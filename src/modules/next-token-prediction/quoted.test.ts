// @vitest-environment node
/// <reference types="node" />
/**
 * Pins the numbers the next-token-prediction lesson (content/standard.mdx, content/plain.mdx,
 * card-info.ts, module.ts, Explore.tsx) quotes from the three shipped character models, so a model
 * re-export or retrain cannot silently stale them. Plan items SH-8 / IM-10 / IM-11.
 *
 * Re-measured after the transformer was retrained (8,000 steps, held-out validation, genuine
 * induction). The GRU and the bigram table are the first release's files and did not change.
 *
 * How the numbers are reproduced:
 *  - RNN and transformer run through the same ONNX files the lab runs, with ONNX Runtime
 *    (onnxruntime-web, WebAssembly CPU provider) in Node. The lab prefers WebGPU when the browser
 *    has it; that is a browser-only runtime, so it is not exercised here. Every percentage quoted
 *    below is to at most two significant figures and sits clear of a rounding boundary, so GPU
 *    float noise cannot move it. The transformer file is the attention module's: this lab shares it.
 *  - The bigram table is bigram.json. Its add-one counts are recovered exactly from the table (see
 *    recoverBigramCounts), so the whole-corpus loss and the capital-J figures are reproduced
 *    without the corpus.
 *  - The loss table is read from language-models.metadata.json and, where the corpus is present
 *    (models/data/tiny-shakespeare.txt, gitignored), re-measured from the ONNX files with the same
 *    protocol train_language_models.py used: consecutive non-overlapping 64-character windows.
 *    Those tests are skipped on a machine without the corpus.
 *  - Nothing here reads Python source text.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";
// Test-only: the attention module ships the transformer (and a byte-identical copy of the
// metadata) that this lab reuses.
import attentionMetadata from "../attention/assets/tiny-transformer.metadata.json";
import bigram from "./assets/bigram.json";
import metadata from "./assets/language-models.metadata.json";
import vocabulary from "./assets/transformer-vocab.json";
import { bitsFromNats, HELD_OUT_PERCENT, LOSS_BITS, TRAINING_STEPS } from "./evaluation";
import { DEFAULT_PROMPT, provenanceOf, TRAINING_LINE, TRAINING_LINE_OFFSET } from "./prompts";
import { encode, entropyBits, formatPercent, logSoftmax, shapeDistribution, surprisalBits } from "./sampling";

const root = process.cwd();
const read = (path: string) => readFileSync(resolve(root, path));
const text = (path: string) => read(path).toString("utf8");

const RNN_PATH = "src/modules/next-token-prediction/assets/character-rnn.onnx";
const TRANSFORMER_PATH = "src/modules/attention/assets/tiny-transformer.onnx";
const CORPUS_PATH = "models/data/tiny-shakespeare.txt";
const PROVENANCE_PATH = "models/data/tiny-shakespeare.provenance.json";

const VOCAB = 66;
/** The lab's default temperature (module.ts initialState). */
const DEFAULT_TEMPERATURE = 0.8;
const TRANSFORMER_CONTEXT = 64;

const stoi = vocabulary.stoi as Record<string, number>;
const itos = vocabulary.itos as string[];
const id = (character: string) => stoi[character];
const isCapital = (character: string) => /^[A-Z]$/.test(character);

let rnn: ort.InferenceSession;
let transformer: ort.InferenceSession;

beforeAll(async () => {
  ort.env.wasm.numThreads = 4;
  // new Uint8Array(buffer) copies, which is what the ONNX reader and ORT both want.
  const open = (path: string) =>
    ort.InferenceSession.create(new Uint8Array(read(path)), { executionProviders: ["wasm"] });
  [rnn, transformer] = await Promise.all([open(RNN_PATH), open(TRANSFORMER_PATH)]);
}, 60_000);

/** Logits [windows.length, width, 66] for equal-length windows of token ids. */
async function runLogits(session: ort.InferenceSession, windows: ReadonlyArray<ReadonlyArray<number>>) {
  const width = windows[0].length;
  const input = new ort.Tensor("int64", BigInt64Array.from(windows.flat(), BigInt), [windows.length, width]);
  const output = await session.run({ input_ids: input });
  return output.logits.data as Float32Array;
}

/** What useCharModels hands the lab: row j is the model's output after reading character j. */
async function rowsFor(model: "rnn" | "transformer", prompt: string) {
  const ids = encode(prompt, vocabulary);
  const windowIds = model === "transformer" ? ids.slice(-TRANSFORMER_CONTEXT) : ids;
  const data = await runLogits(model === "rnn" ? rnn : transformer, [windowIds]);
  return Array.from({ length: windowIds.length }, (_, row) =>
    Array.from(data.subarray(row * VOCAB, (row + 1) * VOCAB)),
  );
}
const lastRow = (rows: number[][]) => rows[rows.length - 1];
const probabilities = (logits: ReadonlyArray<number>) => logSoftmax(logits).map(Math.exp);
const bigramRow = (prompt: string) => bigram.probabilities[encode(prompt, vocabulary).at(-1)!];
const topOf = (values: ReadonlyArray<number>) => {
  const p = probabilities(values);
  const best = p.indexOf(Math.max(...p));
  return { character: itos[best], p: p[best], percent: formatPercent(p[best]) };
};
const totalVariation = (a: ReadonlyArray<number>, b: ReadonlyArray<number>) => {
  const p = probabilities(a);
  const q = probabilities(b);
  return 0.5 * p.reduce((sum, value, index) => sum + Math.abs(value - q[index]), 0);
};
/** Mean surprisal in bits of every character after the first, teacher forced, as the lab's compare card computes it. */
async function meanSurprisal(model: "bigram" | "rnn" | "transformer", prompt: string) {
  const ids = encode(prompt, vocabulary);
  const rows =
    model === "bigram"
      ? ids.map((token) => bigram.probabilities[token].map(Math.log))
      : await rowsFor(model, prompt);
  let total = 0;
  for (let position = 1; position < ids.length; position += 1) {
    total += -logSoftmax(rows[position - 1])[ids[position]] * Math.LOG2E;
  }
  return total / (ids.length - 1);
}

const lastStep = <T extends { step: number }>(history: T[]) => history[history.length - 1];

/**
 * Recover the integer transition counts behind bigram.json. The table is
 * (count + 1) / (row total + 66), so a row's smallest entry m / k fixes the
 * denominator k = row total + 66 when m is a small integer. The recovered counts
 * are checked against the corpus length below, which would fail loudly if the
 * recovery (or the table) were wrong.
 */
function recoverBigramCounts(table: ReadonlyArray<ReadonlyArray<number>>) {
  return table.map((row, from) => {
    const smallest = Math.min(...row);
    for (let m = 1; m <= 200; m += 1) {
      const denominator = Math.round(m / smallest);
      if (row.every((p) => Math.abs(p * denominator - Math.round(p * denominator)) < 1e-6)) {
        return row.map((p) => Math.round(p * denominator) - 1);
      }
    }
    throw new Error(`Could not recover integer counts for bigram row ${from}.`);
  });
}
const bigramCounts = recoverBigramCounts(bigram.probabilities);
const bigramTransitions = bigramCounts.flat().reduce((sum, count) => sum + count, 0);

// ---------------------------------------------------------------------------
// The default prompt and the training line (standard.mdx "Context", "Seen and unseen";
// module.ts steps 1-2; card-info "Next token distribution"; plain.mdx "How to play with it")
// ---------------------------------------------------------------------------

describe("the two worked prompts", () => {
  it("are what the lesson names, and the lab recognises exactly these two as seen or unseen", () => {
    expect(DEFAULT_PROMPT).toBe("O Juliet, Juliet! wherefore art thou J");
    expect(TRAINING_LINE).toBe("O Romeo, Romeo! wherefore art thou R");
    expect(provenanceOf(DEFAULT_PROMPT)).toBe("unseen");
    expect(provenanceOf(TRAINING_LINE)).toBe("training");
    expect(provenanceOf(`${DEFAULT_PROMPT} `)).toBe("unknown");
    expect(provenanceOf("")).toBe("unknown");
    expect(encode(DEFAULT_PROMPT, vocabulary)).not.toContain(vocabulary.unk);
    expect(encode(TRAINING_LINE, vocabulary)).not.toContain(vocabulary.unk);
  });

  it("module.ts starts the lab on the default prompt at temperature 0.8", async () => {
    const { default: definition } = await import("./module");
    expect(definition.initialState.prompt).toBe(DEFAULT_PROMPT);
    expect(definition.initialState.temperature).toBe(DEFAULT_TEMPERATURE);
  });

  it("the training line sits 43% of the way through the corpus, inside the transformer's first 90%", () => {
    expect(TRAINING_LINE_OFFSET).toBe(478_139);
    expect(Math.round((TRAINING_LINE_OFFSET / metadata.corpus_characters) * 100)).toBe(43);
    expect(TRAINING_LINE_OFFSET + TRAINING_LINE.length).toBeLessThanOrEqual(metadata.split.train_characters);
    expect(metadata.split.train_characters).toBe(1_003_854);
    expect(metadata.split.heldout_characters).toBe(111_540);
    expect(HELD_OUT_PERCENT).toBe(10);
  });
});

describe("default prompt: what each model puts first (standard.mdx 'Context'; module.ts step 2)", () => {
  it("Bigram puts U first at 33%, because it only sees J", () => {
    const row = bigramRow(DEFAULT_PROMPT);
    const ranked = row.map((p, index) => ({ p, character: itos[index] })).sort((a, b) => b.p - a.p);
    expect(ranked[0].character).toBe("U");
    expect(formatPercent(ranked[0].p)).toBe("33%");
    expect(ranked.slice(0, 2).map((entry) => entry.character)).toEqual(["U", "u"]);
    expect(formatPercent(ranked[1].p)).toBe("22%");
    expect(entropyBits(row).toFixed(2)).toBe("3.43");
  });

  it("RNN puts o first at 26% and the transformer u first at 99%", async () => {
    const rnnTop = topOf(lastRow(await rowsFor("rnn", DEFAULT_PROMPT)));
    expect([rnnTop.character, rnnTop.percent]).toEqual(["o", "26%"]);
    const transformerTop = topOf(lastRow(await rowsFor("transformer", DEFAULT_PROMPT)));
    expect([transformerTop.character, transformerTop.percent]).toEqual(["u", "99%"]);
  });

  it("the top characters differ across Bigram, RNN, Transformer as U, o, u (module.ts step 2)", async () => {
    const argmax = (values: ReadonlyArray<number>) => values.indexOf(Math.max(...values));
    const rows = [
      bigramRow(DEFAULT_PROMPT),
      lastRow(await rowsFor("rnn", DEFAULT_PROMPT)),
      lastRow(await rowsFor("transformer", DEFAULT_PROMPT)),
    ];
    expect(rows.map((values) => itos[argmax(values)])).toEqual(["U", "o", "u"]);
  });

  it("next-character entropy is 3.43 (bigram), 3.46 (RNN) and 0.09 bits (transformer) (compare card)", async () => {
    const entropy = async (model: "rnn" | "transformer") =>
      entropyBits(probabilities(lastRow(await rowsFor(model, DEFAULT_PROMPT)))).toFixed(2);
    expect(await entropy("rnn")).toBe("3.46");
    expect(await entropy("transformer")).toBe("0.09");
  });

  it("Mean surprisal on the default prompt is 3.89 (bigram), 3.73 (RNN) and 1.82 bits (transformer)", async () => {
    expect((await meanSurprisal("bigram", DEFAULT_PROMPT)).toFixed(2)).toBe("3.89");
    expect((await meanSurprisal("rnn", DEFAULT_PROMPT)).toFixed(2)).toBe("3.73");
    expect((await meanSurprisal("transformer", DEFAULT_PROMPT)).toFixed(2)).toBe("1.82");
  });

  it("capital J: 125 of the corpus's 320 are followed by U, the start of JULIET (card-info, standard.mdx)", () => {
    const row = bigramCounts[id("J")];
    const total = row.reduce((sum, count) => sum + count, 0);
    expect(total).toBe(320);
    expect(row[id("U")]).toBe(125);
    expect(Math.round((row[id("U")] / total) * 100)).toBe(39);
    // Add-one smoothing: (125 + 1) / (320 + 66) is the 33% the lab shows.
    expect(bigram.probabilities[id("J")][id("U")]).toBeCloseTo(126 / 386, 9);
  });
});

describe("the old default, the training line (standard.mdx 'Seen and unseen')", () => {
  it("Bigram I 16%, RNN o 43%, transformer o at 99.97%", async () => {
    expect(formatPercent(Math.max(...bigramRow(TRAINING_LINE)))).toBe("16%");
    expect(itos[bigramRow(TRAINING_LINE).indexOf(Math.max(...bigramRow(TRAINING_LINE)))]).toBe("I");
    const rnnTop = topOf(lastRow(await rowsFor("rnn", TRAINING_LINE)));
    expect([rnnTop.character, rnnTop.percent]).toEqual(["o", "43%"]);
    const transformerTop = topOf(lastRow(await rowsFor("transformer", TRAINING_LINE)));
    expect(transformerTop.character).toBe("o");
    expect(transformerTop.p.toFixed(4)).toBe("0.9997");
    expect(entropyBits(probabilities(lastRow(await rowsFor("transformer", TRAINING_LINE)))).toFixed(3)).toBe("0.004");
  });

  it("the transformer's mean surprisal is 1.56 bits on the training line and 1.82 on the unseen one: a small gap", async () => {
    const onTraining = await meanSurprisal("transformer", TRAINING_LINE);
    const onDefault = await meanSurprisal("transformer", DEFAULT_PROMPT);
    expect(onTraining.toFixed(2)).toBe("1.56");
    expect(onDefault.toFixed(2)).toBe("1.82");
    expect(onDefault - onTraining).toBeLessThan(0.5);
    // The other two models barely change: they were not near-certain on the line either.
    expect((await meanSurprisal("bigram", TRAINING_LINE)).toFixed(2)).toBe("3.94");
    expect((await meanSurprisal("rnn", TRAINING_LINE)).toFixed(2)).toBe("3.54");
  });

  it("repeating a name is enough for near-certainty: a line not in the corpus that repeats Romeo five times gives o at 99.8%", async () => {
    const top = topOf(lastRow(await rowsFor("transformer", "Romeo, Romeo, Romeo, Romeo, Romeo, R")));
    expect(top.character).toBe("o");
    expect(top.p.toFixed(3)).toBe("0.998");
    // Without the repeated name, and without the whole line, the same last character is far less certain.
    const bare = topOf(lastRow(await rowsFor("transformer", "O Juliet, Juliet! wherefore art thou R")));
    expect(bare.character).toBe("o");
    expect(bare.p).toBeLessThan(0.6);
  });
});

describe("editing the start of the prompt (standard.mdx 'Context'; module.ts step 1)", () => {
  // Names changed, final J kept.
  const jason = "O Jason, Jason! wherefore art thou J";

  it("does not move the bigram, and moves the transformer's top character from u to a", async () => {
    expect(bigramRow(jason)).toEqual(bigramRow(DEFAULT_PROMPT));
    const before = lastRow(await rowsFor("transformer", DEFAULT_PROMPT));
    const after = lastRow(await rowsFor("transformer", jason));
    expect(topOf(before).character).toBe("u");
    const top = topOf(after);
    expect([top.character, top.percent]).toEqual(["a", "87%"]);
    expect(totalVariation(before, after)).toBeGreaterThan(0.9);
  });

  it("does not move the RNN: its distribution is the same to within 0.0001 for every such edit", async () => {
    const rnnBefore = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    for (const edited of [
      jason,
      "O Joan, Joan! wherefore art thou J",
      "O Hamlet, Hamlet! wherefore art thou J",
      "ZZZZZZZZZZZZZZZZ wherefore art thou J",
    ]) {
      expect(totalVariation(rnnBefore, lastRow(await rowsFor("rnn", edited)))).toBeLessThan(1e-4);
    }
  });

  it("the RNN's memory fades: a one-character edit more than about a dozen characters back moves it by less than 0.0001", async () => {
    const ids = encode(DEFAULT_PROMPT, vocabulary);
    const rnnBefore = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const tvAt = async (back: number) => {
      const position = ids.length - 1 - back;
      const swap = DEFAULT_PROMPT[position] === "Z" ? "Q" : "Z";
      const edited = `${DEFAULT_PROMPT.slice(0, position)}${swap}${DEFAULT_PROMPT.slice(position + 1)}`;
      return totalVariation(rnnBefore, lastRow(await rowsFor("rnn", edited)));
    };
    for (const back of [11, 12, 16, 20, 30, 37]) expect(await tvAt(back)).toBeLessThan(1e-4);
    // ...but the character just before the last one still matters.
    expect(await tvAt(2)).toBeGreaterThan(0.05);
    expect(await tvAt(1)).toBeGreaterThan(0.5);
  });

  it("the transformer reacts to the characters it copies, not to every edit", async () => {
    const before = lastRow(await rowsFor("transformer", DEFAULT_PROMPT));
    const tvAt = async (back: number) => {
      const position = DEFAULT_PROMPT.length - 1 - back;
      const edited = `${DEFAULT_PROMPT.slice(0, position)}Z${DEFAULT_PROMPT.slice(position + 1)}`;
      return totalVariation(before, lastRow(await rowsFor("transformer", edited)));
    };
    // The u of the second Juliet is 26 back and the u of the first is 34 back.
    expect(DEFAULT_PROMPT[DEFAULT_PROMPT.length - 1 - 26]).toBe("u");
    expect(DEFAULT_PROMPT[DEFAULT_PROMPT.length - 1 - 34]).toBe("u");
    expect(await tvAt(26)).toBeGreaterThan(0.9);
    expect(await tvAt(34)).toBeGreaterThan(0.5);
    // Edits inside "wherefore" (11 to 19 back) barely move it.
    for (let back = 11; back <= 19; back += 1) expect(await tvAt(back)).toBeLessThan(0.02);
  });
});

// ---------------------------------------------------------------------------
// RNN temperature sweep and cuts (standard.mdx "Shaping", "What to notice")
// ---------------------------------------------------------------------------

describe("RNN on the default prompt: temperature and entropy", () => {
  it("o goes 26% at T = 1, 32% at 0.80, 85% at 0.10 and 11% at 2.00 (standard.mdx 'Shaping')", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const o = id("o");
    const q = (temperature: number) => shapeDistribution(logits, { temperature, topK: VOCAB, topP: 1 }).q[o];
    expect(formatPercent(q(1))).toBe("26%");
    expect(formatPercent(q(0.8))).toBe("32%");
    expect(formatPercent(q(0.1))).toBe("85%");
    expect(formatPercent(q(2))).toBe("11%");
  });

  it("entropy goes from 3.46 bits at T = 1 to 0.62 at 0.10 and 5.12 at 2.00", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const bits = (temperature: number) =>
      entropyBits(shapeDistribution(logits, { temperature, topK: VOCAB, topP: 1 }).q).toFixed(2);
    expect(bits(1)).toBe("3.46");
    expect(bits(0.1)).toBe("0.62");
    expect(bits(2)).toBe("5.12");
    // The lab's "Model entropy" metric is the T = 1 value whatever the slider says.
    expect(entropyBits(shapeDistribution(logits, { temperature: 0.3, topK: VOCAB, topP: 1 }).p).toFixed(2)).toBe("3.46");
  });

  it("even at 0.10 the RNN's o is not certain, because a is close behind; Top-k 1 is greedy", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const cold = shapeDistribution(logits, { temperature: 0.1, topK: VOCAB, topP: 1 });
    expect(cold.q[id("a")]).toBeGreaterThan(0.1);
    const greedy = shapeDistribution(logits, { temperature: 1, topK: 1, topP: 1 });
    expect(greedy.q[id("o")]).toBe(1);
    expect(greedy.keptCount).toBe(1);
  });

  it("the order of the candidates never changes with temperature (standard.mdx, plain.mdx)", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const order = (temperature: number) =>
      shapeDistribution(logits, { temperature, topK: VOCAB, topP: 1 }).candidates.map((candidate) => candidate.index);
    const reference = order(1);
    for (const temperature of [0.1, 0.8, 2]) expect(order(temperature)).toEqual(reference);
  });

  it("the distribution view shows 10 rows plus a final row for the other 56 (card-info)", async () => {
    const shaped = shapeDistribution(lastRow(await rowsFor("rnn", DEFAULT_PROMPT)), {
      temperature: DEFAULT_TEMPERATURE,
      topK: VOCAB,
      topP: 1,
    });
    expect(shaped.candidates.slice(10)).toHaveLength(56);
  });
});

describe("RNN on the default prompt: top-p 0.50 at the default temperature", () => {
  it("keeps one character in the middle of `wherefo`, one or two inside the word, and about six right after a space (standard.mdx 'What to notice'; card-info)", async () => {
    const rows = await rowsFor("rnn", DEFAULT_PROMPT);
    const kept = (position: number) =>
      shapeDistribution(rows[position], { temperature: DEFAULT_TEMPERATURE, topK: VOCAB, topP: 0.5 }).keptCount;
    const characters = Array.from(DEFAULT_PROMPT);
    // Row j is the distribution after reading character j.
    const afterWherefo = DEFAULT_PROMPT.indexOf("wherefo") + "wherefo".length - 1;
    expect(characters[afterWherefo]).toBe("o");
    expect(kept(afterWherefo)).toBe(1);

    const afterSpaces = characters.map((c, index) => (c === " " ? kept(index) : 0)).filter((count) => count > 0);
    expect(afterSpaces).toEqual([11, 4, 7, 5, 5, 5]);
    expect(Math.min(...afterSpaces)).toBeGreaterThanOrEqual(4);
    expect(Math.round(afterSpaces.reduce((a, b) => a + b, 0) / afterSpaces.length)).toBe(6);
    // card-info 'Sampler controls': "one or two characters in the middle of a word".
    const midWord = Array.from("wherefor", (_, offset) => kept(DEFAULT_PROMPT.indexOf("wherefore") + offset));
    expect(midWord.filter((count) => count <= 2).length).toBeGreaterThanOrEqual(midWord.length - 1);
  });
});

describe("min-p on real logits (standard.mdx 'What decoding does' and 'Shaping'; card-info 'Sampler controls')", () => {
  it("Min-p 0.10 keeps 7 RNN characters at T = 1 (cut-off 2.6%) and 19 at T = 2", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const at = (temperature: number, minP: number) => shapeDistribution(logits, { temperature, topK: VOCAB, topP: 1, minP });
    const cool = at(1, 0.1);
    expect(cool.keptCount).toBe(7);
    expect(formatPercent(cool.minPThreshold)).toBe("2.6%");
    expect(at(2, 0.1).keptCount).toBe(19);
    expect(at(0.8, 0.1).keptCount).toBe(6);
    // Flatter distributions let more through, so the count never falls as temperature rises.
    const counts = [0.1, 0.3, 0.8, 1, 1.5, 2].map((temperature) => at(temperature, 0.1).keptCount);
    expect(counts).toEqual([...counts].sort((a, b) => a - b));
  });

  it("it lowers the sampler's entropy below the model's, 3.46 to 2.52 bits at T = 1", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const shaped = shapeDistribution(logits, { temperature: 1, topK: VOCAB, topP: 1, minP: 0.1 });
    expect(entropyBits(shaped.p).toFixed(2)).toBe("3.46");
    expect(entropyBits(shaped.q).toFixed(2)).toBe("2.52");
  });

  it("with the transformer on the default prompt Min-p 0.10 keeps one character at every temperature", async () => {
    const logits = lastRow(await rowsFor("transformer", DEFAULT_PROMPT));
    for (const temperature of [0.1, 0.5, 0.8, 1, 1.5, 2]) {
      expect(shapeDistribution(logits, { temperature, topK: VOCAB, topP: 1, minP: 0.1 }).keptCount).toBe(1);
    }
  });

  it("it keeps the top character's cut-off in tempered probability: 0.10 × p_max", async () => {
    const logits = lastRow(await rowsFor("rnn", DEFAULT_PROMPT));
    const shaped = shapeDistribution(logits, { temperature: DEFAULT_TEMPERATURE, topK: VOCAB, topP: 1, minP: 0.1 });
    expect(shaped.minPThreshold).toBeCloseTo(0.1 * shaped.candidates[0].tempered, 12);
    expect(formatPercent(shaped.candidates[0].tempered)).toBe("32%");
  });
});

// ---------------------------------------------------------------------------
// Greedy decoding loops (standard.mdx "The loop", "Going deeper"; module.ts step 4)
// ---------------------------------------------------------------------------

describe("greedy decoding from the default prompt falls into a loop (Top-k 1)", () => {
  async function greedyRun(model: "bigram" | "rnn" | "transformer", length: number) {
    let ids = encode(DEFAULT_PROMPT, vocabulary);
    let out = "";
    const chosen: number[] = [];
    for (let step = 0; step < length; step += 1) {
      const logits =
        model === "bigram"
          ? bigram.probabilities[ids[ids.length - 1]].map(Math.log)
          : lastRow(await rowsFor(model, Array.from(ids, (token) => itos[token]).join("")));
      const pick = shapeDistribution(logits, { temperature: 1, topK: 1, topP: 1 }).candidates[0];
      out += itos[pick.index];
      chosen.push(pick.p);
      ids = [...ids, pick.index];
    }
    return { text: out, chosen };
  }
  const greedy = async (model: "bigram" | "rnn" | "transformer", length: number) => (await greedyRun(model, length)).text;
  /** Smallest period p such that the last `tail` characters repeat with period p. */
  const loopPeriod = (value: string, tail: number) => {
    const window = value.slice(-tail);
    for (let period = 1; period <= tail / 2; period += 1) {
      if (Array.from(window).every((character, index) => index < period || character === window[index - period])) return period;
    }
    return undefined;
  };

  it("the RNN repeats `the` (period 4) and the transformer repeats a phrase of 22 characters", async () => {
    const rnnText = await greedy("rnn", 120);
    expect(rnnText.startsWith("ore the the the")).toBe(true);
    expect(loopPeriod(rnnText, 60)).toBe(4);
    const transformerText = await greedy("transformer", 120);
    expect(transformerText.startsWith("uliet! wherefore?")).toBe(true);
    expect(transformerText).toContain("the said that we said the said that we said");
    expect(loopPeriod(transformerText, 66)).toBe(22);
  }, 120_000);

  it("the loop feeds itself: the transformer's probability for its chosen character rises from 83% in the first 22 characters to above 96% in the last 22 (standard.mdx 'Going deeper')", async () => {
    const { chosen } = await greedyRun("transformer", 120);
    const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
    expect(mean(chosen.slice(0, 22)).toFixed(2)).toBe("0.83");
    expect(mean(chosen.slice(-22))).toBeGreaterThan(0.96);
  }, 120_000);

  it("the bigram table can only repeat one character: it settles on a newline", async () => {
    const bigramText = await greedy("bigram", 60);
    expect(bigramText.slice(-30)).toBe("\n".repeat(30));
    expect(loopPeriod(bigramText, 30)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Bits per character (standard.mdx "What to notice", "Going deeper"; card-info "Three models,
// one context"; Explore.tsx MODEL_INFO via evaluation.ts)
// ---------------------------------------------------------------------------

describe("bits per character, from the shipped metadata", () => {
  it("held-out tail (nats): transformer 1.5433, GRU 2.1093, bigram fitted on the first 90% 2.4820", () => {
    const tail = metadata.evaluation.heldout_tail;
    expect(tail.characters).toBe(111_540);
    expect(tail.transformer).toBeCloseTo(1.5433, 4);
    expect(tail.rnn).toBeCloseTo(2.1093, 4);
    expect(tail.bigram_fitted_on_training_split).toBeCloseTo(2.482, 4);
  });

  it("whole corpus (nats): transformer 1.2667, GRU 2.1215, bigram fitted on all of it 2.4550", () => {
    const whole = metadata.evaluation.whole_corpus;
    expect(whole.characters).toBe(metadata.corpus_characters);
    expect(whole.transformer).toBeCloseTo(1.2667, 4);
    expect(whole.rnn).toBeCloseTo(2.1215, 4);
    expect(whole.bigram_fitted_on_whole_corpus).toBeCloseTo(2.455, 4);
  });

  it("the card prints held-out bits 3.58 / 3.04 / 2.23 and whole-corpus bits 3.54 / 3.06 / 1.83 (bigram / GRU / transformer)", () => {
    const rounded = (value: number) => value.toFixed(2);
    expect([LOSS_BITS.bigram.heldOut, LOSS_BITS.rnn.heldOut, LOSS_BITS.transformer.heldOut].map(rounded)).toEqual([
      "3.58",
      "3.04",
      "2.23",
    ]);
    expect([LOSS_BITS.bigram.wholeCorpus, LOSS_BITS.rnn.wholeCorpus, LOSS_BITS.transformer.wholeCorpus].map(rounded)).toEqual([
      "3.54",
      "3.06",
      "1.83",
    ]);
    expect(bitsFromNats(metadata.evaluation.heldout_tail.transformer)).toBeCloseTo(2.2265, 3);
    expect(bitsFromNats(metadata.evaluation.whole_corpus.transformer)).toBeCloseTo(1.8275, 3);
  });

  it("the order is transformer < GRU < bigram on both protocols (standard.mdx, Explore.tsx note)", () => {
    for (const key of ["heldOut", "wholeCorpus"] as const) {
      expect(LOSS_BITS.transformer[key]).toBeLessThan(LOSS_BITS.rnn[key]);
      expect(LOSS_BITS.rnn[key]).toBeLessThan(LOSS_BITS.bigram[key]);
    }
  });

  it("the transformer fits its training text better than held-out text; the other two barely differ", () => {
    // 90% of the whole corpus is the transformer's training text.
    expect(LOSS_BITS.transformer.heldOut - LOSS_BITS.transformer.wholeCorpus).toBeGreaterThan(0.35);
    expect(Math.abs(LOSS_BITS.rnn.heldOut - LOSS_BITS.rnn.wholeCorpus)).toBeLessThan(0.05);
    expect(Math.abs(LOSS_BITS.bigram.heldOut - LOSS_BITS.bigram.wholeCorpus)).toBeLessThan(0.05);
  });

  it("the transformer's last validation loss in its history is the held-out tail loss", () => {
    expect(lastStep(metadata.transformer.history).step).toBe(8000);
    expect(lastStep(metadata.transformer.history).validation_loss).toBeCloseTo(metadata.evaluation.heldout_tail.transformer, 9);
  });

  it("the bigram table in this lab was fitted on the whole corpus: its loss over every transition is 2.4550 nats (3.54 bits)", () => {
    // Recovery check: the counts add up to exactly the corpus's transitions.
    expect(bigramTransitions).toBe(metadata.corpus_characters - 1);
    let nats = 0;
    bigramCounts.forEach((row, from) =>
      row.forEach((count, to) => {
        nats += count * -Math.log(bigram.probabilities[from][to]);
      }),
    );
    const mean = nats / bigramTransitions;
    // train_language_models.py scores every transition of the whole corpus, so this is exact.
    expect(mean).toBeCloseTo(metadata.evaluation.whole_corpus.bigram_fitted_on_whole_corpus, 6);
    expect(bitsFromNats(mean).toFixed(2)).toBe("3.54");
  });

  it("a uniform guess over 66 symbols costs log2(66) = 6.04 bits", () => {
    expect(Math.log2(66).toFixed(2)).toBe("6.04");
  });

  it("perplexity 2^bits: about 4.7 characters for the transformer and 8.2 for the GRU on held-out text, 8.3 for the GRU on the whole corpus", () => {
    expect((2 ** LOSS_BITS.transformer.heldOut).toFixed(1)).toBe("4.7");
    expect((2 ** LOSS_BITS.rnn.heldOut).toFixed(1)).toBe("8.2");
    expect((2 ** LOSS_BITS.rnn.wholeCorpus).toFixed(1)).toBe("8.3");
  });

  it("converts nats to bits by log2 e, about 1.443 (card-info)", () => {
    expect(Math.LOG2E.toFixed(3)).toBe("1.443");
  });

  it("6 bits of surprisal is about 1 chance in 64 (module.ts glossary)", () => {
    expect(surprisalBits(1 / 64)).toBeCloseTo(6, 12);
  });

  it("the two shipped copies of the metadata agree (the transformer is shared with Attention)", () => {
    expect(attentionMetadata).toEqual(metadata);
  });
});

describe("training budgets, the caveat the lesson states (standard.mdx 'Where it breaks'; card-info; Explore.tsx note)", () => {
  it("the transformer trained for 8,000 steps and the shipped GRU for 240: 33 times as many", () => {
    expect(TRAINING_STEPS).toEqual({ transformer: 8000, rnn: 240 });
    expect(metadata.transformer.training.steps).toBe(8000);
    expect(lastStep(metadata.rnn.history).step).toBe(240);
    expect(Math.round(TRAINING_STEPS.transformer / TRAINING_STEPS.rnn)).toBe(33);
  });

  it("the GRU's own note says it saw about a third of an epoch, and its file is the first release's", () => {
    expect(metadata.evaluation.note).toContain("about a third of an epoch");
    expect(metadata.rnn.status).toContain("reused");
  });

  it("each transformer batch is 45 corpus windows and 19 synthetic repeated strings: about 30%", () => {
    const { corpus, synthetic } = metadata.transformer.training.windows_per_batch;
    expect([corpus, synthetic]).toEqual([45, 19]);
    expect(corpus + synthetic).toBe(metadata.transformer.training.batch_size);
    expect(Math.round((synthetic / metadata.transformer.training.batch_size) * 100)).toBe(30);
  });

  it("the transformer read about 23 million corpus characters, about 23 passes over its 1.0 million training characters", () => {
    const { steps, windows_per_batch: windows } = metadata.transformer.training;
    const characters = steps * windows.corpus * metadata.transformer.config.block_size;
    expect(characters).toBe(23_040_000);
    expect((characters / metadata.split.train_characters).toFixed(0)).toBe("23");
  });

  it("66-symbol vocabulary: the 65 characters of Tiny Shakespeare plus <unk> (card-info)", () => {
    expect(vocabulary.itos).toHaveLength(66);
    expect(vocabulary.unk).toBe(65);
    expect(vocabulary.itos[vocabulary.unk]).toBe("<unk>");
    expect(Object.keys(vocabulary.stoi)).toHaveLength(65);
    expect(metadata.vocabulary_size).toBe(66);
    // bigram.json is a 66 x 66 probability table whose rows each sum to one (card-info "66 × 66").
    expect(bigram.probabilities).toHaveLength(66);
    for (const row of bigram.probabilities) {
      expect(row).toHaveLength(66);
      expect(row.reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 9);
    }
  });

  it("about 1.1 million characters of Tiny Shakespeare (standard.mdx, card-info)", () => {
    expect(metadata.corpus_characters).toBe(1_115_394);
    expect((metadata.corpus_characters / 1e6).toFixed(1)).toBe("1.1");
  });

  it("the transformer is 2 layers x 4 heads, 256 wide, 768-wide MLP, 64-character context, 1,367,552 parameters (Explore.tsx pipeline)", () => {
    expect(metadata.transformer.config).toMatchObject({
      vocab_size: 66,
      block_size: 64,
      model_width: 256,
      heads: 4,
      layers: 2,
      feed_forward_width: 768,
    });
    expect(metadata.transformer.parameters).toBe(1_367_552);
  });

  it("the GRU is a 2-layer, 192-wide state (Explore.tsx 'sees' text)", () => {
    const width = 192;
    const gruLayer = 3 * (width * width) * 2 + 2 * 3 * width; // weight_ih, weight_hh, bias_ih, bias_hh
    const parameters = VOCAB * width + 2 * gruLayer + (width * VOCAB + VOCAB); // embedding + 2 GRU layers + output
    expect(metadata.rnn.parameters).toBe(parameters);
    expect(metadata.rnn.parameters).toBe(470_082);
  });
});

// ---------------------------------------------------------------------------
// Surprisal figures quoted at standard.mdx "What to notice", measured with the shipped GRU
// ---------------------------------------------------------------------------

describe("GRU surprisal on the default prompt", () => {
  it("pays 8.0, 10.4 and 12.0 bits for its three capital Js", async () => {
    const ids = encode(DEFAULT_PROMPT, vocabulary);
    const rows = await rowsFor("rnn", DEFAULT_PROMPT);
    // Positions of the capital Js in "O Juliet, Juliet! wherefore art thou J"; row k predicts token k + 1.
    const positions = Array.from(DEFAULT_PROMPT).flatMap((character, index) => (character === "J" ? [index] : []));
    expect(positions).toEqual([2, 10, 37]);
    const bits = positions.map((position) => -logSoftmax(rows[position - 1])[ids[position]] / Math.LN2);
    expect(bits.map((value) => Number(value.toFixed(1)))).toEqual([8, 10.4, 12]);
  });
});

// ---------------------------------------------------------------------------
// Corpus-dependent figures: need models/data/tiny-shakespeare.txt (gitignored, so
// present only on a machine that downloaded it with models/download_corpus.py).
// ---------------------------------------------------------------------------

const corpusAvailable = existsSync(resolve(root, CORPUS_PATH));

describe.skipIf(!corpusAvailable)("corpus-dependent figures (skipped when models/data is absent)", () => {
  const corpus = corpusAvailable ? text(CORPUS_PATH) : "";
  const characters = Array.from(corpus);
  const corpusIds = encode(corpus, vocabulary);
  const boundary = Math.floor(corpusIds.length * 0.9);

  it("is the pinned Tiny Shakespeare text, split 90/10 as the metadata says", () => {
    const provenance = JSON.parse(text(PROVENANCE_PATH)) as { sha256: string; bytes: number };
    expect(createHash("sha256").update(read(CORPUS_PATH)).digest("hex")).toBe(provenance.sha256);
    expect(characters).toHaveLength(metadata.corpus_characters);
    expect(boundary).toBe(metadata.split.train_characters);
    expect(corpusIds.length - boundary).toBe(metadata.split.heldout_characters);
  });

  it("the default prompt is not in the corpus; its longest corpus-matching ending is 2 characters (' J')", () => {
    expect(corpus.includes(DEFAULT_PROMPT)).toBe(false);
    let longest = "";
    for (let length = DEFAULT_PROMPT.length; length > 0; length -= 1) {
      if (corpus.includes(DEFAULT_PROMPT.slice(-length))) {
        longest = DEFAULT_PROMPT.slice(-length);
        break;
      }
    }
    expect(longest).toBe(" J");
    // Its parts are in the corpus: `wherefore art thou` once, and Juliet 50 times, all in the training split.
    expect(corpus.split("wherefore art thou").length - 1).toBe(1);
    expect(corpus.slice(0, boundary).split("Juliet").length - 1).toBe(50);
    expect(corpus.slice(boundary).includes("Juliet")).toBe(false);
  });

  it("the contrast prompts the lesson uses are not in the corpus either", () => {
    for (const prompt of [
      "Romeo, Romeo, Romeo, Romeo, Romeo, R",
      "O Juliet, Juliet! wherefore art thou R",
      "O Jason, Jason! wherefore art thou J",
    ]) {
      expect(corpus.includes(prompt), prompt).toBe(false);
    }
  });

  it("the training line appears exactly once, at character 478,139, inside the training split", () => {
    expect(corpus.split(TRAINING_LINE).length - 1).toBe(1);
    expect(corpus.indexOf(TRAINING_LINE)).toBe(TRAINING_LINE_OFFSET);
    expect(corpus.indexOf(TRAINING_LINE) + TRAINING_LINE.length).toBeLessThanOrEqual(boundary);
  });

  it("the bigram counts recovered from bigram.json match the corpus's own capital-J transitions, and JULIET is every JU", () => {
    let capitalJ = 0;
    let followedByU = 0;
    for (let index = 0; index < characters.length - 1; index += 1) {
      if (characters[index] !== "J") continue;
      capitalJ += 1;
      if (characters[index + 1] === "U") followedByU += 1;
    }
    expect([followedByU, capitalJ]).toEqual([125, 320]);
    expect(corpus.split("JULIET").length - 1).toBe(125);
    expect(corpus.split("JU").length - 1).toBe(125);
    expect(corpus.split("JULIET:").length - 1).toBe(125);
  });

  it("an add-one bigram fitted on the first 90% scores 3.58 bits on the held-out tail", () => {
    const counts = Array.from({ length: VOCAB }, () => new Array<number>(VOCAB).fill(1));
    for (let index = 0; index < boundary - 1; index += 1) counts[corpusIds[index]][corpusIds[index + 1]] += 1;
    const totals = counts.map((row) => row.reduce((a, b) => a + b, 0));
    let nats = 0;
    for (let index = boundary; index < corpusIds.length - 1; index += 1) {
      nats += -Math.log(counts[corpusIds[index]][corpusIds[index + 1]] / totals[corpusIds[index]]);
    }
    const mean = nats / (corpusIds.length - 1 - boundary);
    expect(mean).toBeCloseTo(metadata.evaluation.heldout_tail.bigram_fitted_on_training_split, 6);
    expect(bitsFromNats(mean).toFixed(2)).toBe("3.58");
  });

  /** train_language_models.py's window_loss: consecutive, non-overlapping 64-character windows, mean nats per character. */
  async function windowLoss(session: ort.InferenceSession, tokens: number[], stride = 1) {
    const block = TRANSFORMER_CONTEXT;
    const windows = Math.floor((tokens.length - 1) / block);
    const starts = Array.from({ length: Math.ceil(windows / stride) }, (_, index) => index * stride);
    let total = 0;
    let count = 0;
    for (let first = 0; first < starts.length; first += 128) {
      const chunk = starts.slice(first, first + 128);
      const data = await runLogits(
        session,
        chunk.map((window) => tokens.slice(window * block, (window + 1) * block)),
      );
      chunk.forEach((window, row) => {
        for (let position = 0; position < block; position += 1) {
          const logits = Array.from(data.subarray((row * block + position) * VOCAB, (row * block + position + 1) * VOCAB));
          total += -logSoftmax(logits)[tokens[window * block + position + 1]];
          count += 1;
        }
      });
    }
    return total / count;
  }

  it("re-measured from the ONNX files, the held-out tail loss is 1.5433 nats (transformer) and 2.1093 (GRU), the metadata's figures", async () => {
    const tail = corpusIds.slice(boundary);
    expect(await windowLoss(transformer, tail)).toBeCloseTo(metadata.evaluation.heldout_tail.transformer, 4);
    expect(await windowLoss(rnn, tail)).toBeCloseTo(metadata.evaluation.heldout_tail.rnn, 4);
  }, 240_000);

  it("re-measured from the ONNX files on every 16th window of the corpus, the whole-corpus loss agrees with the metadata (1.2667, 2.1215)", async () => {
    // The full pass over 17,428 windows takes minutes; a regular stride samples the same text evenly.
    expect(Math.abs((await windowLoss(transformer, corpusIds, 16)) - metadata.evaluation.whole_corpus.transformer)).toBeLessThan(0.04);
    expect(Math.abs((await windowLoss(rnn, corpusIds, 16)) - metadata.evaluation.whole_corpus.rnn)).toBeLessThan(0.04);
  }, 240_000);

  it("the GRU pays 4.4 bits for a word's first letter, well above a letter inside a word (standard.mdx 'What to notice')", async () => {
    const width = 1024;
    const windows = 100;
    const first = { total: 0, count: 0 };
    const inside = { total: 0, count: 0 };
    for (let start = 0; start < windows; start += 12) {
      const batch = Array.from({ length: Math.min(12, windows - start) }, (_, index) =>
        corpusIds.slice((start + index) * width, (start + index + 1) * width),
      );
      const data = await runLogits(rnn, batch);
      batch.forEach((_, window) => {
        const base = (start + window) * width;
        for (let position = 1; position < width; position += 1) {
          const character = characters[base + position];
          const previous = characters[base + position - 1];
          if (!/[A-Za-z]/.test(character)) continue;
          const row = Array.from(data.subarray((window * width + position - 1) * VOCAB, (window * width + position) * VOCAB));
          const bits = bitsFromNats(-logSoftmax(row)[corpusIds[base + position]]);
          if (/\s/.test(previous)) {
            first.total += bits;
            first.count += 1;
          } else if (/[A-Za-z]/.test(previous)) {
            inside.total += bits;
            inside.count += 1;
          }
        }
      });
    }
    const firstBits = first.total / first.count;
    const insideBits = inside.total / inside.count;
    expect(Math.abs(firstBits - 4.4)).toBeLessThan(0.05);
    // standard.mdx quotes about 3.0 bits for a letter that follows a letter (3.02 measured).
    expect(Math.abs(insideBits - 3.0)).toBeLessThan(0.05);
    expect(firstBits - insideBits).toBeGreaterThan(1);
  }, 60_000);
});
