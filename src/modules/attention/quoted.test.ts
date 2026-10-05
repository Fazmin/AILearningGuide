// @vitest-environment node
/// <reference types="node" />
/**
 * Pins the numbers the attention lesson (content/standard.mdx, content/plain.mdx,
 * card-info.ts, module.ts) quotes from the shipped two-layer character
 * transformer, so a model re-export or retrain cannot silently stale them.
 * Plan items SH-8 / IM-10 (re-truthing after the retrain), IM-6 (mask) and IM-7
 * (scaling, value vectors).
 *
 * How the numbers are reproduced:
 *  - Training facts and the induction check are read from the metadata JSON that
 *    ships beside the model (assets/tiny-transformer.metadata.json). Nothing here
 *    reads the Python training source.
 *  - Behavioural figures are computed with the SDK's plain TypeScript forward pass of
 *    the same ONNX file (teachingTransformer.weightsFromOnnx / runTransformer), then
 *    read through the lab's own helpers (wordSpans, aggregateByWord, whitespaceShare)
 *    with the lab's own inputs, so a test sees what the learner sees.
 *  - Query, key and value tensors, which the TypeScript pass does not expose, come
 *    from the real ONNX Runtime (onnxruntime-web, WebAssembly CPU provider), the same
 *    file the lab runs. The mask, scaling and value-vector controls are tested on
 *    those tensors with the lab's own attention-math helpers. The lab prefers WebGPU
 *    when the browser has it; that is browser-only and is not exercised here (the
 *    figures are quoted to one decimal and sit clear of a rounding boundary).
 *  - The induction re-run uses a seeded sample, so it cannot match the metadata bit
 *    for bit; it is compared with a tolerance above its sampling noise.
 *
 * Layer and head numbers below are 1-based like the lab's labels ("Layer 2, head 3"),
 * converted to the 0-based tensor indices at the one place they are used.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";
import { teachingTransformer } from "@app/module-sdk";
import metadata from "./assets/tiny-transformer.metadata.json";
import vocabulary from "./assets/transformer-vocab.json";
import {
  aggregateByWord,
  attentionMatrix,
  attentionRow,
  bidirectionalMatrix,
  causalMatrix,
  headMatrix,
  laterShare,
  mixRow,
  saturatedRows,
  unownedShare,
  vectorLength,
  whitespaceShare,
  wordSpans,
  zeroValues,
} from "./attention-math";

type RunResult = ReturnType<typeof teachingTransformer.runTransformer>;
const { runTransformer, softmax, weightsFromOnnx } = teachingTransformer;

const root = process.cwd();
const read = (path: string) => readFileSync(resolve(root, path));
const text = (path: string) => read(path).toString("utf8");

// new Uint8Array(buffer) copies, which is what the ONNX reader and ORT both want.
const onnxBytes = new Uint8Array(read("src/modules/attention/assets/tiny-transformer.onnx"));
const weights = weightsFromOnnx(onnxBytes);

// The lab's example inputs (Explore.tsx EXAMPLES); checked against the source below.
const PRONOUN = "The animal did not cross the street because it was tired";
const NAME = "Mark Jones met Anna Blake and then Mark";
const MOVED = "Anna Blake met Mark Jones and then Mark";
const REPEAT_8 = "zqxjkvbwzqxjkvbwzqxjkvbwzqxjkvbw";
const REPEAT_9 = "zqxjkvbwmzqxjkvbwmzqxjkvbwmzqxjk";
const NEW_SURNAME = "Mark Jones met Anna Blake and then Mark Smith";
const MAX_CHARACTERS = 64;
const LAYERS = 2;
const HEADS = 4;
const ALL_HEADS = [1, 2].flatMap((layer) => [1, 2, 3, 4].map((head) => ({ layer, head })));
const label = ({ layer, head }: { layer: number; head: number }) => `L${layer}H${head}`;

const stoi = vocabulary.stoi as Record<string, number>;
/** Same mapping as Explore.tsx: characters outside the vocabulary become <unk>. */
const toIds = (value: string) => Array.from(value).slice(0, MAX_CHARACTERS).map((c) => stoi[c] ?? vocabulary.unk);
/** Same formatting as Explore.tsx. */
const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

const runs = new Map<string, RunResult>();
function forward(value: string) {
  let run = runs.get(value);
  if (!run) {
    run = runTransformer(weights, toIds(value));
    runs.set(value, run);
  }
  return run;
}

/** One row of the model's attention for (1-based) layer and head: weights over key positions 0..sequence-1. */
function characterRow(run: RunResult, layer: number, head: number, query: number) {
  const n = run.sequence;
  const offset = ((head - 1) * n + query) * n;
  return Array.from(run.attention[layer - 1].subarray(offset, offset + n));
}

/** What the Words view and readout strip show for one query word (Explore.tsx), with the last occurrence of a repeated word. */
function labView(value: string, layer: number, head: number, queryWord: string) {
  const characters = Array.from(value).slice(0, MAX_CHARACTERS);
  const spans = wordSpans(characters);
  const selected = spans.map((span) => span.text).lastIndexOf(queryWord);
  expect(selected).toBeGreaterThanOrEqual(0);
  const position = spans[selected].representative;
  const row = characterRow(forward(value), layer, head, position);
  const wordWeights = aggregateByWord(row, spans);
  const strongest = wordWeights.indexOf(Math.max(...wordWeights));
  const strongestCharacter = row.indexOf(Math.max(...row));
  return {
    characters,
    spans,
    position,
    row,
    wordWeights,
    strongestWord: spans[strongest].text,
    strongestWeight: wordWeights[strongest],
    strongestOffset: position - strongestCharacter,
    strongestCharacter: characters[strongestCharacter],
    spaces: whitespaceShare(row.slice(0, position + 1), characters),
  };
}

/** Mean over queries from `fromQuery` on of the weight exactly `offset` characters back. */
function meanWeightAtOffset(run: RunResult, layer: number, head: number, offset: number, fromQuery: number) {
  let total = 0;
  for (let query = fromQuery; query < run.sequence; query += 1) {
    total += characterRow(run, layer, head, query)[query - offset];
  }
  return total / (run.sequence - fromQuery);
}

/** The `count` offsets (1..maxOffset) with the most mean weight, measured over queries >= maxOffset. */
function topOffsets(run: RunResult, layer: number, head: number, maxOffset: number, count: number) {
  return Array.from({ length: maxOffset }, (_, index) => index + 1)
    .map((offset) => ({ offset, weight: meanWeightAtOffset(run, layer, head, offset, maxOffset) }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, count)
    .map((entry) => entry.offset)
    .sort((a, b) => a - b);
}

/**
 * Mean over every query that has at least one earlier occurrence of its next
 * character of the total weight on every earlier occurrence of that character:
 * in a string that repeats every `period`, the character after query q sits at
 * q + 1 - period, q + 1 - 2 * period, and so on.
 */
function trueNextCharacterShare(run: RunResult, layer: number, head: number, period: number) {
  let total = 0;
  let queries = 0;
  for (let query = period - 1; query < run.sequence; query += 1) {
    const row = characterRow(run, layer, head, query);
    for (let key = query + 1 - period; key >= 0; key -= period) total += row[key];
    queries += 1;
  }
  return total / queries;
}

function mulberry32(seed: number) {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A random string of `period` distinct symbols (never the <unk> id), repeated once. */
function repeatedDistinct(random: () => number, period: number) {
  const pool = Array.from({ length: weights.vocabulary - 1 }, (_, index) => index);
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [pool[index], pool[other]] = [pool[other], pool[index]];
  }
  const base = pool.slice(0, period);
  return [...base, ...base];
}

// ---------------------------------------------------------------------------
// The model itself and its recorded training facts
// ---------------------------------------------------------------------------

describe("the transformer behind the lab", () => {
  it("is the two-layer, four-head, width-256, head-64, 66-symbol, 64-context model the lesson describes (standard.mdx, module.ts glossary, card-info)", () => {
    expect(weights.blocks).toHaveLength(LAYERS);
    expect(weights.heads).toBe(HEADS);
    expect(LAYERS * HEADS).toBe(8); // "the model's eight heads (two layers of four)"
    expect(weights.width).toBe(256);
    expect(weights.headWidth).toBe(64); // "256 over 4 gives 64"
    expect(Math.sqrt(weights.headWidth)).toBe(8); // "divides by √64 = 8"
    expect(weights.vocabulary).toBe(66);
    expect(weights.context).toBe(MAX_CHARACTERS); // "input caps at 64 characters, the trained block size"
    expect(metadata.transformer.config).toMatchObject({ block_size: 64, model_width: 256, heads: 4, layers: 2 });
  });

  it("has 1,367,552 parameters: 1.37 million (standard.mdx), about 1.4 million (plain.mdx)", () => {
    const count = (...arrays: ArrayLike<number>[]) => arrays.reduce((sum, array) => sum + array.length, 0);
    const parameters =
      count(
        weights.tokenEmbedding,
        weights.positionEmbedding,
        weights.finalNormWeight,
        weights.finalNormBias,
        weights.unembedding,
      ) + weights.blocks.reduce((sum, block) => sum + count(...Object.values(block)), 0);
    expect(parameters).toBe(1_367_552);
    expect(metadata.transformer.parameters).toBe(1_367_552);
    expect((parameters / 1e6).toFixed(2)).toBe("1.37");
    expect((parameters / 1e6).toFixed(1)).toBe("1.4");
  });

  it("learns its position from 64 learned absolute vectors, one per slot (standard.mdx, card-info, glossary)", () => {
    expect(weights.positionEmbedding.length).toBe(64 * weights.width);
    expect(weights.context).toBe(64);
  });

  it("trained for 8,000 steps on batches of 64 windows: 45 from the training split and 19 synthetic repeated strings (standard.mdx, plain.mdx, card-info)", () => {
    const training = metadata.transformer.training;
    expect(training.steps).toBe(8000);
    expect(training.batch_size).toBe(64);
    expect(training.windows_per_batch).toEqual({ corpus: 45, synthetic: 19 });
    expect(training.windows_per_batch.corpus + training.windows_per_batch.synthetic).toBe(training.batch_size);
    // "about 30 percent of each batch": 19 of 64 is 29.7%.
    expect(Math.round((training.windows_per_batch.synthetic / training.batch_size) * 100)).toBe(30);
    expect(((training.windows_per_batch.synthetic / training.batch_size) * 100).toFixed(1)).toBe("29.7");
    expect(metadata.transformer.history[metadata.transformer.history.length - 1].step).toBe(8000);
    expect(metadata.transformer.config.block_size).toBe(64);
  });

  it("about 23 passes over the 1.0 million character training split (standard.mdx, plain.mdx)", () => {
    const { train_characters, heldout_characters } = metadata.split;
    expect(train_characters).toBe(1_003_854);
    expect((train_characters / 1e6).toFixed(1)).toBe("1.0");
    const training = metadata.transformer.training;
    const seen = training.steps * training.windows_per_batch.corpus * metadata.transformer.config.block_size;
    expect(seen).toBe(23_040_000);
    expect(Math.round(seen / train_characters)).toBe(23);
    // The split is 90 / 10, in corpus order, and the last 10% is what every loss figure is measured on.
    expect(metadata.split.train_fraction).toBe(0.9);
    expect(train_characters + heldout_characters).toBe(metadata.corpus_characters);
    expect(heldout_characters / metadata.corpus_characters).toBeCloseTo(0.1, 4);
  });

  it("the synthetic strings repeat with periods 4 to 31 except 11, 13, 17 and 19 (standard.mdx, plain.mdx, card-info)", () => {
    const curriculum = metadata.transformer.training.synthetic_curriculum;
    const held = [11, 13, 17, 19];
    expect(curriculum.held_out_periods).toEqual(held);
    const expected = Array.from({ length: 28 }, (_, index) => index + 4).filter((period) => !held.includes(period));
    expect(curriculum.training_periods).toEqual(expected);
    expect(curriculum.training_periods).toHaveLength(24);
    expect(Math.min(...curriculum.training_periods)).toBe(4);
    expect(Math.max(...curriculum.training_periods)).toBe(31);
    expect(metadata.transformer.induction_diagnostic.held_out_periods).toEqual(held);
  });

  it("validation loss on the held-out last 10% is 1.54 nats (2.23 bits) against 4.19 nats for a uniform guess (standard.mdx, plain.mdx)", () => {
    const history = metadata.transformer.history;
    const final = history[history.length - 1].validation_loss;
    expect(final).toBeCloseTo(metadata.evaluation.heldout_tail.transformer, 10);
    expect(final.toFixed(2)).toBe("1.54");
    expect((final / Math.LN2).toFixed(2)).toBe("2.23");
    expect(metadata.evaluation.heldout_tail.characters).toBe(metadata.split.heldout_characters);
    expect(Math.log(weights.vocabulary).toFixed(2)).toBe("4.19");
    expect((Math.log(weights.vocabulary) / Math.LN2).toFixed(2)).toBe("6.04");
    // The honest comparison is only the order; the GRU got 240 steps and the transformer 8,000.
    expect(metadata.evaluation.heldout_tail.transformer).toBeLessThan(metadata.evaluation.heldout_tail.rnn);
    expect(metadata.evaluation.heldout_tail.rnn).toBeLessThan(metadata.evaluation.heldout_tail.bigram_fitted_on_training_split);
  });

  it("the induction check: held-out periods, worst-case stripe at offset P-1, thresholds 0.40 and 0.25 (standard.mdx, card-info)", () => {
    const diagnostic = metadata.transformer.induction_diagnostic;
    expect(diagnostic.passes_shipping_threshold).toBe(true);
    expect(diagnostic.gate).toMatchObject({ random_stripe_min: 0.4, corpus_stripe_min: 0.25 });
    const [layerOne, layerTwo] = diagnostic.scores;
    const [corpusOne, corpusTwo] = diagnostic.corpus_scores;
    // Layer 2 heads 2 and 3 are the induction heads: 0.83 and 0.81 on random strings, 0.40 and 0.51 on corpus text.
    expect(layerTwo[1].toFixed(2)).toBe("0.83");
    expect(layerTwo[2].toFixed(2)).toBe("0.81");
    expect(corpusTwo[1].toFixed(2)).toBe("0.40");
    expect(corpusTwo[2].toFixed(2)).toBe("0.51");
    expect(diagnostic.head_qualifies).toEqual([
      [false, false, false, false],
      [false, true, true, false],
    ]);
    // Every other head sits under 0.06 on random strings and under 0.04 on corpus text at each held-out
    // period. The worst-case scores above are each head's minimum over the periods, which flatters a
    // head that does not copy, so these bounds take the maximum over every period instead.
    const { random: heldRandom, corpus: heldCorpus } = diagnostic.held_out;
    const nonInduction = (cell: { induction: number[][] }) => [...cell.induction[0], cell.induction[1][0], cell.induction[1][3]];
    const periods = ["11", "13", "17", "19"] as const;
    expect(Math.max(...periods.flatMap((period) => nonInduction(heldRandom[period])))).toBeLessThan(0.06);
    expect(Math.max(...periods.flatMap((period) => nonInduction(heldCorpus[period])))).toBeLessThan(0.04);
    expect(Math.max(...layerOne)).toBeLessThan(0.01); // "best layer-1 head under 0.01"
    expect(diagnostic.best_layer).toBe(1);
    expect(diagnostic.best_score).toBeCloseTo(0.8270, 3);
  });

  it("on the held-out periods the loss on the second copy falls from about 4.3 to 0.13-0.25 nats (random strings) and from 1.6-1.7 to 0.39-0.53 (corpus text)", () => {
    const { random, corpus } = metadata.transformer.induction_diagnostic.held_out;
    const periods = ["11", "13", "17", "19"] as const;
    const range = (kind: typeof random, key: "loss_first_half" | "loss_second_half") =>
      periods.map((period) => kind[period][key]);
    expect(Math.min(...range(random, "loss_first_half")).toFixed(2)).toBe("4.27");
    expect(Math.max(...range(random, "loss_first_half")).toFixed(2)).toBe("4.32");
    expect(Math.min(...range(random, "loss_second_half")).toFixed(2)).toBe("0.13");
    expect(Math.max(...range(random, "loss_second_half")).toFixed(2)).toBe("0.25");
    expect(Math.min(...range(corpus, "loss_first_half")).toFixed(1)).toBe("1.6");
    expect(Math.max(...range(corpus, "loss_first_half")).toFixed(1)).toBe("1.7");
    expect(Math.min(...range(corpus, "loss_second_half")).toFixed(2)).toBe("0.39");
    expect(Math.max(...range(corpus, "loss_second_half")).toFixed(2)).toBe("0.53");
  });

  it("Layer 1 heads 3 and 4 put 0.99 or more of their weight on the previous character on every diagnostic input; no other head reaches 0.4 (the next, Layer 2 head 1, is about a third on corpus text) (standard.mdx, glossary)", () => {
    const { random, corpus } = metadata.transformer.induction_diagnostic.held_out;
    for (const kind of [random, corpus]) {
      for (const period of ["11", "13", "17", "19"] as const) {
        const previous = kind[period].previous_token;
        expect(previous[0][2]).toBeGreaterThanOrEqual(0.99);
        expect(previous[0][3]).toBeGreaterThanOrEqual(0.99);
        for (const other of [previous[0][0], previous[0][1], ...previous[1]]) expect(other).toBeLessThan(0.4);
      }
    }
  });

  it("no head of the shipped model is a fixed 7-back look-up: the recorded 7-back score is under 0.07 for every head at held-out periods, and equals the stripe only at period 8 (standard.mdx)", () => {
    const { held_out, reference } = metadata.transformer.induction_diagnostic;
    for (const kind of [held_out.random, held_out.corpus]) {
      for (const period of ["11", "13", "17", "19"] as const) {
        for (const row of kind[period].lookback_7) for (const value of row) expect(value).toBeLessThan(0.07);
      }
    }
    // At period 8 the successor sits 7 back, so the two measures coincide.
    expect(reference.random["8"].lookback_7).toEqual(reference.random["8"].induction);
  });

  it("re-running the induction check on the shipped file reproduces the recorded stripes at periods it never trained on", () => {
    const strings = 24;
    const random = mulberry32(942);
    const recorded = metadata.transformer.induction_diagnostic.held_out.random;
    for (const period of [11, 13, 17, 19]) {
      const scores = Array.from({ length: LAYERS }, () => new Array<number>(HEADS).fill(0));
      for (let index = 0; index < strings; index += 1) {
        const ids = repeatedDistinct(random, period);
        const run = runTransformer(weights, ids);
        for (let layer = 0; layer < LAYERS; layer += 1) {
          for (let head = 0; head < HEADS; head += 1) {
            let sum = 0;
            for (let query = period; query < 2 * period; query += 1) {
              sum += run.attention[layer][(head * ids.length + query) * ids.length + (query - period + 1)];
            }
            scores[layer][head] += sum / period / strings;
          }
        }
      }
      const reference = recorded[String(period) as "11"].induction;
      for (let layer = 0; layer < LAYERS; layer += 1) {
        for (let head = 0; head < HEADS; head += 1) {
          expect(Math.abs(scores[layer][head] - reference[layer][head])).toBeLessThan(0.05);
        }
      }
      // Layer 2 heads 2 and 3 are well past the 0.40 gate; nothing else is close.
      expect(scores[1][1]).toBeGreaterThan(0.7);
      expect(scores[1][2]).toBeGreaterThan(0.7);
      expect(Math.max(scores[0][0], scores[0][1], scores[0][2], scores[0][3], scores[1][0], scores[1][3])).toBeLessThan(0.1);
    }
  }, 60_000);

  it("the lab's example inputs are the ones this file uses (Explore.tsx EXAMPLES, state.ts DEFAULT_TEXT)", () => {
    const explore = text("src/modules/attention/Explore.tsx");
    for (const example of [NAME, MOVED, REPEAT_8, REPEAT_9]) expect(explore.includes(`"${example}"`)).toBe(true);
    expect(explore.includes("text: DEFAULT_TEXT")).toBe(true);
    expect(text("src/modules/attention/state.ts").includes(`"${PRONOUN}"`)).toBe(true);
  });

});

// ---------------------------------------------------------------------------
// The pronoun sentence (standard.mdx "Test a route before you believe it",
// plain.mdx "A route that is not what it looks like", card-info)
// ---------------------------------------------------------------------------

describe("pronoun sentence routes", () => {
  it("no head's strongest route from it is animal; the strongest per head is because, it, it, it, it, because, cross, it", () => {
    const strongest = ALL_HEADS.map((head) => labView(PRONOUN, head.layer, head.head, "it").strongestWord);
    expect(strongest).toEqual(["because", "it", "it", "it", "it", "because", "cross", "it"]);
    expect(strongest).not.toContain("animal");
    // Even the largest share any head gives animal is small.
    for (const head of ALL_HEADS) {
      const view = labView(PRONOUN, head.layer, head.head, "it");
      const animal = view.wordWeights[view.spans.findIndex((span) => span.text === "animal")];
      expect(animal, `${label(head)} animal`).toBeLessThan(0.1);
    }
  });

  it("Layer 1 head 3 puts 99.9% of it on the word it, because it reads the previous character, the i inside it (standard.mdx, plain.mdx, card-info)", () => {
    for (const head of [3, 4]) {
      const view = labView(PRONOUN, 1, head, "it");
      expect(view.strongestWord).toBe("it");
      expect(percent(view.strongestWeight)).toBe("99.9%");
      expect(view.strongestOffset).toBe(1);
      expect(view.strongestCharacter).toBe("i");
    }
    // Layer 1 head 1 puts 78.6% on because; Layer 1 head 2 puts 99.4% on it.
    expect(percent(labView(PRONOUN, 1, 1, "it").strongestWeight)).toBe("78.6%");
    expect(percent(labView(PRONOUN, 1, 2, "it").strongestWeight)).toBe("99.4%");
  });

  it("Layer 1 reads a fixed distance: head 1 about 3 back, head 2 2 back, heads 3 and 4 one back, on every query of the sentence (standard.mdx, plain.mdx)", () => {
    const run = forward(PRONOUN);
    const fromQuery = 5;
    expect(meanWeightAtOffset(run, 1, 1, 3, fromQuery)).toBeGreaterThan(0.6);
    expect(meanWeightAtOffset(run, 1, 1, 3, fromQuery)).toBeLessThan(0.7); // "about two thirds"
    expect(meanWeightAtOffset(run, 1, 2, 2, fromQuery)).toBeGreaterThan(0.98);
    expect(meanWeightAtOffset(run, 1, 3, 1, fromQuery)).toBeGreaterThan(0.99);
    expect(meanWeightAtOffset(run, 1, 4, 1, fromQuery)).toBeGreaterThan(0.99);
  });

  it("with the 64 learned position vectors zeroed, those Layer 1 stripes fall from 66-100% to under 6% (standard.mdx, plain.mdx, module.ts glossary)", () => {
    const withoutPositions = { ...weights, positionEmbedding: new Float32Array(weights.positionEmbedding.length) };
    const run = runTransformer(withoutPositions, toIds(PRONOUN));
    for (const [head, offset] of [[1, 3], [2, 2], [3, 1], [4, 1]] as const) {
      expect(meanWeightAtOffset(run, 1, head, offset, 5), `L1H${head}`).toBeLessThan(0.06);
      expect(meanWeightAtOffset(forward(PRONOUN), 1, head, offset, 5), `L1H${head} intact`).toBeGreaterThan(0.6);
    }
    // "between 66 and 100 percent": the weakest intact stripe (Layer 1 head 1, 3 back) is 66%, the strongest 100%.
    const intact = [[1, 3], [2, 2], [3, 1], [4, 1]].map(([head, offset]) => meanWeightAtOffset(forward(PRONOUN), 1, head, offset, 5));
    expect(Math.round(Math.min(...intact) * 100)).toBe(66);
    expect(Math.round(Math.max(...intact) * 100)).toBe(100);
  });

  it("every Words-view row sums to exactly 100% in all eight heads (standard.mdx, card-info)", () => {
    const characters = Array.from(PRONOUN);
    const spans = wordSpans(characters);
    const run = forward(PRONOUN);
    for (const { layer, head } of ALL_HEADS) {
      for (const span of spans) {
        const total = aggregateByWord(characterRow(run, layer, head, span.representative), spans).reduce((a, b) => a + b, 0);
        expect(Math.abs(total - 1)).toBeLessThan(1e-5);
        expect(percent(total)).toBe("100.0%");
      }
    }
  });

  it("the On spaces readout for it: Layer 1 head 2 puts 99.1% on spaces only because two characters back is a space; for because it is about 1% (standard.mdx, plain.mdx, card-info)", () => {
    const it = labView(PRONOUN, 1, 2, "it");
    expect(percent(it.spaces)).toBe("99.1%");
    expect(it.strongestOffset).toBe(2);
    expect(it.strongestCharacter).toBe(" ");
    const because = labView(PRONOUN, 1, 2, "because");
    expect(because.spaces).toBeLessThan(0.02);
    expect(percent(because.spaces)).toBe("1.3%");
    expect(percent(labView(PRONOUN, 2, 4, "it").spaces)).toBe("61.7%");
    // Layer 2 head 4 is the only Layer 2 head above half on spaces for it.
    const layerTwo = [1, 2, 3, 4].filter((head) => labView(PRONOUN, 2, head, "it").spaces > 0.5);
    expect(layerTwo).toEqual([4]);
  });
});

// ---------------------------------------------------------------------------
// Repeated name (Layer 2 head 3): a route that follows content
// ---------------------------------------------------------------------------

describe("Repeated name and Name moved earlier", () => {
  it("Layer 2 head 3 sends the second Mark to Jones, the word after the earlier Mark, with 92.7% then 97.2% (standard.mdx, plain.mdx, card-info)", () => {
    const named = labView(NAME, 2, 3, "Mark");
    expect(named.strongestWord).toBe("Jones");
    expect(percent(named.strongestWeight)).toBe("92.7%");
    const moved = labView(MOVED, 2, 3, "Mark");
    expect(moved.strongestWord).toBe("Jones");
    expect(percent(moved.strongestWeight)).toBe("97.2%");
  });

  it("the route lands on the character after the earlier Mark, so its distance changes (34 back, then 19 back) while its target does not (standard.mdx, plain.mdx)", () => {
    for (const [value, distance] of [[NAME, 34], [MOVED, 19]] as const) {
      const view = labView(value, 2, 3, "Mark");
      expect(view.strongestOffset).toBe(distance);
      expect(view.strongestCharacter).toBe(" ");
      // The earlier Mark ends one character before the attended space.
      const earlier = value.indexOf("Mark") + "Mark".length;
      expect(view.position - view.strongestOffset).toBe(earlier);
    }
  });

  it("with no earlier occurrence to match (a new surname) the route disappears: no word gets 20% from Layer 2 head 3 (standard.mdx)", () => {
    const fresh = labView(NEW_SURNAME, 2, 3, "Smith");
    expect(Math.max(...fresh.wordWeights)).toBeLessThan(0.2);
  });
});

// ---------------------------------------------------------------------------
// Repeats every 8 / every 9 (standard.mdx "How to tell copying from a positional
// habit", plain.mdx, card-info "Attention matrix")
// ---------------------------------------------------------------------------

describe("Repeats every 8 and every 9", () => {
  it("period 8: Layer 2 heads 2 and 3 have their stripes at 7, 15 and 23 and put 94% and 88% on the true next character (standard.mdx, plain.mdx)", () => {
    const run = forward(REPEAT_8);
    expect([1, 2, 3].map((k) => 8 * k - 1)).toEqual([7, 15, 23]);
    expect(topOffsets(run, 2, 2, 23, 3)).toEqual([7, 15, 23]);
    expect(topOffsets(run, 2, 3, 23, 3)).toEqual([7, 15, 23]);
    expect(Math.round(trueNextCharacterShare(run, 2, 2, 8) * 100)).toBe(94);
    expect(Math.round(trueNextCharacterShare(run, 2, 3, 8) * 100)).toBe(88);
  });

  it("period 8: the other six heads put at most 11% on the true next character (standard.mdx, plain.mdx)", () => {
    for (const run of [forward(REPEAT_8)]) {
      for (const head of ALL_HEADS.filter(({ layer, head }) => !(layer === 2 && (head === 2 || head === 3)))) {
        expect(trueNextCharacterShare(run, head.layer, head.head, 8), label(head)).toBeLessThan(0.11);
      }
    }
  });

  it("period 9: the induction stripes move to 8, 17 and 26 (98.5% and 96.2% of mean weight) and leave 7, 15 and 23 (under 1%) (standard.mdx, plain.mdx, card-info)", () => {
    const run = forward(REPEAT_9);
    expect([1, 2, 3].map((k) => 9 * k - 1)).toEqual([8, 17, 26]);
    const at = (head: number, offsets: number[]) =>
      offsets.reduce((sum, offset) => sum + meanWeightAtOffset(run, 2, head, offset, 26), 0);
    expect(topOffsets(run, 2, 2, 26, 3)).toEqual([8, 17, 26]);
    expect(topOffsets(run, 2, 3, 26, 3)).toEqual([8, 17, 26]);
    expect(percent(at(2, [8, 17, 26]))).toBe("98.5%");
    expect(percent(at(3, [8, 17, 26]))).toBe("96.2%");
    expect(at(2, [7, 15, 23])).toBeLessThan(0.01);
    expect(at(3, [7, 15, 23])).toBeLessThan(0.01);
    expect(Math.round(trueNextCharacterShare(run, 2, 2, 9) * 100)).toBe(95);
    expect(Math.round(trueNextCharacterShare(run, 2, 3, 9) * 100)).toBe(89);
  });

  it("period 9: the other six heads put at most 11% on the true next character (standard.mdx, plain.mdx)", () => {
    const run = forward(REPEAT_9);
    for (const head of ALL_HEADS.filter(({ layer, head }) => !(layer === 2 && (head === 2 || head === 3)))) {
      expect(trueNextCharacterShare(run, head.layer, head.head, 9), label(head)).toBeLessThan(0.11);
    }
  });

  it("the fixed-distance heads stay put when the period changes: Layer 1 heads 1 to 4 keep their strongest offset at 3, 2, 1 and 1 at both periods (standard.mdx, plain.mdx)", () => {
    for (const value of [REPEAT_8, REPEAT_9]) {
      const run = forward(value);
      expect([1, 2, 3, 4].map((head) => topOffsets(run, 1, head, 26, 1)[0])).toEqual([3, 2, 1, 1]);
    }
  });
});

// ---------------------------------------------------------------------------
// What zero-ablation shows, and what it does not (standard.mdx "What this model
// demonstrates, and what it does not")
// ---------------------------------------------------------------------------

describe("switching heads off, the evidence the lesson calls limited", () => {
  const period = 13;
  const sample = (() => {
    const random = mulberry32(7);
    return Array.from({ length: 10 }, () => repeatedDistinct(random, period));
  })();
  /** Mean loss on the repeated half, from its second character on, with the given heads switched off. */
  function repeatedHalfLoss(off: Array<{ layer: number; head: number }> = []) {
    const headScale = [0, 1].map((layer) =>
      [0, 1, 2, 3].map((head) => (off.some((entry) => entry.layer === layer + 1 && entry.head === head + 1) ? 0 : 1)),
    );
    let total = 0;
    let count = 0;
    for (const ids of sample) {
      const run = runTransformer(weights, ids, { headScale });
      for (let position = period; position < ids.length - 1; position += 1) {
        const probabilities = softmax(run.logits.subarray(position * weights.vocabulary, (position + 1) * weights.vocabulary));
        total += -Math.log(probabilities[ids[position + 1]]);
        count += 1;
      }
    }
    return total / count;
  }

  it("switching off Layer 2 head 3 raises the repeated-half loss more than tenfold; Layer 2 heads 1 and 4 matter little; Layer 1 head 2, which is not a previous-token head, costs about as much as Layer 1 head 4", () => {
    const intact = repeatedHalfLoss();
    const layerTwoHeadThree = repeatedHalfLoss([{ layer: 2, head: 3 }]);
    expect(intact).toBeLessThan(0.3);
    expect(layerTwoHeadThree / intact).toBeGreaterThan(10);
    expect(Math.abs(repeatedHalfLoss([{ layer: 2, head: 1 }]) - intact)).toBeLessThan(0.2);
    expect(Math.abs(repeatedHalfLoss([{ layer: 2, head: 4 }]) - intact)).toBeLessThan(0.2);
    const layerOneHeadTwo = repeatedHalfLoss([{ layer: 1, head: 2 }]);
    const layerOneHeadFour = repeatedHalfLoss([{ layer: 1, head: 4 }]);
    expect(layerOneHeadTwo).toBeGreaterThan(2);
    expect(Math.abs(layerOneHeadTwo - layerOneHeadFour)).toBeLessThan(0.5);
  }, 60_000);
});

// ---------------------------------------------------------------------------
// The real ONNX Runtime path the lab executes, and the controls built on its
// query, key and value tensors: Mask (IM-6), Score scaling and Value vectors (IM-7)
// ---------------------------------------------------------------------------

describe("ONNX Runtime, the lab's own execution path and the controls on its tensors", () => {
  let session: ort.InferenceSession;

  beforeAll(async () => {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ["wasm"] });
  }, 60_000);

  async function onnxRun(value: string) {
    const ids = toIds(value);
    const output = await session.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
    const floats = (tensor: ort.Tensor) => Float32Array.from(tensor.data as ArrayLike<number>);
    return {
      length: ids.length,
      q: floats(output.q),
      k: floats(output.k),
      v: floats(output.v),
      attention: floats(output.attention),
      qkvShape: [...output.q.dims],
      attentionShape: [...output.attention.dims],
    };
  }

  /** One head's q, k, v and attention as the lab reads them (Explore.tsx headData). */
  async function headTensors(value: string, layer: number, head: number) {
    const result = await onnxRun(value);
    const pick = (tensor: Float32Array, shape: number[]) => headMatrix(tensor, shape, layer - 1, head - 1)!;
    return {
      length: result.length,
      q: pick(result.q, result.qkvShape),
      k: pick(result.k, result.qkvShape),
      v: pick(result.v, result.qkvShape),
      attention: pick(result.attention, result.attentionShape),
    };
  }

  it("the engine, ONNX Runtime's attention tensor and the recomputed softmax row all agree (the 'about 1e-6' of standard.mdx)", async () => {
    let worstRecompute = 0;
    let worstEngine = 0;
    const gap = (left: ArrayLike<number>, right: ArrayLike<number>) => {
      let largest = 0;
      for (let index = 0; index < left.length; index += 1) largest = Math.max(largest, Math.abs(left[index] - right[index]));
      return largest;
    };
    for (const value of [PRONOUN, NAME, MOVED, REPEAT_8, REPEAT_9]) {
      const result = await onnxRun(value);
      expect(result.qkvShape).toEqual([LAYERS, 1, HEADS, result.length, 64]);
      expect(result.attentionShape).toEqual([LAYERS, 1, HEADS, result.length, result.length]);
      const run = forward(value);
      for (const { layer, head } of ALL_HEADS) {
        const q = headMatrix(result.q, result.qkvShape, layer - 1, head - 1)!;
        const k = headMatrix(result.k, result.qkvShape, layer - 1, head - 1)!;
        const modelRows = headMatrix(result.attention, result.attentionShape, layer - 1, head - 1)!;
        const recomputed = attentionMatrix(q, k);
        for (let position = 0; position < result.length; position += 1) {
          const model = modelRows[position].slice(0, position + 1);
          worstRecompute = Math.max(worstRecompute, gap(recomputed[position].slice(0, position + 1), model));
          worstEngine = Math.max(worstEngine, gap(characterRow(run, layer, head, position).slice(0, position + 1), model));
        }
      }
    }
    // "within about 1e-6": the wasm provider agrees with the recomputed row and the engine to a few millionths.
    expect(worstRecompute).toBeLessThan(2e-6);
    expect(worstEngine).toBeLessThan(3e-6);
  }, 60_000);

  it("reading the headline figures straight from the ONNX Runtime tensors gives the lab's 99.9%, 92.7% and 97.2%", async () => {
    const wordView = async (value: string, layer: number, head: number, queryWord: string) => {
      const tensors = await headTensors(value, layer, head);
      const characters = Array.from(value);
      const spans = wordSpans(characters);
      const query = spans[spans.map((span) => span.text).lastIndexOf(queryWord)].representative;
      const weights = aggregateByWord(tensors.attention[query], spans);
      const strongest = weights.indexOf(Math.max(...weights));
      return { word: spans[strongest].text, weight: weights[strongest] };
    };
    const itWord = await wordView(PRONOUN, 1, 3, "it");
    expect([itWord.word, percent(itWord.weight)]).toEqual(["it", "99.9%"]);
    const named = await wordView(NAME, 2, 3, "Mark");
    expect([named.word, percent(named.weight)]).toEqual(["Jones", "92.7%"]);
    const moved = await wordView(MOVED, 2, 3, "Mark");
    expect([moved.word, percent(moved.weight)]).toEqual(["Jones", "97.2%"]);
  }, 60_000);

  // ------------------------- Mask (IM-6) -------------------------

  it("the lab's causal matrix, the attention-math causalMatrix and the model's tensor are one matrix; a bidirectional row sums to 100% over every character", async () => {
    for (const head of [{ layer: 1, head: 3 }, { layer: 2, head: 4 }]) {
      const tensors = await headTensors(PRONOUN, head.layer, head.head);
      const causal = causalMatrix(tensors.q, tensors.k);
      const explicit = attentionMatrix(tensors.q, tensors.k, { causal: true, scaled: true });
      const open = bidirectionalMatrix(tensors.q, tensors.k);
      expect(open).toHaveLength(tensors.length);
      causal.forEach((row, position) => {
        expect(row).toEqual(explicit[position]);
        for (let key = 0; key <= position; key += 1) expect(Math.abs(row[key] - tensors.attention[position][key])).toBeLessThan(2e-6);
        for (let key = position + 1; key < tensors.length; key += 1) expect(row[key]).toBe(0);
        expect(open[position]).toHaveLength(tensors.length);
        expect(Math.abs(open[position].reduce((sum, value) => sum + value, 0) - 1)).toBeLessThan(1e-9);
      });
    }
  }, 60_000);

  it("lifting the mask multiplies every earlier weight by the same factor, one minus the weight now on later characters (standard.mdx, card-info, checkpoint)", async () => {
    for (const { layer, head } of ALL_HEADS) {
      const tensors = await headTensors(PRONOUN, layer, head);
      const causal = causalMatrix(tensors.q, tensors.k);
      const open = bidirectionalMatrix(tensors.q, tensors.k);
      for (let position = 0; position < tensors.length - 1; position += 1) {
        const factor = 1 - laterShare(open[position], position);
        for (let key = 0; key <= position; key += 1) {
          expect(Math.abs(open[position][key] - causal[position][key] * factor), `${label({ layer, head })} ${position},${key}`).toBeLessThan(1e-9);
        }
      }
    }
  }, 60_000);

  it("on the pronoun sentence, lifting the mask moves this much of it's weight onto the ten later characters, by head (standard.mdx, plain.mdx, card-info)", async () => {
    const spans = wordSpans(Array.from(PRONOUN));
    const position = spans.find((span) => span.text === "it")!.representative;
    expect(Array.from(PRONOUN).length - position - 1).toBe(10); // " was tired"
    const shares: string[] = [];
    for (const { layer, head } of ALL_HEADS) {
      const tensors = await headTensors(PRONOUN, layer, head);
      shares.push(percent(laterShare(bidirectionalMatrix(tensors.q, tensors.k)[position], position)));
    }
    // L1H1 L1H2 L1H3 L1H4 | L2H1 L2H2 L2H3 L2H4
    expect(shares).toEqual(["29.2%", "1.8%", "0.1%", "0.2%", "78.0%", "56.3%", "33.7%", "90.2%"]);
  }, 60_000);

  it("with the mask lifted, Layer 2 heads 2 and 3 put 33.8% and 30.6% of it on the very next character, the answer a causal model must predict (standard.mdx, card-info)", async () => {
    const spans = wordSpans(Array.from(PRONOUN));
    const position = spans.find((span) => span.text === "it")!.representative;
    const nextShare = async (head: number) => {
      const tensors = await headTensors(PRONOUN, 2, head);
      return bidirectionalMatrix(tensors.q, tensors.k)[position][position + 1];
    };
    expect(percent(await nextShare(2))).toBe("33.8%");
    expect(percent(await nextShare(3))).toBe("30.6%");
  }, 60_000);

  it("Layer 1's queries and keys depend only on the character and its slot, so the lifted mask is exact for Layer 1; Layer 2's do not (matrix card, maskNote)", async () => {
    // Two sentences that share only the character b at slot 1.
    const first = await onnxRun("abcdefgh");
    const second = await onnxRun("xbzwvuts");
    const slot = (tensor: Float32Array, shape: number[], layer: number, head: number) => headMatrix(tensor, shape, layer, head)![1];
    let layerOne = 0;
    let layerTwo = 0;
    for (let head = 0; head < HEADS; head += 1) {
      for (const [one, two] of [[first.q, second.q], [first.k, second.k]] as const) {
        const a = slot(one, first.qkvShape, 0, head);
        const b = slot(two, second.qkvShape, 0, head);
        a.forEach((value, index) => (layerOne = Math.max(layerOne, Math.abs(value - b[index]))));
        const c = slot(one, first.qkvShape, 1, head);
        const d = slot(two, second.qkvShape, 1, head);
        c.forEach((value, index) => (layerTwo = Math.max(layerTwo, Math.abs(value - d[index]))));
      }
    }
    expect(layerOne).toBeLessThan(1e-4);
    expect(layerTwo).toBeGreaterThan(0.05);
  }, 60_000);

  it("lifting the mask turns the first query's row, which a causal row pins to 100% on itself, into a spread over every character", async () => {
    const tensors = await headTensors(PRONOUN, 2, 2);
    expect(causalMatrix(tensors.q, tensors.k)[0][0]).toBe(1);
    expect(bidirectionalMatrix(tensors.q, tensors.k)[0][0]).toBeLessThan(0.5);
  }, 60_000);

  it("the Words view reads a lifted row over every word, and reports weight on trailing spaces or past the 12th word as not drawn", async () => {
    const value = "a b ";
    const spans = wordSpans(Array.from(value));
    // Two words, then one trailing space no word owns: a uniform row over four characters leaves 25% undrawn.
    const row = [0.25, 0.25, 0.25, 0.25];
    expect(aggregateByWord(row, spans).reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(0.75, 12);
    expect(unownedShare(row, spans)).toBeCloseTo(0.25, 12);
    expect(unownedShare([0.5, 0.5], wordSpans(Array.from("a b")))).toBe(0);
  });

  // ------------------------- Score scaling (IM-7) -------------------------

  it("without the division the softmax sees 8 times larger scores and never gives a row a smaller largest weight (standard.mdx, card-info)", async () => {
    for (const { layer, head } of ALL_HEADS) {
      const tensors = await headTensors(PRONOUN, layer, head);
      for (let position = 0; position < tensors.length; position += 1) {
        const scaled = attentionRow(tensors.q[position], tensors.k, position, { causal: true, scaled: true });
        const raw = attentionRow(tensors.q[position], tensors.k, position, { causal: true, scaled: false });
        expect(raw.scale).toBe(1);
        expect(scaled.scale).toBe(8);
        for (let key = 0; key <= position; key += 1) expect(raw.scaledScores[key]).toBeCloseTo(scaled.scaledScores[key] * 8, 3);
        expect(Math.max(...raw.weights)).toBeGreaterThanOrEqual(Math.max(...scaled.weights) - 1e-12);
      }
    }
  }, 60_000);

  it("Layer 2 head 2 has no row above 99% on one character with the division and 26 of its 55 rows without it; no head has fewer without (standard.mdx, card-info)", async () => {
    const counts: Record<string, { scaled: number; raw: number }> = {};
    for (const head of ALL_HEADS) {
      const tensors = await headTensors(PRONOUN, head.layer, head.head);
      const scaled = saturatedRows(attentionMatrix(tensors.q, tensors.k, { scaled: true }), true);
      const raw = saturatedRows(attentionMatrix(tensors.q, tensors.k, { scaled: false }), true);
      expect(scaled.total).toBe(55);
      expect(raw.total).toBe(55);
      expect(raw.saturated, label(head)).toBeGreaterThanOrEqual(scaled.saturated);
      counts[label(head)] = { scaled: scaled.saturated, raw: raw.saturated };
    }
    expect(counts.L2H2).toEqual({ scaled: 0, raw: 26 });
    expect(counts.L2H4).toEqual({ scaled: 0, raw: 39 });
  }, 60_000);

  it("the saturated-row count is 'rows with one weight at least 99%', excluding a causal first row whose single key always weighs 100%", () => {
    expect(saturatedRows([[1], [0.6, 0.4], [0.995, 0.003, 0.002]], true)).toEqual({ saturated: 1, total: 2 });
    expect(saturatedRows([[1, 0], [0.6, 0.4]], false)).toEqual({ saturated: 1, total: 2 });
  });

  // ------------------------- Value vectors (IM-7) -------------------------

  it("zeroing every value leaves the weights exactly as they were and makes the mix the zero vector (standard.mdx, card-info, checkpoint)", async () => {
    for (const { layer, head } of [{ layer: 2, head: 2 }, { layer: 1, head: 3 }]) {
      const tensors = await headTensors(PRONOUN, layer, head);
      const position = tensors.length - 1;
      const before = attentionRow(tensors.q[position], tensors.k, position);
      const zeroed = zeroValues(tensors.v, 0, tensors.length);
      const after = attentionRow(tensors.q[position], tensors.k, position);
      expect(after.weights).toEqual(before.weights);
      const mixed = mixRow(before.weights, zeroed);
      expect(vectorLength(mixed)).toBe(0);
      expect(vectorLength(mixRow(before.weights, tensors.v))).toBeGreaterThan(0.1);
    }
  }, 60_000);

  it("zeroing one word's values removes exactly that word's share of the mix: mix(before) - mix(after) = the sum of weight x value over its characters", async () => {
    const tensors = await headTensors(PRONOUN, 2, 2);
    const spans = wordSpans(Array.from(PRONOUN));
    const query = spans.find((span) => span.text === "it")!.representative;
    const inspected = spans.find((span) => span.text === "because")!;
    const row = tensors.attention[query];
    const before = mixRow(row, tensors.v);
    const after = mixRow(row, zeroValues(tensors.v, inspected.start, inspected.end));
    const contribution = Array.from({ length: 64 }, (_, dimension) => {
      let total = 0;
      for (let key = inspected.start; key < inspected.end; key += 1) total += row[key] * tensors.v[key][dimension];
      return total;
    });
    before.forEach((value, dimension) => expect(Math.abs(value - after[dimension] - contribution[dimension])).toBeLessThan(1e-9));
    // The word total the Word total readout shows is the weight those values carried.
    expect(aggregateByWord(row, spans)[spans.indexOf(inspected)]).toBeGreaterThan(0.1);
    expect(vectorLength(after)).not.toBe(vectorLength(before));
  }, 60_000);
});
