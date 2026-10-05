// @vitest-environment node
/// <reference types="node" />
/**
 * Renders the lab with the real models' outputs (ONNX Runtime for the GRU and transformer, the
 * SDK's TypeScript forward pass for the logit lens, the shipped bigram table) and checks that what
 * a learner reads in the cards is what quoted.test.ts and lens.test.ts pin from the same tensors.
 * The two data hooks are replaced by the values this file computes; everything else is the lab's own code.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { teachingTransformer, type ModuleState } from "@app/module-sdk";
import bigram from "./assets/bigram.json";
import vocabulary from "./assets/transformer-vocab.json";
import { logitLens } from "./lens";
import { DEFAULT_PROMPT, TRAINING_LINE } from "./prompts";
import { encode } from "./sampling";

const shared = vi.hoisted(() => ({
  models: undefined as unknown as (ids: ReadonlyArray<number>) => unknown,
  lens: undefined as unknown as (ids: ReadonlyArray<number>) => unknown,
}));
vi.mock("./useCharModels", () => ({ useCharModels: (ids: ReadonlyArray<number>) => shared.models(ids) }));
vi.mock("./useLogitLens", () => ({ useLogitLens: (ids: ReadonlyArray<number>) => shared.lens(ids) }));

const { default: definition } = await import("./module");

const onnxBytes = new Uint8Array(readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx")));
const weights = teachingTransformer.weightsFromOnnx(onnxBytes);
const VOCAB = 66;

let rnn: ort.InferenceSession;
let transformer: ort.InferenceSession;
beforeAll(async () => {
  ort.env.wasm.numThreads = 1;
  rnn = await ort.InferenceSession.create(new Uint8Array(readFileSync(resolve(process.cwd(), "src/modules/next-token-prediction/assets/character-rnn.onnx"))), {
    executionProviders: ["wasm"],
  });
  transformer = await ort.InferenceSession.create(onnxBytes, { executionProviders: ["wasm"] });
}, 60_000);

async function rowsFrom(session: ort.InferenceSession, ids: number[]) {
  const output = await session.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
  const data = output.logits.data as Float32Array;
  return Array.from({ length: ids.length }, (_, row) => Array.from(data.subarray(row * VOCAB, (row + 1) * VOCAB)));
}

/** Makes the mocked hooks serve the real outputs for `prompt`. */
async function serve(prompt: string) {
  const ids = encode(prompt, vocabulary);
  const rnnRows = await rowsFrom(rnn, ids);
  const transformerRows = await rowsFrom(transformer, ids.slice(-64));
  const view = (rows: number[][], offset: number) => ({ status: "ready", rows, offset, provider: "wasm", stale: false });
  shared.models = () => ({
    bigram: { status: "ready", rows: ids.map((id) => bigram.probabilities[id].map(Math.log)), offset: 0, stale: false },
    rnn: view(rnnRows, 0),
    transformer: view(transformerRows, ids.length - Math.min(ids.length, 64)),
  });
  shared.lens = () => ({ status: "ready", result: logitLens(weights, ids), stale: false });
}

/** The visible text of the rendered lab: tags become spaces so a label and its value read in sequence. */
const visible = (state: ModuleState, step = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, { state, setState: () => undefined, currentStep: step, mode: "standard", narrate: () => undefined }),
  )
    .replace(/<[^>]*>/g, " ")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&gt;/g, ">")
    .replace(/&lt;/g, "<")
    .replace(/\s+/g, " ");

describe("the default prompt on screen", () => {
  it("says the prompt was not in training, and shows the models' top characters", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState });
    expect(text).toContain("Seen in training? No. This exact text is not in Tiny Shakespeare");
    // The compare card lists each model's top three characters with percentages, in column order
    // (each row is read out once for screen readers and once as the visible bar label).
    expect(text).toContain('Bigram table sees only the last character "U" 33% U 33% "u" 22% u 22% "o" 12% o 12%');
    expect(text).toContain('Character RNN (GRU) sees all 38 characters, through a 2-layer, 192-wide state "o" 26% o 26% "a" 22% a 22% "u" 10% u 10%');
    expect(text).toContain('Tiny transformer sees the last 38 characters, each directly "u" 99% u 99% "o" 0.5% o 0.5% "a" 0.2% a 0.2%');
  });

  it("prints the held-out and whole-corpus losses with the training behind each, in the compare card", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState });
    for (const figure of ["3.58 bits", "3.04 bits", "2.23 bits", "3.54 bits", "3.06 bits", "1.83 bits"]) {
      expect(text, figure).toContain(figure);
    }
    expect(text).toContain("8,000 training steps");
    expect(text).toContain("240 training steps");
    expect(text).toContain("not a fair test of the architectures");
    expect(text).toContain("for 8,000 steps and the shipped GRU for 240");
    // The retired claims are gone.
    expect(text).not.toMatch(/undertrained|does worse than both|Loss on training text/);
  });

  it("states that nothing ends the text", async () => {
    await serve(DEFAULT_PROMPT);
    expect(visible({ ...definition.initialState })).toContain("no end-of-sequence token");
  });
});

describe("Layer by layer on screen", () => {
  it("reads the three stages for the default prompt, with entropy and where the answer first appears", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState }, 4);
    expect(text).toContain("Logit lens · the last character of the text · the 2-layer transformer");
    expect(text).toContain("What each stage of the model would predict");
    expect(text).toContain(
      "The finished model's first choice, “u”, first heads the list after layer 2. Where it ranks: 56th after the embeddings, 3rd after layer 1, 1st after layer 2.",
    );
    expect(text).toContain("4.85 → 4.52 → 0.09 bits");
    expect(text).toContain("After embeddings");
    expect(text).toContain("After layer 1");
    expect(text).toContain("After layer 2");
    expect(text).toContain("First on top after layer 2");
    expect(text).toMatch(/Entropy of this readout 0\.09 bits/);
    expect(text).toContain("lens reads here");
    expect(text).toContain("Final norm + unembedding");
    expect(text).toContain("256 numbers → 66 logits");
  });

  it("follows the prompt: with Jason the finished choice is a", async () => {
    const jason = "O Jason, Jason! wherefore art thou J";
    await serve(jason);
    const text = visible({ ...definition.initialState, prompt: jason }, 4);
    expect(text).toContain("The finished model's first choice, “a”");
    expect(text).toContain("87%");
  });

  it("shows a status instead of numbers while the lens is still loading", async () => {
    await serve(DEFAULT_PROMPT);
    shared.lens = () => ({ status: "loading", result: undefined, stale: false });
    const text = visible({ ...definition.initialState }, 4);
    expect(text).toContain("Reading the transformer's weights");
    expect(text).not.toContain("The finished model's first choice");
    expect(text).not.toContain("Entropy of this readout");
  });

  it("says so when the lens could not run, and keeps the other cards", async () => {
    await serve(DEFAULT_PROMPT);
    shared.lens = () => ({ status: "error", error: "HTTP 404", stale: false });
    const text = visible({ ...definition.initialState }, 4);
    expect(text).toContain("The layer-by-layer view could not run (HTTP 404). The other cards still work.");
    expect(text).toContain("Odds for the next character");
  });
});

describe("Seen in training? (the corpus is not shipped, so only the two preset lines are known)", () => {
  it("says yes for the training line, 43% of the way through and inside the first 90%", async () => {
    await serve(TRAINING_LINE);
    const text = visible({ ...definition.initialState, prompt: TRAINING_LINE });
    expect(text).toContain("Seen in training? Yes. This line is in Tiny Shakespeare, 43% of the way through, inside the first 90% the transformer trained on.");
    expect(text).toContain("The GRU and the bigram table saw the whole corpus.");
  });

  it("says Unknown for any other text rather than guessing", async () => {
    await serve("Good morrow to you all");
    const text = visible({ ...definition.initialState, prompt: "Good morrow to you all" });
    expect(text).toContain("Seen in training? Unknown. The lab does not ship the corpus");
  });
});

describe("Min-p on screen", () => {
  it("shows the cut-off and the characters it cut, for the RNN at temperature 1", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState, model: "rnn", temperature: 1, minP: 0.1 }, 2);
    expect(text).toContain("Min-p cut-off: keep a character if its probability ≥ p_min × p_max");
    expect(text).toContain("0.10 × 26% = 2.6%");
    expect(text).toContain("7 kept");
    expect(text).toContain("cut · min-p");
    expect(text).toContain("Min-p");
    expect(text).toContain("drop what falls below the min-p cut-off");
  });

  it("shows no min-p row while it is off, and reads off on the control", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState }, 2);
    expect(text).not.toContain("Min-p cut-off: keep a character");
    expect(text).toContain("off · 0.00");
    expect(text).not.toContain("cut · min-p");
  });

  it("keeps a single character for the transformer on the default prompt at min-p 0.10", async () => {
    await serve(DEFAULT_PROMPT);
    const text = visible({ ...definition.initialState, model: "transformer", temperature: 1, minP: 0.1 }, 2);
    expect(text).toContain("1 kept");
  });
});
