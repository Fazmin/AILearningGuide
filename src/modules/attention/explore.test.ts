// @vitest-environment node
/// <reference types="node" />
/**
 * Renders the lab with the real model's query, key, value and attention tensors (ONNX Runtime,
 * WebAssembly CPU provider) and checks that what a learner reads in the readout strip, the matrix
 * card and the inspector is what quoted.test.ts pins from the same tensors: the Mask, Score scaling
 * and Value vectors controls reach the screen with the right numbers. The runtime hook is replaced
 * by the tensors this file computes; everything else is the lab's own code.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import vocabulary from "./assets/transformer-vocab.json";

const shared = vi.hoisted(() => ({ result: undefined as unknown }));
vi.mock("@app/module-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@app/module-sdk")>();
  return {
    ...actual,
    useTeachingTransformer: () => ({
      status: shared.result ? "ready" : "idle",
      result: shared.result,
      error: "",
    }),
  };
});

const { default: definition } = await import("./module");

const PRONOUN = "The animal did not cross the street because it was tired";
const NAME = "Mark Jones met Anna Blake and then Mark";
const onnxBytes = new Uint8Array(readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx")));
const stoi = vocabulary.stoi as Record<string, number>;

let session: ort.InferenceSession;
beforeAll(async () => {
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ["wasm"] });
}, 60_000);

async function load(text: string) {
  const ids = Array.from(text).map((character) => stoi[character] ?? vocabulary.unk);
  const output = await session.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
  const floats = (tensor: ort.Tensor) => Float32Array.from(tensor.data as ArrayLike<number>);
  shared.result = {
    provider: "wasm",
    q: floats(output.q),
    k: floats(output.k),
    v: floats(output.v),
    attention: floats(output.attention),
    qkvShape: [...output.q.dims],
    attentionShape: [...output.attention.dims],
  };
}

/** The visible text of the rendered lab: tags become spaces so a label and its value read in sequence. */
function screen(state: ModuleState, step = 0) {
  const html = renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...state },
      setState: () => undefined,
      currentStep: step,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
  return html.replace(/<[^>]*>/g, " ").replace(/&#x27;/g, "'").replace(/\s+/g, " ");
}

/** Layer and head are 0-based in state; the lab shows them 1-based. */
const layerOneHeadThree = { text: PRONOUN, layer: 0, head: 2, selected: 8, inspected: 1 };
const layerTwoHeadFour = { text: PRONOUN, layer: 1, head: 3, selected: 8, inspected: 1 };
const layerTwoHeadTwo = { text: PRONOUN, layer: 1, head: 1, selected: 8, inspected: 7 };

describe("the lab with the real model's tensors", () => {
  it("renders the model's own routes with the model's own settings: Layer 1 head 3 sends it 99.9% to it", async () => {
    await load(PRONOUN);
    const view = screen(layerOneHeadThree);
    expect(view).toContain("ONNX model");
    expect(view).toMatch(/strongest route it weight 99\.9% on spaces 0\.0%/);
    expect(view).not.toContain("on later characters");
    expect(view).toContain("Layer 1 · head 3");
    expect(view).toMatch(/Softmax .* Model reports/);
  });

  it("Repeated name: Layer 2 head 3 sends the second Mark 92.7% to Jones", async () => {
    await load(NAME);
    expect(screen({ text: NAME, layer: 1, head: 2, selected: 7, inspected: 1 })).toMatch(/strongest route Jones weight 92\.7%/);
  });

  it("Mask: Bidirectional shows the share on later characters and the next character, notes what the view is, and keeps the weights above the diagonal", async () => {
    await load(PRONOUN);
    const causal = screen({ ...layerTwoHeadFour, mask: "causal" });
    const open = screen({ ...layerTwoHeadFour, mask: "bidirectional" });
    expect(causal).not.toContain("on later characters");
    expect(causal).not.toContain("mask lifted");
    expect(open).toMatch(/on later characters 90\.2% on the next character 11\.6%/);
    expect(open).toContain("mask lifted");
    expect(open).toContain("Mask lifted for this head only");
    expect(open).toContain("not what a trained bidirectional model would learn");
    // Layer 2 queries and keys were built from a causal Layer 1; Layer 1 is exact.
    expect(open).toContain("still built from a Layer 1 that stayed causal");
    expect(screen({ ...layerOneHeadThree, mask: "bidirectional" })).toContain("so for this head the numbers are exact");
    // Layer 1 heads 3 and 4 move almost nothing forward.
    expect(screen({ ...layerOneHeadThree, mask: "bidirectional" })).toMatch(/on later characters 0\.1% /);
  });

  it("Mask: the Characters view stops hatching and the weights-as-text table lists every character", async () => {
    await load(PRONOUN);
    const causalHtml = renderToStaticMarkup(
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...layerTwoHeadFour, matrixView: "characters", mask: "causal" },
        setState: () => undefined,
        currentStep: 0,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    );
    const openHtml = renderToStaticMarkup(
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...layerTwoHeadFour, matrixView: "characters", mask: "bidirectional" },
        setState: () => undefined,
        currentStep: 0,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    );
    expect(causalHtml).toContain("atn-char-heatmap__mask");
    expect(openHtml).not.toContain("atn-char-heatmap__mask");
    const rows = (html: string) => (html.match(/<tr>/g) ?? []).length;
    // "it" ends at character 45 of 56: 46 rows while masked, 56 once lifted (plus one header row).
    expect(rows(causalHtml)).toBe(46 + 1);
    expect(rows(openHtml)).toBe(56 + 1);
  });

  it("Score scaling: No scaling saturates Layer 2 head 2 (0 of 55 rows above 99% becomes 26) and says so", async () => {
    await load(PRONOUN);
    const scaled = screen({ ...layerTwoHeadTwo, scaling: "scaled" });
    const raw = screen({ ...layerTwoHeadTwo, scaling: "raw" });
    expect(scaled).toMatch(/Saturated rows 0 of 55/);
    expect(raw).toMatch(/Saturated rows 26 of 55/);
    expect(raw).toContain("Scaling is off");
    expect(raw).toContain("raw score q·k, no ÷ √d");
    expect(scaled).toContain("scaled score q·k ÷ √d");
    expect(scaled).toContain("÷ √64");
    const largest = (view: string) => Number(/Largest weight ([\d.]+)%/.exec(view)?.[1]);
    expect(largest(raw)).toBeGreaterThanOrEqual(largest(scaled));
  });

  it("Value vectors: Zero all leaves every weight as it was and drops the mix to zero; Zero word changes the mix by that word's share", async () => {
    await load(PRONOUN);
    const model = screen({ ...layerTwoHeadTwo, values: "model" });
    const all = screen({ ...layerTwoHeadTwo, values: "all" });
    const word = screen({ ...layerTwoHeadTwo, values: "inspected" });
    const mixLength = (view: string) => Number(/Mix length ([\d.]+)/.exec(view)?.[1]);
    expect(mixLength(all)).toBe(0);
    expect(mixLength(model)).toBeGreaterThan(0.1);
    expect(mixLength(word)).not.toBe(mixLength(model));
    for (const pattern of [/Selected weight ([\d.]+)%/, /Largest weight ([\d.]+)%/, /On spaces ([\d.]+)%/, /strongest route (\w+) weight ([\d.]+)%/]) {
      expect(pattern.exec(all)?.[0]).toBe(pattern.exec(model)?.[0]);
      expect(pattern.exec(word)?.[0]).toBe(pattern.exec(model)?.[0]);
    }
    expect(all).toContain("Every value is zero");
    expect(word).toContain("now carry zero values");
  });

  it("every control carries its label and the model's setting is pressed by default", async () => {
    await load(PRONOUN);
    const html = renderToStaticMarkup(
      createElement(definition.Explore, {
        state: { ...definition.initialState },
        setState: () => undefined,
        currentStep: 0,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    );
    for (const control of ["Matrix view", "Mask", "Score scaling", "Value vectors (V)"]) expect(html).toContain(`<legend>${control}</legend>`);
    const pressed = (label: string) => {
      const fieldset = new RegExp(`<legend>${label.replace(/[()]/g, "\\$&")}</legend><div>(.*?)</div>`).exec(html)?.[1] ?? "";
      return /aria-pressed="true"[^>]*>([^<]*)</.exec(fieldset)?.[1];
    };
    expect(pressed("Mask")).toBe("Causal");
    expect(pressed("Score scaling")).toBe("÷ √64");
    expect(pressed("Value vectors (V)")).toBe("Model");
    // Every step renders, including the new Lift the mask step.
    expect(definition.steps).toHaveLength(6);
    for (let step = 0; step < definition.steps.length; step += 1) expect(screen({}, step).length).toBeGreaterThan(200);
  });
});
