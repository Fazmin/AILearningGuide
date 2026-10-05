// @vitest-environment node
/// <reference types="node" />
/**
 * Pins the "Layer by layer" card (lens.ts): the readout after the embeddings, layer 1 and layer 2 of
 * the shipped transformer, on the lab's default prompt and on a few contrast prompts. The weights
 * are read out of the same tiny-transformer.onnx the lab ships, with the SDK's TypeScript forward
 * pass, so every figure here is what the card prints. Plan items IM-5, IM-10.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { teachingTransformer } from "@app/module-sdk";
import vocabulary from "./assets/transformer-vocab.json";
import { LENS_STAGES, logitLens, readout } from "./lens";
import { DEFAULT_PROMPT, TRAINING_LINE } from "./prompts";
import { encode, entropyBits, formatPercent } from "./sampling";

const weights = teachingTransformer.weightsFromOnnx(
  // new Uint8Array copies, which is what the ONNX reader wants outside a browser.
  new Uint8Array(readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx"))),
);
const itos = vocabulary.itos as string[];
const lens = (prompt: string) => logitLens(weights, encode(prompt, vocabulary));
const topCharacters = (prompt: string, stage: number) => lens(prompt).stages[stage].top.map((entry) => itos[entry.index]);

describe("the readout is the model's own output layer", () => {
  it("is a 256 by 66 unembedding with no bias, behind a final norm (glossary, card-info)", () => {
    expect(weights.width).toBe(256);
    expect(weights.vocabulary).toBe(66);
    expect(weights.blocks).toHaveLength(2);
    expect(weights.unembedding).toHaveLength(256 * 66);
    expect(weights.finalNormWeight).toHaveLength(256);
  });

  it("reproduces the model's own logits after layer 2, to rounding error (card-info 'Layer by layer')", () => {
    for (const prompt of [DEFAULT_PROMPT, TRAINING_LINE, "the", "To be, or not to be, that is the question", "ZZZZZZZZZZZZZZZZ wherefore art thou J"]) {
      expect(lens(prompt).readoutError, prompt).toBeLessThan(1e-5);
    }
  });

  it("applies the same readout to any residual vector, so earlier stages are read through the final norm", () => {
    const ids = encode(DEFAULT_PROMPT, vocabulary);
    const run = teachingTransformer.runTransformer(weights, ids);
    const last = ids.length - 1;
    const afterLayer1 = readout(weights, teachingTransformer.rowOf(run.residuals[0], last, 256));
    const result = lens(DEFAULT_PROMPT);
    result.stages[1].logits.forEach((value, index) => expect(value).toBeCloseTo(afterLayer1[index], 9));
    // The three stages are three different readouts, not one repeated.
    expect(result.stages[0].logits).not.toEqual(result.stages[1].logits);
    expect(result.stages[1].logits).not.toEqual(result.stages[2].logits);
  });

  it("has the three stages the card names, in order", () => {
    expect(LENS_STAGES.map((stage) => stage.name)).toEqual(["After embeddings", "After layer 1", "After layer 2"]);
    expect(lens(DEFAULT_PROMPT).stages.map((stage) => stage.id)).toEqual(["embeddings", "layer-1", "layer-2"]);
  });

  it("reads at most the last 64 characters, as the model does", () => {
    const long = DEFAULT_PROMPT.repeat(4);
    expect(long.length).toBeGreaterThan(64);
    const result = lens(long);
    expect(result.length).toBe(64);
    expect(result.stages[2].top[0].p).toBeGreaterThan(0.5);
  });
});

describe("the default prompt, stage by stage (card-info, standard.mdx 'What to notice', plain.mdx)", () => {
  const result = lens(DEFAULT_PROMPT);
  const [embeddings, layer1, layer2] = result.stages;

  it("the finished model's first choice is u, at 99%", () => {
    expect(itos[result.finalIndex]).toBe("u");
    expect(formatPercent(layer2.finalProbability)).toBe("99%");
  });

  it("u ranks 56th after the embeddings, 3rd after layer 1 and 1st after layer 2", () => {
    expect(result.stages.map((stage) => stage.finalRank)).toEqual([56, 3, 1]);
    expect(formatPercent(embeddings.finalProbability)).toBe("0.1%");
    expect(formatPercent(layer1.finalProbability)).toBe("6.6%");
  });

  it("the answer first heads the list after layer 2, and layer 1 puts o on top, not u", () => {
    expect(result.firstStage).toBe(2);
    expect(LENS_STAGES[result.firstStage].name).toBe("After layer 2");
    expect(topCharacters(DEFAULT_PROMPT, 1)[0]).toBe("o");
    expect(formatPercent(layer1.top[0].p)).toBe("19%");
  });

  it("entropy is 4.85, 4.52 and 0.09 bits: nearly all of the drop is in layer 2", () => {
    expect(result.stages.map((stage) => stage.entropyBits.toFixed(2))).toEqual(["4.85", "4.52", "0.09"]);
    expect(embeddings.entropyBits - layer1.entropyBits).toBeLessThan(0.5);
    expect(layer1.entropyBits - layer2.entropyBits).toBeGreaterThan(4);
    // The stage's reported entropy is the entropy of its probabilities.
    expect(entropyBits(Array.from(layer2.probabilities))).toBeCloseTo(layer2.entropyBits, 12);
  });

  it("the readouts are probability distributions over 66 characters", () => {
    for (const stage of result.stages) {
      expect(stage.probabilities).toHaveLength(66);
      expect(Array.from(stage.probabilities).reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 9);
      expect(stage.top).toHaveLength(5);
    }
  });

  it("the after-embeddings top characters are not the answer or anything near it (honest limit)", () => {
    expect(topCharacters(DEFAULT_PROMPT, 0)).toEqual(["'", "t", "h", "o", "m"]);
  });
});

describe("what the earlier stages can and cannot know", () => {
  it("after the embeddings the readout depends only on the last character and its position (card-info, checkpoint)", () => {
    // Same length (both names have six letters) and same last character, different earlier words.
    const a = lens("O Juliet, Juliet! wherefore art thou J");
    const b = lens("O Hamlet, Hamlet! wherefore art thou J");
    expect("O Juliet, Juliet! wherefore art thou J".length).toBe("O Hamlet, Hamlet! wherefore art thou J".length);
    expect(Array.from(a.stages[0].logits)).toEqual(Array.from(b.stages[0].logits));
    // The layers mix the earlier characters in, so the finished readouts differ.
    expect(a.stages[1].logits).not.toEqual(b.stages[1].logits);
    expect(a.stages[2].top[0].p - b.stages[2].top[0].p).toBeGreaterThan(0.05);
  });

  it("changing both Juliets to Jason moves layer 2's top character from u to a (step 5)", () => {
    const jason = lens("O Jason, Jason! wherefore art thou J");
    expect(itos[jason.finalIndex]).toBe("a");
    expect(formatPercent(jason.stages[2].top[0].p)).toBe("87%");
    expect(jason.firstStage).toBe(2);
    // The answer is not near the top before layer 2: layer 2 is where it appears.
    expect(jason.stages[1].finalRank).toBeGreaterThan(10);
  });

  it("switching off layer 2 heads 2 and 3 puts u back on top for the Jason prompt: the answer a is copied there (standard.mdx 'Going deeper')", () => {
    const ids = encode("O Jason, Jason! wherefore art thou J", vocabulary);
    const finalRow = (headScale?: number[][]) => {
      const run = teachingTransformer.runTransformer(weights, ids, headScale ? { headScale } : {});
      return teachingTransformer.softmax(teachingTransformer.rowOf(run.logits, ids.length - 1, 66));
    };
    const top = (p: ArrayLike<number>) => Array.from(p).indexOf(Math.max(...Array.from(p)));
    const intact = finalRow();
    expect(itos[top(intact)]).toBe("a");
    expect(formatPercent(intact[top(intact)])).toBe("87%");
    // Zero-ablation of layer 2 heads 2 and 3 (indices 1 and 2), the two induction heads the model card names.
    const ablated = finalRow([
      [1, 1, 1, 1],
      [1, 0, 0, 1],
    ]);
    expect(itos[top(ablated)]).toBe("u");
    expect(formatPercent(ablated[top(ablated)])).toBe("82%");
    // On the default prompt u was already the likely answer, so the same edit costs it little.
    const defaultIds = encode(DEFAULT_PROMPT, vocabulary);
    const defaultRun = teachingTransformer.runTransformer(weights, defaultIds, {
      headScale: [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
      ],
    });
    const defaultAblated = teachingTransformer.softmax(teachingTransformer.rowOf(defaultRun.logits, defaultIds.length - 1, 66));
    expect(itos[top(defaultAblated)]).toBe("u");
    expect(formatPercent(defaultAblated[top(defaultAblated)])).toBe("78%");
  });

  it("the entropy can rise from the embeddings to layer 1 on other text (card-info 'Layer by layer' notice)", () => {
    const result = lens("To be, or not to be, that is the question");
    expect(result.stages[1].entropyBits).toBeGreaterThan(result.stages[0].entropyBits);
    expect(result.stages.map((stage) => stage.entropyBits.toFixed(2))).toEqual(["2.62", "3.34", "2.86"]);
  });

  it("for the training line, layer 1 already puts the answer first and layer 2 makes it certain", () => {
    const result = lens(TRAINING_LINE);
    expect(itos[result.finalIndex]).toBe("o");
    expect(result.firstStage).toBe(1);
    expect(result.stages[2].entropyBits).toBeLessThan(0.01);
  });
});
