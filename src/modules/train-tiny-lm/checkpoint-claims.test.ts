import { describe, expect, it } from "vitest";
import { TINY_CORPORA, trainTinyFactored, trainTinyModel } from "@app/module-sdk";

/**
 * Pins the claim in the second checkpoint question's explanation: switching corpus leaves the
 * weight count (set by Model size alone) unchanged but changes the training text, so the step
 * count follows the corpus. Mirrors splitCorpus in Explore.tsx.
 */
function trainSplit(text: string) {
  const sentences = text.split(".").map((part) => part.trim()).filter(Boolean);
  const held = Math.max(1, Math.round(sentences.length * 0.25));
  return `${sentences.slice(0, sentences.length - held).join(". ")}.`;
}
const recipe = { batchSize: 16, learningRate: 0.5, seed: 9 } as const;

describe("train-tiny-lm checkpoint claims", () => {
  it("keeps the weight count when the corpus changes but not the step count or the training text", () => {
    const harbor = trainSplit(TINY_CORPORA.harbor.text);
    const proverbs = trainSplit(TINY_CORPORA.proverbs.text);
    expect(harbor).not.toBe(proverbs);
    const runs = [harbor, proverbs].map((text) => trainTinyFactored({ text, rank: 8, epochs: 20, ...recipe }));
    expect(runs[0].parameters).toBe(runs[1].parameters);
    expect(runs[0].parameters).toBe(60 * 8);
    const full = [harbor, proverbs].map((text) => trainTinyModel({ text, epochs: 20, ...recipe }));
    expect(full[0].steps).not.toBe(full[1].steps);
  });
});
