import { describe, expect, it } from "vitest";
import { TINY_CORPORA, tinyCrossEntropy, trainTinyModel } from "@app/module-sdk";
import { buildFineTuneData, SOURCE_SENTENCES, TARGET_SENTENCES } from "./holdout";
import { HELD_OUT_MAX, REPLAY_MAX, REPLAY_STEP } from "./state";

const base = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 60, seed: 1 });

describe("the replay and held-out split", () => {
  it("splits the eight recipe sentences and replays whole harbor sentences", () => {
    expect(TARGET_SENTENCES).toHaveLength(8);
    expect(SOURCE_SENTENCES).toHaveLength(8);
    const data = buildFineTuneData(3, 50);
    expect(data.trainSentences).toBe(5);
    expect(data.heldSentences).toBe(3);
    expect(data.replaySentences).toBe(4);
    expect(data.trainTarget).toBe(`${TARGET_SENTENCES.slice(0, 5).join(". ")}.`);
    expect(data.heldTarget).toBe(`${TARGET_SENTENCES.slice(5).join(". ")}.`);
    expect(data.replay).toBe(`${SOURCE_SENTENCES.slice(0, 4).join(". ")}.`);
    expect(data.notReplayed).toBe(`${SOURCE_SENTENCES.slice(4).join(". ")}.`);
    expect(data.text).toBe(`${data.trainTarget} ${data.replay}`);
    expect(data.replayFraction).toBeCloseTo(4 / 9, 10);
  });

  it("never lets a held-out sentence into the fine-tuning text, at any setting", () => {
    for (let held = 0; held <= HELD_OUT_MAX; held += 1) {
      for (let share = 0; share <= REPLAY_MAX; share += REPLAY_STEP) {
        const data = buildFineTuneData(held, share);
        for (const sentence of TARGET_SENTENCES.slice(TARGET_SENTENCES.length - held)) {
          expect(data.text, `${held} held, ${share}% replay`).not.toContain(sentence);
        }
        // One slider notch is exactly one original sentence.
        expect(data.replaySentences).toBe(Math.round(share / REPLAY_STEP));
      }
    }
  });

  it("holds nothing out and replays nothing at the neutral settings, giving back the whole target corpus", () => {
    const data = buildFineTuneData(0, 0);
    expect(data.heldTarget).toBe("");
    expect(data.replay).toBe("");
    expect(data.notReplayed).toBe(`${SOURCE_SENTENCES.join(". ")}.`);
    // The same encoded text as the original corpus, so the first card's run is reproduced exactly.
    const original = trainTinyModel({ text: TINY_CORPORA.recipes.text, epochs: 20, learningRate: 0.6, init: base.weights, seed: 2, checkpoints: 9 });
    const split = trainTinyModel({ text: data.text, epochs: 20, learningRate: 0.6, init: base.weights, seed: 2, checkpoints: 9 });
    expect(Array.from(split.weights)).toEqual(Array.from(original.weights));
    expect(tinyCrossEntropy(split.weights, data.trainTarget)).toBe(tinyCrossEntropy(original.weights, TINY_CORPORA.recipes.text));
  });

  it("clamps what it is given instead of trusting it", () => {
    expect(buildFineTuneData(99, 1e9).heldSentences).toBe(7);
    expect(buildFineTuneData(-4, -50).heldSentences).toBe(0);
    expect(buildFineTuneData(-4, -50).replaySentences).toBe(0);
    expect(buildFineTuneData(3, 1e9).replaySentences).toBe(8);
    expect(buildFineTuneData(3, 100).notReplayed).toBe("");
  });
});
