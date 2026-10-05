import { describe, expect, it } from "vitest";
import { describeRecipe, formatRecipe, parseRecipe, RECIPE_HEADER, RECIPE_MAX_LENGTH } from "./recipe";
import { readSettings, sanitizeState } from "./state";

const settings = readSettings({});

describe("the shareable recipe", () => {
  it("writes the default recipe as one line of key=value pairs", () => {
    expect(formatRecipe(settings)).toBe(
      "tiny-lm-recipe v1; corpus=harbor; rank=8; epochs=20; lr=0.5; schedule=constant; batch=16; clip=off; data=cleaned; seed=9",
    );
    expect(formatRecipe(settings).startsWith(RECIPE_HEADER)).toBe(true);
    expect(describeRecipe(settings)).toBe("20 epochs · batch 16 · learning rate 0.5 · constant · no clipping · seed 9");
  });

  it("reads back exactly what it wrote, for every setting the controls can reach", () => {
    const states = [
      readSettings({ corpusId: "proverbs", rank: 12, epochs: 30, learningRate: 16, schedule: "one-cycle", batchSize: 64, clipNorm: 1.5, data: "raw" }),
      readSettings({ corpusId: "recipes", rank: 1, epochs: 1, learningRate: 0.125, schedule: "cosine", batchSize: 2, clipNorm: 0.05 }),
      readSettings({ clipNorm: 0.35, learningRate: 4, schedule: "warmup-decay" }),
    ];
    for (const state of states) {
      const parsed = parseRecipe(formatRecipe(state));
      expect(parsed).toEqual({
        corpusId: state.corpusId,
        rank: state.rank,
        epochs: state.epochs,
        learningRate: state.learningRate,
        schedule: state.schedule,
        batchSize: state.batchSize,
        clipNorm: state.clipNorm,
        data: state.data,
      });
    }
  });

  it("applies a pasted recipe through the same validation as any other state", () => {
    const parsed = parseRecipe("tiny-lm-recipe v1; lr=3; batch=1000; clip=40; rank=99; epochs=-5; schedule=cosine");
    expect(parsed).toEqual({ learningRate: 4, batchSize: 64, clipNorm: 1.5, rank: 12, epochs: 1, schedule: "cosine" });
    expect(sanitizeState({ ...settings, ...parsed }).learningRate).toBe(4);
  });

  it("skips values that are not options and keys it does not know, and ignores the seed", () => {
    expect(parseRecipe("corpus=<img src=x>; schedule=sawtooth; data=all; seed=1; colour=blue")).toBeNull();
    expect(parseRecipe("lr=fast; batch=; epochs=Infinity")).toBeNull();
    expect(parseRecipe("")).toBeNull();
    expect(parseRecipe("no pairs here")).toBeNull();
    expect(parseRecipe("seed=1; lr=2")).toEqual({ learningRate: 2 });
    expect(parseRecipe("LR = 2 ; Schedule = Cosine")).toEqual({ learningRate: 2, schedule: "cosine" });
  });

  it("caps how much of a pasted recipe it reads", () => {
    const padded = `${" ".repeat(RECIPE_MAX_LENGTH)}lr=2`;
    expect(parseRecipe(padded)).toBeNull();
    expect(parseRecipe(`lr=2${";x=".repeat(5000)}`)).toEqual({ learningRate: 2 });
  });
});
