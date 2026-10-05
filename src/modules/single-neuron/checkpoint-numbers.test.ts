import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import { activate, cornersCorrect, levelChord, slope, weightedSum } from "./neuron";
import definition from "./module";

/**
 * Pins the figures and comparisons the checkpoint questions rely on, so a change to
 * neuron.ts cannot silently make a question's answer wrong.
 */
const defaults = { w1: 1.2, w2: -0.8, bias: -0.1 };
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

describe("single-neuron checkpoint questions", () => {
  it("question 1: x = (1.00, 0.50) with the default weights gives z = 0.7, and each wrong option is a named slip", () => {
    expect(weightedSum(1, 0.5, defaults)).toBeCloseTo(0.7, 12);
    expect(correctOption(0)).toMatch(/^0\.7:/);
    // 1.1: only the first product, then the bias.
    expect(1 * defaults.w1 + defaults.bias).toBeCloseTo(1.1, 12);
    // 0.8: both products, bias left out.
    expect(1 * defaults.w1 + 0.5 * defaults.w2).toBeCloseTo(0.8, 12);
    // 1.5: the negative weight treated as positive, then the bias.
    expect(1 * defaults.w1 + 0.5 * Math.abs(defaults.w2) + defaults.bias).toBeCloseTo(1.5, 12);
    expect(questions[0].options.map((option) => option.split(":")[0]).sort()).toEqual(["0.7", "0.8", "1.1", "1.5"]);
  });

  it("question 2: from z = 3 to z = 6 sigmoid barely moves, ReLU doubles and step stays at 1", () => {
    expect(activate(3, "sigmoid")).toBeGreaterThan(0.95);
    expect(activate(6, "sigmoid") - activate(3, "sigmoid")).toBeLessThan(0.05);
    expect(activate(6, "relu")).toBe(2 * activate(3, "relu"));
    expect(activate(3, "step")).toBe(1);
    expect(activate(6, "step")).toBe(1);
  });

  it("question 3: slope is about 0 for a saturated sigmoid, for ReLU below zero, and for step away from the jump", () => {
    expect(slope(6, "sigmoid")).toBeLessThan(0.003);
    expect(slope(-6, "sigmoid")).toBeLessThan(0.003);
    expect(slope(-1, "relu")).toBe(0);
    expect(slope(1, "step")).toBe(0);
    expect(slope(-1, "step")).toBe(0);
  });

  it("question 4: doubling w and b keeps the line and pushes sigmoid away from 0.5; doubling only the weights moves the line", () => {
    const doubled = { w1: 2.4, w2: -1.6, bias: -0.2 };
    expect(levelChord(doubled)).toEqual(levelChord(defaults));
    const before = activate(weightedSum(0.7, 0.35, defaults), "sigmoid");
    const after = activate(weightedSum(0.7, 0.35, doubled), "sigmoid");
    expect(Math.abs(after - 0.5)).toBeGreaterThan(Math.abs(before - 0.5));
    expect(levelChord({ ...doubled, bias: -0.1 })).not.toEqual(levelChord(defaults));
  });

  it("question 5: no setting of the sliders beyond the grid helps XOR, because scaling does not move the line", () => {
    let best = 0;
    for (let w1 = -8; w1 <= 8.001; w1 += 0.5) {
      for (let w2 = -8; w2 <= 8.001; w2 += 0.5) {
        for (let bias = -8; bias <= 8.001; bias += 0.5) {
          best = Math.max(best, cornersCorrect("xor", { w1, w2, bias }));
        }
      }
    }
    expect(best).toBe(3);
  });
});
