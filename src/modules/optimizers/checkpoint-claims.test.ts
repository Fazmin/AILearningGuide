import { describe, expect, it } from "vitest";
import { checkpointQuestions, newtonSchulz, teachingMemory } from "@app/module-sdk";
import definition from "./module";
import { adamSpec, baseSpec, LARGE_PARAMETERS, muonSpectra, rowsKept, rowSteps, runSweepNow, trainRun, type AdamSettings } from "./race";
import { newtonSchulzScalar, singularValues, ZERO_SINGULAR } from "./spectrum";

/**
 * Pins the factual claims in the checkpoint explanations, question by question, against the lab's own
 * runs (seed 1 unless a line says otherwise) and the memory arithmetic.
 */
const adam = (extra: Partial<AdamSettings> = {}) =>
  adamSpec({ rate: 0.03, biasCorrection: true, warmup: 0, weightDecay: 0, decayMode: "decoupled", ...extra });
const three = (value: number) => value.toFixed(3);
const perVisit = (run: ReturnType<typeof trainRun>, label: string) => rowSteps(run).find((row) => row.label === label)!.perVisit;

describe("optimizers checkpoint claims", () => {
  it("has eight questions covering all five objectives", () => {
    const questions = checkpointQuestions(definition);
    expect(questions).toHaveLength(8);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("question 1: SGD moves q 0.59 times as far per visit as space, AdamW 14.9 times, at a rate 33 times smaller", () => {
    const sgd = trainRun(baseSpec("sgd", 1));
    const adamw = trainRun(baseSpec("adamw", 0.03));
    expect((perVisit(sgd, "q") / perVisit(sgd, "space")).toFixed(2)).toBe("0.59");
    expect((perVisit(adamw, "q") / perVisit(adamw, "space")).toFixed(1)).toBe("14.9");
    expect(Math.floor(1 / 0.03)).toBe(33);
    // Under SGD the rarely seen q row travels less per visit than the busy space row; under AdamW, more.
    expect(perVisit(sgd, "space")).toBeGreaterThan(perVisit(sgd, "q"));
    expect(perVisit(adamw, "q")).toBeGreaterThan(perVisit(adamw, "space"));
  });

  it("question 2: without the correction the first step is 3.16 times the rate, 0.949 at 0.3, and the peak 7.853", () => {
    const corrected = trainRun(adam({ rate: 0.3 }));
    const raw = trainRun(adam({ rate: 0.3, biasCorrection: false }));
    expect((0.1 / Math.sqrt(0.001)).toFixed(2)).toBe("3.16");
    expect([three(corrected.firstStepMove), three(raw.firstStepMove)]).toEqual(["0.300", "0.949"]);
    expect([three(raw.earlyPeakLoss), three(corrected.earlyPeakLoss)]).toEqual(["7.853", "3.499"]);
  });

  it("question 3: a 10% warmup lowers the uncorrected peak from 7.853 to 3.849, and 20% keeps it at 3.401", () => {
    const raw = trainRun(adam({ rate: 0.3, biasCorrection: false }));
    const ten = trainRun(adam({ rate: 0.3, biasCorrection: false, warmup: 0.1 }));
    const twenty = trainRun(adam({ rate: 0.3, biasCorrection: false, warmup: 0.2 }));
    expect([three(raw.earlyPeakLoss), three(ten.earlyPeakLoss), three(twenty.earlyPeakLoss)]).toEqual(["7.853", "3.849", "3.401"]);
    expect(twenty.earlyPeakLoss).toBe(twenty.batchLoss[0]);
  });

  it("question 4: at rate 0.03 and decay 0.1, L2 ends near 3.27 with norm 1.19 and q keeping 0.12%; decoupled near 2.06, 59.45 and 60%", () => {
    const reference = trainRun(adam());
    const l2 = trainRun(adam({ weightDecay: 0.1, decayMode: "l2" }));
    const decoupled = trainRun(adam({ weightDecay: 0.1 }));
    expect([l2.heldOut.toFixed(2), decoupled.heldOut.toFixed(2)]).toEqual(["3.27", "2.06"]);
    expect([l2.weightNorm.toFixed(2), decoupled.weightNorm.toFixed(2)]).toEqual(["1.19", "59.45"]);
    const q = (run: ReturnType<typeof trainRun>) => rowsKept(run, reference).find((row) => row.label === "q")!.kept;
    expect((q(l2) * 100).toFixed(2)).toBe("0.12");
    expect(Math.round(q(decoupled) * 100)).toBe(60);
    // Decoupled decay takes the same 0.3% of every weight each step.
    expect(1 - 0.03 * 0.1).toBeCloseTo(0.997, 12);
  });

  it("question 5: at Step 1 the 21 zero directions stay zero, and the 9 others go from 0.211–0.537 to 0.682–1.121", () => {
    const zero = muonSpectra(0.05, "first", 0)!;
    const five = muonSpectra(0.05, "first", 5)!;
    expect([zero.before.nonzero, five.after.zeros]).toEqual([9, 21]);
    expect([three(zero.before.smallestNonzero), three(zero.before.largest)]).toEqual(["0.211", "0.537"]);
    expect([three(five.after.smallestNonzero), three(five.after.largest)]).toEqual(["0.682", "1.121"]);
    // The polynomial sends zero to zero, and so does the matrix iteration on a zero matrix.
    expect(newtonSchulzScalar(0, 5)).toBe(0);
    expect(singularValues(newtonSchulz(new Float64Array(9), 3, 3, 5), 3, 3).every((value) => value <= ZERO_SINGULAR)).toBe(true);
  });

  it("question 6: tuned Muon finishes last at 2.059 against AdamW's 2.041", () => {
    const table = runSweepNow();
    const best = Object.fromEntries(table.map((row) => [row.optimizer, row.best.heldOut]));
    expect([three(best.muon), three(best.adamw)]).toEqual(["2.059", "2.041"]);
    expect(Math.max(...Object.values(best))).toBe(best.muon);
  });

  it("question 7: SGD falls from 2.116 at rate 1 to 2.041 at rate 3, a 0.0003 gap to AdamW, inside a 0.006–0.011 seed spread", () => {
    const table = runSweepNow();
    const sgd = table.find((row) => row.optimizer === "sgd")!;
    const adamw = table.find((row) => row.optimizer === "adamw")!;
    expect(three(trainRun(baseSpec("sgd", 1)).heldOut) + " " + three(trainRun(baseSpec("adamw", 0.03)).heldOut)).toBe("2.116 2.041");
    expect([sgd.best.rate, three(sgd.best.heldOut), three(sgd.atDefault.heldOut)]).toEqual([3, "2.041", "2.116"]);
    expect(adamw.best.rate).toBe(0.03);
    expect(Math.abs(sgd.best.heldOut - adamw.best.heldOut).toFixed(4)).toBe("0.0003");
    expect(three(trainRun(baseSpec("sgd", 1)).heldOut - trainRun(baseSpec("adamw", 0.03)).heldOut)).toBe("0.075");
    const spread = (spec: ReturnType<typeof baseSpec>) => {
      const values = [1, 2, 3, 4, 5].map((seed) => trainRun({ ...spec, seed }).heldOut);
      return Math.max(...values) - Math.min(...values);
    };
    expect([three(spread(baseSpec("adamw", 0.03))), three(spread(baseSpec("sgd", 3)))]).toEqual(["0.006", "0.011"]);
  });

  it("question 8: AdamW keeps 60 GB of fp32 state at 7.5 billion weights, Muon 30 GB; mixed precision 120 GB against 90 GB", () => {
    const state = (row: teachingMemory.OptimizerId) => teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "fp32", row).optimizerState;
    expect([state("adam"), state("momentum")]).toEqual([60e9, 30e9]);
    expect(teachingMemory.memoryBreakdown(1, "fp32", "adam").stateBytesPerParameter).toBe(8);
    expect(teachingMemory.memoryBreakdown(1, "fp32", "momentum").stateBytesPerParameter).toBe(4);
    const total = (row: teachingMemory.OptimizerId) => teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "mixed", row).total;
    expect([total("adam"), total("momentum")]).toEqual([120e9, 90e9]);
    // Muon's trainer keeps one float per weight, its momentum; the orthogonalized matrix is not kept.
    expect(trainRun(baseSpec("muon", 0.05)).stateFloats).toBe(900);
    expect(trainRun(baseSpec("adamw", 0.03)).stateFloats).toBe(1800);
  });
});
