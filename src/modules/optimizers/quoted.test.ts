import { describe, expect, it } from "vitest";
import { bigramPairs, extendedScheduleFactor, teachingMemory, TINY_VOCAB, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import {
  adamSpec,
  BATCH_SIZE,
  baseSpec,
  DECAY_LADDER,
  EARLY_STEPS,
  EPOCHS,
  formatSmall,
  HELD_OUT_TEXT,
  idleShrink,
  LARGE_PARAMETERS,
  muonSpectra,
  rowsKept,
  rowSteps,
  runSweepNow,
  stateRows,
  TOTAL_STEPS,
  TRAIN_TEXT,
  trainRun,
  type AdamSettings,
  type OptimizerId,
  type RunSpec,
} from "./race";
import { BAND_HIGH, BAND_LOW, ZERO_SINGULAR } from "./spectrum";

/**
 * Pins every figure the three lessons and card-info.ts quote, as the lab prints them. All of them are
 * measured from the lab's own runs (seed 1 unless a line says otherwise), except the memory figures,
 * which are arithmetic, and Shrink per step, which is the closed-form idle-weight illustration.
 */
const three = (value: number) => value.toFixed(3);
const adam = (extra: Partial<AdamSettings> = {}): RunSpec =>
  adamSpec({ rate: 0.03, biasCorrection: true, warmup: 0, weightDecay: 0, decayMode: "decoupled", ...extra });
const defaults: Record<OptimizerId, RunSpec> = {
  sgd: baseSpec("sgd", 1),
  momentum: baseSpec("momentum", 0.1),
  adamw: baseSpec("adamw", 0.03),
  muon: baseSpec("muon", 0.05),
};
const run = (id: OptimizerId) => trainRun(defaults[id]);
const row = (spec: RunSpec, label: string) => rowSteps(trainRun(spec)).find((entry) => entry.label === label)!;
const seeds = (spec: RunSpec) => [1, 2, 3, 4, 5].map((seed) => trainRun({ ...spec, seed }).heldOut);

describe("the setup", () => {
  it("trains 900 weights on 378 pairs for 600 steps and holds out 118", () => {
    expect(TINY_VOCAB.length ** 2).toBe(900);
    expect(bigramPairs(TRAIN_TEXT).inputs.length).toBe(378);
    expect(bigramPairs(HELD_OUT_TEXT).inputs.length).toBe(118);
    expect([EPOCHS, BATCH_SIZE, TOTAL_STEPS, EARLY_STEPS]).toEqual([25, 16, 600, 60]);
    expect(run("sgd").trainCurve).toHaveLength(26);
    expect(three(UNIFORM_CROSS_ENTROPY)).toBe("3.401");
    // "averaged with fifteen other pairs"
    expect(BATCH_SIZE - 1).toBe(15);
  });

  it("counts the visits the lesson quotes", () => {
    const visits = (label: string) => row(defaults.sgd, label).visits;
    expect([visits("space"), visits("q"), visits("z"), visits("k")]).toEqual([580, 25, 25, 50]);
    const starts = (character: string) =>
      Array.from(bigramPairs(TRAIN_TEXT).inputs).filter((id) => id === TINY_VOCAB.indexOf(character)).length;
    expect([starts(" "), starts("q"), starts("z"), starts("k")]).toEqual([70, 1, 1, 2]);
  });

  it("warms up from 2% of the rate, 60 steps at 10%", () => {
    expect(extendedScheduleFactor("wsd", 0, 0.1, 0)).toBe(0.02);
    expect(Math.round(0.1 * (TOTAL_STEPS - 1))).toBe(60);
  });
});

describe("Race at the defaults", () => {
  it("matches the table under the chart", () => {
    expect([three(run("sgd").trainLoss), three(run("sgd").heldOut)]).toEqual(["2.040", "2.116"]);
    expect([three(run("momentum").trainLoss), three(run("momentum").heldOut)]).toEqual(["2.040", "2.115"]);
    expect([three(run("adamw").trainLoss), three(run("adamw").heldOut)]).toEqual(["1.819", "2.041"]);
    expect([three(run("muon").trainLoss), three(run("muon").heldOut)]).toEqual(["1.832", "2.059"]);
    expect(three(run("sgd").heldOut - run("adamw").heldOut)).toBe("0.075");
  });

  it("bottoms AdamW's held-out curve at epoch 13 while SGD's is still falling", () => {
    const curve = run("adamw").heldCurve;
    const lowest = Math.min(...curve);
    expect(three(lowest)).toBe("2.025");
    expect(curve.indexOf(lowest)).toBe(13);
    expect(three(curve[EPOCHS])).toBe("2.041");
    const sgd = run("sgd").heldCurve;
    for (let epoch = 20; epoch <= EPOCHS; epoch += 1) expect(sgd[epoch]).toBeLessThan(sgd[epoch - 1]);
  });

  it("puts momentum at 0.1 within 0.001 of SGD at 1, a velocity of up to ten times the gradient", () => {
    expect(Math.abs(run("momentum").heldOut - run("sgd").heldOut)).toBeLessThan(0.001);
    expect(Math.abs(run("momentum").trainLoss - run("sgd").trainLoss)).toBeLessThan(0.001);
    expect(1 / (1 - 0.9)).toBeCloseTo(10, 12);
  });
});

describe("Rare rows and busy rows", () => {
  it("matches the per-visit distances and q ÷ space", () => {
    const sgdSpace = row(defaults.sgd, "space").perVisit;
    const sgdQ = row(defaults.sgd, "q").perVisit;
    expect([three(sgdSpace), three(sgdQ), (sgdQ / sgdSpace).toFixed(2)]).toEqual(["0.101", "0.059", "0.59"]);
    const adamSpace = row(defaults.adamw, "space").perVisit;
    const adamQ = row(defaults.adamw, "q").perVisit;
    expect([three(adamSpace), three(adamQ), (adamQ / adamSpace).toFixed(1)]).toEqual(["0.036", "0.540", "14.9"]);
    // "33 times smaller" and "about nine times as far"
    expect(Math.floor(1 / 0.03)).toBe(33);
    expect((adamQ / sgdQ).toFixed(0)).toBe("9");
  });

  it("matches momentum's and Muon's rows", () => {
    expect([three(row(defaults.momentum, "q").perVisit), three(row(defaults.momentum, "space").perVisit)]).toEqual([
      "0.059",
      "0.022",
    ]);
    expect((row(defaults.muon, "q").perVisit / row(defaults.muon, "space").perVisit).toFixed(1)).toBe("9.4");
  });
});

describe("Break Adam's first step", () => {
  const at = (extra: Partial<AdamSettings>) => trainRun(adam({ rate: 0.3, ...extra }));

  it("matches the readouts at AdamW rate 0.3", () => {
    const corrected = at({});
    expect([three(corrected.firstStepMove), three(corrected.earlyPeakLoss), three(corrected.heldOut)]).toEqual([
      "0.300",
      "3.499",
      "2.213",
    ]);
    const raw = at({ biasCorrection: false });
    expect([three(raw.firstStepMove), three(raw.earlyPeakLoss), three(raw.heldOut)]).toEqual(["0.949", "7.853", "2.529"]);
    // The peak is on step 22, the 22nd batch of the first epoch.
    expect(raw.batchLoss.indexOf(raw.earlyPeakLoss) + 1).toBe(22);
    const warmed = at({ biasCorrection: false, warmup: 0.1 });
    expect([three(warmed.firstStepMove), three(warmed.earlyPeakLoss), three(warmed.heldOut)]).toEqual([
      "0.019",
      "3.849",
      "2.342",
    ]);
    const longer = at({ biasCorrection: false, warmup: 0.2 });
    expect(longer.earlyPeakLoss).toBe(longer.batchLoss[0]);
    expect(three(longer.earlyPeakLoss)).toBe("3.401");
    expect(three(at({ warmup: 0.1 }).heldOut)).toBe("2.187");
  });

  it("matches the 3.16 ratio, exactly what the first step shows", () => {
    expect((0.1 / Math.sqrt(0.001)).toFixed(2)).toBe("3.16");
    expect((at({ biasCorrection: false }).firstStepMove / at({}).firstStepMove).toFixed(2)).toBe("3.16");
  });

  it("matches the uncorrected inflation for a steady gradient: 3.16 at step 1, 6.6 at its step-12 peak, 1.5 at 600", () => {
    const inflation = (step: number) => (1 - 0.9 ** step) / Math.sqrt(1 - 0.999 ** step);
    const steps = Array.from({ length: TOTAL_STEPS }, (_, index) => index + 1);
    const peak = steps.reduce((best, step) => (inflation(step) > inflation(best) ? step : best), 1);
    expect([inflation(1).toFixed(2), peak, inflation(peak).toFixed(1), inflation(TOTAL_STEPS).toFixed(1)]).toEqual([
      "3.16",
      12,
      "6.6",
      "1.5",
    ]);
  });

  it("matches the default-rate comparison and the 0.09 cross-check", () => {
    const corrected = trainRun(adam());
    const raw = trainRun(adam({ biasCorrection: false }));
    expect([three(corrected.firstStepMove), three(raw.firstStepMove)]).toEqual(["0.030", "0.095"]);
    // Nothing spikes: both peaks are the first batch's uniform loss.
    expect(corrected.earlyPeakLoss).toBe(corrected.batchLoss[0]);
    expect(raw.earlyPeakLoss).toBe(raw.batchLoss[0]);
    expect([three(raw.heldOut), three(corrected.heldOut)]).toEqual(["2.103", "2.041"]);
    expect(three(trainRun(baseSpec("adamw", 0.09)).heldOut)).toBe("2.107");
    const filled = 1 - 0.999 ** TOTAL_STEPS;
    expect(filled).toBeLessThan(0.46);
    expect(filled).toBeGreaterThan(0.45);
  });
});

describe("Two ways to decay", () => {
  it("matches the readouts at rate 0.03 and decay 0.1", () => {
    const decoupled = trainRun(adam({ weightDecay: 0.1 }));
    const l2 = trainRun(adam({ weightDecay: 0.1, decayMode: "l2" }));
    expect([decoupled.weightNorm.toFixed(2), three(decoupled.heldOut)]).toEqual(["59.45", "2.061"]);
    expect([l2.weightNorm.toFixed(2), three(l2.heldOut)]).toEqual(["1.19", "3.274"]);
    // The checkpoint question rounds these to "near 3.27" and "near 2.06".
    expect([l2.heldOut.toFixed(2), decoupled.heldOut.toFixed(2)]).toEqual(["3.27", "2.06"]);
    const settings = { rate: 0.03, weightDecay: 0.1, biasCorrection: true } as const;
    expect(formatSmall(idleShrink({ ...settings, decayMode: "decoupled" }))).toBe("0.0030");
    expect(formatSmall(idleShrink({ ...settings, decayMode: "l2" }))).toBe("0.030");
    expect(`${((0.03 * 0.1) * 100).toFixed(1)}%`).toBe("0.3%");
  });

  it("matches the size each row keeps", () => {
    const reference = trainRun(adam());
    const kept = (spec: RunSpec, label: string) => rowsKept(trainRun(spec), reference).find((entry) => entry.label === label)!.kept;
    const l2 = adam({ weightDecay: 0.1, decayMode: "l2" });
    const decoupled = adam({ weightDecay: 0.1 });
    expect(`${(kept(l2, "q") * 100).toFixed(2)}%`).toBe("0.12%");
    expect(`${(kept(l2, "space") * 100).toFixed(1)}%`).toBe("3.6%");
    expect([Math.round(kept(decoupled, "q") * 100), Math.round(kept(decoupled, "space") * 100)]).toEqual([60, 68]);
    const small = adam({ weightDecay: 0.001, decayMode: "l2" });
    expect([Math.round(kept(small, "space") * 100), Math.round(kept(small, "q") * 100)]).toEqual([39, 14]);
    const smallDecoupled = adam({ weightDecay: 0.001 });
    for (const label of ["space", "k", "q"]) expect(Math.round(kept(smallDecoupled, label) * 100)).toBe(99);
  });

  it("keeps decoupled decay between 2.031 and 2.061 from off to 0.1, and L2 good only at 0.0001", () => {
    const decoupled = DECAY_LADDER.filter((decay) => decay <= 0.1).map((decay) => trainRun(adam({ weightDecay: decay })).heldOut);
    expect([three(Math.min(...decoupled)), three(Math.max(...decoupled))]).toEqual(["2.031", "2.061"]);
    expect(three(trainRun(adam({ weightDecay: 0.03 })).heldOut)).toBe("2.031");
    const none = trainRun(adam()).heldOut;
    const l2 = (decay: number) => trainRun(adam({ weightDecay: decay, decayMode: "l2" })).heldOut;
    expect(three(l2(0.0001))).toBe("2.035");
    expect(l2(0.0001)).toBeLessThan(none);
    expect(three(l2(0.0003))).toBe("2.074");
    for (const decay of DECAY_LADDER.filter((value) => value >= 0.0003)) expect(l2(decay)).toBeGreaterThan(none);
    for (const decay of DECAY_LADDER.filter((value) => value > 0)) {
      expect(formatSmall(idleShrink({ rate: 0.03, weightDecay: decay, decayMode: "l2", biasCorrection: true }))).toBe("0.030");
    }
  });

  it("ties the two best settings over five seeds", () => {
    const l2 = seeds(adam({ weightDecay: 0.0001, decayMode: "l2" }));
    const common = seeds(adam({ weightDecay: 0.1 }));
    const best = seeds(adam({ weightDecay: 0.03 }));
    expect([three(Math.min(...l2)), three(Math.max(...l2))]).toEqual(["2.034", "2.039"]);
    expect([three(Math.min(...common)), three(Math.max(...common))]).toEqual(["2.061", "2.065"]);
    expect([three(Math.min(...best)), three(Math.max(...best))]).toEqual(["2.031", "2.036"]);
    // L2 at 0.0001 beats decoupled 0.1 on every seed; decoupled 0.03 overlaps L2's range.
    expect(Math.max(...l2)).toBeLessThan(Math.min(...common));
    expect(Math.max(...best)).toBeGreaterThan(Math.min(...l2));
  });
});

describe("Watch Newton–Schulz work", () => {
  it("matches the Step 300 spectrum and the counts per iteration", () => {
    const zero = muonSpectra(0.05, "middle", 0)!;
    expect([zero.before.nonzero, zero.before.zeros]).toEqual([26, 4]);
    expect(three(zero.before.largest)).toBe("0.543");
    expect(formatSmall(zero.before.smallestNonzero)).toBe("2.1 × 10⁻⁶");
    expect(zero.before.values.filter((value) => value > ZERO_SINGULAR && value < 0.07)).toHaveLength(14);
    expect([1, 2, 3, 5, 8].map((steps) => muonSpectra(0.05, "middle", steps)!.after.inBand)).toEqual([9, 15, 19, 22, 25]);
    const five = muonSpectra(0.05, "middle", 5)!;
    const inBand = five.after.values.filter((value) => value >= BAND_LOW && value <= BAND_HIGH);
    expect([Math.min(...inBand).toFixed(2), Math.max(...inBand).toFixed(2)]).toEqual(["0.68", "1.20"]);
    expect(three(five.after.smallestNonzero)).toBe("0.001");
    expect(five.after.zeros).toBe(4);
    // Each pass multiplies a tiny value by about 3.4.
    expect((five.after.smallestNonzero / zero.before.smallestNonzero) ** (1 / 5)).toBeCloseTo(3.4445, 2);
  });

  it("matches Step 1: nine directions in, nine near 1 out, 21 zeros", () => {
    const zero = muonSpectra(0.05, "first", 0)!;
    const nonzero = zero.before.values.filter((value) => value > ZERO_SINGULAR);
    expect(nonzero).toHaveLength(9);
    expect([three(Math.min(...nonzero)), three(Math.max(...nonzero))]).toEqual(["0.211", "0.537"]);
    const five = muonSpectra(0.05, "first", 5)!;
    expect([three(five.after.smallestNonzero), three(five.after.largest), five.after.zeros]).toEqual(["0.682", "1.121", 21]);
  });

  it("never passes about 1.20, and reads every exact zero at about 1e-8 or less", () => {
    for (const capture of ["first", "middle"] as const) {
      for (let steps = 0; steps <= 8; steps += 1) {
        const spectra = muonSpectra(0.05, capture, steps)!;
        expect(spectra.after.largest).toBeLessThan(1.2027);
        // Twenty times below the 1e-7 threshold; the weakest real direction is about 2e-6.
        for (const value of spectra.after.values.filter((entry) => entry <= ZERO_SINGULAR)) expect(value).toBeLessThan(2e-8);
      }
    }
  });

  it("traces the four zero directions to the rows that never start a pair", () => {
    const run = trainRun(baseSpec("sgd", 1));
    const never = Array.from(TINY_VOCAB).filter((_, index) => run.rowVisits[index] === 0);
    expect(never).toEqual(["j", "x", "<", ">"]);
  });
});

describe("Tune each", () => {
  const table = runSweepNow();
  const of = (id: OptimizerId) => table.find((entry) => entry.optimizer === id)!;

  it("matches each optimizer's best rate against its default", () => {
    expect([of("sgd").best.rate, three(of("sgd").best.heldOut), three(of("sgd").atDefault.heldOut)]).toEqual([3, "2.041", "2.116"]);
    expect([of("momentum").best.rate, three(of("momentum").best.heldOut), three(of("momentum").atDefault.heldOut)]).toEqual([
      0.3,
      "2.035",
      "2.115",
    ]);
    expect([of("adamw").best.rate, three(of("adamw").best.heldOut)]).toEqual([0.03, "2.041"]);
    expect([of("muon").best.rate, three(of("muon").best.heldOut)]).toEqual([0.05, "2.059"]);
    expect(table.flatMap((entry) => entry.points)).toHaveLength(20);
  });

  it("matches the tuned gaps", () => {
    expect(Math.abs(of("sgd").best.heldOut - of("adamw").best.heldOut).toFixed(4)).toBe("0.0003");
    expect(three(of("adamw").best.heldOut - of("momentum").best.heldOut)).toBe("0.006");
    expect(three(of("muon").best.heldOut - of("adamw").best.heldOut)).toBe("0.018");
  });

  it("is U-shaped in the rate for every optimizer", () => {
    for (const entry of table) {
      const held = entry.points.map((point) => point.heldOut);
      const bottom = held.indexOf(Math.min(...held));
      expect(bottom).toBeGreaterThan(0);
      expect(bottom).toBeLessThan(held.length - 1);
      for (let index = 1; index <= bottom; index += 1) expect(held[index]).toBeLessThan(held[index - 1]);
      for (let index = bottom + 1; index < held.length; index += 1) expect(held[index]).toBeGreaterThan(held[index - 1]);
    }
  });

  it("matches the five-seed spreads and means at the tuned rates", () => {
    const sgd = seeds(baseSpec("sgd", 3));
    const momentum = seeds(baseSpec("momentum", 0.3));
    const adamw = seeds(baseSpec("adamw", 0.03));
    const muon = seeds(baseSpec("muon", 0.05));
    const spread = (values: number[]) => Math.max(...values) - Math.min(...values);
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    const top = [spread(sgd), spread(momentum), spread(adamw)];
    expect([three(Math.min(...top)), three(Math.max(...top))]).toEqual(["0.006", "0.011"]);
    expect(three(spread(muon))).toBe("0.015");
    expect([three(mean(momentum)), three(mean(sgd)), three(mean(adamw))]).toEqual(["2.036", "2.039", "2.044"]);
    // All five Muon seeds finish behind all five AdamW seeds.
    expect(Math.min(...muon)).toBeGreaterThan(Math.max(...adamw));
  });
});

describe("Optimizer state", () => {
  const rows = stateRows({
    sgd: run("sgd").stateFloats,
    momentum: run("momentum").stateFloats,
    adamw: run("adamw").stateFloats,
    muon: run("muon").stateFloats,
  });
  const bytes = teachingMemory.formatBytes;

  it("matches the table and the two metrics", () => {
    expect(rows.map((entry) => bytes(entry.labBytes))).toEqual(["0 B", "3.6 KB", "7.2 KB", "3.6 KB"]);
    expect(rows.map((entry) => bytes(entry.largeBytes))).toEqual(["0 B", "30 GB", "60 GB", "30 GB"]);
    expect(bytes(rows[2].largeBytes - rows[0].largeBytes)).toBe("60 GB");
    expect(LARGE_PARAMETERS).toBe(7.5e9);
    expect(bytes(teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "mixed", "adam").total)).toBe("120 GB");
    expect(bytes(teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "mixed", "momentum").total)).toBe("90 GB");
  });

  it("matches the bytes per weight", () => {
    expect(teachingMemory.memoryBreakdown(1, "fp32", "adam").stateBytesPerParameter).toBe(8);
    expect(teachingMemory.memoryBreakdown(1, "fp32", "momentum").stateBytesPerParameter).toBe(4);
    const mixed = teachingMemory.memoryBreakdown(1, "mixed", "adam");
    expect([mixed.stateBytesPerParameter, mixed.totalBytesPerParameter]).toEqual([12, 16]);
    // The 8-bit optimizer paper's arithmetic for a 1-billion-parameter model: 4 and 8 bytes cut to 1 and 2.
    expect([bytes(1e9 * 4), bytes(1e9 * 8), bytes(1e9 * 1), bytes(1e9 * 2)]).toEqual(["4 GB", "8 GB", "1 GB", "2 GB"]);
    expect(0.00000001).toBe(1e-8);
  });
});
