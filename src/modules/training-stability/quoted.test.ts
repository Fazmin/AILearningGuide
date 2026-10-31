import { describe, expect, it } from "vitest";
import {
  bigramPairs,
  extendedScheduleFactor,
  tinyCrossEntropy,
  trainTinyExtended,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  encodeTinyText,
  UNIFORM_CROSS_ENTROPY,
  type TinyExtendedOptions,
  type TinyExtendedRun,
} from "@app/module-sdk";
import { normalBand, quantile, stepPoints, summarizeRun, values } from "./dashboard";
import { auditHalfGradients } from "./fp16-audit";
import { garbageFault, halfWindow, HALF_MAX, healthyOptions, NUMBER_FORMATS, runOptions, TRAINING_TEXT, type FaultSettings } from "./faults";
import { rollbackPlan, spikeReport } from "./rollback";
import { defaultState } from "./state";

/** Pins every measured value the three lessons and card-info.ts quote. */
const cache = new Map<string, TinyExtendedRun>();
const train = (options: TinyExtendedOptions) => {
  const key = JSON.stringify(options);
  if (!cache.has(key)) cache.set(key, trainTinyExtended(options));
  return cache.get(key)!;
};
const run = (patch: Partial<FaultSettings>) => train(runOptions({ ...defaultState, ...patch }));
const r3 = (value: number | null | undefined) => (value === null || value === undefined ? value : Number(value.toFixed(3)));
const sig2 = (value: number) => Number(value.toPrecision(2));
const median = (list: number[]) => quantile(list, 0.5);
const logPartitions = (target: TinyExtendedRun) =>
  stepPoints(target)
    .map((point) => point.logPartition)
    .filter((value): value is number => value !== null);

describe("the model and its data", () => {
  it("matches the sizes the lessons give", () => {
    expect(encodeTinyText(TRAINING_TEXT)).toHaveLength(379);
    expect(bigramPairs(TRAINING_TEXT).inputs.length).toBe(378);
    expect(TINY_VOCAB_SIZE).toBe(30);
    expect(2 * 8 * 30).toBe(480);
    expect(run({}).steps).toBe(480);
    expect(r3(UNIFORM_CROSS_ENTROPY)).toBe(3.401);
    expect(rollbackPlan(400, 5).checkpointStep).toBe(350);
    // Every 4 steps: 121 whole-corpus checkpoints from step 0 to step 480.
    expect(run({}).checkpoints.map((point) => point.step).slice(0, 3)).toEqual([0, 4, 8]);
  });

  it("never lets j or x start a pair in the clean text, while q and z do", () => {
    const starts = new Set(Array.from(bigramPairs(TRAINING_TEXT).inputs).map((id) => TINY_VOCAB[id]));
    expect(starts.has("j")).toBe(false);
    expect(starts.has("x")).toBe(false);
    expect(starts.has("q")).toBe(true);
    expect(starts.has("z")).toBe(true);
  });
});

describe("Read a healthy dashboard", () => {
  const healthy = run({});
  const points = stepPoints(healthy);

  it("finishes at 2.039 with the quoted normal ranges", () => {
    expect(r3(healthy.finalLoss)).toBe(2.039);
    const loss = normalBand(points, "loss")!;
    expect([loss.low.toFixed(2), loss.high.toFixed(2)]).toEqual(["1.79", "3.38"]);
    const norm = normalBand(points, "norm")!;
    expect([r3(norm.low), r3(norm.high)]).toEqual([0.114, 0.7]);
    expect(r3(summarizeRun(healthy).peakNorm)).toBe(0.955);
    const logit = normalBand(points, "maxLogit")!;
    expect([r3(logit.low), logit.high.toFixed(2)]).toEqual([0.118, "6.23"]);
    expect(r3(summarizeRun(healthy).peakLogit)).toBe(6.707);
    const ratio = normalBand(points, "ratio")!;
    expect([ratio.low.toFixed(3), ratio.high.toFixed(3)]).toEqual(["0.021", "0.040"]);
    expect(median(values(points, "ratio"))!.toFixed(3)).toBe("0.026");
  });

  it("sits about 26 times, and twenty to forty times, above the 10⁻³ rule of thumb", () => {
    const ratio = normalBand(points, "ratio")!;
    expect(Math.round(median(values(points, "ratio"))! / 1e-3)).toBe(26);
    expect(Math.floor(ratio.low / 1e-3)).toBe(20);
    expect(Math.round(ratio.high / 1e-3)).toBe(40);
  });
});

describe("Push the learning rate", () => {
  const explosion = run({ fault: "rate" });
  const healthy = run({});
  const points = stepPoints(explosion);
  const healthyPoints = stepPoints(healthy);
  const band = normalBand(healthyPoints, "norm")!;
  const logitBand = normalBand(healthyPoints, "maxLogit")!;

  it("matches the healthy run on step 1 in everything but the step", () => {
    expect(8 / 0.5).toBe(16);
    expect(explosion.trace[0].loss).toBe(healthy.trace[0].loss);
    expect(explosion.trace[0].gradientNorm).toBe(healthy.trace[0].gradientNorm);
    expect(explosion.trace[0].maxLogit).toBe(healthy.trace[0].maxLogit);
    expect(r3(points[0].ratio)).toBe(0.31);
    expect(r3(healthyPoints[0].ratio)).toBe(0.021);
    expect(Math.round(points[0].ratio! / healthyPoints[0].ratio!)).toBe(15);
  });

  it("rises above the healthy bands on step 6, passes 10 at step 36 and halts at 154", () => {
    expect(points.find((point) => (point.norm ?? 0) > band.high)?.step).toBe(6);
    expect(points.find((point) => (point.maxLogit ?? 0) > logitBand.high)?.step).toBe(6);
    expect(points.find((point) => (point.norm ?? 0) > 10)?.step).toBe(36);
    expect(explosion.haltedAt).toBe(154);
    expect(sig2(points[152].norm!)).toBe(1.1e19);
    expect(sig2(points[152].maxLogit!)).toBe(3.1e38);
    // The largest 32-bit float is about 3.4 × 10³⁸.
    expect(sig2(NUMBER_FORMATS[0].largest)).toBe(3.4e38);
  });

  it("matches the clip and schedule results", () => {
    const clip = run({ fault: "rate", clipNorm: 0.5 });
    expect([r3(clip.finalLoss), clip.clippedSteps, clip.steps]).toEqual([3.172, 459, 480]);
    const both = run({ fault: "rate", clipNorm: 0.5, schedule: "warmup-cosine" });
    expect(r3(both.finalLoss)).toBe(1.904);
    expect(both.finalLoss).toBeLessThan(healthy.finalLoss);
    expect(r3(run({ fault: "rate", clipNorm: 1 }).finalLoss)).toBe(4.05);
    expect(run({ fault: "rate", clipNorm: 1 }).finalLoss).toBeGreaterThan(UNIFORM_CROSS_ENTROPY);
    expect(clip.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY);
  });

  it("stays finite but ruined with Warmup + cosine and no clip", () => {
    const ruined = run({ fault: "rate", schedule: "warmup-cosine" });
    const ruinedPoints = stepPoints(ruined);
    expect(ruined.haltedAt).toBeNull();
    expect(r3(ruined.finalLoss)).toBe(19.152);
    expect(sig2(summarizeRun(ruined).peakLogit!)).toBe(1.2e28);
    expect(sig2(summarizeRun(ruined).peakNorm!)).toBe(6.2e13);
    expect(Number(median(values(ruinedPoints.slice(-50), "ratio"))!.toPrecision(2))).toBe(0.0026);
  });

  it("uses the One cycle shape for Warmup + cosine", () => {
    expect(extendedScheduleFactor("one-cycle", 0, 0.15)).toBeCloseTo(0.1, 12);
    expect(extendedScheduleFactor("one-cycle", 0.15, 0.15)).toBeCloseTo(1, 12);
    expect(extendedScheduleFactor("one-cycle", 1, 0.15)).toBeCloseTo(0.02, 12);
  });
});

describe("Overflow in fp16", () => {
  it("overflows from 0.062 at a scale of 2²⁰ and halts on step 33", () => {
    expect(2 ** 20).toBe(1048576);
    expect(halfWindow(20).overflowFrom.toPrecision(2)).toBe("0.062");
    expect(halfWindow(20).flushBelow.toPrecision(2)).toBe("2.8e-14");
    const fixed = run({ fault: "fp16" });
    expect([summarizeRun(fixed).firstOverflow, fixed.haltedAt]).toEqual([33, 33]);
    expect(HALF_MAX).toBe(65504);
  });

  it("skips four steps under dynamic scaling and ends almost level with fp32", () => {
    const dynamic = run({ fault: "fp16", scaling: "dynamic" });
    const summary = summarizeRun(dynamic);
    expect(summary.overflowSkips).toEqual([33, 62, 184, 430]);
    expect(Math.log2(summary.endScale)).toBe(17);
    expect(r3(dynamic.finalLoss)).toBe(2.04);
    expect(r3(run({}).finalLoss)).toBe(2.039);
    // Three halvings settled the scale at 2¹⁷; it doubled after 200 clean steps and was halved again at step 430.
    const scaleAfter = (step: number) => Math.log2(dynamic.trace[step - 1].lossScale);
    expect([scaleAfter(33), scaleAfter(62), scaleAfter(184), scaleAfter(383), scaleAfter(384), scaleAfter(429), scaleAfter(430)]).toEqual([
      19, 18, 17, 17, 18, 18, 17,
    ]);
  });

  it("finishes at 2¹⁷ and halts at step 184 at 2¹⁸, where the window's top falls from about 0.5 to 0.25", () => {
    expect(run({ fault: "fp16", lossScale: 17 }).haltedAt).toBeNull();
    expect(run({ fault: "fp16", lossScale: 18 }).haltedAt).toBe(184);
    expect(halfWindow(17).overflowFrom.toFixed(2)).toBe("0.50");
    expect(halfWindow(18).overflowFrom.toFixed(2)).toBe("0.25");
    expect(run({ fault: "fp16", lossScale: 0 }).haltedAt).toBeNull();
    expect(run({ fault: "fp16", lossScale: 0 }).overflowSteps).toBe(0);
  });

  it("never flushes a gradient entry to zero, even with no scaling", () => {
    const recipe = { text: TRAINING_TEXT, rank: 8, epochs: 20, batchSize: 16, learningRate: 0.5, seed: 9 };
    const plain = auditHalfGradients({ ...recipe, scaleExponent: null });
    expect(plain.nonzero).toBe(153008);
    expect(sig2(plain.smallest)).toBe(3.6e-8);
    expect(sig2(2 ** -25)).toBe(3e-8);
    expect(plain.smallest).toBeGreaterThan(2 ** -25);
    for (const exponent of [0, 4, 8, 12, 17]) {
      const audit = auditHalfGradients({ ...recipe, scaleExponent: exponent });
      expect([audit.nonzero, audit.flushed], `2^${exponent}`).toEqual([153008, 0]);
    }
    expect(auditHalfGradients({ ...recipe, scaleExponent: 0 }).subnormal).toBe(949);
    // Runs that overflow halt with nothing flushed either.
    expect(auditHalfGradients({ ...recipe, scaleExponent: 20 }).flushed).toBe(0);
  });

  it("matches the format table on the Number formats card", () => {
    const byId = Object.fromEntries(NUMBER_FORMATS.map((format) => [format.id, format]));
    expect(sig2(byId.fp16.smallestNormal)).toBe(6.1e-5);
    expect(sig2(byId.fp32.smallestNormal)).toBe(1.2e-38);
    expect(sig2(byId.bf16.largest)).toBe(3.4e38);
    expect(2 ** 16).toBe(65536);
  });
});

describe("Survive a bad batch", () => {
  const clean = run({ fault: "none" });
  const adamClean = train({ ...runOptions({ ...defaultState, fault: "bad-batch" }), faults: [] });
  const faulty = run({ fault: "bad-batch" });
  const points = stepPoints(faulty);

  it("spikes the whole-corpus loss by 1.395 to 3.447 at step 404", () => {
    expect(r3(adamClean.finalLoss)).toBe(1.991);
    expect(r3(points[399].loss)).toBe(6.636);
    expect(r3(points[399].norm)).toBe(2.806);
    const band = normalBand(stepPoints(adamClean), "norm")!;
    expect([r3(band.low), r3(band.high)]).toEqual([0.483, 0.851]);
    const spike = spikeReport(faulty.checkpoints, 400)!;
    expect([r3(spike.before), r3(spike.peak), spike.peakStep, r3(spike.rise)]).toEqual([2.051, 3.447, 404, 1.395]);
    expect(spike.peak).toBeGreaterThan(UNIFORM_CROSS_ENTROPY);
    expect(r3(faulty.finalLoss)).toBe(2.022);
    // The batch loss jumps on step 400, the first garbage step.
    expect(points.findIndex((point) => (point.loss ?? 0) > 5) + 1).toBe(400);
  });

  it("roughly halves the spike with a clip of 1.00", () => {
    const clipped = run({ fault: "bad-batch", clipNorm: 1 });
    const spike = spikeReport(clipped.checkpoints, 400)!;
    expect([r3(spike.rise), spike.peakStep]).toEqual([0.734, 408]);
    const ratio = spike.rise / spikeReport(faulty.checkpoints, 400)!.rise;
    expect(ratio).toBeGreaterThan(0.45);
    expect(ratio).toBeLessThan(0.6);
  });

  it("recovers with Roll back and skip", () => {
    const plan = rollbackPlan(400, 5);
    expect([plan.checkpointStep, plan.replayFrom, plan.replayTo, plan.skipFrom, plan.skipTo]).toEqual([350, 351, 399, 400, 404]);
    expect(plan.replayTo - plan.replayFrom + 1).toBe(49);
    const replay = run({ fault: "bad-batch", rollback: true });
    expect(r3(replay.finalLoss)).toBe(1.993);
  });

  it("barely moves plain SGD, whose garbage gradient norm stays in the healthy band", () => {
    const sgd = train({ ...healthyOptions(), faults: [garbageFault()] });
    const spike = spikeReport(sgd.checkpoints, 400)!;
    expect(r3(spike.rise)).toBe(0.044);
    const sgdPoints = stepPoints(sgd);
    expect(r3(sgdPoints[399].norm)).toBe(0.608);
    const band = normalBand(stepPoints(clean), "norm")!;
    for (const point of sgdPoints.slice(399, 404)) {
      expect(point.norm!).toBeGreaterThanOrEqual(band.low);
      expect(point.norm!).toBeLessThanOrEqual(band.high);
    }
  });
});

describe("Tame logit growth", () => {
  const growth = run({ fault: "logits" });
  const points = stepPoints(growth);

  it("moves every weight by about the rate on step 1 and ends far above chance", () => {
    expect(r3(points[0].ratio)).toBe(0.989);
    expect(growth.checkpoints.find((point) => point.step > 0 && point.loss > UNIFORM_CROSS_ENTROPY)?.step).toBe(4);
    expect(r3(growth.finalLoss)).toBe(11.515);
    expect(growth.haltedAt).toBeNull();
    expect(Math.round(median(values(points, "maxLogit"))!)).toBe(66);
    expect(Math.round(summarizeRun(growth).peakLogit!)).toBe(395);
  });

  it("sits 63 of 378 pairs at the one-in-a-trillion floor, about 27.6 nats each", () => {
    expect(-Math.log(1e-12)).toBeCloseTo(27.6, 1);
    const V = TINY_VOCAB_SIZE;
    const probability = (row: number, target: number) => {
      let maximum = -Infinity;
      for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, growth.weights[row * V + k]);
      let total = 0;
      for (let k = 0; k < V; k += 1) total += Math.exp(growth.weights[row * V + k] - maximum);
      return Math.exp(growth.weights[row * V + target] - maximum) / total;
    };
    const { inputs, targets } = bigramPairs(TRAINING_TEXT);
    let floored = 0;
    let capped = 0;
    for (let index = 0; index < inputs.length; index += 1) {
      const p = probability(inputs[index], targets[index]);
      if (p < 1e-12) floored += 1;
      capped -= Math.log(Math.max(1e-12, p));
    }
    expect([floored, inputs.length]).toEqual([63, 378]);
    // The floor is what tinyCrossEntropy applies, so the quoted loss is the capped one.
    expect(capped / inputs.length).toBeCloseTo(tinyCrossEntropy(growth.weights, TRAINING_TEXT), 3);
  });

  it("pulls log Z from 19.7 to 2.2 with z-loss 0.01, yet ends at 3.277", () => {
    const z = run({ fault: "logits", zLoss: 0.01 });
    expect(median(logPartitions(growth))!.toFixed(1)).toBe("19.7");
    expect(median(logPartitions(z))!.toFixed(1)).toBe("2.2");
    expect(Math.round(summarizeRun(z).peakLogit!)).toBe(179);
    expect(r3(z.finalLoss)).toBe(3.277);
    expect(z.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY);
  });

  it("lowers log Z at every rate on the control, and ends healthy at 0.1", () => {
    for (const adamRate of [0.01, 0.03, 0.1, 0.3, 1]) {
      const without = median(logPartitions(run({ fault: "logits", adamRate })))!;
      const withZ = median(logPartitions(run({ fault: "logits", adamRate, zLoss: 0.01 })))!;
      expect(withZ, `rate ${adamRate}`).toBeLessThan(without);
    }
    const lowZ = run({ fault: "logits", adamRate: 0.1, zLoss: 0.01 });
    const low = run({ fault: "logits", adamRate: 0.1 });
    expect([r3(lowZ.finalLoss), r3(low.finalLoss)]).toEqual([1.985, 1.991]);
    expect([summarizeRun(low).peakLogit!.toFixed(1), summarizeRun(lowZ).peakLogit!.toFixed(1)]).toEqual(["12.2", "6.1"]);
  });

  it("is reined in, but not fixed, by AdamW's decay of 0.1", () => {
    const decay = train({ ...runOptions({ ...defaultState, fault: "logits" }), weightDecay: 0.1, optimizer: { kind: "adamw" } });
    expect(summarizeRun(decay).peakLogit!.toFixed(1)).toBe("17.8");
    expect(r3(decay.finalLoss)).toBe(3.136);
  });
});

describe("the worked examples", () => {
  it("does the arithmetic the lessons show", () => {
    // Update ratio: a step of length 0.05 on weights of length 2.
    expect(0.05 / 2).toBe(0.025);
    // Loss scaling at 2²⁰: a tiny push lands safely, a push of 0.1 overflows.
    expect(4e-8).toBeGreaterThan(2 ** -25);
    expect((4e-8 * 2 ** 20).toFixed(3)).toBe("0.042");
    expect(Math.round((0.1 * 2 ** 20) / 1000)).toBe(105);
    expect(0.1 * 2 ** 20).toBeGreaterThan(HALF_MAX);
    // z-loss: the penalty at log Z of 20 and of 2, with a coefficient of 0.01.
    expect(0.01 * 20 * 20).toBe(4);
    expect(Number((0.01 * 2 * 2).toFixed(4))).toBe(0.04);
    expect(4).toBeGreaterThan(run({}).finalLoss);
    // Clipping: pushes of 3 and 4 have length 5, and a cap of 1 divides both by 5.
    expect(Math.hypot(3, 4)).toBe(5);
    expect([3 / 5, 4 / 5]).toEqual([0.6, 0.8]);
    // 1.1 × 10¹⁹ is 11 followed by 18 zeros.
    expect(1.1e19).toBe(11 * 10 ** 18);
    // The z-loss term never enters the printed loss: z-loss changes the run, not the scale it is scored on.
    const z = run({ fault: "logits", adamRate: 0.1, zLoss: 0.01 });
    expect(z.finalLoss).toBe(tinyCrossEntropy(z.weights, TRAINING_TEXT));
  });
});

describe("the explosion takes about 150 steps", () => {
  it("halts after 153 finite steps", () => {
    expect(run({ fault: "rate" }).haltedAt! - 1).toBe(153);
  });
});
