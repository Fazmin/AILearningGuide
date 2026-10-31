import { describe, expect, it } from "vitest";
import { checkpointQuestions, trainTinyExtended, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import definition from "./module";
import { normalBand, quantile, stepPoints, summarizeRun, values } from "./dashboard";
import { auditHalfGradients } from "./fp16-audit";
import { runOptions, TRAINING_TEXT, type FaultSettings } from "./faults";
import { firstDifference } from "./rollback";
import { defaultState } from "./state";

/** Pins the factual claims in the checkpoint explanations. Settings are the lab's own. */
const run = (patch: Partial<FaultSettings>) => trainTinyExtended(runOptions({ ...defaultState, ...patch }));
const r3 = (value: number | null | undefined) => (value === null || value === undefined ? value : Number(value.toFixed(3)));
const healthy = run({});
const healthyPoints = stepPoints(healthy);

describe("training-stability checkpoint claims", () => {
  it("question 1: the healthy ratio band sits far above 10⁻³ while the run finishes well", () => {
    const band = normalBand(healthyPoints, "ratio")!;
    expect([band.low.toFixed(3), band.high.toFixed(3)]).toEqual(["0.021", "0.040"]);
    expect(r3(healthy.finalLoss)).toBe(2.039);
    const explosion = stepPoints(run({ fault: "rate" }));
    expect(r3(explosion[0].ratio)).toBe(0.31);
    expect(Math.round(explosion[0].ratio! / healthyPoints[0].ratio!)).toBe(15);
  });

  it("question 2: on step 1 only the update ratio differs, and the gradient norm leaves its band late", () => {
    const explosion = run({ fault: "rate" });
    const [a, b] = [explosion.trace[0], healthy.trace[0]];
    expect([a.loss, a.gradientNorm, a.maxLogit]).toEqual([b.loss, b.gradientNorm, b.maxLogit]);
    expect(a.updateRatio).toBeGreaterThan(10 * b.updateRatio);
    const points = stepPoints(explosion);
    const band = normalBand(healthyPoints, "norm")!;
    expect(points.find((point) => (point.norm ?? 0) > band.high)?.step).toBe(6);
    expect(points.find((point) => (point.norm ?? 0) > 10)?.step).toBe(36);
  });

  it("question 3: only the clip and the schedule together finish below the healthy run", () => {
    const scheduleOnly = run({ fault: "rate", schedule: "warmup-cosine" });
    expect(r3(scheduleOnly.finalLoss)).toBe(19.152);
    expect(Math.floor(Math.log10(summarizeRun(scheduleOnly).peakLogit!))).toBe(28);
    const clipOnly = run({ fault: "rate", clipNorm: 0.5 });
    expect(r3(clipOnly.finalLoss)).toBe(3.172);
    expect(clipOnly.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY);
    const wideClip = run({ fault: "rate", clipNorm: 1 });
    expect(r3(wideClip.finalLoss)).toBe(4.05);
    expect(wideClip.finalLoss).toBeGreaterThan(UNIFORM_CROSS_ENTROPY);
    const both = run({ fault: "rate", clipNorm: 0.5, schedule: "warmup-cosine" });
    expect(r3(both.finalLoss)).toBe(1.904);
    for (const other of [scheduleOnly, clipOnly, wideClip]) expect(both.finalLoss).toBeLessThan(other.finalLoss);
    expect(both.finalLoss).toBeLessThan(healthy.finalLoss);
  });

  it("question 4: dynamic scaling skips the overflowing steps instead of halting", () => {
    const fixed = run({ fault: "fp16" });
    expect(fixed.haltedAt).toBe(33);
    expect(fixed.trace[32].overflow).toBe(true);
    expect(fixed.trace[32].skipped).toBe(false);
    const dynamic = run({ fault: "fp16", scaling: "dynamic" });
    const summary = summarizeRun(dynamic);
    expect(summary.overflowSkips).toEqual([33, 62, 184, 430]);
    expect(summary.skipped).toBe(summary.overflow);
    expect(Math.log2(summary.endScale)).toBe(17);
    expect([r3(dynamic.finalLoss), r3(healthy.finalLoss)]).toEqual([2.04, 2.039]);
    // "Almost level": four skipped steps cost about a thousandth of a nat.
    expect(Math.abs(dynamic.finalLoss - healthy.finalLoss)).toBeLessThan(0.002);
  });

  it("question 5: no gradient entry flushes to zero, even at a scale of 1", () => {
    const recipe = { text: TRAINING_TEXT, rank: 8, epochs: 20, batchSize: 16, learningRate: 0.5, seed: 9 };
    const unscaled = auditHalfGradients({ ...recipe, scaleExponent: 0 });
    expect([unscaled.nonzero, unscaled.flushed, unscaled.subnormal]).toEqual([153008, 0, 949]);
    const plain = auditHalfGradients({ ...recipe, scaleExponent: null });
    expect(Number(plain.smallest.toPrecision(2))).toBe(3.6e-8);
    expect(Number((2 ** -25).toPrecision(2))).toBe(3e-8);
  });

  it("question 6: the replay matches the original exactly up to the skipped steps", () => {
    const original = run({ fault: "bad-batch" });
    const replay = run({ fault: "bad-batch", rollback: true });
    expect(firstDifference(original.trace, replay.trace)).toBe(400);
    expect(replay.trace.filter((point) => point.skipped).map((point) => point.step + 1)).toEqual([400, 401, 402, 403, 404]);
    const clean = trainTinyExtended({ ...runOptions({ ...defaultState, fault: "bad-batch" }), faults: [] });
    expect([r3(replay.finalLoss), r3(original.finalLoss), r3(clean.finalLoss)]).toEqual([1.993, 2.022, 1.991]);
  });

  it("question 7: the lab's spike has a known cause, the garbage batches", () => {
    const original = run({ fault: "bad-batch" });
    const garbage = original.trace.filter((point) => point.faulty).map((point) => point.step + 1);
    expect(garbage).toEqual([400, 401, 402, 403, 404]);
    // Without the garbage the same run has no spike at all.
    const clean = trainTinyExtended({ ...runOptions({ ...defaultState, fault: "bad-batch" }), faults: [] });
    const before = clean.checkpoints.find((point) => point.step === 396)!.loss;
    const after = Math.max(...clean.checkpoints.filter((point) => point.step >= 400).map((point) => point.loss));
    expect(after - before).toBeLessThan(0.1);
  });

  it("question 8: z-loss shrinks the log-partition but only a lower rate rescues the run", () => {
    const logZ = (patch: Partial<FaultSettings>) =>
      quantile(
        stepPoints(run({ fault: "logits", ...patch }))
          .map((point) => point.logPartition)
          .filter((value): value is number => value !== null),
        0.5,
      )!;
    expect(logZ({}).toFixed(1)).toBe("19.7");
    expect(logZ({ zLoss: 0.01 }).toFixed(1)).toBe("2.2");
    const z = run({ fault: "logits", zLoss: 0.01 });
    expect(r3(z.finalLoss)).toBe(3.277);
    expect(z.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY);
    expect(Math.round(summarizeRun(run({ fault: "logits" })).peakLogit!)).toBe(395);
    expect(Math.round(summarizeRun(z).peakLogit!)).toBe(179);
    expect(r3(run({ fault: "logits", zLoss: 0.01, adamRate: 0.1 }).finalLoss)).toBe(1.985);
    // A z-loss of 0.1, past the control's top, is worse than none at a rate of 1.
    const heavy = trainTinyExtended({ ...runOptions({ ...defaultState, fault: "logits" }), zLoss: 0.1 });
    expect([r3(heavy.finalLoss), r3(run({ fault: "logits" }).finalLoss)]).toEqual([13.418, 11.515]);
    // On its first step Adam moves each weight by about the learning rate.
    const first = run({ fault: "logits" }).trace[0];
    expect(first.maxUpdate).toBeGreaterThan(0.99);
    expect(first.maxUpdate).toBeLessThanOrEqual(1 + 1e-6);
  });

  it("does not hand the answer to the longest option", () => {
    const questions = checkpointQuestions(definition);
    const longest = questions.filter((question) => {
      const lengths = question.options.map((option) => option.length);
      const best = Math.max(...lengths);
      return lengths[question.answer] === best && lengths.filter((length) => length === best).length === 1;
    }).length;
    expect(longest).toBeLessThanOrEqual(Math.floor(questions.length / 2));
    expect(values(healthyPoints, "ratio").length).toBe(480);
  });
});
