import { describe, expect, it } from "vitest";
import { teachingServing } from "@app/module-sdk";
import { DEFAULT_RECIPE, runPipeline, type ShipRecipe } from "./pipeline";

const { GIB } = teachingServing;
const run = (patch: Partial<ShipRecipe> = {}) => runPipeline({ ...DEFAULT_RECIPE, ...patch });
const round = (value: number, digits = 3) => Number(value.toFixed(digits));
const losses = (release: ReturnType<typeof run>) => ({
  target: release.stages.map((stage) => round(stage.metrics.targetLoss)),
  control: release.stages.map((stage) => round(stage.metrics.controlLoss)),
});
const overlap = (a: { low: number; high: number }, b: { low: number; high: number }) => a.low <= b.high && b.low <= a.high;

describe("figures quoted in the lessons at the starting recipe", () => {
  const release = run();
  const [base, adapted, aligned, shipped] = release.stages;

  it("reads the four stages' held-out losses", () => {
    // standard.mdx and plain.mdx "How to play with it": 2.540 to 2.461 on recipe text, 2.098 to 2.248 on harbor text.
    expect(losses(release)).toEqual({
      target: [2.54, 2.461, 2.496, 2.497],
      control: [2.098, 2.248, 2.29, 2.292],
    });
    expect(adapted.metrics.controlLoss).toBeGreaterThan(base.metrics.controlLoss);
  });

  it("fails the harbor check by 0.194 nats against an allowance of 0.100 and passes the recipe check", () => {
    // "a gain of 0.043 against a bar of 0.030", "0.194 nats worse than the base against an allowance of 0.100".
    expect(round(base.metrics.targetLoss - shipped.metrics.targetLoss)).toBe(0.043);
    expect(round(shipped.metrics.controlLoss - base.metrics.controlLoss)).toBe(0.194);
    expect(DEFAULT_RECIPE.gateTarget).toBe(0.03);
    expect(DEFAULT_RECIPE.gateControl).toBe(0.1);
    expect(release.verdict).toBe("hold");
  });

  it("lists forgetting at the adapting stage and the alignment tax at the aligning stage", () => {
    expect(round(adapted.metrics.controlLoss - base.metrics.controlLoss)).toBe(0.15);
    expect(round(aligned.metrics.targetLoss - adapted.metrics.targetLoss)).toBe(0.035);
    expect(round(aligned.metrics.controlLoss - adapted.metrics.controlLoss)).toBe(0.042);
  });
});

describe("figures quoted with four replayed harbor sentences", () => {
  const release = run({ replay: 4 });
  const [base, adapted, aligned, shipped] = release.stages;

  it("ends harbor at 2.086 and recipe at 2.367, which ships", () => {
    // "four leaves harbor loss at 2.086, slightly better than the base, with recipe loss at 2.367".
    expect(round(shipped.metrics.controlLoss)).toBe(2.086);
    expect(round(shipped.metrics.targetLoss)).toBe(2.367);
    expect(shipped.metrics.controlLoss).toBeLessThan(base.metrics.controlLoss);
    expect(release.verdict).toBe("ship");
  });

  it("is 0.173 better on recipe text and no worse on harbor text", () => {
    expect(round(base.metrics.targetLoss - shipped.metrics.targetLoss)).toBe(0.173);
    expect(round(shipped.metrics.controlLoss - base.metrics.controlLoss)).toBe(-0.012);
  });

  it("prices alignment at 0.030 on recipe text and 0.040 on harbor text, with all three pairs ordered", () => {
    expect(round(aligned.metrics.targetLoss - adapted.metrics.targetLoss)).toBe(0.03);
    expect(round(aligned.metrics.controlLoss - adapted.metrics.controlLoss)).toBe(0.04);
    expect(release.alignment?.accuracy).toBe(1);
    expect(release.regressions).toEqual([
      "after aligning: recipe loss rose 0.030 nats",
      "after aligning: harbor loss rose 0.040 nats",
    ]);
  });

  it("reads the recipe benchmark at 1 of 6 before and after, with overlapping intervals", () => {
    expect(base.metrics.target.correct).toBe(1);
    expect(shipped.metrics.target.correct).toBe(1);
    expect(overlap(base.metrics.target, shipped.metrics.target)).toBe(true);
  });
});

describe("replay and adapter epochs, as the card and lessons describe", () => {
  it("passes with one replayed sentence", () => {
    // "One sentence is enough to pass".
    const release = run({ replay: 1 });
    expect(release.verdict).toBe("ship");
    expect(round(release.stages[3].metrics.controlLoss - release.stages[0].metrics.controlLoss)).toBe(0.057);
  });

  it("raises harbor loss at every adapter-epoch setting without replay, and recipe loss ends worse than the base", () => {
    // card-info "Training recipe" controls.
    const rises = [8, 16, 24, 32, 40].map((loraEpochs) => {
      const release = run({ loraEpochs });
      return round(release.stages[3].metrics.controlLoss - release.stages[0].metrics.controlLoss);
    });
    expect(rises).toEqual([0.04, 0.189, 0.194, 0.215, 0.284]);
    expect(run({ loraEpochs: 40 }).stages[3].metrics.targetLoss).toBeGreaterThan(run({ loraEpochs: 40 }).stages[0].metrics.targetLoss);
  });

  it("costs harbor loss more than it gains recipe loss at 16 epochs or more without replay", () => {
    // card-info "Stage by stage" notice.
    for (const loraEpochs of [16, 24, 32, 40]) {
      const [base, adapted] = run({ loraEpochs }).stages;
      expect(adapted.metrics.controlLoss - base.metrics.controlLoss, `${loraEpochs} epochs`).toBeGreaterThan(
        base.metrics.targetLoss - adapted.metrics.targetLoss,
      );
    }
  });

  it("keeps harbor loss within 0.02 nats of the base and recipe loss falling at every epoch setting with four replayed sentences", () => {
    for (const loraEpochs of [8, 16, 24, 32, 40]) {
      const release = run({ replay: 4, loraEpochs });
      const [base, , , shipped] = release.stages;
      expect(shipped.metrics.controlLoss - base.metrics.controlLoss, `${loraEpochs} epochs`).toBeLessThanOrEqual(0.02);
      expect(base.metrics.targetLoss - shipped.metrics.targetLoss, `${loraEpochs} epochs`).toBeGreaterThan(0.03);
    }
  });

  it("barely moves the model at rank 1", () => {
    const [base, , , shipped] = run({ replay: 4, loraRank: 1 }).stages;
    expect(round(base.metrics.targetLoss - shipped.metrics.targetLoss)).toBe(-0.004);
  });
});

describe("compression figures", () => {
  it("moves neither loss by 0.02 nats between F16 and Q4_0", () => {
    const f16 = run({ replay: 4, format: "F16" }).stages[3].metrics;
    const q4 = run({ replay: 4, format: "Q4_0" }).stages[3].metrics;
    expect(Math.abs(q4.targetLoss - f16.targetLoss)).toBeLessThan(0.02);
    expect(Math.abs(q4.controlLoss - f16.controlLoss)).toBeLessThan(0.02);
  });

  it("raises recipe loss by 0.022 at 50% pruning and 0.180 at 90%, against After aligning", () => {
    const rise = (prune: number) => {
      const [, , aligned, shipped] = run({ replay: 4, prune }).stages;
      return round(shipped.metrics.targetLoss - aligned.metrics.targetLoss);
    };
    expect(rise(0.5)).toBe(0.022);
    expect(rise(0.9)).toBe(0.18);
  });

  it("returns recipe loss to 2.530 at 90% pruning, nearly the base model's 2.540, and fails both checks", () => {
    const release = run({ replay: 4, prune: 0.9 });
    expect(round(release.stages[3].metrics.targetLoss)).toBe(2.53);
    expect(round(release.stages[0].metrics.targetLoss)).toBe(2.54);
    expect(release.gate.checks.map((check) => check.passed)).toEqual([false, false]);
    // checkpoint explanation: "the recipe gain nearly disappears".
    expect(release.stages[0].metrics.targetLoss - release.stages[3].metrics.targetLoss).toBeLessThan(0.03);
  });

  it("makes harbor loss grow faster than pruning does, so the harm is not proportional", () => {
    const harm = (prune: number) => {
      const [, , aligned, shipped] = run({ replay: 4, prune }).stages;
      return shipped.metrics.controlLoss - aligned.metrics.controlLoss;
    };
    expect(harm(0.9)).toBeGreaterThan(5 * harm(0.5));
  });
});

describe("serving figures", () => {
  it("fits the starting load in 9.58 GiB with 36 streams and 43.4 tokens per second each", () => {
    const serving = run().shipping.serving;
    expect(round(serving.totalBytes / GIB, 2)).toBe(9.58);
    expect(serving.maxStreams).toBe(36);
    expect(round(serving.perUserTokensPerSecond, 1)).toBe(43.4);
  });

  it("needs 261.58 GiB at Q4_K_M and 271.96 GiB at F16 for 64 users at 32,768 tokens", () => {
    const q4 = run({ users: 64, context: 32768 }).shipping.serving;
    const f16 = run({ users: 64, context: 32768, format: "F16" }).shipping.serving;
    expect(round(q4.totalBytes / GIB, 2)).toBe(261.58);
    expect(round(f16.totalBytes / GIB, 2)).toBe(271.96);
    // "the cache, not the weights, overflows": the cache is the larger share at both formats.
    expect(q4.kvBytes).toBeGreaterThan(10 * q4.weightBytes);
    expect(f16.kvBytes).toBeGreaterThan(10 * f16.weightBytes);
  });
});

describe("a loosened gate", () => {
  it("ships the starting recipe once the harbor allowance is at its maximum", () => {
    // standard.mdx "Where it breaks": "drag Allowed harbor-loss rise to its maximum and the defaults ship".
    expect(run().verdict).toBe("hold");
    expect(run({ gateControl: 0.4 }).verdict).toBe("ship");
  });
});
