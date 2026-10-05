import { describe, expect, it } from "vitest";
import {
  CONTROL_BENCHMARK,
  DEFAULT_RECIPE,
  FORMAT_NAMES,
  PREFERENCE_PAIRS,
  TARGET_BENCHMARK,
  TRAIN_TEXT,
  evaluateGate,
  measure,
  runPipeline,
  type ShipRecipe,
} from "./pipeline";

const run = (patch: Partial<ShipRecipe> = {}) => runPipeline({ ...DEFAULT_RECIPE, ...patch });

describe("benchmark hygiene", () => {
  it("has twelve items and none reuses a phrase from any training text", () => {
    expect(TARGET_BENCHMARK).toHaveLength(6);
    expect(CONTROL_BENCHMARK).toHaveLength(6);
    const train = `${TRAIN_TEXT.harbor} ${TRAIN_TEXT.recipes}`.replace(/\s+/g, " ");
    for (const item of [...TARGET_BENCHMARK, ...CONTROL_BENCHMARK]) {
      expect(train.includes(`${item.context} ${item.correct}`), `${item.context} ${item.correct}`).toBe(false);
    }
  });

  it("never trains on a preference pair's chosen text as a benchmark answer", () => {
    for (const pair of PREFERENCE_PAIRS) {
      for (const item of [...TARGET_BENCHMARK, ...CONTROL_BENCHMARK]) {
        expect(`${pair.prompt}${pair.chosen}`.includes(`${item.context} ${item.correct}`)).toBe(false);
      }
    }
  });
});

describe("runPipeline", () => {
  it("is deterministic in the recipe", () => {
    const first = run({ replay: 3 });
    const second = run({ replay: 3 });
    expect(second.stages.map((stage) => stage.metrics)).toEqual(first.stages.map((stage) => stage.metrics));
    expect(second.verdict).toBe(first.verdict);
  });

  it("returns four stages in order, each with held-out metrics", () => {
    const release = run();
    expect(release.stages.map((stage) => stage.id)).toEqual(["base", "adapted", "aligned", "shipped"]);
    for (const stage of release.stages) {
      expect(stage.weights).toHaveLength(900);
      expect(Number.isFinite(stage.metrics.targetLoss)).toBe(true);
      expect(Number.isFinite(stage.metrics.controlLoss)).toBe(true);
    }
  });

  it("changes nothing when there is no adapter training and no preference training", () => {
    const release = run({ loraEpochs: 0, dpoSteps: 0, format: "F16" });
    const [base, adapted, aligned] = release.stages;
    expect(adapted.metrics.targetLoss).toBeCloseTo(base.metrics.targetLoss, 9);
    expect(aligned.metrics.controlLoss).toBeCloseTo(adapted.metrics.controlLoss, 9);
    expect(release.alignment).toBeNull();
  });

  it("measures the gate on the artifact it would ship, not the one it trained", () => {
    const release = run({ replay: 4, prune: 0.9 });
    const [, , aligned, shipped] = release.stages;
    expect(shipped.metrics.targetLoss).toBeGreaterThan(aligned.metrics.targetLoss + 0.1);
    expect(release.gate.checks.map((check) => check.passed)).toEqual([false, false]);
  });

  it("holds the default recipe because the harbor check fails, and says so", () => {
    const release = run();
    expect(release.verdict).toBe("hold");
    expect(release.gate.checks.map((check) => [check.id, check.passed])).toEqual([
      ["target", true],
      ["control", false],
    ]);
    expect(release.reasons).toHaveLength(1);
    expect(release.reasons[0]).toContain("Harbor text");
  });

  it("ships once any harbor text is replayed into the adaptation data", () => {
    for (const replay of [1, 2, 3, 4, 5, 6]) expect(run({ replay }).verdict, `replay ${replay}`).toBe("ship");
  });

  it("fails the recipe check when the adapter barely moves the model", () => {
    const release = run({ replay: 4, loraRank: 1 });
    expect(release.gate.checks.find((check) => check.id === "target")?.passed).toBe(false);
    expect(release.gate.checks.find((check) => check.id === "control")?.passed).toBe(true);
  });

  it("holds a passing recipe that does not fit the memory budget", () => {
    const release = run({ replay: 4, format: "F16", users: 64, context: 32768 });
    expect(release.gate.passed).toBe(true);
    expect(release.shipping.serving.fits).toBe(false);
    expect(release.verdict).toBe("hold");
    expect(release.reasons.some((reason) => reason.includes("past the 24 GiB accelerator"))).toBe(true);
  });

  it("lets a loosened gate pass a model it should not, which is the point of fixing bars in advance", () => {
    expect(run({ gateControl: 0.4 }).gate.passed).toBe(true);
    expect(run({ gateControl: 0.4 }).stages[3].metrics.controlLoss).toBeGreaterThan(
      run().stages[0].metrics.controlLoss + 0.15,
    );
  });

  it("lists a regression when a stage makes either text worse by more than 0.02 nats", () => {
    const release = run();
    expect(release.regressions.some((entry) => entry.startsWith("after adapting: harbor loss rose"))).toBe(true);
    expect(run({ replay: 4, dpoSteps: 0, format: "F16" }).regressions).toEqual([]);
  });

  it("offers every format the serving lab has a file size for", () => {
    expect(FORMAT_NAMES).toEqual(["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]);
    for (const format of FORMAT_NAMES) expect(run({ replay: 4, format }).shipping.format).toBe(format);
  });

  it("falls back to a known format for an unknown name rather than throwing", () => {
    expect(run({ format: "nope" }).shipping.format).toBe("Q4_K_M");
  });
});

describe("evaluateGate", () => {
  it("passes only when both checks do", () => {
    const base = measure(run().stages[0].weights);
    const better = { ...base, targetLoss: base.targetLoss - 0.1 };
    expect(evaluateGate(base, better, DEFAULT_RECIPE).passed).toBe(true);
    expect(evaluateGate(base, { ...better, controlLoss: base.controlLoss + 0.2 }, DEFAULT_RECIPE).passed).toBe(false);
    expect(evaluateGate(base, { ...base, targetLoss: base.targetLoss - 0.01 }, DEFAULT_RECIPE).passed).toBe(false);
  });
});
