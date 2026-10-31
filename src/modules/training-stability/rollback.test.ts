import { describe, expect, it } from "vitest";
import { trainTinyExtended } from "@app/module-sdk";
import { GARBAGE_COUNT, GARBAGE_FIRST_STEP, runOptions, type FaultSettings } from "./faults";
import { CHECKPOINT_EVERY, firstDifference, lastCheckpointBefore, rollbackPlan, spikeReport } from "./rollback";
import { defaultState } from "./state";

const settings = (patch: Partial<FaultSettings>): FaultSettings => ({ ...defaultState, ...patch });

describe("rollback plan", () => {
  it("restarts from the last checkpoint saved before the bad step", () => {
    expect(CHECKPOINT_EVERY).toBe(50);
    expect(lastCheckpointBefore(400)).toBe(350);
    expect(lastCheckpointBefore(401)).toBe(400);
    expect(lastCheckpointBefore(351)).toBe(350);
    expect(lastCheckpointBefore(1)).toBe(0);
    expect(lastCheckpointBefore(-7)).toBe(0);
    expect(lastCheckpointBefore(10, 0)).toBe(9);
  });

  it("replays from the checkpoint and skips exactly the garbage steps", () => {
    const plan = rollbackPlan(GARBAGE_FIRST_STEP, GARBAGE_COUNT);
    expect(plan).toEqual({ checkpointStep: 350, replayFrom: 351, replayTo: 399, skipFrom: 400, skipTo: 404, skipSteps: [[399, 404]] });
  });
});

describe("the replay", () => {
  const original = trainTinyExtended(runOptions(settings({ fault: "bad-batch" })));
  const replay = trainTinyExtended(runOptions(settings({ fault: "bad-batch", rollback: true })));

  it("is bit-identical to the original run at every step before the first skipped one", () => {
    expect(firstDifference(original.trace, replay.trace)).toBe(400);
    expect(replay.trace.slice(0, 399)).toEqual(original.trace.slice(0, 399));
    // Every whole-corpus checkpoint before the garbage matches too, the last at step 396.
    const before = (run: typeof original) => run.checkpoints.filter((point) => point.step < 400).map((point) => [point.step, point.loss]);
    expect(before(replay)).toEqual(before(original));
    expect(before(replay).at(-1)?.[0]).toBe(396);
  });

  it("reads the garbage batches but never applies them", () => {
    const skipped = replay.trace.filter((point) => point.skipped).map((point) => point.step + 1);
    expect(skipped).toEqual([400, 401, 402, 403, 404]);
    for (const point of replay.trace.slice(399, 404)) {
      expect(point.faulty).toBe(true);
      expect(point.maxUpdate).toBe(0);
    }
    expect(replay.skippedSteps).toBe(5);
    expect(replay.steps).toBe(480);
  });

  it("holds with the clip on as well", () => {
    const clippedOriginal = trainTinyExtended(runOptions(settings({ fault: "bad-batch", clipNorm: 1 })));
    const clippedReplay = trainTinyExtended(runOptions(settings({ fault: "bad-batch", clipNorm: 1, rollback: true })));
    expect(firstDifference(clippedOriginal.trace, clippedReplay.trace)).toBe(400);
  });

  it("reports no difference for identical traces and the first step for different ones", () => {
    expect(firstDifference(original.trace, original.trace)).toBeNull();
    const other = trainTinyExtended(runOptions(settings({ fault: "logits" })));
    expect(firstDifference(original.trace, other.trace)).toBe(1);
  });
});

describe("spike report", () => {
  it("measures the rise from the last checkpoint before the bad step to the highest one after", () => {
    const checkpoints = [
      { step: 392, loss: 2.1 },
      { step: 396, loss: 2 },
      { step: 400, loss: 2.5 },
      { step: 404, loss: 3.25 },
      { step: 408, loss: 2.75 },
    ].map((point) => ({ ...point, progress: 0, weights: new Float32Array(0) }));
    expect(spikeReport(checkpoints, 400)).toEqual({ before: 2, peak: 3.25, peakStep: 404, rise: 1.25 });
  });

  it("ignores checkpoints that are not numbers and gives up when one side is empty", () => {
    const point = (step: number, loss: number) => ({ step, loss, progress: 0, weights: new Float32Array(0) });
    expect(spikeReport([point(396, 2), point(400, Number.NaN), point(404, 3)], 400)?.peak).toBe(3);
    expect(spikeReport([point(404, 3)], 400)).toBeNull();
    expect(spikeReport([point(396, 2)], 400)).toBeNull();
  });
});
