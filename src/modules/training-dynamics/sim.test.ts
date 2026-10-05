import { describe, expect, it } from "vitest";
import {
  CHECKPOINTS,
  checkpointAt,
  features,
  gradientNoise,
  makeDataset,
  MAX_EPOCHS,
  NOISE_VARIANCE,
  OVERFIT_RATIO,
  predict,
  statusAt,
  train,
  TRAIN_SIZE,
  VALIDATION_SIZE,
} from "./sim";
import definition from "./module";

const BASE = { learningRate: 0.1, weightDecay: 0, batchSize: 16, seed: 1 };
const last = (run: ReturnType<typeof train>) => run.checkpoints[run.checkpoints.length - 1];

describe("training-dynamics simulation", () => {
  it("splits thirty-two seeded samples into sixteen and sixteen, reproducibly", () => {
    const first = makeDataset(1);
    const again = makeDataset(1);
    expect(first.train).toHaveLength(TRAIN_SIZE);
    expect(first.validation).toHaveLength(VALIDATION_SIZE);
    expect(again).toEqual(first);
    expect(makeDataset(2)).not.toEqual(first);
    for (let index = 1; index < first.train.length; index += 1) {
      expect(first.train[index].x).toBeGreaterThan(first.validation[index - 1].x);
    }
  });

  it("scales Legendre features so every degree has comparable size", () => {
    const atOne = features(1, 4);
    for (let k = 0; k <= 4; k += 1) expect(atOne[k]).toBeCloseTo(Math.sqrt(2 * k + 1), 12);
    expect(features(0, 2)[2]).toBeCloseTo(-0.5 * Math.sqrt(5), 12);
  });

  it("records checkpoints from epoch 0 to the last epoch on a log grid", () => {
    expect(CHECKPOINTS[0]).toBe(0);
    expect(CHECKPOINTS[CHECKPOINTS.length - 1]).toBe(MAX_EPOCHS);
    expect(CHECKPOINTS).toContain(100);
    const run = train({ ...BASE, degree: 5 });
    expect(run.checkpoints.map((checkpoint) => checkpoint.epoch)).toEqual([...CHECKPOINTS]);
  });

  it("is deterministic for a given seed and batch size", () => {
    const a = train({ ...BASE, degree: 7, batchSize: 4 });
    const b = train({ ...BASE, degree: 7, batchSize: 4 });
    expect(last(a).validation).toBe(last(b).validation);
  });

  it("reports the loss of the model it stores", () => {
    const run = train({ ...BASE, degree: 5 });
    const { train: samples } = makeDataset(1);
    const checkpoint = run.checkpoints[checkpointAt(run, 100)];
    const recomputed =
      samples.reduce((sum, sample) => sum + (predict(checkpoint.theta, sample.x) - sample.y) ** 2, 0) /
      samples.length;
    expect(recomputed).toBeCloseTo(checkpoint.train, 10);
  });

  it("underfits with a straight line: both losses stay high", () => {
    const run = train({ ...BASE, degree: 1 });
    expect(last(run).train).toBeGreaterThan(4 * NOISE_VARIANCE);
    expect(last(run).validation).toBeGreaterThan(4 * NOISE_VARIANCE);
    expect(statusAt(run, MAX_EPOCHS)).toBe("underfitting");
  });

  it("overfits at degree 11: training keeps falling while validation climbs far above its early best", () => {
    const run = train({ ...BASE, degree: 11 });
    const best = run.checkpoints[run.bestIndex];
    expect(best.epoch).toBeLessThan(30);
    expect(best.validation).toBeLessThan(0.11);
    expect(last(run).validation).toBeGreaterThan(20 * best.validation);
    expect(last(run).train).toBeLessThan(best.train);
    expect(last(run).train).toBeLessThan(run.trueLoss.train);
    expect(statusAt(run, MAX_EPOCHS)).toBe("overfitting");
    expect(statusAt(run, best.epoch)).toBe("fitting");
  });

  it("keeps degree 5 close to its best validation loss", () => {
    const run = train({ ...BASE, degree: 5 });
    expect(last(run).validation).toBeLessThan(OVERFIT_RATIO * run.checkpoints[run.bestIndex].validation);
    expect(statusAt(run, MAX_EPOCHS)).toBe("fitting");
  });

  it("weight decay 0.03 holds degree 11 near its best validation loss", () => {
    const run = train({ ...BASE, degree: 11, weightDecay: 0.03 });
    expect(last(run).validation).toBeLessThan(0.15);
    expect(statusAt(run, MAX_EPOCHS)).toBe("fitting");
  });

  it("a ten times smaller learning rate reaches the same best about ten times later", () => {
    const fast = train({ ...BASE, degree: 11, learningRate: 0.1 });
    const slow = train({ ...BASE, degree: 11, learningRate: 0.01 });
    const ratio = slow.checkpoints[slow.bestIndex].epoch / fast.checkpoints[fast.bestIndex].epoch;
    expect(ratio).toBeGreaterThan(6);
    expect(ratio).toBeLessThan(16);
    expect(slow.checkpoints[slow.bestIndex].validation).toBeCloseTo(fast.checkpoints[fast.bestIndex].validation, 2);
  });

  it("diverges when the learning rate is too large for the model or the batch", () => {
    expect(train({ ...BASE, degree: 11, learningRate: 0.6 }).divergedAt).not.toBeNull();
    expect(train({ ...BASE, degree: 5, learningRate: 0.6 }).divergedAt).toBeNull();
    expect(train({ ...BASE, degree: 5, learningRate: 0.3, batchSize: 4 }).divergedAt).not.toBeNull();
    expect(train({ ...BASE, degree: 5, learningRate: 0.3, batchSize: 16 }).divergedAt).toBeNull();
  });

  it("measures more gradient noise for smaller batches and none for the full batch", () => {
    const run = train({ ...BASE, degree: 11 });
    const theta = run.checkpoints[checkpointAt(run, 100)].theta;
    const four = gradientNoise(theta, 4, 1);
    const eight = gradientNoise(theta, 8, 1);
    const full = gradientNoise(theta, 16, 1);
    expect(full.noise).toBe(0);
    expect(four.noise).toBeGreaterThan(eight.noise);
    expect(four.full).toBeCloseTo(full.full, 12);
  });
});

describe("training-dynamics state", () => {
  it("migrates the version 1 slider payload", () => {
    const state = definition.hydrateState(JSON.stringify({ epoch: 35, capacity: 62, regularization: 28 }));
    expect(state.capacity).toBeUndefined();
    expect(state.regularization).toBeUndefined();
    expect(state.epoch).toBeUndefined();
    expect(state.degree).toBe(8);
    expect(state.weightDecayIndex).toBe(2);
    expect(state.epochIndex).toBe(29);
  });
});
