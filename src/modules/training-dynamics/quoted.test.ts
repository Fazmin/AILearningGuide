import { describe, expect, it } from "vitest";
import {
  CHECKPOINTS,
  checkpointAt,
  gradientNoise,
  DIVERGED_LOSS,
  NOISE_SD,
  NOISE_VARIANCE,
  OVERFIT_RATIO,
  predict,
  statusAt,
  train,
  UNDERFIT_RATIO,
  type TrainRun,
} from "./sim";

/**
 * Pins the measured values the training-dynamics lesson and card text quote
 * (module.ts, content/standard.mdx, content/plain.mdx, card-info.ts).
 *
 * Every run uses the lab's defaults unless a comment says otherwise: seed 1,
 * eta 0.1, batch 16 (full), lambda 0, 10,000 epochs. The lab reads a checkpoint
 * by epoch and formats losses with `formatLoss` (3 decimals below 1, 2 decimals
 * from 1 to 10, 1 decimal from 10), so each assertion rounds the same way.
 */
const BASE = { learningRate: 0.1, weightDecay: 0, batchSize: 16, seed: 1 };
const MAX = 10_000;

const run = (degree: number, extra: Partial<typeof BASE> = {}) => train({ ...BASE, ...extra, degree });
const at = (trainRun: TrainRun, epoch: number) => trainRun.checkpoints[checkpointAt(trainRun, epoch)];
const best = (trainRun: TrainRun) => trainRun.checkpoints[trainRun.bestIndex];
const round = (value: number, digits: number) => Number(value.toFixed(digits));
const norm = (theta: number[]) => Math.sqrt(theta.reduce((sum, value) => sum + value * value, 0));
/** The lab's "peak |y-hat|": the largest |prediction| over 161 evenly spaced points on [-1, 1]. */
const peak = (theta: number[]) =>
  Array.from({ length: 161 }, (_, index) => Math.abs(predict(theta, -1 + index / 80))).reduce((a, b) => Math.max(a, b), 0);

describe("training-dynamics quoted values", () => {
  it("matches the setup figures: 83 checkpoints, sigma^2 = 0.0625, 2 sigma^2 = 0.125", () => {
    // standard.mdx:18 and card-info.ts:18,45 ("83 times", "83 checkpoints")
    expect(CHECKPOINTS).toHaveLength(83);
    // standard.mdx:14-15 ("noise of standard deviation 0.25"), standard.mdx:19 and card-info.ts:10 ("σ² = 0.0625")
    expect(NOISE_SD).toBe(0.25);
    expect(NOISE_VARIANCE).toBe(0.0625);
    // standard.mdx:90 and card-info.ts:19: the status thresholds the prose names (1.5 times the best, 10^6 for diverged)
    expect(OVERFIT_RATIO).toBe(1.5);
    expect(DIVERGED_LOSS).toBe(1e6);
    // card-info.ts:19 ("above 2σ² = 0.125")
    expect(UNDERFIT_RATIO * NOISE_VARIANCE).toBe(0.125);
  });

  it("underfits with a straight line: 0.322 training and 0.279 validation once settled", () => {
    // standard.mdx:23-24: "degree 1 scores 0.322 on training and 0.279 on validation at every epoch".
    const line = run(1);
    const settled = line.checkpoints.filter((checkpoint) => checkpoint.epoch >= 24);
    expect(settled.length).toBeGreaterThan(50);
    for (const checkpoint of settled) {
      expect(round(checkpoint.train, 3)).toBe(0.322);
      expect(round(checkpoint.validation, 3)).toBe(0.279);
    }
    expect(round(at(line, MAX).train, 3)).toBe(0.322);
    expect(round(at(line, MAX).validation, 3)).toBe(0.279);
    // standard.mdx now says the pair is reached "by about epoch 24 and stays there": epoch 0 reads 0.582 / 0.565,
    // epoch 13 reads 0.322 / 0.282, and the settled pair holds from epoch 24 on (pinned above).
    expect(round(at(line, 0).train, 3)).toBe(0.582);
    expect(round(at(line, 0).validation, 3)).toBe(0.565);
  });

  it("matches the default degree-5 run (the Fitting regime and the card's default readout)", () => {
    const five = run(5);
    const low = best(five);
    // standard.mdx:25 ("validation reaches 0.100 by epoch 11") and :59 ("0.100 at epoch 11");
    // card-info.ts:13 ("0.100 at epoch 11")
    expect(low.epoch).toBe(11);
    expect(round(low.validation, 3)).toBe(0.1);
    // standard.mdx:25-26: "ends at 0.111"
    expect(round(at(five, MAX).validation, 3)).toBe(0.111);
    // card-info.ts:13: at epoch 100 the metrics read training 0.017, validation 0.111, gap 0.093, status fitting
    const hundred = at(five, 100);
    expect(hundred.epoch).toBe(100);
    expect(round(hundred.train, 3)).toBe(0.017);
    expect(round(hundred.validation, 3)).toBe(0.111);
    expect(round(hundred.validation - hundred.train, 3)).toBe(0.093);
    expect(statusAt(five, 100)).toBe("fitting");
    // standard.mdx:73 ("At degree 5 the gap settles near 0.09 and stays there")
    for (const checkpoint of five.checkpoints.filter((point) => point.epoch >= 100)) {
      expect(round(checkpoint.validation - checkpoint.train, 2)).toBe(0.09);
    }
  });

  it("matches the degree-11 overfitting run", () => {
    const eleven = run(11);
    const low = best(eleven);
    const final = at(eleven, MAX);
    // module.ts:53,122, standard.mdx:28, plain.mdx:53, card-info.ts:23: lowest validation 0.098 at epoch 13
    expect(low.epoch).toBe(13);
    expect(round(low.validation, 3)).toBe(0.098);
    // module.ts:53,122, standard.mdx:28,65, plain.mdx:58, card-info.ts:23,56,113: validation ends at 8.63
    expect(round(final.validation, 2)).toBe(8.63);
    // standard.mdx:28 and card-info.ts:23: training loss falls to 0.007
    expect(round(final.train, 3)).toBe(0.007);
    // standard.mdx:29 and card-info.ts:28: the true curve scores 0.081 on the training points
    expect(round(eleven.trueLoss.train, 3)).toBe(0.081);
    // card-info.ts:28: "Training loss below the dotted noise line": 0.007 is below sigma^2 = 0.0625
    expect(final.train).toBeLessThan(NOISE_VARIANCE);
    // standard.mdx:61 and card-info.ts:23: the status reads overfitting from epoch 215
    const firstOverfitting = eleven.checkpoints.find((checkpoint) => statusAt(eleven, checkpoint.epoch) === "overfitting");
    expect(firstOverfitting?.epoch).toBe(215);
    expect(statusAt(eleven, 13)).toBe("fitting");
    // plain.mdx:54 and standard.mdx:71-72: validation ends 88 times worse than its best
    expect(Math.round(final.validation / low.validation)).toBe(88);
    // standard.mdx:71: training loss "falls at every checkpoint"
    for (let index = 1; index < eleven.checkpoints.length; index += 1) {
      expect(eleven.checkpoints[index].train).toBeLessThan(eleven.checkpoints[index - 1].train);
    }
    // standard.mdx:62 and card-info.ts:86: the curve swings to a peak |y-hat| of about 24 near the right edge
    expect(round(peak(final.theta), 0)).toBe(24);
    // standard.mdx:113 and card-info.ts:90: weight size 2.48 with no decay at epoch 10,000
    expect(round(norm(final.theta), 2)).toBe(2.48);
  });

  it("matches the weight-decay figures at degree 11", () => {
    // standard.mdx:65, plain.mdx:58, card-info.ts:56: 0.01, 0.03, 0.1 give 0.310, 0.135, 0.102 at epoch 10,000
    const finals = [0.01, 0.03, 0.1].map((weightDecay) => at(run(11, { weightDecay }), MAX));
    expect(finals.map((checkpoint) => round(checkpoint.validation, 3))).toEqual([0.31, 0.135, 0.102]);
    // card-info.ts:61: two fixes reach almost the same validation loss: epoch 13 gives 0.098, lambda 0.1 gives 0.102
    expect(round(best(run(11)).validation, 3)).toBe(0.098);
    // standard.mdx:113 and card-info.ts:90: weight size 0.74 with lambda 0.03 (against 2.48 with none);
    // plain.mdx:99 "about three times smaller"
    const decayed = norm(finals[1].theta);
    const plain = norm(at(run(11), MAX).theta);
    expect(round(decayed, 2)).toBe(0.74);
    expect(round(plain / decayed, 0)).toBe(3);
  });

  it("matches the learning-rate and batch-size figures", () => {
    // standard.mdx:75-76 and card-info.ts:24: eta 0.01 reaches its best, 0.097, at epoch 147 instead of 13
    const slow = run(11, { learningRate: 0.01 });
    expect(best(slow).epoch).toBe(147);
    expect(round(best(slow).validation, 3)).toBe(0.097);
    expect(best(run(11)).epoch).toBe(13);
    // standard.mdx:76 and card-info.ts:24: "On the log axis the curves slide one decade to the right"
    expect(Math.round(Math.log10(best(slow).epoch / best(run(11)).epoch))).toBe(1);
    // standard.mdx:77-78 and card-info.ts:57: eta 0.3 at degree 5 diverges at epoch 22 with batch 4,
    // trains normally with batch 8 or 16
    expect(run(5, { learningRate: 0.3, batchSize: 4 }).divergedAt).toBe(22);
    expect(run(5, { learningRate: 0.3, batchSize: 8 }).divergedAt).toBeNull();
    expect(run(5, { learningRate: 0.3, batchSize: 16 }).divergedAt).toBeNull();
    // card-info.ts:58: eta 0.6 diverges at epoch 562 at degree 11, while degree 5 is stable
    expect(run(11, { learningRate: 0.6 }).divergedAt).toBe(562);
    expect(run(5, { learningRate: 0.6 }).divergedAt).toBeNull();
    // card-info.ts:53 ("4 steps at batch 4, 1 at the full 16"): 16 training points / batch size
    expect([4, 16].map((batchSize) => 16 / batchSize)).toEqual([4, 1]);
  });

  it("matches the gradient readouts at degree 11, batch 4, epoch 100", () => {
    // card-info.ts:62: Minibatch noise 0.199 and Full-batch |grad L| 0.051. The lab trains with the chosen batch size,
    // then reads the epoch-100 weights of that same run.
    const small = run(11, { batchSize: 4 });
    const theta = at(small, 100).theta;
    const { full, noise } = gradientNoise(theta, 4, BASE.seed);
    expect(round(noise, 3)).toBe(0.199);
    expect(round(full, 3)).toBe(0.051);
    // "several times": 0.199 / 0.051 is about 3.9
    expect(noise / full).toBeGreaterThan(3);
  });

  it("matches the capacity-sweep figures", () => {
    const sweep = Array.from({ length: 13 }, (_, offset) => run(offset + 1));
    const validationAt = (epoch: number) => sweep.map((entry) => at(entry, epoch).validation);
    // card-info.ts:113 ("at epoch 10,000 it climbs from 0.100 at degree 7 to 0.246 at degree 9,
    // 8.63 at degree 11, and 10.6 at degree 12")
    const late = validationAt(MAX);
    expect(round(late[6], 3)).toBe(0.1);
    expect(round(late[8], 3)).toBe(0.246);
    expect(round(late[10], 2)).toBe(8.63);
    expect(round(late[11], 1)).toBe(10.6);
    // card-info.ts:113: at epoch 13 validation sits between 0.096 and 0.119 for every degree from 5 to 13
    const early = validationAt(13).slice(4);
    expect(early).toHaveLength(9);
    expect(round(Math.min(...early), 3)).toBe(0.096);
    for (const value of early) {
      expect(value).toBeGreaterThanOrEqual(0.0955);
      expect(value).toBeLessThanOrEqual(0.1195);
    }
    // card-info.ts now quotes "between 0.096 and 0.117": the largest value over degrees 5 to 13 is 0.117 (degree 13);
    // 0.119 is the degree-3 value, outside that span.
    expect(round(Math.max(...early), 3)).toBe(0.117);
    // card-info.ts:113 and standard.mdx:80: "barely matters at epoch 13, matters enormously at epoch 10,000"
    expect(Math.max(...early) / Math.min(...early)).toBeLessThan(1.25);
    expect(Math.max(...late.slice(4)) / Math.min(...late.slice(4))).toBeGreaterThan(50);
    // card-info.ts:116 ("Degrees 1 and 2 stay high at every epoch"): validation stays above 4 sigma^2 at every checkpoint
    for (const degree of [1, 2]) {
      for (const checkpoint of sweep[degree - 1].checkpoints) {
        expect(checkpoint.validation).toBeGreaterThan(4 * NOISE_VARIANCE);
      }
    }
  });
});
