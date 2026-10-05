import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import { checkpointAt, train, type TrainRun } from "./sim";
import definition from "./module";

/**
 * Pins the comparisons the checkpoint questions rely on, as comparisons rather than
 * figures so a retrain or a tweak to the trainer cannot silently stale a question.
 * (The exact figures the lesson quotes are pinned in quoted.test.ts.)
 *
 * Lab defaults: seed 1, learning rate 0.1, batch 16 (full), weight decay 0, degree 5.
 */
const BASE = { learningRate: 0.1, weightDecay: 0, batchSize: 16, seed: 1 };
const MAX = 10_000;
const run = (degree: number, extra: Partial<typeof BASE> = {}) => train({ ...BASE, ...extra, degree });
const at = (trainRun: TrainRun, epoch: number) => trainRun.checkpoints[checkpointAt(trainRun, epoch)];
const best = (trainRun: TrainRun) => trainRun.checkpoints[trainRun.bestIndex];
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

describe("training-dynamics checkpoint questions", () => {
  it("question 1: at degree 5 validation sits above training and the gap is steady from epoch 100, while at degree 11 it keeps widening", () => {
    const five = run(5);
    const gaps = five.checkpoints.filter((checkpoint) => checkpoint.epoch >= 100).map((checkpoint) => checkpoint.validation - checkpoint.train);
    expect(Math.min(...gaps)).toBeGreaterThan(0);
    expect(Math.max(...gaps) - Math.min(...gaps)).toBeLessThan(0.005);
    const eleven = run(11);
    const gapAt = (epoch: number) => at(eleven, epoch).validation - at(eleven, epoch).train;
    expect(gapAt(1000)).toBeGreaterThan(gapAt(100));
    expect(gapAt(MAX)).toBeGreaterThan(10 * gapAt(100));
    expect(correctOption(0)).toMatch(/^A normal difference between fitting seen and unseen points/);
  });

  it("question 3: at degree 11 validation bottoms out early, then climbs for thousands of epochs while training keeps falling", () => {
    const eleven = run(11);
    const low = best(eleven);
    expect(low.epoch).toBeLessThan(100);
    expect(at(eleven, MAX).validation).toBeGreaterThan(10 * low.validation);
    expect(at(eleven, MAX).train).toBeLessThan(low.train);
    let previous = Number.POSITIVE_INFINITY;
    for (const checkpoint of eleven.checkpoints.filter((point) => point.epoch >= low.epoch)) {
      expect(checkpoint.train).toBeLessThan(previous);
      previous = checkpoint.train;
    }
    expect(correctOption(2)).toMatch(/^The early model/);
  });

  it("question 4: weight decay 0.03 and early stopping both beat the final unconstrained model on validation, and both leave training loss higher", () => {
    const plain = run(11);
    const decayed = run(11, { weightDecay: 0.03 });
    const final = at(plain, MAX);
    // Weight decay: much better validation, slightly worse training loss.
    expect(at(decayed, MAX).validation).toBeLessThan(final.validation / 10);
    expect(at(decayed, MAX).train).toBeGreaterThan(final.train);
    // Early stopping: much better validation, higher training loss than the last epoch.
    expect(best(plain).validation).toBeLessThan(final.validation / 10);
    expect(best(plain).train).toBeGreaterThan(final.train);
    expect(correctOption(3)).toMatch(/^Each limits how far the curve can bend toward noise/);
  });

  it("question 5: degree 5 and 11 validate alike at epoch 13 but degree 11 is far worse at epoch 10,000", () => {
    const five = run(5);
    const eleven = run(11);
    const early = [at(five, 13).validation, at(eleven, 13).validation];
    expect(Math.max(...early) / Math.min(...early)).toBeLessThan(1.25);
    expect(at(eleven, MAX).validation).toBeGreaterThan(10 * at(five, MAX).validation);
    expect(correctOption(4)).toMatch(/^Similar at epoch 13, but far worse for degree 11/);
  });

  it("question 6: a ten times smaller learning rate reaches its best about ten times later, with a similar best, and still overfits", () => {
    const fast = run(11);
    const slow = run(11, { learningRate: 0.01 });
    expect(best(slow).epoch / best(fast).epoch).toBeGreaterThan(5);
    expect(best(slow).epoch / best(fast).epoch).toBeLessThan(20);
    expect(Math.abs(best(slow).validation - best(fast).validation) / best(fast).validation).toBeLessThan(0.1);
    expect(at(slow, MAX).validation).toBeGreaterThan(5 * best(slow).validation);
    expect(correctOption(5)).toMatch(/^Much the same run, only slower/);
  });
});
