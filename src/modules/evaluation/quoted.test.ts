import { describe, expect, it } from "vitest";
import { TINY_CORPORA, tinyPerplexity, trainTinyModel, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import definition from "./module";
import {
  itemsForHalfWidth,
  meanInterval,
  pairedDifference,
  parseItems,
  passAtK,
  proportionStandardError,
  scoreContinuation,
  wilsonInterval,
  type BenchmarkItem,
} from "./eval";

/**
 * Quote locations are given by file and section, not line number, because the lesson text is still being edited.
 * Pins the measured values the evaluation lesson (standard.mdx and plain.mdx), the card text (card-info.ts)
 * and the glossary quote. Mirrors Explore: the benchmark is the last three Harbor sentences, held out of
 * training; the control is the Proverbs corpus; four checkpoints at 3, 10, 30 and 80 epochs (batch 16,
 * learning rate 0.6, seed 1) are trained clean and with the benchmark source leaked in, and scored with the
 * same helpers (scoreContinuation, wilsonInterval, meanInterval, pairedDifference).
 */
const sentences = TINY_CORPORA.harbor.text
  .split(".")
  .map((part) => part.trim())
  .filter(Boolean);
const BENCHMARK_SOURCE = `${sentences.slice(sentences.length - 3).join(". ")}.`;
const TRAINING_TEXT = `${sentences.slice(0, sentences.length - 3).join(". ")}.`;
const LEAKED_TEXT = `${TRAINING_TEXT} ${BENCHMARK_SOURCE}`;
const CONTROL_TEXT = TINY_CORPORA.proverbs.text;
const BUDGETS = [3, 10, 30, 80];
const ITEMS_TEXT = definition.initialState.items as string;
const items = parseItems(ITEMS_TEXT);

/** The epoch counts Explore derives from the training budget multiplier. */
const epochsFor = (scale: number) => BUDGETS.map((budget) => Math.max(1, Math.round(budget * scale)));

function runLeaderboard(text: string, benchmark: BenchmarkItem[]) {
  return BUDGETS.map((epochs) => {
    const run = trainTinyModel({ text, epochs, batchSize: 16, learningRate: 0.6, seed: 1 });
    const results = benchmark.map((item) => {
      const correctScore = scoreContinuation(run.weights, item.context, item.correct).mean;
      const distractorScore = scoreContinuation(run.weights, item.context, item.distractor).mean;
      return { ...item, margin: correctScore - distractorScore, passed: correctScore > distractorScore };
    });
    const passes = results.filter((result) => result.passed).length;
    return {
      epochs,
      results,
      passes,
      passRate: passes / benchmark.length,
      wilson: wilsonInterval(passes, benchmark.length),
      margin: meanInterval(results.map((result) => result.margin)),
      benchmarkPerplexity: tinyPerplexity(run.weights, BENCHMARK_SOURCE),
      controlPerplexity: tinyPerplexity(run.weights, CONTROL_TEXT),
    };
  });
}
type Board = ReturnType<typeof runLeaderboard>;
const rank = (board: Board) =>
  [...board].sort((left, right) => right.passRate - left.passRate || right.margin.mean - left.margin.mean);
const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
const round = (value: number, digits: number) => Number(value.toFixed(digits));

const clean = runLeaderboard(TRAINING_TEXT, items);
const leaked = runLeaderboard(LEAKED_TEXT, items);
const byEpochs = (board: Board, epochs: number) => board.find((row) => row.epochs === epochs)!;

describe("evaluation quoted values", () => {
  it("uses the default lab state", () => {
    expect(items).toHaveLength(6);
    expect(definition.initialState.epochsScale).toBe(1);
    expect(definition.initialState.contaminated).toBe(false);
    // standard.mdx "What it is" (The held-out half), plain.mdx "What it is": a model that learned nothing scores ln(30) = 3.401 nats, perplexity 30.
    expect(UNIFORM_CROSS_ENTROPY).toBeCloseTo(3.401, 3);
    expect(Math.exp(UNIFORM_CROSS_ENTROPY)).toBeCloseTo(30, 10);
  });

  it("matches the clean leaderboard at the defaults", () => {
    const ranked = rank(clean);
    const [leader, runnerUp] = ranked;
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Leaderboard" controls: the 80- and 30-epoch checkpoints lead, both 3 of 6.
    expect([leader.epochs, runnerUp.epochs]).toEqual([80, 30]);
    expect([leader.passes, runnerUp.passes]).toEqual([3, 3]);
    // Interval 19%-81% for both.
    for (const row of [leader, runnerUp]) expect([pct(row.wilson[0]), pct(row.wilson[1])]).toEqual(["19%", "81%"]);
    // standard.mdx "How to play with it": mean margins 0.012 and 0.011.
    expect(leader.margin.mean).toBeCloseTo(0.012, 3);
    expect(runnerUp.margin.mean).toBeCloseTo(0.011, 3);
    // standard.mdx and plain.mdx "What to notice": "All three other intervals overlap the leader's".
    expect(ranked.slice(1).filter((row) => row.wilson[1] >= leader.wilson[0])).toHaveLength(3);
  });

  it("matches the paired leader-versus-runner-up callout", () => {
    const [leader, runnerUp] = rank(clean);
    const gap = pairedDifference(
      leader.results.map((result) => result.margin),
      runnerUp.results.map((result) => result.margin),
    );
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Leaderboard" controls: +0.002 with an interval from -0.220 to +0.224.
    expect(gap.mean).toBeCloseTo(0.002, 3);
    expect(gap.low).toBeCloseTo(-0.22, 3);
    expect(gap.high).toBeCloseTo(0.224, 3);
    // The interval includes zero, so the ranking is "a coin flip".
    expect(gap.low).toBeLessThan(0);
    expect(gap.high).toBeGreaterThan(0);
  });

  it("matches the item-level results at 80 epochs", () => {
    const margins = Object.fromEntries(byEpochs(clean, 80).results.map((result) => [result.context, result.margin]));
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Item-level results" controls: "the boats stay in | port | town" fails by -0.840.
    expect(margins["the boats stay in"]).toBeCloseTo(-0.84, 3);
    // standard.mdx "How to play with it", card-info.ts "Item-level results" notice: "the water stays | warm | wide" passes by +1.213, the biggest margin.
    expect(margins["the water stays"]).toBeCloseTo(1.213, 3);
    expect(Math.max(...Object.values(margins))).toBe(margins["the water stays"]);
    // card-info.ts "Item-level results" controls: leak the source and the boats item passes by +0.012.
    const leakedBoats = byEpochs(leaked, 80).results.find((result) => result.context === "the boats stay in")!;
    expect(leakedBoats.margin).toBeCloseTo(0.012, 3);
    expect(leakedBoats.passed).toBe(true);
  });

  it("matches the sample-size figures", () => {
    // standard.mdx "What it is" (The error bar), plain.mdx "How sure is a score?", Confidence interval glossary entry: six items at 50% give plus or minus 20 points (one standard error).
    expect(Math.round(proportionStandardError(0.5, 6) * 100)).toBe(20);
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Leaderboard" howItWorks: a +/-5 point interval at 50% needs 385 items.
    expect(itemsForHalfWidth(0.5, 0.05)).toBe(385);
    // The lab's Items for +/-5 pts metric reads 385 at the leader's 3/6 pass rate.
    expect(itemsForHalfWidth(rank(clean)[0].passRate, 0.05)).toBe(385);
  });

  it("matches the contamination shift at 80 epochs", () => {
    const cleanStrongest = byEpochs(clean, 80);
    const leakedStrongest = byEpochs(leaked, 80);
    const shift = pairedDifference(
      leakedStrongest.results.map((result) => result.margin),
      cleanStrongest.results.map((result) => result.margin),
    );
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Contamination control" controls: +0.519 margin per item, interval +0.220 to +0.817.
    expect(shift.mean).toBeCloseTo(0.519, 3);
    expect(shift.low).toBeCloseTo(0.22, 3);
    expect(shift.high).toBeCloseTo(0.817, 3);
    // standard.mdx and plain.mdx "What to notice": the leak takes the leaderboard to 6 of 6.
    expect(leakedStrongest.passes).toBe(6);
    // standard.mdx "What to notice": benchmark perplexity from 9.33 to 5.83.
    expect(round(cleanStrongest.benchmarkPerplexity, 2)).toBe(9.33);
    expect(round(leakedStrongest.benchmarkPerplexity, 2)).toBe(5.83);
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Contamination control" controls: control perplexity moves -2.8%.
    const controlShift = leakedStrongest.controlPerplexity / cleanStrongest.controlPerplexity - 1;
    expect(round(controlShift * 100, 1)).toBe(-2.8);
    // The benchmark moves far more than the control (the disproportion is the signature).
    expect(Math.abs(shift.mean / cleanStrongest.margin.mean)).toBeGreaterThan(10);
    expect(Math.abs(controlShift)).toBeLessThan(0.05);
  });

  it("matches the contamination shift at 30 epochs", () => {
    const cleanMid = byEpochs(clean, 30);
    const leakedMid = byEpochs(leaked, 30);
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Contamination control" notice: 3/6 becomes 5/6; perplexity falls from 10.08 to 6.65.
    expect([cleanMid.passes, leakedMid.passes]).toEqual([3, 5]);
    expect(round(cleanMid.benchmarkPerplexity, 2)).toBe(10.08);
    expect(round(leakedMid.benchmarkPerplexity, 2)).toBe(6.65);
    // standard.mdx "What to notice": the control moves -5.7% at 30 epochs.
    expect(round((leakedMid.controlPerplexity / cleanMid.controlPerplexity - 1) * 100, 1)).toBe(-5.7);
  });

  it("matches the leader's interval under contamination", () => {
    const leader = rank(leaked)[0];
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Leaderboard" notice: at 6 of 6 the standard error reads +/-0.0 points
    // and the Wilson interval still runs from 61% to 100%.
    expect(leader.passes).toBe(6);
    expect((proportionStandardError(leader.passRate, items.length) * 100).toFixed(1)).toBe("0.0");
    expect([pct(leader.wilson[0]), pct(leader.wilson[1])]).toEqual(["61%", "100%"]);
  });

  it("makes the xyzzy distractor an easy pass", () => {
    // card-info.ts "Benchmark editor" controls: replace `morning | evening` with `morning | xyzzy` and the item becomes an easy pass.
    const swapped = parseItems(ITEMS_TEXT.replace("morning | evening", "morning | xyzzy"));
    const swappedItem = byEpochs(runLeaderboard(TRAINING_TEXT, swapped), 80).results.find(
      (result) => result.context === "the fog returns in the",
    )!;
    const original = byEpochs(clean, 80).results.find((result) => result.context === "the fog returns in the")!;
    expect(swappedItem.passed).toBe(true);
    expect(swappedItem.margin).toBeGreaterThan(1);
    expect(swappedItem.margin).toBeGreaterThan(original.margin);
  });

  it("matches the budgets the multiplier produces", () => {
    // card-info.ts "Benchmark editor" howItWorks: at 0.25x the budgets become 1, 3, 8 and 20 epochs; at 3x they become 9, 30, 90 and 240.
    expect(epochsFor(0.25)).toEqual([1, 3, 8, 20]);
    expect(epochsFor(3)).toEqual([9, 30, 90, 240]);
    expect(epochsFor(1)).toEqual(BUDGETS);
  });

  it("matches the pass@k worked example", () => {
    // standard.mdx and plain.mdx "Going deeper", Pass@k glossary entry: n = 10, c = 3, k = 5 is 1 - C(7,5)/C(10,5) = 1 - 21/252 = 0.917.
    expect(passAtK(10, 3, 5)).toBeCloseTo(1 - 21 / 252, 12);
    expect(round(passAtK(10, 3, 5), 3)).toBe(0.917);
    // standard.mdx "Going deeper": the plug-in estimate 1 - (1 - c/n)^k gives 0.832.
    expect(round(1 - (1 - 3 / 10) ** 5, 3)).toBe(0.832);
  });
});
