import { describe, expect, it } from "vitest";
import {
  CHALLENGE_DRAWS,
  CHALLENGE_GOAL,
  CHALLENGE_ROUNDS,
  EQUAL_BARS,
  MAX_BAR,
  MIN_BAR,
  ONE_SD_SHARE,
  PROMPTS,
  SKY_BARS,
  SKY_SCORES,
  TEST_COUNT,
  TEST_SIZES,
  WORDS,
  biggestGap,
  challengeScore,
  challengeTally,
  conditional,
  crossEntropy,
  drawSamples,
  entropy,
  expectedValue,
  fairChoices,
  gaussianPdf,
  isBars,
  klDivergence,
  likelihoodOf,
  mulberry32,
  normalize,
  pick,
  pileStats,
  posterior,
  priorFor,
  roundToTotal,
  runningShare,
  setBar,
  shapeShares,
  sigmoid,
  simulateTests,
  softmax,
  sum,
  surprise,
  tally,
  variance,
} from "./chance";

const sky = normalize(SKY_BARS);

describe("bars that add to one", () => {
  it("normalizes any non-negative list, and an empty one to all-equal", () => {
    expect(sum(normalize([3, 1]))).toBeCloseTo(1, 12);
    expect(normalize([3, 1])).toEqual([0.75, 0.25]);
    expect(normalize([0, 0, 0, 0])).toEqual([0.25, 0.25, 0.25, 0.25]);
  });

  it("rounds shares to whole numbers that keep the total", () => {
    expect(roundToTotal([1, 1, 1], 100)).toEqual([34, 33, 33]);
    expect(roundToTotal([3, 4, 20, 11, 2], 100)).toEqual([7, 10, 50, 28, 5]);
    expect(sum(roundToTotal([0.2, 0.3, 0.5], 7))).toBe(7);
  });

  it("recognises a valid set of bars and nothing else", () => {
    expect(isBars([...SKY_BARS])).toBe(true);
    expect(isBars([...EQUAL_BARS])).toBe(true);
    expect(isBars([50, 25, 12, 8])).toBe(false);
    expect(isBars([50, 25, 12, 8, 6])).toBe(false);
    expect(isBars([100, 0, 0, 0, 0])).toBe(false);
    expect(isBars([50.5, 24.5, 12, 8, 5])).toBe(false);
    expect(isBars("50,25,12,8,5")).toBe(false);
    expect(isBars(null)).toBe(false);
  });

  it("keeps the others in proportion when one bar moves", () => {
    expect(setBar(SKY_BARS, 0, 80)).toEqual([80, 10, 5, 3, 2]);
    expect(setBar(EQUAL_BARS, 0, 60)).toEqual([60, 10, 10, 10, 10]);
    expect(setBar(SKY_BARS, 0, 50)).toEqual([...SKY_BARS]);
  });

  it("always returns valid bars, whatever it is asked for", () => {
    const starts = [[...SKY_BARS], [...EQUAL_BARS], [96, 1, 1, 1, 1], [1, 1, 1, 1, 96], [40, 30, 20, 9, 1]];
    for (const start of starts) {
      for (let index = 0; index < WORDS.length; index += 1) {
        for (const value of [-50, 0, 1, 2, 33, 50, 95, 96, 97, 150, 12.4, Number.NaN, Number.POSITIVE_INFINITY]) {
          const next = setBar(start, index, value);
          expect(isBars(next), `${start} [${index}] → ${value} gave ${next}`).toBe(true);
          if (Number.isFinite(value)) expect(next[index]).toBe(Math.min(MAX_BAR, Math.max(MIN_BAR, Math.round(value))));
        }
      }
    }
  });
});

describe("the count table", () => {
  it("has 200 stories: 100, 50 and 50", () => {
    expect(PROMPTS.map((prompt) => sum(prompt.counts))).toEqual([100, 50, 50]);
  });

  it("turns a row into bars by dividing by the row's own total", () => {
    expect(conditional(0)).toEqual([0.5, 0.25, 0.12, 0.08, 0.05]);
    expect(conditional(1)).toEqual([0.1, 0.06, 0.1, 0.04, 0.7]);
    expect(conditional(2)).toEqual([0.4, 0.2, 0.16, 0.1, 0.14]);
    for (let prompt = 0; prompt < PROMPTS.length; prompt += 1) expect(sum(conditional(prompt))).toBeCloseTo(1, 12);
  });

  it("reads a column as likelihoods", () => {
    expect(likelihoodOf(0)).toEqual([0.5, 0.1, 0.4]);
    expect(likelihoodOf(4)).toEqual([0.05, 0.7, 0.14]);
  });
});

describe("Bayes' rule", () => {
  it("matches counting straight from the table when the prior is the table's own", () => {
    // Column "blue" holds 50, 5 and 20 stories: 75 in all.
    const result = posterior(priorFor(0.5), likelihoodOf(0));
    expect(result.joint.map((value) => value * 200)).toEqual([50, 5, 20]);
    expect(result.evidence).toBeCloseTo(75 / 200, 12);
    expect(result.posterior[0]).toBeCloseTo(50 / 75, 12);
    expect(sum(result.posterior)).toBeCloseTo(1, 12);
  });

  it("lowers the posterior when the prior falls, with the likelihoods unchanged", () => {
    const rare = posterior(priorFor(0.1), likelihoodOf(0));
    expect(rare.posterior[0]).toBeCloseTo(0.05 / 0.275, 12);
    const at = [0.05, 0.1, 0.2, 0.5, 0.9].map((share) => posterior(priorFor(share), likelihoodOf(0)).posterior[0]);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
  });

  it("splits what is left of the prior equally between the other two prompts", () => {
    expect(priorFor(0.5)).toEqual([0.5, 0.25, 0.25]);
    expect(sum(priorFor(0.13))).toBeCloseTo(1, 12);
  });
});

describe("sampling", () => {
  it("is repeatable for a seed and different for another", () => {
    expect(drawSamples(sky, 50, 13)).toEqual(drawSamples(sky, 50, 13));
    expect(drawSamples(sky, 50, 13)).not.toEqual(drawSamples(sky, 50, 14));
    const random = mulberry32(7);
    const values = Array.from({ length: 1000 }, () => random());
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
  });

  it("maps a number between 0 and 1 onto the outcome whose stretch it lands in", () => {
    expect(pick(sky, 0)).toBe(0);
    expect(pick(sky, 0.4999)).toBe(0);
    expect(pick(sky, 0.5)).toBe(1);
    expect(pick(sky, 0.96)).toBe(4);
    expect(pick(sky, 0.999999)).toBe(4);
  });

  it("keeps the first draws when more are added, so a tally only grows", () => {
    const hundred = drawSamples(sky, 100, 13);
    expect(drawSamples(sky, 10, 13)).toEqual(hundred.slice(0, 10));
    expect(sum(tally(hundred))).toBe(100);
  });

  it("settles onto the bars as draws pile up (law of large numbers)", () => {
    const draws = drawSamples(sky, 1000, 13);
    const gaps = [10, 100, 1000].map((count) => biggestGap(tally(draws.slice(0, count)), sky));
    expect(gaps[0]).toBeGreaterThan(gaps[1]);
    expect(gaps[1]).toBeGreaterThan(gaps[2]);
    expect(gaps[2]).toBeLessThan(0.02);
    const running = runningShare(draws, 0);
    expect(running).toHaveLength(1000);
    expect(running[999]).toBeCloseTo(tally(draws)[0] / 1000, 12);
    expect(biggestGap([0, 0, 0, 0, 0], sky)).toBe(0);
  });
});

describe("mean, variance and the bell", () => {
  it("computes the expected value and variance of a number attached to each outcome", () => {
    const points = [1, 0, 0, 0, 0];
    expect(expectedValue(sky, points)).toBeCloseTo(0.5, 12);
    expect(variance(sky, points)).toBeCloseTo(0.25, 12);
    expect(expectedValue([0.25, 0.25, 0.25, 0.25], [-1, 0, 1, 2])).toBeCloseTo(0.5, 12);
    expect(variance([0.25, 0.25, 0.25, 0.25], [-1, 0, 1, 2])).toBeCloseTo(1.25, 12);
  });

  it("simulates the stated number of tests, each with a score between 0 and 1", () => {
    for (const size of TEST_SIZES) {
      const pile = simulateTests(0.5, size);
      expect(pile).toHaveLength(size + 1);
      expect(sum(pile)).toBe(TEST_COUNT);
    }
  });

  it("piles up around the mean with a spread close to sqrt(variance / questions)", () => {
    for (const size of TEST_SIZES) {
      const stats = pileStats(simulateTests(0.5, size));
      expect(Math.abs(stats.mean - 0.5)).toBeLessThan(0.01);
      expect(Math.abs(stats.sd - Math.sqrt(0.25 / size)) / Math.sqrt(0.25 / size)).toBeLessThan(0.05);
      expect(stats.sd).toBeCloseTo(Math.sqrt(stats.variance), 12);
    }
  });

  it("halves the spread when the test is four times as long", () => {
    const short = pileStats(simulateTests(0.5, 25)).sd;
    const long = pileStats(simulateTests(0.5, 100)).sd;
    expect(short / long).toBeGreaterThan(1.9);
    expect(short / long).toBeLessThan(2.1);
  });

  it("draws a bell whose area is 1 and whose one-standard-deviation band holds about 68%", () => {
    const area = (low: number, high: number, mean: number, spread: number) => {
      const steps = 2000;
      const width = (high - low) / steps;
      let total = 0;
      for (let step = 0; step < steps; step += 1) total += gaussianPdf(low + (step + 0.5) * width, mean, spread) * width;
      return total;
    };
    expect(area(-10, 10, 0.5, 0.01)).toBeCloseTo(1, 6);
    expect(area(0.4, 0.6, 0.5, 0.01)).toBeCloseTo(ONE_SD_SHARE, 3);
    expect(area(-2, 2, 0, 4)).toBeCloseTo(ONE_SD_SHARE, 3);
  });
});

describe("surprise, entropy and cross-entropy", () => {
  it("adds exactly one bit each time a chance halves", () => {
    expect(surprise(1)).toBe(0);
    expect(surprise(0.5)).toBe(1);
    expect(surprise(0.25)).toBe(2);
    expect(surprise(0.125)).toBe(3);
    expect(surprise(0.2) - surprise(0.4)).toBeCloseTo(1, 12);
    expect(surprise(0)).toBe(Number.POSITIVE_INFINITY);
  });

  it("turns a product of chances into a sum of surprises", () => {
    expect(surprise(0.5 * 0.25)).toBeCloseTo(surprise(0.5) + surprise(0.25), 12);
    expect(surprise(0.12 * 0.05)).toBeCloseTo(surprise(0.12) + surprise(0.05), 12);
  });

  it("is zero for a sure thing and largest for all-equal bars", () => {
    expect(entropy([1, 0, 0, 0, 0])).toBe(0);
    expect(entropy(normalize(EQUAL_BARS))).toBeCloseTo(Math.log2(5), 12);
    expect(entropy(sky)).toBeLessThan(Math.log2(5));
    expect(fairChoices(entropy(normalize(EQUAL_BARS)))).toBeCloseTo(5, 10);
    expect(fairChoices(0)).toBe(1);
  });

  it("rises all the way along the shape slider", () => {
    const values = Array.from({ length: 21 }, (_, index) => entropy(shapeShares(index * 5)));
    for (let index = 1; index < values.length; index += 1) expect(values[index]).toBeGreaterThan(values[index - 1]);
    expect(shapeShares(0)).toEqual([1, 0, 0, 0, 0]);
    expect(shapeShares(50)).toEqual(sky);
    for (const share of shapeShares(100)) expect(share).toBeCloseTo(0.2, 12);
    expect(sum(shapeShares(35))).toBeCloseTo(1, 12);
    expect(shapeShares(-20)).toEqual(shapeShares(0));
  });

  it("never lets the bill fall below the entropy, for any model on the 1% grid edge or inside it", () => {
    const random = mulberry32(99);
    for (let trial = 0; trial < 500; trial += 1) {
      const q = normalize(Array.from({ length: 5 }, () => 0.01 + random()));
      expect(crossEntropy(sky, q)).toBeGreaterThanOrEqual(entropy(sky) - 1e-12);
      expect(klDivergence(sky, q)).toBeGreaterThanOrEqual(0);
    }
    expect(crossEntropy(sky, sky)).toBeCloseTo(entropy(sky), 12);
    expect(klDivergence(sky, sky)).toBe(0);
  });

  it("charges a confident wrong model far more than a know-nothing one", () => {
    const wrong = normalize([1, 1, 1, 1, 96]);
    expect(klDivergence(sky, wrong)).toBeGreaterThan(klDivergence(sky, normalize(EQUAL_BARS)) * 5);
  });
});

describe("softmax, temperature and the sigmoid", () => {
  it("gives positive shares that add to one", () => {
    const result = softmax(SKY_SCORES);
    expect(sum(result.shares)).toBeCloseTo(1, 12);
    expect(result.total).toBeCloseTo(sum(result.powers), 12);
    result.shares.forEach((share, index) => expect(share).toBeCloseTo(result.powers[index] / result.total, 12));
  });

  it("gives the sky bars to the nearest percent from the authored scores", () => {
    expect(softmax(SKY_SCORES).shares.map((share) => Math.round(share * 100))).toEqual([...SKY_BARS]);
  });

  it("does not move when the same amount is added to every score", () => {
    const base = softmax(SKY_SCORES).shares;
    for (const shift of [-3, -1, 1, 4]) {
      softmax(SKY_SCORES.map((score) => score + shift)).shares.forEach((share, index) => expect(share).toBeCloseTo(base[index], 12));
    }
  });

  it("sharpens below temperature 1 and flattens above it, keeping the order", () => {
    const sharp = softmax(SKY_SCORES, 0.5).shares;
    const plain = softmax(SKY_SCORES, 1).shares;
    const flat = softmax(SKY_SCORES, 2).shares;
    expect(sharp[0]).toBeGreaterThan(plain[0]);
    expect(flat[0]).toBeLessThan(plain[0]);
    expect(entropy(sharp)).toBeLessThan(entropy(plain));
    expect(entropy(flat)).toBeGreaterThan(entropy(plain));
    for (const shares of [sharp, plain, flat]) expect([...shares].sort((a, b) => b - a)).toEqual(shares);
  });

  it("survives huge scores without overflowing", () => {
    const shares = softmax([800, 799, -800, 0, 5]).shares;
    expect(shares.every(Number.isFinite)).toBe(true);
    expect(sum(shares)).toBeCloseTo(1, 12);
  });

  it("is the sigmoid of the score gap when there are only two words", () => {
    for (const [a, b] of [[2, 1.3], [0, 0], [-3, 4], [5.5, -0.2]]) {
      expect(softmax([a, b]).shares[0]).toBeCloseTo(sigmoid(a - b), 12);
    }
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(3) + sigmoid(-3)).toBeCloseTo(1, 12);
    expect(sigmoid(1000)).toBe(1);
    expect(sigmoid(-1000)).toBeGreaterThanOrEqual(0);
  });
});

describe("the challenge", () => {
  it("draws 40 samples per round and never leaves a word uncounted", () => {
    CHALLENGE_ROUNDS.forEach((round, index) => {
      const counts = challengeTally(index);
      expect(sum(counts)).toBe(CHALLENGE_DRAWS);
      expect(Math.min(...counts)).toBeGreaterThan(0);
      expect(isBars([...round.truth])).toBe(true);
    });
  });

  it("is won by counting and dividing, whichever way the half percents are rounded", () => {
    CHALLENGE_ROUNDS.forEach((_, index) => {
      const counts = challengeTally(index);
      expect(challengeScore(index, roundToTotal(counts, 100)).gap).toBeLessThan(CHALLENGE_GOAL / 2);
      // A reader may round 7.5% down where the lab would round it up; try nudging every pair by a point.
      const base = roundToTotal(counts, 100);
      for (let up = 0; up < base.length; up += 1) {
        for (let down = 0; down < base.length; down += 1) {
          if (up === down || base[down] <= MIN_BAR) continue;
          const nudged = base.map((bar, at) => bar + (at === up ? 1 : at === down ? -1 : 0));
          expect(challengeScore(index, nudged).gap).toBeLessThan(CHALLENGE_GOAL);
        }
      }
    });
  });

  it("is not won by the starting bars, by the sky bars, or by betting everything on the top word", () => {
    CHALLENGE_ROUNDS.forEach((_, index) => {
      const counts = challengeTally(index);
      const top = counts.indexOf(Math.max(...counts));
      expect(challengeScore(index, EQUAL_BARS).gap).toBeGreaterThan(CHALLENGE_GOAL * 3);
      expect(challengeScore(index, SKY_BARS).gap).toBeGreaterThan(CHALLENGE_GOAL * 3);
      expect(challengeScore(index, setBar(EQUAL_BARS, top, MAX_BAR)).gap).toBeGreaterThan(CHALLENGE_GOAL * 3);
    });
  });

  it("cannot be scored below zero, and scores zero only at the truth", () => {
    CHALLENGE_ROUNDS.forEach((round, index) => {
      expect(challengeScore(index, round.truth).gap).toBe(0);
      expect(challengeScore(index, EQUAL_BARS).bill).toBeGreaterThan(challengeScore(index, round.truth).best);
    });
  });
});
