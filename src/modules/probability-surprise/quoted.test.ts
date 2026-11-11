import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  CHALLENGE_DRAWS,
  CHALLENGE_GOAL,
  CHALLENGE_ROUNDS,
  EQUAL_BARS,
  PROMPTS,
  SKY_BARS,
  SKY_SCORES,
  TEST_COUNT,
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
  klDivergence,
  likelihoodOf,
  mulberry32,
  normalize,
  pick,
  pileStats,
  posterior,
  priorFor,
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
import { DEFAULT_LUCK } from "./state";

/**
 * Every number the three lessons, the checkpoint and the card explanations quote, pinned to the
 * arithmetic the lab runs and to the text the lab prints at the setting the sentence names.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const lesson = (mode: string) => readFileSync(path.join(here, "content", `${mode}.mdx`), "utf8");
const render = (step: number, patch: Record<string, string | number> = {}) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep: step,
      mode: "plain" as const,
      narrate: () => undefined,
    }),
  ).replace(/<[^>]*>/g, "\u0001");
const sky = normalize(SKY_BARS);
const fixed = (value: number, digits: number) => value.toFixed(digits);

describe("Bars that add to one", () => {
  it("starts at 50, 25, 12, 8 and 5, which add to one whole", () => {
    expect([...SKY_BARS]).toEqual([50, 25, 12, 8, 5]);
    expect(sum(sky)).toBeCloseTo(1, 12);
    const text = render(0);
    expect(text).toContain("0.50 + 0.25 + 0.12 + 0.08 + 0.05");
    expect(text).toContain("= 1.00");
    expect(text).toContain("P(blue) = 0.50");
  });

  it("leaves 20% for the others when blue goes to 80%, and grey keeps half of it", () => {
    const next = setBar(SKY_BARS, 0, 80);
    expect(next).toEqual([80, 10, 5, 3, 2]);
    expect(sum(next.slice(1))).toBe(20);
    expect(SKY_BARS[1] / sum(SKY_BARS.slice(1))).toBe(0.5);
    expect(next[1] / sum(next.slice(1))).toBe(0.5);
  });

  it("reads 50% as 50 of 100 stories in the counts view", () => {
    expect(render(0, { view: "counts" })).toContain("50 of 100");
  });
});

describe("Draw from the bars", () => {
  const draws = drawSamples(sky, 1000, DEFAULT_LUCK);

  it("draws grey first, although blue is twice as likely", () => {
    expect(WORDS[draws[0]]).toBe("grey");
    expect(sky[0] / sky[1]).toBe(2);
  });

  it("tallies 7, 2, 0, 1, 0 after 10 draws, a gap of 20.0 points", () => {
    const counts = tally(draws.slice(0, 10));
    expect(counts).toEqual([7, 2, 0, 1, 0]);
    expect(fixed(biggestGap(counts, sky) * 100, 1)).toBe("20.0");
    const text = render(1, { draws: 10 });
    expect(text).toContain("20.0 points");
    expect(text).toContain("70%");
  });

  it("closes the gap to 9.0 points at 100 draws and 0.9 at 1,000, with blue at 507", () => {
    expect(fixed(biggestGap(tally(draws.slice(0, 100)), sky) * 100, 1)).toBe("9.0");
    const counts = tally(draws);
    expect(counts[0]).toBe(507);
    expect(fixed(biggestGap(counts, sky) * 100, 1)).toBe("0.9");
    expect(render(1, { draws: 100 })).toContain("9.0 points");
    const text = render(1, { draws: 1000 });
    expect(text).toContain("0.9 points");
    expect(text).toContain("50.7%");
  });

  it("gives a different tally of ten with new luck", () => {
    expect(tally(drawSamples(sky, 10, DEFAULT_LUCK + 1))).not.toEqual(tally(draws.slice(0, 10)));
  });
});

describe("Averages pile up in a bell", () => {
  const points = [1, 0, 0, 0, 0];

  it("gives one answer a mean of 0.50, a variance of 0.25 and a standard deviation of 0.5", () => {
    expect(expectedValue(sky, points)).toBeCloseTo(0.5, 12);
    expect(variance(sky, points)).toBeCloseTo(0.25, 12);
    expect(Math.sqrt(variance(sky, points))).toBeCloseTo(0.5, 12);
    const text = render(2);
    expect(text).toContain("0.2500");
    expect(text).toContain("the mean (expected value)");
  });

  it("piles 2,000 tests of 25 questions around 0.50 with a standard deviation of 0.101", () => {
    expect(TEST_COUNT).toBe(2000);
    const stats = pileStats(simulateTests(0.5, 25));
    expect(fixed(stats.mean, 2)).toBe("0.50");
    expect(fixed(stats.sd, 3)).toBe("0.101");
    const text = render(2);
    expect(text).toContain("0.101");
    expect(text).toContain("0.2500 ÷ 25 = 0.0100");
    expect(text).toContain("0.100");
    expect(text).toContain("About 68% of the area");
  });

  it("halves the standard deviation to 0.050 at 100 questions and to 0.025 at 400", () => {
    expect(fixed(pileStats(simulateTests(0.5, 100)).sd, 3)).toBe("0.050");
    expect(fixed(pileStats(simulateTests(0.5, 400)).sd, 3)).toBe("0.025");
    expect(Math.sqrt(0.25 / 25)).toBeCloseTo(0.1, 12);
    expect(render(2, { testSize: 100 })).toContain("0.050");
    expect(render(2, { testSize: 400 })).toContain("0.025");
  });
});

describe("Context changes the bars", () => {
  it("counts 200 stories, with 35 of the 50 grass stories saying green", () => {
    expect(sum(PROMPTS.flatMap((prompt) => prompt.counts))).toBe(200);
    expect(PROMPTS[1].counts[4]).toBe(35);
    expect(sum(PROMPTS[1].counts)).toBe(50);
    expect(conditional(1)[4]).toBeCloseTo(0.7, 12);
    expect(conditional(1)[0]).toBeCloseTo(0.1, 12);
    const text = render(3, { prompt: 1 });
    expect(text).toContain("P(green | the grass is) = 35 ÷ 50");
    expect(text).toContain("= 0.70");
  });

  it("gives blue 20 of the 50 sea stories, 0.40", () => {
    expect(PROMPTS[2].counts[0]).toBe(20);
    expect(conditional(2)[0]).toBeCloseTo(0.4, 12);
    expect(render(3, { prompt: 2 })).toContain("P(blue | the sea is) = 20 ÷ 50");
  });

  it("has 47 stories that say green, the divisor of the reverse question in the checkpoint", () => {
    expect(sum(PROMPTS.map((prompt) => prompt.counts[4]))).toBe(47);
    expect(fixed(35 / 47, 2)).toBe("0.74");
    expect(fixed(35 / 200, 3)).toBe("0.175");
  });
});

describe("Read the table backwards", () => {
  it("finds 75 stories that say blue, 50 of them from the sky: about 67%", () => {
    const column = PROMPTS.map((prompt) => prompt.counts[0]);
    expect(column).toEqual([50, 5, 20]);
    expect(sum(column)).toBe(75);
    expect(Math.round((50 / 75) * 100)).toBe(67);
  });

  it("multiplies priors 0.50, 0.25, 0.25 by likelihoods 0.50, 0.10, 0.40 to get 0.25, 0.025, 0.10", () => {
    expect(priorFor(0.5)).toEqual([0.5, 0.25, 0.25]);
    expect(likelihoodOf(0)).toEqual([0.5, 0.1, 0.4]);
    const result = posterior(priorFor(0.5), likelihoodOf(0));
    expect(result.joint.map((value) => +value.toFixed(6))).toEqual([0.25, 0.025, 0.1]);
    expect(result.evidence).toBeCloseTo(0.375, 12);
    expect(fixed(result.posterior[0], 3)).toBe("0.667");
    const text = render(3);
    expect(text).toContain("0.3750");
    expect(text).toContain("66.7%");
    expect(text).toContain("= 0.667");
  });

  it("drops the sky's posterior from 67% to 18% at a prior of 10%, with the sea at about 65%", () => {
    const result = posterior(priorFor(0.1), likelihoodOf(0));
    expect(Math.round(result.posterior[0] * 100)).toBe(18);
    expect(fixed(result.posterior[2] * 100, 1)).toBe("65.5");
    const text = render(3, { prior: 10 });
    expect(text).toContain("18.2%");
    expect(text).toContain("65.5%");
  });
});

describe("Surprise in bits", () => {
  it("gives the sky bars 1.00, 2.00, 3.06, 3.64 and 4.32 bits", () => {
    expect(sky.map((share) => fixed(surprise(share), 2))).toEqual(["1.00", "2.00", "3.06", "3.64", "4.32"]);
    const text = render(4);
    for (const value of ["1.00 bits", "2.00 bits", "3.06 bits", "3.64 bits", "4.32 bits"]) expect(text).toContain(value);
    expect(text).toContain("1 in 4");
    expect(text).toContain("1 in 8");
  });

  it("moves blue from 1 bit to 2 bits when its bar is dragged from 50% to 25%", () => {
    const next = setBar(SKY_BARS, 0, 25);
    expect(surprise(next[0] / 100)).toBe(2);
    expect(surprise(SKY_BARS[0] / 100)).toBe(1);
    // The lesson then asks for Reset the bars, which the card offers once a bar has moved.
    expect(render(4, { bars: next.join(",") })).toContain("Reset the bars");
    expect(setBar(next, 0, 50)).toEqual([...SKY_BARS]);
  });

  it("keeps the two-word numbers the lesson quotes only at temperature 1, which Reset scores restores", () => {
    expect(render(6, { temperature: 2 })).toContain("Reset scores");
    expect(fixed(sigmoid(0.7 / 2) * 100, 1)).not.toBe("66.8");
  });

  it("multiplies 0.50 by 0.25 to 0.125 and adds 1.00 and 2.00 to 3.00 bits", () => {
    expect(sky[0] * sky[1]).toBe(0.125);
    expect(surprise(sky[0]) + surprise(sky[1])).toBe(3);
    expect(surprise(0.125)).toBe(3);
    const text = render(4);
    expect(text).toContain("0.50 × 0.25 = 0.1250");
    expect(text).toContain("= 3.00 bits");
  });

  it("puts a 1% chance at 6.64 bits, 20% at 2.32 and 10% at 3.32, and a bit at about 0.69 nats", () => {
    expect(fixed(surprise(0.01), 2)).toBe("6.64");
    expect(fixed(surprise(0.2), 2)).toBe("2.32");
    expect(fixed(surprise(0.1), 2)).toBe("3.32");
    expect(fixed(Math.log(2), 2)).toBe("0.69");
  });
});

describe("Average surprise", () => {
  it("averages the sky bars' surprises to 1.87 bits, a fair choice among 3.67 words", () => {
    expect(fixed(entropy(sky), 2)).toBe("1.87");
    expect(fixed(fairChoices(entropy(sky)), 2)).toBe("3.67");
    const text = render(4);
    expect(text).toContain("0.50×1.00 + 0.25×2.00 + 0.12×3.06 + 0.08×3.64 + 0.05×4.32");
    expect(text).toContain("1.87 bits");
    expect(text).toContain("3.67 words");
  });

  it("reads 0.00 bits for a sure thing and 2.32 bits for five equal bars", () => {
    expect(entropy(shapeShares(0))).toBe(0);
    expect(fairChoices(entropy(shapeShares(0)))).toBe(1);
    expect(fixed(entropy(shapeShares(100)), 2)).toBe("2.32");
    expect(fixed(fairChoices(entropy(shapeShares(100))), 2)).toBe("5.00");
    expect(render(4, { shape: 0 })).toContain("0.00 bits");
    const equal = render(4, { shape: 100 });
    expect(equal).toContain("2.32 bits");
    expect(equal).toContain("5.00 words");
  });
});

describe("Your guess against what happens", () => {
  it("bills a know-nothing model 2.32 bits against a best possible 1.87, wasting 0.45", () => {
    const q = normalize(EQUAL_BARS);
    expect(fixed(crossEntropy(sky, q), 2)).toBe("2.32");
    expect(fixed(klDivergence(sky, q), 2)).toBe("0.45");
    const text = render(5);
    expect(text).toContain("costs 2.32 bits");
    expect(text).toContain("2.32 bits");
    expect(text).toContain("1.87 bits");
    expect(text).toContain("0.45 bits");
  });

  it("bills 2.03 bits when the model puts 60% on blue and 10% on each other word", () => {
    const bars = setBar(EQUAL_BARS, 0, 60);
    expect(bars).toEqual([60, 10, 10, 10, 10]);
    const q = normalize(bars);
    expect(fixed(surprise(q[0]), 2)).toBe("0.74");
    expect(fixed(surprise(q[1]), 2)).toBe("3.32");
    expect(fixed(crossEntropy(sky, q), 2)).toBe("2.03");
    expect(sum(sky.slice(1))).toBeCloseTo(0.5, 12);
    expect(render(5, { guess: "60,10,10,10,10" })).toContain("2.03 bits");
  });

  it("bills a confidently wrong model 6.31 bits, 4.44 of them wasted", () => {
    const q = normalize([1, 1, 1, 1, 96]);
    expect(fixed(crossEntropy(sky, q), 2)).toBe("6.31");
    expect(fixed(klDivergence(sky, q), 2)).toBe("4.44");
    const text = render(5, { guess: "1,1,1,1,96" });
    expect(text).toContain("6.31 bits");
    expect(text).toContain("4.44 bits");
    expect(text).toContain("costs 6.64 bits");
  });

  it("wastes 0.00 bits and stops at 1.87 when the model matches the truth", () => {
    expect(klDivergence(sky, sky)).toBe(0);
    const text = render(5, { guess: "50,25,12,8,5" });
    expect(text).toContain("0.00 bits");
    expect(text).toContain("1.87 bits");
  });
});

describe("From scores to bars", () => {
  it("raises e to 2.0, 1.3, 0.6, 0.2 and −0.3 to get 7.39, 3.67, 1.82, 1.22 and 0.74, total 14.84", () => {
    expect([...SKY_SCORES]).toEqual([2, 1.3, 0.6, 0.2, -0.3]);
    const result = softmax(SKY_SCORES);
    expect(result.powers.map((value) => fixed(value, 2))).toEqual(["7.39", "3.67", "1.82", "1.22", "0.74"]);
    expect(fixed(result.total, 2)).toBe("14.84");
    expect(fixed(result.shares[0], 3)).toBe("0.498");
    expect(fixed(Math.E, 3)).toBe("2.718");
    const text = render(6);
    expect(text).toContain("7.39 + 3.67 + 1.82 + 1.22 + 0.74");
    expect(text).toContain("14.84");
    expect(text).toContain("= 0.498");
  });

  it("gives the sky bars back to the nearest percent", () => {
    expect(softmax(SKY_SCORES).shares.map((share) => Math.round(share * 100))).toEqual([50, 25, 12, 8, 5]);
  });

  it("grows the total to 40.35 when 1 is added to every score, without moving a bar", () => {
    const shifted = SKY_SCORES.map((score) => +(score + 1).toFixed(1));
    expect(shifted).toEqual([3, 2.3, 1.6, 1.2, 0.7]);
    expect(fixed(softmax(shifted).total, 2)).toBe("40.35");
    softmax(shifted).shares.forEach((share, index) => expect(share).toBeCloseTo(softmax(SKY_SCORES).shares[index], 12));
    expect(render(6, { scores: "3,2.3,1.6,1.2,0.7" })).toContain("40.35");
  });

  it("puts blue at 74% at temperature 0.5, and blue at 34% with green at 11% at 2.0", () => {
    expect(Math.round(softmax(SKY_SCORES, 0.5).shares[0] * 100)).toBe(74);
    const flat = softmax(SKY_SCORES, 2).shares;
    expect(Math.round(flat[0] * 100)).toBe(34);
    expect(Math.round(flat[4] * 100)).toBe(11);
    expect(render(6, { temperature: 0.5 })).toContain("74%");
    const text = render(6, { temperature: 2 });
    expect(text).toContain("34%");
    expect(text).toContain("11%");
  });

  it("gives blue 66.8% of the pair with grey, at a score gap of 0.7, by softmax and by sigmoid", () => {
    expect(SKY_SCORES[0] - SKY_SCORES[1]).toBeCloseTo(0.7, 12);
    expect(fixed(sigmoid(0.7) * 100, 1)).toBe("66.8");
    expect(fixed(softmax([SKY_SCORES[0], SKY_SCORES[1]]).shares[0] * 100, 1)).toBe("66.8");
    expect(sigmoid(0)).toBe(0.5);
    const text = render(6);
    expect(text).toContain("+0.70");
    expect(text.split("66.8%").length - 1).toBeGreaterThanOrEqual(2);
  });
});

describe("the challenge", () => {
  it("shows a tally of 3, 4, 20, 11 and 2 for the night is, from 40 draws", () => {
    expect(CHALLENGE_ROUNDS[0].text).toBe("the night is");
    expect(CHALLENGE_DRAWS).toBe(40);
    expect(challengeTally(0)).toEqual([3, 4, 20, 11, 2]);
    expect((11 / 40) * 100).toBeCloseTo(27.5, 12);
    expect((20 / 40) * 100).toBe(50);
  });

  it("wastes 0.583 bits at the starting bars and 0.010 at 7, 10, 50, 28 and 5", () => {
    expect(fixed(challengeScore(0, EQUAL_BARS).gap, 3)).toBe("0.583");
    expect(fixed(challengeScore(0, [7, 10, 50, 28, 5]).gap, 3)).toBe("0.010");
    expect(CHALLENGE_GOAL).toBe(0.05);
    expect(render(7, { scored: 1, tries: 1 })).toContain("0.583 bits wasted");
    expect(render(7, { answer: "7,10,50,28,5", scored: 1, tries: 2 })).toContain("0.010 bits wasted");
  });

  it("also meets the goal with 8, 10, 50, 27 and 5, the other way of rounding the two half percents", () => {
    expect(challengeScore(0, [8, 10, 50, 27, 5]).gap).toBeLessThan(CHALLENGE_GOAL);
  });

  it("hides a truth with dark at 55%, not the tally's 50%", () => {
    expect(CHALLENGE_ROUNDS[0].truth[2]).toBe(55);
  });

  it("is typically missed by counting 40 draws: the expected waste is about 0.07 bits", () => {
    // Asymptotic value for k outcomes and n draws: (k − 1) ÷ (2 n ln 2).
    expect(fixed((WORDS.length - 1) / (2 * CHALLENGE_DRAWS * Math.log(2)), 2)).toBe("0.07");
    // And by simulation on round one's hidden bars, with half a count added so no share is zero.
    const truth = normalize(CHALLENGE_ROUNDS[0].truth);
    const random = mulberry32(2026);
    let total = 0;
    const runs = 4000;
    for (let run = 0; run < runs; run += 1) {
      const counts = [0, 0, 0, 0, 0];
      for (let draw = 0; draw < CHALLENGE_DRAWS; draw += 1) counts[pick(truth, random())] += 1;
      total += klDivergence(truth, normalize(counts.map((count) => count + 0.5)));
    }
    expect(total / runs).toBeGreaterThan(CHALLENGE_GOAL);
    expect(total / runs).toBeLessThan(0.1);
  });
});

describe("the lessons", () => {
  const figures = [
    "0.50 + 0.25 + 0.12 + 0.08 + 0.05 = 1.00",
    "20.0 points",
    "9.0 points",
    "0.9 points",
    "507",
    "50.7%",
    "0.101",
    "0.050",
    "0.025",
    "35 ÷ 50 = 0.70",
    "20 ÷ 50 = 0.40",
    "50 ÷ 75",
    "0.375",
    "18%",
    "3.06 bits",
    "4.32 bits",
    "0.125",
    "1.87 bits",
    "2.32 bits",
    "3.67",
    "0.45 bits",
    "2.03",
    "6.31 bits",
    "4.44",
    "14.84",
    "40.35",
    "74%",
    "34%",
    "66.8%",
    "0.583",
    "0.010",
    "55%",
  ];

  it.each(["simple", "plain", "standard"])("%s quotes the same figures as the other modes", (mode) => {
    const text = lesson(mode).replace(/\*\*|`/g, "");
    expect(figures.filter((figure) => !text.includes(figure)), `${mode} does not quote these figures`).toEqual([]);
  });

  it("names every later lab the same way in all three modes", () => {
    const labs = [
      "Next-token prediction",
      "Loss & gradient descent",
      "Train one from scratch",
      "How we know a model works",
      "Evaluation",
      "Classical machine learning",
      "Distillation & pruning",
      "Post-training",
      "Reinforcement learning foundations",
    ];
    for (const mode of ["simple", "plain", "standard"]) {
      const text = lesson(mode);
      expect(labs.filter((lab) => !text.includes(`*${lab}*`)), `${mode} does not name these labs`).toEqual([]);
    }
  });
});
