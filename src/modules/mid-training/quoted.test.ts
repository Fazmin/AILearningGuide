import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TINY_VOCAB_SIZE, UNIFORM_CROSS_ENTROPY, type ModuleState } from "@app/module-sdk";
import { ARITHMETIC, DOMAINS, pairCount } from "./corpora";
import definition from "./module";
import {
  arithmeticSweep,
  BATCH_SIZE,
  BUDGET_PAIRS,
  CHECKPOINTS,
  DECAY_FRACTION,
  gainPerEpoch,
  LEARNING_RATE,
  PRESETS,
  runEvenSpread,
  runMixture,
  SEED,
  sharesOf,
  STEPS,
  SWITCH_PROGRESS,
  SWITCH_STEP,
  TARGET_MIX_SHARE,
  WARMUP_FRACTION,
  WEIGHT_MAX,
  WEIGHT_MIN,
} from "./mixture";
import { ANSWER_WORDS, answerProbe } from "./probe";
import {
  BUDGET_MAX,
  BUDGET_MIN,
  BUDGET_STEP,
  DEFAULT_BUDGET,
  REPEAT_CHECKPOINTS,
  repetitionSweep,
  runRepetition,
  SWEEP_UNIQUE,
  UNIQUE_MAX,
  UNIQUE_MIN,
  uniqueText,
} from "./repetition";
import { initialState } from "./state";

/**
 * Pins every measured or configured number that content/standard.mdx, content/plain.mdx, content/simple.mdx
 * and card-info.ts quote. Every run here is the lab's own function with the lab's own settings, so a
 * change to the recipe, the corpora or the SDK trainer fails this file before it reaches a lesson.
 */
const f3 = (value: number) => Number(value.toFixed(3));
const f2 = (value: number) => Number(value.toFixed(2));
const f1 = (value: number) => Number(value.toFixed(1));

const GENERAL_HEAVY = [3, 3, 3, 1];
const UNIFORM = [1, 1, 1, 1];
const generalHeavy = runMixture(GENERAL_HEAVY, "same", "wsd");
const uniform = runMixture(UNIFORM, "same", "wsd");

describe("the recipe and the corpora", () => {
  it("matches the budget, the optimizer, the schedule and the seed", () => {
    // standard "600 steps, 9,600 character pairs", "SGD at rate 3, 16 pairs a batch, seed 7"; plain/simple "600 steps
    // of 16 pairs", "9,600 pairs"; card-info "600 steps of 16 pairs", "plain SGD at rate 3"
    expect([STEPS, BATCH_SIZE, BUDGET_PAIRS, LEARNING_RATE, SEED]).toEqual([600, 16, 9600, 3, 7]);
    // "the first 5% of steps", "the last 20%, from step 480", "the last 120 steps", "progress 0.8"
    expect([WARMUP_FRACTION, DECAY_FRACTION, SWITCH_STEP, SWITCH_PROGRESS]).toEqual([0.05, 0.2, 480, 0.8]);
    expect(STEPS - SWITCH_STEP).toBe(120);
    // card-info "the stable shares for 80% of the steps plus the late shares for 20%"
    expect(SWITCH_STEP / STEPS).toBe(0.8);
    // "the full 900-weight table", "ln 30 = 3.401"
    expect(TINY_VOCAB_SIZE ** 2).toBe(900);
    expect(TINY_VOCAB_SIZE).toBe(30);
    expect(f3(UNIFORM_CROSS_ENTROPY)).toBe(3.401);
    // card-info "every 30 steps from step 90", "21 checkpoints"
    expect(generalHeavy.curve[1].step).toBe(30);
    expect(CHECKPOINTS).toBe(21);
    expect(REPEAT_CHECKPOINTS).toBe(21);
    // "Target mix makes half of every late batch Arithmetic"
    expect(TARGET_MIX_SHARE).toBe(0.5);
  });

  it("matches the four domains, their sizes and their held-out sentences", () => {
    // "Harbor (8 sentences), Recipes (8) and Proverbs (10)", "160 facts", "2 of Harbor's 8, 2 of Recipes' 8, 3 of
    // Proverbs' 10 and 40 of the 160", "Harbor's six training sentences ... Proverbs' seven ... Arithmetic's 120"
    expect(DOMAINS.map((domain) => domain.sentences.length)).toEqual([8, 8, 10, 160]);
    expect(DOMAINS.map((domain) => domain.heldOut.length)).toEqual([2, 2, 3, 40]);
    expect(DOMAINS.map((domain) => domain.train.length)).toEqual([6, 6, 7, 120]);
    // "378 pairs ... 292 ... 195 ... 3,417"
    expect(DOMAINS.map((domain) => pairCount(domain.trainText))).toEqual([378, 292, 195, 3417]);
    // "Arithmetic is 80% of the training text"
    const total = DOMAINS.reduce((sum, domain) => sum + pairCount(domain.trainText), 0);
    expect(Math.round((pairCount(ARITHMETIC.trainText) / total) * 100)).toBe(80);
    // "three templates", "four plus eight is twelve.", numbers spelled out from zero to twenty, "21 answers"
    expect(ARITHMETIC.sentences.every((sentence) => /^\w+ (plus|minus|times) \w+ is \w+\.$/.test(sentence))).toBe(true);
    expect(ANSWER_WORDS).toHaveLength(21);
    expect([ANSWER_WORDS[0], ANSWER_WORDS[20]]).toEqual(["zero", "twenty"]);
  });

  it("matches the controls' ranges and the defaults", () => {
    // "four sliders ... each from 0 to 8", "Uniform (1, 1, 1, 1) and General-heavy (3, 3, 3, 1)"
    expect([WEIGHT_MIN, WEIGHT_MAX]).toEqual([0, 8]);
    expect(PRESETS.map((preset) => [preset.label, [...preset.weights]])).toEqual([
      ["Uniform", [1, 1, 1, 1]],
      ["General-heavy", [3, 3, 3, 1]],
    ]);
    expect([initialState.harborWeight, initialState.recipesWeight, initialState.proverbsWeight, initialState.arithmeticWeight]).toEqual(
      GENERAL_HEAVY,
    );
    // "30% each ... and 10% for Arithmetic", "a quarter of the batches", "3, 3, 3 and 1 and 6, 6, 6 and 2"
    expect(sharesOf(GENERAL_HEAVY)).toEqual([0.3, 0.3, 0.3, 0.1]);
    expect(sharesOf(UNIFORM)).toEqual([0.25, 0.25, 0.25, 0.25]);
    expect(sharesOf([6, 6, 6, 2])).toEqual(sharesOf(GENERAL_HEAVY));
    // "Budget from 2,000 to 16,000 pairs in steps of 2,000", "Unique sentences from 2 to 120", "keep Budget at 8,000"
    expect([BUDGET_MIN, BUDGET_MAX, BUDGET_STEP, DEFAULT_BUDGET]).toEqual([2000, 16000, 2000, 8000]);
    expect([UNIQUE_MIN, UNIQUE_MAX]).toEqual([2, 120]);
    expect(initialState.repeatBudget).toBe(8000);
    expect(initialState.uniqueSentences).toBe(120);
    // "Nine runs", "weights 0 to 8"
    expect(arithmeticSweep(GENERAL_HEAVY, "same", "wsd")).toHaveLength(9);
  });
});

describe("Step 1: the Data mixture card", () => {
  it("matches Uniform's epochs and held-out losses", () => {
    // "6.16 for Harbor, 8.34 for Recipes, 12.48 for Proverbs and 0.70 for Arithmetic"
    expect(uniform.domains.map((domain) => f2(domain.epochs))).toEqual([6.16, 8.34, 12.48, 0.7]);
    // "2.197, 2.498, 2.400 and 1.967, and General 2.365"
    expect(uniform.domains.map((domain) => f3(domain.held))).toEqual([2.197, 2.498, 2.4, 1.967]);
    expect(f3(uniform.general)).toBe(2.365);
    // "past 12 passes" / "more than 12 passes"
    expect(uniform.domains[2].epochs).toBeGreaterThan(12);
  });

  it("matches General-heavy's epochs and held-out losses", () => {
    // "Harbor 7.42, Recipes 10.21, Proverbs 14.72 and Arithmetic 0.28"
    expect(generalHeavy.domains.map((domain) => f2(domain.epochs))).toEqual([7.42, 10.21, 14.72, 0.28]);
    // "falls to 2.097", "2.427", "moves from 2.400 to 2.408", "Arithmetic pays, from 1.967 to 2.348", "2.365 to 2.311"
    expect(generalHeavy.domains.map((domain) => f3(domain.held))).toEqual([2.097, 2.427, 2.408, 2.348]);
    expect(f3(generalHeavy.general)).toBe(2.311);
    // Harbor and Recipes gain, Proverbs does not, Arithmetic pays, General improves
    expect(generalHeavy.domains[0].held).toBeLessThan(uniform.domains[0].held);
    expect(generalHeavy.domains[1].held).toBeLessThan(uniform.domains[1].held);
    expect(generalHeavy.domains[2].held).toBeGreaterThan(uniform.domains[2].held);
    expect(generalHeavy.arithmetic).toBeGreaterThan(uniform.arithmetic);
    expect(generalHeavy.general).toBeLessThan(uniform.general);
  });

  it("spends every pair of the budget", () => {
    // "every bar that falls is paid for by one that rises": the budget is fixed
    for (const run of [uniform, generalHeavy]) expect(run.domains.reduce((sum, domain) => sum + domain.seen, 0)).toBe(9600);
  });
});

describe("Step 2: the Arithmetic weight sweep", () => {
  const sweep = arithmeticSweep(GENERAL_HEAVY, "same", "wsd");
  const gains = gainPerEpoch(sweep);

  it("matches the ends of the sweep", () => {
    // "falls from 2.832 to 1.706 while General held-out rises from 2.320 to 2.515"
    expect([f3(sweep[0].run!.arithmetic), f3(sweep[8].run!.arithmetic)]).toEqual([2.832, 1.706]);
    expect([f3(sweep[0].run!.general), f3(sweep[8].run!.general)]).toEqual([2.32, 2.515]);
    // "Even weight 8 buys only 1.32 epochs"
    expect(f2(sweep[8].run!.domains[3].epochs)).toBe(1.32);
    // "Weight 0 still scores 2.832, below 3.401"
    expect(sweep[0].run!.arithmetic).toBeLessThan(UNIFORM_CROSS_ENTROPY);
    expect(sweep[0].run!.domains[3].seen).toBe(0);
  });

  it("matches the shrinking gain per extra epoch", () => {
    // "1.75 nats per epoch from weight 0 to 1, 1.04 from 1 to 2, 0.25 from 7 to 8"
    expect([f2(gains[1]!), f2(gains[2]!), f2(gains[8]!)]).toEqual([1.75, 1.04, 0.25]);
    for (let index = 2; index < gains.length; index += 1) expect(gains[index]!).toBeLessThan(gains[index - 1]!);
  });

  it("matches what General pays", () => {
    // card-info "from weight 1 to 8 it rises from 2.311 to 2.515", at every step up but the first
    expect(f3(sweep[1].run!.general)).toBe(2.311);
    for (let index = 2; index < sweep.length; index += 1) {
      expect(sweep[index].run!.general).toBeGreaterThan(sweep[index - 1].run!.general);
    }
    // The weight-1 row is the default run on the Data mixture card
    expect(sweep[1].run).toBe(generalHeavy);
  });

  it("matches the seed band at the default mixture", () => {
    // standard/plain/simple and card-info: "seeds 1 to 8 move Arithmetic held-out by up to 0.07 nats and General
    // held-out by about 0.01"
    const runs = [1, 2, 3, 4, 5, 6, 7, 8].map((seed) => runMixture(GENERAL_HEAVY, "same", "wsd", seed));
    const spread = (values: number[]) => Math.max(...values) - Math.min(...values);
    expect(f2(spread(runs.map((run) => run.arithmetic)))).toBe(0.07);
    expect(f2(spread(runs.map((run) => run.general)))).toBe(0.01);
  });
});

describe("Step 3: Repeated data at 8,000 pairs", () => {
  const bars = repetitionSweep(8000);

  it("matches every bar", () => {
    // "120 sentences: 2.3 epochs, held-out 1.388" ... "2: 135.6 epochs, 2.542"; card-info "120, 60, 30, 15, 8, 4 and 2"
    expect([...SWEEP_UNIQUE]).toEqual([120, 60, 30, 15, 8, 4, 2]);
    expect(bars.map((run) => [run.unique, f1(run.epochs), f3(run.held)])).toEqual([
      [120, 2.3, 1.388],
      [60, 4.7, 1.4],
      [30, 9.4, 1.458],
      [15, 18.8, 1.577],
      [8, 34.6, 1.731],
      [4, 70.8, 1.961],
      [2, 135.6, 2.542],
    ]);
    // "500 steps"
    expect(bars[0].steps).toBe(500);
  });

  it("matches the cost of repeating against all 120 sentences", () => {
    // "Versus 120 unique reads +0.011 for 60 sentences and +0.069 for 30": the metric subtracts unrounded losses
    expect(f3(bars[1].held - bars[0].held)).toBe(0.011);
    expect(f3(bars[2].held - bars[0].held)).toBe(0.069);
    // "so it can differ by 0.001 from the rounded bars": 1.400 − 1.388 is 0.012 and 1.458 − 1.388 is 0.070
    expect(f3(f3(bars[1].held) - f3(bars[0].held))).toBe(0.012);
    expect(f3(f3(bars[2].held) - f3(bars[0].held))).toBe(0.07);
  });

  it("matches the two-sentence run that bottoms out and climbs", () => {
    // "Lowest along the run reads 2.217 at 27.1 epochs", "ends at 2.542"; checkpoint "59 pairs of text, about 136
    // passes"
    const two = runRepetition(2, 8000);
    expect([f3(two.lowest.held), f1(two.lowest.epochs), f3(two.held)]).toEqual([2.217, 27.1, 2.542]);
    expect(pairCount(uniqueText(2))).toBe(59);
    expect(Math.round(two.epochs)).toBe(136);
    // "every pass after that point made the score worse": the curve rises from its lowest point to the end
    const after = two.curve.filter((point) => point.pairs >= two.lowest.pairs);
    for (let index = 1; index < after.length; index += 1) expect(after[index].held).toBeGreaterThan(after[index - 1].held);
    // "the solid line for all 120 sentences levels off at its lowest": it ends within 0.001 of its lowest point,
    // after the lowest point comes at its last-but-one checkpoint
    expect(bars[0].held - bars[0].lowest.held).toBeLessThan(0.001);
    expect(bars[0].lowest.pairs).toBeGreaterThanOrEqual(0.9 * 8000);
    // ...while the dashed two-sentence line "dips and then climbs back up", by more than 0.3 nats
    expect(two.held - two.lowest.held).toBeGreaterThan(0.3);
  });
});

describe("Steps 4 and 5: the late phase", () => {
  const mix = runMixture(GENERAL_HEAVY, "mix", "wsd");
  const mixEven = runEvenSpread(GENERAL_HEAVY, "mix", "wsd");
  const only = runMixture(GENERAL_HEAVY, "only", "wsd");
  const onlyEven = runEvenSpread(GENERAL_HEAVY, "only", "wsd");
  const row = (run: typeof mix) => [f3(run.arithmetic), f3(run.general), f3(run.forgetting), run.domains[3].seen];

  it("matches the three columns for Target mix", () => {
    // "Switched at decay: Arithmetic 1.911, General 2.398, Forgetting +0.060, 1,721 Arithmetic pairs"
    expect(row(mix)).toEqual([1.911, 2.398, 0.06, 1721]);
    // "Spread evenly: 2.110, 2.339, −0.027, 1,707 pairs"
    expect(row(mixEven)).toEqual([2.11, 2.339, -0.027, 1707]);
    // "No switch: 2.348, 2.311, −0.027, 943 pairs"
    expect(row(generalHeavy)).toEqual([2.348, 2.311, -0.027, 943]);
    // "end 0.199 nats apart on Arithmetic"
    expect(f3(mixEven.arithmetic - mix.arithmetic)).toBe(0.199);
    expect(f3(f3(mixEven.arithmetic) - f3(mix.arithmetic))).toBe(0.199);
    // "Arithmetic falls from 2.356 at step 480 to 1.911 in the last 120 steps"
    expect(mix.atSwitch.step).toBe(480);
    expect(f3(mix.atSwitch.held[3])).toBe(2.356);
  });

  it("matches Target only", () => {
    // "Arithmetic 1.645, General 2.691 and Forgetting +0.353", "Spread evenly reads 1.935 and 2.375"
    expect(row(only).slice(0, 3)).toEqual([1.645, 2.691, 0.353]);
    expect([f3(onlyEven.arithmetic), f3(onlyEven.general)]).toEqual([1.935, 2.375]);
    // "Target only buys 0.266 more nats on Arithmetic and costs 0.293 on General"
    expect(f3(mix.arithmetic - only.arithmetic)).toBe(0.266);
    expect(f3(only.general - mix.general)).toBe(0.293);
    expect(f3(f3(mix.arithmetic) - f3(only.arithmetic))).toBe(0.266);
    expect(f3(f3(only.general) - f3(mix.general))).toBe(0.293);
  });

  it("matches the constant-rate comparison", () => {
    // "Arithmetic ends at 1.785 ... General at 2.462 ... Forgetting doubles, from +0.060 to +0.127"
    const constant = runMixture(GENERAL_HEAVY, "mix", "constant");
    expect(row(constant).slice(0, 3)).toEqual([1.785, 2.462, 0.127]);
    expect(constant.arithmetic).toBeLessThan(mix.arithmetic);
    expect(constant.general).toBeGreaterThan(mix.general);
    expect(Math.round(constant.forgetting / mix.forgetting)).toBe(2);
    // "General falls by 0.027 over the last 20% of steps with the decay and by 0.022 at a constant rate"
    expect(f3(-generalHeavy.forgetting)).toBe(0.027);
    expect(f3(-runMixture(GENERAL_HEAVY, "same", "constant").forgetting)).toBe(0.022);
  });
});

describe("the Answer probe", () => {
  it("matches the default table: ten for every question, 0 of 40", () => {
    const probe = answerProbe(generalHeavy.weights);
    expect([probe.pick, probe.correct, probe.total, probe.distinctPicks]).toEqual(["ten", 0, 40, 1]);
    // "ten is never the answer to a held-out fact here"
    expect(probe.facts.some((fact) => fact.truth === "ten")).toBe(false);
    // card-info "the five likeliest answers all have three or four letters", shares adding to 100%
    for (const entry of probe.ranking.slice(0, 5)) expect(entry.word.length).toBeLessThanOrEqual(4);
    expect(probe.ranking.reduce((sum, entry) => sum + entry.share, 0)).toBeCloseTo(1, 12);
  });

  it("matches Arithmetic weight 0: one for every question, 3 of 40 by luck, with a worse loss", () => {
    const none = runMixture([3, 3, 3, 0], "same", "wsd");
    const probe = answerProbe(none.weights);
    expect([probe.pick, probe.correct]).toEqual(["one", 3]);
    expect(probe.facts.filter((fact) => fact.truth === "one")).toHaveLength(3);
    // "rises to 2.832", "from 2.832 to 2.348"
    expect(f3(none.arithmetic)).toBe(2.832);
  });

  it("gets nothing right however far the Arithmetic loss falls", () => {
    // checkpoint: "Arithmetic held-out falls from 2.832 to 1.706 ... 0 of 40"
    for (const [weights, late] of [
      [[3, 3, 3, 8], "same"],
      [[3, 3, 3, 1], "only"],
      [[0, 0, 0, 1], "same"],
    ] as const) {
      expect(answerProbe(runMixture(weights, late, "wsd").weights).correct).toBe(0);
    }
  });
});

describe("what the cards print", () => {
  it("matches the late-phase chart footnote and the default readouts", () => {
    const render = (state: ModuleState) =>
      renderToStaticMarkup(
        createElement(definition.Explore, {
          state,
          setState: () => undefined,
          currentStep: 0,
          mode: "standard" as const,
          narrate: () => undefined,
        }),
      );
    const html = render({ ...initialState, latePhase: "mix" });
    // card-info "every 30 steps from step 90"
    expect(html).toContain("Measured every 30 steps from step 90");
    // "Versus 120 unique reads +0.011 for 60 sentences"
    expect(render({ ...initialState, uniqueSentences: 60 })).toContain("<span>Versus 120 unique</span><strong>+0.011</strong>");
    // "Lowest along the run reads 2.217 at 27.1 epochs"
    expect(render({ ...initialState, uniqueSentences: 2 })).toContain("<strong>2.217 at 27.1 epochs</strong>");
    // "Right answers", "0 of 40", "Different answers" 1
    expect(render(initialState)).toContain("<span>Right answers</span><strong>0 of 40</strong>");
  });
});

describe("real-world figures", () => {
  /** Each figure the lessons quote from a paper, and words its reference note must contain. */
  const figures: Array<{ reference: string; note: string[] }> = [
    { reference: "The Llama 3 Herd of Models", note: ["50 percent", "25 percent", "17 percent", "8 percent", "40 million tokens", "24.0 percent", "6.4 percent", "405B", "40 billion tokens", "30 percent"] },
    { reference: "2 OLMo 2 Furious", note: ["5 to 10 percent", "Dolmino Mix 1124", "4 trillion tokens", "24.1 to 67.5"] },
    { reference: "Scaling Data-Constrained Language Models", note: ["4 epochs"] },
    { reference: "Scaling Laws and Interpretability of Learning from Repeated Data", note: ["0.1%", "100 times", "800-million"] },
    { reference: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining", note: ["6.5 points", "2.6 times"] },
    { reference: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", note: ["10 percent"] },
    { reference: "Does your data spark joy? Performance gains from domain upsampling at the end of training", note: ["7B", "1 trillion tokens", "6.90", "8.26", "6.17", "10 to 20 percent"] },
  ];

  it("appear in the notes of the references that back them", () => {
    const references = definition.references ?? [];
    for (const { reference, note } of figures) {
      const entry = references.find((candidate) => candidate.title === reference);
      expect(entry, reference).toBeDefined();
      for (const phrase of note) expect(entry!.note, `${reference}: ${phrase}`).toContain(phrase);
    }
  });
});
