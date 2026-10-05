import {
  TINY_CORPORA,
  teachingEval,
  tinyPerplexity,
  tinyRandom,
  trainTinyModel,
  type ModuleState,
} from "@app/module-sdk";

/**
 * The leaderboard's scoring rules, separated from the React card so the numbers the lesson quotes can be pinned.
 *
 * Three rules score the same trained checkpoints on the same items:
 *   normalized  an item passes when the correct continuation has the higher MEAN log-probability per
 *               character (the idea behind lm-evaluation-harness's acc_norm);
 *   raw         an item passes when the correct continuation has the higher TOTAL log-probability, which
 *               favours the shorter option because every extra character multiplies in another probability
 *               below one;
 *   sample      the checkpoint ANSWERS by sampling: it picks one of the two options in proportion to the
 *               probability it gives each full continuation. We draw SAMPLES_PER_ITEM real seeded samples per
 *               item, count c correct, and score the item with the unbiased pass@k estimate
 *               1 − C(n − c, k) / C(n, k) (teachingEval.passAtK). The checkpoint scores the mean over items.
 *
 * The draws use a seed that depends on the item only, so every checkpoint (and the clean and leaked runs) reuse
 * the same dice and differences between rows come from the models, not from the random numbers.
 */

export const RULES = ["normalized", "raw", "sample"] as const;
export type Rule = (typeof RULES)[number];
export const RULE_LABELS: Record<Rule, string> = {
  normalized: "Length-normalised",
  raw: "Raw log-probability",
  sample: "Sample and check",
};

export const SAMPLES_PER_ITEM = 20;
export const ATTEMPTS_RANGE = { min: 1, max: 10 } as const;
export const SAMPLE_SEED = 101;
export const BUDGETS = [3, 10, 30, 80] as const;
export const EPOCHS_SCALE_RANGE = { min: 0.25, max: 3 } as const;

/** Bounds that keep a pasted benchmark from freezing the page: every item is scored at every budget, twice. */
export const MAX_ITEMS = 60;
export const MAX_LINE_LENGTH = 160;
export const ITEMS_TEXT_MAX = MAX_ITEMS * (MAX_LINE_LENGTH + 1);

export const DEFAULT_ITEMS = [
  "in winter the same harbor | freezes | flowers",
  "the boats stay in | port | town",
  "in summer the harbor fills with | sails | trails",
  "the water stays | warm | wide",
  "the fog returns in the | morning | evening",
  "and the pattern | repeats | retreats",
];

const sentences = TINY_CORPORA.harbor.text
  .split(".")
  .map((part) => part.trim())
  .filter(Boolean);
/** The benchmark is written from these three sentences, so they are held out. */
export const BENCHMARK_SOURCE = `${sentences.slice(sentences.length - 3).join(". ")}.`;
export const TRAINING_TEXT = `${sentences.slice(0, sentences.length - 3).join(". ")}.`;
export const LEAKED_TEXT = `${TRAINING_TEXT} ${BENCHMARK_SOURCE}`;
/** A second unseen corpus that contamination never touches. */
export const CONTROL_TEXT = TINY_CORPORA.proverbs.text;

const Z95 = 1.959964;

export const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** How the lab prints a share, so tests can pin exactly what the table shows. */
export const percent = (value: number) => (Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "—");

/** The first MAX_ITEMS lines, each cut to MAX_LINE_LENGTH characters. */
export function boundedItemsText(value: unknown, fallback: string): string {
  if (typeof value !== "string") return fallback;
  return value
    .split("\n")
    .slice(0, MAX_ITEMS)
    .map((line) => line.slice(0, MAX_LINE_LENGTH))
    .join("\n");
}

export const initialState: ModuleState = {
  items: DEFAULT_ITEMS.join("\n"),
  contaminated: false,
  epochsScale: 1,
  rule: "normalized",
  attempts: 1,
};

/**
 * Rebuilds a state from untrusted input: unknown keys are dropped and every field is validated or clamped.
 * Payloads from before the scoring-rule selector (stateVersion 1) lack `rule` and `attempts` and receive defaults.
 */
export function sanitizeEvaluationState(value: unknown): ModuleState {
  const initial = initialState;
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initial };
  const source = value as Record<string, unknown>;
  const scale = typeof source.epochsScale === "number" && Number.isFinite(source.epochsScale) ? source.epochsScale : 1;
  const attempts =
    typeof source.attempts === "number" && Number.isFinite(source.attempts) ? Math.round(source.attempts) : 1;
  return {
    items: boundedItemsText(source.items, initial.items as string),
    contaminated: source.contaminated === true,
    epochsScale: clamp(Math.round(scale * 4) / 4, EPOCHS_SCALE_RANGE.min, EPOCHS_SCALE_RANGE.max),
    rule:
      typeof source.rule === "string" && (RULES as readonly string[]).includes(source.rule)
        ? source.rule
        : (initial.rule as string),
    attempts: clamp(attempts, ATTEMPTS_RANGE.min, Math.min(ATTEMPTS_RANGE.max, SAMPLES_PER_ITEM)),
  };
}

/** Everything a rule needs about one item under one checkpoint, computed once. */
export interface ItemScores {
  context: string;
  correct: string;
  distractor: string;
  correctTotal: number;
  distractorTotal: number;
  correctMean: number;
  distractorMean: number;
  /** Chance a sample picks the correct option: its share of the two options' total probability. */
  share: number;
}

export interface CheckpointRun {
  epochs: number;
  items: ItemScores[];
  benchmarkPerplexity: number;
  controlPerplexity: number;
}

export function scoreItem(weights: Float32Array, item: teachingEval.BenchmarkItem): ItemScores {
  const correct = teachingEval.scoreContinuation(weights, item.context, item.correct);
  const distractor = teachingEval.scoreContinuation(weights, item.context, item.distractor);
  return {
    ...item,
    correctTotal: correct.total,
    distractorTotal: distractor.total,
    correctMean: correct.mean,
    distractorMean: distractor.mean,
    share: 1 / (1 + Math.exp(distractor.total - correct.total)),
  };
}

/** Trains the four checkpoints (3, 10, 30, 80 epochs times the multiplier) and scores every item once. */
export function trainCheckpoints(
  text: string,
  items: teachingEval.BenchmarkItem[],
  epochsScale: number,
): CheckpointRun[] {
  return BUDGETS.map((budget) => {
    const epochs = Math.max(1, Math.round(budget * epochsScale));
    const run = trainTinyModel({ text, epochs, batchSize: 16, learningRate: 0.6, seed: 1 });
    return {
      epochs,
      items: items.map((item) => scoreItem(run.weights, item)),
      benchmarkPerplexity: tinyPerplexity(run.weights, BENCHMARK_SOURCE),
      controlPerplexity: tinyPerplexity(run.weights, CONTROL_TEXT),
    };
  });
}

/** Real draws: sample j picks the correct option when a seeded uniform falls below its share. */
export function drawCorrect(share: number, samples: number, seed: number): number {
  const random = tinyRandom(seed);
  let correct = 0;
  for (let draw = 0; draw < samples; draw += 1) if (random() < share) correct += 1;
  return correct;
}

export interface RatedItem extends ItemScores {
  /** What the rule scores this item: 1 or 0 for the likelihood rules, pass@k for sample and check. */
  score: number;
  passed: boolean;
  /** Right-hand panel value: the margin in the rule's units, or pass@1 for sample and check. */
  side: number;
  /** What a paired comparison between two checkpoints subtracts: the margin, or pass@k. */
  paired: number;
  /** Samples (of SAMPLES_PER_ITEM) that picked the correct option. Zero unless the rule is sample and check. */
  correctSamples: number;
}

export interface RatedCheckpoint {
  epochs: number;
  rule: Rule;
  items: RatedItem[];
  /** Items with a passing score: likelihood rules only. */
  passes: number;
  /** The headline score: pass rate, or mean pass@k. */
  score: number;
  /** 95% interval for the headline score: Wilson for pass rates, Student's t over items for mean pass@k. */
  interval: [number, number];
  side: ReturnType<typeof teachingEval.meanInterval>;
  standardError: number;
  benchmarkPerplexity: number;
  controlPerplexity: number;
}

function rateItem(item: ItemScores, index: number, rule: Rule, attempts: number): RatedItem {
  if (rule === "sample") {
    const correctSamples = drawCorrect(item.share, SAMPLES_PER_ITEM, SAMPLE_SEED + index);
    const score = teachingEval.passAtK(SAMPLES_PER_ITEM, correctSamples, Math.min(attempts, SAMPLES_PER_ITEM));
    return {
      ...item,
      score,
      passed: score >= 0.5,
      side: correctSamples / SAMPLES_PER_ITEM,
      paired: score,
      correctSamples,
    };
  }
  const margin =
    rule === "raw" ? item.correctTotal - item.distractorTotal : item.correctMean - item.distractorMean;
  const passed = margin > 0;
  return { ...item, score: passed ? 1 : 0, passed, side: margin, paired: margin, correctSamples: 0 };
}

export function rateCheckpoints(runs: CheckpointRun[], rule: Rule, attempts: number): RatedCheckpoint[] {
  return runs.map((run) => {
    const items = run.items.map((item, index) => rateItem(item, index, rule, attempts));
    const n = items.length;
    const passes = items.filter((item) => item.passed).length;
    const scores = items.map((item) => item.score);
    const score = n === 0 ? 0 : scores.reduce((sum, value) => sum + value, 0) / n;
    let interval: [number, number];
    let standardError: number;
    if (rule === "sample") {
      const stats = teachingEval.meanInterval(scores);
      interval = Number.isFinite(stats.low) ? [clamp(stats.low, 0, 1), clamp(stats.high, 0, 1)] : [0, 1];
      standardError = stats.standardError;
    } else {
      interval = teachingEval.wilsonInterval(passes, n);
      standardError = teachingEval.proportionStandardError(score, n);
    }
    return {
      epochs: run.epochs,
      rule,
      items,
      passes,
      score,
      interval,
      side: teachingEval.meanInterval(items.map((item) => item.side)),
      standardError,
      benchmarkPerplexity: run.benchmarkPerplexity,
      controlPerplexity: run.controlPerplexity,
    };
  });
}

/** Highest headline score first, ties broken by the right-hand panel's mean. */
export function rankCheckpoints(rows: RatedCheckpoint[]): RatedCheckpoint[] {
  return [...rows].sort((left, right) => right.score - left.score || right.side.mean - left.side.mean);
}

/**
 * Items needed for a 95% interval of ± halfWidth. Pass rates use p(1 − p); sample and check uses the per-item
 * variance of pass@k, which shrinks as scores saturate, and never goes below the two items a standard error needs.
 */
export function itemsNeeded(row: RatedCheckpoint, halfWidth: number): number {
  const n = row.items.length;
  if (row.rule !== "sample") return teachingEval.itemsForHalfWidth(clamp(row.score, 0.05, 0.95), halfWidth);
  const variance = Number.isFinite(row.standardError) ? row.standardError ** 2 * n : 0.25;
  return Math.max(2, Math.ceil((Z95 * Z95 * Math.max(variance, 1e-9)) / (halfWidth * halfWidth)));
}

/** Elo update for one pairwise vote: a wins (true) or loses (false) against b. */
export function eloUpdate(ratingA: number, ratingB: number, aWon: boolean, k = 32): { a: number; b: number } {
  const expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
  const delta = k * ((aWon ? 1 : 0) - expectedA);
  return { a: ratingA + delta, b: ratingB - delta };
}
