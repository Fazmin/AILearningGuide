/**
 * The same recipe trained over five shuffle seeds, so one run's number can be read against the
 * spread the seed alone produces. The model starts from zeroed weights, so the seed changes only the
 * order the training pairs are shuffled in; every other difference between runs is the recipe.
 */
import { TINY_CORPORA, trainTinyModel, type TinySchedule, type TinyTrainRun } from "@app/module-sdk";

export const CORPUS_TEXT = TINY_CORPORA.harbor.text;
export const EPOCHS = 25;
/** Seed 1 is the lab's own seed, so the single-seed curves and the first column of the table agree. */
export const SEED_SET = [1, 2, 3, 4, 5] as const;

export interface Recipe {
  peakLearningRate: number;
  warmup: number;
  clipNorm: number;
  weightDecay: number;
  batchSize: number;
}

export type SeedsMode = "one" | "five";

/** One run with the lab's settings. `full` keeps a history point at every step for the gradient trace. */
export function trainRecipe(recipe: Recipe, schedule: TinySchedule, seed: number, full: boolean): TinyTrainRun {
  return trainTinyModel({
    text: CORPUS_TEXT,
    epochs: EPOCHS,
    batchSize: recipe.batchSize,
    learningRate: recipe.peakLearningRate,
    schedule,
    warmupFraction: recipe.warmup,
    clipNorm: recipe.clipNorm,
    weightDecay: recipe.weightDecay,
    seed,
    // One checkpoint per epoch boundary, each scored on the whole corpus.
    checkpoints: EPOCHS + 1,
    historyPoints: full ? Number.MAX_SAFE_INTEGER : 1,
  });
}

export interface SeedSummary {
  /** Final whole-corpus loss for each seed, in SEED_SET order. NaN marks a run that diverged. */
  finals: number[];
  /** Mean over the seeds, or Infinity when any run diverged, so it ranks last. */
  mean: number;
  min: number;
  max: number;
  /** max − min of the final loss: the width of the band at the last epoch. */
  spread: number;
  /** Lowest and highest whole-corpus loss across the seeds at each epoch boundary. */
  band: { low: number; high: number }[];
}

export function summarizeSeeds(runs: ReadonlyArray<TinyTrainRun>): SeedSummary {
  const finals = runs.map((run) => run.finalLoss);
  const finite = finals.filter(Number.isFinite);
  const allFinite = finite.length === finals.length && finals.length > 0;
  const min = finite.length ? Math.min(...finite) : Number.NaN;
  const max = finite.length ? Math.max(...finite) : Number.NaN;
  const epochs = runs[0]?.checkpoints.length ?? 0;
  const band = Array.from({ length: epochs }, (_, epoch) => {
    const values = runs.map((run) => run.checkpoints[epoch]?.loss).filter((value): value is number => Number.isFinite(value));
    return values.length ? { low: Math.min(...values), high: Math.max(...values) } : { low: Number.NaN, high: Number.NaN };
  });
  return {
    finals,
    mean: allFinite ? finite.reduce((sum, value) => sum + value, 0) / finite.length : Number.POSITIVE_INFINITY,
    min,
    max,
    spread: finite.length ? max - min : Number.NaN,
    band,
  };
}

export interface NamedSummary {
  label: string;
  summary: SeedSummary;
}

export interface Comparison {
  /** The two schedules compared, better mean first. */
  better: string;
  other: string;
  gap: number;
  /** The wider of the two seed spreads. */
  noise: number;
  /** True when the gap is larger than that spread, so five seeds agree on the order. */
  outsideBand: boolean;
}

const compare = (left: NamedSummary, right: NamedSummary): Comparison => {
  const [better, other] = left.summary.mean <= right.summary.mean ? [left, right] : [right, left];
  const gap = other.summary.mean - better.summary.mean;
  const noise = Math.max(better.summary.spread, other.summary.spread);
  return { better: better.label, other: other.label, gap, noise, outsideBand: gap > noise };
};

export interface SeedVerdict {
  /** Best versus second-best mean. */
  top: Comparison | null;
  /** The two schedules whose means are closest, among those that finished. */
  closest: Comparison | null;
}

/** Ranks the schedules by their five-seed mean and compares the top two and the closest pair with their spreads. */
export function seedVerdict(entries: ReadonlyArray<NamedSummary>): SeedVerdict {
  const finished = entries.filter((entry) => Number.isFinite(entry.summary.mean));
  if (finished.length < 2) return { top: null, closest: null };
  const ranked = [...finished].sort((left, right) => left.summary.mean - right.summary.mean);
  const top = compare(ranked[0], ranked[1]);
  let closest: Comparison | null = null;
  for (let first = 0; first < ranked.length; first += 1) {
    for (let second = first + 1; second < ranked.length; second += 1) {
      const candidate = compare(ranked[first], ranked[second]);
      if (!closest || candidate.gap < closest.gap) closest = candidate;
    }
  }
  return { top, closest };
}
