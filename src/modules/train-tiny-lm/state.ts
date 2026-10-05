import { TINY_SCHEDULES, type ModuleState, type TinySchedule } from "@app/module-sdk";

/**
 * Every control of the lab, with the bounds the sliders use. Training runs synchronously inside a
 * useMemo, so a value from a share link or a hand-edited snapshot must never be able to ask for an
 * unbounded run: `sanitizeState` clamps or snaps every field, and the lab reads its state only
 * through it.
 */
export const CORPUS_IDS = ["harbor", "recipes", "proverbs"] as const;
export const WIZARD_MAX = 3;
export const RANK_MIN = 1;
export const RANK_MAX = 12;
export const EPOCHS_MIN = 1;
export const EPOCHS_MAX = 30;
/** Learning rates double at each notch, the way a real rate sweep is spaced. */
export const RATE_LADDER = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16] as const;
export const BATCH_LADDER = [2, 4, 8, 16, 32, 64] as const;
export const CLIP_MAX = 1.5;
export const CLIP_STEP = 0.05;
export const DATA_MODES = ["cleaned", "raw"] as const;
export type DataMode = (typeof DATA_MODES)[number];
/** Every run in the lab uses this seed, so a recipe reproduces its result exactly. */
export const SEED = 9;

export const DEFAULT_RATE = 0.5;
export const DEFAULT_BATCH = 16;

export interface LabSettings {
  wizard: number;
  corpusId: (typeof CORPUS_IDS)[number];
  rank: number;
  epochs: number;
  learningRate: number;
  schedule: TinySchedule;
  batchSize: number;
  clipNorm: number;
  data: DataMode;
}

/**
 * Version 1 stored only wizard, corpusId, rank and epochs, and trained with batch 16, rate 0.5, a
 * constant schedule, no clipping and the corpus's own training sentences. Version 2 makes those
 * settings controls and defaults them to exactly that, so an older payload keeps its meaning and
 * needs no translation.
 */
export const initialState: ModuleState = {
  wizard: 0,
  corpusId: "harbor",
  rank: 8,
  epochs: 20,
  learningRate: DEFAULT_RATE,
  schedule: "constant",
  batchSize: DEFAULT_BATCH,
  clipNorm: 0,
  data: "cleaned",
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** A finite number clamped to [low, high], or the fallback for anything else. */
export function boundedNumber(value: unknown, low: number, high: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, low, high) : fallback;
}

/** The ladder rung closest to the value on a log scale; the fallback for anything that is not a number. */
export function nearestRung(value: unknown, ladder: ReadonlyArray<number>, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const target = Math.log(Math.max(1e-9, value));
  let best = ladder[0];
  for (const rung of ladder) {
    if (Math.abs(Math.log(rung) - target) < Math.abs(Math.log(best) - target)) best = rung;
  }
  return best;
}

const pick = <T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T =>
  typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;

const SCHEDULE_IDS = TINY_SCHEDULES.map((entry) => entry.value);

/** Reads every control from untrusted input, falling back to the default for anything invalid. */
export function readSettings(value: unknown): LabSettings {
  const source = value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  return {
    wizard: Math.round(boundedNumber(source.wizard, 0, WIZARD_MAX, 0)),
    corpusId: pick(source.corpusId, CORPUS_IDS, "harbor"),
    rank: Math.round(boundedNumber(source.rank, RANK_MIN, RANK_MAX, 8)),
    epochs: Math.round(boundedNumber(source.epochs, EPOCHS_MIN, EPOCHS_MAX, 20)),
    learningRate: nearestRung(source.learningRate, RATE_LADDER, DEFAULT_RATE),
    schedule: pick(source.schedule, SCHEDULE_IDS, "constant"),
    batchSize: nearestRung(source.batchSize, BATCH_LADDER, DEFAULT_BATCH),
    clipNorm: Math.round(boundedNumber(source.clipNorm, 0, CLIP_MAX, 0) / CLIP_STEP) * CLIP_STEP,
    data: pick(source.data, DATA_MODES, "cleaned"),
  };
}

/** The state exactly as the lab stores it: unknown keys are dropped and every field is validated. */
export function sanitizeState(value: unknown): ModuleState {
  const settings = readSettings(value);
  return { ...settings, clipNorm: Number(settings.clipNorm.toFixed(2)) };
}
