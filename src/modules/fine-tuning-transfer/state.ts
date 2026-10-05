import { TINY_VOCAB_SIZE, type ModuleState } from "@app/module-sdk";
import { DECISION_KEYS, DEFAULT_DECISION, type DecisionAnswers } from "./decision";

/**
 * Bounds for the controls. They are the slider ranges in Explore.tsx, so every state the
 * interface can produce passes through `sanitizeState` unchanged. Fine-tuning runs
 * synchronously inside a useMemo, so an unbounded value from a shared link would freeze
 * the lab.
 */
export const EPOCHS_MIN = 0;
export const EPOCHS_MAX = 60;
export const LEARNING_RATE_MIN = 0.05;
export const LEARNING_RATE_MAX = 2;

export const DEFAULT_EPOCHS = 20;
export const DEFAULT_LEARNING_RATE = 0.6;

/**
 * The replay share is the percentage of the original corpus's sentences mixed into the fine-tuning
 * data. One notch is exactly one of its eight sentences, so every position the slider can reach is a
 * whole number of sentences.
 */
export const REPLAY_MAX = 100;
export const REPLAY_STEP = 12.5;
export const DEFAULT_REPLAY = 0;
/** How many of the eight target sentences the held-out split keeps out of fine-tuning. */
export const HELD_OUT_MIN = 0;
export const HELD_OUT_MAX = 4;
export const DEFAULT_HELD_OUT = 3;

/** The freeze grid groups the weight table into blocks of this many context rows. */
export const BLOCK_SIZE = 6;
export const BLOCK_COUNT = Math.ceil(TINY_VOCAB_SIZE / BLOCK_SIZE);
export const BLOCK_IDS: readonly string[] = Array.from({ length: BLOCK_COUNT }, (_, index) => `block-${index}`);

/**
 * Version 1 stored only fineTuneEpochs, learningRate and frozen. The four decision answers are
 * new in version 2 and default to "no". Version 3 adds replayShare and heldOutSentences, which belong
 * to a card of their own and change nothing the other cards compute, so an older payload keeps its
 * meaning once they take their defaults; nothing in a version 1 or 2 payload needs translating.
 */
export const initialState: ModuleState = {
  fineTuneEpochs: DEFAULT_EPOCHS,
  learningRate: DEFAULT_LEARNING_RATE,
  frozen: [],
  ...DEFAULT_DECISION,
  replayShare: DEFAULT_REPLAY,
  heldOutSentences: DEFAULT_HELD_OUT,
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** A finite number clamped to [low, high], or the fallback for anything else. */
export function boundedNumber(value: unknown, low: number, high: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, low, high) : fallback;
}

/** The epoch count is a whole number of passes, as the slider (step 1) can only produce. */
export function boundedEpochs(value: unknown): number {
  return Math.round(boundedNumber(value, EPOCHS_MIN, EPOCHS_MAX, DEFAULT_EPOCHS));
}

export function boundedLearningRate(value: unknown): number {
  return boundedNumber(value, LEARNING_RATE_MIN, LEARNING_RATE_MAX, DEFAULT_LEARNING_RATE);
}

/** A multiple of the slider's 12.5% notch between 0 and 100. */
export function boundedReplay(value: unknown): number {
  const share = boundedNumber(value, 0, REPLAY_MAX, DEFAULT_REPLAY);
  return Math.round(share / REPLAY_STEP) * REPLAY_STEP;
}

/** A whole number of target sentences, 0 to 4. */
export function boundedHeldOut(value: unknown): number {
  return Math.round(boundedNumber(value, HELD_OUT_MIN, HELD_OUT_MAX, DEFAULT_HELD_OUT));
}

/** Only known block ids, each once, in the order the learner froze them. */
export function boundedFrozen(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const kept: string[] = [];
  for (const entry of value) {
    if (typeof entry === "string" && BLOCK_IDS.includes(entry) && !kept.includes(entry)) kept.push(entry);
  }
  return kept;
}

/** A decision answer is a real boolean; anything else falls back to "no". */
export function boundedFlag(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

/** The four yes/no answers of the "Should you fine-tune?" card. */
export function boundedDecision(source: Record<string, unknown>): DecisionAnswers {
  const answers = { ...DEFAULT_DECISION };
  for (const key of DECISION_KEYS) answers[key] = boundedFlag(source[key], DEFAULT_DECISION[key]);
  return answers;
}

/**
 * Rebuilds a state from untrusted input: unknown keys are dropped and every field is
 * validated or clamped to the control's range.
 */
export function sanitizeState(value: unknown): ModuleState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initialState, frozen: [] };
  const source = value as Record<string, unknown>;
  return {
    fineTuneEpochs: boundedEpochs(source.fineTuneEpochs),
    learningRate: boundedLearningRate(source.learningRate),
    frozen: boundedFrozen(source.frozen),
    ...boundedDecision(source),
    replayShare: boundedReplay(source.replayShare),
    heldOutSentences: boundedHeldOut(source.heldOutSentences),
  };
}
