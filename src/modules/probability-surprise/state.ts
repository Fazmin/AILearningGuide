import type { ModuleState } from "@app/module-sdk";
import { CHALLENGE_ROUNDS, EQUAL_BARS, PROMPTS, SKY_BARS, SKY_SCORES, TEST_SIZES, WORD_COUNT, isBars } from "./chance";

/** Most draws the tally card keeps. Enough to press "1,000 draws" and add a few by hand. */
export const MAX_DRAWS = 2000;
export const SCORE_LIMIT = 6;
export const TEMPERATURE_MIN = 0.2;
export const TEMPERATURE_MAX = 4;
export const PRIOR_MIN = 5;
export const PRIOR_MAX = 90;
export const DEFAULT_LUCK = 13;

const join = (values: ReadonlyArray<number>) => values.join(",");

/**
 * Everything the reader can change. The three sets of bars and the scores are stored as comma
 * lists ("50,25,12,8,5") so each is one key that a lab scene can set in a single line.
 */
export const initialState: ModuleState = {
  view: "percent",
  bars: join(SKY_BARS),
  draws: 0,
  luck: DEFAULT_LUCK,
  follow: 0,
  predictDraws: -1,
  testSize: 25,
  prompt: 0,
  clue: 0,
  prior: 50,
  predictPrior: -1,
  word: 0,
  first: 0,
  second: 1,
  shape: 50,
  predictShape: -1,
  guess: join(EQUAL_BARS),
  predictBill: -1,
  scores: join(SKY_SCORES),
  temperature: 1,
  round: 0,
  answer: join(EQUAL_BARS),
  scored: 0,
  tries: 0,
  revealed: 0,
};

export interface LabState {
  view: "percent" | "counts";
  bars: number[];
  draws: number;
  luck: number;
  follow: number;
  predictDraws: number;
  testSize: number;
  prompt: number;
  clue: number;
  prior: number;
  predictPrior: number;
  word: number;
  first: number;
  second: number;
  shape: number;
  predictShape: number;
  guess: number[];
  predictBill: number;
  scores: number[];
  temperature: number;
  round: number;
  answer: number[];
  scored: number;
  tries: number;
  revealed: number;
}

const whole = (value: unknown, fallback: number, low: number, high: number) =>
  typeof value === "number" && Number.isInteger(value) && value >= low && value <= high ? value : fallback;
const ranged = (value: unknown, fallback: number, low: number, high: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;
const tenth = (value: number) => Math.round(value * 10) / 10;
const numbersOf = (value: unknown) =>
  typeof value === "string" && /^-?\d+(\.\d+)?(,-?\d+(\.\d+)?)*$/.test(value) ? value.split(",").map(Number) : null;

/** A stored comma list as five whole percents that add to 100, or the fallback. */
export function parseBars(value: unknown, fallback: ReadonlyArray<number>): number[] {
  const parsed = numbersOf(value);
  return isBars(parsed) ? parsed : [...fallback];
}

/** A stored comma list as five scores, each clamped to the slider range and rounded to a tenth. */
export function parseScores(value: unknown): number[] {
  const parsed = numbersOf(value);
  if (!parsed || parsed.length !== WORD_COUNT || !parsed.every(Number.isFinite)) return [...SKY_SCORES];
  return parsed.map((score) => tenth(Math.min(SCORE_LIMIT, Math.max(-SCORE_LIMIT, score))));
}

export const formatList = join;

/** Read stored state into typed, in-range values. Anything missing or malformed falls back to its default. */
export function readState(state: Record<string, unknown>): LabState {
  const d = initialState as Record<string, number | string>;
  const lastWord = WORD_COUNT - 1;
  const pickIndex = (value: unknown, fallback: number) => whole(value, fallback, -1, 2);
  return {
    view: state.view === "counts" ? "counts" : "percent",
    bars: parseBars(state.bars, SKY_BARS),
    draws: whole(state.draws, 0, 0, MAX_DRAWS),
    luck: whole(state.luck, DEFAULT_LUCK, 1, 999),
    follow: whole(state.follow, 0, 0, lastWord),
    predictDraws: pickIndex(state.predictDraws, -1),
    testSize: (TEST_SIZES as readonly number[]).includes(state.testSize as number) ? (state.testSize as number) : (d.testSize as number),
    prompt: whole(state.prompt, 0, 0, PROMPTS.length - 1),
    clue: whole(state.clue, 0, 0, lastWord),
    prior: whole(state.prior, 50, PRIOR_MIN, PRIOR_MAX),
    predictPrior: pickIndex(state.predictPrior, -1),
    word: whole(state.word, 0, 0, lastWord),
    first: whole(state.first, 0, 0, lastWord),
    second: whole(state.second, 1, 0, lastWord),
    shape: whole(state.shape, 50, 0, 100),
    predictShape: pickIndex(state.predictShape, -1),
    guess: parseBars(state.guess, EQUAL_BARS),
    predictBill: pickIndex(state.predictBill, -1),
    scores: parseScores(state.scores),
    temperature: tenth(ranged(state.temperature, 1, TEMPERATURE_MIN, TEMPERATURE_MAX)),
    round: whole(state.round, 0, 0, CHALLENGE_ROUNDS.length - 1),
    answer: parseBars(state.answer, EQUAL_BARS),
    scored: whole(state.scored, 0, 0, 1),
    tries: whole(state.tries, 0, 0, 999),
    revealed: whole(state.revealed, 0, 0, 1),
  };
}

/** The typed state back in its stored, JSON-safe form. */
export function storeState(state: LabState): ModuleState {
  return {
    ...state,
    bars: join(state.bars),
    guess: join(state.guess),
    scores: join(state.scores),
    answer: join(state.answer),
  };
}

/** Parse a saved or shared state string. Malformed input gives the defaults, never an exception. */
export function hydrate(value: string): ModuleState {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    return storeState(readState(parsed as Record<string, unknown>));
  } catch {
    return { ...initialState };
  }
}
