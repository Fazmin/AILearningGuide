import type { ModuleState } from "@app/module-sdk";
import {
  AUTO_MAX,
  DOTS_KINDS,
  HAND_LIMIT,
  HAND_START,
  INPUT_STOPS,
  LINE_START,
  MAX_LAYERS,
  MAX_UNITS,
  OUTPUT_STOPS,
  RATE_STOPS,
  SMALL_NAMES,
  TRAIN_STEPS,
  WIDTH_STOPS,
  X_RANGE,
  type DotsKind,
} from "./model";
import { clamp } from "./net";

/** How many options each predict-first question has. A stored guess outside the range means "not answered". */
export const GUESS_OPTIONS = { lineGuess: 3, comboGuess: 3, sameGuess: 3, shapeGuess: 3, wiresGuess: 3 } as const;

export const PLAY_MAX_WIDTH = 6;
export const MAX_SEED = 999;

export const initialState: ModuleState = {
  // Card 1: two knobs
  lineW: LINE_START.w,
  lineB: LINE_START.b,
  lineX: 1,
  lineGuess: -1,
  // Cards 2 and 3: seven knobs, one machine. k0…k6 are the knobs as set by hand (or by a scramble);
  // `auto` is how many steps the machine has taken from there.
  k0: HAND_START[0],
  k1: HAND_START[1],
  k2: HAND_START[2],
  k3: HAND_START[3],
  k4: HAND_START[4],
  k5: HAND_START[5],
  k6: HAND_START[6],
  pick: 0,
  scramble: 1,
  comboGuess: -1,
  auto: 0,
  sameGuess: -1,
  // Card 4: more neurons
  units: 1,
  unitsSeed: 1,
  shapeGuess: -1,
  // Card 5: count the knobs
  inputs: 1,
  layers: 1,
  width: 2,
  outputs: 1,
  wiresGuess: -1,
  // Card 6: the playground
  pgDots: "wave",
  pgLayers: 1,
  pgWidth: 2,
  pgRate: 0.05,
  pgMomentum: "on",
  pgSeed: 1,
  pgSteps: 0,
};

export interface LabState {
  lineW: number;
  lineB: number;
  lineX: number;
  lineGuess: number;
  /** The seven hand-set knobs, in the order of `SMALL_NAMES`. */
  hand: number[];
  pick: number;
  scramble: number;
  comboGuess: number;
  auto: number;
  sameGuess: number;
  units: number;
  unitsSeed: number;
  shapeGuess: number;
  inputs: number;
  layers: number;
  width: number;
  outputs: number;
  wiresGuess: number;
  pgDots: DotsKind;
  pgLayers: number;
  pgWidth: number;
  pgRate: number;
  pgMomentum: boolean;
  pgSeed: number;
  pgSteps: number;
}

const finite = (value: unknown, fallback: number) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);

/** A finite number inside [low, high], kept to `digits` decimals; anything else falls back. */
const number = (value: unknown, fallback: number, low: number, high: number, digits = 2) =>
  Number(clamp(finite(value, fallback), low, high).toFixed(digits)) + 0;

const whole = (value: unknown, fallback: number, low: number, high: number) => Math.round(clamp(finite(value, fallback), low, high));

const member = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

const guess = (value: unknown, options: number) =>
  typeof value === "number" && Number.isInteger(value) && value >= 0 && value < options ? value : -1;

/** The offered stop closest to whatever was stored. */
const nearest = (value: unknown, stops: readonly number[], fallback: number) => {
  const target = finite(value, fallback);
  return stops.reduce((best, stop) => (Math.abs(stop - target) < Math.abs(best - target) ? stop : best));
};

/**
 * Read lab state from anything: a share link, an old snapshot, a hand-edited store. Every value comes
 * back inside its control's range, so the lab can draw and train from it without further checks.
 */
export function readState(state: Record<string, unknown> | null | undefined): LabState {
  const source = state && typeof state === "object" ? state : {};
  return {
    lineW: number(source.lineW, LINE_START.w, -HAND_LIMIT, HAND_LIMIT),
    lineB: number(source.lineB, LINE_START.b, -HAND_LIMIT, HAND_LIMIT),
    lineX: number(source.lineX, 1, X_RANGE[0], X_RANGE[1]),
    lineGuess: guess(source.lineGuess, GUESS_OPTIONS.lineGuess),
    hand: SMALL_NAMES.map((_, index) => number(source[`k${index}`], HAND_START[index], -HAND_LIMIT, HAND_LIMIT, 3)),
    pick: whole(source.pick, 0, 0, SMALL_NAMES.length - 1),
    scramble: whole(source.scramble, 1, 1, MAX_SEED),
    comboGuess: guess(source.comboGuess, GUESS_OPTIONS.comboGuess),
    auto: whole(source.auto, 0, 0, AUTO_MAX),
    sameGuess: guess(source.sameGuess, GUESS_OPTIONS.sameGuess),
    units: whole(source.units, 1, 1, MAX_UNITS),
    unitsSeed: whole(source.unitsSeed, 1, 1, MAX_SEED),
    shapeGuess: guess(source.shapeGuess, GUESS_OPTIONS.shapeGuess),
    inputs: nearest(source.inputs, INPUT_STOPS, 1),
    layers: whole(source.layers, 1, 1, MAX_LAYERS),
    width: nearest(source.width, WIDTH_STOPS, 2),
    outputs: nearest(source.outputs, OUTPUT_STOPS, 1),
    wiresGuess: guess(source.wiresGuess, GUESS_OPTIONS.wiresGuess),
    pgDots: member(source.pgDots, DOTS_KINDS, "wave"),
    pgLayers: whole(source.pgLayers, 1, 1, 2),
    pgWidth: whole(source.pgWidth, 2, 1, PLAY_MAX_WIDTH),
    pgRate: nearest(source.pgRate, RATE_STOPS, 0.05),
    pgMomentum: member(source.pgMomentum, ["on", "off"] as const, "on") === "on",
    pgSeed: whole(source.pgSeed, 1, 1, MAX_SEED),
    pgSteps: whole(source.pgSteps, 0, 0, TRAIN_STEPS),
  };
}

/** The same reading, laid back out as the flat record the shell stores. */
export function hydrateState(value: string): ModuleState {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    return { ...initialState };
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
  const lab = readState(parsed as Record<string, unknown>);
  const { hand, pgMomentum, ...rest } = lab;
  const knobs = Object.fromEntries(hand.map((knob, index) => [`k${index}`, knob]));
  return { ...rest, ...knobs, pgMomentum: pgMomentum ? "on" : "off" };
}

/** A patch that sets all seven hand knobs and sends the machine back to step 0. */
export function handPatch(knobs: readonly number[]): Partial<ModuleState> {
  const patch: Record<string, number> = { auto: 0 };
  knobs.forEach((knob, index) => {
    patch[`k${index}`] = Number(clamp(knob, -HAND_LIMIT, HAND_LIMIT).toFixed(3)) + 0;
  });
  return patch;
}
