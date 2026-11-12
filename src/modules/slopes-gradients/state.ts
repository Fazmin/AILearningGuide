import type { ModuleState } from "@app/module-sdk";
import { MAX_WALK_STEPS, NUDGES, RACE_BUDGET, clamp, type Curve, type Surface } from "./curves";

export type Legs = "w" | "b" | "both";
export const LEGS: readonly Legs[] = ["w", "b", "both"];
/** The two pictures of a two-dial meter: the contour map seen from above, and the surface it describes. */
export type Picture = "map" | "surface";
export const PICTURES: readonly Picture[] = ["map", "surface"];
/** The two curves each one-dial card offers. */
export const SLOPE_CURVES: readonly Curve[] = ["valleys", "shelf"];
export const WALK_CURVES: readonly Curve[] = ["valleys", "bowl"];
const MAP_SURFACES: readonly Surface[] = ["bowl", "valley", "saddle"];

/** How many options each predict-first question has. A stored guess outside the range means "not answered". */
export const GUESS_OPTIONS = { nudgeGuess: 3, arrowGuess: 3, chainGuess: 4, walkGuess: 3 } as const;

export const LIMITS = {
  nudgeW: [0, 5],
  dial: [0, 6],
  walkStep: [0.05, 1.2],
  raceStep: [0.05, 1.1],
} as const;

export const initialState: ModuleState = {
  // Card 1: nudge the dial
  nudgeW: 4,
  nudge: 1,
  nudgeGuess: -1,
  // Card 2: a slope at every point
  slopeCurve: "valleys",
  slopeW: 1,
  // Card 3: two dials, one slope each
  surface: "bowl",
  mapW: 4.5,
  mapB: 4,
  legs: "w",
  arrowGuess: -1,
  mapView: "map",
  // Card 4: dials that drive dials
  chainW: 3,
  chainGuess: -1,
  // Card 5: walk downhill
  walkCurve: "valleys",
  walkStart: 2.6,
  walkStep: 0.2,
  walkSteps: 0,
  walkGuess: -1,
  // Card 6: the challenge
  raceStep: 0.1,
  raceSteps: 0,
  racePrev: 0,
  raceView: "map",
};

export interface LabState {
  nudgeW: number;
  nudge: number;
  nudgeGuess: number;
  slopeCurve: Curve;
  slopeW: number;
  surface: Surface;
  mapW: number;
  mapB: number;
  legs: Legs;
  arrowGuess: number;
  mapView: Picture;
  chainW: number;
  chainGuess: number;
  walkCurve: Curve;
  walkStart: number;
  walkStep: number;
  walkSteps: number;
  walkGuess: number;
  raceStep: number;
  raceSteps: number;
  racePrev: number;
  raceView: Picture;
}

const round = (value: number, digits = 2) => Number(value.toFixed(digits));

/** A finite number inside [low, high], on a two-decimal grid; anything else falls back. */
const number = (value: unknown, fallback: number, low: number, high: number) =>
  round(clamp(typeof value === "number" && Number.isFinite(value) ? value : fallback, low, high));

const whole = (value: unknown, fallback: number, low: number, high: number) =>
  Math.round(clamp(typeof value === "number" && Number.isFinite(value) ? value : fallback, low, high));

const member = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

const guess = (value: unknown, options: number) =>
  typeof value === "number" && Number.isInteger(value) && value >= 0 && value < options ? value : -1;

/** The offered nudge closest to whatever was stored. */
const nearestNudge = (value: unknown) => {
  const target = typeof value === "number" && Number.isFinite(value) ? value : (initialState.nudge as number);
  return NUDGES.reduce((best, option) => (Math.abs(option - target) < Math.abs(best - target) ? option : best));
};

/**
 * Read lab state from anything: a share link, an old snapshot, a hand-edited store. Every value comes
 * back inside its control's range, so the lab can draw it without further checks.
 */
export function readState(state: Record<string, unknown> | null | undefined): LabState {
  const source = state && typeof state === "object" ? state : {};
  const racePrev = source.racePrev;
  return {
    nudgeW: number(source.nudgeW, 4, LIMITS.nudgeW[0], LIMITS.nudgeW[1]),
    nudge: nearestNudge(source.nudge),
    nudgeGuess: guess(source.nudgeGuess, GUESS_OPTIONS.nudgeGuess),
    slopeCurve: member(source.slopeCurve, SLOPE_CURVES, "valleys"),
    slopeW: number(source.slopeW, 1, LIMITS.dial[0], LIMITS.dial[1]),
    surface: member(source.surface, MAP_SURFACES, "bowl"),
    mapW: number(source.mapW, 4.5, LIMITS.dial[0], LIMITS.dial[1]),
    mapB: number(source.mapB, 4, LIMITS.dial[0], LIMITS.dial[1]),
    legs: member(source.legs, LEGS, "w"),
    arrowGuess: guess(source.arrowGuess, GUESS_OPTIONS.arrowGuess),
    mapView: member(source.mapView, PICTURES, "map"),
    chainW: number(source.chainW, 3, LIMITS.dial[0], LIMITS.dial[1]),
    chainGuess: guess(source.chainGuess, GUESS_OPTIONS.chainGuess),
    walkCurve: member(source.walkCurve, WALK_CURVES, "valleys"),
    walkStart: number(source.walkStart, 2.6, LIMITS.dial[0], LIMITS.dial[1]),
    walkStep: number(source.walkStep, 0.2, LIMITS.walkStep[0], LIMITS.walkStep[1]),
    walkSteps: whole(source.walkSteps, 0, 0, MAX_WALK_STEPS),
    walkGuess: guess(source.walkGuess, GUESS_OPTIONS.walkGuess),
    raceStep: number(source.raceStep, 0.1, LIMITS.raceStep[0], LIMITS.raceStep[1]),
    raceSteps: whole(source.raceSteps, 0, 0, RACE_BUDGET),
    // 0 means "no earlier attempt to compare with".
    racePrev:
      typeof racePrev === "number" && Number.isFinite(racePrev) && racePrev >= LIMITS.raceStep[0]
        ? number(racePrev, 0, LIMITS.raceStep[0], LIMITS.raceStep[1])
        : 0,
    raceView: member(source.raceView, PICTURES, "map"),
  };
}

export function hydrateState(value: string): ModuleState {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    return { ...readState(parsed as Record<string, unknown>) };
  } catch {
    return { ...initialState };
  }
}
