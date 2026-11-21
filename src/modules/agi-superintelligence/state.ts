import type { ModuleState } from "@app/module-sdk";
import {
  AHEAD_MAX,
  ASI_CELL,
  BOTTLENECK_STOPS,
  CELL_COUNT,
  CLAIMS,
  DOUBLING_STOPS,
  HORIZON,
  MEASURED_DOUBLING_MONTHS,
  MULTIPLIER_STOPS,
  PROFILE_MODELS,
  PROJECT_STOPS,
  RETURN_STOPS,
  SHARE_STOPS,
  SYSTEMS,
  VERDICTS,
  clamp,
  decodePlacement,
  decodeTags,
  encodePlacement,
  encodeTags,
  type ProfileModel,
  type Verdict,
} from "./model";

/** How many options each predict-first question has. A stored guess outside the range means "not answered". */
export const GUESS_OPTIONS = {
  levelGuess: 3,
  profileGuess: 3,
  speedGuess: 3,
  loopGuess: 3,
  surveyGuess: 3,
} as const;

export const initialState: ModuleState = {
  // Card 1
  cell: ASI_CELL,
  chip: 0,
  place: "-".repeat(SYSTEMS.length),
  levelGuess: -1,
  // Card 2
  profile: "gpt5",
  rule: "sum",
  profileGuess: -1,
  // Card 3
  work: 100,
  speed: 1,
  copies: 1,
  inOrder: 0,
  world: 0,
  speedGuess: -1,
  // Card 4
  returns: 1.5,
  bottleneck: 0,
  prevReturns: 0,
  prevBottleneck: 0,
  loopYear: 20,
  loopGuess: -1,
  // Card 5
  survey: "2023",
  ahead: 5,
  doubling: MEASURED_DOUBLING_MONTHS,
  surveyGuess: -1,
  // Card 6
  tags: "-".repeat(CLAIMS.length),
  claim: 0,
};

export interface LabState {
  cell: number;
  chip: number;
  placement: number[];
  levelGuess: number;
  profile: ProfileModel;
  rule: Verdict;
  profileGuess: number;
  work: number;
  speed: number;
  copies: number;
  inOrder: number;
  world: number;
  speedGuess: number;
  returns: number;
  bottleneck: number;
  /** The previous run on card 4, kept as a ghost; returns 0 means there is none. */
  prevReturns: number;
  prevBottleneck: number;
  loopYear: number;
  loopGuess: number;
  survey: "2022" | "2023";
  ahead: number;
  doubling: number;
  surveyGuess: number;
  tags: ReturnType<typeof decodeTags>;
  claim: number;
}

const finite = (value: unknown, fallback: number) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
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
/** The same on a ratio scale, for stops that span several powers of ten. */
const nearestRatio = (value: unknown, stops: readonly number[], fallback: number) => {
  const target = finite(value, fallback);
  if (!(target > 0)) return stops[0];
  return stops.reduce((best, stop) => (Math.abs(Math.log(stop / target)) < Math.abs(Math.log(best / target)) ? stop : best));
};

/**
 * Read lab state from anything: a share link, an old snapshot, a hand-edited store. Every value comes
 * back inside its control's range, so the lab can draw it without further checks.
 */
export function readState(state: Record<string, unknown> | null | undefined): LabState {
  const source = state && typeof state === "object" ? state : {};
  const prevReturns = finite(source.prevReturns, 0);
  return {
    cell: whole(source.cell, ASI_CELL, 0, CELL_COUNT - 1),
    chip: whole(source.chip, 0, 0, SYSTEMS.length - 1),
    placement: decodePlacement(typeof source.place === "string" ? source.place : ""),
    levelGuess: guess(source.levelGuess, GUESS_OPTIONS.levelGuess),
    profile: member(source.profile, PROFILE_MODELS, "gpt5"),
    rule: member(source.rule, VERDICTS, "sum"),
    profileGuess: guess(source.profileGuess, GUESS_OPTIONS.profileGuess),
    work: nearestRatio(source.work, PROJECT_STOPS, 100),
    speed: nearestRatio(source.speed, MULTIPLIER_STOPS, 1),
    copies: nearestRatio(source.copies, MULTIPLIER_STOPS, 1),
    inOrder: nearest(source.inOrder, SHARE_STOPS, 0),
    world: nearest(source.world, SHARE_STOPS, 0),
    speedGuess: guess(source.speedGuess, GUESS_OPTIONS.speedGuess),
    returns: nearest(source.returns, RETURN_STOPS, 1.5),
    bottleneck: nearest(source.bottleneck, BOTTLENECK_STOPS, 0),
    prevReturns: prevReturns > 0 ? nearest(prevReturns, RETURN_STOPS, 1.5) : 0,
    prevBottleneck: nearest(source.prevBottleneck, BOTTLENECK_STOPS, 0),
    loopYear: Number(clamp(finite(source.loopYear, 20), 0, HORIZON).toFixed(1)) + 0,
    loopGuess: guess(source.loopGuess, GUESS_OPTIONS.loopGuess),
    survey: member(source.survey, ["2022", "2023"] as const, "2023"),
    ahead: Number(clamp(finite(source.ahead, 5), 0, AHEAD_MAX).toFixed(1)) + 0,
    doubling: nearest(source.doubling, DOUBLING_STOPS, MEASURED_DOUBLING_MONTHS),
    surveyGuess: guess(source.surveyGuess, GUESS_OPTIONS.surveyGuess),
    tags: decodeTags(typeof source.tags === "string" ? source.tags : ""),
    claim: whole(source.claim, 0, 0, CLAIMS.length - 1),
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
  const { placement, tags, ...rest } = readState(parsed as Record<string, unknown>);
  return { ...rest, place: encodePlacement(placement), tags: encodeTags(tags) };
}
