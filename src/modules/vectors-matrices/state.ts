import type { ModuleState } from "@app/module-sdk";
import { PIECES, PUSH_IDS, ROUNDS, SEQUENCE_IDS, WORD_IDS, type Mat, type Vec } from "./math";

/**
 * The lab's saved state. One followed word runs through the first four cards; every other key
 * belongs to one card. A predict key holds the picked option, or -1 before the reader has guessed.
 */
export const initialState: ModuleState = {
  word: "horse",
  tipX: 0.9,
  tipY: 1.2,
  push: "bigger",
  times: 1,
  other: "whale",
  turn: 0,
  stretch: 1,
  predictDot: -1,
  ma: 1,
  mb: 0.5,
  mc: 0,
  md: 1,
  first: "rotate",
  second: "stretch",
  stage: 2,
  predictOrder: -1,
  keep: 1,
  undo: "leave",
  predictUndo: -1,
  forgetView: "flat",
  keepFast: 1,
  pieces: 2,
  predictPieces: -1,
  round: 0,
  ca: 1,
  cb: 0,
  cc: 0,
  cd: 1,
  solved: 0,
  hint: 0,
};

export const UNDOS = ["leave", "stretch"] as const;
/** The rank card's two pictures: the flat map of two numbers per word, and three numbers per word in space. */
export const FORGET_VIEWS = ["flat", "space"] as const;

const member = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

const bounded = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const whole = (value: unknown, fallback: number, low: number, high: number) =>
  Math.round(bounded(value, fallback, low, high));

/** Slider ranges, shared by the controls and the state check so they cannot drift apart. */
export const RANGES = {
  tipX: [-3, 3],
  tipY: [-3, 3],
  times: [-2, 2],
  turn: [0, 360],
  stretch: [0.5, 1.5],
  matrix: [-2, 2],
  keep: [0, 1],
  keepFast: [0, 1],
} as const;

export interface LabState {
  word: string;
  tip: Vec;
  push: string;
  times: number;
  other: string;
  turn: number;
  stretch: number;
  predictDot: number;
  matrix: Mat;
  first: string;
  second: string;
  stage: number;
  predictOrder: number;
  keep: number;
  undo: (typeof UNDOS)[number];
  predictUndo: number;
  forgetView: (typeof FORGET_VIEWS)[number];
  keepFast: number;
  pieces: number;
  predictPieces: number;
  round: number;
  aim: Mat;
  solved: number;
  hint: boolean;
}

/** Read any stored object as valid lab state: unknown keys are dropped, bad values fall back or clamp. */
export const cleanState = (raw: Record<string, unknown>): ModuleState => {
  const d = initialState as Record<string, number | string>;
  const num = (key: string, low: number, high: number) => bounded(raw[key], d[key] as number, low, high);
  const int = (key: string, low: number, high: number) => whole(raw[key], d[key] as number, low, high);
  return {
    word: member(raw.word, WORD_IDS, d.word as string),
    tipX: num("tipX", ...RANGES.tipX),
    tipY: num("tipY", ...RANGES.tipY),
    push: member(raw.push, PUSH_IDS, d.push as string),
    times: num("times", ...RANGES.times),
    other: member(raw.other, WORD_IDS, d.other as string),
    turn: num("turn", ...RANGES.turn),
    stretch: num("stretch", ...RANGES.stretch),
    predictDot: int("predictDot", -1, 2),
    ma: num("ma", ...RANGES.matrix),
    mb: num("mb", ...RANGES.matrix),
    mc: num("mc", ...RANGES.matrix),
    md: num("md", ...RANGES.matrix),
    first: member(raw.first, SEQUENCE_IDS, d.first as (typeof SEQUENCE_IDS)[number]),
    second: member(raw.second, SEQUENCE_IDS, d.second as (typeof SEQUENCE_IDS)[number]),
    stage: int("stage", 0, 2),
    predictOrder: int("predictOrder", -1, 2),
    keep: num("keep", ...RANGES.keep),
    undo: member(raw.undo, UNDOS, "leave"),
    predictUndo: int("predictUndo", -1, 2),
    forgetView: member(raw.forgetView, FORGET_VIEWS, "flat"),
    keepFast: num("keepFast", ...RANGES.keepFast),
    pieces: int("pieces", 1, PIECES.length),
    predictPieces: int("predictPieces", -1, 2),
    round: int("round", 0, ROUNDS.length - 1),
    ca: num("ca", ...RANGES.matrix),
    cb: num("cb", ...RANGES.matrix),
    cc: num("cc", ...RANGES.matrix),
    cd: num("cd", ...RANGES.matrix),
    solved: int("solved", 0, 2 ** ROUNDS.length - 1),
    hint: int("hint", 0, 1),
  };
};

/** The state as the cards use it. Safe on raw, zeroed or hand-edited state. */
export const readState = (state: ModuleState): LabState => {
  const clean = cleanState(state);
  const n = (key: string) => clean[key] as number;
  const t = (key: string) => clean[key] as string;
  return {
    word: t("word"),
    tip: [n("tipX"), n("tipY")],
    push: t("push"),
    times: n("times"),
    other: t("other"),
    turn: n("turn"),
    stretch: n("stretch"),
    predictDot: n("predictDot"),
    matrix: [n("ma"), n("mb"), n("mc"), n("md")],
    first: t("first"),
    second: t("second"),
    stage: n("stage"),
    predictOrder: n("predictOrder"),
    keep: n("keep"),
    undo: t("undo") as (typeof UNDOS)[number],
    predictUndo: n("predictUndo"),
    forgetView: t("forgetView") as (typeof FORGET_VIEWS)[number],
    keepFast: n("keepFast"),
    pieces: n("pieces"),
    predictPieces: n("predictPieces"),
    round: n("round"),
    aim: [n("ca"), n("cb"), n("cc"), n("cd")],
    solved: n("solved"),
    hint: n("hint") === 1,
  };
};

export const serializeState = (state: ModuleState) => JSON.stringify(state);

/**
 * Accepts the current shape and every older one. Versions 1 to 3 belonged to "The math you need"
 * (keys such as `topic`, `ax`, `m00`, `optW`, `probIdea`): none of those settings exists any more,
 * so they are dropped and the reader starts from this lab's defaults.
 */
export const hydrateState = (value: string): ModuleState => {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    return cleanState(parsed as Record<string, unknown>);
  } catch {
    return { ...initialState };
  }
};
