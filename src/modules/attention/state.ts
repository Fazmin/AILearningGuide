import type { ModuleState } from "@app/module-sdk";
import { MAX_WORDS } from "./attention-math";

/**
 * The lab's saved state. Version 3 adds the three experiment switches (`mask`, `scaling`, `values`).
 * A version 2 payload has none of them, so it keeps its sentence, head and edits and gets the model's
 * own mask, scaling and value vectors.
 */
export const DEFAULT_TEXT = "The animal did not cross the street because it was tired";
export const MAX_CHARACTERS = 64;
export const MODEL_LAYERS = 2;
export const MODEL_HEADS = 4;
export const STAGE_COUNT = 5;
/** The Q and K chips edit this many leading dimensions. */
export const EDIT_DIMENSIONS = 4;
/** Hand-edited or shared values beyond this are clamped; trained Q and K entries are far smaller. */
export const EDIT_LIMIT = 100;

export const VIEWS = ["words", "characters"] as const;
export const MASKS = ["causal", "bidirectional"] as const;
export const SCALINGS = ["scaled", "raw"] as const;
export const VALUE_MODES = ["model", "inspected", "all"] as const;

export type MatrixView = (typeof VIEWS)[number];
export type MaskMode = (typeof MASKS)[number];
export type ScalingMode = (typeof SCALINGS)[number];
export type ValueMode = (typeof VALUE_MODES)[number];

export interface Controls {
  text: string;
  layer: number;
  head: number;
  selected: number;
  inspected: number;
  stage: number;
  editVectors: boolean;
  matrixView: MatrixView;
  mask: MaskMode;
  scaling: ScalingMode;
  values: ValueMode;
  /** Edited leading dimensions of the Q and K chips, keyed `q0`..`q3` and `k0`..`k3`. */
  edits: Record<string, number>;
}

export const initialState: ModuleState = {
  text: DEFAULT_TEXT,
  layer: 1,
  head: 1,
  selected: 8,
  inspected: 1,
  stage: 0,
  editVectors: false,
  matrixView: "words",
  mask: "causal",
  scaling: "scaled",
  values: "model",
};

const asInteger = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const asMember = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

/** Every key validated and clamped, whatever the payload held (share links, snapshots, old versions). */
export function readState(raw: unknown): Controls {
  const source = (typeof raw === "object" && raw !== null ? raw : {}) as Record<string, unknown>;
  const edits: Record<string, number> = {};
  for (const prefix of ["q", "k"]) {
    for (let index = 0; index < EDIT_DIMENSIONS; index += 1) {
      const value = source[`${prefix}${index}`];
      if (typeof value === "number" && Number.isFinite(value)) {
        edits[`${prefix}${index}`] = Math.min(EDIT_LIMIT, Math.max(-EDIT_LIMIT, value));
      }
    }
  }
  return {
    text:
      typeof source.text === "string"
        ? Array.from(source.text).slice(0, MAX_CHARACTERS).join("")
        : DEFAULT_TEXT,
    layer: asInteger(source.layer, 1, 0, MODEL_LAYERS - 1),
    head: asInteger(source.head, 1, 0, MODEL_HEADS - 1),
    selected: asInteger(source.selected, 8, 0, MAX_WORDS - 1),
    inspected: asInteger(source.inspected, 1, 0, MAX_WORDS - 1),
    stage: asInteger(source.stage, 0, 0, STAGE_COUNT - 1),
    editVectors: source.editVectors === true,
    matrixView: asMember(source.matrixView, VIEWS, "words"),
    mask: asMember(source.mask, MASKS, "causal"),
    scaling: asMember(source.scaling, SCALINGS, "scaled"),
    values: asMember(source.values, VALUE_MODES, "model"),
    edits,
  };
}

/** The validated controls as a plain module state, edits included only where the payload had them. */
export function normalizeState(raw: unknown): ModuleState {
  const { edits, ...controls } = readState(raw);
  return { ...controls, ...edits };
}
