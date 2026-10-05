import type { ModuleState } from "@app/module-sdk";
import { CAP_MAX, TASKS, type TaskId } from "./loop";
import {
  CATCH_RANGE,
  DEFAULT_CATCH,
  DEFAULT_GATE,
  DEFAULT_RUN_STEPS,
  DEFAULT_STEP_SUCCESS,
  GATE_RANGE,
  RUN_STEPS_RANGE,
  STEP_SUCCESS_RANGE,
} from "./reliability";

/**
 * Version 1 replayed authored Observe/Plan/Act rows; its tick indexed those rows, so it is reset.
 * Version 2 stored { task, stepCap, tick, guard }.
 * Version 3 adds the "repair" task id and the four Reliability controls. A version 2 payload needs
 * no translation: the new keys take their defaults and every old value keeps its meaning.
 */
export const initialState: ModuleState = {
  task: "hours",
  stepCap: 4,
  tick: 99,
  guard: "off",
  stepSuccess: DEFAULT_STEP_SUCCESS,
  runSteps: DEFAULT_RUN_STEPS,
  gateAfter: DEFAULT_GATE,
  catchRate: DEFAULT_CATCH,
};

/** A finite number rounded to a whole value and clamped to [low, high], or the fallback. */
export function boundedInteger(value: unknown, low: number, high: number, fallback: number): number {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
}

export const asTask = (value: unknown): TaskId =>
  typeof value === "string" && (Object.keys(TASKS) as string[]).includes(value) ? (value as TaskId) : "hours";

/** The reviewer's catch rate moves in steps of five, like its slider. */
export const boundedCatch = (value: unknown) =>
  Math.round(boundedInteger(value, CATCH_RANGE.min, CATCH_RANGE.max, DEFAULT_CATCH) / 5) * 5;

/** Rebuilds a state from untrusted input: unknown keys dropped, every number clamped to its slider. */
export function sanitizeState(value: unknown): ModuleState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initialState };
  const source = value as Record<string, unknown>;
  const isV1 = !("guard" in source);
  return {
    task: asTask(source.task),
    stepCap: boundedInteger(source.stepCap, 1, CAP_MAX, 4),
    tick: isV1 ? 99 : boundedInteger(source.tick, 0, 99, 99),
    guard: source.guard === "on" ? "on" : "off",
    stepSuccess: boundedInteger(source.stepSuccess, STEP_SUCCESS_RANGE.min, STEP_SUCCESS_RANGE.max, DEFAULT_STEP_SUCCESS),
    runSteps: boundedInteger(source.runSteps, RUN_STEPS_RANGE.min, RUN_STEPS_RANGE.max, DEFAULT_RUN_STEPS),
    gateAfter: boundedInteger(source.gateAfter, GATE_RANGE.min, GATE_RANGE.max, DEFAULT_GATE),
    catchRate: boundedCatch(source.catchRate),
  };
}
