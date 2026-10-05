import type { ModuleState } from "@app/module-sdk";
import { PRESETS, presetById } from "./host";

/**
 * Both call boxes are free text that a shared link or a hand-edited snapshot can fill with anything,
 * so they are length-capped. The longest authored call is 80 characters; 600 leaves room for a long
 * email body (the schema allows 200) and for experiments.
 */
export const DRAFT_MAX = 600;

/**
 * Version 1 stored { task: "hours" | "add" | "mail", step: 0-3 } over canned strings.
 * Version 2 stored { preset, draft, approve }.
 * Version 3 adds `retry` (the second call of an error-then-retry preset) and `turn` (which call the
 * pipeline shows). Neither needs translating: a version 2 payload keeps its meaning with retry empty.
 */
export const initialState: ModuleState = {
  preset: "valid",
  draft: PRESETS[0].output,
  retry: "",
  approve: "no",
  turn: "1",
};

const V1_TASKS: Record<string, string> = { hours: "valid", add: "note", mail: "email" };

export const cappedText = (value: unknown, fallback: string) =>
  typeof value === "string" ? value.slice(0, DRAFT_MAX) : fallback;

/** Rebuilds a state from untrusted input: unknown keys dropped, every field validated. */
export function sanitizeState(value: unknown): ModuleState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initialState };
  const source = value as Record<string, unknown>;
  const fromV1 = typeof source.task === "string" ? V1_TASKS[source.task] : undefined;
  const presetId =
    fromV1 ?? (typeof source.preset === "string" && PRESETS.some((item) => item.id === source.preset) ? source.preset : "valid");
  const preset = presetById(presetId);
  const draft = fromV1 || typeof source.draft !== "string" ? preset.output : cappedText(source.draft, preset.output);
  // Only the error-then-retry presets have a second call; any other preset ignores a stray one.
  const retry = preset.retry === undefined ? "" : cappedText(source.retry, preset.retry);
  return {
    preset: presetId,
    draft,
    retry,
    approve: source.approve === "yes" ? "yes" : "no",
    turn: source.turn === "2" ? "2" : "1",
  };
}
