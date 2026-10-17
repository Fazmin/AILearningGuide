import type { ExplanationMode } from "./types";

/** Every explanation mode, in the order the interface offers them: easiest to read first. */
export const explanationModes = ["simple", "plain", "standard"] as const satisfies readonly ExplanationMode[];

export const explanationModeLabel: Record<ExplanationMode, string> = {
  simple: "Simple",
  plain: "Plain",
  standard: "Standard",
};
