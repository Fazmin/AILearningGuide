import type { LabScene, ModuleState } from "@app/module-sdk";
import { normalizeHeading } from "./learn-sections";

/**
 * Where each lab scene's card goes in the lesson being shown.
 *
 * A scene lists headings in order of preference, because the reading modes do not share every
 * subsection: it goes under the first one this lesson has. A heading holds one card, so the first
 * scene to claim it keeps it. Keys are normalized headings.
 */
export function placeScenes(scenes: LabScene[], headings: string[]) {
  const present = new Set(headings.map(normalizeHeading));
  const placed = new Map<string, LabScene>();
  for (const scene of scenes) {
    const heading = scene.sections.map(normalizeHeading).find((section) => present.has(section));
    if (heading && !placed.has(heading)) placed.set(heading, scene);
  }
  return placed;
}

/** The lab state a scene shows: its settings over the module's defaults, whatever the reader had before. */
export function sceneState(initialState: ModuleState, scene: LabScene): ModuleState {
  const state: ModuleState = { ...initialState };
  for (const [key, value] of Object.entries(scene.state ?? {})) {
    if (value !== undefined) state[key] = value;
  }
  return state;
}
