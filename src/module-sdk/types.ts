import type { ComponentType } from "react";

export type ExplanationMode = "standard" | "plain";
export type ModuleGroupId =
  | "groundwork"
  | "foundations"
  | "inside-models"
  | "building-with-models"
  | "frontiers"
  | "training-adapting";
export type ModuleState = Record<string, string | number | boolean | string[]>;

export interface GlossaryTerm {
  term: string;
  definition: string;
}

/** A published source the lesson draws on, listed at the bottom of the Learn panel in both modes. */
export interface Reference {
  /** As printed on the work, with "et al." after the third author. */
  authors: string;
  title: string;
  /** Journal with volume and pages, publisher, or site. */
  source: string;
  year: number;
  url: string;
  /** One plain sentence on what the source adds to this lesson. */
  note: string;
}

export interface Checkpoint {
  prompt: string;
  options: string[];
  answer: number;
  explanation: string;
  /** Index into the module's `objectives` that this question tests. */
  objective?: number;
}

export interface ModuleContext {
  state: ModuleState;
  setState: (patch: Partial<ModuleState>) => void;
  currentStep: number;
  mode: ExplanationMode;
  narrate: (message: string) => void;
}

export interface ModuleDefinition {
  id: string;
  slug: string;
  title: string;
  group: ModuleGroupId;
  order: number;
  icon: string;
  accent: string;
  prerequisites: string[];
  estimatedMinutes: number;
  steps: string[];
  stepInstructions: string[];
  stateVersion: number;
  tagline: string;
  objectives: string[];
  glossary: GlossaryTerm[];
  /** Further reading: seven to fifteen verified sources (registry.test.ts enforces the count). */
  references?: Reference[];
  /** One question, or a list. Read it through `checkpointQuestions`. */
  checkpoint: Checkpoint | Checkpoint[];
  initialState: ModuleState;
  Explore: ComponentType<ModuleContext>;
  serializeState: (state: ModuleState) => string;
  hydrateState: (value: string) => ModuleState;
}

export interface MdxModule {
  default: ComponentType;
}
