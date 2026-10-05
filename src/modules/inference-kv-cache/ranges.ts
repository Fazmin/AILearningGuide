/** Control ranges shared by Explore.tsx and hydrateState. */
export const PROMPT_RANGE = { min: 3, max: 14 };
export const GENERATED_RANGE = { min: 0, max: 10 };
/** Reference-model context length, in tokens, stepped in powers of two. */
export const CONTEXT_RANGE = { min: 512, max: 131072 };
export const SEQUENCE_RANGE = { min: 1, max: 64 };
