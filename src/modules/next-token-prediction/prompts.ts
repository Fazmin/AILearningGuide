/**
 * The lab's two worked prompts. Earlier releases opened on the famous line below, which is in the
 * transformer's training text (and in the GRU's and the bigram's), so the three models were being
 * compared on text one of them had seen. The default is now a twin of that line, one name swapped,
 * which is not anywhere in the corpus (quoted.test.ts checks this against models/data when the
 * corpus is present).
 */
export const DEFAULT_PROMPT = "O Juliet, Juliet! wherefore art thou J";

/** Appears once in Tiny Shakespeare, at character 478,139 of 1,115,394: inside the first 90% the transformer trained on. */
export const TRAINING_LINE = "O Romeo, Romeo! wherefore art thou R";
export const TRAINING_LINE_OFFSET = 478_139;

/** What the lab can honestly say about whether a prompt was in the models' training text. */
export type Provenance = "unseen" | "training" | "unknown";

/**
 * The corpus is not shipped with the app, so the lab cannot search it: it knows only the two
 * prompts above, and says so for anything else.
 */
export function provenanceOf(prompt: string): Provenance {
  if (prompt === DEFAULT_PROMPT) return "unseen";
  if (prompt === TRAINING_LINE) return "training";
  return "unknown";
}
