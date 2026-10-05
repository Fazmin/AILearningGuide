/**
 * Checkpoint results per question: `true` or `false` is the learner's FIRST answer, `null`
 * means unanswered. Only the first answer counts, so retrying a question to read its
 * explanation never inflates the score.
 */
export type CheckpointResults = Array<boolean | null>;

export function recordCheckpointResult(
  existing: CheckpointResults | undefined,
  index: number,
  correct: boolean,
  total: number,
): CheckpointResults {
  const size = Math.max(total, index + 1, existing?.length ?? 0);
  const next: CheckpointResults = Array.from({ length: size }, (_, position) => existing?.[position] ?? null);
  if (next[index] === null) next[index] = correct;
  return next;
}

export function checkpointTally(results: CheckpointResults | undefined, total: number) {
  const answered = Math.min(total, (results ?? []).slice(0, total).filter((item) => item !== null).length);
  const correct = (results ?? []).slice(0, total).filter((item) => item === true).length;
  return { answered, correct, total };
}

/** Fraction of the module's questions answered correctly first time, for the desktop database. */
export function checkpointScore(results: CheckpointResults | undefined, total: number) {
  return total > 0 ? checkpointTally(results, total).correct / total : 0;
}

function cleanResults(value: unknown): CheckpointResults | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.slice(0, 64).map((item) => (item === true || item === false ? item : null));
}

/**
 * Brings a stored or imported progress entry up to the current shape. Version 1 stored one
 * boolean, `checkpointCorrect`; the only question it could have answered is the first one.
 */
export function normalizeProgressEntry<T extends Record<string, unknown>>(entry: T): Omit<T, "checkpointCorrect"> & {
  checkpointResults: CheckpointResults;
} {
  const { checkpointCorrect, ...rest } = entry as T & { checkpointCorrect?: unknown };
  const results =
    cleanResults(rest.checkpointResults) ?? (checkpointCorrect === true ? [true] : []);
  return { ...rest, checkpointResults: results };
}

export function migratePersistedState(persisted: unknown, version: number): unknown {
  if (version >= 2 || !persisted || typeof persisted !== "object") return persisted;
  const state = persisted as { progress?: Record<string, Record<string, unknown>> };
  if (!state.progress || typeof state.progress !== "object") return persisted;
  return {
    ...state,
    progress: Object.fromEntries(
      Object.entries(state.progress).map(([moduleId, entry]) => [
        moduleId,
        entry && typeof entry === "object" ? normalizeProgressEntry(entry) : entry,
      ]),
    ),
  };
}
