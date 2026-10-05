import { describe, expect, it } from "vitest";
import {
  checkpointScore,
  checkpointTally,
  migratePersistedState,
  normalizeProgressEntry,
  recordCheckpointResult,
} from "./checkpoint-progress";

describe("recordCheckpointResult", () => {
  it("pads to the module's question count and records the answer", () => {
    expect(recordCheckpointResult(undefined, 1, true, 3)).toEqual([null, true, null]);
  });

  it("counts only the first answer to a question", () => {
    const first = recordCheckpointResult(undefined, 0, false, 3);
    const retry = recordCheckpointResult(first, 0, true, 3);
    expect(retry).toEqual([false, null, null]);
  });

  it("grows when a module gains questions after progress was stored", () => {
    expect(recordCheckpointResult([true], 3, false, 4)).toEqual([true, null, null, false]);
  });
});

describe("checkpointTally and checkpointScore", () => {
  it("reports answered, correct, and the fraction correct first time", () => {
    const results = [true, false, null];
    expect(checkpointTally(results, 3)).toEqual({ answered: 2, correct: 1, total: 3 });
    expect(checkpointScore(results, 3)).toBeCloseTo(1 / 3);
  });

  it("treats missing progress as nothing answered", () => {
    expect(checkpointTally(undefined, 3)).toEqual({ answered: 0, correct: 0, total: 3 });
    expect(checkpointScore(undefined, 3)).toBe(0);
  });

  it("ignores stored entries beyond the module's current question count", () => {
    expect(checkpointTally([true, true, true, true], 2)).toEqual({ answered: 2, correct: 2, total: 2 });
  });
});

describe("normalizeProgressEntry", () => {
  it("maps the version 1 boolean onto the only question it could have answered", () => {
    expect(normalizeProgressEntry({ visited: true, checkpointCorrect: true })).toEqual({
      visited: true,
      checkpointResults: [true],
    });
    expect(normalizeProgressEntry({ visited: true, checkpointCorrect: false })).toEqual({
      visited: true,
      checkpointResults: [],
    });
  });

  it("keeps valid results and discards junk entries", () => {
    const entry = normalizeProgressEntry({ checkpointResults: [true, "yes", null, false, 1] });
    expect(entry.checkpointResults).toEqual([true, null, null, false, null]);
  });

  it("falls back to empty results for a malformed field", () => {
    expect(normalizeProgressEntry({ checkpointResults: "all" }).checkpointResults).toEqual([]);
  });
});

describe("migratePersistedState", () => {
  it("upgrades version 1 progress and leaves everything else alone", () => {
    const migrated = migratePersistedState(
      {
        theme: "dark",
        progress: {
          "module-a": { completed: true, checkpointCorrect: true },
          "module-b": { completed: false, checkpointCorrect: false },
        },
      },
      1,
    ) as { theme: string; progress: Record<string, { checkpointResults: unknown; checkpointCorrect?: unknown }> };
    expect(migrated.theme).toBe("dark");
    expect(migrated.progress["module-a"].checkpointResults).toEqual([true]);
    expect(migrated.progress["module-b"].checkpointResults).toEqual([]);
    expect(migrated.progress["module-a"]).not.toHaveProperty("checkpointCorrect");
  });

  it("does not touch current-version state or non-objects", () => {
    const current = { progress: { a: { checkpointResults: [true] } } };
    expect(migratePersistedState(current, 2)).toBe(current);
    expect(migratePersistedState(null, 1)).toBeNull();
  });
});
