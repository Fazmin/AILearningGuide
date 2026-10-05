import { beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./app-store";

const state = () => useAppStore.getState();

beforeEach(() => {
  state().resetLearningData();
});

describe("checkpoint progress", () => {
  it("records the first answer to each question and ignores retries", () => {
    state().markCheckpoint("module-a", 0, false, 3);
    state().markCheckpoint("module-a", 0, true, 3);
    state().markCheckpoint("module-a", 2, true, 3);
    expect(state().progress["module-a"].checkpointResults).toEqual([false, null, true]);
  });

  it("marks the module visited when a question is answered", () => {
    state().markCheckpoint("module-a", 0, true, 1);
    expect(state().progress["module-a"].visited).toBe(true);
  });

  it("lets a learner retake the checkpoint without touching other progress", () => {
    state().toggleModuleComplete("module-a");
    state().markCheckpoint("module-a", 0, false, 3);
    state().resetCheckpoint("module-a");
    expect(state().progress["module-a"].checkpointResults).toEqual([]);
    expect(state().progress["module-a"].completed).toBe(true);
  });
});

describe("importLearningData", () => {
  const exported = (progress: Record<string, unknown>) => ({
    format: "the-ai-guide-export",
    version: 1,
    progress,
    moduleStates: {},
    snapshots: [],
  });

  it("reads an export written before multi-question checkpoints", () => {
    const ok = state().importLearningData(
      exported({
        "module-a": { visited: true, completed: true, checkpointCorrect: true, currentStep: 2, updatedAt: "2026-01-01T00:00:00.000Z" },
        "module-b": { visited: true, completed: false, checkpointCorrect: false, currentStep: 0, updatedAt: "2026-01-01T00:00:00.000Z" },
      }),
    );
    expect(ok).toBe(true);
    expect(state().progress["module-a"].checkpointResults).toEqual([true]);
    expect(state().progress["module-a"].completed).toBe(true);
    expect(state().progress["module-a"].currentStep).toBe(2);
    expect(state().progress["module-b"].checkpointResults).toEqual([]);
  });

  it("reads a current export unchanged", () => {
    state().importLearningData(
      exported({ "module-a": { visited: true, completed: false, checkpointResults: [true, false, null], currentStep: 1, updatedAt: "x" } }),
    );
    expect(state().progress["module-a"].checkpointResults).toEqual([true, false, null]);
  });

  it("drops malformed progress entries and still fills in missing fields", () => {
    state().importLearningData(exported({ bad: "nope", worse: null, arr: [], "module-a": {} }));
    expect(Object.keys(state().progress)).toEqual(["module-a"]);
    expect(state().progress["module-a"]).toMatchObject({ visited: false, completed: false, checkpointResults: [], currentStep: 0 });
  });

  it("rejects data that is not an export", () => {
    expect(state().importLearningData({ format: "other", version: 1 })).toBe(false);
    expect(state().importLearningData(null)).toBe(false);
  });
});
