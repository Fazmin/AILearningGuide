import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./app-store";
import {
  hasLearningHistory,
  migrateOnboardedFlag,
  shouldShowOnboarding,
  type OnboardingState,
} from "./onboarding";

const state = () => useAppStore.getState();

const fresh = (overrides: Partial<OnboardingState> = {}): OnboardingState => ({
  onboarded: false,
  onboardingReplay: false,
  progress: {},
  snapshots: [],
  moduleStates: {},
  ...overrides,
});

const STORAGE_KEY = "the-ai-guide";

/** Put the store back to a device that has never run the app. */
const freshDevice = () => {
  localStorage.removeItem(STORAGE_KEY);
  useAppStore.setState({
    onboarded: false,
    onboardingReplay: false,
    progress: {},
    snapshots: [],
    moduleStates: {},
    theme: "system",
  });
};

beforeEach(freshDevice);
afterEach(() => localStorage.removeItem(STORAGE_KEY));

describe("hasLearningHistory", () => {
  it("is false for a learner with nothing saved", () => {
    expect(hasLearningHistory(fresh())).toBe(false);
  });

  it("is true for any progress, snapshot, or saved lab state", () => {
    expect(hasLearningHistory(fresh({ progress: { a: {} } }))).toBe(true);
    expect(hasLearningHistory(fresh({ snapshots: [{}] }))).toBe(true);
    expect(hasLearningHistory(fresh({ moduleStates: { a: { x: 1 } } }))).toBe(true);
  });
});

describe("shouldShowOnboarding", () => {
  it("shows for a fresh learner who has not finished it", () => {
    expect(shouldShowOnboarding(fresh())).toBe(true);
  });

  it("never shows to a learner who already has history, even if the flag is unset", () => {
    expect(shouldShowOnboarding(fresh({ progress: { a: {} } }))).toBe(false);
    expect(shouldShowOnboarding(fresh({ snapshots: [{}] }))).toBe(false);
    expect(shouldShowOnboarding(fresh({ moduleStates: { a: {} } }))).toBe(false);
  });

  it("does not show once it has been finished or skipped", () => {
    expect(shouldShowOnboarding(fresh({ onboarded: true }))).toBe(false);
  });

  it("shows on replay whatever the history, and only because replay was asked for", () => {
    expect(shouldShowOnboarding(fresh({ onboarded: true, progress: { a: {} }, onboardingReplay: true }))).toBe(true);
  });
});

describe("migrateOnboardedFlag", () => {
  it("marks state written by any earlier version as onboarded and keeps the rest", () => {
    const migrated = migrateOnboardedFlag({ theme: "dark", progress: {} }, 2) as Record<string, unknown>;
    expect(migrated).toMatchObject({ theme: "dark", onboarded: true });
    expect((migrateOnboardedFlag({ theme: "dark" }, 1) as Record<string, unknown>).onboarded).toBe(true);
  });

  it("leaves current-version state, including an explicit false, and non-objects alone", () => {
    const current = { onboarded: false };
    expect(migrateOnboardedFlag(current, 3)).toBe(current);
    expect(migrateOnboardedFlag(null, 2)).toBeNull();
    expect(migrateOnboardedFlag("junk", 2)).toBe("junk");
  });
});

describe("the persisted store", () => {
  it("starts a device that has never run the app as not onboarded", () => {
    expect(useAppStore.getInitialState().onboarded).toBe(false);
    expect(useAppStore.getInitialState().onboardingReplay).toBe(false);
  });

  it("marks a version 2 learner onboarded when it loads, without touching their data", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: { theme: "dark", progress: { "module-a": { visited: true, completed: true, currentStep: 1, checkpointResults: [true], updatedAt: "x" } } },
        version: 2,
      }),
    );
    await useAppStore.persist.rehydrate();
    expect(state().onboarded).toBe(true);
    expect(state().theme).toBe("dark");
    expect(state().progress["module-a"].completed).toBe(true);
    expect(shouldShowOnboarding(state())).toBe(false);
  });

  it("also marks a version 2 learner with nothing saved as onboarded", async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ state: { theme: "light" }, version: 2 }));
    await useAppStore.persist.rehydrate();
    expect(state().onboarded).toBe(true);
  });

  it("still runs the checkpoint migration first, so a version 1 store passes through both steps", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          progress: {
            "module-a": { visited: true, completed: false, checkpointCorrect: true, currentStep: 0, updatedAt: "x" },
            "module-b": { visited: true, completed: false, checkpointCorrect: false, currentStep: 0, updatedAt: "x" },
          },
        },
        version: 1,
      }),
    );
    await useAppStore.persist.rehydrate();
    expect(state().progress["module-a"].checkpointResults).toEqual([true]);
    expect(state().progress["module-b"].checkpointResults).toEqual([]);
    expect(state().progress["module-a"]).not.toHaveProperty("checkpointCorrect");
    expect(state().onboarded).toBe(true);
  });

  it("keeps a current-version learner's choice as stored", async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ state: { onboarded: false }, version: 3 }));
    await useAppStore.persist.rehydrate();
    expect(state().onboarded).toBe(false);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ state: { onboarded: true }, version: 3 }));
    await useAppStore.persist.rehydrate();
    expect(state().onboarded).toBe(true);
  });

  it("persists the flag but not the replay switch", () => {
    state().completeOnboarding();
    state().replayOnboarding();
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    expect(stored.version).toBe(3);
    expect(stored.state.onboarded).toBe(true);
    expect(stored.state).not.toHaveProperty("onboardingReplay");
  });
});

describe("onboarding actions", () => {
  it("completing marks it done and closes a replay", () => {
    state().replayOnboarding();
    expect(state().onboardingReplay).toBe(true);
    state().completeOnboarding();
    expect(state().onboarded).toBe(true);
    expect(state().onboardingReplay).toBe(false);
  });

  it("replaying leaves the finished flag alone", () => {
    state().completeOnboarding();
    state().replayOnboarding();
    expect(state().onboarded).toBe(true);
    expect(shouldShowOnboarding(state())).toBe(true);
    state().completeOnboarding();
    expect(shouldShowOnboarding(state())).toBe(false);
  });

  it("does not bring the tour back after a reset of learning data", () => {
    state().openModule("module-a");
    expect(state().onboarded).toBe(false);
    expect(shouldShowOnboarding(state())).toBe(false);
    state().resetLearningData();
    expect(state().progress).toEqual({});
    expect(shouldShowOnboarding(state())).toBe(false);
  });
});

describe("import and export", () => {
  it("does not read the flag from an import, old or new", () => {
    const base = { format: "the-ai-guide-export", version: 1, progress: {}, moduleStates: {}, snapshots: [] };
    expect(state().importLearningData({ ...base, onboarded: true })).toBe(true);
    expect(state().onboarded).toBe(false);
    state().completeOnboarding();
    expect(state().importLearningData({ ...base, onboarded: false, preferences: { onboarded: false } })).toBe(true);
    expect(state().onboarded).toBe(true);
  });

  it("imports an export written before the flag existed, and the imported progress exempts the learner", () => {
    const legacy = {
      format: "the-ai-guide-export",
      version: 1,
      exportedAt: "2026-01-01T00:00:00.000Z",
      progress: { "module-a": { visited: true, completed: true, checkpointCorrect: true, currentStep: 2, updatedAt: "x" } },
      snapshots: [],
      moduleStates: {},
      preferences: { theme: "dark", mode: "plain" },
    };
    expect(state().importLearningData(legacy)).toBe(true);
    expect(state().progress["module-a"].checkpointResults).toEqual([true]);
    expect(shouldShowOnboarding(state())).toBe(false);
  });
});
