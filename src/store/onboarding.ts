/**
 * First-run onboarding rules, kept pure so they can be tested without the store.
 *
 * The tour appears only for a learner who has not finished it AND has nothing to lose or learn
 * from skipping it: no progress, no snapshots, no saved lab state. Anyone with history, and anyone
 * whose device already held persisted state from an earlier build, counts as onboarded.
 */

export interface OnboardingState {
  /** Persisted. True once the tour was finished or skipped, or the learner predates it. */
  onboarded: boolean;
  /** Not persisted. True while the learner replays the tour from Settings > About. */
  onboardingReplay: boolean;
  progress: Record<string, unknown>;
  snapshots: unknown[];
  moduleStates: Record<string, unknown>;
}

/** Version of the persisted store in which `onboarded` was introduced. */
export const ONBOARDING_PERSIST_VERSION = 3;

export function hasLearningHistory(state: Pick<OnboardingState, "progress" | "snapshots" | "moduleStates">) {
  return (
    Object.keys(state.progress).length > 0 ||
    state.snapshots.length > 0 ||
    Object.keys(state.moduleStates).length > 0
  );
}

export function shouldShowOnboarding(state: OnboardingState) {
  if (state.onboardingReplay) return true;
  return !state.onboarded && !hasLearningHistory(state);
}

/**
 * Persist migration step for version 3. Any state written by an earlier version proves the app has
 * already run on this device, so that learner is marked onboarded and is never pushed through the tour.
 * A brand-new device has no stored state at all, so this never runs for it.
 */
export function migrateOnboardedFlag(persisted: unknown, version: number): unknown {
  if (version >= ONBOARDING_PERSIST_VERSION || !persisted || typeof persisted !== "object") return persisted;
  return { ...(persisted as Record<string, unknown>), onboarded: true };
}
