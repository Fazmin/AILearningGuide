import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ExplanationMode, ModuleState } from "@app/module-sdk";
import type { SearchFocus } from "@app/search/types";
import {
  checkpointScore,
  migratePersistedState,
  normalizeProgressEntry,
  recordCheckpointResult,
  type CheckpointResults,
} from "./checkpoint-progress";
import { migrateOnboardedFlag } from "./onboarding";

export type ThemePreference = "light" | "dark" | "system";
export type Typeface = "sans" | "elegant";
export type ColorVision = "default" | "deuteranopia" | "protanopia" | "tritanopia";
export type AppView = "home" | "module" | "settings" | "connections";
export type SettingsTab =
  | "appearance"
  | "accessibility"
  | "explanations"
  | "local"
  | "cloud"
  | "data"
  | "about";

export interface ModuleProgress {
  visited: boolean;
  completed: boolean;
  /** First answer to each checkpoint question; `null` is unanswered. */
  checkpointResults: CheckpointResults;
  currentStep: number;
  updatedAt: string;
}

export interface Snapshot {
  id: string;
  moduleId: string;
  name: string;
  state: ModuleState;
  createdAt: string;
}

interface AppState {
  view: AppView;
  activeModuleId: string | null;
  /** The lab selected on the connections map. Not persisted: the map opens clear each launch. */
  connectionsFocusId: string | null;
  settingsTab: SettingsTab;
  searchOpen: boolean;
  searchFocus: SearchFocus | null;
  theme: ThemePreference;
  typeface: Typeface;
  mode: ExplanationMode;
  fontStep: number;
  sidebarCollapsed: boolean;
  learnOpen: boolean;
  learnExpanded: boolean;
  chatOpen: boolean;
  accessibilityOpen: boolean;
  highContrast: boolean;
  reducedMotion: boolean;
  verboseNarration: boolean;
  colorVision: ColorVision;
  moduleStates: Record<string, ModuleState>;
  progress: Record<string, ModuleProgress>;
  snapshots: Snapshot[];
  provider: "none" | "openai" | "anthropic" | "google" | "compatible";
  providerModel: string;
  providerBaseUrl: string;
  localModelReady: boolean;
  /**
   * The first-run tour was finished, skipped, or the learner predates it. Persisted, but never
   * exported or imported: it describes this device, not a learner's data.
   */
  onboarded: boolean;
  /** The tour was reopened from Settings > About. Not persisted. */
  onboardingReplay: boolean;
  setView: (view: AppView) => void;
  openModule: (moduleId: string) => void;
  /** Open the connections map, selecting a lab if given and keeping the last selection if not. */
  openConnections: (moduleId?: string) => void;
  setConnectionsFocus: (moduleId: string | null) => void;
  setSettingsTab: (tab: SettingsTab) => void;
  setSearchOpen: (open: boolean) => void;
  setSearchFocus: (focus: SearchFocus | null) => void;
  setTheme: (theme: ThemePreference) => void;
  setTypeface: (typeface: Typeface) => void;
  setMode: (mode: ExplanationMode) => void;
  setFontStep: (fontStep: number) => void;
  toggleSidebar: () => void;
  setLearnOpen: (open: boolean) => void;
  /** Learn toggle: closed -> open -> expanded -> closed. */
  cycleLearnPanel: () => void;
  setChatOpen: (open: boolean) => void;
  setAccessibilityOpen: (open: boolean) => void;
  setHighContrast: (enabled: boolean) => void;
  setReducedMotion: (enabled: boolean) => void;
  setVerboseNarration: (enabled: boolean) => void;
  setColorVision: (colorVision: ColorVision) => void;
  setModuleState: (moduleId: string, patch: Partial<ModuleState>) => void;
  resetModuleState: (moduleId: string, initialState: ModuleState) => void;
  setCurrentStep: (moduleId: string, step: number) => void;
  markCheckpoint: (moduleId: string, questionIndex: number, correct: boolean, total: number) => void;
  resetCheckpoint: (moduleId: string) => void;
  toggleModuleComplete: (moduleId: string) => void;
  saveSnapshot: (moduleId: string, name: string, state: ModuleState) => Snapshot;
  loadSnapshot: (snapshot: Snapshot) => void;
  deleteSnapshot: (id: string) => void;
  importLearningData: (data: unknown) => boolean;
  resetLearningData: () => void;
  setProvider: (provider: AppState["provider"]) => void;
  setProviderModel: (model: string) => void;
  setProviderBaseUrl: (baseUrl: string) => void;
  setLocalModelReady: (ready: boolean) => void;
  /** Mark the tour done and close it, whether it ran for the first time or as a replay. */
  completeOnboarding: () => void;
  /** Reopen the tour (Settings > About). Leaves `onboarded` alone. */
  replayOnboarding: () => void;
}

const fontSteps = [0.875, 1, 1.125, 1.25, 1.5, 1.75, 2];

export function fontScaleForStep(step: number) {
  return fontSteps[Math.max(0, Math.min(fontSteps.length - 1, step))];
}

function syncSetting(key: string, value: unknown) {
  if (!window.__TAURI_INTERNALS__) return;
  void invoke("set_setting", { key, value: JSON.stringify(value) }).catch(() => {
    // Local persistence remains the offline fallback if the desktop database is unavailable.
  });
}

function syncProgress(moduleId: string, progress: ModuleProgress) {
  if (!window.__TAURI_INTERNALS__) return;
  void invoke("upsert_module_progress", {
    progress: {
      moduleId,
      currentStep: progress.currentStep,
      checkpointScore: checkpointScore(progress.checkpointResults, progress.checkpointResults.length),
      completed: progress.completed,
      updatedAt: progress.updatedAt,
    },
  }).catch(() => undefined);
}

function progressEntry(existing?: ModuleProgress): ModuleProgress {
  return existing ?? {
    visited: false,
    completed: false,
    checkpointResults: [],
    currentStep: 0,
    updatedAt: new Date().toISOString(),
  };
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      view: "home",
      activeModuleId: null,
      connectionsFocusId: null,
      settingsTab: "appearance",
      searchOpen: false,
      searchFocus: null,
      theme: "system",
      typeface: "sans",
      mode: "plain",
      fontStep: 1,
      sidebarCollapsed: false,
      learnOpen: true,
      learnExpanded: false,
      chatOpen: false,
      accessibilityOpen: false,
      highContrast: false,
      reducedMotion: false,
      verboseNarration: false,
      colorVision: "default",
      moduleStates: {},
      progress: {},
      snapshots: [],
      provider: "none",
      providerModel: "",
      providerBaseUrl: "",
      localModelReady: false,
      onboarded: false,
      onboardingReplay: false,

      setView: (view) => set({ view, chatOpen: false }),
      openConnections: (moduleId) =>
        set((state) => ({
          view: "connections",
          chatOpen: false,
          connectionsFocusId: moduleId ?? state.connectionsFocusId,
        })),
      setConnectionsFocus: (connectionsFocusId) => set({ connectionsFocusId }),
      setSettingsTab: (settingsTab) => set({ settingsTab }),
      setSearchOpen: (searchOpen) => set({ searchOpen, accessibilityOpen: searchOpen ? false : get().accessibilityOpen }),
      setSearchFocus: (searchFocus) => set({ searchFocus }),
      openModule: (moduleId) => {
        const existing = get().progress[moduleId];
        set((state) => ({
          view: "module",
          activeModuleId: moduleId,
          chatOpen: false,
          progress: {
            ...state.progress,
            [moduleId]: {
              ...progressEntry(existing),
              visited: true,
              updatedAt: new Date().toISOString(),
            },
          },
        }));
        syncProgress(moduleId, get().progress[moduleId]);
      },
      setTheme: (theme) => {
        set({ theme });
        syncSetting("theme", theme);
      },
      setTypeface: (typeface) => {
        set({ typeface });
        syncSetting("typeface", typeface);
      },
      setMode: (mode) => {
        set({ mode });
        syncSetting("explanation_mode", mode);
      },
      setFontStep: (fontStep) => {
        const next = Math.max(0, Math.min(fontSteps.length - 1, fontStep));
        set({ fontStep: next });
        syncSetting("font_step", next);
      },
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setLearnOpen: (learnOpen) => set(learnOpen ? { learnOpen } : { learnOpen, learnExpanded: false }),
      cycleLearnPanel: () =>
        set((state) => {
          if (!state.learnOpen) return { learnOpen: true, learnExpanded: false };
          if (!state.learnExpanded) return { learnExpanded: true };
          return { learnOpen: false, learnExpanded: false };
        }),
      setChatOpen: (chatOpen) => set({ chatOpen }),
      setAccessibilityOpen: (accessibilityOpen) => set({ accessibilityOpen }),
      setHighContrast: (highContrast) => {
        set({ highContrast });
        syncSetting("high_contrast", highContrast);
      },
      setReducedMotion: (reducedMotion) => {
        set({ reducedMotion });
        syncSetting("reduced_motion", reducedMotion);
      },
      setVerboseNarration: (verboseNarration) => {
        set({ verboseNarration });
        syncSetting("verbose_narration", verboseNarration);
      },
      setColorVision: (colorVision) => {
        set({ colorVision });
        syncSetting("color_vision", colorVision);
      },
      setModuleState: (moduleId, patch) =>
        set((state) => {
          const next: ModuleState = { ...(state.moduleStates[moduleId] ?? {}) };
          Object.entries(patch).forEach(([key, value]) => {
            if (value !== undefined) next[key] = value;
          });
          return {
            moduleStates: {
              ...state.moduleStates,
              [moduleId]: next,
            },
          };
        }),
      resetModuleState: (moduleId, initialState) =>
        set((state) => ({
          moduleStates: { ...state.moduleStates, [moduleId]: { ...initialState } },
          progress: {
            ...state.progress,
            [moduleId]: {
              ...progressEntry(state.progress[moduleId]),
              currentStep: 0,
              updatedAt: new Date().toISOString(),
            },
          },
        })),
      setCurrentStep: (moduleId, currentStep) => {
        set((state) => ({
          progress: {
            ...state.progress,
            [moduleId]: {
              ...progressEntry(state.progress[moduleId]),
              visited: true,
              currentStep,
              updatedAt: new Date().toISOString(),
            },
          },
        }));
        syncProgress(moduleId, get().progress[moduleId]);
      },
      markCheckpoint: (moduleId, questionIndex, correct, total) => {
        set((state) => {
          const current = progressEntry(state.progress[moduleId]);
          return {
            progress: {
              ...state.progress,
              [moduleId]: {
                ...current,
                visited: true,
                checkpointResults: recordCheckpointResult(current.checkpointResults, questionIndex, correct, total),
                updatedAt: new Date().toISOString(),
              },
            },
          };
        });
        syncProgress(moduleId, get().progress[moduleId]);
      },
      resetCheckpoint: (moduleId) => {
        set((state) => ({
          progress: {
            ...state.progress,
            [moduleId]: {
              ...progressEntry(state.progress[moduleId]),
              checkpointResults: [],
              updatedAt: new Date().toISOString(),
            },
          },
        }));
        syncProgress(moduleId, get().progress[moduleId]);
      },
      toggleModuleComplete: (moduleId) => {
        set((state) => {
          const current = progressEntry(state.progress[moduleId]);
          return {
            progress: {
              ...state.progress,
              [moduleId]: {
                ...current,
                visited: true,
                completed: !current.completed,
                updatedAt: new Date().toISOString(),
              },
            },
          };
        });
        syncProgress(moduleId, get().progress[moduleId]);
      },
      saveSnapshot: (moduleId, name, moduleState) => {
        const snapshot: Snapshot = {
          id: crypto.randomUUID(),
          moduleId,
          name,
          state: { ...moduleState },
          createdAt: new Date().toISOString(),
        };
        set((state) => ({ snapshots: [snapshot, ...state.snapshots].slice(0, 50) }));
        if (window.__TAURI_INTERNALS__) {
          void invoke("create_snapshot", {
            snapshot: {
              id: snapshot.id,
              moduleId,
              name,
              stateJson: JSON.stringify(moduleState),
              stateVersion: 1,
            },
          }).catch(() => undefined);
        }
        return snapshot;
      },
      loadSnapshot: (snapshot) =>
        set((state) => ({
          moduleStates: { ...state.moduleStates, [snapshot.moduleId]: { ...snapshot.state } },
          view: "module",
          activeModuleId: snapshot.moduleId,
        })),
      deleteSnapshot: (id) => {
        set((state) => ({ snapshots: state.snapshots.filter((item) => item.id !== id) }));
        if (window.__TAURI_INTERNALS__) {
          void invoke("delete_snapshot", { id }).catch(() => undefined);
        }
      },
      importLearningData: (data) => {
        if (!data || typeof data !== "object") return false;
        const candidate = data as Record<string, unknown>;
        if (candidate.format !== "the-ai-guide-export" || candidate.version !== 1) return false;
        // Exports written before multi-question checkpoints carry `checkpointCorrect`.
        const importedProgress: Record<string, ModuleProgress> =
          candidate.progress && typeof candidate.progress === "object"
            ? Object.fromEntries(
                Object.entries(candidate.progress as Record<string, unknown>)
                  .filter(([, entry]) => entry && typeof entry === "object" && !Array.isArray(entry))
                  .map(([moduleId, entry]) => [
                    moduleId,
                    {
                      ...progressEntry(),
                      ...normalizeProgressEntry(entry as Record<string, unknown>),
                    } as ModuleProgress,
                  ]),
              )
            : {};
        const importedStates =
          candidate.moduleStates && typeof candidate.moduleStates === "object"
            ? (candidate.moduleStates as Record<string, ModuleState>)
            : {};
        const importedSnapshots = Array.isArray(candidate.snapshots)
          ? (candidate.snapshots as Snapshot[]).filter(
              (snapshot) =>
                snapshot &&
                typeof snapshot.id === "string" &&
                typeof snapshot.moduleId === "string" &&
                snapshot.state &&
                typeof snapshot.state === "object",
            )
          : [];
        set({
          progress: importedProgress,
          moduleStates: importedStates,
          snapshots: importedSnapshots.slice(0, 50),
        });
        return true;
      },
      // Resetting empties the history that exempts a learner from the tour, so it also records that
      // they are past it: a reset must never send someone back through first-run onboarding.
      resetLearningData: () => set({ progress: {}, moduleStates: {}, snapshots: [], onboarded: true }),
      setProvider: (provider) => {
        set({ provider });
        syncSetting("provider", provider);
      },
      setProviderModel: (providerModel) => {
        set({ providerModel });
        syncSetting("provider_model", providerModel);
      },
      setProviderBaseUrl: (providerBaseUrl) => {
        set({ providerBaseUrl });
        syncSetting("provider_base_url", providerBaseUrl);
      },
      setLocalModelReady: (localModelReady) => set({ localModelReady }),
      completeOnboarding: () => set({ onboarded: true, onboardingReplay: false }),
      replayOnboarding: () => set({ onboardingReplay: true }),
    }),
    {
      name: "the-ai-guide",
      // Version 2: `checkpointCorrect` (one boolean) became `checkpointResults` (one entry per question).
      // Version 3: added `onboarded`; state from any earlier version marks the learner as onboarded.
      // The steps compose, oldest first, so a version 1 store passes through both.
      version: 3,
      migrate: (persisted, version) =>
        migrateOnboardedFlag(migratePersistedState(persisted, version), version) as AppState,
      partialize: (state) => ({
        theme: state.theme,
        typeface: state.typeface,
        mode: state.mode,
        fontStep: state.fontStep,
        sidebarCollapsed: state.sidebarCollapsed,
        learnOpen: state.learnOpen,
        highContrast: state.highContrast,
        reducedMotion: state.reducedMotion,
        verboseNarration: state.verboseNarration,
        colorVision: state.colorVision,
        moduleStates: state.moduleStates,
        progress: state.progress,
        snapshots: state.snapshots,
        provider: state.provider,
        providerModel: state.providerModel,
        providerBaseUrl: state.providerBaseUrl,
        localModelReady: state.localModelReady,
        onboarded: state.onboarded,
      }),
    },
  ),
);
