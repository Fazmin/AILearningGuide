import type { SettingsTab } from "@app/store/app-store";
import type { SearchHit } from "./types";
import { focusFromDocument } from "./types";

const settingsTabs: SettingsTab[] = [
  "appearance",
  "accessibility",
  "explanations",
  "local",
  "cloud",
  "data",
  "about",
];

export function asSettingsTab(value: string | null | undefined): SettingsTab | null {
  return value && settingsTabs.includes(value as SettingsTab) ? (value as SettingsTab) : null;
}

interface SearchNavigation {
  setSearchOpen: (open: boolean) => void;
  setSearchFocus: (focus: ReturnType<typeof focusFromDocument>) => void;
  setView: (view: "home" | "module" | "settings") => void;
  openModule: (moduleId: string) => void;
  setMode: (mode: "plain" | "standard") => void;
  setLearnOpen: (open: boolean) => void;
  setCurrentStep: (moduleId: string, step: number) => void;
  setSettingsTab: (tab: SettingsTab) => void;
}

export function navigateToSearchHit(store: SearchNavigation, hit: SearchHit) {
  const focus = focusFromDocument(hit);
  store.setSearchOpen(false);
  store.setSearchFocus(focus);
  if (!focus) return;

  if (focus.kind === "screen") {
    if (focus.screen === "settings") {
      const tab = asSettingsTab(focus.settingsTab);
      if (tab) store.setSettingsTab(tab);
      store.setView("settings");
      return;
    }
    store.setView("home");
    return;
  }

  if (focus.kind === "learn") {
    store.setMode(focus.mode);
    store.setLearnOpen(true);
  }
  if (focus.kind === "glossary" || focus.kind === "checkpoint") {
    store.setLearnOpen(true);
  }
  if (focus.kind === "step") {
    store.setCurrentStep(focus.moduleId, focus.stepIndex);
  }
  store.openModule(focus.moduleId);
}
