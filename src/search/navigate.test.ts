import { describe, expect, it, vi } from "vitest";
import { navigateToSearchHit } from "./navigate";
import type { SearchHit } from "./types";

function hit(partial: Partial<SearchHit> & Pick<SearchHit, "id" | "kind" | "title">): SearchHit {
  return {
    subtitle: "",
    body: "",
    moduleId: null,
    moduleTitle: null,
    moduleSlug: null,
    groupId: null,
    mode: null,
    section: null,
    stepIndex: null,
    term: null,
    cardLabel: null,
    screen: null,
    settingsTab: null,
    rankBoost: 1,
    score: 1,
    snippet: "",
    ...partial,
  };
}

function storeSpy() {
  return {
    setSearchOpen: vi.fn(),
    setSearchFocus: vi.fn(),
    setView: vi.fn(),
    openModule: vi.fn(),
    setMode: vi.fn(),
    setLearnOpen: vi.fn(),
    setCurrentStep: vi.fn(),
    setSettingsTab: vi.fn(),
  };
}

describe("search navigation", () => {
  it("opens a glossary term in the matching lab", () => {
    const store = storeSpy();
    navigateToSearchHit(
      store,
      hit({
        id: "glossary:attention:query",
        kind: "glossary",
        title: "Query",
        moduleId: "module-08-attention",
        term: "Query",
      }),
    );
    expect(store.setLearnOpen).toHaveBeenCalledWith(true);
    expect(store.openModule).toHaveBeenCalledWith("module-08-attention");
    expect(store.setSearchFocus).toHaveBeenCalledWith({
      kind: "glossary",
      moduleId: "module-08-attention",
      term: "Query",
    });
  });

  it("jumps to a settings tab and a learn section", () => {
    const store = storeSpy();
    navigateToSearchHit(
      store,
      hit({
        id: "screen:settings:cloud",
        kind: "screen",
        title: "Cloud providers",
        screen: "settings",
        settingsTab: "cloud",
      }),
    );
    expect(store.setSettingsTab).toHaveBeenCalledWith("cloud");
    expect(store.setView).toHaveBeenCalledWith("settings");

    navigateToSearchHit(
      store,
      hit({
        id: "learn:attention:plain:what-it-is",
        kind: "learn",
        title: "What it is",
        moduleId: "module-08-attention",
        mode: "plain",
        section: "What it is",
      }),
    );
    expect(store.setMode).toHaveBeenCalledWith("plain");
    expect(store.setLearnOpen).toHaveBeenCalledWith(true);
    expect(store.openModule).toHaveBeenCalledWith("module-08-attention");
  });
});
