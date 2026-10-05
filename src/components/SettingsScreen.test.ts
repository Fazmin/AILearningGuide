import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAppStore } from "@app/store/app-store";
import { shouldShowOnboarding } from "@app/store/onboarding";
import { SettingsScreen } from "./SettingsScreen";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const state = () => useAppStore.getState();

let container: HTMLDivElement;
let root: Root;

const mount = (tab: "about" | "data") => {
  useAppStore.setState({ settingsTab: tab, view: "settings" });
  act(() => {
    root.render(createElement(SettingsScreen));
  });
};
const buttonByText = (text: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((button) => button.textContent?.includes(text));
const click = (element: Element | undefined) =>
  act(() => {
    element!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

beforeEach(() => {
  useAppStore.setState({
    onboarded: true,
    onboardingReplay: false,
    progress: {},
    snapshots: [],
    moduleStates: {},
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

describe("Settings > About", () => {
  it("offers to replay the welcome tour", () => {
    mount("about");
    const replay = buttonByText("Replay the welcome tour");
    expect(replay).toBeDefined();
    click(replay);
    expect(state().onboardingReplay).toBe(true);
    expect(state().onboarded).toBe(true);
    expect(shouldShowOnboarding(state())).toBe(true);
  });

  it("carries the licences and attributions, with the app's copyright line", () => {
    mount("about");
    expect(container.querySelector("#licences-title")!.textContent).toBe("Licences and attributions");
    expect(container.textContent).toContain("Copyright © 2026 Discover AI contributors");
    expect(container.querySelector(".licence-entry")).not.toBeNull();
  });
});

describe("Settings > Data", () => {
  const exported = async () => {
    const blobs: Blob[] = [];
    (URL as unknown as { createObjectURL: (blob: Blob) => string }).createObjectURL = (blob) => {
      blobs.push(blob);
      return "blob:test";
    };
    (URL as unknown as { revokeObjectURL: () => void }).revokeObjectURL = () => undefined;
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    click(buttonByText("Export JSON"));
    // jsdom's Blob has no text(), so read it the way a browser would for a file input.
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(blobs[0]);
    });
    return JSON.parse(text) as Record<string, unknown>;
  };

  it("exports learning data without the onboarding flag, in the format old versions read", async () => {
    useAppStore.setState({ onboarded: true });
    state().openModule("module-a");
    mount("data");
    const data = await exported();
    expect(data).toMatchObject({ format: "the-ai-guide-export", version: 1 });
    expect(Object.keys(data).sort()).toEqual(["exportedAt", "format", "moduleStates", "preferences", "progress", "snapshots", "version"]);
    expect(JSON.stringify(data)).not.toContain("onboard");
    expect(state().importLearningData(data)).toBe(true);
  });

  it("resets learning data without sending the learner back through the tour", () => {
    state().openModule("module-a");
    useAppStore.setState({ onboarded: false });
    mount("data");
    vi.spyOn(window, "confirm").mockReturnValue(true);
    click(buttonByText("Reset learning data"));
    expect(state().progress).toEqual({});
    expect(state().snapshots).toEqual([]);
    expect(state().onboarded).toBe(true);
    expect(shouldShowOnboarding(state())).toBe(false);
  });
});
