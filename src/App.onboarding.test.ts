import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { modulesById } from "./modules/registry";
import { useAppStore } from "./store/app-store";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const mountApp = async () => {
  await act(async () => {
    root.render(createElement(App));
  });
};
const dialog = () => document.querySelector('[role="dialog"][aria-modal="true"]');

beforeEach(() => {
  vi.stubGlobal(
    "matchMedia",
    (query: string) =>
      ({ matches: false, media: query, addEventListener: () => undefined, removeEventListener: () => undefined }) as unknown as MediaQueryList,
  );
  window.history.replaceState(null, "", "/");
  useAppStore.setState({
    onboarded: false,
    onboardingReplay: false,
    progress: {},
    snapshots: [],
    moduleStates: {},
    view: "home",
    activeModuleId: null,
    searchOpen: false,
    accessibilityOpen: false,
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

describe("onboarding in the app", () => {
  it("greets a first-time learner with the tour, over the app, and the app behind it is inert", async () => {
    await mountApp();
    expect(dialog()).not.toBeNull();
    expect(document.querySelector(".app-shell")!.hasAttribute("inert")).toBe(true);
    expect(document.querySelector(".titlebar")!.hasAttribute("inert")).toBe(false);
  });

  it("lets a learner with progress straight in", async () => {
    useAppStore.setState({
      progress: { a: { visited: true, completed: false, checkpointResults: [], currentStep: 0, updatedAt: "x" } },
    });
    await mountApp();
    expect(dialog()).toBeNull();
    expect(document.querySelector(".app-shell")!.hasAttribute("inert")).toBe(false);
  });

  it("never shows the tour on top of a shared or deep link into a lab", async () => {
    const lab = [...modulesById.values()][0];
    window.history.replaceState(null, "", `/#/${lab.slug}`);
    await mountApp();
    expect(useAppStore.getState().view).toBe("module");
    expect(dialog()).toBeNull();
  });

  it("goes away for good after Escape, and comes back only on request", async () => {
    await mountApp();
    await act(async () => {
      document.querySelector('[role="dialog"] h2')!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(dialog()).toBeNull();
    expect(useAppStore.getState().onboarded).toBe(true);
    await act(async () => useAppStore.getState().replayOnboarding());
    expect(dialog()).not.toBeNull();
  });
});
