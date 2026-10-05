import { readFileSync } from "node:fs";
import path from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ONBOARDING_SAMPLE } from "@app/lib/onboarding-sample";
import { useAppStore } from "@app/store/app-store";
import { Onboarding } from "./Onboarding";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const state = () => useAppStore.getState();

let container: HTMLDivElement;
let root: Root;
let behind: HTMLButtonElement;

const freshLearner = () =>
  useAppStore.setState({
    onboarded: false,
    onboardingReplay: false,
    progress: {},
    snapshots: [],
    moduleStates: {},
    mode: "plain",
    fontStep: 1,
    highContrast: false,
    reducedMotion: false,
    verboseNarration: false,
    colorVision: "default",
    view: "home",
    settingsTab: "appearance",
    searchOpen: false,
    accessibilityOpen: false,
  });

/** The tour inside a stand-in for the app: a title bar, the app shell, and the tour as the last child. */
const mount = () =>
  act(() => {
    root.render(
      createElement(
        "div",
        { className: "app" },
        createElement("div", { className: "titlebar" }, createElement("button", { type: "button", id: "titlebar-button" }, "Title bar")),
        createElement(
          "div",
          { className: "app-shell" },
          createElement("button", { type: "button", id: "behind" }, "Behind the tour"),
        ),
        createElement(Onboarding),
      ),
    );
  });

const dialog = () => container.querySelector<HTMLElement>('[role="dialog"]');
const heading = () => dialog()?.querySelector<HTMLElement>("h2") ?? null;
const buttons = () => Array.from(dialog()?.querySelectorAll<HTMLButtonElement>("button") ?? []);
const buttonByText = (text: string) => buttons().find((button) => button.textContent?.trim().includes(text));
const click = (element: Element | undefined | null) =>
  act(() => {
    element!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
const press = (text: string) => {
  const button = buttonByText(text);
  expect(button, `no button "${text}"`).toBeDefined();
  click(button);
};
const key = (target: Element, init: KeyboardEventInit) =>
  act(() => {
    target.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init }));
  });
const toStep = (count: number) => {
  for (let index = 0; index < count; index += 1) press("Next");
};

beforeEach(() => {
  freshLearner();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

describe("when it shows", () => {
  it("opens for a fresh learner as a labelled, modal dialog on step 1 of 3", () => {
    mount();
    const element = dialog();
    expect(element).not.toBeNull();
    expect(element!.getAttribute("aria-modal")).toBe("true");
    const labelId = element!.getAttribute("aria-labelledby")!;
    const label = document.getElementById(labelId);
    expect(label?.tagName).toBe("H2");
    expect(label?.textContent).toContain("Step 1 of 3");
    expect(label?.textContent).toContain("Choose how explanations read");
  });

  it("stays away from a learner who already has progress, snapshots, or saved lab state", () => {
    useAppStore.setState({ progress: { a: { visited: true, completed: false, checkpointResults: [], currentStep: 0, updatedAt: "x" } } });
    mount();
    expect(dialog()).toBeNull();

    act(() => useAppStore.setState({ progress: {}, snapshots: [{ id: "s", moduleId: "a", name: "n", state: {}, createdAt: "x" }] }));
    expect(dialog()).toBeNull();

    act(() => useAppStore.setState({ snapshots: [], moduleStates: { a: { x: 1 } } }));
    expect(dialog()).toBeNull();
  });

  it("stays away once it has been finished", () => {
    useAppStore.setState({ onboarded: true });
    mount();
    expect(dialog()).toBeNull();
  });

  it("appears when the learner who has finished it asks to replay it", () => {
    useAppStore.setState({ onboarded: true, progress: { a: { visited: true, completed: false, checkpointResults: [], currentStep: 0, updatedAt: "x" } } });
    mount();
    expect(dialog()).toBeNull();
    act(() => state().replayOnboarding());
    expect(dialog()).not.toBeNull();
    expect(heading()?.textContent).toContain("Step 1 of 3");
  });
});

describe("moving through the three steps", () => {
  it("steps forward and back, saying where you are and what comes last", () => {
    mount();
    expect(buttonByText("Back")).toBeUndefined();
    expect(buttonByText("Skip tour")).toBeDefined();

    press("Next");
    expect(heading()?.textContent).toContain("Step 2 of 3");
    expect(heading()?.textContent).toContain("optional guide");
    expect(buttonByText("Back")).toBeDefined();

    press("Next");
    expect(heading()?.textContent).toContain("Step 3 of 3");
    expect(buttonByText("Next")).toBeUndefined();
    expect(buttonByText("Start learning")).toBeDefined();
    expect(buttonByText("Skip tour")).toBeUndefined();

    press("Back");
    expect(heading()?.textContent).toContain("Step 2 of 3");
    press("Back");
    expect(heading()?.textContent).toContain("Step 1 of 3");
  });

  it("marks reached steps in the progress list and says which one is current", () => {
    mount();
    const items = () => Array.from(container.querySelectorAll(".onboarding__progress li"));
    expect(items()).toHaveLength(3);
    expect(items().map((item) => item.classList.contains("is-reached"))).toEqual([true, false, false]);
    press("Next");
    expect(items().map((item) => item.classList.contains("is-reached"))).toEqual([true, true, false]);
    expect(items()[1].getAttribute("aria-current")).toBe("step");
  });
});

describe("step 1: the two explanation modes", () => {
  it("shows the real Plain and Standard paragraphs from the lab", () => {
    mount();
    const text = dialog()!.textContent ?? "";
    // The sample paragraph is plain text up to its first mark, so its opening sentence renders verbatim.
    expect(text).toContain(ONBOARDING_SAMPLE.plain.split(". ")[0]);
    expect(text).toContain("That loop is what autoregressive means.");
    expect(text).toContain(ONBOARDING_SAMPLE.moduleTitle);
    // The lesson's own bold and code marks are rendered, not shown as raw punctuation.
    expect(text).not.toContain("**");
    expect(dialog()!.querySelector(".onboarding-mode__sample code")?.textContent).toContain("exp(z_i / T)");
  });

  it("lets the learner choose, and the choice is the store's mode", () => {
    mount();
    expect(buttonByText("Use Plain")!.getAttribute("aria-pressed")).toBe("true");
    expect(buttonByText("Use Standard")!.getAttribute("aria-pressed")).toBe("false");
    press("Use Standard");
    expect(state().mode).toBe("standard");
    expect(buttonByText("Use Standard")!.getAttribute("aria-pressed")).toBe("true");
    expect(buttonByText("Use Plain")!.getAttribute("aria-pressed")).toBe("false");
    press("Use Plain");
    expect(state().mode).toBe("plain");
  });
});

describe("step 2: the optional guide", () => {
  it("explains that it is optional, private or cloud, and can wait", () => {
    mount();
    toStep(1);
    const text = dialog()!.textContent ?? "";
    expect(text).toContain("It is optional");
    expect(text).toContain("Private, on this computer");
    expect(text).toContain("A cloud provider");
    expect(text).toContain("Settings, under Local model or Cloud providers");
    expect(buttonByText("Skip, set up later")).toBeDefined();
  });

  it("'Skip, set up later' moves on with nothing to open afterwards", () => {
    mount();
    toStep(1);
    press("Take me to Local model settings");
    press("Skip, set up later");
    expect(heading()?.textContent).toContain("Step 3 of 3");
    expect(buttonByText("settings")).toBeUndefined();
    expect(buttonByText("Start learning")).toBeDefined();
  });

  it("offers to open Local model settings after finishing, when asked", () => {
    mount();
    toStep(1);
    press("Take me to Local model settings");
    expect(buttonByText("Take me to Local model settings")!.getAttribute("aria-pressed")).toBe("true");
    toStep(1);
    expect(state().onboarded).toBe(false);
    press("Open Local model settings");
    expect(state().onboarded).toBe(true);
    expect(state().view).toBe("settings");
    expect(state().settingsTab).toBe("local");
    expect(dialog()).toBeNull();
  });

  it("offers Cloud providers the same way, and the choice can be undone", () => {
    mount();
    toStep(1);
    press("Take me to Cloud provider settings");
    press("Take me to Cloud provider settings");
    toStep(1);
    expect(buttonByText("Open Cloud providers settings")).toBeUndefined();
    press("Back");
    press("Take me to Cloud provider settings");
    toStep(1);
    press("Open Cloud providers settings");
    expect(state().view).toBe("settings");
    expect(state().settingsTab).toBe("cloud");
  });
});

describe("step 3: accessibility, live", () => {
  const switchByLabel = (label: string) =>
    Array.from(dialog()!.querySelectorAll<HTMLButtonElement>('[role="switch"]')).find((item) => item.textContent?.includes(label))!;

  it("drives the same store settings as the accessibility menu", () => {
    mount();
    toStep(2);
    click(switchByLabel("Higher contrast"));
    expect(state().highContrast).toBe(true);
    click(switchByLabel("Reduce motion"));
    expect(state().reducedMotion).toBe(true);
    click(switchByLabel("Verbose narration"));
    expect(state().verboseNarration).toBe(true);
    expect(switchByLabel("Higher contrast").getAttribute("aria-checked")).toBe("true");
    click(switchByLabel("Higher contrast"));
    expect(state().highContrast).toBe(false);

    press("Deuteranopia");
    expect(state().colorVision).toBe("deuteranopia");
    expect(buttonByText("Deuteranopia")!.getAttribute("aria-pressed")).toBe("true");
    press("Default");
    expect(state().colorVision).toBe("default");
  });

  it("changes text size with the buttons and the slider, stopping at the ends", () => {
    mount();
    toStep(2);
    click(dialog()!.querySelector('[aria-label="Increase text size"]'));
    expect(state().fontStep).toBe(2);
    click(dialog()!.querySelector('[aria-label="Decrease text size"]'));
    click(dialog()!.querySelector('[aria-label="Decrease text size"]'));
    expect(state().fontStep).toBe(0);
    expect((dialog()!.querySelector('[aria-label="Decrease text size"]') as HTMLButtonElement).disabled).toBe(true);

    const slider = dialog()!.querySelector<HTMLInputElement>('input[type="range"]')!;
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(slider, "6");
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(state().fontStep).toBe(6);
    expect(dialog()!.textContent).toContain("200%");
    expect(slider.getAttribute("aria-labelledby")).toBe("onboarding-text-size");
  });

  it("'Start learning' finishes the tour and lands on Home", () => {
    useAppStore.setState({ view: "settings" });
    mount();
    toStep(2);
    press("Start learning");
    expect(state().onboarded).toBe(true);
    expect(state().view).toBe("home");
    expect(dialog()).toBeNull();
  });
});

describe("skipping", () => {
  it("'Skip tour' marks it done without moving the learner", () => {
    useAppStore.setState({ view: "settings", settingsTab: "data" });
    mount();
    press("Skip tour");
    expect(state().onboarded).toBe(true);
    expect(state().view).toBe("settings");
    expect(state().settingsTab).toBe("data");
    expect(dialog()).toBeNull();
  });

  it("Escape skips from any step, keeping the choices already made", () => {
    mount();
    press("Use Standard");
    toStep(1);
    key(heading()!, { key: "Escape" });
    expect(state().onboarded).toBe(true);
    expect(state().mode).toBe("standard");
    expect(dialog()).toBeNull();
  });

  it("closing a replay leaves the learner where they were and still onboarded", () => {
    useAppStore.setState({ onboarded: true, view: "settings", settingsTab: "about", progress: { a: { visited: true, completed: false, checkpointResults: [], currentStep: 0, updatedAt: "x" } } });
    mount();
    act(() => state().replayOnboarding());
    key(heading()!, { key: "Escape" });
    expect(dialog()).toBeNull();
    expect(state().onboarded).toBe(true);
    expect(state().onboardingReplay).toBe(false);
    expect(state().settingsTab).toBe("about");
  });
});

describe("keyboard focus", () => {
  it("moves focus to the heading on open and to each new step's heading", () => {
    mount();
    expect(document.activeElement).toBe(heading());
    press("Next");
    expect(document.activeElement).toBe(heading());
    expect(heading()?.textContent).toContain("Step 2 of 3");
  });

  it("keeps Tab and Shift+Tab inside the dialog", () => {
    mount();
    const focusable = buttons();
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    expect(first.textContent).toContain("Use Plain");

    last.focus();
    expect(document.activeElement).toBe(last);
    const tab = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    act(() => {
      last.dispatchEvent(tab);
    });
    expect(tab.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(first);

    const back = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true });
    act(() => {
      first.dispatchEvent(back);
    });
    expect(back.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(last);
  });

  it("does not interfere with Tab between the first and last control", () => {
    mount();
    const focusable = buttons();
    focusable[1].focus();
    const tab = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    act(() => {
      focusable[1].dispatchEvent(tab);
    });
    expect(tab.defaultPrevented).toBe(false);
  });

  it("pulls focus back if something outside the dialog takes it", () => {
    mount();
    const outside = container.querySelector<HTMLButtonElement>("#behind")!;
    act(() => outside.focus());
    expect(dialog()!.contains(document.activeElement)).toBe(true);
  });

  it("hides the app behind it while open, but leaves the title bar usable", () => {
    mount();
    expect(container.querySelector(".app-shell")!.hasAttribute("inert")).toBe(true);
    expect(container.querySelector(".titlebar")!.hasAttribute("inert")).toBe(false);
    press("Skip tour");
    expect(container.querySelector(".app-shell")!.hasAttribute("inert")).toBe(false);
  });

  it("returns focus to where it was when the tour closes", () => {
    useAppStore.setState({ onboarded: true });
    mount();
    const trigger = container.querySelector<HTMLButtonElement>("#behind")!;
    trigger.focus();
    act(() => state().replayOnboarding());
    expect(dialog()!.contains(document.activeElement)).toBe(true);
    press("Skip tour");
    expect(document.activeElement).toBe(trigger);
  });

  it("swallows the search shortcut instead of opening a palette behind the dialog", () => {
    mount();
    const seen = vi.fn();
    window.addEventListener("keydown", seen);
    key(heading()!, { key: "k", ctrlKey: true });
    window.removeEventListener("keydown", seen);
    expect(seen).not.toHaveBeenCalled();
  });

  it("closes a search palette or accessibility menu that was open underneath", () => {
    mount();
    act(() => useAppStore.setState({ searchOpen: true, accessibilityOpen: true }));
    expect(state().searchOpen).toBe(false);
    expect(state().accessibilityOpen).toBe(false);
  });
});

describe("motion and layout", () => {
  const css = readFileSync(path.resolve(__dirname, "..", "styles.css"), "utf8");

  it("has no entrance animation when motion is reduced, by setting or by system preference", () => {
    expect(css).toMatch(/:root\[data-motion="reduced"\] \.onboarding \{\s*animation: none;/);
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\) \{\s*\.onboarding \{\s*animation: none;/);
  });

  it("scrolls inside the dialog and sits below the title bar, so a small window or large text still reaches every control", () => {
    expect(css).toMatch(/\.onboarding-overlay \{[^}]*inset: var\(--titlebar-height\) 0 0 0;/);
    expect(css).toMatch(/\.onboarding__body \{[^}]*overflow-y: auto;/);
    expect(css).toMatch(/\.onboarding \{[^}]*max-height: 100%;/);
  });
});
