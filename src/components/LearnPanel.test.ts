import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { ModuleWorkspace } from "./ModuleWorkspace";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const attention = modulesBySlug.get("attention")!;
const state = () => useAppStore.getState();

let container: HTMLDivElement;
let root: Root;
let scrolled: string[];

const mount = () => {
  act(() => {
    root.render(createElement(ModuleWorkspace, { module: attention }));
  });
};
// Waits on animation frames rather than a timeout: jsdom stops its frame timer whenever no callback
// is queued, so a frame requested from inside the last queued one can otherwise stall.
const frames = (count = 6) =>
  act(async () => {
    for (let index = 0; index < count; index += 1) {
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }
  });
const click = (element: Element | null | undefined) =>
  act(() => {
    element!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
const contents = () => container.querySelector<HTMLElement>(".learn-contents")!;
const heading = (text: string) => container.querySelector<HTMLElement>(`.learn-copy [data-learn-section="${text}"]`);

beforeEach(() => {
  scrolled = [];
  Element.prototype.scrollIntoView = function scrollIntoView(this: Element) {
    scrolled.push((this as HTMLElement).dataset.learnSection ?? "");
  };
  useAppStore.setState({
    onboarded: true,
    view: "module",
    activeModuleId: attention.id,
    mode: "plain",
    learnOpen: true,
    learnExpanded: false,
    searchFocus: null,
    bookmarks: [],
    readingPositions: {},
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

describe("Learn panel contents", () => {
  it("lists the seven lesson sections in order, then the checkpoint", () => {
    mount();
    const sections = [...contents().querySelectorAll(".learn-contents__section")].map((button) => button.textContent);
    expect(sections.slice(0, 8)).toEqual([
      "01The big picture",
      "02What it is",
      "03Why it’s here",
      "04How to play with it",
      "05What to notice",
      "06Where it breaks",
      "07Going deeper",
      "Checkpoint",
    ]);
  });

  it("jumps to a section and closes the drawer", () => {
    mount();
    click(container.querySelector(".learn-sectionbar__toggle"));
    expect(contents().dataset.open).toBe("true");
    const target = [...contents().querySelectorAll("button")].find((button) => button.textContent === "06Where it breaks");
    click(target);
    expect(scrolled).toEqual(["Where it breaks"]);
    expect(contents().dataset.open).toBe("false");
  });
});

describe("Bookmarks", () => {
  it("bookmarks a subsection from its heading, under its section", () => {
    mount();
    const toggle = heading("Why divide by 8?")!.parentElement!.querySelector<HTMLButtonElement>(".learn-heading__bookmark");
    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    click(toggle);
    expect(state().bookmarks).toMatchObject([
      { moduleId: attention.id, mode: "plain", section: "Why divide by 8?", parent: "What it is" },
    ]);
    expect(toggle?.getAttribute("aria-pressed")).toBe("true");
    // The heading itself still reads as just its title.
    expect(heading("Why divide by 8?")!.textContent).toBe("Why divide by 8?");
  });

  it("opens the bookmarks panel from the dock and lists this lab's bookmarks", () => {
    state().toggleBookmark({ moduleId: attention.id, mode: "plain", section: "Why divide by 8?", parent: "What it is" });
    mount();
    click(container.querySelector('[aria-label="Bookmarks"]'));
    const panel = container.querySelector('[role="dialog"][aria-label="Bookmarks"]');
    expect(panel).not.toBeNull();
    expect(panel!.textContent).toContain("This lab");
    expect(panel!.textContent).toContain("Why divide by 8?");
  });

  it("switches to the bookmark's reading mode only when the current one lacks the heading", async () => {
    // "Words as lists of numbers" is a Plain subsection with no Standard counterpart.
    state().toggleBookmark({ moduleId: attention.id, mode: "plain", section: "Words as lists of numbers", parent: "What it is" });
    state().toggleBookmark({ moduleId: attention.id, mode: "plain", section: "Where it breaks", parent: "Where it breaks" });
    useAppStore.setState({ mode: "standard" });
    mount();

    click(container.querySelector('[aria-label="Bookmarks"]'));
    click([...container.querySelectorAll(".bookmark-list button")].find((button) => button.textContent?.startsWith("Where it breaks")));
    await frames();
    expect(state().mode).toBe("standard");
    expect(scrolled.at(-1)).toBe("Where it breaks");

    click(container.querySelector('[aria-label="Bookmarks"]'));
    click([...container.querySelectorAll(".bookmark-list button")].find((button) => button.textContent?.startsWith("Words as lists")));
    await frames();
    expect(state().mode).toBe("plain");
    expect(scrolled.at(-1)).toBe("Words as lists of numbers");
    expect(state().searchFocus).toBeNull();
  });
});

describe("Picking up where you left off", () => {
  it("offers the saved section and goes there", () => {
    state().setReadingPosition(attention.id, { mode: "plain", section: "The one-way rule", parent: "What it is" });
    mount();
    const prompt = container.querySelector(".learn-resume");
    expect(prompt?.textContent).toContain("What it is · The one-way rule");
    click(prompt!.querySelector(".learn-resume__go"));
    expect(scrolled).toEqual(["The one-way rule"]);
    expect(container.querySelector(".learn-resume")).toBeNull();
  });

  it("lands on the section when the subsection is not in the current mode", () => {
    state().setReadingPosition(attention.id, { mode: "plain", section: "Words as lists of numbers", parent: "What it is" });
    useAppStore.setState({ mode: "standard" });
    mount();
    expect(container.querySelector(".learn-resume")?.textContent).toContain("What it is");
  });

  it("stays quiet when the learner was still at the start", () => {
    state().setReadingPosition(attention.id, { mode: "plain", section: "The big picture", parent: "The big picture" });
    mount();
    expect(container.querySelector(".learn-resume")).toBeNull();
  });
});
