import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { ModuleWorkspace } from "./ModuleWorkspace";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// A lab that keeps the seven-section lesson shape (MODULE_AUTHORING.md).
const lab = modulesBySlug.get("next-token-prediction")!;
const state = () => useAppStore.getState();

let container: HTMLDivElement;
let root: Root;
let scrolled: string[];

const mount = () => {
  act(() => {
    root.render(createElement(ModuleWorkspace, { module: lab }));
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
    activeModuleId: lab.id,
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
    const toggle = heading("Where generation stops")!.parentElement!.querySelector<HTMLButtonElement>(".learn-heading__bookmark");
    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    click(toggle);
    expect(state().bookmarks).toMatchObject([
      { moduleId: lab.id, mode: "plain", section: "Where generation stops", parent: "What it is" },
    ]);
    expect(toggle?.getAttribute("aria-pressed")).toBe("true");
    // The heading itself still reads as just its title.
    expect(heading("Where generation stops")!.textContent).toBe("Where generation stops");
  });

  it("opens the bookmarks panel from the dock and lists this lab's bookmarks", () => {
    state().toggleBookmark({ moduleId: lab.id, mode: "plain", section: "Where generation stops", parent: "What it is" });
    mount();
    click(container.querySelector('[aria-label="Bookmarks"]'));
    const panel = container.querySelector('[role="dialog"][aria-label="Bookmarks"]');
    expect(panel).not.toBeNull();
    expect(panel!.textContent).toContain("This lab");
    expect(panel!.textContent).toContain("Where generation stops");
  });

  it("switches to the bookmark's reading mode only when the current one lacks the heading", async () => {
    // "One step, repeated" is a Plain subsection with no Standard counterpart.
    state().toggleBookmark({ moduleId: lab.id, mode: "plain", section: "One step, repeated", parent: "What it is" });
    state().toggleBookmark({ moduleId: lab.id, mode: "plain", section: "Where it breaks", parent: "Where it breaks" });
    useAppStore.setState({ mode: "standard" });
    mount();

    click(container.querySelector('[aria-label="Bookmarks"]'));
    click([...container.querySelectorAll(".bookmark-list button")].find((button) => button.textContent?.startsWith("Where it breaks")));
    await frames();
    expect(state().mode).toBe("standard");
    expect(scrolled.at(-1)).toBe("Where it breaks");

    click(container.querySelector('[aria-label="Bookmarks"]'));
    click([...container.querySelectorAll(".bookmark-list button")].find((button) => button.textContent?.startsWith("One step, repeated")));
    await frames();
    expect(state().mode).toBe("plain");
    expect(scrolled.at(-1)).toBe("One step, repeated");
    expect(state().searchFocus).toBeNull();
  });
});

describe("Picking up where you left off", () => {
  it("offers the saved section and goes there", () => {
    state().setReadingPosition(lab.id, { mode: "plain", section: "From scores to odds", parent: "What it is" });
    mount();
    const prompt = container.querySelector(".learn-resume");
    expect(prompt?.textContent).toContain("What it is · From scores to odds");
    click(prompt!.querySelector(".learn-resume__go"));
    expect(scrolled).toEqual(["From scores to odds"]);
    expect(container.querySelector(".learn-resume")).toBeNull();
  });

  it("lands on the section when the subsection is not in the current mode", () => {
    state().setReadingPosition(lab.id, { mode: "plain", section: "One step, repeated", parent: "What it is" });
    useAppStore.setState({ mode: "standard" });
    mount();
    expect(container.querySelector(".learn-resume")?.textContent).toContain("What it is");
  });

  it("stays quiet when the learner was still at the start", () => {
    state().setReadingPosition(lab.id, { mode: "plain", section: "The big picture", parent: "The big picture" });
    mount();
    expect(container.querySelector(".learn-resume")).toBeNull();
  });
});
