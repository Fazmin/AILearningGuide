import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildGraph, defaultLayoutOptions, focusOf, layoutGraph } from "@app/lib/module-graph";
import { modules, modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { ConnectionsScreen } from "./ConnectionsScreen";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  // jsdom has no layout engine and does not implement element scrolling.
  HTMLElement.prototype.scrollTo = vi.fn() as unknown as typeof HTMLElement.prototype.scrollTo;
  useAppStore.getState().resetLearningData();
  useAppStore.getState().setConnectionsFocus(null);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(createElement(ConnectionsScreen)));
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const nodes = () => [...container.querySelectorAll<HTMLButtonElement>(".conn-node")];
const nodeFor = (slug: string) => nodes().find((node) => node.textContent?.includes(modulesBySlug.get(slug)!.title))!;
const click = (element: Element) => act(() => element.dispatchEvent(new MouseEvent("click", { bubbles: true })));
const panel = () => container.querySelector(".conn-panel")!;

describe("the real guide's connections layout", () => {
  const graph = buildGraph(modules);
  const layout = layoutGraph(graph);
  const { nodeWidth, nodeHeight } = defaultLayoutOptions;

  it("places every lab and every prerequisite link, with nothing overlapping", () => {
    expect(layout.nodes).toHaveLength(modules.length);
    expect(layout.links).toHaveLength(modules.reduce((sum, module) => sum + (graph.prerequisites.get(module.id)?.length ?? 0), 0));

    const sorted = [...layout.nodes].sort((a, b) => a.row - b.row || a.x - b.x);
    sorted.slice(1).forEach((node, i) => {
      if (node.row === sorted[i].row) expect(node.x, `${node.id} overlaps ${sorted[i].id}`).toBeGreaterThanOrEqual(sorted[i].x + nodeWidth);
    });
  });

  it("never routes a link through a lab it does not connect", () => {
    for (const link of layout.links) {
      for (const point of link.points.slice(1, -1)) {
        for (const node of layout.nodes) {
          if (node.id === link.from || node.id === link.to) continue;
          const inside = point.x > node.x && point.x < node.x + nodeWidth && point.y > node.y && point.y < node.y + nodeHeight;
          expect(inside, `${link.from} → ${link.to} runs through ${node.id}`).toBe(false);
        }
      }
    }
  });
});

describe("ConnectionsScreen", () => {
  it("shows one button per lab and invites a choice before anything is selected", () => {
    expect(nodes()).toHaveLength(modules.length);
    expect(nodes().every((node) => node.getAttribute("aria-pressed") === "false")).toBe(true);
    expect(panel().textContent).toContain("Pick a lab");
  });

  it("marks what a lab builds on, what builds on it, and what is unrelated", () => {
    const attention = modulesBySlug.get("attention")!;
    click(nodeFor("attention"));
    const focus = focusOf(buildGraph(modules), attention.id);

    expect(nodeFor("attention").getAttribute("aria-pressed")).toBe("true");
    for (const module of modules) {
      if (module.id === attention.id) continue;
      const expected = focus.beforeSet.has(module.id) ? "is-before" : focus.afterSet.has(module.id) ? "is-after" : "is-apart";
      const node = nodeFor(module.slug);
      expect(node.classList.contains(expected), `${module.slug} should be ${expected}`).toBe(true);
    }
    expect(container.querySelectorAll(".conn-link.is-before").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".conn-link.is-after").length).toBeGreaterThan(0);
  });

  it("describes the selection in the panel, with counts that match the graph", () => {
    const module = modulesBySlug.get("transformer-block")!;
    click(nodeFor("transformer-block"));
    const focus = focusOf(buildGraph(modules), module.id);

    expect(panel().querySelector("h2")?.textContent).toBe(module.title);
    const summary = panel().querySelector(".conn-panel__summary")!.textContent!;
    expect(summary).toContain(`Builds on ${focus.before.length} lab`);
    expect(summary).toContain(`Opens up ${focus.after.length} lab`);
    expect(panel().textContent).toContain("Builds directly on");
    expect(panel().textContent).toContain("Leads directly to");
  });

  it("follows a connection from the panel to the lab it names", () => {
    click(nodeFor("attention"));
    const next = [...panel().querySelectorAll<HTMLButtonElement>(".conn-lab")].find((button) =>
      button.textContent?.includes(modulesBySlug.get("next-token-prediction")!.title),
    )!;
    click(next);
    expect(useAppStore.getState().connectionsFocusId).toBe(modulesBySlug.get("next-token-prediction")!.id);
    expect(nodeFor("next-token-prediction").getAttribute("aria-pressed")).toBe("true");
    expect(nodeFor("attention").getAttribute("aria-pressed")).toBe("false");
  });

  it("clears on a second click, on Clear, and on Escape", () => {
    const select = () => click(nodeFor("attention"));
    select();
    click(nodeFor("attention"));
    expect(useAppStore.getState().connectionsFocusId).toBeNull();

    select();
    click([...panel().querySelectorAll("button")].find((button) => button.textContent?.includes("Clear"))!);
    expect(useAppStore.getState().connectionsFocusId).toBeNull();

    select();
    act(() => {
      nodeFor("attention").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(useAppStore.getState().connectionsFocusId).toBeNull();
    expect(panel().textContent).toContain("Pick a lab");
  });

  it("says plainly when a lab is the start or the end of the guide", () => {
    click(nodeFor("what-ai-is"));
    expect(panel().textContent).toContain("This is where the guide starts");
    expect(panel().textContent).toContain(`Opens up ${modules.length - 1} labs`);

    const leaf = modules.find((module) => buildGraph(modules).dependents.get(module.id)!.length === 0)!;
    click(nodeFor(leaf.slug));
    expect(panel().textContent).toContain("Nothing else builds on this lab");
  });

  it("reflects the learner's progress in the panel and on the map", () => {
    const attention = modulesBySlug.get("attention")!;
    const focus = focusOf(buildGraph(modules), attention.id);
    act(() => {
      for (const id of focus.before.slice(0, 3)) useAppStore.getState().toggleModuleComplete(id);
    });
    click(nodeFor("attention"));
    expect(panel().textContent).toContain(`You have finished 3 of the ${focus.before.length} labs`);
    expect(container.querySelectorAll(".conn-node.is-done")).toHaveLength(3);
  });

  it("opens the selected lab", () => {
    click(nodeFor("attention"));
    click([...panel().querySelectorAll("button")].find((button) => button.textContent?.includes("Open lab"))!);
    expect(useAppStore.getState().view).toBe("module");
    expect(useAppStore.getState().activeModuleId).toBe(modulesBySlug.get("attention")!.id);
  });

  it("opens with a lab already selected when another screen sends one", () => {
    act(() => root.unmount());
    root = createRoot(container);
    useAppStore.getState().openConnections(modulesBySlug.get("quantization")!.id);
    act(() => root.render(createElement(ConnectionsScreen)));
    expect(useAppStore.getState().view).toBe("connections");
    expect(nodeFor("quantization").getAttribute("aria-pressed")).toBe("true");
  });
});
