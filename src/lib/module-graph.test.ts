import { describe, expect, it } from "vitest";
import {
  buildGraph,
  defaultLayoutOptions,
  focusOf,
  layoutGraph,
  type GraphModule,
} from "./module-graph";

const lab = (id: string, order: number, ...prerequisites: string[]): GraphModule => ({
  id,
  order,
  prerequisites,
});

// a ─ b ─ d ─ e, with a ─ c ─ d, and a long link a → e that skips every row between.
const diamond = [
  lab("e", 5, "d", "a"),
  lab("a", 1),
  lab("c", 3, "a"),
  lab("b", 2, "a"),
  lab("d", 4, "b", "c"),
];

describe("buildGraph", () => {
  it("orders labs by learning-map order and lists direct links both ways", () => {
    const graph = buildGraph(diamond);
    expect(graph.ids).toEqual(["a", "b", "c", "d", "e"]);
    expect(graph.prerequisites.get("e")).toEqual(["a", "d"]);
    expect(graph.dependents.get("a")).toEqual(["b", "c", "e"]);
    expect(graph.dependents.get("e")).toEqual([]);
  });

  it("drops unknown ids, self links, and repeats", () => {
    const graph = buildGraph([lab("a", 1), lab("b", 2, "a", "a", "b", "gone")]);
    expect(graph.prerequisites.get("b")).toEqual(["a"]);
    expect(graph.dependents.get("a")).toEqual(["b"]);
  });
});

describe("focusOf", () => {
  const graph = buildGraph(diamond);

  it("separates what a lab builds on from what builds on it, through any number of steps", () => {
    const focus = focusOf(graph, "d");
    expect(focus.before).toEqual(["a", "b", "c"]);
    expect(focus.after).toEqual(["e"]);
    expect(focus.directBefore).toEqual(["b", "c"]);
    expect(focus.directAfter).toEqual(["e"]);
    expect(focus.beforeSet.has("a")).toBe(true);
    expect(focus.afterSet.has("a")).toBe(false);
  });

  it("never lists the lab itself, even when the data loops back", () => {
    const loop = buildGraph([lab("a", 1, "b"), lab("b", 2, "a")]);
    expect(focusOf(loop, "a").before).toEqual(["b"]);
    expect(focusOf(loop, "a").after).toEqual(["b"]);
  });

  it("finds the roots and leaves", () => {
    expect(focusOf(graph, "a").before).toEqual([]);
    expect(focusOf(graph, "e").after).toEqual([]);
    expect(focusOf(graph, "e").before).toEqual(["a", "b", "c", "d"]);
  });
});

describe("layoutGraph", () => {
  const { nodeWidth, nodeHeight } = defaultLayoutOptions;

  it("puts every lab on a row below everything it builds on", () => {
    const layout = layoutGraph(buildGraph(diamond));
    const row = new Map(layout.nodes.map((node) => [node.id, node.row]));
    expect([...row.values()]).toEqual([0, 1, 1, 2, 3]);
    for (const { from, to } of layout.links) expect(row.get(from)!).toBeLessThan(row.get(to)!);
  });

  it("runs a link that skips rows through a lane instead of behind another lab", () => {
    const layout = layoutGraph(buildGraph(diamond));
    const long = layout.links.find((link) => link.from === "a" && link.to === "e")!;
    // Leaves a, then enters and leaves a lane on each of the two rows it crosses, then enters e.
    expect(long.points).toHaveLength(2 + 2 * 2);
    for (const point of long.points.slice(1, -1)) {
      for (const node of layout.nodes) {
        if (node.id === "a" || node.id === "e") continue;
        const inside =
          point.x > node.x && point.x < node.x + nodeWidth && point.y > node.y && point.y < node.y + nodeHeight;
        expect(inside, `lane point ${point.x},${point.y} is inside ${node.id}`).toBe(false);
      }
    }
  });

  it("starts links at the bottom of the prerequisite and ends them at the top of the lab", () => {
    const layout = layoutGraph(buildGraph(diamond));
    const byId = new Map(layout.nodes.map((node) => [node.id, node]));
    for (const link of layout.links) {
      const from = byId.get(link.from)!;
      const to = byId.get(link.to)!;
      expect(link.points[0]).toEqual({ x: from.x + nodeWidth / 2, y: from.y + nodeHeight });
      expect(link.points.at(-1)).toEqual({ x: to.x + nodeWidth / 2, y: to.y });
      link.points.slice(1).forEach((point, i) => expect(point.y).toBeGreaterThanOrEqual(link.points[i].y));
    }
  });

  it("keeps every lab inside the canvas and never overlaps neighbours on a row", () => {
    const layout = layoutGraph(buildGraph(diamond));
    for (const node of layout.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.x + nodeWidth).toBeLessThanOrEqual(layout.width);
      expect(node.y + nodeHeight).toBeLessThanOrEqual(layout.height);
    }
    const sorted = [...layout.nodes].sort((a, b) => a.row - b.row || a.x - b.x);
    sorted.slice(1).forEach((node, i) => {
      if (node.row === sorted[i].row) expect(node.x).toBeGreaterThanOrEqual(sorted[i].x + nodeWidth);
    });
  });

  it("gives the same layout every time", () => {
    expect(layoutGraph(buildGraph(diamond))).toEqual(layoutGraph(buildGraph([...diamond].reverse())));
  });

  it("lays out a wide, tangled graph without overlaps or labs inside a lane", () => {
    // A deterministic pseudo-random DAG: each lab builds on up to three earlier ones.
    let seed = 7;
    const next = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const labs = Array.from({ length: 60 }, (_, i) => {
      const picks = new Set<string>();
      for (let k = 0; k < Math.floor(next() * 4) && i > 0; k += 1) picks.add(`n${Math.floor(next() * i)}`);
      return lab(`n${i}`, i + 1, ...picks);
    });
    const layout = layoutGraph(buildGraph(labs));
    const byRow = new Map<number, number[]>();
    for (const node of layout.nodes) byRow.set(node.row, [...(byRow.get(node.row) ?? []), node.x]);
    for (const xs of byRow.values()) {
      xs.sort((a, b) => a - b);
      xs.slice(1).forEach((x, i) => expect(x - xs[i]).toBeGreaterThanOrEqual(nodeWidth));
    }
    for (const link of layout.links) {
      for (const point of link.points.slice(1, -1)) {
        for (const node of layout.nodes) {
          if (node.id === link.from || node.id === link.to) continue;
          const inside =
            point.x > node.x && point.x < node.x + nodeWidth && point.y > node.y && point.y < node.y + nodeHeight;
          expect(inside).toBe(false);
        }
      }
    }
    expect(layout.links).toHaveLength(labs.reduce((sum, module) => sum + module.prerequisites.length, 0));
  });

  it("survives a prerequisite loop and an empty guide", () => {
    const loop = layoutGraph(buildGraph([lab("a", 1, "b"), lab("b", 2, "a")]));
    expect(loop.nodes).toHaveLength(2);
    expect(loop.links.length).toBeLessThanOrEqual(1);
    const empty = layoutGraph(buildGraph([]));
    expect(empty).toMatchObject({ nodes: [], links: [], rows: 0 });
    expect(Number.isFinite(empty.width) && Number.isFinite(empty.height)).toBe(true);
  });
});
