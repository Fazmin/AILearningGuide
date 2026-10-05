import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  MAX_BUDGET,
  MAX_HOPS,
  MIN_BUDGET,
  MIN_HOPS,
  NODES,
  START_NODE,
  TABLE,
  correctAnswer,
  follow,
  nextPass,
  passesNeeded,
  promptLines,
  runScratchpad,
  sweepHops,
} from "./scratchpad";

describe("the lookup table", () => {
  it("is one cycle through all ten nodes, so no two hop counts up to the maximum collide", () => {
    expect(new Set(Object.values(TABLE)).size).toBe(NODES.length);
    const cycle = follow(START_NODE, NODES.length);
    expect(cycle.slice(0, NODES.length)).toEqual(["C", "H", "A", "J", "E", "B", "G", "D", "I", "F"]);
    expect(cycle[NODES.length]).toBe(START_NODE);
    const visited = follow(START_NODE, MAX_HOPS);
    expect(new Set(visited).size).toBe(MAX_HOPS + 1);
  });

  it("gives the answers the lesson quotes", () => {
    expect(Array.from({ length: MAX_HOPS }, (_, index) => correctAnswer(index + 1))).toEqual([
      "H",
      "A",
      "J",
      "E",
      "B",
      "G",
      "D",
      "I",
    ]);
  });

  it("prints the table and the task as the prompt", () => {
    const lines = promptLines(6);
    expect(lines[0]).toContain("C→H");
    expect(lines[1]).toBe("start C, follow the table 6 times");
    expect(promptLines(1)[1]).toBe("start C, follow the table 1 time");
  });
});

describe("one forward pass at a time", () => {
  it("answers with where it got to when it runs out of budget and has no scratchpad", () => {
    const run = runScratchpad(6, 3, false);
    expect(run.passes).toHaveLength(1);
    expect(run.passes[0]).toMatchObject({ from: "C", readFrom: "prompt", hopsDone: 3, writes: "J", kind: "answer", outOfBudget: true });
    expect(run.passes[0].lookups).toEqual(["C", "H", "A", "J"]);
    expect(run.answer).toBe("J");
    expect(run.target).toBe("G");
    expect(run.correct).toBe(false);
    expect(run.tokens).toBe(1);
  });

  it("continues from the token it wrote when the scratchpad is on", () => {
    const run = runScratchpad(6, 3, true);
    expect(run.written).toEqual(["J", "G"]);
    expect(run.passes.map((pass) => [pass.index, pass.from, pass.readFrom, pass.kind, pass.writes])).toEqual([
      [1, "C", "prompt", "scratch", "J"],
      [2, "J", "scratchpad", "answer", "G"],
    ]);
    expect(run.passes[1].lookups).toEqual(["J", "E", "B", "G"]);
    expect(run.answer).toBe("G");
    expect(run.correct).toBe(true);
    expect(run.tokens).toBe(2);
  });

  it("costs one pass and one token per lookup at a budget of 1", () => {
    const run = runScratchpad(8, 1, true);
    expect(run.tokens).toBe(8);
    expect(run.written).toEqual(["H", "A", "J", "E", "B", "G", "D", "I"]);
    expect(run.correct).toBe(true);
  });

  it("needs the scratchpad for nothing when one pass is enough", () => {
    for (const hops of [1, 2, 3]) {
      const on = runScratchpad(hops, 3, true);
      const off = runScratchpad(hops, 3, false);
      expect(on.written).toEqual(off.written);
      expect(on.passes).toHaveLength(1);
      expect(on.passes[0].outOfBudget).toBe(false);
      expect(on.correct).toBe(true);
    }
  });

  it("reads only the written context to resume", () => {
    const pass = nextPass(["J"], 6, 3, true);
    expect(pass).toMatchObject({ from: "J", readFrom: "scratchpad", hopsDone: 6, writes: "G", kind: "answer" });
    // A corrupted scratch token changes the final answer, so the trace is what carries the computation.
    const tampered = runScratchpad(6, 3, true, (pass) => (pass.kind === "scratch" ? "B" : pass.writes));
    expect(tampered.written[0]).toBe("B");
    expect(tampered.answer).toBe(follow("B", 3)[3]);
    expect(tampered.answer).not.toBe(tampered.target);
    expect(tampered.correct).toBe(false);
  });
});

describe("every setting", () => {
  it("is right without a scratchpad exactly when the chain fits in one pass", () => {
    for (let budget = MIN_BUDGET; budget <= MAX_BUDGET; budget += 1) {
      for (let hops = MIN_HOPS; hops <= MAX_HOPS; hops += 1) {
        const off = runScratchpad(hops, budget, false);
        expect(off.correct, `k ${hops}, b ${budget} without`).toBe(hops <= budget);
        expect(off.tokens).toBe(1);
      }
    }
  });

  it("is always right with a scratchpad, at ceil(k / b) passes", () => {
    for (let budget = MIN_BUDGET; budget <= MAX_BUDGET; budget += 1) {
      for (let hops = MIN_HOPS; hops <= MAX_HOPS; hops += 1) {
        const on = runScratchpad(hops, budget, true);
        expect(on.correct, `k ${hops}, b ${budget} with`).toBe(true);
        expect(on.tokens).toBe(passesNeeded(hops, budget));
        expect(on.passes.at(-1)?.kind).toBe("answer");
        expect(on.passes.slice(0, -1).every((pass) => pass.kind === "scratch")).toBe(true);
      }
    }
  });

  it("gives the budget-3 table the lesson and checkpoint quote", () => {
    const rows = sweepHops(3);
    expect(rows.map((row) => row.without.correct)).toEqual([true, true, true, false, false, false, false, false]);
    expect(rows.every((row) => row.with.correct)).toBe(true);
    expect(rows.map((row) => row.with.passes)).toEqual([1, 1, 1, 2, 2, 2, 3, 3]);
    expect(rows[3].without.answer).toBe("J");
    expect(rows[7].without.answer).toBe("J");
  });

  it("moves the first failure down as the budget rises", () => {
    const firstFailure = (budget: number) => sweepHops(budget).find((row) => !row.without.correct)?.hops;
    expect(firstFailure(1)).toBe(2);
    expect(firstFailure(3)).toBe(4);
    expect(firstFailure(5)).toBe(6);
    expect(firstFailure(8)).toBeUndefined();
  });
});

describe("the scratchpad card", () => {
  const render = (state: Record<string, string | number>) =>
    renderToStaticMarkup(
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...state },
        setState: () => undefined,
        currentStep: 3,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    ).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");

  it("shows the wrong one-pass answer with the scratchpad off", () => {
    const text = render({ hops: 6, budget: 3, scratchpad: "off" });
    expect(text).toContain("budget used up after 3");
    expect(text).toContain("answer wrong");
    expect(text).not.toContain("read back");
  });

  it("shows the second pass reading the first one's token back with the scratchpad on", () => {
    const text = render({ hops: 6, budget: 3, scratchpad: "on" });
    expect(text).toContain("J · read back");
    expect(text).toContain("answer right");
    expect(text).toContain("scratch J");
  });

  it("falls back to safe values for out-of-range state", () => {
    const text = render({ hops: 0, budget: 99, scratchpad: "" });
    expect(text).toContain("k = 1 dependent lookups");
    expect(text).toContain("b = 8 per pass");
  });
});
