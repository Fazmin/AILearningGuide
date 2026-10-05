import { describe, expect, it } from "vitest";
import { nextInSequence, pathComplete, pathState, prerequisitesOf } from "./learning-path";

const a = { id: "a", order: 1, prerequisites: [] as string[] };
const b = { id: "b", order: 2, prerequisites: ["a"] };
const c = { id: "c", order: 3, prerequisites: ["a", "b"] };
const all = [c, a, b];
const byId = new Map(all.map((module) => [module.id, module]));

describe("pathState", () => {
  it("separates complete, started, and untouched", () => {
    expect(pathState({ completed: true, visited: true })).toBe("done");
    expect(pathState({ visited: true })).toBe("started");
    expect(pathState({})).toBe("new");
    expect(pathState(undefined)).toBe("new");
  });
});

describe("prerequisitesOf", () => {
  it("lists what a lab builds on with the learner's progress in each", () => {
    const result = prerequisitesOf(c, byId, { a: { completed: true }, b: { visited: true } });
    expect(result.map((item) => [item.module.id, item.state])).toEqual([
      ["a", "done"],
      ["b", "started"],
    ]);
  });

  it("returns nothing for a lab with no prerequisites and skips unknown ids", () => {
    expect(prerequisitesOf(a, byId, {})).toEqual([]);
    expect(prerequisitesOf({ id: "x", order: 9, prerequisites: ["gone"] }, byId, {})).toEqual([]);
  });
});

describe("nextInSequence", () => {
  it("starts at the first lab and moves past completed ones in map order", () => {
    expect(nextInSequence(all, {})?.id).toBe("a");
    expect(nextInSequence(all, { a: { completed: true } })?.id).toBe("b");
    expect(nextInSequence(all, { a: { completed: true }, b: { completed: true } })?.id).toBe("c");
  });

  it("fills a gap rather than following the most recent lab", () => {
    expect(nextInSequence(all, { b: { completed: true }, c: { completed: true } })?.id).toBe("a");
  });

  it("is undefined once everything is complete", () => {
    const done = { a: { completed: true }, b: { completed: true }, c: { completed: true } };
    expect(nextInSequence(all, done)).toBeUndefined();
  });
});

describe("pathComplete", () => {
  it("is true only when every lab is complete", () => {
    expect(pathComplete(all, { a: { completed: true }, b: { completed: true } })).toBe(false);
    expect(pathComplete(all, { a: { completed: true }, b: { completed: true }, c: { completed: true } })).toBe(true);
  });

  it("is false for an empty path", () => {
    expect(pathComplete([], {})).toBe(false);
  });
});
