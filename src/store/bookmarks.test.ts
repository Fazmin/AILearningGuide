import { describe, expect, it } from "vitest";
import { sanitizeBookmarks, sanitizeReadingPositions, toggleBookmarkIn, type LearnBookmark } from "./bookmarks";

let counter = 0;
const stamp = () => ({ id: `b${++counter}`, createdAt: "2026-10-20T00:00:00.000Z" });

const target = { moduleId: "attention", mode: "plain" as const, section: "Why divide by 8?", parent: "What it is" };

describe("toggleBookmarkIn", () => {
  it("adds a bookmark at the front, then removes it on the second toggle", () => {
    const older: LearnBookmark = { ...target, section: "The one-way rule", ...stamp() };
    const added = toggleBookmarkIn([older], target, stamp);
    expect(added.map((bookmark) => bookmark.section)).toEqual(["Why divide by 8?", "The one-way rule"]);
    expect(toggleBookmarkIn(added, target, stamp)).toEqual([older]);
  });

  it("treats the same heading in another mode, or with other quotes, as the same bookmark", () => {
    const added = toggleBookmarkIn([], { ...target, section: "Why it’s here", parent: "Why it’s here" }, stamp);
    expect(toggleBookmarkIn(added, { ...target, mode: "standard", section: "Why it's here", parent: "Why it's here" }, stamp)).toEqual([]);
  });

  it("keeps the same heading in different labs apart", () => {
    const added = toggleBookmarkIn([], target, stamp);
    expect(toggleBookmarkIn(added, { ...target, moduleId: "embeddings-search" }, stamp)).toHaveLength(2);
  });
});

describe("sanitizeBookmarks", () => {
  it("keeps well-formed bookmarks once each and fills in a missing parent", () => {
    const kept = sanitizeBookmarks([
      { id: "a", moduleId: "attention", mode: "plain", section: "What it is", createdAt: "x" },
      { id: "b", moduleId: "attention", mode: "standard", section: "What it is", parent: "What it is", createdAt: "y" },
      { id: "c", moduleId: "attention", mode: "loud", section: "What it is", createdAt: "x" },
      { id: "d", moduleId: "attention", mode: "plain", section: " ", createdAt: "x" },
      null,
      "nope",
    ]);
    expect(kept).toEqual([{ id: "a", moduleId: "attention", mode: "plain", section: "What it is", parent: "What it is", createdAt: "x" }]);
    expect(sanitizeBookmarks(undefined)).toEqual([]);
  });
});

describe("sanitizeReadingPositions", () => {
  it("keeps one well-formed position per lab", () => {
    expect(
      sanitizeReadingPositions({
        attention: { mode: "simple", section: "Why divide by 8?", parent: "What it is", updatedAt: "x" },
        broken: { mode: "simple" },
        other: "nope",
      }),
    ).toEqual({ attention: { mode: "simple", section: "Why divide by 8?", parent: "What it is", updatedAt: "x" } });
    expect(sanitizeReadingPositions([])).toEqual({});
  });
});
