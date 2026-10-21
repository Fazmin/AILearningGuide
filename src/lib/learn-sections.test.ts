import { createElement } from "react";
import { describe, expect, it } from "vitest";
import {
  activeLearnIndex,
  collectLearnEntries,
  findLearnElement,
  findLearnEntry,
  groupLearnEntries,
  headingText,
  sameSection,
} from "./learn-sections";

const lesson = () => {
  const root = document.createElement("div");
  root.innerHTML = `
    <article>
      <h2 data-learn-section="The big picture" data-learn-level="2"></h2>
      <h3 data-learn-section="Where you have met it" data-learn-level="3"></h3>
      <h2 data-learn-section="What it is" data-learn-level="2"></h2>
      <h3 data-learn-section="Why divide by 8?" data-learn-level="3"></h3>
      <h3 data-learn-section="The one-way rule" data-learn-level="3"></h3>
      <h2 data-learn-section="Why it’s here" data-learn-level="2"></h2>
    </article>
    <div data-learn-section="Checkpoint" data-learn-level="2" data-learn-extra=""></div>
    <h2 data-learn-section="References" data-learn-level="2" data-learn-extra=""></h2>`;
  return root;
};

describe("collectLearnEntries", () => {
  it("reads headings in order, numbering only the lesson's own sections", () => {
    expect(collectLearnEntries(lesson())).toEqual([
      { text: "The big picture", level: 2, parent: "The big picture", number: 1 },
      { text: "Where you have met it", level: 3, parent: "The big picture", number: null },
      { text: "What it is", level: 2, parent: "What it is", number: 2 },
      { text: "Why divide by 8?", level: 3, parent: "What it is", number: null },
      { text: "The one-way rule", level: 3, parent: "What it is", number: null },
      { text: "Why it’s here", level: 2, parent: "Why it’s here", number: 3 },
      { text: "Checkpoint", level: 2, parent: "Checkpoint", number: null },
      { text: "References", level: 2, parent: "References", number: null },
    ]);
  });

  it("groups subsections under their section", () => {
    const groups = groupLearnEntries(collectLearnEntries(lesson()));
    expect(groups.map((group) => [group.entry.text, group.index, group.children.map((child) => child.index)])).toEqual([
      ["The big picture", 0, [1]],
      ["What it is", 2, [3, 4]],
      ["Why it’s here", 5, []],
      ["Checkpoint", 6, []],
      ["References", 7, []],
    ]);
  });
});

describe("finding a section", () => {
  const entries = collectLearnEntries(lesson());

  it("matches headings whatever the quote style", () => {
    expect(sameSection("Why it's here", "Why it’s here")).toBe(true);
    expect(findLearnEntry(entries, "why it's here")).toBe(5);
    expect(findLearnElement(lesson(), "Why it's here")?.dataset.learnSection).toBe("Why it’s here");
  });

  it("falls back to the top-level section when a subsection is missing", () => {
    expect(findLearnEntry(entries, "Words as lists of numbers", "What it is")).toBe(2);
    expect(findLearnEntry(entries, "Words as lists of numbers")).toBe(-1);
  });
});

describe("activeLearnIndex", () => {
  it("is the last heading to pass the threshold", () => {
    expect(activeLearnIndex([0, 400, 900], 600, 96, false)).toBe(0);
    expect(activeLearnIndex([-500, 80, 600], 600, 96, false)).toBe(1);
  });

  it("is the last heading on screen once the panel cannot scroll further", () => {
    expect(activeLearnIndex([-900, -200, 300, 550], 600, 96, true)).toBe(3);
  });
});

describe("headingText", () => {
  it("reads the text a heading will render, including inline code", () => {
    expect(headingText(["Why ", createElement("code", null, "q8_0"), " is not\n exactly half"])).toBe(
      "Why q8_0 is not exactly half",
    );
  });
});
