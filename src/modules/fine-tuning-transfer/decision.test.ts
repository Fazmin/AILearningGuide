import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  allCombinations,
  DECISION_KEYS,
  DECISION_LABELS,
  DECISION_QUESTIONS,
  DECISION_RULES,
  recommend,
  ROUTES,
  type DecisionAnswers,
  type RouteId,
} from "./decision";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";

/**
 * The "Should you fine-tune?" card is a hand-written rule of thumb, so there is nothing to
 * measure; these tests pin the table itself: every combination of answers reaches exactly one
 * route, the table matches an independently written expectation, and no printed reason claims
 * something the answers contradict. The counts quoted in the lesson and the card text (four
 * questions, 16 combinations, five rules, five routes) are asserted here.
 */

const answers = (changes: boolean, cites: boolean, format: boolean, examples: boolean): DecisionAnswers => ({
  knowledgeChanges: changes,
  needsCitations: cites,
  needsConsistency: format,
  hasExamples: examples,
});

/**
 * The expected route for all 16 combinations, written out by hand rather than derived from the
 * rule code. Columns: changing facts, citations, consistent format, examples + evaluation set.
 */
const EXPECTED: ReadonlyArray<readonly [boolean, boolean, boolean, boolean, RouteId]> = [
  [false, false, false, false, "prompt"],
  [false, false, false, true, "prompt"],
  [false, false, true, false, "examples"],
  [false, false, true, true, "fine-tune"],
  [true, false, false, false, "retrieval"],
  [true, false, false, true, "retrieval"],
  [true, false, true, false, "retrieval"],
  [true, false, true, true, "retrieval-and-fine-tune"],
  [false, true, false, false, "retrieval"],
  [false, true, false, true, "retrieval"],
  [false, true, true, false, "retrieval"],
  [false, true, true, true, "retrieval-and-fine-tune"],
  [true, true, false, false, "retrieval"],
  [true, true, false, true, "retrieval"],
  [true, true, true, false, "retrieval"],
  [true, true, true, true, "retrieval-and-fine-tune"],
];

describe("the decision table", () => {
  it("has the sizes the lesson and card text quote", () => {
    expect(DECISION_KEYS).toHaveLength(4);
    expect(allCombinations()).toHaveLength(16);
    expect(DECISION_RULES).toHaveLength(5);
    expect(Object.keys(ROUTES)).toHaveLength(5);
    // The default is four "no" answers, which lands on the cheapest route.
    expect(recommend(answers(false, false, false, false)).route.id).toBe("prompt");
  });

  it("enumerates every combination exactly once", () => {
    const keys = allCombinations().map((combination) => DECISION_KEYS.map((key) => (combination[key] ? 1 : 0)).join(""));
    expect(new Set(keys).size).toBe(16);
  });

  it("gives every combination a defined route, matching the hand-written table", () => {
    expect(EXPECTED).toHaveLength(16);
    for (const [changes, cites, format, examples, route] of EXPECTED) {
      const label = `changes=${changes} cites=${cites} format=${format} examples=${examples}`;
      const result = recommend(answers(changes, cites, format, examples));
      expect(result.route.id, label).toBe(route);
      expect(ROUTES[result.route.id], label).toBe(result.route);
      expect(result.rule, label).toBeGreaterThanOrEqual(0);
      expect(result.rule, label).toBeLessThan(DECISION_RULES.length);
      expect(DECISION_RULES[result.rule].route, label).toBe(route);
      expect(result.because.length, label).toBeGreaterThan(40);
      expect(result.route.title.length, label).toBeGreaterThan(0);
      expect(result.route.next.length, label).toBeGreaterThan(40);
    }
  });

  it("reaches every route, and every rule fires for at least one combination", () => {
    const routes = new Set(allCombinations().map((combination) => recommend(combination).route.id));
    expect([...routes].sort()).toEqual(Object.keys(ROUTES).sort());
    const rules = new Set(allCombinations().map((combination) => recommend(combination).rule));
    expect([...rules].sort()).toEqual([0, 1, 2, 3, 4]);
  });

  it("is a first-match-wins table: the rule that fired is the earliest that matches", () => {
    for (const combination of allCombinations()) {
      const result = recommend(combination);
      for (let earlier = 0; earlier < result.rule; earlier += 1) {
        expect(DECISION_RULES[earlier].matches(combination)).toBe(false);
      }
      expect(DECISION_RULES[result.rule].matches(combination)).toBe(true);
    }
  });

  it("recommends fine-tuning only when behaviour is wanted and could be trained and checked", () => {
    for (const combination of allCombinations()) {
      const id = recommend(combination).route.id;
      const fineTunes = id === "fine-tune" || id === "retrieval-and-fine-tune";
      expect(fineTunes).toBe(combination.needsConsistency && combination.hasExamples);
      const retrieves = id === "retrieval" || id === "retrieval-and-fine-tune";
      expect(retrieves).toBe(combination.knowledgeChanges || combination.needsCitations);
    }
  });

  it("never treats having examples as a reason to fine-tune on its own", () => {
    for (const changes of [false, true]) {
      for (const cites of [false, true]) {
        const without = recommend(answers(changes, cites, false, false)).route.id;
        const withData = recommend(answers(changes, cites, false, true)).route.id;
        expect(withData).toBe(without);
      }
    }
  });
});

describe("the printed reasons", () => {
  it("prints one line per question, in control order, each matching the answer given", () => {
    for (const combination of allCombinations()) {
      const result = recommend(combination);
      expect(result.lines.map((line) => line.key)).toEqual([...DECISION_KEYS]);
      for (const line of result.lines) {
        expect(line.value, `${line.key} line must describe the answer given`).toBe(combination[line.key]);
        expect(line.text.length).toBeGreaterThan(30);
      }
    }
  });

  it("never prints a line for the opposite answer, and never repeats a line", () => {
    const seenByAnswer = new Map<string, string>();
    for (const combination of allCombinations()) {
      const result = recommend(combination);
      const texts = result.lines.map((line) => line.text);
      expect(new Set(texts).size, "no two lines of one recommendation repeat").toBe(texts.length);
      for (const line of result.lines) {
        const slot = `${line.key}:${line.value}`;
        const known = seenByAnswer.get(slot);
        if (known !== undefined) expect(line.text).toBe(known);
        seenByAnswer.set(slot, line.text);
      }
    }
    // Yes and No for one question must read differently, and no line is shared between questions.
    expect(seenByAnswer.size).toBe(DECISION_KEYS.length * 2);
    expect(new Set(seenByAnswer.values()).size).toBe(seenByAnswer.size);
  });

  it("never states a reason that the answers contradict", () => {
    // Each phrase is only true under a condition; if a recommendation prints it, the condition must hold.
    const claims: ReadonlyArray<readonly [string, (a: DecisionAnswers) => boolean]> = [
      ["Facts that change or need a source belong in retrieval", (a) => a.knowledgeChanges || a.needsCitations],
      ["Your facts are stable and uncited", (a) => !a.knowledgeChanges && !a.needsCitations],
      ["you have the data and the evaluation set", (a) => a.hasExamples && a.needsConsistency],
      ["cannot yet train a fine-tune or check one", (a) => a.needsConsistency && !a.hasExamples],
      ["you need consistent behaviour", (a) => a.needsConsistency],
      ["The format is behaviour", (a) => a.needsConsistency],
      ["Nothing here needs more than a good prompt", (a) => !a.knowledgeChanges && !a.needsCitations && !a.needsConsistency],
    ];
    let checked = 0;
    for (const combination of allCombinations()) {
      const { because } = recommend(combination);
      for (const [phrase, holds] of claims) {
        if (because.toLowerCase().includes(phrase.toLowerCase())) {
          checked += 1;
          expect(holds(combination), `"${phrase}" printed for ${JSON.stringify(combination)}`).toBe(true);
        }
      }
    }
    // The claims are not vacuous: they were exercised for several combinations.
    expect(checked).toBeGreaterThan(16);
    // The same rule never prints two different explanations.
    const byRule = new Map<number, string>();
    for (const combination of allCombinations()) {
      const { rule, because } = recommend(combination);
      expect(byRule.get(rule) ?? because).toBe(because);
      byRule.set(rule, because);
    }
  });

  it("gives each rule its own explanation and the table its own rows", () => {
    expect(new Set(DECISION_RULES.map((rule) => rule.because)).size).toBe(DECISION_RULES.length);
    expect(new Set(DECISION_RULES.map((rule) => rule.when)).size).toBe(DECISION_RULES.length);
  });

  it("labels every question and answer in words", () => {
    for (const key of DECISION_KEYS) {
      expect(DECISION_QUESTIONS[key].endsWith("?")).toBe(true);
      expect(DECISION_LABELS[key].length).toBeGreaterThan(3);
    }
  });
});

describe("the decision card as rendered", () => {
  const render = (state: Record<string, unknown>) =>
    renderToStaticMarkup(
      createElement(definition.Explore, {
        // Hostile values (null, wrong types) are deliberate, so the type is cast on purpose.
        state: { ...definition.initialState, ...state } as ModuleState,
        setState: () => undefined,
        currentStep: 4,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    );

  it("is labelled a rule of thumb, with keyboard-operable yes/no buttons for each question", () => {
    const html = render({});
    expect(html).toContain("A hand-written rule of thumb, not a measurement");
    expect(html).toContain("rule of thumb that someone wrote by hand");
    for (const key of DECISION_KEYS) expect(html).toContain(DECISION_QUESTIONS[key]);
    // A real <button> per answer, so Tab and Enter work; the pressed one is exposed to assistive tech.
    expect(html.match(/<button[^>]*aria-pressed="true"[^>]*>No<\/button>/g)).toHaveLength(4);
    expect(html.match(/<button[^>]*aria-pressed="false"[^>]*>Yes<\/button>/g)).toHaveLength(4);
  });

  it("states the route, the matched row and each answer in text, not by colour", () => {
    const html = render({ needsConsistency: true, hasExamples: true });
    expect(html).toContain("Fine-tune");
    expect(html).toContain("(matched)");
    expect(html).toContain('aria-current="true"');
    expect(html).toContain("Format or behaviour: Yes.");
    expect(html).toContain("Changing facts: No.");
    expect(html).toContain("rule 3 of 5 matched");
  });

  it("replaces non-boolean stored answers with No", () => {
    const html = render({ knowledgeChanges: "yes", needsCitations: 1, needsConsistency: null, hasExamples: [] });
    expect(html).toContain("Prompt first");
    expect(html).toContain("rule 5 of 5 matched");
  });
});
