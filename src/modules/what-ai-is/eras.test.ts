import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import { DEFAULT_ERA_YEAR, ERAS, ERA_YEARS, OLD_ERA_YEARS, eraYearFromState, nearestEraIndex } from "./eras";

const render = (patch: Record<string, string | number | string[]>, currentStep = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

describe("the timeline", () => {
  it("has eleven stops in strictly increasing year order, early AI first", () => {
    expect(ERAS).toHaveLength(11);
    expect(ERA_YEARS).toEqual([1956, 1958, 1969, 1974, 1980, 1986, 1987, 1997, 2012, 2017, 2022]);
    for (let index = 1; index < ERA_YEARS.length; index += 1) expect(ERA_YEARS[index]).toBeGreaterThan(ERA_YEARS[index - 1]);
    expect(DEFAULT_ERA_YEAR).toBe(1956);
  });

  it("adds the Dartmouth workshop, the two AI winters, and expert systems, each with an idea and a stall", () => {
    const byYear = (year: number) => ERAS.find((era) => era.year === year)!;
    expect(byYear(1956).title).toBe("Dartmouth workshop");
    expect(byYear(1974).title).toBe("First AI winter");
    expect(byYear(1980).title).toBe("Expert systems");
    expect(byYear(1987).title).toBe("Second AI winter");
    for (const era of ERAS) {
      expect(era.idea.length, era.title).toBeGreaterThan(20);
      expect(era.stalled.length, era.title).toBeGreaterThan(20);
    }
    // The figures the early stops quote.
    expect(byYear(1956).stalled).toContain("seven open problems");
    expect(byYear(1980).idea).toContain("written in 1978");
    expect(byYear(1980).idea).toContain("first went into use in 1980");
    expect(byYear(1980).idea).toContain("about 2,500 rules");
    expect(byYear(1974).idea).toContain("1973");
    expect(byYear(1974).idea).toContain("roughly 1974 to 1980");
    expect(byYear(1987).idea).toContain("about 1987");
  });

  it("keeps the seven original stops, so a version 3 position maps to the same stop", () => {
    expect(OLD_ERA_YEARS).toEqual([1958, 1969, 1986, 1997, 2012, 2017, 2022]);
    for (const year of OLD_ERA_YEARS) expect(ERA_YEARS).toContain(year);
    const titles = OLD_ERA_YEARS.map((year) => ERAS.find((era) => era.year === year)!.title);
    expect(titles).toEqual(["Perceptron", "Perceptrons critique", "Backpropagation", "LSTM memory", "AlexNet", "Transformer", "Assistants"]);
  });
});

describe("era state, version 4", () => {
  const { hydrateState, initialState, serializeState } = definition;

  it("round-trips the defaults and starts at the Dartmouth workshop", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(initialState.eraYear).toBe(1956);
    expect(initialState).not.toHaveProperty("era");
    expect(definition.stateVersion).toBe(4);
  });

  it("translates a version 3 era index into that stop's year", () => {
    for (let index = 0; index < OLD_ERA_YEARS.length; index += 1) {
      expect(hydrateState(JSON.stringify({ era: index })).eraYear).toBe(OLD_ERA_YEARS[index]);
    }
    expect(hydrateState(JSON.stringify({ era: 99 })).eraYear).toBe(2022);
    expect(hydrateState(JSON.stringify({ era: -4 })).eraYear).toBe(1958);
    expect(hydrateState(JSON.stringify({ era: 2, ring: "dl" }))).toMatchObject({ eraYear: 1986, ring: "dl" });
    expect(hydrateState(JSON.stringify({ era: 2 }))).not.toHaveProperty("era");
  });

  it("prefers eraYear and snaps any number to a real stop", () => {
    expect(hydrateState(JSON.stringify({ eraYear: 1980, era: 0 })).eraYear).toBe(1980);
    expect(hydrateState(JSON.stringify({ eraYear: 1983 })).eraYear).toBe(1980);
    expect(hydrateState(JSON.stringify({ eraYear: 1985 })).eraYear).toBe(1986);
    expect(hydrateState(JSON.stringify({ eraYear: 1e9 })).eraYear).toBe(2022);
    expect(hydrateState(JSON.stringify({ eraYear: -1e9 })).eraYear).toBe(1956);
    expect(hydrateState(JSON.stringify({ eraYear: "1980" })).eraYear).toBe(1956);
    expect(eraYearFromState(Number.NaN, undefined)).toBe(1956);
    expect(eraYearFromState(undefined, Number.POSITIVE_INFINITY)).toBe(1956);
    expect(nearestEraIndex(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe("the lab shows the honest rings and the timeline", () => {
  it("prints the timeline at every stop without a broken value", () => {
    for (const year of ERA_YEARS) {
      const html = render({ eraYear: year }, 3);
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
      const text = visible(html);
      const era = ERAS.find((item) => item.year === year)!;
      expect(text).toContain(era.title);
      expect(text).toContain(`${year}`);
    }
    const early = visible(render({ eraYear: 1956 }, 3));
    expect(early).toContain("Dartmouth workshop");
    expect(early).toContain("seven open problems");
  });

  it("marks the n-gram inside ML and outside DL, and says what the drawing cannot show", () => {
    const html = render({});
    expect(html).toContain("ai-ring-marker");
    expect(html).toContain("n-gram");
    expect(html).toContain("A marker for an n-gram sits inside the ML circle and outside the deep-learning circle");
    const text = visible(html);
    expect(text).toContain("What the drawing cannot show");
    expect(text).toContain("The Gen ring holds the neural generative models only");
    expect(text).toContain("A common stack: AI ⊃ ML ⊃ DL ⊃ neural generative models ⊃ LLMs");
  });

  it("defines AGI in the scope cut and the glossary, and claims no system reaches it", () => {
    const text = visible(render({ distinction: "scope" }));
    expect(text).toContain("AGI, artificial general intelligence");
    expect(text).toContain("no agreed test");
    const glossary = definition.glossary.find((item) => item.term === "Artificial intelligence")!;
    expect(glossary.definition).toContain("AGI, artificial general intelligence");
    expect(glossary.definition).toContain("no agreed test");
  });

  it("covers every objective with a question, including the timeline and AGI questions", () => {
    const questions = checkpointQuestions(definition);
    expect(questions).toHaveLength(8);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3]));
    expect(questions.some((question) => question.prompt.includes("expert systems of 1980"))).toBe(true);
    expect(questions.some((question) => question.prompt.includes("AGI"))).toBe(true);
  });
});
