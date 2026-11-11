import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import cardInfo from "./card-info";
import { STEPS } from "./steps";

const render = (patch: Record<string, string | number> = {}, currentStep = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep,
      mode: "plain" as const,
      narrate: () => undefined,
    }),
  );
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const cardsIn = (html: string) => [...html.matchAll(/data-lab-surface="([^"]*)"/g)].map((match) => match[1].replace(/&#x27;/g, "'"));

const CARDS_BY_STEP = [
  ["Bars that add to one"],
  ["Draw from the bars"],
  ["Averages pile up in a bell"],
  ["Context changes the bars", "Read the table backwards"],
  ["Surprise in bits", "Average surprise"],
  ["Your guess against what happens"],
  ["From scores to bars"],
  ["Challenge: learn it from samples"],
];

/** Settings at both ends of every control, to be rendered at every step. */
const EXTREMES: [string, Record<string, string | number>][] = [
  ["the defaults", {}],
  ["one bar at its ceiling, counts view", { bars: "96,1,1,1,1", view: "counts", word: 4, first: 4, second: 4 }],
  ["the last bar at its ceiling", { bars: "1,1,1,1,96", follow: 4, word: 0 }],
  ["one draw", { draws: 1 }],
  ["every draw the lab keeps", { draws: 2000, luck: 999, follow: 4 }],
  ["the smallest test", { testSize: 10, bars: "96,1,1,1,1" }],
  ["the largest test", { testSize: 400 }],
  ["the last prompt and the lowest prior", { prompt: 2, clue: 4, prior: 5 }],
  ["the highest prior", { prompt: 1, clue: 2, prior: 90 }],
  ["one sure thing", { shape: 0 }],
  ["all equal", { shape: 100 }],
  ["a confidently wrong model", { guess: "1,1,1,1,96" }],
  ["a model that matches the truth", { guess: "50,25,12,8,5" }],
  ["the coldest temperature and extreme scores", { temperature: 0.2, scores: "6,-6,6,-6,0" }],
  ["the hottest temperature", { temperature: 4, scores: "-6,-6,-6,-6,-6" }],
  ["a scored and revealed last round", { round: 2, answer: "8,10,35,5,42", scored: 1, tries: 2, revealed: 1 }],
  ["a scored miss", { round: 1, answer: "96,1,1,1,1", scored: 1, tries: 5 }],
  ["every prediction answered", { predictDraws: 1, predictPrior: 0, predictShape: 2, predictBill: 1 }],
];

describe("probability-surprise lab", () => {
  it("shows each step's own cards, in order", () => {
    CARDS_BY_STEP.forEach((cards, step) => {
      expect(cardsIn(render({}, step))).toEqual(cards);
    });
    expect(CARDS_BY_STEP).toHaveLength(STEPS.length);
  });

  it("has an explanation for every card it draws, and none for a card it does not", () => {
    expect(Object.keys(cardInfo).sort()).toEqual(CARDS_BY_STEP.flat().sort());
  });

  it("falls back to a real step when the step index is out of range", () => {
    expect(cardsIn(render({}, -3))).toEqual(CARDS_BY_STEP[0]);
    expect(cardsIn(render({}, 99))).toEqual(CARDS_BY_STEP[CARDS_BY_STEP.length - 1]);
    expect(cardsIn(render({}, Number.NaN))).toEqual(CARDS_BY_STEP[0]);
  });

  it.each(EXTREMES)("renders every step with %s", (_name, patch) => {
    for (let step = 0; step < STEPS.length; step += 1) {
      const html = render(patch, step);
      const text = visible(html);
      expect(cardsIn(html).length).toBeGreaterThan(0);
      expect(text, `step ${step}`).not.toMatch(/\bNaN\b|\bundefined\b|Infinity|∞/);
      expect(html, `step ${step}`).not.toMatch(/="[^"]*\bNaN\b[^"]*"/);
      // No readout runs to a long tail of decimals.
      expect(text, `step ${step}`).not.toMatch(/\d\.\d{6,}/);
      expect(text).toContain(`Step ${step + 1} of ${STEPS.length}`);
    }
  });

  it("gives every picture a text alternative that states its reading", () => {
    for (let step = 0; step < STEPS.length; step += 1) {
      const html = render({ draws: 10 }, step);
      for (const match of html.matchAll(/<svg[^>]*>/g)) {
        expect(match[0], `step ${step}`).toMatch(/role="img"/);
        expect(match[0], `step ${step}`).toMatch(/aria-label="[^"]{40,}"/);
      }
    }
  });

  it("gives every draggable bar a labelled slider role with its value", () => {
    const html = render({}, 0);
    const sliders = [...html.matchAll(/role="slider"[^>]*>/g)].map((match) => match[0]);
    expect(sliders).toHaveLength(5);
    for (const slider of sliders) {
      expect(slider).toMatch(/aria-label="Bar for (blue|grey|dark|clear|green)"/);
      expect(slider).toMatch(/aria-valuenow="\d+"/);
      expect(slider).toMatch(/tabindex="0"/);
    }
    expect(html.match(/type="range"/g)).toHaveLength(5);
  });

  it("hides the challenge's true bars until the reader asks for them", () => {
    const hidden = render({ round: 0, scored: 1, tries: 1 }, 7);
    expect(hidden).not.toContain("ps-col__ghost");
    expect(visible(hidden)).not.toContain("dashed outline: the true bars");
    const shown = render({ round: 0, scored: 1, tries: 1, revealed: 1 }, 7);
    expect(shown).toContain("ps-col__ghost");
  });

  it("says the win in real-model terms only once the goal is met", () => {
    expect(visible(render({ answer: "7,10,50,28,5", scored: 1, tries: 1 }, 7))).toContain("You just trained a model.");
    expect(visible(render({ answer: "7,10,50,28,5", scored: 0 }, 7))).not.toContain("You just trained a model.");
    expect(visible(render({ scored: 1, tries: 1 }, 7))).not.toContain("You just trained a model.");
  });
});
