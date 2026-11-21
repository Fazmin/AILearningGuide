import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider } from "@app/module-sdk";
import cardInfo from "./card-info";
import definition from "./module";
import scenes from "./scenes";
import { SYSTEMS, CLAIMS, encodePlacement, encodeTags } from "./model";
import { hydrateState, initialState, readState, GUESS_OPTIONS } from "./state";
import { pct, span, times, workSpan } from "./kit";

const LABELS = [
  "Narrow, general, and how good",
  "One profile, three verdicts",
  "Faster, more, or better",
  "A loop that feeds itself",
  "Reading forecasts",
  "Challenge: sort the claims",
];

const html = (patch: Record<string, unknown>, currentStep: number) =>
  renderToStaticMarkup(
    createElement(
      CardInfoProvider,
      { value: cardInfo },
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...patch } as typeof definition.initialState,
        setState: () => undefined,
        currentStep,
        mode: "plain" as const,
        narrate: () => undefined,
      }),
    ),
  ).replace(/&#x27;/g, "'");
const text = (patch: Record<string, unknown>, step: number) => html(patch, step).replace(/<[^>]*>/g, "\u0001");
/** The value printed in the metric tile with this label. */
const tile = (patch: Record<string, unknown>, step: number, label: string) => {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = html(patch, step).match(new RegExp(`<div class="metric[^"]*"><span>${escaped}</span><strong>([^<]*)</strong>`));
  expect(match, `no tile "${label}"`).not.toBeNull();
  return match![1];
};
const expectClean = (markup: string, where: string) => {
  expect(markup.match(/\bNaN\b|\bundefined\b|\bInfinity\b/)?.[0], where).toBeUndefined();
  expect(markup.match(/&amp;(?:gt|lt|amp|quot|#\d+);/)?.[0], where).toBeUndefined();
  expect(markup.replace(/<[^>]*>/g, "\u0001").match(/\d\.\d{6,}/)?.[0], where).toBeUndefined();
};

describe("the lab renders", () => {
  it("has one step per card, in the cards' order", () => {
    expect(definition.steps).toHaveLength(LABELS.length);
    expect(Object.keys(cardInfo)).toEqual(LABELS);
  });

  it.each(LABELS.map((label, step) => [step, label] as const))("at step %i it shows the card %s and no other", (step, label) => {
    const markup = html({}, step);
    expectClean(markup, `step ${step}`);
    expect([...markup.matchAll(/<section class="lab-surface[^"]*" aria-label="([^"]*)"/g)].map((card) => card[1])).toEqual([label]);
    expect(markup).toContain(`Step ${step + 1} of ${LABELS.length}`);
  });

  it("gives every picture a text alternative that states its reading", () => {
    const labels = LABELS.flatMap((_, step) => [...html({}, step).matchAll(/<svg class="asi-plot[^"]*"[^>]*role="img" aria-label="([^"]*)"/g)].map((match) => match[1]));
    expect(labels.length).toBeGreaterThanOrEqual(5);
    for (const label of labels) {
      expect(label.length).toBeGreaterThan(80);
      expect(label).toMatch(/\d/);
    }
    expect(html({}, 0)).toContain('aria-label="Level 5, Superhuman, general: artificial superintelligence. Nothing placed."');
  });

  it("asks five predict-first questions and reveals nothing until they are answered", () => {
    const all = LABELS.map((_, step) => text({}, step)).join("");
    expect((all.match(/Predict first/g) ?? []).length).toBe(5);
    expect(all).not.toContain("Not this time.");
    const right = { levelGuess: 1, profileGuess: 2, speedGuess: 2, loopGuess: 1, surveyGuess: 2 };
    expect((LABELS.map((_, step) => text(right, step)).join("").match(/Right\./g) ?? []).length).toBe(5);
  });

  it("marks placements only once all eight systems are placed", () => {
    const partial = { place: encodePlacement(SYSTEMS.map((system, index) => (index < 7 ? system.cell : -1))) };
    expect(text(partial, 0)).toContain("7 of 8 placed");
    expect(html(partial, 0)).not.toContain("is-right");
    const right = { place: encodePlacement(SYSTEMS.map((system) => system.cell)) };
    expect(text(right, 0)).toContain("every system where the table puts it");
    const wrong = { place: encodePlacement(SYSTEMS.map(() => 11)) };
    expect(text(wrong, 0)).toContain("0 of 8");
    expect((html(wrong, 0).match(/class="is-wrong"/g) ?? []).length).toBe(SYSTEMS.length);
  });

  it("prints the readings the lessons quote", () => {
    expect(tile({}, 1, "GPT-5's AGI score")).toBe("57%");
    expect(tile({}, 1, "Areas at adult level")).toBe("2 of 10");
    expect(tile({ profile: "gpt4" }, 1, "GPT-4's AGI score")).toBe("27%");
    expect(text({ rule: "weakest" }, 1)).toContain("long-term memory storage, 0 of 10");
    expect(tile({}, 2, "One expert alone")).toBe("100 years");
    expect(tile({ speed: 1000 }, 2, "With these settings")).toBe("37 days");
    expect(tile({ speed: 1000, copies: 1000000 }, 2, "With these settings")).toBe("3.2 seconds");
    expect(tile({ speed: 1000, copies: 1000000, world: 0.1 }, 2, "With these settings")).toBe("10 years");
    expect(tile({ speed: 1000, copies: 1000000, world: 0.1 }, 2, "Fastest possible here")).toBe("10 years");
    expect(tile({}, 3, "Matches people")).toBe("year 6.0");
    expect(tile({}, 3, "The climb from 1× to 100×")).toBe("12.0 years");
    expect(tile({ bottleneck: 0.2 }, 3, "Reaches 100×")).toBe("not by year 40");
    expect(tile({ bottleneck: 0.01 }, 3, "Reaches 100×")).toBe("year 19.8");
    expect(tile({ bottleneck: 0.01 }, 3, "Leaves the chart")).toBe("no");
    expect(tile({ returns: 1 }, 3, "The climb from 1× to 100×")).toBe("30.7 years");
    expect(tile({ returns: 3 }, 3, "The climb from 1× to 100×")).toBe("3.3 years");
    expect(tile({}, 4, "Carried forward")).toBe("1.9 working months");
    expect(tile({ doubling: 12 }, 4, "Carried forward")).toBe("3.3 working days");
    expect(tile({}, 4, "Survey's 50% year, every task")).toBe("2047");
    expect(tile({ survey: "2022" }, 4, "Survey's 50% year, every task")).toBe("2060");
    expect(text({ surveyGuess: 0 }, 4)).toContain("2047 for every task, 2116 for every occupation: 69 years");
  });

  it("scores the claims and explains each answer", () => {
    const all = { tags: encodeTags(CLAIMS.map((claim) => claim.kind)) };
    expect(text(all, 5)).toContain("8 of 8 · every claim sorted");
    expect((text(all, 5).match(/Right\./g) ?? []).length).toBe(8);
    const wrong = { tags: encodeTags(CLAIMS.map(() => "m")) };
    expect(text(wrong, 5)).toContain("2 of 8 right · 8 sorted");
  });

  const states: [string, Record<string, unknown>][] = [
    ["every slider at its top", { work: 1000, speed: 1000000, copies: 1000000, inOrder: 0.5, world: 0.5, returns: 3, bottleneck: 0.2, loopYear: 40, ahead: 10, doubling: 4 }],
    ["every slider at its bottom", { work: 1, speed: 1, copies: 1, inOrder: 0, world: 0, returns: 0.5, bottleneck: 0, loopYear: 0, ahead: 0, doubling: 24 }],
    ["a ghost run and the cursor past the runaway", { returns: 3, prevReturns: 0.5, prevBottleneck: 0.2, loopYear: 39 }],
    ["GPT-4 with the strict rule", { profile: "gpt4", rule: "count" }],
    ["everything placed in the ASI corner", { place: "llllllll", cell: 11, chip: 7 }],
  ];
  it.each(states)("with %s", (name, patch) => {
    for (let step = 0; step < LABELS.length; step += 1) expectClean(html(patch, step), name);
  });

  it("opens every scene on the card it names", () => {
    for (const scene of scenes) {
      const markup = html(scene.state ?? {}, scene.step ?? 0);
      expect(markup, scene.id).toContain(`data-lab-surface="${scene.focus}"`);
    }
  });
});

describe("lab state", () => {
  it("round-trips and fills in defaults", () => {
    expect(hydrateState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateState("{}")).toEqual(initialState);
    expect(hydrateState("nope")).toEqual(initialState);
    expect(hydrateState("[]")).toEqual(initialState);
  });

  it("snaps every control to a stop it offers and forgets impossible guesses", () => {
    const state = readState({ work: 50, speed: 3000, copies: 0, inOrder: 0.3, world: 9, returns: 1.1, bottleneck: 0.03, doubling: 9, ahead: 99, loopYear: -4 });
    expect(state).toMatchObject({ work: 100, speed: 1000, copies: 1, inOrder: 0.25, world: 0.5, returns: 1, bottleneck: 0.01, doubling: 7, ahead: 10, loopYear: 0 });
    for (const key of Object.keys(GUESS_OPTIONS)) expect(readState({ [key]: 7 })[key as keyof typeof GUESS_OPTIONS]).toBe(-1);
    expect(readState({ prevReturns: -1 }).prevReturns).toBe(0);
    expect(readState({ profile: "gpt9", rule: "vibes", survey: "1999" })).toMatchObject({ profile: "gpt5", rule: "sum", survey: "2023" });
  });
});

describe("formatting", () => {
  it("says spans, multipliers, shares and work in plain units", () => {
    expect(span(100)).toBe("100 years");
    expect(span(0)).toBe("none");
    expect(span(1e-7)).toBe("3.2 seconds");
    expect(span(1e-12)).toBe("under a millisecond");
    expect(times(1000000)).toBe("1 million×");
    expect(times(1000)).toBe("1,000×");
    expect(pct(0.001)).toBe("0.1%");
    expect(pct(0.57)).toBe("57%");
    expect(workSpan(50)).toBe("50 minutes");
    expect(workSpan(60 * 26.7)).toBe("3.3 working days");
  });
});

describe("the lessons", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const lessons = ["simple", "plain", "standard"].map((mode) => [mode, readFileSync(path.join(here, "content", `${mode}.mdx`), "utf8").replace(/\s+/g, " ")] as const);
  it.each(["57%", "0%", "2047", "2116", "2060", "2,778", "1.9 working months", "19.8", "10 years"])("quote %s in every mode", (figure) => {
    for (const [mode, source] of lessons) expect(source.includes(figure), `${mode} should say ${figure}`).toBe(true);
  });
});
