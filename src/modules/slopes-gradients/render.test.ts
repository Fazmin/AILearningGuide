import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider } from "@app/module-sdk";
import cardInfo from "./card-info";
import definition from "./module";
import scenes from "./scenes";
import { fmt, fmtChange, fmtExact, signed, snap } from "./kit";

const LABELS = [
  "Nudge the dial",
  "A slope at every point",
  "Two dials, one slope each",
  "Dials that drive dials",
  "Walk downhill",
  "Challenge: reach the valley floor",
];

const renderStep = (patch: Record<string, unknown>, currentStep: number) =>
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
  );
/** One step's page, or (with no step) every page one after another: the lab shows one card per step. */
const render = (patch: Record<string, unknown> = {}, step?: number) =>
  step === undefined ? LABELS.map((_, index) => renderStep(patch, index)).join("") : renderStep(patch, step);
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const expectClean = (html: string, where: string) => {
  expect(html.match(/\bNaN\b|\bundefined\b|\bInfinity\b/)?.[0], where).toBeUndefined();
  expect(html.match(/&amp;(?:gt|lt|amp|quot|#\d+);/)?.[0], where).toBeUndefined();
  expect(visible(html).match(/\d\.\d{6,}/)?.[0], where).toBeUndefined();
};

describe("the lab renders", () => {
  it("has one step per card, in the cards' order", () => {
    expect(definition.steps).toHaveLength(LABELS.length);
    expect(definition.stepInstructions).toHaveLength(LABELS.length);
    expect(Object.keys(cardInfo)).toEqual(LABELS);
  });

  it.each(LABELS.map((label, step) => [step, label] as const))("at step %i it shows the card %s and no other", (step, label) => {
    const html = render({}, step);
    expectClean(html, `step ${step}`);
    const cards = [...html.matchAll(/<section class="lab-surface[^"]*" aria-label="([^"]*)"/g)];
    expect(cards.map((card) => card[1])).toEqual([label]);
    expect((html.match(/class="card-info-trigger"/g) ?? []).length).toBe(1);
    expect(html).toContain(`Step ${step + 1} of ${LABELS.length}`);
  });

  it("shows every card once across the steps, each drawn from the reader's settings", () => {
    const patch = { chainW: 5, walkSteps: 7, raceStep: 0.6, raceSteps: 12, surface: "saddle" };
    const cards = [...render(patch).matchAll(/<section class="lab-surface[^"]*" aria-label="([^"]*)"/g)];
    expect(cards.map((card) => card[1])).toEqual(LABELS);
    // Paging away and back draws the same card from the same state.
    expect(render(patch, 3)).toBe(render(patch, 3));
    expect(visible(render(patch, 3))).not.toBe(visible(render({ ...patch, chainW: 3 }, 3)));
  });

  it("gives every picture a text alternative that states its reading", () => {
    const html = render();
    const pictures = [...html.matchAll(/<svg class="sg-plot[^"]*"[^>]*role="img" aria-label="([^"]*)"/g)].map((match) => match[1]);
    expect(pictures).toHaveLength(6);
    for (const label of pictures) {
      expect(label.length).toBeGreaterThan(120);
      expect(label).toMatch(/\d/);
    }
    expect(html).toContain('role="slider"');
    expect(html).toContain('aria-label="Dial w, on the plot"');
    expect(html).toContain('aria-label="Start w, on the plot"');
  });

  it("asks four predict-first questions and reveals nothing until they are answered", () => {
    const text = visible(render());
    expect((text.match(/Predict first/g) ?? []).length).toBe(4);
    expect(text).not.toContain("Not this time.");
    expect(text).not.toContain("Right.");
    const right = visible(render({ nudgeGuess: 2, arrowGuess: 1, chainGuess: 3, walkGuess: 0 }));
    expect((right.match(/Right\./g) ?? []).length).toBe(4);
    const wrong = visible(render({ nudgeGuess: 0, arrowGuess: 0, chainGuess: 2, walkGuess: 1 }));
    expect((wrong.match(/Not this time\./g) ?? []).length).toBe(4);
  });

  const states: [string, Record<string, unknown>][] = [
    ["the smallest nudge at the left end of the dial", { nudgeW: 0, nudge: 0.01 }],
    ["the largest nudge at the right end of the dial", { nudgeW: 5, nudge: 1 }],
    ["the bottom of the bowl", { nudgeW: 3, nudge: 0.05 }],
    ["the slope cursor at both ends", { slopeW: 0 }],
    ["the slope cursor at the far end of the shelf", { slopeCurve: "shelf", slopeW: 6 }],
    ["a corner of the long valley", { surface: "valley", mapW: 0, mapB: 6, legs: "both" }],
    ["the flat centre of the saddle", { surface: "saddle", mapW: 3, mapB: 3, legs: "both" }],
    ["one leg on the bowl", { legs: "b", mapW: 6, mapB: 0 }],
    ["the chain with the dial at 0", { chainW: 0 }],
    ["the chain with the dial at 6", { chainW: 6 }],
    ["a walk that has hit the dial stop", { walkCurve: "bowl", walkStart: 5, walkStep: 1.2, walkSteps: 40 }],
    ["a walk that has settled", { walkSteps: 40 }],
    ["a walk from the dial's end", { walkStart: 0, walkStep: 0.05, walkSteps: 3 }],
    ["a challenge attempt that flew out of the valley", { raceStep: 1.1, raceSteps: 12, racePrev: 0.05 }],
    ["a challenge attempt part-way through", { raceStep: 0.75, raceSteps: 5, racePrev: 1 }],
  ];
  it.each(states)("with %s", (name, patch) => {
    for (const step of [0, 5]) expectClean(render(patch, step), name);
  });

  it("survives state that never went through hydrateState", () => {
    const junk = { nudge: "big", nudgeW: null, slopeCurve: 7, surface: "moon", legs: [], walkSteps: 1e9, raceStep: -4, racePrev: "x", chainW: 1e308 };
    expectClean(render(junk), "junk state");
    expect(visible(render(junk))).toContain("Reach the valley floor in 12 steps");
  });

  it("draws every scene's card in the scene's own setup", () => {
    expect(scenes.length).toBeGreaterThanOrEqual(4);
    for (const scene of scenes) {
      const html = render(scene.state ?? {}, scene.step ?? 0);
      expectClean(html, scene.id);
      expect(html, scene.id).toContain(`data-lab-surface="${scene.focus}"`);
      expect(LABELS[scene.step ?? 0], scene.id).toBe(scene.focus);
    }
  });
});

describe("number formatting", () => {
  it("uses a real minus sign and never prints a negative zero", () => {
    expect(fmt(-1.5)).toBe("−1.50");
    expect(fmt(-0.001)).toBe("0.00");
    expect(fmt(-0)).toBe("0.00");
    expect(signed(0)).toBe("+0.00");
    expect(signed(-0.004)).toBe("+0.00");
    expect(signed(2, 1)).toBe("+2.0");
    expect(fmt(Number.NaN)).toBe("0.00");
  });

  it("prints a reading exactly enough for the ratio to check by hand", () => {
    expect([3, 1.25, 0.5625, 0.21, 0.1025, 0.0201, 3.2201].map(fmtExact)).toEqual(["3.00", "1.25", "0.5625", "0.21", "0.1025", "0.0201", "3.2201"]);
    expect(fmtChange(-0.0299)).toBe("−0.0299");
    expect(fmtChange(1.25)).toBe("+1.25");
    expect(fmtChange(0.1 + 0.2 - 0.09)).toBe("+0.21");
  });

  it("snaps a dragged value to the slider's grid inside its range", () => {
    expect(snap(1.234, 0, 6)).toBe(1.25);
    expect(snap(-3, 0, 6)).toBe(0);
    expect(snap(9, 0, 5)).toBe(5);
    expect(snap(2.999, 0, 6)).toBe(3);
  });
});

describe("the 3D surface pictures", () => {
  it("keeps the flat map as the default and swaps it for the surface on request", () => {
    const flat = render({}, 2);
    expect(flat).toContain("sg-map");
    expect(flat).not.toContain("stage3d__frame");
    const lifted = render({ mapView: "surface" }, 2);
    expect(lifted).toContain("stage3d__frame");
    expect(lifted).not.toContain("sg-map");
    expectClean(lifted, "card 3 in 3D");
    // The card, its switch and its controls are the same either way.
    for (const html of [flat, lifted]) {
      expect(visible(html)).toContain("Flat map");
      expect(visible(html)).toContain("3D surface");
      expect(html).toContain('aria-label="Dial w"');
      expect(html).toContain('aria-label="Dial b"');
    }
  });

  it("describes the surface, the ball and the cut in words", () => {
    const html = render({ mapView: "surface", surface: "saddle", mapW: 4.6, mapB: 1.8, legs: "both" }, 2);
    const label = html.match(/<canvas[^>]*aria-label="([^"]*)"/)?.[1] ?? "";
    expect(label).toContain("a saddle");
    expect(label).toContain("w = 4.60, b = 1.80");
    expect(label).toContain("meter reads 5.56");
    expect(label).toContain("+1.60");
    expect(label).toContain("+1.20");
    expect(label).toContain("2.00 long");
    const cut = render({ mapView: "surface", legs: "b" }, 2);
    expect(cut.match(/<canvas[^>]*aria-label="([^"]*)"/)?.[1]).toContain("cuts the ground along dial b with w held still");
    expect(visible(cut)).toContain("Only b turns, so the ball can only travel along the coloured cut");
  });

  it("draws the challenge on the valley with the walk's reading in words", () => {
    const html = render({ raceView: "surface", raceStep: 0.9, raceSteps: 12, racePrev: 0.3 }, 5);
    expectClean(html, "card 6 in 3D");
    const label = html.match(/<canvas[^>]*aria-label="([^"]*)"/)?.[1] ?? "";
    expect(label).toContain("After 12 steps");
    expect(label).toContain("meter reads 1.012");
    expect(visible(html)).toContain("last try, step size 0.30");
    expect(visible(html)).toContain("goal met");
  });

  it("survives every surface, leg and end of the dials in 3D", () => {
    for (const surface of ["bowl", "valley", "saddle"]) {
      for (const legs of ["w", "b", "both"]) {
        for (const [mapW, mapB] of [[0, 0], [6, 6], [3, 3], [0, 6]]) {
          expectClean(render({ mapView: "surface", surface, legs, mapW, mapB }, 2), `${surface} ${legs} ${mapW} ${mapB}`);
        }
      }
    }
    expectClean(render({ mapView: "moon", raceView: 7 }), "junk picture choice");
  });
});
