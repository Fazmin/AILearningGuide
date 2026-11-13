import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider } from "@app/module-sdk";
import cardInfo from "./card-info";
import definition from "./module";
import scenes from "./scenes";
import { count, countSaid, dialArc, duration, fmt, fmtLoss, knobAngle, onDial, signed, snapKnob } from "./kit";

const LABELS = [
  "A machine with two knobs",
  "Seven knobs by hand",
  "Let the machine turn the knobs",
  "More neurons, more shapes",
  "Count the knobs",
  "Playground: train a network",
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

  it("treats a step outside the lab as its nearest card", () => {
    expect(render({}, -3)).toBe(render({}, 0));
    expect(render({}, 99)).toBe(render({}, 5));
    expect(render({}, Number.NaN)).toBe(render({}, 0));
  });

  it("draws each card from the reader's settings", () => {
    const patch = { lineW: 2, k0: 2.5, auto: 30, units: 5, width: 256, pgWidth: 6, pgSteps: 200 };
    const cards = [...render(patch).matchAll(/<section class="lab-surface[^"]*" aria-label="([^"]*)"/g)];
    expect(cards.map((card) => card[1])).toEqual(LABELS);
    for (let step = 0; step < LABELS.length; step += 1) {
      expect(render(patch, step)).toBe(render(patch, step));
      expect(visible(render(patch, step))).not.toBe(visible(render({}, step)));
    }
  });

  it("gives every picture a text alternative that states its reading", () => {
    const html = render();
    const pictures = [...html.matchAll(/<svg class="nk-plot[^"]*"[^>]*role="(?:img|group)" aria-label="([^"]*)"/g)].map((match) => match[1]);
    // Six plots or sketches, the two-knob machine, the seven-knob diagram twice, and two walls of dials.
    expect(pictures.length).toBeGreaterThanOrEqual(10);
    for (const label of pictures) {
      expect(label.length).toBeGreaterThan(60);
      expect(label).toMatch(/\d/);
    }
  });

  it("makes every knob that can be turned a labelled, keyboard-reachable slider", () => {
    const machine = render({}, 0);
    const sliders = [...machine.matchAll(/<g class="nk-knob[^"]*" role="slider" tabindex="0" aria-label="([^"]*)" aria-valuemin="-3" aria-valuemax="3" aria-valuenow="([^"]*)" aria-valuetext="([^"]*)"/g)];
    expect(sliders.map((slider) => slider[1])).toEqual([
      "Knob w, on the machine: the input is multiplied by it",
      "Knob b, on the machine: it is added at the end",
    ]);
    expect(sliders.map((slider) => slider[2])).toEqual(["-0.5", "-1"]);
    expect(sliders.map((slider) => slider[3])).toEqual(["−0.50", "−1.00"]);
    for (const step of [1, 2]) {
      const network = render({}, step);
      const knobs = [...network.matchAll(/role="slider" tabindex="0" aria-label="(Knob [^"]*)"/g)].map((match) => match[1]);
      expect(knobs).toHaveLength(7);
      expect(knobs[0]).toBe("Knob w₁, the knob on the wire from the input to neuron 1");
      expect(knobs[6].replace(/&#x27;/g, "'")).toBe("Knob c, the output's own knob");
    }
    // The same two knobs are also plain range inputs, for anyone who would rather not drag a dial.
    for (const label of ["Knob w", "Knob b", "Input x"]) expect(machine).toContain(`aria-label="${label}"`);
    expect(render({}, 1)).toContain('aria-label="Turn the selected knob"');
  });

  it("draws turn hints only on the card where the machine steps, and none once the slopes are spent", () => {
    expect(render({}, 1)).not.toContain("nk-knob__hint");
    expect((render({}, 2).match(/class="nk-knob__hint"/g) ?? []).length).toBeGreaterThanOrEqual(5);
    // At the setting the dots were made with, every slope is 0: there is nothing to hint at.
    const solved = { k0: 2, k1: 2, k2: 2, k3: -2, k4: 1.5, k5: -1.5, k6: -1 };
    expect(render(solved, 2)).not.toContain("nk-knob__hint");
    expect(visible(render(solved, 2))).toContain("none: all near 0");
  });

  it("asks five predict-first questions and reveals nothing until they are answered", () => {
    const text = visible(render());
    expect((text.match(/Predict first/g) ?? []).length).toBe(5);
    expect(text).not.toContain("Not this time.");
    expect(text).not.toContain("Right.");
    const right = visible(render({ lineGuess: 0, comboGuess: 2, sameGuess: 1, shapeGuess: 2, wiresGuess: 1 }));
    expect((right.match(/Right\./g) ?? []).length).toBe(5);
    const wrong = visible(render({ lineGuess: 1, comboGuess: 0, sameGuess: 0, shapeGuess: 0, wiresGuess: 2 }));
    expect((wrong.match(/Not this time\./g) ?? []).length).toBe(5);
  });

  const states: [string, Record<string, unknown>][] = [
    ["both hand knobs at their stops", { lineW: 3, lineB: -3, lineX: -2 }],
    ["the two-knob machine fitted", { lineW: 0.8, lineB: 0.5, lineX: 2 }],
    ["every one of the seven knobs at a stop", { k0: 3, k1: -3, k2: 3, k3: -3, k4: 3, k5: -3, k6: 3, pick: 6 }],
    ["every one of the seven knobs at zero", { k0: 0, k1: 0, k2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }],
    ["the machine at its last step", { auto: 300 }],
    ["the seven knobs fitted exactly", { k0: 2, k1: 2, k2: 2, k3: -2, k4: 1.5, k5: -1.5, k6: -1, auto: 12 }],
    ["one hidden neuron on a late scramble", { units: 1, unitsSeed: 999 }],
    ["six hidden neurons on a stuck start", { units: 6, unitsSeed: 5 }],
    ["the largest network the sliders build", { inputs: 784, layers: 6, width: 4096, outputs: 10 }],
    ["the smallest network the sliders build", { inputs: 1, layers: 1, width: 2, outputs: 1 }],
    ["a playground run that has blown up", { pgWidth: 4, pgRate: 0.5, pgMomentum: "off", pgSteps: 600 }],
    ["the widest, deepest playground network part-way", { pgLayers: 2, pgWidth: 6, pgSteps: 123, pgDots: "step" }],
    ["the smallest playground network at the end", { pgWidth: 1, pgDots: "hill", pgSteps: 600, pgRate: 0.01 }],
    ["a playground run with no momentum on a late scramble", { pgSeed: 999, pgMomentum: "off", pgDots: "slope", pgSteps: 1 }],
  ];
  it.each(states)("with %s", (name, patch) => {
    for (let step = 0; step < LABELS.length; step += 1) expectClean(render(patch, step), name);
  });

  it("opens every lesson scene on the card it names", () => {
    expect(scenes.length).toBeGreaterThanOrEqual(8);
    for (const scene of scenes) {
      const html = render(scene.state ?? {}, scene.step ?? 0);
      expect(html, scene.id).toContain(`data-lab-surface="${scene.focus}"`);
      expectClean(html, scene.id);
    }
  });
});

describe("number formatting", () => {
  it("prints real minus signs and never a negative zero", () => {
    expect(fmt(-1.5)).toBe("−1.50");
    expect(fmt(-0.001)).toBe("0.00");
    expect(signed(0.8)).toBe("+0.80");
    expect(signed(-0.8)).toBe("−0.80");
    expect(signed(Number.NaN)).toBe("+0.00");
  });

  it("prints a loss at the precision that suits its size", () => {
    expect(fmtLoss(0.0019)).toBe("0.002");
    expect(fmtLoss(0.589)).toBe("0.589");
    expect(fmtLoss(4.785)).toBe("4.79");
    expect(fmtLoss(1595.034)).toBe("1,595");
    expect(fmtLoss(-1)).toBe("0.000");
    expect(fmtLoss(Number.POSITIVE_INFINITY)).toBe("0.000");
  });

  it("says big counts in words and long waits in the unit that fits", () => {
    expect(count(13002)).toBe("13,002");
    expect(countSaid(13002)).toBe("13,002");
    expect(countSaid(87_162_890)).toBe("87.2 million");
    expect(countSaid(175_000_000_000)).toBe("175 billion");
    expect(countSaid(2_000_000_000_000)).toBe("2 trillion");
    expect(duration(7)).toBe("7 seconds");
    expect(duration(13002)).toBe("3.6 hours");
    expect(duration(175_000_000_000)).toBe("5,500 years");
    expect(duration(86_400)).toBe("1 day");
  });
});

describe("the dial", () => {
  it("puts 0 straight up and the stops 135° either side", () => {
    expect(knobAngle(0, 3)).toBe(0);
    expect(knobAngle(3, 3)).toBe(135);
    expect(knobAngle(-3, 3)).toBe(-135);
    expect(knobAngle(99, 3)).toBe(135);
    expect(knobAngle(Number.NaN, 3)).toBe(0);
    const [x, y] = onDial(100, 100, 10, 0);
    expect(x).toBeCloseTo(100, 9);
    expect(y).toBeCloseTo(90, 9);
    const [rx, ry] = onDial(100, 100, 10, 90);
    expect(rx).toBeCloseTo(110, 9);
    expect(ry).toBeCloseTo(100, 9);
  });

  it("clicks in tenths, stops at its limits, and draws its arc the short or the long way round", () => {
    expect(snapKnob(0.84, 3, 0.1)).toBe(0.8);
    expect(snapKnob(7, 3, 0.1)).toBe(3);
    expect(snapKnob(-7, 3, 0.1)).toBe(-3);
    expect(Object.is(snapKnob(-0.01, 3, 0.1), -0)).toBe(false);
    expect(dialArc(0, 0, 10, -135, 135)).toContain(" 1 1 ");
    expect(dialArc(0, 0, 10, 0, 45)).toContain(" 0 1 ");
    expect(dialArc(0, 0, 10, 0, -45)).toContain(" 0 0 ");
  });
});
