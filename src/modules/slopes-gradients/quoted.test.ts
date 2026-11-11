import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  RACE_BUDGET,
  RACE_FLOOR,
  RACE_START,
  RACE_TARGET,
  chainReading,
  curveSlope,
  curveValue,
  flatSpots,
  nudgeRatio,
  raceReadings,
  surfaceGradient,
  surfaceValue,
  walk,
} from "./curves";

/**
 * Every number the three lessons, the checkpoint and the card explanations quote, pinned twice:
 * once against the formulas, and once against what the lab actually prints with those settings.
 */

const MINUS = "−";
/** The lab shows one card per step, so the text of the whole lab is every step's page in turn. */
const render = (patch: Record<string, string | number> = {}) =>
  definition.steps
    .map((_, currentStep) =>
      renderToStaticMarkup(
        createElement(definition.Explore, {
          state: { ...definition.initialState, ...patch },
          setState: () => undefined,
          currentStep,
          mode: "plain" as const,
          narrate: () => undefined,
        }),
      ),
    )
    .join("")
    .replace(/<[^>]*>/g, "\u0001")
    .replace(/&#x27;/g, "'");

/** The visible text of one card: everything between its heading and the next card's. */
const card = (text: string, title: string, nextTitle?: string) => {
  const start = text.indexOf(title);
  expect(start, `card "${title}" not found`).toBeGreaterThan(-1);
  const end = nextTitle ? text.indexOf(nextTitle, start) : text.length;
  return text.slice(start, end);
};
const TITLES = [
  "Nudge the dial and see how far the meter moves",
  "Read the slope at every dial setting",
  "Two dials: one slope each, then one arrow",
  "A dial that reaches the meter through another machine",
  "Walk downhill, one step at a time",
  "Reach the valley floor in 12 steps",
];
const cardText = (index: number, patch: Record<string, string | number> = {}) =>
  card(render(patch), TITLES[index], TITLES[index + 1]);
const round = (value: number, digits = 2) => Number(value.toFixed(digits));

describe("card 1, nudge the dial", () => {
  it("at w = 4 the meter reads 2, and a nudge of 1 takes it to 5: a ratio of 3", () => {
    expect(curveValue("bowl", 4)).toBe(2);
    expect(curveValue("bowl", 5)).toBe(5);
    expect(nudgeRatio("bowl", 4, 1)).toBe(3);
    const text = cardText(0);
    for (const printed of ["Meter at w = 4.00", "2.00", "Meter at w = 5.00", "5.00", "3.000", "+3.00 ÷ +1.00"]) {
      expect(text).toContain(printed);
    }
  });

  it("the ratio reads 2.5, 2.25, 2.1, 2.05 and 2.01 as the nudge shrinks, settling on 2", () => {
    const ladder: [number, number, string][] = [
      [0.5, 2.5, "2.500"],
      [0.25, 2.25, "2.250"],
      [0.1, 2.1, "2.100"],
      [0.05, 2.05, "2.050"],
      [0.01, 2.01, "2.010"],
    ];
    for (const [nudge, ratio, printed] of ladder) {
      expect(nudgeRatio("bowl", 4, nudge)).toBeCloseTo(ratio, 9);
      const text = cardText(0, { nudge });
      expect(text).toContain(printed);
      expect(text).toContain("Slope it settles on");
    }
    expect(curveSlope("bowl", 4)).toBe(2);
    expect(curveValue("bowl", 4.5)).toBe(3.25);
    expect(cardText(0, { nudge: 0.5 })).toContain("+1.25 ÷ +0.50");
    expect(cardText(0, { nudge: 0.5 })).toContain("3.25");
  });

  it("the predict-first reveal quotes 3.00 sliding to 2.50 and settling on 2.00", () => {
    const text = cardText(0, { nudgeGuess: 2 });
    expect(text).toContain("3.00");
    expect(text).toContain("2.50");
    expect(text).toContain("settles on 2.00");
  });

  it("at w = 1.5 the slope is −3, and at w = 3 it is 0", () => {
    expect(curveSlope("bowl", 1.5)).toBe(-3);
    expect(curveSlope("bowl", 3)).toBe(0);
    expect(cardText(0, { nudgeW: 1.5, nudge: 0.01 })).toContain(`${MINUS}3.00`);
    expect(cardText(0, { nudgeW: 3 })).toContain("0.00");
  });
});

describe("card 2, a slope at every point", () => {
  it("at w = 1 the meter reads 2.79 and the slope is −1.82, so downhill is to the right", () => {
    expect(round(curveValue("valleys", 1))).toBe(2.79);
    expect(round(curveSlope("valleys", 1))).toBe(-1.82);
    const text = cardText(1);
    for (const printed of ["2.79", `${MINUS}1.82`, "to the right"]) expect(text).toContain(printed);
  });

  it("the slope is zero at 1.57, 2.88 and 4.62, where the meter reads 2.28, 3.12 and 1.36", () => {
    const spots = flatSpots("valleys");
    expect(spots.map((spot) => [round(spot.w), round(spot.value), spot.kind])).toEqual([
      [1.57, 2.28, "bottom"],
      [2.88, 3.12, "top"],
      [4.62, 1.36, "bottom"],
    ]);
    for (const [w, meter, kind] of [
      [1.57, "2.28", "a bottom"],
      [2.88, "3.12", "a top"],
      [4.62, "1.36", "a bottom"],
    ] as const) {
      const text = cardText(1, { slopeW: w });
      expect(text).toContain(meter);
      expect(text).toContain(`This flat spot is ${kind}`);
      expect(text).toContain("level");
    }
  });

  it("the shelf is level at w = 3, where the meter reads 5.00, and lower all the way to the left", () => {
    expect(curveValue("shelf", 3)).toBe(5);
    expect(curveSlope("shelf", 3)).toBe(0);
    for (let w = 0; w < 3; w += 0.25) expect(curveValue("shelf", w)).toBeLessThan(5);
    const text = cardText(1, { slopeCurve: "shelf", slopeW: 3 });
    expect(text).toContain("5.00");
    expect(text).toContain("This flat spot is a shelf");
  });
});

describe("card 3, two dials", () => {
  it("on the round bowl at w = 4.5, b = 4 the meter reads 2.63 and the slopes are +1.50 and +1.00", () => {
    expect(surfaceValue("bowl", 4.5, 4)).toBe(2.625);
    expect(surfaceGradient("bowl", 4.5, 4)).toEqual({ x: 1.5, y: 1 });
    expect(round(Math.hypot(1.5, 1))).toBe(1.8);
    expect(surfaceValue("bowl", 3, 3)).toBe(1);
    const text = cardText(2);
    for (const printed of ["2.63", "+1.50", "+1.00", "1.80", "(+1.50, +1.00)"]) expect(text).toContain(printed);
  });

  it("on the long valley at w = 1, b = 4 the slopes are −0.50 and +2.00", () => {
    expect(surfaceGradient("valley", 1, 4)).toEqual({ x: -0.5, y: 2 });
    const text = cardText(2, { surface: "valley", mapW: 1, mapB: 4, legs: "both" });
    expect(text).toContain(`(${MINUS}0.50, +2.00)`);
    // Straight away from the lowest point would be the direction (−2, +1); the gradient is not that.
    const away = { x: 1 - 3, y: 4 - 3 };
    expect(Math.abs(away.x * 2 - away.y * -0.5)).toBeGreaterThan(1);
  });

  it("the saddle is flat at its centre, reads 5.00 there, 7.00 at w = 5 and 3.00 at b = 5", () => {
    expect(surfaceValue("saddle", 3, 3)).toBe(5);
    expect(surfaceValue("saddle", 5, 3)).toBe(7);
    expect(surfaceValue("saddle", 3, 5)).toBe(3);
    expect(cardText(2, { surface: "saddle", mapW: 3, mapB: 3, legs: "both" })).toContain("5.00");
    expect(cardText(2, { surface: "saddle", mapW: 5, mapB: 3 })).toContain("7.00");
    expect(cardText(2, { surface: "saddle", mapW: 3, mapB: 5 })).toContain("3.00");
  });
});

describe("card 4, the chain", () => {
  it("at w = 3 the middle reading is 9, the meter 5, the slopes 3 and 2, and the overall slope 6", () => {
    const reading = chainReading(3);
    expect(reading).toMatchObject({ u: 9, meter: 5, slopeA: 3, slopeB: 2, overall: 6 });
    expect(round(reading.predictedU, 3)).toBe(0.3);
    expect(round(reading.predictedMeter, 3)).toBe(0.6);
    expect(round(reading.actualMeter, 3)).toBe(0.602);
    const text = cardText(3);
    for (const printed of ["9.00", "5.00", "× 3.00", "× 2.00", "3.00 × 2.00", "6.00", "+0.30", "+0.60", "+0.600", "+0.602"]) {
      expect(text).toContain(printed);
    }
  });

  it("at w = 5 machine A's slope is 4.00, machine B's is 0.029, the overall slope 0.12 and the meter 9.96", () => {
    const reading = chainReading(5);
    expect(reading.slopeA).toBe(4);
    expect(round(reading.slopeB, 3)).toBe(0.029);
    expect(round(reading.overall)).toBe(0.12);
    expect(round(reading.meter)).toBe(9.96);
    expect(round(reading.predictedMeter, 3)).toBe(0.012);
    const text = cardText(3, { chainW: 5 });
    for (const printed of ["× 4.00", "× 0.029", "0.12", "9.96", "+0.012"]) expect(text).toContain(printed);
  });
});

describe("card 5, walking downhill", () => {
  it("the first step from 2.6 on two valleys is 2.600 − 0.20 × 0.608 = 2.478", () => {
    expect(round(curveSlope("valleys", 2.6), 3)).toBe(0.608);
    expect(round(walk("valleys", 2.6, 0.2, 1)[1], 3)).toBe(2.478);
    const text = cardText(4);
    expect(text).toContain(`2.600 ${MINUS} 0.20 × 0.608`);
    expect(text).toContain("2.478");
  });

  it("ten steps end in the left valley near 1.57, where the meter reads 2.28; the right valley reads 1.36", () => {
    const end = walk("valleys", 2.6, 0.2, 10).pop()!;
    expect(round(end)).toBe(1.57);
    expect(round(curveValue("valleys", end))).toBe(2.28);
    const [left, hump, right] = flatSpots("valleys");
    expect(2.6).toBeLessThan(hump.w);
    expect(round(left.value)).toBe(2.28);
    expect(round(right.value)).toBe(1.36);
    const text = cardText(4, { walkSteps: 10 });
    for (const printed of ["1.574", "2.279", "L 1.36", "L 2.28"]) expect(text).toContain(printed);
  });

  it("on the bowl from 5: 0.05 reaches 3.697 in ten steps and 0.5 lands on 3 in one", () => {
    expect(round(walk("bowl", 5, 0.05, 10).pop()!, 3)).toBe(3.697);
    expect(walk("bowl", 5, 0.5, 1)).toEqual([5, 3]);
    expect(cardText(4, { walkCurve: "bowl", walkStart: 5, walkStep: 0.05, walkSteps: 10 })).toContain("3.697");
    expect(cardText(4, { walkCurve: "bowl", walkStart: 5, walkStep: 0.5, walkSteps: 1 })).toContain("3.000");
  });

  it("on the bowl from 5: 0.9 jumps to 1.4 and back to 4.28, with the bounces shrinking", () => {
    const path = walk("bowl", 5, 0.9, 12);
    expect(round(path[1])).toBe(1.4);
    expect(round(path[2])).toBe(4.28);
    const distances = path.map((w) => Math.abs(w - 3));
    distances.slice(1).forEach((distance, index) => expect(distance).toBeLessThan(distances[index]));
    expect(cardText(4, { walkCurve: "bowl", walkStart: 5, walkStep: 0.9, walkSteps: 2 })).toContain("4.280");
  });

  it("on the bowl from 5: 1.1 makes the meter climb from 5.00 to 6.76 to 9.29 and then hits the dial stop", () => {
    const path = walk("bowl", 5, 1.1, 3);
    expect(path.slice(0, 3).map((w) => round(curveValue("bowl", w)))).toEqual([5, 6.76, 9.29]);
    expect(path[3]).toBe(0);
    expect(cardText(4, { walkCurve: "bowl", walkStart: 5, walkStep: 1.1, walkSteps: 1 })).toContain("6.760");
    expect(cardText(4, { walkCurve: "bowl", walkStart: 5, walkStep: 1.1, walkSteps: 2 })).toContain("9.294");
  });

  it("on two valleys a step size of 0.8 never settles, alternating between about 1.30 and 1.98", () => {
    const path = walk("valleys", 2.6, 0.8, 40);
    expect(round(path[39])).toBe(1.98);
    expect(round(path[40])).toBe(1.3);
    expect(round(path[37])).toBe(1.98);
    expect(round(path[38])).toBe(1.3);
    // The left valley's curvature is about 2.86, so its stable step sizes end near 2 / 2.86 = 0.70.
    const left = flatSpots("valleys")[0].w;
    const curvature = (curveSlope("valleys", left + 1e-4) - curveSlope("valleys", left - 1e-4)) / 2e-4;
    expect(round(curvature)).toBe(2.86);
  });
});

describe("card 6, the challenge", () => {
  const end = (stepSize: number) => raceReadings(stepSize, RACE_BUDGET)[RACE_BUDGET];

  it("starts at w = 0.5, b = 4.5 where the meter reads 4.03, with a floor of 1.00 and a goal of 1.05 in 12 steps", () => {
    expect(RACE_START).toEqual({ x: 0.5, y: 4.5 });
    expect(round(raceReadings(0.1, 0)[0])).toBe(4.03);
    expect([RACE_FLOOR, RACE_TARGET, RACE_BUDGET]).toEqual([1, 1.05, 12]);
    const text = cardText(5);
    for (const printed of ["4.03", "goal under 1.05", "Run all 12 steps", "reads 1.00"]) expect(text).toContain(printed);
  });

  it("ends at 1.436 with a step size of 0.10, at 3.25 with 1.00, and at 1.016 with 0.60", () => {
    expect(round(end(0.1), 3)).toBe(1.436);
    expect(round(end(1))).toBe(3.25);
    expect(round(end(0.6), 3)).toBe(1.016);
    expect(round(0.975 ** 12, 3)).toBe(0.738);
    expect(cardText(5, { raceStep: 0.1, raceSteps: 12 })).toContain("1.436");
    expect(cardText(5, { raceStep: 1, raceSteps: 12 })).toContain("3.25");
    const won = cardText(5, { raceStep: 0.6, raceSteps: 12 });
    expect(won).toContain("1.016");
    expect(won).toContain("goal met");
    expect(won).toContain("You just chose a learning rate");
  });

  it("is won by every step size from 0.45 to 0.90 on the slider's grid, and by no other", () => {
    for (let index = 1; index <= 22; index += 1) {
      const stepSize = round(index * 0.05);
      expect(end(stepSize) < RACE_TARGET, `step size ${stepSize}`).toBe(stepSize >= 0.45 && stepSize <= 0.9);
    }
  });

  it("tells a crawl from a bounce when the steps run out", () => {
    expect(cardText(5, { raceStep: 0.1, raceSteps: 12 })).toContain("Try a bigger step size");
    expect(cardText(5, { raceStep: 1, raceSteps: 12 })).toContain("Try a smaller step size");
    expect(cardText(5, { raceStep: 0.95, raceSteps: 12 })).toContain("Try a smaller step size");
    expect(cardText(5, { raceStep: 0.4, raceSteps: 12 })).toContain("Try a bigger step size");
  });

  it("prints the first step at the default step size as −0.10 × (−0.63, +3.00)", () => {
    expect(surfaceGradient("valley", 0.5, 4.5)).toEqual({ x: -0.625, y: 3 });
    expect(cardText(5)).toContain(`${MINUS}0.10 × (${MINUS}0.63, +3.00)`);
  });
});

describe("numbers quoted outside the lessons", () => {
  it("the checkpoint's chain question: slopes of 4 and 0.03 move the meter by about 0.012", () => {
    expect(round(0.1 * 4 * 0.03, 3)).toBe(0.012);
    const options = definition.checkpoint as { prompt: string; options: string[]; answer: number }[];
    const chain = options.find((question) => question.prompt.includes("machine A's slope is 4"))!;
    expect(chain.options[chain.answer]).toContain("0.012");
  });

  it("every step instruction names a control the lab really has", () => {
    const text = render();
    const named = [
      "Predict first",
      "Nudge size",
      "Ratio at this nudge",
      "Slope it settles on",
      "Dial w",
      "Two valleys",
      "Slope here",
      "Meter curve",
      "A shelf",
      "w only",
      "b only",
      "Both together",
      "Dials turning",
      "Surface",
      "Long valley",
      "Overall slope",
      "Take 10 steps",
      "Bowl",
      "Start w",
      "Step size",
      "Run all 12 steps",
      "Meter L",
    ];
    const instructions = definition.stepInstructions.join(" ");
    for (const name of named) {
      expect(instructions, `instructions mention ${name}`).toContain(name);
      expect(text, `the lab shows ${name}`).toContain(name);
    }
  });
});
