import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import cardInfo from "./card-info";
import definition from "./module";
import {
  AUTO_GOAL,
  AUTO_MAX,
  HAND_START,
  HILL_DOTS,
  HILL_MADE_WITH,
  LANDMARKS,
  LINE_DOTS,
  PLAY_GOAL,
  RATE_STOPS,
  SMALL_NET,
  TRAIN_STEPS,
  autoRun,
  builtSizes,
  firstUnder,
  lineLoss,
  playNet,
  playRun,
  scrambleSmall,
  swapNeurons,
  wideRun,
  type PlaySetup,
} from "./model";
import { knobCount, knobsPerLayer, loss, lossAndGradient } from "./net";
import { duration, fmtLoss } from "./kit";

/**
 * Every number the three lessons, the checkpoint, the step instructions and the card explanations
 * quote, pinned twice: once against the lab's own arithmetic, and once against what the lab prints
 * with those settings.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const lesson = (mode: string) => readFileSync(path.join(here, "content", `${mode}.mdx`), "utf8").replace(/\s+/g, " ");
const LESSONS = { simple: lesson("simple"), plain: lesson("plain"), standard: lesson("standard") };
const everyLesson = (text: string | RegExp, where: string) => {
  for (const [mode, source] of Object.entries(LESSONS)) {
    if (typeof text === "string") expect(source.includes(text), `${mode} lesson should say "${text}" (${where})`).toBe(true);
    else expect(text.test(source), `${mode} lesson should match ${text} (${where})`).toBe(true);
  }
};
const cardText = (label: string) => JSON.stringify(cardInfo[label]);

const page = (step: number, patch: Record<string, string | number> = {}) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep: step,
      mode: "plain" as const,
      narrate: () => undefined,
    }),
  )
    // A second marker where each metric tile starts, so a tile is never confused with a goal bar of the same name.
    .replace(/<div class="metric[^"]*">/g, "\u0002")
    .replace(/<[^>]*>/g, "\u0001")
    .replace(/&#x27;/g, "'");
/** The value printed in the metric tile with this label. */
const tile = (text: string, label: string) => {
  const match = text.match(new RegExp(`\u0002\u0001*${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\u0001+([^\u0001\u0002]+)[\u0001\u0002]`));
  expect(match, `no tile labelled "${label}"`).not.toBeNull();
  return match![1];
};

const hill = (knobs: readonly number[]) => loss(SMALL_NET, knobs, HILL_DOTS);
const made = { k0: 2, k1: 2, k2: 2, k3: -2, k4: 1.5, k5: -1.5, k6: -1 };
const wave = (patch: Partial<PlaySetup> = {}): PlaySetup => ({ dots: "wave", layers: 1, width: 4, rate: 0.05, momentum: true, seed: 1, ...patch });

describe("card 1: two knobs", () => {
  it("opens at a loss of 4.79 and reaches 0.000 only at the setting the dots were made with", () => {
    expect(fmtLoss(lineLoss(-0.5, -1))).toBe("4.79");
    expect(lineLoss(-0.5, -1)).toBeCloseTo(4.785, 3);
    expect(fmtLoss(lineLoss(0.8, -1))).toBe("2.25");
    expect(lineLoss(0.8, 0.5)).toBe(0);
    expect(fmtLoss(lineLoss(0.8, 0.6))).toBe("0.010");
    expect(fmtLoss(lineLoss(0.9, 0.5))).toBe("0.015");
    // On the tenths grid no other setting gets under the goal.
    let fitted = 0;
    for (let w = -30; w <= 30; w += 1) for (let b = -30; b <= 30; b += 1) if (lineLoss(w / 10, b / 10) < 0.005) fitted += 1;
    expect(fitted).toBe(1);
    expect(LINE_DOTS.xs).toHaveLength(17);
    everyLesson("4.79", "opening loss");
    everyLesson("2.25", "loss with the tilt right");
    expect(cardText("A machine with two knobs")).toContain("4.785");
  });

  it("prints those readings", () => {
    const open = page(0);
    expect(tile(open, "Loss")).toBe("4.79");
    expect(tile(open, "Knob w")).toBe("−0.50");
    expect(open).toContain("(−0.50) × 1.00 − 1.00");
    expect(tile(page(0, { lineW: 0.8 }), "Loss")).toBe("2.25");
    const fitted = page(0, { lineW: 0.8, lineB: 0.5 });
    expect(tile(fitted, "Loss")).toBe("0.000");
    expect(fitted).toContain("every dot is on the line");
    // The predict-first answer: 2 × 0.5 + 1.
    expect(page(0, { lineW: 0.5, lineB: 1, lineX: 2 })).toContain("0.50 × 2.00 + 1.00");
  });
});

describe("card 2: seven knobs", () => {
  it("opens at 1.79 and gets worse before it gets better on the way to the made-with setting", () => {
    expect(fmtLoss(hill(HAND_START))).toBe("1.79");
    const one = [...HAND_START];
    [one[0], one[1], one[4]] = [2, 2, 1.5];
    expect(fmtLoss(hill(one))).toBe("2.81");
    const two = [...one];
    [two[2], two[3], two[5]] = [2, -2, -1.5];
    expect(fmtLoss(hill(two))).toBe("2.89");
    two[6] = -1;
    expect(two).toEqual([...HILL_MADE_WITH]);
    expect(hill(two)).toBe(0);
    for (const text of ["1.79", "2.81", "2.89"]) everyLesson(text, "hand-tuning walk-through");
    everyLesson(/ten million/i, "combinations of seven knobs");
    everyLesson("116 days", "one try per second");
    expect(Math.round(1e7 / 86400)).toBe(116);
  });

  it("leaves the curve where it was when the neurons trade places or one flips its signs", () => {
    expect(hill(swapNeurons(HAND_START))).toBeCloseTo(hill(HAND_START), 12);
    expect(hill(swapNeurons(HILL_MADE_WITH))).toBeCloseTo(0, 12);
    expect(hill([-2, -2, 2, -2, -1.5, -1.5, -1])).toBeCloseTo(0, 12);
    const swapped = swapNeurons(HAND_START);
    expect(swapped.filter((knob, index) => knob !== HAND_START[index])).toHaveLength(6);
  });

  it("prints those readings", () => {
    expect(tile(page(1), "Loss")).toBe("1.79");
    expect(tile(page(1, { k0: 2, k1: 2, k4: 1.5 }), "Loss")).toBe("2.81");
    expect(tile(page(1, { ...made, k6: 0.7 }), "Loss")).toBe("2.89");
    const fitted = page(1, made);
    expect(tile(fitted, "Loss")).toBe("0.000");
    expect(fitted).toContain("the curve follows the dots");
  });
});

describe("card 3: the machine turns the knobs", () => {
  const run = autoRun(HAND_START);

  it("takes the opening setting from 1.79 to 1.36 in one step and under 0.01 in 62", () => {
    expect(fmtLoss(run.losses[0])).toBe("1.79");
    expect(fmtLoss(run.losses[1])).toBe("1.36");
    expect(run.losses[1]).toBeCloseTo(1.358, 3);
    expect(firstUnder(run, AUTO_GOAL)).toBe(62);
    const slopes = lossAndGradient(SMALL_NET, HAND_START, HILL_DOTS).gradient;
    expect(slopes[0]).toBeCloseTo(-0.12, 2);
    // The strongest hint at the start is on v₁, and it says turn up.
    expect(slopes.map((slope) => Math.abs(slope)).indexOf(Math.max(...slopes.map((slope) => Math.abs(slope))))).toBe(4);
    expect(slopes[4]).toBeLessThan(0);
    everyLesson("1.36", "loss after one step");
    everyLesson(/\b62 steps\b/, "steps to the goal");
    expect(cardText("Let the machine turn the knobs")).toContain("after 62");
  });

  it("ends near a setting that is not the one the dots were made with", () => {
    const end = run.knobs[62];
    expect(end[0]).toBeCloseTo(-1.4, 1);
    expect(end[1]).toBeCloseTo(1.3, 1);
    expect(end[4]).toBeCloseTo(1.7, 1);
    expect(end.some((knob, index) => Math.abs(knob - HILL_MADE_WITH[index]) > 1)).toBe(true);
    everyLesson("−1.4", "where w₁ ends");
  });

  it("reaches the goal from the opening setting and the next nine scrambles, between step 46 and step 147", () => {
    const steps = Array.from({ length: 10 }, (_, index) => firstUnder(autoRun(scrambleSmall(index + 1)), AUTO_GOAL));
    expect(Math.min(...steps)).toBe(46);
    expect(Math.max(...steps)).toBe(147);
    expect(steps.every((step) => step > 0 && step <= AUTO_MAX)).toBe(true);
    everyLesson("46", "fastest of ten starts");
    everyLesson("147", "slowest of ten starts");
  });

  it("prints those readings", () => {
    const open = page(2);
    expect(tile(open, "Loss")).toBe("1.79");
    expect(open).toContain("−1.50 − 0.2 × (−0.12)");
    expect(tile(open, "Strongest hint")).toBe("v₁: turn up");
    expect(tile(page(2, { auto: 1 }), "Loss")).toBe("1.36");
    const done = page(2, { auto: 62 });
    expect(done).toContain("goal met after 62 steps");
    expect(done).toContain("the knobs do not show the numbers the dots were made with");
    expect(page(2, { auto: 61 })).not.toContain("goal met");
  });

  it("the checkpoint's step is the rule the card applies", () => {
    expect(1 - 0.2 * 0.8).toBeCloseTo(0.84, 12);
  });
});

describe("card 4: more neurons", () => {
  const final = (units: number, seed: number) => fmtLoss(wideRun(units, seed).losses[TRAIN_STEPS]);

  it("leaves 0.648, 0.589, 0.299, 0.002, 0.003 and 0.003 at the opening start", () => {
    expect([1, 2, 3, 4, 5, 6].map((units) => final(units, 1))).toEqual(["0.648", "0.589", "0.299", "0.002", "0.003", "0.003"]);
    for (const text of ["0.648", "0.589", "0.299", "0.002"]) everyLesson(text, "losses by width");
    expect(cardText("More neurons, more shapes")).toContain("0.648, 0.589, 0.299, 0.002, 0.003 and 0.003");
  });

  it("keeps one neuron above 0.6 at every start tried", () => {
    for (let seed = 1; seed <= 10; seed += 1) expect(wideRun(1, seed).losses[TRAIN_STEPS]).toBeGreaterThan(0.6);
  });

  it("gets four neurons stuck after two scrambles, while three do better", () => {
    expect(final(4, 2)).toBe("0.003");
    expect(final(4, 3)).toBe("0.290");
    expect(final(3, 3)).toBe("0.018");
    everyLesson("0.290", "four neurons stuck");
  });

  it("prints those readings", () => {
    expect(tile(page(3), "Loss after 600 steps")).toBe("0.648");
    const four = page(3, { units: 4 });
    expect(tile(four, "Loss after 600 steps")).toBe("0.002");
    expect(tile(four, "Knobs")).toBe("13");
    expect(four).toContain("passes through the dots");
    const stuck = page(3, { units: 4, unitsSeed: 3 });
    expect(tile(stuck, "Loss after 600 steps")).toBe("0.290");
    expect(stuck).toContain("the smallest network on this card that gets the loss under 0.01 has 5 neurons");
    expect(stuck).toContain("Whether this size could do it from a better start is a separate question");
    // A smaller network fitted from the same start: this one had the knobs and got stuck.
    const smallerFits = page(3, { units: 5, unitsSeed: 5 });
    expect(tile(smallerFits, "Loss after 600 steps")).toBe("0.267");
    expect(smallerFits).toContain("A network with only 4 neurons got under 0.01 from this same start");
  });
});

describe("card 5: count the knobs", () => {
  it("counts 13,002 for the digit reader and 10,100 for a row of 100 after a row of 100", () => {
    const reader = builtSizes(784, 2, 16, 10);
    expect(knobsPerLayer(reader)).toEqual([12560, 272, 170]);
    expect(knobCount(reader)).toBe(13002);
    expect(100 * 100 + 100).toBe(10100);
    expect(200 * 200 + 200).toBe(40200);
    expect(knobsPerLayer(builtSizes(784, 2, 64, 10))[1]).toBe(4160);
    expect(4160 / 272).toBeGreaterThan(15);
    expect(knobCount(builtSizes(784, 6, 4096, 10))).toBe(87162890);
    for (const text of ["13,002", "12,560", "10,100", "4,160"]) everyLesson(text, "knob counts");
  });

  it("uses the counts the four papers report", () => {
    expect(LANDMARKS.map((mark) => mark.knobs)).toEqual([60_000, 60_000_000, 1_500_000_000, 175_000_000_000]);
    expect(duration(13002)).toBe("3.6 hours");
    expect(duration(175_000_000_000)).toBe("5,500 years");
    // With card 2's bar one millimetre long, GPT-3's is 25 billion millimetres: 25,000 kilometres.
    expect(175e9 / 7 / 1e6).toBe(25000);
    for (const text of ["60 thousand", "60 million", "1.5 billion", "175 billion", "5,500 years"]) everyLesson(text, "published counts");
  });

  it("prints those readings", () => {
    const reader = page(4, { inputs: 784, layers: 2, width: 16, outputs: 10 });
    expect(reader).toContain("12,560 + 272 + 170");
    expect(tile(reader, "Knobs in your network")).toBe("13,002");
    expect(tile(reader, "One second per knob")).toBe("3.6 hours");
    expect(tile(page(4), "Knobs in your network")).toBe("7");
    expect(tile(page(4), "GPT-3 is larger by")).toBe("× 25 billion");
    expect(tile(page(4, { inputs: 784, layers: 6, width: 4096, outputs: 10 }), "Knobs in your network")).toBe("87.2 million");
    expect(page(4)).toContain("about 5,500 years");
  });
});

describe("card 6: the playground", () => {
  const end = (setup: PlaySetup) => fmtLoss(playRun(setup).losses[TRAIN_STEPS]);

  it("stalls at 0.589 with 2 neurons and passes the goal at step 188 with 4", () => {
    expect(knobCount(playNet(wave({ width: 2 })).sizes)).toBe(7);
    expect(end(wave({ width: 2 }))).toBe("0.589");
    expect(firstUnder(playRun(wave({ width: 2 })), PLAY_GOAL)).toBe(-1);
    expect(knobCount(playNet(wave()).sizes)).toBe(13);
    expect(firstUnder(playRun(wave()), PLAY_GOAL)).toBe(188);
    expect(end(wave())).toBe("0.002");
    everyLesson("188", "step at which 4 neurons pass the goal");
  });

  it("misses without momentum, blows up with long plain steps, and passes sooner with two layers", () => {
    expect(end(wave({ momentum: false }))).toBe("0.063");
    expect(end(wave({ rate: 0.5, momentum: false }))).toBe("1,595");
    expect(knobCount(playNet(wave({ layers: 2 })).sizes)).toBe(33);
    expect(firstUnder(playRun(wave({ layers: 2 })), PLAY_GOAL)).toBe(76);
    expect(knobCount(playNet(wave({ layers: 2, width: 2 })).sizes)).toBe(13);
    expect(end(wave({ layers: 2, width: 2 }))).toBe("0.580");
    for (const text of ["0.063", "1,595", "33", "76"]) everyLesson(text, "playground variations");
  });

  it("finds no network under 13 knobs that fits the wave, across eight starts and every setting", () => {
    let fewest = Number.POSITIVE_INFINITY;
    for (let seed = 1; seed <= 8; seed += 1) {
      for (const layers of [1, 2]) {
        for (let width = 1; width <= 3; width += 1) {
          const knobs = knobCount(playNet({ layers, width }).sizes);
          if (knobs >= 13) continue;
          for (const rate of RATE_STOPS) {
            for (const momentum of [true, false]) {
              if (playRun({ dots: "wave", layers, width, rate, momentum, seed }).losses[TRAIN_STEPS] < PLAY_GOAL) fewest = Math.min(fewest, knobs);
            }
          }
        }
      }
    }
    expect(fewest).toBe(Number.POSITIVE_INFINITY);
  }, 30000);

  it("prints those readings", () => {
    const first = page(5, { pgSteps: 600 });
    expect(tile(first, "Loss")).toBe("0.589");
    expect(tile(first, "Knobs")).toBe("7");
    const four = page(5, { pgWidth: 4, pgSteps: 600 });
    expect(four).toContain("goal met with 13 knobs");
    expect(four).toContain("first dips under 0.01 at step 188");
    expect(page(5, { pgWidth: 4, pgSteps: 187 })).not.toContain("goal met");
    const blown = page(5, { pgWidth: 4, pgRate: 0.5, pgMomentum: "off", pgSteps: 600 });
    expect(tile(blown, "Loss")).toBe("1,595");
    expect(blown).toContain("try a smaller Step size");
  });
});

describe("what the module says about itself", () => {
  it("quotes the same figures in its step instructions and card explanations", () => {
    expect(definition.stepInstructions[0]).toContain("0.000");
    expect(definition.stepInstructions[3]).toContain("600");
    const play = cardText("Playground: train a network");
    for (const text of ["0.589", "step 188", "0.063", "1,595", "step 76", "0.580", "fewer than 13 knobs"]) expect(play, text).toContain(text);
    const countCard = cardText("Count the knobs");
    for (const text of ["13,002", "87.2 million", "3.6 hours", "5,500 years"]) expect(countCard, text).toContain(text);
    const hand = cardText("Seven knobs by hand");
    for (const text of ["1.790", "ten million"]) expect(hand, text).toContain(text);
    const auto = cardText("Let the machine turn the knobs");
    for (const text of ["1.358", "step 46", "step 147"]) expect(auto, text).toContain(text);
    const wide = cardText("More neurons, more shapes");
    for (const text of ["0.290", "0.018"]) expect(wide, text).toContain(text);
  });

  it("sizes the checkpoint's widening question correctly", () => {
    expect(16 * 16 + 16).toBe(272);
    expect(64 * 64 + 64).toBe(4160);
    expect(Math.round(4160 / 272)).toBe(15);
  });
});
