import { describe, expect, it } from "vitest";
import {
  descend,
  forwardValues,
  knobCount,
  knobLayout,
  knobsPerLayer,
  loss,
  lossAndGradient,
  netOf,
  output,
  scrambled,
  seeded,
  spaced,
  stepKnobs,
} from "./net";

const dots = { xs: spaced(-2, 2, 9), ys: spaced(-2, 2, 9).map((x) => Math.sin(2 * x)) };

describe("the network", () => {
  it("counts one knob per wire and one per neuron after the input", () => {
    expect(knobCount([1, 2, 1])).toBe(7);
    expect(knobCount([1, 4, 1])).toBe(13);
    expect(knobCount([1, 6, 6, 1])).toBe(61);
    expect(knobCount([784, 16, 16, 10])).toBe(13002);
    expect(knobsPerLayer([784, 16, 16, 10])).toEqual([12560, 272, 170]);
    expect(knobCount([100, 100]) - knobCount([100])).toBe(10100);
    for (let units = 1; units <= 6; units += 1) expect(knobCount([1, units, 1])).toBe(3 * units + 1);
  });

  it("lays the knobs out in the order the forward pass reads them", () => {
    const layout = knobLayout([1, 2, 1]);
    expect(layout).toEqual([
      { layer: 1, to: 0, from: 0 },
      { layer: 1, to: 0, from: -1 },
      { layer: 1, to: 1, from: 0 },
      { layer: 1, to: 1, from: -1 },
      { layer: 2, to: 0, from: 0 },
      { layer: 2, to: 0, from: 1 },
      { layer: 2, to: 0, from: -1 },
    ]);
    expect(knobLayout([1, 6, 6, 1])).toHaveLength(61);
  });

  it("computes v₁·tanh(w₁x + b₁) + v₂·tanh(w₂x + b₂) + c for the seven-knob network", () => {
    const net = netOf([2]);
    const k = [2, 2, 2, -2, 1.5, -1.5, -1];
    for (const x of [-2, -0.5, 0, 1.25]) {
      const expected = 1.5 * Math.tanh(2 * x + 2) - 1.5 * Math.tanh(2 * x - 2) - 1;
      expect(output(net, k, x)).toBeCloseTo(expected, 12);
    }
    const values = forwardValues(net, k, 0.5);
    expect(values[0]).toEqual([0.5]);
    expect(values[1][0]).toBeCloseTo(Math.tanh(3), 12);
    expect(values[1][1]).toBeCloseTo(Math.tanh(-1), 12);
  });

  it.each([[[2]], [[4]], [[3, 3]], [[6, 6]]])("backpropagation matches nudging each knob, hidden layers %j", (hidden) => {
    const net = netOf(hidden);
    const knobs = scrambled(knobCount(net.sizes), 7, 1.5, 0.01);
    const { loss: value, gradient } = lossAndGradient(net, knobs, dots);
    expect(value).toBeCloseTo(loss(net, knobs, dots), 12);
    knobs.forEach((_, index) => {
      const up = [...knobs];
      const down = [...knobs];
      up[index] += 1e-5;
      down[index] -= 1e-5;
      const numeric = (loss(net, up, dots) - loss(net, down, dots)) / 2e-5;
      expect(gradient[index]).toBeCloseTo(numeric, 6);
    });
  });

  it("gives the same output when hidden neurons trade places or one flips all its signs", () => {
    const net = netOf([2]);
    const k = [0.7, -0.4, -1.3, 0.9, 1.1, -0.6, 0.2];
    const swapped = [k[2], k[3], k[0], k[1], k[5], k[4], k[6]];
    const flipped = [-k[0], -k[1], k[2], k[3], -k[4], k[5], k[6]];
    for (const x of spaced(-2, 2, 9)) {
      expect(output(net, swapped, x)).toBeCloseTo(output(net, k, x), 12);
      expect(output(net, flipped, x)).toBeCloseTo(output(net, k, x), 12);
    }
  });
});

describe("stepping", () => {
  it("moves every knob against its slope and stops it at the limit", () => {
    expect(stepKnobs([1, -1, 2.95], [0.8, -0.5, -1], 0.2, 3)).toEqual([1 - 0.16, -1 + 0.1, 3]);
    expect(stepKnobs([1], [Number.NaN], 0.2, 3)).toEqual([1]);
  });

  it("lowers the loss with a small step and keeps every setting it passes through", () => {
    const net = netOf([3]);
    const start = scrambled(10, 2, 1.5);
    const run = descend(net, start, dots, 0.05, 40, 8);
    expect(run.knobs).toHaveLength(41);
    expect(run.losses).toHaveLength(41);
    expect(run.knobs[0]).toEqual(start);
    expect(run.losses[40]).toBeLessThan(run.losses[0]);
    run.losses.forEach((value, step) => expect(value).toBeCloseTo(loss(net, run.knobs[step], dots), 12));
    for (let step = 1; step <= 40; step += 1) expect(run.losses[step]).toBeLessThanOrEqual(run.losses[step - 1] + 1e-9);
  });

  it("is repeatable, and momentum changes the path", () => {
    const net = netOf([4]);
    const start = scrambled(13, 5, 1.5);
    expect(descend(net, start, dots, 0.05, 30, 8, 0.9)).toEqual(descend(net, start, dots, 0.05, 30, 8, 0.9));
    expect(descend(net, start, dots, 0.05, 30, 8, 0.9).knobs[30]).not.toEqual(descend(net, start, dots, 0.05, 30, 8).knobs[30]);
  });

  it("never leaves the limits or produces a number that cannot be drawn, however large the step", () => {
    const net = netOf([6, 6]);
    for (const stepSize of [5, 1e6, Number.POSITIVE_INFINITY]) {
      const run = descend(net, scrambled(61, 3, 1.5), dots, stepSize, 60, 8, 0.9);
      for (const setting of run.knobs) for (const knob of setting) expect(Math.abs(knob)).toBeLessThanOrEqual(8);
      for (const value of run.losses) expect(Number.isFinite(value)).toBe(true);
    }
    const wild = descend(net, Array.from({ length: 61 }, () => Number.NaN), dots, 0.05, 3, 8);
    for (const value of wild.losses) expect(Number.isFinite(value)).toBe(true);
  });
});

describe("scrambling", () => {
  it("draws the same knobs for the same seed, on the grid and inside the spread", () => {
    expect(scrambled(7, 1, 1.5)).toEqual(scrambled(7, 1, 1.5));
    expect(scrambled(7, 1, 1.5)).not.toEqual(scrambled(7, 2, 1.5));
    for (let seed = 1; seed <= 40; seed += 1) {
      for (const knob of scrambled(19, seed, 1.5)) {
        expect(Math.abs(knob)).toBeLessThanOrEqual(1.5);
        expect(Math.abs(knob * 10 - Math.round(knob * 10))).toBeLessThan(1e-9);
        expect(Object.is(knob, -0)).toBe(false);
      }
    }
  });

  it("spreads its draws across the range", () => {
    const next = seeded(11);
    const draws = Array.from({ length: 2000 }, () => next());
    expect(Math.min(...draws)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...draws)).toBeLessThan(1);
    const mean = draws.reduce((total, value) => total + value, 0) / draws.length;
    expect(mean).toBeGreaterThan(0.45);
    expect(mean).toBeLessThan(0.55);
  });
});
