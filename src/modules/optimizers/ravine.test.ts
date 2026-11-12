import { describe, expect, it } from "vitest";
import {
  RAVINE_OPTIMIZERS,
  RAVINE_RANGE,
  RAVINE_STEPS,
  SGD_LIMIT,
  ravine,
  ravineFloor,
  ravineLost,
  ravineRate,
  ravineSettled,
  ravineStart,
  ravineWalk,
  type RavineOptimizer,
  type RavineRates,
  type RavineTurn,
} from "./ravine";

const settled = (optimizer: RavineOptimizer, turn: RavineTurn, rates: RavineRates) =>
  ravineSettled(ravineWalk(optimizer, turn, ravineRate(optimizer, rates)));

describe("the ravine", () => {
  it("is a quadratic valley a hundred times steeper across than along, with slopes that match the loss", () => {
    for (const turn of ["axis", "diagonal"] as const) {
      expect(ravine(turn, 0, 0).loss).toBe(0);
      const h = 1e-5;
      for (const [x, y] of [[-0.7, 0.3], [0.4, -0.9], [0.8, 0.8]]) {
        const here = ravine(turn, x, y);
        expect(here.gx).toBeCloseTo((ravine(turn, x + h, y).loss - ravine(turn, x - h, y).loss) / (2 * h), 5);
        expect(here.gy).toBeCloseTo((ravine(turn, x, y + h).loss - ravine(turn, x, y - h).loss) / (2 * h), 5);
      }
    }
    expect(ravine("axis", 1, 0).loss).toBeCloseTo(0.009, 10);
    expect(ravine("axis", 0, 1).loss).toBeCloseTo(0.9, 10);
  });

  it("starts every walk the same distance along the floor and up the wall, inside the drawn square", () => {
    expect(ravineStart("axis")).toEqual({ x: -0.8, y: 0.4 });
    for (const turn of ["axis", "diagonal"] as const) {
      const start = ravineStart(turn);
      expect(ravine(turn, start.x, start.y).loss).toBeCloseTo(0.14976, 10);
      expect(Math.max(Math.abs(start.x), Math.abs(start.y))).toBeLessThan(RAVINE_RANGE[1]);
      for (const end of ravineFloor(turn)) {
        expect(Math.max(Math.abs(end.x), Math.abs(end.y))).toBeLessThanOrEqual(RAVINE_RANGE[1]);
        expect(ravine(turn, end.x, end.y).loss).toBeLessThan(0.02);
      }
    }
  });
});

describe("the three walks", () => {
  it("settle in the number of steps the card and the lesson quote", () => {
    expect(settled("sgd", "axis", "default")).toBe(49);
    expect(settled("momentum", "axis", "default")).toBe(59);
    expect(settled("adam", "axis", "default")).toBe(35);
    expect(settled("sgd", "axis", "half")).toBe(97);
    expect(settled("momentum", "axis", "half")).toBe(99);
    expect(settled("adam", "axis", "half")).toBe(52);
    expect(settled("momentum", "axis", "double")).toBe(46);
    expect(settled("adam", "axis", "double")).toBe(35);
  });

  it("send SGD out of the ravine at double its rate, which is past its limit", () => {
    expect(ravineRate("sgd", "double")).toBeGreaterThan(SGD_LIMIT);
    expect(ravineRate("sgd", "default")).toBeLessThan(SGD_LIMIT);
    const walk = ravineWalk("sgd", "axis", ravineRate("sgd", "double"));
    expect(ravineLost(walk)).toBe(true);
    expect(ravineSettled(walk)).toBeNull();
  });

  it("turn with the ravine for SGD and momentum, and not for Adam", () => {
    for (const rates of ["half", "default", "double"] as const) {
      expect(settled("sgd", "diagonal", rates)).toBe(settled("sgd", "axis", rates));
      expect(settled("momentum", "diagonal", rates)).toBe(settled("momentum", "axis", rates));
    }
    // Adam sizes its step per weight, so a ravine that runs between the weights takes that help away.
    expect(settled("adam", "diagonal", "default")).toBeNull();
    expect(settled("adam", "diagonal", "half")).toBeNull();
    expect(settled("adam", "diagonal", "double")).toBe(105);
    const turned = ravineWalk("adam", "diagonal", ravineRate("adam", "default"));
    expect(turned[RAVINE_STEPS].loss).toBeGreaterThan(0.001);
    expect(turned[RAVINE_STEPS].loss).toBeLessThan(0.003);
  });

  it("never leave the drawn square while they are still running", () => {
    for (const { id } of RAVINE_OPTIMIZERS) {
      for (const turn of ["axis", "diagonal"] as const) {
        for (const rates of ["half", "default", "double"] as const) {
          const walk = ravineWalk(id, turn, ravineRate(id, rates));
          if (ravineLost(walk)) continue;
          for (const point of walk) {
            expect(Math.abs(point.x)).toBeLessThanOrEqual(RAVINE_RANGE[1]);
            expect(Math.abs(point.y)).toBeLessThanOrEqual(RAVINE_RANGE[1]);
          }
        }
      }
    }
  });
});
