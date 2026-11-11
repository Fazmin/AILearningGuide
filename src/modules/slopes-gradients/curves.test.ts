import { describe, expect, it } from "vitest";
import {
  CHAIN_NUDGE,
  CURVES,
  DIAL_MAX,
  DIAL_MIN,
  MAX_WALK_STEPS,
  NUDGES,
  RACE_BUDGET,
  RACE_FLOOR,
  RACE_START,
  RACE_TARGET,
  SURFACES,
  chainA,
  chainASlope,
  chainB,
  chainBSlope,
  chainReading,
  contourLevels,
  contourSegments,
  curveSlope,
  curveValue,
  flatSpots,
  nudgeRatio,
  raceReadings,
  slopeBands,
  stepDown,
  stepDownSurface,
  surfaceGradient,
  surfaceValue,
  walk,
  walkSurface,
} from "./curves";

/** A symmetric difference quotient: an independent estimate of a slope. */
const central = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);
const dialSettings = Array.from({ length: 25 }, (_, index) => DIAL_MIN + (index * (DIAL_MAX - DIAL_MIN)) / 24);

describe("one-dial curves", () => {
  it.each(CURVES)("%s: the hand-derived slope matches a numerical one everywhere on the dial", (curve) => {
    for (const w of dialSettings) {
      expect(curveSlope(curve, w)).toBeCloseTo(central((x) => curveValue(curve, x), w), 5);
    }
  });

  it("the bowl's nudge ratio is exactly the slope plus the nudge, so it settles on the slope", () => {
    for (const w of [0, 1.5, 3, 4, 5]) {
      for (const nudge of NUDGES) {
        expect(nudgeRatio("bowl", w, nudge)).toBeCloseTo(curveSlope("bowl", w) + nudge, 9);
      }
      const gaps = NUDGES.map((nudge) => Math.abs(nudgeRatio("bowl", w, nudge) - curveSlope("bowl", w)));
      expect(gaps).toEqual([...gaps].sort((a, b) => b - a));
    }
  });

  it("offers nudges from largest to smallest, with halvings at the top", () => {
    expect(NUDGES).toEqual([...NUDGES].sort((a, b) => b - a));
    expect(NUDGES[1]).toBe(NUDGES[0] / 2);
    expect(NUDGES[2]).toBe(NUDGES[1] / 2);
  });

  it("finds each curve's flat spots and names their kind", () => {
    expect(flatSpots("bowl")).toEqual([{ w: 3, value: 1, kind: "bottom" }]);
    expect(flatSpots("shelf")).toEqual([{ w: 3, value: 5, kind: "shelf" }]);
    const valleys = flatSpots("valleys");
    expect(valleys.map((spot) => spot.kind)).toEqual(["bottom", "top", "bottom"]);
    for (const spot of valleys) {
      expect(Math.abs(curveSlope("valleys", spot.w))).toBeLessThan(1e-9);
      expect(spot.value).toBeCloseTo(curveValue("valleys", spot.w), 12);
    }
    // The right-hand valley is the deeper one, and the hump stands above both.
    expect(valleys[2].value).toBeLessThan(valleys[0].value);
    expect(valleys[1].value).toBeGreaterThan(valleys[0].value);
  });

  it("the shelf's slope touches zero without changing sign", () => {
    for (const w of dialSettings) expect(curveSlope("shelf", w)).toBeGreaterThanOrEqual(0);
    expect(curveSlope("shelf", 3)).toBe(0);
    expect(curveValue("shelf", 2)).toBeLessThan(curveValue("shelf", 3));
    expect(curveValue("shelf", 4)).toBeGreaterThan(curveValue("shelf", 3));
  });

  it.each(CURVES)("%s: the bands cover the dial and agree with the sign of the slope", (curve) => {
    const bands = slopeBands(curve);
    expect(bands[0].from).toBe(DIAL_MIN);
    expect(bands[bands.length - 1].to).toBe(DIAL_MAX);
    bands.forEach((band, index) => {
      if (index > 0) expect(band.from).toBeCloseTo(bands[index - 1].to, 12);
      for (const share of [0.1, 0.5, 0.9]) {
        const slope = curveSlope(curve, band.from + (band.to - band.from) * share);
        expect(band.direction === "rising" ? slope > 0 : slope < 0).toBe(true);
      }
    });
  });

  it("names two falling and two rising stretches on the two-valley curve, and only rising ones on the shelf", () => {
    expect(slopeBands("valleys").map((band) => band.direction)).toEqual(["falling", "rising", "falling", "rising"]);
    expect(slopeBands("shelf").map((band) => band.direction)).toEqual(["rising", "rising"]);
  });
});

describe("walking downhill on one dial", () => {
  it("a step moves against the slope by step size times slope", () => {
    expect(stepDown("bowl", 5, 0.1)).toBeCloseTo(5 - 0.1 * 4, 12);
    expect(stepDown("bowl", 1, 0.1)).toBeCloseTo(1 + 0.1 * 4, 12);
    expect(stepDown("valleys", 2.6, 0.2)).toBeLessThan(2.6);
    expect(stepDown("bowl", 3, 0.7)).toBe(3);
  });

  it("stops at the dial's ends", () => {
    expect(stepDown("bowl", 5.9, 1.2)).toBe(DIAL_MIN);
    expect(stepDown("bowl", 0.1, 1.2)).toBe(DIAL_MAX);
    expect(walk("bowl", 99, 0.1, 0)).toEqual([DIAL_MAX]);
  });

  it("returns the start plus one setting per step", () => {
    expect(walk("valleys", 2.6, 0.2, 0)).toEqual([2.6]);
    expect(walk("valleys", 2.6, 0.2, MAX_WALK_STEPS)).toHaveLength(MAX_WALK_STEPS + 1);
  });

  it("on the bowl each step scales the distance from the bottom by 1 minus twice the step size", () => {
    for (const stepSize of [0.05, 0.3, 0.5, 0.9]) {
      const path = walk("bowl", 5, stepSize, 6);
      path.slice(1).forEach((w, index) => {
        expect(w - 3).toBeCloseTo((path[index] - 3) * (1 - 2 * stepSize), 9);
      });
    }
  });

  it("with a small step the meter never rises, and with too large a step it does", () => {
    for (const curve of ["bowl", "valleys"] as const) {
      const readings = walk(curve, 2.6, 0.1, 30).map((w) => curveValue(curve, w));
      readings.slice(1).forEach((reading, index) => expect(reading).toBeLessThanOrEqual(readings[index] + 1e-12));
    }
    const climbing = walk("bowl", 5, 1.1, 2).map((w) => curveValue("bowl", w));
    expect(climbing[1]).toBeGreaterThan(climbing[0]);
    expect(climbing[2]).toBeGreaterThan(climbing[1]);
  });

  it("settles in the valley on the same side of the hump as the start", () => {
    const [left, hump, right] = flatSpots("valleys");
    for (const start of [0.5, 2, hump.w - 0.2]) {
      expect(walk("valleys", start, 0.2, MAX_WALK_STEPS).pop()).toBeCloseTo(left.w, 3);
    }
    for (const start of [hump.w + 0.2, 4, 5.5]) {
      expect(walk("valleys", start, 0.2, MAX_WALK_STEPS).pop()).toBeCloseTo(right.w, 3);
    }
  });
});

describe("two-dial surfaces", () => {
  const points = [
    [0, 0],
    [1, 4],
    [3, 3],
    [4.5, 4],
    [5.5, 0.5],
    [6, 6],
  ];

  it.each(SURFACES)("%s: each entry of the gradient is the slope along one dial with the other held still", (surface) => {
    for (const [w, b] of points) {
      const gradient = surfaceGradient(surface, w, b);
      expect(gradient.x).toBeCloseTo(central((x) => surfaceValue(surface, x, b), w), 5);
      expect(gradient.y).toBeCloseTo(central((y) => surfaceValue(surface, w, y), b), 5);
    }
  });

  it("the gradient points uphill and the opposite way points downhill", () => {
    for (const surface of SURFACES) {
      for (const [w, b] of points) {
        const gradient = surfaceGradient(surface, w, b);
        const size = Math.hypot(gradient.x, gradient.y);
        if (size < 1e-9) continue;
        const here = surfaceValue(surface, w, b);
        const small = 1e-3 / size;
        expect(surfaceValue(surface, w + gradient.x * small, b + gradient.y * small)).toBeGreaterThan(here);
        expect(surfaceValue(surface, w - gradient.x * small, b - gradient.y * small)).toBeLessThan(here);
      }
    }
  });

  it("no direction climbs faster than the gradient's own", () => {
    const [w, b] = [1, 4];
    const gradient = surfaceGradient("valley", w, b);
    const size = Math.hypot(gradient.x, gradient.y);
    const rise = (dx: number, dy: number) => surfaceValue("valley", w + dx * 1e-4, b + dy * 1e-4) - surfaceValue("valley", w, b);
    const best = rise(gradient.x / size, gradient.y / size);
    for (let turn = 0; turn < 36; turn += 1) {
      const angle = (turn * Math.PI) / 18;
      expect(rise(Math.cos(angle), Math.sin(angle))).toBeLessThanOrEqual(best + 1e-9);
    }
  });

  it("the gradient is at a right angle to the contour line through the point", () => {
    for (const surface of SURFACES) {
      const [w, b] = [4.5, 4];
      const gradient = surfaceGradient(surface, w, b);
      const size = Math.hypot(gradient.x, gradient.y);
      // A small step along the contour's direction leaves the meter unchanged to first order.
      const along = { x: -gradient.y / size, y: gradient.x / size };
      const change = surfaceValue(surface, w + along.x * 1e-4, b + along.y * 1e-4) - surfaceValue(surface, w, b);
      expect(Math.abs(change)).toBeLessThan(1e-7);
    }
  });

  it("is flat at the centre of every surface, where only the saddle is not a low point", () => {
    for (const surface of SURFACES) {
      const gradient = surfaceGradient(surface, 3, 3);
      expect(Math.hypot(gradient.x, gradient.y)).toBe(0);
    }
    expect(surfaceValue("bowl", 3, 3)).toBe(1);
    expect(surfaceValue("valley", 3, 3)).toBe(RACE_FLOOR);
    expect(surfaceValue("saddle", 5, 3)).toBeGreaterThan(surfaceValue("saddle", 3, 3));
    expect(surfaceValue("saddle", 3, 5)).toBeLessThan(surfaceValue("saddle", 3, 3));
  });

  it.each(SURFACES)("%s: contour segments lie on their level and stay on the map", (surface) => {
    for (const level of contourLevels(surface)) {
      const segments = contourSegments((x, y) => surfaceValue(surface, x, y), level);
      expect(segments.length, `level ${level}`).toBeGreaterThan(0);
      for (const [x1, y1, x2, y2] of segments) {
        for (const value of [x1, y1, x2, y2]) {
          expect(value).toBeGreaterThanOrEqual(DIAL_MIN - 1e-9);
          expect(value).toBeLessThanOrEqual(DIAL_MAX + 1e-9);
        }
        expect(surfaceValue(surface, x1, y1)).toBeCloseTo(level, 1);
        expect(surfaceValue(surface, x2, y2)).toBeCloseTo(level, 1);
      }
    }
  });

  it("draws nothing for a level the surface never reaches", () => {
    expect(contourSegments((x, y) => surfaceValue("bowl", x, y), 0.5)).toEqual([]);
    expect(contourSegments((x, y) => surfaceValue("bowl", x, y), 50)).toEqual([]);
  });

  it("a surface step moves both dials against their slopes and stops at the dial's ends", () => {
    const next = stepDownSurface("valley", { x: 1, y: 4 }, 0.1);
    expect(next.x).toBeCloseTo(1 + 0.1 * 0.5, 12);
    expect(next.y).toBeCloseTo(4 - 0.1 * 2, 12);
    expect(stepDownSurface("valley", { x: 3, y: 6 }, 1.1)).toEqual({ x: 3, y: DIAL_MIN });
    expect(walkSurface("valley", RACE_START, 0.5, RACE_BUDGET)).toHaveLength(RACE_BUDGET + 1);
  });
});

describe("the challenge", () => {
  const stepSizes = Array.from({ length: 22 }, (_, index) => Number(((index + 1) * 0.05).toFixed(2)));
  const wins = stepSizes.filter((stepSize) => raceReadings(stepSize, RACE_BUDGET)[RACE_BUDGET] < RACE_TARGET);

  it("can be won, but only inside one unbroken range of step sizes", () => {
    expect(wins.length).toBeGreaterThan(0);
    expect(wins).toEqual(stepSizes.slice(stepSizes.indexOf(wins[0]), stepSizes.indexOf(wins[wins.length - 1]) + 1));
    // Both ends of the slider lose: too small crawls, too large bounces.
    expect(wins[0]).toBeGreaterThan(stepSizes[0] * 3);
    expect(wins[wins.length - 1]).toBeLessThan(1);
  });

  it("is lost at the slider's default, so the first run teaches something", () => {
    expect(raceReadings(0.1, RACE_BUDGET)[RACE_BUDGET]).toBeGreaterThan(RACE_TARGET);
  });

  it("too small a step is still far along the floor, and too large a step is still high on a wall", () => {
    const slow = walkSurface("valley", RACE_START, 0.1, RACE_BUDGET).pop()!;
    expect(Math.abs(slow.y - 3)).toBeLessThan(0.15);
    expect(Math.abs(slow.x - 3)).toBeGreaterThan(1.5);
    const bouncing = walkSurface("valley", RACE_START, 1, RACE_BUDGET);
    for (const point of bouncing) expect(Math.abs(point.y - 3)).toBeCloseTo(1.5, 9);
    expect(Math.sign(bouncing[1].y - 3)).toBe(-Math.sign(bouncing[0].y - 3));
  });

  it("never reads below the valley floor", () => {
    for (const stepSize of stepSizes) {
      for (const reading of raceReadings(stepSize, RACE_BUDGET)) expect(reading).toBeGreaterThanOrEqual(RACE_FLOOR);
    }
  });
});

describe("the chain of two machines", () => {
  it("each machine's hand-derived slope matches a numerical one", () => {
    for (const w of dialSettings) {
      expect(chainASlope(w)).toBeCloseTo(central(chainA, w), 5);
      expect(chainBSlope(chainA(w))).toBeCloseTo(central(chainB, chainA(w)), 5);
    }
  });

  it("the overall slope is the product of the two, and matches the slope of the whole chain", () => {
    for (const w of dialSettings) {
      const reading = chainReading(w);
      expect(reading.overall).toBeCloseTo(reading.slopeA * reading.slopeB, 12);
      expect(reading.overall).toBeCloseTo(central((x) => chainB(chainA(x)), w), 5);
      expect(reading.predictedMeter).toBeCloseTo(reading.overall * CHAIN_NUDGE, 12);
    }
  });

  it("the slopes predict a small nudge better than a large one", () => {
    const gap = (nudge: number) => {
      const reading = chainReading(3, nudge);
      return Math.abs(reading.actualMeter - reading.predictedMeter) / nudge;
    };
    expect(gap(0.01)).toBeLessThan(gap(0.1));
    expect(gap(0.1)).toBeLessThan(gap(1));
  });

  it("machine B goes flat at both ends while machine A keeps getting steeper", () => {
    expect(chainASlope(6)).toBeGreaterThan(chainASlope(3));
    expect(chainBSlope(chainA(6))).toBeLessThan(0.01);
    expect(chainBSlope(chainA(0))).toBeLessThan(0.05);
    expect(chainReading(6).overall).toBeLessThan(0.01);
    // Flat and low at one end, flat and high at the other.
    expect(chainReading(0).meter).toBeLessThan(0.1);
    expect(chainReading(6).meter).toBeGreaterThan(9.9);
  });

  it("stays finite far outside the dial", () => {
    for (const u of [-1e9, 1e9]) {
      expect(Number.isFinite(chainB(u))).toBe(true);
      expect(Number.isFinite(chainBSlope(u))).toBe(true);
    }
  });
});
