/**
 * The closed forms behind the Slopes & gradients lab. Every readout and every mark on the page is
 * computed here from one of these small formulas; nothing is measured from a trained model.
 */

/** Every dial in the lab turns between these two stops. */
export const DIAL_MIN = 0;
export const DIAL_MAX = 6;

export const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const clampDial = (value: number) => clamp(value, DIAL_MIN, DIAL_MAX);

/* ------------------------------------------------------------------ one dial */

export type Curve = "bowl" | "valleys" | "shelf";
export const CURVES: readonly Curve[] = ["bowl", "valleys", "shelf"];

/** Two valleys: a shallow bowl plus a ripple, tilted so the right-hand valley is the deeper one. */
const VALLEYS = { bowl: 0.2, ripple: 0.9, rate: 1.8, tilt: -0.3, lift: 2.2 } as const;

/** The meter reading L(w) for a dial setting w. */
export function curveValue(curve: Curve, w: number): number {
  const u = w - 3;
  if (curve === "bowl") return u * u + 1;
  if (curve === "shelf") return 5 + 0.15 * u * u * u;
  return VALLEYS.bowl * u * u + VALLEYS.ripple * Math.cos(VALLEYS.rate * u) + VALLEYS.tilt * u + VALLEYS.lift;
}

/** The exact slope of L at w: the derivative of the formula above, worked out by hand. */
export function curveSlope(curve: Curve, w: number): number {
  const u = w - 3;
  if (curve === "bowl") return 2 * u;
  if (curve === "shelf") return 0.45 * u * u;
  return 2 * VALLEYS.bowl * u - VALLEYS.ripple * VALLEYS.rate * Math.sin(VALLEYS.rate * u) + VALLEYS.tilt;
}

/** Change in the meter divided by change in the dial, across a nudge of the given size. */
export function nudgeRatio(curve: Curve, w: number, nudge: number): number {
  return (curveValue(curve, w + nudge) - curveValue(curve, w)) / nudge;
}

/** The nudge sizes the first card offers, largest first. Each is small enough to halve by eye. */
export const NUDGES: readonly number[] = [1, 0.5, 0.25, 0.1, 0.05, 0.01];

export type FlatKind = "bottom" | "top" | "shelf";
export interface FlatSpot {
  w: number;
  value: number;
  kind: FlatKind;
}

/**
 * Where the slope is zero between the dial stops, and what kind of flat spot each is. A sign change
 * from falling to rising is a bottom, rising to falling is a top, and a slope that touches zero
 * without changing sign is a shelf.
 */
export function flatSpots(curve: Curve): FlatSpot[] {
  if (curve === "bowl") return [{ w: 3, value: curveValue("bowl", 3), kind: "bottom" }];
  if (curve === "shelf") return [{ w: 3, value: curveValue("shelf", 3), kind: "shelf" }];
  const found: FlatSpot[] = [];
  const pieces = 600;
  for (let index = 0; index < pieces; index += 1) {
    let low = DIAL_MIN + ((DIAL_MAX - DIAL_MIN) * index) / pieces;
    let high = DIAL_MIN + ((DIAL_MAX - DIAL_MIN) * (index + 1)) / pieces;
    const before = Math.sign(curveSlope(curve, low));
    const after = Math.sign(curveSlope(curve, high));
    if (before === after || before === 0) continue;
    for (let round = 0; round < 60; round += 1) {
      const middle = (low + high) / 2;
      if (Math.sign(curveSlope(curve, middle)) === before) low = middle;
      else high = middle;
    }
    const w = (low + high) / 2;
    found.push({ w, value: curveValue(curve, w), kind: before < 0 ? "bottom" : "top" });
  }
  return found;
}

export interface Band {
  from: number;
  to: number;
  direction: "rising" | "falling";
}

/** The stretches of the dial where the meter rises and where it falls, split at the flat spots. */
export function slopeBands(curve: Curve): Band[] {
  const cuts = [DIAL_MIN, ...flatSpots(curve).map((spot) => spot.w), DIAL_MAX];
  const bands: Band[] = [];
  for (let index = 0; index < cuts.length - 1; index += 1) {
    const from = cuts[index];
    const to = cuts[index + 1];
    if (to - from < 1e-6) continue;
    bands.push({ from, to, direction: curveSlope(curve, (from + to) / 2) >= 0 ? "rising" : "falling" });
  }
  return bands;
}

/** One downhill step: move the dial against the slope, by step size times slope, and stop at the dial's ends. */
export function stepDown(curve: Curve, w: number, stepSize: number): number {
  return clampDial(w - stepSize * curveSlope(curve, w));
}

/** The dial settings visited by `steps` downhill steps from `start`, the start included. */
export function walk(curve: Curve, start: number, stepSize: number, steps: number): number[] {
  const path = [clampDial(start)];
  for (let index = 0; index < steps; index += 1) path.push(stepDown(curve, path[path.length - 1], stepSize));
  return path;
}

/** The longest walk the lab will draw. */
export const MAX_WALK_STEPS = 40;

/* ------------------------------------------------------------------ two dials */

export type Surface = "bowl" | "valley" | "saddle";
export const SURFACES: readonly Surface[] = ["bowl", "valley", "saddle"];

export interface Point {
  x: number;
  y: number;
}

/** How fast the long valley curves along its floor (dial w) and across its walls (dial b). */
export const VALLEY_FLOOR = 0.25;
export const VALLEY_WALL = 2;

/** The meter reading L(w, b) for two dials. The code calls them w1 and w2. */
export function surfaceValue(surface: Surface, w1: number, w2: number): number {
  const a = w1 - 3;
  const b = w2 - 3;
  if (surface === "bowl") return 1 + 0.5 * (a * a + b * b);
  if (surface === "valley") return 1 + 0.5 * (VALLEY_FLOOR * a * a + VALLEY_WALL * b * b);
  return 5 + 0.5 * (a * a - b * b);
}

/** The gradient: the slope along dial w with b held still, and the slope along dial b with w held still. */
export function surfaceGradient(surface: Surface, w1: number, w2: number): Point {
  const a = w1 - 3;
  const b = w2 - 3;
  if (surface === "bowl") return { x: a, y: b };
  if (surface === "valley") return { x: VALLEY_FLOOR * a, y: VALLEY_WALL * b };
  return { x: a, y: -b };
}

/** The meter levels each map draws a contour line for. */
export function contourLevels(surface: Surface): number[] {
  if (surface === "bowl") return [1.5, 2.5, 4, 6, 8];
  if (surface === "valley") return [1.25, 2, 3, 5, 7.5, 10];
  return [2, 3, 4, 5, 6, 7, 8];
}

export type Segment = [number, number, number, number];

/**
 * Marching squares: the line segments along which `value(x, y)` equals `level`, over the dial square.
 * Each cell's crossing points are placed by linear interpolation along its edges.
 */
export function contourSegments(value: (x: number, y: number) => number, level: number, cells = 48): Segment[] {
  const size = (DIAL_MAX - DIAL_MIN) / cells;
  const grid: number[][] = [];
  for (let row = 0; row <= cells; row += 1) {
    const line: number[] = [];
    for (let column = 0; column <= cells; column += 1) {
      line.push(value(DIAL_MIN + column * size, DIAL_MIN + row * size) - level);
    }
    grid.push(line);
  }
  const segments: Segment[] = [];
  const cross = (x1: number, y1: number, v1: number, x2: number, y2: number, v2: number): [number, number] => {
    const share = v1 / (v1 - v2);
    return [x1 + (x2 - x1) * share, y1 + (y2 - y1) * share];
  };
  for (let row = 0; row < cells; row += 1) {
    for (let column = 0; column < cells; column += 1) {
      const x0 = DIAL_MIN + column * size;
      const y0 = DIAL_MIN + row * size;
      const x1 = x0 + size;
      const y1 = y0 + size;
      const corners = [grid[row][column], grid[row][column + 1], grid[row + 1][column + 1], grid[row + 1][column]];
      const edges: [number, number][] = [];
      if (corners[0] > 0 !== corners[1] > 0) edges.push(cross(x0, y0, corners[0], x1, y0, corners[1]));
      if (corners[1] > 0 !== corners[2] > 0) edges.push(cross(x1, y0, corners[1], x1, y1, corners[2]));
      if (corners[2] > 0 !== corners[3] > 0) edges.push(cross(x1, y1, corners[2], x0, y1, corners[3]));
      if (corners[3] > 0 !== corners[0] > 0) edges.push(cross(x0, y1, corners[3], x0, y0, corners[0]));
      if (edges.length === 2) {
        segments.push([edges[0][0], edges[0][1], edges[1][0], edges[1][1]]);
      } else if (edges.length === 4) {
        // A saddle cell: pair the crossings so the two lines do not cross inside the cell.
        const centre = (corners[0] + corners[1] + corners[2] + corners[3]) / 4;
        const joinFirst = centre > 0 === corners[0] > 0;
        const pairs = joinFirst ? [[0, 1], [2, 3]] : [[0, 3], [1, 2]];
        for (const [from, to] of pairs) segments.push([edges[from][0], edges[from][1], edges[to][0], edges[to][1]]);
      }
    }
  }
  return segments;
}

/** One downhill step on a surface: both dials move against their own slope, by the same step size. */
export function stepDownSurface(surface: Surface, point: Point, stepSize: number): Point {
  const gradient = surfaceGradient(surface, point.x, point.y);
  return { x: clampDial(point.x - stepSize * gradient.x), y: clampDial(point.y - stepSize * gradient.y) };
}

/** The points visited by `steps` downhill steps from `start`, the start included. */
export function walkSurface(surface: Surface, start: Point, stepSize: number, steps: number): Point[] {
  const path = [{ x: clampDial(start.x), y: clampDial(start.y) }];
  for (let index = 0; index < steps; index += 1) path.push(stepDownSurface(surface, path[path.length - 1], stepSize));
  return path;
}

/** The challenge: from this start on the long valley, get the meter under the target within the step budget. */
export const RACE_START: Point = { x: 0.5, y: 4.5 };
export const RACE_BUDGET = 12;
export const RACE_TARGET = 1.05;
export const RACE_FLOOR = 1;

/** The meter after each step of a challenge attempt, the start included. */
export function raceReadings(stepSize: number, steps: number): number[] {
  return walkSurface("valley", RACE_START, stepSize, steps).map((point) => surfaceValue("valley", point.x, point.y));
}

/* ------------------------------------------------------------------ a chain of two machines */

/** Machine A turns the dial w into the middle reading u. It gets steeper as the dial goes up. */
export const chainA = (w: number) => ((w + 3) * (w + 3)) / 4;
export const chainASlope = (w: number) => (w + 3) / 2;

const squash = (z: number) => 1 / (1 + Math.exp(-clamp(z, -40, 40)));

/** Machine B turns the middle reading u into the meter L: an S-shaped curve, flat at both ends. */
export const chainB = (u: number) => 10 * squash(0.8 * (u - 9));
export const chainBSlope = (u: number) => {
  const share = squash(0.8 * (u - 9));
  return 8 * share * (1 - share);
};

/** The nudge the chain card gives the dial. */
export const CHAIN_NUDGE = 0.1;

export interface ChainReading {
  u: number;
  meter: number;
  slopeA: number;
  slopeB: number;
  /** slope A × slope B: how fast the meter moves per unit of dial, right here. */
  overall: number;
  /** What the slopes predict for the nudge: nudge × slope A, then that × slope B. */
  predictedU: number;
  predictedMeter: number;
  /** What the two formulas actually give for the same nudge. */
  actualU: number;
  actualMeter: number;
}

export function chainReading(w: number, nudge = CHAIN_NUDGE): ChainReading {
  const u = chainA(w);
  const slopeA = chainASlope(w);
  const slopeB = chainBSlope(u);
  const nudgedU = chainA(w + nudge);
  return {
    u,
    meter: chainB(u),
    slopeA,
    slopeB,
    overall: slopeA * slopeB,
    predictedU: slopeA * nudge,
    predictedMeter: slopeA * slopeB * nudge,
    actualU: nudgedU - u,
    actualMeter: chainB(nudgedU) - chainB(u),
  };
}
