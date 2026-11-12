/**
 * A ravine in two weights: the picture behind the race. The loss is a plain quadratic valley, a
 * hundred times steeper across than along, and three optimizers walk it from the same start with the
 * update rules and the default rates the race uses (momentum β 0.9; Adam β 0.9 and 0.999, ε 1e-8,
 * bias correction on, no decay). The ravine's size is chosen so those default rates are sensible on it.
 * Every path and every count on the card is computed here. Muon is left out: it reshapes a matrix
 * of weights, and two weights are not a matrix it can do anything with.
 */

/** Curvature across the ravine and along its floor. Their ratio is the ravine's condition number. */
export const RAVINE_STEEP = 1.8;
export const RAVINE_GENTLE = 0.018;
export const RAVINE_STEPS = 120;
/** The square of weights the card draws, and the loss its surface is cut off at. */
export const RAVINE_RANGE = [-1, 1] as const;
export const RAVINE_TOP = 0.9;
/** A walk has settled once its loss stays under this for the rest of the run. */
export const RAVINE_SETTLED = 0.001;
/** A walk whose loss passes this has flown out of the ravine and stops. */
const RAVINE_LOST = 1e4;

/** Where every walk starts when the ravine runs along weight 1: far along the floor, part-way up one wall. */
const START_ALONG = -0.8;
const START_ACROSS = 0.4;

export type RavineOptimizer = "sgd" | "momentum" | "adam";
export type RavineTurn = "axis" | "diagonal";
export type RavineRates = "half" | "default" | "double";

export const RAVINE_OPTIMIZERS: ReadonlyArray<{ id: RavineOptimizer; label: string; rate: number; tone: string }> = [
  { id: "sgd", label: "SGD", rate: 1, tone: "--forward" },
  { id: "momentum", label: "Momentum", rate: 0.1, tone: "--attention" },
  { id: "adam", label: "Adam", rate: 0.03, tone: "--gradient" },
];

export const RATE_FACTOR: Record<RavineRates, number> = { half: 0.5, default: 1, double: 2 };
const TURN_ANGLE: Record<RavineTurn, number> = { axis: 0, diagonal: Math.PI / 4 };

/** Plain SGD on this ravine diverges once rate × steepness passes 2. */
export const SGD_LIMIT = 2 / RAVINE_STEEP;

export interface RavinePoint {
  x: number;
  y: number;
  loss: number;
}

/** The loss and its two slopes at weights (x, y), for a ravine turned by `angle`. */
export function ravine(turn: RavineTurn, x: number, y: number) {
  const angle = TURN_ANGLE[turn];
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  // u runs along the floor, v across it.
  const u = c * x + s * y;
  const v = -s * x + c * y;
  const gu = RAVINE_GENTLE * u;
  const gv = RAVINE_STEEP * v;
  return {
    loss: 0.5 * (RAVINE_GENTLE * u * u + RAVINE_STEEP * v * v),
    gx: c * gu - s * gv,
    gy: s * gu + c * gv,
  };
}

export const ravineLoss = (turn: RavineTurn, x: number, y: number) => ravine(turn, x, y).loss;

/** The start turns with the ravine, so it is always the same distance along the floor and up the wall. */
export function ravineStart(turn: RavineTurn): { x: number; y: number } {
  const angle = TURN_ANGLE[turn];
  return {
    x: Math.cos(angle) * START_ALONG - Math.sin(angle) * START_ACROSS,
    y: Math.sin(angle) * START_ALONG + Math.cos(angle) * START_ACROSS,
  };
}

/** The two ends of the ravine floor inside the drawn square. */
export function ravineFloor(turn: RavineTurn): [{ x: number; y: number }, { x: number; y: number }] {
  const angle = TURN_ANGLE[turn];
  const reach = RAVINE_RANGE[1] / Math.max(Math.abs(Math.cos(angle)), Math.abs(Math.sin(angle))) - 0.0005;
  return [
    { x: -Math.cos(angle) * reach, y: -Math.sin(angle) * reach },
    { x: Math.cos(angle) * reach, y: Math.sin(angle) * reach },
  ];
}

/** The points one optimizer visits, the start included. A walk that flies out stops where it left. */
export function ravineWalk(optimizer: RavineOptimizer, turn: RavineTurn, rate: number, steps = RAVINE_STEPS): RavinePoint[] {
  let { x, y } = ravineStart(turn);
  let vx = 0;
  let vy = 0;
  let mx = 0;
  let my = 0;
  let sx = 0;
  let sy = 0;
  const points: RavinePoint[] = [];
  for (let step = 0; step <= steps; step += 1) {
    const here = ravine(turn, x, y);
    points.push({ x, y, loss: here.loss });
    if (!Number.isFinite(here.loss) || here.loss > RAVINE_LOST) break;
    if (optimizer === "sgd") {
      x -= rate * here.gx;
      y -= rate * here.gy;
    } else if (optimizer === "momentum") {
      vx = 0.9 * vx + here.gx;
      vy = 0.9 * vy + here.gy;
      x -= rate * vx;
      y -= rate * vy;
    } else {
      const count = step + 1;
      mx = 0.9 * mx + 0.1 * here.gx;
      my = 0.9 * my + 0.1 * here.gy;
      sx = 0.999 * sx + 0.001 * here.gx * here.gx;
      sy = 0.999 * sy + 0.001 * here.gy * here.gy;
      const first = 1 - 0.9 ** count;
      const second = 1 - 0.999 ** count;
      x -= (rate * (mx / first)) / (Math.sqrt(sx / second) + 1e-8);
      y -= (rate * (my / first)) / (Math.sqrt(sy / second) + 1e-8);
    }
  }
  return points;
}

/** True when the walk left the ravine before the last step. */
export const ravineLost = (walk: ReadonlyArray<RavinePoint>, steps = RAVINE_STEPS) => walk.length < steps + 1;

/** The first step after which the loss stays under `RAVINE_SETTLED` to the end of the run, or null. */
export function ravineSettled(walk: ReadonlyArray<RavinePoint>): number | null {
  if (ravineLost(walk)) return null;
  let settled: number | null = null;
  walk.forEach((point, step) => {
    if (point.loss <= RAVINE_SETTLED) settled ??= step;
    else settled = null;
  });
  return settled;
}

/** Where a walk is at `step`: its last point if it has already flown out. */
export const ravineAt = (walk: ReadonlyArray<RavinePoint>, step: number) => walk[Math.min(step, walk.length - 1)];

export const ravineRate = (optimizer: RavineOptimizer, rates: RavineRates) =>
  Number(((RAVINE_OPTIMIZERS.find((entry) => entry.id === optimizer)?.rate ?? 0) * RATE_FACTOR[rates]).toPrecision(6));
