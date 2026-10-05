/** Closed-form math for one two-input neuron: z = w₁x₁ + w₂x₂ + b, a = f(z). */

export type ActivationKind = "sigmoid" | "relu" | "step";
export type CornerTask = "off" | "and" | "or" | "xor";

export const ACTIVATIONS: readonly ActivationKind[] = ["sigmoid", "relu", "step"];
export const TASKS: readonly CornerTask[] = ["off", "and", "or", "xor"];

export const INPUT_RANGE = [0, 1] as const;
export const WEIGHT_RANGE = [-2, 2] as const;
export const BIAS_RANGE = [-2, 2] as const;

export interface NeuronParams {
  w1: number;
  w2: number;
  bias: number;
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export function weightedSum(x1: number, x2: number, { w1, w2, bias }: NeuronParams) {
  return x1 * w1 + x2 * w2 + bias;
}

/** Step uses the convention f(0) = 0: the neuron fires only when z is strictly positive. */
export function activate(z: number, kind: ActivationKind) {
  if (kind === "relu") return Math.max(0, z);
  if (kind === "step") return z > 0 ? 1 : 0;
  return sigmoid(z);
}

/**
 * df/dz. Sigmoid: σ(z)(1 − σ(z)). ReLU: 1 above zero, 0 below. Step: 0 away from the
 * jump; at z = 0 neither function has a derivative, and this returns NaN there.
 */
export function slope(z: number, kind: ActivationKind) {
  if (kind === "sigmoid") {
    const s = sigmoid(z);
    return s * (1 - s);
  }
  if (z === 0) return Number.NaN;
  if (kind === "relu") return z > 0 ? 1 : 0;
  return 0;
}

export const weightNorm = ({ w1, w2 }: NeuronParams) => Math.hypot(w1, w2);

/** Signed Euclidean distance from (x1, x2) to the line z = 0, positive on the side w points to. */
export function signedDistance(x1: number, x2: number, params: NeuronParams) {
  const norm = weightNorm(params);
  return norm > 1e-12 ? weightedSum(x1, x2, params) / norm : Number.NaN;
}

/** Points where the level set z = level crosses the square [lo, hi]², or null if it misses. */
export function levelChord(params: NeuronParams, level = 0, lo = 0, hi = 1) {
  const { w1, w2 } = params;
  const bias = params.bias - level;
  const hits: Array<[number, number]> = [];
  const inside = (value: number) => value >= lo - 1e-9 && value <= hi + 1e-9;
  const push = (x1: number, x2: number) => {
    if (inside(x1) && inside(x2) && !hits.some(([a, b]) => Math.hypot(a - x1, b - x2) < 1e-6)) {
      hits.push([x1, x2]);
    }
  };
  if (Math.abs(w2) > 1e-9) {
    push(lo, -(w1 * lo + bias) / w2);
    push(hi, -(w1 * hi + bias) / w2);
  }
  if (Math.abs(w1) > 1e-9) {
    push(-(w2 * lo + bias) / w1, lo);
    push(-(w2 * hi + bias) / w1, hi);
  }
  return hits.length >= 2 ? ([hits[0], hits[1]] as const) : null;
}

/** Foot of the perpendicular from a point to the line z = 0. */
export function footOnBoundary(x1: number, x2: number, params: NeuronParams) {
  const norm = weightNorm(params);
  if (norm < 1e-12) return null;
  const distance = weightedSum(x1, x2, params) / norm;
  return [x1 - (distance * params.w1) / norm, x2 - (distance * params.w2) / norm] as const;
}

export interface Corner {
  x1: 0 | 1;
  x2: 0 | 1;
  target: 0 | 1;
}

export function cornersFor(task: CornerTask): Corner[] {
  if (task === "off") return [];
  const table: Array<[0 | 1, 0 | 1]> = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  return table.map(([x1, x2]) => ({
    x1,
    x2,
    target: (task === "and" ? x1 & x2 : task === "or" ? x1 | x2 : x1 ^ x2) as 0 | 1,
  }));
}

/** The neuron predicts 1 where z > 0, the same rule as the Step activation. */
export function cornersCorrect(task: CornerTask, params: NeuronParams) {
  return cornersFor(task).filter((corner) => (weightedSum(corner.x1, corner.x2, params) > 0 ? 1 : 0) === corner.target)
    .length;
}
