/**
 * Parametric scaling-law arithmetic for the scaling-laws lab.
 *
 * Loss model (Hoffmann et al. 2022, "approach 3"):
 *   L(N, D) = E + A / N^alpha + B / D^beta
 * with N the parameter count and D the training-token count, both as raw counts,
 * and L in nats per token on the fit's own data and tokenizer.
 *
 * Training compute is the dense-transformer rule of thumb C = 6 N D FLOPs.
 * Minimising L along a fixed C gives a closed form:
 *   N*(C) = G (C/6)^a,  D*(C) = (C/6) / N*(C),
 *   G = (alpha A / (beta B))^(1 / (alpha + beta)),  a = beta / (alpha + beta).
 */

export type FitId = "hoffmann" | "epoch";

export interface ScalingFit {
  id: FitId;
  label: string;
  source: string;
  E: number;
  A: number;
  B: number;
  alpha: number;
  beta: number;
}

export const FITS: Record<FitId, ScalingFit> = {
  hoffmann: {
    id: "hoffmann",
    label: "Hoffmann 2022",
    source: "Hoffmann et al. 2022, eq. 10 as printed",
    E: 1.69,
    A: 406.4,
    B: 410.7,
    alpha: 0.34,
    beta: 0.28,
  },
  epoch: {
    id: "epoch",
    label: "Epoch 2024 refit",
    source: "Besiroglu et al. 2024, Table 1",
    E: 1.8172,
    A: 482.01,
    B: 2085.43,
    alpha: 0.3478,
    beta: 0.3658,
  },
};

export const FIT_IDS = ["hoffmann", "epoch"] as const;

/** Chinchilla's own training budget: 70B parameters on 1.4T tokens. */
export const CHINCHILLA = { parameters: 70e9, tokens: 1.4e12 };

export const RULE_OF_THUMB_RATIO = 20;

/** Slider ranges in billions, matching the stored `parameters` and `data` state keys. */
export const PARAMETER_RANGE = { min: 0.1, max: 1000 };
export const TOKEN_RANGE = { min: 1, max: 30000 };

export function predictedLoss(fit: ScalingFit, parameters: number, tokens: number) {
  return fit.E + fit.A / Math.pow(parameters, fit.alpha) + fit.B / Math.pow(tokens, fit.beta);
}

export function lossTerms(fit: ScalingFit, parameters: number, tokens: number) {
  const size = fit.A / Math.pow(parameters, fit.alpha);
  const data = fit.B / Math.pow(tokens, fit.beta);
  return { floor: fit.E, size, data, total: fit.E + size + data };
}

export function trainingFlops(parameters: number, tokens: number) {
  return 6 * parameters * tokens;
}

/** Exponent a in N*(C) ∝ C^a. The token exponent is 1 − a. */
export function parameterExponent(fit: ScalingFit) {
  return fit.beta / (fit.alpha + fit.beta);
}

/** Reducible loss on the frontier falls as C^(−alpha beta / (alpha + beta)). */
export function frontierLossExponent(fit: ScalingFit) {
  return (fit.alpha * fit.beta) / (fit.alpha + fit.beta);
}

export function optimalAllocation(fit: ScalingFit, compute: number) {
  const k = compute / 6;
  const g = Math.pow((fit.alpha * fit.A) / (fit.beta * fit.B), 1 / (fit.alpha + fit.beta));
  const parameters = g * Math.pow(k, parameterExponent(fit));
  const tokens = k / parameters;
  return {
    parameters,
    tokens,
    ratio: tokens / parameters,
    loss: predictedLoss(fit, parameters, tokens),
  };
}

/** The allocation that spends the budget at a fixed tokens-per-parameter ratio. */
export function ratioAllocation(fit: ScalingFit, compute: number, ratio = RULE_OF_THUMB_RATIO) {
  const parameters = Math.sqrt(compute / (6 * ratio));
  const tokens = ratio * parameters;
  return { parameters, tokens, ratio, loss: predictedLoss(fit, parameters, tokens) };
}

/** Loss along one isoFLOP curve, sampled evenly in log10(N). */
export function isoFlopCurve(fit: ScalingFit, compute: number, logMin: number, logMax: number, samples = 96) {
  const points: { parameters: number; tokens: number; loss: number }[] = [];
  for (let index = 0; index <= samples; index += 1) {
    const parameters = Math.pow(10, logMin + ((logMax - logMin) * index) / samples);
    const tokens = compute / (6 * parameters);
    points.push({ parameters, tokens, loss: predictedLoss(fit, parameters, tokens) });
  }
  return points;
}

/** Brute-force minimum along an isoFLOP curve, used to check the closed form. */
export function numericOptimum(fit: ScalingFit, compute: number) {
  let low = 4;
  let high = 14;
  const at = (logN: number) => {
    const parameters = Math.pow(10, logN);
    return predictedLoss(fit, parameters, compute / (6 * parameters));
  };
  for (let iteration = 0; iteration < 200; iteration += 1) {
    const left = low + (high - low) / 3;
    const right = high - (high - low) / 3;
    if (at(left) < at(right)) high = right;
    else low = left;
  }
  const parameters = Math.pow(10, (low + high) / 2);
  return { parameters, tokens: compute / (6 * parameters) };
}

/** Human-readable counts: 3.2B, 28B, 1.4T, 450M. */
export function formatCount(value: number) {
  if (value >= 1e12) return `${trim(value / 1e12)}T`;
  if (value >= 1e9) return `${trim(value / 1e9)}B`;
  if (value >= 1e6) return `${trim(value / 1e6)}M`;
  return `${Math.round(value)}`;
}

function trim(value: number) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return String(Number(value.toFixed(1)));
  return String(Number(value.toFixed(2)));
}

const SUPERSCRIPT: Record<string, string> = {
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
  "-": "⁻",
};

export function superscript(exponent: number) {
  return String(exponent)
    .split("")
    .map((character) => SUPERSCRIPT[character] ?? character)
    .join("");
}

/** 5.4 × 10²⁰ */
export function formatFlops(value: number) {
  const exponent = Math.floor(Math.log10(value));
  const mantissa = value / Math.pow(10, exponent);
  const rounded = Number(mantissa.toFixed(1));
  if (rounded >= 10) return `1.0 × 10${superscript(exponent + 1)}`;
  return `${rounded.toFixed(1)} × 10${superscript(exponent)}`;
}

/* -------------------------------------------------------------------------- */
/* From a transformer's dimensions to N                                        */
/* -------------------------------------------------------------------------- */

/** Layer count and width the dimension calculator offers. */
export const LAYER_RANGE = { min: 1, max: 128 };
export const WIDTH_RANGE = { min: 64, max: 16384, step: 64 };

/**
 * Weights in one classic transformer block of width d, biases and norm scales left out:
 * attention has four d x d matrices (queries, keys, values, output) and the MLP has d x 4d and 4d x d.
 */
export const blockWeights = (width: number) => 4 * width * width + 2 * width * (4 * width);

/** N = 12 L d^2: the block count times the weights in one block, embeddings excluded. */
export const denseParameters = (layers: number, width: number) => layers * blockWeights(width);

/** The toy block of The transformer block lab: width 8 with a 4x MLP, plus its biases and norm scales. */
export const TOY_BLOCK = { width: 8, hidden: 32 };
export const toyBlockExtras = ({ width, hidden } = TOY_BLOCK) =>
  hidden /* first MLP bias */ + width /* second MLP bias */ + 4 * width /* two norms, scale and shift */;

/** Token embedding table, one row per vocabulary entry. */
export const embeddingWeights = (vocabulary: number, width: number) => vocabulary * width;

/** Snaps a requested width onto the slider's grid inside its range. */
export const clampWidth = (value: number) =>
  Number.isFinite(value)
    ? Math.min(WIDTH_RANGE.max, Math.max(WIDTH_RANGE.min, Math.round(value / WIDTH_RANGE.step) * WIDTH_RANGE.step))
    : 4096;
