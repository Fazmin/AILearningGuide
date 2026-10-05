/**
 * Diffusion math for the shipped MNIST denoiser.
 *
 * The training script (models/train_mnist_models.py) uses a 100-step cosine
 * schedule (Nichol & Dhariwal 2021) whose parameters are recorded in
 * mnist-models.metadata.json, trains an MLP to predict the added noise epsilon
 * from (x_t, t), and scales pixels to [-1, 1]. The schedule itself ships as
 * alpha-bar in mnist-diffusion-schedule.json and is the only thing this file
 * reads, apart from STEPS, which the denoiser's time embedding also assumes.
 * The script ships no sampler, so the reverse step here is the standard DDPM
 * ancestral update (Ho et al. 2020, Algorithm 2) with sigma_t^2 = beta_t,
 * written for 0-indexed steps. models/train_mnist_models.py repeats the same
 * update to judge samples before it publishes anything.
 */

export const PIXELS = 28 * 28;
export const STEPS = 100;

export interface Schedule {
  /** beta_t, the variance added at forward step t. */
  beta: Float64Array;
  /** alpha_t = 1 - beta_t. */
  alpha: Float64Array;
  /** alpha-bar_t = product of alpha_s for s <= t, as shipped in the schedule JSON. */
  alphaBar: Float64Array;
}

/**
 * Rebuild beta and alpha from the shipped cumulative product. Using the shipped
 * alpha-bar (rather than recomputing a linspace) keeps the sampler on exactly
 * the float values the denoiser was trained against.
 */
export function scheduleFromAlphaBar(alphaCumulative: ReadonlyArray<number>): Schedule {
  const steps = alphaCumulative.length;
  const alphaBar = Float64Array.from(alphaCumulative);
  const alpha = new Float64Array(steps);
  const beta = new Float64Array(steps);
  for (let t = 0; t < steps; t += 1) {
    alpha[t] = t === 0 ? alphaBar[0] : alphaBar[t] / alphaBar[t - 1];
    beta[t] = 1 - alpha[t];
  }
  return { beta, alpha, alphaBar };
}

/**
 * The cosine schedule as models/train_mnist_models.py defines it:
 * alpha-bar_t = f((t + 1) / steps * fraction) / f(0) with
 * f(u) = cos^2((u + offset) / (1 + offset) * pi / 2). Used by tests to check
 * that the shipped JSON really is the schedule the metadata describes.
 */
export function cosineSchedule(steps = STEPS, offset = 0.008, fraction = 0.98): Schedule {
  const f = (position: number) => Math.cos(((position + offset) / (1 + offset)) * (Math.PI / 2)) ** 2;
  const alphaBar = new Float64Array(steps);
  const alpha = new Float64Array(steps);
  const beta = new Float64Array(steps);
  for (let t = 0; t < steps; t += 1) {
    alphaBar[t] = f(((t + 1) / steps) * fraction) / f(0);
    alpha[t] = t === 0 ? alphaBar[0] : alphaBar[t] / alphaBar[t - 1];
    beta[t] = 1 - alpha[t];
  }
  return { beta, alpha, alphaBar };
}

/** Deterministic 32-bit generator (mulberry32). */
export function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seeded stream of standard normal draws (Box–Muller on mulberry32). */
export function normalStream(seed: number) {
  const uniform = mulberry32(seed * 7919 + 17);
  let spare: number | null = null;
  const next = () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    while (u <= 1e-12) u = uniform();
    const v = uniform();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
  return {
    next,
    fill(length = PIXELS) {
      const out = new Float32Array(length);
      for (let index = 0; index < length; index += 1) out[index] = next();
      return out;
    },
  };
}

/** q(x_t | x_0): x_t = sqrt(alpha-bar_t) x_0 + sqrt(1 - alpha-bar_t) epsilon. */
export function forwardNoise(
  x0: ArrayLike<number>,
  epsilon: ArrayLike<number>,
  alphaBar: number,
) {
  const signal = Math.sqrt(alphaBar);
  const noise = Math.sqrt(1 - alphaBar);
  const out = new Float32Array(x0.length);
  for (let index = 0; index < x0.length; index += 1) {
    out[index] = signal * x0[index] + noise * epsilon[index];
  }
  return out;
}

/** Invert the forward formula with a predicted epsilon: x0-hat = (x_t - sqrt(1-a) eps) / sqrt(a). */
export function predictCleanImage(
  xt: ArrayLike<number>,
  epsilonHat: ArrayLike<number>,
  alphaBar: number,
) {
  const signal = Math.sqrt(alphaBar);
  const noise = Math.sqrt(1 - alphaBar);
  const out = new Float32Array(xt.length);
  for (let index = 0; index < xt.length; index += 1) {
    out[index] = (xt[index] - noise * epsilonHat[index]) / signal;
  }
  return out;
}

/** Coefficients of one DDPM ancestral step, exposed so the lab can print them. */
export function stepCoefficients(schedule: Schedule, t: number) {
  const beta = schedule.beta[t];
  const alpha = schedule.alpha[t];
  const alphaBar = schedule.alphaBar[t];
  return {
    beta,
    alpha,
    alphaBar,
    /** Multiplies epsilon-hat inside the bracket. */
    epsilonScale: beta / Math.sqrt(1 - alphaBar),
    /** 1 / sqrt(alpha_t), applied to the bracket. */
    rescale: 1 / Math.sqrt(alpha),
    /** Standard deviation of the fresh noise added (0 on the final step). */
    sigma: t > 0 ? Math.sqrt(beta) : 0,
  };
}

/**
 * x_{t-1} = (x_t - beta_t / sqrt(1 - alpha-bar_t) * eps-hat) / sqrt(alpha_t) + sigma_t z,
 * with sigma_t = sqrt(beta_t) and no fresh noise when t = 0.
 */
export function ddpmStep(
  xt: ArrayLike<number>,
  epsilonHat: ArrayLike<number>,
  t: number,
  schedule: Schedule,
  z: ArrayLike<number> | null,
) {
  const { epsilonScale, rescale, sigma } = stepCoefficients(schedule, t);
  const out = new Float32Array(xt.length);
  for (let index = 0; index < xt.length; index += 1) {
    const mean = (xt[index] - epsilonScale * epsilonHat[index]) * rescale;
    out[index] = mean + (z && sigma > 0 ? sigma * z[index] : 0);
  }
  return out;
}

export interface IdealPrediction {
  x0: Float32Array;
  epsilon: Float32Array;
  /** Posterior weight of each reference image; sums to one. */
  weights: Float64Array;
}

/**
 * The exact minimum-mean-squared-error denoiser for a finite reference set:
 * E[x0 | x_t] = sum_i w_i r_i, with w = softmax(-||x_t - sqrt(a) r_i||^2 / (2 (1 - a))).
 * It is what a perfectly trained epsilon-predictor converges to on that set, so
 * sampling with it can only ever reproduce one of the references.
 */
export function idealDenoise(
  xt: ArrayLike<number>,
  t: number,
  references: ReadonlyArray<ArrayLike<number>>,
  schedule: Schedule,
): IdealPrediction {
  const alphaBar = schedule.alphaBar[t];
  const signal = Math.sqrt(alphaBar);
  const variance = 1 - alphaBar;
  const logits = references.map((reference) => {
    let distance = 0;
    for (let index = 0; index < xt.length; index += 1) {
      const gap = xt[index] - signal * reference[index];
      distance += gap * gap;
    }
    return -distance / (2 * variance);
  });
  const peak = Math.max(...logits);
  const weights = new Float64Array(references.length);
  let total = 0;
  logits.forEach((logit, index) => {
    weights[index] = Math.exp(logit - peak);
    total += weights[index];
  });
  const x0 = new Float32Array(xt.length);
  references.forEach((reference, referenceIndex) => {
    const weight = weights[referenceIndex] / total;
    weights[referenceIndex] = weight;
    if (weight < 1e-12) return;
    for (let index = 0; index < xt.length; index += 1) {
      x0[index] += weight * reference[index];
    }
  });
  const epsilon = new Float32Array(xt.length);
  const noise = Math.sqrt(variance);
  for (let index = 0; index < xt.length; index += 1) {
    epsilon[index] = (xt[index] - signal * x0[index]) / noise;
  }
  return { x0, epsilon, weights };
}

export type StartKind = "noise" | "digit";

export interface ReverseStart {
  /** x at the highest noise level t = STEPS - 1, before any denoiser call. */
  x: Float32Array;
  /** Fresh noise for each ancestral step, indexed by denoiser call k (t = 99 - k). */
  z: Float32Array[];
}

/**
 * Seeded start for the reverse process. The first 784 draws are the start
 * noise; the next draws are the fresh noise z for each step, so the trained
 * network and the ideal denoiser see identical randomness for one seed.
 */
export function reverseStart(
  seed: number,
  start: StartKind,
  x0: ArrayLike<number> | null,
  schedule: Schedule,
): ReverseStart {
  const stream = normalStream(seed);
  const epsilon = stream.fill();
  const last = schedule.alphaBar.length - 1;
  const x = start === "digit" && x0 ? forwardNoise(x0, epsilon, schedule.alphaBar[last]) : epsilon;
  const z = Array.from({ length: schedule.alphaBar.length }, () => stream.fill());
  return { x, z };
}

export interface Trajectory {
  /** states[k] is x after k denoiser calls; states[0] is the start, states[STEPS] the sample. */
  states: Float32Array[];
  /** Predicted clean image at call k (made from states[k]). */
  cleanPredictions: Float32Array[];
  /** Predicted noise at call k. */
  noisePredictions: Float32Array[];
  /** Ideal denoiser only: the reference weights at call k. */
  weights?: Float64Array[];
}

/** Map denoiser call k (0-based) to the noise level it removes. */
export const timestepForCall = (call: number, steps = STEPS) => steps - 1 - call;

/** One denoiser call: the predicted noise for an image at noise level t. */
export type NoisePredictor = (x: Float32Array, t: number) => Promise<Float32Array>;

/**
 * Run all reverse steps with a trained network, one awaited call per step. The
 * lab supplies a predictor backed by the ONNX model worker; tests supply one
 * backed by onnxruntime-web, so both exercise this same loop. `beforeCall` runs
 * ahead of every call and may throw to cancel.
 */
export async function runNetworkReverse(
  start: ReverseStart,
  predict: NoisePredictor,
  schedule: Schedule,
  beforeCall?: (call: number) => void,
): Promise<Trajectory> {
  const steps = schedule.alphaBar.length;
  const states = [start.x];
  const cleanPredictions: Float32Array[] = [];
  const noisePredictions: Float32Array[] = [];
  let x = start.x;
  for (let call = 0; call < steps; call += 1) {
    beforeCall?.(call);
    const t = timestepForCall(call, steps);
    const epsilon = await predict(x, t);
    noisePredictions.push(epsilon);
    cleanPredictions.push(predictCleanImage(x, epsilon, schedule.alphaBar[t]));
    x = ddpmStep(x, epsilon, t, schedule, t > 0 ? start.z[call] : null);
    states.push(x);
  }
  return { states, cleanPredictions, noisePredictions };
}

/** Predict the noise for a batch of images, one noise level each, in one call. */
export type BatchNoisePredictor = (
  images: ReadonlyArray<Float32Array>,
  timesteps: ReadonlyArray<number>,
) => Promise<ReadonlyArray<Float32Array>>;

/**
 * The network's mean squared error at predicting epsilon at every noise level,
 * for one clean image and one noise draw, against the error of predicting zero
 * (the mean of epsilon squared). This is the Forward process card's error chart.
 */
export async function noiseErrorByStep(
  x0: ArrayLike<number>,
  epsilon: Float32Array,
  schedule: Schedule,
  predict: BatchNoisePredictor,
) {
  const timesteps = Array.from({ length: schedule.alphaBar.length }, (_, t) => t);
  const noisy = timesteps.map((t) => forwardNoise(x0, epsilon, schedule.alphaBar[t]));
  const predictions = await predict(noisy, timesteps);
  return {
    network: predictions.map((prediction) => meanSquaredError(prediction, epsilon)),
    baseline: meanSquare(epsilon),
  };
}

/** Run all reverse steps with the ideal denoiser. Pure and synchronous. */
export function runIdealReverse(
  start: ReverseStart,
  references: ReadonlyArray<ArrayLike<number>>,
  schedule: Schedule,
): Trajectory {
  const steps = schedule.alphaBar.length;
  const states = [start.x];
  const cleanPredictions: Float32Array[] = [];
  const noisePredictions: Float32Array[] = [];
  const weights: Float64Array[] = [];
  let x = start.x;
  for (let call = 0; call < steps; call += 1) {
    const t = timestepForCall(call, steps);
    const prediction = idealDenoise(x, t, references, schedule);
    cleanPredictions.push(prediction.x0);
    noisePredictions.push(prediction.epsilon);
    weights.push(prediction.weights);
    x = ddpmStep(x, prediction.epsilon, t, schedule, t > 0 ? start.z[call] : null);
    states.push(x);
  }
  return { states, cleanPredictions, noisePredictions, weights };
}

export function meanSquaredError(a: ArrayLike<number>, b: ArrayLike<number>) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    const gap = a[index] - b[index];
    total += gap * gap;
  }
  return total / Math.max(1, a.length);
}

/**
 * Share of pixels at or below -0.9, the black background of an MNIST-style digit.
 * Standard-normal noise has about 0.18 of its pixels there and real digits about
 * 0.8, so unlike pixel RMS (about 1.0 for noise and 0.96 for digits) it separates
 * the two ends of the reverse process.
 */
export function backgroundShare(image: ArrayLike<number>, threshold = -0.9) {
  let count = 0;
  for (let index = 0; index < image.length; index += 1) {
    if (image[index] <= threshold) count += 1;
  }
  return count / Math.max(1, image.length);
}

export function meanSquare(a: ArrayLike<number>) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) total += a[index] * a[index];
  return total / Math.max(1, a.length);
}

/** Pixel values in [0, 1] (decoder output) to the [-1, 1] range diffusion trains on. */
export function toSigned(image: ArrayLike<number>) {
  return Float32Array.from(image, (value) => value * 2 - 1);
}
