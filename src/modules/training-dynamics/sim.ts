/**
 * A real, seeded training run for the training-dynamics lab.
 *
 * A polynomial model ŷ = Σ θₖ φₖ(x) is fitted by minibatch gradient descent to
 * sixteen noisy samples of a known curve and scored every checkpoint on sixteen
 * held-out samples. The features φₖ are Legendre polynomials scaled by √(2k+1)
 * so that every degree has a comparable size on [−1, 1]. Every loss the lab
 * plots comes from `train`; nothing is a drawn curve.
 */

export const TRAIN_SIZE = 16;
export const VALIDATION_SIZE = 16;
export const NOISE_SD = 0.25;
export const NOISE_VARIANCE = NOISE_SD * NOISE_SD;
export const MAX_EPOCHS = 10_000;
export const DEGREE_MIN = 1;
export const DEGREE_MAX = 13;
export const LEARNING_RATES = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6] as const;
export const WEIGHT_DECAYS = [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1] as const;
export const BATCH_SIZES = [4, 8, 16] as const;
/** A run whose training loss passes this is reported as diverged and stops. */
export const DIVERGED_LOSS = 1e6;
/** Status reads "overfitting" once validation loss is this many times its best so far. */
export const OVERFIT_RATIO = 1.5;
/** Status reads "underfitting" while training loss is this many times the noise variance. */
export const UNDERFIT_RATIO = 2;

/** The pattern the data came from. The model never sees this function. */
export const trueFunction = (x: number) => 0.8 * Math.sin(4 * x) + 0.4 * x;

/** Deterministic 32-bit generator, so every run is reproducible from its seed. */
export function mulberry32(seed: number) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller normal sample from a uniform generator. */
export function gaussian(random: () => number) {
  let u = 0;
  while (u === 0) u = random();
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** φ₀ … φ_degree at x: Legendre polynomials times √(2k+1). */
export function features(x: number, degree: number) {
  const values = new Float64Array(degree + 1);
  let previous = 1;
  let currentValue = x;
  values[0] = 1;
  if (degree >= 1) values[1] = x;
  for (let k = 2; k <= degree; k += 1) {
    const next = ((2 * k - 1) * x * currentValue - (k - 1) * previous) / k;
    previous = currentValue;
    currentValue = next;
    values[k] = next;
  }
  for (let k = 0; k <= degree; k += 1) values[k] *= Math.sqrt(2 * k + 1);
  return values;
}

export interface Sample {
  x: number;
  y: number;
}

/**
 * Thirty-two noisy samples on a jittered grid over [−1, 1]. Alternate samples
 * go to training and validation, so both splits cover the whole range.
 */
export function makeDataset(seed: number) {
  const random = mulberry32(seed);
  const total = TRAIN_SIZE + VALIDATION_SIZE;
  const train: Sample[] = [];
  const validation: Sample[] = [];
  for (let index = 0; index < total; index += 1) {
    const x = -1 + ((index + 0.15 + 0.7 * random()) * 2) / total;
    const y = trueFunction(x) + NOISE_SD * gaussian(random);
    (index % 2 === 0 ? train : validation).push({ x, y });
  }
  return { train, validation };
}

/** Epochs at which losses are recorded: 0, then about 24 per decade up to MAX_EPOCHS. */
export const CHECKPOINTS: readonly number[] = (() => {
  const epochs = [0];
  for (let index = 0; index <= 96; index += 1) {
    const epoch = Math.round(10 ** (index / 24));
    if (epoch > epochs[epochs.length - 1]) epochs.push(epoch);
  }
  return epochs;
})();

export interface Checkpoint {
  epoch: number;
  train: number;
  validation: number;
  theta: number[];
}

export interface TrainRun {
  degree: number;
  checkpoints: Checkpoint[];
  /** Epoch at which the training loss passed DIVERGED_LOSS, or null. */
  divergedAt: number | null;
  /** Index into checkpoints of the lowest validation loss. */
  bestIndex: number;
  /** Losses of the true curve on the same samples: what fitting only the pattern would score. */
  trueLoss: { train: number; validation: number };
}

export interface TrainOptions {
  degree: number;
  learningRate: number;
  weightDecay: number;
  batchSize: number;
  seed: number;
}

function meanSquaredError(theta: Float64Array, rows: Float64Array[], samples: Sample[]) {
  let sum = 0;
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    let prediction = 0;
    for (let k = 0; k < theta.length; k += 1) prediction += theta[k] * row[k];
    const residual = prediction - samples[index].y;
    sum += residual * residual;
  }
  return sum / rows.length;
}

/**
 * Minibatch gradient descent from θ = 0 on L = mean (ŷ − y)² + λ Σ_{k≥1} θₖ².
 * Each epoch reshuffles the sixteen training samples and takes 16 / batch
 * steps. Recorded training loss is the data term only, without the penalty.
 * The constant feature φ₀ is not decayed, as biases usually are not.
 */
export function train({ degree, learningRate, weightDecay, batchSize, seed }: TrainOptions): TrainRun {
  const { train: trainSet, validation } = makeDataset(seed);
  const width = degree + 1;
  const trainRows = trainSet.map((sample) => features(sample.x, degree));
  const validationRows = validation.map((sample) => features(sample.x, degree));
  const theta = new Float64Array(width);
  const gradient = new Float64Array(width);
  const order = trainSet.map((_, index) => index);
  const shuffle = mulberry32(seed * 7919 + 17);
  const batch = Math.max(1, Math.min(trainSet.length, Math.round(batchSize)));
  const checkpoints: Checkpoint[] = [];
  let divergedAt: number | null = null;
  let nextCheckpoint = 0;

  const record = (epoch: number) => {
    checkpoints.push({
      epoch,
      train: meanSquaredError(theta, trainRows, trainSet),
      validation: meanSquaredError(theta, validationRows, validation),
      theta: Array.from(theta),
    });
  };

  record(0);
  nextCheckpoint = 1;

  for (let epoch = 1; epoch <= MAX_EPOCHS; epoch += 1) {
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(shuffle() * (index + 1));
      [order[index], order[swap]] = [order[swap], order[index]];
    }
    for (let start = 0; start < order.length; start += batch) {
      const end = Math.min(order.length, start + batch);
      gradient.fill(0);
      for (let position = start; position < end; position += 1) {
        const row = trainRows[order[position]];
        let prediction = 0;
        for (let k = 0; k < width; k += 1) prediction += theta[k] * row[k];
        const scaled = (2 * (prediction - trainSet[order[position]].y)) / (end - start);
        for (let k = 0; k < width; k += 1) gradient[k] += scaled * row[k];
      }
      for (let k = 0; k < width; k += 1) {
        const decay = k === 0 ? 0 : 2 * weightDecay * theta[k];
        theta[k] -= learningRate * (gradient[k] + decay);
      }
    }

    if (epoch === CHECKPOINTS[nextCheckpoint]) {
      record(epoch);
      nextCheckpoint += 1;
      const latest = checkpoints[checkpoints.length - 1];
      if (!Number.isFinite(latest.train) || latest.train > DIVERGED_LOSS) {
        divergedAt = epoch;
        break;
      }
    }
  }

  let bestIndex = 0;
  checkpoints.forEach((checkpoint, index) => {
    if (checkpoint.validation < checkpoints[bestIndex].validation) bestIndex = index;
  });

  const trueLoss = (samples: Sample[]) =>
    samples.reduce((sum, sample) => sum + (trueFunction(sample.x) - sample.y) ** 2, 0) / samples.length;

  return {
    degree,
    checkpoints,
    divergedAt,
    bestIndex,
    trueLoss: { train: trueLoss(trainSet), validation: trueLoss(validation) },
  };
}

export function predict(theta: ReadonlyArray<number>, x: number) {
  const row = features(x, theta.length - 1);
  let value = 0;
  for (let k = 0; k < theta.length; k += 1) value += theta[k] * row[k];
  return value;
}

/** Index of the last recorded checkpoint at or before `epoch`. */
export function checkpointAt(run: TrainRun, epoch: number) {
  let found = 0;
  run.checkpoints.forEach((checkpoint, index) => {
    if (checkpoint.epoch <= epoch) found = index;
  });
  return found;
}

export type Status = "diverged" | "underfitting" | "overfitting" | "fitting";

/**
 * The status label is a rule applied to measured losses, not a verdict:
 * diverged once training loss blew past DIVERGED_LOSS; overfitting once
 * validation loss sits OVERFIT_RATIO times above the best it reached earlier
 * in the same run; underfitting while training loss is still more than
 * UNDERFIT_RATIO times the noise variance; otherwise fitting.
 */
export function statusAt(run: TrainRun, epoch: number): Status {
  if (run.divergedAt !== null && epoch >= run.divergedAt) return "diverged";
  const index = checkpointAt(run, epoch);
  const current = run.checkpoints[index];
  let bestSoFar = Infinity;
  for (let position = 0; position <= index; position += 1) {
    bestSoFar = Math.min(bestSoFar, run.checkpoints[position].validation);
  }
  if (current.validation > OVERFIT_RATIO * bestSoFar) return "overfitting";
  if (current.train > UNDERFIT_RATIO * NOISE_VARIANCE) return "underfitting";
  return "fitting";
}

/**
 * Gradient noise at θ: the full-batch gradient's length, and the root-mean-square
 * distance between each minibatch gradient and it, over one pass in index order.
 */
export function gradientNoise(theta: ReadonlyArray<number>, batchSize: number, seed: number) {
  const { train: trainSet } = makeDataset(seed);
  const degree = theta.length - 1;
  const rows = trainSet.map((sample) => features(sample.x, degree));
  const gradientOf = (indices: number[]) => {
    const g = new Float64Array(theta.length);
    for (const index of indices) {
      let prediction = 0;
      for (let k = 0; k < theta.length; k += 1) prediction += theta[k] * rows[index][k];
      const scaled = (2 * (prediction - trainSet[index].y)) / indices.length;
      for (let k = 0; k < theta.length; k += 1) g[k] += scaled * rows[index][k];
    }
    return g;
  };
  const all = trainSet.map((_, index) => index);
  const full = gradientOf(all);
  const norm = (g: Float64Array) => Math.sqrt(g.reduce((sum, value) => sum + value * value, 0));
  const batch = Math.max(1, Math.min(all.length, Math.round(batchSize)));
  let squared = 0;
  let count = 0;
  for (let start = 0; start < all.length; start += batch) {
    const g = gradientOf(all.slice(start, start + batch));
    let distance = 0;
    for (let k = 0; k < g.length; k += 1) distance += (g[k] - full[k]) ** 2;
    squared += distance;
    count += 1;
  }
  return { full: norm(full), noise: Math.sqrt(squared / Math.max(1, count)) };
}
