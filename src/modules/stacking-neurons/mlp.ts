/**
 * A real, tiny multilayer perceptron for the stacking-neurons lab.
 *
 * Two inputs, `depth` hidden layers of `width` units each, one sigmoid output.
 * Trained by full-batch gradient descent on mean binary cross-entropy, so every
 * number the lab prints comes from an actual forward and backward pass.
 */

export type HiddenActivation = "tanh" | "relu" | "linear";
export type DatasetKind = "xor" | "ring" | "checker";

export const HIDDEN_ACTIVATIONS: readonly HiddenActivation[] = ["tanh", "relu", "linear"];
export const DATASETS: readonly DatasetKind[] = ["xor", "ring", "checker"];

export const MIN_DEPTH = 1;
export const MAX_DEPTH = 3;
export const MIN_WIDTH = 1;
export const MAX_WIDTH = 6;
export const MAX_EPOCHS = 300;
export const LEARNING_RATE = 0.5;
export const MOMENTUM = 0.9;

export interface Example {
  x1: number;
  x2: number;
  label: 0 | 1;
}

export interface Layer {
  /** weights[out][in] */
  weights: number[][];
  biases: number[];
}

export interface Network {
  layers: Layer[];
}

export interface ForwardPass {
  /** Pre-activation z for every non-input layer. */
  pre: number[][];
  /** Activations: post[0] is the input, the last entry is the sigmoid output. */
  post: number[][];
  output: number;
}

export interface TrainConfig {
  dataset: DatasetKind;
  depth: number;
  width: number;
  activation: HiddenActivation;
  seed: number;
  epochs?: number;
  /** Train on these points instead of the dataset's own (the held-out card uses a noisy ring). */
  data?: readonly Example[];
  /** Points the network never trains on; their loss and accuracy are recorded for every epoch. */
  heldOut?: readonly Example[];
}

export interface TrainHistory {
  data: Example[];
  sizes: number[];
  /** snapshots[e] is the network after e epochs (e = 0 is the initialization). */
  snapshots: Network[];
  /** losses[e] is the mean cross-entropy after e epochs (e = 0 is the initialization). */
  losses: number[];
  accuracies: number[];
  /** Held-out loss and accuracy after e epochs, present only when `heldOut` was given. */
  heldLosses?: number[];
  heldAccuracies?: number[];
}

/** Deterministic 32-bit PRNG so a seed always rebuilds the same run. */
export function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number) {
  const u = Math.max(1e-12, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const clampUnit = (value: number) => Math.min(1, Math.max(-1, value));

/**
 * Fixed training sets on the square [−1, 1]².
 * - xor: four clusters of 12 points; opposite corners share a label.
 * - ring: 24 points inside a disc (label 1) and 36 on a surrounding ring (label 0).
 * - checker: a 4 × 4 checkerboard, four points per cell, 64 in all.
 */
export function makeDataset(kind: DatasetKind): Example[] {
  if (kind === "checker") return checkerPoints();
  if (kind === "ring") return ringPoints(24, 36, 23);
  const rand = mulberry32(11);
  const points: Example[] = [];
  const centres: Array<[number, number, 0 | 1]> = [
    [-0.6, -0.6, 0],
    [0.6, 0.6, 0],
    [-0.6, 0.6, 1],
    [0.6, -0.6, 1],
  ];
  for (const [cx, cy, label] of centres) {
    for (let index = 0; index < 12; index += 1) {
      points.push({
        x1: clampUnit(cx + 0.15 * gaussian(rand)),
        x2: clampUnit(cy + 0.15 * gaussian(rand)),
        label,
      });
    }
  }
  return points;
}

/** `inner` points inside a disc (label 1) and `outer` on a surrounding ring (label 0). */
function ringPoints(inner: number, outer: number, seed: number): Example[] {
  const rand = mulberry32(seed);
  const points: Example[] = [];
  for (let index = 0; index < inner; index += 1) {
    const radius = 0.4 * Math.sqrt(rand());
    const angle = 2 * Math.PI * rand();
    points.push({ x1: radius * Math.cos(angle), x2: radius * Math.sin(angle), label: 1 });
  }
  for (let index = 0; index < outer; index += 1) {
    const radius = 0.66 + 0.24 * rand();
    const angle = 2 * Math.PI * (index / outer) + 0.12 * gaussian(rand);
    points.push({ x1: radius * Math.cos(angle), x2: radius * Math.sin(angle), label: 0 });
  }
  return points;
}

export const CHECKER_CELLS = 4;
export const CHECKER_PER_CELL = 4;

/** A 4 × 4 checkerboard: four points in the middle half of every cell, label = parity of the cell. */
function checkerPoints(): Example[] {
  const rand = mulberry32(5);
  const margin = 0.25;
  const points: Example[] = [];
  for (let row = 0; row < CHECKER_CELLS; row += 1) {
    for (let column = 0; column < CHECKER_CELLS; column += 1) {
      for (let index = 0; index < CHECKER_PER_CELL; index += 1) {
        points.push({
          x1: -1 + (2 * (row + margin + (1 - 2 * margin) * rand())) / CHECKER_CELLS,
          x2: -1 + (2 * (column + margin + (1 - 2 * margin) * rand())) / CHECKER_CELLS,
          label: ((row + column) % 2) as 0 | 1,
        });
      }
    }
  }
  return points;
}

export function layerSizes(depth: number, width: number) {
  return [2, ...Array.from({ length: depth }, () => width), 1];
}

/** Weights plus biases: Σ (n_in × n_out + n_out) over every layer. */
export function parameterCount(sizes: readonly number[]) {
  let total = 0;
  for (let index = 1; index < sizes.length; index += 1) {
    total += sizes[index - 1] * sizes[index] + sizes[index];
  }
  return total;
}

/** Glorot-uniform weights (He-uniform for ReLU) and zero biases. */
export function initNetwork(sizes: readonly number[], seed: number, activation: HiddenActivation): Network {
  const rand = mulberry32(seed * 7919 + 17);
  const layers: Layer[] = [];
  for (let index = 1; index < sizes.length; index += 1) {
    const fanIn = sizes[index - 1];
    const fanOut = sizes[index];
    const isOutput = index === sizes.length - 1;
    const limit =
      activation === "relu" && !isOutput ? Math.sqrt(6 / fanIn) : Math.sqrt(6 / (fanIn + fanOut));
    layers.push({
      weights: Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => (rand() * 2 - 1) * limit),
      ),
      biases: Array.from({ length: fanOut }, () => (activation === "relu" && !isOutput ? 0.1 : 0)),
    });
  }
  return { layers };
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export function activate(z: number, activation: HiddenActivation) {
  if (activation === "tanh") return Math.tanh(z);
  if (activation === "relu") return z > 0 ? z : 0;
  return z;
}

/** Derivative with respect to z, given z and the already computed activation a. */
function activationSlope(z: number, a: number, activation: HiddenActivation) {
  if (activation === "tanh") return 1 - a * a;
  if (activation === "relu") return z > 0 ? 1 : 0;
  return 1;
}

export function forward(network: Network, x: readonly [number, number], activation: HiddenActivation): ForwardPass {
  const pre: number[][] = [];
  const post: number[][] = [[x[0], x[1]]];
  network.layers.forEach((layer, layerIndex) => {
    const input = post[post.length - 1];
    const isOutput = layerIndex === network.layers.length - 1;
    const z = layer.weights.map(
      (row, unit) => row.reduce((sum, weight, index) => sum + weight * input[index], layer.biases[unit]),
    );
    pre.push(z);
    post.push(z.map((value) => (isOutput ? sigmoid(value) : activate(value, activation))));
  });
  return { pre, post, output: post[post.length - 1][0] };
}

const EPS = 1e-12;

/** Mean binary cross-entropy −[y ln p + (1−y) ln(1−p)] over the examples. */
export function meanLoss(network: Network, data: readonly Example[], activation: HiddenActivation) {
  let total = 0;
  for (const example of data) {
    const p = forward(network, [example.x1, example.x2], activation).output;
    total -= example.label * Math.log(Math.max(EPS, p)) + (1 - example.label) * Math.log(Math.max(EPS, 1 - p));
  }
  return total / data.length;
}

export function accuracy(network: Network, data: readonly Example[], activation: HiddenActivation) {
  let correct = 0;
  for (const example of data) {
    const p = forward(network, [example.x1, example.x2], activation).output;
    if ((p >= 0.5 ? 1 : 0) === example.label) correct += 1;
  }
  return correct / data.length;
}

/**
 * Exact gradient of the mean cross-entropy by backpropagation, plus the loss and
 * accuracy at the parameters the gradient was taken at.
 */
export function gradients(network: Network, data: readonly Example[], activation: HiddenActivation) {
  const grads: Layer[] = network.layers.map((layer) => ({
    weights: layer.weights.map((row) => row.map(() => 0)),
    biases: layer.biases.map(() => 0),
  }));
  const scale = 1 / data.length;
  let loss = 0;
  let correct = 0;
  for (const example of data) {
    const pass = forward(network, [example.x1, example.x2], activation);
    const p = pass.output;
    loss -= example.label * Math.log(Math.max(EPS, p)) + (1 - example.label) * Math.log(Math.max(EPS, 1 - p));
    if ((p >= 0.5 ? 1 : 0) === example.label) correct += 1;
    // Sigmoid output with cross-entropy: dL/dz_out = p − y.
    let delta = [p - example.label];
    for (let layerIndex = network.layers.length - 1; layerIndex >= 0; layerIndex -= 1) {
      const layer = network.layers[layerIndex];
      const input = pass.post[layerIndex];
      const grad = grads[layerIndex];
      for (let unit = 0; unit < delta.length; unit += 1) {
        grad.biases[unit] += scale * delta[unit];
        for (let index = 0; index < input.length; index += 1) {
          grad.weights[unit][index] += scale * delta[unit] * input[index];
        }
      }
      if (layerIndex === 0) break;
      const z = pass.pre[layerIndex - 1];
      const a = pass.post[layerIndex];
      const next: number[] = [];
      for (let index = 0; index < input.length; index += 1) {
        let sum = 0;
        for (let unit = 0; unit < delta.length; unit += 1) sum += layer.weights[unit][index] * delta[unit];
        next.push(sum * activationSlope(z[index], a[index], activation));
      }
      delta = next;
    }
  }
  return { grads, loss: loss / data.length, accuracy: correct / data.length };
}

const cloneNetwork = (network: Network): Network => ({
  layers: network.layers.map((layer) => ({
    weights: layer.weights.map((row) => [...row]),
    biases: [...layer.biases],
  })),
});

/**
 * Full-batch gradient descent with heavy-ball momentum. One epoch is one pass over
 * every training example followed by one update:
 * v ← μ·v − η·∇L,  θ ← θ + v.
 */
export function train(config: TrainConfig): TrainHistory {
  const epochs = config.epochs ?? MAX_EPOCHS;
  const data = config.data ? [...config.data] : makeDataset(config.dataset);
  const sizes = layerSizes(config.depth, config.width);
  const network = initNetwork(sizes, config.seed, config.activation);
  const velocity: Layer[] = network.layers.map((layer) => ({
    weights: layer.weights.map((row) => row.map(() => 0)),
    biases: layer.biases.map(() => 0),
  }));
  const snapshots: Network[] = [cloneNetwork(network)];
  const losses: number[] = [];
  const accuracies: number[] = [];

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const step = gradients(network, data, config.activation);
    losses.push(step.loss);
    accuracies.push(step.accuracy);
    network.layers.forEach((layer, layerIndex) => {
      const v = velocity[layerIndex];
      const g = step.grads[layerIndex];
      layer.weights.forEach((row, unit) => {
        for (let index = 0; index < row.length; index += 1) {
          v.weights[unit][index] = MOMENTUM * v.weights[unit][index] - LEARNING_RATE * g.weights[unit][index];
          row[index] += v.weights[unit][index];
        }
      });
      for (let unit = 0; unit < layer.biases.length; unit += 1) {
        v.biases[unit] = MOMENTUM * v.biases[unit] - LEARNING_RATE * g.biases[unit];
        layer.biases[unit] += v.biases[unit];
      }
    });
    snapshots.push(cloneNetwork(network));
  }
  losses.push(meanLoss(network, data, config.activation));
  accuracies.push(accuracy(network, data, config.activation));

  const history: TrainHistory = { data, sizes, snapshots, losses, accuracies };
  if (config.heldOut) {
    const held = [...config.heldOut];
    const scores = snapshots.map((snapshot) => evaluate(snapshot, held, config.activation));
    history.heldLosses = scores.map((score) => score.loss);
    history.heldAccuracies = scores.map((score) => score.accuracy);
  }
  return history;
}

/** Loss and accuracy in one forward pass over the examples. */
function evaluate(network: Network, data: readonly Example[], activation: HiddenActivation) {
  let loss = 0;
  let correct = 0;
  for (const example of data) {
    const p = forward(network, [example.x1, example.x2], activation).output;
    loss -= example.label * Math.log(Math.max(EPS, p)) + (1 - example.label) * Math.log(Math.max(EPS, 1 - p));
    if ((p >= 0.5 ? 1 : 0) === example.label) correct += 1;
  }
  return { loss: loss / data.length, accuracy: correct / data.length };
}

/**
 * Training is deterministic, so a finished run is kept and reused when the same controls come back
 * (the epoch slider and the probe never retrain). Histories are read-only to every caller.
 */
const remembered = new Map<string, TrainHistory>();
const REMEMBER_LIMIT = 40;
function remember(key: string, build: () => TrainHistory) {
  const found = remembered.get(key);
  if (found) return found;
  const made = build();
  if (remembered.size >= REMEMBER_LIMIT) remembered.delete(remembered.keys().next().value as string);
  remembered.set(key, made);
  return made;
}

/** `train` for the dataset's own points, reusing an identical earlier run. */
export function trainCached(config: Omit<TrainConfig, "data" | "heldOut">): TrainHistory {
  const { dataset, depth, width, activation, seed, epochs } = config;
  return remember(`${dataset}|${depth}|${width}|${activation}|${seed}|${epochs ?? MAX_EPOCHS}`, () => train(config));
}

/* ------------------------------------------------------------ held-out data */

/** Networks compared on the held-out card, and the labels flipped to make the training set noisy. */
export const NARROW_NET = { depth: 1, width: 3 } as const;
export const LARGE_NET = { depth: 2, width: 6 } as const;
export const NOISE_FLIPS = 12;
const NOISE_SEED = 7;
const HELD_OUT_SEED = 91;

/** 120 clean ring points the networks never train on: 48 inside the disc, 72 on the ring. */
export function makeRingHeldOut(): Example[] {
  return ringPoints(48, 72, HELD_OUT_SEED);
}

/** Flip the label of exactly `count` points, chosen by a seeded shuffle so the same ones flip every time. */
export function flipLabels(data: readonly Example[], count: number, seed = NOISE_SEED): Example[] {
  const rand = mulberry32(seed);
  const order = data.map((_, index) => index);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rand() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  const chosen = new Set(order.slice(0, Math.min(count, data.length)));
  return data.map((point, index) => (chosen.has(index) ? { ...point, label: (1 - point.label) as 0 | 1 } : { ...point }));
}

export interface RingSplitConfig {
  depth: number;
  width: number;
  seed: number;
  /** Flip NOISE_FLIPS of the 60 training labels (20%); the held-out labels are always clean. */
  noisy: boolean;
  activation?: HiddenActivation;
}

/** Train on the ring (clean or with flipped labels) and record loss and accuracy on the held-out ring points. */
export function trainRingSplit(config: RingSplitConfig): TrainHistory {
  const activation = config.activation ?? "tanh";
  const key = `split|${config.depth}|${config.width}|${activation}|${config.seed}|${config.noisy}`;
  return remember(key, () => {
    const clean = makeDataset("ring");
    return train({
      dataset: "ring",
      depth: config.depth,
      width: config.width,
      activation,
      seed: config.seed,
      data: config.noisy ? flipLabels(clean, NOISE_FLIPS) : clean,
      heldOut: makeRingHeldOut(),
    });
  });
}

/* ------------------------------------------------------- equal-size shapes */

/** Two shapes with the same number of parameters: one wide layer against two narrow ones. */
export const SHALLOW_WIDE_NET = { depth: 1, width: 21 } as const;
export const DEEP_NARROW_NET = { depth: 2, width: 7 } as const;

/** The network after `epoch` epochs of training. */
export function snapshotAt(history: TrainHistory, epoch: number) {
  const index = Math.min(history.snapshots.length - 1, Math.max(0, Math.round(epoch)));
  return history.snapshots[index];
}

/** The straight line where a unit's pre-activation w·x + b is zero, clipped to [−1, 1]². */
export function unitBoundary(weights: readonly number[], bias: number) {
  const [w1, w2] = weights;
  const hits: Array<[number, number]> = [];
  const inside = (value: number) => value >= -1 - 1e-9 && value <= 1 + 1e-9;
  const push = (x1: number, x2: number) => {
    if (inside(x1) && inside(x2) && !hits.some(([a, b]) => Math.hypot(a - x1, b - x2) < 1e-6)) {
      hits.push([x1 + 0, x2 + 0]);
    }
  };
  if (Math.abs(w2) > 1e-9) {
    push(-1, -(bias - w1) / w2);
    push(1, -(bias + w1) / w2);
  }
  if (Math.abs(w1) > 1e-9) {
    push(-(bias - w2) / w1, -1);
    push(-(bias + w2) / w1, 1);
  }
  return hits.length >= 2 ? ([hits[0], hits[1]] as const) : null;
}
