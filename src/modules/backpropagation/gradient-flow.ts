/**
 * Gradient flow through a deep multilayer perceptron at initialization.
 *
 * One fixed batch goes forward through `depth` hidden layers of 16 units, one sigmoid
 * output reads the last of them, and the mean binary cross-entropy is backpropagated by
 * hand. Every number the lab prints is the Frobenius norm of the exact gradient of that
 * loss with respect to one layer's weight matrix, measured before any training step.
 *
 * Nothing here is trained and nothing is normalized: this isolates how the multiplied
 * local factors of the chain rule behave as layers are stacked. Biases are zero and left
 * out of the norms.
 */

export type FlowActivation = "sigmoid" | "tanh" | "relu";
export type FlowInit = "small" | "glorot" | "he" | "large";

export const FLOW_ACTIVATIONS: readonly FlowActivation[] = ["sigmoid", "tanh", "relu"];
export const FLOW_INITS: readonly FlowInit[] = ["small", "glorot", "he", "large"];

/** Every hidden matrix is 16 × 16 (inputs are 16-dimensional too), so layer norms compare like with like. */
export const FLOW_WIDTH = 16;
export const FLOW_BATCH = 32;
export const MIN_FLOW_DEPTH = 1;
export const MAX_FLOW_DEPTH = 12;
export const FLOW_SEED = 7;

export interface FlowConfig {
  depth: number;
  activation: FlowActivation;
  init: FlowInit;
  /** Residual connections: h_l = h_(l−1) + f(W_l h_(l−1)) for every layer after the first. */
  skip: boolean;
  seed?: number;
}

export interface FlowLayer {
  /** 1 is the layer nearest the input. */
  layer: number;
  /** Frobenius norm of ∂L/∂W for this layer's 16 × 16 weights. */
  gradNorm: number;
  /** Mean of f′(z) over this layer's units and the batch: the local factor the chain rule multiplies in. */
  meanSlope: number;
  /** Root-mean-square of this layer's output over units and the batch. */
  activationRms: number;
}

export interface FlowResult {
  config: Required<FlowConfig>;
  loss: number;
  layers: FlowLayer[];
  /** First hidden layer's gradient norm divided by the last hidden layer's; 1 when depth is 1. */
  firstOverLast: number;
}

export interface FlowNetwork {
  /** hidden[l] is the 16 × 16 weight matrix of hidden layer l + 1, indexed [unit][input]. */
  hidden: number[][][];
  /** The output unit's 16 weights. */
  output: number[];
}

export interface FlowGradients {
  loss: number;
  hidden: number[][][];
  output: number[];
  /** Per hidden layer: Σ f′(z) and Σ h² over units and examples, for the readouts above. */
  slopeSums: number[];
  activationSquares: number[];
}

/** Deterministic 32-bit PRNG so a seed always rebuilds the same batch and weights. */
export function flowRandom(seed: number) {
  let state = Math.floor(seed) >>> 0;
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

/**
 * Standard deviation of every weight in a layer with the given fan-in and fan-out.
 * Glorot: √(2 / (fan-in + fan-out)). He: √(2 / fan-in). The other two are fixed.
 */
export function initStd(init: FlowInit, fanIn: number, fanOut: number) {
  if (init === "small") return 0.1;
  if (init === "large") return 1;
  if (init === "he") return Math.sqrt(2 / fanIn);
  return Math.sqrt(2 / (fanIn + fanOut));
}

export interface FlowBatch {
  inputs: number[][];
  labels: number[];
}

/** 32 standard-normal points in 16 dimensions, labelled by the sign of a fixed random direction. */
export function makeFlowBatch(seed = FLOW_SEED): FlowBatch {
  const rand = flowRandom(seed * 104729 + 3);
  const teacher = Array.from({ length: FLOW_WIDTH }, () => gaussian(rand));
  const inputs = Array.from({ length: FLOW_BATCH }, () => Array.from({ length: FLOW_WIDTH }, () => gaussian(rand)));
  const labels = inputs.map((x) => (x.reduce((sum, value, index) => sum + value * teacher[index], 0) > 0 ? 1 : 0));
  return { inputs, labels };
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

function activate(z: number, activation: FlowActivation) {
  if (activation === "sigmoid") return sigmoid(z);
  if (activation === "tanh") return Math.tanh(z);
  return z > 0 ? z : 0;
}

/** f′(z), written in terms of the already computed activation where that is exact. */
function slope(z: number, a: number, activation: FlowActivation) {
  if (activation === "sigmoid") return a * (1 - a);
  if (activation === "tanh") return 1 - a * a;
  return z > 0 ? 1 : 0;
}

export const flowDepth = (value: number) =>
  Number.isFinite(value) ? Math.min(MAX_FLOW_DEPTH, Math.max(MIN_FLOW_DEPTH, Math.round(value))) : MIN_FLOW_DEPTH;

const EPS = 1e-12;

/** Draw the weights for `depth` hidden layers and the output unit. */
export function buildFlowNetwork(config: Pick<FlowConfig, "depth" | "init" | "seed">): FlowNetwork {
  const depth = flowDepth(config.depth);
  const rand = flowRandom((config.seed ?? FLOW_SEED) * 7919 + 17);
  const hiddenStd = initStd(config.init, FLOW_WIDTH, FLOW_WIDTH);
  const hidden = Array.from({ length: depth }, () =>
    Array.from({ length: FLOW_WIDTH }, () => Array.from({ length: FLOW_WIDTH }, () => hiddenStd * gaussian(rand))),
  );
  const outputStd = initStd(config.init, FLOW_WIDTH, 1);
  const output = Array.from({ length: FLOW_WIDTH }, () => outputStd * gaussian(rand));
  return { hidden, output };
}

/**
 * Forward the batch, then backpropagate the mean cross-entropy to every weight.
 * With a sigmoid output, ∂L/∂logit = p − y. A skip connection adds the identity path,
 * so ∂L/∂h_(l−1) = ∂L/∂h_l + Wᵀδ_l instead of Wᵀδ_l alone.
 */
export function flowGradients(
  network: FlowNetwork,
  batch: FlowBatch,
  activation: FlowActivation,
  skip: boolean,
): FlowGradients {
  const depth = network.hidden.length;
  const scale = 1 / batch.inputs.length;
  const hidden = network.hidden.map((layer) => layer.map((row) => row.map(() => 0)));
  const output = network.output.map(() => 0);
  const slopeSums = Array.from({ length: depth }, () => 0);
  const activationSquares = Array.from({ length: depth }, () => 0);
  let loss = 0;

  batch.inputs.forEach((x, example) => {
    const layerInputs: number[][] = [];
    const pre: number[][] = [];
    const post: number[][] = [];
    let h = x;
    for (let layer = 0; layer < depth; layer += 1) {
      layerInputs.push(h);
      const z = network.hidden[layer].map((row) => row.reduce((sum, weight, index) => sum + weight * h[index], 0));
      const a = z.map((value) => activate(value, activation));
      pre.push(z);
      post.push(a);
      h = skip && layer > 0 ? a.map((value, unit) => value + h[unit]) : a;
      for (let unit = 0; unit < FLOW_WIDTH; unit += 1) {
        slopeSums[layer] += slope(z[unit], a[unit], activation);
        activationSquares[layer] += h[unit] * h[unit];
      }
    }
    const logit = h.reduce((sum, value, index) => sum + value * network.output[index], 0);
    const p = sigmoid(logit);
    const label = batch.labels[example];
    loss -= label * Math.log(Math.max(EPS, p)) + (1 - label) * Math.log(Math.max(EPS, 1 - p));

    const dLogit = p - label;
    for (let unit = 0; unit < FLOW_WIDTH; unit += 1) output[unit] += scale * dLogit * h[unit];
    let upstream = network.output.map((weight) => weight * dLogit);
    for (let layer = depth - 1; layer >= 0; layer -= 1) {
      const delta = upstream.map((value, unit) => value * slope(pre[layer][unit], post[layer][unit], activation));
      for (let unit = 0; unit < FLOW_WIDTH; unit += 1) {
        for (let index = 0; index < FLOW_WIDTH; index += 1) {
          hidden[layer][unit][index] += scale * delta[unit] * layerInputs[layer][index];
        }
      }
      if (layer === 0) break;
      const through = Array.from({ length: FLOW_WIDTH }, (_, index) =>
        delta.reduce((sum, value, unit) => sum + network.hidden[layer][unit][index] * value, 0),
      );
      upstream = skip ? through.map((value, index) => value + upstream[index]) : through;
    }
  });

  return { loss: loss / batch.inputs.length, hidden, output, slopeSums, activationSquares };
}

/** Mean cross-entropy of the batch alone, used to check the gradients against finite differences. */
export function flowLoss(network: FlowNetwork, batch: FlowBatch, activation: FlowActivation, skip: boolean) {
  return flowGradients(network, batch, activation, skip).loss;
}

const frobenius = (matrix: number[][]) => Math.sqrt(matrix.reduce((sum, row) => sum + row.reduce((inner, value) => inner + value * value, 0), 0));

/** Build the weights, run the batch forward, and report each layer's gradient norm. */
export function measureGradientFlow(input: FlowConfig): FlowResult {
  const depth = flowDepth(input.depth);
  const seed = input.seed ?? FLOW_SEED;
  const network = buildFlowNetwork({ depth, init: input.init, seed });
  const result = flowGradients(network, makeFlowBatch(seed), input.activation, input.skip);
  const cells = FLOW_WIDTH * FLOW_BATCH;
  const layers: FlowLayer[] = result.hidden.map((matrix, layer) => ({
    layer: layer + 1,
    gradNorm: frobenius(matrix),
    meanSlope: result.slopeSums[layer] / cells,
    activationRms: Math.sqrt(result.activationSquares[layer] / cells),
  }));
  const first = layers[0].gradNorm;
  const last = layers[layers.length - 1].gradNorm;
  return {
    config: { depth, activation: input.activation, init: input.init, skip: input.skip, seed },
    loss: result.loss,
    layers,
    firstOverLast: last > 0 ? first / last : 1,
  };
}
