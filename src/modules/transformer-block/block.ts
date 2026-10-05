/**
 * A tiny, seeded, untrained transformer block computed in the browser: model
 * width 8, two causal attention heads of width 4, a GELU MLP of width 32 (4×),
 * and LayerNorm. Every number the lab shows comes from these functions.
 */
import { stableSoftmax, tinyRandom } from "@app/module-sdk";

export const WIDTH = 8;
export const HEADS = 2;
export const HEAD_WIDTH = WIDTH / HEADS;
export const HIDDEN = 4 * WIDTH;
export const LN_EPSILON = 1e-5;
export const WEIGHT_SEED = 17;

export const TOKENS = ["the", "cat", "sat", "on", "the", "mat"] as const;
export const OTHER_TOKENS = ["a", "dog", "ran", "to", "a", "rug"] as const;

export type Vector = number[];
export type Matrix = number[][];
export type Placement = "pre" | "post";

export interface BlockWeights {
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
  wo: Matrix;
  w1: Matrix;
  b1: Vector;
  w2: Matrix;
  b2: Vector;
  gamma1: Vector;
  beta1: Vector;
  gamma2: Vector;
  beta2: Vector;
}

function normalSampler(seed: number) {
  const random = tinyRandom(seed);
  return () => {
    const u = Math.max(1e-12, random());
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

function randomMatrix(rows: number, columns: number, std: number, next: () => number): Matrix {
  return Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => next() * std),
  );
}

/** Seeded Gaussian initialisation with std 1/√fan-in; norm scales start at 1, shifts at 0. */
export function createBlockWeights(seed = WEIGHT_SEED): BlockWeights {
  const next = normalSampler(seed);
  const std = 1 / Math.sqrt(WIDTH);
  return {
    wq: randomMatrix(WIDTH, WIDTH, std, next),
    wk: randomMatrix(WIDTH, WIDTH, std, next),
    wv: randomMatrix(WIDTH, WIDTH, std, next),
    wo: randomMatrix(WIDTH, WIDTH, std, next),
    w1: randomMatrix(WIDTH, HIDDEN, std, next),
    b1: Array.from({ length: HIDDEN }, () => 0),
    w2: randomMatrix(HIDDEN, WIDTH, 1 / Math.sqrt(HIDDEN), next),
    b2: Array.from({ length: WIDTH }, () => 0),
    gamma1: Array.from({ length: WIDTH }, () => 1),
    beta1: Array.from({ length: WIDTH }, () => 0),
    gamma2: Array.from({ length: WIDTH }, () => 1),
    beta2: Array.from({ length: WIDTH }, () => 0),
  };
}

function hash(text: string) {
  let value = 2166136261;
  for (const character of text) {
    value ^= character.charCodeAt(0);
    value = Math.imul(value, 16777619);
  }
  return (value >>> 0) / 4294967296;
}

/** Seeded token vector, entries ~ N(0, 1). The same token always gets the same vector. */
export function tokenVector(token: string): Vector {
  const next = normalSampler(1000 + hash(token) * 1e6);
  return Array.from({ length: WIDTH }, () => next());
}

/** Sinusoidal position vector of amplitude 0.5: a sine/cosine pair per frequency. */
export function positionVector(position: number): Vector {
  return Array.from({ length: WIDTH }, (_, dimension) => {
    const pair = Math.floor(dimension / 2);
    const angle = position / 10000 ** ((2 * pair) / WIDTH);
    return 0.5 * (dimension % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
  });
}

/** Token vector plus position vector: what the lab's block reads as its input. */
export function embed(token: string, position: number): Vector {
  const content = tokenVector(token);
  const place = positionVector(position);
  return content.map((value, dimension) => value + place[dimension]);
}

export const add = (left: Vector, right: Vector) => left.map((value, index) => value + (right[index] ?? 0));
export const scale = (vector: Vector, factor: number) => vector.map((value) => value * factor);
export const norm = (vector: Vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
export const zeros = (length = WIDTH) => Array.from({ length }, () => 0);

export function cosine(left: Vector, right: Vector) {
  const denominator = norm(left) * norm(right);
  if (denominator < 1e-12) return 0;
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0) / denominator;
}

/** x · W for a row vector x and a matrix W stored as rows = inputs. */
export function matVec(vector: Vector, matrix: Matrix): Vector {
  const columns = matrix[0]?.length ?? 0;
  return Array.from({ length: columns }, (_, column) =>
    vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0),
  );
}

export function mean(vector: Vector) {
  return vector.reduce((sum, value) => sum + value, 0) / Math.max(1, vector.length);
}

export function layerNorm(vector: Vector, gamma: Vector, beta: Vector, epsilon = LN_EPSILON) {
  const mu = mean(vector);
  const variance = mean(vector.map((value) => (value - mu) ** 2));
  const inverse = 1 / Math.sqrt(variance + epsilon);
  return vector.map((value, index) => (value - mu) * inverse * gamma[index] + beta[index]);
}

export function rmsNorm(vector: Vector, gamma: Vector, epsilon = LN_EPSILON) {
  const rms = Math.sqrt(mean(vector.map((value) => value * value)) + epsilon);
  return vector.map((value, index) => (value / rms) * gamma[index]);
}

/** GELU, tanh approximation as used by GPT-2. */
export function gelu(value: number) {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

/**
 * Multi-head self-attention over a sequence, causal by default. Returns each
 * position's write (after the output projection) and each head's weights per
 * position. With `causal` false every position reads every position, which is
 * what the Position card needs to isolate order from the mask.
 */
export function selfAttention(inputs: ReadonlyArray<Vector>, weights: BlockWeights, causal = true) {
  const queries = inputs.map((x) => matVec(x, weights.wq));
  const keys = inputs.map((x) => matVec(x, weights.wk));
  const values = inputs.map((x) => matVec(x, weights.wv));
  const headWeights: number[][][] = inputs.map(() => []);
  const outputs = inputs.map((_, position) => {
    const concatenated: Vector = [];
    for (let head = 0; head < HEADS; head += 1) {
      const from = head * HEAD_WIDTH;
      const slice = (vector: Vector) => vector.slice(from, from + HEAD_WIDTH);
      const q = slice(queries[position]);
      const scores = keys
        .slice(0, causal ? position + 1 : keys.length)
        .map((key) => slice(key).reduce((sum, value, index) => sum + value * q[index], 0) / Math.sqrt(HEAD_WIDTH));
      const probabilities = stableSoftmax(scores);
      headWeights[position][head] = probabilities;
      for (let dimension = 0; dimension < HEAD_WIDTH; dimension += 1) {
        concatenated.push(
          probabilities.reduce((sum, weight, index) => sum + weight * values[index][from + dimension], 0),
        );
      }
    }
    return matVec(concatenated, weights.wo);
  });
  return { outputs, headWeights };
}

export function mlp(vector: Vector, weights: BlockWeights) {
  const preActivation = add(matVec(vector, weights.w1), weights.b1);
  const hidden = preActivation.map(gelu);
  return { hidden, output: add(matVec(hidden, weights.w2), weights.b2) };
}

export interface BlockOptions {
  norm: boolean;
  attention: boolean;
  mlp: boolean;
  placement: Placement;
  /** Causal mask on (the default) or every position reads every position. */
  causal?: boolean;
}

export interface BlockTrace {
  input: Vector[];
  attentionIn: Vector[];
  attentionWrite: Vector[];
  afterAttention: Vector[];
  mlpIn: Vector[];
  mlpHidden: Vector[];
  mlpWrite: Vector[];
  output: Vector[];
  headWeights: number[][][];
}

/**
 * One block, either pre-norm (x + Attn(LN(x)), then h + MLP(LN(h))) or post-norm
 * (LN(x + Attn(x)), then LN(h + MLP(h))). A disabled sublayer writes zeros; a
 * disabled norm is the identity.
 */
export function runBlock(input: ReadonlyArray<Vector>, weights: BlockWeights, options: BlockOptions): BlockTrace {
  const causal = options.causal !== false;
  const ln1 = (x: Vector) => (options.norm ? layerNorm(x, weights.gamma1, weights.beta1) : x);
  const ln2 = (x: Vector) => (options.norm ? layerNorm(x, weights.gamma2, weights.beta2) : x);
  const x = input.map((vector) => [...vector]);

  if (options.placement === "pre") {
    const attentionIn = x.map(ln1);
    const attention = selfAttention(attentionIn, weights, causal);
    const attentionWrite = options.attention ? attention.outputs : x.map(() => zeros());
    const afterAttention = x.map((vector, index) => add(vector, attentionWrite[index]));
    const mlpIn = afterAttention.map(ln2);
    const mlpRuns = mlpIn.map((vector) => mlp(vector, weights));
    const mlpWrite = options.mlp ? mlpRuns.map((run) => run.output) : x.map(() => zeros());
    const output = afterAttention.map((vector, index) => add(vector, mlpWrite[index]));
    return {
      input: x,
      attentionIn,
      attentionWrite,
      afterAttention,
      mlpIn,
      mlpHidden: mlpRuns.map((run) => run.hidden),
      mlpWrite,
      output,
      headWeights: attention.headWeights,
    };
  }

  const attention = selfAttention(x, weights, causal);
  const attentionWrite = options.attention ? attention.outputs : x.map(() => zeros());
  const afterAttention = x.map((vector, index) => ln1(add(vector, attentionWrite[index])));
  const mlpRuns = afterAttention.map((vector) => mlp(vector, weights));
  const mlpWrite = options.mlp ? mlpRuns.map((run) => run.output) : x.map(() => zeros());
  const output = afterAttention.map((vector, index) => ln2(add(vector, mlpWrite[index])));
  return {
    input: x,
    attentionIn: x,
    attentionWrite,
    afterAttention,
    mlpIn: afterAttention,
    mlpHidden: mlpRuns.map((run) => run.hidden),
    mlpWrite,
    output,
    headWeights: attention.headWeights,
  };
}

export function sequenceFor(tokens: ReadonlyArray<string>, inputScale: number) {
  return tokens.map((token, position) => scale(embed(token, position), inputScale));
}

/**
 * How far the output at `position` moves when every other position's token is
 * replaced. Only attention reads other positions, so this is zero whenever
 * attention is off, and for the first position under the causal mask.
 */
export function crossPositionShift(
  weights: BlockWeights,
  options: BlockOptions,
  inputScale: number,
  position: number,
) {
  const original = runBlock(sequenceFor(TOKENS, inputScale), weights, options).output[position];
  const swappedTokens = TOKENS.map((token, index) => (index === position ? token : OTHER_TOKENS[index]));
  const swapped = runBlock(sequenceFor(swappedTokens, inputScale), weights, options).output[position];
  return norm(original.map((value, index) => value - swapped[index]));
}

/* Position test: reorder the same six words and see whether the block notices. */

export const ORDER_IDS = ["original", "swap", "shuffle"] as const;
export type OrderId = (typeof ORDER_IDS)[number];

/** `ORDERS[id][slot]` is the index in TOKENS of the word placed at that slot. */
export const ORDERS: Record<OrderId, readonly number[]> = {
  original: [0, 1, 2, 3, 4, 5],
  swap: [0, 5, 2, 3, 4, 1],
  shuffle: [5, 3, 1, 4, 0, 2],
};

/** Input vectors for a word list: with position information, the lab's own `embed`; without, the token vector alone. */
export function sequenceWithPosition(tokens: ReadonlyArray<string>, inputScale: number, usePosition: boolean) {
  return tokens.map((token, position) =>
    scale(usePosition ? embed(token, position) : tokenVector(token), inputScale),
  );
}

export interface OrderTestRow {
  /** The word, and where it sat in the original sentence. */
  word: string;
  from: number;
  /** Where the same word sits in the reordered sentence. */
  to: number;
  /** Distance between its block output in the two orders. */
  shift: number;
}

export interface OrderTest {
  original: string[];
  reordered: string[];
  rows: OrderTestRow[];
  largestShift: number;
  /** Words whose output moved by more than `ORDER_TOLERANCE`. */
  changed: number;
}

/** A shift below this is floating-point noise from summing the same terms in a different order. */
export const ORDER_TOLERANCE = 1e-9;

/**
 * Run the block on the original sentence and on a reordering of it, with every
 * position allowed to read every other (no causal mask), and compare each word's
 * output with itself across the two runs. Without position information the
 * block's outputs are the same six vectors in a different order, so every shift
 * is zero; with it, a word's output depends on where it sits.
 */
export function orderTest(
  weights: BlockWeights,
  options: BlockOptions,
  inputScale: number,
  order: OrderId,
  usePosition: boolean,
): OrderTest {
  const permutation = ORDERS[order];
  const original = [...TOKENS];
  const reordered = permutation.map((index) => TOKENS[index]);
  const run = (tokens: ReadonlyArray<string>) =>
    runBlock(sequenceWithPosition(tokens, inputScale, usePosition), weights, { ...options, causal: false }).output;
  const before = run(original);
  const after = run(reordered);
  const rows = original.map((word, from) => {
    const to = permutation.indexOf(from);
    return {
      word,
      from,
      to,
      shift: norm(before[from].map((value, index) => value - after[to][index])),
    };
  });
  return {
    original,
    reordered,
    rows,
    largestShift: Math.max(...rows.map((row) => row.shift)),
    changed: rows.filter((row) => row.shift > ORDER_TOLERANCE).length,
  };
}

/* Parameter accounting for one block (weights only unless biases are requested). */

export type MlpKind = "gelu" | "swiglu";
export type NormKind = "layernorm" | "rmsnorm";

/** SwiGLU hidden width as in Llama: 8d/3 rounded up to a multiple of 256. */
export function swigluHidden(width: number, multipleOf = 256) {
  return multipleOf * Math.ceil(Math.floor((8 * width) / 3) / multipleOf);
}

export function blockParameters(width: number, mlpKind: MlpKind, normKind: NormKind, biases = false) {
  const attention = 4 * width * width + (biases ? 4 * width : 0);
  const hidden = mlpKind === "gelu" ? 4 * width : swigluHidden(width);
  const mlpParameters =
    mlpKind === "gelu"
      ? 2 * width * hidden + (biases ? hidden + width : 0)
      : 3 * width * hidden;
  const norms = 2 * (normKind === "layernorm" ? 2 * width : width);
  return {
    attention,
    mlp: mlpParameters,
    norms,
    hidden,
    total: attention + mlpParameters + norms,
    twelveDSquared: 12 * width * width,
  };
}
