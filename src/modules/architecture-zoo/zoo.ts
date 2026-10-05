/**
 * Small, exact computations behind the architecture zoo. Every number the lab
 * shows is produced here from the drawn digit and the controls.
 */

export const SIZE = 8;

export const FILTERS = [
  { id: "edge", label: "vertical edge", kernel: [-1, 0, 1, -1, 0, 1, -1, 0, 1] },
  { id: "horiz", label: "horizontal edge", kernel: [-1, -1, -1, 0, 0, 0, 1, 1, 1] },
  { id: "center", label: "center blob", kernel: [0, -1, 0, -1, 4, -1, 0, -1, 0] },
] as const;

export type FilterId = (typeof FILTERS)[number]["id"];

export const DEFAULT_PIXELS = [
  0, 0, 1, 1, 1, 1, 0, 0,
  0, 1, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 0, 1, 0,
  0, 0, 0, 0, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 1, 0,
  0, 1, 0, 0, 0, 0, 1, 0,
  0, 0, 1, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
].join("");

export const parsePixels = (value: string) =>
  value
    .padEnd(SIZE * SIZE, "0")
    .slice(0, SIZE * SIZE)
    .split("")
    .map((bit) => (bit === "1" ? 1 : 0));

/** The 3×3 neighbourhood of (row, column), zero outside the pad ("same" padding). */
export function windowAt(pixels: ReadonlyArray<number>, row: number, column: number) {
  const values: number[] = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const y = row + dy;
      const x = column + dx;
      values.push(y < 0 || y >= SIZE || x < 0 || x >= SIZE ? 0 : pixels[y * SIZE + x] ?? 0);
    }
  }
  return values;
}

/**
 * One 3×3 filter slid over the pad with zero padding. Like every deep-learning
 * library this is cross-correlation: the kernel is not flipped.
 */
export function convolve(pixels: ReadonlyArray<number>, kernel: ReadonlyArray<number>) {
  return Array.from({ length: SIZE }, (_, row) =>
    Array.from({ length: SIZE }, (_, column) =>
      windowAt(pixels, row, column).reduce((sum, value, index) => sum + value * kernel[index], 0),
    ),
  );
}

/** Fraction of inked pixels in each column: the eight inputs the RNN reads left to right. */
export const columnMeans = (pixels: ReadonlyArray<number>) =>
  Array.from({ length: SIZE }, (_, column) => {
    let sum = 0;
    for (let row = 0; row < SIZE; row += 1) sum += pixels[row * SIZE + column] ?? 0;
    return sum / SIZE;
  });

/**
 * A scalar recurrent net h_t = tanh(w·h_{t−1} + x_t), h_{−1} = 0.
 * jacobian[t] = ∂h_t/∂h_{t−1} = w·(1 − h_t²).
 * influence[t] = ∂h_T/∂x_t = (1 − h_t²) · Π_{k=t+1..T} jacobian[k], T = last step.
 */
export function rnnForward(inputs: ReadonlyArray<number>, recurrentWeight: number) {
  const hidden: number[] = [];
  let previous = 0;
  for (const input of inputs) {
    previous = Math.tanh(recurrentWeight * previous + input);
    hidden.push(previous);
  }
  const jacobian = hidden.map((value) => recurrentWeight * (1 - value * value));
  const influence = hidden.map((value, step) => {
    let product = 1 - value * value;
    for (let later = step + 1; later < hidden.length; later += 1) product *= jacobian[later];
    return product;
  });
  return { hidden, jacobian, influence };
}

export const WIRINGS = [
  { id: "mlp", label: "Dense MLP" },
  { id: "ffn", label: "Per-position MLP" },
  { id: "conv", label: "Convolution" },
  { id: "rnn", label: "Recurrent" },
  { id: "attn", label: "Attention" },
  { id: "ssm", label: "State-space" },
] as const;

export type WiringId = (typeof WIRINGS)[number]["id"];

/**
 * Which input positions each output position depends on after `layers` layers,
 * as the number of layer or step hops on the shortest path (null = cannot see).
 * Rows are outputs t, columns are inputs s. The convolution is centred with
 * kernel 3 (radius 1); attention, recurrence and the SSM are causal.
 */
export function dependencyMatrix(wiring: WiringId, positions: number, layers: number) {
  return Array.from({ length: positions }, (_, t) =>
    Array.from({ length: positions }, (_, s): number | null => {
      const distance = Math.abs(t - s);
      switch (wiring) {
        case "mlp":
          return 1;
        case "ffn":
          return t === s ? 1 : null;
        case "conv":
          return distance <= layers ? Math.max(1, distance) : null;
        case "attn":
          return s <= t ? 1 : null;
        case "rnn":
        case "ssm":
          return s <= t ? t - s + 1 : null;
      }
    }),
  );
}

export interface WiringStats {
  seesFuture: boolean;
  weights: number;
  weightsFormula: string;
  sequential: string;
  longestPath: number;
  /** Numbers that must be kept to extend the sequence by one position. */
  carried: (positions: number) => number;
  carriedFormula: string;
}

/** Per-layer figures at width d, kernel k, SSM state N, over n positions. */
export function wiringStats(wiring: WiringId, positions: number, width: number, kernel = 3, state = 16): WiringStats {
  const radius = (kernel - 1) / 2;
  switch (wiring) {
    case "mlp":
      return {
        seesFuture: true,
        weights: (positions * width) ** 2,
        weightsFormula: "(n·d)²",
        sequential: "1",
        longestPath: 1,
        carried: (n) => n * width,
        carriedFormula: "whole window, n·d",
      };
    case "ffn":
      return {
        seesFuture: false,
        weights: 8 * width * width,
        weightsFormula: "2·d·4d",
        sequential: "1",
        longestPath: Number.POSITIVE_INFINITY,
        carried: () => 0,
        carriedFormula: "nothing",
      };
    case "conv":
      return {
        seesFuture: true,
        weights: kernel * width * width,
        weightsFormula: "k·d²",
        sequential: "1",
        longestPath: Math.ceil((positions - 1) / radius),
        carried: () => (kernel - 1) * width,
        carriedFormula: "(k−1)·d per layer",
      };
    case "rnn":
      return {
        seesFuture: false,
        weights: 2 * width * width,
        weightsFormula: "2·d²",
        sequential: `${positions}`,
        longestPath: positions,
        carried: () => width,
        carriedFormula: "d",
      };
    case "attn":
      return {
        seesFuture: false,
        weights: 4 * width * width,
        weightsFormula: "4·d²",
        sequential: "1",
        longestPath: 1,
        carried: (n) => 2 * n * width,
        carriedFormula: "2·n·d (KV cache)",
      };
    case "ssm":
      return {
        seesFuture: false,
        weights: 3 * width * state,
        weightsFormula: "3·d·N",
        sequential: `${positions} as a loop, ${Math.ceil(Math.log2(positions))} as a scan`,
        longestPath: positions,
        carried: () => width * state,
        carriedFormula: "d·N",
      };
  }
}

/** Deterministic pseudo-random numbers in [−1, 1) for the untrained router. */
export function seededWeights(count: number, seed: number) {
  let value = seed >>> 0;
  return Array.from({ length: count }, () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return (((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  });
}

export const MAX_EXPERTS = 16;
const ROUTER_SEED = 35;

/**
 * Top-k routing of each token (one pad column, eight 0/1 values) to experts.
 * logits = x·W + b with a fixed seeded W (8 × 16) and b; the first `experts`
 * columns are used. Gates are a softmax over the k chosen logits, as in Mixtral.
 */
export function routeTokens(tokens: ReadonlyArray<ReadonlyArray<number>>, experts: number, perToken: number) {
  const weights = seededWeights(SIZE * MAX_EXPERTS, ROUTER_SEED);
  const bias = seededWeights(MAX_EXPERTS, ROUTER_SEED + 1).map((value) => value * 0.5);
  const k = Math.max(1, Math.min(perToken, experts));
  return tokens.map((token) => {
    const logits = Array.from({ length: experts }, (_, expert) =>
      token.reduce((sum, value, index) => sum + value * weights[index * MAX_EXPERTS + expert], bias[expert]),
    );
    const chosen = logits
      .map((logit, expert) => ({ logit, expert }))
      .sort((a, b) => b.logit - a.logit || a.expert - b.expert)
      .slice(0, k);
    const top = chosen[0].logit;
    const exps = chosen.map((item) => Math.exp(item.logit - top));
    const total = exps.reduce((sum, value) => sum + value, 0);
    const gates = new Array<number>(experts).fill(0);
    chosen.forEach((item, index) => {
      gates[item.expert] = exps[index] / total;
    });
    return { logits, gates, chosen: chosen.map((item) => item.expert) };
  });
}

export const columnVectors = (pixels: ReadonlyArray<number>) =>
  Array.from({ length: SIZE }, (_, column) =>
    Array.from({ length: SIZE }, (_, row) => pixels[row * SIZE + column] ?? 0),
  );

/** Feed-forward expert d → d_ff → d has 2·d·d_ff weights; only k of E run per token. */
export function moeParameters(width: number, hidden: number, experts: number, perToken: number) {
  const perExpert = 2 * width * hidden;
  return {
    perExpert,
    total: experts * perExpert,
    active: Math.min(perToken, experts) * perExpert,
    router: width * experts,
  };
}
