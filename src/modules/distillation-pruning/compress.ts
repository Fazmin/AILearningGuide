import { bigramPairs, pruneTinyWeights, quantizeTinyWeights, TINY_VOCAB_SIZE, tinyPerplexity } from "@app/module-sdk";

const V = TINY_VOCAB_SIZE;

/** Softmax of one context row at temperature T. */
export function rowDistribution(weights: Float32Array, row: number, temperature = 1): number[] {
  const t = Math.max(0.01, temperature);
  let maximum = -Infinity;
  for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, weights[row * V + k] / t);
  const values = Array.from({ length: V }, (_, k) => Math.exp(weights[row * V + k] / t - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

/** How often each character follows `row` in the corpus: what one-hot labels average to. */
export function labelFrequencies(text: string, row: number): number[] {
  const { inputs, targets } = bigramPairs(text);
  const counts = new Array<number>(V).fill(0);
  let total = 0;
  for (let index = 0; index < inputs.length; index += 1) {
    if (inputs[index] !== row) continue;
    counts[targets[index]] += 1;
    total += 1;
  }
  return counts.map((count) => (total ? count / total : 0));
}

export function klDivergence(p: ReadonlyArray<number>, q: ReadonlyArray<number>) {
  let total = 0;
  for (let k = 0; k < p.length; k += 1) {
    if (p[k] > 1e-12) total += p[k] * Math.log(p[k] / Math.max(1e-12, q[k]));
  }
  return total;
}

/**
 * KL(teacher ‖ student) averaged over the corpus's own positions, so each
 * context row counts as often as the data uses it. Rows the corpus never
 * visits (four of the thirty here) get no weight: the teacher never learned
 * them and the student was never trained on them.
 */
export function corpusKl(teacher: Float32Array, student: Float32Array, text: string, temperature = 1) {
  const { inputs } = bigramPairs(text);
  if (inputs.length === 0) return Number.NaN;
  const counts = new Array<number>(V).fill(0);
  for (const row of inputs) counts[row] += 1;
  let total = 0;
  for (let row = 0; row < V; row += 1) {
    if (!counts[row]) continue;
    total +=
      counts[row] * klDivergence(rowDistribution(teacher, row, temperature), rowDistribution(student, row, temperature));
  }
  return total / inputs.length;
}

/** Context rows the corpus never visits. */
export function unseenRows(text: string) {
  const { inputs } = bigramPairs(text);
  const seen = new Set<number>(Array.from(inputs));
  return Array.from({ length: V }, (_, row) => row).filter((row) => !seen.has(row));
}

export type PruneMode = "unstructured" | "semi" | "structured";

/**
 * N:M semi-structured magnitude pruning: in every group of M consecutive
 * weights along the input (context) dimension of one output column, keep the N
 * largest. A 30-row column has seven groups of four and one group of two; the
 * last keeps one, so the share is exactly (M − N) / M.
 */
export function pruneNofM(source: Float32Array, keep = 2, group = 4) {
  const weights = Float32Array.from(source);
  const mask = new Array<boolean>(source.length).fill(true);
  let removed = 0;
  for (let column = 0; column < V; column += 1) {
    for (let start = 0; start < V; start += group) {
      const rows = Array.from({ length: Math.min(group, V - start) }, (_, offset) => start + offset);
      const keepHere = Math.round((keep / group) * rows.length);
      const order = rows.sort((a, b) => Math.abs(source[a * V + column]) - Math.abs(source[b * V + column]));
      for (const row of order.slice(0, rows.length - keepHere)) {
        weights[row * V + column] = 0;
        mask[row * V + column] = false;
        removed += 1;
      }
    }
  }
  return { weights, mask, removed, removedFraction: removed / source.length };
}

/** One entry point for the three shapes, returning a boolean keep-mask alongside the weights. */
export function prune(source: Float32Array, mode: PruneMode, fraction: number) {
  if (mode === "semi") {
    return { ...pruneNofM(source), removedRows: [] as number[] };
  }
  const result = pruneTinyWeights(source, { fraction, mode });
  const zeroed = new Set<number>();
  if (mode === "structured") {
    for (const row of result.removedRows) for (let k = 0; k < V; k += 1) zeroed.add(row * V + k);
  } else {
    const order = Array.from(source, (value, index) => ({ index, magnitude: Math.abs(value) })).sort(
      (a, b) => a.magnitude - b.magnitude,
    );
    for (const entry of order.slice(0, result.removed)) zeroed.add(entry.index);
  }
  return {
    weights: result.weights,
    removed: result.removed,
    removedFraction: result.removedFraction,
    removedRows: result.removedRows,
    mask: Array.from(source, (_, index) => !zeroed.has(index)),
  };
}

/* -------------------------------------------------------------------------- */
/* Compose: distil, prune, quantize                                            */
/* -------------------------------------------------------------------------- */

/** Bits per stored value before quantization: the lab's float32 baseline, as in the Quantization lab's "Against 32-bit". */
export const STORED_BITS = 32;
/** Weights per quantization block in the compose pipeline, matching llama.cpp's Q4_0 and Q4_1 blocks. */
export const COMPOSE_GROUP = 32;

export type ImportanceScore = "magnitude" | "activation" | "ablation";

export const IMPORTANCE_SCORES: ReadonlyArray<{ id: ImportanceScore; label: string }> = [
  { id: "magnitude", label: "Magnitude" },
  { id: "activation", label: "Activation-weighted" },
  { id: "ablation", label: "Measured ablation" },
];

/** The corpus as a table of how often each next character follows each context, which is all a bigram loss needs. */
export interface CorpusCounts {
  rowCounts: number[];
  pairCounts: Float64Array;
  positions: number;
}

export function corpusCounts(text: string): CorpusCounts {
  const { inputs, targets } = bigramPairs(text);
  const rowCounts = new Array<number>(V).fill(0);
  const pairCounts = new Float64Array(V * V);
  for (let index = 0; index < inputs.length; index += 1) {
    rowCounts[inputs[index]] += 1;
    pairCounts[inputs[index] * V + targets[index]] += 1;
  }
  return { rowCounts, pairCounts, positions: inputs.length };
}

/**
 * One-sided Jacobi (Hestenes) factorization of a V x V table into U (V x rank) and W (rank x V) with U times W equal to
 * the table's best rank-limited form, split evenly so both factors carry the square root of each singular value. The
 * distilled student is exactly rank-limited, so its factors reproduce its table to float precision.
 */
export function factorizeTable(table: Float32Array, rank: number) {
  const safeRank = Math.max(1, Math.min(V, Math.round(rank)));
  const columns = Float64Array.from(table);
  const rotation = new Float64Array(V * V);
  for (let index = 0; index < V; index += 1) rotation[index * V + index] = 1;
  for (let sweep = 0; sweep < 60; sweep += 1) {
    let rotated = 0;
    for (let p = 0; p < V - 1; p += 1) {
      for (let q = p + 1; q < V; q += 1) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let row = 0; row < V; row += 1) {
          alpha += columns[row * V + p] ** 2;
          beta += columns[row * V + q] ** 2;
          gamma += columns[row * V + p] * columns[row * V + q];
        }
        if (gamma === 0 || Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta)) continue;
        rotated += 1;
        const zeta = (beta - alpha) / (2 * gamma);
        const tangent = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cosine = 1 / Math.sqrt(1 + tangent * tangent);
        const sine = cosine * tangent;
        for (let row = 0; row < V; row += 1) {
          const columnP = columns[row * V + p];
          const columnQ = columns[row * V + q];
          columns[row * V + p] = cosine * columnP - sine * columnQ;
          columns[row * V + q] = sine * columnP + cosine * columnQ;
          const rotationP = rotation[row * V + p];
          const rotationQ = rotation[row * V + q];
          rotation[row * V + p] = cosine * rotationP - sine * rotationQ;
          rotation[row * V + q] = sine * rotationP + cosine * rotationQ;
        }
      }
    }
    if (rotated === 0) break;
  }
  const singular = Array.from({ length: V }, (_, column) => {
    let sum = 0;
    for (let row = 0; row < V; row += 1) sum += columns[row * V + column] ** 2;
    return Math.sqrt(sum);
  });
  const order = singular.map((value, column) => ({ value, column })).sort((a, b) => b.value - a.value || a.column - b.column);
  const left = new Float32Array(V * safeRank);
  const right = new Float32Array(safeRank * V);
  for (let component = 0; component < safeRank; component += 1) {
    const { value, column } = order[component];
    const root = Math.sqrt(value);
    for (let index = 0; index < V; index += 1) {
      left[index * safeRank + component] = value > 1e-12 ? (columns[index * V + column] / value) * root : 0;
      right[component * V + index] = value > 1e-12 ? rotation[index * V + column] * root : 0;
    }
  }
  return { left, right, rank: safeRank, singularValues: order.map((entry) => entry.value) };
}

/** The V x V table a pair of factors computes. */
export function multiplyFactors(left: Float32Array, right: Float32Array, rank: number): Float32Array {
  const table = new Float32Array(V * V);
  for (let row = 0; row < V; row += 1) {
    for (let column = 0; column < V; column += 1) {
      let sum = 0;
      for (let component = 0; component < rank; component += 1) sum += left[row * rank + component] * right[component * V + column];
      table[row * V + column] = sum;
    }
  }
  return table;
}

/**
 * Importance of each factor entry, larger meaning more important, for the two factors of a student whose logits are
 * `U[context] · W`.
 * - magnitude: |w|.
 * - activation: |w| times the norm of the input that multiplies it over the corpus (a Wanda-style score). The input to
 *   U is a one-hot context, whose norm over the corpus is the square root of how often that context occurs; the input to
 *   W is the hidden vector U[context], whose norm for component k is sqrt(sum over contexts of count x U[context][k]^2).
 * - ablation: the measured rise in corpus cross-entropy when that one entry is set to zero, every other weight kept.
 */
export function importanceScores(
  kind: ImportanceScore,
  left: Float32Array,
  right: Float32Array,
  rank: number,
  counts: CorpusCounts,
): { left: Float64Array; right: Float64Array } {
  if (kind === "magnitude") {
    return { left: Float64Array.from(left, Math.abs), right: Float64Array.from(right, Math.abs) };
  }
  if (kind === "activation") {
    const hidden = Array.from({ length: rank }, (_, component) => {
      let sum = 0;
      for (let context = 0; context < V; context += 1) sum += counts.rowCounts[context] * left[context * rank + component] ** 2;
      return Math.sqrt(sum);
    });
    return {
      left: Float64Array.from(left, (weight, index) => Math.abs(weight) * Math.sqrt(counts.rowCounts[Math.floor(index / rank)])),
      right: Float64Array.from(right, (weight, index) => Math.abs(weight) * hidden[Math.floor(index / V)]),
    };
  }
  // Measured ablation. Per context row i the corpus loss is n_i lse(z_i) - sum_t C[i][t] z_i[t], where z_i is the row of logits.
  const logits = new Float64Array(V * V);
  for (let row = 0; row < V; row += 1) {
    for (let column = 0; column < V; column += 1) {
      let sum = 0;
      for (let component = 0; component < rank; component += 1) sum += left[row * rank + component] * right[component * V + column];
      logits[row * V + column] = sum;
    }
  }
  const logSumExp = (values: ArrayLike<number>) => {
    let maximum = -Infinity;
    for (let index = 0; index < V; index += 1) maximum = Math.max(maximum, values[index]);
    let total = 0;
    for (let index = 0; index < V; index += 1) total += Math.exp(values[index] - maximum);
    return maximum + Math.log(total);
  };
  const row = new Float64Array(V);
  const rowStats = Array.from({ length: V }, (_, context) => {
    for (let column = 0; column < V; column += 1) row[column] = logits[context * V + column];
    let linear = 0;
    for (let column = 0; column < V; column += 1) linear += counts.pairCounts[context * V + column] * logits[context * V + column];
    return { lse: logSumExp(row), linear };
  });
  const rowLoss = (context: number, lse: number, linear: number) => counts.rowCounts[context] * lse - linear;
  const positions = Math.max(1, counts.positions);

  const leftScores = new Float64Array(left.length);
  for (let context = 0; context < V; context += 1) {
    if (counts.rowCounts[context] === 0) continue; // a context the corpus never visits cannot change the loss
    for (let component = 0; component < rank; component += 1) {
      const weight = left[context * rank + component];
      let linear = 0;
      for (let column = 0; column < V; column += 1) {
        row[column] = logits[context * V + column] - weight * right[component * V + column];
        linear += counts.pairCounts[context * V + column] * row[column];
      }
      const before = rowLoss(context, rowStats[context].lse, rowStats[context].linear);
      leftScores[context * rank + component] = (rowLoss(context, logSumExp(row), linear) - before) / positions;
    }
  }
  const rightScores = new Float64Array(right.length);
  for (let component = 0; component < rank; component += 1) {
    for (let column = 0; column < V; column += 1) {
      const weight = right[component * V + column];
      let delta = 0;
      for (let context = 0; context < V; context += 1) {
        if (counts.rowCounts[context] === 0) continue;
        const before = logits[context * V + column];
        const after = before - left[context * rank + component] * weight;
        const { lse, linear } = rowStats[context];
        const lseAfter = Math.log(Math.max(1e-300, Math.exp(lse) - Math.exp(before) + Math.exp(after)));
        const linearAfter = linear + counts.pairCounts[context * V + column] * (after - before);
        delta += rowLoss(context, lseAfter, linearAfter) - rowLoss(context, lse, linear);
      }
      rightScores[component * V + column] = delta / positions;
    }
  }
  return { left: leftScores, right: rightScores };
}

/** Zero the `round(fraction x length)` entries with the lowest score, lowest first, ties by position. Returns the keep-mask. */
export function pruneByScore(values: Float32Array, scores: ArrayLike<number>, fraction: number) {
  const output = Float32Array.from(values);
  const mask = new Array<boolean>(values.length).fill(true);
  const order = Array.from({ length: values.length }, (_, index) => index).sort(
    (a, b) => scores[a] - scores[b] || a - b,
  );
  const count = Math.round(Math.min(1, Math.max(0, fraction)) * values.length);
  for (const index of order.slice(0, count)) {
    output[index] = 0;
    mask[index] = false;
  }
  return { values: output, mask, removed: count };
}

export interface ComposeStage {
  id: "teacher" | "student" | "pruned" | "quantized";
  label: string;
  /** Values stored: every table entry, every factor entry, or only the entries that survived pruning. */
  values: number;
  /** Total bits of the stage, metadata and mask included. */
  bits: number;
  bytes: number;
  perplexity: number;
  weights: Float32Array;
}

export interface ComposeResult {
  stages: ComposeStage[];
  removed: number;
  kept: number;
  blocks: number;
  /** Passes over the calibration text needed to score every entry: 0, 1, or one per factor entry. */
  scoringPasses: number;
}

export interface ComposeOptions {
  teacher: Float32Array;
  student: Float32Array;
  rank: number;
  text: string;
  importance: ImportanceScore;
  share: number;
  bits: number;
  counts?: CorpusCounts;
}

/**
 * distil -> prune -> quantize on one pipeline, with the size and perplexity after each stage.
 * Sizes: the teacher stores 900 float32 values; the student stores its two factors (2 x rank x 30 float32 values); pruning
 * keeps the highest-scored entries of each factor and adds a one-bit keep-mask per position when anything was removed;
 * quantizing then rounds the kept values, in storage order, through the SDK's group quantizer (32 weights per block, two
 * 16-bit numbers per block) and drops them to `bits` each. Pruned entries stay zero after quantizing.
 */
export function composePipeline(options: ComposeOptions): ComposeResult {
  const rank = Math.max(1, Math.min(V, Math.round(options.rank)));
  const counts = options.counts ?? corpusCounts(options.text);
  const factors = factorizeTable(options.student, rank);
  const scores = importanceScores(options.importance, factors.left, factors.right, rank, counts);
  const left = pruneByScore(factors.left, scores.left, options.share);
  const right = pruneByScore(factors.right, scores.right, options.share);
  const entries = factors.left.length + factors.right.length;
  const removed = left.removed + right.removed;
  const kept = entries - removed;
  const maskBits = removed > 0 ? entries : 0;

  const prunedWeights = multiplyFactors(left.values, right.values, rank);
  const keptValues = Float32Array.from([
    ...left.values.filter((_, index) => left.mask[index]),
    ...right.values.filter((_, index) => right.mask[index]),
  ]);
  const quantized = quantizeTinyWeights(keptValues, { bits: options.bits, scope: "group", groupSize: COMPOSE_GROUP });
  let cursor = 0;
  const restore = (values: Float32Array, mask: boolean[]) =>
    Float32Array.from(values, (_, index) => (mask[index] ? quantized.weights[cursor++] : 0));
  const quantLeft = restore(left.values, left.mask);
  const quantRight = restore(right.values, right.mask);
  const quantizedWeights = multiplyFactors(quantLeft, quantRight, rank);

  const stage = (id: ComposeStage["id"], label: string, values: number, bits: number, weights: Float32Array): ComposeStage => ({
    id,
    label,
    values,
    bits,
    bytes: Math.ceil(bits / 8),
    perplexity: tinyPerplexity(weights, options.text),
    weights,
  });
  const safeBits = Math.max(1, Math.min(16, Math.round(options.bits)));
  return {
    removed,
    kept,
    blocks: quantized.blocks,
    scoringPasses: options.importance === "magnitude" ? 0 : options.importance === "activation" ? 1 : entries,
    stages: [
      stage("teacher", "Teacher", options.teacher.length, options.teacher.length * STORED_BITS, options.teacher),
      stage("student", "Distilled student", entries, entries * STORED_BITS, options.student),
      stage("pruned", "Pruned", kept, kept * STORED_BITS + maskBits, prunedWeights),
      stage("quantized", "Quantized", kept, kept * safeBits + quantized.blocks * 32 + maskBits, quantizedWeights),
    ],
  };
}
