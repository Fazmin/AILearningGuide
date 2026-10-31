/**
 * A genuinely trained character bigram language model, small enough to fit in a
 * synchronous render and honest enough to teach from. Every number a lab shows
 * comes from these functions: real cross-entropy, real gradients, real
 * quantization error. Nothing here simulates a result it did not compute.
 *
 * The model is a single V x V logit table: row `x` holds the logits for the
 * token that follows `x`. That is the smallest object which still has a loss
 * surface, a checkpoint, a low-rank update, and a quantization budget.
 */

/**
 * Angle brackets are in the vocabulary but never appear in any corpus, so the
 * chat-template lab can use them the way a real tokenizer uses reserved ids:
 * they only ever mean "control marker" because user text cannot contain them.
 */
export const TINY_VOCAB = " abcdefghijklmnopqrstuvwxyz.<>";
export const TINY_VOCAB_SIZE = TINY_VOCAB.length;
export const TINY_RESERVED = "<>";
export const UNIFORM_CROSS_ENTROPY = Math.log(TINY_VOCAB_SIZE);

const V = TINY_VOCAB_SIZE;

export interface TinyCorpus {
  id: string;
  name: string;
  detail: string;
  text: string;
}

export const TINY_CORPORA: Record<string, TinyCorpus> = {
  harbor: {
    id: "harbor",
    name: "Harbor weather notes",
    detail: "Repetitive descriptive prose. The easiest corpus to fit.",
    text: `the morning fog settles over the harbor and the air feels cool.
      by noon the clouds break apart and the sun warms the stone streets.
      a light wind moves in from the sea and the temperature drops again.
      rain arrives before evening and the gutters fill with water.
      the storm passes and the sky clears over the quiet town.
      in winter the same harbor freezes and the boats stay in port.
      in summer the harbor fills with sails and the water stays warm.
      the fog returns in the morning and the pattern repeats.`,
  },
  recipes: {
    id: "recipes",
    name: "Recipe steps",
    detail: "Imperative instructions. A distinct style to adapt toward.",
    text: `warm the pan over medium heat and add a spoon of oil.
      stir the onions until they turn soft and sweet.
      add the garlic and cook for one minute more.
      pour in the tomatoes and season with salt.
      simmer the sauce until it thickens and tastes round.
      boil the pasta in salted water until it is tender.
      drain the pasta and toss it with the warm sauce.
      finish the dish with fresh basil and a little cheese.`,
  },
  proverbs: {
    id: "proverbs",
    name: "Proverbs",
    detail: "Short unrelated sentences. Hard to fit, useful for evaluation.",
    text: `a stitch in time saves nine.
      the early bird gets the worm.
      still water runs deep.
      what goes around comes around.
      a watched pot never boils.
      actions speak louder than words.
      better late than never.
      every cloud has a silver lining.
      practice makes perfect.
      the pen is mightier than the sword.`,
  },
};

/** Maps text into the toy vocabulary. Unknown characters become one space. */
export function encodeTinyText(text: string): number[] {
  const ids: number[] = [];
  for (const character of text.toLowerCase()) {
    const index = TINY_VOCAB.indexOf(character);
    const id = index >= 0 ? index : 0;
    if (id === 0 && ids[ids.length - 1] === 0) continue;
    ids.push(id);
  }
  while (ids[0] === 0) ids.shift();
  return ids;
}

export function decodeTinyIds(ids: ReadonlyArray<number>): string {
  return ids.map((id) => TINY_VOCAB[id] ?? " ").join("");
}

/** Deterministic PRNG so every lab result is reproducible from its state. */
export function tinyRandom(seed: number) {
  let state = (Math.floor(Math.abs(seed) * 2654435761) + 0x9e3779b9) >>> 0 || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function createTinyWeights(): Float32Array {
  return new Float32Array(V * V);
}

function rowSoftmax(
  weights: Float32Array,
  row: number,
  into: Float32Array,
  temperature = 1,
) {
  const offset = row * V;
  const safeTemperature = Math.max(0.01, temperature);
  let maximum = -Infinity;
  for (let k = 0; k < V; k += 1) {
    const value = weights[offset + k] / safeTemperature;
    if (value > maximum) maximum = value;
  }
  let total = 0;
  for (let k = 0; k < V; k += 1) {
    const value = Math.exp(weights[offset + k] / safeTemperature - maximum);
    into[k] = value;
    total += value;
  }
  for (let k = 0; k < V; k += 1) into[k] /= total;
  return into;
}

export type TinySchedule = "constant" | "cosine" | "warmup-decay" | "one-cycle";

export const TINY_SCHEDULES: ReadonlyArray<{
  value: TinySchedule;
  label: string;
  detail: string;
}> = [
  {
    value: "constant",
    label: "Constant",
    detail: "The same step size from the first batch to the last.",
  },
  {
    value: "cosine",
    label: "Cosine decay",
    detail: "Starts at the peak and eases toward a small final step.",
  },
  {
    value: "warmup-decay",
    label: "Warmup + decay",
    detail: "Ramps up from near zero, then decays linearly.",
  },
  {
    value: "one-cycle",
    label: "One cycle",
    detail: "Warms up to the peak, then cosines down below the start.",
  },
];

/** Multiplier applied to the peak learning rate at a point in the run. */
export function scheduleFactor(
  schedule: TinySchedule,
  progress: number,
  warmupFraction = 0.15,
): number {
  const point = Math.min(1, Math.max(0, progress));
  const warmup = Math.min(0.9, Math.max(0.01, warmupFraction));
  switch (schedule) {
    case "constant":
      return 1;
    case "cosine":
      return 0.03 + 0.97 * (0.5 * (1 + Math.cos(Math.PI * point)));
    case "warmup-decay":
      return point < warmup
        ? Math.max(0.02, point / warmup)
        : 1 - 0.95 * ((point - warmup) / (1 - warmup));
    case "one-cycle": {
      if (point < warmup) return 0.1 + 0.9 * (point / warmup);
      const after = (point - warmup) / (1 - warmup);
      return 0.02 + 0.98 * (0.5 * (1 + Math.cos(Math.PI * after)));
    }
  }
}

export interface TinyTrainOptions {
  text: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  schedule?: TinySchedule;
  warmupFraction?: number;
  weightDecay?: number;
  /** Rescales any batch gradient whose norm exceeds this value. 0 disables it. */
  clipNorm?: number;
  seed?: number;
  /** Starting weights. Omit to train from a zeroed table. */
  init?: Float32Array;
  /** Rows the optimizer may update. Frozen rows still contribute loss. */
  trainableRows?: ReadonlyArray<boolean>;
  checkpoints?: number;
  historyPoints?: number;
}

export interface TinyHistoryPoint {
  step: number;
  progress: number;
  loss: number;
  learningRate: number;
  gradientNorm: number;
}

export interface TinyCheckpoint {
  step: number;
  progress: number;
  loss: number;
  weights: Float32Array;
}

export interface TinyTrainRun {
  weights: Float32Array;
  history: TinyHistoryPoint[];
  checkpoints: TinyCheckpoint[];
  steps: number;
  tokens: number;
  updatedParameters: number;
  finalLoss: number;
  diverged: boolean;
  peakGradientNorm: number;
}

interface BigramPairs {
  inputs: Int32Array;
  targets: Int32Array;
}

export function bigramPairs(text: string): BigramPairs {
  const ids = encodeTinyText(text);
  const count = Math.max(0, ids.length - 1);
  const inputs = new Int32Array(count);
  const targets = new Int32Array(count);
  for (let index = 0; index < count; index += 1) {
    inputs[index] = ids[index];
    targets[index] = ids[index + 1];
  }
  return { inputs, targets };
}

function shuffledOrder(count: number, random: () => number): Int32Array {
  const order = new Int32Array(count);
  for (let index = 0; index < count; index += 1) order[index] = index;
  for (let index = count - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const held = order[index];
    order[index] = order[swap];
    order[swap] = held;
  }
  return order;
}

/**
 * One full training run: shuffled minibatch SGD on softmax cross-entropy with
 * an optional schedule, weight decay, gradient clipping, and frozen rows.
 */
export function trainTinyModel(options: TinyTrainOptions): TinyTrainRun {
  const {
    text,
    epochs = 8,
    batchSize = 16,
    learningRate = 0.6,
    schedule = "constant",
    warmupFraction = 0.15,
    weightDecay = 0,
    clipNorm = 0,
    seed = 7,
    init,
    trainableRows,
    checkpoints = 6,
    historyPoints = 56,
  } = options;

  const weights = init ? Float32Array.from(init) : createTinyWeights();
  const { inputs, targets } = bigramPairs(text);
  const pairCount = inputs.length;
  const safeBatch = Math.max(1, Math.min(batchSize, Math.max(1, pairCount)));
  const batchesPerEpoch = pairCount === 0 ? 0 : Math.ceil(pairCount / safeBatch);
  const totalSteps = batchesPerEpoch * Math.max(0, epochs);
  const history: TinyHistoryPoint[] = [];
  const recorded: TinyCheckpoint[] = [];

  if (totalSteps === 0) {
    return {
      weights,
      history,
      checkpoints: [
        { step: 0, progress: 0, loss: UNIFORM_CROSS_ENTROPY, weights: Float32Array.from(weights) },
      ],
      steps: 0,
      tokens: pairCount,
      updatedParameters: 0,
      finalLoss: tinyCrossEntropy(weights, text),
      diverged: false,
      peakGradientNorm: 0,
    };
  }

  const random = tinyRandom(seed);
  const gradient = new Float32Array(V * V);
  const probabilities = new Float32Array(V);
  const touched = new Set<number>();
  const updatedRows = new Set<number>();
  const historyEvery = Math.max(1, Math.floor(totalSteps / historyPoints));
  const checkpointEvery = Math.max(1, Math.floor(totalSteps / Math.max(1, checkpoints - 1)));
  let step = 0;
  let windowLoss = 0;
  let windowSteps = 0;
  let peakGradientNorm = 0;
  let diverged = false;

  recorded.push({
    step: 0,
    progress: 0,
    loss: tinyCrossEntropy(weights, text),
    weights: Float32Array.from(weights),
  });

  for (let epoch = 0; epoch < epochs && !diverged; epoch += 1) {
    const order = shuffledOrder(pairCount, random);
    for (let batch = 0; batch < batchesPerEpoch && !diverged; batch += 1) {
      const start = batch * safeBatch;
      const end = Math.min(pairCount, start + safeBatch);
      const size = end - start;
      const inverseSize = 1 / size;
      let batchLoss = 0;
      touched.clear();

      for (let index = start; index < end; index += 1) {
        const pair = order[index];
        const row = inputs[pair];
        const target = targets[pair];
        rowSoftmax(weights, row, probabilities);
        batchLoss -= Math.log(Math.max(1e-12, probabilities[target]));
        const offset = row * V;
        for (let k = 0; k < V; k += 1) {
          gradient[offset + k] +=
            (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
        }
        touched.add(row);
      }

      batchLoss *= inverseSize;
      let squaredNorm = 0;
      for (const row of touched) {
        const offset = row * V;
        for (let k = 0; k < V; k += 1) squaredNorm += gradient[offset + k] ** 2;
      }
      const gradientNorm = Math.sqrt(squaredNorm);
      peakGradientNorm = Math.max(peakGradientNorm, gradientNorm);
      const clipScale =
        clipNorm > 0 && gradientNorm > clipNorm ? clipNorm / gradientNorm : 1;
      const progress = totalSteps <= 1 ? 1 : step / (totalSteps - 1);
      const stepRate =
        learningRate * scheduleFactor(schedule, progress, warmupFraction);

      for (const row of touched) {
        const offset = row * V;
        const frozen = trainableRows ? trainableRows[row] === false : false;
        for (let k = 0; k < V; k += 1) {
          if (!frozen) {
            const decay = weightDecay * weights[offset + k];
            weights[offset + k] -=
              stepRate * (gradient[offset + k] * clipScale + decay);
          }
          gradient[offset + k] = 0;
        }
        if (!frozen) updatedRows.add(row);
      }

      if (!Number.isFinite(batchLoss) || !Number.isFinite(weights[0])) {
        diverged = true;
      }

      windowLoss += Number.isFinite(batchLoss) ? batchLoss : UNIFORM_CROSS_ENTROPY * 4;
      windowSteps += 1;
      step += 1;

      if (step % historyEvery === 0 || step === totalSteps || diverged) {
        history.push({
          step,
          progress,
          loss: windowLoss / Math.max(1, windowSteps),
          learningRate: stepRate,
          gradientNorm,
        });
        windowLoss = 0;
        windowSteps = 0;
      }
      if (step % checkpointEvery === 0 || step === totalSteps) {
        recorded.push({
          step,
          progress,
          loss: diverged ? Number.NaN : tinyCrossEntropy(weights, text),
          weights: Float32Array.from(weights),
        });
      }
    }
  }

  return {
    weights,
    history,
    checkpoints: recorded,
    steps: step,
    tokens: pairCount,
    updatedParameters: updatedRows.size * V,
    finalLoss: diverged ? Number.NaN : tinyCrossEntropy(weights, text),
    diverged,
    peakGradientNorm,
  };
}

/** Mean cross-entropy in nats per token. Lower is a better fit. */
export function tinyCrossEntropy(weights: Float32Array, text: string): number {
  const { inputs, targets } = bigramPairs(text);
  if (inputs.length === 0) return UNIFORM_CROSS_ENTROPY;
  const probabilities = new Float32Array(V);
  let total = 0;
  for (let index = 0; index < inputs.length; index += 1) {
    rowSoftmax(weights, inputs[index], probabilities);
    total -= Math.log(Math.max(1e-12, probabilities[targets[index]]));
  }
  return total / inputs.length;
}

export function tinyPerplexity(weights: Float32Array, text: string): number {
  return Math.exp(tinyCrossEntropy(weights, text));
}

export function tinyNextDistribution(
  weights: Float32Array,
  context: string,
  temperature = 1,
): number[] {
  const ids = encodeTinyText(context);
  const row = ids.length ? ids[ids.length - 1] : 0;
  const probabilities = rowSoftmax(weights, row, new Float32Array(V), temperature);
  return Array.from(probabilities);
}

export function tinyTopTokens(
  weights: Float32Array,
  context: string,
  count = 5,
  temperature = 1,
): { token: string; label: string; probability: number }[] {
  return tinyNextDistribution(weights, context, temperature)
    .map((probability, id) => ({
      token: TINY_VOCAB[id],
      label: TINY_VOCAB[id] === " " ? "␣" : TINY_VOCAB[id],
      probability,
    }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, count);
}

/**
 * Samples one character at a time. Reserved control symbols are suppressed and
 * the distribution renormalized, which is what a real decoder does with special
 * tokens unless it is explicitly asked to emit them.
 */
export function sampleTinyText(
  weights: Float32Array,
  {
    prompt = "the ",
    length = 64,
    temperature = 0.8,
    seed = 11,
    allowReserved = false,
  }: {
    prompt?: string;
    length?: number;
    temperature?: number;
    seed?: number;
    allowReserved?: boolean;
  } = {},
): string {
  const random = tinyRandom(seed);
  const ids = encodeTinyText(prompt);
  const generated: number[] = [];
  let row = ids.length ? ids[ids.length - 1] : 0;
  const probabilities = new Float32Array(V);
  const suppressed = allowReserved
    ? []
    : Array.from(TINY_RESERVED, (character) => TINY_VOCAB.indexOf(character));
  for (let index = 0; index < length; index += 1) {
    rowSoftmax(weights, row, probabilities, temperature);
    if (suppressed.length > 0) {
      for (const id of suppressed) probabilities[id] = 0;
      let remaining = 0;
      for (let k = 0; k < V; k += 1) remaining += probabilities[k];
      for (let k = 0; k < V; k += 1) probabilities[k] /= Math.max(1e-12, remaining);
    }
    let draw = random();
    let chosen = V - 1;
    for (let k = 0; k < V; k += 1) {
      draw -= probabilities[k];
      if (draw <= 0) {
        chosen = k;
        break;
      }
    }
    generated.push(chosen);
    row = chosen;
  }
  return decodeTinyIds(generated);
}

/** Average KL(left ‖ right) over every context row. A drift measurement. */
export function tinyRowDivergence(
  left: Float32Array,
  right: Float32Array,
): number {
  const leftRow = new Float32Array(V);
  const rightRow = new Float32Array(V);
  let total = 0;
  for (let row = 0; row < V; row += 1) {
    rowSoftmax(left, row, leftRow);
    rowSoftmax(right, row, rightRow);
    for (let k = 0; k < V; k += 1) {
      if (leftRow[k] > 1e-9) {
        total += leftRow[k] * Math.log(leftRow[k] / Math.max(1e-12, rightRow[k]));
      }
    }
  }
  return total / V;
}

/* -------------------------------------------------------------------------- */
/* Parameter-efficient adaptation                                              */
/* -------------------------------------------------------------------------- */

export interface TinyLoraOptions {
  base: Float32Array;
  text: string;
  rank: number;
  alpha?: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  seed?: number;
  historyPoints?: number;
}

export interface TinyLoraRun {
  /** V x rank. Zero-initialized, so an untrained adapter changes nothing. */
  up: Float32Array;
  /** rank x V. Randomly initialized. */
  down: Float32Array;
  delta: Float32Array;
  merged: Float32Array;
  scale: number;
  history: TinyHistoryPoint[];
  trainableParameters: number;
  fullParameters: number;
  finalLoss: number;
}

function loraDelta(
  up: Float32Array,
  down: Float32Array,
  rank: number,
  scale: number,
): Float32Array {
  const delta = new Float32Array(V * V);
  for (let row = 0; row < V; row += 1) {
    for (let j = 0; j < rank; j += 1) {
      const factor = up[row * rank + j] * scale;
      if (factor === 0) continue;
      for (let k = 0; k < V; k += 1) {
        delta[row * V + k] += factor * down[j * V + k];
      }
    }
  }
  return delta;
}

/**
 * Trains only a rank-r pair of factors. The base table is read but never
 * written, which is exactly why the adapter can be swapped or merged later.
 */
export function trainTinyLora(options: TinyLoraOptions): TinyLoraRun {
  const {
    base,
    text,
    rank,
    alpha = rank,
    epochs = 10,
    batchSize = 16,
    learningRate = 0.6,
    seed = 5,
    historyPoints = 48,
  } = options;

  const safeRank = Math.max(1, Math.min(V, Math.round(rank)));
  const scale = alpha / safeRank;
  const random = tinyRandom(seed);
  const up = new Float32Array(V * safeRank);
  const down = new Float32Array(safeRank * V);
  for (let index = 0; index < down.length; index += 1) {
    down[index] = (random() - 0.5) * 0.4;
  }

  const { inputs, targets } = bigramPairs(text);
  const pairCount = inputs.length;
  const safeBatch = Math.max(1, Math.min(batchSize, Math.max(1, pairCount)));
  const batchesPerEpoch = pairCount === 0 ? 0 : Math.ceil(pairCount / safeBatch);
  const totalSteps = batchesPerEpoch * Math.max(0, epochs);
  const history: TinyHistoryPoint[] = [];
  const probabilities = new Float32Array(V);
  const rowLogits = new Float32Array(V);
  const upGradient = new Float32Array(up.length);
  const downGradient = new Float32Array(down.length);
  const historyEvery = Math.max(1, Math.floor(Math.max(1, totalSteps) / historyPoints));
  let step = 0;
  let windowLoss = 0;
  let windowSteps = 0;

  for (let epoch = 0; epoch < epochs && totalSteps > 0; epoch += 1) {
    const order = shuffledOrder(pairCount, random);
    for (let batch = 0; batch < batchesPerEpoch; batch += 1) {
      const start = batch * safeBatch;
      const end = Math.min(pairCount, start + safeBatch);
      const size = end - start;
      const inverseSize = 1 / size;
      let batchLoss = 0;
      upGradient.fill(0);
      downGradient.fill(0);

      for (let index = start; index < end; index += 1) {
        const pair = order[index];
        const row = inputs[pair];
        const target = targets[pair];
        for (let k = 0; k < V; k += 1) {
          let adaptation = 0;
          for (let j = 0; j < safeRank; j += 1) {
            adaptation += up[row * safeRank + j] * down[j * V + k];
          }
          rowLogits[k] = base[row * V + k] + scale * adaptation;
        }
        let maximum = -Infinity;
        for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, rowLogits[k]);
        let total = 0;
        for (let k = 0; k < V; k += 1) {
          probabilities[k] = Math.exp(rowLogits[k] - maximum);
          total += probabilities[k];
        }
        for (let k = 0; k < V; k += 1) probabilities[k] /= total;
        batchLoss -= Math.log(Math.max(1e-12, probabilities[target]));

        for (let j = 0; j < safeRank; j += 1) {
          let upTerm = 0;
          for (let k = 0; k < V; k += 1) {
            const outer = (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
            upTerm += outer * down[j * V + k];
            downGradient[j * V + k] += outer * up[row * safeRank + j] * scale;
          }
          upGradient[row * safeRank + j] += upTerm * scale;
        }
      }

      batchLoss *= inverseSize;
      for (let index = 0; index < up.length; index += 1) {
        up[index] -= learningRate * upGradient[index];
      }
      for (let index = 0; index < down.length; index += 1) {
        down[index] -= learningRate * downGradient[index];
      }

      windowLoss += batchLoss;
      windowSteps += 1;
      step += 1;
      if (step % historyEvery === 0 || step === totalSteps) {
        history.push({
          step,
          progress: totalSteps <= 1 ? 1 : step / totalSteps,
          loss: windowLoss / Math.max(1, windowSteps),
          learningRate,
          gradientNorm: 0,
        });
        windowLoss = 0;
        windowSteps = 0;
      }
    }
  }

  const delta = loraDelta(up, down, safeRank, scale);
  const merged = Float32Array.from(base);
  for (let index = 0; index < merged.length; index += 1) merged[index] += delta[index];

  return {
    up,
    down,
    delta,
    merged,
    scale,
    history,
    trainableParameters: up.length + down.length,
    fullParameters: V * V,
    finalLoss: tinyCrossEntropy(merged, text),
  };
}

/* -------------------------------------------------------------------------- */
/* Preference optimization                                                     */
/* -------------------------------------------------------------------------- */

export interface TinyPreferencePair {
  prompt: string;
  chosen: string;
  rejected: string;
}

interface Transition {
  row: number;
  target: number;
}

function completionTransitions(prompt: string, completion: string): Transition[] {
  const promptIds = encodeTinyText(prompt);
  const completionIds = encodeTinyText(completion);
  const transitions: Transition[] = [];
  let previous = promptIds.length ? promptIds[promptIds.length - 1] : 0;
  for (const id of completionIds) {
    transitions.push({ row: previous, target: id });
    previous = id;
  }
  return transitions;
}

/** Total log-probability the model assigns to a completion after a prompt. */
export function tinySequenceLogProb(
  weights: Float32Array,
  prompt: string,
  completion: string,
): number {
  const probabilities = new Float32Array(V);
  let total = 0;
  for (const { row, target } of completionTransitions(prompt, completion)) {
    rowSoftmax(weights, row, probabilities);
    total += Math.log(Math.max(1e-12, probabilities[target]));
  }
  return total;
}

export function tinyImplicitReward(
  policy: Float32Array,
  reference: Float32Array,
  pair: { prompt: string; completion: string },
  beta: number,
): number {
  return (
    beta *
    (tinySequenceLogProb(policy, pair.prompt, pair.completion) -
      tinySequenceLogProb(reference, pair.prompt, pair.completion))
  );
}

export interface TinyDpoOptions {
  reference: Float32Array;
  pairs: ReadonlyArray<TinyPreferencePair>;
  beta?: number;
  steps?: number;
  learningRate?: number;
}

export interface TinyDpoPoint {
  step: number;
  loss: number;
  margin: number;
  chosenShift: number;
  rejectedShift: number;
  drift: number;
}

export interface TinyDpoRun {
  weights: Float32Array;
  history: TinyDpoPoint[];
  finalMargin: number;
  finalDrift: number;
  /**
   * Share of pairs whose implicit reward margin is positive. Raw log-probability
   * is the wrong comparison here because a longer completion always scores
   * lower; DPO only ever claims to order the reference-relative rewards.
   */
  accuracy: number;
}

/**
 * Direct preference optimization on the real objective
 * L = -log sigmoid(beta * ((logp_c - logp_c_ref) - (logp_r - logp_r_ref))).
 */
export function trainTinyDpo(options: TinyDpoOptions): TinyDpoRun {
  const { reference, pairs, beta = 0.4, steps = 40, learningRate = 0.35 } = options;
  const weights = Float32Array.from(reference);
  const history: TinyDpoPoint[] = [];
  const probabilities = new Float32Array(V);
  const gradient = new Float32Array(V * V);

  const referenceMargins = pairs.map(
    (pair) =>
      tinySequenceLogProb(reference, pair.prompt, pair.chosen) -
      tinySequenceLogProb(reference, pair.prompt, pair.rejected),
  );

  const measure = (step: number, loss: number) => {
    let margin = 0;
    let chosenShift = 0;
    let rejectedShift = 0;
    let correct = 0;
    pairs.forEach((pair) => {
      const chosen = tinySequenceLogProb(weights, pair.prompt, pair.chosen);
      const rejected = tinySequenceLogProb(weights, pair.prompt, pair.rejected);
      const chosenReference = tinySequenceLogProb(reference, pair.prompt, pair.chosen);
      const rejectedReference = tinySequenceLogProb(
        reference,
        pair.prompt,
        pair.rejected,
      );
      const pairMargin =
        beta * (chosen - chosenReference - (rejected - rejectedReference));
      margin += pairMargin;
      chosenShift += beta * (chosen - chosenReference);
      rejectedShift += beta * (rejected - rejectedReference);
      if (pairMargin > 0) correct += 1;
    });
    const count = Math.max(1, pairs.length);
    history.push({
      step,
      loss,
      margin: margin / count,
      chosenShift: chosenShift / count,
      rejectedShift: rejectedShift / count,
      drift: tinyRowDivergence(weights, reference),
    });
    return correct / count;
  };

  let accuracy = measure(0, dpoLoss(weights, reference, pairs, beta));

  for (let step = 1; step <= steps && pairs.length > 0; step += 1) {
    gradient.fill(0);
    let loss = 0;
    pairs.forEach((pair, index) => {
      const chosen = completionTransitions(pair.prompt, pair.chosen);
      const rejected = completionTransitions(pair.prompt, pair.rejected);
      const policyMargin =
        tinySequenceLogProb(weights, pair.prompt, pair.chosen) -
        tinySequenceLogProb(weights, pair.prompt, pair.rejected);
      const logit = beta * (policyMargin - referenceMargins[index]);
      loss += softplus(-logit);
      // d/dlogit of -log sigmoid(logit)
      const outer = -1 / (1 + Math.exp(logit));
      const factor = (outer * beta) / pairs.length;
      for (const { row, target } of chosen) {
        rowSoftmax(weights, row, probabilities);
        const offset = row * V;
        for (let k = 0; k < V; k += 1) {
          gradient[offset + k] += factor * ((k === target ? 1 : 0) - probabilities[k]);
        }
      }
      for (const { row, target } of rejected) {
        rowSoftmax(weights, row, probabilities);
        const offset = row * V;
        for (let k = 0; k < V; k += 1) {
          gradient[offset + k] -= factor * ((k === target ? 1 : 0) - probabilities[k]);
        }
      }
    });
    for (let index = 0; index < weights.length; index += 1) {
      weights[index] -= learningRate * gradient[index];
    }
    accuracy = measure(step, loss / Math.max(1, pairs.length));
  }

  const last = history[history.length - 1];
  return {
    weights,
    history,
    finalMargin: last?.margin ?? 0,
    finalDrift: last?.drift ?? 0,
    accuracy,
  };
}

function dpoLoss(
  policy: Float32Array,
  reference: Float32Array,
  pairs: ReadonlyArray<TinyPreferencePair>,
  beta: number,
): number {
  if (pairs.length === 0) return 0;
  let total = 0;
  for (const pair of pairs) {
    const policyMargin =
      tinySequenceLogProb(policy, pair.prompt, pair.chosen) -
      tinySequenceLogProb(policy, pair.prompt, pair.rejected);
    const referenceMargin =
      tinySequenceLogProb(reference, pair.prompt, pair.chosen) -
      tinySequenceLogProb(reference, pair.prompt, pair.rejected);
    total += softplus(-beta * (policyMargin - referenceMargin));
  }
  return total / pairs.length;
}

/** log(1 + e^z) without overflow: -log sigmoid(x) is softplus(-x). */
function softplus(z: number) {
  return z >= 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/* -------------------------------------------------------------------------- */
/* Compression                                                                 */
/* -------------------------------------------------------------------------- */

export type TinyQuantScope = "tensor" | "row" | "group";

export interface TinyQuantOptions {
  bits: number;
  scope: TinyQuantScope;
  groupSize?: number;
}

export interface TinyQuantResult {
  weights: Float32Array;
  rmsError: number;
  maxError: number;
  levels: number;
  blocks: number;
  bytesPerWeight: number;
  totalBytes: number;
}

/**
 * Affine min/max quantization. Each block keeps one scale and one zero point,
 * stored as two 16-bit values, which is where the per-block overhead comes from.
 */
export function quantizeTinyWeights(
  source: Float32Array,
  { bits, scope, groupSize = 8 }: TinyQuantOptions,
): TinyQuantResult {
  const safeBits = Math.max(1, Math.min(16, Math.round(bits)));
  const levels = 2 ** safeBits - 1;
  const weights = Float32Array.from(source);
  const blockSize =
    scope === "tensor" ? source.length : scope === "row" ? V : Math.max(1, groupSize);
  const blocks = Math.ceil(source.length / blockSize);

  for (let block = 0; block < blocks; block += 1) {
    const start = block * blockSize;
    const end = Math.min(source.length, start + blockSize);
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let index = start; index < end; index += 1) {
      minimum = Math.min(minimum, source[index]);
      maximum = Math.max(maximum, source[index]);
    }
    const span = maximum - minimum;
    const scale = span === 0 ? 1 : span / levels;
    for (let index = start; index < end; index += 1) {
      const code = Math.round((source[index] - minimum) / scale);
      weights[index] = Math.min(levels, Math.max(0, code)) * scale + minimum;
    }
  }

  let squared = 0;
  let maxError = 0;
  for (let index = 0; index < source.length; index += 1) {
    const error = weights[index] - source[index];
    squared += error * error;
    maxError = Math.max(maxError, Math.abs(error));
  }

  const payloadBits = source.length * safeBits + blocks * 32;
  return {
    weights,
    rmsError: Math.sqrt(squared / source.length),
    maxError,
    levels: levels + 1,
    blocks,
    bytesPerWeight: payloadBits / 8 / source.length,
    totalBytes: Math.ceil(payloadBits / 8),
  };
}

export const GGUF_FORMATS: ReadonlyArray<{
  name: string;
  bits: number;
  scope: TinyQuantScope;
  groupSize: number;
  note: string;
}> = [
  { name: "F16", bits: 16, scope: "tensor", groupSize: 900, note: "Half precision. No quantization loss worth measuring." },
  { name: "Q8_0", bits: 8, scope: "group", groupSize: 32, note: "8-bit, 32-weight blocks. Usually indistinguishable from F16." },
  { name: "Q5_K_M", bits: 5, scope: "group", groupSize: 32, note: "5-bit K-quant. The common quality-first choice." },
  { name: "Q4_K_M", bits: 4, scope: "group", groupSize: 32, note: "4-bit K-quant, medium mix. The usual default download." },
  { name: "Q4_0", bits: 4, scope: "row", groupSize: 28, note: "Older 4-bit layout with coarser scaling." },
  { name: "Q2_K", bits: 2, scope: "group", groupSize: 16, note: "2-bit. Runs anywhere, degrades visibly." },
];

export type TinyPruneMode = "unstructured" | "structured";

export interface TinyPruneResult {
  weights: Float32Array;
  removed: number;
  removedFraction: number;
  removedRows: number[];
}

/**
 * Magnitude pruning. Unstructured zeroes individual logits; structured removes
 * whole context rows, the toy stand-in for dropping a head or MLP neuron.
 */
export function pruneTinyWeights(
  source: Float32Array,
  { fraction, mode }: { fraction: number; mode: TinyPruneMode },
): TinyPruneResult {
  const weights = Float32Array.from(source);
  const share = Math.max(0, Math.min(1, fraction));

  if (mode === "structured") {
    const norms = Array.from({ length: V }, (_, row) => {
      let squared = 0;
      for (let k = 0; k < V; k += 1) squared += source[row * V + k] ** 2;
      return { row, norm: Math.sqrt(squared) };
    }).sort((left, right) => left.norm - right.norm);
    const count = Math.round(share * V);
    const removedRows = norms.slice(0, count).map((entry) => entry.row);
    for (const row of removedRows) {
      for (let k = 0; k < V; k += 1) weights[row * V + k] = 0;
    }
    return {
      weights,
      removed: removedRows.length * V,
      removedFraction: removedRows.length / V,
      removedRows,
    };
  }

  const magnitudes = Array.from(source, (value, index) => ({
    index,
    magnitude: Math.abs(value),
  })).sort((left, right) => left.magnitude - right.magnitude);
  const count = Math.round(share * source.length);
  for (let index = 0; index < count; index += 1) {
    weights[magnitudes[index].index] = 0;
  }
  return {
    weights,
    removed: count,
    removedFraction: count / source.length,
    removedRows: [],
  };
}

export interface TinyFactoredOptions {
  text: string;
  /** Inner dimension of the U·V factorization. Parameter count is 2 · rank · V. */
  rank: number;
  /** Measured against when provided, and imitated when softLabels is set. */
  teacher?: Float32Array;
  softLabels?: boolean;
  temperature?: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  /** Shapes the step size over the run, exactly as in `trainTinyModel`. Default: constant. */
  schedule?: TinySchedule;
  warmupFraction?: number;
  /** L2 pull toward zero, applied to both factors. Default: none. */
  weightDecay?: number;
  /**
   * Rescales any batch gradient whose global norm (both factors together) exceeds
   * this value. 0 disables it.
   */
  clipNorm?: number;
  seed?: number;
  historyPoints?: number;
  checkpoints?: number;
}

export interface TinyFactoredRun {
  weights: Float32Array;
  history: TinyHistoryPoint[];
  checkpoints: TinyCheckpoint[];
  parameters: number;
  fullParameters: number;
  finalLoss: number;
  /** Average KL from the teacher, or NaN when no teacher was supplied. */
  teacherDivergence: number;
  /** Optimizer steps actually taken (fewer than planned if the run diverged). */
  steps: number;
  /** Largest global gradient norm seen on any batch, measured before clipping. */
  peakGradientNorm: number;
  /** Steps on which the gradient was rescaled by `clipNorm`. */
  clippedSteps: number;
  /** True when the loss or a weight stopped being finite and training halted. */
  diverged: boolean;
}

/**
 * Trains a rank-limited model whose logits are the product of two small
 * matrices. With a teacher and softLabels it is distillation; without them it is
 * ordinary training of a smaller model. Both are reported with the same
 * hard-label cross-entropy so the comparison is fair.
 */
export function trainTinyFactored(options: TinyFactoredOptions): TinyFactoredRun {
  const {
    teacher,
    text,
    rank,
    temperature = 2,
    softLabels = false,
    epochs = 12,
    batchSize = 16,
    learningRate = 0.5,
    schedule = "constant",
    warmupFraction = 0.15,
    weightDecay = 0,
    clipNorm = 0,
    seed = 3,
    historyPoints = 48,
    checkpoints = 4,
  } = options;
  const imitate = softLabels && Boolean(teacher);

  const safeRank = Math.max(1, Math.min(V, Math.round(rank)));
  const safeTemperature = Math.max(0.5, temperature);
  const random = tinyRandom(seed);
  const left = new Float32Array(V * safeRank);
  const right = new Float32Array(safeRank * V);
  for (let index = 0; index < left.length; index += 1) left[index] = (random() - 0.5) * 0.3;
  for (let index = 0; index < right.length; index += 1) right[index] = (random() - 0.5) * 0.3;

  const { inputs, targets } = bigramPairs(text);
  const pairCount = inputs.length;
  const safeBatch = Math.max(1, Math.min(batchSize, Math.max(1, pairCount)));
  const batchesPerEpoch = pairCount === 0 ? 0 : Math.ceil(pairCount / safeBatch);
  const totalSteps = batchesPerEpoch * Math.max(0, epochs);
  const history: TinyHistoryPoint[] = [];
  const logits = new Float32Array(V);
  const studentSoft = new Float32Array(V);
  const teacherSoft = new Float32Array(V);
  const leftGradient = new Float32Array(left.length);
  const rightGradient = new Float32Array(right.length);
  const historyEvery = Math.max(1, Math.floor(Math.max(1, totalSteps) / historyPoints));
  let step = 0;
  let windowLoss = 0;
  let windowSteps = 0;
  let peakGradientNorm = 0;
  let clippedSteps = 0;
  let diverged = false;

  const studentWeights = () => {
    const weights = new Float32Array(V * V);
    for (let row = 0; row < V; row += 1) {
      for (let k = 0; k < V; k += 1) {
        let sum = 0;
        for (let j = 0; j < safeRank; j += 1) {
          sum += left[row * safeRank + j] * right[j * V + k];
        }
        weights[row * V + k] = sum;
      }
    }
    return weights;
  };

  const recorded: TinyCheckpoint[] = [];
  const checkpointEvery = Math.max(
    1,
    Math.floor(Math.max(1, totalSteps) / Math.max(1, checkpoints - 1)),
  );
  const record = (atStep: number) => {
    const weights = studentWeights();
    recorded.push({
      step: atStep,
      progress: totalSteps <= 0 ? 1 : atStep / totalSteps,
      loss: tinyCrossEntropy(weights, text),
      weights,
    });
  };
  record(0);

  for (let epoch = 0; epoch < epochs && totalSteps > 0 && !diverged; epoch += 1) {
    const order = shuffledOrder(pairCount, random);
    for (let batch = 0; batch < batchesPerEpoch && !diverged; batch += 1) {
      const start = batch * safeBatch;
      const end = Math.min(pairCount, start + safeBatch);
      const size = end - start;
      const inverseSize = 1 / size;
      let batchLoss = 0;
      leftGradient.fill(0);
      rightGradient.fill(0);

      for (let index = start; index < end; index += 1) {
        const pair = order[index];
        const row = inputs[pair];
        const target = targets[pair];
        for (let k = 0; k < V; k += 1) {
          let sum = 0;
          for (let j = 0; j < safeRank; j += 1) {
            sum += left[row * safeRank + j] * right[j * V + k];
          }
          logits[k] = sum;
        }

        const softmaxInto = (into: Float32Array, source: Float32Array, scale: number) => {
          let maximum = -Infinity;
          for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, source[k] / scale);
          let total = 0;
          for (let k = 0; k < V; k += 1) {
            into[k] = Math.exp(source[k] / scale - maximum);
            total += into[k];
          }
          for (let k = 0; k < V; k += 1) into[k] /= total;
        };

        softmaxInto(studentSoft, logits, 1);
        batchLoss -= Math.log(Math.max(1e-12, studentSoft[target]));

        let outer: Float32Array;
        if (imitate && teacher) {
          const teacherRow = new Float32Array(V);
          for (let k = 0; k < V; k += 1) teacherRow[k] = teacher[row * V + k];
          softmaxInto(teacherSoft, teacherRow, safeTemperature);
          const scaled = new Float32Array(V);
          softmaxInto(scaled, logits, safeTemperature);
          outer = new Float32Array(V);
          // The soft objective is scaled by T², the standard correction that
          // keeps gradient magnitudes comparable across temperatures. Without it
          // a high temperature simply trains more slowly, which looks like
          // distillation failing when it is only a smaller step size.
          for (let k = 0; k < V; k += 1) {
            outer[k] = (scaled[k] - teacherSoft[k]) * safeTemperature * inverseSize;
          }
        } else {
          outer = new Float32Array(V);
          for (let k = 0; k < V; k += 1) {
            outer[k] = (studentSoft[k] - (k === target ? 1 : 0)) * inverseSize;
          }
        }

        for (let j = 0; j < safeRank; j += 1) {
          let leftTerm = 0;
          for (let k = 0; k < V; k += 1) {
            leftTerm += outer[k] * right[j * V + k];
            rightGradient[j * V + k] += outer[k] * left[row * safeRank + j];
          }
          leftGradient[row * safeRank + j] += leftTerm;
        }
      }

      batchLoss *= inverseSize;
      let squaredNorm = 0;
      for (let index = 0; index < left.length; index += 1) {
        squaredNorm += leftGradient[index] ** 2;
      }
      for (let index = 0; index < right.length; index += 1) {
        squaredNorm += rightGradient[index] ** 2;
      }
      const gradientNorm = Math.sqrt(squaredNorm);
      peakGradientNorm = Math.max(peakGradientNorm, gradientNorm);
      const clipped = clipNorm > 0 && gradientNorm > clipNorm;
      if (clipped) clippedSteps += 1;
      const clipScale = clipped ? clipNorm / gradientNorm : 1;
      const stepRate =
        learningRate *
        scheduleFactor(schedule, totalSteps <= 1 ? 1 : step / (totalSteps - 1), warmupFraction);

      if (weightDecay === 0) {
        for (let index = 0; index < left.length; index += 1) {
          left[index] -= stepRate * (leftGradient[index] * clipScale);
        }
        for (let index = 0; index < right.length; index += 1) {
          right[index] -= stepRate * (rightGradient[index] * clipScale);
        }
      } else {
        for (let index = 0; index < left.length; index += 1) {
          left[index] -=
            stepRate * (leftGradient[index] * clipScale + weightDecay * left[index]);
        }
        for (let index = 0; index < right.length; index += 1) {
          right[index] -=
            stepRate * (rightGradient[index] * clipScale + weightDecay * right[index]);
        }
      }

      if (!Number.isFinite(batchLoss) || !Number.isFinite(left[0]) || !Number.isFinite(right[0])) {
        diverged = true;
      }

      windowLoss += Number.isFinite(batchLoss) ? batchLoss : UNIFORM_CROSS_ENTROPY * 4;
      windowSteps += 1;
      step += 1;
      if (step % historyEvery === 0 || step === totalSteps || diverged) {
        history.push({
          step,
          progress: totalSteps <= 1 ? 1 : step / totalSteps,
          loss: windowLoss / Math.max(1, windowSteps),
          learningRate: stepRate,
          // Kept at 0 so results recorded before clipping existed stay identical;
          // the measured norm is reported as `peakGradientNorm` on the run.
          gradientNorm: 0,
        });
        windowLoss = 0;
        windowSteps = 0;
      }
      if (!diverged && (step % checkpointEvery === 0 || step === totalSteps)) record(step);
    }
  }

  const weights = studentWeights();
  return {
    weights,
    history,
    checkpoints: recorded,
    parameters: left.length + right.length,
    fullParameters: V * V,
    finalLoss: diverged ? Number.NaN : tinyCrossEntropy(weights, text),
    teacherDivergence: teacher ? tinyRowDivergence(teacher, weights) : Number.NaN,
    steps: step,
    peakGradientNorm,
    clippedSteps,
    diverged,
  };
}

/* -------------------------------------------------------------------------- */
/* Extended training: optimizers, faults, guards, data mixtures, and a trace. */
/* -------------------------------------------------------------------------- */

/**
 * `trainTinyExtended` is a separate entry point so `trainTinyModel` and `trainTinyFactored`, which
 * about forty pinned lesson figures depend on, never change. With plain SGD on the table it runs the
 * same arithmetic in the same order as `trainTinyModel`, and `tiny-lm.test.ts` checks that the two
 * agree bit for bit; with SGD on the factored model it matches `trainTinyFactored`'s weights.
 */
export type TinyOptimizerKind = "sgd" | "momentum" | "adam" | "adamw" | "muon";

export interface TinyOptimizerOptions {
  kind: TinyOptimizerKind;
  /** Momentum coefficient, or Adam's first-moment decay. Default 0.9, and 0.95 for Muon. */
  beta1?: number;
  /** Adam's second-moment decay. Default 0.999. */
  beta2?: number;
  /** Added to Adam's denominator. Default 1e-8. */
  epsilon?: number;
  /** Adam only: divide each moment by 1 − βᵗ so early steps are not shrunk toward zero. Default true. */
  biasCorrection?: boolean;
  /** Muon only: Nesterov momentum, as in the reference implementation. Default true. */
  nesterov?: boolean;
  /** Muon only: Newton–Schulz iterations, 0 to 8. Default 5. */
  newtonSchulzSteps?: number;
}

/** `wsd` is warmup, a long constant stretch, then a linear decay to zero over `decayFraction`. */
export type TinyExtendedSchedule = TinySchedule | "wsd";

export type TinyFault =
  /** Replaces `count` consecutive batches from `atStep` with pairs drawn from `text`. */
  | { kind: "bad-batch"; atStep: number; count?: number; text: string }
  /**
   * Emulates an fp16 backward pass: gradients are multiplied by `scale`, rounded to half precision,
   * then divided back. A static scale lets an overflow reach the weights; a dynamic one skips that
   * step, halves the scale, and doubles it again after `growthInterval` clean steps.
   */
  | { kind: "loss-scale"; scale: number; dynamic?: boolean; growthInterval?: number };

/** From `from` (a fraction of the run) onward, batches draw from the sources in these proportions. */
export interface TinyMixturePhase {
  from: number;
  weights: ReadonlyArray<number>;
}

export type TinyExtendedModel =
  | { kind: "table"; init?: Float32Array }
  /** Logits are U·V with an inner width of `rank`; both factors start uniform in ±initScale/2. */
  | { kind: "factored"; rank: number; initScale?: number };

export interface TinyExtendedOptions {
  /** Training text in epoch mode; in step mode only the text the checkpoints and final loss read. */
  text: string;
  model?: TinyExtendedModel;
  epochs?: number;
  /** Step mode: train for exactly this many batches, sampled from `sources` by `mixture`. */
  steps?: number;
  batchSize?: number;
  learningRate?: number;
  schedule?: TinyExtendedSchedule;
  warmupFraction?: number;
  /** `wsd` only: the closing share of the run spent decaying. Default 0.2. */
  decayFraction?: number;
  /** L2 for SGD, momentum and Adam; decoupled for AdamW and Muon. */
  weightDecay?: number;
  /** Rescales any batch gradient whose global norm exceeds this value. 0 disables it. */
  clipNorm?: number;
  seed?: number;
  optimizer?: TinyOptimizerOptions;
  faults?: ReadonlyArray<TinyFault>;
  /** Adds z · (log Z)² per example, which pulls the log-partition toward zero (PaLM uses 1e-4). */
  zLoss?: number;
  /** Half-open step ranges [from, to) whose batches are read but never applied. */
  skipSteps?: ReadonlyArray<readonly [number, number]>;
  sources?: ReadonlyArray<{ id: string; text: string }>;
  mixture?: ReadonlyArray<TinyMixturePhase>;
  checkpoints?: number;
  historyPoints?: number;
  /** Record one `TinyTracePoint` per step. */
  trace?: boolean;
  /** Keep the first parameter matrix's gradient and update at this step (0-based). */
  captureStep?: number;
}

export interface TinyTracePoint {
  step: number;
  /** Cross-entropy of this batch, before the update. */
  loss: number;
  learningRate: number;
  /** Global gradient norm after unscaling and before clipping; Infinity on an overflow. */
  gradientNorm: number;
  clipped: boolean;
  /** Largest logit any example in the batch produced. */
  maxLogit: number;
  /** Mean log-partition, log Σ exp(logit), over the batch. */
  logPartition: number;
  /** ‖Δw‖ / ‖w‖ after the step: how far the step moved the weights relative to their size. */
  updateRatio: number;
  /** Largest single-weight change in the step. */
  maxUpdate: number;
  skipped: boolean;
  overflow: boolean;
  lossScale: number;
  /** The batch came from a `bad-batch` fault. */
  faulty: boolean;
}

export interface TinyExtendedRun extends TinyTrainRun {
  parameters: number;
  clippedSteps: number;
  skippedSteps: number;
  overflowSteps: number;
  /** The 1-based step on which the loss or a weight stopped being finite, or null. */
  haltedAt: number | null;
  /** Floats the optimizer keeps between steps. */
  stateFloats: number;
  /** Table model: total ‖Δrow‖ over the run, and the number of batches whose inputs included the row. */
  rowMovement: Float64Array;
  rowVisits: Int32Array;
  /** Step mode: pairs drawn from each source, and the pairs each source holds. */
  sourceSeen: number[];
  sourcePairs: number[];
  trace: TinyTracePoint[];
  captured: { gradient: Float32Array; update: Float32Array; rows: number; cols: number } | null;
}

/** Multiplier on the peak rate; every schedule but `wsd` is `scheduleFactor`. */
export function extendedScheduleFactor(
  schedule: TinyExtendedSchedule,
  progress: number,
  warmupFraction = 0.15,
  decayFraction = 0.2,
): number {
  if (schedule !== "wsd") return scheduleFactor(schedule, progress, warmupFraction);
  const point = Math.min(1, Math.max(0, progress));
  const warmup = Math.min(0.9, Math.max(0, warmupFraction));
  const decay = Math.min(1 - warmup, Math.max(0, decayFraction));
  if (warmup > 0 && point < warmup) return Math.max(0.02, point / warmup);
  if (decay > 0 && point > 1 - decay) return Math.max(0, (1 - point) / decay);
  return 1;
}

const HALF_MAX_FINITE = 65504;
const HALF_MIN_NORMAL = 2 ** -14;
const HALF_SUBNORMAL_STEP = 2 ** -24;

/**
 * Rounds a number to IEEE half precision: an 11-bit significand, ±Infinity from 65520 up, and
 * subnormal steps of 2⁻²⁴ below 2⁻¹⁴. Ties round away from zero, a simplification of round-to-even.
 */
export function toHalf(value: number): number {
  if (!Number.isFinite(value)) return value;
  if (value === 0) return 0;
  const sign = value < 0 ? -1 : 1;
  const magnitude = Math.abs(value);
  let rounded: number;
  if (magnitude < HALF_MIN_NORMAL) {
    rounded = Math.round(magnitude / HALF_SUBNORMAL_STEP) * HALF_SUBNORMAL_STEP;
  } else {
    const quantum = 2 ** (Math.floor(Math.log2(magnitude)) - 10);
    rounded = Math.round(magnitude / quantum) * quantum;
  }
  if (rounded > HALF_MAX_FINITE) return sign * Infinity;
  return sign * rounded;
}

const NEWTON_SCHULZ = [3.4445, -4.775, 2.0315] as const;

/**
 * Muon's quintic Newton–Schulz iteration (Jordan et al., 2024): normalizes a matrix and pushes its
 * singular values toward 1 while keeping its singular vectors, an approximate orthogonalization.
 * Zero directions stay zero. Returns a new rows × cols matrix.
 */
export function newtonSchulz(
  source: ArrayLike<number>,
  rows: number,
  cols: number,
  steps = 5,
): Float64Array {
  const transpose = rows > cols;
  const r = transpose ? cols : rows;
  const c = transpose ? rows : cols;
  let norm = 0;
  for (let index = 0; index < rows * cols; index += 1) norm += source[index] * source[index];
  norm = Math.sqrt(norm) + 1e-7;
  let x = new Float64Array(r * c);
  for (let i = 0; i < r; i += 1) {
    for (let j = 0; j < c; j += 1) {
      x[i * c + j] = (transpose ? source[j * cols + i] : source[i * cols + j]) / norm;
    }
  }
  const [a, b, cc] = NEWTON_SCHULZ;
  const gram = new Float64Array(r * r);
  const poly = new Float64Array(r * r);
  let next = new Float64Array(r * c);
  const iterations = Math.max(0, Math.min(8, Math.round(steps)));
  for (let step = 0; step < iterations; step += 1) {
    for (let i = 0; i < r; i += 1) {
      for (let k = 0; k < r; k += 1) {
        let sum = 0;
        for (let j = 0; j < c; j += 1) sum += x[i * c + j] * x[k * c + j];
        gram[i * r + k] = sum;
      }
    }
    for (let i = 0; i < r; i += 1) {
      for (let k = 0; k < r; k += 1) {
        let sum = 0;
        for (let l = 0; l < r; l += 1) sum += gram[i * r + l] * gram[l * r + k];
        poly[i * r + k] = b * gram[i * r + k] + cc * sum;
      }
    }
    for (let i = 0; i < r; i += 1) {
      for (let j = 0; j < c; j += 1) {
        let sum = 0;
        for (let l = 0; l < r; l += 1) sum += poly[i * r + l] * x[l * c + j];
        next[i * c + j] = a * x[i * c + j] + sum;
      }
    }
    [x, next] = [next, x];
  }
  const out = new Float64Array(rows * cols);
  for (let i = 0; i < r; i += 1) {
    for (let j = 0; j < c; j += 1) {
      if (transpose) out[j * cols + i] = x[i * c + j];
      else out[i * cols + j] = x[i * c + j];
    }
  }
  return out;
}

interface ParameterMatrix {
  data: Float32Array;
  grad: Float32Array;
  rows: number;
  cols: number;
  first: Float32Array | null;
  second: Float32Array | null;
}

/** Trains the table or factored model with any optimizer, fault, guard or mixture. */
export function trainTinyExtended(options: TinyExtendedOptions): TinyExtendedRun {
  const {
    text,
    model = { kind: "table" },
    epochs = 8,
    batchSize = 16,
    learningRate = 0.6,
    schedule = "constant",
    warmupFraction = 0.15,
    decayFraction = 0.2,
    weightDecay = 0,
    clipNorm = 0,
    seed = 7,
    optimizer = { kind: "sgd" },
    faults = [],
    zLoss = 0,
    skipSteps = [],
    checkpoints = 6,
    historyPoints = 56,
    trace = false,
    captureStep,
  } = options;
  const kind = optimizer.kind;
  const beta1 = optimizer.beta1 ?? (kind === "muon" ? 0.95 : 0.9);
  const beta2 = optimizer.beta2 ?? 0.999;
  const epsilon = optimizer.epsilon ?? 1e-8;
  const biasCorrection = optimizer.biasCorrection ?? true;
  const nesterov = optimizer.nesterov ?? true;
  const nsSteps = optimizer.newtonSchulzSteps ?? 5;
  const stepMode = options.steps !== undefined || options.sources !== undefined;
  const watch = trace || zLoss > 0;

  const random = tinyRandom(seed);
  const table = model.kind === "table";
  const rank = table ? 0 : Math.max(1, Math.min(V, Math.round(model.rank)));
  const matrices: ParameterMatrix[] = [];
  const makeMatrix = (rows: number, cols: number): ParameterMatrix => ({
    data: new Float32Array(rows * cols),
    grad: new Float32Array(rows * cols),
    rows,
    cols,
    first: kind === "sgd" ? null : new Float32Array(rows * cols),
    second: kind === "adam" || kind === "adamw" ? new Float32Array(rows * cols) : null,
  });
  if (model.kind === "table") {
    const only = makeMatrix(V, V);
    if (model.init) only.data.set(model.init.subarray(0, V * V));
    matrices.push(only);
  } else {
    const scale = model.initScale ?? 0.3;
    const left = makeMatrix(V, rank);
    const right = makeMatrix(rank, V);
    for (let index = 0; index < left.data.length; index += 1) left.data[index] = (random() - 0.5) * scale;
    for (let index = 0; index < right.data.length; index += 1) right.data[index] = (random() - 0.5) * scale;
    matrices.push(left, right);
  }
  const parameters = matrices.reduce((sum, matrix) => sum + matrix.data.length, 0);
  const effectiveWeights = () => {
    if (table) return Float32Array.from(matrices[0].data);
    const [left, right] = matrices;
    const weights = new Float32Array(V * V);
    for (let row = 0; row < V; row += 1) {
      for (let k = 0; k < V; k += 1) {
        let sum = 0;
        for (let j = 0; j < rank; j += 1) sum += left.data[row * rank + j] * right.data[j * V + k];
        weights[row * V + k] = sum;
      }
    }
    return weights;
  };

  // Data: one shuffled pass per epoch, or per-source cursors sampled by the mixture.
  const sources = stepMode
    ? options.sources && options.sources.length > 0
      ? options.sources
      : [{ id: "text", text }]
    : [{ id: "text", text }];
  const sourceData = sources.map((source) => bigramPairs(source.text));
  const pairCount = sourceData[0].inputs.length;
  const safeBatch = Math.max(1, Math.min(batchSize, stepMode ? batchSize : Math.max(1, pairCount)));
  const batchesPerEpoch = pairCount === 0 ? 0 : Math.ceil(pairCount / safeBatch);
  const usable = sourceData.some((data) => data.inputs.length > 0);
  const totalSteps = stepMode
    ? usable
      ? Math.max(0, Math.round(options.steps ?? 0))
      : 0
    : batchesPerEpoch * Math.max(0, epochs);
  const sourceSeen = sources.map(() => 0);
  const sourcePairs = sourceData.map((data) => data.inputs.length);

  const history: TinyHistoryPoint[] = [];
  const recorded: TinyCheckpoint[] = [];
  const traced: TinyTracePoint[] = [];
  const rowMovement = new Float64Array(V);
  const rowVisits = new Int32Array(V);
  const stateFloats =
    kind === "sgd" ? 0 : kind === "adam" || kind === "adamw" ? 2 * parameters : parameters;

  const emptyRun = (): TinyExtendedRun => {
    const weights = effectiveWeights();
    return {
      weights,
      history,
      checkpoints: [{ step: 0, progress: 0, loss: UNIFORM_CROSS_ENTROPY, weights: Float32Array.from(weights) }],
      steps: 0,
      tokens: pairCount,
      updatedParameters: 0,
      finalLoss: tinyCrossEntropy(weights, text),
      diverged: false,
      peakGradientNorm: 0,
      parameters,
      clippedSteps: 0,
      skippedSteps: 0,
      overflowSteps: 0,
      haltedAt: null,
      stateFloats,
      rowMovement,
      rowVisits,
      sourceSeen,
      sourcePairs,
      trace: traced,
      captured: null,
    };
  };
  if (totalSteps === 0) return emptyRun();

  const cursors = sourceData.map((data) => ({ order: shuffledOrder(0, random), at: 0, size: data.inputs.length }));
  if (stepMode) {
    for (const cursor of cursors) {
      cursor.order = shuffledOrder(cursor.size, random);
    }
  }
  const mixturePhases = [...(options.mixture ?? [{ from: 0, weights: sources.map(() => 1) }])].sort(
    (left, right) => left.from - right.from,
  );
  const badBatch = faults.find((fault) => fault.kind === "bad-batch");
  const badPairs = badBatch ? bigramPairs(badBatch.text) : null;
  const lossScaleFault = faults.find((fault) => fault.kind === "loss-scale");
  let lossScale = lossScaleFault ? Math.max(1, lossScaleFault.scale) : 1;
  const growthInterval = lossScaleFault?.dynamic ? Math.max(1, lossScaleFault.growthInterval ?? 200) : 0;
  let cleanSteps = 0;

  const batchInputs = new Int32Array(safeBatch);
  const batchTargets = new Int32Array(safeBatch);
  const probabilities = new Float32Array(V);
  const logits = new Float32Array(V);
  const outer = new Float32Array(V);
  const touched = new Set<number>();
  const updatedRows = new Set<number>();
  const delta = new Float32Array(Math.max(...matrices.map((matrix) => matrix.data.length)));
  const historyEvery = Math.max(1, Math.floor(totalSteps / historyPoints));
  const checkpointEvery = Math.max(1, Math.floor(totalSteps / Math.max(1, checkpoints - 1)));
  let step = 0;
  let adamStep = 0;
  let windowLoss = 0;
  let windowSteps = 0;
  let peakGradientNorm = 0;
  let clippedSteps = 0;
  let skippedSteps = 0;
  let overflowSteps = 0;
  let diverged = false;
  let haltedAt: number | null = null;
  let captured: TinyExtendedRun["captured"] = null;
  let order = shuffledOrder(0, random);

  recorded.push({ step: 0, progress: 0, loss: tinyCrossEntropy(effectiveWeights(), text), weights: effectiveWeights() });

  const softmaxInto = (into: Float32Array, source: Float32Array) => {
    let maximum = -Infinity;
    for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, source[k] / 1);
    let total = 0;
    for (let k = 0; k < V; k += 1) {
      into[k] = Math.exp(source[k] / 1 - maximum);
      total += into[k];
    }
    for (let k = 0; k < V; k += 1) into[k] /= total;
  };

  const drawFromSource = (source: number) => {
    const cursor = cursors[source];
    if (cursor.at >= cursor.size) {
      cursor.order = shuffledOrder(cursor.size, random);
      cursor.at = 0;
    }
    const pair = cursor.order[cursor.at];
    cursor.at += 1;
    sourceSeen[source] += 1;
    return pair;
  };

  while (step < totalSteps && !diverged) {
    // Assemble this step's batch.
    let size: number;
    if (stepMode) {
      const progressNow = totalSteps <= 1 ? 1 : step / (totalSteps - 1);
      let phase = mixturePhases[0];
      for (const candidate of mixturePhases) if (candidate.from <= progressNow) phase = candidate;
      const weights = sources.map((_, index) =>
        sourceData[index].inputs.length > 0 ? Math.max(0, phase.weights[index] ?? 0) : 0,
      );
      const total = weights.reduce((sum, value) => sum + value, 0);
      const shares = total > 0 ? weights.map((value) => value / total) : weights.map((_, index) => (sourceData[index].inputs.length > 0 ? 1 : 0));
      const shareTotal = shares.reduce((sum, value) => sum + value, 0);
      size = safeBatch;
      for (let slot = 0; slot < size; slot += 1) {
        let pick = random() * shareTotal;
        let source = 0;
        while (source < shares.length - 1 && pick >= shares[source]) {
          pick -= shares[source];
          source += 1;
        }
        while (shares[source] === 0 && source > 0) source -= 1;
        const pair = drawFromSource(source);
        batchInputs[slot] = sourceData[source].inputs[pair];
        batchTargets[slot] = sourceData[source].targets[pair];
      }
    } else {
      const batch = step % batchesPerEpoch;
      if (batch === 0) order = shuffledOrder(pairCount, random);
      const start = batch * safeBatch;
      const end = Math.min(pairCount, start + safeBatch);
      size = end - start;
      for (let slot = 0; slot < size; slot += 1) {
        const pair = order[start + slot];
        batchInputs[slot] = sourceData[0].inputs[pair];
        batchTargets[slot] = sourceData[0].targets[pair];
      }
    }
    const faulty =
      badBatch !== undefined &&
      badPairs !== null &&
      badPairs.inputs.length > 0 &&
      step >= badBatch.atStep &&
      step < badBatch.atStep + Math.max(1, badBatch.count ?? 1);
    if (faulty && badPairs) {
      const offset = (step - (badBatch?.atStep ?? 0)) * size;
      for (let slot = 0; slot < size; slot += 1) {
        const pair = (offset + slot) % badPairs.inputs.length;
        batchInputs[slot] = badPairs.inputs[pair];
        batchTargets[slot] = badPairs.targets[pair];
      }
    }

    // Forward and backward.
    const inverseSize = 1 / size;
    let batchLoss = 0;
    let maxLogit = -Infinity;
    let partitionSum = 0;
    touched.clear();
    for (let slot = 0; slot < size; slot += 1) {
      const row = batchInputs[slot];
      const target = batchTargets[slot];
      if (table) {
        const weights = matrices[0].data;
        const gradient = matrices[0].grad;
        rowSoftmax(weights, row, probabilities);
        batchLoss -= Math.log(Math.max(1e-12, probabilities[target]));
        const offset = row * V;
        let logPartition = 0;
        if (watch) {
          let maximum = -Infinity;
          for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, weights[offset + k]);
          let total = 0;
          for (let k = 0; k < V; k += 1) total += Math.exp(weights[offset + k] - maximum);
          logPartition = maximum + Math.log(total);
          maxLogit = Math.max(maxLogit, maximum);
          partitionSum += logPartition;
        }
        if (zLoss > 0) {
          for (let k = 0; k < V; k += 1) {
            gradient[offset + k] +=
              ((probabilities[k] - (k === target ? 1 : 0)) + 2 * zLoss * logPartition * probabilities[k]) * inverseSize;
          }
        } else {
          for (let k = 0; k < V; k += 1) {
            gradient[offset + k] += (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
          }
        }
        touched.add(row);
      } else {
        const [left, right] = matrices;
        for (let k = 0; k < V; k += 1) {
          let sum = 0;
          for (let j = 0; j < rank; j += 1) sum += left.data[row * rank + j] * right.data[j * V + k];
          logits[k] = sum;
        }
        softmaxInto(probabilities, logits);
        batchLoss -= Math.log(Math.max(1e-12, probabilities[target]));
        let logPartition = 0;
        if (watch) {
          let maximum = -Infinity;
          for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, logits[k]);
          let total = 0;
          for (let k = 0; k < V; k += 1) total += Math.exp(logits[k] - maximum);
          logPartition = maximum + Math.log(total);
          maxLogit = Math.max(maxLogit, maximum);
          partitionSum += logPartition;
        }
        for (let k = 0; k < V; k += 1) {
          outer[k] =
            zLoss > 0
              ? ((probabilities[k] - (k === target ? 1 : 0)) + 2 * zLoss * logPartition * probabilities[k]) * inverseSize
              : (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
        }
        for (let j = 0; j < rank; j += 1) {
          let leftTerm = 0;
          for (let k = 0; k < V; k += 1) {
            leftTerm += outer[k] * right.data[j * V + k];
            right.grad[j * V + k] += outer[k] * left.data[row * rank + j];
          }
          left.grad[row * rank + j] += leftTerm;
        }
        touched.add(row);
      }
    }
    for (const row of touched) rowVisits[row] += 1;
    batchLoss *= inverseSize;

    // Emulated fp16 backward: scale, round, unscale.
    let overflow = false;
    if (lossScaleFault) {
      for (const matrix of matrices) {
        const gradient = matrix.grad;
        for (let index = 0; index < gradient.length; index += 1) {
          if (gradient[index] === 0) continue;
          const half = toHalf(gradient[index] * lossScale);
          if (!Number.isFinite(half)) overflow = true;
          gradient[index] = half / lossScale;
        }
      }
    }

    // Gradient norm, summed in the same order as trainTinyModel and trainTinyFactored.
    let squaredNorm = 0;
    if (table) {
      const gradient = matrices[0].grad;
      for (const row of touched) {
        const offset = row * V;
        for (let k = 0; k < V; k += 1) squaredNorm += gradient[offset + k] ** 2;
      }
    } else {
      for (const matrix of matrices) {
        for (let index = 0; index < matrix.grad.length; index += 1) squaredNorm += matrix.grad[index] ** 2;
      }
    }
    const gradientNorm = overflow ? Infinity : Math.sqrt(squaredNorm);
    if (Number.isFinite(gradientNorm)) peakGradientNorm = Math.max(peakGradientNorm, gradientNorm);
    const progress = totalSteps <= 1 ? 1 : step / (totalSteps - 1);
    const stepRate = learningRate * extendedScheduleFactor(schedule, progress, warmupFraction, decayFraction);

    const inSkipRange = skipSteps.some(([from, to]) => step >= from && step < to);
    const dynamicSkip = overflow && growthInterval > 0;
    const skipped = inSkipRange || dynamicSkip;
    const clipped = !skipped && clipNorm > 0 && gradientNorm > clipNorm;
    const clipScale = clipped ? clipNorm / gradientNorm : 1;
    let deltaSquared = 0;
    let weightSquared = 0;
    let maxUpdate = 0;

    if (skipped) {
      skippedSteps += 1;
      if (dynamicSkip) {
        overflowSteps += 1;
        lossScale = Math.max(1, lossScale / 2);
        cleanSteps = 0;
      }
      for (const matrix of matrices) matrix.grad.fill(0);
    } else {
      if (clipped) clippedSteps += 1;
      if (overflow) overflowSteps += 1;
      if (growthInterval > 0) {
        cleanSteps += 1;
        if (cleanSteps >= growthInterval) {
          lossScale *= 2;
          cleanSteps = 0;
        }
      }
      if (kind === "adam" || kind === "adamw") adamStep += 1;
      for (let which = 0; which < matrices.length; which += 1) {
        const matrix = matrices[which];
        const { data, grad, rows, cols } = matrix;
        const capture = which === 0 && captureStep === step;
        if (kind === "sgd") {
          if (table) {
            for (const row of touched) {
              const offset = row * V;
              for (let k = 0; k < V; k += 1) {
                const before = data[offset + k];
                const decay = weightDecay * data[offset + k];
                data[offset + k] -= stepRate * (grad[offset + k] * clipScale + decay);
                delta[offset + k] = data[offset + k] - before;
              }
              updatedRows.add(row);
            }
            if (capture) captured = { gradient: Float32Array.from(grad), update: Float32Array.from(grad), rows, cols };
            for (const row of touched) grad.fill(0, row * V, row * V + V);
          } else {
            if (capture) captured = { gradient: Float32Array.from(grad), update: Float32Array.from(grad), rows, cols };
            for (let index = 0; index < data.length; index += 1) {
              const before = data[index];
              if (weightDecay === 0) data[index] -= stepRate * (grad[index] * clipScale);
              else data[index] -= stepRate * (grad[index] * clipScale + weightDecay * data[index]);
              delta[index] = data[index] - before;
            }
            grad.fill(0);
          }
        } else if (kind === "momentum") {
          const velocity = matrix.first!;
          for (let index = 0; index < data.length; index += 1) {
            const before = data[index];
            velocity[index] = beta1 * velocity[index] + (grad[index] * clipScale + weightDecay * data[index]);
            data[index] -= stepRate * velocity[index];
            delta[index] = data[index] - before;
          }
          if (capture) captured = { gradient: Float32Array.from(grad), update: Float32Array.from(velocity), rows, cols };
          grad.fill(0);
        } else if (kind === "adam" || kind === "adamw") {
          const first = matrix.first!;
          const second = matrix.second!;
          const firstCorrection = biasCorrection ? 1 - beta1 ** adamStep : 1;
          const secondCorrection = biasCorrection ? 1 - beta2 ** adamStep : 1;
          const update = capture ? new Float32Array(data.length) : null;
          for (let index = 0; index < data.length; index += 1) {
            const before = data[index];
            let g = grad[index] * clipScale;
            if (kind === "adam") g += weightDecay * data[index];
            else data[index] *= 1 - stepRate * weightDecay;
            first[index] = beta1 * first[index] + (1 - beta1) * g;
            second[index] = beta2 * second[index] + (1 - beta2) * g * g;
            const step = first[index] / firstCorrection / (Math.sqrt(second[index] / secondCorrection) + epsilon);
            if (update) update[index] = step;
            data[index] -= stepRate * step;
            delta[index] = data[index] - before;
          }
          if (capture && update) captured = { gradient: Float32Array.from(grad), update, rows, cols };
          grad.fill(0);
        } else {
          const momentum = matrix.first!;
          const direction = new Float64Array(data.length);
          for (let index = 0; index < data.length; index += 1) {
            const g = grad[index] * clipScale;
            momentum[index] = beta1 * momentum[index] + g;
            direction[index] = nesterov ? g + beta1 * momentum[index] : momentum[index];
          }
          const orthogonal = newtonSchulz(direction, rows, cols, nsSteps);
          const aspect = Math.sqrt(Math.max(1, rows / cols));
          for (let index = 0; index < data.length; index += 1) {
            const before = data[index];
            data[index] *= 1 - stepRate * weightDecay;
            data[index] -= stepRate * orthogonal[index] * aspect;
            delta[index] = data[index] - before;
          }
          if (capture) {
            captured = { gradient: Float32Array.from(direction), update: Float32Array.from(orthogonal), rows, cols };
          }
          grad.fill(0);
        }
        const touchedAll = !(kind === "sgd" && table);
        const length = touchedAll ? data.length : 0;
        for (let index = 0; index < length; index += 1) {
          deltaSquared += delta[index] * delta[index];
          maxUpdate = Math.max(maxUpdate, Math.abs(delta[index]));
        }
        if (!touchedAll) {
          for (const row of touched) {
            for (let k = 0; k < V; k += 1) {
              const change = delta[row * V + k];
              deltaSquared += change * change;
              maxUpdate = Math.max(maxUpdate, Math.abs(change));
            }
          }
        }
        if (table) {
          const rowsMoved = touchedAll ? Array.from({ length: V }, (_, row) => row) : [...touched];
          for (const row of rowsMoved) {
            let rowSquared = 0;
            for (let k = 0; k < V; k += 1) rowSquared += delta[row * V + k] * delta[row * V + k];
            rowMovement[row] += Math.sqrt(rowSquared);
          }
          if (touchedAll) for (let row = 0; row < V; row += 1) updatedRows.add(row);
        }
        for (let index = 0; index < data.length; index += 1) weightSquared += data[index] * data[index];
        if (!touchedAll) delta.fill(0);
      }
    }

    let finite = Number.isFinite(batchLoss);
    for (const matrix of matrices) {
      if (!finite) break;
      for (let index = 0; index < matrix.data.length; index += 1) {
        if (!Number.isFinite(matrix.data[index])) {
          finite = false;
          break;
        }
      }
    }
    if (!finite) {
      diverged = true;
      haltedAt = step + 1;
    }

    if (trace) {
      traced.push({
        step,
        loss: batchLoss,
        learningRate: stepRate,
        gradientNorm,
        clipped,
        maxLogit: Number.isFinite(maxLogit) ? maxLogit : 0,
        logPartition: size > 0 ? partitionSum / size : 0,
        updateRatio: weightSquared > 0 ? Math.sqrt(deltaSquared / weightSquared) : 0,
        maxUpdate,
        skipped,
        overflow,
        lossScale,
        faulty,
      });
    }

    windowLoss += Number.isFinite(batchLoss) ? batchLoss : UNIFORM_CROSS_ENTROPY * 4;
    windowSteps += 1;
    step += 1;

    if (step % historyEvery === 0 || step === totalSteps || diverged) {
      history.push({
        step,
        progress,
        loss: windowLoss / Math.max(1, windowSteps),
        learningRate: stepRate,
        gradientNorm,
      });
      windowLoss = 0;
      windowSteps = 0;
    }
    if (step % checkpointEvery === 0 || step === totalSteps) {
      const weights = effectiveWeights();
      recorded.push({ step, progress, loss: diverged ? Number.NaN : tinyCrossEntropy(weights, text), weights });
    }
  }

  const weights = effectiveWeights();
  return {
    weights,
    history,
    checkpoints: recorded,
    steps: step,
    tokens: pairCount,
    updatedParameters: table ? updatedRows.size * V : parameters,
    finalLoss: diverged ? Number.NaN : tinyCrossEntropy(weights, text),
    diverged,
    peakGradientNorm,
    parameters,
    clippedSteps,
    skippedSteps,
    overflowSteps,
    haltedAt,
    stateFloats,
    rowMovement,
    rowVisits,
    sourceSeen,
    sourcePairs,
    trace: traced,
    captured,
  };
}
