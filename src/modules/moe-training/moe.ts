/**
 * A mixture-of-experts layer small enough to train inside a render, on the same character pairs
 * every tiny-LM lab uses. Every loss, load share, gate and drop rate the lab shows is measured here
 * from a real run; the parameter counts are formulas, and the Mixtral figures are authored constants.
 *
 * The model reads one previous character x and predicts the next one.
 *
 *   h      = [E[x] (4 learned numbers), 1]                        the input, D = 5 numbers
 *   z      = h · R                                                one router score per expert (R is 5 × N)
 *   p      = softmax(z)                                           the gate values
 *   chosen = the top k experts by z + b                           b is the balancing bias, 0 unless used
 *   logits = Σ over chosen experts e of p_e · (h · W_e)            each W_e is 5 × 30
 *
 * The bias b changes only which experts are chosen, never a gate value. The constant 1 at the end
 * of h gives every expert a next-character bias that all of its tokens train, and gives the router a
 * per-expert bias: that pair is what lets one expert get richer as it gets more tokens.
 *
 * Training is plain minibatch SGD on cross-entropy, learning rate 0.5, batches of 16 pairs, over
 * all three tiny corpora, from weights drawn uniformly from the seed (the router's bias row starts at
 * zero). Two optional balancing methods act on the router:
 *
 *   - Auxiliary loss: adds α · N · Σ_e f_e · P_e to each batch's loss, where f_e is the share of
 *     the batch whose first choice is e (held constant) and P_e is the batch's mean gate value for e
 *     (Switch Transformer; f from first choices as in GShard).
 *   - Bias update: after each batch, b_e += γ · sign(B·k/N − c_e), where c_e counts the batch's
 *     choices of e (auxiliary-loss-free balancing, as in DeepSeek-V3).
 *
 * A capacity factor cf caps every expert at ⌈cf · B · k / N⌉ seats per batch. Every token's first
 * choice is seated before any second choice, each in batch order; a choice that finds its expert
 * full is dropped, and a token that loses every expert predicts uniformly (ln 30), because this toy
 * has no residual stream to carry it past the layer.
 */
import { bigramPairs, TINY_CORPORA, TINY_VOCAB_SIZE, tinyRandom, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";

export const VOCAB = TINY_VOCAB_SIZE;
/** Learned numbers per character: the width of the embedding. */
export const EMBED = 4;
/** The router's and every expert's input: the embedding plus the constant 1. */
export const INPUT = EMBED + 1;
export const LEARNING_RATE = 0.5;
export const BATCH = 16;
export const MAX_EXPERTS = 8;
export const MAX_TOP_K = 2;
/** Uniform spread of the starting weights: embedding, router, experts. */
export const INIT_SCALE: Readonly<InitScale> = { embed: 0.5, router: 0.1, expert: 0.1 };

export interface InitScale {
  embed: number;
  router: number;
  expert: number;
}

export type Balancing = "none" | "aux" | "bias";

export const CORPUS_IDS = ["harbor", "recipes", "proverbs"] as const;
export type CorpusId = (typeof CORPUS_IDS)[number];

export interface MoeData {
  inputs: Int32Array;
  targets: Int32Array;
  /** Index into CORPUS_IDS for every pair. */
  corpus: Int32Array;
  count: number;
  /** How many pairs each corpus contributes, in CORPUS_IDS order. */
  corpusCounts: number[];
  /** How many pairs start with each character. */
  characterCounts: number[];
}

let cachedData: MoeData | null = null;

/** Every bigram pair of the three corpora, corpus by corpus, so no pair spans two texts. */
export function moeData(): MoeData {
  if (cachedData) return cachedData;
  const parts = CORPUS_IDS.map((id) => bigramPairs(TINY_CORPORA[id].text));
  const count = parts.reduce((sum, part) => sum + part.inputs.length, 0);
  const inputs = new Int32Array(count);
  const targets = new Int32Array(count);
  const corpus = new Int32Array(count);
  const characterCounts = new Array<number>(VOCAB).fill(0);
  let at = 0;
  parts.forEach((part, index) => {
    inputs.set(part.inputs, at);
    targets.set(part.targets, at);
    corpus.fill(index, at, at + part.inputs.length);
    at += part.inputs.length;
  });
  for (const input of inputs) characterCounts[input] += 1;
  cachedData = { inputs, targets, corpus, count, corpusCounts: parts.map((part) => part.inputs.length), characterCounts };
  return cachedData;
}

export interface MoeSettings {
  experts: number;
  topK: number;
  balancing: Balancing;
  auxWeight: number;
  biasStep: number;
  /** Capacity factor; 0 switches the cap off. */
  capacity: number;
  epochs: number;
  seed: number;
}

export interface MoeParams {
  experts: number;
  topK: number;
  /** VOCAB × EMBED. */
  embed: Float64Array;
  /** INPUT × experts: entry (i, e) at i · experts + e. */
  router: Float64Array;
  /** experts × INPUT × VOCAB: entry (e, i, v) at (e · INPUT + i) · VOCAB + v. */
  weights: Float64Array;
  /** Balancing bias, one per expert. It moves choices and never gates. */
  bias: Float64Array;
}

/** Bounds every run respects, whatever it was asked for, so no input can request an unbounded run. */
export function safeSettings(settings: MoeSettings): MoeSettings {
  const whole = (value: number, low: number, high: number) =>
    Number.isFinite(value) ? Math.min(high, Math.max(low, Math.round(value))) : low;
  const experts = whole(settings.experts, 1, MAX_EXPERTS);
  const real = (value: number, high: number) => (Number.isFinite(value) ? Math.min(high, Math.max(0, value)) : 0);
  return {
    experts,
    topK: whole(settings.topK, 1, Math.min(MAX_TOP_K, experts)),
    balancing: settings.balancing === "aux" || settings.balancing === "bias" ? settings.balancing : "none",
    auxWeight: real(settings.auxWeight, 10),
    biasStep: real(settings.biasStep, 1),
    capacity: real(settings.capacity, 8),
    epochs: whole(settings.epochs, 0, 60),
    seed: Number.isFinite(settings.seed) ? Math.round(settings.seed) : 1,
  };
}

/**
 * Starting weights, drawn uniformly from ±scale in a fixed order (embedding, router, experts) from the
 * seed. The router's bias row, the one the constant 1 multiplies, starts at zero, as biases usually do:
 * a random router bias would hand most characters to one expert before any training, and collapse would
 * look like bad luck at the start instead of something training does.
 */
export function initParams(
  experts: number,
  topK: number,
  seed: number,
  scale: Readonly<InitScale> = INIT_SCALE,
): { params: MoeParams; random: () => number } {
  const random = tinyRandom(seed);
  const draw = (length: number, spread: number) => {
    const values = new Float64Array(length);
    for (let index = 0; index < length; index += 1) values[index] = (random() * 2 - 1) * spread;
    return values;
  };
  const params: MoeParams = {
    experts,
    topK,
    embed: draw(VOCAB * EMBED, scale.embed),
    router: draw(INPUT * experts, scale.router),
    weights: draw(experts * INPUT * VOCAB, scale.expert),
    bias: new Float64Array(experts),
  };
  params.router.fill(0, EMBED * experts);
  return { params, random };
}

/** Total weights stored, and the weights one token uses: the embedding, the whole router, k experts. */
export function parameterCounts(experts: number, topK: number) {
  const embedding = VOCAB * EMBED;
  const router = INPUT * experts;
  const perExpert = INPUT * VOCAB;
  return {
    embedding,
    router,
    perExpert,
    total: embedding + router + experts * perExpert,
    active: embedding + router + Math.min(topK, experts) * perExpert,
  };
}

/** Seats per expert in a batch of `batch` pairs: ⌈cf · B · k / N⌉, or Infinity with the cap off. */
export function capacitySeats(capacity: number, batch: number, topK: number, experts: number): number {
  return capacity > 0 ? Math.ceil((capacity * batch * topK) / experts) : Number.POSITIVE_INFINITY;
}

/* -------------------------------------------------------------------------- */
/* One batch: routing, capacity, loss and gradients                           */
/* -------------------------------------------------------------------------- */

/** Scratch space for one batch, reused across batches so a run allocates almost nothing. */
interface Workspace {
  h: Float64Array; // B × INPUT
  z: Float64Array; // B × N
  p: Float64Array; // B × N
  chosen: Int32Array; // B × k, best first
  kept: Uint8Array; // B × k
  logits: Float64Array; // VOCAB
  q: Float64Array; // VOCAB
  expertOut: Float64Array; // k × VOCAB
  gradEmbed: Float64Array;
  gradRouter: Float64Array;
  gradWeights: Float64Array;
  seats: Float64Array; // N
  firstChoice: Float64Array; // N
  choices: Float64Array; // N
}

function workspace(experts: number, topK: number): Workspace {
  return {
    h: new Float64Array(BATCH * INPUT),
    z: new Float64Array(BATCH * experts),
    p: new Float64Array(BATCH * experts),
    chosen: new Int32Array(BATCH * topK),
    kept: new Uint8Array(BATCH * topK),
    logits: new Float64Array(VOCAB),
    q: new Float64Array(VOCAB),
    expertOut: new Float64Array(topK * VOCAB),
    gradEmbed: new Float64Array(VOCAB * EMBED),
    gradRouter: new Float64Array(INPUT * experts),
    gradWeights: new Float64Array(experts * INPUT * VOCAB),
    seats: new Float64Array(experts),
    firstChoice: new Float64Array(experts),
    choices: new Float64Array(experts),
  };
}

/**
 * Routes the `size` pairs listed in `order` from `start`: fills h, z, p, the chosen experts (top k of
 * z + b, lower index first on a tie) and which choices keep a seat under the capacity cap.
 */
function routeBatch(
  params: MoeParams,
  data: MoeData,
  order: ArrayLike<number>,
  start: number,
  size: number,
  capacity: number,
  space: Workspace,
) {
  const { experts: N, topK: k, embed, router, bias } = params;
  const { h, z, p, chosen, kept, seats, firstChoice, choices } = space;
  firstChoice.fill(0);
  choices.fill(0);
  for (let t = 0; t < size; t += 1) {
    const x = data.inputs[order[start + t]];
    const hRow = t * INPUT;
    for (let i = 0; i < EMBED; i += 1) h[hRow + i] = embed[x * EMBED + i];
    h[hRow + EMBED] = 1;
    const zRow = t * N;
    let top = -Infinity;
    for (let e = 0; e < N; e += 1) {
      let sum = 0;
      for (let i = 0; i < INPUT; i += 1) sum += h[hRow + i] * router[i * N + e];
      z[zRow + e] = sum;
      if (sum > top) top = sum;
    }
    let total = 0;
    for (let e = 0; e < N; e += 1) {
      const value = Math.exp(z[zRow + e] - top);
      p[zRow + e] = value;
      total += value;
    }
    for (let e = 0; e < N; e += 1) p[zRow + e] /= total;
    // Top k of z + b, best first. k is at most 2, so two linear scans are enough.
    for (let slot = 0; slot < k; slot += 1) {
      let best = -1;
      let bestScore = -Infinity;
      for (let e = 0; e < N; e += 1) {
        let taken = false;
        for (let earlier = 0; earlier < slot; earlier += 1) if (chosen[t * k + earlier] === e) taken = true;
        if (taken) continue;
        const score = z[zRow + e] + bias[e];
        if (best < 0 || score > bestScore) {
          best = e;
          bestScore = score;
        }
      }
      chosen[t * k + slot] = best;
      choices[best] += 1;
    }
    firstChoice[chosen[t * k]] += 1;
  }
  assignSeats(chosen, kept, size, k, seats.fill(capacitySeats(capacity, size, k, N)));
}

/**
 * Seats the choices of one batch: every token's first choice in batch order, then every second
 * choice, as GShard does. `chosen` holds k experts per token, best first; `seats` holds each expert's
 * free seats and is used up; `kept` receives 1 for a seated choice and 0 for a dropped one.
 */
export function assignSeats(
  chosen: ArrayLike<number>,
  kept: Uint8Array,
  size: number,
  k: number,
  seats: Float64Array,
): Uint8Array {
  for (let slot = 0; slot < k; slot += 1) {
    for (let t = 0; t < size; t += 1) {
      const e = chosen[t * k + slot];
      if (seats[e] > 0) {
        seats[e] -= 1;
        kept[t * k + slot] = 1;
      } else {
        kept[t * k + slot] = 0;
      }
    }
  }
  return kept;
}

/** Cross-entropy of token t given the routing in `space`; leaves q (the prediction) and each kept expert's output. */
function tokenLoss(params: MoeParams, space: Workspace, t: number, target: number): number {
  const { experts: N, topK: k, weights } = params;
  const { h, p, chosen, kept, logits, q, expertOut } = space;
  const hRow = t * INPUT;
  logits.fill(0);
  for (let slot = 0; slot < k; slot += 1) {
    if (!kept[t * k + slot]) continue;
    const e = chosen[t * k + slot];
    const gate = p[t * N + e];
    const out = slot * VOCAB;
    for (let v = 0; v < VOCAB; v += 1) {
      let sum = 0;
      for (let i = 0; i < INPUT; i += 1) sum += h[hRow + i] * weights[(e * INPUT + i) * VOCAB + v];
      expertOut[out + v] = sum;
      logits[v] += gate * sum;
    }
  }
  let top = -Infinity;
  for (let v = 0; v < VOCAB; v += 1) if (logits[v] > top) top = logits[v];
  let total = 0;
  for (let v = 0; v < VOCAB; v += 1) {
    q[v] = Math.exp(logits[v] - top);
    total += q[v];
  }
  for (let v = 0; v < VOCAB; v += 1) q[v] /= total;
  return -Math.log(Math.max(1e-300, q[target]));
}

/**
 * The batch objective (mean cross-entropy plus the auxiliary term) and, when `gradients` is true, its
 * gradient with respect to the embedding, the router and the experts. The routing, the seats and the
 * first-choice shares f are held fixed, as in every MoE implementation: they are counts, not functions
 * the gradient can pass through.
 */
function batchObjective(
  params: MoeParams,
  data: MoeData,
  order: ArrayLike<number>,
  start: number,
  size: number,
  settings: Pick<MoeSettings, "balancing" | "auxWeight" | "capacity">,
  space: Workspace,
  gradients: boolean,
) {
  const { experts: N, topK: k, router, weights } = params;
  routeBatch(params, data, order, start, size, settings.capacity, space);
  const { h, p, chosen, kept, q, expertOut, gradEmbed, gradRouter, gradWeights, firstChoice } = space;
  const auxWeight = settings.balancing === "aux" ? settings.auxWeight : 0;
  if (gradients) {
    gradEmbed.fill(0);
    gradRouter.fill(0);
    gradWeights.fill(0);
  }
  const inverse = 1 / size;
  // P_e: the batch's mean gate value for each expert. f_e: its share of first choices.
  let balance = 0;
  for (let e = 0; e < N; e += 1) {
    let mean = 0;
    for (let t = 0; t < size; t += 1) mean += p[t * N + e];
    balance += (firstChoice[e] * inverse) * (mean * inverse);
  }
  balance *= N;
  let loss = 0;
  const a = new Float64Array(N);
  const dz = new Float64Array(N);
  const dh = new Float64Array(INPUT);
  for (let t = 0; t < size; t += 1) {
    const pair = order[start + t];
    const target = data.targets[pair];
    loss += tokenLoss(params, space, t, target);
    if (!gradients) continue;
    const hRow = t * INPUT;
    const pRow = t * N;
    // g = q − onehot(target), the gradient of the cross-entropy with respect to the logits.
    q[target] -= 1;
    a.fill(0);
    dh.fill(0);
    for (let slot = 0; slot < k; slot += 1) {
      if (!kept[t * k + slot]) continue;
      const e = chosen[t * k + slot];
      const gate = p[pRow + e];
      const out = slot * VOCAB;
      let along = 0;
      for (let v = 0; v < VOCAB; v += 1) along += q[v] * expertOut[out + v];
      a[e] = along;
      for (let i = 0; i < INPUT; i += 1) {
        const scaled = gate * h[hRow + i] * inverse;
        const row = (e * INPUT + i) * VOCAB;
        let back = 0;
        for (let v = 0; v < VOCAB; v += 1) {
          gradWeights[row + v] += scaled * q[v];
          back += weights[row + v] * q[v];
        }
        dh[i] += gate * back * inverse;
      }
    }
    // Through the softmax gate: dz_j = p_j (a_j − Σ a_e p_e), plus the auxiliary term's own pull.
    let meanA = 0;
    let meanF = 0;
    for (let e = 0; e < N; e += 1) {
      meanA += a[e] * p[pRow + e];
      meanF += firstChoice[e] * inverse * p[pRow + e];
    }
    for (let j = 0; j < N; j += 1) {
      const pj = p[pRow + j];
      dz[j] = pj * (a[j] - meanA) * inverse + auxWeight * N * inverse * pj * (firstChoice[j] * inverse - meanF);
    }
    for (let i = 0; i < INPUT; i += 1) {
      let back = 0;
      for (let j = 0; j < N; j += 1) {
        gradRouter[i * N + j] += h[hRow + i] * dz[j];
        back += router[i * N + j] * dz[j];
      }
      dh[i] += back;
    }
    const x = data.inputs[pair];
    for (let i = 0; i < EMBED; i += 1) gradEmbed[x * EMBED + i] += dh[i];
  }
  return { loss: loss * inverse, balance, objective: loss * inverse + auxWeight * balance };
}

/**
 * One batch's objective and copies of its gradients, for checking the hand-written backward pass
 * against finite differences. Training itself reuses one workspace instead.
 */
export function batchGradients(
  params: MoeParams,
  data: MoeData,
  order: ArrayLike<number>,
  start: number,
  size: number,
  settings: Pick<MoeSettings, "balancing" | "auxWeight" | "capacity">,
) {
  const space = workspace(params.experts, params.topK);
  const result = batchObjective(params, data, order, start, size, settings, space, true);
  return {
    ...result,
    embed: Float64Array.from(space.gradEmbed),
    router: Float64Array.from(space.gradRouter),
    weights: Float64Array.from(space.gradWeights),
    kept: Uint8Array.from(space.kept.subarray(0, size * params.topK)),
    chosen: Int32Array.from(space.chosen.subarray(0, size * params.topK)),
    gates: Float64Array.from(space.p.subarray(0, size * params.experts)),
  };
}

/** The same batch's objective without gradients, for the finite differences. */
export function batchLoss(
  params: MoeParams,
  data: MoeData,
  order: ArrayLike<number>,
  start: number,
  size: number,
  settings: Pick<MoeSettings, "balancing" | "auxWeight" | "capacity">,
) {
  return batchObjective(params, data, order, start, size, settings, workspace(params.experts, params.topK), false);
}

/* -------------------------------------------------------------------------- */
/* Evaluation: one pass of fixed weights over every pair                       */
/* -------------------------------------------------------------------------- */

export interface MoeEvaluation {
  /** Mean cross-entropy over every pair, in nats. Dropped tokens count at ln 30. */
  loss: number;
  /** Each expert's share of the router's choices: choices / (pairs · k). Sums to 1. */
  shares: number[];
  /** exp(−Σ s ln s) over the shares: 1 when one expert takes everything, N when all are even. */
  effective: number;
  /** N · Σ f_e · P_e over every pair: 1 when balanced, N when one expert takes everything. */
  balance: number;
  /** Share of choices that found their expert full. */
  dropped: number;
  /** Share of tokens that lost every expert and predicted uniformly. */
  unserved: number;
  /** Mean gate value of the experts that ran. */
  meanGate: number;
}

export interface MoeRouting extends MoeEvaluation {
  /** Choices per expert for each character as the input, VOCAB × N. */
  byCharacter: number[][];
  /** Choices per expert for each corpus, 3 × N. */
  byCorpus: number[][];
}

/**
 * Scores fixed weights on every pair in corpus order, 16 at a time, with the same routing and the same
 * capacity rule as training. Nothing is updated.
 */
export function evaluate(params: MoeParams, capacity: number, data: MoeData = moeData()): MoeRouting {
  const { experts: N, topK: k } = params;
  const space = workspace(N, k);
  const identity = new Int32Array(data.count);
  for (let index = 0; index < data.count; index += 1) identity[index] = index;
  const choices = new Float64Array(N);
  const firstChoices = new Float64Array(N);
  const gateSums = new Float64Array(N);
  const byCharacter = Array.from({ length: VOCAB }, () => new Array<number>(N).fill(0));
  const byCorpus = CORPUS_IDS.map(() => new Array<number>(N).fill(0));
  let loss = 0;
  let dropped = 0;
  let unserved = 0;
  let gateTotal = 0;
  let ran = 0;
  for (let start = 0; start < data.count; start += BATCH) {
    const size = Math.min(BATCH, data.count - start);
    routeBatch(params, data, identity, start, size, capacity, space);
    for (let t = 0; t < size; t += 1) {
      const pair = start + t;
      loss += tokenLoss(params, space, t, data.targets[pair]);
      let served = false;
      for (let slot = 0; slot < k; slot += 1) {
        const e = space.chosen[t * k + slot];
        choices[e] += 1;
        byCharacter[data.inputs[pair]][e] += 1;
        byCorpus[data.corpus[pair]][e] += 1;
        if (space.kept[t * k + slot]) {
          served = true;
          gateTotal += space.p[t * N + e];
          ran += 1;
        } else {
          dropped += 1;
        }
      }
      if (!served) unserved += 1;
      firstChoices[space.chosen[t * k]] += 1;
      for (let e = 0; e < N; e += 1) gateSums[e] += space.p[t * N + e];
    }
  }
  const assignments = data.count * k;
  const shares = Array.from(choices, (value) => value / assignments);
  const entropy = shares.reduce((sum, share) => (share > 0 ? sum - share * Math.log(share) : sum), 0);
  let balance = 0;
  for (let e = 0; e < N; e += 1) balance += (firstChoices[e] / data.count) * (gateSums[e] / data.count);
  return {
    loss: loss / data.count,
    shares,
    effective: Math.exp(entropy),
    balance: balance * N,
    dropped: dropped / assignments,
    unserved: unserved / data.count,
    meanGate: ran > 0 ? gateTotal / ran : 0,
    byCharacter,
    byCorpus,
  };
}

/* -------------------------------------------------------------------------- */
/* A whole run                                                                 */
/* -------------------------------------------------------------------------- */

export interface MoeRun {
  settings: MoeSettings;
  params: MoeParams;
  /** One evaluation per epoch boundary, epoch 0 (the starting weights) to the last. */
  history: MoeEvaluation[];
  /** The final weights' evaluation, with the routing tables. */
  final: MoeRouting;
  /** The balancing bias after every epoch, epoch 0 to the last. */
  biasHistory: number[][];
  steps: number;
  pairs: number;
}

export function shuffledOrder(count: number, random: () => number): Int32Array {
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

const strip = ({ byCharacter: _character, byCorpus: _corpus, ...rest }: MoeRouting): MoeEvaluation => rest;

/** `scale` exists for the tests that check how much the collapse depends on the starting weights. */
export function trainMoe(requested: MoeSettings, scale: Readonly<InitScale> = INIT_SCALE): MoeRun {
  const settings = safeSettings(requested);
  const { experts: N, topK: k, epochs } = settings;
  const data = moeData();
  const { params, random } = initParams(N, k, settings.seed, scale);
  const space = workspace(N, k);
  const history: MoeEvaluation[] = [strip(evaluate(params, settings.capacity, data))];
  const biasHistory: number[][] = [Array.from(params.bias)];
  const { embed, router, weights, bias } = params;
  let steps = 0;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const order = shuffledOrder(data.count, random);
    for (let start = 0; start < data.count; start += BATCH) {
      const size = Math.min(BATCH, data.count - start);
      batchObjective(params, data, order, start, size, settings, space, true);
      for (let index = 0; index < embed.length; index += 1) embed[index] -= LEARNING_RATE * space.gradEmbed[index];
      for (let index = 0; index < router.length; index += 1) router[index] -= LEARNING_RATE * space.gradRouter[index];
      for (let index = 0; index < weights.length; index += 1) weights[index] -= LEARNING_RATE * space.gradWeights[index];
      if (settings.balancing === "bias" && settings.biasStep > 0) {
        const even = (size * k) / N;
        for (let e = 0; e < N; e += 1) bias[e] += settings.biasStep * Math.sign(even - space.choices[e]);
      }
      steps += 1;
    }
    // The last epoch boundary is the final evaluation below, which also builds the routing tables.
    if (epoch < epochs - 1) history.push(strip(evaluate(params, settings.capacity, data)));
    biasHistory.push(Array.from(bias));
  }
  const final = evaluate(params, settings.capacity, data);
  if (epochs > 0) history.push(strip(final));
  return { settings, params, history, final, biasHistory, steps, pairs: data.count };
}

/** The dense baseline: one expert, so every token runs the same 5 × 30 map with a gate of exactly 1. */
export function denseSettings(settings: Pick<MoeSettings, "epochs" | "seed">): MoeSettings {
  return { experts: 1, topK: 1, balancing: "none", auxWeight: 0, biasStep: 0, capacity: 0, epochs: settings.epochs, seed: settings.seed };
}

export const uniformLoss = UNIFORM_CROSS_ENTROPY;

/**
 * The router's view of one input character with trained weights: its input h, every expert's score z
 * and gate p, the balancing bias, which experts run, and the next-character prediction they make
 * together (no capacity cap: one token alone never fills a seat).
 */
export function routeCharacter(params: MoeParams, character: number) {
  const { experts: N, topK: k, embed, router, weights, bias } = params;
  const h = [...Array.from(embed.subarray(character * EMBED, character * EMBED + EMBED)), 1];
  const z = Array.from({ length: N }, (_, e) => h.reduce((sum, value, i) => sum + value * router[i * N + e], 0));
  const top = Math.max(...z);
  const exps = z.map((value) => Math.exp(value - top));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const gates = exps.map((value) => value / total);
  const chosen = z
    .map((score, e) => ({ e, score: score + bias[e] }))
    .sort((left, right) => right.score - left.score || left.e - right.e)
    .slice(0, k)
    .map((entry) => entry.e);
  const logits = new Array<number>(VOCAB).fill(0);
  for (const e of chosen) {
    for (let v = 0; v < VOCAB; v += 1) {
      let sum = 0;
      for (let i = 0; i < INPUT; i += 1) sum += h[i] * weights[(e * INPUT + i) * VOCAB + v];
      logits[v] += gates[e] * sum;
    }
  }
  const peak = Math.max(...logits);
  const odds = logits.map((value) => Math.exp(value - peak));
  const sum = odds.reduce((acc, value) => acc + value, 0);
  return { h, z, bias: Array.from(bias), gates, chosen, next: odds.map((value) => value / sum) };
}

/**
 * Published totals for the "at real scale" row, authored rather than computed: Mistral AI's announcement
 * gives 46.7B parameters in total and 12.9B used per token (the paper rounds them to 47B and 13B), with
 * a router choosing 2 of 8 experts in every layer.
 */
export const MIXTRAL_8X7B = { name: "Mixtral 8x7B", total: 46.7e9, active: 12.9e9, experts: 8, topK: 2 } as const;
