/**
 * Recursive training on generated text, computed exactly.
 *
 * The model is the same 30 x 30 character-bigram table the rest of the track
 * trains with SGD, but here it is fitted in closed form: each row is the
 * normalized count of what followed that character. That is the maximum-
 * likelihood optimum of the table, so every generation is the best possible
 * fit to its training text, and every change between generations comes from
 * the data the previous generation wrote. Nothing below is simulated.
 */
import {
  decodeTinyIds,
  encodeTinyText,
  TINY_CORPORA,
  TINY_RESERVED,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyNextDistribution,
  tinyRandom,
  trainTinyModel,
} from "@app/module-sdk";

const V = TINY_VOCAB_SIZE;
const SPACE = 0;

export const SEED_TEXT = TINY_CORPORA.harbor.text;
export const SEED_IDS = encodeTinyText(SEED_TEXT);
export const GENERATIONS = 10;
/** Independent sampling seeds per regime. The chart draws every one of them. */
export const RUN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8] as const;
/** A verifier may need many drafts per kept character; this caps the work. */
export const DRAFT_BUDGET = 40;

export type Regime = "replace" | "accumulate";
export type FilterKind = "none" | "words";
/** Who writes each generation's training text: the model fitted last round, or a fixed stronger teacher. */
export type Generator = "self" | "teacher";

/**
 * More harbor notes, written by hand in the seed's style, that only the teacher was trained on. The
 * student never sees them. Together with the seed they are about four times as much text.
 */
export const TEACHER_EXTRA_TEXT = `a gull circles the pier and the fishing boats rock in the swell.
the lighthouse beam sweeps across the water as the fog thickens.
ropes creak against the dock while the tide pulls the hulls toward the sea.
by afternoon the wind turns north and the sky grows heavy with clouds.
the market opens at dawn and the smell of salt and bread drifts down the street.
children watch the ferry leave and wave until it disappears behind the point.
in autumn the leaves gather in the gutters and the rain comes in from the west.
the harbor master writes the forecast on a board beside the old stone gate.
a thin mist hangs over the quiet water and the lamps burn late into the night.
when the storm breaks the waves climb the sea wall and spray the empty street.
the baker lights the oven before sunrise and the warm air fills the narrow lane.
fishermen mend their nets on the quay and talk about the weather ahead.
the town clock strikes noon and the bells echo across the harbor and the hills.
a cold front moves through at night and the morning brings clear sky and frost.
boats return at dusk with the catch and the gulls follow them into port.
the sea turns grey beneath the clouds and the horizon fades into the mist.
in spring the snow melts from the roofs and the streams run down to the water.
the pilot reads the wind and guides the ship past the rocks into the bay.
lanterns glow along the pier while the last ferry crosses the dark water.
a warm breeze carries the scent of rain over the roofs and the stone streets.`;

/** Harbor-style sentences that appear in no training set, the student's or the teacher's. They score what was learned. */
export const HELD_OUT_TEXT = `the fog lifts from the bay and the boats leave the harbor one by one.
a cool wind comes in from the sea and the clouds gather over the town.
the rain stops before noon and the sun warms the wet stone streets.
in winter the water stays cold and the harbor is quiet until the spring.
the ferry waits at the pier while the fishermen carry the catch to the market.
storm clouds build over the hills and the wind drops before the rain arrives.
the evening mist settles on the water and the lamps come on along the quay.
a clear sky follows the storm and the sea is calm again by the morning.`;
export const HELD_OUT_IDS = encodeTinyText(HELD_OUT_TEXT);

/** How the teacher is fitted with the SDK trainer: minibatch SGD at a rate near the sweet spot the hyperparameters lab finds. */
export const TEACHER_TRAINING = { epochs: 40, batchSize: 16, learningRate: 6, seed: 5 } as const;

let teacherCache: Float64Array | null = null;

/**
 * The teacher: a bigram trained by the SDK's SGD trainer on the seed plus the extra notes. SGD never
 * drives a probability exactly to zero, so it keeps a little mass on pairs its data never contained;
 * `teacherUnseenMass` measures how much. The reserved angle brackets are masked and each row renormalized.
 */
export function teacherModel(): Float64Array {
  if (teacherCache) return teacherCache;
  const run = trainTinyModel({ text: `${SEED_TEXT} ${TEACHER_EXTRA_TEXT}`, ...TEACHER_TRAINING });
  const probabilities = new Float64Array(V * V);
  const reserved = new Set(Array.from(TINY_RESERVED, (character) => TINY_VOCAB.indexOf(character)));
  for (let row = 0; row < V; row += 1) {
    const distribution = tinyNextDistribution(run.weights, TINY_VOCAB[row]);
    let total = 0;
    for (let next = 0; next < V; next += 1) if (!reserved.has(next)) total += distribution[next];
    for (let next = 0; next < V; next += 1) {
      probabilities[row * V + next] = reserved.has(next) ? 0 : distribution[next] / total;
    }
  }
  teacherCache = probabilities;
  return probabilities;
}

/** Share of the held-out text's transitions that the counts can produce at all. */
export function heldOutCoverage(counts: Float64Array): number {
  let covered = 0;
  let total = 0;
  for (let index = 0; index + 1 < HELD_OUT_IDS.length; index += 1) {
    total += 1;
    if (counts[HELD_OUT_IDS[index] * V + HELD_OUT_IDS[index + 1]] > 0) covered += 1;
  }
  return total === 0 ? 0 : covered / total;
}

/** Average probability the teacher puts on pairs absent from its training text, over the rows that text uses. */
export function teacherUnseenMass(): number {
  const known = countTransitions(encodeTinyText(`${SEED_TEXT} ${TEACHER_EXTRA_TEXT}`));
  const probabilities = teacherModel();
  let mass = 0;
  let rows = 0;
  for (let row = 0; row < V; row += 1) {
    let used = false;
    for (let next = 0; next < V; next += 1) if (known[row * V + next] > 0) used = true;
    if (!used) continue;
    rows += 1;
    for (let next = 0; next < V; next += 1) if (known[row * V + next] === 0) mass += probabilities[row * V + next];
  }
  return rows === 0 ? 0 : mass / rows;
}

/** Every word in the human seed. The verifier accepts only these. */
export const SEED_WORDS: ReadonlySet<string> = new Set(
  decodeTinyIds(SEED_IDS)
    .split(/[ .]+/)
    .filter(Boolean),
);

export function countTransitions(ids: ReadonlyArray<number>): Float64Array {
  const counts = new Float64Array(V * V);
  for (let index = 0; index + 1 < ids.length; index += 1) {
    counts[ids[index] * V + ids[index + 1]] += 1;
  }
  return counts;
}

export const SEED_COUNTS = countTransitions(SEED_IDS);

/**
 * Row-normalized counts. A character with no recorded successor (at most the
 * last character of the training text) falls back to emitting a space so the
 * chain can move on. Support, coverage, and tail loss read the counts, so the
 * fallback never counts as something the model learned.
 */
export function fitCounts(counts: Float64Array): Float64Array {
  const probabilities = new Float64Array(V * V);
  for (let row = 0; row < V; row += 1) {
    let total = 0;
    for (let next = 0; next < V; next += 1) total += counts[row * V + next];
    if (total === 0) {
      probabilities[row * V + SPACE] = 1;
      continue;
    }
    for (let next = 0; next < V; next += 1) {
      probabilities[row * V + next] = counts[row * V + next] / total;
    }
  }
  return probabilities;
}

/** Sparse cumulative rows after temperature: p^(1/T), renormalized. */
function samplerRows(probabilities: Float64Array, temperature: number) {
  const inverse = 1 / Math.max(0.05, temperature);
  return Array.from({ length: V }, (_, row) => {
    const ids: number[] = [];
    const weights: number[] = [];
    let total = 0;
    for (let next = 0; next < V; next += 1) {
      const probability = probabilities[row * V + next];
      if (probability <= 0) continue;
      const weight = probability ** inverse;
      ids.push(next);
      weights.push(weight);
      total += weight;
    }
    let running = 0;
    const cumulative = weights.map((weight) => (running += weight / total));
    return { ids, cumulative };
  });
}

export function sampleIds(
  probabilities: Float64Array,
  length: number,
  temperature: number,
  random: () => number,
  start = SPACE,
): number[] {
  const rows = samplerRows(probabilities, temperature);
  const ids: number[] = [];
  let current = start;
  for (let index = 0; index < length; index += 1) {
    const { ids: choices, cumulative } = rows[current];
    const draw = random();
    let chosen = choices[choices.length - 1] ?? SPACE;
    for (let k = 0; k < cumulative.length; k += 1) {
      if (draw <= cumulative[k]) {
        chosen = choices[k];
        break;
      }
    }
    ids.push(chosen);
    current = chosen;
  }
  return ids;
}

const stripWord = (token: string) => token.replace(/\./g, "");

/** True when a space-separated token is a word that occurs in the seed. */
export function isSeedWord(token: string) {
  const word = stripWord(token);
  return word.length > 0 && SEED_WORDS.has(word);
}

/** The verifier: keep only tokens that are real seed words, in the order drafted. */
export function verifyWords(text: string): string {
  return text
    .split(" ")
    .filter((token) => isSeedWord(token))
    .join(" ");
}

/** Share of word tokens in a text that are real seed words. */
export function realWordShare(text: string): number {
  const tokens = text.split(/[ .]+/).filter(Boolean);
  if (tokens.length === 0) return 0;
  return tokens.filter((token) => SEED_WORDS.has(token)).length / tokens.length;
}

export interface GenerationStats {
  generation: number;
  /** Distinct (context, next) transitions the model can emit. */
  support: number;
  /** Share of the seed's transition tokens the model can still emit. */
  coverage: number;
  /** Next-character entropy in nats, averaged over contexts as the seed uses them. */
  entropy: number;
  /** Share of words in this generation's own writing that are real seed words. */
  realWords: number;
  /** Share of the held-out harbor text's transitions that the training counts can produce. */
  heldOut: number;
  /** Characters this generation was trained on (the seed for generation 0). */
  trainedChars: number;
  /** Share of drafted characters the filter accepted when building this generation's data. */
  keptShare: number;
}

/** Distinct transitions present in the training counts, which is what the fit can emit from evidence. */
export function supportSize(counts: Float64Array): number {
  let size = 0;
  for (const value of counts) if (value > 0) size += 1;
  return size;
}

/** Share of the seed's transition tokens whose transition still has evidence in the training counts. */
export function seedCoverage(counts: Float64Array): number {
  let total = 0;
  let covered = 0;
  for (let index = 0; index < V * V; index += 1) {
    const count = SEED_COUNTS[index];
    if (count === 0) continue;
    total += count;
    if (counts[index] > 0) covered += count;
  }
  return total === 0 ? 0 : covered / total;
}

export function seedWeightedEntropy(probabilities: Float64Array): number {
  let entropy = 0;
  let total = 0;
  for (let row = 0; row < V; row += 1) {
    let uses = 0;
    for (let next = 0; next < V; next += 1) uses += SEED_COUNTS[row * V + next];
    if (uses === 0) continue;
    let rowEntropy = 0;
    for (let next = 0; next < V; next += 1) {
      const probability = probabilities[row * V + next];
      if (probability > 0) rowEntropy -= probability * Math.log(probability);
    }
    entropy += uses * rowEntropy;
    total += uses;
  }
  return total === 0 ? 0 : entropy / total;
}

export interface ChainOptions {
  regime: Regime;
  temperature: number;
  filter: FilterKind;
  /** Defaults to the model writing its own successor's data. */
  generator?: Generator;
  sampleSize: number;
  generations?: number;
  seed: number;
  probeLength?: number;
}

export interface ChainRun {
  stats: GenerationStats[];
  models: Float64Array[];
  /** The training counts behind each generation's fit. */
  counts: Float64Array[];
  /** What each generation writes at the chosen temperature, before any filter. */
  probes: string[];
}

/** Draft text from a model until the verifier has kept `size` characters. */
function draftTrainingText(
  probabilities: Float64Array,
  size: number,
  temperature: number,
  filter: FilterKind,
  random: () => number,
) {
  if (filter === "none") {
    const ids = sampleIds(probabilities, size, temperature, random);
    return { ids, acceptance: 1 };
  }
  const kept: string[] = [];
  let keptLength = 0;
  let drafted = 0;
  const chunk = Math.max(200, size);
  while (keptLength < size && drafted < size * DRAFT_BUDGET) {
    const draft = decodeTinyIds(sampleIds(probabilities, chunk, temperature, random));
    drafted += chunk;
    const accepted = verifyWords(draft);
    if (accepted) {
      kept.push(accepted);
      keptLength += accepted.length + 1;
    }
  }
  const ids = encodeTinyText(kept.join(" ")).slice(0, size);
  return { ids, acceptance: drafted === 0 ? 0 : Math.min(1, keptLength / drafted) };
}

export function runChain({
  regime,
  temperature,
  filter,
  generator = "self",
  sampleSize,
  generations = GENERATIONS,
  seed,
  probeLength = 2000,
}: ChainOptions): ChainRun {
  const random = tinyRandom(seed * 7919 + 17);
  const probeRandom = tinyRandom(seed * 104729 + 3);
  let data: Float64Array = Float64Array.from(SEED_COUNTS);
  let model = fitCounts(data);
  const models = [model];
  const counts = [data];
  const probes: string[] = [];
  const stats: GenerationStats[] = [];

  const record = (generation: number, trainedChars: number, keptShare: number) => {
    const probe = decodeTinyIds(sampleIds(model, probeLength, temperature, probeRandom));
    probes.push(probe);
    stats.push({
      generation,
      support: supportSize(data),
      coverage: seedCoverage(data),
      entropy: seedWeightedEntropy(model),
      realWords: realWordShare(probe),
      heldOut: heldOutCoverage(data),
      trainedChars,
      keptShare,
    });
  };

  record(0, SEED_IDS.length, 1);
  for (let generation = 1; generation <= generations; generation += 1) {
    // A teacher is fixed, so nothing the student gets wrong can reach the next round's text.
    const writer = generator === "teacher" ? teacherModel() : model;
    const { ids, acceptance } = draftTrainingText(writer, sampleSize, temperature, filter, random);
    const fresh = countTransitions(ids);
    if (regime === "replace") {
      data = fresh;
    } else {
      const merged = new Float64Array(V * V);
      for (let index = 0; index < merged.length; index += 1) merged[index] = data[index] + fresh[index];
      data = merged;
    }
    model = fitCounts(data);
    models.push(model);
    counts.push(data);
    record(generation, ids.length, acceptance);
  }
  return { stats, models, counts, probes };
}

export interface ChainFamily {
  runs: ChainRun[];
  mean: GenerationStats[];
}

export function runFamily(options: Omit<ChainOptions, "seed">): ChainFamily {
  const runs = RUN_SEEDS.map((seed) => runChain({ ...options, seed }));
  const count = runs.length;
  const mean = runs[0].stats.map((_, generation) => {
    const average = (key: keyof GenerationStats) =>
      runs.reduce((sum, run) => sum + run.stats[generation][key], 0) / count;
    return {
      generation,
      support: average("support"),
      coverage: average("coverage"),
      entropy: average("entropy"),
      realWords: average("realWords"),
      heldOut: average("heldOut"),
      trainedChars: average("trainedChars"),
      keptShare: average("keptShare"),
    };
  });
  return { runs, mean };
}

export interface TailBucket {
  id: "once" | "few" | "common";
  label: string;
  total: number;
  lost: number;
}

/** How many of the seed's transitions are gone from the training counts, grouped by how often the seed used them. */
export function tailLoss(counts: Float64Array): TailBucket[] {
  const buckets: TailBucket[] = [
    { id: "once", label: "seen once", total: 0, lost: 0 },
    { id: "few", label: "seen 2–4 times", total: 0, lost: 0 },
    { id: "common", label: "seen 5+ times", total: 0, lost: 0 },
  ];
  for (let index = 0; index < V * V; index += 1) {
    const count = SEED_COUNTS[index];
    if (count === 0) continue;
    const bucket = count === 1 ? buckets[0] : count < 5 ? buckets[1] : buckets[2];
    bucket.total += 1;
    if (counts[index] <= 0) bucket.lost += 1;
  }
  return buckets;
}

/** The characters the heatmap shows: every symbol the seed can contain. */
export const GRID_IDS = Array.from(TINY_VOCAB, (_, id) => id).filter(
  (id) => TINY_VOCAB[id] !== "<" && TINY_VOCAB[id] !== ">",
);
export const glyph = (id: number) => (id === SPACE ? "␣" : TINY_VOCAB[id]);

/** Splits a sample into space-separated tokens with the verifier's verdict on each. */
export function verdictTokens(text: string): { token: string; real: boolean }[] {
  return text
    .split(" ")
    .filter(Boolean)
    .map((token) => ({ token, real: isSeedWord(token) }));
}

/** Share of a text's characters the verifier would keep, counting one space per kept word. */
export function acceptanceOf(text: string): number {
  if (text.length === 0) return 0;
  const kept = verifyWords(text);
  return kept.length === 0 ? 0 : Math.min(1, (kept.length + 1) / text.length);
}
