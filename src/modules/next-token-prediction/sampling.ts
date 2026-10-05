/**
 * Decoding math for the next-token lab: softmax with temperature, top-k, top-p
 * and min-p truncation, seeded inverse-transform sampling, and surprisal.
 * Everything operates on real logits produced by the module's bigram table,
 * character RNN, or tiny transformer.
 */

export const MODELS = [
  { id: "bigram", label: "Bigram" },
  { id: "rnn", label: "RNN" },
  { id: "transformer", label: "Transformer" },
] as const;

export type ModelId = (typeof MODELS)[number]["id"];

export const VOCAB_SIZE = 66;
export const TRANSFORMER_CONTEXT = 64;
export const MAX_PROMPT = 120;
export const MAX_GENERATED = 120;
/** The largest min-p the lab offers: only characters at least half as likely as the top one survive. */
export const MIN_P_MAX = 0.5;
/** Stored in place of the <unk> token so generated text stays a plain string. */
export const UNKNOWN_CHARACTER = "�";

export const isModelId = (value: unknown): value is ModelId =>
  typeof value === "string" && MODELS.some((model) => model.id === value);

/** Numerically stable log-softmax of logits divided by a temperature. */
export function logSoftmax(logits: ReadonlyArray<number>, temperature = 1): number[] {
  const scaled = logits.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const logSum = Math.log(scaled.reduce((sum, value) => sum + Math.exp(value - max), 0)) + max;
  return scaled.map((value) => value - logSum);
}

export const softmax = (logits: ReadonlyArray<number>, temperature = 1) =>
  logSoftmax(logits, temperature).map(Math.exp);

export function entropyBits(probabilities: ReadonlyArray<number>) {
  return probabilities.reduce((sum, p) => (p > 0 ? sum - p * Math.log2(p) : sum), 0);
}

export const surprisalBits = (probability: number) => -Math.log2(Math.max(probability, Number.MIN_VALUE));

export interface SamplerSettings {
  temperature: number;
  /** Keep the k most likely tokens; k ≥ vocabulary size disables it. */
  topK: number;
  /** Keep the smallest prefix whose mass reaches p; p ≥ 1 disables it. */
  topP: number;
  /**
   * Keep tokens whose probability is at least minP times the top token's probability
   * (Nguyen et al., "Turning Up the Heat: Min-p Sampling", ICLR 2025, section 3.1:
   * p_scaled = p_base × p_max, keep P(v) ≥ p_scaled). 0 or absent disables it.
   */
  minP?: number;
}

export interface Candidate {
  index: number;
  rank: number;
  logit: number;
  /** Model probability at temperature 1, before any truncation. */
  p: number;
  /** Probability after temperature, before truncation. */
  tempered: number;
  kept: boolean;
  cutBy: "top-k" | "top-p" | "min-p" | null;
  /** Final sampling probability: tempered, truncated, renormalised. */
  q: number;
}

export interface ShapedDistribution {
  /** Every vocabulary entry, most likely first. */
  candidates: Candidate[];
  /** Final sampling probability by vocabulary index. */
  q: number[];
  p: number[];
  keptCount: number;
  /** Tempered mass of the kept set before renormalising. */
  keptMass: number;
  /** The min-p cut-off, minP × the top tempered probability; 0 when min-p is off. */
  minPThreshold: number;
}

/**
 * The decoding pipeline: divide logits by T, keep the top k, keep the top-p nucleus of what is
 * left (renormalised), then drop every survivor below the min-p cut-off, and renormalise.
 *
 * Min-p compares each tempered probability with the largest one. Both top-k and top-p always keep
 * the top character, and renormalising rescales every probability by the same factor, so the ratio
 * to the top character, and so the min-p set, is the same whether it is measured before or after them.
 */
export function shapeDistribution(logits: ReadonlyArray<number>, settings: SamplerSettings): ShapedDistribution {
  const temperature = Math.max(1e-3, settings.temperature);
  const p = softmax(logits, 1);
  const tempered = softmax(logits, temperature);
  const order = logits
    .map((logit, index) => ({ logit, index }))
    .sort((a, b) => b.logit - a.logit || a.index - b.index)
    .map((item) => item.index);

  const k = Math.max(1, Math.min(order.length, Math.round(settings.topK)));
  const afterK = order.slice(0, k);
  const massAfterK = afterK.reduce((sum, index) => sum + tempered[index], 0);

  let nucleus = afterK.length;
  if (settings.topP < 1) {
    let cumulative = 0;
    nucleus = 0;
    for (const index of afterK) {
      cumulative += tempered[index] / massAfterK;
      nucleus += 1;
      if (cumulative >= settings.topP - 1e-12) break;
    }
  }
  const afterP = afterK.slice(0, Math.max(1, nucleus));
  const minP = settings.minP !== undefined && Number.isFinite(settings.minP) ? Math.max(0, settings.minP) : 0;
  const minPThreshold = minP > 0 ? minP * tempered[order[0]] : 0;
  const keptSet = new Set(afterP.filter((index, position) => position === 0 || tempered[index] >= minPThreshold));
  const afterPSet = new Set(afterP);
  const keptMass = [...keptSet].reduce((sum, index) => sum + tempered[index], 0);
  const q = tempered.map((value, index) => (keptSet.has(index) ? value / keptMass : 0));

  const candidates = order.map((index, rank): Candidate => ({
    index,
    rank,
    logit: logits[index],
    p: p[index],
    tempered: tempered[index],
    kept: keptSet.has(index),
    cutBy: keptSet.has(index) ? null : rank >= k ? "top-k" : !afterPSet.has(index) ? "top-p" : "min-p",
    q: q[index],
  }));

  return { candidates, q, p, keptCount: keptSet.size, keptMass, minPThreshold };
}

/**
 * A deterministic uniform number in [0, 1) for draw number `index` under `seed`,
 * so the same seed, prompt and settings always reproduce the same text.
 */
export function seededUniform(seed: number, index: number) {
  let h = (Math.imul(Math.round(seed) | 0, 0x9e3779b1) ^ Math.imul(index + 1, 0x85ebca77)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  h = (h ^ (h >>> 15)) >>> 0;
  return h / 4294967296;
}

/** Inverse-transform sampling: walk the kept candidates, most likely first, until the running sum passes u. */
export function drawCandidate(shaped: ShapedDistribution, u: number): Candidate {
  let cumulative = 0;
  let last = shaped.candidates[0];
  for (const candidate of shaped.candidates) {
    if (!candidate.kept) continue;
    last = candidate;
    cumulative += candidate.q;
    if (u < cumulative) return candidate;
  }
  return last;
}

export interface Vocabulary {
  stoi: Record<string, number>;
  itos: string[];
  unk: number;
}

export function encode(text: string, vocabulary: Vocabulary) {
  return Array.from(text).map((character) => vocabulary.stoi[character] ?? vocabulary.unk);
}

export function tokenText(index: number, vocabulary: Vocabulary) {
  return index === vocabulary.unk ? UNKNOWN_CHARACTER : vocabulary.itos[index] ?? UNKNOWN_CHARACTER;
}

/** A visible stand-in for whitespace and the unknown token. */
export function displayToken(character: string) {
  if (character === " ") return "␣";
  if (character === "\n") return "↵";
  if (character === UNKNOWN_CHARACTER) return "<unk>";
  return character;
}

export function describeToken(character: string) {
  if (character === " ") return "space";
  if (character === "\n") return "newline";
  if (character === UNKNOWN_CHARACTER) return "unknown token";
  return `"${character}"`;
}

export interface DrawRecord {
  /** The drawn character. */
  c: string;
  /** Model probability at temperature 1. */
  p: number;
  /** Sampling probability after temperature, top-k and top-p. */
  q: number;
  /** The uniform number that selected it. */
  u: number;
  m: ModelId;
}

const compact = (value: number) => Number(value.toPrecision(6));

/**
 * Module state only holds strings, numbers, booleans and string arrays, so each
 * draw is stored as "model|u|p|q". The drawn character itself is the matching
 * character of the generated text.
 */
export function encodeDraw(record: Omit<DrawRecord, "c">) {
  return `${record.m}|${compact(record.u)}|${compact(record.p)}|${compact(record.q)}`;
}

export function decodeDraw(value: unknown, character: string): DrawRecord | undefined {
  if (typeof value !== "string") return undefined;
  const [m, ...numbers] = value.split("|");
  const [u, p, q] = numbers.map(Number);
  if (!isModelId(m) || numbers.length !== 3 || ![u, p, q].every(Number.isFinite)) return undefined;
  if (u < 0 || u >= 1 || p < 0 || p > 1 || q < 0 || q > 1) return undefined;
  return { c: character, m, u, p, q };
}

export const formatPercent = (value: number) => {
  const percent = value * 100;
  if (percent === 0) return "0%";
  if (percent < 0.1) return "<0.1%";
  return `${percent.toFixed(percent < 10 ? 1 : 0)}%`;
};
