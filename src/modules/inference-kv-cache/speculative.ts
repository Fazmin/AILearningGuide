/**
 * Speculative decoding on toy distributions, with the standard accept/reject rule of
 * Leviathan, Kalman and Matias (2023), "Fast Inference from Transformers via
 * Speculative Decoding", Algorithm 1.
 *
 * The target is a character bigram trained on the Harbor and Recipes text. The
 * draft is the same table with its logits scaled by a quality factor, which flattens
 * every row: quality 1 is the target itself, quality near 0 is guessing. Everything
 * here is computed; nothing is timed, because a character bigram has no forward pass
 * worth saving.
 */

import {
  TINY_CORPORA,
  TINY_RESERVED,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  teachingServing,
  tinyRandom,
  trainTinyModel,
} from "@app/module-sdk";

const V = TINY_VOCAB_SIZE;

export const GAMMA_RANGE = { min: 1, max: 8 };
/** Draft quality in steps of 0.05. Below 0.2 the draft is close to uniform and nothing is accepted. */
export const QUALITY_RANGE = { min: 0.2, max: 1 };
export const SEED_RANGE = { min: 1, max: 999 };

/** Quality in steps of 0.05 between the range limits; anything else becomes the default. */
export const clampQuality = (value: number) =>
  Number.isFinite(value)
    ? Math.min(QUALITY_RANGE.max, Math.max(QUALITY_RANGE.min, Math.round(value * 20) / 20))
    : 0.7;
/** Verification passes in the seeded simulation. */
export const SIM_PASSES = 4000;
/** Passes shown as text in the transcript. */
export const TRANSCRIPT_PASSES = 6;
export const PROMPT = "the ";
export const TARGET_EPOCHS = 50;
export const TARGET_SEED = 1;

const targetText = `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`;

/** One probability row per context character, as a flat V x V table. */
export type Rows = Float64Array;

const reserved = Array.from(TINY_RESERVED, (character) => TINY_VOCAB.indexOf(character));

function normalizeRow(rows: Rows, row: number) {
  let total = 0;
  for (let k = 0; k < V; k += 1) total += rows[row * V + k];
  for (let k = 0; k < V; k += 1) rows[row * V + k] /= total;
}

/** Trains the target bigram and returns its rows with the reserved marker characters removed. */
export function trainTargetRows(): Rows {
  const { weights } = trainTinyModel({ text: targetText, epochs: TARGET_EPOCHS, seed: TARGET_SEED });
  const rows = new Float64Array(V * V);
  for (let row = 0; row < V; row += 1) {
    let maximum = -Infinity;
    for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, weights[row * V + k]);
    for (let k = 0; k < V; k += 1) rows[row * V + k] = Math.exp(weights[row * V + k] - maximum);
    for (const id of reserved) rows[row * V + id] = 0;
    normalizeRow(rows, row);
  }
  return rows;
}

/**
 * The draft: every target row raised to the power `quality` and renormalized, which is
 * what scaling the target's logits by `quality` does. A power below 1 flattens a row.
 */
export function draftRows(target: Rows, quality: number): Rows {
  const rows = new Float64Array(V * V);
  for (let row = 0; row < V; row += 1) {
    for (let k = 0; k < V; k += 1) {
      const probability = target[row * V + k];
      rows[row * V + k] = probability > 0 ? Math.pow(probability, quality) : 0;
    }
    normalizeRow(rows, row);
  }
  return rows;
}

/** Per-context acceptance rate: the chance a drafted token is kept, the sum over tokens of min(p, q). */
export function rowAcceptance(p: Rows, q: Rows): Float64Array {
  const out = new Float64Array(V);
  for (let row = 0; row < V; row += 1) {
    let total = 0;
    for (let k = 0; k < V; k += 1) total += Math.min(p[row * V + k], q[row * V + k]);
    out[row] = total;
  }
  return out;
}

/** The target's long-run share of text spent in each context character, by power iteration. */
export function stationary(p: Rows, iterations = 400): Float64Array {
  let current = new Float64Array(V);
  current[TINY_VOCAB.indexOf(" ")] = 1;
  for (let step = 0; step < iterations; step += 1) {
    const next = new Float64Array(V);
    for (let row = 0; row < V; row += 1) {
      const mass = current[row];
      if (mass === 0) continue;
      for (let k = 0; k < V; k += 1) next[k] += mass * p[row * V + k];
    }
    current = next;
  }
  return current;
}

/** α = E[β]: the per-context acceptance rate averaged over the contexts the target visits. */
export function meanAcceptance(p: Rows, q: Rows): number {
  const share = stationary(p);
  const byRow = rowAcceptance(p, q);
  let total = 0;
  for (let row = 0; row < V; row += 1) total += share[row] * byRow[row];
  return Math.min(1, Math.max(0, total));
}

/** The closed form, from the SDK that Deployment & serving also uses. */
export const closedForm = (alpha: number, gamma: number) => teachingServing.speculativeTokensPerPass(alpha, gamma);

/**
 * Exact expected tokens per pass for this Markov toy with no independence assumption:
 * one token always, plus the chance that the first i drafted tokens are all accepted, for i up to gamma.
 * A drafted token x after context c is kept with probability min(p, q)(x | c).
 */
export function exactTokensPerPass(p: Rows, q: Rows, gamma: number): number {
  const share = stationary(p);
  let alive = Float64Array.from(share);
  let total = 1;
  for (let step = 1; step <= gamma; step += 1) {
    const next = new Float64Array(V);
    for (let row = 0; row < V; row += 1) {
      const mass = alive[row];
      if (mass === 0) continue;
      for (let k = 0; k < V; k += 1) next[k] += mass * Math.min(p[row * V + k], q[row * V + k]);
    }
    alive = next;
    total += alive.reduce((sum, value) => sum + value, 0);
  }
  return total;
}

function draw(rows: Rows, row: number, random: () => number): number {
  let remaining = random();
  for (let k = 0; k < V; k += 1) {
    remaining -= rows[row * V + k];
    if (remaining <= 0) return k;
  }
  // Rounding left a sliver of mass: take the last character that has any.
  for (let k = V - 1; k >= 0; k -= 1) if (rows[row * V + k] > 0) return k;
  return 0;
}

export interface PassRecord {
  /** Characters the draft proposed, in order. */
  proposed: string;
  /** How many of them the target kept. */
  accepted: number;
  /** The first proposal the target refused, if any. */
  rejected: string | null;
  /** The target's own token for the position after the accepted ones. */
  fromTarget: string;
  /** True when the target's token is the bonus token after all gamma proposals were kept. */
  bonus: boolean;
  /** Tokens this pass added to the text: accepted plus one. */
  produced: number;
}

export interface Simulation {
  passes: number;
  meanTokens: number;
  standardError: number;
  /** Drafted tokens kept, divided by drafted tokens the target examined. */
  acceptedShare: number;
  transcript: PassRecord[];
  /** The text the transcript passes wrote, starting after the prompt. */
  text: string;
}

const glyph = (id: number) => TINY_VOCAB[id] ?? " ";

/** Algorithm 1 of Leviathan et al., repeated `passes` times from `PROMPT`, seeded. */
export function simulate({
  p,
  q,
  gamma,
  seed,
  passes = SIM_PASSES,
  transcriptPasses = TRANSCRIPT_PASSES,
}: {
  p: Rows;
  q: Rows;
  gamma: number;
  seed: number;
  passes?: number;
  transcriptPasses?: number;
}): Simulation {
  const random = tinyRandom(seed);
  let context = TINY_VOCAB.indexOf(PROMPT[PROMPT.length - 1]);
  let produced = 0;
  let producedSquares = 0;
  let examined = 0;
  let kept = 0;
  const transcript: PassRecord[] = [];
  let text = "";
  const proposals = new Int32Array(gamma);

  for (let pass = 0; pass < passes; pass += 1) {
    let previous = context;
    for (let i = 0; i < gamma; i += 1) {
      proposals[i] = draw(q, previous, random);
      previous = proposals[i];
    }
    // The target scores every proposal in one pass; here that is a table lookup per position.
    let accepted = 0;
    let rejectedId = -1;
    previous = context;
    for (let i = 0; i < gamma; i += 1) {
      const token = proposals[i];
      const ratio = q[previous * V + token] > 0 ? p[previous * V + token] / q[previous * V + token] : 1;
      examined += 1;
      if (random() <= ratio) {
        accepted += 1;
        kept += 1;
        previous = token;
      } else {
        rejectedId = token;
        break;
      }
    }
    // After a rejection the target resamples from norm(max(0, p - q)); after gamma keeps it takes a bonus token from p.
    let replacement: number;
    if (accepted < gamma) {
      const adjusted = new Float64Array(V);
      let total = 0;
      for (let k = 0; k < V; k += 1) {
        adjusted[k] = Math.max(0, p[previous * V + k] - q[previous * V + k]);
        total += adjusted[k];
      }
      if (total <= 0) {
        for (let k = 0; k < V; k += 1) adjusted[k] = p[previous * V + k];
      } else {
        for (let k = 0; k < V; k += 1) adjusted[k] /= total;
      }
      replacement = draw(adjusted, 0, random);
    } else {
      replacement = draw(p, previous, random);
    }
    const count = accepted + 1;
    produced += count;
    producedSquares += count * count;
    if (pass < transcriptPasses) {
      let proposed = "";
      for (let i = 0; i < gamma; i += 1) proposed += glyph(proposals[i]);
      transcript.push({
        proposed,
        accepted,
        rejected: rejectedId >= 0 ? glyph(rejectedId) : null,
        fromTarget: glyph(replacement),
        bonus: accepted === gamma,
        produced: count,
      });
      for (let i = 0; i < accepted; i += 1) text += glyph(proposals[i]);
      text += glyph(replacement);
    }
    context = replacement;
  }

  const mean = produced / Math.max(1, passes);
  const variance = Math.max(0, producedSquares / Math.max(1, passes) - mean * mean);
  return {
    passes,
    meanTokens: mean,
    standardError: Math.sqrt(variance / Math.max(1, passes)),
    acceptedShare: examined > 0 ? kept / examined : 0,
    transcript,
    text,
  };
}
