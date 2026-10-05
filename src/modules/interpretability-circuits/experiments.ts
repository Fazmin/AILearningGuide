/**
 * The experiments this lab runs on the shipped transformer: a repeated-letter
 * copy task, attention scores per head, head ablation (zero and mean), and a
 * residual-stream activation-patching sweep. Everything here is a real forward
 * pass through the engine; nothing is interpolated or drawn from a formula.
 */

import {
  finalLogitsWithPatch,
  rowOf,
  runTransformer,
  softmax,
  type RunResult,
  type TransformerWeights,
} from "./engine";
import { meanAblatedWeights, meanHeadOutputs, referenceStrings, type HeadMeans } from "./mean-ablation";
import { encodeText, ITOS, tokenId } from "./vocabulary";

export const PROMPT_LENGTH = 24;
/** Twenty distinct capitals; a period-P prompt repeats the first P of them. */
export const LETTER_POOL = "QXZRKWMPJVBHDLTGNSCF";
/** The replacement letter for the corrupted run; it never appears in a prompt. */
export const CORRUPT_LETTER = "Y";
/** Repeat periods the lab offers. Every one fits at least one full repeat in a 24-letter prompt. */
export const PERIODS = [6, 7, 8, 9, 10, 11, 12, 13] as const;
/**
 * Periods the model's synthetic training windows never used (models/train_language_models.py holds out
 * 11, 13, 17 and 19; 17 and 19 leave too few repeated letters in a 24-letter prompt, so the lab offers 11 and 13).
 * Every other period in PERIODS was in the training mix.
 */
export const HELD_OUT_PERIODS: readonly number[] = [11, 13];
export const DEFAULT_PERIOD = 8;
export const HEADS = 4;
export const LAYERS = 2;

export interface CopyPrompt {
  period: number;
  text: string;
  ids: number[];
  /** The letter that should come next if the model copies the repeat. */
  target: string;
  /** Position of the target's previous occurrence: length - period. */
  source: number;
}

/** Any text that repeats every `period` letters, as a prompt. */
export function copyPromptFromText(text: string, period: number): CopyPrompt {
  return { period, text, ids: encodeText(text), target: text[text.length % period], source: text.length - period };
}

export function copyPrompt(period: number, length = PROMPT_LENGTH): CopyPrompt {
  const base = LETTER_POOL.slice(0, period);
  let text = "";
  while (text.length < length) text += base;
  return copyPromptFromText(text.slice(0, length), period);
}

/**
 * The unrelated letter replaced in the control corruption: inside the first repeat, half a period
 * away from the letter being copied. Replacing a letter the model reads when it copies (the letter
 * being copied in a later repeat, or the letters just before it) moves the answer; this position is
 * far from all of them, and a test pins that the metric barely moves at every period the lab offers.
 */
export function controlPosition(prompt: CopyPrompt) {
  return (prompt.source + Math.floor(prompt.period / 2)) % prompt.period;
}

/** Next letter after position i if the sequence keeps repeating. */
export function expectedNext(prompt: CopyPrompt, position: number) {
  return prompt.text[(position + 1) % prompt.period];
}

/**
 * Positions that can be scored for copying: the second repeat onward, from `period` to the end.
 * There the current letter has already appeared once, so an earlier occurrence exists to copy
 * from. Position `period - 1` is left out: its next letter repeats the first letter of the
 * string, but nothing before it can say so.
 */
export function copyPositions(prompt: CopyPrompt) {
  const positions: number[] = [];
  for (let position = prompt.period; position < prompt.ids.length; position += 1) positions.push(position);
  return positions;
}

export interface CopyScore {
  /** Mean probability of the correct next letter over copyPositions. */
  mean: number;
  perPosition: number[];
  /** Final position: probability of the target and the top three predictions. */
  finalTarget: number;
  top: { token: string; probability: number }[];
}

/** The three most probable next characters at one position. */
export function topAt(run: RunResult, position: number, count = 3) {
  const probabilities = softmax(rowOf(run.logits, position, ITOS.length));
  return Array.from(probabilities, (probability, index) => ({ token: ITOS[index], probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, count);
}

export function copyScore(run: RunResult, prompt: CopyPrompt): CopyScore {
  const vocabulary = ITOS.length;
  const perPosition = copyPositions(prompt).map((position) => {
    const probabilities = softmax(rowOf(run.logits, position, vocabulary));
    return probabilities[tokenId(expectedNext(prompt, position))];
  });
  const last = softmax(rowOf(run.logits, prompt.ids.length - 1, vocabulary));
  return {
    mean: perPosition.reduce((sum, value) => sum + value, 0) / Math.max(1, perPosition.length),
    perPosition,
    finalTarget: last[tokenId(prompt.target)],
    top: topAt(run, prompt.ids.length - 1),
  };
}

export interface HeadScore {
  layer: number;
  head: number;
  /** Mean attention from i to i - period + 1: the token after the previous occurrence (induction). */
  induction: number;
  /** Mean attention from i to i - 1: the previous letter. */
  previous: number;
  /** The single offset (i - key) that takes the most attention on average, and its share. */
  strongestOffset: number;
  strongestShare: number;
}

/** Scores over queries in the second repeat onward (i >= period), where induction is defined. */
export function headScores(run: RunResult, prompt: CopyPrompt): HeadScore[] {
  const sequence = prompt.ids.length;
  const scores: HeadScore[] = [];
  for (let layer = 0; layer < run.attention.length; layer += 1) {
    for (let head = 0; head < HEADS; head += 1) {
      const byOffset = new Float64Array(sequence);
      let count = 0;
      for (let query = prompt.period; query < sequence; query += 1) {
        const row = (head * sequence + query) * sequence;
        for (let key = 0; key <= query; key += 1) byOffset[query - key] += run.attention[layer][row + key];
        count += 1;
      }
      const mean = (offset: number) => (offset < sequence ? byOffset[offset] / Math.max(1, count) : 0);
      let strongestOffset = 0;
      for (let offset = 1; offset < sequence; offset += 1) {
        if (byOffset[offset] > byOffset[strongestOffset]) strongestOffset = offset;
      }
      scores.push({
        layer,
        head,
        induction: mean(prompt.period - 1),
        previous: mean(1),
        strongestOffset,
        strongestShare: mean(strongestOffset),
      });
    }
  }
  return scores;
}

export const headId = (layer: number, head: number) => `L${layer + 1}H${head + 1}`;

export function headScaleFor(ablated: ReadonlyArray<string>) {
  return Array.from({ length: LAYERS }, (_, layer) =>
    Array.from({ length: HEADS }, (_, head) => (ablated.includes(headId(layer, head)) ? 0 : 1)),
  );
}

/** What replaces an ablated head's output: zero, or its average over a reference batch. */
export type AblationMode = "zero" | "mean";

/** Every head's average output on seeded random repeats at this prompt's period (second repeat onward). */
export function referenceMeans(weights: TransformerWeights, prompt: CopyPrompt): HeadMeans {
  return meanHeadOutputs(weights, referenceStrings(prompt.period, prompt.ids.length), prompt.period);
}

export function runWithAblation(
  weights: TransformerWeights,
  prompt: CopyPrompt,
  ablated: ReadonlyArray<string>,
  mode: AblationMode = "zero",
  means?: HeadMeans,
) {
  const headScale = headScaleFor(ablated);
  if (mode === "mean") {
    if (!means) throw new Error("Mean ablation needs the reference means (see referenceMeans).");
    return runTransformer(meanAblatedWeights(weights, means, headScale), prompt.ids, { headScale });
  }
  return runTransformer(weights, prompt.ids, { headScale });
}

/** Copy score with each single head ablated, in L1H1 … L2H4 order. */
export function singleHeadEffects(
  weights: TransformerWeights,
  prompt: CopyPrompt,
  mode: AblationMode = "zero",
  means?: HeadMeans,
) {
  const effects: { id: string; mean: number }[] = [];
  for (let layer = 0; layer < LAYERS; layer += 1) {
    for (let head = 0; head < HEADS; head += 1) {
      const id = headId(layer, head);
      effects.push({ id, mean: copyScore(runWithAblation(weights, prompt, [id], mode, means), prompt).mean });
    }
  }
  return effects;
}

/** The mean attention the two layer-2 induction heads put on the induction key, from headScores. */
export const INDUCTION_HEADS = ["L2H2", "L2H3"] as const;

export function inductionStripe(run: RunResult, prompt: CopyPrompt) {
  const scores = headScores(run, prompt);
  return INDUCTION_HEADS.map((id) => {
    const found = scores.find((entry) => headId(entry.layer, entry.head) === id);
    return { id, induction: found?.induction ?? 0 };
  });
}

/** Which letter is replaced in the corrupted run. */
export type Corruption = "source" | "control";

export interface PatchSweep {
  id: string;
  clean: string;
  corrupt: string;
  /** Positions where the clean and corrupted prompts differ. */
  differing: number[];
  /** What is measured at the final position. */
  metric: string;
  cleanScore: number;
  corruptScore: number;
  /** recovery[layer][position] = (patched - corrupt) / (clean - corrupt). */
  recovery: number[][];
  /** The metric itself after each patch, when available. */
  patched?: number[][];
  /** Most probable next character at the last position in each run, when available (live sweeps). */
  cleanTop?: { token: string; probability: number };
  corruptTop?: { token: string; probability: number };
}

function differingPositions(a: string, b: string) {
  const out: number[] = [];
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) if (a[index] !== b[index]) out.push(index);
  return out;
}

/** The corrupted run's text and position for a prompt: the letter to copy, or the unrelated control letter. */
export function corruptedPrompt(prompt: CopyPrompt, corruption: Corruption = "source") {
  const position = corruption === "source" ? prompt.source : controlPosition(prompt);
  return {
    position,
    text: prompt.text.slice(0, position) + CORRUPT_LETTER + prompt.text.slice(position + 1),
  };
}

/** The two runs a patching sweep compares, and the metric on each: logit(target) - logit(CORRUPT_LETTER) at the last position. */
export function patchEndpoints(weights: TransformerWeights, prompt: CopyPrompt, corruption: Corruption = "source") {
  const { position, text } = corruptedPrompt(prompt, corruption);
  const target = tokenId(prompt.target);
  const foil = tokenId(CORRUPT_LETTER);
  const last = prompt.ids.length - 1;
  const clean = runTransformer(weights, prompt.ids);
  const corrupt = runTransformer(weights, encodeText(text));
  const difference = (logits: ArrayLike<number>) => logits[target] - logits[foil];
  return {
    position,
    corruptText: text,
    clean,
    corrupt,
    difference,
    cleanScore: difference(rowOf(clean.logits, last, ITOS.length)),
    corruptScore: difference(rowOf(corrupt.logits, last, ITOS.length)),
  };
}

/**
 * Clean: the copy prompt. Corrupted: one letter replaced by CORRUPT_LETTER, either the target's
 * previous occurrence (`source`, the letter the model should copy) or an unrelated letter in the
 * first repeat (`control`). Metric: logit(target) - logit(CORRUPT_LETTER) at the last position.
 * Each cell copies one clean residual vector (after block 1 or 2, at one position) into the
 * corrupted run and re-runs the blocks after it.
 */
export function copyPatchSweep(
  weights: TransformerWeights,
  prompt: CopyPrompt,
  corruption: Corruption = "source",
): PatchSweep {
  const { position, corruptText, clean, corrupt, difference, cleanScore, corruptScore } = patchEndpoints(
    weights,
    prompt,
    corruption,
  );
  const last = prompt.ids.length - 1;
  const patched = clean.residuals.map((residual, layer) =>
    Array.from({ length: prompt.ids.length }, (_, at) =>
      difference(finalLogitsWithPatch(weights, corrupt.residuals, layer, at, rowOf(residual, at, weights.width))),
    ),
  );
  const denominator = cleanScore - corruptScore;
  return {
    id: "live",
    clean: prompt.text,
    corrupt: corruptText,
    differing: [position],
    metric: `logit(${prompt.target}) − logit(${CORRUPT_LETTER})`,
    cleanScore,
    corruptScore,
    recovery: patched.map((row) => row.map((value) => (value - corruptScore) / denominator)),
    patched,
    cleanTop: topAt(clean, last, 1)[0],
    corruptTop: topAt(corrupt, last, 1)[0],
  };
}

export interface CachedPatchRecord {
  id: string;
  clean: string;
  corrupt: string;
  target: string;
  tokens: string[];
  clean_logit: number;
  corrupt_logit: number;
  recovery_by_layer_and_position: number[][];
}

/** The precomputed sweeps in interpretability-cache.json use the target logit alone as the metric. */
export function cachedPatchSweep(record: CachedPatchRecord): PatchSweep {
  const length = record.tokens.length;
  const clean = record.clean.slice(0, length);
  const corrupt = record.corrupt.slice(0, length);
  return {
    id: record.id,
    clean,
    corrupt,
    differing: differingPositions(clean, corrupt),
    metric: `logit(${record.target === " " ? "␣" : record.target})`,
    cleanScore: record.clean_logit,
    corruptScore: record.corrupt_logit,
    recovery: record.recovery_by_layer_and_position,
  };
}

/** Recovery is only meaningful when the clean and corrupted runs differ by a real margin. */
export const STABLE_MARGIN = 1;
