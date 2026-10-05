/**
 * A count-based character n-gram, used to show when a model recites its training text.
 *
 * The model is not trained: it is the table of counts itself. For a context length `k`, every run of `k`
 * characters in the training text records which character came next, and sampling draws the next
 * character in proportion to those counts. With `k` = 1 this is the same one-character-of-context
 * bigram the rest of the track uses. Everything below is counting, so every figure the card shows is exact.
 */
import { decodeTinyIds, encodeTinyText, tinyRandom } from "@app/module-sdk";

export const CONTEXT_MIN = 1;
export const CONTEXT_MAX = 12;
export const DEFAULT_CONTEXT = 3;
/** Characters sampled for each model. */
export const SAMPLE_LENGTH = 300;
/** A character counts as copied when it sits inside a window of this many characters found verbatim in the training text. */
export const COPY_WINDOW = 16;
/** Fixed sampling seed, so a given context length always prints the same text. */
export const SAMPLE_SEED = 7;
/** The site furniture phrases the lab's boilerplate filter targets, counted in the sample. */
export const FOOTER_PHRASES = ["copyright", "all rights reserved", "click here", "subscribe", "share this page"] as const;

/** The training text as the 30-symbol model sees it: lowercase, one space at most, unknown characters spaced. */
export const normalized = (text: string) => decodeTinyIds(encodeTinyText(text));

export interface RecitationSample {
  context: number;
  /** The characters the model wrote after being given the first `context` characters of its training text. */
  text: string;
  /** Number of distinct contexts seen in training. */
  contexts: number;
  /** Contexts that were followed by exactly one character, so the model has no choice there. */
  forcedContexts: number;
}

/** Samples `length` characters. When a context never recurs (the final run of the text), it starts again from the top. */
export function sampleRecitation(trainingText: string, context: number, length = SAMPLE_LENGTH, seed = SAMPLE_SEED): RecitationSample {
  const k = Number.isFinite(context)
    ? Math.max(CONTEXT_MIN, Math.min(CONTEXT_MAX, Math.round(context)))
    : DEFAULT_CONTEXT;
  const source = normalized(trainingText);
  const table = new Map<string, Map<string, number>>();
  for (let index = 0; index + k < source.length; index += 1) {
    const key = source.slice(index, index + k);
    const row = table.get(key) ?? new Map<string, number>();
    row.set(source[index + k], (row.get(source[index + k]) ?? 0) + 1);
    table.set(key, row);
  }
  const random = tinyRandom(seed);
  let written = "";
  let window = source.slice(0, k);
  for (let index = 0; index < length; index += 1) {
    const row = table.get(window);
    let next: string;
    if (!row) {
      window = source.slice(0, k);
      next = " ";
    } else {
      let total = 0;
      for (const count of row.values()) total += count;
      let draw = random() * total;
      next = "";
      for (const [character, count] of row) {
        next = character;
        draw -= count;
        if (draw <= 0) break;
      }
      window = (window + next).slice(-k);
    }
    written += next;
  }
  let forced = 0;
  for (const row of table.values()) if (row.size === 1) forced += 1;
  return { context: k, text: written, contexts: table.size, forcedContexts: forced };
}

/** Share of the sample's characters that lie inside a `window`-character run found verbatim in the training text. */
export function copiedShare(sample: string, trainingText: string, window = COPY_WINDOW): number {
  const source = normalized(trainingText);
  if (sample.length === 0) return 0;
  const covered = new Array<boolean>(sample.length).fill(false);
  for (let start = 0; start + window <= sample.length; start += 1) {
    if (source.includes(sample.slice(start, start + window))) {
      for (let offset = start; offset < start + window; offset += 1) covered[offset] = true;
    }
  }
  return covered.filter(Boolean).length / sample.length;
}

/** Length in characters of the longest run of the sample that appears verbatim in the training text. */
export function longestCopiedRun(sample: string, trainingText: string): number {
  const source = normalized(trainingText);
  let best = 0;
  for (let start = 0; start < sample.length; start += 1) {
    let length = best + 1;
    while (start + length <= sample.length && source.includes(sample.slice(start, start + length))) length += 1;
    best = Math.max(best, length - 1);
  }
  return best;
}

/** Share of the sample's characters that belong to a copy of one of the footer phrases. */
export function footerShare(sample: string): number {
  if (sample.length === 0) return 0;
  const covered = new Array<boolean>(sample.length).fill(false);
  for (const phrase of FOOTER_PHRASES) {
    let at = sample.indexOf(phrase);
    while (at >= 0) {
      for (let offset = at; offset < at + phrase.length; offset += 1) covered[offset] = true;
      at = sample.indexOf(phrase, at + 1);
    }
  }
  return covered.filter(Boolean).length / sample.length;
}

export interface RecitationReport extends RecitationSample {
  copied: number;
  longest: number;
  footer: number;
}

export function recitationReport(trainingText: string, context: number): RecitationReport {
  const sample = sampleRecitation(trainingText, context);
  return {
    ...sample,
    copied: copiedShare(sample.text, trainingText),
    longest: longestCopiedRun(sample.text, trainingText),
    footer: footerShare(sample.text),
  };
}
