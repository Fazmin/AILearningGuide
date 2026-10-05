/**
 * The cleaning pipeline for the dataset-building lab. Every verdict, count,
 * similarity, and overlap on the page is computed here from the lines below.
 */
import { encodeTinyText } from "@app/module-sdk";

/**
 * A scraped-looking sample: real sentences, site furniture, an exact repeat, a
 * spelling variant of an earlier line, and two useless fragments.
 */
export const RAW_LINES = [
  "copyright 2019 all rights reserved.",
  "the morning fog settles over the harbor and the air feels cool.",
  "click here to subscribe to our newsletter.",
  "copyright 2019 all rights reserved.",
  "by noon the clouds break apart and the sun warms the stone streets.",
  "share this page on social media.",
  "the morning fog settles over the harbor and the air feels cool.",
  "copyright 2019 all rights reserved.",
  "a light wind moves in from the sea and the temperature drops again.",
  "ok.",
  "click here to subscribe to our newsletter.",
  "the morning fog settles over the harbour and the air feels cool.",
  "rain arrives before evening and the gutters fill with water.",
  "share this page on social media.",
  "yes.",
  "the storm passes and the sky clears over the quiet town.",
  "copyright 2019 all rights reserved.",
  "in winter the same harbor freezes and the boats stay in port.",
  "click here to subscribe to our newsletter.",
  "the boats stay in port until the wind drops and the fog lifts.",
] as const;

/** Never in either training set unless the leak switch copies one sentence in. */
export const HELD_OUT_SENTENCES = [
  "in summer the harbor fills with sails and the water stays warm until late.",
  "the fog returns in the morning and the pattern repeats.",
  "every season brings the same clouds and the same quiet streets.",
] as const;
export const VALIDATION_TEXT = HELD_OUT_SENTENCES.join(" ");

/** The sentence the leak switch copies into the scrape, the way a quoted benchmark item ends up in a crawl. */
export const LEAKED_INDEX = 1;
export const LEAKED_LINE = HELD_OUT_SENTENCES[LEAKED_INDEX];
/** Held-out text without the leaked sentence, for scoring what the leak did not touch. */
export const UNLEAKED_TEXT = HELD_OUT_SENTENCES.filter((_, index) => index !== LEAKED_INDEX).join(" ");
/** Where the leaked copy lands in the scrape: after the rain line. */
const LEAK_POSITION = 13;

export const BOILERPLATE = [
  "all rights reserved",
  "click here",
  "subscribe",
  "share this page",
  "copyright",
] as const;

export const NEAR_THRESHOLD = 0.7;
/** Word n-gram length for the decontamination check. */
export const CONTAMINATION_N = 8;

export type LineVerdict = "kept" | "boilerplate" | "short" | "duplicate" | "near" | "contaminated";

export interface PipelineOptions {
  dropBoilerplate: boolean;
  minLength: number;
  dropExact: boolean;
  dropNear: boolean;
  decontaminate: boolean;
  leak: boolean;
}

export interface LineResult {
  line: string;
  verdict: LineVerdict;
  /** For near-duplicates: the kept line it matched and their trigram Jaccard. */
  match?: { index: number; similarity: number };
  leaked: boolean;
}

export function scrapeLines(leak: boolean): { line: string; leaked: boolean }[] {
  const lines = RAW_LINES.map((line) => ({ line: line as string, leaked: false }));
  if (leak) lines.splice(LEAK_POSITION, 0, { line: LEAKED_LINE, leaked: true });
  return lines;
}

export function trigrams(text: string): Set<string> {
  const set = new Set<string>();
  for (let index = 0; index + 3 <= text.length; index += 1) set.add(text.slice(index, index + 3));
  return set;
}

/** Exact Jaccard similarity of two lines' character-trigram sets. */
export function jaccard(left: string, right: string): number {
  const a = trigrams(left);
  const b = trigrams(right);
  let shared = 0;
  for (const gram of a) if (b.has(gram)) shared += 1;
  const union = a.size + b.size - shared;
  return union === 0 ? 1 : shared / union;
}

function words(text: string): string[] {
  return text.replace(/\./g, " ").split(/\s+/).filter(Boolean);
}

function wordNgrams(text: string, n: number): Set<string> {
  const tokens = words(text);
  const set = new Set<string>();
  for (let index = 0; index + n <= tokens.length; index += 1) {
    set.add(tokens.slice(index, index + n).join(" "));
  }
  return set;
}

const heldOutNgrams = wordNgrams(VALIDATION_TEXT, CONTAMINATION_N);

/** True when a line shares any CONTAMINATION_N-word sequence with the held-out text. */
export function overlapsHeldOut(line: string): boolean {
  for (const gram of wordNgrams(line, CONTAMINATION_N)) if (heldOutNgrams.has(gram)) return true;
  return false;
}

/**
 * Stages run in a fixed order and each line gets the first verdict that fires:
 * boilerplate, length floor, exact duplicate, near-duplicate, contamination.
 * Duplicate checks compare only against lines already kept.
 */
export function runPipeline(options: PipelineOptions): LineResult[] {
  const kept: { line: string; index: number }[] = [];
  return scrapeLines(options.leak).map(({ line, leaked }, index) => {
    if (options.dropBoilerplate && BOILERPLATE.some((phrase) => line.includes(phrase))) {
      return { line, verdict: "boilerplate", leaked };
    }
    if (line.length < options.minLength) return { line, verdict: "short", leaked };
    if (options.dropExact && kept.some((entry) => entry.line === line)) {
      return { line, verdict: "duplicate", leaked };
    }
    if (options.dropNear) {
      let best: { index: number; similarity: number } | undefined;
      for (const entry of kept) {
        const similarity = jaccard(entry.line, line);
        if (similarity >= NEAR_THRESHOLD && (!best || similarity > best.similarity)) {
          best = { index: entry.index, similarity };
        }
      }
      if (best) return { line, verdict: "near", match: best, leaked };
    }
    if (options.decontaminate && overlapsHeldOut(line)) {
      return { line, verdict: "contaminated", leaked };
    }
    kept.push({ line, index });
    return { line, verdict: "kept", leaked };
  });
}

export interface FunnelRow {
  id: "scraped" | "boilerplate" | "short" | "duplicate" | "near" | "contaminated";
  label: string;
  active: boolean;
  removedLines: number;
  removedChars: number;
  /** Lines and encoded training characters still standing after this stage. */
  lines: number;
  chars: number;
}

const STAGE_ORDER: ReadonlyArray<{ id: Exclude<FunnelRow["id"], "scraped">; label: string; key: keyof PipelineOptions }> = [
  { id: "boilerplate", label: "Strip boilerplate", key: "dropBoilerplate" },
  { id: "short", label: "Length floor", key: "minLength" },
  { id: "duplicate", label: "Exact dedup", key: "dropExact" },
  { id: "near", label: "Near-dup", key: "dropNear" },
  { id: "contaminated", label: "Decontaminate", key: "decontaminate" },
];

/**
 * Survivors after each stage in pipeline order. Characters are counted after
 * encoding into the model's 30-symbol vocabulary, exactly as the trainer sees them.
 */
export function funnel(results: ReadonlyArray<LineResult>, options: PipelineOptions): FunnelRow[] {
  const size = (entries: ReadonlyArray<LineResult>) =>
    encodeTinyText(entries.map((entry) => entry.line).join(" ")).length;
  let standing = results.slice();
  const rows: FunnelRow[] = [
    {
      id: "scraped",
      label: "Scraped",
      active: true,
      removedLines: 0,
      removedChars: 0,
      lines: standing.length,
      chars: size(standing),
    },
  ];
  for (const stage of STAGE_ORDER) {
    const value = options[stage.key];
    const active = typeof value === "number" ? value > 0 : Boolean(value);
    const before = size(standing);
    const removed = standing.filter((entry) => entry.verdict === stage.id);
    standing = standing.filter((entry) => entry.verdict !== stage.id);
    const after = size(standing);
    rows.push({
      id: stage.id,
      label: stage.label,
      active,
      removedLines: removed.length,
      removedChars: before - after,
      lines: standing.length,
      chars: after,
    });
  }
  return rows;
}

/** Longest run of consecutive words shared by two texts. */
export function longestSharedSpan(left: string, right: string): { length: number; span: string } {
  const a = words(left);
  const b = words(right);
  let best = 0;
  let span = "";
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < b.length; j += 1) {
      let k = 0;
      while (i + k < a.length && j + k < b.length && a[i + k] === b[j + k]) k += 1;
      if (k > best) {
        best = k;
        span = a.slice(i, i + k).join(" ");
      }
    }
  }
  return { length: best, span };
}

export function trigramStats(text: string) {
  const grams = new Map<string, number>();
  for (let index = 0; index + 3 <= text.length; index += 1) {
    const gram = text.slice(index, index + 3);
    grams.set(gram, (grams.get(gram) ?? 0) + 1);
  }
  const total = Math.max(1, text.length - 2);
  return { distinct: grams.size, total, ratio: grams.size / total };
}

/* -------------------------------------------------------------------------- */
/* MinHash                                                                     */
/* -------------------------------------------------------------------------- */

/** 32-bit FNV-1a hash of a string. */
function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** A 32-bit integer mix, so each seed acts as an independent hash function. */
function mix(value: number, seed: number): number {
  let x = (value ^ Math.imul(seed + 1, 0x9e3779b1)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/** The minimum of each of `count` hash functions over a line's trigram set. */
export function minhashSignature(text: string, count: number): number[] {
  const hashes = Array.from(trigrams(text), fnv1a);
  return Array.from({ length: count }, (_, seed) => {
    let minimum = 0xffffffff;
    for (const hash of hashes) minimum = Math.min(minimum, mix(hash, seed));
    return minimum;
  });
}

/**
 * The share of signature positions that agree. For each hash function, the
 * probability that two sets share their minimum equals their Jaccard similarity,
 * so this is an unbiased estimate whose error shrinks like 1 / sqrt(count).
 */
export function minhashEstimate(left: string, right: string, count: number): number {
  const a = minhashSignature(left, count);
  const b = minhashSignature(right, count);
  let agree = 0;
  for (let index = 0; index < count; index += 1) if (a[index] === b[index]) agree += 1;
  return agree / count;
}

/** The spelling-variant pair the near-duplicate stage exists to catch. */
export const VARIANT_PAIR = [RAW_LINES[1], RAW_LINES[11]] as const;
export const MINHASH_SIZE = 64;
