/**
 * The training text of the capstone lab. It keeps the corpus split the lab has always used and adds a
 * small, deterministic version of the scrape-and-clean step from the dataset-building lab: a raw
 * scrape is the training sentences plus site furniture, repeats and fragments, and cleaning removes
 * them again. Nothing here is random, so the same corpus always yields the same raw and cleaned text.
 */
import { encodeTinyText } from "@app/module-sdk";

export interface CorpusSplit {
  /** Training sentences without their full stops. */
  sentences: string[];
  train: string;
  held: string;
}

/** Keeps the last quarter of a corpus's sentences (at least one) out of training. */
export function splitCorpus(text: string): CorpusSplit {
  const all = text
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);
  const heldCount = Math.max(1, Math.round(all.length * 0.25));
  const sentences = all.slice(0, all.length - heldCount);
  return {
    sentences,
    train: `${sentences.join(". ")}.`,
    held: `${all.slice(all.length - heldCount).join(". ")}.`,
  };
}

export type ScrapeKind = "sentence" | "boilerplate" | "fragment" | "repeat" | "variant";

export interface ScrapeLine {
  text: string;
  kind: ScrapeKind;
}

/** Lines a crawl picks up from page furniture rather than from the document's prose. */
export const BOILERPLATE_LINES = [
  "copyright 2019 all rights reserved.",
  "click here to subscribe to our newsletter.",
  "share this page on social media.",
] as const;
export const BOILERPLATE_PHRASES = [
  "all rights reserved",
  "click here",
  "subscribe",
  "share this page",
  "copyright",
] as const;
export const FRAGMENTS = ["ok.", "yes."] as const;
/** A line shorter than this many characters is a fragment, not a sentence. */
export const MIN_LINE_LENGTH = 12;
/** Character-trigram Jaccard similarity at or above which a line is a near-duplicate of a kept one. */
export const NEAR_DUPLICATE = 0.7;

/** The same sentence with its first long word misspelled by doubling that word's second letter. */
export function typoVariant(sentence: string): string {
  const words = sentence.split(" ");
  const index = words.findIndex((word) => word.length >= 5);
  if (index < 0) return sentence;
  const word = words[index];
  words[index] = `${word.slice(0, 2)}${word[1]}${word.slice(2)}`;
  return words.join(" ");
}

/**
 * Lays the scrape out: every training sentence in order, with furniture, exact repeats, a misspelled
 * copy and two fragments slotted in after fixed sentence positions.
 */
export function buildScrape(sentences: ReadonlyArray<string>): ScrapeLine[] {
  const line = (sentence: string, kind: ScrapeKind): ScrapeLine => ({ text: `${sentence}.`, kind });
  const furniture = (index: number): ScrapeLine => ({
    text: BOILERPLATE_LINES[index % BOILERPLATE_LINES.length],
    kind: "boilerplate",
  });
  const lines: ScrapeLine[] = [];
  sentences.forEach((sentence, index) => {
    lines.push(line(sentence, "sentence"));
    const extras: ScrapeLine[] = [];
    if (index === 0) extras.push(furniture(0));
    if (index === 1) extras.push(line(sentences[0], "repeat"), { text: FRAGMENTS[0], kind: "fragment" });
    if (index === 2) extras.push(furniture(1));
    if (index === 3) extras.push(line(typoVariant(sentences[1]), "variant"));
    if (index === 4) extras.push(furniture(2), line(sentences[2], "repeat"));
    if (index === 5) extras.push({ text: FRAGMENTS[1], kind: "fragment" });
    if (index === 6) extras.push(furniture(3));
    lines.push(...extras);
  });
  return lines;
}

function trigrams(text: string): Set<string> {
  const grams = new Set<string>();
  for (let index = 0; index + 3 <= text.length; index += 1) grams.add(text.slice(index, index + 3));
  return grams;
}

/** Exact Jaccard similarity of two lines' character-trigram sets. */
export function trigramSimilarity(left: string, right: string): number {
  const a = trigrams(left);
  const b = trigrams(right);
  let shared = 0;
  for (const gram of a) if (b.has(gram)) shared += 1;
  const union = a.size + b.size - shared;
  return union === 0 ? 1 : shared / union;
}

export type Verdict = "kept" | "boilerplate" | "short" | "duplicate" | "near";

export interface CleaningResult {
  verdicts: Verdict[];
  kept: string[];
  /** The kept lines joined with single spaces: the cleaned scrape as the trainer reads it. */
  text: string;
  removed: Record<Exclude<Verdict, "kept">, number>;
}

/**
 * Runs the stages in a fixed order and gives each line the first verdict that fires: boilerplate,
 * length floor, exact duplicate of a kept line, near-duplicate of a kept line.
 */
export function cleanScrape(scrape: ReadonlyArray<ScrapeLine>): CleaningResult {
  const kept: string[] = [];
  const removed = { boilerplate: 0, short: 0, duplicate: 0, near: 0 };
  const verdicts = scrape.map(({ text }): Verdict => {
    if (BOILERPLATE_PHRASES.some((phrase) => text.includes(phrase))) {
      removed.boilerplate += 1;
      return "boilerplate";
    }
    if (text.length < MIN_LINE_LENGTH) {
      removed.short += 1;
      return "short";
    }
    if (kept.includes(text)) {
      removed.duplicate += 1;
      return "duplicate";
    }
    if (kept.some((entry) => trigramSimilarity(entry, text) >= NEAR_DUPLICATE)) {
      removed.near += 1;
      return "near";
    }
    kept.push(text);
    return "kept";
  });
  return { verdicts, kept, text: kept.join(" "), removed };
}

export const scrapeText = (scrape: ReadonlyArray<ScrapeLine>) => scrape.map((entry) => entry.text).join(" ");

/** Training characters after encoding into the 30-symbol vocabulary, as the trainer counts them. */
export const encodedLength = (text: string) => encodeTinyText(text).length;
