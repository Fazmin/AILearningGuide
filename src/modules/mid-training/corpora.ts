/**
 * The four domains the mid-training lab mixes. Three are the SDK's fixed corpora (Harbor weather
 * notes, Recipe steps, Proverbs); the fourth is a synthetic arithmetic set generated here from three
 * templates, with every number spelled out because the tiny vocabulary has no digits. Each domain
 * keeps its last quarter of sentences out of training, so every loss the lab reports per domain is
 * scored on sentences that no batch ever contained.
 */
import { encodeTinyText, TINY_CORPORA, tinyRandom } from "@app/module-sdk";

export type DomainId = "harbor" | "recipes" | "proverbs" | "arithmetic";

export interface Domain {
  id: DomainId;
  /** The name the lab's controls and rows use. */
  name: string;
  /** True for the three SDK corpora that together make up "General". */
  general: boolean;
  /** Every sentence, in order, each ending with a full stop. */
  sentences: string[];
  /** The sentences training may draw from: all but the last quarter. */
  train: string[];
  /** The last quarter, never trained on. */
  heldOut: string[];
  trainText: string;
  heldOutText: string;
}

/** The share of each domain's sentences, from the end, kept out of training (rounded to whole sentences). */
export const HELD_OUT_SHARE = 0.25;

const NUMBER_WORDS = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
  "twenty",
] as const;

/** Spells 0 to 20 in words; the vocabulary is letters, space and full stop only. */
export function numberWord(value: number): string {
  const word = NUMBER_WORDS[value];
  if (word === undefined) throw new RangeError(`numberWord covers 0 to 20, not ${value}`);
  return word;
}

/** How many arithmetic sentences the generator keeps, and the seed of its shuffle. */
export const ARITHMETIC_SENTENCES = 160;
export const ARITHMETIC_SEED = 46;

/**
 * Every true fact the three templates can state with operands and answers from 0 to 20:
 * "a plus b is c", "a minus b is c" and "a times b is c", with no operand of zero and no product
 * by one, so no fact is trivially the other operand.
 */
export function arithmeticFacts(): string[] {
  const facts: string[] = [];
  for (let a = 1; a <= 20; a += 1) {
    for (let b = 1; b <= 20; b += 1) {
      if (a + b <= 20) facts.push(`${numberWord(a)} plus ${numberWord(b)} is ${numberWord(a + b)}.`);
      if (a > b) facts.push(`${numberWord(a)} minus ${numberWord(b)} is ${numberWord(a - b)}.`);
      if (a >= 2 && b >= 2 && a * b <= 20) facts.push(`${numberWord(a)} times ${numberWord(b)} is ${numberWord(a * b)}.`);
    }
  }
  return facts;
}

/**
 * A fixed, shuffled draw of `count` distinct facts. The shuffle is a seeded Fisher–Yates pass, so the
 * same seed always yields the same sentences in the same order.
 */
export function generateArithmetic(count = ARITHMETIC_SENTENCES, seed = ARITHMETIC_SEED): string[] {
  const facts = arithmeticFacts();
  const random = tinyRandom(seed);
  for (let index = facts.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [facts[index], facts[swap]] = [facts[swap], facts[index]];
  }
  return facts.slice(0, Math.max(0, Math.min(facts.length, Math.round(count))));
}

/** A corpus's sentences, each ending in a full stop, with the layout whitespace removed. */
export function sentencesOf(text: string): string[] {
  return text
    .split(".")
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .map((part) => `${part}.`);
}

/** How many of `count` sentences the held-out quarter takes. */
export const heldOutCount = (count: number) => Math.round(count * HELD_OUT_SHARE);

export const joinSentences = (sentences: ReadonlyArray<string>) => sentences.join(" ");

function makeDomain(id: DomainId, name: string, general: boolean, sentences: string[]): Domain {
  const held = heldOutCount(sentences.length);
  const train = sentences.slice(0, sentences.length - held);
  const heldOut = sentences.slice(sentences.length - held);
  return {
    id,
    name,
    general,
    sentences,
    train,
    heldOut,
    trainText: joinSentences(train),
    heldOutText: joinSentences(heldOut),
  };
}

export const DOMAINS: ReadonlyArray<Domain> = [
  makeDomain("harbor", "Harbor", true, sentencesOf(TINY_CORPORA.harbor.text)),
  makeDomain("recipes", "Recipes", true, sentencesOf(TINY_CORPORA.recipes.text)),
  makeDomain("proverbs", "Proverbs", true, sentencesOf(TINY_CORPORA.proverbs.text)),
  makeDomain("arithmetic", "Arithmetic", false, generateArithmetic()),
];

export const DOMAIN_IDS: ReadonlyArray<DomainId> = DOMAINS.map((domain) => domain.id);
export const GENERAL_DOMAINS = DOMAINS.filter((domain) => domain.general);
export const ARITHMETIC = DOMAINS[3];

/** Training pairs (character bigrams) in a text, as the trainer counts them. */
export const pairCount = (text: string) => Math.max(0, encodeTinyText(text).length - 1);
