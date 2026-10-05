import { describe, expect, it } from "vitest";
import { TINY_CORPORA } from "@app/module-sdk";
import {
  BOILERPLATE_LINES,
  BOILERPLATE_PHRASES,
  buildScrape,
  cleanScrape,
  MIN_LINE_LENGTH,
  NEAR_DUPLICATE,
  scrapeText,
  splitCorpus,
  trigramSimilarity,
  typoVariant,
} from "./data";

const ids = ["harbor", "recipes", "proverbs"] as const;

describe("the lab's scrape and cleaning step", () => {
  it("splits each corpus so the last quarter of its sentences is held out", () => {
    const harbor = splitCorpus(TINY_CORPORA.harbor.text);
    expect(harbor.sentences).toHaveLength(6);
    expect(harbor.held.split(".").filter((part) => part.trim())).toHaveLength(2);
    const proverbs = splitCorpus(TINY_CORPORA.proverbs.text);
    expect(proverbs.sentences).toHaveLength(7);
    for (const id of ids) {
      const split = splitCorpus(TINY_CORPORA[id].text);
      expect(split.train.endsWith(".")).toBe(true);
      expect(split.train).toBe(`${split.sentences.join(". ")}.`);
    }
  });

  it.each(ids)("cleans the %s scrape back to exactly its training text", (id) => {
    const split = splitCorpus(TINY_CORPORA[id].text);
    const scrape = buildScrape(split.sentences);
    const cleaning = cleanScrape(scrape);
    expect(cleaning.text).toBe(split.train);
    expect(cleaning.kept).toHaveLength(split.sentences.length);
    // Every non-sentence line was removed by exactly one stage.
    const junk = scrape.filter((line) => line.kind !== "sentence").length;
    expect(Object.values(cleaning.removed).reduce((sum, count) => sum + count, 0)).toBe(junk);
    expect(cleaning.verdicts).toHaveLength(scrape.length);
  });

  it("is deterministic: the same sentences always give the same scrape", () => {
    const sentences = splitCorpus(TINY_CORPORA.recipes.text).sentences;
    expect(buildScrape(sentences)).toEqual(buildScrape(sentences));
    expect(scrapeText(buildScrape(sentences))).toBe(scrapeText(buildScrape([...sentences])));
  });

  it("gives each junk kind the verdict of the stage that should catch it", () => {
    const split = splitCorpus(TINY_CORPORA.harbor.text);
    const scrape = buildScrape(split.sentences);
    const { verdicts } = cleanScrape(scrape);
    const expected = { sentence: "kept", boilerplate: "boilerplate", fragment: "short", repeat: "duplicate", variant: "near" };
    scrape.forEach((line, index) => expect(verdicts[index], line.text).toBe(expected[line.kind]));
  });

  it("builds the near-duplicate by doubling a letter, and it clears the similarity bar on every corpus", () => {
    expect(typoVariant("the morning fog settles")).toBe("the moorning fog settles");
    expect(typoVariant("a b c")).toBe("a b c");
    for (const id of ids) {
      const split = splitCorpus(TINY_CORPORA[id].text);
      const variant = typoVariant(split.sentences[1]);
      expect(variant).not.toBe(split.sentences[1]);
      expect(trigramSimilarity(`${split.sentences[1]}.`, `${variant}.`)).toBeGreaterThanOrEqual(NEAR_DUPLICATE);
    }
  });

  it("keeps every footer line out of the cleaned text and every sentence out of the boilerplate stage", () => {
    for (const line of BOILERPLATE_LINES) {
      expect(BOILERPLATE_PHRASES.some((phrase) => line.includes(phrase))).toBe(true);
      expect(line.length).toBeGreaterThan(MIN_LINE_LENGTH);
    }
    for (const id of ids) {
      for (const sentence of splitCorpus(TINY_CORPORA[id].text).sentences) {
        expect(BOILERPLATE_PHRASES.some((phrase) => sentence.includes(phrase))).toBe(false);
        expect(sentence.length + 1).toBeGreaterThanOrEqual(MIN_LINE_LENGTH);
      }
    }
  });

  it("measures similarity as the Jaccard overlap of character trigrams", () => {
    expect(trigramSimilarity("abcd", "abcd")).toBe(1);
    expect(trigramSimilarity("abcd", "wxyz")).toBe(0);
    expect(trigramSimilarity("abcd", "bcde")).toBeCloseTo(1 / 3, 10);
  });
});
