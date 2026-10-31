import { describe, expect, it } from "vitest";
import { encodeTinyText, TINY_CORPORA, TINY_VOCAB } from "@app/module-sdk";
import {
  ARITHMETIC,
  ARITHMETIC_SENTENCES,
  arithmeticFacts,
  DOMAIN_IDS,
  DOMAINS,
  generateArithmetic,
  GENERAL_DOMAINS,
  heldOutCount,
  numberWord,
  pairCount,
  sentencesOf,
} from "./corpora";

const WORDS = Array.from({ length: 21 }, (_, value) => numberWord(value));
const valueOf = (word: string) => WORDS.indexOf(word);

describe("mid-training corpora", () => {
  it("spells 0 to 20 and refuses anything else", () => {
    expect(numberWord(0)).toBe("zero");
    expect(numberWord(12)).toBe("twelve");
    expect(numberWord(20)).toBe("twenty");
    expect(() => numberWord(21)).toThrow(RangeError);
    expect(() => numberWord(-1)).toThrow(RangeError);
  });

  it("states only true facts from the three templates, with no operand of zero and no product by one", () => {
    const facts = arithmeticFacts();
    expect(new Set(facts).size).toBe(facts.length);
    for (const fact of facts) {
      const match = fact.match(/^(\w+) (plus|minus|times) (\w+) is (\w+)\.$/);
      expect(match, fact).not.toBeNull();
      const [, left, operation, right, answer] = match!;
      const [a, b, c] = [valueOf(left), valueOf(right), valueOf(answer)];
      expect(Math.min(a, b, c), fact).toBeGreaterThanOrEqual(0);
      expect(Math.min(a, b), fact).toBeGreaterThanOrEqual(1);
      if (operation === "plus") expect(a + b).toBe(c);
      if (operation === "minus") expect(a - b).toBe(c);
      if (operation === "times") {
        expect(a * b).toBe(c);
        expect(Math.min(a, b)).toBeGreaterThanOrEqual(2);
      }
    }
    // 190 sums, 190 differences and 27 products have operands and answers from 1 to 20.
    expect(facts.length).toBe(407);
  });

  it("generates the same 160 distinct sentences in the same order every time", () => {
    const first = generateArithmetic();
    expect(first).toHaveLength(ARITHMETIC_SENTENCES);
    expect(ARITHMETIC_SENTENCES).toBe(160);
    expect(generateArithmetic()).toEqual(first);
    expect(new Set(first).size).toBe(first.length);
    expect(ARITHMETIC.sentences).toEqual(first);
    // A different seed draws a different set.
    expect(generateArithmetic(160, 47)).not.toEqual(first);
    // The example the lesson quotes is a fact the templates state.
    expect(arithmeticFacts()).toContain("four plus eight is twelve.");
  });

  it("uses only letters, spaces and full stops, so encoding never drops a character", () => {
    for (const domain of DOMAINS) {
      for (const sentence of domain.sentences) {
        expect(sentence, domain.id).toMatch(/^[a-z ]+\.$/);
        expect(encodeTinyText(sentence).length).toBe(sentence.length);
      }
    }
    expect([...TINY_VOCAB].filter((character) => /[0-9]/.test(character))).toEqual([]);
  });

  it("holds out the last quarter of each domain, rounded, and never trains on it", () => {
    expect(DOMAIN_IDS).toEqual(["harbor", "recipes", "proverbs", "arithmetic"]);
    expect(DOMAINS.map((domain) => [domain.train.length, domain.heldOut.length])).toEqual([
      [6, 2],
      [6, 2],
      [7, 3],
      [120, 40],
    ]);
    for (const domain of DOMAINS) {
      expect(domain.heldOut.length).toBe(heldOutCount(domain.sentences.length));
      expect([...domain.train, ...domain.heldOut]).toEqual(domain.sentences);
      expect(domain.heldOut).toEqual(domain.sentences.slice(domain.sentences.length - domain.heldOut.length));
      for (const sentence of domain.heldOut) {
        expect(domain.train, `${domain.id}: "${sentence}"`).not.toContain(sentence);
        expect(domain.trainText.includes(sentence.slice(0, -1)), `${domain.id}: "${sentence}"`).toBe(false);
      }
    }
  });

  it("keeps every held-out sentence out of every domain's training text, not only its own", () => {
    const allTraining = DOMAINS.map((domain) => domain.trainText).join(" ");
    for (const domain of DOMAINS) {
      for (const sentence of domain.heldOut) expect(allTraining.includes(sentence)).toBe(false);
    }
  });

  it("splits the SDK corpora into their sentences unchanged", () => {
    expect(GENERAL_DOMAINS.map((domain) => domain.name)).toEqual(["Harbor", "Recipes", "Proverbs"]);
    expect(sentencesOf(TINY_CORPORA.harbor.text)).toHaveLength(8);
    expect(sentencesOf(TINY_CORPORA.recipes.text)).toHaveLength(8);
    expect(sentencesOf(TINY_CORPORA.proverbs.text)).toHaveLength(10);
    // Joining the sentences with spaces encodes to exactly what the SDK corpus encodes to.
    expect(encodeTinyText(DOMAINS[0].sentences.join(" "))).toEqual(encodeTinyText(TINY_CORPORA.harbor.text));
  });

  it("counts training pairs the way the trainer does", () => {
    expect(DOMAINS.map((domain) => pairCount(domain.trainText))).toEqual([378, 292, 195, 3417]);
    expect(pairCount("")).toBe(0);
    expect(pairCount("ab")).toBe(1);
  });
});
