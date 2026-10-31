import { describe, expect, it } from "vitest";
import { bigramPairs, tinyCrossEntropy, TINY_VOCAB_SIZE } from "@app/module-sdk";
import { ARITHMETIC } from "./corpora";
import { runMixture } from "./mixture";
import { ANSWER_WORDS, answerLogProb, answerProbe, splitFact } from "./probe";

/** Total log-probability of `text` under the table: minus the summed cross-entropy of its pairs. */
const totalLogProb = (weights: Float32Array, text: string) => -tinyCrossEntropy(weights, text) * bigramPairs(text).inputs.length;

describe("answer probe", () => {
  it("splits a fact into its question and answer", () => {
    expect(splitFact("four plus eight is twelve.")).toEqual({ question: "four plus eight is", truth: "twelve" });
    expect(splitFact("eighteen minus fifteen is three.")).toEqual({ question: "eighteen minus fifteen is", truth: "three" });
    expect(ANSWER_WORDS).toHaveLength(21);
    expect(ANSWER_WORDS[0]).toBe("zero");
    expect(ANSWER_WORDS[20]).toBe("twenty");
  });

  it("scores an answer as the log-probability of its letters and full stop after a space", () => {
    const { weights } = runMixture([3, 3, 3, 1], "same", "wsd");
    for (const fact of ARITHMETIC.heldOut.slice(0, 6)) {
      const { question } = splitFact(fact);
      for (const word of ["ten", "twelve", "one"]) {
        // The completion's probability given the whole question, from the trainer's own cross-entropy.
        const given = totalLogProb(weights, `${question} ${word}.`) - totalLogProb(weights, `${question} `);
        expect(answerLogProb(weights, word)).toBeCloseTo(given, 4);
      }
    }
  });

  it("ranks the 21 answers the same way for every question, so it gives one answer to all of them", () => {
    const { weights } = runMixture([3, 3, 3, 1], "same", "wsd");
    const probe = answerProbe(weights);
    expect(probe.total).toBe(40);
    expect(probe.distinctPicks).toBe(1);
    expect(probe.ranking).toHaveLength(21);
    expect(probe.ranking.reduce((total, entry) => total + entry.share, 0)).toBeCloseTo(1, 12);
    for (let index = 1; index < probe.ranking.length; index += 1) {
      expect(probe.ranking[index].logProb).toBeLessThanOrEqual(probe.ranking[index - 1].logProb);
    }
    // Scoring each question with its full prefix picks the same word every time.
    for (const fact of ARITHMETIC.heldOut) {
      const { question } = splitFact(fact);
      const best = ANSWER_WORDS.reduce((winner, word) =>
        totalLogProb(weights, `${question} ${word}.`) > totalLogProb(weights, `${question} ${winner}.`) ? word : winner,
      );
      expect(best, fact).toBe(probe.pick);
    }
    expect(probe.correct).toBe(probe.facts.filter((fact) => fact.truth === probe.pick).length);
  });

  it("gives a uniform table no favourite beyond answer length", () => {
    const probe = answerProbe(new Float32Array(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE));
    // Every character is 1 in 30, so the shortest words share the top.
    const shortest = Math.min(...ANSWER_WORDS.map((word) => word.length));
    expect(probe.pick.length).toBe(shortest);
    expect(probe.distinctPicks).toBe(1);
  });
});
