/**
 * The Answer probe: does a lower Arithmetic loss mean the model can do arithmetic? For each held-out
 * fact ("four plus eight is twelve.") the probe scores all 21 answer words from zero to twenty as the
 * completion after "is ", by the probability the table gives to the word and its full stop, and picks
 * the likeliest. A bigram conditions only on the previous character, and the character before every
 * answer is the same space, so the ranking cannot depend on the question.
 */
import { encodeTinyText, TINY_VOCAB, TINY_VOCAB_SIZE } from "@app/module-sdk";
import { ARITHMETIC, numberWord } from "./corpora";

export const ANSWER_WORDS: ReadonlyArray<string> = Array.from({ length: 21 }, (_, value) => numberWord(value));

const SPACE = TINY_VOCAB.indexOf(" ");

export interface ProbeFact {
  /** The fact up to and including "is". */
  question: string;
  truth: string;
  pick: string;
}

export interface AnswerRanking {
  word: string;
  /** Natural-log probability of the word and its full stop after a space. */
  logProb: number;
  /** The word's share of the probability the 21 answers receive together. */
  share: number;
}

export interface AnswerProbe {
  ranking: AnswerRanking[];
  pick: string;
  correct: number;
  total: number;
  /** How many different answers the probe gave across all the facts. */
  distinctPicks: number;
  facts: ProbeFact[];
}

/** Splits "four plus eight is twelve." into its question and answer. */
export function splitFact(sentence: string): { question: string; truth: string } {
  const at = sentence.lastIndexOf(" is ");
  return { question: sentence.slice(0, at + 3), truth: sentence.slice(at + 4).replace(/\.$/, "") };
}

function logSoftmaxRow(weights: Float32Array, row: number): Float64Array {
  const offset = row * TINY_VOCAB_SIZE;
  let maximum = -Infinity;
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) maximum = Math.max(maximum, weights[offset + k]);
  let total = 0;
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) total += Math.exp(weights[offset + k] - maximum);
  const logTotal = maximum + Math.log(total);
  const out = new Float64Array(TINY_VOCAB_SIZE);
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) out[k] = weights[offset + k] - logTotal;
  return out;
}

/** log P(word + "." | the character before it is a space) under the bigram table. */
export function answerLogProb(weights: Float32Array, word: string): number {
  let previous = SPACE;
  let total = 0;
  for (const id of encodeTinyText(`${word}.`)) {
    total += logSoftmaxRow(weights, previous)[id];
    previous = id;
  }
  return total;
}

export function answerProbe(weights: Float32Array): AnswerProbe {
  const scored = ANSWER_WORDS.map((word) => ({ word, logProb: answerLogProb(weights, word) }));
  const top = Math.max(...scored.map((entry) => entry.logProb));
  const mass = scored.reduce((sum, entry) => sum + Math.exp(entry.logProb - top), 0);
  const ranking = scored
    .map((entry) => ({ ...entry, share: Math.exp(entry.logProb - top) / mass }))
    .sort((left, right) => right.logProb - left.logProb);
  const pick = ranking[0].word;
  const facts = ARITHMETIC.heldOut.map((sentence) => {
    const { question, truth } = splitFact(sentence);
    // The score of each answer depends only on the space before it, so every question gets the same pick.
    return { question, truth, pick };
  });
  return {
    ranking,
    pick,
    correct: facts.filter((fact) => fact.truth === fact.pick).length,
    total: facts.length,
    distinctPicks: new Set(facts.map((fact) => fact.pick)).size,
    facts,
  };
}
