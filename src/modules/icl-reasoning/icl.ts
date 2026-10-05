/**
 * Two honest toys for in-context learning: what the examples rule out, and what copying can do.
 *
 * 1. Rule elimination: six candidate rules are checked against the literal examples
 *    in the prompt. A uniform prior over the rules that survive gives an answer
 *    distribution for the query (a noise-free version of the "ICL as implicit
 *    Bayesian inference" view).
 * 2. A hand-set induction head over the prompt's word tokens: attend to the token
 *    after an earlier occurrence of the current token, preferring occurrences whose
 *    previous token also matches, then copy it.
 *
 * The chain-of-thought scratchpad toy lives in scratchpad.ts. Sampling and voting are covered
 * by the Reasoning models module.
 */

export interface Example {
  word: string;
  answer: number;
}

export const EXAMPLES: readonly Example[] = [
  { word: "glim", answer: 4 },
  { word: "plint", answer: 5 },
  { word: "soravel", answer: 7 },
  { word: "trelix", answer: 6 },
  { word: "ballot", answer: 6 },
];

export const QUERIES = ["florin", "letter", "soravel"] as const;
export type Query = (typeof QUERIES)[number];

export const INSTRUCTION = "Count the letters in each word.";

const VOWELS = new Set(["a", "e", "i", "o", "u"]);
const vowels = (word: string) => [...word].filter((char) => VOWELS.has(char)).length;
const consonants = (word: string) => [...word].filter((char) => /[a-z]/.test(char) && !VOWELS.has(char)).length;

export interface Rule {
  id: string;
  name: string;
  apply: (word: string) => number;
  /** True for the one rule the instruction line describes. */
  named: boolean;
}

export const RULES: readonly Rule[] = [
  { id: "letters", name: "count the letters", apply: (word) => word.length, named: true },
  { id: "distinct", name: "count distinct letters", apply: (word) => new Set(word).size, named: false },
  { id: "consonants-plus-1", name: "consonants + 1", apply: (word) => consonants(word) + 1, named: false },
  { id: "vowels-plus-3", name: "vowels + 3", apply: (word) => vowels(word) + 3, named: false },
  { id: "consonants", name: "count consonants", apply: consonants, named: false },
  { id: "vowels", name: "count vowels", apply: vowels, named: false },
];

/** The first example (by position) that a rule gets wrong, or -1 if it fits every shown example. */
export function firstMiss(rule: Rule, shown: number) {
  return EXAMPLES.slice(0, shown).findIndex((example) => rule.apply(example.word) !== example.answer);
}

/**
 * Posterior over rules: prior (uniform, or all on the named rule when the instruction
 * is present) times a noise-free likelihood (1 if the rule reproduces every shown
 * example, else 0), normalized.
 */
export function rulePosterior(shown: number, instruction: boolean) {
  const prior = RULES.map((rule) => (instruction ? (rule.named ? 1 : 0) : 1 / RULES.length));
  const unnormalized = RULES.map((rule, index) => prior[index] * (firstMiss(rule, shown) === -1 ? 1 : 0));
  const total = unnormalized.reduce((sum, value) => sum + value, 0);
  return unnormalized.map((value) => (total > 0 ? value / total : 0));
}

export function answerDistribution(shown: number, instruction: boolean, query: string) {
  const posterior = rulePosterior(shown, instruction);
  const mass = new Map<number, number>();
  RULES.forEach((rule, index) => {
    if (posterior[index] <= 0) return;
    const answer = rule.apply(query);
    mass.set(answer, (mass.get(answer) ?? 0) + posterior[index]);
  });
  return [...mass.entries()]
    .map(([answer, probability]) => ({ answer, probability }))
    .sort((a, b) => b.probability - a.probability || a.answer - b.answer);
}

/* ------------------------------------------------------------ induction */

export const NEWLINE = "⏎";
export const ARROW = "→";
export const MATCH_SHARPNESS = 4;

export function promptTokens(shown: number, instruction: boolean, query: string) {
  const tokens: string[] = [];
  if (instruction) tokens.push(...INSTRUCTION.split(" "), NEWLINE);
  EXAMPLES.slice(0, shown).forEach((example) => tokens.push(example.word, ARROW, String(example.answer), NEWLINE));
  tokens.push(query, ARROW);
  return tokens;
}

export function promptText(shown: number, instruction: boolean, query: string) {
  const lines: string[] = [];
  if (instruction) lines.push(INSTRUCTION);
  EXAMPLES.slice(0, shown).forEach((example) => lines.push(`${example.word} ${ARROW} ${example.answer}`));
  lines.push(`${query} ${ARROW}`);
  return lines.join("\n");
}

/**
 * Induction head at the last position t. Position i scores
 *   m_i = [x_{i−1} = x_t] · (1 + [x_{i−2} = x_{t−1}])
 * so a token that follows an earlier copy of the current token scores 1, and 2 if the
 * token before that also matches. Weights are softmax(κ·m) over i < t. The head's output
 * is the attention-weighted vote for the token at each attended position.
 */
export function inductionHead(tokens: ReadonlyArray<string>, sharpness = MATCH_SHARPNESS) {
  const t = tokens.length - 1;
  const scores = tokens.slice(0, t).map((_, i) => {
    if (i < 1 || tokens[i - 1] !== tokens[t]) return 0;
    return 1 + (i >= 2 && tokens[i - 2] === tokens[t - 1] ? 1 : 0);
  });
  const top = Math.max(...scores, 0);
  const exps = scores.map((score) => Math.exp(sharpness * (score - top)));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const weights = exps.map((value) => value / total);
  const votes = new Map<string, number>();
  weights.forEach((weight, i) => votes.set(tokens[i], (votes.get(tokens[i]) ?? 0) + weight));
  const output = [...votes.entries()]
    .map(([token, probability]) => ({ token, probability }))
    .sort((a, b) => b.probability - a.probability);
  return { scores, weights, output };
}
