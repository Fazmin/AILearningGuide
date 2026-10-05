/**
 * The "Should you fine-tune?" rule of thumb.
 *
 * This is a hand-written decision table, not a measurement and not a model: four yes/no
 * answers go in, a recommended route and the reasons come out, and every rule that can fire
 * is listed on screen. Real decisions also weigh cost, latency, privacy and risk, which the
 * four questions do not capture.
 */

export type DecisionKey = "knowledgeChanges" | "needsCitations" | "needsConsistency" | "hasExamples";

export type DecisionAnswers = Record<DecisionKey, boolean>;

/** Control order on screen and in the stored state. */
export const DECISION_KEYS: readonly DecisionKey[] = [
  "knowledgeChanges",
  "needsCitations",
  "needsConsistency",
  "hasExamples",
];

/** Every answer starts as "no", which lands on the cheapest route. */
export const DEFAULT_DECISION: DecisionAnswers = {
  knowledgeChanges: false,
  needsCitations: false,
  needsConsistency: false,
  hasExamples: false,
};

export const DECISION_QUESTIONS: Record<DecisionKey, string> = {
  knowledgeChanges: "Does the knowledge the answers depend on change often?",
  needsCitations: "Must every answer cite its source?",
  needsConsistency: "Do you need a consistent format, style or behaviour?",
  hasExamples: "Do you have hundreds of good examples and an evaluation set?",
};

/** Short names for the four answers, used to label each printed reason. */
export const DECISION_LABELS: Record<DecisionKey, string> = {
  knowledgeChanges: "Changing facts",
  needsCitations: "Citations",
  needsConsistency: "Format or behaviour",
  hasExamples: "Examples and evaluation set",
};

export type RouteId = "prompt" | "examples" | "retrieval" | "retrieval-and-fine-tune" | "fine-tune";

export interface Route {
  id: RouteId;
  title: string;
  /** What to do next, in one sentence. */
  next: string;
}

export const ROUTES: Record<RouteId, Route> = {
  prompt: {
    id: "prompt",
    title: "Prompt first",
    next: "Write a clear prompt and measure it on a few real cases. Only a gap you can name justifies more.",
  },
  examples: {
    id: "examples",
    title: "Prompt with examples, then collect data",
    next: "Show the format in the prompt with a few worked examples, and start collecting examples and an evaluation set so a fine-tune becomes possible later.",
  },
  retrieval: {
    id: "retrieval",
    title: "Retrieval",
    next: "Index the source documents and put the matching passages in the prompt, so answers use current text and can point to it.",
  },
  "retrieval-and-fine-tune": {
    id: "retrieval-and-fine-tune",
    title: "Retrieval for the facts, fine-tuning for the style",
    next: "Keep the facts in retrieval, fine-tune only the format or behaviour, and score both against your evaluation set, including the old behaviour you must not lose.",
  },
  "fine-tune": {
    id: "fine-tune",
    title: "Fine-tune",
    next: "Train on your examples and score the result on the evaluation set and on the behaviour you want to keep, then budget for retraining when the base model changes.",
  },
};

/** One sentence per answer. Each is tied to its question and its value, and says nothing else. */
const LINES: Record<DecisionKey, Record<"yes" | "no", string>> = {
  knowledgeChanges: {
    yes: "Facts change often. A fine-tuned model's facts are frozen when training stops, so retrieval is how they stay current.",
    no: "Facts are stable, so staleness gives no reason to retrieve.",
  },
  needsCitations: {
    yes: "Answers must cite sources. A fine-tuned model cannot point to where a fact came from; retrieval can show the passage it read.",
    no: "No citations are needed, so source tracking gives no reason to retrieve.",
  },
  needsConsistency: {
    yes: "A consistent format or behaviour is needed. Examples teach behaviour, in the prompt first and in the weights when there are enough.",
    no: "No fixed format or behaviour is needed, so a clear prompt can ask for how the answer should look.",
  },
  hasExamples: {
    yes: "You have hundreds of good examples and an evaluation set: enough to train a fine-tune and to check it for regressions.",
    no: "You lack the examples or the evaluation set, so a fine-tune could neither be trained well nor checked.",
  },
};

export interface DecisionLine {
  key: DecisionKey;
  /** The answer this line describes; it always equals the learner's current answer. */
  value: boolean;
  text: string;
}

export interface DecisionRule {
  /** The condition in words, as printed in the rule table. */
  when: string;
  matches: (answers: DecisionAnswers) => boolean;
  route: RouteId;
  because: string;
}

const needsRetrieval = (answers: DecisionAnswers) => answers.knowledgeChanges || answers.needsCitations;
const canFineTune = (answers: DecisionAnswers) => answers.needsConsistency && answers.hasExamples;

/** The first rule that matches wins, so the order is part of the rule. */
export const DECISION_RULES: readonly DecisionRule[] = [
  {
    when: "Facts change or need citations, and you need a consistent format, and you have examples and an evaluation set",
    matches: (answers) => needsRetrieval(answers) && canFineTune(answers),
    route: "retrieval-and-fine-tune",
    because:
      "Facts that change or need a source belong in retrieval. The format is behaviour, and you have the data and the evaluation set to fine-tune it and check it.",
  },
  {
    when: "Facts change or need citations",
    matches: needsRetrieval,
    route: "retrieval",
    because:
      "Facts that change or need a source belong in retrieval, not in the weights. Ask for any format in the prompt; fine-tuning for it can wait for data and an evaluation set.",
  },
  {
    when: "You need a consistent format or behaviour, and you have examples and an evaluation set",
    matches: canFineTune,
    route: "fine-tune",
    because:
      "Your facts are stable and uncited, you need consistent behaviour, and you have the data and the evaluation set to train it and to check what it forgets.",
  },
  {
    when: "You need a consistent format or behaviour",
    matches: (answers) => answers.needsConsistency,
    route: "examples",
    because:
      "You need consistent behaviour but cannot yet train a fine-tune or check one. Worked examples in the prompt come first, and they become training data later.",
  },
  {
    when: "None of the rows above matched",
    matches: () => true,
    route: "prompt",
    because:
      "Nothing here needs more than a good prompt. Start there, and keep any examples you gather as an evaluation set before you consider a fine-tune.",
  },
];

export interface Recommendation {
  /** Zero-based index into DECISION_RULES of the rule that fired. */
  rule: number;
  route: Route;
  because: string;
  /** One line per question, in control order. */
  lines: DecisionLine[];
}

export function recommend(answers: DecisionAnswers): Recommendation {
  const rule = DECISION_RULES.findIndex((candidate) => candidate.matches(answers));
  const fired = DECISION_RULES[rule];
  return {
    rule,
    route: ROUTES[fired.route],
    because: fired.because,
    lines: DECISION_KEYS.map((key) => ({
      key,
      value: answers[key],
      text: LINES[key][answers[key] ? "yes" : "no"],
    })),
  };
}

/** Every one of the 16 answer combinations, in a fixed order, for tests and for the lesson's counts. */
export function allCombinations(): DecisionAnswers[] {
  return Array.from({ length: 1 << DECISION_KEYS.length }, (_, mask) => ({
    knowledgeChanges: Boolean(mask & 1),
    needsCitations: Boolean(mask & 2),
    needsConsistency: Boolean(mask & 4),
    hasExamples: Boolean(mask & 8),
  }));
}
