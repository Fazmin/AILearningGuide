/**
 * The "what is the learning signal" sort. Each authored scenario has exactly one best answer: where the
 * training target comes from. The output type (classification, regression, structure, action) is a
 * separate axis, shown beside the answer so the two are not confused.
 */

export const SIGNALS = [
  {
    id: "supervised",
    label: "supervised",
    source: "Human labels",
    body: "Every example arrives with its answer, written by a person or kept in a record. The model is scored against those answers. Semi-supervised learning is the same idea with labels on only a few examples.",
  },
  {
    id: "unsupervised",
    label: "unsupervised",
    source: "Structure in raw data",
    body: "There is no answer column. The model finds structure in the inputs themselves, such as which records sit close together or which readings look unusual.",
  },
  {
    id: "self",
    label: "self-supervised",
    source: "Labels made from the data itself",
    body: "The data hides part of itself and the model must restore it, so the label is already in the data. No person labels anything, which is why next-word prediction can train on a huge pile of raw text.",
  },
  {
    id: "reward",
    label: "reinforcement",
    source: "Reward after acting",
    body: "No example says the right move. The learner acts, then receives a reward or penalty afterward, and has to work out which choices earned it.",
  },
] as const;

export type SignalId = (typeof SIGNALS)[number]["id"];

export const OUTPUT_KINDS = {
  classification: "classification · the output is one category from a fixed set",
  regression: "regression · the output is a number on a scale",
  structure: "no target · the output is a grouping or a score, not a predicted answer",
  action: "an action · the output is a choice the learner makes",
} as const;

export type OutputKind = keyof typeof OUTPUT_KINDS;

export type Scenario = {
  id: string;
  short: string;
  text: string;
  signal: SignalId;
  output: OutputKind;
  outputNote: string;
  why: string;
};

export const SCENARIOS: readonly Scenario[] = [
  {
    id: "house",
    short: "House sale prices",
    text: "Sale records where every house has a known price.",
    signal: "supervised",
    output: "regression",
    outputNote: "a price",
    why: "A person recorded the price of every house, so each example comes with its answer. Predicting a number is regression.",
  },
  {
    id: "spam",
    short: "Spam emails",
    text: "Emails that people marked as spam or not spam.",
    signal: "supervised",
    output: "classification",
    outputNote: "spam or not spam",
    why: "People supplied the label for each email. Predicting a category is classification.",
  },
  {
    id: "customers",
    short: "Customer records",
    text: "A pile of customer records with no answer column, grouped by similar behavior.",
    signal: "unsupervised",
    output: "structure",
    outputNote: "a group for each record",
    why: "No label exists. The model looks for structure in the inputs, such as which records sit close together. The k-means method in the classical lab works this way.",
  },
  {
    id: "sensors",
    short: "Sensor readings",
    text: "Untagged sensor readings, flagged when they sit far from the usual pattern.",
    signal: "unsupervised",
    output: "structure",
    outputNote: "an unusualness score for each reading",
    why: "Nobody marked which readings are faulty. The model learns what usual looks like from the raw readings and scores the distance from it.",
  },
  {
    id: "text",
    short: "Raw sentences",
    text: "Raw sentences, where the next word of each sentence is the label.",
    signal: "self",
    output: "classification",
    outputNote: "one word from a fixed vocabulary",
    why: "The next word is already in the text, so the label comes free from the data and nobody has to annotate anything. Large language models are pretrained this way. The output is a category, so it is classification over a very large vocabulary.",
  },
  {
    id: "patch",
    short: "Blanked-out photos",
    text: "Photos with a patch blanked out, where the model must paint the hidden pixels back in.",
    signal: "self",
    output: "regression",
    outputNote: "pixel values",
    why: "The hidden pixels are part of the original photo, so the data supplies its own answer. The output is numbers, so this self-supervised task is regression: the signal and the output type are separate choices.",
  },
  {
    id: "game",
    short: "Win-or-loss game",
    text: "A game where only the final win or loss is seen and no single move is marked right or wrong.",
    signal: "reward",
    output: "action",
    outputNote: "the next move",
    why: "No one labels the correct move. The learner plays, receives a reward at the end, and has to work out which moves earned it.",
  },
  {
    id: "robot",
    short: "Charger robot",
    text: "A robot that gets +1 for reaching its charger and −1 for bumping a wall.",
    signal: "reward",
    output: "action",
    outputNote: "the direction to move",
    why: "The only feedback is a reward after the robot acts. No example says which direction was correct in a given room.",
  },
];

export const SCENARIO_IDS: readonly string[] = SCENARIOS.map((item) => item.id);
export const SIGNAL_IDS: readonly string[] = SIGNALS.map((item) => item.id);

export const scenarioById = (id: string) => SCENARIOS.find((item) => item.id === id) ?? SCENARIOS[0];
export const signalById = (id: string) => SIGNALS.find((item) => item.id === id) ?? SIGNALS[0];

/**
 * Placements are stored as `scenarioId:signalId`. Anything malformed, unknown, or repeated is dropped,
 * and a later placement of the same scenario replaces the earlier one.
 */
export function parseSignalPlacements(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const byScenario = new Map<string, string>();
  for (const entry of value) {
    if (typeof entry !== "string") continue;
    const [scenario, signal, ...rest] = entry.split(":");
    if (rest.length > 0 || !SCENARIO_IDS.includes(scenario) || !SIGNAL_IDS.includes(signal)) continue;
    byScenario.set(scenario, signal);
  }
  return SCENARIOS.filter((item) => byScenario.has(item.id)).map((item) => `${item.id}:${byScenario.get(item.id)}`);
}

/** How many scenarios are placed, and how many sit under their own signal. */
export function scoreSignalPlacements(placements: readonly string[]) {
  const placed = new Map(placements.map((entry) => entry.split(":") as [string, string]));
  const matched = SCENARIOS.filter((item) => placed.get(item.id) === item.signal).length;
  return { placed: placed.size, matched, total: SCENARIOS.length };
}
