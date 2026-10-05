/**
 * The timeline card's stops, oldest first. The first stops follow the early-AI record, and each early claim
 * names its source in the lesson and in card-info.ts: the 1955 Dartmouth proposal (McCarthy, Minsky, Rochester,
 * and Shannon, reprinted in AI Magazine 27(4), 2006); the Wikipedia articles on the Lighthill report and on AI
 * winters, which cite Lighthill's 1973 report, the Mansfield Amendment, and the 1987 collapse of the Lisp-machine
 * market; and the Wikipedia article on XCON, which cites McDermott's 1980 AAAI paper.
 */
export type Era = {
  year: number;
  title: string;
  idea: string;
  stalled: string;
};

export const ERAS: readonly Era[] = [
  {
    year: 1956,
    title: "Dartmouth workshop",
    idea: "McCarthy, Minsky, Rochester, and Shannon proposed a two-month summer study of artificial intelligence at Dartmouth College, on the conjecture that any feature of intelligence can be described precisely enough for a machine to simulate it.",
    stalled: "It named problems, not a method. The proposal lists seven open problems, among them language, neuron nets, and self-improvement, and expects a significant advance on one or more in a single summer.",
  },
  {
    year: 1958,
    title: "Perceptron",
    idea: "A linear threshold unit that learns weights from labeled examples.",
    stalled: "A single layer cannot represent XOR, so the method looked finished.",
  },
  {
    year: 1969,
    title: "Perceptrons critique",
    idea: "Minsky and Papert's book Perceptrons proved what a single layer cannot compute, XOR among it.",
    stalled: "Funding and attention moved from neural nets to rule lists and search for over a decade.",
  },
  {
    year: 1974,
    title: "First AI winter",
    idea: "Sir James Lighthill's 1973 report for the UK Science Research Council found that in no part of the field had discoveries produced the major impact promised. It became the basis for the British government's decision to end support for AI research in most British universities, and by 1974 funding for AI projects was hard to find. The first winter is dated roughly 1974 to 1980.",
    stalled: "Techniques that worked in small, restricted domains did not scale to realistic problems: the detailed knowledge a program needed grew too large to enter by hand.",
  },
  {
    year: 1980,
    title: "Expert systems",
    idea: "Specialist knowledge written as if-then rules. XCON, a rule-based configurer for Digital Equipment Corporation's VAX orders, was written in 1978 and first went into use in 1980; it eventually held about 2,500 rules. No parameters are fitted, so this is AI that is not machine learning.",
    stalled: "The earliest successful expert systems, XCON among them, proved too expensive to maintain: they were difficult to update, could not learn, and were brittle outside the cases their authors anticipated.",
  },
  {
    year: 1986,
    title: "Backpropagation",
    idea: "Rumelhart, Hinton, and Williams showed a stack of layers can train by sending a slope backward through the graph.",
    stalled: "Compute and labeled data were still too thin for deep nets to win.",
  },
  {
    year: 1987,
    title: "Second AI winter",
    idea: "In 1987 the market for specialized Lisp-based AI hardware collapsed, as workstations and portable Lisp environments made dedicated Lisp machines economically unviable. The second winter is dated from about 1987.",
    stalled: "Hand-written rule bases were costly to keep current, and the special-purpose machines that ran them lost their market to cheaper general-purpose computers.",
  },
  {
    year: 1997,
    title: "LSTM memory",
    idea: "Hochreiter and Schmidhuber's gated hidden state carries a signal, and its gradient, across a longer sequence.",
    stalled: "It reads one step at a time, so training cannot run a sequence in parallel, and very long spans still fade.",
  },
  {
    year: 2012,
    title: "AlexNet",
    idea: "Two GPUs plus 1.2 million labeled ImageNet images let a deep convnet win the 2012 challenge by a wide margin.",
    stalled: "Each task still needed its own labeled set and its own architecture.",
  },
  {
    year: 2017,
    title: "Transformer",
    idea: "Attention lets every position read every other position in one parallel block, with no recurrence.",
    stalled: "Full attention costs grow with the square of the sequence length, so long context stayed expensive.",
  },
  {
    year: 2022,
    title: "Assistants",
    idea: "Instruction tuning and human-feedback post-training turn a next-token model into a tool people talk to.",
    stalled: "Fluency is not a check on truth: a confident continuation can still be wrong.",
  },
] as const;

export const ERA_YEARS: readonly number[] = ERAS.map((era) => era.year);
export const DEFAULT_ERA_YEAR = ERA_YEARS[0];

/** A version 3 payload stored `era` as an index into the seven stops it then had. */
export const OLD_ERA_YEARS = [1958, 1969, 1986, 1997, 2012, 2017, 2022] as const;

/** Index of the stop whose year is nearest to the given one, so any stored number lands on a real stop. */
export const nearestEraIndex = (year: number) => {
  if (!Number.isFinite(year)) return 0;
  let best = 0;
  for (let index = 1; index < ERA_YEARS.length; index += 1) {
    if (Math.abs(ERA_YEARS[index] - year) < Math.abs(ERA_YEARS[best] - year)) best = index;
  }
  return best;
};

/**
 * Reads a saved timeline position. `eraYear` is the current key. A version 3 payload has only `era`, an index
 * into the old seven stops, which is translated to that stop's year; anything else gives the first stop.
 */
export const eraYearFromState = (eraYear: unknown, oldEra: unknown): number => {
  if (typeof eraYear === "number" && Number.isFinite(eraYear)) return ERA_YEARS[nearestEraIndex(eraYear)];
  if (typeof oldEra === "number" && Number.isFinite(oldEra)) {
    const index = Math.min(OLD_ERA_YEARS.length - 1, Math.max(0, Math.round(oldEra)));
    return OLD_ERA_YEARS[index];
  }
  return DEFAULT_ERA_YEAR;
};
