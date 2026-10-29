import type { LabScene } from "@app/module-sdk";

/** The six default items, `context | correct | plausible distractor`, as `initialState.items` joins them. */
const DEFAULT_ITEMS = [
  "in winter the same harbor | freezes | flowers",
  "the boats stay in | port | town",
  "in summer the harbor fills with | sails | trails",
  "the water stays | warm | wide",
  "the fog returns in the | morning | evening",
  "and the pattern | repeats | retreats",
];

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the six
 * default items, the benchmark source held out, the 1.00× budget multiplier and the Length-normalised
 * rule, so the cards show the lesson's own numbers unless a scene changes a setting. Item-level results
 * always shows the most-trained checkpoint, 80 epochs at the default multiplier.
 */
const scenes: LabScene[] = [
  {
    id: "two-endings",
    sections: ["A test made of two-ending questions", "What it is"],
    title: "Questions with two endings",
    look: "Each line in Benchmark items is a context, the right ending and a believable wrong one, split by upright bars.",
    step: 0,
    focus: "Benchmark editor",
  },
  {
    id: "nonsense-distractor",
    sections: ["1. Write the items"],
    title: "A nonsense distractor",
    look: "The fog item’s wrong ending is now xyzzy, so its bar in Item-level results reaches further right than any other row.",
    step: 0,
    state: { items: DEFAULT_ITEMS.map((line) => line.replace("morning | evening", "morning | xyzzy")).join("\n") },
    focus: "Item-level results",
  },
  {
    id: "read-leaderboard",
    sections: ["2. Read the leaderboard", "How to play with it"],
    title: "The leaderboard at its defaults",
    look: "Each row has a dot for its pass rate and a line for its 95% range, and the leader’s range is shaded down the whole chart.",
    step: 1,
    focus: "Leaderboard",
  },
  {
    id: "error-bar",
    sections: ["How sure is a score?", "The error bar"],
    title: "How wide the range is",
    look: "Read Leader’s standard error under the table, then compare it with how far the leader’s line stretches on the chart.",
    step: 1,
    focus: "Leaderboard",
  },
  {
    id: "paired-callout",
    sections: ["Comparing two checkpoints item by item"],
    title: "The top two, item by item",
    look: "The callout at the bottom subtracts the runner-up’s margin from the leader’s on every item and gives that difference its own range.",
    step: 1,
    focus: "Leaderboard",
  },
  {
    id: "length-head-start",
    sections: ["Why length gets in the way"],
    title: "Totals favour short endings",
    look: "Under Raw log-probability each row compares totals, so the shorter sails and repeats now beat their longer distractors.",
    step: 2,
    state: { rule: "raw" },
    focus: "Item-level results",
  },
  {
    id: "rule-is-choice",
    sections: ["The scoring rule is a choice", "Three scoring rules"],
    title: "Same checkpoints, new rule",
    look: "Raw log-probability is chosen, so the same checkpoints pass more items; switch back to Length-normalised and compare the Passed column.",
    step: 2,
    state: { rule: "raw" },
    focus: "Leaderboard",
  },
  {
    id: "five-attempts",
    sections: ["3. Change the scoring rule"],
    title: "Five attempts each",
    look: "Under Sample and check with Attempts allowed (k) at 5, the least-trained checkpoint leads; drag k down to 1 and watch the order shuffle.",
    step: 2,
    state: { rule: "sample", attempts: 5 },
    focus: "Leaderboard",
  },
  {
    id: "inspect-items",
    sections: ["4. Inspect the items"],
    title: "Item by item at 80 epochs",
    look: "Find the boats row, whose bar grows left of centre for a fail, and the water row, whose bar reaches furthest right.",
    step: 3,
    focus: "Item-level results",
  },
  {
    id: "held-out-control",
    sections: ["The held-out set and the control", "The held-out half"],
    title: "The held-out sentences",
    look: "The Held-out benchmark source box shows the three sentences kept out of training, and the switch above it reads held out.",
    focus: "Contamination control",
  },
  {
    id: "contaminate",
    sections: ["5. Contaminate it"],
    title: "The benchmark leaked",
    look: "The switch now reads leaking; compare Margin shift with Control shift, then press it again to see the clean Leaderboard.",
    step: 4,
    state: { contaminated: true },
    focus: "Contamination control",
  },
  {
    id: "confident-mistake",
    sections: ["What to notice"],
    title: "A confident mistake",
    look: "Even with Attempts allowed (k) at 10, the boats row stays at zero, because none of the 80-epoch picks lands on port.",
    step: 3,
    state: { rule: "sample", attempts: 10 },
    focus: "Item-level results",
  },
];

export default scenes;
