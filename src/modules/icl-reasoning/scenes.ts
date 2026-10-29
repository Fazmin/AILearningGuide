import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `examples` counts how many of the five example lines
 * (glim, plint, soravel, trelix, ballot) the prompt shows, in that order. `hops` and `budget` are the
 * scratchpad card's Hops k and Per-pass budget b; the defaults, k = 6 and b = 3 with the scratchpad off,
 * are the lesson's own walk-through.
 */
const scenes: LabScene[] = [
  {
    id: "weights-fixed",
    sections: ["Learning without changing: in-context learning", "What it is"],
    title: "Only the prompt changes",
    look: "Change Examples, Prompt strategy or Query as much as you like: the answer bars move, but Weights changed stays at 0.",
    step: 0,
    focus: "Prompt as context",
  },
  {
    id: "rules-drop-out",
    sections: ["The examples are clues: task inference"],
    title: "Three clues, two rules left",
    look: "Each ✗ marks a miss and the Weight column names the example that knocked the rule out, so raise Examples to 4 and see that trelix knocks out nothing.",
    step: 0,
    state: { examples: 3 },
    focus: "Prompt as context",
  },
  {
    id: "florin-from-zero",
    sections: ["From rules to a guess about the answer", "Examples as evidence"],
    title: "No examples for florin",
    look: "Read P(florin → 6) and the answer bars, then raise Examples one step at a time and watch the chance on 6 climb.",
    step: 0,
    state: { examples: 0 },
    focus: "Prompt as context",
  },
  {
    id: "letter-split",
    sections: ["When the clues run out"],
    title: "Two rules that disagree",
    look: "The two rules left give different answers for letter, so the bars split; add the fifth example or switch Prompt strategy to State a rule, and one rule wins.",
    step: 1,
    state: { query: "letter", examples: 4 },
    focus: "Prompt as context",
  },
  {
    id: "copy-soravel",
    sections: ["Copying is not the same as working it out", "Copying is not the same as inferring"],
    title: "Copying a seen word",
    look: "The shading under the tokens piles onto the 7 after the earlier soravel line, and the top bar copies it; then set Query to florin and the old answers tie.",
    step: 2,
    state: { query: "soravel", examples: 4 },
    focus: "Copy from context",
  },
  {
    id: "arrow-puzzle",
    sections: ["The lab's arrow puzzle", "Spending compute at answer time"],
    title: "One pass, then two",
    look: "With Scratchpad off the badge says answer wrong, so switch Scratchpad to on and watch the scratch line and a second pass appear.",
    step: 3,
    state: { hops: 6, budget: 3, scratchpad: "off" },
    focus: "Chain-of-thought scratchpad",
  },
  {
    id: "scratchpad-cost",
    sections: ["What the scratchpad costs"],
    title: "The price in passes",
    look: "Passes (one token each) shows what the scratchpad costs here, and the Scratchpad column in the table at the bottom gives the cost for every Hops k.",
    step: 4,
    state: { hops: 7, budget: 3, scratchpad: "on" },
    focus: "Chain-of-thought scratchpad",
  },
  {
    id: "one-pass-limit",
    sections: ["What to notice"],
    title: "Where one pass runs out",
    look: "In the table at the bottom, No scratchpad turns to ✗ once Hops k passes b while Scratchpad stays ✓, so raise Per-pass budget b and watch the first ✗ move down.",
    step: 4,
    state: { budget: 3 },
    focus: "Chain-of-thought scratchpad",
  },
];

export default scenes;
