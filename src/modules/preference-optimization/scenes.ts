import type { LabScene } from "@app/module-sdk";

/** The editor's three starting lines, `prompt | preferred | rejected`, as `initialState.pairsText` joins them. */
const DEFAULT_PAIRS = [
  "the | fog settles over the harbor. | fog fog fog fog fog.",
  "warm the | pan and add a spoon of oil. | pan and the pan and the pan.",
  "simmer the | sauce until it thickens. | sauce and simmer the sauce.",
];

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the three
 * starting pairs, Beta 0.4, 40 DPO steps, learning rate 0.35 and the base model as the reference, so the
 * cards show the lesson's own numbers unless a scene changes a setting.
 */
const scenes: LabScene[] = [
  {
    id: "implicit-reward",
    sections: ["The implicit reward and the margin", "What it is"],
    title: "One pair’s two rewards",
    look: "Pair 1’s dumbbell puts a circle at the chosen reward and a square at the rejected reward, and the bar between them is the margin.",
    step: 2,
    focus: "Per-pair rewards",
  },
  {
    id: "run-optimizer",
    sections: ["Run the optimizer", "How to play with it"],
    title: "Training from step 0",
    look: "DPO steps reads untrained, so every line sits at zero; raise it in Preference pair editor and watch DPO loss fall and the ordered correctly badge climb.",
    step: 1,
    state: { steps: 0 },
    focus: "Implicit reward margin during training",
  },
  {
    id: "two-sides",
    sections: ["Read the margin and its two sides"],
    title: "The margin and its two sides",
    look: "The dashed chosen reward line rises only a little while the dotted rejected reward line falls far, and the solid margin line is the gap between them.",
    step: 2,
    focus: "Implicit reward margin during training",
  },
  {
    id: "pairs-on-curve",
    sections: ["Watch where each pair sits"],
    title: "Pairs still on the slope",
    look: "At 4 DPO steps the numbered circles still sit on the steep part of the curve; raise DPO steps and watch them slide right while each row’s pull shrinks.",
    step: 1,
    state: { steps: 4 },
    focus: "Where each pair sits on the loss",
  },
  {
    id: "sweep-beta",
    sections: ["Sweep beta"],
    title: "Three betas side by side",
    look: "Each legend entry gives one run’s gap and KL; slide Beta in Preference pair editor and watch all three curves change together.",
    step: 3,
    focus: "Drift from the reference model",
  },
  {
    id: "drift-order-flips",
    sections: ["Beta does two jobs"],
    title: "The drift order flips",
    look: "At only 4 DPO steps the KL order in the legend runs the other way, with the highest beta drifting most.",
    step: 3,
    state: { steps: 4 },
    focus: "Drift from the reference model",
  },
  {
    id: "very-low-beta",
    sections: ["What beta trades away"],
    title: "A very low beta",
    look: "At Beta 0.05 the runs are still far from done, so in the legend the higher beta now gives both the bigger gap and more drift.",
    step: 3,
    state: { beta: 0.05 },
    focus: "Drift from the reference model",
  },
  {
    id: "held-out",
    sections: ["Check the held-out pairs", "Held-out pairs"],
    title: "Pairs it never trained on",
    look: "Below the three training pairs sit H1, H2 and H3, which no step trains on; read which of their blocks say ordered correctly.",
    step: 4,
    focus: "Per-pair rewards",
  },
  {
    id: "own-pair",
    sections: ["Write your own pair"],
    title: "A fourth pair about rain",
    look: "The rain pair, block 4, ends ordered correctly, yet the Held-out pairs heading below still counts the same number ordered.",
    step: 0,
    state: { pairsText: [...DEFAULT_PAIRS, "the | rain falls on the boats. | rain rain rain rain."].join("\n") },
    focus: "Per-pair rewards",
  },
  {
    id: "one-pair",
    sections: ["Train one pair at a time"],
    title: "Training on the pan pair alone",
    look: "Only the pan pair trains now, so read H3’s margin, then swap the line for the fog pair and read it again.",
    step: 4,
    state: { pairsText: DEFAULT_PAIRS[1] },
    focus: "Per-pair rewards",
  },
  {
    id: "sft-reference",
    sections: ["Switch the reference", "Change the reference"],
    title: "An SFT checkpoint as reference",
    look: "With SFT checkpoint as the Reference model, two training footers say the reference already preferred it, yet only one H pair ends ordered.",
    step: 4,
    state: { reference: "sft" },
    focus: "Per-pair rewards",
  },
  {
    id: "chosen-falls",
    sections: ["The margin is bought lopsidedly"],
    title: "Both sides fall",
    look: "With SFT checkpoint as the Reference model, the dashed chosen reward line ends below zero, yet the margin line still climbs.",
    step: 2,
    state: { reference: "sft" },
    focus: "Implicit reward margin during training",
  },
  {
    id: "two-badges",
    sections: ["Passing the practice pairs is not learning the preference", "What to notice"],
    title: "Practice pairs against held-out",
    look: "Compare the ordered correctly badge with the held-out badge beside it, then raise DPO steps and watch the held-out margin line stay near zero.",
    step: 4,
    focus: "Implicit reward margin during training",
  },
];

export default scenes;
