import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `attnLayer` and `attnHead` are 0-based, so
 * `attnLayer: 0, attnHead: 3` is L1 H4 and the default `1, 2` is L2 H3; `ablated` holds head ids
 * such as "L2H3". `patch` is the selected patching position: the lab moves it to the corrupted
 * letter whenever the period or Corrupt changes, so scenes that change either set it too (period 13
 * puts the letter to copy at 11, and the control letter at period 8 sits at 4).
 */
const scenes: LabScene[] = [
  {
    id: "two-halves",
    sections: ["Two halves of the copying trick", "The model and the behavior"],
    title: "The previous-letter half",
    look: "L1 H4’s dark cells sit on the dashed outlines just left of the diagonal, and pressing L2 H3 moves them onto the green outlines.",
    step: 0,
    state: { attnLayer: 0, attnHead: 3 },
    focus: "Copy task and attention",
  },
  {
    id: "find-induction",
    sections: ["Step 1: find an induction pattern", "How to play with it"],
    title: "Find the induction stripe",
    look: "L2 H3’s dark cells follow the solid green outlines down the grid, and the table beside it scores all eight heads in two columns.",
    step: 0,
    focus: "Copy task and attention",
  },
  {
    id: "ablate-head",
    sections: ["Step 2: ablate a head"],
    title: "L2 H3 switched off",
    look: "With L2 H3 off, the red bars collapse while Attention on the induction key reads the same on both sides of the arrow.",
    step: 1,
    state: { ablated: ["L2H3"] },
    focus: "Head ablation",
  },
  {
    id: "patch-step",
    sections: ["Step 3: patch activations"],
    title: "Patch one square",
    look: "The After block 1 square under the amber column is already picked, so read its recovery under the grid, then click the last square in that row.",
    step: 2,
    focus: "Activation patching",
  },
  {
    id: "unseen-period",
    sections: ["Step 4: try an unseen period"],
    title: "Period 13, never trained",
    look: "Repeat period is on 13 · unseen, so compare Copy accuracy, the green outlines and the L2 H3 row of the table with what you saw at 8.",
    step: 3,
    state: { period: 13, patch: 11 },
    focus: "Copy task and attention",
  },
  {
    id: "textbook-picture",
    sections: ["The attention picture matches the textbook"],
    title: "Four heads stand out",
    look: "Read Copy accuracy, then the table beside the grid, where two layer-1 heads fill the previous letter column and two layer-2 heads fill the other one.",
    step: 0,
    focus: "Copy task and attention",
  },
  {
    id: "pattern-not-distance",
    sections: ["The heads follow the pattern, not a fixed distance", "The period test passes"],
    title: "The stripe moves with the period",
    look: "At period 13 the green outlines sit further back, and L2 H3’s dark cells move with them, as the line under the grid says.",
    step: 3,
    state: { period: 13, patch: 11 },
    focus: "Copy task and attention",
  },
  {
    id: "single-heads",
    sections: ["Single heads are needed", "Single heads are necessary"],
    title: "One head off at a time",
    look: "The alone figure on each head button shows how many points Copy accuracy changes when only that head is off.",
    step: 1,
    focus: "Head ablation",
  },
  {
    id: "layer-one-feeds",
    sections: ["Layer 1 feeds layer 2, as far as ablation can tell", "Layer 1 feeds layer 2, as far as ablation can say"],
    title: "L1 H4 switched off",
    look: "With L1 H4 off, the numbers after the arrow in Attention on the induction key fall, because the layer-2 heads lose their aim.",
    step: 1,
    state: { ablated: ["L1H4"] },
    focus: "Head ablation",
  },
  {
    id: "mean-ablation",
    sections: ["Zero exaggerates, and the layer-1 result holds up", "Zero overstates, and the layer-1 result survives"],
    title: "Mean instead of zero",
    look: "With Ablation on Mean, switching off L2 H3 costs far fewer points than under Zero, while the layer-1 heads’ alone figures stay large.",
    step: 1,
    state: { ablation: "mean", ablated: ["L2H3"] },
    focus: "Head ablation",
  },
  {
    id: "one-route",
    sections: ["Patching finds one route", "Patching finds one route, with a large gap"],
    title: "One route through the grid",
    look: "In the After block 1 row only the amber column turns green, and in the After block 2 row only the last square does.",
    step: 2,
    focus: "Activation patching",
  },
  {
    id: "control-letter",
    sections: ["A control shows what a bad measurement looks like"],
    title: "An unrelated letter",
    look: "Corrupt is on An unrelated letter, so read the Gap and the warning before you trust any square in the grid.",
    step: 2,
    state: { corruption: "control", patch: 4, layer: 0 },
    focus: "Activation patching",
  },
];

export default scenes;
