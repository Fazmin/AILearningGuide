import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: 3.2B
 * parameters on 28B tokens under the Hoffmann 2022 fit, Budget on Free, and 32 layers of width 4,096
 * in N from dimensions. `parameters` and `data` are in billions, so `parameters: 10, data: 2` is
 * 10B parameters on 2B tokens.
 */
const scenes: LabScene[] = [
  {
    id: "formula",
    sections: ["The formula: a floor plus two shrinking parts", "What it is"],
    title: "The formula, filled in",
    look: "The last line of the formula fills in your run: the floor, then the solid size term, then the hatched data term, the bigger of the two here.",
    focus: "Loss breakdown",
  },
  {
    id: "best-split",
    sections: ["The best split for one budget", "Why a budget must be split"],
    title: "The bottom of the U",
    look: "Your thick curve is one budget: the diamond marks the bottom of its U, and the hollow square marks 20 tokens per parameter.",
    step: 2,
    focus: "IsoFLOP curves",
  },
  {
    id: "frontier-line",
    sections: ["A straight line on a special graph"],
    title: "A straight frontier",
    look: "The dashed line is the best loss at each budget with the floor taken off, and on these axes it runs perfectly straight.",
    step: 4,
    focus: "Frontier power law",
  },
  {
    id: "starting-run",
    sections: ["Read the starting run", "How to play with it"],
    title: "The starting run",
    look: "Read the four readouts under the chart, then find your filled dot, the diamond and the hollow square on the thick curve.",
    step: 2,
    focus: "IsoFLOP curves",
  },
  {
    id: "change-size",
    sections: ["Change the size, change the budget"],
    title: "A bigger model costs more",
    look: "Drag Parameters and watch your dot jump from one thin curve to another, with the thick curve following it.",
    step: 0,
    focus: "IsoFLOP curves",
  },
  {
    id: "more-text",
    sections: ["Feed the model more text"],
    title: "Feed it more text",
    look: "Drag Training tokens upward and watch the hatched bar shrink until Doubling N lowers L by overtakes Doubling D lowers L by.",
    step: 1,
    focus: "Loss breakdown",
  },
  {
    id: "two-bad-splits",
    sections: ["Two bad splits on one curve", "Two failures on one curve"],
    title: "Too big for its text",
    look: "Your dot sits high on the right side of its curve; set Parameters to 200M and Training tokens to 100B, and it lands on the left side of the same curve.",
    step: 2,
    state: { parameters: 10, data: 2 },
    focus: "IsoFLOP curves",
  },
  {
    id: "walk-curve",
    sections: ["Walk one curve", "Walk the curve"],
    title: "Walk one curve",
    look: "With Budget on Hold compute, drag Parameters toward the diamond and watch Gap to optimum shrink as Training tokens moves the other way.",
    step: 3,
    state: { budget: "hold" },
    focus: "IsoFLOP curves",
  },
  {
    id: "read-frontier",
    sections: ["Read the frontier"],
    title: "Slope of the frontier",
    look: "Read Slope and 10× compute keeps, then switch Fit to Epoch 2024 refit and watch the line stay straight while both readouts change.",
    step: 4,
    focus: "Frontier power law",
  },
  {
    id: "build-n",
    sections: ["Build N from dimensions"],
    title: "N from a model’s shape",
    look: "Under N from dimensions, press Use as Parameters, then double Width d and compare the new N on the line with the old one.",
    step: 5,
    focus: "Scaling law controls",
  },
  {
    id: "loss-floor",
    sections: ["Loss has a floor"],
    title: "The largest run allowed",
    look: "At the largest run the sliders allow, the size and data terms on the formula line are tiny, yet the total still sits above the floor.",
    step: 2,
    state: { parameters: 1000, data: 30000 },
    focus: "Loss breakdown",
  },
];

export default scenes;
