import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults:
 * degree 5, epoch 100, η 0.1, no weight decay, batch 16, seed 1. `epochIndex`,
 * `learningRateIndex` and `weightDecayIndex` index the lab's fixed lists, so epochIndex 13 is
 * epoch 13, 82 is epoch 10,000 and 0 is epoch 0; learningRateIndex 4 is η 0.3; weightDecayIndex
 * 5 is λ 0.03.
 */
const scenes: LabScene[] = [
  {
    id: "two-scores",
    sections: ["Two scores from one model", "What it is"],
    title: "Two scores from one model",
    look: "The solid blue line is the training loss, the dashed orange line is the validation loss, and Gap (val − train) reads the distance between them.",
    step: 1,
    focus: "Training curves",
  },
  {
    id: "true-curve",
    sections: ["The model in this lab", "The run in this lab is real"],
    title: "The dots and the true curve",
    look: "The dashed true curve is hidden from the model, and every dot sits a small random nudge above or below it.",
    step: 0,
    focus: "Fitted function",
  },
  {
    id: "underfitting",
    sections: ["Three ways a run can go", "The three regimes"],
    title: "A straight line underfits",
    look: "At degree 1 the badge reads underfitting and both lines level off high, so raise Polynomial degree and watch the badge change.",
    step: 1,
    state: { degree: 1 },
    focus: "Training curves",
  },
  {
    id: "flat-start",
    sections: ["How training moves the weights"],
    title: "Training starts flat",
    look: "At epoch 0 every weight is zero, so the curve is a flat line, and dragging Epoch shows it bend toward the dots.",
    step: 1,
    state: { epochIndex: 0 },
    focus: "Fitted function",
  },
  {
    id: "bends-to-noise",
    sections: ["Training cannot tell pattern from noise"],
    title: "Bending toward the noise",
    look: "The curve bends toward the filled training dots and misses hollow ones, most of all at the right edge, where no filled dot holds it in place.",
    step: 2,
    state: { degree: 11, epochIndex: 82 },
    focus: "Fitted function",
  },
  {
    id: "damage-late",
    sections: ["Why the damage shows up late", "Why the damage arrives late"],
    title: "Before the damage starts",
    look: "The marker sits at the green ring, where the pattern is learned but little noise is matched, so drag Epoch on and watch validation climb.",
    step: 2,
    state: { degree: 11, epochIndex: 13 },
    focus: "Training curves",
  },
  {
    id: "split-strip",
    sections: ["Split the data", "How to play with it"],
    title: "Split the data",
    look: "Filled dots above the line on the split strip train the model and hollow dots below it only score it, so find the same dots on Fitted function.",
    step: 0,
    focus: "Training dynamics controls",
  },
  {
    id: "watch-curves",
    sections: ["Watch the curves"],
    title: "Watch the curves",
    look: "Drag Epoch slowly to the end and follow the two dots as both lines drop fast and then level off, with the green ring at validation’s lowest point.",
    step: 1,
    state: { epochIndex: 0 },
    focus: "Training curves",
  },
  {
    id: "trigger-overfitting",
    sections: ["Trigger overfitting"],
    title: "Trigger overfitting",
    look: "Drag Epoch to the end and watch the badge switch to overfitting as validation climbs while the training line keeps falling.",
    step: 2,
    state: { degree: 11 },
    focus: "Training curves",
  },
  {
    id: "regularize",
    sections: ["Regularize"],
    title: "Weight decay at degree 11",
    look: "With Weight decay on, validation at the last epoch stays close to its best, so set it back to off and press Stop at lowest validation to compare.",
    step: 3,
    state: { degree: 11, weightDecayIndex: 5, epochIndex: 82 },
    focus: "Training curves",
  },
  {
    id: "small-batch",
    sections: ["More comparisons to run"],
    title: "A batch too small",
    look: "With Batch size 4 at this Learning rate the badge reads diverged, so switch Batch size to 8 or 16 and watch the run train normally.",
    state: { learningRateIndex: 4, batchSize: 4 },
    focus: "Training curves",
  },
  {
    id: "capacity-over-time",
    sections: ["What to notice"],
    title: "Capacity changes with time",
    look: "This early in training the dashed validation line is nearly flat from degree 5 up, so drag Epoch to the end and watch it spread apart.",
    step: 1,
    state: { epochIndex: 13 },
    focus: "Capacity sweep",
  },
];

export default scenes;
