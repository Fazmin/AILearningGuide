import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `stage` runs from 0 (Dataset) to 6 (Checkpoint).
 * `checkpoint` indexes the run's saved copies: the default run saves nine, every 48 steps, so 0 is
 * step 0 and 8 is step 384, while at Batch size 2 it saves ten, every 373 steps, so 4 is step 1492.
 * A value past the end, such as 99, means the last copy, which is what the Batch size slider sets.
 */
const scenes: LabScene[] = [
  {
    id: "sheet-of-guesses",
    sections: ["The model: a sheet of guesses", "What it is"],
    title: "One row of the sheet",
    look: "The Forward stage reads only the “o” row after “the fo”, and Top prediction and Runner-up name its two likeliest next characters.",
    step: 0,
    state: { stage: 2 },
    focus: "One pass through the training loop",
  },
  {
    id: "scoring",
    sections: ["Scoring a guess: cross-entropy and nats"],
    title: "Scoring in nats",
    look: "The Loss stage shows Loss at this checkpoint, Its perplexity and the Uniform baseline for the checkpoint picked with the Checkpoint slider.",
    step: 0,
    state: { stage: 3 },
    focus: "One pass through the training loop",
  },
  {
    id: "backward-gradient",
    sections: ["Calculating the gradient: the backward pass"],
    title: "The gradient for o → g",
    look: "At step 0 every bar in the Backward strip stands at the same small height except g’s, then drag Checkpoint to the end and watch g’s bar barely move.",
    step: 0,
    state: { stage: 4, checkpoint: 0 },
    focus: "One pass through the training loop",
  },
  {
    id: "three-counts",
    sections: ["Three ways to count: batch, step and epoch"],
    title: "Batches, steps and epochs",
    look: "The Batch stage shows Batch size, Batches / epoch and Total steps, and moving the Batch size slider changes all three.",
    step: 0,
    state: { stage: 1 },
    focus: "One pass through the training loop",
  },
  {
    id: "counting-passes",
    sections: ["Counting passes is not counting learning", "Why it’s here"],
    title: "Twelve epochs at batch size 2",
    look: "Batch size is 2 at 12 Epochs, so read the step count and Final loss on the loss chart, then drag Batch size to 64 and compare.",
    step: 2,
    state: { batchSize: 2, checkpoint: 99, stage: 1 },
    focus: "Measured training loss",
  },
  {
    id: "walk-loop",
    sections: ["Walk the loop", "How to play with it"],
    title: "Walk the loop",
    look: "Click each stage in the Training loop stages strip in turn, and at Backward find the one long bar in the strip of 30 bars.",
    step: 0,
    state: { stage: 0 },
    focus: "One pass through the training loop",
  },
  {
    id: "resize-batch",
    sections: ["Resize the batch"],
    title: "Resize the batch",
    look: "Drag Batch size from one end to the other with Epochs at 12, and watch Total steps on the Batch stage and Final loss on the loss chart.",
    step: 2,
    state: { stage: 1 },
    focus: "Run controls and checkpoint timeline",
  },
  {
    id: "reload-checkpoint",
    sections: ["Reload a checkpoint"],
    title: "Back to step 0",
    look: "Drag Checkpoint from step 0 to the end and watch Its loss, the big dot on the chart, the sample text and the next-character list move together.",
    step: 3,
    state: { checkpoint: 0, stage: 6 },
    focus: "Run controls and checkpoint timeline",
  },
  {
    id: "step-one-batch",
    sections: ["Step one batch"],
    title: "The first real update",
    look: "One press from step 0 is already applied, so compare the two before → after losses and read the table of rows that moved, then press again.",
    step: 4,
    state: { checkpoint: 0, stepsTaken: 1 },
    focus: "Step one batch",
  },
  {
    id: "step-backfire",
    sections: ["Watch a step backfire"],
    title: "Handfuls of two",
    look: "Press Step one batch again and again, and watch Batch loss fall every time while Whole-corpus loss often goes up.",
    step: 4,
    state: { batchSize: 2, checkpoint: 4 },
    focus: "Step one batch",
  },
  {
    id: "count-memory",
    sections: ["Count the memory"],
    title: "Bytes per parameter",
    look: "Read Bytes per parameter and the three table rows, then switch Precision to fp32 and Optimizer to SGD and watch which row and total change.",
    step: 5,
    focus: "Where the memory goes",
  },
  {
    id: "jagged-and-smooth",
    sections: ["What to notice"],
    title: "Jagged line, smooth line",
    look: "On the chart the thin batch-loss line stays jagged while the thick whole-corpus line falls smoothly, and the ticks under the axis mark each pass.",
    step: 1,
    focus: "Measured training loss",
  },
];

export default scenes;
