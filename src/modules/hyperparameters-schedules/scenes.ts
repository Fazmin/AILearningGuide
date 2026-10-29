import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults
 * (Cosine decay, peak 0.6, warmup 15%, no clip, no weight decay, batch 16, seed 1, and a
 * 24-layer depth card at ×1.20), so the readouts show the lesson's own numbers unless a scene
 * sets a dial. `inspect` is the Inspect epoch, from 0 to 25.
 */
const scenes: LabScene[] = [
  {
    id: "overshoot",
    sections: ["The picture: walking downhill in fog", "Why it’s here"],
    title: "Strides that overshoot",
    look: "At a peak of 40 the Constant curve jumps up and down and ends above the dotted line, while the three decaying curves still settle low.",
    step: 2,
    state: { peakLearningRate: 40 },
    focus: "Loss for all four schedules",
  },
  {
    id: "four-shapes",
    sections: ["The four shapes", "What it is"],
    title: "Four schedule shapes",
    look: "Warmup + decay is chosen, so its dot at epoch 12 shows the rate after the climb, then click the other Schedule buttons to compare the shapes.",
    step: 0,
    state: { schedule: "warmup-decay" },
    focus: "Learning-rate schedule shapes",
  },
  {
    id: "read-schedule",
    sections: ["Read the schedule before any loss", "How to play with it"],
    title: "Inside the warmup",
    look: "At epoch 1 the dot on Warmup + decay is still in the shaded warmup, so drag Inspect epoch and watch its lr label climb to the peak and fall.",
    step: 0,
    state: { schedule: "warmup-decay", inspect: 1 },
    focus: "Learning-rate schedule shapes",
  },
  {
    id: "watch-winner",
    sections: ["Watch the winner change"],
    title: "Watch the winner change",
    look: "Walk Peak learning rate up from 0.6 and watch Best schedule here under the loss chart change hands.",
    step: 1,
    focus: "Loss for all four schedules",
  },
  {
    id: "break-cap",
    sections: ["Break it, then cap it"],
    title: "Peak 30, batches of 2",
    look: "At a peak of 30 with Batch size 2 the Constant curve runs off the top of the chart, then raise Gradient clip and watch it come back down.",
    step: 3,
    state: { peakLearningRate: 30, batchSize: 2 },
    focus: "Loss for all four schedules",
  },
  {
    id: "seed-band",
    sections: ["Read the seed band"],
    title: "Five shuffles per schedule",
    look: "With Five seeds on, every curve gets a shaded band and the table under the chart lists each schedule’s Spread, so compare it with the gaps between schedules.",
    step: 4,
    state: { seeds: "five" },
    focus: "Loss for all four schedules",
  },
  {
    id: "explode-depth",
    sections: ["Explode a gradient, then try to rescue a vanished one"],
    title: "An exploding gradient",
    look: "Read Input ÷ output gradient and the boxed note, then set Depth clip to 1.0 and compare the table’s Before the clip and After the clip columns.",
    step: 5,
    focus: "Gradient through depth",
  },
  {
    id: "schedule-notice",
    sections: ["On the schedule chart", "What to notice"],
    title: "What Warmup fraction moves",
    look: "Drag Warmup fraction and only the Warmup + decay and One cycle curves change shape, and none of the four curves reaches zero at the right edge.",
    step: 0,
    focus: "Learning-rate schedule shapes",
  },
  {
    id: "stability-notice",
    sections: ["On the stability card"],
    title: "One gradient per step",
    look: "Most dots on the Stability readout sit low, while a row of higher dots, one per epoch, comes from each epoch’s tiny last batch.",
    step: 3,
    focus: "Stability readout",
  },
  {
    id: "weight-decay",
    sections: ["Weight decay is not a schedule"],
    title: "Weight decay on Constant",
    look: "With Weight decay at 0.020, Constant final loss sits well above where it is with Weight decay off.",
    step: 1,
    state: { schedule: "constant", weightDecay: 0.02 },
    focus: "Loss for all four schedules",
  },
  {
    id: "seed-table",
    sections: ["On the seed table"],
    title: "A wide band at peak 30",
    look: "At a peak of 30 the Constant band is far wider than the decaying schedules’ bands, so a single Constant run there is close to a coin flip.",
    step: 4,
    state: { seeds: "five", peakLearningRate: 30 },
    focus: "Loss for all four schedules",
  },
  {
    id: "depth-notice",
    sections: ["On the depth card"],
    title: "A gain of exactly 1",
    look: "At a Gain per layer of 1.00 every layer’s gradient has the same length, so the line lies flat, and moving the gain either way tilts it.",
    step: 5,
    state: { depthGain: 1 },
    focus: "Gradient through depth",
  },
];

export default scenes;
