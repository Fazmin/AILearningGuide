import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the
 * start at w −1.00 and b 1.20, Learning rate 0.10, Plain descent, at step 0. The `step` key inside
 * `state` is the lab's own Step slider (0 to 80), not the shell step.
 */
const scenes: LabScene[] = [
  {
    id: "residuals",
    sections: ["Loss: one number for how wrong the line is", "The lab's loss is real"],
    title: "Misses, squared and averaged",
    look: "Each orange segment is one dot’s miss, each bar below is that miss squared, and the dashed line across the bars is their average, Loss L.",
    step: 0,
    focus: "Data and fit",
  },
  {
    id: "one-step",
    sections: ["One step, worked out", "What it is"],
    title: "One step, worked out",
    look: "The ball has moved one step up and to the right of the start and Loss L has dropped, so press Back to step 0 to compare.",
    step: 2,
    state: { step: 1 },
    focus: "Loss surface",
  },
  {
    id: "steep-axis",
    sections: ["Curvature: how sharply the valley bends", "Curvature sets the speed limit"],
    title: "The speed limit across the valley",
    look: "Steep axis × shows how much of the gap across the valley each step leaves, so raise Learning rate and watch it turn negative and then pass −1.",
    step: 2,
    state: { learningRate: 0.2 },
    focus: "Loss per step",
  },
  {
    id: "slow-floor",
    sections: ["The slow direction along the floor"],
    title: "The slow valley floor",
    look: "At this rate Steep axis × and Shallow axis × are about the same size, the best this valley allows, yet the run still needs many steps to settle.",
    step: 2,
    state: { learningRate: 0.43 },
    focus: "Loss per step",
  },
  {
    id: "broken-words",
    sections: ["Where the words for broken training come from", "Why it’s here"],
    title: "The loss exploded",
    look: "The loss curve climbs faster and faster until it runs off the top of the chart, where a red note marks the step it passed 100.",
    step: 2,
    state: { learningRate: 0.48, step: 30 },
    focus: "Loss per step",
  },
  {
    id: "measure-by-hand",
    sections: ["1. Measure the error by hand", "How to play with it"],
    title: "Measure the error by hand",
    look: "Drag Start w and Start b and watch the purple line, the orange misses and the bars change, with Loss L always the bars’ average.",
    step: 0,
    focus: "Data and fit",
  },
  {
    id: "predict-step",
    sections: ["2. Predict the first step"],
    title: "Predict the first step",
    look: "Read the signs of ∂L/∂w and ∂L/∂b under the map, then check that the pink −∇L arrow crosses the nearest contour at a right angle and misses the green ×.",
    step: 1,
    focus: "Loss surface",
  },
  {
    id: "step-size",
    sections: ["3. Take steps and change the step size"],
    title: "Change the step size",
    look: "Press Take one gradient step and watch the orange ball follow the dashed path, then change Learning rate and watch the path and the badge change.",
    step: 2,
    focus: "Loss surface",
  },
  {
    id: "compare-momentum",
    sections: ["4. Compare momentum", "What momentum adds"],
    title: "Compare momentum",
    look: "Move the Momentum β slider and read Settles by each time, then switch Optimizer back to Plain descent to compare.",
    step: 3,
    state: { learningRate: 0.05, optimizer: "momentum" },
    focus: "Loss per step",
  },
  {
    id: "drop-then-crawl",
    sections: ["A fast drop, then a long crawl", "What to notice"],
    title: "A fast drop, then a crawl",
    look: "The loss curve drops fast for a few steps, then bends into a long, gentle slope as the run creeps along the valley floor.",
    step: 2,
    state: { learningRate: 0.05, step: 80 },
    focus: "Loss per step",
  },
  {
    id: "zigzag",
    sections: ["Zigzagging is not the same as blowing up"],
    title: "A zigzag that still settles",
    look: "The path bounces from wall to wall across the valley while the loss keeps falling, so press Take one gradient step to follow it down.",
    step: 2,
    state: { learningRate: 0.4 },
    focus: "Loss surface",
  },
  {
    id: "too-much-momentum",
    sections: ["Momentum helps, up to a point"],
    title: "Too much momentum",
    look: "The loss rises and falls in waves instead of settling, so lower Momentum β and watch the waves calm down.",
    step: 3,
    state: { learningRate: 0.05, optimizer: "momentum", momentum: 0.95, step: 80 },
    focus: "Loss per step",
  },
];

export default scenes;
