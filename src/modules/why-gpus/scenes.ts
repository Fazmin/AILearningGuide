import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the
 * fp16 matmul at 1 batch row, layer width 4096 and 2000 GB/s. Batch rows and Layer width are
 * powers of two, which the lab's hydrateState keeps as they are.
 */
const scenes: LabScene[] = [
  {
    id: "count-work",
    sections: ["Counting the work", "What it is"],
    title: "Counting the work",
    look: "The Work versus traffic row counts the operations and the bytes for this shape, and both change when you move Batch rows or Layer width.",
    step: 0,
    focus: "Workload knobs",
  },
  {
    id: "two-clocks",
    sections: ["Two speeds, not one"],
    title: "Two clocks",
    look: "The Arithmetic at peak bar and the Memory traffic bar are two clocks, and the longer one sets the wall-clock time.",
    step: 0,
    focus: "Where the time goes",
  },
  {
    id: "ridge-point",
    sections: ["The ridge point: where the two limits meet"],
    title: "The ridge point",
    look: "The Ridge readout is the chip’s peak divided by the memory speed, and the dot sits far to the left of the dashed ridge line, so Bound reads memory.",
    step: 0,
    focus: "Roofline",
  },
  {
    id: "roofline-graph",
    sections: ["The roofline graph", "The roofline"],
    title: "Reading the roofline",
    look: "The slanted part of the roof is the memory limit and the flat part is the chip’s peak, and Attainable reads the roof’s height above the dot.",
    step: 0,
    focus: "Roofline",
  },
  {
    id: "batch-lever",
    sections: ["Why batch is the lever"],
    title: "Rows reuse the weights",
    look: "At this batch the Intensity readout sits just under the 2·B / bytes shortcut in the note under Work versus traffic, because each weight is fetched once for every row.",
    step: 0,
    state: { batch: 64 },
    focus: "Roofline",
  },
  {
    id: "parallel-not-enough",
    sections: ["Why it’s here"],
    title: "Plenty of work, idle chip",
    look: "Parallel outputs shows thousands of independent dot products, yet the line at the top of this card shows only a sliver of peak arithmetic in use.",
    step: 0,
    focus: "Where the time goes",
  },
  {
    id: "grow-batch",
    sections: ["Grow the batch", "How to play with it"],
    title: "Grow the batch",
    look: "Drag Batch rows up one step at a time and watch the dot climb the slanted roof, cross the dashed ridge and run along the flat top.",
    step: 0,
    focus: "Roofline",
  },
  {
    id: "change-precision",
    sections: ["Change precision"],
    title: "Fewer bytes per number",
    look: "Switch Precision between fp16, fp32 and int8 and watch the Memory traffic bar grow or shrink with the bytes each number takes.",
    step: 1,
    focus: "Where the time goes",
  },
  {
    id: "starve-bandwidth",
    sections: ["Starve the bandwidth"],
    title: "Slow memory, moved ridge",
    look: "The dashed ridge now sits to the right of the dot and Bound reads memory, so drag Memory bandwidth back up and watch the ridge slide back past it.",
    step: 2,
    state: { batch: 1024, bandwidth: 100 },
    focus: "Roofline",
  },
  {
    id: "elementwise-add",
    sections: ["Try an elementwise op"],
    title: "An add never catches up",
    look: "The add’s dot stays far left of the ridge at every batch, while the diamond marks the matmul of the same shape far to its right.",
    step: 3,
    state: { operation: "add", batch: 4096 },
    focus: "Roofline",
  },
  {
    id: "table-flips",
    sections: ["What to notice"],
    title: "Which speed-up helps",
    look: "Now that Bound reads compute, the 2× peak arithmetic row helps and the 2× memory bandwidth row reads no change.",
    step: 0,
    state: { batch: 1024 },
    focus: "Where the time goes",
  },
];

export default scenes;
