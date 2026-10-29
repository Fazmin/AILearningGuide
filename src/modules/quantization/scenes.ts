import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: 4 bits,
 * asymmetric grids per group of 8, Inspect block on the block that holds the largest weight, the file
 * grouped by block type and 0.02 in Float formats. The lesson's worked examples use one grid for the
 * whole table, so those scenes set `scope: "tensor"`.
 */
const scenes: LabScene[] = [
  {
    id: "round-weights",
    sections: ["Round the weights", "How to play with it"],
    title: "From 8 bits down to 2",
    look: "One grid covers the whole table at 8 bits; pull Bit width down to 2 and watch the dots in the lower strip form a sawtooth whose teeth widen.",
    step: 0,
    state: { scope: "tensor", bits: 8 },
    focus: "Rounding grid and per-weight error",
  },
  {
    id: "worked-example",
    sections: ["A worked example from the lab"],
    title: "One grid at 4 bits",
    look: "With one grid Per tensor at 4 bits, Step in this block is the gap between ticks, and Largest error stays just under half of it.",
    step: 0,
    state: { scope: "tensor" },
    focus: "Rounding grid and per-weight error",
  },
  {
    id: "measuring-damage",
    sections: ["Measuring the damage"],
    title: "What the rounding costs",
    look: "Full precision and Quantized give the perplexity before and after rounding, and Quality cost says how much higher the rounded one is.",
    state: { scope: "tensor" },
    focus: "Quality against bits per weight",
  },
  {
    id: "symmetric-ruler",
    sections: ["Two ways to lay the ruler", "Two ways to place the grid"],
    title: "A grid centred on zero",
    look: "Range mapping is on Symmetric · absmax, so the Bit width label counts one level fewer and the note says each block stores only one fp16 scale.",
    step: 2,
    state: { scope: "tensor", zeroPoint: "symmetric" },
    focus: "Quantization controls",
  },
  {
    id: "per-row",
    sections: ["Change the granularity", "Where the blocks come from"],
    title: "One grid per row",
    look: "Per row starts Inspect block on the full stop’s row, where one big score stretches Step in this block; drag Inspect block to compare other rows.",
    step: 1,
    state: { scope: "row" },
    focus: "Rounding grid and per-weight error",
  },
  {
    id: "zero-point",
    sections: ["Move the zero point"],
    title: "Wasted ticks below zero",
    look: "The grid is now centred on zero, so the dashed lines below the smallest weight are ticks no weight lands on; read Levels used in this block.",
    step: 2,
    state: { scope: "tensor", zeroPoint: "symmetric" },
    focus: "Rounding grid and per-weight error",
  },
  {
    id: "price-tradeoff",
    sections: ["Price the tradeoff"],
    title: "Storage against damage",
    look: "Find the 4-bit point on each of the three lines and the Q4_K diamond, then compare how far right and how far down each one sits.",
    step: 3,
    focus: "Quality against bits per weight",
  },
  {
    id: "format-names",
    sections: ["Read the format names"],
    title: "Three 4-bit block types",
    look: "Compare the highlighted Q4_0, Q4_1 and Q4_K rows, which share 4-bit codes, across Bits per weight, Toy perplexity and RMS error.",
    step: 4,
    focus: "Published quantization formats",
  },
  {
    id: "file-by-role",
    sections: ["Weigh the file you downloaded"],
    title: "The file, part by part",
    look: "With Group tensors by on Role in the network, each bar is one part of the model with its own bits per weight, and the costly parts show a warning colour.",
    step: 5,
    state: { fileGroup: "role" },
    focus: "The file you downloaded",
  },
  {
    id: "float-overflow",
    sections: ["Climb the float ladder"],
    title: "Too big for fp16",
    look: "Example value is 100,000, so the fp16 and fp8 rows read overflow; then set it to 0.02 and compare the Error column.",
    step: 6,
    state: { floatValue: 100000 },
    focus: "Float formats",
  },
  {
    id: "float-layout",
    sections: ["How computers write ordinary numbers"],
    title: "Sign, exponent, mantissa",
    look: "Each row splits its bits into sign, exponent and mantissa, and at 0.001 the fp8 E4M3 row is marked subnormal, where its steps stop shrinking.",
    step: 6,
    state: { floatValue: 0.001 },
    focus: "Float formats",
  },
  {
    id: "group-of-two",
    sections: ["What to notice"],
    title: "Groups of two",
    look: "With Group size at 2, Bits per weight costs more than a plain 16-bit copy, even though every code is still 4 bits.",
    step: 1,
    state: { groupSize: 2 },
    focus: "Quantization controls",
  },
];

export default scenes;
