import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults
 * (inputs 0.70 and 0.35, weights 1.2 and −0.8, bias −0.1, Sigmoid), so z is 0.460 unless a scene
 * moves a slider. Inputs 0.75 and 1.00 put the probe on the line at those weights.
 */
const scenes: LabScene[] = [
  {
    id: "weigh-and-add",
    sections: ["Job one: weigh and add", "What it is"],
    title: "Weigh and add",
    look: "Each line into the z circle shows its weight and the product it adds, and the Weighted sum row adds them up with the bias.",
    step: 0,
    focus: "Neuron equation",
  },
  {
    id: "three-curves",
    sections: ["Job two: the activation", "What the activation adds"],
    title: "Three curves, one total",
    look: "Switch Activation between Sigmoid, ReLU and Step, and watch the dot jump to a new height while z stays the same.",
    step: 2,
    focus: "Activation curve",
  },
  {
    id: "on-the-line",
    sections: ["The line where the neuron is undecided"],
    title: "A point on the line",
    look: "The probe sits on the solid dark line between the green and red sides, and the z badge reads zero.",
    step: 1,
    state: { x1: 0.75, x2: 1 },
    focus: "Decision boundary and controls",
  },
  {
    id: "arrow-distance",
    sections: ["The arrow and the distance", "The geometry of z = w · x + b"],
    title: "The arrow and the distance",
    look: "The arrow w crosses the line at a right angle, the short line from the probe is its shortest path to it, and ‖w‖ and Distance to boundary sit under the square.",
    step: 1,
    focus: "Decision boundary and controls",
  },
  {
    id: "no-bias",
    sections: ["Why the bias is there"],
    title: "Bias set to zero",
    look: "With Bias b at zero, the line runs through the bottom-left corner, where the probe sits right on it.",
    step: 1,
    state: { bias: 0, x1: 0, x2: 0 },
    focus: "Decision boundary and controls",
  },
  {
    id: "move-inputs",
    sections: ["1. Move the inputs", "How to play with it"],
    title: "Move the inputs",
    look: "Drag Input x₁ and Input x₂ and watch the probe and the z badge move while the line stays still.",
    step: 0,
    focus: "Decision boundary and controls",
  },
  {
    id: "turn-and-slide",
    sections: ["2. Turn and slide the line"],
    title: "Turn and slide the line",
    look: "Drag Weight w₁ slowly down from 2 and the line turns with the arrow w, then move Bias b and the line slides instead.",
    step: 1,
    state: { w1: 2 },
    focus: "Decision boundary and controls",
  },
  {
    id: "one-total",
    sections: ["3. One total, three answers"],
    title: "One total, three answers",
    look: "Switch Activation between Sigmoid, ReLU and Step, and watch the dot, which now sits in the middle of the plot, change height.",
    step: 2,
    state: { x1: 0.75, x2: 1 },
    focus: "Activation curve",
  },
  {
    id: "double-everything",
    sections: ["4. Double everything", "What to notice"],
    title: "Double everything",
    look: "Read z and Distance to boundary, then double Weight w₁, Weight w₂ and Bias b, and watch the line stay put while the dashed lines move in.",
    step: 3,
    state: { w1: 1, w2: -0.5, bias: -0.2 },
    focus: "Decision boundary and controls",
  },
  {
    id: "corner-puzzle",
    sections: ["5. The corner puzzle"],
    title: "The AND corners",
    look: "Each corner mark shows ✓ or ✗ and Corners correct counts them, so move the sliders until every mark shows ✓.",
    step: 4,
    state: { task: "and" },
    focus: "Decision boundary and controls",
  },
  {
    id: "flat-zone",
    sections: ["The flat zone"],
    title: "The flat zone",
    look: "The dot sits far out in the shaded band on the right, where the curve is almost flat and Slope f′(z) turns red.",
    step: 2,
    state: { x1: 1, x2: 1, w1: 2, w2: 2, bias: 2 },
    focus: "Activation curve",
  },
  {
    id: "and-solved",
    sections: ["Solving AND and OR"],
    title: "AND solved",
    look: "All four corner marks show ✓, so switch Corner task to OR and raise Bias b until they all show ✓ again.",
    step: 4,
    state: { task: "and", w1: 1, w2: 1, bias: -1.5 },
    focus: "Decision boundary and controls",
  },
  {
    id: "xor-limit",
    sections: ["One straight line is all it gets"],
    title: "XOR, one line short",
    look: "The filled circles sit on one diagonal and the open squares on the other, so no slider setting turns every mark to ✓.",
    step: 4,
    state: { task: "xor" },
    focus: "Decision boundary and controls",
  },
];

export default scenes;
