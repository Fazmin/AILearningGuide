import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `phase` indexes the trace, from 0 (Multiply) to 5
 * (Product → weight). Every scene starts from the lab's defaults, so the graph shows the lesson's
 * own numbers unless a scene sets the sliders.
 */
const scenes: LabScene[] = [
  {
    id: "walk-forward",
    sections: ["Walk the small graph one edge at a time", "How to play with it"],
    title: "The forward pass, one edge at a time",
    look: "Press Advance one edge and watch z, then ŷ, then the loss L fill in from left to right.",
    step: 0,
    state: { phase: 0 },
    focus: "Interactive backpropagation computation graph",
  },
  {
    id: "going-backward",
    sections: ["Going backward", "Why backward rather than forward"],
    title: "The backward pass",
    look: "Each pink × tag is one station’s factor, and the pills under the circles carry the running total back to the weight.",
    step: 2,
    state: { phase: 5 },
    focus: "Interactive backpropagation computation graph",
  },
  {
    id: "chain-rule",
    sections: ["The chain rule: multiply along the path", "The three local derivatives"],
    title: "The chain rule, with numbers",
    look: "The Chain rule row multiplies the three local factors into the weight’s slope, using the current slider values.",
    step: 2,
    state: { phase: 5 },
    focus: "Backpropagation controls and chain rule",
  },
  {
    id: "break-slope",
    sections: ["Break the slope on purpose"],
    title: "A saturated sigmoid",
    look: "The loss is much bigger, yet the slope on the weight is smaller: the sigmoid’s factor in the Chain rule row has collapsed.",
    step: 2,
    state: { x: 1.5, weight: 3, target: 0.1, phase: 5 },
    focus: "Backpropagation controls and chain rule",
  },
  {
    id: "spend-slope",
    sections: ["Spend the slope"],
    title: "One step downhill",
    look: "Press Apply w ← w − η∂L/∂w, then watch the ball slide downhill on the Loss versus weight tangent chart.",
    step: 3,
    state: { phase: 5 },
    focus: "Backpropagation controls and chain rule",
  },
  {
    id: "measure-slope",
    sections: ["Check the slope by measuring"],
    title: "Measure the slope",
    look: "Compare Analytic slope with Numeric check under the chart, then drag Weight w and watch them stay level.",
    step: 3,
    state: { phase: 5 },
    focus: "Loss versus weight tangent chart",
  },
  {
    id: "long-product",
    sections: ["Many layers: the product gets long", "The weak point: long products"],
    title: "Twelve sigmoid layers",
    look: "Compare the Layer 1 bar with the last layer’s: the slope shrinks by orders of magnitude on the way back.",
    step: 4,
    state: { flowDepth: 12, flowActivation: "sigmoid", flowInit: "glorot", flowSkip: "off" },
    focus: "Gradient flow through depth",
  },
  {
    id: "stack-layers",
    sections: ["Stack the layers"],
    title: "Stack the layers",
    look: "Raise Network depth one step at a time and watch Layer 1 norm fall while the last layer’s stays put.",
    step: 4,
    state: { flowDepth: 1, flowActivation: "sigmoid", flowInit: "glorot", flowSkip: "off" },
    focus: "Gradient flow through depth",
  },
  {
    id: "activation",
    sections: ["The first control: Activation"],
    title: "tanh instead of sigmoid",
    look: "With tanh at twelve layers, Layer 1’s bar now reaches about as far as the last layer’s.",
    step: 4,
    state: { flowDepth: 12, flowActivation: "tanh" },
    focus: "Gradient flow through depth",
  },
  {
    id: "init-scale",
    sections: ["The second control: Initial weight scale"],
    title: "Starting weights decide",
    look: "Step Initial weight scale from small to large and watch every bar move together.",
    step: 5,
    state: { flowDepth: 12, flowActivation: "relu", flowInit: "small" },
    focus: "Gradient flow through depth",
  },
  {
    id: "skip-on",
    sections: ["The third control: Skip connections"],
    title: "Skip connections on",
    look: "Compare Layer 1 ÷ last and Loss at the start with the twelve sigmoid layers you saw without the shortcut.",
    step: 5,
    state: { flowDepth: 12, flowSkip: "on" },
    focus: "Gradient flow through depth",
  },
];

export default scenes;
