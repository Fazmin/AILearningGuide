import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the
 * blocky 3 on the pad, vertical edge, Recurrent weight 0.90 at t=0, Convolution with 1 layer, and
 * 8 experts with 2 per token. `cell` counts pad squares row by row from 0, so 11 is r2 c4 and 20
 * is r3 c5; `rnnStep` t is column t + 1.
 */
const scenes: LabScene[] = [
  {
    id: "convolution",
    sections: ["Convolution: one small window, slid everywhere"],
    title: "One kernel, every position",
    look: "The formula row adds up the inked pixels under the kernel’s weights at r2 c4, so move Selected cell to r8 c4 and watch the same weights give the same total.",
    step: 1,
    state: { filter: "horiz", cell: 11 },
    focus: "Live filter",
  },
  {
    id: "early-steps-fade",
    sections: ["Why early steps fade", "Recurrence"],
    title: "Why early steps fade",
    look: "The formula row multiplies column 1’s effect by every factor on the arrows, and its bar is the shortest in the ∂h7/∂x list.",
    step: 2,
    focus: "Unrolled RNN",
  },
  {
    id: "attention-hop",
    sections: ["Attention: one direct look back"],
    title: "One hop to every earlier position",
    look: "Every square on or below the diagonal reads 1, so each output reaches every earlier input in one hop, while the dots above it are the hidden future.",
    step: 3,
    state: { wiring: "attn" },
    focus: "Who sees whom",
  },
  {
    id: "state-space",
    sections: ["State-space model: a recurrent network built for speed"],
    title: "A recurrent grid, run as a scan",
    look: "The grid looks exactly like Recurrent, but the State-space row of the table shows it can also run as a scan.",
    step: 3,
    state: { wiring: "ssm" },
    focus: "Who sees whom",
  },
  {
    id: "many-experts",
    sections: ["Mixture of experts: many stored, few used"],
    title: "Many stored, few used",
    look: "Each row sends one column of the drawing to a couple of experts, and Total expert weights sits beside the much smaller Active per token.",
    step: 4,
    focus: "Sparse experts",
  },
  {
    id: "conv-reach",
    sections: ["The problem of distance", "What to notice"],
    title: "Seven layers to reach",
    look: "With Conv layers at 7 the band of numbers finally fills row 8, so output 8 can reach input 1, and lowering Conv layers narrows the band again.",
    step: 3,
    state: { wiring: "conv", convLayers: 7 },
    focus: "Who sees whom",
  },
  {
    id: "draw-digit",
    sections: ["1. Draw the digit"],
    title: "Draw the digit",
    look: "The outlined 3×3 window is all one filter output can see, so flip a pixel inside it and then one outside it and watch the number in Live filter.",
    step: 0,
    state: { cell: 20 },
    focus: "Drawn digit",
  },
  {
    id: "pick-filter",
    sections: ["2. Pick a filter", "Locality and sharing"],
    title: "Pick a filter",
    look: "Switch Filter between vertical edge, horizontal edge and center blob, and watch the grid of totals change while the drawing stays the same.",
    step: 1,
    focus: "Live filter",
  },
  {
    id: "walk-rnn",
    sections: ["3. Walk the RNN"],
    title: "Walk the RNN",
    look: "Column 8 needs no hops and has the tallest bar, so move RNN step down to t=0 and compare column 1’s bar.",
    step: 2,
    state: { rnnStep: 7 },
    focus: "Unrolled RNN",
  },
  {
    id: "compare-wirings",
    sections: ["4. Compare wirings", "Wiring and experts"],
    title: "Recurrent against attention",
    look: "Row 8, column 1 shows how many steps input 1 needs to reach output 8, so switch Wiring to Attention and read it again.",
    step: 3,
    state: { wiring: "rnn" },
    focus: "Who sees whom",
  },
  {
    id: "route-experts",
    sections: ["5. Route to experts"],
    title: "Route to experts",
    look: "Raise Experts and watch Total expert weights grow while Active per token stays the same.",
    step: 4,
    state: { expertsPerToken: 1 },
    focus: "Sparse experts",
  },
  {
    id: "big-weight",
    sections: ["Fading on the RNN card"],
    title: "A bigger weight fades faster",
    look: "With Recurrent weight at its top the states crowd near 1, so the factors on the arrows shrink and column 1’s gradient gets smaller still.",
    step: 2,
    state: { recurrentWeight: 2 },
    focus: "Unrolled RNN",
  },
];

export default scenes;
