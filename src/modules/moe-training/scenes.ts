import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: no
 * balancing, 4 experts, top-1, no capacity cap, 30 epochs, seed 1, and the grid by previous
 * character. A scene that needs balancing sets it, so the readouts show the lesson's own numbers.
 */
const scenes: LabScene[] = [
  {
    id: "router-gates",
    sections: ["Scores, gates and top-k"],
    title: "Scores, biases and gates",
    look: "The table lists each expert's score, bias and gate for the letter t, and the highlighted row is the expert that runs.",
    step: 2,
    state: { balancing: "bias", biasStep: 0.03 },
    focus: "Router and experts",
  },
  {
    id: "one-expert-wins",
    sections: ["Why one expert wins"],
    title: "Eight experts, one winner",
    look: "With 8 experts the lines start spread out, and within two epochs a single line holds every token.",
    step: 0,
    state: { experts: 8 },
    focus: "Expert load over training",
  },
  {
    id: "watch-collapse",
    sections: ["Watch it collapse"],
    title: "Watch it collapse",
    look: "One line climbs to 100%, Effective experts reads 1.00, and Loss sits right next to Dense baseline.",
    step: 0,
    focus: "Expert load over training",
  },
  {
    id: "balancing-loss",
    sections: ["Repair 1: a penalty for lopsided work"],
    title: "A balancing loss",
    look: "At Aux weight 0.1 every expert gets work, so step Aux weight down to 0.01 and up to 1 and compare Loss with Dense baseline.",
    step: 1,
    state: { balancing: "aux", auxWeight: 0.1 },
    focus: "Balancing",
  },
  {
    id: "bias-nudge",
    sections: ["Repair 2: a thumb on the scale"],
    title: "A bias on each expert",
    look: "Each bar is one expert's bias after training, so lower Bias step to 0.001 and watch most tokens go back to one expert.",
    step: 2,
    state: { balancing: "bias", biasStep: 0.03 },
    focus: "Balancing",
  },
  {
    id: "cap-experts",
    sections: ["Repair 3: a seat limit"],
    title: "Five seats each",
    look: "With no balancing many choices find their expert full, so switch Balancing to Auxiliary loss and watch Dropped tokens fall.",
    step: 3,
    state: { capacity: 1.25, auxWeight: 0.1 },
    focus: "Capacity and dropped tokens",
  },
  {
    id: "read-specialists",
    sections: ["Who gets which letters"],
    title: "Who gets which letters",
    look: "Each row shows where one character's tokens go, so switch Grid to By corpus and see every corpus spread its tokens the same way.",
    step: 4,
    state: { balancing: "aux", auxWeight: 0.1 },
    focus: "Specialisation",
  },
  {
    id: "count-what-runs",
    sections: ["Stored against used"],
    title: "Stored against used",
    look: "Raise Experts and watch Total parameters climb while Active per token barely moves, then read the Mixtral 8x7B row.",
    step: 5,
    state: { experts: 1, balancing: "aux", auxWeight: 0.1 },
    focus: "Total vs active parameters",
  },
  {
    id: "too-strong",
    sections: ["Too much of a good thing"],
    title: "Balanced, but worse",
    look: "At Aux weight 1 every expert stays busy, but Loss sits above Dense baseline and Mean gate drops.",
    step: 1,
    state: { balancing: "aux", auxWeight: 1 },
    focus: "Balancing",
  },
  {
    id: "corpus-blind",
    sections: ["What the router could see"],
    title: "A router that cannot see the text",
    look: "The three corpus rows look almost the same, because the router never learns which text a pair came from.",
    step: 4,
    state: { balancing: "aux", auxWeight: 0.1, specialView: "corpus" },
    focus: "Specialisation",
  },
];

export default scenes;
