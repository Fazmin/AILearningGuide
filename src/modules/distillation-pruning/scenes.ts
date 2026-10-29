import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (rank 6, T = 1, 18 epochs,
 * soft labels, 30% unstructured, context “h”, and a 50% magnitude prune at 4 bits on the compose card), so the readouts
 * show the lesson's own numbers unless a scene moves a control.
 */
const scenes: LabScene[] = [
  {
    id: "whole-teacher",
    sections: ["The teacher, a model small enough to see whole"],
    title: "The whole teacher",
    look: "With nothing pruned, the grid on the Pruning card is the teacher’s whole table, and the more solid a cell looks, the bigger its weight.",
    state: { pruneFraction: 0 },
    focus: "Pruning",
  },
  {
    id: "student-size",
    sections: ["How the student is built", "What it is"],
    title: "How big the student is",
    look: "Drag Student size and watch the weight count beside it, and the badge on the Distillation card, rise and fall with the rank.",
    focus: "Student controls",
  },
  {
    id: "teacher-guesses",
    sections: ["What the teacher knows that the answer key does not", "The distillation objective"],
    title: "The teacher’s guesses after “h”",
    look: "In each row, the top bar is how often that character really follows “h”, and the middle bar is the teacher’s guess.",
    step: 0,
    focus: "Soft targets",
  },
  {
    id: "softer-teacher",
    sections: ["What temperature does", "Temperature and dark knowledge"],
    title: "A softer teacher",
    look: "The teacher’s middle bars flatten, and the Same, at T = 2.0 readout climbs far above the T = 1 one beside it.",
    step: 1,
    state: { temperature: 2 },
    focus: "Soft targets",
  },
  {
    id: "copy-teacher",
    sections: ["Copy the teacher", "How to play with it"],
    title: "Copy the teacher",
    look: "Flip Training target and watch KL to teacher, then compare the two KL bars and the two loss curves on the Distillation card.",
    step: 0,
    focus: "Student controls",
  },
  {
    id: "temperature-sweep",
    sections: ["Tune the temperature"],
    title: "Sweep the temperature",
    look: "Move Distillation temperature up one notch at a time and read KL to teacher, which falls, reaches its lowest point, then climbs again.",
    step: 1,
    state: { temperature: 0.5 },
    focus: "Student controls",
  },
  {
    id: "prune-from-zero",
    sections: ["Prune by magnitude"],
    title: "Prune from zero",
    look: "Raise Pruned share one notch at a time and watch the perplexity badge on the Pruning card, which hardly moves at first and then climbs.",
    step: 2,
    state: { pruneFraction: 0 },
    focus: "Pruning controls",
  },
  {
    id: "two-in-four",
    sections: ["Three ways to prune", "Three shapes of pruning"],
    title: "Two of every four",
    look: "The grid shows 2:4 mode, with two blanks in every four cells down each column, and Pruning mode switches it to scattered blanks or empty rows.",
    step: 3,
    state: { pruneMode: "semi" },
    focus: "Pruning",
  },
  {
    id: "compare-shapes",
    sections: ["Compare the three shapes"],
    title: "Same share, three shapes",
    look: "Switch Pruning mode and watch the grid, the perplexity badge and Rows removed on the Pruning card change with each pattern.",
    step: 3,
    state: { pruneFraction: 0.5 },
    focus: "Pruning",
  },
  {
    id: "weigh-each-stage",
    sections: ["Running the three steps together"],
    title: "Weighing each stage",
    look: "Read the bytes on each row of the first bar list, then the note at the bottom of the card, which counts the mask bits and the groups of 32.",
    step: 4,
    focus: "Compose the three steps",
  },
  {
    id: "compose-stages",
    sections: ["Compose the stages"],
    title: "Distil, prune, quantize",
    look: "Read the first bar list from Teacher down to Quantized: each bar is shorter than the one above, and its perplexity a little higher.",
    step: 4,
    focus: "Compose the three steps",
  },
  {
    id: "importance-scores",
    sections: ["Change the importance score"],
    title: "Three ways to choose",
    look: "Switch Importance score and compare the second bar list, which shows all three scores with the passes each one needs.",
    step: 5,
    state: { composeShare: 0.7 },
    focus: "Compose the three steps",
  },
];

export default scenes;
