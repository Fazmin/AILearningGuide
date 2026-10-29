import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults
 * (Generation 5, Synthetic only, Transitions, the model itself, temperature 1.0, no filter,
 * 500 characters), so the chart shows the lesson's own numbers unless a scene changes a control.
 * `regime` is "replace" for Synthetic only and "accumulate" for Real + synthetic; `measure`
 * "heldOut" is Unseen text.
 */
const scenes: LabScene[] = [
  {
    id: "table-of-pairs",
    sections: ["The model: a table of letter pairs", "What it is"],
    title: "The seed’s letter pairs",
    look: "At generation 0 every pair the seed contains is a filled box in the grid, darker for pairs the seed used more, and hovering a box shows its count.",
    step: 0,
    state: { generation: 0 },
    focus: "Which transitions survive",
  },
  {
    id: "how-it-writes",
    sections: ["How the model writes"],
    title: "Real pairs, made-up words",
    look: "The sample glues real letter pairs into made-up words, which is why Real words here stays so low.",
    step: 0,
    state: { generation: 0 },
    focus: "What a generation writes",
  },
  {
    id: "teacher-writes",
    sections: ["Who does the writing"],
    title: "A stronger teacher",
    look: "With Generator on A stronger teacher, a note under the controls says what the teacher learned from and that it never changes.",
    step: 4,
    state: { generator: "teacher" },
    focus: "Generator and filter",
  },
  {
    id: "read-one-generation",
    sections: ["Step 1: read one generation"],
    title: "Generation 1’s writing",
    look: "Generation 1 learned only from the first model’s writing, so Trained on now counts synthetic characters, then set Generation to 0 to compare the samples.",
    step: 0,
    state: { generation: 1 },
    focus: "What a generation writes",
  },
  {
    id: "synthetic-only",
    sections: ["Step 2: let the loop run on synthetic text only", "How to play with it"],
    title: "Ten rounds on synthetic text",
    look: "Drag Generation from 0 to 10 and watch the solid Synthetic only line fall while the dashed Real + synthetic line stays flat.",
    step: 1,
    state: { generation: 0 },
    focus: "Recursive training",
  },
  {
    id: "coverage-entropy",
    sections: ["Two more views of the same loss"],
    title: "Coverage at generation 10",
    look: "Measure is on Coverage at generation 10, so compare how far it has fallen with the Transitions line, then switch Measure to Entropy.",
    step: 1,
    state: { measure: "coverage", generation: 10 },
    focus: "Recursive training",
  },
  {
    id: "keep-real",
    sections: ["Step 3: keep the real data"],
    title: "Keep the human notes",
    look: "With Training data on Real + synthetic, the dashed line stays flat at the top through generation 10, then switch Measure to Entropy to see what still drifts.",
    step: 2,
    state: { regime: "accumulate", generation: 10 },
    focus: "Recursive training",
  },
  {
    id: "filter-sharpen",
    sections: ["Step 4: filter and sharpen"],
    title: "The real-words filter",
    look: "With Filter on Real words only, the words the verifier would reject are struck through in the sample, and Filter keeps shows how much survives.",
    step: 3,
    state: { filter: "words", measure: "realWords" },
    focus: "What a generation writes",
  },
  {
    id: "stronger-teacher",
    sections: ["Step 5: bring in a stronger teacher"],
    title: "Unseen text at generation 10",
    look: "Measure is on Unseen text, so compare the seed’s score with generation 10’s, then switch Generator to A stronger teacher and Training data to Real + synthetic.",
    step: 4,
    state: { measure: "heldOut", generation: 10 },
    focus: "Recursive training",
  },
  {
    id: "tail-first",
    sections: ["What to notice"],
    title: "The rare pairs go first",
    look: "At generation 10 the three bars under the grid show the seen once pairs lost far more often than the seen 5+ times ones.",
    step: 1,
    state: { generation: 10 },
    focus: "Which transitions survive",
  },
];

export default scenes;
