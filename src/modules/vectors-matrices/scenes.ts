import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections: one per card, in the lesson's order. Every scene
 * but one uses the lab's defaults, which are the numbers the lesson works by hand (horse, the
 * step "bigger", horse beside whale, the shear, rotate then stretch, two pieces). "The map pressed
 * flat" sets the squeeze to zero, the state its section builds up to.
 */
const scenes: LabScene[] = [
  {
    id: "horse-as-arrow",
    sections: ["A word as two numbers"],
    title: "Horse as an arrow",
    look: "The green arrow runs from the centre to horse, and its blue and purple legs are the two numbers in the list.",
    step: 0,
    focus: "A word as two numbers",
  },
  {
    id: "add-a-step",
    sections: ["Add and stretch"],
    title: "Horse plus a step",
    look: "Slide Times and watch the magenta step stretch, shrink and flip while the green result arrow follows it.",
    step: 1,
    focus: "Add and stretch",
  },
  {
    id: "two-arrows",
    sections: ["How alike are two arrows?"],
    title: "Horse beside whale",
    look: "The three bars multiply the matching numbers and add them, and the round marker beside the map shows the cosine.",
    step: 2,
    focus: "How alike are two arrows",
  },
  {
    id: "leaning-map",
    sections: ["A matrix moves every arrow at once"],
    title: "A matrix that leans the map",
    look: "The thick blue and purple arrows end where the First column and Second column sliders say.",
    step: 3,
    focus: "A matrix moves every arrow",
  },
  {
    id: "two-in-a-row",
    sections: ["One after another"],
    title: "Rotate, then stretch",
    look: "Press Swap the order and watch every word jump to its hollow mark.",
    step: 4,
    focus: "One after another",
  },
  {
    id: "pressed-flat",
    sections: ["When a matrix forgets"],
    title: "The map pressed flat",
    look: "Every word lies on one line, and a red ring marks the two words that now share a spot.",
    step: 5,
    state: { keep: 0 },
    focus: "When a matrix forgets",
  },
  {
    id: "two-pieces",
    sections: ["A table from a few pieces"],
    title: "Two pieces of four",
    look: "Move Pieces kept and compare the Numbers stored tile with the Average miss per cell tile.",
    step: 6,
    focus: "A table from a few pieces",
  },
  {
    id: "aim-round-one",
    sections: ["Your turn: aim the map"],
    title: "Round one",
    look: "Set the four sliders until every word sits inside its dashed mark and the meter turns green.",
    step: 7,
    focus: "Challenge: aim the map",
  },
];

export default scenes;
