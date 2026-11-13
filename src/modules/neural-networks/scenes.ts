import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its headings. The five building beats open on the lab's defaults,
 * because each starts from its card as it first appears; the playground and limits scenes load the
 * settings their paragraphs describe.
 */
const scenes: LabScene[] = [
  {
    id: "two-knobs",
    sections: ["Two knobs"],
    title: "A machine with two knobs",
    look: "Turn Knob w and Knob b and watch the line and the Loss.",
    step: 0,
    focus: "A machine with two knobs",
  },
  {
    id: "seven-knobs",
    sections: ["Seven knobs"],
    title: "Seven knobs, scrambled",
    look: "Turn any knob in the diagram and watch the thin line of its neuron and the thick curve.",
    step: 1,
    focus: "Seven knobs by hand",
  },
  {
    id: "turn-hints",
    sections: ["Let the machine turn them"],
    title: "A hint on every knob",
    look: "Press Take one step and watch every knob move along its pink arrow while the Loss falls.",
    step: 2,
    focus: "Let the machine turn the knobs",
  },
  {
    id: "more-neurons",
    sections: ["More neurons, more shapes"],
    title: "One hidden neuron",
    look: "Drag Hidden neurons up from 1 and compare the curve and the bars at each size.",
    step: 3,
    focus: "More neurons, more shapes",
  },
  {
    id: "digit-reader",
    sections: ["Billions of knobs"],
    title: "A small digit reader",
    look: "Read the Knobs line, then raise Neurons per layer and watch your bar move toward the published networks.",
    step: 4,
    state: { inputs: 784, layers: 2, width: 16, outputs: 10 },
    focus: "Count the knobs",
  },
  {
    id: "playground-first-try",
    sections: ["A first try"],
    title: "Wave, two neurons",
    look: "Press Train and watch the curve settle while the Loss stays far above the goal.",
    step: 5,
    focus: "Playground: train a network",
  },
  {
    id: "playground-more-knobs",
    sections: ["Give it more knobs"],
    title: "Wave, four neurons",
    look: "Drag Training steps back to 0 and forward again to replay the curve finding the wave.",
    step: 5,
    state: { pgWidth: 4, pgSteps: 600 },
    focus: "Playground: train a network",
  },
  {
    id: "stuck-start",
    sections: ["When training gets stuck"],
    title: "Four neurons, stuck",
    look: "Compare the bar for 4 neurons with the bar for 3, then press Scramble the knobs.",
    step: 3,
    state: { units: 4, unitsSeed: 3 },
    focus: "More neurons, more shapes",
  },
  {
    id: "steps-too-long",
    sections: ["When the steps are too long"],
    title: "Step size 0.5, no momentum",
    look: "Drag Training steps from 0 to 600 and watch the Loss climb instead of fall.",
    step: 5,
    state: { pgWidth: 4, pgRate: 0.5, pgMomentum: "off", pgSteps: 600 },
    focus: "Playground: train a network",
  },
];

export default scenes;
