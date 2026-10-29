import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: XOR,
 * one hidden layer of two tanh units, seed 2 at epoch 150, the probe at (0.6, −0.6), and 20% flipped
 * labels on Held-out data. Training is deterministic for a seed, so the scores the lesson quotes
 * appear as soon as a scene loads.
 */
const scenes: LabScene[] = [
  {
    id: "new-description",
    sections: ["Representation: a new description of each dot", "How two units solve XOR"],
    title: "A new description of each dot",
    look: "The probe starts on the bottom-right “yes” corner, and the numbers under the h1 and h2 tiles are that dot’s new description.",
    step: 3,
    focus: "Network builder",
  },
  {
    id: "output-sum",
    sections: ["Reading the output neuron's sum"],
    title: "The output neuron’s sum",
    look: "The Output unit at the probe row multiplies h1 and h2 by the output neuron’s weights and adds its bias, so move Probe x₂ up to the top-right corner and watch ŷ drop.",
    step: 3,
    focus: "Network builder",
  },
  {
    id: "random-start",
    sections: ["What training does here"],
    title: "Training from a random start",
    look: "This is the random start, so press Train 50 epochs and watch the marker slide down the loss curve while Accuracy climbs.",
    step: 2,
    state: { epochs: 0 },
    focus: "Network training controls",
  },
  {
    id: "one-neuron-stack",
    sections: ["The puzzle one neuron could not solve"],
    title: "One neuron per layer",
    look: "Three hidden layers of one neuron each still draw one dashed line, so some dots stay ringed as wrong.",
    step: 0,
    state: { width: 1, depth: 3 },
    focus: "Decision surface",
  },
  {
    id: "stuck-seed",
    sections: ["Nobody tells the hidden neurons what to look for"],
    title: "A start that gets stuck",
    look: "This random start, trained all the way, never gets every dot right, and the note under the loss chart gives the best score it reached.",
    step: 2,
    state: { seed: 3, epochs: 300 },
    focus: "Network training controls",
  },
  {
    id: "solve-xor",
    sections: ["Solve XOR", "How to play with it"],
    title: "Solve XOR",
    look: "With one neuron the badge stalls short of every dot, so set Neurons per layer to 2 and watch the dashed lines h1 and h2 split the square.",
    step: 0,
    state: { width: 1 },
    focus: "Decision surface",
  },
  {
    id: "look-inside",
    sections: ["Look inside"],
    title: "Look inside",
    look: "The probe sits on the top-right “no” corner, so read h1 and h2 under the tiles, then move Probe x₁ and Probe x₂ to the other corners.",
    step: 3,
    state: { probeX: 0.6, probeY: 0.6 },
    focus: "Network builder",
  },
  {
    id: "no-bend",
    sections: ["Remove the bend", "Why the activation is load-bearing"],
    title: "No bend between layers",
    look: "Change Hidden layers and Neurons per layer as much as you like, and the loss curve still settles onto the dotted always-50% line.",
    step: 4,
    state: { activation: "linear" },
    focus: "Network training controls",
  },
  {
    id: "ring-width",
    sections: ["Need more width"],
    title: "Two lines around a ring",
    look: "Two dashed lines leave the green “yes” area open on one side, so raise Neurons per layer to 3 and watch a third line close it around the middle disc.",
    step: 5,
    state: { dataset: "ring" },
    focus: "Decision surface",
  },
  {
    id: "match-size",
    sections: ["Match the size", "Depth against width at the same size"],
    title: "Same size, two shapes",
    look: "Compare the two networks’ Accuracy readouts and surfaces, then switch Hidden activation to ReLU and watch the badge at the top.",
    step: 6,
    state: { dataset: "checker", epochs: 300 },
    focus: "Depth against width",
  },
  {
    id: "held-out",
    sections: ["Hold data out", "Held-out data and label noise"],
    title: "Flipped labels, fresh dots",
    look: "Switch Training labels between Clean and 20% flipped and watch the large network’s Gap readout grow and shrink.",
    step: 7,
    state: { epochs: 300 },
    focus: "Held-out data",
  },
  {
    id: "tiles-bend",
    sections: ["What to notice"],
    title: "Second-layer tiles bend",
    look: "Each tile in the hidden 1 column is split by one straight edge, while the tiles in the hidden 2 column bend.",
    step: 1,
    state: { depth: 2 },
    focus: "Network builder",
  },
];

export default scenes;
