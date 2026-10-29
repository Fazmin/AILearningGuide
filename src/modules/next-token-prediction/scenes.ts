import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the
 * unseen line "O Juliet, Juliet! wherefore art thou J" with Model on RNN and Temperature at 0.80,
 * so the bars show the lesson's own numbers unless a scene changes them. The Layer by layer card
 * always reads the transformer, whatever Model is set to.
 */
const scenes: LabScene[] = [
  {
    id: "two-bars",
    sections: ["Two bars for every character"],
    title: "Two bars per character",
    look: "In each row of Next token distribution, the outlined bar is the model’s own odds and the filled bar is the sampler’s; drag Temperature and only the filled bars move.",
    step: 2,
    focus: "Next token distribution",
  },
  {
    id: "spinner-strip",
    sections: ["Picking one character"],
    title: "The spinner, unrolled",
    look: "Under the bars, the Inverse-CDF draw strip lays the characters end to end, and its marker shows where u lands; move Seed and it lands somewhere else.",
    step: 3,
    focus: "Next token distribution",
  },
  {
    id: "min-p-cutoff",
    sections: ["Reshaping the odds before the draw", "What decoding does"],
    title: "The min-p cut-off",
    look: "With Min-p on, a second formula row appears under the first one: it works out the cut-off from the top character’s odds and counts how many are kept.",
    step: 2,
    state: { temperature: 1, minP: 0.1 },
    focus: "Sampler controls",
  },
  {
    id: "surprisal",
    sections: ["How surprised was the model?"],
    title: "How surprised it was",
    look: "Press Sample next token, then read Surprisal −log2 p under the tape; the bar under each character on the tape shows the same measure.",
    step: 3,
    focus: "Generation loop",
  },
  {
    id: "entropy",
    sections: ["How spread out the odds are: entropy"],
    title: "Model and sampler entropy",
    look: "Under the bars, Model entropy reads the outlined bars and Sampler entropy the filled ones; drag Temperature and only Sampler entropy moves.",
    step: 2,
    focus: "Next token distribution",
  },
  {
    id: "three-models",
    sections: ["Three models, one context"],
    title: "Three models side by side",
    look: "Each column lists one model’s three favourite next characters and how much of the text it sees; switch Model and the highlighted column follows.",
    step: 1,
    focus: "Three models, one context",
  },
  {
    id: "pipeline",
    sections: ["Inside the transformer, start to finish", "The whole model, end to end"],
    title: "Seven steps to one character",
    look: "The numbered steps at the top of Layer by layer trace the path from characters to one draw, and the three tagged lens reads here are where the panels below take their readings.",
    step: 4,
    focus: "Layer by layer",
  },
  {
    id: "read-context",
    sections: ["Read the context", "Context"],
    title: "What the bigram sees",
    look: "With Model on Bigram, read the outlined bars for the character after the final J, then switch Model to RNN and then Transformer.",
    step: 0,
    state: { model: "bigram" },
    focus: "Next token distribution",
  },
  {
    id: "seen-unseen",
    sections: ["Seen and unseen"],
    title: "A line it trained on",
    look: "With the training line loaded, read Mean surprisal on this text in the Tiny transformer column, then compare its Held-out loss with its Whole-corpus loss.",
    step: 1,
    state: { prompt: "O Romeo, Romeo! wherefore art thou R", model: "transformer" },
    focus: "Three models, one context",
  },
  {
    id: "temperature-sweep",
    sections: ["Shape the odds", "Shaping"],
    title: "Temperature from low to high",
    look: "Drag Temperature from 0.10 up to 2.00 and watch the filled bars spread out while the order of the rows stays the same.",
    step: 2,
    state: { temperature: 0.1 },
    focus: "Next token distribution",
  },
  {
    id: "greedy-loop",
    sections: ["Run the loop", "The loop"],
    title: "Greedy decoding loops",
    look: "With Top-k at 1 · greedy, press Sample 10 more three times and watch the tape go round the same few words.",
    step: 3,
    state: { topK: 1 },
    focus: "Generation loop",
  },
  {
    id: "read-layers",
    sections: ["Read the layers", "Layer by layer"],
    title: "Three readings of one stream",
    look: "Compare the Entropy of this readout line in the three panels, and find the one tagged final answer first on top here.",
    step: 4,
    focus: "Layer by layer",
  },
];

export default scenes;
