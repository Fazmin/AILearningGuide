import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `selected` is the 0-based patch index in reading
 * order, so 1 is patch #2; it must stay below the patch count for the chosen Patch size. Every scene
 * keeps the default blocky 5 on the pad, and Real-scale arithmetic starts at 336px with 14×14 tiles,
 * a CLS token and the LLaVA connector.
 */
const scenes: LabScene[] = [
  {
    id: "cut-patches",
    sections: ["Step 1: cut the picture into patches"],
    title: "Cut into patches",
    look: "Heavy lines cut the pad into tiles, and Patch tokens works out how many there are from the pad side and the tile side.",
    step: 1,
    state: { patchSize: 4, selected: 1 },
    focus: "Image to patches",
  },
  {
    id: "filters",
    sections: ["Step 2: turn each patch into a list of numbers"],
    title: "Four filters, four numbers",
    look: "Each filter tile marks its + and − cells, and the number under it is that filter’s answer for patch #2.",
    step: 2,
    state: { patchSize: 4, selected: 1 },
    focus: "Patch embedding",
  },
  {
    id: "position-code",
    sections: ["Step 3: add a position code"],
    title: "Twin tiles, different places",
    look: "At 2×2, step Selected patch between #2 and #3: the W·x column stays the same and only the + position column changes.",
    step: 2,
    state: { patchSize: 2, selected: 1 },
    focus: "Patch embedding",
  },
  {
    id: "three-designs",
    sections: ["Three ways to hand a picture to a language model", "Getting images into a language model"],
    title: "Where the picture goes",
    look: "With LLaVA (projector), the last stage puts every patch token into the line, and Context share of 4,096 shows how much room one picture takes.",
    step: 4,
    state: { side: 336, vitPatch: 14, cls: "on", connector: "llava" },
    focus: "Real-scale arithmetic",
  },
  {
    id: "contrastive",
    sections: ["How contrastive learning works"],
    title: "Three pictures, three captions",
    look: "Each cell holds one picture-and-caption score with its row and column shares, and the outlined cells are the labelled matches.",
    step: 5,
    focus: "Contrastive pairs",
  },
  {
    id: "temperature",
    sections: ["How the temperature changes the shares"],
    title: "A low temperature",
    look: "At τ = 0.05, row C’s labelled caption takes almost the whole row; drag Temperature τ up to 1.00 and watch the shares even out.",
    step: 5,
    state: { temperature: 0.05 },
    focus: "Contrastive pairs",
  },
  {
    id: "cut-grid",
    sections: ["Cut the grid"],
    title: "Smaller tiles, more tokens",
    look: "Patch tokens counts the 2×2 tiles; set Patch size to 8×8 and watch it, and the picture chips in Token sequence, drop to one.",
    step: 1,
    state: { patchSize: 2, selected: 0 },
    focus: "Image to patches",
  },
  {
    id: "read-token",
    sections: ["Read one token"],
    title: "Read one token",
    look: "Move Selected patch to #2 of 4 and read the = token column, then step back to #1, whose ink is balanced.",
    step: 2,
    state: { patchSize: 4, selected: 0 },
    focus: "Patch embedding",
  },
  {
    id: "token-line",
    sections: ["See the line of tokens"],
    title: "Pictures and words in one line",
    look: "The line runs USER:, one chip per tile in reading order, then the question and ASSISTANT:; click a chip to select its patch.",
    focus: "Token sequence",
  },
  {
    id: "scale-up",
    sections: ["Scale it up"],
    title: "A 224px picture",
    look: "Encoder sequence is one more than Patch tokens because of the CLS token; then set Image side to 336px and Encoder patch to 14×14.",
    step: 3,
    state: { side: 224, vitPatch: 16, cls: "on" },
    focus: "Real-scale arithmetic",
  },
  {
    id: "compare-connectors",
    sections: ["Compare the three designs", "Compare connectors"],
    title: "CLIP, LLaVA, Flamingo",
    look: "With CLIP (contrastive), read the five stages and the context share, then switch Connector to LLaVA (projector) and Flamingo-style (cross-attn).",
    step: 4,
    state: { side: 336, vitPatch: 14, cls: "on", connector: "clip" },
    focus: "Real-scale arithmetic",
  },
  {
    id: "shuffled-labels",
    sections: ["Match pictures to captions", "Match images to captions"],
    title: "Shuffled labels",
    look: "The labels now point at other captions: no score changed, yet Contrastive loss jumps; set Which pairs count as matches back to true pairs to compare.",
    step: 5,
    state: { pairing: "shuffled" },
    focus: "Contrastive pairs",
  },
];

export default scenes;
