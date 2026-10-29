import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: 8 KV
 * heads, full span at 8,192 tokens, 4-key tiles with all 4 processed, and RoPE pair 1 with the query
 * at 20 and the key at 7 on base 10,000. `tiles` counts processed tiles, so `tiles: 2` with 4-key
 * tiles is the step where the second tile brings the higher score.
 */
const scenes: LabScene[] = [
  {
    id: "sharing-shelves",
    sections: ["Trick 1: sharing shelves"],
    title: "Four readers per shelf",
    look: "In the head diagram, the 32 query heads share 8 wider KV blocks, four to each; switch KV heads to watch the blocks merge or split.",
    step: 0,
    focus: "KV cache budget",
  },
  {
    id: "old-folders",
    sections: ["Trick 2: throwing old folders away"],
    title: "A sliding window",
    look: "With Attention span on Sliding window 4,096, the KV cache row multiplies by the window instead of the whole text, and the note under it says how far back a token can reach.",
    step: 1,
    state: { span: "window" },
    focus: "KV cache budget",
  },
  {
    id: "new-leader",
    sections: ["Trick 3: working in small pieces"],
    title: "A new leader arrives",
    look: "At 2 of 4, the second tile brings a higher score, so the Rescale column shrinks everything counted so far by one factor.",
    step: 2,
    state: { tiles: 2 },
    focus: "FlashAttention tiles",
  },
  {
    id: "rotation-circle",
    sections: ["Trick 4: position as a rotation"],
    title: "Two arrows, turned",
    look: "The circle shows pair 1’s query and key arrows, dashed before rotation and solid after, and Pair angle is the angle between the solid ones.",
    step: 3,
    focus: "RoPE rotation",
  },
  {
    id: "notes-stack",
    sections: ["A stack of notes that grows with every token", "Why it’s here"],
    title: "The cache, multiplied out",
    look: "Read the KV cache row from left to right, then compare Per token with Cache and drag Sequence length to watch Cache grow.",
    step: 0,
    focus: "KV cache budget",
  },
  {
    id: "square-growth",
    sections: ["Comparisons that grow with the square"],
    title: "Attention against the weights",
    look: "Follow the dashed Attention, full span line as it curves up past the solid Weight matmuls line, and read Attention share and Crossover under the chart.",
    step: 1,
    focus: "Quadratic cost",
  },
  {
    id: "score-matrix",
    sections: ["The hidden cost of moving numbers"],
    title: "The grid nobody stores",
    look: "Under the chart, compare Score matrix if stored with FlashAttention keeps, then drag Sequence length and watch the first grow much faster.",
    focus: "Quadratic cost",
  },
  {
    id: "share-kv-heads",
    sections: ["Share key-value heads", "How to play with it"],
    title: "From MHA to MQA",
    look: "Starting at 32 · MHA, switch KV heads to 8 · GQA and then 1 · MQA, and watch Per token, Cache and the bars shrink.",
    step: 0,
    state: { kvHeads: 32 },
    focus: "KV cache budget",
  },
  {
    id: "stretch-sequence",
    sections: ["Stretch the sequence"],
    title: "A 128K-token text",
    look: "At 128K tokens the dashed full-span line has curved far above the solid Weight matmuls line; step Sequence length down and watch Attention share fall.",
    step: 1,
    state: { seqLen: 131072 },
    focus: "Quadratic cost",
  },
  {
    id: "tile-walk",
    sections: ["Tile the softmax"],
    title: "Tile by tile",
    look: "Step Tiles processed up one tile at a time and watch the filled bars settle onto their dashed outlines by the last tile.",
    step: 2,
    state: { tiles: 0 },
    focus: "FlashAttention tiles",
  },
  {
    id: "shift-both",
    sections: ["Rotate by position"],
    title: "Shift both positions",
    look: "Press Shift both +16: both solid arrows swing round, yet Score and Pair angle stay the same.",
    step: 3,
    focus: "RoPE rotation",
  },
];

export default scenes;
