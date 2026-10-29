import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: 7 prompt
 * tokens (`context`) and 4 generated with the cache on, Llama 3 8B at 8,192 tokens for 16 sequences,
 * and a draft of 4 at quality 0.70 with seed 7. Those defaults are the numbers the lesson quotes, so
 * most scenes only move the lab to a step and a card.
 */
const scenes: LabScene[] = [
  {
    id: "prefill",
    sections: ["Prefill: reading the whole prompt at once"],
    title: "The prefill pass",
    look: "With Generated tokens at 0 · prefill, the whole lower triangle lights up at once, because every prompt token goes through in one pass.",
    step: 0,
    state: { generated: 0 },
    focus: "Causal attention work",
  },
  {
    id: "decode",
    sections: ["Decode: writing one token per pass"],
    title: "One token per pass",
    look: "Raise Generated tokens one step at a time: each pass adds one row to the grid, and the caption names the token it feeds in and the one it predicts.",
    step: 1,
    state: { generated: 1 },
    focus: "Causal attention work",
  },
  {
    id: "delivery",
    sections: ["Why writing is slow: a delivery problem"],
    title: "Waiting for the weights",
    look: "With one sequence, the Decode step, cache on panel’s memory bar is far longer than its arithmetic bar, while the prefill panel is the other way round.",
    state: { sequences: 1 },
    focus: "Prefill versus decode",
  },
  {
    id: "cache-notes",
    sections: ["The KV cache: notes that never change", "What the cache stores and why it works"],
    title: "Notes kept, not redone",
    look: "Only the newest row is worked out, and the K and V strips under the grid read the earlier slots and write one new one; switch KV cache to Recompute to compare.",
    step: 2,
    focus: "Causal attention work",
  },
  {
    id: "memory-price",
    sections: ["The price: memory", "The size of the trade"],
    title: "What the notes weigh",
    look: "Read KV per token, then compare the KV block on the bar with the weights block beside it.",
    focus: "Memory budget",
  },
  {
    id: "draft-then-check",
    sections: ["Speculative decoding: draft, then check"],
    title: "Draft, then check",
    look: "Each pass line shows what the draft proposed, how many guesses the target kept, and the token it wrote itself.",
    step: 4,
    focus: "Speculative decoding",
  },
  {
    id: "redoing-the-past",
    sections: ["The problem the cache solves: redoing the past"],
    title: "Recomputing every step",
    look: "With KV cache on Recompute, every decode step lights up the whole triangle again, and the dashed cells repeat work the last pass already did.",
    step: 3,
    state: { useCache: false },
    focus: "Causal attention work",
  },
  {
    id: "whole-run",
    sections: ["The triangle"],
    title: "The whole run, both ways",
    look: "At the longest settings with KV cache on Recompute, read Q·K whole run, then switch to Use cache and read it again.",
    step: 3,
    state: { context: 14, generated: 10, useCache: false },
    focus: "Causal attention work",
  },
  {
    id: "memory-over",
    sections: ["The memory bar"],
    title: "Too many conversations",
    look: "With Concurrent sequences at 64, the KV block runs past the 80 GB marker and turns red, and Memory left becomes Over by.",
    state: { sequences: 64 },
    focus: "Memory budget",
  },
  {
    id: "timing-bars",
    sections: ["The timing bars", "Real scale"],
    title: "Prefill against decode",
    look: "In each panel the solid bar sets the time, so compare the printed times rather than the bar lengths.",
    step: 3,
    focus: "Prefill versus decode",
  },
  {
    id: "speculative-readouts",
    sections: ["Speculative decoding"],
    title: "Tokens per check",
    look: "Read Acceptance rate α and the two tokens-per-pass readouts, then set Draft quality to 1.00 and watch every pass keep all its guesses.",
    step: 4,
    focus: "Speculative decoding",
  },
];

export default scenes;
