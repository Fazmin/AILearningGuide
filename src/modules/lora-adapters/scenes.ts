import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (rank 4,
 * alpha 8, 24 adapter epochs, adapter attached and unmerged, the calculator on Llama 3 8B with every
 * matrix at rank 16 and 16-bit), so the readouts show the lesson's own numbers. A named calculator
 * preset fixes its own size, so a scene that picks one also sets the `calcD` and `calcLayers` it implies.
 */
const scenes: LabScene[] = [
  {
    id: "size-update",
    sections: ["Size the update", "How to play with it"],
    title: "Rank sets the cost",
    look: "Drag Rank up from 1 and watch B widen, A grow taller and the LoRA bar under Trainable parameter count climb.",
    step: 0,
    state: { rank: 1 },
    focus: "The low-rank update",
  },
  {
    id: "read-diagram",
    sections: ["What the diagram shows"],
    title: "A, B and their product",
    look: "A runs across the top, B down the left and ΔW where they meet, with the full fine-tune’s map and the directions chart below.",
    step: 0,
    focus: "The low-rank update",
  },
  {
    id: "fresh-adapter",
    sections: ["Why a fresh adapter changes nothing", "Why an untrained adapter does nothing"],
    title: "An untrained adapter",
    look: "With Adapter epochs at untrained, B and the ΔW grid are blank while A keeps its random colours.",
    step: 1,
    state: { epochs: 0 },
    focus: "The low-rank update",
  },
  {
    id: "alpha-scale",
    sections: ["What Alpha does", "What Alpha is for"],
    title: "Alpha and the scale",
    look: "The top line reads rank, alpha and scale, so hold Alpha still, slide Rank and watch the scale change with it.",
    step: 0,
    focus: "The low-rank update",
  },
  {
    id: "train-adapter",
    sections: ["Train the adapter"],
    title: "From untrained to trained",
    look: "Adapter epochs reads untrained, so the solid line sits on the dotted one and every pair of bars matches until you raise it.",
    step: 1,
    state: { epochs: 0, probe: "s" },
    focus: "Adapter training and behaviour",
  },
  {
    id: "compare-ranks",
    sections: ["Compare ranks fairly"],
    title: "A fair rank sweep",
    look: "Slide Rank and read the target loss on the LoRA bar, then press Set alpha = rank, for a scale of 1 and read it again.",
    step: 0,
    state: { rank: 1 },
    focus: "The low-rank update",
  },
  {
    id: "hot-swap",
    sections: ["Hot-swap it"],
    title: "Adapter off, then on",
    look: "Flip Serving between Base only and Base + adapter, and watch Target loss, Source loss, Serving weights and the sample change together.",
    step: 2,
    state: { adapterOn: false },
    focus: "Hot-swap and merge",
  },
  {
    id: "merge",
    sections: ["Merge it"],
    title: "Fold the adapter in",
    look: "Press Merge the adapter into W, then read Serving weights, Target loss and the logit gap in the callout under the button.",
    step: 3,
    focus: "Hot-swap and merge",
  },
  {
    id: "size-real",
    sections: ["Size a real adapter"],
    title: "An adapter for Llama 3 8B",
    look: "Read Trainable parameters, Share of the base model and Adapter size at 16-bit, then switch the MLP button off and watch them fall.",
    step: 4,
    focus: "Adapter size calculator",
  },
  {
    id: "cheap-keeps-up",
    sections: ["The cheap adapter keeps up", "What to notice"],
    title: "The cheap adapter keeps up",
    look: "Compare the target losses on the two bars, then read the dashed full fine-tune curve on the directions chart at the rank 4 marker.",
    step: 0,
    focus: "The low-rank update",
  },
  {
    id: "rank-twelve",
    sections: ["More rank is not automatically better"],
    title: "Rank 12 at a small scale",
    look: "With Alpha left at 8 the scale has dropped, so press Set alpha = rank, for a scale of 1 and watch the loss on the LoRA bar fall.",
    step: 0,
    state: { rank: 12 },
    focus: "The low-rank update",
  },
  {
    id: "behaviour-not-weights",
    sections: ["Swapping and merging"],
    title: "Weights kept, behaviour not",
    look: "At a scale of 1.00 Source loss still sits well above the base model’s, and flipping Serving to Base only brings it back exactly.",
    step: 2,
    state: { alpha: 4 },
    focus: "Hot-swap and merge",
  },
  {
    id: "real-models",
    sections: ["Real models"],
    title: "The share falls as models grow",
    look: "Switch Architecture from Llama 3.2 1B to Llama 3 8B and then Llama 3 70B, and watch Share of the base model fall.",
    step: 4,
    state: { calcPreset: "llama-3.2-1b", calcD: 2048, calcLayers: 16 },
    focus: "Adapter size calculator",
  },
];

export default scenes;
