import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Image to patches": {
    title: "Tiles, not pixels, become tokens",
    summary:
      "An 8×8 binary image is cut into non-overlapping p×p tiles. Each tile is flattened row by row into a vector x of p² numbers, and each x will become exactly one token. Smaller tiles mean more tokens.",
    whatYouSee: [
      "An 8×8 pad: filled cells are ink, heavy lines are patch borders, tinted cells are the selected patch, and a dashed outline marks Selected pixel.",
      "Patch size (2×2, 4×4 or 8×8) and Selected patch.",
      "Metrics Patch tokens and Pixels per patch, and the selected patch flattened into a strip of p² cells.",
    ],
    howItWorks: [
      "Patch tokens = (8 / p)². Patch k covers rows ⌊k / (8/p)⌋·p to +p and the matching columns; x lists its pixels row-major.",
      "Clicking a cell flips it and selects its patch. Selected pixel plus Flip selected pixel is the keyboard route.",
    ],
    controls: [
      "The pad, Selected pixel, Flip selected pixel, Patch size and Selected patch.",
      "Comparison worth running: Patch size 2×2 against 8×8 — 16 tokens of 4 pixels, or 1 token of 64.",
    ],
    notice: [
      "The top bar of the 5 crosses the vertical border at 4×4, so two tokens each hold part of it.",
      "Nothing about the cut depends on what is drawn. Patching is fixed geometry.",
    ],
    limits: [
      "In this lab: one binary channel at 8×8. Real encoders read three colour channels at hundreds of pixels per side.",
      "In general: some models tile large images into several crops or use variable resolution, which changes the token count but not the idea.",
    ],
  },

  "Patch embedding": {
    title: "z = W·x + position",
    summary:
      "The patch vector x is multiplied by a 4 × p² matrix W. Each row of W is a filter; its output is that filter dotted with the patch. A position code is added so the token remembers where its tile was.",
    whatYouSee: [
      "The selected patch x, then four filters drawn as p×p grids of + (blue) and − (red) weights, each with its output W·x.",
      "A table: each row of W, its W·x, the position value, and the resulting token.",
    ],
    howItWorks: [
      "Filters: ink = 1/p² everywhere; left − right and top − bottom = ±2/p² by half; ╲ − ╱ = ±1/p on the two diagonals. All but ink sum to zero.",
      "position(row, col) = 0.1 × [sin a_r, cos a_r, sin a_c, cos a_c] with a = π(index + ½)/(8/p).",
    ],
    controls: [
      "Selected patch, Patch size, and any edit to the pad.",
      "Comparison worth running: patch #1 against patch #2 at 4×4 — both have ink, but only #2 is lopsided left to right.",
    ],
    notice: [
      "A patch with ink balanced on both halves gives 0 on the contrast filters, even though it is not blank.",
      "Two identical tiles in different places get different tokens only because of the position code.",
    ],
    limits: [
      "In this lab: four hand-set filters and a fixed sine position code. Nothing is learned.",
      "In general: a ViT learns hundreds of filters (768 for ViT-B/16, each 16×16×3) and usually learned position embeddings, and its later layers mix information across patches.",
    ],
  },

  "Token sequence": {
    title: "One sequence, two kinds of token",
    summary:
      "The language model's input is a single sequence. Text tokens come from the tokenizer; image tokens come from the patches, one per tile, in reading order. After a projector sets them to the right width, attention treats them like any other position.",
    whatYouSee: [
      "Text tokens as labelled chips, and image tokens as dashed chips showing their tile, numbered in order.",
      "The selected patch outlined; clicking any image chip selects it.",
    ],
    howItWorks: [
      "Layout follows the LLaVA convention: image tokens replace an image placeholder between the user tag and the question.",
      "Token count = 1 + (8/p)² + 6 here; only the middle part depends on Patch size.",
    ],
    controls: [
      "Click an image chip to select that patch. Patch size changes how many chips there are.",
      "Comparison worth running: 2×2 against 8×8 — the question stays 7 text tokens while the image goes from 16 tokens to 1.",
    ],
    notice: [
      "The image uses context budget exactly like words do.",
      "Reading order is a flattening choice; the position code is what keeps the 2-D layout recoverable.",
    ],
    limits: [
      "In this lab: text tokens are whole words for readability, and no projector or language model runs.",
      "In general: the text tokenizer splits words into subword pieces, and each model uses its own special tokens around images.",
    ],
  },

  "Real-scale arithmetic": {
    title: "Shapes at real resolutions",
    summary:
      "The same cut at production sizes, computed from Image side and Encoder patch. The pipeline shows each stage's tensor shape for the chosen connector: a CLIP-style contrastive head, a LLaVA-style projector, or a Flamingo-style resampler with cross-attention.",
    whatYouSee: [
      "Image side (224, 336, 448 px), Encoder patch (14, 16, 32), CLS token, and Connector.",
      "Five stages with a shape and a note each, from pixels to where the image ends up.",
      "Metrics: Patch tokens, Encoder sequence, Values per patch, and the share of a 4,096-token context the image takes.",
    ],
    howItWorks: [
      "Patch tokens = ⌈side / patch⌉²; values per patch = patch² × 3; encoder sequence = patch tokens + 1 with CLS. A side that does not divide is padded up.",
      "Widths: ViT-L/14 1,024 and ViT-B 768 inside the encoder; CLIP's joint space 768 for ViT-L/14; a 7B language model 4,096; Flamingo-style 64 resampled latents.",
    ],
    controls: [
      "Image side, Encoder patch, CLS token and Connector.",
      "Comparison worth running: 224px with 16×16 (196, or 197 with CLS) against 336px with 14×14 (576).",
    ],
    notice: [
      "Only the LLaVA-style connector puts image tokens into the language model's own sequence: 576 of 4,096 is 14.1%.",
      "CLIP produces one vector per image to compare with captions. It does not generate text.",
    ],
    limits: [
      "In this lab: fixed widths per encoder family and one 4,096-token context. Only the shapes are computed: contrastive training and cross-attention are not run. Flamingo itself used an NFNet image encoder, not a ViT.",
      "In general: newer models merge neighbouring patches, tile high-resolution images, or resample to a fixed count, so token counts vary widely across models.",
    ],
  },

  "Contrastive pairs": {
    title: "Why the diagonal wins",
    summary:
      "Three images and three captions, each a short hand-authored vector. The card computes every image–caption cosine similarity, divides by a temperature, and applies a softmax across each row (which caption fits this image?) and each column (which image fits this caption?). The contrastive loss is the average −log of the labelled match's share, so it is small only when the labelled pairs score highest.",
    whatYouSee: [
      "A 3 × 3 table: rows are images, columns are captions. Each cell shows the cosine similarity in bold, the share that caption gets in its row, the share that image gets in its column, and the words labelled match on the pair the labels call a match.",
      "Metrics: Contrastive loss, its two directions (Image → caption and Caption → image), the chance loss ln 3, and how many images and captions rank their labelled match first.",
      "A second table listing the six hand-authored vectors over four named topics: vehicle, animal, sky and structure.",
    ],
    howItWorks: [
      "cos(a, b) = a·b / (|a||b|), and logit = cos / τ. A row share is the softmax of one image's three logits; a column share is the softmax of one caption's three logits.",
      "Image → caption loss is the mean over images of −log(row share of the labelled match); Caption → image loss is the same down the columns. Contrastive loss is their average, the symmetric cross-entropy that CLIP uses (Radford et al., 2021). The chance loss, ln 3, is what an even guess over three candidates scores.",
      "In training, the gradient on each logit is its share, minus 1 for the labelled match. So the labelled pair is pulled up and every other pairing in the batch, the negative pairs, is pushed down.",
    ],
    controls: [
      "Temperature τ (0.01 to 1.00) and Which pairs count as matches (true pairs or shuffled captions).",
      "Comparison worth running: τ = 0.05 against 1.00 on true pairs — the loss goes from 0.006 to 0.813, and at 1.00 the labelled caption gets only 44%, 46% and 43% of its row.",
      "Second comparison: true pairs against shuffled captions at τ = 0.20 — the same matrix, with loss 0.235 against 2.582.",
    ],
    notice: [
      "Image C, a ship under storm clouds, scores 0.74 with the ferry caption and 0.93 with its own. Its own caption still wins, but at τ = 0.20 it holds only about 70% of the row, because a ship is a ship.",
      "Shuffling the labels changes the loss and nothing else: every cosine stays the same. The loss measures how well the scores agree with the labels.",
      "CLIP's temperature was initialized to the equivalent of 0.07 and clipped so that logits are scaled by no more than 100 (Radford et al., 2021). The slider's lowest value, 0.01, is that limit.",
    ],
    limits: [
      "In this lab: the vectors are hand-authored, four numbers each with topic names for readability. No image or text encoder runs and nothing is trained, so the card shows what the loss measures, not how encoders learn to satisfy it. With three pairs, each image has only two negatives.",
      "In general: CLIP-style training works on very large batches (the CLIP paper reports 32,768 pairs), so each image faces tens of thousands of negatives, and its vectors are anonymous and have hundreds of dimensions. A trained alignment is measured on tasks, not guaranteed.",
    ],
  },
};

export default cardInfo;
