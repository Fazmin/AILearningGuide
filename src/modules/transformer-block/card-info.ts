import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Transformer block": {
    title: "One block, computed stage by stage",
    summary:
      "A real, tiny transformer block running in the page: width 8, two causal attention heads of width 4, a GELU MLP of width 32, and LayerNorm. Each row shows the actual vector for one token at that stage, and the residual stream runs down the left rail.",
    whatYouSee: [
      "Rows in data order. Rows on the rail are the residual stream itself; indented dashed rows are sublayers that read a copy of the stream and write back at the next `+`.",
      "Each row's 8-dimensional vector as diverging bars (positive up, negative down) on one shared scale, plus its Euclidean norm.",
      "On the Self-attention row, both heads' weights from the selected token to every earlier token, as percentages that sum to 100 per head.",
      "On the MLP row, how many of the 32 hidden GELU units are positive for this token.",
      "An `on`/`off` pill on each switchable row. Switched-off rows fade and write zeros.",
    ],
    howItWorks: [
      "Pre-norm, as in GPT-2 and Llama: `h = x + Attn(LN(x))`, then `out = h + MLP(LN(h))`.",
      "Post-norm, as in the 2017 Transformer: `h = LN(x + Attn(x))`, then `out = LN(h + MLP(h))`. The norm rows move onto the rail because they rescale the stream itself.",
      "Attention: q, k, v = x·W, scores q·k ÷ √4 over positions up to this one, softmax, weighted sum of v per head, concatenate, then the output projection.",
      "MLP: x·W₁ (8 → 32), GELU (tanh approximation), then ·W₂ (32 → 8). LayerNorm subtracts the mean, divides by the standard deviation (ε = 1e-5), then applies a scale of 1 and shift of 0.",
      "Inputs are a seeded vector per token plus a sinusoidal position vector, multiplied by Input scale. Weights are seeded Gaussians with standard deviation 1/√fan-in.",
    ],
    controls: [
      "`Placement` switches between Pre-norm and Post-norm.",
      "`Token` picks which of the six positions the rows describe.",
      "`Layer norm`, `Self-attention`, and `MLP` switch their rows. The Layer norm button switches both norms together; `Residual add` rows cannot be switched, because removing the addition removes the stream.",
    ],
    notice: [
      "With Layer norm on in Pre-norm, the Self-attention and MLP writes keep the same size at any Input scale: LN(s·x) = LN(x). Turn it off and push Input scale to 8×: attention weights collapse to 100% on one token.",
      "Self-attention is the only row that mentions other positions. The first token, `the` at position 0, can only attend to itself.",
      "In Post-norm the output norm is always √8 ≈ 2.83 with the norm on, whatever the input: the stream's own scale is erased at every addition.",
    ],
    limits: [
      "In this lab: the weights are random and untrained, so the vectors show the plumbing, not learned features. One shared Layer norm switch stands in for two independent norms, and there is no dropout, no biases in attention, and a width of 8 instead of thousands.",
      "In this lab: switching a sublayer off here is exact, because nothing downstream was trained to expect it.",
      "In general: ablating a sublayer in a trained model leaves the later layers in a state they never saw, so the damage overstates that component's isolated role. Mean-ablation and activation patching in the circuits lab reduce that confound.",
      "In general: production blocks vary what this one fixes: RMSNorm instead of LayerNorm, a gated SwiGLU MLP, grouped-query attention, and mixture-of-experts layers that route each token to a few of many MLPs.",
    ],
  },

  "Residual stream": {
    title: "The sum every sublayer writes into",
    summary:
      "One token's residual vector decomposed dimension by dimension into the input and the two sublayer writes. In a pre-norm block the output is exactly their sum, which is what makes the stream a shared, additive channel.",
    whatYouSee: [
      "Eight rows, one per dimension d0–d7, on a shared horizontal axis centred at zero.",
      "Three lanes per row: the top lane runs from 0 to the input value, the middle lane adds the attention write, and the bottom lane adds the MLP write, each starting where the previous one ended.",
      "A diamond at the block's output value for that dimension.",
      "Input scale, and metrics for the input, attention write, MLP write, and output norms, the cosine between output and input, and how far this token's output moves if the other five words change.",
    ],
    howItWorks: [
      "Lane ends are x_i, x_i + a_i, and x_i + a_i + m_i, where a and m are the attention and MLP writes from the Transformer block card.",
      "Pre-norm: the diamond equals x_i + a_i + m_i exactly. Post-norm: the diamond is LN(LN(x + a) + m)_i, so it generally sits elsewhere.",
      "Output shift if other words change: the block is run a second time with every other position swapped (the cat sat on the mat → a dog ran to a rug, keeping this token), and the lab prints the distance between the two outputs at this position.",
      "Input scale multiplies every input vector before the block runs.",
    ],
    controls: [
      "`Input scale` runs from 0.25× to 8×. Compare Layer norm on and off at 8×.",
      "The sublayer switches, Placement, and Token live on the Transformer block card and change this card too.",
    ],
    notice: [
      "Turn Self-attention off and the output shift is exactly 0.000: the norms and the MLP work on each position alone.",
      "For `mat` in Pre-norm with the norm on, cos(output, input) is 0.995 at Input scale 8×: the writes are a small nudge on a large stream. At 0.25× the same writes dominate and the cosine falls to 0.178.",
      "Switch to Post-norm and the diamonds leave the end of the lanes. The output is no longer the input plus a correction.",
    ],
    limits: [
      "In this lab: one token of one untrained block. A real stream passes through dozens of blocks, and each one's writes accumulate on top of the last.",
      "In general: the stream is a fixed-width channel shared by every sublayer at every depth, so writes compete for the same directions, one mechanism behind superposition in the interpretability labs.",
      "In general: layer normalization is not passive. Its learned scale and its per-token statistics carry information that later layers use.",
    ],
  },

  "Parameter count": {
    title: "Counting a block from its shape",
    summary:
      "Weight counts for one block at a chosen width, split into attention, MLP, and norms, compared with the 12·d² rule of thumb, and multiplied by the layer count. Presets reproduce GPT-2 small's and Llama 2 7B's published totals.",
    whatYouSee: [
      "Preset buttons, Model width, Layers, and choices of MLP and Norm.",
      "A bar split by parameter share: attention, MLP (hatched), and a sliver for the norms.",
      "Formula rows with the exact counts, the block total against 12·d², and metrics for all blocks, the MLP share, and the toy block's 840 parameters.",
    ],
    howItWorks: [
      "Attention: four d × d matrices for Q, K, V, and the output projection, 4d².",
      "GELU MLP: up d × 4d and down 4d × d, 8d². SwiGLU: gate, up, and down with hidden width 8d/3 rounded up to a multiple of 256, also about 8d².",
      "Norms: two per block. LayerNorm has a scale and a shift (2d each); RMSNorm has only a scale (d each).",
      "GPT-2 small preset: 12 blocks with biases, plus 50,257 × 768 token and 1,024 × 768 position embeddings and a final LayerNorm, gives 124,439,808. Llama 2 7B preset: 32 blocks, plus untied 32,000 × 4,096 input and output embeddings and a final RMSNorm, gives 6,738,415,616.",
    ],
    controls: [
      "`GPT-2 small` and `Llama 2 7B` load published configurations.",
      "`Model width` (256 to 8,192) and `Layers` (1 to 128) change the shape; `MLP` and `Norm` change the block's recipe.",
    ],
    notice: [
      "The MLP holds two-thirds of every block with either MLP type: 66.6% for GELU 4×, 66.8% for Llama's SwiGLU at d = 4,096.",
      "Norm parameters are about 1/(3d) of a block, 0.04% at d = 768, so LayerNorm versus RMSNorm matters for speed and stability, not size.",
      "Doubling Model width quadruples the block: parameters grow with d².",
    ],
    limits: [
      "In this lab: counts assume standard multi-head attention. Grouped-query attention shrinks the K and V matrices, so a GQA block has fewer than 4d² attention weights.",
      "In this lab: outside the presets the card counts weights only, with no biases or embeddings.",
      "In general: parameter count sets memory, not compute per token alone. Mixture-of-experts models hold many MLPs per block but run only a few for each token.",
    ],
  },

  "Position": {
    title: "Why a block needs position information",
    summary:
      "The same six words are run through the lab's block in their original order and in a reordering, with every word allowed to read every other. Without position information each word gets the same output in both orders, so the block cannot tell the sentences apart. With it, the outputs differ.",
    whatYouSee: [
      "`Word order`: Original, `Swap cat ↔ mat` (the cat sat on the mat becomes the mat sat on the cat), or `Shuffle all` (mat on cat the the sat). Two rows of numbered chips show both sentences, with the slot of each word.",
      "`Position information`: Off feeds the block each word's own vector; On feeds it the same vector plus a sinusoidal position vector, exactly the input the other cards use.",
      "A table with one row per word of the original sentence: its slot before and after (`1 → 5`), the output shift (the distance between that word's block output in the two orders, with a bar), and a text verdict, `same output` or `different output`.",
      "Metrics for how many of the six words changed output and the largest shift. A note below states the result in words.",
    ],
    howItWorks: [
      "The block runs twice, once on the original word list and once on the reordered one, using the Layer norm, Self-attention, MLP, Placement and Input scale settings from the other cards. Each word's output in the first run is compared with the output of the same word in the second.",
      "Without position information, the input is the seeded token vector alone. Every step of the block is then either position-wise (norms, MLP) or attention, whose scores depend only on the vectors being compared. So a word's output depends on which words are present and not where they sit, and reordering only reorders the outputs.",
      "With position information the input is `embed(word, slot)`: the token vector plus a position vector built from sines and cosines of the slot. The same word in a different slot now enters the block as a different vector.",
      "The card runs attention without the causal mask, so each word reads all six. That isolates order from the mask.",
    ],
    controls: [
      "`Word order`: pick `Swap cat ↔ mat`, then set `Position information` to Off. Every shift reads 0.000 and Words whose output changed reads 0 of 6, though the sentence now says something different.",
      "Switch `Position information` to On with the swap. At the default settings all six words change output: cat and mat move most because their own slots changed, and the other four move less because only their neighbours moved.",
      "Turn `Self-attention` off on the Transformer block card with Position information On and the swap selected: only cat and mat change output, because without attention a word's output depends on its own input alone.",
    ],
    notice: [
      "With position information off, the shifts are zero at every setting of the other cards: Layer norm, MLP, Placement and Input scale cannot supply order, because none of them looks at slots.",
      "`Shuffle all` moves every word, and the result is the same: identical output vectors in new slots with position information off, six different ones with it on.",
      "Position information does not tell a word what is near it. It makes the input depend on the slot, and attention then has to learn to use that signal.",
    ],
    limits: [
      "In this lab: the position vector is a fixed sinusoidal pattern of amplitude 0.5 added to an untrained block's input. Nothing here learns to use order, so the card shows that the block can be sensitive to position, not that it understands it.",
      "In this lab: the card lifts the causal mask. Under the mask a word can still tell how many words precede it, because the first word sees only itself and the last sees all six, so a masked block leaks some order even with no position vector.",
      "In general: this block adds a sinusoidal vector to the embedding. Learned absolute vectors, relative biases such as ALiBi, and rotary embeddings supply order differently; many current models, Llama among them, rotate queries and keys inside attention (see Modern attention at scale) instead of adding any position vector at the input.",
      "In general: position information is necessary but not sufficient. A trained model has to learn to use it, and how well it handles positions beyond those seen in training depends on the scheme chosen.",
    ],
  },
};

export default cardInfo;
