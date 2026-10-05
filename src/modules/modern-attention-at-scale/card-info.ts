import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "KV cache budget": {
    title: "How many keys and values you must keep",
    summary:
      "The decode-time KV cache for a model with Llama 3 8B's shape: 32 layers, 32 query heads of width 128, 16-bit values. Sharing key-value heads across query heads (GQA, MQA) or keeping only a window of recent tokens shrinks it; the attention scores are computed the same way.",
    whatYouSee: [
      "KV heads: 32 is multi-head attention, 16 to 2 are grouped-query attention, 1 is multi-query attention. All six divide the 32 query heads evenly.",
      "Attention span: Full keeps every token's K and V; Sliding window 4,096 keeps only the most recent 4,096.",
      "A head diagram: 32 query-head squares linked to the KV-head blocks they share.",
      "Bars for all six KV-head settings at the current sequence length, with the selected one highlighted, and a formula row that multiplies the terms out.",
      "Metrics: bytes per token across all layers, total cache, and the cache as a percentage of MHA's.",
    ],
    howItWorks: [
      "KV cache = 2 (K and V) × 32 layers × KV heads × 128 × 2 bytes × cached tokens.",
      "Cached tokens are the sequence length, or min(sequence length, 4,096) with the window.",
      "With 8 KV heads that is 128 KiB per token, the figure for Llama 3 8B. At 8,192 tokens: 4.00 GiB for MHA, 1.00 GiB for 8 KV heads, 128 MiB for MQA.",
    ],
    controls: [
      "`KV heads` picks the number of cached key-value heads.",
      "`Attention span` switches between Full and Sliding window 4,096.",
      "`Sequence length` steps through powers of two from 1K to 128K tokens.",
    ],
    notice: [
      "The cache is proportional to KV heads: 8 KV heads cost exactly a quarter of MHA at any length.",
      "At 128K tokens, 8 KV heads need 16 GiB, more than the model's own 16-bit weights (about 15 GiB). The window caps it at 512 MiB.",
      "With the window, a token reaches 4,096 back directly and up to 131,072 back only by relaying through 32 layers.",
    ],
    limits: [
      "In this lab: counts are closed-form for one sequence. Batching multiplies the cache by the batch size, and paged or quantized caches change the constant.",
      "In this lab: a model trained with 32 KV heads cannot simply be switched to fewer; GQA models are trained that way or converted and fine-tuned.",
      "In general: sharing KV heads forces the grouped query heads to read the same stored keys and values, a quality cost that grows as groups get larger. A window drops direct access to older tokens.",
    ],
  },

  "Quadratic cost": {
    title: "When the n² term takes over",
    summary:
      "Prefill FLOPs for one layer of the same model as the prompt grows: the weight matmuls grow linearly with tokens, full causal attention grows with the square, and a 4,096-token window grows linearly again.",
    whatYouSee: [
      "Three curves in TFLOP against sequence length in thousands of tokens: weight matmuls (solid), attention with the full span (dashed), and attention with a 4,096 window (dotted). A vertical line marks the current length.",
      "Metrics: attention's share of the layer's FLOPs at the current length and span, the crossover length where attention equals the weight matmuls, the memory a stored score matrix would take, and what FlashAttention keeps instead.",
    ],
    howItWorks: [
      "Weight matmuls: 2 FLOPs per weight per token, over the layer's 218,103,808 weights with 8 KV heads (Q and output projections, K and V projections, and a SwiGLU MLP of width 14,336).",
      "Attention: 4 × 32 heads × 128 FLOPs per causal query–key pair, half for q·k and half for mixing values. Full causal attention has n(n + 1)/2 pairs; a window caps each query at 4,096 keys.",
      "Score matrix if stored: 32 heads × n² × 2 bytes. FlashAttention keeps one 4-byte log-sum-exp per row per head: 32 × n × 4 bytes.",
    ],
    controls: [
      "This card has no controls of its own. `Sequence length`, `KV heads`, and `Attention span` on the KV cache budget card drive it.",
    ],
    notice: [
      "At 8K tokens attention is 13.3% of the layer's FLOPs; at 128K it is 71.1%. The crossover is 53,247 tokens with 8 KV heads.",
      "KV heads barely moves the curves: every query head still scores every key. Only the K and V projection weights shrink.",
      "A stored score matrix would be 4.00 GiB per layer at 8K tokens and 1.00 TiB at 128K. FlashAttention keeps 1 MiB and 16 MiB.",
    ],
    limits: [
      "In this lab: FLOPs are counted, not timed. Real speed also depends on memory traffic, kernel efficiency, and whether the kernel skips masked tiles.",
      "In this lab: only prefill is shown. During decoding each new token does 2 × weights FLOPs plus attention over the whole cache, and speed is limited by reading the weights and, at long context, the cache from memory.",
      "In general: sub-quadratic alternatives such as sliding windows, sparse patterns, linear attention, and state-space layers trade away some direct long-range access for lower cost.",
    ],
  },

  "FlashAttention tiles": {
    title: "Exact softmax without the whole row",
    summary:
      "One query's attention over 16 keys, processed in tiles the way FlashAttention does: keep a running max, a running sum, and a running output, and rescale them whenever a tile brings a larger score. After the last tile the output equals the ordinary softmax to rounding.",
    whatYouSee: [
      "A row of 16 keys with their scores q·k. Finished tiles are shaded, the tile currently in fast on-chip memory (SRAM) is outlined, and waiting tiles are grey.",
      "Weights below each key: a dashed outline for the full softmax weight and a filled bar for exp(s − m) ÷ ℓ using the statistics so far.",
      "A table with one row per processed tile: tile max, running max m, rescale factor, running sum ℓ, and the normalised output o ÷ ℓ.",
      "A final row comparing the online output with the full softmax output.",
    ],
    howItWorks: [
      "For each tile: m_new = max(m, tile max); scale ℓ and o by exp(m − m_new); add exp(s − m_new) to ℓ and exp(s − m_new)·v to o.",
      "The output is o ÷ ℓ. Because every earlier term was rescaled to the same final max, it equals Σ softmax(s)ᵢ vᵢ exactly, apart from floating-point rounding (8.3e-17 here with 4-key tiles).",
      "Scores and values are fixed teaching numbers; values are 2-dimensional unit vectors.",
    ],
    controls: [
      "`Tile size` picks 2, 4, 8, or 16 keys per tile and shows the finished result.",
      "`Tiles processed` steps from none to all of them.",
    ],
    notice: [
      "When a tile raises the running max, the filled bars of earlier keys shrink: the rescale factor, 0.4966 for tile 2 with 4-key tiles, multiplies everything seen so far.",
      "Before the last tile the partial bars overshoot the outlines, because ℓ has not seen every key. They land on the outlines exactly at the end.",
      "Every tile size gives the same final output. Tiling changes the order of work, not the answer.",
    ],
    limits: [
      "In this lab: one query row and 16 keys. The real kernel tiles blocks of queries too, runs the backward pass by recomputing scores from the saved log-sum-exp, and is tuned to a GPU's SRAM size.",
      "In general: FlashAttention cuts memory traffic and the n² score matrix, which makes long contexts affordable, but it still does quadratic arithmetic and does not shrink the KV cache.",
    ],
  },

  "RoPE rotation": {
    title: "Relative position from absolute rotations",
    summary:
      "Rotary position embedding rotates each pair of query and key dimensions by position × θ for that pair. Rotating q by m and k by n makes their dot product depend only on m − n. The lab uses a toy 8-dimensional head so every pair is visible.",
    whatYouSee: [
      "A unit circle for the selected pair: dashed rays are the unrotated q and k, solid rays are q rotated by m·θ (round dot) and k rotated by n·θ (square).",
      "Controls for query position m, key position n, both shifted together, the dimension pair, and the RoPE base.",
      "A Score row with the full 8-dimensional dot product after rotation, and the same score after shifting both positions.",
      "A chart of the selected pair's share of q·k against distance, for base 10,000 and base 500,000.",
    ],
    howItWorks: [
      "θᵢ = base^(−2i/8) for pair i. Each pair (x, y) becomes (x cos a − y sin a, x sin a + y cos a) with a = position × θᵢ.",
      "Rotations preserve length, and rot(q, m)·rot(k, n) = Σᵢ |qᵢ||kᵢ| cos(φᵢ + (m − n)θᵢ), so absolute positions cancel.",
      "Wavelength 2π/θ is how many tokens one full turn of that pair takes.",
      "q and k are fixed teaching vectors, not weights from a model.",
    ],
    controls: [
      "`Query position` and `Key position` run from 0 to 96.",
      "`Shift both −16` and `Shift both +16` move both positions together.",
      "`Dimension pair` selects dims 0,1 through 6,7; `RoPE base` switches between 10,000 and 500,000.",
    ],
    notice: [
      "Shift both +16 turns both rays but leaves the Score unchanged: 2.0865 at m = 20, n = 7 and at m = 36, n = 23.",
      "Pair 0 turns a full circle every 6.3 tokens; pair 3 takes 6,283 with base 10,000. Fast pairs resolve nearby order, slow pairs change over long distances.",
      "Raising the base to 500,000 stretches pair 1's wavelength from 63 to 167 tokens.",
    ],
    limits: [
      "In this lab: 8 dimensions and one q, k pair. Llama heads have 128 dimensions, 64 pairs, whose slowest wavelength is about 54,000 tokens at base 10,000 and 2.6 million at 500,000.",
      "In general: RoPE encodes relative distance but does not make a model generalise past the lengths it was trained on. Beyond that, attention degrades unless the frequencies are rescaled (position interpolation, NTK-aware scaling, YaRN) and the model is usually trained further.",
    ],
  },
};

export default cardInfo;
