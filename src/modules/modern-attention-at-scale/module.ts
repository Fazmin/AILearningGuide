import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const KV_OPTIONS = [32, 16, 8, 4, 2, 1];
const TILE_SIZES = [2, 4, 8, 16];

const initialState: ModuleState = {
  kvHeads: 8,
  span: "full",
  seqLen: 8192,
  tileSize: 4,
  tiles: 4,
  queryPos: 20,
  keyPos: 7,
  pair: 1,
  ropeBase: 10000,
};

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const nearest = (value: number, options: number[]) =>
  options.reduce((best, option) => (Math.abs(option - value) < Math.abs(best - value) ? option : best), options[0]);

/**
 * Version 1 had a Mechanism switch (mha, gqa, ssm), a linear 128–32768 token
 * slider, a free 1–32 KV-head slider, and one RoPE position. Version 2 drops
 * the SSM scan, keeps KV heads to divisors of 32, uses powers of two from 1K
 * to 128K tokens, and has separate query and key positions.
 */
export function hydrateModernAttentionState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    const { mechanism, position, ...rest } = parsed;
    const merged: ModuleState = { ...initialState, ...rest };

    if (typeof mechanism === "string") {
      if (mechanism === "mha") merged.kvHeads = 32;
      else if (mechanism === "gqa") merged.kvHeads = nearest(clampInt(parsed.kvHeads, 8, 1, 32), KV_OPTIONS);
      else merged.kvHeads = 8;
    }
    if (typeof position === "number" && typeof parsed.queryPos !== "number") {
      merged.queryPos = clampInt(position, 20, 0, 96);
      merged.keyPos = 0;
    }

    merged.kvHeads = nearest(clampInt(merged.kvHeads, 8, 1, 32), KV_OPTIONS);
    merged.span = merged.span === "window" ? "window" : "full";
    const tokens = clampInt(merged.seqLen, 8192, 1, 1 << 20);
    merged.seqLen = 2 ** Math.min(17, Math.max(10, Math.round(Math.log2(tokens))));
    merged.tileSize = nearest(clampInt(merged.tileSize, 4, 1, 16), TILE_SIZES);
    merged.tiles = clampInt(merged.tiles, 16 / (merged.tileSize as number), 0, 16 / (merged.tileSize as number));
    merged.queryPos = clampInt(merged.queryPos, 20, 0, 96);
    merged.keyPos = clampInt(merged.keyPos, 7, 0, 96);
    merged.pair = clampInt(merged.pair, 1, 0, 3);
    merged.ropeBase = merged.ropeBase === 500000 ? 500000 : 10000;
    return merged;
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-36-modern-attention-at-scale",
  slug: "modern-attention-at-scale",
  title: "Modern attention at scale",
  group: "inside-models",
  order: 19,
  icon: "Focus",
  accent: "#7b4fc4",
  prerequisites: ["module-09-transformer-block", "module-11-inference-kv-cache"],
  estimatedMinutes: 15,
  steps: ["Share key-value heads", "Stretch the sequence", "Tile the softmax", "Rotate by position"],
  stepInstructions: [
    "Set KV heads to 32 · MHA, then 8 · GQA, then 1 · MQA, and compare Per token and the KV cache bars.",
    "Drag Sequence length to 128K and watch the attention curve pass the weight matmuls in Quadratic cost, then set Attention span to Sliding window 4,096.",
    "In FlashAttention tiles, step Tiles processed from 0 to the end and watch a new running max rescale the earlier weights; at the end the gap to the full softmax is at rounding level.",
    "Press Shift both +16 in RoPE rotation: both vectors turn, but the Score stays the same, because only m − n enters the dot product.",
  ],
  stateVersion: 2,
  tagline: "Keep the attention scores and change how they are stored and computed: shared key-value heads, sliding windows, tiled exact softmax, and position as rotation, each costed on a real model shape.",
  objectives: ["Separate the attention scoring rule from the memory layout used to compute it","Explain what grouped-query and sliding-window attention trade away","Explain why FlashAttention gives the exact same output, and which costs it leaves in place","Read rotary position embedding as a rotation whose query-key score depends only on the distance between positions"],
  glossary: [
  {
    "term": "Multi-head attention",
    "definition": "Several attention heads computed in parallel, each with its own query, key, and value projections. At decode time every head's keys and values must be cached."
  },
  {
    "term": "Grouped-query attention",
    "definition": "Query heads split into groups that share one key-value head, so the cache holds only as many K and V heads as groups. Memory falls in proportion; the grouped queries must read the same stored keys."
  },
  {
    "term": "Multi-query attention",
    "definition": "The extreme of grouped-query attention: every query head shares a single key-value head. It gives the smallest cache and the most quality risk."
  },
  {
    "term": "KV cache",
    "definition": "The stored keys and values of tokens already processed, reused while decoding. Its size is 2 times layers times KV heads times head dimension times bytes per value, per token."
  },
  {
    "term": "Sliding-window attention",
    "definition": "Each query attends only to the most recent W keys. Cost and cache stop growing past W, and longer-range information must hop through layers."
  },
  {
    "term": "Receptive field",
    "definition": "How far back information can reach. With a window W and L stacked layers it is up to L times W, but only by relaying through intermediate positions."
  },
  {
    "term": "Flash attention",
    "definition": "An exact attention kernel that tiles queries, keys, and values through fast on-chip memory and never writes the full score matrix to main memory. It changes memory traffic, not the result."
  },
  {
    "term": "Online softmax",
    "definition": "Computing softmax in one pass over tiles by keeping a running maximum and running sum, and rescaling earlier partial results by exp of the old maximum minus the new one. The result equals the ordinary softmax."
  },
  {
    "term": "HBM and SRAM",
    "definition": "A GPU's large, slower high-bandwidth main memory and its small, fast on-chip memory. Attention at long lengths is often limited by traffic between them rather than by arithmetic."
  },
  {
    "term": "Rotary position embedding",
    "definition": "RoPE rotates each pair of query and key dimensions by the token's position times a per-pair frequency. The query-key dot product then depends only on the distance between the two positions."
  },
  {
    "term": "RoPE base",
    "definition": "The constant that sets RoPE's frequencies, theta i equals base to the power minus 2i over d. A larger base makes the slow pairs rotate more slowly; Llama 3 uses 500,000."
  },
  {
    "term": "Context length",
    "definition": "How many tokens the model is trained to condition on. Memory for the KV cache, attention's quadratic cost, and the positions seen in training decide whether a length works."
  },
  {
    "term": "Context extension",
    "definition": "Stretching a trained model to longer inputs by rescaling RoPE, as in position interpolation, NTK-aware scaling, or YaRN, usually with some further training. Plain RoPE degrades beyond the trained length."
  }
],
  references: [
    {
      authors: "Noam Shazeer",
      title: "Fast Transformer Decoding: One Write-Head is All You Need",
      source: "arXiv preprint arXiv:1911.02150",
      year: 2019,
      url: "https://arxiv.org/abs/1911.02150",
      note: "Introduces multi-query attention, where all heads share one set of keys and values. It explains that decoding is slow because those keys and values must be loaded again for every new token, which is the 1 · MQA end of the KV heads control.",
    },
    {
      authors: "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.",
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 4895–4901",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.298/",
      note: "Defines grouped-query attention, which uses more than one but fewer key-value heads than query heads. It reports that MQA can lose quality while GQA stays close to multi-head attention and decodes almost as fast as MQA, the trade the KV cache budget card shows.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "The source for the model shape every count in the lab uses: the 8B model's 32 layers, 32 attention heads, 8 key-value heads, and MLP width of 14,336. It also states that Llama 3 uses grouped-query attention and raises the RoPE base to 500,000.",
    },
    {
      authors: "Iz Beltagy, Matthew E. Peters, and Arman Cohan",
      title: "Longformer: The Long-Document Transformer",
      source: "arXiv preprint arXiv:2004.05150",
      year: 2020,
      url: "https://arxiv.org/abs/2004.05150",
      note: "Describes sliding-window attention, whose cost grows in a straight line with text length. It shows that with L layers and a window of w, the top layer can reach back L times w tokens, the receptive field in the lesson's glossary.",
    },
    {
      authors: "Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al.",
      title: "Mistral 7B",
      source: "arXiv preprint arXiv:2310.06825",
      year: 2023,
      url: "https://arxiv.org/abs/2310.06825",
      note: "A model that combines grouped-query attention with a sliding window. It explains the rolling buffer cache named under the KV cache formula, where each new token pushes out the oldest, and how information moves up to k times W tokens after k layers.",
    },
    {
      authors: "Gemma Team",
      title: "Gemma 2: Improving Open Language Models at a Practical Size",
      source: "arXiv preprint arXiv:2408.00118",
      year: 2024,
      url: "https://arxiv.org/abs/2408.00118",
      note: "The architecture section says Gemma 2 alternates sliding-window layers of 4,096 tokens with full-attention layers in every other layer. It backs the lesson's point that long-context models mix windowed and full layers.",
    },
    {
      authors: "Gemma Team",
      title: "Gemma 3 Technical Report",
      source: "arXiv preprint arXiv:2503.19786",
      year: 2025,
      url: "https://arxiv.org/abs/2503.19786",
      note: "Explains that the KV cache grows too large at long context and cuts it by placing five local layers, each seeing only 1,024 tokens, between full-attention layers. It is a newer example of the window trade-off in the Attention span control.",
    },
    {
      authors: "Maxim Milakov and Natalia Gimelshein",
      title: "Online normalizer calculation for softmax",
      source: "arXiv preprint arXiv:1805.02867",
      year: 2018,
      url: "https://arxiv.org/abs/1805.02867",
      note: "Shows how to compute softmax in one pass by keeping a running max and a running sum, and multiplying the old sum by exp of the old max minus the new one when a larger value arrives. That is the rescaling you watch in the FlashAttention tiles card.",
    },
    {
      authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.",
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html",
      note: "The paper behind the FlashAttention tiles card. It computes exact attention in tiles to cut reads and writes between the GPU's HBM and on-chip SRAM, and it stores only the softmax statistics so the scores can be recomputed in the backward pass.",
    },
    {
      authors: "Tri Dao",
      title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
      source: "The Twelfth International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html",
      note: "The second version splits the work better across the GPU, including running a single head across many thread blocks, while staying exact. It backs the lesson's line that later versions improve parallelism and the result stays the same.",
    },
    {
      authors: "Jay Shah, Ganesh Bikshandi, Ying Zhang, et al.",
      title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html",
      note: "The third version targets newer Hopper GPUs. It overlaps data movement with arithmetic and adds support for 8-bit (FP8) numbers, the smaller number formats the lesson mentions.",
    },
    {
      authors: "Jianlin Su, Murtadha Ahmed, Yu Lu, et al.",
      title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      source: "Neurocomputing 568, 127063",
      year: 2024,
      url: "https://arxiv.org/abs/2104.09864",
      note: "The paper that introduced RoPE. It rotates each pair of query and key numbers by position, sets each pair's speed with base 10,000, and shows the query-key score depends only on the distance between positions, as Shift both +16 demonstrates.",
    },
    {
      authors: "Shouyuan Chen, Sherman Wong, Liangjian Chen, et al.",
      title: "Extending Context Window of Large Language Models via Positional Interpolation",
      source: "arXiv preprint arXiv:2306.15595",
      year: 2023,
      url: "https://arxiv.org/abs/2306.15595",
      note: "Introduces position interpolation, which squeezes position numbers down so a longer text fits the range of angles seen in training, then fine-tunes briefly. It explains why plain extrapolation past the trained length can break attention.",
    },
    {
      authors: "Bowen Peng, Jeffrey Quesnelle, Honglu Fan, et al.",
      title: "YaRN: Efficient Context Window Extension of Large Language Models",
      source: "The Twelfth International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://proceedings.iclr.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html",
      note: "Starts from the finding that RoPE models fail past their trained length, then reviews position interpolation and NTK-aware scaling. YaRN treats fast and slow pairs differently and adds an attention temperature, the three context extension methods named in the lesson.",
    },
  ],
  checkpoint: [
    {
      prompt: "On KV cache budget you set KV heads from 32 to 8. What happens to the cache, and to the attention arithmetic on Quadratic cost?",
      options: [
        "Both fall to a quarter, because there are a quarter as many key-value heads doing the attention work",
        "The cache is unchanged, because its size follows the number of query heads rather than key-value heads",
        "The cache falls to a quarter; the arithmetic barely moves, since every query head still scores every key",
      ],
      answer: 2,
      explanation: "The cache stores one key and one value per key-value head, so 8 of 32 heads is exactly a quarter of the bytes. Each of the 32 query heads still computes scores against every key, so the n-squared work is the same, and only the key and value projection weights shrink. Grouped-query attention trades memory, not scoring.",
      objective: 1,
    },
    {
      prompt: "A sliding window of 4,096 stops the cache growing at 128K tokens. What does a window give up?",
      options: [
        "Direct access to older tokens, which must now be relayed through intermediate positions across layers",
        "Nothing, because the window only drops keys the model was never going to attend to in the first place",
        "Exactness of the softmax over the keys it keeps, because a window only approximates the weights",
      ],
      answer: 0,
      explanation: "Each query still gets an exact softmax over its window, so the keys kept are scored normally. But a token older than the window cannot be read directly, and information from further back survives only by hopping through intermediate positions, up to layers times window, and it degrades on the way. That is why some models mix windowed and full layers.",
      objective: 1,
    },
    {
      prompt: "You switch a model's attention kernel to FlashAttention. What changes?",
      options: [
        "The attention output, because tiles only approximate the full softmax and lose a little accuracy",
        "How the same scores are computed: in tiles, never storing the full score matrix in main memory",
        "The KV cache size, because FlashAttention stores fewer key-value heads than the standard kernel",
      ],
      answer: 1,
      explanation: "FlashAttention computes exactly the same softmax attention. Online softmax rescales earlier tiles when a larger score arrives, so the final output matches to rounding, as the tiles card shows. It saves memory traffic and the n-squared score matrix, not the KV cache, which only fewer KV heads or a window shrink.",
      objective: 0,
    },
    {
      prompt: "In FlashAttention tiles you step Tiles processed and a later tile brings a larger score. What happens to the bars for the earlier keys?",
      options: [
        "They stay where they are, because earlier tiles are finished and their results are never revisited",
        "They grow, because a larger score adds weight to every key that was already seen in earlier tiles",
        "They shrink, because the running max rose and earlier partial results are rescaled",
      ],
      answer: 2,
      explanation: "Online softmax keeps a running max, sum and output. When a tile raises the max, everything accumulated so far is multiplied by exp of the old max minus the new one, which is below one, so the earlier shares shrink. After the last tile the output equals the ordinary softmax to rounding.",
      objective: 2,
    },
    {
      prompt: "On Quadratic cost, attention's share of FLOPs grows with sequence length. Does FlashAttention flatten that curve?",
      options: [
        "Yes: tiling skips the lower-scoring keys, so far fewer query-key pairs are computed per query",
        "No: it drops the stored score matrix and most memory traffic, but the arithmetic stays quadratic",
        "Yes: it caches keys, so each query-key product is computed only once and reused for later tokens",
      ],
      answer: 1,
      explanation: "FlashAttention is exact, so it still scores every causal query-key pair. What it removes is the stored n-squared score matrix and the traffic to main memory. Only a sliding window changes how many pairs each query scores, which is what bends the curve. Caching keys is a separate idea, the KV cache.",
      objective: 2,
    },
    {
      prompt: "In RoPE rotation you press Shift both +16. Both vectors turn, but the Score stays the same. Why?",
      options: [
        "The dot product depends only on the position difference, so shifting both positions cancels out",
        "The score is computed from the unrotated vectors, so position never reaches it",
        "Rotations change the vectors' lengths in equal amounts, so the score is rescaled back to the same value",
      ],
      answer: 0,
      explanation: "Each pair is rotated by position times its frequency, and a rotation preserves length. The dot product of a rotated query and a rotated key then contains the angle difference, which is the position difference times the frequency. Moving both by 16 leaves that difference unchanged, so the Score is unchanged.",
      objective: 3,
    },
    {
      prompt: "RoPE makes the score depend on relative distance. Does that let a model trained on 8K tokens handle 128K at inference?",
      options: [
        "Yes: because the score depends only on the distance between positions, any length behaves the same way",
        "No: RoPE keeps a table of absolute position vectors, so every position beyond 8K has no vector at all",
        "No: unseen distances still degrade attention unless the frequencies are rescaled and the model is retrained",
      ],
      answer: 2,
      explanation: "RoPE has no table of positions, but the angles for distances beyond the training range were never seen, and the model's attention degrades there. Context extension methods such as position interpolation, NTK-aware scaling and YaRN rescale the frequencies, usually with some further training. Relative encoding is necessary for extension, not sufficient.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateModernAttentionState,
};

export default definition;
