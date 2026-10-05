import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Causal attention work": {
    title: "Which query-key products each pass computes",
    summary:
      "The causal attention score matrix for a short sequence, with rows as queries and columns as keys. Prefill fills the whole lower triangle in one pass. With the cache, each decode step adds one row; without it, each step computes the whole triangle again. The counts are exact for this toy sequence.",
    whatYouSee: [
      "A square grid with one row and one column per position. Prompt positions read `t1`, `t2`…; generated ones read `+1`, `+2`… in the accent colour, split from the prompt by a dashed line. Cells above the diagonal are faint outlines: the causal mask forbids a query from seeing later keys.",
      "Solid accent cells are computed in the pass being shown, and the outlined row is the newest position. Pale cells were computed by an earlier pass; their scores were used and discarded, since only keys and values are stored. Dashed red cells are the same products computed again.",
      "Two rows under the triangle, `K` and `V`, one slot per position: solid blue is written this pass, pale blue is read from the cache, dashed red is recomputed and then dropped.",
      "Readouts: `New Q, K, V` (positions projected this pass), `Q·K this pass`, `K/V read`, and `Q·K whole run`, the products summed over prefill and every decode step so far.",
    ],
    howItWorks: [
      "Prefill over P prompt tokens computes P(P+1)/2 products and writes P key/value slots. Decode step g feeds token +g at position n = P + g: with the cache it projects one position and computes n products against the stored keys; without it, it projects all n positions and computes n(n+1)/2.",
      "Past rows are still needed without a cache, because each layer's outputs for every position feed the next layer's keys and values. The waste is that those numbers are identical to the ones the previous pass produced.",
      "Counts are per attention head per layer. Llama 3 8B repeats them across 32 query heads and 32 layers.",
    ],
    controls: [
      "`Generated tokens` 0 shows the prefill pass; each higher value shows the next decode step.",
      "`KV cache` flips between `Use cache` and `Recompute`. At 7 prompt and 4 generated tokens, `Q·K this pass` goes from 11 to 66 and `Q·K whole run` from 66 to 230.",
    ],
    notice: [
      "With the cache, the whole run costs exactly one triangle: every product is computed once. Without it, the run costs a triangle per step, 2,145 products instead of 300 at 14 prompt and 10 generated tokens.",
      "Nothing in the `K` and `V` rows is ever rewritten. Each step appends one slot, which is why the cache is valid.",
      "The cache does not change prefill: `Use cache` and `Recompute` draw the same triangle at `Generated tokens` 0.",
    ],
    limits: [
      "In this lab: the sequence tops out at 24 positions and the grid shows one head in one layer. The counts are exact for that toy, but no attention scores are computed, so the cells carry no values.",
      "In general: real kernels such as FlashAttention compute these products in tiles without storing the score matrix, and sliding-window or sparse attention layers skip parts of the triangle altogether.",
    ],
  },

  "KV cache controls": {
    title: "The toy sequence and the caching choice",
    summary:
      "Three inputs for the triangle: how many prompt tokens prefill handles in one pass, how many tokens have been generated since, and whether keys and values are kept or recomputed. The real-scale cards below have their own controls.",
    whatYouSee: [
      "`KV cache`, a two-way switch between `Use cache` and `Recompute`.",
      "`Prompt tokens` from 3 to 14 and `Generated tokens` from 0 to 10; at 0 the readout says `prefill`.",
      "A note stating what the shown pass reads and projects.",
    ],
    howItWorks: [
      "`Prompt tokens` and `Generated tokens` set the sequence length n = P + G on the triangle and the cache rows.",
      "Prefill also produces the first new token, from the last prompt row. So `Generated tokens` 1 already shows the first decode step, feeding +1 and predicting +2.",
    ],
    controls: [
      "Set `Prompt tokens` to 14 and `Generated tokens` to 1: the step projects 1 position and reads 14 stored keys and values.",
      "Raise `Generated tokens` to 10 under `Recompute` and watch `Q·K whole run` grow to 2,145, then switch to `Use cache` for 300.",
    ],
    notice: [
      "The cached per-step cost grows by one product per step, linearly with length. The recomputed per-step cost grows with the square of length.",
      "`Use cache` changes nothing about the prompt pass. It pays off from the first decode step onward.",
    ],
    limits: [
      "In this lab: the switch selects which cells are drawn and counted; there is no model running, and the toy stops at 24 positions.",
      "In general: real servers never recompute; the practical choices are how much cache to keep, at what precision, and for how many sequences at once.",
    ],
  },

  "Memory budget": {
    title: "Weights versus cache on one accelerator",
    summary:
      "Real arithmetic for a named model: Llama 3 8B in bf16 on an 80 GB accelerator. The weights take a fixed 16.1 GB. The cache takes 131,072 bytes per token per sequence, so it grows with both context length and the number of sequences served at once.",
    whatYouSee: [
      "`Context length`, stepping by powers of two from 512 to 131,072 tokens, and `Concurrent sequences` from 1 to 64.",
      "A bar split into weights (solid) and KV cache (hatched), with a marker at 80 GB. If the two exceed it, the cache segment turns red and runs past the marker.",
      "`KV per token`, `KV cache total`, `Memory left` or `Over by`, and `Sequences that fit` at this context length.",
    ],
    howItWorks: [
      "KV bytes per token = 2 × layers × KV heads × head dim × bytes per value = 2 × 32 × 8 × 128 × 2 = 131,072 bytes, which is 128 KiB. The leading 2 counts keys and values.",
      "Weights are 8.03B parameters × 2 bytes = 16.1 GB. The cache total is sequences × context × 131,072 bytes, and `Sequences that fit` is (80 GB − weights) ÷ (context × 131,072).",
      "Llama 3 8B uses grouped-query attention: 32 query heads share 8 key/value heads. With one key/value head per query head the cache would be four times larger, 524 kB per token.",
    ],
    controls: [
      "At the default 8,192 tokens and 16 sequences the cache is 17.2 GB, more than the 16.1 GB of weights.",
      "Push `Context length` to 131,072 with one sequence: that single conversation holds 17.2 GB, and only 3 fit.",
      "Raise `Concurrent sequences` to 64 at 8,192 tokens and the bar runs past 80 GB.",
    ],
    notice: [
      "The weights never change with load; the cache is the part that scales with users and context. That is why serving capacity is usually limited by cache memory, not by weights.",
      "Your toy sequence of a couple of dozen tokens needs only a few megabytes. The problem only appears at real context lengths.",
    ],
    limits: [
      "In this lab: `Sequences that fit` ignores activations, the runtime's own memory, and fragmentation, so real capacity is lower. The 80 GB is nominal capacity.",
      "In general: servers shrink this cost with paged allocation, prefix sharing, cache quantization to 8 or 4 bits, sliding-window layers, and latent-attention schemes that store a compressed representation.",
    ],
  },

  "Prefill versus decode": {
    title: "Arithmetic time against memory time",
    summary:
      "A roofline estimate of three passes at the context length and sequence count from Memory budget. For each pass, the arithmetic time is FLOPs ÷ peak FLOP/s and the memory time is bytes ÷ bandwidth; the larger one sets the modeled time. It is a model built from spec-sheet peaks, not a measurement.",
    whatYouSee: [
      "Three panels: prefill of one prompt, one cached decode step for all sequences, and one recompute decode step. Each has an `arithmetic` bar and a `memory` bar on a log time axis, with the binding one drawn solid and a `compute-bound` or `memory-bound` tag.",
      "Under each panel: FLOPs, bytes moved, arithmetic intensity in FLOP per byte, and for cached decode the total and per-sequence tokens per second.",
    ],
    howItWorks: [
      "Peaks are the H100 SXM spec sheet: 989 TFLOP/s dense bf16 and 3.35 TB/s. Their ratio, about 295 FLOP per byte, is the ridge: below it a pass waits on memory.",
      "Matmul FLOPs are 2 × 7.5B per token (all weights except the embedding lookup). Attention adds 4 × layers × heads × head dim × context per token. Prefill of T tokens reads the weights once and writes T cache slots.",
      "A cached decode step reads the weights once for the whole batch plus every sequence's cache. A recompute step reruns each sequence's prefix, which is a full prefill per new token.",
    ],
    controls: [
      "At the defaults, prefill of 8,192 tokens is compute-bound at 142 ms, and a cached decode step for 16 sequences is memory-bound at 9.6 ms, about 104 tokens per second per sequence.",
      "Set `Concurrent sequences` to 1: the decode step drops to 4.8 ms and 208 tokens per second, nearly all of it spent reading 15 GB of weights.",
    ],
    notice: [
      "Decode's arithmetic bar is tiny next to its memory bar: at one sequence the intensity is about 1.2 FLOP per byte, far below 295.",
      "Batching amortizes the weight read but not the cache reads, since each sequence reads its own cache. At long contexts the cache reads dominate the step.",
      "Recompute turns a memory-bound step into a compute-bound one: 2.27 s per step at the defaults, against 9.6 ms with the cache.",
    ],
    limits: [
      "In this lab: the model assumes 100% of peak throughput and ignores activations, kernel launch overhead, sampling, and communication, so real times are longer. At very short prompts prefill itself becomes memory-bound, which the model reproduces.",
      "In general: production engines split prefill and decode across steps or machines (chunked prefill, disaggregated serving) because the two phases want different hardware balance.",
    ],
  },

  "Speculative decoding": {
    title: "Cheap guesses, checked in one target pass",
    summary:
      "A weak draft model proposes `Draft length` characters. The target scores all of them in one pass and keeps a proposal when a random draw falls under p ÷ q; the first one it refuses is replaced by a sample from the probability the target had left over, and a pass that keeps everything adds one more token of its own. The card reports tokens per pass from the SDK's closed form and from a seeded simulation on toy distributions.",
    whatYouSee: [
      "`Draft length` from 1 to 8 proposals per pass, `Draft quality` from 0.20 to 1.00, and `Seed` from 1 to 999.",
      "Six pass lines written as text: what the draft proposed, how many the target kept and which characters, which one it refused and what it wrote instead (or the extra character it wrote when it kept all of them), and the tokens that pass added. Spaces are drawn as ␣.",
      "The text those six passes wrote after the prompt `the␣`.",
      "Four readouts: `Acceptance rate α`, `Closed form, tokens per pass`, `Simulated, tokens per pass` with its standard error, and `Without drafting`, which is always 1. A note gives the simulation minus the closed form in tokens and in standard errors.",
      "A chart of tokens per verification pass against draft length 1 to 8: the closed form solid, the simulation dashed, no drafting dotted. The legend repeats each last value.",
    ],
    howItWorks: [
      "The target is the lab's character bigram, trained 50 epochs on Harbor weather notes and Recipe steps, with the marker characters removed. The draft raises every target row to the power `Draft quality` and renormalizes, which flattens it: 1.00 is the target itself and 0.20 is close to guessing.",
      "A drafted token x after context c is kept when a uniform draw is at most p(x|c) ÷ q(x|c). The first refusal is replaced by a draw from max(0, p − q), renormalized. If all of them are kept, the target draws one more token from its own row. This is Algorithm 1 of Leviathan, Kalman and Matias (2023).",
      "α is the sum over tokens of min(p, q), averaged over the contexts the target visits. The closed form is the SDK's `speculativeTokensPerPass`, (1 − α^(γ+1)) ÷ (1 − α), the formula `Deployment & serving` also states.",
      "The simulation runs 4,000 passes for each draft length from the prompt, each pass continuing from the last token written. The standard error is the spread of tokens per pass divided by √4,000.",
    ],
    controls: [
      "At the defaults, length 4, quality 0.70 and seed 7, α is 0.831, the closed form 3.57 and the simulation 3.60 ± 0.02. Set `Draft quality` to 1.00 and every pass yields 5, the draft length plus one; set it to 0.30 and the closed form falls to 2.21.",
      "Raise `Draft length` at the default quality: the closed form reads 1.83, 3.57, 3.97 and 4.79 for lengths 1, 4, 5 and 8, and can never pass 1 ÷ (1 − α) = 5.91.",
      "`Seed` re-runs every simulation. The simulated value moves by about a standard error and the closed form does not move.",
    ],
    notice: [
      "A pass always yields at least one token and never more than the draft length plus one.",
      "From length 1 to 4 the gain is larger than from 5 to 8: a later proposal counts only if every earlier one was kept.",
      "The text written is distributed as the target's own, whatever the draft quality, because a refused proposal is replaced from the probability the target had left over. A poor draft costs tokens per pass, not correctness.",
      "`Deployment & serving` states the same formula, 3.36 tokens per pass at α = 0.8 and γ = 4.",
    ],
    limits: [
      "In this lab: the target is also a character bigram, so checking proposals in one pass is a table lookup, not a transformer forward pass. The card counts tokens per pass and never times anything.",
      "In this lab: the draft is the target with flattened rows, a stand-in for a weaker model. The closed form assumes each proposal is kept independently with the same chance; here the chance depends on the context, and the formula is off from the exact expectation by at most about 0.02 tokens across the controls.",
      "In general: a real draft is a small model or extra prediction heads, acceptance varies with the text, and a pass costs the target's step plus the draft's γ steps. Tokens per pass is only half of the speed-up.",
    ],
  },
};

export default cardInfo;
