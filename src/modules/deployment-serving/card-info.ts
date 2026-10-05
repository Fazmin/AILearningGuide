import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Serving controls": {
    title: "Model, format, load, and the scheduler",
    summary:
      "The five knobs both charts describe: which model shape, which weight format, how many concurrent streams, how long a context each stream reserves, and whether streams share a decode step.",
    whatYouSee: [
      "`Model`: `1.5B`, `3B`, `8B`, or `2B · this app`. `Weight format`: `F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`, or `Q4_0`; for `2B · this app` only `F16` and `Q4_K_M`, the file it ships.",
      "`Concurrent users` from 1 to 64 streams, and `Context length` from 512 to 32,768 tokens in steps of 512.",
      "`Scheduler`: `Continuous batching` or `One at a time`.",
      "A note with the selected model's shape and the format's bits per weight and where that figure comes from.",
    ],
    howItWorks: [
      "The presets use the published shapes of Qwen2.5-1.5B (28 layers, 2 KV heads), Qwen2.5-3B (36 layers, 2 KV heads), and Llama-3.1-8B (32 layers, 8 KV heads), all with 128-dimensional heads.",
      "Bits per weight: F16 is 16. The quantized formats use llama.cpp's reported file sizes for Llama-3-8B divided by its 8.03 B parameters: Q8_0 8.52, Q5_K_M 5.70, Q4_K_M 4.90, Q4_0 4.64. That rate is applied to the three Qwen2.5 and Llama presets; `2B · this app` uses its own file's rate.",
      "`2B · this app` is Qwen3.5 2B, the model Settings downloads: 1,881,825,088 parameters and a 1,280,835,840-byte file, 5.445 bits per weight. It has 24 layers but only every fourth is full attention (6 layers, 2 key/value heads of 256 dimensions), so its cache is 2 × 6 × 2 × 256 × 2 bytes = 12 KiB a token. The other 18 are Gated DeltaNet layers with a fixed state per stream: (128 × 2,048 + 3 × (2,048 + 2 × 16 × 128)) × 4 bytes × 18 layers = 20,201,472 bytes, 19.27 MiB.",
      "Sources, read 30 September 2026: the GGUF header of the pinned file (layers, heads, head dimension, state sizes, tensor table), the Qwen/Qwen3.5-2B model card (layer layout), and llama.cpp b10991 (`llama-hparams.cpp` for the state size and `llama-model.cpp` for float32 recurrent memory).",
      "Shared state from a link is clamped to these ranges.",
    ],
    controls: [
      "Every control is live on both charts; `Scheduler` does not change memory.",
      "Comparison worth running: lock 8B and Q4_K_M at 4,096 tokens and raise users until Streams that fit is exceeded at 37.",
    ],
    notice: [
      "Changing the format moves only the weights. Users and context move only the KV cache.",
      "Context here is the length reserved per stream, the worst case, not the tokens generated so far.",
    ],
    limits: [
      "In this lab: this card and the two after it use one fixed hardware profile (the next card changes it), no second GPU, no CPU offload, and a user count rather than an arrival process.",
      "In general: a format is a three-way trade of memory, decode speed, and quality. This page prices the first two; the Quantization lab measures the third.",
    ],
  },

  "Memory budget": {
    title: "Weights are fixed; the KV cache scales with load",
    summary:
      "A stacked bar of the configuration's memory against a 24 GiB limit: weights, a KV cache divided into one slice per stream, and a flat 1 GiB for activations and runtime buffers. Streams that fit is how many full-context streams the leftover memory can hold.",
    whatYouSee: [
      "A bar on a GiB axis with segments for weights, KV cache, and activations + runtime, thin dividers between streams inside the KV segment, a heavy line at the 24 GiB limit, and a dashed outline on any overflow.",
      "A legend with each segment's size, and metrics `Per token, per stream`, `Per stream at this context`, `Streams that fit`, and `Headroom`.",
      "A callout that names the headroom or the overflow.",
    ],
    howItWorks: [
      "Weight bytes = parameters × bits per weight / 8.",
      "KV bytes per token = 2 × layers × KV heads × head dim × 2 bytes (keys and values, fp16): 128 KiB for the 8B shape, 36 KiB for 3B, 28 KiB for 1.5B. Times context times users is the cache.",
      "Streams that fit = floor((24 GiB − weights − 1 GiB) / (KV bytes per token × context)).",
      "For `2B · this app` only the 6 full-attention layers are charged per token, and a fixed 19.27 MiB of recurrent state is added per stream, shown as `Fixed state per stream` and drawn in the same segment as the cache.",
    ],
    controls: [
      "`Model`, `Weight format`, `Concurrent users`, and `Context length` redraw the bar.",
      "Comparison worth running: at 8B Q4_K_M with one user, raise Context length from 4,096 to 32,768. The cache grows from 0.50 to 4.00 GiB and Streams that fit falls from 36 to 4.",
    ],
    notice: [
      "At 8 users and 4,096 tokens the cache (4.00 GiB) is already close to the weights (4.58 GiB).",
      "The 8B shape has 8 KV heads and the 3B shape has 2, so the bigger model pays 3.6 times more cache per token.",
    ],
    limits: [
      "In this lab: every byte is arithmetic. The activation term is flat, the cache is fp16, and each stream reserves its full context up front.",
      "In general: paged KV allocation, cache quantization, sliding windows, and prefix sharing all shrink the cache segment in production servers.",
    ],
  },

  "Throughput and latency": {
    title: "Total tokens rise; each user waits longer",
    summary:
      "Two panels against concurrency from one formula: a decode step reads every weight once plus every stream's cache, then emits one token per stream. The shaded region is where the KV cache no longer fits in memory.",
    whatYouSee: [
      "Top panel: total tokens per second across all streams. Bottom panel: tokens per second each user sees. Both have a marker at the current user count.",
      "A shaded region past the last stream count that fits, labelled with the limit.",
      "Metrics `Total`, `Per user`, `Time per output token`, and `Time to first token`, and a paragraph of stated assumptions.",
    ],
    howItWorks: [
      "Batched: step time = max((weights + all caches) / 400 GB/s, users / compute ceiling). Total = users / step; per user = 1 / step; time per output token = step.",
      "One at a time: each step serves one stream, so total = 1 / ((weights + one cache) / 400 GB/s), per user = total / users, and each user waits users × step between tokens.",
      "Compute ceiling = 120 TFLOP/s / (2 × parameters): 7,472 tokens/s for 8B. Time to first token = max(2 × parameters × context / 120 TFLOP/s, one weight read), a prompt that fills the context.",
    ],
    controls: [
      "All five serving controls redraw the panels.",
      "Comparison worth running: at 8B Q4_K_M and 4,096 tokens, 1 user gets 73.3 tokens/s; 8 users get 347 in total and 43.4 each. Switch to One at a time: total falls to 73.3 and each user gets 9.2.",
    ],
    notice: [
      "Per-user speed barely falls while weights dominate each read, then approaches 1 / users as the caches come to dominate. It never falls faster than 1 / users.",
      "Q8_0 and F16 slow decode because each step reads more bytes, even though they may be more accurate.",
    ],
    limits: [
      "In this lab: closed-form arithmetic at 100% of peak bandwidth and compute, with attention FLOPs and prefill interference ignored.",
      "In general: decode is bandwidth bound and prefill compute bound. Speculative decoding, CUDA graphs, and prefix caching move real numbers (the next card adds speculation); load tests with real traffic measure tail latency, which averages hide.",
    ],
  },

  "Hardware, levers, and cost": {
    title: "Other hardware, three levers, and what a million tokens costs",
    summary:
      "The same model, format, load, and scheduler as the cards above, on hardware you choose and with paged attention, a quantized KV cache, and speculative decoding switched on or off. It reports how many streams fit, how fast each decodes, and the cost per million tokens from an hourly price you can edit. Every number is closed-form arithmetic.",
    whatYouSee: [
      "`Hardware` presets (`Lab default`, `RTX 4090`, `H100 SXM`, `M4 Max laptop`, `CPU box`) that set four editable inputs, `Memory`, `Memory bandwidth`, `Compute`, and `Price`, plus `Busy share of each hour`. No preset is pressed once you edit a number, and the note under them gives each figure's source and date.",
      "Switches for `Paged attention` and `Speculative decoding`; `KV cache type` (`fp16`, `q8_0`, `q4_0`, with bytes per value); sliders for `Average fill of the reserved context`, `Acceptance rate` (α), `Draft length` (γ), and `Draft cost` (c).",
      "Metrics `Streams that fit`, `Per user`, `Total`, and `Cost per million tokens` for your selection, and a bar list with `No levers`, each lever alone, and `Your selection`: bar length is total tokens per second, the label is the cost, and the detail line gives streams that fit and speeds.",
    ],
    howItWorks: [
      "Memory = weights + users × (KV bytes per stream + state) + 1 GiB. KV bytes per stream = 2 × cache layers × KV heads × head dimension × bytes per value × tokens held. Tokens held is the whole context, or with paging ceil(fill × context / 16) × 16.",
      "A decode step takes max((weights + every stream's cache and state) / bandwidth, streams / compute ceiling), where the ceiling is compute / (2 × parameters). Total and per-user speed follow as on the Throughput card.",
      "A speculative pass takes max(memory time, streams × (γ + 1) / ceiling) + γ × c × one target step, and yields (1 − α^(γ+1)) / (1 − α) tokens per stream. While decode is bandwidth bound that is Leviathan et al.'s walltime improvement (1 − α^(γ+1)) / ((1 − α)(γc + 1)): 2.40 at α = 0.8, γ = 4, c = 0.1.",
      "Cost per million tokens = price an hour / (total tokens per second × 3,600 × busy share) × 1,000,000, for output tokens only. It reads as over memory when the load does not fit.",
      "Hardware, read 30 September 2026. RTX 4090: NVIDIA Ada GPU Architecture whitepaper v2.02, Appendix A (24 GB GDDR6X, 1008 GB/s, 165.2 TFLOP/s FP16 tensor with FP32 accumulate, dense). H100 SXM: nvidia.com/en-us/data-center/h100 (80 GB, 3.35 TB/s, 1,979 TFLOP/s FP16 tensor with sparsity; the datasheet footnote says one half without, so 989.5). M4 Max: Apple's October 2024 newsroom release and 14-inch MacBook Pro tech specs (up to 128 GB, 546 GB/s; Apple lists no FLOP/s, so 18 is a third-party figure from flopper.io). Core i9-14900K: Intel's specification page (DDR5 5600 MT/s, two channels, 89.6 GB/s; the 1 TFLOP/s and the 64 GiB are assumptions, not figures from that page).",
      "Prices, read 30 September 2026: runpod.io/pricing, page dated 27 September 2026, lists $0.74 an hour for an RTX 4090 pod and $3.49 for an H100 SXM pod; lambda.ai/pricing lists $4.29 for one H100 SXM instance. The laptop and CPU box are priced at 0 because you own them, and the lab default's $1.00 is a placeholder.",
      "Paged attention: vLLM's default block is 16 tokens, and the PagedAttention paper (Kwon et al., 2023) profiled existing systems storing token states in only 20.4% to 38.2% of KV cache memory. Cache types: q8_0 is 34 bytes per 32 values (8.5 bits) and q4_0 is 18 bytes (4.5 bits); llama.cpp b10991's server accepts them for `--cache-type-k` and `--cache-type-v`, with f16 the default.",
    ],
    controls: [
      "`Hardware` loads a preset into the four sliders, which you can then edit; the three levers and their sliders apply to every preset. The Serving controls above still set the model, format, load, and scheduler.",
      "Comparison worth running at 8B, Q4_K_M, 4,096 tokens: with 1 user the RTX 4090 decodes at 185 tok/s and the H100 at 614, but 36 streams fit on one and 148 on the other. With 8 users the cost is $0.23 against $0.33 per million tokens at the listed prices.",
      "On the lab default, switch on `Speculative decoding`: per-user speed goes from 43.4 to 104 tok/s and the cost from $0.80 to $0.33. Then pick `CPU box` (its compute is an assumption): 8 streams are already compute bound at 62.3 tok/s, and the same switch drops them to 38.8.",
    ],
    notice: [
      "Bandwidth and capacity do different jobs. Bandwidth sets how fast one stream decodes: the H100 is 3.3 times the 4090 in bandwidth and in single-stream speed, against 6.0 times in compute. Capacity sets how many streams share each read of the weights.",
      "A lever is worth what the bottleneck allows. Paging and a quantized cache buy capacity and a smaller cache to read; speculation buys per-user speed while decode is bandwidth bound, and gives it back once verifying γ + 1 tokens per stream makes the step compute bound.",
      "`q8_0` is not half the cost of `fp16`: 8.5 bits against 16 is 0.53, because each block of 32 stores a 16-bit scale. The 2B preset's recurrent state stays float32 under every cache type.",
    ],
    limits: [
      "In this lab: every figure is arithmetic on stated inputs at 100% of peak bandwidth and compute. Paging uses the average fill, not a distribution of stream lengths; speculation assumes one acceptance rate and a draft whose cost is a fixed share of a target step; prefill is not billed; the draft model's memory is ignored.",
      "In general: engines reach a fraction of peak bandwidth, acceptance rate depends on the text and on how well the draft matches the target, hardware lists change, and prices differ by provider and tier: the same H100 SXM is $3.49 an hour at one provider and $4.29 at another, and owned hardware has no hourly price at all.",
    ],
  },

  "Export the model you trained": {
    title: "Five steps from a trained table to a served file",
    summary:
      "The export path applied to the 900-weight table from these labs at the selected format: merge, quantize, write GGUF, verify, serve with the cap you just computed. The same steps apply at 8B; only the byte counts change.",
    whatYouSee: [
      "A kicker `900 weights → FORMAT` and a badge with the toy file's size.",
      "Five steps with details: `Merge the adapter`, `Quantize to FORMAT` with its block type and bits per weight, `Write the GGUF file`, `Verify before running`, and `Serve it` with the stream cap.",
      "A closing note with the production-scale weight and cache bytes.",
    ],
    howItWorks: [
      "The toy size is 900 × the block type's exact bits per weight / 8: Q4_K_M maps to Q4_K at 4.5 (507 bytes), Q5_K_M to Q5_K at 5.5, Q8_0 at 8.5, Q4_0 at 4.5, F16 at 16 (1,800 bytes).",
      "The stream cap is the smaller of the current user count and Streams that fit.",
    ],
    controls: [
      "`Weight format` changes step 2, step 3, and the badge; `Concurrent users` and `Context length` change the cap in step 5.",
      "Comparison worth running: Q4_K_M against F16 on the badge, then read the production weight bytes in the closing note.",
    ],
    notice: [
      "The toy uses block rates, not file-mix rates: with one tensor there is nothing to keep at a wider type.",
      "Capping concurrency is what turns an out-of-memory crash into a queue.",
    ],
    limits: [
      "In this lab: merge, write, verify, and serve are named, not executed, and no rounding is measured here. The Quantization lab measures it.",
      "In general: a production export also freezes the tokenizer, chat template, and license, and the engine checks the hash before the first token.",
    ],
  },
};

export default cardInfo;
