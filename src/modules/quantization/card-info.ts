import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Quantization controls": {
    title: "Bit width, block size, and where the grid sits",
    summary:
      "The three decisions a weight quantizer makes: how many integer codes each weight gets, how many weights share one grid, and whether that grid is centred on zero (symmetric) or stretched over the block's own range (asymmetric). Every other card re-quantizes the same trained table when these change.",
    whatYouSee: [
      "`Bit width` from 2 to 8, printed with the number of levels: 2^b for asymmetric grids, 2^b − 1 for symmetric ones.",
      "`Scaling granularity`: `Per tensor` (one grid for all 900 weights), `Per row` (30 grids, one per context character), or `Per group`.",
      "`Group size` from 2 to 30, shown only in per-group mode.",
      "`Range mapping`: `Asymmetric · min/max` or `Symmetric · absmax`.",
      "Three metrics: `Bits per weight` including metadata, `Model size` in bytes, and `Against 32-bit` as a share of 900 × 4 bytes.",
    ],
    howItWorks: [
      "Asymmetric: scale = (max − min) / (2^b − 1), code = round((w − min) / scale), w′ = min + code × scale. The block stores an fp16 scale and an fp16 min: 32 bits.",
      "Symmetric: scale = max|w| / (2^(b−1) − 1), code = round(w / scale) clamped to ±(2^(b−1) − 1), w′ = code × scale. The block stores one fp16 scale: 16 bits.",
      "Bits per weight = b + blocks × metadata bits / 900. Model size is that times 900 / 8, rounded up.",
    ],
    controls: [
      "`Bit width`, `Scaling granularity`, `Group size`, and `Range mapping`.",
      "Comparison worth running: at 4 bits, read Bits per weight for Per tensor (4.04), Per row (5.07), and Per group of 8 (8.02). Then set Group size to 2: 20 bits per weight, more than F16.",
    ],
    notice: [
      "Symmetric metadata is half the size of asymmetric, which is why the same granularity costs fewer bits per weight on the symmetric setting.",
      "A group of 2 stores two weights exactly with an asymmetric grid, because each pair's min and max are its two weights. The error is zero and the storage is absurd.",
    ],
    limits: [
      "In this lab: the scales are kept at full float precision while being charged 16 bits each, and bits are assumed to pack with no padding.",
      "In general: real formats pick block sizes that divide a row (32 or 256 weights in llama.cpp) and quantize the scales themselves to shrink this overhead.",
    ],
  },

  "Rounding grid and per-weight error": {
    title: "The grid each weight snaps to, and what it cost",
    summary:
      "Top: the 900 trained weights as a histogram, with one block's rounding grid drawn over it. Bottom: every weight's rounding error w′ − w, plotted at its original value. Under per-tensor scaling the errors trace a sawtooth that is the grid itself; under finer scaling each block has its own, smaller teeth.",
    whatYouSee: [
      "A histogram of the original weights on a square-root count axis, so the few large logits stay visible beside the pile near zero.",
      "Vertical lines at the inspected block's levels. Solid lines are levels at least one of the block's weights landed on; dashed lines are levels nobody used. Above 64 levels the lines are too dense and a note says so.",
      "A shaded band covering the inspected block's grid, and a marker at the table's largest weight (the logit for “.” followed by a space, 5.93).",
      "An error strip: grey dots for every weight, accent dots for the inspected block, and dashed lines at ± half of that block's step.",
      "`Inspect block` when there is more than one block, and metrics `RMS rounding error`, `Largest error`, `Step in this block`, and `Levels used in this block`.",
    ],
    howItWorks: [
      "The weights are a 30 × 30 character bigram trained for 80 epochs on Harbor weather notes at seed 1, perplexity 6.600. Nothing is retrained after rounding.",
      "For a weight inside its block's grid, the error is at most half a step. That is why the dots stay between the dashed lines for the inspected block.",
      "`Inspect block` starts on the block that contains the largest weight. `Levels used` counts distinct codes in that block.",
    ],
    controls: [
      "`Inspect block` picks which block's grid is drawn; all quantization dials live on the controls card.",
      "Comparison worth running: at 4 bits per row, the “.” row uses 2 of 16 levels because one 5.93 logit stretches its step to 0.41; drag Inspect block to another row and the step shrinks.",
    ],
    notice: [
      "Switch Range mapping to Symmetric at 4 bits per tensor: the grid spans −5.93 to +5.93, the table only reaches −1.71, and 10 of 15 levels are used.",
      "The sawtooth is exact, not noise: per tensor, every weight with the same value gets the same error.",
    ],
    limits: [
      "In this lab: one small weight matrix, weights only. Activations and the KV cache are not quantized, and the histogram is binned for display.",
      "In general: rounding error is not quality loss. The next card measures perplexity, and production recipes minimize output error on calibration data rather than weight error.",
    ],
  },

  "Quality against bits per weight": {
    title: "Measured quality cost against the real storage bill",
    summary:
      "Each point re-quantizes the trained table at one bit width and granularity, re-measures perplexity on the harbor corpus, and plots the percent increase over full precision against bits per weight including scales. The llama.cpp block types from the table below are plotted as diamonds.",
    whatYouSee: [
      "A log-scale y axis of perplexity increase, from ≤0.01% to 1000%. Points at or below 0.01% (including tiny negative changes) sit on the floor as hollow markers.",
      "Three series for the current range mapping: `per tensor` (circles, solid), `per row` (squares, dashed), `per group of G` (triangles, dotted), each at 2, 3, 4, 5, 6, and 8 bits.",
      "Diamonds for Q8_0, Q6_K, Q5_K, Q4_K, Q4_1, Q4_0, Q3_K, and Q2_K, and a ring on the current setting.",
      "Metrics `Full precision`, `Quantized`, and `Quality cost`, and two samples at temperature 0.7 from the same seed.",
    ],
    howItWorks: [
      "Cost = (quantized perplexity / 6.600 − 1) × 100%. The x position is the same bits per weight the controls card reports.",
      "Points beyond 17 bits per weight (small asymmetric groups) are left off the axis and counted in the caption.",
      "Both samples use prompt “the ”, 76 characters, seed 5, so a different string means a different next-character table.",
    ],
    controls: [
      "`Bit width` moves the ring; `Scaling granularity` moves it between series; `Group size` and `Range mapping` redraw every series.",
      "Comparison worth running: find 4-bit per tensor (4.04 bpw, +0.94%) and 4-bit per row (5.07 bpw, +0.17%), then the Q4_K diamond at 4.5 bpw and +0.19%.",
    ],
    notice: [
      "Down and to the left is better. The Q4_K diamond sits left of the per-row series at the same cost, because its quantized scales cost 0.5 bits per weight instead of 1.07.",
      "Per-tensor cost grows about fivefold or more for each bit removed below 5 (6.6×, 5.3×, then 4.8×): +0.14%, +0.94%, +5.0%, +23.9%.",
    ],
    limits: [
      "In this lab: perplexity is measured on the training corpus of a 900-weight bigram, so it measures reconstruction of this table, not held-out quality.",
      "In general: quality should be checked on held-out tasks, and low-bit formats can hold perplexity while losing ground on long contexts, code, or rare languages.",
    ],
  },

  "Published quantization formats": {
    title: "The block layouts behind GGUF names",
    summary:
      "Nine llama.cpp block types applied to the same table in storage order: the exact bits per weight from each block's struct, the perplexity and rounding error measured here, and llama.cpp's own file size for Llama-3-8B where a file mix is built on that block.",
    whatYouSee: [
      "Columns `Block type` (with the byte layout and a one-line description), `Bits per weight` (with a bar out of 16), `Toy perplexity` (with its percent cost), `RMS error`, and `llama.cpp file, Llama-3-8B`.",
      "Rows whose code width matches the Bit width slider are highlighted.",
      "A note decoding a file name like Q4_K_M as a mix of block types.",
    ],
    howItWorks: [
      "Q8_0: 32 int8 codes plus an fp16 scale, 34 bytes, 8.5 bpw. Q4_0: 32 4-bit codes plus an fp16 scale, 4.5 bpw. Q4_1 adds an fp16 min, 5.0 bpw.",
      "K-quants use 256-weight super-blocks whose sub-block scales are quantized: Q4_K and Q5_K store 6-bit scales and mins for eight 32-weight sub-blocks (4.5 and 5.5 bpw); Q6_K and Q3_K store signed scales for sixteen 16-weight sub-blocks (6.5625 and 3.4375 bpw); Q2_K stores 4-bit scales and mins (2.625 bpw).",
      "The emulation follows those layouts and rounds every stored scale to fp16, but skips llama.cpp's search for the best scale, so its errors are an upper bound.",
      "The file column converts llama.cpp's reported GiB for Llama-3-8B, whose mixes keep some tensors in a larger type.",
    ],
    controls: [
      "This card has no controls. `Bit width` only moves the highlight.",
      "Comparison worth running: Q4_0, Q4_1, and Q4_K all use 4-bit codes. Read their bits per weight (4.5, 5.0, 4.5) against their RMS error (0.127, 0.063, 0.062).",
    ],
    notice: [
      "Q4_0 is symmetric and pays for it on this lopsided table: double the RMS error of Q4_1 at the same code width.",
      "Q4_K matches Q4_1's error while spending half a bit less, which is the whole point of quantizing the scales.",
    ],
    limits: [
      "In this lab: llama.cpp quantizes each row separately and needs rows that are a multiple of the block size; this 30-wide table is quantized in flattened order instead, and the _S / _M / _L mixes need more than one tensor.",
      "In general: the name is a contract about storage, not quality. Quality still has to be measured on the task and model you care about.",
    ],
  },

  "The file you downloaded": {
    title: "What a Q4_K_M file really costs per weight",
    summary:
      "The model file this app downloads, weighed the way the rest of the lab weighs a table: its bytes over its parameters give effective bits per weight, set against the nominal Q4_K rate and against F16. The bars show which tensors carry the extra bits.",
    whatYouSee: [
      "Metrics `File size` (1.19 GiB), `Parameters in the file` (1.88 B), `Effective bits per weight` (5.445), and `Share of the F16 size` (34.0%).",
      "`Group tensors by`: `Block type` (Q6_K, Q4_K, Q5_K, Q8_0, F32) or `Role in the network` (embedding, feed-forward, Gated DeltaNet, full attention, norms).",
      "One bar per group for its share of the file's tensor bytes, with its tensor count, its share of the parameters, and its own bits per weight. Groups above 5 bits per weight are drawn in the loss colour.",
      "A note with the exact byte and parameter counts, the gap to Q4_K, the header's cost, and the calibration the file's metadata records.",
    ],
    howItWorks: [
      "Effective bits per weight = 8 × file bytes / parameters = 8 × 1,280,835,840 / 1,881,825,088 = 5.445. A group's figure is 8 × its bytes / its parameters, and each block type reproduces its layout rate: Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625, Q8_0 8.5, F32 32.",
      "The tensor bytes sum to 1,269,873,920; the header (metadata, the 248,320-token vocabulary, the chat template) is the other 10,961,920, which is 0.047 bits per weight.",
      "Q4_K_M is a recipe across tensors: the token embedding (27.0% of the parameters, tied to the output) is Q6_K at 6.5625; feed-forward layers average 4.84, Gated DeltaNet layers 5.34, full-attention layers 4.60.",
      "Source and date: the counts were read on 30 September 2026 from the GGUF header of the pinned revision (unsloth/Qwen3.5-2B-GGUF at f6d5376, the file Settings downloads); Hugging Face's API reports the same 1,881,825,088 parameters.",
    ],
    controls: [
      "`Group tensors by` switches the bars between the two groupings; the metrics do not change.",
      "Comparison worth running: by `Block type`, Q6_K holds 35.3% of the parameters but 42.9% of the bytes. By `Role in the network`, the token embedding costs 6.56 bits per weight while the feed-forward layers cost 4.84.",
    ],
    notice: [
      "The file is 0.945 bits per weight above a Q4_K block, 21.0% more bytes than a file stored entirely as Q4_K, because about 51% of the parameters are held at Q5_K, Q6_K, or wider. The header explains under 1% of the file.",
      "This 2B model averages 5.445 where llama.cpp's Llama-3-8B Q4_K_M file averages 4.90: the embedding table is a much larger share of a small model, so a recipe's average depends on the model it is applied to.",
    ],
    limits: [
      "In this lab: the numbers are one file, authored from its header on 30 September 2026. Nothing here is downloaded or re-measured when you open the card, and the quality of the file is not measured, only its size.",
      "In general: effective bits per weight is a storage figure. It ignores the runtime cost of the KV cache and recurrent state, and says nothing about quality, which depends on the model, the recipe, and the calibration text.",
    ],
  },

  "Float formats": {
    title: "From fp32 to fp8: range from exponent bits, precision from mantissa bits",
    summary:
      "Five floating-point formats laid out as sign, exponent, and mantissa bits, with their range and spacing computed from those layouts, one value you choose rounded into each of them, and the trained 900-weight table rounded through each to see what it costs.",
    whatYouSee: [
      "`Example value` (0.001, 0.02, 300, 100,000) and `Value to store`, a log slider from 1e-6 to 1e6 that sets the number rounded in every row.",
      "Per format: sign, exponent, and mantissa bits and the exponent bias; the smallest normal and largest finite value; the spacing at 1.0 as 2 to the minus the mantissa bits, with the worst relative rounding error; what your value is stored as and its percent error, or `overflow`.",
      "Toy columns: RMS rounding error of the 900 trained weights through each format, and the perplexity against the full-precision 6.600.",
    ],
    howItWorks: [
      "Largest finite value = (2 − 2^(−mantissa bits)) × 2^(top exponent − bias): 65,504 for fp16, 448 for E4M3, 57,344 for E5M2. E4M3 gives its top exponent to numbers and keeps one mantissa pattern for not-a-number; the others reserve that exponent for infinity and not-a-number.",
      "Rounding finds the value's binade, takes the spacing 2^(exponent − mantissa bits), and rounds to the nearest multiple with ties to even. Below the smallest normal value the spacing stops shrinking, so values go subnormal and then to zero; a result past the largest finite value overflows.",
      "The toy columns round each float32 weight this way and re-measure perplexity on the harbor corpus. No per-tensor scale is applied.",
    ],
    controls: [
      "`Example value` jumps to a preset; `Value to store` moves the number by a twentieth of a decade, about 12%, per step.",
      "Comparison worth running: at 100,000 only bf16 and fp32 hold the value; fp16 stops at 65,504 and the fp8 formats at 57,344 and 448. At 0.02, fp16 is off by 0.021%, bf16 by 0.098%, and both fp8 formats by 2.3%.",
    ],
    notice: [
      "bf16 and fp16 are both 16 bits. bf16 keeps fp32's eight exponent bits and so its range, and is eight times coarser than fp16 at 1.0; on this table fp16's RMS error is more than eight times smaller than bf16's.",
      "Eight bits per weight with no scale and no blocks costs +0.17% (E4M3) or +0.36% (E5M2) perplexity here, while Q8_0 spends 8.5 bits for a change below 0.01%: integer blocks with a scale spend their bits on precision, float formats on range.",
    ],
    limits: [
      "In this lab: rounding is applied to a stored weight table with no per-tensor scale and no accumulation, and perplexity is measured on the corpus the table was trained on.",
      "In general: formats are used differently. Training keeps fp32 or bf16 accumulators and often scales fp8 tensors so their largest value lands near the format's maximum; overflow handling (saturate, infinity, or not-a-number) depends on the conversion mode.",
    ],
  },
};

export default cardInfo;
