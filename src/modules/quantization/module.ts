import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {
  bits: 4,
  scope: "group",
  groupSize: 8,
  zeroPoint: "asymmetric",
  /** Block drawn on the grid plot. -1 follows the block that holds the largest weight. */
  inspect: -1,
  /** How "The file you downloaded" sums its tensors: by block type or by role in the network. */
  fileGroup: "type",
  /** The number "Float formats" stores in each format, clamped to FLOAT_VALUE_RANGE. */
  floatValue: 0.02,
};

const SCOPES = ["tensor", "row", "group"];
const FILE_GROUPS = ["type", "role"];
/** Positive values only: the float card rounds a magnitude, and a log slider spans these six decades each way. */
const FLOAT_VALUE_RANGE = { min: 1e-6, max: 1e6 };

/**
 * Version 1 had no zeroPoint or inspect, and version 2 had no fileGroup or floatValue; each defaults, and every
 * field is clamped to its control.
 */
export function hydrateQuantizationState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const number = (key: string, fallback: number) =>
      typeof parsed[key] === "number" && Number.isFinite(parsed[key]) ? (parsed[key] as number) : fallback;
    return {
      ...initialState,
      bits: Math.min(8, Math.max(2, Math.round(number("bits", 4)))),
      scope: typeof parsed.scope === "string" && SCOPES.includes(parsed.scope) ? parsed.scope : initialState.scope,
      groupSize: Math.min(30, Math.max(2, Math.round(number("groupSize", 8) / 2) * 2)),
      zeroPoint: parsed.zeroPoint === "symmetric" ? "symmetric" : "asymmetric",
      inspect: Math.max(-1, Math.round(number("inspect", -1))),
      fileGroup:
        typeof parsed.fileGroup === "string" && FILE_GROUPS.includes(parsed.fileGroup) ? parsed.fileGroup : initialState.fileGroup,
      floatValue: Math.min(FLOAT_VALUE_RANGE.max, Math.max(FLOAT_VALUE_RANGE.min, number("floatValue", 0.02))),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-26-quantization",
  slug: "quantization",
  title: "Quantization",
  group: "training-adapting",
  order: 48,
  icon: "Ruler",
  accent: "#8d6a2c",
  prerequisites: ["module-11-inference-kv-cache", "module-20-train-tiny-lm"],
  estimatedMinutes: 18,
  steps: [
    "Round the weights",
    "Change the granularity",
    "Move the zero point",
    "Price the tradeoff",
    "Read the format names",
    "Weigh the file you downloaded",
    "Climb the float ladder",
  ],
  stepInstructions: [
    "Set Scaling granularity to Per tensor, then pull Bit width from 8 down to 2 and watch the grid lines spread apart and the error sawtooth grow.",
    "At 4 bits, switch Scaling granularity between Per tensor, Per row, and Per group, compare RMS rounding error, and use Inspect block to compare the block that holds the largest weight with any other block.",
    "At 4 bits Per tensor, switch Range mapping from Asymmetric to Symmetric and read Levels used in this block and Quality cost.",
    "On the Quality against bits per weight chart, find the per-tensor, per-row, and per-group points at 4 bits and compare how far right and how far down each one sits.",
    "In the Published quantization formats table, compare Q4_0, Q4_1, and Q4_K: same 4-bit codes, different metadata, different bits per weight and error.",
    "In The file you downloaded, read Effective bits per weight, then switch Group tensors by from Block type to Role in the network and compare the bits per weight of the token embedding with the feed-forward layers.",
    "In Float formats, set Example value to 100,000 and compare which formats overflow, then set it to 0.02 and compare the Error column for fp16 and bf16.",
  ],
  stateVersion: 3,
  tagline:
    "Round a trained weight table onto integer grids, see every weight's rounding error, price the scales in bits per weight, decode the block layouts behind GGUF names, weigh the model file this app downloads, and climb the float formats from fp32 to fp8.",
  objectives: [
    "Explain what a scale and a zero point do, and when a symmetric grid wastes levels",
    "Trade bit width against scaling granularity in real bits per weight",
    "Read a GGUF block type or file name and predict its size and rough quality",
    "Compare fp32, fp16, bf16, and fp8 by their exponent and mantissa bits, and say whether a value runs into range or precision",
  ],
  glossary: [
    {
      term: "Quantization",
      definition:
        "Storing each weight as a small integer code plus shared per-block metadata that turns codes back into numbers. The weight keeps its meaning and loses resolution.",
    },
    {
      term: "Scale",
      definition:
        "The value one integer step represents inside a block. The largest possible rounding error for a weight inside the grid is half a step, so a wider block range means a bigger step and more error.",
    },
    {
      term: "Zero point",
      definition:
        "The offset that places the integer grid on the number line. An asymmetric grid stores it so the grid can cover a lopsided range; llama.cpp stores it as a float minimum rather than an integer.",
    },
    {
      term: "Symmetric quantization",
      definition:
        "A grid centred on zero with scale = max|w| / (2^(b−1) − 1) and no stored offset. Cheaper metadata and simpler kernels, and it wastes levels on whichever side of zero the block does not use.",
    },
    {
      term: "Per-tensor scaling",
      definition:
        "One grid for an entire matrix. Cheapest to store and the most exposed to a single outlier weight stretching the range.",
    },
    {
      term: "Group scaling",
      definition:
        "One grid per small block of consecutive weights, typically 32 in llama.cpp. Per-row (per-channel) scaling is the same idea with one block per row.",
    },
    {
      term: "Bits per weight",
      definition:
        "The real storage cost: code bits plus the block's scale and offset bits spread over its weights. Q4_0 stores 4-bit codes and one 16-bit scale per 32 weights, so it costs 4.5; a whole file's bytes over its parameters give the effective figure, higher whenever some tensors stay wider.",
    },
    {
      term: "Outlier",
      definition:
        "A weight or activation far outside its block's typical range. It sets the block's scale on its own, so every other weight in the block gets coarser steps.",
    },
    {
      term: "GGUF",
      definition:
        "The single-file format llama.cpp loads: tensors in named block types, quantization metadata, and the tokenizer in one file with a verifiable header.",
    },
    {
      term: "K-quant",
      definition:
        "llama.cpp's block family with 256-weight super-blocks whose sub-block scales are themselves quantized to 4, 6, or 8 bits. That shrinks the metadata to about half a bit per weight. The _S, _M, and _L suffixes are file mixes that keep some tensors in a larger type.",
    },
    {
      term: "Post-training quantization",
      definition:
        "Quantizing an already-trained model, as here. Calibration-based variants adjust weights using sample data; quantization-aware training instead exposes the rounding during training.",
    },
    {
      term: "Floating-point format",
      definition:
        "A bit layout of one sign bit, exponent bits that set the range, and mantissa bits that set the precision: fp32 is 1+8+23, fp16 is 1+5+10, bf16 is 1+8+7, and the fp8 formats E4M3 and E5M2 are 1+4+3 and 1+5+2. Fewer exponent bits mean a narrower range; fewer mantissa bits mean coarser spacing.",
    },
    {
      term: "Calibration",
      definition:
        "Running sample text through a model to see which weights or activation channels matter most, then using that to pick scales or protect channels when quantizing. llama.cpp's importance matrix, GPTQ, and AWQ all calibrate; plain rounding does not.",
    },
  ],
  references: [
    {
      authors: "Benoit Jacob, Skirmantas Kligys, Bo Chen, et al.",
      title: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
      source: "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018), 2704–2713",
      year: 2018,
      url: "https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html",
      note: "Sets out the affine mapping from integer codes to real numbers through a scale and a zero point, the same pair the Asymmetric grid stores. It also trains with simulated rounding, the quantization-aware training named at the end of the lesson.",
    },
    {
      authors: "Markus Nagel, Marios Fournarakis, Rana Ali Amjad, et al.",
      title: "A White Paper on Neural Network Quantization",
      source: "arXiv preprint arXiv:2106.08295",
      year: 2021,
      url: "https://arxiv.org/abs/2106.08295",
      note: "A clear guide to asymmetric and symmetric grids and to per-tensor, per-channel, and per-group scaling, the three Scaling granularity choices in the lab. It also compares post-training quantization with quantization-aware training.",
    },
    {
      authors: "Hugging Face",
      title: "GGUF",
      source: "Hugging Face Hub documentation",
      year: 2026,
      url: "https://huggingface.co/docs/hub/gguf",
      note: "Has a table of GGUF block types with the formula each one decodes by: Q4_0 as scale times code, Q4_1 with a block minimum added, and Q4_K as 32-weight blocks in super-blocks at 4.5 bits per weight. These are the rows of Published quantization formats.",
    },
    {
      authors: "ggml-org (llama.cpp contributors)",
      title: "llama.cpp quantize tool README",
      source: "llama.cpp documentation, release b10991",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/b10991/tools/quantize/README.md",
      note: "Shows how llama-quantize turns a model into a file mix such as Q4_K_M, with options to keep some tensors wider or to use an importance matrix. Its tables list bits per weight and file size for each type on Llama 3.1 8B, the source of the 4.58 GiB Q4_K_M figure.",
    },
    {
      authors: "ggml-org (ggml contributors)",
      title: "GGUF file format specification",
      source: "ggml documentation",
      year: 2026,
      url: "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md",
      note: "Describes GGUF as a single file holding the header, key-value metadata, tensor descriptions, and tensor data. It also explains the file naming convention, which backs the GGUF glossary entry and the card The file you downloaded.",
    },
    {
      authors: "ggml-org (llama.cpp contributors)",
      title: "llama.cpp imatrix tool README",
      source: "llama.cpp documentation, release b10991",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/b10991/tools/imatrix/README.md",
      note: "Explains how llama-imatrix runs a text file through a model to build an importance matrix that quantization can then use to improve quality. This is the calibration step behind the app's own downloaded file.",
    },
    {
      authors: "IEEE Computer Society",
      title: "IEEE Standard for Floating-Point Arithmetic (IEEE Std 754-2019)",
      source: "IEEE Standards Association",
      year: 2019,
      url: "https://standards.ieee.org/ieee/754/6210/",
      note: "The standard that defines binary floating-point formats, including the 32-bit and 16-bit ones at the top of the float ladder. The lesson notes that the fp8 format E5M2 follows its rules for special values.",
    },
    {
      authors: "Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, et al.",
      title: "A Study of BFLOAT16 for Deep Learning Training",
      source: "arXiv preprint arXiv:1905.12322",
      year: 2019,
      url: "https://arxiv.org/abs/1905.12322",
      note: "Explains why bf16 keeps the same range as fp32 and why that lets training run without the tuning fp16 needs. It backs the Float formats card's point that bf16 holds 100,000 while fp16 overflows.",
    },
    {
      authors: "Paulius Micikevicius, Dusan Stosic, Neil Burgess, et al.",
      title: "FP8 Formats for Deep Learning",
      source: "arXiv preprint arXiv:2209.05433",
      year: 2022,
      url: "https://arxiv.org/abs/2209.05433",
      note: "Proposes the two fp8 formats in the lab, E4M3 and E5M2. It explains how E4M3 gives up infinities and most NaN codes to reach a larger maximum, and it tests fp8 for both training and post-training quantization.",
    },
    {
      authors: "Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer",
      title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2208.07339",
      note: "Finds outlier feature dimensions in large transformers and sends them through a 16-bit multiply while more than 99.9% of values use 8 bits. It backs the lesson's point that activation outliers make real models harder than this table.",
    },
    {
      authors: "Guangxuan Xiao, Ji Lin, Mickael Seznec, et al.",
      title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 38087–38099",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/xiao23c.html",
      note: "Moves the difficulty of activation outliers into the weights with an equivalent rescaling, so both weights and activations can be stored in 8 bits. It is the second outlier method named under Where it breaks.",
    },
    {
      authors: "Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh",
      title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2210.17323",
      note: "A one-shot weight quantizer that uses approximate second-order information and reaches 3 or 4 bits per weight on 175-billion-parameter models. It is one of the calibration methods the lesson compares with plain rounding.",
    },
    {
      authors: "Ji Lin, Jiaming Tang, Haotian Tang, et al.",
      title: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration",
      source: "Proceedings of Machine Learning and Systems 6 (MLSys 2024), 87–100",
      year: 2024,
      url: "https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html",
      note: "Shows that protecting about 1% of salient weight channels, chosen from activation statistics, greatly reduces rounding error. It protects them by scaling, with no backpropagation, as the Calibration methods section describes.",
    },
    {
      authors: "ggml-org (llama.cpp contributors)",
      title: "llama.cpp server README",
      source: "llama.cpp documentation, release b10991",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/b10991/tools/server/README.md",
      note: "Lists the --cache-type-k and --cache-type-v options, which store the KV cache as q8_0, q4_0, or other types instead of the f16 default. It backs the Quantizing the cache section.",
    },
    {
      authors: "Zirui Liu, Jiayi Yuan, Hongye Jin, et al.",
      title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 32332–32344",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/liu24bz.html",
      note: "Studies how keys and values are spread out and finds keys should be grouped per channel and values per token. Its 2-bit cache cuts peak memory 2.6 times, the KIVI result quoted under Quantizing the cache.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At 4 bits per tensor you switch Range mapping from Asymmetric to Symmetric, and Levels used in this block drops below the number of levels. Why?",
      options: [
        "A symmetric grid stores a zero point as well as a scale, and that stored offset takes one level away from the codes",
        "Symmetric quantization rounds most of the weights to exactly zero, so the other levels rarely come up at all",
        "The grid is centred on zero and spans the largest magnitude, but the weights lean one way, so levels go unused",
        "The scale becomes a float minimum instead of a step size, which merges neighbouring levels into a single code",
      ],
      answer: 2,
      explanation:
        "A symmetric grid is centred on zero and its step comes from the largest magnitude, so it spreads levels evenly across both signs. This table reaches much further on the positive side than the negative, so the negative levels go unused. An asymmetric grid stretches over the block's own minimum and maximum and wastes none, at the price of storing an offset.",
      objective: 0,
    },
    {
      prompt:
        "At 4 bits per row, one row holds a single weight of 5.93 beside many small ones. What does that outlier do to the row's grid?",
      options: [
        "It is clipped to the largest level, so the smaller weights are quantized exactly as they would be without it",
        "It sets the row's scale alone, so the step is wide and the smaller weights crowd into a few levels",
        "It shifts the row's zero point, so every smaller weight picks up an error equal to the outlier's size",
        "It forces the row onto per-tensor scaling, so the other rows share its step as well",
      ],
      answer: 1,
      explanation:
        "A block's step is its range divided by the number of levels. One large weight stretches the range, so the step grows and the rest of the row crowds into a few of its 16 levels. Here the other weights in the row happen to be identical and are rounded exactly, but with varied weights the coarse step would blur them. Smaller blocks confine the stretch: per-group scaling lets only the outlier's own group pay for it.",
      objective: 0,
    },
    {
      prompt: "At 4 bits you move Scaling granularity from Per tensor to Per group of 8. Which readouts move, and in which directions?",
      options: [
        "Neither changes, because the code width is still 4 bits and that width alone sets the storage cost per weight",
        "Both fall, because smaller groups need fewer levels to cover their narrower range of weights and offsets",
        "RMS rounding error rises while Bits per weight falls, because each group holds fewer weights to be stored",
        "RMS rounding error falls while Bits per weight rises, because every group stores its own scale and offset",
      ],
      answer: 3,
      explanation:
        "Smaller blocks track the local range, so the step and the rounding error shrink. The price is metadata: each asymmetric block stores two 16-bit numbers, now spread over fewer weights, so bits per weight rises. The code width is never the whole storage cost, which is why formats settle on blocks of about 32 rather than one scale per weight.",
      objective: 1,
    },
    {
      prompt: "Q4_0, Q4_1 and Q4_K all store 4-bit codes. What does the Published quantization formats table show about them?",
      options: [
        "Q4_K matches Q4_1's error at half a bit less by quantizing its scales, while Q4_0 pays for its symmetric grid",
        "Their bits per weight are identical, because the codes are identical and only the file names differ",
        "Q4_0 has the lowest error, because it is the oldest and most heavily tested layout of the three",
        "Q4_K is the largest of the three, because its 256-weight super-blocks carry the most metadata per weight",
      ],
      answer: 0,
      explanation:
        "Bits per weight is the code width plus the metadata spread over each block: 4.5 for Q4_0, 5.0 for Q4_1 and 4.5 for Q4_K. Q4_K stores its sub-block scales in a few bits against one super-block scale, so it keeps Q4_1's accuracy for half a bit less, while Q4_0's symmetric grid wastes levels on this lopsided table.",
      objective: 2,
    },
    {
      prompt: "The lab shows only a small perplexity cost for a 4-bit format. Why is that a weak reason to ship it?",
      options: [
        "Perplexity on the training text only measures how well this table was reconstructed; held-out tasks can still lose ground",
        "Perplexity always falls when weights are rounded, so a small reading means the rounding did not really run",
        "The quantized table is retrained before perplexity is read, so the reading says nothing about the rounding",
        "A name such as Q4_K_M already fixes a format's quality, so measuring anything further is redundant",
      ],
      answer: 0,
      explanation:
        "Perplexity here is computed on the corpus the table was trained on, so it reports reconstruction, not held-out quality. In real systems a format can hold perplexity on prose and still lose ground on long contexts, code, or rare languages. Nothing is retrained after rounding, and a file name is a contract about storage, not quality.",
      objective: 2,
    },
    {
      prompt:
        "The Q4_K_M file this app downloads averages well above the 4.5 bits per weight of a Q4_K block. What accounts for most of the gap?",
      options: [
        "The tokenizer and chat template in the header add roughly one extra bit to every weight, which lifts the whole average",
        "Many tensors stay at Q5_K or Q6_K, above all the large embedding table, so the average sits above the Q4_K rate",
        "Every weight keeps its own fp16 scale beside its 4-bit code, which is how each Q4 format stores its numbers",
        "The importance matrix is saved with the weights and counted in the file, which doubles the per-block metadata",
      ],
      answer: 1,
      explanation:
        "A file name such as Q4_K_M is a recipe across tensors: mostly Q4_K, with some tensors kept at Q5_K or Q6_K. Here the 248,320-token embedding table, 27% of the parameters, is kept at Q6_K. The header is under 1% of the file, and the importance matrix steers the rounding without being stored in the weights.",
      objective: 2,
    },
    {
      prompt:
        "You set Value to store to 100,000. Which formats can still hold it, and what does that say about their bits?",
      options: [
        "fp16 and bf16 can, because both are 16 bits wide and so reach the same range of values",
        "Only fp32 can, because every 16-bit format has too few mantissa bits to write a number that large",
        "fp16 and E5M2 can, because five exponent bits cover any weight or activation a model produces",
        "bf16 and fp32 can, because both give eight bits to the exponent; fp16 and the fp8 formats run out of range",
      ],
      answer: 3,
      explanation:
        "Range comes from the exponent bits. bf16 keeps fp32's eight, so 100,000 fits and is stored a little coarsely. fp16 tops out at 65,504, E5M2 at 57,344, and E4M3 at 448, so all three overflow. Mantissa bits set how finely a value that fits is rounded, not how large it can be.",
      objective: 3,
    },
    {
      prompt:
        "bf16 and fp16 both take 16 bits. Rounding the trained table through each, which stays closer to the original weights, and why?",
      options: [
        "bf16, because its eight exponent bits give it finer spacing between neighbouring values near 1.0, as in fp32",
        "fp16, because it spends ten bits on the mantissa to bf16's seven, and this table never needs bf16's range",
        "They tie, because both store 16 bits and rounding error depends only on that total width",
        "bf16, because it matches fp32's range and a table inside that range is rounded without any error",
      ],
      answer: 1,
      explanation:
        "Spacing near 1.0 is two to the minus the mantissa bits: about 0.001 for fp16 and 0.008 for bf16, so the fp16 table lands more than eight times closer in RMS error. bf16 spends its extra bits on range, which pays off for activations and gradients that can be huge but buys nothing for weights between about −1.7 and 5.9.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateQuantizationState,
};

export default definition;
