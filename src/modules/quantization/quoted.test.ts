import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { TINY_CORPORA, TINY_VOCAB, TINY_VOCAB_SIZE, tinyPerplexity, trainTinyModel } from "@app/module-sdk";
import { hydrateQuantizationState } from "./module";
import {
  bitsPerWeightOf,
  codesUsed,
  DOWNLOADED_FILE,
  errorStats,
  fileBitsPerWeight,
  fileEffectiveBitsPerWeight,
  fileSlices,
  FLOAT_FORMATS,
  floatFormatStats,
  GGUF_BLOCK_FORMATS,
  quantizeWeights,
  roundTableToFloatFormat,
  roundToFloatFormat,
  toHalf,
  type QuantScope,
  type ZeroPoint,
} from "./quant";

/** The embedding table is vocabulary x hidden size parameters: the lab note and card quote 248,320 tokens. */
const FILE_VOCABULARY = 248_320;

/**
 * Pins the measured values the quantization lesson (content/standard.mdx, content/plain.mdx), card-info.ts and the
 * Explore lab notes quote. Mirrors Explore.tsx: the table is `trainTinyModel` on the harbor corpus for 80 epochs at
 * seed 1, the sweeps use `quantizeWeights` with `rowLength: TINY_VOCAB_SIZE` and the module's default group size,
 * and "quality cost" is (quantized perplexity / full-precision perplexity - 1) x 100.
 */
const text = TINY_CORPORA.harbor.text;
const defaults = hydrateQuantizationState("{}");
const base = trainTinyModel({ text, epochs: 80, seed: 1 });
const weights = base.weights;
const basePerplexity = tinyPerplexity(weights, text);
const cost = (rounded: Float32Array) => (tinyPerplexity(rounded, text) / basePerplexity - 1) * 100;
const quantize = (bits: number, scope: QuantScope, zeroPoint: ZeroPoint = "asymmetric", groupSize = Number(defaults.groupSize)) =>
  quantizeWeights(weights, { bits, scope, groupSize, zeroPoint, rowLength: TINY_VOCAB_SIZE });
const format = (name: string) => GGUF_BLOCK_FORMATS.find((entry) => entry.name === name)!;
const round = (value: number, digits: number) => Number(value.toFixed(digits));

let largestIndex = 0;
for (let k = 1; k < weights.length; k += 1) if (Math.abs(weights[k]) > Math.abs(weights[largestIndex])) largestIndex = k;
const blockHolding = (result: ReturnType<typeof quantize>) =>
  result.blocks.find((block) => largestIndex >= block.start && largestIndex < block.end)!;

describe("quantization quoted values", () => {
  it("matches the trained table and the defaults the lesson assumes", () => {
    // standard.mdx / plain.mdx "What it is": 900 weights, 80 epochs, perplexity 6.600.
    expect(weights.length).toBe(900);
    expect(basePerplexity).toBeCloseTo(6.6, 3);
    // "groups of 8" and "At 4 bits" are the lab defaults (module.ts initialState).
    expect(defaults).toMatchObject({ bits: 4, scope: "group", groupSize: 8, zeroPoint: "asymmetric" });
    // Explore.tsx symmetric note: "lopsided (-1.71 to +5.93)"; standard.mdx: largest logit 5.93 is "." followed by a space.
    expect(round(Math.min(...weights), 2)).toBe(-1.71);
    expect(round(Math.max(...weights), 2)).toBe(5.93);
    expect(weights[largestIndex]).toBe(Math.max(...weights));
    expect([TINY_VOCAB[Math.floor(largestIndex / TINY_VOCAB_SIZE)], TINY_VOCAB[largestIndex % TINY_VOCAB_SIZE]]).toEqual([".", " "]);
  });

  it("matches the per-tensor bit-width sweep (RMS error and quality cost)", () => {
    // standard.mdx and plain.mdx "How to play with it": RMS error 0.008, 0.074, 0.147, 0.353, 0.967 at 8, 5, 4, 3, 2 bits;
    // Quality cost +0.01%, +0.14%, +0.94%, +5.03%, +23.90%. card-info.ts repeats +0.14%, +0.94%, +5.0%, +23.9%.
    const rows = [8, 5, 4, 3, 2].map((bits) => {
      const result = quantize(bits, "tensor");
      return [bits, round(result.rmsError, 3), round(cost(result.weights), 2)];
    });
    expect(rows).toEqual([
      [8, 0.008, 0.01],
      [5, 0.074, 0.14],
      [4, 0.147, 0.94],
      [3, 0.353, 5.03],
      [2, 0.967, 23.9],
    ]);
  });

  it("matches the growth in per-tensor cost for each bit removed below 5", () => {
    // quantization/card-info.ts "Quality against bits per weight" notice: "about fivefold or more for each bit removed
    // below 5 (6.6×, 5.3×, then 4.8×)". The last step is just under fivefold, which is why the wording is "about".
    const costs = Object.fromEntries([5, 4, 3, 2].map((bits) => [bits, cost(quantize(bits, "tensor").weights)]));
    expect(round(costs[4] / costs[5], 1)).toBe(6.6);
    expect(round(costs[3] / costs[4], 1)).toBe(5.3);
    expect(round(costs[2] / costs[3], 1)).toBe(4.8);
  });

  it("matches the 4-bit granularity comparison (error, bits per weight, cost)", () => {
    // standard.mdx / plain.mdx: RMS error 0.147 per tensor, 0.031 per row, 0.027 for groups of 8.
    // Bits per weight 4.04, 5.07, 8.02 (also card-info.ts "Quantization controls"); card-info.ts "Quality against bits per weight": per row +0.17%.
    const tensor = quantize(4, "tensor");
    const row = quantize(4, "row");
    const group = quantize(4, "group");
    expect([tensor, row, group].map((result) => round(result.rmsError, 3))).toEqual([0.147, 0.031, 0.027]);
    expect([tensor, row, group].map((result) => round(result.bitsPerWeight, 2))).toEqual([4.04, 5.07, 8.02]);
    expect(tensor.blocks.length).toBe(1);
    expect(row.blocks.length).toBe(30);
    expect(round(cost(row.weights), 2)).toBe(0.17);
    // plain.mdx: "every ruler costs 32 bits to store" (asymmetric); card-info.ts: symmetric stores one fp16 scale, 16 bits.
    expect(tensor.metadataBitsPerBlock).toBe(32);
    expect(quantize(4, "tensor", "symmetric").metadataBitsPerBlock).toBe(16);
    // plain.mdx: "4 bits gives 16" ticks.
    expect(tensor.levels).toBe(16);
  });

  it("matches the group-of-2 extreme", () => {
    // standard.mdx / plain.mdx / card-info.ts: a group of 2 costs 20 bits per weight at 4 bits, more than F16 (16),
    // and "stores every pair exactly" (error is zero).
    const pairs = quantize(4, "group", "asymmetric", 2);
    expect(pairs.bitsPerWeight).toBe(20);
    expect(pairs.bitsPerWeight).toBeGreaterThan(format("F16").bitsPerWeight);
    expect(pairs.rmsError).toBeLessThan(1e-6);
    // Explore.tsx note: "At a group of 2 the scales cost at least as much as the codes" (16 metadata bits per weight against the widest 8-bit code).
    expect(pairs.metadataBitsPerBlock / 2).toBeGreaterThanOrEqual(8);
  });

  it("matches the outlier row at 4 bits per row", () => {
    // standard.mdx / plain.mdx / card-info.ts: the 5.93 logit stretches its row's 4-bit step to 0.41, and that row uses 2 of its 16 levels.
    const row = quantize(4, "row");
    const outlierBlock = blockHolding(row);
    expect(round(outlierBlock.scale, 2)).toBe(0.41);
    expect(codesUsed(row, outlierBlock)).toBe(2);
    expect(row.levels).toBe(16);
  });

  it("matches the symmetric grid at 4 bits per tensor", () => {
    // standard.mdx / plain.mdx / card-info.ts: grid spans -5.93 to +5.93, table only reaches -1.71, 10 of 15 levels used, cost +0.94% -> +2.33%.
    const symmetric = quantize(4, "tensor", "symmetric");
    const block = symmetric.blocks[0];
    expect(round(block.scale * block.maxCode, 2)).toBe(5.93);
    expect(round(block.scale * block.minCode, 2)).toBe(-5.93);
    expect(codesUsed(symmetric, block)).toBe(10);
    expect(symmetric.levels).toBe(15);
    expect(round(cost(quantize(4, "tensor").weights), 2)).toBe(0.94);
    expect(round(cost(symmetric.weights), 2)).toBe(2.33);
  });

  it("matches the 2-bit symmetric per-tensor perplexity", () => {
    // standard.mdx "How to play with it": three levels (-5.93, 0, +5.93) leave perplexity at 22.55.
    const symmetric = quantize(2, "tensor", "symmetric");
    expect(symmetric.levels).toBe(3);
    expect(round(symmetric.blocks[0].scale, 2)).toBe(5.93);
    expect(tinyPerplexity(symmetric.weights, text)).toBeCloseTo(22.55, 2);
  });

  it("matches the Published quantization formats table", () => {
    // standard.mdx / plain.mdx / card-info.ts: Q4_0, Q4_1, Q4_K at 4.5, 5.0, 4.5 bits per weight;
    // RMS error 0.127, 0.063, 0.062; cost +0.20%, +0.16%, +0.19%.
    const names = ["Q4_0", "Q4_1", "Q4_K"];
    const rows = names.map((name) => {
      const rounded = format(name).quantize(weights);
      return [name, format(name).bitsPerWeight, round(errorStats(weights, rounded).rmsError, 3), round(cost(rounded), 2)];
    });
    expect(rows).toEqual([
      ["Q4_0", 4.5, 0.127, 0.2],
      ["Q4_1", 5, 0.063, 0.16],
      ["Q4_K", 4.5, 0.062, 0.19],
    ]);
    // "Q4_0 is symmetric and has double the RMS error of Q4_1 here."
    const rms = (name: string) => errorStats(weights, format(name).quantize(weights)).rmsError;
    expect(rms("Q4_0") / rms("Q4_1")).toBeGreaterThan(1.9);
    expect(rms("Q4_0") / rms("Q4_1")).toBeLessThan(2.1);
    // "Q4_K matches Q4_1's error while spending half a bit less."
    expect(round(rms("Q4_K"), 3)).toBeLessThanOrEqual(round(rms("Q4_1"), 3));
    expect(format("Q4_1").bitsPerWeight - format("Q4_K").bitsPerWeight).toBe(0.5);
  });

  it("matches the K-quant metadata story and the bits-per-weight layout arithmetic", () => {
    // standard.mdx "Reading the names": Q4_K is 144 bytes per 256 weights, 4.5 bits per weight; Q4_0 is also 4.5.
    expect((144 * 8) / 256).toBe(4.5);
    expect(format("Q4_K").bitsPerWeight).toBe(4.5);
    expect(format("Q4_0").bitsPerWeight).toBe(4.5);
    // glossary "Bits per weight": Q4_0 stores 4-bit codes and one 16-bit scale per 32 weights, so it costs 4.5.
    expect(4 + 16 / 32).toBe(4.5);
    // glossary "K-quant" / standard.mdx / plain.mdx: scales cost "about half a bit per weight" (Q4_K) "instead of one" (Q4_1).
    expect(format("Q4_K").bitsPerWeight - format("Q4_K").codeBits).toBe(0.5);
    expect(format("Q4_1").bitsPerWeight - format("Q4_1").codeBits).toBe(1);
    // standard.mdx / card-info.ts: per-row scales cost 1.07 bits per weight, against the Q4_K 0.5.
    expect(round(quantize(4, "row").bitsPerWeight - 4, 2)).toBe(1.07);
    // card-info.ts "Published quantization formats": Q8_0 8.5 (34 bytes), Q5_K 5.5, Q6_K 6.5625, Q3_K 3.4375, Q2_K 2.625.
    expect((34 * 8) / 32).toBe(8.5);
    expect(Object.fromEntries(GGUF_BLOCK_FORMATS.map((entry) => [entry.name, entry.bitsPerWeight]))).toMatchObject({
      F16: 16,
      Q8_0: 8.5,
      Q6_K: 6.5625,
      Q5_K: 5.5,
      Q4_K: 4.5,
      Q4_1: 5,
      Q4_0: 4.5,
      Q3_K: 3.4375,
      Q2_K: 2.625,
    });
    // card-info.ts: "Nine llama.cpp block types", plotted as diamonds for all but F16 in this order.
    expect(GGUF_BLOCK_FORMATS).toHaveLength(9);
    expect(GGUF_BLOCK_FORMATS.filter((entry) => entry.name !== "F16").map((entry) => entry.name)).toEqual([
      "Q8_0",
      "Q6_K",
      "Q5_K",
      "Q4_K",
      "Q4_1",
      "Q4_0",
      "Q3_K",
      "Q2_K",
    ]);
  });

  it("converts llama.cpp's Q4_K_M Llama-3-8B file size to bits per weight", () => {
    // standard.mdx "Reading the names", plain.mdx "A common mix-up", Explore.tsx note: 4.58 GiB, about 4.90 bits per weight, not 4.5.
    // The 4.58 GiB file size is llama.cpp's own published figure for Llama-3-8B (external source); only the conversion is pinned here.
    expect(fileBitsPerWeight(4.58)).toBeCloseTo(4.9, 2);
  });

  it("puts Q4_K left of the per-row series at about the same cost", () => {
    // standard.mdx / plain.mdx / card-info.ts "Quality against bits per weight": Q4_K at 4.5 bpw and +0.19%,
    // per-row 4-bit at 5.07 bpw and +0.17%.
    const row = quantize(4, "row");
    const q4k = format("Q4_K");
    expect(q4k.bitsPerWeight).toBeLessThan(row.bitsPerWeight);
    expect(round(cost(q4k.quantize(weights)), 2)).toBe(0.19);
    expect(Math.abs(cost(q4k.quantize(weights)) - cost(row.weights))).toBeLessThan(0.05);
  });

  it("keeps every error within half a step on the real table", () => {
    // standard.mdx "Two ways to place the grid", card-info.ts and glossary "Scale": for a weight inside its grid the error w' - w is at most half a step.
    for (const [scope, zeroPoint] of [["tensor", "asymmetric"], ["row", "asymmetric"], ["group", "asymmetric"], ["tensor", "symmetric"]] as const) {
      const result = quantize(4, scope, zeroPoint);
      for (const block of result.blocks) {
        for (let index = block.start; index < block.end; index += 1) {
          expect(Math.abs(result.weights[index] - weights[index])).toBeLessThanOrEqual(block.scale / 2 + 1e-6);
        }
      }
    }
  });

  it("makes 2 bits look free on a fresh or barely trained table", () => {
    // standard.mdx "What comes after rounding": "on a fresh table, 2 bits would look free" (which is why the lab pretrains for 80 epochs);
    // plain.mdx: "A barely trained model hardly cares about rounding". Per tensor at 2 bits, a fresh table (0 epochs) is
    // untouched and a 1-epoch table does not get worse, while the 80-epoch table costs +23.90%.
    for (const epochs of [0, 1]) {
      const fresh = trainTinyModel({ text, epochs, seed: 1 }).weights;
      const rounded = quantizeWeights(fresh, { bits: 2, scope: "tensor", zeroPoint: "asymmetric", rowLength: TINY_VOCAB_SIZE }).weights;
      const change = (tinyPerplexity(rounded, text) / tinyPerplexity(fresh, text) - 1) * 100;
      expect(change, `${epochs} epochs`).toBeLessThanOrEqual(0.01);
    }
    expect(cost(quantize(2, "tensor").weights)).toBeGreaterThan(20);
  });

  it("lets quality cost sit on the floor with tiny negative changes", () => {
    // card-info.ts "Quality against bits per weight": points at or below 0.01% "(including tiny negative changes)" sit on the floor.
    // Per tensor at 6 bits the measured change is slightly negative; at 8 bits it is below 0.01%.
    const sixBit = cost(quantize(6, "tensor").weights);
    expect(sixBit).toBeLessThan(0);
    expect(sixBit).toBeGreaterThan(-0.1);
    expect(cost(quantize(8, "tensor").weights)).toBeLessThan(0.01);
  });
});

/** module.ts checkpoint questions 1-5 rest on these comparisons; they pin the directions the answers claim. */
describe("quantization checkpoint claims", () => {
  it("question 1: symmetric 4-bit per tensor leaves levels unused because the table is lopsided", () => {
    const symmetric = quantize(4, "tensor", "symmetric");
    const asymmetric = quantize(4, "tensor", "asymmetric");
    expect(codesUsed(symmetric, symmetric.blocks[0])).toBeLessThan(symmetric.levels);
    expect(Math.abs(Math.min(...weights))).toBeLessThan(Math.max(...weights) / 2);
    expect(asymmetric.metadataBitsPerBlock).toBeGreaterThan(symmetric.metadataBitsPerBlock);
  });

  it("question 2: one outlier sets its row's step alone, and the rest of the row crowds into a few levels", () => {
    const row = quantize(4, "row");
    const holder = blockHolding(row);
    const rowWeights = Array.from(weights.slice(holder.start, holder.end));
    const rest = rowWeights.filter((_, index) => holder.start + index !== largestIndex);
    // The step is the row's range over 15; every other weight in the row sits within a hair of one value.
    expect(holder.scale).toBeCloseTo((Math.max(...rowWeights) - Math.min(...rowWeights)) / (row.levels - 1), 6);
    expect(Math.max(...rest) - Math.min(...rest)).toBeLessThan(0.01);
    expect(codesUsed(row, holder)).toBe(2);
  });

  it("question 3: per group of 8 lowers the error and raises the bits per weight against per tensor", () => {
    const tensor = quantize(4, "tensor");
    const group = quantize(4, "group");
    expect(group.rmsError).toBeLessThan(tensor.rmsError);
    expect(group.bitsPerWeight).toBeGreaterThan(tensor.bitsPerWeight);
  });

  it("question 4: Q4_K matches Q4_1's error at half a bit less, and symmetric Q4_0 has the highest error", () => {
    expect(["Q4_0", "Q4_1", "Q4_K"].map((name) => format(name).bitsPerWeight)).toEqual([4.5, 5.0, 4.5]);
    const rms = (name: string) => errorStats(weights, format(name).quantize(weights)).rmsError;
    expect(rms("Q4_0")).toBeGreaterThan(rms("Q4_1") * 1.5);
    expect(Math.abs(rms("Q4_K") - rms("Q4_1"))).toBeLessThan(0.005);
  });
});

/* -------------------------------------------------------------------------- */
/* The file this app downloads                                                 */
/* -------------------------------------------------------------------------- */

describe("quantization: the file you downloaded", () => {
  const readText = (path: string) => readFileSync(resolve(process.cwd(), path), "utf8");

  it("is the file the desktop app actually pins", () => {
    // The lab's constants must track what Settings downloads: src-tauri/src/local_ai.rs and LocalModelSettings.tsx.
    expect(readText("src-tauri/src/local_ai.rs")).toContain(`const MODEL_BYTES: u64 = ${DOWNLOADED_FILE.bytes.toLocaleString("en-US").replaceAll(",", "_")};`);
    expect(readText("src/components/LocalModelSettings.tsx")).toContain(`const MODEL_BYTES = ${DOWNLOADED_FILE.bytes.toLocaleString("en-US").replaceAll(",", "_")};`);
    expect(readText("src-tauri/src/local_ai.rs")).toContain(DOWNLOADED_FILE.revision);
    expect(readText("src-tauri/src/local_ai.rs")).toContain(DOWNLOADED_FILE.file);
    // standard.mdx / card-info.ts: "1,280,835,840 bytes", "1.19 GiB".
    expect(DOWNLOADED_FILE.bytes).toBe(1_280_835_840);
    expect(round(DOWNLOADED_FILE.bytes / 1024 ** 3, 2)).toBe(1.19);
  });

  it("adds up to the file: tensor rows plus the header are exactly the bytes and parameters read from the GGUF header", () => {
    // Authored from the header of the pinned revision on 2026-09-30 (provenance in quant.ts); the arithmetic is what is checked here.
    const sum = (key: "tensors" | "parameters" | "bytes") => DOWNLOADED_FILE.groups.reduce((total, group) => total + group[key], 0);
    expect(sum("tensors")).toBe(DOWNLOADED_FILE.tensorCount);
    expect(sum("tensors")).toBe(320);
    expect(sum("parameters")).toBe(1_881_825_088);
    expect(sum("parameters")).toBe(DOWNLOADED_FILE.parameters);
    expect(sum("bytes")).toBe(1_269_873_920);
    expect(sum("bytes") + DOWNLOADED_FILE.headerBytes).toBe(DOWNLOADED_FILE.bytes);
    // 24 blocks, every 4th one full attention (6), the other 18 Gated DeltaNet.
    expect(DOWNLOADED_FILE.layers).toBe(24);
    expect(DOWNLOADED_FILE.fullAttentionLayers).toBe(DOWNLOADED_FILE.layers / 4);
  });

  it("gives every tensor type exactly the block rate the Published quantization formats table lists", () => {
    // standard.mdx / card-info.ts quote Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625 and Q8_0 8.5 bits per weight for the blocks; the file's own tensors reproduce them.
    const byType = Object.fromEntries(fileSlices("type").map((slice) => [slice.key, slice]));
    for (const name of ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]) {
      expect(byType[name].bitsPerWeight, name).toBe(format(name).bitsPerWeight);
    }
    expect(byType.F32.bitsPerWeight).toBe(32);
    expect(byType.Q6_K.tensors).toBe(17);
    expect(byType.Q4_K.tensors).toBe(98);
    expect(byType.Q5_K.tensors).toBe(36);
  });

  it("computes the effective bits per weight and the comparisons the lesson quotes", () => {
    // standard.mdx / plain.mdx / card-info.ts: 1,280,835,840 bytes over 1,881,825,088 parameters is 5.445 bits per weight,
    // 0.945 above the Q4_K block's 4.5, and 34.0% of the F16 size; llama.cpp's Llama-3-8B Q4_K_M file is 4.90.
    const effective = fileEffectiveBitsPerWeight();
    expect(round(effective, 3)).toBe(5.445);
    expect(round(effective, 1)).toBe(5.4);
    expect(round(effective - format("Q4_K").bitsPerWeight, 3)).toBe(0.945);
    expect(round((effective / 16) * 100, 1)).toBe(34);
    expect(round((DOWNLOADED_FILE.parameters * 2) / 1024 ** 3, 2)).toBe(3.51);
    expect(round(fileBitsPerWeight(4.58), 2)).toBe(4.9);
    expect(effective).toBeGreaterThan(fileBitsPerWeight(4.58));
    // "more bytes than a file that stored every tensor as Q4_K": +21.0%, and that file would be 0.99 GiB.
    const allQ4K = (DOWNLOADED_FILE.parameters * format("Q4_K").bitsPerWeight) / 8;
    expect(round((DOWNLOADED_FILE.bytes / allQ4K - 1) * 100, 1)).toBe(21);
    expect(round(allQ4K / 1024 ** 3, 2)).toBe(0.99);
    // Checkpoint question 6: the gap is mostly wider tensors, not the header.
    expect(bitsPerWeightOf(DOWNLOADED_FILE.headerBytes, DOWNLOADED_FILE.parameters)).toBeLessThan(0.05);
    expect(DOWNLOADED_FILE.headerBytes / DOWNLOADED_FILE.bytes).toBeLessThan(0.01);
    expect(round(bitsPerWeightOf(DOWNLOADED_FILE.headerBytes, DOWNLOADED_FILE.parameters), 3)).toBe(0.047);
    expect(round((DOWNLOADED_FILE.headerBytes / DOWNLOADED_FILE.bytes) * 100, 2)).toBe(0.86);
    // Tensor bytes alone are 5.398 bits per weight; the header adds the other 0.047.
    expect(round(((DOWNLOADED_FILE.bytes - DOWNLOADED_FILE.headerBytes) * 8) / DOWNLOADED_FILE.parameters, 3)).toBe(5.398);
  });

  it("matches the groupings: the embedding table is 27% of the parameters at Q6_K, and the layers sit between 4.6 and 5.3", () => {
    const roles = Object.fromEntries(fileSlices("role").map((slice) => [slice.key, slice]));
    expect(round(roles.embedding.parameterShare * 100, 1)).toBe(27);
    expect(round(roles.embedding.byteShare * 100, 1)).toBe(32.9);
    expect(round(roles.embedding.bitsPerWeight, 2)).toBe(6.56);
    expect(roles.embedding.parameters).toBe(FILE_VOCABULARY * DOWNLOADED_FILE.hiddenSize);
    expect(round(roles.ffn.bitsPerWeight, 2)).toBe(4.84);
    expect(round(roles.linear.bitsPerWeight, 2)).toBe(5.34);
    expect(round(roles.attention.bitsPerWeight, 2)).toBe(4.6);
    expect(roles.norms.bitsPerWeight).toBe(32);
    // The embedding table is the biggest single tensor group by bytes after the feed-forward layers, and costs the most per weight of the big groups.
    const bigGroups = fileSlices("role").filter((slice) => slice.parameterShare > 0.1);
    expect(bigGroups.sort((a, b) => b.bitsPerWeight - a.bitsPerWeight)[0].key).toBe("embedding");
    // By type: Q6_K holds 35.3% of the parameters but 42.9% of the bytes; Q4_K 48.6% and 40.5%.
    const types = Object.fromEntries(fileSlices("type").map((slice) => [slice.key, slice]));
    expect([types.Q6_K, types.Q4_K].map((slice) => round(slice.parameterShare * 100, 1))).toEqual([35.3, 48.6]);
    expect([types.Q6_K, types.Q4_K].map((slice) => round(slice.byteShare * 100, 1))).toEqual([42.9, 40.5]);
    // card-info.ts "The file you downloaded" notice: "about 51% of the parameters are held at Q5_K, Q6_K, or wider".
    expect(round((1 - types.Q4_K.parameterShare) * 100, 0)).toBe(51);
    // The groupings are sorted by bytes, largest first.
    for (const by of ["type", "role"] as const) {
      const bytes = fileSlices(by).map((slice) => slice.bytes);
      expect([...bytes].sort((a, b) => b - a)).toEqual(bytes);
    }
    // And each grouping accounts for every parameter.
    for (const by of ["type", "role"] as const) {
      expect(fileSlices(by).reduce((total, slice) => total + slice.parameters, 0)).toBe(DOWNLOADED_FILE.parameters);
    }
  });

  it("records that the file was calibrated, as its metadata says", () => {
    // Authored from the header: quantize.imatrix.dataset, .chunks_count and .entries_count.
    expect(DOWNLOADED_FILE.calibration).toEqual({ dataset: "unsloth_calibration_Qwen3.5-2B.txt", chunks: 80, entries: 186 });
  });
});

/* -------------------------------------------------------------------------- */
/* Float formats                                                               */
/* -------------------------------------------------------------------------- */

const floatFormat = (id: string) => FLOAT_FORMATS.find((entry) => entry.id === id)!;

/** An independent oracle for the 8-bit formats: decode every bit pattern from the Table 1 layout of Micikevicius et al. (2022). */
function decodeAll(exponentBits: number, mantissaBits: number, bias: number, kind: "e4m3" | "ieee") {
  const values: number[] = [];
  const topExponent = 2 ** exponentBits - 1;
  for (let code = 0; code < 2 ** (exponentBits + mantissaBits); code += 1) {
    const mantissa = code % 2 ** mantissaBits;
    const exponent = Math.floor(code / 2 ** mantissaBits);
    if (exponent === topExponent && (kind === "ieee" || mantissa === 2 ** mantissaBits - 1)) continue; // infinity or NaN
    values.push(exponent === 0 ? (mantissa / 2 ** mantissaBits) * 2 ** (1 - bias) : (1 + mantissa / 2 ** mantissaBits) * 2 ** (exponent - bias));
  }
  return [...new Set(values)].sort((a, b) => a - b);
}
/** Nearest value in a sorted list, ties to the one with an even code (even index parity matches an even mantissa for these grids). */
const nearest = (grid: number[], x: number) => {
  let best = grid[0];
  for (const candidate of grid) if (Math.abs(candidate - x) < Math.abs(best - x)) best = candidate;
  return best;
};

describe("quantization: float formats", () => {
  const stats = (id: string) => floatFormatStats(floatFormat(id));

  it("computes range and precision from the bit layouts", () => {
    // standard.mdx / plain.mdx / card-info.ts: fp32 1+8+23, fp16 1+5+10, bf16 1+8+7, E4M3 1+4+3, E5M2 1+5+2.
    expect(FLOAT_FORMATS.map((entry) => [entry.id, entry.exponentBits, entry.mantissaBits, 1 + entry.exponentBits + entry.mantissaBits])).toEqual([
      ["fp32", 8, 23, 32],
      ["fp16", 5, 10, 16],
      ["bf16", 8, 7, 16],
      ["e4m3", 4, 3, 8],
      ["e5m2", 5, 2, 8],
    ]);
    // Largest finite values: fp16 65,504, E4M3 448, E5M2 57,344 (Micikevicius et al. 2022, Table 1; IEEE 754 binary16).
    expect(stats("fp16").maxFinite).toBe(65504);
    expect(stats("e4m3").maxFinite).toBe(448);
    expect(stats("e5m2").maxFinite).toBe(57344);
    // fp32 and bf16 share the exponent, so bf16's maximum is fp32's less its missing mantissa bits: 3.39e38 against 3.40e38.
    expect(stats("fp32").maxFinite).toBe(Math.fround(3.4028234663852886e38));
    expect(Number(stats("bf16").maxFinite.toPrecision(3))).toBe(3.39e38);
    expect(Number(stats("fp32").maxFinite.toPrecision(3))).toBe(3.4e38);
    // Smallest normal values: 2^-14 for fp16 and E5M2, 2^-6 for E4M3, 2^-126 for fp32 and bf16.
    expect(stats("fp16").minNormal).toBe(2 ** -14);
    expect(stats("e5m2").minNormal).toBe(2 ** -14);
    expect(stats("e4m3").minNormal).toBe(2 ** -6);
    expect(stats("bf16").minNormal).toBe(stats("fp32").minNormal);
    // Smallest subnormals: fp16 2^-24, E4M3 2^-9, E5M2 2^-16.
    expect([stats("fp16").minSubnormal, stats("e4m3").minSubnormal, stats("e5m2").minSubnormal]).toEqual([2 ** -24, 2 ** -9, 2 ** -16]);
    // Spacing at 1.0 is 2^-mantissa bits: bf16 is exactly eight times coarser than fp16.
    expect(stats("bf16").spacingAtOne / stats("fp16").spacingAtOne).toBe(8);
    expect([stats("fp16").spacingAtOne, stats("bf16").spacingAtOne, stats("e4m3").spacingAtOne, stats("e5m2").spacingAtOne]).toEqual([2 ** -10, 2 ** -7, 0.125, 0.25]);
    expect(stats("e4m3").maxRelativeError).toBe(0.0625);
    expect(stats("e5m2").maxRelativeError).toBe(0.125);
    // "253 finite values" for E4M3 and 247 for E5M2: 256 codes less the NaNs and infinities, with +0 and -0 counted once.
    expect([stats("e4m3").finiteValues, stats("e5m2").finiteValues, stats("fp16").finiteValues]).toEqual([253, 247, 63487]);
  });

  it("agrees with independent oracles for each format", () => {
    const probes = [0.001, 0.02, 0.0123, 0.5, 1, 1.0001, 1.5, 2.5, 5.93, 12.34, 300, 448, 449, 1000, 57344, 65504, 65519, 100000, 1e-5, 1e-3 * 3, 6.2e-5, 1e-7, 3e-8];
    const bf16Oracle = (x: number) => {
      const view = new DataView(new ArrayBuffer(4));
      view.setFloat32(0, x);
      const bits = view.getUint32(0);
      view.setUint32(0, ((bits + 0x7fff + ((bits >> 16) & 1)) >>> 16) << 16);
      return view.getFloat32(0);
    };
    for (const x of probes) {
      expect(roundToFloatFormat(x, floatFormat("fp32")).value, `fp32 ${x}`).toBe(Math.fround(x));
      if (x < 65520) expect(roundToFloatFormat(x, floatFormat("fp16")).value, `fp16 ${x}`).toBe(toHalf(x));
      expect(roundToFloatFormat(x, floatFormat("bf16")).value, `bf16 ${x}`).toBe(bf16Oracle(x));
      const e4m3 = decodeAll(4, 3, 7, "e4m3");
      const e5m2 = decodeAll(5, 2, 15, "ieee");
      // Inside the finite range the nearest decoded value (ties do not occur on these probes) is the answer.
      if (x <= 448) expect(roundToFloatFormat(x, floatFormat("e4m3")).value, `e4m3 ${x}`).toBe(nearest(e4m3, x));
      if (x <= 57344) expect(roundToFloatFormat(x, floatFormat("e5m2")).value, `e5m2 ${x}`).toBe(nearest(e5m2, x));
    }
    // The decoded grids hold exactly as many values as the formulas say, and top out where Table 1 says.
    expect(decodeAll(4, 3, 7, "e4m3").length * 2 - 1).toBe(253);
    expect(Math.max(...decodeAll(4, 3, 7, "e4m3"))).toBe(448);
    expect(decodeAll(5, 2, 15, "ieee").length * 2 - 1).toBe(247);
    expect(Math.max(...decodeAll(5, 2, 15, "ieee"))).toBe(57344);
  });

  it("rounds ties to even and flags overflow and underflow", () => {
    // fp16 spacing at 1 is 2^-10, so 1 + 2^-11 is a tie between 1 and 1 + 2^-10: even mantissa wins, 1. The next tie rounds up to the even neighbour.
    expect(roundToFloatFormat(1 + 2 ** -11, floatFormat("fp16")).value).toBe(1);
    expect(roundToFloatFormat(1 + 3 * 2 ** -11, floatFormat("fp16")).value).toBe(1 + 2 ** -9);
    // 65,519 still rounds to 65,504; 65,520 is a tie that rounds up past the largest finite value, so it overflows.
    expect(roundToFloatFormat(65519, floatFormat("fp16"))).toMatchObject({ value: 65504, overflow: false });
    expect(roundToFloatFormat(65520, floatFormat("fp16")).overflow).toBe(true);
    // Checkpoint question 7: at 100,000 fp16, E4M3, and E5M2 overflow; bf16 and fp32 hold the value.
    const overflows = (id: string) => roundToFloatFormat(100000, floatFormat(id)).overflow;
    expect(["fp32", "fp16", "bf16", "e4m3", "e5m2"].map(overflows)).toEqual([false, true, false, true, true]);
    // bf16 stores 100,000 as 99,840, 0.16% low.
    expect(roundToFloatFormat(100000, floatFormat("bf16")).value).toBe(99840);
    expect(round(((99840 - 100000) / 100000) * 100, 2)).toBe(-0.16);
    // Underflow and subnormals: 1e-5 is zero in E4M3 (smallest subnormal 2^-9), subnormal in fp16, and 0.001 is subnormal in E4M3.
    expect(roundToFloatFormat(1e-5, floatFormat("e4m3")).value).toBe(0);
    expect(roundToFloatFormat(1e-5, floatFormat("fp16")).subnormal).toBe(true);
    expect(roundToFloatFormat(0.001, floatFormat("e4m3"))).toMatchObject({ value: 2 ** -9, subnormal: true });
    // Signs and zero survive.
    expect(roundToFloatFormat(-0.02, floatFormat("fp16")).value).toBe(-roundToFloatFormat(0.02, floatFormat("fp16")).value);
    expect(roundToFloatFormat(0, floatFormat("e4m3")).value).toBe(0);
  });

  it("matches the example values the lesson and card quote", () => {
    // card-info.ts / standard.mdx: 0.02 is stored 2.0004e-2 in fp16, 2.0020e-2 in bf16 and 1.9531e-2 (2.3% low) in both fp8 formats;
    // 5.93 becomes 5.9297 in fp16, 5.9375 in bf16 and 6 in both fp8 formats.
    const stored = (value: number, id: string) => roundToFloatFormat(value, floatFormat(id)).value;
    expect([0.02, 5.93].map((x) => ["fp16", "bf16", "e4m3", "e5m2"].map((id) => Number(stored(x, id).toPrecision(5))))).toEqual([
      [0.020004, 0.02002, 0.019531, 0.019531],
      [5.9297, 5.9375, 6, 6],
    ]);
    // The Error column prints two significant digits: 0.021%, 0.098%, and 2.3% (the exact errors are 0.0214%, 0.0977%, and 2.34%).
    const errorText = (id: string) => Math.abs(((stored(0.02, id) - 0.02) / 0.02) * 100).toPrecision(2);
    expect(["fp16", "bf16", "e4m3", "e5m2"].map(errorText)).toEqual(["0.021", "0.098", "2.3", "2.3"]);
    expect(round(((stored(0.02, "e4m3") - 0.02) / 0.02) * 100, 2)).toBe(-2.34);
  });

  it("measures the toy table through each format", () => {
    // card-info.ts / standard.mdx: RMS error 0.0002 (fp16), 0.0017 (bf16), 0.0267 (E4M3), 0.0531 (E5M2); perplexity cost +0.17% (E4M3), +0.36% (E5M2),
    // and nothing measurable for fp32, fp16 or bf16. Checkpoint question 8: fp16 lands more than eight times closer than bf16.
    const rows = Object.fromEntries(
      FLOAT_FORMATS.map((entry) => {
        const rounded = roundTableToFloatFormat(weights, entry);
        return [entry.id, { rms: errorStats(weights, rounded).rmsError, cost: cost(rounded) }];
      }),
    );
    expect(rows.fp32.rms).toBe(0);
    expect(round(rows.fp16.rms, 4)).toBe(0.0002);
    expect(round(rows.bf16.rms, 4)).toBe(0.0017);
    expect(round(rows.e4m3.rms, 4)).toBe(0.0267);
    expect(round(rows.e5m2.rms, 4)).toBe(0.0531);
    expect(rows.bf16.rms / rows.fp16.rms).toBeGreaterThan(8);
    expect(rows.bf16.rms / rows.fp16.rms).toBeLessThan(9);
    expect(round(rows.e4m3.cost, 2)).toBe(0.17);
    expect(round(rows.e5m2.cost, 2)).toBe(0.36);
    for (const id of ["fp32", "fp16", "bf16"]) expect(Math.abs(rows[id].cost), id).toBeLessThan(0.005);
    // The table sits inside every format's range, so no weight overflows (the lab note says so): -1.71 to +5.93.
    for (const entry of FLOAT_FORMATS) {
      for (const weight of weights) expect(roundToFloatFormat(weight, entry).overflow).toBe(false);
    }
    // card-info.ts: "Q8_0 spends 8.5 bits for a change below 0.01%", against +0.17% for the 8-bit float without a scale.
    expect(cost(format("Q8_0").quantize(weights))).toBeLessThan(0.01);
    expect(rows.e4m3.cost).toBeGreaterThan(cost(format("Q8_0").quantize(weights)) * 10);
  });
});

describe("quantization: shared state", () => {
  it("translates a version-2 payload, which had no fileGroup or floatValue", () => {
    const v2 = JSON.stringify({ bits: 3, scope: "row", groupSize: 12, zeroPoint: "symmetric", inspect: 2 });
    expect(hydrateQuantizationState(v2)).toEqual({
      bits: 3,
      scope: "row",
      groupSize: 12,
      zeroPoint: "symmetric",
      inspect: 2,
      fileGroup: "type",
      floatValue: 0.02,
    });
  });

  it("clamps and validates the new keys", () => {
    expect(hydrateQuantizationState(JSON.stringify({ floatValue: 1e9 })).floatValue).toBe(1e6);
    expect(hydrateQuantizationState(JSON.stringify({ floatValue: -1e9 })).floatValue).toBe(1e-6);
    expect(hydrateQuantizationState(JSON.stringify({ floatValue: "1e999" })).floatValue).toBe(0.02);
    expect(hydrateQuantizationState('{"floatValue": 1e999}').floatValue).toBe(0.02);
    expect(hydrateQuantizationState(JSON.stringify({ fileGroup: "role" })).fileGroup).toBe("role");
    expect(hydrateQuantizationState(JSON.stringify({ fileGroup: "<script>" })).fileGroup).toBe("type");
    expect(hydrateQuantizationState(JSON.stringify({ fileGroup: 7 })).fileGroup).toBe("type");
  });
});
