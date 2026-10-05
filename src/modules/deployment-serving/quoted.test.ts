import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { TINY_VOCAB_SIZE } from "@app/module-sdk";
import definition, { hydrateServingState } from "./module";
import {
  hardwareFrom,
  HARDWARE_PRESETS,
  KV_TYPES,
  LOCAL_ARCHITECTURE,
  LOCAL_FORMATS,
  LOCAL_MODEL,
  NO_LEVERS,
  OPERATING_MODELS,
  operatingPoint,
  type Levers,
} from "./operate";
import {
  BLOCK_TYPE,
  GIB,
  HARDWARE,
  kvBytesPerToken,
  MODELS,
  SERVE_FORMATS,
  servingPoint,
  speculativeTokensPerPass,
} from "./serving";

/**
 * Pins the figures the deployment-serving lesson (content/standard.mdx, content/plain.mdx), card-info.ts and the
 * glossary quote. Everything here is closed-form arithmetic from `servingPoint`, the way Explore.tsx calls it:
 * `servingPoint(model, format, users, context, batching)` on the stated hardware. Nothing is timed.
 *
 * External sources: the per-format bits per weight come from llama.cpp's own reported Llama-3-8B file sizes
 * (F16 exact; Q8_0 7.96 GiB, Q5_K_M 5.33 GiB, Q4_K_M 4.58 GiB, Q4_0 4.34 GiB) over 8.03 B parameters. The file sizes
 * are authored constants taken from llama.cpp's published output; only what serving.ts computes from them is pinned.
 * The model shapes are the published configs of Qwen2.5-1.5B, Qwen2.5-3B and Llama-3.1-8B, likewise only pinned
 * through the KV bytes they produce.
 */
const [small, medium, large] = MODELS;
const defaults = definition.hydrateState("{}");
const model = (id: unknown) => MODELS.find((entry) => entry.id === id)!;
const format = (name: unknown) => SERVE_FORMATS.find((entry) => entry.name === name)!;
const point = (modelId: string, formatName: string, users: number, context: number, batching = true) =>
  servingPoint(model(modelId), format(formatName), users, context, batching);

/** Mirrors Explore's display helpers so a pinned figure is the one the lab prints. */
const tokens = (value: number) => (value >= 100 ? value.toFixed(0) : value.toFixed(1));
const milliseconds = (seconds: number) => Number((seconds * 1000).toFixed(0));
const gib = (bytes: number) => Number((bytes / GIB).toFixed(2));

describe("deployment-serving quoted values", () => {
  it("starts from the configuration the lesson walks through", () => {
    // standard.mdx / plain.mdx "How to play with it": "Start at 8B, Q4_K_M, 8 users, 4,096 tokens, Continuous batching".
    expect(defaults).toMatchObject({ modelId: "large", formatName: "Q4_K_M", users: 8, context: 4096, batching: true });
    expect(model(defaults.modelId)).toBe(large);
  });

  it("prices the KV cache per token from the model shape", () => {
    // standard.mdx / plain.mdx / card-info.ts: 8B costs 128 KiB per token, 3B 36 KiB, 1.5B 28 KiB.
    expect([large, medium, small].map((entry) => kvBytesPerToken(entry) / 1024)).toEqual([128, 36, 28]);
    // standard.mdx / card-info.ts: "the 8B shape pays 3.6 times the 3B shape's cache per token".
    expect(Number((kvBytesPerToken(large) / kvBytesPerToken(medium)).toFixed(1))).toBe(3.6);
  });

  it("converts the llama.cpp file sizes to bits per weight", () => {
    // standard.mdx: Q4_K_M is 4.58 GiB, 4.90 bits per weight.
    // card-info.ts "Serving controls": F16 is 16; Q8_0 8.52, Q5_K_M 5.70, Q4_K_M 4.90, Q4_0 4.64.
    expect(Object.fromEntries(SERVE_FORMATS.map((entry) => [entry.name, Number(entry.bitsPerWeight.toFixed(2))]))).toEqual({
      F16: 16,
      Q8_0: 8.52,
      Q5_K_M: 5.7,
      Q4_K_M: 4.9,
      Q4_0: 4.64,
    });
    // The quoted file sizes round-trip: weights for the 8.03 B model at each format equal llama.cpp's reported GiB.
    const gibByFormat = Object.fromEntries(SERVE_FORMATS.map((entry) => [entry.name, gib(point("large", entry.name, 1, 4096).weightBytes)]));
    expect(gibByFormat).toMatchObject({ Q8_0: 7.96, Q5_K_M: 5.33, Q4_K_M: 4.58, Q4_0: 4.34 });
    // standard.mdx / plain.mdx: F16 weights are 14.96 GiB (2 bytes x 8.03 B).
    expect(gibByFormat.F16).toBe(14.96);
  });

  it("matches the memory budget at the starting configuration", () => {
    // standard.mdx / plain.mdx: weights 4.58 GiB, cache 4.00 GiB, 36 streams fit. card-info.ts: 4.00 GiB cache is close to the 4.58 GiB weights.
    const start = point("large", "Q4_K_M", 8, 4096);
    expect(gib(start.weightBytes)).toBe(4.58);
    expect(gib(start.kvBytes)).toBe(4);
    expect(start.maxStreams).toBe(36);
    expect(start.kvBytes).toBeLessThan(start.weightBytes);
    // card-info.ts: Streams that fit = floor((24 GiB - weights - 1 GiB) / (KV bytes per token x context)).
    // At 4,096 tokens one stream's cache is 128 KiB x 4,096 = 0.5 GiB, so floor((24 - 4.58 - 1) / 0.5) = 36.
    expect(Math.floor((24 - 4.58 - 1) / 0.5)).toBe(36);
    expect(start.maxStreams).toBe(Math.floor((HARDWARE.vramBytes - start.weightBytes - GIB) / (kvBytesPerToken(large) * 4096)));
  });

  it("matches the users sweep: throughput, latency, and where memory runs out", () => {
    // standard.mdx / plain.mdx / card-info.ts: one user gets 73.3 tokens/s; eight users get 347 in total and 43.4 each, 23 ms per output token.
    const one = point("large", "Q4_K_M", 1, 4096);
    expect(tokens(one.totalTokensPerSecond)).toBe("73.3");
    expect(tokens(one.perUserTokensPerSecond)).toBe("73.3");
    const eight = point("large", "Q4_K_M", 8, 4096);
    expect(tokens(eight.totalTokensPerSecond)).toBe("347");
    expect(tokens(eight.perUserTokensPerSecond)).toBe("43.4");
    expect(milliseconds(eight.timePerOutputToken)).toBe(23);
    // standard.mdx: "At 64 the arithmetic asks for 37.58 GiB".
    const sixtyFour = point("large", "Q4_K_M", 64, 4096);
    expect(gib(sixtyFour.totalBytes)).toBe(37.58);
    expect(sixtyFour.fits).toBe(false);
    // standard.mdx / plain.mdx: shaded past 36 users. card-info.ts: "Streams that fit is exceeded at 37".
    expect(point("large", "Q4_K_M", 36, 4096).fits).toBe(true);
    expect(point("large", "Q4_K_M", 37, 4096).fits).toBe(false);
    // standard.mdx / plain.mdx: batching raises total and lowers per-user speed.
    expect(eight.totalTokensPerSecond).toBeGreaterThan(one.totalTokensPerSecond);
    expect(eight.perUserTokensPerSecond).toBeLessThan(one.perUserTokensPerSecond);
  });

  it("matches the context sweep at one user", () => {
    // standard.mdx / plain.mdx: at 32,768 tokens the cache reaches 4.00 GiB, Streams that fit drops to 4,
    // and time to first token grows from 548 ms to 4385 ms.
    const short = point("large", "Q4_K_M", 1, 4096);
    const long = point("large", "Q4_K_M", 1, 32768);
    expect(gib(long.kvBytes)).toBe(4);
    expect(long.maxStreams).toBe(4);
    expect([short, long].map((entry) => milliseconds(entry.timeToFirstToken))).toEqual([548, 4385]);
    // card-info.ts "Memory budget": cache grows from 0.50 to 4.00 GiB and Streams that fit falls from 36 to 4.
    expect([short, long].map((entry) => gib(entry.kvBytes))).toEqual([0.5, 4]);
    expect([short, long].map((entry) => entry.maxStreams)).toEqual([36, 4]);
    // standard.mdx step instruction: the KV cache segment "approaches the weights" at 32,768 tokens (4.00 against 4.58 GiB).
    expect(long.kvBytes).toBeLessThan(long.weightBytes);
    expect(long.kvBytes / long.weightBytes).toBeGreaterThan(0.85);
    // card-info.ts "Throughput and latency": time to first token = max(2 x params x context / 120 TFLOP/s, one weight read).
    expect(long.timeToFirstToken).toBeCloseTo((2 * large.params * 32768) / HARDWARE.computeFlops, 12);
  });

  it("matches the F16 format switch", () => {
    // standard.mdx / plain.mdx: "weights jump to 14.96 GiB and one user drops to 24.1 tokens/s" (73.3 at Q4_K_M).
    // The 24.1 holds at the starting context of 4,096 tokens, one user.
    const f16 = point("large", "F16", 1, 4096);
    expect(gib(f16.weightBytes)).toBe(14.96);
    expect(tokens(f16.perUserTokensPerSecond)).toBe("24.1");
    // The lessons now say "set Context length back to 4,096" first, because the paragraph before leaves the lab at
    // 32,768 tokens, where F16 gives 19.7 tokens/s (and Q4_K_M gives 43.4, not 73.3).
    expect(tokens(point("large", "F16", 1, 32768).perUserTokensPerSecond)).toBe("19.7");
    // card-info.ts: "Q8_0 and F16 slow decode because each step reads more bytes".
    const rates = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0", "F16"].map((name) => point("large", name, 8, 4096).totalTokensPerSecond);
    expect([...rates].sort((a, b) => b - a)).toEqual(rates);
    // standard.mdx: "Halving bits per weight roughly halves the weight read" (F16 -> Q8_0 is 0.53x of the bytes).
    const bytes = (name: string) => point("large", name, 1, 4096).weightBytes;
    expect(bytes("Q8_0") / bytes("F16")).toBeGreaterThan(0.5);
    expect(bytes("Q8_0") / bytes("F16")).toBeLessThan(0.6);
  });

  it("matches the One at a time scheduler", () => {
    // standard.mdx / plain.mdx: 8 users share 73.3 tokens/s, 9.2 each, and each waits 109 ms between tokens.
    // card-info.ts "Throughput and latency": total falls to 73.3 and each user gets 9.2.
    const serial = point("large", "Q4_K_M", 8, 4096, false);
    expect(tokens(serial.totalTokensPerSecond)).toBe("73.3");
    expect(tokens(serial.perUserTokensPerSecond)).toBe("9.2");
    expect(milliseconds(serial.timePerOutputToken)).toBe(109);
  });

  it("matches the compute ceiling and the bandwidth-bound claim", () => {
    // standard.mdx / card-info.ts: 8B compute ceiling is 120 TFLOP/s / (2 x 8.03 B) = 7,472 tokens/s.
    const eight = point("large", "Q4_K_M", 8, 4096);
    expect(Math.round(eight.computeCeiling)).toBe(7472);
    expect(HARDWARE.computeFlops / (2 * 8.03e9)).toBeCloseTo(7471.98, 2);
    // standard.mdx: "64 streams at 4,096 tokens reach only 652".
    expect(tokens(point("large", "Q4_K_M", 64, 4096).totalTokensPerSecond)).toBe("652");
    // standard.mdx "What to notice": "The badge says bandwidth bound everywhere" (both schedulers, every user count at the defaults).
    for (const batching of [true, false]) {
      for (let users = 1; users <= 64; users += 1) {
        expect(point("large", "Q4_K_M", users, 4096, batching).computeBound).toBe(false);
      }
    }
  });

  it("matches the per-user slope claim", () => {
    // card-info.ts "Throughput and latency" notice (:80): "Per-user speed falls slower than 1 / users while weights
    // dominate each read, and faster once the caches do."
    // Pinned: at 8B Q4_K_M / 4,096 tokens, batched per-user speed is always above the 1 / users line, and the fall
    // steepens toward it as the cache share of each read grows.
    // NOTE (qualitative, not a numeric mismatch): per-user speed never falls faster than 1 / users in this model
    // (that would need negative weight bytes); it approaches 1 / users from above as cache bytes dominate. "Faster"
    // is true only as "the log-log slope steepens", which the owner may want to reword.
    const perUser = (users: number) => point("large", "Q4_K_M", users, 4096).perUserTokensPerSecond;
    const slope = (users: number) => Math.log(perUser(users + 1) / perUser(users)) / Math.log((users + 1) / users);
    for (const users of [2, 8, 32, 64]) expect(perUser(users)).toBeGreaterThan(perUser(1) / users);
    expect(slope(1)).toBeGreaterThan(slope(8));
    expect(slope(8)).toBeGreaterThan(slope(32));
    expect(slope(32)).toBeGreaterThan(-1);
  });

  it("matches the speculative decoding expectation", () => {
    // standard.mdx "Going deeper" and glossary: (1 - alpha^(gamma+1)) / (1 - alpha) = 3.36 at alpha = 0.8, gamma = 4.
    expect(speculativeTokensPerPass(0.8, 4)).toBeCloseTo(3.36, 2);
  });

  it("matches the toy export sizes", () => {
    // card-info.ts "Export the model you trained": the toy size is 900 x the block type's bits per weight / 8, rounded up
    // (Explore.tsx toyBytes). Q4_K_M maps to Q4_K at 4.5 (507 bytes), Q5_K_M to Q5_K at 5.5, Q8_0 at 8.5, Q4_0 at 4.5, F16 at 16 (1,800 bytes).
    const toyWeights = TINY_VOCAB_SIZE * TINY_VOCAB_SIZE;
    expect(toyWeights).toBe(900);
    const toyBytes = (name: string) => Math.ceil((toyWeights * BLOCK_TYPE[name].bitsPerWeight) / 8);
    expect(BLOCK_TYPE.Q4_K_M).toMatchObject({ block: "Q4_K", bitsPerWeight: 4.5 });
    expect(BLOCK_TYPE.Q5_K_M).toMatchObject({ block: "Q5_K", bitsPerWeight: 5.5 });
    expect(BLOCK_TYPE.Q8_0.bitsPerWeight).toBe(8.5);
    expect(BLOCK_TYPE.Q4_0.bitsPerWeight).toBe(4.5);
    expect(toyBytes("Q4_K_M")).toBe(507);
    expect(toyBytes("F16")).toBe(1800);
    // The block-level table covers every selectable format.
    for (const entry of SERVE_FORMATS) expect(BLOCK_TYPE[entry.name]).toBeDefined();
  });

  it("separates GiB from GB in the plain-language common mix-up", () => {
    // plain.mdx's common mix-up now says "the model is 4.6 GiB": the lab's Q4_K_M 8B weights are 4.58 GiB, which is
    // 4.92 GB in decimal units. Both readings of the same weights are pinned:
    const weights = point("large", "Q4_K_M", 1, 4096).weightBytes;
    expect(Number((weights / GIB).toFixed(1))).toBe(4.6);
    expect(Number((weights / 1e9).toFixed(1))).toBe(4.9);
  });
});

/** module.ts checkpoint questions 1-5 compare directions rather than quote numbers; these pin the directions. */
describe("deployment-serving checkpoint claims", () => {
  it("question 1: users move the KV cache and never the weights", () => {
    const few = point("large", "Q4_K_M", 8, 4096);
    const many = point("large", "Q4_K_M", 64, 4096);
    expect(many.weightBytes).toBe(few.weightBytes);
    expect(many.kvBytes).toBeGreaterThan(few.kvBytes);
    expect(many.totalBytes).toBeGreaterThan(few.totalBytes);
  });

  it("question 2: every stream reserves its full context, which is the worst case paged attention avoids", () => {
    const one = point("large", "Q4_K_M", 1, 4096);
    expect(one.kvBytesPerStream).toBe(kvBytesPerToken(large) * 4096);
    expect(point("large", "Q4_K_M", 8, 4096).kvBytes).toBe(8 * one.kvBytesPerStream);
  });

  it("question 3: batching raises Total and lowers Per user, and One at a time splits a fixed total evenly", () => {
    const one = point("large", "Q4_K_M", 1, 4096);
    const eight = point("large", "Q4_K_M", 8, 4096);
    expect(eight.totalTokensPerSecond).toBeGreaterThan(one.totalTokensPerSecond);
    expect(eight.perUserTokensPerSecond).toBeLessThan(one.perUserTokensPerSecond);
    const roundRobin = point("large", "Q4_K_M", 8, 4096, false);
    expect(roundRobin.totalTokensPerSecond).toBeCloseTo(one.totalTokensPerSecond, 6);
    expect(roundRobin.perUserTokensPerSecond).toBeCloseTo(one.totalTokensPerSecond / 8, 6);
  });

  it("question 4: decode stays bandwidth bound with a compute ceiling far above the throughput reached", () => {
    const busy = point("large", "Q4_K_M", 64, 4096);
    expect(busy.computeBound).toBe(false);
    expect(busy.computeCeiling).toBeGreaterThan(busy.totalTokensPerSecond * 5);
    // Two FLOPs per parameter per token: the ceiling is compute / (2 x parameters).
    expect(busy.computeCeiling).toBeCloseTo(HARDWARE.computeFlops / (2 * large.params), 6);
  });
});


/* -------------------------------------------------------------------------- */
/* The model this app runs, hardware, levers, cost, and operating it           */
/* -------------------------------------------------------------------------- */

const readText = (path: string) => readFileSync(resolve(process.cwd(), path), "utf8");
const rust = readText("src-tauri/src/local_ai.rs");
const preset = (id: string) => HARDWARE_PRESETS.find((entry) => entry.id === id)!;
const largeModel = OPERATING_MODELS.find((entry) => entry.id === "large")!;
const q4km = SERVE_FORMATS.find((entry) => entry.name === "Q4_K_M")!;
const localQ4 = LOCAL_FORMATS.find((entry) => entry.name === "Q4_K_M")!;
const on = (id: string, users: number, levers: Levers = NO_LEVERS, batching = true, context = 4096) =>
  operatingPoint(largeModel, q4km, users, context, batching, hardwareFrom(preset(id), 1), levers);
const labSpeculative: Levers = { ...NO_LEVERS, speculative: true, acceptance: 0.8, draftLength: 4, draftCost: 0.1 };
/** The dollar figures the lab prints: two decimals from $0.10, three below. */
const dollars = (value: number) => (value >= 0.1 ? value.toFixed(2) : value.toFixed(3));

describe("deployment-serving: the model this app runs", () => {
  it("is the file the desktop app pins, run the way the app runs it", () => {
    // standard.mdx / card-info.ts: 1,280,835,840 bytes, llama.cpp b10991, a 4,096-token context, every layer on the GPU
    // on macOS and none elsewhere, a loopback address on a random port.
    expect(rust).toContain("const MODEL_BYTES: u64 = 1_280_835_840;");
    expect(rust).toContain('const ENGINE_VERSION: &str = "b10991";');
    expect(rust).toContain('TcpListener::bind("127.0.0.1:0")');
    expect(rust).toContain('.arg("--ctx-size")\n            .arg("4096")');
    expect(rust).toContain('if cfg!(target_os = "macos") { "99" } else { "0" }');
    expect(LOCAL_ARCHITECTURE.fileBytes).toBe(1_280_835_840);
    expect(localQ4.bitsPerWeight).toBeCloseTo(5.445, 3);
  });

  it("charges 6 of 24 layers per token and a fixed state on the other 18", () => {
    // standard.mdx / plain.mdx / card-info.ts / checkpoint question 6: 12 KiB a token against 48 KiB for 24 caching layers,
    // 19.27 MiB of float32 state per stream, 48 MiB of cache for a 4,096-token stream.
    const local = operatingPoint(LOCAL_MODEL, localQ4, 1, 4096, true, hardwareFrom(preset("lab")));
    expect(local.kvBytesPerToken / 1024).toBe(12);
    expect((local.kvBytesPerToken * 24) / LOCAL_MODEL.layers / 1024).toBe(48);
    expect(local.kvBytesPerStream / 1024 ** 2).toBe(48);
    expect(round2(local.stateBytesPerStream / 1024 ** 2)).toBe(19.27);
    expect(LOCAL_MODEL.stateBytesPerStream).toBe(20_201_472);
    // "a quarter of what 24 caching layers would cost".
    expect(LOCAL_MODEL.layers / LOCAL_ARCHITECTURE.totalLayers).toBe(0.25);
    // "331 streams fit where the 8B model fits 36" on the lab's 24 GiB.
    expect(local.maxStreams).toBe(331);
    expect(on("lab", 1).maxStreams).toBe(36);
    // The file is 1.19 GiB at Q4_K_M and 3.51 GiB at F16.
    expect(round2(local.weightBytes / GIB)).toBe(1.19);
    expect(round2(operatingPoint(LOCAL_MODEL, LOCAL_FORMATS[0], 1, 4096, true, hardwareFrom(preset("lab"))).weightBytes / GIB)).toBe(3.51);
    // The Memory budget bar draws the state in the cache segment, so the segments add up to the total.
    const four = operatingPoint(LOCAL_MODEL, localQ4, 4, 4096, true, hardwareFrom(preset("lab")));
    expect(four.weightBytes + four.kvBytes + four.stateBytes + HARDWARE.overheadBytes).toBeCloseTo(four.totalBytes, 0);
  });

  it("gives the 2B model's decode ceiling on the laptop and the CPU box, which depends on bandwidth alone", () => {
    // standard.mdx / plain.mdx "Operating it": 404 tokens/s on the M4 Max preset and 66 on the CPU box (1 user, 4,096 tokens).
    const ceiling = (id: string) => operatingPoint(LOCAL_MODEL, localQ4, 1, 4096, true, hardwareFrom(preset(id), 1)).perUserTokensPerSecond;
    expect([ceiling("laptop"), ceiling("cpu")].map(Math.round)).toEqual([404, 66]);
    // Bandwidth bound: the number is bytes over bandwidth, so it does not move with the compute assumption.
    const computeBumped = hardwareFrom({ ...preset("cpu"), computeTflops: 100 }, 1);
    expect(operatingPoint(LOCAL_MODEL, localQ4, 1, 4096, true, computeBumped).perUserTokensPerSecond).toBe(ceiling("cpu"));
  });
});

describe("deployment-serving: hardware and cost", () => {
  it("matches the single-stream and capacity comparison across the presets", () => {
    // standard.mdx / card-info.ts / plain.mdx: 8B Q4_K_M, 4,096 tokens, 1 user: the lab prints 73.3, 185, 614, 100, and 16.4 tok/s;
    // streams that fit are 36, 36, 148, 116, and 116.
    const ids = ["lab", "consumer", "datacenter", "laptop", "cpu"];
    expect(ids.map((id) => tokens(on(id, 1).perUserTokensPerSecond))).toEqual(["73.3", "185", "614", "100", "16.4"]);
    expect(ids.map((id) => on(id, 1).maxStreams)).toEqual([36, 36, 148, 116, 116]);
    // "3.3 times faster, the bandwidth ratio and not the 6.0 times compute ratio" (checkpoint question 7).
    const bandwidth = preset("datacenter").bandwidthGBs / preset("consumer").bandwidthGBs;
    const compute = preset("datacenter").computeTflops / preset("consumer").computeTflops;
    expect([bandwidth, compute].map((value) => Number(value.toFixed(1)))).toEqual([3.3, 6]);
    expect(on("datacenter", 1).perUserTokensPerSecond / on("consumer", 1).perUserTokensPerSecond).toBeCloseTo(bandwidth, 9);
    // ...because decode stays far below the compute ceiling on both.
    for (const id of ["consumer", "datacenter"]) expect(on(id, 1).computeCeiling).toBeGreaterThan(on(id, 36).totalTokensPerSecond * 5);
  });

  it("prices a million tokens at the listed prices", () => {
    // standard.mdx / plain.mdx / card-info.ts: at 8 users $0.23 on the 4090 against $0.33 on the H100; the lab default's placeholder price gives $0.80.
    expect([on("consumer", 8), on("datacenter", 8), on("lab", 8)].map((entry) => dollars(entry.costPerMillionTokens))).toEqual(["0.23", "0.33", "0.80"]);
    // "from $1.11 at 1 user to $0.14 at 36" on the 4090, whose memory holds 36 streams.
    expect(dollars(on("consumer", 1).costPerMillionTokens)).toBe("1.11");
    expect(dollars(on("consumer", 36).costPerMillionTokens)).toBe("0.14");
    expect(on("consumer", 36).fits).toBe(true);
    expect(on("consumer", 37).fits).toBe(false);
    // The price is an input: the same H100 SXM is $3.49 and $4.29 an hour on the two pages read on 30 September 2026.
    expect(preset("datacenter").pricePerHour).toBe(3.49);
    expect(preset("datacenter").basis).toContain("$4.29");
    expect(preset("consumer").pricePerHour).toBe(0.74);
    expect(Number((3.49 / 0.74).toFixed(1))).toBe(4.7);
  });
});

describe("deployment-serving: the preset notes", () => {
  it("keeps each preset's source note in step with its numbers", () => {
    // card-info.ts quotes the same figures with their sources; if a number changes, its note has to change too.
    expect(preset("lab").basis).toContain("24 GiB, 400 GB/s, 120 TFLOP/s");
    expect(preset("consumer").basis).toContain(`${preset("consumer").bandwidthGBs} GB/s`);
    expect(preset("consumer").basis).toContain(`${preset("consumer").computeTflops} TFLOP/s`);
    expect(preset("consumer").basis).toContain(`$${preset("consumer").pricePerHour}`);
    expect(preset("datacenter").basis).toContain(`${preset("datacenter").computeTflops} dense`);
    expect(preset("datacenter").basis).toContain("3.35 TB/s");
    expect(preset("datacenter").basis).toContain(`$${preset("datacenter").pricePerHour}`);
    expect(preset("laptop").basis).toContain(`${preset("laptop").bandwidthGBs} GB/s`);
    expect(preset("laptop").basis).toContain(`${preset("laptop").computeTflops} TFLOP/s`);
    expect(preset("cpu").basis).toContain(`${preset("cpu").bandwidthGBs} GB/s`);
    expect(preset("cpu").basis).toContain(`${preset("cpu").computeTflops} TFLOP/s`);
    expect(preset("cpu").basis).toContain(`${preset("cpu").memoryGiB} GiB`);
    // Every note names the date its figures were read, or says they are the lab's own assumptions.
    for (const entry of HARDWARE_PRESETS.filter((candidate) => candidate.id !== "lab")) expect(entry.basis).toContain("30 September 2026");
    // The two owned-hardware presets carry no hourly price.
    expect([preset("laptop").pricePerHour, preset("cpu").pricePerHour]).toEqual([0, 0]);
  });
});

describe("deployment-serving: the levers", () => {
  it("matches the lever comparison on the lab default at 8 users", () => {
    // standard.mdx / plain.mdx: paged at 50% fill takes 36 streams to 73 and 43.4 tok/s to 56.6; q8_0 gives 69 and 55.6;
    // speculation gives 104 tok/s each, 2.4 times, and $0.80 down to $0.33.
    const base = on("lab", 8);
    const paged = on("lab", 8, { ...NO_LEVERS, paged: true, pagedFill: 0.5 });
    const q8 = on("lab", 8, { ...NO_LEVERS, kvBits: KV_TYPES[1].bitsPerValue });
    const spec = on("lab", 8, labSpeculative);
    expect([base, paged, q8, spec].map((entry) => entry.maxStreams)).toEqual([36, 73, 69, 36]);
    expect([base, paged, q8, spec].map((entry) => tokens(entry.perUserTokensPerSecond))).toEqual(["43.4", "56.6", "55.6", "104"]);
    expect(Number((spec.perUserTokensPerSecond / base.perUserTokensPerSecond).toFixed(1))).toBe(2.4);
    expect([base, spec].map((entry) => dollars(entry.costPerMillionTokens))).toEqual(["0.80", "0.33"]);
    expect(Number(spec.tokensPerPass.toFixed(2))).toBe(3.36);
  });

  it("gives the checkpoint question 8 numbers: 3.36 tokens a pass, 2.4 after the draft's cost", () => {
    expect(Number(speculativeTokensPerPass(0.8, 4).toFixed(2))).toBe(3.36);
    expect(Number((speculativeTokensPerPass(0.8, 4) / (1 + 4 * 0.1)).toFixed(2))).toBe(2.4);
    // Speculation helps per-user speed but never changes memory, and does not apply to the first token only.
    expect(on("lab", 8, labSpeculative).totalBytes).toBe(on("lab", 8).totalBytes);
  });

  it("separates what paging changes from what the cache type changes (checkpoint question 9)", () => {
    const base = on("lab", 8);
    const paged = on("lab", 8, { ...NO_LEVERS, paged: true, pagedFill: 0.5 });
    const q8 = on("lab", 8, { ...NO_LEVERS, kvBits: 8.5 });
    expect(paged.tokensHeldPerStream).toBeLessThan(base.tokensHeldPerStream);
    expect(paged.kvBytesPerToken).toBe(base.kvBytesPerToken);
    expect(q8.tokensHeldPerStream).toBe(base.tokensHeldPerStream);
    expect(q8.kvBytesPerToken).toBeLessThan(base.kvBytesPerToken);
    expect(q8.weightBytes).toBe(base.weightBytes);
    expect(paged.weightBytes).toBe(base.weightBytes);
  });

  it("turns speculation into a slowdown once decode is compute bound (the CPU box)", () => {
    // standard.mdx / plain.mdx / card-info.ts: on the CPU box 8 streams are compute bound at 62.3 tok/s and speculation drops them to 38.8.
    const base = on("cpu", 8);
    const spec = on("cpu", 8, labSpeculative);
    expect(base.computeBound).toBe(true);
    expect([base, spec].map((entry) => tokens(entry.totalTokensPerSecond))).toEqual(["62.3", "38.8"]);
    // The compute figure behind that is an assumption, which the preset says.
    expect(preset("cpu").basis).toContain("assumption");
    expect(preset("laptop").basis).toContain("assumption");
  });

  it("matches the drift example: average context from 4,096 to 16,384 tokens", () => {
    // standard.mdx / plain.mdx "Operating it": Streams that fit falls from 36 to 9 on the 8B shape.
    expect([4096, 16384].map((context) => on("lab", 8, NO_LEVERS, true, context).maxStreams)).toEqual([36, 9]);
  });

  it("states the cache-type ratios the lesson quotes", () => {
    // standard.mdx / plain.mdx / card-info.ts: q8_0 and q4_0 cut the cache to 0.53 and 0.28 of fp16.
    expect(KV_TYPES.slice(1).map((type) => Number((type.bitsPerValue / 16).toFixed(2)))).toEqual([0.53, 0.28]);
  });
});

describe("deployment-serving: shared state", () => {
  it("translates a version-1 payload, which stored only five keys", () => {
    const v1 = JSON.stringify({ modelId: "medium", formatName: "Q8_0", users: 12, context: 2048, batching: false });
    const hydrated = hydrateServingState(v1);
    expect(hydrated).toMatchObject({ modelId: "medium", formatName: "Q8_0", users: 12, context: 2048, batching: false });
    // The keys version 2 added take the lab's original hardware and every lever off.
    expect(hydrated).toMatchObject({ memoryGiB: 24, bandwidthGBs: 400, computeTflops: 120, paged: false, speculative: false, kvType: "f16" });
  });

  it("validates every key, including the ones an old payload could carry in the wrong shape", () => {
    expect(hydrateServingState("{").users).toBe(8);
    expect(hydrateServingState("null")).toEqual(definition.initialState);
    expect(hydrateServingState("[1,2]")).toEqual(definition.initialState);
    expect(hydrateServingState(JSON.stringify({ users: 1e9, context: -1e9, memoryGiB: "lots", paged: "yes" }))).toMatchObject({
      users: 64,
      context: 512,
      memoryGiB: 24,
      paged: false,
    });
    expect(hydrateServingState('{"users": 1e999, "bandwidthGBs": -1e999}')).toMatchObject({ users: 8, bandwidthGBs: 400 });
  });
});

describe("deployment-serving: checkpoint claims for the new questions", () => {
  it("question 10 and 11: the rollout and privacy answers rest on how the app pins and serves its model", () => {
    // The app checks a known SHA-256 for both downloads before running them, and serves on loopback.
    expect(rust).toContain("MODEL_SHA256");
    expect(rust).toContain("const MODEL_SHA256: &str =");
    expect(readText("ARCHITECTURE.md")).toContain("random loopback-only port");
    expect(readText("ARCHITECTURE.md")).toContain("Both downloads require known SHA-256 hashes before execution");
  });
});

const round2 = (value: number) => Number(value.toFixed(2));
