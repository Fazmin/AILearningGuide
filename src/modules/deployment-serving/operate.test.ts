import { describe, expect, it } from "vitest";
import { speculativeTokensPerPass } from "./serving";
import {
  activePreset,
  DEFAULT_CONFIG,
  formatsFor,
  HARDWARE_PRESETS,
  hardwareFrom,
  KV_TYPES,
  LIMITS,
  leversFrom,
  LOCAL_ARCHITECTURE,
  LOCAL_FORMATS,
  LOCAL_MODEL,
  NO_LEVERS,
  normalizeConfig,
  OPERATING_MODELS,
  operatingPoint,
  PAGE_TOKENS,
  recurrentStateBytesPerStream,
  type Levers,
  type OperatingHardware,
} from "./operate";
import { GIB, HARDWARE, MODELS, SERVE_FORMATS, servingPoint } from "./serving";

const lab = hardwareFrom(HARDWARE_PRESETS[0]);
const large = OPERATING_MODELS.find((model) => model.id === "large")!;
const q4km = SERVE_FORMATS.find((format) => format.name === "Q4_K_M")!;
const localQ4 = LOCAL_FORMATS.find((format) => format.name === "Q4_K_M")!;
const preset = (id: string) => HARDWARE_PRESETS.find((entry) => entry.id === id)!;
const point = (users: number, hardware: OperatingHardware = lab, levers: Levers = NO_LEVERS, batching = true, context = 4096) =>
  operatingPoint(large, q4km, users, context, batching, hardware, levers);

describe("operatingPoint with no levers", () => {
  it("reproduces the SDK's servingPoint for every shipped preset, format, load, and scheduler", () => {
    const keys = [
      "weightBytes",
      "kvBytesPerToken",
      "kvBytesPerStream",
      "kvBytes",
      "totalBytes",
      "maxStreams",
      "stepSeconds",
      "totalTokensPerSecond",
      "perUserTokensPerSecond",
      "timePerOutputToken",
      "computeCeiling",
      "timeToFirstToken",
    ] as const;
    for (const model of MODELS) {
      const operating = OPERATING_MODELS.find((entry) => entry.id === model.id)!;
      for (const format of SERVE_FORMATS) {
        for (const users of [1, 8, 37, 64]) {
          for (const context of [512, 4096, 32768]) {
            for (const batching of [true, false]) {
              const expected = servingPoint(model, format, users, context, batching);
              const actual = operatingPoint(operating, format, users, context, batching, lab);
              for (const key of keys) expect(actual[key], `${model.id} ${format.name} ${users} ${context} ${key}`).toBeCloseTo(expected[key], 9);
              expect(actual.fits).toBe(expected.fits);
              expect(actual.computeBound).toBe(expected.computeBound);
            }
          }
        }
      }
    }
  });

  it("starts from the lab's original hardware, so the cards that existed before keep their numbers", () => {
    expect(lab).toMatchObject({
      vramBytes: HARDWARE.vramBytes,
      bandwidthBytesPerSecond: HARDWARE.bandwidthBytesPerSecond,
      computeFlops: HARDWARE.computeFlops,
      overheadBytes: HARDWARE.overheadBytes,
    });
    expect(HARDWARE_PRESETS[0]).toMatchObject({ memoryGiB: 24, bandwidthGBs: 400, computeTflops: 120 });
  });
});

describe("a hybrid model", () => {
  it("counts the recurrent state from llama.cpp's formulas in float32", () => {
    // n_embd_s = ssm_d_state x ssm_d_inner = 128 x 2048; n_embd_r = (d_conv - 1) x (d_inner + 2 x n_group x d_state) = 3 x 6144.
    expect(128 * 2048).toBe(262_144);
    expect(3 * (2048 + 2 * 16 * 128)).toBe(18_432);
    expect(recurrentStateBytesPerStream({ recurrentLayers: 1, stateSize: 128, innerSize: 2048, convKernel: 4, groupCount: 16 })).toBe(1_122_304);
    expect(LOCAL_MODEL.stateBytesPerStream).toBe(20_201_472);
    expect(Number((LOCAL_MODEL.stateBytesPerStream / 1024 ** 2).toFixed(2))).toBe(19.27);
    expect(LOCAL_ARCHITECTURE.recurrentLayers + LOCAL_ARCHITECTURE.fullAttentionLayers).toBe(LOCAL_ARCHITECTURE.totalLayers);
    expect(LOCAL_ARCHITECTURE.fullAttentionLayers).toBe(LOCAL_ARCHITECTURE.totalLayers / 4);
  });

  it("charges the cache on the full-attention layers only", () => {
    const local = operatingPoint(LOCAL_MODEL, localQ4, 1, 4096, true, lab);
    // 2 x 6 layers x 2 KV heads x 256 dims x 2 bytes = 12,288 bytes, 12 KiB a token; 24 layers would cost four times that.
    expect(local.kvBytesPerToken).toBe(12_288);
    expect(operatingPoint({ ...LOCAL_MODEL, layers: 24 }, localQ4, 1, 4096, true, lab).kvBytesPerToken).toBe(4 * local.kvBytesPerToken);
    // The state is per stream and independent of context, the cache is per token.
    const long = operatingPoint(LOCAL_MODEL, localQ4, 1, 32768, true, lab);
    expect(long.stateBytesPerStream).toBe(local.stateBytesPerStream);
    expect(long.kvBytesPerStream).toBe(8 * local.kvBytesPerStream);
    // Both are charged per stream in the memory total and the streams-that-fit count.
    const four = operatingPoint(LOCAL_MODEL, localQ4, 4, 4096, true, lab);
    expect(four.totalBytes).toBeCloseTo(four.weightBytes + 4 * (four.kvBytesPerStream + four.stateBytesPerStream) + HARDWARE.overheadBytes, 0);
    expect(four.stateBytes).toBe(4 * LOCAL_MODEL.stateBytesPerStream);
  });

  it("weighs the weights as the file this app downloads", () => {
    expect(formatsFor(LOCAL_MODEL).map((format) => format.name)).toEqual(["F16", "Q4_K_M"]);
    expect(formatsFor(large)).toBe(SERVE_FORMATS);
    const file = operatingPoint(LOCAL_MODEL, localQ4, 1, 4096, true, lab).weightBytes;
    expect(file).toBeCloseTo(LOCAL_ARCHITECTURE.fileBytes, 0);
    expect(localQ4.bitsPerWeight).toBeCloseTo(5.445, 3);
    expect(operatingPoint(LOCAL_MODEL, LOCAL_FORMATS[0], 1, 4096, true, lab).weightBytes).toBe(2 * LOCAL_ARCHITECTURE.parameters);
  });

  it("keeps the recurrent state out of the cache-type lever", () => {
    const fp16 = operatingPoint(LOCAL_MODEL, localQ4, 8, 4096, true, lab, NO_LEVERS);
    const q4 = operatingPoint(LOCAL_MODEL, localQ4, 8, 4096, true, lab, { ...NO_LEVERS, kvBits: 4.5 });
    expect(q4.stateBytesPerStream).toBe(fp16.stateBytesPerStream);
    expect(q4.kvBytesPerStream).toBeCloseTo((fp16.kvBytesPerStream * 4.5) / 16, 6);
  });
});

describe("paged attention", () => {
  it("charges the filled share of the context in whole pages", () => {
    const paged = (fill: number, context = 4096) => point(1, lab, { ...NO_LEVERS, paged: true, pagedFill: fill }, true, context);
    expect(paged(0.5).tokensHeldPerStream).toBe(2048);
    // 10% of 4,096 is 409.6 tokens, which rounds up to 26 pages of 16, 416 tokens.
    expect(paged(0.1).tokensHeldPerStream).toBe(Math.ceil((4096 * 0.1) / PAGE_TOKENS) * PAGE_TOKENS);
    expect(paged(0.1).tokensHeldPerStream % PAGE_TOKENS).toBe(0);
    // Full fill is the unpaged reserve, and a stream never holds more than its context.
    expect(paged(1).tokensHeldPerStream).toBe(4096);
    expect(paged(1).kvBytesPerStream).toBe(point(1).kvBytesPerStream);
    for (const fill of [0.1, 0.35, 0.5, 0.95, 1]) expect(paged(fill).tokensHeldPerStream).toBeLessThanOrEqual(4096);
  });

  it("raises streams that fit and lowers the bytes read each step", () => {
    const base = point(8);
    const paged = point(8, lab, { ...NO_LEVERS, paged: true, pagedFill: 0.5 });
    expect(paged.maxStreams).toBeGreaterThan(base.maxStreams);
    expect(paged.perUserTokensPerSecond).toBeGreaterThan(base.perUserTokensPerSecond);
    expect(paged.weightBytes).toBe(base.weightBytes);
  });
});

describe("KV-cache quantization", () => {
  it("scales the cache by the block layout's bits per value", () => {
    expect(KV_TYPES.map((type) => type.bitsPerValue)).toEqual([16, 8.5, 4.5]);
    // q8_0 is 34 bytes per 32 values and q4_0 is 18: the same layouts as the Quantization lab's Q8_0 and Q4_0 blocks.
    expect((34 * 8) / 32).toBe(8.5);
    expect((18 * 8) / 32).toBe(4.5);
    const base = point(1);
    for (const type of KV_TYPES) {
      const quantized = point(1, lab, { ...NO_LEVERS, kvBits: type.bitsPerValue });
      expect(quantized.kvBytesPerToken).toBeCloseTo((base.kvBytesPerToken * type.bitsPerValue) / 16, 6);
      expect(quantized.maxStreams).toBeGreaterThanOrEqual(base.maxStreams);
    }
    expect(8.5 / 16).toBeCloseTo(0.53, 2);
    expect(4.5 / 16).toBeCloseTo(0.28, 2);
  });
});

describe("speculative decoding", () => {
  const speculative = (alpha: number, gamma: number, cost: number): Levers => ({
    ...NO_LEVERS,
    speculative: true,
    acceptance: alpha,
    draftLength: gamma,
    draftCost: cost,
  });

  it("matches Leviathan et al.'s walltime improvement while decode is bandwidth bound", () => {
    // Theorem 3.8: (1 - a^(g+1)) / ((1 - a)(g c + 1)).
    for (const [alpha, gamma, cost] of [
      [0.8, 4, 0.1],
      [0.5, 2, 0.05],
      [0.9, 8, 0.3],
    ] as const) {
      const expected = (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * cost + 1));
      for (const users of [1, 8, 36]) {
        const gain = point(users, lab, speculative(alpha, gamma, cost)).totalTokensPerSecond / point(users).totalTokensPerSecond;
        expect(gain, `${alpha} ${gamma} ${cost} ${users}`).toBeCloseTo(expected, 9);
      }
    }
    expect(speculativeTokensPerPass(0.8, 4)).toBeCloseTo(3.362, 3);
    expect(speculativeTokensPerPass(0.8, 4) / (1 + 4 * 0.1)).toBeCloseTo(2.401, 3);
  });

  it("is off by default and never changes memory", () => {
    expect(NO_LEVERS.speculative).toBe(false);
    expect(point(8, lab, speculative(0.8, 4, 0.1)).totalBytes).toBe(point(8).totalBytes);
    expect(point(8, lab, { ...speculative(0.8, 4, 0.1), speculative: false }).totalTokensPerSecond).toBe(point(8).totalTokensPerSecond);
  });

  it("stops helping, and then hurts, once verifying gamma + 1 tokens per stream is compute bound", () => {
    // On the CPU box (compute is an assumption) 8 streams of the 8B model are already compute bound.
    const cpu = hardwareFrom(preset("cpu"), 1);
    const base = point(8, cpu);
    const spec = point(8, cpu, speculative(0.8, 4, 0.1));
    expect(base.computeBound).toBe(true);
    expect(spec.totalTokensPerSecond).toBeLessThan(base.totalTokensPerSecond);
    // At one stream it is bandwidth bound with room to spare, and speculation still helps, though less than 2.40.
    const single = point(1, cpu, speculative(0.8, 4, 0.1)).totalTokensPerSecond / point(1, cpu).totalTokensPerSecond;
    expect(single).toBeGreaterThan(1);
    expect(single).toBeLessThan(2.401);
  });

  it("works with the one-at-a-time scheduler", () => {
    const serial = point(8, lab, speculative(0.8, 4, 0.1), false);
    const base = point(8, lab, NO_LEVERS, false);
    expect(serial.totalTokensPerSecond / base.totalTokensPerSecond).toBeCloseTo(2.401, 3);
    expect(serial.perUserTokensPerSecond).toBeCloseTo(serial.totalTokensPerSecond / 8, 9);
  });
});

describe("cost per million tokens", () => {
  it("is the hourly price over the tokens generated in an hour", () => {
    const base = point(8);
    expect(base.costPerMillionTokens).toBeCloseTo((1 / (base.totalTokensPerSecond * 3600)) * 1e6, 9);
    // Doubling the price doubles the cost, halving the busy share doubles it, and a free machine costs nothing.
    expect(point(8, hardwareFrom({ ...preset("lab"), pricePerHour: 2 })).costPerMillionTokens).toBeCloseTo(2 * base.costPerMillionTokens, 9);
    expect(point(8, hardwareFrom(preset("lab"), 0.5)).costPerMillionTokens).toBeCloseTo(2 * base.costPerMillionTokens, 9);
    expect(point(8, hardwareFrom({ ...preset("lab"), pricePerHour: 0 })).costPerMillionTokens).toBe(0);
  });

  it("falls as users share each read of the weights, and is infinite past the memory limit", () => {
    const costs = [1, 2, 4, 8, 16, 36].map((users) => point(users).costPerMillionTokens);
    for (let index = 1; index < costs.length; index += 1) expect(costs[index]).toBeLessThan(costs[index - 1]);
    expect(point(36).fits).toBe(true);
    expect(point(37).fits).toBe(false);
    expect(point(37).costPerMillionTokens).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("hardware presets", () => {
  it("name only figures that agree with the published arithmetic", () => {
    // RTX 4090: 384-bit bus at 21 Gbps is 1008 GB/s (NVIDIA Ada whitepaper v2.02, Appendix A).
    expect((384 / 8) * 21).toBe(preset("consumer").bandwidthGBs);
    // H100 SXM: NVIDIA lists 1,979 TFLOP/s with sparsity and its footnote says one half without.
    expect(1979 / 2).toBe(preset("datacenter").computeTflops);
    // Core i9-14900K: two channels of DDR5-5600, 8 bytes wide each, is 89.6 GB/s (Intel's spec page lists the same).
    expect((2 * 8 * 5600) / 1000).toBe(preset("cpu").bandwidthGBs);
    expect([preset("consumer").memoryGiB, preset("datacenter").memoryGiB, preset("laptop").memoryGiB]).toEqual([24, 80, 64]);
    expect(preset("laptop").bandwidthGBs).toBe(546);
    expect(preset("datacenter").bandwidthGBs).toBe(3350);
  });

  it("round-trips through the state normalizer unchanged, so a preset button stays pressed after a reload", () => {
    for (const entry of HARDWARE_PRESETS) {
      const config = normalizeConfig({ ...DEFAULT_CONFIG, ...entry });
      expect(activePreset(config)?.id).toBe(entry.id);
    }
    expect(activePreset({ ...DEFAULT_CONFIG, memoryGiB: 25 })).toBeUndefined();
  });

  it("fits every preset's inputs inside the slider limits", () => {
    for (const entry of HARDWARE_PRESETS) {
      expect(entry.memoryGiB).toBeGreaterThanOrEqual(LIMITS.memoryGiB.min);
      expect(entry.memoryGiB).toBeLessThanOrEqual(LIMITS.memoryGiB.max);
      expect(entry.bandwidthGBs).toBeLessThanOrEqual(LIMITS.bandwidthGBs.max);
      expect(entry.computeTflops).toBeLessThanOrEqual(LIMITS.computeTflops.max);
      expect(entry.pricePerHour).toBeLessThanOrEqual(LIMITS.pricePerHour.max);
      expect(entry.basis.length).toBeGreaterThan(40);
    }
  });
});

describe("normalizeConfig", () => {
  it("returns the defaults for an empty or malformed payload", () => {
    expect(normalizeConfig({})).toEqual(DEFAULT_CONFIG);
    expect(normalizeConfig({ users: "many", batching: "yes", kvType: 4, modelId: 3, formatName: ["Q4_K_M"] })).toEqual(DEFAULT_CONFIG);
  });

  it("clamps every numeric key to its limits for hostile values", () => {
    const numericKeys = Object.entries(DEFAULT_CONFIG)
      .filter(([, value]) => typeof value === "number")
      .map(([key]) => key as keyof typeof DEFAULT_CONFIG);
    for (const hostile of [1e9, -1e9, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0]) {
      const config = normalizeConfig(Object.fromEntries(numericKeys.map((key) => [key, hostile])));
      for (const key of numericKeys) {
        const value = config[key] as number;
        expect(Number.isFinite(value), `${key} ${hostile}`).toBe(true);
        const limits = (LIMITS as Record<string, { min: number; max: number }>)[key];
        expect(value, `${key} ${hostile}`).toBeGreaterThanOrEqual(limits.min);
        expect(value, `${key} ${hostile}`).toBeLessThanOrEqual(limits.max);
      }
    }
    expect(normalizeConfig({ users: 1e9 }).users).toBe(64);
    expect(normalizeConfig({ users: -1e9 }).users).toBe(1);
    expect(normalizeConfig({ context: 700 }).context).toBe(512);
    expect(normalizeConfig({ context: 5000 }).context).toBe(5120);
  });

  it("snaps to each control's step without floating-point noise", () => {
    expect(normalizeConfig({ acceptance: 0.35 }).acceptance).toBe(0.35);
    expect(normalizeConfig({ bandwidthGBs: 89.6 }).bandwidthGBs).toBe(89.6);
    expect(normalizeConfig({ pricePerHour: 3.49 }).pricePerHour).toBe(3.49);
    expect(normalizeConfig({ draftCost: 0.07 }).draftCost).toBe(0.07);
    expect(normalizeConfig({ utilization: 0.6 }).utilization).toBe(0.6);
  });

  it("accepts the keys of a version-1 payload and defaults the rest", () => {
    // Version 1 stored only these five keys.
    const v1 = { modelId: "medium", formatName: "Q8_0", users: 12, context: 2048, batching: false };
    expect(normalizeConfig(v1)).toEqual({ ...DEFAULT_CONFIG, ...v1 });
  });

  it("rejects unknown ids and names", () => {
    expect(normalizeConfig({ modelId: "huge" }).modelId).toBe("large");
    expect(normalizeConfig({ formatName: "Q1_Z" }).formatName).toBe("Q4_K_M");
    expect(normalizeConfig({ kvType: "q1" }).kvType).toBe("f16");
    expect(normalizeConfig({ modelId: "local" }).modelId).toBe("local");
  });

  it("builds levers that match the cache type's bits", () => {
    expect(leversFrom({ ...DEFAULT_CONFIG, kvType: "q8_0" }).kvBits).toBe(8.5);
    expect(leversFrom({ ...DEFAULT_CONFIG, kvType: "q4_0" }).kvBits).toBe(4.5);
    expect(leversFrom(DEFAULT_CONFIG).kvBits).toBe(16);
  });
});

describe("the GiB the memory limit is read in", () => {
  it("converts a preset's GiB to bytes with binary units", () => {
    expect(hardwareFrom(preset("datacenter")).vramBytes).toBe(80 * GIB);
  });
});

describe("the 2B preset on the lab's hardware", () => {
  it("stays bandwidth bound at every control setting, so the first three cards' wording holds for it", () => {
    for (const format of LOCAL_FORMATS) {
      for (const users of [1, 16, 64]) {
        for (const context of [512, 4096, 32768]) {
          for (const batching of [true, false]) {
            const result = operatingPoint(LOCAL_MODEL, format, users, context, batching, lab);
            expect(result.computeBound, `${format.name} ${users} ${context} ${batching}`).toBe(false);
          }
        }
      }
    }
  });
});
