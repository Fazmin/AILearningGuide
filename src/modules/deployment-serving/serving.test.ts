import { describe, expect, it } from "vitest";
import { GIB, HARDWARE, kvBytesPerToken, MODELS, SERVE_FORMATS, servingPoint, speculativeTokensPerPass } from "./serving";

const large = MODELS[2];
const q4km = SERVE_FORMATS.find((format) => format.name === "Q4_K_M")!;

describe("serving arithmetic", () => {
  it("prices the KV cache per token from the model shape", () => {
    expect(kvBytesPerToken(large)).toBe(128 * 1024);
    expect(kvBytesPerToken(MODELS[1])).toBe(36 * 1024);
    expect(kvBytesPerToken(MODELS[0])).toBe(28 * 1024);
  });

  it("splits memory into weights, cache, and overhead", () => {
    const point = servingPoint(large, q4km, 8, 4096, true);
    expect(point.weightBytes / GIB).toBeCloseTo(4.58, 2);
    expect(point.kvBytes).toBe(8 * 4096 * 128 * 1024);
    expect(point.totalBytes).toBeCloseTo(point.weightBytes + point.kvBytes + HARDWARE.overheadBytes, 0);
    expect(point.maxStreams).toBe(36);
  });

  it("reads weights once per batched step and once per stream when serial", () => {
    const batched = servingPoint(large, q4km, 8, 4096, true);
    const serial = servingPoint(large, q4km, 8, 4096, false);
    const step = (batched.weightBytes + batched.kvBytes) / HARDWARE.bandwidthBytesPerSecond;
    expect(batched.totalTokensPerSecond).toBeCloseTo(8 / step, 6);
    expect(batched.perUserTokensPerSecond).toBeCloseTo(1 / step, 6);
    expect(serial.totalTokensPerSecond).toBeCloseTo(servingPoint(large, q4km, 1, 4096, true).totalTokensPerSecond, 6);
    expect(serial.timePerOutputToken).toBeCloseTo(8 / serial.totalTokensPerSecond, 9);
  });

  it("stays bandwidth bound across every control setting", () => {
    for (const model of MODELS) {
      for (const format of SERVE_FORMATS) {
        for (const users of [1, 16, 64]) {
          for (const context of [512, 4096, 32768]) {
            expect(servingPoint(model, format, users, context, true).computeBound).toBe(false);
          }
        }
      }
    }
  });

  it("computes the speculative decoding expectation", () => {
    expect(speculativeTokensPerPass(0.8, 4)).toBeCloseTo((1 - 0.8 ** 5) / 0.2, 12);
    expect(speculativeTokensPerPass(1, 4)).toBe(5);
  });
});
