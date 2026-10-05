import { describe, expect, it } from "vitest";
import { createTinyWeights, TINY_VOCAB_SIZE } from "@app/module-sdk";
import { devicesNeeded, formatBytes, MODEL_SIZES, memoryBreakdown, OPTIMIZERS, PRECISIONS } from "./memory";

/**
 * Pins every figure the Where the memory goes card, its card text, and the lesson quote. All of them are
 * products of the per-parameter byte counts; the two cross-checks are the ZeRO paper's own examples
 * (Rajbhandari et al., 2019, arXiv 1910.02054: 16 bytes per parameter for mixed-precision Adam, at least
 * 24 GB for a 1.5 billion parameter model, and 120 GB for 7.5 billion).
 */
const sizeOf = (id: string) => MODEL_SIZES.find((entry) => entry.id === id)!.parameters;

describe("memory arithmetic", () => {
  it("reproduces the ZeRO paper's 16 bytes per parameter for mixed-precision Adam", () => {
    const adam = memoryBreakdown(1, "mixed", "adam");
    expect(adam.weightBytesPerParameter).toBe(2);
    expect(adam.gradientBytesPerParameter).toBe(2);
    expect(adam.stateBytesPerParameter).toBe(12);
    expect(adam.stateParts).toEqual([
      { name: "fp32 master copy", bytesPerParameter: 4 },
      { name: "momentum", bytesPerParameter: 4 },
      { name: "variance", bytesPerParameter: 4 },
    ]);
    expect(adam.totalBytesPerParameter).toBe(16);
  });

  it("matches the paper's two worked examples", () => {
    expect(memoryBreakdown(sizeOf("1.5b"), "mixed", "adam").total).toBe(24e9);
    const example = memoryBreakdown(sizeOf("7.5b"), "mixed", "adam");
    expect(example.total).toBe(120e9);
    expect(example.weights).toBe(15e9);
    expect(example.gradients).toBe(15e9);
    expect(example.optimizerState).toBe(90e9);
  });

  it("gives every precision and optimizer pair its bytes per parameter", () => {
    const table = Object.fromEntries(
      PRECISIONS.flatMap((precision) =>
        OPTIMIZERS.map((optimizer) => [
          `${precision.id}/${optimizer.id}`,
          memoryBreakdown(1, precision.id, optimizer.id).totalBytesPerParameter,
        ]),
      ),
    );
    expect(table).toEqual({
      "fp32/sgd": 8,
      "fp32/momentum": 12,
      "fp32/adam": 16,
      "mixed/sgd": 8,
      "mixed/momentum": 12,
      "mixed/adam": 16,
    });
  });

  it("shows that 16-bit weights do not shrink Adam's model state, only move it into the optimizer", () => {
    const fp32 = memoryBreakdown(sizeOf("7.5b"), "fp32", "adam");
    const mixed = memoryBreakdown(sizeOf("7.5b"), "mixed", "adam");
    expect(fp32.total).toBe(mixed.total);
    expect(fp32.optimizerState / fp32.weights).toBe(2);
    expect(mixed.optimizerState / mixed.weights).toBe(6);
  });

  it("totals the 7.5 billion case under SGD", () => {
    // 2 + 2 weights and gradients, plus the 4-byte master copy: 8 bytes per parameter.
    expect(memoryBreakdown(sizeOf("7.5b"), "mixed", "sgd").total).toBe(60e9);
    expect(memoryBreakdown(sizeOf("7.5b"), "mixed", "sgd").optimizerState).toBe(30e9);
  });

  it("totals the other presets", () => {
    expect(memoryBreakdown(sizeOf("125m"), "mixed", "adam").total).toBe(2e9);
    expect(memoryBreakdown(sizeOf("70b"), "mixed", "adam").total).toBe(1.12e12);
    expect(memoryBreakdown(sizeOf("70b"), "mixed", "adam").optimizerState).toBe(840e9);
  });

  it("counts the lab's own model from the trainer's real buffers", () => {
    expect(sizeOf("lab")).toBe(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE);
    // The trainer holds the weights and one gradient buffer as Float32Array, and plain SGD keeps no state.
    expect(createTinyWeights().byteLength).toBe(3600);
    const lab = memoryBreakdown(sizeOf("lab"), "fp32", "sgd");
    expect(lab.weights).toBe(3600);
    expect(lab.gradients).toBe(3600);
    expect(lab.optimizerState).toBe(0);
    expect(lab.total).toBe(7200);
    expect(memoryBreakdown(sizeOf("lab"), "fp32", "adam").total).toBe(14400);
  });

  it("counts how many 32 GB devices the state alone needs", () => {
    expect(devicesNeeded(memoryBreakdown(sizeOf("7.5b"), "mixed", "adam").total)).toBe(4);
    expect(devicesNeeded(memoryBreakdown(sizeOf("1.5b"), "mixed", "adam").total)).toBe(1);
    expect(devicesNeeded(memoryBreakdown(sizeOf("70b"), "mixed", "adam").total)).toBe(35);
    expect(devicesNeeded(7200)).toBe(1);
  });

  it("formats decimal units without losing a trailing zero of a whole number", () => {
    expect(formatBytes(3600)).toBe("3.6 KB");
    expect(formatBytes(7200)).toBe("7.2 KB");
    expect(formatBytes(120e9)).toBe("120 GB");
    expect(formatBytes(15e9)).toBe("15 GB");
    expect(formatBytes(2e9)).toBe("2 GB");
    expect(formatBytes(1.12e12)).toBe("1.12 TB");
    expect(formatBytes(840e9)).toBe("840 GB");
    expect(formatBytes(24e9)).toBe("24 GB");
    expect(formatBytes(500)).toBe("500 B");
  });
});
