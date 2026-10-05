import { describe, expect, it } from "vitest";
import {
  attainable,
  batchAtRidge,
  evaluate,
  formatSeconds,
  precisionById,
  ridgePoint,
  whatIf,
  workload,
} from "./roofline";

const fp16 = precisionById("fp16");

describe("workload counts", () => {
  it("counts a square matmul exactly like the classic 2n³ / 3n² model", () => {
    const n = 1024;
    const work = workload("matmul", n, n, 2);
    expect(work.flops).toBe(2 * n ** 3);
    expect(work.bytes).toBe(3 * n * n * 2);
  });

  it("gives a batch-1 matmul about one FLOP per fp16 byte", () => {
    const work = workload("matmul", 1, 4096, 2);
    expect(work.flops).toBe(2 * 4096 * 4096);
    expect(work.intensity).toBeCloseTo(1, 2);
  });

  it("approaches 2·batch / bytes when batch is much smaller than width", () => {
    const work = workload("matmul", 8, 16384, 2);
    expect(work.intensity).toBeCloseTo((2 * 8) / 2, 1);
  });

  it("keeps an elementwise add at 1 / (3·bytes) whatever the batch", () => {
    for (const batch of [1, 64, 4096]) {
      expect(workload("add", batch, 4096, 2).intensity).toBeCloseTo(1 / 6, 10);
    }
  });
});

describe("roofline", () => {
  it("puts the ridge at peak / bandwidth", () => {
    expect(ridgePoint(300e12, 2000e9)).toBeCloseTo(150, 10);
    expect(attainable(300e12, 2000e9, 1)).toBeCloseTo(2e12, 0);
    expect(attainable(300e12, 2000e9, 1000)).toBe(300e12);
  });

  it("marks batch-1 fp16 decode-like work as memory-bound at under 1% of peak", () => {
    const { time, ridge } = evaluate({ operation: "matmul", batch: 1, width: 4096, precision: fp16, bandwidth: 2000 });
    expect(ridge).toBeCloseTo(150, 6);
    expect(time.bound).toBe("memory");
    expect(time.memorySeconds * 1e6).toBeCloseTo(16.8, 1);
    expect(time.utilization).toBeLessThan(0.01);
  });

  it("flips a 1024-row matmul to compute-bound, and back to memory-bound at 100 GB/s", () => {
    const base = { operation: "matmul" as const, batch: 1024, width: 4096, precision: fp16 };
    expect(evaluate({ ...base, bandwidth: 2000 }).time.bound).toBe("compute");
    expect(evaluate({ ...base, bandwidth: 100 }).time.bound).toBe("memory");
  });

  it("solves for the batch that reaches the ridge", () => {
    const rows = batchAtRidge(4096, 2, 150);
    expect(rows).toBeCloseTo(161.85, 1);
    const at = workload("matmul", rows, 4096, 2).intensity;
    expect(at).toBeCloseTo(150, 6);
    expect(batchAtRidge(256, 2, 150)).toBe(Number.POSITIVE_INFINITY);
  });

  it("shows that doubling peak does nothing for memory-bound work while doubling bandwidth halves it", () => {
    const scenario = { operation: "matmul" as const, batch: 1, width: 4096, precision: fp16, bandwidth: 2000 };
    const result = whatIf(scenario);
    expect(result.doublePeak).toBeCloseTo(result.base, 12);
    expect(result.doubleBandwidth).toBeCloseTo(result.base / 2, 12);
    expect(result.smallerLabel).toBe("int8");
    expect(result.smallerPrecision!).toBeCloseTo(result.base / 2, 7);
  });
});

describe("formatting", () => {
  it("chooses a readable unit", () => {
    expect(formatSeconds(16.785e-6)).toBe("16.8 µs");
    expect(formatSeconds(114.5e-6)).toBe("115 µs");
    expect(formatSeconds(2.5e-3)).toBe("2.50 ms");
    expect(formatSeconds(1.12e-7)).toBe("112.0 ns");
  });
});
