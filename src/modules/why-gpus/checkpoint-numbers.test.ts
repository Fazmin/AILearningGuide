import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import { evaluate, PRECISIONS, precisionById, ridgePoint, whatIf, workload } from "./roofline";
import definition from "./module";

/**
 * Pins the comparisons the checkpoint questions rely on. Lab defaults: matmul, batch 1,
 * width 4096, fp16, 2000 GB/s.
 */
const fp16 = precisionById("fp16");
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

describe("why-gpus checkpoint questions", () => {
  it("question 1: a batch-1 matmul exposes 4,096 independent dots of length 4096 yet uses under 1% of peak", () => {
    const work = workload("matmul", 1, 4096, fp16.bytes);
    expect(work.outputs).toBe(4096);
    expect(work.dotLength).toBe(4096);
    expect(work.outputs.toLocaleString("en-US")).toBe("4,096");
    const { time } = evaluate({ operation: "matmul", batch: 1, width: 4096, precision: fp16, bandwidth: 2000 });
    expect(time.utilization).toBeLessThan(0.01);
    expect(correctOption(0)).toMatch(/^Each output is its own dot product/);
  });

  it("question 2: more rows per weight fetch or fewer bytes help at batch 1, while doubling peak does nothing", () => {
    const base = { operation: "matmul" as const, batch: 1, width: 4096, precision: fp16, bandwidth: 2000 };
    const start = evaluate(base);
    expect(start.time.bound).toBe("memory");
    expect(evaluate({ ...base, batch: 8 }).work.intensity).toBeGreaterThan(5 * start.work.intensity);
    expect(evaluate({ ...base, batch: 8 }).time.attainableFlops).toBeGreaterThan(5 * start.time.attainableFlops);
    const options = whatIf(base);
    expect(options.doublePeak).toBeCloseTo(options.base, 12);
    expect(options.smallerPrecision!).toBeLessThan(options.base);
    expect(correctOption(1)).toMatch(/^Reusing each fetched weight for more rows/);
  });

  it("question 3: the 1024-row matmul is compute-bound at 2000 GB/s and memory-bound at 100 GB/s because the ridge moves past its intensity", () => {
    const base = { operation: "matmul" as const, batch: 1024, width: 4096, precision: fp16 };
    const intensity = workload("matmul", 1024, 4096, fp16.bytes).intensity;
    expect(ridgePoint(fp16.peakTflops * 1e12, 2000e9)).toBeLessThan(intensity);
    expect(ridgePoint(fp16.peakTflops * 1e12, 100e9)).toBeGreaterThan(intensity);
    expect(evaluate({ ...base, bandwidth: 2000 }).time.bound).toBe("compute");
    expect(evaluate({ ...base, bandwidth: 100 }).time.bound).toBe("memory");
    // The operation's arithmetic did not change when only bandwidth moved.
    expect(evaluate({ ...base, bandwidth: 100 }).work.flops).toBe(evaluate({ ...base, bandwidth: 2000 }).work.flops);
    expect(correctOption(2)).toMatch(/^Memory: the ridge moves right past the dot/);
  });

  it("question 4: an elementwise add stays memory-bound at every batch and precision, with intensity independent of the batch", () => {
    for (const precision of PRECISIONS) {
      const intensities = [1, 64, 4096].map((batch) => workload("add", batch, 4096, precision.bytes).intensity);
      for (const intensity of intensities) expect(intensity).toBeCloseTo(1 / (3 * precision.bytes), 12);
      for (const batch of [1, 64, 4096]) {
        const result = evaluate({ operation: "add", batch, width: 4096, precision, bandwidth: 2000 });
        expect(result.time.bound).toBe("memory");
      }
    }
    expect(correctOption(3)).toMatch(/^Each value is read, added once, and written/);
  });

  it("question 5: every readout is the best case, since time is the larger of two clocks with perfect overlap", () => {
    const result = evaluate({ operation: "matmul", batch: 1024, width: 4096, precision: fp16, bandwidth: 2000 });
    expect(result.time.seconds).toBe(Math.max(result.time.computeSeconds, result.time.memorySeconds));
    expect(result.time.utilization).toBeLessThanOrEqual(1);
    expect(correctOption(4)).toMatch(/^As a ceiling/);
  });
});
