import { describe, expect, it } from "vitest";
import {
  cosine,
  dareApply,
  dropMask,
  mergeTaskVectors,
  norm,
  pickCoordinates,
  signStats,
  slerp,
  taskVector,
  tiesMerge,
  trimTopK,
  type MergeParams,
} from "./merge";
import { buildTaskVectors, lossesFor } from "./lab";

const vec = (...values: number[]) => Float32Array.from(values);
const close = (actual: ArrayLike<number>, expected: number[], digits = 6) =>
  expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, digits));

const params: MergeParams = { mix: 0.5, lambda: 1, density: 0.6, dropRate: 0.5, maskSeed: 1 };

describe("task vectors", () => {
  it("are the tuned weights minus the base weights", () => {
    close(taskVector(vec(1, 2, 3), vec(0.5, 2, 4)), [0.5, 0, -1]);
  });

  it("counts overlaps and sign conflicts", () => {
    const stats = signStats(vec(1, -1, 0, 2), vec(2, 1, 3, 0));
    expect(stats).toEqual({ overlap: 2, conflicts: 1, onlyA: 1, onlyB: 1 });
    expect(cosine(vec(1, 0), vec(0, 1))).toBeCloseTo(0, 12);
    expect(Number.isNaN(cosine(vec(0, 0), vec(1, 1)))).toBe(true);
  });
});

describe("TIES-Merging on explicit five-entry vectors", () => {
  const a = vec(0.9, -0.2, 0.5, 0.05, -0.6);
  const b = vec(-0.7, -0.4, 0.4, 0.8, 0.1);

  it("trims each vector to its largest entries", () => {
    close(trimTopK(a, 0.6), [0.9, 0, 0.5, 0, -0.6]);
    // |−0.4| and |0.4| tie; the lower index wins so exactly three survive.
    close(trimTopK(b, 0.6), [-0.7, -0.4, 0, 0.8, 0]);
  });

  it("elects the sign with more mass and averages only agreeing entries", () => {
    const { merged, elected } = tiesMerge([a, b], 0.6);
    // Entry 0 conflicts: +0.9 against −0.7 sums positive, so −0.7 is discarded.
    expect(Array.from(elected)).toEqual([1, -1, 1, 1, -1]);
    close(merged, [0.9, -0.4, 0.5, 0.8, -0.6]);
  });

  it("differs from the plain average exactly where trimming or signs intervene", () => {
    const average = mergeTaskVectors("linear", a, b, params)!.vector;
    close(average, [0.1, -0.3, 0.45, 0.425, -0.25]);
  });

  it("with density 1 and no conflicts, reduces to the mean of the two", () => {
    const { merged } = tiesMerge([vec(1, 2), vec(3, 4)], 1);
    close(merged, [2, 3]);
  });
});

describe("DARE", () => {
  it("drops entries and rescales the survivors by 1 / (1 − p)", () => {
    const mask = dropMask(8, 0.5, 3);
    const out = dareApply(vec(1, 1, 1, 1, 1, 1, 1, 1), mask, 0.5);
    out.forEach((value, index) => expect(value).toBe(mask[index] ? 2 : 0));
  });

  it("keeps each entry's expected value", () => {
    const source = vec(0.3, -1.2, 0.8);
    const totals = [0, 0, 0];
    const trials = 4000;
    for (let seed = 0; seed < trials; seed += 1) {
      const out = dareApply(source, dropMask(3, 0.7, seed), 0.7);
      out.forEach((value, index) => (totals[index] += value));
    }
    totals.forEach((total, index) => expect(total / trials).toBeCloseTo(source[index], 1));
  });
});

describe("SLERP", () => {
  it("keeps the norm that linear interpolation loses between orthogonal vectors", () => {
    const result = slerp(vec(1, 0), vec(0, 1), 0.5)!;
    expect(result.angle).toBeCloseTo(90, 6);
    close(result.vector, [Math.SQRT1_2, Math.SQRT1_2]);
    expect(norm(result.vector)).toBeCloseTo(1, 6);
    expect(norm(mergeTaskVectors("linear", vec(1, 0), vec(0, 1), params)!.vector)).toBeCloseTo(
      Math.SQRT1_2,
      6,
    );
  });

  it("returns the endpoints at t = 0 and t = 1 and refuses a zero vector", () => {
    close(slerp(vec(2, 1), vec(-1, 3), 0)!.vector, [2, 1]);
    close(slerp(vec(2, 1), vec(-1, 3), 1)!.vector, [-1, 3]);
    expect(slerp(vec(0, 0), vec(1, 0), 0.5)).toBeNull();
  });
});

describe("merging with the other task removed", () => {
  it("isolates each method's own shrinkage from interference", () => {
    const a = vec(0.9, -0.2, 0.5);
    const zero = vec(0, 0, 0);
    close(mergeTaskVectors("linear", a, zero, { ...params, mix: 0.3 })!.vector, [0.63, -0.14, 0.35]);
    close(mergeTaskVectors("task", a, zero, { ...params, lambda: 0.8 })!.vector, [0.72, -0.16, 0.4]);
    close(mergeTaskVectors("ties", a, zero, { ...params, density: 2 / 3 })!.vector, [0.9, 0, 0.5]);
    expect(mergeTaskVectors("slerp", a, zero, params)).toBeNull();
  });

  it("picks the largest conflicts first, then the largest agreements", () => {
    const picked = pickCoordinates(vec(1, -2, 0.1, 3), vec(-1, -1, 0.2, 0), 1);
    expect(picked).toEqual([
      { index: 0, conflict: true },
      { index: 1, conflict: false },
    ]);
  });
});

describe("merging real task vectors from the tiny language model", () => {
  const vectors = buildTaskVectors("full", 20, 20);

  it("starts every fine-tune from the same base, so task vectors share a coordinate system", () => {
    expect(vectors.taskA.length).toBe(900);
    expect(norm(vectors.taskA)).toBeGreaterThan(0);
    expect(cosine(vectors.taskA, vectors.taskB)).toBeGreaterThan(0);
  });

  it("makes averaging worse than each specialist, mostly by halving each vector", () => {
    const merged = lossesFor(vectors, "linear", params);
    expect(merged.recipes).toBeGreaterThan(vectors.specialistA.recipes);
    expect(merged.proverbs).toBeGreaterThan(vectors.specialistB.proverbs);
    // Removing the other task barely changes the average: little interference.
    expect(Math.abs(merged.interferenceA)).toBeLessThan(0.1);
    expect(merged.recipes - vectors.specialistA.recipes).toBeGreaterThan(0.1);
  });

  it("shows interference when both full vectors are added", () => {
    const summed = lossesFor(vectors, "task", { ...params, lambda: 1 });
    expect(summed.interferenceA).toBeGreaterThan(0.05);
    expect(summed.interferenceB).toBeGreaterThan(0.05);
  });

  it("collapses under DARE at a high drop rate because these deltas are not redundant", () => {
    const light = lossesFor(vectors, "dare", { ...params, dropRate: 0.2 });
    const heavy = lossesFor(vectors, "dare", { ...params, dropRate: 0.9 });
    expect(heavy.recipes).toBeGreaterThan(light.recipes + 1);
  });

  it("leaves the base untouched with zero training epochs", () => {
    const untrained = buildTaskVectors("lora", 0, 0);
    expect(norm(untrained.taskA)).toBe(0);
    expect(untrained.specialistA.recipes).toBeCloseTo(untrained.base.recipes, 10);
  });
});
