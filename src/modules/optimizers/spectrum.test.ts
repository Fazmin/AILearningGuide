import { describe, expect, it } from "vitest";
import { newtonSchulz, tinyRandom } from "@app/module-sdk";
import { captureMuon } from "./race";
import {
  NEWTON_SCHULZ_COEFFICIENTS,
  newtonSchulzScalar,
  singularValues,
  summarizeSpectrum,
  symmetricEigenvalues,
  ZERO_SINGULAR,
} from "./spectrum";

const randomMatrix = (rows: number, cols: number, seed: number) => {
  const random = tinyRandom(seed);
  return Float64Array.from({ length: rows * cols }, () => random() - 0.5);
};

describe("Jacobi eigenvalues", () => {
  it("solves small symmetric matrices exactly", () => {
    const pair = symmetricEigenvalues([2, 1, 1, 2], 2);
    expect(pair[0]).toBeCloseTo(3, 12);
    expect(pair[1]).toBeCloseTo(1, 12);
    expect(symmetricEigenvalues([4, 0, 0, 0, 9, 0, 0, 0, 1], 3)).toEqual([9, 4, 1]);
  });

  it("keeps the trace and returns eigenvalues largest first", () => {
    const m = randomMatrix(8, 8, 3);
    const symmetric = Float64Array.from({ length: 64 }, (_, index) => {
      const row = Math.floor(index / 8);
      const col = index % 8;
      return m[row * 8 + col] + m[col * 8 + row];
    });
    const values = symmetricEigenvalues(symmetric, 8);
    const trace = Array.from({ length: 8 }, (_, index) => symmetric[index * 9]).reduce((sum, value) => sum + value, 0);
    expect(values.reduce((sum, value) => sum + value, 0)).toBeCloseTo(trace, 10);
    expect(values).toEqual([...values].sort((left, right) => right - left));
  });
});

describe("singular values", () => {
  it("reads a diagonal matrix's diagonal, zero included", () => {
    const source = new Float64Array(6 * 4);
    [5, 1, 0.2, 0].forEach((value, index) => (source[index * 4 + index] = value));
    const values = singularValues(source, 6, 4);
    [5, 1, 0.2, 0].forEach((value, index) => expect(values[index]).toBeCloseTo(value, 10));
  });

  it("gives a rank-one matrix one singular value, the product of its two lengths", () => {
    const u = [1, 2, 2];
    const v = [3, 4];
    const values = singularValues(u.flatMap((a) => v.map((b) => a * b)), 3, 2);
    expect(values[0]).toBeCloseTo(3 * 5, 10);
    expect(values[1]).toBeLessThan(1e-7);
  });

  it("keeps the Frobenius norm: the squares of the singular values add up to the sum of squares", () => {
    const m = randomMatrix(30, 30, 11);
    const values = singularValues(m, 30, 30);
    const frobenius = m.reduce((sum, value) => sum + value * value, 0);
    expect(values.reduce((sum, value) => sum + value * value, 0)).toBeCloseTo(frobenius, 8);
  });

  it("reads an orthogonal matrix as all ones", () => {
    const angle = 0.7;
    const values = singularValues([Math.cos(angle), -Math.sin(angle), Math.sin(angle), Math.cos(angle)], 2, 2);
    expect(values[0]).toBeCloseTo(1, 12);
    expect(values[1]).toBeCloseTo(1, 12);
  });
});

describe("Newton–Schulz, read through the singular values", () => {
  it("uses Muon's quintic coefficients and sends zero to zero", () => {
    expect(NEWTON_SCHULZ_COEFFICIENTS).toEqual([3.4445, -4.775, 2.0315]);
    expect(newtonSchulzScalar(0, 5)).toBe(0);
    expect(newtonSchulzScalar(0.3, 0)).toBe(0.3);
    // A tiny value grows by about 3.4445 per pass.
    expect(newtonSchulzScalar(1e-9, 5) / 1e-9).toBeCloseTo(3.4445 ** 5, 3);
  });

  it("never pushes a value from (0, 1] above about 1.2026, the polynomial's peak", () => {
    let highest = 0;
    for (let index = 1; index <= 2000; index += 1) {
      for (const steps of [1, 2, 5, 8]) highest = Math.max(highest, newtonSchulzScalar(index / 2000, steps));
    }
    expect(highest).toBeGreaterThan(1.2);
    expect(highest).toBeLessThan(1.2027);
  });

  it("matches the matrix iteration on the captured Muon direction, value for value", () => {
    for (const capture of ["first", "middle"] as const) {
      const captured = captureMuon(0.05, capture)!;
      const before = singularValues(newtonSchulz(captured.direction, 30, 30, 0), 30, 30);
      for (const steps of [1, 3, 5, 8]) {
        const after = singularValues(newtonSchulz(captured.direction, 30, 30, steps), 30, 30);
        // An exact zero reads as about 1e-9 through MᵀM; the matrix iteration keeps it at exactly zero.
        const mapped = before
          .map((value) => newtonSchulzScalar(value <= ZERO_SINGULAR ? 0 : value, steps))
          .sort((left, right) => right - left);
        after.forEach((value, index) => expect(Math.abs(value - mapped[index])).toBeLessThan(1e-5));
      }
    }
  });

  it("counts zero directions, nonzero ones and those inside the band", () => {
    const summary = summarizeSpectrum([1.1, 0.7, 0.3, 1e-4, 1e-9, 0]);
    expect(summary).toMatchObject({ nonzero: 4, zeros: 2, inBand: 2, largest: 1.1, smallestNonzero: 1e-4 });
    expect(ZERO_SINGULAR).toBe(1e-7);
  });
});
