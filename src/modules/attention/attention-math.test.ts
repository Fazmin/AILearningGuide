import { describe, expect, it } from "vitest";
import {
  aggregateByWord,
  attentionMatrix,
  attentionRow,
  bidirectionalMatrix,
  causalMatrix,
  causalRow,
  characterLabel,
  fallbackVectors,
  headMatrix,
  largestAbsoluteDifference,
  laterShare,
  mixRow,
  saturatedRows,
  unownedShare,
  vectorLength,
  whitespaceShare,
  wordSpans,
  zeroValues,
} from "./attention-math";

const chars = (text: string) => Array.from(text);

describe("wordSpans", () => {
  it("gives each word the whitespace in front of it", () => {
    const spans = wordSpans(chars("  ab cd"));
    expect(spans).toEqual([
      { text: "ab", start: 0, wordStart: 2, end: 4, representative: 3 },
      { text: "cd", start: 4, wordStart: 5, end: 7, representative: 6 },
    ]);
  });

  it("leaves trailing whitespace unowned and caps the word count", () => {
    const spans = wordSpans(chars("a b c d "), 3);
    expect(spans.map((span) => span.text)).toEqual(["a", "b", "c"]);
    expect(spans[2]).toMatchObject({ start: 3, wordStart: 4, end: 5 });
  });

  it("returns no spans for whitespace-only text", () => {
    expect(wordSpans(chars("   "))).toEqual([]);
  });
});

describe("causal rows", () => {
  const text = chars("The cat sat on it");
  const queries = fallbackVectors(text, 0, 0, 0, 8);
  const keys = fallbackVectors(text, 0, 0, 1, 8);
  const matrix = causalMatrix(queries, keys);

  it("puts zero weight on later positions and sums to one", () => {
    matrix.forEach((row, position) => {
      expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
      row.slice(position + 1).forEach((value) => expect(value).toBe(0));
    });
  });

  it("matches a hand-computed softmax of q·k / √d", () => {
    const position = 5;
    const scaled = keys
      .slice(0, position + 1)
      .map(
        (key) =>
          key.reduce((sum, value, index) => sum + value * queries[position][index], 0) /
          Math.sqrt(8),
      );
    const peak = Math.max(...scaled);
    const exps = scaled.map((value) => Math.exp(value - peak));
    const total = exps.reduce((sum, value) => sum + value, 0);
    const row = causalRow(queries[position], keys, position);
    exps.forEach((value, index) => expect(row.weights[index]).toBeCloseTo(value / total, 12));
    expect(row.scale).toBeCloseTo(Math.sqrt(8), 12);
  });

  it("conserves every row's mass when aggregated to words", () => {
    const spans = wordSpans(text);
    spans.forEach((span) => {
      const words = aggregateByWord(matrix[span.representative], spans);
      expect(words.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    });
  });

  it("reports the whitespace share of a row", () => {
    const row = [0.1, 0.2, 0.3, 0.4];
    expect(whitespaceShare(row, chars("a bc"))).toBeCloseTo(0.2, 12);
    expect(whitespaceShare(row, chars("a b "))).toBeCloseTo(0.6, 12);
  });
});

describe("headMatrix", () => {
  it("reads one head from a [layers, batch, heads, sequence, width] tensor", () => {
    const shape = [2, 1, 3, 4, 2];
    const tensor = Array.from({ length: 2 * 3 * 4 * 2 }, (_, index) => index);
    const matrix = headMatrix(tensor, shape, 1, 2);
    // layer 1, head 2 starts at (1 * 3 + 2) * 4 * 2 = 40
    expect(matrix).toEqual([
      [40, 41],
      [42, 43],
      [44, 45],
      [46, 47],
    ]);
    expect(headMatrix(tensor, shape, 2, 0)).toBeUndefined();
    expect(headMatrix(tensor, [2, 3], 0, 0)).toBeUndefined();
  });
});

describe("small helpers", () => {
  it("mixes value vectors by weight", () => {
    expect(mixRow([0.25, 0.75], [[4, 0], [0, 8]])).toEqual([1, 6]);
  });

  it("measures the largest absolute gap", () => {
    expect(largestAbsoluteDifference([0.1, 0.5], [0.1, 0.25, 0.3])).toBeCloseTo(0.3, 12);
  });

  it("labels whitespace visibly", () => {
    expect(characterLabel(" ")).toBe("␣");
    expect(characterLabel("\n")).toBe("↵");
    expect(characterLabel("a")).toBe("a");
  });
});

describe("the mask, scaling and value switches", () => {
  const text = chars("The cat sat on it");
  const queries = fallbackVectors(text, 0, 0, 0, 8);
  const keys = fallbackVectors(text, 0, 0, 1, 8);
  const values = fallbackVectors(text, 0, 0, 2, 8);

  it("attentionRow with the model's own settings is exactly causalRow", () => {
    queries.forEach((query, position) => {
      const row = attentionRow(query, keys, position);
      const reference = causalRow(query, keys, position);
      expect(row.weights).toEqual(reference.weights);
      expect(row.scale).toBe(reference.scale);
    });
    expect(attentionMatrix(queries, keys)).toEqual(causalMatrix(queries, keys));
  });

  it("lifting the mask reads every position, sums to one, and gives later positions weight", () => {
    const open = bidirectionalMatrix(queries, keys);
    open.forEach((row, position) => {
      expect(row).toHaveLength(text.length);
      expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
      if (position < text.length - 1) expect(laterShare(row, position)).toBeGreaterThan(0);
    });
    expect(laterShare(open[text.length - 1], text.length - 1)).toBe(0);
  });

  it("lifting the mask scales every earlier weight by the same factor, one minus the later share", () => {
    const causal = causalMatrix(queries, keys);
    const open = bidirectionalMatrix(queries, keys);
    open.forEach((row, position) => {
      const factor = 1 - laterShare(row, position);
      for (let key = 0; key <= position; key += 1) expect(row[key]).toBeCloseTo(causal[position][key] * factor, 12);
    });
  });

  it("switching the scaling off is softmax of the raw dot products, which is softmax of the scaled scores times the square root of the width", () => {
    const position = 9;
    const scaled = attentionRow(queries[position], keys, position, { scaled: true });
    const raw = attentionRow(queries[position], keys, position, { scaled: false });
    expect(raw.scale).toBe(1);
    expect(scaled.scale).toBeCloseTo(Math.sqrt(8), 12);
    for (let key = 0; key <= position; key += 1) {
      expect(raw.scaledScores[key]).toBeCloseTo(scaled.scaledScores[key] * Math.sqrt(8), 12);
    }
    const peak = Math.max(...raw.scaledScores.slice(0, position + 1));
    const exps = raw.scaledScores.slice(0, position + 1).map((value) => Math.exp(value - peak));
    const total = exps.reduce((sum, value) => sum + value, 0);
    exps.forEach((value, key) => expect(raw.weights[key]).toBeCloseTo(value / total, 12));
    // A larger multiplier never lowers the largest weight.
    expect(Math.max(...raw.weights)).toBeGreaterThanOrEqual(Math.max(...scaled.weights));
  });

  it("combines the two switches: no mask and no scaling reads every position on raw scores", () => {
    const row = attentionRow(queries[3], keys, 3, { causal: false, scaled: false });
    expect(row.weights).toHaveLength(text.length);
    expect(row.scaledScores.every(Number.isFinite)).toBe(true);
    expect(row.weights.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
  });

  it("zeroes a span of value vectors and leaves the others and the original array alone", () => {
    const zeroed = zeroValues(values, 4, 7);
    zeroed.forEach((vector, position) => {
      if (position >= 4 && position < 7) expect(vector.every((value) => value === 0)).toBe(true);
      else expect(vector).toEqual(values[position]);
    });
    expect(values[5].some((value) => value !== 0)).toBe(true);
    expect(zeroed[5]).toHaveLength(values[5].length);
  });

  it("a mix of zeroed values is the zero vector whatever the weights are; the weights never read V", () => {
    const weights = causalRow(queries[9], keys, 9).weights;
    expect(vectorLength(mixRow(weights, zeroValues(values, 0, values.length)))).toBe(0);
    expect(vectorLength(mixRow(weights, values))).toBeGreaterThan(0);
    // Zeroing a span removes exactly that span's weight-times-value share.
    const full = mixRow(weights, values);
    const without = mixRow(weights, zeroValues(values, 4, 7));
    full.forEach((value, dimension) => {
      let share = 0;
      for (let key = 4; key < 7; key += 1) share += weights[key] * values[key][dimension];
      expect(value - without[dimension]).toBeCloseTo(share, 12);
    });
  });

  it("counts saturated rows, skipping a causal first row that is 100% by construction", () => {
    expect(saturatedRows([[1], [0.6, 0.4], [0.995, 0.003, 0.002]], true)).toEqual({ saturated: 1, total: 2 });
    expect(saturatedRows([[1, 0], [0.6, 0.4]], false)).toEqual({ saturated: 1, total: 2 });
    expect(saturatedRows([], true)).toEqual({ saturated: 0, total: 0 });
  });

  it("reports the weight a lifted row puts outside every word", () => {
    const spans = wordSpans(chars("a b "));
    expect(unownedShare([0.25, 0.25, 0.25, 0.25], spans)).toBeCloseTo(0.25, 12);
    expect(unownedShare([0.5, 0.5], wordSpans(chars("a b")))).toBe(0);
  });

  it("measures a vector's length", () => {
    expect(vectorLength([3, 4])).toBe(5);
    expect(vectorLength([])).toBe(0);
  });
});
