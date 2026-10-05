import { describe, expect, it } from "vitest";
import {
  add,
  blockParameters,
  createBlockWeights,
  crossPositionShift,
  gelu,
  HIDDEN,
  layerNorm,
  mean,
  norm,
  embed,
  ORDER_IDS,
  ORDER_TOLERANCE,
  ORDERS,
  orderTest,
  positionVector,
  rmsNorm,
  runBlock,
  sequenceFor,
  sequenceWithPosition,
  swigluHidden,
  tokenVector,
  TOKENS,
  WIDTH,
  type BlockOptions,
} from "./block";

const weights = createBlockWeights();
const on: BlockOptions = { norm: true, attention: true, mlp: true, placement: "pre" };
const ones = Array.from({ length: WIDTH }, () => 1);
const zeros = Array.from({ length: WIDTH }, () => 0);

describe("normalisation", () => {
  it("LayerNorm gives zero mean and unit variance with unit scale", () => {
    const out = layerNorm([1, 2, 3, 4, 5, 6, 7, 20], ones, zeros);
    expect(mean(out)).toBeCloseTo(0, 10);
    expect(mean(out.map((value) => value * value))).toBeCloseTo(1, 4);
  });

  it("RMSNorm gives unit root-mean-square without removing the mean", () => {
    const vector = [1, 2, 3, 4, 5, 6, 7, 20];
    const out = rmsNorm(vector, ones);
    expect(Math.sqrt(mean(out.map((value) => value * value)))).toBeCloseTo(1, 4);
    expect(mean(out)).toBeGreaterThan(0.5);
  });

  it("GELU matches reference values of the tanh approximation", () => {
    expect(gelu(0)).toBe(0);
    expect(gelu(1)).toBeCloseTo(0.841192, 5);
    expect(gelu(-1)).toBeCloseTo(-0.158808, 5);
  });
});

describe("pre-norm block", () => {
  it("is exactly input + attention write + MLP write", () => {
    const trace = runBlock(sequenceFor(TOKENS, 1), weights, on);
    trace.output.forEach((vector, position) => {
      const sum = add(add(trace.input[position], trace.attentionWrite[position]), trace.mlpWrite[position]);
      vector.forEach((value, index) => expect(value).toBeCloseTo(sum[index], 12));
    });
    expect(trace.mlpHidden[0]).toHaveLength(HIDDEN);
  });

  it("ablating both sublayers passes the stream through unchanged", () => {
    const input = sequenceFor(TOKENS, 1);
    const trace = runBlock(input, weights, { ...on, attention: false, mlp: false });
    trace.output.forEach((vector, position) =>
      vector.forEach((value, index) => expect(value).toBeCloseTo(input[position][index], 12)),
    );
  });

  it("with the norm on, the attention write ignores the input's scale", () => {
    const small = runBlock(sequenceFor(TOKENS, 1), weights, on).attentionWrite[5];
    const large = runBlock(sequenceFor(TOKENS, 8), weights, on).attentionWrite[5];
    large.forEach((value, index) => expect(value).toBeCloseTo(small[index], 3));
  });

  it("without the norm, large inputs saturate attention", () => {
    const trace = runBlock(sequenceFor(TOKENS, 8), weights, { ...on, norm: false });
    expect(Math.max(...trace.headWeights[5][0])).toBeGreaterThan(0.95);
  });

  it("attention weights are causal and sum to one per head", () => {
    const trace = runBlock(sequenceFor(TOKENS, 1), weights, on);
    trace.headWeights.forEach((heads, position) =>
      heads.forEach((row) => {
        expect(row).toHaveLength(position + 1);
        expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
      }),
    );
  });
});

describe("cross-position flow", () => {
  it("only attention lets other tokens change a position's output", () => {
    expect(crossPositionShift(weights, on, 1, 5)).toBeGreaterThan(0.1);
    expect(crossPositionShift(weights, { ...on, attention: false }, 1, 5)).toBe(0);
    expect(crossPositionShift(weights, on, 1, 0)).toBe(0);
  });
});

describe("post-norm block", () => {
  it("renormalises the stream, so the output norm is √d whatever the input scale", () => {
    for (const inputScale of [0.25, 1, 8]) {
      const trace = runBlock(sequenceFor(TOKENS, inputScale), weights, { ...on, placement: "post" });
      expect(norm(trace.output[3])).toBeCloseTo(Math.sqrt(WIDTH), 3);
    }
  });
});

describe("parameter accounting", () => {
  it("a classic GELU block is 12·d² weights plus norms", () => {
    const counts = blockParameters(768, "gelu", "layernorm");
    expect(counts.attention).toBe(4 * 768 * 768);
    expect(counts.mlp).toBe(8 * 768 * 768);
    expect(counts.total - counts.norms).toBe(counts.twelveDSquared);
  });

  it("GPT-2 small adds up to its published 124,439,808 parameters", () => {
    const block = blockParameters(768, "gelu", "layernorm", true).total;
    const embeddings = 50257 * 768 + 1024 * 768;
    expect(12 * block + embeddings + 2 * 768).toBe(124_439_808);
  });

  it("Llama 2 7B adds up to its published 6,738,415,616 parameters", () => {
    expect(swigluHidden(4096)).toBe(11008);
    const block = blockParameters(4096, "swiglu", "rmsnorm").total;
    expect(32 * block + 2 * 32000 * 4096 + 4096).toBe(6_738_415_616);
  });
});

/**
 * Figures quoted by the Position card, its card-info entry and both lessons
 * (content/standard.mdx, content/plain.mdx). They come from the seeded block, so
 * they change only if block.ts, the seed, or the embedding changes.
 */
describe("position: does the block notice word order?", () => {
  const distance = (left: number[], right: number[]) => norm(left.map((value, index) => value - right[index]));

  it("builds the lab's embedding from a token vector plus a position vector", () => {
    const expected = tokenVector("cat").map((value, index) => value + positionVector(3)[index]);
    embed("cat", 3).forEach((value, index) => expect(value).toBe(expected[index]));
    // Same token, same vector, wherever it sits: the position vector is what differs.
    expect(distance(tokenVector("the"), tokenVector("the"))).toBe(0);
    expect(distance(embed("the", 0), embed("the", 4))).toBeGreaterThan(0.1);
  });

  it("defines every reordering as a permutation of the six words", () => {
    for (const id of ORDER_IDS) {
      expect([...ORDERS[id]].sort()).toEqual([0, 1, 2, 3, 4, 5]);
    }
    expect(orderTest(createBlockWeights(), on, 1, "swap", true).reordered.join(" ")).toBe("the mat sat on the cat");
    expect(orderTest(createBlockWeights(), on, 1, "shuffle", true).reordered.join(" ")).toBe("mat on cat the the sat");
  });

  it("without position information, reordering only reorders the outputs, at every setting", () => {
    for (const order of ORDER_IDS) {
      for (const placement of ["pre", "post"] as const) {
        for (const inputScale of [0.25, 1, 8]) {
          for (const toggles of [{}, { attention: false }, { mlp: false }, { norm: false }]) {
            const result = orderTest(weights, { ...on, ...toggles, placement }, inputScale, order, false);
            expect(result.largestShift, `${order} ${placement} ${inputScale} ${JSON.stringify(toggles)}`).toBeLessThan(ORDER_TOLERANCE);
            expect(result.changed).toBe(0);
          }
        }
      }
    }
  });

  it("with position information, every word's output depends on where it sits", () => {
    const swap = orderTest(weights, on, 1, "swap", true);
    expect(swap.changed).toBe(6);
    // Quoted in the lessons and the Position card: swapping cat and mat, default settings.
    expect(swap.rows.map((row) => Number(row.shift.toFixed(3)))).toEqual([0.391, 1.565, 0.414, 0.465, 0.389, 1.286]);
    expect(swap.largestShift).toBeCloseTo(1.565, 3);
    const shuffle = orderTest(weights, on, 1, "shuffle", true);
    expect(shuffle.changed).toBe(6);
    expect(shuffle.largestShift).toBeCloseTo(1.391, 3);
  });

  it("keeps the original order as a zero baseline", () => {
    expect(orderTest(weights, on, 1, "original", true).largestShift).toBe(0);
  });

  it("with attention off, position information only changes the words that moved", () => {
    const rows = orderTest(weights, { ...on, attention: false }, 1, "swap", true).rows;
    expect(rows.map((row) => row.shift > ORDER_TOLERANCE)).toEqual([false, true, false, false, false, true]);
  });

  it("the causal mask leaks a little order, which is why the card lifts it", () => {
    // Without position information and without the mask, the first and second "the" get identical outputs.
    const open = runBlock(sequenceWithPosition(TOKENS, 1, false), weights, { ...on, causal: false }).output;
    expect(distance(open[0], open[4])).toBeLessThan(ORDER_TOLERANCE);
    // With the mask, the first "the" sees only itself and the second sees five words, so they differ.
    const masked = runBlock(sequenceWithPosition(TOKENS, 1, false), weights, on).output;
    expect(distance(masked[0], masked[4])).toBeGreaterThan(3);
    // Yet the last word still cannot see the order of the words before it...
    const before = ["the", "cat", "sat", "on", "the", "mat"];
    const after = ["on", "the", "the", "sat", "cat", "mat"];
    const last = (words: string[], usePosition: boolean) =>
      runBlock(sequenceWithPosition(words, 1, usePosition), weights, on).output[5];
    expect(distance(last(before, false), last(after, false))).toBeLessThan(ORDER_TOLERANCE);
    // ...unless position information is on.
    expect(distance(last(before, true), last(after, true))).toBeGreaterThan(0.1);
  });

  it("leaves the causal block's numbers untouched by the new option", () => {
    const implicit = runBlock(sequenceFor(TOKENS, 1), weights, on);
    const explicit = runBlock(sequenceFor(TOKENS, 1), weights, { ...on, causal: true });
    expect(explicit.output).toEqual(implicit.output);
    expect(implicit.headWeights.map((heads) => heads[0].length)).toEqual([1, 2, 3, 4, 5, 6]);
    const open = runBlock(sequenceFor(TOKENS, 1), weights, { ...on, causal: false });
    expect(open.headWeights.every((heads) => heads.every((row) => row.length === TOKENS.length))).toBe(true);
  });
});
