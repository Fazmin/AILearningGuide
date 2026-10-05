import { describe, expect, it } from "vitest";
import vocabulary from "./assets/transformer-vocab.json";
import { DEFAULT_PROMPT } from "./prompts";
import {
  displayToken,
  drawCandidate,
  encode,
  entropyBits,
  decodeDraw,
  encodeDraw,
  logSoftmax,
  seededUniform,
  shapeDistribution,
  softmax,
  surprisalBits,
  tokenText,
} from "./sampling";

const logits = [2, 1, 0.5, 0, -1, -3];
const off = { temperature: 1, topK: 66, topP: 1 };

describe("softmax and temperature", () => {
  it("sums to one and is shift invariant", () => {
    const p = softmax(logits);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    const shifted = softmax(logits.map((value) => value + 100));
    p.forEach((value, index) => expect(shifted[index]).toBeCloseTo(value, 12));
  });

  it("recovers a probability table from its logarithm", () => {
    const table = [0.5, 0.25, 0.125, 0.125];
    softmax(table.map(Math.log)).forEach((value, index) => expect(value).toBeCloseTo(table[index], 12));
  });

  it("sharpens below one, flattens above one, and never reorders", () => {
    const cold = shapeDistribution(logits, { ...off, temperature: 0.25 });
    const hot = shapeDistribution(logits, { ...off, temperature: 2 });
    const base = shapeDistribution(logits, off);
    expect(entropyBits(cold.q)).toBeLessThan(entropyBits(base.q));
    expect(entropyBits(hot.q)).toBeGreaterThan(entropyBits(base.q));
    expect(cold.candidates.map((c) => c.index)).toEqual(hot.candidates.map((c) => c.index));
  });

  it("keeps log-softmax finite for extreme logits", () => {
    const values = logSoftmax([1000, 0, -1000], 0.1);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values[0]).toBeCloseTo(0, 12);
  });
});

describe("truncation", () => {
  it("top-k keeps exactly k candidates and renormalises them", () => {
    const shaped = shapeDistribution(logits, { ...off, topK: 2 });
    expect(shaped.keptCount).toBe(2);
    const p = softmax(logits);
    expect(shaped.q[0]).toBeCloseTo(p[0] / (p[0] + p[1]), 12);
    expect(shaped.candidates[2].cutBy).toBe("top-k");
  });

  it("top-p keeps the smallest prefix whose mass reaches p", () => {
    const p = softmax(logits);
    const target = p[0] + p[1] - 1e-6;
    const shaped = shapeDistribution(logits, { ...off, topP: target });
    expect(shaped.keptCount).toBe(2);
    expect(shaped.candidates[2].cutBy).toBe("top-p");
    const wider = shapeDistribution(logits, { ...off, topP: p[0] + p[1] + 1e-6 });
    expect(wider.keptCount).toBe(3);
  });

  it("applies top-p to the distribution left after top-k", () => {
    const shaped = shapeDistribution(logits, { ...off, topK: 3, topP: 0.95 });
    expect(shaped.keptCount).toBeLessThanOrEqual(3);
    expect(shaped.q.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it("always keeps at least the top candidate", () => {
    const shaped = shapeDistribution(logits, { temperature: 1, topK: 66, topP: 0.01 });
    expect(shaped.keptCount).toBe(1);
    expect(shaped.q[0]).toBe(1);
  });
});

describe("min-p", () => {
  // Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs"
  // (ICLR 2025), section 3.1: p_scaled = p_base × p_max; keep every token with P(v) >= p_scaled.
  const minP = (value: number, temperature = 1) => shapeDistribution(logits, { ...off, temperature, minP: value });

  it("keeps exactly the characters whose probability is at least p_min times the top probability", () => {
    const p = softmax(logits);
    for (const threshold of [0.05, 0.2, 0.5, 0.9]) {
      const shaped = minP(threshold);
      expect(shaped.minPThreshold).toBeCloseTo(threshold * p[0], 12);
      const expected = p.filter((value) => value >= threshold * p[0]).length;
      expect(shaped.keptCount).toBe(expected);
      shaped.candidates.forEach((candidate) => expect(candidate.kept).toBe(p[candidate.index] >= threshold * p[0]));
    }
    // The ratios to the top are exp(-1), exp(-1.5), exp(-2), exp(-3), exp(-5) = 0.368, 0.223, 0.135, 0.050, 0.007.
    expect([0.05, 0.2, 0.5, 0.9].map((value) => minP(value).keptCount)).toEqual([4, 3, 1, 1]);
  });

  it("is a test on the logit gap: (z_i - z_max) / T >= ln(p_min)", () => {
    for (const temperature of [0.5, 1, 2]) {
      const shaped = minP(0.2, temperature);
      shaped.candidates.forEach((candidate) =>
        expect(candidate.kept).toBe((candidate.logit - logits[0]) / temperature >= Math.log(0.2) - 1e-12),
      );
    }
  });

  it("renormalises the survivors and always keeps the top character", () => {
    const shaped = minP(0.5);
    expect(shaped.q.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(shaped.candidates[0].kept).toBe(true);
    expect(minP(1).keptCount).toBe(1);
    expect(minP(1).q[0]).toBe(1);
  });

  it("changes nothing when it is 0 or absent, and never reorders", () => {
    const base = shapeDistribution(logits, off);
    const zero = minP(0);
    expect(zero.q).toEqual(base.q);
    expect(zero.minPThreshold).toBe(0);
    expect(zero.keptCount).toBe(logits.length);
    expect(minP(0.3).candidates.map((candidate) => candidate.index)).toEqual(base.candidates.map((candidate) => candidate.index));
  });

  it("adapts to confidence: a peaked distribution keeps fewer characters than a flat one at the same p_min", () => {
    const peaked = shapeDistribution([8, 1, 0.5, 0, -1, -3], { ...off, minP: 0.1 });
    const flat = shapeDistribution([1, 0.9, 0.8, 0.7, 0.6, 0.5], { ...off, minP: 0.1 });
    expect(peaked.keptCount).toBe(1);
    expect(flat.keptCount).toBe(6);
  });

  it("keeps more characters as temperature rises, because the ratios to the top move toward 1", () => {
    const counts = [0.25, 0.5, 1, 2, 4].map((temperature) => minP(0.2, temperature).keptCount);
    expect(counts).toEqual([...counts].sort((a, b) => a - b));
    expect(counts[0]).toBeLessThan(counts[counts.length - 1]);
  });

  it("measures the ratio to the top character, so top-k and top-p renormalising does not move the cut-off", () => {
    const alone = minP(0.2);
    const withTopK = shapeDistribution(logits, { ...off, topK: 4, minP: 0.2 });
    // The cut-off is a tempered probability of the whole vocabulary, not of what is left after top-k.
    expect(withTopK.minPThreshold).toBeCloseTo(alone.minPThreshold, 12);
    const keptAlone = new Set(alone.candidates.filter((candidate) => candidate.kept).map((candidate) => candidate.index));
    const keptBoth = withTopK.candidates.filter((candidate) => candidate.kept).map((candidate) => candidate.index);
    expect(keptBoth.every((index) => keptAlone.has(index))).toBe(true);
    // Both filters together keep the intersection of what each keeps alone.
    const topK = shapeDistribution(logits, { ...off, topK: 4 });
    const intersection = topK.candidates.filter((candidate) => candidate.kept && keptAlone.has(candidate.index)).length;
    expect(withTopK.keptCount).toBe(intersection);
  });

  it("labels what it cut, after top-k and top-p have had their turn", () => {
    const shaped = shapeDistribution(logits, { ...off, topK: 5, topP: 0.99, minP: 0.3 });
    const cuts = new Map(shaped.candidates.map((candidate) => [candidate.rank, candidate.cutBy]));
    expect(cuts.get(0)).toBeNull();
    expect(cuts.get(5)).toBe("top-k");
    expect([...cuts.values()]).toContain("min-p");
  });

  it("treats a non-finite or negative setting as off", () => {
    expect(minP(Number.NaN).keptCount).toBe(logits.length);
    expect(minP(-1).keptCount).toBe(logits.length);
  });
});

describe("seeded sampling", () => {
  it("is deterministic and roughly uniform", () => {
    expect(seededUniform(7, 3)).toBe(seededUniform(7, 3));
    expect(seededUniform(7, 3)).not.toBe(seededUniform(8, 3));
    const draws = Array.from({ length: 20000 }, (_, index) => seededUniform(11, index));
    expect(Math.min(...draws)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...draws)).toBeLessThan(1);
    const bins = new Array(10).fill(0);
    draws.forEach((value) => (bins[Math.floor(value * 10)] += 1));
    bins.forEach((count) => expect(Math.abs(count - 2000)).toBeLessThan(200));
  });

  it("draws by inverse CDF over the kept candidates in rank order", () => {
    const shaped = shapeDistribution(logits, off);
    expect(drawCandidate(shaped, 0).index).toBe(0);
    expect(drawCandidate(shaped, shaped.q[0] - 1e-9).index).toBe(0);
    expect(drawCandidate(shaped, shaped.q[0] + 1e-9).index).toBe(1);
    expect(drawCandidate(shaped, 0.9999999999).kept).toBe(true);
  });

  it("matches the sampling probabilities over many seeded draws", () => {
    const shaped = shapeDistribution(logits, { temperature: 0.8, topK: 4, topP: 1 });
    const counts = new Array(logits.length).fill(0);
    const n = 40000;
    for (let index = 0; index < n; index += 1) counts[drawCandidate(shaped, seededUniform(3, index)).index] += 1;
    shaped.q.forEach((q, index) => expect(counts[index] / n).toBeCloseTo(q, 1));
    expect(counts[4]).toBe(0);
  });
});

describe("vocabulary and surprisal", () => {
  it("encodes the default prompt with no unknown characters", () => {
    const ids = encode(DEFAULT_PROMPT, vocabulary);
    expect(ids).not.toContain(vocabulary.unk);
    expect(ids).toHaveLength(38);
    expect(ids.map((id) => tokenText(id, vocabulary)).join("")).toBe(DEFAULT_PROMPT);
  });

  it("maps characters outside the 65-symbol alphabet to <unk>", () => {
    expect(encode("7", vocabulary)).toEqual([vocabulary.unk]);
    expect(displayToken(tokenText(vocabulary.unk, vocabulary))).toBe("<unk>");
  });

  it("measures surprisal in bits", () => {
    expect(surprisalBits(0.5)).toBeCloseTo(1, 12);
    expect(surprisalBits(1 / 66)).toBeCloseTo(Math.log2(66), 12);
  });

  it("round-trips stored draws and rejects malformed ones", () => {
    const stored = encodeDraw({ m: "rnn", u: 0.123456789, p: 0.2, q: 0.30000001 });
    expect(stored).toBe("rnn|0.123457|0.2|0.3");
    expect(decodeDraw(stored, "a")).toEqual({ c: "a", m: "rnn", u: 0.123457, p: 0.2, q: 0.3 });
    expect(decodeDraw("gpt|0.1|0.2|0.3", "a")).toBeUndefined();
    expect(decodeDraw("rnn|1.5|0.2|0.3", "a")).toBeUndefined();
    expect(decodeDraw("rnn|0.1|0.2", "a")).toBeUndefined();
    expect(decodeDraw(42, "a")).toBeUndefined();
  });
});
