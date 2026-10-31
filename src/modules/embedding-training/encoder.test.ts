import { describe, expect, it } from "vitest";
import { BUCKETS, bucketsOf, dot, encode, fnv1a, initialWeights, offsetDirection, textTrigrams, trigramsOf, words } from "./encoder";
import { PASSAGES } from "./data";
import { meanPairCosine } from "./metrics";
import { PASSAGE_BUCKETS } from "./train";

describe("the trigram encoder", () => {
  it("pads words with # and cuts them into three-letter pieces", () => {
    expect(trigramsOf("dock")).toEqual(["#do", "doc", "ock", "ck#"]);
    expect(trigramsOf("a")).toEqual(["#a#"]);
    expect(words("When does the North dock close at 18:00?")).toEqual(["when", "north", "dock", "close", "18:00"]);
    expect(textTrigrams("the dock")).toEqual(["#do", "doc", "ock", "ck#"]);
  });

  it("shares most pieces between a word and its typo or plural, as the lesson says", () => {
    const north = new Set(trigramsOf("north"));
    expect(trigramsOf("norht").filter((piece) => north.has(piece))).toEqual(["#no", "nor"]);
    expect(north.size).toBe(5);
    const dock = new Set(trigramsOf("dock"));
    expect(trigramsOf("docks").filter((piece) => dock.has(piece))).toHaveLength(3);
    expect(dock.size).toBe(4);
  });

  it("hashes with 32-bit FNV-1a into 512 buckets", () => {
    // Published FNV-1a test vectors.
    expect(fnv1a("")).toBe(0x811c9dc5);
    expect(fnv1a("a")).toBe(0xe40c292c);
    expect(fnv1a("foobar")).toBe(0xbf9cf968);
    for (const passage of PASSAGES) {
      for (const bucket of bucketsOf(passage.text)) {
        expect(bucket).toBeGreaterThanOrEqual(0);
        expect(bucket).toBeLessThan(BUCKETS);
      }
    }
  });

  it("mean-pools the rows and returns a unit vector", () => {
    const weights = initialWeights(3, 16);
    const buckets = bucketsOf("north dock");
    const encoded = encode(weights, 16, buckets);
    const manual = new Float64Array(16);
    for (const bucket of buckets) for (let k = 0; k < 16; k += 1) manual[k] += weights[bucket * 16 + k] / buckets.length;
    for (let k = 0; k < 16; k += 1) expect(encoded.pooled[k]).toBeCloseTo(manual[k], 12);
    expect(Math.hypot(...encoded.unit)).toBeCloseTo(1, 12);
    expect(encode(weights, 16, new Int32Array(0)).norm).toBe(0);
  });

  it("starts every vector leaning the same way because every row shares one offset", () => {
    const lean = offsetDirection(1, 16);
    expect(Math.hypot(...lean)).toBeCloseTo(1, 12);
    const weights = initialWeights(1, 16);
    const passages = PASSAGE_BUCKETS.map((buckets) => encode(weights, 16, buckets).unit);
    expect(meanPairCosine(passages)).toBeGreaterThan(0.9);
    // Without the offset the same noise leaves the passages far less alike.
    const noOffset = initialWeights(1, 16, undefined, 0);
    const spread = PASSAGE_BUCKETS.map((buckets) => encode(noOffset, 16, buckets).unit);
    expect(meanPairCosine(spread)).toBeLessThan(0.3);
    expect(dot(passages[0], lean)).toBeGreaterThan(0.9);
  });
});
