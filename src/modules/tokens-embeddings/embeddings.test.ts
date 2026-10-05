import { describe, expect, it } from "vitest";
import index from "./assets/embedding-10k.index.json";
import projections from "./assets/embedding-10k.projections.json";
import {
  analogy,
  buildSpace,
  cosine,
  nearest,
  neighbourOverlap,
  neighboursOf,
  projectedNeighbours,
  readFloat32LE,
  unitRow,
} from "./embeddings";

const tinyProjection = (count: number) =>
  Array.from({ length: count }, (_, index) => [index, 0] as const);

function tinySpace() {
  // Four 2-D vectors: a man→woman offset that repeats from king to queen.
  const words = ["king", "queen", "man", "woman", "apple"];
  const vectors = new Float32Array([3, 1, 3, 2, 1, 1, 1, 2, -1, -3]);
  return buildSpace(words, 2, vectors, { umap: tinyProjection(5), pca: tinyProjection(5) });
}

describe("embedding geometry", () => {
  it("reads little-endian float32 regardless of platform order", () => {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setFloat32(0, 1.5, true);
    view.setFloat32(4, -2.25, true);
    expect(Array.from(readFloat32LE(buffer, 2))).toEqual([1.5, -2.25]);
    expect(() => readFloat32LE(buffer, 3)).toThrow();
  });

  it("normalizes rows so a dot product is a cosine", () => {
    const space = tinySpace();
    const row = unitRow(space, 0);
    expect(Math.hypot(row[0], row[1])).toBeCloseTo(1, 6);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([2, 0], [5, 0])).toBe(1);
  });

  it("ranks neighbours by cosine and skips excluded rows", () => {
    const space = tinySpace();
    const result = nearest(space, [1, 0], 2, new Set([0]));
    expect(result.map((entry) => entry.word)).toEqual(["queen", "man"]);
    expect(result[0].cosine).toBeGreaterThan(result[1].cosine);
    expect(neighboursOf(space, "missing", 3)).toEqual([]);
  });

  it("computes 3CosAdd and reports the unfiltered winner separately", () => {
    const space = tinySpace();
    const result = analogy(space, "king", "man", "woman", 2);
    expect(result).not.toBeNull();
    expect(result?.results.map((entry) => entry.word)).not.toContain("king");
    expect(analogy(space, "king", "man", "nobody", 2)).toBeNull();
  });

  it("finds 2-D neighbours by Euclidean distance", () => {
    const points = new Float32Array([0, 0, 1, 0, 5, 5, 0.5, 0]);
    expect(projectedNeighbours(points, 0, 2)).toEqual([3, 1]);
  });
});

// The binary vector file is read from disk the way the app fetches it: as raw bytes.
// Node's fs is imported by name so the app's browser-only type config stays untouched.
const fsModule = "node:fs";
const { readFileSync } = (await import(/* @vite-ignore */ fsModule)) as {
  readFileSync: (path: string) => Uint8Array;
};
const vectorBytes = readFileSync("src/modules/tokens-embeddings/assets/embedding-10k.f32");
const vectorBuffer = vectorBytes.buffer.slice(
  vectorBytes.byteOffset,
  vectorBytes.byteOffset + vectorBytes.byteLength,
) as ArrayBuffer;

describe("the shipped GloVe subset", () => {
  const space = buildSpace(
    index.words,
    index.dimensions,
    readFloat32LE(vectorBuffer, index.count * index.dimensions),
    {
      umap: projections.umap as Array<[number, number]>,
      pca: projections.pca as Array<[number, number]>,
    },
  );

  it("holds 10,000 words of 50 dimensions", () => {
    expect(space.words).toHaveLength(10_000);
    expect(space.dims).toBe(50);
  });

  // These measured values are quoted in the lesson and the card explanations.
  it("king − man + woman → queen at cosine 0.852 once the inputs are excluded", () => {
    const result = analogy(space, "king", "man", "woman", 5);
    expect(result?.results[0].word).toBe("queen");
    expect(result?.results[0].cosine).toBeCloseTo(0.852, 3);
    expect(result?.unfiltered[0].word).toBe("king");
    expect(result?.unfiltered[0].cosine).toBeCloseTo(0.885, 3);
  });

  it("bigger − big + small → larger, not smaller", () => {
    const result = analogy(space, "bigger", "big", "small", 3);
    expect(result?.results.map((entry) => entry.word)).toEqual(["larger", "smaller", "large"]);
  });

  it("queen's nearest neighbour is princess at 0.852; hot sits next to cold", () => {
    const queen = neighboursOf(space, "queen", 3);
    expect(queen[0].word).toBe("princess");
    expect(queen[0].cosine).toBeCloseTo(0.852, 3);
    expect(neighboursOf(space, "hot", 2).map((entry) => entry.word)).toEqual(["cool", "cold"]);
  });

  it("UMAP keeps 5 of queen's 10 nearest neighbours; PCA keeps none", () => {
    const row = space.index.get("queen") ?? -1;
    expect(neighbourOverlap(space, "umap", row, 10).kept).toBe(5);
    expect(neighbourOverlap(space, "pca", row, 10).kept).toBe(0);
  });
});
