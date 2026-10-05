/**
 * Cosine geometry over the shipped GloVe subset: 10,000 lowercase ASCII words,
 * 50 dimensions each, stored row-major as little-endian float32.
 */

export type ProjectionKind = "umap" | "pca";

export interface EmbeddingSpace {
  words: string[];
  index: Map<string, number>;
  dims: number;
  /** Raw vectors, row-major. */
  vectors: Float32Array;
  /** Each row divided by its L2 norm, so a dot product is a cosine. */
  unit: Float32Array;
  norms: Float32Array;
  projections: Record<ProjectionKind, Float32Array>;
}

export interface Neighbour {
  index: number;
  word: string;
  cosine: number;
}

export interface AnalogyResult {
  /** unit(a) − unit(b) + unit(c), before renormalizing. */
  target: Float32Array;
  results: Neighbour[];
  /** The best match when the three query words are allowed to win. */
  unfiltered: Neighbour[];
}

export function readFloat32LE(buffer: ArrayBuffer, count: number) {
  if (buffer.byteLength < count * 4) {
    throw new Error(`Vector file holds ${buffer.byteLength} bytes; expected ${count * 4}.`);
  }
  const view = new DataView(buffer);
  const values = new Float32Array(count);
  for (let index = 0; index < count; index += 1) values[index] = view.getFloat32(index * 4, true);
  return values;
}

export function buildSpace(
  words: string[],
  dims: number,
  vectors: Float32Array,
  projections: Record<ProjectionKind, ReadonlyArray<readonly [number, number]>>,
): EmbeddingSpace {
  const count = words.length;
  if (vectors.length !== count * dims) {
    throw new Error(`Expected ${count} × ${dims} values, found ${vectors.length}.`);
  }
  const unit = new Float32Array(vectors.length);
  const norms = new Float32Array(count);
  for (let row = 0; row < count; row += 1) {
    let sum = 0;
    for (let column = 0; column < dims; column += 1) sum += vectors[row * dims + column] ** 2;
    const norm = Math.sqrt(sum) || 1;
    norms[row] = norm;
    for (let column = 0; column < dims; column += 1) {
      unit[row * dims + column] = vectors[row * dims + column] / norm;
    }
  }
  const flatten = (points: ReadonlyArray<readonly [number, number]>) => {
    if (points.length !== count) throw new Error("Projection and vocabulary lengths differ.");
    const flat = new Float32Array(count * 2);
    points.forEach(([x, y], index) => {
      flat[index * 2] = x;
      flat[index * 2 + 1] = y;
    });
    return flat;
  };
  return {
    words,
    index: new Map(words.map((word, position) => [word, position])),
    dims,
    vectors,
    unit,
    norms,
    projections: { umap: flatten(projections.umap), pca: flatten(projections.pca) },
  };
}

export function unitRow(space: EmbeddingSpace, row: number) {
  return space.unit.subarray(row * space.dims, (row + 1) * space.dims);
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>) {
  let dotValue = 0;
  let aa = 0;
  let bb = 0;
  for (let index = 0; index < a.length; index += 1) {
    dotValue += a[index] * b[index];
    aa += a[index] * a[index];
    bb += b[index] * b[index];
  }
  return aa && bb ? dotValue / Math.sqrt(aa * bb) : 0;
}

/** The k rows with the highest cosine to `query`, skipping any row in `exclude`. */
export function nearest(
  space: EmbeddingSpace,
  query: ArrayLike<number>,
  k: number,
  exclude: ReadonlySet<number> = new Set(),
): Neighbour[] {
  let queryNorm = 0;
  for (let index = 0; index < space.dims; index += 1) queryNorm += query[index] * query[index];
  queryNorm = Math.sqrt(queryNorm) || 1;
  const best: Neighbour[] = [];
  for (let row = 0; row < space.words.length; row += 1) {
    if (exclude.has(row)) continue;
    let score = 0;
    const offset = row * space.dims;
    for (let column = 0; column < space.dims; column += 1) score += space.unit[offset + column] * query[column];
    score /= queryNorm;
    if (best.length < k || score > best[best.length - 1].cosine) {
      const entry = { index: row, word: space.words[row], cosine: score };
      let position = best.length;
      while (position > 0 && best[position - 1].cosine < score) position -= 1;
      best.splice(position, 0, entry);
      if (best.length > k) best.pop();
    }
  }
  return best;
}

export function neighboursOf(space: EmbeddingSpace, word: string, k: number) {
  const row = space.index.get(word);
  if (row === undefined) return [];
  return nearest(space, unitRow(space, row), k, new Set([row]));
}

/**
 * 3CosAdd analogy: rank words by cosine to unit(a) − unit(b) + unit(c).
 * "a is to b as ? is to c" read as king − man + woman.
 */
export function analogy(space: EmbeddingSpace, a: string, b: string, c: string, k: number): AnalogyResult | null {
  const rows = [a, b, c].map((word) => space.index.get(word));
  if (rows.some((row) => row === undefined)) return null;
  const [ra, rb, rc] = rows as number[];
  const target = new Float32Array(space.dims);
  const ua = unitRow(space, ra);
  const ub = unitRow(space, rb);
  const uc = unitRow(space, rc);
  for (let column = 0; column < space.dims; column += 1) target[column] = ua[column] - ub[column] + uc[column];
  return {
    target,
    results: nearest(space, target, k, new Set([ra, rb, rc])),
    unfiltered: nearest(space, target, k),
  };
}

/** Indices of the k points nearest to `row` by Euclidean distance in a 2-D projection. */
export function projectedNeighbours(points: Float32Array, row: number, k: number) {
  const x = points[row * 2];
  const y = points[row * 2 + 1];
  const best: Array<{ index: number; distance: number }> = [];
  const count = points.length / 2;
  for (let index = 0; index < count; index += 1) {
    if (index === row) continue;
    const distance = (points[index * 2] - x) ** 2 + (points[index * 2 + 1] - y) ** 2;
    if (best.length < k || distance < best[best.length - 1].distance) {
      let position = best.length;
      while (position > 0 && best[position - 1].distance > distance) position -= 1;
      best.splice(position, 0, { index, distance });
      if (best.length > k) best.pop();
    }
  }
  return best.map((entry) => entry.index);
}

/** How many of a word's k nearest 50-D neighbours are also among its k nearest in 2-D. */
export function neighbourOverlap(space: EmbeddingSpace, kind: ProjectionKind, row: number, k: number) {
  const high = new Set(nearest(space, unitRow(space, row), k, new Set([row])).map((entry) => entry.index));
  const low = projectedNeighbours(space.projections[kind], row, k);
  return { kept: low.filter((index) => high.has(index)).length, of: k };
}
