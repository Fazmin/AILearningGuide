export const ROWS = [
  { id: 0, name: "Alder 12", rooms: 2, color: "red", park: "yes", price: 1 },
  { id: 1, name: "Birch 4", rooms: 1, color: "blue", park: "no", price: 0 },
  { id: 2, name: "Cedar 9", rooms: 3, color: "green", park: "yes", price: 1 },
  { id: 3, name: "Dock 2", rooms: 4, color: "red", park: "no", price: 1 },
  { id: 4, name: "Elm 7", rooms: 2, color: "blue", park: "yes", price: 0 },
  { id: 5, name: "Fir 1", rooms: 1, color: "green", park: "no", price: 0 },
] as const;

export type Apartment = (typeof ROWS)[number];
export type ColorName = Apartment["color"];
export type Encoding = "integer" | "onehot" | "normalized";
export type FeatureAxis = "rooms" | "color" | "park";
export type ColorCodes = Record<ColorName, number>;

export const ENCODINGS = ["integer", "onehot", "normalized"] as const;
export const AXES = ["rooms", "color", "park"] as const;
export const MAX_ROOMS = 4;
export const MAX_COLOR_CODE = 5;

/** Authored 2-D coordinates that put similar park-and-price rows nearby. Not a trained embedding. */
export const EMBEDDED: Record<number, [number, number]> = {
  0: [0.16, 0.88],
  1: [0.88, 0.12],
  2: [0.4, 0.62],
  3: [0.72, 0.86],
  4: [0.2, 0.2],
  5: [0.78, 0.36],
};

export const DEFAULT_CODES: ColorCodes = { red: 1, blue: 2, green: 3 };

export const euclid = (left: readonly number[], right: readonly number[]) => {
  const n = Math.max(left.length, right.length);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const d = (left[i] ?? 0) - (right[i] ?? 0);
    sum += d * d;
  }
  return Math.sqrt(sum);
};

export const colorOneHot = (color: ColorName) => [
  color === "red" ? 1 : 0,
  color === "blue" ? 1 : 0,
  color === "green" ? 1 : 0,
];

export const featureNames = (encoding: Encoding) =>
  encoding === "onehot" ? ["rooms", "red", "blue", "green", "park"] : ["rooms", "color", "park"];

export const encodeRow = (row: Apartment, encoding: Encoding, codes: ColorCodes) => {
  const park = row.park === "yes" ? 1 : 0;
  const rooms = encoding === "normalized" ? row.rooms / MAX_ROOMS : row.rooms;
  if (encoding === "onehot") {
    return [rooms, ...colorOneHot(row.color), park];
  }
  const color = encoding === "normalized" ? codes[row.color] / MAX_COLOR_CODE : codes[row.color];
  return [rooms, color, park];
};

export const axisValue = (
  row: Apartment,
  axis: FeatureAxis,
  encoding: Encoding,
  codes: ColorCodes,
) => {
  if (axis === "rooms") return encoding === "normalized" ? row.rooms / MAX_ROOMS : row.rooms;
  if (axis === "park") return row.park === "yes" ? 1 : 0;
  if (encoding === "onehot") return row.color === "red" ? 1 : 0;
  return encoding === "normalized" ? codes[row.color] / MAX_COLOR_CODE : codes[row.color];
};

export const neighborOrder = (features: readonly number[][], from: number) => {
  const here = features[from];
  return features
    .map((vector, id) => ({ id, dist: id === from ? 0 : euclid(vector, here) }))
    .filter((row) => row.id !== from)
    .sort((left, right) => left.dist - right.dist || left.id - right.id);
};

export const planeNeighbor = (features: readonly number[][], from: number) => {
  const here = features[from];
  return features
    .map((vector, id) => ({
      id,
      dist: id === from ? 0 : Math.hypot((vector[0] ?? 0) - (here[0] ?? 0), (vector[1] ?? 0) - (here[1] ?? 0)),
    }))
    .filter((row) => row.id !== from)
    .sort((left, right) => left.dist - right.dist || left.id - right.id);
};

/** Every row whose distance equals the nearest one, so a tie is reported instead of hidden by a sort. */
export const nearestTies = (ranks: ReadonlyArray<{ id: number; dist: number }>) => {
  const first = ranks[0];
  if (!first) return [];
  return ranks.filter((rank) => Math.abs(rank.dist - first.dist) < 1e-9);
};
