/** Classical supervised fits for the 16-apartment loop. Every number is computed. */

export const ENCODINGS = ["integer", "onehot", "numeric"] as const;
export const ALGORITHMS = ["linear", "logistic", "tree"] as const;
export const LOSSES = ["mse", "logloss"] as const;

export type Encoding = (typeof ENCODINGS)[number];
export type Algorithm = (typeof ALGORITHMS)[number];
export type LossKind = (typeof LOSSES)[number];
export type ColorName = "red" | "blue" | "green";
export type SplitName = "train" | "holdout";

export type Apartment = {
  id: number;
  name: string;
  rooms: number;
  color: ColorName;
  park: 0 | 1;
  price: 0 | 1;
  split: SplitName;
};

/**
 * Authored rows. The usual label is high price when rooms ≥ 3 or (rooms = 2 and
 * park). Rowan 9 is a written exception so a deep tree has something to isolate.
 * Color is a spurious correlate on the train slice only.
 */
export const APARTMENTS: Apartment[] = [
  { id: 0, name: "Alder 12", rooms: 2, color: "red", park: 1, price: 1, split: "train" },
  { id: 1, name: "Birch 4", rooms: 1, color: "blue", park: 0, price: 0, split: "train" },
  { id: 2, name: "Cedar 9", rooms: 3, color: "green", park: 1, price: 1, split: "train" },
  { id: 3, name: "Dock 2", rooms: 4, color: "red", park: 0, price: 1, split: "train" },
  { id: 4, name: "Elm 7", rooms: 2, color: "blue", park: 0, price: 0, split: "train" },
  { id: 5, name: "Fir 1", rooms: 1, color: "green", park: 0, price: 0, split: "train" },
  { id: 6, name: "Grove 5", rooms: 3, color: "red", park: 0, price: 1, split: "train" },
  { id: 7, name: "Holly 3", rooms: 2, color: "red", park: 1, price: 1, split: "train" },
  { id: 8, name: "Ivy 8", rooms: 1, color: "red", park: 1, price: 0, split: "train" },
  { id: 9, name: "Juniper 6", rooms: 4, color: "blue", park: 1, price: 1, split: "train" },
  { id: 10, name: "Knot 11", rooms: 2, color: "green", park: 0, price: 0, split: "train" },
  { id: 11, name: "Rowan 9", rooms: 3, color: "blue", park: 1, price: 0, split: "train" },
  { id: 12, name: "Laurel 2", rooms: 2, color: "red", park: 0, price: 0, split: "holdout" },
  { id: 13, name: "Maple 10", rooms: 3, color: "blue", park: 0, price: 1, split: "holdout" },
  { id: 14, name: "Oak 14", rooms: 1, color: "red", park: 0, price: 0, split: "holdout" },
  { id: 15, name: "Pine 6", rooms: 4, color: "green", park: 1, price: 1, split: "holdout" },
];

/**
 * Which four rows are the holdout: the original four, drawn from the same process as the train rows, or four
 * rows from a different process.
 */
export const HOLDOUT_SOURCES = ["same", "shifted"] as const;
export type HoldoutSource = (typeof HOLDOUT_SOURCES)[number];

/**
 * Four rows from a market where the rule that links the features to price is different: small flats are the
 * expensive ones (high price when rooms ≤ 2). The inputs look like the train rows, but the rule that
 * produced the labels changed, so a rule fitted on the train slice is wrong on three of the four. Ids 12 to 15
 * replace the same-process holdout rows one for one.
 */
export const SHIFTED_HOLDOUT: Apartment[] = [
  { id: 12, name: "Quay 3", rooms: 1, color: "blue", park: 0, price: 1, split: "holdout" },
  { id: 13, name: "Reed 5", rooms: 2, color: "green", park: 1, price: 1, split: "holdout" },
  { id: 14, name: "Slate 7", rooms: 3, color: "red", park: 1, price: 0, split: "holdout" },
  { id: 15, name: "Tarn 11", rooms: 4, color: "blue", park: 0, price: 0, split: "holdout" },
];

const SHIFTED_ROWS: Apartment[] = [...APARTMENTS.filter((row) => row.split === "train"), ...SHIFTED_HOLDOUT];

/** The sixteen rows with the chosen holdout. The twelve train rows never change. */
export const apartmentsFor = (source: HoldoutSource): Apartment[] => (source === "shifted" ? SHIFTED_ROWS : APARTMENTS);

export const COLOR_CODE: Record<ColorName, number> = { red: 1, blue: 2, green: 3 };
export const GD_STEPS = 48;
export const TREE_MAX_DEPTH = 5;
export const RIDGE = 1e-4;

export type TreeNode =
  | { kind: "leaf"; probability: number }
  | { kind: "split"; axis: number; threshold: number; left: TreeNode; right: TreeNode };

export type LinearState = {
  weights: number[];
  bias: number;
};

export type FitSnapshot = {
  step: number;
  weights: number[];
  bias: number;
  tree: TreeNode | null;
  trainLoss: number;
  holdoutLoss: number;
  trainAccuracy: number;
  holdoutAccuracy: number;
  gradNorm: number;
};

const EPS = 1e-9;

export const sigmoid = (z: number) => {
  const clipped = Math.max(-20, Math.min(20, z));
  return 1 / (1 + Math.exp(-clipped));
};

export const dot = (left: readonly number[], right: readonly number[]) => {
  let sum = 0;
  const n = Math.max(left.length, right.length);
  for (let i = 0; i < n; i += 1) sum += (left[i] ?? 0) * (right[i] ?? 0);
  return sum;
};

export const featureNames = (encoding: Encoding): string[] => {
  if (encoding === "onehot") return ["rooms", "red", "blue", "green", "park"];
  if (encoding === "numeric") return ["rooms", "park"];
  return ["rooms", "color", "park"];
};

export const encodeApartment = (row: Apartment, encoding: Encoding): number[] => {
  if (encoding === "onehot") {
    return [
      row.rooms,
      row.color === "red" ? 1 : 0,
      row.color === "blue" ? 1 : 0,
      row.color === "green" ? 1 : 0,
      row.park,
    ];
  }
  if (encoding === "numeric") return [row.rooms, row.park];
  return [row.rooms, COLOR_CODE[row.color], row.park];
};

/** Features at a rooms×park query. Color bits come from the slice row; park stays continuous. */
export const featuresAt = (
  rooms: number,
  park: number,
  encoding: Encoding,
  slice: Apartment,
): number[] => {
  if (encoding === "onehot") {
    return [
      rooms,
      slice.color === "red" ? 1 : 0,
      slice.color === "blue" ? 1 : 0,
      slice.color === "green" ? 1 : 0,
      park,
    ];
  }
  if (encoding === "numeric") return [rooms, park];
  return [rooms, COLOR_CODE[slice.color], park];
};

export const encodeAll = (encoding: Encoding, source: HoldoutSource = "same") =>
  apartmentsFor(source).map((row) => ({
    row,
    x: encodeApartment(row, encoding),
    y: row.price,
  }));

const logProb = (p: number) => Math.log(Math.min(1 - EPS, Math.max(EPS, p)));

export const pointLoss = (prediction: number, y: number, loss: LossKind) => {
  if (loss === "mse") return (prediction - y) ** 2;
  return -(y * logProb(prediction) + (1 - y) * logProb(1 - prediction));
};

export const meanLoss = (predictions: readonly number[], labels: readonly number[], loss: LossKind) => {
  if (predictions.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < predictions.length; i += 1) sum += pointLoss(predictions[i], labels[i], loss);
  return sum / predictions.length;
};

export const accuracyOf = (predictions: readonly number[], labels: readonly number[]) => {
  if (predictions.length === 0) return 0;
  let hits = 0;
  for (let i = 0; i < predictions.length; i += 1) {
    if ((predictions[i] >= 0.5 ? 1 : 0) === labels[i]) hits += 1;
  }
  return hits / predictions.length;
};

const scoreLinear = (x: readonly number[], model: LinearState) => model.bias + dot(model.weights, x);

export const predictProbability = (
  x: readonly number[],
  algorithm: Algorithm,
  loss: LossKind,
  linear: LinearState,
  tree: TreeNode | null,
) => {
  if (algorithm === "tree") return tree ? treeProbability(tree, x) : 0.5;
  const raw = scoreLinear(x, linear);
  if (algorithm === "logistic" || loss === "logloss") return sigmoid(raw);
  return raw;
};

const predictMany = (
  rows: ReadonlyArray<{ x: number[] }>,
  algorithm: Algorithm,
  loss: LossKind,
  linear: LinearState,
  tree: TreeNode | null,
) => rows.map((row) => predictProbability(row.x, algorithm, loss, linear, tree));

const zeros = (dimension: number) => Array.from({ length: dimension }, () => 0);

const cloneWeights = (weights: readonly number[]) => weights.slice();

export const solveLeastSquares = (X: readonly number[][], y: readonly number[]): LinearState => {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const dim = d + 1;
  const A = Array.from({ length: dim }, () => zeros(dim));
  const b = zeros(dim);
  for (let i = 0; i < n; i += 1) {
    const row = [1, ...X[i]];
    for (let r = 0; r < dim; r += 1) {
      b[r] += row[r] * y[i];
      for (let c = 0; c < dim; c += 1) A[r][c] += row[r] * row[c];
    }
  }
  for (let i = 0; i < dim; i += 1) A[i][i] += RIDGE;
  const solved = solveSystem(A, b);
  return { bias: solved[0] ?? 0, weights: solved.slice(1) };
};

const solveSystem = (rawA: number[][], rawB: number[]) => {
  const n = rawB.length;
  const A = rawA.map((row, index) => [...row, rawB[index]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) pivot = row;
    }
    [A[col], A[pivot]] = [A[pivot], A[col]];
    const denom = A[col][col];
    const scale = Math.abs(denom) < 1e-12 ? 1 : denom;
    for (let j = col; j <= n; j += 1) A[col][j] /= scale;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = A[row][col];
      for (let j = col; j <= n; j += 1) A[row][j] -= factor * A[col][j];
    }
  }
  return A.map((row) => row[n]);
};

/**
 * Exact gradient of the mean training loss with respect to (w, b).
 * MSE: ∂/∂z (p − y)² = 2(p − y), times p(1 − p) when p = σ(z). Log-loss with p = σ(z): ∂/∂z = p − y.
 */
export const linearGradient = (
  rows: ReadonlyArray<{ x: number[]; y: number }>,
  model: LinearState,
  algorithm: Algorithm,
  loss: LossKind,
) => {
  const gW = zeros(model.weights.length);
  let gB = 0;
  const n = Math.max(1, rows.length);
  for (const row of rows) {
    const z = scoreLinear(row.x, model);
    const prediction = algorithm === "logistic" || loss === "logloss" ? sigmoid(z) : z;
    let residual: number;
    if (loss === "logloss") {
      residual = prediction - row.y;
    } else if (algorithm === "logistic") {
      residual = 2 * (prediction - row.y) * prediction * (1 - prediction);
    } else {
      residual = 2 * (prediction - row.y);
    }
    gB += residual;
    for (let i = 0; i < gW.length; i += 1) gW[i] += residual * (row.x[i] ?? 0);
  }
  return {
    gW: gW.map((value) => value / n),
    gB: gB / n,
  };
};

/** Step sizes η for full-batch gradient descent on the mean loss. */
export const stepSize = (algorithm: Algorithm, loss: LossKind) => {
  if (algorithm === "logistic" && loss === "mse") return 0.7;
  if (loss === "logloss") return 0.85;
  return 0.06;
};

const impurity = (labels: readonly number[], loss: LossKind) => {
  if (labels.length === 0) return 0;
  const mean = labels.reduce((sum, value) => sum + value, 0) / labels.length;
  if (loss === "logloss") {
    const p = Math.min(1 - EPS, Math.max(EPS, mean));
    return -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
  }
  return 2 * mean * (1 - mean);
};

const bestSplit = (X: readonly number[][], y: readonly number[], loss: LossKind) => {
  let winner = { axis: 0, threshold: 0, gain: -1 };
  const parent = impurity(y, loss);
  const n = y.length;
  const dim = X[0]?.length ?? 0;
  for (let axis = 0; axis < dim; axis += 1) {
    const values = [...new Set(X.map((row) => row[axis]))].sort((a, b) => a - b);
    for (let index = 0; index < values.length - 1; index += 1) {
      const threshold = (values[index] + values[index + 1]) / 2;
      const leftY: number[] = [];
      const rightY: number[] = [];
      for (let row = 0; row < n; row += 1) {
        if (X[row][axis] < threshold) leftY.push(y[row]);
        else rightY.push(y[row]);
      }
      if (leftY.length === 0 || rightY.length === 0) continue;
      const gain = parent - (leftY.length / n) * impurity(leftY, loss) - (rightY.length / n) * impurity(rightY, loss);
      if (gain > winner.gain) winner = { axis, threshold, gain };
    }
  }
  return winner;
};

export const buildTree = (
  X: readonly number[][],
  y: readonly number[],
  depth: number,
  loss: LossKind,
): TreeNode => {
  const probability = y.length === 0 ? 0.5 : y.reduce((sum, value) => sum + value, 0) / y.length;
  if (depth <= 0 || y.length === 0 || y.every((value) => value === y[0])) {
    return { kind: "leaf", probability };
  }
  const split = bestSplit(X, y, loss);
  if (split.gain <= 1e-9) return { kind: "leaf", probability };
  const leftX: number[][] = [];
  const leftY: number[] = [];
  const rightX: number[][] = [];
  const rightY: number[] = [];
  for (let i = 0; i < X.length; i += 1) {
    if (X[i][split.axis] < split.threshold) {
      leftX.push(X[i]);
      leftY.push(y[i]);
    } else {
      rightX.push(X[i]);
      rightY.push(y[i]);
    }
  }
  if (leftY.length === 0 || rightY.length === 0) return { kind: "leaf", probability };
  return {
    kind: "split",
    axis: split.axis,
    threshold: split.threshold,
    left: buildTree(leftX, leftY, depth - 1, loss),
    right: buildTree(rightX, rightY, depth - 1, loss),
  };
};

export const treeProbability = (node: TreeNode, x: readonly number[]): number => {
  if (node.kind === "leaf") return node.probability;
  return treeProbability(x[node.axis] < node.threshold ? node.left : node.right, x);
};

export const collectSplits = (
  node: TreeNode,
  acc: { axis: number; threshold: number }[] = [],
): { axis: number; threshold: number }[] => {
  if (node.kind === "split") {
    acc.push({ axis: node.axis, threshold: node.threshold });
    collectSplits(node.left, acc);
    collectSplits(node.right, acc);
  }
  return acc;
};

const snapshotOf = (
  step: number,
  rows: ReadonlyArray<{ x: number[]; y: number }>,
  holdout: ReadonlyArray<{ x: number[]; y: number }>,
  algorithm: Algorithm,
  loss: LossKind,
  linear: LinearState,
  tree: TreeNode | null,
  gradNorm: number,
): FitSnapshot => {
  const trainP = predictMany(rows, algorithm, loss, linear, tree);
  const holdP = predictMany(holdout, algorithm, loss, linear, tree);
  return {
    step,
    weights: cloneWeights(linear.weights),
    bias: linear.bias,
    tree,
    trainLoss: meanLoss(trainP, rows.map((row) => row.y), loss),
    holdoutLoss: meanLoss(holdP, holdout.map((row) => row.y), loss),
    trainAccuracy: accuracyOf(trainP, rows.map((row) => row.y)),
    holdoutAccuracy: accuracyOf(holdP, holdout.map((row) => row.y)),
    gradNorm,
  };
};

export const runFit = (
  encoding: Encoding,
  algorithm: Algorithm,
  loss: LossKind,
  depth: number,
  source: HoldoutSource = "same",
): FitSnapshot[] => {
  const encoded = encodeAll(encoding, source);
  const train = encoded.filter((item) => item.row.split === "train").map((item) => ({ x: item.x, y: item.y }));
  const holdout = encoded.filter((item) => item.row.split === "holdout").map((item) => ({ x: item.x, y: item.y }));
  const dimension = train[0]?.x.length ?? 0;

  if (algorithm === "tree") {
    const cap = Math.max(0, Math.min(TREE_MAX_DEPTH, Math.round(depth)));
    return Array.from({ length: cap + 1 }, (_, step) => {
      const tree = buildTree(
        train.map((row) => row.x),
        train.map((row) => row.y),
        step,
        loss,
      );
      return snapshotOf(step, train, holdout, algorithm, loss, { weights: zeros(dimension), bias: 0 }, tree, 0);
    });
  }

  let model: LinearState = { weights: zeros(dimension), bias: 0 };
  const rate = stepSize(algorithm, loss);
  const history: FitSnapshot[] = [];
  for (let step = 0; step <= GD_STEPS; step += 1) {
    const { gW, gB } = linearGradient(train, model, algorithm, loss);
    const gradNorm = Math.sqrt(gB * gB + gW.reduce((sum, value) => sum + value * value, 0));
    history.push(snapshotOf(step, train, holdout, algorithm, loss, model, null, gradNorm));
    if (step === GD_STEPS) break;
    model = {
      bias: model.bias - rate * gB,
      weights: model.weights.map((value, index) => value - rate * (gW[index] ?? 0)),
    };
  }
  return history;
};

export const closedFormLinear = (encoding: Encoding) => {
  const train = encodeAll(encoding)
    .filter((item) => item.row.split === "train")
    .map((item) => ({ x: item.x, y: item.y }));
  return solveLeastSquares(
    train.map((row) => row.x),
    train.map((row) => row.y),
  );
};

export type SliceSegment = { axis: "rooms" | "park"; threshold: number; from: number; to: number };

/**
 * The tree's cuts as they appear in the rooms × park slice at the selected row's color. A color split is
 * not a line in this plane: the slice takes the branch its color falls into, and cuts in the other
 * branch are not drawn. A rooms or park cut spans only the rectangle its parent owns.
 */
export const sliceSegments = (
  node: TreeNode,
  encoding: Encoding,
  slice: Apartment,
  bounds = { rooms0: 0.6, rooms1: 4.4, park0: -0.2, park1: 1.2 },
  acc: SliceSegment[] = [],
): SliceSegment[] => {
  if (node.kind === "leaf") return acc;
  const names = featureNames(encoding);
  const name = names[node.axis];
  if (name === "rooms") {
    acc.push({ axis: "rooms", threshold: node.threshold, from: bounds.park0, to: bounds.park1 });
    sliceSegments(node.left, encoding, slice, { ...bounds, rooms1: Math.min(bounds.rooms1, node.threshold) }, acc);
    sliceSegments(node.right, encoding, slice, { ...bounds, rooms0: Math.max(bounds.rooms0, node.threshold) }, acc);
    return acc;
  }
  if (name === "park") {
    acc.push({ axis: "park", threshold: node.threshold, from: bounds.rooms0, to: bounds.rooms1 });
    sliceSegments(node.left, encoding, slice, { ...bounds, park1: Math.min(bounds.park1, node.threshold) }, acc);
    sliceSegments(node.right, encoding, slice, { ...bounds, park0: Math.max(bounds.park0, node.threshold) }, acc);
    return acc;
  }
  const value = encodeApartment(slice, encoding)[node.axis] ?? 0;
  return sliceSegments(value < node.threshold ? node.left : node.right, encoding, slice, bounds, acc);
};

export const contourThreshold = (algorithm: Algorithm, loss: LossKind) =>
  algorithm === "linear" && loss === "mse" ? 0.5 : 0;

export const planeLine = (
  weights: readonly number[],
  bias: number,
  encoding: Encoding,
  slice: Apartment,
  target: number,
) => {
  const names = featureNames(encoding);
  const roomsIndex = names.indexOf("rooms");
  const parkIndex = names.indexOf("park");
  if (roomsIndex < 0 || parkIndex < 0) return null;
  const extra = encodeApartment(slice, encoding).map((value, index) =>
    index === roomsIndex || index === parkIndex ? 0 : value,
  );
  const shift = bias + dot(weights, extra);
  const wRooms = weights[roomsIndex] ?? 0;
  const wPark = weights[parkIndex] ?? 0;
  const xMin = 0.6;
  const xMax = 4.4;
  const yMin = -0.2;
  const yMax = 1.2;
  const hits: { rooms: number; park: number }[] = [];
  const pushUnique = (rooms: number, park: number) => {
    if (rooms < xMin - 1e-6 || rooms > xMax + 1e-6 || park < yMin - 1e-6 || park > yMax + 1e-6) return;
    if (hits.some((hit) => Math.abs(hit.rooms - rooms) < 1e-6 && Math.abs(hit.park - park) < 1e-6)) return;
    hits.push({ rooms, park });
  };
  if (Math.abs(wPark) >= 1e-6) {
    const yAt = (rooms: number) => (target - shift - wRooms * rooms) / wPark;
    pushUnique(xMin, yAt(xMin));
    pushUnique(xMax, yAt(xMax));
  }
  if (Math.abs(wRooms) >= 1e-6) {
    const xAt = (park: number) => (target - shift - wPark * park) / wRooms;
    pushUnique(xAt(yMin), yMin);
    pushUnique(xAt(yMax), yMax);
  }
  if (hits.length < 2) return null;
  return { x1: hits[0].rooms, y1: hits[0].park, x2: hits[1].rooms, y2: hits[1].park, kind: "line" as const };
};
