/** Eight classical fits on one authored 18-point XOR-ish set. Every readout is computed here. */

export type Point = { x: number; y: number; label: 0 | 1 };
export type Weights = { w0: number; w1: number; w2: number };
export type TreeNode =
  | { kind: "leaf"; label: 0 | 1; count: number; positives: number }
  | { kind: "split"; axis: "x" | "y"; threshold: number; gain: number; left: TreeNode; right: TreeNode };
export type GaussClass = { prior: number; mx: number; my: number; vx: number; vy: number };

export const ALGORITHMS = ["linear", "logistic", "svm", "knn", "tree", "forest", "naivebayes", "kmeans"] as const;
export type Algorithm = (typeof ALGORITHMS)[number];

/**
 * Twelve class-0 points in two diagonal clusters (bottom-left, top-right) and six class-1 points in the
 * other two corners: four top-left, two bottom-right. The uneven XOR lets a straight line take the larger
 * class-1 corner, but never both.
 */
export const POINTS: Point[] = [
  { x: 0.16, y: 0.22, label: 0 },
  { x: 0.22, y: 0.18, label: 0 },
  { x: 0.28, y: 0.3, label: 0 },
  { x: 0.18, y: 0.36, label: 0 },
  { x: 0.34, y: 0.2, label: 0 },
  { x: 0.3, y: 0.42, label: 0 },
  { x: 0.72, y: 0.7, label: 0 },
  { x: 0.8, y: 0.76, label: 0 },
  { x: 0.66, y: 0.82, label: 0 },
  { x: 0.78, y: 0.62, label: 0 },
  { x: 0.88, y: 0.7, label: 0 },
  { x: 0.84, y: 0.84, label: 0 },
  { x: 0.22, y: 0.8, label: 1 },
  { x: 0.14, y: 0.7, label: 1 },
  { x: 0.3, y: 0.9, label: 1 },
  { x: 0.1, y: 0.88, label: 1 },
  { x: 0.8, y: 0.22, label: 1 },
  { x: 0.88, y: 0.3, label: 1 },
];

export const FOREST_SIZE = 7;
export const LOGISTIC_STEPS = 80;
export const LOGISTIC_RATE = 0.35;
export const SVM_STEPS = 3000;
export const SVM_RATE = 0.05;
export const SVM_C = 8;
export const KMEANS_STEPS = 12;

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));

export const score = (weights: Weights, x: number, y: number) => weights.w0 + weights.w1 * x + weights.w2 * y;

const invert3 = (m: number[][]) => {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const s = 1 / (det || 1);
  return [
    [
      s * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
      s * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
      s * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
    ],
    [
      s * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
      s * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
      s * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
    ],
    [
      s * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
      s * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
      s * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
    ],
  ];
};

const dotRow = (row: number[], vector: number[]) => row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2];

/** Ordinary least squares on labels in {0,1}: solves the 3×3 normal equation exactly. */
export const fitLinear = (points: readonly Point[]): Weights => {
  let xx = 0;
  let yy = 0;
  let xy = 0;
  let x1 = 0;
  let y1 = 0;
  let t1 = 0;
  let xt = 0;
  let yt = 0;
  for (const point of points) {
    xx += point.x * point.x;
    yy += point.y * point.y;
    xy += point.x * point.y;
    x1 += point.x;
    y1 += point.y;
    t1 += point.label;
    xt += point.x * point.label;
    yt += point.y * point.label;
  }
  const inverse = invert3([
    [points.length, x1, y1],
    [x1, xx, xy],
    [y1, xy, yy],
  ]);
  const target = [t1, xt, yt];
  return { w0: dotRow(inverse[0], target), w1: dotRow(inverse[1], target), w2: dotRow(inverse[2], target) };
};

/** Full-batch gradient descent on summed log-loss from w = 0. */
export const fitLogistic = (points: readonly Point[], steps = LOGISTIC_STEPS): Weights => {
  let w0 = 0;
  let w1 = 0;
  let w2 = 0;
  for (let step = 0; step < steps; step += 1) {
    let g0 = 0;
    let g1 = 0;
    let g2 = 0;
    for (const point of points) {
      const error = sigmoid(w0 + w1 * point.x + w2 * point.y) - point.label;
      g0 += error;
      g1 += error * point.x;
      g2 += error * point.y;
    }
    w0 -= LOGISTIC_RATE * g0;
    w1 -= LOGISTIC_RATE * g1;
    w2 -= LOGISTIC_RATE * g2;
  }
  return { w0, w1, w2 };
};

export const logLoss = (weights: Weights, points: readonly Point[]) =>
  points.reduce((sum, point) => {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(score(weights, point.x, point.y))));
    return sum - (point.label * Math.log(p) + (1 - point.label) * Math.log(1 - p));
  }, 0) / points.length;

/**
 * Soft-margin linear SVM, ½‖w‖² + C Σ max(0, 1 − t(w·x+b)) with t ∈ {−1,+1}, the bias unregularized.
 * Primal subgradient steps with a shrinking step size η₀/√(step+1); the lowest-objective iterate is kept.
 */
export const fitSvm = (points: readonly Point[], steps = SVM_STEPS): Weights => {
  let current: Weights = { w0: 0, w1: 0, w2: 0 };
  let best = current;
  let bestObjective = svmObjective(current, points);
  for (let step = 0; step < steps; step += 1) {
    let g0 = 0;
    let g1 = current.w1;
    let g2 = current.w2;
    for (const point of points) {
      const t = point.label === 1 ? 1 : -1;
      if (t * score(current, point.x, point.y) < 1) {
        g0 -= SVM_C * t;
        g1 -= SVM_C * t * point.x;
        g2 -= SVM_C * t * point.y;
      }
    }
    const rate = SVM_RATE / Math.sqrt(step + 1);
    current = { w0: current.w0 - rate * g0, w1: current.w1 - rate * g1, w2: current.w2 - rate * g2 };
    const objective = svmObjective(current, points);
    if (objective < bestObjective) {
      best = current;
      bestObjective = objective;
    }
  }
  return best;
};

export function svmObjective(weights: Weights, points: readonly Point[]) {
  return (
  0.5 * (weights.w1 ** 2 + weights.w2 ** 2) +
  SVM_C *
    points.reduce((sum, point) => {
      const t = point.label === 1 ? 1 : -1;
      return sum + Math.max(0, 1 - t * score(weights, point.x, point.y));
    }, 0)
  );
}

/** Points on or inside the margin: t(w·x+b) ≤ 1. */
export const supportVectors = (weights: Weights, points: readonly Point[]) =>
  points.filter((point) => {
    const t = point.label === 1 ? 1 : -1;
    return t * score(weights, point.x, point.y) <= 1 + 1e-3;
  });

export const knnRanked = (x: number, y: number, points: readonly Point[] = POINTS) =>
  points
    .map((point, index) => ({ ...point, index, distance: Math.hypot(point.x - x, point.y - y) }))
    .sort((left, right) => left.distance - right.distance || left.index - right.index);

/** Majority of the k nearest. An even split goes to class 1. */
export const knnVote = (x: number, y: number, k: number, points: readonly Point[] = POINTS): 0 | 1 => {
  const votes = knnRanked(x, y, points)
    .slice(0, k)
    .reduce((sum, point) => sum + point.label, 0);
  return votes * 2 >= k ? 1 : 0;
};

/** Binary Gini impurity 1 − p² − (1−p)² = 2p(1−p). */
export const gini = (points: readonly Point[]) => {
  if (points.length === 0) return 0;
  const p = points.reduce((sum, point) => sum + point.label, 0) / points.length;
  return 2 * p * (1 - p);
};

export const bestSplit = (points: readonly Point[]) => {
  let winner = { axis: "x" as "x" | "y", threshold: 0.5, gain: -1 };
  const parent = gini(points);
  for (const axis of ["x", "y"] as const) {
    const values = [...new Set(points.map((point) => point[axis]))].sort((a, b) => a - b);
    for (let index = 0; index < values.length - 1; index += 1) {
      const threshold = (values[index] + values[index + 1]) / 2;
      const left = points.filter((point) => point[axis] < threshold);
      const right = points.filter((point) => point[axis] >= threshold);
      const gain =
        parent - (left.length / points.length) * gini(left) - (right.length / points.length) * gini(right);
      if (gain > winner.gain) winner = { axis, threshold, gain };
    }
  }
  return winner;
};

/** The numbers behind one candidate cut: parent and child Gini, the size-weighted child Gini, and the gain. */
export const scoreSplit = (points: readonly Point[], axis: "x" | "y", threshold: number) => {
  const left = points.filter((point) => point[axis] < threshold);
  const right = points.filter((point) => point[axis] >= threshold);
  const parent = gini(points);
  const weighted =
    (left.length / points.length) * gini(left) + (right.length / points.length) * gini(right);
  return {
    parent,
    leftCount: left.length,
    rightCount: right.length,
    leftPositives: left.reduce((sum, point) => sum + point.label, 0),
    rightPositives: right.reduce((sum, point) => sum + point.label, 0),
    leftGini: gini(left),
    rightGini: gini(right),
    weighted,
    gain: parent - weighted,
  };
};

const leafOf = (points: readonly Point[]): TreeNode => {
  const positives = points.reduce((sum, point) => sum + point.label, 0);
  return { kind: "leaf", label: positives * 2 >= points.length ? 1 : 0, count: points.length, positives };
};

/** Greedy CART on Gini gain. A tied leaf goes to class 1. */
export const buildTree = (points: readonly Point[], depth: number): TreeNode => {
  if (depth <= 0 || points.length === 0 || points.every((point) => point.label === points[0].label)) {
    return leafOf(points);
  }
  const split = bestSplit(points);
  if (split.gain <= 1e-6) return leafOf(points);
  const left = points.filter((point) => point[split.axis] < split.threshold);
  const right = points.filter((point) => point[split.axis] >= split.threshold);
  if (left.length === 0 || right.length === 0) return leafOf(points);
  return {
    kind: "split",
    axis: split.axis,
    threshold: split.threshold,
    gain: split.gain,
    left: buildTree(left, depth - 1),
    right: buildTree(right, depth - 1),
  };
};

export const treePredict = (node: TreeNode, x: number, y: number): 0 | 1 => {
  if (node.kind === "leaf") return node.label;
  return treePredict((node.axis === "x" ? x : y) < node.threshold ? node.left : node.right, x, y);
};

export const collectSplits = (node: TreeNode, acc: { axis: "x" | "y"; threshold: number }[] = []) => {
  if (node.kind === "split") {
    acc.push({ axis: node.axis, threshold: node.threshold });
    collectSplits(node.left, acc);
    collectSplits(node.right, acc);
  }
  return acc;
};

export const countLeaves = (node: TreeNode): number =>
  node.kind === "leaf" ? 1 : countLeaves(node.left) + countLeaves(node.right);

/** Deterministic stand-in for a random draw, so the forest is the same on every render. */
const unitRandom = (seed: number) => {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
};

export const bootstrapSample = (tree: number, points: readonly Point[] = POINTS) =>
  Array.from({ length: points.length }, (_, index) => points[Math.floor(unitRandom(tree * 31 + index + 3) * points.length)]);

export const buildForest = (depth: number, points: readonly Point[] = POINTS) =>
  Array.from({ length: FOREST_SIZE }, (_, tree) => buildTree(bootstrapSample(tree, points), depth));

export const forestVotes = (trees: readonly TreeNode[], x: number, y: number) =>
  trees.reduce((sum, tree) => sum + treePredict(tree, x, y), 0);

export const fitKMeans = (k: number, points: readonly Point[] = POINTS) => {
  let centers = Array.from({ length: k }, (_, index) => ({
    x: 0.2 + (index * 0.6) / Math.max(1, k - 1),
    y: 0.25 + ((index * 17) % 7) * 0.08,
  }));
  const nearest = (x: number, y: number) => {
    let best = 0;
    let bestDistance = Infinity;
    centers.forEach((center, index) => {
      const distance = (x - center.x) ** 2 + (y - center.y) ** 2;
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  };
  let assignments = points.map(() => 0);
  for (let step = 0; step < KMEANS_STEPS; step += 1) {
    assignments = points.map((point) => nearest(point.x, point.y));
    centers = centers.map((center, index) => {
      const members = points.filter((_, pointIndex) => assignments[pointIndex] === index);
      if (members.length === 0) return center;
      return {
        x: members.reduce((sum, point) => sum + point.x, 0) / members.length,
        y: members.reduce((sum, point) => sum + point.y, 0) / members.length,
      };
    });
  }
  const inertia = points.reduce((sum, point) => {
    const center = centers[nearest(point.x, point.y)];
    return sum + (point.x - center.x) ** 2 + (point.y - center.y) ** 2;
  }, 0);
  return { centers, assignments, inertia };
};

export const nearestCenter = (centers: ReadonlyArray<{ x: number; y: number }>, x: number, y: number) => {
  let best = 0;
  let bestDistance = Infinity;
  centers.forEach((center, index) => {
    const distance = (x - center.x) ** 2 + (y - center.y) ** 2;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
};

/** Per class: prior, and a 1-D Gaussian (maximum-likelihood mean and variance) on each coordinate. */
export const fitNaiveBayes = (points: readonly Point[]): GaussClass[] =>
  ([0, 1] as const).map((label) => {
    const members = points.filter((point) => point.label === label);
    const n = members.length || 1;
    const mx = members.reduce((sum, point) => sum + point.x, 0) / n;
    const my = members.reduce((sum, point) => sum + point.y, 0) / n;
    const vx = Math.max(1e-3, members.reduce((sum, point) => sum + (point.x - mx) ** 2, 0) / n);
    const vy = Math.max(1e-3, members.reduce((sum, point) => sum + (point.y - my) ** 2, 0) / n);
    return { prior: members.length / points.length, mx, my, vx, vy };
  });

const logGauss = (value: number, mean: number, variance: number) =>
  -0.5 * Math.log(2 * Math.PI * variance) - (value - mean) ** 2 / (2 * variance);

export const naiveLogScore = (model: GaussClass, x: number, y: number) =>
  Math.log(Math.max(1e-9, model.prior)) + logGauss(x, model.mx, model.vx) + logGauss(y, model.my, model.vy);

/** P(c=1 | x, y) under the naive factorization, normalized over the two classes. */
export const naivePosterior = (classes: readonly GaussClass[], x: number, y: number) => {
  const log0 = naiveLogScore(classes[0], x, y);
  const log1 = naiveLogScore(classes[1], x, y);
  const shift = Math.max(log0, log1);
  const p0 = Math.exp(log0 - shift);
  const p1 = Math.exp(log1 - shift);
  return p1 / (p0 + p1);
};

/** Height of a 1-D Gaussian with the given mean and variance. */
export const gaussianDensity = (value: number, mean: number, variance: number) =>
  Math.exp(-((value - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

/** The three factors of each class's unnormalized score: prior, density of x, density of y, and their product. */
export const naiveFactors = (classes: readonly GaussClass[], x: number, y: number) =>
  classes.map((model) => {
    const densityX = gaussianDensity(x, model.mx, model.vx);
    const densityY = gaussianDensity(y, model.my, model.vy);
    return { prior: model.prior, densityX, densityY, score: model.prior * densityX * densityY };
  });

/** Pearson correlation of x and y within one class: the quantity the naive assumption sets to zero. */
export const classCorrelation = (points: readonly Point[], label: 0 | 1) => {
  const members = points.filter((point) => point.label === label);
  const mx = members.reduce((sum, point) => sum + point.x, 0) / members.length;
  const my = members.reduce((sum, point) => sum + point.y, 0) / members.length;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (const point of members) {
    covariance += (point.x - mx) * (point.y - my);
    varianceX += (point.x - mx) ** 2;
    varianceY += (point.y - my) ** 2;
  }
  return covariance / Math.sqrt(varianceX * varianceY);
};

/** Street width 2/‖w‖ of a linear classifier. */
export const marginWidthOf = (weights: Weights) => 2 / Math.max(1e-9, Math.hypot(weights.w1, weights.w2));

/** Where w0 + w1 x + w2 y = target crosses the unit square, or null if it misses the square. */
export const lineInSquare = (weights: Weights, target: number) => {
  const hits: { x: number; y: number }[] = [];
  const push = (x: number, y: number) => {
    if (x < -1e-9 || x > 1 + 1e-9 || y < -1e-9 || y > 1 + 1e-9) return;
    if (hits.some((hit) => Math.abs(hit.x - x) < 1e-9 && Math.abs(hit.y - y) < 1e-9)) return;
    hits.push({ x, y });
  };
  if (Math.abs(weights.w2) > 1e-9) {
    push(0, (target - weights.w0) / weights.w2);
    push(1, (target - weights.w0 - weights.w1) / weights.w2);
  }
  if (Math.abs(weights.w1) > 1e-9) {
    push((target - weights.w0) / weights.w1, 0);
    push((target - weights.w0 - weights.w2) / weights.w1, 1);
  }
  if (hits.length < 2) return null;
  return { x1: hits[0].x, y1: hits[0].y, x2: hits[1].x, y2: hits[1].y };
};

export type SplitSegment = { axis: "x" | "y"; threshold: number; from: number; to: number; depth: number };

/**
 * Each split drawn only across the region its parent node owns. A depth-2 cut does not span the square;
 * it lives inside the rectangle the first cut created.
 */
export const treeSegments = (
  node: TreeNode,
  bounds = { x0: 0, x1: 1, y0: 0, y1: 1 },
  depth = 0,
  acc: SplitSegment[] = [],
): SplitSegment[] => {
  if (node.kind === "leaf") return acc;
  if (node.axis === "x") {
    acc.push({ axis: "x", threshold: node.threshold, from: bounds.y0, to: bounds.y1, depth });
    treeSegments(node.left, { ...bounds, x1: Math.min(bounds.x1, node.threshold) }, depth + 1, acc);
    treeSegments(node.right, { ...bounds, x0: Math.max(bounds.x0, node.threshold) }, depth + 1, acc);
  } else {
    acc.push({ axis: "y", threshold: node.threshold, from: bounds.x0, to: bounds.x1, depth });
    treeSegments(node.left, { ...bounds, y1: Math.min(bounds.y1, node.threshold) }, depth + 1, acc);
    treeSegments(node.right, { ...bounds, y0: Math.max(bounds.y0, node.threshold) }, depth + 1, acc);
  }
  return acc;
};

/** Number of stored numbers after fitting, so "what it stored" is a count, not a slogan. */
export const storedCount = (algorithm: Algorithm, k: number, depth: number) => {
  if (algorithm === "linear" || algorithm === "logistic" || algorithm === "svm") return 3;
  if (algorithm === "knn") return POINTS.length * 3;
  if (algorithm === "naivebayes") return 2 * 4 + 1;
  if (algorithm === "kmeans") return k * 2;
  const tree = buildTree(POINTS, depth);
  if (algorithm === "tree") return 2 * (countLeaves(tree) - 1) + countLeaves(tree);
  return buildForest(depth).reduce((sum, item) => sum + 2 * (countLeaves(item) - 1) + countLeaves(item), 0);
};
