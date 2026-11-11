import {
  POINTS,
  buildForest,
  buildTree,
  fitKMeans,
  fitLinear,
  fitLogistic,
  fitNaiveBayes,
  fitSvm,
  forestVotes,
  knnVote,
  naivePosterior,
  nearestCenter,
  score,
  sigmoid,
  treePredict,
  type Algorithm,
  type GaussClass,
  type TreeNode,
  type Weights,
} from "./fit";

/** One fitted method, in the shape the lab's cards read. Built only from fit.ts, so no number changes here. */
export type Model =
  | { kind: "linear"; weights: Weights }
  | { kind: "logistic"; weights: Weights }
  | { kind: "svm"; weights: Weights }
  | { kind: "knn"; k: number }
  | { kind: "tree"; tree: TreeNode }
  | { kind: "forest"; trees: TreeNode[] }
  | { kind: "naivebayes"; classes: GaussClass[] }
  | ({ kind: "kmeans" } & ReturnType<typeof fitKMeans>);

export const buildModel = (algorithm: Algorithm, k: number, depth: number): Model => {
  if (algorithm === "linear") return { kind: "linear", weights: fitLinear(POINTS) };
  if (algorithm === "logistic") return { kind: "logistic", weights: fitLogistic(POINTS) };
  if (algorithm === "svm") return { kind: "svm", weights: fitSvm(POINTS) };
  if (algorithm === "knn") return { kind: "knn", k };
  if (algorithm === "tree") return { kind: "tree", tree: buildTree(POINTS, depth) };
  if (algorithm === "forest") return { kind: "forest", trees: buildForest(depth) };
  if (algorithm === "naivebayes") return { kind: "naivebayes", classes: fitNaiveBayes(POINTS) };
  return { kind: "kmeans", ...fitKMeans(k) };
};

/** P(y=1 | x) for the two methods that report one; null for the rest. */
export const probabilityOf = (model: Model, x: number, y: number): number | null => {
  if (model.kind === "logistic") return sigmoid(score(model.weights, x, y));
  if (model.kind === "naivebayes") return naivePosterior(model.classes, x, y);
  return null;
};

/** The predicted class, or the cluster id under k-means. */
export const classify = (model: Model, x: number, y: number): number => {
  if (model.kind === "linear") return score(model.weights, x, y) >= 0.5 ? 1 : 0;
  if (model.kind === "logistic" || model.kind === "naivebayes") return probabilityOf(model, x, y)! >= 0.5 ? 1 : 0;
  if (model.kind === "svm") return score(model.weights, x, y) >= 0 ? 1 : 0;
  if (model.kind === "knn") return knnVote(x, y, model.k);
  if (model.kind === "tree") return treePredict(model.tree, x, y);
  if (model.kind === "forest") return forestVotes(model.trees, x, y) * 2 >= model.trees.length ? 1 : 0;
  return nearestCenter(model.centers, x, y);
};

/** The questions a tree asks about one point, root to leaf, and the leaf it lands in. */
export const treePath = (tree: TreeNode, x: number, y: number) => {
  const steps: { axis: "x" | "y"; threshold: number; value: number; yes: boolean }[] = [];
  let node = tree;
  while (node.kind === "split") {
    const value = node.axis === "x" ? x : y;
    const yes = value < node.threshold;
    steps.push({ axis: node.axis, threshold: node.threshold, value, yes });
    node = yes ? node.left : node.right;
  }
  return { steps, leaf: node };
};
