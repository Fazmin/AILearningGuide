import { sigmoid } from "@app/lib/math";

export interface BackpropSnapshot {
  input: number;
  weight: number;
  target: number;
  product: number;
  prediction: number;
  loss: number;
  dLossPrediction: number;
  dPredictionProduct: number;
  dProductWeight: number;
  dProductInput: number;
  dLossProduct: number;
  dLossWeight: number;
  dLossInput: number;
}

export function backpropSnapshot(
  input: number,
  weight: number,
  target: number,
): BackpropSnapshot {
  const product = input * weight;
  const prediction = sigmoid(product);
  const difference = prediction - target;
  const dLossPrediction = 2 * difference;
  const dPredictionProduct = prediction * (1 - prediction);
  const dProductWeight = input;
  const dProductInput = weight;
  const dLossProduct = dLossPrediction * dPredictionProduct;

  return {
    input,
    weight,
    target,
    product,
    prediction,
    loss: difference ** 2,
    dLossPrediction,
    dPredictionProduct,
    dProductWeight,
    dProductInput,
    dLossProduct,
    dLossWeight: dLossProduct * dProductWeight,
    dLossInput: dLossProduct * dProductInput,
  };
}

export function numericWeightGradient(
  input: number,
  weight: number,
  target: number,
  epsilon = 1e-5,
) {
  const lossAt = (candidate: number) =>
    backpropSnapshot(input, candidate, target).loss;
  return (lossAt(weight + epsilon) - lossAt(weight - epsilon)) / (2 * epsilon);
}

export function relativeGradientError(analytic: number, numeric: number) {
  const scale = Math.max(1e-12, Math.abs(analytic), Math.abs(numeric));
  return Math.abs(analytic - numeric) / scale;
}

export function dot(left: ReadonlyArray<number>, right: ReadonlyArray<number>) {
  return left.reduce(
    (sum, value, index) => sum + value * (right[index] ?? 0),
    0,
  );
}

export function stableSoftmax(values: ReadonlyArray<number>) {
  if (values.length === 0) return [];
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function scaledDotProductAttention(
  query: ReadonlyArray<number>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
  options?: { causalIndex?: number },
) {
  const scale = Math.sqrt(Math.max(1, query.length));
  const causalIndex = options?.causalIndex;
  const scores = keys.map((key) => dot(query, key));
  const scaledScores = scores.map((score, index) =>
    causalIndex !== undefined && index > causalIndex
      ? Number.NEGATIVE_INFINITY
      : score / scale,
  );
  return {
    scores,
    scaledScores,
    weights: stableSoftmax(scaledScores),
    scale,
  };
}

export function mixVectors(
  weights: ReadonlyArray<number>,
  values: ReadonlyArray<ReadonlyArray<number>>,
) {
  const width = values[0]?.length ?? 0;
  return Array.from({ length: width }, (_, dimension) =>
    values.reduce(
      (sum, vector, index) =>
        sum + (weights[index] ?? 0) * (vector[dimension] ?? 0),
      0,
    ),
  );
}
