/**
 * The lab's experiment: one harbor base, two fine-tunes (recipes and proverbs)
 * from that same base, and the losses of any merge on both target corpora.
 */

import {
  TINY_CORPORA,
  tinyCrossEntropy,
  trainTinyLora,
  trainTinyModel,
} from "@app/module-sdk";
import {
  addToBase,
  mergeTaskVectors,
  taskVector,
  type MergeMethod,
  type MergeParams,
  type MergeResult,
} from "./merge";

export type VectorSource = "full" | "lora";

export const BASE_TEXT = TINY_CORPORA.harbor.text;
export const TASK_A_TEXT = TINY_CORPORA.recipes.text;
export const TASK_B_TEXT = TINY_CORPORA.proverbs.text;
export const BASE_EPOCHS = 48;
export const LORA_RANK = 4;
export const LORA_ALPHA = 8;

export interface TaskLosses {
  recipes: number;
  proverbs: number;
}

export interface TaskVectors {
  source: VectorSource;
  baseWeights: Float32Array;
  taskA: Float32Array;
  taskB: Float32Array;
  base: TaskLosses;
  specialistA: TaskLosses;
  specialistB: TaskLosses;
  trainableA: number;
  trainableB: number;
}

let cachedBase: Float32Array | null = null;

export function baseWeights(): Float32Array {
  if (!cachedBase) {
    cachedBase = trainTinyModel({ text: BASE_TEXT, epochs: BASE_EPOCHS, seed: 1 }).weights;
  }
  return cachedBase;
}

export function lossesOf(weights: Float32Array): TaskLosses {
  return {
    recipes: tinyCrossEntropy(weights, TASK_A_TEXT),
    proverbs: tinyCrossEntropy(weights, TASK_B_TEXT),
  };
}

function fineTune(base: Float32Array, text: string, epochs: number, seed: number, source: VectorSource) {
  if (source === "lora") {
    const run = trainTinyLora({ base, text, rank: LORA_RANK, alpha: LORA_ALPHA, epochs, seed });
    return { vector: run.delta, trainable: run.trainableParameters };
  }
  const run = trainTinyModel({ text, epochs, init: base, seed });
  return { vector: taskVector(run.weights, base), trainable: base.length };
}

export function buildTaskVectors(
  source: VectorSource,
  epochsA: number,
  epochsB: number,
): TaskVectors {
  const base = baseWeights();
  const a = fineTune(base, TASK_A_TEXT, epochsA, 3, source);
  const b = fineTune(base, TASK_B_TEXT, epochsB, 4, source);
  return {
    source,
    baseWeights: base,
    taskA: a.vector,
    taskB: b.vector,
    base: lossesOf(base),
    specialistA: lossesOf(addToBase(base, a.vector)),
    specialistB: lossesOf(addToBase(base, b.vector)),
    trainableA: a.trainable,
    trainableB: b.trainable,
  };
}

export interface MergeOutcome extends TaskLosses {
  result: MergeResult | null;
  /** Merged recipes loss minus the same recipe applied with the proverb vector removed. */
  interferenceA: number;
  /** Merged proverbs loss minus the same recipe applied with the recipe vector removed. */
  interferenceB: number;
}

/**
 * Scores one merge on both corpora and separates interference from shrinkage:
 * the same method is re-run with the other task vector set to zero, so any
 * remaining difference is caused by the other task's presence.
 */
export function lossesFor(
  vectors: TaskVectors,
  method: MergeMethod,
  params: MergeParams,
): MergeOutcome {
  const zero = new Float32Array(vectors.taskA.length);
  const result = mergeTaskVectors(method, vectors.taskA, vectors.taskB, params);
  const merged = result ? lossesOf(addToBase(vectors.baseWeights, result.vector)) : vectors.base;
  const aloneA = mergeTaskVectors(method, vectors.taskA, zero, params);
  const aloneB = mergeTaskVectors(method, zero, vectors.taskB, params);
  const interferenceA = aloneA
    ? merged.recipes - lossesOf(addToBase(vectors.baseWeights, aloneA.vector)).recipes
    : Number.NaN;
  const interferenceB = aloneB
    ? merged.proverbs - lossesOf(addToBase(vectors.baseWeights, aloneB.vector)).proverbs
    : Number.NaN;
  return { ...merged, result, interferenceA, interferenceB };
}
