import { teachingTransformer } from "@app/module-sdk";

/**
 * The in-browser transformer runner (weights from the ONNX file, forward pass, and interventions) lives in
 * the SDK (`teaching-transformer.ts`) so other labs can run the same shipped model. This file keeps this
 * module's import path stable.
 */
export type BlockWeights = teachingTransformer.BlockWeights;
export type TransformerWeights = teachingTransformer.TransformerWeights;
export type Intervention = teachingTransformer.Intervention;
export type RunResult = teachingTransformer.RunResult;
export const {
  erf,
  finalLogitsWithPatch,
  rowOf,
  runTransformer,
  softmax,
  weightsFromOnnx,
} = teachingTransformer;
