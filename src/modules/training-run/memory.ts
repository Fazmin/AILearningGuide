import { teachingMemory } from "@app/module-sdk";

/**
 * The training-memory arithmetic lives in the SDK (`teaching-memory.ts`) so other labs, such as
 * Optimizers in practice and Splitting training across GPUs, can count the same bytes. This file
 * keeps this module's import path stable.
 */
export type PrecisionId = teachingMemory.PrecisionId;
export type OptimizerId = teachingMemory.OptimizerId;
export type PrecisionChoice = teachingMemory.PrecisionChoice;
export type OptimizerChoice = teachingMemory.OptimizerChoice;
export type ModelSizeChoice = teachingMemory.ModelSizeChoice;
export type MemoryBreakdown = teachingMemory.MemoryBreakdown;
export const {
  DEFAULT_MODEL_SIZE,
  DEFAULT_OPTIMIZER,
  DEFAULT_PRECISION,
  devicesNeeded,
  formatBytes,
  memoryBreakdown,
  MODEL_SIZES,
  OPTIMIZERS,
  PRECISIONS,
  REFERENCE_DEVICE_BYTES,
  STATE_BYTES,
} = teachingMemory;
