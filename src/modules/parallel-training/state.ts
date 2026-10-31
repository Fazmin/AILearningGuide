import type { ModuleState } from "@app/module-sdk";
import {
  CHECKPOINT_MINUTES,
  DATA_LADDER,
  EXPERT_LADDER,
  MICRO_BATCH_LADDER,
  MICROBATCHES_LADDER,
  MODELS,
  PIPELINE_LADDER,
  RECOMPUTE,
  RESTART_MINUTES,
  SAVE_SECONDS,
  snapToLadder,
  TENSOR_LADDER,
  type ModelId,
  type Recompute,
  type ZeroStage,
} from "./parallel";

/**
 * Everything the lab stores: the controls, never a result. Every value lands on a value its control can set, so a
 * shared link always shows a layout the sliders could have made. The parallel degrees, the micro-batch size, and
 * the microbatch count snap down to their power-of-two ladders; Expert parallel never exceeds Data parallel, because
 * the expert groups are carved out of the data-parallel group.
 */
export interface ParallelState {
  modelId: ModelId;
  tensor: number;
  pipeline: number;
  data: number;
  expert: number;
  zero: ZeroStage;
  microBatch: number;
  microbatches: number;
  recompute: Recompute;
  overlap: boolean;
  checkpointMinutes: number;
  saveSeconds: number;
  restartMinutes: number;
}

/** The first step's setup: the 8B model on 64 data-parallel GPUs with no sharding, so its state does not fit. */
export const defaultState: ParallelState = {
  modelId: "8b",
  tensor: 1,
  pipeline: 1,
  data: 64,
  expert: 1,
  zero: 0,
  microBatch: 1,
  microbatches: 8,
  recompute: "selective",
  overlap: true,
  checkpointMinutes: 60,
  saveSeconds: 60,
  restartMinutes: 10,
};

export const initialState: ModuleState = { ...defaultState };

/**
 * The Llama 3 405B layout at 16,384 GPUs (Llama 3 paper, Table 4): tensor 8, pipeline 16, data 128, 16 sequences
 * of 8,192 tokens per data-parallel rank per step. ZeRO stage 2 stands for its FSDP setting, which shards the
 * optimizer state and gradients and keeps a stage's parameters gathered from the forward to the backward pass.
 */
export const LLAMA3_LAYOUT: Pick<
  ParallelState,
  "modelId" | "tensor" | "pipeline" | "data" | "expert" | "zero" | "microBatch" | "microbatches" | "recompute" | "overlap"
> = {
  modelId: "405b",
  tensor: 8,
  pipeline: 16,
  data: 128,
  expert: 1,
  zero: 2,
  microBatch: 1,
  microbatches: 16,
  recompute: "selective",
  overlap: true,
};

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** A ladder value, or the fallback for anything that is not a finite number. */
function ladder(value: unknown, steps: ReadonlyArray<number>, fallback: number): number {
  return finite(value) ? snapToLadder(value, steps) : fallback;
}

/** A finite number rounded to the control's step and clamped to its range, or the fallback. */
function stepped(value: unknown, range: { min: number; max: number; step: number }, fallback: number): number {
  if (!finite(value)) return fallback;
  const rounded = Math.round(value / range.step) * range.step;
  return Math.min(range.max, Math.max(range.min, rounded));
}

function oneOf<T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T {
  return typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;
}

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated, snapped, or clamped. */
export function readState(value: unknown): ParallelState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  const data = ladder(source.data, DATA_LADDER, defaultState.data);
  const zero = finite(source.zero) ? (Math.min(3, Math.max(0, Math.round(source.zero))) as ZeroStage) : defaultState.zero;
  return {
    modelId: oneOf(
      source.modelId,
      MODELS.map((model) => model.id),
      defaultState.modelId,
    ),
    tensor: ladder(source.tensor, TENSOR_LADDER, defaultState.tensor),
    pipeline: ladder(source.pipeline, PIPELINE_LADDER, defaultState.pipeline),
    data,
    expert: Math.min(data, ladder(source.expert, EXPERT_LADDER, defaultState.expert)),
    zero,
    microBatch: ladder(source.microBatch, MICRO_BATCH_LADDER, defaultState.microBatch),
    microbatches: ladder(source.microbatches, MICROBATCHES_LADDER, defaultState.microbatches),
    recompute: oneOf(source.recompute, RECOMPUTE, defaultState.recompute),
    overlap: typeof source.overlap === "boolean" ? source.overlap : defaultState.overlap,
    checkpointMinutes: stepped(source.checkpointMinutes, CHECKPOINT_MINUTES, defaultState.checkpointMinutes),
    saveSeconds: stepped(source.saveSeconds, SAVE_SECONDS, defaultState.saveSeconds),
    restartMinutes: stepped(source.restartMinutes, RESTART_MINUTES, defaultState.restartMinutes),
  };
}

export function hydrateParallelState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
