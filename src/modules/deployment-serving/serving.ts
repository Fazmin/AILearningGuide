import { teachingServing } from "@app/module-sdk";

/**
 * The serving arithmetic lives in the SDK (`teaching-serving.ts`) so other labs,
 * such as the capstone, can use the same stated assumptions. This file keeps this
 * module's import path stable.
 */
export type ServingModel = teachingServing.ServingModel;
export type ServeFormat = teachingServing.ServeFormat;
export type ServingPoint = teachingServing.ServingPoint;
export const {
  BLOCK_TYPE,
  formatBytes,
  GIB,
  HARDWARE,
  kvBytesPerToken,
  MODELS,
  SERVE_FORMATS,
  servingPoint,
  speculativeTokensPerPass,
} = teachingServing;
