import { teachingEval } from "@app/module-sdk";

/**
 * The scoring rule and the statistics behind the leaderboard live in the SDK
 * (`teaching-eval.ts`) so other labs, such as the capstone, can use the same
 * arithmetic. This file keeps this module's import path stable.
 */
export type BenchmarkItem = teachingEval.BenchmarkItem;
export const {
  itemsForHalfWidth,
  meanInterval,
  pairedDifference,
  parseItems,
  passAtK,
  proportionStandardError,
  scoreContinuation,
  tCritical,
  wilsonInterval,
} = teachingEval;
