export { PlannedLab } from "./PlannedLab";
export * as teachingEmbeddings from "./teaching-embeddings";
export * as teachingTransformer from "./teaching-transformer";
export * as teachingEval from "./teaching-eval";
export * as teachingServing from "./teaching-serving";
export * as teachingMemory from "./teaching-memory";
export * as teachingRl from "./teaching-rl";
export * as teachingRetrieval from "./teaching-retrieval";
export { checkpointQuestions, shuffleCheckpoint } from "./checkpoints";
export type { ShuffledCheckpoint } from "./checkpoints";
export { CardInfoButton, CardInfoProvider, useCardInfo } from "./card-info";
export type { CardInfo, ModuleCardInfo } from "./card-info";
export {
  ArcDiagram,
  BarList,
  computationLinkPath,
  FormulaWithValues,
  Heatmap,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  StageFlow,
  StepThroughController,
  SurfaceHeading,
  smoothLinePath,
  VectorChip,
} from "./visualizations";
export type {
  BarListItem,
  LineSeries,
  PlotPoint,
  SeriesTone,
  StageFlowStage,
} from "./visualizations";
export {
  bigramPairs,
  createTinyWeights,
  decodeTinyIds,
  encodeTinyText,
  extendedScheduleFactor,
  GGUF_FORMATS,
  newtonSchulz,
  pruneTinyWeights,
  quantizeTinyWeights,
  sampleTinyText,
  scheduleFactor,
  TINY_CORPORA,
  TINY_RESERVED,
  TINY_SCHEDULES,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  tinyImplicitReward,
  tinyNextDistribution,
  tinyPerplexity,
  tinyRandom,
  tinyRowDivergence,
  tinySequenceLogProb,
  tinyTopTokens,
  toHalf,
  trainTinyFactored,
  trainTinyDpo,
  trainTinyExtended,
  trainTinyLora,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
} from "./tiny-lm";
export type {
  TinyCheckpoint,
  TinyCorpus,
  TinyFactoredOptions,
  TinyFactoredRun,
  TinyDpoOptions,
  TinyDpoPoint,
  TinyDpoRun,
  TinyExtendedModel,
  TinyExtendedOptions,
  TinyExtendedRun,
  TinyExtendedSchedule,
  TinyFault,
  TinyHistoryPoint,
  TinyLoraOptions,
  TinyLoraRun,
  TinyMixturePhase,
  TinyOptimizerKind,
  TinyOptimizerOptions,
  TinyPreferencePair,
  TinyPruneMode,
  TinyPruneResult,
  TinyQuantOptions,
  TinyQuantResult,
  TinyQuantScope,
  TinySchedule,
  TinyTrainOptions,
  TinyTracePoint,
  TinyTrainRun,
} from "./tiny-lm";
export {
  backpropSnapshot,
  dot,
  mixVectors,
  numericWeightGradient,
  relativeGradientError,
  scaledDotProductAttention,
  stableSoftmax,
} from "./teaching-math";
export type { BackpropSnapshot } from "./teaching-math";
export {
  runTeachingModel,
  runTeachingTransformer,
} from "@app/model-runtime/client";
export {
  teachingAssetCount,
  teachingAssetUrl,
} from "@app/model-runtime/asset-registry";
export type {
  ModelRunResult,
  TransformerRunResult,
  WireTensor,
} from "@app/model-runtime/types";
export { useTeachingTransformer } from "./useTeachingTransformer";
export type { TeachingModelStatus } from "./useTeachingTransformer";
export type {
  Checkpoint,
  ExplanationMode,
  GlossaryTerm,
  LabScene,
  Reference,
  MdxModule,
  ModuleContext,
  ModuleDefinition,
  ModuleGroupId,
  ModuleState,
} from "./types";
export { explanationModeLabel, explanationModes } from "./explanation-modes";
