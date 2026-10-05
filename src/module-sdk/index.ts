export { PlannedLab } from "./PlannedLab";
export * as teachingEmbeddings from "./teaching-embeddings";
export * as teachingTransformer from "./teaching-transformer";
export * as teachingEval from "./teaching-eval";
export * as teachingServing from "./teaching-serving";
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
  GGUF_FORMATS,
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
  trainTinyFactored,
  trainTinyDpo,
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
  TinyHistoryPoint,
  TinyLoraOptions,
  TinyLoraRun,
  TinyPreferencePair,
  TinyPruneMode,
  TinyPruneResult,
  TinyQuantOptions,
  TinyQuantResult,
  TinyQuantScope,
  TinySchedule,
  TinyTrainOptions,
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
  Reference,
  MdxModule,
  ModuleContext,
  ModuleDefinition,
  ModuleGroupId,
  ModuleState,
} from "./types";
