import { teachingEmbeddings } from "@app/module-sdk";

/**
 * The hand-authored toy encoder, its search arithmetic, and the passage corpus live in the SDK
 * (`teaching-embeddings.ts`) so other labs, such as retrieval-augmented generation, can reuse the
 * same vectors. This file keeps this module's import path stable.
 */
export type Dimension = teachingEmbeddings.Dimension;
export type Passage = teachingEmbeddings.Passage;
export type EncodedToken = teachingEmbeddings.EncodedToken;
export type Metric = teachingEmbeddings.Metric;
export type IndexedPassage = teachingEmbeddings.IndexedPassage;
export type Scored = teachingEmbeddings.Scored;
export type Cells = teachingEmbeddings.Cells;
export type ApproxResult = teachingEmbeddings.ApproxResult;
export const {
  approximateSearch,
  buildIndex,
  contentWords,
  CORPUS,
  cosine,
  DIM,
  DIMENSIONS,
  dot,
  embed,
  exactSearch,
  kmeans,
  lookup,
  nearestCell,
  norm,
  normalize,
  PRESETS,
  recallAtK,
  sharedWords,
  tokens,
  USER_ID,
  words,
  wordVector,
} = teachingEmbeddings;
