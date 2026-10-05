import metadata from "./assets/language-models.metadata.json";

/**
 * The loss figures the "Three models, one context" card and the lesson quote, read from
 * `language-models.metadata.json` (written by models/train_language_models.py) and converted from
 * nats to bits. quoted.test.ts re-measures the GRU and transformer figures from the shipped ONNX
 * files and the bigram figures from the corpus.
 *
 * Two protocols, both over consecutive non-overlapping 64-character windows:
 *  - `heldOut`: the last 10% of Tiny Shakespeare. The transformer never trained on it; the GRU
 *    trained on random windows of the whole corpus for a third of an epoch, so for it the tail is
 *    nearly, not strictly, held out; the bigram figure comes from a table fitted on the first 90%
 *    (the shipped table was fitted on all of it).
 *  - `wholeCorpus`: every window. For the transformer that is 90% training text.
 */
export const bitsFromNats = (nats: number) => nats * Math.LOG2E;

const { heldout_tail: tail, whole_corpus: whole } = metadata.evaluation;

export const LOSS_BITS = {
  bigram: {
    heldOut: bitsFromNats(tail.bigram_fitted_on_training_split),
    wholeCorpus: bitsFromNats(whole.bigram_fitted_on_whole_corpus),
  },
  rnn: { heldOut: bitsFromNats(tail.rnn), wholeCorpus: bitsFromNats(whole.rnn) },
  transformer: { heldOut: bitsFromNats(tail.transformer), wholeCorpus: bitsFromNats(whole.transformer) },
} as const;

const rnnHistory = metadata.rnn.history;

/** Optimiser steps each neural model was trained for (the bigram is a count table, not trained by steps). */
export const TRAINING_STEPS = {
  transformer: metadata.transformer.training.steps,
  rnn: rnnHistory[rnnHistory.length - 1]?.step ?? 0,
} as const;

/** Share of the corpus held out from the transformer, as a whole percentage. */
export const HELD_OUT_PERCENT = Math.round((metadata.split.heldout_characters / metadata.corpus_characters) * 100);
