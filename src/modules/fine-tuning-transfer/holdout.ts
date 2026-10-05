/**
 * The data behind the "Replay and held-out target" card. The target corpus (Recipe steps) is split by
 * sentence: the first sentences are fine-tuned on and the last few are never trained on, which is the
 * only way to learn whether a fine-tune generalized. Replay mixes the first sentences of the original
 * corpus (Harbor weather notes) back into the fine-tuning text. Both are deterministic, so a state of
 * two numbers reproduces the whole run.
 */
import { TINY_CORPORA, encodeTinyText } from "@app/module-sdk";
import { REPLAY_MAX } from "./state";

const sentencesOf = (text: string) =>
  text
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);

export const SOURCE_SENTENCES = sentencesOf(TINY_CORPORA.harbor.text);
export const TARGET_SENTENCES = sentencesOf(TINY_CORPORA.recipes.text);

const asText = (sentences: ReadonlyArray<string>) => (sentences.length ? `${sentences.join(". ")}.` : "");

export interface FineTuneData {
  /** Target sentences the fine-tune trains on. */
  trainTarget: string;
  /** Target sentences it never trains on. Empty when none are held out. */
  heldTarget: string;
  /** Original-corpus sentences mixed into the fine-tuning text. Empty at 0% replay. */
  replay: string;
  /** The original-corpus sentences that were not replayed. Empty at 100% replay. */
  notReplayed: string;
  /** Everything the optimizer sees: the training target followed by the replay. */
  text: string;
  trainSentences: number;
  heldSentences: number;
  replaySentences: number;
  /** Replay sentences as a share of all fine-tuning sentences, 0 to 1. */
  replayFraction: number;
}

export function buildFineTuneData(heldOut: number, replayShare: number): FineTuneData {
  const held = Math.max(0, Math.min(TARGET_SENTENCES.length - 1, Math.round(heldOut)));
  const trainSentences = TARGET_SENTENCES.length - held;
  const replaySentences = Math.round((Math.max(0, Math.min(REPLAY_MAX, replayShare)) / 100) * SOURCE_SENTENCES.length);
  const trainTarget = asText(TARGET_SENTENCES.slice(0, trainSentences));
  const replay = asText(SOURCE_SENTENCES.slice(0, replaySentences));
  const notReplayed = asText(SOURCE_SENTENCES.slice(replaySentences));
  return {
    trainTarget,
    heldTarget: asText(TARGET_SENTENCES.slice(trainSentences)),
    replay,
    notReplayed,
    text: replay ? `${trainTarget} ${replay}` : trainTarget,
    trainSentences,
    heldSentences: held,
    replaySentences,
    replayFraction: replaySentences / (replaySentences + trainSentences),
  };
}

export const encodedLength = (text: string) => encodeTinyText(text).length;
