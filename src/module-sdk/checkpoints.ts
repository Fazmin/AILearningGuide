import type { Checkpoint, ModuleDefinition } from "./types";

/** A module may declare one checkpoint question or a list; consumers always see a list. */
export function checkpointQuestions(definition: Pick<ModuleDefinition, "checkpoint">): Checkpoint[] {
  return Array.isArray(definition.checkpoint) ? definition.checkpoint : [definition.checkpoint];
}

function hashSeed(seed: string) {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed: number) {
  let state = seed || 1;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export interface ShuffledCheckpoint {
  options: string[];
  /** Index of the correct option within the shuffled `options`. */
  answer: number;
}

/**
 * Orders a question's options deterministically from `seed`, so the correct option's
 * position stops depending on how the author happened to write it while a learner who
 * reopens the module sees the same order. Options must therefore never refer to each
 * other by position ("both A and B", "all of the above").
 */
export function shuffleCheckpoint(question: Checkpoint, seed: string): ShuffledCheckpoint {
  const order = question.options.map((_, index) => index);
  const random = seededRandom(hashSeed(seed));
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  return {
    options: order.map((original) => question.options[original]),
    answer: order.indexOf(question.answer),
  };
}
