/**
 * The rank card again with three numbers per word. The third number, "fast", is invented for this
 * lab like the other two: above zero moves quicker than a person walking, below zero slower. It is
 * tuned so that cat and bicycle, which share a size, differ in fast: squeezing "alive" to zero no
 * longer merges them, and squeezing "fast" as well does.
 */
import { WORDS, type Word } from "./math";

export type Vec3 = readonly [number, number, number];

const FAST: Record<string, number> = {
  mouse: -0.6,
  cat: 0.4,
  dog: 0.7,
  horse: 1.4,
  whale: -0.2,
  bicycle: -0.4,
  car: 1.6,
  bus: 0.6,
};

/** A word as three numbers: size, alive, fast. */
export const vec3Of = (word: Word): Vec3 => [word.x, word.y, FAST[word.id] ?? 0];

/** Multiply the alive number by one factor and the fast number by another; size is left alone. */
export const squeeze3 = (point: Vec3, keepAlive: number, keepFast: number): Vec3 => [point[0], point[1] * keepAlive, point[2] * keepFast];

/** How many of the three directions survive: a direction multiplied by zero is gone. */
export const directionsLeft = (keepAlive: number, keepFast: number) => 1 + (keepAlive > 1e-9 ? 1 : 0) + (keepFast > 1e-9 ? 1 : 0);

const distance3 = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** The two words that end up closest together after the squeeze, and how many distinct spots are left. */
export function closestPair3(keepAlive: number, keepFast: number) {
  const moved = WORDS.map((word) => ({ word, at: squeeze3(vec3Of(word), keepAlive, keepFast) }));
  let pair: [Word, Word] = [WORDS[0], WORDS[1]];
  let gap = Infinity;
  for (let i = 0; i < moved.length; i += 1) {
    for (let j = i + 1; j < moved.length; j += 1) {
      const d = distance3(moved[i].at, moved[j].at);
      if (d < gap - 1e-12) {
        gap = d;
        pair = [moved[i].word, moved[j].word];
      }
    }
  }
  const spots: Vec3[] = [];
  for (const { at } of moved) {
    if (!spots.some((spot) => distance3(spot, at) < 1e-9)) spots.push(at);
  }
  return { pair, gap, distinct: spots.length, merged: gap < 1e-9 };
}
