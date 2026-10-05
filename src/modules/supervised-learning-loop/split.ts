/** Counting how a random holdout could fall. Every number is an exact integer count. */

/** n choose k, exact for the small n used here (the running product stays well inside 2^53). */
export const choose = (n: number, k: number) => {
  if (!Number.isInteger(n) || !Number.isInteger(k) || k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= k; i += 1) result = (result * (n - k + i)) / i;
  return Math.round(result);
};

export type HoldoutMix = {
  /** How many high-price rows land in the holdout. */
  highInHoldout: number;
  /** How many of the equally likely holdouts have exactly that many. */
  ways: number;
  /** ways divided by every possible holdout. */
  share: number;
};

/**
 * Every way to draw `holdout` rows out of `high + low` rows, grouped by how many high-price rows the holdout
 * receives: choose(high, k) · choose(low, holdout − k) draws have exactly k.
 */
export const holdoutMix = (high: number, low: number, holdout: number): HoldoutMix[] => {
  const total = choose(high + low, holdout);
  return Array.from({ length: holdout + 1 }, (_, highInHoldout) => {
    const ways = choose(high, highInHoldout) * choose(low, holdout - highInHoldout);
    return { highInHoldout, ways, share: total === 0 ? 0 : ways / total };
  });
};
