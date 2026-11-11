/**
 * The arithmetic behind Probability & surprise. Every number the lab shows comes from these
 * functions and from the small authored tables below. Nothing here is measured from a real model:
 * the counts are invented so the sums can be checked by hand.
 */

/** The five candidate next words, in the order every picture draws them. */
export const WORDS = ["blue", "grey", "dark", "clear", "green"] as const;
export const WORD_COUNT = WORDS.length;

/** A bar may not go below 1% or above 96%, so no word is ever impossible and the five still fit. */
export const MIN_BAR = 1;
export const MAX_BAR = 100 - MIN_BAR * (WORD_COUNT - 1);

/**
 * The authored count table: 200 made-up stories, sorted by how they begin (rows) and which word
 * comes next (columns, in WORDS order). The rows have different totals on purpose, so that turning
 * a row into bars needs a real division.
 */
export const PROMPTS = [
  { id: "sky", text: "the sky is", counts: [50, 25, 12, 8, 5] },
  { id: "grass", text: "the grass is", counts: [5, 3, 5, 2, 35] },
  { id: "sea", text: "the sea is", counts: [20, 10, 8, 5, 7] },
] as const;

export const sum = (values: ReadonlyArray<number>) => values.reduce((total, value) => total + value, 0);

/** Shares that add to one, from any non-negative numbers. An all-zero list becomes all-equal. */
export function normalize(values: ReadonlyArray<number>): number[] {
  const total = sum(values);
  if (!(total > 0)) return values.map(() => 1 / Math.max(1, values.length));
  return values.map((value) => value / total);
}

/** "The sky is" bars in whole percent: the lab's default distribution, 50 / 25 / 12 / 8 / 5. */
export const SKY_BARS: readonly number[] = PROMPTS[0].counts;
export const EQUAL_BARS: readonly number[] = [20, 20, 20, 20, 20];

/** P(next word | prompt): one row of the table divided by its own total. */
export const conditional = (prompt: number) => normalize(PROMPTS[prompt]?.counts ?? PROMPTS[0].counts);

/** Whole numbers with a fixed total, as close as possible to the given shares (largest remainder). */
export function roundToTotal(shares: ReadonlyArray<number>, total = 100): number[] {
  const exact = normalize(shares).map((share) => share * total);
  const floors = exact.map((value) => Math.floor(value + 1e-9));
  let left = total - sum(floors);
  const order = exact
    .map((value, index) => ({ index, remainder: value - floors[index] }))
    .sort((a, b) => b.remainder - a.remainder || a.index - b.index);
  for (let turn = 0; left > 0 && order.length > 0; turn += 1, left -= 1) floors[order[turn % order.length].index] += 1;
  return floors;
}

/** True when a list is five whole percents, each between MIN_BAR and MAX_BAR, adding to 100. */
export function isBars(values: unknown): values is number[] {
  return (
    Array.isArray(values) &&
    values.length === WORD_COUNT &&
    values.every((value) => Number.isInteger(value) && value >= MIN_BAR && value <= MAX_BAR) &&
    sum(values as number[]) === 100
  );
}

/**
 * Set one bar to a new whole percent and rescale the others so the five still add to 100. The
 * others keep their proportions to one another, which is what "sharing out one whole" means.
 */
export function setBar(bars: ReadonlyArray<number>, index: number, value: number): number[] {
  const target = Math.min(MAX_BAR, Math.max(MIN_BAR, Math.round(Number.isFinite(value) ? value : bars[index])));
  const others = bars.map((bar, at) => (at === index ? 0 : Math.max(0, bar)));
  const weights = sum(others) > 0 ? others : others.map((_, at) => (at === index ? 0 : 1));
  // Share what is left in proportion, in whole percents.
  const next = roundToTotal(weights, 100 - target);
  // Then lift any bar that fell under the 1% floor, taking the difference from the tallest of the rest.
  for (let at = 0; at < next.length; at += 1) {
    if (at === index) continue;
    while (next[at] < MIN_BAR) {
      let tallest = -1;
      for (let other = 0; other < next.length; other += 1) {
        if (other !== index && other !== at && (tallest < 0 || next[other] > next[tallest])) tallest = other;
      }
      if (tallest < 0 || next[tallest] <= MIN_BAR) break;
      next[tallest] -= 1;
      next[at] += 1;
    }
  }
  next[index] = target;
  return next;
}

/** A small seeded generator (mulberry32), so every "random" draw in the lab can be repeated. */
export function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/** Which outcome a number between 0 and 1 lands on, when the outcomes tile that stretch in order. */
export function pick(shares: ReadonlyArray<number>, u: number) {
  let edge = 0;
  for (let index = 0; index < shares.length; index += 1) {
    edge += shares[index];
    if (u < edge) return index;
  }
  return shares.length - 1;
}

/** `count` seeded draws from a distribution, as outcome indices. */
export function drawSamples(shares: ReadonlyArray<number>, count: number, seed: number): number[] {
  const random = mulberry32(seed);
  const p = normalize(shares);
  return Array.from({ length: Math.max(0, Math.floor(count)) }, () => pick(p, random()));
}

/** How many times each outcome came up. */
export function tally(draws: ReadonlyArray<number>, outcomes = WORD_COUNT): number[] {
  const counts = Array.from({ length: outcomes }, () => 0);
  for (const draw of draws) counts[draw] += 1;
  return counts;
}

/** One outcome's share of the draws so far, after each draw: the line that settles. */
export function runningShare(draws: ReadonlyArray<number>, outcome: number): number[] {
  let hits = 0;
  return draws.map((draw, index) => {
    if (draw === outcome) hits += 1;
    return hits / (index + 1);
  });
}

/** The largest gap, over the outcomes, between a tally's shares and the true shares. */
export function biggestGap(counts: ReadonlyArray<number>, shares: ReadonlyArray<number>) {
  const total = sum(counts);
  if (total === 0) return 0;
  const p = normalize(shares);
  return Math.max(...counts.map((count, index) => Math.abs(count / total - p[index])));
}

/** The long-run average of a number attached to each outcome: sum of chance × value. */
export const expectedValue = (shares: ReadonlyArray<number>, values: ReadonlyArray<number>) =>
  sum(shares.map((share, index) => share * values[index]));

/** The average squared distance from the expected value: sum of chance × (value − mean)². */
export function variance(shares: ReadonlyArray<number>, values: ReadonlyArray<number>) {
  const mean = expectedValue(shares, values);
  return sum(shares.map((share, index) => share * (values[index] - mean) ** 2));
}

/** How many simulated tests the bell card runs for each test size. */
export const TEST_COUNT = 2000;
export const TEST_SIZES = [10, 25, 100, 400] as const;
export const TEST_SEED = 2;

/**
 * Run `tests` tests of `size` questions each, where each answer is a hit with chance `hit`. Returns
 * how many tests ended with 0, 1, …, `size` hits. The score of a test is hits ÷ size.
 */
export function simulateTests(hit: number, size: number, tests = TEST_COUNT, seed = TEST_SEED): number[] {
  const random = mulberry32(seed);
  const pile = Array.from({ length: size + 1 }, () => 0);
  for (let test = 0; test < tests; test += 1) {
    let hits = 0;
    for (let question = 0; question < size; question += 1) if (random() < hit) hits += 1;
    pile[hits] += 1;
  }
  return pile;
}

/** Mean and variance of the test scores in a pile from `simulateTests` (divide by the count). */
export function pileStats(pile: ReadonlyArray<number>) {
  const size = pile.length - 1;
  const tests = sum(pile);
  const scores = pile.map((_, hits) => hits / size);
  const shares = pile.map((count) => count / Math.max(1, tests));
  const mean = expectedValue(shares, scores);
  const spread = variance(shares, scores);
  return { mean, variance: spread, sd: Math.sqrt(spread) };
}

/** Height of the Gaussian (normal) bell with the given mean and variance at x. A density, not a chance. */
export const gaussianPdf = (x: number, mean: number, spread: number) =>
  Math.exp(-((x - mean) ** 2) / (2 * spread)) / Math.sqrt(2 * Math.PI * spread);

/** Share of a bell's area within one standard deviation of its centre. The same for every bell. */
export const ONE_SD_SHARE = 0.6827;

/** Bayes' rule: prior × likelihood for each explanation, rescaled to add to one. */
export function posterior(prior: ReadonlyArray<number>, likelihood: ReadonlyArray<number>) {
  const joint = prior.map((share, index) => share * likelihood[index]);
  const evidence = sum(joint);
  return { joint, evidence, posterior: joint.map((value) => (evidence > 0 ? value / evidence : 0)) };
}

/**
 * The prior over the three prompts when "the sky is" has the given share and the other two split
 * the rest equally, as they do in the count table (100, 50 and 50 stories).
 */
export const priorFor = (skyShare: number) => [skyShare, (1 - skyShare) / 2, (1 - skyShare) / 2];

/** P(word | prompt) for each prompt: one column of the table, each cell divided by its row total. */
export const likelihoodOf = (word: number) => PROMPTS.map((_, prompt) => conditional(prompt)[word]);

/** Surprise in bits: how many halvings it takes to get from 100% down to p. */
export const surprise = (p: number) => (p > 0 ? Math.max(0, -Math.log2(p)) : Infinity);

/** Average surprise in bits. An outcome that never happens adds nothing. */
export const entropy = (p: ReadonlyArray<number>) =>
  sum(p.map((share) => (share > 0 ? share * surprise(share) : 0)));

/** Average surprise when outcomes happen with p but surprise is computed from q. */
export const crossEntropy = (p: ReadonlyArray<number>, q: ReadonlyArray<number>) =>
  sum(p.map((share, index) => (share > 0 ? share * surprise(q[index]) : 0)));

/** The extra bits paid for believing q when the world is p. Never below zero. */
export const klDivergence = (p: ReadonlyArray<number>, q: ReadonlyArray<number>) =>
  Math.max(0, crossEntropy(p, q) - entropy(p));

/** "As uncertain as a fair choice among N": two to the power of the average surprise. */
export const fairChoices = (bits: number) => 2 ** bits;

/**
 * The entropy card's slider. 0 is one sure thing (all on the first word), 50 is the sky bars, 100
 * is all equal; in between the shares move in a straight line from one shape to the next.
 */
export function shapeShares(position: number): number[] {
  const at = Math.min(100, Math.max(0, position));
  const sky = normalize(SKY_BARS);
  const sure = sky.map((_, index) => (index === 0 ? 1 : 0));
  const equal = sky.map(() => 1 / sky.length);
  const [from, to, mix] = at <= 50 ? [sure, sky, at / 50] : [sky, equal, (at - 50) / 50];
  return from.map((value, index) => value + (to[index] - value) * mix);
}

/** Softmax with a temperature: divide each score by T, raise e to it, divide by the total. */
export function softmax(scores: ReadonlyArray<number>, temperature = 1) {
  const t = Math.max(1e-6, temperature);
  const scaled = scores.map((score) => score / t);
  const powers = scaled.map((value) => Math.exp(value));
  const total = sum(powers);
  // The shares are computed from shifted scores so a large score cannot overflow.
  const top = Math.max(...scaled);
  const safe = scaled.map((value) => Math.exp(value - top));
  return { scaled, powers, total, shares: normalize(safe) };
}

/** The sigmoid: squeezes any number into a share between 0 and 1, with sigmoid(0) = 0.5. */
export const sigmoid = (gap: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, gap))));

/** Authored scores whose softmax at temperature 1 gives the sky bars to the nearest percent. */
export const SKY_SCORES: readonly number[] = [2, 1.3, 0.6, 0.2, -0.3];

/** The challenge: three prompts whose true bars stay hidden, each with a seeded tally of 40 draws. */
export const CHALLENGE_DRAWS = 40;
export const CHALLENGE_GOAL = 0.05;
export const CHALLENGE_ROUNDS = [
  { id: "night", text: "the night is", truth: [5, 10, 55, 25, 5], seed: 26 },
  { id: "water", text: "the water is", truth: [25, 10, 12, 38, 15], seed: 46 },
  { id: "forest", text: "the forest is", truth: [6, 8, 35, 6, 45], seed: 36 },
] as const;

export function challengeTally(round: number) {
  const entry = CHALLENGE_ROUNDS[round] ?? CHALLENGE_ROUNDS[0];
  return tally(drawSamples(entry.truth, CHALLENGE_DRAWS, entry.seed));
}

/** Score a guess q against a round's hidden truth: the bill, the best possible bill, and the gap. */
export function challengeScore(round: number, guess: ReadonlyArray<number>) {
  const entry = CHALLENGE_ROUNDS[round] ?? CHALLENGE_ROUNDS[0];
  const p = normalize(entry.truth);
  const q = normalize(guess);
  const best = entropy(p);
  const bill = crossEntropy(p, q);
  return { bill, best, gap: Math.max(0, bill - best) };
}
