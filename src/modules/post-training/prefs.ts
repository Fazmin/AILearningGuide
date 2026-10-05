/**
 * A whole post-training pipeline on one prompt and six candidate responses.
 *
 * - The "policy" is a softmax over the six responses (each response is one action,
 *   the bandit view of RLHF). Base-model logits are authored.
 * - SFT takes real cross-entropy gradient steps toward two demonstrations.
 * - The reward model is linear in three features the code measures from the text,
 *   fitted to your labeled pairs by Bradley–Terry maximum likelihood.
 * - Policy optimization uses the exact optimum of  E_π[r] − β·KL(π‖π_ref),
 *   which is π*(y) ∝ π_ref(y)·exp(r(y)/β).
 * - "Gold" is an authored careful-rater score used only to measure the result.
 */

export const PROMPT = "Explain why the sky is blue.";

export interface Candidate {
  id: string;
  tag: string;
  text: string;
  /** Authored: does the response name wavelength-dependent scattering? */
  mechanism: 0 | 1;
  /** Authored careful-rater score in [0, 1]. Never used for training. */
  gold: number;
  /** Authored pretrained-model logit. */
  baseLogit: number;
}

export const CANDIDATES: readonly Candidate[] = [
  {
    id: "A",
    tag: "another exam question",
    text: "Q2. Why is the sunset red? Q3. Why is grass green? Q4. Why is the sea blue?",
    mechanism: 0,
    gold: 0,
    baseLogit: 1.6,
  },
  {
    id: "B",
    tag: "short, names scattering",
    text: "Air molecules scatter short blue wavelengths far more than long red ones, so scattered blue light reaches your eyes from every part of the sky.",
    mechanism: 1,
    gold: 0.95,
    baseLogit: -0.3,
  },
  {
    id: "C",
    tag: "praise, then scattering",
    text: "Great question! Sunlight contains every color. As it crosses the atmosphere, nitrogen and oxygen molecules scatter it, and short wavelengths scatter far more strongly than long ones, roughly as one over wavelength to the fourth power. That redirected blue light reaches your eyes from every direction, so the whole sky looks blue.",
    mechanism: 1,
    gold: 0.85,
    baseLogit: -0.8,
  },
  {
    id: "D",
    tag: "short, wrong",
    text: "Because the sky reflects the blue of the oceans.",
    mechanism: 0,
    gold: 0.05,
    baseLogit: 0.7,
  },
  {
    id: "E",
    tag: "praise, long, no mechanism",
    text: "Great question! The color of the sky is a fascinating subject that people have wondered about for centuries. Many different factors play a part, including sunlight, the air, the atmosphere, the time of day, and even the oceans, and all of them work together in complex and wonderful ways to give the sky its familiar and beautiful blue appearance.",
    mechanism: 0,
    gold: 0.1,
    baseLogit: 0.1,
  },
  {
    id: "F",
    tag: "medium, names scattering",
    text: "Sunlight scatters off the gas molecules in air, and shorter wavelengths scatter much more. Blue light is bounced across the whole sky, while at sunset the long path through the air scatters the blue away and leaves the reds.",
    mechanism: 1,
    gold: 1,
    baseLogit: -0.5,
  },
];

export const FEATURE_NAMES = ["Words ÷ 10", "Names mechanism", "Opens with praise"] as const;

export const wordCount = (text: string) => text.trim().split(/\s+/).filter(Boolean).length;
export const opensWithPraise = (text: string) => (/^great question/i.test(text.trim()) ? 1 : 0);

/** φ(y): the only view of a response the reward model gets. Words and praise are measured from the text. */
export const features = (candidate: Candidate) => [
  wordCount(candidate.text) / 10,
  candidate.mechanism,
  opensWithPraise(candidate.text),
];

export const indexOf = (id: string) => CANDIDATES.findIndex((candidate) => candidate.id === id);

export interface PairSpec {
  a: string;
  b: string;
}

/** Six comparisons. Pair 5 is the only one where the response that names the mechanism is the shorter one. */
export const PAIRS: readonly PairSpec[] = [
  { a: "B", b: "D" },
  { a: "F", b: "A" },
  { a: "C", b: "D" },
  { a: "C", b: "B" },
  { a: "E", b: "B" },
  { a: "E", b: "A" },
];

export type Label = "a" | "b" | "none";
export const DEFAULT_LABELS: readonly Label[] = ["a", "a", "a", "a", "none", "a"];

/** Labels persist as a six-character string such as "aaaana"; anything else reads as unlabelled. */
export const parseLabels = (value: unknown): Label[] => {
  const text = typeof value === "string" ? value : "";
  return PAIRS.map((_, index) => {
    const char = text[index];
    return char === "a" ? "a" : char === "b" ? "b" : "none";
  });
};
export const encodeLabels = (labels: ReadonlyArray<Label>) => labels.map((label) => (label === "none" ? "n" : label)).join("");

export const SFT_DEMOS = ["B", "F"] as const;
export const SFT_RATE = 0.8;
export const MAX_SFT_STEPS = 12;

export const softmax = (logits: ReadonlyArray<number>) => {
  const top = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - top));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
};

export const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

/** Cross-entropy SFT on the demonstrations: logits ← logits − η·(p − target), target uniform over the demos. */
export function sftLogits(steps: number, rate = SFT_RATE) {
  const target = CANDIDATES.map((candidate) =>
    (SFT_DEMOS as readonly string[]).includes(candidate.id) ? 1 / SFT_DEMOS.length : 0,
  );
  let logits = CANDIDATES.map((candidate) => candidate.baseLogit);
  const losses = [sftLoss(logits)];
  for (let step = 0; step < steps; step += 1) {
    const p = softmax(logits);
    logits = logits.map((value, index) => value - rate * (p[index] - target[index]));
    losses.push(sftLoss(logits));
  }
  return { logits, losses };
}

export function sftLoss(logits: ReadonlyArray<number>) {
  const p = softmax(logits);
  return -SFT_DEMOS.reduce((sum, id) => sum + Math.log(p[indexOf(id)]), 0) / SFT_DEMOS.length;
}

export interface Comparison {
  winner: number;
  loser: number;
  pair: number;
}

export function comparisonsFrom(labels: ReadonlyArray<Label>): Comparison[] {
  return PAIRS.flatMap((pair, index) => {
    const label = labels[index];
    if (label !== "a" && label !== "b") return [];
    const a = indexOf(pair.a);
    const b = indexOf(pair.b);
    return [{ winner: label === "a" ? a : b, loser: label === "a" ? b : a, pair: index }];
  });
}

export const RM_STEPS = 200;
export const RM_RATE = 0.5;
export const RM_L2 = 0.05;

const dot = (w: ReadonlyArray<number>, x: ReadonlyArray<number>) => w.reduce((sum, value, index) => sum + value * x[index], 0);

export const rewardOf = (weights: ReadonlyArray<number>, index: number) => dot(weights, features(CANDIDATES[index]));

/**
 * Bradley–Terry: P(winner ≻ loser) = σ(r_w − r_l) with r = w·φ. Full-batch gradient
 * ascent on the mean log-likelihood minus (λ/2)‖w‖², from w = 0.
 */
export function fitRewardModel(comparisons: ReadonlyArray<Comparison>, steps = RM_STEPS, rate = RM_RATE, l2 = RM_L2) {
  const dims = FEATURE_NAMES.length;
  let weights = Array(dims).fill(0) as number[];
  const history: { step: number; loss: number }[] = [];
  const phi = CANDIDATES.map(features);
  const loss = (w: number[]) =>
    comparisons.length === 0
      ? Math.log(2)
      : comparisons.reduce((sum, c) => sum - Math.log(sigmoid(dot(w, phi[c.winner]) - dot(w, phi[c.loser]))), 0) /
        comparisons.length;
  history.push({ step: 0, loss: loss(weights) });
  for (let step = 1; step <= steps; step += 1) {
    const gradient = Array(dims).fill(0) as number[];
    for (const c of comparisons) {
      const diff = phi[c.winner].map((value, index) => value - phi[c.loser][index]);
      const scale = 1 - sigmoid(dot(weights, diff));
      diff.forEach((value, index) => {
        gradient[index] += (scale * value) / comparisons.length;
      });
    }
    weights = weights.map((value, index) => value + rate * (gradient[index] - l2 * value));
    history.push({ step, loss: loss(weights) });
  }
  return { weights, history };
}

/** π_β(y) ∝ π_ref(y)·exp(r(y)/β), the maximizer of E_π[r] − β·KL(π‖π_ref). */
export function tiltedPolicy(reference: ReadonlyArray<number>, rewards: ReadonlyArray<number>, beta: number) {
  const logits = reference.map((p, index) => Math.log(p) + rewards[index] / beta);
  return softmax(logits);
}

export const klDivergence = (p: ReadonlyArray<number>, q: ReadonlyArray<number>) =>
  p.reduce((sum, value, index) => (value > 0 ? sum + value * Math.log(value / q[index]) : sum), 0);

export const expectation = (p: ReadonlyArray<number>, values: ReadonlyArray<number>) =>
  p.reduce((sum, value, index) => sum + value * values[index], 0);

export const BETA_MIN = 0.02;
export const BETA_MAX = 20;

export function betaSweep(reference: ReadonlyArray<number>, rewards: ReadonlyArray<number>, points = 60) {
  const gold = CANDIDATES.map((candidate) => candidate.gold);
  return Array.from({ length: points }, (_, index) => {
    const t = index / (points - 1);
    const beta = BETA_MAX * (BETA_MIN / BETA_MAX) ** t;
    const policy = tiltedPolicy(reference, rewards, beta);
    return {
      beta,
      kl: klDivergence(policy, reference),
      reward: expectation(policy, rewards),
      gold: expectation(policy, gold),
    };
  });
}

/** Share of the 15 unordered response pairs that the reward model orders the same way as gold. */
export function goldAgreement(rewards: ReadonlyArray<number>) {
  let agree = 0;
  let total = 0;
  for (let i = 0; i < CANDIDATES.length; i += 1) {
    for (let j = i + 1; j < CANDIDATES.length; j += 1) {
      const goldGap = CANDIDATES[i].gold - CANDIDATES[j].gold;
      if (goldGap === 0) continue;
      total += 1;
      if (Math.sign(rewards[i] - rewards[j]) === Math.sign(goldGap)) agree += 1;
    }
  }
  return { agree, total };
}

/** The labels a rater who scored exactly by gold would give. */
export const goldLabels = (): Label[] =>
  PAIRS.map((pair) => (CANDIDATES[indexOf(pair.a)].gold >= CANDIDATES[indexOf(pair.b)].gold ? "a" : "b"));
