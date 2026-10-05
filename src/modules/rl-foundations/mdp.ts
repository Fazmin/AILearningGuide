/**
 * A 5×4 gridworld MDP solved by value iteration, a seeded rollout of the greedy
 * policy, and a seeded Bernoulli bandit for ε-greedy exploration. Everything the
 * lab draws is computed here, so the numbers on screen are the numbers the
 * Bellman backup produces.
 */

export const COLS = 5;
export const ROWS = 4;
export const CELLS = COLS * ROWS;
export const START = 15;
export const GOAL = 19;
export const TRAP = 17;
export const BONUS = 4;

export const STEP_REWARD = -0.04;
export const GOAL_REWARD = 1;
export const TRAP_REWARD = -1;
export const BONUS_REWARD = 0.3;

export const MAX_SWEEPS = 60;
export const MAX_EPISODE_STEPS = 24;

export const ACTIONS = ["up", "right", "down", "left"] as const;
export type Action = (typeof ACTIONS)[number];
export const ARROWS: Record<Action, string> = { up: "↑", right: "→", down: "↓", left: "←" };

export interface WorldConfig {
  gamma: number;
  slip: number;
  bonus: boolean;
}

export const rowOf = (cell: number) => Math.floor(cell / COLS);
export const colOf = (cell: number) => cell % COLS;
export const isTerminal = (cell: number) => cell === GOAL || cell === TRAP;

/** Deterministic move. Walking off the grid leaves the agent where it was. */
export function move(cell: number, action: Action) {
  const row = rowOf(cell);
  const col = colOf(cell);
  if (action === "up") return row > 0 ? cell - COLS : cell;
  if (action === "down") return row < ROWS - 1 ? cell + COLS : cell;
  if (action === "left") return col > 0 ? cell - 1 : cell;
  return col < COLS - 1 ? cell + 1 : cell;
}

const perpendicular = (action: Action): [Action, Action] =>
  action === "up" || action === "down" ? ["left", "right"] : ["up", "down"];

/** P(s' | s, a): the intended move with 1 − slip, each perpendicular move with slip / 2. */
export function transitions(cell: number, action: Action, slip: number) {
  const outcomes = new Map<number, number>();
  const add = (next: number, probability: number) => {
    if (probability <= 0) return;
    outcomes.set(next, (outcomes.get(next) ?? 0) + probability);
  };
  add(move(cell, action), 1 - slip);
  const [a, b] = perpendicular(action);
  add(move(cell, a), slip / 2);
  add(move(cell, b), slip / 2);
  return [...outcomes.entries()].map(([next, probability]) => ({ next, probability }));
}

/**
 * r(s, a, s'): paid on entering s'. Bumping a wall re-enters nothing, so it costs a
 * step. The bonus tile pays every time it is entered and does not end the episode.
 */
export function reward(from: number, to: number, bonus: boolean) {
  if (to === from) return STEP_REWARD;
  if (to === GOAL) return GOAL_REWARD;
  if (to === TRAP) return TRAP_REWARD;
  if (bonus && to === BONUS) return BONUS_REWARD;
  return STEP_REWARD;
}

/** Q(s, a) = Σ_s' P(s'|s,a) [ r(s,a,s') + γ V(s') ]. */
export function qValue(values: ReadonlyArray<number>, cell: number, action: Action, config: WorldConfig) {
  return transitions(cell, action, config.slip).reduce(
    (sum, { next, probability }) =>
      sum + probability * (reward(cell, next, config.bonus) + config.gamma * (isTerminal(next) ? 0 : values[next])),
    0,
  );
}

export function qValues(values: ReadonlyArray<number>, cell: number, config: WorldConfig) {
  return ACTIONS.map((action) => ({ action, q: qValue(values, cell, action, config) }));
}

export interface Sweep {
  values: number[];
  /** max_s |V_k(s) − V_{k−1}(s)|; 0 for k = 0. */
  residual: number;
}

/** Value iteration from V_0 = 0. Entry k holds V_k after k synchronous Bellman optimality backups. */
export function valueIteration(config: WorldConfig, sweeps = MAX_SWEEPS): Sweep[] {
  const history: Sweep[] = [{ values: Array(CELLS).fill(0), residual: 0 }];
  for (let k = 1; k <= sweeps; k += 1) {
    const previous = history[k - 1].values;
    const next = previous.map((_, cell) =>
      isTerminal(cell) ? 0 : Math.max(...qValues(previous, cell, config).map((entry) => entry.q)),
    );
    const residual = Math.max(...next.map((value, cell) => Math.abs(value - previous[cell])));
    history.push({ values: next, residual });
  }
  return history;
}

const TIE = 1e-9;

/** Greedy actions with respect to V. More than one entry means a tie. */
export function greedyActions(values: ReadonlyArray<number>, cell: number, config: WorldConfig): Action[] {
  if (isTerminal(cell)) return [];
  const scored = qValues(values, cell, config);
  const best = Math.max(...scored.map((entry) => entry.q));
  return scored.filter((entry) => best - entry.q < TIE).map((entry) => entry.action);
}

/** A small, fast, seedable PRNG (mulberry32). */
export function seededRandom(seed: number) {
  let state = (Math.floor(seed) * 0x9e3779b1) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface EpisodeStep {
  from: number;
  action: Action;
  to: number;
  reward: number;
  slipped: boolean;
}

export interface Episode {
  steps: EpisodeStep[];
  cells: number[];
  outcome: "goal" | "trap" | "cap";
}

/**
 * Roll out the greedy policy for V from START. Ties take the first action in
 * up, right, down, left order. Slips draw from the seeded generator.
 */
export function rollout(values: ReadonlyArray<number>, config: WorldConfig, seed: number): Episode {
  const random = seededRandom(seed);
  const steps: EpisodeStep[] = [];
  const cells = [START];
  let cell = START;
  while (!isTerminal(cell) && steps.length < MAX_EPISODE_STEPS) {
    const action = greedyActions(values, cell, config)[0] ?? "right";
    const draw = random();
    let direction: Action = action;
    if (draw >= 1 - config.slip) {
      const [a, b] = perpendicular(action);
      direction = draw < 1 - config.slip / 2 ? a : b;
    }
    const to = move(cell, direction);
    steps.push({ from: cell, action, to, reward: reward(cell, to, config.bonus), slipped: direction !== action });
    cells.push(to);
    cell = to;
  }
  return { steps, cells, outcome: cell === GOAL ? "goal" : cell === TRAP ? "trap" : "cap" };
}

/** G_0 = Σ_t γ^t r_{t+1} over the first `upto` rewards. */
export function discountedReturn(rewards: ReadonlyArray<number>, gamma: number, upto = rewards.length) {
  let total = 0;
  for (let t = 0; t < Math.min(upto, rewards.length); t += 1) total += gamma ** t * rewards[t];
  return total;
}

/* ------------------------------------------------------------------ bandit */

export const ARM_MEANS = [0.3, 0.55, 0.45, 0.7] as const;
export const BANDIT_PULLS = 400;
export const BANDIT_RUNS = 300;
export const BEST_ARM = ARM_MEANS.indexOf(Math.max(...ARM_MEANS) as (typeof ARM_MEANS)[number]);

export interface BanditRun {
  pulls: number[];
  estimates: number[];
  rewards: number[];
  choices: number[];
}

/**
 * ε-greedy on Bernoulli arms with sample-average estimates starting at 0. With
 * probability ε pick an arm uniformly; otherwise pick the highest estimate, breaking
 * ties uniformly at random.
 */
export function runBandit(epsilon: number, seed: number, pulls = BANDIT_PULLS): BanditRun {
  const random = seededRandom(seed * 7919 + 17);
  const arms = ARM_MEANS.length;
  const counts = Array(arms).fill(0);
  const estimates = Array(arms).fill(0);
  const rewards: number[] = [];
  const choices: number[] = [];
  for (let t = 0; t < pulls; t += 1) {
    let arm: number;
    if (random() < epsilon) {
      arm = Math.floor(random() * arms);
    } else {
      const best = Math.max(...estimates);
      const tied = estimates.map((value, index) => (value === best ? index : -1)).filter((index) => index >= 0);
      arm = tied[Math.floor(random() * tied.length)];
    }
    const paid = random() < ARM_MEANS[arm] ? 1 : 0;
    counts[arm] += 1;
    estimates[arm] += (paid - estimates[arm]) / counts[arm];
    rewards.push(paid);
    choices.push(arm);
  }
  return { pulls: counts, estimates, rewards, choices };
}

/** Mean over many seeded runs of the running-average reward and of the best-arm share. */
export function banditAverages(epsilon: number, runs = BANDIT_RUNS, pulls = BANDIT_PULLS) {
  const rewardAt = Array(pulls).fill(0);
  const bestAt = Array(pulls).fill(0);
  let finalBestShare = 0;
  for (let run = 1; run <= runs; run += 1) {
    const result = runBandit(epsilon, 1000 + run, pulls);
    result.rewards.forEach((value, t) => {
      rewardAt[t] += value;
    });
    result.choices.forEach((arm, t) => {
      if (arm === BEST_ARM) bestAt[t] += 1;
    });
    finalBestShare += result.pulls[BEST_ARM] / pulls;
  }
  let cumulative = 0;
  const meanRewardSoFar = rewardAt.map((value, t) => {
    cumulative += value / runs;
    return cumulative / (t + 1);
  });
  return {
    meanRewardSoFar,
    bestArmRate: bestAt.map((value) => value / runs),
    finalBestShare: finalBestShare / runs,
  };
}

/* ---------------------------------------------------------- policy gradient */

export const PG_MIN_RATE = 0.02;
export const PG_MAX_RATE = 1;

export interface PolicyGradientRun {
  /** The policy's probability on the best arm before each pull. */
  bestProbability: number[];
  /** The policy over the four arms after the last update. */
  probabilities: number[];
  rewards: number[];
  choices: number[];
}

/** Softmax of the arm preferences, shifted by the largest so nothing overflows. */
export function softmax(preferences: ReadonlyArray<number>) {
  const largest = Math.max(...preferences);
  const exponentials = preferences.map((value) => Math.exp(value - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/** ∂ ln π(arm) / ∂θ_i for a softmax policy: 1 for the pulled arm, minus the probability of each arm. */
export function logPolicyGradient(policy: ReadonlyArray<number>, arm: number) {
  return policy.map((probability, index) => (index === arm ? 1 : 0) - probability);
}

/**
 * REINFORCE on the same four Bernoulli arms. The policy is a softmax over four preferences θ, all 0 at the
 * start, so each arm has probability 0.25. Every pull samples an arm from the policy and a 0 or 1 reward from
 * the arm, then moves each preference by α·(r − b)·(1[i = arm] − π_i), which is α·(r − b)·∂ln π(arm)/∂θ_i.
 * With the baseline on, b is the average reward of the earlier pulls; with it off, b = 0.
 */
export function runPolicyGradient(
  learningRate: number,
  baseline: boolean,
  seed: number,
  pulls = BANDIT_PULLS,
): PolicyGradientRun {
  const random = seededRandom(seed * 7919 + 17);
  const preferences = Array<number>(ARM_MEANS.length).fill(0);
  const bestProbability: number[] = [];
  const rewards: number[] = [];
  const choices: number[] = [];
  let average = 0;
  for (let pull = 0; pull < pulls; pull += 1) {
    const policy = softmax(preferences);
    bestProbability.push(policy[BEST_ARM]);
    let draw = random();
    let arm = policy.length - 1;
    for (let index = 0; index < policy.length - 1; index += 1) {
      if (draw < policy[index]) {
        arm = index;
        break;
      }
      draw -= policy[index];
    }
    const paid = random() < ARM_MEANS[arm] ? 1 : 0;
    const advantage = paid - (baseline ? average : 0);
    const direction = logPolicyGradient(policy, arm);
    for (let index = 0; index < preferences.length; index += 1) {
      preferences[index] += learningRate * advantage * direction[index];
    }
    average += (paid - average) / (pull + 1);
    rewards.push(paid);
    choices.push(arm);
  }
  return { bestProbability, probabilities: softmax(preferences), rewards, choices };
}

type PolicyGradientAverages = ReturnType<typeof computePolicyGradientAverages>;
const averagesMemo = new Map<string, PolicyGradientAverages>();

/**
 * Means over many seeded runs, the same seeds `banditAverages` uses so the two cards compare like with like.
 * The result depends only on its arguments, so an identical earlier answer is reused.
 */
export function policyGradientAverages(
  learningRate: number,
  baseline: boolean,
  runs = BANDIT_RUNS,
  pulls = BANDIT_PULLS,
): PolicyGradientAverages {
  const key = `${learningRate}|${baseline}|${runs}|${pulls}`;
  const found = averagesMemo.get(key);
  if (found) return found;
  const made = computePolicyGradientAverages(learningRate, baseline, runs, pulls);
  if (averagesMemo.size >= 40) averagesMemo.delete(averagesMemo.keys().next().value as string);
  averagesMemo.set(key, made);
  return made;
}

function computePolicyGradientAverages(
  learningRate: number,
  baseline: boolean,
  runs: number,
  pulls: number,
) {
  const rewardAt = Array<number>(pulls).fill(0);
  const bestAt = Array<number>(pulls).fill(0);
  let finalBest = 0;
  let stuck = 0;
  let lastWindow = 0;
  for (let run = 1; run <= runs; run += 1) {
    const result = runPolicyGradient(learningRate, baseline, 1000 + run, pulls);
    result.rewards.forEach((value, pull) => {
      rewardAt[pull] += value;
    });
    result.bestProbability.forEach((value, pull) => {
      bestAt[pull] += value;
    });
    const final = result.probabilities[BEST_ARM];
    finalBest += final;
    if (final < 0.5) stuck += 1;
    lastWindow += result.rewards.slice(-100).reduce((sum, value) => sum + value, 0) / 100;
  }
  let cumulative = 0;
  return {
    meanRewardSoFar: rewardAt.map((value, pull) => {
      cumulative += value / runs;
      return cumulative / (pull + 1);
    }),
    bestProbability: bestAt.map((value) => value / runs),
    /** Mean over runs of the probability on the best arm after the last pull. */
    finalBestProbability: finalBest / runs,
    /** Runs that end with less than half their probability on the best arm. */
    stuckRuns: stuck,
    runs,
    /** Mean reward over each run's last 100 pulls. */
    lastWindowReward: lastWindow / runs,
  };
}
