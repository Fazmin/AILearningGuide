import { teachingRl, tinyRandom } from "@app/module-sdk";
import {
  CLOSE,
  MAX_TURNS,
  TOOL_COUNT,
  features,
  outcomeOf,
  sampleTransition,
  scoreEpisode,
  startState,
  trainingKinds,
  transitions,
  type Checker,
  type DeskState,
  type Outcome,
  type RewardDesign,
  type TicketSet,
} from "./env";
import {
  WEIGHT_COUNT,
  addKlGradient,
  addLogProbGradient,
  inputOf,
  klDivergence,
  referenceWeights,
  toolProbabilities,
  type Weights,
} from "./policy";

/**
 * Group-relative policy-gradient training (GRPO-style) of the 48-weight policy on the ticket desk.
 *
 * One update:
 *   1. pick the next training ticket kind, round-robin;
 *   2. sample a group of G episodes from the current policy on that ticket;
 *   3. reward_i = score_i − stepCost × turns_i, scored by the chosen reward design;
 *   4. advantage_i = teachingRl.groupAdvantages(rewards, { scale }): reward minus the group mean,
 *      divided by the group standard deviation when scale is "std" and left as it is when "none";
 *   5. w += rate / G × Σ_i [ advantage_i Σ_t ∇log π(a_t | s_t) − β Σ_t ∇KL(π(·|s_t) ‖ π_ref(·|s_t)) ].
 * The KL term uses the exact gradient of the KL at every visited state, not a sampled estimate.
 * A group whose rewards are all equal has zero advantage, so only the KL term can move the policy.
 *
 * Every readout is an exact expectation of the policy at that update, computed by dynamic
 * programming over the desk's states for at most MAX_TURNS turns, so curves carry no Monte Carlo
 * noise. Only the training path itself is sampled, so the seed matters.
 *
 * Left out on purpose: PPO's probability ratio and clipping (one gradient step per group, so the
 * ratio is always 1), many tickets per batch, and any tool arguments or text.
 */

export const UPDATES = 200;
export const LEARNING_RATE = 0.6;
export const GROUP_RANGE = { min: 2, max: 32 } as const;
export const KL_RANGE = { min: 0, max: 0.5, step: 0.05 } as const;
export const COST_RANGE = { min: 0, max: 0.1, step: 0.01 } as const;
export const SEED_RANGE = { min: 1, max: 20 } as const;
export const STRIP_SEEDS = 10;
/** Weights are kept inside this box so no setting can overflow the softmax. */
export const WEIGHT_LIMIT = 40;

export type AdvantageScale = "std" | "none";

export interface TrainOptions {
  reward: RewardDesign;
  checker: Checker;
  tickets: TicketSet;
  group: number;
  /** β, the weight of the KL penalty to the reference policy. */
  kl: number;
  stepCost: number;
  scale: AdvantageScale;
  seed: number;
  updates?: number;
  rate?: number;
}

/** Exact expectations for one value class (tickets that give the value, or leave it out). */
export interface ClassReadout {
  /** E[score − stepCost × turns] under the training reward. */
  reward: number;
  /** P(Strict grader passes). */
  strict: number;
  /** P(the sandbox test passes). */
  sandbox: number;
  /** P(edit_check was called). */
  edited: number;
  /** P(the ticket was closed with the record still wrong). */
  closedNotFixed: number;
  /** P(close_ticket on the first turn). */
  quitFirst: number;
  /** P(the episode hit the turn cap without closing). */
  capped: number;
  /** P(look_up was called). */
  lookedUp: number;
  /** E[turns]. */
  turns: number;
  /** E[Σ over turns of KL(π ‖ π_ref)] in nats. */
  klSum: number;
}

export interface Readout {
  /** Expected training reward over the training tickets, weighted as round-robin samples them. */
  trainingReward: number;
  /** Strict grader success on the two held-out tickets, averaged. */
  trueSuccess: number;
  /** Held-out success on the ticket that gives its value and the one that leaves it out. */
  successGiven: number;
  successMissing: number;
  /** Strict grader success on the training tickets, weighted like training reward. */
  trainingSuccess: number;
  /** Share of held-out episodes that call edit_check. */
  hackRate: number;
  /** Share of held-out episodes closed with the record still wrong. */
  closedNotFixed: number;
  /** Share of held-out episodes on the value-given ticket that call look_up anyway. */
  needlessLookups: number;
  /** Share of held-out episodes that close on the first turn. */
  quitFirst: number;
  /** Mean tool calls per held-out episode. */
  meanTurns: number;
  /** Mean KL to the reference per held-out turn, in nats. */
  klPerTurn: number;
}

export interface TrainPoint extends Readout {
  update: number;
  /** Updates so far whose sampled group had identical rewards. */
  flatSoFar: number;
}

export interface TrainRun {
  points: TrainPoint[];
  /** The policy after each update; weights[0] is the reference. */
  weights: Weights[];
  flatGroups: number;
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/* ---------------------------------------------------------------- exact evaluation */

/**
 * Desk states as small integers: bit 0 given, 1 read, 2 looked, 3 updated, 4 right, 5 edited, and
 * bits 6–7 the last check (0 none, 1 pass, 2 fail). The closed flag is not part of the key,
 * because close_ticket ends the episode.
 */
const KEY_COUNT = 192;
const CHECKS = ["none", "pass", "fail"] as const;
const encode = (state: DeskState) =>
  (state.given ? 1 : 0) |
  (state.read ? 2 : 0) |
  (state.looked ? 4 : 0) |
  (state.updated ? 8 : 0) |
  (state.right ? 16 : 0) |
  (state.edited ? 32 : 0) |
  (CHECKS.indexOf(state.check) << 6);
const decode = (key: number): DeskState => ({
  given: (key & 1) !== 0,
  read: (key & 2) !== 0,
  looked: (key & 4) !== 0,
  updated: (key & 8) !== 0,
  right: (key & 16) !== 0,
  edited: (key & 32) !== 0,
  check: CHECKS[(key >> 6) & 3] ?? "none",
});
const featureKey = (values: ReadonlyArray<number>) => values.reduce((key, value, index) => key | (value << index), 0);

interface Edge {
  probability: number;
  /** The next key, or −1 when the call closed the ticket (the state is unchanged). */
  next: number;
}

/** Every transition of the desk, built once from env.ts so the evaluator cannot drift from it. */
const TABLE: { features: number[]; feature: number; edges: Edge[][] }[] = Array.from({ length: KEY_COUNT }, (_, key) => {
  const state = decode(key);
  const values = features(state);
  return {
    features: values,
    feature: featureKey(values),
    edges: Array.from({ length: TOOL_COUNT }, (_, tool) =>
      transitions(state, tool).map((transition) => ({
        probability: transition.probability,
        next: transition.closed ? -1 : encode(transition.next),
      })),
    ),
  };
});

/** π and KL(π ‖ π_ref) per feature vector, filled lazily for one policy. */
function policyLookup(weights: Weights, reference: Weights) {
  const cache: Array<{ policy: number[]; kl: number } | undefined> = new Array(128);
  return (key: number) => {
    const row = TABLE[key];
    let entry = cache[row.feature];
    if (!entry) {
      const policy = toolProbabilities(weights, row.features);
      entry = { policy, kl: klDivergence(policy, toolProbabilities(reference, row.features)) };
      cache[row.feature] = entry;
    }
    return entry;
  };
}

type Lookup = ReturnType<typeof policyLookup>;
interface Design {
  reward: RewardDesign;
  checker: Checker;
  stepCost: number;
}

function classReadout(lookup: Lookup, given: boolean, design: Design): ClassReadout {
  const result: ClassReadout = {
    reward: 0,
    strict: 0,
    sandbox: 0,
    edited: 0,
    closedNotFixed: 0,
    quitFirst: 0,
    capped: 0,
    lookedUp: 0,
    turns: 0,
    klSum: 0,
  };
  const finish = (key: number, closed: boolean, turns: number, probability: number) => {
    const outcome = outcomeOf(decode(key), closed, turns);
    result.reward += probability * (scoreEpisode(design.reward, design.checker, outcome) - design.stepCost * turns);
    result.strict += probability * scoreEpisode("outcome", "strict", outcome);
    result.sandbox += probability * scoreEpisode("outcome", "sandbox", outcome);
    if (outcome.edited) result.edited += probability;
    if (closed && !outcome.right) result.closedNotFixed += probability;
    if (closed && turns === 1) result.quitFirst += probability;
    if (!closed) result.capped += probability;
    if (outcome.looked) result.lookedUp += probability;
    result.turns += probability * turns;
  };

  let current = new Float64Array(KEY_COUNT);
  current[encode(startState(given))] = 1;
  for (let turn = 1; turn <= MAX_TURNS; turn += 1) {
    const next = new Float64Array(KEY_COUNT);
    for (let key = 0; key < KEY_COUNT; key += 1) {
      const probability = current[key];
      if (probability === 0) continue;
      const { policy, kl } = lookup(key);
      result.klSum += probability * kl;
      const edges = TABLE[key].edges;
      for (let tool = 0; tool < TOOL_COUNT; tool += 1) {
        const mass = probability * policy[tool];
        if (mass === 0) continue;
        for (const edge of edges[tool]) {
          if (edge.next < 0) finish(key, true, turn, mass * edge.probability);
          else next[edge.next] += mass * edge.probability;
        }
      }
    }
    current = next;
  }
  for (let key = 0; key < KEY_COUNT; key += 1) if (current[key] > 0) finish(key, false, MAX_TURNS, current[key]);
  return result;
}

/**
 * Exact expectations for one value class by dynamic programming: the probability of every desk
 * state is pushed forward one turn at a time, for at most MAX_TURNS turns.
 */
export function evaluateClass(
  weights: Weights,
  given: boolean,
  design: Design,
  reference: Weights = referenceWeights(),
): ClassReadout {
  return classReadout(policyLookup(weights, reference), given, design);
}

/** Both value classes, and the readouts the lab shows, for one policy. */
export function evaluatePolicy(
  weights: Weights,
  options: Pick<TrainOptions, "reward" | "checker" | "stepCost" | "tickets">,
  reference: Weights = referenceWeights(),
): Readout {
  const design = { reward: options.reward, checker: options.checker, stepCost: options.stepCost };
  const lookup = policyLookup(weights, reference);
  const given = classReadout(lookup, true, design);
  const missing = classReadout(lookup, false, design);
  const kinds = trainingKinds(options.tickets);
  const givenShare = kinds.filter((kind) => kind.given).length / kinds.length;
  const mix = (a: number, b: number) => givenShare * a + (1 - givenShare) * b;
  const mean = (a: number, b: number) => (a + b) / 2;
  return {
    trainingReward: mix(given.reward, missing.reward),
    trueSuccess: mean(given.strict, missing.strict),
    successGiven: given.strict,
    successMissing: missing.strict,
    trainingSuccess: mix(given.strict, missing.strict),
    hackRate: mean(given.edited, missing.edited),
    closedNotFixed: mean(given.closedNotFixed, missing.closedNotFixed),
    needlessLookups: given.lookedUp,
    quitFirst: mean(given.quitFirst, missing.quitFirst),
    meanTurns: mean(given.turns, missing.turns),
    klPerTurn: (given.klSum + missing.klSum) / Math.max(1e-12, given.turns + missing.turns),
  };
}

/* ---------------------------------------------------------------- sampling */

export interface EpisodeStep {
  turn: number;
  /** The features the policy saw before this call. */
  features: number[];
  /** π over the six tools at this state. */
  probabilities: number[];
  tool: number;
  /** For update_record: whether the write saved. */
  saved?: boolean;
  /** The desk state before and after the call. */
  before: DeskState;
  after: DeskState;
}

export interface Episode {
  steps: EpisodeStep[];
  outcome: Outcome;
}

/** Draws an index from a distribution, never one with zero probability. */
function drawIndex(probabilities: ReadonlyArray<number>, draw: number) {
  let remaining = draw;
  let last = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    if (probabilities[index] <= 0) continue;
    last = index;
    remaining -= probabilities[index];
    if (remaining < 0) return index;
  }
  return last;
}

/** One episode from the policy, on a ticket that gives its value or leaves it out. */
export function sampleEpisode(weights: Weights, given: boolean, random: () => number): Episode {
  let state = startState(given);
  const steps: EpisodeStep[] = [];
  for (let turn = 1; turn <= MAX_TURNS; turn += 1) {
    const values = features(state);
    const probabilities = toolProbabilities(weights, values);
    const tool = drawIndex(probabilities, random());
    const transition = sampleTransition(state, tool, random());
    steps.push({ turn, features: values, probabilities, tool, saved: transition.saved, before: state, after: transition.next });
    state = transition.next;
    if (tool === CLOSE) return { steps, outcome: outcomeOf(state, true, turn) };
  }
  return { steps, outcome: outcomeOf(state, false, MAX_TURNS) };
}

/* ---------------------------------------------------------------- training */

const finiteOr = (value: number | undefined, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

export const sanitizeOptions = (options: TrainOptions): Required<TrainOptions> => ({
  ...options,
  group: Math.round(clamp(finiteOr(options.group, 8), GROUP_RANGE.min, GROUP_RANGE.max)),
  kl: clamp(finiteOr(options.kl, 0), KL_RANGE.min, KL_RANGE.max),
  stepCost: clamp(finiteOr(options.stepCost, 0), COST_RANGE.min, COST_RANGE.max),
  seed: Math.round(clamp(finiteOr(options.seed, 1), SEED_RANGE.min, SEED_RANGE.max)),
  updates: Math.round(clamp(finiteOr(options.updates, UPDATES), 0, UPDATES)),
  rate: clamp(finiteOr(options.rate, LEARNING_RATE), 0, 2),
});

export interface UpdateSettings {
  scale: AdvantageScale;
  /** β, the weight of the KL penalty. */
  beta: number;
  rate: number;
  /** π_ref at a feature vector; train() caches it. */
  referenceAt: (features: ReadonlyArray<number>) => number[];
}

/** Equal rewards carry no signal. */
export const isFlat = (rewards: ReadonlyArray<number>) => Math.max(...rewards) - Math.min(...rewards) < 1e-12;

/**
 * One group-relative update, in place. Returns whether the group was flat. A flat group gets
 * exactly zero advantage: a floating-point mean can leave a 1e-17 residue that the std division
 * would blow up to 1e-11, so it is not left to the division.
 */
export function applyGroupUpdate(
  weights: Weights,
  episodes: ReadonlyArray<Episode>,
  rewards: ReadonlyArray<number>,
  settings: UpdateSettings,
): boolean {
  const flat = isFlat(rewards);
  const advantages = flat ? rewards.map(() => 0) : teachingRl.groupAdvantages(rewards, { scale: settings.scale });
  const gradient = new Float64Array(WEIGHT_COUNT);
  episodes.forEach((episode, member) => {
    for (const step of episode.steps) {
      const input = inputOf(step.features);
      if (advantages[member] !== 0) addLogProbGradient(gradient, step.probabilities, input, step.tool, advantages[member]);
      if (settings.beta > 0) addKlGradient(gradient, step.probabilities, settings.referenceAt(step.features), input, -settings.beta);
    }
  });
  const step = settings.rate / Math.max(1, episodes.length);
  for (let index = 0; index < WEIGHT_COUNT; index += 1) {
    weights[index] = clamp(weights[index] + step * gradient[index], -WEIGHT_LIMIT, WEIGHT_LIMIT);
  }
  return flat;
}

/**
 * Runs the whole training loop. Deterministic in the options: the same seed gives the same run.
 * With `detail` false only the final policy is evaluated, which is what the 10-seed strip needs.
 */
export function train(input: TrainOptions, detail = true): TrainRun {
  const options = sanitizeOptions(input);
  const { group, kl: beta, stepCost, scale, rate, updates } = options;
  const kinds = trainingKinds(options.tickets);
  const random = tinyRandom(options.seed);
  const reference = referenceWeights();
  const referenceCache = new Map<number, number[]>();
  const referenceAt = (values: ReadonlyArray<number>) => {
    const key = featureKey(values);
    let probabilities = referenceCache.get(key);
    if (!probabilities) {
      probabilities = toolProbabilities(reference, values);
      referenceCache.set(key, probabilities);
    }
    return probabilities;
  };
  const settings: UpdateSettings = { scale, beta, rate, referenceAt };
  const weights = referenceWeights();
  const history: Weights[] = [Float64Array.from(weights)];
  const points: TrainPoint[] = [];
  let flatGroups = 0;
  const record = (update: number) =>
    points.push({ update, flatSoFar: flatGroups, ...evaluatePolicy(weights, options, reference) });
  if (detail) record(0);

  for (let update = 1; update <= updates; update += 1) {
    const kind = kinds[(update - 1) % kinds.length];
    const episodes: Episode[] = [];
    const rewards: number[] = [];
    for (let member = 0; member < group; member += 1) {
      const episode = sampleEpisode(weights, kind.given, random);
      episodes.push(episode);
      rewards.push(scoreEpisode(options.reward, options.checker, episode.outcome) - stepCost * episode.outcome.turns);
    }
    if (applyGroupUpdate(weights, episodes, rewards, settings)) flatGroups += 1;
    if (detail) {
      history.push(Float64Array.from(weights));
      record(update);
    }
  }
  if (!detail) {
    history.push(Float64Array.from(weights));
    record(updates);
  }
  return { points, weights: history, flatGroups };
}

/* ---------------------------------------------------------------- ten seeds */

export interface SeedResult {
  seed: number;
  final: TrainPoint;
}

export interface SeedSummary {
  seeds: SeedResult[];
  meanSuccess: number;
  meanHack: number;
  meanSuccessMissing: number;
  /** Seeds whose final policy edits the test in more than half of held-out episodes. */
  hackedSeeds: number;
  /** Seeds whose final policy closes on the first turn in more than half of held-out episodes. */
  quitSeeds: number;
}

const stripCache = new Map<string, SeedSummary>();
const STRIP_CACHE_LIMIT = 48;

/**
 * The final readouts of seeds 1 to 10 with every other control as given. Memoised on the
 * controls, so moving Seed or Training update does not retrain ten runs.
 */
export function acrossSeeds(input: Omit<TrainOptions, "seed">): SeedSummary {
  const options = sanitizeOptions({ ...input, seed: 1 });
  const key = JSON.stringify([
    options.reward,
    options.checker,
    options.tickets,
    options.group,
    options.kl,
    options.stepCost,
    options.scale,
    options.updates,
    options.rate,
  ]);
  const cached = stripCache.get(key);
  if (cached) return cached;
  const seeds: SeedResult[] = [];
  for (let seed = 1; seed <= STRIP_SEEDS; seed += 1) {
    const run = train({ ...options, seed }, false);
    seeds.push({ seed, final: run.points[run.points.length - 1] });
  }
  const average = (pick: (point: TrainPoint) => number) =>
    seeds.reduce((sum, entry) => sum + pick(entry.final), 0) / seeds.length;
  const summary: SeedSummary = {
    seeds,
    meanSuccess: average((point) => point.trueSuccess),
    meanHack: average((point) => point.hackRate),
    meanSuccessMissing: average((point) => point.successMissing),
    hackedSeeds: seeds.filter((entry) => entry.final.hackRate > 0.5).length,
    quitSeeds: seeds.filter((entry) => entry.final.quitFirst > 0.5).length,
  };
  if (stripCache.size >= STRIP_CACHE_LIMIT) {
    const oldest = stripCache.keys().next().value;
    if (oldest !== undefined) stripCache.delete(oldest);
  }
  stripCache.set(key, summary);
  return summary;
}
