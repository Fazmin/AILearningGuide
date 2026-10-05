import {
  GGUF_FORMATS,
  TINY_CORPORA,
  pruneTinyWeights,
  quantizeTinyWeights,
  teachingEval,
  teachingServing,
  tinyCrossEntropy,
  tinyPerplexity,
  trainTinyDpo,
  trainTinyLora,
  trainTinyModel,
  type TinyPreferencePair,
} from "@app/module-sdk";

/**
 * The capstone pipeline: pretrain, adapt, align, compress, gate, serve.
 *
 * Every stage is a function of the recipe alone, so the lab stores controls and recomputes
 * the whole run from them. Nothing here is measured on a real model: the weights are the
 * 900-weight character bigram table from tiny-lm, and the serving budget is first-order
 * arithmetic for an 8B-class model on a stated 24 GiB accelerator.
 */

export interface ShipRecipe {
  baseEpochs: number;
  loraRank: number;
  loraEpochs: number;
  /** Harbor sentences mixed back into the adaptation data, from none to all six. */
  replay: number;
  dpoSteps: number;
  beta: number;
  format: string;
  prune: number;
  /** Smallest acceptable fall in recipe-text loss, in nats per character, against the base model. */
  gateTarget: number;
  /** Largest tolerated rise in harbor-text loss, in nats per character, against the base model. */
  gateControl: number;
  users: number;
  context: number;
}

export const DEFAULT_RECIPE: ShipRecipe = {
  baseEpochs: 48,
  loraRank: 4,
  loraEpochs: 24,
  replay: 0,
  dpoSteps: 40,
  beta: 0.4,
  format: "Q4_K_M",
  prune: 0,
  gateTarget: 0.03,
  gateControl: 0.1,
  users: 8,
  context: 4096,
};

/** Slider bounds, shared by the lab and by state validation. */
export const RECIPE_BOUNDS = {
  baseEpochs: [8, 60],
  loraRank: [1, 12],
  loraEpochs: [0, 40],
  replay: [0, 6],
  dpoSteps: [0, 80],
  beta: [0.1, 1.5],
  prune: [0, 0.9],
  gateTarget: [0, 0.2],
  gateControl: [0, 0.4],
  users: [1, 64],
  context: [512, 32768],
} as const satisfies Record<string, readonly [number, number]>;

/** Formats that have both a toy block layout and a published whole-file size to scale from. */
export const FORMAT_NAMES: string[] = teachingServing.SERVE_FORMATS.map((format) => format.name);

const sentences = (text: string) =>
  text
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);

/** The last quarter of a corpus is never trained on; it is the held-out text the losses read. */
function split(text: string) {
  const parts = sentences(text);
  const held = Math.max(1, Math.round(parts.length * 0.25));
  return {
    train: `${parts.slice(0, parts.length - held).join(". ")}.`,
    held: `${parts.slice(parts.length - held).join(". ")}.`,
  };
}

const harbor = split(TINY_CORPORA.harbor.text);
const recipes = split(TINY_CORPORA.recipes.text);

export const HELD_OUT = { harbor: harbor.held, recipes: recipes.held };
export const TRAIN_TEXT = { harbor: harbor.train, recipes: recipes.train };

/**
 * Benchmark items come only from text the model never trained on: the held-out last quarter of
 * each corpus, and a few fresh sentences written for this lab. A distractor is wrong but ordinary,
 * and the bigram reads each candidate only through its own characters and the space before it.
 */
const TARGET_ITEMS = `
the pasta is drained and tossed with the warm | sauce | trails
finish the dish with fresh | basil | breeze
finish the dish with fresh basil and a little | cheese | gale
slice the garlic and cook it until it turns | golden | stormy
toss the noodles in the pan with a little | butter | harbor
season the soup with salt and a pinch of | pepper | rain
`;

const CONTROL_ITEMS = `
in summer the harbor fills with | sails | garlic
in summer the harbor fills with sails and the water stays | warm | pasta
the fog returns in the | morning | basil
the fog returns in the morning and the pattern | repeats | simmers
a cold wind crosses the harbor at | night | sauce
the tide pulls the boats out to | sea | cheese
`;

export const TARGET_BENCHMARK = teachingEval.parseItems(TARGET_ITEMS);
export const CONTROL_BENCHMARK = teachingEval.parseItems(CONTROL_ITEMS);

export const PREFERENCE_PAIRS: TinyPreferencePair[] = [
  { prompt: "warm the", chosen: " pan and add a spoon of oil.", rejected: " pan and the pan and the pan." },
  { prompt: "simmer the", chosen: " sauce until it thickens.", rejected: " sauce and simmer the sauce." },
  { prompt: "stir the", chosen: " onions until they turn soft.", rejected: " onions onions onions onions." },
];

export interface PassRate {
  correct: number;
  total: number;
  low: number;
  high: number;
}

export function passRate(weights: Float32Array, items: teachingEval.BenchmarkItem[]): PassRate {
  const correct = items.filter(
    (item) =>
      teachingEval.scoreContinuation(weights, item.context, item.correct).mean >
      teachingEval.scoreContinuation(weights, item.context, item.distractor).mean,
  ).length;
  const [low, high] = teachingEval.wilsonInterval(correct, items.length);
  return { correct, total: items.length, low, high };
}

export interface StageMetrics {
  /** Held-out recipe text, nats per character. */
  targetLoss: number;
  /** Held-out harbor text, nats per character. */
  controlLoss: number;
  targetPerplexity: number;
  target: PassRate;
  control: PassRate;
}

export function measure(weights: Float32Array): StageMetrics {
  return {
    targetLoss: tinyCrossEntropy(weights, recipes.held),
    controlLoss: tinyCrossEntropy(weights, harbor.held),
    targetPerplexity: tinyPerplexity(weights, recipes.held),
    target: passRate(weights, TARGET_BENCHMARK),
    control: passRate(weights, CONTROL_BENCHMARK),
  };
}

export type StageId = "base" | "adapted" | "aligned" | "shipped";

export const STAGE_LABELS: Record<StageId, string> = {
  base: "Base model",
  adapted: "After adapting",
  aligned: "After aligning",
  shipped: "As shipped",
};

export interface Stage {
  id: StageId;
  weights: Float32Array;
  metrics: StageMetrics;
}

export interface GateCheck {
  id: "target" | "control";
  passed: boolean;
  /** What the gate compared, in words the report can print. */
  detail: string;
}

export interface Gate {
  passed: boolean;
  checks: GateCheck[];
  /** The benchmark counts for the same two comparisons, with how far their intervals overlap. */
  advisory: string;
}

const nats = (value: number) => value.toFixed(3);
const percent = (value: number) => `${Math.round(value * 100)}%`;
const overlaps = (a: PassRate, b: PassRate) => a.low <= b.high && b.low <= a.high;

/** The gate reads the artifact you would ship, not the one you trained. */
export function evaluateGate(base: StageMetrics, shipped: StageMetrics, recipe: ShipRecipe): Gate {
  const gain = base.targetLoss - shipped.targetLoss;
  const rise = shipped.controlLoss - base.controlLoss;
  const checks: GateCheck[] = [
    {
      id: "target",
      passed: gain >= recipe.gateTarget,
      detail: `Recipe text: ${nats(shipped.targetLoss)} nats per character, ${gain >= 0 ? "down" : "up"} ${nats(
        Math.abs(gain),
      )} from the base model's ${nats(base.targetLoss)}. The bar is a fall of at least ${nats(recipe.gateTarget)}.`,
    },
    {
      id: "control",
      passed: rise <= recipe.gateControl,
      detail: `Harbor text: ${nats(shipped.controlLoss)} nats per character, ${rise >= 0 ? "up" : "down"} ${nats(
        Math.abs(rise),
      )} from the base model's ${nats(base.controlLoss)}. The most it may rise is ${nats(recipe.gateControl)}.`,
    },
  ];
  const targetDecisive = !overlaps(base.target, shipped.target);
  const controlDecisive = !overlaps(base.control, shipped.control);
  const advisory =
    `Benchmark, ${shipped.target.total} items each: recipe ${base.target.correct} → ${shipped.target.correct} correct ` +
    `(95% intervals ${percent(base.target.low)}–${percent(base.target.high)} and ${percent(shipped.target.low)}–${percent(shipped.target.high)}, ` +
    `${targetDecisive ? "separate" : "overlapping, so not decisive"}); harbor ${base.control.correct} → ${shipped.control.correct} ` +
    `(${controlDecisive ? "separate" : "overlapping, so not decisive"}). Six items cannot settle a close call, which is why the gate reads held-out loss.`;
  return { passed: checks.every((check) => check.passed), checks, advisory };
}

export interface Shipping {
  format: string;
  /** Toy bytes for the 900-weight table at the chosen format's block layout. */
  toyBytes: number;
  toyBitsPerWeight: number;
  /** The same recipe imagined on an 8B model, on the stated 24 GiB accelerator. */
  serving: teachingServing.ServingPoint;
}

export interface Alignment {
  /** Share of preference pairs whose reference-relative reward margin is positive after DPO. */
  accuracy: number;
  margin: number;
}

export interface Release {
  stages: Stage[];
  alignment: Alignment | null;
  gate: Gate;
  shipping: Shipping;
  /** Stage-to-stage losses worse than 0.02 nats, in the order they happened. */
  regressions: string[];
  verdict: "ship" | "hold";
  /** Why the verdict is what it is. */
  reasons: string[];
}

export function runPipeline(recipe: ShipRecipe): Release {
  const base = trainTinyModel({ text: harbor.train, epochs: recipe.baseEpochs, seed: 1 }).weights;

  const replayText = sentences(harbor.train)
    .slice(0, recipe.replay)
    .map((part) => `${part}.`)
    .join(" ");
  const adapted = trainTinyLora({
    base,
    text: replayText ? `${recipes.train} ${replayText}` : recipes.train,
    rank: recipe.loraRank,
    epochs: recipe.loraEpochs,
    seed: 5,
  }).merged;

  const dpo = recipe.dpoSteps
    ? trainTinyDpo({ reference: adapted, pairs: PREFERENCE_PAIRS, beta: recipe.beta, steps: recipe.dpoSteps })
    : null;
  const aligned = dpo ? dpo.weights : adapted;

  const format = GGUF_FORMATS.find((entry) => entry.name === recipe.format) ?? GGUF_FORMATS[3];
  const quantized = quantizeTinyWeights(aligned, { bits: format.bits, scope: format.scope, groupSize: format.groupSize });
  const shippedWeights =
    recipe.prune > 0
      ? pruneTinyWeights(quantized.weights, { fraction: recipe.prune, mode: "unstructured" }).weights
      : quantized.weights;

  const stages: Stage[] = [
    { id: "base", weights: base, metrics: measure(base) },
    { id: "adapted", weights: adapted, metrics: measure(adapted) },
    { id: "aligned", weights: aligned, metrics: measure(aligned) },
    { id: "shipped", weights: shippedWeights, metrics: measure(shippedWeights) },
  ];

  const gate = evaluateGate(stages[0].metrics, stages[3].metrics, recipe);

  const served = teachingServing.SERVE_FORMATS.find((entry) => entry.name === recipe.format) ?? teachingServing.SERVE_FORMATS[3];
  const serving = teachingServing.servingPoint(
    teachingServing.MODELS[2],
    served,
    recipe.users,
    recipe.context,
    true,
    teachingServing.HARDWARE,
  );
  const shipping: Shipping = {
    format: format.name,
    toyBytes: quantized.totalBytes,
    toyBitsPerWeight: quantized.bytesPerWeight * 8,
    serving,
  };

  const regressions: string[] = [];
  for (let index = 1; index < stages.length; index += 1) {
    const before = stages[index - 1].metrics;
    const after = stages[index].metrics;
    const label = STAGE_LABELS[stages[index].id].toLowerCase();
    if (after.targetLoss - before.targetLoss > 0.02) {
      regressions.push(`${label}: recipe loss rose ${nats(after.targetLoss - before.targetLoss)} nats`);
    }
    if (after.controlLoss - before.controlLoss > 0.02) {
      regressions.push(`${label}: harbor loss rose ${nats(after.controlLoss - before.controlLoss)} nats`);
    }
  }

  const reasons = gate.checks.filter((check) => !check.passed).map((check) => check.detail);
  if (!serving.fits) {
    reasons.push(
      `At ${recipe.users} users and ${recipe.context.toLocaleString("en-US")} tokens the 8B-class budget is ${(
        serving.totalBytes /
        teachingServing.GIB
      ).toFixed(1)} GiB, past the ${(teachingServing.HARDWARE.vramBytes / teachingServing.GIB).toFixed(0)} GiB accelerator.`,
    );
  }

  return {
    stages,
    alignment: dpo ? { accuracy: dpo.accuracy, margin: dpo.finalMargin } : null,
    gate,
    shipping,
    regressions,
    verdict: reasons.length === 0 ? "ship" : "hold",
    reasons,
  };
}
