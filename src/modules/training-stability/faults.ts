/**
 * The five runs the lab can show, as options for `trainTinyExtended`. Every run trains the same
 * rank-8 factored model as Train one from scratch, on the same six Harbor sentences, with the same
 * batch size, epoch count and seed, so the only thing that changes between faults is the fault and
 * the fix the learner applied. Each fault also names a clean reference run, so the dashboard can
 * shade the normal range of every signal from a run that had nothing wrong with it.
 */
import {
  bigramPairs,
  TINY_CORPORA,
  type TinyExtendedOptions,
  type TinyFault,
  type TinyOptimizerKind,
} from "@app/module-sdk";
import { formatCoefficient, powerOfTwo } from "./dashboard";
import { rollbackPlan } from "./rollback";

export const FAULT_IDS = ["none", "rate", "fp16", "bad-batch", "logits"] as const;
export type FaultId = (typeof FAULT_IDS)[number];

export const SCHEDULE_IDS = ["constant", "warmup-cosine"] as const;
export type ScheduleId = (typeof SCHEDULE_IDS)[number];

export const SCALING_IDS = ["static", "dynamic"] as const;
export type ScalingId = (typeof SCALING_IDS)[number];

export const FAULTS: ReadonlyArray<{ id: FaultId; label: string; detail: string }> = [
  { id: "none", label: "None", detail: "Plain SGD at learning rate 0.5, the recipe Train one from scratch uses." },
  { id: "rate", label: "Learning rate too high", detail: "The same SGD run at learning rate 8, sixteen times the healthy rate." },
  {
    id: "fp16",
    label: "fp16 overflow",
    detail: "The healthy SGD run, with every gradient multiplied by the loss scale and rounded to half precision.",
  },
  {
    id: "bad-batch",
    label: "Garbage batch",
    detail: "Adam at 0.1, with five batches of junk text in place of the real batches at steps 400 to 404.",
  },
  { id: "logits", label: "Logit growth", detail: "Adam at learning rate 1, ten times the rate the garbage-batch run uses." },
];

/** The model and data every run shares. */
export const RANK = 8;
export const EPOCHS = 20;
export const BATCH_SIZE = 16;
export const SEED = 9;
/** A whole-corpus loss every 4 steps: 121 checkpoints over 480 steps, so step 404 is one of them. */
export const CHECKPOINTS = 121;

/** Plain SGD at 0.5 is the healthy run; 8 is the rate that halts it in Train one from scratch. */
export const SGD_RATE = 0.5;
export const HIGH_RATE = 8;
/** Adam with no weight decay, the optimizer of the garbage-batch run and the clean reference for logit growth. */
export const ADAM_RATE = 0.1;
export const ADAM_RATES = [0.01, 0.03, 0.1, 0.3, 1] as const;
export const DEFAULT_ADAM_RATE = 1;
export const Z_LOSSES = [0, 1e-4, 1e-3, 1e-2] as const;

export const CLIP_MAX = 1.5;
export const CLIP_STEP = 0.05;
/** Loss scale exponents: 2^0 to 2^24. */
export const SCALE_MIN = 0;
export const SCALE_MAX = 24;
export const DEFAULT_SCALE = 20;
/** Clean steps before dynamic scaling doubles the scale again. PyTorch's GradScaler waits 2,000. */
export const GROWTH_INTERVAL = 200;

/** Rare letters, repeated: the kind of junk a broken scrape produces. */
export const GARBAGE_TEXT = "qqqq zzzz xxxx jjjj";
/** 1-based: the first of the five garbage batches is the 400th optimizer step. */
export const GARBAGE_FIRST_STEP = 400;
export const GARBAGE_COUNT = 5;

/**
 * The six training sentences of Harbor weather notes: the corpus with its last quarter of sentences
 * held out, exactly the split Train one from scratch trains on (379 characters, 378 pairs).
 */
export function harborTrainingText(): string {
  const sentences = TINY_CORPORA.harbor.text
    .split(".")
    .map((part) => part.trim())
    .filter(Boolean);
  const held = Math.max(1, Math.round(sentences.length * 0.25));
  return `${sentences.slice(0, sentences.length - held).join(". ")}.`;
}

export const TRAINING_TEXT = harborTrainingText();
/** 378 training pairs, so 24 batches of 16 per epoch (the last holds 10) and 480 steps in 20 epochs. */
export const PAIRS = bigramPairs(TRAINING_TEXT).inputs.length;
export const PLANNED_STEPS = Math.ceil(PAIRS / BATCH_SIZE) * EPOCHS;

export interface FaultSettings {
  fault: FaultId;
  /** 0 turns the clip off. Applies to the Learning rate too high and Garbage batch runs. */
  clipNorm: number;
  schedule: ScheduleId;
  /** The loss scale is 2 to this power. */
  lossScale: number;
  scaling: ScalingId;
  /** Garbage batch only: restart from the last clean checkpoint and skip the garbage. */
  rollback: boolean;
  zLoss: number;
  /** Logit growth only: Adam's learning rate. */
  adamRate: number;
}

export type Optimizer = "SGD" | "Adam";

export interface Recipe {
  optimizer: Optimizer;
  learningRate: number;
  scheduleLabel: string;
  /** Extra settings worth printing, already worded. */
  extras: string[];
}

const shared = {
  text: TRAINING_TEXT,
  model: { kind: "factored" as const, rank: RANK },
  epochs: EPOCHS,
  batchSize: BATCH_SIZE,
  seed: SEED,
  trace: true,
  checkpoints: CHECKPOINTS,
};

const optimizerOf = (kind: TinyOptimizerKind) => ({ optimizer: { kind } });

/** The healthy run: what Fault: None trains, and the reference for the SGD faults. */
export const healthyOptions = (): TinyExtendedOptions => ({ ...shared, learningRate: SGD_RATE, ...optimizerOf("sgd") });

/** Adam at 0.1 with nothing wrong, and the clip the learner chose. */
export const adamCleanOptions = (clipNorm = 0): TinyExtendedOptions => ({
  ...shared,
  learningRate: ADAM_RATE,
  clipNorm,
  ...optimizerOf("adam"),
});

export const garbageFault = (): TinyFault => ({
  kind: "bad-batch",
  atStep: GARBAGE_FIRST_STEP - 1,
  count: GARBAGE_COUNT,
  text: GARBAGE_TEXT,
});

export const lossScaleFault = (exponent: number, scaling: ScalingId): TinyFault => ({
  kind: "loss-scale",
  scale: 2 ** exponent,
  dynamic: scaling === "dynamic",
  growthInterval: GROWTH_INTERVAL,
});

/**
 * The run the dashboard draws. For Garbage batch with Roll back and skip pressed, that is the
 * replay: the same options with the garbage steps skipped, which is identical to the original run
 * up to the first garbage step.
 */
export function runOptions(settings: FaultSettings): TinyExtendedOptions {
  switch (settings.fault) {
    case "none":
      return healthyOptions();
    case "rate":
      return {
        ...shared,
        learningRate: HIGH_RATE,
        clipNorm: settings.clipNorm,
        // The SDK's one-cycle curve: a tenth of the rate, up to the full rate over the first 15
        // percent of steps, then half a cosine down to 2 percent.
        schedule: settings.schedule === "warmup-cosine" ? "one-cycle" : "constant",
        warmupFraction: 0.15,
        ...optimizerOf("sgd"),
      };
    case "fp16":
      return { ...healthyOptions(), faults: [lossScaleFault(settings.lossScale, settings.scaling)] };
    case "bad-batch":
      return {
        ...adamCleanOptions(settings.clipNorm),
        faults: [garbageFault()],
        skipSteps: settings.rollback ? rollbackPlan(GARBAGE_FIRST_STEP, GARBAGE_COUNT).skipSteps : [],
      };
    case "logits":
      return { ...shared, learningRate: settings.adamRate, zLoss: settings.zLoss, ...optimizerOf("adam") };
  }
}

/** The faulty original, drawn faintly behind the replay once Roll back and skip is pressed. */
export function originalOptions(settings: FaultSettings): TinyExtendedOptions | null {
  if (settings.fault !== "bad-batch" || !settings.rollback) return null;
  return runOptions({ ...settings, rollback: false });
}

/** The clean run whose percentiles shade each panel's normal range. Null when the run is its own reference. */
export function referenceOptions(settings: FaultSettings): TinyExtendedOptions | null {
  switch (settings.fault) {
    case "none":
      return null;
    case "rate":
    case "fp16":
      return healthyOptions();
    case "bad-batch":
      return adamCleanOptions(settings.clipNorm);
    case "logits":
      return adamCleanOptions(0);
  }
}

export function referenceLabel(settings: FaultSettings): string {
  switch (settings.fault) {
    case "none":
      return "this run";
    case "rate":
    case "fp16":
      return "the healthy run (SGD at 0.5)";
    case "bad-batch":
      return "the same run with no garbage";
    case "logits":
      return "Adam at 0.1 with no z-loss";
  }
}

export function recipeOf(settings: FaultSettings): Recipe {
  switch (settings.fault) {
    case "none":
      return { optimizer: "SGD", learningRate: SGD_RATE, scheduleLabel: "constant", extras: [] };
    case "rate":
      return {
        optimizer: "SGD",
        learningRate: HIGH_RATE,
        scheduleLabel: settings.schedule === "warmup-cosine" ? "warmup + cosine" : "constant",
        extras: [settings.clipNorm > 0 ? `clip ${settings.clipNorm.toFixed(2)}` : "no clip"],
      };
    case "fp16":
      return {
        optimizer: "SGD",
        learningRate: SGD_RATE,
        scheduleLabel: "constant",
        extras: [
          settings.scaling === "dynamic"
            ? `fp16 gradients, dynamic loss scale from ${powerOfTwo(settings.lossScale)}`
            : `fp16 gradients, static loss scale ${powerOfTwo(settings.lossScale)}`,
        ],
      };
    case "bad-batch":
      return {
        optimizer: "Adam",
        learningRate: ADAM_RATE,
        scheduleLabel: "constant",
        extras: [
          settings.clipNorm > 0 ? `clip ${settings.clipNorm.toFixed(2)}` : "no clip",
          `garbage at steps ${GARBAGE_FIRST_STEP} to ${GARBAGE_FIRST_STEP + GARBAGE_COUNT - 1}`,
          ...(settings.rollback ? ["rolled back and skipped"] : []),
        ],
      };
    case "logits":
      return {
        optimizer: "Adam",
        learningRate: settings.adamRate,
        scheduleLabel: "constant",
        extras: [settings.zLoss > 0 ? `z-loss ${formatCoefficient(settings.zLoss)}` : "no z-loss"],
      };
  }
}

export const describeRecipe = (recipe: Recipe) =>
  [`${recipe.optimizer} at ${recipe.learningRate}`, recipe.scheduleLabel, ...recipe.extras].join(" · ");

/* -------------------------------------------------------------------------- */
/* Number formats. Closed-form facts, not measurements.                        */
/* -------------------------------------------------------------------------- */

/** Largest finite half-precision value; anything that rounds above it becomes Infinity. */
export const HALF_MAX = 65504;
/** A magnitude from here up rounds past HALF_MAX in the SDK's toHalf, so it overflows. */
export const HALF_OVERFLOW = 65520;
/** Smallest positive normal half-precision value, 2^-14. */
export const HALF_MIN_NORMAL = 2 ** -14;
/** Smallest positive subnormal half-precision value, 2^-24. Half of it and below rounds to zero. */
export const HALF_MIN_SUBNORMAL = 2 ** -24;

export interface NumberFormat {
  id: "fp32" | "bf16" | "fp16";
  label: string;
  bits: number;
  exponentBits: number;
  /** Stored significand bits; one more is implied. */
  fractionBits: number;
  largest: number;
  smallestNormal: number;
  smallestSubnormal: number;
}

export const NUMBER_FORMATS: ReadonlyArray<NumberFormat> = [
  {
    id: "fp32",
    label: "fp32",
    bits: 32,
    exponentBits: 8,
    fractionBits: 23,
    largest: (2 - 2 ** -23) * 2 ** 127,
    smallestNormal: 2 ** -126,
    smallestSubnormal: 2 ** -149,
  },
  {
    id: "bf16",
    label: "bf16",
    bits: 16,
    exponentBits: 8,
    fractionBits: 7,
    largest: (2 - 2 ** -7) * 2 ** 127,
    smallestNormal: 2 ** -126,
    smallestSubnormal: 2 ** -133,
  },
  {
    id: "fp16",
    label: "fp16",
    bits: 16,
    exponentBits: 5,
    fractionBits: 10,
    largest: HALF_MAX,
    smallestNormal: HALF_MIN_NORMAL,
    smallestSubnormal: HALF_MIN_SUBNORMAL,
  },
];

/**
 * The gradient entries that survive the emulated fp16 cast at a loss scale of 2^exponent: below
 * `flushBelow` an entry rounds to zero, and from `overflowFrom` up it becomes Infinity.
 */
export function halfWindow(exponent: number) {
  const scale = 2 ** exponent;
  return { scale, flushBelow: HALF_MIN_SUBNORMAL / 2 / scale, overflowFrom: HALF_OVERFLOW / scale };
}
