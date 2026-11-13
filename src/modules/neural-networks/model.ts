import { descend, knobCount, netOf, output, scrambled, spaced, type Dots, type Net, type Run } from "./net";

/**
 * Everything the six cards share: the dots each one fits, the fixed recipes that train them, and the
 * real networks on the size ladder. Every number a card prints comes from here and from net.ts.
 */

/** Inputs run from −2 to 2, and every set of dots has one dot at each of these 17 inputs. */
export const X_RANGE = [-2, 2] as const;
export const XS = spaced(X_RANGE[0], X_RANGE[1], 17);
/** The plots' vertical range. */
export const Y_RANGE = [-2.6, 2.6] as const;

const dotsOf = (rule: (x: number) => number): Dots => ({ xs: XS, ys: XS.map(rule) });

/* ---------------- card 1: two knobs ---------------- */

/** A knob turned by hand stops at ±3 and clicks in tenths. */
export const HAND_LIMIT = 3;
export const HAND_GRID = 0.1;
/** The setting the line's dots were made with. */
export const LINE_MADE_WITH = { w: 0.8, b: 0.5 } as const;
export const LINE_DOTS = dotsOf((x) => LINE_MADE_WITH.w * x + LINE_MADE_WITH.b);
export const LINE_START = { w: -0.5, b: -1 } as const;
/** Under this the two-knob machine counts as fitted; on the tenths grid only the exact setting gets there. */
export const LINE_GOAL = 0.005;

export const lineOutput = (w: number, b: number, x: number) => w * x + b;
export function lineLoss(w: number, b: number): number {
  let total = 0;
  for (let index = 0; index < XS.length; index += 1) total += (lineOutput(w, b, XS[index]) - LINE_DOTS.ys[index]) ** 2;
  return total / XS.length;
}

/* ---------------- cards 2 and 3: seven knobs ---------------- */

/** One input, two hidden neurons, one output. Knob order: w₁ b₁ w₂ b₂ v₁ v₂ c. */
export const SMALL_NET: Net = netOf([2]);
export const SMALL_NAMES = ["w₁", "b₁", "w₂", "b₂", "v₁", "v₂", "c"] as const;
export const SMALL_ROLES = [
  "the knob on the wire from the input to neuron 1",
  "neuron 1's own knob",
  "the knob on the wire from the input to neuron 2",
  "neuron 2's own knob",
  "the knob on the wire from neuron 1 to the output",
  "the knob on the wire from neuron 2 to the output",
  "the output's own knob",
] as const;
/** The setting the hill's dots were made with, so a loss of exactly 0 is within reach of the knobs. */
export const HILL_MADE_WITH = [2, 2, 2, -2, 1.5, -1.5, -1] as const;
export const HILL_DOTS = dotsOf((x) => output(SMALL_NET, HILL_MADE_WITH, x));
/** Where the seven knobs sit when the lab opens: the first scramble. */
export const SCRAMBLE_SPREAD = 1.5;
export const scrambleSmall = (seed: number) => scrambled(SMALL_NAMES.length, seed, SCRAMBLE_SPREAD, HAND_GRID);
export const HAND_START = scrambleSmall(1);
export const HAND_GOAL = 0.05;

/** Card 3's rule: plain steps of this size, at most this many. */
export const AUTO_STEP = 0.2;
export const AUTO_MAX = 300;
export const AUTO_GOAL = 0.01;

const cache = new Map<string, Run>();
/** Training is repeatable, so a run is worked out once per setting and kept. */
function remembered(key: string, make: () => Run): Run {
  const found = cache.get(key);
  if (found) return found;
  if (cache.size > 96) cache.clear();
  const run = make();
  cache.set(key, run);
  return run;
}

/** Every setting the machine passes through from the hand-set knobs, for all `AUTO_MAX` steps. */
export function autoRun(hand: readonly number[]): Run {
  return remembered(`auto:${hand.join(",")}`, () => descend(SMALL_NET, hand, HILL_DOTS, AUTO_STEP, AUTO_MAX, HAND_LIMIT));
}

/** Trade the two hidden neurons' knobs. The network's output cannot change: a sum does not care about order. */
export const swapNeurons = (k: readonly number[]) => [k[2], k[3], k[0], k[1], k[5], k[4], k[6]];

/** Each hidden neuron's share of the output at one input: vᵢ × tanh(wᵢ·x + bᵢ). */
export function smallPieces(k: readonly number[], x: number): [number, number] {
  return [k[4] * Math.tanh(k[0] * x + k[1]), k[5] * Math.tanh(k[2] * x + k[3])];
}

/* ---------------- card 4: more neurons ---------------- */

export const WAVE_DOTS = dotsOf((x) => -1.2 * Math.cos(Math.PI * x));
export const MAX_UNITS = 6;
/** Knobs trained by the machine may travel this far. */
export const TRAIN_LIMIT = 8;
/** The fixed recipe for cards 4 and 6: this many steps, each knob keeping 0.9 of its last move. */
export const TRAIN_STEPS = 600;
export const WIDE_STEP = 0.05;
export const MOMENTUM = 0.9;

export const wideNet = (units: number): Net => netOf([units]);
export function wideRun(units: number, seed: number): Run {
  return remembered(`wide:${units}:${seed}`, () => {
    const net = wideNet(units);
    return descend(net, scrambled(knobCount(net.sizes), seed, SCRAMBLE_SPREAD), WAVE_DOTS, WIDE_STEP, TRAIN_STEPS, TRAIN_LIMIT, MOMENTUM);
  });
}

/** One hidden neuron's share of a one-hidden-layer network's output: vᵢ × tanh(wᵢ·x + bᵢ). */
export function widePiece(units: number, knobs: readonly number[], unit: number, x: number): number {
  return knobs[2 * units + unit] * Math.tanh(knobs[2 * unit] * x + knobs[2 * unit + 1]);
}

/* ---------------- card 5: count the knobs ---------------- */

export const INPUT_STOPS = [1, 2, 16, 784] as const;
export const WIDTH_STOPS = [2, 4, 16, 64, 256, 1024, 4096] as const;
export const OUTPUT_STOPS = [1, 10] as const;
export const MAX_LAYERS = 6;

export const builtSizes = (inputs: number, layers: number, width: number, outputs: number) => [
  inputs,
  ...Array.from({ length: layers }, () => width),
  outputs,
];

export interface Landmark {
  id: string;
  name: string;
  what: string;
  knobs: number;
  /** The count as its paper rounds it, in words. */
  said: string;
}

/** Published parameter counts, each cited in the module's references. */
export const LANDMARKS: readonly Landmark[] = [
  { id: "lenet", name: "LeNet-5 (1998)", what: "reads handwritten digits", knobs: 60_000, said: "60 thousand" },
  { id: "alexnet", name: "AlexNet (2012)", what: "names the object in a photo", knobs: 60_000_000, said: "60 million" },
  { id: "gpt2", name: "GPT-2 (2019)", what: "writes text; the largest of its four sizes", knobs: 1_500_000_000, said: "1.5 billion" },
  { id: "gpt3", name: "GPT-3 (2020)", what: "writes text", knobs: 175_000_000_000, said: "175 billion" },
];

/* ---------------- card 6: the playground ---------------- */

export type DotsKind = "slope" | "hill" | "wave" | "step";
export const DOTS_KINDS: readonly DotsKind[] = ["slope", "hill", "wave", "step"];
export const PLAY_DOTS: Record<DotsKind, Dots> = {
  slope: LINE_DOTS,
  hill: HILL_DOTS,
  wave: WAVE_DOTS,
  step: dotsOf((x) => Math.sign(x)),
};
export const DOTS_NAMES: Record<DotsKind, string> = { slope: "Slope", hill: "Hill", wave: "Wave", step: "Step" };
export const RATE_STOPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] as const;
export const PLAY_GOAL = 0.01;

export interface PlaySetup {
  dots: DotsKind;
  layers: number;
  width: number;
  rate: number;
  momentum: boolean;
  seed: number;
}

export const playNet = (setup: Pick<PlaySetup, "layers" | "width">): Net =>
  netOf(Array.from({ length: setup.layers }, () => setup.width));

export function playRun(setup: PlaySetup): Run {
  const key = `play:${setup.dots}:${setup.layers}:${setup.width}:${setup.rate}:${setup.momentum ? 1 : 0}:${setup.seed}`;
  return remembered(key, () => {
    const net = playNet(setup);
    return descend(
      net,
      scrambled(knobCount(net.sizes), setup.seed, SCRAMBLE_SPREAD),
      PLAY_DOTS[setup.dots],
      setup.rate,
      TRAIN_STEPS,
      TRAIN_LIMIT,
      setup.momentum ? MOMENTUM : 0,
    );
  });
}

/** The first step at which a run's loss is under `goal`, or -1. */
export const firstUnder = (run: Run, goal: number) => run.losses.findIndex((value) => value < goal);
