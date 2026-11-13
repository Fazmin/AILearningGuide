/**
 * The network every card in this lab draws: one input, some hidden layers, one output. A hidden
 * neuron computes tanh(weights · inputs + bias); the output neuron is a plain weighted sum plus a
 * bias. Every weight and every bias is one "knob", kept in a flat list in the order `knobLayout` gives.
 *
 * The loss is the mean squared gap between the output and the dots, and the gradient is exact
 * backpropagation (net.test.ts checks it against nudging each knob).
 */

export interface Net {
  /** Layer sizes from input to output, such as [1, 2, 1]. */
  sizes: number[];
}

export interface Dots {
  xs: number[];
  ys: number[];
}

/** One knob: the wire it sits on (`from` ≥ 0) or the neuron whose bias it is (`from` = -1). */
export interface KnobPlace {
  /** 1 for the first layer of knobs (input to first hidden layer), and so on. */
  layer: number;
  /** Neuron in that layer the knob feeds. */
  to: number;
  /** Neuron in the layer before, or -1 for a bias. */
  from: number;
}

export const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

export const netOf = (hidden: readonly number[]): Net => ({ sizes: [1, ...hidden, 1] });

/** How many knobs a list of layer sizes has: one per wire, plus one bias per neuron after the input. */
export function knobCount(sizes: readonly number[]): number {
  let total = 0;
  for (let layer = 1; layer < sizes.length; layer += 1) total += sizes[layer] * sizes[layer - 1] + sizes[layer];
  return total;
}

/** Knobs per layer of wires, biases included: [16, 72, 9] for 1 → 8 → 8 → 1. */
export function knobsPerLayer(sizes: readonly number[]): number[] {
  const counts: number[] = [];
  for (let layer = 1; layer < sizes.length; layer += 1) counts.push(sizes[layer] * sizes[layer - 1] + sizes[layer]);
  return counts;
}

/**
 * Where each knob in the flat list lives. Within a layer the order is every wire into neuron 0, then
 * neuron 0's bias, then neuron 1's wires and bias, and so on.
 */
export function knobLayout(sizes: readonly number[]): KnobPlace[] {
  const places: KnobPlace[] = [];
  for (let layer = 1; layer < sizes.length; layer += 1) {
    for (let to = 0; to < sizes[layer]; to += 1) {
      for (let from = 0; from < sizes[layer - 1]; from += 1) places.push({ layer, to, from });
      places.push({ layer, to, from: -1 });
    }
  }
  return places;
}


/** Every neuron's value for one input: `values[0]` is [x], the last entry is [output]. */
export function forwardValues(net: Net, knobs: readonly number[], x: number): number[][] {
  const values: number[][] = [[x]];
  let at = 0;
  const last = net.sizes.length - 1;
  for (let layer = 1; layer <= last; layer += 1) {
    const before = values[layer - 1];
    const row: number[] = [];
    for (let to = 0; to < net.sizes[layer]; to += 1) {
      let z = 0;
      for (let from = 0; from < before.length; from += 1) z += knobs[at + from] * before[from];
      z += knobs[at + before.length];
      at += before.length + 1;
      row.push(layer === last ? z : Math.tanh(z));
    }
    values.push(row);
  }
  return values;
}

export function output(net: Net, knobs: readonly number[], x: number): number {
  const values = forwardValues(net, knobs, x);
  return values[values.length - 1][0];
}

/** Mean squared gap between the network's output and the dots. */
export function loss(net: Net, knobs: readonly number[], dots: Dots): number {
  let total = 0;
  for (let index = 0; index < dots.xs.length; index += 1) {
    const gap = output(net, knobs, dots.xs[index]) - dots.ys[index];
    total += gap * gap;
  }
  return total / Math.max(1, dots.xs.length);
}

/** The loss and its slope along every knob, by backpropagation. */
export function lossAndGradient(net: Net, knobs: readonly number[], dots: Dots): { loss: number; gradient: number[] } {
  const gradient = new Array<number>(knobs.length).fill(0);
  const last = net.sizes.length - 1;
  const count = Math.max(1, dots.xs.length);
  // Where each layer's knobs start in the flat list.
  const starts: number[] = [0, 0];
  for (let layer = 2; layer <= last; layer += 1) {
    starts[layer] = starts[layer - 1] + net.sizes[layer - 1] * (net.sizes[layer - 2] + 1);
  }
  let total = 0;
  for (let index = 0; index < dots.xs.length; index += 1) {
    const values: number[][] = [[dots.xs[index]]];
    for (let layer = 1; layer <= last; layer += 1) {
      const before = values[layer - 1];
      const row: number[] = [];
      for (let to = 0; to < net.sizes[layer]; to += 1) {
        const base = starts[layer] + to * (before.length + 1);
        let z = knobs[base + before.length];
        for (let from = 0; from < before.length; from += 1) z += knobs[base + from] * before[from];
        row.push(layer === last ? z : Math.tanh(z));
      }
      values.push(row);
    }
    const gap = values[last][0] - dots.ys[index];
    total += gap * gap;
    // Backward: `delta` is the loss's slope along each neuron's weighted sum.
    let delta = [(2 * gap) / count];
    for (let layer = last; layer >= 1; layer -= 1) {
      const before = values[layer - 1];
      const earlier = new Array<number>(before.length).fill(0);
      for (let to = 0; to < net.sizes[layer]; to += 1) {
        const base = starts[layer] + to * (before.length + 1);
        for (let from = 0; from < before.length; from += 1) {
          gradient[base + from] += delta[to] * before[from];
          earlier[from] += delta[to] * knobs[base + from];
        }
        gradient[base + before.length] += delta[to];
      }
      // tanh's slope, written with its own output a: 1 − a².
      if (layer > 1) delta = earlier.map((value, unit) => value * (1 - values[layer - 1][unit] ** 2));
    }
  }
  return { loss: total / count, gradient };
}

/** One plain step: every knob moves against its own slope by step size × slope, and stops at ±limit. */
export function stepKnobs(knobs: readonly number[], gradient: readonly number[], stepSize: number, limit: number): number[] {
  return knobs.map((value, index) => {
    const next = value - stepSize * gradient[index];
    return Number.isFinite(next) ? clamp(next, -limit, limit) : clamp(value, -limit, limit);
  });
}

export interface Run {
  /** Knob settings after 0, 1, 2 … steps. */
  knobs: number[][];
  /** Loss at each of those settings. */
  losses: number[];
}

/**
 * Repeat the step `steps` times from `start`, keeping every setting on the way. With `momentum`
 * above 0 each knob also keeps that share of its previous move: move = momentum × last move −
 * step size × slope. A knob never passes ±limit, so a step size that is far too large cannot
 * produce a number the lab cannot draw.
 */
export function descend(
  net: Net,
  start: readonly number[],
  dots: Dots,
  stepSize: number,
  steps: number,
  limit: number,
  momentum = 0,
): Run {
  const knobs: number[][] = [start.map((value) => clamp(Number.isFinite(value) ? value : 0, -limit, limit))];
  const losses: number[] = [];
  let moves = start.map(() => 0);
  for (let step = 0; step < steps; step += 1) {
    const current = knobs[knobs.length - 1];
    const { loss: value, gradient } = lossAndGradient(net, current, dots);
    losses.push(value);
    moves = moves.map((move, index) => {
      const next = momentum * move - stepSize * gradient[index];
      return Number.isFinite(next) ? next : 0;
    });
    knobs.push(current.map((knob, index) => clamp(knob + moves[index], -limit, limit)));
  }
  losses.push(loss(net, knobs[knobs.length - 1], dots));
  return { knobs, losses };
}

/** A small seeded generator (mulberry32): the same seed always scrambles the knobs the same way. */
export function seeded(seed: number): () => number {
  let state = (Math.floor(seed) >>> 0) + 0x6d2b79f5;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random starting knobs between −spread and +spread, on a grid of `grid`. */
export function scrambled(count: number, seed: number, spread: number, grid = 0.1): number[] {
  const next = seeded(seed);
  return Array.from({ length: count }, () => {
    const value = (next() * 2 - 1) * spread;
    // "+ 0" turns a negative zero into a plain zero, so a stored knob reads back as it was written.
    return Number((Math.round(value / grid) * grid).toFixed(3)) + 0;
  });
}

/** `count` evenly spaced inputs from `low` to `high`. */
export function spaced(low: number, high: number, count: number): number[] {
  return Array.from({ length: count }, (_, index) => Number((low + ((high - low) * index) / (count - 1)).toFixed(6)));
}
