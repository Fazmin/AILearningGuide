import { describe, expect, it } from "vitest";
import {
  accuracy,
  DEEP_NARROW_NET,
  flipLabels,
  forward,
  gradients,
  initNetwork,
  LARGE_NET,
  layerSizes,
  makeDataset,
  makeRingHeldOut,
  meanLoss,
  NARROW_NET,
  NOISE_FLIPS,
  parameterCount,
  SHALLOW_WIDE_NET,
  train,
  trainRingSplit,
  unitBoundary,
  type HiddenActivation,
  type Network,
} from "./mlp";

describe("tiny MLP", () => {
  it("builds balanced fixed datasets", () => {
    const xor = makeDataset("xor");
    const ring = makeDataset("ring");
    expect(xor).toHaveLength(48);
    expect(xor.filter((point) => point.label === 1)).toHaveLength(24);
    expect(ring).toHaveLength(60);
    for (const point of [...xor, ...ring]) {
      expect(Math.abs(point.x1)).toBeLessThanOrEqual(1);
      expect(Math.abs(point.x2)).toBeLessThanOrEqual(1);
    }
    expect(makeDataset("xor")).toEqual(xor);
  });

  it("counts weights plus biases", () => {
    expect(parameterCount(layerSizes(1, 2))).toBe(9);
    expect(parameterCount(layerSizes(3, 4))).toBe(12 + 20 + 20 + 5);
    expect(parameterCount(layerSizes(3, 6))).toBe(18 + 42 + 42 + 7);
  });

  it("runs the forward pass exactly as written", () => {
    // Hand-set XOR solution: h1 fires for x1 + x2 > −0.6, h2 for x1 + x2 > 0.6,
    // and the output fires between the two lines.
    const network: Network = {
      layers: [
        { weights: [[10, 10], [10, 10]], biases: [6, -6] },
        { weights: [[5, -5]], biases: [-5] },
      ],
    };
    const pass = forward(network, [0.6, -0.6], "tanh");
    expect(pass.pre[0][0]).toBeCloseTo(6, 10);
    expect(pass.post[1][0]).toBeCloseTo(Math.tanh(6), 10);
    expect(pass.output).toBeGreaterThan(0.5);
    expect(forward(network, [0.6, 0.6], "tanh").output).toBeLessThan(0.5);
    expect(accuracy(network, makeDataset("xor"), "tanh")).toBe(1);
  });

  it.each<HiddenActivation>(["tanh", "relu", "linear"])(
    "backpropagation matches finite differences (%s)",
    (activation) => {
      const data = makeDataset("xor").slice(0, 12);
      const network = initNetwork(layerSizes(2, 3), 5, activation);
      const { grads } = gradients(network, data, activation);
      const h = 1e-6;
      for (const [layer, unit, input] of [
        [0, 0, 0],
        [0, 2, 1],
        [1, 1, 2],
        [2, 0, 1],
      ] as const) {
        const original = network.layers[layer].weights[unit][input];
        network.layers[layer].weights[unit][input] = original + h;
        const up = meanLoss(network, data, activation);
        network.layers[layer].weights[unit][input] = original - h;
        const down = meanLoss(network, data, activation);
        network.layers[layer].weights[unit][input] = original;
        expect(grads[layer].weights[unit][input]).toBeCloseTo((up - down) / (2 * h), 6);
      }
    },
  );

  it("records loss and accuracy for every epoch, starting from the initialization", () => {
    const run = train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 2 });
    expect(run.snapshots).toHaveLength(301);
    expect(run.losses).toHaveLength(301);
    expect(run.losses[0]).toBeCloseTo(meanLoss(run.snapshots[0], run.data, "tanh"), 10);
    expect(run.losses[300]).toBeCloseTo(meanLoss(run.snapshots[300], run.data, "tanh"), 10);
  });

  // The measured claims quoted in the lesson.
  it("the default run (XOR, tanh, one layer of 2, seed 2) reaches 100% by epoch 28", () => {
    const run = train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 2 });
    expect(run.accuracies.findIndex((value) => value === 1)).toBe(28);
    expect(run.accuracies[300]).toBe(1);
    expect(run.losses[300]).toBeLessThan(0.05);
  });

  it("one hidden unit cannot solve XOR at any depth", () => {
    for (let depth = 1; depth <= 3; depth += 1) {
      for (let seed = 1; seed <= 4; seed += 1) {
        const run = train({ dataset: "xor", depth, width: 1, activation: "tanh", seed });
        expect(Math.max(...run.accuracies)).toBeLessThanOrEqual(0.75);
      }
    }
  });

  it("without a nonlinearity, depth and width do not help on XOR", () => {
    for (const [depth, width] of [
      [1, 6],
      [3, 6],
    ]) {
      const run = train({ dataset: "xor", depth, width, activation: "linear", seed: 2 });
      expect(run.accuracies[300]).toBeLessThanOrEqual(0.75);
    }
  });

  it("some seeds leave two tanh units stuck on XOR", () => {
    const run = train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 1 });
    expect(run.accuracies[300]).toBeLessThan(1);
  });

  it("the ring needs three units in one hidden layer; two plateau below 90%", () => {
    for (let seed = 1; seed <= 4; seed += 1) {
      const narrow = train({ dataset: "ring", depth: 1, width: 2, activation: "tanh", seed });
      expect(Math.max(...narrow.accuracies)).toBeLessThan(0.9);
      const wide = train({ dataset: "ring", depth: 1, width: 3, activation: "tanh", seed });
      expect(wide.accuracies[300]).toBe(1);
    }
  });

  it("reproduces the other numbers the lesson quotes", () => {
    const first = (run: ReturnType<typeof train>) => run.accuracies.findIndex((value) => value === 1);
    expect(train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 2 }).accuracies[0]).toBeCloseTo(0.479, 3);
    expect(train({ dataset: "xor", depth: 1, width: 1, activation: "tanh", seed: 2 }).losses[150]).toBeCloseTo(0.482, 3);
    expect(train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 1 }).accuracies[300]).toBe(0.625);
    expect(train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 3 }).accuracies[300]).toBe(0.6875);
    expect(first(train({ dataset: "xor", depth: 2, width: 3, activation: "tanh", seed: 2 }))).toBe(10);
    expect(train({ dataset: "ring", depth: 1, width: 2, activation: "tanh", seed: 2 }).accuracies[300]).toBeCloseTo(52 / 60, 10);
    expect(first(train({ dataset: "ring", depth: 1, width: 3, activation: "tanh", seed: 2 }))).toBe(74);
    const linear = train({ dataset: "xor", depth: 3, width: 6, activation: "linear", seed: 2 });
    expect(linear.accuracies[300]).toBe(0.5);
    expect(linear.losses[300]).toBeCloseTo(Math.LN2, 3);
  });

  it("with three or more tanh units per layer, seeds 1 to 6 all solve XOR", { timeout: 60000 }, () => {
    for (const depth of [1, 2, 3]) {
      for (const width of [3, 4, 6]) {
        for (let seed = 1; seed <= 6; seed += 1) {
          expect(train({ dataset: "xor", depth, width, activation: "tanh", seed }).accuracies[300]).toBe(1);
        }
      }
    }
  });

  it("clips a unit's zero line to the square", () => {
    expect(unitBoundary([1, 0], 0)).toEqual([
      [0, -1],
      [0, 1],
    ]);
    expect(unitBoundary([1, 1], 5)).toBeNull();
  });
});

const SEEDS = [1, 2, 3, 4, 5, 6];
const gapOf = (history: ReturnType<typeof train>, epoch = 300) =>
  history.accuracies[epoch] - (history.heldAccuracies as number[])[epoch];

describe("checkerboard dataset", () => {
  it("is a 4 × 4 board with four points in the middle half of every cell", { timeout: 60000 }, () => {
    const board = makeDataset("checker");
    expect(board).toHaveLength(64);
    expect(board.filter((point) => point.label === 1)).toHaveLength(32);
    board.forEach((point, index) => {
      const cell = Math.floor(index / 4);
      const row = Math.floor(cell / 4);
      const column = cell % 4;
      expect(point.label).toBe((row + column) % 2);
      // Cell [row, column] spans 0.5 in each direction; points stay in its middle half.
      expect(point.x1).toBeGreaterThanOrEqual(-1 + (row + 0.25) * 0.5 - 1e-12);
      expect(point.x1).toBeLessThanOrEqual(-1 + (row + 0.75) * 0.5 + 1e-12);
      expect(point.x2).toBeGreaterThanOrEqual(-1 + (column + 0.25) * 0.5 - 1e-12);
      expect(point.x2).toBeLessThanOrEqual(-1 + (column + 0.75) * 0.5 + 1e-12);
    });
    expect(makeDataset("checker")).toEqual(board);
  });
});

describe("depth against width at the same parameter count", () => {
  it("gives the two shapes exactly the same number of parameters", { timeout: 60000 }, () => {
    expect(parameterCount(layerSizes(SHALLOW_WIDE_NET.depth, SHALLOW_WIDE_NET.width))).toBe(85);
    expect(parameterCount(layerSizes(DEEP_NARROW_NET.depth, DEEP_NARROW_NET.width))).toBe(85);
    expect(parameterCount(layerSizes(NARROW_NET.depth, NARROW_NET.width))).toBe(13);
    expect(parameterCount(layerSizes(LARGE_NET.depth, LARGE_NET.width))).toBe(67);
  });

  it("on XOR and Ring both shapes reach 100% at seeds 1 to 6", { timeout: 60000 }, () => {
    for (const dataset of ["xor", "ring"] as const) {
      for (const seed of SEEDS) {
        for (const shape of [SHALLOW_WIDE_NET, DEEP_NARROW_NET]) {
          expect(train({ dataset, ...shape, activation: "tanh", seed }).accuracies[300]).toBe(1);
        }
      }
    }
  });

  it("on the checkerboard with tanh, one layer of 21 stays near chance and two layers of 7 fit most of the board", { timeout: 60000 }, () => {
    for (const seed of SEEDS) {
      const wide = train({ dataset: "checker", ...SHALLOW_WIDE_NET, activation: "tanh", seed });
      const deep = train({ dataset: "checker", ...DEEP_NARROW_NET, activation: "tanh", seed });
      expect(wide.accuracies[300]).toBeLessThanOrEqual(0.6);
      expect(deep.accuracies[300]).toBeGreaterThanOrEqual(0.8);
      expect(deep.accuracies[300] - wide.accuracies[300]).toBeGreaterThan(0.2);
    }
    // The numbers the lesson and the step instruction quote, at the lab's default seed.
    const wide = train({ dataset: "checker", ...SHALLOW_WIDE_NET, activation: "tanh", seed: 2 });
    const deep = train({ dataset: "checker", ...DEEP_NARROW_NET, activation: "tanh", seed: 2 });
    expect(wide.accuracies[300]).toBeCloseTo(36 / 64, 10);
    expect(deep.accuracies[300]).toBeCloseTo(56 / 64, 10);
    expect(wide.accuracies[150]).toBeCloseTo(29 / 64, 10);
    expect(deep.accuracies[150]).toBeCloseTo(36 / 64, 10);
  });

  it("with ReLU units the wide layer fits as well as the deep one or better, so the advantage is not a law", { timeout: 30000 }, () => {
    let strictlyBetter = 0;
    for (const seed of SEEDS) {
      const wide = train({ dataset: "checker", ...SHALLOW_WIDE_NET, activation: "relu", seed });
      const deep = train({ dataset: "checker", ...DEEP_NARROW_NET, activation: "relu", seed });
      expect(wide.accuracies[300]).toBeGreaterThanOrEqual(deep.accuracies[300]);
      expect(wide.accuracies[300]).toBeGreaterThanOrEqual(0.85);
      if (wide.accuracies[300] > deep.accuracies[300]) strictlyBetter += 1;
    }
    expect(strictlyBetter).toBe(4);
    const minimum = (shape: typeof SHALLOW_WIDE_NET | typeof DEEP_NARROW_NET) =>
      Math.min(...SEEDS.map((seed) => train({ dataset: "checker", ...shape, activation: "relu", seed }).accuracies[300]));
    expect(minimum(SHALLOW_WIDE_NET)).toBeCloseTo(55 / 64, 10);
    expect(minimum(DEEP_NARROW_NET)).toBeCloseTo(44 / 64, 10);
    const maximum = (shape: typeof SHALLOW_WIDE_NET | typeof DEEP_NARROW_NET) =>
      Math.max(...SEEDS.map((seed) => train({ dataset: "checker", ...shape, activation: "relu", seed }).accuracies[300]));
    expect(maximum(SHALLOW_WIDE_NET)).toBe(1);
    expect(maximum(DEEP_NARROW_NET)).toBe(1);
  });

  it("without a nonlinearity both shapes sit at the majority-class score and loss ln 2", { timeout: 60000 }, () => {
    for (const shape of [SHALLOW_WIDE_NET, DEEP_NARROW_NET]) {
      const run = train({ dataset: "checker", ...shape, activation: "linear", seed: 2 });
      expect(run.accuracies[300]).toBeCloseTo(33 / 64, 10);
      expect(run.losses[300]).toBeCloseTo(Math.LN2, 3);
    }
  });
});

describe("held-out ring and label noise", () => {
  it("builds 120 clean held-out points that are not the training points", { timeout: 60000 }, () => {
    const held = makeRingHeldOut();
    const training = makeDataset("ring");
    expect(held).toHaveLength(120);
    expect(held.filter((point) => point.label === 1)).toHaveLength(48);
    for (const point of held) {
      const radius = Math.hypot(point.x1, point.x2);
      if (point.label === 1) expect(radius).toBeLessThanOrEqual(0.4 + 1e-9);
      else expect(radius).toBeGreaterThanOrEqual(0.66 - 1e-9);
      expect(training.some((other) => other.x1 === point.x1 && other.x2 === point.x2)).toBe(false);
    }
    expect(makeRingHeldOut()).toEqual(held);
  });

  it("flips exactly twelve of the sixty training labels, the same twelve every time", { timeout: 60000 }, () => {
    const clean = makeDataset("ring");
    const noisy = flipLabels(clean, NOISE_FLIPS);
    expect(NOISE_FLIPS / clean.length).toBeCloseTo(0.2, 12);
    const changed = noisy.filter((point, index) => point.label !== clean[index].label);
    expect(changed).toHaveLength(12);
    noisy.forEach((point, index) => {
      expect(point.x1).toBe(clean[index].x1);
      expect(point.x2).toBe(clean[index].x2);
    });
    expect(flipLabels(clean, NOISE_FLIPS)).toEqual(noisy);
    expect(clean.filter((point) => point.label === 1)).toHaveLength(24);
  });

  it("records held-out loss and accuracy for every epoch from the saved networks", { timeout: 60000 }, () => {
    const run = trainRingSplit({ ...LARGE_NET, seed: 2, noisy: true });
    const held = makeRingHeldOut();
    expect(run.heldLosses).toHaveLength(301);
    expect(run.heldAccuracies).toHaveLength(301);
    expect(run.heldLosses?.[120]).toBeCloseTo(meanLoss(run.snapshots[120], held, "tanh"), 12);
    expect(run.heldAccuracies?.[300]).toBeCloseTo(accuracy(run.snapshots[300], held, "tanh"), 12);
    expect(train({ dataset: "ring", depth: 1, width: 2, activation: "tanh", seed: 2 }).heldLosses).toBeUndefined();
  });

  it("with clean labels both networks reach 100% on training points and at least 99% on held-out points", { timeout: 60000 }, () => {
    for (const seed of SEEDS) {
      const large = trainRingSplit({ ...LARGE_NET, seed, noisy: false });
      const narrow = trainRingSplit({ ...NARROW_NET, seed, noisy: false });
      expect(large.accuracies[300]).toBe(1);
      expect(large.heldAccuracies?.[300]).toBe(1);
      expect(narrow.accuracies[300]).toBe(1);
      expect(narrow.heldAccuracies?.[300]).toBeGreaterThanOrEqual(0.99);
    }
  });

  it("with 20% flipped labels the large network's gap is at least 15 points and the narrow network's at most 7, at seeds 1 to 6", { timeout: 60000 }, () => {
    for (const seed of SEEDS) {
      const large = trainRingSplit({ ...LARGE_NET, seed, noisy: true });
      const narrow = trainRingSplit({ ...NARROW_NET, seed, noisy: true });
      expect(gapOf(large)).toBeGreaterThanOrEqual(0.15 - 1e-9);
      expect(gapOf(narrow)).toBeLessThanOrEqual(0.07);
      expect(gapOf(large) - gapOf(narrow)).toBeGreaterThanOrEqual(0.1);
      // The large network's held-out loss climbs well above its own best while its training loss stays tiny.
      const best = Math.min(...(large.heldLosses as number[]));
      expect(large.heldLosses?.[300]).toBeGreaterThan(2 * best);
      expect(large.losses[300]).toBeLessThan(0.15);
    }
  });

  it("pins the numbers the step instruction and lesson quote at seed 2", { timeout: 60000 }, () => {
    const large = trainRingSplit({ ...LARGE_NET, seed: 2, noisy: true });
    const narrow = trainRingSplit({ ...NARROW_NET, seed: 2, noisy: true });
    expect(large.accuracies[300]).toBeCloseTo(58 / 60, 10);
    expect(large.heldAccuracies?.[300]).toBeCloseTo(95 / 120, 10);
    expect(narrow.accuracies[300]).toBeCloseTo(54 / 60, 10);
    expect(narrow.heldAccuracies?.[300]).toBeCloseTo(110 / 120, 10);
    // The default marker is epoch 150.
    expect(large.accuracies[150]).toBeCloseTo(53 / 60, 10);
    expect(large.heldAccuracies?.[150]).toBeCloseTo(91 / 120, 10);
    expect(narrow.accuracies[150]).toBeCloseTo(54 / 60, 10);
    expect(narrow.heldAccuracies?.[150]).toBeCloseTo(108 / 120, 10);
  });
});

describe("numbers the lesson and card explanations quote", () => {
  it("quotes the checkerboard ranges across seeds 1 to 6 for tanh", { timeout: 60000 }, () => {
    const finals = (shape: typeof SHALLOW_WIDE_NET | typeof DEEP_NARROW_NET) =>
      SEEDS.map((seed) => train({ dataset: "checker", ...shape, activation: "tanh", seed }).accuracies[300]);
    const wide = finals(SHALLOW_WIDE_NET);
    const deep = finals(DEEP_NARROW_NET);
    expect(Math.min(...wide)).toBeCloseTo(30 / 64, 10);
    expect(Math.max(...wide)).toBeCloseTo(37 / 64, 10);
    expect(Math.min(...deep)).toBeCloseTo(52 / 64, 10);
    expect(Math.max(...deep)).toBe(1);
  });

  it("quotes the main lab's checkerboard ladder at seed 2: 54.7%, 93.8%, 100% for 1, 2, 3 layers of 6", { timeout: 60000 }, () => {
    const at = (depth: number) => train({ dataset: "checker", depth, width: 6, activation: "tanh", seed: 2 }).accuracies[300];
    expect(at(1)).toBeCloseTo(35 / 64, 10);
    expect(at(2)).toBeCloseTo(60 / 64, 10);
    expect(at(3)).toBe(1);
    expect(parameterCount(layerSizes(1, 6))).toBe(25);
    expect(parameterCount(layerSizes(2, 6))).toBe(67);
    expect(parameterCount(layerSizes(3, 6))).toBe(109);
  });

  it("quotes the held-out curves at seed 2: held-out loss bottoms near 0.28 at epoch 71, then climbs to 0.99 while training loss falls to 0.07", { timeout: 60000 }, () => {
    const large = trainRingSplit({ ...LARGE_NET, seed: 2, noisy: true });
    const held = large.heldLosses as number[];
    const lowest = Math.min(...held);
    expect(held.indexOf(lowest)).toBe(71);
    expect(lowest).toBeCloseTo(0.283, 3);
    expect(held[300]).toBeCloseTo(0.987, 3);
    expect(large.losses[300]).toBeCloseTo(0.072, 3);
    const peak = Math.max(...(large.heldAccuracies as number[]));
    expect(peak).toBeCloseTo(109 / 120, 10);
    expect((large.heldAccuracies as number[]).indexOf(peak)).toBe(69);
    const narrow = trainRingSplit({ ...NARROW_NET, seed: 2, noisy: true });
    expect(narrow.losses[300]).toBeCloseTo(0.32, 2);
    expect(narrow.heldLosses?.[300]).toBeCloseTo(0.281, 3);
  });

  it("changes the size of the gap when a different set of twelve labels is flipped", { timeout: 60000 }, () => {
    const clean = makeDataset("ring");
    const held = makeRingHeldOut();
    const gaps = [1, 6, 7, 8].map((flipSeed) => {
      const run = train({
        dataset: "ring",
        depth: 2,
        width: 6,
        activation: "tanh",
        seed: 2,
        data: flipLabels(clean, NOISE_FLIPS, flipSeed),
        heldOut: held,
      });
      return gapOf(run);
    });
    expect(Math.max(...gaps) - Math.min(...gaps)).toBeGreaterThan(0.1);
    expect(Math.min(...gaps)).toBeLessThan(0.05);
  });
});
