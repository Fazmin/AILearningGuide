import { describe, expect, it } from "vitest";
import {
  FLOW_ACTIVATIONS,
  FLOW_BATCH,
  FLOW_INITS,
  FLOW_WIDTH,
  buildFlowNetwork,
  flowDepth,
  flowGradients,
  flowLoss,
  initStd,
  makeFlowBatch,
  measureGradientFlow,
  type FlowActivation,
  type FlowInit,
} from "./gradient-flow";

const SEEDS = [1, 2, 3, 4, 5, 6, 7];
const flow = (depth: number, activation: FlowActivation, init: FlowInit, skip = false, seed = 7) =>
  measureGradientFlow({ depth, activation, init, skip, seed });
/** The value as the lab prints it, at two significant figures. */
const sci = (value: number) => Number(value.toExponential(1));

describe("gradient flow: the computation", () => {
  it("draws the fixed batch and the layer sizes the card describes", () => {
    const batch = makeFlowBatch();
    expect(batch.inputs).toHaveLength(FLOW_BATCH);
    expect(FLOW_BATCH).toBe(32);
    expect(FLOW_WIDTH).toBe(16);
    for (const input of batch.inputs) expect(input).toHaveLength(FLOW_WIDTH);
    const ones = batch.labels.filter((label) => label === 1).length;
    expect(ones).toBeGreaterThan(8);
    expect(ones).toBeLessThan(24);
    expect(makeFlowBatch()).toEqual(batch);
    const network = buildFlowNetwork({ depth: 5, init: "he" });
    expect(network.hidden).toHaveLength(5);
    expect(network.hidden[0]).toHaveLength(16);
    expect(network.hidden[0][0]).toHaveLength(16);
    expect(network.output).toHaveLength(16);
  });

  it("uses the standard deviations it names: Glorot √(2/(in+out)), He √(2/in), 0.1, and 1", () => {
    expect(initStd("glorot", 16, 16)).toBeCloseTo(0.25, 12);
    expect(initStd("he", 16, 16)).toBeCloseTo(Math.sqrt(2 / 16), 12);
    expect(initStd("small", 16, 16)).toBe(0.1);
    expect(initStd("large", 16, 16)).toBe(1);
    expect(initStd("glorot", 16, 1)).toBeCloseTo(Math.sqrt(2 / 17), 12);
  });

  it.each(FLOW_ACTIVATIONS.flatMap((activation) => [false, true].map((skip) => [activation, skip] as const)))(
    "backpropagation matches finite differences (%s, skip %s)",
    (activation, skip) => {
      const batch = makeFlowBatch();
      const network = buildFlowNetwork({ depth: 4, init: "he" });
      const { hidden, output } = flowGradients(network, batch, activation, skip);
      // A larger step than usual: with skip connections the activations grow and the loss saturates, so tiny steps drown in rounding.
      const h = 1e-4;
      const check = (analytic: number, read: () => number, write: (value: number) => void) => {
        const original = read();
        write(original + h);
        const up = flowLoss(network, batch, activation, skip);
        write(original - h);
        const down = flowLoss(network, batch, activation, skip);
        write(original);
        const numeric = (up - down) / (2 * h);
        expect(Math.abs(analytic - numeric)).toBeLessThanOrEqual(1e-6 + 1e-4 * Math.max(Math.abs(analytic), Math.abs(numeric)));
      };
      for (const [layer, unit, input] of [
        [0, 0, 0],
        [0, 9, 15],
        [1, 3, 4],
        [2, 15, 2],
        [3, 7, 7],
      ] as const) {
        check(
          hidden[layer][unit][input],
          () => network.hidden[layer][unit][input],
          (value) => {
            network.hidden[layer][unit][input] = value;
          },
        );
      }
      for (const unit of [0, 5, 15]) {
        check(
          output[unit],
          () => network.output[unit],
          (value) => {
            network.output[unit] = value;
          },
        );
      }
    },
  );

  it("is deterministic, and depth 1 has one layer whose ratio to itself is 1", () => {
    const a = flow(8, "tanh", "glorot");
    expect(flow(8, "tanh", "glorot")).toEqual(a);
    const one = flow(1, "sigmoid", "glorot");
    expect(one.layers).toHaveLength(1);
    expect(one.firstOverLast).toBe(1);
    expect(a.layers).toHaveLength(8);
    expect(a.layers.map((layer) => layer.layer)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("clamps depth into 1 to 12 and never returns a non-finite number for any control combination", () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 1e9, -1e9, 0, 6.6]) {
      const depth = flowDepth(value);
      expect(depth).toBeGreaterThanOrEqual(1);
      expect(depth).toBeLessThanOrEqual(12);
    }
    expect(flowDepth(6.6)).toBe(7);
    for (const activation of FLOW_ACTIVATIONS) {
      for (const init of FLOW_INITS) {
        for (const skip of [false, true]) {
          for (const depth of [1, 6, 12]) {
            const result = flow(depth, activation, init, skip);
            expect(Number.isFinite(result.loss)).toBe(true);
            expect(Number.isFinite(result.firstOverLast)).toBe(true);
            for (const layer of result.layers) {
              expect(Number.isFinite(layer.gradNorm)).toBe(true);
              expect(Number.isFinite(layer.meanSlope)).toBe(true);
              expect(Number.isFinite(layer.activationRms)).toBe(true);
            }
          }
        }
      }
    }
  });
});

describe("gradient flow: the claims the lesson makes at depth 12", () => {
  it("sigmoid vanishes at every starting scale: the first-to-last ratio is below 0.004 at seeds 1 to 7", () => {
    for (const seed of SEEDS) {
      for (const init of FLOW_INITS) {
        expect(flow(12, "sigmoid", init, false, seed).firstOverLast).toBeLessThan(4e-3);
      }
    }
    expect(sci(flow(12, "sigmoid", "he").layers[0].gradNorm)).toBe(4.9e-7);
    expect(sci(flow(12, "sigmoid", "he").firstOverLast)).toBe(2.3e-6);
  });

  it("quotes the default view: depth 6, sigmoid, Glorot, skips off", () => {
    const view = flow(6, "sigmoid", "glorot");
    expect(sci(view.layers[0].gradNorm)).toBe(1.4e-4);
    expect(sci(view.layers[5].gradNorm)).toBe(2.3e-2);
    expect(sci(view.firstOverLast)).toBe(6e-3);
    expect(view.loss.toFixed(2)).toBe("0.69");
  });

  it("sigmoid with Glorot weights: Layer 1's gradient is about 1e-7 of Layer 12's at seeds 1 to 7, and a skip connection restores it", () => {
    for (const seed of SEEDS) {
      const plain = flow(12, "sigmoid", "glorot", false, seed);
      const skipped = flow(12, "sigmoid", "glorot", true, seed);
      expect(plain.firstOverLast).toBeGreaterThan(9e-8);
      expect(plain.firstOverLast).toBeLessThan(3e-7);
      expect(skipped.firstOverLast).toBeGreaterThan(7e-2);
      expect(skipped.layers[0].gradNorm).toBeGreaterThan(1e6 * plain.layers[0].gradNorm);
      expect(skipped.loss).toBeGreaterThan(plain.loss);
    }
    // The figures quoted for the default seed.
    const plain = flow(12, "sigmoid", "glorot");
    const skipped = flow(12, "sigmoid", "glorot", true);
    expect(sci(plain.layers[0].gradNorm)).toBe(2.6e-8);
    expect(plain.layers[11].gradNorm.toFixed(2)).toBe("0.21");
    expect(sci(plain.firstOverLast)).toBe(1.2e-7);
    expect(plain.loss.toFixed(2)).toBe("0.90");
    expect(skipped.layers[0].gradNorm.toFixed(2)).toBe("0.24");
    expect(skipped.firstOverLast.toFixed(3)).toBe("0.077");
    expect(skipped.loss.toFixed(2)).toBe("7.30");
  });

  it("each sigmoid layer passes back about a quarter of the gradient: the geometric mean per layer is about 0.24", () => {
    const plain = flow(12, "sigmoid", "glorot");
    expect(Math.pow(plain.firstOverLast, 1 / 11)).toBeCloseTo(0.235, 2);
    // The sigmoid's slope never exceeds 0.25, and the mean slope is what the table prints.
    for (const layer of plain.layers) {
      expect(layer.meanSlope).toBeLessThanOrEqual(0.25);
      expect(layer.meanSlope).toBeGreaterThan(0.15);
    }
  });

  it("the gradient reaching Layer 1 falls as depth grows for sigmoid, by a bigger factor each time", () => {
    const first = [1, 2, 4, 6, 8, 12].map((depth) => flow(depth, "sigmoid", "glorot").layers[0].gradNorm);
    for (let index = 1; index < first.length; index += 1) expect(first[index]).toBeLessThan(first[index - 1]);
    expect(first[0].toFixed(2)).toBe("0.15");
    expect(sci(first[3])).toBe(1.4e-4);
  });

  it("tanh with Glorot weights keeps every layer within a factor of three of Layer 12, while sigmoid does not", () => {
    for (const seed of SEEDS) {
      const tanh = flow(12, "tanh", "glorot", false, seed);
      expect(tanh.firstOverLast).toBeGreaterThan(1 / 3);
      expect(tanh.firstOverLast).toBeLessThan(3);
      const he = flow(12, "tanh", "he", false, seed).firstOverLast;
      expect(he).toBeGreaterThan(1 / 3);
      expect(he).toBeLessThan(3);
      expect(flow(12, "sigmoid", "he", false, seed).firstOverLast).toBeLessThan(1e-5);
    }
    expect(flow(12, "tanh", "glorot").firstOverLast.toFixed(2)).toBe("0.92");
  });

  it("with large weights tanh's gradient grows toward the input: Layer 1 is more than ten times Layer 12", () => {
    for (const seed of SEEDS) expect(flow(12, "tanh", "large", false, seed).firstOverLast).toBeGreaterThan(10);
    expect(sci(flow(12, "tanh", "large").layers[0].gradNorm)).toBe(57);
  });

  it("ReLU: the starting scale sets how big every layer's gradient is, in the order small < Glorot < He < large", () => {
    for (const seed of SEEDS) {
      const norms = FLOW_INITS.map((init) => flow(12, "relu", init, false, seed).layers[0].gradNorm);
      expect(norms[0]).toBeLessThan(norms[1]);
      expect(norms[1]).toBeLessThan(norms[2]);
      expect(norms[2]).toBeLessThan(norms[3]);
      expect(norms[0]).toBeLessThan(1e-6);
      expect(norms[1]).toBeLessThan(0.1);
      expect(norms[3]).toBeGreaterThan(1e4);
      // The scale moves every layer together: first and last stay within a factor of ten of each other.
      for (const init of FLOW_INITS) {
        const ratio = flow(12, "relu", init, false, seed).firstOverLast;
        expect(ratio).toBeGreaterThan(0.1);
        expect(ratio).toBeLessThan(10);
      }
    }
    const quoted = FLOW_INITS.map((init) => sci(flow(12, "relu", init).layers[0].gradNorm));
    expect(quoted).toEqual([2.4e-7, 2e-2, 1.3, 3.5e5]);
  });
});
