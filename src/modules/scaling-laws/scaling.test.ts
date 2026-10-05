import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  blockWeights,
  clampWidth,
  denseParameters,
  embeddingWeights,
  FITS,
  formatCount,
  formatFlops,
  frontierLossExponent,
  lossTerms,
  numericOptimum,
  optimalAllocation,
  LAYER_RANGE,
  PARAMETER_RANGE,
  predictedLoss,
  ratioAllocation,
  toyBlockExtras,
  trainingFlops,
  WIDTH_RANGE,
} from "./scaling";

describe("scaling-law arithmetic", () => {
  it("evaluates the printed Chinchilla fit with raw counts", () => {
    // 1.69 + 406.4 / (3.2e9)^0.34 + 410.7 / (2.8e10)^0.28
    expect(predictedLoss(FITS.hoffmann, 3.2e9, 28e9)).toBeCloseTo(2.416, 3);
    const terms = lossTerms(FITS.hoffmann, 3.2e9, 28e9);
    expect(terms.size).toBeCloseTo(0.238, 3);
    expect(terms.data).toBeCloseTo(0.488, 3);
    expect(terms.floor + terms.size + terms.data).toBeCloseTo(terms.total, 12);
  });

  it("uses C = 6ND", () => {
    expect(trainingFlops(3.2e9, 28e9)).toBeCloseTo(5.376e20, -16);
    expect(trainingFlops(70e9, 1.4e12)).toBeCloseTo(5.88e23, -19);
  });

  it("matches the closed-form optimum to a brute-force search on every isoFLOP curve", () => {
    for (const fit of Object.values(FITS)) {
      for (const compute of [1e19, 1e21, 5.76e23, 1e25]) {
        const closed = optimalAllocation(fit, compute);
        const numeric = numericOptimum(fit, compute);
        expect(closed.parameters / numeric.parameters).toBeCloseTo(1, 4);
        expect(6 * closed.parameters * closed.tokens).toBeCloseTo(compute, -Math.floor(Math.log10(compute)) + 6);
        // Nothing on the curve beats the optimum.
        for (const factor of [0.5, 0.9, 1.1, 2]) {
          const parameters = closed.parameters * factor;
          expect(predictedLoss(fit, parameters, compute / (6 * parameters))).toBeGreaterThan(closed.loss);
        }
      }
    }
  });

  it("puts the printed Hoffmann optimum at Chinchilla's budget far above 20 tokens per parameter", () => {
    const printed = optimalAllocation(FITS.hoffmann, 5.76e23);
    expect(printed.parameters / 1e9).toBeCloseTo(32.2, 1);
    expect(printed.ratio).toBeGreaterThan(90);
    const refit = optimalAllocation(FITS.epoch, 5.76e23);
    expect(refit.parameters / 1e9).toBeCloseTo(72.2, 1);
    expect(refit.ratio).toBeGreaterThan(17);
    expect(refit.ratio).toBeLessThan(20);
  });

  it("balances marginal returns at the optimum", () => {
    for (const fit of Object.values(FITS)) {
      const best = optimalAllocation(fit, 1e22);
      const terms = lossTerms(fit, best.parameters, best.tokens);
      expect(fit.alpha * terms.size).toBeCloseTo(fit.beta * terms.data, 10);
    }
  });

  it("makes reducible loss on the frontier an exact power law in compute", () => {
    for (const fit of Object.values(FITS)) {
      const low = optimalAllocation(fit, 1e20).loss - fit.E;
      const high = optimalAllocation(fit, 1e21).loss - fit.E;
      expect(high / low).toBeCloseTo(Math.pow(10, -frontierLossExponent(fit)), 10);
    }
    expect(frontierLossExponent(FITS.hoffmann)).toBeCloseTo(0.1535, 4);
  });

  it("pins the frontier slope and the share of reducible loss a tenfold budget keeps under both fits (step 5, card info, checkpoint 1)", () => {
    // Frontier power law card: Slope and "10× compute keeps".
    const hoffmann = frontierLossExponent(FITS.hoffmann);
    const epoch = frontierLossExponent(FITS.epoch);
    expect(hoffmann).toBeCloseTo(0.1535, 4);
    expect(epoch).toBeCloseTo(0.1783, 4);
    expect(Math.round(Math.pow(10, -hoffmann) * 100)).toBe(70);
    expect(Math.round(Math.pow(10, -epoch) * 100)).toBe(66);
    // Step 5 says the refit's line is steeper: it keeps less of L − E per tenfold budget.
    expect(epoch).toBeGreaterThan(hoffmann);
    // Halving the reducible loss takes about 90 times the compute under the printed fit.
    expect(Math.pow(0.5, -1 / hoffmann)).toBeGreaterThan(85);
    expect(Math.pow(0.5, -1 / hoffmann)).toBeLessThan(95);
  });

  it("keeps the frontier straight in log-log once the floor is removed, but not in raw loss (checkpoint 2)", () => {
    const fit = FITS.hoffmann;
    const reducible = (compute: number) => optimalAllocation(fit, compute).loss - fit.E;
    const raw = (compute: number) => optimalAllocation(fit, compute).loss;
    const slopeAfter = (f: (c: number) => number, c: number) => Math.log10(f(c * 10) / f(c));
    // Reducible loss: the same log-log slope at every decade.
    expect(slopeAfter(reducible, 1e20)).toBeCloseTo(slopeAfter(reducible, 1e24), 10);
    // Raw loss: the slope flattens toward the floor E.
    expect(slopeAfter(raw, 1e24)).toBeGreaterThan(slopeAfter(raw, 1e20));
    // Even at 1T parameters on 30T tokens the loss stays above E.
    expect(predictedLoss(fit, 1e12, 30e12)).toBeGreaterThan(fit.E);
  });

  it("drags Parameters along a held-compute curve: past the optimum tokens fall and loss rises (checkpoint 4)", () => {
    const fit = FITS.hoffmann;
    const compute = 5.376e20;
    const best = optimalAllocation(fit, compute);
    let previousLoss = best.loss;
    let previousTokens = best.tokens;
    for (const factor of [1.5, 2, 4, 8]) {
      const parameters = best.parameters * factor;
      const tokens = compute / (6 * parameters);
      expect(tokens).toBeLessThan(previousTokens);
      expect(predictedLoss(fit, parameters, tokens)).toBeGreaterThan(previousLoss);
      previousLoss = predictedLoss(fit, parameters, tokens);
      previousTokens = tokens;
    }
  });

  it("makes doubling tokens beat doubling parameters when the data term dominates, as at the default run (checkpoint 5)", () => {
    const fit = FITS.hoffmann;
    const terms = lossTerms(fit, 3.2e9, 28e9);
    expect(terms.data).toBeGreaterThan(terms.size);
    const doubleN = terms.size * (1 - Math.pow(2, -fit.alpha));
    const doubleD = terms.data * (1 - Math.pow(2, -fit.beta));
    expect(doubleD).toBeGreaterThan(doubleN);
    // At the optimum the two doublings are not equal in size, but the marginal returns match.
    const best = optimalAllocation(fit, 5.376e20);
    const atBest = lossTerms(fit, best.parameters, best.tokens);
    expect(fit.alpha * atBest.size).toBeCloseTo(fit.beta * atBest.data, 10);
  });

  it("spends a budget at 20 tokens per parameter", () => {
    const rule = ratioAllocation(FITS.hoffmann, 1.2e20);
    expect(rule.parameters / 1e9).toBeCloseTo(1, 6);
    expect(rule.tokens / 1e9).toBeCloseTo(20, 6);
  });

  it("formats counts and FLOPs", () => {
    expect(formatCount(3.2e9)).toBe("3.2B");
    expect(formatCount(28e9)).toBe("28B");
    expect(formatCount(1.4e12)).toBe("1.4T");
    expect(formatCount(1.38e9)).toBe("1.38B");
    expect(formatFlops(5.376e20)).toBe("5.4 × 10²⁰");
  });
});

/**
 * Quote locations: standard.mdx and plain.mdx "Going deeper" ("From dimensions to N") and card-info.ts "Scaling law controls".
 * Dimensions: the toy block of The transformer block lab (width 8, MLP 32, one layer); LLaMA 6.7B and 65.2B from Table 2 of
 * Touvron et al. 2023 (arXiv 2302.13971); GPT-2 small from its Hugging Face configuration (n_embd 768, n_layer 12, vocab_size 50257).
 */
describe("N from dimensions", () => {
  it("counts 12 d squared weights per block, 4 d squared in attention and 8 d squared in the MLP", () => {
    expect(blockWeights(8)).toBe(768);
    expect(blockWeights(8)).toBe(4 * 8 * 8 + 2 * 8 * 32);
    expect(denseParameters(1, 8)).toBe(12 * 64);
    // The transformer block lab states 840 for its toy block: 768 weights plus 72 biases and norm scales and shifts.
    expect(toyBlockExtras()).toBe(72);
    expect(denseParameters(1, 8) + toyBlockExtras()).toBe(840);
  });

  it("puts LLaMA's published shapes within a few percent under their published counts", () => {
    // standard.mdx and plain.mdx "From dimensions to N": 12 x 32 x 4,096^2 = 6.44B against 6.7B, and 12 x 80 x 8,192^2 = 64.4B against 65.2B.
    expect(denseParameters(32, 4096)).toBe(6442450944);
    expect(formatCount(denseParameters(32, 4096))).toBe("6.44B");
    expect(denseParameters(80, 8192)).toBe(64424509440);
    expect(formatCount(denseParameters(80, 8192))).toBe("64.4B");
    expect(Math.round((1 - denseParameters(32, 4096) / 6.7e9) * 100)).toBe(4);
    expect(Math.round((1 - denseParameters(80, 8192) / 65.2e9) * 100)).toBe(1);
  });

  it("shows the embedding table is a large share of a small model's blocks and a small share of a large one's", () => {
    // standard.mdx and plain.mdx "From dimensions to N", module.ts checkpoint question 8: GPT-2 small, 84.9M in the blocks and 38.6M in the token table.
    expect(denseParameters(12, 768)).toBe(84934656);
    expect(formatCount(denseParameters(12, 768))).toBe("84.9M");
    expect(embeddingWeights(50257, 768)).toBe(38597376);
    expect(formatCount(embeddingWeights(50257, 768))).toBe("38.6M");
    const share = (layers: number, width: number, vocabulary: number) =>
      embeddingWeights(vocabulary, width) / denseParameters(layers, width);
    expect(share(12, 768, 50257)).toBeGreaterThan(0.45);
    expect(share(32, 4096, 50257)).toBeLessThan(0.05);
    expect(share(12, 768, 50257)).toBeGreaterThan(share(32, 4096, 50257));
  });

  it("grows fourfold when width doubles and twofold when layers double (checkpoint question 7)", () => {
    expect(denseParameters(32, 2 * 4096) / denseParameters(32, 4096)).toBe(4);
    expect(denseParameters(2 * 32, 4096) / denseParameters(32, 4096)).toBe(2);
    // card-info.ts "Scaling law controls" controls: from the default 6.44B, double the width for 25.8B or the layers for 12.9B.
    expect(formatCount(denseParameters(32, 2 * 4096))).toBe("25.8B");
    expect(formatCount(denseParameters(2 * 32, 4096))).toBe("12.9B");
  });

  it("starts the calculator inside the Parameters slider and keeps the whole range finite", () => {
    expect(denseParameters(32, 4096) / 1e9).toBeGreaterThan(PARAMETER_RANGE.min);
    expect(denseParameters(32, 4096) / 1e9).toBeLessThan(PARAMETER_RANGE.max);
    expect(denseParameters(LAYER_RANGE.max, WIDTH_RANGE.max)).toBeLessThan(PARAMETER_RANGE.max * 1e9);
    expect(Number.isFinite(denseParameters(LAYER_RANGE.min, WIDTH_RANGE.min))).toBe(true);
  });

  it("snaps width onto the slider grid inside its range", () => {
    expect(clampWidth(4100)).toBe(4096);
    expect(clampWidth(1e9)).toBe(WIDTH_RANGE.max);
    expect(clampWidth(-1e9)).toBe(WIDTH_RANGE.min);
    expect(clampWidth(Number.NaN)).toBe(4096);
    expect(clampWidth(Infinity)).toBe(4096);
  });
});

describe("scaling-laws state, dimensions", () => {
  it("reads a version 1 payload, which had no dimensions, and fills in 32 layers of width 4,096", () => {
    expect(definition.stateVersion).toBe(2);
    expect(definition.hydrateState(JSON.stringify({ parameters: 3.2, data: 28, fit: "epoch", budget: "hold" }))).toEqual({
      parameters: 3.2,
      data: 28,
      fit: "epoch",
      budget: "hold",
      layers: 32,
      width: 4096,
    });
  });

  it("clamps the dimension controls", () => {
    expect(definition.hydrateState(JSON.stringify({ layers: 1e9, width: 1e9 }))).toMatchObject({ layers: 128, width: 16384 });
    expect(definition.hydrateState(JSON.stringify({ layers: -1e9, width: -1e9 }))).toMatchObject({ layers: 1, width: 64 });
    expect(definition.hydrateState(JSON.stringify({ layers: 12.6, width: "wide" }))).toMatchObject({ layers: 13, width: 4096 });
    expect(definition.hydrateState('{"layers":1e999,"width":1e999}')).toMatchObject({ layers: 32, width: 4096 });
  });
});

describe("scaling-laws state", () => {
  it("clamps and fills old payloads", () => {
    const old = definition.hydrateState(JSON.stringify({ parameters: 3.2, data: 28 }));
    expect(old).toMatchObject({ parameters: 3.2, data: 28, fit: "hoffmann", budget: "free" });
    const wild = definition.hydrateState(JSON.stringify({ parameters: 1e9, data: -4, fit: "kaplan", budget: 3 }));
    expect(wild).toMatchObject({ parameters: 1000, data: 1, fit: "hoffmann", budget: "free" });
    expect(definition.hydrateState("not json")).toEqual(definition.initialState);
  });
});
