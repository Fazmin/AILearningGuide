import { describe, expect, it } from "vitest";
import { TINY_VOCAB, tinyRandom } from "@app/module-sdk";
import {
  assignSeats,
  batchGradients,
  batchLoss,
  capacitySeats,
  denseSettings,
  EMBED,
  evaluate,
  initParams,
  INPUT,
  moeData,
  parameterCounts,
  routeCharacter,
  safeSettings,
  shuffledOrder,
  trainMoe,
  VOCAB,
  type MoeParams,
  type MoeSettings,
} from "./moe";

const BASE: MoeSettings = { experts: 4, topK: 1, balancing: "none", auxWeight: 0.01, biasStep: 0.001, capacity: 0, epochs: 30, seed: 1 };
const SEEDS = [1, 2, 3, 4, 5, 6] as const;
const run = (patch: Partial<MoeSettings>) => trainMoe({ ...BASE, ...patch });
const perSeed = (patch: Partial<MoeSettings>) => SEEDS.map((seed) => run({ ...patch, seed }).final);
const denseLoss = SEEDS.map((seed) => trainMoe(denseSettings({ epochs: 30, seed })).final.loss);

/** Parameters with every entry random, including the router's bias row, so every gradient is exercised. */
function randomParams(experts: number, topK: number, seed: number): MoeParams {
  const { params } = initParams(experts, topK, seed);
  const random = tinyRandom(seed + 100);
  for (const tensor of [params.embed, params.router, params.weights]) {
    for (let index = 0; index < tensor.length; index += 1) tensor[index] = (random() * 2 - 1) * 0.8;
  }
  return params;
}

describe("the backward pass", () => {
  const data = moeData();
  const order = shuffledOrder(data.count, tinyRandom(7));
  const tensors = ["embed", "router", "weights"] as const;

  it.each([
    { label: "top-1, no balancing", experts: 4, topK: 1, settings: { balancing: "none", auxWeight: 0, capacity: 0 } },
    { label: "top-1 with the auxiliary term", experts: 4, topK: 1, settings: { balancing: "aux", auxWeight: 0.7, capacity: 0 } },
    { label: "top-2 with the auxiliary term and a capacity cap", experts: 5, topK: 2, settings: { balancing: "aux", auxWeight: 0.3, capacity: 1 } },
    { label: "top-2 over 8 experts, aux weight 1", experts: 8, topK: 2, settings: { balancing: "aux", auxWeight: 1, capacity: 0 } },
  ] as const)("matches finite differences for E, R and W: $label", ({ experts, topK, settings }) => {
    const params = randomParams(experts, topK, 11);
    const start = 32;
    const analytic = batchGradients(params, data, order, start, 16, settings);
    // The cap must bind somewhere in the capped case, or that case would test nothing extra.
    if (settings.capacity > 0) expect(Array.from(analytic.kept)).toContain(0);
    const epsilon = 1e-6;
    let checked = 0;
    for (const name of tensors) {
      const tensor = params[name];
      const gradient = analytic[name];
      // Every entry with a non-negligible gradient, plus a spread of the rest.
      const indexes = Array.from({ length: tensor.length }, (_, index) => index).filter(
        (index) => Math.abs(gradient[index]) > 1e-4 || index % 7 === 0,
      );
      for (const index of indexes) {
        const held = tensor[index];
        tensor[index] = held + epsilon;
        const up = batchLoss(params, data, order, start, 16, settings).objective;
        tensor[index] = held - epsilon;
        const down = batchLoss(params, data, order, start, 16, settings).objective;
        tensor[index] = held;
        const numeric = (up - down) / (2 * epsilon);
        expect(gradient[index], `${name}[${index}]`).toBeCloseTo(numeric, 6);
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("gives the auxiliary term a gradient of its own, and none when it is off", () => {
    const params = randomParams(4, 1, 5);
    const off = batchGradients(params, data, order, 0, 16, { balancing: "none", auxWeight: 0, capacity: 0 });
    const on = batchGradients(params, data, order, 0, 16, { balancing: "aux", auxWeight: 0.5, capacity: 0 });
    expect(on.objective - on.loss).toBeCloseTo(0.5 * on.balance, 12);
    expect(off.objective).toBe(off.loss);
    expect(Array.from(on.router)).not.toEqual(Array.from(off.router));
    // The term reaches the router and, through it, the embedding, but never an expert.
    expect(Array.from(on.weights)).toEqual(Array.from(off.weights));
  });
});

describe("routing", () => {
  it("lets the bias change which experts are chosen and never a gate value", () => {
    const params = randomParams(4, 1, 3);
    const character = TINY_VOCAB.indexOf("e");
    const plain = routeCharacter(params, character);
    const preferred = plain.chosen[0];
    const other = (preferred + 1) % 4;
    params.bias[other] = plain.z[preferred] - plain.z[other] + 1;
    const nudged = routeCharacter(params, character);
    expect(nudged.chosen).toEqual([other]);
    expect(nudged.gates).toEqual(plain.gates);
    expect(nudged.z).toEqual(plain.z);
    // A bias shared by every expert moves no choice at all.
    params.bias.fill(0);
    const shifted = { ...params, bias: new Float64Array(4).fill(5) };
    expect(routeCharacter(shifted, character).chosen).toEqual(plain.chosen);
  });

  it("keeps every token's gate values identical with or without the bias, while the choices move", () => {
    const data = moeData();
    const order = Array.from({ length: 16 }, (_, index) => index);
    const params = randomParams(4, 2, 9);
    const settings = { balancing: "none", auxWeight: 0, capacity: 0 } as const;
    const before = batchGradients(params, data, order, 0, 16, settings);
    const flat = evaluate(params, 0);
    params.bias.set([3, -3, 0, 0]);
    const after = batchGradients(params, data, order, 0, 16, settings);
    expect(Array.from(after.chosen)).not.toEqual(Array.from(before.chosen));
    expect(Array.from(after.gates)).toEqual(Array.from(before.gates));
    expect(evaluate(params, 0).shares).not.toEqual(flat.shares);
    params.bias.fill(0);
    expect(evaluate(params, 0).shares).toEqual(flat.shares);
  });

  it("runs exactly k experts per token and gates them with the softmax of z over all experts", () => {
    const params = randomParams(6, 2, 4);
    const route = routeCharacter(params, TINY_VOCAB.indexOf("t"));
    expect(route.chosen).toHaveLength(2);
    expect(new Set(route.chosen).size).toBe(2);
    const exps = route.z.map((value) => Math.exp(value));
    const total = exps.reduce((sum, value) => sum + value, 0);
    route.gates.forEach((gate, e) => expect(gate).toBeCloseTo(exps[e] / total, 12));
    expect(route.gates[route.chosen[0]] + route.gates[route.chosen[1]]).toBeLessThan(1);
    expect(route.next.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
  });

  it("seats every first choice before any second choice, each in batch order", () => {
    // Tokens 0–7 pick E1 then E2; tokens 8–15 pick E2 then E1. With 8 seats each, every first
    // choice is seated and every second choice dropped, so no token is left without an expert.
    // Seating token by token instead would fill both experts with tokens 0–7 and strand 8–15.
    const chosen = Array.from({ length: 16 }, (_, t) => (t < 8 ? [0, 1] : [1, 0])).flat();
    const kept = assignSeats(chosen, new Uint8Array(32), 16, 2, new Float64Array([8, 8]));
    for (let t = 0; t < 16; t += 1) {
      expect(kept[t * 2]).toBe(1);
      expect(kept[t * 2 + 1]).toBe(0);
    }
    expect(capacitySeats(0.5, 16, 2, 2)).toBe(8);
    expect(capacitySeats(1.25, 16, 1, 4)).toBe(5);
    expect(capacitySeats(1.25, 13, 1, 4)).toBe(5);
    expect(capacitySeats(0, 16, 1, 4)).toBe(Number.POSITIVE_INFINITY);
  });

  it("never reroutes a capped token: the cap changes what is served, not what the router chose", () => {
    const params = randomParams(4, 1, 2);
    const open = evaluate(params, 0);
    const capped = evaluate(params, 1);
    expect(capped.shares).toEqual(open.shares);
    expect(open.dropped).toBe(0);
    expect(capped.dropped).toBeGreaterThan(0);
    expect(capped.unserved).toBe(capped.dropped);
    expect(capped.loss).toBeGreaterThan(open.loss);
  });

  it("scores a token that loses every expert at ln 30", () => {
    // Sixteen copies of one pair, t then h: every token picks the same expert, which has one seat.
    const t = TINY_VOCAB.indexOf("t");
    const h = TINY_VOCAB.indexOf("h");
    const data = {
      inputs: new Int32Array(16).fill(t),
      targets: new Int32Array(16).fill(h),
      corpus: new Int32Array(16),
      count: 16,
      corpusCounts: [16, 0, 0],
      characterCounts: Array.from({ length: VOCAB }, (_, c) => (c === t ? 16 : 0)),
    };
    const params = randomParams(4, 1, 8);
    const open = evaluate(params, 0, data);
    const capped = evaluate(params, 0.001, data);
    expect(capacitySeats(0.001, 16, 1, 4)).toBe(1);
    expect(capped.unserved).toBe(15 / 16);
    expect(capped.loss * 16).toBeCloseTo(open.loss + 15 * Math.log(30), 10);
  });
});

describe("runs", () => {
  it("is deterministic for a seed and differs between seeds", () => {
    const first = run({ balancing: "aux", auxWeight: 0.1, epochs: 5 });
    const again = run({ balancing: "aux", auxWeight: 0.1, epochs: 5 });
    expect(again.final.loss).toBe(first.final.loss);
    expect(Array.from(again.params.router)).toEqual(Array.from(first.params.router));
    expect(again.history.map((point) => point.shares)).toEqual(first.history.map((point) => point.shares));
    expect(run({ balancing: "aux", auxWeight: 0.1, epochs: 5, seed: 2 }).final.loss).not.toBe(first.final.loss);
  });

  it("records one evaluation per epoch boundary and 74 steps per epoch", () => {
    const result = run({ epochs: 3 });
    expect(result.history).toHaveLength(4);
    expect(result.biasHistory).toHaveLength(4);
    expect(result.steps).toBe(3 * 74);
    expect(result.pairs).toBe(1181);
    const { byCharacter: _character, byCorpus: _corpus, ...finalSummary } = result.final;
    expect(result.history[3]).toEqual(finalSummary);
  });

  it("counts total and active parameters from the stored arrays", () => {
    for (const [experts, topK, total, active] of [
      [1, 1, 275, 275],
      [2, 1, 430, 280],
      [4, 1, 740, 290],
      [4, 2, 740, 440],
      [8, 1, 1360, 310],
      [8, 2, 1360, 460],
    ] as const) {
      const counts = parameterCounts(experts, topK);
      expect([counts.total, counts.active]).toEqual([total, active]);
      const { params } = initParams(experts, topK, 1);
      expect(params.embed.length + params.router.length + params.weights.length).toBe(total);
      expect(params.embed.length).toBe(VOCAB * EMBED);
      expect(params.weights.length / experts).toBe(INPUT * VOCAB);
    }
  });

  it("starts the router's bias row at zero", () => {
    const { params } = initParams(4, 1, 1);
    expect(Array.from(params.router.subarray(EMBED * 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(params.router.subarray(0, EMBED * 4)).every((value) => value !== 0)).toBe(true);
  });

  it("clamps any request to a bounded run", () => {
    const wild = safeSettings({ experts: 1e9, topK: 1e9, balancing: "x" as never, auxWeight: -1, biasStep: Infinity, capacity: NaN, epochs: 1e9, seed: NaN });
    expect(wild).toEqual({ experts: 8, topK: 2, balancing: "none", auxWeight: 0, biasStep: 0, capacity: 0, epochs: 60, seed: 1 });
    expect(safeSettings({ ...BASE, experts: 1, topK: 2 }).topK).toBe(1);
  });
});

describe("what training does, seed by seed", () => {
  it("collapses onto one expert with no balancing at 2, 4 and 8 experts, every seed, at about the dense loss", () => {
    for (const experts of [2, 4, 8]) {
      perSeed({ experts }).forEach((final, index) => {
        expect(final.effective, `${experts} experts, seed ${index + 1}`).toBeLessThan(1.01);
        expect(Math.abs(final.loss - denseLoss[index]), `${experts} experts, seed ${index + 1}`).toBeLessThan(0.04);
      });
    }
  }, 60_000);

  it("still collapses at Switch's aux weight of 0.01 in five of six seeds", () => {
    const effective = perSeed({ balancing: "aux", auxWeight: 0.01 }).map((final) => final.effective);
    expect(effective.filter((value) => value < 1.01)).toHaveLength(5);
    expect(effective[2]).toBeCloseTo(1.61, 2);
  }, 30_000);

  it("balances at aux weight 0.1 and beats the dense baseline in every seed", () => {
    perSeed({ balancing: "aux", auxWeight: 0.1 }).forEach((final, index) => {
      expect(final.effective).toBeGreaterThan(3.5);
      expect(final.loss).toBeLessThan(denseLoss[index]);
    });
  }, 30_000);

  it("pays for aux weight 1 in loss: worse than 0.1 in every seed, worse than dense in five, with flatter gates", () => {
    const strong = perSeed({ balancing: "aux", auxWeight: 1 });
    const moderate = perSeed({ balancing: "aux", auxWeight: 0.1 });
    strong.forEach((final, index) => {
      expect(final.effective).toBeGreaterThan(3.3);
      expect(final.loss).toBeGreaterThan(moderate[index].loss);
      expect(final.meanGate).toBeLessThan(moderate[index].meanGate);
    });
    expect(strong.filter((final, index) => final.loss > denseLoss[index])).toHaveLength(5);
  }, 30_000);

  it("balances with a bias step of 0.03 and beats the dense baseline in every seed; 0.001 is too slow", () => {
    perSeed({ balancing: "bias", biasStep: 0.03 }).forEach((final, index) => {
      expect(final.effective).toBeGreaterThan(3.6);
      expect(final.loss).toBeLessThan(denseLoss[index]);
    });
    perSeed({ balancing: "bias", biasStep: 0.001 }).forEach((final) => expect(final.effective).toBeLessThan(2));
  }, 30_000);

  it("does worse with 8 balanced experts than with 4", () => {
    const four = perSeed({ balancing: "aux", auxWeight: 0.1 });
    const eight = perSeed({ experts: 8, balancing: "aux", auxWeight: 0.1 });
    expect(eight.filter((final, index) => final.loss > four[index].loss)).toHaveLength(5);
    const fourBias = perSeed({ balancing: "bias", biasStep: 0.03 });
    const eightBias = perSeed({ experts: 8, balancing: "bias", biasStep: 0.03 });
    expect(eightBias.filter((final, index) => final.loss > fourBias[index].loss)).toHaveLength(6);
    expect(eightBias.filter((final, index) => final.loss > denseLoss[index])).toHaveLength(5);
  }, 60_000);

  it("drops far more tokens under a 1.25 cap without balancing than with it", () => {
    const none = perSeed({ capacity: 1.25 }).map((final) => final.dropped);
    const aux = perSeed({ capacity: 1.25, balancing: "aux", auxWeight: 0.1 }).map((final) => final.dropped);
    const bias = perSeed({ capacity: 1.25, balancing: "bias", biasStep: 0.03 }).map((final) => final.dropped);
    expect(Math.min(...none)).toBeGreaterThan(0.39);
    // A fully collapsed router keeps 5 of every 16: 73 full batches drop 11 each, the last 13-pair batch drops 8.
    expect(Math.max(...none)).toBeCloseTo((73 * 11 + 8) / 1181, 12);
    expect(Math.max(...aux)).toBeLessThan(0.16);
    expect(Math.max(...bias)).toBeLessThan(0.09);
    none.forEach((value, index) => expect(value).toBeGreaterThan(aux[index]));
  }, 60_000);
});

describe("cost", () => {
  it("trains the slowest page, three runs at the largest settings, well inside a render", () => {
    const worst: MoeSettings = { experts: 8, topK: 2, balancing: "aux", auxWeight: 1, biasStep: 0.1, capacity: 2, epochs: 40, seed: 6 };
    const started = performance.now();
    trainMoe(worst);
    trainMoe({ ...worst, balancing: "none" });
    trainMoe(denseSettings(worst));
    // About 0.2 s on a quiet machine; the ceiling is generous so a busy machine does not flake.
    expect(performance.now() - started).toBeLessThan(2500);
  }, 30_000);
});
