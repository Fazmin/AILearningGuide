import { describe, expect, it } from "vitest";
import { checkpointQuestions, TINY_VOCAB, tinyRandom } from "@app/module-sdk";
import definition from "./module";
import {
  capacitySeats,
  EMBED,
  evaluate,
  initParams,
  INPUT,
  MIXTRAL_8X7B,
  moeData,
  parameterCounts,
  routeCharacter,
  trainMoe,
  VOCAB,
  type MoeSettings,
} from "./moe";

/**
 * Pins the factual claims in the checkpoint explanations. The exact figures the lessons quote are
 * pinned in quoted.test.ts.
 */
const questions = checkpointQuestions(definition);
const correct = (index: number) => questions[index].options[questions[index].answer];
const BASE: MoeSettings = { experts: 4, topK: 1, balancing: "none", auxWeight: 0.01, biasStep: 0.001, capacity: 0, epochs: 30, seed: 1 };
const SEEDS = [1, 2, 3, 4, 5, 6];
const run = (patch: Partial<MoeSettings>) => trainMoe({ ...BASE, ...patch });

describe("moe-training checkpoint claims", () => {
  it("question 1: a bias that overrides the router's ranking runs that expert, weighted by its own unbiased gate, and only k experts run", () => {
    const { params } = initParams(4, 1, 2);
    const random = tinyRandom(40);
    for (const tensor of [params.embed, params.router, params.weights]) {
      for (let index = 0; index < tensor.length; index += 1) tensor[index] = (random() * 2 - 1) * 0.8;
    }
    const character = TINY_VOCAB.indexOf("e");
    const plain = routeCharacter(params, character);
    const first = plain.chosen[0];
    const target = (first + 2) % 4;
    params.bias[target] = plain.z[first] - plain.z[target] + 0.5;
    const nudged = routeCharacter(params, character);
    expect(nudged.chosen).toEqual([target]);
    // The gate is the softmax of z, computed without the bias, and it is below 1.
    expect(nudged.gates[target]).toBe(plain.gates[target]);
    expect(nudged.gates[target]).toBeLessThan(1);
    // The prediction is that expert's scores times its gate, and nothing from the router's favourite.
    const h = [...Array.from(params.embed.subarray(character * EMBED, character * EMBED + EMBED)), 1];
    const logits = Array.from({ length: VOCAB }, (_, v) =>
      nudged.gates[target] * h.reduce((sum, value, i) => sum + value * params.weights[(target * INPUT + i) * VOCAB + v], 0),
    );
    const peak = Math.max(...logits);
    const total = logits.reduce((sum, value) => sum + Math.exp(value - peak), 0);
    logits.forEach((value, v) => expect(nudged.next[v]).toBeCloseTo(Math.exp(value - peak) / total, 12));
    for (const topK of [1, 2]) {
      const { params: wide } = initParams(6, topK, 3);
      expect(routeCharacter(wide, character).chosen).toHaveLength(topK);
    }
    expect(correct(0)).toMatch(/^E3 runs, scaled by its own gate/);
  });

  it("question 2: the split starts fairly even, from a zero router bias, and one expert takes over within an epoch", () => {
    const { params } = initParams(4, 1, 1);
    expect(Array.from(params.router.subarray(EMBED * 4))).toEqual([0, 0, 0, 0]);
    const collapse = run({});
    expect(collapse.history[0].effective).toBeCloseTo(3.66, 2);
    expect(collapse.history[1].effective).toBeLessThan(1.001);
    // In every seed the winner is made by training: none of them starts collapsed.
    for (const seed of SEEDS) {
      const history = run({ seed }).history;
      expect(history[0].effective).toBeGreaterThan(3.4);
      expect(history.at(-1)!.effective).toBeLessThan(1.001);
    }
    expect(correct(1)).toMatch(/^An expert that wins a few more tokens/);
  }, 30_000);

  it("question 3: aux weight 1 ends above 0.1 in every seed, with flatter gates; Switch's 0.01 still collapses", () => {
    for (const seed of SEEDS) {
      const strong = run({ balancing: "aux", auxWeight: 1, seed }).final;
      const moderate = run({ balancing: "aux", auxWeight: 0.1, seed }).final;
      expect(strong.loss).toBeGreaterThan(moderate.loss);
      expect(strong.meanGate).toBeLessThan(moderate.meanGate);
      expect(strong.effective).toBeGreaterThan(3.3);
    }
    expect(run({ balancing: "aux", auxWeight: 0.01 }).final.effective).toBeLessThan(1.001);
    // The bias method leaves the gates sharp.
    expect(run({ balancing: "bias", biasStep: 0.03 }).final.meanGate).toBeGreaterThan(run({ balancing: "aux", auxWeight: 1 }).final.meanGate);
    expect(correct(2)).toMatch(/^The balancing pull now outweighs/);
  }, 60_000);

  it("question 4: 5 seats for 16 tokens drops 11 of 16 when collapsed; seed 4 drops 69%, seed 1 40%; a cap never reroutes", () => {
    expect(capacitySeats(1.25, 16, 1, 4)).toBe(5);
    expect(Math.round((11 / 16) * 100)).toBe(69);
    expect(Math.round(run({ capacity: 1.25, seed: 4 }).final.dropped * 100)).toBe(69);
    expect(Math.round(run({ capacity: 1.25 }).final.dropped * 100)).toBe(40);
    const trained = run({ capacity: 1.25 }).params;
    expect(evaluate(trained, 1.25).shares).toEqual(evaluate(trained, 0).shares);
    expect(correct(3)).toMatch(/^About 69%, because that expert has 5 seats/);
  }, 30_000);

  it("question 5: each corpus's split is exactly the mix of its characters' routes, so it sits within a few hundredths of the overall split", () => {
    const data = moeData();
    for (const seed of SEEDS) {
      const final = run({ balancing: "aux", auxWeight: 0.1, seed }).final;
      // Character routes: every character sends all its choices to one expert at top-1.
      const route = final.byCharacter.map((row) => {
        const total = row.reduce((sum, value) => sum + value, 0);
        return row.map((value) => (total > 0 ? value / total : 0));
      });
      final.byCorpus.forEach((row, corpus) => {
        const counts = new Array<number>(VOCAB).fill(0);
        for (let pair = 0; pair < data.count; pair += 1) if (data.corpus[pair] === corpus) counts[data.inputs[pair]] += 1;
        const total = counts.reduce((sum, value) => sum + value, 0);
        row.forEach((value, e) => {
          const fromCharacters = counts.reduce((sum, count, character) => sum + count * route[character][e], 0);
          expect(value).toBeCloseTo(fromCharacters, 9);
          expect(Math.abs(value / total - final.shares[e])).toBeLessThan(0.05);
        });
        expect(total).toBe(data.corpusCounts[corpus]);
      });
    }
    expect(correct(4)).toMatch(/^Nothing: the router sees only the previous character/);
  }, 30_000);

  it("question 6: four more experts add 600 stored weights plus 20 router weights, 740 to 1,360, while active goes 290 to 310", () => {
    const four = parameterCounts(4, 1);
    const eight = parameterCounts(8, 1);
    expect(four.perExpert).toBe(150);
    expect(eight.total - four.total).toBe(4 * 150 + 20);
    expect([four.total, eight.total, four.active, eight.active]).toEqual([740, 1360, 290, 310]);
    expect([MIXTRAL_8X7B.total, MIXTRAL_8X7B.active]).toEqual([46.7e9, 12.9e9]);
    expect(correct(5)).toMatch(/^Total nearly doubles, while Active per token grows only by the router's extra weights/);
  });
});
