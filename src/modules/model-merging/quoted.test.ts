import { describe, expect, it } from "vitest";
import { bigramPairs, TINY_CORPORA, TINY_VOCAB, TINY_VOCAB_SIZE, tinyCrossEntropy, trainTinyModel } from "@app/module-sdk";
import { BASE_EPOCHS, buildTaskVectors, lossesFor, lossesOf } from "./lab";
import {
  addToBase,
  cosine,
  mergeTaskVectors,
  norm,
  pickCoordinates,
  signStats,
  slerp,
  trimTopK,
  type MergeMethod,
  type MergeParams,
} from "./merge";

/**
 * Pins the measured values the model-merging lesson and card text quote
 * (module.ts, content/standard.mdx, content/plain.mdx, card-info.ts).
 *
 * Everything is computed through lab.ts and merge.ts, the code the lab runs: one harbor base trained for 48
 * epochs (seed 1), two full fine-tunes from it (recipes seed 3, proverbs seed 4, 20 epochs each, learning rate
 * 0.6), and the lab's default merge settings. The LoRA source swaps in rank-4, alpha-8 adapters.
 */
const V = TINY_VOCAB_SIZE;
/** The lab's initial state: Blend t 50%, Scale lambda 1.00, Keep top 20%, Drop rate p 0.5, mask #4. */
const DEFAULTS: MergeParams = { mix: 0.5, lambda: 1, density: 0.2, dropRate: 0.5, maskSeed: 4 };
const round = (value: number, digits = 3) => Number(value.toFixed(digits));

const full = buildTaskVectors("full", 20, 20);
const lora = buildTaskVectors("lora", 20, 20);
const at = (method: MergeMethod, params: Partial<MergeParams> = {}, vectors = full) =>
  lossesFor(vectors, method, { ...DEFAULTS, ...params });

describe("model-merging quoted values", () => {
  it("matches the base and the two specialists", () => {
    // standard.mdx:13-16, card-info.ts:12: the 48-epoch base scores 2.553 on recipes and 2.684 on Proverbs;
    // the recipe specialist reaches 2.123 and the proverb specialist 2.169 (20 epochs each)
    expect(round(full.base.recipes)).toBe(2.553);
    expect(round(full.base.proverbs)).toBe(2.684);
    expect(round(full.specialistA.recipes)).toBe(2.123);
    expect(round(full.specialistB.proverbs)).toBe(2.169);
    // standard.mdx:13, plain.mdx:13: a 900-weight table
    expect(full.taskA).toHaveLength(900);
    // standard.mdx:13,18-20, plain.mdx:17-19, card-info.ts:15: this lab's base trains 48 epochs, not the 60-epoch Harbor
    // table of Fine-tuning & transfer and LoRA adapters, which reads 2.556 on recipes
    expect(BASE_EPOCHS).toBe(48);
    const sixtyEpochBase = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 60, seed: 1 });
    expect(round(tinyCrossEntropy(sixtyEpochBase.weights, TINY_CORPORA.recipes.text))).toBe(2.556);
    expect(round(full.base.recipes)).not.toBe(2.556);
    // standard.mdx:90, card-info.ts:94: "eight-to-ten-sentence corpora"
    const sentences = (text: string) => text.split(".").filter((part) => part.trim()).length;
    expect(sentences(TINY_CORPORA.recipes.text)).toBe(8);
    expect(sentences(TINY_CORPORA.proverbs.text)).toBe(10);
  });

  it("matches the task-vector statistics at 20 epochs each", () => {
    const stats = signStats(full.taskA, full.taskB);
    // standard.mdx:53-54, plain.mdx:55-56, card-info.ts:22: cosine 0.34; 720 of 900 overlap; 96 sign conflicts (13%)
    expect(round(cosine(full.taskA, full.taskB), 2)).toBe(0.34);
    expect(stats.overlap).toBe(720);
    expect(stats.conflicts).toBe(96);
    expect(Math.round((stats.conflicts / stats.overlap) * 100)).toBe(13);
    // card-info.ts:25: "87% of those entries agree in sign"
    expect(Math.round((1 - stats.conflicts / stats.overlap) * 100)).toBe(87);
    // card-info.ts:25: both corpora use the same 24 context characters, so both vectors touch the same 720 entries
    const contexts = (text: string) => new Set(Array.from(bigramPairs(text).inputs, (id) => TINY_VOCAB[id]));
    const recipes = contexts(TINY_CORPORA.recipes.text);
    const proverbs = contexts(TINY_CORPORA.proverbs.text);
    expect(recipes.size).toBe(24);
    expect([...recipes].every((character) => proverbs.has(character))).toBe(true);
    expect(stats.overlap).toBe(24 * V);
    expect(stats.onlyA + stats.onlyB).toBe(0);
    // standard.mdx:93 ("The strip shows 10 of 900 coordinates") and card-info.ts:43: five conflicts then five agreements
    expect(pickCoordinates(full.taskA, full.taskB, 5)).toHaveLength(10);
  });

  it("matches the Linear blend at t = 0.5: 2.257 and 2.330, gaps +0.134 and +0.161, interference -0.037 and -0.055", () => {
    const linear = at("linear");
    // standard.mdx:57-59, plain.mdx:58-59, card-info.ts:86
    expect(round(linear.recipes)).toBe(2.257);
    expect(round(linear.proverbs)).toBe(2.33);
    expect(round(linear.recipes - full.specialistA.recipes)).toBe(0.134);
    expect(round(linear.proverbs - full.specialistB.proverbs)).toBe(0.161);
    // standard.mdx:59, plain.mdx:62, card-info.ts:86
    expect(round(linear.interferenceA)).toBe(-0.037);
    expect(round(linear.interferenceB)).toBe(-0.055);
    // module.ts:120: "the average's proverb vector slightly helps recipes"
    expect(linear.interferenceA).toBeLessThan(0);
    // standard.mdx:70-72, plain.mdx:71-72: the recipe vector at half strength, alone, scores 2.294;
    // adding half the proverb vector improves that to 2.257
    const zero = new Float32Array(full.taskA.length);
    const aloneHalf = mergeTaskVectors("linear", full.taskA, zero, DEFAULTS)!;
    expect(round(lossesOf(addToBase(full.baseWeights, aloneHalf.vector)).recipes)).toBe(2.294);
    expect(linear.recipes).toBeLessThan(lossesOf(addToBase(full.baseWeights, aloneHalf.vector)).recipes);
    // standard.mdx:34: "At λ = 0.5 it is the average" (the lab formula's detail line says the same)
    const half = mergeTaskVectors("task", full.taskA, full.taskB, { ...DEFAULTS, lambda: 0.5 })!.vector;
    const average = mergeTaskVectors("linear", full.taskA, full.taskB, DEFAULTS)!.vector;
    expect(Array.from(half)).toEqual(Array.from(average));
  });

  it("matches Task arithmetic at lambda 1: 2.279 and 2.258, interference +0.156 and +0.089", () => {
    const summed = at("task", { lambda: 1 });
    // standard.mdx:61-62 and :73-74, plain.mdx:63-64 and :73, card-info.ts:86
    expect(round(summed.recipes)).toBe(2.279);
    expect(round(summed.proverbs)).toBe(2.258);
    expect(round(summed.interferenceA)).toBe(0.156);
    expect(round(summed.interferenceB)).toBe(0.089);
    // module.ts:120: "the full sum at scale 1 is where interference shows"
    expect(summed.interferenceA).toBeGreaterThan(0);
    expect(summed.interferenceB).toBeGreaterThan(0);
  });

  it("matches the lambda sweep: both losses fall until about 0.7 to 0.8 (2.218 and 2.241 at 0.8), then both rise", () => {
    // The dashed path on the plane: lambda from 0 to 1.5 in steps of 0.1 (Explore's `paths`)
    const path = Array.from({ length: 16 }, (_, step) =>
      lossesOf(addToBase(full.baseWeights, mergeTaskVectors("task", full.taskA, full.taskB, { ...DEFAULTS, lambda: step / 10 })!.vector)),
    );
    // standard.mdx:75-76 ("2.218 and 2.241 at 0.8")
    expect(round(path[8].recipes)).toBe(2.218);
    expect(round(path[8].proverbs)).toBe(2.241);
    // standard.mdx:75-76, plain.mdx:74-75, card-info.ts:90: the minimum sits at 0.7 to 0.8, then both rise
    const argmin = (losses: number[]) => losses.indexOf(Math.min(...losses));
    expect(argmin(path.map((point) => point.recipes))).toBe(7);
    expect(argmin(path.map((point) => point.proverbs))).toBe(8);
    for (let step = 1; step <= 7; step += 1) {
      expect(path[step].recipes).toBeLessThan(path[step - 1].recipes);
      expect(path[step].proverbs).toBeLessThan(path[step - 1].proverbs);
    }
    for (let step = 9; step < path.length; step += 1) {
      expect(path[step].recipes).toBeGreaterThan(path[step - 1].recipes);
      expect(path[step].proverbs).toBeGreaterThan(path[step - 1].proverbs);
    }
  });

  it("matches TIES: the h to a column and the 20% trim", () => {
    const ties = at("ties");
    const average = at("linear");
    const index = TINY_VOCAB.indexOf("h") * V + TINY_VOCAB.indexOf("a");
    // standard.mdx:77-78, plain.mdx:76-77: in the h to a column recipes wrote -0.89 and proverbs +1.18; the average
    // keeps +0.14 and TIES keeps +1.18
    expect(round(full.taskA[index], 2)).toBe(-0.89);
    expect(round(full.taskB[index], 2)).toBe(1.18);
    expect(round(average.result!.vector[index], 2)).toBe(0.14);
    expect(round(ties.result!.vector[index], 2)).toBe(1.18);
    expect(ties.result!.elected[index]).toBe(1);
    // standard.mdx:78-79: at 20% kept it lands at 2.273 and 2.280
    expect(round(ties.recipes)).toBe(2.273);
    expect(round(ties.proverbs)).toBe(2.28);
    // card-info.ts:58: at Keep top 20%, 38 of the 96 conflicts survive trimming in both vectors; in the other 58 at least
    // one side was trimmed away first
    const trimmedA = trimTopK(full.taskA, 0.2);
    const trimmedB = trimTopK(full.taskB, 0.2);
    let conflicts = 0;
    let bothKept = 0;
    for (let entry = 0; entry < full.taskA.length; entry += 1) {
      const conflict =
        Math.abs(full.taskA[entry]) > 1e-6 &&
        Math.abs(full.taskB[entry]) > 1e-6 &&
        Math.sign(full.taskA[entry]) !== Math.sign(full.taskB[entry]);
      if (!conflict) continue;
      conflicts += 1;
      if (trimmedA[entry] !== 0 && trimmedB[entry] !== 0) bothKept += 1;
    }
    expect(conflicts).toBe(96);
    expect(bothKept).toBe(38);
    expect(conflicts - bothKept).toBe(58);
  });

  it("matches DARE: 2.762 and 2.602 at p = 0.5, 5.967 and 5.661 at p = 0.9, and the spread over five masks", () => {
    // standard.mdx:80-81, plain.mdx:78-79: the default mask (#4) at p = 0.5
    const half = at("dare", { dropRate: 0.5, maskSeed: 4 });
    expect(round(half.recipes)).toBe(2.762);
    expect(round(half.proverbs)).toBe(2.602);
    // standard.mdx:81, plain.mdx:78-79 ("nearly 6")
    const heavy = at("dare", { dropRate: 0.9, maskSeed: 4 });
    expect(round(heavy.recipes)).toBe(5.967);
    expect(round(heavy.proverbs)).toBe(5.661);
    // card-info.ts:59 and plain.mdx:78-79: at p = 0.5 each survivor is doubled
    expect(1 / (1 - 0.5)).toBe(2);
    // card-info.ts:95: "at p = 0.5, five masks put recipe loss anywhere from 2.50 to 3.11" (masks #1 to #5;
    // the lab's default is #4)
    const spread = [1, 2, 3, 4, 5].map((maskSeed) => at("dare", { dropRate: 0.5, maskSeed }).recipes);
    expect(round(Math.min(...spread), 2)).toBe(2.5);
    expect(round(Math.max(...spread), 2)).toBe(3.11);
    // NOTE (card-info.ts:95): "New random mask" starts from #4 and steps to #5, #6, #7, #8. Those five masks (#4 to #8)
    // give 2.50 to 2.76 here, so the 3.11 end of the quoted range is mask #1 and is only reachable from a shared link
    // or by cycling past #999. The figure reproduces; the path a learner clicks through to see it does not.
    // standard.mdx:81-82, card-info.ts:59 and plain.mdx:78-79: DARE does not survive this model: it is the worst method here
    const others = (["linear", "task", "ties", "slerp"] as MergeMethod[]).map((method) => at(method).recipes);
    for (const value of others) expect(half.recipes).toBeGreaterThan(value);
  });

  it("matches SLERP: 2.226 and 2.284, 69.9 degrees, and a midpoint of length 10.32 against the chord's 8.46", () => {
    const arc = at("slerp");
    const chord = at("linear");
    // standard.mdx:83-84, plain.mdx:80-81
    expect(round(arc.recipes)).toBe(2.226);
    expect(round(arc.proverbs)).toBe(2.284);
    expect(arc.recipes).toBeLessThan(chord.recipes);
    expect(arc.proverbs).toBeLessThan(chord.proverbs);
    // standard.mdx:83-84: "The vectors are 69.9 degrees apart" (the badge's angle)
    expect(round(arc.result!.angle, 1)).toBe(69.9);
    expect(round(slerp(full.taskA, full.taskB, 0.5)!.angle, 1)).toBe(69.9);
    // standard.mdx:84: "the chord's midpoint is only 8.46 long while the arc's is 10.32" (the ‖τ_merged‖ badge)
    expect(round(norm(chord.result!.vector), 2)).toBe(8.46);
    expect(round(norm(arc.result!.vector), 2)).toBe(10.32);
    // card-info.ts:63: on full checkpoints (nearly parallel with a shared base) SLERP is almost a linear blend:
    // its midpoint is under 3% longer, where for the task vectors it is over 20% longer
    const checkpointA = addToBase(full.baseWeights, full.taskA);
    const checkpointB = addToBase(full.baseWeights, full.taskB);
    const checkpointArc = slerp(checkpointA, checkpointB, 0.5)!;
    const checkpointChord = mergeTaskVectors("linear", checkpointA, checkpointB, DEFAULTS)!;
    expect(norm(checkpointArc.vector) / norm(checkpointChord.vector)).toBeLessThan(1.03);
    expect(norm(arc.result!.vector) / norm(chord.result!.vector)).toBeGreaterThan(1.2);
  });

  it("matches the LoRA source: 331 of 720 conflicts (46%), cosine 0.19", () => {
    // standard.mdx:85, card-info.ts:22
    const stats = signStats(lora.taskA, lora.taskB);
    expect(stats.overlap).toBe(720);
    expect(stats.conflicts).toBe(331);
    expect(Math.round((stats.conflicts / stats.overlap) * 100)).toBe(46);
    expect(round(cosine(lora.taskA, lora.taskB), 2)).toBe(0.19);
    // standard.mdx:85 ("LoRA task vectors conflict far more")
    expect(stats.conflicts / stats.overlap).toBeGreaterThan(3 * (96 / 720));
  });

  it("matches the row claims for unused contexts and the row-sum claim", () => {
    // card-info.ts:17: j, q, x, z and the brackets receive no gradient, so those rows stay at zero, for LoRA too
    const zeroRows = (vector: Float32Array) =>
      [...TINY_VOCAB].filter((_, row) => Array.from({ length: V }, (__, k) => vector[row * V + k]).every((value) => value === 0)).join("");
    for (const vectors of [full.taskA, full.taskB, lora.taskA, lora.taskB]) expect(zeroRows(vectors)).toBe("jqxz<>");
    // card-info.ts:31: full fine-tunes never add a constant to a row (each row's gradient sums to zero), but LoRA products do
    const rowSums = (vector: Float32Array) =>
      Array.from({ length: V }, (_, row) => Array.from({ length: V }, (__, k) => vector[row * V + k]).reduce((sum, value) => sum + value, 0));
    for (const vector of [full.taskA, full.taskB]) expect(Math.max(...rowSums(vector).map(Math.abs))).toBeLessThan(1e-4);
    for (const vector of [lora.taskA, lora.taskB]) expect(Math.max(...rowSums(vector).map(Math.abs))).toBeGreaterThan(0.1);
    // card-info.ts:27: an untrained fine-tune is a zero task vector
    expect(norm(buildTaskVectors("full", 0, 20).taskA)).toBe(0);
  });

  it("leaves every merge short of the two-separate-models corner", () => {
    // card-info.ts:91: "No merge reaches the `two separate models` cross"
    const cornerReached = (recipes: number, proverbs: number) =>
      recipes <= full.specialistA.recipes && proverbs <= full.specialistB.proverbs;
    for (const method of ["linear", "task", "ties", "dare", "slerp"] as MergeMethod[]) {
      const outcome = at(method);
      expect(cornerReached(outcome.recipes, outcome.proverbs)).toBe(false);
    }
    for (let step = 0; step <= 10; step += 1) {
      const point = at("linear", { mix: step / 10 });
      expect(cornerReached(point.recipes, point.proverbs)).toBe(false);
    }
    for (let step = 0; step <= 15; step += 1) {
      const point = at("task", { lambda: step / 10 });
      expect(cornerReached(point.recipes, point.proverbs)).toBe(false);
    }
    // standard.mdx:57-59 and plain.mdx:58-60: the 50/50 average is worse than each specialist on both tasks
    const average = at("linear");
    expect(average.recipes).toBeGreaterThan(full.specialistA.recipes);
    expect(average.proverbs).toBeGreaterThan(full.specialistB.proverbs);
  });
});
