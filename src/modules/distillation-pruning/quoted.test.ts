import { describe, expect, it } from "vitest";
import {
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyPerplexity,
  trainTinyFactored,
  trainTinyModel,
} from "@app/module-sdk";
import {
  composePipeline,
  corpusCounts,
  corpusKl,
  labelFrequencies,
  prune,
  rowDistribution,
  unseenRows,
  type ImportanceScore,
  type PruneMode,
} from "./compress";
import { hydrateDistillationState } from "./module";

/**
 * Pins the measured values the distillation-pruning lesson (content/standard.mdx, content/plain.mdx), card-info.ts and
 * the Explore lab text quote. Mirrors Explore.tsx: the teacher is `trainTinyModel` on the harbor corpus for 80 epochs at
 * seed 1; each student is `trainTinyFactored` with batch 16, learning rate 0.5, seed 3, the teacher weights, soft labels
 * at the chosen temperature or hard labels; KL is `corpusKl` at T = 1; pruning is `prune(teacher.weights, mode, share)`
 * measured with `tinyPerplexity` on the harbor corpus.
 */
const text = TINY_CORPORA.harbor.text;
const V = TINY_VOCAB_SIZE;
const defaults = hydrateDistillationState("{}");
const teacher = trainTinyModel({ text, epochs: 80, seed: 1 });
const teacherPerplexity = tinyPerplexity(teacher.weights, text);

const students = (rank: number, temperature: number, epochs = Number(defaults.epochs)) => {
  const shared = { text, rank, epochs, batchSize: 16, learningRate: 0.5, seed: 3, teacher: teacher.weights };
  const soft = trainTinyFactored({ ...shared, softLabels: true, temperature });
  const hard = trainTinyFactored({ ...shared, softLabels: false });
  return {
    soft,
    hard,
    softKl: corpusKl(teacher.weights, soft.weights, text),
    hardKl: corpusKl(teacher.weights, hard.weights, text),
  };
};
const defaultRank = Number(defaults.rank);
const defaultTemperature = Number(defaults.temperature);
const pruned = (mode: PruneMode, share: number) => {
  const result = prune(teacher.weights, mode, share);
  return { ...result, perplexity: tinyPerplexity(result.weights, text) };
};
const round = (value: number, digits: number) => Number(value.toFixed(digits));
const percent = (value: number) => round(value * 100, 1);

/** Mirrors Explore's `unseenMass`: teacher probability on characters that never follow `character`, excluding the reserved `<` and `>`. */
const unseenMass = (character: string, temperature: number) => {
  const row = TINY_VOCAB.indexOf(character);
  const frequencies = labelFrequencies(text, row);
  return rowDistribution(teacher.weights, row, temperature).reduce(
    (sum, value, k) => (frequencies[k] === 0 && TINY_VOCAB[k] !== "<" && TINY_VOCAB[k] !== ">" ? sum + value : sum),
    0,
  );
};

describe("distillation-pruning quoted values", () => {
  it("matches the teacher and the lab defaults the lesson assumes", () => {
    // standard.mdx "What it is": 900-weight table, 80 epochs, loss 1.887, perplexity 6.600.
    expect(teacher.weights.length).toBe(900);
    expect(teacher.finalLoss).toBeCloseTo(1.887, 3);
    expect(teacherPerplexity).toBeCloseTo(6.6, 3);
    // "At rank 6 and T = 1" are the lab defaults (module.ts initialState).
    expect(defaults).toMatchObject({ rank: 6, temperature: 1, epochs: 18, labels: "soft", pruneMode: "unstructured", context: "h" });
    // Student size: "2 x r x 30 weights: 360 at rank 6" (standard.mdx, plain.mdx) and "rank r · 60r weights" (card-info.ts).
    for (const rank of [1, 6, 12]) expect(students(rank, 1).soft.parameters).toBe(60 * rank);
    expect(students(6, 1).soft.parameters).toBe(2 * 6 * V);
    expect(students(6, 1).soft.parameters).toBe(360);
    // card-info.ts: "Teacher 900 weights" and the badge's share of the teacher.
    expect(students(6, 1).soft.fullParameters).toBe(900);
  });

  it("skips the four context rows the corpus never visits", () => {
    // card-info.ts "Distillation" and standard.mdx "Where it breaks": four rows, (j, x, <, >).
    expect(unseenRows(text).map((row) => TINY_VOCAB[row])).toEqual(["j", "x", "<", ">"]);
  });

  it("matches the soft against hard student at rank 6 and T = 1", () => {
    // standard.mdx "How to play with it": soft loss 2.119 and KL 0.1348; one-hot loss 2.175 and KL 0.2577.
    // plain.mdx: KL 0.1348 against 0.2577. card-info.ts "Student controls": 0.1348 against 0.2577 at rank 6.
    const { soft, hard, softKl, hardKl } = students(defaultRank, defaultTemperature);
    expect(soft.finalLoss).toBeCloseTo(2.119, 3);
    expect(softKl).toBeCloseTo(0.1348, 4);
    expect(hard.finalLoss).toBeCloseTo(2.175, 3);
    expect(hardKl).toBeCloseTo(0.2577, 4);
    // standard.mdx / plain.mdx "What to notice": KL differs twofold ("twice as close").
    expect(Math.round(hardKl / softKl)).toBe(2);
  });

  it("matches the temperature sweep of KL to teacher at rank 6", () => {
    // standard.mdx and card-info.ts: KL 0.4037 at T = 0.5, 0.1348 at 1, 0.1340 at 1.5, 0.1407 at 2, 0.2193 at 4, 0.2737 at 6.
    // plain.mdx quotes 0.1348 at 1, 0.1340 at 1.5, 0.2193 at 4, 0.2737 at 6.
    const kl = (temperature: number) => students(defaultRank, temperature).softKl;
    expect(kl(0.5)).toBeCloseTo(0.4037, 4);
    expect(kl(1)).toBeCloseTo(0.1348, 4);
    expect(kl(1.5)).toBeCloseTo(0.134, 4);
    expect(kl(2)).toBeCloseTo(0.1407, 4);
    expect(kl(4)).toBeCloseTo(0.2193, 4);
    expect(kl(6)).toBeCloseTo(0.2737, 4);
    // standard.mdx / card-info.ts: "shallow optimum near 1 to 1.5" - the sweep minimum is at 1.5 and T = 1 is within 1% of it.
    const sweep = [0.5, 1, 1.5, 2, 3, 4, 5, 6].map((temperature) => [temperature, kl(temperature)] as const);
    const best = sweep.reduce((low, point) => (point[1] < low[1] ? point : low));
    expect(best[0]).toBe(1.5);
    expect(kl(1) / kl(1.5)).toBeLessThan(1.01);
    // card-info.ts compares the sweep "against the hard student's 0.2577": T = 4 (0.2193) is still closer to the
    // teacher than the hard student, T = 6 (0.2737) is not.
    const hardKl = students(defaultRank, 1).hardKl;
    expect(kl(4)).toBeLessThan(hardKl);
    expect(kl(6)).toBeGreaterThan(hardKl);
  });

  it("matches the rank sweep: soft labels win at every rank", () => {
    // standard.mdx / card-info.ts: 0.4200 against 0.4668 at rank 2, 0.1348 against 0.2577 at rank 6, 0.0722 against 0.1295 at rank 12 (all at T = 1).
    const at = (rank: number) => students(rank, 1);
    const rank2 = at(2);
    const rank12 = at(12);
    expect([rank2.softKl, rank2.hardKl].map((value) => round(value, 4))).toEqual([0.42, 0.4668]);
    expect([rank12.softKl, rank12.hardKl].map((value) => round(value, 4))).toEqual([0.0722, 0.1295]);
    for (let rank = 1; rank <= 12; rank += 1) {
      const { softKl, hardKl } = at(rank);
      expect(softKl, `rank ${rank}`).toBeLessThan(hardKl);
    }
    // standard.mdx "What to notice": rank 12 gets "nearly six times closer than rank 2".
    expect(rank2.softKl / rank12.softKl).toBeGreaterThan(5.5);
    expect(rank2.softKl / rank12.softKl).toBeLessThan(6);
    // plain.mdx: "rank 12 lands at 0.0722".
    expect(rank12.softKl).toBeCloseTo(0.0722, 4);
  });

  it("matches the unseen-mass figures after h", () => {
    // standard.mdx and card-info.ts: the teacher puts 4.8% on characters that never follow "h" at T = 1, 39.3% at T = 2, 64.6% at T = 4.
    // plain.mdx: 4.8% at temperature 1 growing to 39.3% at 2.
    expect([1, 2, 4].map((temperature) => percent(unseenMass("h", temperature)))).toEqual([4.8, 39.3, 64.6]);
    // standard.mdx step instruction: the mass grows as temperature moves from 0.5 up to 6.
    const masses = [0.5, 1, 1.5, 2, 4, 6].map((temperature) => unseenMass("h", temperature));
    expect([...masses].sort((a, b) => a - b)).toEqual(masses);
  });

  it("matches the teacher against the label frequencies", () => {
    const h = TINY_VOCAB.indexOf("h");
    const e = TINY_VOCAB.indexOf("e");
    // standard.mdx / plain.mdx / card-info.ts: after "h" the teacher says 76.6% "e" against 76.0% in the labels.
    expect(percent(rowDistribution(teacher.weights, h, 1)[e])).toBe(76.6);
    expect(percent(labelFrequencies(text, h)[e])).toBe(76);
  });

  it("matches the hard and soft student after a space", () => {
    // standard.mdx / plain.mdx / card-info.ts: with One-hot labels the hard student gives "a" 46.2% where the labels say 14.1%; the soft student gives 14.6%.
    const { soft, hard } = students(defaultRank, defaultTemperature);
    const space = TINY_VOCAB.indexOf(" ");
    const a = TINY_VOCAB.indexOf("a");
    expect(percent(rowDistribution(hard.weights, space, 1)[a])).toBe(46.2);
    expect(percent(labelFrequencies(text, space)[a])).toBe(14.1);
    expect(percent(rowDistribution(soft.weights, space, 1)[a])).toBe(14.6);
  });

  it("matches the unstructured pruning curve", () => {
    // standard.mdx: 6.60 unpruned, 6.61 at 30%, 6.65 at 50%, 6.84 at 70%, 8.07 at 90%.
    // plain.mdx: 6.65 at 50%. card-info.ts "Pruning": 6.61 at 30%; 6.65 at 50%, 6.84 at 70%, 8.07 at 90%.
    expect([0, 0.3, 0.5, 0.7, 0.9].map((share) => round(pruned("unstructured", share).perplexity, 2))).toEqual([
      6.6, 6.61, 6.65, 6.84, 8.07,
    ]);
    // card-info.ts "Pruning controls": "Unstructured zeroes the round(share x 900) smallest-magnitude weights".
    expect([0.3, 0.5, 0.9].map((share) => pruned("unstructured", share).removed)).toEqual([270, 450, 810]);
  });

  it("matches the 2:4 and structured points", () => {
    // standard.mdx / plain.mdx / card-info.ts: at 50%, 2:4 gives 7.15 and Structured rows 8.22.
    const semi = pruned("semi", 0.5);
    const structured = pruned("structured", 0.5);
    expect(round(semi.perplexity, 2)).toBe(7.15);
    expect(round(structured.perplexity, 2)).toBe(8.22);
    // card-info.ts "Pruning": at 30% structured already costs 6.95.
    expect(round(pruned("structured", 0.3).perplexity, 2)).toBe(6.95);
    // card-info.ts "Pruning controls": 2:4 is "Exactly 450 zeros"; glossary: "exactly 50% sparsity".
    expect(semi.removed).toBe(450);
    expect(semi.removedFraction).toBe(0.5);
    expect(semi.weights.filter((value) => value === 0).length).toBeGreaterThanOrEqual(450);
    // card-info.ts "Pruning controls": structured zeroes round(share x 30) rows, so 15 at 50%.
    expect(structured.removedRows).toHaveLength(15);
    expect(structured.removed).toBe(450);
    // standard.mdx / plain.mdx / card-info.ts: 2:4 sits between the unstructured and structured curves at 50%.
    const unstructured = pruned("unstructured", 0.5).perplexity;
    expect(unstructured).toBeLessThan(semi.perplexity);
    expect(semi.perplexity).toBeLessThan(structured.perplexity);
  });

  it("removes the never-used rows first at 30% unstructured", () => {
    // card-info.ts "Pruning" controls: at 30%, unstructured "removes the four never-used rows and the smallest logits first".
    const result = prune(teacher.weights, "unstructured", 0.3);
    for (const row of unseenRows(text)) {
      for (let k = 0; k < V; k += 1) {
        expect(teacher.weights[row * V + k]).toBe(0);
        expect(result.mask[row * V + k]).toBe(false);
      }
    }
  });
});

/** module.ts checkpoint questions 1-5 compare directions rather than quote numbers; these pin the directions. */
describe("distillation-pruning checkpoint claims", () => {
  it("question 1: the one-hot student ends further from the teacher while its training loss stays close", () => {
    const { soft, hard, softKl, hardKl } = students(defaultRank, defaultTemperature);
    expect(hardKl).toBeGreaterThan(softKl * 1.5);
    expect(Math.abs(hard.finalLoss - soft.finalLoss) / soft.finalLoss).toBeLessThan(0.05);
  });

  it("question 2: the teacher's rejected probabilities are a thin tail next to the corpus frequencies", () => {
    const row = TINY_VOCAB.indexOf("h");
    const frequencies = labelFrequencies(text, row);
    const teacherRow = rowDistribution(teacher.weights, row, 1);
    const e = TINY_VOCAB.indexOf("e");
    expect(Math.abs(teacherRow[e] - frequencies[e])).toBeLessThan(0.02);
    expect(unseenMass("h", 1)).toBeLessThan(0.1);
  });

  it("question 3: raising the temperature from 1 to 4 lifts the mass on unseen characters and ends further from the teacher", () => {
    expect(unseenMass("h", 4)).toBeGreaterThan(unseenMass("h", 1) * 5);
    expect(students(defaultRank, 4).softKl).toBeGreaterThan(students(defaultRank, 1).softKl);
  });

  it("questions 4 and 5: at 50% structured costs the most perplexity, 2:4 sits between, and unstructured the least", () => {
    const unstructured = pruned("unstructured", 0.5).perplexity;
    const semi = pruned("semi", 0.5).perplexity;
    const structured = pruned("structured", 0.5).perplexity;
    expect(unstructured).toBeLessThan(semi);
    expect(semi).toBeLessThan(structured);
  });
});


/* -------------------------------------------------------------------------- */
/* Compose the three steps                                                     */
/* -------------------------------------------------------------------------- */

describe("distillation-pruning: the compose pipeline", () => {
  // Mirrors Explore.tsx: the student selected on the Student controls card (rank 6, T = 1, 18 epochs, soft labels) goes
  // through composePipeline with the module's corpus counts.
  const counts = corpusCounts(text);
  const student = students(defaultRank, defaultTemperature).soft;
  const run = (importance: ImportanceScore, share: number, bits: number, studentWeights = student.weights, rank = defaultRank) =>
    composePipeline({ teacher: teacher.weights, student: studentWeights, rank, text, importance, share, bits, counts });
  const stages = (importance: ImportanceScore, share: number, bits: number) => run(importance, share, bits).stages;

  it("matches the default run quoted in the lesson and card-info", () => {
    // standard.mdx / plain.mdx / card-info.ts: at rank 6, 50% and 4 bits, magnitude: Teacher 3,600 B at 6.600, Distilled student
    // 1,440 B at 8.326, Pruned 765 B at 8.560, Quantized 159 B at 8.619: 22.6 times smaller for +30.6% perplexity.
    const result = run("magnitude", 0.5, 4);
    expect(result.stages.map((stage) => stage.bytes)).toEqual([3600, 1440, 765, 159]);
    expect(result.stages.map((stage) => round(stage.perplexity, 3))).toEqual([6.6, 8.326, 8.56, 8.619]);
    expect(round(result.stages[0].bytes / result.stages[3].bytes, 1)).toBe(22.6);
    expect(round((result.stages[3].perplexity / result.stages[0].perplexity - 1) * 100, 1)).toBe(30.6);
    // card-info.ts: 180 zeros out of 360 values cost 360 mask bits; kept 180 of 360.
    expect([result.kept, result.removed, result.blocks]).toEqual([180, 180, 6]);
    expect(result.stages[2].bits).toBe(180 * 32 + 360);
    // The teacher stage is the same 6.600 as every other card, and the student is the lab's rank-6 student.
    expect(result.stages[0].perplexity).toBeCloseTo(teacherPerplexity, 12);
    expect(result.stages[1].perplexity).toBeCloseTo(tinyPerplexity(student.weights, text), 12);
  });

  it("prices each stage separately", () => {
    // standard.mdx / plain.mdx / card-info.ts / checkpoint question 6 (4-bit run): distilling saves 2,160 B for +1.73 perplexity,
    // pruning 675 B for +0.23, quantizing 606 B for +0.06.
    const [teacherStage, studentStage, prunedStage, quantizedStage] = stages("magnitude", 0.5, 4);
    expect([teacherStage.bytes - studentStage.bytes, studentStage.bytes - prunedStage.bytes, prunedStage.bytes - quantizedStage.bytes]).toEqual([2160, 675, 606]);
    const added = [studentStage.perplexity - teacherStage.perplexity, prunedStage.perplexity - studentStage.perplexity, quantizedStage.perplexity - prunedStage.perplexity];
    expect(added.map((value) => round(value, 2))).toEqual([1.73, 0.23, 0.06]);
    // Quantizing gives up the least perplexity per byte saved, then pruning, then distilling.
    const perPpl = [2160 / added[0], 675 / added[1], 606 / added[2]];
    expect(perPpl[2]).toBeGreaterThan(perPpl[1]);
    expect(perPpl[1]).toBeGreaterThan(perPpl[0]);
    // card-info.ts: "At 2 bits the quantize stage costs +1.34" (50% magnitude), more than the pruning stage costs.
    const twoBit = stages("magnitude", 0.5, 2);
    expect(round(twoBit[3].perplexity - twoBit[2].perplexity, 2)).toBe(1.34);
    expect(twoBit[3].perplexity - twoBit[2].perplexity).toBeGreaterThan(added[1]);
    // standard.mdx / plain.mdx Going deeper: the same 2-bit stage on the full student, no pruning, costs +1.76.
    const unpruned = stages("magnitude", 0, 2);
    expect(round(unpruned[3].perplexity - unpruned[1].perplexity, 2)).toBe(1.76);
  });

  it("compares the three importance scores at 50% and 70% (checkpoint question 7)", () => {
    // standard.mdx / plain.mdx / card-info.ts: at 70%, after pruning: Magnitude 9.682, Activation-weighted 9.999, Measured ablation 9.480.
    const afterPrune = (share: number) =>
      Object.fromEntries((["magnitude", "activation", "ablation"] as const).map((kind) => [kind, stages(kind, share, 4)[2].perplexity]));
    const seventy = afterPrune(0.7);
    expect([seventy.magnitude, seventy.activation, seventy.ablation].map((value) => round(value, 3))).toEqual([9.682, 9.999, 9.48]);
    expect(seventy.ablation).toBeLessThan(seventy.magnitude);
    expect(seventy.magnitude).toBeLessThan(seventy.activation);
    // The same order holds at 50%: 8.560, 8.571, 8.510.
    const fifty = afterPrune(0.5);
    expect([fifty.magnitude, fifty.activation, fifty.ablation].map((value) => round(value, 3))).toEqual([8.56, 8.571, 8.51]);
    expect(fifty.ablation).toBeLessThan(fifty.magnitude);
    expect(fifty.magnitude).toBeLessThan(fifty.activation);
    // The magnitude score is the SDK's rule: the same pruned perplexity as pruning the factors with pruneTinyWeights is covered in compress.test.ts.
  });

  it("counts the passes each score needs (checkpoint question 8)", () => {
    // standard.mdx / plain.mdx / card-info.ts: 0, 1, and 360 passes at rank 6, which is 60 x rank.
    expect((["magnitude", "activation", "ablation"] as const).map((kind) => run(kind, 0.5, 4).scoringPasses)).toEqual([0, 1, 360]);
    expect(360).toBe(60 * defaultRank);
  });

  it("starts the card on the defaults the lesson walks through", () => {
    expect(defaults).toMatchObject({ importance: "magnitude", composeShare: 0.5, composeBits: 4 });
  });
});
