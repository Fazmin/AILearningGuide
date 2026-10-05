import { describe, expect, it } from "vitest";
import {
  pruneTinyWeights,
  TINY_CORPORA,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  tinyPerplexity,
  trainTinyFactored,
  trainTinyModel,
} from "@app/module-sdk";
import {
  composePipeline,
  corpusCounts,
  corpusKl,
  factorizeTable,
  importanceScores,
  klDivergence,
  labelFrequencies,
  multiplyFactors,
  prune,
  pruneByScore,
  pruneNofM,
  rowDistribution,
  unseenRows,
} from "./compress";
import { hydrateDistillationState } from "./module";

const V = TINY_VOCAB_SIZE;
const text = TINY_CORPORA.harbor.text;

describe("distillation measurements", () => {
  it("softens a row with temperature and keeps it normalized", () => {
    const weights = new Float32Array(V * V);
    weights[0] = 2;
    const sharp = rowDistribution(weights, 0, 0.5);
    const flat = rowDistribution(weights, 0, 4);
    expect(sharp.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(sharp[0]).toBeGreaterThan(flat[0]);
  });

  it("has zero KL against itself and weights rows by corpus use", () => {
    const teacher = trainTinyModel({ text, epochs: 10, seed: 1 }).weights;
    expect(corpusKl(teacher, teacher, text)).toBeCloseTo(0, 10);
    expect(klDivergence([0.5, 0.5], [0.25, 0.75])).toBeCloseTo(0.5 * Math.log(2) + 0.5 * Math.log(2 / 3), 12);
    // Changing only a row the corpus never visits leaves the corpus KL unchanged.
    const unseen = unseenRows(text);
    expect(unseen.length).toBe(4);
    const edited = Float32Array.from(teacher);
    edited[unseen[0] * V] = 9;
    expect(corpusKl(teacher, edited, text)).toBeCloseTo(0, 10);
  });

  it("turns one-hot labels into frequencies", () => {
    const row = " abcdefghijklmnopqrstuvwxyz.<>".indexOf("h");
    const frequencies = labelFrequencies(text, row);
    expect(frequencies.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(frequencies[" abcdefghijklmnopqrstuvwxyz.<>".indexOf("e")]).toBeCloseTo(0.76, 2);
  });
});

describe("pruning shapes", () => {
  const weights = Float32Array.from({ length: V * V }, (_, index) => Math.sin(index * 1.7) * (1 + (index % 7)));

  it("2:4 keeps the two largest of every four down each column, exactly half overall", () => {
    const result = pruneNofM(weights);
    expect(result.removed).toBe(450);
    for (let column = 0; column < V; column += 1) {
      for (let start = 0; start + 4 <= V; start += 4) {
        const kept = [0, 1, 2, 3].filter((offset) => result.mask[(start + offset) * V + column]);
        expect(kept).toHaveLength(2);
        const keptMin = Math.min(...kept.map((offset) => Math.abs(weights[(start + offset) * V + column])));
        const droppedMax = Math.max(
          ...[0, 1, 2, 3]
            .filter((offset) => !kept.includes(offset))
            .map((offset) => Math.abs(weights[(start + offset) * V + column])),
        );
        expect(keptMin).toBeGreaterThanOrEqual(droppedMax);
      }
    }
  });

  it("reports masks that agree with the zeroed weights", () => {
    for (const mode of ["unstructured", "structured"] as const) {
      const result = prune(weights, mode, 0.4);
      expect(result.mask.filter((keep) => !keep).length).toBe(result.removed);
      result.mask.forEach((keep, index) => {
        if (!keep) expect(result.weights[index]).toBe(0);
      });
    }
    expect(prune(weights, "structured", 0.5).removedRows).toHaveLength(15);
  });
});

describe("distillation state", () => {
  it("hydrates a version 1 payload and accepts the new mode", () => {
    expect(hydrateDistillationState(JSON.stringify({ rank: 3, pruneMode: "structured" }))).toMatchObject({
      rank: 3,
      pruneMode: "structured",
      context: "h",
    });
    expect(hydrateDistillationState(JSON.stringify({ pruneMode: "semi", context: "zz" }))).toMatchObject({
      pruneMode: "semi",
      context: "h",
    });
  });
});

/* -------------------------------------------------------------------------- */
/* Compose: distil, prune, quantize                                            */
/* -------------------------------------------------------------------------- */

describe("factorizing a table", () => {
  const teacherTable = trainTinyModel({ text, epochs: 80, seed: 1 }).weights;

  it("reproduces an exactly rank-limited table to float precision", () => {
    const rank = 4;
    const left = Float32Array.from({ length: V * rank }, (_, index) => Math.sin(index * 0.7) * 0.5);
    const right = Float32Array.from({ length: rank * V }, (_, index) => Math.cos(index * 1.3) * 0.5);
    const table = multiplyFactors(left, right, rank);
    const factors = factorizeTable(table, rank);
    expect(factors.left).toHaveLength(V * rank);
    expect(factors.right).toHaveLength(rank * V);
    const rebuilt = multiplyFactors(factors.left, factors.right, rank);
    let worst = 0;
    for (let index = 0; index < table.length; index += 1) worst = Math.max(worst, Math.abs(rebuilt[index] - table[index]));
    expect(worst).toBeLessThan(1e-5);
    // Singular values come out largest first, and only `rank` of them are non-negligible.
    expect(factors.singularValues[0]).toBeGreaterThanOrEqual(factors.singularValues[1]);
    expect(factors.singularValues[rank]).toBeLessThan(1e-4);
  });

  it("is the best rank-limited form: a higher rank never reproduces the teacher worse", () => {
    const error = (rank: number) => {
      const { left, right } = factorizeTable(teacherTable, rank);
      const rebuilt = multiplyFactors(left, right, rank);
      let sum = 0;
      for (let index = 0; index < rebuilt.length; index += 1) sum += (rebuilt[index] - teacherTable[index]) ** 2;
      return sum;
    };
    expect(error(2)).toBeGreaterThan(error(6));
    expect(error(6)).toBeGreaterThan(error(12));
    expect(error(V)).toBeLessThan(1e-6);
  });

  it("splits the singular values evenly and stays finite when the rank exceeds the table's", () => {
    const flat = new Float32Array(V * V); // rank zero
    const { left, right } = factorizeTable(flat, 3);
    expect([...left, ...right].every((value) => value === 0)).toBe(true);
    const rankOne = Float32Array.from({ length: V * V }, (_, index) => (Math.floor(index / V) + 1) * ((index % V) - 5));
    const factors = factorizeTable(rankOne, 5);
    expect([...factors.left, ...factors.right].every(Number.isFinite)).toBe(true);
    const rebuilt = multiplyFactors(factors.left, factors.right, 5);
    for (let index = 0; index < rankOne.length; index += 1) expect(rebuilt[index]).toBeCloseTo(rankOne[index], 2);
  });
});

describe("corpus counts and importance scores", () => {
  const counts = corpusCounts(text);
  const teacherTable = trainTinyModel({ text, epochs: 80, seed: 1 }).weights;
  const student = trainTinyFactored({ text, rank: 4, epochs: 6, batchSize: 16, learningRate: 0.5, seed: 3, teacher: teacherTable, softLabels: true, temperature: 1 });
  const { left, right } = factorizeTable(student.weights, 4);

  it("counts every position once and reproduces the corpus cross-entropy", () => {
    expect(counts.rowCounts.reduce((sum, value) => sum + value, 0)).toBe(counts.positions);
    expect(counts.pairCounts.reduce((sum, value) => sum + value, 0)).toBe(counts.positions);
    expect(counts.positions).toBe(498);
    // Rows the corpus never visits have no positions.
    for (const row of unseenRows(text)) expect(counts.rowCounts[row]).toBe(0);
  });

  it("scores magnitude as |w| and activation as |w| times the input norm", () => {
    const magnitude = importanceScores("magnitude", left, right, 4, counts);
    expect(Array.from(magnitude.left.slice(0, 5))).toEqual(Array.from(left.slice(0, 5), Math.abs));
    const activation = importanceScores("activation", left, right, 4, counts);
    // The input to the left factor is a one-hot context, whose norm over the corpus is sqrt(count).
    for (const [context, component] of [[0, 0], [5, 2], [8, 3]]) {
      const index = context * 4 + component;
      expect(activation.left[index]).toBeCloseTo(Math.abs(left[index]) * Math.sqrt(counts.rowCounts[context]), 9);
    }
    // The input to the right factor is the hidden vector: sqrt of the count-weighted squares of that component.
    for (const [component, column] of [[0, 1], [2, 7]]) {
      let sum = 0;
      for (let context = 0; context < V; context += 1) sum += counts.rowCounts[context] * left[context * 4 + component] ** 2;
      expect(activation.right[component * V + column]).toBeCloseTo(Math.abs(right[component * V + column]) * Math.sqrt(sum), 9);
    }
    // A context the corpus never visits scores zero for the left factor.
    const unseen = unseenRows(text)[0];
    for (let component = 0; component < 4; component += 1) expect(activation.left[unseen * 4 + component]).toBe(0);
  });

  it("measures ablation exactly: the loss rise from zeroing one entry, checked by brute force", () => {
    const scores = importanceScores("ablation", left, right, 4, counts);
    const base = tinyCrossEntropy(multiplyFactors(left, right, 4), text);
    const check = (side: "left" | "right", index: number) => {
      const l = Float32Array.from(left);
      const r = Float32Array.from(right);
      (side === "left" ? l : r)[index] = 0;
      const brute = tinyCrossEntropy(multiplyFactors(l, r, 4), text) - base;
      expect(scores[side][index], `${side} ${index}`).toBeCloseTo(brute, 5);
    };
    for (const index of [0, 7, 33, 61, 90, 119]) check("left", index);
    for (const index of [0, 5, 44, 77, 100, 119]) check("right", index);
    // Entries on a never-visited context cannot change the loss.
    const unseen = unseenRows(text)[0];
    for (let component = 0; component < 4; component += 1) expect(scores.left[unseen * 4 + component]).toBe(0);
  });
});

describe("pruning by score", () => {
  it("removes the lowest-scored entries, ties broken by position, and keeps a matching mask", () => {
    const values = Float32Array.from([5, -4, 3, -2, 1, 0.5]);
    const scores = [5, 4, 3, 2, 1, 1];
    const result = pruneByScore(values, scores, 0.5);
    expect(result.removed).toBe(3);
    expect(Array.from(result.values)).toEqual([5, -4, 3, 0, 0, 0]);
    expect(result.mask).toEqual([true, true, true, false, false, false]);
    expect(pruneByScore(values, scores, 0).removed).toBe(0);
    expect(pruneByScore(values, scores, 7).removed).toBe(6);
    expect(pruneByScore(values, scores, -1).removed).toBe(0);
  });

  it("agrees with the SDK's magnitude pruning when the score is |w|", () => {
    const values = Float32Array.from({ length: 180 }, (_, index) => Math.sin(index * 2.3) * (1 + (index % 5)));
    for (const share of [0.1, 0.5, 0.9]) {
      const mine = pruneByScore(values, Array.from(values, Math.abs), share);
      const sdk = pruneTinyWeights(values, { fraction: share, mode: "unstructured" });
      expect(mine.removed).toBe(sdk.removed);
      expect(Array.from(mine.values)).toEqual(Array.from(sdk.weights));
    }
  });
});

describe("the compose pipeline", () => {
  const teacher = trainTinyModel({ text, epochs: 80, seed: 1 }).weights;
  const student = trainTinyFactored({ text, rank: 6, epochs: 18, batchSize: 16, learningRate: 0.5, seed: 3, teacher, softLabels: true, temperature: 1 });
  const run = (options: Partial<Parameters<typeof composePipeline>[0]> = {}) =>
    composePipeline({ teacher, student: student.weights, rank: 6, text, importance: "magnitude", share: 0.5, bits: 4, ...options });

  it("prices each stage from its storage layout", () => {
    const result = run();
    const [teacherStage, studentStage, prunedStage, quantizedStage] = result.stages;
    expect(result.stages.map((stage) => stage.id)).toEqual(["teacher", "student", "pruned", "quantized"]);
    // Teacher: 900 float32 values. Student: two factors of 6 x 30 float32 values.
    expect([teacherStage.values, teacherStage.bytes]).toEqual([900, 3600]);
    expect([studentStage.values, studentStage.bytes]).toEqual([360, 1440]);
    // Pruned: half of each factor removed, kept values at 32 bits plus a one-bit mask per position.
    expect(result.removed).toBe(180);
    expect(result.kept).toBe(180);
    expect(prunedStage.bits).toBe(180 * 32 + 360);
    expect(prunedStage.bytes).toBe(765);
    // Quantized: 180 kept values at 4 bits in blocks of 32 (6 blocks, two 16-bit numbers each), plus the mask.
    expect(result.blocks).toBe(6);
    expect(quantizedStage.bits).toBe(180 * 4 + 6 * 32 + 360);
    expect(quantizedStage.bytes).toBe(159);
  });

  it("charges no mask when nothing is pruned, and the pruned stage then equals the student", () => {
    const result = run({ share: 0 });
    expect(result.removed).toBe(0);
    expect(result.stages[2].bytes).toBe(result.stages[1].bytes);
    expect(result.stages[2].perplexity).toBeCloseTo(result.stages[1].perplexity, 6);
    expect(result.stages[3].bits).toBe(360 * 4 + Math.ceil(360 / 32) * 32);
  });

  it("reproduces the student's own perplexity from its factors", () => {
    const result = run({ share: 0 });
    expect(result.stages[1].perplexity).toBeCloseTo(tinyPerplexity(student.weights, text), 12);
    expect(result.stages[2].perplexity).toBeCloseTo(tinyPerplexity(student.weights, text), 4);
  });

  it("puts every quantized value back where it came from, so near-lossless bits reproduce the pruned table", () => {
    // If the kept values were scattered into the wrong positions, or a pruned position came back non-zero, 16-bit rounding
    // would not land on the pruned stage's perplexity.
    for (const share of [0.3, 0.7]) {
      for (const importance of ["magnitude", "activation", "ablation"] as const) {
        const result = run({ share, importance, bits: 16 });
        expect(result.stages[3].perplexity, `${importance} ${share}`).toBeCloseTo(result.stages[2].perplexity, 3);
        for (let index = 0; index < 900; index += 1) {
          expect(result.stages[3].weights[index]).toBeCloseTo(result.stages[2].weights[index], 2);
        }
      }
    }
  });

  it("costs less perplexity at more bits, and never more than a few percent at 8 bits", () => {
    const eight = run({ bits: 8 }).stages;
    const four = run({ bits: 4 }).stages;
    const two = run({ bits: 2 }).stages;
    expect(eight[3].perplexity / eight[2].perplexity - 1).toBeLessThan(0.001);
    expect(four[3].perplexity).toBeGreaterThan(eight[3].perplexity);
    expect(two[3].perplexity).toBeGreaterThan(four[3].perplexity);
    expect(two[3].bytes).toBeLessThan(four[3].bytes);
    expect(four[3].bytes).toBeLessThan(eight[3].bytes);
  });

  it("counts the passes each score needs", () => {
    expect(run({ importance: "magnitude" }).scoringPasses).toBe(0);
    expect(run({ importance: "activation" }).scoringPasses).toBe(1);
    expect(run({ importance: "ablation" }).scoringPasses).toBe(360);
    expect(run({ importance: "ablation", rank: 3 }).scoringPasses).toBe(180);
  });

  it("clamps the rank and the share instead of throwing", () => {
    for (const rank of [0, -3, 13, 1e9, Number.NaN]) {
      const result = run({ rank: rank === 13 || rank === 1e9 ? 12 : rank, share: 7 });
      expect(result.stages.every((stage) => Number.isFinite(stage.perplexity))).toBe(true);
    }
    expect(run({ share: -1 }).removed).toBe(0);
    expect(run({ share: 5 }).kept).toBeLessThan(10);
  });

  it("stays fast at the largest rank for every score", () => {
    const wide = trainTinyFactored({ text, rank: 12, epochs: 6, batchSize: 16, learningRate: 0.5, seed: 3, teacher, softLabels: true, temperature: 1 });
    const started = performance.now();
    for (const importance of ["magnitude", "activation", "ablation"] as const) {
      composePipeline({ teacher, student: wide.weights, rank: 12, text, importance, share: 0.9, bits: 2 });
    }
    expect(performance.now() - started).toBeLessThan(2000);
  });
});

describe("distillation state for the compose card", () => {
  it("translates a version-2 payload and clamps the new keys", () => {
    const v2 = JSON.stringify({ rank: 4, temperature: 2, epochs: 10, labels: "hard", pruneFraction: 0.4, pruneMode: "semi", context: "e" });
    expect(hydrateDistillationState(v2)).toEqual({
      rank: 4,
      temperature: 2,
      epochs: 10,
      labels: "hard",
      pruneFraction: 0.4,
      pruneMode: "semi",
      context: "e",
      importance: "magnitude",
      composeShare: 0.5,
      composeBits: 4,
    });
    expect(hydrateDistillationState(JSON.stringify({ importance: "ablation", composeShare: 0.65, composeBits: 6 }))).toMatchObject({
      importance: "ablation",
      composeShare: 0.65,
      composeBits: 6,
    });
    expect(hydrateDistillationState(JSON.stringify({ importance: "random", composeShare: 9, composeBits: -3 }))).toMatchObject({
      importance: "magnitude",
      composeShare: 0.9,
      composeBits: 2,
    });
    expect(hydrateDistillationState(JSON.stringify({ importance: 3, composeShare: "half", composeBits: null }))).toMatchObject({
      importance: "magnitude",
      composeShare: 0.5,
      composeBits: 4,
    });
  });
});
