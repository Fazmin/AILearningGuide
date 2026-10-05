import { describe, expect, it } from "vitest";
import {
  GGUF_FORMATS,
  TINY_CORPORA,
  TINY_RESERVED,
  TINY_VOCAB_SIZE,
  UNIFORM_CROSS_ENTROPY,
  createTinyWeights,
  encodeTinyText,
  pruneTinyWeights,
  quantizeTinyWeights,
  sampleTinyText,
  scheduleFactor,
  tinyCrossEntropy,
  tinyNextDistribution,
  tinyPerplexity,
  tinyRowDivergence,
  tinySequenceLogProb,
  trainTinyFactored,
  trainTinyDpo,
  trainTinyLora,
  trainTinyModel,
} from "./tiny-lm";

const harbor = TINY_CORPORA.harbor.text;
const recipes = TINY_CORPORA.recipes.text;

describe("tiny corpus encoding", () => {
  it("keeps only vocabulary characters and collapses whitespace", () => {
    const ids = encodeTinyText("The  FOG\n  settles!");
    expect(ids.every((id) => id >= 0 && id < TINY_VOCAB_SIZE)).toBe(true);
    expect(ids.filter((id, index) => id === 0 && ids[index + 1] === 0)).toHaveLength(0);
  });
});

describe("training a tiny language model", () => {
  it("beats a uniform predictor and reports a finite loss", () => {
    const run = trainTinyModel({ text: harbor, epochs: 6, seed: 1 });
    expect(run.diverged).toBe(false);
    expect(run.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY * 0.75);
    expect(tinyPerplexity(run.weights, harbor)).toBeLessThan(TINY_VOCAB_SIZE);
  });

  it("is reproducible from its seed and responsive to it", () => {
    const first = trainTinyModel({ text: harbor, epochs: 3, seed: 4 });
    const repeat = trainTinyModel({ text: harbor, epochs: 3, seed: 4 });
    const other = trainTinyModel({ text: harbor, epochs: 3, seed: 9 });
    expect(Array.from(first.weights)).toEqual(Array.from(repeat.weights));
    expect(Array.from(first.weights)).not.toEqual(Array.from(other.weights));
  });

  it("improves monotonically with more epochs on the training text", () => {
    const short = trainTinyModel({ text: harbor, epochs: 2, seed: 2 });
    const long = trainTinyModel({ text: harbor, epochs: 12, seed: 2 });
    expect(long.finalLoss).toBeLessThan(short.finalLoss);
  });

  it("records checkpoints in order with the final weights last", () => {
    const run = trainTinyModel({ text: harbor, epochs: 4, checkpoints: 5, seed: 6 });
    expect(run.checkpoints.length).toBeGreaterThanOrEqual(5);
    expect(run.checkpoints[0].step).toBe(0);
    expect(run.checkpoints.at(-1)?.step).toBe(run.steps);
    expect(run.checkpoints.at(-1)?.loss).toBeCloseTo(run.finalLoss, 10);
  });

  it("diverges at an absurd learning rate and survives it with clipping", () => {
    const unstable = trainTinyModel({
      text: harbor,
      epochs: 6,
      learningRate: 900,
      batchSize: 4,
      seed: 3,
    });
    const clipped = trainTinyModel({
      text: harbor,
      epochs: 6,
      learningRate: 900,
      batchSize: 4,
      clipNorm: 0.5,
      seed: 3,
    });
    expect(unstable.finalLoss).toBeGreaterThan(clipped.finalLoss);
    expect(clipped.diverged).toBe(false);
  });

  it("leaves frozen rows untouched", () => {
    const trainableRows = Array.from({ length: TINY_VOCAB_SIZE }, (_, row) => row < 4);
    const run = trainTinyModel({ text: harbor, epochs: 4, trainableRows, seed: 8 });
    for (let row = 4; row < TINY_VOCAB_SIZE; row += 1) {
      for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
        expect(run.weights[row * TINY_VOCAB_SIZE + k]).toBe(0);
      }
    }
    expect(run.updatedParameters).toBeLessThanOrEqual(4 * TINY_VOCAB_SIZE);
  });

  it("produces a normalized next-token distribution and readable samples", () => {
    const run = trainTinyModel({ text: harbor, epochs: 10, seed: 5 });
    const distribution = tinyNextDistribution(run.weights, "the fo");
    const total = distribution.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
    expect(sampleTinyText(run.weights, { length: 40, seed: 2 })).toHaveLength(40);
  });

  it("suppresses reserved control symbols when sampling unless asked", () => {
    // An untrained model is uniform, so it would emit reserved symbols often.
    const untrained = createTinyWeights();
    const suppressed = sampleTinyText(untrained, { length: 400, seed: 3 });
    expect(suppressed).toHaveLength(400);
    for (const character of TINY_RESERVED) {
      expect(suppressed).not.toContain(character);
    }
    const permitted = sampleTinyText(untrained, { length: 400, seed: 3, allowReserved: true });
    expect(
      Array.from(TINY_RESERVED).some((character) => permitted.includes(character)),
    ).toBe(true);
  });

  it("never mutates the weights it was given as a starting point", () => {
    const base = trainTinyModel({ text: harbor, epochs: 6, seed: 1 });
    const copy = Float32Array.from(base.weights);
    trainTinyModel({ text: recipes, epochs: 6, init: base.weights, seed: 2 });
    trainTinyLora({ base: base.weights, text: recipes, rank: 3, epochs: 6, seed: 2 });
    trainTinyDpo({
      reference: base.weights,
      pairs: [{ prompt: "the ", chosen: "harbor", rejected: "zqxjvk" }],
      steps: 5,
    });
    expect(Array.from(base.weights)).toEqual(Array.from(copy));
  });
});

describe("learning-rate schedules against a real run", () => {
  // The claim lab 19 is built on: decay is what makes a large peak rate usable.
  it("loses to a constant rate when the peak is small", () => {
    const shared = { text: harbor, epochs: 25, batchSize: 16, seed: 1 } as const;
    const constant = trainTinyModel({ ...shared, learningRate: 0.6, schedule: "constant" });
    const cosine = trainTinyModel({ ...shared, learningRate: 0.6, schedule: "cosine" });
    expect(constant.finalLoss).toBeLessThan(cosine.finalLoss);
  });

  it("wins decisively when the peak is large enough to overshoot", () => {
    const shared = { text: harbor, epochs: 25, batchSize: 16, seed: 1 } as const;
    const constant = trainTinyModel({ ...shared, learningRate: 30, schedule: "constant" });
    const cosine = trainTinyModel({ ...shared, learningRate: 30, schedule: "cosine" });
    expect(cosine.finalLoss).toBeLessThan(constant.finalLoss);
    expect(constant.finalLoss).toBeGreaterThan(cosine.finalLoss * 1.3);
  });

  it("keeps gradient norms bounded, so this model overshoots rather than exploding", () => {
    const calm = trainTinyModel({ text: harbor, epochs: 25, learningRate: 0.6, seed: 1 });
    const wild = trainTinyModel({ text: harbor, epochs: 25, learningRate: 200, seed: 1 });
    expect(calm.peakGradientNorm).toBeLessThan(2);
    expect(wild.peakGradientNorm).toBeLessThan(2);
    expect(wild.finalLoss).toBeGreaterThan(UNIFORM_CROSS_ENTROPY);
  });

  it("recovers a noisy high-rate run when gradients are clipped", () => {
    const shared = { text: harbor, epochs: 25, batchSize: 2, learningRate: 30, seed: 1 } as const;
    const unclipped = trainTinyModel(shared);
    const clipped = trainTinyModel({ ...shared, clipNorm: 0.5 });
    expect(clipped.finalLoss).toBeLessThan(unclipped.finalLoss);
  });
});

describe("learning-rate schedules", () => {
  it("stays inside the unit range and shapes the run as named", () => {
    for (const schedule of ["constant", "cosine", "warmup-decay", "one-cycle"] as const) {
      for (const progress of [0, 0.25, 0.5, 0.75, 1]) {
        const factor = scheduleFactor(schedule, progress);
        expect(factor).toBeGreaterThan(0);
        expect(factor).toBeLessThanOrEqual(1);
      }
    }
    expect(scheduleFactor("cosine", 0)).toBeGreaterThan(scheduleFactor("cosine", 1));
    expect(scheduleFactor("warmup-decay", 0.02)).toBeLessThan(
      scheduleFactor("warmup-decay", 0.15),
    );
    expect(scheduleFactor("one-cycle", 0)).toBeLessThan(scheduleFactor("one-cycle", 0.15));
  });
});

describe("transfer and forgetting", () => {
  const pretrained = trainTinyModel({ text: harbor, epochs: 60, seed: 1 });

  it("adapts to a new corpus while losing ground on the original", () => {
    const before = tinyCrossEntropy(pretrained.weights, harbor);
    const adapted = trainTinyModel({
      text: recipes,
      epochs: 40,
      init: pretrained.weights,
      seed: 1,
    });
    expect(tinyCrossEntropy(adapted.weights, recipes)).toBeLessThan(
      tinyCrossEntropy(pretrained.weights, recipes),
    );
    expect(tinyCrossEntropy(adapted.weights, harbor)).toBeGreaterThan(before);
  });

  it("forgets monotonically as adaptation continues", () => {
    const run = trainTinyModel({
      text: recipes,
      epochs: 40,
      init: pretrained.weights,
      seed: 1,
      checkpoints: 6,
    });
    const original = run.checkpoints.map((point) =>
      tinyCrossEntropy(point.weights, harbor),
    );
    const target = run.checkpoints.map((point) => tinyCrossEntropy(point.weights, recipes));
    for (let index = 1; index < original.length; index += 1) {
      expect(original[index]).toBeGreaterThan(original[index - 1]);
      expect(target[index]).toBeLessThan(target[index - 1]);
    }
  });

  it("beats training the same budget from scratch early in adaptation", () => {
    const transferred = trainTinyModel({
      text: recipes,
      epochs: 6,
      init: pretrained.weights,
      seed: 1,
    });
    const scratch = trainTinyModel({ text: recipes, epochs: 6, seed: 1 });
    expect(transferred.finalLoss).toBeLessThan(scratch.finalLoss);
  });
});

describe("low-rank adaptation", () => {
  it("trains far fewer parameters and still lowers the target loss", () => {
    const base = trainTinyModel({ text: harbor, epochs: 12, seed: 1 });
    const lora = trainTinyLora({
      base: base.weights,
      text: recipes,
      rank: 4,
      epochs: 12,
      seed: 2,
    });
    expect(lora.trainableParameters).toBe(2 * 4 * TINY_VOCAB_SIZE);
    expect(lora.trainableParameters).toBeLessThan(lora.fullParameters);
    expect(lora.finalLoss).toBeLessThan(tinyCrossEntropy(base.weights, recipes));
  });

  it("moves the zero-initialized factor first", () => {
    // B is zero and A is random, so dL/dB is proportional to A and non-zero
    // while dL/dA is proportional to B and therefore exactly zero on step one.
    const base = trainTinyModel({ text: harbor, epochs: 20, seed: 1 });
    const shared = { base: base.weights, text: recipes, rank: 4, alpha: 8, seed: 5 } as const;
    const before = trainTinyLora({ ...shared, epochs: 0 });
    const after = trainTinyLora({ ...shared, epochs: 1, batchSize: 1000 });
    const moved = (left: Float32Array, right: Float32Array) =>
      Math.max(...Array.from(left, (value, index) => Math.abs(value - right[index])));
    expect(Array.from(before.up).every((value) => value === 0)).toBe(true);
    expect(Array.from(before.down).some((value) => value !== 0)).toBe(true);
    expect(moved(after.up, before.up)).toBeGreaterThan(0);
    expect(moved(after.down, before.down)).toBe(0);
  });

  it("changes nothing before training and merges exactly", () => {
    const base = trainTinyModel({ text: harbor, epochs: 6, seed: 1 });
    const untrained = trainTinyLora({
      base: base.weights,
      text: recipes,
      rank: 3,
      epochs: 0,
      seed: 2,
    });
    expect(Array.from(untrained.delta).every((value) => value === 0)).toBe(true);
    const trained = trainTinyLora({
      base: base.weights,
      text: recipes,
      rank: 3,
      epochs: 8,
      seed: 2,
    });
    trained.merged.forEach((value, index) => {
      expect(value).toBeCloseTo(base.weights[index] + trained.delta[index], 5);
    });
  });

  it("fits the target text better at a higher rank", () => {
    const base = trainTinyModel({ text: harbor, epochs: 10, seed: 1 });
    const lowRank = trainTinyLora({ base: base.weights, text: recipes, rank: 1, epochs: 10, seed: 4 });
    const highRank = trainTinyLora({ base: base.weights, text: recipes, rank: 8, epochs: 10, seed: 4 });
    expect(highRank.finalLoss).toBeLessThan(lowRank.finalLoss);
  });
});

describe("direct preference optimization", () => {
  // Equal-length completions, so raw log-probability is a fair comparison too.
  const pairs = [
    { prompt: "the ", chosen: "harbor", rejected: "zqxjvk" },
    { prompt: "boil the ", chosen: "pasta", rejected: "qxjvz" },
  ];

  it("raises the implicit reward margin and separates the pairs", () => {
    const reference = trainTinyModel({ text: `${harbor} ${recipes}`, epochs: 10, seed: 1 });
    const run = trainTinyDpo({ reference: reference.weights, pairs, steps: 30, beta: 0.4 });
    expect(run.history.at(-1)!.margin).toBeGreaterThan(run.history[0].margin);
    expect(run.history.at(-1)!.loss).toBeLessThan(run.history[0].loss);
    expect(run.accuracy).toBe(1);
    for (const pair of pairs) {
      expect(tinySequenceLogProb(run.weights, pair.prompt, pair.chosen)).toBeGreaterThan(
        tinySequenceLogProb(run.weights, pair.prompt, pair.rejected),
      );
    }
  });

  it("scores the margin, not the raw length-sensitive log-probability", () => {
    const reference = trainTinyModel({ text: harbor, epochs: 10, seed: 1 });
    const long = { prompt: "the ", chosen: "harbor freezes over", rejected: "fog" };
    const run = trainTinyDpo({ reference: reference.weights, pairs: [long], steps: 30 });
    // The preferred completion is six times longer, so it still scores lower in
    // absolute log-probability while its implicit reward is clearly ahead.
    expect(tinySequenceLogProb(run.weights, long.prompt, long.chosen)).toBeLessThan(
      tinySequenceLogProb(run.weights, long.prompt, long.rejected),
    );
    expect(run.finalMargin).toBeGreaterThan(0);
    expect(run.accuracy).toBe(1);
  });

  it("starts at the reference model and drifts away from it", () => {
    const reference = trainTinyModel({ text: harbor, epochs: 8, seed: 1 });
    const run = trainTinyDpo({ reference: reference.weights, pairs, steps: 20 });
    expect(run.history[0].drift).toBeCloseTo(0, 10);
    expect(run.finalDrift).toBeGreaterThan(0);
    expect(tinyRowDivergence(reference.weights, reference.weights)).toBeCloseTo(0, 12);
  });
});

describe("quantization", () => {
  // A well-fit model has spread-out logits, so rounding error actually costs
  // something. An underfit model is almost immune to quantization.
  const trained = trainTinyModel({ text: harbor, epochs: 80, seed: 1 });

  it("loses less precision with more bits", () => {
    const coarse = quantizeTinyWeights(trained.weights, { bits: 2, scope: "tensor" });
    const fine = quantizeTinyWeights(trained.weights, { bits: 8, scope: "tensor" });
    expect(fine.rmsError).toBeLessThan(coarse.rmsError);
    expect(fine.bytesPerWeight).toBeGreaterThan(1);
    expect(coarse.levels).toBe(4);
  });

  it("loses less precision with finer scaling at the same bit width", () => {
    const perTensor = quantizeTinyWeights(trained.weights, { bits: 4, scope: "tensor" });
    const perRow = quantizeTinyWeights(trained.weights, { bits: 4, scope: "row" });
    const perGroup = quantizeTinyWeights(trained.weights, {
      bits: 4,
      scope: "group",
      groupSize: 8,
    });
    expect(perRow.rmsError).toBeLessThan(perTensor.rmsError);
    expect(perGroup.rmsError).toBeLessThan(perRow.rmsError);
    expect(perGroup.bytesPerWeight).toBeGreaterThan(perTensor.bytesPerWeight);
  });

  it("keeps 8-bit perplexity close to the original and 2-bit measurably worse", () => {
    const original = tinyPerplexity(trained.weights, harbor);
    const eightBit = quantizeTinyWeights(trained.weights, { bits: 8, scope: "group", groupSize: 8 });
    const twoBit = quantizeTinyWeights(trained.weights, { bits: 2, scope: "tensor" });
    expect(tinyPerplexity(eightBit.weights, harbor)).toBeCloseTo(original, 1);
    expect(tinyPerplexity(twoBit.weights, harbor)).toBeGreaterThan(original * 1.05);
  });

  it("describes every published format with a usable configuration", () => {
    for (const format of GGUF_FORMATS) {
      const result = quantizeTinyWeights(trained.weights, {
        bits: format.bits,
        scope: format.scope,
        groupSize: format.groupSize,
      });
      expect(Number.isFinite(result.rmsError)).toBe(true);
      expect(result.totalBytes).toBeGreaterThan(0);
    }
  });
});

describe("pruning", () => {
  const trained = trainTinyModel({ text: harbor, epochs: 12, seed: 1 });

  it("removes the requested share and degrades the fit", () => {
    const light = pruneTinyWeights(trained.weights, { fraction: 0.2, mode: "unstructured" });
    const heavy = pruneTinyWeights(trained.weights, { fraction: 0.8, mode: "unstructured" });
    expect(light.removed).toBeLessThan(heavy.removed);
    expect(tinyCrossEntropy(heavy.weights, harbor)).toBeGreaterThan(
      tinyCrossEntropy(light.weights, harbor),
    );
  });

  it("removes whole context rows when structured", () => {
    const structured = pruneTinyWeights(trained.weights, { fraction: 0.25, mode: "structured" });
    expect(structured.removedRows.length).toBe(Math.round(0.25 * TINY_VOCAB_SIZE));
    for (const row of structured.removedRows) {
      for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
        expect(structured.weights[row * TINY_VOCAB_SIZE + k]).toBe(0);
      }
    }
  });

  it("changes nothing at zero pruning", () => {
    const none = pruneTinyWeights(trained.weights, { fraction: 0, mode: "unstructured" });
    expect(Array.from(none.weights)).toEqual(Array.from(trained.weights));
  });
});

describe("distillation", () => {
  const teacher = trainTinyModel({ text: harbor, epochs: 80, seed: 1 });
  const student = (extra: Partial<Parameters<typeof trainTinyFactored>[0]>) =>
    trainTinyFactored({
      teacher: teacher.weights,
      text: harbor,
      rank: 6,
      epochs: 18,
      batchSize: 16,
      learningRate: 0.5,
      seed: 3,
      ...extra,
    });

  it("stays closer to the teacher with soft labels than with one-hot labels", () => {
    const soft = student({ softLabels: true, temperature: 1 });
    const hard = student({ softLabels: false });
    expect(soft.parameters).toBeLessThan(soft.fullParameters);
    expect(soft.teacherDivergence).toBeLessThan(hard.teacherDivergence);
    expect(soft.finalLoss).toBeLessThan(hard.finalLoss);
  });

  it("does not benefit from tempering an already uncertain teacher", () => {
    // Temperature exists to expose the tail of a confident teacher. A bigram
    // model is not confident, so flattening it only distorts the target.
    const neutral = student({ softLabels: true, temperature: 1 });
    for (const temperature of [0.5, 2, 4]) {
      expect(student({ softLabels: true, temperature }).teacherDivergence).toBeGreaterThan(
        neutral.teacherDivergence,
      );
    }
  });

  it("pays off most when the student is much smaller than the teacher", () => {
    const smallSoft = student({ rank: 2, softLabels: true, temperature: 1 });
    const smallHard = student({ rank: 2, softLabels: false });
    const largeSoft = student({ rank: 12, softLabels: true, temperature: 1 });
    const largeHard = student({ rank: 12, softLabels: false });
    expect(smallSoft.teacherDivergence).toBeLessThan(smallHard.teacherDivergence);
    expect(largeSoft.teacherDivergence).toBeGreaterThan(largeHard.teacherDivergence);
  });

  it("gives a bigger student more capacity and reports no teacher when none is given", () => {
    const small = student({ rank: 2, softLabels: true, temperature: 1 });
    const large = student({ rank: 12, softLabels: true, temperature: 1 });
    expect(large.parameters).toBeGreaterThan(small.parameters);
    expect(large.teacherDivergence).toBeLessThan(small.teacherDivergence);
    const alone = trainTinyFactored({ text: harbor, rank: 6, epochs: 8, seed: 3 });
    expect(Number.isNaN(alone.teacherDivergence)).toBe(true);
    expect(alone.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY);
  });

  it("fits better than the full weight table at this data scale", () => {
    // Counterintuitive but measured: every example updates the shared r×30 output
    // factor plus one input row (31r weights), while the full table only updates
    // the row it just saw.
    const factored = trainTinyFactored({
      text: harbor,
      rank: 8,
      epochs: 20,
      batchSize: 16,
      learningRate: 0.5,
      seed: 9,
    });
    const full = trainTinyModel({
      text: harbor,
      epochs: 20,
      batchSize: 16,
      learningRate: 0.5,
      seed: 9,
    });
    expect(factored.parameters).toBeLessThan(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE);
    expect(factored.finalLoss).toBeLessThan(full.finalLoss);
  });
});

/**
 * `trainTinyFactored` gained schedule, clipping and weight decay after other labs had pinned
 * numbers from it. The defaults must leave every one of those results exactly as it was.
 */
describe("factored training options", () => {
  /** FNV-1a over the IEEE bits of every value, so one flipped bit anywhere changes the digest. */
  const digest = (values: ArrayLike<number>) => {
    const holder = new Float64Array(1);
    const halves = new Uint32Array(holder.buffer);
    let hash = 0x811c9dc5;
    for (let index = 0; index < values.length; index += 1) {
      holder[0] = values[index];
      for (const half of halves) {
        hash ^= half;
        hash = Math.imul(hash, 0x01000193) >>> 0;
      }
    }
    return hash >>> 0;
  };
  const fingerprint = (run: ReturnType<typeof trainTinyFactored>) => ({
    weights: digest(run.weights),
    loss: digest([run.finalLoss]),
    history: digest(
      run.history.flatMap((point) => [
        point.step,
        point.progress,
        point.loss,
        point.learningRate,
        point.gradientNorm,
      ]),
    ),
    checkpoints: digest(
      run.checkpoints.flatMap((point) => [point.step, point.progress, point.loss, ...point.weights]),
    ),
    parameters: run.parameters,
  });

  const teacher = trainTinyModel({ text: harbor, epochs: 80, seed: 1 });
  const proverbs = TINY_CORPORA.proverbs.text;
  /** Digests recorded from the implementation before the new options existed. */
  const baseline = [
    {
      name: "the train-tiny-lm default",
      options: { text: harbor, rank: 8, epochs: 20, batchSize: 16, learningRate: 0.5, seed: 9, checkpoints: 21 },
      expected: { weights: 3138085817, loss: 187015788, history: 132716335, checkpoints: 2550807257, parameters: 480 },
    },
    {
      name: "rank 1 on recipes",
      options: { text: recipes, rank: 1, epochs: 30, batchSize: 16, learningRate: 0.5, seed: 9, checkpoints: 31 },
      expected: { weights: 590104584, loss: 776007628, history: 65279457, checkpoints: 4110494339, parameters: 60 },
    },
    {
      name: "a small batch and a high rate",
      options: { text: proverbs, rank: 12, epochs: 7, batchSize: 5, learningRate: 1.5, seed: 4 },
      expected: { weights: 3262411485, loss: 772517243, history: 2292552636, checkpoints: 2085369452, parameters: 720 },
    },
    {
      name: "every option left at its default",
      options: { text: harbor, rank: 6 },
      expected: { weights: 3276196568, loss: 4169652263, history: 600867079, checkpoints: 833761295, parameters: 360 },
    },
    {
      name: "soft-label distillation at temperature 2",
      options: { text: harbor, rank: 6, epochs: 18, seed: 3, teacher: teacher.weights, softLabels: true, temperature: 2 },
      expected: { weights: 2552811518, loss: 612423496, history: 1832079616, checkpoints: 2746511062, parameters: 360 },
    },
    {
      name: "soft-label distillation at temperature 1",
      options: { text: harbor, rank: 3, epochs: 9, seed: 3, teacher: teacher.weights, softLabels: true, temperature: 1 },
      expected: { weights: 3755280173, loss: 3488280006, history: 3985355666, checkpoints: 1812682841, parameters: 180 },
    },
    {
      name: "hard labels measured against a teacher",
      options: { text: harbor, rank: 6, epochs: 18, seed: 3, teacher: teacher.weights, softLabels: false },
      expected: { weights: 1842598852, loss: 3313838414, history: 553199571, checkpoints: 3869735333, parameters: 360 },
    },
    {
      name: "zero epochs",
      options: { text: harbor, rank: 4, epochs: 0 },
      expected: { weights: 1103253592, loss: 715769253, history: 2166136261, checkpoints: 1746686632, parameters: 240 },
    },
  ];

  it.each(baseline.map((entry) => [entry.name, entry] as const))(
    "leaves %s bit-identical to the version without the new options",
    (_name, entry) => {
      expect(fingerprint(trainTinyFactored(entry.options))).toEqual(entry.expected);
    },
  );

  it("treats the explicit defaults exactly like omitting them", () => {
    const plain = trainTinyFactored(baseline[0].options);
    const spelled = trainTinyFactored({
      ...baseline[0].options,
      schedule: "constant",
      warmupFraction: 0.15,
      weightDecay: 0,
      clipNorm: 0,
    });
    expect(fingerprint(spelled)).toEqual(fingerprint(plain));
    expect(spelled.clippedSteps).toBe(0);
    expect(spelled.diverged).toBe(false);
  });

  const recipe = { text: TINY_CORPORA.harbor.text, rank: 8, epochs: 20, batchSize: 16, seed: 9 } as const;

  it("takes the step size from the schedule, from the first batch to the last", () => {
    const peak = 0.5;
    const run = trainTinyFactored({ ...recipe, learningRate: peak, schedule: "cosine", historyPoints: 1000 });
    expect(run.steps).toBe(640);
    expect(run.history[0].learningRate).toBeCloseTo(peak * scheduleFactor("cosine", 0), 12);
    expect(run.history.at(-1)!.learningRate).toBeCloseTo(peak * scheduleFactor("cosine", 1), 12);
    const rates = run.history.map((point) => point.learningRate);
    for (let index = 1; index < rates.length; index += 1) {
      expect(rates[index]).toBeLessThan(rates[index - 1]);
    }
    const cycle = trainTinyFactored({ ...recipe, learningRate: peak, schedule: "one-cycle", historyPoints: 1000 });
    expect(cycle.history[0].learningRate).toBeLessThan(cycle.history[Math.floor(cycle.history.length * 0.15)].learningRate);
    const constant = trainTinyFactored({ ...recipe, learningRate: peak, historyPoints: 1000 });
    expect(new Set(constant.history.map((point) => point.learningRate))).toEqual(new Set([peak]));
  });

  it("changes the result when the schedule changes and keeps it deterministic", () => {
    const options = { ...recipe, learningRate: 0.5 } as const;
    const cosine = trainTinyFactored({ ...options, schedule: "cosine" });
    const again = trainTinyFactored({ ...options, schedule: "cosine" });
    const constant = trainTinyFactored(options);
    expect(Array.from(cosine.weights)).toEqual(Array.from(again.weights));
    expect(cosine.finalLoss).not.toBe(constant.finalLoss);
    // Ending on a small step leaves a run that has not moved as far from its start.
    expect(cosine.finalLoss).toBeGreaterThan(constant.finalLoss);
  });

  it("clips on the global norm of both factors and reports the norm before clipping", () => {
    const options = { ...recipe, learningRate: 0.5 } as const;
    const free = trainTinyFactored(options);
    const roomy = trainTinyFactored({ ...options, clipNorm: 1e6 });
    expect(roomy.clippedSteps).toBe(0);
    expect(Array.from(roomy.weights)).toEqual(Array.from(free.weights));
    expect(roomy.peakGradientNorm).toBe(free.peakGradientNorm);
    expect(free.peakGradientNorm).toBeGreaterThan(0.5);

    const tight = trainTinyFactored({ ...options, clipNorm: 0.05 });
    expect(tight.clippedSteps).toBe(tight.steps);
    expect(tight.peakGradientNorm).toBeGreaterThan(0.05);
    // Every step is shortened, so the clipped run ends higher than the free one.
    expect(tight.finalLoss).toBeGreaterThan(free.finalLoss);
  });

  it("lets clipping rescue a run whose step size is too large", () => {
    // A product of two matrices can explode: each factor's gradient scales with the other.
    const options = { ...recipe, learningRate: 8 } as const;
    const unclipped = trainTinyFactored(options);
    expect(unclipped.diverged).toBe(true);
    expect(unclipped.steps).toBeLessThan(640);
    expect(Number.isNaN(unclipped.finalLoss)).toBe(true);
    const clipped = trainTinyFactored({ ...options, clipNorm: 0.5 });
    expect(clipped.diverged).toBe(false);
    expect(Number.isFinite(clipped.finalLoss)).toBe(true);
    expect(clipped.finalLoss).toBeLessThan(UNIFORM_CROSS_ENTROPY * 1.1);
    expect(clipped.clippedSteps).toBeGreaterThan(clipped.steps / 2);

    const decayed = trainTinyFactored({ ...options, schedule: "cosine" });
    expect(decayed.diverged).toBe(false);
    expect(decayed.peakGradientNorm).toBeGreaterThan(1000);
    const decayedAndClipped = trainTinyFactored({ ...options, schedule: "cosine", clipNorm: 0.5 });
    expect(decayedAndClipped.peakGradientNorm).toBeLessThan(5);
    expect(decayedAndClipped.finalLoss).toBeLessThan(decayed.finalLoss);
  });

  it("pulls both factors toward zero with weight decay", () => {
    const norm = (run: ReturnType<typeof trainTinyFactored>) =>
      Math.sqrt(Array.from(run.weights).reduce((sum, value) => sum + value * value, 0));
    const options = { ...recipe, learningRate: 0.5 } as const;
    const plain = trainTinyFactored(options);
    const decayed = trainTinyFactored({ ...options, weightDecay: 0.05 });
    expect(norm(decayed)).toBeLessThan(norm(plain));
    expect(decayed.diverged).toBe(false);
  });
});
