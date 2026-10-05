import { describe, expect, it } from "vitest";
import {
  bigramPairs,
  sampleTinyText,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
} from "@app/module-sdk";
import { BASE_EPOCHS as MERGING_BASE_EPOCHS, baseWeights as mergingBase } from "../model-merging/lab";
import { buildFineTuneData } from "./holdout";
import { BLOCK_IDS, BLOCK_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE } from "./state";

/**
 * Pins the measured values the fine-tuning-transfer lesson and card text quote
 * (module.ts, content/standard.mdx, content/plain.mdx, card-info.ts).
 *
 * `lab` mirrors Explore.tsx exactly: the base model is harbor, 60 epochs, seed 1; transfer and scratch both
 * train on recipes at seed 2 with `checkpoints: 9`, transfer from the base weights with the frozen rows
 * masked, scratch from zeros. Metrics are the lab's own: Transfer = tinyCrossEntropy(transfer weights),
 * Scratch = scratch.finalLoss, Source before/after on harbor, Forgetting = after - before.
 */
const source = TINY_CORPORA.harbor;
const target = TINY_CORPORA.recipes;
const base = trainTinyModel({ text: source.text, epochs: 60, seed: 1 });
const sourceBefore = tinyCrossEntropy(base.weights, source.text);
const targetBefore = tinyCrossEntropy(base.weights, target.text);

/** The freeze grid's blocks, as Explore builds them: BLOCK_SIZE consecutive vocabulary rows each. */
const blockRows = (id: string) => {
  const index = BLOCK_IDS.indexOf(id);
  return Array.from({ length: BLOCK_SIZE }, (_, offset) => index * BLOCK_SIZE + offset).filter((row) => row < TINY_VOCAB_SIZE);
};
const block = (characters: string) => `block-${TINY_VOCAB.indexOf(characters[0]) / BLOCK_SIZE}`;

function lab(fineTuneEpochs: number, learningRate: number, frozen: string[] = []) {
  const trainableRows = Array.from({ length: TINY_VOCAB_SIZE }, () => true);
  for (const id of frozen) for (const row of blockRows(id)) trainableRows[row] = false;
  const transfer = trainTinyModel({
    text: target.text,
    epochs: fineTuneEpochs,
    learningRate,
    init: base.weights,
    trainableRows,
    seed: 2,
    checkpoints: 9,
  });
  const scratch = trainTinyModel({ text: target.text, epochs: fineTuneEpochs, learningRate, seed: 2, checkpoints: 9 });
  const targetAfter = tinyCrossEntropy(transfer.weights, target.text);
  const sourceAfter = tinyCrossEntropy(transfer.weights, source.text);
  return {
    transfer,
    scratch,
    trainableRows,
    targetAfter,
    sourceAfter,
    scratchLoss: scratch.finalLoss,
    lower: scratch.finalLoss - targetAfter,
    forgetting: sourceAfter - sourceBefore,
    gained: targetBefore - targetAfter,
    sourceSeries: transfer.checkpoints.map((point) => tinyCrossEntropy(point.weights, source.text)),
    targetSeries: transfer.checkpoints.map((point) => tinyCrossEntropy(point.weights, target.text)),
  };
}
const round = (value: number, digits = 3) => Number(value.toFixed(digits));
/** How far each context row moved: the L2 norm of that row of W_after - W_before, as the row strip draws it. */
const rowMovement = (weights: Float32Array) =>
  Array.from({ length: TINY_VOCAB_SIZE }, (_, row) => {
    let squared = 0;
    for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) squared += (weights[row * TINY_VOCAB_SIZE + k] - base.weights[row * TINY_VOCAB_SIZE + k]) ** 2;
    return Math.sqrt(squared);
  });

const defaults = lab(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE);
const twoBlocks = lab(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, [block(" abcde"), block("fghijk")]);

describe("fine-tuning-transfer quoted values", () => {
  it("uses the defaults the lesson names: 20 epochs at learning rate 0.6", () => {
    // standard.mdx:49, plain.mdx:57, card-info.ts:23,83: "the defaults, 20 epochs at learning rate 0.6"
    expect(DEFAULT_EPOCHS).toBe(20);
    expect(DEFAULT_LEARNING_RATE).toBe(0.6);
    // card-info.ts:79: the base used the trainer default of 0.6 (and batch 16) for its 60 epochs
    const explicit = trainTinyModel({ text: source.text, epochs: 60, seed: 1, learningRate: 0.6, batchSize: 16 });
    expect(explicit.weights).toEqual(base.weights);
  });

  it("matches the base model's two losses and its size", () => {
    // standard.mdx:9,13-14, plain.mdx:10,14-15, card-info.ts:7,15,32: 900 weights; scores 1.921 on Harbor, 2.556 on Recipe steps
    expect(base.weights).toHaveLength(900);
    expect(round(sourceBefore)).toBe(1.921);
    expect(round(targetBefore)).toBe(2.556);
    // standard.mdx:24, plain.mdx:25: "the same 30 symbols"
    expect(TINY_VOCAB_SIZE).toBe(30);
    // plain.mdx:104, standard.mdx:100, card-info.ts:31: "eight sentences per corpus"
    for (const corpus of [source, target]) {
      expect(corpus.text.split(".").filter((part) => part.trim()).length).toBe(8);
    }
  });

  it("matches the cross-lab note: Model merging's own 48-epoch base reads 1.953 on Harbor and 2.553 on Recipe steps", () => {
    // standard.mdx:106-107 and plain.mdx:110-111: "Model merging trains its own for 48 epochs, so its base reads 1.953
    // on Harbor and 2.553 on Recipe steps" (read through that lab's own baseWeights()), against this lab's 60-epoch table
    expect(MERGING_BASE_EPOCHS).toBe(48);
    const other = mergingBase();
    expect(round(tinyCrossEntropy(other, source.text))).toBe(1.953);
    expect(round(tinyCrossEntropy(other, target.text))).toBe(2.553);
    expect(round(sourceBefore)).not.toBe(1.953);
  });

  it("matches the zero-epoch head start: 2.556 against ln(30) = 3.401, a gap of 0.845", () => {
    const zero = lab(0, DEFAULT_LEARNING_RATE);
    // standard.mdx:44-46, plain.mdx:52-54, card-info.ts:17,23: "a zeroed table ... loss is exactly ln(30)", 3.401
    expect(round(UNIFORM_CROSS_ENTROPY)).toBe(3.401);
    expect(zero.scratchLoss).toBeCloseTo(UNIFORM_CROSS_ENTROPY, 6);
    expect(round(zero.scratchLoss)).toBe(3.401);
    // The transfer curve is a single point: the pretrained model's loss on recipes
    expect(zero.transfer.checkpoints).toHaveLength(1);
    expect(round(zero.targetAfter)).toBe(2.556);
    // card-info.ts:23: "the gap is 0.845 (2.556 against 3.401)"
    expect(round(zero.lower)).toBe(0.845);
    // card-info.ts:78: epochs 0 keeps the pretrained weights, scratch stays at zeros, Target gained is 0
    expect(zero.transfer.weights).toEqual(base.weights);
    expect(zero.scratch.weights.every((value) => value === 0)).toBe(true);
    expect(zero.gained).toBe(0);
    expect(zero.forgetting).toBe(0);
    // card-info.ts:51: "At 0 fine-tune epochs ... the samples match" (the 'Before' sample is the base)
    const sample = (weights: Float32Array) => sampleTinyText(weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 12 });
    expect(sample(zero.transfer.weights)).toBe(sample(base.weights));
  });

  it("matches the default run: transfer 2.123, scratch 2.247, lower by 0.123, forgetting +0.144", () => {
    // standard.mdx:49-52, plain.mdx:57-60, card-info.ts:23
    expect(round(defaults.targetAfter)).toBe(2.123);
    expect(round(defaults.scratchLoss)).toBe(2.247);
    expect(round(defaults.lower)).toBe(0.123);
    expect(round(sourceBefore)).toBe(1.921);
    expect(round(defaults.sourceAfter)).toBe(2.065);
    expect(round(defaults.forgetting)).toBe(0.144);
    // "Target gained reports the same 2.556-to-2.123 drop": the drop is exactly targetBefore - targetAfter
    expect(round(targetBefore)).toBe(2.556);
    expect(round(defaults.gained)).toBe(round(targetBefore - defaults.targetAfter));
    // standard.mdx:70-71 and plain.mdx:76-77: "grows checkpoint by checkpoint": Harbor loss rises at every checkpoint
    for (let index = 1; index < defaults.sourceSeries.length; index += 1) {
      expect(defaults.sourceSeries[index]).toBeGreaterThan(defaults.sourceSeries[index - 1]);
    }
  });

  it("matches the 60-epoch run: Recipe about 1.96, Harbor about 2.22, forgetting roughly +0.30", () => {
    const long = lab(60, DEFAULT_LEARNING_RATE);
    // standard.mdx:54-55 and plain.mdx:63 ("about 1.96", "about 2.22")
    expect(round(long.targetAfter, 2)).toBe(1.96);
    expect(round(long.sourceAfter, 2)).toBe(2.22);
    // standard.mdx:70-71 and plain.mdx:76-77 ("roughly +0.30 at 60")
    expect(round(long.forgetting, 2)).toBe(0.3);
    // The forgetting curve keeps rising with the budget (checkpoint by checkpoint, never falling)
    for (let index = 1; index < long.sourceSeries.length; index += 1) {
      expect(long.sourceSeries[index]).toBeGreaterThanOrEqual(long.sourceSeries[index - 1]);
    }
    expect(long.forgetting).toBeGreaterThan(defaults.forgetting);
  });

  it("shows the head start shrinking as the budget grows (transfer buys speed, not a better ceiling)", () => {
    // standard.mdx:66-69, plain.mdx:73-75, card-info.ts:27,83: the two target curves sit far apart at small
    // budgets and from-scratch closes most of the distance
    const gap = (epochs: number) => lab(epochs, DEFAULT_LEARNING_RATE).lower;
    const gaps = [0, 10, 20, 60].map(gap);
    for (let index = 1; index < gaps.length; index += 1) expect(gaps[index]).toBeLessThan(gaps[index - 1]);
    expect(gaps[0]).toBeGreaterThan(0.8);
    expect(gaps.at(-1)).toBeLessThan(0.1 * gaps[0]);
    // Transfer is lower than scratch at every budget on this page's default rate
    for (const value of gaps) expect(value).toBeGreaterThanOrEqual(0);
  });

  it("matches the freezing figures: two blocks, 12 / 30 rows, 540 weights, 2.345 and 1.990", () => {
    const frozen = twoBlocks.trainableRows.filter((value) => !value).length;
    // standard.mdx:57-58, plain.mdx:66-67, card-info.ts:105,113: Frozen rows 12 / 30, Trainable weights 540
    expect(blockRows(block(" abcde"))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(TINY_VOCAB.slice(6, 12)).toBe("fghijk");
    expect(frozen).toBe(12);
    expect(TINY_VOCAB_SIZE - frozen).toBe(18);
    expect((TINY_VOCAB_SIZE - frozen) * TINY_VOCAB_SIZE).toBe(540);
    // card-info.ts:105,113: "five blocks of exactly six, so each frozen block removes 180 trainable weights"
    expect(BLOCK_IDS).toHaveLength(5);
    expect(BLOCK_SIZE * TINY_VOCAB_SIZE).toBe(180);
    // standard.mdx:58-59, plain.mdx:67-68: Recipe steps only reaches 2.345 instead of 2.123; Harbor holds at
    // 1.990 instead of 2.065
    expect(round(twoBlocks.targetAfter)).toBe(2.345);
    expect(round(twoBlocks.sourceAfter)).toBe(1.99);
    // standard.mdx:72-73, plain.mdx:78-79: 12 frozen rows cost 0.222 of adaptation and save 0.075 of forgetting
    expect(round(twoBlocks.targetAfter - defaults.targetAfter)).toBe(0.222);
    expect(round(defaults.sourceAfter - twoBlocks.sourceAfter)).toBe(0.075);
    // The freeze is a dial between the two: less adaptation and less forgetting than unfrozen
    expect(twoBlocks.forgetting).toBeLessThan(defaults.forgetting);
    expect(twoBlocks.gained).toBeLessThan(defaults.gained);
    // card-info.ts:22: "a large freeze can make transfer lose to scratch": Transfer is higher by (negative gap) here
    expect(twoBlocks.lower).toBeLessThan(0);
  });

  it("freezing every block leaves the transfer curve flat and lets scratch pass it", () => {
    // card-info.ts:28,111: "transfer cannot move and the solid curve is flat; scratch will pass it"
    const all = lab(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, [...BLOCK_IDS]);
    expect(all.transfer.weights).toEqual(base.weights);
    for (const value of all.targetSeries) expect(value).toBeCloseTo(targetBefore, 6);
    expect(all.scratchLoss).toBeLessThan(all.targetAfter);
  });

  it("matches the row strip: which rows never move", () => {
    // plain.mdx:82-83 and card-info.ts:121: j, q, x, z, < and > never move even when trainable
    const movement = rowMovement(defaults.transfer.weights);
    const still = [...TINY_VOCAB].filter((_, row) => movement[row] === 0);
    expect(still.sort()).toEqual(["<", ">", "j", "q", "x", "z"].sort());
    // They never occur as a context in Recipe steps, so they get no gradient
    const contexts = new Set(Array.from(bigramPairs(target.text).inputs, (id) => TINY_VOCAB[id]));
    expect([...TINY_VOCAB].filter((character) => !contexts.has(character)).sort()).toEqual(still.sort());
    // standard.mdx:59-60: "those twelve rows flat at zero" (the two frozen blocks are twelve rows)
    const frozenMovement = rowMovement(twoBlocks.transfer.weights);
    for (const id of [block(" abcde"), block("fghijk")]) for (const row of blockRows(id)) expect(frozenMovement[row]).toBe(0);
    expect(blockRows(block(" abcde")).length + blockRows(block("fghijk")).length).toBe(12);
  });

  it("matches the block claims: xyz.<> alone barely matters, the space block is the opposite extreme", () => {
    // standard.mdx:78-81 and plain.mdx:84-86: freeze `xyz.<>` alone and almost nothing moves
    expect(TINY_VOCAB.slice(24, 30)).toBe("xyz.<>");
    const tail = lab(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, [block("xyz.<>")]);
    expect(Math.abs(tail.targetAfter - defaults.targetAfter)).toBeLessThan(0.002);
    expect(Math.abs(tail.sourceAfter - defaults.sourceAfter)).toBeLessThan(0.002);
    // "the block holding the space row is the opposite extreme": space is the most common context in Recipe steps
    const counts = new Map<string, number>();
    for (const id of bigramPairs(target.text).inputs) counts.set(TINY_VOCAB[id], (counts.get(TINY_VOCAB[id]) ?? 0) + 1);
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    expect(ranked[0][0]).toBe(" ");
    // ...and freezing it costs far more adaptation than freezing xyz.<>
    const head = lab(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, [block(" abcde")]);
    expect(head.targetAfter - defaults.targetAfter).toBeGreaterThan(0.1);
    // card-info.ts:55,117: freezing the space/e block makes forgetting shrink
    expect(head.forgetting).toBeLessThan(defaults.forgetting);
  });

  it("treats epochs times learning rate as one budget", () => {
    // standard.mdx:62-63 and plain.mdx:68-69 ("halve epochs, double the learning rate"), module.ts glossary
    // "Adaptation budget: epochs times learning rate, roughly": the forgetting stays within 0.01 nats
    const halved = lab(DEFAULT_EPOCHS / 2, DEFAULT_LEARNING_RATE * 2);
    expect(Math.abs(halved.forgetting - defaults.forgetting)).toBeLessThan(0.01);
    expect(Math.abs(halved.targetAfter - defaults.targetAfter)).toBeLessThan(0.01);
  });

  it("matches the controls card: 5 epochs at learning rate 2.00 gains about 0.40, and gains stay positive", () => {
    // card-info.ts:88 ("even 5 epochs at learning rate 2.00 gains about 0.40 nats")
    expect(round(lab(5, 2).gained, 2)).toBe(0.4);
    // card-info.ts:84: learning rate 2.00 "will move harbor numbers quickly and usually for the worse"
    expect(lab(20, 2).forgetting).toBeGreaterThan(defaults.forgetting);
    // card-info.ts:88 ("Target gained stays positive across this page's ranges"), on a grid over both sliders
    for (const epochs of [1, 5, 20, 60]) {
      for (const learningRate of [0.05, 0.6, 1.5, 2]) expect(lab(epochs, learningRate).gained).toBeGreaterThan(0);
    }
  });

  // card-info.ts:59 now says the after sample is "mostly letter fragments, not recipe words". At the defaults it is
  //   "t. ond ta a sspand tond panstheswithe ithestoog tite the and a bor t parnd t"
  // (prompt "the ", 76 characters, temperature 0.7, seed 12); none of warm, stir or sauce appears in it or in any
  // after sample over epochs 0 to 60 and learning rates 0.05 to 2, which is why the card no longer says it does.
});

/* -------------------------------------------------------------------------- */
/* The Replay and held-out target card                                         */
/* -------------------------------------------------------------------------- */

/**
 * Mirrors the card in Explore.tsx: its own fine-tune from the same base, on the first recipe sentences plus
 * the first harbor sentences, with every row free, seed 2 and `checkpoints: 9`. Forgetting is the loss on harbor
 * text after minus before, for all of harbor, for the replayed sentences, and for the sentences not replayed.
 */
const replayLab = (held: number, replay: number, epochs = 20, learningRate = 0.6) => {
  const data = buildFineTuneData(held, replay);
  const run = trainTinyModel({ text: data.text, epochs, learningRate, init: base.weights, seed: 2, checkpoints: 9 });
  const change = (text: string) => (text ? tinyCrossEntropy(run.weights, text) - tinyCrossEntropy(base.weights, text) : Number.NaN);
  return {
    data,
    run,
    trained: tinyCrossEntropy(run.weights, data.trainTarget),
    held: data.heldTarget ? tinyCrossEntropy(run.weights, data.heldTarget) : Number.NaN,
    forgetting: change(source.text),
    replayed: change(data.replay),
    notReplayed: change(data.notReplayed),
    heldCurve: run.checkpoints.map((point) => (data.heldTarget ? tinyCrossEntropy(point.weights, data.heldTarget) : Number.NaN)),
  };
};
/** The no-replay fine-tune, scored on the original-corpus groups of a chosen replay setting. */
const plainOn = (held: number, replay: number, epochs = 20) => {
  const chosen = buildFineTuneData(held, replay);
  const plain = replayLab(held, 0, epochs);
  const change = (text: string) => tinyCrossEntropy(plain.run.weights, text) - tinyCrossEntropy(base.weights, text);
  return { replayed: change(chosen.replay), notReplayed: change(chosen.notReplayed) };
};
const f3 = (value: number) => Number(value.toFixed(3));

describe("fine-tuning-transfer replay and held-out values", () => {
  it("matches the defaults: trained-on target, held-out target, forgetting, and the gap", () => {
    const lab20 = replayLab(3, 0);
    expect([f3(lab20.trained), f3(lab20.held), f3(lab20.forgetting)]).toEqual([2.162, 2.424, 0.123]);
    expect(f3(lab20.held - lab20.trained)).toBe(0.262);
    expect(lab20.run.steps).toBe(320);
  });

  it("matches how the gap widens with epochs while held-out loss flattens", () => {
    const gaps = [10, 20, 60].map((epochs) => {
      const lab = replayLab(3, 0, epochs);
      return [epochs, f3(lab.trained), f3(lab.held), f3(lab.held - lab.trained), f3(lab.forgetting)];
    });
    expect(gaps).toEqual([
      [10, 2.295, 2.451, 0.157, 0.061],
      [20, 2.162, 2.424, 0.262, 0.123],
      [60, 1.927, 2.399, 0.472, 0.296],
    ]);
    // Held-out loss moves by less than a tenth of a nat over the last 40 epochs; trained-on loss by a quarter.
    const sixty = replayLab(3, 0, 60);
    const twenty = replayLab(3, 0, 20);
    expect(twenty.held - sixty.held).toBeLessThan(0.05);
    expect(twenty.trained - sixty.trained).toBeGreaterThan(0.2);
  });

  it("matches the replay sweep at 20 epochs: forgetting falls, the fit slows, held-out barely moves", () => {
    const rows = [0, 50, 100].map((share) => {
      const lab = replayLab(3, share);
      return [share, lab.run.steps, f3(lab.trained), f3(lab.held), f3(lab.forgetting)];
    });
    expect(rows).toEqual([
      [0, 320, 2.162, 2.424, 0.123],
      [50, 640, 2.232, 2.417, 0.042],
      [100, 940, 2.256, 2.419, 0.001],
    ]);
    expect(replayLab(3, 50).data.replayFraction).toBeCloseTo(4 / 9, 10);
  });

  it("shows replay protecting what it replays and little else, at 50% and 20 epochs", () => {
    const withReplay = replayLab(3, 50);
    const without = plainOn(3, 50);
    expect([f3(withReplay.replayed), f3(without.replayed)]).toEqual([-0.038, 0.119]);
    expect([f3(withReplay.notReplayed), f3(without.notReplayed)]).toEqual([0.129, 0.128]);
    // The sentences that were not replayed forget within a hundredth of a nat of the no-replay run.
    expect(Math.abs(withReplay.notReplayed - without.notReplayed)).toBeLessThan(0.01);
    expect(withReplay.replayed).toBeLessThan(0);
  });

  it("matches the 60-epoch replay sweep", () => {
    const rows = [0, 50, 100].map((share) => {
      const lab = replayLab(3, share, 60);
      return [share, lab.run.steps, f3(lab.trained), f3(lab.held), f3(lab.forgetting)];
    });
    expect(rows).toEqual([
      [0, 960, 1.927, 2.399, 0.296],
      [50, 1920, 2.052, 2.382, 0.083],
      [100, 2820, 2.102, 2.382, -0.005],
    ]);
  });

  it("shows small-data overfitting: with 4 sentences held out the held-out curve bottoms out and turns up", () => {
    const lab = replayLab(4, 0, 60);
    expect([f3(lab.trained), f3(lab.held)]).toEqual([1.865, 2.57]);
    const lowest = Math.min(...lab.heldCurve);
    expect(f3(lowest)).toBe(2.541);
    expect(lab.heldCurve.at(-1)! - lowest).toBeGreaterThan(0.02);
    expect(f3(lab.held - lab.trained)).toBe(0.705);
  });

  it("reproduces the first card's target loss when nothing is held out and nothing replayed", () => {
    expect(f3(replayLab(0, 0).trained)).toBe(2.123);
    expect(f3(replayLab(0, 0).forgetting)).toBe(0.144);
  });
});
