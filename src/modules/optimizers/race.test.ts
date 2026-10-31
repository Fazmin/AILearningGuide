import { afterEach, describe, expect, it, vi } from "vitest";
import { bigramPairs, tinyCrossEntropy, TINY_CORPORA, trainTinyExtended } from "@app/module-sdk";
import {
  adamSpec,
  baseSpec,
  BATCH_SIZE,
  captureMuon,
  DECAY_LADDER,
  EPOCHS,
  formatRate,
  formatSmall,
  HELD_OUT_TEXT,
  idleShrink,
  ladderIndex,
  muonSpectra,
  OPTIMIZERS,
  rateLadder,
  rowSteps,
  runSweepNow,
  snapToLadder,
  startSweep,
  stateRows,
  STEPS_PER_EPOCH,
  subscribeSweep,
  sweepProgress,
  sweepRates,
  sweepTable,
  sweepTasks,
  TOTAL_STEPS,
  TRAIN_TEXT,
  trainerKind,
  trainRun,
} from "./race";

describe("the data split", () => {
  it("trains on Harbor's first six sentences and holds out the last two", () => {
    const sentences = TINY_CORPORA.harbor.text.split(".").map((part) => part.trim()).filter(Boolean);
    expect(sentences).toHaveLength(8);
    expect(TRAIN_TEXT.startsWith("the morning fog settles over the harbor")).toBe(true);
    expect(HELD_OUT_TEXT).toBe(`${sentences[6]}. ${sentences[7]}.`);
    expect(TRAIN_TEXT.includes(sentences[6])).toBe(false);
    expect(bigramPairs(TRAIN_TEXT).inputs.length).toBe(378);
    expect(bigramPairs(HELD_OUT_TEXT).inputs.length).toBe(118);
  });

  it("runs 24 batches of 16 per epoch, 600 steps in 25 epochs", () => {
    expect(Math.ceil(378 / BATCH_SIZE)).toBe(STEPS_PER_EPOCH);
    expect(TOTAL_STEPS).toBe(600);
    const run = trainRun(baseSpec("sgd", 1));
    expect(run.steps).toBe(600);
    expect(run.trainCurve).toHaveLength(EPOCHS + 1);
    expect(run.heldCurve).toHaveLength(EPOCHS + 1);
    // Zeroed weights give every character 1/30; the trainer scores in 32-bit floats.
    expect(run.trainCurve[0]).toBeCloseTo(Math.log(30), 6);
  });
});

describe("rate notches", () => {
  it("gives every optimizer nine notches from a tenth of its default to ten times it", () => {
    expect(rateLadder("sgd")).toEqual([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]);
    expect(rateLadder("momentum")).toEqual([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]);
    expect(rateLadder("adamw")).toEqual([0.003, 0.006, 0.009, 0.015, 0.03, 0.06, 0.09, 0.15, 0.3]);
    expect(rateLadder("muon")).toEqual([0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5]);
    for (const choice of OPTIMIZERS) {
      expect(rateLadder(choice.id)[4]).toBe(choice.defaultRate);
      // Every sweep rate is a notch, so a slider can reproduce any cell of the tuning table.
      for (const rate of sweepRates(choice.id)) expect(rateLadder(choice.id)).toContain(rate);
    }
  });

  it("snaps to the nearest notch in log space and finds its index", () => {
    expect(snapToLadder(0.04, rateLadder("adamw"), 0.03)).toBe(0.03);
    expect(snapToLadder(0.045, rateLadder("adamw"), 0.03)).toBe(0.06);
    expect(snapToLadder("0.3", rateLadder("adamw"), 0.03)).toBe(0.03);
    expect(snapToLadder(0, DECAY_LADDER, 0)).toBe(0);
    expect(ladderIndex(0.3, rateLadder("adamw"))).toBe(8);
    expect(ladderIndex(0.1, DECAY_LADDER)).toBe(7);
  });
});

describe("runs", () => {
  it("is the trainer's own run, scored on the held-out text", () => {
    const run = trainRun(baseSpec("adamw", 0.03));
    const direct = trainTinyExtended({
      text: TRAIN_TEXT,
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      learningRate: 0.03,
      seed: 1,
      optimizer: { kind: "adamw" },
    });
    expect(run.trainLoss).toBe(direct.finalLoss);
    expect(run.heldOut).toBe(tinyCrossEntropy(direct.weights, HELD_OUT_TEXT));
    expect(run.heldCurve.at(-1)).toBe(run.heldOut);
  });

  it("caches equal runs and ignores Adam settings for the other optimizers", () => {
    const first = trainRun(baseSpec("sgd", 1));
    expect(trainRun({ ...baseSpec("sgd", 1), biasCorrection: false, warmup: 0.2, weightDecay: 0.1 })).toBe(first);
    const decoupled = trainRun(adamSpec({ rate: 0.03, biasCorrection: true, warmup: 0, weightDecay: 0, decayMode: "decoupled" }));
    expect(trainRun(adamSpec({ rate: 0.03, biasCorrection: true, warmup: 0, weightDecay: 0, decayMode: "l2" }))).toBe(decoupled);
  });

  it("sends L2 decay to the trainer's adam and decoupled decay to its adamw", () => {
    expect(trainerKind({ ...baseSpec("adamw", 0.03), weightDecay: 0.1, decayMode: "l2" })).toBe("adam");
    expect(trainerKind({ ...baseSpec("adamw", 0.03), weightDecay: 0.1, decayMode: "decoupled" })).toBe("adamw");
    expect(trainerKind({ ...baseSpec("adamw", 0.03), weightDecay: 0, decayMode: "l2" })).toBe("adamw");
    expect(trainerKind(baseSpec("muon", 0.05))).toBe("muon");
  });

  it("traces only the AdamW run, and starts its early window at the uniform loss", () => {
    const adam = trainRun(baseSpec("adamw", 0.03));
    expect(adam.batchLoss).toHaveLength(600);
    expect(adam.batchLoss[0]).toBeCloseTo(Math.log(30), 4);
    expect(adam.earlyPeakLoss).toBe(Math.max(...adam.batchLoss.slice(0, 60)));
    expect(trainRun(baseSpec("sgd", 1)).batchLoss).toHaveLength(0);
  });
});

describe("step size per row", () => {
  it("divides each row's travelled distance by the batches that held it", () => {
    const run = trainRun(baseSpec("adamw", 0.03));
    const rows = rowSteps(run);
    expect(rows.map((row) => row.label)).toEqual(["space", "e", "t", "a", "o", "k", "q", "z"]);
    const space = rows[0];
    expect(space.visits).toBe(run.rowVisits[0]);
    expect(space.perVisit).toBe(run.rowMovement[0] / run.rowVisits[0]);
    expect(rows.find((row) => row.label === "q")?.visits).toBe(25);
    expect(rows.find((row) => row.label === "k")?.visits).toBe(50);
  });

  it("counts the pairs each row starts in the training text", () => {
    const { inputs } = bigramPairs(TRAIN_TEXT);
    const starts = (character: string) => Array.from(inputs).filter((id) => id === " abcdefghijklmnopqrstuvwxyz.<>".indexOf(character)).length;
    expect([starts(" "), starts("q"), starts("z"), starts("k")]).toEqual([70, 1, 1, 2]);
    expect(["j", "x", "<", ">"].map(starts)).toEqual([0, 0, 0, 0]);
  });
});

describe("idle-weight shrink", () => {
  it("takes rate × decay under decoupled decay and about the full rate under L2", () => {
    const decoupled = idleShrink({ rate: 0.03, weightDecay: 0.1, decayMode: "decoupled", biasCorrection: true });
    expect(decoupled).toBeCloseTo((1 - (1 - 0.003) ** 10) / 10, 12);
    for (const weightDecay of DECAY_LADDER.filter((value) => value > 0)) {
      const l2 = idleShrink({ rate: 0.03, weightDecay, decayMode: "l2", biasCorrection: true });
      expect(l2).toBeGreaterThan(0.029);
      expect(l2).toBeLessThanOrEqual(0.03);
    }
    expect(idleShrink({ rate: 0.03, weightDecay: 0, decayMode: "l2", biasCorrection: true })).toBe(0);
    expect(idleShrink({ rate: 0.03, weightDecay: 0.1, decayMode: "l2", biasCorrection: false })).toBeGreaterThan(0.03);
  });
});

describe("Muon capture", () => {
  it("captures the same matrix from a run cut short as from the full race run", () => {
    const short = captureMuon(0.05, "middle")!;
    const full = trainRun({ ...baseSpec("muon", 0.05), captureStep: 299 }).captured!;
    expect(Array.from(short.direction)).toEqual(Array.from(full.direction));
    expect([short.rows, short.cols]).toEqual([30, 30]);
  });

  it("reads before at zero iterations and after at the chosen count", () => {
    const zero = muonSpectra(0.05, "middle", 0)!;
    expect(zero.after.values).toEqual(zero.before.values);
    const five = muonSpectra(0.05, "middle", 5)!;
    expect(five.iterations).toBe(5);
    expect(five.after.zeros).toBe(five.before.zeros);
  });
});

describe("Tune each", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("trains one run per macrotask and tells listeners as each lands", () => {
    vi.useFakeTimers();
    let calls = 0;
    const unsubscribe = subscribeSweep(() => {
      calls += 1;
    });
    expect(sweepTable()).toBeNull();
    startSweep();
    expect(sweepProgress()).toMatchObject({ done: 0, total: 20, running: true, complete: false });
    vi.advanceTimersToNextTimer();
    expect(sweepProgress().done).toBe(1);
    vi.runAllTimers();
    expect(sweepProgress()).toMatchObject({ done: 20, running: false, complete: true });
    expect(calls).toBe(21);
    unsubscribe();
  });

  it("covers a tenth to ten times each default and marks the lowest held-out loss", () => {
    expect(sweepTasks()).toHaveLength(20);
    const table = runSweepNow();
    for (const row of table) {
      expect(row.points.map((point) => point.factor)).toEqual([0.1, 0.3, 1, 3, 10]);
      expect(row.atDefault.rate).toBe(OPTIMIZERS.find((choice) => choice.id === row.optimizer)!.defaultRate);
      expect(Math.min(...row.points.map((point) => point.heldOut))).toBe(row.best.heldOut);
      expect(row.points.map((point) => point.heldOut)).toEqual(
        row.points.map((point) => trainRun(baseSpec(row.optimizer, point.rate)).heldOut),
      );
    }
  });
});

describe("optimizer state", () => {
  it("measures the lab's buffers and multiplies out 7.5 billion", () => {
    const floats = {
      sgd: trainRun(baseSpec("sgd", 1)).stateFloats,
      momentum: trainRun(baseSpec("momentum", 0.1)).stateFloats,
      adamw: trainRun(baseSpec("adamw", 0.03)).stateFloats,
      muon: trainRun(baseSpec("muon", 0.05)).stateFloats,
    };
    expect(floats).toEqual({ sgd: 0, momentum: 900, adamw: 1800, muon: 900 });
    const rows = stateRows(floats);
    expect(rows.map((row) => [row.optimizer, row.buffers, row.labBytes, row.largeBytes, row.largeMixedTotal])).toEqual([
      ["sgd", 0, 0, 0, 60e9],
      ["momentum", 1, 3600, 30e9, 90e9],
      ["adamw", 2, 7200, 60e9, 120e9],
      ["muon", 1, 3600, 30e9, 90e9],
    ]);
  });
});

describe("number formats", () => {
  it("prints rates and small values without long decimals", () => {
    expect([formatRate(0.009), formatRate(0.015), formatRate(1), formatRate(0.0001), formatRate(0)]).toEqual([
      "0.009",
      "0.015",
      "1",
      "0.0001",
      "0",
    ]);
    expect([formatSmall(0.029628), formatSmall(0.00296), formatSmall(3e-6), formatSmall(0)]).toEqual([
      "0.030",
      "0.0030",
      "3.0 × 10⁻⁶",
      "0",
    ]);
  });
});
