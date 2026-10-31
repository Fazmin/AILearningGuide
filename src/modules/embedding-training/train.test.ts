import { describe, expect, it } from "vitest";
import { DUPLICATE_OF, HELD_OUT_QUERIES, PASSAGES, passageIndex, TRAIN_QUERIES } from "./data";
import { BUCKETS, initialWeights, words } from "./encoder";
import {
  BATCH_SIZES,
  batchGradient,
  batchLoss,
  bm25Negatives,
  buildBatch,
  nearestNegatives,
  pickNegative,
  REFRESH_EVERY,
  SNAPSHOT_EVERY,
  STEPS,
  train,
  TRAIN_ANSWERS,
} from "./train";

const query = (id: string) => TRAIN_QUERIES.findIndex((entry) => entry.id === id);

/** Central differences on a few coordinates of the rows a batch actually reads. */
function finiteDifferenceCheck(weights: Float64Array, dim: number, batch: ReturnType<typeof buildBatch>, temperature: number) {
  const { gradient } = batchGradient(weights, dim, batch, temperature);
  const used = [...new Set(gradient.map((value, index) => (value !== 0 ? Math.floor(index / dim) : -1)).filter((row) => row >= 0))];
  const coordinates = used.slice(0, 12).flatMap((row) => [row * dim, row * dim + (dim - 1)]);
  const h = 1e-6;
  let worst = 0;
  const numerics: number[] = [];
  const analytics: number[] = [];
  for (const coordinate of coordinates) {
    const original = weights[coordinate];
    weights[coordinate] = original + h;
    const up = batchLoss(weights, dim, batch, temperature);
    weights[coordinate] = original - h;
    const down = batchLoss(weights, dim, batch, temperature);
    weights[coordinate] = original;
    const numeric = (up - down) / (2 * h);
    // Relative to the largest gradient entry checked, so a near-zero entry cannot inflate the error.
    numerics.push(numeric);
    analytics.push(gradient[coordinate]);
  }
  const scale = Math.max(...analytics.map(Math.abs), 1e-12);
  numerics.forEach((numeric, index) => (worst = Math.max(worst, Math.abs(numeric - analytics[index]) / scale)));
  return { worst, checked: coordinates.length };
}

describe("the InfoNCE gradient", () => {
  it("matches finite differences, including through the L2 normalisation", () => {
    const rows = [query("t01"), query("t04"), query("t10"), query("t12"), query("t25")];
    for (const temperature of [0.05, 0.1, 1]) {
      // Scaling W by 3 leaves every unit vector the same but makes each pooled length far from 1, so the
      // check exercises the 1 / |u| factor and the projection that removes the component along u.
      const weights = initialWeights(4, 16).map((value) => value * 3);
      const plain = finiteDifferenceCheck(weights, 16, buildBatch(rows, null, false), temperature);
      expect(plain.checked).toBeGreaterThan(10);
      expect(plain.worst).toBeLessThan(1e-5);
      // Mined negatives and the duplicate mask take the same path.
      const mined = nearestNegatives(weights, 16, false);
      const masked = finiteDifferenceCheck(weights, 16, buildBatch(rows, mined, true), temperature);
      expect(masked.worst).toBeLessThan(1e-5);
    }
  });

  it("is unchanged by rescaling W, because only directions are compared", () => {
    const rows = [query("t02"), query("t30"), query("t44")];
    const weights = initialWeights(2, 8);
    const batch = buildBatch(rows, null, false);
    const loss = batchLoss(weights, 8, batch, 0.1);
    expect(batchLoss(weights.map((value) => value * 5), 8, batch, 0.1)).toBeCloseTo(loss, 10);
  });

  it("starts at ln B when every candidate scores the same", () => {
    // Every row of W equal: every text encodes to the same unit vector, so every cosine is 1.
    const flat = new Float64Array(BUCKETS * 16);
    for (let bucket = 0; bucket < BUCKETS; bucket += 1) flat[bucket * 16] = 1;
    const distinctAnswers: number[] = [];
    TRAIN_QUERIES.forEach((_, index) => {
      if (!distinctAnswers.some((row) => TRAIN_ANSWERS[row] === TRAIN_ANSWERS[index])) distinctAnswers.push(index);
    });
    for (const size of BATCH_SIZES) {
      const batch = buildBatch(distinctAnswers.slice(0, size), null, false);
      expect(batch.candidates).toHaveLength(size);
      expect(batchLoss(flat, 16, batch, 0.1)).toBeCloseTo(Math.log(size), 12);
      expect(batchLoss(flat, 16, batch, 1)).toBeCloseTo(Math.log(size), 12);
    }
  });
});

describe("negatives", () => {
  it("never mines the labelled answer, and never its twin when skipping", () => {
    for (const skip of [false, true]) {
      const lists = [bm25Negatives(skip), ...[1, 7].map((seed) => nearestNegatives(initialWeights(seed, 16), 16, skip))];
      const run = train({ batch: 8, temperature: 0.1, negatives: "nearest", skipDuplicates: skip, dim: 16, seed: 1 }, { steps: 40 });
      lists.push(nearestNegatives(run.snapshots.at(-1)!.weights, 16, skip));
      for (const mined of lists) {
        mined.forEach((negative, index) => {
          expect(negative).not.toBe(TRAIN_ANSWERS[index]);
          const twin = DUPLICATE_OF[TRAIN_ANSWERS[index]];
          if (skip && twin >= 0) expect(negative).not.toBe(twin);
        });
      }
    }
    expect(pickNegative([5, 9, 1], 1, false)).toBe(0);
    expect(pickNegative([0, 3, 0], 1, false, true)).toBe(-1);
  });

  it("lets BM25 mine only passages that share a keyword: 22 of 90 training queries get none", () => {
    expect(bm25Negatives(false).filter((negative) => negative < 0)).toHaveLength(22);
  });

  it("masks exactly the labelled duplicates when Skip known duplicates is on", () => {
    // t10 and t11 label Sunday hours, t12 labels its twin Sunday notice, t25 Ferry fares, t27 its twin Ticket desk.
    const rows = ["t10", "t11", "t12", "t25", "t27", "t01"].map(query);
    const skipped = buildBatch(rows, null, true);
    const kept = buildBatch(rows, null, false);
    let masks = 0;
    skipped.rows.forEach((row, r) => {
      skipped.candidates.forEach((passage, c) => {
        const twin = DUPLICATE_OF[TRAIN_ANSWERS[row]];
        expect(skipped.masked[r][c]).toBe(passage === twin);
        if (skipped.masked[r][c]) masks += 1;
        expect(kept.masked[r][c]).toBe(false);
      });
    });
    // Sunday hours' two rows skip Sunday notice, Sunday notice's row skips Sunday hours, and the two fare rows skip each other.
    expect(masks).toBe(5);
    // A masked cell gets no share and no gradient.
    const run = train({ batch: 32, temperature: 0.1, negatives: "nearest", skipDuplicates: true, dim: 8, seed: 3 }, { steps: 30 });
    for (const snapshot of run.snapshots) {
      snapshot.batch.rows.forEach((row, r) => {
        snapshot.batch.candidates.forEach((passage, c) => {
          expect(snapshot.batch.masked[r][c]).toBe(passage === DUPLICATE_OF[TRAIN_ANSWERS[row]]);
          if (snapshot.batch.masked[r][c]) expect(snapshot.scores.shares[r][c]).toBe(0);
        });
      });
    }
  });

  it("counts each answer once and shares mined negatives across the batch, as DPR does", () => {
    const rows = ["t01", "t02", "t04"].map(query);
    const batch = buildBatch(rows, bm25Negatives(false), false);
    expect(batch.candidates.filter((passage) => passage === passageIndex("north-hours"))).toHaveLength(1);
    expect(batch.answerColumn[0]).toBe(batch.answerColumn[1]);
    batch.minedColumn.forEach((column, r) => {
      if (column >= 0) expect(batch.candidates[column]).not.toBe(TRAIN_ANSWERS[rows[r]]);
    });
  });
});

describe("the training run", () => {
  it("keeps every held-out query out of training", () => {
    const normalise = (text: string) => words(text).join(" ");
    const trained = new Set(TRAIN_QUERIES.map((entry) => normalise(entry.text)));
    for (const held of HELD_OUT_QUERIES) expect(trained.has(normalise(held.text)), held.text).toBe(false);
    const run = train({ batch: 32, temperature: 0.1, negatives: "in-batch", skipDuplicates: false, dim: 8, seed: 1 }, { steps: 20 });
    for (const snapshot of run.snapshots) for (const row of snapshot.batch.rows) expect(row).toBeLessThan(TRAIN_QUERIES.length);
    expect(new Set(HELD_OUT_QUERIES.map((entry) => entry.id)).size).toBe(30);
    expect(TRAIN_QUERIES).toHaveLength(90);
    expect(PASSAGES).toHaveLength(41);
  });

  it("gives each unseen-word query a word that no training query or passage contains", () => {
    const vocabulary = new Set([...TRAIN_QUERIES.map((entry) => entry.text), ...PASSAGES.map((entry) => entry.text)].flatMap(words));
    for (const held of HELD_OUT_QUERIES.filter((entry) => entry.kind === "unseen")) {
      expect(words(held.text).some((word) => !vocabulary.has(word)), held.text).toBe(true);
    }
  });

  it("is deterministic, draws distinct rows, and snapshots every 10 steps", () => {
    const config = { batch: 16, temperature: 0.1, negatives: "nearest" as const, skipDuplicates: false, dim: 8, seed: 5 };
    const first = train(config, { steps: 50 });
    const second = train(config, { steps: 50 });
    expect(second.losses).toEqual(first.losses);
    expect(first.snapshots.map((snapshot) => snapshot.step)).toEqual([0, 10, 20, 30, 40, 50]);
    for (const snapshot of first.snapshots) expect(new Set(snapshot.batch.rows).size).toBe(16);
    expect(SNAPSHOT_EVERY).toBe(10);
    expect(REFRESH_EVERY).toBe(10);
    expect(STEPS).toBe(300);
  });

  it("finishes the worst case, batch 32 with Nearest wrong, well under a second", () => {
    const started = performance.now();
    const run = train({ batch: 32, temperature: 0.02, negatives: "nearest", skipDuplicates: false, dim: 16, seed: 20 });
    expect(performance.now() - started).toBeLessThan(1000);
    expect(run.losses.every(Number.isFinite)).toBe(true);
  });
});
