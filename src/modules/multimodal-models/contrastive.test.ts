import { describe, expect, it } from "vitest";
import { audioFrames, audioPositions, AUDIO, pageImageTokens, videoTokens } from "./modalities";
import {
  boundedTemperature,
  CAPTIONS,
  CHANCE_LOSS,
  contrast,
  contrastFor,
  cosine,
  DEFAULT_TEMPERATURE,
  IMAGES,
  pairingOf,
  POSITIVES,
  SIMILARITY,
  TEMPERATURE_RANGE,
} from "./contrastive";
import { CONTEXT, realScale } from "./patches";

const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);
const pct = (value: number) => Math.round(value * 100);

describe("the similarity matrix", () => {
  it("is cosine similarity of hand-authored vectors, highest on the diagonal", () => {
    expect(SIMILARITY.map((row) => row.map((value) => value.toFixed(2)))).toEqual([
      ["0.97", "0.41", "0.62"],
      ["0.58", "0.98", "0.32"],
      ["0.74", "0.20", "0.93"],
    ]);
    SIMILARITY.forEach((row, i) => expect(row.indexOf(Math.max(...row))).toBe(i));
    expect(cosine(IMAGES[0].vector, IMAGES[0].vector)).toBeCloseTo(1, 12);
    expect(cosine([0, 0], [1, 2])).toBe(0);
    expect(IMAGES).toHaveLength(3);
    expect(CAPTIONS).toHaveLength(3);
  });

  it("keeps the strongest rival of each image below its own caption, with image C closest", () => {
    const margins = SIMILARITY.map((row, i) => row[i] - Math.max(...row.filter((_, j) => j !== i)));
    expect(margins.every((margin) => margin > 0)).toBe(true);
    expect(margins.indexOf(Math.min(...margins))).toBe(2);
    expect(SIMILARITY[2][0].toFixed(2)).toBe("0.74");
    expect(SIMILARITY[2][2].toFixed(2)).toBe("0.93");
  });
});

describe("the contrastive loss", () => {
  it("pins the figures the lesson and card quote", () => {
    expect(CHANCE_LOSS.toFixed(3)).toBe("1.099");
    expect(contrast(0.2, "true").loss.toFixed(3)).toBe("0.235");
    expect(contrast(0.05, "true").loss.toFixed(3)).toBe("0.006");
    expect(contrast(1, "true").loss.toFixed(3)).toBe("0.813");
    expect(contrast(0.2, "shuffled").loss.toFixed(3)).toBe("2.582");
    expect(contrast(0.07, "true").loss.toFixed(3)).toBe("0.022");
    expect(contrast(1, "true").rowShares.map((row, i) => pct(row[i]))).toEqual([44, 46, 43]);
    expect(pct(contrast(0.2, "true").rowShares[2][2])).toBe(70);
    expect(pct(contrast(0.2, "true").rowShares[2][0])).toBe(28);
  });

  it("is the mean of two cross-entropies, and each share list sums to one", () => {
    const result = contrast(0.2, "true");
    expect(result.loss).toBeCloseTo((result.imageToCaption + result.captionToImage) / 2, 12);
    const direct =
      result.rowShares.reduce((total, row, i) => total - Math.log(row[result.positives[i]]), 0) / result.rowShares.length;
    expect(result.imageToCaption).toBeCloseTo(direct, 10);
    result.rowShares.forEach((row) => expect(sum(row)).toBeCloseTo(1, 12));
    for (let j = 0; j < 3; j += 1) expect(sum(result.columnShares.map((row) => row[j]))).toBeCloseTo(1, 12);
  });

  it("falls as temperature drops on true pairs and rises as it drops on shuffled pairs", () => {
    const temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1];
    const trueLosses = temperatures.map((t) => contrast(t, "true").loss);
    const shuffledLosses = temperatures.map((t) => contrast(t, "shuffled").loss);
    for (let i = 1; i < temperatures.length; i += 1) {
      expect(trueLosses[i]).toBeGreaterThan(trueLosses[i - 1]);
      expect(shuffledLosses[i]).toBeLessThan(shuffledLosses[i - 1]);
    }
    expect(trueLosses.every((loss) => loss < CHANCE_LOSS)).toBe(true);
    expect(trueLosses[0]).toBeLessThan(1e-6);
  });

  it("scores far higher when the labels call low-scoring pairs the matches, with the matrix unchanged", () => {
    const right = contrast(0.2, "true");
    const wrong = contrast(0.2, "shuffled");
    expect(wrong.loss).toBeGreaterThan(right.loss * 5);
    expect(wrong.rowShares).toEqual(right.rowShares);
    expect([right.imagesRight, right.captionsRight, wrong.imagesRight, wrong.captionsRight]).toEqual([3, 3, 0, 0]);
    expect(POSITIVES.shuffled).toEqual([1, 2, 0]);
  });

  it("reaches the chance loss when every similarity is equal, at any temperature", () => {
    const flat = [
      [0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
    ];
    for (const tau of [0.01, 0.3, 1]) expect(contrastFor(flat, tau, [0, 1, 2]).loss).toBeCloseTo(CHANCE_LOSS, 10);
  });

  it("has the gradient the card describes: (share − label) / (N·τ), averaged over both directions", () => {
    const tau = 0.2;
    const positives = POSITIVES.true;
    const base = contrastFor(SIMILARITY, tau, positives);
    const step = 1e-6;
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        const bump = SIMILARITY.map((row, a) => row.map((value, b) => (a === i && b === j ? value + step : value)));
        const numeric = (contrastFor(bump, tau, positives).loss - base.loss) / step;
        const label = positives[i] === j ? 1 : 0;
        const analytic = ((base.rowShares[i][j] + base.columnShares[i][j]) / 2 - label) / (3 * tau);
        expect(numeric).toBeCloseTo(analytic, 4);
      }
    }
  });

  it("stays finite at both ends of the slider for either pairing", () => {
    for (const pairing of ["true", "shuffled"] as const) {
      for (const tau of [0.01, 1]) {
        const result = contrast(tau, pairing);
        expect(Number.isFinite(result.loss)).toBe(true);
        expect(result.rowShares.flat().every(Number.isFinite)).toBe(true);
      }
    }
  });
});

describe("control bounds", () => {
  it("clamps temperature to the slider and rounds to hundredths", () => {
    expect([boundedTemperature(1e9), boundedTemperature(-1e9), boundedTemperature(0)]).toEqual([1, 0.01, 0.01]);
    expect(boundedTemperature(Infinity)).toBe(DEFAULT_TEMPERATURE);
    expect(boundedTemperature(Number.NaN)).toBe(DEFAULT_TEMPERATURE);
    expect(boundedTemperature("0.5")).toBe(DEFAULT_TEMPERATURE);
    expect(boundedTemperature(0.234)).toBe(0.23);
    // CLIP clips its learned temperature so that logits are scaled by at most 100, i.e. τ is at least 1/100.
    expect(TEMPERATURE_RANGE.min).toBe(1 / 100);
    expect(pairingOf("shuffled")).toBe("shuffled");
    expect(pairingOf("anything else")).toBe("true");
  });
});

describe("tokens beyond a still image", () => {
  it("computes Whisper's 30-second segment as 3,000 frames and 1,500 positions", () => {
    expect(audioFrames(AUDIO.segmentSeconds, AUDIO.strideMs)).toBe(3000);
    expect(audioPositions(AUDIO.segmentSeconds, AUDIO.strideMs, AUDIO.convStride)).toBe(1500);
    expect(audioPositions(AUDIO.segmentSeconds, AUDIO.strideMs, AUDIO.convStride) / AUDIO.segmentSeconds).toBe(50);
  });

  it("computes a one-minute clip at one frame per second from the lab's own 576-token frame", () => {
    const frame = realScale(336, 14, true, "llava").llmImageTokens;
    expect(frame).toBe(576);
    expect(videoTokens(60, 1, frame)).toBe(34560);
    expect(videoTokens(60, 1, frame) / CONTEXT).toBeGreaterThan(8);
    expect(videoTokens(60, 1, frame) / CONTEXT).toBeLessThan(9);
    expect((videoTokens(60, 1, frame) / CONTEXT).toFixed(1)).toBe("8.4");
  });

  it("computes a 448-pixel page with 14-pixel patches as 1,024 tokens, and a four-times-larger side as sixteen times more", () => {
    expect(pageImageTokens(448, 14)).toBe(1024);
    expect(pageImageTokens(448, 14)).toBe((448 / 14) ** 2);
    expect(pageImageTokens(224, 14)).toBe(256);
  });
});
