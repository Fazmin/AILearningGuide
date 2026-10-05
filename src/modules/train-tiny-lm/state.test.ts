import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import {
  BATCH_LADDER,
  CLIP_MAX,
  EPOCHS_MAX,
  EPOCHS_MIN,
  RANK_MAX,
  RANK_MIN,
  RATE_LADDER,
  readSettings,
  sanitizeState,
} from "./state";

const hydrate = (value: string) => definition.hydrateState(value);
const roundTrip = (state: ModuleState) => hydrate(definition.serializeState(state));
const renderLab = (state: ModuleState) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state,
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const text = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const broken = /\bNaN\b|\bundefined\b|\bInfinity\b/;

const KEYS = ["batchSize", "clipNorm", "corpusId", "data", "epochs", "learningRate", "rank", "schedule", "wizard"];

function expectBounded(state: ModuleState) {
  expect(Object.keys(state).sort()).toEqual(KEYS);
  expect(Number.isInteger(state.rank)).toBe(true);
  expect(state.rank as number).toBeGreaterThanOrEqual(RANK_MIN);
  expect(state.rank as number).toBeLessThanOrEqual(RANK_MAX);
  expect(Number.isInteger(state.epochs)).toBe(true);
  expect(state.epochs as number).toBeGreaterThanOrEqual(EPOCHS_MIN);
  expect(state.epochs as number).toBeLessThanOrEqual(EPOCHS_MAX);
  expect(RATE_LADDER as readonly number[]).toContain(state.learningRate);
  expect(BATCH_LADDER as readonly number[]).toContain(state.batchSize);
  expect(state.clipNorm as number).toBeGreaterThanOrEqual(0);
  expect(state.clipNorm as number).toBeLessThanOrEqual(CLIP_MAX);
  expect(["constant", "cosine", "warmup-decay", "one-cycle"]).toContain(state.schedule);
  expect(["cleaned", "raw"]).toContain(state.data);
  expect(["harbor", "recipes", "proverbs"]).toContain(state.corpusId);
  expect(state.wizard as number).toBeGreaterThanOrEqual(0);
  expect(state.wizard as number).toBeLessThanOrEqual(3);
}

describe("train-tiny-lm state", () => {
  it("keeps the defaults the lab used before the recipe card: rate 0.5, batch 16, constant, no clip", () => {
    expect(definition.initialState).toEqual({
      wizard: 0,
      corpusId: "harbor",
      rank: 8,
      epochs: 20,
      learningRate: 0.5,
      schedule: "constant",
      batchSize: 16,
      clipNorm: 0,
      data: "cleaned",
    });
    expect(hydrate("{}")).toEqual(definition.initialState);
    expect(definition.stateVersion).toBe(2);
  });

  it("translates a version 1 payload by keeping its four settings and adding the old fixed recipe", () => {
    const version1 = JSON.stringify({ wizard: 2, corpusId: "proverbs", rank: 12, epochs: 30 });
    expect(hydrate(version1)).toEqual({
      wizard: 2,
      corpusId: "proverbs",
      rank: 12,
      epochs: 30,
      learningRate: 0.5,
      schedule: "constant",
      batchSize: 16,
      clipNorm: 0,
      data: "cleaned",
    });
  });

  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [
      definition.initialState,
      { wizard: 3, corpusId: "recipes", rank: 1, epochs: 1, learningRate: 0.125, schedule: "cosine", batchSize: 2, clipNorm: 0.05, data: "raw" },
      { wizard: 2, corpusId: "proverbs", rank: 12, epochs: 30, learningRate: 16, schedule: "one-cycle", batchSize: 64, clipNorm: 1.5, data: "raw" },
      { wizard: 1, corpusId: "harbor", rank: 5, epochs: 17, learningRate: 4, schedule: "warmup-decay", batchSize: 8, clipNorm: 0.35, data: "cleaned" },
    ];
    for (const state of states) expect(roundTrip(state)).toEqual(state);
  });

  it("clamps numbers, snaps rates and batches to their notches, and falls back on anything invalid", () => {
    const wild = hydrate(
      JSON.stringify({
        wizard: 99,
        corpusId: "<script>",
        rank: 1e9,
        epochs: -1e9,
        learningRate: 3,
        schedule: "sawtooth",
        batchSize: 1000,
        clipNorm: 40,
        data: "everything",
        injected: "dropped",
      }),
    );
    expect(wild).toEqual({
      wizard: 3,
      corpusId: "harbor",
      rank: 12,
      epochs: 1,
      learningRate: 4,
      schedule: "constant",
      batchSize: 64,
      clipNorm: 1.5,
      data: "cleaned",
    });
    expectBounded(hydrate(JSON.stringify({ rank: "8", epochs: null, learningRate: "fast", clipNorm: [1] })));
    expect(hydrate("[1,2,3]")).toEqual(definition.initialState);
    expect(hydrate("null")).toEqual(definition.initialState);
    expect(hydrate("not json")).toEqual(definition.initialState);
  });

  it("stays bounded for 1e9, -1e9 and Infinity in every numeric key", () => {
    for (const value of ["1e9", "-1e9", "1e999", "-1e999"]) {
      const numeric = Object.keys(definition.initialState).filter((key) => typeof definition.initialState[key] === "number");
      const payload = `{${numeric.map((key) => `"${key}":${value}`).join(",")}}`;
      expectBounded(hydrate(payload));
    }
  });

  it("clips to two decimals so the slider's own values survive the floating-point round trip", () => {
    for (let notch = 0; notch <= 30; notch += 1) {
      const clipNorm = Number((notch * 0.05).toFixed(2));
      expect(sanitizeState({ clipNorm }).clipNorm).toBe(clipNorm);
    }
    expect(readSettings({ clipNorm: 0.37 }).clipNorm).toBeCloseTo(0.35, 10);
  });

  it("renders every extreme the controls allow without printing a broken value", { timeout: 60_000 }, () => {
    const extremes: ModuleState[] = [];
    for (const learningRate of [RATE_LADDER[0], RATE_LADDER[RATE_LADDER.length - 1]]) {
      for (const batchSize of [BATCH_LADDER[0], BATCH_LADDER[BATCH_LADDER.length - 1]]) {
        for (const data of ["cleaned", "raw"]) {
          extremes.push({ ...definition.initialState, learningRate, batchSize, data, rank: 12, epochs: 30 });
        }
      }
    }
    for (const schedule of ["constant", "cosine", "warmup-decay", "one-cycle"]) {
      for (const clipNorm of [0, 0.5, CLIP_MAX]) extremes.push({ ...definition.initialState, learningRate: 16, schedule, clipNorm });
    }
    for (const state of extremes) {
      const html = text(renderLab(state));
      expect(html, JSON.stringify(state)).not.toMatch(broken);
      expect(html, JSON.stringify(state)).not.toMatch(/\d\.\d{6,}/);
    }
  });

  it("says a run diverged instead of printing its loss, and a clip lets the same run finish", () => {
    const diverging = { ...definition.initialState, learningRate: 8, schedule: "constant", clipNorm: 0 };
    const stopped = text(renderLab(diverging));
    expect(stopped).toContain("This run diverged at step 154");
    expect(stopped).toContain("diverged");
    const clipped = text(renderLab({ ...diverging, clipNorm: 0.5 }));
    expect(clipped).not.toContain("This run diverged");
    expect(clipped).toContain("459 of 480");
  });

  it("renders the zeroed and emptied payloads the registry feeds it", () => {
    const zeroed = Object.fromEntries(
      Object.entries(definition.initialState).map(([key, value]) => [key, typeof value === "number" ? 0 : ""]),
    ) as ModuleState;
    expect(text(renderLab(zeroed))).not.toMatch(broken);
  });
});
