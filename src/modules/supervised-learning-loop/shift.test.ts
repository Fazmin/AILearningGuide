import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import { APARTMENTS, SHIFTED_HOLDOUT, apartmentsFor, runFit } from "./fit";
import { choose, holdoutMix } from "./split";

const last = <T>(items: T[]) => items[items.length - 1];
const round = (value: number, digits = 2) => Number(value.toFixed(digits));
const usualRule = (row: { rooms: number; park: number }) => (row.rooms >= 3 || (row.rooms === 2 && row.park === 1) ? 1 : 0);

describe("the different-process holdout", () => {
  it("replaces only the four holdout rows, one for one, and leaves the twelve train rows alone", () => {
    const same = apartmentsFor("same");
    const shifted = apartmentsFor("shifted");
    expect(same).toBe(APARTMENTS);
    expect(shifted).toHaveLength(16);
    expect(shifted.slice(0, 12)).toEqual(same.slice(0, 12));
    expect(shifted.slice(12).map((row) => row.id)).toEqual([12, 13, 14, 15]);
    expect(shifted.slice(12).every((row) => row.split === "holdout")).toBe(true);
    expect(new Set(shifted.map((row) => row.name)).size).toBe(16);
  });

  it("labels the shifted rows by a reversed rule (high price when rooms ≤ 2), which the usual rule gets wrong on three of four", () => {
    for (const row of SHIFTED_HOLDOUT) expect(row.price).toBe(row.rooms <= 2 ? 1 : 0);
    expect(SHIFTED_HOLDOUT.filter((row) => usualRule(row) !== row.price)).toHaveLength(3);
    for (const row of APARTMENTS.filter((item) => item.split === "holdout")) expect(usualRule(row)).toBe(row.price);
  });

  it("collapses the same fit: holdout accuracy 100% to 25% and log-loss 0.31 to 1.39, with train loss unchanged", () => {
    const same = last(runFit("numeric", "logistic", "logloss", 2, "same"));
    const shifted = last(runFit("numeric", "logistic", "logloss", 2, "shifted"));
    expect(shifted.trainLoss).toBe(same.trainLoss);
    expect(round(same.trainLoss)).toBe(0.45);
    expect([same.holdoutAccuracy, shifted.holdoutAccuracy]).toEqual([1, 0.25]);
    expect([round(same.holdoutLoss), round(shifted.holdoutLoss)]).toEqual([0.31, 1.39]);
    expect(round(shifted.holdoutLoss / same.holdoutLoss, 1)).toBe(4.5);
    expect(shifted.holdoutLoss).toBeGreaterThan(Math.LN2);
    expect(round(Math.LN2)).toBe(0.69);
  });

  it("collapses every encoding, algorithm, and loss: the shifted holdout is never better than the same-process one", () => {
    for (const encoding of ["numeric", "integer", "onehot"] as const) {
      for (const algorithm of ["linear", "logistic", "tree"] as const) {
        for (const loss of ["logloss", "mse"] as const) {
          const same = last(runFit(encoding, algorithm, loss, 2, "same"));
          const shifted = last(runFit(encoding, algorithm, loss, 2, "shifted"));
          const label = `${encoding}/${algorithm}/${loss}`;
          expect(shifted.holdoutLoss, label).toBeGreaterThan(same.holdoutLoss);
          expect(shifted.holdoutAccuracy, label).toBeLessThanOrEqual(same.holdoutAccuracy);
          expect(shifted.trainLoss, label).toBe(same.trainLoss);
        }
      }
    }
  });

  it("defaults to the same process and reads a version 1 payload as same-process", () => {
    expect(definition.initialState.holdoutSource).toBe("same");
    expect(definition.stateVersion).toBe(2);
    const version1 = JSON.stringify({ stage: 4, encoding: "onehot", algorithm: "tree", loss: "mse", step: 3, depth: 3, selected: 5, lastKnob: "depth" });
    expect(definition.hydrateState(version1)).toMatchObject({
      encoding: "onehot",
      algorithm: "tree",
      depth: 3,
      selected: 5,
      lastKnob: "depth",
      holdoutSource: "same",
    });
    expect(definition.hydrateState(JSON.stringify({ holdoutSource: "shifted", lastKnob: "holdout" }))).toMatchObject({
      holdoutSource: "shifted",
      lastKnob: "holdout",
    });
    expect(definition.hydrateState(JSON.stringify({ holdoutSource: "elsewhere", lastKnob: 7 }))).toMatchObject({
      holdoutSource: "same",
      lastKnob: "algorithm",
    });
  });
});

describe("the worked split", () => {
  it("has 8 high-priced and 8 low-priced apartments: 6 and 6 in train, 2 and 2 in holdout", () => {
    const count = (split: string, price: number) => APARTMENTS.filter((row) => row.split === split && row.price === price).length;
    expect([count("train", 1), count("train", 0), count("holdout", 1), count("holdout", 0)]).toEqual([6, 6, 2, 2]);
    expect(APARTMENTS.filter((row) => row.price === 1)).toHaveLength(8);
  });

  it("counts 1,820 ways to hold out 4 of 16 rows: 140 all one label, 784 two and two, 896 three and one", () => {
    expect(choose(16, 4)).toBe(1820);
    const mix = holdoutMix(8, 8, 4);
    expect(mix.map((item) => item.ways)).toEqual([70, 448, 784, 448, 70]);
    expect(mix.reduce((sum, item) => sum + item.ways, 0)).toBe(1820);
    const allOneSide = mix[0].ways + mix[4].ways;
    const mixed = mix[1].ways + mix[3].ways;
    expect([allOneSide, mix[2].ways, mixed]).toEqual([140, 784, 896]);
    expect([allOneSide, mix[2].ways, mixed].map((ways) => round((ways / 1820) * 100, 1))).toEqual([7.7, 43.1, 49.2]);
  });

  it("returns 0 for impossible draws", () => {
    expect(choose(4, 5)).toBe(0);
    expect(choose(4, -1)).toBe(0);
    expect(choose(0, 0)).toBe(1);
  });
});

describe("the lab renders both holdouts", () => {
  const render = (patch: Record<string, string | number>, currentStep = 0) =>
    renderToStaticMarkup(
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...patch },
        setState: () => undefined,
        currentStep,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    );
  const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

  it("prints the collapse the lesson quotes, at every step", () => {
    for (const holdoutSource of ["same", "shifted"]) {
      for (let step = 0; step < definition.steps.length; step += 1) {
        const html = render({ holdoutSource }, step);
        expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
        expect(visible(html)).not.toMatch(/\d\.\d{6,}/);
      }
    }
    const text = visible(render({ holdoutSource: "shifted" }));
    expect(text).toContain("Quay 3");
    expect(text).toContain("1.392");
    expect(text).toContain("0.308");
    expect(text).toContain("4.5×");
    expect(text).toContain("-75 pts");
    expect(text).toContain("1 of 4");
    expect(text).toContain("4 of 4");
  });

  it("covers all four objectives with questions and keeps seven steps", () => {
    expect(definition.steps).toHaveLength(7);
    expect(definition.stepInstructions).toHaveLength(7);
    expect(definition.objectives).toHaveLength(4);
    const covered = new Set(checkpointQuestions(definition).map((question) => question.objective));
    expect([...covered].sort()).toEqual([0, 1, 2, 3]);
  });
});
