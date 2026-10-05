import { describe, expect, it } from "vitest";
import { APARTMENTS, runFit } from "./fit";

/** Pins the numbers the lesson and the checkpoint rely on. */
const last = <T>(items: T[]) => items[items.length - 1];
const round = (value: number, digits = 2) => Number(value.toFixed(digits));

describe("supervised-loop quoted values", () => {
  it("lets a depth-5 tree on integer codes take train loss to 0 while holdout log-loss climbs from 0.48 to about 10", () => {
    const path = runFit("integer", "tree", "logloss", 5);
    expect(round(path[1].holdoutLoss)).toBe(0.48);
    expect(path[5].trainLoss).toBeLessThan(1e-6);
    expect(round(path[5].holdoutLoss, 1)).toBe(10.4);
  });

  it("changing only the encoding on logistic + log-loss lowers train loss and raises holdout loss as color enters x", () => {
    const fit = (encoding: "numeric" | "integer" | "onehot") => last(runFit(encoding, "logistic", "logloss", 2));
    const plain = fit("numeric");
    const integer = fit("integer");
    const oneHot = fit("onehot");
    expect([round(plain.trainLoss), round(plain.holdoutLoss)]).toEqual([0.45, 0.31]);
    expect([round(oneHot.trainLoss), round(oneHot.holdoutLoss)]).toEqual([0.26, 0.75]);
    expect(plain.trainLoss).toBeGreaterThan(integer.trainLoss);
    expect(integer.trainLoss).toBeGreaterThan(oneHot.trainLoss);
    expect(plain.holdoutLoss).toBeLessThan(integer.holdoutLoss);
    expect(integer.holdoutLoss).toBeLessThan(oneHot.holdoutLoss);
  });

  it("makes color a spurious cue: 4 of 5 red train rows are high-priced and neither red holdout row is", () => {
    const red = (split: string) => APARTMENTS.filter((row) => row.color === "red" && row.split === split);
    expect(red("train")).toHaveLength(5);
    expect(red("train").filter((row) => row.price === 1)).toHaveLength(4);
    expect(red("holdout")).toHaveLength(2);
    expect(red("holdout").filter((row) => row.price === 1)).toHaveLength(0);
  });

  it("reads the same fit on a different scale when only the loss changes", () => {
    const logloss = last(runFit("numeric", "logistic", "logloss", 2));
    const mse = last(runFit("numeric", "logistic", "mse", 2));
    expect(mse.holdoutLoss).not.toBeCloseTo(logloss.holdoutLoss, 2);
    expect(mse.trainLoss).not.toBeCloseTo(logloss.trainLoss, 2);
  });

  it("cannot memorize Rowan 9 on rooms + park, because it shares the vector (3, 1) with Cedar 9", () => {
    const rowan = APARTMENTS.find((row) => row.name === "Rowan 9")!;
    const cedar = APARTMENTS.find((row) => row.name === "Cedar 9")!;
    expect([rowan.rooms, rowan.park]).toEqual([cedar.rooms, cedar.park]);
    expect(rowan.price).not.toBe(cedar.price);
    expect(last(runFit("numeric", "tree", "logloss", 5)).trainLoss).toBeGreaterThan(0.05);
  });
});
