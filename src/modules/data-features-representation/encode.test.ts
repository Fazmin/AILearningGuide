import { describe, expect, it } from "vitest";
import {
  DEFAULT_CODES,
  ROWS,
  axisValue,
  encodeRow,
  euclid,
  neighborOrder,
  nearestTies,
  planeNeighbor,
} from "./encode";

const alder = ROWS[0];
const elm = ROWS[4];

describe("encodeRow", () => {
  it("writes integer codes as [rooms, color, park]", () => {
    expect(encodeRow(alder, "integer", DEFAULT_CODES)).toEqual([2, 1, 1]);
  });

  it("writes one-hot color as three exclusive slots and ignores the integer codes", () => {
    expect(encodeRow(alder, "onehot", { red: 5, blue: 5, green: 5 })).toEqual([2, 1, 0, 0, 1]);
  });

  it("rescales rooms by 4 and color codes by 5", () => {
    expect(encodeRow(alder, "normalized", DEFAULT_CODES)).toEqual([0.5, 0.2, 1]);
  });
});

describe("neighbors", () => {
  it("puts Elm nearest to Alder under default integer codes", () => {
    const features = ROWS.map((row) => encodeRow(row, "integer", DEFAULT_CODES));
    expect(neighborOrder(features, 0)[0]?.id).toBe(elm.id);
  });

  it("moves Elm away from Alder when blue is stamped 5", () => {
    const codes = { ...DEFAULT_CODES, blue: 5 };
    const features = ROWS.map((row) => encodeRow(row, "integer", codes));
    expect(neighborOrder(features, 0)[0]?.id).not.toBe(elm.id);
  });

  it("keeps Elm nearest to Alder under one-hot even after the blue stamp changes", () => {
    const features = ROWS.map((row) => encodeRow(row, "onehot", { ...DEFAULT_CODES, blue: 5 }));
    expect(neighborOrder(features, 0)[0]?.id).toBe(elm.id);
    expect(euclid(features[0], features[4])).toBeCloseTo(Math.SQRT2);
  });

  it("can name a different neighbor in the first two coordinates than in the full slip", () => {
    const features = ROWS.map((row) => encodeRow(row, "onehot", DEFAULT_CODES));
    const full = neighborOrder(features, 3)[0];
    const plane = planeNeighbor(features, 3)[0];
    expect(full && plane).toBeTruthy();
    expect(full?.dist).not.toBe(plane?.dist);
  });
});

describe("axisValue", () => {
  it("reads only the red bit when color is one-hot", () => {
    expect(axisValue(alder, "color", "onehot", DEFAULT_CODES)).toBe(1);
    expect(axisValue(elm, "color", "onehot", DEFAULT_CODES)).toBe(0);
  });

  it("uses the stamped integer for color otherwise", () => {
    expect(axisValue(elm, "color", "integer", { ...DEFAULT_CODES, blue: 5 })).toBe(5);
  });
});

describe("ties and scaling", () => {
  it("reports Cedar 9 and Dock 2 as tied nearest to Alder 12 when blue is stamped 5", () => {
    const features = ROWS.map((row) => encodeRow(row, "integer", { ...DEFAULT_CODES, blue: 5 }));
    const tied = nearestTies(neighborOrder(features, 0));
    expect(tied.map((rank) => rank.id)).toEqual([2, 3]);
    expect(tied[0].dist).toBeCloseTo(Math.sqrt(5), 12);
  });

  it("reports a single nearest when there is no tie", () => {
    const features = ROWS.map((row) => encodeRow(row, "integer", DEFAULT_CODES));
    expect(nearestTies(neighborOrder(features, 0)).map((rank) => rank.id)).toEqual([elm.id]);
  });

  it("makes the 0/1 park slot the widest axis once rooms and stamps are rescaled", () => {
    const features = ROWS.map((row) => encodeRow(row, "normalized", DEFAULT_CODES));
    const range = (index: number) =>
      Math.max(...features.map((row) => row[index])) - Math.min(...features.map((row) => row[index]));
    expect(range(2)).toBe(1);
    expect(range(0)).toBeCloseTo(0.75, 12);
    expect(range(1)).toBeCloseTo(0.4, 12);
  });
});
