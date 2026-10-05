import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import { DEFAULT_CODES, ROWS, encodeRow, euclid, neighborOrder, type Apartment } from "./encode";
import {
  CLEAN,
  DEFAULT_PIXELS,
  MISSING_ROW,
  ONE_HOT_GAP,
  SENTENCES,
  TEMPLATES,
  distanceToQuery,
  encodeSentence,
  idGap,
  leaveOneOut,
  messyRows,
  nearestAnswer,
  parsePixels,
  pixelVector,
  summarize,
  templateDistances,
  togglePixel,
} from "./preview";

const round = (value: number, digits = 2) => Number(value.toFixed(digits));
const messed = (patch: Partial<typeof CLEAN>) => messyRows({ ...CLEAN, ...patch });

describe("pixels as numbers", () => {
  it("validates the drawing and falls back to the plus", () => {
    expect(DEFAULT_PIXELS).toBe("010111010");
    for (const bad of [undefined, 5, "", "01011101", "0101110101", "01x111010", ["010111010"]]) {
      expect(parsePixels(bad)).toBe(DEFAULT_PIXELS);
    }
    expect(parsePixels("111000111")).toBe("111000111");
  });

  it("reads a drawing as nine numbers, row by row, and flips exactly one pixel per press", () => {
    expect(pixelVector("010111010")).toEqual([0, 1, 0, 1, 1, 1, 0, 1, 0]);
    expect(togglePixel("010111010", 0)).toBe("110111010");
    expect(togglePixel("010111010", 4)).toBe("010101010");
    expect(togglePixel(togglePixel("010111010", 8), 8)).toBe("010111010");
    expect(togglePixel("010111010", 9)).toBe("010111010");
    expect(togglePixel("010111010", -1)).toBe("010111010");
    expect(togglePixel("010111010", 1.5)).toBe("010111010");
  });

  it("puts the plus at 0, the bar at √2, and the ring at √5 from the default drawing", () => {
    expect(TEMPLATES.map((template) => template.id)).toEqual(["plus", "ring", "bar"]);
    const ranks = templateDistances("010111010");
    expect(ranks.map((item) => item.id)).toEqual(["plus", "bar", "ring"]);
    expect(ranks[0].distance).toBe(0);
    expect(ranks[1].distance).toBeCloseTo(Math.SQRT2, 12);
    expect(ranks[2].distance).toBeCloseTo(Math.sqrt(5), 12);
    expect(ranks.map((item) => round(item.distance))).toEqual([0, 1.41, 2.24]);
  });

  it("turns the plus into the bar by switching off both arms: the bar is 0.00 away and the plus 1.41", () => {
    const bar = togglePixel(togglePixel("010111010", 3), 5);
    expect(bar).toBe("010010010");
    const ranks = templateDistances(bar);
    expect(ranks.map((item) => [item.id, round(item.distance)])).toEqual([
      ["bar", 0],
      ["plus", 1.41],
      ["ring", 2.65],
    ]);
  });

  it("makes one flipped pixel change the squared distance by exactly 1", () => {
    for (let index = 0; index < 9; index += 1) {
      const flipped = templateDistances(togglePixel("010111010", index)).find((item) => item.id === "plus")!;
      expect(flipped.distance ** 2).toBeCloseTo(1, 12);
    }
  });

  it("keeps ties in template order", () => {
    const ranks = templateDistances("010110010");
    expect(ranks[0].distance).toBeCloseTo(1, 12);
    expect(ranks[1].distance).toBeCloseTo(1, 12);
    expect(ranks.map((item) => item.id).slice(0, 2)).toEqual(["plus", "bar"]);
  });
});

describe("a sentence as IDs", () => {
  it("looks each word up in the fixed vocabulary", () => {
    expect(encodeSentence("the cat sat on the mat").map((token) => token.id)).toEqual([1, 2, 3, 4, 1, 5]);
    expect(encodeSentence("the dog ran").map((token) => token.id)).toEqual([1, 6, 7]);
    expect(encodeSentence("the cat ran on the mat").map((token) => token.id)).toEqual([1, 2, 7, 4, 1, 5]);
  });

  it("sends a word that is not on the list to ID 0 and keeps the sentence length", () => {
    const tokens = encodeSentence("the bird sat on the mat");
    expect(tokens).toHaveLength(6);
    expect(tokens.map((token) => token.id)).toEqual([1, 0, 3, 4, 1, 5]);
    expect(tokens.filter((token) => !token.known).map((token) => token.word)).toEqual(["bird"]);
    expect(encodeSentence("zebra  Cobra").map((token) => token.id)).toEqual([0, 0]);
    expect(encodeSentence("")).toEqual([]);
  });

  it("makes cat nearer to mat (gap 3) than to dog (gap 4) by subtraction, while one-hot gives every pair √2", () => {
    expect(idGap("cat", "dog")).toBe(4);
    expect(idGap("cat", "mat")).toBe(3);
    expect(idGap("dog", "mat")).toBe(1);
    const oneHot = (id: number) => Array.from({ length: 8 }, (_, slot) => (slot === id ? 1 : 0));
    expect(euclid(oneHot(2), oneHot(6))).toBeCloseTo(ONE_HOT_GAP, 12);
    expect(euclid(oneHot(2), oneHot(5))).toBeCloseTo(ONE_HOT_GAP, 12);
    expect(round(ONE_HOT_GAP)).toBe(1.41);
    expect(SENTENCES.map((sentence) => sentence.id)).toEqual(["cat-mat", "dog-ran", "cat-ran", "bird-mat"]);
  });
});

describe("messy rows", () => {
  it("is the clean street when nothing is switched on: 6 rows, mean rooms 13/6, half of them high-priced", () => {
    const rows = messed({});
    expect(rows).toHaveLength(6);
    expect(rows.every((row) => row.flag === "" && !row.dropped)).toBe(true);
    const summary = summarize(rows);
    expect(summary.count).toBe(6);
    expect(summary.meanRooms).toBeCloseTo(13 / 6, 12);
    expect(round(summary.meanRooms)).toBe(2.17);
    expect(summary.highShare).toBe(0.5);
    const answer = nearestAnswer(rows);
    expect(answer.neighbors.map((item) => item.row.name)).toEqual(["Dock 2", "Alder 12", "Elm 7"]);
    expect(answer.neighbors.map((item) => round(item.distance))).toEqual([1, 1.41, 1.73]);
    expect(answer.high).toBe(2);
    expect(answer.answer).toBe(1);
    const loo = leaveOneOut(rows);
    expect([loo.correct, loo.total, round(loo.accuracy * 100, 1)]).toEqual([3, 6, 50]);
  });

  it("matches the integer-stamp distances used on the street cards", () => {
    const features = ROWS.map((row) => encodeRow(row, "integer", DEFAULT_CODES));
    const query = [3, 1, 0];
    const streetOrder = ROWS.map((_, id) => ({ id, d: euclid(features[id], query) }))
      .sort((a, b) => a.d - b.d || a.id - b.id)
      .slice(0, 3)
      .map((item) => ROWS[item.id].name);
    expect(streetOrder).toEqual(["Dock 2", "Alder 12", "Elm 7"]);
    expect(neighborOrder(features, 0)[0].id).toBe(4);
  });

  it("reads a blank room count as 0: Dock 2 moves from 1.00 to 3.00 away, mean rooms falls to 1.5, and the answer flips to low", () => {
    const rows = messed({ missing: true, handling: "zero" });
    expect(rows[MISSING_ROW]).toMatchObject({ name: "Dock 2", rooms: 0, flag: "rooms missing, read as 0" });
    expect(distanceToQuery(messyRows(CLEAN)[MISSING_ROW])).toBe(1);
    expect(distanceToQuery(rows[MISSING_ROW])).toBe(3);
    expect(summarize(rows)).toMatchObject({ count: 6, meanRooms: 1.5, highShare: 0.5 });
    const answer = nearestAnswer(rows);
    expect(answer.neighbors.map((item) => item.row.name)).toEqual(["Alder 12", "Elm 7", "Birch 4"]);
    expect(answer.answer).toBe(0);
    expect(leaveOneOut(rows).correct).toBe(2);
  });

  it("drops the row: 5 rows, mean rooms 1.8, 40 percent high-priced, and the answer is still low", () => {
    const rows = messed({ missing: true, handling: "drop" });
    expect(rows).toHaveLength(6);
    expect(rows[MISSING_ROW].dropped).toBe(true);
    expect(summarize(rows)).toMatchObject({ count: 5, meanRooms: 1.8, highShare: 0.4 });
    expect(nearestAnswer(rows).neighbors.map((item) => item.row.name)).not.toContain("Dock 2");
    expect(nearestAnswer(rows).answer).toBe(0);
    expect(leaveOneOut(rows)).toMatchObject({ correct: 2, total: 5 });
  });

  it("fills with the mean of the other five rooms, 1.8: Dock 2 stays 1.20 away and the answer stays high", () => {
    const rows = messed({ missing: true, handling: "mean" });
    expect(rows[MISSING_ROW].rooms).toBeCloseTo(1.8, 12);
    expect(rows[MISSING_ROW].flag).toBe("rooms missing, filled with 1.8");
    expect(distanceToQuery(rows[MISSING_ROW])).toBeCloseTo(1.2, 12);
    expect(summarize(rows).meanRooms).toBeCloseTo(1.8, 12);
    expect(nearestAnswer(rows).answer).toBe(1);
    expect(nearestAnswer(rows).neighbors[0].row.name).toBe("Dock 2");
  });

  it("lists Alder 12 twice: 7 rows, 57.1 percent high-priced, and leave-one-out accuracy rises from 3 of 6 to 5 of 7", () => {
    const rows = messed({ duplicate: true });
    expect(rows).toHaveLength(7);
    expect(rows[6]).toMatchObject({ name: "Alder 12", flag: "duplicate of Alder 12" });
    const summary = summarize(rows);
    expect(summary.count).toBe(7);
    expect(summary.meanRooms).toBeCloseTo(15 / 7, 12);
    expect(round(summary.highShare * 100, 1)).toBe(57.1);
    const loo = leaveOneOut(rows);
    expect([loo.correct, loo.total, round(loo.accuracy * 100, 1)]).toEqual([5, 7, 71.4]);
    expect(leaveOneOut(messed({})).accuracy).toBeLessThan(loo.accuracy);
    expect(nearestAnswer(rows).answer).toBe(1);
  });

  it("flips Dock 2's tag: 33.3 percent high-priced, and the nearest listing now votes the other way", () => {
    const rows = messed({ mislabel: true });
    expect(rows[3]).toMatchObject({ name: "Dock 2", price: 0, flag: "tag flipped from high to low" });
    expect(round(summarize(rows).highShare * 100, 1)).toBe(33.3);
    expect(nearestAnswer(rows).neighbors.map((item) => [item.row.name, item.row.price])).toEqual([
      ["Dock 2", 0],
      ["Alder 12", 1],
      ["Elm 7", 0],
    ]);
    expect(nearestAnswer(rows).answer).toBe(0);
  });

  it("composes: every option on at once still returns finite numbers", () => {
    const rows = messed({ missing: true, duplicate: true, mislabel: true, handling: "zero" });
    expect(rows).toHaveLength(7);
    expect(rows[MISSING_ROW].flag).toContain("rooms missing, read as 0");
    expect(rows[MISSING_ROW].flag).toContain("tag flipped");
    const summary = summarize(rows);
    expect(Number.isFinite(summary.meanRooms)).toBe(true);
    expect(Number.isFinite(leaveOneOut(rows).accuracy)).toBe(true);
  });
});

describe("why distances flip: Alder 12 and Elm 7 under two encodings", () => {
  const alder = ROWS[0];
  const elm = ROWS[4];
  const cedar = ROWS[2];
  const distance = (a: Apartment, b: Apartment, encoding: "integer" | "onehot", codes = DEFAULT_CODES) =>
    euclid(encodeRow(a, encoding, codes), encodeRow(b, encoding, codes));

  it("puts Elm 7 at 1.00 on the default stamps, 4.00 once blue is stamped 5, and √2 under one-hot", () => {
    expect(distance(alder, elm, "integer")).toBeCloseTo(1, 12);
    expect(distance(alder, elm, "integer", { ...DEFAULT_CODES, blue: 5 })).toBeCloseTo(4, 12);
    expect(distance(alder, elm, "onehot")).toBeCloseTo(Math.SQRT2, 12);
    expect(distance(alder, elm, "onehot", { ...DEFAULT_CODES, blue: 5 })).toBeCloseTo(Math.SQRT2, 12);
    expect(round(distance(alder, elm, "onehot"))).toBe(1.41);
  });

  it("holds Cedar 9 at √5 ≈ 2.24 from Alder 12 whatever blue is stamped, so Elm 7 and Cedar 9 swap places", () => {
    const stamped = { ...DEFAULT_CODES, blue: 5 };
    expect(distance(alder, cedar, "integer")).toBeCloseTo(Math.sqrt(5), 12);
    expect(distance(alder, cedar, "integer", stamped)).toBeCloseTo(Math.sqrt(5), 12);
    expect(round(distance(alder, cedar, "integer"))).toBe(2.24);
    expect(distance(alder, elm, "integer")).toBeLessThan(distance(alder, cedar, "integer"));
    expect(distance(alder, elm, "integer", stamped)).toBeGreaterThan(distance(alder, cedar, "integer", stamped));
  });
});

describe("state, version 4", () => {
  const { hydrateState, initialState, serializeState } = definition;

  it("round-trips the defaults and bumps the version", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(definition.stateVersion).toBe(4);
    expect(initialState).toMatchObject({
      pixels: "010111010",
      sentence: "cat-mat",
      messMissing: false,
      messDuplicate: false,
      messMislabel: false,
      missingHandling: "zero",
    });
  });

  it("reads a version 3 payload, which has none of the new keys, and keeps the street choices", () => {
    const version3 = JSON.stringify({
      encoding: "onehot",
      redCode: 2,
      blueCode: 5,
      greenCode: 3,
      selected: 3,
      featureAxis: "color",
      threshold: 1.5,
    });
    expect(hydrateState(version3)).toEqual({
      ...initialState,
      encoding: "onehot",
      redCode: 2,
      blueCode: 5,
      greenCode: 3,
      selected: 3,
      featureAxis: "color",
      threshold: 1.5,
    });
  });

  it("validates every new key and drops unknown ones", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        pixels: "2222",
        sentence: "elephant",
        messMissing: "yes",
        messDuplicate: 1,
        messMislabel: true,
        missingHandling: "guess",
        extra: "ignored",
      }),
    );
    expect(hydrated).toEqual({ ...initialState, messMislabel: true });
    expect(hydrateState(JSON.stringify({ pixels: "111000111", sentence: "bird-mat", missingHandling: "mean", messMissing: true }))).toMatchObject({
      pixels: "111000111",
      sentence: "bird-mat",
      missingHandling: "mean",
      messMissing: true,
    });
  });
});

describe("the new cards render", () => {
  const render = (patch: Record<string, string | number | boolean>, currentStep = 0) =>
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
  const cases: [string, Record<string, string | number | boolean>][] = [
    ["defaults", {}],
    ["blank picture", { pixels: "000000000" }],
    ["full picture", { pixels: "111111111" }],
    ["bird sentence", { sentence: "bird-mat" }],
    ["missing, zero", { messMissing: true, missingHandling: "zero" }],
    ["missing, drop", { messMissing: true, missingHandling: "drop" }],
    ["missing, mean", { messMissing: true, missingHandling: "mean" }],
    ["duplicate", { messDuplicate: true }],
    ["mislabel", { messMislabel: true }],
    ["everything", { messMissing: true, messDuplicate: true, messMislabel: true, missingHandling: "drop" }],
  ];
  it.each(cases)("%s prints no broken value at any step", (_name, patch) => {
    for (let step = 0; step < definition.steps.length; step += 1) {
      const html = render(patch, step);
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
      expect(visible(html)).not.toMatch(/\d\.\d{6,}/);
    }
  });

  it("prints the numbers the lesson quotes", () => {
    const base = visible(render({}));
    expect(base).toContain("0.00");
    expect(base).toContain("1.41");
    expect(base).toContain("2.24");
    expect(base).toContain("[1, 2, 3, 4, 1, 5]");
    expect(base).toContain("3 of 6 · 50.0%");
    expect(base).toContain("2.17");
    expect(visible(render({ sentence: "bird-mat" }))).toContain("[1, 0, 3, 4, 1, 5]");
    expect(visible(render({ messMissing: true, missingHandling: "zero" }))).toContain("1.50");
    expect(visible(render({ messMissing: true, missingHandling: "zero" }))).toContain("changed");
    expect(visible(render({ messMissing: true, missingHandling: "mean" }))).toContain("filled with 1.8");
    expect(visible(render({ messDuplicate: true }))).toContain("5 of 7 · 71.4%");
    expect(visible(render({ messMislabel: true }))).toContain("33.3%");
  });

  it("gives every pixel a text name and a visible value, so ink does not rest on colour alone", () => {
    const html = render({});
    expect(html).toContain('aria-label="Row 1, column 2, value 1"');
    expect(html).toContain('aria-label="Row 1, column 1, value 0"');
    expect(html.match(/aria-pressed="true"/g)!.length).toBeGreaterThanOrEqual(5);
  });

  it("has eight steps, four objectives, and a question for each objective", () => {
    expect(definition.steps).toHaveLength(8);
    expect(definition.stepInstructions).toHaveLength(8);
    expect(definition.objectives).toHaveLength(4);
    const covered = new Set(checkpointQuestions(definition).map((question) => question.objective));
    expect([...covered].sort()).toEqual([0, 1, 2, 3]);
  });
});

describe("the rescaled gaps quoted in the lesson", () => {
  it("shrinks Elm 7's gap to 0.20 and Cedar 9's to 0.47 under same-length rulers", () => {
    const alder = encodeRow(ROWS[0], "normalized", DEFAULT_CODES);
    expect(round(euclid(alder, encodeRow(ROWS[4], "normalized", DEFAULT_CODES)))).toBe(0.2);
    expect(round(euclid(alder, encodeRow(ROWS[2], "normalized", DEFAULT_CODES)))).toBe(0.47);
  });
});
