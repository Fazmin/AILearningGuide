// @vitest-environment node
/// <reference types="node" />
/**
 * Pins every number the three lessons, the checkpoint, the predict prompts and the card
 * explanations quote from the lab, so a change to the map of words, a preset or a readout cannot
 * silently make the writing wrong.
 *
 * Each figure is checked twice where it can be: against the math helpers, and against the text
 * the lab really prints at the setting the lesson names. The last block checks that all three
 * lessons still state the headline figures.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import {
  FULL_TABLE,
  IDENTITY,
  PIECES,
  ROUNDS,
  add,
  aimMiss,
  aimSolved,
  angleBetween,
  apply,
  closestPair,
  columns,
  compose,
  cosine,
  determinant,
  dot,
  length,
  nearestWord,
  orderGap,
  presetById,
  rotate,
  scale,
  shadowLength,
  squeezeMatrix,
  storedNumbers,
  tableMiss,
  undoSqueeze,
  vecOf,
  wordById,
  type Mat,
} from "./math";
import definition from "./module";

const round =(value: number, digits = 2) => Number(value.toFixed(digits));
const at = (id: string) => vecOf(wordById(id));
/** The lab shows one card per step, so the text of the whole lab is every step's page in turn. */
const text = (patch: Partial<ModuleState> = {}) =>
  definition.steps
    .map((_, currentStep) =>
      renderToStaticMarkup(
        createElement(definition.Explore, {
          state: { ...definition.initialState, ...patch } as ModuleState,
          setState: () => undefined,
          currentStep,
          mode: "plain" as const,
          narrate: () => undefined,
        }),
      ),
    )
    .join("")
    .replace(/<[^>]*>/g, "\u0001")
    .replace(/&amp;/g, "&");
/** The text of one card: from its heading to the next card's kicker. */
const card = (all: string, title: string, next?: string) => {
  const start = all.indexOf(title);
  expect(start, `no card titled ${title}`).toBeGreaterThan(-1);
  const end = next ? all.indexOf(next, start) : all.length;
  return all.slice(start, end > start ? end : all.length);
};
const lesson = (mode: string) =>
  readFileSync(resolve(process.cwd(), "src/modules/vectors-matrices/content", `${mode}.mdx`), "utf8").replace(/\s+/g, " ");

describe("a word as two numbers", () => {
  it("reads horse as (0.9, 1.2) with squares 0.81 and 1.44 and length 1.5", () => {
    expect(at("horse")).toEqual([0.9, 1.2]);
    expect(round(0.9 ** 2)).toBe(0.81);
    expect(round(1.2 ** 2)).toBe(1.44);
    expect(round(0.9 ** 2 + 1.2 ** 2)).toBe(2.25);
    expect(length(at("horse"))).toBeCloseTo(1.5, 12);
    const shown = card(text(), "A word as two numbers", "2 · ");
    expect(shown).toContain("√(0.9² + 1.2²) = √(0.81 + 1.44) = √2.25");
    expect(shown).toContain("1.50");
    expect(shown).toContain("big, living");
    expect(shown).toContain("horse (right on it)");
  });

  it("doubles the length to 3.00 at (1.8, 2.4) without changing the direction", () => {
    expect(length([1.8, 2.4])).toBeCloseTo(3, 12);
    const shown = card(text({ tipX: 1.8, tipY: 2.4 }), "A word as two numbers", "2 · ");
    expect(shown).toContain("3.00");
    expect(shown).toContain("big, living");
    expect(shown).toContain("Put the arrow back on horse");
  });

  it("gives mouse (−1.6, 1.0) a length of 1.89 and the direction small, living", () => {
    expect(at("mouse")).toEqual([-1.6, 1.0]);
    expect(round(length(at("mouse")))).toBe(1.89);
    const shown = card(text({ word: "mouse", tipX: -1.6, tipY: 1 }), "A word as two numbers", "2 · ");
    expect(shown).toContain("1.89");
    expect(shown).toContain("small, living");
  });

  it("puts a round bracket round a negative number before squaring it", () => {
    expect(card(text({ tipX: 1.4, tipY: -1.1 }), "A word as two numbers", "2 · ")).toContain("√(1.4² + (−1.1)²)");
  });
});

describe("add and stretch", () => {
  it("lands horse plus bigger on (1.9, 1.2), 0.32 from whale", () => {
    const result = add(at("horse"), [1, 0]);
    expect(result[0]).toBeCloseTo(1.9, 12);
    expect(result[1]).toBeCloseTo(1.2, 12);
    const near = nearestWord(result);
    expect(near.word.id).toBe("whale");
    expect(round(near.gap)).toBe(0.32);
    const shown = card(text(), "Add and stretch", "3 · ");
    expect(shown).toContain("whale, 0.32 away");
    expect(shown).toContain("× 1.0");
  });

  it("lands horse minus 1.7 × bigger exactly on cat, (−0.8, 1.2)", () => {
    expect(at("cat")).toEqual([-0.8, 1.2]);
    const result = add(at("horse"), scale([1, 0], -1.7));
    expect(result[0]).toBeCloseTo(-0.8, 12);
    expect(result[1]).toBeCloseTo(1.2, 12);
    expect(nearestWord(result).word.id).toBe("cat");
    const shown = card(text({ times: -1.7 }), "Add and stretch", "3 · ");
    expect(shown).toContain("× −1.7");
    expect(shown).toContain("cat (right on it)");
    expect(shown).toContain("− 1.7 × bigger");
    expect(shown).toContain("−0.8");
    expect(shown).toContain("flipped and stretched");
  });

  it("offers the steps more alive (0, 1) and bigger and less alive (1, −1)", () => {
    const shown = card(text(), "Add and stretch", "3 · ");
    expect(shown).toContain("bigger (1, 0)");
    expect(shown).toContain("more alive (0, 1)");
    expect(shown).toContain("bigger and less alive (1, −1)");
  });
});

describe("how alike are two arrows", () => {
  const horse = at("horse");
  const whale = at("whale");
  const bus = at("bus");
  const shownCard = (patch: Partial<ModuleState> = {}) => card(text(patch), "How alike are two arrows?", "4 · ");

  it("gives horse · whale = 1.80 + 1.08 = 2.88", () => {
    expect(whale).toEqual([2.0, 0.9]);
    expect(round(horse[0] * whale[0])).toBe(1.8);
    expect(round(horse[1] * whale[1])).toBe(1.08);
    expect(round(dot(horse, whale))).toBe(2.88);
    const shown = shownCard();
    expect(shown).toContain("+1.80");
    expect(shown).toContain("+1.08");
    expect(shown).toContain("+2.88");
  });

  it("cancels to exactly zero for horse and bus, at a right angle", () => {
    expect(bus).toEqual([1.6, -1.2]);
    expect(round(horse[0] * bus[0])).toBe(1.44);
    expect(round(horse[1] * bus[1])).toBe(-1.44);
    expect(dot(horse, bus)).toBeCloseTo(0, 12);
    expect(angleBetween(horse, bus)).toBeCloseTo(90, 9);
    const shown = shownCard({ other: "bus" });
    expect(shown).toContain("+1.44");
    expect(shown).toContain("−1.44");
    expect(shown).toContain("90°");
    expect(shown).toContain("At right angles");
  });

  it("flips to −2.88 when the second arrow turns half a turn", () => {
    expect(round(dot(horse, rotate(whale, 180)))).toBe(-2.88);
    const shown = shownCard({ turn: 180 });
    expect(shown).toContain("−2.88");
    expect(shown).toContain("−1.80");
    expect(shown).toContain("−1.08");
    expect(shown).toContain("−0.88");
  });

  it("casts a shadow of 1.31 on whale, whose length is 2.19", () => {
    expect(round(shadowLength(horse, whale))).toBe(1.31);
    expect(round(length(whale))).toBe(2.19);
    expect(shadowLength(horse, whale) * length(whale)).toBeCloseTo(dot(horse, whale), 12);
    expect(round(shadowLength(horse, rotate(whale, 180)))).toBe(-1.31);
    const shown = shownCard();
    expect(shown).toContain("shadow 1.31");
    expect(shown).toContain("1.31 × 2.19 = +2.88");
    expect(shown).toContain("29°");
  });

  it("raises the dot product to 4.32 at × 1.5 and leaves the cosine at 0.88", () => {
    expect(round(dot(horse, scale(whale, 1.5)))).toBe(4.32);
    expect(round(cosine(horse, whale))).toBe(0.88);
    expect(round(cosine(horse, scale(whale, 1.5)))).toBe(0.88);
    expect(round(2.88 / (1.5 * 2.19))).toBe(0.88);
    const plain = shownCard();
    expect(plain).toContain("+2.88 ÷ (1.50 × 2.19)");
    expect(plain).toContain("+0.88");
    const stretched = shownCard({ stretch: 1.5 });
    expect(stretched).toContain("+4.32");
    expect(stretched).toContain("+0.88");
  });
});

describe("a matrix moves every arrow at once", () => {
  const shownCard = (patch: Partial<ModuleState> = {}) => card(text(patch), "A matrix moves every arrow at once", "5 · ");

  it("opens on the shear with rows (1.0, 0.5) and (0.0, 1.0), which sends horse to (1.5, 1.2)", () => {
    expect(definition.initialState).toMatchObject({ ma: 1, mb: 0.5, mc: 0, md: 1 });
    const shear = presetById("shear").matrix;
    expect(shear).toEqual([1, 0.5, 0, 1]);
    const landed = apply(shear, at("horse"));
    expect(landed[0]).toBeCloseTo(1.5, 12);
    expect(landed[1]).toBeCloseTo(1.2, 12);
    expect(columns(shear)).toEqual([
      [1, 0],
      [0.5, 1],
    ]);
    const shown = shownCard();
    expect(shown).toContain("= 1.50");
    expect(shown).toContain("= 1.20");
    expect(shown).toContain("1st column (1.0, 0.0)");
    expect(shown).toContain("2nd column (0.5, 1.0)");
    expect(shown).toContain("its area is now 1.00");
  });

  it("moves horse to (2.1, 1.2) and bicycle to (−1.8, −1.0) when Top right is 1.0", () => {
    const m: Mat = [1, 1, 0, 1];
    expect(at("bicycle")).toEqual([-0.8, -1.0]);
    expect(apply(m, at("horse"))[0]).toBeCloseTo(2.1, 12);
    expect(apply(m, at("bicycle"))[0]).toBeCloseTo(-1.8, 12);
    expect(apply(m, at("bicycle"))[1]).toBeCloseTo(-1.0, 12);
    // Every animal moves right and every machine moves left.
    for (const id of ["mouse", "cat", "dog", "horse", "whale"]) expect(apply(m, at(id))[0]).toBeGreaterThan(at(id)[0]);
    for (const id of ["bicycle", "car", "bus"]) expect(apply(m, at(id))[0]).toBeLessThan(at(id)[0]);
    expect(shownCard({ mb: 1 })).toContain("= 2.10");
  });

  it("has the six ready-made matrices the lesson lists by their columns", () => {
    const cols = (id: string) => columns(presetById(id).matrix);
    expect(cols("same")).toEqual([[1, 0], [0, 1]]);
    expect(cols("stretch")).toEqual([[1.5, 0], [0, 1]]);
    expect(cols("rotate")).toEqual([[0, 1], [-1, 0]]);
    expect(cols("shear")).toEqual([[1, 0], [0.5, 1]]);
    expect(cols("flip")).toEqual([[-1, 0], [0, 1]]);
    expect(cols("squash")).toEqual([[1, 0], [0, 0.4]]);
    const turned = apply(presetById("rotate").matrix, at("horse"));
    expect(turned[0]).toBeCloseTo(-1.2, 12);
    expect(turned[1]).toBeCloseTo(0.9, 12);
    expect(determinant(presetById("stretch").matrix)).toBe(1.5);
    expect(shownCard({ ma: 1.5, mb: 0 })).toContain("its area is now 1.50");
  });

  it("merges cat and horse when leave alone has its Top left set to 0", () => {
    const m: Mat = [0, 0, 0, 1];
    expect(apply(m, at("cat"))).toEqual(apply(m, at("horse")));
    expect(at("cat")[1]).toBe(1.2);
    expect(at("horse")[1]).toBe(1.2);
  });
});

describe("one after another", () => {
  const rotateM = presetById("rotate").matrix;
  const stretch = presetById("stretch").matrix;
  const shownCard = (patch: Partial<ModuleState> = {}) => card(text(patch), "One after another", "6 · ");

  it("sends horse to (−1.8, 0.9) by rotate then stretch, and to (−1.2, 1.35) the other way", () => {
    const turned = apply(rotateM, at("horse"));
    expect(round(turned[0])).toBe(-1.2);
    expect(round(turned[1])).toBe(0.9);
    const both = apply(stretch, turned);
    expect(round(both[0])).toBe(-1.8);
    expect(round(both[1])).toBe(0.9);
    const wide = apply(stretch, at("horse"));
    expect(round(wide[0])).toBe(1.35);
    expect(round(wide[1])).toBe(1.2);
    const other = apply(rotateM, wide);
    expect(round(other[0])).toBe(-1.2);
    expect(round(other[1])).toBe(1.35);
  });

  it("writes the two single matrices the lessons give", () => {
    expect(compose(stretch, rotateM)).toEqual([0, -1.5, 1, 0]);
    expect(compose(rotateM, stretch)).toEqual([0, -1, 1.5, 0]);
    const shown = shownCard();
    expect(shown).toContain("one matrix that does both");
    for (const number of ["−1.5", "1.5", "−1.0"]) expect(shown).toContain(number);
    expect(shown).toContain("(1, 0) goes to (0.0, 1.0) in step 1, then to (0.0, 1.0) in step 2");
  });

  it("moves 8 of 8 words when rotate and stretch swap, and 0 of 8 for stretch and flip", () => {
    expect(orderGap(compose(stretch, rotateM), compose(rotateM, stretch)).differ).toBe(8);
    const flip = presetById("flip").matrix;
    expect(orderGap(compose(flip, stretch), compose(stretch, flip)).differ).toBe(0);
    expect(shownCard()).toContain("8 of 8");
    expect(shownCard({ first: "stretch", second: "flip" })).toContain("0 of 8");
  });
});

describe("when a matrix forgets", () => {
  const shownCard = (patch: Partial<ModuleState> = {}) => card(text(patch), "When a matrix forgets", "7 · ");

  it("halves the area at × 0.50 and keeps all eight words apart", () => {
    expect(determinant(squeezeMatrix("alive", 0.5))).toBe(0.5);
    expect(closestPair(squeezeMatrix("alive", 0.5)).distinct).toBe(8);
    const shown = shownCard({ keep: 0.5 });
    expect(shown).toContain("0.50");
    expect(shown).toContain("8 of 8");
  });

  it("puts cat and bicycle on (−0.8, 0.0) at × 0.00, leaving 7 of 8", () => {
    const flat = squeezeMatrix("alive", 0);
    expect(apply(flat, at("cat"))).toEqual([-0.8, 0]);
    expect(apply(flat, at("bicycle"))).toEqual([-0.8, -0]);
    expect(closestPair(flat)).toMatchObject({ distinct: 7, merged: true });
    const shown = shownCard({ keep: 0 });
    expect(shown).toContain("7 of 8");
    expect(shown).toContain("(−0.8, 0.0)");
    expect(shown).toContain("same spot");
  });

  it("can be stretched back by 5 at × 0.20 and by 20 at × 0.05, and by nothing at zero", () => {
    expect(undoSqueeze("alive", 0.2)?.[3]).toBeCloseTo(5, 12);
    expect(undoSqueeze("alive", 0.05)?.[3]).toBeCloseTo(20, 12);
    expect(undoSqueeze("alive", 0)).toBeNull();
    expect(closestPair(squeezeMatrix("alive", 0.05)).distinct).toBe(8);
    expect(shownCard({ keep: 0.2, undo: "stretch" })).toContain("1 ÷ 0.20 = 5.00");
    expect(shownCard({ keep: 0, undo: "stretch" })).toContain("1 ÷ 0");
  });
});

describe("a table from a few pieces", () => {
  /** Rank by elimination on exact small integers (as fractions kept in floats: the entries stay tiny). */
  const rankOf = (table: number[][]) => {
    const rows = table.map((row) => [...row]);
    let rank = 0;
    for (let col = 0; col < rows[0].length && rank < rows.length; col += 1) {
      const pivot = rows.findIndex((row, index) => index >= rank && Math.abs(row[col]) > 1e-9);
      if (pivot === -1) continue;
      [rows[rank], rows[pivot]] = [rows[pivot], rows[rank]];
      for (let r = 0; r < rows.length; r += 1) {
        if (r === rank) continue;
        const factor = rows[r][col] / rows[rank][col];
        for (let c = col; c < rows[0].length; c += 1) rows[r][c] -= factor * rows[rank][c];
      }
      rank += 1;
    }
    return rank;
  };

  it("holds 64 numbers and has rank 4, built from pieces of rank 1", () => {
    expect(FULL_TABLE.flat()).toHaveLength(64);
    expect(rankOf(FULL_TABLE)).toBe(4);
    for (const part of PIECES) {
      expect(rankOf(part.column.map((c) => part.row.map((r) => c * r)))).toBe(1);
    }
    expect(PIECES[0].column.every((value) => Math.abs(value) === 8)).toBe(true);
    expect(PIECES[0].row.every((value) => Math.abs(value) === 1)).toBe(true);
  });

  it("stores 16, 32, 48 and 64 numbers for an average miss of 4, 2, 1 and 0", () => {
    expect([1, 2, 3, 4].map((kept) => storedNumbers(kept))).toEqual([16, 32, 48, 64]);
    expect([1, 2, 3, 4].map((kept) => tableMiss(kept).average)).toEqual([4, 2, 1, 0]);
    const all = text();
    const shown = card(all, "A table from a few pieces", "8 · ");
    expect(shown).toContain("2 × (8 + 8) = 32");
    expect(shown).toContain("64 (50%)");
    expect(shown).toContain("2.00");
    expect(card(text({ pieces: 3 }), "A table from a few pieces", "8 · ")).toContain("3 × (8 + 8) = 48");
    expect(card(text({ pieces: 1 }), "A table from a few pieces", "8 · ")).toContain("4.00");
  });

  it("makes eight pieces of a 4,096 by 4,096 table cost 65,536 of its 16,777,216 numbers", () => {
    expect(4096 * 4096).toBe(16777216);
    expect(storedNumbers(8, 4096, 4096)).toBe(65536);
    expect(65536 / 16777216).toBeLessThan(0.005);
    expect(round((65536 / 16777216) * 100, 1)).toBe(0.4);
  });
});

describe("the challenge", () => {
  it("needs two numbers changed in round 1 and has the columns the lessons give for rounds 2 and 3", () => {
    const changed = ROUNDS[0].target.filter((value, index) => value !== IDENTITY[index]);
    expect(changed).toHaveLength(2);
    expect(ROUNDS[0].target).toEqual([1.5, 0, 0, 0.5]);
    expect(columns(ROUNDS[1].target)).toEqual([[0, -1], [1, 0]]);
    expect(columns(ROUNDS[2].target)).toEqual([[-1, 0], [0.5, 1]]);
    const shown = card(text({ round: 2, hint: 1 }), "Challenge: aim the map");
    expect(shown).toContain("(−1.0, 0.0) and (0.5, 1.0)");
  });

  it("keeps every round-1 target on its word's side of the centre, further out sideways and closer in vertically", () => {
    for (const id of ["mouse", "cat", "dog", "horse", "whale", "bicycle", "car", "bus"]) {
      const from = at(id);
      const to = apply(ROUNDS[0].target, from);
      expect(Math.sign(to[0])).toBe(Math.sign(from[0]));
      expect(Math.sign(to[1])).toBe(Math.sign(from[1]));
      expect(Math.abs(to[0])).toBeGreaterThan(Math.abs(from[0]));
      expect(Math.abs(to[1])).toBeLessThan(Math.abs(from[1]));
    }
  });

  it("can be won in every round by moving one slider at a time and keeping what lowers the meter", () => {
    const step = (value: number, by: number) => Math.min(2, Math.max(-2, Math.round((value + by) * 10) / 10));
    for (const { target } of ROUNDS) {
      let m: number[] = [...IDENTITY];
      let sweeps = 0;
      while (!aimSolved(m as unknown as Mat, target) && sweeps < 40) {
        for (let index = 0; index < 4; index += 1) {
          // Walk this slider downhill, one notch at a time, until neither direction helps.
          for (let moved = true; moved; ) {
            moved = false;
            const here = aimMiss(m as unknown as Mat, target).average;
            for (const by of [0.1, -0.1]) {
              const next: number[] = [...m];
              next[index] = step(m[index], by);
              if (aimMiss(next as unknown as Mat, target).average < here - 1e-12) {
                m = next;
                moved = true;
                break;
              }
            }
          }
        }
        sweeps += 1;
      }
      expect(aimSolved(m as unknown as Mat, target), `round with target ${target.join(", ")}`).toBe(true);
    }
  });
});

describe("what the three lessons state", () => {
  const figures = [
    "(0.9, 1.2)",
    "0.81",
    "1.44",
    "2.25",
    "(1.8, 2.4)",
    "3.00",
    "(−1.6, 1.0)",
    "1.89",
    "(1.9, 1.2)",
    "(−0.8, 1.2)",
    "−1.7",
    "1.80",
    "1.08",
    "2.88",
    "(1.6, −1.2)",
    "−2.88",
    "1.31",
    "2.19",
    "4.32",
    "0.88",
    "(1.5, 1.2)",
    "(2.1, 1.2)",
    "(−1.8, −1.0)",
    "(−1.2, 0.9)",
    "(−1.8, 0.9)",
    "(1.35, 1.2)",
    "(−1.2, 1.35)",
    "8 of 8",
    "7 of 8",
    "(−0.8, 0.0)",
    "16,777,216",
    "65,536",
  ];

  it.each(["plain", "simple", "standard"])("the %s lesson keeps every headline figure", (mode) => {
    const source = lesson(mode);
    for (const figure of figures) expect(source, `${mode} is missing ${figure}`).toContain(figure);
  });

  it.each(["plain", "simple", "standard"])("the %s lesson names every lab it points forward to", (mode) => {
    const source = lesson(mode);
    for (const title of [
      "*Data, features, and representation*",
      "*Tokens & embeddings*",
      "*A single neuron*",
      "*Attention*",
      "*LoRA & adapters*",
      "*Model merging*",
      "*Slopes & gradients*",
    ]) {
      expect(source, `${mode} does not name ${title}`).toContain(title);
    }
  });

  it("the checkpoint and the predict prompts quote figures the lab produces", () => {
    const checkpoint = JSON.stringify(definition.checkpoint);
    for (const figure of ["(−0.8, 1.2)", "2.88", "4.32", "0.88", "(−1.2, 0.9)", "(−0.8, 0.0)", "32 to 48"]) {
      expect(checkpoint).toContain(figure);
    }
    const all = text({ predictDot: 2, predictOrder: 1, predictUndo: 1, predictPieces: 1 });
    expect(all).toContain("+2.88 becomes −2.88");
    expect(all).toContain("both words become the list (−0.8, 0.0)");
    expect(all).toContain("stretching by 20 brings it back");
    expect(all).toContain("8 + 8 = 16");
  });
});
