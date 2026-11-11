import { describe, expect, it } from "vitest";
import {
  FULL_TABLE,
  IDENTITY,
  PATTERNS,
  PIECES,
  PRESETS,
  PUSHES,
  ROUNDS,
  TABLE_SIZE,
  WORDS,
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
  distance,
  dot,
  fmt,
  kindOf,
  length,
  matchingPreset,
  nearestWord,
  orderGap,
  pair,
  pieceTable,
  presetById,
  rebuildTable,
  rotate,
  scale,
  shadowLength,
  shadowPoint,
  signed,
  squeezeMatrix,
  storedNumbers,
  subtract,
  tableMiss,
  undoSqueeze,
  vecOf,
  wordById,
  type Mat,
  type Vec,
} from "./math";

const close = (a: Vec, b: Vec) => {
  expect(a[0]).toBeCloseTo(b[0], 10);
  expect(a[1]).toBeCloseTo(b[1], 10);
};

describe("the map of words", () => {
  it("has eight distinct words, five living and three machines, all inside the plane", () => {
    expect(WORDS).toHaveLength(8);
    expect(new Set(WORDS.map((word) => word.id)).size).toBe(8);
    expect(WORDS.filter((word) => word.y > 0)).toHaveLength(5);
    expect(WORDS.filter((word) => word.y < 0)).toHaveLength(3);
    for (const word of WORDS) {
      expect(Math.abs(word.x)).toBeLessThanOrEqual(2);
      expect(Math.abs(word.y)).toBeLessThanOrEqual(2);
    }
  });

  it("falls back to horse for an unknown word", () => {
    expect(wordById("unicorn").id).toBe("horse");
    expect(wordById("bus").id).toBe("bus");
  });

  it("finds the nearest word and its gap", () => {
    expect(nearestWord([0.9, 1.2])).toMatchObject({ word: { id: "horse" }, gap: 0 });
    const near = nearestWord([1.9, 1.2]);
    expect(near.word.id).toBe("whale");
    expect(near.gap).toBeCloseTo(Math.hypot(0.1, 0.3), 12);
  });

  it("names a direction from the signs of the two numbers", () => {
    expect(kindOf([0.9, 1.2])).toBe("big, living");
    expect(kindOf([-0.8, -1])).toBe("small, machine");
    expect(kindOf([0, 1])).toBe("middling size, living");
  });
});

describe("vector arithmetic", () => {
  it("adds, subtracts and scales number by number", () => {
    close(add([1, 2], [3, -5]), [4, -3]);
    close(subtract([1, 2], [3, -5]), [-2, 7]);
    close(scale([1, -2], -1.5), [-1.5, 3]);
  });

  it("measures length with the right-triangle rule, and never below zero", () => {
    expect(length([3, 4])).toBe(5);
    expect(length([-3, -4])).toBe(5);
    expect(length([0, 0])).toBe(0);
    expect(distance([1, 1], [4, 5])).toBe(5);
  });

  it("computes the dot product two ways that agree: pairs, and lengths times cosine", () => {
    const a: Vec = [0.9, 1.2];
    const b: Vec = [2, 0.9];
    expect(dot(a, b)).toBeCloseTo(0.9 * 2 + 1.2 * 0.9, 12);
    expect(length(a) * length(b) * cosine(a, b)).toBeCloseTo(dot(a, b), 12);
    expect(shadowLength(a, b) * length(b)).toBeCloseTo(dot(a, b), 12);
  });

  it("keeps cosine between −1 and 1, unchanged by stretching, and 0 for a zero arrow", () => {
    const a: Vec = [0.9, 1.2];
    const b: Vec = [2, 0.9];
    expect(cosine(a, scale(b, 7))).toBeCloseTo(cosine(a, b), 12);
    expect(cosine(a, scale(a, 3))).toBeCloseTo(1, 12);
    expect(cosine(a, scale(a, -0.2))).toBeCloseTo(-1, 12);
    expect(cosine(a, [0, 0])).toBe(0);
    expect(angleBetween([1, 0], [0, 2])).toBeCloseTo(90, 10);
    expect(angleBetween([1, 0], [-3, 0])).toBeCloseTo(180, 10);
  });

  it("drops the shadow at a right angle to the second arrow", () => {
    const a: Vec = [0.9, 1.2];
    const b: Vec = [2, 0.9];
    const foot = shadowPoint(a, b);
    expect(dot(subtract(a, foot), b)).toBeCloseTo(0, 12);
    expect(length(foot)).toBeCloseTo(Math.abs(shadowLength(a, b)), 12);
    // Turning b half a turn keeps the line, so the foot stays put while the signed length flips.
    close(shadowPoint(a, rotate(b, 180)), foot);
    expect(shadowLength(a, rotate(b, 180))).toBeCloseTo(-shadowLength(a, b), 12);
    close(shadowPoint(a, [0, 0]), [0, 0]);
    expect(shadowLength(a, [0, 0])).toBe(0);
  });

  it("rotates exactly at quarter turns and keeps length at any angle", () => {
    expect(rotate([2, 0.9], 90)).toEqual([-0.9, 2]);
    expect(rotate([2, 0.9], 180)).toEqual([-2, -0.9]);
    expect(rotate([2, 0.9], 270)).toEqual([0.9, -2]);
    expect(rotate([2, 0.9], 360)).toEqual([2, 0.9]);
    expect(length(rotate([2, 0.9], 45))).toBeCloseTo(length([2, 0.9]), 12);
    expect(angleBetween([2, 0.9], rotate([2, 0.9], 45))).toBeCloseTo(45, 9);
  });
});

describe("matrices", () => {
  it("applies a matrix as one dot product per row, and as a mix of its columns", () => {
    const m: Mat = [1, 0.5, 0, 1];
    const v: Vec = [0.9, 1.2];
    close(apply(m, v), [dot([1, 0.5], v), dot([0, 1], v)]);
    const [first, second] = columns(m);
    close(apply(m, v), add(scale(first, v[0]), scale(second, v[1])));
    close(apply(m, [1, 0]), first);
    close(apply(m, [0, 1]), second);
  });

  it("composes so that the product does the first matrix and then the second", () => {
    const rotateM = presetById("rotate").matrix;
    const stretch = presetById("stretch").matrix;
    for (const word of WORDS) {
      close(apply(compose(stretch, rotateM), vecOf(word)), apply(stretch, apply(rotateM, vecOf(word))));
    }
    expect(compose(stretch, rotateM)).not.toEqual(compose(rotateM, stretch));
    expect(compose(IDENTITY, stretch)).toEqual(stretch);
  });

  it("reads the determinant as the area factor of each preset", () => {
    expect(determinant(presetById("same").matrix)).toBe(1);
    expect(determinant(presetById("stretch").matrix)).toBe(1.5);
    expect(determinant(presetById("rotate").matrix)).toBe(1);
    expect(determinant(presetById("shear").matrix)).toBe(1);
    expect(determinant(presetById("flip").matrix)).toBe(-1);
    expect(determinant(presetById("squash").matrix)).toBe(0.4);
    expect(determinant(compose(presetById("stretch").matrix, presetById("flip").matrix))).toBeCloseTo(-1.5, 12);
  });

  it("recognises a preset from its four numbers and nothing else", () => {
    for (const preset of PRESETS) expect(matchingPreset(preset.matrix)?.id).toBe(preset.id);
    expect(matchingPreset([1, 0.3, 0, 1])).toBeUndefined();
    expect(presetById("nope").id).toBe("same");
  });

  it("keeps every preset, and every pair of sequence moves, inside the drawn plane", () => {
    for (const first of PRESETS) {
      for (const second of PRESETS) {
        for (const word of WORDS) {
          const at = apply(compose(second.matrix, first.matrix), vecOf(word));
          expect(Math.max(Math.abs(at[0]), Math.abs(at[1]))).toBeLessThanOrEqual(4.6);
        }
      }
    }
    for (const preset of PRESETS) {
      for (const word of WORDS) {
        const at = apply(preset.matrix, vecOf(word));
        expect(Math.max(Math.abs(at[0]), Math.abs(at[1]))).toBeLessThanOrEqual(3.2);
      }
    }
  });

  it("counts how many words two orders disagree on", () => {
    const rotateM = presetById("rotate").matrix;
    const stretch = presetById("stretch").matrix;
    const squash = presetById("squash").matrix;
    expect(orderGap(compose(stretch, rotateM), compose(rotateM, stretch)).differ).toBe(8);
    // Two moves along the axes commute.
    expect(orderGap(compose(stretch, squash), compose(squash, stretch))).toMatchObject({ differ: 0, widest: 0 });
  });
});

describe("squeezing the map", () => {
  it("multiplies one number and leaves the other", () => {
    expect(squeezeMatrix("alive", 0.3)).toEqual([1, 0, 0, 0.3]);
    expect(squeezeMatrix("size", 0.3)).toEqual([0.3, 0, 0, 1]);
    expect(determinant(squeezeMatrix("alive", 0.3))).toBeCloseTo(0.3, 12);
  });

  it("can be undone at any factor above zero and not at zero", () => {
    for (const keep of [1, 0.5, 0.2, 0.05]) {
      const undo = undoSqueeze("alive", keep);
      expect(undo).not.toBeNull();
      const back = compose(undo as Mat, squeezeMatrix("alive", keep));
      back.forEach((value, index) => expect(value).toBeCloseTo(IDENTITY[index], 10));
    }
    expect(undoSqueeze("alive", 0)).toBeNull();
    expect(undoSqueeze("size", 0)).toBeNull();
  });

  it("merges cat and bicycle only at zero", () => {
    expect(closestPair(IDENTITY)).toMatchObject({ distinct: 8, merged: false });
    expect(closestPair(squeezeMatrix("alive", 0.05))).toMatchObject({ distinct: 8, merged: false });
    const flat = closestPair(squeezeMatrix("alive", 0));
    expect(flat.pair.map((word) => word.id).sort()).toEqual(["bicycle", "cat"]);
    expect(flat).toMatchObject({ distinct: 7, merged: true, gap: 0 });
  });

  it("merges cat and horse instead when size is the number forgotten", () => {
    const flat = closestPair(squeezeMatrix("size", 0));
    expect(flat.pair.map((word) => word.id).sort()).toEqual(["cat", "horse"]);
    expect(flat.merged).toBe(true);
  });
});

describe("the table built from pieces", () => {
  it("uses eight patterns of plus and minus one that are orthogonal to one another", () => {
    expect(PATTERNS).toHaveLength(TABLE_SIZE);
    for (let i = 0; i < PATTERNS.length; i += 1) {
      for (const sign of PATTERNS[i]) expect(Math.abs(sign)).toBe(1);
      for (let j = 0; j < PATTERNS.length; j += 1) {
        const product = PATTERNS[i].reduce((sum, value, k) => sum + value * PATTERNS[j][k], 0);
        expect(product).toBe(i === j ? TABLE_SIZE : 0);
      }
    }
  });

  it("has four pieces, each half as strong as the last, with orthogonal columns and rows", () => {
    expect(PIECES.map((part) => part.strength)).toEqual([8, 4, 2, 1]);
    for (let i = 0; i < PIECES.length; i += 1) {
      for (let j = i + 1; j < PIECES.length; j += 1) {
        expect(PIECES[i].column.reduce((sum, value, k) => sum + value * PIECES[j].column[k], 0)).toBe(0);
        expect(PIECES[i].row.reduce((sum, value, k) => sum + value * PIECES[j].row[k], 0)).toBe(0);
      }
    }
  });

  it("makes every cell of a piece its column entry times its row entry", () => {
    const cells = pieceTable(PIECES[1]);
    for (let i = 0; i < TABLE_SIZE; i += 1) {
      for (let j = 0; j < TABLE_SIZE; j += 1) {
        expect(cells[i][j]).toBe(PIECES[1].column[i] * PIECES[1].row[j]);
        expect(Math.abs(cells[i][j])).toBe(4);
      }
    }
  });

  it("rebuilds the full table from all four pieces, in odd whole numbers up to 15", () => {
    expect(rebuildTable(4)).toEqual(FULL_TABLE);
    const values = FULL_TABLE.flat();
    expect(values).toHaveLength(64);
    for (const value of values) expect(Math.abs(value) % 2).toBe(1);
    expect(Math.max(...values)).toBe(15);
    expect(Math.min(...values)).toBe(-15);
    expect(rebuildTable(0).flat().every((value) => value === 0)).toBe(true);
  });

  it("halves the average miss with each piece and stores r × (8 + 8) numbers", () => {
    expect([1, 2, 3, 4].map((kept) => tableMiss(kept).average)).toEqual([4, 2, 1, 0]);
    expect([1, 2, 3, 4].map((kept) => tableMiss(kept).largest)).toEqual([7, 3, 1, 0]);
    expect([1, 2, 3, 4].map((kept) => storedNumbers(kept))).toEqual([16, 32, 48, 64]);
    expect(storedNumbers(8, 4096, 4096)).toBe(65536);
  });

  it("keeps the strongest pieces first, so no other order of r pieces misses by less", () => {
    // With orthogonal pieces the squared miss is the sum of the dropped strengths squared (times 64).
    const squaredMiss = (kept: number[]) => {
      let total = 0;
      for (let i = 0; i < TABLE_SIZE; i += 1) {
        for (let j = 0; j < TABLE_SIZE; j += 1) {
          const rebuilt = kept.reduce((sum, index) => sum + pieceTable(PIECES[index])[i][j], 0);
          total += (FULL_TABLE[i][j] - rebuilt) ** 2;
        }
      }
      return total;
    };
    expect(squaredMiss([0, 1])).toBe(64 * (2 * 2 + 1 * 1));
    for (const other of [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) {
      expect(squaredMiss(other)).toBeGreaterThan(squaredMiss([0, 1]));
    }
  });
});

describe("the challenge", () => {
  it("has three rounds whose targets the sliders can reach exactly", () => {
    expect(ROUNDS).toHaveLength(3);
    for (const round of ROUNDS) {
      for (const value of round.target) {
        expect(Math.abs(value)).toBeLessThanOrEqual(2);
        expect(Math.round(value * 10)).toBeCloseTo(value * 10, 9);
      }
      expect(aimSolved(round.target, round.target)).toBe(true);
      expect(aimSolved(IDENTITY, round.target)).toBe(false);
      expect(aimMiss(round.target, round.target)).toEqual({ average: 0, largest: 0 });
      for (const word of WORDS) {
        const at = apply(round.target, vecOf(word));
        expect(Math.max(Math.abs(at[0]), Math.abs(at[1]))).toBeLessThanOrEqual(3.1);
      }
    }
  });

  it("is not solved by a matrix one slider step away from the target", () => {
    for (const round of ROUNDS) {
      for (let index = 0; index < 4; index += 1) {
        for (const nudge of [-0.1, 0.1]) {
          const near = round.target.map((value, i) => (i === index ? value + nudge : value)) as unknown as Mat;
          expect(aimSolved(near, round.target)).toBe(false);
        }
      }
    }
  });

  it("falls steadily as one slider moves toward its target, so trial and error works", () => {
    const target = ROUNDS[0].target;
    let last = Infinity;
    for (let step = 0; step <= 5; step += 1) {
      const miss = aimMiss([1 + step * 0.1, 0, 0, 1], target).average;
      expect(miss).toBeLessThan(last);
      last = miss;
    }
  });
});

describe("the steps a word can take", () => {
  it("offers three steps, the first along size and the second along alive", () => {
    expect(PUSHES.map((push) => push.vec)).toEqual([
      [1, 0],
      [0, 1],
      [1, -1],
    ]);
  });
});

describe("formatting", () => {
  it("uses a real minus sign and never prints minus zero", () => {
    expect(fmt(-1.25, 1)).toBe("−1.3");
    expect(fmt(-0.04, 1)).toBe("0.0");
    expect(fmt(0.9 - 1.7, 1)).toBe("−0.8");
    expect(fmt(Number.NaN)).toBe("0.0");
    expect(fmt(Infinity, 2)).toBe("0.00");
  });

  it("adds a plus sign to positive values only", () => {
    expect(signed(2.88)).toBe("+2.88");
    expect(signed(-2.88)).toBe("−2.88");
    expect(signed(0)).toBe("0.00");
    expect(signed(0.001)).toBe("0.00");
  });

  it("writes a vector as a bracketed pair", () => {
    expect(pair([0.9, 1.2])).toBe("(0.9, 1.2)");
    expect(pair([-0.8, 0], 1)).toBe("(−0.8, 0.0)");
  });
});
