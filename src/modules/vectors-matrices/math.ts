/**
 * The arithmetic behind the Vectors & matrices lab. Every number the lab shows comes from here:
 * a hand-written map of eight words, and small closed forms for arrows, 2×2 matrices and an 8×8
 * table built from four authored pieces. Nothing is measured from a model.
 */

export type Vec = readonly [number, number];
/** A 2×2 matrix in reading order: top-left, top-right, bottom-left, bottom-right. */
export type Mat = readonly [number, number, number, number];

export interface Word {
  id: string;
  /** First number, "size": below zero is smaller than a person, above zero is bigger. */
  x: number;
  /** Second number, "alive": above zero is a living thing, below zero is a machine. */
  y: number;
  /** Where the label sits beside the dot, in pixels, so neighbours do not collide at rest. */
  dx: number;
  dy: number;
}

/**
 * The tiny map of meaning. The two numbers per word are invented for this lab and tuned so the
 * lesson's arithmetic is clean: horse has length 1.5, horse and bus are at right angles, cat plus
 * 1.7 "bigger" is horse, and cat and bicycle share a size.
 */
export const WORDS: readonly Word[] = [
  { id: "mouse", x: -1.6, y: 1.0, dx: 0, dy: -9 },
  { id: "cat", x: -0.8, y: 1.2, dx: -8, dy: -9 },
  { id: "dog", x: -0.3, y: 1.3, dx: 6, dy: -9 },
  { id: "horse", x: 0.9, y: 1.2, dx: 0, dy: -9 },
  { id: "whale", x: 2.0, y: 0.9, dx: 0, dy: -9 },
  { id: "bicycle", x: -0.8, y: -1.0, dx: 0, dy: 17 },
  { id: "car", x: 1.1, y: -1.3, dx: -4, dy: 17 },
  { id: "bus", x: 1.6, y: -1.2, dx: 8, dy: 17 },
];

export const WORD_IDS = WORDS.map((word) => word.id);
export const wordById = (id: string): Word => WORDS.find((word) => word.id === id) ?? WORDS[3];
export const vecOf = (word: Word): Vec => [word.x, word.y];

export const add = (a: Vec, b: Vec): Vec => [a[0] + b[0], a[1] + b[1]];
export const subtract = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1]];
export const scale = (a: Vec, k: number): Vec => [a[0] * k, a[1] * k];
export const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
export const length = (a: Vec) => Math.hypot(a[0], a[1]);
export const distance = (a: Vec, b: Vec) => length(subtract(a, b));

/** Dot product divided by both lengths: 1 same direction, 0 at right angles, −1 opposite. Zero for a zero arrow. */
export const cosine = (a: Vec, b: Vec) => {
  const lengths = length(a) * length(b);
  return lengths < 1e-9 ? 0 : Math.max(-1, Math.min(1, dot(a, b) / lengths));
};

/** The angle between two arrows, in degrees from 0 to 180. */
export const angleBetween = (a: Vec, b: Vec) => (Math.acos(cosine(a, b)) * 180) / Math.PI;

/** Signed length of the shadow a casts on the line through b: negative when a leans away from b. */
export const shadowLength = (a: Vec, b: Vec) => (length(b) < 1e-9 ? 0 : dot(a, b) / length(b));

/** The point on b's line under the tip of a: the foot of the shadow. */
export const shadowPoint = (a: Vec, b: Vec): Vec => {
  const squared = dot(b, b);
  return squared < 1e-9 ? [0, 0] : scale(b, dot(a, b) / squared);
};

/** Turn an arrow anticlockwise by a number of degrees, keeping its length. Quarter turns are exact. */
export const rotate = (a: Vec, degrees: number): Vec => {
  const quarter = ((Math.round(degrees) % 360) + 360) % 360;
  if (quarter === 0) return a;
  if (quarter === 90) return [-a[1], a[0]];
  if (quarter === 180) return [-a[0], -a[1]];
  if (quarter === 270) return [a[1], -a[0]];
  const radians = (degrees * Math.PI) / 180;
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return [a[0] * c - a[1] * s, a[0] * s + a[1] * c];
};

/** The word whose dot is closest to a point, and how far away it is. */
export const nearestWord = (point: Vec) => {
  let best = WORDS[0];
  let gap = Infinity;
  for (const word of WORDS) {
    const d = distance(point, vecOf(word));
    if (d < gap - 1e-12) {
      best = word;
      gap = d;
    }
  }
  return { word: best, gap };
};

/** Words for the direction an arrow points on this map, read from the signs of its two numbers. */
export const kindOf = (a: Vec) => {
  const size = Math.abs(a[0]) < 0.05 ? "middling size" : a[0] > 0 ? "big" : "small";
  const alive = Math.abs(a[1]) < 0.05 ? "neither living nor machine" : a[1] > 0 ? "living" : "machine";
  return `${size}, ${alive}`;
};

/* ── steps you can add to a word (card 2) ── */

export interface Push {
  id: string;
  label: string;
  vec: Vec;
}

export const PUSHES: readonly Push[] = [
  { id: "bigger", label: "bigger", vec: [1, 0] },
  { id: "alive", label: "more alive", vec: [0, 1] },
  { id: "both", label: "bigger and less alive", vec: [1, -1] },
];
export const PUSH_IDS = PUSHES.map((push) => push.id);
export const pushById = (id: string): Push => PUSHES.find((push) => push.id === id) ?? PUSHES[0];

/* ── matrices ── */

export const IDENTITY: Mat = [1, 0, 0, 1];

/** Matrix times arrow. Each output number is one row of the matrix dotted with the arrow. */
export const apply = (m: Mat, v: Vec): Vec => [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]];

/** The two columns: where the arrows (1, 0) and (0, 1) land. */
export const columns = (m: Mat): readonly [Vec, Vec] => [
  [m[0], m[2]],
  [m[1], m[3]],
];

/** The one matrix that does `first` and then `second`: second × first. */
export const compose = (second: Mat, first: Mat): Mat => [
  second[0] * first[0] + second[1] * first[2],
  second[0] * first[1] + second[1] * first[3],
  second[2] * first[0] + second[3] * first[2],
  second[2] * first[1] + second[3] * first[3],
];

/** Signed area of the unit square after the move. Zero means the map is flattened onto a line (or a point). */
export const determinant = (m: Mat) => m[0] * m[3] - m[1] * m[2];

export interface Preset {
  id: string;
  label: string;
  matrix: Mat;
  /** What it does, in words. */
  says: string;
}

export const PRESETS: readonly Preset[] = [
  { id: "same", label: "leave alone", matrix: [1, 0, 0, 1], says: "every word stays where it is" },
  { id: "stretch", label: "stretch", matrix: [1.5, 0, 0, 1], says: "every size is multiplied by 1.5" },
  { id: "rotate", label: "rotate", matrix: [0, -1, 1, 0], says: "the whole map makes a quarter turn to the left" },
  { id: "shear", label: "shear", matrix: [1, 0.5, 0, 1], says: "each word slides sideways by half its alive number" },
  { id: "flip", label: "flip", matrix: [-1, 0, 0, 1], says: "every size changes sign, like a mirror" },
  { id: "squash", label: "squash", matrix: [1, 0, 0, 0.4], says: "every alive number shrinks to 0.4 of itself" },
];
export const presetById = (id: string): Preset => PRESETS.find((preset) => preset.id === id) ?? PRESETS[0];
export const matchingPreset = (m: Mat) =>
  PRESETS.find((preset) => preset.matrix.every((value, index) => Math.abs(value - m[index]) < 1e-9));

/** The moves offered in "One after another". */
export const SEQUENCE_IDS = ["rotate", "stretch", "shear", "flip"] as const;

/** Where every word lands under a matrix. */
export const moveWords = (m: Mat) => WORDS.map((word) => ({ word, at: apply(m, vecOf(word)) }));

/** How many words land in a different place under two matrices, and the widest gap between the two landings. */
export const orderGap = (one: Mat, other: Mat) => {
  let differ = 0;
  let widest = 0;
  let widestWord = WORDS[0];
  for (const word of WORDS) {
    const gap = distance(apply(one, vecOf(word)), apply(other, vecOf(word)));
    if (gap > 1e-9) differ += 1;
    if (gap > widest) {
      widest = gap;
      widestWord = word;
    }
  }
  return { differ, widest, widestWord };
};

/* ── squeezing the map onto a line (card 6) ── */

export type Forget = "alive" | "size";

/** Multiply one of the two numbers by `keep` and leave the other alone. */
export const squeezeMatrix = (forget: Forget, keep: number): Mat =>
  forget === "alive" ? [1, 0, 0, keep] : [keep, 0, 0, 1];

/** The matrix that would undo a squeeze, or null when the squeeze reached zero and nothing can. */
export const undoSqueeze = (forget: Forget, keep: number): Mat | null =>
  keep <= 1e-9 ? null : forget === "alive" ? [1, 0, 0, 1 / keep] : [1 / keep, 0, 0, 1];

/** The two words that end up closest together under a matrix, and the count of distinct landing spots. */
export const closestPair = (m: Mat) => {
  const moved = moveWords(m);
  let pair: [Word, Word] = [WORDS[0], WORDS[1]];
  let gap = Infinity;
  for (let i = 0; i < moved.length; i += 1) {
    for (let j = i + 1; j < moved.length; j += 1) {
      const d = distance(moved[i].at, moved[j].at);
      if (d < gap - 1e-12) {
        gap = d;
        pair = [moved[i].word, moved[j].word];
      }
    }
  }
  const spots: Vec[] = [];
  for (const { at } of moved) {
    if (!spots.some((spot) => distance(spot, at) < 1e-9)) spots.push(at);
  }
  return { pair, gap, distinct: spots.length, merged: gap < 1e-9 };
};

/* ── an 8×8 table built from "one column times one row" pieces (card 7) ── */

export const TABLE_SIZE = 8;

/** Eight patterns of +1 and −1. Any two different ones multiply and add to zero (they are orthogonal). */
export const PATTERNS: readonly (readonly number[])[] = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, -1, -1, -1, -1],
  [1, 1, -1, -1, -1, -1, 1, 1],
  [1, 1, -1, -1, 1, 1, -1, -1],
  [1, -1, -1, 1, 1, -1, -1, 1],
  [1, -1, -1, 1, -1, 1, 1, -1],
  [1, -1, 1, -1, -1, 1, -1, 1],
  [1, -1, 1, -1, 1, -1, 1, -1],
];

export interface Piece {
  /** How loud this piece is in the table: every cell of the piece is plus or minus this number. */
  strength: number;
  /** The column of eight numbers (one per table row), strength included. */
  column: readonly number[];
  /** The row of eight numbers (one per table column). */
  row: readonly number[];
}

const piece = (strength: number, columnPattern: number, rowPattern: number): Piece => ({
  strength,
  column: PATTERNS[columnPattern].map((sign) => sign * strength),
  row: PATTERNS[rowPattern],
});

/** Four authored pieces, each half as strong as the one before. Their columns are orthogonal, and so are their rows. */
export const PIECES: readonly Piece[] = [piece(8, 1, 1), piece(4, 3, 2), piece(2, 2, 5), piece(1, 7, 7)];

/** One piece written out as a full table: cell = column entry × row entry. */
export const pieceTable = (part: Piece) => part.column.map((c) => part.row.map((r) => c * r));

/** The sum of the first `kept` pieces. With all four it is the full table. */
export const rebuildTable = (kept: number) => {
  const table = Array.from({ length: TABLE_SIZE }, () => Array.from({ length: TABLE_SIZE }, () => 0));
  PIECES.slice(0, Math.max(0, kept)).forEach((part) => {
    const cells = pieceTable(part);
    for (let i = 0; i < TABLE_SIZE; i += 1) for (let j = 0; j < TABLE_SIZE; j += 1) table[i][j] += cells[i][j];
  });
  return table;
};

export const FULL_TABLE = rebuildTable(PIECES.length);

/** How far the rebuilt table is from the full one: the average and the largest miss over the 64 cells. */
export const tableMiss = (kept: number) => {
  const rebuilt = rebuildTable(kept);
  let total = 0;
  let largest = 0;
  for (let i = 0; i < TABLE_SIZE; i += 1) {
    for (let j = 0; j < TABLE_SIZE; j += 1) {
      const miss = Math.abs(FULL_TABLE[i][j] - rebuilt[i][j]);
      total += miss;
      largest = Math.max(largest, miss);
    }
  }
  return { average: total / (TABLE_SIZE * TABLE_SIZE), largest };
};

/** Numbers you must store for `kept` pieces of an n-by-m table: each piece is one column plus one row. */
export const storedNumbers = (kept: number, rows = TABLE_SIZE, cols = TABLE_SIZE) => kept * (rows + cols);

/* ── the challenge (card 8) ── */

export interface Round {
  id: string;
  title: string;
  target: Mat;
  /** One sentence on what kind of move this is, shown once solved. */
  solvedText: string;
}

export const ROUNDS: readonly Round[] = [
  {
    id: "stretch",
    title: "Round 1: a stretch",
    target: [1.5, 0, 0, 0.5],
    solvedText: "Size is multiplied by 1.5 and alive by 0.5: two numbers on the diagonal, zeros elsewhere.",
  },
  {
    id: "rotate",
    title: "Round 2: a rotation",
    target: [0, 1, -1, 0],
    solvedText: "A quarter turn to the right: the size arrow lands pointing down, the alive arrow pointing right.",
  },
  {
    id: "shear-flip",
    title: "Round 3: a shear with a flip",
    target: [-1, 0.5, 0, 1],
    solvedText: "Sizes are mirrored, then every word slides sideways by half its alive number.",
  },
];

/** Average and largest distance between where a matrix puts the words and where the target puts them. */
export const aimMiss = (m: Mat, target: Mat) => {
  let total = 0;
  let largest = 0;
  for (const word of WORDS) {
    const gap = distance(apply(m, vecOf(word)), apply(target, vecOf(word)));
    total += gap;
    largest = Math.max(largest, gap);
  }
  return { average: total / WORDS.length, largest };
};

/** A round is won when every word sits on its target (to within a hair, for float noise). */
export const aimSolved = (m: Mat, target: Mat) => aimMiss(m, target).largest < 0.02;

/* ── formatting ── */

/** A number with a real minus sign and no "−0.0". */
export const fmt = (value: number, digits = 1) => {
  const safe = Number.isFinite(value) ? value : 0;
  const text = safe.toFixed(digits);
  return Number(text) === 0 ? (0).toFixed(digits) : text.replace("-", "−");
};

/** The same, with an explicit plus sign on positive values. */
export const signed = (value: number, digits = 2) => {
  const text = fmt(value, digits);
  return Number(text.replace("−", "-")) > 0 ? `+${text}` : text;
};

export const pair = (v: Vec, digits = 1) => `(${fmt(v[0], digits)}, ${fmt(v[1], digits)})`;
