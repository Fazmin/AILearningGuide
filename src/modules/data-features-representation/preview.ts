import { ROWS, euclid } from "./encode";

/* ---------------------------------------------------------------- pixels */

/** A 3×3 picture, row by row, written as nine 0/1 characters. */
export const PIXEL_COUNT = 9;
export const DEFAULT_PIXELS = "010111010";

/** Three fixed reference pictures. The drawing is compared with each as a vector of nine numbers. */
export const TEMPLATES = [
  { id: "plus", label: "plus", pixels: "010111010" },
  { id: "ring", label: "ring", pixels: "111101111" },
  { id: "bar", label: "bar", pixels: "010010010" },
] as const;

/** Anything that is not exactly nine 0/1 characters falls back to the default drawing. */
export const parsePixels = (value: unknown): string =>
  typeof value === "string" && /^[01]{9}$/.test(value) ? value : DEFAULT_PIXELS;

export const pixelVector = (pixels: string): number[] => Array.from(parsePixels(pixels), (char) => Number(char));

export const togglePixel = (pixels: string, index: number): string => {
  const safe = parsePixels(pixels);
  if (!Number.isInteger(index) || index < 0 || index >= PIXEL_COUNT) return safe;
  return `${safe.slice(0, index)}${safe[index] === "1" ? "0" : "1"}${safe.slice(index + 1)}`;
};

/** Straight-line distance between the drawing and each template, nearest first (ties keep template order). */
export const templateDistances = (pixels: string) => {
  const drawing = pixelVector(pixels);
  return TEMPLATES.map((template, order) => ({
    id: template.id,
    label: template.label,
    pixels: template.pixels,
    distance: euclid(drawing, pixelVector(template.pixels)),
    order,
  })).sort((left, right) => left.distance - right.distance || left.order - right.order);
};

/* ----------------------------------------------------------------- words */

/** A tiny fixed vocabulary, defined here for the lab. It is not a real tokenizer. */
export const VOCABULARY = ["<unk>", "the", "cat", "sat", "on", "mat", "dog", "ran"] as const;

export const SENTENCES = [
  { id: "cat-mat", text: "the cat sat on the mat" },
  { id: "dog-ran", text: "the dog ran" },
  { id: "cat-ran", text: "the cat ran on the mat" },
  { id: "bird-mat", text: "the bird sat on the mat" },
] as const;

export const SENTENCE_IDS: readonly string[] = SENTENCES.map((sentence) => sentence.id);

/** Each word becomes its position in the vocabulary; a word that is not in it becomes ID 0, <unk>. */
export const encodeSentence = (text: string) =>
  text
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .map((word) => {
      const found = (VOCABULARY as readonly string[]).indexOf(word);
      return { word, id: found < 0 ? 0 : found, known: found > 0 };
    });

export const idOf = (word: string) => Math.max(0, (VOCABULARY as readonly string[]).indexOf(word));

/** The subtraction a model would do if it read IDs as amounts. */
export const idGap = (left: string, right: string) => Math.abs(idOf(left) - idOf(right));

/** One-hot vectors for two different words are always this far apart. */
export const ONE_HOT_GAP = Math.SQRT2;

/* ------------------------------------------------------------ messy rows */

export const HANDLINGS = ["zero", "drop", "mean"] as const;
export type Handling = (typeof HANDLINGS)[number];

export type Mess = { missing: boolean; duplicate: boolean; mislabel: boolean; handling: Handling };
export const CLEAN: Mess = { missing: false, duplicate: false, mislabel: false, handling: "zero" };

/** Which listing the messy options act on. They are authored so each failure can be seen. */
export const MISSING_ROW = 3; // Dock 2: its rooms are blank
export const DUPLICATE_ROW = 0; // Alder 12: listed twice
export const MISLABEL_ROW = 3; // Dock 2: its price tag is flipped

/** A new listing the nearest-neighbour answer is asked about: 3 rooms, red door, no park. */
export const QUERY = { rooms: 3, color: "red", park: 0, code: 1 } as const;
export const K = 3;

const COLOR_STAMP: Record<string, number> = { red: 1, blue: 2, green: 3 };

export type MessyRow = {
  key: string;
  name: string;
  rooms: number;
  color: string;
  park: number;
  price: number;
  /** Plain-text note on what is wrong with this row, or an empty string. */
  flag: string;
  /** A dropped row stays in the table so you can see it go, but nothing reads it. */
  dropped: boolean;
};

const mean = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

/** The six listings with the chosen mess applied. Features use the default integer stamps. */
export const messyRows = (mess: Mess): MessyRow[] => {
  const base: MessyRow[] = ROWS.map((row) => ({
    key: `row-${row.id}`,
    name: row.name,
    rooms: row.rooms,
    color: row.color,
    park: row.park === "yes" ? 1 : 0,
    price: row.price,
    flag: "",
    dropped: false,
  }));

  if (mess.missing) {
    const others = base.filter((_, index) => index !== MISSING_ROW).map((row) => row.rooms);
    const filled = mean(others);
    const target = base[MISSING_ROW];
    if (mess.handling === "zero") {
      base[MISSING_ROW] = { ...target, rooms: 0, flag: "rooms missing, read as 0" };
    } else if (mess.handling === "mean") {
      base[MISSING_ROW] = { ...target, rooms: filled, flag: `rooms missing, filled with ${filled.toFixed(1)}` };
    } else {
      base[MISSING_ROW] = { ...target, flag: "rooms missing, row dropped", dropped: true };
    }
  }

  if (mess.mislabel) {
    const target = base[MISLABEL_ROW];
    const flipped = 1 - target.price;
    const note = `tag flipped from ${target.price ? "high" : "low"} to ${flipped ? "high" : "low"}`;
    base[MISLABEL_ROW] = { ...target, price: flipped, flag: target.flag ? `${target.flag}; ${note}` : note };
  }

  if (mess.duplicate) {
    const source = base[DUPLICATE_ROW];
    base.push({ ...source, key: "row-duplicate", flag: `duplicate of ${source.name}` });
  }

  return base;
};

const featuresOf = (row: { rooms: number; color: string; park: number }) => [row.rooms, COLOR_STAMP[row.color] ?? 0, row.park];

/** Straight-line distance from one listing to the new listing, on the default integer stamps. */
export const distanceToQuery = (row: { rooms: number; color: string; park: number }) =>
  euclid(featuresOf(row), [QUERY.rooms, QUERY.code, QUERY.park]);

export const summarize = (rows: readonly MessyRow[]) => {
  const live = rows.filter((row) => !row.dropped);
  return {
    count: live.length,
    meanRooms: mean(live.map((row) => row.rooms)),
    highShare: live.filter((row) => row.price === 1).length / live.length,
  };
};

/** The K nearest live rows to the new listing, nearest first (ties keep table order), and their majority tag. */
export const nearestAnswer = (rows: readonly MessyRow[], k = K) => {
  const query = [QUERY.rooms, QUERY.code, QUERY.park];
  const neighbors = rows
    .map((row, order) => ({ row, order, distance: euclid(featuresOf(row), query) }))
    .filter((item) => !item.row.dropped)
    .sort((left, right) => left.distance - right.distance || left.order - right.order)
    .slice(0, k);
  const high = neighbors.filter((item) => item.row.price === 1).length;
  return { neighbors, high, answer: high * 2 > neighbors.length ? 1 : 0 };
};

/**
 * Leave-one-out 1-nearest-neighbour accuracy: every live row is predicted from its nearest other row. A
 * duplicate is its twin's nearest neighbour at distance 0, so it is always "predicted" correctly.
 */
export const leaveOneOut = (rows: readonly MessyRow[]) => {
  const live = rows.filter((row) => !row.dropped);
  let correct = 0;
  live.forEach((row, index) => {
    let best = -1;
    let bestDistance = Infinity;
    live.forEach((other, otherIndex) => {
      if (otherIndex === index) return;
      const distance = euclid(featuresOf(row), featuresOf(other));
      if (distance < bestDistance) {
        bestDistance = distance;
        best = otherIndex;
      }
    });
    if (best >= 0 && live[best].price === row.price) correct += 1;
  });
  return { correct, total: live.length, accuracy: live.length === 0 ? 0 : correct / live.length };
};
