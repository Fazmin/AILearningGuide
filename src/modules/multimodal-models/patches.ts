/**
 * Patch tokenization, computed for real on an 8×8 binary image, plus the shape
 * arithmetic of real vision encoders and connectors.
 *
 * The patch embedding is a genuine linear map z = W·x + p. W holds four
 * hand-set filters instead of learned ones, so each output number is readable.
 */

export const SIZE = 8;
export const PATCH_SIZES = [2, 4, 8] as const;
export type PatchSize = (typeof PATCH_SIZES)[number];

/** A blocky 5, row by row. */
export const DEFAULT_PIXELS = [
  "01111100",
  "01000000",
  "01000000",
  "01111000",
  "00000100",
  "00000100",
  "01111000",
  "00000000",
].join("");

export const parsePixels = (value: string) =>
  value
    .replace(/[^01]/g, "0")
    .padEnd(SIZE * SIZE, "0")
    .slice(0, SIZE * SIZE)
    .split("")
    .map((bit): number => (bit === "1" ? 1 : 0));

export const FILTER_NAMES = ["ink", "left − right", "top − bottom", "╲ − ╱"] as const;

/** Four p×p filters: mean ink, horizontal contrast, vertical contrast, diagonal contrast. */
export function filters(p: number): number[][] {
  const cells = p * p;
  const half = cells / 2;
  const ink: number[] = [];
  const leftRight: number[] = [];
  const topBottom: number[] = [];
  const diagonal: number[] = [];
  for (let row = 0; row < p; row += 1) {
    for (let column = 0; column < p; column += 1) {
      ink.push(1 / cells);
      leftRight.push(column < p / 2 ? 1 / half : -1 / half);
      topBottom.push(row < p / 2 ? 1 / half : -1 / half);
      diagonal.push(row === column ? 1 / p : row + column === p - 1 ? -1 / p : 0);
    }
  }
  return [ink, leftRight, topBottom, diagonal];
}

/** Fixed 2-D sinusoidal position code, scaled small; a stand-in for ViT's learned position embeddings. */
export function positionCode(row: number, column: number, perSide: number): number[] {
  const angle = (index: number) => (Math.PI * (index + 0.5)) / Math.max(1, perSide);
  return [0.1 * Math.sin(angle(row)), 0.1 * Math.cos(angle(row)), 0.1 * Math.sin(angle(column)), 0.1 * Math.cos(angle(column))];
}

export interface Patch {
  id: number;
  row: number;
  column: number;
  /** Flattened pixels, row-major: the vector x the projection reads. */
  x: number[];
  /** W·x: one number per filter. */
  wx: number[];
  position: number[];
  /** The token: W·x + position. */
  token: number[];
}

export function patchify(pixels: readonly number[], p: number): Patch[] {
  const perSide = SIZE / p;
  const bank = filters(p);
  const patches: Patch[] = [];
  for (let row = 0; row < perSide; row += 1) {
    for (let column = 0; column < perSide; column += 1) {
      const x: number[] = [];
      for (let y = 0; y < p; y += 1) {
        for (let xOffset = 0; xOffset < p; xOffset += 1) {
          x.push(pixels[(row * p + y) * SIZE + column * p + xOffset] ?? 0);
        }
      }
      const wx = bank.map((filter) => filter.reduce((sum, weight, index) => sum + weight * x[index], 0));
      const position = positionCode(row, column, perSide);
      patches.push({
        id: row * perSide + column,
        row,
        column,
        x,
        wx,
        position,
        token: wx.map((value, index) => value + position[index]),
      });
    }
  }
  return patches;
}

/* ---- Real-scale arithmetic ---- */

export const SIDES = [224, 336, 448] as const;
export const VIT_PATCHES = [14, 16, 32] as const;
export type Connector = "clip" | "llava" | "flamingo";

/** Encoder hidden width by patch size, following the common CLIP ViT variants. */
export const ENCODERS: Record<number, { name: string; width: number }> = {
  14: { name: "ViT-L/14", width: 1024 },
  16: { name: "ViT-B/16", width: 768 },
  32: { name: "ViT-B/32", width: 768 },
};

export const LLM_WIDTH = 4096;
export const CONTEXT = 4096;
export const RESAMPLER_LATENTS = 64;

export interface ScaleStage {
  name: string;
  shape: string;
  note: string;
}

export interface Scale {
  perSide: number;
  padded: number;
  patchTokens: number;
  encoderSequence: number;
  rawPerPatch: number;
  encoder: { name: string; width: number };
  llmImageTokens: number;
  stages: ScaleStage[];
  contextShare: number;
}

export function realScale(side: number, patch: number, cls: boolean, connector: Connector): Scale {
  const perSide = Math.ceil(side / patch);
  const padded = perSide * patch;
  const patchTokens = perSide * perSide;
  const encoderSequence = patchTokens + (cls ? 1 : 0);
  const rawPerPatch = patch * patch * 3;
  const encoder = ENCODERS[patch] ?? ENCODERS[16];
  const llmImageTokens = connector === "llava" ? patchTokens : 0;
  const fmt = (value: number) => value.toLocaleString("en-US");
  const stages: ScaleStage[] = [
    {
      name: "Pixels",
      shape: `${fmt(side)} × ${fmt(side)} × 3`,
      note: padded === side ? "RGB image" : `${side} is not a multiple of ${patch}; resized or padded to ${padded}`,
    },
    {
      name: "Patches",
      shape: `${fmt(patchTokens)} × ${fmt(rawPerPatch)}`,
      note: `${perSide} × ${perSide} tiles, each ${patch}·${patch}·3 values`,
    },
    {
      name: `${encoder.name} encoder`,
      shape: `${fmt(encoderSequence)} × ${fmt(encoder.width)}`,
      note: cls ? "linear patch embedding, + position, + 1 CLS token, then transformer layers" : "linear patch embedding, + position, then transformer layers",
    },
  ];
  if (connector === "clip") {
    stages.push(
      {
        name: "Pooled + projected",
        shape: `1 × ${patch === 14 ? 768 : 512}`,
        note: "one vector per image in CLIP's shared image–text space (768 for ViT-L/14, 512 for ViT-B)",
      },
      { name: "Contrastive score", shape: "1 number per caption", note: "cosine with each caption's text vector; nothing enters an LLM" },
    );
  } else if (connector === "llava") {
    stages.push(
      { name: "MLP projector", shape: `${fmt(patchTokens)} × ${fmt(LLM_WIDTH)}`, note: "patch features only (CLS dropped), two layers into the LLM's embedding width" },
      { name: "LLM sequence", shape: `${fmt(patchTokens)} image tokens + text`, note: "replaces the <image> placeholder; ordinary self-attention" },
    );
  } else {
    stages.push(
      { name: "Perceiver resampler", shape: `${RESAMPLER_LATENTS} × d`, note: `${RESAMPLER_LATENTS} learned queries cross-attend to all ${fmt(encoderSequence)} features` },
      { name: "Gated cross-attention", shape: "text tokens only in the LLM", note: "inserted layers let text queries read the 64 visual vectors" },
    );
  }
  return {
    perSide,
    padded,
    patchTokens,
    encoderSequence,
    rawPerPatch,
    encoder,
    llmImageTokens,
    stages,
    contextShare: llmImageTokens / CONTEXT,
  };
}
