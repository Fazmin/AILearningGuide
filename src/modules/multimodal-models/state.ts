import type { ModuleState } from "@app/module-sdk";
import { boundedTemperature, DEFAULT_TEMPERATURE, pairingOf } from "./contrastive";
import { DEFAULT_PIXELS, PATCH_SIZES, SIDES, SIZE, VIT_PATCHES } from "./patches";

/**
 * Version 1 stored only pixels, patchSize, selected and pixel. Version 2 added side, vitPatch, cls and
 * connector. Version 3 adds the two Contrastive pairs controls, `temperature` and `pairing`. No older
 * value needs translating: every missing key takes its default.
 */
export const initialState: ModuleState = {
  pixels: DEFAULT_PIXELS,
  patchSize: 4,
  selected: 1,
  pixel: 0,
  side: 336,
  vitPatch: 14,
  cls: "on",
  connector: "llava",
  temperature: DEFAULT_TEMPERATURE,
  pairing: "true",
};

const clampWhole = (value: unknown, low: number, high: number, fallback: number) =>
  Math.min(high, Math.max(low, typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback));

function oneOf<T extends number>(value: unknown, allowed: readonly T[], fallback: T): T {
  return (allowed as readonly unknown[]).includes(value) ? (value as T) : fallback;
}

/** The pad is 64 cells; longer text is cut before it is read, so a pasted megabyte costs nothing. */
export const cleanPixels = (value: unknown) =>
  typeof value === "string"
    ? value
        .slice(0, SIZE * SIZE)
        .replace(/[^01]/g, "0")
        .padEnd(SIZE * SIZE, "0")
    : DEFAULT_PIXELS;

/** Rebuilds a state from untrusted input: unknown keys dropped, every field validated or clamped. */
export function sanitizeState(value: unknown): ModuleState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initialState };
  const source = value as Record<string, unknown>;
  const patchSize = oneOf(source.patchSize, PATCH_SIZES, 4);
  const patches = (SIZE / patchSize) ** 2;
  return {
    pixels: cleanPixels(source.pixels),
    patchSize,
    selected: clampWhole(source.selected, 0, patches - 1, Math.min(patches - 1, initialState.selected as number)),
    pixel: clampWhole(source.pixel, 0, SIZE * SIZE - 1, 0),
    side: oneOf(source.side, SIDES, 336),
    vitPatch: oneOf(source.vitPatch, VIT_PATCHES, 14),
    cls: source.cls === "off" ? "off" : "on",
    connector: source.connector === "clip" || source.connector === "flamingo" ? source.connector : "llava",
    temperature: boundedTemperature(source.temperature),
    pairing: pairingOf(source.pairing),
  };
}
