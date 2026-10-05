/**
 * A closed-form teaching model of a gradient passing back through a deep stack.
 *
 * It is not a trained network. Each layer multiplies the gradient that reaches it by the same gain,
 * which is what a product of Jacobians does when every layer's Jacobian has the same size. The
 * gradient arriving at the output layer has length 1, so the layer `k` steps back from the output sees
 * a gradient of length `gain ^ k`. A global-norm clip then rescales all of those lengths together by
 * `min(1, clip / total)`, where `total` is the length of the whole gradient across every layer.
 */
export const DEPTH_LAYERS_MIN = 2;
export const DEPTH_LAYERS_MAX = 48;
export const DEPTH_GAIN_MIN = 0.7;
export const DEPTH_GAIN_MAX = 1.4;
export const DEPTH_CLIP_MIN = 0;
export const DEPTH_CLIP_MAX = 10;

export interface DepthReport {
  layers: number;
  gain: number;
  clip: number;
  /** Gradient length at layer 1 (the input side) through layer `layers` (the output), before clipping. */
  norms: number[];
  /** The same lengths after the clip's scale is applied. */
  clipped: number[];
  /** Length of the whole gradient: the square root of the sum of squared per-layer lengths. */
  total: number;
  /** The factor the clip multiplies everything by: 1 when it never binds or is off. */
  scale: number;
  /** Input-layer length divided by output-layer length, which is `gain ^ (layers − 1)`. */
  inputToOutput: number;
  /** True when a clip is set and the total exceeds it. */
  binding: boolean;
}

export function depthReport(layers: number, gain: number, clip: number): DepthReport {
  const norms = Array.from({ length: layers }, (_, index) => gain ** (layers - 1 - index));
  const total = Math.sqrt(norms.reduce((sum, value) => sum + value * value, 0));
  const binding = clip > 0 && total > clip;
  const scale = binding ? clip / total : 1;
  return {
    layers,
    gain,
    clip,
    norms,
    clipped: norms.map((value) => value * scale),
    total,
    scale,
    inputToOutput: norms[0] / norms[layers - 1],
    binding,
  };
}

const SUPERSCRIPT: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

/** Compact, finite text for lengths that span many orders of magnitude, such as 1.22 × 10³. */
export function formatMagnitude(value: number): string {
  if (!Number.isFinite(value)) return "off the chart";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) {
    const [mantissa, exponent] = value.toExponential(2).split("e");
    const power = String(Number(exponent)).replace(/./g, (character) => SUPERSCRIPT[character] ?? character);
    return `${mantissa} × 10${power}`;
  }
  return magnitude >= 100 ? value.toFixed(0) : magnitude >= 10 ? value.toFixed(1) : value.toFixed(3);
}
