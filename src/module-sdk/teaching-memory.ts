/**
 * Training-memory arithmetic: how many bytes the model state costs per parameter.
 *
 * Everything here is multiplication, not measurement. It follows the accounting in the ZeRO paper
 * (Rajbhandari et al., 2019, section 3.1): mixed-precision Adam keeps a 16-bit copy of the weights and
 * of the gradients (2 + 2 bytes per parameter) and 12 bytes of fp32 optimizer state (a master copy of
 * the weights, the momentum, and the variance), 16 bytes in all. The other rows apply the same rule
 * to plain fp32 training and to optimizers with fewer buffers. Activations are deliberately absent.
 */
export type PrecisionId = "fp32" | "mixed";
export type OptimizerId = "sgd" | "momentum" | "adam";

export interface PrecisionChoice {
  id: PrecisionId;
  label: string;
  detail: string;
  /** Bytes per parameter for the weights the forward and backward passes read. */
  weightBytes: number;
  /** Bytes per parameter for the gradient. */
  gradientBytes: number;
  /** Bytes per parameter for an extra fp32 master copy of the weights. Zero when the weights are already fp32. */
  masterBytes: number;
}

export interface OptimizerChoice {
  id: OptimizerId;
  label: string;
  detail: string;
  /** fp32 buffers per parameter: none, one (momentum), or two (Adam's momentum and variance). */
  buffers: number;
  bufferNames: ReadonlyArray<string>;
}

export const PRECISIONS: ReadonlyArray<PrecisionChoice> = [
  {
    id: "fp32",
    label: "fp32",
    detail: "Weights and gradients are 4-byte floats, so there is no separate master copy.",
    weightBytes: 4,
    gradientBytes: 4,
    masterBytes: 0,
  },
  {
    id: "mixed",
    label: "Mixed 16-bit",
    detail: "Weights and gradients are 2-byte floats, and the optimizer keeps a 4-byte master copy.",
    weightBytes: 2,
    gradientBytes: 2,
    masterBytes: 4,
  },
];

export const OPTIMIZERS: ReadonlyArray<OptimizerChoice> = [
  { id: "sgd", label: "SGD", detail: "Keeps no state of its own.", buffers: 0, bufferNames: [] },
  { id: "momentum", label: "Momentum", detail: "Keeps one fp32 velocity per parameter.", buffers: 1, bufferNames: ["momentum"] },
  {
    id: "adam",
    label: "Adam",
    detail: "Keeps two fp32 values per parameter: a running mean and a running variance of the gradient.",
    buffers: 2,
    bufferNames: ["momentum", "variance"],
  },
];

/** Bytes per fp32 optimizer buffer. */
export const STATE_BYTES = 4;

export interface ModelSizeChoice {
  id: string;
  label: string;
  parameters: number;
  detail: string;
}

/** Round parameter counts chosen for clean arithmetic; none of them names a specific released model. */
export const MODEL_SIZES: ReadonlyArray<ModelSizeChoice> = [
  { id: "lab", label: "This lab", parameters: 900, detail: "The 30 × 30 logit table every lab in this track trains." },
  { id: "125m", label: "125 million", parameters: 125_000_000, detail: "A small research-scale transformer." },
  { id: "1.5b", label: "1.5 billion", parameters: 1_500_000_000, detail: "The size of the example in the ZeRO paper's memory analysis." },
  { id: "7.5b", label: "7.5 billion", parameters: 7_500_000_000, detail: "The size of the ZeRO paper's Figure 1 and Table 1." },
  { id: "70b", label: "70 billion", parameters: 70_000_000_000, detail: "A frontier-open-weights scale." },
];

export const DEFAULT_MODEL_SIZE: string = "7.5b";
export const DEFAULT_PRECISION: PrecisionId = "mixed";
export const DEFAULT_OPTIMIZER: OptimizerId = "adam";

export interface MemoryBreakdown {
  parameters: number;
  weightBytesPerParameter: number;
  gradientBytesPerParameter: number;
  /** Master copy plus optimizer buffers: what the ZeRO paper counts as optimizer state. */
  stateBytesPerParameter: number;
  totalBytesPerParameter: number;
  weights: number;
  gradients: number;
  optimizerState: number;
  total: number;
  /** The individual pieces of the optimizer state, in the order they are listed in the table. */
  stateParts: ReadonlyArray<{ name: string; bytesPerParameter: number }>;
}

export function memoryBreakdown(parameters: number, precision: PrecisionId, optimizer: OptimizerId): MemoryBreakdown {
  const precisionChoice = PRECISIONS.find((entry) => entry.id === precision) ?? PRECISIONS[1];
  const optimizerChoice = OPTIMIZERS.find((entry) => entry.id === optimizer) ?? OPTIMIZERS[2];
  const stateParts: { name: string; bytesPerParameter: number }[] = [];
  if (precisionChoice.masterBytes > 0) stateParts.push({ name: "fp32 master copy", bytesPerParameter: precisionChoice.masterBytes });
  for (const name of optimizerChoice.bufferNames) stateParts.push({ name, bytesPerParameter: STATE_BYTES });
  const stateBytesPerParameter = stateParts.reduce((sum, part) => sum + part.bytesPerParameter, 0);
  const totalBytesPerParameter =
    precisionChoice.weightBytes + precisionChoice.gradientBytes + stateBytesPerParameter;
  return {
    parameters,
    weightBytesPerParameter: precisionChoice.weightBytes,
    gradientBytesPerParameter: precisionChoice.gradientBytes,
    stateBytesPerParameter,
    totalBytesPerParameter,
    weights: parameters * precisionChoice.weightBytes,
    gradients: parameters * precisionChoice.gradientBytes,
    optimizerState: parameters * stateBytesPerParameter,
    total: parameters * totalBytesPerParameter,
    stateParts,
  };
}

/** Decimal units, as the ZeRO paper uses: 1 GB is 10^9 bytes. */
export function formatBytes(bytes: number): string {
  const units: ReadonlyArray<[number, string]> = [
    [1e12, "TB"],
    [1e9, "GB"],
    [1e6, "MB"],
    [1e3, "KB"],
  ];
  for (const [size, unit] of units) {
    if (bytes >= size) {
      const value = bytes / size;
      const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
      // Trim trailing zeros after a decimal point only, so 120 stays 120 and 3.60 reads 3.6.
      const text = value.toFixed(digits);
      return `${digits > 0 ? text.replace(/\.?0+$/, "") : text} ${unit}`;
    }
  }
  return `${Math.round(bytes)} B`;
}

/** The device size in the ZeRO paper's Table 1 (32 GB V100s), used only to count how many it would take. */
export const REFERENCE_DEVICE_BYTES = 32e9;

export const devicesNeeded = (bytes: number, deviceBytes = REFERENCE_DEVICE_BYTES) =>
  Math.max(1, Math.ceil(bytes / deviceBytes));
