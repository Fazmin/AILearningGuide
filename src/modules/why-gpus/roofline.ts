/**
 * Roofline arithmetic for one layer operation on an illustrative accelerator.
 *
 * Everything here is closed-form: FLOPs and bytes are counted from the shapes,
 * and time is the larger of "arithmetic at peak" and "bytes at full bandwidth".
 * Nothing runs on a GPU.
 */

export const PRECISIONS = [
  // Round numbers of the same order as an A100-class datacenter GPU:
  // fp32 on ordinary vector lanes, fp16 and int8 on matrix (tensor) units, dense.
  { id: "fp32", label: "fp32", bytes: 4, peakTflops: 20, unit: "TFLOP/s" },
  { id: "fp16", label: "fp16", bytes: 2, peakTflops: 300, unit: "TFLOP/s" },
  { id: "int8", label: "int8", bytes: 1, peakTflops: 600, unit: "TOP/s" },
] as const;

export type PrecisionId = (typeof PRECISIONS)[number]["id"];
export type Precision = (typeof PRECISIONS)[number];

export const OPERATIONS = [
  { id: "matmul", label: "Matmul X·W" },
  { id: "add", label: "Elementwise add" },
] as const;

export type OperationId = (typeof OPERATIONS)[number]["id"];

export const BATCH_MIN = 1;
export const BATCH_MAX = 4096;
export const WIDTH_MIN = 512;
export const WIDTH_MAX = 16384;
export const BANDWIDTH_MIN = 100;
export const BANDWIDTH_MAX = 5000;
export const BATCH_TRAIL = [1, 4, 16, 64, 256, 1024, 4096] as const;

export const precisionById = (id: string): Precision =>
  PRECISIONS.find((item) => item.id === id) ?? PRECISIONS[1];

export interface Workload {
  flops: number;
  bytes: number;
  /** FLOPs per byte moved. */
  intensity: number;
  /** Independent output values the operation produces. */
  outputs: number;
  /** Length of the dot product behind each output (1 for an add). */
  dotLength: number;
}

/**
 * Y = X·W with X of shape batch×width and W of shape width×width, or the
 * elementwise add Y = X + R on two batch×width tensors. Bytes assume each
 * operand is read once and the result written once, the best case that
 * perfect on-chip reuse (tiling) can reach.
 */
export function workload(
  operation: OperationId,
  batch: number,
  width: number,
  bytesPerElement: number,
): Workload {
  if (operation === "matmul") {
    const flops = 2 * batch * width * width;
    const elements = width * width + 2 * batch * width;
    const bytes = elements * bytesPerElement;
    return { flops, bytes, intensity: flops / bytes, outputs: batch * width, dotLength: width };
  }
  const flops = batch * width;
  const bytes = 3 * batch * width * bytesPerElement;
  return { flops, bytes, intensity: flops / bytes, outputs: batch * width, dotLength: 1 };
}

/** The intensity at which the bandwidth roof meets the compute roof, in FLOP per byte. */
export const ridgePoint = (peakFlops: number, bandwidthBytes: number) => peakFlops / bandwidthBytes;

/** Attainable FLOP/s = min(peak, bandwidth × intensity). */
export const attainable = (peakFlops: number, bandwidthBytes: number, intensity: number) =>
  Math.min(peakFlops, bandwidthBytes * intensity);

export interface Timing {
  computeSeconds: number;
  memorySeconds: number;
  seconds: number;
  bound: "memory" | "compute";
  attainableFlops: number;
  utilization: number;
}

/** Roofline time: arithmetic and traffic overlap perfectly, so the slower one sets the clock. */
export function timing(work: Workload, peakFlops: number, bandwidthBytes: number): Timing {
  const computeSeconds = work.flops / peakFlops;
  const memorySeconds = work.bytes / bandwidthBytes;
  const seconds = Math.max(computeSeconds, memorySeconds);
  const attainableFlops = work.flops / seconds;
  return {
    computeSeconds,
    memorySeconds,
    seconds,
    bound: memorySeconds > computeSeconds ? "memory" : "compute",
    attainableFlops,
    utilization: attainableFlops / peakFlops,
  };
}

/**
 * Smallest batch at which a width×width matmul reaches the ridge:
 * 2Bd² / ((d² + 2Bd)·b) = R  ⇒  B = R·b·d / (2(d − R·b)).
 * Returns Infinity when even an unbounded batch stays below the ridge (d/b ≤ R).
 */
export function batchAtRidge(width: number, bytesPerElement: number, ridge: number) {
  const rb = ridge * bytesPerElement;
  if (width <= rb) return Number.POSITIVE_INFINITY;
  return (rb * width) / (2 * (width - rb));
}

export interface Scenario {
  operation: OperationId;
  batch: number;
  width: number;
  precision: Precision;
  /** GB/s */
  bandwidth: number;
}

export function evaluate(scenario: Scenario) {
  const peakFlops = scenario.precision.peakTflops * 1e12;
  const bandwidthBytes = scenario.bandwidth * 1e9;
  const work = workload(scenario.operation, scenario.batch, scenario.width, scenario.precision.bytes);
  const time = timing(work, peakFlops, bandwidthBytes);
  const ridge = ridgePoint(peakFlops, bandwidthBytes);
  return { work, time, ridge, peakFlops, bandwidthBytes };
}

/** Wall-clock under three single changes, so "what would help" is computed rather than asserted. */
export function whatIf(scenario: Scenario) {
  const base = evaluate(scenario).time.seconds;
  const doublePeak = timing(
    workload(scenario.operation, scenario.batch, scenario.width, scenario.precision.bytes),
    scenario.precision.peakTflops * 2e12,
    scenario.bandwidth * 1e9,
  ).seconds;
  const doubleBandwidth = evaluate({ ...scenario, bandwidth: scenario.bandwidth * 2 }).time.seconds;
  const index = PRECISIONS.findIndex((item) => item.id === scenario.precision.id);
  const smaller = PRECISIONS[index + 1];
  const smallerPrecision = smaller ? evaluate({ ...scenario, precision: smaller }).time.seconds : undefined;
  return { base, doublePeak, doubleBandwidth, smallerPrecision, smallerLabel: smaller?.label };
}

export function formatSeconds(seconds: number) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  if (seconds >= 1e-3) return `${(seconds * 1e3).toFixed(seconds >= 0.1 ? 0 : 2)} ms`;
  if (seconds >= 1e-6) return `${(seconds * 1e6).toFixed(seconds >= 1e-4 ? 0 : seconds >= 1e-5 ? 1 : 2)} µs`;
  return `${(seconds * 1e9).toFixed(1)} ns`;
}

export function formatCount(value: number, unit: string) {
  const steps: Array<[number, string]> = [
    [1e12, "T"],
    [1e9, "G"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [scale, prefix] of steps) {
    if (value >= scale) {
      const scaled = value / scale;
      return `${scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2)} ${prefix}${unit}`;
    }
  }
  return `${value.toFixed(0)} ${unit}`;
}

export function formatIntensity(value: number) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
