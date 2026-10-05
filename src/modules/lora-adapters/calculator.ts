/**
 * The closed-form arithmetic behind the "Adapter size calculator" card. Nothing here is trained or
 * measured: a LoRA pair on a weight matrix with `inputs` columns and `outputs` rows holds
 * `rank * (inputs + outputs)` numbers, and the rest is multiplication.
 *
 * The three presets are published architectures. Layers, model dimension, feed-forward width and head
 * counts come from Table 3 of the Llama 3 paper (arXiv 2407.21783) for the 8B and 70B models and from the
 * released config.json files for the 1B model and for the vocabulary size; see the card text for sources.
 */
import type { ModuleState } from "@app/module-sdk";

export type TargetId = "q" | "k" | "v" | "o" | "mlp";

export const TARGETS: ReadonlyArray<{ id: TargetId; label: string; detail: string }> = [
  { id: "q", label: "q", detail: "query projection" },
  { id: "k", label: "k", detail: "key projection" },
  { id: "v", label: "v", detail: "value projection" },
  { id: "o", label: "o", detail: "attention output projection" },
  { id: "mlp", label: "MLP", detail: "gate, up and down projections" },
];
const TARGET_IDS = TARGETS.map((target) => target.id);

export interface Architecture {
  id: string;
  name: string;
  /** Model (hidden) dimension d. */
  d: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  /** Width of the feed-forward (MLP) block. */
  ffn: number;
  vocab: number;
  /** Whether the output projection reuses the embedding table. */
  tied: boolean;
}

export const PRESETS: ReadonlyArray<Architecture> = [
  { id: "llama-3.2-1b", name: "Llama 3.2 1B", d: 2048, layers: 16, heads: 32, kvHeads: 8, headDim: 64, ffn: 8192, vocab: 128256, tied: true },
  { id: "llama-3-8b", name: "Llama 3 8B", d: 4096, layers: 32, heads: 32, kvHeads: 8, headDim: 128, ffn: 14336, vocab: 128256, tied: false },
  { id: "llama-3-70b", name: "Llama 3 70B", d: 8192, layers: 80, heads: 64, kvHeads: 8, headDim: 128, ffn: 28672, vocab: 128256, tied: false },
];
export const CUSTOM = "custom";
export const PRESET_IDS: ReadonlyArray<string> = [...PRESETS.map((preset) => preset.id), CUSTOM];

/** Bounds of the calculator's controls. */
export const D_MIN = 256;
export const D_MAX = 16384;
export const D_STEP = 256;
export const LAYERS_MIN = 1;
export const LAYERS_MAX = 126;
export const RANK_MIN = 1;
export const RANK_MAX = 256;
export const BITS = [32, 16, 8, 4] as const;
export type Bits = (typeof BITS)[number];

export interface CalculatorSettings {
  preset: string;
  d: number;
  layers: number;
  rank: number;
  targets: TargetId[];
  bits: Bits;
}

export const DEFAULT_CALCULATOR: CalculatorSettings = {
  preset: "llama-3-8b",
  d: 4096,
  layers: 32,
  rank: 16,
  targets: ["q", "k", "v", "o", "mlp"],
  bits: 16,
};

/** The state keys the card stores, so the module can tell a calculator patch from the rest. */
export const CALCULATOR_STATE: ModuleState = {
  calcPreset: DEFAULT_CALCULATOR.preset,
  calcD: DEFAULT_CALCULATOR.d,
  calcLayers: DEFAULT_CALCULATOR.layers,
  calcRank: DEFAULT_CALCULATOR.rank,
  calcTargets: [...DEFAULT_CALCULATOR.targets],
  calcBits: DEFAULT_CALCULATOR.bits,
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function readCalculator(source: Record<string, unknown>): CalculatorSettings {
  const preset = typeof source.calcPreset === "string" && PRESET_IDS.includes(source.calcPreset) ? source.calcPreset : DEFAULT_CALCULATOR.preset;
  const known = PRESETS.find((entry) => entry.id === preset);
  // A named preset fixes the shape, so a state can never name one and carry a different size.
  const d = known
    ? known.d
    : finite(source.calcD)
      ? Math.round(clamp(source.calcD, D_MIN, D_MAX) / D_STEP) * D_STEP
      : DEFAULT_CALCULATOR.d;
  const layers = known
    ? known.layers
    : finite(source.calcLayers)
      ? Math.round(clamp(source.calcLayers, LAYERS_MIN, LAYERS_MAX))
      : DEFAULT_CALCULATOR.layers;
  const rank = finite(source.calcRank) ? Math.round(clamp(source.calcRank, RANK_MIN, RANK_MAX)) : DEFAULT_CALCULATOR.rank;
  const requested = Array.isArray(source.calcTargets) ? source.calcTargets : DEFAULT_CALCULATOR.targets;
  const targets = TARGET_IDS.filter((id) => requested.includes(id));
  const bits = BITS.includes(source.calcBits as Bits) ? (source.calcBits as Bits) : DEFAULT_CALCULATOR.bits;
  return { preset, d, layers, rank, targets, bits };
}

/** The settings as the lab stores them. */
export function calculatorState(settings: CalculatorSettings): ModuleState {
  return {
    calcPreset: settings.preset,
    calcD: settings.d,
    calcLayers: settings.layers,
    calcRank: settings.rank,
    calcTargets: [...settings.targets],
    calcBits: settings.bits,
  };
}

/**
 * The architecture a set of settings describes. A preset is used as published; a custom model follows
 * Llama 3's ratios: key and value width a quarter of d, feed-forward width 3.5 times d, the 128,256-entry
 * vocabulary, and an untied output projection.
 */
export function architectureFor(settings: CalculatorSettings): Architecture {
  const known = PRESETS.find((entry) => entry.id === settings.preset);
  if (known) return known;
  const headDim = 128;
  return {
    id: CUSTOM,
    name: "Custom",
    d: settings.d,
    layers: settings.layers,
    heads: Math.max(1, Math.round(settings.d / headDim)),
    kvHeads: Math.max(1, Math.round(settings.d / headDim / 4)),
    headDim,
    ffn: Math.round(3.5 * settings.d),
    vocab: 128256,
    tied: false,
  };
}

export interface MatrixShape {
  inputs: number;
  outputs: number;
}

/** The weight matrices of one transformer layer that a target group adapts. */
export function matricesOf(architecture: Architecture, target: TargetId): MatrixShape[] {
  const { d, ffn } = architecture;
  const queryWidth = architecture.heads * architecture.headDim;
  const keyValueWidth = architecture.kvHeads * architecture.headDim;
  switch (target) {
    case "q":
      return [{ inputs: d, outputs: queryWidth }];
    case "k":
    case "v":
      return [{ inputs: d, outputs: keyValueWidth }];
    case "o":
      return [{ inputs: queryWidth, outputs: d }];
    case "mlp":
      return [
        { inputs: d, outputs: ffn },
        { inputs: d, outputs: ffn },
        { inputs: ffn, outputs: d },
      ];
  }
}

/** Numbers in one LoRA pair on a matrix: B is outputs by rank and A is rank by inputs. */
export const pairParameters = (matrix: MatrixShape, rank: number) => rank * (matrix.inputs + matrix.outputs);

export interface AdapterCount {
  total: number;
  /** Trainable numbers per target group, across every layer. */
  byTarget: Array<{ id: TargetId; label: string; parameters: number }>;
}

export function adapterParameters(architecture: Architecture, targets: ReadonlyArray<TargetId>, rank: number): AdapterCount {
  const byTarget = TARGETS.filter((target) => targets.includes(target.id)).map((target) => ({
    id: target.id,
    label: target.id === "mlp" ? "MLP (gate, up, down)" : `${target.id} projection`,
    parameters:
      architecture.layers *
      matricesOf(architecture, target.id).reduce((sum, matrix) => sum + pairParameters(matrix, rank), 0),
  }));
  return { total: byTarget.reduce((sum, entry) => sum + entry.parameters, 0), byTarget };
}

/**
 * Parameters of the base model: the embedding table (and a separate output table unless tied), every
 * layer's attention and MLP matrices and its two norm vectors, and the final norm vector.
 */
export function baseParameters(architecture: Architecture): number {
  const { d, ffn, layers, vocab } = architecture;
  const queryWidth = architecture.heads * architecture.headDim;
  const keyValueWidth = architecture.kvHeads * architecture.headDim;
  const attention = d * queryWidth + 2 * d * keyValueWidth + queryWidth * d;
  const perLayer = attention + 3 * d * ffn + 2 * d;
  const embeddings = vocab * d * (architecture.tied ? 1 : 2);
  return embeddings + layers * perLayer + d;
}

/** Bytes for `parameters` numbers stored at `bits` bits each. */
export const adapterBytes = (parameters: number, bits: number) => (parameters * bits) / 8;

/** The toy's trainable share: 2 · rank · 30 of its 30 × 30 table. */
export const toyShare = (rank: number, size = 30) => (rank * 2 * size) / (size * size);

/** 41943040 as "41,943,040". */
export function formatCount(value: number): string {
  return Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** A fraction as a percentage with enough digits to show its first significant ones: 0.0424%, 0.522%, 26.7%. */
export function formatShare(fraction: number): string {
  const percent = fraction * 100;
  const digits = percent >= 10 ? 1 : percent >= 1 ? 2 : percent >= 0.1 ? 3 : 4;
  return `${percent.toFixed(digits)}%`;
}

/** Megabytes of 10^6 bytes: two decimals under 10, one decimal under 1,000, grouped digits above. */
export function formatMegabytes(bytes: number): string {
  const megabytes = bytes / 1e6;
  if (megabytes >= 1000) return `${formatCount(megabytes)} MB`;
  return `${megabytes.toFixed(megabytes >= 10 ? 1 : 2)} MB`;
}

/** A parameter count in words: 8.03 billion, 1.24 billion, 524 thousand. */
export function formatBillions(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} billion`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} million`;
  return `${formatCount(value)}`;
}
