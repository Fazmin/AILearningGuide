import { describe, expect, it } from "vitest";
import {
  adapterBytes,
  adapterParameters,
  architectureFor,
  baseParameters,
  BITS,
  CALCULATOR_STATE,
  calculatorState,
  D_MAX,
  D_MIN,
  DEFAULT_CALCULATOR,
  formatBillions,
  formatCount,
  formatMegabytes,
  formatShare,
  LAYERS_MAX,
  matricesOf,
  PRESETS,
  RANK_MAX,
  readCalculator,
  toyShare,
  type Architecture,
  type TargetId,
} from "./calculator";

const preset = (id: string) => PRESETS.find((entry) => entry.id === id) as Architecture;
const small = preset("llama-3.2-1b");
const eight = preset("llama-3-8b");
const seventy = preset("llama-3-70b");
const ALL: TargetId[] = ["q", "k", "v", "o", "mlp"];

describe("the adapter size calculator", () => {
  it("counts a LoRA pair as rank times the sum of the matrix's two dimensions", () => {
    // B is outputs by rank and A is rank by inputs, so a 4096 by 1024 key projection at rank 16 holds 16 * 5120 numbers.
    expect(matricesOf(eight, "k")).toEqual([{ inputs: 4096, outputs: 1024 }]);
    expect(adapterParameters(eight, ["k"], 16).total).toBe(32 * 16 * (4096 + 1024));
    expect(matricesOf(eight, "mlp")).toHaveLength(3);
  });

  it("reproduces the number the PEFT quick tour prints for Llama 3.2 1B with q_proj at rank 8", () => {
    // huggingface.co/docs/peft/quicktour: LoraConfig(target_modules=["q_proj"], r=8) on meta-llama/Llama-3.2-1B prints
    // "trainable params: 524,288 || all params: 1,236,338,688 || trainable%: 0.0424".
    const adapter = adapterParameters(small, ["q"], 8).total;
    expect(adapter).toBe(524_288);
    // PEFT's "all params" includes the adapter itself, so it is the base plus the adapter.
    expect(baseParameters(small) + adapter).toBe(1_236_338_688);
    expect(baseParameters(small)).toBe(1_235_814_400);
    expect(((adapter / (baseParameters(small) + adapter)) * 100).toFixed(4)).toBe("0.0424");
    // Measured against the base model alone the share is the same to the digits PEFT prints.
    expect(formatShare(adapter / baseParameters(small))).toBe("0.0424%");
  });

  it("computes the base sizes of the larger presets from their published shapes", () => {
    expect(baseParameters(eight)).toBe(8_030_261_248);
    expect(formatBillions(baseParameters(eight))).toBe("8.03 billion");
    expect(baseParameters(seventy)).toBe(70_553_706_496);
    expect(formatBillions(baseParameters(seventy))).toBe("70.55 billion");
    expect(formatBillions(baseParameters(small))).toBe("1.24 billion");
  });

  it("matches the lesson's Llama 3 8B figures at rank 16 on every linear layer", () => {
    const all = adapterParameters(eight, ALL, 16);
    expect(all.total).toBe(41_943_040);
    expect(all.byTarget.map((entry) => [entry.id, entry.parameters])).toEqual([
      ["q", 4_194_304],
      ["k", 2_621_440],
      ["v", 2_621_440],
      ["o", 4_194_304],
      ["mlp", 28_311_552],
    ]);
    expect(formatShare(all.total / baseParameters(eight))).toBe("0.522%");
    expect(formatMegabytes(adapterBytes(all.total, 16))).toBe("83.9 MB");
    expect(formatMegabytes(adapterBytes(all.total, 32))).toBe("167.8 MB");
    expect(formatMegabytes(adapterBytes(all.total, 8))).toBe("41.9 MB");
    expect(formatMegabytes(adapterBytes(all.total, 4))).toBe("21.0 MB");
  });

  it("matches the lesson's figure for the attention matrices alone", () => {
    const attention = adapterParameters(eight, ["q", "k", "v", "o"], 16).total;
    expect(attention).toBe(13_631_488);
    expect(formatShare(attention / baseParameters(eight))).toBe("0.170%");
  });

  it("matches the lesson's query-and-value figure and the effect of adding the MLP", () => {
    const qv = adapterParameters(eight, ["q", "v"], 8);
    expect(qv.total).toBe(3_407_872);
    expect(formatShare(qv.total / baseParameters(eight))).toBe("0.0424%");
    // At rank 16 the MLP's three wide matrices dominate: adding them to q and v multiplies the count by 5.15.
    const rank16 = adapterParameters(eight, ["q", "v"], 16).total;
    const withMlp = adapterParameters(eight, ["q", "v", "mlp"], 16).total;
    expect(rank16).toBe(6_815_744);
    expect(withMlp).toBe(35_127_296);
    expect((withMlp / rank16).toFixed(2)).toBe("5.15");
  });

  it("matches the 1B figure and the MLP's share of the 8B adapter", () => {
    const all = adapterParameters(small, ALL, 16);
    expect(all.total).toBe(11_272_192);
    expect(formatShare(all.total / baseParameters(small))).toBe("0.912%");
    const eightAll = adapterParameters(eight, ALL, 16);
    const mlp = eightAll.byTarget.find((entry) => entry.id === "mlp")!.parameters;
    expect(mlp / eightAll.total).toBeGreaterThan(2 / 3);
    expect(((mlp / eightAll.total) * 100).toFixed(1)).toBe("67.5");
  });

  it("matches the 70B figure and the share's fall as the model grows", () => {
    const all = adapterParameters(seventy, ALL, 16);
    expect(all.total).toBe(207_093_760);
    expect(formatShare(all.total / baseParameters(seventy))).toBe("0.294%");
    expect(formatMegabytes(adapterBytes(all.total, 16))).toBe("414.2 MB");
    const shares = [small, eight, seventy].map((entry) => adapterParameters(entry, ALL, 16).total / baseParameters(entry));
    expect(shares[0]).toBeGreaterThan(shares[1]);
    expect(shares[1]).toBeGreaterThan(shares[2]);
  });

  it("scales exactly linearly with rank and with layers", () => {
    for (const entry of PRESETS) {
      const one = adapterParameters(entry, ALL, 1).total;
      expect(adapterParameters(entry, ALL, 64).total).toBe(64 * one);
      expect(adapterParameters({ ...entry, layers: entry.layers * 2 }, ALL, 8).total).toBe(2 * adapterParameters(entry, ALL, 8).total);
    }
    expect(adapterParameters(eight, [], 16).total).toBe(0);
  });

  it("compares the toy's share with a real adapter's at the same rank", () => {
    // The toy trains 2 * rank * 30 of its 900 weights: 240 at rank 4, which is 26.7%.
    expect(toyShare(4) * 900).toBeCloseTo(240, 10);
    expect(formatShare(toyShare(4))).toBe("26.7%");
    const real = adapterParameters(eight, ALL, 4).total / baseParameters(eight);
    expect(formatShare(real)).toBe("0.131%");
    expect(toyShare(4) / real).toBeGreaterThan(200);
    expect(Math.round(toyShare(4) / real)).toBe(204);
    // One matrix of the toy's size is 30 wide; a real one is 4096 wide, so the pair is a far smaller share of it.
    const pair = 4 * (4096 + 4096);
    expect([pair, 4096 * 4096]).toEqual([32_768, 16_777_216]);
    expect(formatShare(pair / (4096 * 4096))).toBe("0.195%");
    expect(formatShare((4 * (30 + 30)) / (30 * 30))).toBe("26.7%");
  });

  it("describes a custom model with Llama 3's ratios", () => {
    const custom = architectureFor({ ...DEFAULT_CALCULATOR, preset: "custom", d: 4096, layers: 32 });
    expect(custom).toMatchObject({ d: 4096, layers: 32, heads: 32, kvHeads: 8, headDim: 128, ffn: 14336, vocab: 128256, tied: false });
    // At Llama 3 8B's own dimension and depth the ratios reproduce the published architecture exactly.
    expect(baseParameters(custom)).toBe(baseParameters(eight));
    expect(adapterParameters(custom, ALL, 16).total).toBe(adapterParameters(eight, ALL, 16).total);
  });

  it("formats counts, shares and sizes", () => {
    expect(formatCount(41_943_040)).toBe("41,943,040");
    expect(formatCount(999)).toBe("999");
    expect(formatCount(0)).toBe("0");
    expect(formatShare(0.2667)).toBe("26.7%");
    expect(formatShare(0.0522)).toBe("5.22%");
    expect(formatShare(0.00522)).toBe("0.522%");
    expect(formatShare(0.000424)).toBe("0.0424%");
    expect(formatMegabytes(1_234_567_890)).toBe("1,235 MB");
    expect(formatMegabytes(5_000_000)).toBe("5.00 MB");
  });
});

describe("the calculator's state", () => {
  it("keeps the defaults: Llama 3 8B, every linear layer, rank 16, 16-bit", () => {
    expect(readCalculator({})).toEqual(DEFAULT_CALCULATOR);
    expect(calculatorState(DEFAULT_CALCULATOR)).toEqual(CALCULATOR_STATE);
    expect(readCalculator(CALCULATOR_STATE)).toEqual(DEFAULT_CALCULATOR);
  });

  it("lets a named preset fix the shape, whatever size the state carries", () => {
    const settings = readCalculator({ calcPreset: "llama-3-70b", calcD: 512, calcLayers: 2 });
    expect([settings.d, settings.layers]).toEqual([8192, 80]);
    const custom = readCalculator({ calcPreset: "custom", calcD: 5000, calcLayers: 40 });
    expect([custom.d, custom.layers]).toEqual([5120, 40]);
  });

  it("clamps every number and drops anything that is not an option", () => {
    const wild = readCalculator({
      calcPreset: "<script>",
      calcD: 1e9,
      calcLayers: 1e9,
      calcRank: 1e9,
      calcTargets: ["q", "q", "mlp", "bogus", 7, null],
      calcBits: 3,
    });
    expect(wild).toEqual({ ...DEFAULT_CALCULATOR, rank: RANK_MAX, targets: ["q", "mlp"] });
    const custom = (patch: Record<string, unknown>) => readCalculator({ calcPreset: "custom", ...patch });
    expect([custom({ calcD: 1e9 }).d, custom({ calcD: -1e9 }).d, custom({ calcD: 1e999 }).d]).toEqual([D_MAX, D_MIN, 4096]);
    expect([custom({ calcLayers: 1e9 }).layers, custom({ calcLayers: -5 }).layers, custom({ calcLayers: NaN }).layers]).toEqual([LAYERS_MAX, 1, 32]);
    expect(readCalculator({ calcRank: -1e9 }).rank).toBe(1);
    expect(readCalculator({ calcRank: "8", calcTargets: "q", calcBits: "16" })).toEqual(DEFAULT_CALCULATOR);
    expect(readCalculator({ calcTargets: [] }).targets).toEqual([]);
    for (const bits of BITS) expect(readCalculator({ calcBits: bits }).bits).toBe(bits);
  });

  it("keeps the targets in the card's order and without repeats", () => {
    expect(readCalculator({ calcTargets: ["mlp", "v", "q", "v"] }).targets).toEqual(["q", "v", "mlp"]);
  });

  it("stays finite and exact at the largest setting the controls allow", () => {
    const largest = architectureFor({ ...DEFAULT_CALCULATOR, preset: "custom", d: D_MAX, layers: LAYERS_MAX, rank: RANK_MAX });
    const count = adapterParameters(largest, ALL, RANK_MAX).total;
    expect(Number.isSafeInteger(count)).toBe(true);
    expect(Number.isSafeInteger(baseParameters(largest))).toBe(true);
    expect(count).toBeLessThan(baseParameters(largest));
  });
});
