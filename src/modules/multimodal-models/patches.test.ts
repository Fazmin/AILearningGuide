import { describe, expect, it } from "vitest";
import { hydrateMultimodalState } from "./module";
import { DEFAULT_PIXELS, filters, parsePixels, patchify, realScale } from "./patches";

describe("patch tokenization", () => {
  it("cuts 8×8 into (8/p)² patches of p² pixels", () => {
    const pixels = parsePixels(DEFAULT_PIXELS);
    for (const [p, count] of [
      [2, 16],
      [4, 4],
      [8, 1],
    ]) {
      const patches = patchify(pixels, p);
      expect(patches).toHaveLength(count);
      expect(patches.every((patch) => patch.x.length === p * p)).toBe(true);
    }
    expect(pixels.reduce((sum, bit) => sum + bit, 0)).toBe(17);
  });

  it("uses zero-sum contrast filters and a mean filter", () => {
    for (const p of [2, 4, 8]) {
      const [ink, leftRight, topBottom, diagonal] = filters(p);
      expect(ink.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      for (const filter of [leftRight, topBottom, diagonal]) expect(filter.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 12);
    }
  });

  it("computes W·x and the token for the default patch #2", () => {
    const patch = patchify(parsePixels(DEFAULT_PIXELS), 4)[1];
    expect(patch.wx.map((value) => value.toFixed(3))).toEqual(["0.188", "0.375", "0.125", "0.000"]);
    expect(patch.token.map((value) => value.toFixed(3))).toEqual(["0.258", "0.446", "0.196", "-0.071"]);
  });

  it("is linear: W(a + b) = Wa + Wb on disjoint ink", () => {
    const a = parsePixels("1".repeat(8) + "0".repeat(56));
    const b = parsePixels("0".repeat(56) + "1".repeat(8));
    const both = a.map((bit, index) => bit + b[index]);
    const [pa, pb, pab] = [a, b, both].map((pixels) => patchify(pixels, 4)[0].wx);
    pab.forEach((value, index) => expect(value).toBeCloseTo(pa[index] + pb[index], 12));
  });
});

describe("real-scale arithmetic", () => {
  it("matches ViT-B/16 at 224 and LLaVA-1.5 at 336", () => {
    const vit = realScale(224, 16, true, "llava");
    expect(vit.patchTokens).toBe(196);
    expect(vit.encoderSequence).toBe(197);
    expect(vit.rawPerPatch).toBe(768);
    const llava = realScale(336, 14, true, "llava");
    expect(llava.patchTokens).toBe(576);
    expect(llava.llmImageTokens).toBe(576);
    expect(llava.rawPerPatch).toBe(588);
    expect((llava.contextShare * 100).toFixed(1)).toBe("14.1");
  });

  it("pads a side that the patch does not divide, and keeps images out of the sequence for CLIP and Flamingo-style", () => {
    const odd = realScale(336, 32, false, "llava");
    expect(odd.padded).toBe(352);
    expect(odd.patchTokens).toBe(121);
    expect(realScale(336, 14, true, "clip").llmImageTokens).toBe(0);
    expect(realScale(336, 14, true, "flamingo").llmImageTokens).toBe(0);
  });

  it("quadruples tokens when the side doubles", () => {
    expect(realScale(448, 16, false, "llava").patchTokens / realScale(224, 16, false, "llava").patchTokens).toBe(4);
    expect([realScale(224, 16, false, "llava").patchTokens, realScale(448, 16, false, "llava").patchTokens]).toEqual([196, 784]);
  });

  it("puts image tokens in the language model's sequence only for the LLaVA-style connector", () => {
    const tokens = (connector: "llava" | "clip" | "flamingo") => realScale(336, 14, true, connector).llmImageTokens;
    expect([tokens("llava"), tokens("clip"), tokens("flamingo")]).toEqual([576, 0, 0]);
  });
});

describe("state", () => {
  it("keeps version 1 pixels and fills the new keys", () => {
    const state = hydrateMultimodalState(JSON.stringify({ pixels: "1".repeat(64), patchSize: 2, selected: 30, pixel: 5 }));
    expect(state.pixels).toBe("1".repeat(64));
    expect(state.selected).toBe(15);
    expect(state.connector).toBe("llava");
    expect(state.side).toBe(336);
    expect(hydrateMultimodalState(JSON.stringify({ patchSize: 6 })).patchSize).toBe(4);
  });
});
