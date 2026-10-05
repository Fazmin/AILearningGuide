import { describe, expect, it } from "vitest";
import definition, { hydrateModernAttentionState } from "./module";

describe("modern-attention-at-scale state", () => {
  it("migrates version-1 payloads", () => {
    const mha = hydrateModernAttentionState(
      JSON.stringify({ mechanism: "mha", seqLen: 4096, kvHeads: 8, position: 8, pair: 0 }),
    );
    expect(mha).toMatchObject({ kvHeads: 32, seqLen: 4096, queryPos: 8, keyPos: 0, pair: 0 });
    expect(mha).not.toHaveProperty("mechanism");
    expect(mha).not.toHaveProperty("position");

    const gqa = hydrateModernAttentionState(JSON.stringify({ mechanism: "gqa", kvHeads: 5, seqLen: 30000 }));
    expect(gqa).toMatchObject({ kvHeads: 4, seqLen: 32768 });

    const ssm = hydrateModernAttentionState(JSON.stringify({ mechanism: "ssm", kvHeads: 3, seqLen: 128 }));
    expect(ssm).toMatchObject({ kvHeads: 8, seqLen: 1024 });
  });

  it("clamps current payloads and rejects malformed JSON", () => {
    expect(hydrateModernAttentionState(JSON.stringify({ tileSize: 5, tiles: 99, ropeBase: 7, span: "x" })))
      .toMatchObject({ tileSize: 4, tiles: 4, ropeBase: 10000, span: "full" });
    expect(hydrateModernAttentionState("{")).toEqual(definition.initialState);
  });

  it("round-trips its own serialized state", () => {
    const state = { ...definition.initialState, kvHeads: 1, span: "window", seqLen: 131072, tiles: 2 };
    expect(hydrateModernAttentionState(definition.serializeState(state))).toEqual(state);
  });
});
