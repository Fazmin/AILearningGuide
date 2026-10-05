import { describe, expect, it } from "vitest";
import definition from "./module";
import { MAX_DATASET_LINE_LENGTH, MAX_DATASET_LINES } from "./sft";

describe("instruction-tuning state", () => {
  it("keeps a version 1 snapshot on whole-sequence loss, which is what it trained with", () => {
    const hydrated = definition.hydrateState(JSON.stringify({ useSystem: false, useRoles: true, useEnd: true, epochs: 12 }));
    expect(hydrated).toMatchObject({ useSystem: false, epochs: 12, lossMask: "all" });
  });

  it("defaults new state to assistant-only loss and rejects unknown values", () => {
    expect(definition.hydrateState("{}").lossMask).toBe("assistant");
    expect(definition.hydrateState(JSON.stringify({ lossMask: "prompt" })).lossMask).toBe("assistant");
  });

  it("reads a version 2 payload, which had no serveWith, and serves the chat template", () => {
    const version2 = JSON.stringify({
      useSystem: true,
      useRoles: false,
      useEnd: true,
      lossMask: "assistant",
      epochs: 20,
      dataset: "a | b.",
    });
    expect(definition.stateVersion).toBe(3);
    expect(definition.hydrateState(version2)).toEqual({
      useSystem: true,
      useRoles: false,
      useEnd: true,
      lossMask: "assistant",
      epochs: 20,
      dataset: "a | b.",
      serveWith: "template",
    });
  });

  it("validates every key, so a crafted link cannot carry a wrong type or a value out of range", () => {
    const hydrated = definition.hydrateState(
      JSON.stringify({ useSystem: "yes", useRoles: 1, useEnd: null, lossMask: "assistant", epochs: 1e9, serveWith: "everything", dataset: 5, extra: "x" }),
    );
    expect(hydrated).toEqual({ ...definition.initialState, epochs: 80 });
    expect(definition.hydrateState(JSON.stringify({ epochs: -1e9 })).epochs).toBe(1);
    expect(definition.hydrateState(JSON.stringify({ serveWith: "none" })).serveWith).toBe("none");
  });

  it("cuts a huge dataset to the editor's limits", () => {
    const huge = Array.from({ length: 2000 }, () => "word ".repeat(400)).join("\n");
    const dataset = definition.hydrateState(JSON.stringify({ dataset: huge })).dataset as string;
    expect(dataset.split("\n")).toHaveLength(MAX_DATASET_LINES);
    expect(dataset.length).toBeLessThanOrEqual(MAX_DATASET_LINES * (MAX_DATASET_LINE_LENGTH + 1));
  });
});
