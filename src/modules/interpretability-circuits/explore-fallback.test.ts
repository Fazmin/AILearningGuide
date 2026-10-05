import { act, createElement, useState } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";

/** When the weights cannot load, the lab falls back to the cached period-13 attention and says so. */
describe("the circuits lab when the weights do not load", () => {
  it("shows the cached period-13 attention with its guides, and no live results", async () => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    vi.stubGlobal("fetch", async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) }));
    const container = document.createElement("div");
    document.body.appendChild(container);
    function Harness() {
      const [state, setState] = useState<ModuleState>({ ...definition.initialState });
      return createElement(definition.Explore, {
        state,
        setState: (patch: Partial<ModuleState>) => setState((previous) => ({ ...previous, ...patch }) as ModuleState),
        currentStep: 0,
        mode: "standard" as const,
        narrate: () => undefined,
      });
    }
    const root = createRoot(container);
    act(() => root.render(createElement(Harness)));
    for (let tries = 0; tries < 100 && !container.textContent?.includes("Weights unavailable"); tries += 1) {
      await act(async () => {
        await new Promise((done) => setTimeout(done, 20));
      });
    }
    const text = container.textContent ?? "";
    expect(text).toContain("Weights unavailable · cached view");
    expect(text).toContain("Cached attention for “QXZRKWMPJVBHDQXZRKWMPJVB”");
    expect(text).toContain("Head ablation needs the model weights");
    expect(text).toContain("The live sweep needs the model weights");
    // The cached example is a period-13 repeat, so the induction-key outlines are drawn for it.
    expect(text).toContain("i − 13 + 1");
    expect(container.querySelectorAll(".ci-attn__guide--induction").length).toBe(11);
    expect(container.querySelectorAll(".ci-attn__guide--previous").length).toBe(11);
    act(() => root.unmount());
    container.remove();
  }, 30000);
});
