/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { teachingTransformer } from "@app/module-sdk";
import type { ModuleContext, ModuleState } from "@app/module-sdk";
import { ACTIVE, encode, reconstruct, FEATURES, WIDTH } from "./sae";
import { sae, transformer } from "./test-support";

/**
 * An end-to-end render of the lab in jsdom. ONNX Runtime and its worker cannot run here, so the
 * model runtime is replaced by the same TypeScript forward pass and SAE encoder the other tests
 * use; everything above that (state, effects, cards, the steering card, the continuation button)
 * is the real code.
 */
const bytes = (path: string) => new Uint8Array(readFileSync(resolve(process.cwd(), path)));
const files = {
  "tiny-transformer": bytes("src/modules/attention/assets/tiny-transformer.onnx"),
  "residual-sae": bytes("src/modules/interpretability-features/assets/residual-sae.onnx"),
};

vi.mock("@app/model-runtime/client", () => ({
  runTeachingTransformer: async () => {
    throw new Error("not used");
  },
  runTeachingModel: async (url: string, feeds: Record<string, { data: number[]; dims: number[] }>) => {
    if (url.includes("tiny-transformer")) {
      const ids = feeds.input_ids.data;
      const run = teachingTransformer.runTransformer(transformer, ids);
      const data = [...Array.from(run.residuals[0]), ...Array.from(run.residuals[1])];
      return { provider: "wasm", outputs: { residual: { type: "float32", dims: [2, 1, ids.length, WIDTH], data } } };
    }
    const rows = feeds.residual.dims[0];
    const features: number[] = [];
    const rebuilt: number[] = [];
    for (let row = 0; row < rows; row += 1) {
      const code = encode(sae, feeds.residual.data.slice(row * WIDTH, (row + 1) * WIDTH));
      features.push(...Array.from(code));
      const active = Array.from(code, (value, feature) => ({ feature, value })).filter((entry) => entry.value > 0);
      rebuilt.push(...Array.from(reconstruct(sae, active)));
    }
    return {
      provider: "wasm",
      outputs: {
        features: { type: "float32", dims: [rows, FEATURES], data: features },
        reconstruction: { type: "float32", dims: [rows, WIDTH], data: rebuilt },
      },
    };
  },
}));

const { default: Explore } = await import("./Explore");
const { default: definition } = await import("./module");

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

beforeAll(() => {
  vi.stubGlobal("fetch", async (url: string) => {
    const key = url.includes("tiny-transformer") ? "tiny-transformer" : "residual-sae";
    const copy = files[key].slice();
    return { ok: true, status: 200, arrayBuffer: async () => copy.buffer };
  });
});

afterEach(() => {
  if (root) act(() => root!.unmount());
  host?.remove();
  root = null;
  host = null;
});

function Harness({ initial, log }: { initial: ModuleState; log: (state: ModuleState) => void }) {
  const [state, set] = useState<ModuleState>(initial);
  log(state);
  const context: ModuleContext = {
    state,
    setState: (patch) => set((previous) => ({ ...previous, ...patch }) as ModuleState),
    currentStep: 0,
    mode: "standard",
    narrate: () => undefined,
  };
  return createElement(Explore, context);
}

async function mount(initial: Partial<ModuleState> = {}) {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  let latest = { ...definition.initialState, ...initial } as ModuleState;
  await act(async () => {
    root!.render(createElement(Harness, { initial: latest, log: (state) => (latest = state) }));
  });
  return { host, state: () => latest };
}

/** Let timers and promises run, inside act, until `done` holds or the time is up. */
async function until(done: () => boolean, milliseconds = 60_000) {
  const deadline = Date.now() + milliseconds;
  while (!done() && Date.now() < deadline) {
    await act(async () => {
      await new Promise((resolveTimer) => setTimeout(resolveTimer, 100));
    });
  }
  expect(done(), "the condition was not reached in time").toBe(true);
}

const metric = (element: HTMLElement, label: string) => {
  const found = Array.from(element.querySelectorAll(".metric")).find((node) => node.querySelector("span")?.textContent === label);
  return found?.querySelector("strong")?.textContent ?? null;
};

describe("the lab, end to end with the TypeScript model runtime", () => {
  it("runs the SAE on the default text and reports the figures the lesson quotes", async () => {
    const { host: element } = await mount();
    await until(() => metric(element, "‖x − x̂‖² / ‖x‖²") === "1.0%");
    expect(metric(element, "Active features (L0)")).toBe("32 / 1,024");
    expect(metric(element, "‖x − x̂‖² / ‖x‖²")).toBe("1.0%");
    await until(() => metric(element, "Keep 1") === "53.5%");
    expect(metric(element, "Keep 8")).toBe("4.6%");
    expect(metric(element, "Keep all 32")).toBe("1.0%");
    // The largest active features at the first space, with the window-start flag on #499.
    const items = Array.from(element.querySelectorAll(".viz-bar-row")).map((node) => node.textContent ?? "");
    expect(items[0]).toContain("#121");
    expect(items[1]).toContain("#499 · window start");
    // The overlap card finds the pair after the decoder has loaded.
    await until(() => metric(element, "Most similar pair") === "#108 · #499", 90_000);
    expect(metric(element, "Their cosine (signed)")).toBe("-0.992");
    expect(metric(element, "Pairs with |cos| above 0.9")).toBe("7");
  }, 180_000);

  it("shows the window position on every cached example of the top-ranked feature, and its flag", async () => {
    const { host: element } = await mount();
    await until(() => metric(element, "Precision") !== "—");
    const badges = Array.from(element.querySelectorAll(".sf-examples em")).map((node) => node.textContent);
    expect(badges).toEqual(["position 1", "position 1", "position 1", "position 1", "position 1", "position 1", "position 2", "position 1"]);
    expect(element.querySelector(".sf-flags")?.textContent).toContain("window-start artefact");
    expect(element.textContent).toContain("Active on 36,380 of 115,200 scanned characters (31.6%)");
    // Default label "First 4 characters" matches all eight cached examples.
    expect(element.textContent).toContain("On these cached examples the label matches 8 of 8");
    // Position 1 of the probe text is a window-start position, marked with a dotted top edge.
    expect(element.querySelectorAll(".sf-text li.is-early")).toHaveLength(4);
  }, 180_000);

  it("scores #121 as a space label on the probe text and switches example views", async () => {
    const { host: element } = await mount({ feature: 121, hypothesis: "space", threshold: 5 });
    await until(() => metric(element, "Precision") === "100%");
    expect(metric(element, "Recall")).toBe("70%");
    expect(element.textContent).toContain("7 hits, 3 misses, 0 false alarms");
    expect(element.querySelector(".sf-flags")).toBeNull();
    // An unflagged feature has no in-context list to switch to.
    expect(element.textContent).toContain("None of this feature's top examples are in the first 4 positions");
  }, 180_000);

  it("steers #160 at the comma after aye: a hyphen rises from 1.3% to 43.0%, and a continuation of hyphens follows", async () => {
    const { host: element } = await mount();
    await until(() => (metric(element, "Change in log p") ?? "—") !== "—", 90_000);
    expect(metric(element, "p(“-”) before → after")).toBe("1.3% → 43.0%");
    expect(metric(element, "Change in log p")).toBe("+3.53");
    expect(metric(element, "20 random: min · median · max")).toBe("−0.53 · +0.26 · +1.36");
    expect(metric(element, "Feature vs random, in SDs")).toBe("+6.9");
    expect(element.textContent).toContain("6.9 standard deviations above their mean: far outside the random spread");
    expect(element.querySelector(".sf-table")?.textContent).toContain("+15");
    expect(element.textContent).toContain("a vector 40% as long as the residual there");

    const button = Array.from(element.querySelectorAll("button.sf-button"))[0] as HTMLButtonElement;
    expect(button.textContent).toBe("Continue 12 characters");
    await act(async () => {
      button.click();
    });
    await until(() => element.querySelector(".sf-chains") !== null, 150_000);
    const chains = Array.from(element.querySelectorAll(".sf-chains li")).map((node) => node.textContent ?? "");
    expect(chains[0]).toContain("No steering");
    expect(chains[0]).toContain("␣my␣lord,↵Th");
    expect(chains[1]).toContain("#160 added");
    expect(chains[1]).toContain("------------");
    expect(chains[2]).toContain("Random direction 1");
    expect(chains[2]).toContain("␣where␣is␣no");
  }, 300_000);

  it("steers #683 and shows the label character falling", async () => {
    const { host: element } = await mount({ steerFeature: 683, strength: 5 });
    await until(() => (metric(element, "Change in log p") ?? "—") !== "—", 90_000);
    expect(metric(element, "Change in log p")).toBe("−1.38");
    expect(metric(element, "p(“y”) before → after")).toBe("0.5% → 0.1%");
    expect(element.textContent).toContain("The label character became less likely, not more.");
    expect(element.textContent).toContain("Largest rises in log probability: “o” +1.36");
  }, 180_000);

  it("steers #122 and says the change cannot be told from a random direction", async () => {
    const { host: element } = await mount({ steerFeature: 122, strength: 5 });
    await until(() => (metric(element, "Change in log p") ?? "—") !== "—", 90_000);
    expect(metric(element, "Change in log p")).toBe("+0.39");
    expect(element.textContent).toContain("inside the random spread, so it cannot be told apart");
  }, 180_000);

  it("keeps working when the text changes: positions clamp and the steering card follows", async () => {
    const { host: element } = await mount({ text: "hath the" });
    await until(() => (metric(element, "Change in log p") ?? "—") !== "—", 90_000);
    // The steering position 43 is past the end of an 8-character text, so it clamps to the last character.
    expect(element.textContent).toContain("Steering position");
    expect(element.querySelector('input[aria-label="Steering position"]')?.getAttribute("max")).toBe("7");
    expect((element.querySelector('input[aria-label="Steering position"]') as HTMLInputElement).value).toBe("7");
  }, 180_000);
});

describe("module wiring", () => {
  it("renders every guided step's focus target", () => {
    expect(definition.steps).toHaveLength(5);
    expect(ACTIVE).toBe(32);
  });
});
