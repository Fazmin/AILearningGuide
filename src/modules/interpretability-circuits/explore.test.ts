/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";

/**
 * The lab, rendered with the real weights in jsdom (fetch serves the shipped ONNX file; the ONNX Runtime worker
 * is unavailable here, which the lab already treats as "no parity check"). It reads the numbers the lab prints, so
 * a wiring mistake between the experiments and the screen cannot hide behind the experiments' own tests.
 */
const bytes = readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx"));
const sleep = (ms: number) => new Promise((done) => setTimeout(done, ms));

let root: Root | null = null;
let container: HTMLElement | null = null;

function mount(initial: Partial<ModuleState> = {}) {
  container = document.createElement("div");
  document.body.appendChild(container);
  const states: ModuleState[] = [];
  function Harness() {
    const [state, setState] = useState<ModuleState>({ ...definition.initialState, ...initial } as ModuleState);
    states.push(state);
    return createElement(definition.Explore, {
      state,
      setState: (patch: Partial<ModuleState>) => setState((previous) => ({ ...previous, ...patch }) as ModuleState),
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    });
  }
  root = createRoot(container);
  act(() => root!.render(createElement(Harness)));
  return { states };
}

const text = () => container?.textContent ?? "";
const buttons = () => Array.from(container?.querySelectorAll("button") ?? []);
const press = async (match: (button: HTMLButtonElement) => boolean) => {
  const target = buttons().find(match);
  expect(target, "button to press").toBeTruthy();
  await act(async () => {
    target!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
};
const pressLabel = (label: string) => press((button) => button.textContent?.trim() === label);
const pressAria = (prefix: string) => press((button) => (button.getAttribute("aria-label") ?? "").startsWith(prefix));
const until = async (condition: () => boolean, what: string, timeout = 20000) => {
  const started = Date.now();
  while (!condition()) {
    if (Date.now() - started > timeout) throw new Error(`Timed out waiting for ${what}. Text: ${text().slice(0, 600)}`);
    await act(async () => {
      await sleep(50);
    });
  }
};

beforeAll(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.stubGlobal(
    "fetch",
    async () => ({
      ok: true,
      status: 200,
      arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    }),
  );
});

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe("the circuits lab with the shipped weights", () => {
  it("shows the copy task, the two attention columns and the unseen periods", async () => {
    mount();
    await until(() => text().includes("Copy accuracy") && text().includes("94.5%"), "period-8 copy accuracy");
    expect(buttons().map((button) => button.textContent)).toEqual(expect.arrayContaining(["11 · unseen", "13 · unseen"]));
    expect(text()).not.toContain("6 · unseen");
    expect(text()).toContain("previous letter");
    expect(text()).toContain("after previous occurrence");
    // L2 H3 is shown first: its largest share is 7 back, the key after the previous occurrence.
    expect(text()).toContain("L2 H3: its largest single share of attention, 67%, goes exactly 7 back (the key after the previous occurrence)");
    // The table rows carry the two columns for all eight heads.
    const rows = Array.from(container!.querySelectorAll(".ci-head-table > button"));
    expect(rows).toHaveLength(8);
    expect(rows[3].textContent).toContain("1.00");
    expect(rows[6].textContent).toContain("0.67");
  }, 60000);

  it("follows an unseen period: copy accuracy, a named offset and the corrupted letter's position", async () => {
    mount();
    await until(() => text().includes("94.5%"), "period-8 copy accuracy");
    await pressLabel("13 · unseen");
    await until(() => text().includes("88.0%"), "period-13 copy accuracy");
    expect(text()).toContain("Period 13 is unseen");
    await pressAria("Layer 2 head 3");
    await until(() => text().includes("exactly 12 back (the key after the previous occurrence)"), "a 12-back caption");
    await until(() => text().includes("Running the sweep") === false && text().includes("18.43"), "the live period-13 gap");
  }, 60000);

  it("ablates heads with zero and with mean, and reports the induction heads' attention", async () => {
    mount();
    await until(() => text().includes("alone") && /alone [+−]?\d/.test(text()), "zero-ablation single-head table");
    await pressAria("L1H4, active");
    await until(() => text().includes("9.9%"), "L1 H4 off under zero");
    expect(text()).toContain("0.66 · 0.67 → 0.19 · 0.24");
    await pressAria("L1H4, ablated");
    await pressAria("L2H3, active");
    await until(() => text().includes("6.0%"), "L2 H3 off under zero");
    await pressLabel("Mean");
    await until(() => text().includes("47.9%"), "L2 H3 off under mean");
    expect(text()).toContain("Mean ablation replaces a head's output");
    await pressAria("L2H3, ablated");
    await pressAria("L1H4, active");
    await until(() => text().includes("24.3%"), "L1 H4 off under mean");
    await pressAria("L1H4, ablated");
    await pressLabel("Ablate layer 2");
    await until(() => text().includes("2.8%"), "layer 2 off under mean");
  }, 90000);

  it("patches live, then shows the control corruption's tiny gap and warning", async () => {
    mount();
    await until(() => text().includes("11.70"), "the period-8 gap");
    expect(text()).toContain("recovers 98.8%");
    expect(text()).toContain("Q 99% → Y 48%");
    expect(text()).not.toContain("is not a stable measurement");
    await pressLabel("An unrelated letter");
    await until(() => text().includes("differ by only 0.16"), "the control warning");
    expect(text()).toContain("is not a stable measurement");
    expect(text()).toContain("Q 99% → Q 99%");
  }, 60000);

  it("switches to the cached sweeps, which both have well-conditioned gaps", async () => {
    mount();
    await until(() => text().includes("11.70"), "the live gap");
    await pressLabel("Cached · letters");
    await until(() => text().includes("11.99"), "the cached letters gap");
    expect(text()).toContain("Patching the clean residual after block 1 at position 11");
    expect(text()).not.toContain("is not a stable measurement");
    await pressLabel("Cached · speaker");
    await until(() => text().includes("8.43"), "the cached speaker gap");
    expect(text()).toContain("recovers 96.0%");
  }, 60000);
});
