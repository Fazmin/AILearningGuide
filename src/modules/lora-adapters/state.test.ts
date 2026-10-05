import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import { CALCULATOR_STATE, DEFAULT_CALCULATOR, readCalculator } from "./calculator";
import definition from "./module";

const renderLab = (state: ModuleState) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state,
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const text = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

describe("lora-adapters state", () => {
  it("migrates a version 1 snapshot by dropping the retired heatmap toggle", () => {
    const hydrated = definition.hydrateState(
      JSON.stringify({ rank: 8, alpha: 8, epochs: 12, adapterOn: false, merged: false, view: "base" }),
    );
    expect(hydrated).not.toHaveProperty("view");
    expect(hydrated).toMatchObject({ rank: 8, alpha: 8, epochs: 12, adapterOn: false, probe: " " });
  });

  it("rejects a probe outside the offered contexts", () => {
    expect(definition.hydrateState(JSON.stringify({ probe: "q" })).probe).toBe(" ");
    expect(definition.hydrateState(JSON.stringify({ probe: "s" })).probe).toBe("s");
  });

  it("migrates a version 2 snapshot: the calculator's six settings take their defaults and nothing else changes", () => {
    // Version 2 stored the six keys of the training and serving cards and no calculator setting.
    const version2 = { rank: 6, alpha: 6, epochs: 30, adapterOn: false, merged: false, probe: "e" };
    const hydrated = definition.hydrateState(JSON.stringify(version2));
    expect(hydrated).toEqual({ ...version2, ...CALCULATOR_STATE });
    expect(definition.stateVersion).toBe(3);
    expect(readCalculator(hydrated)).toEqual(DEFAULT_CALCULATOR);
  });

  it("clamps the calculator's settings from a hostile payload and keeps a preset consistent with its shape", () => {
    const hostile = definition.hydrateState(
      JSON.stringify({ calcPreset: "llama-3-70b", calcD: 1e9, calcLayers: -1e9, calcRank: 1e9, calcTargets: ["q", "<b>"], calcBits: 7 }),
    );
    expect(hostile).toMatchObject({ calcPreset: "llama-3-70b", calcD: 8192, calcLayers: 80, calcRank: 256, calcTargets: ["q"], calcBits: 16 });
    const infinite = definition.hydrateState('{"calcPreset":"custom","calcD":1e999,"calcLayers":-1e999,"calcRank":1e999}');
    // A value that parses to Infinity is not a number the controls could produce, so it falls back to the default.
    expect(infinite).toMatchObject({ calcD: 4096, calcLayers: 32, calcRank: 16 });
  });

  it("renders the calculator at its largest setting without a broken value", () => {
    const largest = { ...definition.initialState, calcPreset: "custom", calcD: 16384, calcLayers: 126, calcRank: 256, calcBits: 32 };
    const html = text(renderLab(largest));
    expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
    expect(html).not.toMatch(/\d\.\d{6,}/);
    const empty = text(renderLab({ ...definition.initialState, calcTargets: [] }));
    expect(empty).toContain("nothing targeted");
    expect(empty).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
  });

  it("prints the default calculator readouts the lesson quotes", () => {
    const html = text(renderLab(definition.initialState));
    expect(html).toContain("41,943,040");
    expect(html).toContain("0.522%");
    expect(html).toContain("83.9 MB");
    expect(html).toContain("8.03 billion base parameters");
    expect(html).toContain("26.7%");
    expect(html).toContain("0.131%");
  });
});
