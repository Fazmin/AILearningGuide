import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition, { hydrateMultimodalState } from "./module";
import { DEFAULT_PIXELS } from "./patches";
import { initialState } from "./state";

describe("state versions", () => {
  it("reads a version 2 payload unchanged and fills the Contrastive pairs controls", () => {
    const version2 = {
      pixels: DEFAULT_PIXELS,
      patchSize: 2,
      selected: 7,
      pixel: 12,
      side: 224,
      vitPatch: 16,
      cls: "off",
      connector: "clip",
    };
    expect(hydrateMultimodalState(JSON.stringify(version2))).toEqual({ ...version2, temperature: 0.2, pairing: "true" });
  });

  it("keeps a version 3 payload", () => {
    const state = hydrateMultimodalState(JSON.stringify({ temperature: 0.05, pairing: "shuffled" }));
    expect([state.temperature, state.pairing]).toEqual([0.05, "shuffled"]);
  });

  it("clamps the new controls, drops unknown keys, and cuts free text before reading it", () => {
    expect(hydrateMultimodalState(JSON.stringify({ temperature: 1e9 })).temperature).toBe(1);
    expect(hydrateMultimodalState(JSON.stringify({ temperature: -1e9 })).temperature).toBe(0.01);
    expect(hydrateMultimodalState('{"temperature": 1e999}').temperature).toBe(0.2);
    expect(hydrateMultimodalState(JSON.stringify({ pairing: "nonsense" })).pairing).toBe("true");
    const state = hydrateMultimodalState(JSON.stringify({ pixels: "1".repeat(200000), extra: 1 }));
    expect(state.pixels).toBe("1".repeat(64));
    expect(state).not.toHaveProperty("extra");
    expect(hydrateMultimodalState("[1]")).toEqual(initialState);
  });
});

describe("the page shows what the lesson quotes", () => {
  const rendered = (state: Record<string, unknown>) =>
    renderToStaticMarkup(
      createElement(definition.Explore, {
        state: hydrateMultimodalState(JSON.stringify(state)),
        setState: () => undefined,
        currentStep: 5,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    )
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ");

  it("shows the default loss, the chance loss and a match marked in every row", () => {
    const page = rendered({});
    expect(page).toContain("Contrastive loss 0.235");
    expect(page).toContain("Chance loss, ln 3 1.099");
    expect(page.match(/labelled match(?!&#x27;|')/g)).toHaveLength(3);
    expect(page).toContain("3 of 3");
  });

  it("shows the temperature and pairing comparisons the lesson names", () => {
    expect(rendered({ temperature: 0.05 })).toContain("Contrastive loss 0.006");
    expect(rendered({ temperature: 1 })).toContain("Contrastive loss 0.813");
    const shuffled = rendered({ pairing: "shuffled" });
    expect(shuffled).toContain("Contrastive loss 2.582");
    expect(shuffled).toContain("0 of 3");
  });
});
