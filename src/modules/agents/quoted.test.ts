import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition, { hydrateAgentsState } from "./module";

/** The text a learner reads on the page, with the markup removed. */
const rendered = (state: Record<string, unknown>) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: hydrateAgentsState(JSON.stringify(state)),
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  )
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ");

/** Pins the figures that standard.mdx, plain.mdx, card-info.ts and the checkpoint quote. */
describe("agents quoted values appear on the page", () => {
  it("shows the Reliability figures the lesson quotes", () => {
    const base = { guard: "off", stepSuccess: 95, runSteps: 20 };
    const plain = rendered(base);
    expect(plain).toContain("35.8%");
    expect(plain).toContain("13 steps");
    expect(rendered({ ...base, stepSuccess: 99 })).toContain("81.8%");
    const gated = rendered({ ...base, gateAfter: 10, catchRate: 100 });
    expect(gated).toContain("50.2%");
    expect(gated).toContain("24.0");
    expect(rendered({ ...base, gateAfter: 10, catchRate: 0 })).not.toContain("50.2%");
    expect(rendered({ ...base, gateAfter: 1 })).toContain("37.6%");
    expect(rendered({ ...base, stepSuccess: 99 })).toContain("68 steps");
  });

  it("shows the Fix an argument run the lesson quotes", () => {
    const page = rendered({ guard: "on", task: "repair", stepCap: 4 });
    expect(page).toContain("3 model calls");
    expect(page).toContain("2 (1 failed)");
    expect(page).toContain("391");
  });
});
