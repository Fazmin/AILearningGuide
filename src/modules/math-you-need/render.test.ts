import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";

const render = (patch: Record<string, string | number>, currentStep = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, topic: "probability", ...patch },
      setState: () => undefined,
      currentStep,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

describe("the probability refresher renders every idea at both ends of its controls", () => {
  const cases: [string, Record<string, string | number>, string][] = [
    ["counts", { probIdea: "counts" }, "P(A,A,B)"],
    ["bayes at the default prior", { probIdea: "bayes" }, "Posterior P(y=1|x=1)"],
    ["bayes at the lowest prior", { probIdea: "bayes", prior: 0.02 }, "Posterior P(y=1|x=1)"],
    ["bayes at the highest prior", { probIdea: "bayes", prior: 0.98 }, "Posterior P(y=1|x=1)"],
    ["gaussian at the default", { probIdea: "gaussian" }, "Density at x"],
    ["gaussian with the smallest variance", { probIdea: "gaussian", gaussVar: 0.1, gaussMu: 2, gaussX: -4 }, "Density at x"],
    ["gaussian with the largest variance", { probIdea: "gaussian", gaussVar: 2, gaussMu: -2, gaussX: 4 }, "Density at x"],
    ["logit at the default", { probIdea: "logit" }, "Sigmoid σ(z)"],
    ["logit at the widest gap", { probIdea: "logit", pA: 0.1, pB: 6 }, "Sigmoid σ(z)"],
  ];

  it.each(cases)("%s", (_name, patch, expected) => {
    for (const step of [0, 4, 5, 6, 8]) {
      const html = render(patch, step);
      const text = visible(html);
      expect(text).toContain(expected);
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
      expect(text).not.toMatch(/\d\.\d{6,}/);
    }
  });

  it("prints the worked numbers the lesson quotes", () => {
    expect(visible(render({ probIdea: "bayes" }))).toContain("0.750");
    expect(visible(render({ probIdea: "bayes", prior: 0.1 }))).toContain("0.250");
    expect(visible(render({ probIdea: "gaussian" }))).toContain("68.3%");
    expect(visible(render({ probIdea: "logit" }))).toContain("0.7858");
  });
});
