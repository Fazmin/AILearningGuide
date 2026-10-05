import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";

const text = (patch: Record<string, string | number>) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep: 2,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  )
    .replace(/<[^>]*>/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#x27;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ");

describe("the probe readouts print the numbers the lesson quotes", () => {
  it("scores the tree's root cut by Gini at every depth", () => {
    for (const depth of [1, 3, 5]) {
      const shown = text({ algorithm: "tree", depth });
      expect(shown).toContain("The root cut x < 0.15 sends 2 points left and 16 right");
      expect(shown).toContain("Gini falls from 0.444 to a size-weighted 0.333, a gain of 0.111");
    }
  });

  it("counts the SVM's support vectors beside the street width", () => {
    const shown = text({ algorithm: "svm" });
    expect(shown).toContain("12 of 18 here");
    expect(shown).toContain("2/‖w‖ = 0.541");
  });

  it("prints each naive-Bayes class score at the default probe", () => {
    const shown = text({ algorithm: "naivebayes" });
    expect(shown).toContain("0.667 × 1.446 × 1.632 = 1.574");
    expect(shown).toContain("0.333 × 1.217 × 1.298 = 0.527");
  });

  it("still renders every other method", () => {
    for (const algorithm of ["linear", "logistic", "knn", "forest", "kmeans"]) {
      expect(text({ algorithm })).not.toMatch(/\bNaN\b|\bundefined\b/);
    }
  });
});
