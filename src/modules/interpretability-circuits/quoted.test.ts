/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import definition from "./module";

/**
 * experiments.test.ts measures every figure the lesson quotes from the shipped model and pins it. This file
 * checks the other direction: that the prose, the card explanations and the module copy still say those
 * figures, and no longer say the ones from the first release of the model (a fixed 7-back look-up learned
 * from period-8 strings), so a retrain cannot be half-applied.
 */
const read = (relative: string) => readFileSync(resolve(process.cwd(), "src/modules/interpretability-circuits", relative), "utf8");
const standard = read("content/standard.mdx");
const plain = read("content/plain.mdx");
const cards = read("card-info.ts");
const moduleSource = read("module.ts");
const flat = (text: string) => text.replace(/\s+/g, " ");
const everything = flat([standard, plain, cards, moduleSource].join("\n"));

describe("figures the lessons and cards quote", () => {
  it("states the measured copying, attention and ablation figures in the Standard lesson", () => {
    for (const figure of [
      "94.5%", "88.0%", "97.2%", "0.66 and 0.67", "0.55 to 0.84", "0.98 or more", "0.89 and 0.93",
      "5.1%", "6.0%", "9.9%", "50.2%", "56.6%", "89.2%", "95.8%", "99.6%", "1.9%", "0.4%",
      "0.19 and 0.24", "0.01 and 0.03", "0.30 and 0.21",
      "47.9%", "75.9%", "27.4%", "24.3%", "48.8%", "1.3%", "2.4%",
      "11.70", "10.69", "−1.01", "98.8%", "−0.5%", "48%", "99.5%", "6.05", "22.35", "20.7", "15.9", "0.16", "0.35", "43%",
    ]) {
      expect(flat(standard), `Standard lesson should say ${figure}`).toContain(figure);
    }
  });

  it("states the same core figures in the Plain lesson", () => {
    for (const figure of [
      "94.5%", "88.0%", "97.2%", "0.66 and 0.67", "0.98 or more", "5.1%", "6.0%", "9.9%", "89.2%",
      "0.19 and 0.24", "47.9%", "98.8%", "11.70", "0.16", "43%",
    ]) {
      expect(flat(plain), `Plain lesson should say ${figure}`).toContain(figure);
    }
  });

  it("states them in the card explanations too", () => {
    for (const figure of [
      "94.5%", "88.0% to 97.2%", "88.5%", "0.55 to 0.84", "0.98 or more", "0.67", "1.00", "0.99",
      "9.9%", "24.3%", "0.66 · 0.67", "0.19 · 0.24", "0.01 · 0.03", "6.0%", "47.9%", "5.1%", "50.2%", "56.6%", "89.2%", "95.8%", "99.6%",
      "75.9%", "27.4%", "48.8%", "1.3%", "2.4%", "95.3%", "0.30 · 0.21",
      "98.8%", "11.70", "0.16", "48%", "99.5%", "6.05", "22.35", "20.7", "15.9", "0.35", "43%", "11.99", "18.43", "0.006", "96%",
    ]) {
      expect(flat(cards), `card-info should say ${figure}`).toContain(figure);
    }
  });

  it("no longer carries the first release's claims", () => {
    for (const stale of [
      "96.8%", "67.0%", "51.0%", "0.00003", "204×", "51×", "trained period", "· trained", "30% curriculum",
      "all eight put", "all eight do the same", "No head is necessary", "No single head is necessary", "fixed positional look-back",
      "learned a fixed look-back", "unstable except", "all training repeats had period 8", "2.4% or less",
      "only practised on", "85%", "0.57 to 0.03",
    ]) {
      expect(everything, `should not say "${stale}"`).not.toContain(stale);
    }
  });

  it("names the unseen periods and the sources it relies on", () => {
    expect(cards).toContain("11, 13, 17 and 19");
    for (const source of ["Elhage", "Olsson", "Wang et al.", "McGrath"]) expect(flat(standard)).toContain(source);
    for (const source of ["Elhage", "Olsson", "Wang et al."]) expect(flat(plain)).toContain(source);
    expect(definition.stateVersion).toBe(3);
  });

  it("uses every glossary term in the Plain lesson as well as the Standard one", () => {
    const lowered = (text: string) => text.toLowerCase().replace(/\s+/g, " ");
    for (const { term } of definition.glossary) {
      expect(lowered(plain.replace(/^---[\s\S]*?---/, "")), `Plain lesson never uses "${term}"`).toContain(term.toLowerCase());
      expect(lowered(standard.replace(/^---[\s\S]*?---/, "")), `Standard lesson never uses "${term}"`).toContain(term.toLowerCase());
    }
  });
});
