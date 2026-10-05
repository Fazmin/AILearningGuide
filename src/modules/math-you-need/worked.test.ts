import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import { crossEntropy, det2, entropy, klDivergence, secantSlope } from "./curves";

const round = (value: number, digits = 3) => Number(value.toFixed(digits));
const render = (patch: Record<string, string | number>) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch },
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

describe("the derivative worked by hand in the lesson", () => {
  it("reads f = 1.44, tangent 2.4, and secant 2.8 at x = 1.2 with ε = 0.4, and 2.45 at ε = 0.05", () => {
    expect(round(1.2 ** 2, 2)).toBe(1.44);
    expect(round(1.6 ** 2, 2)).toBe(2.56);
    expect(round(2 * 1.2, 1)).toBe(2.4);
    expect(secantSlope(1.2, 0.4)).toBeCloseTo(2.8, 12);
    expect(secantSlope(1.2, 0.05)).toBeCloseTo(2.45, 12);
    expect(secantSlope(1.2, 0.4) - 2 * 1.2).toBeCloseTo(0.4, 12);
  });

  it("is what the derivatives widget prints at its defaults", () => {
    expect(definition.initialState).toMatchObject({ x: 1.2, nudge: 0.4 });
    const text = visible(render({ topic: "derivatives" }));
    expect(text).toContain("1.440");
    expect(text).toContain("2.400");
    expect(text).toContain("2.800");
  });
});

describe("the matrix worked by hand in the lesson", () => {
  it("gives Wx + b = (1.28, 0.30) and det W = 1 for the default W, x, and b", () => {
    expect(definition.initialState).toMatchObject({ m00: 1, m01: 0.4, m10: 0, m11: 1, vx: 1, vy: 0.2, b0: 0.2, b1: 0.1 });
    const rowOne = 1 * 1 + 0.4 * 0.2;
    const rowTwo = 0 * 1 + 1 * 0.2;
    expect(round(rowOne, 2)).toBe(1.08);
    expect(round(rowOne + 0.2, 2)).toBe(1.28);
    expect(round(rowTwo + 0.1, 2)).toBe(0.3);
    expect(det2(1, 0.4, 0, 1)).toBe(1);
  });

  it("is what the matrices widget prints at its defaults", () => {
    const text = visible(render({ topic: "matrices" }));
    expect(text).toContain("[1.28, 0.30]");
    expect(text).toContain("1.080");
    expect(text).toContain("det W · area scale");
  });
});

describe("the logarithm worked by hand in the lesson", () => {
  it("gives surprises 0.511 and 0.916 under q = 0.60, H(p,q) 0.815, H(p) 0.562, and KL 0.253 at p = 0.25", () => {
    expect(definition.initialState).toMatchObject({ value: 0.25, modelQ: 0.6 });
    expect(round(-Math.log(0.6))).toBe(0.511);
    expect(round(-Math.log(0.4))).toBe(0.916);
    expect(round(-Math.log(0.25))).toBe(1.386);
    expect(round(-Math.log(0.75))).toBe(0.288);
    expect(round(crossEntropy(0.25, 0.6))).toBe(0.815);
    expect(round(entropy(0.25))).toBe(0.562);
    expect(round(klDivergence(0.25, 0.6))).toBe(0.253);
    expect(klDivergence(0.25, 0.25)).toBeCloseTo(0, 12);
  });

  it("is what the logarithms widget prints at its defaults", () => {
    const text = visible(render({ topic: "logarithms" }));
    for (const printed of ["1.386", "0.562", "0.815", "0.253"]) expect(text).toContain(printed);
  });
});
