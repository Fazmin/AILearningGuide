import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider, type ModuleState } from "@app/module-sdk";
import cardInfo from "./card-info";
import { WORDS, wordById } from "./math";
import definition from "./module";
import { closestPair3, directionsLeft, squeeze3, vec3Of } from "./space";

describe("three numbers per word", () => {
  it("keeps each word's size and alive numbers and adds a third", () => {
    for (const word of WORDS) {
      const [x, y, z] = vec3Of(word);
      expect([x, y]).toEqual([word.x, word.y]);
      expect(Number.isFinite(z)).toBe(true);
      expect(Math.abs(z)).toBeLessThanOrEqual(2);
    }
    // cat and bicycle share a size and differ in the other two numbers.
    expect(vec3Of(wordById("cat"))[0]).toBe(vec3Of(wordById("bicycle"))[0]);
    expect(vec3Of(wordById("cat"))[2]).not.toBe(vec3Of(wordById("bicycle"))[2]);
  });

  it("counts the directions a squeeze leaves", () => {
    expect(directionsLeft(1, 1)).toBe(3);
    expect(directionsLeft(0.05, 0.4)).toBe(3);
    expect(directionsLeft(0, 1)).toBe(2);
    expect(directionsLeft(1, 0)).toBe(2);
    expect(directionsLeft(0, 0)).toBe(1);
    expect(squeeze3([2, -1, 0.5], 0, 0.5)).toEqual([2, -0, 0.25]);
  });

  it("keeps all eight words apart on a sheet and merges cat with bicycle on the line", () => {
    expect(closestPair3(1, 1).distinct).toBe(8);
    const sheet = closestPair3(0, 1);
    expect(sheet.distinct).toBe(8);
    expect(sheet.merged).toBe(false);
    expect(sheet.pair.map((word) => word.id)).toEqual(["horse", "car"]);
    expect(sheet.gap).toBeCloseTo(0.283, 3);
    // Pressing fast flat instead gives back the flat map, where every word has its own spot.
    expect(closestPair3(1, 0).distinct).toBe(8);
    const line = closestPair3(0, 0);
    expect(line.distinct).toBe(7);
    expect(line.merged).toBe(true);
    expect(line.pair.map((word) => word.id)).toEqual(["cat", "bicycle"]);
  });
});

describe("the rank card in 3D", () => {
  const render = (patch: Partial<ModuleState>) =>
    renderToStaticMarkup(
      createElement(
        CardInfoProvider,
        { value: cardInfo },
        createElement(definition.Explore, {
          state: { ...definition.initialState, forgetView: "space", ...patch } as ModuleState,
          setState: () => undefined,
          currentStep: 5,
          mode: "plain" as const,
          narrate: () => undefined,
        }),
      ),
    );
  const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

  it("replaces the flat map with a 3D picture described in words, and keeps the flat map as the default", () => {
    const html = render({});
    expect(html).toContain("stage3d__frame");
    expect(html).toMatch(/aria-label="Eight words as points in space[^"]*3 of 3 directions are left/);
    expect(html).not.toContain("the squeezing matrix</figcaption>");
    const flat = render({ forgetView: "flat" });
    expect(flat).not.toContain("stage3d__frame");
    expect(flat).toContain("Each word is");
  });

  it("counts directions, spots and volume as the two sliders press the cloud flat", () => {
    const full = visible(render({}));
    expect(full).toContain("3 of 3");
    expect(full).toContain("Nothing has moved yet");
    const sheet = visible(render({ keep: 0 }));
    expect(sheet).toContain("2 of 3");
    expect(sheet).toContain("8 of 8");
    expect(sheet).toContain("pressed onto a flat sheet: every alive number is 0");
    const line = visible(render({ keep: 0, keepFast: 0 }));
    expect(line).toContain("1 of 3");
    expect(line).toContain("7 of 8");
    expect(line).toContain("cat = bicycle");
    expect(line).toContain("(−0.8, 0.0, 0.0)");
  });

  it("can stretch back a thin cloud and cannot stretch back a flat one", () => {
    expect(visible(render({ keep: 0.25, keepFast: 0.5, undo: "stretch" }))).toContain("puts every word back");
    const stuck = visible(render({ keep: 0.25, keepFast: 0, undo: "stretch" }));
    expect(stuck).toContain("Stretching fast back would need 1 ÷ 0");
    expect(stuck).toContain("2 of 3");
  });

  it("prints nothing broken at any slider setting", () => {
    for (const keep of [0, 0.05, 0.5, 1]) {
      for (const keepFast of [0, 0.35, 1]) {
        for (const undo of ["leave", "stretch"]) {
          const html = render({ keep, keepFast, undo });
          expect(html, `${keep} ${keepFast} ${undo}`).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
          expect(visible(html)).not.toMatch(/\d\.\d{6,}/);
        }
      }
    }
  });
});
