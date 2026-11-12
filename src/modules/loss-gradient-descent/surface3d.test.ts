import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LEAST_SQUARES, mse, runDescent } from "./descent";
import { SURFACE_TOP } from "./LossSurface3D";
import definition from "./module";

const render = (patch: Record<string, unknown> = {}) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...definition.initialState, ...patch } as typeof definition.initialState,
      setState: () => undefined,
      currentStep: 0,
      mode: "plain" as const,
      narrate: () => undefined,
    }),
  );
const canvasLabel = (html: string) => html.match(/<canvas[^>]*aria-label="([^"]*)"/)?.[1] ?? "";

describe("the Loss surface card in 3D", () => {
  it("shows the flat map unless the reader picks the 3D surface", () => {
    const flat = render();
    expect(flat).toContain('class="lgd-surface"');
    expect(flat).not.toContain("stage3d__frame");
    const lifted = render({ picture: "surface" });
    expect(lifted).toContain("stage3d__frame");
    expect(lifted).not.toContain('class="lgd-surface"');
    expect(render({ picture: "hologram" })).toContain('class="lgd-surface"');
  });

  it("puts the run's current point and loss into the picture's words", () => {
    const run = runDescent({ w0: -1, b0: 1.2, learningRate: 0.3, optimizer: "gd", beta: 0.9 });
    const label = canvasLabel(render({ picture: "surface", learningRate: 0.3, step: 9 }));
    expect(label).toContain(`lowest at w ${LEAST_SQUARES.w.toFixed(2)}, b ${LEAST_SQUARES.b.toFixed(2)}`);
    expect(label).toContain(`At step 9 the ball is at w ${run.points[9].w.toFixed(2)}, b ${run.points[9].b.toFixed(2)}`);
    expect(label).toContain(`cut off above ${SURFACE_TOP}`);
  });

  it("says so when a diverging run has left the drawn ground", () => {
    const html = render({ picture: "surface", learningRate: 0.5, step: 40 });
    expect(canvasLabel(html)).toContain("the run is off the drawn ground");
    expect(html).toContain("Off the drawn ground at step 40");
    expect(html).not.toMatch(/\bNaN\b|\bundefined\b/);
  });

  it("cuts the ground off above every start the sliders allow, so the ball is always on it at step 0", () => {
    for (const w of [-1.5, 3.5]) for (const b of [-1.9, 1.9]) expect(mse(w, b)).toBeGreaterThan(0);
    expect(mse(-1, 1.2)).toBeLessThan(SURFACE_TOP);
  });
});
