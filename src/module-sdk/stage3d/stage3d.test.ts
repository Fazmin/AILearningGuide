import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { contourLines } from "./contours";
import { Stage3D } from "./Stage3D";
import { SurfaceStage } from "./SurfaceStage";

describe("contourLines", () => {
  const bowl = (x: number, y: number) => x * x + y * y;

  it("traces a circle whose every point sits at the level's radius", () => {
    const lines = contourLines(bowl, 4, [-3, 3], [-3, 3], 60);
    expect(lines.length).toBeGreaterThan(200);
    expect(lines.length % 4).toBe(0);
    for (let index = 0; index < lines.length; index += 2) {
      expect(Math.hypot(lines[index], lines[index + 1])).toBeCloseTo(2, 2);
    }
  });

  it("returns nothing for a level the ground never reaches", () => {
    expect(contourLines(bowl, 40, [-3, 3], [-3, 3])).toEqual([]);
    expect(contourLines(bowl, -1, [-3, 3], [-3, 3])).toEqual([]);
  });

  it("works on a rectangle and keeps every point inside it", () => {
    const lines = contourLines((x, y) => x - y, 0.5, [-2, 4], [-1, 1], 24);
    expect(lines.length).toBeGreaterThan(0);
    for (let index = 0; index < lines.length; index += 2) {
      expect(lines[index]).toBeGreaterThanOrEqual(-2);
      expect(lines[index]).toBeLessThanOrEqual(4);
      expect(lines[index + 1]).toBeGreaterThanOrEqual(-1);
      expect(lines[index + 1]).toBeLessThanOrEqual(1);
      expect(lines[index] - lines[index + 1]).toBeCloseTo(0.5, 9);
    }
  });

  it("draws both branches through a saddle without joining them across the centre", () => {
    const lines = contourLines((x, y) => x * x - y * y, 0.5, [-2, 2], [-2, 2], 40);
    // Two branches, one on each side: no segment may cross x = 0.
    for (let index = 0; index < lines.length; index += 4) {
      expect(Math.sign(lines[index]) * Math.sign(lines[index + 2])).toBeGreaterThanOrEqual(0);
      expect(Math.abs(lines[index])).toBeGreaterThan(0.6);
    }
  });
});

describe("Stage3D before the picture has loaded", () => {
  const scene = () => new Promise<never>(() => undefined);
  const view = { azimuth: 0, elevation: 30, distance: 5, target: [0, 0, 0] as const };

  it("gives the canvas the picture in words and says how to turn it from the keyboard", () => {
    const html = renderToStaticMarkup(
      createElement(Stage3D, { scene, props: {}, view, label: "A bowl, lowest in the middle.", hint: "click the ground to move the ball" }),
    );
    expect(html).toContain('role="img"');
    expect(html).toContain('aria-label="A bowl, lowest in the middle. A 3D picture: the arrow keys turn it');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain("Drag to turn · click the ground to move the ball · pinch or Ctrl + scroll to zoom");
    expect(html).toContain("Reset view");
    expect(html).toContain("Drawing the 3D view");
  });

  it("offers the flatten button on a surface, not pressed to begin with", () => {
    const html = renderToStaticMarkup(
      createElement(SurfaceStage, {
        label: "A valley.",
        surfaceKey: "valley",
        value: (x: number, y: number) => x * x + 4 * y * y,
        xRange: [-1, 1] as const,
        yRange: [-1, 1] as const,
        zRange: [0, 5] as const,
        levels: [1, 2],
        axes: { x: "w", y: "b", z: "L" },
        xTicks: [-1, 0, 1],
        yTicks: [-1, 0, 1],
        zTicks: [0, 5],
      }),
    );
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain("Press it flat");
    expect(html).toContain('aria-label="A valley. A 3D picture');
  });
});
