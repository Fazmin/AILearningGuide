import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { explanationModes } from "@app/module-sdk";
import { normalizeHeading } from "@app/lib/learn-sections";
import { sceneState } from "@app/lib/lab-scenes";
import { modules } from "./registry";

const withScenes = modules.filter((module) => module.scenes.length > 0);
// Joined by hand: Vite turns `new URL("./x.mdx", import.meta.url)` into an import of the lesson itself.
const modulesRoot = path.dirname(fileURLToPath(import.meta.url));

/** The `##` and `###` headings of one reading mode's lesson, normalized. */
const headingsOf = (slug: string, mode: string) => {
  const source = readFileSync(path.join(modulesRoot, slug, "content", `${mode}.mdx`), "utf8");
  return [...source.matchAll(/^#{2,3} (.+)$/gm)].map((match) => normalizeHeading(match[1]));
};

const escapeAttribute = (value: string) =>
  value.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

describe.each(withScenes.map((module) => [module.slug, module] as const))("%s lab scenes", (slug, module) => {
  const lessons = explanationModes.map((mode) => ({ mode, headings: headingsOf(slug, mode) }));

  it("have unique ids, a title and a one-sentence look", () => {
    const ids = module.scenes.map((scene) => scene.id);
    expect(new Set(ids).size, "duplicate scene id").toBe(ids.length);
    for (const scene of module.scenes) {
      expect(scene.title.trim(), scene.id).not.toBe("");
      expect(scene.look, scene.id).toMatch(/\S.*\.$/);
      expect(scene.sections.length, `${scene.id} names no section`).toBeGreaterThan(0);
    }
  });

  it("name headings the lessons have, one scene per heading in each mode", () => {
    for (const scene of module.scenes) {
      for (const section of scene.sections) {
        expect(
          lessons.some(({ headings }) => headings.includes(normalizeHeading(section))),
          `${scene.id}: no reading mode has the heading "${section}"`,
        ).toBe(true);
      }
    }
    for (const { mode, headings } of lessons) {
      const claimed = new Map<string, string>();
      for (const scene of module.scenes) {
        const heading = scene.sections.map(normalizeHeading).find((section) => headings.includes(section));
        if (!heading) continue;
        expect(claimed.get(heading), `${mode}: "${heading}" is claimed by ${claimed.get(heading)} and ${scene.id}`).toBeUndefined();
        claimed.set(heading, scene.id);
      }
    }
  });

  it("set real steps and settings the lab keeps as they are", () => {
    for (const scene of module.scenes) {
      if (scene.step !== undefined) {
        expect(Number.isInteger(scene.step) && scene.step >= 0 && scene.step < module.steps.length, scene.id).toBe(true);
      }
      for (const key of Object.keys(scene.state ?? {})) {
        expect(Object.keys(module.initialState), `${scene.id}: "${key}" is not a lab setting`).toContain(key);
      }
      // A value the lab would clamp or replace would show something other than what the scene promises.
      const state = sceneState(module.initialState, scene);
      expect(module.hydrateState(module.serializeState(state)), scene.id).toEqual(state);
    }
  });

  it("point at a card the lab draws in that setup", () => {
    for (const scene of module.scenes) {
      if (!scene.focus) continue;
      const html = renderToStaticMarkup(
        createElement(module.Explore, {
          state: sceneState(module.initialState, scene),
          setState: () => undefined,
          currentStep: scene.step ?? 0,
          mode: "standard" as const,
          narrate: () => undefined,
        }),
      );
      expect(html, `${scene.id}: no "${scene.focus}" card`).toContain(`data-lab-surface="${escapeAttribute(scene.focus)}"`);
    }
  });
});
