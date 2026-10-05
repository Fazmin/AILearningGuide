import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { moduleIcons } from "@app/components/module-icons";
import { moduleGroups } from "./groups";
import { modules, modulesById, modulesInGroup } from "./registry";
import { checkpointQuestions } from "@app/module-sdk";
import type { ModuleState } from "@app/module-sdk";

const render = (
  module: (typeof modules)[number],
  state: ModuleState,
  currentStep: number,
) =>
  renderToStaticMarkup(
    createElement(module.Explore, {
      state,
      setState: () => undefined,
      currentStep,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );

/** Values that reach the DOM only when a calculation or a guard went wrong. */
const brokenValue = /\bNaN\b|\bundefined\b|\bInfinity\b/;
/**
 * A number with this many decimals was never formatted, which usually means a
 * raw float is being shown where a label or a rounded value belongs.
 */
const unformattedNumber = /\d\.\d{6,}/;
/** JSX renders "&gt;" as literal text, so this is always an authoring mistake. */
const literalEntity = /&amp;(?:gt|lt|amp|quot|#\d+);/;

describe("module registry", () => {
  it("discovers every module with a unique id, slug, and order", () => {
    expect(modules.length).toBeGreaterThanOrEqual(28);
    expect(new Set(modules.map((module) => module.id)).size).toBe(modules.length);
    expect(new Set(modules.map((module) => module.slug)).size).toBe(modules.length);
    expect(new Set(modules.map((module) => module.order)).size).toBe(modules.length);
    expect(modules.map((module) => module.order)).toEqual(
      [...modules.map((module) => module.order)].sort((a, b) => a - b),
    );
  });

  it("places every module in a declared group, and every group has modules", () => {
    const declared = new Set(moduleGroups.map((group) => group.id));
    for (const module of modules) expect(declared.has(module.group)).toBe(true);
    for (const group of moduleGroups) expect(modulesInGroup(group.id).length).toBeGreaterThan(0);
  });

  it("names an icon the sidebar can actually resolve", () => {
    for (const module of modules) {
      expect(moduleIcons[module.icon], `${module.slug} icon "${module.icon}"`).toBeTruthy();
    }
  });

  it("lists seven to fifteen complete, distinct references for every module", () => {
    for (const module of modules) {
      const references = module.references ?? [];
      expect(references.length, `${module.slug} references`).toBeGreaterThanOrEqual(7);
      expect(references.length, `${module.slug} references`).toBeLessThanOrEqual(15);
      expect(new Set(references.map((reference) => reference.url)).size, `${module.slug} repeats a URL`).toBe(references.length);
      for (const reference of references) {
        const label = `${module.slug} reference "${reference.title}"`;
        expect(reference.url, label).toMatch(/^https?:\/\/[^\s]+$/);
        for (const field of [reference.authors, reference.title, reference.source, reference.note]) {
          expect(field.trim(), label).not.toBe("");
        }
        expect(reference.year, label).toBeGreaterThan(1900);
        expect(reference.year, label).toBeLessThanOrEqual(new Date().getFullYear());
      }
    }
  });

  it("points every prerequisite at a module that exists and comes earlier", () => {
    for (const module of modules) {
      for (const prerequisite of module.prerequisites) {
        const target = modulesById.get(prerequisite);
        expect(target, `${module.slug} requires ${prerequisite}`).toBeTruthy();
        expect(target!.order).toBeLessThan(module.order);
      }
    }
  });

  it("gives every module checkpoint questions whose answers index real, distinct options", () => {
    for (const module of modules) {
      const questions = checkpointQuestions(module);
      expect(questions.length, `${module.slug} has no checkpoint`).toBeGreaterThan(0);
      questions.forEach((question, index) => {
        const label = `${module.slug} question ${index + 1}`;
        expect(question.options.length, label).toBeGreaterThan(1);
        expect(new Set(question.options).size, `${label} repeats an option`).toBe(question.options.length);
        expect(question.options[question.answer], label).toBeTruthy();
        expect(question.explanation.length, label).toBeGreaterThan(20);
        // Options are shuffled when shown, so none may refer to another by position.
        for (const option of question.options) {
          expect(option, `${label} option refers to another by position`).not.toMatch(
            /\b(?:all|none) of the above\b|\b(?:both|neither) (?:of )?(?:A|B|C|D)\b|\boption [A-D]\b/i,
          );
        }
        if (question.objective !== undefined) {
          expect(Number.isInteger(question.objective), label).toBe(true);
          expect(question.objective, `${label} objective index`).toBeGreaterThanOrEqual(0);
          expect(question.objective, `${label} objective index`).toBeLessThan(module.objectives.length);
        }
      });
    }
  });

  // Every module declares a list; the single-question form is still read by the SDK for old
  // definitions, but nothing in the app uses it.
  const migrated = modules.filter((module) => Array.isArray(module.checkpoint));

  it("declares a list of questions for every module", () => {
    for (const module of modules) {
      expect(Array.isArray(module.checkpoint), `${module.slug} still declares a single question`).toBe(true);
    }
    expect(migrated).toHaveLength(modules.length);
  });

  it("covers every objective of a migrated module with at least one question", () => {
    for (const module of migrated) {
      const questions = checkpointQuestions(module);
      expect(questions.length, `${module.slug} needs at least 3 questions`).toBeGreaterThanOrEqual(3);
      const covered = new Set(questions.map((question) => question.objective));
      module.objectives.forEach((_, objective) => {
        expect(covered.has(objective), `${module.slug} has no question for objective ${objective + 1}`).toBe(true);
      });
    }
  });

  it("does not let the longest option give the answer away", () => {
    let longest = 0;
    let total = 0;
    for (const module of migrated) {
      const questions = checkpointQuestions(module);
      const hits = questions.filter((question) => {
        const lengths = question.options.map((option) => option.length);
        const best = Math.max(...lengths);
        return lengths[question.answer] === best && lengths.filter((length) => length === best).length === 1;
      }).length;
      expect(hits, `${module.slug}: the correct option is the longest in every question`).toBeLessThan(questions.length);
      longest += hits;
      total += questions.length;
    }
    if (total > 0) expect(longest / total, "correct option is the longest too often").toBeLessThanOrEqual(0.5);
  });
});

/**
 * State arrives from share links, snapshots and hand-edited storage, so it can hold any number. A lab
 * that trains or simulates inside the render must clamp it, or one crafted link freezes the window.
 */
describe("every module under hostile state", () => {
  const INFINITE = "__infinite__";
  const hostile = (module: (typeof modules)[number], value: number | string) => {
    const state: Record<string, unknown> = { ...module.initialState };
    for (const [key, original] of Object.entries(state)) {
      if (typeof original === "number") state[key] = value;
    }
    // JSON has no Infinity, but "1e999" parses to it, which is how a hand-edited link can carry one.
    return JSON.stringify(state).replace(new RegExp(`"${INFINITE}"`, "g"), "1e999");
  };

  it.each(modules.map((module) => [module.slug, module] as const))(
    "%s stays bounded and prints no broken value",
    (slug, module) => {
      for (const value of [1e9, -1e9, INFINITE]) {
        const state = module.hydrateState(hostile(module, value));
        const started = performance.now();
        const html = render(module, state, 0);
        const elapsed = performance.now() - started;
        expect(elapsed, `${slug} took ${Math.round(elapsed)} ms for ${value}`).toBeLessThan(4000);
        const text = html.replace(/<[^>]*>/g, "\u0001");
        const match = text.match(brokenValue);
        expect(match?.[0], `${slug} rendered "${match?.[0]}" for ${value}`).toBeUndefined();
      }
    },
    20000,
  );
});

describe("every module interactive", () => {
  it.each(modules.map((module) => [module.slug, module] as const))(
    "%s renders at every step without leaking a broken value",
    (slug, module) => {
      for (let step = 0; step < module.steps.length; step += 1) {
        const html = render(module, module.initialState, step);
        expect(html.length, `${slug} step ${step} rendered nothing`).toBeGreaterThan(200);
        // Attributes legitimately carry long floats (SVG coordinates, widths), so
        // the formatting rule applies to what a reader actually sees.
        const text = html.replace(/<[^>]*>/g, "\u0001");
        for (const [pattern, source] of [
          [brokenValue, html],
          [literalEntity, html],
          [unformattedNumber, text],
        ] as const) {
          const match = source.match(pattern);
          expect(
            match?.[0],
            `${slug} step ${step} rendered "${match?.[0]}" in: …${source
              .slice(Math.max(0, (match?.index ?? 0) - 70), (match?.index ?? 0) + 40)
              .replace(/[\u0001<][^>\u0001]*[>\u0001]?/g, " ")
              .trim()}…`,
          ).toBeUndefined();
        }
      }
    },
  );

  it.each(modules.map((module) => [module.slug, module] as const))(
    "%s survives zeroed, emptied, and malformed state",
    (slug, module) => {
      const zeroed: ModuleState = Object.fromEntries(
        Object.entries(module.initialState).map(([key, value]) => [
          key,
          typeof value === "number" ? 0 : typeof value === "string" ? "" : value,
        ]),
      );
      for (const state of [
        zeroed,
        module.hydrateState("{}"),
        module.hydrateState("not json at all"),
        module.hydrateState(module.serializeState(module.initialState)),
      ]) {
        const html = render(module, state, 0);
        const match = html.match(brokenValue);
        expect(match?.[0], `${slug} rendered ${match?.[0]}`).toBeUndefined();
      }
    },
  );

  it.each(modules.map((module) => [module.slug, module] as const))(
    "%s round-trips its state and merges defaults back in",
    (slug, module) => {
      const serialized = module.serializeState(module.initialState);
      expect(module.hydrateState(serialized)).toEqual(module.initialState);
      expect(module.hydrateState("{}")).toEqual(module.initialState);
      expect(module.hydrateState("⛔")).toEqual(module.initialState);
      expect(module.stateVersion).toBeGreaterThanOrEqual(1);
    },
  );
});
