import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider } from "@app/module-sdk";
import { modules } from "./registry";

const render = (module: (typeof modules)[number], currentStep: number) =>
  renderToStaticMarkup(
    createElement(
      CardInfoProvider,
      { value: module.cardInfo },
      createElement(module.Explore, {
        state: module.initialState,
        setState: () => undefined,
        currentStep,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    ),
  );

/** Surface labels a module actually renders, gathered across every step. */
function renderedLabels(module: (typeof modules)[number]) {
  const labels = new Set<string>();
  let surfaces = 0;
  let triggers = 0;

  for (let step = 0; step < module.steps.length; step += 1) {
    const html = render(module, step);
    const rendered = html.match(/class="lab-surface[^"]*" aria-label="([^"]*)"/g) ?? [];
    surfaces = Math.max(surfaces, rendered.length);
    triggers = Math.max(triggers, (html.match(/class="card-info-trigger"/g) ?? []).length);
    for (const surface of rendered) {
      labels.add(surface.replace(/^.*aria-label="([^"]*)"$/, "$1"));
    }
  }

  return { labels, surfaces, triggers };
}

const cases = modules.map((module) => [module.slug, module] as const);

describe("lab card explanations", () => {
  it.each(cases)("%s puts an info trigger on every lab card it renders", (slug, module) => {
    const { surfaces, triggers } = renderedLabels(module);
    expect(surfaces, `${slug} rendered no lab cards`).toBeGreaterThan(0);
    expect(triggers, `${slug} has ${surfaces} cards but ${triggers} info triggers`).toBe(surfaces);
  });

  it.each(cases)("%s registers no explanation for a card it never renders", (slug, module) => {
    const { labels } = renderedLabels(module);
    for (const key of Object.keys(module.cardInfo)) {
      expect(labels.has(key), `${slug}/card-info.ts: "${key}" matches no rendered card`).toBe(true);
    }
  });

  it.each(cases)("%s explains each card's mechanism, signal, and limits", (slug, module) => {
    for (const [label, info] of Object.entries(module.cardInfo)) {
      const where = `${slug} · ${label}`;
      expect(info.title.length, `${where}: title`).toBeGreaterThan(3);
      expect(info.summary.length, `${where}: summary`).toBeGreaterThan(80);
      expect(info.whatYouSee.length, `${where}: whatYouSee`).toBeGreaterThanOrEqual(2);
      expect(info.howItWorks.length, `${where}: howItWorks`).toBeGreaterThanOrEqual(2);
      expect(info.limits.length, `${where}: limits`).toBeGreaterThanOrEqual(2);

      // The style guide keeps the interactive's limits and the mechanism's limits
      // as separate claims, so neither can stand in for the other.
      expect(
        info.limits.some((limit) => limit.startsWith("In this lab: ")),
        `${where}: limits need an "In this lab: " bullet`,
      ).toBe(true);
      expect(
        info.limits.some((limit) => limit.startsWith("In general: ")),
        `${where}: limits need an "In general: " bullet`,
      ).toBe(true);

      for (const bullet of [
        ...info.whatYouSee,
        ...info.howItWorks,
        ...(info.controls ?? []),
        ...(info.notice ?? []),
        ...info.limits,
      ]) {
        expect(bullet.trim(), `${where}: empty bullet`).not.toBe("");
        expect(bullet, `${where}: unresolved placeholder in "${bullet}"`).not.toMatch(
          /\bTODO\b|\bTBD\b|\bLorem\b|\bundefined\b|\bNaN\b/,
        );
      }
    }
  });
});
