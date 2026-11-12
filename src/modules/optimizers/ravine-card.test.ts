import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RavineCard } from "./RavineCard";
import type { RavineRates, RavineTurn } from "./ravine";
import { defaultState, readState } from "./state";

const render = (turn: RavineTurn, rates: RavineRates, step: number) =>
  renderToStaticMarkup(createElement(RavineCard, { turn, rates, step, onChange: () => undefined, narrate: () => undefined }));
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const rows = (html: string) =>
  [...html.matchAll(/<tr><th scope="row">.*?<\/span>([^<]+)<\/th><td>([^<]*)<\/td><td>([^<]*)<\/td><td>([^<]*)<\/td><\/tr>/g)].map((row) =>
    row.slice(1, 5),
  );

describe("A ravine in two weights", () => {
  it("walks three optimizers at the race's default rates and says when each settles", () => {
    const html = render("axis", "default", 0);
    expect(rows(html)).toEqual([
      ["SGD", "1", "0.1498", "49"],
      ["Momentum", "0.1", "0.1498", "59"],
      ["Adam", "0.03", "0.1498", "35"],
    ]);
    expect(html).toMatch(/aria-label="A loss over two weights drawn as ground: a ravine 100 times steeper across than along/);
    expect(visible(html)).toContain("walls 100× steeper than the floor");
    expect(visible(html)).toContain("step 0 of 120");
  });

  it("shows SGD flying out at double the rate while the other two settle", () => {
    const html = render("axis", "double", 120);
    expect(rows(html)).toEqual([
      ["SGD", "2", "flew out", "never: it flies out"],
      ["Momentum", "0.2", "under 0.0001", "46"],
      ["Adam", "0.06", "under 0.0001", "35"],
    ]);
    expect(visible(html)).toContain("rate of 2 is past 1.11");
  });

  it("leaves SGD and momentum unchanged on the turned ravine and takes Adam's lead away", () => {
    const html = render("diagonal", "default", 120);
    const table = rows(html);
    expect(table.map((row) => row[3])).toEqual(["49", "59", "not in 120"]);
    expect(visible(html)).toContain("Adam has not settled after 120 steps");
    expect(visible(render("axis", "half", 0))).toContain("At half the rate SGD no longer bounces");
  });

  it("prints nothing broken at any setting or step", () => {
    for (const turn of ["axis", "diagonal"] as const) {
      for (const rates of ["half", "default", "double"] as const) {
        for (const step of [0, 1, 37, 120]) {
          const html = render(turn, rates, step);
          expect(html, `${turn} ${rates} ${step}`).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
          expect(visible(html)).not.toMatch(/\d\.\d{6,}/);
        }
      }
    }
  });

  it("stores its three settings and reads them back from anything", () => {
    expect(defaultState).toMatchObject({ ravineTurn: "axis", ravineRates: "default", ravineStep: 0 });
    expect(readState({ ravineTurn: "diagonal", ravineRates: "double", ravineStep: 64.4 })).toMatchObject({
      ravineTurn: "diagonal",
      ravineRates: "double",
      ravineStep: 64,
    });
    expect(readState({ ravineTurn: "sideways", ravineRates: 3, ravineStep: 1e9 })).toMatchObject({
      ravineTurn: "axis",
      ravineRates: "default",
      ravineStep: 120,
    });
    expect(readState({ ravineStep: -5 }).ravineStep).toBe(0);
  });
});
