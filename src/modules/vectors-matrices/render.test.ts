import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CardInfoProvider, type ModuleState } from "@app/module-sdk";
import cardInfo from "./card-info";
import { ROUNDS, WORD_IDS } from "./math";
import definition from "./module";
import scenes from "./scenes";

const CARDS = [
  "A word as two numbers",
  "Add and stretch",
  "How alike are two arrows",
  "A matrix moves every arrow",
  "One after another",
  "When a matrix forgets",
  "A table from a few pieces",
  "Challenge: aim the map",
];

const renderStep = (patch: Partial<ModuleState>, currentStep: number) =>
  renderToStaticMarkup(
    createElement(
      CardInfoProvider,
      { value: cardInfo },
      createElement(definition.Explore, {
        state: { ...definition.initialState, ...patch } as ModuleState,
        setState: () => undefined,
        currentStep,
        mode: "plain" as const,
        narrate: () => undefined,
      }),
    ),
  );

/** One step's page, or (with no step) every page one after another: the lab shows one card per step. */
const render = (patch: Partial<ModuleState> = {}, step?: number) =>
  step === undefined ? CARDS.map((_, index) => renderStep(patch, index)).join("") : renderStep(patch, step);

/** Text a reader sees, with tags replaced by a separator so numbers in attributes do not count. */
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const broken = /\bNaN\b|\bundefined\b|\bInfinity\b/;
const unformatted = /\d\.\d{6,}/;

describe("the lab renders", () => {
  it("has one step, one instruction and one explanation per card", () => {
    expect(definition.steps).toHaveLength(CARDS.length);
    expect(definition.stepInstructions).toHaveLength(CARDS.length);
    expect(Object.keys(cardInfo)).toEqual(CARDS);
  });

  it.each(CARDS.map((label, step) => [step, label] as const))(
    "at step %i it shows the card “%s” and no other",
    (step, label) => {
      const html = render({}, step);
      const shown = [...html.matchAll(/data-lab-surface="([^"]*)"/g)].map((match) => match[1]);
      expect(shown).toEqual([label]);
      expect((html.match(/class="card-info-trigger"/g) ?? []).length).toBe(1);
      expect(html).toContain(`Step ${step + 1} of ${CARDS.length}`);
      expect(html).not.toMatch(broken);
      expect(visible(html)).not.toMatch(unformatted);
    },
  );

  it("shows every card once across the steps, and a step change keeps what the reader set", () => {
    const html = render({ times: -1.7, keep: 0.2 });
    const shown = [...html.matchAll(/data-lab-surface="([^"]*)"/g)].map((match) => match[1]);
    expect(shown).toEqual(CARDS);
    // Paging away and back draws the same card from the same state.
    expect(render({ times: -1.7, keep: 0.2 }, 1)).toBe(render({ times: -1.7, keep: 0.2 }, 1));
    expect(render({ times: -1.7 }, 1)).not.toBe(render({ times: 1 }, 1));
  });

  it("gives every picture a spoken reading and every slider a label", () => {
    const html = render();
    const pictures = html.match(/<svg[^>]*role="img"[^>]*>/g) ?? [];
    expect(pictures.length).toBe(7);
    for (const picture of pictures) expect(picture).toMatch(/aria-label="[^"]{60,}"/);
    const sliders = html.match(/<input type="range"[^>]*>/g) ?? [];
    expect(sliders.length).toBe(15);
    for (const slider of sliders) expect(slider).toMatch(/aria-label="[^"]+"/);
  });

  it("follows each of the eight words on every card that follows one", () => {
    for (const word of WORD_IDS) {
      for (const other of ["whale", "bus", word]) {
        const html = render({ word, other });
        expect(html, `${word} with ${other}`).not.toMatch(broken);
        expect(visible(html)).not.toMatch(unformatted);
      }
    }
  });

  it("survives both ends of every control", () => {
    const extremes: Partial<ModuleState>[] = [
      { tipX: -3, tipY: -3 },
      { tipX: 3, tipY: 3 },
      { tipX: 0, tipY: 0 },
      { times: -2, push: "both" },
      { times: 2, push: "alive", word: "whale" },
      { times: 0 },
      { turn: 360, stretch: 0.5 },
      { turn: 90, stretch: 1.5, other: "horse" },
      { ma: -2, mb: -2, mc: -2, md: -2 },
      { ma: 2, mb: 2, mc: 2, md: 2 },
      { ma: 0, mb: 0, mc: 0, md: 0 },
      { first: "stretch", second: "stretch", stage: 0 },
      { first: "shear", second: "stretch", stage: 1 },
      { first: "flip", second: "flip", stage: 2 },
      { keep: 0, undo: "stretch" },
      { keep: 0.05, undo: "stretch" },
      { keep: 0.5, undo: "leave" },
      { pieces: 1 },
      { pieces: 4 },
      { round: 2, ca: -2, cb: 2, cc: -2, cd: 2, hint: 1 },
      { round: 1, ca: 0, cb: 0, cc: 0, cd: 0 },
      { predictDot: 0, predictOrder: 1, predictUndo: 2, predictPieces: 1 },
    ];
    for (const patch of extremes) {
      for (const step of [0, 7]) {
        const html = render(patch, step);
        expect(html, JSON.stringify(patch)).not.toMatch(broken);
        expect(visible(html), JSON.stringify(patch)).not.toMatch(unformatted);
      }
    }
  });

  it("marks a word a matrix throws off the map instead of losing it", () => {
    expect(visible(render({ ma: 2, mb: 0, mc: 0, md: 2 }))).toContain("off the map");
    expect(visible(render())).not.toContain("off the map");
  });

  it("explains a prediction only after the reader has made one", () => {
    const before = visible(render());
    expect(before).not.toContain("Not this time.");
    expect(before).not.toContain("A half turn flips the sign");
    const right = visible(render({ predictDot: 2 }));
    expect(right).toContain("Right.");
    expect(right).toContain("A half turn flips the sign");
    expect(visible(render({ predictDot: 0 }))).toContain("Not this time.");
  });

  it("wins each round of the challenge with that round's target matrix, and not before", () => {
    ROUNDS.forEach((round, index) => {
      const open = visible(render({ round: index }));
      expect(open).toContain("to go");
      expect(open).not.toContain("All eight words on target");
      const [ca, cb, cc, cd] = round.target;
      const won = visible(render({ round: index, ca, cb, cc, cd, solved: 2 ** (index + 1) - 1 }));
      expect(won).toContain("All eight words on target");
      expect(won).toContain(round.solvedText);
      expect(won.includes("Next round")).toBe(index < ROUNDS.length - 1);
    });
    expect(visible(render({ round: 2, ca: -1, cb: 0.5, cc: 0, cd: 1, solved: 7 }))).toContain("That is all three.");
  });

  it("draws the card each scene points at, in the state the scene sets", () => {
    for (const scene of scenes) {
      const html = render(scene.state ?? {}, scene.step ?? 0);
      expect(html, scene.id).toContain(`data-lab-surface="${scene.focus}"`);
      expect(html, scene.id).not.toMatch(broken);
    }
  });
});
