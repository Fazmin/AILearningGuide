import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TINY_CORPORA, trainTinyModel, type ModuleState } from "@app/module-sdk";
import definition from "./module";
import {
  BLOCK_IDS,
  EPOCHS_MAX,
  EPOCHS_MIN,
  HELD_OUT_MAX,
  HELD_OUT_MIN,
  LEARNING_RATE_MAX,
  LEARNING_RATE_MIN,
  REPLAY_MAX,
  REPLAY_STEP,
} from "./state";

const hydrate = (value: string) => definition.hydrateState(value);
const roundTrip = (state: ModuleState) => hydrate(definition.serializeState(state));

const renderLab = (state: ModuleState) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state,
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );

/** The only keys the lab stores, whatever a payload carried. */
const KEYS = [
  "fineTuneEpochs",
  "frozen",
  "hasExamples",
  "heldOutSentences",
  "knowledgeChanges",
  "learningRate",
  "needsCitations",
  "needsConsistency",
  "replayShare",
];

/** The four decision answers, all "no" until the learner says otherwise. */
const NO = { knowledgeChanges: false, needsCitations: false, needsConsistency: false, hasExamples: false };
/** The replay card's two controls at their defaults: no replay, three target sentences held out. */
const CARD = { replayShare: 0, heldOutSentences: 3 };

function expectBounded(state: ModuleState) {
  expect(Object.keys(state).sort()).toEqual(KEYS);
  expect(Number.isInteger(state.fineTuneEpochs)).toBe(true);
  expect(state.fineTuneEpochs as number).toBeGreaterThanOrEqual(EPOCHS_MIN);
  expect(state.fineTuneEpochs as number).toBeLessThanOrEqual(EPOCHS_MAX);
  expect(Number.isFinite(state.learningRate)).toBe(true);
  expect(state.learningRate as number).toBeGreaterThanOrEqual(LEARNING_RATE_MIN);
  expect(state.learningRate as number).toBeLessThanOrEqual(LEARNING_RATE_MAX);
  expect(Array.isArray(state.frozen)).toBe(true);
  for (const id of state.frozen as string[]) expect(BLOCK_IDS).toContain(id);
  expect(new Set(state.frozen as string[]).size).toBe((state.frozen as string[]).length);
  for (const key of ["knowledgeChanges", "needsCitations", "needsConsistency", "hasExamples"]) {
    expect(typeof state[key], key).toBe("boolean");
  }
  expect(state.replayShare as number).toBeGreaterThanOrEqual(0);
  expect(state.replayShare as number).toBeLessThanOrEqual(REPLAY_MAX);
  expect((state.replayShare as number) % REPLAY_STEP).toBe(0);
  expect(Number.isInteger(state.heldOutSentences)).toBe(true);
  expect(state.heldOutSentences as number).toBeGreaterThanOrEqual(HELD_OUT_MIN);
  expect(state.heldOutSentences as number).toBeLessThanOrEqual(HELD_OUT_MAX);
}

describe("fine-tuning-transfer state", () => {
  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [
      definition.initialState,
      { fineTuneEpochs: 0, learningRate: 0.05, frozen: [], ...NO, ...CARD },
      {
        fineTuneEpochs: 60,
        learningRate: 2,
        frozen: ["block-0", "block-1", "block-2", "block-3", "block-4"],
        knowledgeChanges: true,
        needsCitations: true,
        needsConsistency: true,
        hasExamples: true,
        replayShare: 100,
        heldOutSentences: 4,
      },
      { fineTuneEpochs: 37, learningRate: 1.35, frozen: ["block-3", "block-0"], ...NO, needsConsistency: true, replayShare: 37.5, heldOutSentences: 0 },
      { fineTuneEpochs: 20, learningRate: 0.6, frozen: ["block-1"], ...NO, ...CARD, hasExamples: true },
    ];
    for (const state of states) expect(roundTrip(state)).toEqual(state);
  });

  it("keeps the shipped defaults", () => {
    expect(definition.initialState).toEqual({ fineTuneEpochs: 20, learningRate: 0.6, frozen: [], ...NO, ...CARD });
    expect(hydrate("{}")).toEqual(definition.initialState);
  });

  it("uses the slider bounds the interface offers", () => {
    const html = renderLab(definition.initialState);
    const slider = (label: string) => {
      const match = html.match(new RegExp(`aria-label="${label}" min="([^"]*)" max="([^"]*)"`));
      expect(match, `${label} slider`).not.toBeNull();
      return [Number(match![1]), Number(match![2])];
    };
    expect(slider("Fine-tune epochs")).toEqual([EPOCHS_MIN, EPOCHS_MAX]);
    expect(slider("Fine-tune learning rate")).toEqual([LEARNING_RATE_MIN, LEARNING_RATE_MAX]);
  });

  it("clamps out-of-range numbers to the control's range", () => {
    expect(hydrate(JSON.stringify({ fineTuneEpochs: 1e9 })).fineTuneEpochs).toBe(EPOCHS_MAX);
    expect(hydrate(JSON.stringify({ fineTuneEpochs: 61 })).fineTuneEpochs).toBe(EPOCHS_MAX);
    expect(hydrate(JSON.stringify({ fineTuneEpochs: -5 })).fineTuneEpochs).toBe(EPOCHS_MIN);
    expect(hydrate(JSON.stringify({ fineTuneEpochs: 12.6 })).fineTuneEpochs).toBe(13);
    expect(hydrate(JSON.stringify({ learningRate: 1e9 })).learningRate).toBe(LEARNING_RATE_MAX);
    expect(hydrate(JSON.stringify({ learningRate: -3 })).learningRate).toBe(LEARNING_RATE_MIN);
    expect(hydrate(JSON.stringify({ learningRate: 0 })).learningRate).toBe(LEARNING_RATE_MIN);
  });

  it("replaces non-finite numbers with the defaults", () => {
    // JSON has no NaN or Infinity: stringify writes null, and 1e999 parses to Infinity.
    expect(JSON.stringify({ fineTuneEpochs: Number.NaN })).toBe('{"fineTuneEpochs":null}');
    expect(hydrate(JSON.stringify({ fineTuneEpochs: Number.NaN, learningRate: Infinity }))).toEqual(
      definition.initialState,
    );
    const infinite = hydrate('{"fineTuneEpochs":1e999,"learningRate":-1e999,"frozen":[]}');
    expect(infinite).toEqual(definition.initialState);
    expectBounded(hydrate('{"fineTuneEpochs":1e999}'));
  });

  it("replaces wrong types with the defaults", () => {
    const wrong = [
      { fineTuneEpochs: "40", learningRate: "1.5", frozen: "block-0" },
      { fineTuneEpochs: true, learningRate: false, frozen: 7 },
      { fineTuneEpochs: [30], learningRate: { value: 1 }, frozen: { 0: "block-0" } },
      { fineTuneEpochs: null, learningRate: null, frozen: null },
    ];
    for (const payload of wrong) {
      expect(hydrate(JSON.stringify(payload))).toEqual(definition.initialState);
    }
  });

  it("rejects payloads that are not objects", () => {
    for (const payload of ["[]", "[1,2,3]", '[{"fineTuneEpochs":9}]', "null", "42", '"text"', "true", "", "{", "not json", "{'a':1}"]) {
      expect(hydrate(payload), payload).toEqual(definition.initialState);
    }
  });

  it("drops unknown keys, unknown block ids, non-string entries and duplicates", () => {
    // Written as text: an object literal's __proto__ would never reach the JSON.
    const hydrated = hydrate(
      `{"fineTuneEpochs":10,"learningRate":0.5,"extra":"${"x".repeat(10_000)}","__proto__":{"polluted":true},` +
        '"frozen":["block-1","block-1","block-99","nope",3,null,["block-2"],"block-0"]}',
    );
    expect(hydrated).toEqual({ fineTuneEpochs: 10, learningRate: 0.5, frozen: ["block-1", "block-0"], ...NO, ...CARD });
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    expectBounded(hydrated);
  });

  it("does not share the frozen array between hydrations", () => {
    const first = hydrate("{}");
    (first.frozen as string[]).push("block-0");
    expect(hydrate("{}").frozen).toEqual([]);
    expect(definition.initialState.frozen).toEqual([]);
  });

  it("always yields a bounded state for hostile payloads", () => {
    const hostile = [
      '{"fineTuneEpochs":1e9,"learningRate":1e9}',
      '{"fineTuneEpochs":-1e9,"learningRate":-1e9}',
      '{"fineTuneEpochs":1e999,"learningRate":1e999,"frozen":["block-0"]}',
      '{"fineTuneEpochs":9007199254740993}',
      '{"fineTuneEpochs":"60","learningRate":[2],"frozen":[[],{}]}',
      "[]",
      "null",
      "{",
    ];
    for (const payload of hostile) expectBounded(hydrate(payload));
  });

  it("migrates a version 1 payload: the decision answers and the replay card's controls are new", () => {
    // Version 1 stored only these three keys.
    const legacy = JSON.stringify({ fineTuneEpochs: 33, learningRate: 1.1, frozen: ["block-2"] });
    expect(hydrate(legacy)).toEqual({ fineTuneEpochs: 33, learningRate: 1.1, frozen: ["block-2"], ...NO, ...CARD });
    expect(definition.stateVersion).toBe(3);
  });

  it("migrates a version 2 payload: the replay card's two controls default and nothing else changes", () => {
    // Version 2 stored the three controls and the four decision answers, and no replay or held-out setting.
    const version2 = JSON.stringify({
      fineTuneEpochs: 45,
      learningRate: 0.9,
      frozen: ["block-1", "block-4"],
      knowledgeChanges: true,
      needsCitations: false,
      needsConsistency: true,
      hasExamples: true,
    });
    expect(hydrate(version2)).toEqual({
      fineTuneEpochs: 45,
      learningRate: 0.9,
      frozen: ["block-1", "block-4"],
      knowledgeChanges: true,
      needsCitations: false,
      needsConsistency: true,
      hasExamples: true,
      ...CARD,
    });
  });

  it("snaps the replay share to a whole sentence and clamps both of the card's controls", () => {
    const share = (value: unknown) => hydrate(JSON.stringify({ replayShare: value })).replayShare;
    const held = (value: unknown) => hydrate(JSON.stringify({ heldOutSentences: value })).heldOutSentences;
    expect([share(0), share(12.5), share(50), share(100)]).toEqual([0, 12.5, 50, 100]);
    expect([share(30), share(1e9), share(-1e9), share(6)]).toEqual([25, 100, 0, 0]);
    expect([share("50"), share(null), share(true), share([50])]).toEqual([0, 0, 0, 0]);
    expect([held(0), held(4), held(2.6), held(1e9), held(-1e9)]).toEqual([0, 4, 3, 4, 0]);
    expect([held("2"), held(null), held(false), held({})]).toEqual([3, 3, 3, 3]);
    expect(hydrate('{"replayShare":1e999,"heldOutSentences":-1e999}')).toEqual(definition.initialState);
  });

  it("keeps the four decision answers only when they are real booleans", () => {
    const answered = hydrate(
      JSON.stringify({ knowledgeChanges: true, needsCitations: true, needsConsistency: false, hasExamples: true }),
    );
    expect(answered).toMatchObject({
      knowledgeChanges: true,
      needsCitations: true,
      needsConsistency: false,
      hasExamples: true,
    });
    const wrong = hydrate(
      JSON.stringify({ knowledgeChanges: "true", needsCitations: 1, needsConsistency: null, hasExamples: [true] }),
    );
    expect(wrong).toEqual(definition.initialState);
    expectBounded(answered);
    expectBounded(wrong);
  });

  it("finishes the clamped worst case quickly", () => {
    const { fineTuneEpochs, learningRate } = hydrate('{"fineTuneEpochs":1e9,"learningRate":1e9}');
    expect(fineTuneEpochs as number).toBeLessThanOrEqual(EPOCHS_MAX);

    const started = performance.now();
    const run = trainTinyModel({
      text: TINY_CORPORA.recipes.text,
      epochs: fineTuneEpochs as number,
      learningRate: learningRate as number,
      seed: 2,
      checkpoints: 9,
    });
    const elapsed = performance.now() - started;

    // One minibatch step per 16 pairs: the bound on epochs is a bound on work.
    expect(run.steps).toBeLessThanOrEqual(Math.ceil(TINY_CORPORA.recipes.text.length / 16) * EPOCHS_MAX);
    expect(elapsed).toBeLessThan(2000);
  });

  it("renders a hostile live state at the slider maximum instead of hanging", () => {
    const html = renderLab({ fineTuneEpochs: 1e9, learningRate: Number.NaN, frozen: ["block-0", "bogus"] });
    expect(html).toContain(`<output>${EPOCHS_MAX} passes</output>`);
    expect(html).toMatch(new RegExp(`aria-label="Fine-tune epochs"[^>]*value="${EPOCHS_MAX}"`));
    // The non-finite learning rate falls back to the default and the unknown block id is ignored.
    expect(html).toContain("<output>0.60</output>");
    expect(html).toContain("<span>Frozen rows</span><strong>6 / 30</strong>");
    expect(html).not.toContain("NaN");
  });
});
