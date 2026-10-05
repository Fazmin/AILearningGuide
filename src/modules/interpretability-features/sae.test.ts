import { beforeAll, describe, expect, it } from "vitest";
import cache from "./assets/sae-top-activations.json";
import {
  ACTIVE,
  activeFeatures,
  decoderCosine,
  decoderOverlap,
  dominantFocus,
  EARLY_POSITIONS,
  examplesFor,
  featureFlags,
  FEATURES,
  isStartArtefact,
  matchesHypothesis,
  pairActivity,
  reconstruct,
  seenBefore,
  testLabel,
  topKCurve,
  unexplainedFraction,
  WIDTH,
  type CachedRecord,
} from "./sae";
import { DEFAULT_TEXT, fixtureRuns, runText, sae, type TextRun } from "./test-support";

const records = cache.top_activations as unknown as CachedRecord[];
const record = (feature: number) => records.find((entry) => entry.feature === feature)!;

let probe: TextRun;
beforeAll(() => {
  probe = runText(DEFAULT_TEXT);
  fixtureRuns();
}, 120_000);

describe("shipped SAE weights", () => {
  it("reads a 1,024-feature dictionary over the 256-wide residual", () => {
    expect(cache.features).toBe(FEATURES);
    expect(cache.active_features).toBe(ACTIVE);
    expect(cache.layer).toBe(1);
    expect(cache.normalization_mean).toHaveLength(WIDTH);
    expect(cache.normalization_scale).toBeCloseTo(3.5784, 4);
    expect(sae.decoder).toHaveLength(FEATURES * WIDTH);
  });

  it("has unit-length decoder directions, as training renormalized them", () => {
    for (const feature of [0, 17, 160, 876, 1023]) {
      let norm = 0;
      for (let d = 0; d < WIDTH; d += 1) norm += sae.decoder[feature * WIDTH + d] ** 2;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 4);
    }
  });

  it("records the training the lesson quotes", () => {
    expect(cache.training_windows).toBe(1600);
    expect(cache.training_rows).toBe(102400);
    expect(cache.skipped_window_start_positions).toBe(0);
    const last = cache.training[cache.training.length - 1];
    expect(last.step).toBe(10000);
    expect(last.reconstruction).toBeCloseTo(0.0470, 4);
    expect(last.active_fraction).toBe(0.03125);
    // The L1 term adds about 1e-4 to the final loss: sparsity comes from TopK.
    expect(last.loss - last.reconstruction).toBeCloseTo(9.65e-5, 6);
    expect(cache.evaluation.heldout_windows.fraction_of_variance_unexplained).toBeCloseTo(0.0514, 4);
    expect(cache.evaluation.training_windows.fraction_of_variance_unexplained).toBeCloseTo(0.0465, 4);
    expect(cache.evaluation.heldout_windows.fraction_unexplained_position_zero).toBeCloseTo(0.0026, 4);
    expect(cache.evaluation.heldout_windows.fraction_unexplained_from_position_4).toBeCloseTo(0.0564, 4);
    expect(cache.position_summary.scanned_windows).toBe(1800);
    expect(cache.position_summary.scanned_tokens).toBe(115200);
    expect(cache.position_summary.dead_features).toBe(0);
  });
});

describe("decoder overlap: superposition, and one pair that is not what it seems", () => {
  const overlap = decoderOverlap(sae);

  it("packs 1,024 directions into 256 dimensions with small but nonzero typical overlap", () => {
    expect(overlap.pairs).toBe((FEATURES * (FEATURES - 1)) / 2);
    expect(overlap.meanAbs).toBeCloseTo(0.0694, 4);
    expect(overlap.above).toBeCloseTo(0.0029, 4);
  });

  it("finds a largest overlap of 0.992 between features 108 and 499, whose cosine is negative", () => {
    // The retrain moved this from 0.78 to 0.99, so the old 0.85 bound no longer holds.
    expect(overlap.maxAbs).toBeCloseTo(0.9919, 4);
    expect(overlap.strongest).toEqual({ a: 108, b: 499, cosine: expect.closeTo(-0.9919, 4) });
    expect(decoderCosine(sae, 108, 499)).toBeCloseTo(-0.9919, 4);
  });

  it("has exactly seven pairs above 0.9, and every one of them points the opposite way", () => {
    expect(overlap.nearIdentical).toBe(7);
    const strong: Array<[number, number, number]> = [];
    for (let a = 0; a < FEATURES; a += 1) {
      for (let b = a + 1; b < FEATURES; b += 1) {
        const cosine = decoderCosine(sae, a, b);
        if (Math.abs(cosine) > 0.9) strong.push([a, b, cosine]);
      }
    }
    expect(strong.map(([a, b]) => `${a}/${b}`)).toEqual(["108/499", "121/870", "222/511", "259/902", "449/876", "740/750", "934/962"]);
    expect(strong.every(([, , cosine]) => cosine < -0.95)).toBe(true);
  });

  it("never has both features of such a pair active on the same character, but one of them almost always", () => {
    const rows = fixtureRuns().flatMap((run) => run.features);
    expect(rows).toHaveLength(768);
    for (const [a, b] of [[108, 499], [121, 870], [222, 511], [259, 902], [449, 876], [740, 750], [934, 962]]) {
      expect(pairActivity(rows, a, b).both, `${a}/${b}`).toBe(0);
    }
    expect(pairActivity(rows, 108, 499)).toEqual({ both: 0, either: 734, total: 768 });
    expect(pairActivity(probe.features, 108, 499).both).toBe(0);
  });

  it("splits the text between 499 (spaces, word boundaries) and 108 (vowels, inside words) from position 4 on", () => {
    const cells = fixtureRuns().flatMap((run) =>
      Array.from(run.text, (character, position) => ({ character, position, features: run.features[position] })).filter((cell) => cell.position >= 4),
    );
    const count = (matches: (character: string) => boolean) => {
      const chosen = cells.filter((cell) => matches(cell.character));
      return {
        n: chosen.length,
        a499: chosen.filter((cell) => cell.features[499] > 0).length,
        a108: chosen.filter((cell) => cell.features[108] > 0).length,
      };
    };
    expect(count((character) => character === " ")).toEqual({ n: 111, a499: 101, a108: 6 });
    expect(count((character) => "aeiou".includes(character))).toEqual({ n: 181, a499: 41, a108: 132 });
  });
});

describe("the default probe text: reconstruction figures the lesson quotes", () => {
  const position = 4;
  const code = () => {
    const active = activeFeatures(probe.features[position]);
    return { active, x: probe.x[position] };
  };

  it("keeps exactly 32 features at the first space and leaves 1.0% unexplained", () => {
    const { active, x } = code();
    expect(probe.text[position]).toBe(" ");
    expect(active).toHaveLength(ACTIVE);
    const fraction = unexplainedFraction(x, reconstruct(sae, active));
    expect(fraction).toBeCloseTo(0.009624, 5);
    let dot = 0;
    let a = 0;
    let b = 0;
    const rebuilt = reconstruct(sae, active);
    for (let index = 0; index < WIDTH; index += 1) {
      dot += x[index] * rebuilt[index];
      a += x[index] ** 2;
      b += rebuilt[index] ** 2;
    }
    expect(dot / Math.sqrt(a * b)).toBeCloseTo(0.995, 3);
  });

  it("falls steeply, flattens, and never reaches zero: 53.5% kept at 1, 4.6% at 8, 1.0% at 32", () => {
    const { active, x } = code();
    const curve = topKCurve(sae, x, active);
    expect(curve).toHaveLength(ACTIVE);
    expect(curve[0]).toBeCloseTo(0.5349, 4);
    expect(curve[7]).toBeCloseTo(0.04558, 5);
    expect(curve[ACTIVE - 1]).toBeCloseTo(0.009624, 5);
    expect(curve[ACTIVE - 1]).toBeGreaterThan(0);
    expect(curve[0] - curve[7]).toBeGreaterThan(curve[7] - curve[ACTIVE - 1]);
    // Not strictly falling: a feature fitted for the whole code can add error when kept alone with a few others.
    const risesAt = (values: number[]) => values.flatMap((value, index) => (index > 0 && value > values[index - 1] + 1e-9 ? [index + 1] : []));
    expect(risesAt(curve)).toEqual([29]);
    expect(curve[28]).toBeCloseTo(0.01026, 5);
    expect(curve[27]).toBeCloseTo(0.00998, 5);
    const k = probe.text.indexOf("K");
    const kCurve = topKCurve(sae, probe.x[k], activeFeatures(probe.features[k]));
    expect(risesAt(kCurve)).toEqual([16, 29, 30, 31]);
  });

  it("ticks back up at least once in 42 of the 53 positions of the default text, yet always ends far below k = 1", () => {
    let ticks = 0;
    probe.x.forEach((x, index) => {
      const curve = topKCurve(sae, x, activeFeatures(probe.features[index]));
      if (curve.some((value, k) => k > 0 && value > curve[k - 1] + 1e-9)) ticks += 1;
      expect(curve[ACTIVE - 1]).toBeLessThan(curve[0]);
    });
    expect(ticks).toBe(42);
  });

  it("names the largest features at the first space: 121, 499 (flagged), 222 and 264", () => {
    const { active } = code();
    expect(active.slice(0, 4).map((entry) => entry.feature)).toEqual([121, 499, 222, 264]);
    expect(active[0].value).toBeCloseTo(14.04, 2);
    expect(active[1].value).toBeCloseTo(12.82, 2);
    expect(isStartArtefact(record(499))).toBe(true);
    expect(isStartArtefact(record(121))).toBe(false);
  });

  it("leaves from 0.0% at position 0 to 22.3% at the K of KING unexplained, 7.0% on average", () => {
    const shares = probe.x.map((x, index) => unexplainedFraction(x, reconstruct(sae, activeFeatures(probe.features[index]))));
    expect(shares[0]).toBeLessThan(0.001);
    const largest = Math.max(...shares);
    expect(shares.indexOf(largest)).toBe(28);
    expect(probe.text[28]).toBe("K");
    expect(largest).toBeCloseTo(0.2234, 4);
    expect(shares.reduce((sum, value) => sum + value, 0) / shares.length).toBeCloseTo(0.0696, 4);
  });

  it("has no window-start feature among the six largest at the K of KING", () => {
    const k = probe.text.indexOf("K");
    const leaders = activeFeatures(probe.features[k]).slice(0, 6).map((entry) => entry.feature);
    expect(leaders).toEqual([973, 13, 932, 751, 282, 108]);
    for (const feature of leaders) {
      const entry = records.find((item) => item.feature === feature);
      if (entry) expect(isStartArtefact(entry), `${feature}`).toBe(false);
    }
    for (const [position, feature] of [[0, 959], [1, 876], [2, 876]] as const) {
      expect(activeFeatures(probe.features[position])[0].feature).toBe(feature);
    }
  });

  it("puts the window-start features first at positions 0 to 2 and a clean one first at position 4", () => {
    const leading = (index: number, count: number) =>
      activeFeatures(probe.features[index]).slice(0, count).map((entry) => entry.feature);
    expect(leading(0, 6)).toEqual([959, 924, 133, 605, 482, 203]);
    for (const feature of leading(0, 6)) expect(isStartArtefact(record(feature)), `${feature}`).toBe(true);
    expect(leading(1, 2)).toEqual([876, 356]);
    expect(activeFeatures(probe.features[1])[0].value).toBeCloseTo(7.8, 1);
    expect(isStartArtefact(record(876)) && isStartArtefact(record(356))).toBe(true);
  });
});

describe("what the cache records and does not guarantee", () => {
  it("records a window position for every cached example, and 31 features have an in-context list", () => {
    for (const entry of records) {
      for (const example of entry.examples) {
        expect(Number.isInteger(example.window_position)).toBe(true);
        expect(example.window_position).toBeGreaterThanOrEqual(0);
        expect(example.window_position).toBeLessThan(64);
      }
    }
    expect(records.filter((entry) => entry.examples_in_context).length).toBe(31);
    for (const entry of records.filter((item) => item.examples_in_context)) {
      expect(entry.examples_in_context!.every((example) => example.window_position >= EARLY_POSITIONS)).toBe(true);
    }
    expect(EARLY_POSITIONS).toBe(cache.position_summary.early_positions);
  });

  it("flags 21 of the 128 cached features as window-start artefacts, 18 as position-0 artefacts, none as position-0-only", () => {
    expect(records).toHaveLength(128);
    expect(records.filter((entry) => entry.window_start_artefact)).toHaveLength(21);
    expect(records.filter((entry) => entry.position_zero_artefact)).toHaveLength(18);
    expect(records.filter((entry) => entry.position_zero_only)).toHaveLength(0);
    expect(cache.position_summary.position_zero_only_features).toEqual([]);
    expect(cache.position_summary.position_zero_dominated_features).toHaveLength(24);
    const identical = records.filter((entry) => featureFlags(entry).includes("same activation in every example"));
    expect(identical.map((entry) => entry.feature).sort((a, b) => a - b)).toEqual(
      [...cache.position_summary.selected_identical_activation_features].sort((a, b) => a - b),
    );
    expect(identical).toHaveLength(17);
  });

  it("ranks an artefact first: feature 876 has the largest maximum, 56.9, and its top examples are at positions 1 and 2", () => {
    expect(records[0].feature).toBe(876);
    expect(records.map((entry) => entry.maximum)).toEqual([...records.map((entry) => entry.maximum)].sort((a, b) => b - a));
    const top = record(876);
    expect(top.maximum).toBeCloseTo(56.86, 2);
    expect(top.examples.map((example) => example.window_position)).toEqual([1, 1, 1, 1, 1, 1, 2, 1]);
    expect(top.window_start_artefact).toBe(true);
    expect(top.position_zero_artefact).toBe(false);
    expect(featureFlags(top)).toEqual(["window-start artefact"]);
    expect(top.fires).toBe(36380);
    expect(top.fires / cache.position_summary.scanned_tokens).toBeCloseTo(0.3158, 4);
    // It is not a "first character" feature either: it keeps firing, hard, later in the window.
    const later = examplesFor(top, "context");
    expect(later[0].activation).toBeCloseTo(36.34, 2);
    expect(later[0].window_position).toBe(19);
    expect(later[0].text[later[0].focus_offset]).toBe("&");
    expect(later.every((example) => example.window_position >= EARLY_POSITIONS)).toBe(true);
    expect(examplesFor(top, "top")).toBe(top.examples);
    expect(examplesFor(record(121), "context")).toBe(record(121).examples);
  });

  it("ranks feature 499 second, also flagged, though it fires on a space at position 4 of the default text", () => {
    expect(records[1].feature).toBe(499);
    expect(isStartArtefact(records[1])).toBe(true);
    expect(records[2].feature).toBe(121);
    expect(isStartArtefact(records[2])).toBe(false);
  });

  it("shows that the examples of an unflagged feature are all in context, and a position-0 feature's are all the first character", () => {
    expect(record(121).examples.every((example) => example.window_position >= EARLY_POSITIONS)).toBe(true);
    const zero = record(959);
    expect(zero.position_zero_artefact).toBe(true);
    expect(zero.examples.every((example) => example.window_position === 0)).toBe(true);
    expect(zero.examples.every((example) => example.activation === zero.examples[0].activation)).toBe(true);
    expect(zero.maximum).toBeCloseTo(8.44, 2);
    expect(zero.maximum_after_position_zero).toBeCloseTo(2.18, 2);
    expect(zero.position_zero_share).toBeCloseTo(0.8234, 4);
  });

  it("reproduces cached activations from the excerpt and the window position: the end-to-end check", () => {
    // Position 0: the residual depends on the single character, so "r" alone gives feature 959's 8.44.
    expect(runText("r").features[0][959]).toBeCloseTo(record(959).examples[0].activation, 3);
    // A later example: the excerpt carries 32 characters of left context, the window began window_position back.
    const example = record(740).examples[7];
    expect(example.window_position).toBe(3);
    const window = example.text.slice(example.focus_offset - example.window_position, example.focus_offset + 1);
    expect(runText(window).features[example.window_position][740]).toBeCloseTo(example.activation, 2);
    expect(seenBefore(example)).toBe(3);
  });

  it("writes every line break as a space in an excerpt, so a space can hide a line break", () => {
    // Feature 876's strongest example reads "m" then a space at window position 1: 45.44 in the cache.
    const first = record(876).examples[2];
    expect(first.activation).toBeCloseTo(45.44, 2);
    expect(first.text[first.focus_offset - 1]).toBe("m");
    expect(first.text[first.focus_offset]).toBe(" ");
    expect(runText("m ").features[1][876]).toBeLessThan(16);
    expect(runText("m\n").features[1][876]).toBeCloseTo(first.activation, 2);
  });
});

describe("label tests: the cache's labels are hypotheses", () => {
  const fixture = () => fixtureRuns();
  const score = (feature: number, character: string, threshold: number, hypothesis: Parameters<typeof testLabel>[3] = "char") => {
    let truePositive = 0;
    let falsePositive = 0;
    let falseNegative = 0;
    for (const run of fixture()) {
      const result = testLabel(run.text, run.features.map((row) => row[feature]), threshold, hypothesis, character);
      truePositive += result.truePositive;
      falsePositive += result.falsePositive;
      falseNegative += result.falseNegative;
    }
    return { truePositive, falsePositive, falseNegative };
  };

  it("summarizes the focus characters of cached examples", () => {
    expect(dominantFocus(record(121).examples)).toEqual({ character: " ", count: 8, distinct: 1 });
    expect(dominantFocus(record(511).examples)).toEqual({ character: ",", count: 8, distinct: 1 });
    expect(dominantFocus(record(683).examples)).toEqual({ character: "y", count: 8, distinct: 1 });
    expect(dominantFocus(record(160).examples)).toEqual({ character: "-", count: 8, distinct: 1 });
  });

  it("scores a label as precision and recall at a threshold", () => {
    const result = testLabel("a b", [0, 3, 0.5], 1, "space", "");
    expect(result).toEqual({ truePositive: 1, falsePositive: 0, falseNegative: 0, precision: 1, recall: 1 });
    const loose = testLabel("hah", [5, 4, 0], 1, "char", "h");
    expect(loose.precision).toBe(0.5);
    expect(loose.recall).toBe(0.5);
    expect(testLabel("ab", [0, 0], 1, "space", "").precision).toBeNull();
  });

  it("matches a window-start label by window position, never by excerpt offset", () => {
    expect(matchesHypothesis("start", "x", 3, "")).toBe(true);
    expect(matchesHypothesis("start", "x", 4, "")).toBe(false);
    // Every cached example of #876 is at window position 1 or 2, but its focus_offset is 32.
    const top = record(876);
    const matched = top.examples.filter((example) => matchesHypothesis("start", example.text[example.focus_offset], example.window_position, ""));
    expect(matched).toHaveLength(8);
    expect(top.examples.every((example) => example.focus_offset === 32)).toBe(true);
    expect(matchesHypothesis("newline", "\n", 0, "")).toBe(true);
    expect(matchesHypothesis("newline", " ", 0, "")).toBe(false);
  });

  it("scores feature 121 as a space label: precision falls and recall rises as the threshold drops", () => {
    expect(score(121, " ", 2, "space")).toEqual({ truePositive: 114, falsePositive: 67, falseNegative: 3 });
    expect(score(121, " ", 5, "space")).toEqual({ truePositive: 106, falsePositive: 4, falseNegative: 11 });
    expect(score(121, " ", 8, "space")).toEqual({ truePositive: 65, falsePositive: 0, falseNegative: 52 });
  });

  it("scores feature 121 on the default text itself: 100% precision and 70% recall at 5.0, 90% and 90% at 2.0", () => {
    const at = (threshold: number) =>
      testLabel(probe.text, probe.features.map((row) => row[121]), threshold, "space", "");
    expect(at(5)).toMatchObject({ truePositive: 7, falsePositive: 0, falseNegative: 3, precision: 1, recall: 0.7 });
    expect(at(2)).toMatchObject({ truePositive: 9, falsePositive: 1, falseNegative: 1, precision: 0.9, recall: 0.9 });
    expect(at(8)).toMatchObject({ truePositive: 5, falsePositive: 0, falseNegative: 5 });
  });

  it("scores feature 511 as a comma label from its top examples: near-perfect recall, very poor precision", () => {
    // Its eight cached top examples are all commas, at about 14 (the cache's maximum is 14.0).
    expect(record(511).maximum).toBeCloseTo(14.0, 1);
    expect(score(511, ",", 3)).toEqual({ truePositive: 19, falsePositive: 184, falseNegative: 0 });
    expect(score(511, ",", 5)).toEqual({ truePositive: 18, falsePositive: 102, falseNegative: 1 });
    expect(score(511, ",", 8)).toEqual({ truePositive: 9, falsePositive: 21, falseNegative: 10 });
    expect(score(511, ",", 10)).toEqual({ truePositive: 4, falsePositive: 4, falseNegative: 15 });
    const onProbe = testLabel(probe.text, probe.features.map((row) => row[511]), 3, "char", ",");
    expect(onProbe).toMatchObject({ truePositive: 1, falsePositive: 15, falseNegative: 0 });
    expect(onProbe.precision).toBeCloseTo(0.0625, 4);
  });

  it("scores feature 53 as a space label (0%) and as a line-break label (41% precision, 62% recall at 5.0)", () => {
    expect(dominantFocus(record(53).examples)).toEqual({ character: " ", count: 8, distinct: 1 });
    expect(score(53, " ", 5, "space")).toEqual({ truePositive: 0, falsePositive: 44, falseNegative: 117 });
    expect(score(53, "\n", 5, "newline")).toEqual({ truePositive: 18, falsePositive: 26, falseNegative: 11 });
    expect(score(53, "\n", 10, "newline")).toEqual({ truePositive: 9, falsePositive: 0, falseNegative: 20 });
  });

  it("scores feature 876 as a first-four-characters label: neither precise nor complete", () => {
    expect(score(876, "", 5, "start")).toEqual({ truePositive: 19, falsePositive: 25, falseNegative: 29 });
    expect(score(876, "", 10, "start")).toEqual({ truePositive: 12, falsePositive: 4, falseNegative: 36 });
    const onProbe = testLabel(probe.text, probe.features.map((row) => row[876]), 5, "start", "");
    expect(onProbe).toMatchObject({ truePositive: 2, falsePositive: 2, falseNegative: 2, precision: 0.5, recall: 0.5 });
  });

  it("measures how large and how position-bound feature 876 is on held-out windows", () => {
    const runs = fixtureRuns();
    const at = (position: number) => runs.map((run) => run.features[position][876]);
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(Math.min(...at(1))).toBeGreaterThan(8);
    expect(Math.max(...at(1))).toBeLessThan(18.2);
    expect(mean(at(1))).toBeCloseTo(13.26, 2);
    expect(mean(at(2))).toBeCloseTo(5.57, 2);
    const from8 = runs.flatMap((run) => run.features.slice(8).map((row) => row[876]));
    expect(mean(from8)).toBeCloseTo(0.64, 2);
    // Later in a window it still fires hard now and then: the cache's in-context maximum is 36.3.
    expect(Math.max(...from8)).toBeGreaterThan(10);
  });

  it("measures the residual's size at the start of a window: 32.5 at position 0, 21.5 at position 1, 15.0 from position 4", () => {
    const norm = (row: Float32Array) => Math.sqrt(row.reduce((sum, value) => sum + value * value, 0));
    const runs = fixtureRuns();
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(mean(runs.map((run) => norm(run.x[0])))).toBeCloseTo(32.54, 2);
    expect(mean(runs.map((run) => norm(run.x[1])))).toBeCloseTo(21.46, 2);
    expect(mean(runs.flatMap((run) => run.x.slice(4).map(norm)))).toBeCloseTo(14.98, 2);
  });
});
