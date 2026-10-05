/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import cache from "./assets/sae-top-activations.json";
import {
  activeFeatures,
  decoderOverlap,
  pairActivity,
  reconstruct,
  testLabel,
  topKCurve,
  unexplainedFraction,
  type CachedRecord,
} from "./sae";
import { decoderDirection, distributionOf, nextLogits, prefixRows, steeredResidual, strengthSweep } from "./steering";
import { DEFAULT_TEXT, fixtureRuns, runText, SCALE, sae, transformer, type TextRun } from "./test-support";
import { tokenId } from "./vocabulary";
import { teachingTransformer } from "@app/module-sdk";

/**
 * Pins the figures the lessons and card explanations quote: each figure is measured here with the
 * lab's own code and formatted the way the text writes it, and the text must contain that string.
 * If a retrain, a recipe change or an edit moves a number, this file fails until the prose follows.
 */
const folder = resolve(process.cwd(), "src/modules/interpretability-features");
/** Markdown wraps lines, so a figure and its unit can sit on two lines; compare with whitespace collapsed. */
const read = (path: string) => readFileSync(resolve(folder, path), "utf8").replace(/\s+/g, " ");
const docs = {
  standard: read("content/standard.mdx"),
  plain: read("content/plain.mdx"),
  cards: read("card-info.ts") + read("module.ts"),
};
type Doc = keyof typeof docs;
const everywhere: Doc[] = ["standard", "plain", "cards"];

const records = cache.top_activations as unknown as CachedRecord[];
const record = (feature: number) => records.find((entry) => entry.feature === feature)!;
const pct = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const change = (value: number) => `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(2)}`;

let probe: TextRun;
beforeAll(() => {
  probe = runText(DEFAULT_TEXT);
  fixtureRuns();
}, 120_000);

const checks: Array<{ figure: () => string; docs: Doc[] }> = [];
const quote = (figure: () => string, where: Doc[] = everywhere) => checks.push({ figure, docs: where });

const posFour = () => {
  const active = activeFeatures(probe.features[4]);
  return { active, curve: topKCurve(sae, probe.x[4], active), x: probe.x[4] };
};
const scores = () => probe.x.map((x, index) => unexplainedFraction(x, reconstruct(sae, activeFeatures(probe.features[index]))));
const fixtureScore = (feature: number, character: string, threshold: number, hypothesis: Parameters<typeof testLabel>[3]) => {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  for (const run of fixtureRuns()) {
    const result = testLabel(run.text, run.features.map((row) => row[feature]), threshold, hypothesis, character);
    truePositive += result.truePositive;
    falsePositive += result.falsePositive;
    falseNegative += result.falseNegative;
  }
  return {
    truePositive,
    falsePositive,
    falseNegative,
    precision: truePositive / (truePositive + falsePositive),
    recall: truePositive / (truePositive + falseNegative),
  };
};
const probeScore = (feature: number, character: string, threshold: number, hypothesis: Parameters<typeof testLabel>[3]) =>
  testLabel(probe.text, probe.features.map((row) => row[feature]), threshold, hypothesis, character);
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
const norm = (row: Float32Array) => Math.sqrt(row.reduce((sum, value) => sum + value * value, 0));

// ---- Reconstruction (card 1 and 3, lessons)
quote(() => pct(posFour().x && unexplainedFraction(posFour().x, reconstruct(sae, posFour().active)), 1)); // 1.0%
quote(() => pct(posFour().curve[0], 1)); // 53.5%
quote(() => pct(posFour().curve[7], 1)); // 4.6%
quote(() => pct(scores()[0], 1)); // 0.0%
quote(() => pct(Math.max(...scores()), 1)); // 22.3%
quote(() => pct(mean(scores()), 1), ["standard", "plain", "cards"]); // 7.0%
quote(() => pct(cache.evaluation.heldout_windows.fraction_of_variance_unexplained, 1)); // 5.1%
quote(() => `${pct(posFour().curve[27], 2)} to ${pct(posFour().curve[28], 2)}`, ["standard", "plain", "cards"]); // 1.00% to 1.03%
quote(() => {
  const last = cache.training[cache.training.length - 1];
  return `${((last.loss - last.reconstruction) * 1e5).toFixed(1)} × 10⁻⁵`;
}); // 9.7 × 10⁻⁵
quote(() => cache.training[cache.training.length - 1].loss.toFixed(4)); // 0.0471
quote(() => cache.training[cache.training.length - 1].step.toLocaleString("en-US")); // 10,000
quote(() => cache.training_windows.toLocaleString("en-US")); // 1,600

// ---- Overlap
const overlap = decoderOverlap(sae);
quote(() => overlap.meanAbs.toFixed(3)); // 0.069
quote(() => overlap.maxAbs.toFixed(3)); // 0.992
quote(() => `#${overlap.strongest!.a}`);
quote(() => `#${overlap.strongest!.b}`);
quote(() => String(pairActivity(fixtureRuns().flatMap((run) => run.features), 108, 499).either)); // 734
quote(() => String(fixtureRuns().flatMap((run) => run.features).length)); // 768

const splitCells = () =>
  fixtureRuns().flatMap((run) =>
    Array.from(run.text, (character, index) => ({ character, index, features: run.features[index] })).filter((cell) => cell.index >= 4),
  );
quote(() => {
  const spaces = splitCells().filter((cell) => cell.character === " ");
  return `${spaces.filter((cell) => cell.features[499] > 0).length} of the ${spaces.length} spaces`;
}); // 101 of the 111 spaces
quote(() => {
  const vowels = splitCells().filter((cell) => "aeiou".includes(cell.character));
  return `${vowels.length} lowercase vowels`;
}); // 181 lowercase vowels
const vowelCount108 = () => splitCells().filter((cell) => "aeiou".includes(cell.character) && cell.features[108] > 0).length;
quote(() => `#108 on ${vowelCount108()}`, ["cards"]); // #108 on 132
quote(() => `\`#108\` on ${vowelCount108()}`, ["standard"]); // `#108` on 132
quote(() => `\`#108\` for ${vowelCount108()}`, ["plain"]); // `#108` for 132

// ---- The cache and window position
quote(() => `${records.filter((entry) => entry.window_start_artefact).length} of its 128`, ["standard", "plain"]); // 21 of its 128
quote(() => `${records.filter((entry) => entry.position_zero_artefact).length} of them`, ["standard", "plain"]); // 18 of them
quote(() => `${cache.position_summary.position_zero_dominated_features.length} are dominated`); // 24 are dominated
quote(() => record(876).maximum.toFixed(1)); // 56.9
quote(() => pct(record(876).fires / cache.position_summary.scanned_tokens, 1)); // 31.6%
quote(() => record(876).examples_in_context![0].activation.toFixed(1)); // 36.3
quote(() => mean(fixtureRuns().map((run) => run.features[1][876])).toFixed(1)); // 13.3
quote(() => mean(fixtureRuns().flatMap((run) => run.features.slice(8).map((row) => row[876]))).toFixed(2)); // 0.64
quote(() => mean(fixtureRuns().map((run) => norm(run.x[0]))).toFixed(1), ["standard", "plain"]); // 32.5
quote(() => mean(fixtureRuns().map((run) => norm(run.x[1]))).toFixed(1), ["standard", "plain"]); // 21.5
quote(() => mean(fixtureRuns().flatMap((run) => run.x.slice(4).map(norm))).toFixed(1), ["standard", "plain"]); // 15.0
quote(() => {
  const result = fixtureScore(876, "", 5, "start");
  return `${pct(result.precision)} precision and ${pct(result.recall)} recall`;
}, ["cards"]); // 43% precision and 40% recall

// ---- Labels
quote(() => {
  const result = probeScore(121, " ", 5, "space");
  return `${pct(result.precision!)} precision and ${pct(result.recall!)} recall`;
}); // 100% precision and 70% recall
quote(() => {
  const result = probeScore(121, " ", 2, "space");
  return `${pct(result.precision!)} and ${pct(result.recall!)}`;
}); // 90% and 90%
quote(() => {
  const a = fixtureScore(121, " ", 5, "space");
  return `${pct(a.precision)} and ${pct(a.recall)}`;
}, ["plain", "cards"]); // 96% and 91%
quote(() => {
  const a = fixtureScore(121, " ", 5, "space");
  return `${pct(a.precision)} precision and ${pct(a.recall)} recall`;
}, ["standard"]); // 96% precision and 91% recall
quote(() => {
  const a = fixtureScore(121, " ", 2, "space");
  return `${pct(a.precision)} and ${pct(a.recall)}`;
}); // 63% and 97%
quote(() => {
  const a = fixtureScore(121, " ", 8, "space");
  return `${pct(a.precision)} and ${pct(a.recall)}`;
}); // 100% and 56%
quote(() => {
  const a = fixtureScore(511, ",", 5, "char");
  return `${a.truePositive} of ${a.truePositive + a.falseNegative}`;
}, ["standard", "plain"]); // 18 of 19
quote(() => String(fixtureScore(511, ",", 5, "char").falsePositive)); // 102
quote(() => `${pct(fixtureScore(511, ",", 5, "char").precision)} precision`, ["standard", "plain"]); // 15% precision
quote(() => {
  const a = fixtureScore(53, " ", 5, "space");
  return `${pct(a.precision)} precision and ${pct(a.recall)} recall at 5.0`;
}); // 0% precision and 0% recall at 5.0
quote(() => {
  const a = fixtureScore(53, "\n", 5, "newline");
  return `${pct(a.precision)} precision and ${pct(a.recall)} recall`;
}); // 41% precision and 62% recall
quote(() => `${pct(probeScore(511, ",", 3, "char").precision!)} precision and ${pct(probeScore(511, ",", 3, "char").recall!)} recall`, ["cards"]); // 6% precision and 100% recall
quote(() => `${pct(fixtureScore(511, ",", 3, "char").precision)} and ${pct(fixtureScore(511, ",", 3, "char").recall)}`, ["cards"]); // 9% and 100%

// ---- Steering at the comma after aye
const position = 43;
const steerPrefix = () => prefixRows(probe.residual, position);
const steerRaw = () => teachingTransformer.rowOf(steerPrefix(), position, 256);
const sweep = (feature: number, character: string, strength: number) =>
  strengthSweep(transformer, steerPrefix(), decoderDirection(sae, feature), tokenId(character), SCALE, [strength])[0];
const probabilities = (feature: number, strength: number) =>
  distributionOf(nextLogits(transformer, steerPrefix(), steeredResidual(steerRaw(), decoderDirection(sae, feature), strength, SCALE))).probabilities;
const baseline = () => distributionOf(nextLogits(transformer, steerPrefix())).probabilities;
quote(() => `${pct(baseline()[tokenId("-")], 1)} to ${pct(probabilities(160, 10)[tokenId("-")], 1)}`); // 1.3% to 43.0%
quote(() => change(sweep(160, "-", 10).feature).replace("+", "")); // 3.53
quote(() => `${change(sweep(160, "-", 10).nullMin)} to ${change(sweep(160, "-", 10).nullMax)}`); // −0.53 to +1.36
quote(() => `${sweep(160, "-", 10).score.toFixed(1)} SDs`, ["cards"]); // 6.9 SDs
quote(() => change(sweep(160, "-", 5).feature)); // +2.09
quote(() => `${pct(baseline()[tokenId("y")], 2)} to ${pct(probabilities(683, 5)[tokenId("y")], 2)}`); // 0.55% to 0.14%
quote(() => change(sweep(683, "y", 5).feature), ["cards"]); // −1.38
quote(() => `${Math.abs(sweep(683, "y", 5).score).toFixed(1)} SDs`, ["cards"]); // 3.8 SDs
quote(() => change(sweep(122, "m", 5).feature)); // +0.39
quote(() => change(sweep(122, "m", 5).nullMax)); // +0.37
quote(() => `${Math.abs(sweep(122, "m", 5).score).toFixed(1)} SDs`, ["cards"]); // 1.6 SDs
quote(() => change(sweep(121, " ", 5).feature), ["cards"]); // −0.10
quote(() => `${((10 * SCALE) / norm(steerRaw()) * 100).toFixed(0)}%`, ["cards"]); // 40%
quote(() => norm(steerRaw()).toFixed(1), ["cards"]); // 90.1

// Rank of a feature's direction among all 1,024 for raising one character at strength 5.
const ranking = (feature: number, character: string) => {
  const id = tokenId(character);
  const before = distributionOf(nextLogits(transformer, steerPrefix())).logProbabilities[id];
  const all = Array.from({ length: 1024 }, (_, candidate) => ({
    candidate,
    value:
      distributionOf(nextLogits(transformer, steerPrefix(), steeredResidual(steerRaw(), decoderDirection(sae, candidate), 5, SCALE)))
        .logProbabilities[id] - before,
  })).sort((a, b) => b.value - a.value);
  return { rank: all.findIndex((entry) => entry.candidate === feature) + 1, all };
};
let hyphenRanking: ReturnType<typeof ranking> | null = null;
let yRanking: ReturnType<typeof ranking> | null = null;
quote(() => change((hyphenRanking ??= ranking(160, "-")).all[1].value), ["standard", "plain", "cards"]); // +1.68
quote(() => `#${(hyphenRanking ??= ranking(160, "-")).all[1].candidate}`, ["cards"]); // #563
quote(() => `${(yRanking ??= ranking(683, "y")).rank.toLocaleString("en-US")}th of 1,024`, ["standard", "plain", "cards"]); // 1,011th of 1,024
quote(() => {
  const [a, b, c] = (yRanking ??= ranking(683, "y")).all.slice(0, 3).map((entry) => `#${entry.candidate}`);
  return `${a}, ${b} and ${c}`;
}, ["cards"]); // #167, #902 and #737

describe("the lessons and cards quote what the shipped assets measure", () => {
  it.each(checks.map((entry, index) => [index, entry] as const))("figure %#", (_, entry) => {
    const figure = entry.figure();
    for (const doc of entry.docs) {
      expect(docs[doc].includes(figure), `"${figure}" is missing from ${doc}`).toBe(true);
    }
  }, 120_000);

  it("checked at least sixty figures", () => {
    expect(checks.length).toBeGreaterThanOrEqual(55);
  });
});
