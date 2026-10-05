import { describe, expect, it } from "vitest";
import { RAW_LINES, runPipeline, scrapeLines, type PipelineOptions } from "./clean";
import {
  CONTEXT_MAX,
  COPY_WINDOW,
  copiedShare,
  footerShare,
  longestCopiedRun,
  normalized,
  recitationReport,
  sampleRecitation,
} from "./recite";

/** Pins every figure the Regurgitation probe, its card text, and the lesson quote. */
const defaults: PipelineOptions = {
  dropBoilerplate: true,
  dropExact: true,
  dropNear: false,
  decontaminate: false,
  leak: false,
  minLength: 0,
};
const rawText = scrapeLines(false)
  .map((entry) => entry.line)
  .join(" ");
const cleanedText = runPipeline(defaults)
  .filter((entry) => entry.verdict === "kept")
  .map((entry) => entry.line)
  .join(" ");

describe("count-based n-gram", () => {
  it("is deterministic in its seed and starts from the training text's own prefix", () => {
    const first = sampleRecitation(cleanedText, 6);
    expect(sampleRecitation(cleanedText, 6)).toEqual(first);
    expect(first.text).toHaveLength(300);
    expect(sampleRecitation(cleanedText, 6, 300, 8).text).not.toBe(first.text);
    // With a context of 8 the first characters it writes continue the training text's own opening.
    const opening = normalized(cleanedText).slice(0, 8);
    expect(normalized(cleanedText).startsWith(opening)).toBe(true);
    expect(normalized(cleanedText).slice(8, 30)).toBe(sampleRecitation(cleanedText, 8).text.slice(0, 22));
  });

  it("counts the raw and cleaned texts as the trainer sees them", () => {
    expect(normalized(rawText)).toHaveLength(899);
    expect(normalized(cleanedText)).toHaveLength(516);
    expect(RAW_LINES).toHaveLength(20);
  });

  it("clamps the context length and never reads outside the text", () => {
    expect(sampleRecitation(cleanedText, 1e9).context).toBe(CONTEXT_MAX);
    expect(sampleRecitation(cleanedText, -5).context).toBe(1);
    expect(sampleRecitation(cleanedText, Number.NaN).context).toBeGreaterThanOrEqual(1);
    expect(sampleRecitation("", 3).text).toHaveLength(300);
    expect(copiedShare("", cleanedText)).toBe(0);
    expect(footerShare("")).toBe(0);
  });

  it("recites nothing at one character of context, where it is a bigram like the rest of the track", () => {
    const bigram = recitationReport(cleanedText, 1);
    expect(bigram.copied).toBe(0);
    expect(bigram.longest).toBe(6);
    expect(bigram.forcedContexts).toBe(4);
    expect(bigram.contexts).toBe(26);
    expect(recitationReport(rawText, 1).longest).toBe(5);
    expect(recitationReport(rawText, 1).copied).toBe(0);
  });

  it("matches the sweep quoted in the lesson", () => {
    const row = (text: string, context: number) => recitationReport(text, context);
    expect(row(rawText, 2).copied).toBe(0);
    expect(row(cleanedText, 2).copied).toBe(0);
    expect(row(rawText, 3).copied).toBeCloseTo(0.517, 3);
    expect(row(cleanedText, 3).copied).toBeCloseTo(0.387, 3);
    expect(row(rawText, 4).copied).toBeCloseTo(0.977, 3);
    expect(row(rawText, 5).copied).toBe(1);
    expect(row(cleanedText, 5).copied).toBeCloseTo(0.98, 2);
    expect(row(cleanedText, 4).copied).toBeCloseTo(0.927, 3);
    expect(row(rawText, 8).copied).toBe(1);
    expect(row(cleanedText, 8).copied).toBe(1);
    expect(row(rawText, 8).longest).toBe(164);
    expect(row(cleanedText, 8).longest).toBe(67);
    expect(row(rawText, 8).forcedContexts).toBe(541);
    expect(row(rawText, 8).contexts).toBe(555);
    expect(row(cleanedText, 8).forcedContexts).toBe(409);
    expect(row(cleanedText, 8).contexts).toBe(418);
  });

  it("shows duplicated footer text surviving into the raw model's output but not the cleaned one's", () => {
    expect(recitationReport(rawText, 3).footer).toBeCloseTo(0.093, 3);
    expect(recitationReport(rawText, 8).footer).toBeCloseTo(0.207, 3);
    for (const context of [1, 2, 3, 4, 8, 12]) expect(recitationReport(cleanedText, context).footer).toBe(0);
    // The footer is 4 copyright lines, 3 subscribe lines, 2 share lines in the scrape.
    expect(rawText.split("copyright").length - 1).toBe(4);
  });

  it("counts a copied character only inside a verbatim window", () => {
    const source = "the quick brown fox jumps over the lazy dog";
    expect(copiedShare("the quick brown fox ", source, 10)).toBe(1);
    expect(copiedShare("the quick brown zebra", source, 10)).toBeLessThan(1);
    expect(copiedShare("zzzzzzzzzzzzzzzzzzzz", source, 10)).toBe(0);
    expect(longestCopiedRun("xxthe quick brownxx", source)).toBe(15);
    expect(COPY_WINDOW).toBe(16);
  });
});
