import { describe, expect, it } from "vitest";
import { TINY_VOCAB } from "@app/module-sdk";
import {
  encodeSegments,
  firstMarkerAt,
  HELD_OUT_REPLY,
  inLoss,
  MAX_DATASET_LINE_LENGTH,
  MAX_DATASET_LINES,
  nextDistribution,
  probeServing,
  renderExample,
  renderPlain,
  renderServed,
  responseLogProb,
  sanitizeDataset,
  segmentsText,
  trainMasked,
  transitionLoss,
  transitionsFor,
} from "./sft";

const all = { system: true, roles: true, end: true };
const text = (instruction: string, response: string, options = all) =>
  encodeSegments(renderExample(instruction, response, options)).map((entry) => entry.character).join("");

describe("chat template rendering", () => {
  it("wraps turns in markers that butt against the text", () => {
    expect(text("hi there", "hello.")).toBe(
      "<system>answer briefly and plainly.<end><user>hi there<end><assistant>hello.<end>",
    );
  });

  it("never fuses words when a marker between them is switched off", () => {
    expect(text("how long", "simmer it.", { system: false, roles: false, end: false })).toBe("how long simmer it.");
    expect(text("how long", "simmer it.", { system: true, roles: false, end: false })).toBe(
      "<system>answer briefly and plainly. how long simmer it.",
    );
  });

  it("strips reserved characters from typed text", () => {
    expect(text("<user>hi", "ok<end>", { system: false, roles: false, end: false })).toBe("userhi okend");
  });
});

describe("loss masking", () => {
  const sequence = encodeSegments(renderExample("hi", "yo.", all));
  const graded = sequence.map((_, index) => inLoss(sequence, index, "assistant"));
  const gradedText = sequence.filter((_, index) => graded[index]).map((entry) => entry.character).join("");

  it("grades only the reply and the marker that closes it", () => {
    expect(gradedText).toBe("yo.<end>");
    expect(transitionsFor([sequence], "assistant")).toHaveLength(8);
    expect(transitionsFor([sequence], "all")).toHaveLength(sequence.length - 1);
  });

  it("conditions the first reply character on the assistant marker", () => {
    const first = transitionsFor([sequence], "assistant")[0];
    expect(TINY_VOCAB[first.row]).toBe(">");
    expect(TINY_VOCAB[first.target]).toBe("y");
  });

  it("treats the plain separator as context, not reply", () => {
    const plain = encodeSegments(renderPlain("hi", "yo."));
    const gradedPlain = plain.filter((_, index) => inLoss(plain, index, "assistant")).map((entry) => entry.character);
    expect(gradedPlain.join("")).toBe("yo.");
  });
});

describe("masked training", () => {
  const pairs = [
    ["when does the fog arrive", "the fog settles over the harbor in the morning."],
    ["how do i start the sauce", "warm the pan and add a spoon of oil."],
  ];
  const sequences = pairs.map(([instruction, response]) => encodeSegments(renderExample(instruction, response, all)));

  it("lowers the graded loss and leaves ungraded rows untouched", () => {
    const transitions = transitionsFor(sequences, "assistant");
    const run = trainMasked({ transitions, epochs: 30 });
    expect(run.finalLoss).toBeLessThan(Math.log(30));
    // No graded target ever follows "y", which appears only in "system text".
    const afterY = nextDistribution(run.weights, "y");
    afterY.forEach((value) => expect(value).toBeCloseTo(1 / 30, 6));
  });

  it("learns that a marker follows the end of a reply", () => {
    const run = trainMasked({ transitions: transitionsFor(sequences, "assistant"), epochs: 30 });
    const afterStop = nextDistribution(run.weights, ".");
    const marker = afterStop[TINY_VOCAB.indexOf("<")];
    expect(marker).toBe(Math.max(...afterStop));
    expect(marker).toBeGreaterThan(0.2);
    expect(responseLogProb(run.weights, sequences[0])).toBeLessThan(0);
    expect(transitionLoss(run.weights, transitionsFor(sequences, "assistant"))).toBeCloseTo(run.finalLoss, 10);
  });
});

describe("serving contexts", () => {
  const templateOf = (mode: "template" | "none" | "wrong-role", options = all) =>
    segmentsText(renderServed("hi there", mode, options));

  it("builds the training prompt exactly, up to the first reply character", () => {
    expect(templateOf("template")).toBe("<system>answer briefly and plainly.<end><user>hi there<end><assistant>");
    // The prompt is whatever renderExample writes before the reply, separator spaces included.
    const trained = segmentsText(renderExample("hi there", "yo.", all));
    expect(trained.startsWith(templateOf("template"))).toBe(true);
    const none = { system: false, roles: false, end: false };
    expect(templateOf("template", none)).toBe("hi there ");
    expect(segmentsText(renderExample("hi there", "yo.", none)).startsWith(templateOf("template", none))).toBe(true);
  });

  it("serves the bare instruction and one space with no template", () => {
    expect(templateOf("none")).toBe("hi there ");
  });

  it("swaps the assistant marker for the user marker, or appends one when roles are off", () => {
    expect(templateOf("wrong-role")).toBe("<system>answer briefly and plainly.<end><user>hi there<end><user>");
    expect(templateOf("wrong-role", { system: false, roles: false, end: false })).toBe("hi there <user>");
  });

  it("strips reserved characters typed into the instruction", () => {
    expect(segmentsText(renderServed("<end>hi", "none", all))).toBe("endhi ");
  });

  it("probes a table through the last character of the served string only", () => {
    const sequences = [encodeSegments(renderExample("hi", "yo.", all))];
    const run = trainMasked({ transitions: transitionsFor(sequences, "assistant"), epochs: 30 });
    const probe = (mode: "template" | "none" | "wrong-role") =>
      probeServing({ weights: run.weights, instruction: "hi", reply: "yo.", options: all, mode });
    // The chat template and the wrong role marker end in the same bracket, so a one-character model cannot tell them apart.
    expect(probe("template").lastCharacter).toBe(">");
    expect(probe("wrong-role")).toEqual({ ...probe("template"), served: probe("wrong-role").served });
    expect(probe("none").lastCharacter).toBe(" ");
    expect(probe("template").firstReplyProbability).toBe(nextDistribution(run.weights, ">")[TINY_VOCAB.indexOf("y")]);
    expect(probe("template").heldOutLogProb).toBeLessThan(0);
  });

  it("finds the first marker character in a sample", () => {
    expect(firstMarkerAt("the fog.<end>")).toBe(8);
    expect(firstMarkerAt("no marker here")).toBe(-1);
    expect(firstMarkerAt(">")).toBe(0);
    expect(HELD_OUT_REPLY.length).toBeGreaterThan(0);
  });
});

describe("untrusted dataset text", () => {
  it("keeps the default dataset exactly", () => {
    const text = "a | b.\nc | d.";
    expect(sanitizeDataset(text)).toBe(text);
  });

  it("caps the number of lines and the length of each", () => {
    const huge = Array.from({ length: 5000 }, () => "x".repeat(5000)).join("\n");
    const cut = sanitizeDataset(huge);
    expect(cut.split("\n")).toHaveLength(MAX_DATASET_LINES);
    expect(Math.max(...cut.split("\n").map((line) => line.length))).toBe(MAX_DATASET_LINE_LENGTH);
    expect(cut.length).toBeLessThanOrEqual(MAX_DATASET_LINES * (MAX_DATASET_LINE_LENGTH + 1));
  });
});
