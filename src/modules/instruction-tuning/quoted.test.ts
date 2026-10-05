import { describe, expect, it } from "vitest";
import { sampleTinyText, TINY_VOCAB, TINY_VOCAB_SIZE, tinyTopTokens, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import definition from "./module";
import {
  encodeSegments,
  HELD_OUT_REPLY,
  inLoss,
  probeServing,
  renderExample,
  renderPlain,
  responseLogProb,
  surprisals,
  trainMasked,
  transitionsFor,
  type LossMask,
  type ServeMode,
} from "./sft";

/**
 * Pins the measured values the instruction-tuning lesson (standard.mdx and plain.mdx), the card text
 * (card-info.ts) quote. Quote locations are given by file and section because the lesson text is still being edited.
 * Mirrors the Explore lab: the module's initial dataset parsed the way Explore
 * parses it, the SDK-default trainer (batch 16, learning rate 0.6, seed 6), one run on the templated
 * sequences and one on plain concatenation, both under the same loss mask.
 */
const DEFAULT_EPOCHS = definition.initialState.epochs as number;
const pairs = (definition.initialState.dataset as string)
  .split("\n")
  .map((line) => line.split("|"))
  .filter((parts) => parts.length >= 2)
  .map((parts) => ({
    instruction: parts[0].trim().toLowerCase(),
    response: parts.slice(1).join("|").trim().toLowerCase(),
  }));

function lab({
  system = true,
  roles = true,
  end = true,
  mask = "assistant",
  epochs = DEFAULT_EPOCHS,
}: { system?: boolean; roles?: boolean; end?: boolean; mask?: LossMask; epochs?: number } = {}) {
  const options = { system, roles, end };
  const templated = pairs.map((pair) => encodeSegments(renderExample(pair.instruction, pair.response, options)));
  const plain = pairs.map((pair) => encodeSegments(renderPlain(pair.instruction, pair.response)));
  const templatedRun = trainMasked({ transitions: transitionsFor(templated, mask), epochs });
  const plainRun = trainMasked({ transitions: transitionsFor(plain, mask), epochs });

  // LossMaskStrip: mean surprisal of graded vs context-only characters of the first example.
  const first = templated[0];
  const values = surprisals(templatedRun.weights, first);
  let gradedTotal = 0;
  let gradedCount = 0;
  let contextTotal = 0;
  let contextCount = 0;
  first.forEach((_, index) => {
    if (index === 0) return;
    if (inLoss(first, index, mask)) {
      gradedTotal += values[index];
      gradedCount += 1;
    } else {
      contextTotal += values[index];
      contextCount += 1;
    }
  });

  // "What follows the end of a turn": probability of "<" after the first reply's last character.
  const boundary = pairs[0].response.slice(-1);
  const marker = (weights: Float32Array) =>
    tinyTopTokens(weights, boundary, TINY_VOCAB_SIZE).find((entry) => entry.token === "<")?.probability ?? 0;

  return {
    templatedRun,
    plainRun,
    predicted: first.length - 1,
    gradedCount,
    gradedMean: gradedTotal / gradedCount,
    contextMean: contextTotal / contextCount,
    templatedReply: responseLogProb(templatedRun.weights, first),
    plainReply: responseLogProb(plainRun.weights, plain[0]),
    templatedMarker: marker(templatedRun.weights),
    plainMarker: marker(plainRun.weights),
  };
}

describe("instruction-tuning quoted values", () => {
  it("uses the default lab state", () => {
    expect(pairs).toHaveLength(4);
    expect(pairs[0].response.slice(-1)).toBe(".");
    expect(DEFAULT_EPOCHS).toBe(30);
    // standard.mdx "The markers are reserved symbols" (30 characters) and "Where it breaks" (900 logits); card-info.ts "Chat template editor" howItWorks (30 x 30 table).
    expect(TINY_VOCAB).toHaveLength(30);
    expect(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE).toBe(900);
    // card-info.ts "Two models, two formats" whatYouSee, "y from 0.60 to about 3.50": the chart's upper bound is ln 30 + 0.1.
    expect(UNIFORM_CROSS_ENTROPY + 0.1).toBeCloseTo(3.5, 2);
  });

  it("trains with the documented recipe", () => {
    // card-info.ts "Chat template editor" howItWorks: "minibatch SGD from a zeroed 30 x 30 table ... batch 16, learning rate 0.6, seed 6".
    const transitions = transitionsFor(
      pairs.map((pair) => encodeSegments(renderExample(pair.instruction, pair.response, { system: true, roles: true, end: true }))),
      "assistant",
    );
    expect(trainMasked({ transitions, epochs: 0 }).weights.every((value) => value === 0)).toBe(true);
    const implicit = trainMasked({ transitions, epochs: 3 });
    const explicit = trainMasked({ transitions, epochs: 3, batchSize: 16, learningRate: 0.6, seed: 6 });
    expect(implicit.weights).toEqual(explicit.weights);
    expect(implicit.finalLoss).toBe(explicit.finalLoss);
  });

  it("matches the strip readouts at the defaults", () => {
    const defaults = lab();
    // standard.mdx "Which characters are graded", plain.mdx "How to play with it", card-info.ts "Chat template editor" controls: "52 of its 137 predicted characters are graded".
    expect(defaults.gradedCount).toBe(52);
    expect(defaults.predicted).toBe(137);
    // standard.mdx "How to play with it" and "What to notice", plain.mdx "What to notice", card-info.ts "Chat template editor" controls: graded 2.11 nats, context only 2.92 nats.
    expect(defaults.gradedMean).toBeCloseTo(2.11, 2);
    expect(defaults.contextMean).toBeCloseTo(2.92, 2);
  });

  it("matches the strip readouts with Every character", () => {
    const everything = lab({ mask: "all" });
    // standard.mdx "How to play with it", plain.mdx "How to play with it", card-info.ts "Chat template editor" controls: all 137 graded, averaging 1.99.
    expect(everything.gradedCount).toBe(137);
    expect(everything.gradedMean).toBeCloseTo(1.99, 2);
  });

  it("matches the end-of-turn probabilities", () => {
    const defaults = lab();
    // standard.mdx and plain.mdx "What to notice", card-info.ts "What follows the end of a turn" controls: templated "<" after the full stop reads 54.4%.
    expect(defaults.templatedMarker).toBeCloseTo(0.544, 3);
    // card-info.ts "Two models, two formats" notice: "54% against 3%".
    expect(Math.round(defaults.templatedMarker * 100)).toBe(54);
    expect(Math.round(defaults.plainMarker * 100)).toBe(3);
    // standard.mdx and plain.mdx "What to notice": the plain model puts 3.3% on every character, the uniform 1/30.
    expect(defaults.plainMarker).toBeCloseTo(0.033, 3);
    expect(defaults.plainMarker).toBeCloseTo(1 / 30, 6);

    // standard.mdx and plain.mdx "What to notice", card-info.ts "What follows the end of a turn" controls: end marker off, mask on, falls to 3.3% (uniform).
    const endOff = lab({ end: false });
    expect(endOff.templatedMarker).toBeCloseTo(0.033, 3);
    expect(endOff.templatedMarker).toBeCloseTo(1 / 30, 6);

    // standard.mdx "What to notice", card-info.ts "What follows the end of a turn" controls: Every character with the end marker off still leaves 55%.
    expect(lab({ mask: "all", end: false }).templatedMarker).toBeCloseTo(0.55, 2);
  });

  it("matches the reply log-prob per character", () => {
    const masked = lab();
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Two models, two formats" controls: -2.228 against -2.168 under the mask.
    expect(masked.templatedReply).toBeCloseTo(-2.228, 3);
    expect(masked.plainReply).toBeCloseTo(-2.168, 3);
    // standard.mdx "What to notice", card-info.ts "Two models, two formats" controls: -2.425 against -2.129 with Every character.
    const everything = lab({ mask: "all" });
    expect(everything.templatedReply).toBeCloseTo(-2.425, 3);
    expect(everything.plainReply).toBeCloseTo(-2.129, 3);
  });

  it("matches the graded-target counts", () => {
    const masked = lab();
    // card-info.ts "Two models, two formats" howItWorks: "191 against 171 targets at the defaults" (templated vs plain).
    expect(masked.templatedRun.gradedTargets).toBe(191);
    expect(masked.plainRun.gradedTargets).toBe(171);
    // card-info.ts "Supervised dataset editor" notice: "191 under the mask, 528 with every character graded".
    expect(lab({ mask: "all" }).templatedRun.gradedTargets).toBe(528);
  });

  it("matches the epoch sweep of the end-of-turn marker", () => {
    // card-info.ts "Supervised dataset editor" controls: "at 4 epochs the < probability after the full stop is 6%; at 30 it is 54%; at 80 it is 88%".
    expect(Math.round(lab({ epochs: 4 }).templatedMarker * 100)).toBe(6);
    expect(Math.round(lab({ epochs: 30 }).templatedMarker * 100)).toBe(54);
    expect(Math.round(lab({ epochs: 80 }).templatedMarker * 100)).toBe(88);
  });
});

/** module.ts checkpoint questions 1-4 make comparisons, not new numbers; these pin the facts their answers rest on. */
describe("instruction-tuning checkpoint claims", () => {
  it("question 1: only the templated model has a trained row after the reply's last character", () => {
    const defaults = lab();
    expect(defaults.templatedMarker).toBeGreaterThan(0.5);
    expect(defaults.plainMarker).toBeCloseTo(1 / TINY_VOCAB_SIZE, 6);
  });

  it("question 2: Every character grades every predicted character, leaving nothing context-only", () => {
    const everything = lab({ mask: "all" });
    expect(everything.gradedCount).toBe(everything.predicted);
    expect(lab().gradedCount).toBeLessThan(lab().predicted);
  });

  it("question 3: with the end marker off and the mask on, the templated row is untrained, as in the plain model", () => {
    expect(lab({ end: false }).templatedMarker).toBeCloseTo(1 / TINY_VOCAB_SIZE, 6);
  });

  it("question 4: the model conditions on one preceding character through a vocabulary-squared table", () => {
    expect(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE).toBe(900);
  });
});


/**
 * The "Serve the trained models" card: the same two trained tables, probed with three serving strings.
 * Quote locations: standard.mdx and plain.mdx "How to play with it" and "Where it breaks", card-info.ts "Serve the trained models".
 */
describe("instruction-tuning serving probes", () => {
  const serve = (mode: ServeMode, { mask = "assistant", end = true }: { mask?: LossMask; end?: boolean } = {}) => {
    const run = lab({ mask, end });
    const options = { system: true, roles: true, end };
    const inputs = { instruction: pairs[0].instruction, reply: pairs[0].response, options, mode };
    return {
      templated: probeServing({ weights: run.templatedRun.weights, ...inputs }),
      plain: probeServing({ weights: run.plainRun.weights, ...inputs }),
      weights: run.templatedRun.weights,
    };
  };

  it("gives the first reply character 18% from the templated model and 1/30 from the plain model when serving the chat template", () => {
    const template = serve("template");
    // card-info.ts "Serve the trained models" controls, standard.mdx and plain.mdx "How to play with it": 18% against 3%.
    expect(template.templated.firstReplyProbability).toBeCloseTo(0.18, 2);
    expect(Math.round(template.templated.firstReplyProbability * 100)).toBe(18);
    expect(template.plain.firstReplyProbability).toBeCloseTo(1 / TINY_VOCAB_SIZE, 6);
    expect(Math.round(template.plain.firstReplyProbability * 100)).toBe(3);
  });

  it("serves the same distribution for the chat template and the wrong role marker", () => {
    const template = serve("template");
    const wrong = serve("wrong-role");
    expect(wrong.templated.firstReplyProbability).toBe(template.templated.firstReplyProbability);
    expect(wrong.plain.firstReplyProbability).toBe(template.plain.firstReplyProbability);
    expect(wrong.templated.heldOutLogProb).toBe(template.templated.heldOutLogProb);
    expect(wrong.templated.served).not.toBe(template.templated.served);
  });

  it("barely hurts the templated model, and helps the plain one, to drop the template", () => {
    const none = serve("none");
    // standard.mdx and plain.mdx "Where it breaks", card-info.ts "Serve the trained models" controls: 22% against 25% with no template.
    expect(Math.round(none.templated.firstReplyProbability * 100)).toBe(22);
    expect(Math.round(none.plain.firstReplyProbability * 100)).toBe(25);
    expect(none.templated.firstReplyProbability).toBeGreaterThan(serve("template").templated.firstReplyProbability);
  });

  it("scores a held-out reply worse than the trained one", () => {
    const template = serve("template");
    // card-info.ts "Serve the trained models" notice: held-out reply -2.86 per character against -2.23 for the first trained reply.
    expect(template.templated.heldOutLogProb).toBeCloseTo(-2.86, 2);
    expect(template.plain.heldOutLogProb).toBeCloseTo(-2.87, 2);
    expect(template.templated.heldOutLogProb).toBeLessThan(lab().templatedReply);
    expect(lab().templatedReply).toBeCloseTo(-2.228, 3);
    // The held-out reply is not one of the default pairs.
    expect(pairs.map((pair) => pair.response)).not.toContain(HELD_OUT_REPLY);
  });

  it("puts 37% on a marker after the closing bracket when the prompt was graded too", () => {
    // card-info.ts "Serve the trained models" notice: with Every character, the chat template serve puts 37% on "<".
    const graded = serve("template", { mask: "all" });
    const next = tinyTopTokens(graded.weights, graded.templated.served, TINY_VOCAB_SIZE);
    expect(next[0].token).toBe("<");
    expect(next[0].probability).toBeCloseTo(0.37, 2);
    // standard.mdx and plain.mdx "How to play with it": the card's sample (length 90, temperature 0.6, seed 17) opens with a marker.
    const sample = sampleTinyText(graded.weights, {
      prompt: graded.templated.served,
      length: 90,
      temperature: 0.6,
      seed: 17,
      allowReserved: true,
    });
    expect(sample.startsWith("<")).toBe(true);
    // Under the mask the same row points at reply letters instead.
    const masked = tinyTopTokens(serve("template").weights, "<assistant>", TINY_VOCAB_SIZE);
    expect(masked[0].token).not.toBe("<");
  });

  it("checkpoint question 5: the plain model has no trained row for a closing bracket", () => {
    const plain = lab().plainRun.weights;
    const next = tinyTopTokens(plain, "<assistant>", TINY_VOCAB_SIZE);
    next.forEach((entry) => expect(entry.probability).toBeCloseTo(1 / TINY_VOCAB_SIZE, 6));
  });
});
