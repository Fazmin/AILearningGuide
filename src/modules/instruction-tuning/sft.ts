/**
 * Chat templating and loss-masked supervised fine-tuning for the character
 * bigram model. Every example becomes its own sequence; each next-character
 * target carries the role of the segment it belongs to, and the loss mask
 * decides which targets the optimizer is graded on.
 */

import { TINY_RESERVED, TINY_VOCAB, TINY_VOCAB_SIZE, tinyRandom } from "@app/module-sdk";

export const SYSTEM_TEXT = "answer briefly and plainly.";
const V = TINY_VOCAB_SIZE;
const SPACE = 0;

export type Role = "system" | "user" | "assistant";
export type LossMask = "assistant" | "all";

export interface Segment {
  text: string;
  marker: boolean;
  role: Role;
  /** True for the assistant's reply text and for the marker that closes it. */
  graded: boolean;
}

export interface TemplateOptions {
  system: boolean;
  roles: boolean;
  end: boolean;
}

/** Reserved symbols cannot appear in user text, so they only ever mean "marker". */
export const stripReserved = (text: string) =>
  text.replace(new RegExp(`[${TINY_RESERVED}]`, "g"), "");

/**
 * Renders one instruction–response pair. Markers butt against the text, as in
 * real templates. Two plain-text segments that end up adjacent (a marker was
 * switched off between them) are separated by one space so words never fuse.
 */
export function renderExample(
  instruction: string,
  response: string,
  options: TemplateOptions,
): Segment[] {
  const segments: Segment[] = [];
  if (options.system) {
    segments.push({ text: "<system>", marker: true, role: "system", graded: false });
    segments.push({ text: SYSTEM_TEXT, marker: false, role: "system", graded: false });
    if (options.end) segments.push({ text: "<end>", marker: true, role: "system", graded: false });
  }
  if (options.roles) segments.push({ text: "<user>", marker: true, role: "user", graded: false });
  segments.push({ text: stripReserved(instruction), marker: false, role: "user", graded: false });
  if (options.roles && options.end) segments.push({ text: "<end>", marker: true, role: "user", graded: false });
  if (options.roles) segments.push({ text: "<assistant>", marker: true, role: "assistant", graded: false });
  segments.push({ text: stripReserved(response), marker: false, role: "assistant", graded: true });
  if (options.end) segments.push({ text: "<end>", marker: true, role: "assistant", graded: true });

  const joined: Segment[] = [];
  for (const segment of segments) {
    const previous = joined[joined.length - 1];
    if (previous && !previous.marker && !segment.marker) {
      // Formatting, not content: the separator belongs to the turn it closes and is never graded.
      joined.push({ text: " ", marker: false, role: previous.role, graded: false });
    }
    joined.push(segment);
  }
  return joined;
}

/** The plain baseline: instruction, one space, response, and no markers at all. */
export function renderPlain(instruction: string, response: string): Segment[] {
  return [
    { text: stripReserved(instruction), marker: false, role: "user", graded: false },
    { text: " ", marker: false, role: "user", graded: false },
    { text: stripReserved(response), marker: false, role: "assistant", graded: true },
  ];
}

export interface EncodedCharacter {
  id: number;
  character: string;
  role: Role;
  marker: boolean;
  graded: boolean;
}

/**
 * Encodes segments exactly as the SDK encodes text — lowercase, unknown
 * characters become a space, runs of spaces collapse, no leading space — while
 * keeping each character's role and grading flag.
 */
export function encodeSegments(segments: ReadonlyArray<Segment>): EncodedCharacter[] {
  const out: EncodedCharacter[] = [];
  for (const segment of segments) {
    for (const raw of segment.text.toLowerCase()) {
      const index = TINY_VOCAB.indexOf(raw);
      const id = index >= 0 ? index : SPACE;
      if (id === SPACE && (out.length === 0 || out[out.length - 1].id === SPACE)) continue;
      out.push({
        id,
        character: TINY_VOCAB[id],
        role: segment.role,
        marker: segment.marker,
        graded: segment.graded,
      });
    }
  }
  return out;
}

/** Whether predicting character `index` (from character `index − 1`) is in the loss. */
export function inLoss(sequence: ReadonlyArray<EncodedCharacter>, index: number, mask: LossMask) {
  if (index <= 0) return false;
  return mask === "all" ? true : sequence[index].graded;
}

export interface Transition {
  row: number;
  target: number;
}

export function transitionsFor(
  sequences: ReadonlyArray<ReadonlyArray<EncodedCharacter>>,
  mask: LossMask,
): Transition[] {
  const transitions: Transition[] = [];
  for (const sequence of sequences) {
    for (let index = 1; index < sequence.length; index += 1) {
      if (inLoss(sequence, index, mask)) {
        transitions.push({ row: sequence[index - 1].id, target: sequence[index].id });
      }
    }
  }
  return transitions;
}

export interface MaskedRun {
  weights: Float32Array;
  history: { step: number; loss: number }[];
  steps: number;
  gradedTargets: number;
  finalLoss: number;
}

function softmaxRow(weights: Float32Array, row: number, into: Float64Array) {
  const offset = row * V;
  let maximum = -Infinity;
  for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, weights[offset + k]);
  let total = 0;
  for (let k = 0; k < V; k += 1) {
    into[k] = Math.exp(weights[offset + k] - maximum);
    total += into[k];
  }
  for (let k = 0; k < V; k += 1) into[k] /= total;
  return into;
}

/** Mean cross-entropy over a set of transitions, in nats per graded character. */
export function transitionLoss(weights: Float32Array, transitions: ReadonlyArray<Transition>): number {
  if (transitions.length === 0) return Math.log(V);
  const probabilities = new Float64Array(V);
  let total = 0;
  for (const { row, target } of transitions) {
    softmaxRow(weights, row, probabilities);
    total -= Math.log(Math.max(1e-12, probabilities[target]));
  }
  return total / transitions.length;
}

/**
 * Minibatch SGD on softmax cross-entropy over the graded transitions only,
 * from a zeroed table. Masked targets contribute no gradient; the characters
 * are still read as context by whichever transition follows them.
 */
export function trainMasked({
  transitions,
  epochs,
  batchSize = 16,
  learningRate = 0.6,
  seed = 6,
  historyPoints = 56,
}: {
  transitions: ReadonlyArray<Transition>;
  epochs: number;
  batchSize?: number;
  learningRate?: number;
  seed?: number;
  historyPoints?: number;
}): MaskedRun {
  const weights = new Float32Array(V * V);
  const count = transitions.length;
  const batch = Math.max(1, Math.min(batchSize, Math.max(1, count)));
  const perEpoch = count === 0 ? 0 : Math.ceil(count / batch);
  const totalSteps = perEpoch * Math.max(0, epochs);
  const every = Math.max(1, Math.floor(Math.max(1, totalSteps) / historyPoints));
  const random = tinyRandom(seed);
  const order = Array.from({ length: count }, (_, index) => index);
  const probabilities = new Float64Array(V);
  const gradient = new Float64Array(V * V);
  const touched = new Set<number>();
  const history: { step: number; loss: number }[] = [];
  let step = 0;
  let windowLoss = 0;
  let windowSteps = 0;

  for (let epoch = 0; epoch < epochs && count > 0; epoch += 1) {
    for (let index = count - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      [order[index], order[swap]] = [order[swap], order[index]];
    }
    for (let start = 0; start < count; start += batch) {
      const end = Math.min(count, start + batch);
      const inverse = 1 / (end - start);
      let batchLoss = 0;
      touched.clear();
      for (let position = start; position < end; position += 1) {
        const { row, target } = transitions[order[position]];
        softmaxRow(weights, row, probabilities);
        batchLoss -= Math.log(Math.max(1e-12, probabilities[target]));
        for (let k = 0; k < V; k += 1) {
          gradient[row * V + k] += (probabilities[k] - (k === target ? 1 : 0)) * inverse;
        }
        touched.add(row);
      }
      for (const row of touched) {
        for (let k = 0; k < V; k += 1) {
          weights[row * V + k] -= learningRate * gradient[row * V + k];
          gradient[row * V + k] = 0;
        }
      }
      windowLoss += batchLoss * inverse;
      windowSteps += 1;
      step += 1;
      if (step % every === 0 || step === totalSteps) {
        history.push({ step, loss: windowLoss / windowSteps });
        windowLoss = 0;
        windowSteps = 0;
      }
    }
  }

  return {
    weights,
    history,
    steps: step,
    gradedTargets: count,
    finalLoss: transitionLoss(weights, transitions),
  };
}

/** Per-character surprisal −log p(character | previous character), NaN for the first. */
export function surprisals(weights: Float32Array, sequence: ReadonlyArray<EncodedCharacter>): number[] {
  const probabilities = new Float64Array(V);
  return sequence.map((entry, index) => {
    if (index === 0) return Number.NaN;
    softmaxRow(weights, sequence[index - 1].id, probabilities);
    return -Math.log(Math.max(1e-12, probabilities[entry.id]));
  });
}

/** Mean log-probability per character of `response` given everything before it. */
export function responseLogProb(
  weights: Float32Array,
  sequence: ReadonlyArray<EncodedCharacter>,
): number {
  const values = surprisals(weights, sequence);
  let total = 0;
  let count = 0;
  sequence.forEach((entry, index) => {
    if (index > 0 && entry.role === "assistant" && entry.graded && !entry.marker) {
      total -= values[index];
      count += 1;
    }
  });
  return count ? total / count : 0;
}

export function nextDistribution(weights: Float32Array, character: string): number[] {
  const index = TINY_VOCAB.indexOf(character);
  return Array.from(softmaxRow(weights, index >= 0 ? index : SPACE, new Float64Array(V)));
}

/* -------------------------------------------------------------------------- */
/* Serving: what a stack sends before the first reply character                */
/* -------------------------------------------------------------------------- */

export type ServeMode = "template" | "none" | "wrong-role";

export const SERVE_MODES: ReadonlyArray<{ value: ServeMode; label: string }> = [
  { value: "template", label: "Chat template" },
  { value: "none", label: "No template" },
  { value: "wrong-role", label: "Wrong role marker" },
];

export const isServeMode = (value: unknown): value is ServeMode =>
  value === "template" || value === "none" || value === "wrong-role";

/** A reply that appears in no default pair, scored under every serving context. */
export const HELD_OUT_REPLY = "the rain stops before noon.";

/**
 * Everything a serving stack would send before the model writes its first reply
 * character. `template` follows the training switches exactly, including the
 * separator space a template without a closing marker leaves. `none` is the bare
 * instruction and one space. `wrong-role` is the template with the assistant
 * marker replaced by the user marker, so the next turn belongs to the user.
 */
export function renderServed(instruction: string, mode: ServeMode, options: TemplateOptions): Segment[] {
  if (mode === "none") {
    return [
      { text: stripReserved(instruction), marker: false, role: "user", graded: false },
      { text: " ", marker: false, role: "user", graded: false },
    ];
  }
  // Rendering with a stand-in reply and cutting before it keeps the prompt identical to training.
  const full = renderExample(instruction, "x", options);
  const start = full.findIndex((segment) => segment.graded && !segment.marker);
  const prompt = full.slice(0, Math.max(0, start));
  if (mode === "template") return prompt;
  const last = prompt[prompt.length - 1];
  if (last && last.marker && last.text === "<assistant>") {
    return [...prompt.slice(0, -1), { text: "<user>", marker: true, role: "user", graded: false }];
  }
  return [...prompt, { text: "<user>", marker: true, role: "user", graded: false }];
}

export const segmentsText = (segments: ReadonlyArray<Segment>) => segments.map((segment) => segment.text).join("");

export interface ServeProbe {
  /** The string sent, markers spelled out. */
  served: string;
  /** The only character the one-character-context model reads from it. */
  lastCharacter: string;
  /** Probability the model gives the reply's first character. */
  firstReplyProbability: number;
  /** Mean log-probability per character of `heldOutReply` written after the served string. */
  heldOutLogProb: number;
}

/** Scores one trained table against one serving context, using the same machinery as training. */
export function probeServing({
  weights,
  instruction,
  reply,
  heldOutReply = HELD_OUT_REPLY,
  options,
  mode,
}: {
  weights: Float32Array;
  instruction: string;
  reply: string;
  heldOutReply?: string;
  options: TemplateOptions;
  mode: ServeMode;
}): ServeProbe {
  const segments = renderServed(instruction, mode, options);
  const served = segmentsText(segments);
  const encoded = encodeSegments(segments);
  const last = encoded[encoded.length - 1];
  const lastCharacter = last ? last.character : TINY_VOCAB[SPACE];
  const firstReply = TINY_VOCAB.indexOf(reply.trim().slice(0, 1).toLowerCase());
  const distribution = nextDistribution(weights, lastCharacter);
  const held = encodeSegments([
    ...segments,
    { text: stripReserved(heldOutReply), marker: false, role: "assistant", graded: true },
  ]);
  return {
    served,
    lastCharacter,
    firstReplyProbability: firstReply >= 0 ? distribution[firstReply] : 0,
    heldOutLogProb: responseLogProb(weights, held),
  };
}

/** Position of the first reserved marker character in a sampled continuation, or -1 if it never starts one. */
export function firstMarkerAt(sample: string): number {
  return sample.search(new RegExp(`[${TINY_RESERVED}]`));
}

/* -------------------------------------------------------------------------- */
/* Untrusted dataset text                                                       */
/* -------------------------------------------------------------------------- */

/** Training time grows with these, so a link cannot carry more than the editor could. */
export const MAX_DATASET_LINES = 16;
export const MAX_DATASET_LINE_LENGTH = 200;

export function sanitizeDataset(text: string): string {
  return text
    .split("\n")
    .slice(0, MAX_DATASET_LINES)
    .map((line) => line.slice(0, MAX_DATASET_LINE_LENGTH))
    .join("\n");
}
