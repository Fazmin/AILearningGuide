/**
 * A toy for why writing intermediate results down changes what a fixed amount of
 * computation per forward pass can do.
 *
 * The task is pointer chasing through a lookup table in the prompt: start at a node, follow
 * the table `hops` times, report where you end. Every hop is one dependent lookup, the
 * same "find this key earlier in the prompt and copy what follows it" step an induction head
 * performs, and each hop needs the answer to the one before it.
 *
 * The per-pass limit is IMPOSED here: one forward pass may chain at most `budget` dependent
 * lookups before it has to emit a token. A real model's limit is architectural and implicit
 * (network depth and what training taught it), not a clean integer.
 *
 * - Without a scratchpad the model makes one pass, chains `budget` lookups, and answers with
 *   where it got to, the best a computation that deep can do (the toy's rule, not a measurement
 *   of any real model).
 * - With a scratchpad each pass chains up to `budget` lookups, writes the node it reached as one
 *   token, and the next pass starts from that written token. Each pass emits exactly one token,
 *   so passes and tokens written are the same count.
 *
 * The resume step reads only the written context, never the pass that wrote it, so the final
 * answer follows the trace: change a written token and the answer changes with it.
 */

export const NODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] as const;

/**
 * One cycle through all ten nodes, C → H → A → J → E → B → G → D → I → F → C. A cycle of
 * length 10 means no two different hop counts up to MAX_HOPS can land on the same node, so a
 * wrong answer is never right by coincidence.
 */
export const TABLE: Readonly<Record<string, string>> = {
  A: "J",
  B: "G",
  C: "H",
  D: "I",
  E: "B",
  F: "C",
  G: "D",
  H: "A",
  I: "F",
  J: "E",
};

export const START_NODE = "C";
export const MIN_HOPS = 1;
export const MAX_HOPS = 8;
export const MIN_BUDGET = 1;
export const MAX_BUDGET = 8;

export const hop = (node: string) => TABLE[node] ?? node;

/** The nodes visited from `start`, including `start`, after up to `hops` lookups. */
export function follow(start: string, hops: number) {
  const path = [start];
  for (let step = 0; step < hops; step += 1) path.push(hop(path[path.length - 1]));
  return path;
}

export const correctAnswer = (hops: number) => follow(START_NODE, hops)[hops];

/** The prompt, one line per piece, as the model reads it. */
export function promptLines(hops: number) {
  const table = NODES.map((node) => `${node}→${TABLE[node]}`).join("  ");
  return [`table  ${table}`, `start ${START_NODE}, follow the table ${hops} time${hops === 1 ? "" : "s"}`];
}

export interface Pass {
  /** 1-based. */
  index: number;
  /** The node this pass starts from. */
  from: string;
  /** Where the starting node came from: the prompt, or the last token written to the scratchpad. */
  readFrom: "prompt" | "scratchpad";
  /** The nodes visited by this pass's silent lookups, starting with `from`. Nothing here is written down. */
  lookups: string[];
  /** Dependent lookups completed after this pass, counted from the start. */
  hopsDone: number;
  /** The one token this pass writes. */
  writes: string;
  kind: "scratch" | "answer";
  /** True when the pass ran out of per-pass budget before finishing and answered anyway. */
  outOfBudget: boolean;
}

export interface Run {
  passes: Pass[];
  /** Every token written after the prompt, in order; the last one is the answer. */
  written: string[];
  answer: string;
  target: string;
  correct: boolean;
  /** Forward passes; each writes one token, so this is also the tokens written. */
  tokens: number;
}

/**
 * One forward pass, given only the written context. With a scratchpad on, the written tokens
 * so far are the scratch values, each worth `budget` hops, so the pass counts them to know how
 * many hops are left and resumes from the last one.
 */
export function nextPass(written: ReadonlyArray<string>, hops: number, budget: number, scratchpad: boolean): Pass {
  const resume = scratchpad && written.length > 0;
  const from = resume ? written[written.length - 1] : START_NODE;
  const done = resume ? written.length * budget : 0;
  const remaining = Math.max(0, hops - done);
  const steps = Math.min(budget, remaining);
  const lookups = follow(from, steps);
  const reached = lookups[lookups.length - 1];
  const finished = steps >= remaining;
  const answers = finished || !scratchpad;
  return {
    index: written.length + 1,
    from,
    readFrom: resume ? "scratchpad" : "prompt",
    lookups,
    hopsDone: done + steps,
    writes: reached,
    kind: answers ? "answer" : "scratch",
    outOfBudget: !scratchpad && !finished,
  };
}

/** `tamper` lets a test change what a pass writes before the next pass reads it. */
export function runScratchpad(
  hops: number,
  budget: number,
  scratchpad: boolean,
  tamper?: (pass: Pass) => string,
): Run {
  const written: string[] = [];
  const passes: Pass[] = [];
  // The loop is bounded: with a scratchpad each pass adds `budget` hops, so ceil(hops / budget) passes end it.
  for (let guard = 0; guard <= hops + 1; guard += 1) {
    const pass = nextPass(written, hops, budget, scratchpad);
    passes.push(pass);
    written.push(tamper ? tamper(pass) : pass.writes);
    if (pass.kind === "answer") break;
  }
  const answer = written[written.length - 1];
  const target = correctAnswer(hops);
  return { passes, written, answer, target, correct: answer === target, tokens: written.length };
}

/** Passes the scratchpad needs for `hops` lookups at `budget` per pass. */
export const passesNeeded = (hops: number, budget: number) => Math.ceil(hops / budget);

/** The same task at every hop count, for one budget: what each setting gets right and what it costs. */
export function sweepHops(budget: number) {
  return Array.from({ length: MAX_HOPS }, (_, index) => {
    const hops = index + 1;
    const without = runScratchpad(hops, budget, false);
    const withPad = runScratchpad(hops, budget, true);
    return {
      hops,
      target: without.target,
      without: { answer: without.answer, correct: without.correct },
      with: { answer: withPad.answer, correct: withPad.correct, passes: withPad.tokens },
    };
  });
}
