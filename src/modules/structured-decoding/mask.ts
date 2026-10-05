/**
 * A real, tiny constrained decoder.
 *
 * - Grammars are small patterns compiled into a character-level matcher (a nondeterministic
 *   automaton simulated as a set of states).
 * - A token is legal when every one of its characters can be consumed from the current state,
 *   so a token may cross grammar-symbol boundaries (`{"`, `":`).
 * - The "model" is a table of authored logits per prefix. Softmax, masking and renormalization
 *   are computed exactly.
 */

export type GrammarId = "json" | "call";
export const GRAMMAR_IDS = ["json", "call"] as const;

export const EOS = "⟨eos⟩";

export const VOCAB = [
  '{"',
  "{",
  '"',
  "ok",
  '":',
  ":",
  "true",
  "false",
  "null",
  "}",
  '"}',
  "Sure",
  "The",
  "get",
  "_weather",
  "send",
  "(city",
  "(",
  '="',
  "Os",
  "lo",
  "hello",
  '")',
  EOS,
] as const;

export type Token = (typeof VOCAB)[number];

type Part =
  | { kind: "literal"; text: string }
  | { kind: "choice"; options: readonly string[] }
  | { kind: "letters" };

export interface GrammarSpec {
  id: GrammarId;
  label: string;
  prompt: string;
  pattern: string;
  source: string;
  parts: readonly Part[];
  /** The greedy constrained path, one token per decode step. */
  walk: readonly Token[];
  /** Tokens that straddle two grammar symbols, and the symbols they join. */
  straddles: Partial<Record<Token, string>>;
  /** Authored logits for the salient tokens at each step; every other token gets BASELINE. */
  logits: ReadonlyArray<Partial<Record<Token, number>>>;
}

export const BASELINE_LOGIT = -2;

export const GRAMMARS: Record<GrammarId, GrammarSpec> = {
  json: {
    id: "json",
    label: "JSON object",
    prompt: "Is the service up? Answer as JSON.",
    pattern: '{"ok":true} or {"ok":false}',
    source: 'JSON Schema: an object whose only key is a required boolean "ok"; whitespace omitted to keep the toy small',
    parts: [
      { kind: "literal", text: '{"ok":' },
      { kind: "choice", options: ["true", "false"] },
      { kind: "literal", text: "}" },
    ],
    walk: ['{"', "ok", '":', "true", "}", EOS],
    straddles: { '{"': '{ and "ok"', '":': '"ok" and :', '"}': '"ok" and }' },
    logits: [
      { Sure: 3.0, The: 2.2, '{"': 2.4, "{": 1.5, '"': 0.2 },
      { ok: 2.6, '"}': 0.6, The: 0.8, hello: 0.4, '"': 0.3 },
      { '":': 3.0, '"': 1.4, '"}': 1.2, ":": 0.6 },
      { true: 2.2, '"': 1.9, false: 0.9, null: 0.7, Sure: 0.2 },
      { "}": 2.5, [EOS]: 1.6, '"}': 0.4, The: 0.2 },
      { [EOS]: 1.8, The: 2.1, Sure: 1.1 },
    ],
  },
  call: {
    id: "call",
    label: "Function call",
    prompt: "What is the weather in Oslo? Call a tool.",
    pattern: 'get_weather(city="…")',
    source: "A Python-style tool call; the city argument is a free string of one or more letters",
    parts: [
      { kind: "literal", text: 'get_weather(city="' },
      { kind: "letters" },
      { kind: "literal", text: '")' },
    ],
    walk: ["get", "_weather", "(city", '="', "Os", "lo", '")', EOS],
    straddles: { "(city": "( and city", '="': '= and "', '")': '" and )' },
    logits: [
      { Sure: 2.7, get: 2.2, send: 1.5, "{": 1.0, The: 0.8 },
      { _weather: 3.2, "(": 0.4 },
      { "(city": 2.4, "(": 1.7, '"': 0.2 },
      { '="': 3.0, ":": 0.7 },
      { Os: 2.6, '")': 0.9, hello: 0.7, The: 0.3 },
      { lo: 3.2, '")': 0.6, '"': 0.3 },
      { '")': 3.0, '"': 1.1, lo: 0.3 },
      { [EOS]: 2.5, Sure: 1.4, The: 0.9 },
    ],
  },
};

/* ---------- character-level matcher ---------- */

interface MatchState {
  part: number;
  /** For a choice: the index of the option being followed, or -1 before the first character. */
  option: number;
  offset: number;
}

const LETTER = /^[A-Za-z]$/;

function closure(parts: readonly Part[], states: MatchState[]): MatchState[] {
  const out: MatchState[] = [];
  const seen = new Set<string>();
  const push = (state: MatchState) => {
    const key = `${state.part}:${state.option}:${state.offset}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(state);
    const part = parts[state.part];
    if (!part) return;
    const next = { part: state.part + 1, option: -1, offset: 0 };
    if (part.kind === "literal" && state.offset === part.text.length) push(next);
    if (part.kind === "choice" && state.option >= 0 && state.offset === part.options[state.option].length) push(next);
    if (part.kind === "letters" && state.offset >= 1) push(next);
  };
  states.forEach(push);
  return out;
}

function step(parts: readonly Part[], states: MatchState[], character: string): MatchState[] {
  const next: MatchState[] = [];
  for (const state of states) {
    const part = parts[state.part];
    if (!part) continue;
    if (part.kind === "literal") {
      if (part.text[state.offset] === character) next.push({ ...state, offset: state.offset + 1 });
    } else if (part.kind === "choice") {
      part.options.forEach((option, index) => {
        if ((state.option === -1 || state.option === index) && option[state.offset] === character) {
          next.push({ part: state.part, option: index, offset: state.offset + 1 });
        }
      });
    } else if (LETTER.test(character)) {
      next.push({ ...state, offset: state.offset + 1 });
    }
  }
  return closure(parts, next);
}

export function matchPrefix(grammar: GrammarSpec, text: string): MatchState[] {
  let states = closure(grammar.parts, [{ part: 0, option: -1, offset: 0 }]);
  for (const character of text) {
    states = step(grammar.parts, states, character);
    if (states.length === 0) break;
  }
  return states;
}

export function isComplete(grammar: GrammarSpec, text: string) {
  return matchPrefix(grammar, text).some((state) => state.part === grammar.parts.length);
}

/** Legal iff the text after this token is still a prefix of some string in the grammar. */
export function isLegal(grammar: GrammarSpec, prefix: string, token: Token) {
  if (token === EOS) return isComplete(grammar, prefix);
  return matchPrefix(grammar, prefix + token).length > 0;
}

/** Plain-language description of what the matcher accepts next. */
export function expectation(grammar: GrammarSpec, prefix: string) {
  const states = matchPrefix(grammar, prefix);
  const options = new Set<string>();
  for (const state of states) {
    const part = grammar.parts[state.part];
    if (!part) options.add("end of output");
    else if (part.kind === "literal") {
      if (state.offset < part.text.length) options.add(part.text.slice(state.offset));
    } else if (part.kind === "choice") {
      part.options
        .filter((option, index) => state.option === -1 || state.option === index)
        .filter((option) => option.length > state.offset)
        .forEach((option) => options.add(option.slice(state.offset)));
    } else options.add(state.offset === 0 ? "one or more letters" : "more letters");
  }
  return [...options].join(" or ");
}

/* ---------- distributions ---------- */

export function softmax(logits: readonly number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

export interface TokenRow {
  token: Token;
  logit: number;
  p: number;
  legal: boolean;
  masked: number;
}

export function prefixAt(grammar: GrammarSpec, stepIndex: number) {
  return grammar.walk.slice(0, stepIndex).join("");
}

export function distributionAt(grammar: GrammarSpec, stepIndex: number) {
  const prefix = prefixAt(grammar, stepIndex);
  const authored = grammar.logits[stepIndex] ?? {};
  const logits = VOCAB.map((token) => authored[token] ?? BASELINE_LOGIT);
  const p = softmax(logits);
  const legal = VOCAB.map((token) => isLegal(grammar, prefix, token));
  const z = p.reduce((total, value, index) => total + (legal[index] ? value : 0), 0);
  const rows: TokenRow[] = VOCAB.map((token, index) => ({
    token,
    logit: logits[index],
    p: p[index],
    legal: legal[index],
    masked: legal[index] ? p[index] / z : 0,
  }));
  const rawTop = rows.reduce((best, row) => (row.p > best.p ? row : best));
  const greedy = rows.reduce((best, row) => (row.masked > best.masked ? row : best));
  return { prefix, rows, z, rawTop, greedy, legalCount: legal.filter(Boolean).length };
}

/* ---------- per-step masking versus conditioning on the whole string ---------- */

/**
 * A two-token enum field, `"Oslo"` or `"Bergen"`. The first token is split between `"O`, `"B`
 * and an illegal `"P`. After `"O` the model mostly wants `ttawa"`, which the enum forbids.
 */
export const DETOUR = {
  first: { O: 0.55, B: 0.3, P: 0.15 },
  afterB: { legal: 0.9 },
};

export function distortion(legalAfterO: number) {
  const { O, B } = DETOUR.first;
  const legalAfterB = DETOUR.afterB.legal;
  // Per-step masking renormalizes the first token over the legal ones, then renormalizes again
  // inside each branch, where exactly one continuation is legal.
  const stepwiseOslo = O / (O + B);
  const jointOslo = O * legalAfterO;
  const jointBergen = B * legalAfterB;
  const conditionedOslo = jointOslo / (jointOslo + jointBergen);
  return {
    stepwise: { oslo: stepwiseOslo, bergen: 1 - stepwiseOslo },
    conditioned: { oslo: conditionedOslo, bergen: 1 - conditionedOslo },
    joint: { oslo: jointOslo, bergen: jointBergen, valid: jointOslo + jointBergen },
  };
}
