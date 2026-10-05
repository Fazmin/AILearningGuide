/**
 * Byte-level byte-pair encoding, reimplemented to match the Hugging Face
 * `tokenizers` BPE model that produced `assets/bpe-tokenizer.json`
 * (NFKC normalizer, GPT-2 byte-level pre-tokenizer without a prefix space,
 * `<unk>` for any byte symbol missing from the vocabulary, no byte fallback).
 *
 * Unlike a production encoder this one also records every merge it applies,
 * so the lab can replay the merge sequence one rule at a time.
 */

export interface TokenizerJson {
  added_tokens?: Array<{ id: number; content: string; special?: boolean }>;
  model: {
    type: string;
    unk_token?: string | null;
    vocab: Record<string, number>;
    merges: Array<[string, string] | string>;
  };
}

export interface BpeModel {
  vocab: Map<string, number>;
  idToToken: string[];
  /** Merge rule at each rank; rank 0 was learned first. */
  merges: Array<[string, string]>;
  /** `left \u0000 right` → rank */
  ranks: Map<string, number>;
  unkId: number;
  unkToken: string;
  specialTokens: string[];
  /** Vocabulary entries that exist before any merge: specials plus single byte symbols. */
  baseSize: number;
}

export interface MergeStep {
  rank: number;
  left: string;
  right: string;
  merged: string;
  id: number;
  /** How many adjacent occurrences this rule merged inside the word. */
  count: number;
  /** The word's symbols after this merge. */
  symbols: string[];
}

export interface WordTrace {
  /** The pre-token as normalized text, e.g. " king". */
  text: string;
  /** The pre-token after the byte-to-symbol mapping, e.g. "Ġking". */
  symbolsText: string;
  /** One symbol per byte, before any merge. */
  initial: string[];
  steps: MergeStep[];
  tokens: EncodedToken[];
}

export interface EncodedToken {
  /** The vocabulary string, in byte-level symbols ("Ġthe"). */
  token: string;
  id: number;
  unknown: boolean;
  /** A reserved token such as `<eos>`, matched in the raw text before normalization. */
  special: boolean;
  /** For an unknown token, the byte symbol it replaced. */
  source: string;
  wordIndex: number;
}

export interface Encoding {
  normalized: string;
  words: WordTrace[];
  tokens: EncodedToken[];
  byteCount: number;
}

const PAIR_SEPARATOR = "\u0000";
const pairKey = (left: string, right: string) => `${left}${PAIR_SEPARATOR}${right}`;

/**
 * GPT-2's reversible byte → printable-character table. Printable Latin-1 bytes map to
 * themselves; the rest (space, control bytes, …) are shifted to U+0100 and up, which is
 * why a leading space shows up as `Ġ` (U+0120) and a newline as `Ċ` (U+010A).
 */
export const BYTE_TO_SYMBOL: readonly string[] = (() => {
  const printable: number[] = [];
  for (let byte = 33; byte <= 126; byte += 1) printable.push(byte);
  for (let byte = 161; byte <= 172; byte += 1) printable.push(byte);
  for (let byte = 174; byte <= 255; byte += 1) printable.push(byte);
  const table: string[] = new Array(256);
  printable.forEach((byte) => {
    table[byte] = String.fromCharCode(byte);
  });
  let shift = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (table[byte] === undefined) {
      table[byte] = String.fromCharCode(256 + shift);
      shift += 1;
    }
  }
  return table;
})();

const SYMBOL_TO_BYTE = new Map(BYTE_TO_SYMBOL.map((symbol, byte) => [symbol, byte]));

/** The byte a single byte-level symbol stands for, as `0x30`, or null for a multi-byte token. */
export function symbolHex(symbol: string) {
  const byte = SYMBOL_TO_BYTE.get(symbol);
  return byte === undefined ? null : `0x${byte.toString(16).toUpperCase().padStart(2, "0")}`;
}

/** GPT-2 pre-tokenization pattern, as used by the ByteLevel pre-tokenizer. */
const PRE_TOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export function normalize(text: string) {
  return text.normalize("NFKC").replace(/\n/g, " \n ");
}

export function preTokenize(normalized: string): string[] {
  return normalized.match(PRE_TOKEN) ?? [];
}

const encoder = new TextEncoder();

/** Map a string's UTF-8 bytes to byte-level symbols, one symbol per byte. */
export function toByteSymbols(text: string): string[] {
  return Array.from(encoder.encode(text), (byte) => BYTE_TO_SYMBOL[byte]);
}

export function buildModel(json: TokenizerJson): BpeModel {
  const vocab = new Map<string, number>();
  const idToToken: string[] = [];
  for (const [token, id] of Object.entries(json.model.vocab)) {
    vocab.set(token, id);
    idToToken[id] = token;
  }
  const merges = json.model.merges.map((merge) =>
    typeof merge === "string" ? (merge.split(" ") as [string, string]) : merge,
  );
  const ranks = new Map<string, number>();
  merges.forEach(([left, right], rank) => ranks.set(pairKey(left, right), rank));
  const unkToken = json.model.unk_token ?? "<unk>";
  const specialTokens = (json.added_tokens ?? [])
    .filter((token) => token.special)
    .map((token) => token.content);
  return {
    vocab,
    idToToken,
    merges,
    ranks,
    unkId: vocab.get(unkToken) ?? 0,
    unkToken,
    specialTokens,
    baseSize: vocab.size - merges.length,
  };
}

/**
 * Apply merges to one pre-token. At every step the adjacent pair with the lowest
 * rank merges, left to right, until no permitted pair remains. Only ranks below
 * `mergeLimit` are permitted, which is exactly the tokenizer a shorter training run
 * would have produced.
 */
export function encodeWord(
  model: BpeModel,
  text: string,
  wordIndex = 0,
  mergeLimit = Number.POSITIVE_INFINITY,
): WordTrace {
  const initial = toByteSymbols(text);
  let symbols = [...initial];
  const steps: MergeStep[] = [];

  for (;;) {
    let bestRank = Number.POSITIVE_INFINITY;
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const rank = model.ranks.get(pairKey(symbols[index], symbols[index + 1]));
      if (rank !== undefined && rank < bestRank && rank < mergeLimit) bestRank = rank;
    }
    if (!Number.isFinite(bestRank)) break;
    const [left, right] = model.merges[bestRank];
    const merged = left + right;
    const next: string[] = [];
    let count = 0;
    for (let index = 0; index < symbols.length; index += 1) {
      if (index < symbols.length - 1 && symbols[index] === left && symbols[index + 1] === right) {
        next.push(merged);
        count += 1;
        index += 1;
      } else {
        next.push(symbols[index]);
      }
    }
    symbols = next;
    steps.push({
      rank: bestRank,
      left,
      right,
      merged,
      id: model.vocab.get(merged) ?? model.unkId,
      count,
      symbols: [...symbols],
    });
  }

  const tokens = symbols.map((symbol) => {
    const id = model.vocab.get(symbol);
    return {
      token: id === undefined ? model.unkToken : symbol,
      id: id ?? model.unkId,
      unknown: id === undefined,
      special: false,
      source: symbol,
      wordIndex,
    };
  });

  return { text, symbolsText: initial.join(""), initial, steps, tokens };
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Split raw text around special-token strings. Like the Hugging Face added-vocabulary
 * pass, a literal `<eos>` typed into the text becomes the special token itself.
 */
export function splitSpecial(model: BpeModel, text: string): Array<{ text: string; special: boolean }> {
  if (model.specialTokens.length === 0) return [{ text, special: false }];
  const pattern = new RegExp(`(${model.specialTokens.map(escapeRegExp).join("|")})`, "g");
  return text
    .split(pattern)
    .filter((part) => part.length > 0)
    .map((part) => ({ text: part, special: model.specialTokens.includes(part) }));
}

export function encode(model: BpeModel, text: string, mergeLimit = Number.POSITIVE_INFINITY): Encoding {
  const words: WordTrace[] = [];
  let normalized = "";
  for (const segment of splitSpecial(model, text)) {
    if (segment.special) {
      const id = model.vocab.get(segment.text) ?? model.unkId;
      words.push({
        text: segment.text,
        symbolsText: segment.text,
        initial: [segment.text],
        steps: [],
        tokens: [
          { token: segment.text, id, unknown: false, special: true, source: segment.text, wordIndex: words.length },
        ],
      });
      normalized += segment.text;
      continue;
    }
    const piece = normalize(segment.text);
    normalized += piece;
    for (const word of preTokenize(piece)) words.push(encodeWord(model, word, words.length, mergeLimit));
  }
  return {
    normalized,
    words,
    tokens: words.flatMap((word) => word.tokens),
    byteCount: encoder.encode(normalized).length,
  };
}

const decoder = new TextDecoder("utf-8", { fatal: false });

/**
 * Render a byte-level symbol string as readable text. A leading-space byte becomes
 * `␣` and a newline becomes `↵`, so a learner can see that spacing is part of the
 * token. Bytes that do not form valid UTF-8 on their own are shown as hex.
 */
export function displayToken(symbols: string): string {
  if (/^<[a-z]+>$/.test(symbols)) return symbols;
  const bytes: number[] = [];
  for (const symbol of symbols) {
    const byte = SYMBOL_TO_BYTE.get(symbol);
    if (byte === undefined) return symbols;
    bytes.push(byte);
  }
  const decoded = decoder.decode(new Uint8Array(bytes));
  if (decoded.includes("�")) {
    return bytes.map((byte) => `0x${byte.toString(16).toUpperCase().padStart(2, "0")}`).join(" ");
  }
  return decoded.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
