import { describe, expect, it } from "vitest";
import tokenizerJson from "./assets/bpe-tokenizer.json";
import { buildModel, displayToken, encode, encodeWord, preTokenize, symbolHex, toByteSymbols, type TokenizerJson } from "./bpe";
import { DEFAULT_TEXT } from "./state";

const model = buildModel(tokenizerJson as unknown as TokenizerJson);

/**
 * Expected IDs measured with Hugging Face `tokenizers` 0.23.2:
 * `Tokenizer.from_file("bpe-tokenizer.json").encode(text).ids`.
 * 316 such strings matched during the audit; these are a representative subset.
 */
const REFERENCE: Array<[string, number[]]> = [
  ["attention turns context into meaning", [118, 60, 152, 216, 71, 176, 54, 59, 284, 60, 45, 64, 60, 122, 60, 55, 132, 114, 113]],
  ["The king and the queen", [167, 344, 111, 82, 367, 45, 94]],
  ["the The  the", [60, 72, 68, 167, 68, 82]],
  ["2024 was 33 years", [0, 0, 0, 0, 304, 68, 11, 11, 97, 224, 59]],
  ["naïve café 🙂", [54, 41, 0, 0, 108, 92, 41, 46, 0, 0, 68, 0, 0, 0, 0]],
  ["unbelievably", [425, 42, 366, 288, 62, 41, 42, 172]],
  ["Hello, world!\nNew line", [22, 223, 55, 8, 355, 127, 4, 68, 67, 68, 28, 45, 63, 96, 264]],
  ["tokenization", [60, 55, 51, 94, 49, 66, 118, 216]],
  ["don't you'll", [44, 90, 481, 103, 270]],
  ["  leading spaces", [68, 96, 45, 155, 113, 225, 41, 43, 93]],
  ["trailing   ", [60, 173, 232, 113, 68, 68, 68]],
  ["MIXED case TEXT!!! ???", [27, 23, 38, 19, 18, 92, 41, 120, 68, 34, 19, 38, 34, 4, 4, 4, 68, 14, 14, 14]],
  ["tab\there", [60, 41, 42, 0, 72, 79]],
  ["ﬁne ligature", [46, 264, 174, 47, 118, 356]],
  ["Shall I compare thee to a summer's day?", [33, 80, 88, 106, 273, 56, 298, 226, 102, 73, 212, 53, 53, 87, 135, 506, 14]],
  ["O Romeo, Romeo! wherefore art thou Romeo?", [29, 241, 161, 55, 8, 241, 161, 55, 4, 278, 79, 363, 349, 60, 157, 241, 161, 55, 14]],
  ["soft Spends outlawed reaction bird?", [59, 55, 46, 60, 341, 56, 282, 59, 435, 52, 41, 63, 130, 169, 41, 247, 216, 83, 128, 44, 14]],
  ["Draw beds exchanged,", [18, 173, 63, 119, 44, 59, 149, 64, 43, 222, 47, 130, 8]],
  ["hi<eos> there", [187, 3, 317]],
  ["a <pad> b", [41, 68, 1, 83]],
  ["<bos>The king", [2, 167, 344]],
];

describe("byte-level BPE", () => {
  it("has 4 specials, 65 byte symbols, and 443 merges: 512 entries", () => {
    expect(model.vocab.size).toBe(512);
    expect(model.merges.length).toBe(443);
    expect(model.baseSize).toBe(69);
    expect(model.specialTokens).toEqual(["<unk>", "<pad>", "<bos>", "<eos>"]);
  });

  it("numbers merged tokens in merge order, so ID = 69 + rank", () => {
    model.merges.forEach(([left, right], rank) => {
      expect(model.vocab.get(left + right)).toBe(69 + rank);
    });
  });

  it.each(REFERENCE)("matches Hugging Face tokenizers on %j", (text, ids) => {
    expect(encode(model, text).tokens.map((token) => token.id)).toEqual(ids);
  });

  it("gives a repeated token the same ID wherever it sits, so an embedding row cannot carry order", () => {
    // Quoted in both lessons, the workbench card and a checkpoint question: the default sentence has ␣the twice.
    const tokens = encode(model, DEFAULT_TEXT).tokens;
    const the = tokens.map((token, index) => ({ token: token.token, id: token.id, index })).filter((token) => token.token === "Ġthe");
    expect(the.map((token) => token.id)).toEqual([82, 82]);
    expect(the.map((token) => token.index)).toEqual([3, 9]);
    expect(displayToken(the[0].token)).toBe("␣the");
  });

  it("maps a space byte to Ġ and splits words with the GPT-2 pattern", () => {
    expect(toByteSymbols(" a\n")).toEqual(["Ġ", "a", "Ċ"]);
    expect(preTokenize("don't stop  now")).toEqual(["don", "'t", " stop", " ", " now"]);
  });

  it("records merges in strictly increasing rank, ending at the final tokens", () => {
    const trace = encodeWord(model, " queen");
    const ranks = trace.steps.map((step) => step.rank);
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
    expect(new Set(ranks).size).toBe(ranks.length);
    expect(trace.initial).toEqual(["Ġ", "q", "u", "e", "e", "n"]);
    expect(trace.steps.at(-1)?.symbols).toEqual(trace.tokens.map((token) => token.token));
    expect(trace.tokens.map((token) => token.id)).toEqual([367, 45, 94]);
  });

  it("a smaller merge budget is a prefix of the full merge sequence", () => {
    const full = encodeWord(model, "attention");
    for (const limit of [0, 40, 49, 83, 200, 443]) {
      const limited = encodeWord(model, "attention", 0, limit);
      expect(limited.steps).toEqual(full.steps.filter((step) => step.rank < limit));
      for (const token of limited.tokens) expect(token.id).toBeLessThan(69 + limit);
    }
    expect(encodeWord(model, "attention", 0, 0).tokens).toHaveLength(9);
  });

  it("matches a literal special-token string before normalization", () => {
    const encoding = encode(model, "end<eos>");
    expect(encoding.tokens.at(-1)).toMatchObject({ token: "<eos>", id: 3, special: true });
  });

  it("marks bytes missing from the vocabulary as <unk>", () => {
    const encoding = encode(model, "7");
    expect(encoding.tokens).toEqual([
      { token: "<unk>", id: 0, unknown: true, special: false, source: "7", wordIndex: 0 },
    ]);
  });

  it("renders tokens with visible spacing and hex for partial UTF-8", () => {
    expect(displayToken("Ġthe")).toBe("␣the");
    expect(displayToken("Ċ")).toBe("↵");
    expect(displayToken("<eos>")).toBe("<eos>");
    expect(displayToken(toByteSymbols("ï")[0])).toBe("0xC3");
    expect(symbolHex("0")).toBe("0x30");
    expect(symbolHex("Ġthe")).toBeNull();
  });
});
