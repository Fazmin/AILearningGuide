/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { encodeText, ITOS, showToken, tokenId, UNKNOWN } from "./vocabulary";

const vocabulary = JSON.parse(
  readFileSync(resolve(process.cwd(), "src/modules/attention/assets/transformer-vocab.json"), "utf8"),
) as { stoi: Record<string, number> };

describe("the character vocabulary", () => {
  it("mirrors the transformer's vocabulary file, with <unk> after the 65 characters", () => {
    expect(ITOS).toHaveLength(66);
    expect(UNKNOWN).toBe(65);
    for (const [character, index] of Object.entries(vocabulary.stoi)) expect(ITOS[index], character).toBe(character);
    expect(Object.keys(vocabulary.stoi)).toHaveLength(65);
  });

  it("encodes text, mapping any other character to <unk>", () => {
    expect(encodeText("a b\n")).toEqual([vocabulary.stoi.a, 1, vocabulary.stoi.b, 0]);
    expect(encodeText("é")).toEqual([UNKNOWN]);
    expect(tokenId("-")).toBe(7);
  });

  it("shows spaces and line breaks in a readable way", () => {
    expect(showToken(1)).toBe("␣");
    expect(showToken(0)).toBe("↵");
    expect(showToken(tokenId("y"))).toBe("y");
  });
});
