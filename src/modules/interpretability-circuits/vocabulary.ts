/**
 * The character vocabulary of the shipped transformer: the sorted characters
 * of Tiny Shakespeare plus <unk>. It mirrors attention/assets/transformer-vocab.json
 * (a test checks that), kept here so this module does not import another's files.
 */
export const ITOS = [
  "\n", " ", "!", "$", "&", "'", ",", "-", ".", "3", ":", ";", "?",
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."abcdefghijklmnopqrstuvwxyz",
  "<unk>",
];

export const UNKNOWN = ITOS.length - 1;

const STOI = new Map(ITOS.map((character, index) => [character, index]));

export function encodeText(text: string) {
  return Array.from(text).map((character) => STOI.get(character) ?? UNKNOWN);
}

export function tokenId(character: string) {
  return STOI.get(character) ?? UNKNOWN;
}

/** A printable label for a character token. */
export function showToken(character: string) {
  if (character === " ") return "␣";
  if (character === "\n") return "↵";
  return character;
}
