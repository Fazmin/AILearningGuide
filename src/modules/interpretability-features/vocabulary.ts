/**
 * The character vocabulary of the shipped transformer: the sorted characters of
 * Tiny Shakespeare, then <unk>. It mirrors attention/assets/transformer-vocab.json
 * (a test checks that), kept here so this module does not import another's files.
 */
export const ITOS: ReadonlyArray<string> = [
  "\n", " ", "!", "$", "&", "'", ",", "-", ".", "3", ":", ";", "?",
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."abcdefghijklmnopqrstuvwxyz",
  "<unk>",
];

/** Index of <unk>: any character outside the corpus maps here, and the model never writes it. */
export const UNKNOWN = ITOS.length - 1;

const STOI = new Map(ITOS.map((character, index) => [character, index]));

export function encodeText(text: string) {
  return Array.from(text).map((character) => STOI.get(character) ?? UNKNOWN);
}

export function tokenId(character: string) {
  return STOI.get(character) ?? UNKNOWN;
}

/** A printable form of one token: spaces and line breaks get visible marks. */
export function showToken(id: number) {
  const character = ITOS[id] ?? "<unk>";
  if (character === " ") return "␣";
  if (character === "\n") return "↵";
  return character;
}
