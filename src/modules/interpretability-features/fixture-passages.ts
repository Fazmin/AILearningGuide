/**
 * Twelve 64-character windows of Tiny Shakespeare (public domain) from the held-out tail of
 * the corpus (the last 10%, which neither the transformer nor the SAE trained on), used by
 * the tests to measure every label and statistic the lesson quotes.
 *
 * Eight windows begin at tail offsets 1,000, 10,000, 19,000, ..., spaced 9,000 characters
 * apart, chosen by position alone. Four more were chosen because they contain a double hyphen
 * (offsets 26,104, 37,229, 84,902 and 103,585 of the tail), since hyphens are only 0.2% of
 * the text and the hyphen feature could not otherwise be scored. The sample is small:
 * 768 characters, of which 12 are hyphens, 10 are y, 10 are T, 20 are m and 117 are spaces.
 */
export const PASSAGES: ReadonlyArray<string> = [
  "rina, this I know,\nShe is not for your turn, the more my grief.\n",
  "alt.\n\nKATHARINA:\nGo, fool, and whom thou keep'st command.\n\nPETRU",
  " but the base.\n\nHORTENSIO:\nThe base is right; 'tis the base knav",
  "fit him to our turn,--\nAnd he shall be Vincentio of Pisa;\nAnd ma",
  " and hungerly\nAnd seem'd to ask him sops as he was drinking.\nThi",
  "g:\nThere were none fine but Adam, Ralph, and Gregory;\nThe rest w",
  "ed--\nWhere are those--Sit down, Kate, and welcome.--\nSound, soun",
  "eaty have a present aims;\nIf not, elsewhere they meet with chari",
  " father's care,\nTo have him match'd; and if you please to like\nN",
  "indeed.\n\nBIONDELLO:\nHelp, help, help! here's a madman will murde",
  " dry he was for sway--wi' the King of Naples\nTo give him annual ",
  "d seem to be desert,--\n\nSEBASTIAN:\nHa, ha, ha! So, you're paid.\n",
];
