import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { AXES, DEFAULT_CODES, ENCODINGS } from "./encode";
import { DEFAULT_PIXELS, HANDLINGS, SENTENCE_IDS, parsePixels } from "./preview";

/**
 * Version 4 adds the Pixels and words card (`pixels`, `sentence`) and the Messy rows card (`messMissing`,
 * `messDuplicate`, `messMislabel`, `missingHandling`). A version 3 payload has none of them, so it keeps its
 * encoding choices and gets the clean table and the default drawing.
 */
const initialState: ModuleState = {
  encoding: "integer",
  redCode: DEFAULT_CODES.red,
  blueCode: DEFAULT_CODES.blue,
  greenCode: DEFAULT_CODES.green,
  selected: 0,
  featureAxis: "rooms",
  threshold: 2.5,
  pixels: DEFAULT_PIXELS,
  sentence: "cat-mat",
  messMissing: false,
  messDuplicate: false,
  messMislabel: false,
  missingHandling: "zero",
};

const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;
const asFlag = (value: unknown) => value === true;

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-31-data-features-representation",
  slug: "data-features-representation",
  title: "Data, features, and representation",
  group: "groundwork",
  order: 3,
  icon: "Table2",
  accent: "#6b5cbf",
  prerequisites: ["module-30-math-you-need"],
  estimatedMinutes: 14,
  steps: [
    "Walk the street",
    "Stamp a door color",
    "Watch the lineup",
    "Move the gate",
    "Compare the two maps",
    "Draw a picture as numbers",
    "Write a sentence as IDs",
    "Spoil the table",
  ],
  stepInstructions: [
    "Click Alder 12 on the street so that listing is the selected house.",
    "Leave How to pack the door on integer stamps, then stamp blue as 5 and read the packing slip.",
    "Switch How to pack the door to one cubby each, then to same-length rulers, and see whether the nearest house to Alder 12 stays the same.",
    "On The gate, switch Stamp the gate reads from rooms to door color and move Gate height t.",
    "On Two maps, compare the later-map neighbor of Alder 12 with the one-hot color neighbor.",
    "On Pixels and words, press the pixels to switch off the left and right arms of the plus (row 2, columns 1 and 3), then read the nine-number vector and which template is now nearest.",
    "On Pixels and words, set Sentence to the one with a bird and read which word became ID 0; then compare the ID gap between cat and dog with the gap between cat and mat.",
    "On Messy rows, switch Missing value to on with Handle missing set to read as 0, then to fill with the mean, and compare the neighbour answer; then switch Duplicate row on and read the leave-one-out accuracy.",
  ],
  stateVersion: 4,
  tagline: "Pack six listings into numbers by hand, watch who stands nearest when the packing changes, see the same threshold gate cut a different yard, then turn a picture and a sentence into numbers and spoil the table.",
  objectives: [
    "Encode a table row as a feature vector by hand",
    "Explain why two encodings of the same rows can disagree about which examples are near each other",
    "Show that a threshold gate reading a different feature cuts the street differently, and that an embedding is a geometry fitted for a loss (authored here, learned in Tokens & embeddings)",
    "Turn a 3×3 picture and a short sentence into numbers, and say how a missing value, a duplicate row, or a mislabelled row changes a table summary or a nearest-neighbour answer",
  ],
  glossary: [
    {
      term: "Feature",
      definition: "One measured or constructed attribute used as an input coordinate. Choosing it is a design decision; the model cannot recover a feature you never wrote down.",
    },
    {
      term: "Representation",
      definition: "The numbers that stand in for a raw example after encoding. Distance, clusters, and later weights all act on this representation, not on the original table text.",
    },
    {
      term: "One-hot encoding",
      definition: "A categorical value turned into a vector with a single 1 in the matching slot and 0s elsewhere. Categories become orthogonal, so the encoding invents equal distance between every pair of labels.",
    },
    {
      term: "Normalization",
      definition: "Rescaling a numeric feature so its range or variance is comparable to others. Without it, a column measured in thousands dominates a column measured in fractions.",
    },
    {
      term: "Feature space",
      definition: "The geometry whose axes are the chosen features. Neighbors in that space are neighbors only under that encoding. A 2-D plot of a longer vector is a projection.",
    },
    {
      term: "Distance",
      definition: "The straight-line (Euclidean) gap between two vectors: square each coordinate difference, add them, and take the square root. Two different one-hot colors, (1, 0, 0) and (0, 1, 0), are √2 apart. Every neighbor ranking on this page is by distance, so it changes whenever the encoding does.",
    },
    {
      term: "Categorical variable",
      definition: "A column whose values are labels rather than amounts. Treating a label as an integer invents an order the original column did not have.",
    },
    {
      term: "Embedding",
      definition: "Coordinates fitted so that items the later loss treats as similar end up nearby. The sketch on this page is authored to show that geometry; it is not trained.",
    },
    {
      term: "Label",
      definition: "The target y a supervised model is asked to match. It is not a feature unless you leak it into the input vector.",
    },
    {
      term: "Pixel vector",
      definition: "A picture turned into numbers by reading its pixels in a fixed order, so a 3×3 picture is a point in nine dimensions. Distance between two pictures is then ordinary distance between two vectors. With each pixel 0 or 1, one flipped pixel adds exactly 1 to the squared distance.",
    },
    {
      term: "Vocabulary and token ID",
      definition: "A vocabulary is a fixed list of known words, and a token ID is a word's position in it. IDs are labels, not amounts: subtracting them invents an order, and a word outside the list collapses into one unknown ID. The eight-word list in this lab is not a real tokenizer; real ones split text into pieces (Tokens & embeddings).",
    },
    {
      term: "Missing value",
      definition: "A cell with no recorded value. A model needs a number, so the row must go or something must fill the gap: reading it as 0, dropping the row, and filling with the mean of the others each move the summary and the neighbours differently.",
    },
    {
      term: "Duplicate row",
      definition: "The same record listed more than once. It double-counts in a summary and, if one copy lands in training and one in a test, lets the test row be answered by looking up its twin, so the score reads too high.",
    },
    {
      term: "Label noise",
      definition: "Labels that are wrong on some rows, such as a flipped price tag. A model fits a wrong label as faithfully as a right one, so a share, a nearest-neighbour answer, or a fitted boundary can move because of one bad tag.",
    },
  ],
  references: [
    {
      authors: "Kevin P. Murphy",
      title: "Probabilistic Machine Learning: An Introduction",
      source: "MIT Press",
      year: 2022,
      url: "https://mitpress.mit.edu/9780262046824/probabilistic-machine-learning/",
      note: "Section 1.5 of this textbook covers the same ground as this lab: pictures stored as grids of pixel numbers, one-hot vectors for the colors red, green, and blue, unknown words replaced by one UNK symbol, and blanks filled with the column mean. The author also posts a free draft online.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, et al.",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online",
      year: 2023,
      url: "https://d2l.ai/chapter_preliminaries/pandas.html",
      note: "Section 2.2 loads a tiny table of houses with a room count, a roof type, and a price. It fills a blank room count with the column mean and turns the roof type into one-hot columns, which is the Messy rows card and the one-cubby packing written as a few lines of code.",
    },
    {
      authors: "scikit-learn developers",
      title: "Preprocessing data",
      source: "scikit-learn User Guide",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/preprocessing.html",
      note: "The official guide to a widely used Python toolkit for this step. It warns that integer codes make a model read categories as ordered, offers one-hot encoding instead, and shows how to rescale columns to a shared range, the same three choices as How to pack the door.",
    },
    {
      authors: "Yoshua Bengio, Aaron Courville, and Pascal Vincent",
      title: "Representation Learning: A Review and New Perspectives",
      source: "IEEE Transactions on Pattern Analysis and Machine Intelligence 35(8), 1798–1828",
      year: 2013,
      url: "https://arxiv.org/abs/1206.5538",
      note: "A review of methods that learn features instead of having people design them. It opens with this lesson's main point, that how well a method works depends on how the data is represented, then surveys ways to learn that representation, the idea behind the later map.",
    },
    {
      authors: "Rico Sennrich, Barry Haddow, and Alexandra Birch",
      title: "Neural Machine Translation of Rare Words with Subword Units",
      source: "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016), 1715–1725",
      year: 2016,
      url: "https://aclanthology.org/P16-1162/",
      note: "The paper behind the subword pieces that real tokenizers use. It starts from the problem in the Sentence card, a fixed vocabulary that cannot handle words it has never seen, and solves it by splitting rare words into smaller known pieces.",
    },
    {
      authors: "scikit-learn developers",
      title: "Common pitfalls and recommended practices",
      source: "scikit-learn User Guide",
      year: 2026,
      url: "https://scikit-learn.org/stable/common_pitfalls.html",
      note: "Explains data leakage: information that would not be there at prediction time slips into building the model and makes the score look too good. It shows why rescaling must learn its numbers from the training rows only, the trap described in Where it breaks.",
    },
    {
      authors: "Sayash Kapoor and Arvind Narayanan",
      title: "Leakage and the reproducibility crisis in machine-learning-based science",
      source: "Patterns 4(9), 100804",
      year: 2023,
      url: "https://arxiv.org/abs/2207.07048",
      note: "A survey finding that data leakage is common in research that uses machine learning and has produced results that do not hold up. Its list of leakage types includes features that give away the answer, preprocessing run on training and test rows together, and duplicates in the data.",
    },
    {
      authors: "Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al.",
      title: "Deduplicating Training Data Makes Language Models Better",
      source: "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022), 8424–8445",
      year: 2022,
      url: "https://aclanthology.org/2022.acl-long.577/",
      note: "Finds many near-duplicate examples in large text datasets, with train-test overlap affecting over 4% of the validation set of standard datasets. It is the Duplicate row problem at scale, and removing the copies gives a more accurate test score.",
    },
    {
      authors: "Curtis G. Northcutt, Anish Athalye, and Jonas Mueller",
      title: "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks",
      source: "NeurIPS 2021 Track on Datasets and Benchmarks",
      year: 2021,
      url: "https://arxiv.org/abs/2103.14749",
      note: "Finds wrong labels in the test sets of 10 widely used image, text, and audio datasets, at least 3.3% on average. It shows that label noise can change which model looks best, much as one flipped tag on Dock 2 changes the vote in this lab.",
    },
    {
      authors: "Timnit Gebru, Jamie Morgenstern, Briana Vecchione, et al.",
      title: "Datasheets for Datasets",
      source: "Communications of the ACM 64(12), 86–92",
      year: 2021,
      url: "https://arxiv.org/abs/1803.09010",
      note: "Proposes a standard list of questions to answer about any dataset. Among them: is there a label for each example, is any information missing, and are there errors, noise, or repeated entries, the same checks the Messy rows card walks through.",
    },
  ],
  checkpoint: [
    {
      prompt: "You set How to pack the door to one cubby each. What does Alder 12's packing slip hold?",
      options: [
        "Three numbers: its rooms, the integer code stamped on its door color, and its park bit",
        "Six numbers: the five features plus a slot that carries its high-price tag",
        "Five numbers: its rooms, a 0 or 1 for each of red, blue, and green, and its park bit",
        "Two numbers: its rooms and a single 1 in the slot for its door color",
      ],
      answer: 2,
      explanation: "One cubby each gives every color its own 0/1 slot, so the slip is rooms, red, blue, green, park: five numbers, with Alder 12 reading 2, 1, 0, 0, 1. The three-number slip is the integer packing. The price tag is the label, so it stays off the slip unless you leak it.",
      objective: 0,
    },
    {
      prompt: "With integer stamps, you stamp blue as 5. What happens to Elm 7, Alder 12's nearest house under the default stamps?",
      options: [
        "It stays nearest, because distance depends only on rooms and park",
        "It drops from first place, since the new stamp puts blue far from red",
        "It stays nearest, because the stamp changes only how the door is painted",
        "It moves closer, because a larger stamp means a more similar color",
      ],
      answer: 1,
      explanation: "The slip carries the stamp as a coordinate, so changing blue from 2 to 5 changes the distance from every blue door to every other house. Elm 7 loses first place and two houses tie as nearest. The apartments never moved; the packing did. One cubby each would restore Elm 7, since it never reads the stamps.",
      objective: 1,
    },
    {
      prompt: "Same-length rulers divides rooms by 4 and color stamps by 5. What does that do to the lineup?",
      options: [
        "It adds a new measurement of each house, so distances become more accurate",
        "It makes color the widest axis, because stamps are the largest numbers",
        "It leaves every distance unchanged, because dividing keeps the ordering of houses",
        "It rescales the coordinates, which makes the 0 or 1 park slot the widest axis",
      ],
      answer: 3,
      explanation: "Rescaling is not a new measurement. With the default stamps, rooms then span 0.75 and the color stamps span 0.4 while the 0/1 park slot still spans a full 1, so park dominates most distances and the neighbor order can change. Normalization changes the map, which is why a scaler has to be a deliberate choice.",
      objective: 1,
    },
    {
      prompt: "On The gate you switch Stamp the gate reads from rooms to door color and keep Gate height t. What changes?",
      options: [
        "Nothing changes, because the rule family is the same threshold on a single number",
        "The price tags change, because the gate predicts them from whichever stamp it reads",
        "Different houses reach the high yard, since the rule now reads another feature",
        "The rooms of every house change, because the gate moves the buildings it sorts",
      ],
      answer: 2,
      explanation: "The gate is the rule ŷ = 1[stamp ≥ t]. Changing the stamp it reads changes which houses clear t, so the same rule family draws a different cut through the street. The price tags are the labels and never move; the buildings do not move either.",
      objective: 2,
    },
    {
      prompt: "On Two maps, the later-map neighbor of Alder 12 differs from its one-hot color neighbor. What does the later map show?",
      options: [
        "A trained embedding fitted to the six price tags by minimizing a loss",
        "The first two slip numbers plotted as a plane, as in the lineup callout",
        "Hand-placed coordinates showing what a fitted geometry could look like",
      ],
      answer: 2,
      explanation: "The later map is authored: similar park-and-price listings were placed close together by hand to show the idea of an embedding. Nothing here was trained. Tokens & embeddings fits coordinates against a loss instead of assigning them. The first-two-numbers plane is the projection from the lineup callout, which is a different picture.",
      objective: 2,
    },
    {
      prompt: "On Pixels and words, Sentence is set to the one with a bird, and the word bird comes out as ID 0. What does that show about a fixed vocabulary?",
      options: [
        "Every word outside the list becomes the same unknown ID, so different unseen words look identical",
        "The model learned that bird is unimportant, so it gave the word the smallest number",
        "IDs are given by how common a word is, so a rare word gets a small number",
        "A word outside the list is dropped before IDs are assigned, so the sentence gets shorter",
      ],
      answer: 0,
      explanation: "A vocabulary maps each known word to its position and sends everything else to one shared unknown ID, here 0. Nothing was learned and no frequency was counted: the list is fixed, and the sentence keeps all six positions, with bird occupying one as ID 0. Two different unseen words would therefore be indistinguishable.",
      objective: 3,
    },
    {
      prompt: "In the fixed vocabulary cat is ID 2, dog is ID 6, and mat is ID 5. If a model subtracted IDs as though they were amounts, what would it conclude?",
      options: [
        "That cat is nearer to dog than to mat, because the IDs count how often words appear",
        "That all three words are equally far apart, because every ID is a whole number",
        "That cat is nearer to mat than to dog, an order the numbering invented",
        "That nothing differs from one-hot, because both encodings invent the same distances",
      ],
      answer: 2,
      explanation: "The gap from cat to mat is 3 and from cat to dog is 4, so subtraction ranks mat nearer even though cat and dog are both animals. The numbering made that order up, which is the integer-stamp trap again with words. One-hot would put every pair of different words exactly √2 apart.",
      objective: 3,
    },
    {
      prompt: "Missing value is on with Handle missing set to read as 0, and the new listing's neighbour answer changed from high to low. You switch to fill with the mean. What happens?",
      options: [
        "The answer stays low, because any filled value still throws away the true room count",
        "The answer returns to high, because the filled 1.8 rooms keeps Dock 2 among the three nearest",
        "The answer cannot change, because handling only affects the summary and not the neighbours",
        "The answer returns to high only if the row is dropped, because a filled value always lands far away",
      ],
      answer: 1,
      explanation: "Reading the blank as 0 sends Dock 2 from distance 1.00 to 3.00, so it drops out of the three nearest and the vote flips. Filling with the mean of the other rows (1.8 rooms) puts it at 1.20, still the nearest listing, so the answer is high again. Dropping the row also flips the answer. How a gap is handled is part of the encoding.",
      objective: 3,
    },
    {
      prompt: "Duplicate row is on, and the leave-one-out nearest-neighbour accuracy rises from 50 percent to 71.4 percent. What does that rise measure?",
      options: [
        "Better generalization, because the table now has more examples to learn from",
        "Nothing about the detector: each copy is matched with its twin at distance 0",
        "A better encoding, because duplicated rows pull the neighbours closer together",
        "A fix for the mislabelled row, because two copies outvote one wrong tag",
      ],
      answer: 1,
      explanation: "In leave-one-out, each row is predicted from its nearest other row. A copy's nearest other row is its twin at distance 0, with the same tag, so both copies are counted right for free. The score rose from 3 of 6 to 5 of 7 without any new information. The same leak happens when one copy lands in train and one in test.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      const encoding =
        typeof parsed.encoding === "string" && (ENCODINGS as readonly string[]).includes(parsed.encoding)
          ? parsed.encoding
          : initialState.encoding;
      const featureAxis =
        typeof parsed.featureAxis === "string" && (AXES as readonly string[]).includes(parsed.featureAxis)
          ? parsed.featureAxis
          : initialState.featureAxis;
      return {
        encoding,
        redCode: clampInt(parsed.redCode, 1, 1, 5),
        blueCode: clampInt(parsed.blueCode, 2, 1, 5),
        greenCode: clampInt(parsed.greenCode, 3, 1, 5),
        selected: clampInt(parsed.selected, 0, 0, 5),
        featureAxis,
        threshold: clampNumber(parsed.threshold, 2.5, 0, 5),
        pixels: parsePixels(parsed.pixels),
        sentence: asMember(parsed.sentence, SENTENCE_IDS, initialState.sentence as string),
        messMissing: asFlag(parsed.messMissing),
        messDuplicate: asFlag(parsed.messDuplicate),
        messMislabel: asFlag(parsed.messMislabel),
        missingHandling: asMember(parsed.missingHandling, HANDLINGS, initialState.missingHandling as string),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
