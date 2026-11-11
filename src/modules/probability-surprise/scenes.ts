import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Each scene starts from the lab's defaults, which
 * are the numbers the lesson quotes, plus the one or two settings its section begins from. Bars are
 * stored as comma lists of whole percents in the order blue, grey, dark, clear, green.
 */
const scenes: LabScene[] = [
  {
    id: "bars-add-to-one",
    sections: ["Bars that add to one"],
    title: "Five bars, one whole",
    look: "Drag the blue bar up and watch the other four bars shrink, so the strip above them stays full.",
    step: 0,
    state: { view: "percent" },
    focus: "Bars that add to one",
  },
  {
    id: "first-draws",
    sections: ["Draw from the bars"],
    title: "Draw from the bars",
    look: "Raise Number of draws one setting at a time and watch Biggest gap, tally to bars shrink.",
    step: 1,
    state: { draws: 0 },
    focus: "Draw from the bars",
  },
  {
    id: "bell-of-scores",
    sections: ["Averages pile up in a bell"],
    title: "A pile of test scores",
    look: "Switch Questions per test and watch the pile get narrower while its centre stays where it is.",
    step: 2,
    state: { testSize: 25 },
    focus: "Averages pile up in a bell",
  },
  {
    id: "one-row-of-counts",
    sections: ["Context changes the bars"],
    title: "One row of the table",
    look: "Switch Prompt and watch the highlighted row of the table turn into the bars under it.",
    step: 3,
    state: { prompt: 0 },
    focus: "Context changes the bars",
  },
  {
    id: "which-prompt",
    sections: ["Read the table backwards"],
    title: "Which prompt said blue?",
    look: "Drag the Prior slider down and watch the first part of the strip under the columns shrink.",
    step: 3,
    state: { clue: 0, prior: 50 },
    focus: "Read the table backwards",
  },
  {
    id: "halvings",
    sections: ["Surprise in bits"],
    title: "Surprise under each bar",
    look: "Drag the blue bar down and watch the hatched bar hanging under it grow by one line.",
    step: 4,
    state: { word: 0, first: 0, second: 1 },
    focus: "Surprise in bits",
  },
  {
    id: "row-of-blocks",
    sections: ["Average surprise"],
    title: "The row of blocks",
    look: "Drag Shape of the bars to each end and watch the dashed average line rise and fall.",
    step: 4,
    state: { shape: 50 },
    focus: "Average surprise",
  },
  {
    id: "know-nothing-model",
    sections: ["Your guess against what happens"],
    title: "A model that knows nothing",
    look: "Drag the hatched bars onto the solid ones and watch Wasted (KL divergence) fall to zero.",
    step: 5,
    state: { guess: "20,20,20,20,20" },
    focus: "Your guess against what happens",
  },
  {
    id: "scores-to-bars",
    sections: ["From scores to bars"],
    title: "Three moves from scores to bars",
    look: "Press Add 1 to every score and check that the bars at the bottom do not move.",
    step: 6,
    state: { scores: "2,1.3,0.6,0.2,-0.3", temperature: 1 },
    focus: "From scores to bars",
  },
  {
    id: "challenge-round-one",
    sections: ["Try the challenge"],
    title: "Forty draws, hidden bars",
    look: "Set your hatched bars from the dotted tally bars, then press Score it and read Bits wasted.",
    step: 7,
    state: { round: 0, answer: "20,20,20,20,20", scored: 0 },
    focus: "Challenge: learn it from samples",
  },
];

export default scenes;
