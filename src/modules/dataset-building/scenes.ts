import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults, where
 * Strip boilerplate and Drop exact duplicates are on and everything else is off, so the readouts
 * show the lesson's own numbers unless a scene sets the switches. `minLength` is in characters
 * (0 is off; 12 catches only “ok.” and “yes.”), and `context` is the probe's Context length.
 */
const scenes: LabScene[] = [
  {
    id: "every-junk-tagged",
    sections: ["Four kinds of junk, and one unfair copy", "What it is"],
    title: "Every kind of junk, tagged",
    look: "With every filter but Decontaminate on, each kind of junk wears its own tag in the list: boilerplate, too short, exact repeat and near-duplicate.",
    step: 1,
    state: { dropNear: true, minLength: 12 },
    focus: "Source documents and cleaning verdicts",
  },
  {
    id: "minhash-note",
    sections: ["A shortcut for billions of pages"],
    title: "Exact overlap and its estimate",
    look: "The note at the bottom of Cleaning stages puts the exact Jaccard for the harbour and harbor lines beside the MinHash estimate.",
    step: 1,
    focus: "Cleaning stages",
  },
  {
    id: "keeps-score",
    sections: ["How the lab keeps score"],
    title: "Two runs, one test",
    look: "The dashed raw line and the solid cleaned line are scored on the same held-out text, and the chart’s key shows how many training pairs each had.",
    step: 3,
    focus: "Validation loss for both datasets",
  },
  {
    id: "read-first",
    sections: ["Read before you filter"],
    title: "Read the scrape first",
    look: "With Show on All lines, read every verdict tag, and notice that the harbour line, “ok.” and “yes.” are still kept.",
    step: 0,
    focus: "Source documents and cleaning verdicts",
  },
  {
    id: "walk-switches",
    sections: ["Walk through the switches", "How to play with it"],
    title: "Every switch off",
    look: "With every switch off the two chart lines sit on top of each other, so turn the switches on in order and note Cleaned held-out at each one.",
    step: 1,
    state: { dropBoilerplate: false, dropExact: false },
    focus: "Cleaning stages",
  },
  {
    id: "leak-sentence",
    sections: ["Leak a sentence, then clean it out"],
    title: "One test sentence leaked",
    look: "With the sentence leaked, Cleaned held-out drops and the contamination check is flagged, then turn on Decontaminate and watch both go back.",
    step: 4,
    state: { leak: true },
    focus: "Validation loss for both datasets",
  },
  {
    id: "probe-recitation",
    sections: ["Probe for recitation"],
    title: "One character of context",
    look: "Step Context length up from 1 to 8 one notch at a time and compare Copied verbatim for the two models.",
    step: 5,
    state: { context: 1 },
    focus: "Regurgitation probe",
  },
  {
    id: "filters-overlap",
    sections: ["The filters overlap", "What to notice"],
    title: "Boilerplate off, dedup on",
    look: "With Strip boilerplate off, the Drop exact duplicates badge counts many more lines, then turn boilerplate back on and watch the badge shrink.",
    step: 1,
    state: { dropBoilerplate: false },
    focus: "Cleaning stages",
  },
  {
    id: "lines-and-characters",
    sections: ["Lines and characters tell different stories"],
    title: "Lines against characters",
    look: "In the Survival funnel, the Strip boilerplate row takes away nearly half the lines but a much smaller share of the characters.",
    step: 1,
    focus: "Survival funnel",
  },
  {
    id: "diversity",
    sections: ["Diversity is counted per character"],
    title: "Trigrams and repetition",
    look: "Compare Distinct trigrams and Repetition with Raw repetition, then turn Strip boilerplate off and watch Distinct trigrams climb back.",
    step: 2,
    focus: "Dataset statistics",
  },
  {
    id: "recitation-on",
    sections: ["Recitation switches on quickly"],
    title: "Eight characters of context",
    look: "At 8 characters of context, the note under the raw sample says nearly all its contexts have only one possible next character.",
    step: 5,
    state: { context: 8 },
    focus: "Regurgitation probe",
  },
  {
    id: "footers-recited",
    sections: ["Repeated footers get recited more"],
    title: "One copy of each footer",
    look: "With Strip boilerplate off, each footer survives once in the cleaned set, so compare Footer text in the sample for the two models.",
    step: 5,
    state: { context: 8, dropBoilerplate: false },
    focus: "Regurgitation probe",
  },
];

export default scenes;
