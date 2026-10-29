import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `selected` is a passage id from the shared corpus
 * ("hours" is Dock hours, "ferry" Ferry timetable, "pets" Pets on board), and an empty string selects
 * the top-ranked passage. Most scenes load a preset's exact query, so Preset query lights up to match;
 * the encoder scene types Dock hours' own text into the Query box instead.
 */
const PIER = "When is the pier shut on Sunday?";
const DOG = "Can I bring my dog on the boat?";
const PASSWORD = "What is the harbor master password?";

const scenes: LabScene[] = [
  {
    id: "one-passage-encoded",
    sections: ["Words become points on eight topics", "The encoder on this page"],
    title: "One passage through the encoder",
    look: "The chips under Query mark each word known or ignored, closes is read as close, and the eight bars average only the known words.",
    step: 0,
    state: { query: "The north dock closes at 18:00 on weekdays and at 16:00 on Sundays." },
    focus: "Query and vectors",
  },
  {
    id: "dot-product",
    sections: ["Score one: the dot product", "The two scores"],
    title: "The dot product, row by row",
    look: "Each q × d cell multiplies the q and d beside it, only the time and place rows light up, and the total row adds them.",
    step: 1,
    state: { query: PIER, selected: "hours" },
    focus: "Worked similarity",
  },
  {
    id: "ferry-angle",
    sections: ["Score two: cosine similarity"],
    title: "A wider angle, a lower cosine",
    look: "Ferry timetable is selected, so read its θ and cos θ beside the picture, then compare its dot on the arc with Dock hours’.",
    step: 1,
    state: { query: PIER, selected: "ferry" },
    focus: "Angle to the query",
  },
  {
    id: "cells",
    sections: ["The shortcut: an approximate index"],
    title: "Cells and centroids",
    look: "Each box is one cell with its centroid’s lean under its name, and only the cell marked probed has its passages scored.",
    step: 4,
    state: { query: PASSWORD },
    focus: "Exact versus approximate",
  },
  {
    id: "recall-miss",
    sections: ["Measuring what the shortcut missed: Recall@k"],
    title: "One probe misses Fishing",
    look: "Fishing sits in a skipped cell and is tagged top-k · missed, so Recall@3 counts only the passages the shortcut found.",
    step: 4,
    state: { query: PASSWORD, cells: 4, probe: 1 },
    focus: "Exact versus approximate",
  },
  {
    id: "same-boxes",
    sections: ["Words differ even when the meaning matches"],
    title: "Different words, same boxes",
    look: "The question and Pets on board share no word, yet both the q and d columns have numbers in the travel and animals rows.",
    step: 2,
    state: { query: DOG, selected: "pets" },
    focus: "Worked similarity",
  },
  {
    id: "read-query",
    sections: ["1. Read the query vector", "Read a query vector"],
    title: "Read the query vector",
    look: "The chips under Query mark which words count, and the eight bars beside them, with |q| underneath, are the query vector.",
    step: 0,
    state: { query: PIER },
    focus: "Query and vectors",
  },
  {
    id: "measure-angle",
    sections: ["2. Measure the angle", "Measure an angle"],
    title: "Dock hours, almost on the arrow",
    look: "Dock hours’ dot sits almost on the query arrow, and its dashed line drops to its shadow on the scale below.",
    step: 1,
    state: { query: PIER, selected: "hours" },
    focus: "Angle to the query",
  },
  {
    id: "paraphrase",
    sections: ["3. Paraphrase versus keywords", "Compare a paraphrase with keywords"],
    title: "A match with no shared words",
    look: "Pets on board tops Ranked neighbours while its row reads shared words none, and the note under the list names the keyword winner.",
    step: 2,
    state: { query: DOG },
    focus: "Ranked neighbours",
  },
  {
    id: "cosine-or-dot",
    sections: ["4. Cosine versus dot product", "Switch the score"],
    title: "Cosine or dot product",
    look: "Flip Score to dot product and watch Visitor notice’s dot move out past Lost property’s as the two swap places.",
    step: 3,
    state: { query: PASSWORD },
    focus: "Angle to the query",
  },
  {
    id: "probe-more",
    sections: ["5. Search approximately", "Search approximately"],
    title: "One cell probed, then two",
    look: "Raise Cells probed from 1 to 2 and watch Fishing turn from missed to found while Vectors scored climbs.",
    step: 4,
    state: { query: PASSWORD, cells: 4, probe: 1 },
    focus: "Exact versus approximate",
  },
  {
    id: "your-passage",
    sections: ["More to try"],
    title: "Add your own passage",
    look: "Your notes joins Ranked neighbours ahead of Ferry timetable, though it never mentions the pier.",
    state: { query: PIER, userText: "The ferry café closes at 17:00." },
    focus: "Ranked neighbours",
  },
];

export default scenes;
