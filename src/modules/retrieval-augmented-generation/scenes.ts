import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (North
 * dock, Sunday; 20-word chunks, no overlap, Top-k 2, index built before the June notice, BM25, every
 * Abstain below slider off), so a scene sets only what its section changes. `question` is a question
 * id from rag.ts: hours, ferry, wind, dog or june.
 */
const scenes: LabScene[] = [
  {
    id: "start-to-finish",
    sections: ["One question, start to finish", "What it is"],
    title: "One question, four steps",
    look: "The two retrieved chunks sit in the prompt as [1] and [2], between the instruction and the question, and Reader output below cites [1].",
    step: 0,
    focus: "Assembled prompt",
  },
  {
    id: "bm25-pieces",
    sections: ["Three ways to score a chunk"],
    title: "A BM25 score, piece by piece",
    look: "The small line under each chunk in Retrieval lists what each shared word earned, and those pieces add up to the score beside its title.",
    state: { question: "wind" },
    focus: "Retrieval",
  },
  {
    id: "follow-one-question",
    sections: ["Follow one question"],
    title: "Follow one question",
    look: "The amber words on the handbook ruler are the answer span, and the bar marked ✓ whole answer is the chunk that holds all of it.",
    step: 0,
    focus: "Chunk the documents",
  },
  {
    id: "split-answer",
    sections: ["Cut the answer in half"],
    title: "The answer cut in half",
    look: "The answer span now crosses the line between two bars, no bar earns the ✓, and Answer whole in one chunk reads no — split.",
    step: 1,
    state: { chunkSize: 12 },
    focus: "Chunk the documents",
  },
  {
    id: "overlap",
    sections: ["Overlap the windows"],
    title: "Overlapping windows",
    look: "The bars now overlap, one starts right at north and earns the ✓ again, and Chunks in index and Stored words both climb.",
    step: 2,
    state: { chunkSize: 12, overlap: 7 },
    focus: "Chunk the documents",
  },
  {
    id: "stale-index",
    sections: ["Refresh the index"],
    title: "An index built too early",
    look: "Reader output quotes the old timetable under a stale badge; switch Index built to after notice and watch the quoted time change.",
    step: 3,
    state: { question: "ferry" },
    focus: "Reader output",
  },
  {
    id: "corpus-lacks",
    sections: ["Ask what the corpus lacks"],
    title: "A dog question, a pets answer",
    look: "The handbook’s pets chunk reads no query term — never retrieved, so only a timetable chunk reaches the prompt; then pick Closed for wind now? and read Reader output.",
    step: 4,
    state: { question: "dog" },
    focus: "Retrieval",
  },
  {
    id: "dense-dog",
    sections: ["Switch the retriever"],
    title: "Dense finds the pets chunk",
    look: "With Retriever on Dense the pets chunk arrives as [2] without a shared word, yet Reader output still quotes the ferry sentence.",
    step: 5,
    state: { question: "dog", retriever: "dense" },
    focus: "Retrieval",
  },
  {
    id: "score-table",
    sections: ["Score the pipeline"],
    title: "Five questions, three searches",
    look: "Each cell is one question under one search: read Mean context recall and Answers correct at the bottom, then the wind row.",
    step: 6,
    state: { index: "after", topK: 3 },
    focus: "Score the pipeline",
  },
  {
    id: "threshold",
    sections: ["Set an abstention threshold"],
    title: "Abstain below 2.4",
    look: "Under With the check, wind now reads correctly abstained and Changed in June wrongly abstained; drag Abstain below back toward off and watch the chart.",
    step: 7,
    state: { index: "after", topK: 3, question: "wind", abstainBm25: 2.4 },
    focus: "Grounding check",
  },
  {
    id: "opposite-places",
    sections: ["The searches fail in opposite places"],
    title: "Hybrid at Top-k 2",
    look: "Hybrid (RRF) keeps hits from both searches, but the pets chunk, found by Dense alone, lands just below the cut until Top-k reaches 3.",
    step: 5,
    state: { question: "dog", index: "after", retriever: "hybrid" },
    focus: "Retrieval",
  },
  {
    id: "stale-scores-high",
    sections: ["A score measures a match, not an answer"],
    title: "A stale chunk scores high",
    look: "In Best chunk score, the old timetable’s score for Afternoon ferry sits far above the wind rule’s, yet Reader alone calls its answer a stale time.",
    state: { question: "ferry" },
    focus: "Grounding check",
  },
];

export default scenes;
