import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene trains from the default seed, so the cards
 * show the numbers the lesson quotes. Query ids are held-out questions from data.ts: "h02" is “How late
 * can I use the east berth?”, "h23" the typo “lifevests for kids”, "h26" the unseen-word “Can my hound
 * ride the vessel?”. With Nearest wrong at step 40, row 7 of the shown batch is the Sunday training query
 * whose twin takes most of its share.
 */
const scenes: LabScene[] = [
  {
    id: "untrained-map",
    sections: ["What the lab shows", "What the lab lets you do"],
    title: "Before training: one clump",
    look: "Every dot sits in one tight clump near the edge of the circle, and the topic names pile up on top of each other.",
    step: 0,
    state: { step: 0 },
    focus: "Embedding map",
  },
  {
    id: "typo-trigrams",
    sections: ["The encoder", "How text becomes a list of numbers"],
    title: "A typo still finds its notice",
    look: "The typo query ranks Life jackets first after training, while BM25 finds nothing because no whole word matches.",
    state: { query: "h23" },
    focus: "Query inspector",
  },
  {
    id: "mid-training-batch",
    sections: ["The loss: InfoNCE over a batch", "How a practice round is scored"],
    title: "A batch early in training",
    look: "Each row still spreads its share over several candidates, and in most rows the outlined answer cell holds the biggest part.",
    step: 1,
    state: { step: 20 },
    focus: "Batch similarity",
  },
  {
    id: "even-shares",
    sections: ["A worked example of the starting loss"],
    title: "Even shares at step 0",
    look: "At step 0 the cells in each row hold similar shares, and Loss at step 0 sits a little under Starting loss (ln B).",
    step: 1,
    state: { step: 0 },
    focus: "Batch similarity",
  },
  {
    id: "twin-in-batch",
    sections: ["Duplicates and false negatives", "Twins and false negatives"],
    title: "A twin counted as wrong",
    look: "In the Sunday row, the cell tagged duplicate · false negative takes more share than the answer, so the loss pushes a right answer away.",
    step: 4,
    state: { negatives: "nearest", step: 40 },
    focus: "Batch similarity",
  },
  {
    id: "rerank-east",
    sections: ["Reranking", "A second opinion: the reranker"],
    title: "Reranking the east berth question",
    look: "The reranked row puts East dock hours first, where the trained encoder had put Crab pots.",
    step: 5,
    state: { query: "h02", rerank: true },
    focus: "Query inspector",
  },
  {
    id: "read-untrained",
    sections: ["1. Read the untrained space"],
    title: "Read the untrained space",
    look: "The step 0 row of the table sits far below the BM25 row, and the marker on the chart sits at the left edge.",
    step: 0,
    state: { step: 0 },
    focus: "Held-out retrieval",
  },
  {
    id: "big-batch",
    sections: ["2. Train with in-batch negatives"],
    title: "Batch size 32",
    look: "Switch Batch size between 2 and 32 and watch Duplicate-pair cosine under the map fall while the dots stay spread.",
    step: 1,
    state: { batch: 32 },
    focus: "Embedding map",
  },
  {
    id: "low-temperature",
    sections: ["3. Turn the temperature"],
    title: "A small temperature",
    look: "At τ = 0.03 the dots squeeze toward one side of the circle and Mean cosine between passages climbs.",
    step: 2,
    state: { temperature: 0.03 },
    focus: "Embedding map",
  },
  {
    id: "mined-small-batch",
    sections: ["4. Mine hard negatives"],
    title: "Nearest wrong at batch 2",
    look: "Each row now holds an extra candidate the miner picked, and Closest wrong cosine, run average jumps up.",
    step: 3,
    state: { batch: 2, negatives: "nearest" },
    focus: "Batch similarity",
  },
  {
    id: "false-negative",
    sections: ["5. Spot a false negative"],
    title: "Sunday notice pushed down",
    look: "North dock hours sits above both Sunday passages; switch Skip known duplicates on and Sunday notice climbs back to the top.",
    step: 4,
    state: { negatives: "nearest" },
    focus: "Query inspector",
  },
  {
    id: "rerank-on",
    sections: ["6. Rerank the shortlist"],
    title: "Rerank top 5",
    look: "A reranked row joins the table, and Scoring passes per query rises while Recall@1 creeps up.",
    step: 5,
    state: { rerank: true },
    focus: "Held-out retrieval",
  },
  {
    id: "another-seed",
    sections: ["What to notice"],
    title: "The same recipe, seed 12",
    look: "With Seed 12 the Recall@1 line ends far higher than with seed 1, though nothing else changed.",
    state: { seed: 12 },
    focus: "Held-out retrieval",
  },
  {
    id: "unseen-words",
    sections: ["In the real mechanism", "Where the real method is limited"],
    title: "Words training never saw",
    look: "Can my hound ride the vessel? stays far down every row of the table, before training, after it and under BM25.",
    state: { query: "h26" },
    focus: "Query inspector",
  },
  {
    id: "first-four",
    sections: ["Going deeper"],
    title: "Only the first 4 numbers",
    look: "Recall@1, first 4 of 16 numbers reads far below the full Recall@1 in Held-out retrieval.",
    focus: "Query inspector",
  },
];

export default scenes;
