import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Query and vectors": {
    title: "Text in, eight numbers out",
    summary:
      "The encoder reads each word, looks it up in a hand-set lexicon of 8-dimensional word vectors, and averages the known ones into q (mean pooling). Unknown words add nothing. Your passage goes through the same encoder and joins the index.",
    whatYouSee: [
      "Query, a text box, and Preset query with four ready-made questions.",
      "Token chips: solid for words the lexicon knows (with the entry they matched, such as closes → close), dashed and struck through for ignored words.",
      "Eight labelled bars: the query vector q on the dimensions time, place, travel, weather, animals, access, safety and money, plus its length |q|.",
      "Your passage, a textarea whose text is added to the index as Your notes.",
    ],
    howItWorks: [
      "Words are lowercased; clock times such as 18:00 become one token; a trailing s or es is stripped when that finds a lexicon entry.",
      "q = (1/n) Σ wᵢ over the n known words, where wᵢ is that word's authored 8-number vector. With no known words, q is the zero vector.",
    ],
    controls: [
      "Query, Preset query, and Your passage.",
      "Comparison worth running: type “When does the dock shut?” then “When does the dock close?” — shut and close have the same loading, so q barely moves.",
    ],
    notice: [
      "Stop-words such as “is” and “the” are ignored because they are not in the lexicon, not because of a separate rule.",
      "A query made only of unknown words has length 0, and every similarity becomes 0.",
    ],
    limits: [
      "In this lab: the eight dimensions and every word loading are authored by hand. Nothing here was trained, and word order is thrown away.",
      "In general: a trained encoder is a transformer with hundreds of unnamed dimensions, reads word order and subwords, and never has an ignored word — but its similarities can still be wrong in ways you cannot read off a label.",
    ],
  },

  "Angle to the query": {
    title: "Similarity is an angle, rank is a projection",
    summary:
      "The query points along the x-axis. Each passage is drawn at its true angle θ to q. In cosine mode every dot sits on the unit arc, so its x-coordinate is cos θ. In dot-product mode each dot sits at its raw length |d|, so its x-coordinate is |d| cos θ, proportional to q·d.",
    whatYouSee: [
      "A quarter circle with 30° and 60° spokes and the query arrow along the x-axis, ticked in cosine or dot-product units.",
      "Filled dots are the current top-k, hollow dots are the rest; the selected passage is purple with a line down to its projection on the query axis.",
      "Labels on the right with rank, θ and score, and metrics for the selected passage's θ, cos θ and |d|.",
    ],
    howItWorks: [
      "θ = arccos(q·d / (|q| |d|)), in degrees. All loadings are non-negative, so θ runs from 0° to 90°.",
      "Dot mode: radius = |d| / max|d|, so x = |d| cos θ scaled, and the axis ticks convert back to q·d = |q| |d| cos θ.",
    ],
    controls: [
      "Score switches between cosine and dot product. Clicking a dot selects that passage.",
      "Comparison worth running: preset password, then flip Score and watch Visitor notice move outward past Lost property.",
    ],
    notice: [
      "In cosine mode the order along the x-axis is exactly the ranking.",
      "In dot mode a longer vector at a slightly wider angle can reach further right.",
    ],
    limits: [
      "In this lab: only each passage's angle to the query is real. Two dots that look close to each other may be far apart in 8 dimensions.",
      "In general: most retrieval models either output unit-length vectors or are trained for one score, so use the metric the encoder was trained with.",
    ],
  },

  "Ranked neighbours": {
    title: "Exact search: score everything, sort",
    summary:
      "Every stored vector is scored against q with the chosen metric and sorted. The top-k are highlighted. Each row also lists the words it shares with the query and its rank under a plain shared-word count, for comparison with keyword search.",
    whatYouSee: [
      "Top-k from 1 to 5.",
      "Six ranked rows with rank, title and score, plus the other score, the shared content words and the keyword rank.",
      "A note naming the passage keyword ranking would put first.",
    ],
    howItWorks: [
      "Score = cosine(q, d) or q·d over all twelve passages (thirteen with Your notes). Ties keep corpus order.",
      "Keyword rank sorts by the number of distinct shared content words after removing function words and plural s. It is a crude stand-in for BM25.",
    ],
    controls: [
      "Top-k, and a click on any row to select it for the other cards.",
      "Comparison worth running: preset dog on the boat, then read Pets on board's cosine rank against its keyword rank.",
    ],
    notice: [
      "Pets on board shares no words with “Can I bring my dog on the boat?” and is still #1 by cosine, because dog and pet, boat and travel load on the same dimensions.",
      "Keyword ranking puts Mooring fees first for that query, on the single word “boat”.",
    ],
    limits: [
      "In this lab: exact search over thirteen vectors is instant, and the keyword baseline is a shared-word count, not BM25.",
      "In general: the nearest passage is the most similar under this encoder. It may still be stale, false, hostile, or not an answer to the question.",
    ],
  },

  "Worked similarity": {
    title: "q·d and cos θ, term by term",
    summary:
      "For the selected passage, each dimension multiplies q's value by d's value. The products add up to the dot product. Dividing by |q| and |d| gives the cosine, and arccos gives θ. These are the same numbers the other cards draw.",
    whatYouSee: [
      "The selected passage's text.",
      "A table with one row per dimension: q, d and q × d, highlighted where the product is non-zero, and a total row with |q|, |d| and Σ.",
      "Two formula blocks: the dot product with its rank under dot product, and the cosine with θ.",
    ],
    howItWorks: [
      "q·d = Σᵢ qᵢ dᵢ; |v| = √(Σᵢ vᵢ²); cos θ = q·d / (|q| |d|).",
      "Only dimensions where both q and d are non-zero contribute; a topic the query never mentions cannot add to the score.",
    ],
    controls: [
      "This card has no controls. Select a passage in Ranked neighbours or on Angle to the query.",
      "Comparison worth running: password preset, Lost property against Visitor notice — compare |d| in the total row.",
    ],
    notice: [
      "Visitor notice has the larger |d| (0.565 against 0.427), which is why it wins under dot product and loses under cosine.",
      "If every vector were scaled to length 1 first, the two formulas would print the same number.",
    ],
    limits: [
      "In this lab: eight dimensions, so every term fits on screen.",
      "In general: the same sum runs over 384 to several thousand dimensions, usually in low precision, and the individual terms are not interpretable.",
    ],
  },

  "Exact versus approximate": {
    title: "An IVF index trades recall for work",
    summary:
      "Spherical k-means splits the unit vectors into cells. At query time the cells are ranked by centroid · q̂, only the first few are opened, and only their members are scored. Recall@k compares that result with exact search.",
    whatYouSee: [
      "Cells and Cells probed sliders.",
      "One box per cell with its members, the two dimensions its centroid leans on, and whether it was probed (solid) or skipped (dashed) with its rank.",
      "Exact top-k members are marked found or missed. Metrics: Vectors scored, Recall@k, and the approximate top-k.",
    ],
    howItWorks: [
      "k-means: deterministic farthest-point seeds, then 12 rounds of assign-to-highest-dot-product and re-average-then-normalize.",
      "Recall@k = |approximate top-k ∩ exact top-k| / k. Probing every cell scores every vector, so recall is 1.",
    ],
    controls: [
      "Cells (1 to 4) and Cells probed (1 to Cells).",
      "Comparison worth running: password preset, Cells 4, Cells probed 1 then 2 — Recall@3 goes from 0.67 with 2 vectors scored to 1.00 with 7.",
    ],
    notice: [
      "A true neighbour can sit in a cell whose centroid points elsewhere. That is the miss.",
      "With three cells and one probe, all four presets happen to keep recall 1.00. One good query is not a guarantee.",
    ],
    limits: [
      "In this lab: twelve or thirteen vectors, so approximation saves nothing real. The cells exist to make the trade visible.",
      "In general: IVF, HNSW graphs and product quantization make search over millions of vectors fast. Recall is tuned with parameters such as the number of probes and is measured on sample queries, never guaranteed.",
    ],
  },
};

export default cardInfo;
