import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Batch similarity": {
    title: "One batch, scored the way the loss scores it",
    summary:
      "The batch the trainer draws next at the chosen step, scored with the weights of that step. Every query is compared with every candidate, each row goes through a softmax after dividing by τ, and the loss is −log of the answer's share. The cells are the shares, which is also how hard each wrong candidate gets pushed.",
    whatYouSee: [
      "Batch size, Temperature τ, Negatives and Skip known duplicates: the four training choices this card controls. Changing any of them retrains the whole run in the page.",
      "A table with one row per training query (the first 8 when the batch is larger) and one column per candidate passage. Each cell prints the candidate's share of the row and its cosine with the query.",
      "Text tags, not only shading: answer marks the labelled passage, mined marks the row's own mined negative, duplicate · false negative marks the answer's twin counted as wrong, and duplicate · skipped marks it left out of the softmax. A final column sums any candidates not shown.",
      "Metrics: Starting loss (ln B), Loss at step 0, Loss at this step, and two run averages over every step's batch, Answer's share and Closest wrong cosine.",
    ],
    howItWorks: [
      "share(i, j) = exp(cos(qᵢ, pⱼ) / τ) / Σₖ exp(cos(qᵢ, pₖ) / τ) over the row's candidates; loss = mean over rows of −log share(i, answerᵢ). This is InfoNCE with the query on one side and passages on the other.",
      "Candidates are the batch's labelled answers, each counted once, plus one mined negative per query when Negatives is BM25-mined or Nearest wrong. Every row is scored against every candidate, as in DPR.",
      "The gradient with respect to cos(qᵢ, pⱼ) is (share − 1 for the answer, share otherwise) / (τ B). It flows through the L2 normalisation and the mean of the trigram rows into the weight table, and plain SGD with a learning rate of 8 takes one step.",
      "Nearest wrong rebuilds its index of passage vectors every 10 steps; BM25-mined picks each query's best-scoring wrong passage once. Neither ever returns the labelled answer, and with Skip known duplicates neither returns its twin.",
    ],
    controls: [
      "Batch size 2 to 32, Temperature τ 0.02 to 1.00, Negatives, and Skip known duplicates. Training step, Width and Seed in Held-out retrieval also change this card.",
      "Comparison worth running: set Training step to 0, then 20, then 300, and watch the shares go from nearly even to almost all on the answers.",
    ],
    notice: [
      "When every vector points the same way, every share is about 1 / B and the loss is about ln B. At Batch size 8 that is 2.08; the first batch measures 1.72 because the untrained encoder already favours some answers.",
      "At Batch size 2 most rows hold one other answer from an unrelated topic, so the closest wrong cosine averages 0.39 over the run. Nearest wrong raises it to 0.68: the rows now contain the passages the model actually confuses.",
    ],
    limits: [
      "In this lab: 90 training queries, 41 passages and 300 steps; the batch shown is cut to its first 8 rows, and the run average describes the whole run, not the step you are looking at.",
      "In general: real encoders train on millions of pairs with batches in the thousands, and the learned score can still rank a confident wrong passage first. A low training loss here only says the 90 training pairs are memorized.",
    ],
  },

  "Held-out retrieval": {
    title: "Recall on questions the training never saw",
    summary:
      "The 30 held-out queries are ranked against all 41 passages at every tenth training step. Recall@1 and Recall@5 are hit rates, MRR averages 1 / rank of the first relevant passage, and BM25 runs on the same queries as a fixed keyword baseline.",
    whatYouSee: [
      "Training step, Width, Seed and Rerank top 5.",
      "A line chart of Recall@1, Recall@5 and MRR over training, with BM25's Recall@1 as a flat dashed line and a marker at the chosen step.",
      "Recall@1, Recall@5, MRR and Scoring passes per query for the current pipeline: the bi-encoder at this step, reranked when Rerank top 5 is on.",
      "A table setting BM25, the untrained bi-encoder and the trained one side by side, with hits at rank 1 split into 20 paraphrases, 5 typos and 5 queries with unseen words.",
    ],
    howItWorks: [
      "Recall@k here is the hit rate: the share of queries with at least one relevant passage in the top k, which DPR calls top-k retrieval accuracy. A query whose answer has a duplicate has two relevant passages, and either counts.",
      "MRR = mean over queries of 1 / rank of the first relevant passage, 0 when none is ranked. BM25 (k1 1.2, b 0.75) ranks only passages that share a keyword with the query, so a query with no shared word scores 0.",
      "Scoring passes per query counts model passes: the bi-encoder encodes the query once and compares it with stored passage vectors, and each reranked passage adds one pass over the joined pair.",
    ],
    controls: [
      "Training step 0 to 300 in steps of 10, Width 8 or 16 numbers, Seed 1 to 20, Rerank top 5.",
      "Comparison worth running: Seed 1, then Seed 12, at step 300. The same recipe lands at 63% and 87% Recall@1.",
    ],
    notice: [
      "At step 0 the bi-encoder finds 7% of answers first, far below BM25's 43%; by step 300 it reaches 63%, with 3 of 5 typos found first against BM25's 1.",
      "At the defaults neither ranker finds any of the five unseen-word queries first: trigrams that training never connected to a meaning carry none.",
    ],
    limits: [
      "In this lab: 30 held-out queries written by the same author as the 90 training queries, so they share habits such as jetty for dock. One query is 3.3 percentage points of Recall@1, and seeds 1 to 20 spread from 63% to 87%.",
      "In general: held-out recall on one small, single-domain set says little about recall under domain shift; BEIR finds BM25 robust across datasets where trained dense retrievers often fall behind it.",
    ],
  },

  "Embedding map": {
    title: "A two-direction shadow of the vector space",
    summary:
      "Every passage and held-out query vector at the chosen step, projected onto the two directions that hold the most of the passages' squared length. Every vector has length 1, so every point lands inside the circle; a tight clump means the vectors point nearly the same way.",
    whatYouSee: [
      "Dots for the 41 passages with each topic's name printed at the centre of its passages, small squares for held-out queries, and a thin line from each query to its labelled answer.",
      "The selected query from Query inspector as a larger square, with lines to its answer and to the answer's duplicate, both circled and named.",
      "Metrics: Mean cosine between passages, Duplicate-pair cosine, Alignment and Uniformity. The heading says how much squared length the two axes hold.",
    ],
    howItWorks: [
      "Axes: power iteration on the d × d matrix of averaged outer products of the passage vectors, without centring, so a lean shared by every vector shows up as axis 1; the second axis is kept orthogonal to the first.",
      "Alignment (Wang and Isola) = mean squared distance between each held-out query and its labelled answer. Uniformity = log of the mean of exp(−2 × squared distance) over passage pairs; lower is more spread out.",
      "Mean cosine between passages averages all 820 pairs. Duplicate-pair cosine averages the three pairs that state the same fact.",
    ],
    controls: [
      "No controls of its own. Training step, Temperature τ, Batch size and Skip known duplicates change it most; Query inspector picks the highlighted query.",
      "Comparison worth running: Training step 0 against 300, then Temperature τ 0.10 against 0.03 at step 300.",
    ],
    notice: [
      "At step 0 the mean cosine is 0.94 and the points sit in one clump; at step 300 it is −0.01 and the topics fan out around the circle.",
      "At τ = 0.03 the mean cosine stays at 0.52: the loss is satisfied once each answer leads by a small margin, so nothing pushes the passages further apart.",
    ],
    limits: [
      "In this lab: a projection of 8 or 16 numbers onto 2, so points that look close may be far apart in the directions not drawn. The starting lean is constructed by giving every weight row the same offset.",
      "In general: maps of real embeddings (PCA, t-SNE, UMAP) distort distances too, and uniformity is only one goal: published work finds that pushing it too far can split passages that belong together.",
    ],
  },

  "Query inspector": {
    title: "One held-out query, ranked four ways",
    summary:
      "Pick any held-out query and see where its relevant passages rank before training, after training, under BM25 and after reranking the top 5, along with the passages from the answer's topic, so you can watch confusable passages move.",
    whatYouSee: [
      "Query: 30 held-out questions grouped as paraphrases, typos and unseen words. The heading names the labelled answer and, where there is one, its duplicate.",
      "A table giving the rank of the first relevant passage and the top passage under each ranker.",
      "A table of every passage in the answer's topic plus the current top three, with ranks at step 0 and now, the cosine now and the BM25 rank; the answer and its duplicate are tagged.",
      "Metrics: Margin over the closest wrong passage, the closest wrong passage itself, Recall@1 from the first 4 numbers of each vector, and how many training queries share this answer.",
    ],
    howItWorks: [
      "Ranks sort all 41 passages by cosine with the query vector at the chosen step; #1 is the closest. BM25 ranks only passages sharing a keyword.",
      "The reranker is a logistic regression over five features, cosine, BM25, exact slot-word match, query coverage and trigram Jaccard, trained on each training query's top 5 at the chosen step. It reorders only the top 5.",
      "First 4 numbers: every vector is cut to its first 4 numbers and renormalised before ranking. Nothing in this training rewards the first numbers for carrying more than the rest.",
    ],
    controls: [
      "Query here; Rerank top 5 in Held-out retrieval; every training control upstream.",
      "Comparison worth running: with Negatives on Nearest wrong, switch Skip known duplicates and read Sunday notice's rank for the default query.",
    ],
    notice: [
      "For “When does the north jetty shut on Sundays?” both Sunday notices are relevant, and North dock hours, with weekday times, is the closest wrong passage.",
      "Margin is the cosine of the best relevant passage minus the best wrong one: negative means a wrong passage ranks first.",
    ],
    limits: [
      "In this lab: the reranker is five hand-picked features and five learned weights, a stand-in for a cross-encoder, and its slot words are a hand-written list.",
      "In general: a real cross-encoder is a transformer reading both texts token by token. It is usually more accurate than a bi-encoder but needs one full pass per candidate, which is why it only reranks a short list.",
    ],
  },
};

export default cardInfo;
