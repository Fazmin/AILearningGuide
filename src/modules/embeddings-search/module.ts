import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { PRESETS } from "./vectors";

const initialState: ModuleState = {
  query: PRESETS[0].query,
  userText: "",
  metric: "cosine",
  selected: "",
  topK: 3,
  cells: 3,
  probe: 1,
};

/** Version 1 stored a hashed bag-of-words lab with chunking and a hybrid slider. */
const V1_DEFAULT_QUERY = "When does the north dock close?";
const V1_DEFAULT_NOTES = "Bring a dog on the afternoon ferry and check when the north dock shuts.";

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

export function hydrateEmbeddingsState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object") return { ...initialState };
    const isV1 = "hybrid" in parsed || "chunkSize" in parsed;
    let query = typeof parsed.query === "string" ? parsed.query : (initialState.query as string);
    let userText = typeof parsed.userText === "string" ? parsed.userText : "";
    if (isV1) {
      if (query === V1_DEFAULT_QUERY) query = initialState.query as string;
      if (userText === V1_DEFAULT_NOTES) userText = "";
    }
    const cells = clampInt(parsed.cells, 3, 1, 4);
    const { hybrid: _hybrid, chunkSize: _chunkSize, ...rest } = parsed;
    return {
      ...initialState,
      ...rest,
      query,
      userText,
      metric: parsed.metric === "dot" ? "dot" : "cosine",
      selected: typeof parsed.selected === "string" ? parsed.selected : "",
      topK: clampInt(parsed.topK, 3, 1, 5),
      cells,
      probe: clampInt(parsed.probe, 1, 1, cells),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-38-embeddings-search",
  slug: "embeddings-search",
  title: "Embeddings for search and retrieval",
  group: "building-with-models",
  order: 22,
  icon: "Search",
  accent: "#4a6ec8",
  prerequisites: ["module-01-tokens-embeddings"],
  estimatedMinutes: 14,
  steps: [
    "Read the query vector",
    "Measure the angle",
    "Paraphrase versus keywords",
    "Cosine versus dot",
    "Search approximately",
  ],
  stepInstructions: [
    "Leave Preset query on pier shut. In Query and vectors, read which words are known and which are ignored, then read the eight numbers of q.",
    "In Ranked neighbours click Dock hours. Find its dot on Angle to the query, then read the q × d row and cos θ in Worked similarity.",
    "Set Preset query to dog on the boat. Pets on board shares no words with the query, yet it is #1 by cosine. Read its keyword rank.",
    "Set Preset query to password and switch Score between cosine and dot product. Lost property and Visitor notice swap places.",
    "Keep the password query. Set Cells to 4 and Cells probed to 1, read Recall@3 and Vectors scored, then raise Cells probed to 2.",
  ],
  stateVersion: 2,
  tagline:
    "Turn a query and twelve passages into vectors, rank them by cosine or dot product, and watch an approximate index trade recall for fewer comparisons.",
  objectives: [
    "Compute cosine similarity and dot product between a query vector and passage vectors, and explain when the two rank differently",
    "Explain why dense retrieval can match a paraphrase that keyword search misses, and why the nearest passage is not proof of truth or of an answer",
    "Read recall@k to say what an approximate index gives up in exchange for scoring fewer vectors",
  ],
  glossary: [
    {
      term: "Embedding",
      definition:
        "A fixed-length vector that stands in for a piece of text so that similar use lands in a similar direction. In retrieval it is an index key, not a generator's internal token table.",
    },
    {
      term: "Encoder",
      definition:
        "The function that maps text to an embedding. Here it is a hand-set lexicon with mean pooling. A production encoder is a trained transformer, and the query and documents must go through compatible encoders.",
    },
    {
      term: "Mean pooling",
      definition:
        "Averaging per-word vectors into one passage vector. Word order is lost, and a passage whose words point different ways ends up shorter than one whose words agree.",
    },
    {
      term: "Query vector",
      definition:
        "The embedding of the search text. If none of its words are known to the encoder it is all zeros, and every similarity is zero.",
    },
    {
      term: "Dot product",
      definition:
        "The sum of matching components multiplied together, q·d = |q| |d| cos θ. It rises with the angle match and with both vector lengths.",
    },
    {
      term: "Cosine similarity",
      definition:
        "The dot product divided by both lengths, which leaves only the angle. For unit-length vectors it equals the dot product. High cosine means similar direction, not a verified fact.",
    },
    {
      term: "Nearest neighbour",
      definition:
        "The stored vector with the best score for the query. Nearest is a geometric fact about these vectors. It can still be the wrong passage, or a passage that does not answer the question.",
    },
    {
      term: "Dense retrieval",
      definition:
        "Ranking passages by embedding similarity. It can match a paraphrase with no shared words, and it can also rank a fluent but irrelevant passage highly.",
    },
    {
      term: "Keyword retrieval",
      definition:
        "Ranking by shared words, as in BM25. It is exact about rare tokens such as names and codes and blind to paraphrase. Many production systems combine it with dense retrieval.",
    },
    {
      term: "Approximate nearest neighbour",
      definition:
        "A search that skips most stored vectors to save time and may miss some true neighbours. IVF and HNSW are two common families.",
    },
    {
      term: "IVF index",
      definition:
        "An inverted-file index: k-means splits the vectors into cells, and a query scores only the vectors in the few cells whose centroids are closest to it. More probed cells means higher recall and more work.",
    },
    {
      term: "Recall@k",
      definition:
        "The share of the exact top-k results that an approximate search also returned, which grades the index against exact search, not against relevance (relevance recall, used to evaluate retrieval pipelines, asks how many of the passages a question needs appear in the top k). A recall of 1.00 on one query does not guarantee it on the next.",
    },
  ],
  references: [
    {
      authors: "Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze",
      title: "Introduction to Information Retrieval",
      source: "Cambridge University Press, free to read online",
      year: 2008,
      url: "https://nlp.stanford.edu/IR-book/",
      note: "Chapter 6 turns documents and queries into vectors and ranks them by cosine similarity, which divides out vector length, the same step as Score set to cosine. The Okapi BM25 section in the chapter on probabilistic retrieval explains the keyword scoring that the lab's keyword rank stands in for.",
    },
    {
      authors: "Nils Reimers and Iryna Gurevych",
      title: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
      source:
        "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), 3982–3992",
      year: 2019,
      url: "https://aclanthology.org/D19-1410/",
      note: "The Sentence-BERT paper the lesson names as a production encoder. It builds one fixed-size vector per sentence, by default by taking the mean of the word outputs, like the lab's mean pooling, and compares sentences with cosine similarity.",
    },
    {
      authors: "Vladimir Karpukhin, Barlas Oğuz, Sewon Min, et al.",
      title: "Dense Passage Retrieval for Open-Domain Question Answering",
      source:
        "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), 6769–6781",
      year: 2020,
      url: "https://aclanthology.org/2020.emnlp-main.550/",
      note: "The dense passage retrieval paper the lesson names. It trains a question encoder and a passage encoder with in-batch negatives, and its examples show dense search catching reworded matches while BM25 does better on rare, exact keywords, the trade-off in Paraphrase versus keywords.",
    },
    {
      authors: "Stephen Robertson and Hugo Zaragoza",
      title: "The Probabilistic Relevance Framework: BM25 and Beyond",
      source: "Foundations and Trends in Information Retrieval 3(4), 333–389",
      year: 2009,
      url: "https://doi.org/10.1561/1500000019",
      note: "The full account of BM25 by two of the people who developed it. It explains the ideas behind the keyword scorer that the lesson's glossary and the retrieval-augmented generation lab use next to dense search.",
    },
    {
      authors: "Harald Steck, Chaitanya Ekanadham, and Nathan Kallus",
      title: "Is Cosine-Similarity of Embeddings Really About Similarity?",
      source: "Companion Proceedings of the ACM Web Conference 2024 (WWW 2024), 887–890",
      year: 2024,
      url: "https://arxiv.org/abs/2403.05440",
      note: "Notes that cosine sometimes works better and sometimes worse than the raw dot product, and shows how training choices can make cosine scores of learned embeddings arbitrary. It backs the Cosine versus dot step and the warning that a high cosine is not a verified fact.",
    },
    {
      authors: "Sentence Transformers developers",
      title: "Semantic Search",
      source: "Sentence Transformers documentation",
      year: 2026,
      url: "https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html",
      note: "A practical guide to searching with embeddings. It shows that once vectors are normalized to length 1 the dot product can be used as the score, gives separate query and document encoding with prompts, and says every approximate index trades recall for speed.",
    },
    {
      authors: "Matthijs Douze, Alexandr Guzhva, Chengqi Deng, et al.",
      title: "The Faiss library",
      source: "IEEE Transactions on Big Data 12(2), 346–361",
      year: 2026,
      url: "https://arxiv.org/abs/2401.08281",
      note: "Describes a widely used vector search library. It explains the IVF index, where vectors are clustered into lists and a search visits only the lists nearest the query (nprobe), and it grades approximate search by recall against exact search, as the Exact versus approximate card does.",
    },
    {
      authors: "Hervé Jégou, Matthijs Douze, and Cordelia Schmid",
      title: "Product Quantization for Nearest Neighbor Search",
      source: "IEEE Transactions on Pattern Analysis and Machine Intelligence 33(1), 117–128",
      year: 2011,
      url: "https://www.semanticscholar.org/paper/Product-Quantization-for-Nearest-Neighbor-Search-J%C3%A9gou-Douze/4748d22348e72e6e06c2476486afddbc76e5eca7",
      note: "Introduces product quantization, which squeezes each vector into a short code so more vectors fit in memory, as the lesson's Going deeper section describes. It pairs the codes with an inverted file of cells, the same kind of index as this lab.",
    },
    {
      authors: "Yu. A. Malkov and D. A. Yashunin",
      title: "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs",
      source: "IEEE Transactions on Pattern Analysis and Machine Intelligence 42(4), 824–836",
      year: 2020,
      url: "https://arxiv.org/abs/1603.09320",
      note: "The HNSW paper. It builds layers of proximity graphs and searches by hopping greedily from neighbour to neighbour toward the query, the second approximate family the lesson names beside IVF.",
    },
    {
      authors: "Nandan Thakur, Nils Reimers, Andreas Rücklé, et al.",
      title: "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
      source:
        "Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1 (NeurIPS Datasets and Benchmarks 2021)",
      year: 2021,
      url: "https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html",
      note: "Tests keyword, dense, and reranking retrievers on 18 datasets they were not trained for. BM25 holds up as a strong baseline and dense models often fall behind it, one reason the lesson says production systems keep a keyword half.",
    },
    {
      authors: "Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher",
      title: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods",
      source:
        "Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2009), 758–759",
      year: 2009,
      url: "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
      note: "A two-page paper that defines reciprocal rank fusion, the method the lesson names for merging a BM25 list and a dense list. Each passage scores by its rank in each list, so neither scorer's raw numbers need to match.",
    },
    {
      authors: "Rodrigo Nogueira and Kyunghyun Cho",
      title: "Passage Re-ranking with BERT",
      source: "arXiv preprint arXiv:1901.04085",
      year: 2019,
      url: "https://arxiv.org/abs/1901.04085",
      note: "Takes passages first found by BM25 and reorders them with BERT, which reads the query and each passage together. This is the cross-encoder reranker stage the lesson describes after hybrid search.",
    },
    {
      authors: "Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, et al.",
      title:
        "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection",
      source: "Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec 2023), 79–90",
      year: 2023,
      url: "https://arxiv.org/abs/2302.12173",
      note: "Shows attackers planting instructions in text that a system is likely to retrieve, so the model reads them as if they were commands. It backs the lesson's Visitor notice, a planted instruction that ranks near the top for the password query.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "You switch Score from cosine to dot product and the top passage changes, although the query and the passages did not change. Why?",
      options: [
        "Dot product uses a different encoder, so the passage vectors were recomputed",
        "Dot product grows with vector length, so a longer vector can beat a closer one",
        "Cosine ignores which dimensions match and counts only the words two texts share",
      ],
      answer: 1,
      explanation:
        "q·d = |q| |d| cos θ. Cosine divides out both lengths and keeps only the angle. With raw lengths kept, a longer passage vector at a slightly wider angle can win, which is why Visitor notice overtakes Lost property for the password query. If every stored vector had length 1, the two rankings would be identical.",
      objective: 0,
    },
    {
      prompt:
        "The dog on the boat preset ranks Pets on board first although it shares no word with the query. What produces that match?",
      options: [
        "The encoder looks up dog in a thesaurus and then counts the shared word pets",
        "Pets on board also ranks first by shared keywords, so cosine only confirms the count",
        "Dog and pets both load on animals, so the two vectors point the same way",
      ],
      answer: 2,
      explanation:
        "Nothing is looked up at search time. The lexicon gives dog and pet the same loading on animals, so q and the passage vector point in nearly the same direction and the cosine is high. Counting shared words would put Mooring fees first, on the single word boat, and rank Pets on board sixth.",
      objective: 1,
    },
    {
      prompt:
        "For the password preset, the nearest passage by cosine is about lost keys and bags. What does a top rank show?",
      options: [
        "That its vector points close to the query's, not that it answers the question",
        "That it contains the answer, since the encoder only returns relevant text",
        "That it shares the most words with the query, since cosine counts shared words",
      ],
      answer: 0,
      explanation:
        "A similarity score is a geometric fact about two vectors. The query's words (harbor, master, password) load on place and access, and so do keys, bags and office, so Lost property lands nearest; Visitor notice, a planted instruction, is close behind. Neither is checked for truth or for answering the question, and whatever lands in the prompt later inherits that gap.",
      objective: 1,
    },
    {
      prompt:
        "At Cells 4 and Cells probed 1, Recall@3 is below 1.00 and few vectors are scored. What changes when you raise Cells probed to 2?",
      options: [
        "The same vectors are scored but ranked better, so recall rises at no extra cost",
        "More vectors are scored and recall rises, as the missed neighbour's cell opens",
        "Recall stays below 1.00, because a neighbour outside the first cell is lost for good",
      ],
      answer: 1,
      explanation:
        "The index scores only the vectors inside the cells it opens. With one probe the third true neighbour sits in a cell that stays closed. Probing a second cell scores more vectors, seven of twelve instead of two, and recovers it. Recall is bought with comparisons; probing every cell is exact search.",
      objective: 2,
    },
    {
      prompt: "Recall@3 reads 1.00 for a query in the approximate-search card. What does that tell you?",
      options: [
        "It matches exact search's top 3 for this query and says nothing about relevance",
        "The three passages returned are the ones that actually answer the question asked",
        "The index will return exact search's top 3 for any other query as well as this",
      ],
      answer: 0,
      explanation:
        "Recall@k grades the index against exact search on one query. Exact search can still return the wrong passages, and a cell layout that is perfect for the password query can miss on another. The retrieval-augmented generation lab measures a different recall: whether the passages a question needs reached the prompt.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateEmbeddingsState,
};

export default definition;
