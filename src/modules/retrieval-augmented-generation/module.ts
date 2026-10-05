import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { ABSTAIN_KEY, CHUNK_MAX, CHUNK_MIN, clampThreshold, maxOverlap, QUESTIONS, RETRIEVERS } from "./rag";

const initialState: ModuleState = {
  question: "hours",
  chunkSize: 20,
  overlap: 0,
  topK: 2,
  index: "before",
  retriever: "bm25",
  abstainBm25: 0,
  abstainDense: 0,
  abstainHybrid: 0,
};

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

/**
 * Version 1 had four canned questions and a stuffing toggle over a templated answer.
 * Version 2 had no retriever choice (BM25 only); a version 2 payload gets "bm25" by default.
 * Version 3 had no grounding check; a version 3 payload gets a threshold of 0 (off) for every retriever.
 */
const V1_QUESTIONS: Record<string, string> = { hours: "hours", ferry: "ferry", fresh: "wind", password: "hours" };

export function hydrateRagState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object") return { ...initialState };
    const { grounded: _grounded, ...rest } = parsed;
    const rawQuestion = typeof parsed.question === "string" ? parsed.question : "hours";
    const mapped = "grounded" in parsed ? (V1_QUESTIONS[rawQuestion] ?? "hours") : rawQuestion;
    const question = QUESTIONS.some((item) => item.id === mapped) ? mapped : "hours";
    const chunkSize = clampInt(parsed.chunkSize, 20, CHUNK_MIN, CHUNK_MAX);
    return {
      ...initialState,
      ...rest,
      question,
      chunkSize,
      overlap: clampInt(parsed.overlap, 0, 0, maxOverlap(chunkSize)),
      topK: clampInt(parsed.topK, 2, 1, 4),
      index: parsed.index === "after" ? "after" : "before",
      retriever: RETRIEVERS.find((item) => item === parsed.retriever) ?? "bm25",
      ...Object.fromEntries(RETRIEVERS.map((item) => [ABSTAIN_KEY[item], clampThreshold(item, parsed[ABSTAIN_KEY[item]])])),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-39-retrieval-augmented-generation",
  slug: "retrieval-augmented-generation",
  title: "Retrieval-augmented generation",
  group: "building-with-models",
  order: 23,
  icon: "Library",
  accent: "#8b5a2b",
  prerequisites: ["module-38-embeddings-search"],
  estimatedMinutes: 18,
  steps: [
    "Follow one question",
    "Cut the answer in half",
    "Overlap the windows",
    "Refresh the index",
    "Ask what the corpus lacks",
    "Switch the retriever",
    "Score the pipeline",
    "Set an abstention threshold",
  ],
  stepInstructions: [
    "Leave Question on North dock, Sunday. Find the answer span on the Harbor handbook ruler, the [1] chunk in Retrieval, the same text in Assembled prompt, and 16:00 in Reader output.",
    "Set Chunk size to 12. The answer span now straddles two chunks, the ruler shows no whole-answer chunk, and the reader quotes 18:00 — the weekday time.",
    "Keep Chunk size at 12 and raise Overlap to 7. A window now starts at “north”, the reader returns 16:00 again, and Chunks in index rises from 8 to 15.",
    "Set Chunk size to 20, Overlap to 0 and Question to Afternoon ferry. Built before the June notice, the index yields the stale 15:40. Switch Index built to after, then set Top-k to 2 and read the conflict check.",
    "Set Question to Dog on the ferry, then to Closed for wind now, with Retriever on BM25. Read which chunk wins in Retrieval and what the reader quotes when the right passage never arrives.",
    "With Chunk size 20 and Overlap 0, set Index built to before, Question to Dog on the ferry and Top-k to 2, then switch Retriever from BM25 to Dense: the pets chunk arrives as [2]. Next set Index built to after and Question to Changed in June, and compare BM25, Dense and Hybrid.",
    "Set Chunk size to 20, Overlap to 0, Index built to after and Top-k to 3, with every Abstain below slider off. In Score the pipeline compare the Mean context recall and Answers correct rows across the three retrievers, read which failures sit on the reader side, and find the wind row: it reads wrong because no answer exists.",
    "Keep those settings, choose Retriever BM25 and Question Closed for wind now?, then raise Abstain below in Grounding check from off to 2.4. Wind turns to correctly abstained, but Changed in June (best score 1.87, under wind’s 2.28) turns to wrongly abstained. Switch Retriever to Dense, set 0.9, and compare Handled correctly in Score the pipeline.",
  ],
  stateVersion: 4,
  tagline:
    "Chunk a small corpus, retrieve with BM25, a dense encoder or both, assemble the prompt, score the pipeline to see whether each confident wrong quote is a retrieval failure or a reader failure, and add a grounding check that abstains when the best match scores too low.",
  objectives: [
    "Describe RAG as chunk, retrieve, assemble a prompt, then generate — a pipeline over your index, not a model that looked something up by itself",
    "Explain how chunk size and overlap decide whether an answer can be retrieved whole",
    "Name three retrieval failures — a split answer, a stale index, a vocabulary mismatch — and why each still yields a fluent, cited answer",
    "Compare BM25, dense and hybrid retrieval by what each misses — synonyms for BM25, words the encoder has no vector for for dense — and how reciprocal-rank fusion keeps either one’s hit",
    "Score a pipeline with context recall, a support check and correctness, and use recall to tell a retrieval failure from a reader failure",
    "Use a retrieval-score threshold to make a pipeline abstain, and read the coverage and precision trade-off — a score measures how well a chunk matches the question, not whether it answers it",
  ],
  glossary: [
    {
      term: "Retrieval-augmented generation",
      definition:
        "A pipeline that retrieves passages from an index and places them in the prompt before the model generates. The model is unchanged; only its input is.",
    },
    {
      term: "Chunk",
      definition:
        "A window of text stored and retrieved as one unit. Here a chunk is a fixed number of words (Chunk size) and consecutive windows share Overlap words. Too small a window cuts one fact across chunks, too large a window mixes topics, and more overlap stores and scores more chunks for a likelier whole fact. Real systems often split on sentences, headings or tokens.",
    },
    {
      term: "Index",
      definition:
        "The stored, searchable chunks. It is a snapshot of the documents at build time; a document added later is invisible until the index is rebuilt.",
    },
    {
      term: "Retriever",
      definition:
        "The component that scores chunks against the question and returns the top-k. If it misses, nothing downstream can recover the fact.",
    },
    {
      term: "BM25",
      definition:
        "A keyword score that sums, over query terms in a chunk, a rarity weight (IDF) times a saturating term count, discounted for long chunks. It cannot match synonyms it never sees.",
    },
    {
      term: "Dense retrieval",
      definition:
        "Ranking chunks by the cosine between the question's embedding and each chunk's embedding, so a chunk can match without sharing a word. It is only as good as the encoder: words the encoder has no vector for add nothing, and exact names, codes and dates can blur together.",
    },
    {
      term: "Hybrid retrieval",
      definition:
        "Running a keyword retriever and a dense retriever and merging their rankings, so a chunk that either one finds can reach the prompt. Production systems usually follow the merge with a reranker.",
    },
    {
      term: "Reciprocal rank fusion",
      definition:
        "A way to merge ranked lists: each chunk scores the sum of 1 divided by (60 plus its rank) over every list that contains it. It uses ranks, not scores, so BM25 scores and cosines need no common scale.",
    },
    {
      term: "Top-k",
      definition:
        "How many retrieved chunks go into the prompt. A larger k raises the chance of including the answer and also adds distractors and conflicting passages.",
    },
    {
      term: "Context window",
      definition:
        "The token budget the generator reads at once. The instructions, retrieved chunks and question all share it.",
    },
    {
      term: "Citation",
      definition:
        "A pointer from the answer back to a numbered passage. It shows which passage was quoted, not that the passage was correct, current or the right one.",
    },
    {
      term: "Context recall",
      definition:
        "The share of the gold evidence that reached the prompt. Here it counts the answer-span words that sit inside a retrieved chunk. A wrong answer with low recall points upstream, at chunking, the index or the retriever; a wrong answer with full recall points at the reader.",
    },
    {
      term: "Faithfulness",
      definition:
        "Whether an answer is supported by the retrieved context, which is what grounding an answer means. This lab checks only that the quoted sentence appears word for word in the passage it cites. Support is not correctness: a supported quote can answer a different question, and checking a real generator's claims takes a judge model or a person.",
    },
    {
      term: "Abstention",
      definition:
        "Declining to answer: here the pipeline says the sources do not say. The grounding check abstains when nothing was retrieved or the best retrieved chunk scores below a threshold. A score measures how well a chunk matches the question, not whether it answers it, so a threshold can withhold right answers and let wrong ones through.",
    },
    {
      term: "Precision and coverage",
      definition:
        "Coverage is the share of questions the pipeline answers; precision is the share of those answers that are right. Raising the abstention threshold can only lower coverage; precision rises when the answers it drops were wrong and falls when they were right.",
    },
    {
      term: "Stale data",
      definition:
        "Text that was true when indexed and is no longer. Ranking has no notion of date unless the system adds one, for example as metadata.",
    },
  ],
  references: [
    {
      authors: "Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al.",
      title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html",
      note: "The paper the lesson says the name comes from. It joins a neural retriever over a Wikipedia index with an answer-writing model and trains them together, and it points to updating knowledge and showing where an answer came from as reasons to retrieve.",
    },
    {
      authors: "Yunfan Gao, Yun Xiong, Xinyu Gao, et al.",
      title: "Retrieval-Augmented Generation for Large Language Models: A Survey",
      source: "arXiv preprint arXiv:2312.10997",
      year: 2023,
      url: "https://arxiv.org/abs/2312.10997",
      note: "A broad review of today's retrieve-then-prompt pipelines. Its indexing section says small chunks lose context, large chunks add noise, and fixed windows cut sentences, as Chunk size and Overlap show, and it adds dates to chunks as metadata to avoid stale answers.",
    },
    {
      authors: "Stephen Robertson and Hugo Zaragoza",
      title: "The Probabilistic Relevance Framework: BM25 and Beyond",
      source: "Foundations and Trends in Information Retrieval 3(4), 333–389",
      year: 2009,
      url: "https://doi.org/10.1561/1500000019",
      note: "The full account of BM25 by two of the people who built it. It explains the ideas behind the keyword score that the lab's BM25 retriever computes and the BM25 glossary entry describes.",
    },
    {
      authors: "Vladimir Karpukhin, Barlas Oğuz, Sewon Min, et al.",
      title: "Dense Passage Retrieval for Open-Domain Question Answering",
      source:
        "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), 6769–6781",
      year: 2020,
      url: "https://aclanthology.org/2020.emnlp-main.550/",
      note: "Trains one encoder for questions and one for passages and ranks passages by how close their vectors are. It is the learned version of the lab's Dense retriever, and it reports finding the right passage more often than BM25 on several question sets.",
    },
    {
      authors: "Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher",
      title: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods",
      source:
        "Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2009), 758–759",
      year: 2009,
      url: "https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
      note: "A two-page paper that defines reciprocal rank fusion with the constant 60 that the Hybrid (RRF) retriever uses. A document scores 1/(60 + rank) from each list, so a BM25 list and a cosine list need no common scale.",
    },
    {
      authors: "Rodrigo Nogueira and Kyunghyun Cho",
      title: "Passage Re-ranking with BERT",
      source: "arXiv preprint arXiv:1901.04085",
      year: 2019,
      url: "https://arxiv.org/abs/1901.04085",
      note: "Feeds the query and one passage into BERT together and scores how relevant the passage is. This is the cross-encoder reranker that the lesson says production systems run after the hybrid merge.",
    },
    {
      authors: "Anthropic",
      title: "Introducing Contextual Retrieval",
      source: "Anthropic Engineering blog",
      year: 2024,
      url: "https://www.anthropic.com/engineering/contextual-retrieval",
      note: "Describes a pipeline that combines BM25 and embeddings with rank fusion and then reranks, and adds a short note about the source document to each chunk before indexing. It matches the lesson's Going deeper list of what production pipelines do differently.",
    },
    {
      authors: "Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert",
      title: "RAGAs: Automated Evaluation of Retrieval Augmented Generation",
      source:
        "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL 2024), 150–158",
      year: 2024,
      url: "https://aclanthology.org/2024.eacl-demo.16/",
      note: "Scores a RAG pipeline in parts: whether the retrieved context is relevant, whether the answer is faithful to that context, and whether it answers the question. It backs Score the pipeline's idea of grading retrieval and generation separately.",
    },
    {
      authors: "Nelson F. Liu, Tianyi Zhang, and Percy Liang",
      title: "Evaluating Verifiability in Generative Search Engines",
      source: "Findings of the Association for Computational Linguistics: EMNLP 2023, 7001–7025",
      year: 2023,
      url: "https://aclanthology.org/2023.findings-emnlp.467/",
      note: "People checked the answers of four AI search engines and found that many sentences had no supporting citation and many citations did not support their sentence. It backs the lesson's point that a citation shows where a quote came from, not that it is right.",
    },
    {
      authors: "Pranav Rajpurkar, Robin Jia, and Percy Liang",
      title: "Know What You Don't Know: Unanswerable Questions for SQuAD",
      source:
        "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), Volume 2: Short Papers, 784–789",
      year: 2018,
      url: "https://aclanthology.org/P18-2124/",
      note: "Adds questions with no answer in the passage to a reading test, written to look like answerable ones, so a system must learn to abstain. It is the same test as Closed for wind now?, where answering at all is the mistake.",
    },
    {
      authors: "Amita Kamath, Robin Jia, and Percy Liang",
      title: "Selective Question Answering under Domain Shift",
      source: "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), 5684–5696",
      year: 2020,
      url: "https://aclanthology.org/2020.acl-main.503/",
      note: "Asks a question-answering model to answer as many questions as it can while staying accurate, the coverage and precision trade-off in Grounding check. It finds that abstaining on the model's own confidence score works poorly and trains a separate checker instead.",
    },
    {
      authors: "Freda Shi, Xinyun Chen, Kanishka Misra, et al.",
      title: "Large Language Models Can Be Easily Distracted by Irrelevant Context",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 31210–31227",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/shi23a.html",
      note: "Adds irrelevant sentences to math word problems and shows that language models often get them wrong as a result. It backs the Top-k glossary entry: more retrieved chunks can bring distractors along with the answer.",
    },
    {
      authors: "Nelson F. Liu, Kevin Lin, John Hewitt, et al.",
      title: "Lost in the Middle: How Language Models Use Long Contexts",
      source: "Transactions of the Association for Computational Linguistics 12, 157–173",
      year: 2024,
      url: "https://aclanthology.org/2024.tacl-1.9/",
      note: "Moves the passage with the answer around a long prompt and finds models do best when it sits at the start or end and worse in the middle. It is the source for the lesson's point that context has a budget and a shape.",
    },
    {
      authors: "Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, et al.",
      title:
        "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection",
      source: "Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec 2023), 79–90",
      year: 2023,
      url: "https://arxiv.org/abs/2302.12173",
      note: "Shows attackers planting instructions in text that an AI app is likely to retrieve, so the app follows them. It backs the lesson's warning that retrieved text is untrusted input.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At Chunk size 12 and Overlap 0, North dock, Sunday is answered with 18:00 and a handbook citation. What went wrong?",
      options: [
        "The handbook states the wrong Sunday time, and the reader copied it faithfully",
        "A window edge split the sentence, so the Sunday clause was never retrieved",
        "BM25 ranked a ferry timetable chunk first, so the handbook never reached the prompt",
      ],
      answer: 1,
      explanation:
        "The source is correct and the right document was retrieved. At Chunk size 12 “north dock closes at 18:00 on weekdays” and “and at 16:00 on Sundays” land in different chunks, and at Top-k 2 only the first one is retrieved. The quote is fluent, cited and wrong. Overlap or a larger window keeps the span in one chunk.",
      objective: 1,
    },
    {
      prompt:
        "Switching Index built from before to after changes the Afternoon ferry answer from 15:40 to 16:10. What changed in the pipeline?",
      options: [
        "The notice became searchable, so a chunk holding 16:10 could reach the prompt",
        "The reader learned the new time from the notice and stopped trusting the timetable",
        "The retriever noticed that the 2025 timetable was out of date and demoted it",
      ],
      answer: 0,
      explanation:
        "Nothing about the reader or the retriever's scoring changed. The index is a snapshot, so a document added later is invisible until the index is rebuilt. Once the notice is in it, its short chunk outranks the timetable on the same question words. BM25 has no notion of dates, which is why Top-k 2 puts both times in the prompt.",
      objective: 0,
    },
    {
      prompt:
        "With Retriever on BM25, Dog on the ferry retrieves only timetable chunks. Why is the pets sentence never retrieved?",
      options: [
        "No word in it matches the question, so its chunk scores 0 under BM25",
        "Chunk size 20 is too large for any chunk to hold a whole sentence about pets",
        "BM25 skips handbook chunks whenever the question mentions a ferry or a boat",
      ],
      answer: 0,
      explanation:
        "“Pets may travel on a leash with a tag” has no word in common with “Can I bring my dog on the ferry?”, so its chunk scores 0 and a timetable chunk wins on “ferry”. At Chunk size 20 a chunk holds the whole sentence, and BM25 has no rule about documents. The failure is vocabulary: dog and pets are different strings.",
      objective: 2,
    },
    {
      prompt:
        "With Index built after the notice, Changed in June retrieves the notice under BM25 and nothing under Dense. Why?",
      options: [
        "The notice’s vector points in a different direction from the question’s vector",
        "The notice is too short to be embedded, so Dense search skips it",
        "The question has no vector at all, because the encoder knows none of its words",
      ],
      answer: 2,
      explanation:
        "The toy encoder has no vector for “changed” or “June”, so the question’s vector is all zeros and every cosine is 0. Nothing is retrieved, while BM25 weights the rare word “june” most and finds the notice at once. Real encoders are not blind to rare words, but dates, names and codes are where dense scores are least reliable.",
      objective: 3,
    },
    {
      prompt:
        "BM25 misses the pets chunk for Dog on the ferry and Dense misses the notice for Changed in June. Why can Hybrid retrieve both?",
      options: [
        "It runs BM25 first and falls back to Dense only when BM25 returns nothing",
        "It averages the BM25 score and the cosine, so one high score carries a chunk",
        "Each list gives its chunks 1/(60 + rank), so a chunk in one list scores",
      ],
      answer: 2,
      explanation:
        "Reciprocal-rank fusion adds 1/(60 + rank) for every list that contains the chunk, so a chunk found by only one retriever still scores and a chunk found by both scores about double. It fuses ranks, not scores, because BM25 scores and cosines live on different scales. It is not a fallback: for Dog, BM25 does return something, so a fallback would never call Dense.",
      objective: 3,
    },
    {
      prompt:
        "Dog on the ferry, Retriever Dense, Top-k 2, index before: context recall reads 100% and the answer is still wrong. Where did it fail?",
      options: [
        "In retrieval, because the pets chunk was ranked below a timetable chunk",
        "In the reader, which skipped the pets sentence although it was in the prompt",
        "In the support check, because the quoted sentence is not in any retrieved chunk",
      ],
      answer: 1,
      explanation:
        "The pets chunk arrives as [2], so every answer-span word reached the prompt. The reader quotes the piece that shares the most words with the question, and “Pets may travel on a leash with a tag” shares none, so it quotes the ferry sentence. Full recall with a wrong answer is a reader failure. A language model could use the pets sentence; this stand-in cannot.",
      objective: 4,
    },
    {
      prompt:
        "In Score the pipeline almost every quote is marked supported, yet many answers are wrong. What does that show?",
      options: [
        "The scorer is broken, because a supported quote must be a correct answer",
        "Retrieval worked for those questions, so every remaining failure is in the reader",
        "The support check tests whether the cited passage holds the quote, not the answer",
      ],
      answer: 2,
      explanation:
        "The reader only quotes, so its quote is always inside the passage it cites, whatever the question. Support says nothing about whether that passage was the right one, current, or complete, so it cannot separate a correct answer from a wrong one. Correctness against the gold fact and context recall do that, and only a real generator can fail the support check.",
      objective: 4,
    },
    {
      prompt:
        "Retriever BM25, Index built after, Top-k 3. You raise Abstain below until Closed for wind now? turns to correctly abstained. What does Changed in June do?",
      options: [
        "It turns to wrongly abstained, because its best chunk scores lower than the wind question’s",
        "It stays answered correctly, because the June notice is the only chunk that holds the answer",
        "It turns to correctly abstained, because the index was rebuilt after that notice arrived",
      ],
      answer: 0,
      explanation:
        "The check reads only the best chunk’s score. Wind’s best chunk, the 40-knot rule, matches “harbor” and “wind” and scores 2.28; the June notice matches only the rare word “june” and scores 1.87. Any threshold that drops wind drops June too, although June’s answer is in the corpus. A score measures how well words match, not whether a passage answers.",
      objective: 5,
    },
    {
      prompt: "In Grounding check you keep raising Abstain below. Which statement about coverage and precision holds every time?",
      options: [
        "Coverage never rises, and precision can fall when a right answer is dropped",
        "Precision never falls, because the lowest-scoring answers are always the wrong ones",
        "Both fall together, because every dropped answer takes a right answer with it",
      ],
      answer: 0,
      explanation:
        "Raising the threshold only moves questions from answered to abstained, so coverage can only fall. Precision depends on which answers go. Under BM25 it reads 60% with the check off, 75% once Dog on the ferry (1.25) is dropped, 67% when Changed in June (1.87) goes too, and 100% only after wind (2.28) is dropped as well. Score order and correctness order differ.",
      objective: 5,
    },
    {
      prompt:
        "With Index built before, Afternoon ferry quotes the stale 15:40. Why can an abstention threshold not be relied on to stop that?",
      options: [
        "Its chunk matches the question’s words as well as a current source would",
        "Thresholds apply only to questions the corpus cannot answer, never to ones it can",
        "The reader rescales an old timetable’s score, so it passes whatever the threshold says",
      ],
      answer: 0,
      explanation:
        "A stale passage is a good match for the question; it is only out of date, and a score cannot see dates. Under BM25 its chunk scores 4.38, well above the wind question’s 2.03. A threshold set above 4.38 would drop it, and would then drop the correct 16:10 as well once the notice is indexed, because that chunk scores only 3.75. Dates need metadata and a rule, not a score.",
      objective: 5,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateRagState,
};

export default definition;
