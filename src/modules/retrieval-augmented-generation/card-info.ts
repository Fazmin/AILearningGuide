import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Chunk the documents": {
    title: "Fixed word windows over three documents",
    summary:
      "Every indexed document is cut into windows of Chunk size words that start every Chunk size − Overlap words. Each ruler shows one document as a row of word ticks, the answer span for the current question, and every chunk as a bar. Retrieved chunks are filled and numbered.",
    whatYouSee: [
      "Question with five harbor questions, and the full question text.",
      "Chunk size (6–30 words), Overlap (0 to three quarters of the chunk size), and Index built, before or after the June ferry notice.",
      "One ruler per indexed document: grey ticks are words, amber ticks and the dashed band are the answer span, bars are chunks stacked in rows so overlaps stay visible. A green outline and “✓ whole answer” mark a chunk that holds the entire span.",
      "Metrics: Chunks in index, Stored words (and how many times the original text that is), and whether any chunk holds the answer whole.",
    ],
    howItWorks: [
      "Windows start at word 0, stride, 2 × stride … with stride = size − overlap, and the last window stops at the document's end.",
      "The answer span is authored per question: words 6–17 of the handbook for the Sunday hours, words 5–9 of the notice for the new ferry time, words 27–35 of the handbook for pets. The whole nine-word June notice is the span for Changed in June. The wind question has no span, because the corpus has no answer.",
    ],
    controls: [
      "Question, Chunk size, Overlap and Index built.",
      "Comparison worth running: North dock, Sunday at Chunk size 20, then 12, then 12 with Overlap 7.",
    ],
    notice: [
      "At size 12 with no overlap the span crosses a window boundary, and no bar earns the ✓.",
      "Overlap 7 at size 12 makes a window start exactly at “north” — and nearly doubles the number of chunks (8 to 15).",
    ],
    limits: [
      "In this lab: windows are counted in whitespace words and ignore sentence boundaries, so chunks start and end mid-sentence.",
      "In general: production chunkers split on tokens, sentences, headings or document structure, and attach metadata such as title and date to each chunk.",
    ],
  },

  "Retrieval": {
    title: "Three ways to rank the chunks",
    summary:
      "Retriever chooses how every chunk is scored against the question: BM25 counts shared words, Dense compares vectors from the toy encoder in Embeddings and search, and Hybrid fuses the two rankings. The top-k chunks with a positive score go into the prompt.",
    whatYouSee: [
      "Retriever (BM25, Dense or Hybrid) and Top-k from 1 to 4. The kicker shows what the active retriever reads: the analysed query terms for BM25, the words the encoder knows and ignores for Dense, and both for Hybrid.",
      "The five best chunks, or the top-k if that is more, with prompt number [n] if retrieved (or rank #n if not), document, word range, score, and the chunk text with answer-span words highlighted.",
      "A detail line per chunk: each term's BM25 weight; the cosine and the chunk's BM25 rank for Dense; and for Hybrid each list's rank turned into 1/(60 + rank).",
      "A note with the active formula and how many chunks scored zero.",
    ],
    howItWorks: [
      "BM25: score = Σ idf(t) × tf(k₁+1) / (tf + k₁(1 − b + b·len/avgLen)), with k₁ = 1.2, b = 0.75 and idf(t) = ln(1 + (N − n + 0.5)/(n + 0.5)) over the N chunks. Terms are lowercased words with stop-words removed and one trailing s stripped.",
      "Dense: score = cos θ between the encoder's vector for the question and for the chunk. The encoder is the one in Embeddings and search: eight hand-authored topic dimensions, a small loading per known word, a text's vector is the mean of its known words' vectors, and words outside its lexicon are ignored. Every clock time is one shared token.",
      "Hybrid: reciprocal-rank fusion, rrf(c) = Σ 1/(60 + rank), summed over the BM25 list and the dense list that contain c. A list holds only chunks with a positive score, so a chunk in both lists beats one in a single list, and ties keep document order.",
      "In every mode a chunk with a score of 0 is never retrieved, whatever Top-k says.",
    ],
    controls: [
      "Retriever and Top-k.",
      "Comparisons worth running: Dog on the ferry with Index built before and Top-k 2, BM25 against Dense; then Changed in June with Index built after, BM25 against Dense against Hybrid.",
    ],
    notice: [
      "Dense retrieves the handbook's pets chunk for Dog on the ferry although it shares no word with the question: dog and pets both load the animals dimension. Its cosine is a little below the timetable chunk's (about 0.50 against 0.54), so it needs Top-k 2.",
      "Dense retrieves nothing for Changed in June. “Changed” and “June” have no vector in the encoder, so the question vector is all zeros and every cosine is 0. BM25 gives the rare word “june” the largest weight.",
      "Hybrid keeps both hits: the notice from the BM25 list alone and the pets chunk from the dense list alone. A chunk that both lists rank first scores 2/61, twice what a first place in one list scores.",
      "Changing Chunk size changes every idf and every length, so BM25 scores are not comparable across chunkings.",
    ],
    limits: [
      "In this lab: the dense encoder is the hand-authored toy from Embeddings and search, with eight topic dimensions and a small lexicon, so it ignores words outright. A crude plural stripper stands in for stemming, and there is no reranker, approximate index or metadata filter.",
      "In general: learned encoders have hundreds or thousands of dimensions and rarely ignore a word, yet exact names, codes, dates and numbers are still where dense scores are least reliable, and BM25 still misses synonyms and paraphrases. Production systems usually fuse both and rerank the candidates with a cross-encoder.",
    ],
  },

  "Assembled prompt": {
    title: "The generator's entire view",
    summary:
      "The retrieved chunks are numbered and pasted between a fixed instruction and the question. This string is everything a generator would see: no other chunks, no document dates unless they are in the text, and no record of what was left out.",
    whatYouSee: [
      "A word count and the number of passages in the kicker.",
      "The literal prompt: a system instruction, numbered context lines with document and word range, the question, and “Answer:”.",
    ],
    howItWorks: [
      "Context lines are the retrieved chunks in rank order. If nothing matched, the context says so.",
      "The instruction asks for citations and for “say so” when the context lacks the answer. The lab's reader never reads it; only the Grounding check can make the pipeline abstain.",
    ],
    controls: [
      "This card has no controls. Top-k, Chunk size and Overlap change what it holds.",
      "Comparison worth running: Afternoon ferry, after the notice, Top-k 1 against Top-k 2 — the second puts both 16:10 and 15:40 in front of the model.",
    ],
    notice: [
      "A larger Top-k adds words, distractors and sometimes a contradiction.",
      "The prompt carries the old timetable's “(2025)” only because it is in the title text.",
    ],
    limits: [
      "In this lab: prompts are short, so no context budget is enforced, and no model reads them.",
      "In general: retrieved text shares the context window with instructions and the question. Long context can crowd out or bury the passage that matters, and untrusted text in it can carry instructions.",
    ],
  },

  "Reader output": {
    title: "A real extractive reader, not a language model",
    summary:
      "The reader splits each retrieved chunk into sentence pieces and quotes the piece that shares the most distinct query terms, with its prompt number as the citation. The verdict compares that quote with the fact the question needs.",
    whatYouSee: [
      "The quoted sentence and its [n] citation, with a verdict badge: correct, wrong, stale, missed, or no answer in corpus. When the Grounding check abstains, the line reads “The sources do not say.” and the badge names the outcome instead.",
      "A one-line reason for the verdict or for the abstention.",
      "Checks: context recall (the share of answer-span words inside some retrieved chunk), whether one retrieved chunk holds the whole answer span, whether chunking split it, how many query terms the quote shares, and for the ferry question whether old and new times are both in the prompt.",
    ],
    howItWorks: [
      "Ties go to the higher-ranked chunk, then the earlier piece. A piece is text between sentence marks, so a chunk that starts mid-sentence gives a fragment.",
      "Verdicts: correct if the quote contains the needed fact (16:00, 16:10, leash); stale if it contains 15:40; wrong if it contains another fact; missed if the answer span never reached the prompt. A wrong verdict with full context recall carries a different note, because the failure is the reader's.",
    ],
    controls: [
      "This card has no controls.",
      "Comparisons worth running: Closed for wind now, where the reader still quotes the 40-knot rule, which is a rule and not today's reading, until the Grounding check abstains; and Dog on the ferry under Dense at Top-k 2, where recall is full and the answer is still wrong.",
    ],
    notice: [
      "Every wrong answer here is a real sentence from the corpus with a real citation. Citation does not mean correct.",
      "On its own the reader never abstains. It quotes its best match whatever that match is worth, and it matches words, so it cannot use a passage that shares none with the question even when a retriever fetched it. Abstention comes from the Grounding check, which stops the pipeline before the reader quotes anything.",
    ],
    limits: [
      "In this lab: the reader is extractive and rule-based. It cannot paraphrase, combine two chunks, or invent a fact. Its refusal is a score rule applied before it runs, not a judgement it makes.",
      "In general: a generator can do all three — including answering from its weights and ignoring the passages. Grounding has to be checked, for example by verifying each claim against the cited text.",
    ],
  },

  "Score the pipeline": {
    title: "Grade retrieval and reading separately",
    summary:
      "For every question under each retriever, using the chunk size, overlap, index and Top-k you set, the card computes context recall, a support check and correctness. A wrong answer is blamed on retrieval when the evidence did not reach the prompt, and on the reader when it did.",
    whatYouSee: [
      "A table with one row per question and one column per retriever. The selected retriever's column and the current question's row are marked.",
      "Each cell names the outcome (correct, retrieval failure, reader failure, wrong because no answer exists, correct abstention, or wrong abstention), then the context recall and whether the quote is supported by the passage it cites. A column header also shows its retriever's abstention threshold when the Grounding check has one set.",
      "Footer rows: mean context recall over the answerable questions, quotes supported out of quotes produced, answers correct out of answerable, the count of retrieval failures against reader failures, wrong answers stated with confidence out of all five questions, and questions handled correctly (a right answer or a right abstention) out of all five.",
    ],
    howItWorks: [
      "Context recall = the answer-span words that sit inside at least one retrieved chunk ÷ the words in the span. The spans are the authored ones from Chunk the documents. Two halves of a split span, both retrieved, count as full recall.",
      "Support check = the quoted sentence appears word for word in the passage the citation points to. It is a string test, so the extractive reader passes it whenever it quotes anything.",
      "Correct = the quote contains the fact written for the question (16:00, 16:10 or leash). Wrong with recall below 100% is a retrieval failure; wrong at 100% is a reader failure. The wind question has no answer in the corpus, so recall and correctness leave it out; what scores it is the outcome. Answering it is an answer stated with confidence and no evidence; abstaining is the right behaviour.",
      "Handled correctly = answered correctly + correctly abstained. With every threshold off no pipeline abstains on score, so the wind question can only be wrong; the Grounding check is what makes a right abstention possible.",
      "Every cell is a full run of the same pipeline and diagnosis the other cards use, not a separate calculation.",
    ],
    controls: [
      "This card has no controls. It follows Chunk size, Overlap, Index built, Top-k and each retriever's Abstain below, and marks the column of the Retriever you chose.",
      "Comparison worth running: Chunk size 20, Overlap 0, Index built after, Top-k 3, then Top-k 1. Then Chunk size 12 with Index built before, Top-k 2 against Top-k 3, for the Sunday question. Then set BM25's Abstain below to 2.4 and watch the wind cell and Handled correctly.",
    ],
    notice: [
      "Supported quotes sit next to wrong answers. Support says the quote is in the cited passage, not that the passage answers the question.",
      "Full recall with a wrong answer is the reader's failure: Dog on the ferry under Dense or Hybrid, and the Sunday question at Chunk size 12 once Top-k 3 brings both halves, because this reader cannot join two chunks.",
      "Recall below 100% can still give a correct answer, when the part of the span that carries the fact is the part that arrived.",
    ],
    limits: [
      "In this lab: five questions over one corpus, with the answer spans and the gold facts written by hand, one question with no answer, and no language model reading or judging anything. The support check is a substring test the extractive reader cannot fail. These counts describe five questions, not a sample that supports an error bar.",
      "In general: evaluating a RAG system takes hundreds of labelled questions and confidence intervals. Faithfulness needs claim-level checking by an entailment model, a judge model or a person, and correctness needs graded answers, so retrieval and generation are reported separately.",
    ],
  },

  "Grounding check": {
    title: "Say “the sources do not say” when the best match is weak",
    summary:
      "The check sits between retrieval and the reader. If nothing was retrieved, or the best retrieved chunk scores below the threshold you set, the pipeline abstains instead of quoting. It is a card of its own because it needs a control, an outcome for every question and a trade-off curve, and it feeds the Score the pipeline card.",
    whatYouSee: [
      "Abstain below: a threshold on the selected retriever's own score scale, from off (0) up. Each retriever keeps its own setting, so choosing another Retriever in Retrieval shows that retriever's slider.",
      "A table with one row per question: the best retrieved chunk's score, what the reader alone would do, and the outcome with the check: answered correctly, answered wrongly with confidence, correctly abstained, or wrongly abstained, each with a short reason.",
      "Metrics: Coverage (questions answered out of five), Precision (right answers out of those answered) and Handled correctly (right answers plus right abstentions).",
      "A chart of coverage (dashed) and precision (solid) at every slider position, with a marker at the current threshold.",
    ],
    howItWorks: [
      "Score read: the best retrieved chunk's score on the active retriever's scale, rounded to the number of decimals shown. BM25 is the BM25 score; Dense is the cosine; Hybrid is the reciprocal-rank-fusion score multiplied by 61, so 1.00 is first place in one list and 2.00 is first place in both.",
      "Rule: abstain when nothing was retrieved or the score is below the threshold. A threshold of 0 never abstains on score.",
      "Outcome: answered correctly when it answers and the quote carries the right fact; answered wrongly with confidence when it answers and the quote is wrong, stale, or answers the question the corpus cannot; correctly abstained when it abstains and the corpus has no answer (the wind question); wrongly abstained when it abstains and the corpus does hold the answer.",
      "Coverage = answered ÷ 5. Precision = answered correctly ÷ answered. Handled correctly = answered correctly + correctly abstained.",
    ],
    controls: [
      "Abstain below. The Question, Chunk size, Overlap, Index built, Top-k and Retriever controls change what it scores.",
      "Comparisons worth running, at Chunk size 20, Overlap 0, Index built after and Top-k 3: BM25 at 2.4 against off, then Dense at 0.9 against off, then Hybrid at 2.00 and at 1.98.",
    ],
    notice: [
      "Under BM25 the wind question's best chunk scores 2.28 and Changed in June's scores 1.87. The slider cannot abstain on wind without also abstaining on June, so Handled correctly stays at 3 of 5 while precision reaches 100% and coverage falls to 40%.",
      "Under Dense the same idea works on these five questions: wind scores 0.843, below Sunday and ferry (0.981 and 1.000), so a threshold at 0.9 lifts Handled correctly from 2 to 3. That is a feature of five hand-written questions, not a rule: 0.843 is still a high cosine, and the check cannot tell that nothing here reports today's wind.",
      "Hybrid scores sit at 2.00 for almost every question, because the best chunk is first in both lists. Wind's best chunk is first in one list and second in the other (1.984), so only a threshold of 2.00 separates it, and that also drops Changed in June (1.00).",
      "A stale answer is not caught: with Index built before, the 15:40 chunk scores 4.38 under BM25, far above wind, because it matches the question as well as a right source would.",
    ],
    limits: [
      "In this lab: five questions and one corpus, with the scores of the three toy retrievers, so a threshold is tuned on the questions it is graded on. The reader is extractive, the abstention line is fixed text, and nothing judges whether a passage answers the question. Hybrid's scale is the fused rank score times 61, which this card invents to make it readable.",
      "In general: a retrieval score measures similarity, not truth or relevance to the question, and it is not calibrated across queries, retrievers or index versions. Production systems calibrate a threshold on labelled questions and re-tune it when the index or the encoder changes, and add a verifier (an entailment check or a judge model) or a model trained to say “I do not know”. Abstaining too often is itself a failure that users feel.",
    ],
  },
};

export default cardInfo;
