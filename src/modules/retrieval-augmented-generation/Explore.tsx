import { useMemo, type ReactNode } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  ABSTAIN_KEY,
  ABSTAIN_SCALE,
  analyze,
  CHUNK_MAX,
  CHUNK_MIN,
  clampThreshold,
  DOCS,
  encoderView,
  failureSite,
  ground,
  groundingSweep,
  groundingTotals,
  indexedDocs,
  maxOverlap,
  OUTCOME_LABEL,
  QUESTIONS,
  RETRIEVER_LABEL,
  RETRIEVERS,
  RRF_K,
  runPipeline,
  scoreTable,
  scoreTotals,
  splitWords,
  wordCount,
  type Chunk,
  type PipelineScore,
  type Question,
  type QuestionId,
  type RankedChunk,
  type Retriever,
  type Verdict,
} from "./rag";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const asRetriever = (state: ModuleContext["state"]): Retriever => {
  const value = asString(state, "retriever", "bm25");
  return RETRIEVERS.find((item) => item === value) ?? "bm25";
};
const percent = (value: number | null) => (value === null ? "n/a" : `${Math.round(value * 100)}%`);

const VERDICT: Record<Verdict, { label: string; note: string }> = {
  correct: { label: "correct", note: "The quoted sentence contains the fact that answers the question." },
  wrong: {
    label: "wrong",
    note: "The quoted sentence is real text from the corpus, but it answers a different question.",
  },
  stale: {
    label: "stale",
    note: "The quoted time was true when the timetable was written. The index does not hold anything newer, or ranks it lower.",
  },
  unsupported: {
    label: "no answer in corpus",
    note: "Nothing in the index reports today's wind. The reader still quoted its best-matching sentence — a rule, not a reading.",
  },
  missed: {
    label: "missed",
    note: "The passage that answers this never reached the prompt, so the reader quoted the best of what did.",
  },
};

const READER_FAILURE_NOTE =
  "Every answer-span word is in the prompt, but the reader quoted a different sentence. That is a reader failure, not a retrieval failure. This stand-in quotes the piece that shares the most question words and cannot join two chunks or match a synonym; a language model reads the whole prompt.";

const SITE_LABEL: Record<ReturnType<typeof failureSite>, string> = {
  none: "correct",
  retrieval: "retrieval failure",
  reader: "reader failure",
  "no-evidence": "wrong: no answer exists",
};

/** What the reader alone would have said, before any grounding check. */
const READER_ALONE: Record<Verdict, string> = {
  correct: "correct",
  wrong: "wrong sentence",
  stale: "stale time",
  unsupported: "quotes anyway: no answer exists",
  missed: "answer never retrieved",
};

const percentFormat = (value: number) => `${Math.round(value * 100)}%`;

const MAX_WORDS = Math.max(...DOCS.map((doc) => splitWords(doc.text).length));
const RULER_LEFT = 8;
const RULER_WIDTH = 624;
const wordX = (index: number) => RULER_LEFT + (index / MAX_WORDS) * RULER_WIDTH;

function ChunkRuler({
  docId,
  title,
  words,
  chunks,
  retrievedRank,
  span,
}: {
  docId: string;
  title: string;
  words: string[];
  chunks: Chunk[];
  retrievedRank: Map<string, number>;
  span: Question["span"];
}) {
  const stride = chunks.length > 1 ? chunks[1].start - chunks[0].start : chunks[0]?.end ?? 1;
  const width = chunks[0] ? chunks[0].end - chunks[0].start : 1;
  const rows = Math.max(1, Math.min(chunks.length, Math.ceil(width / Math.max(1, stride))));
  const barTop = 20;
  const rowHeight = 17;
  const height = barTop + rows * rowHeight + 4;
  const ownSpan = span && span.doc === docId ? span : null;
  const wordWidth = RULER_WIDTH / MAX_WORDS;
  const retrievedHere = chunks.filter((chunk) => retrievedRank.has(chunk.id));

  return (
    <figure className="rag-ruler">
      <figcaption>
        <strong>{title}</strong>
        <span>
          {words.length} words · {chunks.length} chunk{chunks.length === 1 ? "" : "s"}
          {retrievedHere.length ? ` · retrieved ${retrievedHere.map((chunk) => `#${retrievedRank.get(chunk.id)}`).join(", ")}` : ""}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 640 ${height}`}
        role="img"
        aria-label={`${title}: ${chunks
          .map(
            (chunk) =>
              `chunk ${chunk.ordinal + 1} words ${chunk.start + 1} to ${chunk.end}${
                retrievedRank.has(chunk.id) ? `, retrieved rank ${retrievedRank.get(chunk.id)}` : ""
              }${ownSpan && chunk.start <= ownSpan.start && chunk.end >= ownSpan.end ? ", holds the whole answer" : ""}`,
          )
          .join("; ")}.${ownSpan ? ` Answer span is words ${ownSpan.start + 1} to ${ownSpan.end}.` : ""}`}
      >
        {ownSpan && (
          <rect
            className="rag-span-band"
            x={wordX(ownSpan.start) - 1}
            y={2}
            width={wordX(ownSpan.end) - wordX(ownSpan.start) + 1}
            height={height - 4}
            rx={3}
          />
        )}
        {words.map((word, index) => {
          const inSpan = ownSpan ? index >= ownSpan.start && index < ownSpan.end : false;
          return (
            <rect
              key={index}
              className={`rag-word${inSpan ? " is-span" : ""}`}
              x={wordX(index) + 0.8}
              y={6}
              width={Math.max(1, wordWidth - 1.6)}
              height={9}
              rx={1.5}
            >
              <title>{`word ${index + 1}: ${word}`}</title>
            </rect>
          );
        })}
        {chunks.map((chunk, index) => {
          const rank = retrievedRank.get(chunk.id);
          const y = barTop + (index % rows) * rowHeight;
          const whole = ownSpan ? chunk.start <= ownSpan.start && chunk.end >= ownSpan.end : false;
          const x = wordX(chunk.start) + 1;
          const w = Math.max(4, wordX(chunk.end) - wordX(chunk.start) - 2);
          return (
            <g key={chunk.id} className={`rag-bar${rank ? " is-retrieved" : ""}${whole ? " is-whole" : ""}`}>
              <rect x={x} y={y} width={w} height={13} rx={3} />
              <text x={x + 4} y={y + 9.6}>
                {rank ? `#${rank}` : `c${chunk.ordinal + 1}`}
                {whole ? " ✓ whole answer" : ""}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

function Highlighted({ chunk, span }: { chunk: Chunk; span: Question["span"] }): ReactNode {
  const words = splitWords(chunk.text);
  if (!span || span.doc !== chunk.doc) return chunk.text;
  return words.map((word, offset) => {
    const index = chunk.start + offset;
    const inSpan = index >= span.start && index < span.end;
    return (
      <span key={offset}>
        {offset > 0 ? " " : ""}
        {inSpan ? <mark>{word}</mark> : word}
      </span>
    );
  });
}

export default function Explore({ state, setState }: ModuleContext) {
  const questionId = (QUESTIONS.some((item) => item.id === asString(state, "question", "hours"))
    ? asString(state, "question", "hours")
    : "hours") as QuestionId;
  const chunkSize = clamp(Math.round(asNumber(state, "chunkSize", 20)), CHUNK_MIN, CHUNK_MAX);
  const overlap = clamp(Math.round(asNumber(state, "overlap", 0)), 0, maxOverlap(chunkSize));
  const topK = clamp(Math.round(asNumber(state, "topK", 2)), 1, 4);
  const afterNotice = asString(state, "index", "before") === "after";
  const retriever = asRetriever(state);

  const thresholds = Object.fromEntries(
    RETRIEVERS.map((item) => [item, clampThreshold(item, state[ABSTAIN_KEY[item]])]),
  ) as Record<Retriever, number>;

  const run = useMemo(
    () => runPipeline(questionId, chunkSize, overlap, afterNotice, topK, retriever),
    [afterNotice, chunkSize, overlap, questionId, retriever, topK],
  );
  const { bm25: bm25Limit, dense: denseLimit, hybrid: hybridLimit } = thresholds;
  const scores = useMemo(
    () =>
      scoreTable(chunkSize, overlap, afterNotice, topK, { bm25: bm25Limit, dense: denseLimit, hybrid: hybridLimit }),
    [afterNotice, bm25Limit, chunkSize, denseLimit, hybridLimit, overlap, topK],
  );
  const sweep = useMemo(
    () => groundingSweep(chunkSize, overlap, afterNotice, topK, retriever),
    [afterNotice, chunkSize, overlap, retriever, topK],
  );
  const { question, chunks, ranked, retrieved, prompt, answer, diagnosis } = run;
  const retrievedRank = new Map(retrieved.map((chunk, index) => [chunk.id, index + 1]));
  const docs = indexedDocs(afterNotice);
  const docWords = docs.reduce((sum, doc) => sum + splitWords(doc.text).length, 0);
  const storedWords = chunks.reduce((sum, chunk) => sum + (chunk.end - chunk.start), 0);
  const verdict = VERDICT[diagnosis.verdict];
  const zeroMatches = ranked.filter((chunk) => chunk.score === 0).length;
  const shown: RankedChunk[] = ranked.slice(0, Math.max(topK, 5));
  const conflicting = diagnosis.staleInContext && diagnosis.freshInContext;
  const queryTerms = [...new Set(analyze(question.text))];
  const encoder = encoderView(question.text);
  const site = failureSite(question, diagnosis);
  const scale = ABSTAIN_SCALE[retriever];
  const threshold = thresholds[retriever];
  const grounding = ground(run, retriever, threshold);
  const shownScore = (value: number) => value.toFixed(scale.scoreDecimals);
  const quotedTerms = grounding.abstained ? [] : answer.matched;
  const verdictNote = grounding.abstained
    ? grounding.reason === "below-threshold"
      ? `The best retrieved chunk scored ${shownScore(grounding.topScore ?? 0)} (${scale.unit}), below the threshold ${threshold.toFixed(scale.decimals)}, so the grounding check stopped the pipeline before anything was quoted. Without the check the reader would have quoted its best match (${READER_ALONE[grounding.ungated]}).`
      : question.correct
        ? "Nothing scored above zero, so there was no passage to quote and the pipeline abstains. The corpus does hold the answer: the retriever failed, and it shows up as an abstention."
        : "Nothing scored above zero, so there was no passage to quote and the pipeline abstains. The corpus holds no answer, so abstaining is right."
    : diagnosis.verdict === "wrong" && site === "reader"
      ? READER_FAILURE_NOTE
      : verdict.note;
  const retrievalKicker =
    retriever === "bm25"
      ? `Query terms after stop-words: ${queryTerms.join(", ")}`
      : retriever === "dense"
        ? `Encoder reads: ${encoder.known.join(", ") || "nothing (every cosine is 0)"} · ignores: ${encoder.ignored.join(", ") || "nothing"}`
        : `BM25 terms: ${queryTerms.join(", ")} · encoder reads: ${encoder.known.join(", ") || "nothing"}`;
  const scoreLabel = (chunk: RankedChunk) =>
    retriever === "bm25" ? chunk.score.toFixed(2) : retriever === "dense" ? chunk.cosine.toFixed(3) : chunk.rrf.toFixed(4);
  const detailLine = (chunk: RankedChunk) => {
    if (retriever === "bm25") {
      return chunk.contributions.length
        ? chunk.contributions.map((entry) => `${entry.term} ${entry.weight.toFixed(2)}`).join(" + ")
        : "no query term — never retrieved";
    }
    if (retriever === "dense") {
      return chunk.cosine > 0
        ? `cosine of the question and chunk vectors · BM25 rank ${chunk.bm25Rank ?? "—"}`
        : "cosine 0 — never retrieved";
    }
    if (chunk.rrf === 0) return "in neither list — never retrieved";
    const part = (label: string, rank: number | null) =>
      rank ? `${label} #${rank} → 1/(${RRF_K}+${rank}) = ${(1 / (RRF_K + rank)).toFixed(4)}` : `${label} —`;
    return `${part("BM25", chunk.bm25Rank)} · ${part("dense", chunk.denseRank)}`;
  };
  const totals = Object.fromEntries(RETRIEVERS.map((item) => [item, scoreTotals(scores, item)])) as Record<
    Retriever,
    ReturnType<typeof scoreTotals>
  >;
  const cellLabel = (score: PipelineScore) =>
    score.outcome === "abstained-right"
      ? "correct abstention"
      : score.outcome === "abstained-wrong"
        ? "wrong abstention"
        : score.outcome === "answered-right"
          ? "correct"
          : SITE_LABEL[score.site];
  const groundRows = scores.map((row) => ({ question: row.question, score: row.scores[retriever] }));
  const groundSummary = groundingTotals(groundRows.map((row) => row.score.outcome));
  const groundDetail = (question: Question, score: PipelineScore) => {
    if (score.outcome === "answered-right") return "quotes the sentence with the right fact";
    if (score.outcome === "answered-wrong")
      return question.span ? "the quote is not the right fact" : "the corpus has no answer, and it quoted anyway";
    if (score.outcome === "abstained-right") return "“The sources do not say.” The corpus has no answer.";
    if (score.topScore === null) return "nothing retrieved, though the corpus holds the answer";
    return score.ungated === "correct"
      ? "dropped an answer that was right"
      : "dropped an answer that was wrong, but the corpus holds the right one";
  };

  return (
    <div className="tg-lab tg-lab--hero rag-lab">
      <LabSurface label="Chunk the documents" className="rag-chunk-card">
        <SurfaceHeading
          kicker={`${docs.length} documents · ${chunks.length} chunks · stride ${chunkSize - overlap} words`}
          title="The index stores windows of words, not documents"
        />
        <div className="rag-question-control">
          <SegmentedControl
            label="Question"
            value={questionId}
            options={QUESTIONS.map((item) => ({ value: item.id, label: item.label }))}
            onChange={(value) => setState({ question: value })}
          />
        </div>
        <p className="rag-question-text">“{question.text}”</p>
        <div className="rag-chunk-controls">
          <RangeControl
            label="Chunk size"
            min={CHUNK_MIN}
            max={CHUNK_MAX}
            step={1}
            value={chunkSize}
            format={(value) => `${value} words`}
            onChange={(value) => setState({ chunkSize: value, overlap: Math.min(overlap, maxOverlap(value)) })}
          />
          <RangeControl
            label="Overlap"
            min={0}
            max={maxOverlap(chunkSize)}
            step={1}
            value={overlap}
            format={(value) => `${value} words`}
            onChange={(value) => setState({ overlap: value })}
          />
          <div className="rag-index-control">
            <SegmentedControl
              label="Index built"
              value={afterNotice ? "after" : "before"}
              options={[
                { value: "before", label: "before notice" },
                { value: "after", label: "after notice" },
              ]}
              onChange={(value) => setState({ index: value })}
            />
          </div>
        </div>
        <div className="rag-rulers">
          {docs.map((doc) => (
            <ChunkRuler
              key={doc.id}
              docId={doc.id}
              title={doc.title}
              words={splitWords(doc.text)}
              chunks={chunks.filter((chunk) => chunk.doc === doc.id)}
              retrievedRank={retrievedRank}
              span={question.span}
            />
          ))}
          {!afterNotice && (
            <p className="rag-ruler-missing">Ferry notice (June 2026) exists but is not in this index.</p>
          )}
        </div>
        <div className="rag-legend" aria-hidden="true">
          <span>
            <i className="is-word" /> one word
          </span>
          <span>
            <i className="is-span" /> answer span
          </span>
          <span>
            <i className="is-bar" /> chunk
          </span>
          <span>
            <i className="is-retrieved" /> retrieved, with rank
          </span>
        </div>
        <div className="metric-row">
          <Metric label="Chunks in index" value={`${chunks.length}`} />
          <Metric label="Stored words" value={`${storedWords} (${(storedWords / docWords).toFixed(2)}× the text)`} />
          <Metric
            label="Answer whole in one chunk"
            value={question.span ? (diagnosis.spanWholeInIndex ? "yes" : "no — split") : "no answer exists"}
            tone={question.span && diagnosis.spanWholeInIndex ? "forward" : "loss"}
          />
        </div>
      </LabSurface>

      <LabSurface label="Retrieval" className="rag-bm25-card">
        <SurfaceHeading kicker={retrievalKicker} title="Score every chunk against the question, keep the top-k" />
        <div className="rag-retriever-control">
          <SegmentedControl
            label="Retriever"
            value={retriever}
            options={RETRIEVERS.map((item) => ({ value: item, label: RETRIEVER_LABEL[item] }))}
            onChange={(value) => setState({ retriever: value })}
          />
        </div>
        <RangeControl
          label="Top-k"
          min={1}
          max={4}
          step={1}
          value={topK}
          format={(value) => `${value} chunk${value === 1 ? "" : "s"}`}
          onChange={(value) => setState({ topK: value })}
        />
        <ol className="rag-rank-list">
          {shown.map((chunk) => {
            const rank = retrievedRank.get(chunk.id);
            return (
              <li key={chunk.id} className={rank ? "is-retrieved" : chunk.score === 0 ? "is-zero" : ""}>
                <header>
                  <span className="rag-rank-no">{rank ? `[${rank}]` : `#${chunk.rank}`}</span>
                  <strong>
                    {chunk.docTitle} · words {chunk.start + 1}–{chunk.end}
                  </strong>
                  <code>{scoreLabel(chunk)}</code>
                </header>
                <p>
                  <Highlighted chunk={chunk} span={question.span} />
                </p>
                <small>{detailLine(chunk)}</small>
              </li>
            );
          })}
        </ol>
        <p className="lab-note">
          {retriever === "bm25" && (
            <>
              BM25: each matched term adds idf × tf(k₁+1) / (tf + k₁(1 − b + b·len/avg)) with k₁ = 1.2 and b = 0.75.
              Rare terms count more; long chunks count less. {zeroMatches} of {chunks.length} chunks match no query
              term at all.
            </>
          )}
          {retriever === "dense" && (
            <>
              Dense: the score is the cosine between the question’s vector and each chunk’s vector, from the same
              hand-authored 8-dimension encoder as Embeddings and search. A word it has no vector for adds nothing.{" "}
              {zeroMatches} of {chunks.length} chunks have cosine 0.
            </>
          )}
          {retriever === "hybrid" && (
            <>
              Hybrid: reciprocal-rank fusion. A chunk scores 1/({RRF_K} + rank) for each list that holds it, the BM25
              list and the dense list, and the two add up. A chunk in both lists beats one in a single list. Ties keep
              document order. {zeroMatches} of {chunks.length} chunks are in neither list.
            </>
          )}
        </p>
      </LabSurface>

      <LabSurface label="Assembled prompt" className="rag-prompt-card">
        <SurfaceHeading
          kicker={`${wordCount(prompt)} words · ${retrieved.length} passage${retrieved.length === 1 ? "" : "s"}`}
          title="What the model would actually read"
        />
        <pre className="rag-prompt" aria-label="Assembled prompt text">
          {prompt}
        </pre>
        <p className="lab-note">
          Retrieval ends here. A generator sees only this text; it cannot see chunks that were not retrieved, or
          which document was newer.
        </p>
      </LabSurface>

      <LabSurface label="Reader output" className="rag-answer-card">
        <SurfaceHeading
          kicker="Extractive reader · quotes one retrieved sentence"
          title="An answer is only as good as the passages it was handed"
          aside={
            grounding.abstained ? (
              <span className={`tg-badge${grounding.outcome === "abstained-right" ? "" : " is-warning"}`}>
                {OUTCOME_LABEL[grounding.outcome]}
              </span>
            ) : (
              <span className={`tg-badge${diagnosis.verdict === "correct" ? "" : " is-warning"}`}>{verdict.label}</span>
            )
          }
        />
        <p className="rag-reader-answer">
          {grounding.abstained ? (
            "The sources do not say."
          ) : answer.sentence ? (
            <>
              {answer.sentence} <strong>[{answer.citation}]</strong>
            </>
          ) : (
            "No retrieved sentence shares a term with the question."
          )}
        </p>
        <p className="rag-verdict-note">{verdictNote}</p>
        <ul className="rag-checks">
          <li className={diagnosis.spanCoverage === null || diagnosis.spanCoverage === 1 ? "is-yes" : "is-no"}>
            <b>{percent(diagnosis.spanCoverage)}</b>
            Context recall: the share of answer-span words that sit inside some retrieved chunk.
          </li>
          <li className={diagnosis.spanWholeRetrieved ? "is-yes" : "is-no"}>
            <b>{diagnosis.spanWholeRetrieved ? "yes" : "no"}</b>
            The whole answer span sits inside one retrieved chunk.
          </li>
          <li className={question.span && !diagnosis.spanWholeInIndex ? "is-no" : "is-yes"}>
            <b>{!question.span ? "n/a" : diagnosis.spanWholeInIndex ? "no" : "yes"}</b>
            The chunking cut the answer span across windows.
          </li>
          <li className={quotedTerms.length ? "is-yes" : "is-no"}>
            <b>{quotedTerms.length}</b>
            Query terms in the quoted sentence
            {quotedTerms.length ? `: ${quotedTerms.join(", ")}` : grounding.abstained ? " (nothing was quoted)" : ""}.
          </li>
          {question.id === "ferry" && (
            <li className={conflicting ? "is-no" : "is-yes"}>
              <b>{conflicting ? "yes" : "no"}</b>
              Both the old time (15:40) and the new time (16:10) are in the prompt.
            </li>
          )}
        </ul>
        <p className="lab-note">
          The reader returns the retrieved sentence piece that shares the most distinct query terms, citing its
          prompt number. It is not a language model, and on its own it never refuses; only the Grounding check below
          can make the pipeline abstain. A generator can also paraphrase, blend two passages, or answer from its
          weights.
        </p>
      </LabSurface>

      <LabSurface label="Score the pipeline" className="rag-score-card">
        <SurfaceHeading
          kicker={`${chunkSize}-word chunks · overlap ${overlap} · index ${afterNotice ? "after" : "before"} the notice · top-${topK}`}
          title="Grade retrieval and reading separately"
        />
        <div className="rag-score-scroll">
          <table className="tg-board rag-score-table">
            <caption>
              Every question under each retriever, using the chunk size, overlap, index and Top-k set above. Each cell
              names the outcome, then context recall and whether the quote is supported by the passage it cites.
            </caption>
            <thead>
              <tr>
                <th scope="col">Question</th>
                {RETRIEVERS.map((item) => (
                  <th key={item} scope="col" aria-current={item === retriever ? "true" : undefined}>
                    {RETRIEVER_LABEL[item]}
                    {item === retriever ? " (selected)" : ""}
                    {thresholds[item] > 0
                      ? ` · abstains below ${thresholds[item].toFixed(ABSTAIN_SCALE[item].decimals)}`
                      : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {scores.map((row) => (
                <tr key={row.question.id} className={row.question.id === question.id ? "is-current" : ""}>
                  <th scope="row">{row.question.label}</th>
                  {RETRIEVERS.map((item) => {
                    const score = row.scores[item];
                    return (
                      <td key={item} className={`is-${score.outcome}`}>
                        <strong>{cellLabel(score)}</strong>
                        <small>
                          {score.contextRecall === null ? "no answer exists" : `recall ${percent(score.contextRecall)}`} ·{" "}
                          {score.faithful === null ? "nothing quoted" : score.faithful ? "supported" : "unsupported"}
                        </small>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">Mean context recall</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {percent(totals[item].meanRecall)} <small>of {totals[item].answerable} answerable</small>
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Quotes supported</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {totals[item].faithful} of {totals[item].quoted}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Answers correct</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {totals[item].correct} of {totals[item].answerable}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Retrieval / reader failures</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {totals[item].retrievalFailures} / {totals[item].readerFailures}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Wrong with confidence</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {totals[item].confidentErrors} of {totals[item].questions}
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">Handled correctly (answer or abstain)</th>
                {RETRIEVERS.map((item) => (
                  <td key={item}>
                    {totals[item].handled} of {totals[item].questions}
                  </td>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>
        <p className="lab-note">
          Context recall is the share of the gold evidence’s words that reached the prompt. The support check passes
          when the quoted sentence appears word for word in the passage it cites; this reader only quotes, so it passes
          whenever it answers, and it cannot tell whether the quote answers the question. A wrong answer with full
          recall is a reader failure; a wrong answer with less is a retrieval failure. The wind question has no answer
          in the corpus, so answering it at all is the error: its cell reads wrong unless the Grounding check below
          makes the pipeline abstain. Handled correctly counts right answers and right abstentions together, out of all
          five questions.
        </p>
      </LabSurface>

      <LabSurface label="Grounding check" className="rag-score-card rag-ground-card">
        <SurfaceHeading
          kicker={`${RETRIEVER_LABEL[retriever]} · the best retrieved chunk’s ${scale.unit} against a threshold`}
          title="Abstain when the best match scores too low"
        />
        <RangeControl
          label="Abstain below"
          min={0}
          max={scale.max}
          step={scale.step}
          value={threshold}
          format={(value) => (value === 0 ? "off" : `${value.toFixed(scale.decimals)} ${scale.unit}`)}
          onChange={(value) => setState({ [ABSTAIN_KEY[retriever]]: clampThreshold(retriever, value) })}
        />
        <div className="rag-ground-scroll">
          <table className="tg-board rag-score-table">
            <caption>
              The five questions under {RETRIEVER_LABEL[retriever]}, using the chunk size, overlap, index and Top-k set
              above. The check abstains when the best retrieved chunk scores below the threshold, or when nothing was
              retrieved.
            </caption>
            <thead>
              <tr>
                <th scope="col">Question</th>
                <th scope="col">Best chunk score</th>
                <th scope="col">Reader alone</th>
                <th scope="col">With the check</th>
              </tr>
            </thead>
            <tbody>
              {groundRows.map(({ question: item, score }) => (
                <tr key={item.id} className={item.id === question.id ? "is-current" : ""}>
                  <th scope="row">{item.label}</th>
                  <td>{score.topScore === null ? "nothing retrieved" : shownScore(score.topScore)}</td>
                  <td>{score.topScore === null ? "nothing to quote" : READER_ALONE[score.ungated]}</td>
                  <td className={`is-${score.outcome}`}>
                    <strong>{OUTCOME_LABEL[score.outcome]}</strong>
                    <small>{groundDetail(item, score)}</small>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">Outcomes</th>
                <td colSpan={3}>
                  {groundSummary.right} answered correctly · {groundSummary.confidentWrong} answered wrongly with
                  confidence · {groundSummary.abstainedRight} correctly abstained · {groundSummary.abstainedWrong}{" "}
                  wrongly abstained
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
        <div className="metric-row">
          <Metric
            label="Coverage"
            value={`${groundSummary.answered} of ${groundSummary.questions} answered (${percent(groundSummary.coverage)})`}
          />
          <Metric
            label="Precision"
            value={
              groundSummary.precision === null
                ? "nothing answered"
                : `${groundSummary.right} of ${groundSummary.answered} right (${percent(groundSummary.precision)})`
            }
            tone={groundSummary.confidentWrong > 0 ? "loss" : "forward"}
          />
          <Metric label="Handled correctly" value={`${groundSummary.handled} of ${groundSummary.questions}`} />
        </div>
        <LineChart
          label="Coverage and precision against the abstention threshold"
          series={[
            {
              id: "coverage",
              name: "Coverage (answered ÷ questions)",
              points: sweep.map((point) => ({ x: point.threshold, y: point.coverage })),
              tone: "gradient",
              dash: "dashed",
              format: percentFormat,
            },
            {
              id: "precision",
              name: "Precision (right ÷ answered)",
              points: sweep.flatMap((point) =>
                point.precision === null ? [] : [{ x: point.threshold, y: point.precision }],
              ),
              tone: "forward",
              format: percentFormat,
            },
          ]}
          xLabel={`Abstain below (${scale.unit})`}
          yLabel="Share of the five questions, or of the answers given"
          xDomain={[0, scale.max]}
          yDomain={[0, 1]}
          marker={{ x: threshold, label: threshold === 0 ? "off" : threshold.toFixed(scale.decimals) }}
          footnote="Each line ends at the right-hand edge of the slider. Precision has no value once nothing is answered, so its line stops where the last answer is dropped."
        />
        <p className="lab-note">
          The check reads only the best retrieved chunk’s score, on the active retriever’s own scale, so the same
          slider position means something different for BM25, Dense and Hybrid. A score measures how well a chunk
          matches the question, not whether it answers it, so no threshold can promise a right answer or a right
          refusal: lowering the wrong answers also lowers the right ones whenever they score alike. Real systems
          calibrate a threshold on labelled questions and add a verifier or a model that can say “I do not know”.
        </p>
      </LabSurface>
    </div>
  );
}
