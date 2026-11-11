import { useMemo, type CSSProperties } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type LineSeries,
  type ModuleContext,
} from "@app/module-sdk";
import {
  DUPLICATE_OF,
  HELD_OUT_KIND_LABEL,
  HELD_OUT_QUERIES,
  PASSAGES,
  passageIndex,
  TOPICS,
  TRAIN_QUERIES,
  type HeldOutKind,
} from "./data";
import { EmbeddingMap } from "./EmbeddingMap";
import {
  alignment,
  BM25_RANKINGS,
  BM25_RESULT,
  buildSpace,
  duplicateCosine,
  firstRelevantRank,
  HELD_OUT_RELEVANT,
  meanPairCosine,
  principalAxes,
  rankOf,
  scoreRankings,
  trainCosines,
  truncatedRecallAt1,
  uniformity,
  type RetrievalScore,
} from "./metrics";
import { CROSS_ENCODER_ONLY_PASSES, passesPerQuery, RERANK_DEPTH, rerankHeldOut, trainReranker } from "./rerank";
import { readState, SEED_MAX, SEED_MIN, TEMPERATURE_MAX, TEMPERATURE_MIN } from "./state";
import { BATCH_SIZES, NEGATIVE_LABEL, NEGATIVE_MODES, SNAPSHOT_EVERY, STEPS, train, TRAIN_ANSWERS } from "./train";

const pct = (value: number) => `${Math.round(value * 100)}%`;
const share = (value: number) => (value < 0.005 ? "<1%" : `${Math.round(value * 100)}%`);
const f2 = (value: number) => (Math.abs(value) < 0.005 ? "0.00" : value.toFixed(2));
const f3 = (value: number) => (Math.abs(value) < 0.0005 ? "0.000" : value.toFixed(3));
const rankText = (rank: number | null) => (rank === null ? "not ranked" : `#${rank}`);

/** Rows shown in Batch similarity; a larger batch is cut to its first rows. */
const MAX_ROWS = 8;
const KINDS: readonly HeldOutKind[] = ["paraphrase", "typo", "unseen"];
const KIND_COUNTS = KINDS.map((kind) => HELD_OUT_QUERIES.filter((query) => query.kind === kind).length);

/** Where the first relevant passage sits among all the passages: #1 at the left end, last at the right. */
function RankTrack({ rank, from }: { rank: number | null; from?: number | null }) {
  const last = PASSAGES.length;
  const at = (value: number) => ((value - 1) / (last - 1)) * 100;
  const moved = from !== undefined && from !== null && rank !== null && from !== rank;
  return (
    <span className={`etr-rank-track${rank === null ? " is-unranked" : rank === 1 ? " is-top" : ""}`} aria-hidden="true">
      <i className="etr-rank-track__end">#1</i>
      <span className="etr-rank-track__line">
        {moved && (
          <u
            className={rank < from ? "is-up" : "is-down"}
            style={{ left: `${at(Math.min(rank, from))}%`, width: `${Math.abs(at(rank) - at(from))}%` }}
          />
        )}
        {from !== undefined && from !== null && <s style={{ left: `${at(from)}%` }} />}
        {rank !== null && <b style={{ left: `${at(rank)}%` }} />}
      </span>
      <i className="etr-rank-track__end">#{last}</i>
    </span>
  );
}

function hitsByKind(rankings: readonly (readonly number[])[]) {
  return KINDS.map((kind) =>
    HELD_OUT_QUERIES.filter((query, index) => query.kind === kind && firstRelevantRank(rankings[index], HELD_OUT_RELEVANT[index]) === 1).length,
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const { step, batch, temperature, negatives, skipDuplicates, dim, seed, query, rerank } = readState(state);

  const run = useMemo(
    () => train({ batch, temperature, negatives, skipDuplicates, dim, seed }),
    [batch, temperature, negatives, skipDuplicates, dim, seed],
  );
  const spaces = useMemo(() => run.snapshots.map((snapshot) => buildSpace(snapshot.weights, dim)), [run, dim]);
  const curve = useMemo(() => spaces.map((space) => scoreRankings(space.rankings, HELD_OUT_RELEVANT)), [spaces]);

  const snapIndex = Math.min(run.snapshots.length - 1, Math.round(step / SNAPSHOT_EVERY));
  const snapshot = run.snapshots[snapIndex];
  const space = spaces[snapIndex];
  const start = spaces[0];
  const score = curve[snapIndex];
  const startScore = curve[0];

  const reranker = useMemo(() => trainReranker(trainCosines(snapshot.weights, space)), [snapshot, space]);
  const reranked = useMemo(() => rerankHeldOut(reranker, space.rankings, space.cosine, HELD_OUT_RELEVANT), [reranker, space]);
  const projection = useMemo(() => principalAxes(space.passages, dim), [space, dim]);
  const startProjection = useMemo(() => principalAxes(start.passages, dim), [start, dim]);
  const firstFour = useMemo(() => truncatedRecallAt1(space, 4), [space]);

  const pipeline: RetrievalScore = rerank ? reranked.score : score;
  const pipelineRankings = rerank ? reranked.rankings : space.rankings;
  const meanCosine = meanPairCosine(space.passages);
  const startCosine = meanPairCosine(start.passages);

  /* ---------- Batch similarity ---------- */
  const shown = snapshot.batch;
  const rowCount = Math.min(MAX_ROWS, shown.rows.length);
  const shownRows = Array.from({ length: rowCount }, (_, r) => r);
  const columnSet = new Set<number>();
  for (const r of shownRows) {
    columnSet.add(shown.answerColumn[r]);
    if (shown.minedColumn[r] >= 0) columnSet.add(shown.minedColumn[r]);
    const twin = DUPLICATE_OF[TRAIN_ANSWERS[shown.rows[r]]];
    const twinColumn = shown.candidates.indexOf(twin);
    if (twin >= 0 && twinColumn >= 0) columnSet.add(twinColumn);
  }
  if (shown.candidates.length <= MAX_ROWS) shown.candidates.forEach((_, column) => columnSet.add(column));
  const columns = [...columnSet].sort((a, b) => a - b);
  const hiddenColumns = shown.candidates.length - columns.length;
  const lossNow = run.losses[Math.min(run.losses.length - 1, step)];

  /* ---------- Held-out retrieval ---------- */
  const steps = run.snapshots.map((entry) => entry.step);
  const series: LineSeries[] = [
    { id: "r1", name: "Recall@1", tone: "forward", points: steps.map((x, i) => ({ x, y: curve[i].recall1 })), format: pct },
    { id: "r5", name: "Recall@5", tone: "attention", dash: "dashed", points: steps.map((x, i) => ({ x, y: curve[i].recall5 })), format: pct },
    { id: "mrr", name: "MRR", tone: "gradient", dash: "dotted", points: steps.map((x, i) => ({ x, y: curve[i].mrr })), format: f2 },
    {
      id: "bm25",
      name: "BM25 Recall@1",
      tone: "muted",
      dash: "dashed",
      points: [
        { x: 0, y: BM25_RESULT.recall1 },
        { x: STEPS, y: BM25_RESULT.recall1 },
      ],
      format: pct,
    },
  ];
  const bm25Kinds = hitsByKind(BM25_RANKINGS);
  const startKinds = hitsByKind(start.rankings);
  const nowKinds = hitsByKind(space.rankings);
  const rerankKinds = hitsByKind(reranked.rankings);
  const scoreRows: { id: string; label: string; result: RetrievalScore; kinds: number[]; passes: string; current?: boolean }[] = [
    { id: "bm25", label: "BM25 (keywords)", result: BM25_RESULT, kinds: bm25Kinds, passes: "index lookup" },
    { id: "start", label: "Bi-encoder, step 0", result: startScore, kinds: startKinds, passes: "1" },
    { id: "now", label: `Bi-encoder, step ${step}`, result: score, kinds: nowKinds, passes: "1", current: !rerank },
  ];
  if (rerank) {
    scoreRows.push({
      id: "rerank",
      label: `Step ${step} + rerank top ${RERANK_DEPTH}`,
      result: reranked.score,
      kinds: rerankKinds,
      passes: String(passesPerQuery(true)),
      current: true,
    });
  }

  /* ---------- Query inspector ---------- */
  const queryIndex = Math.max(0, HELD_OUT_QUERIES.findIndex((entry) => entry.id === query));
  const selected = HELD_OUT_QUERIES[queryIndex];
  const relevant = HELD_OUT_RELEVANT[queryIndex];
  const answer = passageIndex(selected.answer);
  const twin = DUPLICATE_OF[answer];
  const topic = PASSAGES[answer].topic;
  const cosineNow = space.cosine[queryIndex];
  const rankingNow = space.rankings[queryIndex];
  const rankingStart = start.rankings[queryIndex];
  const rankingBm25 = BM25_RANKINGS[queryIndex];
  const rankingRerank = reranked.rankings[queryIndex];
  const rankRows = [
    { id: "start", label: "Before training (step 0)", ranking: rankingStart },
    { id: "now", label: `After training (step ${step})`, ranking: rankingNow },
    { id: "bm25", label: "BM25", ranking: rankingBm25 },
    { id: "rerank", label: `Step ${step} + rerank top ${RERANK_DEPTH}`, ranking: rankingRerank },
  ];
  const watched = PASSAGES.map((passage, index) => ({ passage, index }))
    .filter(({ passage, index }) => passage.topic === topic || rankingNow.slice(0, 3).includes(index))
    .sort((a, b) => (rankOf(rankingNow, a.index) ?? 99) - (rankOf(rankingNow, b.index) ?? 99));
  const bestRelevant = Math.max(...relevant.map((index) => cosineNow[index]));
  const bestWrong = Math.max(...cosineNow.filter((_, index) => !relevant.includes(index)));
  const bestWrongIndex = cosineNow.findIndex((value, index) => value === bestWrong && !relevant.includes(index));

  const trainMatches = TRAIN_QUERIES.filter((entry) => passageIndex(entry.answer) === answer || passageIndex(entry.answer) === twin);

  return (
    <div className="tg-lab tg-lab--hero etr-lab">
      <LabSurface label="Batch similarity" className="etr-batch-card">
        <SurfaceHeading
          kicker={`Training step ${step} · the next batch: ${shown.rows.length} queries, ${shown.candidates.length} candidates · τ = ${temperature.toFixed(2)}`}
          title="Each query has to pick its own answer out of the batch"
          aside={<span className="tg-badge">trained live in this page</span>}
        />
        <div className="etr-batch-controls">
          <div className="etr-batch-size">
            <SegmentedControl
              label="Batch size"
              value={String(batch)}
              options={BATCH_SIZES.map((size) => ({ value: String(size), label: String(size) }))}
              onChange={(value) => setState({ batch: Number(value) })}
            />
          </div>
          <div className="etr-temperature">
            <RangeControl
              label="Temperature τ"
              min={TEMPERATURE_MIN}
              max={TEMPERATURE_MAX}
              step={0.01}
              value={temperature}
              format={(value) => `τ = ${value.toFixed(2)}`}
              onChange={(value) => setState({ temperature: Math.round(value * 100) / 100 })}
            />
          </div>
          <div className="etr-negatives">
            <SegmentedControl
              label="Negatives"
              value={negatives}
              options={NEGATIVE_MODES.map((mode) => ({ value: mode, label: NEGATIVE_LABEL[mode] }))}
              onChange={(value) => setState({ negatives: value })}
            />
          </div>
          <div className="etr-skip">
            <SegmentedControl
              label="Skip known duplicates"
              value={skipDuplicates ? "on" : "off"}
              options={[
                { value: "off", label: "Off" },
                { value: "on", label: "On" },
              ]}
              onChange={(value) => setState({ skipDuplicates: value === "on" })}
            />
          </div>
        </div>
        <div className="source-table etr-batch-table">
          <table
            className="tg-board"
            aria-label={`Softmax share of each candidate in each query's row at training step ${step}. Each row sums to 100%.`}
          >
            <thead>
              <tr>
                <th scope="col">query ↓ · candidate →</th>
                {columns.map((column) => (
                  <th scope="col" key={column}>
                    {PASSAGES[shown.candidates[column]].title}
                  </th>
                ))}
                {hiddenColumns > 0 && <th scope="col">{`${hiddenColumns} other candidates`}</th>}
              </tr>
            </thead>
            <tbody>
              {shownRows.map((r) => {
                const rowAnswer = TRAIN_ANSWERS[shown.rows[r]];
                const rowShares = snapshot.scores.shares[r];
                const hiddenShare = shown.candidates.reduce((sum, _, column) => (columns.includes(column) ? sum : sum + rowShares[column]), 0);
                return (
                  <tr key={shown.rows[r]}>
                    <th scope="row">{TRAIN_QUERIES[shown.rows[r]].text}</th>
                    {columns.map((column) => {
                      const passage = shown.candidates[column];
                      const isAnswer = column === shown.answerColumn[r];
                      const isTwin = passage === DUPLICATE_OF[rowAnswer];
                      const isMined = column === shown.minedColumn[r];
                      const masked = shown.masked[r][column];
                      const tag = isAnswer
                        ? "answer"
                        : isTwin
                          ? masked
                            ? "duplicate · skipped"
                            : "duplicate · false negative"
                          : isMined
                            ? "mined"
                            : "";
                      return (
                        <td
                          key={column}
                          className={`${isAnswer ? "is-answer" : ""}${isTwin ? " is-twin" : ""}${masked ? " is-masked" : ""}`.trim()}
                          style={{ "--etr-share": masked ? 0 : rowShares[column], "--etr-heat": masked ? 0 : Math.sqrt(rowShares[column]) } as CSSProperties}
                        >
                          <b>{masked ? "—" : share(rowShares[column])}</b>
                          <small>cos {f2(snapshot.scores.cosines[r][column])}</small>
                          {tag && <em>{tag}</em>}
                        </td>
                      );
                    })}
                    {hiddenColumns > 0 && (
                      <td style={{ "--etr-share": hiddenShare, "--etr-heat": Math.sqrt(Math.max(0, hiddenShare)) } as CSSProperties}>
                        <b>{share(hiddenShare)}</b>
                        <small>together</small>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <ul className="etr-batch-key" aria-hidden="true">
          <li className="etr-batch-key__ramp">
            <small>0%</small>
            {[0.02, 0.1, 0.3, 0.6, 1].map((part) => (
              <i key={part} style={{ background: `color-mix(in srgb, var(--module-accent) ${Math.sqrt(part) * 52}%, transparent)` }} />
            ))}
            <small>100% of the row</small>
          </li>
          <li>
            <i className="is-answer" /> the row&rsquo;s own answer
          </li>
          <li>
            <i className="is-twin" /> duplicate of the answer
          </li>
        </ul>
        {shown.rows.length > rowCount && (
          <p className="etr-table-note">{`Showing the first ${rowCount} of ${shown.rows.length} rows; the loss averages all of them.`}</p>
        )}
        <div className="metric-row">
          <Metric label="Starting loss (ln B)" value={f2(Math.log(batch))} />
          <Metric label="Loss at step 0" value={f2(run.losses[0])} />
          <Metric label="Loss at this step" value={lossNow >= 0.1 ? f2(lossNow) : f3(lossNow)} tone="loss" />
          <Metric label="Answer's share, run average" value={pct(run.meanAnswerShare)} tone="forward" />
          <Metric label="Closest wrong cosine, run average" value={f2(run.meanClosestWrong)} tone="gradient" />
        </div>
        <p className="lab-note">
          Each cell is one candidate's share of its row: the cosine divided by τ, then a softmax across the row. The
          loss is −log of the answer's share, averaged over the {shown.rows.length} rows, and its gradient pushes every
          wrong candidate down in proportion to its share, so the cells are also where the push lands. Every query
          is scored against every candidate, so one query's answer is another query's negative. A duplicate is the
          same fact in other words; without Skip known duplicates it counts as wrong.
        </p>
      </LabSurface>

      <LabSurface label="Held-out retrieval" className="etr-retrieval-card">
        <SurfaceHeading
          kicker={`${HELD_OUT_QUERIES.length} held-out queries · ${PASSAGES.length} passages · width ${dim} · seed ${seed}`}
          title="Recall on queries the training never saw"
          aside={<span className="tg-badge">measured</span>}
        />
        <div className="etr-retrieval-controls">
          <div className="etr-step-control">
            <RangeControl
              label="Training step"
              min={0}
              max={STEPS}
              step={SNAPSHOT_EVERY}
              value={step}
              format={(value) => `${value} of ${STEPS}`}
              onChange={(value) => setState({ step: value })}
            />
          </div>
          <SegmentedControl
            label="Width"
            value={String(dim)}
            options={[
              { value: "8", label: "8 numbers" },
              { value: "16", label: "16 numbers" },
            ]}
            onChange={(value) => setState({ dim: Number(value) })}
          />
          <RangeControl
            label="Seed"
            min={SEED_MIN}
            max={SEED_MAX}
            step={1}
            value={seed}
            format={(value) => `${value}`}
            onChange={(value) => setState({ seed: value })}
          />
          <div className="etr-rerank">
            <SegmentedControl
              label={`Rerank top ${RERANK_DEPTH}`}
              value={rerank ? "on" : "off"}
              options={[
                { value: "off", label: "Off" },
                { value: "on", label: "On" },
              ]}
              onChange={(value) => setState({ rerank: value === "on" })}
            />
          </div>
        </div>
        <LineChart
          label="Held-out Recall@1, Recall@5 and MRR at every tenth training step, with BM25's Recall@1 as a flat line"
          series={series}
          xLabel={`training step 0 → ${STEPS}`}
          yLabel="share of held-out queries (MRR: mean of 1 / rank)"
          xDomain={[0, STEPS]}
          yDomain={[0, 1]}
          marker={{ x: step, label: `step ${step}` }}
        />
        <div className="metric-row">
          <Metric label="Recall@1" value={pct(pipeline.recall1)} tone="forward" />
          <Metric label="Recall@5" value={pct(pipeline.recall5)} />
          <Metric label="MRR" value={f2(pipeline.mrr)} />
          <Metric label="Scoring passes per query" value={String(passesPerQuery(rerank))} tone={rerank ? "loss" : undefined} />
        </div>
        <div className="source-table etr-score-table">
          <table className="tg-board" aria-label="Held-out scores for BM25, the untrained bi-encoder, and the bi-encoder at this step">
            <thead>
              <tr>
                <th scope="col">ranker</th>
                <th scope="col">Recall@1</th>
                <th scope="col">Recall@5</th>
                <th scope="col">MRR</th>
                {KINDS.map((kind, index) => (
                  <th scope="col" key={kind}>{`${HELD_OUT_KIND_LABEL[kind]} @1 (of ${KIND_COUNTS[index]})`}</th>
                ))}
                <th scope="col">passes per query</th>
              </tr>
            </thead>
            <tbody>
              {scoreRows.map((row) => (
                <tr key={row.id} className={row.current ? "is-leader" : ""}>
                  <th scope="row">{row.label}</th>
                  <td className="etr-databar" style={{ "--etr-fill": row.result.recall1 } as CSSProperties}>
                    <span>{pct(row.result.recall1)}</span>
                  </td>
                  <td className="etr-databar" style={{ "--etr-fill": row.result.recall5 } as CSSProperties}>
                    <span>{pct(row.result.recall5)}</span>
                  </td>
                  <td className="etr-databar" style={{ "--etr-fill": row.result.mrr } as CSSProperties}>
                    <span>{f2(row.result.mrr)}</span>
                  </td>
                  {row.kinds.map((hits, index) => (
                    <td
                      key={KINDS[index]}
                      className="etr-databar is-count"
                      style={{ "--etr-fill": hits / Math.max(1, KIND_COUNTS[index]) } as CSSProperties}
                    >
                      <span>{hits}</span>
                    </td>
                  ))}
                  <td>{row.passes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Recall@k here is the hit rate: the share of the {HELD_OUT_QUERIES.length} held-out queries with at least one
          relevant passage in the top k. Where the answer has a duplicate, either one counts. MRR averages 1 / rank of
          the first relevant passage. The bi-encoder stores every passage vector in advance, so a query costs one
          encoder pass and {PASSAGES.length} dot products; reranking adds one joint pass per reranked passage, and a
          cross-encoder alone would need {CROSS_ENCODER_ONLY_PASSES}.
        </p>
      </LabSurface>

      <LabSurface label="Embedding map" className="etr-map-card">
        <SurfaceHeading
          kicker={`Step ${step} · 2 of ${dim} dimensions · the axes hold ${pct(projection.captured[0] + projection.captured[1])} of the squared length`}
          title="A shadow of the space: bunched before training, spread after"
          aside={<span className="tg-badge">projection</span>}
        />
        <div className="etr-map-pair">
          <EmbeddingMap
            space={start}
            projection={startProjection}
            selectedQuery={queryIndex}
            step={0}
            meanCosine={startCosine}
            heading="Before training · step 0"
            quiet
          />
          <span className="etr-map-pair__arrow" aria-hidden="true">
            →
          </span>
          <EmbeddingMap
            space={space}
            projection={projection}
            selectedQuery={queryIndex}
            step={step}
            meanCosine={meanCosine}
            heading={step === 0 ? "Now · step 0, nothing trained yet" : `Now · step ${step}`}
          />
        </div>
        <ul className="etr-map-legend">
          <li>
            <i className="is-passage" /> passage, named by its topic
          </li>
          <li>
            <i className="is-query" /> held-out query, with a line to its labelled answer
          </li>
          <li>
            <i className="is-answer" /> the selected query's answer (and duplicate)
          </li>
        </ul>
        <div className="metric-row">
          <Metric label="Mean cosine between passages" value={f2(meanCosine)} tone="gradient" />
          <Metric label="Duplicate-pair cosine" value={f2(duplicateCosine(space.passages))} />
          <Metric label="Alignment (lower is closer)" value={f2(alignment(space))} tone="forward" />
          <Metric label="Uniformity (lower is more spread)" value={f2(uniformity(space.passages))} />
        </div>
        <p className="lab-note">
          A projection, not the space: each unit vector is drawn by its coordinates along the two directions that
          hold the most of the passages' squared length, so every point lands inside the circle. Two points that look
          close can still be far apart in the {dim - 2} directions not drawn. At step 0 the mean cosine is{" "}
          {f2(startCosine)}: every vector leans the same way because training starts from weights we gave a shared
          offset.
        </p>
      </LabSurface>

      <LabSurface label="Query inspector" className="etr-inspector-card">
        <SurfaceHeading
          kicker={`${HELD_OUT_KIND_LABEL[selected.kind]} · held out · labelled answer ${PASSAGES[answer].title}${twin >= 0 ? ` · duplicate ${PASSAGES[twin].title}` : ""}`}
          title={`“${selected.text}”`}
        />
        <label className="etr-query-select">
          <span>Query</span>
          <select value={selected.id} onChange={(event) => setState({ query: event.target.value })}>
            {KINDS.map((kind) => (
              <optgroup key={kind} label={HELD_OUT_KIND_LABEL[kind]}>
                {HELD_OUT_QUERIES.filter((entry) => entry.kind === kind).map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.text}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <div className="source-table etr-rank-table">
          <table className="tg-board" aria-label="Rank of the first relevant passage for this query under each ranker">
            <thead>
              <tr>
                <th scope="col">ranker</th>
                <th scope="col">first relevant</th>
                <th scope="col">top passage</th>
              </tr>
            </thead>
            <tbody>
              {rankRows.map((row) => (
                <tr key={row.id} className={(row.id === "rerank") === rerank && row.id !== "start" && row.id !== "bm25" ? "is-leader" : ""}>
                  <th scope="row">{row.label}</th>
                  <td className="etr-rank-cell">
                    <RankTrack rank={firstRelevantRank(row.ranking, relevant)} />
                    <span className="etr-rank-cell__value">{rankText(firstRelevantRank(row.ranking, relevant))}</span>
                  </td>
                  <td className="etr-top-passage">{row.ranking.length ? PASSAGES[row.ranking[0]].title : "no passage shares a word"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="source-table etr-watch-table">
          <table className="tg-board" aria-label={`Ranks of the ${TOPICS.find((entry) => entry.id === topic)?.label} passages and the current top three`}>
            <thead>
              <tr>
                <th scope="col">passage</th>
                <th scope="col">step 0</th>
                <th scope="col">{`step ${step}`}</th>
                <th scope="col" className="etr-move-head">{`step 0 ○ → ● step ${step}`}</th>
                <th scope="col">cosine</th>
                <th scope="col">BM25</th>
              </tr>
            </thead>
            <tbody>
              {watched.map(({ passage, index }) => (
                <tr key={passage.id} className={relevant.includes(index) ? "is-leader" : ""}>
                  <th scope="row">
                    {passage.title}
                    {index === answer && <em className="etr-tag">answer</em>}
                    {index === twin && <em className="etr-tag">duplicate, also relevant</em>}
                  </th>
                  <td>{rankText(rankOf(rankingStart, index))}</td>
                  <td>{rankText(rankOf(rankingNow, index))}</td>
                  <td className="etr-rank-cell">
                    <RankTrack rank={rankOf(rankingNow, index)} from={rankOf(rankingStart, index)} />
                  </td>
                  <td className="etr-databar" style={{ "--etr-fill": Math.max(0, cosineNow[index]) } as CSSProperties}>
                    <span>{f2(cosineNow[index])}</span>
                  </td>
                  <td>{rankText(rankOf(rankingBm25, index))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric
            label="Margin over the closest wrong passage"
            value={`${bestRelevant - bestWrong >= 0 ? "+" : ""}${f2(bestRelevant - bestWrong)}`}
            tone={bestRelevant >= bestWrong ? "forward" : "loss"}
          />
          <Metric label="Closest wrong passage" value={bestWrongIndex >= 0 ? PASSAGES[bestWrongIndex].title : "none"} />
          <Metric label={`Recall@1, first 4 of ${dim} numbers`} value={pct(firstFour)} />
          <Metric label="Training queries with this answer" value={String(trainMatches.length)} />
        </div>
        <p className="lab-note">
          Ranks count from #1, the closest passage. BM25 ranks only passages that share a keyword with the query.
          The reranker reorders the bi-encoder's top {RERANK_DEPTH} with a five-feature logistic model trained on the
          training queries: a stand-in for a cross-encoder, which reads the query and passage together.
          {pipelineRankings === reranked.rankings ? " Rerank top 5 is on, so Held-out retrieval scores the reranked order." : ""}
        </p>
      </LabSurface>
    </div>
  );
}
