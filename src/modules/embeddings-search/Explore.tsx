import { useMemo } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  approximateSearch,
  buildIndex,
  CORPUS,
  DIMENSIONS,
  embed,
  exactSearch,
  kmeans,
  norm,
  PRESETS,
  recallAtK,
  tokens,
  type Metric as ScoreMetric,
  type Scored,
} from "./vectors";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const f3 = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "—");
const f2 = (value: number) => (Number.isFinite(value) ? value.toFixed(2) : "—");

/* Protractor geometry: the query points along +x from the origin. */
const OX = 34;
const OY = 206;
const R = 178;
const LABEL_X = 272;

function Protractor({
  ranked,
  metric,
  topK,
  selectedId,
  queryLength,
  onSelect,
}: {
  ranked: readonly Scored[];
  metric: ScoreMetric;
  topK: number;
  selectedId: string;
  queryLength: number;
  onSelect: (id: string) => void;
}) {
  const maxLength = Math.max(1e-9, ...ranked.map((item) => item.length));
  const place = (item: Scored) => {
    const radius = metric === "cosine" ? R : (R * item.length) / maxLength;
    const theta = (item.angle * Math.PI) / 180;
    return { x: OX + radius * Math.cos(theta), y: OY - radius * Math.sin(theta), radius };
  };
  const labelled = ranked.filter((item) => item.rank <= topK || item.id === selectedId);
  const byAngle = [...labelled].sort((a, b) => a.angle - b.angle);
  const slot = (position: number) => 214 - position * 24;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const tickValue = (fraction: number) =>
    metric === "cosine" ? fraction : fraction * maxLength * queryLength;
  const selected = ranked.find((item) => item.id === selectedId);
  const selectedPoint = selected ? place(selected) : null;
  const zeroVector = queryLength === 0;

  return (
    <svg
      className="emb-protractor"
      viewBox="0 0 420 240"
      role="img"
      aria-label={
        zeroVector
          ? "No known words in the query, so the query vector is zero and no angle can be measured."
          : `Each passage drawn at its true angle to the query. ${ranked
              .slice(0, topK)
              .map((item) => `${item.title} at ${item.angle.toFixed(0)} degrees, ${metric} ${f3(item.score)}`)
              .join("; ")}.`
      }
    >
      <path className="emb-arc" d={`M ${OX + R} ${OY} A ${R} ${R} 0 0 0 ${OX} ${OY - R}`} />
      {[30, 60].map((degrees) => {
        const theta = (degrees * Math.PI) / 180;
        return (
          <g key={degrees}>
            <line
              className="emb-spoke"
              x1={OX}
              y1={OY}
              x2={OX + R * Math.cos(theta)}
              y2={OY - R * Math.sin(theta)}
            />
            <text className="emb-tick" x={OX + (R + 6) * Math.cos(theta)} y={OY - (R + 6) * Math.sin(theta)}>
              {degrees}°
            </text>
          </g>
        );
      })}
      <line className="emb-axis" x1={OX} y1={OY - R} x2={OX} y2={OY} />
      <text className="emb-tick" x={OX - 6} y={OY - R + 3} textAnchor="end">
        90°
      </text>
      <line className="emb-query" x1={OX} y1={OY} x2={OX + R + 10} y2={OY} />
      <text className="emb-query-label" x={OX + R + 12} y={OY - 5}>
        query
      </text>
      {ticks.map((fraction) => (
        <g key={fraction}>
          <line className="emb-axis" x1={OX + R * fraction} x2={OX + R * fraction} y1={OY} y2={OY + 5} />
          <text className="emb-tick" x={OX + R * fraction} y={OY + 16} textAnchor="middle">
            {tickValue(fraction).toFixed(2)}
          </text>
        </g>
      ))}
      <text className="emb-tick" x={OX} y={OY + 30}>
        {metric === "cosine" ? "x = cos θ (unit length)" : "x ∝ q·d = |q| |d| cos θ"}
      </text>
      {selectedPoint && !zeroVector && (
        <g className="emb-drop">
          <line x1={OX} y1={OY} x2={selectedPoint.x} y2={selectedPoint.y} />
          <line x1={selectedPoint.x} y1={selectedPoint.y} x2={selectedPoint.x} y2={OY} strokeDasharray="3 3" />
          <line className="emb-projection" x1={OX} y1={OY} x2={selectedPoint.x} y2={OY} />
        </g>
      )}
      {!zeroVector &&
        ranked.map((item) => {
          const point = place(item);
          const isTop = item.rank <= topK;
          return (
            <circle
              key={item.id}
              cx={point.x}
              cy={point.y}
              r={item.id === selectedId ? 5.2 : isTop ? 4 : 2.6}
              className={`emb-point${isTop ? " is-top" : ""}${item.id === selectedId ? " is-selected" : ""}${
                item.id === "notes" ? " is-user" : ""
              }`}
              onClick={() => onSelect(item.id)}
            >
              <title>{`${item.title}: θ ${item.angle.toFixed(1)}°, cos ${f3(item.cosine)}, dot ${f3(item.dot)}`}</title>
            </circle>
          );
        })}
      {!zeroVector &&
        byAngle.map((item, position) => {
          const point = place(item);
          const y = slot(byAngle.length - 1 - position);
          return (
            <g key={item.id} className={`emb-label${item.id === selectedId ? " is-selected" : ""}`}>
              <line className="emb-leader" x1={point.x} y1={point.y} x2={LABEL_X - 4} y2={y - 4} />
              <text x={LABEL_X} y={y}>
                {`#${item.rank} ${item.title}`}
              </text>
              <text className="emb-label-value" x={LABEL_X} y={y + 10}>
                {`θ ${item.angle.toFixed(1)}° · ${metric} ${f3(item.score)}`}
              </text>
            </g>
          );
        })}
      {zeroVector && (
        <text className="emb-query-label" x={OX + 20} y={OY - R / 2}>
          No known words: the query vector is all zeros.
        </text>
      )}
    </svg>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const query = asString(state, "query", PRESETS[0].query);
  const userText = asString(state, "userText", "");
  const metric: ScoreMetric = asString(state, "metric", "cosine") === "dot" ? "dot" : "cosine";
  const topK = clamp(Math.round(asNumber(state, "topK", 3)), 1, 5);
  const cellCount = clamp(Math.round(asNumber(state, "cells", 3)), 1, 4);
  const probe = clamp(Math.round(asNumber(state, "probe", 1)), 1, cellCount);

  const index = useMemo(() => buildIndex(userText), [userText]);
  const ranked = useMemo(() => exactSearch(index, query, metric), [index, metric, query]);
  const dotRanks = useMemo(
    () => new Map(exactSearch(index, query, "dot").map((item) => [item.id, item.rank])),
    [index, query],
  );
  const queryVector = useMemo(() => embed(query), [query]);
  const queryTokens = useMemo(() => tokens(query), [query]);
  const queryLength = norm(queryVector);

  const selectedRaw = asString(state, "selected", "");
  const selected = ranked.find((item) => item.id === selectedRaw) ?? ranked[0];

  const cells = useMemo(() => kmeans(index.map((item) => item.unit), cellCount), [cellCount, index]);
  const effectiveK = Math.min(topK, index.length);
  const exactTop = ranked.slice(0, effectiveK).map((item) => item.id);
  const approx = useMemo(
    () => approximateSearch(index, cells, query, metric, probe, effectiveK),
    [cells, effectiveK, index, metric, probe, query],
  );
  const recall = recallAtK(exactTop, approx.top);
  const cellsUsed = cells.centroids.length;

  const presetValue = PRESETS.find((preset) => preset.query === query)?.query ?? "";
  const known = queryTokens.filter((token) => token.entry !== null);
  const keywordLeader = [...ranked].sort((a, b) => a.keywordRank - b.keywordRank)[0];
  const products = selected ? queryVector.map((value, dimension) => value * selected.vector[dimension]) : [];
  const maxComponent = Math.max(0.01, ...queryVector, ...(selected?.vector ?? []));

  const select = (id: string) => setState({ selected: id });

  return (
    <div className="tg-lab tg-lab--hero emb-lab">
      <LabSurface label="Query and vectors" className="emb-query-card">
        <SurfaceHeading
          kicker={`${CORPUS.length} harbor passages${userText.trim() ? " + your passage" : ""} · 8 named dimensions`}
          title="A query becomes a vector before anything is ranked"
          aside={<span className="tg-badge">hand-set encoder</span>}
        />
        <div className="emb-query-grid">
          <div className="emb-query-inputs">
            <label className="prompt-input">
              <span>Query</span>
              <input aria-label="Query" value={query} onChange={(event) => setState({ query: event.target.value })} />
            </label>
            <div className="emb-preset-control">
              <SegmentedControl
                label="Preset query"
                value={presetValue}
                options={PRESETS.map((preset) => ({ value: preset.query, label: preset.label }))}
                onChange={(value) => setState({ query: value, selected: "" })}
              />
            </div>
            <div className="emb-tokens" aria-label="How the encoder reads the query">
              {queryTokens.length === 0 && <span className="emb-token is-ignored">empty query</span>}
              {queryTokens.map((token, position) => (
                <span
                  key={`${token.word}-${position}`}
                  className={`emb-token${token.entry ? "" : " is-ignored"}`}
                  title={token.entry ? `read as "${token.entry}"` : "not in the lexicon: ignored"}
                >
                  {token.word}
                  <small>{token.entry ? (token.entry === token.word ? "known" : `→ ${token.entry}`) : "ignored"}</small>
                </span>
              ))}
            </div>
            <label className="tg-editor emb-notes">
              <span>Your passage (added to the index as “Your notes”)</span>
              <textarea
                aria-label="Your passage"
                rows={2}
                placeholder="Type a sentence, e.g. The ferry café closes at 17:00."
                value={userText}
                onChange={(event) => setState({ userText: event.target.value })}
              />
            </label>
          </div>
          <div className="emb-qvec" role="img" aria-label={`Query vector: ${DIMENSIONS.map((name, i) => `${name} ${f2(queryVector[i])}`).join(", ")}. Length ${f3(queryLength)}.`}>
            <span className="emb-qvec-title">
              q = mean of {known.length} known word vector{known.length === 1 ? "" : "s"}
            </span>
            {DIMENSIONS.map((name, dimension) => (
              <div key={name} className="emb-qvec-row">
                <span>{name}</span>
                <i>
                  <b style={{ width: `${Math.min(100, (queryVector[dimension] / maxComponent) * 100)}%` }} />
                </i>
                <code>{f2(queryVector[dimension])}</code>
              </div>
            ))}
            <span className="emb-qvec-foot">|q| = {f3(queryLength)}</span>
          </div>
        </div>
        <p className="lab-note">
          Each dimension is a topic we named and each known word has authored loadings on them. Unknown words add
          nothing. A trained encoder learns hundreds of unnamed dimensions from data instead; the arithmetic after
          this point is the same.
        </p>
      </LabSurface>

      <LabSurface label="Angle to the query" className="emb-angle-card">
        <SurfaceHeading
          kicker={metric === "cosine" ? "Every vector scaled to length 1" : "Raw lengths kept"}
          title="Nearest means furthest along the query direction"
        />
        <div className="emb-angle-grid">
          <Protractor
            ranked={ranked}
            metric={metric}
            topK={topK}
            selectedId={selected?.id ?? ""}
            queryLength={queryLength}
            onSelect={select}
          />
          <div className="emb-angle-side">
            <div className="emb-metric-control">
              <SegmentedControl
                label="Score"
                value={metric}
                options={[
                  { value: "cosine", label: "cosine" },
                  { value: "dot", label: "dot product" },
                ]}
                onChange={(value) => setState({ metric: value })}
              />
            </div>
            {selected && (
              <div className="metric-row emb-angle-metrics">
                <Metric label={`${selected.title} θ`} value={`${selected.angle.toFixed(1)}°`} />
                <Metric label="cos θ" value={f3(selected.cosine)} tone="forward" />
                <Metric label="|d|" value={f3(selected.length)} />
              </div>
            )}
            <ul className="emb-legend">
              <li>
                <i className="is-top" /> top-{topK} by {metric === "cosine" ? "cosine" : "dot product"}
              </li>
              <li>
                <i /> rest of the index
              </li>
              <li>
                <i className="is-selected" /> selected passage, with its projection on the query axis
              </li>
            </ul>
          </div>
        </div>
        <p className="lab-note">
          Each dot sits at its true angle θ to the query. Only that angle is real in this picture — the angles
          between passages are not drawn. {metric === "cosine"
            ? "On the unit arc the x-coordinate is cos θ, so rank is the order along the query axis, right to left."
            : "With raw lengths, a dot's distance from the origin is |d|, and its x-coordinate is |d| cos θ, which ranks exactly like q·d."}
        </p>
      </LabSurface>

      <LabSurface label="Ranked neighbours" className="emb-rank-card">
        <SurfaceHeading kicker={`Exact search · all ${index.length} vectors scored`} title="Top-k by the chosen score" />
        <RangeControl
          label="Top-k"
          min={1}
          max={5}
          step={1}
          value={topK}
          format={(value) => `${value}`}
          onChange={(value) => setState({ topK: value })}
        />
        <ol className="emb-rank-list">
          {ranked.slice(0, Math.max(topK, 6)).map((item) => (
            <li key={item.id} className={`${item.rank <= topK ? "is-top" : ""}${item.id === selected?.id ? " is-selected" : ""}`}>
              <button type="button" aria-pressed={item.id === selected?.id} onClick={() => select(item.id)}>
                <span className="emb-rank-no">#{item.rank}</span>
                <strong>{item.title}</strong>
                <code>{f3(item.score)}</code>
                <small>
                  {metric === "cosine" ? `dot ${f3(item.dot)}` : `cos ${f3(item.cosine)}`} · shared words{" "}
                  {item.shared.length ? item.shared.join(", ") : "none"} · keyword rank #{item.keywordRank}
                </small>
              </button>
            </li>
          ))}
        </ol>
        <p className="lab-note">
          Keyword rank orders passages by shared content words only. It currently puts{" "}
          <strong>{keywordLeader?.title ?? "—"}</strong> first
          {keywordLeader && keywordLeader.shared.length === 0 ? ", with zero shared words — a tie broken by list order" : ""}.
        </p>
      </LabSurface>

      <LabSurface label="Worked similarity" className="emb-worked-card">
        <SurfaceHeading
          kicker={selected ? `q against ${selected.title}` : "No passage"}
          title="Multiply, add, then divide by the lengths"
        />
        {selected && (
          <>
            <p className="emb-passage">“{selected.text}”</p>
            <div className="emb-worked-table">
              <table className="tg-board">
                <thead>
                  <tr>
                    <th scope="col">dimension</th>
                    <th scope="col">q</th>
                    <th scope="col">d</th>
                    <th scope="col">q × d</th>
                  </tr>
                </thead>
                <tbody>
                  {DIMENSIONS.map((name, dimension) => (
                    <tr key={name} className={products[dimension] > 0 ? "is-leader" : ""}>
                      <th scope="row">{name}</th>
                      <td className={queryVector[dimension] > 0 ? "is-live" : ""}>{f2(queryVector[dimension])}</td>
                      <td className={selected.vector[dimension] > 0 ? "is-live" : ""}>{f2(selected.vector[dimension])}</td>
                      <td>{products[dimension] > 0 ? f3(products[dimension]) : "0"}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="row">total</th>
                    <td>|q| {f3(queryLength)}</td>
                    <td>|d| {f3(selected.length)}</td>
                    <td>Σ {f3(selected.dot)}</td>
                  </tr>
                </tfoot>
              </table>
            </div>
            <div className="emb-formulas">
              <FormulaWithValues
                label="Dot product"
                expression={`q·d = Σ qᵢ dᵢ = ${f3(selected.dot)}`}
                result={`rank #${dotRanks.get(selected.id) ?? "—"} by dot`}
                tone="forward"
                detail="Grows with the angle match and with both lengths."
              />
              <FormulaWithValues
                label="Cosine similarity"
                expression={`${f3(selected.dot)} / (${f3(queryLength)} × ${f3(selected.length)}) = ${f3(selected.cosine)}`}
                result={`θ = ${selected.angle.toFixed(1)}°`}
                tone="attention"
                detail="Dividing by both lengths leaves only the direction."
              />
            </div>
          </>
        )}
        <p className="lab-note">
          A zero in q means the query says nothing about that topic, so the passage's value there adds nothing. If
          every vector were stored at length 1, dot product and cosine would be the same number.
        </p>
      </LabSurface>

      <LabSurface label="Exact versus approximate" className="emb-ann-card">
        <SurfaceHeading
          kicker={`IVF index · ${cellsUsed} cell${cellsUsed === 1 ? "" : "s"} from k-means · probe ${Math.min(probe, cellsUsed)}`}
          title="Score only the cells nearest the query, and sometimes miss"
        />
        <div className="emb-ann-controls">
          <RangeControl
            label="Cells"
            min={1}
            max={4}
            step={1}
            value={cellCount}
            format={(value) => `${value}`}
            onChange={(value) => setState({ cells: value, probe: Math.min(probe, value) })}
          />
          {cellCount > 1 ? (
            <RangeControl
              label="Cells probed"
              min={1}
              max={cellCount}
              step={1}
              value={probe}
              format={(value) => `${value} of ${cellCount}`}
              onChange={(value) => setState({ probe: value })}
            />
          ) : (
            <p className="emb-probe-fixed">Cells probed: 1 of 1 — one cell is exact search.</p>
          )}
        </div>
        <div className="emb-cells" style={{ gridTemplateColumns: `repeat(${cellsUsed}, minmax(0, 1fr))` }}>
          {cells.centroids.map((centroid, cell) => {
            const probed = approx.probed.includes(cell);
            const order = approx.cellOrder.indexOf(cell) + 1;
            const members = index.filter((_, position) => cells.assignment[position] === cell);
            const leading = DIMENSIONS.map((name, dimension) => ({ name, value: centroid[dimension] }))
              .sort((a, b) => b.value - a.value)
              .slice(0, 2)
              .filter((entry) => entry.value > 0.05)
              .map((entry) => entry.name)
              .join(" + ");
            return (
              <div key={cell} className={`emb-cell${probed ? " is-probed" : ""}`}>
                <header>
                  <strong>Cell {cell + 1}</strong>
                  <span>{probed ? `probed · #${order}` : `skipped · #${order}`}</span>
                </header>
                <small>centroid leans {leading || "nowhere"}</small>
                <ul>
                  {members.map((member) => {
                    const inExact = exactTop.includes(member.id);
                    const found = approx.top.includes(member.id);
                    return (
                      <li key={member.id} className={`${inExact ? "is-exact" : ""}${inExact && !found ? " is-missed" : ""}`}>
                        {member.title}
                        {inExact && <em>{found ? "top-k · found" : "top-k · missed"}</em>}
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}
        </div>
        <div className="metric-row">
          <Metric label="Vectors scored" value={`${approx.scoredIds.length} of ${index.length}`} tone="forward" />
          <Metric label={`Recall@${effectiveK}`} value={recall.toFixed(2)} tone={recall < 1 ? "loss" : "forward"} />
          <Metric label="Approximate top-k" value={approx.top.map((id) => index.find((item) => item.id === id)?.title ?? id).join(", ") || "none"} />
        </div>
        <p className="lab-note">
          Cells come from spherical k-means on the unit vectors. The query opens the cells whose centroids are most
          similar to it and scores only their members. Recall@k is the share of the exact top-k that survived.
        </p>
      </LabSurface>
    </div>
  );
}
