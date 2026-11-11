import type { CSSProperties } from "react";
import { showToken } from "./vocabulary";

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * One head's causal attention matrix for a short prompt. Rows are queries,
 * columns keys. Two guides outline, for every query in the second repeat,
 * the key an induction head would read (the token after the previous
 * occurrence, a solid outline) and the previous letter (a dashed outline).
 */
export function AttentionGrid({
  tokens,
  weights,
  period,
  label,
}: {
  tokens: ReadonlyArray<string>;
  /** [sequence x sequence], query-major. */
  weights: ArrayLike<number>;
  period: number | null;
  label: string;
}) {
  const n = tokens.length;
  const cell = 14;
  const margin = 16;
  const size = margin + n * cell;
  const cells = [];
  const guides = [];
  for (let query = 0; query < n; query += 1) {
    for (let key = 0; key <= query; key += 1) {
      const value = weights[query * n + key] ?? 0;
      cells.push(
        <rect
          key={`${query}-${key}`}
          x={margin + key * cell}
          y={margin + query * cell}
          width={cell - 1}
          height={cell - 1}
          className="ci-attn__cell"
          style={{ "--w": clamp(Math.sqrt(value), 0, 1) } as CSSProperties}
        >
          <title>{`query ${query} (${showToken(tokens[query])}) → key ${key} (${showToken(tokens[key])}): ${(value * 100).toFixed(1)}%`}</title>
        </rect>,
      );
    }
    if (period !== null && query >= period) {
      const induction = query - period + 1;
      guides.push(
        <rect
          key={`i-${query}`}
          x={margin + induction * cell - 1}
          y={margin + query * cell - 1}
          width={cell + 1}
          height={cell + 1}
          className="ci-attn__guide ci-attn__guide--induction"
        />,
      );
      guides.push(
        <rect
          key={`p-${query}`}
          x={margin + (query - 1) * cell - 1}
          y={margin + query * cell - 1}
          width={cell + 1}
          height={cell + 1}
          className="ci-attn__guide ci-attn__guide--previous"
        />,
      );
    }
  }
  return (
    <svg className="ci-attn" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={label}>
      {tokens.map((token, index) => (
        <g key={index} aria-hidden="true">
          <text x={margin + index * cell + cell / 2 - 0.5} y={11} className="ci-attn__label" textAnchor="middle">
            {showToken(token)}
          </text>
          <text x={8} y={margin + index * cell + cell / 2 + 3} className="ci-attn__label" textAnchor="middle">
            {showToken(token)}
          </text>
        </g>
      ))}
      {cells}
      {guides}
    </svg>
  );
}

/**
 * One row of the attention matrix drawn on the prompt itself: arcs from the last letter back to every
 * letter the chosen head attends to, as thick as the weight. The two keys the grid outlines are outlined
 * here too, so "the arc lands on the green box" is the induction pattern, seen directly.
 */
export function AttentionArcs({
  tokens,
  weights,
  period,
  head,
}: {
  tokens: ReadonlyArray<string>;
  /** [sequence x sequence], query-major. */
  weights: ArrayLike<number>;
  period: number | null;
  head: string;
}) {
  const n = tokens.length;
  const slot = 26;
  const left = 10;
  const width = left * 2 + n * slot;
  const boxTop = 118;
  const height = boxTop + 40;
  const query = n - 1;
  const centre = (index: number) => left + index * slot + slot / 2;
  const row = Array.from({ length: n }, (_, key) => weights[query * n + key] ?? 0);
  const induction = period !== null && query >= period ? query - period + 1 : null;
  const previous = query - 1;
  const arcs = row
    .map((weight, key) => ({ weight, key }))
    .filter((entry) => entry.key !== query && entry.weight >= 0.02)
    .sort((a, b) => a.weight - b.weight);
  const strongest = [...arcs].sort((a, b) => b.weight - a.weight).slice(0, 2);
  const apex = (key: number) => boxTop - 14 - Math.min(92, 18 + (query - key) * 4.2);
  const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
  const where = (key: number) =>
    key === induction ? "the letter after the previous occurrence" : key === previous ? "the previous letter" : `position ${key}`;
  return (
    <svg
      className="ci-arcs"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${head}, looking from the last letter ${showToken(tokens[query])} at position ${query}: ${
        strongest.length
          ? strongest.map((entry) => `${percent(entry.weight)} of its attention on ${where(entry.key)} (${showToken(tokens[entry.key])})`).join(", ")
          : "no earlier letter gets 2% or more"
      }${row[query] >= 0.02 ? `, ${percent(row[query])} on itself` : ""}.`}
    >
      {arcs.map(({ weight, key }) => {
        const x1 = centre(query);
        const x2 = centre(key);
        const top = apex(key);
        return (
          <path
            key={key}
            className="ci-arcs__arc"
            d={`M${x1},${boxTop - 3} C${x1},${top - 10} ${x2},${top - 10} ${x2},${boxTop - 5}`}
            style={{ strokeWidth: 1 + weight * 8, opacity: 0.3 + 0.7 * Math.sqrt(weight) }}
          />
        );
      })}
      {arcs
        .filter((entry) => entry.weight >= 0.08)
        .map(({ key }) => (
          <path
            key={`tip-${key}`}
            className="ci-arcs__tip"
            d={`M${centre(key) - 4.5},${boxTop - 9} L${centre(key)},${boxTop - 2} L${centre(key) + 4.5},${boxTop - 9} Z`}
          />
        ))}
      {strongest.map(({ weight, key }) => {
        const x = (centre(query) + centre(key)) / 2;
        const y = apex(key) + (boxTop - apex(key)) * 0.22 - 6;
        return (
          <text key={`label-${key}`} className="ci-arcs__value" x={x} y={Math.max(11, y)} textAnchor="middle">
            {percent(weight)}
          </text>
        );
      })}
      {row[query] >= 0.02 && (
        <text className="ci-arcs__value" x={centre(query)} y={boxTop - 8} textAnchor="middle">
          {percent(row[query])}
        </text>
      )}
      {tokens.map((token, index) => (
        <g key={index} aria-hidden="true">
          <rect
            className={[
              "ci-arcs__box",
              index === query ? "is-query" : "",
              index === induction ? "is-induction" : "",
              index === previous ? "is-previous" : "",
              period && Math.floor(index / period) % 2 === 1 ? "is-alt" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            x={left + index * slot + 1.5}
            y={boxTop}
            width={slot - 3}
            height={24}
            rx={3}
          />
          <text className="ci-arcs__letter" x={centre(index)} y={boxTop + 16.5} textAnchor="middle">
            {showToken(token)}
          </text>
          <text className="ci-arcs__index" x={centre(index)} y={boxTop + 36} textAnchor="middle">
            {index}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Letters of a prompt as chips, with optional marks for particular positions. */
export function TokenStrip({
  tokens,
  period,
  marks = {},
  picked,
  label,
}: {
  tokens: ReadonlyArray<string>;
  period?: number;
  marks?: Record<number, string>;
  /** A position to point at, for example the column of the selected patching cell. */
  picked?: number;
  label: string;
}) {
  return (
    <ol className="ci-tokens" aria-label={label}>
      {tokens.map((token, index) => (
        <li
          key={index}
          className={[
            period && Math.floor(index / period) % 2 === 1 ? "is-alt" : "",
            marks[index] ? `is-${marks[index]}` : "",
            picked === index ? "is-picked" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          title={`position ${index}`}
        >
          <span>{showToken(token)}</span>
          <small>{index}</small>
        </li>
      ))}
    </ol>
  );
}

/**
 * Activation-patching recovery as a layer x position grid. Fill encodes the
 * recovered share of the metric on a diverging scale clipped to [-1, 1]; the
 * exact value is in each cell's label and in the readout below the grid.
 */
export function PatchGrid({
  tokens,
  recovery,
  selected,
  onSelect,
  rowLabels,
  differing,
}: {
  tokens: ReadonlyArray<string>;
  recovery: ReadonlyArray<ReadonlyArray<number>>;
  selected: { layer: number; position: number };
  onSelect: (layer: number, position: number) => void;
  rowLabels: ReadonlyArray<string>;
  differing: ReadonlyArray<number>;
}) {
  // A three-digit number needs a cell about 1.2rem wide; the cached speaker sweep is too long for that.
  const showNumbers = tokens.length <= 30;
  return (
    <div className="ci-patch" style={{ "--columns": tokens.length } as CSSProperties}>
      <div className="ci-patch__grid" role="grid" aria-label="Recovery by layer and position">
        <div className="ci-patch__row" role="row">
          <span role="columnheader" aria-label="Layer" />
          {tokens.map((token, index) => (
            <span
              key={`h-${index}`}
              role="columnheader"
              aria-label={`position ${index}`}
              className={`ci-patch__column ${differing.includes(index) ? "is-differing" : ""}`}
            >
              {showToken(token)}
            </span>
          ))}
        </div>
        {recovery.map((row, layer) => (
          <div className="ci-patch__row" role="row" key={layer}>
            <span className="ci-patch__row-label" role="rowheader">{rowLabels[layer]}</span>
            {row.map((value, position) => {
              const isSelected = selected.layer === layer && selected.position === position;
              return (
                <button
                  type="button"
                  role="gridcell"
                  key={position}
                  aria-selected={isSelected}
                  aria-label={`${rowLabels[layer]}, position ${position} (${showToken(tokens[position])}): recovery ${(value * 100).toFixed(0)} percent`}
                  className={[
                    isSelected ? "is-selected" : "",
                    value < 0 ? "is-negative" : "",
                    Math.abs(value) >= 0.55 ? "is-strong" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={{ "--r": clamp(Math.abs(value), 0, 1) } as CSSProperties}
                  onClick={() => onSelect(layer, position)}
                >
                  {showNumbers && Math.abs(value) >= 0.095 ? (
                    <span aria-hidden="true">{Math.abs(value) >= 9.995 ? "≫" : Math.round(value * 100)}</span>
                  ) : null}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Per-position probability of the correct next letter, intact vs with ablations. */
export function PositionBars({
  positions,
  tokens,
  intact,
  ablated,
  intactMean,
  ablatedMean,
  label,
}: {
  positions: ReadonlyArray<number>;
  tokens: ReadonlyArray<string>;
  intact: ReadonlyArray<number>;
  ablated: ReadonlyArray<number>;
  /** Copy accuracy, the mean of each set of bars, drawn as a line across them. */
  intactMean: number;
  ablatedMean: number;
  label: string;
}) {
  const width = 600;
  const height = 142;
  const bottom = 112;
  const left = 4;
  const right = 112;
  const rise = 84;
  const slot = (width - left - right) / Math.max(1, positions.length);
  const bar = Math.max(3, slot * 0.34);
  const moved = Math.abs(intactMean - ablatedMean) > 5e-4;
  // Keep the two mean labels from printing on top of each other when the means are close.
  const intactY = bottom - intactMean * rise;
  const rawAblatedY = bottom - ablatedMean * rise;
  const ablatedY = Math.abs(rawAblatedY - intactY) < 12 ? intactY + (rawAblatedY >= intactY ? 12 : -12) : rawAblatedY;
  return (
    <svg className="ci-bars" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      <line x1={left} x2={width - right} y1={bottom} y2={bottom} className="ci-bars__axis" />
      <line x1={left} x2={width - right} y1={bottom - rise} y2={bottom - rise} className="ci-bars__grid" />
      <text x={left} y={bottom - rise - 9} className="ci-bars__tick">
        probability of the right next letter · dashed line at 1
      </text>
      <line x1={left} x2={width - right + 4} y1={intactY} y2={intactY} className="ci-bars__mean ci-bars__mean--intact" />
      <text x={width - right + 7} y={intactY + 3.5} className="ci-bars__mean-label">
        intact mean
      </text>
      {moved && (
        <>
          <line
            x1={left}
            x2={width - right + 4}
            y1={rawAblatedY}
            y2={rawAblatedY}
            className="ci-bars__mean ci-bars__mean--ablated"
          />
          <text x={width - right + 7} y={ablatedY + 3.5} className="ci-bars__mean-label ci-bars__mean-label--ablated">
            mean, heads off
          </text>
        </>
      )}
      {positions.map((position, index) => {
        const x = left + index * slot + slot / 2;
        const a = intact[index] ?? 0;
        const b = ablated[index] ?? 0;
        return (
          <g key={position}>
            <rect x={x - bar - 1} y={bottom - a * rise} width={bar} height={a * rise} className="ci-bars__intact" />
            <rect x={x + 1} y={bottom - b * rise} width={bar} height={b * rise} className="ci-bars__ablated" />
            <text x={x} y={bottom + 13} textAnchor="middle" className="ci-bars__tick">{showToken(tokens[position])}</text>
            <text x={x} y={bottom + 26} textAnchor="middle" className="ci-bars__tick ci-bars__tick--faint">{position}</text>
          </g>
        );
      })}
    </svg>
  );
}

/** Before-and-after bars for one head's attention on the induction key: an outline at the intact value, a fill at the ablated one. */
export function StripeBars({ rows }: { rows: ReadonlyArray<{ name: string; intact: number; ablated: number }> }) {
  return (
    <div
      className="ci-stripe"
      role="img"
      aria-label={`Attention on the induction key. ${rows
        .map((row) => `${row.name}: ${row.intact.toFixed(2)} intact, ${row.ablated.toFixed(2)} with the chosen heads off`)
        .join(". ")}.`}
    >
      {rows.map((row) => (
        <div className="ci-stripe__row" key={row.name} aria-hidden="true">
          <span>{row.name}</span>
          <i>
            <b className="ci-stripe__ablated" style={{ width: `${clamp(row.ablated, 0, 1) * 100}%` }} />
            <b className="ci-stripe__intact" style={{ width: `${clamp(row.intact, 0, 1) * 100}%` }} />
          </i>
        </div>
      ))}
      <p aria-hidden="true">
        <i className="ci-stripe__key ci-stripe__key--intact" /> outline: every head on
        <i className="ci-stripe__key ci-stripe__key--ablated" /> filled: with the chosen heads off
      </p>
    </div>
  );
}
