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

/** Letters of a prompt as chips, with optional marks for particular positions. */
export function TokenStrip({
  tokens,
  period,
  marks = {},
  label,
}: {
  tokens: ReadonlyArray<string>;
  period?: number;
  marks?: Record<number, string>;
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
                  className={`${isSelected ? "is-selected" : ""} ${value < 0 ? "is-negative" : ""}`.trim()}
                  style={{ "--r": clamp(Math.abs(value), 0, 1) } as CSSProperties}
                  onClick={() => onSelect(layer, position)}
                />
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
  label,
}: {
  positions: ReadonlyArray<number>;
  tokens: ReadonlyArray<string>;
  intact: ReadonlyArray<number>;
  ablated: ReadonlyArray<number>;
  label: string;
}) {
  const width = 480;
  const height = 104;
  const bottom = 80;
  const slot = width / Math.max(1, positions.length);
  const bar = Math.max(3, slot * 0.34);
  return (
    <svg className="ci-bars" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      <line x1={0} x2={width} y1={bottom} y2={bottom} className="ci-bars__axis" />
      <line x1={0} x2={width} y1={bottom - 70} y2={bottom - 70} className="ci-bars__grid" />
      <text x={width - 2} y={bottom - 73} textAnchor="end" className="ci-bars__tick">p = 1</text>
      {positions.map((position, index) => {
        const x = index * slot + slot / 2;
        const a = intact[index] ?? 0;
        const b = ablated[index] ?? 0;
        return (
          <g key={position}>
            <rect x={x - bar - 1} y={bottom - a * 70} width={bar} height={a * 70} className="ci-bars__intact" />
            <rect x={x + 1} y={bottom - b * 70} width={bar} height={b * 70} className="ci-bars__ablated" />
            <text x={x} y={bottom + 12} textAnchor="middle" className="ci-bars__tick">{showToken(tokens[position])}</text>
            <text x={x} y={bottom + 22} textAnchor="middle" className="ci-bars__tick ci-bars__tick--faint">{position}</text>
          </g>
        );
      })}
    </svg>
  );
}
