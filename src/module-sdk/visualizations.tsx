import type { CSSProperties, ReactNode } from "react";
import { curveMonotoneX, line, linkHorizontal } from "d3-shape";
import { formatPercent } from "@app/lib/math";
import { CardInfoButton, useCardInfo, type CardInfo } from "./card-info";

export interface PlotPoint {
  x: number;
  y: number;
}

export function computationLinkPath(source: PlotPoint, target: PlotPoint) {
  return (
    linkHorizontal<
      { source: PlotPoint; target: PlotPoint },
      PlotPoint
    >()
      .x((point) => point.x)
      .y((point) => point.y)({ source, target }) ?? ""
  );
}

export function smoothLinePath(points: ReadonlyArray<PlotPoint>) {
  return (
    line<PlotPoint>()
      .x((point) => point.x)
      .y((point) => point.y)
      .curve(curveMonotoneX)(Array.from(points)) ?? ""
  );
}

export function LabSurface({
  children,
  className = "",
  label,
  info,
}: {
  children: ReactNode;
  className?: string;
  label: string;
  /** Overrides the explanation registered for this label in the module's card-info.ts. */
  info?: CardInfo;
}) {
  const registered = useCardInfo(label);
  const explanation = info ?? registered;
  return (
    <section
      className={`lab-surface ${explanation ? "has-card-info" : ""} ${className}`.trim()}
      aria-label={label}
      data-lab-surface={label}
    >
      {explanation && <CardInfoButton info={explanation} label={label} />}
      {children}
    </section>
  );
}

export function SurfaceHeading({
  kicker,
  title,
  icon,
  aside,
}: {
  kicker: string;
  title: string;
  icon?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="surface-heading">
      <div>
        <span>{kicker}</span>
        <h3>{title}</h3>
      </div>
      {aside ?? icon}
    </div>
  );
}

export function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "forward" | "gradient" | "loss";
}) {
  return (
    <div className={`metric ${tone ? `metric--${tone}` : ""}`.trim()}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function RangeControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = String,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  const progress = max > min ? ((value - min) / (max - min)) * 100 : 0;

  return (
    <label className="range-control">
      <span>
        {label}
        <output>{format(value)}</output>
      </span>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--range-progress": `${progress}%` } as CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function SegmentedControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <fieldset className="segmented">
      <legend>{label}</legend>
      <div>
        {options.map((option) => (
          <button
            type="button"
            key={option.value}
            aria-pressed={option.value === value}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function VectorChip({
  label,
  values,
  tone = "attention",
  editable = false,
  onChange,
  precision = 2,
}: {
  label: string;
  values: ReadonlyArray<number>;
  tone?: "forward" | "gradient" | "attention";
  editable?: boolean;
  onChange?: (index: number, value: number) => void;
  precision?: number;
}) {
  return (
    <div className={`vector-chip vector-chip--${tone}`}>
      <span>{label}</span>
      <div aria-label={`${label} vector`}>
        <i>[</i>
        {values.map((value, index) =>
          editable ? (
            <label key={index}>
              <span className="sr-only">{`${label} component ${index + 1}`}</span>
              <input
                type="number"
                step="0.05"
                value={Number(value.toFixed(precision))}
                onChange={(event) => onChange?.(index, Number(event.target.value))}
              />
            </label>
          ) : (
            <code key={index}>{value.toFixed(precision)}</code>
          ),
        )}
        <i>]</i>
      </div>
    </div>
  );
}

export function FormulaWithValues({
  label,
  expression,
  result,
  tone = "attention",
  detail,
}: {
  label: string;
  expression: ReactNode;
  result: string;
  tone?: "forward" | "gradient" | "attention" | "loss";
  detail?: string;
}) {
  return (
    <div className={`formula-live formula-live--${tone}`}>
      <span>{label}</span>
      <code>{expression}</code>
      <strong>{result}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

export function StepThroughController({
  label,
  steps,
  current,
  onChange,
}: {
  label: string;
  steps: ReadonlyArray<string>;
  current: number;
  onChange: (step: number) => void;
}) {
  return (
    <div className="viz-step-controller" aria-label={label}>
      {steps.map((step, index) => (
        <button
          type="button"
          key={step}
          aria-current={current === index ? "step" : undefined}
          className={current === index ? "is-current" : current > index ? "is-done" : ""}
          onClick={() => onChange(index)}
        >
          <span>{index + 1}</span>
          {step}
        </button>
      ))}
    </div>
  );
}

export function ArcDiagram({
  tokens,
  weights,
  queryIndex,
  inspectedIndex,
  onQueryChange,
  label,
}: {
  tokens: ReadonlyArray<string>;
  weights: ReadonlyArray<number>;
  queryIndex: number;
  inspectedIndex?: number;
  onQueryChange: (index: number) => void;
  label: string;
}) {
  const denominator = Math.max(1, tokens.length - 1);
  const pointAt = (index: number) => 45 + (index * 810) / denominator;

  return (
    <div className="viz-arc-diagram">
      <svg viewBox="0 0 900 190" preserveAspectRatio="none" aria-hidden="true">
        {tokens.map((_, index) => {
          if (index === queryIndex) return null;
          const start = pointAt(queryIndex);
          const end = pointAt(index);
          const height = 164 - Math.min(138, Math.abs(end - start) * 0.44);
          const weight = weights[index] ?? 0;

          return (
            <path
              key={index}
              d={`M ${start} 170 Q ${(start + end) / 2} ${height} ${end} 170`}
              className={inspectedIndex === index ? "is-inspected" : ""}
              style={{
                opacity: 0.12 + weight * 2.4,
                strokeWidth: 1 + weight * 13,
              }}
            />
          );
        })}
      </svg>
      <div className="viz-arc-tokens" aria-label={label}>
        {tokens.map((token, index) => (
          <button
            type="button"
            key={`${token}-${index}`}
            className={[
              index === queryIndex ? "is-query" : "",
              index === inspectedIndex ? "is-inspected" : "",
            ].filter(Boolean).join(" ")}
            aria-label={`${token}, attention weight ${formatPercent(weights[index] ?? 0)}${
              index === queryIndex ? ", current query" : ""
            }`}
            aria-pressed={index === queryIndex}
            onClick={() => onQueryChange(index)}
          >
            <span>{token}</span>
            <i style={{ "--token-weight": weights[index] ?? 0 } as CSSProperties} />
          </button>
        ))}
      </div>
    </div>
  );
}

export interface StageFlowStage {
  id: string;
  name: string;
  detail?: string;
}

/**
 * A left-to-right pipeline of named stages. Used wherever a lesson needs the
 * learner to see which part of a loop is running right now.
 */
export function StageFlow({
  label,
  stages,
  current,
  onSelect,
  loops = false,
}: {
  label: string;
  stages: ReadonlyArray<StageFlowStage>;
  current: number;
  onSelect?: (index: number) => void;
  loops?: boolean;
}) {
  return (
    <ol
      className={`viz-stage-flow ${loops ? "viz-stage-flow--loops" : ""}`.trim()}
      aria-label={label}
    >
      {stages.map((stage, index) => {
        const status =
          index === current ? "is-current" : index < current ? "is-done" : "";
        const body = (
          <>
            <span>{index + 1}</span>
            <strong>{stage.name}</strong>
            {stage.detail && <small>{stage.detail}</small>}
          </>
        );
        return (
          <li key={stage.id} className={status}>
            {onSelect ? (
              <button
                type="button"
                aria-current={index === current ? "step" : undefined}
                onClick={() => onSelect(index)}
              >
                {body}
              </button>
            ) : (
              <div aria-current={index === current ? "step" : undefined}>{body}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export type SeriesTone = "forward" | "loss" | "gradient" | "attention" | "muted";

export interface LineSeries {
  id: string;
  name: string;
  points: ReadonlyArray<PlotPoint>;
  tone?: SeriesTone;
  /** Dash pattern, so a series is never distinguished by colour alone. */
  dash?: "solid" | "dashed" | "dotted";
  format?: (value: number) => string;
}

/**
 * A small multi-series chart. The legend repeats every final value as text, so
 * the chart always has a readable data alternative.
 */
/** Compact axis numbers: 1200 → 1.2k, 0.125 → 0.13, 3 → 3. */
function formatAxisTick(value: number) {
  const size = Math.abs(value);
  if (size >= 1e9) return `${+(value / 1e9).toFixed(1)}B`;
  if (size >= 1e6) return `${+(value / 1e6).toFixed(1)}M`;
  if (size >= 1e4) return `${+(value / 1e3).toFixed(1)}k`;
  if (Number.isInteger(value)) return String(value);
  return String(+value.toFixed(size < 1 ? 2 : 1));
}

export function LineChart({
  label,
  series,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  marker,
  footnote,
}: {
  label: string;
  series: ReadonlyArray<LineSeries>;
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  marker?: { x: number; label: string };
  footnote?: string;
}) {
  const points = series.flatMap((entry) => entry.points);
  const left = 48;
  const right = 470;
  const top = 18;
  const bottom = 196;
  const [xMinimum, xMaximum] = xDomain ?? [
    Math.min(...points.map((point) => point.x), 0),
    Math.max(...points.map((point) => point.x), 1),
  ];
  const [yMinimum, yMaximum] = yDomain ?? [
    Math.min(...points.map((point) => point.y), 0),
    Math.max(...points.map((point) => point.y), 1),
  ];
  const xSpan = xMaximum - xMinimum || 1;
  const ySpan = yMaximum - yMinimum || 1;
  // Values outside an explicit domain are pinned to its edge rather than drawn
  // outside the plot area. The legend still reports the true final value.
  const clamp = (value: number, low: number, high: number) =>
    Math.min(high, Math.max(low, Number.isFinite(value) ? value : high));
  const xAt = (value: number) =>
    left + ((clamp(value, xMinimum, xMaximum) - xMinimum) / xSpan) * (right - left);
  const yAt = (value: number) =>
    bottom - ((clamp(value, yMinimum, yMaximum) - yMinimum) / ySpan) * (bottom - top);
  // Ends are always labelled; the midpoint only when it is a readable value.
  const xMiddle = (xMinimum + xMaximum) / 2;
  const integerDomain = Number.isInteger(xMinimum) && Number.isInteger(xMaximum);
  const xTicks =
    xMaximum > xMinimum
      ? !integerDomain || Number.isInteger(xMiddle)
        ? [xMinimum, xMiddle, xMaximum]
        : [xMinimum, xMaximum]
      : [xMinimum];
  const describe = (entry: LineSeries) => {
    const last = entry.points[entry.points.length - 1];
    const formatter = entry.format ?? ((value: number) => value.toFixed(3));
    return last ? formatter(last.y) : "no data";
  };

  return (
    <div className="viz-line-chart">
      <svg
        viewBox="0 0 490 236"
        role="img"
        aria-label={`${label}. ${series
          .map((entry) => `${entry.name} ends at ${describe(entry)}`)
          .join(". ")}.`}
      >
        <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="plot-axis" x1={left} x2={left} y1={top} y2={bottom} />
        {[0, 0.5, 1].map((fraction) => (
          <g key={fraction}>
            <line
              className="viz-line-chart__grid"
              x1={left}
              x2={right}
              y1={yAt(yMinimum + fraction * ySpan)}
              y2={yAt(yMinimum + fraction * ySpan)}
            />
            <text
              className="viz-line-chart__tick"
              x={left - 7}
              y={yAt(yMinimum + fraction * ySpan) + 3}
              textAnchor="end"
            >
              {(yMinimum + fraction * ySpan).toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text
            key={`x-${tick}`}
            className="viz-line-chart__tick"
            x={xAt(tick)}
            y={bottom + 12}
            textAnchor={tick === xMinimum ? "start" : tick === xMaximum ? "end" : "middle"}
          >
            {formatAxisTick(tick)}
          </text>
        ))}
        {marker && (
          <g>
            <line
              className="marker-guide"
              x1={xAt(marker.x)}
              x2={xAt(marker.x)}
              y1={top}
              y2={bottom}
            />
            <text
              className="viz-line-chart__marker-label"
              x={xAt(marker.x) + (xAt(marker.x) > (left + right) / 2 ? -4 : 4)}
              y={top + 9}
              textAnchor={xAt(marker.x) > (left + right) / 2 ? "end" : "start"}
            >
              {marker.label}
            </text>
          </g>
        )}
        {series.map((entry) => (
          <path
            key={entry.id}
            className={`viz-line viz-line--${entry.tone ?? "forward"} viz-line--${entry.dash ?? "solid"}`}
            d={smoothLinePath(
              entry.points.map((point) => ({ x: xAt(point.x), y: yAt(point.y) })),
            )}
          />
        ))}
        <text className="viz-line-chart__axis-label" x={(left + right) / 2} y={230} textAnchor="middle">
          {xLabel}
        </text>
        <text className="viz-line-chart__axis-label" x={left} y={12}>
          {yLabel}
        </text>
      </svg>
      <div className="viz-line-chart__legend">
        {series.map((entry) => (
          <span key={entry.id} className={`viz-legend viz-legend--${entry.tone ?? "forward"}`}>
            <i className={`viz-legend__dash viz-legend__dash--${entry.dash ?? "solid"}`} />
            {entry.name}
            <b>{describe(entry)}</b>
          </span>
        ))}
      </div>
      {footnote && <p className="viz-line-chart__footnote">{footnote}</p>}
    </div>
  );
}

export interface BarListItem {
  id: string;
  label: string;
  value: number;
  display?: string;
  detail?: string;
  tone?: SeriesTone;
  emphasis?: boolean;
}

/** Ranked horizontal bars. The numeric value is always printed beside the bar. */
export function BarList({
  label,
  items,
  max,
  onSelect,
  selectedId,
}: {
  label: string;
  items: ReadonlyArray<BarListItem>;
  max?: number;
  onSelect?: (id: string) => void;
  selectedId?: string;
}) {
  const ceiling = max ?? Math.max(1e-9, ...items.map((item) => Math.abs(item.value)));

  return (
    <div className="viz-bar-list" aria-label={label} role="list">
      {items.map((item) => {
        const width = `${Math.min(100, (Math.abs(item.value) / ceiling) * 100)}%`;
        const body = (
          <>
            <span>{item.label}</span>
            <i>
              <b
                className={`viz-bar--${item.tone ?? "forward"}`}
                style={{ width }}
              />
            </i>
            <strong>{item.display ?? item.value.toFixed(3)}</strong>
            {item.detail && <small>{item.detail}</small>}
          </>
        );
        const className = [
          item.emphasis ? "is-emphasis" : "",
          selectedId === item.id ? "is-selected" : "",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <div className={`viz-bar-row ${className}`.trim()} key={item.id} role="listitem">
            {onSelect ? (
              <button
                type="button"
                aria-pressed={selectedId === item.id}
                onClick={() => onSelect(item.id)}
              >
                {body}
              </button>
            ) : (
              <div>{body}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function Heatmap({
  rows,
  columns,
  values,
  selected,
  onSelect,
  label,
}: {
  rows: ReadonlyArray<string>;
  columns: ReadonlyArray<string>;
  values: ReadonlyArray<ReadonlyArray<number>>;
  selected?: { row: number; column: number };
  onSelect?: (row: number, column: number) => void;
  label: string;
}) {
  return (
    <div className="viz-heatmap-wrap">
      <div
        className="viz-heatmap"
        role="grid"
        aria-label={label}
        style={{ "--heatmap-columns": columns.length } as CSSProperties}
      >
        <span aria-hidden="true" />
        {columns.map((column, index) => (
          <span className="viz-heatmap__column" aria-hidden="true" key={`${column}-${index}`}>
            {column}
          </span>
        ))}
        {rows.map((row, rowIndex) => [
          <span className="viz-heatmap__row" aria-hidden="true" key={`row-${rowIndex}`}>
            {row}
          </span>,
          ...columns.map((column, columnIndex) => {
            const value = values[rowIndex]?.[columnIndex] ?? 0;
            const isSelected =
              selected?.row === rowIndex && selected.column === columnIndex;
            return (
              <button
                type="button"
                role="gridcell"
                key={`${rowIndex}-${columnIndex}`}
                className={isSelected ? "is-selected" : ""}
                aria-label={`${row} to ${column}: ${formatPercent(value)}`}
                aria-selected={isSelected}
                style={{ "--heat-value": value } as CSSProperties}
                onClick={() => onSelect?.(rowIndex, columnIndex)}
              >
                <span>{formatPercent(value)}</span>
              </button>
            );
          }),
        ])}
      </div>
    </div>
  );
}
