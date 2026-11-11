import { useId, type CSSProperties, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { niceTicks, useMeasuredWidth } from "@app/module-sdk";
import { MAX_BAR, MIN_BAR, sigmoid } from "./chance";

/** 0.5 → "50%", 0.125 → "12.5%" when asked for decimals. Trailing zeros are dropped. */
export const pct = (share: number, digits = 0) => {
  const text = (share * 100).toFixed(digits);
  return `${digits > 0 ? String(Number(text)) : text}%`;
};
export const bits = (value: number, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "∞");

export interface BarSeries {
  /** Picks the fill: "truth" is solid, "tally" and "guess" are hatched, so no pair differs by colour alone. */
  id: "truth" | "tally" | "guess";
  name: string;
  shares: ReadonlyArray<number>;
  /** Text above each bar. Defaults to the share as a percent. */
  labels?: ReadonlyArray<string>;
}

/**
 * One column per word, drawn to a fixed 0–100% scale. A column holds one or two bars; one series may
 * be editable, in which case dragging in the column (or the arrow keys on the bar) sets that word's
 * share and the caller rescales the rest.
 */
export function Columns({
  label,
  names,
  series,
  ghost,
  editable,
  selected,
  onSelect,
  footers,
  track = 168,
}: {
  label: string;
  names: ReadonlyArray<string>;
  series: ReadonlyArray<BarSeries>;
  /** Dashed outline of an earlier or default state, as shares. */
  ghost?: ReadonlyArray<number>;
  editable?: { series: number; name: (word: string) => string; onChange: (index: number, percent: number) => void };
  selected?: number;
  onSelect?: (index: number) => void;
  /** One short line under each word's name. */
  footers?: ReadonlyArray<string>;
  /** Height of the plot in pixels. */
  track?: number;
}) {
  const setFromPointer = (index: number, event: ReactPointerEvent<HTMLDivElement>) => {
    if (!editable) return;
    const box = event.currentTarget.getBoundingClientRect();
    if (box.height <= 0) return;
    editable.onChange(index, Math.round((1 - (event.clientY - box.top) / box.height) * 100));
  };
  const onKey = (index: number, current: number) => (event: KeyboardEvent<HTMLElement>) => {
    if (!editable) return;
    const jump: Record<string, number> = {
      ArrowUp: 1,
      ArrowRight: 1,
      ArrowDown: -1,
      ArrowLeft: -1,
      PageUp: 10,
      PageDown: -10,
    };
    const next =
      event.key === "Home" ? MIN_BAR : event.key === "End" ? MAX_BAR : event.key in jump ? current + jump[event.key] : null;
    if (next === null) return;
    event.preventDefault();
    editable.onChange(index, next);
  };

  return (
    <div className="ps-cols" style={{ "--ps-track": `${track}px` } as CSSProperties}>
      <div className="ps-cols__axis" aria-hidden="true">
        {[100, 75, 50, 25, 0].map((tick) => (
          <span key={tick} style={{ bottom: `${tick}%` }}>
            {tick}%
          </span>
        ))}
      </div>
      <div className="ps-cols__plot" role="group" aria-label={label}>
        {names.map((name, index) => (
          <div
            key={name}
            className={["ps-col", selected === index ? "is-selected" : "", editable ? "is-editable" : ""].filter(Boolean).join(" ")}
          >
            <div
              className="ps-col__track"
              onPointerDown={(event) => {
                if (!editable) return;
                try {
                  event.currentTarget.setPointerCapture(event.pointerId);
                } catch {
                  // Capture is a nicety (the drag keeps following outside the column); the click still lands.
                }
                setFromPointer(index, event);
              }}
              onPointerMove={(event) => {
                if (editable && event.buttons === 1) setFromPointer(index, event);
              }}
            >
              {ghost && <i className="ps-col__ghost" style={{ height: `${ghost[index] * 100}%` }} aria-hidden="true" />}
              <div className="ps-col__bars">
                {series.map((entry, at) => {
                  const share = entry.shares[index] ?? 0;
                  const text = entry.labels?.[index] ?? pct(share);
                  const isEditable = editable?.series === at;
                  const percent = Math.round(share * 100);
                  return (
                    <div
                      key={entry.id}
                      className={`ps-bar ps-bar--${entry.id} ${isEditable ? "is-handle" : ""} ${share > 0.86 ? "is-tall" : ""}`.trim()}
                      style={{ height: `${Math.min(100, share * 100)}%` }}
                      {...(isEditable
                        ? {
                            role: "slider",
                            tabIndex: 0,
                            "aria-label": editable.name(name),
                            "aria-valuemin": MIN_BAR,
                            "aria-valuemax": MAX_BAR,
                            "aria-valuenow": percent,
                            "aria-valuetext": `${percent}%`,
                            "aria-orientation": "vertical" as const,
                            onKeyDown: onKey(index, percent),
                          }
                        : { role: "img", "aria-label": `${name}, ${entry.name}: ${text}` })}
                    >
                      <b>{text}</b>
                    </div>
                  );
                })}
              </div>
            </div>
            {onSelect ? (
              <button type="button" className="ps-col__name" aria-pressed={selected === index} onClick={() => onSelect(index)}>
                {name}
              </button>
            ) : (
              <span className="ps-col__name">{name}</span>
            )}
            {footers && <small className="ps-col__foot">{footers[index]}</small>}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Surprise bars that hang down under a Columns chart, one per word, on a scale of whole bits. Each
 * gridline is one more halving: 1 bit is "1 in 2", 2 bits "1 in 4", and so on.
 */
export function SurpriseBars({
  names,
  values,
  selected,
  max = 7,
  track = 176,
}: {
  names: ReadonlyArray<string>;
  values: ReadonlyArray<number>;
  selected?: number;
  max?: number;
  track?: number;
}) {
  const lines = Array.from({ length: max }, (_, index) => index + 1);
  return (
    <div className="ps-hang" style={{ "--ps-track": `${track}px`, "--ps-lines": max } as CSSProperties}>
      <div className="ps-hang__axis" aria-hidden="true">
        <span style={{ top: 0 }}>0 bits · certain</span>
        {lines.slice(0, max - 1).map((line) => (
          <span key={line} style={{ top: `${(line / max) * 100}%` }}>
            {line} {line === 1 ? "bit" : "bits"} · 1 in {2 ** line}
          </span>
        ))}
      </div>
      <div
        className="ps-hang__plot"
        role="img"
        aria-label={`Surprise of each word in bits: ${names.map((name, index) => `${name} ${bits(values[index])}`).join(", ")}.`}
      >
        {names.map((name, index) => (
          <div key={name} className={`ps-hang__col ${selected === index ? "is-selected" : ""}`.trim()}>
            <i style={{ height: `${Math.min(100, (values[index] / max) * 100)}%` }}>
              <b>{bits(values[index])} bits</b>
            </i>
          </div>
        ))}
      </div>
    </div>
  );
}

/** What each fill means, for a chart that draws more than one series. */
export function Legend({ items }: { items: ReadonlyArray<{ id: string; text: string }> }) {
  return (
    <ul className="ps-legend">
      {items.map((item) => (
        <li key={item.id}>
          <i className={`ps-swatch ps-swatch--${item.id}`} aria-hidden="true" />
          {item.text}
        </li>
      ))}
    </ul>
  );
}

/**
 * One whole, cut into parts whose widths are their shares. With `cells`, the same strip is ruled
 * into 100 equal cells, so a 50% part reads as 50 stories out of 100.
 */
export function WholeStrip({
  label,
  parts,
  cells = false,
  caption,
}: {
  label: string;
  parts: ReadonlyArray<{ name: string; share: number; text: string; pattern?: number }>;
  cells?: boolean;
  caption?: string;
}) {
  return (
    <div className="ps-whole">
      <div className={`ps-whole__bar ${cells ? "is-cells" : ""}`.trim()} role="img" aria-label={label}>
        {parts.map((part, index) => (
          <span
            key={part.name}
            className={`ps-whole__part ps-whole__part--${part.pattern ?? index}`}
            style={{ flexGrow: Math.max(0.0001, part.share), flexBasis: 0, "--ps-cells": Math.max(1, Math.round(part.share * 100)) } as CSSProperties}
          >
            {part.share >= 0.14 ? (
              <em>
                <b>{part.name}</b> {part.text}
              </em>
            ) : part.share >= 0.045 ? (
              <em>{part.text}</em>
            ) : null}
          </span>
        ))}
      </div>
      <div className="ps-whole__scale" aria-hidden="true">
        <span>0</span>
        <span>{caption ?? "one whole"}</span>
        <span>{cells ? "100 stories" : "100%"}</span>
      </div>
    </div>
  );
}

/**
 * Bayes' rule as areas. Each prompt is a column as wide as its prior (its share of all stories),
 * filled up to the chance that the word follows it. The filled areas are prior × likelihood; the
 * strip underneath keeps only those areas and stretches them to one whole, which is the posterior.
 */
export function BayesArea({
  label,
  names,
  prior,
  likelihood,
  joint,
  posterior,
  word,
  short,
}: {
  label: string;
  names: ReadonlyArray<string>;
  /** One-word names for columns too narrow for the full prompt. */
  short: ReadonlyArray<string>;
  prior: ReadonlyArray<number>;
  likelihood: ReadonlyArray<number>;
  joint: ReadonlyArray<number>;
  posterior: ReadonlyArray<number>;
  word: string;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(660);
  const width = Math.max(300, measured);
  const left = 44;
  const right = width - 12;
  const top = 46;
  const side = 150;
  const base = top + side;
  const stripTop = base + 78;
  const stripHeight = 34;
  const height = stripTop + stripHeight + 34;
  const span = right - left;
  let edge = 0;
  const columns = prior.map((share, index) => {
    const start = edge;
    edge += share;
    return { index, x0: left + start * span, x1: left + edge * span, share };
  });
  let postEdge = 0;
  const segments = posterior.map((share, index) => {
    const start = postEdge;
    postEdge += share;
    return { index, x0: left + start * span, x1: left + postEdge * span, share };
  });
  const fill = (index: number) => (index === 0 ? undefined : `url(#${id}-p${index})`);

  return (
    <div className="ps-plot" ref={frameRef}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        <defs>
          <pattern id={`${id}-p1`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect className="ps-bayes__base ps-bayes__base--1" width="7" height="7" />
            <line className="ps-bayes__ink" x1="0" y1="0" x2="0" y2="7" />
          </pattern>
          <pattern id={`${id}-p2`} width="7" height="7" patternUnits="userSpaceOnUse">
            <rect className="ps-bayes__base ps-bayes__base--2" width="7" height="7" />
            <circle className="ps-bayes__dot" cx="3.5" cy="3.5" r="1.3" />
          </pattern>
        </defs>
        <text className="ps-axis-label" x={left} y={12}>
          All the stories, one column per prompt. Width = the prior. Filled height = chance of “{word}”.
        </text>
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line className="ps-grid" x1={left} x2={right} y1={base - tick * side} y2={base - tick * side} />
            <text className="ps-tick" x={left - 6} y={base - tick * side + 3.5} textAnchor="end">
              {pct(tick)}
            </text>
          </g>
        ))}
        {columns.map((column) => {
          const wide = column.x1 - column.x0;
          const tall = likelihood[column.index] * side;
          const mid = (column.x0 + column.x1) / 2;
          return (
            <g key={column.index}>
              <rect className="ps-bayes__all" x={column.x0} y={top} width={wide} height={side} />
              <rect
                className={`ps-bayes__fill ps-bayes__fill--${column.index}`}
                x={column.x0}
                y={base - tall}
                width={wide}
                height={tall}
                fill={fill(column.index)}
              />
              {wide >= 30 && (
                <text className="ps-bayes__name" x={mid} y={top - 18} textAnchor="middle">
                  {wide >= 66 ? names[column.index] : short[column.index]}
                </text>
              )}
              {wide >= 30 && (
                <text className="ps-bayes__prior" x={mid} y={top - 5} textAnchor="middle">
                  {wide >= 66 ? "prior " : ""}
                  {pct(column.share)}
                </text>
              )}
              {wide >= 30 && (
                <text className="ps-bayes__like ps-halo" x={mid} y={Math.min(base - 6, Math.max(top + 13, base - tall - 5))} textAnchor="middle">
                  {pct(likelihood[column.index])}
                  {wide >= 76 ? " say it" : ""}
                </text>
              )}
              {/* The funnel from a column's filled part down to its share of the strip. */}
              <path
                className={`ps-bayes__funnel ps-bayes__funnel--${column.index}`}
                d={`M${column.x0} ${base + 26}L${column.x1} ${base + 26}L${segments[column.index].x1} ${stripTop}L${segments[column.index].x0} ${stripTop}Z`}
              />
              {wide >= 30 && (
                <text className="ps-bayes__joint" x={mid} y={base + 17} textAnchor="middle">
                  {wide >= 118 ? `${pct(column.share)} × ${pct(likelihood[column.index])} = ` : wide >= 44 ? "= " : ""}
                  {pct(joint[column.index], 1)}
                </text>
              )}
            </g>
          );
        })}
        <line className="ps-axis" x1={left} x2={right} y1={base} y2={base} />
        {segments.map((segment) => {
          const wide = segment.x1 - segment.x0;
          return (
            <g key={segment.index}>
              <rect
                className={`ps-bayes__fill ps-bayes__fill--${segment.index}`}
                x={segment.x0}
                y={stripTop}
                width={Math.max(0, wide)}
                height={stripHeight}
                fill={fill(segment.index)}
              />
              {wide >= 34 && (
                <text className="ps-bayes__post ps-halo" x={(segment.x0 + segment.x1) / 2} y={stripTop + stripHeight / 2 + 4} textAnchor="middle">
                  {wide >= 124 ? `${names[segment.index]} ` : wide >= 76 ? `${short[segment.index]} ` : ""}
                  {pct(segment.share)}
                </text>
              )}
            </g>
          );
        })}
        <rect className="ps-bayes__frame" x={left} y={stripTop} width={span} height={stripHeight} />
        <text className="ps-axis-label" x={left} y={stripTop + stripHeight + 16}>
          Only the stories that said “{word}”, stretched to one whole: the posterior.
        </text>
      </svg>
    </div>
  );
}

/** The S-shaped sigmoid curve, with the current score gap marked on it. */
export function SigmoidCurve({ gap, first, second }: { gap: number; first: string; second: string }) {
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(660);
  const width = Math.max(300, measured);
  const height = 200;
  const left = 44;
  const right = width - 14;
  const top = 22;
  const bottom = height - 40;
  const xAt = (value: number) => left + ((Math.min(6, Math.max(-6, value)) + 6) / 12) * (right - left);
  const yAt = (value: number) => bottom - value * (bottom - top);
  const curve = Array.from({ length: 121 }, (_, index) => {
    const value = -6 + index / 10;
    return `${index === 0 ? "M" : "L"}${xAt(value).toFixed(1)} ${yAt(sigmoid(value)).toFixed(1)}`;
  }).join("");
  const share = sigmoid(gap);
  const onRight = xAt(gap) > (left + right) / 2;
  return (
    <div className="ps-plot" ref={frameRef}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`The sigmoid curve. The score gap between ${first} and ${second} is ${gap.toFixed(2)}, which gives ${first} a ${pct(share, 1)} share of the two. A gap of zero gives 50%.`}
      >
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line className="ps-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="ps-tick" x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {pct(tick)}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={left - 34} y={12}>
          {first}’s share of the two ↑
        </text>
        <line className="ps-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="ps-zero" x1={xAt(0)} x2={xAt(0)} y1={top} y2={bottom} />
        {[-6, -4, -2, 0, 2, 4, 6].map((tick) => (
          <g key={tick}>
            <line className="ps-axis" x1={xAt(tick)} x2={xAt(tick)} y1={bottom} y2={bottom + 4} />
            <text className="ps-tick" x={xAt(tick)} y={bottom + 15} textAnchor="middle">
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={(left + right) / 2} y={height - 6} textAnchor="middle">
          score gap: {first}’s score minus {second}’s (→ {first} further ahead)
        </text>
        <path className="ps-curve" d={curve} />
        <line className="ps-drop" x1={xAt(gap)} x2={xAt(gap)} y1={yAt(share)} y2={bottom} />
        <line className="ps-drop" x1={left} x2={xAt(gap)} y1={yAt(share)} y2={yAt(share)} />
        <circle className="ps-dot" cx={xAt(gap)} cy={yAt(share)} r={5} />
        <text
          className="ps-dot-name ps-halo"
          x={xAt(gap) + (onRight ? -9 : 9)}
          y={yAt(share) + (share > 0.5 ? 16 : -8)}
          textAnchor={onRight ? "end" : "start"}
        >
          gap {gap >= 0 ? "+" : ""}
          {gap.toFixed(2)} → {pct(share, 1)}
        </text>
      </svg>
    </div>
  );
}

/**
 * Two draws in a row. Left: the chance of both is the area of a rectangle, one chance wide and the
 * other tall. Right: their surprises laid end to end reach exactly the surprise of the pair.
 */
export function PairPicture({
  first,
  second,
  pFirst,
  pSecond,
}: {
  first: string;
  second: string;
  pFirst: number;
  pSecond: number;
}) {
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(660);
  const width = Math.max(300, measured);
  const stacked = width < 520;
  const side = 132;
  const squareLeft = 46;
  const squareTop = 24;
  const barLeft = stacked ? 46 : squareLeft + side + 84;
  const barTop = stacked ? squareTop + side + 70 : 64;
  const barRight = width - 16;
  const height = stacked ? barTop + 92 : squareTop + side + 40;
  const bitsFirst = -Math.log2(pFirst);
  const bitsSecond = -Math.log2(pSecond);
  const both = pFirst * pSecond;
  const bitsBoth = -Math.log2(both);
  const scaleMax = Math.max(4, Math.ceil(bitsBoth + 0.5));
  const bAt = (value: number) => barLeft + (value / scaleMax) * (barRight - barLeft);
  const ticks = niceTicks(0, scaleMax, 4).filter((tick) => Number.isInteger(tick));
  return (
    <div className="ps-plot" ref={frameRef}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Two draws in a row: ${first}, then ${second}. The chances multiply: ${pct(pFirst)} times ${pct(pSecond)} is ${pct(both, 2)}. The surprises add: ${bits(bitsFirst)} plus ${bits(bitsSecond)} is ${bits(bitsBoth)} bits.`}
      >
        <text className="ps-axis-label" x={squareLeft} y={12}>
          chances multiply: an area
        </text>
        <rect className="ps-pair__all" x={squareLeft} y={squareTop} width={side} height={side} />
        <rect
          className="ps-pair__area"
          x={squareLeft}
          y={squareTop + side - pSecond * side}
          width={pFirst * side}
          height={pSecond * side}
        />
        <line className="ps-pair__edge" x1={squareLeft} x2={squareLeft + pFirst * side} y1={squareTop + side + 6} y2={squareTop + side + 6} />
        <text className="ps-tick" x={squareLeft} y={squareTop + side + 19}>
          {first} {pct(pFirst)} wide
        </text>
        <line className="ps-pair__edge" x1={squareLeft - 6} x2={squareLeft - 6} y1={squareTop + side} y2={squareTop + side - pSecond * side} />
        <text
          className="ps-tick"
          x={squareLeft - 12}
          y={squareTop + side}
          transform={`rotate(-90 ${squareLeft - 12} ${squareTop + side})`}
        >
          {second} {pct(pSecond)} tall
        </text>
        <text className="ps-pair__value ps-halo" x={squareLeft + side + 8} y={squareTop + side - Math.max(8, (pSecond * side) / 2) + 4}>
          = {pct(both, 2)}
        </text>
        <text className="ps-axis-label" x={barLeft} y={barTop - 26}>
          surprises add: end to end
        </text>
        <rect className="ps-pair__bits ps-pair__bits--a" x={bAt(0)} y={barTop} width={bAt(bitsFirst) - bAt(0)} height={30} />
        <rect
          className="ps-pair__bits ps-pair__bits--b"
          x={bAt(bitsFirst)}
          y={barTop}
          width={bAt(bitsFirst + bitsSecond) - bAt(bitsFirst)}
          height={30}
        />
        {bAt(bitsFirst) - bAt(0) > 54 && (
          <text className="ps-pair__label" x={(bAt(0) + bAt(bitsFirst)) / 2} y={barTop + 19} textAnchor="middle">
            {first} {bits(bitsFirst)}
          </text>
        )}
        {bAt(bitsFirst + bitsSecond) - bAt(bitsFirst) > 54 && (
          <text className="ps-pair__label" x={(bAt(bitsFirst) + bAt(bitsFirst + bitsSecond)) / 2} y={barTop + 19} textAnchor="middle">
            {second} {bits(bitsSecond)}
          </text>
        )}
        <line className="ps-pair__mark" x1={bAt(bitsBoth)} x2={bAt(bitsBoth)} y1={barTop - 8} y2={barTop + 38} />
        <text
          className="ps-pair__value ps-halo"
          x={bAt(bitsBoth) + (bAt(bitsBoth) > barRight - 230 ? -6 : 6)}
          y={barTop - 10}
          textAnchor={bAt(bitsBoth) > barRight - 230 ? "end" : "start"}
        >
          surprise of {pct(both, 2)} = {bits(bitsBoth)} bits
        </text>
        <line className="ps-axis" x1={barLeft} x2={barRight} y1={barTop + 38} y2={barTop + 38} />
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="ps-axis" x1={bAt(tick)} x2={bAt(tick)} y1={barTop + 38} y2={barTop + 42} />
            <text className="ps-tick" x={bAt(tick)} y={barTop + 54} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={barRight} y={barTop + 70} textAnchor="end">
          bits →
        </text>
      </svg>
    </div>
  );
}

/** "As uncertain as a fair choice among N": N boxes, the last one partly filled. */
export function FairBoxes({ count, total }: { count: number; total: number }) {
  return (
    <div className="ps-fair" role="img" aria-label={`As uncertain as a fair choice among ${count.toFixed(1)} words, out of ${total}.`}>
      {Array.from({ length: total }, (_, index) => {
        const filled = Math.min(1, Math.max(0, count - index));
        return (
          <span key={index} className="ps-fair__box">
            <i style={{ width: `${filled * 100}%` }} />
          </span>
        );
      })}
    </div>
  );
}
