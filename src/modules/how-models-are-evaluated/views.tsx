import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Example, Matrix, Split } from "./metrics";

/** Track the rendered width so a picture is drawn in CSS pixels and its text keeps its size in a narrow card. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry.contentRect.width);
      if (next > 0) setWidth((current) => (Math.abs(current - next) > 2 ? next : current));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

const pct = (value: number | null, digits = 1) => (value === null ? "n/a" : `${(value * 100).toFixed(digits)}%`);

/** One scored row as the mark the score strip uses for it: circle for a negative, diamond for a positive. */
function RowDots({ count, kind }: { count: number; kind: "tn" | "fp" | "fn" | "tp" }) {
  return (
    <span className="eval-cm__dots" aria-hidden="true">
      {Array.from({ length: count }, (_, index) => (
        <i key={index} className={`is-${kind}`} />
      ))}
    </span>
  );
}

/**
 * The four counts as one two-by-two grid. Each cell prints its count and draws its rows with the score
 * strip's own marks, so the strip's dots can be seen sorted into four boxes.
 */
export function ConfusionGrid({ counted }: { counted: Matrix }) {
  const total = Math.max(1, counted.n);
  const share = (count: number) => `${((count / total) * 100).toFixed(0)}% of scored rows`;
  const allMissed = counted.fn > 0 && counted.tp === 0;
  const cell = (kind: "tn" | "fp" | "fn" | "tp", count: number, words: string, flag = false) => (
    <td className={`eval-cm__cell is-${kind}${kind === "tn" || kind === "tp" ? " is-right" : " is-wrong"}${flag ? " is-flagged" : ""}`}>
      <strong>
        {count} {kind.toUpperCase()}
      </strong>
      <span>{words}</span>
      <small>{share(count)}</small>
      <RowDots count={count} kind={kind} />
    </td>
  );
  return (
    <table className="eval-cm" aria-label="Confusion matrix: true class against predicted class, as counts of scored rows">
      <thead>
        <tr>
          <td />
          <th scope="col">
            Pred 0 <small>the alarm stayed quiet</small>
          </th>
          <th scope="col">
            Pred 1 <small>the alarm rang</small>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">
            True 0 <small>really negative</small>
          </th>
          {cell("tn", counted.tn, "quiet, and right")}
          {cell("fp", counted.fp, "false alarm")}
        </tr>
        <tr>
          <th scope="row">
            True 1 <small>really positive</small>
          </th>
          {cell("fn", counted.fn, "missed", allMissed)}
          {cell("tp", counted.tp, "caught")}
        </tr>
      </tbody>
    </table>
  );
}

export type ReadingPart = { code: string; amount: number; good: boolean };
export type Reading = {
  id: string;
  name: string;
  formula: string;
  parts: ReadingPart[];
  value: number | null;
  alert?: boolean;
  empty?: string;
};

/**
 * Each reading as one bar whose whole length is the reading's denominator and whose solid part is its
 * numerator, so the four readings can be seen to be four cuts of the same counts.
 */
export function ReadingBars({ readings, stacked = false, label }: { readings: Reading[]; stacked?: boolean; label: string }) {
  return (
    <div className={`eval-readings${stacked ? " is-stacked" : ""}`} role="group" aria-label={label}>
      {readings.map((reading) => {
        const whole = reading.parts.reduce((sum, part) => sum + part.amount, 0);
        return (
          <div key={reading.id} className={`eval-reading${reading.alert ? " is-alert" : ""}`}>
            <p className="eval-reading__name">
              <strong>{reading.name}</strong>
              <code>{reading.formula}</code>
            </p>
            {reading.value === null || whole <= 0 ? (
              <span className="eval-reading__bar is-empty">{reading.empty ?? "0 / 0 · nothing to divide by"}</span>
            ) : (
              <span className="eval-reading__bar" aria-hidden="true">
                {reading.parts
                  .filter((part) => part.amount > 0)
                  .map((part) => (
                    <i key={part.code} className={part.good ? "is-good" : "is-bad"} style={{ flexGrow: part.amount }}>
                      {part.amount / whole >= 0.1 ? part.code : ""}
                    </i>
                  ))}
              </span>
            )}
            <b className="eval-reading__value">{pct(reading.value)}</b>
          </div>
        );
      })}
    </div>
  );
}

const SLICES: ReadonlyArray<{ id: Split; from: number; to: number }> = [
  { id: "train", from: 0, to: 69 },
  { id: "val", from: 70, to: 84 },
  { id: "test", from: 85, to: 99 },
];

/**
 * The hundred rows in one line, cut into train, val and test, with the slice that chooses t and the slice
 * that is scored for the claim both pointed at. When they are the same slice, the holdout has leaked.
 */
export function SliceRail({
  examples,
  rare,
  tuneSplit,
  label,
}: {
  examples: readonly Example[];
  rare: boolean;
  tuneSplit: Split;
  label: string;
}) {
  const W = 670;
  const pitch = 6.5;
  const x0 = 10;
  const cellX = (id: number) => x0 + id * pitch;
  const top = 26;
  const height = 24;
  const mid = (from: number, to: number) => (cellX(from) + cellX(to) + pitch - 1) / 2;
  const leaked = tuneSplit === "test";
  const tune = SLICES.find((slice) => slice.id === tuneSplit)!;
  const test = SLICES[2];
  const noteY = top + height + 42;
  const claimY = leaked ? noteY : noteY + 15;
  return (
    <svg className="eval-rail" viewBox={`0 0 ${W} ${claimY + 8}`} role="img" aria-label={label}>
      {examples.map((example) => (
        <g key={example.id}>
          <title>{`row ${example.id} · ${example.split} · y=${example.label} · score ${example.score.toFixed(2)}`}</title>
          <rect className={`eval-rail__cell is-${example.split}`} x={cellX(example.id)} y={top} width={pitch - 1.5} height={height} rx={1.2} />
          {example.label === 1 && (
            <path
              className="eval-rail__positive"
              d={`M${cellX(example.id) + (pitch - 1.5) / 2} ${top - 12} l4 5 l-4 5 l-4 -5 z`}
            />
          )}
          {rare && example.id === 12 && <circle className="eval-rail__alarm" cx={cellX(12) + (pitch - 1.5) / 2} cy={top - 7} r={3.6} />}
        </g>
      ))}
      {rare && (
        <>
          <text className="eval-rail__note" x={cellX(12) + 9} y={top - 4}>
            row 12 · score 0.88
          </text>
          <text className="eval-rail__note" x={cellX(99) - 7} y={top - 4} textAnchor="end">
            row 99 · y = 1
          </text>
        </>
      )}
      {!rare && (
        <text className="eval-rail__note" x={x0} y={9}>
          ◆ above a cell: that row is really positive (y = 1)
        </text>
      )}
      {SLICES.map((slice) => (
        <g key={slice.id} className={`eval-rail__slice is-${slice.id}`}>
          <path
            className="eval-rail__brace"
            d={`M${cellX(slice.from)} ${top + height + 4} v5 H${cellX(slice.to) + pitch - 1.5} v-5`}
          />
          <text className="eval-rail__name" x={mid(slice.from, slice.to)} y={top + height + 22} textAnchor="middle">
            {slice.id} · {slice.from}–{slice.to}
          </text>
        </g>
      ))}
      {leaked ? (
        <text className="eval-rail__role is-leaked" x={mid(test.from, test.to) + 6} y={noteY} textAnchor="end">
          t is chosen here and the claim is scored here: leaked ▲
        </text>
      ) : (
        <>
          {tuneSplit === "val" ? (
            <text className="eval-rail__role" x={mid(tune.from, tune.to) + 6} y={noteY} textAnchor="end">
              t is chosen here ▲
            </text>
          ) : (
            <text className="eval-rail__role" x={mid(tune.from, tune.to)} y={noteY} textAnchor="middle">
              ▲ t is chosen here
            </text>
          )}
          <text className="eval-rail__role is-claim" x={mid(test.from, test.to) + 6} y={claimY} textAnchor="end">
            the claim is scored here, with t frozen ▲
          </text>
        </>
      )}
    </svg>
  );
}

/**
 * The claim each path would publish from the test slice against what fresh rows show, on one 0–100% scale.
 * The gap between the two marks is the overstatement.
 */
export function ClaimPlot({
  goalName,
  current,
  paths,
}: {
  goalName: string;
  current: string;
  paths: ReadonlyArray<{ id: string; label: string; claim: number | null; fresh: number | null }>;
}) {
  const W = 670;
  const L = 168;
  const R = 640;
  const rowH = 34;
  const top = 30;
  const xs = (value: number) => L + Math.min(1, Math.max(0, value)) * (R - L);
  const bottom = top + paths.length * rowH;
  const describe = (path: { label: string; claim: number | null; fresh: number | null }) =>
    `${path.label}: test claim ${pct(path.claim)}, fresh rows ${pct(path.fresh)}`;
  return (
    <svg
      className="eval-claims"
      viewBox={`0 0 ${W} ${bottom + 20}`}
      role="img"
      aria-label={`Test ${goalName} claimed by each path against the same threshold on fresh rows. ${paths.map(describe).join(". ")}.`}
    >
      <circle className="eval-claims__claim" cx={L + 5} cy={11} r={5} />
      <text className="eval-claims__key" x={L + 15} y={15}>
        test claim
      </text>
      <path className="eval-claims__fresh" d={`M${L + 104} 5 l6 6 l-6 6 l-6 -6 z`} />
      <text className="eval-claims__key" x={L + 115} y={15}>
        same t on fresh rows
      </text>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line className="eval-claims__grid" x1={xs(tick)} x2={xs(tick)} y1={top - 4} y2={bottom} />
          <text className="eval-claims__tick" x={xs(tick)} y={bottom + 14} textAnchor="middle">
            {Math.round(tick * 100)}%
          </text>
        </g>
      ))}
      {paths.map((path, index) => {
        const y = top + index * rowH + rowH / 2;
        const over = path.claim !== null && path.fresh !== null && path.claim > path.fresh;
        return (
          <g key={path.id} className={path.id === current ? "is-current" : undefined}>
            <text className="eval-claims__name" x={L - 10} y={y + 4} textAnchor="end">
              {path.label}
            </text>
            <line className="eval-claims__track" x1={L} x2={R} y1={y} y2={y} />
            {path.claim !== null && path.fresh !== null && (
              <line className={`eval-claims__gap${over ? " is-over" : ""}`} x1={xs(path.fresh)} x2={xs(path.claim)} y1={y} y2={y} />
            )}
            {path.fresh !== null && (
              <>
                <path className="eval-claims__fresh" d={`M${xs(path.fresh)} ${y - 6.5} l6.5 6.5 l-6.5 6.5 l-6.5 -6.5 z`} />
                <text
                  className="eval-claims__value"
                  x={xs(path.fresh) + (over || path.claim === null ? -10 : 10)}
                  y={y - 8}
                  textAnchor={over || path.claim === null ? "end" : "start"}
                >
                  {pct(path.fresh)}
                </text>
              </>
            )}
            {path.claim !== null ? (
              <>
                <circle className="eval-claims__claim" cx={xs(path.claim)} cy={y} r={5.5} />
                <text
                  className="eval-claims__value is-claim"
                  x={xs(path.claim) + (over || path.fresh === null ? 10 : -10)}
                  y={y - 8}
                  textAnchor={path.claim > 0.86 ? "end" : over || path.fresh === null ? "start" : "end"}
                >
                  {pct(path.claim)}
                </text>
              </>
            ) : (
              <text className="eval-claims__value" x={L + 8} y={y - 8}>
                n/a (0/0)
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Errors as bars on one fixed scale, so dragging the outlier shows MSE pulling away from MAE. */
export function ErrorBars({ rows, max, label }: { rows: ReadonlyArray<{ name: string; value: number; tone: "plain" | "loss"; note: string }>; max: number; label: string }) {
  const [wrapper, measured] = useWidth(670);
  const W = Math.max(360, measured);
  const L = 60;
  const R = W - 196;
  const rowH = 24;
  const top = 4;
  const xs = (value: number) => L + (Math.min(max, Math.max(0, value)) / max) * (R - L);
  const bottom = top + rows.length * rowH;
  return (
    <div ref={wrapper}>
    <svg className="eval-errors" viewBox={`0 0 ${W} ${bottom + 18}`} role="img" aria-label={label}>
      {[0, 10, 20, 30].filter((tick) => tick <= max).map((tick) => (
        <g key={tick}>
          <line className="eval-claims__grid" x1={xs(tick)} x2={xs(tick)} y1={top} y2={bottom} />
          <text className="eval-claims__tick" x={xs(tick)} y={bottom + 13} textAnchor="middle">
            {tick}
          </text>
        </g>
      ))}
      {rows.map((row, index) => {
        const y = top + index * rowH;
        return (
          <g key={row.name}>
            <text className="eval-claims__name" x={L - 8} y={y + 16} textAnchor="end">
              {row.name}
            </text>
            <rect className={`eval-errors__bar is-${row.tone}`} x={L} y={y + 5} width={Math.max(1.5, xs(row.value) - L)} height={14} rx={2} />
            <text className="eval-claims__value" x={xs(row.value) + 6} y={y + 16}>
              {row.value.toFixed(2)}
              <tspan className="eval-errors__note"> · {row.note}</tspan>
            </text>
          </g>
        );
      })}
    </svg>
    </div>
  );
}

/** A fold's twenty rows as cells: held out or trained on, with the positives marked. */
export function FoldCells({ rows, held }: { rows: readonly Example[]; held: boolean }) {
  return (
    <span className={`eval-foldcells${held ? " is-held" : ""}`} aria-hidden="true">
      {rows.map((example) => (
        <i key={example.id} className={example.label === 1 ? "is-positive" : undefined} />
      ))}
    </span>
  );
}

export function InlineBar({ value, children }: { value: number; children: ReactNode }) {
  return (
    <span className="eval-inbar">
      <span className="eval-inbar__track" aria-hidden="true">
        <i style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
      </span>
      {children}
    </span>
  );
}
