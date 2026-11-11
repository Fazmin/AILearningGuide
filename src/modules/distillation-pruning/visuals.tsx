import { useLayoutEffect, useRef, useState } from "react";
import { TINY_VOCAB, TINY_VOCAB_SIZE } from "@app/module-sdk";
import type { ComposeStage, PruneMode } from "./compress";

const V = TINY_VOCAB_SIZE;
const glyph = (character: string) => (character === " " ? "␣" : character);
const withCommas = (value: number) => value.toLocaleString("en-US");
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

/** The rendered width of an element, so each drawing is made at its real pixel size. */
function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const read = () => {
      const next = element.clientWidth;
      if (next > 0) setWidth((current) => (Math.abs(current - next) > 1 ? next : current));
    };
    read();
    const observer = new ResizeObserver(read);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Round tick values inside a range. */
function ticksFor(low: number, high: number, target = 4): number[] {
  const span = high - low;
  if (!(span > 0) || !Number.isFinite(span)) return [low];
  const rough = span / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / magnitude;
  const step = (fraction >= 7.5 ? 10 : fraction >= 3.5 ? 5 : fraction >= 1.5 ? 2 : 1) * magnitude;
  const ticks: number[] = [];
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : +value.toPrecision(12));
  }
  return ticks;
}

/**
 * KL to the teacher against distillation temperature for the current student size and budget, with the one-hot
 * student as a level line and the current temperature marked. Points arrive as the other temperatures are measured.
 */
export function TemperatureSweep({
  points,
  hardKl,
  temperature,
  softKl,
  complete,
}: {
  points: { t: number; kl: number }[];
  hardKl: number;
  temperature: number;
  softKl: number;
  complete: boolean;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const merged = [...points.filter((point) => point.t !== temperature), { t: temperature, kl: softKl }].sort((a, b) => a.t - b.t);
  const left = 44;
  const right = width - 16;
  const top = 24;
  const bottom = 150;
  const peak = Math.max(hardKl, ...merged.map((point) => point.kl)) * 1.18 || 1;
  const xAt = (t: number) => left + ((t - 0.5) / 5.5) * (right - left);
  const yAt = (value: number) => bottom - (value / peak) * (bottom - top);
  const best = complete ? merged.reduce((low, point) => (point.kl < low.kl ? point : low), merged[0]) : null;
  const currentIsBest = best !== null && best.t === temperature;
  const currentText = `your T = ${temperature.toFixed(1)} · ${softKl.toFixed(4)}${currentIsBest ? " · the closest" : ""}`;
  const currentLeft = xAt(temperature) + 10 + currentText.length * 6.4 > right;
  const hardText = `one-hot student ${hardKl.toFixed(4)}`;
  const ticks = ticksFor(0, peak / 1.18, 3);
  return (
    <div className="dp-sweep" ref={ref}>
      <svg
        className="dp-mask dp-sweep-plot"
        width={width}
        height={186}
        viewBox={`0 0 ${width} 186`}
        role="img"
        aria-label={`KL to the teacher against distillation temperature. ${merged.map((point) => `T ${point.t}: ${point.kl.toFixed(4)}`).join(", ")}. One-hot student: ${hardKl.toFixed(4)}. Your temperature ${temperature.toFixed(1)} gives ${softKl.toFixed(4)}.${best ? ` The closest is T ${best.t} at ${best.kl.toFixed(4)}.` : ""}`}
      >
        <text x={left} y={11}>
          KL to teacher · lower is closer{complete ? "" : " · measuring the other temperatures…"}
        </text>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="dp-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {tick === 0 ? "0" : +tick.toFixed(3)}
            </text>
          </g>
        ))}
        {[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6].map((tick) => (
          <text key={tick} x={xAt(tick)} y={bottom + 15} textAnchor="middle">
            {tick}
          </text>
        ))}
        <text x={(left + right) / 2} y={bottom + 31} textAnchor="middle">
          distillation temperature T
        </text>
        <line className="marker-guide" x1={xAt(temperature)} x2={xAt(temperature)} y1={top} y2={bottom} />
        <line className="dp-sweep-hard" x1={left} x2={right} y1={yAt(hardKl)} y2={yAt(hardKl)} />
        <line className="dp-sweep-hard" x1={right - hardText.length * 6.1 - 34} x2={right - hardText.length * 6.1 - 8} y1={8} y2={8} />
        <text className="dp-sweep-hard-label" x={right} y={11} textAnchor="end">
          {hardText}
        </text>
        <path
          className="dp-sweep-line"
          d={merged.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(point.t)},${yAt(point.kl)}`).join(" ")}
        />
        {merged.map((point) => (
          <circle key={point.t} className="dp-sweep-dot" cx={xAt(point.t)} cy={yAt(point.kl)} r={3} />
        ))}
        {best && !currentIsBest && (
          <text
            className="dp-sweep-best"
            x={Math.min(right - 80, Math.max(left + 80, xAt(best.t)))}
            y={yAt(best.kl) + 17 > bottom - 3 ? yAt(best.kl) - 9 : yAt(best.kl) + 17}
            textAnchor="middle"
          >
            ▲ closest at T = {best.t} · {best.kl.toFixed(4)}
          </text>
        )}
        <circle className="dp-sweep-current" cx={xAt(temperature)} cy={yAt(softKl)} r={6} />
        <text
          className="dp-current-label"
          x={xAt(temperature) + (currentLeft ? -10 : 10)}
          y={Math.max(top + 8, yAt(softKl) - 10)}
          textAnchor={currentLeft ? "end" : "start"}
        >
          {currentText}
        </text>
      </svg>
    </div>
  );
}

/**
 * One context's next-character distributions: three bars a row (label frequency, teacher at T, student), a number
 * column for each, and a band over the characters the corpus never puts after this context.
 */
export function SoftTargetBars({
  labels,
  teacher,
  student,
  temperature,
  studentName,
  context,
}: {
  labels: number[];
  teacher: number[];
  student: number[];
  temperature: number;
  studentName: string;
  context: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const order = Array.from({ length: V }, (_, k) => k)
    .filter((k) => TINY_VOCAB[k] !== "<" && TINY_VOCAB[k] !== ">")
    .sort((a, b) => teacher[b] - teacher[a])
    .slice(0, 10);
  const rest = (values: number[]) => 1 - order.reduce((sum, k) => sum + values[k], 0);
  const rows = [
    ...order.map((k) => ({
      key: TINY_VOCAB[k],
      label: glyph(TINY_VOCAB[k]),
      l: labels[k],
      t: teacher[k],
      s: student[k],
      unseen: labels[k] === 0,
    })),
    { key: "rest", label: "rest", l: rest(labels), t: rest(teacher), s: rest(student), unseen: false },
  ];
  const peak = Math.max(...rows.flatMap((row) => [row.l, row.t, row.s]), 0.05);
  const column = 56;
  const left = 40;
  const right = Math.max(left + 120, width - column * 3 - 12);
  const rowHeight = 30;
  const top = 32;
  const bottom = top + rows.length * rowHeight;
  const wAt = (value: number) => (value / peak) * (right - left);
  const columnX = (index: number) => width - column * (2 - index) - 2;
  const ticks = ticksFor(0, peak, 4);
  const firstUnseen = rows.findIndex((row) => row.unseen);
  const series = [
    { id: "label", head: "labels" },
    { id: "teacher", head: "teacher" },
    { id: "student", head: "student" },
  ] as const;
  return (
    <div className="dp-soft" ref={ref}>
      <svg
        className="dp-soft-plot"
        width={width}
        height={bottom + 24}
        viewBox={`0 0 ${width} ${bottom + 24}`}
        role="img"
        aria-label={`Next-character distributions. ${rows
          .map(
            (row) =>
              `${row.label}: labels ${percent(row.l)}, teacher at T ${temperature} ${percent(row.t)}, ${studentName} ${percent(row.s)}`,
          )
          .join("; ")}.`}
      >
        <text x={left} y={12}>
          next character after “{glyph(context)}” · share of the time
        </text>
        {series.map((entry, index) => (
          <g key={entry.id}>
            <text className={`dp-soft-head is-${entry.id}`} x={columnX(index)} y={12} textAnchor="end">
              {entry.head}
            </text>
            <rect className={`dp-bar is-${entry.id}`} x={columnX(index) - 30} y={17} width={30} height={3} />
          </g>
        ))}
        {rows.map(
          (row, index) =>
            row.unseen && <rect key={row.key} className="dp-unseen-band" x={0} y={top + index * rowHeight} width={width} height={rowHeight} />,
        )}
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="dp-grid" x1={left + wAt(tick)} x2={left + wAt(tick)} y1={top - 4} y2={bottom + 3} />
            <text x={left + wAt(tick)} y={bottom + 16} textAnchor="middle">
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        {rows.map((row, index) => {
          const y = top + index * rowHeight;
          return (
            <g key={row.key}>
              <text className={`dp-char ${row.unseen ? "is-unseen" : ""}`.trim()} x={left - 9} y={y + 19} textAnchor="end">
                {row.label}
              </text>
              <rect className="dp-bar is-label" x={left} y={y + 4} width={wAt(row.l)} height={7} />
              <rect className="dp-bar is-teacher" x={left} y={y + 12} width={wAt(row.t)} height={7} />
              <rect className="dp-bar is-student" x={left} y={y + 20} width={wAt(row.s)} height={7} />
              <text className="dp-soft-value is-label" x={columnX(0)} y={y + 19} textAnchor="end">
                {percent(row.l)}
              </text>
              <text className="dp-soft-value is-teacher" x={columnX(1)} y={y + 19} textAnchor="end">
                {percent(row.t)}
              </text>
              <text className="dp-soft-value is-student" x={columnX(2)} y={y + 19} textAnchor="end">
                {percent(row.s)}
              </text>
            </g>
          );
        })}
        {firstUnseen >= 0 && (
          <text className="dp-unseen-label" x={right - 2} y={top + firstUnseen * rowHeight + 19} textAnchor="end">
            shaded rows: never follow “{glyph(context)}” in the corpus
          </text>
        )}
      </svg>
    </div>
  );
}

/** The teacher's probability on characters the corpus never shows after this context, at T = 1 and at the current T. */
export function TailMeter({ atOne, atT, temperature }: { atOne: number; atT: number; temperature: number }) {
  const tiles = [
    { id: "one", label: "Teacher mass on unseen next characters, T = 1", value: atOne },
    { id: "current", label: `Same, at T = ${temperature.toFixed(1)}`, value: atT },
  ];
  return (
    <div className="dp-tail">
      {tiles.map((tile) => (
        <div key={tile.id} className={`dp-tail__tile is-${tile.id}`}>
          <span>{tile.label}</span>
          <i aria-hidden="true">
            <b style={{ width: `${Math.min(100, tile.value * 100)}%` }} />
          </i>
          <strong>{percent(tile.value)}</strong>
          <small aria-hidden="true">
            <span>0%</span>
            <span>of the teacher’s guess</span>
            <span>100%</span>
          </small>
        </div>
      ))}
    </div>
  );
}

/** Perplexity against the share removed, with the current setting marked on the curve of the current mode. */
export function PruneChart({
  sweep,
  semi,
  base,
  share,
  mode,
  current,
}: {
  sweep: Record<"unstructured" | "structured", { x: number; y: number }[]>;
  semi: number;
  base: number;
  share: number;
  mode: PruneMode;
  /** Perplexity at the current share and mode. */
  current: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const left = 40;
  const right = Math.max(left + 200, width - 96);
  const top = 26;
  const bottom = 214;
  const peak = Math.max(...sweep.structured.map((point) => point.y), semi);
  const ticks = ticksFor(base, peak, 4).filter((tick) => tick - base > (peak - base) * 0.12);
  const yMax = Math.max(peak, ticks[ticks.length - 1] ?? peak) * 1.02;
  const yMin = base - (yMax - base) * 0.05;
  const xAt = (value: number) => left + (value / 0.9) * (right - left);
  const yAt = (value: number) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);
  const line = (points: { x: number; y: number }[]) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(point.x)},${yAt(point.y)}`).join(" ");
  const markerX = mode === "semi" ? 0.5 : share;
  const markerText = `your setting · ${Math.round(markerX * 100)}% removed · perplexity ${current.toFixed(2)}`;
  const textWidth = markerText.length * 6.5;
  // The label sits at the top of the guide line, on the side with room, clear of the curves at the bottom.
  const labelLeft = xAt(markerX) + 8 + textWidth > right || markerX >= 0.5;
  const lastUnstructured = sweep.unstructured[sweep.unstructured.length - 1];
  const lastStructured = sweep.structured[sweep.structured.length - 1];
  return (
    <div className="dp-prune-chart" ref={ref}>
      <svg
        className="dp-mask dp-prune-plot"
        width={width}
        height={252}
        viewBox={`0 0 ${width} 252`}
        role="img"
        aria-label={`Perplexity against share of weights removed. Unstructured: ${sweep.unstructured.map((point) => `${Math.round(point.x * 100)}% ${point.y.toFixed(2)}`).join(", ")}. Structured: ${sweep.structured.map((point) => `${Math.round(point.x * 100)}% ${point.y.toFixed(2)}`).join(", ")}. 2:4 semi-structured at 50%: ${semi.toFixed(2)}. Your setting: ${Math.round(markerX * 100)}% removed, perplexity ${current.toFixed(2)}.`}
      >
        <text x={left} y={11}>
          perplexity on the harbor corpus · lower is better
        </text>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="dp-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {+tick.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="dp-baseline" x1={left} x2={right} y1={yAt(base)} y2={yAt(base)} />
        <text x={left - 6} y={yAt(base) + 3.5} textAnchor="end">
          {base.toFixed(1)}
        </text>
        <text className="dp-baseline-label" x={left + 5} y={yAt(base) - 8}>
          unpruned teacher {base.toFixed(2)}
        </text>
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((tick) => (
          <text key={tick} x={xAt(tick)} y={bottom + 15} textAnchor="middle">
            {Math.round(tick * 100)}%
          </text>
        ))}
        <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <text x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
          share of the 900 weights set to zero
        </text>
        <line className="marker-guide" x1={xAt(markerX)} x2={xAt(markerX)} y1={top} y2={bottom} />
        <path className={`dp-curve is-unstructured ${mode === "unstructured" ? "is-active" : ""}`.trim()} d={line(sweep.unstructured)} />
        <path className={`dp-curve is-structured ${mode === "structured" ? "is-active" : ""}`.trim()} d={line(sweep.structured)} />
        <text className="dp-curve-label is-unstructured" x={right + 8} y={yAt(lastUnstructured.y) + 4}>
          unstructured
        </text>
        <text className="dp-curve-label is-structured" x={right + 8} y={yAt(lastStructured.y) + 4}>
          structured
        </text>
        <text className="dp-curve-label is-structured" x={right + 8} y={yAt(lastStructured.y) + 17}>
          (whole rows)
        </text>
        <path className="dp-semi" d={`M${xAt(0.5)},${yAt(semi) - 6} l6,6 l-6,6 l-6,-6 Z`} />
        {mode !== "semi" && (
          <text className="dp-curve-label is-semi" x={xAt(0.5) + 10} y={yAt(semi) - 7}>
            2:4 · {semi.toFixed(2)}
          </text>
        )}
        <circle className={`dp-current is-${mode}`} cx={xAt(markerX)} cy={yAt(current)} r={mode === "semi" ? 10 : 6} />
        <text
          className="dp-current-label"
          x={xAt(markerX) + (labelLeft ? -7 : 7)}
          y={top + 9}
          textAnchor={labelLeft ? "end" : "start"}
        >
          {markerText}
        </text>
      </svg>
    </div>
  );
}

/** The 30 by 30 table as cells: kept weights shaded by magnitude, zeroed weights as empty outlines. */
export function MaskGrid({ mask, source, removedRows }: { mask: boolean[]; source: Float32Array; removedRows: number[] }) {
  const cell = 12;
  const left = 34;
  const top = 34;
  let peak = 0;
  for (const value of source) peak = Math.max(peak, Math.abs(value));
  const kept = mask.filter(Boolean).length;
  const gone = new Set(removedRows);
  const size = V * cell;
  return (
    <svg
      className="dp-mask dp-mask-grid"
      width={left + size + 4}
      height={top + size + 4}
      viewBox={`0 0 ${left + size + 4} ${top + size + 4}`}
      role="img"
      aria-label={`Weight mask: ${kept} of ${mask.length} weights kept. Rows are the context character, columns the next character. Filled cells are kept, shaded by magnitude; empty outlined cells are pruned.`}
    >
      <text className="dp-mask-axis" x={left + size / 2} y={10} textAnchor="middle">
        next character →
      </text>
      <text className="dp-mask-axis" transform={`translate(10 ${top + size / 2}) rotate(-90)`} textAnchor="middle">
        ← context character
      </text>
      {Array.from({ length: V }, (_, index) => (
        <g key={index}>
          <text className={gone.has(index) ? "is-gone" : undefined} x={left - 5} y={top + index * cell + 9.5} textAnchor="end">
            {glyph(TINY_VOCAB[index])}
          </text>
          <text x={left + index * cell + cell / 2} y={top - 6} textAnchor="middle">
            {glyph(TINY_VOCAB[index])}
          </text>
        </g>
      ))}
      {mask.map((keep, index) => {
        const row = Math.floor(index / V);
        const column = index % V;
        const x = left + column * cell;
        const y = top + row * cell;
        return keep ? (
          <rect
            key={index}
            className="dp-cell is-kept"
            x={x + 0.5}
            y={y + 0.5}
            width={cell - 1}
            height={cell - 1}
            style={{ fillOpacity: 0.18 + 0.82 * Math.sqrt(Math.abs(source[index]) / (peak || 1)) }}
          />
        ) : (
          <rect key={index} className="dp-cell is-pruned" x={x + 1.5} y={y + 1.5} width={cell - 3} height={cell - 3} />
        );
      })}
    </svg>
  );
}

/** Small fixed patterns, one per pruning shape: 1 is a kept cell, 0 a zeroed one. */
const SHAPES: { id: PruneMode; name: string; saves: string; cells: string[] }[] = [
  {
    id: "unstructured",
    name: "Zeros anywhere",
    saves:
      "The matrix keeps its shape, and a dense multiply still touches every cell: it saves storage with a sparse format, and little time.",
    cells: ["11011101", "10111011", "11101110", "01110111", "11011011", "10111101", "11110110", "01101111"],
  },
  {
    id: "semi",
    name: "Two of every four",
    saves: "Each group of four down a column keeps two. Sparse tensor cores can skip that pattern, for up to about twice the matmul throughput.",
    cells: ["10101010", "01100101", "10010110", "01011001", "01010101", "10100110", "10011001", "01101010"],
  },
  {
    id: "structured",
    name: "Whole rows",
    saves: "The matrix itself shrinks, so every computer does less work, at a higher cost in quality per weight removed.",
    cells: ["11111111", "00000000", "11111111", "11111111", "00000000", "00000000", "11111111", "11111111"],
  },
];

/** What each pruning shape looks like and what it saves, with the selected shape marked. */
export function ShapeKey({ mode }: { mode: PruneMode }) {
  const cell = 7;
  return (
    <ul className="dp-shapes" aria-label="The three pruning shapes and what each one saves">
      {SHAPES.map((shape) => (
        <li key={shape.id} className={shape.id === mode ? "is-current" : undefined} aria-current={shape.id === mode ? "true" : undefined}>
          <svg width={8 * cell} height={8 * cell} viewBox={`0 0 ${8 * cell} ${8 * cell}`} aria-hidden="true">
            {shape.cells.flatMap((line, row) =>
              [...line].map((kept, column) => (
                <rect
                  key={`${row}-${column}`}
                  className={kept === "1" ? "dp-cell is-kept" : "dp-cell is-pruned"}
                  x={column * cell + (kept === "1" ? 0.5 : 1)}
                  y={row * cell + (kept === "1" ? 0.5 : 1)}
                  width={cell - (kept === "1" ? 1 : 2)}
                  height={cell - (kept === "1" ? 1 : 2)}
                />
              )),
            )}
          </svg>
          <p>
            <strong>{shape.name}</strong>
            <span>{shape.saves}</span>
          </p>
        </li>
      ))}
    </ul>
  );
}

const STEP_NAMES: Record<ComposeStage["id"], (detail: { rank: number; share: number; bits: number }) => string> = {
  teacher: () => "",
  student: ({ rank }) => `distil to rank ${rank}`,
  pruned: ({ share }) => `prune ${Math.round(share * 100)}%`,
  quantized: ({ bits }) => `quantize to ${bits} bits`,
};

/**
 * The compose pipeline as a ladder: one row per stage with its stored size and its perplexity on shared scales, and
 * between rows the step that was taken and what it saved and cost.
 */
export function StageLadder({
  stages,
  rank,
  share,
  bits,
}: {
  stages: ComposeStage[];
  rank: number;
  share: number;
  bits: number;
}) {
  const teacher = stages[0];
  const low = Math.min(...stages.map((stage) => stage.perplexity));
  const high = Math.max(low + 0.5, ...stages.map((stage) => stage.perplexity));
  const at = (value: number) => ((value - low) / (high - low)) * 100;
  const signedPercent = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)}%`;
  return (
    <div className="dp-ladder">
      <div className="dp-ladder__head" aria-hidden="true">
        <span>stage</span>
        <span>stored size · shorter is smaller</span>
        <span>perplexity · further right is worse</span>
      </div>
      <div
        className="dp-ladder__rows"
        role="list"
        aria-label="Stored size in bytes and perplexity after each stage, with what each step saved and cost"
      >
      {stages.map((stage, index) => {
        const previous = stages[index - 1];
        return (
          <div key={stage.id} className="dp-ladder__item" role="listitem">
            {previous && (
              <p className="dp-ladder__step">
                <b>↓ {STEP_NAMES[stage.id]({ rank, share, bits })}</b>
                <span className="is-size">
                  {stage.bytes <= previous.bytes ? "−" : "+"}
                  {withCommas(Math.abs(previous.bytes - stage.bytes))} B
                </span>
                <span className="is-ppl">
                  {stage.perplexity >= previous.perplexity ? "+" : "−"}
                  {Math.abs(stage.perplexity - previous.perplexity).toFixed(3)} perplexity
                </span>
              </p>
            )}
            <div className={`dp-ladder__row ${stage.id === "quantized" ? "is-final" : ""}`.trim()}>
              <strong>{stage.label}</strong>
              <div className="dp-ladder__size">
                <i>
                  <b style={{ width: `${Math.min(100, (stage.bytes / teacher.bytes) * 100)}%` }} />
                </i>
                <em>{withCommas(stage.bytes)} B</em>
              </div>
              <div className="dp-ladder__ppl">
                <i>
                  <b style={{ left: `${at(teacher.perplexity)}%`, width: `${Math.max(0, at(stage.perplexity) - at(teacher.perplexity))}%` }} />
                  <u style={{ left: `${at(stage.perplexity)}%` }} />
                </i>
                <em>
                  <span className="sr-only">perplexity </span>
                  {stage.perplexity.toFixed(3)}{" "}
                  <small>({signedPercent((stage.perplexity / teacher.perplexity - 1) * 100)} on the teacher)</small>
                </em>
              </div>
              <small className="dp-ladder__detail">
                {withCommas(stage.values)} values · {(stage.bits / stage.values).toFixed(1)} bits each, metadata and mask included
              </small>
            </div>
          </div>
        );
      })}
      </div>
    </div>
  );
}
