import { useMemo, useState, type PointerEvent, type ReactNode } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  type ModuleContext,
} from "@app/module-sdk";
import {
  buildTaskVectors,
  lossesOf,
  lossesFor,
  LORA_ALPHA,
  LORA_RANK,
  type TaskLosses,
  type VectorSource,
} from "./lab";
import {
  addToBase,
  cosine,
  mergeTaskVectors,
  norm,
  pickCoordinates,
  signStats,
  type MergeMethod,
  type MergeParams,
} from "./merge";

const V = TINY_VOCAB_SIZE;
const glyph = (index: number) => (TINY_VOCAB[index] === " " ? "␣" : TINY_VOCAB[index]);
const coordinateName = (index: number) => `${glyph(Math.floor(index / V))}→${glyph(index % V)}`;

const METHODS: ReadonlyArray<{ value: MergeMethod; label: string }> = [
  { value: "linear", label: "Linear blend" },
  { value: "task", label: "Task arithmetic" },
  { value: "ties", label: "TIES" },
  { value: "dare", label: "DARE" },
  { value: "slerp", label: "SLERP" },
];

const clamp = (value: unknown, low: number, high: number, fallback: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const signed = (value: number, digits = 3) =>
  Number.isFinite(value) ? `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}` : "—";

/* -------------------------------------------------------------------------- */
/* Task-vector map                                                             */
/* -------------------------------------------------------------------------- */

const CELL = 10;
const GUTTER = 15;
const MAP_SIZE = V * CELL;
const MAP_VIEW = GUTTER + MAP_SIZE + 2;
const EPSILON = 1e-6;

/** Square-root shading, so the many small entries of a task vector still show their sign. */
const cellFill = (value: number, scale: number) => {
  const magnitude = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
  if (magnitude < 0.004) return "var(--panel-muted)";
  return `color-mix(in srgb, ${value >= 0 ? "var(--forward)" : "var(--loss)"} ${Math.round(
    6 + Math.sqrt(magnitude) * 94,
  )}%, var(--panel-solid))`;
};

const isConflict = (a: number, b: number) =>
  Math.abs(a) > EPSILON && Math.abs(b) > EPSILON && Math.sign(a) !== Math.sign(b);

function TaskVectorMap({
  label,
  caption,
  values,
  other,
  scale,
  focus,
  onFocus,
}: {
  label: string;
  caption: string;
  values: Float32Array;
  other: Float32Array;
  scale: number;
  /** The cell ringed on both maps: the one under the pointer, or the lab's default pick. */
  focus: number | null;
  onFocus: (index: number | null) => void;
}) {
  const { cells, conflicts } = useMemo(() => {
    const drawn: ReactNode[] = [];
    let count = 0;
    for (let row = 0; row < V; row += 1) {
      for (let column = 0; column < V; column += 1) {
        const index = row * V + column;
        const x = GUTTER + column * CELL;
        const y = GUTTER + row * CELL;
        drawn.push(
          <rect key={index} x={x} y={y} width={CELL} height={CELL} style={{ fill: cellFill(values[index], scale) }} />,
        );
        if (isConflict(values[index], other[index])) {
          count += 1;
          drawn.push(
            <circle key={`c-${index}`} className="mrg-map__conflict" cx={x + CELL / 2} cy={y + CELL / 2} r={2} />,
          );
        }
      }
    }
    return { cells: drawn, conflicts: count };
  }, [values, other, scale]);

  const focusRow = focus === null ? -1 : Math.floor(focus / V);
  const focusColumn = focus === null ? -1 : focus % V;

  const locate = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const unit = box.width / MAP_VIEW;
    const column = Math.floor(((event.clientX - box.left) / unit - GUTTER) / CELL);
    const row = Math.floor(((event.clientY - box.top) / unit - GUTTER) / CELL);
    onFocus(row >= 0 && row < V && column >= 0 && column < V ? row * V + column : null);
  };

  return (
    <figure className="mrg-map">
      <figcaption>
        <strong>{label}</strong>
        <span>{caption}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${MAP_VIEW} ${MAP_VIEW}`}
        role="img"
        aria-label={`${label}. ${caption}. Rows are the previous character, columns the next character. Largest entry magnitude ${scale.toFixed(2)}. ${conflicts} entries disagree in sign with the other task vector and carry a dot.`}
        onPointerMove={locate}
        onPointerLeave={() => onFocus(null)}
      >
        {Array.from({ length: V }, (_, index) => (
          <g key={`l-${index}`} className="mrg-map__label">
            <text
              className={index === focusColumn ? "is-focus" : undefined}
              x={GUTTER + index * CELL + CELL / 2}
              y={GUTTER - 4}
              textAnchor="middle"
            >
              {glyph(index)}
            </text>
            <text
              className={index === focusRow ? "is-focus" : undefined}
              x={GUTTER - 4}
              y={GUTTER + index * CELL + CELL / 2 + 3}
              textAnchor="end"
            >
              {glyph(index)}
            </text>
          </g>
        ))}
        {cells}
        <rect className="mrg-map__frame" x={GUTTER} y={GUTTER} width={MAP_SIZE} height={MAP_SIZE} />
        {focus !== null && (
          <g className="mrg-map__focus">
            <line x1={GUTTER} x2={GUTTER + focusColumn * CELL} y1={GUTTER + focusRow * CELL + CELL / 2} y2={GUTTER + focusRow * CELL + CELL / 2} />
            <line x1={GUTTER + focusColumn * CELL + CELL / 2} x2={GUTTER + focusColumn * CELL + CELL / 2} y1={GUTTER} y2={GUTTER + focusRow * CELL} />
            <rect x={GUTTER + focusColumn * CELL - 2} y={GUTTER + focusRow * CELL - 2} width={CELL + 4} height={CELL + 4} rx={2} />
          </g>
        )}
      </svg>
    </figure>
  );
}

/** One line under the two maps: the ringed cell's entry in each task vector, and whether they agree. */
function CellProbe({
  index,
  pinned,
  taskA,
  taskB,
}: {
  index: number | null;
  /** True when the pointer is on a map; false when the lab picked the cell itself. */
  pinned: boolean;
  taskA: Float32Array;
  taskB: Float32Array;
}) {
  if (index === null) {
    return (
      <p className="mrg-probe">
        <span className="mrg-probe__hint">Point at a cell on either map to read that weight in both task vectors.</span>
      </p>
    );
  }
  const a = taskA[index];
  const b = taskB[index];
  const word = (value: number) => (Math.abs(value) <= EPSILON ? "unchanged" : value > 0 ? "raises" : "lowers");
  const changedA = Math.abs(a) > EPSILON;
  const changedB = Math.abs(b) > EPSILON;
  const verdict = isConflict(a, b)
    ? { text: "conflict: opposite signs", tone: "conflict" }
    : changedA && changedB
      ? { text: "agree: same sign", tone: "agree" }
      : changedA || changedB
        ? { text: "only one fine-tune changed it", tone: "single" }
        : { text: "unchanged by both", tone: "none" };
  return (
    <p className="mrg-probe">
      <b>{coordinateName(index)}</b>
      <span className="mrg-probe__a">
        τ recipes {signed(a, 2)} <small className={`is-${word(a)}`}>{word(a)}</small>
      </span>
      <span className="mrg-probe__b">
        τ proverbs {signed(b, 2)} <small className={`is-${word(b)}`}>{word(b)}</small>
      </span>
      <strong className={`mrg-probe__verdict is-${verdict.tone}`}>{verdict.text}</strong>
      <span className="mrg-probe__hint">
        {pinned ? "the ringed cell on both maps" : "ringed on both maps · point at any cell"}
      </span>
    </p>
  );
}

/** All 900 weights as one bar: where the two fine-tunes agree, where they conflict, and what neither touched. */
function OverlapBar({ agree, conflicts, single, untouched }: { agree: number; conflicts: number; single: number; untouched: number }) {
  const total = agree + conflicts + single + untouched;
  const overlap = agree + conflicts;
  const segments = [
    { id: "agree", count: agree, name: "agree in sign" },
    { id: "conflict", count: conflicts, name: "conflict" },
    { id: "single", count: single, name: "changed by one only" },
    { id: "untouched", count: untouched, name: "unchanged by both" },
  ].filter((segment) => segment.count > 0);
  return (
    <div
      className="mrg-overlap"
      role="img"
      aria-label={`Of ${total} weights, both fine-tunes changed ${overlap}: ${agree} agree in sign and ${conflicts} conflict. ${single} were changed by one fine-tune only and ${untouched} by neither.`}
    >
      {overlap > 0 && (
        <div className="mrg-overlap__brace" style={{ width: `${(overlap / total) * 100}%` }}>
          <span>{overlap} changed by both fine-tunes</span>
        </div>
      )}
      <div className="mrg-overlap__bar">
        {segments.map((segment) => (
          <i key={segment.id} className={`mrg-overlap__segment is-${segment.id}`} style={{ flexGrow: segment.count }}>
            <b>{segment.count}</b>
          </i>
        ))}
      </div>
      <div className="mrg-overlap__key">
        {segments.map((segment) => (
          <span key={segment.id}>
            <i className={`mrg-overlap__chip is-${segment.id}`} />
            {segment.count} {segment.name}
            {(segment.id === "agree" || segment.id === "conflict") && overlap > 0
              ? ` (${Math.round((segment.count / overlap) * 100)}%)`
              : ""}
          </span>
        ))}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Coordinate strip                                                            */
/* -------------------------------------------------------------------------- */

function CoordinateStrip({
  method,
  coordinates,
  taskA,
  taskB,
  processed,
  merged,
  elected,
}: {
  method: MergeMethod;
  coordinates: { index: number; conflict: boolean }[];
  taskA: Float32Array;
  taskB: Float32Array;
  processed: [Float32Array, Float32Array];
  merged: Float32Array;
  elected: Int8Array;
}) {
  const width = 664;
  const height = 300;
  const left = 44;
  const top = 46;
  const bottom = 206;
  const zero = (top + bottom) / 2;
  const column = (width - left) / Math.max(1, coordinates.length);
  const bar = Math.min(15, column / 4);
  const gap = 2;
  const ceiling = Math.max(
    1e-6,
    ...coordinates.flatMap(({ index }) => [
      Math.abs(taskA[index]),
      Math.abs(taskB[index]),
      Math.abs(processed[0][index]),
      Math.abs(processed[1][index]),
      Math.abs(merged[index]),
    ]),
  );
  const yOf = (value: number) => zero - (value / ceiling) * (zero - top);
  const showsProcessing = method === "ties" || method === "dare";
  const conflictCount = coordinates.filter((entry) => entry.conflict).length;
  const average = (index: number) => (taskA[index] + taskB[index]) / 2;
  const showsAverage = coordinates.some(({ index }) => Math.abs(merged[index] - average(index)) > 0.005);

  const describe = coordinates
    .map(({ index, conflict }) => {
      const parts = [
        `${coordinateName(index)} ${conflict ? "conflict" : "agree"}`,
        `recipes ${taskA[index].toFixed(2)}`,
        `proverbs ${taskB[index].toFixed(2)}`,
        `merged ${merged[index].toFixed(2)}`,
      ];
      return parts.join(", ");
    })
    .join("; ");

  const inputBar = (
    key: string,
    x: number,
    raw: number,
    kept: number,
    tone: "a" | "b",
  ) => {
    const removed = showsProcessing && kept === 0 && Math.abs(raw) > 1e-6;
    const value = showsProcessing ? kept : raw;
    return (
      <g key={key}>
        {removed ? (
          <rect
            className={`mrg-strip__bar mrg-strip__bar--${tone} is-removed`}
            x={x}
            y={Math.min(yOf(raw), zero)}
            width={bar}
            height={Math.max(1, Math.abs(yOf(raw) - zero))}
          />
        ) : (
          <rect
            className={`mrg-strip__bar mrg-strip__bar--${tone}`}
            x={x}
            y={Math.min(yOf(value), zero)}
            width={bar}
            height={Math.max(1, Math.abs(yOf(value) - zero))}
          />
        )}
        {method === "dare" && !removed && Math.abs(raw) > 1e-6 && (
          <line className="mrg-strip__original" x1={x - 2} x2={x + bar + 2} y1={yOf(raw)} y2={yOf(raw)} />
        )}
      </g>
    );
  };

  if (coordinates.length === 0) {
    return (
      <p className="lab-note">
        No weight is changed by both task vectors, so there are no conflicts or agreements to draw.
        An untrained fine-tune is a zero task vector.
      </p>
    );
  }

  const split = left + conflictCount * column;
  const half = ceiling / 2;
  const agreeCount = coordinates.length - conflictCount;

  return (
    <div className="mrg-strip">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Ten coordinates of the two task vectors and the merge. ${describe}.`}>
        {conflictCount > 0 && (
          <>
            <rect className="mrg-strip__band" x={left} y={4} width={split - left - 3} height={height - 6} rx={6} />
            <text className="mrg-strip__group" x={left + 9} y={21}>
              {conflictCount === 1 ? "Largest sign conflict" : `${conflictCount} largest sign conflicts`}
              <tspan className="mrg-strip__group-note"> · r and p point opposite ways</tspan>
            </text>
          </>
        )}
        {agreeCount > 0 && (
          <text className="mrg-strip__group" x={split + 9} y={21}>
            {agreeCount === 1 ? "Largest agreement" : `${agreeCount} largest agreements`}
            <tspan className="mrg-strip__group-note"> · r and p point the same way</tspan>
          </text>
        )}
        {[ceiling, half, -half, -ceiling].map((value) => (
          <line key={value} className="mrg-strip__grid" x1={left} x2={width} y1={yOf(value)} y2={yOf(value)} />
        ))}
        <line className="mrg-strip__zero" x1={left} x2={width} y1={zero} y2={zero} />
        <text className="mrg-strip__axis" x={left - 6} y={top + 3.5} textAnchor="end">
          +{ceiling.toFixed(2)}
        </text>
        <text className="mrg-strip__axis" x={left - 6} y={zero + 3.5} textAnchor="end">
          0
        </text>
        <text className="mrg-strip__axis" x={left - 6} y={bottom + 3.5} textAnchor="end">
          −{ceiling.toFixed(2)}
        </text>
        <text className="mrg-strip__axis-title" transform={`translate(11 ${zero}) rotate(-90)`} textAnchor="middle">
          change to the logit
        </text>
        {coordinates.map(({ index }, position) => {
          const center = left + position * column + column / 2;
          const xA = center - bar * 1.5 - gap;
          const xB = center - bar * 0.5;
          const xM = center + bar * 0.5 + gap;
          const sign = elected[index] ?? 0;
          const mergedY = yOf(merged[index]);
          const averageY = yOf(average(index));
          const mid = xM + bar / 2;
          return (
            <g key={index}>
              {inputBar(`a-${index}`, xA, taskA[index], processed[0][index], "a")}
              {inputBar(`b-${index}`, xB, taskB[index], processed[1][index], "b")}
              <rect
                className="mrg-strip__bar mrg-strip__bar--merged"
                x={xM}
                y={Math.min(mergedY, zero)}
                width={bar}
                height={Math.max(1, Math.abs(mergedY - zero))}
              />
              {showsAverage && (
                <path
                  className="mrg-strip__average"
                  d={`M${mid - 4.5},${averageY} L${mid},${averageY - 4.5} L${mid + 4.5},${averageY} L${mid},${averageY + 4.5} Z`}
                />
              )}
              {method === "ties" && sign !== 0 && (
                <text
                  className="mrg-strip__elected"
                  x={mid}
                  y={sign > 0 ? Math.min(mergedY, zero) - 7 : Math.max(mergedY, zero) + 15}
                  textAnchor="middle"
                >
                  {sign > 0 ? "▲ +" : "▼ −"}
                </text>
              )}
              <text className="mrg-strip__letter mrg-strip__letter--a" x={xA + bar / 2} y={bottom + 32} textAnchor="middle">
                r
              </text>
              <text className="mrg-strip__letter mrg-strip__letter--b" x={xB + bar / 2} y={bottom + 32} textAnchor="middle">
                p
              </text>
              <text className="mrg-strip__letter mrg-strip__letter--merged" x={mid} y={bottom + 32} textAnchor="middle">
                m
              </text>
              <text className="mrg-strip__name" x={center} y={height - 28} textAnchor="middle">
                {coordinateName(index)}
              </text>
              <text className="mrg-strip__value" x={center} y={height - 10} textAnchor="middle">
                m {merged[index] >= 0 ? "+" : "−"}
                {Math.abs(merged[index]).toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mrg-legend" aria-hidden="true">
        <span><i className="mrg-swatch mrg-swatch--a" /><b>r</b> τ recipes (left bar)</span>
        <span><i className="mrg-swatch mrg-swatch--b" /><b>p</b> τ proverbs (middle bar)</span>
        <span><i className="mrg-swatch mrg-swatch--merged" /><b>m</b> merged (right bar, value printed)</span>
        {showsAverage && (
          <span><i className="mrg-swatch mrg-swatch--average" />where a plain 50/50 average would end the merged bar</span>
        )}
        {showsProcessing && (
          <span><i className="mrg-swatch mrg-swatch--removed" />{method === "ties" ? "trimmed away" : "dropped"}</span>
        )}
        {method === "dare" && (
          <span><i className="mrg-swatch mrg-swatch--original" />height before rescaling</span>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Loss plane                                                                  */
/* -------------------------------------------------------------------------- */

interface PlanePoint {
  id: string;
  label: string;
  losses: TaskLosses;
  kind: "base" | "a" | "b" | "ideal" | "current" | "other";
}

interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

const boxesHit = (a: Box, b: Box) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

/** Label positions tried in order: close to the marker first, then further out on a leader line. */
const LABEL_ANGLES = [-28, 28, 152, 208, -90, 90, -60, 60, 120, 240, 0, 180, -45, -14, 14, 45];
const LABEL_RADII = [13, 26, 40, 56, 74, 94, 116, 140, 166];

function LossPlane({
  points,
  linearPath,
  taskPath,
  specialistRecipes,
  specialistProverbs,
}: {
  points: PlanePoint[];
  linearPath: TaskLosses[];
  taskPath: TaskLosses[];
  /** Each specialist's loss on its own corpus: the two guide lines every gap is measured from. */
  specialistRecipes: number;
  specialistProverbs: number;
}) {
  const width = 664;
  const height = 400;
  const left = 54;
  const right = width - 14;
  const top = 14;
  const bottom = height - 44;
  const anchors = points.filter((point) => point.kind !== "other");
  const xs = [...anchors.map((point) => point.losses.recipes), ...linearPath.map((point) => point.recipes)];
  const ys = [...anchors.map((point) => point.losses.proverbs), ...linearPath.map((point) => point.proverbs)];
  const others = points.filter((point) => point.kind === "other");
  const anchorMaxX = Math.max(...xs);
  const anchorMaxY = Math.max(...ys);
  const xMin = Math.min(...xs, ...others.map((point) => point.losses.recipes)) - 0.04;
  const yMin = Math.min(...ys, ...others.map((point) => point.losses.proverbs)) - 0.04;
  // A wild merge (DARE at a high drop rate) is pinned to the edge rather than allowed to rescale the plot.
  const xMax = Math.min(Math.max(anchorMaxX, ...others.map((point) => point.losses.recipes)), anchorMaxX + 0.15) + 0.04;
  const yMax = Math.min(Math.max(anchorMaxY, ...others.map((point) => point.losses.proverbs)), anchorMaxY + 0.15) + 0.04;
  const rawX = (value: number) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
  const rawY = (value: number) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);
  const xAt = (value: number) => rawX(Math.min(xMax, value));
  const yAt = (value: number) => rawY(Math.min(yMax, value));
  const ticks = (low: number, high: number) => [0, 0.25, 0.5, 0.75, 1].map((f) => low + f * (high - low));
  const path = (list: TaskLosses[]) =>
    list
      .map((point, index) => `${index === 0 ? "M" : "L"}${rawX(point.recipes).toFixed(1)},${rawY(point.proverbs).toFixed(1)}`)
      .join(" ");
  const isClipped = (point: PlanePoint) => point.losses.recipes > xMax || point.losses.proverbs > yMax;
  const inPlot = (x: number, y: number) => x >= left && x <= right && y >= top && y <= bottom;

  const marker = (point: PlanePoint) => {
    const x = xAt(point.losses.recipes);
    const y = yAt(point.losses.proverbs);
    const clipped = isClipped(point);
    switch (point.kind) {
      case "base":
        return <circle className="mrg-plane__base" cx={x} cy={y} r={5} />;
      case "a":
        return <rect className="mrg-plane__a" x={x - 5} y={y - 5} width={10} height={10} />;
      case "b":
        return <path className="mrg-plane__b" d={`M${x},${y - 6} L${x + 6},${y + 5} L${x - 6},${y + 5} Z`} />;
      case "ideal":
        return (
          <g className="mrg-plane__ideal">
            <line x1={x - 5} x2={x + 5} y1={y - 5} y2={y + 5} />
            <line x1={x - 5} x2={x + 5} y1={y + 5} y2={y - 5} />
          </g>
        );
      case "current":
        return <circle className={`mrg-plane__current ${clipped ? "is-clipped" : ""}`} cx={x} cy={y} r={7} />;
      default:
        return <circle className={`mrg-plane__other ${clipped ? "is-clipped" : ""}`} cx={x} cy={y} r={4} />;
    }
  };

  // Where λ = 0.5, 1 and 1.5 fall on the task-arithmetic path, so the path reads as a direction of travel.
  const lambdaTicks = [5, 10, 15]
    .filter((step) => step < taskPath.length)
    .map((step) => ({ lambda: step / 10, x: rawX(taskPath[step].recipes), y: rawY(taskPath[step].proverbs) }))
    .filter((tick) => inPlot(tick.x, tick.y));

  // Label placement: every marker and both paths are obstacles. Each label tries positions around
  // its marker, nearest first, and takes the first that overlaps nothing; far ones get a leader line.
  const obstacles: Box[] = points.map((point) => ({
    x0: xAt(point.losses.recipes) - 7,
    x1: xAt(point.losses.recipes) + 7,
    y0: yAt(point.losses.proverbs) - 7,
    y1: yAt(point.losses.proverbs) + 7,
  }));
  for (const list of [linearPath, taskPath]) {
    for (let index = 1; index < list.length; index += 1) {
      const x0 = rawX(list[index - 1].recipes);
      const y0 = rawY(list[index - 1].proverbs);
      const x1 = rawX(list[index].recipes);
      const y1 = rawY(list[index].proverbs);
      const pieces = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 5));
      for (let piece = 0; piece <= pieces; piece += 1) {
        const x = x0 + ((x1 - x0) * piece) / pieces;
        const y = y0 + ((y1 - y0) * piece) / pieces;
        if (inPlot(x, y)) obstacles.push({ x0: x - 2, x1: x + 2, y0: y - 2, y1: y + 2 });
      }
    }
  }
  for (const tick of lambdaTicks) obstacles.push({ x0: tick.x - 3, x1: tick.x + 3, y0: tick.y - 3, y1: tick.y + 3 });

  const place = (x: number, y: number, text: string, charWidth: number, reach: number) => {
    const textWidth = text.length * charWidth;
    let fallback: { box: Box; tx: number; ty: number; anchor: "start" | "end" | "middle"; radius: number; angle: number } | null = null;
    for (const radius of LABEL_RADII.slice(0, reach)) {
      for (const angle of LABEL_ANGLES) {
        const radians = (angle * Math.PI) / 180;
        const dx = Math.cos(radians);
        const dy = Math.sin(radians);
        const anchor = dx > 0.3 ? ("start" as const) : dx < -0.3 ? ("end" as const) : ("middle" as const);
        const tx = x + dx * radius;
        const ty = y + dy * radius + (anchor === "middle" ? (dy > 0 ? 9 : -2) : 4);
        const x0 = anchor === "start" ? tx : anchor === "end" ? tx - textWidth : tx - textWidth / 2;
        const box = { x0: x0 - 2, x1: x0 + textWidth + 2, y0: ty - 10, y1: ty + 3 };
        const candidate = { box, tx, ty, anchor, radius, angle };
        if (box.x0 < left + 2 || box.x1 > right - 1 || box.y0 < top + 1 || box.y1 > bottom - 2) continue;
        fallback ??= candidate;
        if (!obstacles.some((other) => boxesHit(box, other))) return candidate;
      }
    }
    return fallback;
  };

  const rank = (kind: PlanePoint["kind"]) => (kind === "current" ? 0 : kind === "other" ? 2 : 1);
  const order = [...points].sort((a, b) => rank(a.kind) - rank(b.kind));
  const placed = order.flatMap((point) => {
    const x = xAt(point.losses.recipes);
    const y = yAt(point.losses.proverbs);
    const text = isClipped(point)
      ? `${point.label} (${point.losses.recipes.toFixed(2)}, ${point.losses.proverbs.toFixed(2)}) off chart`
      : point.label;
    const spot = place(x, y, text, point.kind === "current" ? 7.2 : 6.2, LABEL_RADII.length);
    if (!spot) return [];
    // A crowded corner (one runaway merge squeezes the rest) can leave no free spot. The table
    // below still names every method, so an unplaceable method label is dropped, not overlapped.
    if (point.kind === "other" && obstacles.some((other) => boxesHit(spot.box, other))) return [];
    obstacles.push(spot.box);
    const radians = (spot.angle * Math.PI) / 180;
    const leader =
      spot.radius > 14
        ? {
            x1: x + Math.cos(radians) * 8,
            y1: y + Math.sin(radians) * 8,
            x2: x + Math.cos(radians) * (spot.radius - 3),
            y2: y + Math.sin(radians) * (spot.radius - 3),
          }
        : null;
    return [{ point, text, x: spot.tx, y: spot.ty, anchor: spot.anchor, leader }];
  });
  const lambdaLabels = lambdaTicks.flatMap((tick) => {
    const text = `λ ${tick.lambda}`;
    const spot = place(tick.x, tick.y, text, 6, 2);
    if (!spot || obstacles.some((other) => boxesHit(spot.box, other))) return [];
    obstacles.push(spot.box);
    return [{ ...tick, text, tx: spot.tx, ty: spot.ty, anchor: spot.anchor }];
  });

  const describe = points
    .map((point) => `${point.label}: recipes ${point.losses.recipes.toFixed(3)}, proverbs ${point.losses.proverbs.toFixed(3)}`)
    .join("; ");

  const spread = Math.max(...points.map((point) => Math.abs(point.losses.recipes - points[0].losses.recipes) + Math.abs(point.losses.proverbs - points[0].losses.proverbs)));
  if (spread < 1e-6) {
    return (
      <p className="lab-note mrg-plane__empty">
        Every point is the base, at recipes {points[0].losses.recipes.toFixed(3)} and proverbs{" "}
        {points[0].losses.proverbs.toFixed(3)}: both task vectors are zero, so every merge adds nothing.
        Raise Recipe epochs or Proverb epochs.
      </p>
    );
  }

  const guideX = rawX(specialistRecipes);
  const guideY = rawY(specialistProverbs);

  return (
    <div className="mrg-plane">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Loss on both tasks for every merge. Lower and further left is better. ${describe}.`}>
        <defs>
          <clipPath id="mrg-plane-clip">
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {ticks(xMin, xMax).map((value) => (
          <g key={`x-${value}`}>
            <line className="mrg-plane__grid" x1={xAt(value)} x2={xAt(value)} y1={top} y2={bottom} />
            <text className="mrg-plane__tick" x={xAt(value)} y={bottom + 15} textAnchor="middle">
              {value.toFixed(2)}
            </text>
          </g>
        ))}
        {ticks(yMin, yMax).map((value) => (
          <g key={`y-${value}`}>
            <line className="mrg-plane__grid" x1={left} x2={right} y1={yAt(value)} y2={yAt(value)} />
            <text className="mrg-plane__tick" x={left - 7} y={yAt(value) + 3.5} textAnchor="end">
              {value.toFixed(2)}
            </text>
          </g>
        ))}
        <line className="mrg-plane__axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="mrg-plane__axis" x1={left} x2={left} y1={top} y2={bottom} />
        <text className="mrg-plane__axis-label" x={(left + right) / 2} y={height - 8} textAnchor="middle">
          ← better · recipe loss (nats/char) · worse →
        </text>
        <text
          className="mrg-plane__axis-label"
          transform={`translate(13 ${(top + bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          ← better · proverb loss (nats/char) · worse →
        </text>
        {inPlot(guideX, guideY) && (
          <>
            <line className="mrg-plane__guide" x1={guideX} x2={guideX} y1={top} y2={bottom} />
            <line className="mrg-plane__guide" x1={left} x2={right} y1={guideY} y2={guideY} />
          </>
        )}
        <g clipPath="url(#mrg-plane-clip)">
          <path className="mrg-plane__path mrg-plane__path--linear" d={path(linearPath)} />
          <path className="mrg-plane__path mrg-plane__path--task" d={path(taskPath)} />
        </g>
        {lambdaTicks.map((tick) => (
          <circle key={`lt-${tick.lambda}`} className="mrg-plane__lambda" cx={tick.x} cy={tick.y} r={2.6} />
        ))}
        {lambdaLabels.map((tick) => (
          <text key={`ll-${tick.lambda}`} className="mrg-plane__lambda-label" x={tick.tx} y={tick.ty} textAnchor={tick.anchor}>
            {tick.text}
          </text>
        ))}
        {placed.map(
          (label) =>
            label.leader && (
              <line
                key={`leader-${label.point.id}`}
                className="mrg-plane__leader"
                x1={label.leader.x1}
                y1={label.leader.y1}
                x2={label.leader.x2}
                y2={label.leader.y2}
              />
            ),
        )}
        {points.filter((point) => point.kind !== "current").map((point) => (
          <g key={point.id}>{marker(point)}</g>
        ))}
        {points.filter((point) => point.kind === "current").map((point) => (
          <g key={point.id}>{marker(point)}</g>
        ))}
        {placed.map((label) => (
          <text
            key={`label-${label.point.id}`}
            className={`mrg-plane__label mrg-plane__label--${label.point.kind}`}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
          >
            {label.text}
          </text>
        ))}
      </svg>
      <div className="mrg-legend" aria-hidden="true">
        <span><i className="mrg-key mrg-key--current" />selected merge</span>
        <span><i className="mrg-key mrg-key--other" />other methods at the same settings</span>
        <span><i className="mrg-line mrg-line--linear" />linear blend, t from 0 to 1</span>
        <span><i className="mrg-line mrg-line--task" />task sum, λ from 0 to 1.5</span>
        <span><i className="mrg-line mrg-line--guide" />each specialist&rsquo;s own loss: a gap is the distance from that line</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Gap bars                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * One task's gap to its specialist, drawn from zero. The solid part is what the merge costs with
 * the other task vector removed; the hatched part is what the other task then added or took off.
 */
function GapBar({ gap, interference, max }: { gap: number; interference: number; max: number }) {
  const share = (value: number) => Math.min(1, Math.max(0, value / max));
  const span = (from: number, to: number) => ({
    left: `${share(Math.min(from, to)) * 100}%`,
    width: `${(share(Math.max(from, to)) - share(Math.min(from, to))) * 100}%`,
  });
  const split = Number.isFinite(interference);
  const own = split ? gap - interference : gap;
  return (
    <span className={`mrg-gap__track${gap > max ? " is-clipped" : ""}`} aria-hidden="true">
      <i className={`mrg-gap__own${split ? "" : " is-whole"}`} style={span(0, own)} />
      {split && interference > 0.0005 && <i className="mrg-gap__hurt" style={span(own, gap)} />}
      {split && interference < -0.0005 && <i className="mrg-gap__helped" style={span(gap, own)} />}
      <i className="mrg-gap__end" style={{ left: `${share(gap) * 100}%` }} />
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Explore                                                                     */
/* -------------------------------------------------------------------------- */

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const source: VectorSource = state.source === "lora" ? "lora" : "full";
  const epochsA = Math.round(clamp(state.epochsA, 0, 40, 20));
  const epochsB = Math.round(clamp(state.epochsB, 0, 40, 20));
  const method: MergeMethod = METHODS.some((entry) => entry.value === state.method)
    ? (state.method as MergeMethod)
    : "linear";
  const mix = clamp(state.mix, 0, 1, 0.5);
  const lambda = clamp(state.lambda, 0, 1.5, 1);
  const density = clamp(state.density, 0.05, 1, 0.2);
  const dropRate = clamp(state.dropRate, 0, 0.9, 0.5);
  const maskSeed = Math.round(clamp(state.maskSeed, 1, 999, 4));

  const vectors = useMemo(() => buildTaskVectors(source, epochsA, epochsB), [source, epochsA, epochsB]);
  const params: MergeParams = useMemo(
    () => ({ mix, lambda, density, dropRate, maskSeed }),
    [mix, lambda, density, dropRate, maskSeed],
  );

  const outcomes = useMemo(
    () => METHODS.map((entry) => ({ ...entry, outcome: lossesFor(vectors, entry.value, params) })),
    [vectors, params],
  );
  const current = outcomes.find((entry) => entry.value === method)!;
  const result =
    current.outcome.result ??
    mergeTaskVectors("linear", vectors.taskA, vectors.taskB, { ...params, mix: 0.5 })!;

  const paths = useMemo(() => {
    const linearPath: TaskLosses[] = [];
    const taskPath: TaskLosses[] = [];
    const fixed: MergeParams = { mix: 0.5, lambda: 1, density: 1, dropRate: 0, maskSeed: 1 };
    const served = (kind: MergeMethod, overrides: Partial<MergeParams>) =>
      lossesOf(
        addToBase(
          vectors.baseWeights,
          mergeTaskVectors(kind, vectors.taskA, vectors.taskB, { ...fixed, ...overrides })!.vector,
        ),
      );
    for (let step = 0; step <= 10; step += 1) linearPath.push(served("linear", { mix: step / 10 }));
    for (let step = 0; step <= 15; step += 1) taskPath.push(served("task", { lambda: step / 10 }));
    return { linearPath, taskPath };
  }, [vectors]);

  const stats = useMemo(() => signStats(vectors.taskA, vectors.taskB), [vectors]);
  const similarity = useMemo(() => cosine(vectors.taskA, vectors.taskB), [vectors]);
  const mapScale = useMemo(() => {
    let maximum = 0;
    for (let index = 0; index < vectors.taskA.length; index += 1) {
      maximum = Math.max(maximum, Math.abs(vectors.taskA[index]), Math.abs(vectors.taskB[index]));
    }
    return maximum;
  }, [vectors]);
  const coordinates = useMemo(() => pickCoordinates(vectors.taskA, vectors.taskB, 5), [vectors]);

  // The cell ringed on both maps. Until the pointer picks one, it is the conflict where both
  // fine-tunes pushed hardest, so the two maps show one opposite pair without any interaction.
  const [hovered, setHovered] = useState<number | null>(null);
  const defaultFocus = useMemo(() => {
    let best = -1;
    let bestSize = 0;
    for (let index = 0; index < vectors.taskA.length; index += 1) {
      if (!isConflict(vectors.taskA[index], vectors.taskB[index])) continue;
      const size = Math.min(Math.abs(vectors.taskA[index]), Math.abs(vectors.taskB[index]));
      if (size > bestSize) {
        best = index;
        bestSize = size;
      }
    }
    return best >= 0 ? best : null;
  }, [vectors]);
  const focus = hovered ?? defaultFocus;

  const methodLabel = METHODS.find((entry) => entry.value === method)!.label;
  const sourceLabel = source === "lora" ? `LoRA rank ${LORA_RANK}, alpha ${LORA_ALPHA}` : "full fine-tunes";

  // One shared scale for every gap bar in the table. A single runaway method (DARE at a high drop
  // rate) is pinned at the end of its track instead of flattening the other four.
  const gapScale = (() => {
    const perMethod = outcomes
      .map((entry) => {
        const gapA = entry.outcome.recipes - vectors.specialistA.recipes;
        const gapB = entry.outcome.proverbs - vectors.specialistB.proverbs;
        const ownA = Number.isFinite(entry.outcome.interferenceA) ? gapA - entry.outcome.interferenceA : gapA;
        const ownB = Number.isFinite(entry.outcome.interferenceB) ? gapB - entry.outcome.interferenceB : gapB;
        return Math.max(gapA, gapB, ownA, ownB);
      })
      .filter((value) => Number.isFinite(value))
      .sort((left, right) => right - left);
    const largest = perMethod[0] ?? 0;
    const second = perMethod[1] ?? largest;
    return Math.max(0.05, Math.min(largest, second * 2.2));
  })();

  const planePoints: PlanePoint[] = [
    { id: "base", label: "harbor base", losses: vectors.base, kind: "base" },
    { id: "a", label: "recipe specialist", losses: vectors.specialistA, kind: "a" },
    { id: "b", label: "proverb specialist", losses: vectors.specialistB, kind: "b" },
    {
      id: "ideal",
      label: "two separate models",
      losses: { recipes: vectors.specialistA.recipes, proverbs: vectors.specialistB.proverbs },
      kind: "ideal",
    },
    ...outcomes
      .filter((entry) => entry.value !== method && entry.outcome.result)
      .map((entry) => ({
        id: entry.value,
        label: entry.label,
        losses: { recipes: entry.outcome.recipes, proverbs: entry.outcome.proverbs },
        kind: "other" as const,
      })),
    {
      id: "current",
      label: methodLabel,
      losses: { recipes: current.outcome.recipes, proverbs: current.outcome.proverbs },
      kind: "current",
    },
  ];

  const formula = (() => {
    switch (method) {
      case "linear":
        return {
          expression: `θ = θ_base + ${(1 - mix).toFixed(2)}·τ_recipes + ${mix.toFixed(2)}·τ_proverbs`,
          detail:
            "Identical to averaging the two fine-tuned checkpoints with weights 1 − t and t, because both share θ_base.",
        };
      case "task":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·(τ_recipes + τ_proverbs)`,
          detail: "Both task vectors at full strength, then one shared scale λ. At λ = 0.5 this is the average.",
        };
      case "ties":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·mean_agreeing(elect(trim_${Math.round(density * 100)}%(τ_recipes), trim_${Math.round(density * 100)}%(τ_proverbs)))`,
          detail: `Trim keeps the largest ${Math.round(density * 100)}% of each vector by magnitude. Elect picks, per weight, the sign with more summed mass. The merge averages only the kept entries with that sign.`,
        };
      case "dare":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·(m_r⊙τ_recipes + m_p⊙τ_proverbs) / (1 − ${dropRate.toFixed(2)})`,
          detail: `Each entry survives its mask with probability ${(1 - dropRate).toFixed(2)}; survivors are multiplied by ${(1 / (1 - dropRate)).toFixed(2)} so every entry keeps its expected value.`,
        };
      case "slerp":
        return {
          expression: `θ = θ_base + (sin(${(1 - mix).toFixed(2)}Ω)·τ_recipes + sin(${mix.toFixed(2)}Ω)·τ_proverbs) / sin Ω`,
          detail: Number.isFinite(result.angle)
            ? `Ω = ${result.angle.toFixed(1)}°, the angle between the two task vectors. The arc keeps the blend's length where a straight line would shrink it.`
            : "SLERP needs two non-zero task vectors. Train both before reading this one.",
        };
    }
  })();

  const interferenceNote = (() => {
    const a = current.outcome.interferenceA;
    const b = current.outcome.interferenceB;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      return "SLERP has no version with one task removed, so its interference is not separated here. Compare its point with Linear blend at the same t instead.";
    }
    const verdict = (value: number) =>
      value > 0.02 ? "hurt it" : value < -0.02 ? "helped it" : "barely changed it";
    return `Adding the proverb vector ${verdict(a)} on recipes (${signed(a)} nats), and adding the recipe vector ${verdict(b)} on proverbs (${signed(b)} nats). The rest of each gap to its specialist is the merge shrinking that task's own vector.`;
  })();

  const methodControls = (() => {
    const mixControl = (
      <RangeControl
        label="Blend t"
        min={0}
        max={1}
        step={0.05}
        value={mix}
        format={(value) => `${Math.round((1 - value) * 100)}% recipes · ${Math.round(value * 100)}% proverbs`}
        onChange={(value) => setState({ mix: value })}
      />
    );
    const lambdaControl = (
      <RangeControl
        label="Scale λ"
        min={0}
        max={1.5}
        step={0.05}
        value={lambda}
        format={(value) => value.toFixed(2)}
        onChange={(value) => setState({ lambda: value })}
      />
    );
    switch (method) {
      case "linear":
      case "slerp":
        return mixControl;
      case "task":
        return lambdaControl;
      case "ties":
        return (
          <>
            <RangeControl
              label="Keep top"
              min={0.05}
              max={1}
              step={0.05}
              value={density}
              format={(value) => `${Math.round(value * 100)}% of each vector`}
              onChange={(value) => setState({ density: value })}
            />
            {lambdaControl}
          </>
        );
      case "dare":
        return (
          <>
            <RangeControl
              label="Drop rate p"
              min={0}
              max={0.9}
              step={0.05}
              value={dropRate}
              format={(value) => `${Math.round(value * 100)}% dropped · ×${(1 / (1 - value)).toFixed(2)}`}
              onChange={(value) => setState({ dropRate: value })}
            />
            {lambdaControl}
            <button
              type="button"
              className="quiet-action mrg-reseed"
              onClick={() => {
                setState({ maskSeed: (maskSeed % 999) + 1 });
                narrate("New random drop masks.");
              }}
            >
              New random mask · #{maskSeed}
            </button>
          </>
        );
    }
  })();

  return (
    <div className="mrg-lab">
      <LabSurface label="Task vectors" className="mrg-vectors-card">
        <SurfaceHeading
          kicker={`Harbor base · both fine-tunes start from it · ${sourceLabel}`}
          title="Two task vectors: τ = θ_tuned − θ_base"
          aside={
            <span className="tg-badge">
              cosine {Number.isFinite(similarity) ? similarity.toFixed(2) : "—"}
            </span>
          }
        />
        <div className="mrg-controls">
          <SegmentedControl
            label="Task vectors from"
            value={source}
            options={[
              { value: "full", label: "Full fine-tunes" },
              { value: "lora", label: `LoRA rank ${LORA_RANK}` },
            ]}
            onChange={(value) => {
              setState({ source: value });
              narrate(value === "lora" ? "Task vectors from LoRA adapters." : "Task vectors from full fine-tunes.");
            }}
          />
          <RangeControl
            label="Recipe epochs"
            min={0}
            max={40}
            step={2}
            value={epochsA}
            format={(value) => (value === 0 ? "untrained · τ = 0" : `${value} passes`)}
            onChange={(value) => setState({ epochsA: value })}
          />
          <RangeControl
            label="Proverb epochs"
            min={0}
            max={40}
            step={2}
            value={epochsB}
            format={(value) => (value === 0 ? "untrained · τ = 0" : `${value} passes`)}
            onChange={(value) => setState({ epochsB: value })}
          />
        </div>
        <p className="mrg-maps__lead">
          Each cell is one of the {V * V} weights: its row is the previous character, its column the next
          character. Both maps use one colour scale.
        </p>
        <div className="mrg-maps">
          <TaskVectorMap
            label="τ recipes"
            caption={`‖τ‖ = ${norm(vectors.taskA).toFixed(2)}`}
            values={vectors.taskA}
            other={vectors.taskB}
            scale={mapScale}
            focus={focus}
            onFocus={setHovered}
          />
          <TaskVectorMap
            label="τ proverbs"
            caption={`‖τ‖ = ${norm(vectors.taskB).toFixed(2)}`}
            values={vectors.taskB}
            other={vectors.taskA}
            scale={mapScale}
            focus={focus}
            onFocus={setHovered}
          />
        </div>
        <CellProbe index={focus} pinned={hovered !== null} taskA={vectors.taskA} taskB={vectors.taskB} />
        <div className="mrg-legend mrg-scale" aria-hidden="true">
          <span>lowers that logit</span>
          <span className="mrg-scale__ramp">
            <small>−{mapScale.toFixed(2)}</small>
            {[-1, -0.45, -0.15, -0.03, 0, 0.03, 0.15, 0.45, 1].map((part) => (
              <i key={part} style={{ background: cellFill(part * mapScale, mapScale) }} />
            ))}
            <small>+{mapScale.toFixed(2)}</small>
          </span>
          <span>raises that logit</span>
          <span><i className="mrg-swatch mrg-swatch--zero" />unchanged</span>
          <span><i className="mrg-swatch mrg-swatch--conflict" />dot: the two vectors disagree in sign</span>
        </div>
        <OverlapBar
          agree={stats.overlap - stats.conflicts}
          conflicts={stats.conflicts}
          single={stats.onlyA + stats.onlyB}
          untouched={V * V - stats.overlap - stats.onlyA - stats.onlyB}
        />
        <div className="metric-row mrg-metrics">
          <Metric label="Recipe specialist, recipes" value={vectors.specialistA.recipes.toFixed(3)} tone="forward" />
          <Metric label="Proverb specialist, proverbs" value={vectors.specialistB.proverbs.toFixed(3)} tone="gradient" />
          <Metric label="Overlapping entries" value={`${stats.overlap} of ${V * V}`} />
          <Metric label="Sign conflicts" value={stats.overlap ? `${stats.conflicts} (${Math.round((stats.conflicts / stats.overlap) * 100)}%)` : "0"} tone="loss" />
        </div>
      </LabSurface>

      <LabSurface label="Merge recipe" className="mrg-recipe-card">
        <SurfaceHeading
          kicker="Arithmetic on the two vectors · no gradient step"
          title={`${methodLabel}, coordinate by coordinate`}
          aside={<span className="tg-badge">‖τ_merged‖ {norm(result.vector).toFixed(2)}</span>}
        />
        <div className="mrg-method">
          <SegmentedControl
            label="Merge method"
            value={method}
            options={METHODS.map((entry) => ({ value: entry.value, label: entry.label }))}
            onChange={(value) => {
              setState({ method: value });
              narrate(`${METHODS.find((entry) => entry.value === value)?.label ?? value} selected.`);
            }}
          />
          <div className="mrg-method__controls">{methodControls}</div>
        </div>
        <FormulaWithValues
          label="Served weights"
          expression={formula.expression}
          result={`recipes ${current.outcome.recipes.toFixed(3)} · proverbs ${current.outcome.proverbs.toFixed(3)}`}
          detail={formula.detail}
          tone="attention"
        />
        <p className="mrg-strip__title">
          Ten of the 900 weights, picked for size by |τ_recipes| + |τ_proverbs|. Each has three bars in one
          order: recipes, proverbs, merged.
        </p>
        <CoordinateStrip
          method={method}
          coordinates={coordinates}
          taskA={vectors.taskA}
          taskB={vectors.taskB}
          processed={result.processed}
          merged={result.vector}
          elected={result.elected}
        />
      </LabSurface>

      <LabSurface label="Two task losses" className="mrg-plane-card">
        <SurfaceHeading
          kicker="Measured on both corpora · lower left is better"
          title="Where each merge lands"
          aside={<span className="tg-badge">{methodLabel}</span>}
        />
        <LossPlane
          points={planePoints}
          linearPath={paths.linearPath}
          taskPath={paths.taskPath}
          specialistRecipes={vectors.specialistA.recipes}
          specialistProverbs={vectors.specialistB.proverbs}
        />
        <table className="mrg-table">
          <caption>
            Every method at the current settings. Gap is merged loss minus that task&rsquo;s specialist;
            interference is merged loss minus the same merge with the other task vector set to zero.
            Each bar draws one gap from zero on a shared scale, split into the part that stays with the
            other task removed and the part the other task caused.
          </caption>
          <thead>
            <tr>
              <th scope="col">Merge</th>
              <th scope="col">Recipes</th>
              <th scope="col">Gap</th>
              <th scope="col">Interference</th>
              <th scope="col">Proverbs</th>
              <th scope="col">Gap</th>
              <th scope="col">Interference</th>
            </tr>
          </thead>
          <tbody>
            {outcomes.map((entry) => {
              const gapA = entry.outcome.recipes - vectors.specialistA.recipes;
              const gapB = entry.outcome.proverbs - vectors.specialistB.proverbs;
              const tone = (value: number) =>
                !Number.isFinite(value) ? "" : value > 0.0005 ? "is-hurt" : value < -0.0005 ? "is-helped" : "";
              return (
                <tr key={entry.value} className={entry.value === method ? "is-selected" : ""} aria-current={entry.value === method ? "true" : undefined}>
                  <th scope="row">
                    {entry.label}
                    {entry.value === method ? " ◆" : ""}
                  </th>
                  <td>{entry.outcome.recipes.toFixed(3)}</td>
                  <td className="mrg-gap">
                    <GapBar gap={gapA} interference={entry.outcome.interferenceA} max={gapScale} />
                    <span>{signed(gapA)}</span>
                  </td>
                  <td className={tone(entry.outcome.interferenceA)}>{signed(entry.outcome.interferenceA)}</td>
                  <td>{entry.outcome.proverbs.toFixed(3)}</td>
                  <td className="mrg-gap">
                    <GapBar gap={gapB} interference={entry.outcome.interferenceB} max={gapScale} />
                    <span>{signed(gapB)}</span>
                  </td>
                  <td className={tone(entry.outcome.interferenceB)}>{signed(entry.outcome.interferenceB)}</td>
                </tr>
              );
            })}
            <tr className="is-reference">
              <th scope="row">Specialists</th>
              <td>{vectors.specialistA.recipes.toFixed(3)}</td>
              <td className="mrg-gap">
                <span className="mrg-gap__track is-zero" aria-hidden="true">
                  <i className="mrg-gap__end" style={{ left: "0%" }} />
                </span>
                <span>—</span>
              </td>
              <td>—</td>
              <td>{vectors.specialistB.proverbs.toFixed(3)}</td>
              <td className="mrg-gap">
                <span className="mrg-gap__track is-zero" aria-hidden="true">
                  <i className="mrg-gap__end" style={{ left: "0%" }} />
                </span>
                <span>—</span>
              </td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
        <div className="mrg-legend mrg-gap-legend" aria-hidden="true">
          <span><i className="mrg-gap__chip is-own" />gap with the other task vector removed (this task&rsquo;s own vector was shrunk)</span>
          <span><i className="mrg-gap__chip is-hurt" />added by the other task</span>
          <span><i className="mrg-gap__chip is-helped" />taken off by the other task</span>
          <span><i className="mrg-gap__chip is-end" />the gap · one scale for every bar, 0 to {gapScale.toFixed(2)} nats; » marks a longer gap</span>
        </div>
        <p className="lab-note">{interferenceNote}</p>
      </LabSurface>
    </div>
  );
}
