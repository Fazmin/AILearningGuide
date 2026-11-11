import { useMeasuredWidth } from "@app/module-sdk";
import { POINTS, type Algorithm } from "./fit";

export type GalleryItem = {
  id: Algorithm;
  /** Predicted class (or cluster id) at each cell of a size × size grid, row by row from the top. */
  cells: readonly number[];
  size: number;
  /** Training points classified correctly, or null when the method reads no labels. */
  hits: number | null;
  /** Which of the 18 points this method gets wrong. */
  misses: readonly boolean[];
};

/**
 * All eight methods on the same 18 points, side by side. Each thumbnail is that method's decision surface at
 * the current k and Tree depth, so the shapes of the eight boundaries can be compared in one look.
 */
export function MethodGallery({
  items,
  current,
  onSelect,
}: {
  items: readonly GalleryItem[];
  current: Algorithm;
  onSelect: (id: Algorithm) => void;
}) {
  return (
    <div className="cml-gallery" role="group" aria-label="The eight methods' decision surfaces, side by side">
      {items.map((item) => {
        // One rect per horizontal run of equal cells keeps eight surfaces light to draw.
        const runs: { x: number; y: number; width: number; value: number }[] = [];
        for (let row = 0; row < item.size; row += 1) {
          let start = 0;
          for (let column = 1; column <= item.size; column += 1) {
            const value = item.cells[row * item.size + start];
            if (column === item.size || item.cells[row * item.size + column] !== value) {
              runs.push({ x: start, y: row, width: column - start, value });
              start = column;
            }
          }
        }
        const clusters = item.hits === null;
        return (
          <button
            key={item.id}
            type="button"
            className={`cml-thumb${item.id === current ? " is-current" : ""}`}
            aria-pressed={item.id === current}
            aria-label={`${item.id}: ${
              item.hits === null ? "clusters, no labels used" : `${item.hits} of ${POINTS.length} training points right`
            }`}
            onClick={() => onSelect(item.id)}
          >
            <svg viewBox={`0 0 ${item.size} ${item.size}`} aria-hidden="true" preserveAspectRatio="none">
              {runs.map((run) => (
                <rect
                  key={`${run.x}-${run.y}`}
                  x={run.x}
                  y={run.y}
                  width={run.width + 0.04}
                  height="1.04"
                  className={clusters ? `is-cluster-${run.value % 4}` : run.value === 1 ? "is-one" : "is-zero"}
                />
              ))}
              {POINTS.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x * item.size}
                  cy={(1 - point.y) * item.size}
                  r={item.size * 0.03}
                  className={`${point.label === 1 ? "is-pos" : "is-neg"}${item.misses[index] ? " is-miss" : ""}`}
                />
              ))}
            </svg>
            <strong>{item.id}</strong>
            <small>{item.hits === null ? "no labels" : `${item.hits}/${POINTS.length} right`}</small>
          </button>
        );
      })}
    </div>
  );
}

/**
 * One number on a line with a cut: the score a linear method gives the probe, which side of the cut it falls
 * on, and (for the SVM) the street around the cut.
 */
export function ScoreLine({
  value,
  min,
  max,
  cut,
  cutLabel,
  valueLabel,
  axisLabel,
  band,
  bandLabel,
  ticks,
}: {
  value: number;
  min: number;
  max: number;
  cut: number;
  cutLabel: string;
  valueLabel: string;
  axisLabel: string;
  band?: readonly [number, number];
  bandLabel?: string;
  ticks: readonly number[];
}) {
  // Drawn at the card's real width, so its labels keep their size in the two-column layout.
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(620);
  const width = Math.min(672, Math.max(300, measured));
  const height = 104;
  const left = 22;
  const right = width - 22;
  const top = 40;
  const bottom = 64;
  const x = (v: number) => left + ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * (right - left);
  const off = value < min || value > max;
  const side = value >= cut ? 1 : 0;
  const anchor = x(value) > right - 90 ? "end" : x(value) < left + 90 ? "start" : "middle";
  return (
    <div className="cml-scoreline-frame" ref={frameRef}>
    <svg
      className="cml-scoreline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${axisLabel}. The probe reads ${valueLabel}. The cut is at ${cutLabel}, so the probe falls on the class ${side} side.`}
    >
      <rect className="cml-scoreline__zero" x={left} y={top} width={x(cut) - left} height={bottom - top} />
      <rect className="cml-scoreline__one" x={x(cut)} y={top} width={right - x(cut)} height={bottom - top} />
      {band && (
        <>
          <rect className="cml-scoreline__band" x={x(band[0])} y={top} width={x(band[1]) - x(band[0])} height={bottom - top} />
          <text className="cml-scoreline__note" x={(x(band[0]) + x(band[1])) / 2} y={bottom + 28} textAnchor="middle">
            {bandLabel}
          </text>
        </>
      )}
      <text className="cml-scoreline__side" x={left + 6} y={top + 16}>
        class 0
      </text>
      <text className="cml-scoreline__side" x={right - 6} y={top + 16} textAnchor="end">
        class 1
      </text>
      <line className="cml-scoreline__cut" x1={x(cut)} x2={x(cut)} y1={top - 8} y2={bottom + 5} />
      <text className="cml-scoreline__note is-cut" x={x(cut)} y={top - 12} textAnchor="middle">
        cut: {cutLabel}
      </text>
      {ticks.map((tick) => (
        <text key={tick} className="cml-scoreline__tick" x={x(tick)} y={bottom + 14} textAnchor="middle">
          {tick}
        </text>
      ))}
      <text className="cml-scoreline__tick" x={right} y={height - 4} textAnchor="end">
        {axisLabel} →
      </text>
      <circle className="cml-scoreline__probe" cx={x(value)} cy={(top + bottom) / 2} r="8" />
      <circle className="cml-scoreline__dot" cx={x(value)} cy={(top + bottom) / 2} r="2.4" />
      <text className="cml-scoreline__value" x={x(value)} y={14} textAnchor={anchor}>
        probe: {valueLabel}
        {off ? " (off this line)" : ""}
      </text>
      <line className="cml-scoreline__stem" x1={x(value)} x2={x(value)} y1={18} y2={(top + bottom) / 2 - 9} />
    </svg>
    </div>
  );
}

/** A vote drawn as ballots: one mark per voter, filled for class 1 and hollow for class 0, then the tally. */
export function Ballots({
  title,
  votes,
  result,
  rule,
}: {
  title: string;
  votes: ReadonlyArray<{ label: number; caption: string }>;
  result: number;
  rule: string;
}) {
  const ones = votes.filter((vote) => vote.label === 1).length;
  return (
    <div
      className="cml-ballots"
      role="img"
      aria-label={`${title}. ${ones} of ${votes.length} vote class 1, so the probe is class ${result}.`}
    >
      <span className="cml-ballots__title">{title}</span>
      <ol>
        {votes.map((vote, index) => (
          <li key={index} className={vote.label === 1 ? "is-one" : "is-zero"}>
            <i />
            <b>class {vote.label}</b>
            <small>{vote.caption}</small>
          </li>
        ))}
      </ol>
      <p>
        <strong>
          {ones} of {votes.length}
        </strong>{" "}
        vote class 1 → the probe is <strong>class {result}</strong>
        <small>{rule}</small>
      </p>
    </div>
  );
}

/** The split questions the probe answers on its way from the root to one leaf. */
export function TreeWalk({
  steps,
  leafLabel,
  leafCount,
  leafPositives,
}: {
  steps: ReadonlyArray<{ axis: "x" | "y"; threshold: number; value: number; yes: boolean }>;
  leafLabel: number;
  leafCount: number;
  leafPositives: number;
}) {
  return (
    <ol
      className="cml-walk"
      aria-label={`The probe's path through the tree: ${steps
        .map((step) => `${step.axis} below ${step.threshold.toFixed(2)}, ${step.yes ? "yes" : "no"}`)
        .join("; ")}; leaf of class ${leafLabel}.`}
    >
      {steps.map((step, index) => (
        <li key={index} className={step.yes ? "is-yes" : "is-no"}>
          <small>{index === 0 ? "root asks" : `cut ${index + 1} asks`}</small>
          <b>
            {step.axis} &lt; {step.threshold.toFixed(2)}?
          </b>
          <span>
            {step.axis} = {step.value.toFixed(2)} → {step.yes ? "yes" : "no"}
          </span>
        </li>
      ))}
      <li className={`is-leaf ${leafLabel === 1 ? "is-one" : "is-zero"}`}>
        <small>leaf</small>
        <b>class {leafLabel}</b>
        <span>
          {leafPositives} of {leafCount} points here are class 1
        </span>
      </li>
    </ol>
  );
}

/** Two amounts as one bar split in proportion: the share the first part takes is the reading. */
export function ShareBar({
  title,
  parts,
}: {
  title: string;
  parts: ReadonlyArray<{ label: string; value: number; tone: "one" | "zero" }>;
}) {
  const total = parts.reduce((sum, part) => sum + part.value, 0) || 1;
  return (
    <div
      className="cml-share"
      role="img"
      aria-label={`${title}: ${parts.map((part) => `${part.label} ${((part.value / total) * 100).toFixed(1)} percent`).join(", ")}.`}
    >
      <span className="cml-ballots__title">{title}</span>
      <div>
        {parts.map((part) => (
          <i key={part.label} className={`is-${part.tone}`} style={{ width: `${(part.value / total) * 100}%` }}>
            <b>
              {part.label} · {((part.value / total) * 100).toFixed(1)}%
            </b>
          </i>
        ))}
        <u aria-hidden="true" />
      </div>
    </div>
  );
}
