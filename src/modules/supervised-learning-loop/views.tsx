import { useEffect, useRef, useState } from "react";
import { sigmoid, type Apartment, type SliceSegment, type TreeNode } from "./fit";

/** The rooms × park window every picture on this page shares. */
export const X_MIN = 0.6;
export const X_MAX = 4.4;
export const Y_MIN = -0.2;
export const Y_MAX = 1.2;

export const toPlotX = (rooms: number) => (rooms - X_MIN) / (X_MAX - X_MIN);
export const toPlotY = (park: number) => (park - Y_MIN) / (Y_MAX - Y_MIN);
export const fromPlotX = (x: number) => X_MIN + x * (X_MAX - X_MIN);
export const fromPlotY = (y: number) => Y_MIN + y * (Y_MAX - Y_MIN);

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

const fmt = (value: number, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const signed = (value: number, digits = 3) => `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;
const plain = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(2));
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * Rows that share one (rooms, park) cell are set one above the other on the plot, so every mark stays
 * visible and clickable. Rooms is never moved, so a mark never changes sides of a rooms cut. The fit
 * ignores this.
 */
export const markOffsets = (apartments: readonly Apartment[]) => {
  const groups = new Map<string, Apartment[]>();
  apartments.forEach((row) => {
    const key = `${row.rooms}:${row.park}`;
    const members = groups.get(key);
    if (members) members.push(row);
    else groups.set(key, [row]);
  });
  const offsets = new Map<number, number>();
  groups.forEach((members) =>
    members.forEach((row, index) => offsets.set(row.id, (index - (members.length - 1) / 2) * 0.15)),
  );
  return offsets;
};

function Mark({ cx, cy, high, holdout, r = 7 }: { cx: number; cy: number; high: boolean; holdout: boolean; r?: number }) {
  const tone = high ? "is-high" : "is-low";
  if (!holdout) return <circle className={`sll-mark is-train ${tone}`} cx={cx} cy={cy} r={r} />;
  return (
    <>
      <circle className={`sll-mark-edge ${tone}`} cx={cx} cy={cy} r={r - 1.6} />
      <circle className={`sll-mark is-holdout ${tone}`} cx={cx} cy={cy} r={r - 1.6} />
    </>
  );
}

/** One glyph of the mark key, drawn with the same classes as the plot. */
function KeyMark({ high, holdout }: { high: boolean; holdout: boolean }) {
  return (
    <svg className="sll-key__glyph" viewBox="0 0 20 20" aria-hidden="true">
      <Mark cx={10} cy={10} high={high} holdout={holdout} />
    </svg>
  );
}

type Cell = { x: number; y: number; value: number; klass: number };

export function DecisionPlot({
  cells,
  grid,
  usesProbability,
  boundary,
  segments,
  apartments,
  selected,
  predictions,
  sliced,
  onSelect,
  label,
}: {
  /** True when color is in x, so the wash is only the slice at the selected row's color. */
  sliced: boolean;
  cells: readonly Cell[];
  grid: number;
  usesProbability: boolean;
  boundary: { x1: number; y1: number; x2: number; y2: number } | null;
  segments: readonly SliceSegment[];
  apartments: readonly Apartment[];
  selected: number;
  predictions: readonly number[];
  onSelect: (id: number) => void;
  label: string;
}) {
  const frame = useRef<SVGRectElement>(null);
  const W = 640;
  const H = 344;
  const L = 78;
  const R = 628;
  const T = 12;
  const B = 290;
  const pw = R - L;
  const ph = B - T;
  const sx = (rooms: number) => L + toPlotX(rooms) * pw;
  const sy = (park: number) => T + (1 - toPlotY(park)) * ph;
  const offsets = markOffsets(apartments);
  const cw = pw / grid;
  const ch = ph / grid;

  // Name each side of the rule where a wide enough run of one class crosses the empty band between the rails.
  const mid = Math.floor(grid / 2);
  const runs: Array<{ klass: number; start: number; end: number }> = [];
  for (let column = 0; column < grid; column += 1) {
    const klass = cells[mid * grid + column]?.klass ?? 0;
    const last = runs[runs.length - 1];
    if (last && last.klass === klass) last.end = column;
    else runs.push({ klass, start: column, end: column });
  }
  const regionLabels = [0, 1]
    .map((klass) => {
      const best = runs
        .filter((run) => run.klass === klass)
        .sort((a, b) => b.end - b.start - (a.end - a.start))[0];
      if (!best || best.end - best.start + 1 < 7) return null;
      return {
        klass,
        x: L + ((best.start + best.end + 1) / 2) * cw,
        text: klass === 1 ? "rule says high price" : "rule says low price",
      };
    })
    .filter((item): item is { klass: number; x: number; text: string } => item !== null);

  const current = apartments[selected];
  const marks = apartments.map((row, index) => {
    const p = predictions[index] ?? 0;
    return {
      row,
      cx: sx(row.rooms),
      cy: sy(row.park + (offsets.get(row.id) ?? 0)),
      wrong: (p >= 0.5 ? 1 : 0) !== row.price,
    };
  });
  const chosen = marks.find((mark) => mark.row.id === selected);

  return (
    <div className="sll-decision">
      <svg
        className="sll-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={label}
        onClick={(event) => {
          const rect = frame.current?.getBoundingClientRect();
          if (!rect || rect.width === 0) return;
          const px = ((event.clientX - rect.left) / rect.width) * pw + L;
          const py = ((event.clientY - rect.top) / rect.height) * ph + T;
          let best = selected;
          let bestDistance = Infinity;
          marks.forEach((mark) => {
            const distance = (mark.cx - px) ** 2 + (mark.cy - py) ** 2;
            if (distance < bestDistance) {
              best = mark.row.id;
              bestDistance = distance;
            }
          });
          onSelect(best);
        }}
      >
        <defs>
          <clipPath id="sll-plot-clip">
            <rect x={L} y={T} width={pw} height={ph} rx="6" />
          </clipPath>
        </defs>
        <g clipPath="url(#sll-plot-clip)">
          {cells.map((cell, index) => (
            <rect
              key={index}
              className={`sll-cell is-class-${cell.klass}`}
              x={L + cell.x * pw - cw / 2}
              y={T + (1 - cell.y) * ph - ch / 2}
              width={cw + 0.5}
              height={ch + 0.5}
              opacity={usesProbability ? 0.34 + clamp(cell.value, 0, 1) * 0.66 : cell.klass ? 0.92 : 0.5}
            />
          ))}
          {[1, 2, 3, 4].map((rooms) => (
            <line key={`v-${rooms}`} className="sll-gridline" x1={sx(rooms)} x2={sx(rooms)} y1={T} y2={B} />
          ))}
          {[0, 1].map((park) => (
            <line key={`h-${park}`} className="sll-railline" x1={L} x2={R} y1={sy(park)} y2={sy(park)} />
          ))}
          {boundary && (
            <line
              className="sll-boundary"
              x1={sx(boundary.x1)}
              y1={sy(boundary.y1)}
              x2={sx(boundary.x2)}
              y2={sy(boundary.y2)}
            />
          )}
          {segments.map((segment, index) =>
            segment.axis === "rooms" ? (
              <line
                key={`s-${index}`}
                className="sll-split"
                x1={sx(segment.threshold)}
                x2={sx(segment.threshold)}
                y1={sy(segment.to)}
                y2={sy(segment.from)}
              />
            ) : (
              <line
                key={`s-${index}`}
                className="sll-split"
                x1={sx(segment.from)}
                x2={sx(segment.to)}
                y1={sy(segment.threshold)}
                y2={sy(segment.threshold)}
              />
            ),
          )}
        </g>
        <rect ref={frame} className="sll-frame" x={L} y={T} width={pw} height={ph} rx="6" />
        {regionLabels.map((item) => (
          <text key={item.klass} className="sll-region" x={item.x} y={T + ph / 2 + 4} textAnchor="middle">
            {item.text}
          </text>
        ))}
        {[1, 2, 3, 4].map((rooms) => (
          <text key={rooms} className="sll-tick" x={sx(rooms)} y={B + 16} textAnchor="middle">
            {rooms} {rooms === 1 ? "room" : "rooms"}
          </text>
        ))}
        <text className="sll-axis" x={R} y={B + 36} textAnchor="end">
          rooms →
        </text>
        <text className="sll-tick" x={L - 9} y={sy(1) + 4} textAnchor="end">
          park
        </text>
        <text className="sll-tick" x={L - 9} y={sy(0) + 4} textAnchor="end">
          no park
        </text>
        {marks.map((mark) => (
          <g key={mark.row.id} className={sliced && mark.row.color !== current.color ? "sll-offslice" : undefined}>
            <title>{`${mark.row.name} · ${mark.row.split} · ${mark.row.price ? "high" : "low"} price`}</title>
            <Mark cx={mark.cx} cy={mark.cy} high={mark.row.price === 1} holdout={mark.row.split === "holdout"} />
            {mark.wrong && (
              <text className="sll-wrong" x={mark.cx + 9} y={mark.cy - 5}>
                ✗
              </text>
            )}
          </g>
        ))}
        {chosen && (
          <g>
            <circle className="sll-chosen" cx={chosen.cx} cy={chosen.cy} r={11.5} />
            <text
              className="sll-chosen-name"
              x={chosen.cx > R - 150 ? chosen.cx - 17 : chosen.cx + (chosen.wrong ? 27 : 17)}
              y={chosen.cy + 4}
              textAnchor={chosen.cx > R - 150 ? "end" : "start"}
            >
              {current.name} · {current.price ? "high" : "low"} price
            </text>
          </g>
        )}
      </svg>
      <ul className="sll-key" aria-hidden="true">
        <li>
          <KeyMark high holdout={false} />
          train, high price
        </li>
        <li>
          <KeyMark high={false} holdout={false} />
          train, low price
        </li>
        <li>
          <KeyMark high holdout />
          holdout, high
        </li>
        <li>
          <KeyMark high={false} holdout />
          holdout, low
        </li>
        <li>
          <b className="sll-key__wrong">✗</b>
          the rule gets this row wrong
        </li>
        {sliced && <li>faded: a row of another color than this slice ({current.color})</li>}
      </ul>
    </div>
  );
}

/**
 * The selected row's score built term by term on one z axis, then dropped onto the curve that turns the score
 * into the model's output. The picture is the same arithmetic as the formula under it.
 */
export function ScoreWalk({
  names,
  weights,
  x,
  bias,
  link,
  output,
  target,
  rowName,
}: {
  names: readonly string[];
  weights: readonly number[];
  x: readonly number[];
  bias: number;
  link: "sigmoid" | "identity";
  output: number;
  target: number;
  rowName: string;
}) {
  const [wrapper, measured] = useWidth(660);
  const W = Math.max(430, measured);
  const A0 = 268;
  const A1 = W - 18;
  const rowH = 22;
  const top = 30;
  const terms = [
    ...names.map((name, index) => {
      const weight = weights[index] ?? 0;
      const value = x[index] ?? 0;
      return {
        key: name,
        name: `w_${name} × ${name}`,
        detail: `${fmt(weight)} × ${plain(value)} = ${signed(weight * value, 2)}`,
        value: weight * value,
      };
    }),
    { key: "bias", name: "b", detail: signed(bias), value: bias },
  ];
  let running = 0;
  const steps = terms.map((term) => {
    const from = running;
    running += term.value;
    return { ...term, from, to: running };
  });
  const z = running;
  const reach = Math.max(3.2, ...steps.map((step) => Math.abs(step.to) * 1.14));
  const xs = (value: number) => A0 + ((value + reach) / (2 * reach)) * (A1 - A0);
  const tickStep = reach <= 4.5 ? 1 : reach <= 9 ? 2 : Math.ceil(reach / 4);
  const ticks: number[] = [];
  for (let value = 0; value <= reach; value += tickStep) {
    ticks.push(value);
    if (value > 0) ticks.push(-value);
  }
  const zRow = top + steps.length * rowH;
  const plotTop = zRow + rowH + 14;
  const plotBottom = plotTop + 92;
  const yLow = link === "identity" ? -0.25 : 0;
  const yHigh = link === "identity" ? 1.25 : 1;
  const ys = (value: number) => plotBottom - ((clamp(value, yLow, yHigh) - yLow) / (yHigh - yLow)) * (plotBottom - plotTop);
  const f = (value: number) => (link === "sigmoid" ? sigmoid(value) : value);
  const curve = Array.from({ length: 81 }, (_, index) => {
    const value = -reach + (2 * reach * index) / 80;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(1)} ${ys(f(value)).toFixed(1)}`;
  }).join(" ");
  const H = plotBottom + 26;
  const predictedHigh = output >= 0.5;
  const right = (predictedHigh ? 1 : 0) === target;
  const outName = link === "sigmoid" ? "P(y=1|x)" : "ŷ";

  return (
    <div className="sll-walk-wrap" ref={wrapper}>
    <svg
      className="sll-walk"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`${rowName}: ${steps
        .map((step) => `${step.name} adds ${signed(step.value)}`)
        .join(", ")}, so the score z is ${fmt(z)}. ${
        link === "sigmoid" ? `The sigmoid turns it into a chance of ${fmt(output)}.` : `The output ŷ equals the score, ${fmt(output)}.`
      } The label is ${target}, and the rule predicts ${predictedHigh ? "high" : "low"} price, which is ${right ? "right" : "wrong"}.`}
    >
      <text className="sll-walk__axis" x={A1} y={12} textAnchor="end">
        score z = w·x + b →
      </text>
      {ticks.map((tick) => (
        <g key={tick}>
          <line className={tick === 0 ? "sll-walk__zero" : "sll-walk__grid"} x1={xs(tick)} x2={xs(tick)} y1={top - 6} y2={plotBottom} />
          <text className="sll-walk__tick" x={xs(tick)} y={plotBottom + 14} textAnchor="middle">
            {tick}
          </text>
        </g>
      ))}
      {steps.map((step, index) => {
        const y = top + index * rowH;
        const x0 = xs(step.from);
        const x1 = xs(step.to);
        const up = step.value >= 0;
        const flat = Math.abs(x1 - x0) < 1.5;
        return (
          <g key={step.key}>
            <text className="sll-walk__name" x={6} y={y + 12}>
              {step.name}
            </text>
            <text className="sll-walk__detail" x={A0 - 14} y={y + 12} textAnchor="end">
              {step.detail}
            </text>
            {flat ? (
              <circle className="sll-walk__still" cx={x1} cy={y + 8} r={3} />
            ) : (
              <>
                <rect
                  className={`sll-walk__bar ${up ? "is-up" : "is-down"}`}
                  x={Math.min(x0, x1)}
                  y={y + 2}
                  width={Math.abs(x1 - x0)}
                  height={12}
                  rx={2}
                />
                <path
                  className={`sll-walk__head ${up ? "is-up" : "is-down"}`}
                  d={up ? `M${x1} ${y + 8} l-7 -8 v16 z` : `M${x1} ${y + 8} l7 -8 v16 z`}
                />
              </>
            )}
            <line className="sll-walk__carry" x1={x1} x2={x1} y1={y + 14} y2={y + rowH + 2} />
          </g>
        );
      })}
      <text className="sll-walk__name is-total" x={6} y={zRow + 12}>
        score z
      </text>
      <text className="sll-walk__detail is-total" x={A0 - 14} y={zRow + 12} textAnchor="end">
        = {fmt(z)}
      </text>
      <line className="sll-walk__drop" x1={xs(z)} x2={xs(z)} y1={zRow + 8} y2={ys(f(z))} />
      <circle className="sll-walk__z" cx={xs(z)} cy={zRow + 8} r={5} />

      <rect className="sll-walk__high" x={A0} y={ys(yHigh)} width={A1 - A0} height={ys(0.5) - ys(yHigh)} />
      <line className="sll-walk__half" x1={A0} x2={A1} y1={ys(0.5)} y2={ys(0.5)} />
      {[0, 0.5, 1].map((tick) => (
        <text key={tick} className="sll-walk__tick" x={A0 - 6} y={ys(tick) + 3.5} textAnchor="end">
          {tick}
        </text>
      ))}
      {A1 - A0 >= 280 && (
        <>
          <text className="sll-walk__note" x={A1 - 4} y={ys(0.5) - 6} textAnchor="end">
            above 0.5: predicts high
          </text>
          <text className="sll-walk__note" x={A1 - 4} y={ys(0.5) + 14} textAnchor="end">
            below 0.5: predicts low
          </text>
        </>
      )}
      <line className="sll-walk__label" x1={A0} x2={A1} y1={ys(target)} y2={ys(target)} />
      <path className="sll-walk__curve" d={curve} />
      <line className="sll-walk__miss" x1={xs(z)} x2={xs(z)} y1={ys(output)} y2={ys(target)} />
      <line className="sll-walk__drop" x1={A0} x2={xs(z)} y1={ys(output)} y2={ys(output)} />
      <circle className="sll-walk__out" cx={xs(z)} cy={ys(output)} r={5} />
      <text className="sll-walk__name" x={6} y={plotTop + 12}>
        {link === "sigmoid" ? "σ(z) squashes z" : "ŷ = z, no squashing"}
      </text>
      <text className="sll-walk__name is-total" x={6} y={plotTop + 30} textAnchor="start">
        {outName} = {fmt(output)}
      </text>
      <text className="sll-walk__detail" x={6} y={plotTop + 48} textAnchor="start">
        label y = {target} (dashed line)
      </text>
      <text className={`sll-walk__verdict ${right ? "is-right" : "is-wrong"}`} x={6} y={plotTop + 68}>
        {right ? "✓" : "✗"} predicts {predictedHigh ? "high" : "low"} price
      </text>
    </svg>
    </div>
  );
}

/** The questions a tree asks the selected row, in order, ending in the leaf whose train fraction is p. */
export function TreePath({
  tree,
  names,
  x,
  output,
  target,
}: {
  tree: TreeNode | null;
  names: readonly string[];
  x: readonly number[];
  output: number;
  target: number;
}) {
  const path: Array<{ name: string; value: number; threshold: number; left: boolean }> = [];
  let node = tree;
  while (node && node.kind === "split") {
    const value = x[node.axis] ?? 0;
    const left = value < node.threshold;
    path.push({ name: names[node.axis] ?? "x", value, threshold: node.threshold, left });
    node = left ? node.left : node.right;
  }
  const predictedHigh = output >= 0.5;
  const right = (predictedHigh ? 1 : 0) === target;
  return (
    <div className="sll-path-wrap">
      <ol className="sll-path" aria-label="The tree's questions for this row, in order">
        {path.map((step, index) => (
          <li key={index}>
            <small>question {index + 1}</small>
            <strong>
              {step.name} &lt; {plain(step.threshold)}?
            </strong>
            <span>
              {step.name} is {plain(step.value)}: {step.left ? "yes" : "no"}
            </span>
          </li>
        ))}
        <li className={`is-leaf ${right ? "is-right" : "is-wrong"}`}>
          <small>{path.length === 0 ? "no question asked" : "leaf"}</small>
          <strong>p = {fmt(output)}</strong>
          <span>
            {right ? "✓" : "✗"} predicts {predictedHigh ? "high" : "low"}, label y = {target}
          </span>
        </li>
      </ol>
      <ProbTrack output={output} target={target} />
    </div>
  );
}

/** A 0-to-1 track: where the model's output sits, which side of 0.5, and how far from the label. */
export function ProbTrack({ output, target }: { output: number; target: number }) {
  const [wrapper, measured] = useWidth(660);
  const W = Math.max(320, measured);
  const L = 36;
  const R = W - 36;
  const xs = (value: number) => L + clamp(value, 0, 1) * (R - L);
  const y = 30;
  return (
    <div className="sll-track-wrap" ref={wrapper}>
    <svg
      className="sll-track"
      viewBox={`0 0 ${W} 62`}
      role="img"
      aria-label={`Leaf probability ${fmt(output)} on a track from 0 to 1. The label is ${target}.`}
    >
      <rect className="sll-track__low" x={L} y={y - 7} width={(R - L) / 2} height={14} />
      <rect className="sll-track__high" x={(L + R) / 2} y={y - 7} width={(R - L) / 2} height={14} />
      <line className="sll-walk__half" x1={(L + R) / 2} x2={(L + R) / 2} y1={y - 13} y2={y + 13} />
      <line className="sll-walk__miss" x1={xs(output)} x2={xs(target)} y1={y} y2={y} />
      <path className="sll-track__label" d={`M${xs(target)} ${y - 9} l6 -9 h-12 z`} />
      <circle className="sll-walk__out" cx={xs(output)} cy={y} r={6} />
      <text className="sll-walk__note" x={L} y={y + 26}>
        0 · predicts low
      </text>
      <text className="sll-walk__note" x={(L + R) / 2} y={y + 26} textAnchor="middle">
        0.5
      </text>
      <text className="sll-walk__note" x={R} y={y + 26} textAnchor="end">
        predicts high · 1
      </text>
      <text className="sll-walk__note" x={xs(target)} y={9} textAnchor={target === 1 ? "end" : "start"}>
        label y = {target}
      </text>
    </svg>
    </div>
  );
}

/** The sixteen apartments as one rail, cut where the split is: twelve the fit may read, four it may not. */
export function SplitRail({
  apartments,
  selected,
  onSelect,
}: {
  apartments: readonly Apartment[];
  selected: number;
  onSelect?: (id: number) => void;
}) {
  const groups = [
    { id: "train", title: "Train · 12", note: "enter X and y", rows: apartments.filter((row) => row.split === "train") },
    { id: "holdout", title: "Holdout · 4", note: "scored after the fit", rows: apartments.filter((row) => row.split === "holdout") },
  ];
  return (
    <div className="sll-rail" aria-hidden={onSelect ? undefined : true}>
      <div className="sll-rail__groups">
      {groups.map((group) => (
        <div key={group.id} className={`sll-rail__group is-${group.id}`} style={{ flexGrow: group.rows.length }}>
          <div className="sll-rail__cells">
            {group.rows.map((row) => {
              const className = `sll-rail__cell${row.price ? " is-high" : ""}${row.id === selected ? " is-selected" : ""}`;
              const initial = row.name.charAt(0);
              return onSelect ? (
                <button
                  type="button"
                  key={row.id}
                  className={className}
                  title={row.name}
                  aria-label={`${row.name}, ${row.split}, ${row.price ? "high" : "low"} price`}
                  aria-pressed={row.id === selected}
                  onClick={() => onSelect(row.id)}
                >
                  {initial}
                </button>
              ) : (
                <i key={row.id} className={className} title={row.name}>
                  {initial}
                </i>
              );
            })}
          </div>
          <p>
            <strong>{group.title}</strong> {group.note}
          </p>
        </div>
      ))}
      </div>
      <p className="sll-rail__key">
        One cell per apartment, marked with the first letter of its name.
        <i className="sll-rail__cell is-high" />
        Outline · high price
      </p>
    </div>
  );
}

/** Loss values as bars on one scale, with reference lines that cross every bar. */
export function LossBars({
  label,
  rows,
  refs = [],
  unit,
  gap = false,
}: {
  /** Bracket the distance between the first two bars and print it as the gap. */
  gap?: boolean;
  label: string;
  rows: ReadonlyArray<{ id: string; name: string; value: number; tone: "train" | "holdout"; current?: boolean }>;
  refs?: ReadonlyArray<{ id: string; name: string; value: number }>;
  unit: string;
}) {
  const W = 640;
  const L = 168;
  const R = 566;
  const rowH = 26;
  const top = 6 + refs.length * 13;
  const max = Math.max(1e-6, ...rows.map((row) => row.value), ...refs.map((ref) => ref.value)) * 1.05;
  const xs = (value: number) => L + (clamp(value, 0, max) / max) * (R - L);
  // The lowest reference line is named on its left and the others on their right, so the names do not collide.
  const lowestRef = refs.length > 1 ? Math.min(...refs.map((ref) => ref.value)) : Number.NaN;
  const leftOf = (ref: { value: number }) => ref.value === lowestRef || xs(ref.value) > R - 150;
  const barsBottom = top + rows.length * rowH;
  const showGap = gap && rows.length >= 2;
  const bottom = barsBottom + (showGap ? 20 : 0);
  const gapFrom = showGap ? xs(Math.min(rows[0].value, rows[1].value)) : 0;
  const gapTo = showGap ? xs(Math.max(rows[0].value, rows[1].value)) : 0;
  return (
    <svg className="sll-bars" viewBox={`0 0 ${W} ${bottom + 18}`} role="img" aria-label={label}>
      {showGap && (
        <g>
          <path className="sll-bars__gap" d={`M${gapFrom} ${barsBottom + 3} v8 H${gapTo} v-8`} />
          <text className="sll-bars__value" x={gapTo + 6} y={barsBottom + 15}>
            gap {signed(rows[1].value - rows[0].value)}
          </text>
        </g>
      )}
      <line className="sll-bars__axis" x1={L} x2={L} y1={top - 2} y2={bottom} />
      {rows.map((row, index) => {
        const y = top + index * rowH;
        return (
          <g key={row.id} className={row.current ? "is-current" : undefined}>
            <text className="sll-bars__name" x={L - 8} y={y + 16} textAnchor="end">
              {row.name}
            </text>
            <rect className={`sll-bars__bar is-${row.tone}`} x={L} y={y + 5} width={Math.max(1.5, xs(row.value) - L)} height={15} rx={2} />
            <text className="sll-bars__value" x={xs(row.value) + 6} y={y + 16}>
              {fmt(row.value)}
            </text>
          </g>
        );
      })}
      {refs.map((ref, index) => (
        <g key={ref.id}>
          <line className="sll-bars__ref" x1={xs(ref.value)} x2={xs(ref.value)} y1={top - 13 * (index + 1) + 2} y2={bottom + 2} />
          <text
            className="sll-bars__refname"
            x={xs(ref.value) + (leftOf(ref) ? -5 : 5)}
            y={top - 13 * index - 4}
            textAnchor={leftOf(ref) ? "end" : "start"}
          >
            {ref.name} {fmt(ref.value, 2)}
          </text>
        </g>
      ))}
      <text className="sll-bars__unit" x={L} y={bottom + 14}>
        0 · {unit}, lower is better
      </text>
    </svg>
  );
}

/** How one table row becomes the vector x under the chosen encoding. */
export function EncodeDiagram({
  row,
  names,
  x,
}: {
  row: Apartment;
  names: readonly string[];
  x: readonly number[];
}) {
  const W = 640;
  const facts = [
    { id: "rooms", name: "rooms", value: String(row.rooms), cx: 290 },
    { id: "color", name: "color", value: row.color, cx: 420 },
    { id: "park", name: "park", value: row.park ? "yes" : "no", cx: 550 },
  ];
  const slotW = names.length > 3 ? 58 : 70;
  const gap = 8;
  const total = names.length * slotW + (names.length - 1) * gap;
  const start = 420 - total / 2;
  const slots = names.map((name, index) => ({
    name,
    value: x[index] ?? 0,
    cx: start + slotW / 2 + index * (slotW + gap),
    from: name === "rooms" ? "rooms" : name === "park" ? "park" : "color",
  }));
  const colorUsed = slots.some((slot) => slot.from === "color");
  const factBottom = 42;
  const slotTop = 92;
  return (
    <svg
      className="sll-encode"
      viewBox={`0 0 ${W} 150`}
      role="img"
      aria-label={`${row.name}: rooms ${row.rooms}, color ${row.color}, park ${row.park ? "yes" : "no"} becomes x = [${x
        .map(plain)
        .join(", ")}], with slots ${names.join(", ")}.${colorUsed ? "" : " Color is dropped."}`}
    >
      <text className="sll-encode__side" x={6} y={22}>
        table row
      </text>
      <text className="sll-encode__side is-name" x={6} y={38}>
        {row.name}
      </text>
      <text className="sll-encode__side" x={6} y={slotTop + 14}>
        vector x
      </text>
      <text className="sll-encode__side is-name" x={6} y={slotTop + 30}>
        {names.length} numbers
      </text>
      {slots.map((slot) => {
        const fact = facts.find((item) => item.id === slot.from)!;
        const live = slot.from !== "color" || slot.name === "color" || slot.value !== 0;
        return (
          <path
            key={`link-${slot.name}`}
            className={`sll-encode__link${live ? "" : " is-zero"}`}
            d={`M${fact.cx} ${factBottom} C ${fact.cx} ${factBottom + 26}, ${slot.cx} ${slotTop - 26}, ${slot.cx} ${slotTop - 2}`}
          />
        );
      })}
      {!colorUsed && (
        <>
          <line className="sll-encode__link is-dropped" x1={420} x2={420} y1={factBottom} y2={factBottom + 16} />
          <text className="sll-encode__dropped" x={420} y={factBottom + 30} textAnchor="middle">
            ✗ dropped
          </text>
        </>
      )}
      {facts.map((fact) => (
        <g key={fact.id} className={fact.id === "color" && !colorUsed ? "is-unused" : undefined}>
          <rect className="sll-encode__fact" x={fact.cx - 50} y={8} width={100} height={factBottom - 8} rx={6} />
          <text className="sll-encode__factname" x={fact.cx - 40} y={30}>
            {fact.name}
          </text>
          <text className="sll-encode__factvalue" x={fact.cx + 40} y={30} textAnchor="end">
            {fact.value}
          </text>
        </g>
      ))}
      {slots.map((slot) => (
        <g key={`slot-${slot.name}`}>
          <rect className="sll-encode__slot" x={slot.cx - slotW / 2} y={slotTop} width={slotW} height={30} rx={5} />
          <text className="sll-encode__slotvalue" x={slot.cx} y={slotTop + 20} textAnchor="middle">
            {plain(slot.value)}
          </text>
          <text className="sll-encode__slotname" x={slot.cx} y={slotTop + 46} textAnchor="middle">
            {slot.name}
          </text>
        </g>
      ))}
    </svg>
  );
}
