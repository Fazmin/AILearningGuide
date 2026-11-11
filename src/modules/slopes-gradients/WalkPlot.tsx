import { useId, useMemo } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { curveSlope, curveValue, flatSpots, stepDown, type Curve } from "./curves";
import { ArrowDefs, curvePath, fmt, linear, signed, sliderKeys, snap, useDrag } from "./kit";

const RANGES: Record<Curve, { top: number; ticks: number[] }> = {
  valleys: { top: 6.2, ticks: [0, 2, 4, 6] },
  bowl: { top: 10.5, ticks: [0, 2, 4, 6, 8, 10] },
  shelf: { top: 10, ticks: [0, 2, 4, 6, 8, 10] },
};

/**
 * Card 5. One dial again: the meter curve, where the walk started, every setting it has visited,
 * the slope under its feet, and the step it would take next.
 */
export function WalkPlot({
  curve,
  path,
  stepSize,
  onStart,
}: {
  curve: Curve;
  path: number[];
  stepSize: number;
  onStart: (w: number) => void;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(664);
  const width = Math.max(280, measured);
  const roomy = width >= 520;
  const height = 336;
  const left = 44;
  const right = width - 12;
  const top = 48;
  const bottom = height - 46;
  const range = RANGES[curve];
  const x = linear(0, 6, left, right);
  const y = linear(0, range.top, bottom, top);
  const spots = useMemo(() => flatSpots(curve), [curve]);
  const deepest = Math.min(...spots.filter((spot) => spot.kind === "bottom").map((spot) => spot.value));
  const bottoms = spots.filter((spot) => spot.kind === "bottom").length;

  const start = path[0];
  const w = path[path.length - 1];
  const value = curveValue(curve, w);
  const slope = curveSlope(curve, w);
  const next = stepDown(curve, w, stepSize);
  const perW = (right - left) / 6;
  const perL = (bottom - top) / range.top;
  const reach = 40 / Math.hypot(perW, slope * perL);
  const yOn = (dial: number) => y(Math.min(range.top, curveValue(curve, dial)));
  const hop = (from: number, to: number) => {
    const lift = Math.min(46, 14 + Math.abs(x(to) - x(from)) * 0.22);
    return `M${x(from).toFixed(1)} ${yOn(from).toFixed(1)}Q${((x(from) + x(to)) / 2).toFixed(1)} ${(Math.min(yOn(from), yOn(to)) - lift).toFixed(1)} ${x(to).toFixed(1)} ${yOn(to).toFixed(1)}`;
  };
  const moving = Math.abs(next - w) > 0.004;
  const atStop = (next === 0 || next === 6) && moving;

  const drag = useDrag(width, (px) => onStart(snap(((px - left) / (right - left)) * 6, 0, 6)));
  const labelLeft = x(w) > right - 120;

  return (
    <div className="sg-frame" ref={frameRef}>
      <svg
        className="sg-plot sg-plot--drag-x"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`The meter L against the dial w, ${
          curve === "bowl" ? "a single bowl with its bottom at w = 3" : "a curve with two valleys and a hump between them"
        }. ${spots
          .map((spot) => `A ${spot.kind} at w = ${fmt(spot.w)} reads ${fmt(spot.value)}`)
          .join(". ")}. The walk started at w = ${fmt(start)} and has taken ${path.length - 1} steps. It is now at w = ${fmt(
          w,
          3,
        )}, where the meter reads ${fmt(value, 3)} and the slope is ${signed(slope, 3)}. The next step would move the dial to ${fmt(next, 3)}.`}
      >
        <ArrowDefs id={id} />
        <defs>
          <clipPath id={`${id}-plot`}>
            <rect x={left} y={top - 4} width={right - left} height={bottom - top + 4} />
          </clipPath>
        </defs>
        {range.ticks.map((tick) => (
          <g key={`y-${tick}`}>
            <line className="sg-grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} />
            <text className="sg-tick" x={left - 6} y={y(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
          <text key={`x-${tick}`} className="sg-tick" x={x(tick)} y={bottom + 20} textAnchor="middle">
            {tick}
          </text>
        ))}
        <line className="sg-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="sg-axis" x1={left} x2={left} y1={top} y2={bottom} />
        <text className="sg-axis-label" x={left - 34} y={14}>
          meter L · higher = more wrong
        </text>
        <text className="sg-axis-label" x={left} y={height - 8}>
          dial w →
        </text>

        <g clipPath={`url(#${id}-plot)`}>
          <path className="sg-curve" d={curvePath(0, 6, 120, x, y, (dial) => curveValue(curve, dial))} />
          {path.slice(0, -1).map((from, index) => (
            <path key={`hop-${index}`} className="sg-trail" d={hop(from, path[index + 1])} />
          ))}
          <line
            className="sg-line sg-line--slope sg-line--thick"
            x1={x(w - reach)}
            y1={y(value - slope * reach)}
            x2={x(w + reach)}
            y2={y(value + slope * reach)}
          />
          {moving && <path className="sg-line sg-line--dial is-dashed" d={hop(w, next)} markerEnd={`url(#${id}-dial)`} />}
        </g>

        {/* The flat spots are named in a band above the plot, each with a dotted leader down to its place. */}
        {spots.map((spot) => {
          const name =
            spot.kind === "top"
              ? "hump"
              : bottoms === 1
                ? `bottom · L ${fmt(spot.value)}`
                : `${spot.value === deepest ? "deeper" : "shallower"} valley${roomy ? " ·" : ""} L ${fmt(spot.value)}`;
          const half = name.length * 3.6;
          return (
            <g key={spot.w}>
              <line className="sg-flat" x1={x(spot.w)} x2={x(spot.w)} y1={top - 6} y2={y(spot.value) - 5} />
              <circle className="sg-dot sg-dot--hollow" cx={x(spot.w)} cy={y(spot.value)} r="3.5" />
              <text
                className="sg-mark is-faint"
                x={Math.min(right - half, Math.max(left + half, x(spot.w)))}
                y={top - 11}
                textAnchor="middle"
              >
                {name}
              </text>
            </g>
          );
        })}

        {path.slice(1, -1).map((visited, index) => (
          <circle key={`dot-${index}`} className="sg-trail-dot" cx={x(visited)} cy={yOn(visited)} r="3" />
        ))}
        {path.length > 1 && (
          <>
            <circle className="sg-dot sg-dot--hollow" cx={x(start)} cy={yOn(start)} r="5" />
            <text className="sg-mark" x={x(start)} y={Math.max(top + 10, yOn(start) - 12)} textAnchor="middle">
              start
            </text>
          </>
        )}
        <circle className="sg-handle" cx={x(w)} cy={yOn(w)} r="6.5" />
        {path.length > 1 && (
          <text
            className="sg-mark is-accent"
            x={x(w) + (labelLeft ? -4 : 4)}
            y={Math.min(bottom - 6, yOn(w) + 24)}
            textAnchor={labelLeft ? "end" : "start"}
          >
            step {path.length - 1} · L {fmt(value)}
          </text>
        )}
        {moving && (
          <text
            className="sg-mark is-dial"
            x={Math.min(right - 60, Math.max(left + 60, (x(w) + x(next)) / 2))}
            y={Math.max(top + 10, Math.min(yOn(w), yOn(next)) - Math.min(46, 14 + Math.abs(x(next) - x(w)) * 0.22) + 2)}
            textAnchor="middle"
          >
            next step {signed(next - w)}
            {atStop ? " (dial stop)" : ""}
          </text>
        )}
        <g
          className="sg-knob"
          tabIndex={0}
          role="slider"
          aria-label="Start w, on the plot"
          aria-valuemin={0}
          aria-valuemax={6}
          aria-valuenow={start}
          aria-valuetext={`start w = ${fmt(start)}`}
          onKeyDown={sliderKeys(start, 0, 6, 0.05, onStart)}
        >
          <circle cx={x(start)} cy={bottom} r="6.5" />
        </g>
        <text className="sg-mark is-accent" x={Math.min(right - 44, Math.max(left + 104, x(start)))} y={height - 8} textAnchor="middle">
          start w = {fmt(start)}
        </text>
        <rect className="sg-hit" x={left} y={top} width={right - left} height={bottom - top + 14} {...drag} />
      </svg>
    </div>
  );
}
