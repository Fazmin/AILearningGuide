import { useId, useMemo } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { curveSlope, curveValue, flatSpots, slopeBands, type Curve } from "./curves";
import { curvePath, fmt, linear, signed, sliderKeys, snap, useDrag } from "./kit";

const RANGES: Record<Curve, { meter: [number, number]; meterTicks: number[]; slope: [number, number]; slopeTicks: number[] }> = {
  valleys: { meter: [0, 6.2], meterTicks: [0, 2, 4, 6], slope: [-3.6, 3.2], slopeTicks: [-2, 0, 2] },
  shelf: { meter: [0, 10], meterTicks: [0, 2, 4, 6, 8, 10], slope: [-1.6, 4.6], slopeTicks: [0, 2, 4] },
  bowl: { meter: [0, 10.5], meterTicks: [0, 2, 4, 6, 8, 10], slope: [-6.5, 6.5], slopeTicks: [-4, 0, 4] },
};

const FLAT_NAMES = { bottom: "bottom", top: "top", shelf: "shelf" } as const;

/**
 * Card 2. The meter curve on top and its slope underneath, on one shared dial axis. Bands mark where
 * the meter falls and rises; a dotted line drops from every flat spot to the place the slope crosses
 * or touches zero.
 */
export function SlopePlot({ curve, w, onDial }: { curve: Curve; w: number; onDial: (w: number) => void }) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(664);
  const width = Math.max(280, measured);
  const left = 44;
  const right = width - 12;
  const topA = 30;
  const bottomA = 226;
  const topB = 262;
  const bottomB = 410;
  const height = 452;
  const range = RANGES[curve];
  const x = linear(0, 6, left, right);
  const yMeter = linear(range.meter[0], range.meter[1], bottomA, topA);
  const ySlope = linear(range.slope[0], range.slope[1], bottomB, topB);
  const spots = useMemo(() => flatSpots(curve), [curve]);
  const bands = useMemo(() => slopeBands(curve), [curve]);

  const value = curveValue(curve, w);
  const slope = curveSlope(curve, w);
  const flatHere = Math.abs(slope) < 0.03;
  // A stub of the tangent, the same on-screen length whatever its angle.
  const perW = (right - left) / 6;
  const perL = (bottomA - topA) / (range.meter[1] - range.meter[0]);
  const reach = 46 / Math.hypot(perW, slope * perL);

  const slopeArea = (direction: "rising" | "falling") =>
    bands
      .filter((band) => band.direction === direction)
      .map(
        (band) =>
          `${curvePath(band.from, band.to, 40, x, ySlope, (dial) => curveSlope(curve, dial))}L${x(band.to).toFixed(1)} ${ySlope(0).toFixed(1)}L${x(band.from).toFixed(1)} ${ySlope(0).toFixed(1)}Z`,
      )
      .join("");

  const drag = useDrag(width, (px) => onDial(snap(((px - left) / (right - left)) * 6, 0, 6)));
  const labelLeft = x(w) > right - 110;

  return (
    <div className="sg-frame" ref={frameRef}>
      <svg
        className="sg-plot sg-plot--drag-x"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Top: the meter L against the dial w. Bottom: the slope of that curve at every dial setting, on the same dial axis. ${bands
          .map((band) => `From w = ${fmt(band.from)} to ${fmt(band.to)} the meter is ${band.direction}`)
          .join(". ")}. ${spots
          .map((spot) => `A flat spot, a ${FLAT_NAMES[spot.kind]}, sits at w = ${fmt(spot.w)}`)
          .join(". ")}. The cursor is at w = ${fmt(w)}, where the meter reads ${fmt(value)} and the slope is ${signed(slope)}.`}
      >
        <defs>
          <clipPath id={`${id}-a`}>
            <rect x={left} y={topA} width={right - left} height={bottomA - topA} />
          </clipPath>
          <clipPath id={`${id}-b`}>
            <rect x={left} y={topB} width={right - left} height={bottomB - topB} />
          </clipPath>
          <pattern id={`${id}-fall`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line className="sg-hatch sg-hatch--fall" x1="0" y1="0" x2="0" y2="7" />
          </pattern>
        </defs>

        {/* Bands: where the meter falls (hatched) and where it rises (plain tint). */}
        {bands.map((band) => (
          <g key={`${band.from}-${band.direction}`}>
            {[
              [topA, bottomA],
              [topB, bottomB],
            ].map(([from, to]) => (
              <rect
                key={from}
                className={`sg-band sg-band--${band.direction}`}
                x={x(band.from)}
                y={from}
                width={x(band.to) - x(band.from)}
                height={to - from}
                fill={band.direction === "falling" ? `url(#${id}-fall)` : undefined}
              />
            ))}
            {x(band.to) - x(band.from) > 58 && (
              <text className="sg-band-label" x={(x(band.from) + x(band.to)) / 2} y={topA - 8} textAnchor="middle">
                {band.direction === "falling" ? "falls ↘" : "rises ↗"}
              </text>
            )}
          </g>
        ))}

        {range.meterTicks.map((tick) => (
          <g key={`m-${tick}`}>
            <line className="sg-grid" x1={left} x2={right} y1={yMeter(tick)} y2={yMeter(tick)} />
            <text className="sg-tick" x={left - 6} y={yMeter(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {range.slopeTicks.map((tick) => (
          <g key={`s-${tick}`}>
            {tick !== 0 && <line className="sg-grid" x1={left} x2={right} y1={ySlope(tick)} y2={ySlope(tick)} />}
            <text className="sg-tick" x={left - 6} y={ySlope(tick) + 3.5} textAnchor="end">
              {tick > 0 ? `+${tick}` : tick < 0 ? `−${-tick}` : 0}
            </text>
          </g>
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
          <text key={`x-${tick}`} className="sg-tick" x={x(tick)} y={bottomB + 20} textAnchor="middle">
            {tick}
          </text>
        ))}
        <line className="sg-axis" x1={left} x2={right} y1={bottomA} y2={bottomA} />
        <line className="sg-axis" x1={left} x2={left} y1={topA} y2={bottomA} />
        <line className="sg-axis" x1={left} x2={left} y1={topB} y2={bottomB} />
        <line className="sg-axis" x1={left} x2={right} y1={bottomB} y2={bottomB} />
        <line className="sg-zero" x1={left} x2={right} y1={ySlope(0)} y2={ySlope(0)} />
        <text className="sg-axis-label" x={left - 34} y={14}>
          meter L
        </text>
        <text className="sg-axis-label" x={left - 34} y={topB - 9}>
          slope of the meter · above the line = rising, below = falling
        </text>
        <text className="sg-axis-label" x={left} y={height - 8}>
          dial w →
        </text>

        <g clipPath={`url(#${id}-b)`}>
          <path className="sg-area sg-area--rising" d={slopeArea("rising")} />
          <path className="sg-area sg-area--falling" d={slopeArea("falling")} />
          <path className="sg-line sg-line--slope" d={curvePath(0, 6, 120, x, ySlope, (dial) => curveSlope(curve, dial))} />
        </g>
        <g clipPath={`url(#${id}-a)`}>
          <path className="sg-curve" d={curvePath(0, 6, 120, x, yMeter, (dial) => curveValue(curve, dial))} />
        </g>

        {/* Flat spots: one dotted line through both pictures, named on the curve, zero on the slope. */}
        {spots.map((spot) => (
          <g key={spot.w}>
            <line className="sg-flat" x1={x(spot.w)} x2={x(spot.w)} y1={yMeter(spot.value)} y2={ySlope(0)} />
            <circle className="sg-dot sg-dot--hollow" cx={x(spot.w)} cy={yMeter(spot.value)} r="4" />
            <circle className="sg-dot sg-dot--hollow" cx={x(spot.w)} cy={ySlope(0)} r="4" />
            <text
              className="sg-mark"
              x={x(spot.w)}
              y={yMeter(spot.value) + (spot.kind === "bottom" ? 19 : Math.abs(x(spot.w) - x(w)) < 84 ? -28 : -11)}
              textAnchor="middle"
            >
              {FLAT_NAMES[spot.kind]}
            </text>
            {Math.abs(x(spot.w) - x(w)) >= 84 && (
              <text className="sg-mark is-slope" x={x(spot.w) + 7} y={ySlope(0) - 7}>
                0
              </text>
            )}
          </g>
        ))}

        {/* The cursor: one dial setting read on both pictures. */}
        <line className="sg-cursor" x1={x(w)} x2={x(w)} y1={topA} y2={bottomB} />
        <g clipPath={`url(#${id}-a)`}>
          <line
            className="sg-line sg-line--slope sg-line--thick"
            x1={x(w - reach)}
            y1={yMeter(value - slope * reach)}
            x2={x(w + reach)}
            y2={yMeter(value + slope * reach)}
          />
        </g>
        <line className="sg-line sg-line--slope sg-line--bar" x1={x(w)} x2={x(w)} y1={ySlope(0)} y2={ySlope(slope)} />
        <circle className="sg-dot sg-dot--slope" cx={x(w)} cy={ySlope(slope)} r="4.5" />
        <circle className="sg-handle" cx={x(w)} cy={yMeter(value)} r="6.5" />
        <text
          className="sg-mark"
          x={x(w) + (labelLeft ? -11 : 11)}
          y={Math.max(topA + 12, yMeter(value) - 12)}
          textAnchor={labelLeft ? "end" : "start"}
        >
          L = {fmt(value)}
        </text>
        <text
          className="sg-mark is-slope"
          x={x(w) + (labelLeft ? -11 : 11)}
          y={Math.min(bottomB - 6, Math.max(topB + 12, ySlope(slope) + (slope >= 0 ? -9 : 17)))}
          textAnchor={labelLeft ? "end" : "start"}
        >
          slope {flatHere ? "0.00" : signed(slope)}
        </text>
        <g
          className="sg-knob"
          tabIndex={0}
          role="slider"
          aria-label="Dial w, on the plot"
          aria-valuemin={0}
          aria-valuemax={6}
          aria-valuenow={w}
          aria-valuetext={`w = ${fmt(w)}, meter ${fmt(value)}, slope ${signed(slope)}`}
          onKeyDown={sliderKeys(w, 0, 6, 0.05, onDial)}
        >
          <circle cx={x(w)} cy={bottomB} r="6.5" />
        </g>
        <text className="sg-mark is-accent" x={Math.min(right - 30, Math.max(left + 92, x(w)))} y={height - 8} textAnchor="middle">
          w = {fmt(w)}
        </text>
        <rect className="sg-hit" x={left} y={topA} width={right - left} height={bottomB - topA + 14} {...drag} />
      </svg>
    </div>
  );
}
