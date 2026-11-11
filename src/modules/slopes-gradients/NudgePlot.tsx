import { useId } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { curveSlope, curveValue, nudgeRatio } from "./curves";
import { curvePath, fmt, fmtChange, linear, signed, sliderKeys, snap, useDrag } from "./kit";

/**
 * Card 1. The bowl-shaped meter, the dial setting and its nudged neighbour, the two changes drawn as
 * the legs of a triangle, the line through both points, and a magnifier that zooms until the nudge
 * fills it, so a tiny nudge is as easy to read as a large one.
 */
export function NudgePlot({
  w,
  nudge,
  onDial,
}: {
  w: number;
  nudge: number;
  onDial: (w: number) => void;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(664);
  const width = Math.max(280, measured);
  const sideBySide = width >= 540;
  const lensSize = sideBySide ? Math.round(Math.min(250, Math.max(196, width * 0.34))) : Math.min(width - 2, 300);
  const plotWidth = sideBySide ? width - lensSize - 14 : width;
  const plotHeight = 300;
  const height = sideBySide ? plotHeight : plotHeight + lensSize + 16;

  const left = 44;
  const right = plotWidth - 10;
  const top = 26;
  const bottom = plotHeight - 46;
  const x = linear(0, 6, left, right);
  const y = linear(0, 10.5, bottom, top);
  const perW = (right - left) / 6;
  const perL = (bottom - top) / 10.5;

  const value = curveValue("bowl", w);
  const nudged = w + nudge;
  const nudgedValue = curveValue("bowl", nudged);
  const change = nudgedValue - value;
  const ratio = nudgeRatio("bowl", w, nudge);
  const slope = curveSlope("bowl", w);
  const ghostRatio = nudgeRatio("bowl", w, 1);

  /** A straight line through (w, value) with the given slope, clipped to the plot by its clip path. */
  const lineThrough = (rate: number) => ({
    x1: x(-1),
    y1: y(value + rate * (-1 - w)),
    x2: x(7),
    y2: y(value + rate * (7 - w)),
  });

  // The magnifier: the same picture, scaled up about the midpoint of the two points until the
  // nudge spans a little under half of it.
  const lensX = sideBySide ? plotWidth + 14 : (width - lensSize) / 2;
  const lensY = sideBySide ? Math.round((plotHeight - lensSize) / 2) : plotHeight + 16;
  const runPx = nudge * perW;
  const risePx = Math.abs(change) * perL;
  const zoom = Math.max(1, Math.min((0.3 * lensSize) / runPx, (0.4 * lensSize) / Math.max(risePx, 1e-6)));
  const midW = w + nudge / 2;
  const midL = (value + nudgedValue) / 2;
  // The triangle sits left of centre, which leaves room for the label on its right-hand leg.
  const lensCx = lensX + lensSize * 0.33;
  const lensCy = lensY + lensSize / 2 + 8;
  const lx = (dial: number) => lensCx + (dial - midW) * perW * zoom;
  const ly = (meter: number) => lensCy - (meter - midL) * perL * zoom;
  const lensHalfW = (lensSize * 0.7) / (perW * zoom);
  const lensSlopeLine = (rate: number) => ({
    x1: lx(midW - lensHalfW * 1.2),
    y1: ly(value + rate * (midW - lensHalfW * 1.2 - w)),
    x2: lx(midW + lensHalfW * 1.2),
    y2: ly(value + rate * (midW + lensHalfW * 1.2 - w)),
  });
  // The patch of the main plot the magnifier shows.
  const boxW = Math.max(7, lensSize / zoom);
  const boxH = Math.max(7, lensSize / zoom);
  const boxX = x(midW) - boxW * 0.33;
  const boxY = y(midL) - boxH / 2 - 8 / zoom;
  const zoomLabel = zoom >= 20 ? `×${Math.round(zoom)}` : `×${zoom.toFixed(1)}`;

  const drag = useDrag(width, (px) => onDial(snap(((px - left) / (right - left)) * 6, 0, 5)));
  // Near the right-hand edge the two labels move to the left of the triangle, where there is room.
  const crowded = x(nudged) + 7 + (fmtChange(change).length + 6) * 7 > plotWidth - 4;
  const riseLabelLeft = lx(nudged) + 8 + (fmtChange(change).length + 6) * 7 > lensX + lensSize;

  return (
    <div className="sg-frame" ref={frameRef}>
      <svg
        className="sg-plot sg-plot--drag-x"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`The meter L drawn against the dial w, a bowl with its bottom at w = 3. The dial is at ${fmt(
          w,
        )}, where the meter reads ${fmt(value)}. A nudge of ${fmt(nudge)} moves the dial to ${fmt(
          nudged,
        )} and the meter to ${fmt(nudgedValue, 3)}, a change of ${fmtChange(change)}. Change in the meter divided by change in the dial is ${fmt(
          ratio,
          3,
        )}. The slope at exactly this point is ${fmt(slope)}. A magnifier shows the two points at ${zoomLabel}.`}
      >
        <defs>
          <clipPath id={`${id}-plot`}>
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
          <clipPath id={`${id}-lens`}>
            <rect x={lensX} y={lensY} width={lensSize} height={lensSize} rx="10" />
          </clipPath>
        </defs>

        {[0, 2, 4, 6, 8, 10].map((tick) => (
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
        <text className="sg-axis-label" x={left} y={plotHeight - 8}>
          dial w →
        </text>

        <g clipPath={`url(#${id}-plot)`}>
          <path className="sg-curve" d={curvePath(0, 6, 96, x, y, (dial) => curveValue("bowl", dial))} />
          {nudge < 1 && <line className="sg-ghost" {...lineThrough(ghostRatio)} />}
          <line className="sg-line sg-line--ink is-dashed" {...lineThrough(slope)} />
          <line className="sg-line sg-line--slope" {...lineThrough(ratio)} />
          {/* Leaders from the two points to the axes they are read on. */}
          <line className="sg-leader" x1={x(w)} y1={y(value)} x2={x(w)} y2={bottom} />
          <line className="sg-leader" x1={x(w)} y1={y(value)} x2={left} y2={y(value)} />
          <line className="sg-leader" x1={x(nudged)} y1={y(nudgedValue)} x2={left} y2={y(nudgedValue)} />
          {/* The triangle: the nudge along the dial, then the change it causes in the meter. */}
          <line className="sg-line sg-line--dial" x1={x(w)} y1={y(value)} x2={x(nudged)} y2={y(value)} />
          <line className="sg-line sg-line--meter" x1={x(nudged)} y1={y(value)} x2={x(nudged)} y2={y(nudgedValue)} />
        </g>
        {/* The meter's change, repeated on the meter's own axis. */}
        <line className="sg-line sg-line--meter sg-line--thick" x1={left} x2={left} y1={y(value)} y2={y(Math.min(10.5, nudgedValue))} />
        <line className="sg-line sg-line--dial sg-line--thick" x1={x(w)} x2={x(nudged)} y1={bottom} y2={bottom} />

        <rect
          className="sg-zoom-box"
          x={boxX}
          y={boxY}
          width={boxW}
          height={boxH}
          rx="2"
        />
        {sideBySide && (
          <>
            <line className="sg-zoom-link" x1={boxX + boxW} y1={boxY} x2={lensX} y2={lensY} />
            <line className="sg-zoom-link" x1={boxX + boxW} y1={boxY + boxH} x2={lensX} y2={lensY + lensSize} />
          </>
        )}

        <circle className="sg-dot sg-dot--meter" cx={x(nudged)} cy={y(nudgedValue)} r="4" />
        <circle className="sg-handle" cx={x(w)} cy={y(value)} r="6.5" />
        {/* The dial itself: a knob on the dial's axis. */}
        <g
          className="sg-knob"
          tabIndex={0}
          role="slider"
          aria-label="Dial w, on the plot"
          aria-valuemin={0}
          aria-valuemax={5}
          aria-valuenow={w}
          aria-valuetext={`w = ${fmt(w)}, meter ${fmt(value)}`}
          onKeyDown={sliderKeys(w, 0, 5, 0.05, onDial)}
        >
          <circle cx={x(w)} cy={bottom} r="6.5" />
        </g>
        <text className="sg-mark is-accent" x={Math.max(left + 92, x(w))} y={plotHeight - 8} textAnchor="middle">
          w = {fmt(w)}
        </text>
        {nudge >= 0.25 && (
          <>
            <text
              className="sg-mark is-dial"
              x={crowded ? x(w) - 10 : x(w) + 10}
              y={y(value) + (change >= 0 ? 16 : -8)}
              textAnchor={crowded ? "end" : "start"}
            >
              nudge {signed(nudge)}
            </text>
            <text
              className="sg-mark is-meter"
              x={crowded ? x(w) - 10 : x(nudged) + 7}
              y={(y(value) + y(Math.min(10.5, nudgedValue))) / 2 + (crowded ? (change >= 0 ? -10 : 14) : 4)}
              textAnchor={crowded ? "end" : "start"}
            >
              meter {fmtChange(change)}
            </text>
          </>
        )}
        <rect className="sg-hit" x={left} y={top} width={right - left} height={bottom - top + 14} {...drag} />

        {/* Magnifier */}
        <rect className="sg-lens" x={lensX} y={lensY} width={lensSize} height={lensSize} rx="10" />
        <g clipPath={`url(#${id}-lens)`}>
          <path
            className="sg-curve"
            d={curvePath(midW - lensHalfW * 1.2, midW + lensHalfW * 1.2, 48, lx, ly, (dial) => curveValue("bowl", dial))}
          />
          <line className="sg-line sg-line--ink is-dashed" {...lensSlopeLine(slope)} />
          <line className="sg-line sg-line--slope" {...lensSlopeLine(ratio)} />
          <line className="sg-line sg-line--dial sg-line--thick" x1={lx(w)} y1={ly(value)} x2={lx(nudged)} y2={ly(value)} />
          <line className="sg-line sg-line--meter sg-line--thick" x1={lx(nudged)} y1={ly(value)} x2={lx(nudged)} y2={ly(nudgedValue)} />
          <circle className="sg-dot sg-dot--meter" cx={lx(nudged)} cy={ly(nudgedValue)} r="4.5" />
          <circle className="sg-handle" cx={lx(w)} cy={ly(value)} r="6" />
          <text className="sg-mark is-dial" x={(lx(w) + lx(nudged)) / 2} y={ly(value) + (change >= 0 ? 16 : -8)} textAnchor="middle">
            nudge {signed(nudge)}
          </text>
          <text
            className="sg-mark is-meter"
            x={riseLabelLeft ? lx(nudged) - 8 : lx(nudged) + 8}
            y={(ly(value) + ly(nudgedValue)) / 2 + 4}
            textAnchor={riseLabelLeft ? "end" : "start"}
          >
            meter {fmtChange(change)}
          </text>
        </g>
        <text className="sg-lens-title" x={lensX + 10} y={lensY + 17}>
          magnified {zoomLabel}
        </text>
      </svg>
      <ul className="sg-key" aria-hidden="true">
        <li><i className="sg-key__line is-slope" />line through both points, slope {fmt(ratio, 3)}</li>
        <li><i className="sg-key__line is-ink is-dashed" />the line it settles on, slope {fmt(slope)}</li>
        {nudge < 1 && <li><i className="sg-key__line is-ghost" />where the line was at nudge 1</li>}
      </ul>
    </div>
  );
}
