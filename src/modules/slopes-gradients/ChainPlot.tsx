import { useId } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { chainA, chainB, chainReading } from "./curves";
import { curvePath, fmt, linear, sliderKeys, snap, useDrag } from "./kit";

/**
 * Card 4. Two machines in a row, each drawn as its own curve with the point it is working at and a
 * stub of its slope there. The middle reading u is the height of the first point and the position of
 * the second.
 */
export function ChainPlot({ w, onDial }: { w: number; onDial: (w: number) => void }) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(664);
  const width = Math.max(280, measured);
  const sideBySide = width >= 520;
  const gap = 34;
  const plotWidth = sideBySide ? (width - gap) / 2 : width;
  const plotHeight = 250;
  const height = sideBySide ? plotHeight : plotHeight * 2 + 14;
  const reading = chainReading(w);

  const frame = (offsetX: number, offsetY: number) => ({
    left: offsetX + 40,
    right: offsetX + plotWidth - 10,
    top: offsetY + 34,
    bottom: offsetY + plotHeight - 44,
  });
  const a = frame(0, 0);
  const b = frame(sideBySide ? plotWidth + gap : 0, sideBySide ? 0 : plotHeight + 14);
  const ax = linear(0, 6, a.left, a.right);
  const ay = linear(0, 21, a.bottom, a.top);
  const bx = linear(0, 21, b.left, b.right);
  const by = linear(0, 10, b.bottom, b.top);

  /** A stub of a tangent with a fixed on-screen length. */
  const stub = (
    px: (value: number) => number,
    py: (value: number) => number,
    perX: number,
    perY: number,
    atX: number,
    atY: number,
    slope: number,
  ) => {
    const reach = 44 / Math.hypot(perX, slope * perY);
    return { x1: px(atX - reach), y1: py(atY - slope * reach), x2: px(atX + reach), y2: py(atY + slope * reach) };
  };
  const stubA = stub(ax, ay, (a.right - a.left) / 6, (a.bottom - a.top) / 21, w, reading.u, reading.slopeA);
  const stubB = stub(bx, by, (b.right - b.left) / 21, (b.bottom - b.top) / 10, reading.u, reading.meter, reading.slopeB);

  const drag = useDrag(width, (px) => onDial(snap(((px - a.left) / (a.right - a.left)) * 6, 0, 6)));
  const aLabelLeft = ax(w) > a.right - 86;
  const bLabelLeft = bx(reading.u) > b.right - 86;

  return (
    <div className="sg-frame" ref={frameRef}>
      <svg
        className="sg-plot sg-plot--drag-x"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Two curves. Machine A turns the dial w into the middle reading u: at w = ${fmt(w)} it gives u = ${fmt(
          reading.u,
        )} and its slope there is ${fmt(reading.slopeA)}. Machine B turns u into the meter L, an S-shaped curve that is flat at both ends: at u = ${fmt(
          reading.u,
        )} it gives L = ${fmt(reading.meter)} and its slope there is ${fmt(reading.slopeB, 3)}.`}
      >
        <defs>
          <clipPath id={`${id}-a`}>
            <rect x={a.left} y={a.top} width={a.right - a.left} height={a.bottom - a.top} />
          </clipPath>
          <clipPath id={`${id}-b`}>
            <rect x={b.left} y={b.top} width={b.right - b.left} height={b.bottom - b.top} />
          </clipPath>
        </defs>

        {/* Machine A */}
        <text className="sg-plot-title" x={a.left - 30} y={a.top - 18}>
          Machine A · dial w in, middle reading u out
        </text>
        {[0, 5, 10, 15, 20].map((tick) => (
          <g key={`au-${tick}`}>
            <line className="sg-grid" x1={a.left} x2={a.right} y1={ay(tick)} y2={ay(tick)} />
            <text className="sg-tick" x={a.left - 6} y={ay(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
          <text key={`aw-${tick}`} className="sg-tick" x={ax(tick)} y={a.bottom + 20} textAnchor="middle">
            {tick}
          </text>
        ))}
        <line className="sg-axis" x1={a.left} x2={a.right} y1={a.bottom} y2={a.bottom} />
        <line className="sg-axis" x1={a.left} x2={a.left} y1={a.top} y2={a.bottom} />
        <text className="sg-axis-label" x={a.left} y={a.bottom + 36}>
          dial w →
        </text>
        <text className="sg-axis-label is-mid" x={a.left + 6} y={a.top + 12}>
          u ↑
        </text>
        <g clipPath={`url(#${id}-a)`}>
          <path className="sg-curve" d={curvePath(0, 6, 60, ax, ay, chainA)} />
          <line className="sg-leader" x1={ax(w)} y1={ay(reading.u)} x2={ax(w)} y2={a.bottom} />
          <line className="sg-leader sg-leader--mid" x1={a.left} y1={ay(reading.u)} x2={ax(w)} y2={ay(reading.u)} />
          <line className="sg-line sg-line--slope sg-line--thick" {...stubA} />
        </g>
        <line className="sg-line sg-line--mid sg-line--tick" x1={a.left - 5} x2={a.left + 5} y1={ay(reading.u)} y2={ay(reading.u)} />
        <circle className="sg-handle" cx={ax(w)} cy={ay(reading.u)} r="6.5" />
        <text className="sg-mark is-slope" x={ax(w) + (aLabelLeft ? -12 : 12)} y={ay(reading.u) + 18} textAnchor={aLabelLeft ? "end" : "start"}>
          slope {fmt(reading.slopeA)}
        </text>
        <text className="sg-mark is-mid" x={a.left + 8} y={ay(reading.u) - 6}>
          u = {fmt(reading.u)}
        </text>
        <g
          className="sg-knob"
          tabIndex={0}
          role="slider"
          aria-label="Dial w, on the plot"
          aria-valuemin={0}
          aria-valuemax={6}
          aria-valuenow={w}
          aria-valuetext={`w = ${fmt(w)}, u = ${fmt(reading.u)}, meter ${fmt(reading.meter)}`}
          onKeyDown={sliderKeys(w, 0, 6, 0.05, onDial)}
        >
          <circle cx={ax(w)} cy={a.bottom} r="6.5" />
        </g>
        <text className="sg-mark is-accent" x={Math.max(a.left + 92, ax(w))} y={a.bottom + 36} textAnchor="middle">
          w = {fmt(w)}
        </text>
        <rect className="sg-hit" x={a.left} y={a.top} width={a.right - a.left} height={a.bottom - a.top + 14} {...drag} />

        {/* Machine B */}
        <text className="sg-plot-title" x={b.left - 30} y={b.top - 18}>
          Machine B · middle reading u in, meter L out
        </text>
        {[0, 2, 4, 6, 8, 10].map((tick) => (
          <g key={`bl-${tick}`}>
            <line className="sg-grid" x1={b.left} x2={b.right} y1={by(tick)} y2={by(tick)} />
            <text className="sg-tick" x={b.left - 6} y={by(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {[0, 5, 10, 15, 20].map((tick) => (
          <text key={`bu-${tick}`} className="sg-tick" x={bx(tick)} y={b.bottom + 20} textAnchor="middle">
            {tick}
          </text>
        ))}
        <line className="sg-axis" x1={b.left} x2={b.right} y1={b.bottom} y2={b.bottom} />
        <line className="sg-axis" x1={b.left} x2={b.left} y1={b.top} y2={b.bottom} />
        <text className="sg-axis-label is-mid" x={b.left} y={b.bottom + 36}>
          u →
        </text>
        <text className="sg-axis-label" x={b.left + 6} y={b.top + 12}>
          meter L ↑
        </text>
        <g clipPath={`url(#${id}-b)`}>
          <path className="sg-curve" d={curvePath(0, 21, 90, bx, by, chainB)} />
          <line className="sg-leader sg-leader--mid" x1={bx(reading.u)} y1={by(reading.meter)} x2={bx(reading.u)} y2={b.bottom} />
          <line className="sg-line sg-line--slope sg-line--thick" {...stubB} />
        </g>
        <line className="sg-line sg-line--mid sg-line--tick" x1={bx(reading.u)} x2={bx(reading.u)} y1={b.bottom - 5} y2={b.bottom + 5} />
        <circle className="sg-dot sg-dot--meter" cx={bx(reading.u)} cy={by(reading.meter)} r="5.5" />
        <text
          className="sg-mark is-slope"
          x={bx(reading.u) + (bLabelLeft ? -11 : 11)}
          y={by(reading.meter) + (reading.meter > 8 ? 30 : -10)}
          textAnchor={bLabelLeft ? "end" : "start"}
        >
          slope {fmt(reading.slopeB, reading.slopeB < 0.1 ? 3 : 2)}
        </text>
        <text className="sg-mark is-mid" x={Math.min(b.right - 40, Math.max(b.left + 78, bx(reading.u)))} y={b.bottom + 36} textAnchor="middle">
          u = {fmt(reading.u)}
        </text>
      </svg>
    </div>
  );
}
