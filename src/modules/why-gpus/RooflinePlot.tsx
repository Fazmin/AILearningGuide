import { useEffect, useRef, useState } from "react";
import { attainable, BATCH_TRAIL, formatIntensity, workload, type OperationId } from "./roofline";

const LEFT = 60;
const TOP = 26;
const BOTTOM = 236;
const HEIGHT = 278;
const X_DECADES: [number, number] = [-2, 4];
const Y_DECADES: [number, number] = [-3, 3];

/** Track the rendered width so the SVG is drawn at 1:1 and its labels keep their pixel size. */
function useMeasuredWidth(fallback: number) {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

const yAt = (tflops: number) => {
  const value = Math.log10(Math.max(tflops, 10 ** Y_DECADES[0]));
  const clamped = Math.min(Y_DECADES[1], value);
  return BOTTOM - ((clamped - Y_DECADES[0]) / (Y_DECADES[1] - Y_DECADES[0])) * (BOTTOM - TOP);
};

const tickLabel = (exponent: number) => {
  const value = 10 ** exponent;
  if (value >= 1000) return `${value / 1000}k`;
  return value >= 1 ? `${value}` : `${value}`;
};

export function RooflinePlot({
  peakTflops,
  bandwidth,
  bytesPerElement,
  operation,
  batch,
  width,
  precisionLabel,
  unit,
}: {
  peakTflops: number;
  /** GB/s */
  bandwidth: number;
  bytesPerElement: number;
  operation: OperationId;
  batch: number;
  width: number;
  precisionLabel: string;
  unit: string;
}) {
  const [figureRef, measured] = useMeasuredWidth(600);
  const viewWidth = Math.max(320, measured || 600);
  const RIGHT = viewWidth - 14;
  const xAt = (intensity: number) => {
    const value = Math.log10(Math.max(intensity, 10 ** X_DECADES[0]));
    const clamped = Math.min(X_DECADES[1], value);
    return LEFT + ((clamped - X_DECADES[0]) / (X_DECADES[1] - X_DECADES[0])) * (RIGHT - LEFT);
  };
  const bandwidthTb = bandwidth / 1000;
  const ridge = peakTflops / bandwidthTb;
  const roofAt = (intensity: number) => attainable(peakTflops, bandwidthTb, intensity);
  const xMin = 10 ** X_DECADES[0];
  const xMax = 10 ** X_DECADES[1];
  const startIntensity = Math.max(xMin, 10 ** Y_DECADES[0] / bandwidthTb);
  const roofPath = `M ${xAt(startIntensity)} ${yAt(roofAt(startIntensity))} L ${xAt(ridge)} ${yAt(peakTflops)} L ${xAt(xMax)} ${yAt(peakTflops)}`;

  const current = workload(operation, batch, width, bytesPerElement);
  const other = workload(operation === "matmul" ? "add" : "matmul", batch, width, bytesPerElement);
  const trail = BATCH_TRAIL.map((rows) => {
    const intensity = workload("matmul", rows, width, bytesPerElement).intensity;
    return { rows, intensity, x: xAt(intensity), y: yAt(roofAt(intensity)) };
  });
  const currentPoint = { x: xAt(current.intensity), y: yAt(roofAt(current.intensity)) };
  const otherPoint = { x: xAt(other.intensity), y: yAt(roofAt(other.intensity)) };
  const ridgeX = xAt(ridge);
  const bound = current.intensity < ridge ? "memory-bound" : "compute-bound";
  const currentLabel = operation === "matmul" ? `matmul, ${batch} row${batch === 1 ? "" : "s"}` : "add";
  const peakY = yAt(peakTflops);
  const ridgeLabelEnd = ridgeX > RIGHT - 90;

  const xTicks = Array.from({ length: X_DECADES[1] - X_DECADES[0] + 1 }, (_, index) => X_DECADES[0] + index);
  const yTicks = Array.from({ length: Y_DECADES[1] - Y_DECADES[0] + 1 }, (_, index) => Y_DECADES[0] + index);

  return (
    <figure className="wg-roofline" ref={figureRef}>
      <svg
        viewBox={`0 0 ${viewWidth} ${HEIGHT}`}
        role="img"
        aria-label={`Roofline on log–log axes for ${precisionLabel} at ${bandwidth.toFixed(0)} GB/s. The slanted roof is bandwidth times intensity; the flat roof is the ${peakTflops} ${unit} peak; they meet at the ridge, ${formatIntensity(ridge)} operations per byte. The current ${currentLabel} sits at ${formatIntensity(current.intensity)} operations per byte and attains ${roofAt(current.intensity).toPrecision(3)} ${unit}, which is ${bound}.`}
      >
        <rect className="wg-roofline__memory-zone" x={LEFT} y={TOP} width={Math.max(0, Math.min(RIGHT, ridgeX) - LEFT)} height={BOTTOM - TOP} />
        <rect className="wg-roofline__compute-zone" x={Math.min(RIGHT, ridgeX)} y={TOP} width={Math.max(0, RIGHT - ridgeX)} height={BOTTOM - TOP} />
        {xTicks.map((exponent) => (
          <g key={`x${exponent}`}>
            <line className="wg-roofline__grid" x1={xAt(10 ** exponent)} x2={xAt(10 ** exponent)} y1={TOP} y2={BOTTOM} />
            <text className="wg-roofline__tick" x={xAt(10 ** exponent)} y={BOTTOM + 14} textAnchor="middle">
              {tickLabel(exponent)}
            </text>
          </g>
        ))}
        {yTicks.map((exponent) => (
          <g key={`y${exponent}`}>
            <line className="wg-roofline__grid" x1={LEFT} x2={RIGHT} y1={yAt(10 ** exponent)} y2={yAt(10 ** exponent)} />
            <text className="wg-roofline__tick" x={LEFT - 6} y={yAt(10 ** exponent) + 3} textAnchor="end">
              {tickLabel(exponent)}
            </text>
          </g>
        ))}
        <line className="wg-roofline__axis" x1={LEFT} x2={RIGHT} y1={BOTTOM} y2={BOTTOM} />
        <line className="wg-roofline__axis" x1={LEFT} x2={LEFT} y1={TOP} y2={BOTTOM} />
        {ridgeX > LEFT + 90 && (
          <text className="wg-roofline__zone-label" x={LEFT + 8} y={peakY + 18}>
            memory-bound
          </text>
        )}
        {ridgeX < RIGHT - 90 && (
          <text className="wg-roofline__zone-label" x={RIGHT - 8} y={BOTTOM - 8} textAnchor="end">
            compute-bound
          </text>
        )}
        <line className="wg-roofline__ridge" x1={ridgeX} x2={ridgeX} y1={peakY} y2={BOTTOM} />
        <line className="wg-roofline__peak-guide" x1={LEFT} x2={ridgeX} y1={peakY} y2={peakY} />
        <text className="wg-roofline__roof-label" x={LEFT + 6} y={peakY - 6}>
          peak {peakTflops} {unit}
        </text>
        <text
          className="wg-roofline__ridge-label"
          x={ridgeX + (ridgeLabelEnd ? -6 : 6)}
          y={peakY - 6}
          textAnchor={ridgeLabelEnd ? "end" : "start"}
        >
          ridge {formatIntensity(ridge)}
        </text>
        <path className="wg-roofline__roof" d={roofPath} />
        {operation === "matmul" &&
          trail.map((point, index) => {
            const onFlat = point.intensity >= ridge;
            return (
              <g key={point.rows} className="wg-roofline__trail">
                <circle cx={point.x} cy={point.y} r={3.2} />
                {point.rows !== batch && index % 2 === 0 && (
                  <text x={onFlat ? point.x : point.x + 6} y={point.y + 15} textAnchor={onFlat ? "middle" : "start"}>
                    {point.rows}
                  </text>
                )}
              </g>
            );
          })}
        <g className="wg-roofline__other">
          <rect x={otherPoint.x - 4} y={otherPoint.y - 4} width={8} height={8} transform={`rotate(45 ${otherPoint.x} ${otherPoint.y})`} />
          <text x={otherPoint.x + 8} y={otherPoint.y + 15}>
            {operation === "matmul" ? "add" : `matmul ×${batch}`}
          </text>
        </g>
        <g className={`wg-roofline__current wg-roofline__current--${bound === "memory-bound" ? "memory" : "compute"}`}>
          <circle cx={currentPoint.x} cy={currentPoint.y} r={6.5} />
        </g>
        <text className="wg-roofline__axis-label" x={(LEFT + RIGHT) / 2} y={HEIGHT - 8} textAnchor="middle">
          Arithmetic intensity, operations per byte (log scale)
        </text>
        <text className="wg-roofline__axis-label" x={14} y={(TOP + BOTTOM) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(TOP + BOTTOM) / 2})`}>
          Attainable {unit} (log scale)
        </text>
      </svg>
      <figcaption className="wg-roofline__legend">
        <span><i className="wg-key wg-key--roof" /> Roof = min(peak, bandwidth × intensity)</span>
        <span><i className="wg-key wg-key--current" /> Current operation</span>
        {operation === "matmul" && <span><i className="wg-key wg-key--trail" /> Same matmul at 1–4096 rows</span>}
        <span><i className="wg-key wg-key--other" /> {operation === "matmul" ? "Elementwise add, same shape" : "Matmul, same shape"}</span>
      </figcaption>
    </figure>
  );
}
