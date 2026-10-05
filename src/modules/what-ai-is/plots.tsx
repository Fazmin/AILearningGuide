import { useId } from "react";
import { B_RANGE, FIT_POINTS, W_RANGE, contour, gradient, leastSquares, residuals, STEP_SIZE } from "./fit";

const scale = (d0: number, d1: number, r0: number, r1: number) => (value: number) =>
  r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

const LEVELS = [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4];

/** Data space: three points, the current line, and each residual drawn as its own square. */
export function DataPlot({ w, b }: { w: number; b: number }) {
  const id = useId().replace(/:/g, "");
  const width = 300;
  const height = 260;
  const left = 30;
  const right = width - 10;
  const top = 22;
  const bottom = height - 26;
  const xs = scale(0, 3, left, right);
  const ys = scale(0, 4, bottom, top);
  const rows = residuals(w, b);
  const unitY = ys(0) - ys(1);
  return (
    <svg
      className="gw-plot wai-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Data space. Line ŷ = ${w.toFixed(2)} x + ${b.toFixed(2)} through three points. Residuals ${rows
        .map((row) => row.residual.toFixed(2))
        .join(", ")}.`}
    >
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
      </defs>
      {[0, 1, 2, 3, 4].map((value) => (
        <g key={`y-${value}`}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[0, 1, 2, 3].map((value) => (
        <text key={`x-${value}`} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        x
      </text>
      <text className="gw-label" x={left - 4} y={12}>
        y
      </text>
      <g clipPath={`url(#${id}-clip)`}>
        {rows.map((row) => {
          // A true square on screen whose side is the residual's drawn length, so areas compare as r².
          const side = Math.abs(row.residual) * unitY;
          return (
            <rect
              key={`sq-${row.x}`}
              className="wai-square"
              x={xs(row.x)}
              y={ys(Math.max(row.y, row.prediction))}
              width={side}
              height={side}
            />
          );
        })}
        <line className="gw-forward" x1={xs(0)} y1={ys(b)} x2={xs(3)} y2={ys(w * 3 + b)} />
        {rows.map((row) => (
          <line
            key={`r-${row.x}`}
            className="gw-loss"
            x1={xs(row.x)}
            y1={ys(row.y)}
            x2={xs(row.x)}
            y2={ys(row.prediction)}
          />
        ))}
      </g>
      {FIT_POINTS.map((point) => (
        <g key={point.x}>
          <circle className="gw-dot is-filled" cx={xs(point.x)} cy={ys(point.y)} r="4" />
          <text
            className="gw-note"
            x={xs(point.x) + (point.x > 2 ? -8 : 8)}
            y={ys(point.y) - 7}
            textAnchor={point.x > 2 ? "end" : "start"}
          >
            ({point.x}, {point.y})
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Parameter space: every (w, b) is one line; contours of L, the bottom of the bowl, and the path walked. */
export function ParamPlot({ w, b, trail }: { w: number; b: number; trail: ReadonlyArray<{ w: number; b: number }> }) {
  const id = useId().replace(/:/g, "");
  const width = 300;
  const height = 260;
  const left = 34;
  const right = width - 10;
  const top = 22;
  const bottom = height - 26;
  const xs = scale(W_RANGE[0], W_RANGE[1], left, right);
  const ys = scale(B_RANGE[0], B_RANGE[1], bottom, top);
  const best = leastSquares();
  const g = gradient(w, b);
  const nextW = w - STEP_SIZE * g.w;
  const nextB = b - STEP_SIZE * g.b;
  return (
    <svg
      className="gw-plot wai-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Parameter space. Contours of the loss over w from 0 to 2 and b from 0 to 2.4. The current point is (${w.toFixed(
        2,
      )}, ${b.toFixed(2)}); the minimum is at (${best.w.toFixed(2)}, ${best.b.toFixed(2)}) with L = ${best.loss.toFixed(
        4,
      )}. The next downhill step points to (${nextW.toFixed(2)}, ${nextB.toFixed(2)}).`}
    >
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
        <marker
          id={`${id}-head`}
          viewBox="0 0 10 10"
          refX="8.5"
          refY="5"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M0 0 L10 5 L0 10 z" className="wai-head" />
        </marker>
      </defs>
      {[0, 0.5, 1, 1.5, 2].map((value) => (
        <text key={`w-${value}`} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      {[0, 0.8, 1.6, 2.4].map((value) => (
        <text key={`b-${value}`} className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
          {value}
        </text>
      ))}
      <rect className="wai-frame" x={left} y={top} width={right - left} height={bottom - top} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        w
      </text>
      <text className="gw-label" x={left - 4} y={12}>
        b
      </text>
      <g clipPath={`url(#${id}-clip)`}>
        {LEVELS.map((level) => {
          const points = contour(level);
          const label = points[Math.round(points.length * 0.25)];
          return (
            <g key={level}>
              <polyline
                className="wai-contour"
                points={points.map((point) => `${xs(point.w).toFixed(1)},${ys(point.b).toFixed(1)}`).join(" ")}
              />
              {label && (
                <text className="wai-level" x={xs(label.w)} y={ys(label.b) + 3} textAnchor="middle">
                  {level}
                </text>
              )}
            </g>
          );
        })}
        {trail.length > 1 && (
          <polyline
            className="wai-trail"
            points={trail.map((point) => `${xs(point.w)},${ys(point.b)}`).join(" ")}
          />
        )}
        {trail.slice(0, -1).map((point, index) => (
          <circle key={index} className="wai-trail-dot" cx={xs(point.w)} cy={ys(point.b)} r="2.2" />
        ))}
        {Math.hypot(xs(nextW) - xs(w), ys(nextB) - ys(b)) > 4 && (
          <line
            className="gw-gradient"
            x1={xs(w)}
            y1={ys(b)}
            x2={xs(nextW)}
            y2={ys(nextB)}
            markerEnd={`url(#${id}-head)`}
          />
        )}
      </g>
      <path
        className="wai-best"
        d={`M${xs(best.w) - 5} ${ys(best.b) - 5}l10 10m0 -10l-10 10`}
      />
      <circle className="gw-dot is-forward" cx={xs(w)} cy={ys(b)} r="5" />
    </svg>
  );
}
