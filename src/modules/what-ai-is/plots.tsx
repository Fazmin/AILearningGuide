import { useId } from "react";
import { B_RANGE, W_RANGE, contour, gradient, leastSquares, residuals, STEP_SIZE } from "./fit";

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
  // Name the line above its right-hand end; if it leaves through the top, name it where it leaves.
  const exitX = w > 1e-6 ? Math.min(3, (4 - b) / w) : 3;
  const lineLabel = {
    x: Math.max(left + 100, xs(exitX) - 16),
    y: Math.max(top + 9, Math.min(bottom - 6, ys(w * exitX + b) - 7)),
  };
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
      {rows.map((row) => {
        // The residual's value, printed past its square on the side the line is not on.
        const side = Math.abs(row.residual) * unitY;
        if (side < 9) return null;
        return (
          <text
            key={`rv-${row.x}`}
            className="wai-miss"
            x={Math.min(xs(row.x) + side + 4, right - 26)}
            y={Math.min(bottom - 4, Math.max(top + 9, (ys(row.y) + ys(row.prediction)) / 2 + 3.5))}
          >
            {row.residual > 0 ? "+" : "−"}
            {Math.abs(row.residual).toFixed(2)}
          </text>
        );
      })}
      {rows.map((row) => {
        // The label sits on the side of the dot the line is not on, so it never covers the residual.
        const below = row.prediction >= row.y;
        const y = Math.min(bottom - 4, Math.max(top + 9, ys(row.y) + (below ? 16 : -9)));
        return (
          <g key={row.x}>
            <circle className="gw-dot is-filled" cx={xs(row.x)} cy={ys(row.y)} r="4" />
            <text
              className="gw-note wai-halo"
              x={xs(row.x) + (row.x > 2 ? -7 : 7)}
              y={y}
              textAnchor={row.x > 2 ? "end" : "start"}
            >
              ({row.x}, {row.y})
            </text>
          </g>
        );
      })}
      <text className="wai-line-name" x={lineLabel.x} y={lineLabel.y} textAnchor="end">
        ŷ = {w.toFixed(2)}x + {b.toFixed(2)}
      </text>
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
      <text
        className="wai-pin is-best"
        x={xs(best.w) + (xs(best.w) > right - 70 ? -9 : 9)}
        y={ys(best.b) + (ys(b) > ys(best.b) + 2 ? -8 : 16)}
        textAnchor={xs(best.w) > right - 70 ? "end" : "start"}
      >
        lowest L
      </text>
      <circle className="gw-dot is-forward" cx={xs(w)} cy={ys(b)} r="5" />
      <text
        className="wai-pin is-you"
        x={xs(w) + (xs(w) > right - 62 ? -9 : 9)}
        y={ys(b) + (ys(b) > ys(best.b) + 2 ? 16 : -8)}
        textAnchor={xs(w) > right - 62 ? "end" : "start"}
      >
        your line
      </text>
    </svg>
  );
}

/**
 * L after each press of the downhill button, on the path walked so far. The dashed floor is the least-squares
 * minimum, so the picture shows both how fast L falls and how little is left to gain.
 */
export function LossPath({ losses, floor }: { losses: readonly number[]; floor: number }) {
  const width = 660;
  const height = 104;
  const left = 46;
  const right = width - 150;
  const top = 18;
  const bottom = height - 24;
  const first = losses[0] ?? 0;
  const last = losses[losses.length - 1] ?? 0;
  const peak = Math.max(0.1, ...losses) * 1.1;
  const count = Math.max(6, losses.length - 1);
  const xs = scale(0, count, left, right);
  const ys = scale(0, peak, bottom, top);
  const path = losses
    .map((value, index) => `${index === 0 ? "M" : "L"}${xs(index).toFixed(1)} ${ys(value).toFixed(1)}`)
    .join("");
  const presses = losses.length - 1;
  const every = Math.ceil(count / 10);
  return (
    <svg
      className="gw-plot wai-plot wai-losspath"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={
        presses > 0
          ? `Loss after each downhill step. L started at ${first.toFixed(3)} and is ${last.toFixed(3)} after ${presses} ${
              presses === 1 ? "press" : "presses"
            }. The lowest possible L is ${floor.toFixed(4)}.`
          : `Loss path. L is ${last.toFixed(3)} and no downhill step has been taken yet. The lowest possible L is ${floor.toFixed(
              4,
            )}.`
      }
    >
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-tick" x={left - 5} y={bottom + 3.5} textAnchor="end">
        0
      </text>
      <text className="gw-tick" x={left - 8} y={ys(first) + 3.5} textAnchor="end">
        {first.toFixed(2)}
      </text>
      <text className="gw-label" x={6} y={11}>
        L after each press
      </text>
      <text className="gw-tick" x={left} y={bottom + 14} textAnchor="middle">
        start
      </text>
      {Array.from({ length: count }, (_, index) => index + 1)
        .filter((press) => press % every === 0)
        .map((press) => (
          <text key={press} className="gw-tick" x={xs(press)} y={bottom + 14} textAnchor="middle">
            {press}
          </text>
        ))}
      <line className="wai-floor" x1={left} x2={right} y1={ys(floor)} y2={ys(floor)} />
      {losses.length > 1 && <path className="gw-loss" d={path} />}
      {losses.slice(0, -1).map((value, index) => (
        <g key={index}>
          <circle className="wai-trail-dot" cx={xs(index)} cy={ys(value)} r="2.6" />
          {losses.length <= 7 && index > 0 && (
            <text className="gw-tick wai-halo" x={xs(index)} y={ys(value) - 7} textAnchor="middle">
              {value.toFixed(3)}
            </text>
          )}
        </g>
      ))}
      <circle className="gw-dot is-loss" cx={xs(losses.length - 1)} cy={ys(last)} r="4.5" />
      <text className="wai-miss" x={xs(losses.length - 1) + 8} y={Math.max(top + 4, ys(last) - 7)}>
        L = {last.toFixed(3)}
      </text>
      <text className="wai-pin is-best" x={right + 8} y={ys(floor) - 5}>
        lowest possible L
      </text>
      <text className="wai-pin is-best" x={right + 8} y={ys(floor) + 8}>
        {floor.toFixed(4)}
      </text>
      {presses === 0 && (
        <text className="gw-note" x={xs(1.4)} y={(top + bottom) / 2 + 4}>
          Each press of Take a downhill step adds a point here.
        </text>
      )}
    </svg>
  );
}
