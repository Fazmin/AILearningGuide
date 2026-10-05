import { useId } from "react";
import { gaussianPdf, lossOf, sigmoid, stationaryPoints, type Surface } from "./curves";

/** Linear map from a data interval onto a pixel interval. */
const scale = (d0: number, d1: number, r0: number, r1: number) => (value: number) =>
  r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

const fmt = (value: number, digits = 2) => value.toFixed(digits);

function Arrowheads({ id }: { id: string }) {
  return (
    <defs>
      {(["forward", "attention", "loss", "muted", "gradient"] as const).map((tone) => (
        <marker
          key={tone}
          id={`${id}-${tone}`}
          viewBox="0 0 10 10"
          refX="8.5"
          refY="5"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto-start-reverse"
        >
          <path d="M0 0 L10 5 L0 10 z" className={`myn-head myn-head--${tone}`} />
        </marker>
      ))}
    </defs>
  );
}

function PlaneGrid({
  px,
  py,
  limit,
  size,
}: {
  px: (value: number) => number;
  py: (value: number) => number;
  limit: number;
  size: number;
}) {
  const ticks = [];
  for (let value = -Math.floor(limit); value <= Math.floor(limit); value += 1) ticks.push(value);
  return (
    <g>
      {ticks
        .filter((value) => value !== 0)
        .map((value) => (
          <g key={value}>
            <line className="gw-grid" x1={px(value)} x2={px(value)} y1={py(limit)} y2={py(-limit)} />
            <line className="gw-grid" x1={px(-limit)} x2={px(limit)} y1={py(value)} y2={py(value)} />
            <text className="gw-tick" x={px(value)} y={py(0) + 12} textAnchor="middle">
              {value}
            </text>
            <text className="gw-tick" x={px(0) - 5} y={py(value) + 3.5} textAnchor="end">
              {value}
            </text>
          </g>
        ))}
      <line className="gw-axis" x1={px(-limit)} x2={px(limit)} y1={py(0)} y2={py(0)} />
      <line className="gw-axis" x1={px(0)} x2={px(0)} y1={py(limit)} y2={py(-limit)} />
      <text className="gw-label" x={size - 8} y={py(0) - 6} textAnchor="end">
        x₁
      </text>
      <text className="gw-label" x={px(0) + 6} y={14}>
        x₂
      </text>
    </g>
  );
}

/** Two arrows, the projection of a onto b, the perpendicular drop, and the angle between them. */
export function VectorPlane({ ax, ay, bx, by }: { ax: number; ay: number; bx: number; by: number }) {
  const id = useId().replace(/:/g, "");
  const size = 300;
  const limit = 2.4;
  const px = scale(-limit, limit, 12, size - 12);
  const py = scale(-limit, limit, size - 12, 12);
  const dot = ax * bx + ay * by;
  const lengthA = Math.hypot(ax, ay);
  const lengthB = Math.hypot(bx, by);
  const projScale = lengthB > 1e-9 ? dot / (lengthB * lengthB) : 0;
  const projX = projScale * bx;
  const projY = projScale * by;
  const angleA = Math.atan2(ay, ax);
  const angleB = Math.atan2(by, bx);
  const cosine = lengthA * lengthB > 1e-9 ? dot / (lengthA * lengthB) : 0;
  const theta = Math.acos(Math.max(-1, Math.min(1, cosine)));
  let sweep = angleB - angleA;
  while (sweep > Math.PI) sweep -= 2 * Math.PI;
  while (sweep < -Math.PI) sweep += 2 * Math.PI;
  const arcRadius = 0.42;
  const arcEnd = angleA + sweep;
  const arc =
    lengthA > 0.15 && lengthB > 0.15
      ? `M${px(arcRadius * Math.cos(angleA))} ${py(arcRadius * Math.sin(angleA))} A${arcRadius * (px(1) - px(0))} ${
          arcRadius * (px(1) - px(0))
        } 0 0 ${sweep > 0 ? 0 : 1} ${px(arcRadius * Math.cos(arcEnd))} ${py(arcRadius * Math.sin(arcEnd))}`
      : null;
  const mid = angleA + sweep / 2;
  const tipLabel = (x: number, y: number) => {
    const length = Math.hypot(x, y) || 1;
    return { x: px(x + (0.22 * x) / length), y: py(y + (0.22 * y) / length) + 4 };
  };
  const aTip = tipLabel(ax, ay);
  const bTip = tipLabel(bx, by);
  return (
    <svg
      className="gw-plot myn-plane"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Arrow a to (${fmt(ax)}, ${fmt(ay)}) and arrow b to (${fmt(bx)}, ${fmt(by)}). Angle ${Math.round(
        (theta * 180) / Math.PI,
      )} degrees, a·b = ${fmt(dot, 3)}. The projection of a onto b ends at (${fmt(projX)}, ${fmt(projY)}).`}
    >
      <Arrowheads id={id} />
      <PlaneGrid px={px} py={py} limit={limit} size={size} />
      {lengthB > 0.05 && (
        <>
          <line
            className="gw-loss is-dotted"
            x1={px(ax)}
            y1={py(ay)}
            x2={px(projX)}
            y2={py(projY)}
          />
          <line
            className="gw-forward is-dashed myn-proj"
            x1={px(0)}
            y1={py(0)}
            x2={px(projX)}
            y2={py(projY)}
          />
          <circle className="gw-dot is-loss" cx={px(projX)} cy={py(projY)} r="3" />
        </>
      )}
      {arc && <path className="myn-arc" d={arc} />}
      {arc && (
        <text className="gw-note" x={px(0.72 * Math.cos(mid))} y={py(0.72 * Math.sin(mid)) + 3} textAnchor="middle">
          θ {Math.round((theta * 180) / Math.PI)}°
        </text>
      )}
      <line
        className="gw-attention"
        x1={px(0)}
        y1={py(0)}
        x2={px(bx)}
        y2={py(by)}
        markerEnd={`url(#${id}-attention)`}
      />
      <line
        className="gw-forward"
        x1={px(0)}
        y1={py(0)}
        x2={px(ax)}
        y2={py(ay)}
        markerEnd={`url(#${id}-forward)`}
      />
      <text className="gw-label is-forward" x={aTip.x} y={aTip.y} textAnchor="middle">
        a
      </text>
      <text className="gw-label is-attention" x={bTip.x} y={bTip.y} textAnchor="middle">
        b
      </text>
    </svg>
  );
}

/** The square with corners (±1, ±1), its image under x ↦ Wx + b, and one input carried through. */
export function MatrixPlane({
  m,
  b,
  v,
}: {
  m: [number, number, number, number];
  b: [number, number];
  v: [number, number];
}) {
  const id = useId().replace(/:/g, "");
  const size = 300;
  const limit = 3.5;
  const px = scale(-limit, limit, 12, size - 12);
  const py = scale(-limit, limit, size - 12, 12);
  const [m00, m01, m10, m11] = m;
  const apply = (x: number, y: number) => ({ x: m00 * x + m01 * y, y: m10 * x + m11 * y });
  const corners = [
    [-1, 1],
    [1, 1],
    [1, -1],
    [-1, -1],
  ].map(([x, y]) => {
    const mapped = apply(x, y);
    return { x: mapped.x + b[0], y: mapped.y + b[1] };
  });
  const wx = apply(v[0], v[1]);
  const out = { x: wx.x + b[0], y: wx.y + b[1] };
  const outside = [...corners, out].some((point) => Math.abs(point.x) > limit || Math.abs(point.y) > limit);
  return (
    <svg
      className="gw-plot myn-plane"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Square with corners plus or minus one, dashed, and its image under W x plus b, shaded. Input x = (${fmt(
        v[0],
      )}, ${fmt(v[1])}) lands at W x = (${fmt(wx.x)}, ${fmt(wx.y)}), then W x + b = (${fmt(out.x)}, ${fmt(out.y)}).`}
    >
      <Arrowheads id={id} />
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={px(-limit)} y={py(limit)} width={px(limit) - px(-limit)} height={py(-limit) - py(limit)} />
        </clipPath>
      </defs>
      <PlaneGrid px={px} py={py} limit={limit} size={size} />
      <g clipPath={`url(#${id}-clip)`}>
        <polygon
          className="myn-square"
          points={[
            [-1, 1],
            [1, 1],
            [1, -1],
            [-1, -1],
          ]
            .map(([x, y]) => `${px(x)},${py(y)}`)
            .join(" ")}
        />
        <polygon className="myn-image" points={corners.map((point) => `${px(point.x)},${py(point.y)}`).join(" ")} />
        <line
          className="myn-muted-arrow"
          x1={px(0)}
          y1={py(0)}
          x2={px(v[0])}
          y2={py(v[1])}
          markerEnd={`url(#${id}-muted)`}
        />
        <line
          className="gw-forward is-dashed"
          x1={px(0)}
          y1={py(0)}
          x2={px(wx.x)}
          y2={py(wx.y)}
        />
        <line className="gw-loss is-dotted" x1={px(wx.x)} y1={py(wx.y)} x2={px(out.x)} y2={py(out.y)} />
        <line
          className="gw-forward"
          x1={px(0)}
          y1={py(0)}
          x2={px(out.x)}
          y2={py(out.y)}
          markerEnd={`url(#${id}-forward)`}
        />
        <circle className="gw-dot" cx={px(wx.x)} cy={py(wx.y)} r="3" />
      </g>
      <text className="gw-label" x={px(v[0]) + 5} y={py(v[1]) + 14}>
        x
      </text>
      <text className="gw-label is-forward" x={Math.min(size - 40, px(out.x) + 6)} y={Math.max(14, py(out.y) - 6)}>
        Wx+b
      </text>
      {outside && (
        <text className="gw-note" x={size - 10} y={size - 16} textAnchor="end">
          part of the image is off this ±3.5 window
        </text>
      )}
    </svg>
  );
}

/** f(x) = x² with the tangent at x and the secant across the nudge ε, plus the rise-over-run triangle. */
export function DerivativePlot({ x, eps }: { x: number; eps: number }) {
  const id = useId().replace(/:/g, "");
  const width = 340;
  const height = 240;
  const left = 34;
  const right = width - 10;
  const top = 12;
  const bottom = height - 26;
  const xs = scale(-2.3, 3, left, right);
  const ys = scale(-0.8, 8.2, bottom, top);
  const fx = x * x;
  const x2 = x + eps;
  const f2 = x2 * x2;
  const slope = 2 * x;
  const secant = (f2 - fx) / eps;
  const curve = Array.from({ length: 81 }, (_, index) => {
    const value = -2.3 + (5.3 * index) / 80;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(2)} ${ys(value * value).toFixed(2)}`;
  }).join("");
  const reach = 1.6;
  return (
    <svg
      className="gw-plot myn-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Parabola f of x equals x squared. At x = ${fmt(x)} the tangent slope is ${fmt(slope, 3)}. The secant to x + ε = ${fmt(
        x2,
      )} has slope ${fmt(secant, 3)}, larger by exactly ε = ${fmt(eps)}.`}
    >
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
      </defs>
      {[0, 2, 4, 6, 8].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[-2, -1, 0, 1, 2, 3].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={ys(0)} y2={ys(0)} />
      <line className="gw-axis" x1={xs(0)} x2={xs(0)} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        x
      </text>
      <g clipPath={`url(#${id}-clip)`}>
        <path className="gw-curve" d={curve} />
        <line
          className="gw-gradient"
          x1={xs(x - reach)}
          y1={ys(fx - slope * reach)}
          x2={xs(x + reach)}
          y2={ys(fx + slope * reach)}
        />
        <line
          className="gw-loss is-dashed"
          x1={xs(x - 0.9)}
          y1={ys(fx - secant * 0.9)}
          x2={xs(x2 + 0.9)}
          y2={ys(f2 + secant * 0.9)}
        />
        <line className="myn-run" x1={xs(x)} y1={ys(fx)} x2={xs(x2)} y2={ys(fx)} />
        <line className="myn-run" x1={xs(x2)} y1={ys(fx)} x2={xs(x2)} y2={ys(f2)} />
      </g>
      <text className="gw-note" x={(xs(x) + xs(x2)) / 2} y={ys(fx) + 12} textAnchor="middle">
        ε
      </text>
      <text className="gw-note" x={xs(x2) + 4} y={(ys(fx) + ys(f2)) / 2 + 3}>
        Δf
      </text>
      <circle className="gw-dot is-filled" cx={xs(x)} cy={ys(fx)} r="3.6" />
      <circle className="gw-dot is-loss" cx={xs(x2)} cy={ys(f2)} r="3.6" />
      <text className="gw-label" x={xs(-2.15)} y={ys(4.62) - 6}>
        f(x) = x²
      </text>
    </svg>
  );
}

/** L(w) with its minima, the current w, its tangent, the next step, and the path walked so far. */
export function LossPlot({
  surface,
  w,
  eta,
  trail,
}: {
  surface: Surface;
  w: number;
  eta: number;
  trail: readonly number[];
}) {
  const id = useId().replace(/:/g, "");
  const width = 340;
  const height = 250;
  const left = 34;
  const right = width - 10;
  const top = 24;
  const bottom = height - 26;
  const yMax = surface === "convex" ? 8.4 : 3.6;
  const xs = scale(-2, 2, left, right);
  const ys = scale(0, yMax, bottom, top);
  const loss = lossOf(surface, w);
  const grad =
    surface === "convex" ? 2 * (w - 0.8) : 4 * w * (w * w - 1) + 0.35;
  const next = Math.min(2, Math.max(-2, w - eta * grad));
  const nextLoss = lossOf(surface, next);
  const curve = Array.from({ length: 161 }, (_, index) => {
    const value = -2 + (4 * index) / 160;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(2)} ${ys(lossOf(surface, value)).toFixed(2)}`;
  }).join("");
  const minima = stationaryPoints(surface).filter((point) => point.kind === "minimum");
  const deepest = Math.min(...minima.map((point) => point.loss));
  const offChart = loss > yMax;
  const ballY = offChart ? top + 6 : ys(loss);
  const pxPerW = (right - left) / 4;
  const pxPerL = (bottom - top) / yMax;
  const tangentReach = 30 / Math.sqrt(pxPerW ** 2 + (grad * pxPerL) ** 2);
  const ticks = surface === "convex" ? [0, 2, 4, 6, 8] : [0, 1, 2, 3];
  return (
    <svg
      className="gw-plot myn-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${surface === "convex" ? "Convex bowl" : "Tilted double well"}. At w = ${fmt(w)}, L = ${fmt(
        loss,
        3,
      )} and L' = ${fmt(grad, 3)}. The next step moves w to ${fmt(next, 3)}. ${minima
        .map((point) => `Minimum at w = ${fmt(point.w)} with L = ${fmt(point.loss, 3)}`)
        .join(". ")}.`}
    >
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
      </defs>
      <Arrowheads id={id} />
      {ticks.map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[-2, -1, 0, 1, 2].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        w
      </text>
      <text className="gw-label" x={left - 4} y={12}>
        L(w)
      </text>
      <g clipPath={`url(#${id}-clip)`}>
        <path className="gw-loss" d={curve} />
        {minima.map((point) => (
          <g key={point.w}>
            <line className="gw-grid" x1={xs(point.w)} x2={xs(point.w)} y1={ys(point.loss)} y2={bottom} />
            <path
              className="myn-min"
              d={`M${xs(point.w)} ${ys(point.loss) + 3} l-4 7 h8 z`}
            />
          </g>
        ))}
        {trail.length > 1 && (
          <polyline
            className="myn-trail"
            points={trail.map((value) => `${xs(value)},${ys(Math.min(yMax, lossOf(surface, value)))}`).join(" ")}
          />
        )}
        {trail.slice(0, -1).map((value, index) => (
          <circle key={`${value}-${index}`} className="myn-trail-dot" cx={xs(value)} cy={ys(Math.min(yMax, lossOf(surface, value)))} r="2.2" />
        ))}
        {!offChart && (
          <line
            className="gw-gradient"
            x1={xs(w - tangentReach)}
            y1={ys(loss - grad * tangentReach)}
            x2={xs(w + tangentReach)}
            y2={ys(loss + grad * tangentReach)}
          />
        )}
        {Math.abs(next - w) > 0.01 && (
          <path
            className="gw-forward is-dashed"
            d={`M${xs(w)} ${ballY} Q${(xs(w) + xs(next)) / 2} ${Math.min(ballY, ys(Math.min(yMax, nextLoss))) - 18} ${xs(next)} ${ys(
              Math.min(yMax, nextLoss),
            )}`}
            markerEnd={`url(#${id}-forward)`}
          />
        )}
      </g>
      {minima.map((point) => (
        <text key={`label-${point.w}`} className="gw-note is-positive" x={xs(point.w)} y={bottom + 24} textAnchor="middle">
          {minima.length === 1 ? "minimum" : point.loss === deepest ? "global min" : "local min"}
        </text>
      ))}
      <circle className="gw-dot is-filled myn-ball" cx={xs(w)} cy={ballY} r="5" />
      {offChart && (
        <text className="gw-note" x={xs(w) + (w > 1 ? -8 : 8)} y={top + 10} textAnchor={w > 1 ? "end" : "start"}>
          ↑ L = {fmt(loss)} (off chart)
        </text>
      )}
    </svg>
  );
}

/** Surprise −ln u, with the model's two surprises and the truth's two surprises, then the weighted sums as bars. */
export function SurprisePlot({ p, q }: { p: number; q: number }) {
  const width = 360;
  const height = 240;
  const left = 32;
  const plotRight = 212;
  const top = 24;
  const bottom = height - 28;
  const yMax = 4;
  const xs = scale(0, 1, left, plotRight);
  const ys = scale(0, yMax, bottom, top);
  const curve = Array.from({ length: 121 }, (_, index) => {
    const value = 0.02 + (0.98 * index) / 120;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(2)} ${ys(-Math.log(value)).toFixed(2)}`;
  }).join("");
  const surprise = (value: number) => -Math.log(value);
  const hp = p * surprise(p) + (1 - p) * surprise(1 - p);
  const hpq = p * surprise(q) + (1 - p) * surprise(1 - q);
  const barMax = Math.max(1, hpq * 1.12);
  const barScale = scale(0, barMax, bottom, top + 12);
  const barWidth = 34;
  const barX = [246, 298];
  const stack = (parts: [number, number]) => [
    { y0: 0, y1: parts[0] },
    { y0: parts[0], y1: parts[0] + parts[1] },
  ];
  const truthStack = stack([p * surprise(p), (1 - p) * surprise(1 - p)]);
  const modelStack = stack([p * surprise(q), (1 - p) * surprise(1 - q)]);
  const clip = (value: number) => Math.min(yMax, value);
  const clipBar = (value: number) => Math.min(barMax, value);
  return (
    <svg
      className="gw-plot myn-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Surprise curve minus ln u. Truth p = ${fmt(p)}, model q = ${fmt(q)}. Entropy H(p) = ${fmt(
        hp,
        3,
      )} nats, cross-entropy H(p,q) = ${fmt(hpq, 3)} nats, KL = ${fmt(hpq - hp, 3)} nats.`}
    >
      {[0, 1, 2, 3, 4].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={plotRight} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[0, 0.5, 1].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={plotRight} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={plotRight} y={bottom + 24} textAnchor="end">
        probability u
      </text>
      <text className="gw-label" x={left - 4} y={12}>
        −ln u (nats)
      </text>
      <text className="gw-label" x={barX[0] - 6} y={12}>
        expected surprise
      </text>
      <path className="gw-curve" d={curve} />
      {[
        { u: q, tone: "is-loss", label: "q" },
        { u: 1 - q, tone: "is-loss", label: "1−q" },
      ].map((mark) => (
        <g key={mark.label}>
          <line className="gw-grid" x1={xs(mark.u)} x2={xs(mark.u)} y1={ys(clip(surprise(mark.u)))} y2={bottom} />
          <circle className={`gw-dot ${mark.tone}`} cx={xs(mark.u)} cy={ys(clip(surprise(mark.u)))} r="3.8" />
          <text className="gw-note" x={xs(mark.u)} y={bottom - 4} textAnchor="middle">
            {mark.label}
          </text>
        </g>
      ))}
      {[
        { u: p, label: "p" },
        { u: 1 - p, label: "1−p" },
      ].map((mark) => (
        <circle key={mark.label} className="gw-dot myn-truth" cx={xs(mark.u)} cy={ys(clip(surprise(mark.u)))} r="5.4" />
      ))}
      {[
        { x: barX[0], parts: truthStack, label: "H(p)", total: hp, tone: "myn-bar-truth" },
        { x: barX[1], parts: modelStack, label: "H(p,q)", total: hpq, tone: "myn-bar-model" },
      ].map((bar) => (
        <g key={bar.label}>
          {bar.parts.map((part, index) => (
            <rect
              key={index}
              className={`${bar.tone} ${index === 0 ? "is-first" : "is-second"}`}
              x={bar.x}
              width={barWidth}
              y={barScale(clipBar(part.y1))}
              height={Math.max(0, barScale(clipBar(part.y0)) - barScale(clipBar(part.y1)))}
            />
          ))}
          <text className="gw-tick" x={bar.x + barWidth / 2} y={bottom + 13} textAnchor="middle">
            {bar.label}
          </text>
          <text className="gw-label" x={bar.x + barWidth / 2} y={barScale(clipBar(bar.total)) - 5} textAnchor="middle">
            {fmt(bar.total, 3)}
          </text>
        </g>
      ))}
      <line className="gw-axis" x1={barX[0] - 6} x2={barX[1] + barWidth + 6} y1={bottom} y2={bottom} />
      {hpq - hp > 0.02 && (
        <g>
          <line
            className="gw-loss"
            x1={barX[1] + barWidth + 3}
            x2={barX[1] + barWidth + 3}
            y1={barScale(clipBar(hpq))}
            y2={barScale(clipBar(hp))}
          />
          <line
            className="gw-grid"
            x1={barX[0]}
            x2={barX[1] + barWidth + 6}
            y1={barScale(clipBar(hp))}
            y2={barScale(clipBar(hp))}
          />
          <text
            className="gw-note is-loss"
            x={width - 2}
            y={(barScale(clipBar(hpq)) + barScale(clipBar(hp))) / 2 + 3}
            textAnchor="end"
          >
            KL
          </text>
        </g>
      )}
    </svg>
  );
}

/** A Gaussian density with its mean, a one-standard-deviation band, a probe, and the four sample points as ticks. */
export function GaussianPlot({
  mean,
  variance,
  x,
  sample,
}: {
  mean: number;
  variance: number;
  x: number;
  sample: readonly number[];
}) {
  const id = useId().replace(/:/g, "");
  const width = 360;
  const height = 230;
  const left = 34;
  const right = width - 10;
  const top = 16;
  const bottom = height - 28;
  const xs = scale(-4, 4, left, right);
  const ys = scale(0, 1.4, bottom, top);
  const sigma = Math.sqrt(variance);
  const curve = Array.from({ length: 161 }, (_, index) => {
    const value = -4 + (8 * index) / 160;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(2)} ${ys(gaussianPdf(value, mean, variance)).toFixed(2)}`;
  }).join("");
  const density = gaussianPdf(x, mean, variance);
  const bandLeft = Math.max(-4, mean - sigma);
  const bandRight = Math.min(4, mean + sigma);
  return (
    <svg
      className="gw-plot myn-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Gaussian density with mean ${fmt(mean)} and variance ${fmt(variance)}, so standard deviation ${fmt(
        sigma,
        3,
      )}. The density at x = ${fmt(x, 1)} is ${fmt(density, 3)}. The band from ${fmt(mean - sigma)} to ${fmt(
        mean + sigma,
      )} holds about 68 percent of the area.`}
    >
      <defs>
        <clipPath id={`${id}-clip`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
      </defs>
      {[0, 0.5, 1].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[-4, -2, 0, 2, 4].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        x
      </text>
      <text className="gw-label" x={left - 4} y={11}>
        density
      </text>
      <g clipPath={`url(#${id}-clip)`}>
        <rect className="myn-image" x={xs(bandLeft)} y={top} width={Math.max(0, xs(bandRight) - xs(bandLeft))} height={bottom - top} />
        <path className="gw-curve" d={curve} />
        <line className="gw-gradient is-dashed" x1={xs(mean)} x2={xs(mean)} y1={top} y2={bottom} />
      </g>
      <text className="gw-note" x={xs(mean) + 4} y={top + 10}>
        μ = {fmt(mean)}
      </text>
      <text className="gw-note" x={xs(bandLeft) + 3} y={bottom - 5}>
        μ ± σ
      </text>
      <line className="myn-run" x1={xs(x)} x2={xs(x)} y1={ys(density)} y2={bottom} />
      <circle className="gw-dot is-loss" cx={xs(x)} cy={ys(density)} r="4" />
      {sample.map((value) => (
        <line key={value} className="gw-loss" x1={xs(value)} x2={xs(value)} y1={bottom} y2={bottom - 8} />
      ))}
    </svg>
  );
}

/** The sigmoid curve, with the current logit marked and the log-odds line for reference. */
export function SigmoidPlot({ z }: { z: number }) {
  const width = 340;
  const height = 220;
  const left = 34;
  const right = width - 10;
  const top = 14;
  const bottom = height - 28;
  const xs = scale(-6, 6, left, right);
  const ys = scale(0, 1, bottom, top);
  const curve = Array.from({ length: 121 }, (_, index) => {
    const value = -6 + (12 * index) / 120;
    return `${index === 0 ? "M" : "L"}${xs(value).toFixed(2)} ${ys(sigmoid(value)).toFixed(2)}`;
  }).join("");
  const clamped = Math.max(-6, Math.min(6, z));
  const p = sigmoid(z);
  return (
    <svg
      className="gw-plot myn-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Sigmoid curve from logit z to probability. At z = ${fmt(z)} the probability is ${fmt(p, 3)}. At z = 0 it is 0.5, and it flattens toward 0 and 1 at either end.`}
    >
      {[0, 0.5, 1].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[-6, -3, 0, 3, 6].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={xs(0)} x2={xs(0)} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 24} textAnchor="end">
        logit z
      </text>
      <text className="gw-label" x={left - 4} y={11}>
        σ(z) = P
      </text>
      <path className="gw-curve" d={curve} />
      <line className="myn-run" x1={xs(clamped)} x2={xs(clamped)} y1={ys(p)} y2={bottom} />
      <line className="myn-run" x1={left} x2={xs(clamped)} y1={ys(p)} y2={ys(p)} />
      <circle className="gw-dot is-forward" cx={xs(clamped)} cy={ys(p)} r="4.2" />
      <text className="gw-note" x={xs(clamped) + (clamped > 3 ? -6 : 6)} y={ys(p) + (p > 0.5 ? 14 : -7)} textAnchor={clamped > 3 ? "end" : "start"}>
        z = {fmt(z, 1)}, P = {fmt(p, 3)}
      </text>
    </svg>
  );
}
