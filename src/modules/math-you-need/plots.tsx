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
      <text className="gw-label myn-halo" x={px(v[0]) - 4} y={py(v[1]) + (out.y >= v[1] ? 15 : -7)} textAnchor="end">
        x
      </text>
      <text
        className="gw-label is-forward myn-halo"
        x={Math.min(size - 40, px(out.x) + 7)}
        y={Math.min(size - 8, Math.max(14, py(out.y) + (out.y >= v[1] ? -6 : 14)))}
      >
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
  // The area under the curve between μ − σ and μ + σ: the 68% the note talks about.
  const area =
    `M${xs(bandLeft).toFixed(2)} ${bottom}` +
    Array.from({ length: 41 }, (_, index) => {
      const value = bandLeft + ((bandRight - bandLeft) * index) / 40;
      return `L${xs(value).toFixed(2)} ${ys(gaussianPdf(value, mean, variance)).toFixed(2)}`;
    }).join("") +
    `L${xs(bandRight).toFixed(2)} ${bottom}Z`;
  const peak = gaussianPdf(mean, mean, variance);
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
        <path className="myn-area" d={area} />
        {[bandLeft, bandRight].map((edge) => (
          <line
            key={edge}
            className="myn-area-edge"
            x1={xs(edge)}
            x2={xs(edge)}
            y1={ys(gaussianPdf(edge, mean, variance))}
            y2={bottom}
          />
        ))}
        <path className="gw-curve" d={curve} />
        <line className="gw-gradient is-dashed" x1={xs(mean)} x2={xs(mean)} y1={top} y2={bottom} />
      </g>
      <text className="gw-note is-gradient myn-halo" x={xs(mean) + 4} y={top + 10}>
        μ = {fmt(mean)}
      </text>
      <text
        className="myn-area-name myn-halo"
        x={Math.min(right - 4, Math.max(left + 4, xs(mean)))}
        y={Math.min(bottom - 26, Math.max(top + 22, ys(peak) - 6))}
        textAnchor={xs(mean) > right - 60 ? "end" : xs(mean) < left + 60 ? "start" : "middle"}
      >
        ≈68% of the area
      </text>
      <text
        className="gw-note is-forward myn-halo"
        x={xs(bandLeft) - 4}
        y={Math.max(top + 9, ys(gaussianPdf(bandLeft, mean, variance)) - 3)}
        textAnchor="end"
      >
        μ − σ
      </text>
      <text
        className="gw-note is-forward myn-halo"
        x={xs(bandRight) + 4}
        y={Math.max(top + 9, ys(gaussianPdf(bandRight, mean, variance)) - 3)}
      >
        μ + σ
      </text>
      <line className="myn-run" x1={xs(x)} x2={xs(x)} y1={ys(density)} y2={bottom} />
      <circle className="gw-dot is-loss" cx={xs(x)} cy={ys(density)} r="4" />
      {sample.map((value) => (
        <g key={value}>
          <line className="gw-loss" x1={xs(value)} x2={xs(value)} y1={bottom + 1} y2={bottom - 9} />
          <circle className="gw-dot is-loss" cx={xs(value)} cy={bottom - 9} r="2.4" />
        </g>
      ))}
    </svg>
  );
}

/** The sigmoid curve, with the current logit marked and the log-odds line for reference. */
export function SigmoidPlot({ z }: { z: number }) {
  const width = 340;
  const height = 228;
  const left = 34;
  const right = width - 10;
  const top = 24;
  const bottom = height - 32;
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
      <text className="gw-label" x={(left + right) / 2} y={bottom + 27} textAnchor="middle">
        logit gap z = A − B
      </text>
      <text className="gw-label" x={left - 4} y={12}>
        σ(z) = P(A)
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

/**
 * Bayes' rule as areas. The square is every case: its width is split by the prior, and each column is filled
 * up to that class's likelihood of x = 1. The two filled areas are the joint weights; the bar on the right
 * keeps only those and rescales them to 1, and the y = 1 share of that bar is the posterior.
 */
export function BayesArea({
  prior,
  likePositive,
  likeNegative,
}: {
  prior: number;
  likePositive: number;
  likeNegative: number;
}) {
  const id = useId().replace(/:/g, "");
  const width = 440;
  const height = 232;
  const x0 = 50;
  const x1 = 262;
  const y0 = 30;
  const y1 = 184;
  const side = y1 - y0;
  const split = x0 + prior * (x1 - x0);
  const jointPositive = prior * likePositive;
  const jointNegative = (1 - prior) * likeNegative;
  const evidence = jointPositive + jointNegative;
  const posterior = jointPositive / evidence;
  const heightPositive = likePositive * side;
  const heightNegative = likeNegative * side;
  const barX = 346;
  const barW = 42;
  const cut = y1 - posterior * side;
  return (
    <svg
      className="gw-plot myn-plot myn-bayes"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Bayes' rule drawn as areas. A unit square is split by the prior: ${fmt(prior)} of its width is class y = 1 and ${fmt(
        1 - prior,
      )} is class y = 0. The y = 1 column is filled to ${fmt(likePositive)} of its height and the y = 0 column to ${fmt(
        likeNegative,
      )}, the shares of each class with x = 1. The filled areas are ${fmt(jointPositive, 3)} and ${fmt(
        jointNegative,
        3,
      )}. Rescaled to add to 1, the y = 1 share is the posterior, ${fmt(posterior, 3)}.`}
    >
      <defs>
        <pattern id={`${id}-hatch`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect className="myn-bayes__neg-base" width="6" height="6" />
          <line className="myn-bayes__neg-line" x1="0" y1="0" x2="0" y2="6" />
        </pattern>
      </defs>
      <text className="gw-label" x={x0} y={14}>
        every case, split by class
      </text>
      <text className="gw-label" x={barX + barW / 2} y={14} textAnchor="middle">
        only x = 1
      </text>
      <rect className="myn-bayes__all" x={x0} y={y0} width={x1 - x0} height={side} />
      <rect className="myn-bayes__pos" x={x0} y={y1 - heightPositive} width={split - x0} height={heightPositive} />
      <rect
        x={split}
        y={y1 - heightNegative}
        width={x1 - split}
        height={heightNegative}
        fill={`url(#${id}-hatch)`}
        className="myn-bayes__neg"
      />
      <line className="myn-bayes__split" x1={split} x2={split} y1={y0 - 4} y2={y1 + 4} />
      <text className="gw-tick is-forward" x={x0 - 5} y={y1 - heightPositive + 3.5} textAnchor="end">
        {fmt(likePositive)}
      </text>
      <text className="gw-tick is-attention" x={x1 + 5} y={y1 - heightNegative + 3.5}>
        {fmt(likeNegative)}
      </text>
      <text className="gw-tick" x={x0 - 5} y={y0 + 3.5} textAnchor="end">
        1
      </text>
      <text className="gw-tick" x={x0 - 5} y={y1 + 3.5} textAnchor="end">
        0
      </text>
      <text className="gw-note" x={x0 - 38} y={(y0 + y1) / 2} transform={`rotate(-90 ${x0 - 38} ${(y0 + y1) / 2})`} textAnchor="middle">
        share with x = 1
      </text>
      {split - x0 > 46 && heightPositive > 30 && (
        <text className="myn-bayes__value" x={(x0 + split) / 2} y={y1 - heightPositive / 2 + 4} textAnchor="middle">
          {fmt(jointPositive, 3)}
        </text>
      )}
      {x1 - split > 46 && heightNegative > 18 && (
        <text className="myn-bayes__value is-neg myn-halo" x={(split + x1) / 2} y={y1 - heightNegative / 2 + 4} textAnchor="middle">
          {fmt(jointNegative, 3)}
        </text>
      )}
      {split - x0 > 46 && side - heightPositive > 26 && (
        <text className="gw-note" x={(x0 + split) / 2} y={y0 + 16} textAnchor="middle">
          x = 0
        </text>
      )}
      {x1 - split > 46 && (
        <text className="gw-note" x={(split + x1) / 2} y={y0 + 16} textAnchor="middle">
          x = 0
        </text>
      )}
      <text className="gw-label is-forward" x={x0} y={y1 + 17}>
        y = 1
      </text>
      <text className="gw-note" x={x0} y={y1 + 30}>
        prior {fmt(prior)}
      </text>
      <text className="gw-label is-attention" x={x1} y={y1 + 17} textAnchor="end">
        y = 0
      </text>
      <text className="gw-note" x={x1} y={y1 + 30} textAnchor="end">
        1 − prior {fmt(1 - prior)}
      </text>
      <path className="myn-bayes__flow" d={`M${x1 + 34} ${(y0 + y1) / 2}H${barX - 12}m-6 -4l6 4l-6 4`} />
      <text className="gw-note" x={(x1 + 34 + barX - 12) / 2} y={(y0 + y1) / 2 - 20} textAnchor="middle">
        keep the
      </text>
      <text className="gw-note" x={(x1 + 34 + barX - 12) / 2} y={(y0 + y1) / 2 - 8} textAnchor="middle">
        filled parts
      </text>
      <text className="gw-note" x={(x1 + 34 + barX - 12) / 2} y={(y0 + y1) / 2 + 17} textAnchor="middle">
        rescale
      </text>
      <text className="gw-note" x={(x1 + 34 + barX - 12) / 2} y={(y0 + y1) / 2 + 29} textAnchor="middle">
        to 1
      </text>
      <rect x={barX} y={y0} width={barW} height={cut - y0} fill={`url(#${id}-hatch)`} className="myn-bayes__neg" />
      <rect className="myn-bayes__pos" x={barX} y={cut} width={barW} height={y1 - cut} />
      <rect className="myn-bayes__frame" x={barX} y={y0} width={barW} height={side} />
      <text
        className="gw-label is-forward"
        x={barX + barW + 6}
        y={Math.min(y1 - 14, Math.max(y0 + 34, (cut + y1) / 2))}
      >
        y = 1
      </text>
      <text
        className="myn-bayes__value is-out"
        x={barX + barW + 6}
        y={Math.min(y1 - 1, Math.max(y0 + 47, (cut + y1) / 2 + 13))}
      >
        {fmt(posterior, 3)}
      </text>
      <text className="gw-note is-attention" x={barX + barW + 6} y={y0 + 10}>
        y = 0
      </text>
      <text className="gw-label" x={barX + barW / 2} y={y1 + 17} textAnchor="middle">
        posterior
      </text>
      <text className="gw-note" x={barX + barW / 2} y={y1 + 30} textAnchor="middle">
        P(y=1|x=1)
      </text>
    </svg>
  );
}

/** Two logits on one number line with the gap between them bracketed: the only thing two-class softmax reads. */
export function LogitGap({ a, b }: { a: number; b: number }) {
  const width = 340;
  const height = 74;
  const left = 34;
  const right = width - 10;
  const axis = 46;
  const xs = scale(0, 6, left, right);
  const low = Math.min(a, b);
  const high = Math.max(a, b);
  const close = Math.abs(xs(a) - xs(b)) < 16;
  return (
    <svg
      className="gw-plot myn-plot myn-gap"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Logit A = ${fmt(a, 1)} and Logit B = ${fmt(b, 1)} on one number line. The gap between them is z = ${fmt(
        a - b,
        1,
      )}. Moving both by the same amount leaves the gap unchanged.`}
    >
      <line className="gw-axis" x1={left} x2={right} y1={axis} y2={axis} />
      {[0, 1, 2, 3, 4, 5, 6].map((value) => (
        <g key={value}>
          <line className="gw-axis" x1={xs(value)} x2={xs(value)} y1={axis} y2={axis + 4} />
          <text className="gw-tick" x={xs(value)} y={axis + 15} textAnchor="middle">
            {value}
          </text>
        </g>
      ))}
      <text className="gw-label" x={6} y={13}>
        the two logits
      </text>
      {high - low > 0.05 && (
        <>
          <path className="myn-gap__brace" d={`M${xs(low)} ${axis - 16}v-6H${xs(high)}v6`} />
          <rect className="myn-gap__span" x={xs(low)} y={axis - 4} width={xs(high) - xs(low)} height={8} />
        </>
      )}
      <text className="myn-gap__z" x={(xs(low) + xs(high)) / 2} y={axis - 27} textAnchor="middle">
        gap z = A − B = {fmt(a - b, 1)}
      </text>
      <circle className="gw-dot is-forward" cx={xs(a)} cy={axis} r="5" />
      <circle className="gw-dot is-attention" cx={xs(b)} cy={axis} r="5" />
      <text className="gw-label is-forward myn-halo" x={xs(a) + (close ? (a >= b ? 8 : -8) : 0)} y={axis - 9} textAnchor={close ? (a >= b ? "start" : "end") : "middle"}>
        A
      </text>
      <text className="gw-label is-attention myn-halo" x={xs(b) + (close ? (a >= b ? -8 : 8) : 0)} y={axis - 9} textAnchor={close ? (a >= b ? "end" : "start") : "middle"}>
        B
      </text>
    </svg>
  );
}

/** A table row's cases drawn as marks: filled for the counted outcome, hollow for the other. */
export function CountMarks({ filled, hollow, label }: { filled: number; hollow: number; label: string }) {
  return (
    <span className="myn-marks" role="img" aria-label={label}>
      {Array.from({ length: hollow }, (_, index) => (
        <i key={`h-${index}`} />
      ))}
      {Array.from({ length: filled }, (_, index) => (
        <i key={`f-${index}`} className="is-on" />
      ))}
    </span>
  );
}
