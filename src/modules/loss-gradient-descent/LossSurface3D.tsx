import type { CSSProperties } from "react";
import { SurfaceStage, type StageView, type SurfaceArrow, type SurfacePath, type SurfacePoint } from "@app/module-sdk";
import { B_RANGE, HESSIAN, LEAST_SQUARES, mse, W_RANGE, type DescentPoint } from "./descent";

/** Loss above this is cut off: the walls keep rising, and drawing them would flatten the valley floor. */
export const SURFACE_TOP = 6;
const HEIGHTS = [0, SURFACE_TOP] as const;
const W_TICKS = [-1, 0, 1, 2, 3];
const B_TICKS = [-2, -1, 0, 1, 2];
/** Looking up the valley from its low end, a little to one side, so both walls and the floor show. */
const VIEW: StageView = { azimuth: 24, elevation: 34, distance: 5.6, target: [0, 0.3, 0] };
const L_TICKS = [0, 2, 4, 6];
const AXES = { x: "slope w →", y: "intercept b →", z: "loss L" };
/** How long the −∇L arrow is drawn, in units of w and b. Direction is the claim; the metrics carry the size. */
const ARROW_LENGTH = 0.75;

const key = (tone: string): CSSProperties => ({ "--key": `var(${tone})` }) as CSSProperties;

/**
 * The Loss surface card's map, lifted: the same mean squared error drawn as ground, with the run
 * lying on it. Every height is `mse(w, b)`; the equal-loss lines are the map's own contour levels.
 */
export function LossSurface3D({
  points,
  step,
  levels,
  startW,
  startB,
  formatLoss,
  onStart,
}: {
  points: ReadonlyArray<DescentPoint>;
  step: number;
  levels: ReadonlyArray<number>;
  startW: number;
  startB: number;
  formatLoss: (value: number) => string;
  onStart: (w: number, b: number) => void;
}) {
  const current = points[step];
  const onMap = (point: { w: number; b: number }) =>
    Number.isFinite(point.w) &&
    Number.isFinite(point.b) &&
    point.w >= W_RANGE[0] &&
    point.w <= W_RANGE[1] &&
    point.b >= B_RANGE[0] &&
    point.b <= B_RANGE[1];
  const visible = onMap(current) && Number.isFinite(current.loss) && current.loss <= SURFACE_TOP;
  const xy = (point: { w: number; b: number }) => ({ x: point.w, y: point.b });
  // A run that leaves the map is drawn up to where it leaves.
  const untilOff = (list: ReadonlyArray<DescentPoint>) => {
    const end = list.findIndex((point) => !Number.isFinite(point.w) || !Number.isFinite(point.b));
    return (end === -1 ? list : list.slice(0, end)).map(xy);
  };
  const floor = [-6, 6].map((t) => ({
    x: LEAST_SQUARES.w + t * HESSIAN.shallowAxis[0],
    y: LEAST_SQUARES.b + t * HESSIAN.shallowAxis[1],
  }));
  // Start the valley-floor line where it enters the map.
  const clipToMap = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    let t0 = 0;
    let t1 = 1;
    for (const [a, d, low, high] of [
      [from.x, to.x - from.x, W_RANGE[0], W_RANGE[1]],
      [from.y, to.y - from.y, B_RANGE[0], B_RANGE[1]],
    ] as const) {
      if (Math.abs(d) < 1e-12) continue;
      const enter = ((d > 0 ? low : high) - a) / d;
      const leave = ((d > 0 ? high : low) - a) / d;
      t0 = Math.max(t0, enter);
      t1 = Math.min(t1, leave);
    }
    const at = (t: number) => ({ x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t });
    return [at(t0 + 1e-6), at(t1 - 1e-6)];
  };

  const paths: SurfacePath[] = [
    { id: "floor", points: clipToMap(floor[0], floor[1]), tone: "--ink", width: 1.4, opacity: 0.55, dashed: true },
    { id: "planned", points: untilOff(points), tone: "--ink", width: 1.5, opacity: 0.45 },
    { id: "taken", points: untilOff(points.slice(0, step + 1)), tone: "--loss", width: 3, dots: true },
  ];
  const norm = Math.hypot(current.gw, current.gb);
  const arrows: SurfaceArrow[] =
    visible && norm > 1e-9 && Number.isFinite(norm)
      ? [
          {
            id: "downhill",
            from: xy(current),
            to: { x: current.w - (current.gw / norm) * ARROW_LENGTH, y: current.b - (current.gb / norm) * ARROW_LENGTH },
            tone: "--gradient",
            width: 3.4,
            label: "−∇L",
            labelTone: "is-gradient",
          },
        ]
      : [];
  const surfacePoints: SurfacePoint[] = [
    { id: "minimum", x: LEAST_SQUARES.w, y: LEAST_SQUARES.b, kind: "cross", tone: "--ink", label: `minimum ${formatLoss(LEAST_SQUARES.loss)}`, labelTone: "is-ink" },
    { id: "start", x: startW, y: startB, kind: "ring", tone: "--ink", label: step === 0 ? undefined : "start", labelTone: "is-faint" },
  ];
  if (visible) {
    surfacePoints.push({
      id: "ball",
      x: current.w,
      y: current.b,
      kind: "ball",
      tone: "--loss",
      drop: true,
      label: `L ${formatLoss(current.loss)}`,
      labelTone: "is-loss",
    });
  }
  const showLevel = visible && current.loss - LEAST_SQUARES.loss > 1e-4;

  return (
    <div className="lgd-surface3d">
      <SurfaceStage
        surfaceKey="mse"
        value={mse}
        xRange={W_RANGE}
        yRange={B_RANGE}
        zRange={HEIGHTS}
        levels={levels.filter((level) => level <= SURFACE_TOP)}
        axes={AXES}
        xTicks={W_TICKS}
        yTicks={B_TICKS}
        zTicks={L_TICKS}
        paths={paths}
        arrows={arrows}
        points={surfacePoints}
        rings={showLevel ? [{ id: "level", level: current.loss, tone: "--ink", width: 1.8 }] : []}
        handle={step === 0 ? "ball" : undefined}
        onPlace={onStart}
        hint="click the ground to move the start"
        aspect={0.8}
        view={VIEW}
        label={`The loss drawn as ground over slope w and intercept b: height is the mean squared error, cut off above ${SURFACE_TOP}. It is a long valley, steep across and nearly level along its floor, lowest at w ${LEAST_SQUARES.w.toFixed(
          2,
        )}, b ${LEAST_SQUARES.b.toFixed(2)} where the loss is ${formatLoss(LEAST_SQUARES.loss)}. ${
          visible
            ? `At step ${step} the ball is at w ${current.w.toFixed(2)}, b ${current.b.toFixed(2)}, loss ${formatLoss(current.loss)}.`
            : `At step ${step} the run is off the drawn ground: w ${Number.isFinite(current.w) ? current.w.toFixed(2) : "overflow"}, loss ${formatLoss(current.loss)}.`
        } The coloured line on the ground is the path so far.`}
      />
      {!visible && (
        <p className="lgd-surface3d__off" role="status">
          Off the drawn ground at step {step}: loss {formatLoss(current.loss)}. The surface is cut off above {SURFACE_TOP}.
        </p>
      )}
      <ul className="stage3d-legend" aria-hidden="true">
        <li><i className="is-shade" />higher loss, stronger colour</li>
        <li><i style={key("--loss")} />steps taken</li>
        <li><i style={{ ...key("--ink"), opacity: 0.5 }} />the full run at this η</li>
        <li><i style={key("--gradient")} />−∇L, steepest downhill</li>
        <li><i className="is-dashed" style={key("--ink")} />valley floor</li>
        <li><i style={key("--ink")} />every point with this step&apos;s loss</li>
      </ul>
    </div>
  );
}
