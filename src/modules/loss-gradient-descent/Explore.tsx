import { useId, useMemo, type MouseEvent } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  B_RANGE,
  contourEllipse,
  DATA_X,
  DATA_Y,
  GD_STABLE_LIMIT,
  HESSIAN,
  LEAST_SQUARES,
  MAX_STEPS,
  modeBehaviour,
  momentumStableLimit,
  regime,
  residuals,
  runDescent,
  stepsToSettle,
  W_RANGE,
  type Optimizer,
  type Regime,
} from "./descent";

const LR_MIN = 0.01;
const LR_MAX = 0.5;
const START_W: [number, number] = [-1.5, 3.5];
const START_B: [number, number] = [-1.9, 1.9];
const CONTOUR_LEVELS = [0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32];
const LABELLED_LEVELS = new Set([0.5, 2, 8]);

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) => {
  const value = state[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
};
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const round2 = (value: number) => Math.round(value * 100) / 100;

/** Loss values span five orders of magnitude, so print them with a fixed number of significant digits. */
function formatLoss(value: number) {
  if (!Number.isFinite(value)) return "overflow";
  if (value >= 1000) return value.toExponential(1).replace("e+", "e");
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(3);
  return value.toFixed(4);
}
const signed = (value: number, digits = 3) => {
  const magnitude = Math.abs(value);
  const body =
    magnitude >= 1000 ? magnitude.toExponential(1).replace("e+", "e") : magnitude >= 100 ? magnitude.toFixed(0) : magnitude.toFixed(digits);
  return `${value < 0 ? "−" : ""}${body}`;
};

const REGIME_TEXT: Record<Regime, string> = {
  "one-sided": "one-sided",
  overshoots: "overshoots",
  diverges: "diverges",
};

/* Surface geometry: equal units per pixel on both axes, so a gradient arrow
 * meets the contour lines at a true right angle. */
const UNIT = 80;
const SURFACE = { left: 40, top: 12, right: 12, bottom: 36 };
const SURFACE_W = SURFACE.left + (W_RANGE[1] - W_RANGE[0]) * UNIT + SURFACE.right;
const SURFACE_H = SURFACE.top + (B_RANGE[1] - B_RANGE[0]) * UNIT + SURFACE.bottom;
const sx = (w: number) => SURFACE.left + (w - W_RANGE[0]) * UNIT;
const sy = (b: number) => SURFACE.top + (B_RANGE[1] - b) * UNIT;
const PLOT_RIGHT = sx(W_RANGE[1]);
const PLOT_BOTTOM = sy(B_RANGE[0]);

/* Loss-per-step chart geometry: log-scaled loss from 0.01 to 100. */
const CURVE = { left: 44, right: 448, top: 14, bottom: 186, width: 460, height: 214 };
const LOG_MIN = -2;
const LOG_MAX = 2;
const cx = (step: number) => CURVE.left + (step / MAX_STEPS) * (CURVE.right - CURVE.left);
const cy = (loss: number) => {
  const log = Math.log10(Math.max(1e-6, loss));
  const bounded = clamp(Number.isFinite(log) ? log : LOG_MAX, LOG_MIN, LOG_MAX);
  return CURVE.bottom - ((bounded - LOG_MIN) / (LOG_MAX - LOG_MIN)) * (CURVE.bottom - CURVE.top);
};

/* Data-and-fit chart geometry. */
const FIT = { left: 38, right: 448, top: 12, bottom: 182, width: 460, height: 208 };
const FIT_X: [number, number] = [0, 2];
const FIT_Y: [number, number] = [-1.5, 3];
const fx = (x: number) => FIT.left + ((x - FIT_X[0]) / (FIT_X[1] - FIT_X[0])) * (FIT.right - FIT.left);
const fy = (y: number) =>
  FIT.bottom - ((clamp(y, FIT_Y[0] - 5, FIT_Y[1] + 5) - FIT_Y[0]) / (FIT_Y[1] - FIT_Y[0])) * (FIT.bottom - FIT.top);
const BARS = { top: 8, bottom: 70, height: 88 };

const polyline = (points: ReadonlyArray<{ x: number; y: number }>) =>
  points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

/** Learning-rate intervals that share one regime, found by sweeping the slider's own values. */
function regimeBands(optimizer: Optimizer, beta: number) {
  const bands: { regime: Regime; from: number; to: number }[] = [];
  for (let hundredths = LR_MIN * 100; hundredths <= LR_MAX * 100; hundredths += 1) {
    const lr = hundredths / 100;
    const current = regime(lr, optimizer, beta);
    const last = bands[bands.length - 1];
    if (last && last.regime === current) last.to = lr;
    else bands.push({ regime: current, from: lr, to: lr });
  }
  return bands;
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const ids = useId().replace(/:/g, "");
  const startW = clamp(asNumber(state, "startW", -1), START_W[0], START_W[1]);
  const startB = clamp(asNumber(state, "startB", 1.2), START_B[0], START_B[1]);
  const learningRate = clamp(asNumber(state, "learningRate", 0.1), LR_MIN, LR_MAX);
  const beta = clamp(asNumber(state, "momentum", 0.9), 0, 0.95);
  const optimizer: Optimizer = state.optimizer === "momentum" ? "momentum" : "gd";
  const step = clamp(Math.round(asNumber(state, "step", 0)), 0, MAX_STEPS);

  const run = useMemo(
    () => runDescent({ w0: startW, b0: startB, learningRate, optimizer, beta }),
    [startW, startB, learningRate, optimizer, beta],
  );
  const bands = useMemo(() => regimeBands(optimizer, beta), [optimizer, beta]);
  const contours = useMemo(
    () =>
      CONTOUR_LEVELS.map((level) => ({
        level,
        points: contourEllipse(level).map((point) => ({ x: sx(point.w), y: sy(point.b) })),
      })),
    [],
  );

  const current = run.points[step];
  const currentRegime = regime(learningRate, optimizer, beta);
  const settled = stepsToSettle(run);
  const steepMode = modeBehaviour(learningRate, HESSIAN.steep, optimizer, beta);
  const shallowMode = modeBehaviour(learningRate, HESSIAN.shallow, optimizer, beta);
  const stableLimit = optimizer === "momentum" ? momentumStableLimit(beta) : GD_STABLE_LIMIT;
  const diverged = run.divergedAt !== null && step >= run.divergedAt;
  const gradientNorm = Math.hypot(current.gw, current.gb);
  const offMap =
    current.w < W_RANGE[0] || current.w > W_RANGE[1] || current.b < B_RANGE[0] || current.b > B_RANGE[1];

  const traversed = run.points.slice(0, step + 1);
  const pathPoints = run.points.map((point) => ({ x: sx(point.w), y: sy(point.b) }));

  // −∇L drawn with a fixed on-screen length: direction is the claim, the metrics carry magnitude.
  const arrow =
    gradientNorm > 1e-9 && Number.isFinite(gradientNorm) && !offMap
      ? {
          x1: sx(current.w),
          y1: sy(current.b),
          x2: sx(current.w) + (-current.gw / gradientNorm) * 56,
          y2: sy(current.b) - (-current.gb / gradientNorm) * 56,
        }
      : null;

  const setStart = (w: number, b: number) =>
    setState({
      startW: round2(clamp(w, START_W[0], START_W[1])),
      startB: round2(clamp(b, START_B[0], START_B[1])),
      step: 0,
    });

  const onSurfaceClick = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * SURFACE_W;
    const py = ((event.clientY - rect.top) / rect.height) * SURFACE_H;
    if (px < SURFACE.left || px > PLOT_RIGHT || py < SURFACE.top || py > PLOT_BOTTOM) return;
    const w = W_RANGE[0] + (px - SURFACE.left) / UNIT;
    const b = B_RANGE[1] - (py - SURFACE.top) / UNIT;
    setStart(w, b);
    narrate(`Start moved to w ${w.toFixed(2)}, b ${b.toFixed(2)}. Step reset to 0.`);
  };

  const takeStep = () => {
    const next = Math.min(MAX_STEPS, step + 1);
    setState({ step: next });
    const point = run.points[next];
    narrate(
      `Step ${next}: w ${point.w.toFixed(3)}, b ${point.b.toFixed(3)}, loss ${formatLoss(point.loss)}.`,
    );
  };

  // Loss chart.
  const lossCurve = run.points.map((point) => ({ x: cx(point.step), y: cy(point.loss) }));
  const firstOffChart = run.points.find((point) => !(point.loss <= 10 ** LOG_MAX));

  // Data-and-fit chart.
  const r = residuals(current.w, current.b);
  const squared = r.map((value) => value * value);
  const lineAt = (x: number) => current.w * x + current.b;
  const bestAt = (x: number) => LEAST_SQUARES.w * x + LEAST_SQUARES.b;
  const barMax = Math.max(0.05, ...squared.filter(Number.isFinite));
  const barY = (value: number) =>
    BARS.bottom - (Math.min(value, barMax) / barMax) * (BARS.bottom - BARS.top);

  const regimeSentence =
    currentRegime === "diverges"
      ? `Diverges: η = ${learningRate.toFixed(2)} is past the stability limit ${stableLimit.toFixed(3)}, so each swing across the valley grows.`
      : currentRegime === "overshoots"
        ? optimizer === "momentum"
          ? `Overshoots: velocity carries the point past the valley floor and it swings back while the swings shrink by about ×${Math.max(steepMode.rate, shallowMode.rate).toFixed(2)} per step.`
          : `Overshoots: η is past 1/λ_max = ${(1 / HESSIAN.steep).toFixed(3)}, so each step jumps across the steep direction and lands on the other side.`
        : `One-sided: every step lands on the same side of the valley floor, and the slow shallow direction shrinks by ×${shallowMode.rate.toFixed(3)} per step.`;

  return (
    <div className="lgd-shell">
    <div className="lgd-lab">
      <LabSurface label="Loss surface" className="lgd-surface-card">
        <SurfaceHeading
          kicker="MSE of ŷ = w·x + b on 8 points"
          title="Contours are equal-loss lines; −∇L crosses each one at a right angle"
          aside={<span className="lgd-badge">{REGIME_TEXT[currentRegime]}</span>}
        />
        <svg
          className="lgd-surface"
          viewBox={`0 0 ${SURFACE_W} ${SURFACE_H}`}
          role="img"
          aria-label={`Loss surface over slope w and intercept b. Minimum ${formatLoss(LEAST_SQUARES.loss)} at w ${LEAST_SQUARES.w.toFixed(2)}, b ${LEAST_SQUARES.b.toFixed(2)}. Step ${step}: w ${current.w.toFixed(2)}, b ${current.b.toFixed(2)}, loss ${formatLoss(current.loss)}. Regime ${currentRegime}.`}
          onClick={onSurfaceClick}
        >
          <defs>
            <clipPath id={`${ids}-plot`}>
              <rect x={SURFACE.left} y={SURFACE.top} width={PLOT_RIGHT - SURFACE.left} height={PLOT_BOTTOM - SURFACE.top} />
            </clipPath>
            <marker id={`${ids}-arrow`} markerWidth="8" markerHeight="8" refX="6.5" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" className="lgd-arrowhead" />
            </marker>
          </defs>
          <g clipPath={`url(#${ids}-plot)`}>
            <rect
              className="lgd-band-base"
              x={SURFACE.left}
              y={SURFACE.top}
              width={PLOT_RIGHT - SURFACE.left}
              height={PLOT_BOTTOM - SURFACE.top}
            />
            {[...contours].reverse().map((contour) => (
              <polygon key={`band-${contour.level}`} className="lgd-band" points={polyline(contour.points)} />
            ))}
            {contours.map((contour) => (
              <polygon key={`line-${contour.level}`} className="lgd-contour" points={polyline(contour.points)} />
            ))}
            <polyline className="lgd-path-planned" points={polyline(pathPoints)} />
            <polyline
              className="lgd-path-taken"
              points={polyline(traversed.map((point) => ({ x: sx(point.w), y: sy(point.b) })))}
            />
            {traversed.slice(1, -1).map((point) => (
              <circle key={point.step} className="lgd-path-dot" cx={sx(point.w)} cy={sy(point.b)} r="2.4" />
            ))}
          </g>
          {contours
            .filter((contour) => LABELLED_LEVELS.has(contour.level))
            .map((contour) => {
              const radius = Math.sqrt((contour.level - LEAST_SQUARES.loss) / (HESSIAN.steep / 2));
              const w = LEAST_SQUARES.w + radius * HESSIAN.steepAxis[0];
              const b = LEAST_SQUARES.b + radius * HESSIAN.steepAxis[1];
              if (w > W_RANGE[1] - 0.2 || b > B_RANGE[1] - 0.1) return null;
              return (
                <text key={`label-${contour.level}`} className="lgd-contour-label" x={sx(w) + 4} y={sy(b) - 3}>
                  {contour.level}
                </text>
              );
            })}
          <g className="lgd-minimum" transform={`translate(${sx(LEAST_SQUARES.w)} ${sy(LEAST_SQUARES.b)})`}>
            <path d="M -5 -5 L 5 5 M -5 5 L 5 -5" />
            <text x="8" y="14">minimum {formatLoss(LEAST_SQUARES.loss)}</text>
          </g>
          <circle className="lgd-start" cx={sx(startW)} cy={sy(startB)} r="6" />
          <text className="lgd-start-label" x={sx(startW) + 9} y={sy(startB) - 8}>
            start
          </text>
          {arrow && (
            <g className="lgd-gradient-arrow">
              <line x1={arrow.x1} y1={arrow.y1} x2={arrow.x2} y2={arrow.y2} markerEnd={`url(#${ids}-arrow)`} />
              <text x={arrow.x2 + (arrow.x2 > arrow.x1 ? 5 : -5)} y={arrow.y2 + (arrow.y2 > arrow.y1 ? 12 : -5)} textAnchor={arrow.x2 > arrow.x1 ? "start" : "end"}>
                −∇L
              </text>
            </g>
          )}
          {!offMap && <circle className="lgd-ball" cx={sx(current.w)} cy={sy(current.b)} r="6.5" />}
          {(offMap || diverged) && (
            <text className="lgd-warning" x={SURFACE.left + 10} y={PLOT_BOTTOM - 10}>
              {`Off the map at step ${step}: loss ${formatLoss(current.loss)}`}
            </text>
          )}
          <line className="plot-axis" x1={SURFACE.left} x2={PLOT_RIGHT} y1={PLOT_BOTTOM} y2={PLOT_BOTTOM} />
          <line className="plot-axis" x1={SURFACE.left} x2={SURFACE.left} y1={SURFACE.top} y2={PLOT_BOTTOM} />
          {[-1, 0, 1, 2, 3].map((w) => (
            <text key={`wt-${w}`} className="lgd-tick" x={sx(w)} y={PLOT_BOTTOM + 13} textAnchor="middle">
              {signed(w, 0)}
            </text>
          ))}
          {[-2, -1, 0, 1, 2].map((b) => (
            <text key={`bt-${b}`} className="lgd-tick" x={SURFACE.left - 6} y={sy(b) + 3} textAnchor="end">
              {signed(b, 0)}
            </text>
          ))}
          <text className="lgd-axis-title" x={PLOT_RIGHT} y={SURFACE_H - 4} textAnchor="end">
            slope w →
          </text>
          <text className="lgd-axis-title" x={SURFACE.left + 4} y={SURFACE.top + 2} dominantBaseline="hanging">
            ↑ intercept b
          </text>
        </svg>
        <div className="lgd-legend" aria-hidden="true">
          <span><i className="lgd-key lgd-key--taken" />steps taken</span>
          <span><i className="lgd-key lgd-key--planned" />the full {MAX_STEPS}-step run at this η</span>
          <span><i className="lgd-key lgd-key--arrow" />−∇L, steepest downhill</span>
          <span>contour labels = loss · darker = higher</span>
        </div>
        <div className="metric-row">
          <Metric label="Step" value={`${step}`} />
          <Metric label="Loss L" value={formatLoss(current.loss)} tone="loss" />
          <Metric label="∂L/∂w" value={Number.isFinite(current.gw) ? signed(current.gw) : "overflow"} tone="gradient" />
          <Metric label="∂L/∂b" value={Number.isFinite(current.gb) ? signed(current.gb) : "overflow"} tone="gradient" />
        </div>
      </LabSurface>

      <LabSurface label="Optimizer controls" className="lgd-controls-card">
        <SegmentedControl
          label="Optimizer"
          value={optimizer}
          options={[
            { value: "gd", label: "Plain descent" },
            { value: "momentum", label: "Momentum" },
          ]}
          onChange={(value) => setState({ optimizer: value === "momentum" ? "momentum" : "gd" })}
        />
        <RangeControl
          label="Learning rate"
          min={LR_MIN}
          max={LR_MAX}
          step={0.01}
          value={learningRate}
          format={(value) => `η ${value.toFixed(2)}`}
          onChange={(value) => setState({ learningRate: value })}
        />
        <div className="lgd-ruler" aria-label={`Learning-rate regimes. Stability limit ${stableLimit.toFixed(3)}.`}>
          <div className="lgd-ruler__track">
            {bands.map((band) => (
              <span
                key={`${band.regime}-${band.from}`}
                className={`lgd-ruler__band lgd-ruler__band--${band.regime}`}
                style={{ flexGrow: Math.round((band.to - band.from + 0.01) * 100) }}
              />
            ))}
            <i
              className="lgd-ruler__marker"
              style={{ left: `${((learningRate - LR_MIN) / (LR_MAX - LR_MIN + 0.01)) * 100}%` }}
            />
          </div>
          <div className="lgd-ruler__labels" aria-hidden="true">
            {bands.map((band) => (
              <span
                key={`${band.regime}-${band.from}`}
                className={band.regime === currentRegime ? "is-current" : undefined}
                style={{ flexGrow: Math.round((band.to - band.from + 0.01) * 100) }}
              >
                {band.to - band.from >= 0.07 ? REGIME_TEXT[band.regime] : ""}
              </span>
            ))}
          </div>
          <small>
            {optimizer === "momentum"
              ? `With β = ${beta.toFixed(2)}, stable up to η = ${stableLimit.toFixed(3)} = 2(1+β)/λ_max`
              : `Overshoot above 1/λ_max = ${(1 / HESSIAN.steep).toFixed(3)} · diverges above 2/λ_max = ${GD_STABLE_LIMIT.toFixed(3)}`}
          </small>
        </div>
        {optimizer === "momentum" && (
          <RangeControl
            label="Momentum β"
            min={0}
            max={0.95}
            step={0.05}
            value={beta}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ momentum: value })}
          />
        )}
        <RangeControl
          label="Step"
          min={0}
          max={MAX_STEPS}
          step={1}
          value={step}
          onChange={(value) => setState({ step: value })}
        />
        <div className="lgd-step-actions">
          <button type="button" className="primary-action" onClick={takeStep} disabled={step >= MAX_STEPS}>
            Take one gradient step
          </button>
          <button type="button" className="quiet-action" onClick={() => setState({ step: 0 })}>
            Back to step 0
          </button>
        </div>
        <div className="lgd-start-controls">
          <RangeControl
            label="Start w"
            min={START_W[0]}
            max={START_W[1]}
            step={0.05}
            value={startW}
            format={(value) => signed(value, 2)}
            onChange={(value) => setStart(value, startB)}
          />
          <RangeControl
            label="Start b"
            min={START_B[0]}
            max={START_B[1]}
            step={0.05}
            value={startB}
            format={(value) => signed(value, 2)}
            onChange={(value) => setStart(startW, value)}
          />
        </div>
        <p className="lab-note">Click the surface to start from any point. Moving the start returns to step 0.</p>
      </LabSurface>

      <LabSurface label="Loss per step" className="lgd-curve-card">
        <SurfaceHeading kicker="The same run, one number per step" title="Loss against step, on a log scale" />
        <svg
          className="lgd-curve"
          viewBox={`0 0 ${CURVE.width} ${CURVE.height}`}
          role="img"
          aria-label={`Loss per step. Step ${step} loss ${formatLoss(current.loss)}. ${settled === null ? `Not settled within ${MAX_STEPS} steps.` : `Settles within 0.01 of the minimum by step ${settled}.`}`}
        >
          {[0.01, 0.1, 1, 10, 100].map((tick) => (
            <g key={tick}>
              <line className="lgd-grid" x1={CURVE.left} x2={CURVE.right} y1={cy(tick)} y2={cy(tick)} />
              <text className="lgd-tick" x={CURVE.left - 6} y={cy(tick) + 3} textAnchor="end">
                {tick}
              </text>
            </g>
          ))}
          {[0, 20, 40, 60, 80].map((tick) => (
            <text key={tick} className="lgd-tick" x={cx(tick)} y={CURVE.bottom + 14} textAnchor="middle">
              {tick}
            </text>
          ))}
          <line className="lgd-floor" x1={CURVE.left} x2={CURVE.right} y1={cy(LEAST_SQUARES.loss)} y2={cy(LEAST_SQUARES.loss)} />
          <text className="lgd-floor-label" x={CURVE.left + 6} y={cy(LEAST_SQUARES.loss) - 5}>
            minimum {formatLoss(LEAST_SQUARES.loss)}
          </text>
          <polyline className="lgd-curve-planned" points={polyline(lossCurve)} />
          <polyline className="lgd-curve-taken" points={polyline(lossCurve.slice(0, step + 1))} />
          <line className="lgd-step-marker" x1={cx(step)} x2={cx(step)} y1={CURVE.top} y2={CURVE.bottom} />
          <circle className="lgd-ball" cx={cx(step)} cy={cy(current.loss)} r="5" />
          {firstOffChart && (
            <text className="lgd-warning" x={cx(firstOffChart.step)} y={CURVE.top + 12} textAnchor={firstOffChart.step > 60 ? "end" : "start"}>
              ↑ above 100 from step {firstOffChart.step}
            </text>
          )}
          <line className="plot-axis" x1={CURVE.left} x2={CURVE.right} y1={CURVE.bottom} y2={CURVE.bottom} />
          <line className="plot-axis" x1={CURVE.left} x2={CURVE.left} y1={CURVE.top} y2={CURVE.bottom} />
          <text className="lgd-axis-title" x={CURVE.right} y={CURVE.height - 2} textAnchor="end">
            step →
          </text>
          <text className="lgd-axis-title" x={CURVE.left + 4} y={CURVE.top + 2} dominantBaseline="hanging">
            ↑ loss
          </text>
        </svg>
        <div className="metric-row">
          <Metric label="Regime" value={REGIME_TEXT[currentRegime]} tone={currentRegime === "diverges" ? "loss" : undefined} />
          <Metric label="Settles by" value={settled === null ? `not in ${MAX_STEPS}` : `step ${settled}`} />
          <Metric
            label="Steep axis ×"
            value={
              optimizer === "gd"
                ? signed(1 - learningRate * HESSIAN.steep, 2)
                : `${steepMode.rate.toFixed(2)}${steepMode.oscillates ? " swing" : ""}`
            }
          />
          <Metric label="Shallow axis ×" value={shallowMode.rate.toFixed(3)} />
        </div>
        <p className="lab-note">{regimeSentence}</p>
      </LabSurface>

      <LabSurface label="Data and fit" className="lgd-fit-card">
        <SurfaceHeading
          kicker={`Step ${step}: ŷ = ${signed(current.w, 2)}·x ${current.b < 0 ? "−" : "+"} ${signed(Math.abs(current.b), 2)}`}
          title="Loss is the average squared vertical miss"
        />
        <svg
          className="lgd-fit"
          viewBox={`0 0 ${FIT.width} ${FIT.height}`}
          role="img"
          aria-label={`Eight data points, the current line, and residuals. Mean squared error ${formatLoss(current.loss)}.`}
        >
          <defs>
            <clipPath id={`${ids}-fit`}>
              <rect x={FIT.left} y={FIT.top} width={FIT.right - FIT.left} height={FIT.bottom - FIT.top} />
            </clipPath>
          </defs>
          {[-1, 0, 1, 2, 3].map((tick) => (
            <g key={tick}>
              <line className="lgd-grid" x1={FIT.left} x2={FIT.right} y1={fy(tick)} y2={fy(tick)} />
              <text className="lgd-tick" x={FIT.left - 6} y={fy(tick) + 3} textAnchor="end">
                {signed(tick, 0)}
              </text>
            </g>
          ))}
          {[0, 0.5, 1, 1.5, 2].map((tick) => (
            <text key={tick} className="lgd-tick" x={fx(tick)} y={FIT.bottom + 14} textAnchor="middle">
              {tick}
            </text>
          ))}
          <g clipPath={`url(#${ids}-fit)`}>
            <line className="lgd-best-line" x1={fx(0)} y1={fy(bestAt(0))} x2={fx(2)} y2={fy(bestAt(2))} />
            {Number.isFinite(current.w) && Number.isFinite(current.b) && (
              <line className="lgd-fit-line" x1={fx(0)} y1={fy(lineAt(0))} x2={fx(2)} y2={fy(lineAt(2))} />
            )}
            {DATA_X.map((x, index) => (
              <line
                key={`res-${x}`}
                className="lgd-residual"
                x1={fx(x)}
                x2={fx(x)}
                y1={fy(DATA_Y[index])}
                y2={fy(lineAt(x))}
              />
            ))}
          </g>
          {DATA_X.map((x, index) => (
            <circle key={`pt-${x}`} className="lgd-point" cx={fx(x)} cy={fy(DATA_Y[index])} r="4.5" />
          ))}
          <line className="plot-axis" x1={FIT.left} x2={FIT.right} y1={FIT.bottom} y2={FIT.bottom} />
          <line className="plot-axis" x1={FIT.left} x2={FIT.left} y1={FIT.top} y2={FIT.bottom} />
          <text className="lgd-axis-title" x={FIT.right} y={FIT.height - 2} textAnchor="end">
            x →
          </text>
          <text className="lgd-axis-title" x={FIT.left + 4} y={FIT.top + 2} dominantBaseline="hanging">
            ↑ y
          </text>
        </svg>
        <svg
          className="lgd-bars"
          viewBox={`0 0 ${FIT.width} ${BARS.height}`}
          role="img"
          aria-label={`Squared residuals: ${squared.map((value) => formatLoss(value)).join(", ")}. Their mean is the loss, ${formatLoss(current.loss)}.`}
        >
          {squared.map((value, index) => (
            <g key={`bar-${DATA_X[index]}`}>
              <rect
                className="lgd-bar"
                x={fx(DATA_X[index]) - 9}
                y={barY(value)}
                width="18"
                height={Math.max(0.5, BARS.bottom - barY(value))}
              />
              <text className="lgd-bar-value" x={fx(DATA_X[index])} y={BARS.bottom + 12} textAnchor="middle">
                {value >= 1000 ? value.toExponential(0).replace("e+", "e") : value >= 10 ? value.toFixed(0) : value.toFixed(2)}
              </text>
            </g>
          ))}
          <line className="lgd-mean-line" x1={FIT.left} x2={FIT.right} y1={barY(current.loss)} y2={barY(current.loss)} />
          <line className="plot-axis" x1={FIT.left} x2={FIT.right} y1={BARS.bottom} y2={BARS.bottom} />
        </svg>
        <div className="lgd-legend" aria-hidden="true">
          <span><i className="lgd-key lgd-key--fit" />current line</span>
          <span><i className="lgd-key lgd-key--best" />least-squares line</span>
          <span><i className="lgd-key lgd-key--residual" />residual ŷ − y</span>
          <span><i className="lgd-key lgd-key--bar" />squared residual</span>
          <span><i className="lgd-key lgd-key--mean" />their mean, L = {formatLoss(current.loss)}</span>
        </div>
        <p className="lab-note">
          L = (1/8) Σ (ŷᵢ − yᵢ)². Squared error suits a numeric target like this one. A classifier would
          instead use cross-entropy, −log of the probability it gave the correct class.
        </p>
      </LabSurface>
    </div>
    </div>
  );
}
