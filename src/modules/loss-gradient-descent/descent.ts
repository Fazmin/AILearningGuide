/**
 * Gradient descent on a real two-parameter loss: the mean squared error of the
 * line ŷ = w·x + b on eight fixed points. Every number the lab shows comes from
 * these functions. Nothing here is a drawn curve.
 */

export const DATA_X = [0.1, 0.4, 0.6, 0.9, 1.1, 1.4, 1.6, 1.9] as const;
export const DATA_Y = [-0.33, -0.2, 0.45, 0.7, 1.27, 1.32, 1.85, 2.15] as const;
const N = DATA_X.length;

/** Plot window for the (w, b) plane. Both axes use the same units per pixel. */
export const W_RANGE = [-1.6, 3.6] as const;
export const B_RANGE = [-2, 2] as const;

export const MAX_STEPS = 80;
/** A run whose loss passes this is reported as diverged and stops updating. */
export const DIVERGED_LOSS = 1e6;

export type Optimizer = "gd" | "momentum";

export interface DescentPoint {
  step: number;
  w: number;
  b: number;
  loss: number;
  /** ∂L/∂w and ∂L/∂b at (w, b). */
  gw: number;
  gb: number;
  /** Length of the move that produced this point (0 at step 0). */
  moved: number;
}

export interface DescentRun {
  points: DescentPoint[];
  /** First step whose loss exceeded DIVERGED_LOSS, or null. */
  divergedAt: number | null;
}

export function predict(w: number, b: number, x: number) {
  return w * x + b;
}

export function residuals(w: number, b: number) {
  return DATA_X.map((x, index) => predict(w, b, x) - DATA_Y[index]);
}

/** L(w, b) = (1/8) Σ (w·xᵢ + b − yᵢ)². */
export function mse(w: number, b: number) {
  let sum = 0;
  for (let index = 0; index < N; index += 1) {
    const r = w * DATA_X[index] + b - DATA_Y[index];
    sum += r * r;
  }
  return sum / N;
}

/** ∂L/∂w = (2/8) Σ rᵢ·xᵢ and ∂L/∂b = (2/8) Σ rᵢ. */
export function gradient(w: number, b: number) {
  let gw = 0;
  let gb = 0;
  for (let index = 0; index < N; index += 1) {
    const r = w * DATA_X[index] + b - DATA_Y[index];
    gw += 2 * r * DATA_X[index];
    gb += 2 * r;
  }
  return { gw: gw / N, gb: gb / N };
}

const meanX = DATA_X.reduce((sum, x) => sum + x, 0) / N;
const meanX2 = DATA_X.reduce((sum, x) => sum + x * x, 0) / N;
const meanY = DATA_Y.reduce((sum, y) => sum + y, 0) / N;
const meanXY = DATA_X.reduce((sum, x, index) => sum + x * DATA_Y[index], 0) / N;

/** The exact minimum, from the normal equations. */
export const LEAST_SQUARES = (() => {
  const w = (meanXY - meanX * meanY) / (meanX2 - meanX * meanX);
  const b = meanY - w * meanX;
  return { w, b, loss: mse(w, b) };
})();

/**
 * The loss is quadratic, so its Hessian is constant:
 * H = 2 [[mean x², mean x], [mean x, 1]]. Its eigenvalues are the curvatures
 * along the steep and shallow axes of the elliptical contours.
 */
export const HESSIAN = (() => {
  const a = 2 * meanX2;
  const c = 2 * meanX;
  const d = 2;
  const trace = a + d;
  const det = a * d - c * c;
  const root = Math.sqrt(trace * trace - 4 * det);
  const steep = (trace + root) / 2;
  const shallow = (trace - root) / 2;
  // Eigenvector for eigenvalue λ of [[a, c], [c, d]] is (c, λ − a).
  const unit = (vx: number, vy: number) => {
    const length = Math.hypot(vx, vy);
    return [vx / length, vy / length] as const;
  };
  return {
    matrix: [
      [a, c],
      [c, d],
    ] as const,
    steep,
    shallow,
    steepAxis: unit(c, steep - a),
    shallowAxis: unit(c, shallow - a),
    condition: steep / shallow,
  };
})();

/** Plain gradient descent diverges once η passes 2/λ_max. */
export const GD_STABLE_LIMIT = 2 / HESSIAN.steep;
/** Past 1/λ_max each plain step overshoots the valley floor along the steep axis. */
export const GD_OVERSHOOT_LIMIT = 1 / HESSIAN.steep;

/** Heavy-ball momentum on a quadratic is stable while η·λ_max < 2(1 + β). */
export function momentumStableLimit(beta: number) {
  return (2 * (1 + beta)) / HESSIAN.steep;
}

/**
 * How the error along one curvature λ evolves per step. For plain descent the
 * error is multiplied by 1 − ηλ. For heavy-ball momentum it follows the
 * eigenvalues of [[1 + β − ηλ, −β], [1, 0]]. `rate` is the magnitude of the
 * dominant factor (below 1 shrinks, above 1 grows); `oscillates` is true when
 * that factor is negative or complex, so the error changes sign as it decays.
 */
export function modeBehaviour(
  learningRate: number,
  curvature: number,
  optimizer: Optimizer,
  beta: number,
) {
  if (optimizer === "gd" || beta <= 0) {
    const factor = 1 - learningRate * curvature;
    return { rate: Math.abs(factor), oscillates: factor < 0 };
  }
  const t = 1 + beta - learningRate * curvature;
  const disc = t * t - 4 * beta;
  if (disc < 0) return { rate: Math.sqrt(beta), oscillates: true };
  const root = Math.sqrt(disc);
  const high = (t + root) / 2;
  const low = (t - root) / 2;
  const dominant = Math.abs(high) >= Math.abs(low) ? high : low;
  return { rate: Math.abs(dominant), oscillates: dominant < 0 };
}

export type Regime = "diverges" | "overshoots" | "one-sided";

export function regime(learningRate: number, optimizer: Optimizer, beta: number): Regime {
  const steep = modeBehaviour(learningRate, HESSIAN.steep, optimizer, beta);
  const shallow = modeBehaviour(learningRate, HESSIAN.shallow, optimizer, beta);
  if (Math.max(steep.rate, shallow.rate) >= 1) return "diverges";
  return steep.oscillates || shallow.oscillates ? "overshoots" : "one-sided";
}

/**
 * Runs the optimizer from (w0, b0). Momentum uses v ← βv + ∇L, θ ← θ − ηv,
 * the same convention as the previous lab, so β = 0 is exactly plain descent.
 */
export function runDescent({
  w0,
  b0,
  learningRate,
  optimizer,
  beta,
  steps = MAX_STEPS,
}: {
  w0: number;
  b0: number;
  learningRate: number;
  optimizer: Optimizer;
  beta: number;
  steps?: number;
}): DescentRun {
  let w = w0;
  let b = b0;
  let vw = 0;
  let vb = 0;
  let divergedAt: number | null = null;
  const first = gradient(w, b);
  const points: DescentPoint[] = [
    { step: 0, w, b, loss: mse(w, b), gw: first.gw, gb: first.gb, moved: 0 },
  ];
  const momentum = optimizer === "momentum" ? beta : 0;

  for (let step = 1; step <= steps; step += 1) {
    const previous = points[step - 1];
    if (divergedAt !== null) {
      points.push({ ...previous, step, moved: 0 });
      continue;
    }
    vw = momentum * vw + previous.gw;
    vb = momentum * vb + previous.gb;
    const nextW = w - learningRate * vw;
    const nextB = b - learningRate * vb;
    const moved = Math.hypot(nextW - w, nextB - b);
    w = nextW;
    b = nextB;
    const loss = mse(w, b);
    const g = gradient(w, b);
    points.push({ step, w, b, loss, gw: g.gw, gb: g.gb, moved });
    if (!Number.isFinite(loss) || loss > DIVERGED_LOSS) divergedAt = step;
  }

  return { points, divergedAt };
}

/**
 * First step after which the loss stays within `tolerance` of the minimum for
 * the rest of the run, or null. A run that dips in and swings back out has not
 * settled, so overshooting optimizers are not credited early.
 */
export function stepsToSettle(run: DescentRun, tolerance = 0.01) {
  const target = LEAST_SQUARES.loss + tolerance;
  let settled: number | null = null;
  for (const point of run.points) {
    if (point.loss <= target) {
      if (settled === null) settled = point.step;
    } else {
      settled = null;
    }
  }
  return run.divergedAt === null ? settled : null;
}

/**
 * Points on the contour L(w, b) = level. Because L is quadratic the contour is
 * an exact ellipse centred on the least-squares minimum:
 * (θ − θ*)ᵀ (H/2) (θ − θ*) = level − L*.
 */
export function contourEllipse(level: number, samples = 96) {
  const excess = level - LEAST_SQUARES.loss;
  if (excess <= 0) return [];
  const along = (curvature: number) => Math.sqrt(excess / (curvature / 2));
  const steepRadius = along(HESSIAN.steep);
  const shallowRadius = along(HESSIAN.shallow);
  const [sx, sy] = HESSIAN.steepAxis;
  const [hx, hy] = HESSIAN.shallowAxis;
  return Array.from({ length: samples + 1 }, (_, index) => {
    const angle = (index / samples) * Math.PI * 2;
    const u = Math.cos(angle) * steepRadius;
    const v = Math.sin(angle) * shallowRadius;
    return {
      w: LEAST_SQUARES.w + u * sx + v * hx,
      b: LEAST_SQUARES.b + u * sy + v * hy,
    };
  });
}
