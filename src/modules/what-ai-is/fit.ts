/** The three-point line fit behind the Learning problem card. Every number on that card comes from here. */

export const FIT_POINTS = [
  { x: 0.5, y: 1.4 },
  { x: 1.5, y: 2.1 },
  { x: 2.5, y: 3.2 },
] as const;

export const STEP_SIZE = 0.12;
export const W_RANGE = [0, 2] as const;
export const B_RANGE = [0, 2.4] as const;

const n = FIT_POINTS.length;
const sumX = FIT_POINTS.reduce((sum, point) => sum + point.x, 0);
const sumY = FIT_POINTS.reduce((sum, point) => sum + point.y, 0);
const sumXX = FIT_POINTS.reduce((sum, point) => sum + point.x * point.x, 0);
const sumXY = FIT_POINTS.reduce((sum, point) => sum + point.x * point.y, 0);

export const residuals = (w: number, b: number) =>
  FIT_POINTS.map((point) => {
    const prediction = w * point.x + b;
    const residual = prediction - point.y;
    return { ...point, prediction, residual, squared: residual * residual };
  });

/** L(w, b) = (1/n) Σ (w x + b − y)². */
export const mse = (w: number, b: number) => residuals(w, b).reduce((sum, row) => sum + row.squared, 0) / n;

/** ∂L/∂w = (2/n) Σ (ŷ − y) x and ∂L/∂b = (2/n) Σ (ŷ − y). */
export const gradient = (w: number, b: number) => {
  const rows = residuals(w, b);
  return {
    w: rows.reduce((sum, row) => sum + 2 * row.residual * row.x, 0) / n,
    b: rows.reduce((sum, row) => sum + 2 * row.residual, 0) / n,
  };
};

const clip = (value: number, [low, high]: readonly [number, number]) => Math.min(high, Math.max(low, value));

/** One downhill step, clipped to the slider ranges. */
export const downhill = (w: number, b: number, rate = STEP_SIZE) => {
  const g = gradient(w, b);
  return { w: clip(w - rate * g.w, W_RANGE), b: clip(b - rate * g.b, B_RANGE) };
};

/** The least-squares line from the normal equations: the bottom of the bowl. */
export const leastSquares = () => {
  const w = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const b = (sumY - w * sumX) / n;
  return { w, b, loss: mse(w, b) };
};

/**
 * L is exactly quadratic in (w, b): L = L* + dᵀ A d with d = θ − θ* and A = (1/n)[[Σx², Σx], [Σx, n]].
 * Each level set L = c is therefore an ellipse; this samples it.
 */
export const contour = (level: number, samples = 96) => {
  const best = leastSquares();
  const excess = level - best.loss;
  if (excess <= 0) return [];
  const a = sumXX / n;
  const c = sumX / n;
  const d = 1;
  const mean = (a + d) / 2;
  const spread = Math.sqrt(((a - d) / 2) ** 2 + c * c);
  const lambda1 = mean + spread;
  const lambda2 = mean - spread;
  const angle = 0.5 * Math.atan2(2 * c, a - d);
  const v1 = { w: Math.cos(angle), b: Math.sin(angle) };
  const v2 = { w: -Math.sin(angle), b: Math.cos(angle) };
  const r1 = Math.sqrt(excess / lambda1);
  const r2 = Math.sqrt(excess / lambda2);
  return Array.from({ length: samples + 1 }, (_, index) => {
    const t = (2 * Math.PI * index) / samples;
    return {
      w: best.w + r1 * Math.cos(t) * v1.w + r2 * Math.sin(t) * v2.w,
      b: best.b + r1 * Math.cos(t) * v1.b + r2 * Math.sin(t) * v2.b,
    };
  });
};
