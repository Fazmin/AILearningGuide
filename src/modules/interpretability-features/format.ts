/** Display helpers shared by the lab's cards. */

export const pct = (value: number | null, digits = 0) => (value === null ? "—" : `${(value * 100).toFixed(digits)}%`);

/** A probability as a percentage that never rounds a nonzero value to a misleading 0.0%. */
export function formatProbability(value: number) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 0.001) return `${(value * 100).toFixed(1)}%`;
  if (value >= 1e-6) return "<0.1%";
  return "~0%";
}

/** A change in log probability (nats) with an explicit sign. */
export function formatChange(value: number) {
  if (!Number.isFinite(value)) return "—";
  const rounded = Math.round(value * 100) / 100;
  return `${rounded >= 0 ? "+" : "−"}${Math.abs(rounded).toFixed(2)}`;
}

/** A count of standard deviations with an explicit sign and one decimal. */
export function formatScore(value: number) {
  if (!Number.isFinite(value)) return "—";
  const rounded = Math.round(value * 10) / 10;
  return `${rounded >= 0 ? "+" : "−"}${Math.abs(rounded).toFixed(1)}`;
}
