const MINUS = "−";

/** A number with fixed decimals and a real minus sign, never "-0.00". */
export function fmt(value: number, digits = 2): string {
  const safe = Number.isFinite(value) ? value : 0;
  const text = Math.abs(safe).toFixed(digits);
  return safe < 0 && Number(text) !== 0 ? `${MINUS}${text}` : text;
}

/** A whole number with thousands separators: 2,778. */
export const count = (value: number) => Math.round(Number.isFinite(value) ? value : 0).toLocaleString("en-US");

/** A share as a percentage: 0.57 → "57%", 0.001 → "0.1%". */
export function pct(share: number): string {
  const value = (Number.isFinite(share) ? share : 0) * 100;
  if (value === 0) return "0%";
  if (value < 1) return `${+value.toFixed(1)}%`;
  return `${Math.round(value)}%`;
}

/** A multiplier: 1 → "1×", 1000 → "1,000×", 1000000 → "1 million×". */
export function times(value: number): string {
  if (value >= 1e6) return `${+(value / 1e6).toPrecision(3)} million×`;
  return `${count(value)}×`;
}

/** A big number said in words: 1.2 million, 3.4 billion. Below a million it keeps its digits. */
export function said(value: number): string {
  const size = Math.abs(value);
  const trim = (number: number) => String(+number.toPrecision(3));
  if (size >= 1e12) return `${trim(value / 1e12)} trillion`;
  if (size >= 1e9) return `${trim(value / 1e9)} billion`;
  if (size >= 1e6) return `${trim(value / 1e6)} million`;
  return count(value);
}

const SECOND = 1;
const MINUTE = 60;
const HOUR = 3600;
const DAY = 86_400;
const YEAR = 365.25 * DAY;

/** A span of years said in the unit that reads naturally: "3.2 seconds", "9 hours", "10 years". */
export function span(years: number): string {
  const seconds = Math.max(0, Number.isFinite(years) ? years : 0) * YEAR;
  if (seconds === 0) return "none";
  const units: Array<[number, string]> = [
    [YEAR, "year"],
    [DAY, "day"],
    [HOUR, "hour"],
    [MINUTE, "minute"],
    [SECOND, "second"],
  ];
  for (const [size, name] of units) {
    if (seconds >= size) {
      const amount = seconds / size;
      const shown = amount >= 1000 ? said(Math.round(amount)) : amount >= 10 ? String(Math.round(amount)) : String(+amount.toPrecision(2));
      return `${shown} ${name}${shown === "1" ? "" : "s"}`;
    }
  }
  if (seconds >= 1e-3) return `${+(seconds * 1000).toPrecision(2)} milliseconds`;
  return "under a millisecond";
}

/** Minutes of human work said as hours, working days, weeks, months or years (8-hour days, 167-hour months). */
export function workSpan(minutes: number): string {
  const hours = Math.max(0, Number.isFinite(minutes) ? minutes : 0) / 60;
  if (hours < 1) return `${Math.round(hours * 60)} minutes`;
  if (hours < 8) return `${+hours.toPrecision(2)} hours`;
  if (hours < 40) return `${+(hours / 8).toPrecision(2)} working days`;
  if (hours < 167) return `${+(hours / 40).toPrecision(2)} working weeks`;
  if (hours < 2000) return `${+(hours / 167).toPrecision(2)} working months`;
  const years = hours / 2000;
  return `${years >= 1000 ? said(Math.round(years)) : String(+years.toPrecision(2))} working years`;
}

export const SECONDS_PER_YEAR = YEAR;

/** Linear map from a data interval onto a pixel interval. */
export const linear = (d0: number, d1: number, r0: number, r1: number) => (value: number) =>
  r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

/** The same on a log scale. Values at or below zero land on the low end. */
export const logScale = (d0: number, d1: number, r0: number, r1: number) => {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  return (value: number) => r0 + ((Math.log10(Math.max(d0, Math.min(d1, value > 0 ? value : d0))) - l0) / (l1 - l0)) * (r1 - r0);
};

/** A log scale that does not clamp, for paths a clip rectangle will trim. */
export const logRaw = (d0: number, d1: number, r0: number, r1: number) => {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  return (value: number) => r0 + ((Math.log10(Math.max(1e-30, value)) - l0) / (l1 - l0)) * (r1 - r0);
};
