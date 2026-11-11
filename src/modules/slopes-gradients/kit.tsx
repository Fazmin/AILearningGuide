import { useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";

/** Linear map from a data interval onto a pixel interval. */
export const linear = (d0: number, d1: number, r0: number, r1: number) => (value: number) =>
  r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

const MINUS = "−";

/** A number with fixed decimals and a real minus sign, never "-0.00". */
export function fmt(value: number, digits = 2): string {
  const safe = Number.isFinite(value) ? value : 0;
  const text = Math.abs(safe).toFixed(digits);
  return safe < 0 && Number(text) !== 0 ? `${MINUS}${text}` : text;
}

/** The same, always carrying its sign: +1.50 or −1.50. */
export function signed(value: number, digits = 2): string {
  const text = fmt(value, digits);
  return text.startsWith(MINUS) ? text : `+${text}`;
}

/**
 * A reading the lesson's arithmetic has to agree with: up to four decimals, with trailing zeros
 * dropped down to two. 3 prints as 3.00 and 0.5625 stays 0.5625, so the printed ratio really is the
 * printed change divided by the printed nudge.
 */
export function fmtExact(value: number): string {
  return fmt(value, 4).replace(/(\.\d\d\d*?)0+$/, "$1");
}

/** The same, always carrying its sign. */
export function fmtChange(value: number): string {
  const text = fmtExact(value);
  return text.startsWith(MINUS) ? text : `+${text}`;
}

/** A polyline path through sampled points of a curve. */
export function curvePath(
  from: number,
  to: number,
  samples: number,
  x: (value: number) => number,
  y: (value: number) => number,
  curve: (value: number) => number,
): string {
  let path = "";
  for (let index = 0; index <= samples; index += 1) {
    const value = from + ((to - from) * index) / samples;
    path += `${index === 0 ? "M" : "L"}${x(value).toFixed(1)} ${y(curve(value)).toFixed(1)}`;
  }
  return path;
}

export type Tone = "dial" | "meter" | "slope" | "mid" | "ink" | "accent" | "goal";
const TONES: readonly Tone[] = ["dial", "meter", "slope", "mid", "ink", "accent", "goal"];

/** One arrowhead per tone, so an arrow's head always matches its shaft. */
export function ArrowDefs({ id }: { id: string }) {
  return (
    <defs>
      {TONES.map((tone) => (
        <marker
          key={tone}
          id={`${id}-${tone}`}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="9"
          markerHeight="9"
          markerUnits="userSpaceOnUse"
          orient="auto-start-reverse"
        >
          <path d="M0 0.8 L10 5 L0 9.2 z" className={`sg-head sg-head--${tone}`} />
        </marker>
      ))}
    </defs>
  );
}

/**
 * Pointer dragging on an SVG drawn at its measured width. `onPoint` receives the pointer in viewBox
 * units, wherever the press started; the element keeps the pointer until release.
 */
export function useDrag(viewWidth: number, onPoint: (x: number, y: number) => void) {
  const active = useRef(false);
  const locate = (event: ReactPointerEvent<SVGElement>) => {
    const svg = event.currentTarget.ownerSVGElement ?? (event.currentTarget as unknown as SVGSVGElement);
    const box = svg.getBoundingClientRect();
    if (box.width <= 0) return;
    const ratio = viewWidth / box.width;
    onPoint((event.clientX - box.left) * ratio, (event.clientY - box.top) * ratio);
  };
  return {
    onPointerDown: (event: ReactPointerEvent<SVGElement>) => {
      active.current = true;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      locate(event);
    },
    onPointerMove: (event: ReactPointerEvent<SVGElement>) => {
      if (active.current) locate(event);
    },
    onPointerUp: () => {
      active.current = false;
    },
    onPointerCancel: () => {
      active.current = false;
    },
  };
}

/** Arrow keys for a one-dimensional handle: left and down lower the value, right and up raise it. */
export function sliderKeys(
  value: number,
  low: number,
  high: number,
  step: number,
  onChange: (value: number) => void,
) {
  return (event: KeyboardEvent<SVGElement>) => {
    const moves: Record<string, number> = {
      ArrowLeft: -step,
      ArrowDown: -step,
      ArrowRight: step,
      ArrowUp: step,
      PageDown: -step * 5,
      PageUp: step * 5,
    };
    let next: number | null = null;
    if (event.key in moves) next = value + moves[event.key];
    else if (event.key === "Home") next = low;
    else if (event.key === "End") next = high;
    if (next === null) return;
    event.preventDefault();
    onChange(Number(Math.min(high, Math.max(low, next)).toFixed(2)));
  };
}

/** Snap a dragged value to the slider's grid, inside its range. */
export const snap = (value: number, low: number, high: number, step = 0.05) =>
  Number((Math.round(Math.min(high, Math.max(low, value)) / step) * step).toFixed(2));
