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

/** A loss: three decimals where it is small, two where it is not, and a rounded figure when it has blown up. */
export function fmtLoss(value: number): string {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  if (safe >= 100) return Math.round(safe).toLocaleString("en-US");
  return fmt(safe, safe < 1 ? 3 : 2);
}

/** A whole number with thousands separators: 13,002. */
export const count = (value: number) => Math.round(value).toLocaleString("en-US");

/** A big count in words: 13,002 stays as it is, 4.2 million, 175 billion. */
export function countSaid(value: number): string {
  const size = Math.abs(value);
  const trim = (number: number) => String(+number.toPrecision(3));
  if (size >= 1e12) return `${trim(value / 1e12)} trillion`;
  if (size >= 1e9) return `${trim(value / 1e9)} billion`;
  if (size >= 1e6) return `${trim(value / 1e6)} million`;
  return count(value);
}

/** How long `seconds` is, in the largest unit that reads naturally. */
export function duration(seconds: number): string {
  const units: Array<[number, string]> = [
    [31_557_600, "year"],
    [86_400, "day"],
    [3_600, "hour"],
    [60, "minute"],
  ];
  for (const [size, name] of units) {
    if (seconds >= size) {
      const amount = seconds / size;
      const shown = amount >= 100 ? countSaid(Number(amount.toPrecision(2))) : String(+amount.toPrecision(2));
      return `${shown} ${name}${shown === "1" ? "" : "s"}`;
    }
  }
  return `${Math.round(seconds)} seconds`;
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

/* ------------------------------------------------------------------ */
/* The knob                                                            */
/* ------------------------------------------------------------------ */

/** A knob sweeps 270°: straight up is 0, fully clockwise is +limit, fully anticlockwise is −limit. */
export const KNOB_SWEEP = 135;

/** Degrees clockwise from straight up for a value on a knob that stops at ±limit. */
export const knobAngle = (value: number, limit: number) =>
  Math.max(-KNOB_SWEEP, Math.min(KNOB_SWEEP, ((Number.isFinite(value) ? value : 0) / limit) * KNOB_SWEEP));

/** A point on a circle, `degrees` clockwise from straight up. */
export function onDial(cx: number, cy: number, radius: number, degrees: number): [number, number] {
  const angle = (degrees * Math.PI) / 180;
  return [cx + radius * Math.sin(angle), cy - radius * Math.cos(angle)];
}

/** An arc around a centre from one dial angle to another. */
export function dialArc(cx: number, cy: number, radius: number, from: number, to: number): string {
  const [x0, y0] = onDial(cx, cy, radius, from);
  const [x1, y1] = onDial(cx, cy, radius, to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  const sweep = to > from ? 1 : 0;
  return `M${x0.toFixed(1)} ${y0.toFixed(1)}A${radius} ${radius} 0 ${large} ${sweep} ${x1.toFixed(1)} ${y1.toFixed(1)}`;
}

/** Snap a value to the knob's clicks, inside its stops. */
export const snapKnob = (value: number, limit: number, grid: number) =>
  Number((Math.round(Math.max(-limit, Math.min(limit, value)) / grid) * grid).toFixed(3)) + 0;

export interface KnobProps {
  cx: number;
  cy: number;
  radius: number;
  value: number;
  limit: number;
  /** Short name drawn beside the knob, such as w₁. */
  name: string;
  /** Full name for a screen reader: "Knob w₁, wire from the input to neuron 1". */
  label: string;
  /** Where the name and value go; `textAt` places them exactly instead. */
  side?: "below" | "above" | "left" | "right";
  textAt?: { x: number; y: number; anchor: "start" | "middle" | "end" };
  selected?: boolean;
  /**
   * The loss's slope along this knob. When given, an arrow shows which way the knob should turn to
   * lower the loss (against the slope) and how strongly.
   */
  slope?: number;
  /** The slope that draws a full-length arrow. */
  slopeScale?: number;
  /** Width of the SVG's viewBox, so a drag can be read in its units. */
  viewWidth: number;
  grid?: number;
  onChange?: (value: number) => void;
  onSelect?: () => void;
}

/**
 * A rotary knob drawn in SVG. Drag around it to turn it, or focus it and use the arrow keys. It is a
 * slider to assistive technology, and a knob that only shows a value (no `onChange`) is a plain image.
 */
export function Knob({
  cx,
  cy,
  radius,
  value,
  limit,
  name,
  label,
  side = "below",
  textAt,
  selected = false,
  slope,
  slopeScale = 1,
  viewWidth,
  grid = 0.1,
  onChange,
  onSelect,
}: KnobProps) {
  /** The press in progress: the dial angle it started from, how far it has turned, and the last pointer angle. */
  const drag = useRef<{ from: number; turned: number; last: number | null } | null>(null);
  const angle = knobAngle(value, limit);
  const [px, py] = onDial(cx, cy, radius - 4, angle);
  const [hx, hy] = onDial(cx, cy, radius * 0.32, angle);
  const text =
    textAt ??
    (side === "below"
      ? { x: cx, y: cy + radius + 15, anchor: "middle" as const }
      : side === "above"
        ? { x: cx, y: cy - radius - 8, anchor: "middle" as const }
        : side === "left"
          ? { x: cx - radius - 7, y: cy + 4, anchor: "end" as const }
          : { x: cx + radius + 7, y: cy + 4, anchor: "start" as const });

  /** Degrees clockwise from straight up of the pointer, seen from the knob's centre; null when on the centre. */
  const pointerAngle = (event: ReactPointerEvent<SVGGElement>) => {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return null;
    const box = svg.getBoundingClientRect();
    if (box.width <= 0) return null;
    const ratio = viewWidth / box.width;
    const dx = (event.clientX - box.left) * ratio - cx;
    const dy = (event.clientY - box.top) * ratio - cy;
    return Math.hypot(dx, dy) < 2 ? null : (Math.atan2(dx, -dy) * 180) / Math.PI;
  };

  /**
   * Turning is relative: the knob moves by however far the pointer has travelled around it since the
   * press, so a press alone selects the knob without changing it.
   */
  const turn = (event: ReactPointerEvent<SVGGElement>) => {
    const grip = drag.current;
    if (!grip || !onChange) return;
    const at = pointerAngle(event);
    if (at === null) return;
    if (grip.last !== null) {
      let delta = at - grip.last;
      if (delta > 180) delta -= 360;
      if (delta < -180) delta += 360;
      // Winding past a stop is not saved up: turning back moves the knob at once.
      grip.turned = Math.max(-KNOB_SWEEP - grip.from, Math.min(KNOB_SWEEP - grip.from, grip.turned + delta));
    }
    grip.last = at;
    const next = snapKnob(((grip.from + grip.turned) / KNOB_SWEEP) * limit, limit, grid);
    if (next !== value) onChange(next);
  };

  const keys = (event: KeyboardEvent<SVGGElement>) => {
    if (!onChange) return;
    const moves: Record<string, number> = {
      ArrowLeft: -grid,
      ArrowDown: -grid,
      ArrowRight: grid,
      ArrowUp: grid,
      PageDown: -grid * 5,
      PageUp: grid * 5,
    };
    let next: number | null = null;
    if (event.key in moves) next = value + moves[event.key];
    else if (event.key === "Home") next = -limit;
    else if (event.key === "End") next = limit;
    if (next === null) return;
    event.preventDefault();
    onChange(snapKnob(next, limit, grid));
  };

  // The hint arrow: starts at the pointer and sweeps the way the knob should turn.
  let hint: { path: string; head: string } | null = null;
  if (slope !== undefined && Number.isFinite(slope)) {
    const strength = Math.min(1, Math.abs(slope) / slopeScale);
    if (strength > 0.03) {
      const turn = -Math.sign(slope);
      const sweep = turn * (18 + 66 * strength);
      const ring = radius + 6.5;
      const [ax, ay] = onDial(cx, cy, ring, angle + sweep);
      const [bx, by] = onDial(cx, cy, ring + 4.5, angle + sweep - turn * 13);
      const [dx, dy] = onDial(cx, cy, ring - 4.5, angle + sweep - turn * 13);
      hint = {
        path: dialArc(cx, cy, ring, angle, angle + sweep),
        head: `M${ax.toFixed(1)} ${ay.toFixed(1)}L${bx.toFixed(1)} ${by.toFixed(1)}L${dx.toFixed(1)} ${dy.toFixed(1)}z`,
      };
    }
  }

  const interactive = Boolean(onChange);
  return (
    <g
      className={`nk-knob${selected ? " is-selected" : ""}${interactive ? " is-live" : ""}${value < 0 ? " is-negative" : value > 0 ? " is-positive" : ""}`}
      {...(interactive
        ? {
            role: "slider",
            tabIndex: 0,
            "aria-label": label,
            "aria-valuemin": -limit,
            "aria-valuemax": limit,
            "aria-valuenow": Number(value.toFixed(2)),
            "aria-valuetext": signed(value),
            onKeyDown: keys,
            onFocus: onSelect,
            onPointerDown: (event: ReactPointerEvent<SVGGElement>) => {
              event.currentTarget.setPointerCapture?.(event.pointerId);
              onSelect?.();
              drag.current = { from: knobAngle(value, limit), turned: 0, last: pointerAngle(event) };
            },
            onPointerMove: (event: ReactPointerEvent<SVGGElement>) => {
              turn(event);
            },
            onPointerUp: () => {
              drag.current = null;
            },
            onPointerCancel: () => {
              drag.current = null;
            },
          }
        : { "aria-hidden": true })}
    >
      {/* A generous target: the whole dial and its label respond to a press. */}
      <circle className="nk-knob__hit" cx={cx} cy={cy} r={radius + 9} />
      <path className="nk-knob__track" d={dialArc(cx, cy, radius, -KNOB_SWEEP, KNOB_SWEEP)} />
      {Math.abs(angle) > 0.5 && <path className="nk-knob__fill" d={dialArc(cx, cy, radius, 0, angle)} />}
      <circle className="nk-knob__cap" cx={cx} cy={cy} r={radius - 4} />
      <line className="nk-knob__pointer" x1={hx.toFixed(1)} y1={hy.toFixed(1)} x2={px.toFixed(1)} y2={py.toFixed(1)} />
      {hint && (
        <>
          <path className="nk-knob__hint" d={hint.path} />
          <path className="nk-knob__hint-head" d={hint.head} />
        </>
      )}
      <text className="nk-knob__text" x={text.x} y={text.y} textAnchor={text.anchor}>
        <tspan className="nk-knob__name">{name}</tspan>
        <tspan dx="4">{signed(value)}</tspan>
      </text>
    </g>
  );
}

/** A small dial that only shows a value: for walls of knobs the machine turns. */
export function MiniKnob({ cx, cy, radius, value, soften }: { cx: number; cy: number; radius: number; value: number; soften: number }) {
  // tanh keeps a far-travelled knob on the dial instead of pinned at its stop.
  const angle = KNOB_SWEEP * Math.tanh((Number.isFinite(value) ? value : 0) / soften);
  const [px, py] = onDial(cx, cy, radius - 1.5, angle);
  return (
    <g className={`nk-mini${value < 0 ? " is-negative" : " is-positive"}`}>
      <circle cx={cx} cy={cy} r={radius} />
      <line x1={cx} y1={cy} x2={px.toFixed(1)} y2={py.toFixed(1)} />
    </g>
  );
}
