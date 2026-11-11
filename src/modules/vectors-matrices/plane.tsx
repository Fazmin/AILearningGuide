import {
  Children,
  useEffect,
  useId,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { apply, type Mat, type Vec, type Word } from "./math";

/** The plane shows −LIMIT to +LIMIT on both axes: room for every preset to move every word without losing one. */
export const LIMIT = 3.2;

export interface Geometry {
  size: number;
  /** Pixels per one unit on the map. */
  unit: number;
  px: (x: number) => number;
  py: (y: number) => number;
  at: (v: Vec) => { x: number; y: number };
  /** Pixel position back to map numbers. */
  fromPixels: (sx: number, sy: number) => Vec;
}

const geometry = (size: number): Geometry => {
  const pad = 6;
  const span = size - pad * 2;
  const unit = span / (LIMIT * 2);
  const px = (x: number) => pad + (x + LIMIT) * unit;
  const py = (y: number) => pad + (LIMIT - y) * unit;
  return {
    size,
    unit,
    px,
    py,
    at: (v) => ({ x: px(v[0]), y: py(v[1]) }),
    fromPixels: (sx, sy) => [(sx - pad) / unit - LIMIT, LIMIT - (sy - pad) / unit],
  };
};

/**
 * The map every card draws on: a square plane at the card's real pixel size, with a faint unit grid,
 * the two axes named in words at both ends, and whatever the card adds on top.
 */
export function Plane({
  label,
  children,
  className = "",
  quiet = false,
}: {
  /** The reading of the picture, for screen readers. */
  label: string;
  children: (geo: Geometry) => ReactNode;
  className?: string;
  /** Fade the resting grid, for cards that draw a moved grid on top. */
  quiet?: boolean;
}) {
  const [frame, measured] = useMeasuredWidth<HTMLDivElement>(410);
  const size = Math.min(540, Math.max(260, measured));
  const geo = geometry(size);
  const clip = useId().replace(/:/g, "");
  const ticks = [-3, -2, -1, 1, 2, 3];
  // The outermost grid lines carry no number: the ends of each axis belong to its name and directions.
  const numbered = (tick: number) => Math.abs(tick) < 3;
  return (
    <div className={`vm-plane ${className}`.trim()} ref={frame}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} role="img" aria-label={label}>
        <defs>
          <clipPath id={clip}>
            <rect x="0" y="0" width={size} height={size} rx="8" />
          </clipPath>
        </defs>
        <g clipPath={`url(#${clip})`}>
          <g className={quiet ? "vm-grid is-quiet" : "vm-grid"}>
            {ticks.map((tick) => (
              <g key={tick}>
                <line x1={geo.px(tick)} x2={geo.px(tick)} y1={0} y2={size} />
                <line x1={0} x2={size} y1={geo.py(tick)} y2={geo.py(tick)} />
              </g>
            ))}
          </g>
          <line className="vm-axis" x1={0} x2={size} y1={geo.py(0)} y2={geo.py(0)} />
          <line className="vm-axis" x1={geo.px(0)} x2={geo.px(0)} y1={0} y2={size} />
          {ticks.map((tick) => (
            <g key={tick} className="vm-tick">
              {numbered(tick) && (
                <text x={geo.px(tick)} y={geo.py(0) + 13} textAnchor="middle">
                  {tick < 0 ? `−${-tick}` : tick}
                </text>
              )}
              {numbered(tick) && (
                <text x={geo.px(0) - 6} y={geo.py(tick) + 3.5} textAnchor="end">
                  {tick < 0 ? `−${-tick}` : tick}
                </text>
              )}
            </g>
          ))}
          <g className="vm-axis-name">
            <text className="vm-axis-name__title" x={size - 8} y={geo.py(0) - 7} textAnchor="end">
              size
            </text>
            <text x={size - 8} y={geo.py(0) + 13.5} textAnchor="end">
              bigger →
            </text>
            <text x={8} y={geo.py(0) + 13.5}>
              ← smaller
            </text>
            <text className="vm-axis-name__title" x={geo.px(0) - 8} y={15} textAnchor="end">
              alive
            </text>
            <text x={geo.px(0) + 8} y={15}>
              ↑ living
            </text>
            <text x={geo.px(0) + 8} y={size - 8}>
              ↓ machine
            </text>
          </g>
          {children(geo)}
        </g>
      </svg>
    </div>
  );
}

/** An arrow from one map point to another, with a head drawn to scale so it never depends on SVG markers. */
export function Arrow({
  geo,
  from = [0, 0],
  to,
  className = "",
  head = 9,
}: {
  geo: Geometry;
  from?: Vec;
  to: Vec;
  className?: string;
  head?: number;
}) {
  const a = geo.at(from);
  const b = geo.at(to);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const span = Math.hypot(dx, dy);
  if (span < 1.5) return <circle className={`vm-arrow__stub ${className}`.trim()} cx={b.x} cy={b.y} r="2.5" />;
  const ux = dx / span;
  const uy = dy / span;
  const size = Math.min(head, span * 0.6);
  const baseX = b.x - ux * size;
  const baseY = b.y - uy * size;
  const half = size * 0.42;
  return (
    <g className={`vm-arrow ${className}`.trim()}>
      <line x1={a.x} y1={a.y} x2={baseX} y2={baseY} />
      <polygon
        points={`${b.x},${b.y} ${baseX - uy * half},${baseY + ux * half} ${baseX + uy * half},${baseY - ux * half}`}
      />
    </g>
  );
}

/** A label on the plane with a halo behind it, so it stays readable over grid lines and arrows. */
export function PlaneLabel({
  x,
  y,
  children,
  className = "",
  anchor = "middle",
  geo,
}: {
  x: number;
  y: number;
  children: ReactNode;
  className?: string;
  anchor?: "start" | "middle" | "end";
  /** Pass the plane's geometry to slide the label sideways until all of it is inside the plane. */
  geo?: Geometry;
}) {
  let fitted = x;
  if (geo) {
    // 11px semi-bold text runs about 6.2px a character: close enough to keep a label from being cut off.
    const width = Children.toArray(children).join("").length * 6.2;
    const left = anchor === "start" ? x : anchor === "end" ? x - width : x - width / 2;
    const kept = Math.min(geo.size - 5 - width, Math.max(5, left));
    fitted = anchor === "start" ? kept : anchor === "end" ? kept + width : kept + width / 2;
  }
  return (
    <text className={`vm-label ${className}`.trim()} x={fitted} y={y} textAnchor={anchor}>
      {children}
    </text>
  );
}

/** Keep a label inside the plane: clamp its anchor point and pick an anchor that grows inward. */
export const labelSpot = (geo: Geometry, x: number, y: number) => {
  const cx = Math.min(geo.size - 6, Math.max(6, x));
  const cy = Math.min(geo.size - 6, Math.max(13, y));
  const anchor: "start" | "middle" | "end" = cx < 44 ? "start" : cx > geo.size - 44 ? "end" : "middle";
  return { x: cx, y: cy, anchor };
};

/** Which way a word left the map, keyed by "sideways,upward" where each is −1, 0 or 1. */
const OFF_ARROWS: Record<string, string> = {
  "1,0": "→",
  "-1,0": "←",
  "0,1": "↑",
  "0,-1": "↓",
  "1,1": "↗",
  "-1,1": "↖",
  "1,-1": "↘",
  "-1,-1": "↙",
};

/**
 * One word on the map: a round dot for a living thing, a square one for a machine (so the two
 * kinds differ by shape, not only by where they sit), with its name beside it.
 */
export function WordMark({
  geo,
  word,
  at,
  tone = "",
  showLabel = true,
  labelSide,
  lift = 0,
  onPick,
}: {
  geo: Geometry;
  word: Word;
  at: Vec;
  /** "is-picked", "is-dim", "is-ghost", "is-flag" or "is-target". */
  tone?: string;
  showLabel?: boolean;
  /** Force the name above or below the dot, when an arrow needs the word's usual side. */
  labelSide?: "above" | "below";
  /** Extra pixels to move the name away from the dot, to clear a drag handle. */
  lift?: number;
  onPick?: () => void;
}) {
  // A word a matrix has thrown off the map is pinned to the nearest edge, hollow, with an arrow saying which way it went.
  const edge = LIMIT - 0.12;
  const sideways = at[0] > LIMIT ? 1 : at[0] < -LIMIT ? -1 : 0;
  const upward = at[1] > LIMIT ? 1 : at[1] < -LIMIT ? -1 : 0;
  const off = sideways !== 0 || upward !== 0;
  const p = geo.at([Math.min(edge, Math.max(-edge, at[0])), Math.min(edge, Math.max(-edge, at[1]))]);
  const away = OFF_ARROWS[`${sideways},${upward}`] ?? "";
  const living = word.y > 0;
  const below = labelSide ? labelSide === "below" : off ? upward > 0 : word.dy > 0;
  const spot = labelSpot(geo, p.x + (labelSide || off ? 0 : word.dx), p.y + (below ? 17 + lift : -9 - lift));
  // A target is drawn larger than a word, so a word that reaches it sits inside the outline.
  const r = tone.includes("is-target") ? 7.5 : 4.2;
  return (
    <g className={`vm-word ${tone} ${off ? "is-off" : ""} ${onPick ? "is-clickable" : ""}`.trim()} onClick={onPick}>
      {onPick && <circle className="vm-word__hit" cx={p.x} cy={p.y} r="13" />}
      {living ? (
        <circle className="vm-word__dot" cx={p.x} cy={p.y} r={r} />
      ) : (
        <rect className="vm-word__dot" x={p.x - r * 0.9} y={p.y - r * 0.9} width={r * 1.8} height={r * 1.8} rx="1" />
      )}
      {showLabel && (
        <text className="vm-word__name" x={spot.x} y={spot.y} textAnchor={spot.anchor}>
          {off ? `${word.id} ${away} off the map` : word.id}
        </text>
      )}
    </g>
  );
}

/** The unit grid after a matrix has moved it: every grid line of the resting map, carried along. */
export function MovedGrid({ geo, m, className = "" }: { geo: Geometry; m: Mat; className?: string }) {
  const reach = 8;
  const lines: ReactNode[] = [];
  for (let k = -reach; k <= reach; k += 1) {
    const a = geo.at(apply(m, [k, -reach]));
    const b = geo.at(apply(m, [k, reach]));
    const c = geo.at(apply(m, [-reach, k]));
    const d = geo.at(apply(m, [reach, k]));
    lines.push(<line key={`v${k}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />);
    lines.push(<line key={`h${k}`} x1={c.x} y1={c.y} x2={d.x} y2={d.y} />);
  }
  return <g className={`vm-moved-grid ${className}`.trim()}>{lines}</g>;
}

/** The square with corners (0,0), (1,0), (1,1), (0,1), after a matrix has moved it. */
export function MovedSquare({ geo, m, className = "" }: { geo: Geometry; m: Mat; className?: string }) {
  const corners: Vec[] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  const points = corners
    .map((corner) => geo.at(apply(m, corner)))
    .map((p) => `${p.x},${p.y}`)
    .join(" ");
  return <polygon className={`vm-square ${className}`.trim()} points={points} />;
}

/** Pointer handlers that turn a drag on an SVG handle into map coordinates. */
export function useTipDrag(geo: Geometry, onMove: (point: Vec) => void) {
  const dragging = useRef(false);
  const read = (event: ReactPointerEvent<SVGElement>): Vec => {
    const svg = event.currentTarget.ownerSVGElement ?? event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const sx = ((event.clientX - rect.left) / Math.max(1, rect.width)) * geo.size;
    const sy = ((event.clientY - rect.top) / Math.max(1, rect.height)) * geo.size;
    return geo.fromPixels(sx, sy);
  };
  return {
    onPointerDown: (event: ReactPointerEvent<SVGElement>) => {
      dragging.current = true;
      // Capture keeps the drag alive when the pointer leaves the small handle. A browser may refuse it
      // (a pointer that has already gone), and the drag still works while the pointer stays on the handle.
      try {
        event.currentTarget.setPointerCapture?.(event.pointerId);
      } catch {
        /* no capture */
      }
      event.preventDefault();
      onMove(read(event));
    },
    onPointerMove: (event: ReactPointerEvent<SVGElement>) => {
      if (dragging.current) onMove(read(event));
    },
    onPointerUp: (event: ReactPointerEvent<SVGElement>) => {
      dragging.current = false;
      try {
        event.currentTarget.releasePointerCapture?.(event.pointerId);
      } catch {
        /* nothing was captured */
      }
    },
    onPointerCancel: () => {
      dragging.current = false;
    },
  };
}

/** The round handle on an arrow tip that the reader drags. */
export function DragHandle({
  geo,
  at,
  onMove,
  title,
}: {
  geo: Geometry;
  at: Vec;
  onMove: (point: Vec) => void;
  title: string;
}) {
  const handlers = useTipDrag(geo, onMove);
  const p = geo.at(at);
  return (
    <g className="vm-handle">
      <circle className="vm-handle__ring" cx={p.x} cy={p.y} r="8" />
      <circle className="vm-handle__hit" cx={p.x} cy={p.y} r="16" {...handlers}>
        <title>{title}</title>
      </circle>
    </g>
  );
}

/**
 * Glide a matrix toward its target when it changes by a jump (a preset, a swap, a stage), so the
 * reader sees the map travel from A to B. Small slider steps apply at once, and reduced motion
 * turns the glide off.
 */
export function useGlide(target: Mat): Mat {
  const [shown, setShown] = useState<Mat>(target);
  const current = useRef<Mat>(target);
  const [a, b, c, d] = target;
  useEffect(() => {
    const goal: Mat = [a, b, c, d];
    const from = current.current;
    const jump = Math.max(...goal.map((value, index) => Math.abs(value - from[index])));
    const reduced = typeof document !== "undefined" && document.documentElement.dataset.motion === "reduced";
    if (jump < 0.15 || reduced || typeof requestAnimationFrame === "undefined") {
      current.current = goal;
      setShown(goal);
      return undefined;
    }
    let frame = 0;
    const started = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / 420);
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      const next = goal.map((value, index) => from[index] + (value - from[index]) * eased) as unknown as Mat;
      current.current = t >= 1 ? goal : next;
      setShown(current.current);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [a, b, c, d]);
  return shown;
}
