import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { PixelImage, PixelMosaic } from "./PixelImage";
import { rasterizeStrokes, type Point, type Stroke } from "./vae";

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * The decoded latent mosaic with a draggable point. The mosaic is the real
 * decoder evaluated on a grid; the overlay marks the prior's 1 and 2 sigma
 * circles and, when present, an encoded drawing's mean with a 2 sigma ellipse.
 */
export function LatentMap({
  tiles,
  columns,
  minimum,
  maximum,
  point,
  onChange,
  encoded,
  label,
}: {
  tiles: ReadonlyArray<ArrayLike<number>> | null;
  columns: number;
  minimum: number;
  maximum: number;
  point: readonly [number, number];
  onChange: (point: [number, number]) => void;
  encoded: { mean: readonly [number, number]; sigma: readonly [number, number] } | null;
  label: string;
}) {
  const span = maximum - minimum;
  const toX = (z1: number) => ((z1 - minimum) / span) * 100;
  const toY = (z2: number) => ((maximum - z2) / span) * 100;
  const dragging = useRef(false);

  const fromEvent = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const fx = clamp((event.clientX - box.left) / box.width, 0, 1);
    const fy = clamp((event.clientY - box.top) / box.height, 0, 1);
    const round = (value: number) => Math.round(value * 20) / 20;
    onChange([round(minimum + fx * span), round(maximum - fy * span)]);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 0.5 : 0.1;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    const round = (value: number) => Math.round(value * 100) / 100;
    onChange([
      round(clamp(point[0] + move[0], minimum, maximum)),
      round(clamp(point[1] + move[1], minimum, maximum)),
    ]);
  };

  const ticks: number[] = [];
  for (let value = Math.ceil(minimum); value < maximum; value += 2) ticks.push(value);

  return (
    <div className="dv-map">
      <div
        className="dv-map__plot"
        tabIndex={0}
        role="group"
        aria-label={`${label}. Point at z1 ${point[0].toFixed(2)}, z2 ${point[1].toFixed(2)}. Arrow keys move it by 0.1, Shift plus arrow by 0.5.`}
        onKeyDown={onKeyDown}
        onPointerDown={(event) => {
          dragging.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          fromEvent(event);
        }}
        onPointerMove={(event) => {
          if (dragging.current) fromEvent(event);
        }}
        onPointerUp={() => {
          dragging.current = false;
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
      >
        <PixelMosaic tiles={tiles} columns={columns} label={`Decoder output on a ${columns} by ${columns} grid of latents from ${minimum} to ${maximum}`} />
        {!tiles && <span className="dv-map__empty">Decoding the grid…</span>}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <line className="dv-map__axis" x1={toX(0)} x2={toX(0)} y1={0} y2={100} />
          <line className="dv-map__axis" x1={0} x2={100} y1={toY(0)} y2={toY(0)} />
          <ellipse className="dv-map__ring" cx={toX(0)} cy={toY(0)} rx={(1 / span) * 100} ry={(1 / span) * 100} />
          <ellipse className="dv-map__ring dv-map__ring--outer" cx={toX(0)} cy={toY(0)} rx={(2 / span) * 100} ry={(2 / span) * 100} />
          {encoded && (
            <g className="dv-map__encoded">
              <ellipse
                cx={toX(encoded.mean[0])}
                cy={toY(encoded.mean[1])}
                rx={Math.max(0.6, ((2 * encoded.sigma[0]) / span) * 100)}
                ry={Math.max(0.6, ((2 * encoded.sigma[1]) / span) * 100)}
              />
              <path
                d={`M ${toX(encoded.mean[0])} ${toY(encoded.mean[1]) - 2.2} L ${toX(encoded.mean[0]) + 2.2} ${toY(encoded.mean[1])} L ${toX(encoded.mean[0])} ${toY(encoded.mean[1]) + 2.2} L ${toX(encoded.mean[0]) - 2.2} ${toY(encoded.mean[1])} Z`}
              />
            </g>
          )}
          <circle className="dv-map__point-halo" cx={toX(point[0])} cy={toY(point[1])} r={3.2} />
          <circle className="dv-map__point" cx={toX(point[0])} cy={toY(point[1])} r={1.7} />
        </svg>
      </div>
      <div className="dv-map__xticks" aria-hidden="true">
        {ticks.map((value) => (
          <span key={value} style={{ left: `${toX(value)}%` }}>{value}</span>
        ))}
        <b>z₁</b>
      </div>
      <div className="dv-map__yticks" aria-hidden="true">
        {ticks.map((value) => (
          <span key={value} style={{ top: `${toY(value)}%` }}>{value}</span>
        ))}
        <b>z₂</b>
      </div>
    </div>
  );
}

/**
 * A 28 x 28 drawing surface. Strokes are stored in pixel coordinates and
 * rasterized with the same soft brush the presets use.
 */
export function DrawingPad({
  strokes,
  onCommit,
  label,
}: {
  strokes: ReadonlyArray<Stroke>;
  onCommit: (strokes: Stroke[]) => void;
  label: string;
}) {
  const [live, setLive] = useState<Point[] | null>(null);
  const shown = useMemo(
    () => rasterizeStrokes(live ? [...strokes, live] : strokes),
    [strokes, live],
  );

  const pointAt = (event: PointerEvent<HTMLDivElement>): Point => {
    const box = event.currentTarget.getBoundingClientRect();
    return [
      clamp(((event.clientX - box.left) / box.width) * 28, 0, 28),
      clamp(((event.clientY - box.top) / box.height) * 28, 0, 28),
    ];
  };

  return (
    <div
      className="dv-pad"
      role="group"
      aria-label={label}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        setLive([pointAt(event)]);
      }}
      onPointerMove={(event) => {
        if (!live) return;
        const next = pointAt(event);
        const last = live[live.length - 1];
        if (Math.hypot(next[0] - last[0], next[1] - last[1]) < 0.6) return;
        setLive([...live, next]);
      }}
      onPointerUp={() => {
        if (live) onCommit([...strokes, live]);
        setLive(null);
      }}
      onPointerCancel={() => setLive(null)}
    >
      <PixelImage data={shown} size={168} label={`${label}: ${strokes.length} stroke${strokes.length === 1 ? "" : "s"}`} />
    </div>
  );
}
