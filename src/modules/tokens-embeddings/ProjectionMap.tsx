import { useEffect, useMemo, useRef, useState } from "react";
import type { EmbeddingSpace, ProjectionKind } from "./embeddings";

export type MarkRole = "query" | "neighbour" | "a" | "b" | "c" | "result";

export interface MapMark {
  index: number;
  word: string;
  role: MarkRole;
}

const MARGIN = 26;

/** Track the rendered width so the drawing is in CSS pixels and labels keep their size. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry.contentRect.width);
      if (next > 0) setWidth((current) => (Math.abs(current - next) > 2 ? next : current));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

interface Box {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function boundsOf(points: Float32Array, indices?: readonly number[]): Box {
  const box = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  const visit = (index: number) => {
    const x = points[index * 2];
    const y = points[index * 2 + 1];
    box.minX = Math.min(box.minX, x);
    box.maxX = Math.max(box.maxX, x);
    box.minY = Math.min(box.minY, y);
    box.maxY = Math.max(box.maxY, y);
  };
  if (indices) indices.forEach(visit);
  else for (let index = 0; index < points.length / 2; index += 1) visit(index);
  return box;
}

/** Pad a box and give it a minimum span so a tight cluster is not blown up to fill the frame. */
function padBox(box: Box, full: Box): Box {
  const fullSpan = Math.max(full.maxX - full.minX, full.maxY - full.minY);
  const minimum = fullSpan * 0.14;
  const grow = (low: number, high: number) => {
    const span = Math.max(minimum, (high - low) * 1.25);
    const middle = (low + high) / 2;
    return [middle - span / 2, middle + span / 2] as const;
  };
  const [minX, maxX] = grow(box.minX, box.maxX);
  const [minY, maxY] = grow(box.minY, box.maxY);
  return { minX, maxX, minY, maxY };
}

const ESTIMATED_CHAR = 6.7;

export function ProjectionMap({
  space,
  kind,
  marks,
  frame,
  mode,
  label,
}: {
  space: EmbeddingSpace | null;
  kind: ProjectionKind;
  marks: MapMark[];
  frame: "fit" | "all";
  mode: "neighbours" | "analogy";
  label: string;
}) {
  const [wrapper, measured] = useWidth(640);
  const WIDTH = Math.max(280, measured);
  const HEIGHT = Math.round(Math.min(420, Math.max(260, WIDTH * 0.6)));
  const points = space?.projections[kind] ?? null;
  const full = useMemo(() => (points ? boundsOf(points) : null), [points]);
  const markKey = marks.map((mark) => mark.index).join(",");

  const view = useMemo(() => {
    if (!points || !full) return null;
    const box =
      frame === "all" || marks.length === 0
        ? padBox(full, { minX: 0, maxX: 0, minY: 0, maxY: 0 })
        : padBox(
            boundsOf(
              points,
              marks.map((mark) => mark.index),
            ),
            full,
          );
    // One scale for both axes: a projection's distances are only comparable if x and y share units.
    const scale = Math.min((WIDTH - 2 * MARGIN) / (box.maxX - box.minX), (HEIGHT - 2 * MARGIN) / (box.maxY - box.minY));
    const offsetX = (WIDTH - (box.maxX - box.minX) * scale) / 2;
    const offsetY = (HEIGHT - (box.maxY - box.minY) * scale) / 2;
    const toX = (x: number) => offsetX + (x - box.minX) * scale;
    const toY = (y: number) => HEIGHT - (offsetY + (y - box.minY) * scale);
    return { box, scale, toX, toY };
    // markKey captures the identity of the framed words.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, full, frame, markKey, WIDTH, HEIGHT]);

  const cloud = useMemo(() => {
    if (!points || !view) return { path: "", shown: 0 };
    let path = "";
    let shown = 0;
    for (let index = 0; index < points.length / 2; index += 1) {
      const x = view.toX(points[index * 2]);
      const y = view.toY(points[index * 2 + 1]);
      if (x < -4 || x > WIDTH + 4 || y < -4 || y > HEIGHT + 4) continue;
      path += `M${x.toFixed(1)} ${y.toFixed(1)}h0`;
      shown += 1;
    }
    return { path, shown };
  }, [points, view, WIDTH, HEIGHT]);

  if (!points || !view) {
    return (
      <div className="te-map te-map--empty" aria-hidden="true" ref={wrapper}>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} />
      </div>
    );
  }

  const at = (index: number) => ({ x: view.toX(points[index * 2]), y: view.toY(points[index * 2 + 1]) });
  const query = marks.find((mark) => mark.role === "query");
  const byRole = (role: MarkRole) => marks.find((mark) => mark.role === role);

  // Greedy label placement: nearest free slot around each dot, with a leader line when the
  // label had to move away. Dots are obstacles too, so a label never hides another mark.
  const dots = marks.map((mark) => at(mark.index));
  const placed: Array<{ x: number; y: number; w: number; h: number }> = [];
  const overlaps = (x: number, y: number, w: number, h: number) =>
    placed.some((box) => x < box.x + box.w && x + w > box.x && y < box.y + box.h && y + h > box.y) ||
    dots.some((dot) => dot.x > x - 4 && dot.x < x + w + 4 && dot.y > y - 4 && dot.y < y + h + 4);
  const labels = marks.map((mark) => {
    const point = at(mark.index);
    const width = mark.word.length * ESTIMATED_CHAR + 4;
    const height = 13;
    let choice: { x: number; y: number; leader: boolean } | null = null;
    for (const radius of [8, 18, 30, 44, 60]) {
      for (const angle of [0, 180, -35, 35, 145, -145, -90, 90]) {
        const radians = (angle * Math.PI) / 180;
        const cx = point.x + radius * Math.cos(radians);
        const cy = point.y + radius * Math.sin(radians);
        // Anchor the box so its edge nearest the dot sits at (cx, cy).
        const x = Math.cos(radians) > 0.3 ? cx : Math.cos(radians) < -0.3 ? cx - width : cx - width / 2;
        const y = Math.sin(radians) > 0.3 ? cy : Math.sin(radians) < -0.3 ? cy - height : cy - height / 2;
        if (x < 2 || x + width > WIDTH - 2 || y < 2 || y + height > HEIGHT - 2) continue;
        if (!overlaps(x, y, width, height)) {
          choice = { x, y, leader: radius > 8 };
          break;
        }
      }
      if (choice) break;
    }
    const final = choice ?? { x: Math.min(WIDTH - width - 2, point.x + 8), y: point.y - height / 2, leader: false };
    placed.push({ x: final.x, y: final.y, w: width, h: height });
    const centreX = final.x + width / 2;
    const centreY = final.y + height / 2;
    // Leader line from the dot to the nearest point on the label box.
    const leaderX = Math.min(final.x + width, Math.max(final.x, point.x));
    const leaderY = Math.min(final.y + height, Math.max(final.y, point.y));
    return {
      mark,
      point,
      textX: centreX,
      textY: centreY + 3.8,
      anchor: "middle" as const,
      leader: final.leader ? { x: leaderX, y: leaderY } : null,
    };
  });

  /** End an arrow just short of its target dot so the head stays visible. */
  const shorten = (from: { x: number; y: number }, to: { x: number; y: number }, gap = 8) => {
    const length = Math.hypot(to.x - from.x, to.y - from.y);
    if (length < gap * 2) return to;
    return { x: to.x - ((to.x - from.x) / length) * gap, y: to.y - ((to.y - from.y) / length) * gap };
  };
  const a = byRole("a");
  const b = byRole("b");
  const c = byRole("c");
  const result = byRole("result");
  const ghost =
    mode === "analogy" && a && b && c
      ? (() => {
          const pa = at(a.index);
          const pb = at(b.index);
          const pc = at(c.index);
          return { from: pc, to: { x: pc.x + (pa.x - pb.x), y: pc.y + (pa.y - pb.y) } };
        })()
      : null;
  const axisNames = kind === "umap" ? ["UMAP 1", "UMAP 2"] : ["PC 1", "PC 2"];

  return (
    <div className="te-map" ref={wrapper}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={label}>
        <defs>
          <marker id="te-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className="te-map__arrowhead" />
          </marker>
          <marker id="te-arrow-ghost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className="te-map__arrowhead te-map__arrowhead--ghost" />
          </marker>
        </defs>
        <rect className="te-map__frame" x="0.5" y="0.5" width={WIDTH - 1} height={HEIGHT - 1} rx="10" />
        <path className="te-map__cloud" d={cloud.path} />
        {mode === "neighbours" &&
          query &&
          marks
            .filter((mark) => mark.role === "neighbour")
            .map((mark) => {
              const from = at(query.index);
              const to = at(mark.index);
              return <line key={`spoke-${mark.index}`} className="te-map__spoke" x1={from.x} y1={from.y} x2={to.x} y2={to.y} />;
            })}
        {mode === "analogy" && a && b && (
          <line
            className="te-map__offset"
            markerEnd="url(#te-arrow)"
            x1={at(b.index).x}
            y1={at(b.index).y}
            x2={shorten(at(b.index), at(a.index)).x}
            y2={shorten(at(b.index), at(a.index)).y}
          />
        )}
        {mode === "analogy" && c && result && (
          <line
            className="te-map__offset"
            markerEnd="url(#te-arrow)"
            x1={at(c.index).x}
            y1={at(c.index).y}
            x2={shorten(at(c.index), at(result.index)).x}
            y2={shorten(at(c.index), at(result.index)).y}
          />
        )}
        {ghost && (
          <line
            className="te-map__ghost"
            markerEnd="url(#te-arrow-ghost)"
            x1={ghost.from.x}
            y1={ghost.from.y}
            x2={ghost.to.x}
            y2={ghost.to.y}
          />
        )}
        {labels.map(({ mark, point, leader }) =>
          leader ? (
            <line key={`leader-${mark.index}`} className="te-map__leader" x1={point.x} y1={point.y} x2={leader.x} y2={leader.y} />
          ) : null,
        )}
        {labels.map(({ mark, point }) => (
          <circle
            key={`dot-${mark.index}`}
            className={`te-map__mark te-map__mark--${mark.role}`}
            cx={point.x}
            cy={point.y}
            r={mark.role === "query" || mark.role === "result" ? 5.5 : 4}
          />
        ))}
        {labels.map(({ mark, textX, textY, anchor }) => (
          <text
            key={`label-${mark.index}`}
            className={`te-map__label te-map__label--${mark.role}`}
            x={textX}
            y={textY}
            textAnchor={anchor}
          >
            {mark.word}
          </text>
        ))}
        <text className="te-map__axis" x={MARGIN - 14} y={HEIGHT - 8}>
          {axisNames[0]} →
        </text>
        <text className="te-map__axis" x={12} y={MARGIN - 10}>
          ↑ {axisNames[1]}
        </text>
      </svg>
      <p className="te-map__caption">
        {cloud.shown.toLocaleString("en-US")} of {(points.length / 2).toLocaleString("en-US")} words in view · one scale on both
        axes, no units
      </p>
    </div>
  );
}
