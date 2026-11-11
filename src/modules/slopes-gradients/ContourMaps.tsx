import { useId, useMemo, type KeyboardEvent, type ReactNode } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import {
  RACE_START,
  RACE_TARGET,
  contourLevels,
  contourSegments,
  surfaceGradient,
  surfaceValue,
  type Point,
  type Segment,
  type Surface,
} from "./curves";
import type { Legs } from "./state";
import { ArrowDefs, fmt, linear, signed, snap, useDrag } from "./kit";

const segmentPath = (segments: Segment[], px: (value: number) => number, py: (value: number) => number) =>
  segments
    .map(([x1, y1, x2, y2]) => `M${px(x1).toFixed(1)} ${py(y1).toFixed(1)}L${px(x2).toFixed(1)} ${py(y2).toFixed(1)}`)
    .join("");

/** Where to print each contour's level: the first place a ray from the centre meets it. */
function labelSpot(surface: Surface, level: number): Point | null {
  const angle =
    surface === "bowl" ? -0.62 : surface === "valley" ? -1.12 : level > 5 ? 0.2 : level < 5 ? 1.37 : Math.PI / 4;
  if (surface === "saddle" && level === 5) return { x: 4.75, y: 4.75 };
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  let previous = surfaceValue(surface, 3, 3) - level;
  for (let distance = 0.05; distance < 4.4; distance += 0.05) {
    const x = 3 + dx * distance;
    const y = 3 + dy * distance;
    if (x < 0.35 || x > 5.65 || y < 0.3 || y > 5.7) return null;
    const current = surfaceValue(surface, x, y) - level;
    if (current > 0 !== previous > 0) return { x, y };
    previous = current;
  }
  return null;
}

interface MapLayout {
  size: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
  px: (value: number) => number;
  py: (value: number) => number;
  /** Pixels per dial unit. */
  unit: number;
}

/** A square map drawn at the width it was given, between 250 and 460 pixels. */
function mapLayout(width: number): MapLayout {
  const size = Math.max(250, Math.min(460, width));
  const left = 38;
  const right = size - 10;
  const top = 12;
  const bottom = size - 40;
  return {
    size,
    left,
    right,
    top,
    bottom,
    px: linear(0, 6, left, right),
    py: linear(0, 6, bottom, top),
    unit: (right - left) / 6,
  };
}

/** The shared map: a square of dial settings, contour lines with their levels, and whatever a card draws on top. */
function ContourMap({
  surface,
  layout,
  id,
  label,
  className = "",
  children,
}: {
  surface: Surface;
  layout: MapLayout;
  id: string;
  label: string;
  className?: string;
  children: ReactNode;
}) {
  const { size, left, right, top, bottom, px, py } = layout;
  const contours = useMemo(
    () =>
      contourLevels(surface).map((level) => ({
        level,
        segments: contourSegments((x, y) => surfaceValue(surface, x, y), level),
        spot: labelSpot(surface, level),
      })),
    [surface],
  );
  return (
    <svg className={`sg-plot sg-map ${className}`.trim()} viewBox={`0 0 ${size} ${size}`} style={{ maxWidth: size }} role="img" aria-label={label}>
      <ArrowDefs id={id} />
      <defs>
        <clipPath id={`${id}-map`}>
          <rect x={left} y={top} width={right - left} height={bottom - top} />
        </clipPath>
      </defs>
      <rect className="sg-map__ground" x={left} y={top} width={right - left} height={bottom - top} />
      {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
        <g key={tick}>
          <text className="sg-tick" x={px(tick)} y={bottom + 14} textAnchor="middle">
            {tick}
          </text>
          <text className="sg-tick" x={left - 6} y={py(tick) + 3.5} textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      <text className="sg-axis-label" x={right} y={size - 8} textAnchor="end">
        dial w →
      </text>
      <text className="sg-axis-label" x={12} y={(top + bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(top + bottom) / 2})`}>
        dial b →
      </text>
      <g clipPath={`url(#${id}-map)`}>
        {contours.map((contour) => (
          <path key={contour.level} className="sg-contour" d={segmentPath(contour.segments, px, py)} />
        ))}
      </g>
      {contours.map(
        (contour) =>
          contour.spot && (
            <text key={contour.level} className="sg-contour-label" x={px(contour.spot.x)} y={py(contour.spot.y) + 3.5} textAnchor="middle">
              {contour.level}
            </text>
          ),
      )}
      <rect className="sg-map__frame" x={left} y={top} width={right - left} height={bottom - top} />
      {children}
    </svg>
  );
}

const CENTRE_NAME: Record<Surface, string> = {
  bowl: "lowest · L 1",
  valley: "lowest · L 1",
  saddle: "flat, not a bottom",
};

const SURFACE_WORDS: Record<Surface, string> = {
  bowl: "a round bowl whose lowest point is at the centre",
  valley: "a long valley: gentle along w, steep along b, lowest at the centre",
  saddle: "a saddle: it rises to the left and right of the centre and falls above and below it",
};

/** Map units of arrow per unit of slope. */
const ARROW_SCALE = 0.6;

/**
 * Card 3. The meter over two dials as a contour map, a draggable point, the slope along each dial
 * drawn as a leg, and the two legs put together as the gradient arrow.
 */
export function GradientMap({
  surface,
  w1,
  w2,
  legs,
  onMove,
}: {
  surface: Surface;
  w1: number;
  w2: number;
  legs: Legs;
  onMove: (w1: number, w2: number) => void;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(420);
  const layout = mapLayout(measured);
  const { px, py, unit, size } = layout;
  const drag = useDrag(size, (x, y) =>
    onMove(snap(((x - px(0)) / (px(6) - px(0))) * 6, 0, 6), snap(((py(0) - y) / (py(0) - py(6))) * 6, 0, 6)),
  );
  const value = surfaceValue(surface, w1, w2);
  const gradient = surfaceGradient(surface, w1, w2);
  const steepness = Math.hypot(gradient.x, gradient.y);
  const here = useMemo(
    () => contourSegments((x, y) => surfaceValue(surface, x, y), surfaceValue(surface, w1, w2), 60),
    [surface, w1, w2],
  );
  const showW1 = legs !== "b";
  const showW2 = legs !== "w";
  const both = legs === "both";
  const flat = steepness < 0.02;
  const keys = (event: KeyboardEvent<SVGElement>) => {
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-0.1, 0],
      ArrowRight: [0.1, 0],
      ArrowUp: [0, 0.1],
      ArrowDown: [0, -0.1],
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    onMove(snap(w1 + move[0], 0, 6), snap(w2 + move[1], 0, 6));
  };

  return (
    <div className="sg-frame sg-frame--map" ref={frameRef}>
      <ContourMap
        surface={surface}
        layout={layout}
        id={id}
        className="sg-plot--drag-xy"
        label={`A contour map of the meter L over two dials, ${SURFACE_WORDS[surface]}. Each line joins settings with the same meter reading. The point is at w = ${fmt(
          w1,
        )}, b = ${fmt(w2)}, where the meter reads ${fmt(value)}. The slope along w is ${signed(gradient.x)} and the slope along b is ${signed(
          gradient.y,
        )}.${both ? ` Together they make the gradient arrow, ${fmt(steepness)} long, pointing straight uphill across the contour line.` : ""}`}
      >
        {(() => {
          const tipX = w1 + gradient.x * ARROW_SCALE;
          const tipY = w2 + gradient.y * ARROW_SCALE;
          const cornerX = showW1 ? tipX : w1;
          const normal = flat ? { x: 1, y: 0 } : { x: gradient.x / steepness, y: gradient.y / steepness };
          const along = { x: -normal.y, y: normal.x };
          const mark = 9 / unit;
          const labelRight = w1 < 4.6;
          return (
            <>
              <g clipPath={`url(#${id}-map)`}>
                <path className="sg-contour sg-contour--here" d={segmentPath(here, px, py)} />
                {legs === "w" && <line className="sg-guide" x1={px(0)} x2={px(6)} y1={py(w2)} y2={py(w2)} />}
                {legs === "b" && <line className="sg-guide" x1={px(w1)} x2={px(w1)} y1={py(0)} y2={py(6)} />}
                {both && !flat && (
                  <>
                    <path
                      className="sg-right-angle"
                      d={`M${px(w1 + normal.x * mark)} ${py(w2 + normal.y * mark)}L${px(w1 + (normal.x + along.x) * mark)} ${py(
                        w2 + (normal.y + along.y) * mark,
                      )}L${px(w1 + along.x * mark)} ${py(w2 + along.y * mark)}`}
                    />
                    <line
                      className="sg-line sg-line--ink is-dashed"
                      x1={px(w1)}
                      y1={py(w2)}
                      x2={px(w1 - gradient.x * ARROW_SCALE * 0.6)}
                      y2={py(w2 - gradient.y * ARROW_SCALE * 0.6)}
                      markerEnd={`url(#${id}-ink)`}
                    />
                  </>
                )}
                {showW1 && Math.abs(gradient.x) > 0.02 && (
                  <line
                    className="sg-line sg-line--dial sg-line--thick"
                    x1={px(w1)}
                    y1={py(w2)}
                    x2={px(tipX)}
                    y2={py(w2)}
                    markerEnd={`url(#${id}-dial)`}
                  />
                )}
                {showW2 && Math.abs(gradient.y) > 0.02 && (
                  <line
                    className="sg-line sg-line--mid sg-line--thick"
                    x1={px(cornerX)}
                    y1={py(w2)}
                    x2={px(cornerX)}
                    y2={py(tipY)}
                    markerEnd={`url(#${id}-mid)`}
                  />
                )}
                {both && !flat && (
                  <line
                    className="sg-line sg-line--slope sg-line--arrow"
                    x1={px(w1)}
                    y1={py(w2)}
                    x2={px(tipX)}
                    y2={py(tipY)}
                    markerEnd={`url(#${id}-slope)`}
                  />
                )}
              </g>
              <circle className="sg-dot sg-dot--hollow" cx={px(3)} cy={py(3)} r="3.5" />
              <text className="sg-mark is-faint" x={px(3)} y={py(3) - 9} textAnchor="middle">
                {CENTRE_NAME[surface]}
              </text>
              {showW1 && Math.abs(gradient.x) > 0.02 && (
                <text
                  className="sg-mark is-dial"
                  x={Math.min(size - 46, Math.max(74, px((w1 + tipX) / 2)))}
                  y={py(w2) + (gradient.y >= 0 || !both ? 16 : -8)}
                  textAnchor="middle"
                >
                  {signed(gradient.x)}
                </text>
              )}
              {showW2 && Math.abs(gradient.y) > 0.02 && (
                <text
                  className="sg-mark is-mid"
                  x={px(cornerX) + (gradient.x >= 0 || !both ? 8 : -8)}
                  y={Math.min(size - 50, Math.max(24, py((w2 + tipY) / 2) + 4))}
                  textAnchor={gradient.x >= 0 || !both ? "start" : "end"}
                >
                  {signed(gradient.y)}
                </text>
              )}
              {both && !flat && (
                <text
                  className="sg-mark is-faint"
                  x={Math.min(size - 40, Math.max(64, px(w1 - gradient.x * ARROW_SCALE * 0.6) + (gradient.x >= 0 ? -6 : 6)))}
                  y={Math.min(size - 50, Math.max(24, py(w2 - gradient.y * ARROW_SCALE * 0.6) + (gradient.y >= 0 ? 15 : -7)))}
                  textAnchor={gradient.x >= 0 ? "end" : "start"}
                >
                  downhill
                </text>
              )}
              {both && !flat && (
                <text
                  className="sg-mark is-slope"
                  x={Math.min(size - 40, Math.max(64, px(Math.min(6, Math.max(0, tipX))) + (gradient.x >= 0 ? -6 : 6)))}
                  y={Math.min(size - 50, Math.max(24, py(Math.min(6, Math.max(0, tipY))) + (gradient.y >= 0 ? -9 : 18)))}
                  textAnchor={gradient.x >= 0 ? "end" : "start"}
                >
                  uphill
                </text>
              )}
              <g
                className="sg-knob sg-knob--map"
                tabIndex={0}
                role="application"
                aria-label={`Point on the map at w ${fmt(w1)}, b ${fmt(w2)}. Arrow keys move it.`}
                onKeyDown={keys}
              >
                <circle cx={px(w1)} cy={py(w2)} r="7" />
              </g>
              {!both && (
                <text
                  className="sg-mark is-accent"
                  x={px(w1) + (labelRight ? 11 : -11)}
                  y={py(w2) - 11}
                  textAnchor={labelRight ? "start" : "end"}
                >
                  L = {fmt(value)}
                </text>
              )}
              <rect className="sg-hit" x={px(0)} y={py(6)} width={px(6) - px(0)} height={py(0) - py(6)} {...drag} />
            </>
          );
        })()}
      </ContourMap>
    </div>
  );
}

/**
 * Card 6. The long valley again, a fixed start, the ring the walk has to get inside, the path of the
 * current attempt, and a faint copy of the attempt before it.
 */
export function RaceMap({
  path,
  ghost,
  ghostStep,
}: {
  path: Point[];
  ghost: Point[] | null;
  ghostStep: number;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(420);
  const layout = mapLayout(measured);
  const { px, py, size } = layout;
  const goal = useMemo(() => contourSegments((x, y) => surfaceValue("valley", x, y), RACE_TARGET, 96), []);
  const last = path[path.length - 1];
  const reading = surfaceValue("valley", last.x, last.y);
  const points = (list: Point[], px: (value: number) => number, py: (value: number) => number) =>
    list.map((point) => `${px(point.x).toFixed(1)},${py(point.y).toFixed(1)}`).join(" ");
  return (
    <div className="sg-frame sg-frame--map" ref={frameRef}>
      <ContourMap
        surface="valley"
        layout={layout}
        id={id}
        label={`The long valley as a contour map. The walk starts at w = ${fmt(RACE_START.x)}, b = ${fmt(
          RACE_START.y,
        )}. The goal is the small dashed ring around the lowest point, where the meter is under ${fmt(RACE_TARGET)}. After ${
          path.length - 1
        } steps the walk is at w = ${fmt(last.x)}, b = ${fmt(last.y)} and the meter reads ${fmt(reading, 3)}.`}
      >
        <>
            <g clipPath={`url(#${id}-map)`}>
              <path className="sg-goal-ring" d={segmentPath(goal, px, py)} />
              {ghost && ghost.length > 1 && <polyline className="sg-trail sg-trail--ghost" points={points(ghost, px, py)} />}
              {path.length > 1 && <polyline className="sg-trail" points={points(path, px, py)} />}
              {path.slice(1, -1).map((point, index) => (
                <circle key={index} className="sg-trail-dot" cx={px(point.x)} cy={py(point.y)} r="2.6" />
              ))}
            </g>
            <text className="sg-mark is-goal" x={px(3)} y={py(3) - 16} textAnchor="middle">
              goal: inside this ring
            </text>
            <circle className="sg-dot sg-dot--hollow" cx={px(RACE_START.x)} cy={py(RACE_START.y)} r="5" />
            <text className="sg-mark" x={px(RACE_START.x) + 9} y={py(RACE_START.y) - 9}>
              start
            </text>
            {ghost && ghost.length > 1 && (
              <text className="sg-mark is-faint" x={size - 14} y={24} textAnchor="end">
                dashed: last try, step size {fmt(ghostStep)}
              </text>
            )}
            <circle className="sg-handle" cx={px(last.x)} cy={py(last.y)} r="6.5" />
        </>
      </ContourMap>
    </div>
  );
}
