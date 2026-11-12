import { useMemo, type CSSProperties } from "react";
import { SurfaceStage, type SurfaceArrow, type SurfacePath, type SurfacePoint, type SurfaceRing, type SurfaceSlice } from "@app/module-sdk";
import {
  RACE_START,
  RACE_TARGET,
  contourLevels,
  surfaceGradient,
  surfaceValue,
  type Point,
  type Surface,
} from "./curves";
import { fmt, signed, snap } from "./kit";
import type { Legs } from "./state";

/**
 * The same three meters as the contour maps, drawn as the ground they describe: height is the meter
 * reading. One height scale serves all three, so the long valley's walls really are that much steeper
 * than the bowl's.
 */
const DIALS = [0, 6] as const;
const HEIGHTS = [0, 11.25] as const;
const DIAL_TICKS = [0, 1, 2, 3, 4, 5, 6];
const HEIGHT_TICKS = [0, 5, 10];
const AXES = { x: "dial w →", y: "dial b →", z: "meter L" };

/** Scene units of arrow per unit of slope: the same scale as the flat map. */
const ARROW_SCALE = 0.6;

const SURFACE_WORDS: Record<Surface, string> = {
  bowl: "a round bowl whose lowest point is at the centre",
  valley: "a long valley: gentle along w, steep along b, lowest along the middle",
  saddle: "a saddle: it rises to the left and right of the centre and falls in front of and behind it",
};

const CENTRE_NAME: Record<Surface, string> = {
  bowl: "lowest · L 1",
  valley: "lowest · L 1",
  saddle: "flat, not a bottom",
};

/** Card 3 in 3D: the point, the cut each dial makes through the surface, and the legs and arrow lying on it. */
export function GradientSurface({
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
  const value = surfaceValue(surface, w1, w2);
  const gradient = surfaceGradient(surface, w1, w2);
  const steepness = Math.hypot(gradient.x, gradient.y);
  const both = legs === "both";
  const flat = steepness < 0.02;
  const levels = useMemo(() => contourLevels(surface), [surface]);

  const tip = { x: w1 + gradient.x * ARROW_SCALE, y: w2 + gradient.y * ARROW_SCALE };
  const corner = legs === "b" ? w1 : tip.x;
  const slices: SurfaceSlice[] = [];
  if (legs === "w") slices.push({ id: "cut-w", along: "x", at: w2, tone: "--forward" });
  if (legs === "b") slices.push({ id: "cut-b", along: "y", at: w1, tone: "--attention" });
  const arrows: SurfaceArrow[] = [];
  if (legs !== "b" && Math.abs(gradient.x) > 0.02) {
    arrows.push({ id: "leg-w", from: { x: w1, y: w2 }, to: { x: tip.x, y: w2 }, tone: "--forward", label: both ? undefined : signed(gradient.x), labelTone: "is-forward" });
  }
  if (legs !== "w" && Math.abs(gradient.y) > 0.02) {
    arrows.push({ id: "leg-b", from: { x: corner, y: w2 }, to: { x: corner, y: tip.y }, tone: "--attention", label: both ? undefined : signed(gradient.y), labelTone: "is-attention" });
  }
  if (both && !flat) {
    arrows.push({ id: "uphill", from: { x: w1, y: w2 }, to: tip, tone: "--gradient", width: 3.6, label: "uphill", labelTone: "is-gradient" });
    arrows.push({
      id: "downhill",
      from: { x: w1, y: w2 },
      to: { x: w1 - gradient.x * ARROW_SCALE * 0.6, y: w2 - gradient.y * ARROW_SCALE * 0.6 },
      tone: "--ink",
      width: 2,
      dashed: true,
      label: "downhill",
      labelTone: "is-faint",
    });
  }
  const rings: SurfaceRing[] = [{ id: "here", level: value, tone: "--module-accent", width: 2 }];
  const points: SurfacePoint[] = [
    { id: "centre", x: 3, y: 3, kind: "ring", tone: "--ink", label: CENTRE_NAME[surface], labelTone: "is-faint", size: 0.8 },
    { id: "here", x: w1, y: w2, kind: "ball", tone: "--module-accent", drop: true, label: `L = ${fmt(value)}`, labelTone: "is-accent" },
  ];

  return (
    <div className="sg-frame sg-frame--surface">
      <SurfaceStage
        surfaceKey={surface}
        value={(x, y) => surfaceValue(surface, x, y)}
        xRange={DIALS}
        yRange={DIALS}
        zRange={HEIGHTS}
        levels={levels}
        axes={AXES}
        xTicks={DIAL_TICKS}
        yTicks={DIAL_TICKS}
        zTicks={HEIGHT_TICKS}
        points={points}
        arrows={arrows}
        slices={slices}
        rings={rings}
        handle="here"
        hint="click the ground to move the ball"
        onPlace={(x, y) => onMove(snap(x, 0, 6), snap(y, 0, 6))}
        label={`The meter L over two dials drawn as ground: height is the meter reading. The shape is ${SURFACE_WORDS[surface]}. The lines on it are the contour lines of the flat map, each at its own height. The ball is at w = ${fmt(
          w1,
        )}, b = ${fmt(w2)}, where the meter reads ${fmt(value)}. The slope along w is ${signed(gradient.x)} and the slope along b is ${signed(
          gradient.y,
        )}.${
          legs === "w"
            ? " The curtain cuts the ground along dial w with b held still: its top edge is a one-dial curve, and the leg is that curve's slope."
            : legs === "b"
              ? " The curtain cuts the ground along dial b with w held still: its top edge is a one-dial curve, and the leg is that curve's slope."
              : flat
                ? " The ground is level here, so there is no arrow."
                : ` The two legs make the gradient arrow, ${fmt(steepness)} long, pointing straight up the slope.`
        }`}
      />
      <ul className="stage3d-legend" aria-hidden="true">
        <li><i className="is-shade" />higher ground, stronger colour</li>
        <li><i style={{ "--key": "var(--module-accent)" } as CSSProperties} />every setting with this reading</li>
        {legs === "w" && <li><i style={{ "--key": "var(--forward)" } as CSSProperties} />the ground cut along w, b held still</li>}
        {legs === "b" && <li><i style={{ "--key": "var(--attention)" } as CSSProperties} />the ground cut along b, w held still</li>}
        {both && <li><i style={{ "--key": "var(--gradient)" } as CSSProperties} />the gradient: straight uphill</li>}
      </ul>
    </div>
  );
}

/** Card 6 in 3D: the long valley, the goal ring on its floor, this attempt's walk and the one before it. */
export function RaceSurface({ path, ghost, ghostStep }: { path: Point[]; ghost: Point[] | null; ghostStep: number }) {
  const levels = useMemo(() => contourLevels("valley"), []);
  const last = path[path.length - 1];
  const reading = surfaceValue("valley", last.x, last.y);
  const paths: SurfacePath[] = [];
  if (ghost && ghost.length > 1) paths.push({ id: "ghost", points: ghost, tone: "--gradient", width: 2, opacity: 0.7, dashed: true });
  if (path.length > 1) paths.push({ id: "walk", points: path, tone: "--module-accent", width: 3, dots: true });
  const points: SurfacePoint[] = [
    { id: "start", x: RACE_START.x, y: RACE_START.y, kind: "ring", tone: "--ink", label: "start", labelTone: "is-ink" },
    { id: "walker", x: last.x, y: last.y, kind: "ball", tone: "--module-accent", drop: true, label: `L = ${fmt(reading, reading < 2 ? 3 : 2)}`, labelTone: "is-accent" },
  ];
  return (
    <div className="sg-frame sg-frame--surface">
      <SurfaceStage
        surfaceKey="valley"
        value={(x, y) => surfaceValue("valley", x, y)}
        xRange={DIALS}
        yRange={DIALS}
        zRange={HEIGHTS}
        levels={levels}
        axes={AXES}
        xTicks={DIAL_TICKS}
        yTicks={DIAL_TICKS}
        zTicks={HEIGHT_TICKS}
        paths={paths}
        points={points}
        rings={[{ id: "goal", level: RACE_TARGET, tone: "--positive", width: 2.4 }]}
        label={`The long valley drawn as ground: steep walls along dial b, a gentle floor along dial w. The walk starts high on one wall at w = ${fmt(
          RACE_START.x,
        )}, b = ${fmt(RACE_START.y)}. The goal is the small ring on the valley floor, where the meter is under ${fmt(RACE_TARGET)}. After ${
          path.length - 1
        } steps the ball is at w = ${fmt(last.x)}, b = ${fmt(last.y)} and the meter reads ${fmt(reading, 3)}.`}
      />
      <ul className="stage3d-legend" aria-hidden="true">
        <li><i style={{ "--key": "var(--module-accent)" } as CSSProperties} />this try, one dot per step</li>
        {ghost && ghost.length > 1 && (
          <li><i className="is-dashed" style={{ "--key": "var(--gradient)" } as CSSProperties} />last try, step size {fmt(ghostStep)}</li>
        )}
        <li><i style={{ "--key": "var(--positive)" } as CSSProperties} />goal: inside this ring</li>
      </ul>
    </div>
  );
}
