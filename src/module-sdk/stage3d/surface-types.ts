/** What a lab hands the shared 3D surface picture. Positions are in the lab's own units, never in pixels. */

/** A CSS variable name such as `--loss`, or any CSS colour. */
export type Tone = string;

export interface SurfaceXY {
  x: number;
  y: number;
}

export interface SurfacePath {
  id: string;
  points: ReadonlyArray<SurfaceXY>;
  tone: Tone;
  width?: number;
  opacity?: number;
  dashed?: boolean;
  /** A small dot at every point but the first and last. */
  dots?: boolean;
}

export interface SurfacePoint extends SurfaceXY {
  id: string;
  tone: Tone;
  /** A ball rests on the surface; a ring and a cross lie flat on it. */
  kind: "ball" | "ring" | "cross";
  label?: string;
  /** Extra class names for the label, such as `is-accent`. */
  labelTone?: string;
  /** Pixels to lift the label further, so the labels of points that meet do not land on each other. */
  labelLift?: number;
  /** A dashed line straight down to the floor, to read the two settings under the point. */
  drop?: boolean;
  size?: number;
}

export interface SurfaceArrow {
  id: string;
  from: SurfaceXY;
  to: SurfaceXY;
  tone: Tone;
  width?: number;
  dashed?: boolean;
  label?: string;
  labelTone?: string;
}

/** The surface cut along one setting while the other is held still. */
export interface SurfaceSlice {
  id: string;
  /** The setting that varies along the cut. */
  along: "x" | "y";
  /** The value the other setting is held at. */
  at: number;
  tone: Tone;
}

/** One equal-height line, drawn heavier than the rest. */
export interface SurfaceRing {
  id: string;
  level: number;
  tone: Tone;
  width?: number;
  dashed?: boolean;
}

export interface SurfaceSceneProps {
  /** The surface is rebuilt when this changes; give each formula its own key. */
  surfaceKey: string;
  value: (x: number, y: number) => number;
  xRange: readonly [number, number];
  yRange: readonly [number, number];
  /** Heights: the first sits on the floor, the second at the top of the box. Higher ground is cut off. */
  zRange: readonly [number, number];
  /** Heights that get an equal-height line. */
  levels: ReadonlyArray<number>;
  /** True presses the surface down into its own contour map, seen from above. */
  flat: boolean;
  axes: { x: string; y: string; z: string };
  xTicks: ReadonlyArray<number>;
  yTicks: ReadonlyArray<number>;
  zTicks: ReadonlyArray<number>;
  paths?: ReadonlyArray<SurfacePath>;
  points?: ReadonlyArray<SurfacePoint>;
  arrows?: ReadonlyArray<SurfaceArrow>;
  slices?: ReadonlyArray<SurfaceSlice>;
  rings?: ReadonlyArray<SurfaceRing>;
  /** A press on the surface, or a drag of the handle point, reports the settings under the pointer. */
  onPlace?: (x: number, y: number) => void;
  /** The id of the point a mouse or pen can drag. */
  handle?: string;
}
