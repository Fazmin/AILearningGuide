import { useState } from "react";
import { Stage3D } from "./Stage3D";
import type { StageView } from "./runtime";
import type { SurfaceSceneProps } from "./surface-types";

const loadSurface = () => import("./surface-scene");

/** Looking down on the surface from the front left, with the far corner highest on screen. */
export const SURFACE_VIEW: StageView = { azimuth: -32, elevation: 31, distance: 5.5, target: [0, 0.32, 0] };

export interface SurfaceStageProps extends Omit<SurfaceSceneProps, "flat"> {
  /** The picture in words. */
  label: string;
  view?: StageView;
  aspect?: number;
  className?: string;
  /** What a click on the ground does, for the line under the picture. */
  hint?: string;
}

/**
 * A height surface a reader can turn, with a button that presses it flat into its own contour map
 * and lifts it back. The surface, the lines on it and the flat map are one object, so the reader
 * watches each equal-height line keep its place while the ground rises under it.
 */
export function SurfaceStage({ label, view = SURFACE_VIEW, aspect = 0.66, className = "", hint, ...scene }: SurfaceStageProps) {
  const [flat, setFlat] = useState(false);
  return (
    <Stage3D
      scene={loadSurface}
      props={{ ...scene, flat }}
      label={label}
      view={view}
      aspect={aspect}
      hint={hint}
      className={`stage3d--surface ${className}`.trim()}
      actions={
        <button type="button" className="stage3d__button" aria-pressed={flat} onClick={() => setFlat((value) => !value)}>
          {flat ? "Lift it back up" : "Press it flat"}
        </button>
      }
    />
  );
}
