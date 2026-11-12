import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { SceneFactory, Stage, StageLimits, StageView } from "./runtime";

type Status = "loading" | "ready" | "unavailable";

export interface Stage3DProps<P> {
  /**
   * The scene file, loaded on demand: `() => import("./my-scene")`. Its default export builds the
   * picture. Keeping it behind an import is what keeps three.js out of the app's first download.
   */
  scene: () => Promise<{ default: SceneFactory<P> }>;
  /** What the scene draws. The scene is redrawn whenever this changes. */
  props: P;
  /** The picture in words, for a screen reader and for a device that cannot draw 3D. */
  label: string;
  view: StageView;
  limits?: Partial<StageLimits>;
  /** Height divided by width of the drawing area. */
  aspect?: number;
  className?: string;
  /** Extra controls for the bar under the picture. */
  actions?: ReactNode;
  /** What else the pointer does in this picture, added to the line under it: "click the ground to move the ball". */
  hint?: string;
}

/** jsdom has a canvas element but no drawing behind it, and says so loudly; do not ask it. */
const canDraw = () =>
  typeof window !== "undefined" && typeof navigator !== "undefined" && !/jsdom/i.test(navigator.userAgent);

/**
 * A 3D picture a reader can turn: drag it, or focus it and use the arrow keys. The scene is drawn
 * with three.js, which loads the first time one of these appears. The picture's colours come from
 * the app's CSS variables and are re-read when the theme, contrast or palette changes.
 */
export function Stage3D<P>({ scene, props, label, view, limits, aspect = 0.7, className = "", actions, hint }: Stage3DProps<P>) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const handleRef = useRef<{ update: (props: P) => void; dispose?: () => void } | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    const frame = frameRef.current;
    const canvas = canvasRef.current;
    const labels = labelsRef.current;
    if (!frame || !canvas || !labels || !canDraw()) {
      if (typeof window !== "undefined") setStatus("unavailable");
      return;
    }
    let cancelled = false;
    Promise.all([import("./runtime"), scene()])
      .then(([runtime, sceneModule]) => {
        if (cancelled) return;
        if (!runtime.webglAvailable()) {
          setStatus("unavailable");
          return;
        }
        const build = () => {
          const stage = stageRef.current;
          if (!stage) return;
          handleRef.current?.dispose?.();
          stage.clear();
          handleRef.current = sceneModule.default(stage);
          handleRef.current.update(latest.current);
          stage.invalidate();
        };
        stageRef.current = runtime.createStage(frame, canvas, labels, { view, limits, onTheme: build });
        build();
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("unavailable");
      });
    return () => {
      cancelled = true;
      handleRef.current?.dispose?.();
      handleRef.current = null;
      stageRef.current?.dispose();
      stageRef.current = null;
    };
    // The scene file and the starting view are fixed for the life of the picture.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    handleRef.current?.update(props);
    stageRef.current?.invalidate();
  });

  return (
    <div className={`stage3d ${className}`.trim()} data-status={status}>
      <div className="stage3d__frame" ref={frameRef} style={{ "--stage3d-aspect": String(aspect) } as CSSProperties}>
        <canvas
          ref={canvasRef}
          className="stage3d__canvas"
          role="img"
          aria-label={`${label} A 3D picture: the arrow keys turn it, plus and minus zoom, Home resets the view.`}
          tabIndex={status === "unavailable" ? -1 : 0}
        />
        <div className="stage3d__labels" ref={labelsRef} aria-hidden="true" />
        {status === "loading" && <p className="stage3d__status">Drawing the 3D view…</p>}
        {status === "unavailable" && (
          <p className="stage3d__status is-fallback">
            <b>This device cannot draw the 3D view.</b> {label}
          </p>
        )}
      </div>
      <div className="stage3d__bar">
        <span className="stage3d__hint">Drag to turn{hint ? ` · ${hint}` : ""} · pinch or Ctrl + scroll to zoom</span>
        <span className="stage3d__actions">
          {actions}
          <button
            type="button"
            className="stage3d__button"
            disabled={status !== "ready"}
            onClick={() => stageRef.current?.resetView()}
          >
            Reset view
          </button>
        </span>
      </div>
    </div>
  );
}
