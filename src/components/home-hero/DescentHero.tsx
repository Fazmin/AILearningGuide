import { useEffect, useRef, useState } from "react";
import { clamp, tween, useHeroScript, wait, type HeroVisualProps } from "./hero-motion";

/** A lopsided valley: steeper on the right, lowest at x = 0.6, loss 0.07. */
const loss = (x: number) => 0.07 + 1.6 * (x - 0.6) ** 2 + 1.5 * Math.max(0, x - 0.6) ** 3;
const slope = (x: number) => 3.2 * (x - 0.6) + 4.5 * Math.max(0, x - 0.6) ** 2;
const LEARNING_RATE = 0.15;
/** Stop once a step would move less than this. */
const SETTLED_STEP = 0.0035;
const MAX_STEPS = 12;
const TOP_LOSS = 0.7;
const starts = [0.05, 0.95, 0.15, 0.9];
const arrowId = "home-hero-descent-arrow";

interface View {
  x: number;
  trail: number[];
  /** Where the slope arrow is drawn, and whether it is showing. */
  tangent: number | null;
  tangentOn: boolean;
  hop: { from: number; to: number; t: number } | null;
  step: number;
  done: boolean;
  fading: boolean;
  /** Bumped on each arrival so the arrival ring replays. */
  pulse: number;
}

const fresh = (x: number): View => ({ x, trail: [], tangent: null, tangentOn: false, hop: null, step: 0, done: false, fading: false, pulse: 0 });

/** The finished run from `start`, with the first step's slope still drawn. */
function finishedRun(start: number): View {
  const view = fresh(start);
  while (view.step < MAX_STEPS) {
    const dx = -LEARNING_RATE * slope(view.x);
    if (Math.abs(dx) < SETTLED_STEP) break;
    view.trail.push(view.x);
    view.x = clamp(view.x + dx, 0.01, 0.99);
    view.step += 1;
  }
  return { ...view, done: true, tangent: view.trail[0] ?? null, tangentOn: view.trail.length > 0 };
}

const still = finishedRun(starts[0]);

export function DescentHero({ size, animate }: HeroVisualProps) {
  const [view, setView] = useState<View>(still);
  const [restartKey, setRestartKey] = useState(0);
  const nextStart = useRef<number | null>(null);

  useEffect(() => {
    if (!animate) setView(still);
  }, [animate]);

  useHeroScript(animate, async (live) => {
    for (let run = 0; ; run += 1) {
      const model = fresh(nextStart.current ?? starts[run % starts.length]);
      nextStart.current = null;
      const show = () => setView({ ...model, trail: [...model.trail] });
      show();
      await wait(550, live);
      while (model.step < MAX_STEPS) {
        const dx = -LEARNING_RATE * slope(model.x);
        if (Math.abs(dx) < SETTLED_STEP) break;
        model.tangent = model.x;
        model.tangentOn = true;
        show();
        await wait(430, live);
        const from = model.x;
        const to = clamp(from + dx, 0.01, 0.99);
        model.trail.push(from);
        model.tangentOn = false;
        await tween(360 + 280 * Math.min(1, Math.abs(to - from) / 0.25), live, (t) => {
          model.hop = { from, to, t };
          show();
        });
        model.hop = null;
        model.x = to;
        model.step += 1;
        show();
        await wait(170, live);
      }
      model.done = true;
      model.pulse += 1;
      show();
      await wait(1900, live);
      model.fading = true;
      show();
      await wait(450, live);
    }
  }, restartKey);

  const left = 34;
  const top = Math.max(118, size.height * 0.27);
  const plotWidth = size.width - left * 2;
  const plotHeight = size.height - top - Math.max(62, size.height * 0.15);
  const floor = top + plotHeight;
  const px = (x: number) => left + x * plotWidth;
  const py = (value: number) => top + (1 - value / TOP_LOSS) * plotHeight;
  const on = (x: number) => [px(x), py(loss(x))] as const;

  let curve = "";
  for (let i = 0; i <= 160; i += 1) {
    const [x, y] = on(i / 160);
    curve += `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }

  let [dotX, dotY] = on(view.x);
  if (view.hop) {
    const { from, to, t } = view.hop;
    const [ax, ay] = on(from);
    const [bx, by] = on(to);
    dotX = ax + (bx - ax) * t;
    dotY = ay + (by - ay) * t - (10 + 0.22 * Math.abs(bx - ax)) * 4 * t * (1 - t);
  }

  let tangent = null;
  if (view.tangent !== null) {
    const g = slope(view.tangent);
    const [tx, ty] = on(view.tangent);
    // The slope in screen pixels, as a unit vector pointing right.
    const rise = -(g * (plotHeight / TOP_LOSS)) / plotWidth;
    const norm = Math.hypot(1, rise);
    const [ux, uy] = [1 / norm, rise / norm];
    const downhill = g > 0 ? -1 : 1;
    const length = 36 + 64 * Math.min(1, Math.abs(g) / 1.8);
    tangent = { tx, ty, ux, uy, downhill, length };
  }

  const restartAt = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = clamp((event.clientX - rect.left - left) / plotWidth, 0.02, 0.98);
    if (animate) {
      nextStart.current = x;
      setRestartKey((key) => key + 1);
    } else {
      setView(finishedRun(x));
    }
  };

  return (
    <>
      <svg
        className={`hero-descent${animate ? "" : " is-still"}`}
        viewBox={`0 0 ${size.width} ${size.height}`}
        aria-hidden="true"
        onClick={restartAt}
      >
        <defs>
          <marker id={arrowId} viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto">
            <path className="hero-descent__arrowhead" d="M0,0.5 L9.5,5 L0,9.5 Z" />
          </marker>
        </defs>
        <path className="hero-descent__area" d={`${curve}L${px(1)},${floor}L${px(0)},${floor}Z`} />
        <path className="hero-descent__curve" d={curve} />
        <line className="hero-descent__floor" x1={px(0)} x2={px(1)} y1={floor} y2={floor} />
        <text className="hero-descent__axis" x={px(0)} y={py(loss(0)) - 14}>loss</text>
        <text className="hero-descent__axis" x={px(0)} y={floor + 20}>weight →</text>
        <g className={`hero-descent__run${view.fading ? " is-fading" : ""}`}>
          {view.trail.map((x, index) => {
            const [cx, cy] = on(x);
            return <circle key={index} className="hero-descent__ghost" cx={cx} cy={cy} r={4} />;
          })}
          {tangent && (
            <g className={`hero-descent__slope${view.tangentOn ? "" : " is-off"}`}>
              <line
                className="hero-descent__tangent"
                x1={tangent.tx - tangent.downhill * tangent.ux * tangent.length * 0.5}
                y1={tangent.ty - tangent.downhill * tangent.uy * tangent.length * 0.5}
                x2={tangent.tx + tangent.downhill * tangent.ux * tangent.length * 1.1}
                y2={tangent.ty + tangent.downhill * tangent.uy * tangent.length * 1.1}
              />
              <line
                className="hero-descent__arrow"
                markerEnd={`url(#${arrowId})`}
                x1={tangent.tx}
                y1={tangent.ty}
                x2={tangent.tx + tangent.downhill * tangent.ux * tangent.length}
                y2={tangent.ty + tangent.downhill * tangent.uy * tangent.length}
              />
            </g>
          )}
          {view.done && <circle className="hero-descent__ring" cx={dotX} cy={dotY} r={13} />}
          {view.done && view.pulse > 0 && <circle key={view.pulse} className="hero-descent__pulse" cx={dotX} cy={dotY} r={8} />}
          <circle className="hero-descent__dot" cx={dotX} cy={dotY} r={7} />
        </g>
      </svg>
      <div className={`hero-descent__readout${view.done ? " is-done" : ""}`}>
        <div><span>loss</span><b className="hero-descent__loss">{loss(view.x).toFixed(3)}</b></div>
        <div><span>step</span><b>{String(view.step).padStart(2, "0")}</b></div>
        <span className="hero-descent__state">converged</span>
      </div>
    </>
  );
}
