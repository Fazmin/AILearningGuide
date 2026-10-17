import { useEffect, useMemo, useRef, useState } from "react";
import { clamp, easeInOut, linear, tween, useHeroScript, wait, type HeroVisualProps, type StageSize } from "./hero-motion";
import { meaningMapWords, type MeaningWord } from "./meaning-map-data";

interface MapPoint extends MeaningWord {
  /** Target position, 0 to 1, after each cluster is spread around its centre. */
  tx: number;
  ty: number;
}

/** Clusters from t-SNE are tight; spreading them leaves room for labels without moving any cluster. */
const SPREAD = 2.35;
const PAD = { left: 30, top: 92, right: 92, bottom: 70 };
const LABEL_LINE = 15;
const anchors = new Set(["cat", "red", "paris", "bread", "two"]);
const tour = ["cat", "purple", "paris", "cow", "bread", "seven"];

const points: MapPoint[] = (() => {
  const centres = new Map<string, { x: number; y: number; n: number }>();
  for (const word of meaningMapWords) {
    const centre = centres.get(word.group) ?? { x: 0, y: 0, n: 0 };
    centre.x += word.x;
    centre.y += word.y;
    centre.n += 1;
    centres.set(word.group, centre);
  }
  const spread = meaningMapWords.map((word) => {
    const centre = centres.get(word.group)!;
    const cx = centre.x / centre.n;
    const cy = centre.y / centre.n;
    return { ...word, tx: cx + (word.x - cx) * SPREAD, ty: cy + (word.y - cy) * SPREAD };
  });
  const xs = spread.map((point) => point.tx);
  const ys = spread.map((point) => point.ty);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return spread.map((point) => ({
    ...point,
    tx: (point.tx - minX) / (maxX - minX),
    ty: (point.ty - minY) / (maxY - minY),
  }));
})();
const byWord = new Map(points.map((point) => [point.word, point]));
const indexOf = new Map(points.map((point, index) => [point.word, index]));

interface Position {
  u: number;
  v: number;
}

/** Final positions for one stage size: labelled words are nudged apart vertically until no two labels overlap. */
function settledPositions({ width, height }: StageSize): Position[] {
  const plotWidth = width - PAD.left - PAD.right;
  const plotHeight = height - PAD.top - PAD.bottom;
  const placed = points.map((point) => ({
    point,
    x: PAD.left + point.tx * plotWidth,
    y: PAD.top + point.ty * plotHeight,
    label: point.word.length * 6.8 + 12,
  }));
  const labelled = placed.filter((entry) => entry.point.neighbours);
  for (let pass = 0; pass < 140; pass += 1) {
    for (let i = 0; i < labelled.length; i += 1) {
      for (let j = i + 1; j < labelled.length; j += 1) {
        const a = labelled[i];
        const b = labelled[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const overlap = LABEL_LINE - Math.abs(dy);
        // Labels sit to the right of their dot, so the left-hand label sets the clearance.
        if (Math.abs(dx) < (dx >= 0 ? a.label : b.label) && overlap > 0) {
          const direction = dy > 0 || (dy === 0 && i % 2) ? 1 : -1;
          a.y -= (direction * overlap) / 2;
          b.y += (direction * overlap) / 2;
        }
      }
    }
  }
  return placed.map((entry) => ({
    u: entry.point.tx,
    v: (clamp(entry.y, PAD.top - 6, PAD.top + plotHeight + 6) - PAD.top) / plotHeight,
  }));
}

interface Scattered extends Position {
  delay: number;
}

interface View {
  /** Random starting positions while the map settles; null once settled. */
  scatter: Scattered[] | null;
  progress: number;
  focus: string | null;
  caption: "random" | "trained" | "focus";
  fading: boolean;
}

const still: View = { scatter: null, progress: 1, focus: "cat", caption: "focus", fading: false };

export function MeaningMapHero({ size, animate }: HeroVisualProps) {
  const [view, setView] = useState<View>(still);
  const [restartKey, setRestartKey] = useState(0);
  const skipSettle = useRef(false);
  const holdFocus = useRef(false);
  const settled = useMemo(() => settledPositions(size), [size]);

  useEffect(() => {
    if (!animate) setView(still);
  }, [animate]);

  useHeroScript(animate, async (live) => {
    while (true) {
      if (skipSettle.current) {
        skipSettle.current = false;
      } else {
        setView((current) => ({ ...current, focus: null, caption: "random", fading: true }));
        await wait(320, live);
        const scatter = points.map(() => ({
          u: 0.02 + Math.random() * 0.96,
          v: 0.02 + Math.random() * 0.96,
          delay: Math.random() * 0.35,
        }));
        setView({ scatter, progress: 0, focus: null, caption: "random", fading: false });
        await wait(900, live);
        setView((current) => ({ ...current, caption: "trained" }));
        await tween(2800, live, (progress) => setView((current) => ({ ...current, progress })), linear);
        setView((current) => ({ ...current, scatter: null, progress: 1 }));
        await wait(700, live);
      }
      if (holdFocus.current) {
        holdFocus.current = false;
        await wait(5200, live);
      }
      for (const word of tour) {
        setView((current) => ({ ...current, focus: word, caption: "focus" }));
        await wait(3300, live);
      }
    }
  }, restartKey);

  const pick = (word: string) => {
    setView({ scatter: null, progress: 1, focus: word, caption: "focus", fading: false });
    if (!animate) return;
    skipSettle.current = true;
    holdFocus.current = true;
    setRestartKey((key) => key + 1);
  };

  const plotWidth = size.width - PAD.left - PAD.right;
  const plotHeight = size.height - PAD.top - PAD.bottom;
  const positions = settled.map((target, index) => {
    const start = view.scatter?.[index];
    if (!start) return target;
    const k = easeInOut(clamp((view.progress - start.delay) / 0.65, 0, 1));
    return { u: start.u + (target.u - start.u) * k, v: start.v + (target.v - start.v) * k };
  });
  const xAt = (index: number) => PAD.left + positions[index].u * plotWidth;
  const yAt = (index: number) => PAD.top + positions[index].v * plotHeight;
  const focus = view.focus ? byWord.get(view.focus) : undefined;
  const focusIndex = focus ? indexOf.get(focus.word)! : -1;
  const near = new Set(focus?.neighbours?.map(([word]) => word));

  return (
    <>
      <svg
        className={`hero-map${animate ? "" : " is-still"}`}
        viewBox={`0 0 ${size.width} ${size.height}`}
        role="group"
        aria-label="Word map"
      >
        <g>
          {focus?.neighbours?.map(([word]) => {
            const to = indexOf.get(word)!;
            return (
              <line
                key={`${focus.word}-${word}`}
                className="hero-map__line"
                pathLength={1}
                x1={xAt(focusIndex)}
                y1={yAt(focusIndex)}
                x2={xAt(to)}
                y2={yAt(to)}
              />
            );
          })}
        </g>
        <g className={`hero-map__dots${view.fading ? " is-fading" : ""}`}>
          {focus && <circle className="hero-map__ring" cx={xAt(focusIndex)} cy={yAt(focusIndex)} r={9} />}
          {points.map((point, index) => {
            const classes = ["hero-map__dot"];
            if (point.neighbours) classes.push("is-word");
            if (focus && point.group === focus.group) classes.push("is-group");
            if (near.has(point.word)) classes.push("is-near");
            if (point === focus) classes.push("is-focus");
            return <circle key={point.word} className={classes.join(" ")} cx={xAt(index)} cy={yAt(index)} r={point.neighbours ? 3.6 : 2.4} />;
          })}
        </g>
        <g>
          {points.map((point, index) => {
            if (!point.neighbours) return null;
            const classes = ["hero-map__label"];
            if (focus && (anchors.has(point.word) || point.group === focus.group || near.has(point.word))) classes.push("is-on");
            if (point === focus || near.has(point.word)) classes.push("is-hot");
            if (point === focus) classes.push("is-focus");
            return (
              <g key={point.word}>
                <text className={classes.join(" ")} x={xAt(index) + 6.5} y={yAt(index) + 3.6}>{point.word}</text>
                <circle
                  className="hero-map__hit"
                  cx={xAt(index)}
                  cy={yAt(index)}
                  r={11}
                  tabIndex={0}
                  role="button"
                  aria-label={`Show the nearest words to ${point.word}`}
                  onClick={() => pick(point.word)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      pick(point.word);
                    }
                  }}
                />
              </g>
            );
          })}
        </g>
      </svg>
      <p className="hero-map__caption">
        {view.caption === "random" && <><b>Random vectors.</b> 75 words, no meaning yet.</>}
        {view.caption === "trained" && <><b>Trained vectors.</b> Words that mean similar things end up close together.</>}
        {view.caption === "focus" && focus?.neighbours && (
          <>
            Nearest to <b>{focus.word}</b>
            {focus.neighbours.map(([word, cosine], index) => (
              <span key={word}>
                {index === 0 ? " " : " · "}
                {word} <span className="hero-map__cosine">{cosine.toFixed(2)}</span>
              </span>
            ))}
          </>
        )}
      </p>
    </>
  );
}
