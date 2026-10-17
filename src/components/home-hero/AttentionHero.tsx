import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "@app/store/app-store";
import { clamp, useHeroScript, wait, type HeroVisualProps } from "./hero-motion";

type LastWord = "tired" | "wide";

const words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too"] as const;
const IT = 7;
const LAST = words.length;
const other: Record<LastWord, LastWord> = { tired: "wide", wide: "tired" };

// Hand-set weights from "it" to every other word, one row per ending; each row sums to 1.
const weights: Record<LastWord, readonly number[]> = {
  tired: [0.03, 0.62, 0.03, 0.05, 0.02, 0.08, 0.04, 0, 0.03, 0.02, 0.08],
  wide: [0.03, 0.09, 0.02, 0.06, 0.03, 0.61, 0.04, 0, 0.03, 0.02, 0.07],
};

/** Words are measured once at this size; widths scale linearly from it. */
const REFERENCE_SIZE = 20;
const GAP = 0.3;

interface Widths {
  words: number[];
  last: Record<LastWord, number>;
}

const estimate = (text: string) => text.length * REFERENCE_SIZE * 0.55;
const estimatedWidths: Widths = {
  words: words.map(estimate),
  last: { tired: estimate("tired"), wide: estimate("wide") },
};

const wordClass = (index: number) =>
  `hero-attention__word${index === IT ? " is-it" : index === LAST ? " is-swap" : ""}`;

export function AttentionHero({ size, animate }: HeroVisualProps) {
  const typeface = useAppStore((state) => state.typeface);
  const [last, setLast] = useState<LastWord>("tired");
  const [restartKey, setRestartKey] = useState(0);
  const [widths, setWidths] = useState<Widths>(estimatedWidths);
  const measureRef = useRef<SVGGElement>(null);

  useLayoutEffect(() => {
    const group = measureRef.current;
    if (!group) return undefined;
    const measure = () => {
      const lengths = [...group.querySelectorAll("text")].map((text) => {
        const measured = typeof text.getComputedTextLength === "function" ? text.getComputedTextLength() : 0;
        return measured || estimate(text.textContent ?? "");
      });
      setWidths({ words: lengths.slice(0, LAST), last: { tired: lengths[LAST], wide: lengths[LAST + 1] } });
    };
    measure();
    let cancelled = false;
    void document.fonts?.ready.then(() => {
      if (!cancelled) measure();
    });
    return () => {
      cancelled = true;
    };
  }, [typeface]);

  useEffect(() => {
    if (!animate) setLast("tired");
  }, [animate]);

  useHeroScript(animate, async (live) => {
    while (true) {
      await wait(3600, live);
      setLast((current) => other[current]);
    }
  }, restartKey);

  const layout = useMemo(() => {
    const { width, height } = size;
    const widest = Math.max(widths.last.tired, widths.last.wide);
    // The sentence's width for each pixel of font size, including the gaps between words.
    const widthPerSize = (widths.words.reduce((sum, w) => sum + w, 0) + widest) / REFERENCE_SIZE + GAP * words.length;
    let fontSize = clamp(width / 31, 11, 19);
    if (fontSize * widthPerSize > width - 36) fontSize = Math.max(8.5, (width - 36) / widthPerSize);
    const scale = fontSize / REFERENCE_SIZE;
    const start = (width - fontSize * widthPerSize) / 2;
    const xs: number[] = [];
    let x = start;
    widths.words.forEach((w) => {
      xs.push(x);
      x += w * scale + fontSize * GAP;
    });
    xs.push(x);
    const baseline = Math.round(height * 0.6);
    const top = baseline - fontSize * 0.95;
    return { fontSize, scale, xs, start, baseline, top, maxArc: Math.max(30, top - 72) };
  }, [size, widths]);

  const lastWidth = widths.last[last] * layout.scale;
  const widthOf = (index: number) => (index === LAST ? lastWidth : widths.words[index] * layout.scale);
  const centre = (index: number) => layout.xs[index] + widthOf(index) / 2;
  const row = weights[last];
  const strongest = row.indexOf(Math.max(...row));
  const from = centre(IT);
  const swap = () => {
    setLast((current) => other[current]);
    if (animate) setRestartKey((key) => key + 1);
  };

  const arcs = row.map((weight, index) => {
    if (index === IT) return null;
    const to = centre(index);
    const height = Math.min(layout.maxArc, 10 + Math.abs(to - from) * 0.55);
    // A cubic's peak sits at three quarters of its control height.
    const control = layout.top - height / 0.75;
    return { index, weight, to, height, d: `M${from},${layout.top} C${from},${control} ${to},${control} ${to},${layout.top}` };
  });
  const peak = arcs[strongest];

  return (
    <>
      <svg
        className={`hero-attention${animate ? "" : " is-still"}`}
        viewBox={`0 0 ${size.width} ${size.height}`}
        aria-hidden="true"
      >
        <g className="hero-attention__measure" ref={measureRef} fontSize={REFERENCE_SIZE}>
          {words.map((word, index) => (
            <text key={word + index} className={wordClass(index)}>{word}</text>
          ))}
          <text className={wordClass(LAST)}>tired</text>
          <text className={wordClass(LAST)}>wide</text>
        </g>
        <g>
          {arcs.map((arc) => {
            if (!arc) return null;
            const opacity = 0.2 + 0.8 * Math.min(1, arc.weight / 0.6);
            const top = arc.index === strongest ? " is-top" : "";
            return (
              <g key={arc.index}>
                <path className={`hero-attention__arc${top}`} d={arc.d} style={{ strokeWidth: 0.8 + arc.weight * 9, opacity }} />
                <circle className={`hero-attention__end${top}`} cx={arc.to} cy={layout.top} r={1.6 + arc.weight * 5} style={{ opacity }} />
              </g>
            );
          })}
          {peak && (
            <text className="hero-attention__peak" x={(from + peak.to) / 2} y={layout.top - peak.height - 9} textAnchor="middle">
              {peak.weight.toFixed(2)}
            </text>
          )}
        </g>
        <g fontSize={layout.fontSize}>
          {words.map((word, index) => (
            <text key={word + index} className={wordClass(index)} x={layout.xs[index]} y={layout.baseline}>{word}</text>
          ))}
          <text key={last} className={wordClass(LAST)} x={layout.xs[LAST]} y={layout.baseline}>{last}</text>
          <line
            key={`${last}-line`}
            className="hero-attention__underline"
            x1={layout.xs[LAST]}
            x2={layout.xs[LAST] + lastWidth}
            y1={layout.baseline + layout.fontSize * 0.3}
            y2={layout.baseline + layout.fontSize * 0.3}
          />
        </g>
      </svg>
      <p className="hero-attention__readout" style={{ top: layout.baseline + layout.fontSize * 1.4, left: Math.max(18, layout.start) }}>
        <span className="hero-attention__it">it</span> attends most to{" "}
        <b>{strongest === LAST ? last : words[strongest]}</b>{" "}
        <span className="hero-attention__weight">{row[strongest].toFixed(2)}</span>
        <br />
        <span className="hero-attention__hint">Click “{last}” to swap it for “{other[last]}”.</span>
      </p>
      <button
        type="button"
        className="hero-attention__swap"
        aria-label={`Swap “${last}” for “${other[last]}”`}
        onClick={swap}
        style={{
          left: layout.xs[LAST] - 5,
          top: layout.baseline - layout.fontSize * 1.05,
          width: lastWidth + 10,
          height: layout.fontSize * 1.55,
        }}
      />
    </>
  );
}
