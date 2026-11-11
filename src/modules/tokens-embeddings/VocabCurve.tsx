import { useEffect, useRef, useState } from "react";

/** Track the rendered width so the curve is drawn in CSS pixels and lines up with the slider above it. */
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

/**
 * How many tokens this text needs at every vocabulary size. The count only changes where the text can use
 * one more merge, so the curve is a staircase: `drops` lists each vocabulary size at which it steps down.
 * The horizontal scale matches the Vocabulary size slider, so the marked point sits under the slider's thumb.
 */
export function VocabCurve({
  drops,
  startTokens,
  minVocab,
  maxVocab,
  vocabSize,
  tokens,
}: {
  drops: ReadonlyArray<{ vocab: number; tokens: number }>;
  startTokens: number;
  minVocab: number;
  maxVocab: number;
  vocabSize: number;
  tokens: number;
}) {
  const [wrapper, measured] = useWidth(660);
  const W = Math.max(260, measured);
  const H = 104;
  const inset = 10;
  const top = 20;
  const bottom = H - 22;
  const endTokens = drops.length ? drops[drops.length - 1].tokens : startTokens;
  const high = Math.max(startTokens, 1);
  const low = Math.min(endTokens, high - 1);
  const xs = (vocab: number) => inset + ((vocab - minVocab) / (maxVocab - minVocab)) * (W - 2 * inset);
  const ys = (count: number) => bottom - ((count - low) / (high - low)) * (bottom - top - 6) - 3;

  let path = `M${xs(minVocab).toFixed(1)} ${ys(startTokens).toFixed(1)}`;
  drops.forEach((drop) => {
    path += ` H${xs(drop.vocab).toFixed(1)} V${ys(drop.tokens).toFixed(1)}`;
  });
  path += ` H${xs(maxVocab).toFixed(1)}`;
  const area = `${path} V${bottom} H${xs(minVocab).toFixed(1)} Z`;
  const cx = xs(vocabSize);
  const cy = ys(tokens);
  const flip = cx > W * 0.62;
  const unused = drops.filter((drop) => drop.vocab > vocabSize).length;

  return (
    <div className="te-curve" ref={wrapper}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Tokens needed for this text at each vocabulary size: ${startTokens} at ${minVocab} entries, falling to ${endTokens} at ${maxVocab}. At ${vocabSize} entries it needs ${tokens}.`}
      >
        <line className="te-curve__base" x1={xs(minVocab)} x2={xs(maxVocab)} y1={bottom} y2={bottom} />
        <path className="te-curve__area" d={area} />
        {drops.map((drop) => (
          <line
            key={drop.vocab}
            className={`te-curve__merge${drop.vocab > vocabSize ? " is-unused" : ""}`}
            x1={xs(drop.vocab)}
            x2={xs(drop.vocab)}
            y1={bottom}
            y2={bottom + 5}
          />
        ))}
        <path className="te-curve__line" d={path} />
        <line className="te-curve__guide" x1={cx} x2={cx} y1={4} y2={bottom} />
        <circle className="te-curve__point" cx={cx} cy={cy} r={5} />
        <text
          className="te-curve__now"
          x={cx + (flip ? -9 : 9)}
          y={Math.max(13, Math.min(cy - 7, bottom - 16))}
          textAnchor={flip ? "end" : "start"}
        >
          {tokens} token{tokens === 1 ? "" : "s"} at {vocabSize}
        </text>
        <text className="te-curve__end" x={xs(minVocab)} y={H - 4}>
          {minVocab}: {startTokens} tokens, no merges
        </text>
        <text className="te-curve__end" x={xs(maxVocab)} y={H - 4} textAnchor="end">
          {maxVocab}: {endTokens} tokens
        </text>
      </svg>
      <p className="te-curve__caption">
        Tokens this text needs at every vocabulary size. Each tick is a merge this text can use
        {unused > 0 ? `; the ${unused} to the right of the marker are switched off` : ""}.
      </p>
    </div>
  );
}
