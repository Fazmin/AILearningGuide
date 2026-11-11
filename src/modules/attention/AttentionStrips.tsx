import { useLayoutEffect, useRef, useState } from "react";
import { characterLabel, type WordSpan } from "./attention-math";

const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 2) => `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;

/** Track the rendered width so each strip is drawn 1:1 and its labels keep their pixel size. */
function useMeasuredWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const next = Math.round(element.getBoundingClientRect().width);
      if (next > 0) setWidth(next);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);
  return [ref, Math.max(280, width || fallback)] as const;
}

const LEFT = 30;
const RIGHT_PAD = 8;

/**
 * One query's row before and after softmax: the scores softmax sees on top
 * (scaled, or raw when scaling is off), weights below, with masked (later)
 * positions hatched while the causal mask is on. Drawn at the card's real
 * width, with the query column and the inspected key named on the picture.
 */
export function SoftmaxStrip({
  characters,
  scaledScores,
  weights,
  queryPosition,
  keyPosition,
  keySpan,
  queryWord,
  causal = true,
  scaled = true,
}: {
  characters: ReadonlyArray<string>;
  scaledScores: ReadonlyArray<number>;
  weights: ReadonlyArray<number>;
  queryPosition: number;
  keyPosition: number;
  keySpan?: WordSpan;
  queryWord?: string;
  causal?: boolean;
  scaled?: boolean;
}) {
  const [ref, width] = useMeasuredWidth(640);
  const n = characters.length;
  const plotWidth = width - LEFT - RIGHT_PAD;
  const step = plotWidth / Math.max(1, n);
  const barWidth = Math.max(1, step * 0.72);
  const scoreTop = 52;
  const scoreHeight = 64;
  const zero = scoreTop + scoreHeight / 2;
  const weightTop = scoreTop + scoreHeight + 26;
  const weightHeight = 56;
  const weightBase = weightTop + weightHeight;
  const showCharacters = step >= 7.5;
  const height = weightBase + (showCharacters ? 20 : 8);
  const visibleCount = causal ? queryPosition + 1 : n;
  const visible = scaledScores.slice(0, visibleCount).filter(Number.isFinite);
  const scoreRange = Math.max(1e-6, ...visible.map((value) => Math.abs(value)));
  const topWeight = Math.max(1e-6, ...weights.slice(0, visibleCount));
  const peak = weights.indexOf(topWeight);
  const xAt = (index: number) => LEFT + index * step + (step - barWidth) / 2;
  const centre = (index: number) => LEFT + index * step + step / 2;
  const masked = causal && queryPosition < n - 1;
  const maskedWidth = (n - queryPosition - 1) * step;
  const keyVisible = keyPosition < visibleCount;
  const keyScore = scaledScores[keyPosition];
  const keyReading =
    width < 430
      ? ""
      : keyVisible && Number.isFinite(keyScore)
        ? ` · its last character: score ${signed(keyScore ?? 0)}, weight ${percent(weights[keyPosition] ?? 0, 2)}`
        : " · masked";
  const clampLabel = (x: number, half: number) => Math.min(width - RIGHT_PAD - half, Math.max(LEFT + half, x));
  const keyCentre = keySpan ? LEFT + ((keySpan.start + keySpan.end) / 2) * step : centre(keyPosition);
  const short = (word: string) => (word.length > 14 ? `${word.slice(0, 13)}…` : word);
  const keyText = `key “${short(keySpan?.text ?? characterLabel(characters[keyPosition] ?? ""))}”${keyReading}`;
  const queryText = `query “${characterLabel(characters[queryPosition] ?? "")}”${queryWord ? ` of ${short(queryWord)}` : ""}`;

  const summary = `${scaled ? "Scaled" : "Raw, unscaled"} scores and softmax weights for ${visibleCount} visible characters. The largest weight, ${percent(topWeight)}, goes to “${characterLabel(characters[peak] ?? "")}” at position ${peak}. ${causal ? `${n - queryPosition - 1} later positions are masked.` : "The causal mask is lifted, so later positions are readable."}`;

  return (
    <div className="atn-strip-wrap" ref={ref}>
      <svg className="atn-strip" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary}>
        <defs>
          <pattern id="atn-strip-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className="atn-hatch-line" />
          </pattern>
        </defs>
        <text className="atn-strip__name atn-strip__name--key" x={clampLabel(keyCentre, keyText.length * 3.2)} y={12} textAnchor="middle">
          {keyText}
        </text>
        <text className="atn-strip__name atn-strip__name--query" x={clampLabel(centre(queryPosition), queryText.length * 3.2)} y={25} textAnchor="middle">
          {queryText}
        </text>
        <text className="atn-strip__title" x={LEFT} y={scoreTop - 8}>
          {scaled ? "scaled score q·k ÷ √d" : "raw score q·k, no ÷ √d"} · scale ±{scoreRange.toFixed(2)}
        </text>
        <text className="atn-strip__title" x={LEFT} y={weightTop - 8}>
          softmax weight · tallest bar {percent(topWeight)}
        </text>
        {keySpan && (
          <rect
            className="atn-strip__key-band"
            x={LEFT + keySpan.start * step}
            y={scoreTop - 2}
            width={(keySpan.end - keySpan.start) * step}
            height={weightBase - scoreTop + 4}
          />
        )}
        {masked && (
          <g>
            <rect className="atn-strip__mask" x={LEFT + (queryPosition + 1) * step} y={scoreTop} width={maskedWidth} height={weightBase - scoreTop} />
            {maskedWidth > 52 && (
              <text className="atn-strip__mask-label" x={LEFT + (queryPosition + 1) * step + maskedWidth / 2} y={zero + 3} textAnchor="middle">
                {maskedWidth > 150 ? "masked: later characters" : "masked"}
              </text>
            )}
          </g>
        )}
        <line className="atn-strip__axis" x1={LEFT} x2={width - RIGHT_PAD} y1={zero} y2={zero} />
        <line className="atn-strip__axis" x1={LEFT} x2={width - RIGHT_PAD} y1={weightBase} y2={weightBase} />
        <text className="atn-strip__tick" x={LEFT - 5} y={zero + 3.5} textAnchor="end">0</text>
        <text className="atn-strip__tick" x={LEFT - 5} y={scoreTop + 8} textAnchor="end">+</text>
        <text className="atn-strip__tick" x={LEFT - 5} y={scoreTop + scoreHeight} textAnchor="end">−</text>
        <text className="atn-strip__tick" x={LEFT - 5} y={weightBase + 3.5} textAnchor="end">0</text>
        {characters.slice(0, visibleCount).map((_, index) => {
          const score = scaledScores[index] ?? 0;
          const barHeight = (Math.abs(score) / scoreRange) * (scoreHeight / 2);
          const weight = weights[index] ?? 0;
          const weightBar = (weight / topWeight) * weightHeight;
          const isKey = index === keyPosition;
          return (
            <g key={index} className={isKey ? "is-key" : ""}>
              <rect className="atn-strip__score" x={xAt(index)} y={score >= 0 ? zero - barHeight : zero} width={barWidth} height={Math.max(0.6, barHeight)} />
              <rect className="atn-strip__weight" x={xAt(index)} y={weightBase - weightBar} width={barWidth} height={Math.max(0.6, weightBar)} />
            </g>
          );
        })}
        <rect className="atn-strip__query" x={LEFT + queryPosition * step} y={scoreTop - 2} width={step} height={weightBase - scoreTop + 4} />
        <line className="atn-strip__query-stem" x1={centre(queryPosition)} x2={centre(queryPosition)} y1={29} y2={scoreTop - 2} />
        {showCharacters &&
          characters.map((character, index) => (
            <text
              key={`c-${index}`}
              className={`atn-strip__char ${index === keyPosition ? "is-key" : index === queryPosition ? "is-query" : ""}`}
              x={centre(index)}
              y={weightBase + 14}
              textAnchor="middle"
            >
              {characterLabel(character)}
            </text>
          ))}
      </svg>
    </div>
  );
}

/**
 * What each visible key hands to the mix: its weight times the length of its value vector.
 * The outlined ghost bars use the model's own values, so zeroing a value leaves a hollow bar.
 */
export function PayloadStrip({
  characters,
  payload,
  modelPayload,
  visibleCount,
  keyPosition,
  keySpan,
}: {
  characters: ReadonlyArray<string>;
  payload: ReadonlyArray<number>;
  modelPayload: ReadonlyArray<number>;
  visibleCount: number;
  keyPosition: number;
  keySpan?: WordSpan;
}) {
  const [ref, width] = useMeasuredWidth(320);
  const n = characters.length;
  const left = 8;
  const step = (width - left - RIGHT_PAD) / Math.max(1, n);
  const barWidth = Math.max(1, step * 0.72);
  const top = 22;
  const plotHeight = 44;
  const base = top + plotHeight;
  const height = base + 8;
  const tallest = Math.max(1e-9, ...modelPayload.slice(0, visibleCount), ...payload.slice(0, visibleCount));
  const emptied = payload.slice(0, visibleCount).filter((value, index) => value === 0 && (modelPayload[index] ?? 0) > 0).length;
  const total = payload.slice(0, visibleCount).reduce((sum, value) => sum + value, 0);

  return (
    <div className="atn-strip-wrap" ref={ref}>
      <svg
        className="atn-strip atn-payload"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`What each of the ${visibleCount} visible characters adds to the mix: its weight times the length of its value vector. ${emptied === 0 ? "Every value is the model's own." : `${emptied} characters carry zero values, so they add nothing; their hollow bars show what the model's own values would add.`}`}
      >
        <text className="atn-strip__title" x={left} y={12}>
          weight × value length, per key{total === 0 ? " · all zero" : ""}
        </text>
        {keySpan && (
          <rect className="atn-strip__key-band" x={left + keySpan.start * step} y={top - 2} width={(keySpan.end - keySpan.start) * step} height={plotHeight + 4} />
        )}
        <line className="atn-strip__axis" x1={left} x2={width - RIGHT_PAD} y1={base} y2={base} />
        {characters.slice(0, visibleCount).map((_, index) => {
          const x = left + index * step + (step - barWidth) / 2;
          const ghost = ((modelPayload[index] ?? 0) / tallest) * plotHeight;
          const solid = ((payload[index] ?? 0) / tallest) * plotHeight;
          return (
            <g key={index} className={index === keyPosition ? "is-key" : ""}>
              <rect className="atn-payload__ghost" x={x} y={base - ghost} width={barWidth} height={Math.max(0.6, ghost)} />
              {solid > 0 && <rect className="atn-payload__bar" x={x} y={base - solid} width={barWidth} height={Math.max(0.6, solid)} />}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * The dot product as a sum of products: the four editable dimensions one by one, the other
 * dimensions together, and the total. Same signs push the score up; opposite signs pull it down.
 */
export function DotProductBars({
  query,
  keyVector,
  score,
  dimensions,
}: {
  query: ReadonlyArray<number>;
  keyVector: ReadonlyArray<number>;
  score: number;
  dimensions: number;
}) {
  const products = query.map((value, index) => value * (keyVector[index] ?? 0));
  const shown = products.reduce((sum, value) => sum + value, 0);
  const rest = score - shown;
  const rows = [
    ...products.map((value, index) => ({ id: `d${index}`, label: `q${index + 1} × k${index + 1}`, value, total: false })),
    { id: "rest", label: `other ${dimensions - products.length}`, value: rest, total: false },
    { id: "sum", label: "score q · k", value: score, total: true },
  ];
  const range = Math.max(1e-9, ...rows.map((row) => Math.abs(row.value)));

  return (
    <div
      className="atn-products"
      role="img"
      aria-label={`The score as a sum of products. ${products.map((value, index) => `Dimension ${index + 1}: ${signed(value)}`).join(". ")}. The other ${dimensions - products.length} dimensions together: ${signed(rest)}. Score: ${signed(score, 4)}.`}
    >
      <span className="atn-products__title" aria-hidden="true">
        The score is a sum of products. Same signs add, opposite signs subtract.
      </span>
      {rows.map((row) => (
        <div key={row.id} className={`atn-products__row${row.total ? " is-total" : ""}`} aria-hidden="true">
          <span>{row.label}</span>
          <span className="atn-products__track">
            <i
              className={row.value < 0 ? "is-negative" : ""}
              style={{ width: `${((Math.abs(row.value) / range) * 50).toFixed(2)}%` }}
            />
          </span>
          <b>{signed(row.value)}</b>
        </div>
      ))}
    </div>
  );
}
