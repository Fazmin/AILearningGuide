import { characterLabel, type WordSpan } from "./attention-math";

const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

/**
 * A character × character attention matrix for one head. Rows are queries,
 * columns are keys; opacity is linear in weight. With the causal mask on, cells
 * above the diagonal are hatched because the mask forbids them; with it lifted
 * nothing is hatched and every cell can carry weight.
 */
export function CharacterHeatmap({
  characters,
  matrix,
  queryRow,
  keySpan,
  label,
  causal = true,
}: {
  characters: ReadonlyArray<string>;
  matrix: ReadonlyArray<ReadonlyArray<number>>;
  queryRow: number;
  keySpan?: WordSpan;
  label: string;
  causal?: boolean;
}) {
  const n = characters.length;
  const cell = 10;
  const pad = n <= 36 ? 14 : 12;
  const size = pad + n * cell;
  const showCharacters = n <= 36;
  const ticks = Array.from({ length: Math.floor((n - 1) / 8) + 1 }, (_, index) => index * 8);

  return (
    <svg
      className="atn-char-heatmap"
      viewBox={`0 0 ${size + 2} ${size + 2}`}
      role="img"
      aria-label={label}
    >
      <defs>
        <pattern id="atn-mask-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" className="atn-hatch-line" />
        </pattern>
      </defs>
      {Array.from({ length: causal ? n - 1 : 0 }, (_, row) => (
        <rect
          key={`mask-${row}`}
          className="atn-char-heatmap__mask"
          x={pad + (row + 1) * cell}
          y={pad + row * cell}
          width={(n - row - 1) * cell}
          height={cell}
        />
      ))}
      {matrix.slice(0, n).map((row, rowIndex) =>
        row.slice(0, causal ? rowIndex + 1 : n).map((weight, column) =>
          weight > 0.004 ? (
            <rect
              key={`${rowIndex}-${column}`}
              className="atn-char-heatmap__cell"
              x={pad + column * cell}
              y={pad + rowIndex * cell}
              width={cell}
              height={cell}
              style={{ fillOpacity: Math.min(1, weight) }}
            />
          ) : null,
        ),
      )}
      {keySpan && (
        <rect
          className="atn-char-heatmap__key"
          x={pad + keySpan.start * cell}
          y={pad}
          width={(keySpan.end - keySpan.start) * cell}
          height={n * cell}
        />
      )}
      {queryRow >= 0 && queryRow < n && (
        <rect
          className="atn-char-heatmap__query"
          x={pad}
          y={pad + queryRow * cell}
          width={n * cell}
          height={cell}
        />
      )}
      <rect className="atn-char-heatmap__frame" x={pad} y={pad} width={n * cell} height={n * cell} />
      {showCharacters
        ? characters.map((character, index) => (
            <g key={`label-${index}`}>
              <text className="atn-char-heatmap__label" x={pad + index * cell + cell / 2} y={pad - 4} textAnchor="middle">
                {characterLabel(character)}
              </text>
              <text className="atn-char-heatmap__label" x={pad - 3} y={pad + index * cell + cell * 0.72} textAnchor="end">
                {characterLabel(character)}
              </text>
            </g>
          ))
        : ticks.map((tick) => (
            <g key={`tick-${tick}`}>
              <text className="atn-char-heatmap__tick" x={pad + tick * cell + cell / 2} y={pad - 3} textAnchor="middle">
                {tick}
              </text>
              <text className="atn-char-heatmap__tick" x={pad - 2} y={pad + tick * cell + cell * 0.72} textAnchor="end">
                {tick}
              </text>
            </g>
          ))}
    </svg>
  );
}

/**
 * One query's row before and after softmax: the scores softmax sees on top
 * (scaled, or raw when scaling is off), weights below, with masked (later)
 * positions hatched while the causal mask is on.
 */
export function SoftmaxStrip({
  characters,
  scaledScores,
  weights,
  queryPosition,
  keyPosition,
  keySpan,
  causal = true,
  scaled = true,
}: {
  characters: ReadonlyArray<string>;
  scaledScores: ReadonlyArray<number>;
  weights: ReadonlyArray<number>;
  queryPosition: number;
  keyPosition: number;
  keySpan?: WordSpan;
  causal?: boolean;
  scaled?: boolean;
}) {
  const n = characters.length;
  const width = 320;
  const left = 26;
  const plotWidth = width - left - 4;
  const step = plotWidth / Math.max(1, n);
  const barWidth = Math.max(1, step * 0.72);
  const scoreTop = 18;
  const scoreHeight = 56;
  const zero = scoreTop + scoreHeight / 2;
  const weightTop = scoreTop + scoreHeight + 20;
  const weightHeight = 48;
  const weightBase = weightTop + weightHeight;
  const labelsY = weightBase + 10;
  const height = labelsY + (n <= 40 ? 4 : -6);
  const visibleCount = causal ? queryPosition + 1 : n;
  const visible = scaledScores.slice(0, visibleCount).filter(Number.isFinite);
  const scoreRange = Math.max(1e-6, ...visible.map((value) => Math.abs(value)));
  const topWeight = Math.max(1e-6, ...weights.slice(0, visibleCount));
  const peak = weights.indexOf(topWeight);
  const xAt = (index: number) => left + index * step + (step - barWidth) / 2;
  const masked = causal && queryPosition < n - 1;

  const summary = `${scaled ? "Scaled" : "Raw, unscaled"} scores and softmax weights for ${visibleCount} visible characters. The largest weight, ${percent(topWeight)}, goes to “${characterLabel(characters[peak] ?? "")}” at position ${peak}. ${causal ? `${n - queryPosition - 1} later positions are masked.` : "The causal mask is lifted, so later positions are readable."}`;

  return (
    <svg className="atn-strip" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary}>
      <defs>
        <pattern id="atn-strip-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" className="atn-hatch-line" />
        </pattern>
      </defs>
      <text className="atn-strip__title" x={left} y={11}>
        {scaled ? "scaled score q·k ÷ √d" : "raw score q·k, no ÷ √d"} · scale ±{scoreRange.toFixed(2)}
      </text>
      <text className="atn-strip__title" x={left} y={weightTop - 7}>
        softmax weight · tallest bar {percent(topWeight)}
      </text>
      {keySpan && (
        <rect
          className="atn-strip__key-band"
          x={left + keySpan.start * step}
          y={scoreTop - 2}
          width={(keySpan.end - keySpan.start) * step}
          height={weightBase - scoreTop + 4}
        />
      )}
      {masked && (
        <g>
          <rect
            className="atn-strip__mask"
            x={left + (queryPosition + 1) * step}
            y={scoreTop}
            width={(n - queryPosition - 1) * step}
            height={weightBase - scoreTop}
          />
          <text
            className="atn-strip__mask-label"
            x={left + (queryPosition + 1 + (n - queryPosition - 1) / 2) * step}
            y={zero + 3}
            textAnchor="middle"
          >
            {n - queryPosition - 1 > 3 ? "masked" : ""}
          </text>
        </g>
      )}
      <line className="atn-strip__axis" x1={left} x2={width - 4} y1={zero} y2={zero} />
      <line className="atn-strip__axis" x1={left} x2={width - 4} y1={weightBase} y2={weightBase} />
      <text className="atn-strip__tick" x={left - 4} y={zero + 3} textAnchor="end">0</text>
      <text className="atn-strip__tick" x={left - 4} y={weightBase + 3} textAnchor="end">0</text>
      {characters.slice(0, visibleCount).map((_, index) => {
        const score = scaledScores[index] ?? 0;
        const barHeight = (Math.abs(score) / scoreRange) * (scoreHeight / 2);
        const weight = weights[index] ?? 0;
        const weightBar = (weight / topWeight) * weightHeight;
        const isKey = index === keyPosition;
        return (
          <g key={index} className={isKey ? "is-key" : ""}>
            <rect
              className="atn-strip__score"
              x={xAt(index)}
              y={score >= 0 ? zero - barHeight : zero}
              width={barWidth}
              height={Math.max(0.6, barHeight)}
            />
            <rect
              className="atn-strip__weight"
              x={xAt(index)}
              y={weightBase - weightBar}
              width={barWidth}
              height={Math.max(0.6, weightBar)}
            />
          </g>
        );
      })}
      {n <= 40 &&
        characters.map((character, index) => (
          <text
            key={`c-${index}`}
            className={`atn-strip__char ${index === keyPosition ? "is-key" : ""}`}
            x={left + index * step + step / 2}
            y={labelsY}
            textAnchor="middle"
          >
            {characterLabel(character)}
          </text>
        ))}
    </svg>
  );
}
