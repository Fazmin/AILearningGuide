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
  const pad = n <= 36 ? 14 : 20;
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
