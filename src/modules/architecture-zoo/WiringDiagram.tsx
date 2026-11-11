import { SIZE, type WiringId } from "./zoo";

const VIEW_WIDTH = 312;
const STEP = 36;
const FIRST = 30;
const TOP = 22;
const BOTTOM = 196;
const HEIGHT = 232;

const xAt = (position: number) => FIRST + position * STEP;

const ranges = (positions: number[]) => {
  if (!positions.length) return "no inputs";
  const parts: string[] = [];
  let start = positions[0];
  let previous = positions[0];
  for (const value of [...positions.slice(1), Number.NaN]) {
    if (value === previous + 1) {
      previous = value;
      continue;
    }
    parts.push(start === previous ? `${start}` : `${start}–${previous}`);
    start = value;
    previous = value;
  }
  return `input${positions.length === 1 ? "" : "s"} ${parts.join(", ")}`;
};

interface Edge {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  lit: boolean;
}

interface Node {
  key: string;
  x: number;
  y: number;
  lit: boolean;
  inner: boolean;
}

/**
 * The same dependencies as the grid, drawn as wires: inputs along the bottom, outputs along the
 * top, and the traced output's wires in the lab accent. A convolution shows one row of units per
 * layer, and the recurrent wirings show the chain of states each input has to travel along.
 */
export function WiringDiagram({
  wiring,
  label,
  layers,
  matrix,
  focus,
  onFocus,
}: {
  wiring: WiringId;
  label: string;
  layers: number;
  matrix: ReadonlyArray<ReadonlyArray<number | null>>;
  focus: number;
  onFocus: (output: number) => void;
}) {
  const edges: Edge[] = [];
  const nodes: Node[] = [];
  const chain = wiring === "rnn" || wiring === "ssm";

  if (wiring === "conv") {
    const rowY = (layer: number) => BOTTOM - (layer / layers) * (BOTTOM - TOP);
    const inCone = (layer: number, position: number) => Math.abs(position - focus) <= layers - layer;
    for (let layer = 0; layer <= layers; layer += 1) {
      for (let position = 0; position < SIZE; position += 1) {
        nodes.push({
          key: `n${layer}-${position}`,
          x: xAt(position),
          y: rowY(layer),
          lit: inCone(layer, position),
          inner: layer > 0 && layer < layers,
        });
        if (layer === layers) continue;
        for (const offset of [-1, 0, 1]) {
          const above = position + offset;
          if (above < 0 || above >= SIZE) continue;
          edges.push({
            key: `e${layer}-${position}-${above}`,
            x1: xAt(position),
            y1: rowY(layer),
            x2: xAt(above),
            y2: rowY(layer + 1),
            lit: inCone(layer, position) && inCone(layer + 1, above),
          });
        }
      }
    }
  } else {
    for (let position = 0; position < SIZE; position += 1) {
      const seen = matrix[focus][position] !== null;
      nodes.push({ key: `in${position}`, x: xAt(position), y: BOTTOM, lit: seen, inner: false });
      nodes.push({ key: `out${position}`, x: xAt(position), y: TOP, lit: chain ? position <= focus : position === focus, inner: false });
    }
    if (chain) {
      for (let position = 0; position < SIZE; position += 1) {
        edges.push({ key: `up${position}`, x1: xAt(position), y1: BOTTOM, x2: xAt(position), y2: TOP, lit: position <= focus });
        if (position > 0) {
          edges.push({ key: `hop${position}`, x1: xAt(position - 1), y1: TOP, x2: xAt(position), y2: TOP, lit: position <= focus });
        }
      }
    } else {
      matrix.forEach((row, output) =>
        row.forEach((hop, input) => {
          if (hop === null) return;
          edges.push({ key: `w${output}-${input}`, x1: xAt(input), y1: BOTTOM, x2: xAt(output), y2: TOP, lit: output === focus });
        }),
      );
    }
  }

  const seenInputs = matrix[focus].flatMap((hop, input) => (hop === null ? [] : [input + 1]));
  const farthest = Math.max(0, ...matrix[focus].map((hop) => hop ?? 0));
  const reading =
    `Output ${focus + 1} reads ${ranges(seenInputs)}` +
    (chain && farthest > 1
      ? `. Input 1 travels ${farthest} steps along the chain.`
      : wiring === "conv"
        ? ` through ${layers} layer${layers === 1 ? "" : "s"}.`
        : seenInputs.length > 1
          ? ", each in one hop."
          : ".");

  return (
    <div className="az-wires">
      <span className="az-wires__caption" aria-hidden="true">
        Output position · pick one to trace its wires
      </span>
      <div className="az-wires__outputs" role="group" aria-label="Output to trace">
        {Array.from({ length: SIZE }, (_, output) => (
          <button
            key={output}
            type="button"
            className={output === focus ? "is-selected" : ""}
            aria-pressed={output === focus}
            aria-label={`Trace output ${output + 1}`}
            onClick={() => onFocus(output)}
          >
            {output + 1}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${VIEW_WIDTH} ${HEIGHT}`} role="img" aria-label={`${label} wiring, outputs on top and inputs below. ${reading}`}>
        <defs>
          <marker id="az-wires-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" />
          </marker>
        </defs>
        {edges.filter((edge) => !edge.lit).map((edge) => (
          <line key={edge.key} className="az-wires__edge" x1={edge.x1} y1={edge.y1} x2={edge.x2} y2={edge.y2} />
        ))}
        {edges.filter((edge) => edge.lit).map((edge) => (
          <line
            key={edge.key}
            className="az-wires__edge is-lit"
            x1={edge.x1}
            y1={edge.y1}
            x2={edge.y1 === edge.y2 ? edge.x2 - 8 : edge.x2}
            y2={edge.y2}
            markerEnd={edge.y1 === edge.y2 ? "url(#az-wires-arrow)" : undefined}
          />
        ))}
        {nodes.map((node) => (
          <circle
            key={node.key}
            className={`az-wires__node${node.lit ? " is-lit" : ""}${node.inner ? " is-inner" : ""}`}
            cx={node.x}
            cy={node.y}
            r={node.inner ? 3.2 : 5}
          />
        ))}
        {Array.from({ length: SIZE }, (_, position) => (
          <text key={position} className="az-wires__tick" x={xAt(position)} y={BOTTOM + 18} textAnchor="middle">
            {position + 1}
          </text>
        ))}
        <text className="az-wires__axis" x={VIEW_WIDTH / 2} y={HEIGHT - 3} textAnchor="middle">
          input position
        </text>
      </svg>
      <p className="az-wires__reading">{reading}</p>
    </div>
  );
}
