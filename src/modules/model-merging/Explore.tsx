import { useMemo } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  type ModuleContext,
} from "@app/module-sdk";
import {
  buildTaskVectors,
  lossesOf,
  lossesFor,
  LORA_ALPHA,
  LORA_RANK,
  type TaskLosses,
  type VectorSource,
} from "./lab";
import {
  addToBase,
  cosine,
  mergeTaskVectors,
  norm,
  pickCoordinates,
  signStats,
  type MergeMethod,
  type MergeParams,
} from "./merge";

const V = TINY_VOCAB_SIZE;
const glyph = (index: number) => (TINY_VOCAB[index] === " " ? "␣" : TINY_VOCAB[index]);
const coordinateName = (index: number) => `${glyph(Math.floor(index / V))}→${glyph(index % V)}`;

const METHODS: ReadonlyArray<{ value: MergeMethod; label: string; tag: string }> = [
  { value: "linear", label: "Linear blend", tag: "Lin" },
  { value: "task", label: "Task arithmetic", tag: "TA" },
  { value: "ties", label: "TIES", tag: "TIES" },
  { value: "dare", label: "DARE", tag: "DARE" },
  { value: "slerp", label: "SLERP", tag: "SL" },
];

const clamp = (value: unknown, low: number, high: number, fallback: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const signed = (value: number, digits = 3) =>
  Number.isFinite(value) ? `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}` : "—";

/* -------------------------------------------------------------------------- */
/* Task-vector map                                                             */
/* -------------------------------------------------------------------------- */

const CELL = 9;
const GUTTER = 14;

function TaskVectorMap({
  label,
  caption,
  values,
  other,
  scale,
}: {
  label: string;
  caption: string;
  values: Float32Array;
  other: Float32Array;
  scale: number;
}) {
  const size = V * CELL;
  const cells = [];
  let conflicts = 0;
  for (let row = 0; row < V; row += 1) {
    for (let column = 0; column < V; column += 1) {
      const index = row * V + column;
      const value = values[index];
      const magnitude = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
      const x = GUTTER + column * CELL;
      const y = GUTTER + row * CELL;
      const conflict =
        Math.abs(value) > 1e-6 &&
        Math.abs(other[index]) > 1e-6 &&
        Math.sign(value) !== Math.sign(other[index]);
      if (conflict) conflicts += 1;
      cells.push(
        <rect
          key={index}
          x={x}
          y={y}
          width={CELL}
          height={CELL}
          style={{
            fill:
              magnitude < 0.004
                ? "var(--panel-muted)"
                : `color-mix(in srgb, ${value >= 0 ? "var(--forward)" : "var(--loss)"} ${Math.round(
                    4 + magnitude * 96,
                  )}%, var(--panel-solid))`,
          }}
        />,
      );
      if (conflict) {
        cells.push(
          <line key={`c-${index}`} className="mrg-map__conflict" x1={x + 1} y1={y + CELL - 1} x2={x + CELL - 1} y2={y + 1} />,
        );
      }
    }
  }
  return (
    <figure className="mrg-map">
      <svg
        viewBox={`0 0 ${GUTTER + size + 2} ${GUTTER + size + 2}`}
        role="img"
        aria-label={`${label}. ${caption}. Largest entry magnitude ${scale.toFixed(2)}. ${conflicts} entries disagree in sign with the other task vector and are struck through.`}
      >
        {Array.from({ length: V }, (_, index) => (
          <g key={`l-${index}`} className="mrg-map__label">
            <text x={GUTTER + index * CELL + CELL / 2} y={GUTTER - 4} textAnchor="middle">
              {glyph(index)}
            </text>
            <text x={GUTTER - 4} y={GUTTER + index * CELL + CELL / 2 + 2.5} textAnchor="end">
              {glyph(index)}
            </text>
          </g>
        ))}
        {cells}
        <rect className="mrg-map__frame" x={GUTTER} y={GUTTER} width={size} height={size} />
      </svg>
      <figcaption>
        <strong>{label}</strong>
        <span>{caption}</span>
      </figcaption>
    </figure>
  );
}

/* -------------------------------------------------------------------------- */
/* Coordinate strip                                                            */
/* -------------------------------------------------------------------------- */

function CoordinateStrip({
  method,
  coordinates,
  taskA,
  taskB,
  processed,
  merged,
  elected,
}: {
  method: MergeMethod;
  coordinates: { index: number; conflict: boolean }[];
  taskA: Float32Array;
  taskB: Float32Array;
  processed: [Float32Array, Float32Array];
  merged: Float32Array;
  elected: Int8Array;
}) {
  const width = 660;
  const height = 250;
  const top = 26;
  const bottom = 186;
  const zero = (top + bottom) / 2;
  const column = width / Math.max(1, coordinates.length);
  const bar = Math.min(13, column / 4.4);
  const ceiling = Math.max(
    1e-6,
    ...coordinates.flatMap(({ index }) => [
      Math.abs(taskA[index]),
      Math.abs(taskB[index]),
      Math.abs(processed[0][index]),
      Math.abs(processed[1][index]),
      Math.abs(merged[index]),
    ]),
  );
  const yOf = (value: number) => zero - (value / ceiling) * (zero - top);
  const showsProcessing = method === "ties" || method === "dare";

  const describe = coordinates
    .map(({ index, conflict }) => {
      const parts = [
        `${coordinateName(index)} ${conflict ? "conflict" : "agree"}`,
        `recipes ${taskA[index].toFixed(2)}`,
        `proverbs ${taskB[index].toFixed(2)}`,
        `merged ${merged[index].toFixed(2)}`,
      ];
      return parts.join(", ");
    })
    .join("; ");

  const inputBar = (
    key: string,
    x: number,
    raw: number,
    kept: number,
    tone: "a" | "b",
  ) => {
    const removed = showsProcessing && kept === 0 && Math.abs(raw) > 1e-6;
    const value = showsProcessing ? kept : raw;
    return (
      <g key={key}>
        {removed ? (
          <rect
            className={`mrg-strip__bar mrg-strip__bar--${tone} is-removed`}
            x={x}
            y={Math.min(yOf(raw), zero)}
            width={bar}
            height={Math.max(1, Math.abs(yOf(raw) - zero))}
          />
        ) : (
          <rect
            className={`mrg-strip__bar mrg-strip__bar--${tone}`}
            x={x}
            y={Math.min(yOf(value), zero)}
            width={bar}
            height={Math.max(1, Math.abs(yOf(value) - zero))}
          />
        )}
        {method === "dare" && !removed && Math.abs(raw) > 1e-6 && (
          <line className="mrg-strip__original" x1={x - 2} x2={x + bar + 2} y1={yOf(raw)} y2={yOf(raw)} />
        )}
      </g>
    );
  };

  if (coordinates.length === 0) {
    return (
      <p className="lab-note">
        No weight is changed by both task vectors, so there are no conflicts or agreements to draw.
        An untrained fine-tune is a zero task vector.
      </p>
    );
  }

  return (
    <div className="mrg-strip">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Ten coordinates of the two task vectors and the merge. ${describe}.`}>
        <line className="mrg-strip__zero" x1={0} x2={width} y1={zero} y2={zero} />
        <text className="mrg-strip__axis" x={2} y={top - 12}>
          +{ceiling.toFixed(2)}
        </text>
        <text className="mrg-strip__axis" x={2} y={bottom + 12}>
          −{ceiling.toFixed(2)}
        </text>
        {coordinates.map(({ index, conflict }, position) => {
          const center = position * column + column / 2;
          const xA = center - bar * 1.6;
          const xB = center - bar * 0.5;
          const xM = center + bar * 0.6;
          const sign = elected[index] ?? 0;
          return (
            <g key={index}>
              {position === coordinates.length / 2 && (
                <line className="mrg-strip__divider" x1={position * column} x2={position * column} y1={8} y2={height - 20} />
              )}
              {inputBar(`a-${index}`, xA, taskA[index], processed[0][index], "a")}
              {inputBar(`b-${index}`, xB, taskB[index], processed[1][index], "b")}
              <rect
                className="mrg-strip__bar mrg-strip__bar--merged"
                x={xM}
                y={Math.min(yOf(merged[index]), zero)}
                width={bar}
                height={Math.max(1, Math.abs(yOf(merged[index]) - zero))}
              />
              {method === "ties" && sign !== 0 && (
                <text className="mrg-strip__elected" x={xM + bar / 2} y={sign > 0 ? top - 2 : bottom + 14} textAnchor="middle">
                  {sign > 0 ? "▲ +" : "▼ −"}
                </text>
              )}
              <text className="mrg-strip__value" x={center} y={height - 34} textAnchor="middle">
                {merged[index] >= 0 ? "+" : "−"}
                {Math.abs(merged[index]).toFixed(2)}
              </text>
              <text className="mrg-strip__name" x={center} y={height - 18} textAnchor="middle">
                {coordinateName(index)}
              </text>
              <text className="mrg-strip__kind" x={center} y={height - 5} textAnchor="middle">
                {conflict ? "conflict" : "agree"}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mrg-legend" aria-hidden="true">
        <span><i className="mrg-swatch mrg-swatch--a" />τ recipes (left bar)</span>
        <span><i className="mrg-swatch mrg-swatch--b" />τ proverbs (middle bar)</span>
        <span><i className="mrg-swatch mrg-swatch--merged" />merged (right bar, value printed)</span>
        {showsProcessing && (
          <span><i className="mrg-swatch mrg-swatch--removed" />{method === "ties" ? "trimmed away" : "dropped"}</span>
        )}
        {method === "dare" && (
          <span><i className="mrg-swatch mrg-swatch--original" />height before rescaling</span>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Loss plane                                                                  */
/* -------------------------------------------------------------------------- */

interface PlanePoint {
  id: string;
  label: string;
  /** A two-letter tag drawn beside unselected methods so their labels do not collide. */
  tag?: string;
  losses: TaskLosses;
  kind: "base" | "a" | "b" | "ideal" | "current" | "other";
}

function LossPlane({
  points,
  linearPath,
  taskPath,
}: {
  points: PlanePoint[];
  linearPath: TaskLosses[];
  taskPath: TaskLosses[];
}) {
  const width = 660;
  const height = 340;
  const left = 58;
  const right = width - 18;
  const top = 18;
  const bottom = height - 46;
  const anchors = points.filter((point) => point.kind !== "other");
  const xs = [...anchors.map((point) => point.losses.recipes), ...linearPath.map((point) => point.recipes)];
  const ys = [...anchors.map((point) => point.losses.proverbs), ...linearPath.map((point) => point.proverbs)];
  const others = points.filter((point) => point.kind === "other");
  const anchorMaxX = Math.max(...xs);
  const anchorMaxY = Math.max(...ys);
  const xMin = Math.min(...xs, ...others.map((point) => point.losses.recipes)) - 0.04;
  const yMin = Math.min(...ys, ...others.map((point) => point.losses.proverbs)) - 0.04;
  // A wild merge (DARE at a high drop rate) is pinned to the edge rather than allowed to rescale the plot.
  const xMax = Math.min(Math.max(anchorMaxX, ...others.map((point) => point.losses.recipes)), anchorMaxX + 0.15) + 0.04;
  const yMax = Math.min(Math.max(anchorMaxY, ...others.map((point) => point.losses.proverbs)), anchorMaxY + 0.15) + 0.04;
  const rawX = (value: number) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
  const rawY = (value: number) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);
  const xAt = (value: number) => rawX(Math.min(xMax, value));
  const yAt = (value: number) => rawY(Math.min(yMax, value));
  const ticks = (low: number, high: number) => [0, 0.25, 0.5, 0.75, 1].map((f) => low + f * (high - low));
  const path = (list: TaskLosses[]) =>
    list
      .map((point, index) => `${index === 0 ? "M" : "L"}${rawX(point.recipes).toFixed(1)},${rawY(point.proverbs).toFixed(1)}`)
      .join(" ");
  const isClipped = (point: PlanePoint) => point.losses.recipes > xMax || point.losses.proverbs > yMax;

  const marker = (point: PlanePoint) => {
    const x = xAt(point.losses.recipes);
    const y = yAt(point.losses.proverbs);
    const clipped = isClipped(point);
    switch (point.kind) {
      case "base":
        return <circle className="mrg-plane__base" cx={x} cy={y} r={5} />;
      case "a":
        return <rect className="mrg-plane__a" x={x - 5} y={y - 5} width={10} height={10} />;
      case "b":
        return <path className="mrg-plane__b" d={`M${x},${y - 6} L${x + 6},${y + 5} L${x - 6},${y + 5} Z`} />;
      case "ideal":
        return (
          <g className="mrg-plane__ideal">
            <line x1={x - 5} x2={x + 5} y1={y - 5} y2={y + 5} />
            <line x1={x - 5} x2={x + 5} y1={y + 5} y2={y - 5} />
          </g>
        );
      case "current":
        return <circle className={`mrg-plane__current ${clipped ? "is-clipped" : ""}`} cx={x} cy={y} r={7} />;
      default:
        return <circle className={`mrg-plane__other ${clipped ? "is-clipped" : ""}`} cx={x} cy={y} r={4} />;
    }
  };

  // Greedy label placement: try four positions around each marker and keep the first that
  // overlaps no label or marker already placed. Selected merge and anchors go first.
  const order = [...points].sort((a, b) => (a.kind === "current" ? -1 : b.kind === "current" ? 1 : a.kind === "other" ? 1 : b.kind === "other" ? -1 : 0));
  const boxes: { x0: number; x1: number; y0: number; y1: number }[] = points.map((point) => ({
    x0: xAt(point.losses.recipes) - 6,
    x1: xAt(point.losses.recipes) + 6,
    y0: yAt(point.losses.proverbs) - 6,
    y1: yAt(point.losses.proverbs) + 6,
  }));
  const placed = order.map((point) => {
    const x = xAt(point.losses.recipes);
    const y = yAt(point.losses.proverbs);
    const name = point.kind === "other" ? point.tag ?? point.label : point.label;
    const text = isClipped(point)
      ? `${name} (${point.losses.recipes.toFixed(2)}, ${point.losses.proverbs.toFixed(2)}) off chart`
      : name;
    const charWidth = point.kind === "current" ? 7 : 6.2;
    const textWidth = text.length * charWidth;
    const candidates = [
      { dx: 10, dy: -8, anchor: "start" as const },
      { dx: 10, dy: 15, anchor: "start" as const },
      { dx: -10, dy: -8, anchor: "end" as const },
      { dx: -10, dy: 15, anchor: "end" as const },
      { dx: 0, dy: -13, anchor: "middle" as const },
      { dx: 0, dy: 22, anchor: "middle" as const },
    ];
    const boxOf = (candidate: (typeof candidates)[number]) => {
      const tx = x + candidate.dx;
      const x0 = candidate.anchor === "start" ? tx : candidate.anchor === "end" ? tx - textWidth : tx - textWidth / 2;
      return { x0, x1: x0 + textWidth, y0: y + candidate.dy - 10, y1: y + candidate.dy + 2 };
    };
    const fits = (box: { x0: number; x1: number; y0: number; y1: number }) =>
      box.x0 >= left - 4 &&
      box.x1 <= width - 2 &&
      box.y0 >= 2 &&
      !boxes.some((other) => box.x0 < other.x1 && box.x1 > other.x0 && box.y0 < other.y1 && box.y1 > other.y0);
    const preferred = x > right - textWidth - 12 ? [2, 3, 0, 1, 4, 5] : [0, 1, 2, 3, 4, 5];
    const choice = preferred.map((index) => candidates[index]).find((candidate) => fits(boxOf(candidate))) ?? candidates[preferred[0]];
    boxes.push(boxOf(choice));
    return { point, text, x: x + choice.dx, y: y + choice.dy, anchor: choice.anchor };
  });

  const describe = points
    .map((point) => `${point.label}: recipes ${point.losses.recipes.toFixed(3)}, proverbs ${point.losses.proverbs.toFixed(3)}`)
    .join("; ");

  const spread = Math.max(...points.map((point) => Math.abs(point.losses.recipes - points[0].losses.recipes) + Math.abs(point.losses.proverbs - points[0].losses.proverbs)));
  if (spread < 1e-6) {
    return (
      <p className="lab-note mrg-plane__empty">
        Every point is the base, at recipes {points[0].losses.recipes.toFixed(3)} and proverbs{" "}
        {points[0].losses.proverbs.toFixed(3)}: both task vectors are zero, so every merge adds nothing.
        Raise Recipe epochs or Proverb epochs.
      </p>
    );
  }

  return (
    <div className="mrg-plane">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Loss on both tasks for every merge. Lower and further left is better. ${describe}.`}>
        <defs>
          <clipPath id="mrg-plane-clip">
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {ticks(xMin, xMax).map((value) => (
          <g key={`x-${value}`}>
            <line className="mrg-plane__grid" x1={xAt(value)} x2={xAt(value)} y1={top} y2={bottom} />
            <text className="mrg-plane__tick" x={xAt(value)} y={bottom + 15} textAnchor="middle">
              {value.toFixed(2)}
            </text>
          </g>
        ))}
        {ticks(yMin, yMax).map((value) => (
          <g key={`y-${value}`}>
            <line className="mrg-plane__grid" x1={left} x2={right} y1={yAt(value)} y2={yAt(value)} />
            <text className="mrg-plane__tick" x={left - 7} y={yAt(value) + 3} textAnchor="end">
              {value.toFixed(2)}
            </text>
          </g>
        ))}
        <line className="mrg-plane__axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="mrg-plane__axis" x1={left} x2={left} y1={top} y2={bottom} />
        <text className="mrg-plane__axis-label" x={(left + right) / 2} y={height - 8} textAnchor="middle">
          recipe loss (nats/char) → worse
        </text>
        <text
          className="mrg-plane__axis-label"
          transform={`translate(14 ${(top + bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          proverb loss (nats/char) → worse
        </text>
        <g clipPath="url(#mrg-plane-clip)">
          <path className="mrg-plane__path mrg-plane__path--linear" d={path(linearPath)} />
          <path className="mrg-plane__path mrg-plane__path--task" d={path(taskPath)} />
        </g>
        {points.filter((point) => point.kind !== "current").map((point) => (
          <g key={point.id}>{marker(point)}</g>
        ))}
        {points.filter((point) => point.kind === "current").map((point) => (
          <g key={point.id}>{marker(point)}</g>
        ))}
        {placed.map((label) => (
          <text
            key={`label-${label.point.id}`}
            className={`mrg-plane__label mrg-plane__label--${label.point.kind}`}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
          >
            {label.text}
          </text>
        ))}
      </svg>
      <div className="mrg-legend" aria-hidden="true">
        <span><i className="mrg-key mrg-key--base" />harbor base</span>
        <span><i className="mrg-key mrg-key--a" />recipe specialist</span>
        <span><i className="mrg-key mrg-key--b" />proverb specialist</span>
        <span><i className="mrg-key mrg-key--ideal" />two separate models</span>
        <span><i className="mrg-key mrg-key--current" />selected merge</span>
        <span><i className="mrg-key mrg-key--other" />other methods at the same settings (Lin, TA, TIES, DARE, SL)</span>
        <span><i className="mrg-line mrg-line--linear" />linear blend, t from 0 to 1</span>
        <span><i className="mrg-line mrg-line--task" />task sum, λ from 0 to 1.5</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Explore                                                                     */
/* -------------------------------------------------------------------------- */

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const source: VectorSource = state.source === "lora" ? "lora" : "full";
  const epochsA = Math.round(clamp(state.epochsA, 0, 40, 20));
  const epochsB = Math.round(clamp(state.epochsB, 0, 40, 20));
  const method: MergeMethod = METHODS.some((entry) => entry.value === state.method)
    ? (state.method as MergeMethod)
    : "linear";
  const mix = clamp(state.mix, 0, 1, 0.5);
  const lambda = clamp(state.lambda, 0, 1.5, 1);
  const density = clamp(state.density, 0.05, 1, 0.2);
  const dropRate = clamp(state.dropRate, 0, 0.9, 0.5);
  const maskSeed = Math.round(clamp(state.maskSeed, 1, 999, 4));

  const vectors = useMemo(() => buildTaskVectors(source, epochsA, epochsB), [source, epochsA, epochsB]);
  const params: MergeParams = useMemo(
    () => ({ mix, lambda, density, dropRate, maskSeed }),
    [mix, lambda, density, dropRate, maskSeed],
  );

  const outcomes = useMemo(
    () => METHODS.map((entry) => ({ ...entry, outcome: lossesFor(vectors, entry.value, params) })),
    [vectors, params],
  );
  const current = outcomes.find((entry) => entry.value === method)!;
  const result =
    current.outcome.result ??
    mergeTaskVectors("linear", vectors.taskA, vectors.taskB, { ...params, mix: 0.5 })!;

  const paths = useMemo(() => {
    const linearPath: TaskLosses[] = [];
    const taskPath: TaskLosses[] = [];
    const fixed: MergeParams = { mix: 0.5, lambda: 1, density: 1, dropRate: 0, maskSeed: 1 };
    const served = (kind: MergeMethod, overrides: Partial<MergeParams>) =>
      lossesOf(
        addToBase(
          vectors.baseWeights,
          mergeTaskVectors(kind, vectors.taskA, vectors.taskB, { ...fixed, ...overrides })!.vector,
        ),
      );
    for (let step = 0; step <= 10; step += 1) linearPath.push(served("linear", { mix: step / 10 }));
    for (let step = 0; step <= 15; step += 1) taskPath.push(served("task", { lambda: step / 10 }));
    return { linearPath, taskPath };
  }, [vectors]);

  const stats = useMemo(() => signStats(vectors.taskA, vectors.taskB), [vectors]);
  const similarity = useMemo(() => cosine(vectors.taskA, vectors.taskB), [vectors]);
  const mapScale = useMemo(() => {
    let maximum = 0;
    for (let index = 0; index < vectors.taskA.length; index += 1) {
      maximum = Math.max(maximum, Math.abs(vectors.taskA[index]), Math.abs(vectors.taskB[index]));
    }
    return maximum;
  }, [vectors]);
  const coordinates = useMemo(() => pickCoordinates(vectors.taskA, vectors.taskB, 5), [vectors]);

  const methodLabel = METHODS.find((entry) => entry.value === method)!.label;
  const sourceLabel = source === "lora" ? `LoRA rank ${LORA_RANK}, alpha ${LORA_ALPHA}` : "full fine-tunes";

  const planePoints: PlanePoint[] = [
    { id: "base", label: "base", losses: vectors.base, kind: "base" },
    { id: "a", label: "recipe specialist", losses: vectors.specialistA, kind: "a" },
    { id: "b", label: "proverb specialist", losses: vectors.specialistB, kind: "b" },
    {
      id: "ideal",
      label: "two models",
      losses: { recipes: vectors.specialistA.recipes, proverbs: vectors.specialistB.proverbs },
      kind: "ideal",
    },
    ...outcomes
      .filter((entry) => entry.value !== method && entry.outcome.result)
      .map((entry) => ({
        id: entry.value,
        label: entry.label,
        tag: entry.tag,
        losses: { recipes: entry.outcome.recipes, proverbs: entry.outcome.proverbs },
        kind: "other" as const,
      })),
    {
      id: "current",
      label: methodLabel,
      losses: { recipes: current.outcome.recipes, proverbs: current.outcome.proverbs },
      kind: "current",
    },
  ];

  const formula = (() => {
    switch (method) {
      case "linear":
        return {
          expression: `θ = θ_base + ${(1 - mix).toFixed(2)}·τ_recipes + ${mix.toFixed(2)}·τ_proverbs`,
          detail:
            "Identical to averaging the two fine-tuned checkpoints with weights 1 − t and t, because both share θ_base.",
        };
      case "task":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·(τ_recipes + τ_proverbs)`,
          detail: "Both task vectors at full strength, then one shared scale λ. At λ = 0.5 this is the average.",
        };
      case "ties":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·mean_agreeing(elect(trim_${Math.round(density * 100)}%(τ_recipes), trim_${Math.round(density * 100)}%(τ_proverbs)))`,
          detail: `Trim keeps the largest ${Math.round(density * 100)}% of each vector by magnitude. Elect picks, per weight, the sign with more summed mass. The merge averages only the kept entries with that sign.`,
        };
      case "dare":
        return {
          expression: `θ = θ_base + ${lambda.toFixed(2)}·(m_r⊙τ_recipes + m_p⊙τ_proverbs) / (1 − ${dropRate.toFixed(2)})`,
          detail: `Each entry survives its mask with probability ${(1 - dropRate).toFixed(2)}; survivors are multiplied by ${(1 / (1 - dropRate)).toFixed(2)} so every entry keeps its expected value.`,
        };
      case "slerp":
        return {
          expression: `θ = θ_base + (sin(${(1 - mix).toFixed(2)}Ω)·τ_recipes + sin(${mix.toFixed(2)}Ω)·τ_proverbs) / sin Ω`,
          detail: Number.isFinite(result.angle)
            ? `Ω = ${result.angle.toFixed(1)}°, the angle between the two task vectors. The arc keeps the blend's length where a straight line would shrink it.`
            : "SLERP needs two non-zero task vectors. Train both before reading this one.",
        };
    }
  })();

  const interferenceNote = (() => {
    const a = current.outcome.interferenceA;
    const b = current.outcome.interferenceB;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      return "SLERP has no version with one task removed, so its interference is not separated here. Compare its point with Linear blend at the same t instead.";
    }
    const verdict = (value: number) =>
      value > 0.02 ? "hurt it" : value < -0.02 ? "helped it" : "barely changed it";
    return `Adding the proverb vector ${verdict(a)} on recipes (${signed(a)} nats), and adding the recipe vector ${verdict(b)} on proverbs (${signed(b)} nats). The rest of each gap to its specialist is the merge shrinking that task's own vector.`;
  })();

  const methodControls = (() => {
    const mixControl = (
      <RangeControl
        label="Blend t"
        min={0}
        max={1}
        step={0.05}
        value={mix}
        format={(value) => `${Math.round((1 - value) * 100)}% recipes · ${Math.round(value * 100)}% proverbs`}
        onChange={(value) => setState({ mix: value })}
      />
    );
    const lambdaControl = (
      <RangeControl
        label="Scale λ"
        min={0}
        max={1.5}
        step={0.05}
        value={lambda}
        format={(value) => value.toFixed(2)}
        onChange={(value) => setState({ lambda: value })}
      />
    );
    switch (method) {
      case "linear":
      case "slerp":
        return mixControl;
      case "task":
        return lambdaControl;
      case "ties":
        return (
          <>
            <RangeControl
              label="Keep top"
              min={0.05}
              max={1}
              step={0.05}
              value={density}
              format={(value) => `${Math.round(value * 100)}% of each vector`}
              onChange={(value) => setState({ density: value })}
            />
            {lambdaControl}
          </>
        );
      case "dare":
        return (
          <>
            <RangeControl
              label="Drop rate p"
              min={0}
              max={0.9}
              step={0.05}
              value={dropRate}
              format={(value) => `${Math.round(value * 100)}% dropped · ×${(1 / (1 - value)).toFixed(2)}`}
              onChange={(value) => setState({ dropRate: value })}
            />
            {lambdaControl}
            <button
              type="button"
              className="quiet-action mrg-reseed"
              onClick={() => {
                setState({ maskSeed: (maskSeed % 999) + 1 });
                narrate("New random drop masks.");
              }}
            >
              New random mask · #{maskSeed}
            </button>
          </>
        );
    }
  })();

  return (
    <div className="mrg-lab">
      <LabSurface label="Task vectors" className="mrg-vectors-card">
        <SurfaceHeading
          kicker={`Harbor base · both fine-tunes start from it · ${sourceLabel}`}
          title="Two task vectors: τ = θ_tuned − θ_base"
          aside={
            <span className="tg-badge">
              cosine {Number.isFinite(similarity) ? similarity.toFixed(2) : "—"}
            </span>
          }
        />
        <div className="mrg-controls">
          <SegmentedControl
            label="Task vectors from"
            value={source}
            options={[
              { value: "full", label: "Full fine-tunes" },
              { value: "lora", label: `LoRA rank ${LORA_RANK}` },
            ]}
            onChange={(value) => {
              setState({ source: value });
              narrate(value === "lora" ? "Task vectors from LoRA adapters." : "Task vectors from full fine-tunes.");
            }}
          />
          <RangeControl
            label="Recipe epochs"
            min={0}
            max={40}
            step={2}
            value={epochsA}
            format={(value) => (value === 0 ? "untrained · τ = 0" : `${value} passes`)}
            onChange={(value) => setState({ epochsA: value })}
          />
          <RangeControl
            label="Proverb epochs"
            min={0}
            max={40}
            step={2}
            value={epochsB}
            format={(value) => (value === 0 ? "untrained · τ = 0" : `${value} passes`)}
            onChange={(value) => setState({ epochsB: value })}
          />
        </div>
        <div className="mrg-maps">
          <TaskVectorMap
            label="τ recipes"
            caption={`‖τ‖ = ${norm(vectors.taskA).toFixed(2)} · rows: previous char, columns: next char`}
            values={vectors.taskA}
            other={vectors.taskB}
            scale={mapScale}
          />
          <TaskVectorMap
            label="τ proverbs"
            caption={`‖τ‖ = ${norm(vectors.taskB).toFixed(2)} · same layout, same colour scale`}
            values={vectors.taskB}
            other={vectors.taskA}
            scale={mapScale}
          />
        </div>
        <div className="mrg-legend" aria-hidden="true">
          <span><i className="mrg-swatch mrg-swatch--pos" />raises that logit</span>
          <span><i className="mrg-swatch mrg-swatch--neg" />lowers that logit</span>
          <span><i className="mrg-swatch mrg-swatch--zero" />unchanged</span>
          <span><i className="mrg-swatch mrg-swatch--conflict" />struck through: the two vectors disagree in sign</span>
          <span>colour depth = |entry| out of {mapScale.toFixed(2)}</span>
        </div>
        <div className="metric-row mrg-metrics">
          <Metric label="Recipe specialist, recipes" value={vectors.specialistA.recipes.toFixed(3)} tone="forward" />
          <Metric label="Proverb specialist, proverbs" value={vectors.specialistB.proverbs.toFixed(3)} tone="gradient" />
          <Metric label="Overlapping entries" value={`${stats.overlap} of ${V * V}`} />
          <Metric label="Sign conflicts" value={stats.overlap ? `${stats.conflicts} (${Math.round((stats.conflicts / stats.overlap) * 100)}%)` : "0"} tone="loss" />
        </div>
      </LabSurface>

      <LabSurface label="Merge recipe" className="mrg-recipe-card">
        <SurfaceHeading
          kicker="Arithmetic on the two vectors · no gradient step"
          title={`${methodLabel}, coordinate by coordinate`}
          aside={<span className="tg-badge">‖τ_merged‖ {norm(result.vector).toFixed(2)}</span>}
        />
        <div className="mrg-method">
          <SegmentedControl
            label="Merge method"
            value={method}
            options={METHODS.map((entry) => ({ value: entry.value, label: entry.label }))}
            onChange={(value) => {
              setState({ method: value });
              narrate(`${METHODS.find((entry) => entry.value === value)?.label ?? value} selected.`);
            }}
          />
          <div className="mrg-method__controls">{methodControls}</div>
        </div>
        <FormulaWithValues
          label="Served weights"
          expression={formula.expression}
          result={`recipes ${current.outcome.recipes.toFixed(3)} · proverbs ${current.outcome.proverbs.toFixed(3)}`}
          detail={formula.detail}
          tone="attention"
        />
        <p className="mrg-strip__title">
          The five largest sign conflicts and the five largest agreements between the two vectors, by |τ_recipes| + |τ_proverbs|.
        </p>
        <CoordinateStrip
          method={method}
          coordinates={coordinates}
          taskA={vectors.taskA}
          taskB={vectors.taskB}
          processed={result.processed}
          merged={result.vector}
          elected={result.elected}
        />
      </LabSurface>

      <LabSurface label="Two task losses" className="mrg-plane-card">
        <SurfaceHeading
          kicker="Measured on both corpora · lower left is better"
          title="Where each merge lands"
          aside={<span className="tg-badge">{methodLabel}</span>}
        />
        <LossPlane points={planePoints} linearPath={paths.linearPath} taskPath={paths.taskPath} />
        <table className="mrg-table">
          <caption>
            Every method at the current settings. Gap is merged loss minus that task&rsquo;s specialist;
            interference is merged loss minus the same merge with the other task vector set to zero.
          </caption>
          <thead>
            <tr>
              <th scope="col">Merge</th>
              <th scope="col">Recipes</th>
              <th scope="col">Gap</th>
              <th scope="col">Interference</th>
              <th scope="col">Proverbs</th>
              <th scope="col">Gap</th>
              <th scope="col">Interference</th>
            </tr>
          </thead>
          <tbody>
            {outcomes.map((entry) => (
              <tr key={entry.value} className={entry.value === method ? "is-selected" : ""} aria-current={entry.value === method ? "true" : undefined}>
                <th scope="row">
                  {entry.label}
                  {entry.value === method ? " ◆" : ""}
                </th>
                <td>{entry.outcome.recipes.toFixed(3)}</td>
                <td>{signed(entry.outcome.recipes - vectors.specialistA.recipes)}</td>
                <td>{signed(entry.outcome.interferenceA)}</td>
                <td>{entry.outcome.proverbs.toFixed(3)}</td>
                <td>{signed(entry.outcome.proverbs - vectors.specialistB.proverbs)}</td>
                <td>{signed(entry.outcome.interferenceB)}</td>
              </tr>
            ))}
            <tr className="is-reference">
              <th scope="row">Specialists</th>
              <td>{vectors.specialistA.recipes.toFixed(3)}</td>
              <td>—</td>
              <td>—</td>
              <td>{vectors.specialistB.proverbs.toFixed(3)}</td>
              <td>—</td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
        <p className="lab-note">{interferenceNote}</p>
      </LabSurface>
    </div>
  );
}
