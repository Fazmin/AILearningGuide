import { useEffect, useMemo, useRef, useState } from "react";
import {
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  DEEP_NARROW_NET,
  forward,
  LARGE_NET,
  MAX_DEPTH,
  MAX_EPOCHS,
  MAX_WIDTH,
  MIN_DEPTH,
  MIN_WIDTH,
  NARROW_NET,
  NOISE_FLIPS,
  parameterCount,
  SHALLOW_WIDE_NET,
  snapshotAt,
  trainCached,
  trainRingSplit,
  unitBoundary,
  type DatasetKind,
  type Example,
  type HiddenActivation,
  type Network,
  type TrainHistory,
} from "./mlp";

const SURFACE_GRID = 40;
const MINI_GRID = 30;
const TILE_GRID = 14;
const PLOT = 300;
const PAD = 30;
const STEP_EPOCHS = 50;

const asNumber = (
  state: ModuleContext["state"],
  key: string,
  fallback: number,
) =>
  typeof state[key] === "number" && Number.isFinite(state[key])
    ? (state[key] as number)
    : fallback;
const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value));
const pick = <T extends string>(
  value: unknown,
  allowed: readonly T[],
  fallback: T,
): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
const fmt = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
/** A difference in percentage points with a true minus sign. */
const points = (value: number) =>
  Math.abs(value * 100) < 0.05
    ? "0.0 points"
    : `${value < 0 ? "−" : "+"}${Math.abs(value * 100).toFixed(1)} points`;
const DATASET_NAMES: Record<DatasetKind, string> = {
  xor: "XOR clusters",
  ring: "Ring",
  checker: "Checkerboard",
};
const ACTIVATION_NAMES: Record<HiddenActivation, string> = {
  tanh: "tanh",
  relu: "ReLU",
  linear: "no nonlinearity",
};
/** Two decimals with a true minus sign, parenthesized when negative so products read cleanly. */
const signedText = (value: number) =>
  value < 0 ? `(−${Math.abs(value).toFixed(2)})` : value.toFixed(2);

/** [−1, 1] → plot pixels. */
const sx = (x: number) => PAD + ((x + 1) / 2) * PLOT;
const sy = (y: number) => PAD + ((1 - y) / 2) * PLOT;

type Tone = "pos" | "neg" | "zero";

/** Track the rendered width so the diagram is drawn in CSS pixels at any card width. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry.contentRect.width);
      if (next > 0)
        setWidth((current) => (Math.abs(current - next) > 2 ? next : current));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Shade an activation for display: sign picks the tone, magnitude the opacity. */
function shade(
  value: number,
  kind: "tanh" | "relu" | "linear" | "output",
  scale: number,
): { tone: Tone; opacity: number } {
  if (kind === "output") {
    return {
      tone: value >= 0.5 ? "pos" : "neg",
      opacity: 0.08 + Math.abs(value - 0.5) * 2 * 0.72,
    };
  }
  if (kind === "relu") {
    return value > 0
      ? { tone: "pos", opacity: 0.12 + Math.min(1, value / scale) * 0.78 }
      : { tone: "zero", opacity: 1 };
  }
  const magnitude =
    kind === "tanh" ? Math.abs(value) : Math.min(1, Math.abs(value) / scale);
  return { tone: value >= 0 ? "pos" : "neg", opacity: 0.08 + magnitude * 0.8 };
}

/**
 * A saved network never changes, so what is computed from it can be reused by any later render of the
 * same epoch (the probe sliders re-render the card without retraining or resampling).
 */
const gridMemo = new WeakMap<Network, Map<string, unknown>>();
function memoGrid<T>(network: Network, key: string, build: () => T): T {
  let entries = gridMemo.get(network);
  if (!entries) {
    entries = new Map();
    gridMemo.set(network, entries);
  }
  if (!entries.has(key)) entries.set(key, build());
  return entries.get(key) as T;
}

/** Evaluate every unit of the network on an n × n grid over [−1, 1]². */
function sampleGrid(network: Network, activation: HiddenActivation, n: number) {
  return memoGrid(network, `units|${activation}|${n}`, () => sampleUnitGrid(network, activation, n));
}

function sampleUnitGrid(network: Network, activation: HiddenActivation, n: number) {
  const values: number[][][] = network.layers.map((layer) =>
    layer.biases.map(() => new Array(n * n).fill(0)),
  );
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      const x = -1 + ((column + 0.5) * 2) / n;
      const y = 1 - ((row + 0.5) * 2) / n;
      const pass = forward(network, [x, y], activation);
      pass.post.slice(1).forEach((layerValues, layerIndex) => {
        layerValues.forEach((value, unit) => {
          values[layerIndex][unit][row * n + column] = value;
        });
      });
    }
  }
  return values;
}

function Tile({
  x,
  y,
  size,
  values,
  kind,
  probe,
  label,
}: {
  x: number;
  y: number;
  size: number;
  values: number[];
  kind: "tanh" | "relu" | "linear" | "output" | "input1" | "input2";
  probe: readonly [number, number];
  label: string;
}) {
  const cell = size / TILE_GRID;
  const scale = Math.max(1e-6, ...values.map((value) => Math.abs(value)));
  return (
    <g>
      <title>{label}</title>
      <rect
        className="nn-tile__frame"
        x={x - 1}
        y={y - 1}
        width={size + 2}
        height={size + 2}
        rx={4}
      />
      {values.map((value, index) => {
        const column = index % TILE_GRID;
        const row = Math.floor(index / TILE_GRID);
        const style =
          kind === "input1" || kind === "input2"
            ? {
                tone: (value >= 0 ? "pos" : "neg") as Tone,
                opacity: 0.08 + Math.abs(value) * 0.6,
              }
            : shade(value, kind, scale);
        return (
          <rect
            key={index}
            className={`nn-cell nn-cell--${style.tone}`}
            x={x + column * cell}
            y={y + row * cell}
            width={cell + 0.3}
            height={cell + 0.3}
            opacity={style.opacity}
          />
        );
      })}
      <circle
        className="nn-tile__probe"
        cx={x + ((probe[0] + 1) / 2) * size}
        cy={y + ((1 - probe[1]) / 2) * size}
        r={2.2}
      />
    </g>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const dataset = pick(state.dataset, ["xor", "ring", "checker"] as const, "xor");
  const noisy = pick(state.noise, ["off", "on"] as const, "on") === "on";
  const heldCurve = pick(state.heldCurve, ["accuracy", "loss"] as const, "accuracy");
  const activation = pick(
    state.activation,
    ["tanh", "relu", "linear"] as const,
    "tanh",
  );
  const depth = clamp(
    Math.round(asNumber(state, "depth", 1)),
    MIN_DEPTH,
    MAX_DEPTH,
  );
  const width = clamp(
    Math.round(asNumber(state, "width", 2)),
    MIN_WIDTH,
    MAX_WIDTH,
  );
  const epochs = clamp(
    Math.round(asNumber(state, "epochs", 150)),
    0,
    MAX_EPOCHS,
  );
  const seed = clamp(Math.round(asNumber(state, "seed", 2)), 1, 999);
  const probeX = clamp(asNumber(state, "probeX", 0.6), -1, 1);
  const probeY = clamp(asNumber(state, "probeY", -0.6), -1, 1);
  const probe = [probeX, probeY] as const;
  const [diagramRef, measuredWidth] = useWidth(640);

  const history = useMemo(
    () => trainCached({ dataset, depth, width, activation, seed }),
    [dataset, depth, width, activation, seed],
  );
  const network = snapshotAt(history, epochs);
  const loss = history.losses[epochs];
  const acc = history.accuracies[epochs];
  const sizes = history.sizes;
  const params = parameterCount(sizes);

  const shallowWide = useMemo(
    () => trainCached({ dataset, ...SHALLOW_WIDE_NET, activation, seed }),
    [dataset, activation, seed],
  );
  const deepNarrow = useMemo(
    () => trainCached({ dataset, ...DEEP_NARROW_NET, activation, seed }),
    [dataset, activation, seed],
  );
  const narrowSplit = useMemo(
    () => trainRingSplit({ ...NARROW_NET, seed, noisy }),
    [seed, noisy],
  );
  const largeSplit = useMemo(
    () => trainRingSplit({ ...LARGE_NET, seed, noisy }),
    [seed, noisy],
  );

  const lossCeiling =
    Math.ceil(
      Math.max(
        0.8,
        ...narrowSplit.losses,
        ...(narrowSplit.heldLosses ?? []),
        ...largeSplit.losses,
        ...(largeSplit.heldLosses ?? []),
      ) * 5,
    ) / 5;

  const surface = useMemo(
    () => sampleGrid(network, activation, SURFACE_GRID),
    [network, activation],
  );
  const tiles = useMemo(
    () => sampleGrid(network, activation, TILE_GRID),
    [network, activation],
  );
  const inputTiles = useMemo(() => {
    const first: number[] = [];
    const second: number[] = [];
    for (let row = 0; row < TILE_GRID; row += 1) {
      for (let column = 0; column < TILE_GRID; column += 1) {
        first.push(-1 + ((column + 0.5) * 2) / TILE_GRID);
        second.push(1 - ((row + 0.5) * 2) / TILE_GRID);
      }
    }
    return [first, second];
  }, []);
  const pass = forward(network, probe, activation);
  const outputLayer = network.layers[network.layers.length - 1];
  const lastHidden = pass.post[pass.post.length - 2];
  const firstLayer = network.layers[0];
  const firstLines = firstLayer.weights.map((row, unit) =>
    unitBoundary(row, firstLayer.biases[unit]),
  );
  const outputs = surface[surface.length - 1][0];
  const firstPerfect = history.accuracies.findIndex((value) => value === 1);

  // Network diagram geometry.
  const columns = sizes.length;
  const maxRows = Math.max(...sizes);
  const tileSize = maxRows <= 2 ? 60 : maxRows <= 4 ? 50 : 42;
  const outputSize = maxRows <= 2 ? 68 : 58;
  const rowGap = tileSize + 26;
  const diagramWidth = Math.max(320, measuredWidth);
  const diagramHeight = Math.max(170, 34 + maxRows * rowGap);
  // When columns are close, print only the value under hidden tiles; the tile's title keeps the name.
  const compactLabels = (diagramWidth - 48 - outputSize) / (columns - 1) < 92;
  const columnX = (index: number) =>
    24 + (index * (diagramWidth - 48 - outputSize)) / (columns - 1);
  const nodeY = (count: number, index: number, size: number) =>
    28 +
    (diagramHeight - 28) / 2 +
    (index - (count - 1) / 2) * rowGap -
    size / 2 -
    7;
  const nodeBox = (layer: number, index: number) => {
    const size = layer === columns - 1 ? outputSize : tileSize;
    return { x: columnX(layer), y: nodeY(sizes[layer], index, size), size };
  };

  const setAndNarrate = (
    patch: Record<string, string | number>,
    message?: string,
  ) => {
    setState(patch);
    if (message) narrate(message);
  };

  return (
    <div className="nn-lab">
      <LabSurface label="Decision surface" className="nn-card">
        <SurfaceHeading
          kicker={`${DATASET_NAMES[dataset]} · ${history.data.length} training points · epoch ${epochs}`}
          title="The trained network's answer across the input square"
          aside={
            <span className={`nn-badge ${acc < 1 ? "is-warning" : ""}`.trim()}>
              {percent(acc)} correct
            </span>
          }
        />
        <svg
          className="nn-surface"
          viewBox={`0 0 ${PLOT + PAD * 2} ${PLOT + PAD * 2 + 8}`}
          role="img"
          aria-label={`Decision surface after ${epochs} epochs: ${percent(acc)} of ${history.data.length} training points correct, loss ${fmt(loss)}. ${width} first-layer boundary lines drawn.`}
        >
          <defs>
            <clipPath id="nn-square">
              <rect x={PAD} y={PAD} width={PLOT} height={PLOT} />
            </clipPath>
          </defs>
          <g clipPath="url(#nn-square)">
            {outputs.map((value, index) => {
              const column = index % SURFACE_GRID;
              const row = Math.floor(index / SURFACE_GRID);
              const style = shade(value, "output", 1);
              return (
                <rect
                  key={index}
                  className={`nn-cell nn-cell--${style.tone}`}
                  x={PAD + (column * PLOT) / SURFACE_GRID}
                  y={PAD + (row * PLOT) / SURFACE_GRID}
                  width={PLOT / SURFACE_GRID + 0.3}
                  height={PLOT / SURFACE_GRID + 0.3}
                  opacity={style.opacity}
                />
              );
            })}
            {firstLines.map((line, unit) =>
              line ? (
                <line
                  key={unit}
                  className="nn-unit-line"
                  x1={sx(line[0][0])}
                  y1={sy(line[0][1])}
                  x2={sx(line[1][0])}
                  y2={sy(line[1][1])}
                />
              ) : null,
            )}
          </g>
          {firstLines.map((line, unit) => {
            if (!line) return null;
            const end = line[1];
            const tx = clamp(sx(end[0]), PAD + 12, PAD + PLOT - 12);
            const ty = clamp(sy(end[1]), PAD + 12, PAD + PLOT - 6);
            return (
              <text
                key={`l-${unit}`}
                className="nn-unit-label"
                x={tx}
                y={ty}
                textAnchor="middle"
              >
                h{unit + 1}
              </text>
            );
          })}
          {history.data.map((point, index) => {
            const p = forward(network, [point.x1, point.x2], activation).output;
            const wrong = (p >= 0.5 ? 1 : 0) !== point.label;
            const cx = sx(point.x1);
            const cy = sy(point.x2);
            return (
              <g
                key={index}
                className={wrong ? "nn-point is-wrong" : "nn-point"}
              >
                {point.label === 1 ? (
                  <circle className="nn-point__one" cx={cx} cy={cy} r={4} />
                ) : (
                  <rect
                    className="nn-point__zero"
                    x={cx - 3.6}
                    y={cy - 3.6}
                    width={7.2}
                    height={7.2}
                  />
                )}
                {wrong && (
                  <circle className="nn-point__ring" cx={cx} cy={cy} r={7.5} />
                )}
              </g>
            );
          })}
          <g className="nn-probe">
            <line
              x1={sx(probeX) - 8}
              x2={sx(probeX) + 8}
              y1={sy(probeY)}
              y2={sy(probeY)}
            />
            <line
              x1={sx(probeX)}
              x2={sx(probeX)}
              y1={sy(probeY) - 8}
              y2={sy(probeY) + 8}
            />
          </g>
          <rect
            className="nn-frame"
            x={PAD}
            y={PAD}
            width={PLOT}
            height={PLOT}
          />
          {[-1, 0, 1].map((tick) => (
            <g key={tick}>
              <text
                className="nn-tick"
                x={sx(tick)}
                y={PAD + PLOT + 14}
                textAnchor="middle"
              >
                {tick}
              </text>
              <text
                className="nn-tick"
                x={PAD - 6}
                y={sy(tick) + 3.5}
                textAnchor="end"
              >
                {tick}
              </text>
            </g>
          ))}
          <text
            className="nn-axis"
            x={PAD + PLOT / 2}
            y={PAD + PLOT + 30}
            textAnchor="middle"
          >
            x₁ →
          </text>
          <text className="nn-axis" x={4} y={PAD - 10}>
            ↑ x₂
          </text>
        </svg>
        <p className="nn-legend">
          Shade is the output ŷ: green toward 1, red toward 0. Circles are label
          1, squares label 0; a ring marks a point the network gets wrong.
          Dashed lines h1…h{width} are where each first-layer unit's weighted
          sum is 0. The cross is the probe.
        </p>
      </LabSurface>

      <LabSurface label="Network training controls" className="nn-card">
        <SurfaceHeading
          kicker="Full-batch gradient descent · learning rate 0.5 · momentum 0.9"
          title="Change the stack, then train it for real"
        />
        <div className="nn-dataset-control">
          <SegmentedControl
            label="Dataset"
            value={dataset}
            options={[
              { value: "xor", label: "XOR" },
              { value: "ring", label: "Ring" },
              { value: "checker", label: "Checkerboard" },
            ]}
            onChange={(value) => setState({ dataset: value as DatasetKind })}
          />
        </div>
        <div className="nn-activation-control">
          <SegmentedControl
            label="Hidden activation"
            value={activation}
            options={[
              { value: "tanh", label: "tanh" },
              { value: "relu", label: "ReLU" },
              { value: "linear", label: "none (linear)" },
            ]}
            onChange={(value) =>
              setState({ activation: value as HiddenActivation })
            }
          />
        </div>
        <RangeControl
          label="Hidden layers"
          min={MIN_DEPTH}
          max={MAX_DEPTH}
          step={1}
          value={depth}
          onChange={(value) => setState({ depth: value })}
        />
        <RangeControl
          label="Neurons per layer"
          min={MIN_WIDTH}
          max={MAX_WIDTH}
          step={1}
          value={width}
          onChange={(value) => setState({ width: value })}
        />
        <RangeControl
          label="Training epochs"
          min={0}
          max={MAX_EPOCHS}
          step={1}
          value={epochs}
          onChange={(value) => setState({ epochs: value })}
        />
        <div className="nn-actions">
          <button
            type="button"
            className="primary-action"
            disabled={epochs >= MAX_EPOCHS}
            onClick={() => {
              const next = Math.min(MAX_EPOCHS, epochs + STEP_EPOCHS);
              setAndNarrate(
                { epochs: next },
                `Epoch ${next}. Loss ${fmt(history.losses[next])}, ${percent(history.accuracies[next])} correct.`,
              );
            }}
          >
            Train {STEP_EPOCHS} epochs
          </button>
          <button
            type="button"
            className="quiet-action"
            onClick={() =>
              setAndNarrate(
                { seed: seed + 1, epochs: 0 },
                `New random initialization, seed ${seed + 1}. Epoch 0.`,
              )
            }
          >
            New initialization
          </button>
        </div>
        <div className="metric-row">
          <Metric label="Training loss" value={fmt(loss)} tone="loss" />
          <Metric
            label="Accuracy"
            value={percent(acc)}
            tone={acc === 1 ? "forward" : undefined}
          />
          <Metric label="Parameters" value={String(params)} />
        </div>
        <LossChart
          losses={history.losses}
          accuracies={history.accuracies}
          epoch={epochs}
        />
        <p className="nn-legend">
          {firstPerfect >= 0
            ? `Seed ${seed}: every training point correct from epoch ${firstPerfect}. The curve is measured, not drawn.`
            : `Seed ${seed}: never reaches 100% within ${MAX_EPOCHS} epochs; best ${percent(Math.max(...history.accuracies))}. The curve is measured, not drawn.`}
        </p>
      </LabSurface>

      <LabSurface label="Network builder" className="nn-card nn-card--wide">
        <SurfaceHeading
          kicker={`${sizes.join(" → ")} units · ${params} parameters · ${activation === "linear" ? "no" : activation} activation`}
          title="Every unit's own map of the input square, and the forward pass at the probe"
        />
        <div ref={diagramRef}>
          <svg
            className="nn-diagram"
            viewBox={`0 0 ${diagramWidth} ${diagramHeight}`}
            role="img"
            aria-label={`Network ${sizes.join(", ")}. At probe (${fmt(probeX, 2)}, ${fmt(probeY, 2)}): ${pass.post
              .slice(1, -1)
              .map(
                (values, index) =>
                  `hidden layer ${index + 1} outputs ${values.map((value) => fmt(value, 2)).join(", ")}`,
              )
              .join("; ")}; output ${fmt(pass.output)}.`}
          >
            {network.layers.map((layer, layerIndex) =>
              layer.weights.map((row, unit) =>
                row.map((weight, input) => {
                  const from = nodeBox(layerIndex, input);
                  const to = nodeBox(layerIndex + 1, unit);
                  return (
                    <line
                      key={`${layerIndex}-${unit}-${input}`}
                      className={`nn-edge ${weight < 0 ? "is-negative" : "is-positive"}`}
                      x1={from.x + from.size}
                      y1={from.y + from.size / 2}
                      x2={to.x}
                      y2={to.y + to.size / 2}
                      strokeWidth={0.4 + Math.min(3, Math.abs(weight)) * 1.2}
                    />
                  );
                }),
              ),
            )}
            {sizes.map((count, layerIndex) =>
              Array.from({ length: count }, (_, index) => {
                const box = nodeBox(layerIndex, index);
                const isInput = layerIndex === 0;
                const isOutput = layerIndex === columns - 1;
                const value = pass.post[layerIndex][index];
                const name = isInput
                  ? `x${index === 0 ? "₁" : "₂"}`
                  : isOutput
                    ? "ŷ"
                    : `h${index + 1}`;
                return (
                  <g key={`${layerIndex}-${index}`}>
                    <Tile
                      x={box.x}
                      y={box.y}
                      size={box.size}
                      values={
                        isInput
                          ? inputTiles[index]
                          : tiles[layerIndex - 1][index]
                      }
                      kind={
                        isInput
                          ? index === 0
                            ? "input1"
                            : "input2"
                          : isOutput
                            ? "output"
                            : activation
                      }
                      probe={probe}
                      label={`${isInput ? "Input" : isOutput ? "Output" : `Layer ${layerIndex} unit ${index + 1}`}: ${fmt(value, 3)} at the probe`}
                    />
                    <text
                      className="nn-node-value"
                      x={box.x + box.size / 2}
                      y={box.y + box.size + 13}
                      textAnchor="middle"
                    >
                      {compactLabels && !isInput && !isOutput ? fmt(value, 2) : `${name} ${fmt(value, 2)}`}
                    </text>
                  </g>
                );
              }),
            )}
            {sizes.map((_, layerIndex) => (
              <text
                key={`c-${layerIndex}`}
                className="nn-column-label"
                x={
                  columnX(layerIndex) +
                  (layerIndex === columns - 1 ? outputSize : tileSize) / 2
                }
                y={16}
                textAnchor="middle"
              >
                {layerIndex === 0
                  ? "input"
                  : layerIndex === columns - 1
                    ? "output"
                    : `hidden ${layerIndex}`}
              </text>
            ))}
          </svg>
        </div>
        <div className="nn-probe-controls">
          <RangeControl
            label="Probe x₁"
            min={-1}
            max={1}
            step={0.05}
            value={probeX}
            format={(value) => fmt(value, 2)}
            onChange={(value) => setState({ probeX: value })}
          />
          <RangeControl
            label="Probe x₂"
            min={-1}
            max={1}
            step={0.05}
            value={probeY}
            format={(value) => fmt(value, 2)}
            onChange={(value) => setState({ probeY: value })}
          />
        </div>
        <FormulaWithValues
          label="Output unit at the probe"
          tone="forward"
          expression={`ŷ = σ(${outputLayer.weights[0]
            .map(
              (weight, index) =>
                `${signedText(weight)}·${signedText(lastHidden[index])}`,
            )
            .join(
              " + ",
            )} ${outputLayer.biases[0] < 0 ? "−" : "+"} ${fmt(Math.abs(outputLayer.biases[0]), 2)})`}
          result={fmt(pass.output)}
          detail={`z = ${fmt(pass.pre[pass.pre.length - 1][0], 2)}. Each tile shades one unit's output over the whole square; the dot inside it is the probe.`}
        />
      </LabSurface>

      <LabSurface label="Depth against width" className="nn-card nn-card--wide nn-width-card">
        <SurfaceHeading
          kicker={`${DATASET_NAMES[dataset]} · ${ACTIVATION_NAMES[activation]} · ${parameterCount(shallowWide.sizes)} parameters each · epoch ${epochs}`}
          title="The same parameter budget, shaped two ways"
          aside={
            <span
              className={`nn-badge ${deepNarrow.accuracies[epochs] > shallowWide.accuracies[epochs] ? "" : "is-warning"}`.trim()}
            >
              two layers minus one layer: {points(deepNarrow.accuracies[epochs] - shallowWide.accuracies[epochs])}
            </span>
          }
        />
        <div className="nn-pair">
          {[
            { key: "wide", title: "One wide layer", shape: SHALLOW_WIDE_NET, history: shallowWide },
            { key: "deep", title: "Two narrow layers", shape: DEEP_NARROW_NET, history: deepNarrow },
          ].map((entry) => (
            <div key={entry.key} className="nn-pair__item">
              <h4>
                {entry.title}: {entry.shape.depth} × {entry.shape.width} units
              </h4>
              <MiniSurface
                network={snapshotAt(entry.history, epochs)}
                activation={activation}
                data={entry.history.data}
                label={`${entry.title}, ${entry.shape.depth} layer${entry.shape.depth === 1 ? "" : "s"} of ${entry.shape.width} units, after ${epochs} epochs on ${DATASET_NAMES[dataset]}: ${percent(entry.history.accuracies[epochs])} of ${entry.history.data.length} training points correct.`}
              />
              <div className="metric-row">
                <Metric
                  label="Accuracy"
                  value={percent(entry.history.accuracies[epochs])}
                  tone={entry.history.accuracies[epochs] === 1 ? "forward" : undefined}
                />
                <Metric label="Training loss" value={fmt(entry.history.losses[epochs])} tone="loss" />
                <Metric label="Parameters" value={String(parameterCount(entry.history.sizes))} />
              </div>
            </div>
          ))}
        </div>
        <LineChart
          label="Training accuracy by epoch, one wide layer against two narrow layers"
          xLabel={`epoch, 0 → ${MAX_EPOCHS}`}
          yLabel="training accuracy"
          xDomain={[0, MAX_EPOCHS]}
          yDomain={[0, 1]}
          marker={{ x: epochs, label: `epoch ${epochs}` }}
          series={[
            {
              id: "wide",
              name: `1 × ${SHALLOW_WIDE_NET.width} at epoch ${MAX_EPOCHS}`,
              tone: "loss",
              dash: "dashed",
              points: shallowWide.accuracies.map((y, x) => ({ x, y })),
              format: percent,
            },
            {
              id: "deep",
              name: `2 × ${DEEP_NARROW_NET.width} at epoch ${MAX_EPOCHS}`,
              tone: "forward",
              dash: "solid",
              points: deepNarrow.accuracies.map((y, x) => ({ x, y })),
              format: percent,
            },
          ]}
        />
        <p className="nn-legend">
          Both networks start from the same seed and train for real with the controls above. The legend prints each
          curve's accuracy at epoch {MAX_EPOCHS}, the end of training; the metrics and surfaces show epoch {epochs}. On XOR and Ring both
          shapes reach 100%; the checkerboard is where they come apart.
        </p>
      </LabSurface>

      <LabSurface label="Held-out data" className="nn-card nn-card--wide nn-held-card">
        <SurfaceHeading
          kicker={`Ring · 60 training points · 120 held-out points · tanh · epoch ${epochs}`}
          title="Does the network learn the ring, or its labelling mistakes?"
          aside={
            <span className={`nn-badge ${noisy ? "is-warning" : ""}`.trim()}>
              {noisy ? `${NOISE_FLIPS} of 60 labels flipped` : "all labels correct"}
            </span>
          }
        />
        <div className="nn-held-controls">
          <SegmentedControl
            label="Training labels"
            value={noisy ? "on" : "off"}
            options={[
              { value: "off", label: "Clean" },
              { value: "on", label: "20% flipped" },
            ]}
            onChange={(value) => {
              setState({ noise: value });
              narrate(value === "on" ? `${NOISE_FLIPS} of 60 training labels flipped.` : "Training labels are clean.");
            }}
          />
          <SegmentedControl
            label="Curves"
            value={heldCurve}
            options={[
              { value: "accuracy", label: "Accuracy" },
              { value: "loss", label: "Loss" },
            ]}
            onChange={(value) => setState({ heldCurve: value })}
          />
        </div>
        <div className="nn-pair">
          <SplitPanel
            title={`Narrow: ${NARROW_NET.depth} × ${NARROW_NET.width} units`}
            history={narrowSplit}
            curve={heldCurve}
            epoch={epochs}
            lossCeiling={lossCeiling}
          />
          <SplitPanel
            title={`Large: ${LARGE_NET.depth} × ${LARGE_NET.width} units`}
            history={largeSplit}
            curve={heldCurve}
            epoch={epochs}
            lossCeiling={lossCeiling}
          />
        </div>
        <p className="nn-legend">
          Training points are the 60 on the Ring dataset. Held-out points are 120 fresh ones from the same ring that
          the networks never train on, and their labels are always correct. Solid curves are training; dashed curves
          are held-out. A gap opens when a network fits something in the training set that the held-out points do not
          repeat.
        </p>
      </LabSurface>
    </div>
  );
}

/** The output of a network over a coarse grid of the input square, row by row from the top. */
function outputGrid(network: Network, activation: HiddenActivation, n: number) {
  return memoGrid(network, `output|${activation}|${n}`, () => sampleOutputGrid(network, activation, n));
}

function sampleOutputGrid(network: Network, activation: HiddenActivation, n: number) {
  const values: number[] = [];
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      const x = -1 + ((column + 0.5) * 2) / n;
      const y = 1 - ((row + 0.5) * 2) / n;
      values.push(forward(network, [x, y], activation).output);
    }
  }
  return values;
}

/** A small decision surface for one network, with its training points drawn over it. */
function MiniSurface({
  network,
  activation,
  data,
  label,
}: {
  network: Network;
  activation: HiddenActivation;
  data: readonly Example[];
  label: string;
}) {
  const size = 150;
  const cells = useMemo(() => outputGrid(network, activation, MINI_GRID), [network, activation]);
  const at = (value: number) => ((value + 1) / 2) * size;
  const atY = (value: number) => ((1 - value) / 2) * size;
  return (
    <svg className="nn-mini" viewBox={`-2 -2 ${size + 4} ${size + 4}`} role="img" aria-label={label}>
      {cells.map((value, index) => {
        const style = shade(value, "output", 1);
        return (
          <rect
            key={index}
            className={`nn-cell nn-cell--${style.tone}`}
            x={(index % MINI_GRID) * (size / MINI_GRID)}
            y={Math.floor(index / MINI_GRID) * (size / MINI_GRID)}
            width={size / MINI_GRID + 0.3}
            height={size / MINI_GRID + 0.3}
            opacity={style.opacity}
          />
        );
      })}
      {data.map((point, index) =>
        point.label === 1 ? (
          <circle key={index} className="nn-point__one" cx={at(point.x1)} cy={atY(point.x2)} r={2.6} />
        ) : (
          <rect
            key={index}
            className="nn-point__zero"
            x={at(point.x1) - 2.2}
            y={atY(point.x2) - 2.2}
            width={4.4}
            height={4.4}
          />
        ),
      )}
      <rect className="nn-frame" x={0} y={0} width={size} height={size} />
    </svg>
  );
}

/** Training against held-out curves for one network, with the numbers at the chosen epoch. */
function SplitPanel({
  title,
  history,
  curve,
  epoch,
  lossCeiling,
}: {
  title: string;
  history: TrainHistory;
  curve: "accuracy" | "loss";
  epoch: number;
  /** Shared by both panels so their loss curves sit on one scale. */
  lossCeiling: number;
}) {
  const heldAccuracies = history.heldAccuracies ?? [];
  const heldLosses = history.heldLosses ?? [];
  const accuracyMode = curve === "accuracy";
  const trainValues = accuracyMode ? history.accuracies : history.losses;
  const heldValues = accuracyMode ? heldAccuracies : heldLosses;
  const format = accuracyMode ? percent : (value: number) => fmt(value, 3);
  const trainAccuracy = history.accuracies[epoch];
  const heldAccuracy = heldAccuracies[epoch];
  return (
    <div className="nn-pair__item">
      <h4>
        {title} · {parameterCount(history.sizes)} parameters
      </h4>
      <LineChart
        label={`${title}: ${accuracyMode ? "accuracy" : "loss"} on training points and on held-out points by epoch`}
        xLabel={`epoch, 0 → ${MAX_EPOCHS}`}
        yLabel={accuracyMode ? "accuracy" : "loss"}
        xDomain={[0, MAX_EPOCHS]}
        yDomain={accuracyMode ? [0, 1] : [0, lossCeiling]}
        marker={{ x: epoch, label: `epoch ${epoch}` }}
        series={[
          {
            id: "train",
            name: `training at epoch ${MAX_EPOCHS}`,
            tone: "forward",
            dash: "solid",
            points: trainValues.map((y, x) => ({ x, y })),
            format,
          },
          {
            id: "held",
            name: `held-out at epoch ${MAX_EPOCHS}`,
            tone: "loss",
            dash: "dashed",
            points: heldValues.map((y, x) => ({ x, y })),
            format,
          },
        ]}
      />
      <div className="metric-row">
        <Metric label="Training accuracy" value={percent(trainAccuracy)} tone="forward" />
        <Metric label="Held-out accuracy" value={percent(heldAccuracy)} tone="loss" />
        <Metric label="Gap" value={points(trainAccuracy - heldAccuracy)} />
        <Metric label="Held-out loss" value={fmt(heldLosses[epoch])} />
      </div>
    </div>
  );
}

function LossChart({
  losses,
  accuracies,
  epoch,
}: {
  losses: readonly number[];
  accuracies: readonly number[];
  epoch: number;
}) {
  const width = 340;
  const height = 158;
  const left = 34;
  const right = width - 10;
  const top = 20;
  const bottom = height - 26;
  const yMax = Math.max(0.8, ...losses);
  const xAt = (value: number) => left + (value / MAX_EPOCHS) * (right - left);
  const yAt = (value: number) =>
    bottom - (Math.min(yMax, value) / yMax) * (bottom - top);
  const path = losses
    .map(
      (value, index) =>
        `${index === 0 ? "M" : "L"}${xAt(index).toFixed(1)} ${yAt(value).toFixed(1)}`,
    )
    .join(" ");
  const firstPerfect = accuracies.findIndex((value) => value === 1);
  const current = losses[epoch];
  return (
    <svg
      className="nn-loss"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Training loss by epoch, from ${fmt(losses[0])} at epoch 0 to ${fmt(losses[losses.length - 1])} at epoch ${MAX_EPOCHS}. Marker at epoch ${epoch}, loss ${fmt(current)}.`}
    >
      {[0, yMax / 2, yMax].map((tick) => (
        <g key={tick}>
          <line
            className="nn-loss__grid"
            x1={left}
            x2={right}
            y1={yAt(tick)}
            y2={yAt(tick)}
          />
          <text
            className="nn-tick"
            x={left - 5}
            y={yAt(tick) + 3.5}
            textAnchor="end"
          >
            {tick.toFixed(tick === 0 ? 0 : 1)}
          </text>
        </g>
      ))}
      {[0, 100, 200, 300].map((tick) => (
        <text
          key={tick}
          className="nn-tick"
          x={xAt(tick)}
          y={bottom + 13}
          textAnchor="middle"
        >
          {tick}
        </text>
      ))}
      <line
        className="nn-loss__ln2"
        x1={left}
        x2={right}
        y1={yAt(Math.LN2)}
        y2={yAt(Math.LN2)}
      />
      <text
        className="nn-loss__ln2-label"
        x={right}
        y={yAt(Math.LN2) - 4}
        textAnchor="end"
      >
        0.693 = always 50%
      </text>
      {firstPerfect >= 0 && (
        <line
          className="nn-loss__perfect"
          x1={xAt(firstPerfect)}
          x2={xAt(firstPerfect)}
          y1={top}
          y2={bottom}
        />
      )}
      <path className="nn-loss__line" d={path} />
      <line
        className="nn-loss__marker"
        x1={xAt(epoch)}
        x2={xAt(epoch)}
        y1={top}
        y2={bottom}
      />
      <circle
        className="nn-loss__dot"
        cx={xAt(epoch)}
        cy={yAt(current)}
        r={4}
      />
      <text className="nn-axis" x={right} y={height - 1} textAnchor="end">
        epoch →
      </text>
      <text className="nn-axis" x={2} y={10}>
        loss
      </text>
    </svg>
  );
}
