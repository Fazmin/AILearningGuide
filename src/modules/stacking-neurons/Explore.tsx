import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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
  makeDataset,
  MAX_DEPTH,
  MAX_EPOCHS,
  MAX_WIDTH,
  MIN_DEPTH,
  MIN_WIDTH,
  NARROW_NET,
  NOISE_FLIPS,
  parameterCount,
  SHALLOW_WIDE_NET,
  sigmoid,
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

  // Which of the 60 ring points carry a flipped label in the run the held-out card trained on.
  const flipped = useMemo(() => {
    const clean = makeDataset("ring");
    const marked = new Set<number>();
    narrowSplit.data.forEach((point, index) => {
      if (point.label !== clean[index].label) marked.add(index);
    });
    return marked;
  }, [narrowSplit]);

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
  const diagramWidth = Math.max(320, measuredWidth);
  // Tiles are as large as the card allows: each one is a picture the lesson asks the reader to compare.
  const roomPerColumn = Math.floor((diagramWidth - 48) / columns) - 22;
  const tileSize = Math.max(34, Math.min(maxRows <= 2 ? 84 : maxRows <= 4 ? 62 : 46, roomPerColumn));
  const outputSize = Math.max(40, Math.min(maxRows <= 2 ? 96 : 72, roomPerColumn + 8));
  const rowGap = tileSize + 26;
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
        <ul className="nn-key" aria-label="Key to the decision surface">
          <li>
            <i className="nn-key__swatch nn-key__swatch--pos" />ŷ toward 1
          </li>
          <li>
            <i className="nn-key__swatch nn-key__swatch--neg" />ŷ toward 0
          </li>
          <li>
            <i className="nn-key__dot" />label 1
          </li>
          <li>
            <i className="nn-key__square" />label 0
          </li>
          <li>
            <i className="nn-key__ring" />wrong answer
          </li>
          <li>
            <i className="nn-key__dash" />h1…h{width}: one first-layer unit's sum is 0
          </li>
          <li>
            <i className="nn-key__cross" />probe
          </li>
        </ul>
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
        <HiddenSpace
          network={network}
          activation={activation}
          data={history.data}
          probe={probe}
          hiddenLayers={depth}
        >
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
        </HiddenSpace>
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
              <div className="nn-shape-row">
                <ShapeGlyph sizes={entry.history.sizes} />
                <MiniSurface
                  network={snapshotAt(entry.history, epochs)}
                  activation={activation}
                  data={entry.history.data}
                  markWrong
                  label={`${entry.title}, ${entry.shape.depth} layer${entry.shape.depth === 1 ? "" : "s"} of ${entry.shape.width} units, after ${epochs} epochs on ${DATASET_NAMES[dataset]}: ${percent(entry.history.accuracies[epochs])} of ${entry.history.data.length} training points correct.`}
                />
              </div>
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
        <ul className="nn-key" aria-label="Key to the two surfaces">
          <li>
            <i className="nn-key__dot" />label 1
          </li>
          <li>
            <i className="nn-key__square" />label 0
          </li>
          <li>
            <i className="nn-key__ring" />wrong answer
          </li>
          <li>columns of dots: the units in each layer</li>
        </ul>
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
            flipped={flipped}
            curve={heldCurve}
            epoch={epochs}
            lossCeiling={lossCeiling}
          />
          <SplitPanel
            title={`Large: ${LARGE_NET.depth} × ${LARGE_NET.width} units`}
            history={largeSplit}
            flipped={flipped}
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

/** A training point: a filled circle for label 1, an open square for label 0, a ring when it is answered wrong. */
function PointMark({
  cx,
  cy,
  label,
  wrong = false,
  flipped = false,
  r = 4,
}: {
  cx: number;
  cy: number;
  label: 0 | 1;
  wrong?: boolean;
  flipped?: boolean;
  r?: number;
}) {
  const half = r * 0.9;
  const reach = r * 2.1;
  return (
    <g>
      {flipped && (
        <path
          className="nn-point__flipped"
          d={`M ${cx} ${cy - reach} L ${cx + reach} ${cy} L ${cx} ${cy + reach} L ${cx - reach} ${cy} Z`}
        />
      )}
      {label === 1 ? (
        <circle className="nn-point__one" cx={cx} cy={cy} r={r} />
      ) : (
        <rect className="nn-point__zero" x={cx - half} y={cy - half} width={half * 2} height={half * 2} />
      )}
      {wrong && <circle className="nn-point__ring" cx={cx} cy={cy} r={r * 1.9} />}
    </g>
  );
}

/** A small decision surface for one network, with its training points drawn over it. */
function MiniSurface({
  network,
  activation,
  data,
  label,
  markWrong = false,
  flipped,
}: {
  network: Network;
  activation: HiddenActivation;
  data: readonly Example[];
  label: string;
  /** Ring every training point the network answers wrong. */
  markWrong?: boolean;
  /** Indexes of training points whose label was flipped; each gets a diamond. */
  flipped?: ReadonlySet<number>;
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
      {data.map((point, index) => (
        <PointMark
          key={index}
          cx={at(point.x1)}
          cy={atY(point.x2)}
          label={point.label}
          r={2.6}
          flipped={flipped?.has(index)}
          wrong={
            markWrong &&
            (forward(network, [point.x1, point.x2], activation).output >= 0.5 ? 1 : 0) !== point.label
          }
        />
      ))}
      <rect className="nn-frame" x={0} y={0} width={size} height={size} />
    </svg>
  );
}

/** A network's shape as columns of dots, one dot per unit, so wide-and-shallow reads differently from narrow-and-deep. */
function ShapeGlyph({ sizes }: { sizes: readonly number[] }) {
  const width = 92;
  const height = 150;
  const most = Math.max(...sizes);
  const pitch = Math.min(12, (height - 26) / most);
  const radius = Math.min(3, pitch * 0.36);
  const x = (layer: number) => 9 + (layer * (width - 18)) / (sizes.length - 1);
  const y = (count: number, index: number) => (height - 18) / 2 + (index - (count - 1) / 2) * pitch;
  return (
    <svg
      className="nn-shape"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Network shape: ${sizes[0]} inputs, ${sizes
        .slice(1, -1)
        .map((count) => `a hidden layer of ${count} units`)
        .join(", then ")}, 1 output.`}
    >
      {sizes.slice(1).map((count, layer) =>
        Array.from({ length: count }, (_, to) =>
          Array.from({ length: sizes[layer] }, (_, from) => (
            <line
              key={`${layer}-${to}-${from}`}
              className="nn-shape__edge"
              x1={x(layer)}
              y1={y(sizes[layer], from)}
              x2={x(layer + 1)}
              y2={y(count, to)}
            />
          )),
        ),
      )}
      {sizes.map((count, layer) =>
        Array.from({ length: count }, (_, index) => (
          <circle
            key={`${layer}-${index}`}
            className={`nn-shape__unit${layer > 0 && layer < sizes.length - 1 ? " is-hidden" : ""}`}
            cx={x(layer)}
            cy={y(count, index)}
            r={radius}
          />
        )),
      )}
      <text className="nn-shape__label" x={width / 2} y={height - 3} textAnchor="middle">
        {sizes.join(" → ")}
      </text>
    </svg>
  );
}

const HS = { width: 248, left: 34, top: 22, right: 14, bottom: 30 };
const HS_PLOT = HS.width - HS.left - HS.right;
const HS_HEIGHT = HS.top + HS_PLOT + HS.bottom;
const HS_GRID = 24;

/**
 * The training points as the output unit receives them. With two units in the last hidden layer the
 * points are drawn at their hidden values, to scale, with the output unit's straight cut. With any other
 * width they are drawn at the one number the output unit computes from them, z, on its sigmoid.
 */
function HiddenSpace({
  network,
  activation,
  data,
  probe,
  hiddenLayers,
  children,
}: {
  network: Network;
  activation: HiddenActivation;
  data: readonly Example[];
  probe: readonly [number, number];
  hiddenLayers: number;
  /** The probe sliders and the output formula, shown beside the plot under its caption. */
  children: ReactNode;
}) {
  const seen = useMemo(
    () =>
      data.map((point) => {
        const pass = forward(network, [point.x1, point.x2], activation);
        return {
          label: point.label,
          hidden: pass.post[pass.post.length - 2],
          z: pass.pre[pass.pre.length - 1][0],
          output: pass.output,
        };
      }),
    [network, activation, data],
  );
  const output = network.layers[network.layers.length - 1];
  const weights = output.weights[0];
  const bias = output.biases[0];
  const probePass = forward(network, probe, activation);
  const probeHidden = probePass.post[probePass.post.length - 2];
  const probeZ = probePass.pre[probePass.pre.length - 1][0];
  const isWrong = (point: (typeof seen)[number]) => (point.output >= 0.5 ? 1 : 0) !== point.label;
  const correct = seen.filter((point) => !isWrong(point)).length;
  const layerName = hiddenLayers === 1 ? "the hidden layer" : `hidden layer ${hiddenLayers}`;
  const title = "The same points, as the output unit sees them";
  const frame = <rect className="nn-frame" x={HS.left} y={HS.top} width={HS_PLOT} height={HS_PLOT} />;
  // Points that land on top of each other are counted, so a pile of twelve does not read as one.
  const clumpLabels = (spots: ReadonlyArray<{ x: number; y: number }>) => {
    const clumps: { x: number; y: number; count: number }[] = [];
    spots.forEach((spot) => {
      const found = clumps.find((clump) => Math.hypot(clump.x - spot.x, clump.y - spot.y) < 11);
      if (found) {
        found.x = (found.x * found.count + spot.x) / (found.count + 1);
        found.y = (found.y * found.count + spot.y) / (found.count + 1);
        found.count += 1;
      } else {
        clumps.push({ ...spot, count: 1 });
      }
    });
    const centreX = HS.left + HS_PLOT / 2;
    const centreY = HS.top + HS_PLOT / 2;
    // Largest piles first; a count that would sit on top of one already placed is left out.
    const placed: { x: number; y: number }[] = [];
    return clumps
      .filter((clump) => clump.count >= 5)
      .sort((a, b) => b.count - a.count)
      .map((clump) => {
        const x = clump.x + (clump.x < centreX ? 12 : -12);
        const y = clump.y + (clump.y < centreY ? 20 : -13);
        if (placed.some((other) => Math.abs(other.x - x) < 64 && Math.abs(other.y - y) < 13)) return null;
        placed.push({ x, y });
        return (
          <text
            key={`${clump.x.toFixed(0)}-${clump.y.toFixed(0)}`}
            className="nn-clump-label"
            x={x}
            y={y}
            textAnchor={clump.x < centreX ? "start" : "end"}
          >
            {clump.count} points
          </text>
        );
      });
  };
  const cross = (x: number, y: number) => (
    <g className="nn-probe">
      <line x1={x - 7} x2={x + 7} y1={y} y2={y} />
      <line x1={x} x2={x} y1={y - 7} y2={y + 7} />
    </g>
  );

  if (weights.length === 2) {
    const values = [...seen.flatMap((point) => point.hidden), ...probeHidden];
    const rawLow = Math.min(0, ...values);
    const rawHigh = Math.max(0, ...values);
    const span = Math.max(1e-6, rawHigh - rawLow);
    const [low, high] = activation === "tanh" ? [-1.15, 1.15] : [rawLow - span * 0.1, rawHigh + span * 0.1];
    const half = (high - low) / 2;
    const middle = (high + low) / 2;
    const px = (value: number) => HS.left + ((value - low) / (high - low)) * HS_PLOT;
    const py = (value: number) => HS.top + (1 - (value - low) / (high - low)) * HS_PLOT;
    // The cut w₁h₁ + w₂h₂ + c = 0, found in the plot's own [−1, 1]² coordinates.
    const cut = unitBoundary([weights[0] * half, weights[1] * half], bias + (weights[0] + weights[1]) * middle);
    const fromUnit = (value: number) => middle + half * value;
    // Name the cut at whichever of three places along it is farthest from the points.
    const cutLabel = cut
      ? [0.25, 0.5, 0.75]
          .map((t) => {
            const x = px(fromUnit(cut[0][0] + (cut[1][0] - cut[0][0]) * t));
            const y = py(fromUnit(cut[0][1] + (cut[1][1] - cut[0][1]) * t));
            const clear = Math.min(
              ...seen.map((point) => Math.hypot(px(point.hidden[0]) - x, py(point.hidden[1]) - y)),
            );
            return { x, y, clear };
          })
          .reduce((best, spot) => (spot.clear > best.clear ? spot : best))
      : { x: 0, y: 0, clear: 0 };
    const ticks =
      activation === "tanh"
        ? [-1, 0, 1]
        : [rawLow, 0, rawHigh].filter(
            (tick, index, all) => all.findIndex((other) => Math.abs(other - tick) < span * 0.14) === index,
          );
    const tickText = (tick: number) =>
      tick === 0
        ? "0"
        : `${tick < 0 ? "−" : ""}${activation === "tanh" ? Math.abs(tick) : Math.abs(tick).toFixed(1)}`;
    const cell = HS_PLOT / HS_GRID;
    return (
      <div className="nn-builder-lower">
        <svg
          className="nn-hidden__plot"
          viewBox={`0 0 ${HS.width} ${HS_HEIGHT}`}
          role="img"
          aria-label={`Hidden space. Each of the ${seen.length} training points is drawn at its two hidden values, h1 across and h2 up. The output unit's straight cut puts ${correct} of them on the correct side.`}
        >
          <defs>
            <clipPath id="nn-hidden-square">
              <rect x={HS.left} y={HS.top} width={HS_PLOT} height={HS_PLOT} />
            </clipPath>
          </defs>
          <g clipPath="url(#nn-hidden-square)">
            {Array.from({ length: HS_GRID * HS_GRID }, (_, index) => {
              const column = index % HS_GRID;
              const row = Math.floor(index / HS_GRID);
              const h1 = low + ((column + 0.5) / HS_GRID) * (high - low);
              const h2 = high - ((row + 0.5) / HS_GRID) * (high - low);
              const style = shade(sigmoid(weights[0] * h1 + weights[1] * h2 + bias), "output", 1);
              return (
                <rect
                  key={index}
                  className={`nn-cell nn-cell--${style.tone}`}
                  x={HS.left + column * cell}
                  y={HS.top + row * cell}
                  width={cell + 0.3}
                  height={cell + 0.3}
                  opacity={style.opacity}
                />
              );
            })}
            {cut && (
              <line
                className="nn-unit-line"
                x1={px(fromUnit(cut[0][0]))}
                y1={py(fromUnit(cut[0][1]))}
                x2={px(fromUnit(cut[1][0]))}
                y2={py(fromUnit(cut[1][1]))}
              />
            )}
          </g>
          {cut && (
            <text
              className="nn-unit-label"
              x={clamp(cutLabel.x, HS.left + 30, HS.left + HS_PLOT - 30)}
              y={clamp(cutLabel.y - 6, HS.top + 12, HS.top + HS_PLOT - 6)}
              textAnchor="middle"
            >
              ŷ = 0.5
            </text>
          )}
          {seen.map((point, index) => (
            <PointMark
              key={index}
              cx={px(point.hidden[0])}
              cy={py(point.hidden[1])}
              label={point.label}
              wrong={isWrong(point)}
              r={3.6}
            />
          ))}
          {clumpLabels(seen.map((point) => ({ x: px(point.hidden[0]), y: py(point.hidden[1]) })))}
          {cross(px(probeHidden[0]), py(probeHidden[1]))}
          {frame}
          {ticks.map((tick) => (
            <g key={tick}>
              <text className="nn-tick" x={px(tick)} y={HS.top + HS_PLOT + 13} textAnchor="middle">
                {tickText(tick)}
              </text>
              <text className="nn-tick" x={HS.left - 5} y={py(tick) + 3.5} textAnchor="end">
                {tickText(tick)}
              </text>
            </g>
          ))}
          <text className="nn-axis" x={HS.left + HS_PLOT} y={HS_HEIGHT - 2} textAnchor="end">
            h1 →
          </text>
          <text className="nn-axis" x={HS.left} y={HS.top - 8}>
            ↑ h2
          </text>
        </svg>
        <div className="nn-builder-probe">
          <p className="nn-hidden__caption">
            <strong>{title}</strong>
            <span>
              Each training point is redrawn at the two numbers {layerName} gives it. The dashed line is the output
              unit's one straight cut: {correct} of {seen.length} points are on their correct side.
            </span>
          </p>
          {children}
        </div>
      </div>
    );
  }

  const reach = Math.max(1, Math.abs(probeZ), ...seen.map((point) => Math.abs(point.z))) * 1.08;
  const px = (z: number) => HS.left + ((z + reach) / (2 * reach)) * HS_PLOT;
  const py = (value: number) => HS.top + 10 + (1 - value) * (HS_PLOT - 20);
  const strips = 30;
  const strip = HS_PLOT / strips;
  const curve = Array.from({ length: 81 }, (_, index) => {
    const z = -reach + (index / 80) * 2 * reach;
    return `${index === 0 ? "M" : "L"}${px(z).toFixed(1)} ${py(sigmoid(z)).toFixed(1)}`;
  }).join(" ");
  const edge = Math.max(1, Math.floor(reach));
  return (
    <div className="nn-builder-lower">
      <svg
        className="nn-hidden__plot"
        viewBox={`0 0 ${HS.width} ${HS_HEIGHT}`}
        role="img"
        aria-label={`Each of the ${seen.length} training points drawn at the output unit's weighted sum z, on the sigmoid curve that turns z into the output. ${correct} of them are on the correct side of z = 0.`}
      >
        {Array.from({ length: strips }, (_, index) => {
          const style = shade(sigmoid(-reach + ((index + 0.5) / strips) * 2 * reach), "output", 1);
          return (
            <rect
              key={index}
              className={`nn-cell nn-cell--${style.tone}`}
              x={HS.left + index * strip}
              y={HS.top}
              width={strip + 0.3}
              height={HS_PLOT}
              opacity={style.opacity * 0.55}
            />
          );
        })}
        <line className="nn-unit-line" x1={px(0)} x2={px(0)} y1={HS.top} y2={HS.top + HS_PLOT} />
        <path className="nn-hidden__curve" d={curve} />
        <text className="nn-unit-label" x={px(0) + 6} y={py(0.3)}>
          ŷ = 0.5
        </text>
        {seen.map((point, index) => (
          <PointMark
            key={index}
            cx={px(point.z)}
            cy={py(point.output)}
            label={point.label}
            wrong={isWrong(point)}
            r={3.6}
          />
        ))}
        {clumpLabels(seen.map((point) => ({ x: px(point.z), y: py(point.output) })))}
        {cross(px(probeZ), py(probePass.output))}
        {frame}
        {[-edge, 0, edge].map((tick) => (
          <text key={tick} className="nn-tick" x={px(tick)} y={HS.top + HS_PLOT + 13} textAnchor="middle">
            {tick < 0 ? `−${-tick}` : tick}
          </text>
        ))}
        {[0, 0.5, 1].map((tick) => (
          <text key={tick} className="nn-tick" x={HS.left - 5} y={py(tick) + 3.5} textAnchor="end">
            {tick}
          </text>
        ))}
        <text className="nn-axis" x={HS.left + HS_PLOT} y={HS_HEIGHT - 2} textAnchor="end">
          weighted sum z →
        </text>
        <text className="nn-axis" x={HS.left} y={HS.top - 8}>
          ↑ ŷ, the output
        </text>
      </svg>
      <div className="nn-builder-probe">
        <p className="nn-hidden__caption">
          <strong>{title}</strong>
          <span>
            The output unit turns the {weights.length === 1 ? "one hidden value" : `${weights.length} hidden values`}{" "}
            of each training point into one number, z, and cuts at z = 0: {correct} of {seen.length} points are on
            their correct side.
          </span>
        </p>
        {children}
      </div>
    </div>
  );
}

/** Training against held-out curves for one network, with the numbers at the chosen epoch. */
function SplitPanel({
  title,
  history,
  curve,
  epoch,
  lossCeiling,
  flipped,
}: {
  title: string;
  history: TrainHistory;
  curve: "accuracy" | "loss";
  epoch: number;
  /** Shared by both panels so their loss curves sit on one scale. */
  lossCeiling: number;
  /** Training points whose label was flipped (empty when the labels are clean). */
  flipped: ReadonlySet<number>;
}) {
  const heldAccuracies = history.heldAccuracies ?? [];
  const heldLosses = history.heldLosses ?? [];
  const accuracyMode = curve === "accuracy";
  const trainValues = accuracyMode ? history.accuracies : history.losses;
  const heldValues = accuracyMode ? heldAccuracies : heldLosses;
  const format = accuracyMode ? percent : (value: number) => fmt(value, 3);
  const trainAccuracy = history.accuracies[epoch];
  const heldAccuracy = heldAccuracies[epoch];
  const gap = trainAccuracy - heldAccuracy;
  return (
    <div className="nn-pair__item">
      <h4>
        {title} · {parameterCount(history.sizes)} parameters
      </h4>
      <MiniSurface
        network={snapshotAt(history, epoch)}
        activation="tanh"
        data={history.data}
        flipped={flipped}
        label={`${title}: its answer across the input square after ${epoch} epochs, with the 60 training points${flipped.size > 0 ? `, ${flipped.size} of them carrying a flipped label` : ""}. Training accuracy ${percent(trainAccuracy)}, held-out accuracy ${percent(heldAccuracy)}.`}
      />
      <p className="nn-mini-note">
        {flipped.size > 0 ? (
          <>
            <i className="nn-key__diamond" />a flipped training label. A pocket of the other colour around one means
            the network learned the mistake.
          </>
        ) : (
          "The 60 training points, all labelled correctly."
        )}
      </p>
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
      <dl className="nn-readout">
        <div className="nn-readout__row">
          <dt>Training accuracy</dt>
          <dd>
            <span className="nn-readout__track" aria-hidden="true">
              <i className="nn-readout__fill nn-readout__fill--train" style={{ width: `${trainAccuracy * 100}%` }} />
            </span>
            <strong className="nn-readout__value nn-readout__value--train">{percent(trainAccuracy)}</strong>
          </dd>
        </div>
        <div className="nn-readout__row">
          <dt>Held-out accuracy</dt>
          <dd>
            <span className="nn-readout__track" aria-hidden="true">
              <i className="nn-readout__fill nn-readout__fill--held" style={{ width: `${heldAccuracy * 100}%` }} />
              <i
                className="nn-readout__gap"
                style={{
                  left: `${Math.min(trainAccuracy, heldAccuracy) * 100}%`,
                  width: `${Math.abs(gap) * 100}%`,
                }}
              />
            </span>
            <strong className="nn-readout__value nn-readout__value--held">{percent(heldAccuracy)}</strong>
          </dd>
        </div>
        <div className="nn-readout__stat">
          <dt>
            <i className="nn-readout__gap-key" />
            Gap
          </dt>
          <dd>{points(gap)}</dd>
        </div>
        <div className="nn-readout__stat">
          <dt>Held-out loss</dt>
          <dd>{fmt(heldLosses[epoch])}</dd>
        </div>
      </dl>
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
  const [chartRef, measured] = useWidth(300);
  const width = Math.max(240, measured);
  const height = 164;
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
    <div ref={chartRef}>
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
    </div>
  );
}
