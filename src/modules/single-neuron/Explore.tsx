import { useMemo } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  activate,
  cornersCorrect,
  cornersFor,
  footOnBoundary,
  levelChord,
  signedDistance,
  slope,
  weightedSum,
  weightNorm,
  type ActivationKind,
  type CornerTask,
} from "./neuron";

const GRID = 30;
const PLANE = 300;
const PAD = 34;
const CONTOURS = [-2, -1, 1, 2];

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pick = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
const fmt = (value: number, digits = 3) =>
  Number.isFinite(value) ? (Object.is(Math.round(value * 10 ** digits), -0) ? 0 : value).toFixed(digits) : "—";
const signed = (value: number, digits = 2) => (value < 0 ? `(−${fmt(-value, digits)})` : fmt(value, digits));

const ACTIVATION_NAME: Record<ActivationKind, string> = { sigmoid: "σ", relu: "ReLU", step: "step" };
const ACTIVATION_FORMULA: Record<ActivationKind, string> = {
  sigmoid: "1 / (1 + e^−z)",
  relu: "max(0, z)",
  step: "1 if z > 0, else 0",
};

/** Plane coordinates: x₁ left → right, x₂ bottom → top, both 0 to 1. */
const px = (x1: number) => PAD + x1 * PLANE;
const py = (x2: number) => PAD + (1 - x2) * PLANE;

function ActivationGlyph({ kind, x, y }: { kind: ActivationKind; x: number; y: number }) {
  const path =
    kind === "sigmoid"
      ? `M ${x - 12} ${y + 7} C ${x - 3} ${y + 7} ${x + 3} ${y - 7} ${x + 12} ${y - 7}`
      : kind === "relu"
        ? `M ${x - 12} ${y + 6} L ${x} ${y + 6} L ${x + 12} ${y - 8}`
        : `M ${x - 12} ${y + 6} L ${x} ${y + 6} L ${x} ${y - 7} L ${x + 12} ${y - 7}`;
  return <path className="sn-glyph" d={path} />;
}

export default function Explore({ state, setState }: ModuleContext) {
  const x1 = clamp(asNumber(state, "x1", 0.7), 0, 1);
  const x2 = clamp(asNumber(state, "x2", 0.35), 0, 1);
  const w1 = clamp(asNumber(state, "w1", 1.2), -2, 2);
  const w2 = clamp(asNumber(state, "w2", -0.8), -2, 2);
  const bias = clamp(asNumber(state, "bias", -0.1), -2, 2);
  const activation = pick(state.activation, ["sigmoid", "relu", "step"] as const, "sigmoid");
  const task = pick(state.task, ["off", "and", "or", "xor"] as const, "off");
  const params = { w1, w2, bias };

  const z = weightedSum(x1, x2, params);
  const a = activate(z, activation);
  const gradient = slope(z, activation);
  const norm = weightNorm(params);
  const distance = signedDistance(x1, x2, params);
  const boundary = levelChord(params);
  const foot = footOnBoundary(x1, x2, params);
  const corners = cornersFor(task);
  const correct = cornersCorrect(task, params);

  const reluCeiling = Math.max(
    1e-9,
    ...[
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ].map(([u, v]) => Math.max(0, weightedSum(u, v, params))),
  );

  const cells = useMemo(() => {
    const tiles: Array<{ x: number; y: number; tone: "pos" | "neg" | "zero"; opacity: number }> = [];
    for (let row = 0; row < GRID; row += 1) {
      for (let column = 0; column < GRID; column += 1) {
        const u = (column + 0.5) / GRID;
        const v = 1 - (row + 0.5) / GRID;
        const value = weightedSum(u, v, { w1, w2, bias });
        const out = activate(value, activation);
        let tone: "pos" | "neg" | "zero" = value > 0 ? "pos" : "neg";
        let opacity = 0.5;
        if (activation === "sigmoid") opacity = 0.08 + Math.abs(out - 0.5) * 2 * 0.72;
        if (activation === "relu") {
          if (value > 0) opacity = 0.1 + Math.min(1, out / reluCeiling) * 0.7;
          else {
            tone = "zero";
            opacity = 1;
          }
        }
        if (activation === "step") opacity = 0.42;
        tiles.push({ x: column, y: row, tone, opacity });
      }
    }
    return tiles;
  }, [w1, w2, bias, activation, reluCeiling]);

  // Weight-vector arrow: drawn from the boundary point nearest the square's centre.
  const centreFoot = footOnBoundary(0.5, 0.5, params);
  const anchor =
    centreFoot && centreFoot[0] >= 0 && centreFoot[0] <= 1 && centreFoot[1] >= 0 && centreFoot[1] <= 1
      ? centreFoot
      : ([0.5, 0.5] as const);
  const arrowLength = norm > 1e-9 ? 0.07 + 0.09 * norm : 0;
  const arrowEnd =
    norm > 1e-9 ? ([anchor[0] + (w1 / norm) * arrowLength, anchor[1] + (w2 / norm) * arrowLength] as const) : null;

  const regionLegend =
    activation === "sigmoid"
      ? "Shade is σ(z): deeper green toward 1, deeper red toward 0, pale near 0.5 on the line."
      : activation === "relu"
        ? "Shade is ReLU(z): green deepens as the output grows; the grey side outputs exactly 0."
        : "Shade is step(z): flat green outputs 1, flat red outputs 0, with nothing in between.";

  const edges = [
    { key: "x1", label: "x₁", value: x1, weight: w1, product: x1 * w1, y: 42 },
    { key: "x2", label: "x₂", value: x2, weight: w2, product: x2 * w2, y: 152 },
  ];
  const sumX = 176;
  const sumY = 97;

  return (
    <div className="sn-lab">
      <LabSurface label="Decision boundary and controls" className="sn-card sn-card--wide">
        <SurfaceHeading
          kicker="Input plane · x₁ and x₂ from 0 to 1"
          title="One straight line where z = 0, turned by w and shifted by b"
          aside={<span className="sn-badge">z = {fmt(z)}</span>}
        />
        <div className="sn-boundary">
          <div className="sn-plane-wrap">
            <svg
              className="sn-plane"
              viewBox={`0 0 ${PLANE + PAD * 2} ${PLANE + PAD * 2 + 12}`}
              role="img"
              aria-label={`Decision plane. Boundary z = 0 ${boundary ? "crosses the square" : "misses the square"}. Probe at x₁ ${fmt(x1, 2)}, x₂ ${fmt(x2, 2)} has z ${fmt(z)}, ${z > 0 ? "positive side" : "negative side"}, distance ${fmt(distance)} from the line.${task === "off" ? "" : ` ${task.toUpperCase()} corners correct: ${correct} of 4.`}`}
            >
              <defs>
                <clipPath id="sn-square">
                  <rect x={PAD} y={PAD} width={PLANE} height={PLANE} />
                </clipPath>
                <marker id="sn-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M0 0 L10 5 L0 10 z" className="sn-arrowhead" />
                </marker>
              </defs>
              <g clipPath="url(#sn-square)">
                {cells.map((cell) => (
                  <rect
                    key={`${cell.x}-${cell.y}`}
                    className={`sn-cell sn-cell--${cell.tone}`}
                    x={PAD + (cell.x * PLANE) / GRID}
                    y={PAD + (cell.y * PLANE) / GRID}
                    width={PLANE / GRID + 0.4}
                    height={PLANE / GRID + 0.4}
                    opacity={cell.opacity}
                  />
                ))}
                {CONTOURS.map((level) => {
                  const chord = levelChord(params, level, -0.5, 1.5);
                  if (!chord) return null;
                  return (
                    <g key={level}>
                      <line
                        className="sn-contour"
                        x1={px(chord[0][0])}
                        y1={py(chord[0][1])}
                        x2={px(chord[1][0])}
                        y2={py(chord[1][1])}
                      />
                    </g>
                  );
                })}
                {boundary && (
                  <line
                    className="sn-boundary-line"
                    x1={px(boundary[0][0])}
                    y1={py(boundary[0][1])}
                    x2={px(boundary[1][0])}
                    y2={py(boundary[1][1])}
                  />
                )}
                {foot && (
                  <line className="sn-perp" x1={px(x1)} y1={py(x2)} x2={px(foot[0])} y2={py(foot[1])} />
                )}
                {arrowEnd && (
                  <g>
                    <line
                      className="sn-weight-arrow"
                      x1={px(anchor[0])}
                      y1={py(anchor[1])}
                      x2={px(arrowEnd[0])}
                      y2={py(arrowEnd[1])}
                      markerEnd="url(#sn-arrow)"
                    />
                    <text className="sn-weight-label" x={px(arrowEnd[0]) + 6} y={py(arrowEnd[1]) - 4}>
                      w
                    </text>
                  </g>
                )}
              </g>
              <rect className="sn-frame" x={PAD} y={PAD} width={PLANE} height={PLANE} />
              {[0, 0.5, 1].map((tick) => (
                <g key={tick}>
                  <text className="sn-tick" x={px(tick) + (tick === 0 ? 12 : tick === 1 ? -12 : 0)} y={PAD + PLANE + 18} textAnchor="middle">
                    {tick}
                  </text>
                  <text className="sn-tick" x={PAD - 12} y={py(tick) + 3.5 + (tick === 0 ? -12 : tick === 1 ? 12 : 0)} textAnchor="end">
                    {tick}
                  </text>
                </g>
              ))}
              <text className="sn-axis" x={PAD + PLANE / 2} y={PAD + PLANE + 32} textAnchor="middle">
                x₁ →
              </text>
              <text className="sn-axis" x={PAD - 26} y={PAD - 14} textAnchor="start">
                ↑ x₂
              </text>
              {corners.map((corner) => {
                const predicted = weightedSum(corner.x1, corner.x2, params) > 0 ? 1 : 0;
                const ok = predicted === corner.target;
                const cx = px(corner.x1);
                const cy = py(corner.x2);
                const dx = corner.x1 === 0 ? 13 : -13;
                return (
                  <g key={`${corner.x1}${corner.x2}`} className={`sn-corner ${ok ? "is-right" : "is-wrong"}`}>
                    {corner.target === 1 ? (
                      <circle cx={cx} cy={cy} r={7.5} className="sn-corner__one" />
                    ) : (
                      <rect x={cx - 7} y={cy - 7} width={14} height={14} className="sn-corner__zero" />
                    )}
                    <text x={cx + dx} y={cy + (corner.x2 === 0 ? -9 : 17)} textAnchor={corner.x1 === 0 ? "start" : "end"}>
                      {corner.target} {ok ? "✓" : "✗"}
                    </text>
                  </g>
                );
              })}
              <circle className="sn-probe" cx={px(x1)} cy={py(x2)} r={6} />
            </svg>
            <p className="sn-legend">{regionLegend} Thin dashed lines are z = ±1 and ±2, spaced 1/‖w‖ apart.</p>
          </div>
          <div className="sn-controls">
            <SegmentedControl
              label="Activation"
              value={activation}
              options={[
                { value: "sigmoid", label: "Sigmoid" },
                { value: "relu", label: "ReLU" },
                { value: "step", label: "Step" },
              ]}
              onChange={(value) => setState({ activation: value })}
            />
            <RangeControl label="Input x₁" min={0} max={1} step={0.01} value={x1} format={(v) => fmt(v, 2)} onChange={(v) => setState({ x1: v })} />
            <RangeControl label="Input x₂" min={0} max={1} step={0.01} value={x2} format={(v) => fmt(v, 2)} onChange={(v) => setState({ x2: v })} />
            <RangeControl label="Weight w₁" min={-2} max={2} step={0.1} value={w1} format={(v) => fmt(v, 1)} onChange={(v) => setState({ w1: v })} />
            <RangeControl label="Weight w₂" min={-2} max={2} step={0.1} value={w2} format={(v) => fmt(v, 1)} onChange={(v) => setState({ w2: v })} />
            <RangeControl label="Bias b" min={-2} max={2} step={0.1} value={bias} format={(v) => fmt(v, 1)} onChange={(v) => setState({ bias: v })} />
            <div className="sn-task">
              <SegmentedControl
                label="Corner task"
                value={task}
                options={[
                  { value: "off", label: "Off" },
                  { value: "and", label: "AND" },
                  { value: "or", label: "OR" },
                  { value: "xor", label: "XOR" },
                ]}
                onChange={(value) => setState({ task: value as CornerTask })}
              />
            </div>
          </div>
        </div>
        <div className="metric-row">
          <Metric label="‖w‖" value={fmt(norm)} />
          <Metric label="Distance to boundary" value={Number.isFinite(distance) ? fmt(distance) : "no line"} tone="forward" />
          <Metric
            label="Corners correct"
            value={task === "off" ? "—" : `${correct} / 4`}
            tone={task !== "off" && correct < 4 ? "loss" : undefined}
          />
        </div>
        <p className="lab-note">
          {task === "xor"
            ? `XOR wants the two diagonal corners on opposite sides of one straight line. No w₁, w₂ and b can do that, so the best any setting reaches is 3 / 4. You have ${correct} / 4.`
            : task === "off"
              ? "Distance to boundary is z / ‖w‖. The arrow w is perpendicular to the line and points toward positive z; the dashed segment is the probe's shortest path to the line."
              : `The neuron predicts 1 where z > 0. ${task.toUpperCase()} is linearly separable, so some setting of the three sliders scores 4 / 4. You have ${correct} / 4.`}
        </p>
      </LabSurface>

      <LabSurface label="Neuron equation" className="sn-card">
        <SurfaceHeading kicker="Forward pass, evaluated live" title="Multiply, add the bias, then activate" />
        <svg
          className="sn-diagram"
          viewBox="0 0 340 196"
          role="img"
          aria-label={`x₁ ${fmt(x1, 2)} times w₁ ${fmt(w1, 1)} is ${fmt(x1 * w1)}. x₂ ${fmt(x2, 2)} times w₂ ${fmt(w2, 1)} is ${fmt(x2 * w2)}. Plus bias ${fmt(bias, 1)} gives z ${fmt(z)}. ${activation} of z is ${fmt(a)}.`}
        >
          {edges.map((edge) => {
            const upper = edge.y < sumY;
            return (
              <g key={edge.key}>
                <line
                  className={`sn-edge ${edge.weight < 0 ? "is-negative" : "is-positive"}`}
                  x1={44}
                  y1={edge.y}
                  x2={sumX - 24}
                  y2={sumY + (edge.y - sumY) * 0.3}
                  strokeWidth={1 + Math.abs(edge.weight) * 2.2}
                />
                <text className="sn-edge-label" x={52} y={upper ? edge.y - 12 : edge.y + 22}>
                  × {fmt(edge.weight, 1)} = {fmt(edge.product)}
                </text>
                <circle className="sn-node" cx={24} cy={edge.y} r={20} />
                <text className="sn-node-name" x={24} y={edge.y - 5} textAnchor="middle">
                  {edge.label}
                </text>
                <text className="sn-node-value" x={24} y={edge.y + 10} textAnchor="middle">
                  {fmt(edge.value, 2)}
                </text>
              </g>
            );
          })}
          <line className="sn-edge sn-edge--bias" x1={sumX} y1={170} x2={sumX} y2={sumY + 26} />
          <text className="sn-edge-label" x={sumX} y={188} textAnchor="middle">
            + b = {fmt(bias, 1)}
          </text>
          <circle className="sn-node sn-node--sum" cx={sumX} cy={sumY} r={26} />
          <text className="sn-node-name" x={sumX} y={sumY - 7} textAnchor="middle">
            z
          </text>
          <text className="sn-node-value" x={sumX} y={sumY + 10} textAnchor="middle">
            {fmt(z)}
          </text>
          <line className="sn-edge" x1={sumX + 26} y1={sumY} x2={236} y2={sumY} />
          <rect className="sn-node sn-node--act" x={236} y={sumY - 19} width={38} height={38} rx={8} />
          <ActivationGlyph kind={activation} x={255} y={sumY} />
          <text className="sn-edge-label" x={255} y={sumY + 34} textAnchor="middle">
            {ACTIVATION_NAME[activation]}
          </text>
          <line className="sn-edge" x1={274} y1={sumY} x2={290} y2={sumY} />
          <circle className="sn-node sn-node--out" cx={313} cy={sumY} r={23} />
          <text className="sn-node-name" x={313} y={sumY - 6} textAnchor="middle">
            a
          </text>
          <text className="sn-node-value" x={313} y={sumY + 10} textAnchor="middle">
            {fmt(a)}
          </text>
        </svg>
        <FormulaWithValues
          label="Weighted sum"
          tone="forward"
          expression={`z = ${fmt(x1, 2)}×${signed(w1, 1)} + ${fmt(x2, 2)}×${signed(w2, 1)} + ${signed(bias, 1)}`}
          result={fmt(z)}
        />
        <FormulaWithValues
          label={`Activation · ${ACTIVATION_NAME[activation]}(z) = ${ACTIVATION_FORMULA[activation]}`}
          tone="attention"
          expression={`a = ${ACTIVATION_NAME[activation]}(${fmt(z)})`}
          result={fmt(a)}
        />
        <p className="sn-edge-key">Line thickness is |w|; dashed red lines carry a negative weight.</p>
      </LabSurface>

      <LabSurface label="Activation curve" className="sn-card">
        <SurfaceHeading kicker={`f(z) for z from −6 to 6 · ${activation}`} title="Where on the curve this z sits, and how steep it is" />
        <ActivationPlot kind={activation} z={z} />
        <div className="metric-row">
          <Metric label="z" value={fmt(z)} tone="forward" />
          <Metric label="f(z)" value={fmt(a)} />
          <Metric
            label="Slope f′(z)"
            value={Number.isFinite(gradient) ? fmt(gradient, 4) : "none at z = 0"}
            tone={Number.isFinite(gradient) && gradient < 0.05 ? "loss" : undefined}
          />
        </div>
      </LabSurface>
    </div>
  );
}

const Z_MIN = -6;
const Z_MAX = 6;

function ActivationPlot({ kind, z }: { kind: ActivationKind; z: number }) {
  const left = 40;
  const right = 390;
  const top = 14;
  const bottom = 170;
  const yMax = kind === "relu" ? 6 : 1;
  const yMin = kind === "relu" ? -0.5 : -0.05;
  const xAt = (value: number) => left + ((value - Z_MIN) / (Z_MAX - Z_MIN)) * (right - left);
  const yAt = (value: number) => bottom - ((Math.min(yMax, Math.max(yMin, value)) - yMin) / (yMax - yMin)) * (bottom - top);
  const curve = (target: ActivationKind) => {
    const parts: string[] = [];
    for (let index = 0; index <= 240; index += 1) {
      const value = Z_MIN + ((Z_MAX - Z_MIN) * index) / 240;
      const next = activate(value, target);
      const previous = index > 0 ? activate(value - (Z_MAX - Z_MIN) / 240, target) : next;
      // Break the path at the step's jump so no vertical segment implies intermediate outputs.
      const jump = target === "step" && index > 0 && next !== previous;
      parts.push(`${index === 0 || jump ? "M" : "L"}${xAt(value).toFixed(1)} ${yAt(next).toFixed(1)}`);
    }
    return parts.join(" ");
  };
  const out = activate(z, kind);
  const gradient = slope(z, kind);
  const span = 1.3;
  const tangent =
    Number.isFinite(gradient) && kind !== "step"
      ? { x1: xAt(z - span), y1: yAt(out - gradient * span), x2: xAt(z + span), y2: yAt(out + gradient * span) }
      : null;
  const flatZone = 2.887; // |z| beyond which σ′(z) < 0.05
  const inPlot = z >= Z_MIN && z <= Z_MAX;

  return (
    <div className="sn-curve">
      <svg
        viewBox="0 0 400 196"
        role="img"
        aria-label={`${kind} curve. At z ${fmt(z)} the output is ${fmt(out)} and the slope is ${Number.isFinite(gradient) ? fmt(gradient, 4) : "not defined at z = 0"}.`}
      >
        {kind === "sigmoid" && (
          <>
            <rect className="sn-flat" x={left} y={top} width={xAt(-flatZone) - left} height={bottom - top} />
            <rect className="sn-flat" x={xAt(flatZone)} y={top} width={right - xAt(flatZone)} height={bottom - top} />
            <text className="sn-flat-label" x={left + 4} y={top + 11}>
              slope &lt; 0.05
            </text>
            <text className="sn-flat-label" x={right - 4} y={top + 11} textAnchor="end">
              slope &lt; 0.05
            </text>
          </>
        )}
        <line className="plot-axis" x1={left} x2={right} y1={yAt(0)} y2={yAt(0)} />
        <line className="plot-axis" x1={xAt(0)} x2={xAt(0)} y1={top} y2={bottom} />
        {[Z_MIN, -3, 0, 3, Z_MAX].map((tick) => (
          <text key={tick} className="sn-tick" x={xAt(tick)} y={bottom + 14} textAnchor="middle">
            {tick}
          </text>
        ))}
        {(kind === "relu" ? [0, 3, 6] : [0, 0.5, 1]).map((tick) => (
          <text key={tick} className="sn-tick" x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
            {tick}
          </text>
        ))}
        {(["sigmoid", "relu", "step"] as const)
          .filter((other) => other !== kind)
          .map((other) => (
            <path key={other} className={`sn-curve-line sn-curve-line--ghost sn-curve-line--${other}`} d={curve(other)} />
          ))}
        <path className="sn-curve-line" d={curve(kind)} />
        {tangent && <line className="sn-tangent" {...tangent} />}
        {inPlot && (
          <>
            <line className="marker-guide" x1={xAt(z)} x2={xAt(z)} y1={top} y2={bottom} />
            <circle className="sn-curve-point" cx={xAt(z)} cy={yAt(out)} r={5} />
          </>
        )}
        <text className="sn-axis" x={right} y={bottom + 28} textAnchor="end">
          z →
        </text>
        <text className="sn-axis" x={left} y={top - 4}>
          f(z)
        </text>
      </svg>
      <p className="sn-legend">
        Solid: {kind}. Faint dashed: the other two, for comparison.{" "}
        {kind === "step" ? "Step has no tangent: its slope is 0 away from the jump and does not exist at z = 0." : "The short line is the tangent; its steepness is f′(z)."}
      </p>
    </div>
  );
}
