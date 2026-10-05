import {
  backpropSnapshot,
  BarList,
  computationLinkPath,
  FormulaWithValues,
  LabSurface,
  Metric,
  numericWeightGradient,
  RangeControl,
  relativeGradientError,
  SegmentedControl,
  smoothLinePath,
  StepThroughController,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { useEffect, useMemo } from "react";
import {
  FLOW_ACTIVATIONS,
  FLOW_BATCH,
  FLOW_INITS,
  FLOW_WIDTH,
  flowDepth,
  MAX_FLOW_DEPTH,
  MIN_FLOW_DEPTH,
  measureGradientFlow,
  type FlowActivation,
  type FlowInit,
} from "./gradient-flow";

const WEIGHT_RANGE: [number, number] = [-3, 3];

const asNumber = (
  state: ModuleContext["state"],
  key: string,
  fallback: number,
) => (typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback);

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pick = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

/** Gradient-flow bars run on a log scale from 1e-13 to 1e7, so every full bar spans twenty decades. */
const FLOW_LOG_FLOOR = -13;
const FLOW_LOG_SPAN = 20;
const scientific = (value: number, digits = 1) => (value === 0 ? "0" : value.toExponential(digits));
/** Plain decimals for ordinary sizes, scientific for anything that would run long. */
const compact = (value: number) => (value !== 0 && (value >= 1000 || value < 0.001) ? value.toExponential(2) : value.toFixed(3));
const ACTIVATION_NAMES: Record<FlowActivation, string> = { sigmoid: "sigmoid", tanh: "tanh", relu: "ReLU" };
const INIT_NAMES: Record<FlowInit, string> = {
  small: "small (std 0.1)",
  glorot: "Glorot",
  he: "He",
  large: "large (std 1.0)",
};

/** Signed fixed-point text with a true minus sign. */
const num = (value: number, digits = 3) => `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;

const nodes = {
  input: { x: 62, y: 70, label: "input x" },
  weight: { x: 62, y: 196, label: "weight w" },
  multiply: { x: 246, y: 132, label: "multiply z" },
  sigmoid: { x: 436, y: 132, label: "sigmoid ŷ" },
  target: { x: 452, y: 250, label: "target y" },
  loss: { x: 632, y: 132, label: "loss L" },
} as const;

type NodeId = keyof typeof nodes;

const traceSteps = [
  "Multiply",
  "Activate",
  "Measure loss",
  "Loss → output",
  "Output → product",
  "Product → weight",
] as const;

/** Phase at which each node's forward value exists (inputs exist from the start). */
const valueFrom: Record<NodeId, number> = {
  input: 0,
  weight: 0,
  target: 0,
  multiply: 0,
  sigmoid: 1,
  loss: 2,
};

/** Phase at which the backward pass has written each node's gradient. */
const gradientFrom: Partial<Record<NodeId, number>> = {
  loss: 3,
  sigmoid: 3,
  multiply: 4,
  weight: 5,
  input: 5,
};

export default function Explore({
  state,
  setState,
  currentStep,
  narrate,
}: ModuleContext) {
  const input = clamp(asNumber(state, "x", 0.8), 0.1, 1.5);
  const weight = clamp(asNumber(state, "weight", 0.6), WEIGHT_RANGE[0], WEIGHT_RANGE[1]);
  const target = clamp(asNumber(state, "target", 0.9), 0, 1);
  const phase = Math.max(
    0,
    Math.min(traceSteps.length - 1, Math.round(asNumber(state, "phase", 0))),
  );
  // The two gradient-flow steps leave the trace where the learner last had it, at the end of the backward pass.
  const outerPhase = [0, 2, 3, 5, 5, 5][currentStep] ?? 0;

  const depth = flowDepth(asNumber(state, "flowDepth", 6));
  const activation = pick(state.flowActivation, FLOW_ACTIVATIONS, "sigmoid");
  const init = pick(state.flowInit, FLOW_INITS, "glorot");
  const skip = pick(state.flowSkip, ["off", "on"] as const, "off") === "on";
  const flow = useMemo(
    () => measureGradientFlow({ depth, activation, init, skip }),
    [depth, activation, init, skip],
  );

  useEffect(() => {
    if (phase !== outerPhase) setState({ phase: outerPhase });
    // Only an outer step change should move the trace; learner clicks must stick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  const learningRate = asNumber(state, "learningRate", 0.7);
  const values = backpropSnapshot(input, weight, target);
  const numeric = numericWeightGradient(input, weight, target);
  const gradientError = relativeGradientError(values.dLossWeight, numeric);
  const backward = phase >= 3;

  const nodeValue: Record<NodeId, number> = {
    input,
    weight,
    multiply: values.product,
    sigmoid: values.prediction,
    target,
    loss: values.loss,
  };
  const nodeGradient: Partial<Record<NodeId, { symbol: string; value: number }>> = {
    loss: { symbol: "∂L/∂L", value: 1 },
    sigmoid: { symbol: "∂L/∂ŷ", value: values.dLossPrediction },
    multiply: { symbol: "∂L/∂z", value: values.dLossProduct },
    weight: { symbol: "∂L/∂w", value: values.dLossWeight },
    input: { symbol: "∂L/∂x", value: values.dLossInput },
  };

  const edgeData = [
    {
      id: "input-product",
      source: nodes.input,
      target: nodes.multiply,
      forwardStep: 0,
      backwardStep: 5,
      label: `x = ${input.toFixed(2)}`,
      local: `∂z/∂x = w = ${num(weight)}`,
      factor: `× ${num(weight)}`,
    },
    {
      id: "weight-product",
      source: nodes.weight,
      target: nodes.multiply,
      forwardStep: 0,
      backwardStep: 5,
      label: `w = ${num(weight, 2)}`,
      local: `∂z/∂w = x = ${num(input)}`,
      factor: `× ${num(input)}`,
    },
    {
      id: "product-sigmoid",
      source: nodes.multiply,
      target: nodes.sigmoid,
      forwardStep: 1,
      backwardStep: 4,
      label: `z = ${num(values.product)}`,
      local: `∂ŷ/∂z = ŷ(1−ŷ) = ${num(values.dPredictionProduct)}`,
      factor: `× ${num(values.dPredictionProduct)}`,
    },
    {
      id: "sigmoid-loss",
      source: nodes.sigmoid,
      target: nodes.loss,
      forwardStep: 2,
      backwardStep: 3,
      label: `ŷ = ${num(values.prediction)}`,
      local: `∂L/∂ŷ = 2(ŷ−y) = ${num(values.dLossPrediction)}`,
      factor: `× ${num(values.dLossPrediction)}`,
    },
    {
      id: "target-loss",
      source: nodes.target,
      target: nodes.loss,
      forwardStep: 2,
      backwardStep: -1,
      label: `y = ${target.toFixed(2)}`,
      local: "data: no gradient needed",
      factor: "",
    },
  ];

  const tape = [
    {
      step: 0,
      name: "z = x · w",
      work: `${num(input, 2)} × ${num(weight, 2)}`,
      result: num(values.product),
    },
    {
      step: 1,
      name: "ŷ = σ(z)",
      work: `1 / (1 + e^(${values.product >= 0 ? "−" : ""}${Math.abs(values.product).toFixed(3)}))`,
      result: num(values.prediction),
    },
    {
      step: 2,
      name: "L = (ŷ − y)²",
      work: `(${num(values.prediction)} − ${target.toFixed(2)})²`,
      result: num(values.loss, 4),
    },
    {
      step: 3,
      name: "∂L/∂ŷ = 2(ŷ − y)",
      work: `2 × (${num(values.prediction)} − ${target.toFixed(2)})`,
      result: num(values.dLossPrediction),
    },
    {
      step: 4,
      name: "∂L/∂z = ∂L/∂ŷ · ŷ(1 − ŷ)",
      work: `${num(values.dLossPrediction)} × ${num(values.dPredictionProduct)}`,
      result: num(values.dLossProduct, 4),
    },
    {
      step: 5,
      name: "∂L/∂w = ∂L/∂z · x",
      work: `${num(values.dLossProduct, 4)} × ${num(input)}`,
      result: num(values.dLossWeight, 4),
      also: `∂L/∂x = ∂L/∂z · w = ${num(values.dLossProduct, 4)} × ${num(weight)} = ${num(values.dLossInput, 4)}`,
    },
  ];

  const chart = {
    left: 44,
    right: 388,
    top: 16,
    bottom: 188,
  };
  const samples = Array.from({ length: 121 }, (_, index) => {
    const candidate = WEIGHT_RANGE[0] + (index / 120) * (WEIGHT_RANGE[1] - WEIGHT_RANGE[0]);
    return { weight: candidate, loss: backpropSnapshot(input, candidate, target).loss };
  });
  const maximumLoss = Math.max(0.2, ...samples.map((sample) => sample.loss));
  const xAt = (candidate: number) =>
    chart.left + ((candidate - WEIGHT_RANGE[0]) / (WEIGHT_RANGE[1] - WEIGHT_RANGE[0])) * (chart.right - chart.left);
  const yAt = (loss: number) => chart.bottom - (loss / maximumLoss) * (chart.bottom - chart.top);
  const curve = smoothLinePath(samples.map((sample) => ({ x: xAt(sample.weight), y: yAt(sample.loss) })));
  const tangentStart = Math.max(WEIGHT_RANGE[0], weight - 0.72);
  const tangentEnd = Math.min(WEIGHT_RANGE[1], weight + 0.72);
  const tangentAt = (candidate: number) => values.loss + values.dLossWeight * (candidate - weight);
  // The loss reaches zero where σ(x·w) = y, i.e. w* = logit(y) / x, when that lies in view.
  const bestWeight =
    target > 0 && target < 1 ? Math.log(target / (1 - target)) / input : Number.NaN;
  const bestInView = Number.isFinite(bestWeight) && bestWeight >= WEIGHT_RANGE[0] && bestWeight <= WEIGHT_RANGE[1];

  const setPhase = (next: number) => {
    setState({ phase: next });
    const direction = next < 3 ? "forward" : "backward";
    narrate(`${traceSteps[next]}. Tracing ${direction}: ${tape[next].name} = ${tape[next].result}.`);
  };

  const onEdge = (edge: (typeof edgeData)[number]) => {
    if (backward) {
      if (edge.backwardStep >= 0) setPhase(edge.backwardStep);
      else narrate("The target is data, not a parameter, so the backward pass computes no gradient for it.");
      return;
    }
    setPhase(edge.forwardStep);
  };

  return (
    <div className="phase4-backprop">
      <LabSurface
        label="Interactive backpropagation computation graph"
        className="backprop-bench"
      >
        <SurfaceHeading
          kicker={backward ? "Backward pass" : "Forward pass"}
          title="Values move right; gradients move left, one multiplication per edge"
          aside={
            <span className={`trace-direction trace-direction--${backward ? "backward" : "forward"}`}>
              {backward ? "← gradients" : "values →"}
            </span>
          }
        />

        <svg
          className="backprop-d3-graph"
          viewBox="0 0 700 300"
          role="img"
          aria-label={`Computation graph at ${traceSteps[phase]}. z ${num(values.product)}, prediction ${num(values.prediction)}, loss ${num(values.loss, 4)}.${backward ? ` Gradients so far: ${Object.entries(gradientFrom)
            .filter(([, from]) => (from ?? 99) <= phase)
            .map(([id]) => `${nodeGradient[id as NodeId]?.symbol} ${num(nodeGradient[id as NodeId]?.value ?? 0, 4)}`)
            .join(", ")}.` : ""}`}
        >
          <defs>
            <marker id="forward-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" />
            </marker>
            <marker id="gradient-arrow" markerWidth="8" markerHeight="8" refX="2" refY="4" orient="auto">
              <path d="M 8 0 L 0 4 L 8 8 z" />
            </marker>
          </defs>

          {edgeData.map((edge) => {
            const forwardActive = phase >= edge.forwardStep;
            const backwardActive = edge.backwardStep >= 0 && phase >= edge.backwardStep;
            const selected = backward ? edge.backwardStep === phase : edge.forwardStep === phase;
            const midX = (edge.source.x + edge.target.x) / 2;
            const midY = (edge.source.y + edge.target.y) / 2;
            return (
              <g
                key={edge.id}
                className={[
                  "backprop-edge",
                  forwardActive ? "is-forward" : "",
                  backwardActive ? "is-backward" : "",
                  selected ? "is-selected" : "",
                ].filter(Boolean).join(" ")}
                role="button"
                tabIndex={0}
                aria-label={`${edge.source.label} to ${edge.target.label}. ${edge.label}. ${edge.local}.`}
                onClick={() => onEdge(edge)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onEdge(edge);
                  }
                }}
              >
                <path
                  className="edge-forward-path"
                  d={computationLinkPath(edge.source, edge.target)}
                />
                {edge.backwardStep >= 0 && (
                  <path
                    className="edge-gradient-path"
                    d={computationLinkPath(edge.source, edge.target)}
                  />
                )}
                <text className="bp-edge-value" x={midX} y={midY - 10} textAnchor="middle">
                  {edge.label}
                </text>
                {backwardActive && (
                  <text className="bp-edge-local" x={midX} y={midY + 17} textAnchor="middle">
                    {edge.factor}
                  </text>
                )}
              </g>
            );
          })}

          {(Object.keys(nodes) as NodeId[]).map((id) => {
            const node = nodes[id];
            const known = phase >= valueFrom[id];
            const gradientStep = gradientFrom[id];
            const gradient = gradientStep !== undefined && phase >= gradientStep ? nodeGradient[id] : undefined;
            return (
              <g
                className={`backprop-node backprop-node--${id}${known ? "" : " is-pending"}${gradient ? " has-gradient" : ""}`}
                key={id}
                transform={`translate(${node.x} ${node.y})`}
              >
                <circle r={id === "loss" ? 38 : 34} />
                <text className="backprop-node__name" textAnchor="middle" y="-5">
                  {node.label}
                </text>
                <text className="backprop-node__value" textAnchor="middle" y="13">
                  {known ? num(nodeValue[id], id === "loss" ? 4 : 3) : "?"}
                </text>
                {id === "target" && backward && (
                  <text className="bp-no-gradient" textAnchor="start" x="40" y="4">
                    no gradient: data
                  </text>
                )}
                {gradient && (
                  <g className="bp-grad-pill" transform={`translate(0 ${id === "loss" ? 52 : 48})`}>
                    <rect x="-50" y="-11" width="100" height="21" rx="10.5" />
                    <text textAnchor="middle" y="4">
                      {`${gradient.symbol} ${num(gradient.value, id === "loss" ? 0 : 4)}`}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        <ol className="bp-chain-strip" aria-label="Tape of forward values and backward products">
          {tape.map((row) => {
            const status = row.step === phase ? "is-current" : row.step < phase ? "is-done" : "is-pending";
            return (
              <li
                key={row.step}
                className={`bp-tape-row bp-tape-row--${row.step < 3 ? "forward" : "backward"} ${status}`}
                aria-current={row.step === phase ? "step" : undefined}
              >
                <span className="bp-tape-row__index">{row.step + 1}</span>
                <code className="bp-tape-row__name">{row.name}</code>
                <code className="bp-tape-row__work">{row.step <= phase ? row.work : "…"}</code>
                <strong className="bp-tape-row__result">{row.step <= phase ? row.result : "?"}</strong>
                {row.also && row.step <= phase && <small className="bp-tape-row__also">{row.also}</small>}
              </li>
            );
          })}
        </ol>

        <StepThroughController
          label="Backpropagation trace"
          steps={traceSteps}
          current={phase}
          onChange={setPhase}
        />
        <button
          type="button"
          className="primary-action backprop-next-edge"
          onClick={() => setPhase((phase + 1) % traceSteps.length)}
        >
          {phase === traceSteps.length - 1 ? "Run the trace again" : "Advance one edge"}
        </button>
      </LabSurface>

      <div className="backprop-analysis">
        <LabSurface
          label="Loss versus weight tangent chart"
          className="backprop-tangent-panel"
        >
          <SurfaceHeading
            kicker="Loss versus weight, x and y held fixed"
            title="The backward pass predicts this slope"
          />
          <svg
            viewBox="0 0 400 214"
            role="img"
            aria-label={`Loss curve over weight −3 to 3 with tangent at weight ${num(weight, 2)}. Slope ${num(values.dLossWeight, 4)}.${bestInView ? ` Loss reaches zero at weight ${num(bestWeight, 2)}.` : ""}`}
          >
            <line className="plot-axis" x1={chart.left} x2={chart.right} y1={chart.bottom} y2={chart.bottom} />
            <line className="plot-axis" x1={chart.left} x2={chart.left} y1={chart.top} y2={chart.bottom} />
            {[-3, -2, -1, 0, 1, 2, 3].map((tick) => (
              <text key={tick} className="bp-tick" x={xAt(tick)} y={chart.bottom + 13} textAnchor="middle">
                {tick < 0 ? `−${-tick}` : tick}
              </text>
            ))}
            {[0, maximumLoss / 2, maximumLoss].map((tick) => (
              <text key={tick} className="bp-tick" x={chart.left - 5} y={yAt(tick) + 3} textAnchor="end">
                {tick.toFixed(2)}
              </text>
            ))}
            {bestInView && (
              <g className="bp-best">
                <line x1={xAt(bestWeight)} x2={xAt(bestWeight)} y1={chart.top} y2={chart.bottom} />
                <text x={xAt(bestWeight) + (bestWeight > 1.8 ? -4 : 4)} y={chart.top + 10} textAnchor={bestWeight > 1.8 ? "end" : "start"}>
                  ŷ = y at w {num(bestWeight, 2)}
                </text>
              </g>
            )}
            <path className="loss-curve" d={curve} />
            <line
              className="tangent-line"
              x1={xAt(tangentStart)}
              y1={yAt(tangentAt(tangentStart))}
              x2={xAt(tangentEnd)}
              y2={yAt(tangentAt(tangentEnd))}
            />
            <line className="marker-guide" x1={xAt(weight)} x2={xAt(weight)} y1={yAt(values.loss)} y2={chart.bottom} />
            <circle className="loss-ball" cx={xAt(weight)} cy={yAt(values.loss)} r="7" />
            <text className="bp-axis-title" x={chart.left + 4} y={chart.top - 4}>
              loss L
            </text>
            <text className="bp-axis-title" x={chart.right} y={chart.bottom + 25} textAnchor="end">
              weight w →
            </text>
          </svg>
          <div className="metric-row">
            <Metric label="Loss" value={values.loss.toFixed(5)} tone="loss" />
            <Metric label="Analytic slope" value={num(values.dLossWeight, 5)} tone="gradient" />
            <Metric label="Numeric check" value={num(numeric, 5)} tone="gradient" />
          </div>
        </LabSurface>

        <LabSurface
          label="Backpropagation controls and chain rule"
          className="backprop-control-panel"
        >
          <SurfaceHeading
            kicker="Parameter bench"
            title="Move the weight, then check the derivative"
          />
          <RangeControl
            label="Input x"
            min={0.1}
            max={1.5}
            step={0.05}
            value={input}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ x: value, phase: 0 })}
          />
          <RangeControl
            label="Weight w"
            min={WEIGHT_RANGE[0]}
            max={WEIGHT_RANGE[1]}
            step={0.02}
            value={weight}
            format={(value) => num(value, 2)}
            onChange={(value) => setState({ weight: value, phase: 0 })}
          />
          <RangeControl
            label="Target y"
            min={0}
            max={1}
            step={0.05}
            value={target}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ target: value, phase: 0 })}
          />
          <RangeControl
            label="Learning rate"
            min={0.05}
            max={2}
            step={0.05}
            value={learningRate}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ learningRate: value })}
          />
          <FormulaWithValues
            label="Chain rule"
            tone="gradient"
            expression={`${num(values.dLossPrediction)} × ${num(values.dPredictionProduct)} × ${num(values.dProductWeight)}`}
            result={`∂L/∂w = ${num(values.dLossWeight, 5)}`}
            detail={`Numeric agreement: ${(gradientError * 100).toExponential(1)}% relative error`}
          />
          <button
            type="button"
            className="primary-action"
            onClick={() => {
              const next = clamp(weight - learningRate * values.dLossWeight, WEIGHT_RANGE[0], WEIGHT_RANGE[1]);
              setState({ weight: next, phase: 0 });
              narrate(
                `Applied one update. Weight moved from ${weight.toFixed(3)} to ${next.toFixed(3)} and loss is now ${backpropSnapshot(input, next, target).loss.toFixed(4)}.`,
              );
            }}
          >
            Apply w ← w − η∂L/∂w
          </button>
        </LabSurface>
      </div>

      <LabSurface label="Gradient flow through depth" className="backprop-flow-card">
        <SurfaceHeading
          kicker={`${depth} hidden layer${depth === 1 ? "" : "s"} of ${FLOW_WIDTH} units · ${ACTIVATION_NAMES[activation]} · ${INIT_NAMES[init]} start · skip connections ${skip ? "on" : "off"} · ${FLOW_BATCH} fixed examples`}
          title="How much gradient reaches each layer before any training"
          aside={
            <span className="trace-direction trace-direction--backward">
              layer 1 ÷ layer {depth}: {scientific(flow.firstOverLast, 1)}
            </span>
          }
        />
        <div className="backprop-flow-controls">
          <div className="backprop-flow-depth">
            <RangeControl
              label="Network depth"
              min={MIN_FLOW_DEPTH}
              max={MAX_FLOW_DEPTH}
              step={1}
              value={depth}
              onChange={(value) => setState({ flowDepth: value })}
            />
          </div>
          <div className="backprop-flow-activation">
            <SegmentedControl
              label="Activation"
              value={activation}
              options={FLOW_ACTIVATIONS.map((value) => ({ value, label: ACTIVATION_NAMES[value] }))}
              onChange={(value) => setState({ flowActivation: value })}
            />
          </div>
          <div className="backprop-flow-fix">
            <SegmentedControl
              label="Initial weight scale"
              value={init}
              options={FLOW_INITS.map((value) => ({ value, label: INIT_NAMES[value] }))}
              onChange={(value) => setState({ flowInit: value })}
            />
            <SegmentedControl
              label="Skip connections"
              value={skip ? "on" : "off"}
              options={[
                { value: "off", label: "off" },
                { value: "on", label: "on" },
              ]}
              onChange={(value) => {
                setState({ flowSkip: value });
                narrate(value === "on" ? "Skip connections on." : "Skip connections off.");
              }}
            />
          </div>
        </div>
        <BarList
          label={`Gradient norm of each layer's weights, log scale from 1e-13 to 1e7, ${depth} layers`}
          max={FLOW_LOG_SPAN}
          items={flow.layers.map((layer) => ({
            id: `flow-${layer.layer}`,
            label: `Layer ${layer.layer}${layer.layer === 1 ? " · in" : layer.layer === depth ? " · out" : ""}`,
            value: clamp(Math.log10(Math.max(layer.gradNorm, 1e-300)) - FLOW_LOG_FLOOR, 0, FLOW_LOG_SPAN),
            display: scientific(layer.gradNorm, 1),
            tone: "gradient",
            emphasis: layer.layer === 1 || layer.layer === depth,
          }))}
        />
        <div className="metric-row">
          <Metric label="Layer 1 norm" value={scientific(flow.layers[0].gradNorm, 2)} tone="gradient" />
          <Metric label={`Layer ${depth} norm`} value={scientific(flow.layers[depth - 1].gradNorm, 2)} tone="gradient" />
          <Metric label="Layer 1 ÷ last" value={scientific(flow.firstOverLast, 2)} />
          <Metric label="Loss at the start" value={flow.loss.toFixed(2)} tone="loss" />
        </div>
        <details className="attention-data-table">
          <summary>Table of the same numbers</summary>
          <table>
            <thead>
              <tr>
                <th scope="col">Layer</th>
                <th scope="col">Gradient norm</th>
                <th scope="col">Mean slope f′(z)</th>
                <th scope="col">Output RMS</th>
              </tr>
            </thead>
            <tbody>
              {flow.layers.map((layer) => (
                <tr key={layer.layer}>
                  <th scope="row">{layer.layer}</th>
                  <td>{scientific(layer.gradNorm, 2)}</td>
                  <td>{layer.meanSlope.toFixed(3)}</td>
                  <td>{compact(layer.activationRms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
        <p className="lab-note">
          Bar length is the number of decades above 1e-13, so a bar half as long is not half the gradient. The printed
          number is exact. Each norm is the Frobenius norm of ∂L/∂W for one 16 × 16 weight matrix, measured on one
          fixed batch before any update, with one random draw of the weights.
        </p>
      </LabSurface>
    </div>
  );
}
