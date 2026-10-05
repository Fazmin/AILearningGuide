import { useMemo, type CSSProperties } from "react";
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
  add,
  blockParameters,
  cosine,
  createBlockWeights,
  crossPositionShift,
  HIDDEN,
  mean,
  norm,
  ORDER_IDS,
  ORDER_TOLERANCE,
  orderTest,
  runBlock,
  sequenceFor,
  TOKENS,
  WIDTH,
  type BlockOptions,
  type MlpKind,
  type NormKind,
  type OrderId,
  type Placement,
  type Vector,
} from "./block";

const weights = createBlockWeights();

const PRESETS = {
  gpt2: {
    label: "GPT-2 small",
    width: 768,
    layers: 12,
    mlp: "gelu" as MlpKind,
    norm: "layernorm" as NormKind,
  },
  llama2: {
    label: "Llama 2 7B",
    width: 4096,
    layers: 32,
    mlp: "swiglu" as MlpKind,
    norm: "rmsnorm" as NormKind,
  },
} as const;

const ORDER_LABELS: Record<OrderId, string> = {
  original: "Original",
  swap: "Swap cat ↔ mat",
  shuffle: "Shuffle all",
};

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asBoolean = (state: ModuleContext["state"], key: string, fallback: boolean) =>
  typeof state[key] === "boolean" ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

const fixed = (value: number, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const count = (value: number) => {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
};
const grouped = (value: number) => value.toLocaleString("en-US");

/** Eight diverging bars on a shared scale, so rows can be compared by length. */
function VectorBars({ values, scale, label }: { values: Vector; scale: number; label: string }) {
  const width = 132;
  const height = 30;
  const step = width / values.length;
  const middle = height / 2;
  return (
    <svg
      className="tb-bars"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${label}: ${values.map((value) => value.toFixed(2)).join(", ")}`}
    >
      <line className="tb-bars__zero" x1={0} x2={width} y1={middle} y2={middle} />
      {values.map((value, index) => {
        const barHeight = (Math.min(Math.abs(value), scale) / scale) * (middle - 1);
        return (
          <rect
            key={index}
            className={value >= 0 ? "tb-bars__pos" : "tb-bars__neg"}
            x={index * step + step * 0.18}
            y={value >= 0 ? middle - barHeight : middle}
            width={step * 0.64}
            height={Math.max(0.5, barHeight)}
          />
        );
      })}
    </svg>
  );
}

interface FlowRow {
  id: string;
  kind: "stream" | "branch";
  name: string;
  detail: string;
  vector: Vector;
  toggle?: "norm" | "attention" | "mlp";
  enabled?: boolean;
  focus: "norm" | "attention" | "add" | "mlp" | "input";
  extra?: "attention" | "mlp";
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const attentionOn = asBoolean(state, "attention", true);
  const mlpOn = asBoolean(state, "mlp", true);
  const normOn = asBoolean(state, "norm", true);
  const placement: Placement = asString(state, "placement", "pre") === "post" ? "post" : "pre";
  const inputScale = clamp(asNumber(state, "inputScale", 1), 0.25, 8);
  const position = clamp(Math.round(asNumber(state, "position", 5)), 0, TOKENS.length - 1);

  const orderValue = asString(state, "order", "swap");
  const order: OrderId = (ORDER_IDS as readonly string[]).includes(orderValue) ? (orderValue as OrderId) : "swap";
  const positionInfo = asBoolean(state, "positionInfo", true);

  const presetId = asString(state, "paramPreset", "gpt2");
  const paramWidth = clamp(Math.round(asNumber(state, "paramWidth", 768) / 256) * 256 || 256, 256, 8192);
  const paramLayers = clamp(Math.round(asNumber(state, "paramLayers", 12)), 1, 128);
  const paramMlp: MlpKind = asString(state, "paramMlp", "gelu") === "swiglu" ? "swiglu" : "gelu";
  const paramNorm: NormKind = asString(state, "paramNorm", "layernorm") === "rmsnorm" ? "rmsnorm" : "layernorm";

  const options: BlockOptions = useMemo(
    () => ({ norm: normOn, attention: attentionOn, mlp: mlpOn, placement }),
    [attentionOn, mlpOn, normOn, placement],
  );
  const trace = useMemo(
    () => runBlock(sequenceFor(TOKENS, inputScale), weights, options),
    [options, inputScale],
  );
  const shift = useMemo(
    () => crossPositionShift(weights, options, inputScale, position),
    [options, inputScale, position],
  );

  const orderResult = useMemo(
    () => orderTest(weights, options, inputScale, order, positionInfo),
    [options, inputScale, order, positionInfo],
  );

  const x = trace.input[position];
  const attentionWrite = trace.attentionWrite[position];
  const mlpWrite = trace.mlpWrite[position];
  const output = trace.output[position];
  const hidden = trace.mlpHidden[position];
  const heads = trace.headWeights[position];
  const activeUnits = hidden.filter((value) => value > 0).length;
  const xMean = mean(x);
  const xStd = Math.sqrt(mean(x.map((value) => (value - xMean) ** 2)));

  const rows: FlowRow[] =
    placement === "pre"
      ? [
          { id: "input", kind: "stream", name: "Input x", detail: `“${TOKENS[position]}” embedding + position, × ${inputScale}`, vector: x, focus: "input" },
          {
            id: "norm1",
            kind: "branch",
            name: "Layer norm",
            detail: normOn ? `reads x: (x − ${fixed(xMean)}) ÷ ${fixed(xStd)}` : "off: the sublayer reads raw x",
            vector: trace.attentionIn[position],
            toggle: "norm",
            enabled: normOn,
            focus: "norm",
          },
          {
            id: "attention",
            kind: "branch",
            name: "Self-attention",
            detail: attentionOn ? `reads positions 0–${position}, writes back` : "off: writes zeros",
            vector: attentionWrite,
            toggle: "attention",
            enabled: attentionOn,
            focus: "attention",
            extra: "attention",
          },
          { id: "add1", kind: "stream", name: "Residual add", detail: "x + attention write", vector: trace.afterAttention[position], focus: "add" },
          {
            id: "norm2",
            kind: "branch",
            name: "Layer norm",
            detail: normOn ? "reads the stream, normalises a copy" : "off: the MLP reads the raw stream",
            vector: trace.mlpIn[position],
            toggle: "norm",
            enabled: normOn,
            focus: "norm",
          },
          {
            id: "mlp",
            kind: "branch",
            name: "MLP",
            detail: mlpOn ? `8 → ${HIDDEN} (GELU, ${activeUnits} positive) → 8` : "off: writes zeros",
            vector: mlpWrite,
            toggle: "mlp",
            enabled: mlpOn,
            focus: "mlp",
            extra: "mlp",
          },
          { id: "add2", kind: "stream", name: "Residual add", detail: "stream + MLP write = block output", vector: output, focus: "add" },
        ]
      : [
          { id: "input", kind: "stream", name: "Input x", detail: `“${TOKENS[position]}” embedding + position, × ${inputScale}`, vector: x, focus: "input" },
          {
            id: "attention",
            kind: "branch",
            name: "Self-attention",
            detail: attentionOn ? `reads raw x at positions 0–${position}` : "off: writes zeros",
            vector: attentionWrite,
            toggle: "attention",
            enabled: attentionOn,
            focus: "attention",
            extra: "attention",
          },
          { id: "add1", kind: "stream", name: "Residual add", detail: "x + attention write", vector: add(x, attentionWrite), focus: "add" },
          {
            id: "norm1",
            kind: "stream",
            name: "Layer norm",
            detail: normOn ? "on the main line: rescales the stream itself" : "off",
            vector: trace.afterAttention[position],
            toggle: "norm",
            enabled: normOn,
            focus: "norm",
          },
          {
            id: "mlp",
            kind: "branch",
            name: "MLP",
            detail: mlpOn ? `8 → ${HIDDEN} (GELU, ${activeUnits} positive) → 8` : "off: writes zeros",
            vector: mlpWrite,
            toggle: "mlp",
            enabled: mlpOn,
            focus: "mlp",
            extra: "mlp",
          },
          { id: "add2", kind: "stream", name: "Residual add", detail: "stream + MLP write", vector: add(trace.afterAttention[position], mlpWrite), focus: "add" },
          {
            id: "norm2",
            kind: "stream",
            name: "Layer norm",
            detail: normOn ? "on the main line: block output" : "off: block output",
            vector: output,
            toggle: "norm",
            enabled: normOn,
            focus: "norm",
          },
        ];

  const barScale = Math.max(1e-6, ...rows.flatMap((row) => row.vector.map((value) => Math.abs(value))));

  const toggle = (key: "norm" | "attention" | "mlp") => {
    const current = key === "norm" ? normOn : key === "attention" ? attentionOn : mlpOn;
    setState({ [key]: !current });
    const name = key === "norm" ? "Layer norm" : key === "attention" ? "Self-attention" : "MLP";
    narrate(`${name} ${current ? "off" : "on"}.`);
  };

  const describeOrder = (nextOrder: OrderId, nextPosition: boolean) => {
    if (nextOrder === "original") return "Original order: there is nothing to compare.";
    const result = orderTest(weights, options, inputScale, nextOrder, nextPosition);
    return `Position information ${nextPosition ? "on" : "off"}, order ${ORDER_LABELS[nextOrder]}: ${result.changed} of ${TOKENS.length} words changed output, largest shift ${fixed(result.largestShift, 3)}.`;
  };

  /* Residual stream waterfall */
  const sums = x.map((value, index) => ({
    start: value,
    afterAttention: value + attentionWrite[index],
    afterMlp: value + attentionWrite[index] + mlpWrite[index],
    output: output[index],
  }));
  const domain = Math.max(
    1e-6,
    ...sums.flatMap((entry) => [entry.start, entry.afterAttention, entry.afterMlp, entry.output].map(Math.abs)),
  );
  const waterfallWidth = 520;
  const rowHeight = 20;
  const left = 34;
  const plot = waterfallWidth - left - 10;
  const xAt = (value: number) => left + plot / 2 + (value / domain) * (plot / 2);
  const waterfallHeight = WIDTH * rowHeight + 18;

  /* Parameter count */
  const counts = blockParameters(paramWidth, paramMlp, paramNorm);
  const preset = presetId === "gpt2" || presetId === "llama2" ? PRESETS[presetId] : undefined;
  const presetMatches =
    preset !== undefined &&
    preset.width === paramWidth &&
    preset.layers === paramLayers &&
    preset.mlp === paramMlp &&
    preset.norm === paramNorm;
  const publishedTotal =
    presetMatches && presetId === "gpt2"
      ? 12 * blockParameters(768, "gelu", "layernorm", true).total + 50257 * 768 + 1024 * 768 + 2 * 768
      : presetMatches && presetId === "llama2"
        ? 32 * counts.total + 2 * 32000 * 4096 + 4096
        : undefined;
  const share = (value: number) => value / counts.total;
  // The toy block has no attention biases; its MLP has both biases.
  const toyParameters = 4 * WIDTH * WIDTH + 2 * WIDTH * HIDDEN + HIDDEN + WIDTH + 2 * 2 * WIDTH;

  return (
    <div className="tb-lab">
      <LabSurface label="Transformer block" className="tb-block-card">
        <SurfaceHeading
          kicker={`${placement === "pre" ? "Pre-norm" : "Post-norm"} block · d = ${WIDTH} · 2 heads · MLP ${WIDTH} → ${HIDDEN} → ${WIDTH}`}
          title="One block, computed for one token"
        />
        <div className="tb-controls">
          <div className="tb-placement-control">
            <SegmentedControl
              label="Placement"
              value={placement}
              options={[
                { value: "pre", label: "Pre-norm" },
                { value: "post", label: "Post-norm" },
              ]}
              onChange={(value) => setState({ placement: value })}
            />
          </div>
          <div className="tb-positions" role="group" aria-label="Token position">
            <span>Token</span>
            {TOKENS.map((token, index) => (
              <button
                type="button"
                key={`${token}-${index}`}
                aria-pressed={index === position}
                onClick={() => setState({ position: index })}
              >
                <small>{index}</small>
                {token}
              </button>
            ))}
          </div>
        </div>

        <ol className={`tb-flow tb-flow--${placement}`}>
          {rows.map((row) => (
            <li
              key={row.id}
              className={[
                "tb-row",
                `tb-row--${row.kind}`,
                `tb-row--${row.focus}`,
                row.enabled === false ? "is-off" : "",
              ].filter(Boolean).join(" ")}
            >
              <span className="tb-rail" aria-hidden="true">
                <i>{row.focus === "add" ? "+" : ""}</i>
              </span>
              <div className="tb-row__body">
                <div className="tb-row__name">
                  {row.toggle ? (
                    <button
                      type="button"
                      aria-pressed={row.enabled}
                      onClick={() => toggle(row.toggle!)}
                    >
                      <span>{row.name}</span>
                      <em>{row.enabled ? "on" : "off"}</em>
                    </button>
                  ) : (
                    <strong>{row.name}</strong>
                  )}
                  <small>{row.detail}</small>
                </div>
                <VectorBars values={row.vector} scale={barScale} label={`${row.name} vector`} />
                <code className="tb-row__norm">norm {fixed(norm(row.vector))}</code>
                {row.extra === "attention" && attentionOn && (
                  <div className="tb-heads" aria-label="Attention weights for this token">
                    {heads.map((weightsForHead, head) => (
                      <div key={head} className="tb-heads__row">
                        <span>head {head + 1}</span>
                        {weightsForHead.map((weight, index) => (
                          <b
                            key={index}
                            style={{ "--tb-weight": weight } as CSSProperties}
                            title={`${TOKENS[index]}: ${(weight * 100).toFixed(1)}%`}
                          >
                            <small>{TOKENS[index]}</small>
                            {(weight * 100).toFixed(0)}%
                          </b>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>
        <p className="tb-note">
          The stream runs down the left rail. Branch rows read a copy of it and write back at the next +.
          Bars share one scale, so lengths compare across rows. Weights are seeded and untrained: this
          shows the plumbing, not learned behaviour.
        </p>
      </LabSurface>

      <LabSurface label="Residual stream" className="tb-stream-card">
        <SurfaceHeading
          kicker={`Token “${TOKENS[position]}” · ${WIDTH} dimensions`}
          title="Every write adds to the same vector"
        />
        <svg
          className="tb-waterfall"
          viewBox={`0 0 ${waterfallWidth} ${waterfallHeight}`}
          role="img"
          aria-label={`Residual stream for “${TOKENS[position]}”. Input norm ${fixed(norm(x))}, attention write ${fixed(norm(attentionWrite))}, MLP write ${fixed(norm(mlpWrite))}, output ${fixed(norm(output))}.`}
        >
          <line className="tb-waterfall__zero" x1={xAt(0)} x2={xAt(0)} y1={4} y2={waterfallHeight - 12} />
          {sums.map((entry, index) => {
            const top = 6 + index * rowHeight;
            const lane = (from: number, to: number, offset: number, className: string) => (
              <rect
                className={className}
                x={Math.min(xAt(from), xAt(to))}
                y={top + offset}
                width={Math.max(0.8, Math.abs(xAt(to) - xAt(from)))}
                height={4.2}
              />
            );
            return (
              <g key={index}>
                <text className="tb-waterfall__label" x={left - 6} y={top + 11} textAnchor="end">
                  d{index}
                </text>
                {lane(0, entry.start, 1, "tb-waterfall__input")}
                {lane(entry.start, entry.afterAttention, 6.5, "tb-waterfall__attention")}
                {lane(entry.afterAttention, entry.afterMlp, 12, "tb-waterfall__mlp")}
                <path
                  className="tb-waterfall__output"
                  d={`M ${xAt(entry.output)} ${top + 3} l 4 5.5 l -4 5.5 l -4 -5.5 z`}
                />
              </g>
            );
          })}
          <text className="tb-waterfall__tick" x={xAt(-domain)} y={waterfallHeight - 2}>
            −{fixed(domain, 1)}
          </text>
          <text className="tb-waterfall__tick" x={xAt(0)} y={waterfallHeight - 2} textAnchor="middle">
            0
          </text>
          <text className="tb-waterfall__tick" x={xAt(domain)} y={waterfallHeight - 2} textAnchor="end">
            {fixed(domain, 1)}
          </text>
        </svg>
        <div className="tb-legend" aria-hidden="true">
          <span><i className="tb-legend__input" />input x (top lane)</span>
          <span><i className="tb-legend__attention" />+ attention write (middle)</span>
          <span><i className="tb-legend__mlp" />+ MLP write (bottom)</span>
          <span><i className="tb-legend__output" />output</span>
        </div>
        <p className="tb-note">
          {placement === "pre"
            ? "Pre-norm: each diamond sits exactly where the three lanes end, because output = x + attention write + MLP write."
            : "Post-norm: the diamond is the renormalised output, not the end of the lanes. The stream is rescaled after each addition, so its norm is always √8 ≈ 2.83 with the norm on."}
        </p>
        <RangeControl
          label="Input scale"
          min={0.25}
          max={8}
          step={0.25}
          value={inputScale}
          format={(value) => `${value}×`}
          onChange={(value) => setState({ inputScale: value })}
        />
        <div className="metric-row">
          <Metric label="Input norm" value={fixed(norm(x))} />
          <Metric label="Attention write" value={fixed(norm(attentionWrite))} tone="gradient" />
          <Metric label="MLP write" value={fixed(norm(mlpWrite))} tone="forward" />
          <Metric label="Output norm" value={fixed(norm(output))} />
        </div>
        <div className="metric-row">
          <Metric label="Output shift if other words change" value={fixed(shift, 3)} tone={shift > 0 ? "gradient" : undefined} />
          <Metric label="cos(output, input)" value={fixed(cosine(output, x), 3)} />
        </div>
      </LabSurface>

      <LabSurface label="Parameter count" className="tb-param-card">
        <SurfaceHeading
          kicker={`d = ${grouped(paramWidth)} · ${paramLayers} layers`}
          title="Where a block's parameters live"
        />
        <div className="tb-presets" role="group" aria-label="Parameter presets">
          <span>Presets</span>
          {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((id) => (
            <button
              type="button"
              key={id}
              aria-pressed={presetMatches && presetId === id}
              onClick={() => {
                const entry = PRESETS[id];
                setState({
                  paramPreset: id,
                  paramWidth: entry.width,
                  paramLayers: entry.layers,
                  paramMlp: entry.mlp,
                  paramNorm: entry.norm,
                });
              }}
            >
              {PRESETS[id].label}
            </button>
          ))}
        </div>
        <RangeControl
          label="Model width"
          min={256}
          max={8192}
          step={256}
          value={paramWidth}
          format={(value) => `d = ${grouped(value)}`}
          onChange={(value) => setState({ paramWidth: value, paramPreset: "custom" })}
        />
        <RangeControl
          label="Layers"
          min={1}
          max={128}
          step={1}
          value={paramLayers}
          onChange={(value) => setState({ paramLayers: value, paramPreset: "custom" })}
        />
        <div className="tb-param-segments">
          <SegmentedControl
            label="MLP"
            value={paramMlp}
            options={[
              { value: "gelu", label: "GELU 4×" },
              { value: "swiglu", label: "SwiGLU ≈8/3×" },
            ]}
            onChange={(value) => setState({ paramMlp: value, paramPreset: "custom" })}
          />
          <SegmentedControl
            label="Norm"
            value={paramNorm}
            options={[
              { value: "layernorm", label: "LayerNorm" },
              { value: "rmsnorm", label: "RMSNorm" },
            ]}
            onChange={(value) => setState({ paramNorm: value, paramPreset: "custom" })}
          />
        </div>

        <div
          className="tb-param-bar"
          role="img"
          aria-label={`One block: attention ${(share(counts.attention) * 100).toFixed(1)} percent, MLP ${(share(counts.mlp) * 100).toFixed(1)} percent, norms ${(share(counts.norms) * 100).toFixed(3)} percent.`}
        >
          <span className="tb-param-bar__attention" style={{ flexGrow: counts.attention }}>
            attention {(share(counts.attention) * 100).toFixed(1)}%
          </span>
          <span className="tb-param-bar__mlp" style={{ flexGrow: counts.mlp }}>
            MLP {(share(counts.mlp) * 100).toFixed(1)}%
          </span>
          <span className="tb-param-bar__norms" style={{ flexGrow: Math.max(counts.norms, counts.total * 0.004) }} title="norms" />
        </div>

        <div className="tb-param-formulas">
          <FormulaWithValues label="Attention" expression="4 · d²  (Q, K, V, output)" result={grouped(counts.attention)} tone="gradient" />
          <FormulaWithValues
            label="MLP"
            expression={paramMlp === "gelu" ? `2 · d · ${grouped(counts.hidden)}  (up, down)` : `3 · d · ${grouped(counts.hidden)}  (gate, up, down)`}
            result={grouped(counts.mlp)}
            tone="forward"
            detail={paramMlp === "gelu" ? "Hidden width 4d" : "Hidden width 8d/3, rounded up to a multiple of 256 as in Llama"}
          />
          <FormulaWithValues
            label="Norms"
            expression={paramNorm === "layernorm" ? "2 norms · 2d  (scale, shift)" : "2 norms · d  (scale only)"}
            result={grouped(counts.norms)}
          />
          <FormulaWithValues
            label="Block"
            expression={`sum vs 12 · d² = ${grouped(counts.twelveDSquared)}`}
            result={grouped(counts.total)}
            tone="loss"
            detail={`${((counts.total / counts.twelveDSquared) * 100).toFixed(2)}% of 12·d²`}
          />
        </div>
        <div className="metric-row">
          <Metric label="All blocks" value={count(counts.total * paramLayers)} />
          <Metric label="MLP share" value={`${(share(counts.mlp) * 100).toFixed(1)}%`} tone="forward" />
          <Metric label="This lab's toy block" value={`${toyParameters} params`} />
        </div>
        <p className="tb-note">
          {publishedTotal !== undefined
            ? `${preset?.label}: blocks ${count(counts.total * paramLayers)}${presetId === "gpt2" ? " (plus GPT-2's biases)" : ""}, plus embeddings and the final norm, gives ${grouped(publishedTotal)}, the published count.`
            : "Weights only, no biases or embeddings. Pick a preset to see a published model's total reproduced."}
        </p>
      </LabSurface>

      <LabSurface label="Position" className="tb-position-card">
        <SurfaceHeading
          kicker={`Same six words · ${attentionOn ? "every word reads every word" : "Self-attention off"} · ${placement === "pre" ? "pre-norm" : "post-norm"}`}
          title="Can the block tell which word came first?"
        />
        <div className="tb-position-controls">
          <SegmentedControl
            label="Word order"
            value={order}
            options={ORDER_IDS.map((id) => ({ value: id, label: ORDER_LABELS[id] }))}
            onChange={(value) => {
              setState({ order: value });
              narrate(describeOrder(value as OrderId, positionInfo));
            }}
          />
          <SegmentedControl
            label="Position information"
            value={positionInfo ? "on" : "off"}
            options={[
              { value: "off", label: "Off" },
              { value: "on", label: "On" },
            ]}
            onChange={(value) => {
              setState({ positionInfo: value === "on" });
              narrate(describeOrder(order, value === "on"));
            }}
          />
        </div>
        <div className="tb-order-pair" role="group" aria-label="The two sentences being compared">
          {[
            { name: "Original", words: orderResult.original },
            { name: "Reordered", words: orderResult.reordered },
          ].map((line) => (
            <div key={line.name} className="tb-order-line">
              <span>{line.name}</span>
              <ol aria-label={`${line.name} order: ${line.words.join(" ")}`}>
                {line.words.map((word, slot) => (
                  <li key={slot}>
                    <small>{slot}</small>
                    {word}
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
        <table className="tb-order-table">
          <caption>
            Each word's block output in the original order against the reordered one, {positionInfo ? "with" : "without"}{" "}
            position information
          </caption>
          <thead>
            <tr>
              <th scope="col">Word</th>
              <th scope="col">Slot</th>
              <th scope="col">Output shift</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {orderResult.rows.map((row) => {
              const moved = row.shift > ORDER_TOLERANCE;
              return (
                <tr key={row.from} className={moved ? "is-changed" : "is-same"}>
                  <th scope="row">{row.word}</th>
                  <td>
                    {row.from} → {row.to}
                  </td>
                  <td>
                    <span className="tb-order-bar" aria-hidden="true">
                      <i style={{ width: `${Math.min(100, (row.shift / Math.max(1, orderResult.largestShift)) * 100)}%` }} />
                    </span>
                    <code>{fixed(row.shift, 3)}</code>
                  </td>
                  <td>{moved ? "different output" : "same output"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric
            label="Words whose output changed"
            value={`${orderResult.changed} of ${TOKENS.length}`}
            tone={orderResult.changed > 0 ? "gradient" : undefined}
          />
          <Metric label="Largest output shift" value={fixed(orderResult.largestShift, 3)} />
        </div>
        <p className="tb-note">
          {order === "original"
            ? "The two sentences are the same, so every shift is 0.000. Pick a reordering to compare."
            : positionInfo
              ? `Position information is on, so each word's input carries its slot and ${orderResult.changed} of ${TOKENS.length} outputs changed. The block can tell the two sentences apart.`
              : `Position information is off, so the block holds only each word's own vector. The six outputs are the same vectors as before, only in new slots (largest shift ${fixed(orderResult.largestShift, 3)}), and it cannot tell the two sentences apart.`}{" "}
          This card lifts the causal mask, so with Self-attention on every word reads every word. That keeps the test about
          order alone.
        </p>
      </LabSurface>
    </div>
  );
}
