import { useMemo } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { PipelineSchedule } from "./PipelineSchedule";
import {
  CHECKPOINT_MINUTES,
  DATA_LADDER,
  effectiveExpert,
  EXPERT_LADDER,
  failurePlan,
  formatBytes,
  formatCount,
  formatHours,
  formatMinutes,
  formatPercent,
  formatSeconds,
  gpuCount,
  HARDWARE,
  isMoe,
  linkFor,
  MEMORY_LABEL,
  memoryPerGpu,
  MICRO_BATCH_LADDER,
  MICROBATCHES_LADDER,
  modelById,
  MODELS,
  parameterCounts,
  PIPELINE_LADDER,
  RECOMPUTE,
  RECOMPUTE_LABELS,
  RESTART_MINUTES,
  SAVE_SECONDS,
  SEQUENCE,
  stepTime,
  TENSOR_LADDER,
  timeLost,
  ZERO_STAGES,
  type Layout,
  type MemoryPerGpu,
  type Recompute,
  type ZeroStage,
} from "./parallel";
import { LLAMA3_LAYOUT, readState } from "./state";

const ZERO_NOTES: Record<ZeroStage, string> = {
  0: "Stage 0 shards nothing: every data-parallel GPU keeps its own full copy of the weights, gradients, and optimizer state.",
  1: "Stage 1 shards the optimizer state: each data-parallel GPU keeps 1/d of the fp32 master copy, momentum, and variance.",
  2: "Stage 2 also shards the gradients: each GPU reduce-scatters them and keeps only the 1/d it updates.",
  3: "Stage 3 also shards the weights: each GPU keeps 1/d and gathers the rest layer by layer, at 1.5 times the traffic.",
};

const RECOMPUTE_NOTES: Record<Recompute, string> = {
  none: "None keeps every intermediate of every layer, including the attention scores, which grow with the square of the sequence length.",
  selective: "Selective drops the attention-score intermediates and recomputes them in the backward pass, which FlashAttention-style kernels do anyway.",
  full: "Full keeps only each layer's input and reruns the whole forward pass in the backward pass: a third more arithmetic.",
};

const indexOn = (ladder: ReadonlyArray<number>, value: number) => Math.max(0, ladder.indexOf(value));
const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;

function MemoryBar({ memory }: { memory: MemoryPerGpu }) {
  const left = 12;
  const right = 708;
  const limit = HARDWARE.memoryBytes;
  const scale = Math.max(limit * 1.12, memory.total * 1.04);
  const xAt = (bytes: number) => left + (bytes / scale) * (right - left);
  const top = 30;
  const barHeight = 34;
  const segments = [
    { id: "weights", bytes: memory.weights },
    { id: "gradients", bytes: memory.gradients },
    { id: "optimizer", bytes: memory.optimizer },
    { id: "activations", bytes: memory.activations },
  ];
  let start = 0;
  const placed = segments.map((segment) => {
    const at = start;
    start += segment.bytes;
    return { ...segment, start: at };
  });
  const limitOnRight = xAt(limit) > (left + right) / 2;
  const tickStep = [10e9, 20e9, 50e9, 100e9, 200e9, 500e9, 1000e9].find((step) => scale / step <= 9) ?? 2000e9;
  const ticks: number[] = [];
  for (let bytes = 0; bytes <= scale && ticks.length < 12; bytes += tickStep) ticks.push(bytes);
  return (
    <svg
      className="par-plot par-memory-bar"
      viewBox="0 0 720 96"
      role="img"
      aria-label={`Memory on one GPU: weights ${formatBytes(memory.weights)}, gradients ${formatBytes(memory.gradients)}, optimizer state ${formatBytes(memory.optimizer)}, activations ${formatBytes(memory.activations)}; ${formatBytes(memory.total)} in all against ${MEMORY_LABEL}. ${memory.fits ? "It fits." : "It does not fit."}`}
    >
      {ticks.map((bytes) => (
        <g key={bytes}>
          <line className="par-grid" x1={xAt(bytes)} x2={xAt(bytes)} y1={top - 4} y2={top + barHeight + 4} />
          <text x={xAt(bytes)} y={top + barHeight + 16} textAnchor="middle">
            {Math.round(bytes / 1e9)}
          </text>
        </g>
      ))}
      {placed.map((segment) => (
        <rect
          key={segment.id}
          className={`par-seg is-${segment.id}`}
          x={xAt(segment.start)}
          y={top}
          width={Math.max(0, xAt(segment.start + segment.bytes) - xAt(segment.start))}
          height={barHeight}
        />
      ))}
      {!memory.fits && (
        <rect className="par-over" x={xAt(limit)} y={top - 6} width={Math.max(0, xAt(memory.total) - xAt(limit))} height={barHeight + 12} />
      )}
      <line className="par-limit" x1={xAt(limit)} x2={xAt(limit)} y1={top - 12} y2={top + barHeight + 6} />
      {/* The limit's label sits on whichever side of its line has room; the axis note takes the other end. */}
      <text
        className="par-limit-label"
        x={xAt(limit) + (limitOnRight ? -4 : 4)}
        y={top - 16}
        textAnchor={limitOnRight ? "end" : "start"}
      >
        {MEMORY_LABEL} on one H100
      </text>
      <text x={limitOnRight ? left : right} y={top - 16} textAnchor={limitOnRight ? "start" : "end"}>
        {memory.fits ? `used ${formatBytes(memory.total)} · axis in GB` : "axis in GB"}
      </text>
    </svg>
  );
}

function LostPlot({
  mtbfSeconds,
  saveSeconds,
  restartSeconds,
  intervalMinutes,
  youngSeconds,
  dalySeconds,
}: {
  mtbfSeconds: number;
  saveSeconds: number;
  restartSeconds: number;
  intervalMinutes: number;
  youngSeconds: number;
  dalySeconds: number;
}) {
  const left = 52;
  const right = 704;
  const top = 18;
  const bottom = 176;
  const { min, max } = CHECKPOINT_MINUTES;
  const xAt = (minutes: number) => left + ((minutes - min) / (max - min)) * (right - left);
  const lost = (minutes: number) => timeLost(minutes * 60, saveSeconds, mtbfSeconds, restartSeconds);
  const samples = Array.from({ length: max - min + 1 }, (_, index) => min + index);
  const values = samples.map(lost);
  const ceiling = Math.min(1, Math.max(0.02, ...values.map((value) => Math.min(value, 1))) * 1.08);
  const yAt = (fraction: number) => bottom - (Math.min(fraction, ceiling) / ceiling) * (bottom - top);
  const path = samples.map((minutes, index) => `${index === 0 ? "M" : "L"}${xAt(minutes).toFixed(1)},${yAt(values[index]).toFixed(1)}`).join(" ");
  const yTickStep = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((step) => ceiling / step <= 5) ?? 0.25;
  const yTicks: number[] = [];
  for (let fraction = 0; fraction <= ceiling + 1e-9 && yTicks.length < 8; fraction += yTickStep) yTicks.push(fraction);
  const markers = [
    { id: "young", label: "Young", seconds: youngSeconds },
    { id: "daly", label: "Daly", seconds: dalySeconds },
  ];
  const currentLost = lost(intervalMinutes);
  return (
    <svg
      className="par-plot par-lost-plot"
      viewBox="0 0 720 214"
      role="img"
      aria-label={`Share of time lost against the checkpoint interval from ${min} to ${max} minutes. At ${intervalMinutes} minutes it is ${formatPercent(Math.min(currentLost, 1))}${currentLost >= 1 ? " or more" : ""}. Young's optimum is ${formatMinutes(youngSeconds)} and Daly's is ${formatMinutes(dalySeconds)}.`}
    >
      {yTicks.map((fraction) => (
        <g key={fraction}>
          <line className="par-grid" x1={left} x2={right} y1={yAt(fraction)} y2={yAt(fraction)} />
          <text x={left - 6} y={yAt(fraction) + 3} textAnchor="end">
            {Math.round(fraction * 100)}%
          </text>
        </g>
      ))}
      {[5, 20, 40, 60, 80, 100, 120].map((minutes) => (
        <text key={minutes} x={xAt(minutes)} y={bottom + 15} textAnchor="middle">
          {minutes}
        </text>
      ))}
      <text x={(left + right) / 2} y={bottom + 31} textAnchor="middle">
        checkpoint every … minutes
      </text>
      <path className="par-lost-curve" d={path} />
      {markers.map((marker, index) => {
        const minutes = marker.seconds / 60;
        const inside = minutes >= min && minutes <= max;
        const labelY = top + 10 + index * 13;
        if (!inside) {
          return (
            <text key={marker.id} className={`par-optimum-label is-${marker.id}`} x={minutes > max ? right : left + 4} y={labelY} textAnchor={minutes > max ? "end" : "start"}>
              {marker.label}: {formatMinutes(marker.seconds)} {minutes > max ? "→" : "←"}
            </text>
          );
        }
        return (
          <g key={marker.id} className={`par-optimum is-${marker.id}`}>
            <line x1={xAt(minutes)} x2={xAt(minutes)} y1={top} y2={bottom} />
            <text x={xAt(minutes) + (index === 0 ? 5 : -5)} y={labelY} textAnchor={index === 0 ? "start" : "end"}>
              {marker.label} {formatMinutes(marker.seconds)}
            </text>
          </g>
        );
      })}
      <line className="marker-guide" x1={xAt(intervalMinutes)} x2={xAt(intervalMinutes)} y1={top} y2={bottom} />
      <circle className="par-lost-dot" cx={xAt(intervalMinutes)} cy={yAt(currentLost)} r={5} />
    </svg>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const config = readState(state);
  const model = modelById(config.modelId);
  const moe = isMoe(model);
  const counts = parameterCounts(model);
  const layout: Layout = useMemo(
    () => ({
      model,
      tensor: config.tensor,
      pipeline: config.pipeline,
      data: config.data,
      expert: config.expert,
      zero: config.zero,
      microBatch: config.microBatch,
      microbatches: config.microbatches,
      recompute: config.recompute,
      overlap: config.overlap,
    }),
    [
      model,
      config.tensor,
      config.pipeline,
      config.data,
      config.expert,
      config.zero,
      config.microBatch,
      config.microbatches,
      config.recompute,
      config.overlap,
    ],
  );
  const memory = useMemo(() => memoryPerGpu(layout), [layout]);
  const timing = useMemo(() => stepTime(layout), [layout]);
  const gpus = gpuCount(layout);
  const expert = effectiveExpert(layout);
  const plan = failurePlan(model, gpus, config.checkpointMinutes, config.saveSeconds, config.restartMinutes);
  const nodes = Math.ceil(gpus / HARDWARE.gpusPerNode);
  const layersPerStage = model.layers / config.pipeline;
  const stageLayers = Number.isInteger(layersPerStage) ? `${layersPerStage}` : `${Math.floor(layersPerStage)}–${Math.ceil(layersPerStage)}`;
  const tensorLink = linkFor(1, config.tensor);
  const dataLink = linkFor(config.tensor, config.data);
  const pipelineLink = linkFor(config.tensor * config.data, config.pipeline);
  const expertLink = linkFor(config.tensor, expert);
  const lostCapped = Math.min(plan.lost, 1);
  /** More than two percentage points worse than Young's optimum, or past the point where the formula holds. */
  const lostBadly = plan.lost >= 1 || plan.lost - plan.lostAtYoung > 0.02;
  const stageLayerCount = Number.isInteger(layersPerStage) ? `${layersPerStage}` : layersPerStage.toFixed(1);
  /** Mixtral's experts are replicated only d ÷ e times, so ZeRO and the gradient sync split them over fewer GPUs. */
  const expertSharded = moe && expert > 1;
  const shardNote = (stage: number) =>
    config.zero >= stage
      ? `, ÷ ${config.data}${expertSharded ? ` (experts ÷ ${config.data / expert})` : ""} by ZeRO ${stage === 3 ? "3" : `${stage} and up`}`
      : "";

  const update = (patch: Partial<typeof config>, lead: string) => {
    setState(patch);
    const next = readState({ ...config, ...patch });
    const nextLayout: Layout = { ...layout, ...next, model: modelById(next.modelId) };
    const nextMemory = memoryPerGpu(nextLayout);
    const nextStep = stepTime(nextLayout);
    narrate(
      `${lead} ${gpuCount(nextLayout).toLocaleString("en-US")} GPUs. ${formatBytes(nextMemory.total)} per GPU, ${nextMemory.fits ? "fits" : "does not fit"} in 80 GB. Step ${formatSeconds(nextStep.step)}, MFU ${formatPercent(nextStep.mfu)}.`,
    );
  };

  const stepParts = [
    {
      id: "compute",
      label: "Compute",
      value: timing.parts.compute,
      tone: "forward" as const,
      detail: `${config.microbatches} microbatches × ${formatSeconds(timing.computePerMicrobatch / (config.recompute === "full" ? 4 / 3 : 1))} of model arithmetic at ${Math.round(HARDWARE.efficiency * 100)}% of peak`,
    },
    {
      id: "recompute",
      label: "Recompute",
      value: timing.parts.recompute,
      tone: "gradient" as const,
      detail:
        config.recompute === "full"
          ? "the forward pass run a second time during the backward pass: a third more arithmetic"
          : "none charged: only Full recompute reruns the forward pass here",
    },
    {
      id: "communication",
      label: "TP, PP and EP traffic",
      value: timing.parts.communication,
      tone: "attention" as const,
      detail: `${config.microbatches} microbatches × ${formatSeconds(timing.tensor.seconds + timing.pipelineSend.seconds + timing.expertAllToAll.seconds)} on the critical path, not overlapped`,
    },
    {
      id: "bubble",
      label: "Pipeline bubble",
      value: timing.parts.bubble,
      tone: "muted" as const,
      detail: `${config.pipeline - 1} idle microbatch slot${config.pipeline - 1 === 1 ? "" : "s"} of ${formatSeconds(timing.microbatchTime)} while the pipeline fills and drains`,
    },
    {
      id: "sync",
      label: "Exposed DP sync",
      value: timing.parts.exposedDataSync,
      tone: "loss" as const,
      detail: config.overlap
        ? "what is left of the gradient sync after hiding it under the last backward pass"
        : "the whole gradient sync, waited for after the last backward pass",
    },
  ];

  const commRows = [
    {
      id: "tp",
      label: "TP all-reduce",
      value: config.microbatches * timing.tensor.seconds,
      display: config.tensor > 1 ? formatSeconds(config.microbatches * timing.tensor.seconds) : "none",
      tone: "attention" as const,
      detail:
        config.tensor > 1
          ? `${formatSeconds(timing.tensor.seconds)} per microbatch: ${config.recompute === "full" ? 6 : 4} all-reduces per layer × ${stageLayerCount} layers, ${formatBytes(timing.tensor.bytesPerCall)} each, over ${plural(config.tensor, "GPU")} on ${tensorLink.label}${tensorLink.id === "network" ? ", because the group no longer fits in one node" : ""}`
          : "Tensor parallel is 1, so no layer is split and nothing is all-reduced",
    },
    {
      id: "pp",
      label: "PP send/receive",
      value: config.microbatches * timing.pipelineSend.seconds,
      display: config.pipeline > 1 ? formatSeconds(config.microbatches * timing.pipelineSend.seconds) : "none",
      tone: "gradient" as const,
      detail:
        config.pipeline > 1
          ? `${formatSeconds(timing.pipelineSend.seconds)} per microbatch: activations forward and gradients back, ${formatBytes(timing.pipelineSend.bytesPerCall)} each, on ${pipelineLink.label}`
          : "Pipeline is 1, so there are no stage boundaries to cross",
    },
    ...(moe
      ? [
          {
            id: "ep",
            label: "EP all-to-all",
            value: config.microbatches * timing.expertAllToAll.seconds,
            display: expert > 1 ? formatSeconds(config.microbatches * timing.expertAllToAll.seconds) : "none",
            tone: "forward" as const,
            detail:
              expert > 1
                ? `${formatSeconds(timing.expertAllToAll.seconds)} per microbatch: 4 all-to-alls of ${formatBytes(timing.expertAllToAll.bytesPerCall)} per MoE layer across ${plural(expert, "GPU")} on ${expertLink.label}`
                : "Expert parallel is 1, so every GPU holds all 8 experts and no tokens move",
          },
        ]
      : []),
    {
      id: "dp",
      label: "DP gradient sync",
      value: timing.dataSync.seconds,
      display: config.data > 1 ? formatSeconds(timing.dataSync.seconds) : "none",
      tone: "loss" as const,
      detail:
        config.data > 1
          ? `once per step: ${formatBytes(timing.dataSync.bytesPerCall)} of bf16 gradients over ${plural(config.data, "replica")}${expertSharded ? ` (each expert's over ${config.data / expert})` : ""} on ${dataLink.label}${config.zero === 3 ? ", × 1.5 to gather ZeRO-3 weights" : ""}; ${formatSeconds(timing.exposedDataSync)} left exposed`
          : "Data parallel is 1, so there is no other replica to sync with",
    },
  ];

  return (
    <div className="tg-lab par-lab">
      <LabSurface label="Parallel layout" className="par-layout-card">
        <SurfaceHeading
          kicker={`${model.label} · ${formatCount(counts.total)} parameters${moe ? `, ${formatCount(counts.active)} active` : ""} · ${SEQUENCE.toLocaleString("en-US")}-token sequences · ${HARDWARE.name}`}
          title="Split one training step across GPUs"
          aside={<span className="tg-badge">{plural(gpus, "GPU")}</span>}
        />
        <div className="par-model-row">
          <SegmentedControl
            label="Model"
            value={model.id}
            options={MODELS.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => update({ modelId: modelById(value).id }, `${modelById(value).label}.`)}
          />
          <button
            type="button"
            className="quiet-action par-preset"
            onClick={() => update({ ...LLAMA3_LAYOUT }, "16,384-GPU layout: Llama 3.1 405B on tensor 8, pipeline 16, data 128.")}
          >
            16,384-GPU layout
          </button>
        </div>
        <div className="par-controls">
          <RangeControl
            label="Tensor parallel"
            min={0}
            max={TENSOR_LADDER.length - 1}
            step={1}
            value={indexOn(TENSOR_LADDER, config.tensor)}
            format={(index) => plural(TENSOR_LADDER[index] ?? 1, "GPU") + " per layer"}
            onChange={(index) => update({ tensor: TENSOR_LADDER[index] ?? 1 }, `Tensor parallel ${TENSOR_LADDER[index] ?? 1}.`)}
          />
          <RangeControl
            label="Pipeline"
            min={0}
            max={PIPELINE_LADDER.length - 1}
            step={1}
            value={indexOn(PIPELINE_LADDER, config.pipeline)}
            format={(index) => plural(PIPELINE_LADDER[index] ?? 1, "stage")}
            onChange={(index) => update({ pipeline: PIPELINE_LADDER[index] ?? 1 }, `Pipeline ${PIPELINE_LADDER[index] ?? 1}.`)}
          />
          <RangeControl
            label="Data parallel"
            min={0}
            max={DATA_LADDER.length - 1}
            step={1}
            value={indexOn(DATA_LADDER, config.data)}
            format={(index) => plural(DATA_LADDER[index] ?? 1, "replica")}
            onChange={(index) => {
              const data = DATA_LADDER[index] ?? 1;
              update({ data, expert: Math.min(config.expert, data) }, `Data parallel ${data}.`);
            }}
          />
          {moe && (
            <RangeControl
              label="Expert parallel"
              min={0}
              max={Math.min(EXPERT_LADDER.length - 1, Math.log2(config.data))}
              step={1}
              value={indexOn(EXPERT_LADDER, expert)}
              format={(index) => {
                const e = EXPERT_LADDER[index] ?? 1;
                return `${plural(e, "GPU")} · ${plural(model.experts / e, "expert")} each`;
              }}
              onChange={(index) => update({ expert: EXPERT_LADDER[index] ?? 1 }, `Expert parallel ${EXPERT_LADDER[index] ?? 1}.`)}
            />
          )}
          <RangeControl
            label="Micro-batch size"
            min={0}
            max={MICRO_BATCH_LADDER.length - 1}
            step={1}
            value={indexOn(MICRO_BATCH_LADDER, config.microBatch)}
            format={(index) => `${plural(MICRO_BATCH_LADDER[index] ?? 1, "sequence")} of ${SEQUENCE.toLocaleString("en-US")}`}
            onChange={(index) => update({ microBatch: MICRO_BATCH_LADDER[index] ?? 1 }, `Micro-batch size ${MICRO_BATCH_LADDER[index] ?? 1}.`)}
          />
          <RangeControl
            label="Microbatches"
            min={0}
            max={MICROBATCHES_LADDER.length - 1}
            step={1}
            value={indexOn(MICROBATCHES_LADDER, config.microbatches)}
            format={(index) => `${MICROBATCHES_LADDER[index] ?? 1} per step`}
            onChange={(index) => update({ microbatches: MICROBATCHES_LADDER[index] ?? 1 }, `${MICROBATCHES_LADDER[index] ?? 1} microbatches.`)}
          />
        </div>
        <div className="metric-row">
          <Metric label="GPUs" value={gpus.toLocaleString("en-US")} />
          <Metric label="Nodes of 8" value={nodes.toLocaleString("en-US")} />
          <Metric label="Tokens per step" value={formatCount(timing.tokensPerStep)} />
          <Metric label="Parameters per GPU" value={formatCount(memory.sliceParameters)} />
        </div>
        <ul className="par-groups" aria-label="What each parallel degree splits and sends">
          <li>
            <strong>Tensor parallel · {config.tensor}</strong>
            <span>
              {config.tensor > 1
                ? `Splits every weight matrix ${config.tensor} ways and all-reduces activations ${config.recompute === "full" ? 6 : 4} times per layer per microbatch, over ${tensorLink.label}${tensorLink.id === "nvlink" ? " inside one node" : " between nodes"}.`
                : "Off: each GPU holds whole layers."}
            </span>
          </li>
          <li>
            <strong>Pipeline · {config.pipeline}</strong>
            <span>
              {config.pipeline > 1
                ? `Splits the ${model.layers} layers into ${config.pipeline} stages of ${stageLayers} and passes activations and gradients between neighbours, over ${pipelineLink.label}.`
                : `Off: every GPU runs all ${model.layers} layers.`}
            </span>
          </li>
          <li>
            <strong>Data parallel · {config.data}</strong>
            <span>
              {config.data > 1
                ? `Splits the batch into ${config.data} shares and all-reduces gradients once per step, over ${dataLink.label}.`
                : "Off: one replica sees the whole batch."}
            </span>
          </li>
          {moe && (
            <li>
              <strong>Expert parallel · {expert}</strong>
              <span>
                {expert > 1
                  ? `Spreads the ${model.experts} experts of each layer over ${expert} of the data-parallel GPUs and sends each token to its ${model.topK} experts by all-to-all, over ${expertLink.label}.`
                  : `Off: every GPU holds all ${model.experts} experts of each layer.`}
              </span>
            </li>
          )}
        </ul>
        <p className="lab-note">
          GPUs = tensor × pipeline × data = {config.tensor} × {config.pipeline} × {config.data}. Ranks are placed
          tensor-first, {HARDWARE.gpusPerNode} to a node: a group whose GPUs all sit in one node talks over NVLink at{" "}
          {HARDWARE.nvlinkBytesPerSecond / 1e9} GB/s each way, and any group that spans nodes is limited by InfiniBand at{" "}
          {HARDWARE.networkBytesPerSecond / 1e9} GB/s per GPU. Every readout below is closed-form arithmetic on these
          stated numbers and the model's published shape, not a measurement.
        </p>
      </LabSurface>

      <LabSurface label="Memory per GPU" className="par-memory-card">
        <SurfaceHeading
          kicker={`ZeRO stage ${config.zero} · ${RECOMPUTE_LABELS[config.recompute]} recompute · ${plural(config.microBatch, "sequence")} per microbatch`}
          title="What one GPU has to hold"
          aside={
            <span className={`tg-badge ${memory.fits ? "" : "is-warning"}`.trim()}>
              {formatBytes(memory.total)} of {MEMORY_LABEL}
            </span>
          }
        />
        <div className="par-memory-controls">
          <div className="par-zero-control">
            <SegmentedControl
              label="ZeRO stage"
              value={`${config.zero}`}
              options={ZERO_STAGES.map((stage) => ({ value: `${stage}`, label: `${stage}` }))}
              onChange={(value) => update({ zero: Number(value) as ZeroStage }, `ZeRO stage ${value}.`)}
            />
          </div>
          <div className="par-recompute-control">
            <SegmentedControl
              label="Recompute"
              value={config.recompute}
              options={RECOMPUTE.map((entry) => ({ value: entry, label: RECOMPUTE_LABELS[entry] }))}
              onChange={(value) => {
                const next = RECOMPUTE.find((entry) => entry === value) ?? "selective";
                update({ recompute: next }, `${RECOMPUTE_LABELS[next]} recompute.`);
              }}
            />
          </div>
        </div>
        <div className="par-budget">
          <MemoryBar memory={memory} />
          <div className="par-legend" aria-hidden="true">
            <span className="is-weights">weights</span>
            <span className="is-gradients">gradients</span>
            <span className="is-optimizer">optimizer state</span>
            <span className="is-activations">activations</span>
          </div>
        </div>
        <BarList
          label="Memory per GPU, part by part"
          max={Math.max(1, memory.weights, memory.gradients, memory.optimizer, memory.activations)}
          items={[
            {
              id: "weights",
              label: "Weights",
              value: memory.weights,
              display: formatBytes(memory.weights),
              tone: "forward",
              detail: `2 bytes × ${formatCount(memory.sliceParameters)} parameters${shardNote(3)}`,
            },
            {
              id: "gradients",
              label: "Gradients",
              value: memory.gradients,
              display: formatBytes(memory.gradients),
              tone: "attention",
              detail: `2 bytes × ${formatCount(memory.sliceParameters)}${shardNote(2)}`,
            },
            {
              id: "optimizer",
              label: "Optimizer state",
              value: memory.optimizer,
              display: formatBytes(memory.optimizer),
              tone: "gradient",
              detail: `12 bytes (fp32 master copy, momentum, variance) × ${formatCount(memory.sliceParameters)}${shardNote(1)}`,
            },
            {
              id: "activations",
              label: "Activations",
              value: memory.activations,
              display: formatBytes(memory.activations),
              tone: "loss",
              detail: `${plural(memory.layersPerStage, "layer")} × ${plural(memory.inFlight, "microbatch", "microbatches")} in flight × ${formatBytes(memory.activationPerLayer)}`,
            },
          ]}
        />
        <div className="metric-row">
          <Metric label="Model state" value={formatBytes(memory.modelState)} />
          <Metric label="Activations" value={formatBytes(memory.activations)} />
          <Metric label="Total per GPU" value={formatBytes(memory.total)} tone={memory.fits ? "forward" : "loss"} />
          <Metric label="Fits in 80 GB" value={memory.fits ? "yes" : "no"} tone={memory.fits ? "forward" : "loss"} />
        </div>
        <p className="lab-note">
          {ZERO_NOTES[config.zero]} {RECOMPUTE_NOTES[config.recompute]} Tensor and pipeline parallelism divide every
          part; data parallelism divides only what ZeRO shards. Activations follow Korthikanti et al.'s per-layer count for
          16-bit activations, with sequence parallelism when tensor parallel is above 1, and the first stage holds{" "}
          {plural(memory.inFlight, "microbatch", "microbatches")} at once under 1F1B. Temporary buffers, the output logits,
          and fragmentation are left out, so a configuration that fits on paper needs headroom in practice.
        </p>
      </LabSurface>

      <LabSurface label="Pipeline schedule" className="par-schedule-card">
        <SurfaceHeading
          kicker={`${plural(config.pipeline, "stage")} × ${plural(config.microbatches, "microbatch", "microbatches")} · one forward, one backward (1F1B)`}
          title="Idle time while the pipeline fills and drains"
          aside={<span className={`tg-badge ${timing.bubbleFraction >= 0.5 ? "is-warning" : ""}`.trim()}>bubble {formatPercent(timing.bubbleFraction)}</span>}
        />
        <PipelineSchedule
          stages={config.pipeline}
          microbatches={config.microbatches}
          microbatchSeconds={timing.microbatchTime}
          exposedSyncSeconds={timing.exposedDataSync}
          bubbleFraction={timing.bubbleFraction}
        />
        <div className="metric-row">
          <Metric label="Bubble" value={formatPercent(timing.bubbleFraction)} tone={timing.bubbleFraction >= 0.25 ? "loss" : "forward"} />
          <Metric label="Microbatch time" value={formatSeconds(timing.microbatchTime)} />
          <Metric label="Pipeline time" value={formatSeconds(timing.pipelineTime)} />
          <Metric label="In flight on stage 1" value={`${memory.inFlight}`} />
        </div>
        <p className="lab-note">
          Bubble = (p − 1) / (m + p − 1) = ({config.pipeline} − 1) / ({config.microbatches} + {config.pipeline} − 1). A
          stage cannot start a microbatch until the stage before it has finished that microbatch, so the last stage waits
          at the start and the first waits at the end. More microbatches shrink the bubble without changing the work, but
          each one is a gradient-accumulation step, so they also grow the batch. Every stage is charged an equal share of
          the layers.
        </p>
      </LabSurface>

      <LabSurface label="Communication per step" className="par-comm-card">
        <SurfaceHeading
          kicker={`NVLink ${HARDWARE.nvlinkBytesPerSecond / 1e9} GB/s inside a node · InfiniBand ${HARDWARE.networkBytesPerSecond / 1e9} GB/s per GPU between nodes`}
          title="What crosses the wires in one step"
          aside={<span className="tg-badge">{formatSeconds(timing.exposedDataSync)} exposed</span>}
        />
        <div className="tg-switch-list par-overlap-switch">
          <button
            type="button"
            className="tg-switch"
            aria-pressed={config.overlap}
            onClick={() => update({ overlap: !config.overlap }, config.overlap ? "Overlap off." : "Overlap on.")}
          >
            <strong>Overlap gradient sync</strong>
            <small>
              Start all-reducing each layer's gradients during the last microbatch's backward pass instead of waiting for it
              to finish.
            </small>
            <b>{config.overlap ? "on" : "off"}</b>
          </button>
        </div>
        <BarList
          label="Seconds of communication per step for each kind of parallelism"
          max={Math.max(1e-9, ...commRows.map((row) => row.value))}
          items={commRows}
        />
        <div className="metric-row">
          <Metric label="Exposed DP time" value={formatSeconds(timing.exposedDataSync)} tone={timing.exposedDataSync > 0 ? "loss" : "forward"} />
          <Metric label="TP group" value={config.tensor > 1 ? `${config.tensor} · ${tensorLink.label}` : "off"} tone={tensorLink.id === "network" ? "loss" : undefined} />
          <Metric label="DP group" value={config.data > 1 ? `${config.data} · ${dataLink.label}` : "off"} />
        </div>
        <p className="lab-note">
          Bandwidth terms only: a ring all-reduce over n GPUs moves 2(n − 1)/n of the buffer, a reduce-scatter, all-gather,
          or all-to-all moves (n − 1)/n, and a pipeline send moves the buffer once. Latency, congestion, and stragglers are
          left out. The DP sync is the step's only once-per-step collective; with overlap on, everything beyond two thirds of
          one microbatch's compute, the last backward pass, is left exposed.
        </p>
      </LabSurface>

      <LabSurface label="Step time and MFU" className="par-step-card">
        <SurfaceHeading
          kicker={`${formatCount(timing.tokensPerStep)} tokens per step on ${plural(gpus, "GPU")} · kernel efficiency η = ${HARDWARE.efficiency} of ${HARDWARE.peakFlops / 1e12} TFLOP/s`}
          title="Compute, exposed communication, and idle time"
          aside={<span className="tg-badge">MFU {formatPercent(timing.mfu)}</span>}
        />
        <div
          className="par-step-strip"
          role="img"
          aria-label={`Step of ${formatSeconds(timing.step)}: ${stepParts
            .filter((part) => part.value > 0)
            .map((part) => `${part.label.toLowerCase()} ${formatPercent(part.value / timing.step)}`)
            .join(", ")}.`}
        >
          {stepParts
            .filter((part) => part.value / timing.step >= 0.002)
            .map((part) => (
              <i key={part.id} className={`is-${part.id}`} style={{ flexGrow: part.value }}>
                {part.value / timing.step >= 0.09 && <span>{part.label}</span>}
              </i>
            ))}
        </div>
        <BarList
          label="The step time split into what the GPUs were doing"
          max={Math.max(1e-9, timing.step)}
          items={stepParts.map((part) => ({
            id: part.id,
            label: part.label,
            value: part.value,
            display: formatSeconds(part.value),
            tone: part.tone,
            detail: `${formatPercent(part.value / timing.step)} of the step · ${part.detail}`,
          }))}
        />
        <div className="metric-row">
          <Metric label="Step time" value={formatSeconds(timing.step)} />
          <Metric label="Tokens/s" value={formatCount(timing.tokensPerSecond)} tone="forward" />
          <Metric label="MFU" value={formatPercent(timing.mfu)} tone="forward" />
          <Metric label="HFU" value={formatPercent(timing.hfu)} tone="gradient" />
        </div>
        <p className="lab-note">
          MFU = model FLOPs per token × tokens per second ÷ (GPUs × peak), with model FLOPs per token = 6 × active
          parameters + 12 × layers × hidden size × sequence length, the PaLM definition: recomputed work does not count.
          HFU counts it, so Full recompute raises HFU and lowers MFU. Kernel efficiency η = {HARDWARE.efficiency} is a stated
          constant that caps MFU at {Math.round(HARDWARE.efficiency * 100)}%; tensor, pipeline, and expert traffic is charged in
          full, never overlapped, and the optimizer step itself is not timed.
        </p>
      </LabSurface>

      <LabSurface label="Failures and checkpoints" className="par-failure-card">
        <SurfaceHeading
          kicker={`${plural(gpus, "GPU")} · one interruption every ${formatHours(plan.mtbfHours)} on average · ${formatBytes(plan.checkpointBytes)} per checkpoint`}
          title="How often to save"
          aside={<span className={`tg-badge ${lostBadly ? "is-warning" : ""}`.trim()}>{plan.lost >= 1 ? "100%+" : formatPercent(lostCapped)} lost</span>}
        />
        <div className="par-controls par-failure-controls">
          <RangeControl
            label="Checkpoint every"
            min={CHECKPOINT_MINUTES.min}
            max={CHECKPOINT_MINUTES.max}
            step={CHECKPOINT_MINUTES.step}
            value={config.checkpointMinutes}
            format={(value) => `${value} min`}
            onChange={(value) => {
              setState({ checkpointMinutes: value });
              const lost = timeLost(value * 60, config.saveSeconds, plan.mtbfSeconds, config.restartMinutes * 60);
              narrate(`Checkpoint every ${value} minutes: ${lost >= 1 ? "100 percent or more" : formatPercent(lost)} of time lost.`);
            }}
          />
          <RangeControl
            label="Save time"
            min={SAVE_SECONDS.min}
            max={SAVE_SECONDS.max}
            step={SAVE_SECONDS.step}
            value={config.saveSeconds}
            format={(value) => `${value} s`}
            onChange={(value) => setState({ saveSeconds: value })}
          />
          <RangeControl
            label="Restart time"
            min={RESTART_MINUTES.min}
            max={RESTART_MINUTES.max}
            step={RESTART_MINUTES.step}
            value={config.restartMinutes}
            format={(value) => `${value} min`}
            onChange={(value) => setState({ restartMinutes: value })}
          />
        </div>
        <LostPlot
          mtbfSeconds={plan.mtbfSeconds}
          saveSeconds={config.saveSeconds}
          restartSeconds={config.restartMinutes * 60}
          intervalMinutes={config.checkpointMinutes}
          youngSeconds={plan.youngSeconds}
          dalySeconds={plan.dalySeconds}
        />
        <div className="metric-row">
          <Metric label="Job MTBF" value={formatHours(plan.mtbfHours)} />
          <Metric label="Young optimum" value={formatMinutes(plan.youngSeconds)} />
          <Metric label="Daly optimum" value={formatMinutes(plan.dalySeconds)} />
          <Metric label="Time lost" value={plan.lost >= 1 ? "100%+" : formatPercent(lostCapped)} tone={lostBadly ? "loss" : "forward"} />
        </div>
        <p className="lab-note">
          Job MTBF = 50,677 GPU-hours ÷ {gpus.toLocaleString("en-US")} GPUs. The 50,677 comes from the Llama 3 paper's
          54-day snapshot, 419 unexpected interruptions, assuming all 16,384 GPUs ran throughout. Time lost = save time ÷
          interval + (half an interval of redone work + restart) ÷ MTBF, which Young's interval √(2 × save × MTBF) minimizes;
          Daly's higher-order estimate is a little shorter. A checkpoint holds 14 bytes per parameter: bf16 weights plus
          the fp32 master copy, momentum, and variance{plan.lost >= 1 ? ". Here failures come faster than the job can save and restart, so the first-order formula no longer applies" : ""}.
        </p>
      </LabSurface>
    </div>
  );
}
