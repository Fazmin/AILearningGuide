import { useId, useMemo } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { LayoutMap } from "./LayoutMap";
import { GpuBlock } from "./GpuBlock";
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

/** Short names for the step strip, where a segment may be only a few characters wide. */
const STRIP_LABELS: Record<string, string> = {
  communication: "Traffic",
  bubble: "Bubble",
  sync: "Sync",
};

const indexOn = (ladder: ReadonlyArray<number>, value: number) => Math.max(0, ladder.indexOf(value));
const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;

const MEMORY_PARTS = [
  { id: "weights", name: "weights" },
  { id: "gradients", name: "gradients" },
  { id: "optimizer", name: "optimizer" },
  { id: "activations", name: "activations" },
] as const;

const placeParts = (memory: MemoryPerGpu) => {
  let start = 0;
  return MEMORY_PARTS.map((part) => {
    const at = start;
    start += memory[part.id];
    return { ...part, bytes: memory[part.id], start: at };
  });
};

/**
 * The stacked bar for one GPU, with the same layout at ZeRO stage 0 drawn underneath on the same axis. The axis
 * always reaches the stage-0 total, so sharding visibly shortens the bar instead of rescaling the picture.
 */
function MemoryBar({ memory, unsharded, zero }: { memory: MemoryPerGpu; unsharded: MemoryPerGpu; zero: ZeroStage }) {
  const left = 12;
  const right = 708;
  const limit = HARDWARE.memoryBytes;
  const scale = Math.max(limit * 1.12, memory.total * 1.04, unsharded.total * 1.04);
  const xAt = (bytes: number) => left + (bytes / scale) * (right - left);
  const top = 30;
  const barHeight = 34;
  const ghostTop = top + barHeight + 5;
  const ghostHeight = 15;
  const axisY = ghostTop + ghostHeight + 4;
  const placed = placeParts(memory);
  const ghost = placeParts(unsharded);
  const ghostLabel = zero === 0 ? `ZeRO stage 0, as now: ${formatBytes(unsharded.total)}` : `at ZeRO stage 0: ${formatBytes(unsharded.total)}`;
  const ghostWidth = xAt(unsharded.total) - left;
  const ghostLabelInside = ghostWidth >= ghostLabel.length * 6.9 + 12;
  const limitOnRight = xAt(limit) > (left + right) / 2;
  const tickStep = [10e9, 20e9, 50e9, 100e9, 200e9, 500e9, 1000e9].find((step) => scale / step <= 9) ?? 2000e9;
  const ticks: number[] = [];
  for (let bytes = 0; bytes <= scale && ticks.length < 12; bytes += tickStep) ticks.push(bytes);
  return (
    <svg
      className="par-plot par-memory-bar"
      viewBox={`0 0 720 ${axisY + 16}`}
      role="img"
      aria-label={`Memory on one GPU: weights ${formatBytes(memory.weights)}, gradients ${formatBytes(memory.gradients)}, optimizer state ${formatBytes(memory.optimizer)}, activations ${formatBytes(memory.activations)}; ${formatBytes(memory.total)} in all against ${MEMORY_LABEL}. ${memory.fits ? "It fits." : "It does not fit."} ${zero === 0 ? "Nothing is sharded yet." : `The same layout at ZeRO stage 0 would hold ${formatBytes(unsharded.total)}.`}`}
    >
      {ticks.map((bytes) => (
        <g key={bytes}>
          <line className="par-grid" x1={xAt(bytes)} x2={xAt(bytes)} y1={top - 4} y2={axisY} />
          <text x={xAt(bytes)} y={axisY + 12} textAnchor="middle">
            {Math.round(bytes / 1e9)}
          </text>
        </g>
      ))}
      {placed.map((segment) => {
        const width = Math.max(0, xAt(segment.start + segment.bytes) - xAt(segment.start));
        const value = formatBytes(segment.bytes);
        const full = `${segment.name} ${value}`;
        const text = width >= full.length * 7.4 + 8 ? full : width >= value.length * 7.4 + 6 ? value : "";
        return (
          <g key={segment.id}>
            <rect className={`par-seg is-${segment.id}`} x={xAt(segment.start)} y={top} width={width} height={barHeight} />
            {text && (
              <text className="par-seg-label" x={xAt(segment.start) + width / 2} y={top + barHeight / 2 + 4} textAnchor="middle">
                {text}
              </text>
            )}
          </g>
        );
      })}
      {ghost.map((segment) => (
        <rect
          key={segment.id}
          className={`par-seg par-ghost is-${segment.id}`}
          x={xAt(segment.start)}
          y={ghostTop}
          width={Math.max(0, xAt(segment.start + segment.bytes) - xAt(segment.start))}
          height={ghostHeight}
        />
      ))}
      <text
        className="par-ghost-label"
        x={ghostLabelInside ? xAt(unsharded.total) - 6 : xAt(unsharded.total) + 6}
        y={ghostTop + ghostHeight / 2 + 4}
        textAnchor={ghostLabelInside ? "end" : "start"}
      >
        {ghostLabel}
      </text>
      {!memory.fits && (
        <rect className="par-over" x={xAt(limit)} y={top - 6} width={Math.max(0, xAt(memory.total) - xAt(limit))} height={barHeight + 12} />
      )}
      <line className="par-limit" x1={xAt(limit)} x2={xAt(limit)} y1={top - 12} y2={axisY} />
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
  /** The two terms of the same formula, drawn apart: saving falls with the interval, redone work rises with it. */
  const saving = (minutes: number) => saveSeconds / (minutes * 60);
  const redone = (minutes: number) => ((minutes * 60) / 2 + restartSeconds) / mtbfSeconds;
  const partPath = (part: (minutes: number) => number) =>
    samples.map((minutes, index) => `${index === 0 ? "M" : "L"}${xAt(minutes).toFixed(1)},${yAt(part(minutes)).toFixed(1)}`).join(" ");
  const dotX = xAt(intervalMinutes);
  const dotY = yAt(currentLost);
  // The curve falls towards Young's optimum, so the label sits on the side where the curve is lower.
  const dotLabelLeft = intervalMinutes * 60 > youngSeconds && dotX - 200 > left;
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
      <text x={left - 44} y={top - 7}>
        share of time lost
      </text>
      <path className="par-lost-part is-saving" d={partPath(saving)} />
      <path className="par-lost-part is-redone" d={partPath(redone)} />
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
      <circle className="par-lost-part-dot is-saving" cx={dotX} cy={yAt(saving(intervalMinutes))} r={3.2} />
      <circle className="par-lost-part-dot is-redone" cx={dotX} cy={yAt(redone(intervalMinutes))} r={3.2} />
      <circle className="par-lost-dot" cx={dotX} cy={dotY} r={5} />
      <text
        className="par-lost-dot-label"
        x={dotX + (dotLabelLeft ? -9 : 9)}
        y={Math.max(top + 38, dotY - 9)}
        textAnchor={dotLabelLeft ? "end" : "start"}
      >
        your setting: {currentLost >= 1 ? "100%+" : formatPercent(currentLost)} lost
      </text>
    </svg>
  );
}

/**
 * The data-parallel sync against the last backward pass on one time axis. With overlap on the sync starts with
 * that backward pass and only what sticks out past its end is waited for; with overlap off it starts at the end.
 */
function OverlapTimeline({
  backwardSeconds,
  syncSeconds,
  exposedSeconds,
  overlap,
  replicas,
}: {
  backwardSeconds: number;
  syncSeconds: number;
  exposedSeconds: number;
  overlap: boolean;
  replicas: number;
}) {
  const hatch = `par-hidden-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const left = 150;
  const right = 708;
  const total = Math.max(1e-9, backwardSeconds + syncSeconds);
  const xAt = (seconds: number) => left + (seconds / total) * (right - left);
  const rowHeight = 20;
  const backwardY = 26;
  const syncY = backwardY + rowHeight + 8;
  const none = replicas <= 1 || syncSeconds <= 0;
  const syncStart = overlap ? 0 : backwardSeconds;
  const hiddenSeconds = Math.max(0, syncSeconds - exposedSeconds);
  const exposedStart = syncStart + hiddenSeconds;
  const hiddenWidth = xAt(exposedStart) - xAt(syncStart);
  const exposedWidth = xAt(exposedStart + exposedSeconds) - xAt(exposedStart);
  const endX = xAt(backwardSeconds);
  const exposedLabel = `exposed ${formatSeconds(exposedSeconds)}`;
  const exposedInside = exposedWidth >= exposedLabel.length * 7.4 + 10;
  return (
    <svg
      className="par-plot par-overlap-plot"
      viewBox={`0 0 720 ${syncY + rowHeight + 22}`}
      role="img"
      aria-label={
        none
          ? "Data parallel is 1, so there is no gradient sync to overlap."
          : `The last backward pass takes ${formatSeconds(backwardSeconds)} and the gradient sync ${formatSeconds(syncSeconds)}. ${overlap ? `With overlap on the sync starts with the backward pass, so ${formatSeconds(hiddenSeconds)} is hidden under it and ${formatSeconds(exposedSeconds)} is left exposed.` : `With overlap off the sync starts when the backward pass ends, so all ${formatSeconds(exposedSeconds)} is exposed.`}`
      }
    >
      <defs>
        <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect className="par-overlap__hidden-fill" width="6" height="6" />
          <line className="par-overlap__hidden-line" x1="0" y1="0" x2="0" y2="6" />
        </pattern>
      </defs>
      <text x={left - 10} y={backwardY + 14} textAnchor="end">
        last backward pass
      </text>
      <text x={left - 10} y={syncY + 14} textAnchor="end">
        DP gradient sync
      </text>
      <rect className="par-overlap__backward" x={left} y={backwardY} width={Math.max(1, endX - left)} height={rowHeight} rx={2} />
      {endX - left >= 60 && (
        <text className="par-seg-label" x={(left + endX) / 2} y={backwardY + 14} textAnchor="middle">
          {formatSeconds(backwardSeconds)}
        </text>
      )}
      {none ? (
        <text x={left + 4} y={syncY + 14}>
          no sync: one replica has nobody to average with
        </text>
      ) : (
        <>
          {hiddenSeconds > 0 && (
            <rect className="par-overlap__hidden" x={xAt(syncStart)} y={syncY} width={Math.max(1, hiddenWidth)} height={rowHeight} rx={2} fill={`url(#${hatch})`} />
          )}
          {hiddenSeconds > 0 && hiddenWidth >= 110 && (
            <text className="par-seg-label" x={xAt(syncStart) + hiddenWidth / 2} y={syncY + 14} textAnchor="middle">
              hidden {formatSeconds(hiddenSeconds)}
            </text>
          )}
          {exposedSeconds > 0 && (
            <rect className="par-overlap__exposed" x={xAt(exposedStart)} y={syncY} width={Math.max(1.5, exposedWidth)} height={rowHeight} rx={2} />
          )}
          <text
            className={exposedSeconds > 0 ? "par-overlap__exposed-label" : undefined}
            x={exposedInside ? xAt(exposedStart) + exposedWidth / 2 : Math.min(xAt(exposedStart + exposedSeconds) + 6, right)}
            y={exposedInside ? syncY + 14 : syncY + rowHeight + 14}
            textAnchor={exposedInside ? "middle" : xAt(exposedStart + exposedSeconds) + 6 + exposedLabel.length * 7.4 > right ? "end" : "start"}
          >
            {exposedSeconds > 0 ? exposedLabel : "all of it hidden: nothing to wait for"}
          </text>
        </>
      )}
      <line className="par-overlap__end" x1={endX} x2={endX} y1={backwardY - 8} y2={syncY + rowHeight + 4} />
      <text x={endX > (left + right) / 2 ? endX - 5 : endX + 5} y={backwardY - 10} textAnchor={endX > (left + right) / 2 ? "end" : "start"}>
        backward pass ends: the step can finish here
      </text>
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
  /** The same layout with nothing sharded: the baseline the Memory per GPU bar is compared against. */
  const unsharded = useMemo(() => memoryPerGpu({ ...layout, zero: 0 }), [layout]);
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
        <div className="stage3d-switch par-picture">
          <SegmentedControl
            label="Picture"
            value={config.layoutView}
            options={[
              { value: "flat", label: "Slab and rack" },
              { value: "block", label: "3D block of GPUs" },
            ]}
            onChange={(value) => {
              setState({ layoutView: value });
              narrate(
                value === "block"
                  ? "3D block: every GPU is one cube. Pipeline stages run across, tensor slices up, data replicas back."
                  : "Slab and rack: the model cut into pieces, and the first two nodes.",
              );
            }}
          />
        </div>
        {config.layoutView === "block" ? (
          <GpuBlock tensor={config.tensor} pipeline={config.pipeline} data={config.data} gpus={gpus} />
        ) : (
          <LayoutMap
            model={model}
            tensor={config.tensor}
            pipeline={config.pipeline}
            data={config.data}
            expert={expert}
            moe={moe}
            sliceParameters={memory.sliceParameters}
            gpus={gpus}
          />
        )}
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
          <MemoryBar memory={memory} unsharded={unsharded} zero={config.zero} />
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
        <OverlapTimeline
          backwardSeconds={(2 / 3) * timing.computePerMicrobatch}
          syncSeconds={timing.dataSync.seconds}
          exposedSeconds={timing.exposedDataSync}
          overlap={config.overlap}
          replicas={config.data}
        />
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
                {part.value / timing.step >= 0.09 && <span>{STRIP_LABELS[part.id] ?? part.label}</span>}
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
        <div
          className="par-mfu"
          role="img"
          aria-label={`Of the GPUs' peak arithmetic, ${formatPercent(timing.mfu)} is model work (MFU)${timing.hfu > timing.mfu ? `, ${formatPercent(timing.hfu - timing.mfu)} is recomputation (HFU ${formatPercent(timing.hfu)})` : ""}, ${formatPercent(Math.max(0, HARDWARE.efficiency - timing.hfu))} is lost to traffic, bubble, and exposed sync, and ${formatPercent(1 - HARDWARE.efficiency)} is out of reach above the kernel efficiency of ${formatPercent(HARDWARE.efficiency)}.`}
        >
          <span className="par-mfu__title">Share of the GPUs' peak arithmetic</span>
          <div className="par-mfu__bar">
            <i className="is-mfu" style={{ width: `${timing.mfu * 100}%` }}>
              {timing.mfu >= 0.16 && <span>MFU {formatPercent(timing.mfu)}</span>}
            </i>
            {timing.hfu - timing.mfu > 0.0005 && (
              <i className="is-recompute" style={{ width: `${(timing.hfu - timing.mfu) * 100}%` }}>
                {timing.hfu - timing.mfu >= 0.13 && <span>recompute</span>}
              </i>
            )}
            <i className="is-waiting" style={{ width: `${Math.max(0, HARDWARE.efficiency - timing.hfu) * 100}%` }}>
              {HARDWARE.efficiency - timing.hfu >= 0.22 ? (
                <span>traffic, bubble, sync</span>
              ) : (
                HARDWARE.efficiency - timing.hfu >= 0.09 && <span>waiting</span>
              )}
            </i>
            <i className="is-kernel" style={{ width: `${(1 - HARDWARE.efficiency) * 100}%` }}>
              <span>out of reach</span>
            </i>
            <b className="par-mfu__cap" style={{ left: `${HARDWARE.efficiency * 100}%` }}>
              <span>cap {Math.round(HARDWARE.efficiency * 100)}% · kernel efficiency</span>
            </b>
          </div>
          <div className="par-mfu__axis" aria-hidden="true">
            <span>0%</span>
            <span>25%</span>
            <span>50%</span>
            <span>75%</span>
            <span>100% of peak</span>
          </div>
        </div>
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
        <div className="par-lost-legend">
          <span className="is-total">
            <i /> time lost <b>{plan.lost >= 1 ? "100%+" : formatPercent(lostCapped)}</b>
          </span>
          <span>=</span>
          <span className="is-saving">
            <i /> saving (save time ÷ interval) <b>{formatPercent(Math.min(1, config.saveSeconds / (config.checkpointMinutes * 60)))}</b>
          </span>
          <span>+</span>
          <span className="is-redone">
            <i /> redone work and restarts <b>{formatPercent(Math.min(1, ((config.checkpointMinutes * 60) / 2 + config.restartMinutes * 60) / plan.mtbfSeconds))}</b>
          </span>
        </div>
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
