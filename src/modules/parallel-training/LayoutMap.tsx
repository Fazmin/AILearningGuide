import { useMeasuredWidth } from "./PipelineSchedule";
import { formatCount, HARDWARE, linkFor, type ModelShape } from "./parallel";

const NODE = HARDWARE.gpusPerNode;
/** GPUs drawn in the rack: the first two nodes. */
const SHOWN = NODE * 2;
const count = (value: number) => value.toLocaleString("en-US");

/** Layers on stage k of p when the stack is cut as evenly as it can be. */
function stageLayerRange(layers: number, stages: number, stage: number) {
  const from = Math.round((layers * stage) / stages) + 1;
  const to = Math.round((layers * (stage + 1)) / stages);
  return from === to ? `layer ${from}` : `layers ${from}–${to}`;
}

/**
 * The model drawn as one slab: pipeline cuts it into columns (stages of layers), tensor parallel cuts every column
 * into rows (slices of each layer's matrices), and data parallel stacks whole copies behind it. One cell is what one
 * GPU holds, so the cell shrinks as the cuts multiply and the stack deepens as replicas are added.
 */
function ModelSlab({
  model,
  tensor,
  pipeline,
  data,
  expert,
  sliceParameters,
  width,
}: {
  model: ModelShape;
  tensor: number;
  pipeline: number;
  data: number;
  /** The expert-parallel degree in use; above 1 a GPU holds only some of each layer's experts. */
  expert: number;
  sliceParameters: number;
  width: number;
}) {
  const wide = width >= 560;
  const x0 = wide ? 108 : 78;
  const copies = Math.min(3, data - 1);
  const x1 = wide ? width - 176 : width - 8 - copies * 7;
  const y0 = 46;
  const slabHeight = 112;
  const y1 = y0 + slabHeight;
  const columnWidth = (x1 - x0) / pipeline;
  const rowHeight = slabHeight / tensor;
  const pieces = tensor * pipeline;
  const expertShare = expert > 1 ? `${model.experts / expert} of ${model.experts} experts per layer` : "";
  const holds =
    pieces === 1
      ? expert > 1
        ? `the shared layers and ${expertShare}`
        : "the whole model"
      : `1 of ${count(pieces)} pieces${expert > 1 ? `, with ${expertShare}` : ""}`;
  const stageLabel = (stage: number) => (columnWidth >= 62 ? `stage ${stage + 1}` : `${stage + 1}`);
  const showStage = (stage: number) => columnWidth >= 20 || stage % 2 === 0;
  const height = y1 + (wide ? 40 : 104);
  const replicaLines =
    data > 1
      ? ["data parallel: the same", "pieces again, each copy", "on its own share of", "the batch"]
      : ["data parallel is off:", "one copy reads the", "whole batch"];
  return (
    <svg
      className="par-slab"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`The model as one slab cut into ${count(pieces)} piece${pieces === 1 ? "" : "s"}: ${pipeline} pipeline stage${pipeline === 1 ? "" : "s"} across and ${tensor} tensor slice${tensor === 1 ? "" : "s"} down, with ${count(data)} data-parallel ${data === 1 ? "copy" : "copies"} stacked behind. One GPU holds ${holds}, ${formatCount(sliceParameters)} parameters.`}
    >
      {Array.from({ length: copies }, (_, index) => {
        const offset = (copies - index) * 7;
        return (
          <rect
            key={index}
            className="par-slab__copy"
            x={x0 + offset}
            y={y0 - offset}
            width={x1 - x0}
            height={slabHeight}
            rx={3}
          />
        );
      })}
      <rect className="par-slab__body" x={x0} y={y0} width={x1 - x0} height={slabHeight} rx={3} />
      <rect
        className="par-slab__piece"
        x={x0 + 1}
        y={y0 + 1}
        width={Math.max(2, columnWidth - 2)}
        height={Math.max(2, rowHeight - 2)}
        rx={pieces === 1 ? 3 : 1.5}
      />
      {Array.from({ length: tensor - 1 }, (_, index) => (
        <line
          key={`t${index}`}
          className="par-slab__cut is-tensor"
          x1={x0}
          x2={x1}
          y1={y0 + (index + 1) * rowHeight}
          y2={y0 + (index + 1) * rowHeight}
        />
      ))}
      {Array.from({ length: pipeline - 1 }, (_, index) => (
        <line
          key={`p${index}`}
          className="par-slab__cut is-pipeline"
          x1={x0 + (index + 1) * columnWidth}
          x2={x0 + (index + 1) * columnWidth}
          y1={y0 - 3}
          y2={y1 + 3}
        />
      ))}

      {/* What one GPU holds: named from the highlighted piece. */}
      <line className="par-slab__leader" x1={x0 + 5} x2={x0 + 5} y1={17} y2={y0 + 2} />
      <rect className="par-slab__piece" x={x0} y={5} width={10} height={10} rx={2} />
      <text className="par-slab__lead" x={x0 + 16} y={14}>
        one GPU holds {holds}: {formatCount(sliceParameters)} parameters
      </text>

      {/* Tensor parallel: the cuts down the side. */}
      <path className="par-slab__bracket is-tensor" d={`M${x0 - 8},${y0} h-5 v${slabHeight} h5`} />
      <text className="par-slab__label is-tensor" x={x0 - 18} y={y0 + slabHeight / 2 - 10} textAnchor="end">
        <tspan className="par-slab__strong">{wide ? `tensor × ${tensor}` : `TP × ${tensor}`}</tspan>
        {wide && (
          <>
            <tspan x={x0 - 18} dy={14}>
              {tensor > 1 ? `${tensor} GPUs share` : "each GPU holds"}
            </tspan>
            <tspan x={x0 - 18} dy={13}>
              {tensor > 1 ? "every layer" : "whole layers"}
            </tspan>
          </>
        )}
      </text>

      {/* Pipeline: the stages along the bottom, with what passes between neighbours. */}
      <text className="par-slab__label is-pipeline" x={x0 - 18} y={y1 + 16} textAnchor="end">
        <tspan className="par-slab__strong">{wide ? `pipeline × ${pipeline}` : `PP × ${pipeline}`}</tspan>
      </text>
      {Array.from({ length: pipeline }, (_, stage) => (
        <g key={stage}>
          {showStage(stage) && (
            <text className="par-slab__stage" x={x0 + (stage + 0.5) * columnWidth} y={y1 + 16} textAnchor="middle">
              {stageLabel(stage)}
            </text>
          )}
          {columnWidth >= 96 && (
            <text className="par-slab__stage is-soft" x={x0 + (stage + 0.5) * columnWidth} y={y1 + 30} textAnchor="middle">
              {stageLayerRange(model.layers, pipeline, stage)}
            </text>
          )}
          {stage > 0 && columnWidth >= 62 && (
            <text className="par-slab__hop" x={x0 + stage * columnWidth} y={y1 + 16} textAnchor="middle">
              ⇄
            </text>
          )}
        </g>
      ))}
      {wide && columnWidth < 96 && (
        <text className="par-slab__stage is-soft" x={x0} y={y1 + 31}>
          {pipeline > 1
            ? `stages of ${Number.isInteger(model.layers / pipeline) ? model.layers / pipeline : `${Math.floor(model.layers / pipeline)}–${Math.ceil(model.layers / pipeline)}`} layers · activations pass forward, gradients back`
            : `one stage runs all ${model.layers} layers`}
        </text>
      )}

      {/* Data parallel: the stack of copies. */}
      <text
        className="par-slab__label is-data"
        x={wide ? x1 + copies * 7 + 14 : x0}
        y={wide ? y0 + 8 : y1 + 40}
      >
        <tspan className="par-slab__strong">
          {data > 1 ? `× ${count(data)} replicas` : "× 1 replica"}
        </tspan>
        {replicaLines.map((line, index) => (
          <tspan key={line} x={wide ? x1 + copies * 7 + 14 : x0} dy={index === 0 ? 15 : 13}>
            {line}
          </tspan>
        ))}
      </text>
    </svg>
  );
}

interface Group {
  id: "tensor" | "expert" | "data" | "pipeline";
  name: string;
  size: number;
  stride: number;
  unit: string;
}

/**
 * The first two 8-GPU nodes with the groups that GPU 0 belongs to. Ranks are numbered tensor-first, so a group's
 * members sit `stride` apart: a line that stays inside one node box runs on NVLink, and one that has to leave the
 * box is limited by InfiniBand. Uses the same rule as `linkFor`, which prices the traffic.
 */
function NodeRack({ groups, gpus, width }: { groups: ReadonlyArray<Group>; gpus: number; width: number }) {
  const wide = width >= 560;
  const left = wide ? 108 : 78;
  const right = width - 6;
  const gap = wide ? 26 : 14;
  const nodeWidth = (right - left - gap) / 2;
  const pitch = (nodeWidth - 12) / NODE;
  const chipTop = 28;
  const chipHeight = 24;
  const rowHeight = 30;
  const rowsTop = chipTop + chipHeight + 10;
  const bottom = rowsTop + groups.length * rowHeight + 2;
  const nodes = Math.ceil(gpus / NODE);
  const chipX = (rank: number) => left + Math.floor(rank / NODE) * (nodeWidth + gap) + 6 + (rank % NODE) * pitch;
  const centre = (rank: number) => chipX(rank) + (pitch - 4) / 2;
  const summary = groups
    .map((group) =>
      group.size > 1
        ? `${group.name} group of ${count(group.size)}, ${count(group.stride)} apart, on ${linkFor(group.stride, group.size).label}`
        : `${group.name} off`,
    )
    .join("; ");
  return (
    <svg
      className="par-rack"
      viewBox={`0 0 ${width} ${bottom + 6}`}
      role="img"
      aria-label={`The first ${SHOWN} of ${count(gpus)} GPUs, ${NODE} to a node, with the groups GPU 0 belongs to: ${summary}. A group that fits inside one node uses NVLink; one that spans nodes uses InfiniBand.`}
    >
      {[0, 1].map((node) => {
        const x = left + node * (nodeWidth + gap);
        const used = gpus > node * NODE;
        return (
          <g key={node}>
            <rect
              className={`par-rack__node${used ? "" : " is-empty"}`}
              x={x}
              y={20}
              width={nodeWidth}
              height={bottom - 20}
              rx={6}
            />
            <text className="par-rack__node-label" x={x + 2} y={13}>
              {used
                ? `node ${node + 1}${wide ? ` · ${NODE} GPUs on NVLink` : ""}`
                : `node ${node + 1} · not used`}
            </text>
          </g>
        );
      })}
      {nodes > 2 && (
        <text className="par-rack__node-label" x={right} y={13} textAnchor="end">
          + {count(nodes - 2)} more node{nodes - 2 === 1 ? "" : "s"}
        </text>
      )}
      {Array.from({ length: SHOWN }, (_, rank) => (
        <g key={rank} className={`par-rack__chip${rank >= gpus ? " is-empty" : ""}${rank === 0 ? " is-self" : ""}`}>
          <rect x={chipX(rank)} y={chipTop} width={pitch - 4} height={chipHeight} rx={3} />
          {rank < gpus && pitch >= 16 && (
            <text x={centre(rank)} y={chipTop + chipHeight / 2 + 3.5} textAnchor="middle">
              {rank}
            </text>
          )}
        </g>
      ))}
      <text className="par-rack__label" x={left - 12} y={chipTop + chipHeight / 2 + 3.5} textAnchor="end">
        GPU rank
      </text>
      {groups.map((group, index) => {
        const y = rowsTop + index * rowHeight + 15;
        const link = linkFor(group.stride, group.size);
        const members: number[] = [];
        for (let k = 0; k < group.size && k * group.stride < SHOWN; k += 1) members.push(k * group.stride);
        const hidden = group.size - members.length;
        const last = members[members.length - 1] ?? 0;
        const endX = hidden > 0 ? right - 12 : centre(last);
        const off = group.size <= 1;
        const nextRank = members.length * group.stride;
        return (
          <g key={group.id} className={`par-rack__group is-${group.id} is-${link.id}${off ? " is-off" : ""}`}>
            <text className="par-rack__label" x={left - 12} y={y - 2} textAnchor="end">
              <tspan className="par-slab__strong">
                {group.name} × {count(group.size)}
              </tspan>
              <tspan className="par-rack__link" x={left - 12} dy={12}>
                {off ? "off" : link.label}
              </tspan>
            </text>
            {!off && <line className="par-rack__wire" x1={centre(0)} x2={endX} y1={y} y2={y} />}
            {!off && hidden > 0 && <path className="par-rack__arrow" d={`M${right - 12},${y - 4} l7,4 l-7,4 z`} />}
            {members.map((rank) => (
              <circle key={rank} className="par-rack__dot" cx={centre(rank)} cy={y} r={off ? 3 : 4} />
            ))}
            {!off && hidden > 0 && (
              <text className="par-rack__more" x={right - 16} y={y - 7} textAnchor="end">
                {members.length === 1
                  ? `next ${group.unit}: GPU ${count(nextRank)}, in node ${count(Math.floor(nextRank / NODE) + 1)}`
                  : `+ ${count(hidden)} more ${group.unit}${hidden === 1 ? "" : "s"}`}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Two pictures of the same layout: the model cut into the pieces one GPU holds, and where those GPUs sit in
 * 8-GPU nodes, which decides whether each kind of traffic rides NVLink or InfiniBand.
 */
export function LayoutMap({
  model,
  tensor,
  pipeline,
  data,
  expert,
  moe,
  sliceParameters,
  gpus,
}: {
  model: ModelShape;
  tensor: number;
  pipeline: number;
  data: number;
  /** The expert-parallel degree in use; 1 for a dense model. */
  expert: number;
  moe: boolean;
  sliceParameters: number;
  gpus: number;
}) {
  const [figureRef, measured] = useMeasuredWidth(680);
  const width = Math.max(300, measured || 680);
  const groups: Group[] = [
    { id: "tensor", name: "tensor", size: tensor, stride: 1, unit: "slice" },
    ...(moe ? [{ id: "expert" as const, name: "expert", size: expert, stride: tensor, unit: "expert GPU" }] : []),
    { id: "data", name: "data", size: data, stride: tensor, unit: "replica" },
    { id: "pipeline", name: "pipeline", size: pipeline, stride: tensor * data, unit: "stage" },
  ];
  return (
    <figure className="par-map" ref={figureRef}>
      <ModelSlab
        model={model}
        tensor={tensor}
        pipeline={pipeline}
        data={data}
        expert={expert}
        sliceParameters={sliceParameters}
        width={width}
      />
      <NodeRack groups={groups} gpus={gpus} width={width} />
      <figcaption className="par-map__legend">
        <span>
          <i className="par-map__key is-nvlink" /> stays inside a node: NVLink, {HARDWARE.nvlinkBytesPerSecond / 1e9} GB/s
        </span>
        <span>
          <i className="par-map__key is-network" /> leaves the node: InfiniBand, {HARDWARE.networkBytesPerSecond / 1e9} GB/s per GPU
        </span>
      </figcaption>
    </figure>
  );
}
