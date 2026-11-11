import { useLayoutEffect, useRef, useState } from "react";
import { teachingServing } from "@app/module-sdk";
import { STAGE_LABELS, type ShipRecipe, type Stage } from "./pipeline";

const { formatBytes, GIB } = teachingServing;

const signed = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
/** The same noise floor the table uses for its worse and better marks. */
const MOVED = 0.02;
/** Rough width of a string in the 10px mono used inside these drawings. */
const monoWidth = (text: string, size = 10) => text.length * size * 0.61;

/** The rendered width of an element, so each drawing is made at its real pixel size. */
function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const read = () => {
      const next = element.clientWidth;
      if (next > 0) setWidth((current) => (Math.abs(current - next) > 1 ? next : current));
    };
    read();
    const observer = new ResizeObserver(read);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Round tick values inside a range. */
function ticksFor(low: number, high: number, target = 4): number[] {
  const span = high - low;
  if (!(span > 0) || !Number.isFinite(span)) return [low];
  const rough = span / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / magnitude;
  const step = (fraction >= 7.5 ? 10 : fraction >= 3.5 ? 5 : fraction >= 1.5 ? 2 : 1) * magnitude;
  const ticks: number[] = [];
  for (let value = Math.ceil(low / step - 1e-9) * step; value <= high + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : +value.toPrecision(12));
  }
  return ticks;
}

const tickText = (value: number) => (value === 0 ? "0" : `${value > 0 ? "+" : "−"}${+Math.abs(value).toFixed(3)}`);

/**
 * The pipeline as a picture: each stage is an arrow, and the two held-out losses are read where the arrow ends.
 * Both lines are drawn as the change from the base model, so they share one scale and start together at zero.
 */
export function StageTrack({ stages, recipe }: { stages: Stage[]; recipe: ShipRecipe }) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const base = stages[0].metrics;
  const series = [
    {
      id: "recipe",
      name: "recipe text",
      values: stages.map((stage) => stage.metrics.targetLoss - base.targetLoss),
      baseLoss: base.targetLoss,
    },
    {
      id: "harbor",
      name: "harbor text",
      values: stages.map((stage) => stage.metrics.controlLoss - base.controlLoss),
      baseLoss: base.controlLoss,
    },
  ];
  const operations = [
    { name: "1 · Pretrain", lines: [`${recipe.baseEpochs} epochs`, "on harbor text"] },
    {
      name: "2 · Adapt",
      lines: [`rank ${recipe.loraRank} · ${recipe.loraEpochs} epochs`, recipe.replay === 0 ? "no replay" : `replay ${recipe.replay} of 6`],
    },
    {
      name: "3 · Align",
      lines: recipe.dpoSteps === 0 ? ["DPO off", "nothing changes"] : [`DPO ${recipe.dpoSteps} steps`, `beta ${recipe.beta.toFixed(2)}`],
    },
    {
      name: "4 · Compress",
      lines: [recipe.prune > 0 ? `prune ${Math.round(recipe.prune * 100)}%` : "no pruning", `then ${recipe.format}`],
    },
  ];

  const left = 50;
  const right = Math.max(left + 240, width - 94);
  const segment = (right - left) / 4;
  const xAt = (index: number) => left + segment * (index + 1) - 8;
  const ribbonTop = 2;
  const ribbonBottom = 52;
  const top = 74;
  const bottom = 244;
  const height = 282;

  const all = series.flatMap((entry) => entry.values);
  const low = Math.min(-0.05, ...all);
  const high = Math.max(0.05, ...all);
  const pad = (high - low) * 0.2;
  const yAt = (value: number) => bottom - ((value - (low - pad)) / (high - low + 2 * pad)) * (bottom - top);
  const ticks = ticksFor(low - pad * 0.4, high + pad * 0.4, 4);

  const ends = series.map((entry) => yAt(entry.values[3]));
  // Keep the two end labels apart when the lines finish close together.
  const labelY = [...ends];
  if (Math.abs(ends[0] - ends[1]) < 28) {
    const middle = (ends[0] + ends[1]) / 2;
    const recipeAbove = ends[0] <= ends[1];
    labelY[0] = middle + (recipeAbove ? -14 : 14);
    labelY[1] = middle + (recipeAbove ? 14 : -14);
  }

  const reading = series
    .map(
      (entry) =>
        `${entry.name}: ${stages.map((stage, index) => `${STAGE_LABELS[stage.id]} ${signed(entry.values[index])}`).join(", ")}`,
    )
    .join(". ");

  return (
    <div className="ship-track" ref={ref}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Change in held-out loss from the base model after each stage, in nats per character; up is worse. ${reading}.`}
      >
        {operations.map((operation, index) => {
          const start = index === 0 ? left : xAt(index - 1) + 4;
          const end = xAt(index) - 4;
          const middle = (ribbonTop + ribbonBottom) / 2;
          const notch = index === 0 ? "" : `L${start + 9},${middle} `;
          return (
            <g key={operation.name}>
              <path
                className="ship-track__arrow"
                d={`M${start},${ribbonTop} L${end - 10},${ribbonTop} L${end},${middle} L${end - 10},${ribbonBottom} L${start},${ribbonBottom} ${notch}Z`}
              />
              <text className="ship-track__op" x={start + (index === 0 ? 8 : 16)} y={ribbonTop + 15}>
                {operation.name}
              </text>
              {operation.lines.map((line, row) => (
                <text key={line} className="ship-track__detail" x={start + (index === 0 ? 8 : 16)} y={ribbonTop + 29 + row * 12}>
                  {line}
                </text>
              ))}
            </g>
          );
        })}

        {ticks.map((tick) => (
          <g key={tick}>
            <line className={tick === 0 ? "ship-track__zero" : "ship-track__grid"} x1={left} x2={xAt(3) + 8} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="ship-track__tick" x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {tickText(tick)}
            </text>
          </g>
        ))}
        <text className="ship-track__hint" x={left + 4} y={top + 4}>
          ↑ worse than base
        </text>
        <text className="ship-track__hint" x={left + 4} y={bottom - 2}>
          ↓ better than base
        </text>

        {stages.map((stage, index) => (
          <g key={stage.id}>
            <line className="ship-track__measure" x1={xAt(index)} x2={xAt(index)} y1={ribbonBottom + 2} y2={bottom + 6} />
            <text className="ship-track__stage" x={xAt(index)} y={bottom + 20} textAnchor="middle">
              {STAGE_LABELS[stage.id]}
            </text>
            <text className="ship-track__tick" x={xAt(index)} y={bottom + 33} textAnchor="middle">
              {index === 3 ? "the gate reads this" : "measured here"}
            </text>
          </g>
        ))}

        {series.map((entry) => (
          <path
            key={entry.id}
            className={`ship-track__line is-${entry.id}`}
            d={entry.values.map((value, index) => `${index === 0 ? "M" : "L"}${xAt(index)},${yAt(value)}`).join(" ")}
          />
        ))}

        {[1, 2, 3].map((index) => {
          const moves = series.map((entry) => ({
            id: entry.id,
            delta: entry.values[index] - entry.values[index - 1],
            middle: (yAt(entry.values[index]) + yAt(entry.values[index - 1])) / 2,
          }));
          const upper = moves[0].middle <= moves[1].middle ? moves[0].id : moves[1].id;
          return moves
            .filter((move) => Math.abs(move.delta) > MOVED)
            .map((move) => (
              <text
                key={`${index}-${move.id}`}
                className={`ship-track__delta ${move.delta > 0 ? "is-worse" : "is-better"}`}
                x={(xAt(index) + xAt(index - 1)) / 2}
                y={move.middle + (move.id === upper ? -9 : 17)}
                textAnchor="middle"
              >
                {signed(move.delta)} {move.delta > 0 ? "worse" : "better"}
              </text>
            ));
        })}

        {series.map((entry) =>
          entry.values.map((value, index) =>
            entry.id === "recipe" ? (
              <circle key={`${entry.id}-${index}`} className="ship-track__mark is-recipe" cx={xAt(index)} cy={yAt(value)} r={4.5} />
            ) : (
              <rect
                key={`${entry.id}-${index}`}
                className="ship-track__mark is-harbor"
                x={xAt(index) - 4.5}
                y={yAt(value) - 4.5}
                width={9}
                height={9}
              />
            ),
          ),
        )}

        {series.map((entry, index) => (
          <g key={entry.id} className={`ship-track__end is-${entry.id}`}>
            <text x={xAt(3) + 14} y={labelY[index] - 1}>
              {entry.name}
            </text>
            <text className="ship-track__end-value" x={xAt(3) + 14} y={labelY[index] + 12}>
              {signed(entry.values[3])}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/**
 * One gate check as a number line: the shipped model's change from the base model, the pass mark the slider sets,
 * and the side of the mark that passes. Both checks share one scale.
 */
export function GateGauge({
  kind,
  change,
  mark,
  passed,
  low,
  high,
}: {
  kind: "recipe" | "harbor";
  /** Shipped loss minus base loss, in nats per character. */
  change: number;
  /** The change at which the check flips: at or below it passes. */
  mark: number;
  passed: boolean;
  low: number;
  high: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(540);
  const left = 8;
  const right = width - 8;
  const axis = 44;
  const xAt = (value: number) => left + ((value - low) / (high - low)) * (right - left);
  const ticks = ticksFor(low, high, 6);
  const valueText = `as shipped ${signed(change)}`;
  const markText = `pass mark ${signed(mark)}`;
  const clampText = (x: number, text: string, size = 10) => {
    const half = monoWidth(text, size) / 2;
    return Math.min(right - half, Math.max(left + half, x));
  };
  const markRoomLeft = xAt(mark) - left > monoWidth(markText) + 8;
  return (
    <div className="ship-gauge" ref={ref}>
      <svg
        width={width}
        height={66}
        viewBox={`0 0 ${width} 66`}
        role="img"
        aria-label={`${kind === "recipe" ? "Recipe" : "Harbor"} loss as shipped is ${signed(change)} nats from the base model. The pass mark is ${signed(mark)}; at or below it passes. This check ${passed ? "passes" : "fails"}.`}
      >
        <rect className="ship-gauge__pass" x={left} y={axis - 10} width={Math.max(0, xAt(mark) - left)} height={20} />
        <line className="ship-gauge__axis" x1={left} x2={right} y1={axis} y2={axis} />
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="ship-gauge__tick" x1={xAt(tick)} x2={xAt(tick)} y1={axis - 3} y2={axis + 3} />
            <text className="ship-gauge__tick-label" x={clampText(xAt(tick), tick === 0 ? "base model" : tickText(tick), 9.5)} y={axis + 19} textAnchor="middle">
              {tick === 0 ? "base model" : tickText(tick)}
            </text>
          </g>
        ))}
        <line className="ship-gauge__base" x1={xAt(0)} x2={xAt(0)} y1={axis - 12} y2={axis + 6} />
        <line className={`ship-gauge__move ${passed ? "is-pass" : "is-fail"}`} x1={xAt(0)} x2={xAt(change)} y1={axis} y2={axis} />
        <line className="ship-gauge__mark" x1={xAt(mark)} x2={xAt(mark)} y1={axis - 22} y2={axis + 10} />
        <text
          className="ship-gauge__mark-label"
          x={xAt(mark) + (markRoomLeft ? -5 : 5)}
          y={axis - 14}
          textAnchor={markRoomLeft ? "end" : "start"}
        >
          {markText} · passes to the left
        </text>
        <circle className={`ship-gauge__dot ${passed ? "is-pass" : "is-fail"}`} cx={xAt(change)} cy={axis} r={6} />
        <text className={`ship-gauge__value ${passed ? "is-pass" : "is-fail"}`} x={clampText(xAt(change), valueText, 11)} y={10} textAnchor="middle">
          {valueText}
        </text>
        <line className="ship-gauge__leader" x1={xAt(change)} x2={xAt(change)} y1={14} y2={axis - 7} />
      </svg>
    </div>
  );
}

/**
 * The six-item quiz as ranges: each count's 95% interval before and after, with the stretch the two share shaded.
 * Overlapping ranges are why the gate does not read the quiz.
 */
export function QuizRanges({ base, shipped }: { base: Stage; shipped: Stage }) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const groups = [
    { id: "recipe", name: "recipe items", before: base.metrics.target, after: shipped.metrics.target },
    { id: "harbor", name: "harbor items", before: base.metrics.control, after: shipped.metrics.control },
  ];
  const left = 138;
  const right = width - 14;
  const xAt = (share: number) => left + share * (right - left);
  const rowHeight = 20;
  const head = 16;
  const groupHeight = head + rowHeight * 2 + 8;
  const top = 2;
  const axis = top + groups.length * groupHeight;
  return (
    <div className="ship-quiz" ref={ref}>
      <svg
        width={width}
        height={axis + 30}
        viewBox={`0 0 ${width} ${axis + 30}`}
        role="img"
        aria-label={`Benchmark counts with 95% ranges. ${groups
          .map((group) => {
            const overlap = group.before.low <= group.after.high && group.after.low <= group.before.high;
            return `${group.name}: base model ${group.before.correct} of ${group.before.total}, range ${Math.round(group.before.low * 100)}% to ${Math.round(group.before.high * 100)}%; as shipped ${group.after.correct} of ${group.after.total}, range ${Math.round(group.after.low * 100)}% to ${Math.round(group.after.high * 100)}%; the ranges ${overlap ? "overlap" : "are separate"}`;
          })
          .join(". ")}.`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className="ship-drift__grid" x1={xAt(tick)} x2={xAt(tick)} y1={top} y2={axis - 6} />
            <text x={xAt(tick)} y={axis + 8} textAnchor="middle">
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        <text x={right} y={axis + 22} textAnchor="end">
          share of six items answered correctly · dot is the count, bar is its 95% range
        </text>
        {groups.map((group, index) => {
          const y = top + index * groupHeight;
          const low = Math.max(group.before.low, group.after.low);
          const high = Math.min(group.before.high, group.after.high);
          const overlap = low <= high;
          const rows = [
            { id: "before", label: `base model · ${group.before.correct} of ${group.before.total}`, rate: group.before },
            { id: "after", label: `as shipped · ${group.after.correct} of ${group.after.total}`, rate: group.after },
          ];
          return (
            <g key={group.id}>
              {overlap && (
                <rect className="ship-quiz__overlap" x={xAt(low)} y={y + head - 1} width={Math.max(1, xAt(high) - xAt(low))} height={rowHeight * 2 + 2} />
              )}
              <text className="ship-quiz__group" x={0} y={y + 11}>
                {group.name}
              </text>
              {rows.map((row, line) => {
                const rowY = y + head + line * rowHeight + rowHeight / 2;
                return (
                  <g key={row.id}>
                    <text className="ship-quiz__row" x={left - 10} y={rowY + 3.5} textAnchor="end">
                      {row.label}
                    </text>
                    <path
                      className={`ship-quiz__range is-${row.id}`}
                      d={`M${xAt(row.rate.low)},${rowY - 4} v8 M${xAt(row.rate.low)},${rowY} H${xAt(row.rate.high)} M${xAt(row.rate.high)},${rowY - 4} v8`}
                    />
                    <circle className={`ship-quiz__dot is-${row.id}`} cx={xAt(row.rate.correct / row.rate.total)} cy={rowY} r={4.5} />
                  </g>
                );
              })}
              <text
                className={`ship-quiz__verdict ${overlap ? "" : "is-separate"}`.trim()}
                x={right}
                y={y + 11}
                textAnchor="end"
              >
                {overlap ? "shaded: the two ranges overlap, so not decisive" : "the two ranges are separate"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** What pruning and quantizing did: each held-out loss before compression (ring) and as shipped (dot), on one scale. */
export function CompressDrift({ aligned, shipped }: { aligned: Stage; shipped: Stage }) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const rows = [
    { id: "recipe", name: "recipe text", from: aligned.metrics.targetLoss, to: shipped.metrics.targetLoss },
    { id: "harbor", name: "harbor text", from: aligned.metrics.controlLoss, to: shipped.metrics.controlLoss },
  ];
  const values = rows.flatMap((row) => [row.from, row.to]);
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (high - low < 0.4) {
    const middle = (high + low) / 2;
    low = middle - 0.2;
    high = middle + 0.2;
  }
  const pad = (high - low) * 0.12;
  low -= pad;
  high += pad;
  const left = 86;
  const right = width - 12;
  const xAt = (value: number) => left + ((value - low) / (high - low)) * (right - left);
  const rowHeight = 40;
  const top = 6;
  const axis = top + rows.length * rowHeight + 2;
  const ticks = ticksFor(low, high, 5);
  return (
    <div className="ship-drift" ref={ref}>
      <svg
        width={width}
        height={axis + 30}
        viewBox={`0 0 ${width} ${axis + 30}`}
        role="img"
        aria-label={`Held-out loss before and after compressing. ${rows
          .map((row) => `${row.name}: ${row.from.toFixed(3)} after aligning, ${row.to.toFixed(3)} as shipped, ${signed(row.to - row.from)}`)
          .join(". ")}.`}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="ship-drift__grid" x1={xAt(tick)} x2={xAt(tick)} y1={top} y2={axis} />
            <text className="ship-drift__tick" x={xAt(tick)} y={axis + 13} textAnchor="middle">
              {+tick.toFixed(3)}
            </text>
          </g>
        ))}
        <text className="ship-drift__tick" x={right} y={axis + 26} textAnchor="end">
          held-out loss, nats per character · further right is worse
        </text>
        {rows.map((row, index) => {
          const y = top + index * rowHeight + rowHeight / 2 + 8;
          const delta = row.to - row.from;
          const moved = Math.abs(delta) > MOVED;
          const gap = xAt(row.to) - xAt(row.from);
          const text = `${row.to.toFixed(3)} as shipped · ${signed(delta)}${moved ? (delta > 0 ? " worse" : " better") : ""}`;
          const half = monoWidth(text) / 2 + 2;
          const labelX = Math.min(right - half, Math.max(left + half, xAt(row.to)));
          return (
            <g key={row.id}>
              <text className="ship-drift__name" x={left - 12} y={y + 3.5} textAnchor="end">
                {row.name}
              </text>
              <line className="ship-drift__lane" x1={left} x2={right} y1={y} y2={y} />
              {Math.abs(gap) > 14 && (
                <path
                  className={`ship-drift__move ${delta > 0 ? "is-worse" : "is-better"}`}
                  d={`M${xAt(row.from) + Math.sign(gap) * 6},${y} L${xAt(row.to) - Math.sign(gap) * 8},${y} m${-Math.sign(gap) * 6},-4 l${Math.sign(gap) * 6},4 l${-Math.sign(gap) * 6},4`}
                />
              )}
              <circle className="ship-drift__from" cx={xAt(row.from)} cy={y} r={6} />
              <circle className={`ship-drift__to is-${row.id}`} cx={xAt(row.to)} cy={y} r={4.5} />
              <text
                className={`ship-drift__value ${moved ? (delta > 0 ? "is-worse" : "is-better") : ""}`}
                x={labelX}
                y={y - 11}
                textAnchor="middle"
              >
                {text}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="ship-drift__key" aria-hidden="true">
        <span className="is-from">ring: after aligning</span>
        <span className="is-to">dot: as shipped</span>
      </p>
    </div>
  );
}

/** The imagined deployment's memory as one bar: fixed costs first, one slice per stream, then the room left before the limit. */
export function BudgetBar({
  serving,
  users,
  limitBytes,
  overheadBytes,
}: {
  serving: teachingServing.ServingPoint;
  users: number;
  limitBytes: number;
  overheadBytes: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const left = 6;
  const right = width - 6;
  const scale = Math.max(limitBytes, serving.totalBytes) * 1.03;
  const xAt = (bytes: number) => left + (bytes / scale) * (right - left);
  const top = 30;
  const barHeight = 30;
  const fixed = serving.weightBytes + overheadBytes;
  const segments = [
    { id: "overhead", text: `runtime ${formatBytes(overheadBytes)}`, start: 0, bytes: overheadBytes },
    { id: "weights", text: `weights ${formatBytes(serving.weightBytes)}`, start: overheadBytes, bytes: serving.weightBytes },
    {
      id: "kv",
      text: `KV cache ${formatBytes(serving.kvBytes)} · ${users} stream${users === 1 ? "" : "s"}`,
      start: fixed,
      bytes: serving.kvBytes,
    },
  ];
  // Labels sit above the bar at their segment's start, pushed right when the one before needs the room.
  let cursor = left;
  const labels = segments.map((segment) => {
    const x = Math.max(xAt(segment.start), cursor);
    cursor = x + monoWidth(segment.text) + 12;
    return { ...segment, x };
  });
  const slot = xAt(serving.kvBytesPerStream) - xAt(0);
  const spare = Math.max(0, serving.maxStreams - users);
  const streamEdges = slot >= 3 && users > 1 ? Array.from({ length: users - 1 }, (_, index) => fixed + serving.kvBytesPerStream * (index + 1)) : [];
  const freeEdges =
    slot >= 4 && serving.fits ? Array.from({ length: spare }, (_, index) => fixed + serving.kvBytesPerStream * (users + index + 1)) : [];
  const freeText = `room for ${spare} more`;
  const freeWidth = xAt(limitBytes) - xAt(serving.totalBytes);
  const step = [4, 8, 16, 32, 64].find((value) => scale / GIB / value <= 8) ?? 128;
  const ticks: number[] = [];
  for (let gib = 0; gib * GIB <= scale; gib += step) ticks.push(gib);
  const limitText = `${formatBytes(limitBytes)} limit`;
  const limitX = Math.max(left + monoWidth(limitText) / 2, Math.min(right - monoWidth(limitText) / 2, xAt(limitBytes)));
  return (
    <div className="ship-budget" ref={ref}>
      <svg
        width={width}
        height={88}
        viewBox={`0 0 ${width} 88`}
        role="img"
        aria-label={`Memory for the imagined 8B-class deployment: weights ${formatBytes(serving.weightBytes)}, runtime ${formatBytes(overheadBytes)}, KV cache ${formatBytes(serving.kvBytes)} for ${users} streams; ${formatBytes(serving.totalBytes)} against a ${formatBytes(limitBytes)} limit. ${serving.maxStreams} streams fit.`}
      >
        {labels.map((label) => (
          <g key={label.id}>
            <path className="ship-budget__leader" d={`M${xAt(label.start) + 1},${top - 1} V${top - 8} H${label.x + 1}`} />
            <text className={`ship-budget__label is-${label.id}`} x={label.x} y={top - 13}>
              {label.text}
            </text>
          </g>
        ))}
        <rect className="ship-budget__room" x={left} y={top} width={xAt(limitBytes) - left} height={barHeight} />
        {segments.map((segment) => (
          <rect
            key={segment.id}
            className={`ship-budget__seg is-${segment.id}`}
            x={xAt(segment.start)}
            y={top}
            width={Math.max(0, xAt(segment.start + segment.bytes) - xAt(segment.start))}
            height={barHeight}
          />
        ))}
        {streamEdges.map((bytes) => (
          <line key={bytes} className="ship-budget__edge" x1={xAt(bytes)} x2={xAt(bytes)} y1={top + 3} y2={top + barHeight - 3} />
        ))}
        {freeEdges.map((bytes) => (
          <line key={bytes} className="ship-budget__slot" x1={xAt(bytes)} x2={xAt(bytes)} y1={top + 3} y2={top + barHeight - 3} />
        ))}
        {serving.fits && spare > 0 && freeWidth > monoWidth(freeText) + 16 && (
          <text className="ship-budget__free" x={xAt(serving.totalBytes) + freeWidth / 2} y={top + barHeight / 2 + 3.5} textAnchor="middle">
            {freeText}
          </text>
        )}
        {!serving.fits && (
          <rect
            className="ship-budget__over"
            x={xAt(limitBytes)}
            y={top - 4}
            width={xAt(serving.totalBytes) - xAt(limitBytes)}
            height={barHeight + 8}
          />
        )}
        {ticks
          .filter((gib) => Math.abs(xAt(gib * GIB) - limitX) > monoWidth(limitText) / 2 + 32)
          .map((gib) => (
            <text key={gib} className="ship-budget__tick" x={xAt(gib * GIB)} y={top + barHeight + 16} textAnchor="middle">
              {gib === 0 ? "0 GiB" : gib}
            </text>
          ))}
        <line className="ship-budget__limit" x1={xAt(limitBytes)} x2={xAt(limitBytes)} y1={top - 7} y2={top + barHeight + 7} />
        <text
          className="ship-budget__limit-label"
          x={limitX}
          y={top + barHeight + 16}
          textAnchor="middle"
        >
          {limitText}
        </text>
      </svg>
    </div>
  );
}
