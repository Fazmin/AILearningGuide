import { useId, useLayoutEffect, useRef, useState } from "react";
import { LIMITS, type OperatingPoint } from "./operate";
import { formatBytes, GIB, HARDWARE } from "./serving";

const MAX_USERS = LIMITS.users.max;
const tokens = (value: number) => (value >= 100 ? value.toFixed(0) : value.toFixed(1));
const ms = (seconds: number) => (seconds >= 10 ? `${seconds.toFixed(1)} s` : `${(seconds * 1000).toFixed(0)} ms`);
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

/**
 * Memory as one bar: the two fixed costs first, then one slice of cache per stream, then the empty slots more
 * streams could take before the limit.
 */
export function MemoryBar({ point, users }: { point: OperatingPoint; users: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(650);
  const left = 6;
  const right = width - 6;
  const scaleBytes = Math.max(HARDWARE.vramBytes, point.totalBytes) * 1.03;
  const xAt = (bytes: number) => left + (bytes / scaleBytes) * (right - left);
  const top = 32;
  const barHeight = 34;
  const fixed = point.weightBytes + HARDWARE.overheadBytes;
  const perStream = point.kvBytesPerStream + point.stateBytesPerStream;
  const segments = [
    { id: "overhead", text: `runtime ${formatBytes(HARDWARE.overheadBytes)}`, start: 0, bytes: HARDWARE.overheadBytes },
    { id: "weights", text: `weights ${formatBytes(point.weightBytes)}`, start: HARDWARE.overheadBytes, bytes: point.weightBytes },
    {
      id: "kv",
      text: `${point.stateBytes > 0 ? "KV cache + state" : "KV cache"} ${formatBytes(point.kvBytes + point.stateBytes)} · ${users} stream${users === 1 ? "" : "s"}`,
      start: fixed,
      bytes: point.kvBytes + point.stateBytes,
    },
  ];
  // Labels sit above the bar at their segment's start, pushed right when the one before needs the room.
  let cursor = left;
  const labels = segments.map((segment) => {
    const x = Math.min(Math.max(xAt(segment.start), cursor), Math.max(cursor, right - monoWidth(segment.text)));
    cursor = x + monoWidth(segment.text) + 12;
    return { ...segment, x };
  });
  const slot = xAt(perStream) - xAt(0);
  const spare = Math.max(0, point.maxStreams - users);
  const streamEdges =
    users > 1 && users <= MAX_USERS && slot >= 2.5 ? Array.from({ length: users - 1 }, (_, index) => fixed + perStream * (index + 1)) : [];
  const freeEdges = point.fits && slot >= 4 ? Array.from({ length: spare }, (_, index) => fixed + perStream * (users + index + 1)) : [];
  const freeText = `room for ${spare} more`;
  const freeWidth = xAt(HARDWARE.vramBytes) - xAt(point.totalBytes);
  const stepGiB = [4, 8, 16, 32, 64].find((step) => scaleBytes / GIB / step <= 8) ?? 128;
  const ticks: number[] = [];
  for (let gib = 0; gib * GIB <= scaleBytes; gib += stepGiB) ticks.push(gib);
  const limitText = `${formatBytes(HARDWARE.vramBytes)} limit`;
  const limitX = Math.max(left + monoWidth(limitText) / 2, Math.min(right - monoWidth(limitText) / 2, xAt(HARDWARE.vramBytes)));
  return (
    <div className="ds-memory-wrap" ref={ref}>
      <svg
        className="ds-plot ds-memory"
        width={width}
        height={top + barHeight + 24}
        viewBox={`0 0 ${width} ${top + barHeight + 24}`}
        role="img"
        aria-label={`Memory: weights ${formatBytes(point.weightBytes)}, activations and runtime ${formatBytes(HARDWARE.overheadBytes)}, KV cache ${formatBytes(point.kvBytes)}${point.stateBytes > 0 ? ` and recurrent state ${formatBytes(point.stateBytes)}` : ""} for ${users} streams; total ${formatBytes(point.totalBytes)} against a ${formatBytes(HARDWARE.vramBytes)} limit. At this context, ${point.maxStreams} streams fit.`}
      >
        {labels.map((label) => (
          <g key={label.id}>
            <path className="ds-leader" d={`M${xAt(label.start) + 1},${top - 1} V${top - 9} H${label.x + 1}`} />
            <text className="ds-seg-label" x={label.x} y={top - 14}>
              {label.text}
            </text>
          </g>
        ))}
        <rect className="ds-room" x={left} y={top} width={xAt(HARDWARE.vramBytes) - left} height={barHeight} />
        {segments.map((segment) => (
          <rect
            key={segment.id}
            className={`ds-seg is-${segment.id}`}
            x={xAt(segment.start)}
            y={top}
            width={Math.max(0, xAt(segment.start + segment.bytes) - xAt(segment.start))}
            height={barHeight}
          />
        ))}
        {streamEdges.map((bytes) => (
          <line key={bytes} className="ds-stream-edge" x1={xAt(bytes)} x2={xAt(bytes)} y1={top + 4} y2={top + barHeight - 4} />
        ))}
        {freeEdges.map((bytes) => (
          <line key={bytes} className="ds-slot" x1={xAt(bytes)} x2={xAt(bytes)} y1={top + 4} y2={top + barHeight - 4} />
        ))}
        {point.fits && spare > 0 && freeWidth > monoWidth(freeText) + 16 && (
          <text className="ds-free-label" x={xAt(point.totalBytes) + freeWidth / 2} y={top + barHeight / 2 + 3.5} textAnchor="middle">
            {freeText}
          </text>
        )}
        {!point.fits && (
          <rect
            className="ds-over"
            x={xAt(HARDWARE.vramBytes)}
            y={top - 5}
            width={xAt(point.totalBytes) - xAt(HARDWARE.vramBytes)}
            height={barHeight + 10}
          />
        )}
        {ticks
          .filter((gib) => Math.abs(xAt(gib * GIB) - limitX) > monoWidth(limitText) / 2 + 32)
          .map((gib) => (
            <text key={gib} x={Math.max(left + 14, xAt(gib * GIB))} y={top + barHeight + 17} textAnchor="middle">
              {gib === 0 ? "0 GiB" : gib}
            </text>
          ))}
        <line className="ds-limit" x1={xAt(HARDWARE.vramBytes)} x2={xAt(HARDWARE.vramBytes)} y1={top - 8} y2={top + barHeight + 7} />
        <text className="ds-limit-label" x={limitX} y={top + barHeight + 17} textAnchor="middle">
          {limitText}
        </text>
      </svg>
    </div>
  );
}

const niceCeil = (value: number) => [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].find((candidate) => candidate >= value) ?? 200;

/** The stretch of time the decode timeline draws: a round figure that holds about eight single-stream steps. */
export const timelineSpan = (soloStep: number) => niceCeil(8 * soloStep);

/**
 * A stretch of decoding drawn to scale: each block on the GPU lane is one decode step (read the weights, then the
 * caches it serves), and each dot is a token landing in a stream. Batched steps feed every stream; one-at-a-time
 * steps feed one.
 */
export function DecodeTimeline({
  point,
  users,
  batching,
  soloStep,
  bandwidth,
}: {
  point: OperatingPoint;
  users: number;
  batching: boolean;
  /** Seconds one step takes with a single stream: it sets the window, so adding users visibly stretches the steps. */
  soloStep: number;
  bandwidth: number;
}) {
  const clipId = useId();
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const step = point.stepSeconds;
  const span = timelineSpan(soloStep);
  const left = 70;
  const right = width - 12;
  const xAt = (seconds: number) => left + (seconds / span) * (right - left);
  const cacheBytes = batching ? point.kvBytes + point.stateBytes : point.kvBytesPerStream + point.stateBytesPerStream;
  const weightTime = Math.min(step, point.weightBytes / bandwidth);
  const cacheTime = Math.min(step - weightTime, cacheBytes / bandwidth);
  const steps = Math.min(200, Math.ceil(span / step));
  const shown = Math.min(users, 5);
  const gpuTop = 36;
  const gpuHeight = 24;
  const laneTop = gpuTop + gpuHeight + 38;
  const laneGap = 20;
  const laneY = (lane: number) => laneTop + lane * laneGap;
  const axis = laneY(shown - 1) + 18;
  const height = axis + 30 + (users > shown ? 16 : 0);
  const stepPx = xAt(step) - xAt(0);
  const landings: { key: string; x: number; lanes: number[] }[] = [];
  for (let index = 1; index <= steps && index * step <= span + 1e-9; index += 1) {
    const lanes = batching ? Array.from({ length: shown }, (_, lane) => lane) : [(index - 1) % users].filter((lane) => lane < shown);
    landings.push({ key: `step-${index}`, x: xAt(index * step), lanes });
  }
  const wait = point.timePerOutputToken;
  const secondX = xAt(step + wait);
  const gapFits = step + wait <= span + 1e-9 && secondX - xAt(step) > 30;
  const gapText = `${ms(wait)} between one user’s tokens`;
  const stepText = `one decode step · ${ms(step)}`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * span);
  return (
    <div className="ds-timeline" ref={ref}>
      <svg
        className="ds-plot"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${ms(span)} of decoding. Each decode step takes ${ms(step)}: ${ms(weightTime)} to read the weights and ${ms(cacheTime)} to read ${batching ? `all ${users} streams' caches` : "one stream's cache"}. ${batching ? `Every step gives each of the ${users} streams one token` : `Each step gives one stream a token, so a stream waits for the other ${users - 1}`}. One user gets a token every ${ms(wait)}.`}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={left} y={0} width={right - left} height={height} />
          </clipPath>
        </defs>
        <text className="ds-lane-label" x={left - 8} y={gpuTop + gpuHeight / 2 + 3.5} textAnchor="end">
          GPU
        </text>
        <g clipPath={`url(#${clipId})`}>
          {Array.from({ length: steps }, (_, index) => {
            const start = index * step;
            return (
              <g key={index}>
                <rect className="ds-step is-weights" x={xAt(start)} y={gpuTop} width={Math.max(0, xAt(weightTime) - xAt(0))} height={gpuHeight} />
                <rect
                  className="ds-step is-kv"
                  x={xAt(start + weightTime)}
                  y={gpuTop}
                  width={Math.max(0, xAt(cacheTime) - xAt(0))}
                  height={gpuHeight}
                />
                <rect className="ds-step-frame" x={xAt(start)} y={gpuTop} width={stepPx} height={gpuHeight} />
              </g>
            );
          })}
          {xAt(weightTime) - xAt(0) > 50 && (
            <text className="ds-step-text" x={xAt(weightTime / 2)} y={gpuTop + gpuHeight / 2 + 3.5} textAnchor="middle">
              weights
            </text>
          )}
          {xAt(cacheTime) - xAt(0) > (batching ? 46 : 40) && (
            <text className="ds-step-text" x={xAt(weightTime + cacheTime / 2)} y={gpuTop + gpuHeight / 2 + 3.5} textAnchor="middle">
              {batching ? "caches" : "cache"}
            </text>
          )}
          {landings.map((landing) =>
            landing.lanes.length > 0 ? (
              <line
                key={landing.key}
                className="ds-landing"
                x1={landing.x}
                x2={landing.x}
                y1={gpuTop + gpuHeight}
                y2={laneY(Math.max(...landing.lanes))}
              />
            ) : null,
          )}
        </g>
        <path className="ds-bracket" d={`M${xAt(0) + 1},${gpuTop - 4} v-5 H${Math.min(right, xAt(step)) - 1} v5`} />
        <text className="ds-step-label" x={xAt(0)} y={gpuTop - 14}>
          {stepText}
          {width < 520 ? "" : " · reads the "}
          {width >= 520 && <tspan className="ds-step-word is-weights">■ weights</tspan>}
          {width < 520 ? "" : batching ? " once, then " : ", then "}
          {width >= 520 && <tspan className="ds-step-word is-kv">■ {batching ? "every stream’s cache" : "one stream’s cache"}</tspan>}
        </text>
        {Array.from({ length: shown }, (_, lane) => (
          <g key={lane}>
            <text className="ds-lane-label" x={left - 8} y={laneY(lane) + 3.5} textAnchor="end">
              stream {lane + 1}
            </text>
            <line className="ds-lane" x1={left} x2={right} y1={laneY(lane)} y2={laneY(lane)} />
          </g>
        ))}
        {landings.flatMap((landing) =>
          landing.lanes.map((lane) => <circle key={`${landing.key}-${lane}`} className="ds-token" cx={landing.x} cy={laneY(lane)} r={4} />),
        )}
        {gapFits ? (
          <g>
            <path
              className="ds-gap"
              d={`M${xAt(step) + 2},${laneY(0) - 12} H${secondX - 2} M${xAt(step) + 7},${laneY(0) - 16} l-5,4 l5,4 M${secondX - 7},${laneY(0) - 16} l5,4 l-5,4`}
            />
            <text
              className="ds-gap-label"
              x={Math.min(right - monoWidth(gapText) / 2, Math.max(left + monoWidth(gapText) / 2, (xAt(step) + secondX) / 2))}
              y={laneY(0) - 20}
              textAnchor="middle"
            >
              {gapText}
            </text>
          </g>
        ) : (
          <text className="ds-gap-label" x={Math.min(right - monoWidth(`stream 1 waits ${ms(wait)} for its next token →`), xAt(step) + 10)} y={laneY(0) - 14}>
            stream 1 waits {ms(wait)} for its next token →
          </text>
        )}
        {users > shown && (
          <text x={left} y={axis + 2}>
            + {users - shown} more stream{users - shown === 1 ? "" : "s"}, {batching ? "fed by the same steps" : "each waiting for its turn"}
          </text>
        )}
        {ticks.map((tick) => (
          <text key={tick} x={xAt(tick)} y={height - 14} textAnchor={tick === 0 ? "start" : tick === span ? "end" : "middle"}>
            {tick === 0 ? "0" : ms(tick)}
          </text>
        ))}
        <text x={(left + right) / 2} y={height - 1} textAnchor="middle">
          time
        </text>
      </svg>
    </div>
  );
}

/** Total and per-user tokens per second against concurrency, with the current load marked and read out. */
export function ThroughputPlot({
  sweep,
  users,
  maxStreams,
  ceiling,
}: {
  sweep: { users: number; total: number; perUser: number }[];
  users: number;
  maxStreams: number;
  ceiling: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(668);
  const left = 44;
  const right = width - 14;
  const panels = [
    { key: "total" as const, title: "total tokens/s across all streams", top: 22, bottom: 128 },
    { key: "perUser" as const, title: "tokens/s each user sees", top: 162, bottom: 254 },
  ];
  const xAt = (n: number) => left + ((n - 1) / (MAX_USERS - 1)) * (right - left);
  const fitEdge = maxStreams < MAX_USERS ? Math.max(1, maxStreams + 0.5) : null;
  const current = sweep[users - 1];
  const nofitText = `over ${formatBytes(HARDWARE.vramBytes)} beyond ${maxStreams}`;
  return (
    <div className="ds-throughput-wrap" ref={ref}>
      <svg
        className="ds-plot ds-throughput"
        width={width}
        height={292}
        viewBox={`0 0 ${width} 292`}
        role="img"
        aria-label={`Decode throughput against concurrent streams. At ${users}: ${tokens(current.total)} tokens per second in total, ${tokens(current.perUser)} per user. ${fitEdge ? `Beyond ${maxStreams} streams the KV cache no longer fits.` : "Every point on the chart fits in memory."} Compute ceiling ${ceiling.toFixed(0)} tokens per second.`}
      >
        {panels.map((panel) => {
          const top = Math.max(...sweep.map((point) => point[panel.key]));
          const peak = top * 1.28;
          const yAt = (value: number) => panel.bottom - (value / peak) * (panel.bottom - panel.top);
          const path = sweep.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(point.users)},${yAt(point[panel.key])}`).join(" ");
          const text = `${tokens(current[panel.key])} tok/s ${panel.key === "total" ? "in total" : "each"} at ${users}`;
          const textLeft = xAt(users) + 10 + monoWidth(text, 10.5) > right;
          // Total climbs to the right and per-user falls, so each label goes on the side its curve leaves empty.
          const above = panel.key === "total" ? textLeft : !textLeft;
          return (
            <g key={panel.key}>
              <text x={left} y={panel.top - 8}>
                {panel.title}
              </text>
              {fitEdge && (
                <rect className="ds-nofit" x={xAt(fitEdge)} y={panel.top} width={right - xAt(fitEdge)} height={panel.bottom - panel.top} />
              )}
              {ticksFor(0, top, 3).map((tick) => (
                <g key={tick}>
                  <line className="ds-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
                  <text x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
                    {tick}
                  </text>
                </g>
              ))}
              <path className={`ds-curve is-${panel.key}`} d={path} />
              <circle className={`ds-dot is-${panel.key}`} cx={xAt(users)} cy={yAt(current[panel.key])} r={5} />
              <text
                className={`ds-dot-label is-${panel.key}`}
                x={xAt(users) + (textLeft ? -9 : 9)}
                y={yAt(current[panel.key]) + (above ? -9 : 16)}
                textAnchor={textLeft ? "end" : "start"}
              >
                {text}
              </text>
            </g>
          );
        })}
        {fitEdge && (
          <text
            className="ds-nofit-label"
            x={Math.min(xAt(fitEdge) + 6, right - monoWidth(nofitText))}
            y={panels[0].bottom - 6}
          >
            {nofitText}
          </text>
        )}
        <line className="marker-guide" x1={xAt(users)} x2={xAt(users)} y1={22} y2={254} />
        {[1, 8, 16, 24, 32, 40, 48, 56, 64].map((tick) => (
          <text key={tick} x={xAt(tick)} y={270} textAnchor="middle">
            {tick}
          </text>
        ))}
        <text x={(left + right) / 2} y={287} textAnchor="middle">
          concurrent streams
        </text>
      </svg>
    </div>
  );
}

export interface LeverRow {
  id: string;
  label: string;
  point: OperatingPoint;
  cost: string;
}

/**
 * Each lever on its own and the current selection, as three aligned bars a row: streams that fit, each user's speed,
 * and total speed. A tick on every track marks the no-lever value, so what a lever moved is visible at a glance.
 */
export function LeverTable({ rows, label }: { rows: LeverRow[]; label: string }) {
  const baseline = rows[0].point;
  const columns = [
    { id: "streams", head: "streams that fit", value: (point: OperatingPoint) => point.maxStreams, text: (point: OperatingPoint) => `${point.maxStreams}`, needsFit: false },
    {
      id: "perUser",
      head: "tok/s each user",
      value: (point: OperatingPoint) => point.perUserTokensPerSecond,
      text: (point: OperatingPoint) => tokens(point.perUserTokensPerSecond),
      needsFit: true,
    },
    {
      id: "total",
      head: "tok/s in total",
      value: (point: OperatingPoint) => point.totalTokensPerSecond,
      text: (point: OperatingPoint) => tokens(point.totalTokensPerSecond),
      needsFit: true,
    },
  ];
  const peaks = columns.map((column) =>
    Math.max(1e-9, ...rows.filter((row) => !column.needsFit || row.point.fits).map((row) => column.value(row.point))),
  );
  return (
    <div className="ds-lever-table">
      <div className="ds-lever-head" aria-hidden="true">
        <span />
        {columns.map((column) => (
          <span key={column.id}>{column.head}</span>
        ))}
        <span>cost per M tokens</span>
      </div>
      <div className="ds-lever-rows" role="list" aria-label={label}>
      {rows.map((row) => (
        <div
          key={row.id}
          className={`ds-lever-row ${row.id === "selected" ? "is-selected" : ""} ${row.id === "none" ? "is-baseline" : ""}`.trim()}
          role="listitem"
        >
          <strong>{row.label}</strong>
          {columns.map((column, index) => {
            const off = column.needsFit && !row.point.fits;
            const value = off ? 0 : column.value(row.point);
            return (
              <span key={column.id} className={`ds-lever-cell is-${column.id}`}>
                <i>
                  <b style={{ width: `${Math.min(100, (value / peaks[index]) * 100)}%` }} />
                  {row.id !== "none" && (!column.needsFit || baseline.fits) && (
                    <u style={{ left: `${Math.min(100, (column.value(baseline) / peaks[index]) * 100)}%` }} />
                  )}
                </i>
                <em>{off ? "—" : column.text(row.point)}</em>
                <small className="ds-lever-unit">{column.head}</small>
              </span>
            );
          })}
          <span className="ds-lever-cost">
            {row.cost}
            <span className="sr-only"> per million tokens</span>
          </span>
        </div>
      ))}
      </div>
      <p className="ds-lever-key" aria-hidden="true">
        <u /> the tick on each track is the No levers value
      </p>
    </div>
  );
}
