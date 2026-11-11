import { useEffect, useId, useMemo, useRef, useState } from "react";
import { formatPercent, formatSeconds, oneForwardOneBackward } from "./parallel";

const LEFT = 58;
const TOP = 14;
const AXIS_GAP = 26;

/** Track the rendered width so the SVG is drawn at 1:1 and its labels keep their pixel size. */
export function useMeasuredWidth(fallback: number) {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Round seconds per tick: 1, 2, or 5 times a power of ten, about five ticks across the axis. */
function tickStep(total: number) {
  const raw = total / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const scaled = raw / power;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
}

/**
 * One training step on one pipeline under the 1F1B schedule: a row per stage, a block per forward (one unit) and
 * backward (two units) of each microbatch, hatched idle time for the bubble, and the data-parallel sync that is
 * left exposed after the last backward pass. Time runs left to right in seconds.
 */
export function PipelineSchedule({
  stages,
  microbatches,
  microbatchSeconds,
  exposedSyncSeconds,
  bubbleFraction,
}: {
  stages: number;
  microbatches: number;
  /** Compute plus tensor, pipeline, and expert communication for one microbatch on one stage. */
  microbatchSeconds: number;
  exposedSyncSeconds: number;
  bubbleFraction: number;
}) {
  const [figureRef, measured] = useMeasuredWidth(640);
  const hatch = `par-idle-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const ops = useMemo(() => oneForwardOneBackward(stages, microbatches), [stages, microbatches]);
  const viewWidth = Math.max(320, measured || 640);
  const right = viewWidth - 12;
  const rowHeight = Math.max(11, Math.min(26, 210 / stages));
  const plotBottom = TOP + rowHeight * stages;
  const height = plotBottom + AXIS_GAP + 4;

  const unitSeconds = Math.max(0, microbatchSeconds) / 3;
  const pipelineSeconds = 3 * (microbatches + stages - 1) * unitSeconds;
  const exposed = Math.max(0, exposedSyncSeconds);
  const totalSeconds = Math.max(1e-9, pipelineSeconds + exposed);
  const xAt = (seconds: number) => LEFT + (seconds / totalSeconds) * (right - LEFT);
  const step = tickStep(totalSeconds);
  const ticks: number[] = [];
  for (let value = 0; value <= totalSeconds * 1.0001 && ticks.length < 12; value += step) ticks.push(value);
  const busyPerStage = 3 * microbatches;
  const idlePerStage = 3 * (stages - 1);

  return (
    <figure className="par-schedule" ref={figureRef}>
      <svg
        viewBox={`0 0 ${viewWidth} ${height}`}
        role="img"
        aria-label={`One step on one pipeline under the one-forward-one-backward schedule: ${stages} stage${stages === 1 ? "" : "s"} and ${microbatches} microbatch${microbatches === 1 ? "" : "es"}. Each stage works for ${busyPerStage} of ${busyPerStage + idlePerStage} time units and waits for ${idlePerStage}, so the bubble is ${formatPercent(bubbleFraction)} of the ${formatSeconds(pipelineSeconds)} pipeline. ${exposed > 0 ? `The data-parallel sync then adds ${formatSeconds(exposed)} with every stage waiting.` : "The data-parallel sync is hidden under the last backward pass."}`}
      >
        <defs>
          <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect className="par-schedule__idle-fill" width="6" height="6" />
            <line className="par-schedule__idle-line" x1="0" y1="0" x2="0" y2="6" />
          </pattern>
        </defs>
        {Array.from({ length: stages }, (_, stage) => {
          const y = TOP + stage * rowHeight;
          return (
            <g key={stage}>
              <rect
                className="par-schedule__row"
                x={xAt(0)}
                y={y + 1}
                width={Math.max(0, xAt(pipelineSeconds) - xAt(0))}
                height={rowHeight - 2}
                fill={`url(#${hatch})`}
              />
              {(stages <= 8 || stage % 2 === 0 || stage === stages - 1) && (
                <text className="par-schedule__stage" x={LEFT - 6} y={y + rowHeight / 2 + 3.5} textAnchor="end">
                  stage {stage + 1}
                </text>
              )}
            </g>
          );
        })}
        {ops.map((op) => {
          const x = xAt(op.start * unitSeconds);
          const width = Math.max(0.6, xAt(op.end * unitSeconds) - x - 0.6);
          const y = TOP + op.stage * rowHeight + 1;
          const showLabel = width >= 15 && rowHeight >= 13;
          return (
            <g key={`${op.kind}-${op.stage}-${op.microbatch}`} className={`par-schedule__op par-schedule__op--${op.kind}`}>
              <rect x={x} y={y} width={width} height={rowHeight - 2} rx={1.5} />
              {showLabel && (
                <text x={x + width / 2} y={y + rowHeight / 2 + 2.5} textAnchor="middle">
                  {op.kind === "forward" ? "F" : "B"}
                  {op.microbatch + 1}
                </text>
              )}
            </g>
          );
        })}
        {exposed > 0 && (
          <g className="par-schedule__sync">
            <rect
              x={xAt(pipelineSeconds) + 0.6}
              y={TOP + 1}
              width={Math.max(0.8, xAt(totalSeconds) - xAt(pipelineSeconds) - 0.6)}
              height={plotBottom - TOP - 2}
              rx={1.5}
            />
            {xAt(totalSeconds) - xAt(pipelineSeconds) >= 34 && (
              <text
                x={(xAt(pipelineSeconds) + xAt(totalSeconds)) / 2}
                y={(TOP + plotBottom) / 2 + 3}
                textAnchor="middle"
              >
                DP sync
              </text>
            )}
          </g>
        )}
        <line className="par-schedule__axis" x1={LEFT} x2={right} y1={plotBottom + 2} y2={plotBottom + 2} />
        {ticks.map((value) => (
          <g key={value}>
            <line className="par-schedule__tick-mark" x1={xAt(value)} x2={xAt(value)} y1={plotBottom + 2} y2={plotBottom + 6} />
            <text className="par-schedule__tick" x={xAt(value)} y={plotBottom + 17} textAnchor="middle">
              {formatSeconds(value)}
            </text>
          </g>
        ))}
      </svg>
      <figcaption className="par-schedule__legend">
        <span>
          <i className="par-key par-key--forward" /> F: forward pass of a microbatch (one unit)
        </span>
        <span>
          <i className="par-key par-key--backward" /> B: backward pass (two units)
        </span>
        <span>
          <i className="par-key par-key--idle" /> Hatched: waiting, the pipeline bubble
        </span>
        <span>
          <i className="par-key par-key--sync" /> DP sync left exposed after the last backward
        </span>
      </figcaption>
    </figure>
  );
}
