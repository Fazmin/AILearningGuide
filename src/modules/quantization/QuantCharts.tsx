import type { QuantBlock, QuantResult } from "./quant";
import { histogram } from "./quant";

const fmt = (value: number, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "—");

/**
 * Two stacked panels on one weight axis: the trained weights as a histogram
 * with the inspected block's rounding grid drawn over it, and every weight's
 * rounding error (dequantized minus original) directly underneath.
 */
export function GridAndErrorPlot({
  source,
  result,
  block,
  blockIndex,
  outlier,
}: {
  source: Float32Array;
  result: QuantResult;
  block: QuantBlock;
  blockIndex: number;
  outlier: { value: number; label: string };
}) {
  const width = 720;
  const left = 58;
  const right = 704;
  const histTop = 26;
  const histBottom = 168;
  const errTop = 196;
  const errBottom = 292;
  const axisY = errBottom;

  let dataMin = Infinity;
  let dataMax = -Infinity;
  for (const value of source) {
    dataMin = Math.min(dataMin, value);
    dataMax = Math.max(dataMax, value);
  }
  const gridLow = block.offset + block.minCode * block.scale;
  const gridHigh = block.offset + block.maxCode * block.scale;
  const xMin = Math.min(dataMin, gridLow) - 0.25;
  const xMax = Math.max(dataMax, gridHigh) + 0.25;
  const xAt = (value: number) => left + ((value - xMin) / (xMax - xMin)) * (right - left);

  const bins = 90;
  const counts = histogram(source, xMin, xMax, bins);
  const peak = Math.max(...counts, 1);
  const binWidth = (right - left) / bins;
  const hAt = (count: number) => (Math.sqrt(count) / Math.sqrt(peak)) * (histBottom - histTop);

  const levelCount = block.maxCode - block.minCode + 1;
  const drawLevels = levelCount <= 64;
  const used = new Set<number>();
  for (let index = block.start; index < block.end; index += 1) used.add(result.codes[index]);

  const errorSpan = Math.max(result.maxError, block.scale / 2, 1e-4) * 1.12;
  const yErr = (error: number) => (errTop + errBottom) / 2 - (error / errorSpan) * ((errBottom - errTop) / 2);
  const half = block.scale / 2;

  const ticks: number[] = [];
  for (let tick = Math.ceil(xMin); tick <= Math.floor(xMax); tick += 1) ticks.push(tick);
  const countTicks = [1, 10, 50, 150].filter((tick) => tick <= peak);

  const inBlock = (index: number) => index >= block.start && index < block.end;
  const dots = [];
  for (let index = 0; index < source.length; index += 1) {
    if (inBlock(index)) continue;
    dots.push(
      <circle
        key={index}
        className="qz-dot"
        cx={xAt(source[index])}
        cy={yErr(result.weights[index] - source[index])}
        r={1.7}
      />,
    );
  }
  for (let index = block.start; index < block.end; index += 1) {
    dots.push(
      <circle
        key={index}
        className="qz-dot is-block"
        cx={xAt(source[index])}
        cy={yErr(result.weights[index] - source[index])}
        r={3}
      />,
    );
  }

  const blockName = result.blocks.length === 1 ? "the tensor's grid" : `block ${blockIndex + 1}'s grid`;
  const summary = `Histogram of ${source.length} trained weights from ${fmt(dataMin)} to ${fmt(dataMax)}, with ${blockName}: ${levelCount} levels ${fmt(block.scale, 3)} apart from ${fmt(gridLow)} to ${fmt(gridHigh)}, ${used.size} of them used. Below, each weight's rounding error; the largest is ${fmt(result.maxError, 3)} and ±half a step is ±${fmt(half, 3)}.`;

  return (
    <svg className="qz-plot" viewBox={`0 0 ${width} 330`} role="img" aria-label={summary}>
      <text className="qz-axis-label" x={left} y={14}>
        weights per bin (square-root scale) · {blockName}: {levelCount} levels
      </text>
      <rect
        className="qz-span"
        x={xAt(gridLow)}
        y={histTop}
        width={Math.max(1, xAt(gridHigh) - xAt(gridLow))}
        height={histBottom - histTop}
      />
      {counts.map((count, index) =>
        count > 0 ? (
          <rect
            key={index}
            className="qz-bar"
            x={left + index * binWidth + 0.3}
            y={histBottom - hAt(count)}
            width={Math.max(0.6, binWidth - 0.6)}
            height={hAt(count)}
          />
        ) : null,
      )}
      {countTicks.map((tick) => (
        <g key={tick}>
          <line className="qz-grid" x1={left} x2={right} y1={histBottom - hAt(tick)} y2={histBottom - hAt(tick)} />
          <text className="qz-tick" x={left - 6} y={histBottom - hAt(tick) + 3} textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      {drawLevels &&
        Array.from({ length: levelCount }, (_, offset) => {
          const code = block.minCode + offset;
          const value = block.offset + code * block.scale;
          return (
            <line
              key={code}
              className={`qz-level ${used.has(code) ? "is-used" : "is-unused"}`}
              x1={xAt(value)}
              x2={xAt(value)}
              y1={histTop}
              y2={histBottom}
            />
          );
        })}
      {!drawLevels && (
        <text className="qz-note" x={right} y={histTop + 12} textAnchor="end">
          {levelCount} levels {fmt(block.scale, 3)} apart — too dense to draw
        </text>
      )}
      <line className="plot-axis" x1={left} x2={right} y1={histBottom} y2={histBottom} />
      <g className="qz-outlier">
        <line x1={xAt(outlier.value)} x2={xAt(outlier.value)} y1={histBottom - 26} y2={histBottom - 6} />
        <text x={xAt(outlier.value) - 4} y={histBottom - 30} textAnchor="end">
          largest weight {outlier.label} = {fmt(outlier.value)}
        </text>
      </g>

      <text className="qz-axis-label" x={left} y={errTop - 8}>
        rounding error w′ − w for every weight · dashed = ± half a step of {blockName}
      </text>
      <line className="qz-grid" x1={left} x2={right} y1={yErr(0)} y2={yErr(0)} />
      <line className="qz-half" x1={left} x2={right} y1={yErr(half)} y2={yErr(half)} />
      <line className="qz-half" x1={left} x2={right} y1={yErr(-half)} y2={yErr(-half)} />
      {[errorSpan / 1.12, 0, -errorSpan / 1.12].map((value) => (
        <text key={value} className="qz-tick" x={left - 6} y={yErr(value) + 3} textAnchor="end">
          {value > 0 ? "+" : ""}
          {fmt(value, 3)}
        </text>
      ))}
      {dots}
      <line className="plot-axis" x1={left} x2={right} y1={axisY} y2={axisY} />
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="plot-axis" x1={xAt(tick)} x2={xAt(tick)} y1={axisY} y2={axisY + 4} />
          <text className="qz-tick" x={xAt(tick)} y={axisY + 16} textAnchor="middle">
            {tick}
          </text>
        </g>
      ))}
      <text className="qz-axis-label" x={(left + right) / 2} y={axisY + 32} textAnchor="middle">
        weight value (a logit in the 30 × 30 bigram table)
      </text>
    </svg>
  );
}

export interface FrontierSeries {
  id: string;
  name: string;
  shape: "circle" | "square" | "triangle";
  dash: "solid" | "dashed" | "dotted";
  points: { bits: number; bpw: number; cost: number }[];
}

const COST_FLOOR = 0.01;
const COST_CEIL = 1000;

/** Perplexity cost against real storage cost, on a log cost axis. */
export function FrontierPlot({
  series,
  formats,
  current,
  maxBpw,
}: {
  series: FrontierSeries[];
  formats: { name: string; bpw: number; cost: number }[];
  current: { bpw: number; cost: number };
  maxBpw: number;
}) {
  const left = 58;
  const right = 700;
  const top = 16;
  const bottom = 214;
  const xMin = 1.5;
  const xMax = maxBpw;
  const xAt = (value: number) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
  const logMin = Math.log10(COST_FLOOR);
  const logMax = Math.log10(COST_CEIL);
  const clampCost = (cost: number) => Math.min(COST_CEIL, Math.max(COST_FLOOR, cost));
  const yAt = (cost: number) => bottom - ((Math.log10(clampCost(cost)) - logMin) / (logMax - logMin)) * (bottom - top);
  const visible = (point: { bpw: number }) => point.bpw <= xMax;

  const marker = (shape: FrontierSeries["shape"], x: number, y: number, key: string, floor: boolean) => {
    const className = `qz-marker${floor ? " is-floor" : ""}`;
    if (shape === "square")
      return <rect key={key} className={className} x={x - 3.4} y={y - 3.4} width={6.8} height={6.8} />;
    if (shape === "triangle")
      return (
        <path key={key} className={className} d={`M${x},${y - 4.4} L${x + 4.2},${y + 3.2} L${x - 4.2},${y + 3.2} Z`} />
      );
    return <circle key={key} className={className} cx={x} cy={y} r={3.6} />;
  };

  /** Greedy label placement so formats that share a bit width do not print on top of each other. */
  const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
  const labels = [...formats]
    .sort((a, b) => a.bpw - b.bpw || a.cost - b.cost)
    .map((format) => {
      const x = xAt(format.bpw);
      const y = yAt(format.cost);
      const width = format.name.length * 5.6;
      const candidates = [
        { dx: 7, dy: -6, anchor: "start" as const },
        { dx: 7, dy: 12, anchor: "start" as const },
        { dx: -7, dy: -6, anchor: "end" as const },
        { dx: -7, dy: 12, anchor: "end" as const },
        { dx: 7, dy: 24, anchor: "start" as const },
      ];
      const box = (candidate: (typeof candidates)[number]) => {
        const x0 = candidate.anchor === "start" ? x + candidate.dx : x + candidate.dx - width;
        return { x0, x1: x0 + width, y0: y + candidate.dy - 9, y1: y + candidate.dy + 1 };
      };
      const free = candidates.find((candidate) => {
        const next = box(candidate);
        return placed.every((other) => next.x1 < other.x0 || next.x0 > other.x1 || next.y1 < other.y0 || next.y0 > other.y1);
      });
      const chosen = free ?? candidates[0];
      placed.push(box(chosen));
      return { ...format, x, y, lx: x + chosen.dx, ly: y + chosen.dy, anchor: chosen.anchor };
    });

  const hidden = series.reduce((total, entry) => total + entry.points.filter((point) => !visible(point)).length, 0);
  const xTicks = [];
  for (let tick = 2; tick <= xMax; tick += 2) xTicks.push(tick);

  return (
    <figure className="qz-frontier">
      <svg
        viewBox="0 0 720 262"
        role="img"
        aria-label={`Perplexity increase against bits per weight including scales. ${series
          .map(
            (entry) =>
              `${entry.name}: ${entry.points
                .filter(visible)
                .map((point) => `${point.bits}-bit at ${point.bpw.toFixed(2)} bpw costs ${point.cost.toFixed(2)}%`)
                .join(", ")}`,
          )
          .join(
            ". ",
          )}. Formats: ${formats.map((format) => `${format.name} ${format.bpw} bpw ${format.cost.toFixed(2)}%`).join(", ")}.`}
      >
        {[0.01, 0.1, 1, 10, 100, 1000].map((tick) => (
          <g key={tick}>
            <line className="qz-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="qz-tick" x={left - 6} y={yAt(tick) + 3} textAnchor="end">
              {tick === COST_FLOOR ? "≤0.01%" : `${tick}%`}
            </text>
          </g>
        ))}
        <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="plot-axis" x1={left} x2={left} y1={top} y2={bottom} />
        {xTicks.map((tick) => (
          <g key={tick}>
            <line className="plot-axis" x1={xAt(tick)} x2={xAt(tick)} y1={bottom} y2={bottom + 4} />
            <text className="qz-tick" x={xAt(tick)} y={bottom + 15} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        {series.map((entry) => {
          const shown = entry.points.filter(visible);
          return (
            <g key={entry.id} className={`qz-series is-${entry.id}`}>
              <polyline
                className={`qz-line is-${entry.dash}`}
                points={shown.map((point) => `${xAt(point.bpw)},${yAt(point.cost)}`).join(" ")}
              />
              {shown.map((point) =>
                marker(
                  entry.shape,
                  xAt(point.bpw),
                  yAt(point.cost),
                  `${entry.id}-${point.bits}`,
                  point.cost <= COST_FLOOR,
                ),
              )}
            </g>
          );
        })}
        {labels.map((format) => (
          <g key={format.name} className="qz-format-point">
            <path d={`M${format.x},${format.y - 4.6} l4.6,4.6 l-4.6,4.6 l-4.6,-4.6 Z`} />
            <text x={format.lx} y={format.ly} textAnchor={format.anchor}>
              {format.name}
            </text>
          </g>
        ))}
        <circle className="qz-current" cx={xAt(current.bpw)} cy={yAt(current.cost)} r={8} />
        <text className="qz-axis-label" x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
          bits per weight, scales included
          {hidden > 0 ? ` · ${hidden} point${hidden === 1 ? "" : "s"} beyond ${xMax} not drawn` : ""}
        </text>
        <text className="qz-axis-label" x={left} y={10}>
          perplexity increase over full precision (log scale)
        </text>
      </svg>
      <figcaption className="qz-legend">
        {series.map((entry) => (
          <span key={entry.id} className={`is-${entry.id}`}>
            <svg viewBox="0 0 26 10" aria-hidden="true">
              <line className={`qz-line is-${entry.dash}`} x1={1} x2={25} y1={5} y2={5} />
              {marker(entry.shape, 13, 5, "legend", false)}
            </svg>
            {entry.name}
          </span>
        ))}
        <span className="is-format">
          <svg viewBox="0 0 26 10" aria-hidden="true">
            <path d="M13,0.6 l4.4,4.4 l-4.4,4.4 l-4.4,-4.4 Z" />
          </svg>
          llama.cpp block
        </span>
        <span className="is-current">
          <svg viewBox="0 0 26 10" aria-hidden="true">
            <circle cx={13} cy={5} r={4} />
          </svg>
          current setting
        </span>
      </figcaption>
    </figure>
  );
}
