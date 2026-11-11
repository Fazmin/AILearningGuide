import type { FileSlice, FloatFormat, FloatFormatStats, QuantBlock, QuantResult } from "./quant";
import { histogram } from "./quant";

const fmt = (value: number, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const signed = (value: number, digits = 2) => `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;

/**
 * Three bands, top to bottom. The snap band zooms on the inspected block: its weights on the top
 * rail, its levels on the bottom rail, and one line per weight to the level it rounds to. A funnel
 * places that grid on the histogram of all the trained weights, and every weight's rounding error
 * (dequantized minus original) sits directly underneath on the same weight axis.
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
  outlier: { index: number; value: number; label: string };
}) {
  const width = 720;
  const left = 58;
  const right = 704;
  const railTop = 42;
  const railBottom = 102;
  const codeY = railBottom + 18;
  const bracketY = railBottom + 26;
  const valueY = railBottom + 40;
  const funnelTop = railBottom + 46;
  const histTop = 190;
  const histBottom = 268;
  const errTop = 298;
  const errBottom = 372;
  const axisY = errBottom;
  const height = axisY + 38;

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
  const perCode = new Map<number, number>();
  for (let index = block.start; index < block.end; index += 1) {
    perCode.set(result.codes[index], (perCode.get(result.codes[index]) ?? 0) + 1);
  }
  const usedCount = perCode.size;
  const busiest = Math.max(1, ...perCode.values());
  const blockSize = block.end - block.start;

  // The snap band's own axis: the block's grid from its first level to its last, edge to edge.
  const inset = 16;
  const gridSpan = gridHigh - gridLow || 1;
  const bx = (value: number) =>
    left + inset + (Math.min(1, Math.max(0, (value - gridLow) / gridSpan)) * (right - left - 2 * inset));
  const gap = levelCount > 1 ? (right - left - 2 * inset) / (levelCount - 1) : right - left;
  const drawNodes = gap >= 9;
  const nodeMax = Math.min(7, gap * 0.45);
  const labelEvery = [1, 2, 4, 8, 16, 32, 64].find((step) => step * gap >= 24) ?? 64;
  const snapOpacity = Math.min(0.9, Math.max(0.3, 8 / Math.sqrt(blockSize)));
  const outlierInBlock = outlier.index >= block.start && outlier.index < block.end;
  const middle = block.minCode + Math.max(0, Math.floor(levelCount / 2) - 1);
  const levelValue = (code: number) => block.offset + code * block.scale;

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
  const snaps = [];
  const marks = [];
  for (let index = block.start; index < block.end; index += 1) {
    const from = bx(source[index]);
    dots.push(
      <circle
        key={index}
        className="qz-dot is-block"
        cx={xAt(source[index])}
        cy={yErr(result.weights[index] - source[index])}
        r={3}
      />,
    );
    snaps.push(
      <line
        key={index}
        className="qz-snap"
        x1={from}
        x2={bx(result.weights[index])}
        y1={railTop}
        y2={railBottom}
        strokeOpacity={snapOpacity}
      />,
    );
    marks.push(
      blockSize <= 64 ? (
        <circle
          key={index}
          className={`qz-weight${index === outlier.index ? " is-outlier" : ""}`}
          cx={from}
          cy={railTop}
          r={3.4}
        />
      ) : (
        <line key={index} className="qz-weight-tick" x1={from} x2={from} y1={railTop - 7} y2={railTop} />
      ),
    );
  }

  const single = result.blocks.length === 1;
  const blockName = single ? "the tensor's grid" : `block ${blockIndex + 1}'s grid`;
  const owner = single ? "all" : `block ${blockIndex + 1}'s`;
  const summary = `${owner} ${blockSize} weight${blockSize === 1 ? "" : "s"}, each joined to the nearest of ${levelCount} levels ${fmt(block.scale, 3)} apart, from ${fmt(gridLow)} to ${fmt(gridHigh)}; ${usedCount} of the levels are used. Below, a histogram of ${source.length} trained weights from ${fmt(dataMin)} to ${fmt(dataMax)} with that grid drawn over it, and each weight's rounding error; the largest is ${fmt(result.maxError, 3)} and ±half a step is ±${fmt(half, 3)}.`;

  return (
    <svg className="qz-plot qz-plot--grid" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary}>
      <text className="qz-axis-label is-lead" x={left} y={12}>
        {owner} {blockSize} weight{blockSize === 1 ? "" : "s"} (top) · each snaps to the nearest of {levelCount} levels
        (bottom)
      </text>
      {outlierInBlock && (
        <text
          className="qz-outlier-label"
          x={bx(outlier.value) > (left + right) / 2 ? Math.min(right, bx(outlier.value) + 6) : bx(outlier.value) - 6}
          y={30}
          textAnchor={bx(outlier.value) > (left + right) / 2 ? "end" : "start"}
        >
          largest weight {outlier.label} = {fmt(outlier.value)} ↓
        </text>
      )}
      {drawNodes &&
        Array.from({ length: levelCount }, (_, offset) => {
          const centre = left + inset + offset * gap;
          const x0 = Math.max(left, centre - gap / 2);
          const x1 = Math.min(right, centre + gap / 2);
          return (
            <rect
              key={offset}
              className={`qz-catch ${offset % 2 === 0 ? "is-even" : "is-odd"}`}
              x={x0}
              y={railTop}
              width={x1 - x0}
              height={railBottom - railTop}
            />
          );
        })}
      {!drawNodes && (
        <rect className="qz-catch is-even" x={left} y={railTop} width={right - left} height={railBottom - railTop} />
      )}
      <line className="qz-rail" x1={left} x2={right} y1={railTop} y2={railTop} />
      <line className="qz-rail" x1={left} x2={right} y1={railBottom} y2={railBottom} />
      <text className="qz-tick" x={left - 6} y={railTop + 4} textAnchor="end">
        weight
      </text>
      <text className="qz-tick" x={left - 6} y={railBottom + 4} textAnchor="end">
        level
      </text>
      <text className="qz-tick" x={left - 6} y={codeY} textAnchor="end">
        code
      </text>
      {snaps}
      {marks}
      {Array.from({ length: levelCount }, (_, offset) => {
        const code = block.minCode + offset;
        const x = left + inset + offset * gap;
        const hits = perCode.get(code) ?? 0;
        const labelled = (block.minCode === 0 ? offset : code) % labelEvery === 0;
        return (
          <g key={code}>
            {drawNodes ? (
              <circle
                className={`qz-node ${hits > 0 ? "is-used" : "is-unused"}`}
                cx={x}
                cy={railBottom}
                r={hits > 0 ? 2.2 + (nodeMax - 2.2) * Math.sqrt(hits / busiest) : Math.min(3, gap * 0.3)}
              />
            ) : (
              <line
                className={`qz-level ${hits > 0 ? "is-used" : "is-unused"}`}
                x1={x}
                x2={x}
                y1={railBottom - 4}
                y2={railBottom + 5}
              />
            )}
            {labelled && (
              <text className="qz-tick" x={x} y={codeY} textAnchor="middle">
                {code < 0 ? `−${-code}` : code}
              </text>
            )}
          </g>
        );
      })}
      {gap >= 12 && levelCount > 1 && (
        <path
          className="qz-bracket"
          d={`M${bx(levelValue(middle))},${bracketY - 3} v3 H${bx(levelValue(middle + 1))} v-3`}
        />
      )}
      <text className="qz-tick" x={left + inset} y={valueY} textAnchor="middle">
        {signed(gridLow)}
      </text>
      <text className="qz-tick" x={right - inset} y={valueY} textAnchor="middle">
        {signed(gridHigh)}
      </text>
      <text className="qz-step-label" x={(bx(levelValue(middle)) + bx(levelValue(middle + 1))) / 2} y={valueY} textAnchor="middle">
        {gap >= 12 ? `one step = ${fmt(block.scale, 4)}` : `${levelCount} levels, ${fmt(block.scale, 4)} apart`}
      </text>

      <path
        className="qz-funnel"
        d={`M${left + inset},${funnelTop} H${right - inset} L${xAt(gridHigh)},${histTop} H${xAt(gridLow)} Z`}
      />
      <text className="qz-axis-label" x={left} y={histTop - 7}>
        all {source.length} weights per bin (square-root scale) · shaded = where {blockName} sits
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
          const value = levelValue(code);
          return (
            <line
              key={code}
              className={`qz-level ${perCode.has(code) ? "is-used" : "is-unused"}`}
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
      {!outlierInBlock && (
        <g className="qz-outlier">
          <line x1={xAt(outlier.value)} x2={xAt(outlier.value)} y1={histBottom - 26} y2={histBottom - 6} />
          <text x={xAt(outlier.value) - 4} y={histBottom - 30} textAnchor="end">
            largest weight {outlier.label} = {fmt(outlier.value)}
          </text>
        </g>
      )}

      <text className="qz-axis-label" x={left} y={errTop - 9}>
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

/**
 * The storage bill as one bar on a 0 to 32 bit scale: the solid part is each weight's code, the
 * hatched part is the per-block scales shared out over the weights.
 */
export function StorageBill({ codeBits, bitsPerWeight }: { codeBits: number; bitsPerWeight: number }) {
  const overhead = Math.max(0, bitsPerWeight - codeBits);
  const share = (value: number) => `${Math.min(100, (value / 32) * 100)}%`;
  return (
    <figure
      className="qz-bill"
      role="img"
      aria-label={`Bits per weight on a 0 to 32 scale: ${codeBits} code bits plus ${overhead.toFixed(3)} bits of scales is ${bitsPerWeight.toFixed(3)}, against 16 for fp16 and 32 for float32.`}
    >
      <figcaption>
        <span className="qz-bill__key is-codes">codes {codeBits}</span>
        <span className="qz-bill__key is-scales">+ scales {overhead.toFixed(3)}</span>
        <strong>= {bitsPerWeight.toFixed(3)} bits per weight</strong>
      </figcaption>
      <div className="qz-bill__track">
        <i className="qz-bill__codes" style={{ width: share(codeBits) }} />
        <i className="qz-bill__scales" style={{ width: share(overhead) }} />
        <b className="qz-bill__mark" style={{ left: "50%" }} />
      </div>
      <div className="qz-bill__axis">
        <span style={{ left: "0%" }}>0</span>
        <span style={{ left: "25%" }}>8</span>
        <span style={{ left: "50%" }}>16 · fp16</span>
        <span style={{ left: "75%" }}>24</span>
        <span className="is-end" style={{ left: "100%" }}>
          32 · float32
        </span>
      </div>
    </figure>
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
  bits,
  maxBpw,
}: {
  series: FrontierSeries[];
  formats: { name: string; bpw: number; cost: number }[];
  current: { bpw: number; cost: number };
  bits: number;
  maxBpw: number;
}) {
  const left = 58;
  const right = 700;
  const top = 20;
  const bottom = 226;
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
      const width = format.name.length * 6.4;
      const candidates = [
        { dx: 7, dy: -6, anchor: "start" as const },
        { dx: 7, dy: 13, anchor: "start" as const },
        { dx: -7, dy: -6, anchor: "end" as const },
        { dx: -7, dy: 13, anchor: "end" as const },
        { dx: 7, dy: 26, anchor: "start" as const },
      ];
      const box = (candidate: (typeof candidates)[number]) => {
        const x0 = candidate.anchor === "start" ? x + candidate.dx : x + candidate.dx - width;
        return { x0, x1: x0 + width, y0: y + candidate.dy - 10, y1: y + candidate.dy + 1 };
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

  // The same code width at each granularity: the comparison the lesson asks for.
  const sameBits = series
    .map((entry) => entry.points.find((point) => point.bits === bits))
    .filter((point): point is FrontierSeries["points"][number] => point !== undefined && visible(point))
    .sort((a, b) => a.bpw - b.bpw);

  const cx = xAt(Math.min(current.bpw, xMax));
  const cy = yAt(current.cost);
  const currentShown = current.bpw <= xMax;
  const calloutEnd = cx > right - 210;
  const costText =
    current.cost <= COST_FLOOR ? "≤0.01%" : `+${current.cost >= 10 ? current.cost.toFixed(0) : current.cost.toFixed(2)}%`;

  return (
    <figure className="qz-frontier">
      <svg
        viewBox="0 0 720 274"
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
          )}. Formats: ${formats.map((format) => `${format.name} ${format.bpw} bpw ${format.cost.toFixed(2)}%`).join(", ")}. Current setting: ${current.bpw.toFixed(2)} bpw, ${current.cost.toFixed(2)}%.`}
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
            <text className="qz-tick" x={xAt(tick)} y={bottom + 16} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        <text className="qz-better" x={right - 6} y={top + 14} textAnchor="end">
          ↙ better: fewer bits, smaller loss
        </text>
        {currentShown && (
          <path className="qz-guide" d={`M${left},${cy} H${cx} V${bottom}`} />
        )}
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
        {sameBits.length > 1 && (
          <g className="qz-iso">
            <polyline points={sameBits.map((point) => `${xAt(point.bpw)},${yAt(point.cost)}`).join(" ")} />
            {sameBits.map((point) => (
              <circle key={point.bpw} cx={xAt(point.bpw)} cy={yAt(point.cost)} r={6.5} />
            ))}
          </g>
        )}
        {labels.map((format) => (
          <g key={format.name} className="qz-format-point">
            <path d={`M${format.x},${format.y - 4.6} l4.6,4.6 l-4.6,4.6 l-4.6,-4.6 Z`} />
            <text x={format.lx} y={format.ly} textAnchor={format.anchor}>
              {format.name}
            </text>
          </g>
        ))}
        {currentShown && (
          <g className="qz-callout">
            <circle className="qz-current" cx={cx} cy={cy} r={9} />
            <text x={calloutEnd ? cx - 12 : cx + 12} y={cy < top + 26 ? cy + 20 : cy - 12} textAnchor={calloutEnd ? "end" : "start"}>
              your setting: {current.bpw.toFixed(2)} bits, {costText}
            </text>
          </g>
        )}
        <text className="qz-axis-label" x={(left + right) / 2} y={bottom + 34} textAnchor="middle">
          bits per weight, scales included
          {hidden > 0 ? ` · ${hidden} point${hidden === 1 ? "" : "s"} beyond ${xMax} not drawn` : ""}
          {currentShown ? "" : " · your setting is beyond the axis"}
        </text>
        <text className="qz-axis-label" x={left} y={11}>
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
        {sameBits.length > 1 && (
          <span className="is-iso">
            <svg viewBox="0 0 26 10" aria-hidden="true">
              <line x1={1} x2={25} y1={5} y2={5} />
              <circle cx={13} cy={5} r={4} />
            </svg>
            the same {bits}-bit codes at each granularity
          </span>
        )}
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

const MOSAIC_CEILING = 9;
const NOMINAL_BITS = 4.5;

/**
 * The downloaded file as blocks: width is a group's share of the parameters, height its bits per
 * weight, so area is its bytes. Whatever rises above the dashed 4.5 line is what the file spends
 * beyond a pure Q4_K file.
 */
export function FileMosaic({
  slices,
  shortLabels,
  effective,
}: {
  slices: ReadonlyArray<FileSlice>;
  shortLabels: Record<string, string>;
  effective: number;
}) {
  const left = 40;
  const right = 704;
  const top = 26;
  const bottom = 212;
  const yAt = (bitsPerWeight: number) => bottom - (Math.min(bitsPerWeight, MOSAIC_CEILING) / MOSAIC_CEILING) * (bottom - top);
  let cursor = left;
  const blocks = slices.map((slice, index) => {
    const x = cursor;
    const width = slice.parameterShare * (right - left);
    cursor += width;
    return { slice, x, width, tag: String.fromCharCode(65 + index) };
  });
  const thin = blocks.filter((entry) => entry.width < 3);
  const effectiveY = yAt(effective);
  const name = (slice: FileSlice) => shortLabels[slice.key] ?? slice.label;
  const summary = `Each group of tensors as a block whose width is its share of the parameters and whose height is its bits per weight. ${slices
    .map(
      (slice) =>
        `${slice.label}: ${(slice.parameterShare * 100).toFixed(1)}% of parameters at ${slice.bitsPerWeight.toFixed(2)} bits per weight, ${(slice.byteShare * 100).toFixed(1)}% of bytes`,
    )
    .join("; ")}. A Q4_K block is ${NOMINAL_BITS} bits per weight and the whole file averages ${effective.toFixed(3)}.`;

  return (
    <svg className="qz-plot qz-mosaic" viewBox="0 0 720 248" role="img" aria-label={summary}>
      <defs>
        <pattern id="qz-mosaic-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line className="qz-mosaic__hatch" x1="0" x2="0" y1="0" y2="6" />
        </pattern>
      </defs>
      <text className="qz-axis-label" x={left} y={12}>
        width = share of parameters · height = bits per weight · area = bytes · hatched = bits above a Q4_K block
      </text>
      {[0, 2, 4, 6, 8].map((tick) => (
        <g key={tick}>
          <line className="qz-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
          <text className="qz-tick" x={left - 6} y={yAt(tick) + 4} textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      {blocks.map(({ slice, x, width, tag }) => {
        if (width < 3) return null;
        const blockTop = yAt(slice.bitsPerWeight);
        const baseTop = yAt(Math.min(slice.bitsPerWeight, NOMINAL_BITS));
        const centre = x + width / 2;
        const roomy = width >= 96;
        return (
          <g key={slice.key}>
            <rect className="qz-mosaic__base" x={x + 1} y={baseTop} width={width - 2} height={bottom - baseTop} />
            {slice.bitsPerWeight > NOMINAL_BITS && (
              <>
                <rect className="qz-mosaic__extra" x={x + 1} y={blockTop} width={width - 2} height={baseTop - blockTop} />
                <rect
                  className="qz-mosaic__extra-hatch"
                  x={x + 1}
                  y={blockTop}
                  width={width - 2}
                  height={baseTop - blockTop}
                  fill="url(#qz-mosaic-hatch)"
                />
              </>
            )}
            <text
              className="qz-mosaic__tag"
              x={centre}
              y={effectiveY > blockTop - 19 && effectiveY < blockTop - 2 ? effectiveY - 6 : blockTop - 6}
              textAnchor="middle"
            >
              {width >= 60 ? `${tag} · ${slice.bitsPerWeight.toFixed(2)}` : tag}
            </text>
            {roomy && (
              <>
                <text className="qz-mosaic__name" x={centre} y={bottom - 27} textAnchor="middle">
                  {name(slice)}
                </text>
                <text className="qz-mosaic__detail" x={centre} y={bottom - 11} textAnchor="middle">
                  {(slice.byteShare * 100).toFixed(1)}% of bytes
                </text>
              </>
            )}
          </g>
        );
      })}
      <line className="qz-mosaic__nominal" x1={left} x2={right} y1={yAt(NOMINAL_BITS)} y2={yAt(NOMINAL_BITS)} />
      <line className="qz-mosaic__effective" x1={left} x2={right} y1={yAt(effective)} y2={yAt(effective)} />
      <text className="qz-mosaic__line-label is-effective" x={left + 6} y={yAt(effective) - 5}>
        whole file {effective.toFixed(3)}
      </text>
      <text className="qz-mosaic__line-label" x={left + 6} y={yAt(NOMINAL_BITS) + 14}>
        a Q4_K block {NOMINAL_BITS}
      </text>
      {thin.length > 0 && (
        <text className="qz-mosaic__line-label" x={right - 4} y={top + 6} textAnchor="end">
          {thin.map((entry) => entry.tag).join(", ")}: too thin to draw ▸
        </text>
      )}
      <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line className="plot-axis" x1={left + tick * (right - left)} x2={left + tick * (right - left)} y1={bottom} y2={bottom + 4} />
          <text
            className="qz-tick"
            x={left + tick * (right - left)}
            y={bottom + 16}
            textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"}
          >
            {(tick * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <text className="qz-axis-label" x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
        share of the file's {slices.reduce((total, slice) => total + slice.tensors, 0)} tensors' parameters
      </text>
    </svg>
  );
}

/** One float format's bits as cells: a sign bit, then the exponent, then the mantissa. */
export function BitStrip({ format }: { format: FloatFormat }) {
  return (
    <span className="qz-bits" aria-hidden="true">
      <i className="is-sign" />
      {Array.from({ length: format.exponentBits }, (_, index) => (
        <i key={`e${index}`} className="is-exponent" />
      ))}
      {Array.from({ length: format.mantissaBits }, (_, index) => (
        <i key={`m${index}`} className="is-mantissa" />
      ))}
    </span>
  );
}

/** The values a format can store between 1 and 2, drawn as a ruler: one tick per value. */
export function SpacingRuler({ format }: { format: FloatFormat }) {
  const width = 112;
  const steps = 2 ** format.mantissaBits;
  const dense = steps > width / 2;
  return (
    <svg className="qz-ruler" viewBox={`0 0 ${width} 14`} aria-hidden="true">
      {dense ? (
        <rect className="qz-ruler__solid" x={0} y={2} width={width} height={10} style={{ opacity: steps > width * 4 ? 1 : 0.6 }} />
      ) : (
        Array.from({ length: steps + 1 }, (_, index) => (
          <line key={index} x1={0.5 + (index / steps) * (width - 1)} x2={0.5 + (index / steps) * (width - 1)} y1={2} y2={12} />
        ))
      )}
    </svg>
  );
}

export const RANGE_AXIS = { low: -8, high: 8 };
const rangeShare = (value: number) =>
  Math.min(1, Math.max(0, (Math.log10(value) - RANGE_AXIS.low) / (RANGE_AXIS.high - RANGE_AXIS.low)));

/**
 * A format's range on a shared log axis from 1e-8 to 1e8: hatched where values are subnormal, solid
 * from the smallest normal to the largest finite value, with the learner's value as a marker.
 */
export function RangeBar({
  stats,
  value,
  state,
}: {
  stats: FloatFormatStats;
  value: number;
  state: "normal" | "subnormal" | "zero" | "overflow";
}) {
  const start = rangeShare(stats.minNormal);
  const end = rangeShare(stats.maxFinite);
  const subStart = rangeShare(stats.minSubnormal);
  const pastLow = Math.log10(stats.minSubnormal) < RANGE_AXIS.low;
  const pastHigh = Math.log10(stats.maxFinite) > RANGE_AXIS.high;
  const percentOf = (share: number) => `${share * 100}%`;
  return (
    <span className={`qz-range is-${state}`} aria-hidden="true">
      {subStart < start && (
        <i className="qz-range__sub" style={{ left: percentOf(subStart), width: percentOf(start - subStart) }} />
      )}
      <i className="qz-range__bar" style={{ left: percentOf(start), width: percentOf(end - start) }} />
      {pastLow && <b className="qz-range__more is-low">◂</b>}
      {pastHigh && <b className="qz-range__more is-high">▸</b>}
      <u className="qz-range__value" style={{ left: percentOf(rangeShare(value)) }} />
    </span>
  );
}

/** Tick labels for the range column's shared log axis. */
export function RangeAxis({ value, label }: { value: number; label: string }) {
  const ticks = [-8, -4, 0, 4, 8];
  return (
    <span className="qz-range-axis" aria-hidden="true">
      {ticks.map((tick) => (
        <span key={tick} style={{ left: `${((tick - RANGE_AXIS.low) / (RANGE_AXIS.high - RANGE_AXIS.low)) * 100}%` }}>
          {tick === 0 ? "1" : `1e${tick < 0 ? "−" : ""}${Math.abs(tick)}`}
        </span>
      ))}
      <em style={{ left: `${rangeShare(value) * 100}%` }}>
        {label}
        <i>▼</i>
      </em>
    </span>
  );
}
