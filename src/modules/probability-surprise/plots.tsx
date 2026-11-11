import { useId } from "react";
import { niceTicks, useMeasuredWidth } from "@app/module-sdk";
import { ONE_SD_SHARE, gaussianPdf, sum } from "./chance";
import { bits, pct } from "./pictures";

/**
 * The picture of an average surprise. Each word is a block: as wide as how often it happens, as tall
 * as its surprise in bits. The shaded area is the average, and the dashed line is the level a flat
 * block of the same area would reach. `ghost` draws a second outline (the truth's own surprises),
 * so the extra a wrong guess pays is the gap between the two levels, bracketed in the right margin.
 */
export function Skyline({
  label,
  names,
  widths,
  heights,
  average,
  averageName,
  ghost,
  tone = "truth",
}: {
  label: string;
  names: ReadonlyArray<string>;
  widths: ReadonlyArray<number>;
  heights: ReadonlyArray<number>;
  average: number;
  averageName: string;
  ghost?: { heights: ReadonlyArray<number>; average: number; name: string };
  tone?: "truth" | "guess";
}) {
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(660);
  const width = Math.max(300, measured);
  const margin = width < 460 ? 100 : 128;
  const left = 34;
  const right = width - margin;
  const top = 24;
  const bottom = top + 196;
  const height = bottom + 78;
  const shown = (index: number) => (widths[index] ?? 0) > 0;
  const finite = [
    ...heights.filter((value, index) => shown(index) && Number.isFinite(value)),
    ...(ghost?.heights ?? []).filter((value, index) => shown(index) && Number.isFinite(value)),
  ];
  const yMax = Math.max(3, Math.ceil(Math.max(0, ...finite) + 0.3));
  const xAt = (share: number) => left + share * (right - left);
  const yAt = (value: number) => bottom - (Math.min(yMax, value) / yMax) * (bottom - top);
  let edge = 0;
  const blocks = widths.map((share, index) => {
    const start = edge;
    edge += share;
    return { index, start, end: edge, share, bits: heights[index], ghost: ghost?.heights[index] };
  });
  const visible = blocks.filter((block) => block.share > 0 && Number.isFinite(block.bits));
  const yTicks = niceTicks(0, yMax, 4).filter((tick) => Number.isInteger(tick));
  const yLevel = yAt(average);
  const yGhost = ghost ? yAt(ghost.average) : yLevel;
  const gap = ghost ? Math.max(0, average - ghost.average) : 0;

  return (
    <div className="ps-plot" ref={frameRef}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="ps-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="ps-tick" x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={left - 26} y={11}>
          surprise, bits ↑
        </text>
        {visible.map((block) => {
          const x0 = xAt(block.start);
          const x1 = xAt(block.end);
          const wide = x1 - x0;
          const mid = (x0 + x1) / 2;
          const y = yAt(block.bits);
          const tall = bottom - y;
          const name = names[block.index];
          // Keep the value clear of the dashed outline when that sits just above the block's top.
          const outline = Number.isFinite(block.ghost) ? yAt(block.ghost ?? 0) : null;
          const crowded = outline !== null && outline < y - 1 && outline > y - 19;
          const valueY = !crowded ? y - 5 : tall >= 20 ? y + 14 : (outline ?? y) - 5;
          return (
            <g key={block.index}>
              <rect
                className={`ps-block ps-block--${tone} ps-block--${block.index % 2 === 0 ? "even" : "odd"}`}
                x={x0}
                y={y}
                width={Math.max(0, wide)}
                height={Math.max(0, tall)}
              />
              {wide >= 58 ? (
                <text className="ps-block-bits ps-halo" x={mid} y={valueY} textAnchor="middle">
                  {bits(block.bits)} bits
                </text>
              ) : wide >= 13 ? (
                tall >= 62 ? (
                  <text className="ps-block-bits ps-halo" x={mid + 4} y={y + 6} textAnchor="end" transform={`rotate(-90 ${mid + 4} ${y + 6})`}>
                    {bits(block.bits)} bits
                  </text>
                ) : (
                  <text className="ps-block-bits ps-halo" x={mid + 4} y={y - 5} transform={`rotate(-90 ${mid + 4} ${y - 5})`}>
                    {bits(block.bits)} bits
                  </text>
                )
              ) : null}
              {wide >= 104 && tall >= (crowded ? 56 : 34) && (
                <text className="ps-block-area ps-halo" x={mid} y={y + tall / 2 + (crowded ? 11 : 4)} textAnchor="middle">
                  {pct(block.share, 1)} × {bits(block.bits)} = {bits(block.share * block.bits)}
                </text>
              )}
              {wide >= name.length * 7 + 6 ? (
                <text className="ps-block-name" x={mid} y={bottom + 16} textAnchor="middle">
                  {name}
                </text>
              ) : wide >= 12 ? (
                <text className="ps-block-name" x={mid + 4} y={bottom + 6} textAnchor="end" transform={`rotate(-90 ${mid + 4} ${bottom + 6})`}>
                  {name}
                </text>
              ) : null}
            </g>
          );
        })}
        {ghost &&
          visible.map((block) =>
            Number.isFinite(block.ghost) ? (
              <path
                key={`ghost-${block.index}`}
                className="ps-block-ghost"
                d={`M${xAt(block.start) + 1} ${bottom}V${yAt(block.ghost ?? 0)}H${xAt(block.end) - 1}V${bottom}`}
              />
            ) : null,
          )}
        <line className="ps-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <line className="ps-axis" x1={left} x2={left} y1={top} y2={bottom} />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <text
            key={tick}
            className="ps-tick"
            x={xAt(tick)}
            y={bottom + 58}
            textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"}
          >
            {pct(tick)}
          </text>
        ))}
        <text className="ps-axis-label" x={(left + right) / 2} y={height - 5} textAnchor="middle">
          share of the time each word happens (widths add to 100%) →
        </text>
        {ghost && (
          <g>
            <line className="ps-level ps-level--ghost" x1={left} x2={right + 6} y1={yGhost} y2={yGhost} />
            <text className="ps-level-name ps-level-name--ghost" x={right + 12} y={yGhost + 14}>
              {ghost.name} {bits(ghost.average)}
            </text>
          </g>
        )}
        <line className={`ps-level ps-level--${tone}`} x1={left} x2={right + 6} y1={yLevel} y2={yLevel} />
        {ghost ? (
          <text className="ps-level-name" x={right + 12} y={yLevel - 6}>
            {averageName} {bits(average)} bits
          </text>
        ) : (
          <>
            <text className="ps-level-name" x={right + 12} y={yLevel - 6}>
              {averageName}
            </text>
            <text className="ps-level-name" x={right + 12} y={yLevel + 14}>
              {bits(average)} bits
            </text>
          </>
        )}
        {ghost && yGhost - yLevel >= 4 && <line className="ps-bracket" x1={right + 5} x2={right + 5} y1={yLevel + 1} y2={yGhost - 1} />}
        {ghost && yGhost - yLevel >= 18 && (
          <text className="ps-bracket-name" x={right + 12} y={(yLevel + yGhost) / 2 + 4}>
            wasted {bits(gap)}
          </text>
        )}
      </svg>
    </div>
  );
}

/**
 * The pile of simulated test scores, one bar per possible score (grouped into 0.01-wide bins for the
 * largest test), with the bell drawn from the pile's own mean and variance laid over it. Bars within
 * one standard deviation of the mean are filled differently from the rest.
 */
export function BellPile({
  pile,
  mean,
  sd,
  label,
}: {
  pile: ReadonlyArray<number>;
  mean: number;
  sd: number;
  label: string;
}) {
  const id = useId().replace(/:/g, "");
  const [frameRef, measured] = useMeasuredWidth<HTMLDivElement>(660);
  const width = Math.max(300, measured);
  const left = 44;
  const right = width - 14;
  const top = 58;
  const bottom = top + 178;
  const height = bottom + 40;
  const size = pile.length - 1;
  const tests = Math.max(1, sum(pile));
  // One bar per possible score, except that 400 questions are grouped four scores to a 0.01 bin.
  const group = size > 100 ? Math.round(size / 100) : 1;
  const binWidth = group / size;
  const bins = new Map<number, number>();
  pile.forEach((count, hits) => {
    const centre = group > 1 ? Math.floor((hits + group / 2) / group) * binWidth : hits / size;
    bins.set(centre, (bins.get(centre) ?? 0) + count / tests);
  });
  const spread = Math.max(1e-9, sd * sd);
  const bell = (x: number) => gaussianPdf(x, mean, spread) * binWidth;
  const yMax = Math.max(0.02, ...bins.values(), bell(mean)) * 1.1;
  const xAt = (score: number) => left + Math.min(1, Math.max(0, score)) * (right - left);
  const yAt = (share: number) => bottom - (Math.min(yMax, share) / yMax) * (bottom - top);
  const slot = binWidth * (right - left);
  const barWidth = Math.max(1.5, slot - (slot > 6 ? 2 : 0.6));
  const curve = Array.from({ length: 201 }, (_, index) => {
    const x = index / 200;
    return `${index === 0 ? "M" : "L"}${xAt(x).toFixed(1)} ${yAt(bell(x)).toFixed(1)}`;
  }).join("");
  const low = mean - sd;
  const high = mean + sd;
  const yTicks = niceTicks(0, yMax, 3).filter((tick) => tick <= yMax);
  const labelX = Math.min(right - 150, Math.max(left + 150, xAt(mean)));
  const anchorX = right - left < 320 ? (left + right) / 2 : labelX;

  return (
    <div className="ps-plot" ref={frameRef}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        <defs>
          <clipPath id={`${id}-clip`}>
            <rect x={left} y={top - 2} width={right - left} height={bottom - top + 2} />
          </clipPath>
        </defs>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="ps-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="ps-tick" x={left - 6} y={yAt(tick) + 3.5} textAnchor="end">
              {pct(tick, tick > 0 && tick < 0.1 ? 1 : 0)}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={left - 34} y={11}>
          share of the {tests.toLocaleString("en-US")} tests ↑
        </text>
        <g clipPath={`url(#${id}-clip)`}>
          <rect className="ps-band" x={xAt(low)} y={top} width={Math.max(0, xAt(high) - xAt(low))} height={bottom - top} />
          {[...bins.entries()].map(([centre, share]) =>
            share > 0 ? (
              <rect
                key={centre}
                className={`ps-pile ${centre >= low && centre <= high ? "is-inside" : ""}`.trim()}
                x={xAt(centre) - barWidth / 2}
                y={yAt(share)}
                width={barWidth}
                height={bottom - yAt(share)}
              />
            ) : null,
          )}
          <path className="ps-bell" d={curve} />
          <line className="ps-mean" x1={xAt(mean)} x2={xAt(mean)} y1={top} y2={bottom} />
        </g>
        <line className="ps-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className="ps-axis" x1={xAt(tick)} x2={xAt(tick)} y1={bottom} y2={bottom + 4} />
            <text
              className="ps-tick"
              x={xAt(tick)}
              y={bottom + 15}
              textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"}
            >
              {tick}
            </text>
          </g>
        ))}
        <text className="ps-axis-label" x={(left + right) / 2} y={height - 6} textAnchor="middle">
          a test’s score: average points per question (0 = all misses, 1 = all hits) →
        </text>
        <text className="ps-mean-name" x={anchorX} y={29} textAnchor="middle">
          dashed line: the mean, {mean.toFixed(2)}
        </text>
        <text className="ps-band-name" x={anchorX} y={43} textAnchor="middle">
          shaded band: 1 standard deviation each side ({sd.toFixed(3)})
        </text>
        <path className="ps-band-span" d={`M${xAt(low)} ${top - 5}v-4M${xAt(low)} ${top - 7}H${xAt(high)}M${xAt(high)} ${top - 5}v-4`} />
      </svg>
      <p className="ps-plot__note">
        About {Math.round(ONE_SD_SHARE * 100)}% of the area under any bell lies inside its shaded band.
      </p>
    </div>
  );
}
