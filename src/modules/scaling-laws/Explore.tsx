import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  blockWeights,
  CHINCHILLA,
  clampWidth,
  denseParameters,
  FITS,
  formatCount,
  formatFlops,
  frontierLossExponent,
  isoFlopCurve,
  LAYER_RANGE,
  lossTerms,
  optimalAllocation,
  PARAMETER_RANGE,
  parameterExponent,
  ratioAllocation,
  superscript,
  TOKEN_RANGE,
  trainingFlops,
  WIDTH_RANGE,
  type FitId,
  type ScalingFit,
} from "./scaling";

const GOPHER = { parameters: 280e9, tokens: 300e9 };

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const round3 = (value: number) => Number(value.toPrecision(3));
const sci = (value: number) => {
  const exponent = Math.floor(Math.log10(value));
  return `${(value / Math.pow(10, exponent)).toFixed(2)} × 10${superscript(exponent)}`;
};

function LogRange({
  label,
  value,
  min,
  max,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  format: (value: number) => string;
}) {
  const low = Math.log10(min);
  const high = Math.log10(max);
  const logValue = Math.log10(value);
  const progress = ((logValue - low) / (high - low)) * 100;
  return (
    <label className="range-control">
      <span>
        {label}
        <output>{format(value)}</output>
      </span>
      <input
        type="range"
        aria-label={label}
        aria-valuetext={format(value)}
        min={low}
        max={high}
        step={0.02}
        value={logValue}
        style={{ "--range-progress": `${progress}%` } as CSSProperties}
        onChange={(event) => onChange(Math.pow(10, Number(event.target.value)))}
      />
    </label>
  );
}

/* ---------- log-scale tracks in the controls card ---------- */

const trackPct = (value: number, low: number, high: number) =>
  `${clamp(((Math.log10(value) - low) / (high - low)) * 100, 0, 100)}%`;

/** Compute on a log track: both sliders push the same dot, and Hold compute pins it. */
function BudgetMeter({ parameters, tokens, hold }: { parameters: number; tokens: number; hold: boolean }) {
  const compute = trainingFlops(parameters, tokens);
  const low = 17.5;
  const high = 26.5;
  return (
    <div
      className={`sl-meter ${hold ? "is-held" : ""}`}
      role="img"
      aria-label={`Compute is 6 times ${formatCount(parameters)} parameters times ${formatCount(tokens)} tokens, ${formatFlops(compute)} FLOPs. ${hold ? "It is held fixed." : "It moves when either slider moves."}`}
    >
      <p className="sl-meter__sum">
        <span>C =</span> 6 × <b>{formatCount(parameters)}</b> × <b>{formatCount(tokens)}</b> ={" "}
        <strong>{formatFlops(compute)}</strong> <span>FLOPs</span>
      </p>
      <div className="sl-track">
        {FAMILY.map((exponent) => (
          <i key={exponent} className="sl-track__tick" style={{ left: trackPct(Math.pow(10, exponent), low, high) }}>
            {exponent % 2 === 1 && <small>{`10${superscript(exponent)}`}</small>}
          </i>
        ))}
        <b className="sl-track__dot" style={{ left: trackPct(compute, low, high) }}>
          <small>{hold ? "held" : "your budget"}</small>
        </b>
      </div>
    </div>
  );
}

/** N from the two dimension sliders against the Parameters slider, on one log track from 100M to 1T. */
function DimensionTrack({ fromDimensions, parameters }: { fromDimensions: number; parameters: number }) {
  const rows = [
    { id: "dims", name: "12 · L · d²", value: fromDimensions },
    { id: "slider", name: "Parameters", value: parameters },
  ];
  const outside = fromDimensions < PARAMETER_RANGE.min * 1e9 || fromDimensions > PARAMETER_RANGE.max * 1e9;
  return (
    <div
      className="sl-dimtrack"
      role="img"
      aria-label={`12 times L times d squared is ${formatCount(fromDimensions)}${outside ? ", outside the slider's range" : ""}; the Parameters slider is at ${formatCount(parameters)}.`}
    >
      {rows.map((row) => (
        <div key={row.id} className={`sl-dimtrack__row sl-dimtrack__row--${row.id}`}>
          <span>{row.name}</span>
          <div className="sl-track">
            {[8, 9, 10, 11, 12].map((exponent) => (
              <i key={exponent} className="sl-track__tick" style={{ left: trackPct(Math.pow(10, exponent), 8, 12) }} />
            ))}
            <b className="sl-track__dot" style={{ left: trackPct(row.value, 8, 12) }} />
          </div>
          <strong>{formatCount(row.value)}</strong>
        </div>
      ))}
      <div className="sl-dimtrack__row sl-dimtrack__row--axis" aria-hidden="true">
        <span />
        <div className="sl-track sl-track--axis">
          {["100M", "1B", "10B", "100B", "1T"].map((label, index) => (
            <i key={label} className="sl-track__tick" style={{ left: `${index * 25}%` }}>
              <small>{label}</small>
            </i>
          ))}
        </div>
        <strong />
      </div>
    </div>
  );
}

/* ---------- isoFLOP chart ---------- */

const ISO_X = { min: 8, max: 12 };
const ISO_Y = { min: 1.6, max: 3.8 };
const FAMILY = [19, 20, 21, 22, 23, 24, 25];

type CurvePoint = { parameters: number; loss: number };
type Pixel = { x: number; y: number };
type Box = { x: number; y: number; width: number; height: number };

/** Distance from a point to a box, 0 when the point is inside it. */
const gapToBox = (point: Pixel, box: Box) =>
  Math.hypot(
    Math.max(box.x - point.x, 0, point.x - box.x - box.width),
    Math.max(box.y - point.y, 0, point.y - box.y - box.height),
  );

const boxesTouch = (a: Box, b: Box, pad = 3) =>
  a.x < b.x + b.width + pad && b.x < a.x + a.width + pad && a.y < b.y + b.height + pad && b.y < a.y + a.height + pad;

/**
 * Picks the spot around `anchor` where a short text label is furthest from every drawn line.
 * The lines it avoids do not depend on the reader's run, so the label stays put while sliders move.
 */
function placeLabel(
  anchor: Pixel,
  text: string,
  obstacles: ReadonlyArray<Pixel>,
  bounds: { left: number; right: number; top: number; bottom: number },
  taken: ReadonlyArray<Box>,
) {
  const width = text.length * 6.4 + 4;
  const height = 12;
  const fallback: Box = { x: anchor.x - width / 2, y: anchor.y + 9, width, height };
  let best = { box: fallback, score: -Infinity, reach: 9 };
  for (const dy of [-8, 19, 23, 27, 31, 35, 39, 43, -20, -32]) {
    for (const side of [1, -1, 0]) {
      const box: Box = {
        x: side === 1 ? anchor.x + 7 : side === -1 ? anchor.x - 7 - width : anchor.x - width / 2,
        y: anchor.y + dy - 10,
        width,
        height,
      };
      if (
        box.x < bounds.left + 2 ||
        box.x + width > bounds.right - 2 ||
        box.y < bounds.top + 2 ||
        box.y + height > bounds.bottom - 1
      )
        continue;
      let clear = 9;
      for (const obstacle of obstacles) {
        clear = Math.min(clear, gapToBox(obstacle, box));
        if (clear === 0) break;
      }
      if (taken.some((other) => boxesTouch(box, other))) clear = -4;
      const reach = gapToBox(anchor, box);
      const score = clear - reach * 0.06;
      if (score > best.score) best = { box, score, reach };
    }
  }
  return best;
}

/**
 * Chart geometry in CSS pixels. The chart is drawn at its measured width so that
 * text stays the same size on wide and narrow layouts.
 */
function isoGeometry(measured: number) {
  const width = Math.max(320, Math.round(measured));
  const box = {
    width,
    height: Math.round(Math.min(380, Math.max(270, width * 0.55))),
    left: 50,
    right: 34,
    top: 24,
    bottom: 42,
  };
  const isoX = (parameters: number) =>
    box.left +
    ((Math.log10(parameters) - ISO_X.min) / (ISO_X.max - ISO_X.min)) * (box.width - box.left - box.right);
  const isoY = (loss: number) =>
    box.top + ((ISO_Y.max - loss) / (ISO_Y.max - ISO_Y.min)) * (box.height - box.top - box.bottom);
  const inIsoX = (parameters: number) => {
    const logN = Math.log10(parameters);
    return logN >= ISO_X.min - 1e-9 && logN <= ISO_X.max + 1e-9;
  };
  /** Where a curve's right branch leaves the plot: through the top edge or the right edge. */
  const curveExit = (points: ReadonlyArray<CurvePoint>) => {
    let last = points[points.length - 1];
    for (let index = points.length - 1; index >= 0; index -= 1) {
      if (points[index].loss <= ISO_Y.max) {
        last = points[index];
        break;
      }
    }
    const atRight = last === points[points.length - 1];
    return atRight
      ? { x: box.width - box.right + 4, y: isoY(last.loss) + 3.5, anchor: "start" as const }
      : { x: isoX(last.parameters), y: box.top - 7, anchor: "middle" as const };
  };
  const curvePath = (points: ReadonlyArray<CurvePoint>) => {
    let path = "";
    let open = false;
    for (const point of points) {
      const visible = point.loss <= ISO_Y.max + 0.4 && Number.isFinite(point.loss);
      if (!visible) {
        open = false;
        continue;
      }
      path += `${open ? "L" : "M"}${isoX(point.parameters).toFixed(1)} ${isoY(point.loss).toFixed(1)} `;
      open = true;
    }
    return path.trim();
  };
  /** Pixel samples along a curve, at most 3px apart, for keeping labels off it. */
  const curvePixels = (points: ReadonlyArray<CurvePoint>) => {
    const pixels: Pixel[] = [];
    let previous: Pixel | undefined;
    for (const point of points) {
      if (!Number.isFinite(point.loss) || point.loss > ISO_Y.max + 0.4) {
        previous = undefined;
        continue;
      }
      const next = { x: isoX(point.parameters), y: isoY(point.loss) };
      if (previous) {
        const steps = Math.max(1, Math.ceil(Math.hypot(next.x - previous.x, next.y - previous.y) / 3));
        for (let step = 1; step <= steps; step += 1) {
          pixels.push({
            x: previous.x + ((next.x - previous.x) * step) / steps,
            y: previous.y + ((next.y - previous.y) * step) / steps,
          });
        }
      } else pixels.push(next);
      previous = next;
    }
    return pixels;
  };
  return { box, isoX, isoY, inIsoX, curveExit, curvePath, curvePixels };
}

function useMeasuredWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next) setWidth((current) => (Math.abs(current - next) > 1 ? Math.round(next) : current));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Spreads labels that share an edge so that no two sit closer than `gap`, staying inside low..high. */
function spread(values: number[], gap: number, low: number, high: number) {
  const order = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  for (let index = 1; index < order.length; index += 1)
    order[index].value = Math.max(order[index].value, order[index - 1].value + gap);
  for (let index = order.length - 1; index >= 0; index -= 1) {
    const limit = index === order.length - 1 ? high : order[index + 1].value - gap;
    order[index].value = Math.min(order[index].value, limit);
  }
  const result = [...values];
  for (const entry of order) result[entry.index] = Math.max(low, entry.value);
  return result;
}

function IsoFlopChart({ fit, parameters, tokens }: { fit: ScalingFit; parameters: number; tokens: number }) {
  const [frame, measured] = useMeasuredWidth(640);
  const { box: ISO, isoX, isoY, inIsoX, curveExit, curvePath, curvePixels } = isoGeometry(measured);
  const compute = trainingFlops(parameters, tokens);
  const loss = lossTerms(fit, parameters, tokens).total;
  const optimum = optimalAllocation(fit, compute);
  const rule = ratioAllocation(fit, compute);
  const plotRight = ISO.width - ISO.right;
  const plotBottom = ISO.height - ISO.bottom;
  const floorY = isoY(fit.E);
  const wide = ISO.width >= 540;

  // Everything that does not move with the reader's run: the budget curves, the two loci and the
  // labels for the two real models, which are placed in the widest gap between those lines.
  const fixed = useMemo(() => {
    const family = FAMILY.map((exponent) => {
      const points = isoFlopCurve(fit, Math.pow(10, exponent), ISO_X.min, ISO_X.max, 120);
      return { exponent, points, best: optimalAllocation(fit, Math.pow(10, exponent)), exit: curveExit(points) };
    });
    const frontier = Array.from({ length: 81 }, (_, index) =>
      optimalAllocation(fit, Math.pow(10, 17.5 + index * 0.1)),
    );
    const ruleLocus = Array.from({ length: 81 }, (_, index) =>
      ratioAllocation(fit, Math.pow(10, 17.5 + index * 0.1)),
    );
    const references = [
      { id: "chinchilla", name: "Chinchilla", ...CHINCHILLA },
      { id: "gopher", name: "Gopher", ...GOPHER },
    ].map((entry) => ({ ...entry, loss: lossTerms(fit, entry.parameters, entry.tokens).total }));
    const obstacles = [
      ...family.flatMap((curve) => curvePixels(curve.points)),
      ...curvePixels(frontier),
      ...curvePixels(ruleLocus),
      ...references.map((entry) => ({ x: isoX(entry.parameters), y: isoY(entry.loss) })),
    ];
    const bounds = { left: ISO.left, right: ISO.width - ISO.right, top: ISO.top, bottom: isoY(fit.E) };
    const taken: Box[] = [];
    const placed = references.map((entry) => {
      const anchor = { x: isoX(entry.parameters), y: isoY(entry.loss) };
      const own = obstacles.filter((point) => point.x !== anchor.x || point.y !== anchor.y);
      const spot = placeLabel(anchor, entry.name, own, bounds, taken);
      taken.push(spot.box);
      return { ...entry, anchor, spot };
    });
    // Budget labels: keep neighbours on the right edge and along the top edge apart.
    const rightIndex = family.map((curve, index) => (curve.exit.anchor === "start" ? index : -1)).filter((index) => index >= 0);
    const rightY = spread(rightIndex.map((index) => family[index].exit.y), 12, ISO.top + 8, ISO.height - ISO.bottom + 3);
    const topIndex = family.map((curve, index) => (curve.exit.anchor === "middle" ? index : -1)).filter((index) => index >= 0);
    const topX = spread(topIndex.map((index) => family[index].exit.x), 30, ISO.left + 150, ISO.width - ISO.right - 4);
    const labels = family.map((curve, index) => {
      const right = rightIndex.indexOf(index);
      const top = topIndex.indexOf(index);
      return {
        exponent: curve.exponent,
        anchor: curve.exit.anchor,
        x: top >= 0 ? topX[top] : curve.exit.x,
        y: right >= 0 ? rightY[right] : curve.exit.y,
      };
    });
    // Room for the frontier caption in the empty corner under the curves, above the floor caption.
    const captionRight = ISO.left + 10 + 140;
    const lowestCurve = Math.max(
      ISO.top,
      ...obstacles.filter((point) => point.x >= ISO.left && point.x <= captionRight).map((point) => point.y),
    );
    const captionRoom = isoY(fit.E) - 15 - (lowestCurve + 4);
    const caption = { top: lowestCurve + 4, lines: captionRoom >= 26 ? 2 : captionRoom >= 13 ? 1 : 0 };
    return { family, frontier, ruleLocus, placed, labels, caption };
    // The geometry helpers depend only on the measured width.
  }, [fit, ISO.width]);

  const xTicks = [
    { value: 1e8, label: "100M" },
    { value: 1e9, label: "1B" },
    { value: 1e10, label: "10B" },
    { value: 1e11, label: "100B" },
    { value: 1e12, label: "1T" },
  ];
  const yTicks = [2, 2.5, 3, 3.5];

  // Marks on the reader's own curve.
  const user = { x: isoX(parameters), y: isoY(loss) };
  const best = { x: isoX(optimum.parameters), y: isoY(optimum.loss) };
  const twenty = { x: isoX(rule.parameters), y: isoY(rule.loss) };
  const userVisible = loss <= ISO_Y.max;
  const gap = loss - optimum.loss;
  const gapPx = best.y - Math.max(user.y, ISO.top);
  const showGap = gapPx >= 14 && inIsoX(optimum.parameters);
  const userRight = user.x >= best.x;
  const bestLeft = best.x <= twenty.x;
  // The frontier label sits in the empty corner under the frontier; its leader ends on the dashed line.
  const frontierTip = optimalAllocation(fit, Math.pow(10, 21.5));
  const frontierLabel = { x: ISO.left + 10, y: fixed.caption.top + 9 };
  // The corner caption shares the top margin with the budget labels of curves that leave through the top.
  const firstTopLabel = Math.min(...fixed.labels.filter((label) => label.anchor === "middle").map((label) => label.x));
  const cornerRoom = wide && firstTopLabel > ISO.left + 272;

  return (
    <div ref={frame} className="sl-iso-frame">
      <svg
        className="sl-iso"
        width={ISO.width}
        height={ISO.height}
        viewBox={`0 0 ${ISO.width} ${ISO.height}`}
        role="img"
        aria-label={`IsoFLOP curves from the ${fit.label} fit. Your run: ${formatCount(parameters)} parameters on ${formatCount(tokens)} tokens, ${formatFlops(compute)} FLOPs, predicted loss ${loss.toFixed(3)}. The fit's optimum at this compute is ${formatCount(optimum.parameters)} parameters on ${formatCount(optimum.tokens)} tokens, loss ${optimum.loss.toFixed(3)}. Twenty tokens per parameter would be ${formatCount(rule.parameters)} parameters, loss ${rule.loss.toFixed(3)}. No curve reaches the floor E = ${fit.E}.`}
      >
        <defs>
          <clipPath id="sl-iso-plot">
            <rect x={ISO.left} y={ISO.top} width={plotRight - ISO.left} height={plotBottom - ISO.top} />
          </clipPath>
          <pattern id="sl-iso-floor" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line className="sl-floor-hatch" x1="0" x2="0" y1="0" y2="6" />
          </pattern>
        </defs>
        {xTicks.map((tick) => (
          <g key={tick.label}>
            <line
              className="sl-grid"
              x1={isoX(tick.value)}
              x2={isoX(tick.value)}
              y1={ISO.top}
              y2={plotBottom}
            />
            <text className="sl-tick" x={isoX(tick.value)} y={plotBottom + 15} textAnchor="middle">
              {tick.label}
            </text>
          </g>
        ))}
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="sl-grid" x1={ISO.left} x2={plotRight} y1={isoY(tick)} y2={isoY(tick)} />
            <text className="sl-tick" x={ISO.left - 7} y={isoY(tick) + 3.5} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <rect
          className="sl-floor-zone"
          x={ISO.left}
          y={floorY}
          width={plotRight - ISO.left}
          height={Math.max(0, plotBottom - floorY)}
          fill="url(#sl-iso-floor)"
        />
        <line className="sl-floor" x1={ISO.left} x2={plotRight} y1={floorY} y2={floorY} />
        <line className="sl-axis" x1={ISO.left} x2={plotRight} y1={plotBottom} y2={plotBottom} />
        <line className="sl-axis" x1={ISO.left} x2={ISO.left} y1={ISO.top} y2={plotBottom} />
        <text className="sl-axis-label" x={(ISO.left + plotRight) / 2} y={ISO.height - 6} textAnchor="middle">
          parameters N (log scale)
        </text>
        <text className="sl-axis-label sl-axis-label--hint" x={ISO.left} y={ISO.height - 6}>
          {wide ? "← smaller model, more tokens" : "← more tokens"}
        </text>
        <text className="sl-axis-label sl-axis-label--hint" x={plotRight} y={ISO.height - 6} textAnchor="end">
          {wide ? "bigger model, fewer tokens →" : "fewer tokens →"}
        </text>
        <text
          className="sl-axis-label"
          transform={`translate(13 ${(ISO.top + plotBottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          predicted loss (nats/token)
        </text>

        <g clipPath="url(#sl-iso-plot)">
          {fixed.family.map((curve) => (
            <g key={curve.exponent} className="sl-family">
              <path d={curvePath(curve.points)} />
              {inIsoX(curve.best.parameters) && (
                <circle cx={isoX(curve.best.parameters)} cy={isoY(curve.best.loss)} r={2.6} />
              )}
            </g>
          ))}
          <path className="sl-frontier" d={curvePath(fixed.frontier)} />
          <path className="sl-rule-locus" d={curvePath(fixed.ruleLocus)} />
          <path className="sl-user-curve" d={curvePath(isoFlopCurve(fit, compute, ISO_X.min, ISO_X.max, 160))} />

          <text className="sl-note sl-note--floor" x={ISO.left + 10} y={floorY - 6}>
            {wide ? `floor E = ${fit.E} · no budget gets below this line` : `floor E = ${fit.E}`}
          </text>
          {wide && fixed.caption.lines > 0 && (
            <g className="sl-note sl-note--frontier">
              <line
                x1={frontierLabel.x + 138}
                y1={frontierLabel.y - 4}
                x2={isoX(frontierTip.parameters) - 4}
                y2={isoY(frontierTip.loss) + 5}
              />
              <text x={frontierLabel.x} y={frontierLabel.y}>
                {fixed.caption.lines > 1 ? "compute-optimal frontier:" : "compute-optimal frontier"}
              </text>
              {fixed.caption.lines > 1 && (
                <text x={frontierLabel.x} y={frontierLabel.y + 13}>
                  the bottom of every curve
                </text>
              )}
            </g>
          )}

          {fixed.placed.map((entry) => {
            const { box, reach } = entry.spot;
            const tip = {
              x: Math.min(Math.max(entry.anchor.x, box.x + 6), box.x + box.width - 6),
              y: box.y > entry.anchor.y ? box.y + 1 : box.y + box.height - 1,
            };
            return (
              <g key={entry.id} className="sl-reference">
                {reach > 12 && (
                  <line
                    className="sl-reference__leader"
                    x1={entry.anchor.x}
                    y1={entry.anchor.y + (tip.y > entry.anchor.y ? 6 : -6)}
                    x2={tip.x}
                    y2={tip.y}
                  />
                )}
                <path
                  d={`M${entry.anchor.x - 4} ${entry.anchor.y - 4} l8 8 M${entry.anchor.x - 4} ${entry.anchor.y + 4} l8 -8`}
                />
                <text x={box.x + 2} y={box.y + 10}>
                  {entry.name}
                </text>
              </g>
            );
          })}

          {showGap && (
            <g className="sl-gap">
              <line className="sl-gap__level" x1={best.x} x2={user.x} y1={best.y} y2={best.y} />
              <line className="sl-gap__drop" x1={user.x} x2={user.x} y1={Math.max(user.y, ISO.top)} y2={best.y} />
              <text
                x={user.x + (userRight ? -7 : 7)}
                y={(Math.max(user.y, ISO.top) + best.y) / 2 + 4}
                textAnchor={userRight ? "end" : "start"}
              >
                {`gap ${gap.toFixed(3)}`}
              </text>
            </g>
          )}
          {inIsoX(rule.parameters) && (
            <g className="sl-mark">
              <rect className="sl-rule-mark" x={twenty.x - 4.5} y={twenty.y - 4.5} width={9} height={9} />
              <text x={twenty.x + (bestLeft ? 8 : -8)} y={twenty.y + 17} textAnchor={bestLeft ? "start" : "end"}>
                20:1
              </text>
            </g>
          )}
          {inIsoX(optimum.parameters) && (
            <g className="sl-mark">
              <path
                className="sl-optimum-mark"
                d={`M${best.x} ${best.y - 6.5} l6.5 6.5 l-6.5 6.5 l-6.5 -6.5 Z`}
              />
              <text x={best.x + (bestLeft ? -8 : 8)} y={best.y + 17} textAnchor={bestLeft ? "end" : "start"}>
                best split
              </text>
            </g>
          )}
          <circle className="sl-user-halo" cx={user.x} cy={user.y} r={10} />
          <circle className="sl-user-mark" cx={user.x} cy={user.y} r={5.5} />
          {userVisible && (
            <g className="sl-mark sl-mark--user">
              <text
                x={user.x + (userRight ? 9 : -9)}
                y={user.y - 11 < ISO.top + 10 ? user.y + 20 : user.y - 11}
                textAnchor={userRight ? "start" : "end"}
              >
                your run
              </text>
            </g>
          )}
        </g>
        {fixed.labels.map((label) => (
          <text key={label.exponent} className="sl-curve-label" x={label.x} y={label.y} textAnchor={label.anchor}>
            {`10${superscript(label.exponent)}`}
          </text>
        ))}
        <text className="sl-curve-label sl-curve-label--corner" x={ISO.left + 4} y={ISO.top - 7}>
          {cornerRoom ? "each thin curve: one budget C, in FLOPs" : "C in FLOPs:"}
        </text>
      </svg>
    </div>
  );
}

/* ---------- frontier power-law chart ---------- */

const PL_X = { min: 18, max: 26 };
const PL_Y = { min: Math.log10(0.05), max: Math.log10(3) };

function FrontierChart({ fit, parameters, tokens }: { fit: ScalingFit; parameters: number; tokens: number }) {
  const [frame, measured] = useMeasuredWidth(302);
  const width = Math.max(260, Math.round(measured));
  const PL = { width, height: Math.round(Math.min(330, Math.max(230, width * 0.8))), left: 50, right: 12, top: 12, bottom: 38 };
  const plX = (c: number) =>
    PL.left + ((Math.log10(c) - PL_X.min) / (PL_X.max - PL_X.min)) * (PL.width - PL.left - PL.right);
  const plY = (reducibleLoss: number) =>
    PL.top +
    ((PL_Y.max - Math.log10(Math.max(1e-6, reducibleLoss))) / (PL_Y.max - PL_Y.min)) *
      (PL.height - PL.top - PL.bottom);
  const compute = trainingFlops(parameters, tokens);
  const reducible = lossTerms(fit, parameters, tokens).total - fit.E;
  const exponent = frontierLossExponent(fit);
  const line = Array.from({ length: 41 }, (_, index) => {
    const c = Math.pow(10, PL_X.min + (index * (PL_X.max - PL_X.min)) / 40);
    return { c, value: optimalAllocation(fit, c).loss - fit.E };
  });
  const rule = Array.from({ length: 41 }, (_, index) => {
    const c = Math.pow(10, PL_X.min + (index * (PL_X.max - PL_X.min)) / 40);
    return { c, value: ratioAllocation(fit, c).loss - fit.E };
  });
  const path = (points: ReadonlyArray<{ c: number; value: number }>) =>
    points
      .map((point, index) => `${index ? "L" : "M"}${plX(point.c).toFixed(1)} ${plY(point.value).toFixed(1)}`)
      .join(" ");
  const plotRight = PL.width - PL.right;
  const plotBottom = PL.height - PL.bottom;
  const onFrontier = optimalAllocation(fit, compute).loss - fit.E;
  const keeps = Math.pow(10, -exponent);

  // One tenfold step drawn on the line: a decade across, and the drop it buys. It moves out of the
  // way when the reader's own dot is nearby.
  const logC = Math.log10(compute);
  const stepFrom = logC > 21.4 && logC < 24.9 ? 19 : 23;
  const stepA = { x: plX(Math.pow(10, stepFrom)), y: plY(optimalAllocation(fit, Math.pow(10, stepFrom)).loss - fit.E) };
  const stepB = {
    x: plX(Math.pow(10, stepFrom + 1)),
    y: plY(optimalAllocation(fit, Math.pow(10, stepFrom + 1)).loss - fit.E),
  };
  const user = { x: plX(compute), y: plY(reducible) };
  const userLabelRight = user.x < plotRight - 70;

  return (
    <div ref={frame} className="sl-power-frame">
      <svg
        className="sl-power"
        width={PL.width}
        height={PL.height}
        viewBox={`0 0 ${PL.width} ${PL.height}`}
        role="img"
        aria-label={`Log-log plot of reducible loss, predicted loss minus the floor ${fit.E}, against compute. The frontier is a straight line with slope minus ${exponent.toFixed(3)}, so ten times the compute keeps ${(keeps * 100).toFixed(0)} percent of it. Your run sits at ${reducible.toFixed(3)} against ${onFrontier.toFixed(3)} on the frontier at the same compute.`}
      >
        {[18, 20, 22, 24, 26].map((tick) => (
          <g key={tick}>
            <line
              className="sl-grid"
              x1={plX(Math.pow(10, tick))}
              x2={plX(Math.pow(10, tick))}
              y1={PL.top}
              y2={plotBottom}
            />
            <text className="sl-tick" x={plX(Math.pow(10, tick))} y={plotBottom + 14} textAnchor="middle">
              {`10${superscript(tick)}`}
            </text>
          </g>
        ))}
        {[0.1, 0.2, 0.5, 1, 2].map((tick) => (
          <g key={tick}>
            <line className="sl-grid" x1={PL.left} x2={plotRight} y1={plY(tick)} y2={plY(tick)} />
            <text className="sl-tick" x={PL.left - 6} y={plY(tick) + 3.5} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        <line className="sl-axis" x1={PL.left} x2={plotRight} y1={plotBottom} y2={plotBottom} />
        <line className="sl-axis" x1={PL.left} x2={PL.left} y1={PL.top} y2={plotBottom} />
        <text className="sl-axis-label" x={(PL.left + plotRight) / 2} y={PL.height - 6} textAnchor="middle">
          training compute C (FLOPs, log scale)
        </text>
        <text
          className="sl-axis-label"
          transform={`translate(12 ${(PL.top + plotBottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {`loss above floor, L − E (log)`}
        </text>
        <path className="sl-rule-locus" d={path(rule)} />
        <path className="sl-frontier" d={path(line)} />
        <g className="sl-step">
          <path d={`M${stepA.x} ${stepA.y} H${stepB.x} V${stepB.y}`} />
          <text x={(stepA.x + stepB.x) / 2} y={stepA.y - 6} textAnchor="middle">
            10× compute
          </text>
          <text x={stepB.x + 6} y={(stepA.y + stepB.y) / 2 + 4}>
            {`keeps ${(keeps * 100).toFixed(0)}%`}
          </text>
        </g>
        <text className="sl-note sl-note--frontier" x={PL.left + 8} y={plotBottom - 22}>
          dashed: best split at each budget
        </text>
        <text className="sl-note sl-note--rule" x={PL.left + 8} y={plotBottom - 9}>
          dotted: 20 tokens per parameter
        </text>
        <line className="sl-drop" x1={user.x} x2={user.x} y1={user.y} y2={plY(onFrontier)} />
        <circle className="sl-user-mark" cx={user.x} cy={user.y} r={5} />
        <text
          className="sl-mark sl-mark--user"
          x={user.x + (userLabelRight ? 9 : -9)}
          y={Math.max(PL.top + 10, user.y - 9)}
          textAnchor={userLabelRight ? "start" : "end"}
        >
          your run
        </text>
      </svg>
    </div>
  );
}

/* ---------- loss breakdown bars ---------- */

function TermBar({
  label,
  terms,
  scale,
}: {
  label: string;
  terms: { floor: number; size: number; data: number; total: number };
  scale: number;
}) {
  const width = (value: number) => `${(value / scale) * 100}%`;
  const reducible = terms.size + terms.data;
  return (
    <div className="sl-term-row">
      <span className="sl-term-row__label">{label}</span>
      <div
        className="sl-term-bar"
        role="img"
        aria-label={`${label}: size term ${terms.size.toFixed(3)} plus data term ${terms.data.toFixed(3)} equals ${reducible.toFixed(3)} above the floor; total loss ${terms.total.toFixed(3)}.`}
      >
        <i className="sl-term sl-term--size" style={{ width: width(terms.size) }}>
          <b>{terms.size.toFixed(3)}</b>
        </i>
        <i className="sl-term sl-term--data" style={{ width: width(terms.data) }}>
          <b>{terms.data.toFixed(3)}</b>
        </i>
      </div>
      <strong>{reducible.toFixed(3)}</strong>
    </div>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const parametersB = clamp(asNumber(state, "parameters", 3.2), PARAMETER_RANGE.min, PARAMETER_RANGE.max);
  const tokensB = clamp(asNumber(state, "data", 28), TOKEN_RANGE.min, TOKEN_RANGE.max);
  const fitId: FitId = state.fit === "epoch" ? "epoch" : "hoffmann";
  const hold = state.budget === "hold";
  const fit = FITS[fitId];
  const layers = Math.round(clamp(asNumber(state, "layers", 32), LAYER_RANGE.min, LAYER_RANGE.max));
  const width = clampWidth(asNumber(state, "width", 4096));

  const parameters = parametersB * 1e9;
  const tokens = tokensB * 1e9;
  const compute = trainingFlops(parameters, tokens);
  const terms = lossTerms(fit, parameters, tokens);
  const optimum = optimalAllocation(fit, compute);
  const optimumTerms = lossTerms(fit, optimum.parameters, optimum.tokens);
  const rule = ratioAllocation(fit, compute);
  const ratio = tokens / parameters;
  const gap = terms.total - optimum.loss;
  const doubleN = terms.size * (1 - Math.pow(2, -fit.alpha));
  const doubleD = terms.data * (1 - Math.pow(2, -fit.beta));
  const exponent = frontierLossExponent(fit);
  const scale = Math.max(terms.size + terms.data, optimumTerms.size + optimumTerms.data) * 1.02;

  // Under "Hold compute" the product N·D is fixed, so each slider stops where the
  // other would leave its range instead of silently changing the budget.
  const product = parametersB * tokensB;
  const setParameters = (next: number) => {
    if (!hold) return setState({ parameters: round3(clamp(next, PARAMETER_RANGE.min, PARAMETER_RANGE.max)) });
    const low = Math.max(PARAMETER_RANGE.min, product / TOKEN_RANGE.max);
    const high = Math.min(PARAMETER_RANGE.max, product / TOKEN_RANGE.min);
    const value = clamp(next, low, high);
    setState({ parameters: round3(value), data: round3(product / value) });
  };
  const setTokens = (next: number) => {
    if (!hold) return setState({ data: round3(clamp(next, TOKEN_RANGE.min, TOKEN_RANGE.max)) });
    const low = Math.max(TOKEN_RANGE.min, product / PARAMETER_RANGE.max);
    const high = Math.min(TOKEN_RANGE.max, product / PARAMETER_RANGE.min);
    const value = clamp(next, low, high);
    setState({ data: round3(value), parameters: round3(product / value) });
  };

  // N from dimensions: 12 L d², embeddings excluded. The slider can hold 100M to 1T, so a count outside it is clamped.
  const dimensionN = denseParameters(layers, width);
  const dimensionNB = dimensionN / 1e9;
  const dimensionClamped = dimensionNB < PARAMETER_RANGE.min || dimensionNB > PARAMETER_RANGE.max;

  const verdict =
    Math.abs(doubleN - doubleD) < 0.1 * Math.max(doubleN, doubleD)
      ? "Near the optimum: doubling either resource buys about the same."
      : doubleD > doubleN
        ? "The data term dominates: the next doubling of compute buys more as tokens."
        : "The size term dominates: the next doubling of compute buys more as parameters.";

  return (
    <div className="sl-lab">
      <LabSurface label="Scaling law controls" className="sl-controls-card">
        <SurfaceHeading kicker="Two resources, one bill" title="Choose N and D; compute is their product" />
        <div className="sl-controls">
          <div className="sl-controls__sliders">
            <LogRange
              label="Parameters"
              value={parametersB}
              min={PARAMETER_RANGE.min}
              max={PARAMETER_RANGE.max}
              onChange={(value) => setParameters(value)}
              format={(value) => formatCount(value * 1e9)}
            />
            <LogRange
              label="Training tokens"
              value={tokensB}
              min={TOKEN_RANGE.min}
              max={TOKEN_RANGE.max}
              onChange={(value) => setTokens(value)}
              format={(value) => formatCount(value * 1e9)}
            />
            <SegmentedControl
              label="Fit"
              value={fitId}
              options={[
                { value: "hoffmann", label: "Hoffmann 2022" },
                { value: "epoch", label: "Epoch 2024 refit" },
              ]}
              onChange={(value) => setState({ fit: value })}
            />
          </div>
          <div className="sl-budget">
            <SegmentedControl
              label="Budget"
              value={hold ? "hold" : "free"}
              options={[
                { value: "free", label: "Free" },
                { value: "hold", label: "Hold compute" },
              ]}
              onChange={(value) => setState({ budget: value })}
            />
            <BudgetMeter parameters={parameters} tokens={tokens} hold={hold} />
            <p>
              {hold
                ? `Held at ${formatFlops(compute)} FLOPs. Moving one slider now moves the other, so your dot slides along the highlighted curve.`
                : "Each slider moves independently, so every move also changes the compute budget and jumps to a different curve."}
            </p>
            <p className="sl-budget__target">
              <span>Fit optimum here</span>
              <strong>
                {formatCount(optimum.parameters)} on {formatCount(optimum.tokens)} ({optimum.ratio.toFixed(0)}{" "}
                tok/param)
              </strong>
            </p>
          </div>
        </div>
        <div className="sl-dimensions" role="group" aria-label="N from dimensions">
          <p className="sl-dimensions__head">
            <strong>N from dimensions</strong>
            <span>A dense transformer has about 12 · L · d² weights outside its embeddings.</span>
          </p>
          <RangeControl
            label="Layers L"
            min={LAYER_RANGE.min}
            max={LAYER_RANGE.max}
            step={1}
            value={layers}
            onChange={(value) => setState({ layers: value })}
          />
          <RangeControl
            label="Width d"
            min={WIDTH_RANGE.min}
            max={WIDTH_RANGE.max}
            step={WIDTH_RANGE.step}
            value={width}
            format={(value) => value.toLocaleString("en-US")}
            onChange={(value) => setState({ width: clampWidth(value) })}
          />
          <DimensionTrack fromDimensions={dimensionN} parameters={parameters} />
          <p className="sl-dimensions__result">
            12 × {layers} × {width.toLocaleString("en-US")}² = <strong>{formatCount(dimensionN)}</strong> weights
            ({blockWeights(width).toLocaleString("en-US")} per block).{" "}
            {dimensionClamped
              ? `That is outside the Parameters slider, so the button clamps it to ${dimensionNB < PARAMETER_RANGE.min ? "100M" : "1T"}.`
              : "Embeddings are left out, so a real model's total is a little higher."}
          </p>
          <button type="button" className="sl-dimensions__use" onClick={() => setParameters(dimensionNB)}>
            Use as Parameters
          </button>
        </div>
      </LabSurface>

      <LabSurface label="IsoFLOP curves" className="scaling-dashboard sl-iso-card">
        <SurfaceHeading
          kicker={`${fit.source} · L = E + A/N^α + B/D^β`}
          title="Each curve spends one compute budget; its lowest point is the best split"
        />
        <IsoFlopChart fit={fit} parameters={parameters} tokens={tokens} />
        <ul className="sl-legend">
          <li>
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <circle className="sl-user-mark" cx="7" cy="7" r="4.5" />
            </svg>
            <span>Your run</span>
            <b>
              {formatCount(parameters)} · {formatCount(tokens)} tok · L {terms.total.toFixed(3)}
            </b>
          </li>
          <li>
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <path className="sl-optimum-mark" d="M7 1 l6 6 l-6 6 l-6 -6 Z" />
            </svg>
            <span>Fit optimum, same C</span>
            <b>
              {formatCount(optimum.parameters)} · {formatCount(optimum.tokens)} tok · L{" "}
              {optimum.loss.toFixed(3)}
            </b>
          </li>
          <li>
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <rect className="sl-rule-mark" x="2.5" y="2.5" width="9" height="9" />
            </svg>
            <span>20 tokens/param, same C</span>
            <b>
              {formatCount(rule.parameters)} · {formatCount(rule.tokens)} tok · L {rule.loss.toFixed(3)}
            </b>
          </li>
          <li>
            <svg viewBox="0 0 22 14" aria-hidden="true">
              <line className="sl-frontier" x1="1" x2="21" y1="7" y2="7" />
            </svg>
            <span>Compute-optimal frontier</span>
          </li>
          <li>
            <svg viewBox="0 0 22 14" aria-hidden="true">
              <line className="sl-rule-locus" x1="1" x2="21" y1="7" y2="7" />
            </svg>
            <span>20:1 locus</span>
          </li>
          <li>
            <svg viewBox="0 0 22 14" aria-hidden="true">
              <path className="sl-reference" d="M7 3 l8 8 M7 11 l8 -8" />
            </svg>
            <span>Real runs (70B/1.4T, 280B/300B)</span>
          </li>
        </ul>
        <div className="metric-row">
          <Metric label="Predicted loss" value={terms.total.toFixed(3)} tone="loss" />
          <Metric label="Compute C = 6ND" value={`${formatFlops(compute)} FLOPs`} />
          <Metric label="Tokens per parameter" value={ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1)} />
          <Metric
            label="Gap to optimum"
            value={`${gap.toFixed(3)} nats`}
            tone={gap > 0.05 ? "loss" : "forward"}
          />
        </div>
      </LabSurface>

      <LabSurface label="Loss breakdown" className="sl-breakdown-card">
        <SurfaceHeading kicker="Floor + size term + data term" title="Which term is holding the loss up?" />
        <div className="sl-formula" role="group" aria-label="The fit evaluated for your run">
          <code>
            L = {fit.E} + {fit.A}/N<sup>{fit.alpha}</sup> + {fit.B}/D<sup>{fit.beta}</sup>
          </code>
          <code>
            N = {sci(parameters)}, D = {sci(tokens)}
          </code>
          <code>
            = {fit.E} + <b className="sl-formula__size">{terms.size.toFixed(3)}</b> +{" "}
            <b className="sl-formula__data">{terms.data.toFixed(3)}</b> ={" "}
            <strong>{terms.total.toFixed(3)}</strong>
          </code>
        </div>
        <div className="sl-terms">
          <TermBar label="Your run" terms={terms} scale={scale} />
          <TermBar label="Optimum, same C" terms={optimumTerms} scale={scale} />
          <div className="sl-term-key" aria-hidden="true">
            <span>
              <i className="sl-term--size" />
              A/N^α, size term (solid)
            </span>
            <span>
              <i className="sl-term--data" />
              B/D^β, data term (hatched)
            </span>
            <span>Bars and totals are loss above the floor E = {fit.E}.</span>
          </div>
        </div>
        <div className="sl-doubling" role="group" aria-label="What the next doubling of each resource buys">
          {[
            { id: "size", label: "Doubling N lowers L by", value: doubleN, wins: doubleN > doubleD },
            { id: "data", label: "Doubling D lowers L by", value: doubleD, wins: doubleD > doubleN },
          ].map((row) => (
            <div key={row.id} className={`sl-doubling__row ${row.wins ? "is-better" : ""}`}>
              <span>{row.label}</span>
              <strong>{row.value.toFixed(3)}</strong>
              <div className="sl-doubling__bar" aria-hidden="true">
                <i
                  className={`sl-term--${row.id}`}
                  style={{ width: `${(row.value / Math.max(doubleN, doubleD, 1e-9)) * 100}%` }}
                />
              </div>
              <em>{row.wins ? "better buy" : ""}</em>
            </div>
          ))}
        </div>
        <p className="lab-note">
          {verdict} Either doubling doubles C. At the optimum the marginal gains match: α·(A/N^α) = β·(B/D^β).
        </p>
      </LabSurface>

      <LabSurface label="Frontier power law" className="sl-power-card">
        <SurfaceHeading
          kicker="Reducible loss on log-log axes"
          title="Subtract the floor and the frontier is a straight line"
        />
        <FrontierChart fit={fit} parameters={parameters} tokens={tokens} />
        <div className="metric-row">
          <Metric label="Slope" value={`−${exponent.toFixed(3)}`} />
          <Metric
            label="10× compute keeps"
            value={`${(Math.pow(10, -exponent) * 100).toFixed(0)}% of L − E`}
          />
        </div>
        <p className="lab-note">
          Dashed: the fit&apos;s optimum at every budget, where N* ∝ C
          <sup>{parameterExponent(fit).toFixed(2)}</sup> and D* ∝ C
          <sup>{(1 - parameterExponent(fit)).toFixed(2)}</sup>. Dotted: 20 tokens per parameter. The two
          nearly overlap because each isoFLOP curve is flat at the bottom. Your dot sits on or above the
          dashed line.
        </p>
      </LabSurface>
    </div>
  );
}
