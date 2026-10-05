import { useEffect, useRef, useState, type CSSProperties } from "react";
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

/* ---------- isoFLOP chart ---------- */

const ISO_X = { min: 8, max: 12 };
const ISO_Y = { min: 1.7, max: 3.8 };
const FAMILY = [19, 20, 21, 22, 23, 24, 25];

type CurvePoint = { parameters: number; loss: number };

/**
 * Chart geometry in CSS pixels. The chart is drawn at its measured width so that
 * text stays the same size on wide and narrow layouts.
 */
function isoGeometry(measured: number) {
  const width = Math.max(320, Math.round(measured));
  const box = {
    width,
    height: Math.round(Math.min(360, Math.max(250, width * 0.52))),
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
  return { box, isoX, isoY, inIsoX, curveExit, curvePath };
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

function IsoFlopChart({ fit, parameters, tokens }: { fit: ScalingFit; parameters: number; tokens: number }) {
  const [frame, measured] = useMeasuredWidth(640);
  const { box: ISO, isoX, isoY, inIsoX, curveExit, curvePath } = isoGeometry(measured);
  const compute = trainingFlops(parameters, tokens);
  const loss = lossTerms(fit, parameters, tokens).total;
  const optimum = optimalAllocation(fit, compute);
  const rule = ratioAllocation(fit, compute);
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
  const xTicks = [
    { value: 1e8, label: "100M" },
    { value: 1e9, label: "1B" },
    { value: 1e10, label: "10B" },
    { value: 1e11, label: "100B" },
    { value: 1e12, label: "1T" },
  ];
  const yTicks = [2, 2.5, 3, 3.5];
  const plotRight = ISO.width - ISO.right;
  const plotBottom = ISO.height - ISO.bottom;

  return (
    <div ref={frame} className="sl-iso-frame">
      <svg
        className="sl-iso"
        width={ISO.width}
        height={ISO.height}
        viewBox={`0 0 ${ISO.width} ${ISO.height}`}
        role="img"
        aria-label={`IsoFLOP curves from the ${fit.label} fit. Your run: ${formatCount(parameters)} parameters on ${formatCount(tokens)} tokens, ${formatFlops(compute)} FLOPs, predicted loss ${loss.toFixed(3)}. The fit's optimum at this compute is ${formatCount(optimum.parameters)} parameters on ${formatCount(optimum.tokens)} tokens, loss ${optimum.loss.toFixed(3)}. Twenty tokens per parameter would be ${formatCount(rule.parameters)} parameters, loss ${rule.loss.toFixed(3)}.`}
      >
        <defs>
          <clipPath id="sl-iso-plot">
            <rect x={ISO.left} y={ISO.top} width={plotRight - ISO.left} height={plotBottom - ISO.top} />
          </clipPath>
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
        <line className="sl-axis" x1={ISO.left} x2={plotRight} y1={plotBottom} y2={plotBottom} />
        <line className="sl-axis" x1={ISO.left} x2={ISO.left} y1={ISO.top} y2={plotBottom} />
        <text className="sl-axis-label" x={(ISO.left + plotRight) / 2} y={ISO.height - 6} textAnchor="middle">
          parameters N (log scale) · each curve holds C = 6ND fixed
        </text>
        <text
          className="sl-axis-label"
          transform={`translate(13 ${(ISO.top + plotBottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          predicted loss (nats/token)
        </text>

        <g clipPath="url(#sl-iso-plot)">
          {FAMILY.map((exponent) => {
            const points = isoFlopCurve(fit, Math.pow(10, exponent), ISO_X.min, ISO_X.max, 120);
            const best = optimalAllocation(fit, Math.pow(10, exponent));
            return (
              <g key={exponent} className="sl-family">
                <path d={curvePath(points)} />
                {inIsoX(best.parameters) && (
                  <circle cx={isoX(best.parameters)} cy={isoY(best.loss)} r={2.6} />
                )}
              </g>
            );
          })}
          <path
            className="sl-frontier"
            d={curvePath(frontier.map((point) => ({ parameters: point.parameters, loss: point.loss })))}
          />
          <path
            className="sl-rule-locus"
            d={curvePath(ruleLocus.map((point) => ({ parameters: point.parameters, loss: point.loss })))}
          />
          <path
            className="sl-user-curve"
            d={curvePath(isoFlopCurve(fit, compute, ISO_X.min, ISO_X.max, 160))}
          />
          {references.map((entry) => (
            <g key={entry.id} className="sl-reference">
              <path
                d={`M${isoX(entry.parameters) - 4} ${isoY(entry.loss) - 4} l8 8 M${isoX(entry.parameters) - 4} ${isoY(entry.loss) + 4} l8 -8`}
              />
              <text
                x={isoX(entry.parameters) + (entry.id === "gopher" ? -8 : 0)}
                y={isoY(entry.loss) + (entry.id === "gopher" ? -8 : 17)}
                textAnchor={entry.id === "gopher" ? "end" : "middle"}
              >
                {entry.name}
              </text>
            </g>
          ))}
          {inIsoX(rule.parameters) && (
            <rect
              className="sl-rule-mark"
              x={isoX(rule.parameters) - 4.5}
              y={isoY(rule.loss) - 4.5}
              width={9}
              height={9}
            />
          )}
          {inIsoX(optimum.parameters) && (
            <path
              className="sl-optimum-mark"
              d={`M${isoX(optimum.parameters)} ${isoY(optimum.loss) - 6.5} l6.5 6.5 l-6.5 6.5 l-6.5 -6.5 Z`}
            />
          )}
          <circle className="sl-user-halo" cx={isoX(parameters)} cy={isoY(loss)} r={10} />
          <circle className="sl-user-mark" cx={isoX(parameters)} cy={isoY(loss)} r={5.5} />
        </g>
        {FAMILY.map((exponent) => {
          const exit = curveExit(isoFlopCurve(fit, Math.pow(10, exponent), ISO_X.min, ISO_X.max, 120));
          return (
            <text key={exponent} className="sl-curve-label" x={exit.x} y={exit.y} textAnchor={exit.anchor}>
              {`10${superscript(exponent)}`}
            </text>
          );
        })}
        <text className="sl-curve-label sl-curve-label--corner" x={ISO.left + 4} y={ISO.top - 6}>
          C in FLOPs:
        </text>
      </svg>
    </div>
  );
}

/* ---------- frontier power-law chart ---------- */

const PL = { width: 330, height: 236, left: 48, right: 10, top: 10, bottom: 38 };
const PL_X = { min: 18, max: 26 };
const PL_Y = { min: Math.log10(0.05), max: Math.log10(3) };
const plX = (compute: number) =>
  PL.left + ((Math.log10(compute) - PL_X.min) / (PL_X.max - PL_X.min)) * (PL.width - PL.left - PL.right);
const plY = (reducible: number) =>
  PL.top +
  ((PL_Y.max - Math.log10(Math.max(1e-6, reducible))) / (PL_Y.max - PL_Y.min)) *
    (PL.height - PL.top - PL.bottom);

function FrontierChart({ fit, parameters, tokens }: { fit: ScalingFit; parameters: number; tokens: number }) {
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

  return (
    <svg
      className="sl-power"
      viewBox={`0 0 ${PL.width} ${PL.height}`}
      role="img"
      aria-label={`Log-log plot of reducible loss, predicted loss minus the floor ${fit.E}, against compute. The frontier is a straight line with slope minus ${exponent.toFixed(3)}. Your run sits at ${reducible.toFixed(3)} against ${onFrontier.toFixed(3)} on the frontier at the same compute.`}
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
        {`L − E (log)`}
      </text>
      <path className="sl-rule-locus" d={path(rule)} />
      <path className="sl-frontier" d={path(line)} />
      <line
        className="sl-drop"
        x1={plX(compute)}
        x2={plX(compute)}
        y1={plY(reducible)}
        y2={plY(onFrontier)}
      />
      <circle className="sl-user-mark" cx={plX(compute)} cy={plY(reducible)} r={5} />
    </svg>
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
        <div className="metric-row">
          <Metric
            label="Doubling N lowers L by"
            value={doubleN.toFixed(3)}
            tone={doubleN > doubleD ? "forward" : undefined}
          />
          <Metric
            label="Doubling D lowers L by"
            value={doubleD.toFixed(3)}
            tone={doubleD > doubleN ? "forward" : undefined}
          />
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
