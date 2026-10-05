import { useId, useMemo, type CSSProperties } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  BATCH_SIZES,
  CHECKPOINTS,
  checkpointAt,
  DEGREE_MAX,
  DEGREE_MIN,
  gradientNoise,
  LEARNING_RATES,
  makeDataset,
  NOISE_VARIANCE,
  OVERFIT_RATIO,
  predict,
  statusAt,
  train,
  trueFunction,
  UNDERFIT_RATIO,
  WEIGHT_DECAYS,
  type Status,
  type TrainRun,
} from "./sim";

const SEEDS = [1, 2, 3, 4] as const;
const LAST_INDEX = CHECKPOINTS.length - 1;

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) => {
  const value = state[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
};
const clampInt = (value: number, low: number, high: number) => Math.min(high, Math.max(low, Math.round(value)));

function formatLoss(value: number) {
  if (!Number.isFinite(value)) return "overflow";
  if (value >= 1000) return value.toExponential(1).replace("e+", "e");
  if (value >= 10) return value.toFixed(1);
  if (value >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
const formatSmall = (value: number) =>
  !Number.isFinite(value) ? "overflow" : value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : value.toFixed(3);
const formatEpoch = (epoch: number) => epoch.toLocaleString("en-US");
const formatDecay = (value: number) => (value === 0 ? "0 (off)" : `${value}`);

const STATUS_TEXT: Record<Status, string> = {
  diverged: "diverged",
  underfitting: "underfitting",
  overfitting: "overfitting",
  fitting: "fitting",
};

/** A slider over a fixed list of values, announced by value rather than by position. */
function IndexedRange({
  label,
  index,
  count,
  describe,
  onChange,
}: {
  label: string;
  index: number;
  count: number;
  describe: (index: number) => string;
  onChange: (index: number) => void;
}) {
  const progress = (index / Math.max(1, count - 1)) * 100;
  return (
    <label className="range-control">
      <span>
        {label}
        <output>{describe(index)}</output>
      </span>
      <input
        type="range"
        aria-label={label}
        aria-valuetext={describe(index)}
        min={0}
        max={count - 1}
        step={1}
        value={index}
        style={{ "--range-progress": `${progress}%` } as CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

/* Shared log-loss axis for the two loss charts: 0.003 to 30. */
const LOSS_LOG_MIN = Math.log10(0.003);
const LOSS_LOG_MAX = Math.log10(30);
const LOSS_TICKS = [0.01, 0.1, 1, 10];
const logY = (loss: number, top: number, bottom: number) => {
  const log = Number.isFinite(loss) && loss > 0 ? Math.log10(loss) : LOSS_LOG_MAX;
  const bounded = Math.min(LOSS_LOG_MAX, Math.max(LOSS_LOG_MIN, log));
  return bottom - ((bounded - LOSS_LOG_MIN) / (LOSS_LOG_MAX - LOSS_LOG_MIN)) * (bottom - top);
};

/* Training curves geometry: epoch 0 sits at the axis, then 1 → 10,000 on a log scale. */
const CURVE = { left: 46, right: 466, top: 14, bottom: 206, width: 480, height: 236 };
const EPOCH_GAP = 16;
const ex = (epoch: number) =>
  epoch <= 0
    ? CURVE.left
    : CURVE.left + EPOCH_GAP + (Math.log10(epoch) / 4) * (CURVE.right - CURVE.left - EPOCH_GAP);
const cy = (loss: number) => logY(loss, CURVE.top, CURVE.bottom);

/* Fitted function geometry. */
const FIT = { left: 34, right: 466, top: 12, bottom: 196, width: 480, height: 222 };
const FIT_Y = 2;
const fx = (x: number) => FIT.left + ((x + 1) / 2) * (FIT.right - FIT.left);
const fy = (y: number) => {
  const bounded = Math.min(FIT_Y + 0.6, Math.max(-FIT_Y - 0.6, y));
  return FIT.bottom - ((bounded + FIT_Y) / (2 * FIT_Y)) * (FIT.bottom - FIT.top);
};

/* Capacity sweep geometry. */
const SWEEP = { left: 46, right: 466, top: 14, bottom: 178, width: 480, height: 206 };
const dx = (degree: number) =>
  SWEEP.left + 10 + ((degree - DEGREE_MIN) / (DEGREE_MAX - DEGREE_MIN)) * (SWEEP.right - SWEEP.left - 20);
const dy = (loss: number) => logY(loss, SWEEP.top, SWEEP.bottom);

/* Split strip geometry. */
const STRIP = { left: 12, right: 468, width: 480, height: 44 };

const toPoints = (points: ReadonlyArray<{ x: number; y: number }>) =>
  points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

function lossAtEpoch(run: TrainRun, epoch: number) {
  const index = checkpointAt(run, epoch);
  return run.checkpoints[index];
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const ids = useId().replace(/:/g, "");
  const degree = clampInt(asNumber(state, "degree", 5), DEGREE_MIN, DEGREE_MAX);
  const epochIndex = clampInt(asNumber(state, "epochIndex", CHECKPOINTS.indexOf(100)), 0, LAST_INDEX);
  const learningRateIndex = clampInt(asNumber(state, "learningRateIndex", 3), 0, LEARNING_RATES.length - 1);
  const weightDecayIndex = clampInt(asNumber(state, "weightDecayIndex", 0), 0, WEIGHT_DECAYS.length - 1);
  const requestedBatch = asNumber(state, "batchSize", 16);
  const batchSize = (BATCH_SIZES as readonly number[]).includes(requestedBatch) ? requestedBatch : 16;
  const requestedSeed = asNumber(state, "seed", 1);
  const seed = (SEEDS as readonly number[]).includes(requestedSeed) ? requestedSeed : 1;

  const learningRate = LEARNING_RATES[learningRateIndex];
  const weightDecay = WEIGHT_DECAYS[weightDecayIndex];
  const epoch = CHECKPOINTS[epochIndex];

  const dataset = useMemo(() => makeDataset(seed), [seed]);
  // Every degree trained with the current optimizer settings: the capacity sweep
  // reads all of them and the main chart reads the selected one.
  const sweep = useMemo(
    () =>
      Array.from({ length: DEGREE_MAX - DEGREE_MIN + 1 }, (_, offset) =>
        train({ degree: DEGREE_MIN + offset, learningRate, weightDecay, batchSize, seed }),
      ),
    [learningRate, weightDecay, batchSize, seed],
  );
  const run = sweep[degree - DEGREE_MIN];
  const shownIndex = checkpointAt(run, epoch);
  const shown = run.checkpoints[shownIndex];
  const best = run.checkpoints[run.bestIndex];
  const status = statusAt(run, epoch);
  const diverged = run.divergedAt !== null && epoch >= run.divergedAt;
  const gap = shown.validation - shown.train;
  const noise = useMemo(() => gradientNoise(shown.theta, batchSize, seed), [shown, batchSize, seed]);
  const weightNorm = Math.sqrt(shown.theta.reduce((sum, value) => sum + value * value, 0));

  // Training curves.
  const trainLine = run.checkpoints.map((checkpoint) => ({ x: ex(checkpoint.epoch), y: cy(checkpoint.train) }));
  const validationLine = run.checkpoints.map((checkpoint) => ({ x: ex(checkpoint.epoch), y: cy(checkpoint.validation) }));
  const gapArea = [...trainLine, ...[...validationLine].reverse()];

  // Fitted function.
  const modelCurve = diverged
    ? []
    : Array.from({ length: 161 }, (_, index) => {
        const x = -1 + index / 80;
        return { x: fx(x), y: fy(predict(shown.theta, x)), raw: predict(shown.theta, x) };
      });
  const peak = modelCurve.reduce((max, point) => Math.max(max, Math.abs(point.raw)), 0);
  const trueCurve = Array.from({ length: 161 }, (_, index) => {
    const x = -1 + index / 80;
    return { x: fx(x), y: fy(trueFunction(x)) };
  });

  // Capacity sweep at the current epoch.
  const sweepRows = sweep.map((entry) => {
    const point = lossAtEpoch(entry, epoch);
    const gone = entry.divergedAt !== null && epoch >= entry.divergedAt;
    return { degree: entry.degree, train: point.train, validation: point.validation, gone };
  });
  const bestDegree = sweepRows
    .filter((row) => !row.gone)
    .reduce<(typeof sweepRows)[number] | null>(
      (winner, row) => (winner === null || row.validation < winner.validation ? row : winner),
      null,
    );

  // Split strip: all 32 samples in x order, marked by split.
  const stripSamples = [
    ...dataset.train.map((sample) => ({ ...sample, split: "train" as const })),
    ...dataset.validation.map((sample) => ({ ...sample, split: "validation" as const })),
  ].sort((a, b) => a.x - b.x);

  const stopEarly = () => {
    const index = CHECKPOINTS.indexOf(best.epoch);
    setState({ epochIndex: index });
    narrate(
      `Moved to epoch ${formatEpoch(best.epoch)}, the lowest validation loss in this run: ${formatLoss(best.validation)}.`,
    );
  };

  const statusNote =
    status === "diverged"
      ? `Diverged at epoch ${formatEpoch(run.divergedAt ?? 0)}: learning rate ${learningRate} with batch ${batchSize} is too large for degree ${degree}, so each update overshoots and the loss explodes.`
      : status === "overfitting"
        ? `Validation is ${(shown.validation / best.validation).toFixed(1)}× its best (${formatLoss(best.validation)} at epoch ${formatEpoch(best.epoch)}) while training loss kept falling. The extra fit is noise.`
        : status === "underfitting"
          ? `Training loss ${formatLoss(shown.train)} is still more than ${UNDERFIT_RATIO}× the noise variance ${NOISE_VARIANCE.toFixed(4)}: the model has not captured the pattern yet${degree <= 2 ? ", and at this degree it cannot" : ""}.`
          : `Validation is within ${OVERFIT_RATIO}× of its best so far (${formatLoss(best.validation)} at epoch ${formatEpoch(best.epoch)}).`;

  return (
    <div className="td-shell">
      <div className="td-lab">
        <LabSurface label="Training curves" className="td-curve-card">
          <SurfaceHeading
            kicker={`Degree ${degree} · η ${learningRate} · batch ${batchSize} · λ ${weightDecay}`}
            title="Training and validation loss from one real run"
            aside={<span className={`td-badge td-badge--${status}`}>{STATUS_TEXT[status]}</span>}
          />
          <svg
            className="td-chart"
            viewBox={`0 0 ${CURVE.width} ${CURVE.height}`}
            role="img"
            aria-label={`Training and validation loss by epoch for degree ${degree}. At epoch ${formatEpoch(epoch)}, training ${formatLoss(shown.train)}, validation ${formatLoss(shown.validation)}. Lowest validation ${formatLoss(best.validation)} at epoch ${formatEpoch(best.epoch)}. Status ${status}.`}
          >
            <defs>
              <clipPath id={`${ids}-curve`}>
                <rect x={CURVE.left} y={CURVE.top - 2} width={CURVE.right - CURVE.left + 2} height={CURVE.bottom - CURVE.top + 2} />
              </clipPath>
            </defs>
            {LOSS_TICKS.map((tick) => (
              <g key={tick}>
                <line className="td-grid" x1={CURVE.left} x2={CURVE.right} y1={cy(tick)} y2={cy(tick)} />
                <text className="td-tick" x={CURVE.left - 6} y={cy(tick) + 3} textAnchor="end">
                  {tick}
                </text>
              </g>
            ))}
            {[0, 1, 10, 100, 1000, 10000].map((tick) => (
              <text key={tick} className="td-tick" x={ex(tick)} y={CURVE.bottom + 14} textAnchor="middle">
                {tick >= 1000 ? `${tick / 1000}k` : tick}
              </text>
            ))}
            <g clipPath={`url(#${ids}-curve)`}>
              <polygon className="td-gap-area" points={toPoints(gapArea)} />
              <line className="td-floor" x1={CURVE.left} x2={CURVE.right} y1={cy(NOISE_VARIANCE)} y2={cy(NOISE_VARIANCE)} />
              <polyline className="td-line td-line--train" points={toPoints(trainLine)} />
              <polyline className="td-line td-line--validation" points={toPoints(validationLine)} />
            </g>
            <text className="td-floor-label" x={CURVE.right - 2} y={cy(NOISE_VARIANCE) + 12} textAnchor="end">
              noise variance {NOISE_VARIANCE.toFixed(4)}
            </text>
            <line className="td-epoch-marker" x1={ex(epoch)} x2={ex(epoch)} y1={CURVE.top} y2={CURVE.bottom} />
            <circle className="td-best" cx={ex(best.epoch)} cy={cy(best.validation)} r="6.5" />
            <text
              className="td-best-label"
              x={ex(best.epoch)}
              y={cy(best.validation) + 18}
              textAnchor={ex(best.epoch) > CURVE.right - 90 ? "end" : "middle"}
            >
              lowest validation
            </text>
            {!diverged && (
              <>
                <circle className="td-dot td-dot--train" cx={ex(shown.epoch)} cy={cy(shown.train)} r="4.5" />
                <circle className="td-dot td-dot--validation" cx={ex(shown.epoch)} cy={cy(shown.validation)} r="4.5" />
              </>
            )}
            {run.divergedAt !== null && (
              <text className="td-warning" x={CURVE.right - 4} y={CURVE.bottom - 8} textAnchor="end">
                diverged at epoch {formatEpoch(run.divergedAt)}: loss above 10⁶
              </text>
            )}
            <line className="plot-axis" x1={CURVE.left} x2={CURVE.right} y1={CURVE.bottom} y2={CURVE.bottom} />
            <line className="plot-axis" x1={CURVE.left} x2={CURVE.left} y1={CURVE.top} y2={CURVE.bottom} />
            <text className="td-axis-title" x={CURVE.right} y={CURVE.height - 2} textAnchor="end">
              epoch, log scale →
            </text>
            <text className="td-axis-title" x={CURVE.left + 4} y={CURVE.top + 1} dominantBaseline="hanging">
              ↑ mean squared error, log scale
            </text>
          </svg>
          <div className="td-legend" aria-hidden="true">
            <span><i className="td-key td-key--train" />training (16 fitted points)</span>
            <span><i className="td-key td-key--validation" />validation (16 held-out points)</span>
            <span><i className="td-key td-key--gap" />gap</span>
            <span><i className="td-key td-key--floor" />noise variance σ²</span>
          </div>
          <div className="metric-row">
            <Metric label={`Training · epoch ${formatEpoch(shown.epoch)}`} value={formatLoss(shown.train)} tone="forward" />
            <Metric label="Validation" value={formatLoss(shown.validation)} tone="loss" />
            <Metric label="Gap (val − train)" value={diverged ? "—" : formatLoss(gap)} />
            <Metric label="Lowest validation" value={`${formatLoss(best.validation)} @ ${formatEpoch(best.epoch)}`} />
          </div>
          <p className="lab-note">{statusNote}</p>
        </LabSurface>

        <LabSurface label="Training dynamics controls" className="td-controls-card">
          <div className="td-split">
            <span className="td-split__title">
              32 samples → <b>16 train</b> · <b>16 validation</b>
            </span>
            <svg
              viewBox={`0 0 ${STRIP.width} ${STRIP.height}`}
              role="img"
              aria-label="Thirty-two samples along x. Alternate samples go to training (filled) and validation (hollow)."
            >
              <line className="td-strip-axis" x1={STRIP.left} x2={STRIP.right} y1="22" y2="22" />
              {stripSamples.map((sample) => (
                <circle
                  key={`${sample.split}-${sample.x}`}
                  className={`td-strip-dot td-strip-dot--${sample.split}`}
                  cx={STRIP.left + ((sample.x + 1) / 2) * (STRIP.right - STRIP.left)}
                  cy={sample.split === "train" ? 15 : 29}
                  r="4.2"
                />
              ))}
            </svg>
            <span className="td-split__note">Filled trains the model. Hollow only scores it. No test split is drawn.</span>
          </div>
          <IndexedRange
            label="Epoch"
            index={epochIndex}
            count={CHECKPOINTS.length}
            describe={(index) => formatEpoch(CHECKPOINTS[index])}
            onChange={(index) => setState({ epochIndex: index })}
          />
          <button type="button" className="quiet-action td-stop-action" onClick={stopEarly}>
            Stop at lowest validation (epoch {formatEpoch(best.epoch)})
          </button>
          <RangeControl
            label="Polynomial degree"
            min={DEGREE_MIN}
            max={DEGREE_MAX}
            step={1}
            value={degree}
            format={(value) => `${value} · ${value + 1} weights`}
            onChange={(value) => setState({ degree: value })}
          />
          <IndexedRange
            label="Weight decay"
            index={weightDecayIndex}
            count={WEIGHT_DECAYS.length}
            describe={(index) => `λ ${formatDecay(WEIGHT_DECAYS[index])}`}
            onChange={(index) => setState({ weightDecayIndex: index })}
          />
          <IndexedRange
            label="Learning rate"
            index={learningRateIndex}
            count={LEARNING_RATES.length}
            describe={(index) => `η ${LEARNING_RATES[index]}`}
            onChange={(index) => setState({ learningRateIndex: index })}
          />
          <SegmentedControl
            label="Batch size"
            value={String(batchSize)}
            options={BATCH_SIZES.map((size) => ({ value: String(size), label: size === 16 ? "16 (full)" : String(size) }))}
            onChange={(value) => setState({ batchSize: Number(value) })}
          />
          <SegmentedControl
            label="Noise draw"
            value={String(seed)}
            options={SEEDS.map((value) => ({ value: String(value), label: `seed ${value}` }))}
            onChange={(value) => setState({ seed: Number(value) })}
          />
          <div className="metric-row">
            <Metric label="Full-batch |∇L|" value={diverged ? "—" : formatSmall(noise.full)} tone="gradient" />
            <Metric label="Minibatch noise" value={diverged ? "—" : batchSize === 16 ? "0 (full)" : formatSmall(noise.noise)} tone="gradient" />
          </div>
        </LabSurface>

        <LabSurface label="Fitted function" className="td-fit-card">
          <SurfaceHeading
            kicker={`Degree ${degree} at epoch ${formatEpoch(shown.epoch)}`}
            title="What the model has learned so far"
          />
          <svg
            className="td-chart"
            viewBox={`0 0 ${FIT.width} ${FIT.height}`}
            role="img"
            aria-label={`Fitted curve at epoch ${formatEpoch(shown.epoch)} with sixteen training and sixteen validation points and the true curve. Weight size ${weightNorm.toFixed(2)}.`}
          >
            <defs>
              <clipPath id={`${ids}-fit`}>
                <rect x={FIT.left} y={FIT.top} width={FIT.right - FIT.left} height={FIT.bottom - FIT.top} />
              </clipPath>
            </defs>
            {[-2, -1, 0, 1, 2].map((tick) => (
              <g key={tick}>
                <line className="td-grid" x1={FIT.left} x2={FIT.right} y1={fy(tick)} y2={fy(tick)} />
                <text className="td-tick" x={FIT.left - 6} y={fy(tick) + 3} textAnchor="end">
                  {tick < 0 ? `−${-tick}` : tick}
                </text>
              </g>
            ))}
            {[-1, -0.5, 0, 0.5, 1].map((tick) => (
              <text key={tick} className="td-tick" x={fx(tick)} y={FIT.bottom + 14} textAnchor="middle">
                {tick < 0 ? `−${-tick}` : tick}
              </text>
            ))}
            <g clipPath={`url(#${ids}-fit)`}>
              <polyline className="td-true-curve" points={toPoints(trueCurve)} />
              {modelCurve.length > 0 && <polyline className="td-model-curve" points={toPoints(modelCurve)} />}
            </g>
            {dataset.validation.map((sample) => (
              <circle key={`v-${sample.x}`} className="td-point td-point--validation" cx={fx(sample.x)} cy={fy(sample.y)} r="4.2" />
            ))}
            {dataset.train.map((sample) => (
              <circle key={`t-${sample.x}`} className="td-point td-point--train" cx={fx(sample.x)} cy={fy(sample.y)} r="4.2" />
            ))}
            {(peak > FIT_Y || diverged) && (
              <text className="td-warning" x={FIT.left + 8} y={FIT.top + 12}>
                {diverged ? "run diverged: no curve to draw" : `curve leaves the plot: peak |ŷ| = ${formatLoss(peak)}`}
              </text>
            )}
            <line className="plot-axis" x1={FIT.left} x2={FIT.right} y1={FIT.bottom} y2={FIT.bottom} />
            <line className="plot-axis" x1={FIT.left} x2={FIT.left} y1={FIT.top} y2={FIT.bottom} />
            <text className="td-axis-title" x={FIT.right} y={FIT.height - 2} textAnchor="end">
              x →
            </text>
          </svg>
          <div className="td-legend" aria-hidden="true">
            <span><i className="td-key td-key--model" />model ŷ</span>
            <span><i className="td-key td-key--true" />true curve (hidden from the model)</span>
            <span><i className="td-dot-key td-dot-key--train" />train</span>
            <span><i className="td-dot-key td-dot-key--validation" />validation</span>
          </div>
          <div className="metric-row">
            <Metric label="Weights" value={`${degree + 1}`} />
            <Metric label="Weight size ‖θ‖" value={diverged ? "overflow" : weightNorm >= 100 ? weightNorm.toFixed(0) : weightNorm.toFixed(2)} tone="gradient" />
          </div>
        </LabSurface>

        <LabSurface label="Capacity sweep" className="td-sweep-card">
          <SurfaceHeading
            kicker={`Every degree, same settings, epoch ${formatEpoch(epoch)}`}
            title="Loss against capacity at this moment of training"
          />
          <svg
            className="td-chart"
            viewBox={`0 0 ${SWEEP.width} ${SWEEP.height}`}
            role="img"
            aria-label={`Loss by polynomial degree at epoch ${formatEpoch(epoch)}. ${bestDegree ? `Lowest validation at degree ${bestDegree.degree}: ${formatLoss(bestDegree.validation)}.` : "Every run diverged."}`}
          >
            {LOSS_TICKS.map((tick) => (
              <g key={tick}>
                <line className="td-grid" x1={SWEEP.left} x2={SWEEP.right} y1={dy(tick)} y2={dy(tick)} />
                <text className="td-tick" x={SWEEP.left - 6} y={dy(tick) + 3} textAnchor="end">
                  {tick}
                </text>
              </g>
            ))}
            <rect
              className="td-sweep-current"
              x={dx(degree) - 12}
              y={SWEEP.top}
              width="24"
              height={SWEEP.bottom - SWEEP.top}
            />
            {sweepRows.map((row) => (
              <text key={`d-${row.degree}`} className={`td-tick${row.degree === degree ? " td-tick--current" : ""}`} x={dx(row.degree)} y={SWEEP.bottom + 14} textAnchor="middle">
                {row.degree}
              </text>
            ))}
            <line className="td-floor" x1={SWEEP.left} x2={SWEEP.right} y1={dy(NOISE_VARIANCE)} y2={dy(NOISE_VARIANCE)} />
            <polyline
              className="td-line td-line--train"
              points={toPoints(sweepRows.filter((row) => !row.gone).map((row) => ({ x: dx(row.degree), y: dy(row.train) })))}
            />
            <polyline
              className="td-line td-line--validation"
              points={toPoints(sweepRows.filter((row) => !row.gone).map((row) => ({ x: dx(row.degree), y: dy(row.validation) })))}
            />
            {sweepRows.map((row) =>
              row.gone ? (
                <text key={`x-${row.degree}`} className="td-warning" x={dx(row.degree)} y={SWEEP.top + 10} textAnchor="middle">
                  ×
                </text>
              ) : (
                <g key={`p-${row.degree}`}>
                  <circle className="td-dot td-dot--train" cx={dx(row.degree)} cy={dy(row.train)} r="3" />
                  <circle className="td-dot td-dot--validation" cx={dx(row.degree)} cy={dy(row.validation)} r="3" />
                </g>
              ),
            )}
            {bestDegree && (
              <circle className="td-best" cx={dx(bestDegree.degree)} cy={dy(bestDegree.validation)} r="6.5" />
            )}
            <line className="plot-axis" x1={SWEEP.left} x2={SWEEP.right} y1={SWEEP.bottom} y2={SWEEP.bottom} />
            <line className="plot-axis" x1={SWEEP.left} x2={SWEEP.left} y1={SWEEP.top} y2={SWEEP.bottom} />
            <text className="td-axis-title" x={SWEEP.right} y={SWEEP.height - 2} textAnchor="end">
              polynomial degree →
            </text>
          </svg>
          <div className="metric-row">
            <Metric
              label="Best degree now"
              value={bestDegree ? `${bestDegree.degree} (val ${formatLoss(bestDegree.validation)})` : "none"}
              tone="loss"
            />
            <Metric
              label="Diverged runs"
              value={`${sweepRows.filter((row) => row.gone).length} of ${sweepRows.length}`}
            />
          </div>
        </LabSurface>
      </div>
    </div>
  );
}
