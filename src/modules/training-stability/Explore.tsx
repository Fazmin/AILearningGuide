import { useId, useLayoutEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  trainTinyExtended,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
  type TinyExtendedRun,
} from "@app/module-sdk";
import {
  corpusPoints,
  decadeTicks,
  formatCoefficient,
  formatLoss,
  formatNumber,
  logDomain,
  logPosition,
  normalBand,
  powerOfTen,
  powerOfTwo,
  quantile,
  stepPoints,
  stepRanges,
  summarizeRun,
  values,
  type LogDomain,
  type StepPoint,
} from "./dashboard";
import {
  ADAM_RATES,
  CLIP_MAX,
  CLIP_STEP,
  describeRecipe,
  FAULTS,
  GARBAGE_COUNT,
  GARBAGE_FIRST_STEP,
  GARBAGE_TEXT,
  GROWTH_INTERVAL,
  halfWindow,
  NUMBER_FORMATS,
  originalOptions,
  PLANNED_STEPS,
  recipeOf,
  referenceLabel,
  referenceOptions,
  runOptions,
  SCALE_MAX,
  SCALE_MIN,
  Z_LOSSES,
  type FaultSettings,
} from "./faults";
import { rollbackPlan, spikeReport } from "./rollback";
import { readState } from "./state";

const UPDATE_RULE = 1e-3;
/** The clean run each fault is compared with, short enough to label a mark. */
const REFERENCE_SHORT: Record<FaultSettings["fault"], string> = {
  none: "this run",
  rate: "healthy run",
  fp16: "healthy run",
  "bad-batch": "no garbage",
  logits: "Adam at 0.1",
};
const ROLLBACK = rollbackPlan(GARBAGE_FIRST_STEP, GARBAGE_COUNT);

/** Runs a set of options once per distinct value; the key is the options themselves. */
function useRun(options: ReturnType<typeof runOptions> | null): TinyExtendedRun | null {
  // The options object is rebuilt every render, so the memo is keyed on its content.
  const key = options ? JSON.stringify(options) : "";
  return useMemo(() => (options ? trainTinyExtended(options) : null), [key]);
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const settings = readState(state);
  const { fault, clipNorm, lossScale, rollback } = settings;

  const run = useRun(runOptions(settings)) as TinyExtendedRun;
  const reference = useRun(referenceOptions(settings));
  const original = useRun(originalOptions(settings));

  const points = useMemo(() => stepPoints(run), [run]);
  const referencePoints = useMemo(() => (reference ? stepPoints(reference) : points), [reference, points]);
  const originalPoints = useMemo(() => (original ? stepPoints(original) : null), [original]);
  const summary = useMemo(() => summarizeRun(run, points), [run, points]);
  const corpus = useMemo(() => corpusPoints(run.checkpoints), [run]);
  const referenceCorpus = useMemo(() => (reference ? corpusPoints(reference.checkpoints) : null), [reference]);
  const originalCorpus = useMemo(() => (original ? corpusPoints(original.checkpoints) : null), [original]);

  const recipe = recipeOf(settings);
  const faultEntry = FAULTS.find((entry) => entry.id === fault) ?? FAULTS[0];
  const refLabel = referenceLabel(settings);
  const markers: Markers = {
    haltedAt: summary.haltedAt,
    skipped: stepRanges(points, (point) => point.skipped),
    garbage: stepRanges(points, (point) => point.faulty),
    overflow: points.filter((point) => point.overflow).map((point) => point.step),
    rollbackAt: fault === "bad-batch" && rollback ? ROLLBACK.checkpointStep : null,
  };

  const spike = fault === "bad-batch" ? spikeReport(run.checkpoints, GARBAGE_FIRST_STEP) : null;
  const originalSpike = original ? spikeReport(original.checkpoints, GARBAGE_FIRST_STEP) : null;
  const half = halfWindow(lossScale);
  const finiteNorms = values(points, "norm");
  const medianLogPartition = quantile(
    points.map((point) => point.logPartition).filter((value): value is number => value !== null),
    0.5,
  );
  const worseThanChance = summary.finalLoss !== null && summary.finalLoss > UNIFORM_CROSS_ENTROPY;
  const status = summary.haltedAt !== null ? `halted at step ${summary.haltedAt}` : worseThanChance ? "finished, worse than chance" : "finished";

  const setFault = (value: string) => {
    setState({ fault: value });
    narrate(`Fault: ${FAULTS.find((entry) => entry.id === value)?.label ?? "None"}.`);
  };

  return (
    <div className="tg-lab tg-lab--hero ts-lab">
      <LabSurface label="Fault and fixes" className="ts-controls-card">
        <SurfaceHeading
          kicker="Break one thing, then fix it"
          title="Every change retrains the same tiny model from step 1"
          aside={<span className={`tg-badge${summary.haltedAt !== null || worseThanChance ? " is-warning" : ""}`}>{status}</span>}
        />
        <div className="ts-fault">
          <SegmentedControl
            label="Fault"
            value={fault}
            options={FAULTS.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={setFault}
          />
          <p className="lab-note">
            {faultEntry.detail} Recipe: {describeRecipe(recipe)} · batch 16 · 480 planned steps · seed 9.
          </p>
        </div>
        <div className="ts-remedies" data-fault={fault}>
          <Remedies settings={settings} setState={setState} narrate={narrate} />
        </div>
      </LabSurface>

      <LabSurface label="Run monitor" className="ts-monitor">
        <SurfaceHeading
          kicker={`Factored rank 8 · six Harbor sentences · ${summary.steps} of ${PLANNED_STEPS} steps run`}
          title="Four signals from every optimizer step"
          aside={<span className="tg-badge">{recipe.optimizer} at {recipe.learningRate}</span>}
        />
        <Monitor
          fault={fault}
          points={points}
          referencePoints={referencePoints}
          corpus={corpus}
          referenceCorpus={referenceCorpus}
          originalCorpus={originalCorpus}
          markers={markers}
          summary={summary}
          clipNorm={clipNorm}
          medianLogPartition={medianLogPartition}
        />
        <Legend fault={fault} refLabel={refLabel} rollback={rollback} markers={markers} />
      </LabSurface>

      <LabSurface label="Run report" className="ts-report">
        <SurfaceHeading kicker="Measured on this run" title="What the dashboard is telling you" />
        <FinalLossStrip
          finalLoss={summary.finalLoss}
          haltedAt={summary.haltedAt}
          reference={
            reference && !reference.diverged && Number.isFinite(reference.finalLoss)
              ? { label: REFERENCE_SHORT[fault], loss: reference.finalLoss }
              : null
          }
          original={original && !original.diverged && Number.isFinite(original.finalLoss) ? original.finalLoss : null}
        />
        <div className="metric-row">
          <Metric
            label="Final loss"
            value={summary.finalLoss === null ? "halted" : `${formatLoss(summary.finalLoss)} nats`}
            tone="loss"
          />
          <Metric label="Halted at" value={summary.haltedAt === null ? "never" : `step ${summary.haltedAt}`} />
          <Metric label="Peak gradient norm" value={formatNumber(summary.peakNorm)} tone="gradient" />
        </div>
        <div className="metric-row">
          <Metric
            label="Clipped steps"
            value={(fault === "rate" || fault === "bad-batch") && clipNorm > 0 ? `${summary.clipped} of ${summary.steps}` : "clip off"}
          />
          <Metric label="Skipped steps" value={`${summary.skipped}`} />
          <Metric label="Overflow steps" value={`${summary.overflow}`} />
        </div>
        <div className="metric-row">
          <Metric label="Peak max logit" value={formatNumber(summary.peakLogit)} />
          <Metric label="Median update ÷ weight" value={formatNumber(summary.medianRatio)} tone="forward" />
          {fault === "fp16" ? (
            <Metric label="Loss scale" value={`${powerOfTwo(lossScale)} → ${powerOfTwo(Math.round(Math.log2(summary.endScale)))}`} />
          ) : fault === "bad-batch" ? (
            <Metric
              label="Spike"
              value={rollback ? "skipped" : spike ? `+${formatLoss(spike.rise)} at step ${spike.peakStep}` : "—"}
              tone="loss"
            />
          ) : (
            <Metric label="Median log-partition" value={formatNumber(medianLogPartition)} />
          )}
        </div>
        <Diagnosis
          settings={settings}
          summary={summary}
          points={points}
          spike={spike}
          originalSpike={originalSpike}
          originalFinal={original && !original.diverged ? original.finalLoss : null}
        />
      </LabSurface>

      <LabSurface label="Number formats" className="ts-formats">
        <SurfaceHeading
          kicker="Closed-form facts, not measurements"
          title="Where fp16 runs out of room"
          aside={<span className="tg-badge">loss scale {powerOfTwo(lossScale)}</span>}
        />
        <FormatLine
          flushBelow={half.flushBelow}
          overflowFrom={half.overflowFrom}
          scaleExponent={lossScale}
          norms={fault === "fp16" ? finiteNorms : []}
        />
        <div className="hp-table-wrap">
          <table className="hp-depth-table ts-format-table">
            <caption>Each format's exponent and fraction bits set its range and its precision.</caption>
            <thead>
              <tr>
                <th scope="col">Format</th>
                <th scope="col">Exponent / fraction bits</th>
                <th scope="col">Largest</th>
                <th scope="col">Smallest normal</th>
              </tr>
            </thead>
            <tbody>
              {NUMBER_FORMATS.map((format) => (
                <tr key={format.id}>
                  <th scope="row">{format.label}</th>
                  <td>
                    {format.exponentBits} / {format.fractionBits}
                  </td>
                  <td>{format.largest > 1e5 ? formatNumber(format.largest, 1) : format.largest.toLocaleString("en-US")}</td>
                  <td>{formatNumber(format.smallestNormal, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          At a loss scale of {powerOfTwo(lossScale)}, a gradient entry survives the emulated fp16 cast only between{" "}
          {formatNumber(half.flushBelow, 1)} and {formatNumber(half.overflowFrom, 2)}: smaller rounds to zero, larger becomes
          infinity.{" "}
          {fault === "fp16" && finiteNorms.length > 0
            ? `This run's gradient norm, which caps every entry, reached ${formatNumber(Math.max(...finiteNorms))}.`
            : "Pick fp16 overflow to draw this run's gradient norms against the window."}
        </p>
      </LabSurface>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Controls                                                                    */
/* -------------------------------------------------------------------------- */

function Remedies({
  settings,
  setState,
  narrate,
}: {
  settings: FaultSettings;
  setState: ModuleContext["setState"];
  narrate: ModuleContext["narrate"];
}) {
  const { fault, clipNorm, schedule, lossScale, scaling, rollback, zLoss, adamRate } = settings;
  const clip = (
    <RangeControl
      label="Gradient clip"
      min={0}
      max={CLIP_MAX}
      step={CLIP_STEP}
      value={clipNorm}
      format={(value) => (value === 0 ? "off" : `norm ≤ ${value.toFixed(2)}`)}
      onChange={(value) => {
        const next = Number((Math.round(value / CLIP_STEP) * CLIP_STEP).toFixed(2));
        setState({ clipNorm: next });
        narrate(next === 0 ? "Gradient clipping off." : `Clipping the gradient norm at ${next.toFixed(2)}.`);
      }}
    />
  );
  if (fault === "none") {
    return (
      <p className="lab-note ts-remedies__empty">
        Nothing is broken, so there is nothing to fix. Read the shaded normal range on each panel, then pick a fault.
      </p>
    );
  }
  if (fault === "rate") {
    return (
      <>
        {clip}
        <SegmentedControl
          label="Schedule"
          value={schedule}
          options={[
            { value: "constant", label: "Constant" },
            { value: "warmup-cosine", label: "Warmup + cosine" },
          ]}
          onChange={(value) => {
            setState({ schedule: value });
            narrate(value === "warmup-cosine" ? "Warmup then cosine decay." : "Constant learning rate.");
          }}
        />
        <p className="lab-note">
          Warmup + cosine starts at a tenth of the rate, climbs to the full rate over the first 15 percent of steps, then follows
          half a cosine down to 2 percent: the One cycle curve from Hyperparameters &amp; schedules.
        </p>
      </>
    );
  }
  if (fault === "fp16") {
    return (
      <>
        <RangeControl
          label="Loss scale"
          min={SCALE_MIN}
          max={SCALE_MAX}
          step={1}
          value={lossScale}
          format={(value) => `${powerOfTwo(value)} = ${(2 ** value).toLocaleString("en-US")}`}
          onChange={(value) => {
            setState({ lossScale: Math.round(value) });
            narrate(`Loss scale 2 to the ${Math.round(value)}.`);
          }}
        />
        <SegmentedControl
          label="Scaling"
          value={scaling}
          options={[
            { value: "static", label: "Static" },
            { value: "dynamic", label: "Dynamic" },
          ]}
          onChange={(value) => {
            setState({ scaling: value });
            narrate(value === "dynamic" ? "Dynamic loss scaling: skip and halve on overflow." : "Static loss scale.");
          }}
        />
        <p className="lab-note">
          Static keeps one scale for the whole run. Dynamic skips any step whose gradient overflowed, halves the scale, and
          doubles it again after {GROWTH_INTERVAL} clean steps.
        </p>
      </>
    );
  }
  if (fault === "bad-batch") {
    return (
      <>
        {clip}
        <div className="ts-rollback">
          <button
            type="button"
            className={rollback ? "quiet-action" : "primary-action"}
            aria-pressed={rollback}
            onClick={() => {
              setState({ rollback: !rollback });
              narrate(
                rollback
                  ? "Showing the run without the rollback."
                  : `Rolled back to the checkpoint at step ${ROLLBACK.checkpointStep} and skipped steps ${ROLLBACK.skipFrom} to ${ROLLBACK.skipTo}.`,
              );
            }}
          >
            {rollback ? "Undo the rollback" : "Roll back and skip"}
          </button>
          <p className="lab-note">
            {rollback
              ? `Restarted from the checkpoint saved after step ${ROLLBACK.checkpointStep}. Steps ${ROLLBACK.replayFrom} to ${ROLLBACK.replayTo} replay exactly, and steps ${ROLLBACK.skipFrom} to ${ROLLBACK.skipTo} are read but never applied.`
              : `The run saves a checkpoint every 50 steps. Steps ${GARBAGE_FIRST_STEP} to ${GARBAGE_FIRST_STEP + GARBAGE_COUNT - 1} get “${GARBAGE_TEXT}” in place of real text.`}
          </p>
        </div>
      </>
    );
  }
  const zIndex = Math.max(0, Z_LOSSES.indexOf(zLoss as (typeof Z_LOSSES)[number]));
  const rateIndex = Math.max(0, ADAM_RATES.indexOf(adamRate as (typeof ADAM_RATES)[number]));
  return (
    <>
      <RangeControl
        label="z-loss"
        min={0}
        max={Z_LOSSES.length - 1}
        step={1}
        value={zIndex}
        format={(value) => formatCoefficient(Z_LOSSES[value] ?? 0)}
        onChange={(value) => {
          const next = Z_LOSSES[Math.round(value)] ?? 0;
          setState({ zLoss: next });
          narrate(next === 0 ? "z-loss off." : `z-loss ${next}.`);
        }}
      />
      <RangeControl
        label="Learning rate"
        min={0}
        max={ADAM_RATES.length - 1}
        step={1}
        value={rateIndex}
        format={(value) => `${ADAM_RATES[value] ?? ADAM_RATES[ADAM_RATES.length - 1]}`}
        onChange={(value) => {
          const next = ADAM_RATES[Math.round(value)] ?? ADAM_RATES[ADAM_RATES.length - 1];
          setState({ adamRate: next });
          narrate(`Adam learning rate ${next}.`);
        }}
      />
      <p className="lab-note">
        z-loss adds the coefficient times the squared log-partition to every example's loss, pulling log Z toward zero. PaLM
        used 0.0001.
      </p>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Diagnosis                                                                   */
/* -------------------------------------------------------------------------- */

function Diagnosis({
  settings,
  summary,
  points,
  spike,
  originalSpike,
  originalFinal,
}: {
  settings: FaultSettings;
  summary: ReturnType<typeof summarizeRun>;
  points: ReadonlyArray<StepPoint>;
  spike: ReturnType<typeof spikeReport>;
  originalSpike: ReturnType<typeof spikeReport>;
  originalFinal: number | null;
}) {
  const { fault, clipNorm, schedule, rollback, zLoss, adamRate } = settings;
  const finished = summary.finalLoss !== null;
  const worse = finished && (summary.finalLoss ?? 0) > UNIFORM_CROSS_ENTROPY;
  const firstRatio = points[0]?.ratio ?? null;
  let warning = summary.haltedAt !== null || worse;
  let headline: string;
  let detail: string;

  if (summary.haltedAt !== null) {
    headline = `The run halted at step ${summary.haltedAt}: a weight left the range of a 32-bit float and the loss stopped being a number.`;
    detail =
      fault === "fp16"
        ? `With a static loss scale, the first overflow sends infinity straight into the update. The first overflow came at step ${
            summary.firstOverflow ?? summary.haltedAt
          }. Switch Scaling to Dynamic, or lower Loss scale.`
        : `The two matrices multiply, so each one's gradient grows with the other and every oversized step makes the next one larger. Update ÷ weight was already ${formatNumber(
            firstRatio,
          )} at step 1, before the gradient norm looked wrong. Set Gradient clip to 0.50.`;
  } else if (fault === "none") {
    warning = false;
    headline = `A healthy run: it finished at ${formatLoss(summary.finalLoss)} nats, below the ${UNIFORM_CROSS_ENTROPY.toFixed(3)} of a model that knows nothing.`;
    detail =
      "The shaded band on each panel is this run's own 5th to 95th percentile. Those bands are the normal range every other fault is read against.";
  } else if (fault === "rate") {
    headline = worse
      ? `It finished, but at ${formatLoss(summary.finalLoss)} nats it is worse than a model that knows nothing.`
      : `It finished at ${formatLoss(summary.finalLoss)} nats with the clip binding on ${summary.clipped} of ${summary.steps} steps.`;
    detail =
      clipNorm === 0
        ? "Without a clip, the schedule only slows the blow-up: the weights stay finite but huge, and the loss ends far above chance. Add Gradient clip."
        : schedule === "constant"
          ? "The clip caps every step's length, so the run survives, but a constant rate of 8 keeps every capped step long. Now set Schedule to Warmup + cosine."
          : "The clip caps each step, and the schedule shrinks the steps late in the run, so the same rate of 8 can settle. Neither fix does this on its own.";
  } else if (fault === "fp16") {
    headline =
      summary.overflow === 0
        ? `No gradient entry overflowed at this loss scale, and the run finished at ${formatLoss(summary.finalLoss)} nats.`
        : `Dynamic scaling skipped ${summary.overflowSkips.length} step${summary.overflowSkips.length === 1 ? "" : "s"} and finished at ${formatLoss(summary.finalLoss)} nats.`;
    detail =
      summary.overflow === 0
        ? "Rounding each gradient to half precision barely moves the run. Raise Loss scale until an entry times the scale passes 65,504, the largest fp16 value."
        : `Each overflow halved the scale before the bad gradient could touch a weight. The skipped steps were ${summary.overflowSkips.join(", ")}.`;
  } else if (fault === "bad-batch") {
    if (rollback) {
      headline = `Rolled back: it finished at ${formatLoss(summary.finalLoss)} nats, against ${formatLoss(originalFinal)} without the rollback.`;
      detail = `The replay is identical to the original run until step ${ROLLBACK.skipFrom}, then never applies the garbage.${
        originalSpike ? ` The original spiked by ${formatLoss(originalSpike.rise)} nats.` : ""
      }`;
    } else {
      warning = true;
      headline = spike
        ? `A loss spike: whole-corpus loss rose by ${formatLoss(spike.rise)} nats, to ${formatLoss(spike.peak)} at step ${spike.peakStep}.`
        : "The garbage batches went in at steps 400 to 404.";
      detail =
        clipNorm > 0
          ? "The clip shortened the garbage steps, so the spike is smaller, but the garbage still went in. Press Roll back and skip."
          : "Adam divides each weight's step by that weight's recent gradient size. The rows for j and x never start a pair in the clean text, so the garbage moves them by full-sized steps. Try Gradient clip at 1.00.";
    }
  } else {
    headline = worse
      ? `Logits ran away: the loss ended at ${formatLoss(summary.finalLoss)} nats, worse than a model that knows nothing.`
      : `It finished at ${formatLoss(summary.finalLoss)} nats with max logit peaking at ${formatNumber(summary.peakLogit)}.`;
    detail =
      adamRate >= 1
        ? zLoss > 0
          ? "z-loss pulls the log-partition toward zero, so the logits stay smaller, but steps this large still wreck the fit. Lower Learning rate."
          : "Adam moves each weight by roughly the learning rate on every step, so at 1 the scores lurch by large jumps. Raise z-loss, then lower Learning rate."
        : "A smaller rate keeps each step small, so the scores grow only as fast as the data asks.";
  }

  return (
    <div className={`tg-callout ${warning ? "is-warning" : "is-quiet"}`} role="status">
      <strong>{headline}</strong>
      <span>{detail}</span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Panels                                                                      */
/* -------------------------------------------------------------------------- */

interface Markers {
  haltedAt: number | null;
  skipped: Array<[number, number]>;
  garbage: Array<[number, number]>;
  overflow: number[];
  rollbackAt: number | null;
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** The rendered width of an element, so each chart is drawn at its real size and its text stays readable. */
function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const read = () => {
      const next = Math.round(element.getBoundingClientRect().width);
      if (next > 0) setWidth((current) => (Math.abs(current - next) > 1 ? next : current));
    };
    read();
    const observer = new ResizeObserver(read);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Ids safe to use inside url(#…). */
const useSvgId = () => `ts${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

/** Approximate width of a 10px monospace label. */
const textWidth = (text: string) => text.length * 6.05;

type Zoom = "all" | "first" | "event";
const FIRST_STEPS = 40;
const PANEL_GAP = 18;
/** Room to the right of each plot for the names of its horizontal guides. */
const GUTTER = 62;

/** One panel's pixel geometry, the steps it shows, and the step the pointer is on. All four panels share it. */
interface Frame {
  W: number;
  H: number;
  L: number;
  R: number;
  T: number;
  B: number;
  from: number;
  to: number;
  ticks: number[];
  /** Few enough steps are on screen to draw each one as its own dot. */
  dense: boolean;
  stepWidth: number;
  xAt: (step: number) => number;
  probe: number | null;
  onProbe: (step: number | null) => void;
}

type SeriesPoint = { step: number; value: number | null };
type Scale = (value: number) => number | null;

const inWindow = (frame: Frame, step: number) => step >= frame.from && step <= frame.to;
/** The points a panel draws: the window plus one step either side, so lines run to the plot's edge. */
const visibleSteps = <T extends { step: number }>(list: ReadonlyArray<T>, frame: Frame) =>
  list.filter((point) => point.step >= frame.from - 1 && point.step <= frame.to + 1);

function linePath(list: ReadonlyArray<SeriesPoint>, frame: Frame, yAt: Scale) {
  let path = "";
  let drawing = false;
  for (const point of list) {
    const y = point.value === null ? null : yAt(point.value);
    if (y === null || !Number.isFinite(y)) {
      drawing = false;
      continue;
    }
    path += `${drawing ? "L" : "M"}${frame.xAt(point.step).toFixed(1)},${y.toFixed(1)}`;
    drawing = true;
  }
  return path;
}

function dotPath(list: ReadonlyArray<SeriesPoint>, frame: Frame, yAt: Scale) {
  return list
    .map((point) => {
      const y = point.value === null ? null : yAt(point.value);
      return y === null || !Number.isFinite(y) ? "" : `M${frame.xAt(point.step).toFixed(1)},${y.toFixed(1)}h0.01`;
    })
    .join("");
}

/**
 * The area between a line and a horizontal level, one closed shape per unbroken stretch of the line.
 * Clipped to one side of the level, it paints exactly where the signal is outside its normal range.
 */
function areaToLevel(list: ReadonlyArray<SeriesPoint>, frame: Frame, yAt: Scale, level: number) {
  let path = "";
  let lastX: number | null = null;
  const close = () => {
    if (lastX !== null) path += `L${lastX.toFixed(1)},${level.toFixed(1)}Z`;
    lastX = null;
  };
  for (const point of list) {
    const y = point.value === null ? null : yAt(point.value);
    if (y === null || !Number.isFinite(y)) {
      close();
      continue;
    }
    const x = frame.xAt(point.step);
    path += lastX === null ? `M${x.toFixed(1)},${level.toFixed(1)}L${x.toFixed(1)},${y.toFixed(1)}` : `L${x.toFixed(1)},${y.toFixed(1)}`;
    lastX = x;
  }
  close();
  return path;
}

/** Tints the part of a signal that is above or below its normal range. */
function OutOfRange({
  list,
  frame,
  yAt,
  band,
}: {
  list: ReadonlyArray<SeriesPoint>;
  frame: Frame;
  yAt: Scale;
  band: { low: number; high: number } | null;
}) {
  const id = useSvgId();
  if (!band) return null;
  const top = yAt(band.high);
  const bottom = yAt(band.low);
  if (top === null || bottom === null) return null;
  const { L, R, T, B } = frame;
  return (
    <g className="ts-excess">
      <clipPath id={`${id}a`}>
        <rect x={L} y={T - 2} width={R - L} height={Math.max(0, top - T + 2)} />
      </clipPath>
      <clipPath id={`${id}b`}>
        <rect x={L} y={bottom} width={R - L} height={Math.max(0, B - bottom + 2)} />
      </clipPath>
      <path d={areaToLevel(list, frame, yAt, top)} clipPath={`url(#${id}a)`} />
      <path d={areaToLevel(list, frame, yAt, bottom)} clipPath={`url(#${id}b)`} />
    </g>
  );
}

function MarkerLayer({ markers, frame, detail }: { markers: Markers; frame: Frame; detail: boolean }) {
  const { L, R, T, B, xAt, stepWidth } = frame;
  const height = B - T;
  const skippedGarbage = markers.garbage.some(([first]) => markers.skipped.some(([skipFirst]) => skipFirst === first));
  const firstSkip = markers.skipped[0];
  const halted = markers.haltedAt;
  const haltLabel = halted === null ? "" : detail ? `halted at step ${halted}` : "halted";
  const haltX = halted === null ? 0 : xAt(halted);
  const haltFits = R - haltX > textWidth(haltLabel) + 10;
  return (
    <g>
      {markers.garbage.map(([first, last]) => {
        const x = xAt(first) - stepWidth / 2;
        return (
          <g key={`garbage-${first}`}>
            <rect className="ts-garbage" x={x.toFixed(1)} y={T} width={Math.max(2, (last - first + 1) * stepWidth).toFixed(1)} height={height} />
            {inWindow(frame, first) && (
              <text className="ts-marker-label ts-marker-label--garbage" x={(x - 4).toFixed(1)} y={T + 11} textAnchor="end">
                {skippedGarbage ? "garbage, skipped" : "garbage"}
              </text>
            )}
          </g>
        );
      })}
      {markers.skipped.map(([first, last]) => (
        <rect
          key={`skip-${first}`}
          className="ts-skip"
          x={(xAt(first) - stepWidth / 2).toFixed(1)}
          y={T}
          width={Math.max(1.6, (last - first + 1) * stepWidth).toFixed(1)}
          height={height}
        />
      ))}
      {detail && firstSkip && !skippedGarbage && inWindow(frame, firstSkip[0]) && (
        <text className="ts-marker-label" x={(xAt(firstSkip[0]) + 4).toFixed(1)} y={B - 6}>
          skipped
        </text>
      )}
      {markers.rollbackAt !== null && inWindow(frame, markers.rollbackAt) && (
        <g>
          <line className="ts-rollback-line" x1={xAt(markers.rollbackAt).toFixed(1)} x2={xAt(markers.rollbackAt).toFixed(1)} y1={T} y2={B} />
          <text className="ts-marker-label ts-marker-label--rollback" x={(xAt(markers.rollbackAt) - 4).toFixed(1)} y={B - 6} textAnchor="end">
            {detail ? `restart from ${markers.rollbackAt}` : "restart"}
          </text>
        </g>
      )}
      {halted !== null && haltX <= R && (
        <g>
          <rect className="ts-halted-zone" x={Math.max(L, haltX).toFixed(1)} y={T} width={Math.max(0, R - Math.max(L, haltX)).toFixed(1)} height={height} />
          {haltX >= L && <line className="ts-halted" x1={haltX.toFixed(1)} x2={haltX.toFixed(1)} y1={T} y2={B} />}
          {haltX >= L && (
            <text
              className="ts-marker-label ts-marker-label--halted"
              x={(haltFits ? haltX + 8 : haltX - 6).toFixed(1)}
              y={detail ? (T + B) / 2 + 4 : T + 11}
              textAnchor={haltFits ? "start" : "end"}
            >
              {haltLabel}
            </text>
          )}
        </g>
      )}
    </g>
  );
}

function StepAxis({ frame }: { frame: Frame }) {
  const { L, R, T, B, H, xAt } = frame;
  return (
    <g>
      <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
      <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
      {frame.ticks.map((step, index) => (
        <text
          key={step}
          className="ts-tick"
          x={xAt(step).toFixed(1)}
          y={B + 13}
          textAnchor={index === 0 ? "start" : index === frame.ticks.length - 1 ? "end" : "middle"}
        >
          {step}
        </text>
      ))}
      <text className="ts-tick ts-tick--title" x={(L + R) / 2} y={H - 3} textAnchor="middle">
        optimizer step
      </text>
    </g>
  );
}

interface GutterItem {
  y: number;
  label: string;
  kind: "band" | "clip" | "rule";
  /** The band's top and bottom, drawn as a bracket. */
  span?: [number, number];
}

/** Names the horizontal guides beside the plot, nudged apart so no two labels overlap. */
function Gutter({ frame, items }: { frame: Frame; items: GutterItem[] }) {
  const placed = [...items].sort((a, b) => a.y - b.y).map((item) => ({ ...item, at: item.y }));
  placed.forEach((item, index) => {
    item.at = Math.max(frame.T + 6, item.at, index > 0 ? placed[index - 1].at + 12 : -Infinity);
  });
  for (let index = placed.length - 1; index >= 0; index -= 1) {
    const limit = index === placed.length - 1 ? frame.B - 2 : placed[index + 1].at - 12;
    placed[index].at = Math.min(placed[index].at, limit);
  }
  return (
    <g>
      {placed.map((item) => (
        <g key={item.label} className={`ts-gutter ts-gutter--${item.kind}`}>
          {item.span && <path d={`M${frame.R + 2},${item.span[0].toFixed(1)}h3V${item.span[1].toFixed(1)}h-3`} />}
          <text x={frame.R + 9} y={(item.at + 3.5).toFixed(1)}>
            {item.label}
          </text>
        </g>
      ))}
    </g>
  );
}

function PanelFrame({
  frame,
  title,
  readout,
  tone,
  ariaLabel,
  caption,
  children,
}: {
  frame: Frame;
  title: string;
  readout: string;
  tone: string;
  ariaLabel: string;
  caption: string;
  children: ReactNode;
}) {
  const move = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - box.left) / Math.max(1, box.width)) * frame.W;
    if (x < frame.L - 8 || x > frame.R + 8) {
      frame.onProbe(null);
      return;
    }
    const step = Math.round(frame.from + ((x - frame.L) / (frame.R - frame.L)) * (frame.to - frame.from));
    frame.onProbe(clamp(step, frame.from, frame.to));
  };
  return (
    <figure className={`ts-panel ts-tone--${tone}`}>
      <div className="ts-panel__head">
        <h4>{title}</h4>
        <span>{readout}</span>
      </div>
      <svg
        viewBox={`0 0 ${frame.W} ${frame.H}`}
        role="img"
        aria-label={ariaLabel}
        onPointerMove={move}
        onPointerDown={move}
        onPointerLeave={() => frame.onProbe(null)}
      >
        {children}
        {frame.probe !== null && (
          <line className="ts-probe" x1={frame.xAt(frame.probe).toFixed(1)} x2={frame.xAt(frame.probe).toFixed(1)} y1={frame.T} y2={frame.B} />
        )}
      </svg>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function BandRect({ frame, top, bottom }: { frame: Frame; top: number | null; bottom: number | null }) {
  if (top === null || bottom === null || !Number.isFinite(top) || !Number.isFinite(bottom)) return null;
  const y = Math.min(top, bottom);
  return <rect className="ts-band" x={frame.L} y={y.toFixed(1)} width={frame.R - frame.L} height={Math.max(1.5, Math.abs(bottom - top)).toFixed(1)} />;
}

/** What happened on a step besides its value, for the readout under the pointer. */
function stepTags(point: StepPoint | undefined, haltedAt: number | null) {
  if (!point) return haltedAt !== null ? " · after the halt" : "";
  const tags: string[] = [];
  if (point.step === haltedAt) tags.push("halted");
  if (point.faulty) tags.push("garbage");
  if (point.overflow) tags.push("overflow");
  if (point.skipped) tags.push("skipped");
  if (point.clipped) tags.push("clipped");
  return tags.length ? ` · ${tags.join(", ")}` : "";
}

function LossPanel({
  frame,
  points,
  corpus,
  referenceCorpus,
  originalCorpus,
  band,
  markers,
}: {
  frame: Frame;
  points: ReadonlyArray<StepPoint>;
  corpus: ReadonlyArray<{ step: number; loss: number }>;
  referenceCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  originalCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  band: { low: number; high: number } | null;
  markers: Markers;
}) {
  const clip = useSvgId();
  const { L, R, T, B } = frame;
  const batch: SeriesPoint[] = visibleSteps(points, frame).map((point) => ({ step: point.step, value: point.loss }));
  const shown = (list: ReadonlyArray<{ step: number; loss: number }>): SeriesPoint[] =>
    visibleSteps(list, frame)
      .filter((point) => point.step > 0)
      .map((point) => ({ step: point.step, value: point.loss }));
  const corpusLine = shown(corpus);
  const referenceLine = referenceCorpus ? shown(referenceCorpus) : null;
  const originalLine = originalCorpus ? shown(originalCorpus) : null;
  const visible = [...batch, ...corpusLine, ...(referenceLine ?? []), ...(originalLine ?? [])]
    .map((point) => point.value)
    .filter((value): value is number => value !== null);
  const top = Math.min(28, Math.max(4, Math.ceil(Math.max(0, ...visible))));
  const yAt = (value: number) => B - (Math.min(top, Math.max(0, value)) / top) * (B - T);
  const stride = top <= 6 ? 1 : top <= 12 ? 2 : top <= 20 ? 5 : 7;
  const ticks: number[] = [];
  for (let tick = 0; tick <= top; tick += stride) ticks.push(tick);
  const last = corpus.at(-1);
  const chanceY = yAt(UNIFORM_CROSS_ENTROPY);
  const probe = frame.probe;
  const probed = probe === null ? undefined : points.find((point) => point.step === probe);
  const probedCorpus = probe === null ? undefined : [...corpus].reverse().find((point) => point.step > 0 && point.step <= probe);
  const readout =
    probe !== null
      ? `step ${probe} · batch ${formatLoss(probed?.loss)}${probedCorpus ? ` · corpus ${formatLoss(probedCorpus.loss)}` : ""}`
      : last
        ? `corpus ${formatLoss(last.loss)} at step ${last.step}`
        : "no finite loss";
  const gutter: GutterItem[] = band
    ? [{ y: (yAt(band.high) + yAt(band.low)) / 2, label: "normal", kind: "band", span: [yAt(band.high), yAt(band.low)] }]
    : [];
  return (
    <PanelFrame
      frame={frame}
      title="Loss"
      tone="loss"
      readout={readout}
      ariaLabel={`Loss at optimizer steps ${frame.from} to ${frame.to}. The thin line is each batch's loss, the thick line the whole-corpus loss every 4 steps, ${
        last ? `ending at ${formatLoss(last.loss)} at step ${last.step}` : "with no finite value"
      }. The dotted line is ${UNIFORM_CROSS_ENTROPY.toFixed(3)}, a model that knows nothing.`}
      caption={`Thin: each batch's own loss. Thick: whole-corpus loss every 4 steps. Dotted: ln 30 = ${UNIFORM_CROSS_ENTROPY.toFixed(3)}, a model that knows nothing. Values above ${top} are pinned to the top.`}
    >
      <clipPath id={clip}>
        <rect x={L} y={T - 2} width={R - L} height={B - T + 4} />
      </clipPath>
      {band && <BandRect frame={frame} top={yAt(band.high)} bottom={yAt(band.low)} />}
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="ts-grid" x1={L} x2={R} y1={yAt(tick).toFixed(1)} y2={yAt(tick).toFixed(1)} />
          <text className="ts-tick" x={L - 5} y={(yAt(tick) + 3.5).toFixed(1)} textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      <g clipPath={`url(#${clip})`}>
        <OutOfRange list={batch} frame={frame} yAt={yAt} band={band} />
        <MarkerLayer markers={markers} frame={frame} detail />
      </g>
      <line className="ts-chance" x1={L} x2={R} y1={chanceY.toFixed(1)} y2={chanceY.toFixed(1)} />
      <StepAxis frame={frame} />
      <g clipPath={`url(#${clip})`}>
        <path className="ts-line ts-line--batch" d={linePath(batch, frame, yAt)} />
        {frame.dense && <path className="ts-dots ts-dots--steps ts-dots--faint" d={dotPath(batch, frame, yAt)} />}
        {referenceLine && <path className="ts-line ts-line--reference" d={linePath(referenceLine, frame, yAt)} />}
        {originalLine && <path className="ts-line ts-line--original" d={linePath(originalLine, frame, yAt)} />}
        <path className="ts-line ts-line--corpus" d={linePath(corpusLine, frame, yAt)} />
        {frame.dense && <path className="ts-dots ts-dots--corpus" d={dotPath(corpusLine, frame, yAt)} />}
      </g>
      <text className="ts-marker-label ts-marker-label--chance" x={L + 6} y={(chanceY - 4).toFixed(1)}>
        knows nothing · {UNIFORM_CROSS_ENTROPY.toFixed(3)}
      </text>
      <Gutter frame={frame} items={gutter} />
      {probe !== null && probed && probed.loss !== null && (
        <circle className="ts-probe-dot ts-probe-dot--small" cx={frame.xAt(probe).toFixed(1)} cy={yAt(probed.loss).toFixed(1)} r={2.6} />
      )}
      {probe !== null && probedCorpus && inWindow(frame, probedCorpus.step) && (
        <circle className="ts-probe-dot" cx={frame.xAt(probedCorpus.step).toFixed(1)} cy={yAt(probedCorpus.loss).toFixed(1)} r={3.6} />
      )}
    </PanelFrame>
  );
}

function LogPanel({
  frame,
  title,
  tone,
  points,
  signal,
  band,
  markers,
  floor,
  guides,
  showClipped = false,
  showOverflow = false,
  readout,
  caption,
}: {
  frame: Frame;
  title: string;
  tone: string;
  points: ReadonlyArray<StepPoint>;
  signal: "norm" | "maxLogit" | "ratio";
  band: { low: number; high: number } | null;
  markers: Markers;
  floor: number;
  guides: ReadonlyArray<{ value: number; label: string; kind: "clip" | "rule" }>;
  showClipped?: boolean;
  showOverflow?: boolean;
  readout: string;
  caption: string;
}) {
  const clip = useSvgId();
  const { L, R, T, B, xAt } = frame;
  const shown = visibleSteps(points, frame);
  const series: SeriesPoint[] = shown.map((point) => ({ step: point.step, value: point[signal] }));
  const list = values(shown, signal);
  const domain: LogDomain = logDomain([...list, ...(band ? [band.low, band.high] : []), ...guides.map((guide) => guide.value)], {
    floor,
    minDecades: 2,
  });
  const yAt = (value: number) => {
    const position = logPosition(value, domain);
    return position === null ? null : B - position * (B - T);
  };
  const ticks = decadeTicks(domain, 5);
  const clipped = showClipped ? series.filter((point, index) => shown[index].clipped && point.value !== null) : [];
  const probe = frame.probe;
  const probed = probe === null ? undefined : points.find((point) => point.step === probe);
  const probedValue = probed ? probed[signal] : null;
  const probedY = probedValue === null ? null : yAt(probedValue);
  const bandTop = band ? yAt(band.high) : null;
  const bandBottom = band ? yAt(band.low) : null;
  const gutter: GutterItem[] = [];
  if (bandTop !== null && bandBottom !== null) {
    gutter.push({ y: (bandTop + bandBottom) / 2, label: "normal", kind: "band", span: [bandTop, bandBottom] });
  }
  for (const guide of guides) {
    const y = yAt(guide.value);
    if (y !== null) gutter.push({ y, label: guide.label, kind: guide.kind });
  }
  return (
    <PanelFrame
      frame={frame}
      title={title}
      tone={tone}
      readout={
        probe !== null
          ? `step ${probe} · ${probedValue === null ? "no value" : formatNumber(probedValue)}${stepTags(probed, markers.haltedAt)}`
          : readout
      }
      ariaLabel={`${title} at optimizer steps ${frame.from} to ${frame.to} on a log scale from ${powerOfTen(domain.low)} to ${powerOfTen(domain.high)}. ${readout}.`}
      caption={caption}
    >
      <clipPath id={clip}>
        <rect x={L} y={T - 2} width={R - L} height={B - T + 4} />
      </clipPath>
      {band && <BandRect frame={frame} top={bandTop} bottom={bandBottom} />}
      {ticks.map((exponent) => {
        const y = yAt(10 ** exponent) ?? B;
        return (
          <g key={exponent}>
            <line className="ts-grid" x1={L} x2={R} y1={y.toFixed(1)} y2={y.toFixed(1)} />
            <text className="ts-tick" x={L - 5} y={(y + 3.5).toFixed(1)} textAnchor="end">
              {powerOfTen(exponent)}
            </text>
          </g>
        );
      })}
      <g clipPath={`url(#${clip})`}>
        <OutOfRange list={series} frame={frame} yAt={yAt} band={band} />
        <MarkerLayer markers={markers} frame={frame} detail={false} />
      </g>
      {guides.map((guide) => {
        const y = yAt(guide.value);
        if (y === null) return null;
        return <line key={guide.label} className={`ts-guide ts-guide--${guide.kind}`} x1={L} x2={R} y1={y.toFixed(1)} y2={y.toFixed(1)} />;
      })}
      <StepAxis frame={frame} />
      <g clipPath={`url(#${clip})`}>
        <path className="ts-line ts-line--signal" d={linePath(series, frame, yAt)} />
        {frame.dense && <path className="ts-dots ts-dots--steps" d={dotPath(series, frame, yAt)} />}
        {clipped.length > 0 && <path className={`ts-dots ts-dots--clipped${frame.dense ? " is-dense" : ""}`} d={dotPath(clipped, frame, yAt)} />}
      </g>
      {showOverflow &&
        markers.overflow
          .filter((step) => inWindow(frame, step))
          .map((step) => (
            <path
              key={`overflow-${step}`}
              className="ts-overflow"
              d={`M${(xAt(step) - 4.5).toFixed(1)},${T - 2}L${(xAt(step) + 4.5).toFixed(1)},${T - 2}L${xAt(step).toFixed(1)},${T + 7}Z`}
            />
          ))}
      <Gutter frame={frame} items={gutter} />
      {probe !== null && probedY !== null && <circle className="ts-probe-dot" cx={xAt(probe).toFixed(1)} cy={probedY.toFixed(1)} r={3.6} />}
    </PanelFrame>
  );
}

/** The four panels, drawn at the card's real width, with one zoom and one pointer shared between them. */
function Monitor({
  fault,
  points,
  referencePoints,
  corpus,
  referenceCorpus,
  originalCorpus,
  markers,
  summary,
  clipNorm,
  medianLogPartition,
}: {
  fault: FaultSettings["fault"];
  points: ReadonlyArray<StepPoint>;
  referencePoints: ReadonlyArray<StepPoint>;
  corpus: ReadonlyArray<{ step: number; loss: number }>;
  referenceCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  originalCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  markers: Markers;
  summary: ReturnType<typeof summarizeRun>;
  clipNorm: number;
  medianLogPartition: number | null;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(664);
  const [zoom, setZoom] = useState<Zoom>("all");
  const [probe, setProbe] = useState<number | null>(null);
  const garbage = markers.garbage[0];
  const eventWindow: [number, number] | null = garbage ? [garbage[0] - 20, Math.min(PLANNED_STEPS, garbage[0] + 40)] : null;
  const active: Zoom = zoom === "event" && !eventWindow ? "all" : zoom;
  const [from, to] = active === "first" ? [1, FIRST_STEPS] : active === "event" && eventWindow ? eventWindow : [1, PLANNED_STEPS];
  const columns = width >= 540 ? 2 : 1;
  const W = Math.max(240, Math.floor((width - 8 - (columns - 1) * PANEL_GAP) / columns));
  const H = clamp(Math.round(W * 0.56), 172, 236);
  const L = 44;
  const R = W - GUTTER;
  const T = 14;
  const B = H - 30;
  const ticks =
    active === "all"
      ? [1, 120, 240, 360, 480]
      : active === "first"
        ? [1, 10, 20, 30, 40]
        : Array.from({ length: Math.floor((to - from) / 20) + 1 }, (_, index) => from + index * 20);
  const frame: Frame = {
    W,
    H,
    L,
    R,
    T,
    B,
    from,
    to,
    ticks,
    dense: to - from <= 80,
    stepWidth: (R - L) / Math.max(1, to - from),
    xAt: (step) => L + ((step - from) / Math.max(1, to - from)) * (R - L),
    probe: probe !== null && probe >= from && probe <= to ? probe : null,
    onProbe: setProbe,
  };
  const clipShown = clipNorm > 0 && (fault === "rate" || fault === "bad-batch");
  return (
    <>
      <div className="ts-zoom">
        <SegmentedControl
          label="Steps shown"
          value={active}
          options={[
            { value: "all", label: `All ${PLANNED_STEPS} steps` },
            { value: "first", label: `First ${FIRST_STEPS} steps` },
            ...(eventWindow ? [{ value: "event", label: `Steps ${eventWindow[0]} to ${eventWindow[1]}` }] : []),
          ]}
          onChange={(value) => setZoom(value as Zoom)}
        />
        <p className="ts-zoom__hint">Point at any panel to read all four at the same step.</p>
      </div>
      <div className="ts-panels" ref={ref} data-columns={columns}>
        <LossPanel
          frame={frame}
          points={points}
          corpus={corpus}
          referenceCorpus={referenceCorpus}
          originalCorpus={originalCorpus}
          band={normalBand(referencePoints, "loss")}
          markers={markers}
        />
        <LogPanel
          frame={frame}
          title="Gradient norm"
          tone="gradient"
          points={points}
          signal="norm"
          band={normalBand(referencePoints, "norm")}
          markers={markers}
          floor={-3}
          guides={clipShown ? [{ value: clipNorm, label: `clip ${clipNorm.toFixed(2)}`, kind: "clip" }] : []}
          showClipped
          showOverflow
          readout={`peak ${formatNumber(summary.peakNorm)}`}
          caption={
            summary.peakNorm === null
              ? "No finite gradient norm was recorded before the run stopped."
              : `Length of the whole batch gradient before clipping, on a log scale. Peak ${formatNumber(summary.peakNorm)}${
                  summary.clipped > 0 ? `; the clip rescaled ${summary.clipped} of ${summary.steps} steps (dark dots)` : ""
                }${summary.overflow > 0 ? `; ${summary.overflow} step${summary.overflow === 1 ? "" : "s"} overflowed in fp16 (triangles)` : ""}.`
          }
        />
        <LogPanel
          frame={frame}
          title="Max logit"
          tone="logit"
          points={points}
          signal="maxLogit"
          band={normalBand(referencePoints, "maxLogit")}
          markers={markers}
          floor={-2}
          guides={[]}
          readout={`peak ${formatNumber(summary.peakLogit)}`}
          caption={`Largest score any example in the batch gave a character, on a log scale. Peak ${formatNumber(
            summary.peakLogit,
          )}; median log-partition ${formatNumber(medianLogPartition)}.`}
        />
        <LogPanel
          frame={frame}
          title="Update ÷ weight"
          tone="ratio"
          points={points}
          signal="ratio"
          band={normalBand(referencePoints, "ratio")}
          markers={markers}
          floor={-4}
          guides={[{ value: UPDATE_RULE, label: "10⁻³ rule", kind: "rule" }]}
          readout={`median ${formatNumber(summary.medianRatio)}`}
          caption={`How far each step moved the weights, as a share of their size, on a log scale. Median ${formatNumber(
            summary.medianRatio,
          )}; the dashed line is the 10⁻³ rule of thumb for large networks.`}
        />
      </div>
    </>
  );
}

function Legend({
  fault,
  refLabel,
  rollback,
  markers,
}: {
  fault: FaultSettings["fault"];
  refLabel: string;
  rollback: boolean;
  markers: Markers;
}) {
  return (
    <div className="ts-legend">
      <span className="ts-legend__item">
        <i className="ts-key ts-key--band" /> normal range: 5th to 95th percentile of {refLabel}
      </span>
      <span className="ts-legend__item">
        <i className="ts-key ts-key--excess" /> stronger tint: the signal is outside that range
      </span>
      {fault !== "none" && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--reference" /> whole-corpus loss of {refLabel}
        </span>
      )}
      {rollback && fault === "bad-batch" && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--original" /> whole-corpus loss without the rollback
        </span>
      )}
      {markers.garbage.length > 0 && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--garbage" /> garbage batches
        </span>
      )}
      {markers.skipped.length > 0 && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--skip" /> skipped steps (read, never applied)
        </span>
      )}
      {markers.overflow.length > 0 && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--overflow" /> fp16 overflow
        </span>
      )}
      {markers.haltedAt !== null && (
        <span className="ts-legend__item">
          <i className="ts-key ts-key--halted" /> halted: no values after this step
        </span>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Final loss on one scale                                                     */
/* -------------------------------------------------------------------------- */

const STRIP_LOW = 1.5;
const STRIP_HIGH = 4;

/** Where this run's final loss lands next to its clean reference and a model that knows nothing. */
function FinalLossStrip({
  finalLoss,
  haltedAt,
  reference,
  original,
}: {
  finalLoss: number | null;
  haltedAt: number | null;
  reference: { label: string; loss: number } | null;
  original: number | null;
}) {
  const [ref, width] = useWidth<HTMLElement>(664);
  const W = clamp(width, 260, 1200);
  const H = 96;
  const left = 14;
  const right = W - 56;
  const slot = W - 20;
  const track = 40;
  const xOf = (value: number) =>
    value > STRIP_HIGH ? slot : left + ((clamp(value, STRIP_LOW, STRIP_HIGH) - STRIP_LOW) / (STRIP_HIGH - STRIP_LOW)) * (right - left);
  const chanceX = xOf(UNIFORM_CROSS_ENTROPY);
  const ticks = [1.5, 2, 2.5, 3, 3.5, 4];

  const below = [
    { x: chanceX, text: `knows nothing ${UNIFORM_CROSS_ENTROPY.toFixed(3)}`, kind: "chance" },
    ...(reference ? [{ x: xOf(reference.loss), text: `${reference.label} ${formatLoss(reference.loss)}`, kind: "reference" }] : []),
    ...(original !== null ? [{ x: xOf(original), text: `without rollback ${formatLoss(original)}`, kind: "original" }] : []),
  ]
    .sort((a, b) => a.x - b.x)
    .map((item) => ({ ...item, at: item.x, half: textWidth(item.text) / 2 + 5 }));
  // Push labels apart left to right, then pull them back inside the frame right to left.
  below.forEach((item, index) => {
    const previous = below[index - 1];
    item.at = Math.max(item.half + 2, item.at, previous ? previous.at + previous.half + item.half : -Infinity);
  });
  for (let index = below.length - 1; index >= 0; index -= 1) {
    const next = below[index + 1];
    below[index].at = Math.min(below[index].at, next ? next.at - next.half - below[index].half : W - below[index].half - 2);
  }

  const runX = finalLoss === null ? slot : xOf(finalLoss);
  const runText =
    finalLoss === null
      ? `this run halted${haltedAt !== null ? ` at step ${haltedAt}` : ""}: no final loss`
      : `this run ${formatLoss(finalLoss)}${finalLoss > STRIP_HIGH ? " →" : ""}`;
  const runHalf = (runText.length * 6.7) / 2 + 4;
  const runAt = clamp(runX, runHalf + 2, W - runHalf - 2);
  const worse = finalLoss !== null && finalLoss > UNIFORM_CROSS_ENTROPY;
  const verdict =
    finalLoss === null
      ? "The run halted, so it has no final loss."
      : `This run ended at ${formatLoss(finalLoss)} nats, ${worse ? "worse than" : "better than"} a model that knows nothing (${UNIFORM_CROSS_ENTROPY.toFixed(3)}).`;
  return (
    <figure className="ts-final" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Final whole-corpus loss on a scale from ${STRIP_LOW} to ${STRIP_HIGH} nats, lower is better. ${verdict}${
          reference ? ` ${reference.label}: ${formatLoss(reference.loss)}.` : ""
        }${original !== null ? ` Without the rollback: ${formatLoss(original)}.` : ""}`}
      >
        <rect className="ts-final__worse" x={chanceX.toFixed(1)} y={track - 13} width={(W - 4 - chanceX).toFixed(1)} height={26} rx={3} />
        <line className="ts-final__track" x1={left} x2={right} y1={track} y2={track} />
        <path className="ts-final__break" d={`M${right + 6},${track + 5}l5,-10M${right + 12},${track + 5}l5,-10`} />
        {ticks.map((tick) => {
          const x = xOf(tick);
          const hidden = below.some((item) => Math.abs((item.x + item.at) / 2 - x) < 17);
          return (
            <g key={tick}>
              <line className="ts-final__tick" x1={x.toFixed(1)} x2={x.toFixed(1)} y1={track - 4} y2={track + 4} />
              {!hidden && (
                <text className="ts-tick" x={x.toFixed(1)} y={track + 19} textAnchor="middle">
                  {tick.toFixed(1)}
                </text>
              )}
            </g>
          );
        })}
        <line className="ts-chance" x1={chanceX.toFixed(1)} x2={chanceX.toFixed(1)} y1={track - 15} y2={track + 15} />
        {below.map((item) => (
          <g key={item.kind} className={`ts-final__ref ts-final__ref--${item.kind}`}>
            <path d={`M${item.x.toFixed(1)},${track + (item.kind === "chance" ? 16 : 14)}L${item.at.toFixed(1)},${H - 20}`} />
            <text x={item.at.toFixed(1)} y={H - 8} textAnchor="middle">
              {item.text}
            </text>
          </g>
        ))}
        {reference && (
          <g className="ts-final__mark ts-final__mark--reference">
            <line x1={xOf(reference.loss).toFixed(1)} x2={xOf(reference.loss).toFixed(1)} y1={track - 9} y2={track + 9} />
            <path d={`M${xOf(reference.loss).toFixed(1)},${track + 5}l4,4l-4,4l-4,-4Z`} />
          </g>
        )}
        {original !== null && (
          <g className="ts-final__mark ts-final__mark--original">
            <line x1={xOf(original).toFixed(1)} x2={xOf(original).toFixed(1)} y1={track - 9} y2={track + 9} />
            <circle cx={xOf(original).toFixed(1)} cy={track + 9} r={3.6} />
          </g>
        )}
        {finalLoss === null ? (
          <path className="ts-final__halt" d={`M${slot - 5},${track - 5}l10,10M${slot + 5},${track - 5}l-10,10`} />
        ) : (
          <circle className="ts-final__mark--run" cx={runX.toFixed(1)} cy={track} r={5.5} />
        )}
        <text className={`ts-final__run${finalLoss === null || worse ? " is-bad" : ""}`} x={runAt.toFixed(1)} y={track - 20} textAnchor="middle">
          {runText}
        </text>
      </svg>
      <figcaption>
        Final loss on all the training pairs, in nats per character. Left is better; the tinted stretch is worse than a model
        that knows nothing.
      </figcaption>
    </figure>
  );
}

/* -------------------------------------------------------------------------- */
/* Number formats                                                              */
/* -------------------------------------------------------------------------- */

const F_LOW = -46;
const F_HIGH = 40;
/** The lower picture zooms in on the decades where fp16 and its survival window live. */
const Z_LOW = -20;
const Z_HIGH = 10;
const exponentOf = (value: number) => Math.log10(Math.max(1e-300, value));

function FormatLine({
  flushBelow,
  overflowFrom,
  scaleExponent,
  norms,
}: {
  flushBelow: number;
  overflowFrom: number;
  scaleExponent: number;
  norms: ReadonlyArray<number>;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(664);
  const W = clamp(width, 280, 920);
  const left = 74;
  const right = W - 14;
  const span = right - left;
  const wideAt = (value: number) => left + ((clamp(exponentOf(value), F_LOW, F_HIGH) - F_LOW) / (F_HIGH - F_LOW)) * span;
  const zoomAt = (value: number) => left + ((clamp(exponentOf(value), Z_LOW, Z_HIGH) - Z_LOW) / (Z_HIGH - Z_LOW)) * span;

  const rows = NUMBER_FORMATS.map((format, index) => ({ format, y: 16 + index * 18 }));
  const wideWindowY = 16 + rows.length * 18;
  const wideAxis = wideWindowY + 14;
  const zoomTop = wideAxis + 44;
  const fp16 = NUMBER_FORMATS.find((format) => format.id === "fp16") ?? NUMBER_FORMATS[NUMBER_FORMATS.length - 1];
  const fpY = zoomTop + 24;
  const arrowY = fpY + 22;
  const windowY = arrowY + 22;
  const hasNorms = norms.length > 0;
  const normLow = hasNorms ? Math.min(...norms) : 0;
  const normHigh = hasNorms ? Math.max(...norms) : 0;
  const normY = windowY + 40;
  const zoneBottom = (hasNorms ? normY : windowY + 12) + 12;
  const zoomAxis = zoneBottom + 6;
  const H = zoomAxis + 30;

  const windowLeft = zoomAt(flushBelow);
  const windowRight = zoomAt(overflowFrom);
  const fpRight = zoomAt(fp16.largest);
  const zeroLabel = "← rounds to zero";
  const infinityLabel = "becomes infinity →";
  const overflowing = hasNorms && normHigh >= overflowFrom;
  const normText = hasNorms ? `${formatNumber(normLow)} to ${formatNumber(normHigh)}` : "";
  const normLeft = zoomAt(normLow) - 1;
  const normRight = zoomAt(normHigh) + 1;
  const normLabel = `gradient norms ${normText}`;
  // The label sits right of the box unless it would run off the chart or, for a run inside the window, across its edge.
  const normLabelRight = (overflowing ? right : windowRight) - normRight > textWidth(normLabel) + 10;

  return (
    <div className="ts-format" ref={ref}>
      <svg
        className="ts-format-line"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Positive ranges of three number formats on a log scale from ${powerOfTen(F_LOW)} to ${powerOfTen(
          F_HIGH,
        )}. fp32 and bf16 reach about 3.4 × 10³⁸; fp16 stops at 65,504. Below, a closer view from ${powerOfTen(Z_LOW)} to ${powerOfTen(
          Z_HIGH,
        )}: dividing fp16's range by the loss scale of ${powerOfTwo(scaleExponent)} gives the window a gradient entry survives in, from ${formatNumber(
          flushBelow,
          1,
        )} to ${formatNumber(overflowFrom, 2)}.${
          hasNorms ? ` This run's gradient norms run from ${normText}, ${overflowing ? "past" : "inside"} the top of that window.` : ""
        }`}
      >
        {/* The whole range of each format */}
        {[-40, -20, 0, 20, 40].map((exponent) => (
          <g key={exponent}>
            <line className="ts-grid" x1={wideAt(10 ** exponent).toFixed(1)} x2={wideAt(10 ** exponent).toFixed(1)} y1={6} y2={wideAxis} />
            <text className="ts-tick" x={wideAt(10 ** exponent).toFixed(1)} y={wideAxis + 12} textAnchor={exponent === 40 ? "end" : "middle"}>
              {powerOfTen(exponent)}
            </text>
          </g>
        ))}
        {rows.map(({ format, y }) => (
          <g key={format.id}>
            <text className="ts-format-label" x={left - 8} y={y + 3.5} textAnchor="end">
              {format.label}
            </text>
            <rect
              className="ts-format-sub"
              x={wideAt(format.smallestSubnormal).toFixed(1)}
              y={y - 5}
              width={Math.max(1, wideAt(format.smallestNormal) - wideAt(format.smallestSubnormal)).toFixed(1)}
              height={10}
            />
            <rect
              className={`ts-format-bar ts-format-bar--${format.id}`}
              x={wideAt(format.smallestNormal).toFixed(1)}
              y={y - 5}
              width={Math.max(1, wideAt(format.largest) - wideAt(format.smallestNormal)).toFixed(1)}
              height={10}
            />
          </g>
        ))}
        <text className="ts-format-label" x={left - 8} y={wideWindowY + 3.5} textAnchor="end">
          survives
        </text>
        <rect
          className="ts-format-window"
          x={wideAt(flushBelow).toFixed(1)}
          y={wideWindowY - 5}
          width={Math.max(1, wideAt(overflowFrom) - wideAt(flushBelow)).toFixed(1)}
          height={10}
        />
        <text className="ts-format-end" x={(wideAt(fp16.largest) + 6).toFixed(1)} y={rows[rows.length - 1].y + 3.5}>
          stops at 65,504
        </text>

        {/* The zoom: which stretch of the upper axis the lower one magnifies */}
        <path
          className="ts-format-zoom"
          d={`M${wideAt(10 ** Z_LOW).toFixed(1)},${wideAxis + 18}H${wideAt(10 ** Z_HIGH).toFixed(1)}L${right},${zoomTop}H${left}Z`}
        />
        <text className="ts-format-zoom-label" x={((wideAt(10 ** Z_LOW) + wideAt(10 ** Z_HIGH)) / 2).toFixed(1)} y={zoomTop - 8} textAnchor="middle">
          closer view: {powerOfTen(Z_LOW)} to {powerOfTen(Z_HIGH)}
        </text>

        {/* Zones outside the survival window */}
        <rect className="ts-format-zero" x={left} y={windowY - 12} width={Math.max(0, windowLeft - left).toFixed(1)} height={zoneBottom - windowY + 12} />
        <rect
          className="ts-format-infinity"
          x={windowRight.toFixed(1)}
          y={windowY - 12}
          width={Math.max(0, right - windowRight).toFixed(1)}
          height={zoneBottom - windowY + 12}
        />
        {[-20, -15, -10, -5, 0, 5, 10].map((exponent) => (
          <g key={exponent}>
            <line className="ts-grid" x1={zoomAt(10 ** exponent).toFixed(1)} x2={zoomAt(10 ** exponent).toFixed(1)} y1={zoomTop + 6} y2={zoomAxis} />
            <text
              className="ts-tick"
              x={zoomAt(10 ** exponent).toFixed(1)}
              y={zoomAxis + 12}
              textAnchor={exponent === Z_HIGH ? "end" : exponent === Z_LOW ? "start" : "middle"}
            >
              {powerOfTen(exponent)}
            </text>
          </g>
        ))}

        <text className="ts-format-label" x={left - 8} y={fpY + 3.5} textAnchor="end">
          fp16
        </text>
        <rect
          className="ts-format-sub"
          x={zoomAt(fp16.smallestSubnormal).toFixed(1)}
          y={fpY - 6}
          width={Math.max(1, zoomAt(fp16.smallestNormal) - zoomAt(fp16.smallestSubnormal)).toFixed(1)}
          height={12}
        />
        <rect
          className="ts-format-bar ts-format-bar--fp16"
          x={zoomAt(fp16.smallestNormal).toFixed(1)}
          y={fpY - 6}
          width={Math.max(1, fpRight - zoomAt(fp16.smallestNormal)).toFixed(1)}
          height={12}
        />
        <text className="ts-format-end" x={(fpRight - 4).toFixed(1)} y={fpY - 10} textAnchor="end">
          largest 65,504
        </text>

        {/* The loss scale slides fp16's range down to the window */}
        {scaleExponent > 0 ? (
          <g className="ts-format-shift">
            <path d={`M${fpRight.toFixed(1)},${fpY + 8}V${arrowY}H${windowRight.toFixed(1)}V${windowY - 10}`} />
            <path className="ts-format-shift__head" d={`M${(windowRight - 4).toFixed(1)},${windowY - 13}l4,5l4,-5Z`} />
            <text x={(windowRight - 8).toFixed(1)} y={arrowY + 3.5} textAnchor="end">
              divide by the loss scale, {powerOfTwo(scaleExponent)}
            </text>
          </g>
        ) : (
          <text className="ts-format-shift-note" x={(fpRight - 4).toFixed(1)} y={arrowY + 3.5} textAnchor="end">
            ÷ 2⁰ = 1: the window is fp16's own range
          </text>
        )}

        <text className="ts-format-label" x={left - 8} y={windowY + 3.5} textAnchor="end">
          survives
        </text>
        <rect className="ts-format-window" x={windowLeft.toFixed(1)} y={windowY - 6} width={Math.max(1, windowRight - windowLeft).toFixed(1)} height={12} />
        <text className="ts-format-end ts-format-end--window" x={windowLeft.toFixed(1)} y={windowY + 19}>
          {formatNumber(flushBelow, 1)}
        </text>
        <text className="ts-format-end ts-format-end--window" x={windowRight.toFixed(1)} y={windowY + 19} textAnchor="end">
          {formatNumber(overflowFrom, 2)}
        </text>
        {windowLeft - left > textWidth(zeroLabel) + 10 && (
          <text className="ts-format-zone" x={(windowLeft - 6).toFixed(1)} y={windowY + 3.5} textAnchor="end">
            {zeroLabel}
          </text>
        )}
        {right - windowRight > textWidth(infinityLabel) + 10 && (
          <text className="ts-format-zone ts-format-zone--infinity" x={(windowRight + 6).toFixed(1)} y={windowY + 3.5}>
            {infinityLabel}
          </text>
        )}
        <line className="ts-format-edge" x1={windowRight.toFixed(1)} x2={windowRight.toFixed(1)} y1={windowY - 10} y2={zoneBottom} />

        {hasNorms && (
          <g>
            <text className="ts-format-label" x={left - 8} y={normY + 3.5} textAnchor="end">
              this run
            </text>
            <rect className="ts-format-norms" x={normLeft.toFixed(1)} y={normY - 7} width={Math.max(3, normRight - normLeft).toFixed(1)} height={14} />
            <text
              className={`ts-format-end ts-format-end--norms${overflowing ? " is-over" : ""}`}
              x={(normLabelRight ? normRight + 6 : normLeft - 6).toFixed(1)}
              y={normY + 3.5}
              textAnchor={normLabelRight ? "start" : "end"}
            >
              {normLabel}
            </text>
          </g>
        )}
        <text className="ts-tick ts-tick--title" x={(left + right) / 2} y={H - 3} textAnchor="middle">
          magnitude, log scale (lighter: subnormal)
        </text>
      </svg>
    </div>
  );
}
