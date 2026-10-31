import { useMemo, type ReactNode } from "react";
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
        <div className="ts-panels">
          <LossPanel
            points={points}
            corpus={corpus}
            referenceCorpus={referenceCorpus}
            originalCorpus={originalCorpus}
            band={normalBand(referencePoints, "loss")}
            markers={markers}
          />
          <LogPanel
            title="Gradient norm"
            tone="gradient"
            points={points}
            signal="norm"
            band={normalBand(referencePoints, "norm")}
            markers={markers}
            floor={-3}
            guides={clipNorm > 0 && (fault === "rate" || fault === "bad-batch") ? [{ value: clipNorm, label: `clip ${clipNorm.toFixed(2)}`, kind: "clip" }] : []}
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
            title="Update ÷ weight"
            tone="ratio"
            points={points}
            signal="ratio"
            band={normalBand(referencePoints, "ratio")}
            markers={markers}
            floor={-4}
            guides={[{ value: UPDATE_RULE, label: "10⁻³ rule of thumb", kind: "rule" }]}
            readout={`median ${formatNumber(summary.medianRatio)}`}
            caption={`How far each step moved the weights, as a share of their size, on a log scale. Median ${formatNumber(
              summary.medianRatio,
            )}; the dashed line is the 10⁻³ rule of thumb for large networks.`}
          />
        </div>
        <Legend fault={fault} refLabel={refLabel} rollback={rollback} markers={markers} />
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Run report" className="ts-report">
          <SurfaceHeading kicker="Measured on this run" title="What the dashboard is telling you" />
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
          <FormatLine flushBelow={half.flushBelow} overflowFrom={half.overflowFrom} norms={fault === "fp16" ? finiteNorms : []} />
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

const W = 320;
const H = 150;
const L = 40;
const R = 312;
const T = 12;
const B = 122;
const STEP_TICKS = [1, 120, 240, 360, 480];
const xAt = (step: number) => L + ((Math.min(PLANNED_STEPS, Math.max(1, step)) - 1) / Math.max(1, PLANNED_STEPS - 1)) * (R - L);
const stepWidth = (R - L) / Math.max(1, PLANNED_STEPS - 1);

function linePath(list: ReadonlyArray<{ step: number; value: number | null }>, yAt: (value: number) => number | null) {
  let path = "";
  let drawing = false;
  for (const point of list) {
    const y = point.value === null ? null : yAt(point.value);
    if (y === null || !Number.isFinite(y)) {
      drawing = false;
      continue;
    }
    path += `${drawing ? "L" : "M"}${xAt(point.step).toFixed(1)},${y.toFixed(1)}`;
    drawing = true;
  }
  return path;
}

function dotPath(list: ReadonlyArray<{ step: number; value: number }>, yAt: (value: number) => number | null) {
  return list
    .map((point) => {
      const y = yAt(point.value);
      return y === null || !Number.isFinite(y) ? "" : `M${xAt(point.step).toFixed(1)},${y.toFixed(1)}h0.01`;
    })
    .join("");
}

function MarkerLayer({ markers, label }: { markers: Markers; label: boolean }) {
  return (
    <g>
      {markers.garbage.map(([first, last]) => (
        <g key={`garbage-${first}`}>
          <rect
            className="ts-garbage"
            x={(xAt(first) - stepWidth / 2).toFixed(1)}
            y={T}
            width={Math.max(2, (last - first + 1) * stepWidth).toFixed(1)}
            height={B - T}
          />
          {label && (
            <text className="ts-marker-label" x={(xAt(first) - 3).toFixed(1)} y={T + 9} textAnchor="end">
              garbage
            </text>
          )}
        </g>
      ))}
      {markers.skipped.map(([first, last]) => (
        <rect
          key={`skip-${first}`}
          className="ts-skip"
          x={(xAt(first) - stepWidth / 2).toFixed(1)}
          y={T}
          width={Math.max(1.4, (last - first + 1) * stepWidth).toFixed(1)}
          height={B - T}
        />
      ))}
      {markers.rollbackAt !== null && (
        <g>
          <line className="ts-rollback-line" x1={xAt(markers.rollbackAt).toFixed(1)} x2={xAt(markers.rollbackAt).toFixed(1)} y1={T} y2={B} />
          {label && (
            <text className="ts-marker-label" x={(xAt(markers.rollbackAt) - 3).toFixed(1)} y={B - 4} textAnchor="end">
              restart {markers.rollbackAt}
            </text>
          )}
        </g>
      )}
      {markers.haltedAt !== null && (
        <g>
          <rect className="ts-halted-zone" x={xAt(markers.haltedAt).toFixed(1)} y={T} width={Math.max(0, R - xAt(markers.haltedAt)).toFixed(1)} height={B - T} />
          <line className="ts-halted" x1={xAt(markers.haltedAt).toFixed(1)} x2={xAt(markers.haltedAt).toFixed(1)} y1={T} y2={B} />
          {label && (
            <text className="ts-marker-label ts-marker-label--halted" x={(xAt(markers.haltedAt) + 4).toFixed(1)} y={T + 9}>
              halted at step {markers.haltedAt}
            </text>
          )}
        </g>
      )}
    </g>
  );
}

function StepAxis() {
  return (
    <g>
      <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
      <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
      {STEP_TICKS.map((step) => (
        <text key={step} className="ts-tick" x={xAt(step).toFixed(1)} y={B + 11} textAnchor={step === 1 ? "start" : step === PLANNED_STEPS ? "end" : "middle"}>
          {step}
        </text>
      ))}
      <text className="ts-tick" x={(L + R) / 2} y={H - 3} textAnchor="middle">
        optimizer step
      </text>
    </g>
  );
}

function PanelFrame({
  title,
  readout,
  tone,
  ariaLabel,
  caption,
  children,
}: {
  title: string;
  readout: string;
  tone: string;
  ariaLabel: string;
  caption: string;
  children: ReactNode;
}) {
  return (
    <figure className={`ts-panel ts-tone--${tone}`}>
      <div className="ts-panel__head">
        <h4>{title}</h4>
        <span>{readout}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
        {children}
      </svg>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function BandRect({ top, bottom }: { top: number | null; bottom: number | null }) {
  if (top === null || bottom === null || !Number.isFinite(top) || !Number.isFinite(bottom)) return null;
  const y = Math.min(top, bottom);
  return <rect className="ts-band" x={L} y={y.toFixed(1)} width={R - L} height={Math.max(1, Math.abs(bottom - top)).toFixed(1)} />;
}

function LossPanel({
  points,
  corpus,
  referenceCorpus,
  originalCorpus,
  band,
  markers,
}: {
  points: ReadonlyArray<StepPoint>;
  corpus: ReadonlyArray<{ step: number; loss: number }>;
  referenceCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  originalCorpus: ReadonlyArray<{ step: number; loss: number }> | null;
  band: { low: number; high: number } | null;
  markers: Markers;
}) {
  const visible = [
    ...values(points, "loss"),
    ...corpus.map((point) => point.loss),
    ...(referenceCorpus ?? []).map((point) => point.loss),
    ...(originalCorpus ?? []).map((point) => point.loss),
  ];
  const top = Math.min(28, Math.max(4, Math.ceil(Math.max(0, ...visible))));
  const yAt = (value: number) => B - (Math.min(top, Math.max(0, value)) / top) * (B - T);
  const ticks = [0, top / 2, top];
  const last = corpus.at(-1);
  const corpusLine = (list: ReadonlyArray<{ step: number; loss: number }>) =>
    linePath(
      list.filter((point) => point.step > 0).map((point) => ({ step: point.step, value: point.loss })),
      yAt,
    );
  return (
    <PanelFrame
      title="Loss"
      tone="loss"
      readout={last ? `corpus ${formatLoss(last.loss)} at step ${last.step}` : "no finite loss"}
      ariaLabel={`Loss at each optimizer step. The thin line is each batch's loss, the thick line the whole-corpus loss every 4 steps, ${
        last ? `ending at ${formatLoss(last.loss)} at step ${last.step}` : "with no finite value"
      }. The dotted line is ${UNIFORM_CROSS_ENTROPY.toFixed(3)}, a model that knows nothing.`}
      caption={`Thin: each batch's own loss. Thick: whole-corpus loss every 4 steps. Dotted: ln 30 = ${UNIFORM_CROSS_ENTROPY.toFixed(3)}, a model that knows nothing. Values above ${top} are pinned to the top.`}
    >
      {band && <BandRect top={yAt(band.high)} bottom={yAt(band.low)} />}
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="ts-grid" x1={L} x2={R} y1={yAt(tick).toFixed(1)} y2={yAt(tick).toFixed(1)} />
          <text className="ts-tick" x={L - 4} y={(yAt(tick) + 3).toFixed(1)} textAnchor="end">
            {Number.isInteger(tick) ? tick : tick.toFixed(1)}
          </text>
        </g>
      ))}
      <line className="ts-chance" x1={L} x2={R} y1={yAt(UNIFORM_CROSS_ENTROPY).toFixed(1)} y2={yAt(UNIFORM_CROSS_ENTROPY).toFixed(1)} />
      <MarkerLayer markers={markers} label />
      <StepAxis />
      <path className="ts-line ts-line--batch" d={linePath(points.map((point) => ({ step: point.step, value: point.loss })), yAt)} />
      {referenceCorpus && <path className="ts-line ts-line--reference" d={corpusLine(referenceCorpus)} />}
      {originalCorpus && <path className="ts-line ts-line--original" d={corpusLine(originalCorpus)} />}
      <path className="ts-line ts-line--corpus" d={corpusLine(corpus)} />
    </PanelFrame>
  );
}

function LogPanel({
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
  const list = values(points, signal);
  const domain: LogDomain = logDomain([...list, ...(band ? [band.low, band.high] : []), ...guides.map((guide) => guide.value)], {
    floor,
    minDecades: 2,
  });
  const yAt = (value: number) => {
    const position = logPosition(value, domain);
    return position === null ? null : B - position * (B - T);
  };
  const ticks = decadeTicks(domain, 5);
  const clipped = showClipped
    ? points.filter((point) => point.clipped && point[signal] !== null).map((point) => ({ step: point.step, value: point[signal] as number }))
    : [];
  return (
    <PanelFrame
      title={title}
      tone={tone}
      readout={readout}
      ariaLabel={`${title} at each optimizer step on a log scale from ${powerOfTen(domain.low)} to ${powerOfTen(domain.high)}. ${readout}.`}
      caption={caption}
    >
      {band && <BandRect top={yAt(band.high)} bottom={yAt(band.low)} />}
      {ticks.map((exponent) => {
        const y = yAt(10 ** exponent) ?? B;
        return (
          <g key={exponent}>
            <line className="ts-grid" x1={L} x2={R} y1={y.toFixed(1)} y2={y.toFixed(1)} />
            <text className="ts-tick" x={L - 4} y={(y + 3).toFixed(1)} textAnchor="end">
              {powerOfTen(exponent)}
            </text>
          </g>
        );
      })}
      {guides.map((guide) => {
        const y = yAt(guide.value);
        if (y === null) return null;
        return (
          <g key={guide.label}>
            <line className={`ts-guide ts-guide--${guide.kind}`} x1={L} x2={R} y1={y.toFixed(1)} y2={y.toFixed(1)} />
            <text className={`ts-guide-label ts-guide-label--${guide.kind}`} x={R - 2} y={(y - 3).toFixed(1)} textAnchor="end">
              {guide.label}
            </text>
          </g>
        );
      })}
      <MarkerLayer markers={markers} label={false} />
      <StepAxis />
      <path className="ts-line ts-line--signal" d={linePath(points.map((point) => ({ step: point.step, value: point[signal] })), yAt)} />
      {clipped.length > 0 && <path className="ts-dots ts-dots--clipped" d={dotPath(clipped, yAt)} />}
      {showOverflow &&
        markers.overflow.map((step) => (
          <path
            key={`overflow-${step}`}
            className="ts-overflow"
            d={`M${(xAt(step) - 3.5).toFixed(1)},${T}L${(xAt(step) + 3.5).toFixed(1)},${T}L${xAt(step).toFixed(1)},${T + 6}Z`}
          />
        ))}
    </PanelFrame>
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
/* Number formats                                                              */
/* -------------------------------------------------------------------------- */

const F_W = 320;
const F_H = 118;
const F_L = 60;
const F_R = 312;
const F_LOW = -46;
const F_HIGH = 40;
const fxAt = (value: number) =>
  F_L + ((Math.min(F_HIGH, Math.max(F_LOW, Math.log10(Math.max(1e-300, value)))) - F_LOW) / (F_HIGH - F_LOW)) * (F_R - F_L);

function FormatLine({ flushBelow, overflowFrom, norms }: { flushBelow: number; overflowFrom: number; norms: ReadonlyArray<number> }) {
  const rows = NUMBER_FORMATS.map((format, index) => ({ format, y: 14 + index * 18 }));
  const windowY = 14 + rows.length * 18;
  const normLow = norms.length ? Math.min(...norms) : null;
  const normHigh = norms.length ? Math.max(...norms) : null;
  return (
    <svg
      className="ts-format-line"
      viewBox={`0 0 ${F_W} ${F_H}`}
      role="img"
      aria-label={`Positive ranges of three number formats on a log scale from ${powerOfTen(F_LOW)} to ${powerOfTen(
        F_HIGH,
      )}. fp32 and bf16 reach about 3.4 × 10³⁸; fp16 stops at 65,504. At this loss scale a gradient entry survives the fp16 cast between ${formatNumber(
        flushBelow,
        1,
      )} and ${formatNumber(overflowFrom, 2)}.`}
    >
      {[-40, -20, 0, 20, 40].map((exponent) => (
        <g key={exponent}>
          <line className="ts-grid" x1={fxAt(10 ** exponent).toFixed(1)} x2={fxAt(10 ** exponent).toFixed(1)} y1={6} y2={windowY + 10} />
          <text className="ts-tick" x={fxAt(10 ** exponent).toFixed(1)} y={F_H - 14} textAnchor="middle">
            {powerOfTen(exponent)}
          </text>
        </g>
      ))}
      {rows.map(({ format, y }) => (
        <g key={format.id}>
          <text className="ts-format-label" x={F_L - 6} y={y + 4} textAnchor="end">
            {format.label}
          </text>
          <rect
            className="ts-format-sub"
            x={fxAt(format.smallestSubnormal).toFixed(1)}
            y={y - 4}
            width={Math.max(1, fxAt(format.smallestNormal) - fxAt(format.smallestSubnormal)).toFixed(1)}
            height={8}
          />
          <rect
            className={`ts-format-bar ts-format-bar--${format.id}`}
            x={fxAt(format.smallestNormal).toFixed(1)}
            y={y - 4}
            width={Math.max(1, fxAt(format.largest) - fxAt(format.smallestNormal)).toFixed(1)}
            height={8}
          />
        </g>
      ))}
      <text className="ts-format-label" x={F_L - 6} y={windowY + 4} textAnchor="end">
        survives
      </text>
      <rect
        className="ts-format-window"
        x={fxAt(flushBelow).toFixed(1)}
        y={windowY - 4}
        width={Math.max(1, fxAt(overflowFrom) - fxAt(flushBelow)).toFixed(1)}
        height={8}
      />
      {normLow !== null && normHigh !== null && (
        <rect
          className="ts-format-norms"
          x={(fxAt(normLow) - 1).toFixed(1)}
          y={windowY - 7}
          width={Math.max(2, fxAt(normHigh) - fxAt(normLow) + 2).toFixed(1)}
          height={14}
        />
      )}
      <text className="ts-tick" x={(F_L + F_R) / 2} y={F_H - 2} textAnchor="middle">
        magnitude, log scale (lighter: subnormal)
      </text>
    </svg>
  );
}
