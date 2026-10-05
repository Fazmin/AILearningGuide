import { useMemo } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  scheduleFactor,
  SegmentedControl,
  SurfaceHeading,
  TINY_SCHEDULES,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
  type TinySchedule,
} from "@app/module-sdk";
import {
  DEPTH_CLIP_MAX,
  DEPTH_CLIP_MIN,
  DEPTH_GAIN_MAX,
  DEPTH_GAIN_MIN,
  DEPTH_LAYERS_MAX,
  DEPTH_LAYERS_MIN,
  depthReport,
  formatMagnitude,
} from "./depth";
import {
  EPOCHS,
  SEED_SET,
  seedVerdict,
  summarizeSeeds,
  trainRecipe,
  type Comparison,
  type SeedSummary,
} from "./seeds";
import { readState } from "./state";

const GRADIENT_BOUND = Math.SQRT2;
const tones = ["forward", "attention", "gradient", "loss"] as const;
const dashes = ["", "7 4", "1.6 3.4", "10 3 2 3"] as const;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Where each schedule's multiplier ends, read from the formula itself. */
const finalShare = (schedule: TinySchedule) => scheduleFactor(schedule, 1);
const usesWarmup = (schedule: TinySchedule) => schedule === "warmup-decay" || schedule === "one-cycle";

/* Shared chart frame, in viewBox units. */
const W = 560;
const L = 52;
const R = 540;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const {
    schedule: scheduleValue,
    peakLearningRate,
    warmup,
    clipNorm,
    weightDecay,
    batchSize,
    inspect,
    seeds,
    depthLayers,
    depthGain,
    depthClip,
  } = readState(state);
  const schedule = scheduleValue as TinySchedule;

  const runs = useMemo(
    () =>
      TINY_SCHEDULES.map((entry) => ({
        schedule: entry,
        // Seed 1, with a history point at every optimizer step, so the gradient trace misses nothing.
        run: trainRecipe({ peakLearningRate, warmup, clipNorm, weightDecay, batchSize }, entry.value, 1, true),
      })),
    [batchSize, clipNorm, peakLearningRate, warmup, weightDecay],
  );

  // Five seeds retrain the other four seeds of every schedule, so this runs only when asked for.
  const seedSummaries = useMemo<SeedSummary[] | null>(
    () =>
      seeds === "five"
        ? runs.map((entry) =>
            summarizeSeeds([
              entry.run,
              ...SEED_SET.slice(1).map((seed) =>
                trainRecipe({ peakLearningRate, warmup, clipNorm, weightDecay, batchSize }, entry.schedule.value, seed, false),
              ),
            ]),
          )
        : null,
    [seeds, runs, batchSize, clipNorm, peakLearningRate, warmup, weightDecay],
  );
  const verdict = useMemo(
    () =>
      seedSummaries
        ? seedVerdict(runs.map((entry, index) => ({ label: entry.schedule.label, summary: seedSummaries[index] })))
        : null,
    [seedSummaries, runs],
  );

  const selectedIndex = Math.max(0, runs.findIndex((entry) => entry.schedule.value === schedule));
  const selected = runs[selectedIndex];
  const steps = selected.run.steps;
  const stepsPerEpoch = Math.max(1, Math.round(steps / EPOCHS));
  const inspectStep = inspect * stepsPerEpoch;
  const inspectProgress = steps <= 1 ? 1 : Math.min(1, inspectStep / (steps - 1));
  const inspectRate = peakLearningRate * scheduleFactor(schedule, inspectProgress, warmup);

  /** What each schedule is ranked by: its seed-1 final loss, or the mean of five seeds when that is on. */
  const rankValue = (index: number) => (seedSummaries ? seedSummaries[index].mean : runs[index].run.finalLoss);
  const finiteIndexes = runs.map((_, index) => index).filter((index) => Number.isFinite(rankValue(index)));
  const bestIndex = finiteIndexes.reduce(
    (winner, index) => (rankValue(index) < rankValue(winner) ? index : winner),
    finiteIndexes[0] ?? 0,
  );
  const best = runs[bestIndex];
  const bestValue = rankValue(bestIndex);
  /** A run that ends worse than a model which learned nothing is unstable. */
  const unstable = runs.filter(
    (entry) => !Number.isFinite(entry.run.finalLoss) || entry.run.finalLoss > UNIFORM_CROSS_ENTROPY,
  );
  const peakGradient = Math.max(...runs.map((entry) => entry.run.peakGradientNorm));
  const lossAt = (entryIndex: number, epoch: number) => runs[entryIndex].run.checkpoints[epoch]?.loss ?? Number.NaN;

  const lossValues = [
    ...runs.flatMap((entry) => entry.run.checkpoints.map((point) => point.loss)),
    ...(seedSummaries ?? []).flatMap((summary) => summary.band.flatMap((point) => [point.low, point.high])),
  ].filter(Number.isFinite);
  const lossTop = clamp(Math.max(...lossValues) + 0.15, UNIFORM_CROSS_ENTROPY + 0.2, 7.5);
  const lossBottom = Math.min(1.5, Math.floor((Math.min(...lossValues) - 0.1) * 10) / 10);

  const scheduleLabel = TINY_SCHEDULES[selectedIndex].label;
  const warmupSteps = Math.round(warmup * (steps - 1));
  const depth = depthReport(depthLayers, depthGain, depthClip);
  const middleLayer = Math.ceil(depthLayers / 2);
  const depthRows = [
    { name: `Output layer (${depthLayers})`, raw: depth.norms[depthLayers - 1], clipped: depth.clipped[depthLayers - 1] },
    { name: `Middle layer (${middleLayer})`, raw: depth.norms[middleLayer - 1], clipped: depth.clipped[middleLayer - 1] },
    { name: "Input layer (1)", raw: depth.norms[0], clipped: depth.clipped[0] },
    { name: "Whole gradient", raw: depth.total, clipped: depth.total * depth.scale },
  ];

  return (
    <div className="tg-lab tg-lab--hero a10-lab">
      <LabSurface label="Learning-rate schedule shapes" className="tg-schedule-panel">
        <SurfaceHeading
          kicker="Schedule editor"
          title="The step size is a function of how far through the run you are"
          aside={<span className="tg-badge">peak {peakLearningRate.toFixed(2)}</span>}
        />
        <SegmentedControl
          label="Schedule"
          value={schedule}
          options={TINY_SCHEDULES.map((entry) => ({ value: entry.value, label: entry.label }))}
          onChange={(value) => {
            setState({ schedule: value });
            narrate(
              `${TINY_SCHEDULES.find((entry) => entry.value === value)?.detail ?? ""} Final loss ${
                runs.find((entry) => entry.schedule.value === value)?.run.finalLoss.toFixed(3) ?? "—"
              }.`,
            );
          }}
        />
        <ScheduleChart
          schedule={schedule}
          peak={peakLearningRate}
          warmup={warmup}
          steps={steps}
          inspect={inspect}
          inspectRate={inspectRate}
          inspectStep={inspectStep}
        />
        <div className="hp-inspect-control">
          <RangeControl
            label="Inspect epoch"
            min={0}
            max={EPOCHS}
            step={1}
            value={inspect}
            format={(value) => `epoch ${value} · step ${value * stepsPerEpoch}`}
            onChange={(value) => {
              setState({ inspect: value });
              narrate(
                `Epoch ${value}. ${scheduleLabel} learning rate ${(
                  peakLearningRate *
                  scheduleFactor(schedule, Math.min(1, (value * stepsPerEpoch) / Math.max(1, steps - 1)), warmup)
                ).toFixed(3)}.`,
              );
            }}
          />
        </div>
        <p className="lab-note">
          {TINY_SCHEDULES[selectedIndex].detail}{" "}
          {schedule === "constant"
            ? "Warmup fraction does not apply."
            : schedule === "cosine"
              ? "It ignores Warmup fraction and ends at 3% of the peak."
              : `Warmup lasts ${warmupSteps} of ${steps} steps, then the rate falls to ${(finalShare(schedule) * 100).toFixed(0)}% of the peak.`}
        </p>
      </LabSurface>

      <LabSurface label="Loss for all four schedules" className="tg-compare-panel">
        <SurfaceHeading
          kicker={`Four real runs, ${EPOCHS} epochs each · ${steps} steps`}
          title={seeds === "five" ? "Same data, same peak rate, five seeds" : "Same data, same seed, same peak rate"}
          aside={<span className="tg-badge">batch {batchSize}</span>}
        />
        <div className="hp-seeds-control">
          <SegmentedControl
            label="Seeds"
            value={seeds}
            options={[
              { value: "one", label: "Seed 1 only" },
              { value: "five", label: "Five seeds" },
            ]}
            onChange={(value) => {
              setState({ seeds: value });
              narrate(
                value === "five"
                  ? "Each schedule now trains with five shuffle seeds. The shaded band is the lowest to the highest loss across them."
                  : "Back to the single shuffle seed.",
              );
            }}
          />
        </div>
        <LossChart
          runs={runs.map((entry) => entry.run.checkpoints.map((point) => point.loss))}
          labels={runs.map((entry) => entry.schedule.label)}
          selected={selectedIndex}
          inspect={inspect}
          domain={[lossBottom, lossTop]}
          bands={seedSummaries ? seedSummaries.map((summary) => summary.band) : null}
        />
        <div className="hp-legend">
          {runs.map((entry, index) => (
            <span key={entry.schedule.value} className={`hp-legend__item hp-tone--${tones[index]}${index === selectedIndex ? " is-selected" : ""}`}>
              <svg viewBox="0 0 22 6" aria-hidden="true">
                <line x1="1" x2="21" y1="3" y2="3" strokeDasharray={dashes[index]} />
              </svg>
              {entry.schedule.label}
              <b>
                ep {inspect}: {Number.isFinite(lossAt(index, inspect)) ? lossAt(index, inspect).toFixed(3) : "—"} · final{" "}
                {Number.isFinite(entry.run.finalLoss) ? entry.run.finalLoss.toFixed(3) : "diverged"}
                {seedSummaries && Number.isFinite(seedSummaries[index].spread)
                  ? ` · seeds ${seedSummaries[index].min.toFixed(3)}–${seedSummaries[index].max.toFixed(3)}`
                  : ""}
              </b>
            </span>
          ))}
        </div>
        <p className="lab-note">
          Each point is the cross-entropy over the whole corpus at an epoch boundary, measured on the saved
          weights, so the last point of each curve is its final loss. The dotted line at{" "}
          {UNIFORM_CROSS_ENTROPY.toFixed(3)} is a model that learned nothing; values above {lossTop.toFixed(1)} are
          pinned to the top edge.
          {seedSummaries
            ? " Each shaded band runs from the lowest to the highest loss that schedule reached across the five seeds at that epoch. The line is seed 1. Five seeds retrain twenty runs, so the sliders respond more slowly."
            : " Seed 1 is the only shuffle order here; switch Seeds to Five seeds to see how much the order alone moves each curve."}
        </p>
        {seedSummaries && verdict && (
          <SeedTable
            labels={runs.map((entry) => entry.schedule.label)}
            summaries={seedSummaries}
            selected={selectedIndex}
            verdict={verdict}
          />
        )}
        <div className="metric-row">
          <Metric
            label={`${selected.schedule.label} final loss`}
            value={Number.isFinite(selected.run.finalLoss) ? `${selected.run.finalLoss.toFixed(3)} nats` : "diverged"}
            tone="loss"
          />
          <Metric label="Best schedule here" value={best.schedule.label} tone="forward" />
          <Metric
            label={seedSummaries ? "Best mean of five seeds" : "Best final loss"}
            value={Number.isFinite(bestValue) ? `${bestValue.toFixed(3)} nats` : "diverged"}
            tone="forward"
          />
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Optimizer controls" className="tg-hyper-panel">
          <SurfaceHeading kicker="Hyperparameters" title="Five dials, one of them dangerous" />
          <RangeControl
            label="Peak learning rate"
            min={0.1}
            max={60}
            step={0.1}
            value={peakLearningRate}
            format={(value) => value.toFixed(2)}
            onChange={(value) => {
              setState({ peakLearningRate: value });
              narrate(`Peak learning rate ${value.toFixed(2)}.`);
            }}
          />
          <RangeControl
            label="Warmup fraction"
            min={0.02}
            max={0.6}
            step={0.01}
            value={warmup}
            format={(value) => `${Math.round(value * 100)}% of the run`}
            onChange={(value) => setState({ warmup: value })}
          />
          <RangeControl
            label="Gradient clip"
            // Row gradients are bounded by √2, so a threshold above ~1.4 could
            // never bind. The slider stops just past the bound to stay useful.
            min={0}
            max={1.5}
            step={0.05}
            value={clipNorm}
            format={(value) => (value === 0 ? "off" : `norm ≤ ${value.toFixed(2)}`)}
            onChange={(value) => {
              setState({ clipNorm: value });
              narrate(value === 0 ? "Gradient clipping off." : `Clipping batch gradients to norm ${value.toFixed(2)}.`);
            }}
          />
          <RangeControl
            label="Weight decay"
            min={0}
            max={0.08}
            step={0.002}
            value={weightDecay}
            format={(value) => (value === 0 ? "off" : value.toFixed(3))}
            onChange={(value) => setState({ weightDecay: value })}
          />
          <RangeControl
            label="Batch size"
            min={2}
            max={64}
            step={2}
            value={batchSize}
            format={(value) => `${value} pairs`}
            onChange={(value) => setState({ batchSize: value })}
          />
        </LabSurface>

        <LabSurface label="Stability readout" className="tg-stability-panel">
          <SurfaceHeading kicker="Stability" title="Which runs are still learning" />
          <GradientTrace
            points={selected.run.history.map((point) => ({ step: point.step, norm: point.gradientNorm }))}
            steps={steps}
            clip={clipNorm}
            label={scheduleLabel}
          />
          <div className="metric-row">
            <Metric label="Peak gradient norm" value={Number.isFinite(peakGradient) ? peakGradient.toFixed(3) : "diverged"} tone="gradient" />
            <Metric
              label="Clip"
              value={
                clipNorm === 0
                  ? "off"
                  : peakGradient > clipNorm
                    ? `${clipNorm.toFixed(2)} · binding`
                    : `${clipNorm.toFixed(2)} · never binds`
              }
              tone={clipNorm > 0 && peakGradient > clipNorm ? "forward" : undefined}
            />
            <Metric
              label="Worse than chance"
              value={`${unstable.length} / ${runs.length}`}
              tone={unstable.length > 0 ? "loss" : undefined}
            />
          </div>
          <div className={`tg-callout ${unstable.length > 0 ? "is-warning" : "is-quiet"}`}>
            <strong>
              {unstable.length > 0
                ? `${unstable.map((entry) => entry.schedule.label).join(" and ")} ${unstable.length === 1 ? "is" : "are"} now worse than a model that learned nothing.`
                : peakLearningRate > 6
                  ? "Every run survives, but the decaying schedules are pulling ahead."
                  : "All four runs are stable at this peak rate."}
            </strong>
            <span>
              {unstable.length > 0
                ? "The steps are longer than the local slope justifies, so each update overshoots. Decaying the rate fixes it by shrinking steps over time; clipping helps by capping how far any single step can travel."
                : "Past a peak of about 6 the constant schedule's final loss starts rising while the decaying ones keep improving, and between 35 and 38 it crosses the dotted baseline. That gap is the reason schedules exist."}
            </span>
          </div>
          <p className="lab-note">
            Gradients here are bounded. For softmax cross-entropy each row's gradient is a probability vector
            minus a one, so its norm cannot exceed √2 ≈ 1.41, and a clip above that can never bind. This model
            overshoots rather than exploding, so clipping helps most at the low end of the slider and at small
            batch sizes, where single batches are noisiest.
          </p>
        </LabSurface>
      </div>

      <LabSurface label="Gradient through depth" className="hp-depth-card">
        <SurfaceHeading
          kicker="Closed form, not a trained network · every layer multiplies the gradient by the same gain"
          title="A deep stack can explode a gradient, and a clip caps the step"
          aside={<span className="tg-badge">{depthLayers} layers</span>}
        />
        <div className="hp-depth-controls">
          <RangeControl
            label="Layers"
            min={DEPTH_LAYERS_MIN}
            max={DEPTH_LAYERS_MAX}
            step={1}
            value={depthLayers}
            format={(value) => `${value} layers`}
            onChange={(value) => setState({ depthLayers: value })}
          />
          <RangeControl
            label="Gain per layer"
            min={DEPTH_GAIN_MIN}
            max={DEPTH_GAIN_MAX}
            step={0.01}
            value={depthGain}
            format={(value) => `×${value.toFixed(2)}${value > 1 ? " · grows" : value < 1 ? " · shrinks" : ""}`}
            onChange={(value) => {
              setState({ depthGain: Math.round(value * 100) / 100 });
              narrate(`Gain ${value.toFixed(2)} per layer over ${depthLayers} layers.`);
            }}
          />
          <RangeControl
            label="Depth clip"
            min={DEPTH_CLIP_MIN}
            max={DEPTH_CLIP_MAX}
            step={0.5}
            value={depthClip}
            format={(value) => (value === 0 ? "off" : `total ≤ ${value.toFixed(1)}`)}
            onChange={(value) => {
              setState({ depthClip: value });
              narrate(value === 0 ? "Depth clip off." : `Depth clip ${value.toFixed(1)}.`);
            }}
          />
        </div>
        <DepthChart report={depth} />
        <div className="hp-table-wrap">
          <table className="hp-depth-table">
            <caption>
              Gradient length at three layers and over the whole stack, before and after the clip. The output layer
              starts at 1.
            </caption>
            <thead>
              <tr>
                <th scope="col">Where</th>
                <th scope="col">Before the clip</th>
                <th scope="col">After the clip</th>
              </tr>
            </thead>
            <tbody>
              {depthRows.map((row) => (
                <tr key={row.name}>
                  <th scope="row">{row.name}</th>
                  <td>{formatMagnitude(row.raw)}</td>
                  <td>{formatMagnitude(row.clipped)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric label="Input ÷ output gradient" value={formatMagnitude(depth.inputToOutput)} tone="gradient" />
          <Metric label="Whole gradient length" value={formatMagnitude(depth.total)} tone="gradient" />
          <Metric
            label="Clip"
            value={
              depthClip === 0
                ? "off"
                : depth.binding
                  ? `binding · scales by ${formatMagnitude(depth.scale)}`
                  : "never binds"
            }
            tone={depth.binding ? "forward" : undefined}
          />
        </div>
        <div className={`tg-callout ${depth.total > 100 ? "is-warning" : "is-quiet"}`}>
          <strong>
            {depth.total > 100 && depthClip === 0
              ? "The gradient has exploded: the whole gradient is more than 100 times as long as the one at the output."
              : depth.total > 100 && depth.binding
                ? "Clipped: the direction is kept and the whole gradient is scaled down to the threshold."
                : depth.inputToOutput < 0.01
                  ? "The gradient has vanished before it reaches the early layers, and a clip cannot bring it back."
                  : "The gradient stays within a few times its starting length."}
          </strong>
          <span>
            A clip only shrinks a gradient that is too long. It does nothing for one that is too short, so vanishing
            needs a different remedy, such as the residual connections and initialization from earlier labs.
          </span>
        </div>
      </LabSurface>
    </div>
  );
}

function ScheduleChart({
  schedule,
  peak,
  warmup,
  steps,
  inspect,
  inspectRate,
  inspectStep,
}: {
  schedule: TinySchedule;
  peak: number;
  warmup: number;
  steps: number;
  inspect: number;
  inspectRate: number;
  inspectStep: number;
}) {
  const H = 232;
  const T = 30;
  const B = 196;
  const top = peak * 1.08;
  const xAt = (progress: number) => L + progress * (R - L);
  const yAt = (rate: number) => B - (clamp(rate, 0, top) / top) * (B - T);
  const curve = (value: TinySchedule) =>
    Array.from({ length: 121 }, (_, tick) => {
      const progress = tick / 120;
      return `${tick === 0 ? "M" : "L"}${xAt(progress).toFixed(1)},${yAt(peak * scheduleFactor(value, progress, warmup)).toFixed(1)}`;
    }).join(" ");
  const warmupEnd = Math.min(0.9, Math.max(0.01, warmup));
  const endShare = finalShare(schedule);
  const inspectX = xAt(Math.min(1, inspect / EPOCHS));
  const endText = `${(endShare * 100).toFixed(0)}% of peak (${(peak * endShare).toFixed(peak * endShare < 1 ? 3 : 2)})`;
  const phases = usesWarmup(schedule)
    ? [
        {
          from: 0,
          to: warmupEnd,
          label: `warmup${schedule === "one-cycle" ? " from 10%" : ""} · ${Math.round(warmupEnd * (steps - 1))} steps`,
          kind: "warmup",
        },
        {
          from: warmupEnd,
          to: 1,
          label: `${schedule === "one-cycle" ? "cosine" : "linear"} decay to ${endText}`,
          kind: "decay",
        },
      ]
    : schedule === "cosine"
      ? [{ from: 0, to: 1, label: `cosine decay from the first step to ${endText}`, kind: "decay" }]
      : [{ from: 0, to: 1, label: "constant at the peak for every step", kind: "flat" }];

  return (
    <svg
      className="hp-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Learning rate against epoch for all four schedules at peak ${peak.toFixed(2)}. ${
        TINY_SCHEDULES.find((entry) => entry.value === schedule)?.label
      } is highlighted${usesWarmup(schedule) ? `, warming up over ${Math.round(warmupEnd * 100)}% of the run` : ""}, ending at ${(
        endShare * 100
      ).toFixed(0)}% of the peak. At epoch ${inspect}, step ${inspectStep}, its rate is ${inspectRate.toFixed(3)}.`}
    >
      {phases.map((phase, index) => (
        <g key={phase.kind}>
          <rect
            className={`hp-phase hp-phase--${phase.kind}`}
            x={xAt(phase.from)}
            y={T}
            width={Math.max(0, xAt(phase.to) - xAt(phase.from))}
            height={B - T}
          />
          <text
            className="hp-phase-label"
            x={index === phases.length - 1 && phases.length > 1 ? xAt(phase.to) - 2 : xAt(phase.from) + 2}
            y={T - 6}
            textAnchor={index === phases.length - 1 && phases.length > 1 ? "end" : "start"}
          >
            {phase.label}
          </text>
        </g>
      ))}
      {[0, 0.5, 1].map((fraction) => (
        <g key={fraction}>
          <line className="hp-grid" x1={L} x2={R} y1={yAt(peak * fraction)} y2={yAt(peak * fraction)} />
          <text className="hp-tick" x={L - 6} y={yAt(peak * fraction) + 3} textAnchor="end">
            {(peak * fraction).toFixed(peak < 1 ? 2 : 1)}
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((epoch) => (
        <text key={epoch} className="hp-tick" x={xAt(epoch / EPOCHS)} y={B + 13} textAnchor="middle">
          {epoch}
        </text>
      ))}
      <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
      <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
      {TINY_SCHEDULES.map((entry, index) =>
        entry.value === schedule ? null : (
          <path key={entry.value} className="hp-shape is-muted" d={curve(entry.value)} strokeDasharray={dashes[index] || "7 4"} />
        ),
      )}
      <path className="hp-shape is-selected" d={curve(schedule)} />
      <line className="hp-marker" x1={inspectX} x2={inspectX} y1={T} y2={B} />
      <circle className="hp-dot" cx={inspectX} cy={yAt(inspectRate)} r={4.2} />
      <text
        className="hp-dot-label"
        x={inspectX + (inspect > EPOCHS * 0.7 ? -8 : 8)}
        y={Math.max(T + 12, yAt(inspectRate) - 8)}
        textAnchor={inspect > EPOCHS * 0.7 ? "end" : "start"}
      >
        lr {inspectRate.toFixed(3)} at step {inspectStep}
      </text>
      <text className="hp-axis-label" x={L} y={H - 4}>
        epoch (progress through the run)
      </text>
      <text className="hp-axis-label" x={L - 44} y={12}>
        learning rate
      </text>
    </svg>
  );
}

function LossChart({
  runs,
  labels,
  selected,
  inspect,
  domain,
  bands,
}: {
  runs: ReadonlyArray<ReadonlyArray<number>>;
  labels: ReadonlyArray<string>;
  selected: number;
  inspect: number;
  domain: [number, number];
  /** Per schedule, the lowest and highest loss across the five seeds at each epoch; null when seeds are off. */
  bands: ReadonlyArray<ReadonlyArray<{ low: number; high: number }>> | null;
}) {
  const H = 254;
  const T = 24;
  const B = 226;
  const LW = 430;
  const LL = 40;
  const LR = 424;
  const [low, high] = domain;
  const xAt = (epoch: number) => LL + (epoch / EPOCHS) * (LR - LL);
  const yAt = (loss: number) =>
    B - ((clamp(Number.isFinite(loss) ? loss : high, low, high) - low) / (high - low)) * (B - T);
  const ticks = [low, (low + high) / 2, high];
  const order = runs.map((_, index) => index).filter((index) => index !== selected).concat(selected);
  return (
    <svg
      className="hp-chart hp-chart--loss"
      viewBox={`0 0 ${LW} ${H}`}
      role="img"
      aria-label={`Whole-corpus training loss at each epoch for four schedules. ${labels
        .map((label, index) => `${label}: ${runs[index][inspect]?.toFixed(3) ?? "—"} at epoch ${inspect}, ${runs[index].at(-1)?.toFixed(3) ?? "—"} at the end`)
        .join("; ")}. Uniform baseline ${UNIFORM_CROSS_ENTROPY.toFixed(3)}.${
        bands
          ? ` Shaded bands show the range across five seeds; at the last epoch ${labels
              .map((label, index) => {
                const last = bands[index].at(-1);
                return last && Number.isFinite(last.low) ? `${label} ${last.low.toFixed(3)} to ${last.high.toFixed(3)}` : `${label} diverged`;
              })
              .join("; ")}.`
          : ""
      }`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="hp-grid" x1={LL} x2={LR} y1={yAt(tick)} y2={yAt(tick)} />
          <text className="hp-tick" x={LL - 5} y={yAt(tick) + 3} textAnchor="end">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((epoch) => (
        <text key={epoch} className="hp-tick" x={xAt(epoch)} y={B + 13} textAnchor="middle">
          {epoch}
        </text>
      ))}
      <line className="plot-axis" x1={LL} x2={LR} y1={B} y2={B} />
      <line className="plot-axis" x1={LL} x2={LL} y1={T} y2={B} />
      <line className="hp-baseline" x1={LL} x2={LR} y1={yAt(UNIFORM_CROSS_ENTROPY)} y2={yAt(UNIFORM_CROSS_ENTROPY)} />
      <text className="hp-baseline-label" x={LR - 2} y={yAt(UNIFORM_CROSS_ENTROPY) - 4} textAnchor="end">
        knows nothing · ln 30 = {UNIFORM_CROSS_ENTROPY.toFixed(3)}
      </text>
      <line className="hp-marker" x1={xAt(inspect)} x2={xAt(inspect)} y1={T} y2={B} />
      {bands &&
        order.map((index) => {
          const band = bands[index];
          const upper = band.map((point, epoch) => `${epoch === 0 ? "M" : "L"}${xAt(epoch).toFixed(1)},${yAt(point.high).toFixed(1)}`);
          const lower = band
            .map((_, epoch) => band.length - 1 - epoch)
            .map((epoch) => `L${xAt(epoch).toFixed(1)},${yAt(band[epoch].low).toFixed(1)}`);
          return (
            <path
              key={`band-${labels[index]}`}
              className={`hp-band hp-tone--${tones[index]}${index === selected ? " is-selected" : ""}`}
              d={`${upper.join(" ")} ${lower.join(" ")} Z`}
            />
          );
        })}
      {order.map((index) => (
        <g key={labels[index]} className={`hp-series hp-tone--${tones[index]}${index === selected ? " is-selected" : ""}`}>
          <path
            d={runs[index].map((loss, epoch) => `${epoch === 0 ? "M" : "L"}${xAt(epoch).toFixed(1)},${yAt(loss).toFixed(1)}`).join(" ")}
            strokeDasharray={dashes[index]}
          />
          <circle cx={xAt(inspect)} cy={yAt(runs[index][inspect])} r={index === selected ? 4.2 : 3} />
        </g>
      ))}
      <text className="hp-axis-label" x={LL} y={H - 4}>
        epoch
      </text>
      <text className="hp-axis-label" x={4} y={10}>
        loss (nats/token)
      </text>
    </svg>
  );
}

function GradientTrace({
  points,
  steps,
  clip,
  label,
}: {
  points: ReadonlyArray<{ step: number; norm: number }>;
  steps: number;
  clip: number;
  label: string;
}) {
  const w = 260;
  const h = 124;
  const left = 26;
  const right = 256;
  const T = 10;
  const B = 100;
  const top = 1.5;
  const xAt = (step: number) => left + (step / Math.max(1, steps)) * (right - left);
  const yAt = (norm: number) => B - (clamp(norm, 0, top) / top) * (B - T);
  const above = clip > 0 ? points.filter((point) => point.norm > clip).length : 0;
  const dots = (subset: ReadonlyArray<{ step: number; norm: number }>) =>
    subset.map((point) => `M${xAt(point.step).toFixed(1)},${yAt(point.norm).toFixed(1)}h0.01`).join("");
  return (
    <figure className="hp-trace">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`Batch gradient norm at every one of ${points.length} optimizer steps for ${label}. ${
          clip > 0 ? `${above} of them exceed the clip threshold ${clip.toFixed(2)} and are rescaled.` : "Clipping is off."
        } The bound √2 is 1.414.`}
      >
        {[0, 0.5, 1, 1.5].map((tick) => (
          <g key={tick}>
            <line className="hp-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="hp-tick" x={left - 4} y={yAt(tick) + 3} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="hp-bound" x1={left} x2={right} y1={yAt(GRADIENT_BOUND)} y2={yAt(GRADIENT_BOUND)} />
        <text className="hp-bound-label" x={right} y={yAt(GRADIENT_BOUND) - 3} textAnchor="end">
          √2 bound
        </text>
        {clip > 0 && (
          <>
            <line className="hp-clip" x1={left} x2={right} y1={yAt(clip)} y2={yAt(clip)} />
            <text className="hp-clip-label" x={left + 4} y={yAt(clip) - 3}>
              clip {clip.toFixed(2)}
            </text>
          </>
        )}
        <path className="hp-trace-dots" d={dots(points.filter((point) => !(clip > 0 && point.norm > clip)))} />
        <path className="hp-trace-dots is-clipped" d={dots(points.filter((point) => clip > 0 && point.norm > clip))} />
        <text className="hp-axis-label" x={left} y={h - 4}>
          optimizer step · {label}
        </text>
      </svg>
      <figcaption>
        {clip > 0
          ? `The clip binds on ${above} of ${points.length} steps (dark dots): each of those batch gradients is rescaled to norm ${clip.toFixed(2)} before the update.`
          : `Batch-gradient norm at each of ${points.length} steps for ${label}. Set Gradient clip to draw the threshold.`}
      </figcaption>
    </figure>
  );
}

function SeedTable({
  labels,
  summaries,
  selected,
  verdict,
}: {
  labels: ReadonlyArray<string>;
  summaries: ReadonlyArray<SeedSummary>;
  selected: number;
  verdict: { top: Comparison | null; closest: Comparison | null };
}) {
  const say = (comparison: Comparison) =>
    `${comparison.better} is ahead of ${comparison.other} by ${comparison.gap.toFixed(4)} nats on average, against a seed spread of ${comparison.noise.toFixed(
      4,
    )}: ${
      comparison.outsideBand
        ? "the gap is larger than the band, so all five seeds agree on the order."
        : "the gap is inside the band, so a single run cannot tell them apart."
    }`;
  return (
    <div className="hp-seed-table">
      <div className="hp-table-wrap">
        <table>
          <caption>
            Final whole-corpus loss for each of the five seeds, in nats per token. Spread is the highest minus the
            lowest.
          </caption>
          <thead>
            <tr>
              <th scope="col">Schedule</th>
              {SEED_SET.map((seed) => (
                <th key={seed} scope="col">
                  Seed {seed}
                </th>
              ))}
              <th scope="col">Spread</th>
            </tr>
          </thead>
          <tbody>
            {summaries.map((summary, index) => (
              <tr key={labels[index]} className={index === selected ? "is-selected" : undefined}>
                <th scope="row">{labels[index]}</th>
                {summary.finals.map((value, seedIndex) => (
                  <td key={seedIndex}>{Number.isFinite(value) ? value.toFixed(3) : "diverged"}</td>
                ))}
                <td>{Number.isFinite(summary.spread) ? summary.spread.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={`tg-callout ${verdict.closest && !verdict.closest.outsideBand ? "is-warning" : "is-quiet"}`}>
        <strong>
          {verdict.closest && !verdict.closest.outsideBand
            ? "Two schedules here cannot be told apart with one run."
            : "Every pair of schedules is further apart than the seed band."}
        </strong>
        <span>
          {verdict.top ? `Best two: ${say(verdict.top)}` : "Fewer than two schedules finished, so there is nothing to compare."}
          {verdict.closest && verdict.top && verdict.closest.gap !== verdict.top.gap ? ` Closest pair: ${say(verdict.closest)}` : ""}
        </span>
      </div>
    </div>
  );
}

const DEPTH_W = 560;
const DEPTH_H = 240;
const DEPTH_L = 52;
const DEPTH_R = 540;
const DEPTH_T = 22;
const DEPTH_B = 200;
const LOG_LOW = -8;
const LOG_HIGH = 8;

function DepthChart({ report }: { report: ReturnType<typeof depthReport> }) {
  const layers = report.layers;
  const xAt = (layer: number) => DEPTH_L + ((layer - 1) / Math.max(1, layers - 1)) * (DEPTH_R - DEPTH_L);
  const yAt = (norm: number) => {
    const value = norm > 0 ? Math.log10(norm) : LOG_LOW;
    return DEPTH_B - ((clamp(value, LOG_LOW, LOG_HIGH) - LOG_LOW) / (LOG_HIGH - LOG_LOW)) * (DEPTH_B - DEPTH_T);
  };
  const line = (values: ReadonlyArray<number>) =>
    values.map((norm, index) => `${index === 0 ? "M" : "L"}${xAt(index + 1).toFixed(1)},${yAt(norm).toFixed(1)}`).join(" ");
  const input = report.norms[0];
  const output = report.norms[layers - 1];
  return (
    <svg
      className="hp-chart hp-depth-chart"
      viewBox={`0 0 ${DEPTH_W} ${DEPTH_H}`}
      role="img"
      aria-label={`Gradient length at each of ${layers} layers on a log scale, with a gain of ${report.gain.toFixed(
        2,
      )} per layer. The output layer sees ${formatMagnitude(output)} and the input layer ${formatMagnitude(input)}. ${
        report.clip > 0
          ? report.binding
            ? `The total gradient length ${formatMagnitude(report.total)} exceeds the clip ${report.clip.toFixed(1)}, so every layer's gradient is scaled by ${formatMagnitude(report.scale)}; the input layer then sees ${formatMagnitude(report.clipped[0])}.`
            : `The total gradient length ${formatMagnitude(report.total)} is below the clip ${report.clip.toFixed(1)}, so the clip never binds.`
          : "Clipping is off."
      }`}
    >
      {[-8, -4, 0, 4, 8].map((power) => (
        <g key={power}>
          <line className="hp-grid" x1={DEPTH_L} x2={DEPTH_R} y1={yAt(10 ** power)} y2={yAt(10 ** power)} />
          <text className="hp-tick" x={DEPTH_L - 6} y={yAt(10 ** power) + 3} textAnchor="end">
            1e{power}
          </text>
        </g>
      ))}
      <line className="plot-axis" x1={DEPTH_L} x2={DEPTH_R} y1={DEPTH_B} y2={DEPTH_B} />
      <line className="plot-axis" x1={DEPTH_L} x2={DEPTH_L} y1={DEPTH_T} y2={DEPTH_B} />
      {report.clip > 0 && (
        <>
          <line className="hp-clip" x1={DEPTH_L} x2={DEPTH_R} y1={yAt(report.clip)} y2={yAt(report.clip)} />
          <text className="hp-clip-label" x={DEPTH_L + 4} y={yAt(report.clip) - 3}>
            clip {report.clip.toFixed(1)} on the total
          </text>
        </>
      )}
      <path className="hp-depth-line is-raw" d={line(report.norms)} />
      {report.clip > 0 && <path className="hp-depth-line is-clipped" d={line(report.clipped)} />}
      <text className="hp-tick" x={DEPTH_L} y={DEPTH_B + 14}>
        layer 1 (input)
      </text>
      <text className="hp-tick" x={DEPTH_R} y={DEPTH_B + 14} textAnchor="end">
        layer {layers} (output)
      </text>
      <text className="hp-axis-label" x={4} y={12}>
        gradient length, log scale (values beyond 1e±8 pinned to the edge)
      </text>
    </svg>
  );
}
