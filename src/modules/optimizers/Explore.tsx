import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  teachingMemory,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  adamSpec,
  baseSpec,
  CAPTURES,
  DECAY_LADDER,
  EARLY_STEPS,
  EPOCHS,
  formatRate,
  formatSmall,
  idleShrink,
  IDLE_STEPS,
  ladderIndex,
  LAB_PARAMETERS,
  LARGE_PARAMETERS,
  muonSpectra,
  NS_MAX,
  OPTIMIZERS,
  optimizerChoice,
  RATE_FACTORS,
  rateLadder,
  RARE_CHARACTERS,
  rowsKept,
  rowSteps,
  startSweep,
  stateRows,
  subscribeSweep,
  sweepProgress,
  sweepTable,
  SWEEP_FACTORS,
  TOTAL_STEPS,
  trainRun,
  WARMUP_MAX,
  WARMUP_STEP,
  type AdamSettings,
  type OptimizerId,
  type RaceRun,
  type SweepRow,
} from "./race";
import { BAND_HIGH, BAND_LOW, ZERO_SINGULAR, type SpectrumSummary } from "./spectrum";
import { RavineCard } from "./RavineCard";
import { readState } from "./state";

const TONES: Record<OptimizerId, string> = {
  sgd: "forward",
  momentum: "attention",
  adamw: "gradient",
  muon: "loss",
};
/** Every series has its own dash, so no curve is told apart by colour alone. */
const DASHES: Record<OptimizerId, string> = {
  sgd: "",
  momentum: "7 4",
  adamw: "1.6 3.4",
  muon: "10 3 2 3",
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** The rendered width of an element, so a chart in a narrow card is drawn at its real size. */
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
const loss = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "diverged");
const percent = (value: number) => `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;
/** Three decimals down to 0.001, then scientific: 0.543, 0.001, 2.1 × 10⁻⁶. */
const singular = (value: number) => (value >= 0.001 ? value.toFixed(3) : formatSmall(value));

/** The race's name for the Adam-family run: decay in the gradient makes it Adam with L2. */
const adamLabel = (settings: AdamSettings) =>
  settings.decayMode === "l2" && settings.weightDecay > 0 ? "Adam + L2" : "AdamW";

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const settings = readState(state);
  const adam: AdamSettings = {
    rate: settings.adamRate,
    biasCorrection: settings.biasCorrection === "on",
    warmup: settings.warmup,
    weightDecay: settings.weightDecay,
    decayMode: settings.decayMode,
  };
  const { rate: adamRate, biasCorrection, warmup, weightDecay, decayMode } = adam;

  const sgd = useMemo(() => trainRun(baseSpec("sgd", settings.sgdRate)), [settings.sgdRate]);
  const momentum = useMemo(() => trainRun(baseSpec("momentum", settings.momentumRate)), [settings.momentumRate]);
  const adamw = useMemo(
    () => trainRun(adamSpec({ rate: adamRate, biasCorrection, warmup, weightDecay, decayMode })),
    [adamRate, biasCorrection, warmup, weightDecay, decayMode],
  );
  const muon = useMemo(() => trainRun(baseSpec("muon", settings.muonRate)), [settings.muonRate]);
  // The same AdamW run with bias correction flipped, drawn faint on the early-steps chart.
  const counterpart = useMemo(
    () => trainRun(adamSpec({ rate: adamRate, biasCorrection: !biasCorrection, warmup, weightDecay, decayMode })),
    [adamRate, biasCorrection, warmup, weightDecay, decayMode],
  );
  // The same AdamW run with no decay, so each row's size can be read as a share of it.
  const undecayed = useMemo(
    () => trainRun(adamSpec({ rate: adamRate, biasCorrection, warmup, weightDecay: 0, decayMode })),
    [adamRate, biasCorrection, warmup, decayMode],
  );
  const spectra = useMemo(
    () => muonSpectra(settings.muonRate, settings.capture, settings.nsSteps),
    [settings.muonRate, settings.capture, settings.nsSteps],
  );

  // Tune each runs at module scope, one training per macrotask; this only re-renders as runs land.
  const [, setSweepTick] = useState(0);
  useEffect(() => subscribeSweep(() => setSweepTick((tick) => tick + 1)), []);
  const progress = sweepProgress();
  const tuned = sweepTable();

  const runs: Record<OptimizerId, RaceRun> = { sgd, momentum, adamw, muon };
  const rates: Record<OptimizerId, number> = {
    sgd: settings.sgdRate,
    momentum: settings.momentumRate,
    adamw: settings.adamRate,
    muon: settings.muonRate,
  };
  const labelOf = (id: OptimizerId) => (id === "adamw" ? adamLabel(adam) : optimizerChoice(id).label);
  const selected = settings.optimizer;

  const finished = OPTIMIZERS.filter((choice) => Number.isFinite(runs[choice.id].heldOut));
  const lowest = (pickValue: (run: RaceRun) => number) =>
    finished.reduce(
      (winner, choice) => (pickValue(runs[choice.id]) < pickValue(runs[winner.id]) ? choice : winner),
      finished[0] ?? OPTIMIZERS[0],
    );
  const bestHeld = lowest((run) => run.heldOut);
  const bestTrain = lowest((run) => run.trainLoss);
  const heldValues = finished.map((choice) => runs[choice.id].heldOut);
  const heldSpread = heldValues.length ? Math.max(...heldValues) - Math.min(...heldValues) : Number.NaN;

  const rows = rowSteps(runs[selected]);
  const spaceRow = rows.find((row) => row.character === " ");
  const qRow = rows.find((row) => row.character === "q");
  const rareOverSpace = spaceRow && qRow && spaceRow.perVisit > 0 ? qRow.perVisit / spaceRow.perVisit : Number.NaN;

  const kept = rowsKept(adamw, undecayed);
  const shrink = idleShrink(adam);
  const warmupSteps = Math.round(warmup * (TOTAL_STEPS - 1));

  const stateFloats: Record<OptimizerId, number> = {
    sgd: sgd.stateFloats,
    momentum: momentum.stateFloats,
    adamw: adamw.stateFloats,
    muon: muon.stateFloats,
  };
  const memory = stateRows(stateFloats);
  const adamMixed = teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "mixed", "adam");
  const largeBytesOf = (id: OptimizerId) => memory.find((row) => row.optimizer === id)?.largeBytes ?? 0;
  const largestState = Math.max(1, ...memory.map((row) => row.largeBytes));

  return (
    <div className="tg-lab op-lab">
      <LabSurface label="Optimizer race" className="op-wide op-race-card">
        <SurfaceHeading
          kicker={`Four real runs · ${TOTAL_STEPS} steps each · same data, batches and seed`}
          title="Four optimizers, each at its own learning rate"
          aside={<span className="tg-badge">batch 16 · {EPOCHS} epochs</span>}
        />
        <div className="op-rate-grid op-race-controls">
          {OPTIMIZERS.map((choice) => {
            const ladder = rateLadder(choice.id);
            return (
              <RangeControl
                key={choice.id}
                label={choice.rateLabel}
                min={0}
                max={ladder.length - 1}
                step={1}
                value={ladderIndex(rates[choice.id], ladder)}
                format={(index) =>
                  `${formatRate(ladder[index] ?? ladder[0])} · ${
                    RATE_FACTORS[index] === 1 ? "default" : `${RATE_FACTORS[index] ?? 1}× default`
                  }`
                }
                onChange={(index) => {
                  const rate = ladder[clamp(Math.round(index), 0, ladder.length - 1)];
                  setState({ [choice.rateKey]: rate });
                  narrate(`${choice.label} learning rate ${formatRate(rate)}.`);
                }}
              />
            );
          })}
        </div>
        <div className="op-curves-control">
          <SegmentedControl
            label="Curves"
            value={settings.curves}
            options={[
              { value: "train", label: "Training loss" },
              { value: "held", label: "Held-out loss" },
            ]}
            onChange={(value) => {
              setState({ curves: value });
              narrate(value === "held" ? "Showing loss on the held-out sentences." : "Showing loss on the training sentences.");
            }}
          />
        </div>
        <RaceChart runs={runs} labelOf={labelOf} selected={selected} curves={settings.curves} />
        <div className="op-table-wrap">
          <table className="op-table">
            <caption>
              Final loss of each run in nats per character. Training loss is measured on the six training
              sentences, held out on the two sentences no run ever trains on.
            </caption>
            <thead>
              <tr>
                <th scope="col">Optimizer</th>
                <th scope="col">Learning rate</th>
                <th scope="col">Training loss</th>
                <th scope="col">Held out</th>
              </tr>
            </thead>
            <tbody>
              {OPTIMIZERS.map((choice) => (
                <tr key={choice.id} className={choice.id === selected ? "is-selected" : undefined}>
                  <th scope="row">
                    <span className={`op-swatch op-tone--${TONES[choice.id]}`} aria-hidden="true">
                      <svg viewBox="0 0 22 6">
                        <line x1="1" x2="21" y1="3" y2="3" strokeDasharray={DASHES[choice.id]} />
                      </svg>
                    </span>
                    {labelOf(choice.id)}
                    {choice.id === "adamw" && adamNotes(adam) ? <small> · {adamNotes(adam)}</small> : null}
                  </th>
                  <td>{formatRate(rates[choice.id])}</td>
                  <td>{loss(runs[choice.id].trainLoss)}</td>
                  <td className={choice.id === bestHeld.id ? "is-best" : undefined}>{loss(runs[choice.id].heldOut)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric
            label="Lowest held out"
            value={`${labelOf(bestHeld.id)} · ${loss(runs[bestHeld.id].heldOut)}`}
            tone="forward"
          />
          <Metric label="Lowest training loss" value={`${labelOf(bestTrain.id)} · ${loss(runs[bestTrain.id].trainLoss)}`} />
          <Metric label="Held-out spread" value={Number.isFinite(heldSpread) ? heldSpread.toFixed(3) : "—"} tone="loss" />
        </div>
        <p className="lab-note">
          Every run starts from the same zeroed table, sees the same batches in the same order, and stops after{" "}
          {TOTAL_STEPS} steps; only the optimizer and its rate differ. A rate on one optimizer's slider is not
          comparable with another's: AdamW's 0.03 and SGD's 1 are each that optimizer's default. The dotted line at{" "}
          {UNIFORM_CROSS_ENTROPY.toFixed(3)} is a model that learned nothing.
        </p>
      </LabSurface>

      <LabSurface label="A ravine in two weights" className="op-wide op-ravine-card">
        <RavineCard
          turn={settings.ravineTurn}
          rates={settings.ravineRates}
          step={settings.ravineStep}
          onChange={(patch) => setState(patch)}
          narrate={narrate}
        />
      </LabSurface>

      <LabSurface label="Step size per row" className="op-rows-card">
        <SurfaceHeading
          kicker="Measured over the whole run"
          title="How far each row moves per visit"
          aside={
            <span className="tg-badge">
              {labelOf(selected)} at {formatRate(rates[selected])}
            </span>
          }
        />
        <div className="op-rows-control">
          <SegmentedControl
            label="Optimizer"
            value={selected}
            options={OPTIMIZERS.map((choice) => ({ value: choice.id, label: choice.label }))}
            onChange={(value) => {
              setState({ optimizer: value });
              const run = runs[value as OptimizerId];
              const steps = rowSteps(run);
              const space = steps.find((row) => row.character === " ");
              const q = steps.find((row) => row.character === "q");
              narrate(
                `${optimizerChoice(value as OptimizerId).label}: the space row moves ${space?.perVisit.toFixed(3) ?? "—"} per visit, the q row ${
                  q?.perVisit.toFixed(3) ?? "—"
                }.`,
              );
            }}
          />
        </div>
        <BarList
          label={`Distance each row travelled per visit under ${labelOf(selected)}`}
          items={rows.map((row) => ({
            id: row.character,
            label: row.label,
            value: row.perVisit,
            display: row.perVisit.toFixed(3),
            detail: `${row.visits} visits`,
            tone: (RARE_CHARACTERS as ReadonlyArray<string>).includes(row.character) ? "gradient" : "forward",
            emphasis: row.character === " " || row.character === "q",
          }))}
        />
        <div className="metric-row">
          <Metric label="Space row per visit" value={spaceRow ? spaceRow.perVisit.toFixed(3) : "—"} tone="forward" />
          <Metric label="q row per visit" value={qRow ? qRow.perVisit.toFixed(3) : "—"} tone="gradient" />
          <Metric
            label="q ÷ space"
            value={Number.isFinite(rareOverSpace) ? rareOverSpace.toFixed(rareOverSpace < 10 ? 2 : 1) : "—"}
          />
        </div>
        <p className="lab-note">
          Each bar is the total length of every step's change to that row, over all {TOTAL_STEPS} steps, divided by
          the batches that held the row's character. Space starts 70 of the 378 training pairs and q, z and k one or
          two each. SGD moves a row only when it is in the batch; momentum, AdamW and Muon keep moving it for several
          steps afterwards.
        </p>
      </LabSurface>

      <LabSurface label="Bias correction and warmup" className="op-bias-card">
        <SurfaceHeading
          kicker={`${adamLabel(adam)}'s first ${EARLY_STEPS} steps`}
          title="How big the first steps are"
          aside={<span className="tg-badge">AdamW rate {formatRate(adamRate)}</span>}
        />
        <div className="op-bias-controls">
          <SegmentedControl
            label="Bias correction"
            value={settings.biasCorrection}
            options={[
              { value: "on", label: "On" },
              { value: "off", label: "Off" },
            ]}
            onChange={(value) => {
              setState({ biasCorrection: value });
              const run = trainRun(adamSpec({ ...adam, biasCorrection: value === "on" }));
              narrate(
                `Bias correction ${value}. First-step move ${run.firstStepMove.toFixed(3)}, early peak loss ${run.earlyPeakLoss.toFixed(3)}.`,
              );
            }}
          />
          <RangeControl
            label="Warmup"
            min={0}
            max={WARMUP_MAX}
            step={WARMUP_STEP}
            value={warmup}
            format={(value) =>
              value === 0 ? "off" : `${Math.round(value * 100)}% of the run · ${Math.round(value * (TOTAL_STEPS - 1))} steps`
            }
            onChange={(value) => {
              const notch = Number((Math.round(value / WARMUP_STEP) * WARMUP_STEP).toFixed(2));
              setState({ warmup: notch });
              narrate(notch === 0 ? "Warmup off." : `Warmup over ${Math.round(notch * 100)} percent of the run.`);
            }}
          />
        </div>
        <EarlyChart current={adamw} other={counterpart} biasOn={biasCorrection} warmupSteps={warmupSteps} />
        <FirstStepBars current={adamw.firstStepMove} other={counterpart.firstStepMove} biasOn={biasCorrection} rate={adamRate} />
        <div className="metric-row">
          <Metric label="First-step move" value={adamw.firstStepMove.toFixed(3)} tone="gradient" />
          <Metric label="Early peak loss" value={loss(adamw.earlyPeakLoss)} tone="loss" />
          <Metric label="Held out" value={loss(adamw.heldOut)} />
        </div>
        <p className="lab-note">
          First-step move is the largest change any one weight made on step 1; Early peak loss is the highest batch
          loss in steps 1 to {EARLY_STEPS}. With bias correction, Adam's first step is the gradient divided by its own
          size, so every weight with a gradient moves by exactly the rate. Without it the moments start at a tenth
          and a thousandth of their true size, and the first step is 0.1 ÷ √0.001 ≈ 3.16 times the rate.
        </p>
      </LabSurface>

      <LabSurface label="Weight decay" className="op-decay-card">
        <SurfaceHeading
          kicker="The same pull toward zero, applied two ways"
          title="L2 in the gradient or decoupled"
          aside={<span className="tg-badge">{adamLabel(adam)} · rate {formatRate(adamRate)}</span>}
        />
        <div className="op-decay-controls">
          <RangeControl
            label="Weight decay"
            min={0}
            max={DECAY_LADDER.length - 1}
            step={1}
            value={ladderIndex(weightDecay, DECAY_LADDER)}
            format={(index) => {
              const value = DECAY_LADDER[index] ?? 0;
              return value === 0 ? "off" : formatRate(value);
            }}
            onChange={(index) => {
              const value = DECAY_LADDER[clamp(Math.round(index), 0, DECAY_LADDER.length - 1)];
              setState({ weightDecay: value });
              narrate(value === 0 ? "Weight decay off." : `Weight decay ${formatRate(value)}.`);
            }}
          />
          <SegmentedControl
            label="Decay mode"
            value={decayMode}
            options={[
              { value: "l2", label: "L2 in gradient" },
              { value: "decoupled", label: "Decoupled" },
            ]}
            onChange={(value) => {
              setState({ decayMode: value });
              const run = trainRun(adamSpec({ ...adam, decayMode: value === "l2" ? "l2" : "decoupled" }));
              narrate(
                `${value === "l2" ? "L2 in the gradient" : "Decoupled decay"}: weight norm ${run.weightNorm.toFixed(2)}, held out ${loss(run.heldOut)}.`,
              );
            }}
          />
        </div>
        <div className="metric-row">
          <Metric label="Weight norm" value={adamw.weightNorm.toFixed(2)} tone="gradient" />
          <Metric label="Held out" value={loss(adamw.heldOut)} tone="forward" />
          <Metric label="Shrink per step" value={formatSmall(shrink)} tone="loss" />
        </div>
        <div className="op-table-wrap">
          <table className="op-table op-kept-table">
            <caption>
              Each row's size after training, as a share of the same run with no decay. Space is the busiest input
              row; k and q are among the rarest. The bars use a log scale from 0.1% to 100%, with marks at 1% and 10%.
            </caption>
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Visits</th>
                <th scope="col">Size kept</th>
              </tr>
            </thead>
            <tbody>
              {kept.map((row) => (
                <tr key={row.character}>
                  <th scope="row">{row.label}</th>
                  <td>{row.visits}</td>
                  <td>
                    <span className="op-kept-cell">
                      <span className="op-kept-bar" aria-hidden="true">
                        <i style={{ width: `${(keptShare(row.kept) * 100).toFixed(1)}%` }} />
                      </span>
                      <span>{percent(row.kept)}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={`tg-callout ${weightDecay > 0 && decayMode === "l2" ? "is-warning" : "is-quiet"}`}>
          <strong>
            {weightDecay === 0
              ? "Weight decay is off, so the two modes train the same run."
              : decayMode === "l2"
                ? "The pull goes through Adam's division, so it is not the same for every weight."
                : `Every weight shrinks by the same ${percent(adamRate * weightDecay)} each step.`}
          </strong>
          <span>
            Shrink per step follows one weight of 1.0 that has no gradient of its own for {IDLE_STEPS} steps and
            averages its drop: a closed-form illustration, not a measurement of the run. Decoupled decay takes rate ×
            decay of it. L2 hands Adam a gradient of decay × weight, which Adam divides by that gradient's own size, so
            the idle weight moves by about the full rate however small the decay is, while a weight with a large
            gradient of its own barely feels the pull.
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Orthogonalized update" className="op-muon-card">
        <SurfaceHeading
          kicker={`Muon at rate ${formatRate(settings.muonRate)} · one captured step`}
          title="Newton–Schulz pushes every direction toward the same size"
          aside={<span className="tg-badge">{CAPTURES[settings.capture].label}</span>}
        />
        <div className="op-muon-controls">
          <SegmentedControl
            label="Captured step"
            value={settings.capture}
            options={[
              { value: "first", label: CAPTURES.first.label },
              { value: "middle", label: CAPTURES.middle.label },
            ]}
            onChange={(value) => {
              setState({ capture: value });
              narrate(`Showing Muon's update at ${value === "first" ? CAPTURES.first.label : CAPTURES.middle.label}.`);
            }}
          />
          <RangeControl
            label="Newton–Schulz iterations"
            min={0}
            max={NS_MAX}
            step={1}
            value={settings.nsSteps}
            format={(value) => `${value} ${value === 1 ? "iteration" : "iterations"}${value === 5 ? " · Muon's default" : ""}`}
            onChange={(value) => {
              const iterations = clamp(Math.round(value), 0, NS_MAX);
              setState({ nsSteps: iterations });
              const next = muonSpectra(settings.muonRate, settings.capture, iterations);
              narrate(
                next
                  ? `${iterations} iterations: ${next.after.inBand} of ${next.after.nonzero} directions between ${BAND_LOW} and ${BAND_HIGH}.`
                  : `${iterations} iterations.`,
              );
            }}
          />
        </div>
        {spectra ? (
          <>
            <SpectrumChart before={spectra.before} after={spectra.after} iterations={spectra.iterations} />
            <div className="metric-row">
              <Metric label="Directions with gradient" value={`${spectra.before.nonzero} of 30`} tone="forward" />
              <Metric
                label={`Between ${BAND_LOW} and ${BAND_HIGH}`}
                value={`${spectra.after.inBand} of ${spectra.after.nonzero}`}
                tone="gradient"
              />
              <Metric label="Zero directions" value={`${spectra.after.zeros} stay 0`} />
            </div>
            <p className="lab-note">
              Before: the singular values of Muon's momentum direction at {CAPTURES[settings.capture].label}, divided
              by its overall size, which is where Newton–Schulz starts. After: the same matrix after{" "}
              {spectra.iterations} {spectra.iterations === 1 ? "iteration" : "iterations"}. Largest before{" "}
              {singular(spectra.before.largest)}, smallest nonzero before {singular(spectra.before.smallestNonzero)};
              after, they read {singular(spectra.after.largest)} and {singular(spectra.after.smallestNonzero)}. The
              race's Muon run always uses 5 iterations; this slider only redoes the iteration on the captured matrix.
            </p>
          </>
        ) : (
          <p className="lab-note">The captured step is past the end of the run.</p>
        )}
      </LabSurface>

      <LabSurface label="Tuned against tuned" className="op-tune-card">
        <SurfaceHeading
          kicker={`${SWEEP_FACTORS.length} rates per optimizer · ${SWEEP_FACTORS.length * OPTIMIZERS.length} runs`}
          title="Give every optimizer the same search"
          aside={<span className="tg-badge">{progress.done} of {progress.total} trained</span>}
        />
        <div className="op-tune-controls">
          <button
            type="button"
            className="primary-action"
            disabled={progress.running || progress.complete}
            onClick={() => {
              startSweep();
              narrate("Training every optimizer at five learning rates.");
            }}
          >
            {progress.complete ? "Tuned" : progress.running ? `Tuning · ${progress.done} of ${progress.total}` : "Tune each"}
          </button>
          {progress.running || (progress.done > 0 && !progress.complete) ? (
            <progress className="op-progress" max={progress.total} value={progress.done} aria-label="Tuning progress" />
          ) : null}
        </div>
        {tuned ? (
          <TuneTable rows={tuned} />
        ) : (
          <p className="lab-note">
            Tune each trains every optimizer at a tenth, three tenths, one, three and ten times its default rate, with
            every other setting at the lab's defaults, and keeps the rate with the lowest held-out loss. That is{" "}
            {SWEEP_FACTORS.length * OPTIMIZERS.length} runs, trained one at a time; Muon's five take the longest,
            because every one of their steps runs five Newton–Schulz iterations.
          </p>
        )}
      </LabSurface>

      <LabSurface label="Optimizer state" className="op-state-card">
        <SurfaceHeading
          kicker="Bytes kept between steps"
          title="What each optimizer remembers"
          aside={<span className="tg-badge">fp32 · 4 bytes a float</span>}
        />
        <div className="op-table-wrap">
          <table className="op-table op-state-table">
            <caption>
              Optimizer state is the buffers an optimizer carries from one step to the next. The lab column is the
              trainer's own count; the 7.5 billion column is arithmetic.
            </caption>
            <thead>
              <tr>
                <th scope="col">Optimizer</th>
                <th scope="col">Floats per weight</th>
                <th scope="col">{LAB_PARAMETERS} weights</th>
                <th scope="col">7.5 billion weights</th>
              </tr>
            </thead>
            <tbody>
              {memory.map((row) => (
                <tr key={row.optimizer} className={row.optimizer === selected ? "is-selected" : undefined}>
                  <th scope="row">{optimizerChoice(row.optimizer).label}</th>
                  <td>
                    <span className="op-buffers" aria-hidden="true">
                      {Array.from({ length: row.buffers }, (_, index) => (
                        <i key={index} />
                      ))}
                    </span>
                    {row.buffers}
                  </td>
                  <td>{teachingMemory.formatBytes(row.labBytes)}</td>
                  <td>
                    <span className="op-state-cell">
                      <span className="op-state-bar" aria-hidden="true">
                        <i style={{ width: `${((row.largeBytes / largestState) * 100).toFixed(1)}%` }} />
                      </span>
                      <span>{teachingMemory.formatBytes(row.largeBytes)}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric
            label="AdamW over SGD, 7.5 billion"
            value={`+${teachingMemory.formatBytes(largeBytesOf("adamw") - largeBytesOf("sgd"))}`}
            tone="loss"
          />
          <Metric label="Mixed-precision Adam, all model states" value={teachingMemory.formatBytes(adamMixed.total)} />
        </div>
        <p className="lab-note">
          AdamW keeps two floats per weight and is counted with the memory helper's Adam row; momentum and Muon
          keep one and are counted with its momentum row. Training in mixed precision adds a 4-byte fp32 master copy of every weight, which is how the
          ZeRO paper reaches 16 bytes per weight, 120 GB, for Adam at 7.5 billion weights. A Muon run still keeps
          Adam's two floats for its embedding and output layers.
        </p>
      </LabSurface>
    </div>
  );
}

/** A kept share's place on a log scale from 0.1% (0) to 100% (1), so 3.6% and 0.12% can be told apart. */
const keptShare = (kept: number) => clamp((Math.log10(Math.max(1e-9, kept)) + 3) / 3, 0.012, 1);

/** Non-default Adam settings, listed beside its row in the race table. */
function adamNotes(settings: AdamSettings): string {
  const notes: string[] = [];
  if (!settings.biasCorrection) notes.push("no bias correction");
  if (settings.warmup > 0) notes.push(`${Math.round(settings.warmup * 100)}% warmup`);
  if (settings.weightDecay > 0) {
    notes.push(`decay ${formatRate(settings.weightDecay)}${settings.decayMode === "l2" ? " in the gradient" : ""}`);
  }
  return notes.join(", ");
}

/* The race chart spans the full lab width, so its frame is wider than the half-width cards' charts. */
const CW = 720;
const CL = 46;
const CR = 600;

function RaceChart({
  runs,
  labelOf,
  selected,
  curves,
}: {
  runs: Record<OptimizerId, RaceRun>;
  labelOf: (id: OptimizerId) => string;
  selected: OptimizerId;
  curves: "train" | "held";
}) {
  const H = 250;
  const T = 16;
  const B = 222;
  const series = OPTIMIZERS.map((choice) => ({
    id: choice.id,
    values: curves === "held" ? runs[choice.id].heldCurve : runs[choice.id].trainCurve,
  }));
  const finite = series.flatMap((entry) => entry.values).filter(Number.isFinite);
  const low = clamp(Math.floor(((finite.length ? Math.min(...finite) : 1.8) - 0.05) * 10) / 10, 0, 3);
  const high = clamp(Math.ceil(((finite.length ? Math.max(...finite) : 3.4) + 0.05) * 10) / 10, 3.5, 6);
  const xAt = (epoch: number) => CL + (epoch / EPOCHS) * (CR - CL);
  const yAt = (value: number) => B - ((clamp(Number.isFinite(value) ? value : high, low, high) - low) / (high - low)) * (B - T);
  const ticks = [low, (low + high) / 2, high];
  const order = series.filter((entry) => entry.id !== selected).concat(series.filter((entry) => entry.id === selected));
  const what = curves === "held" ? "held-out" : "training";
  // Each line is named where it ends, with its final loss; labels are nudged apart so none overlap.
  const ends = series
    .map((entry) => {
      const final = entry.values[entry.values.length - 1] ?? Number.NaN;
      return { id: entry.id, final, y: yAt(final), at: yAt(final) };
    })
    .sort((a, b) => a.y - b.y);
  ends.forEach((end, index) => {
    end.at = Math.max(T + 4, end.at, index > 0 ? ends[index - 1].at + 14 : -Infinity);
  });
  for (let index = ends.length - 1; index >= 0; index -= 1) {
    ends[index].at = Math.min(ends[index].at, index === ends.length - 1 ? B - 2 : ends[index + 1].at - 14);
  }
  return (
    <svg
      className="op-chart"
      viewBox={`0 0 ${CW} ${H}`}
      role="img"
      aria-label={`Whole-split ${what} loss at each epoch for four optimizers. ${series
        .map((entry) => `${labelOf(entry.id)} ends at ${loss(entry.values[entry.values.length - 1] ?? Number.NaN)}`)
        .join("; ")}. A model that learned nothing scores ${UNIFORM_CROSS_ENTROPY.toFixed(3)}.`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="op-grid" x1={CL} x2={CR} y1={yAt(tick)} y2={yAt(tick)} />
          <text className="op-tick" x={CL - 6} y={yAt(tick) + 3} textAnchor="end">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {[0, 5, 10, 15, 20, 25].map((epoch) => (
        <text key={epoch} className="op-tick" x={xAt(epoch)} y={B + 13} textAnchor="middle">
          {epoch}
        </text>
      ))}
      <line className="plot-axis" x1={CL} x2={CR} y1={B} y2={B} />
      <line className="plot-axis" x1={CL} x2={CL} y1={T} y2={B} />
      <line className="op-baseline" x1={CL} x2={CR} y1={yAt(UNIFORM_CROSS_ENTROPY)} y2={yAt(UNIFORM_CROSS_ENTROPY)} />
      <text className="op-baseline-label" x={CR - 2} y={yAt(UNIFORM_CROSS_ENTROPY) - 4} textAnchor="end">
        knows nothing · ln 30 = {UNIFORM_CROSS_ENTROPY.toFixed(3)}
      </text>
      {order.map((entry) => (
        <path
          key={entry.id}
          className={`op-series op-tone--${TONES[entry.id]}${entry.id === selected ? " is-selected" : ""}`}
          d={entry.values.map((value, epoch) => `${epoch === 0 ? "M" : "L"}${xAt(epoch).toFixed(1)},${yAt(value).toFixed(1)}`).join(" ")}
          strokeDasharray={DASHES[entry.id] || undefined}
        />
      ))}
      {ends.map((end) => (
        <g key={end.id} className={`op-end op-tone--${TONES[end.id]}${end.id === selected ? " is-selected" : ""}`}>
          <path d={`M${CR + 3},${end.y.toFixed(1)}L${CR + 12},${end.at.toFixed(1)}`} />
          <circle cx={CR} cy={end.y.toFixed(1)} r={end.id === selected ? 3.6 : 2.8} />
          <text x={CR + 15} y={(end.at + 3.5).toFixed(1)}>
            {labelOf(end.id)} {loss(end.final)}
          </text>
        </g>
      ))}
      <text className="op-axis-label" x={CL} y={H - 2}>
        epoch
      </text>
      <text className="op-axis-label" x={4} y={10}>
        {what} loss (nats per character)
      </text>
    </svg>
  );
}

function EarlyChart({
  current,
  other,
  biasOn,
  warmupSteps,
}: {
  current: RaceRun;
  other: RaceRun;
  biasOn: boolean;
  warmupSteps: number;
}) {
  const [ref, width] = useWidth<HTMLElement>(302);
  const W = clamp(width, 240, 620);
  const H = clamp(Math.round(W * 0.62), 180, 250);
  const L = 34;
  const R = W - 10;
  const T = 14;
  const B = H - 34;
  const shown = (run: RaceRun) => run.batchLoss.slice(0, EARLY_STEPS);
  const values = [...shown(current), ...shown(other)].filter(Number.isFinite);
  const high = clamp(Math.ceil((values.length ? Math.max(...values) : 3.6) + 0.2), 4, 9);
  const low = clamp(Math.floor((values.length ? Math.min(...values) : 3) - 0.1), 0, 3);
  const xAt = (step: number) => L + (step / Math.max(1, EARLY_STEPS - 1)) * (R - L);
  const yAt = (value: number) => B - ((clamp(Number.isFinite(value) ? value : high, low, high) - low) / (high - low)) * (B - T);
  const line = (run: RaceRun) =>
    shown(run)
      .map((value, step) => `${step === 0 ? "M" : "L"}${xAt(step).toFixed(1)},${yAt(value).toFixed(1)}`)
      .join(" ");
  const warmEnd = Math.min(EARLY_STEPS - 1, warmupSteps);
  // Each line is named at its right end: the higher one above itself, the lower one below.
  const endOf = (run: RaceRun) => yAt(shown(run)[EARLY_STEPS - 1] ?? Number.NaN);
  const currentAbove = endOf(current) <= endOf(other);
  const labelY = (y: number, above: boolean) => clamp(above ? y - 8 : y + 15, T + 9, B - 5);
  return (
    <figure className="op-early" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Batch loss over the first ${EARLY_STEPS} steps with bias correction ${biasOn ? "on" : "off"}, peaking at ${loss(
          current.earlyPeakLoss,
        )}; with it ${biasOn ? "off" : "on"} the peak is ${loss(other.earlyPeakLoss)}.`}
      >
        {warmupSteps > 0 && (
          <>
            <rect className="op-warmup" x={L} y={T} width={Math.max(0, xAt(warmEnd) - L)} height={B - T} />
            <text className="op-chart-note" x={L + 5} y={B - 6}>
              warmup{warmupSteps > EARLY_STEPS ? ` (runs to step ${warmupSteps})` : ""}
            </text>
          </>
        )}
        {[low, (low + high) / 2, high].map((tick) => (
          <g key={tick}>
            <line className="op-grid" x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="op-tick" x={L - 5} y={yAt(tick) + 3.5} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
        <line className="op-baseline" x1={L} x2={R} y1={yAt(UNIFORM_CROSS_ENTROPY)} y2={yAt(UNIFORM_CROSS_ENTROPY)} />
        <text className="op-chart-note" x={L + 5} y={yAt(UNIFORM_CROSS_ENTROPY) - 5}>
          knows nothing · {UNIFORM_CROSS_ENTROPY.toFixed(3)}
        </text>
        <path className="op-early-line is-other" d={line(other)} />
        <path className="op-early-line is-current" d={line(current)} />
        <text className="op-line-name is-other" x={R - 2} y={labelY(endOf(other), !currentAbove)} textAnchor="end">
          correction {biasOn ? "off" : "on"}
        </text>
        <text className="op-line-name is-current" x={R - 2} y={labelY(endOf(current), currentAbove)} textAnchor="end">
          correction {biasOn ? "on" : "off"} · this run
        </text>
        {[1, 20, 40, EARLY_STEPS].map((step) => (
          <text key={step} className="op-tick" x={xAt(step - 1)} y={B + 13} textAnchor={step === EARLY_STEPS ? "end" : "middle"}>
            {step}
          </text>
        ))}
        <text className="op-axis-label" x={L} y={H - 4}>
          step · batch loss before the update
        </text>
      </svg>
      <figcaption>
        Solid: bias correction {biasOn ? "on" : "off"}, the run the metrics read. Dashed: the same run with it{" "}
        {biasOn ? "off" : "on"}. {warmupSteps > 0 ? `The shaded strip is the warmup, ${warmupSteps} steps long.` : "No warmup."}
      </figcaption>
    </figure>
  );
}

/** The first step of the run and of its counterpart as two bars on one scale, with the rate marked. */
function FirstStepBars({ current, other, biasOn, rate }: { current: number; other: number; biasOn: boolean; rate: number }) {
  const top = Math.max(current, other, rate) * 1.08 || 1;
  const share = (value: number) => `${((Math.max(0, value) / top) * 100).toFixed(1)}%`;
  const times = (value: number) => (rate > 0 ? `${(value / rate).toFixed(2)}× the rate` : "");
  const rows = [
    { key: "current", name: `Correction ${biasOn ? "on" : "off"} · this run`, value: current },
    { key: "other", name: `Correction ${biasOn ? "off" : "on"}`, value: other },
  ];
  return (
    <div
      className="op-first"
      role="img"
      aria-label={`Largest change to any one weight on step 1. Bias correction ${biasOn ? "on" : "off"}: ${current.toFixed(3)}, ${times(
        current,
      )}. Bias correction ${biasOn ? "off" : "on"}: ${other.toFixed(3)}, ${times(other)}. The rate is ${formatRate(rate)}.`}
    >
      <p className="op-first__title">First-step move: the largest change to one weight on step 1</p>
      <div className="op-first__rows" style={{ "--op-rate": share(rate) } as CSSProperties}>
        {rows.map((row) => (
          <div key={row.key} className={`op-first__row is-${row.key}`}>
            <span className="op-first__name">{row.name}</span>
            <span className="op-first__track">
              <i style={{ width: share(row.value) }} />
            </span>
            <b>
              {row.value.toFixed(3)} <small>{times(row.value)}</small>
            </b>
          </div>
        ))}
        <span className={`op-first__rate${rate / top > 0.6 ? " is-flipped" : ""}`}>
          <span>{rate / top > 0.6 ? `rate ${formatRate(rate)} ↑` : `↑ rate ${formatRate(rate)}`}</span>
        </span>
      </div>
    </div>
  );
}

function SpectrumChart({
  before,
  after,
  iterations,
}: {
  before: SpectrumSummary;
  after: SpectrumSummary;
  iterations: number;
}) {
  const [ref, width] = useWidth<HTMLElement>(302);
  const W = clamp(width, 240, 720);
  const H = clamp(Math.round(W * 0.6), 190, 260);
  const L = 30;
  const R = W - 6;
  const T = 14;
  const B = H - 32;
  const top = 1.3;
  const slots = Math.max(before.values.length, after.values.length, 1);
  const slot = (R - L) / slots;
  const yAt = (value: number) => B - (clamp(value, 0, top) / top) * (B - T);
  return (
    <figure className="op-spectrum" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Singular values of Muon's update, largest first. Before Newton–Schulz: largest ${before.largest.toFixed(
          3,
        )}, ${before.nonzero} nonzero. After ${iterations} iterations: largest ${after.largest.toFixed(3)}, ${after.inBand} of ${
          after.nonzero
        } between ${BAND_LOW} and ${BAND_HIGH}, ${after.zeros} still zero.`}
      >
        <rect className="op-band" x={L} y={yAt(top)} width={R - L} height={yAt(BAND_LOW) - yAt(top)} />
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line className={tick === 1 ? "op-unit" : "op-grid"} x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="op-tick" x={L - 5} y={yAt(tick) + 3.5} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
        {Array.from({ length: slots }, (_, index) => {
          const x = L + index * slot;
          const was = before.values[index] ?? 0;
          const now = after.values[index] ?? 0;
          const barWidth = Math.max(1.5, slot * 0.4);
          return (
            <g key={index}>
              <rect className="op-bar is-before" x={x + slot * 0.08} y={yAt(was)} width={barWidth} height={B - yAt(was)} />
              <rect className="op-bar is-after" x={x + slot * 0.5} y={yAt(now)} width={barWidth} height={B - yAt(now)} />
              {now <= ZERO_SINGULAR && <circle className="op-zero" cx={x + slot * 0.5} cy={B - 4} r={Math.min(3, slot * 0.3)} />}
            </g>
          );
        })}
        <text className="op-chart-note op-chart-note--unit" x={R - 3} y={yAt(1) - 5} textAnchor="end">
          all equal
        </text>
        <text className="op-tick" x={L + slot * 0.5} y={B + 13} textAnchor="middle">
          1
        </text>
        <text className="op-tick" x={R - slot * 0.5} y={B + 13} textAnchor="middle">
          {slots}
        </text>
        <text className="op-axis-label" x={(L + R) / 2} y={B + 13} textAnchor="middle">
          direction, largest first
        </text>
        <text className="op-axis-label" x={L} y={H - 4}>
          singular value · shaded: {BAND_LOW} to {BAND_HIGH}
        </text>
      </svg>
      <figcaption>
        <span className="op-key is-before" aria-hidden="true" /> before, normalized{" "}
        <span className="op-key is-after" aria-hidden="true" /> after {iterations} {iterations === 1 ? "iteration" : "iterations"}{" "}
        <span className="op-key is-zero" aria-hidden="true" /> zero direction
      </figcaption>
    </figure>
  );
}

/** Held-out loss against the rate multiple: one U-shaped line per optimizer, its lowest point ringed. */
function TuneChart({ rows }: { rows: ReadonlyArray<SweepRow> }) {
  const [ref, width] = useWidth<HTMLElement>(302);
  const W = clamp(width, 240, 720);
  const H = clamp(Math.round(W * 0.62), 190, 260);
  const L = 36;
  const R = W - 70;
  const T = 20;
  const B = H - 34;
  const finite = rows.flatMap((row) => row.points.map((point) => point.heldOut)).filter(Number.isFinite);
  const low = Math.floor(((finite.length ? Math.min(...finite) : 2) - 0.02) * 10) / 10;
  const high = Math.min(UNIFORM_CROSS_ENTROPY, Math.ceil(((finite.length ? Math.max(...finite) : 3) + 0.02) * 10) / 10);
  const xAt = (index: number) => L + 8 + (index / Math.max(1, SWEEP_FACTORS.length - 1)) * (R - L - 16);
  const yAt = (value: number) => B - ((clamp(Number.isFinite(value) ? value : high, low, high) - low) / Math.max(1e-9, high - low)) * (B - T);
  const ticks = [low, (low + high) / 2, high];
  const ends = rows
    .map((row) => {
      const y = yAt(row.points[row.points.length - 1]?.heldOut ?? Number.NaN);
      return { id: row.optimizer, y, at: y };
    })
    .sort((a, b) => a.y - b.y);
  ends.forEach((end, index) => {
    end.at = Math.max(T + 4, end.at, index > 0 ? ends[index - 1].at + 13 : -Infinity);
  });
  for (let index = ends.length - 1; index >= 0; index -= 1) {
    ends[index].at = Math.min(ends[index].at, index === ends.length - 1 ? B - 2 : ends[index + 1].at - 13);
  }
  return (
    <figure className="op-tune-chart" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Held-out loss at five multiples of each optimizer's default rate. ${rows
          .map(
            (row) =>
              `${optimizerChoice(row.optimizer).label}: lowest ${loss(row.best.heldOut)} at ${formatRate(row.best.rate)}, ${loss(
                row.atDefault.heldOut,
              )} at its default`,
          )
          .join("; ")}.`}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="op-grid" x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
            <text className="op-tick" x={L - 5} y={yAt(tick) + 3.5} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
        {SWEEP_FACTORS.map((factor, index) => (
          <text key={factor} className="op-tick" x={xAt(index)} y={B + 13} textAnchor="middle">
            {factor === 1 ? "default" : `×${factor}`}
          </text>
        ))}
        {rows.map((row) => (
          <g key={row.optimizer} className={`op-tune-line op-tone--${TONES[row.optimizer]}`}>
            <path
              d={row.points.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(index).toFixed(1)},${yAt(point.heldOut).toFixed(1)}`).join(" ")}
              strokeDasharray={DASHES[row.optimizer] || undefined}
            />
            {row.points.map((point, index) => (
              <circle key={point.rate} className={point === row.best ? "is-best" : undefined} cx={xAt(index)} cy={yAt(point.heldOut)} r={point === row.best ? 4.5 : 2.2} />
            ))}
          </g>
        ))}
        {ends.map((end) => (
          <g key={end.id} className={`op-end op-tone--${TONES[end.id]}`}>
            <path d={`M${xAt(SWEEP_FACTORS.length - 1) + 4},${end.y.toFixed(1)}L${R + 4},${end.at.toFixed(1)}`} />
            <text x={R + 7} y={(end.at + 3.5).toFixed(1)}>
              {optimizerChoice(end.id).label}
            </text>
          </g>
        ))}
        <text className="op-axis-label" x={L} y={H - 4}>
          rate, as a multiple of the default
        </text>
        <text className="op-axis-label" x={4} y={9}>
          held-out loss
        </text>
      </svg>
      <figcaption>Each line is one optimizer at five rates. The ring marks its lowest held-out loss; lower is better.</figcaption>
    </figure>
  );
}

function TuneTable({ rows }: { rows: ReadonlyArray<SweepRow> }) {
  const ranked = [...rows].sort((left, right) => left.best.heldOut - right.best.heldOut);
  const winner = ranked[0];
  const second = ranked[1];
  const label = (id: OptimizerId) => optimizerChoice(id).label;
  const adam = rows.find((row) => row.optimizer === "adamw");
  const sgd = rows.find((row) => row.optimizer === "sgd");
  return (
    <>
      <TuneChart rows={rows} />
      <div className="op-table-wrap">
        <table className="op-table op-tune-table">
          <caption>
            Held-out loss at each rate of the grid, as a multiple of the optimizer's default; the rate is under each
            value. The lowest in each row is marked.
          </caption>
          <thead>
            <tr>
              <th scope="col">Optimizer</th>
              {SWEEP_FACTORS.map((factor) => (
                <th key={factor} scope="col">
                  {factor === 1 ? "default" : `×${factor}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.optimizer}>
                <th scope="row">{label(row.optimizer)}</th>
                {row.points.map((point) => (
                  <td key={point.rate} className={point === row.best ? "is-best" : undefined}>
                    {loss(point.heldOut)}
                    <small>{formatRate(point.rate)}</small>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="metric-row">
        <Metric label="Tuned winner" value={`${label(winner.optimizer)} · ${loss(winner.best.heldOut)}`} tone="forward" />
        {sgd && (
          <Metric label="SGD, default to tuned" value={`${loss(sgd.atDefault.heldOut)} → ${loss(sgd.best.heldOut)}`} />
        )}
        {adam && (
          <Metric label="AdamW, default to tuned" value={`${loss(adam.atDefault.heldOut)} → ${loss(adam.best.heldOut)}`} />
        )}
      </div>
      <div className="tg-callout is-quiet">
        <strong>
          Tuned, {label(winner.optimizer)} leads {label(second.optimizer)} by {(second.best.heldOut - winner.best.heldOut).toFixed(3)}.
        </strong>
        <span>
          {ranked.map((row) => `${label(row.optimizer)} ${loss(row.best.heldOut)} at ${formatRate(row.best.rate)}`).join(", ")}.
          Five shuffle seeds move each tuned result by 0.006 to 0.015 (measured for the lesson), so gaps that small are
          not a ranking. The rate was picked on the same held-out sentences it is scored on, which flatters every row a
          little.
        </span>
      </div>
    </>
  );
}
