import { useMemo } from "react";
import {
  BarList,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  UNIFORM_CROSS_ENTROPY,
  type LineSeries,
  type ModuleContext,
} from "@app/module-sdk";
import { ARITHMETIC, DOMAINS, pairCount } from "./corpora";
import {
  arithmeticShareCurve,
  arithmeticSweep,
  BATCH_SIZE,
  BUDGET_PAIRS,
  DECAY_FRACTION,
  gainPerEpoch,
  LATE_PHASES,
  LEARNING_RATE,
  PRESETS,
  runEvenSpread,
  runMixture,
  SCHEDULES,
  scheduleCurve,
  SEED,
  sharesOf,
  STEPS,
  SWITCH_STEP,
  TARGET_MIX_SHARE,
  WARMUP_FRACTION,
  WEIGHT_MAX,
  WEIGHT_MIN,
  type CurvePoint,
  type LatePhase,
  type MixtureRun,
  type ScheduleKind,
} from "./mixture";
import { answerProbe } from "./probe";
import {
  BUDGET_MAX,
  BUDGET_MIN,
  BUDGET_STEP,
  repetitionSweep,
  runRepetition,
  uniqueText,
  UNIQUE_MAX,
  UNIQUE_MIN,
} from "./repetition";
import { readState, WEIGHT_KEYS, weightsOf } from "./state";

const loss = (value: number) => value.toFixed(3);
const signed = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
const percent = (share: number) => `${Math.round(share * 1000) / 10}%`;
const count = (value: number) => Math.round(value).toLocaleString("en-US");

const lateLabel = (value: LatePhase) => LATE_PHASES.find((entry) => entry.value === value)?.label ?? "Same mix";
const scheduleLabel = (value: ScheduleKind) => SCHEDULES.find((entry) => entry.value === value)?.label ?? "Warmup-stable-decay";

/** Held-out curves start at the zero table's 3.401; the chart frames the part of the run after the first drop. */
const CURVE_FROM_STEP = 90;

function curveDomain(curves: ReadonlyArray<ReadonlyArray<number>>): [number, number] {
  const values = curves.flat().filter(Number.isFinite);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = Math.max(0.04, (high - low) * 0.1);
  return [Math.floor((low - pad) * 20) / 20, Math.ceil((high + pad) * 20) / 20];
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  // Every card trains inside this render, so state is only ever read through the clamping readState.
  const settings = readState(state);
  const { latePhase, schedule, uniqueSentences, repeatBudget } = settings;
  const [harbor, recipes, proverbs, arithmetic] = weightsOf(settings);
  const stable = useMemo(() => [harbor, recipes, proverbs, arithmetic], [harbor, recipes, proverbs, arithmetic]);
  const shares = useMemo(() => sharesOf(stable), [stable]);
  const switched = latePhase !== "same";

  const run = useMemo(() => runMixture(stable, latePhase, schedule), [stable, latePhase, schedule]);
  const sweep = useMemo(
    () => arithmeticSweep([harbor, recipes, proverbs, 0], latePhase, schedule),
    [harbor, recipes, proverbs, latePhase, schedule],
  );
  const gains = useMemo(() => gainPerEpoch(sweep), [sweep]);
  const even = useMemo(
    () => (switched ? runEvenSpread(stable, latePhase, schedule) : null),
    [switched, stable, latePhase, schedule],
  );
  const noSwitch = useMemo(
    () => (switched ? runMixture(stable, "same", schedule) : run),
    [switched, stable, schedule, run],
  );
  const probe = useMemo(() => answerProbe(run.weights), [run]);

  const repeat = useMemo(() => runRepetition(uniqueSentences, repeatBudget), [uniqueSentences, repeatBudget]);
  const fresh = useMemo(() => runRepetition(UNIQUE_MAX, repeatBudget), [repeatBudget]);
  const repeatBars = useMemo(() => repetitionSweep(repeatBudget), [repeatBudget]);
  const repeatPairs = useMemo(() => pairCount(uniqueText(uniqueSentences)), [uniqueSentences]);

  const preset = PRESETS.find((entry) => entry.weights.every((weight, index) => weight === stable[index]));
  const setWeight = (index: number, value: number) => {
    const next = [...stable];
    next[index] = value;
    setState({ [WEIGHT_KEYS[index]]: value });
    const nextShares = sharesOf(next);
    narrate(
      next.every((weight) => weight === 0)
        ? "All four weights at zero leave nothing to draw from, so the mixture falls back to Uniform."
        : `${DOMAINS[index].name} weight ${value}: ${percent(nextShares[index])} of every batch.`,
    );
  };

  const sweepRow = sweep.find((point) => point.weight === arithmetic);
  const sweepEpochs = sweep.flatMap((point) => (point.run ? [point.run.domains[3].epochs] : []));
  const sweepSeries: LineSeries[] = [
    {
      id: "arithmetic",
      name: `Arithmetic held-out, weight 0 to ${WEIGHT_MAX}`,
      tone: "attention",
      points: sweep.flatMap((point) => (point.run ? [{ x: point.run.domains[3].epochs, y: point.run.arithmetic }] : [])),
    },
    {
      id: "general",
      name: `General held-out, weight 0 to ${WEIGHT_MAX}`,
      tone: "forward",
      dash: "dashed",
      points: sweep.flatMap((point) => (point.run ? [{ x: point.run.domains[3].epochs, y: point.run.general }] : [])),
    },
  ];
  const sweepDomain = curveDomain(sweep.flatMap((point) => (point.run ? [[point.run.arithmetic, point.run.general]] : [])));
  const maxSweepEpochs = Math.max(0.1, ...sweepEpochs);

  const lateRows = (entry: MixtureRun) => [
    loss(entry.arithmetic),
    loss(entry.general),
    signed(entry.forgetting),
    count(entry.domains[3].seen),
  ];
  const lateColumns: Array<{ id: string; title: string; values: string[] }> = switched
    ? [
        { id: "switched", title: "Switched at decay", values: lateRows(run) },
        ...(even ? [{ id: "even", title: "Spread evenly", values: lateRows(even) }] : []),
        { id: "none", title: "No switch", values: lateRows(noSwitch) },
      ]
    : [{ id: "none", title: "No switch (this run)", values: lateRows(run) }];

  const visible = (curve: ReadonlyArray<CurvePoint>) => curve.filter((point) => point.step >= CURVE_FROM_STEP);
  const lossSeries: LineSeries[] = [
    {
      id: "arithmetic-run",
      name: switched ? "Arithmetic, switched" : "Arithmetic",
      tone: "attention",
      points: visible(run.curve).map((point) => ({ x: point.step, y: point.held[3] })),
    },
    {
      id: "general-run",
      name: switched ? "General, switched" : "General",
      tone: "forward",
      points: visible(run.curve).map((point) => ({ x: point.step, y: point.general })),
    },
    ...(even
      ? [
          {
            id: "arithmetic-even",
            name: "Arithmetic, spread evenly",
            tone: "attention" as const,
            dash: "dashed" as const,
            points: visible(even.curve).map((point) => ({ x: point.step, y: point.held[3] })),
          },
          {
            id: "general-even",
            name: "General, spread evenly",
            tone: "forward" as const,
            dash: "dashed" as const,
            points: visible(even.curve).map((point) => ({ x: point.step, y: point.general })),
          },
        ]
      : []),
  ];
  const lossDomain = curveDomain(lossSeries.map((series) => series.points.map((point) => point.y)));

  const repeatSeries: LineSeries[] = [
    {
      id: "fresh",
      name: `all ${UNIQUE_MAX} unique sentences`,
      tone: "forward",
      points: fresh.curve.map((point) => ({ x: point.pairs, y: point.held })),
    },
    ...(uniqueSentences === UNIQUE_MAX
      ? []
      : [
          {
            id: "repeated",
            name: `${uniqueSentences} unique sentences`,
            tone: "loss" as const,
            dash: "dashed" as const,
            points: repeat.curve.map((point) => ({ x: point.pairs, y: point.held })),
          },
        ]),
  ];

  return (
    <div className="tg-lab tg-lab--hero mt-lab">
      <LabSurface label="Data mixture" className="mt-mixture-card">
        <SurfaceHeading
          kicker={`${STEPS} steps × ${BATCH_SIZE} pairs = ${count(BUDGET_PAIRS)} pairs · SGD at rate ${LEARNING_RATE} · ${scheduleLabel(schedule).toLowerCase()}`}
          title="Four domains share one training budget"
          aside={<span className="tg-badge">seed {SEED}</span>}
        />
        <div className="mt-mixture">
          <div className="mt-weights">
            <SegmentedControl
              label="Mixture preset"
              value={preset?.id ?? "custom"}
              options={PRESETS.map((entry) => ({ value: entry.id, label: entry.label }))}
              onChange={(value) => {
                const chosen = PRESETS.find((entry) => entry.id === value);
                if (!chosen) return;
                setState(Object.fromEntries(WEIGHT_KEYS.map((key, index) => [key, chosen.weights[index]])));
                narrate(
                  `${chosen.label}: ${DOMAINS.map((domain, index) => `${domain.name} ${percent(sharesOf(chosen.weights)[index])}`).join(", ")}.`,
                );
              }}
            />
            {DOMAINS.map((domain, index) => (
              <RangeControl
                key={domain.id}
                label={`${domain.name} weight`}
                min={WEIGHT_MIN}
                max={WEIGHT_MAX}
                step={1}
                value={stable[index]}
                format={(value) => `${value} · ${percent(shares[index])} of each batch`}
                onChange={(value) => setWeight(index, value)}
              />
            ))}
            <p className="mt-weights__note">
              Only the ratio matters: 3, 3, 3, 1 and 6, 6, 6, 2 draw the same shares. All four at 0 falls back to
              Uniform.
            </p>
          </div>
          <div className="mt-mixture__readouts">
            <div
              className="mt-budget-bar"
              role="img"
              aria-label={`Each batch is drawn ${DOMAINS.map((domain, index) => `${percent(shares[index])} ${domain.name}`).join(", ")}${switched ? " until the late phase" : ""}.`}
            >
              {DOMAINS.map((domain, index) =>
                shares[index] > 0 ? (
                  <i
                    key={domain.id}
                    className={`mt-budget-bar__${domain.general ? "general" : "target"} mt-budget-bar__${domain.id}`}
                    style={{ flexGrow: shares[index] }}
                  >
                    {domain.name} {percent(shares[index])}
                  </i>
                ) : null,
              )}
            </div>
            <div className="mt-table-wrap">
              <table className="tg-board mt-mixture-table">
                <caption>
                  {switched
                    ? `Shares hold for the first ${Math.round((1 - DECAY_FRACTION) * 100)}% of steps; ${lateLabel(latePhase)} takes over for the last ${Math.round(DECAY_FRACTION * 100)}%, and Pairs drawn counts both.`
                    : "Epochs are pairs drawn divided by the training pairs a domain holds: one epoch is one full pass."}
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Domain</th>
                    <th scope="col">Weight</th>
                    <th scope="col">Share</th>
                    <th scope="col">Pairs drawn</th>
                    <th scope="col">Training pairs</th>
                    <th scope="col">Epochs</th>
                  </tr>
                </thead>
                <tbody>
                  {DOMAINS.map((domain, index) => {
                    const result = run.domains[index];
                    return (
                      <tr key={domain.id}>
                        <th scope="row">{domain.name}</th>
                        <td>{stable[index]}</td>
                        <td>{percent(shares[index])}</td>
                        <td>{count(result.seen)}</td>
                        <td>{count(result.pairs)}</td>
                        <td>{result.epochs.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <span className="mt-subhead">Held-out loss by domain (nats per character, lower is better)</span>
            <BarList
              label="Held-out loss by domain"
              max={UNIFORM_CROSS_ENTROPY}
              items={[
                ...DOMAINS.map((domain, index) => ({
                  id: domain.id,
                  label: domain.name,
                  value: run.domains[index].held,
                  display: loss(run.domains[index].held),
                  detail: `${domain.heldOut.length} held-out sentences, never trained on`,
                  tone: domain.general ? ("forward" as const) : ("attention" as const),
                })),
                {
                  id: "general",
                  label: "General",
                  value: run.general,
                  display: loss(run.general),
                  detail: "The mean of Harbor, Recipes and Proverbs, each counted once",
                  tone: "muted" as const,
                  emphasis: true,
                },
              ]}
            />
          </div>
        </div>
        <div className="metric-row">
          <Metric label="General held-out" value={`${loss(run.general)} nats`} tone="forward" />
          <Metric label="Arithmetic held-out" value={`${loss(run.arithmetic)} nats`} tone="gradient" />
          <Metric label="Late phase" value={lateLabel(latePhase)} />
        </div>
        <p className="lab-note">
          One model, trained live from a zeroed table on {count(BUDGET_PAIRS)} character pairs. Each batch slot picks a
          domain with probability equal to its share, then takes that domain&apos;s next pair. Every loss here is
          measured on held-out sentences, the last quarter of each domain; a bar at {UNIFORM_CROSS_ENTROPY.toFixed(3)}{" "}
          would be a model that knows nothing.
        </p>
      </LabSurface>

      <LabSurface label="Arithmetic weight sweep" className="mt-sweep-card">
        <SurfaceHeading
          kicker={`Nine runs · Arithmetic weight 0 to ${WEIGHT_MAX} · Harbor ${harbor}, Recipes ${recipes}, Proverbs ${proverbs}`}
          title="What each extra epoch of Arithmetic buys, and what General pays"
          aside={<span className="tg-badge">weight {arithmetic}</span>}
        />
        <LineChart
          label="Held-out loss against Arithmetic epochs"
          xLabel="Arithmetic epochs"
          yLabel="held-out loss (nats/char)"
          xDomain={[0, maxSweepEpochs]}
          yDomain={sweepDomain}
          series={sweepSeries}
          marker={sweepRow?.run ? { x: sweepRow.run.domains[3].epochs, label: `weight ${arithmetic}` } : undefined}
        />
        <div className="mt-table-wrap">
          <table className="tg-board mt-sweep-table">
            <caption>
              Gain per extra epoch is the fall in Arithmetic held-out loss from the row above, divided by the epochs
              added.
            </caption>
            <thead>
              <tr>
                <th scope="col">Arithmetic weight</th>
                <th scope="col">Epochs</th>
                <th scope="col">Arithmetic held-out</th>
                <th scope="col">Gain per extra epoch</th>
                <th scope="col">General held-out</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((point, index) => (
                <tr key={point.weight} className={point.weight === arithmetic ? "is-leader" : undefined}>
                  <th scope="row">{point.weight}</th>
                  <td>{point.run ? point.run.domains[3].epochs.toFixed(2) : "—"}</td>
                  <td>{point.run ? loss(point.run.arithmetic) : "no data"}</td>
                  <td>{gains[index] === null ? "—" : (gains[index] as number).toFixed(2)}</td>
                  <td>{point.run ? loss(point.run.general) : "no data"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Each row is a full run with the other three weights, the late phase and the schedule as set; the highlighted
          row is the run on the Data mixture card. Gain per extra epoch is in nats per epoch.
        </p>
      </LabSurface>

      <LabSurface label="Answer probe" className="mt-probe-card">
        <SurfaceHeading
          kicker={`${probe.total} held-out facts · 21 answers`}
          title="A lower loss is not the same as adding up"
          aside={
            <span className="tg-badge">
              {probe.correct} of {probe.total} right
            </span>
          }
        />
        <div className="metric-row">
          <Metric label="Arithmetic held-out" value={`${loss(run.arithmetic)} nats`} tone="gradient" />
          <Metric label="Right answers" value={`${probe.correct} of ${probe.total}`} tone={probe.correct > 0 ? "forward" : "loss"} />
          <Metric label="Different answers" value={`${probe.distinctPicks}`} />
        </div>
        <span className="mt-subhead">The five likeliest answers, as shares of what the 21 receive</span>
        <BarList
          label="Answer probabilities after is"
          max={1}
          items={probe.ranking.slice(0, 5).map((entry, index) => ({
            id: entry.word,
            label: entry.word,
            value: entry.share,
            display: percent(entry.share),
            tone: index === 0 ? ("attention" as const) : ("muted" as const),
            emphasis: index === 0,
          }))}
        />
        <ul className="mt-probe-facts" aria-label="Three held-out facts with the model's answer">
          {probe.facts.slice(0, 3).map((fact) => (
            <li key={fact.question}>
              <span>{fact.question} …</span>
              <b>model: {fact.pick}</b>
              <i>true: {fact.truth}</i>
            </li>
          ))}
        </ul>
        <p className="lab-note">
          The table sees only the character before its guess, and before every answer that is the same space. So it
          ranks the 21 answers the same way for every question, whatever the numbers in it. Its loss on arithmetic
          text falls by learning the words, not the sums.
        </p>
      </LabSurface>

      <LabSurface label="Repeated data" className="mt-repeat-card">
        <SurfaceHeading
          kicker={`Arithmetic only · ${count(repeatBudget)} pairs · the mixture run's recipe`}
          title="Repeat the sentences you have, or find more?"
          aside={<span className="tg-badge">{repeat.epochs.toFixed(1)} epochs</span>}
        />
        <div className="mt-repeat-controls">
          <RangeControl
            label="Budget"
            min={BUDGET_MIN}
            max={BUDGET_MAX}
            step={BUDGET_STEP}
            value={repeatBudget}
            format={(value) => `${count(value)} pairs · ${count(value / BATCH_SIZE)} steps`}
            onChange={(value) => {
              setState({ repeatBudget: value });
              narrate(`Budget ${count(value)} pairs.`);
            }}
          />
          <RangeControl
            label="Unique sentences"
            min={UNIQUE_MIN}
            max={UNIQUE_MAX}
            step={1}
            value={uniqueSentences}
            format={(value) => `${value} of ${UNIQUE_MAX} · ${count(pairCount(uniqueText(value)))} pairs`}
            onChange={(value) => {
              setState({ uniqueSentences: value });
              narrate(`${value} unique sentences: ${(repeatBudget / pairCount(uniqueText(value))).toFixed(1)} epochs at this budget.`);
            }}
          />
        </div>
        <div className="mt-repeat-body">
          <LineChart
            label="Held-out arithmetic loss as the budget is spent"
            xLabel="pairs drawn"
            yLabel="held-out loss (nats/char)"
            xDomain={[0, repeatBudget]}
            yDomain={[1.2, UNIFORM_CROSS_ENTROPY + 0.05]}
            series={repeatSeries}
            footnote={`Scored on the ${ARITHMETIC.heldOut.length} held-out arithmetic sentences at ${repeat.curve.length} checkpoints. The first ${uniqueSentences} training sentences hold ${count(repeatPairs)} pairs.`}
          />
          <div>
            <span className="mt-subhead">At this budget, by unique sentences</span>
            <BarList
              label="Held-out arithmetic loss at this budget by unique sentences"
              max={UNIFORM_CROSS_ENTROPY}
              items={repeatBars.map((entry) => ({
                id: `unique-${entry.unique}`,
                label: `${entry.unique} unique`,
                value: entry.held,
                display: loss(entry.held),
                detail: `${entry.epochs.toFixed(1)} epochs`,
                tone: entry.unique === UNIQUE_MAX ? ("forward" as const) : ("loss" as const),
                emphasis: entry.unique === uniqueSentences,
              }))}
            />
          </div>
        </div>
        <div className="metric-row">
          <Metric label="Epochs" value={repeat.epochs.toFixed(1)} />
          <Metric label="Held-out loss" value={`${loss(repeat.held)} nats`} tone="loss" />
          <Metric label={`Versus ${UNIQUE_MAX} unique`} value={signed(repeat.held - fresh.held)} />
          <Metric label="Lowest along the run" value={`${loss(repeat.lowest.held)} at ${repeat.lowest.epochs.toFixed(1)} epochs`} />
        </div>
        <p className="lab-note">
          A separate run on Arithmetic alone: SGD at rate {LEARNING_RATE}, batch {BATCH_SIZE}, warmup-stable-decay, seed{" "}
          {SEED}. The Late phase and Schedule controls do not apply here. Fewer unique sentences means the same budget
          is spent on more passes over each one.
        </p>
      </LabSurface>

      <LabSurface label="Late phase and annealing" className="mt-late-card">
        <SurfaceHeading
          kicker={`The last ${Math.round(DECAY_FRACTION * 100)}% of steps · from step ${SWITCH_STEP} of ${STEPS}`}
          title="Switch the mixture while the learning rate decays"
          aside={<span className="tg-badge">{lateLabel(latePhase)}</span>}
        />
        <div className="mt-late-controls">
          <SegmentedControl
            label="Late phase"
            value={latePhase}
            options={LATE_PHASES}
            onChange={(value) => {
              setState({ latePhase: value });
              narrate(
                value === "same"
                  ? "No switch: the same mixture to the end."
                  : value === "mix"
                    ? `Target mix: from step ${SWITCH_STEP}, half of every batch is Arithmetic.`
                    : `Target only: from step ${SWITCH_STEP}, every batch is Arithmetic.`,
              );
            }}
          />
          <SegmentedControl
            label="Schedule"
            value={schedule}
            options={SCHEDULES}
            onChange={(value) => {
              setState({ schedule: value });
              narrate(value === "wsd" ? "Warmup-stable-decay." : "Constant rate, no warmup and no decay.");
            }}
          />
        </div>
        <div className="mt-late-charts">
          <LineChart
            label="Learning-rate multiplier and Arithmetic share across the run"
            xLabel="step"
            yLabel="multiplier or share"
            xDomain={[0, STEPS]}
            yDomain={[0, 1]}
            marker={{ x: SWITCH_STEP, label: schedule === "wsd" ? "decay starts" : "late phase starts" }}
            series={[
              {
                id: "rate",
                name: "Learning-rate multiplier",
                tone: "forward",
                points: scheduleCurve(schedule).map((point) => ({ x: point.x * STEPS, y: point.y })),
                format: (value) => `×${value.toFixed(2)}`,
              },
              {
                id: "share",
                name: "Arithmetic share of each batch",
                tone: "attention",
                dash: "dashed",
                points: arithmeticShareCurve(stable, latePhase).map((point) => ({ x: point.x * STEPS, y: point.y })),
                format: (value) => percent(value),
              },
            ]}
          />
          <LineChart
            label="Held-out loss along the run"
            xLabel="step"
            yLabel="held-out loss (nats/char)"
            xDomain={[CURVE_FROM_STEP, STEPS]}
            yDomain={lossDomain}
            marker={{ x: SWITCH_STEP, label: `step ${SWITCH_STEP}` }}
            series={lossSeries}
            footnote={`Measured every 30 steps from step ${CURVE_FROM_STEP}; the zero table at step 0 scores ${UNIFORM_CROSS_ENTROPY.toFixed(3)}.`}
          />
        </div>
        <div className="mt-table-wrap">
          <table className="tg-board mt-late-table">
            <caption>
              {switched
                ? "Spread evenly draws, on average, the same pairs from each domain as the switched run, in one constant mixture. Losses are held-out, in nats per character; Forgetting is General at the end minus General at the switch."
                : "Choose Target mix or Target only to switch the mixture when the decay starts. Losses are held-out, in nats per character; Forgetting is General at the end minus General at the switch."}
            </caption>
            <thead>
              <tr>
                <th scope="col">At the end of the run</th>
                {lateColumns.map((column) => (
                  <th scope="col" key={column.id}>
                    {column.title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {["Arithmetic held-out", "General held-out", "Forgetting", "Arithmetic pairs drawn"].map((row, rowIndex) => (
                <tr key={row}>
                  <th scope="row">{row}</th>
                  {lateColumns.map((column) => (
                    <td key={column.id}>{column.values[rowIndex]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Target mix keeps the general domains in their own proportions and makes {percent(TARGET_MIX_SHARE)} of every
          late batch Arithmetic; Target only makes it all Arithmetic. Warmup-stable-decay ramps up over the first{" "}
          {Math.round(WARMUP_FRACTION * 100)}% of steps, holds, then falls linearly to zero over the last{" "}
          {Math.round(DECAY_FRACTION * 100)}%; Constant rate keeps rate {LEARNING_RATE} throughout.
        </p>
      </LabSurface>
    </div>
  );
}
