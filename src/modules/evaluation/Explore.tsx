import { useMemo } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  encodeTinyText,
  type ModuleContext,
} from "@app/module-sdk";
import { itemsForHalfWidth, pairedDifference, parseItems } from "./eval";
import {
  ATTEMPTS_RANGE,
  BENCHMARK_SOURCE,
  BUDGETS,
  CONTROL_TEXT,
  ITEMS_TEXT_MAX,
  LEAKED_TEXT,
  RULES,
  RULE_LABELS,
  SAMPLES_PER_ITEM,
  TRAINING_TEXT,
  itemsNeeded,
  percent,
  rankCheckpoints,
  rateCheckpoints,
  sanitizeEvaluationState,
  trainCheckpoints,
  type RatedCheckpoint,
  type Rule,
} from "./scoring";

const signed = (value: number, digits = 3) =>
  Number.isFinite(value) ? `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}` : "—";
const pct = percent;
const fixed = (value: number, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");

const RULE_NOTES: Record<Rule, string> = {
  normalized:
    "Each continuation is scored by its mean log-probability per character, so a long option is not punished for being long. An item passes when the correct option scores higher.",
  raw: "Each continuation is scored by its total log-probability. Every extra character multiplies in another probability below one, so this rule hands the shorter option a head start whatever the model knows.",
  sample: `Each checkpoint answers by sampling: it picks one of the two options in proportion to the probability it gives each full continuation, ${SAMPLES_PER_ITEM} seeded draws per item. An item scores pass@k, the chance that at least one of k attempts picks the key, estimated from those draws; the checkpoint scores the mean over items.`,
};

/** One row per checkpoint: the headline score with its interval, and the rule's second quantity with its interval. */
function ForestPlot({
  ranked,
  leader,
  n,
  rule,
  attempts,
}: {
  ranked: RatedCheckpoint[];
  leader: RatedCheckpoint;
  n: number;
  rule: Rule;
  attempts: number;
}) {
  const rowHeight = 30;
  const top = 26;
  const height = top + ranked.length * rowHeight + 30;
  const passLeft = 104;
  const passRight = 340;
  const marginLeft = 424;
  const marginRight = 700;
  const passAt = (value: number) => passLeft + value * (passRight - passLeft);
  const sample = rule === "sample";
  const extent = Math.max(
    rule === "raw" ? 1 : 0.25,
    ...ranked.flatMap((row) =>
      Number.isFinite(row.side.low) ? [Math.abs(row.side.low), Math.abs(row.side.high)] : [Math.abs(row.side.mean)],
    ),
  );
  const sideLow = sample ? 0 : -extent;
  const sideHigh = sample ? 1 : extent;
  const sideAt = (value: number) =>
    marginLeft +
    ((Math.max(sideLow, Math.min(sideHigh, value)) - sideLow) / (sideHigh - sideLow)) * (marginRight - marginLeft);
  const yAt = (index: number) => top + index * rowHeight + rowHeight / 2;
  const sideTicks = sample
    ? [0, 0.25, 0.5, 0.75, 1]
    : [-extent, -extent / 2, 0, extent / 2, extent];
  const leftTitle = sample ? `mean pass@${attempts} · t 95% interval` : "pass rate · Wilson 95% interval";
  const rightTitle = sample ? "mean pass@1 · t 95% interval" : "mean margin · t 95% interval";

  const summary = ranked
    .map(
      (row) =>
        `${row.epochs} epochs: ${
          sample ? `mean pass@${attempts} ${pct(row.score)}` : `${row.passes} of ${n} passed`
        }, 95% interval ${pct(row.interval[0])} to ${pct(row.interval[1])}; ${sample ? "mean pass@1" : "mean margin"} ${fixed(row.side.mean)}${Number.isFinite(row.side.low) ? `, interval ${row.side.low.toFixed(3)} to ${row.side.high.toFixed(3)}` : ""}`,
    )
    .join(". ");

  return (
    <svg className="ev-forest" viewBox={`0 0 720 ${height}`} role="img" aria-label={`Leaderboard with 95% intervals. ${summary}.`}>
      <text x={passLeft} y={12}>
        {leftTitle}
      </text>
      <text x={marginLeft} y={12}>
        {rightTitle}
      </text>
      <rect
        className="ev-leader-band"
        x={passAt(leader.interval[0])}
        y={top - 4}
        width={Math.max(1, passAt(leader.interval[1]) - passAt(leader.interval[0]))}
        height={ranked.length * rowHeight + 8}
      />
      {Number.isFinite(leader.side.low) && (
        <rect
          className="ev-leader-band"
          x={sideAt(leader.side.low)}
          y={top - 4}
          width={Math.max(1, sideAt(leader.side.high) - sideAt(leader.side.low))}
          height={ranked.length * rowHeight + 8}
        />
      )}
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={`p${tick}`}>
          <line className="ev-grid" x1={passAt(tick)} x2={passAt(tick)} y1={top - 4} y2={top + ranked.length * rowHeight + 4} />
          <text x={passAt(tick)} y={height - 12} textAnchor="middle">
            {pct(tick)}
          </text>
        </g>
      ))}
      {sideTicks.map((tick, index) => (
        <g key={`m${index}`}>
          <line
            className={!sample && tick === 0 ? "ev-zero" : "ev-grid"}
            x1={sideAt(tick)}
            x2={sideAt(tick)}
            y1={top - 4}
            y2={top + ranked.length * rowHeight + 4}
          />
          <text x={sideAt(tick)} y={height - 12} textAnchor="middle">
            {sample ? pct(tick) : tick === 0 ? "0" : signed(tick, 2)}
          </text>
        </g>
      ))}
      {ranked.map((row, index) => {
        const y = yAt(index);
        const isLeader = row === leader;
        return (
          <g key={row.epochs} className={`ev-row${isLeader ? " is-leader" : ""}`}>
            <text className="ev-row-label" x={8} y={y + 4}>
              {row.epochs} epochs {isLeader ? "★" : ""}
            </text>
            <line className="ev-whisker" x1={passAt(row.interval[0])} x2={passAt(row.interval[1])} y1={y} y2={y} />
            <line className="ev-whisker" x1={passAt(row.interval[0])} x2={passAt(row.interval[0])} y1={y - 5} y2={y + 5} />
            <line className="ev-whisker" x1={passAt(row.interval[1])} x2={passAt(row.interval[1])} y1={y - 5} y2={y + 5} />
            <circle className="ev-point" cx={passAt(row.score)} cy={y} r={4.5} />
            <text className="ev-value" x={passRight + 8} y={y + 4}>
              {sample ? pct(row.score) : `${row.passes}/${n}`}
            </text>
            {Number.isFinite(row.side.low) && (
              <>
                <line className="ev-whisker" x1={sideAt(row.side.low)} x2={sideAt(row.side.high)} y1={y} y2={y} />
                <line className="ev-whisker" x1={sideAt(row.side.low)} x2={sideAt(row.side.low)} y1={y - 5} y2={y + 5} />
                <line className="ev-whisker" x1={sideAt(row.side.high)} x2={sideAt(row.side.high)} y1={y - 5} y2={y + 5} />
              </>
            )}
            <rect className="ev-point" x={sideAt(row.side.mean) - 4} y={y - 4} width={8} height={8} />
          </g>
        );
      })}
    </svg>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const settings = sanitizeEvaluationState(state);
  const itemsText = settings.items as string;
  const contaminated = settings.contaminated as boolean;
  const epochsScale = settings.epochsScale as number;
  const rule = settings.rule as Rule;
  const attempts = settings.attempts as number;
  const sample = rule === "sample";
  const pairedName = sample ? `pass@${attempts}` : "margin";

  const items = useMemo(() => parseItems(itemsText), [itemsText]);

  /** Both conditions are trained, so the leak can be compared item by item against the clean run. */
  const cleanRuns = useMemo(() => trainCheckpoints(TRAINING_TEXT, items, epochsScale), [epochsScale, items]);
  const leakedRuns = useMemo(() => trainCheckpoints(LEAKED_TEXT, items, epochsScale), [epochsScale, items]);
  /** Changing the rule re-scores the trained checkpoints; it never retrains them. */
  const clean = useMemo(() => rateCheckpoints(cleanRuns, rule, attempts), [cleanRuns, rule, attempts]);
  const leaked = useMemo(() => rateCheckpoints(leakedRuns, rule, attempts), [leakedRuns, rule, attempts]);
  const scored = contaminated ? leaked : clean;
  const trainingText = contaminated ? LEAKED_TEXT : TRAINING_TEXT;

  const ranked = rankCheckpoints(scored);
  const leader = ranked[0];
  const runnerUp = ranked[1];
  const strongest = scored[scored.length - 1];
  const n = items.length;

  const overlap = ranked.slice(1).filter((row) => row.interval[1] >= leader.interval[0]).length;
  const leaderGap = runnerUp
    ? pairedDifference(
        leader.items.map((item) => item.paired),
        runnerUp.items.map((item) => item.paired),
      )
    : null;
  const gapResolved = leaderGap && Number.isFinite(leaderGap.low) && (leaderGap.low > 0 || leaderGap.high < 0);
  const needed = n > 0 ? itemsNeeded(leader, 0.05) : itemsForHalfWidth(0.5, 0.05);

  const cleanStrongest = clean[clean.length - 1];
  const leakedStrongest = leaked[leaked.length - 1];
  const leakShift = pairedDifference(
    leakedStrongest.items.map((item) => item.paired),
    cleanStrongest.items.map((item) => item.paired),
  );
  const controlShift = leakedStrongest.controlPerplexity / cleanStrongest.controlPerplexity - 1;
  const marginExtent = Math.max(
    rule === "raw" ? 1 : 0.2,
    ...strongest.items.map((item) => Math.abs(item.side)),
  );

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="Leaderboard" className="tg-board-panel">
        <SurfaceHeading
          kicker={`${n} multiple-choice items · ${BUDGETS.length} checkpoints · 95% intervals · ${RULE_LABELS[rule]}`}
          title={contaminated ? "Leaderboard, with the benchmark source leaked into training" : "Leaderboard"}
          aside={
            <span className={`tg-badge ${contaminated ? "is-warning" : ""}`.trim()}>
              {contaminated ? "contaminated" : "clean"}
            </span>
          }
        />
        <div className="ev-rule-controls">
          <SegmentedControl
            label="Scoring rule"
            value={rule}
            options={RULES.map((value) => ({ value, label: RULE_LABELS[value] }))}
            onChange={(value) => {
              setState({ rule: value });
              narrate(`Scoring rule: ${RULE_LABELS[value as Rule]}. The checkpoints are re-scored, not retrained.`);
            }}
          />
          {sample && (
            <RangeControl
              label="Attempts allowed (k)"
              min={ATTEMPTS_RANGE.min}
              max={Math.min(ATTEMPTS_RANGE.max, SAMPLES_PER_ITEM)}
              step={1}
              value={attempts}
              format={(value) => `k = ${value}`}
              onChange={(value) => setState({ attempts: value })}
            />
          )}
        </div>
        <p className="lab-note">{RULE_NOTES[rule]}</p>
        {n > 0 && (
          <div className="ev-forest-wrap">
            <ForestPlot ranked={ranked} leader={leader} n={n} rule={rule} attempts={attempts} />
          </div>
        )}
        <div className="ev-board-scroll">
          <table className="tg-board">
            <thead>
              <tr>
                <th scope="col">Checkpoint</th>
                <th scope="col">{sample ? `Mean pass@${attempts}` : "Passed"}</th>
                <th scope="col">95% interval</th>
                <th scope="col">{sample ? "Mean pass@1" : "Mean margin"}</th>
                <th scope="col">Benchmark perplexity</th>
                <th scope="col">Control perplexity</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((model) => (
                <tr key={model.epochs} className={model === leader ? "is-leader" : ""}>
                  <th scope="row">{model.epochs} epochs</th>
                  <td>{sample ? pct(model.score) : `${model.passes} / ${n}`}</td>
                  <td>
                    {pct(model.interval[0])}–{pct(model.interval[1])}
                  </td>
                  <td>{sample ? pct(model.side.mean) : fixed(model.side.mean)}</td>
                  <td>{model.benchmarkPerplexity.toFixed(2)}</td>
                  <td>{model.controlPerplexity.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric
            label="Leader's standard error"
            value={n > 0 ? `± ${fixed(leader.standardError * 100, 1)} pts` : "—"}
          />
          <Metric
            label="Intervals overlapping the leader's"
            value={`${overlap} of ${Math.max(0, ranked.length - 1)}`}
            tone={overlap > 0 ? "loss" : "forward"}
          />
          <Metric label="Items for ±5 pts" value={`${needed}`} />
        </div>
        <div className={`tg-callout ${gapResolved ? "is-quiet" : "is-warning"}`}>
          <strong>
            {leaderGap && Number.isFinite(leaderGap.low)
              ? `Leader minus runner-up, paired over the same ${n} items: ${signed(leaderGap.mean)} ${pairedName}, 95% interval ${signed(leaderGap.low)} to ${signed(leaderGap.high)}.`
              : "Add at least two items to compare checkpoints."}
          </strong>
          <span>
            {gapResolved
              ? `That interval excludes zero, so on these items the leader's ${pairedName} is reliably higher. It still says nothing about items you did not write.`
              : "That interval includes zero, so this item set cannot say which of the two is better. A rank printed without an interval hides exactly this."}{" "}
            {sample
              ? `Mean pass@${attempts} is drawn with a t interval over items, because it averages fractional item scores rather than counting passes; the ${SAMPLES_PER_ITEM} draws per item add sampling noise it does not include.`
              : `Pass rates are drawn with Wilson intervals because p ± 1.96·√(p(1−p)/n) collapses to zero width at 0/${n} or ${n}/${n}.`}
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Item-level results" className="tg-eval-summary">
        <SurfaceHeading
          kicker={`Largest checkpoint · ${strongest.epochs} epochs · ${RULE_LABELS[rule]}`}
          title="Item by item, with both scores"
        />
        <ol className="tg-item-list ev-item-list">
          {strongest.items.map((item, index) => (
            <li key={index} className={item.passed ? "is-pass" : "is-fail"}>
              <span>{item.context}</span>
              <code>
                {sample
                  ? `${item.correctSamples} of ${SAMPLES_PER_ITEM} samples`
                  : rule === "raw"
                    ? `${item.correctTotal.toFixed(2)} ${item.passed ? "›" : "‹"} ${item.distractorTotal.toFixed(2)}`
                    : `${item.correctMean.toFixed(2)} ${item.passed ? "›" : "‹"} ${item.distractorMean.toFixed(2)}`}
              </code>
              <b>{sample ? `${pct(item.score)} pass@${attempts}` : signed(item.side)}</b>
              <i>
                {sample
                  ? `“${item.correct}” was drawn ${item.correctSamples} times against “${item.distractor}”`
                  : `“${item.correct}” ${item.passed ? "beat" : "lost to"} “${item.distractor}”`}
              </i>
              <em className="ev-margin-bar" aria-hidden="true">
                <u
                  style={
                    sample
                      ? { left: "0%", width: `${item.score * 100}%` }
                      : {
                          left: item.side >= 0 ? "50%" : `${50 - (Math.abs(item.side) / marginExtent) * 50}%`,
                          width: `${(Math.abs(item.side) / marginExtent) * 50}%`,
                        }
                  }
                />
              </em>
            </li>
          ))}
          {strongest.items.length === 0 && (
            <p className="lab-note">
              Each line needs three parts: a context, the correct continuation, and a plausible wrong one.
            </p>
          )}
        </ol>
        <div className="metric-row">
          <Metric
            label={sample ? `Mean pass@${attempts}` : "Pass rate"}
            value={`${pct(strongest.score)} (${pct(strongest.interval[0])}–${pct(strongest.interval[1])})`}
            tone="forward"
          />
          <Metric
            label={sample ? "Mean pass@1" : "Mean margin"}
            value={sample ? pct(strongest.side.mean) : fixed(strongest.side.mean)}
          />
          <Metric label="Control perplexity" value={strongest.controlPerplexity.toFixed(2)} tone="loss" />
        </div>
        <div className={`tg-callout ${contaminated ? "is-warning" : "is-quiet"}`}>
          <strong>
            {contaminated
              ? `Leak shift at ${strongest.epochs} epochs: ${pairedName} ${signed(leakShift.mean)} per item (95% interval ${signed(leakShift.low)} to ${signed(leakShift.high)}), control perplexity ${signed(controlShift * 100, 1)}%.`
              : `Benchmark ${sample ? "mean pass@1" : "margin"} ${sample ? pct(strongest.side.mean) : fixed(strongest.side.mean)} on text the model has never seen.`}
          </strong>
          <span>
            {contaminated
              ? `Both numbers compare the leaked run with the clean run at the same budget and seed. The control improves a little because the leak added real English, but the benchmark ${pairedName} moves far more. That disproportion, not a frozen control, is the signature.`
              : `Turn contamination on and compare how far the benchmark ${pairedName} moves against how far the control perplexity moves. The disproportion between them is the signature contamination leaves.`}
          </span>
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Benchmark editor" className="tg-tests-panel">
          <SurfaceHeading kicker="Your harness" title="Write the items" />
          <label className="tg-editor">
            <span>context | correct | plausible distractor</span>
            <textarea
              aria-label="Benchmark items"
              rows={7}
              maxLength={ITEMS_TEXT_MAX}
              value={itemsText}
              onChange={(event) => setState({ items: event.target.value })}
            />
          </label>
          <RangeControl
            label="Training budget multiplier"
            min={0.25}
            max={3}
            step={0.25}
            value={epochsScale}
            format={(value) => `${value.toFixed(2)}× epochs`}
            onChange={(value) => setState({ epochsScale: value })}
          />
          <div className="metric-row">
            <Metric label="Training characters" value={`${encodeTinyText(trainingText).length}`} />
            <Metric label="Benchmark source" value={`${encodeTinyText(BENCHMARK_SOURCE).length}`} />
            <Metric label="Control" value={`${encodeTinyText(CONTROL_TEXT).length}`} />
          </div>
          <p className="lab-note">
            A good distractor is wrong but ordinary: “evening” shares almost every character pattern with “morning”, so
            only a model that learned this particular text can separate them.
          </p>
        </LabSurface>

        <LabSurface label="Contamination control" className="tg-contamination-panel">
          <SurfaceHeading kicker="The experiment that ruins benchmarks" title="Leak the source text" />
          <div className="tg-switch-list">
            <button
              type="button"
              className="tg-switch"
              aria-pressed={contaminated}
              onClick={() => {
                setState({ contaminated: !contaminated });
                narrate(
                  contaminated
                    ? "Contamination off. The benchmark sentences are held out again."
                    : "Contamination on. The three benchmark sentences are now in the training corpus.",
                );
              }}
            >
              <strong>Add the benchmark source sentences to the corpus</strong>
              <small>
                Not the questions themselves — the three sentences they were written from, which is how a scraped page
                leaks a public benchmark.
              </small>
              <b>{contaminated ? "leaking" : "held out"}</b>
            </button>
          </div>
          <div className="tg-sample">
            <span>Held-out benchmark source</span>
            <p>{BENCHMARK_SOURCE}</p>
          </div>
          <div className="metric-row">
            <Metric label={sample ? `Pass@${attempts} shift` : "Margin shift"} value={signed(leakShift.mean)} tone="loss" />
            <Metric label="Control shift" value={`${signed(controlShift * 100, 1)}%`} />
          </div>
          <p className="lab-note">
            Both shifts compare the {leakedStrongest.epochs}-epoch run trained with the leak against the same run
            without it, so they are measured whichever way the switch is set.
          </p>
        </LabSurface>
      </div>
    </div>
  );
}
