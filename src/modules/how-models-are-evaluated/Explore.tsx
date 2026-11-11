import { useMemo } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  COMMON_PROCESS,
  FRESH_ROWS,
  GOALS,
  bestEpoch,
  bestThreshold,
  countMatrix,
  curvePoints,
  curveValue,
  examplesFor,
  foldOf,
  leakReport,
  precisionAtPrevalence,
  readingOf,
  regressionErrors,
  roc,
  type Baseline,
  type CurveKind,
  type Goal,
  type Population,
} from "./metrics";
import { ResidualPlot, RocPlot, ScoreStrip, SweepPlot } from "./plots";
import { ClaimPlot, ConfusionGrid, ErrorBars, FoldCells, InlineBar, ReadingBars, SliceRail, type Reading } from "./views";

const REGRESSION = [
  { x: 0, y: 0.2 },
  { x: 1, y: 1.1 },
  { x: 2, y: 1.9 },
  { x: 3, y: 3.2 },
] as const;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** A reading that is 0/0 on this slice prints n/a rather than a misleading 0%. */
const percent = (value: number | null, digits = 1) => (value === null ? "n/a" : `${(value * 100).toFixed(digits)}%`);
/** A signed gap in percentage points, or n/a when either reading is 0/0 or was not computed. */
const points = (value: number | null) =>
  value === null ? "n/a" : `${value < 0 ? "−" : "+"}${Math.abs(value * 100).toFixed(1)} pts`;
const GOAL_NAME: Record<Goal, string> = { accuracy: "accuracy", recall: "recall", f1: "F1" };

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const threshold = clamp(asNumber(state, "threshold", 0.5), 0, 1);
  const baseline = (["learned", "always-negative", "always-positive"].includes(
    asString(state, "baseline", "always-negative"),
  )
    ? asString(state, "baseline", "always-negative")
    : "always-negative") as Baseline;
  const split = ["all", "train", "val", "test"].includes(asString(state, "split", "all"))
    ? asString(state, "split", "all")
    : "all";
  const tuneSplit = ["train", "val", "test"].includes(asString(state, "tuneSplit", "val"))
    ? asString(state, "tuneSplit", "val")
    : "val";
  const goal = ((GOALS as readonly string[]).includes(asString(state, "goal", "accuracy"))
    ? asString(state, "goal", "accuracy")
    : "accuracy") as Goal;
  const population: Population = asString(state, "population", "rare") === "common" ? "common" : "rare";
  const rare = population === "rare";
  const curve = (["underfit", "good", "overfit"].includes(asString(state, "curve", "overfit"))
    ? asString(state, "curve", "overfit")
    : "overfit") as CurveKind;
  const fold = clamp(Math.round(asNumber(state, "fold", 4)), 0, 4);
  const outlier = clamp(asNumber(state, "outlier", 12), 3, 16);

  const examples = useMemo(() => examplesFor(population), [population]);
  const rows = useMemo(() => examples.filter((example) => split === "all" || example.split === split), [examples, split]);
  const scoredIds = useMemo(() => new Set(rows.map((example) => example.id)), [rows]);
  const counted = useMemo(() => countMatrix(rows, baseline, threshold), [baseline, rows, threshold]);
  const positives = rows.filter((example) => example.label === 1).length;
  const ranking = useMemo(() => roc(rows), [rows]);
  const positivesIn = (name: "train" | "val" | "test") =>
    examples.filter((example) => example.split === name && example.label === 1).length;
  const bestAccuracy = useMemo(() => bestThreshold(rows, "accuracy"), [rows]);
  const bestF1 = useMemo(() => bestThreshold(rows, "f1"), [rows]);
  const rateAtOneInFive = precisionAtPrevalence(counted, 0.2);
  const rateAtOneInHundred = precisionAtPrevalence(counted, 0.01);

  const tunePool = examples.filter((example) => example.split === tuneSplit);
  const testPool = examples.filter((example) => example.split === "test");
  const chosen = bestThreshold(tunePool, goal);
  const testAfterTune = countMatrix(testPool, "learned", chosen.threshold);
  const leaked = tuneSplit === "test";
  const tuneHasPositives = tunePool.some((example) => example.label === 1);
  const leak = useMemo(() => leakReport(examples, goal, rare ? null : FRESH_ROWS), [examples, goal, rare]);
  const goalName = GOAL_NAME[goal];

    const trainCurve = useMemo(() => curvePoints(curve, "train"), [curve]);
  const valCurve = useMemo(() => curvePoints(curve, "val"), [curve]);
  const stopEpoch = bestEpoch(curve);

  const foldRows = examples.filter((example) => foldOf(example) === fold);
  const foldCounted = countMatrix(foldRows, "always-negative", 1);
  const foldMatrices = [0, 1, 2, 3, 4].map((index) =>
    countMatrix(examples.filter((example) => foldOf(example) === index), "always-negative", 1),
  );
  const foldAccuracies = foldMatrices.map((matrix) => matrix.accuracy);
  const foldPositives = foldMatrices.map((matrix) => matrix.tp + matrix.fn);
  const foldsWithPositives = foldPositives.filter((count) => count > 0).length;
  const cvMean = foldAccuracies.reduce((sum, value) => sum + value, 0) / foldAccuracies.length;

  const regression = [...REGRESSION, { x: 4, y: outlier }];
  const residuals = regression.map((point) => point.y - point.x);
  const { mae, mse, rmse } = regressionErrors(residuals);

  const applyTuned = () => {
    setState({ threshold: chosen.threshold, baseline: "learned" });
    narrate(
      leaked
        ? `Threshold ${chosen.threshold.toFixed(2)} was chosen on test. The test ${goalName} is no longer a holdout.`
        : `Threshold ${chosen.threshold.toFixed(2)} was chosen on ${tuneSplit} to raise ${goalName}. Test is still unused for that choice.`,
    );
  };

  const readings: Reading[] = [
    {
      id: "accuracy",
      name: "Accuracy",
      formula: `(TP+TN)/N = ${counted.tp + counted.tn}/${counted.n}`,
      parts: [
        { code: `TP ${counted.tp}`, amount: counted.tp, good: true },
        { code: `TN ${counted.tn}`, amount: counted.tn, good: true },
        { code: `FP ${counted.fp}`, amount: counted.fp, good: false },
        { code: `FN ${counted.fn}`, amount: counted.fn, good: false },
      ],
      value: counted.accuracy,
      alert: counted.accuracy > 0.75 && (counted.recall ?? 0) === 0,
    },
    {
      id: "precision",
      name: "Precision",
      formula: `TP/(TP+FP) = ${counted.tp}/${counted.tp + counted.fp}`,
      parts: [
        { code: `TP ${counted.tp}`, amount: counted.tp, good: true },
        { code: `FP ${counted.fp}`, amount: counted.fp, good: false },
      ],
      value: counted.precision,
      empty: "n/a · nothing was predicted positive (0/0)",
    },
    {
      id: "recall",
      name: "Recall",
      formula: `TP/(TP+FN) = ${counted.tp}/${counted.tp + counted.fn}`,
      parts: [
        { code: `TP ${counted.tp}`, amount: counted.tp, good: true },
        { code: `FN ${counted.fn}`, amount: counted.fn, good: false },
      ],
      value: counted.recall,
      alert: counted.recall === 0,
      empty: "n/a · no positives in this split (0/0)",
    },
    {
      id: "f1",
      name: "F1",
      formula: `2TP/(2TP+FP+FN) = ${2 * counted.tp}/${2 * counted.tp + counted.fp + counted.fn}`,
      parts: [
        { code: `2 × TP ${counted.tp}`, amount: 2 * counted.tp, good: true },
        { code: `FP ${counted.fp}`, amount: counted.fp, good: false },
        { code: `FN ${counted.fn}`, amount: counted.fn, good: false },
      ],
      value: counted.f1,
      empty: "n/a · no positives and no alarms (0/0)",
    },
  ];
  const baseParts = (precision: number | null) =>
    precision === null
      ? []
      : [
          { code: `${(precision * 100).toFixed(0)} real`, amount: precision, good: true },
          { code: `${((1 - precision) * 100).toFixed(0)} false alarms`, amount: 1 - precision, good: false },
        ];

  const firing =
    baseline === "always-negative"
      ? "Always negative never fires, whatever the threshold."
      : baseline === "always-positive"
        ? "Always positive fires on every row, whatever the threshold."
        : !rare
          ? `At t = ${threshold.toFixed(2)} the rule fires on ${counted.tp + counted.fp} of ${rows.length} scored rows, ${counted.tp} of them real positives. Raising t drops false alarms first and then real positives: the two groups of scores overlap, so no t separates them.`
          : threshold > 0.88
            ? "Above 0.88 nothing fires: not row 12, not row 99."
            : threshold > 0.42
              ? "Between 0.43 and 0.88 only row 12, a negative, fires: one false alarm, and row 99 is missed."
              : threshold > 0.256
                ? "At 0.42 or below row 99 fires too: one true positive and one false alarm."
                : "At 0.25 or below the ordinary negatives, which score up to 0.256, start firing as false alarms.";

  return (
    <div className="gw-shell">
      <div className="tg-lab tg-lab--hero gw-lab eval-lab">
        <LabSurface label="Threshold and baseline" className="threshold-card">
          <SurfaceHeading
            kicker={`${rows.length} scored rows · ${positives} real positive${positives === 1 ? "" : "s"}`}
            title={rare ? "A detector on a deliberately rare class" : "A detector on a class that is one row in five"}
            aside={
              <span className="tg-badge">
                {baseline === "learned" ? `score ≥ ${threshold.toFixed(2)}` : baseline.replace("-", " ")}
              </span>
            }
          />
          <div className="population-control">
            <SegmentedControl
              label="Population"
              value={population}
              options={[
                { value: "rare", label: "1 in 100 (rare)" },
                { value: "common", label: "1 in 5 (about 20%)" },
              ]}
              onChange={(value) => {
                setState({ population: value });
                narrate(
                  value === "rare"
                    ? "Population set to 1 in 100: one positive among 100 rows."
                    : "Population set to 1 in 5: twenty positives among 100 rows.",
                );
              }}
            />
          </div>
          <div className="baseline-control">
            <SegmentedControl
              label="Baseline"
              value={baseline}
              options={[
                { value: "always-negative", label: "always negative" },
                { value: "always-positive", label: "always positive" },
                { value: "learned", label: "score ≥ threshold" },
              ]}
              onChange={(value) => setState({ baseline: value })}
            />
          </div>
          <div className="split-control">
            <SegmentedControl
              label="Scored split"
              value={split}
              options={[
                { value: "all", label: "all 100" },
                { value: "train", label: "train" },
                { value: "val", label: "val" },
                { value: "test", label: "test" },
              ]}
              onChange={(value) => setState({ split: value })}
            />
          </div>
          <RangeControl
            label="Decision threshold"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ threshold: value })}
          />
          <ScoreStrip examples={examples} population={population} threshold={threshold} baseline={baseline} scored={scoredIds} />
          <ul className="gw-legend" aria-hidden="true">
            {rare ? (
              <>
                <li>
                  <b className="eval-key is-negative">●</b>negative row
                </li>
                <li>
                  <b className="eval-key is-positive">◆</b>row 99, the one positive
                </li>
                <li>
                  <b className="eval-key is-alarm">●</b>row 12, high-scoring negative
                </li>
              </>
            ) : (
              <>
                <li>
                  <b className="eval-key is-negative">●</b>negative row (circle)
                </li>
                <li>
                  <b className="eval-key is-positive">◆</b>positive row (diamond)
                </li>
              </>
            )}
            <li>
              <b className="eval-key is-fired">●</b>false alarm (FP)
            </li>
            <li>
              <b className="eval-key is-missed">◇</b>missed positive (FN)
            </li>
            <li>
              <b className="eval-key is-faded">●</b>outside the scored split
            </li>
          </ul>
          <p className="lab-note">
            {firing}{" "}
            {rare
              ? "The only real positive is row 99 in the test slice, with score 0.42. Row 12 is a negative the detector scores 0.88: a confident false alarm. The scores are authored, not a trained loss; the metrics below are readings of the thresholded matrix."
              : `Twenty of the 100 rows are positive: ${positivesIn("train")} in train, ${positivesIn("val")} in val, ${positivesIn("test")} in test. Positives score about ${COMMON_PROCESS.positiveMean.toFixed(2)} and negatives about ${COMMON_PROCESS.negativeMean.toFixed(2)}, with a spread. The scores are drawn from a fixed seed, not a trained loss; the metrics below are readings of the thresholded matrix.`}
          </p>
        </LabSurface>

        <LabSurface label="Confusion matrix" className="confusion-card gw-wide">
          <SurfaceHeading kicker="True class versus predicted class" title="Four counts, one table" />
          <ConfusionGrid counted={counted} />
          <p className="lab-note">
            Each cell holds its share of the {rows.length} scored rows, drawn with the marks from the score strip. These
            four counts are what every scalar score on this page is computed from. Accuracy = (TP+TN)/N, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 =
            2PR/(P+R) = 2TP/(2TP+FP+FN).
          </p>
        </LabSurface>

        <LabSurface label="Four readings" className="metrics-card gw-wide">
          <SurfaceHeading kicker="Accuracy is not the other three" title="The same matrix, four readings" />
          <ReadingBars label="Four readings of the confusion matrix" readings={readings} />
          <ul className="gw-legend eval-reading-key" aria-hidden="true">
            <li>
              <i className="eval-swatch is-good" />
              solid: rows that count for the reading
            </li>
            <li>
              <i className="eval-swatch is-bad" />
              hatched: rows that count against it
            </li>
          </ul>
          <p className="lab-note">
            {counted.tp === 0 && counted.fn > 0 && counted.accuracy > 0.75
              ? `${percent(counted.accuracy, 0)} accuracy, zero recall. The detector never found a positive. Majority guessing is the baseline that looks fine and is worthless.`
              : counted.recall === null
                ? `Recall is n/a: this split holds no positives, so TP + FN = 0 and there is nothing to find. ${
                    counted.precision === null
                      ? "Precision is n/a too, because nothing was predicted positive."
                      : `Every positive prediction here is a false alarm, so precision is ${percent(counted.precision)}.`
                  } Accuracy can still print 100%.`
                : `Precision is TP/(TP+FP) = ${counted.tp}/${counted.tp + counted.fp}${
                    counted.precision === null ? " (nothing predicted positive, so n/a)" : ""
                  }. Recall is TP/(TP+FN) = ${counted.tp}/${counted.tp + counted.fn}. F1 is their harmonic mean.`}
          </p>
          <div className="eval-curves">
            <SweepPlot rows={rows} threshold={threshold} />
            <RocPlot rows={rows} threshold={threshold} />
          </div>
          <ul className="gw-legend" aria-hidden="true">
            <li>
              <i className="is-forward" />
              accuracy
            </li>
            <li>
              <i className="is-attention is-dashed" />
              precision
            </li>
            <li>
              <i className="is-loss is-dotted" />
              recall
            </li>
            <li>
              <i className="is-positive is-dashdot" />
              F1
            </li>
            <li>
              <i className="is-ink" />
              current t
            </li>
          </ul>
          <p className="gw-caption">
            Both plots score the score ≥ t rule on the scored rows, even while a constant baseline is selected above.
            The sweep shows every reading at every t; a gap is a threshold where that reading is 0/0.{" "}
            {rare
              ? "Accuracy stays near its ceiling almost everywhere; recall is a cliff at 0.42. "
              : `Recall only falls as t rises and precision mostly rises, which is the trade-off; F1 peaks at t = ${bestF1.threshold.toFixed(2)} on these rows. `}
            The ROC traces true positive rate against false positive rate as t falls; the dot is the current t.
          </p>
          <div className="metric-row">
            <Metric label="AUC (ranking)" value={ranking ? ranking.auc.toFixed(3) : "n/a"} tone="forward" />
            <Metric
              label="What AUC means here"
              value={
                ranking
                  ? rare && positives === 1
                    ? `row 99 outranks ${Math.round(ranking.auc * (rows.length - positives))} of ${rows.length - positives} negatives`
                    : `${(ranking.auc * 100).toFixed(1)}% of positive-negative pairs ranked right`
                  : "no positive to rank"
              }
            />
          </div>
          <details className="eval-more">
            <summary>What AUC grades, and what it does not</summary>
            <p className="lab-note">
              AUC is the chance a random positive scores above a random negative. It reads the ranking, not any one
              threshold, so a detector can have AUC near 1 while the published threshold never fires.{" "}
              {rare
                ? "With a single positive the curve is one step and the number is fragile. "
                : "With twenty positives the curve has many small steps, so one row moves it far less than with a single positive. "}
              Calibration asks a different question — do rows scored 0.4 turn out positive 40% of the time? — and a
              reliability plot needs many rows per bin, so this page does not draw one for either population.
            </p>
          </details>
          <ReadingBars
            stacked
            label="Precision of this rule at two base rates"
            readings={[
              {
                id: "fifth",
                name: "Precision if 1 row in 5 is positive",
                formula: "of every 100 alarms",
                parts: baseParts(rateAtOneInFive),
                value: rateAtOneInFive,
                empty: "n/a · no alarms, or a rate that is 0/0 on this split",
              },
              {
                id: "hundredth",
                name: "Precision if 1 row in 100 is positive",
                formula: "of every 100 alarms",
                parts: baseParts(rateAtOneInHundred),
                value: rateAtOneInHundred,
                alert: true,
                empty: "n/a · no alarms, or a rate that is 0/0 on this split",
              },
            ]}
          />
          <p className="lab-note">
            Base rate: these two readings keep this rule&apos;s true positive rate and false positive rate and change only
            the share of positives, using precision = TPR·π / (TPR·π + FPR·(1−π)). The detector does not change; its
            precision does, because the false alarms come from the negatives and there are far more of them when
            positives are rare.
          </p>
          <details className="eval-more">
            <summary>Why these readings are not the training loss</summary>
            <p className="lab-note">
              Loss versus metric: training would minimize a differentiable loss on scores (for example cross-entropy).
              These four numbers are published readings of a thresholded matrix.{" "}
              {rare
                ? "Maximizing accuracy on this set prefers never firing; maximizing recall prefers t ≤ 0.42. Those two searches disagree."
                : `On these rows accuracy peaks at t = ${bestAccuracy.threshold.toFixed(2)} and F1 at t = ${bestF1.threshold.toFixed(2)}, while recall alone is best at the lowest t that still catches every positive. The searches disagree.`}
            </p>
          </details>
        </LabSurface>

        <LabSurface label="Train, val, test" className="split-card">
          <SurfaceHeading
            kicker={
              rare
                ? "70 / 15 / 15 · one positive, in test"
                : `70 / 15 / 15 · ${positivesIn("train")} / ${positivesIn("val")} / ${positivesIn("test")} positives`
            }
            title="A holdout dies if you choose t on it"
          />
          <SliceRail
            examples={examples}
            rare={rare}
            tuneSplit={tuneSplit as "train" | "val" | "test"}
            label={
              (rare
                ? "One hundred rows: 70 train, 15 val, 15 test. Row 99 is the only positive. Row 12 is a confident false alarm."
                : `One hundred rows: 70 train, 15 val, 15 test. Twenty are positive: ${positivesIn("train")} in train, ${positivesIn("val")} in val, ${positivesIn("test")} in test, each marked with a diamond.`) +
              (leaked
                ? " The threshold is chosen on test, the same slice the claim is scored on."
                : ` The threshold is chosen on ${tuneSplit}; the claim is scored on test.`)
            }
          />
          <div className="tune-control">
            <SegmentedControl
              label="Choose t using"
              value={tuneSplit}
              options={[
                { value: "train", label: "train" },
                { value: "val", label: "val" },
                { value: "test", label: "test" },
              ]}
              onChange={(value) => setState({ tuneSplit: value })}
            />
            <SegmentedControl
              label="Maximize"
              value={goal}
              options={[
                { value: "accuracy", label: "accuracy" },
                { value: "recall", label: "recall" },
                { value: "f1", label: "F1" },
              ]}
              onChange={(value) => setState({ goal: value })}
            />
            <button type="button" className="primary-action" onClick={applyTuned}>
              Set t from {tuneSplit} ({chosen.threshold.toFixed(2)})
            </button>
          </div>
          <div className="metric-row">
            <Metric label="t chosen" value={chosen.threshold.toFixed(2)} />
            <Metric
              label={`${tuneSplit} ${goalName} at that t`}
              value={percent(chosen.score)}
            />
            <Metric
              label={`Test ${goalName}`}
              value={percent(readingOf(testAfterTune, goal))}
              tone={leaked ? "loss" : "forward"}
            />
            <Metric label="Holdout?" value={leaked ? "leaked" : "intact"} tone={leaked ? "loss" : "forward"} />
          </div>
          <ClaimPlot
            goalName={goalName}
            current={tuneSplit}
            paths={[
              { id: "val", label: "Intact · t from val", claim: leak.intactTest, fresh: leak.intactFresh },
              { id: "test", label: "Leaked · t from test", claim: leak.leakedTest, fresh: leak.leakedFresh },
            ]}
          />
          <table className="tg-board fit-table eval-leak-table" aria-label={`Intact versus leaked threshold choice, scored on ${goalName}`}>
            <thead>
              <tr>
                <th scope="col">Path · {goalName}</th>
                <th scope="col">t</th>
                <th scope="col">Test claim</th>
                <th scope="col">Fresh rows</th>
                <th scope="col">Claim − fresh</th>
              </tr>
            </thead>
            <tbody>
              {[
                { id: "val", label: "Intact · t from val", t: leak.valThreshold, claim: leak.intactTest, fresh: leak.intactFresh },
                { id: "test", label: "Leaked · t from test", t: leak.testThreshold, claim: leak.leakedTest, fresh: leak.leakedFresh },
              ].map((path) => (
                <tr key={path.id} className={path.id === tuneSplit ? "is-leader" : ""}>
                  <th scope="row">{path.label}</th>
                  <td>{path.t.toFixed(2)}</td>
                  <td>{percent(path.claim)}</td>
                  <td>{percent(path.fresh)}</td>
                  <td>{path.claim === null || path.fresh === null ? "n/a" : points(path.claim - path.fresh)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="metric-row">
            <Metric
              label={`Leak inflation, test ${goalName}`}
              value={points(leak.inflation)}
              tone={leak.inflation !== null && leak.inflation > 0 ? "loss" : undefined}
            />
          </div>
          <p className="lab-note">
            {leaked
              ? rare
                ? "Choosing t on test is leakage. The test number then describes a threshold that already saw the holdout, including row 99. With one positive the gain is a single row; switch Population to 1 in 5 to measure it against rows the choice never saw."
                : `Choosing t on test is leakage. The search kept whatever was lucky about these 15 rows, so the test number rises while the fresh-rows number, scored on ${FRESH_ROWS.length} rows that took no part in the choice, does not follow it.`
              : goal !== "accuracy" && !tuneHasPositives
                ? goal === "recall"
                  ? `${tuneSplit === "train" ? "Train" : "Val"} holds no positives, so recall is 0/0 there for every t: the search has nothing to find and falls back to t = 1.00, which never fires. The intact path is honest and still misses row 99. Stratifying would need at least one positive per slice, and one positive cannot sit in all three.`
                  : `${tuneSplit === "train" ? "Train" : "Val"} holds no positives, so F1 is 0 wherever the rule fires and 0/0 where it does not. The search keeps the highest t that fires at all, a threshold chosen on nothing. The intact path is honest and still misses row 99.`
                : `t is chosen on ${tuneSplit} to raise ${goalName}, then the test slice is scored with that frozen t.${
                    rare
                      ? " Maximizing accuracy on an all-negative slice picks a t that never fires."
                      : ` The Fresh rows column scores each t on ${FRESH_ROWS.length} more rows from the same process that took no part in either choice; a claim above its fresh-rows reading is an overstatement.`
                  }`}
          </p>
        </LabSurface>

        <LabSurface label="Learning curves" className="curves-card">
          <SurfaceHeading kicker="Illustrative train versus val loss" title="Overfitting is a pair of curves, not a vibe" />
          <div className="curve-control">
            <SegmentedControl
              label="Pattern"
              value={curve}
              options={[
                { value: "underfit", label: "underfit" },
                { value: "good", label: "good fit" },
                { value: "overfit", label: "overfit" },
              ]}
              onChange={(value) => setState({ curve: value })}
            />
          </div>
          <LineChart
            label="Train and validation loss"
            xLabel="epoch"
            yLabel="loss (illustrative)"
            xDomain={[0, 20]}
            yDomain={[0, 1.1]}
            marker={{ x: stopEpoch, label: "lowest val loss" }}
            series={[
              { id: "train", name: "train loss at epoch 20", points: trainCurve, tone: "forward", dash: "solid" },
              { id: "val", name: "val loss at epoch 20", points: valCurve, tone: "loss", dash: "dashed" },
            ]}
            footnote={
              curve === "underfit"
                ? "Both curves flatten early and stay high, close together. The hypothesis class or the training budget is too weak for the pattern."
                : curve === "good"
                  ? "Both fall and stay close. The gap is the usual generalization remainder, not a blow-up."
                  : "Train keeps falling; val bottoms out and turns up. That divergence is the diagnosis of overfitting, not a low train number alone."
            }
          />
          <div className="metric-row">
            <Metric label="Lowest val loss" value={`epoch ${stopEpoch} · ${curveValue(curve, "val", stopEpoch).toFixed(3)}`} tone="forward" />
            <Metric
              label="Val rise since then"
              value={(curveValue(curve, "val", 20) - curveValue(curve, "val", stopEpoch)).toFixed(3)}
              tone={curve === "overfit" ? "loss" : undefined}
            />
            <Metric
              label="Val − train at 20"
              value={(curveValue(curve, "val", 20) - curveValue(curve, "train", 20)).toFixed(3)}
            />
          </div>
          <p className="lab-note">
            The vertical guide marks the epoch with the lowest val loss — where early stopping would keep the weights.
            These three pairs are closed-form sketches of the pattern, not a model trained on the 100 rows above; all
            three start near the same loss, as a freshly initialized model would on both slices.
          </p>
        </LabSurface>

        <LabSurface label="Cross-validation" className="cv-card">
          <SurfaceHeading kicker="Five folds of 20 · always-negative accuracy" title="Each slice is held out once" />
          <div className="fold-control">
            <RangeControl
              label="Held-out fold"
              min={0}
              max={4}
              step={1}
              value={fold}
              format={(value) => `fold ${value} · rows ${value * 20}–${value * 20 + 19}`}
              onChange={(value) => setState({ fold: value })}
            />
          </div>
          <table className="tg-board fit-table eval-cv-table" aria-label={`Five folds of twenty rows. Fold ${fold} is held out.`}>
            <thead>
              <tr>
                <th scope="col">Fold</th>
                <th scope="col" className="eval-cv-rows">Its 20 rows · ◆ positive</th>
                <th scope="col">Always-neg acc</th>
                <th scope="col">Recall</th>
                <th scope="col">Positives in fold</th>
              </tr>
            </thead>
            <tbody>
              {foldMatrices.map((matrix, index) => (
                <tr key={index} className={index === fold ? "is-leader" : ""}>
                  <td>
                    {index}
                    {index === fold ? " · held out" : ""}
                  </td>
                  <td className="eval-cv-rows">
                    <FoldCells rows={examples.filter((example) => foldOf(example) === index)} held={index === fold} />
                  </td>
                  <td>
                    <InlineBar value={matrix.accuracy}>{(matrix.accuracy * 100).toFixed(0)}%</InlineBar>
                  </td>
                  <td>{percent(matrix.recall, 0)}</td>
                  <td>
                    {foldPositives[index]}
                    {rare && foldPositives[index] > 0 ? " · row 99" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="metric-row">
            <Metric label="This fold acc" value={`${(foldCounted.accuracy * 100).toFixed(0)}%`} />
            <Metric label="CV mean acc" value={`${(cvMean * 100).toFixed(1)}%`} tone="loss" />
            <Metric label="This fold recall" value={percent(foldCounted.recall, 0)} tone="loss" />
          </div>
          <p className="lab-note">
            Cross-validation rotates which slice is held out so you do not spend the only holdout on a lucky split. On
            this set, mean accuracy of a never-fire rule is {`${(cvMean * 100).toFixed(1)}%`}.{" "}
            {foldsWithPositives === 5
              ? `Every fold holds ${foldPositives[0]} positives, so recall is measured on all five folds, and it is 0% on each: the stratified placement of positives lets every fold catch the failure.`
              : `Fold ${foldPositives.findIndex((count) => count > 0)} is the only fold that can measure recall; on the other four it is 0/0.`}{" "}
            For simplicity these folds span all 100 rows; in practice cross-validation rotates inside train + val and the
            test slice stays out.
          </p>
        </LabSurface>

        <LabSurface label="Regression residuals" className="regression-card">
          <SurfaceHeading kicker="ŷ = x on five points, one outlier" title="MSE squares the miss; MAE does not" />
          <RangeControl
            label="Outlier y"
            min={3}
            max={16}
            step={0.1}
            value={outlier}
            format={(value) => value.toFixed(1)}
            onChange={(value) => setState({ outlier: value })}
          />
          <ResidualPlot points={regression} />
          <ErrorBars
            max={30}
            label={`Mean absolute error ${mae.toFixed(2)}, root mean squared error ${rmse.toFixed(2)}, mean squared error ${mse.toFixed(2)}, on one scale from 0 to 30.`}
            rows={[
              { name: "MAE", value: mae, tone: "plain", note: "mean of |y − ŷ|" },
              { name: "RMSE", value: rmse, tone: "loss", note: "√MSE, back in y's units" },
              { name: "MSE", value: mse, tone: "loss", note: "mean of (y − ŷ)²" },
            ]}
          />
          <table className="tg-board fit-table">
            <thead>
              <tr>
                <th scope="col">x</th>
                <th scope="col">y</th>
                <th scope="col">ŷ = x</th>
                <th scope="col">y − ŷ</th>
                <th scope="col">|y − ŷ|</th>
                <th scope="col">(y − ŷ)²</th>
              </tr>
            </thead>
            <tbody>
              {regression.map((point, index) => (
                <tr key={point.x} className={index === 4 ? "is-leader" : ""}>
                  <td>{point.x}</td>
                  <td>{point.y.toFixed(1)}</td>
                  <td>{point.x.toFixed(1)}</td>
                  <td>{residuals[index].toFixed(2)}</td>
                  <td>{Math.abs(residuals[index]).toFixed(2)}</td>
                  <td>{(residuals[index] ** 2).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="metric-row">
            <Metric label="MAE" value={mae.toFixed(2)} />
            <Metric label="MSE" value={mse.toFixed(2)} tone="loss" />
            <Metric label="RMSE" value={rmse.toFixed(2)} tone="loss" />
            <Metric
              label="Outlier's share of MSE"
              value={`${(((residuals[4] ** 2) / Math.max(1e-9, mse * residuals.length)) * 100).toFixed(1)}%`}
            />
          </div>
          <p className="lab-note">
            The model is the identity ŷ = x, not a fitted line. MAE is the mean absolute residual. MSE is the mean
            square, so the outlier dominates. Drag Outlier y and watch MSE leave MAE behind. This card is a separate toy
            from the 100-row detector.
          </p>
        </LabSurface>
      </div>
    </div>
  );
}
