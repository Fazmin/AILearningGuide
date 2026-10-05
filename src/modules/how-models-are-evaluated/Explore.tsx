import { useMemo } from "react";
import {
  Heatmap,
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
  const heat = [
    [counted.tn / Math.max(1, rows.length), counted.fp / Math.max(1, rows.length)],
    [counted.fn / Math.max(1, rows.length), counted.tp / Math.max(1, rows.length)],
  ];
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
          <RangeControl
            label="Decision threshold"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ threshold: value })}
          />
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

        <LabSurface label="Train, val, test" className="split-card">
          <SurfaceHeading
            kicker={
              rare
                ? "70 / 15 / 15 · one positive, in test"
                : `70 / 15 / 15 · ${positivesIn("train")} / ${positivesIn("val")} / ${positivesIn("test")} positives`
            }
            title="A holdout dies if you choose t on it"
          />
          <div
            className="split-strip"
            role="img"
            aria-label={
              rare
                ? "One hundred rows: 70 train, 15 val, 15 test. Row 99 is the only positive. Row 12 is a confident false alarm."
                : `One hundred rows: 70 train, 15 val, 15 test. Twenty are positive: ${positivesIn("train")} in train, ${positivesIn("val")} in val, ${positivesIn("test")} in test, outlined in the strip.`
            }
          >
            {examples.map((example) => (
              <i
                key={example.id}
                className={`is-${example.split}${example.label === 1 ? " is-positive" : ""}${rare && example.id === 12 ? " is-alarm" : ""}`}
                title={`row ${example.id} · ${example.split} · y=${example.label} · score ${example.score.toFixed(2)}`}
              />
            ))}
          </div>
          <div className="split-legend eval-split-legend">
            <span>
              <i className="is-train" />
              train 0–69
            </span>
            <span>
              <i className="is-val" />
              val 70–84
            </span>
            <span>
              <i className="is-test" />
              test 85–99
            </span>
            {rare ? (
              <>
                <span>
                  <i className="is-positive" />
                  row 99 · y = 1
                </span>
                <span>
                  <i className="is-alarm" />
                  row 12 · score 0.88
                </span>
              </>
            ) : (
              <span>
                <i className="is-positive" />
                outlined · y = 1
              </span>
            )}
          </div>
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

        <LabSurface label="Confusion matrix" className="confusion-card">
          <SurfaceHeading kicker="True class versus predicted class" title="Four counts, one table" />
          <div className="confusion-matrix">
            <Heatmap
              label="Confusion matrix as shares of the scored rows"
              rows={["true 0", "true 1"]}
              columns={["pred 0", "pred 1"]}
              values={heat}
            />
            <table className="tg-board">
              <thead>
                <tr>
                  <th scope="col"> </th>
                  <th scope="col">Pred 0</th>
                  <th scope="col">Pred 1</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">True 0</th>
                  <td>{counted.tn} TN</td>
                  <td>{counted.fp} FP</td>
                </tr>
                <tr className={counted.fn > 0 && counted.tp === 0 ? "is-leader" : ""}>
                  <th scope="row">True 1</th>
                  <td>{counted.fn} FN</td>
                  <td>{counted.tp} TP</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="lab-note">
            Heatmap cells are shares of the {rows.length} scored rows. The table is the raw counts every scalar score on
            this page is computed from. Accuracy = (TP+TN)/N, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 =
            2PR/(P+R) = 2TP/(2TP+FP+FN).
          </p>
        </LabSurface>

        <LabSurface label="Four readings" className="metrics-card">
          <SurfaceHeading kicker="Accuracy is not the other three" title="The same matrix, four readings" />
          <div className="metric-row">
            <Metric
              label="Accuracy"
              value={percent(counted.accuracy)}
              tone={counted.accuracy > 0.75 && (counted.recall ?? 0) === 0 ? "loss" : "forward"}
            />
            <Metric label="Precision" value={percent(counted.precision)} />
            <Metric label="Recall" value={percent(counted.recall)} tone={counted.recall === 0 ? "loss" : "forward"} />
            <Metric label="F1" value={percent(counted.f1)} />
          </div>
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
          <p className="lab-note">
            AUC is the chance a random positive scores above a random negative. It reads the ranking, not any one
            threshold, so a detector can have AUC near 1 while the published threshold never fires.{" "}
            {rare
              ? "With a single positive the curve is one step and the number is fragile. "
              : "With twenty positives the curve has many small steps, so one row moves it far less than with a single positive. "}
            Calibration asks a different question — do rows scored 0.4 turn out positive 40% of the time? — and a
            reliability plot needs many rows per bin, so this page does not draw one for either population.
          </p>
          <div className="metric-row">
            <Metric label="Precision if 1 row in 5 is positive" value={percent(rateAtOneInFive)} />
            <Metric label="Precision if 1 row in 100 is positive" value={percent(rateAtOneInHundred)} tone="loss" />
          </div>
          <p className="lab-note">
            Base rate: these two readings keep this rule&apos;s true positive rate and false positive rate and change only
            the share of positives, using precision = TPR·π / (TPR·π + FPR·(1−π)). The detector does not change; its
            precision does, because the false alarms come from the negatives and there are far more of them when
            positives are rare.
          </p>
          <p className="lab-note">
            Loss versus metric: training would minimize a differentiable loss on scores (for example cross-entropy).
            These four numbers are published readings of a thresholded matrix.{" "}
            {rare
              ? "Maximizing accuracy on this set prefers never firing; maximizing recall prefers t ≤ 0.42. Those two searches disagree."
              : `On these rows accuracy peaks at t = ${bestAccuracy.threshold.toFixed(2)} and F1 at t = ${bestF1.threshold.toFixed(2)}, while recall alone is best at the lowest t that still catches every positive. The searches disagree.`}
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
          <div className="split-strip eval-fold-strip" role="img" aria-label={`Five folds of twenty. Fold ${fold} is held out.`}>
            {examples.map((example) => (
              <i
                key={example.id}
                className={`${foldOf(example) === fold ? "is-val" : "is-train"}${example.label === 1 ? " is-positive" : ""}`}
              />
            ))}
          </div>
          <table className="tg-board fit-table">
            <thead>
              <tr>
                <th scope="col">Fold</th>
                <th scope="col">Always-neg acc</th>
                <th scope="col">Recall</th>
                <th scope="col">Positives in fold</th>
              </tr>
            </thead>
            <tbody>
              {foldMatrices.map((matrix, index) => (
                <tr key={index} className={index === fold ? "is-leader" : ""}>
                  <td>{index}</td>
                  <td>{(matrix.accuracy * 100).toFixed(0)}%</td>
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
