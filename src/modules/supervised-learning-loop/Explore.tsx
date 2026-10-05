import { useMemo } from "react";
import {
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  StageFlow,
  SurfaceHeading,
  VectorChip,
  type ModuleContext,
} from "@app/module-sdk";
import {
  ALGORITHMS,
  apartmentsFor,
  closedFormLinear,
  collectSplits,
  contourThreshold,
  encodeApartment,
  ENCODINGS,
  featureNames,
  featuresAt,
  GD_STEPS,
  HOLDOUT_SOURCES,
  LOSSES,
  meanLoss,
  planeLine,
  pointLoss,
  predictProbability,
  runFit,
  sliceSegments,
  stepSize,
  TREE_MAX_DEPTH,
  type Algorithm,
  type Encoding,
  type HoldoutSource,
  type LossKind,
} from "./fit";

const GRID = 28;
const X_MIN = 0.6;
const X_MAX = 4.4;
const Y_MIN = -0.2;
const Y_MAX = 1.2;
const STAGES = [
  { id: "encode", name: "Encode", detail: "Row → x" },
  { id: "split", name: "Split", detail: "Train / holdout" },
  { id: "hypothesis", name: "Hypothesis", detail: "Pick H" },
  { id: "fit", name: "Fit", detail: "Drive L down" },
  { id: "holdout", name: "Holdout", detail: "Read, do not fit" },
] as const;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pick = <T extends string>(value: string, allowed: readonly T[], fallback: T): T =>
  (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

const toPlotX = (rooms: number) => (rooms - X_MIN) / (X_MAX - X_MIN);
const toPlotY = (park: number) => (park - Y_MIN) / (Y_MAX - Y_MIN);
const fromPlotX = (x: number) => X_MIN + x * (X_MAX - X_MIN);
const fromPlotY = (y: number) => Y_MIN + y * (Y_MAX - Y_MIN);

/** Nudge only the mark so stacked (rooms, park) rows stay clickable. The fit ignores this. */
const displayNudge = (id: number) => {
  const angle = id * 2.399 + 0.55;
  return { dx: 0.1 * Math.cos(angle), dy: 0.055 * Math.sin(angle) };
};

const fmt = (value: number, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const stage = clamp(Math.round(asNumber(state, "stage", 3)), 0, 4);
  const encoding = pick(asString(state, "encoding", "numeric"), ENCODINGS, "numeric");
  const algorithm = pick(asString(state, "algorithm", "logistic"), ALGORITHMS, "logistic");
  const loss = pick(asString(state, "loss", "logloss"), LOSSES, "logloss");
  const depth = clamp(Math.round(asNumber(state, "depth", 2)), 0, TREE_MAX_DEPTH);
  const holdoutSource: HoldoutSource = pick(asString(state, "holdoutSource", "same"), HOLDOUT_SOURCES, "same");
  const otherSource: HoldoutSource = holdoutSource === "same" ? "shifted" : "same";
  const apartments = apartmentsFor(holdoutSource);
  const selected = clamp(Math.round(asNumber(state, "selected", 0)), 0, apartments.length - 1);
  const lastKnob = pick(
    asString(state, "lastKnob", "algorithm"),
    ["encoding", "algorithm", "loss", "depth", "holdout"],
    "algorithm",
  );
  const maxStep = algorithm === "tree" ? depth : GD_STEPS;
  const step = clamp(Math.round(asNumber(state, "step", maxStep)), 0, maxStep);

  const names = featureNames(encoding);
  const current = apartments[selected];
  const currentX = encodeApartment(current, encoding);

  const history = useMemo(
    () => runFit(encoding, algorithm, loss, depth, holdoutSource),
    [algorithm, depth, encoding, holdoutSource, loss],
  );
  const otherHistory = useMemo(
    () => runFit(encoding, algorithm, loss, depth, otherSource),
    [algorithm, depth, encoding, loss, otherSource],
  );
  const snapshot = history[Math.min(step, history.length - 1)] ?? history[0];
  const otherSnapshot = otherHistory[Math.min(step, otherHistory.length - 1)] ?? otherHistory[0];
  const sameSnapshot = holdoutSource === "same" ? snapshot : otherSnapshot;
  const shiftedSnapshot = holdoutSource === "same" ? otherSnapshot : snapshot;
  const linear = { weights: snapshot.weights, bias: snapshot.bias };

  const leastSquares = useMemo(() => {
    if (algorithm !== "linear" || loss !== "mse") return null;
    const solved = closedFormLinear(encoding);
    const train = apartments.filter((row) => row.split === "train");
    const hold = apartments.filter((row) => row.split === "holdout");
    const pred = (row: (typeof apartments)[number]) =>
      predictProbability(encodeApartment(row, encoding), "linear", "mse", solved, null);
    return {
      ...solved,
      trainLoss: meanLoss(train.map(pred), train.map((row) => row.price), "mse"),
      holdoutLoss: meanLoss(hold.map(pred), hold.map((row) => row.price), "mse"),
    };
  }, [algorithm, apartments, encoding, loss]);

  const probability = (x: number[]) => predictProbability(x, algorithm, loss, linear, snapshot.tree);
  const usesProbability = algorithm !== "linear" || loss === "logloss";
  const contour = contourThreshold(algorithm, loss);
  const boundary = algorithm === "tree" ? null : planeLine(linear.weights, linear.bias, encoding, current, contour);
  const splits = snapshot.tree ? collectSplits(snapshot.tree) : [];
  const segments = snapshot.tree ? sliceSegments(snapshot.tree, encoding, current) : [];
  const roomsAxis = names.indexOf("rooms");
  const parkAxis = names.indexOf("park");

  const cells = useMemo(() => {
    const tiles = [];
    for (let row = 0; row < GRID; row += 1) {
      for (let column = 0; column < GRID; column += 1) {
        const x = (column + 0.5) / GRID;
        const y = 1 - (row + 0.5) / GRID;
        const rooms = fromPlotX(x);
        const park = fromPlotY(y);
        const features = featuresAt(rooms, park, encoding, current);
        const value = probability(features);
        tiles.push({
          x,
          y,
          value,
          klass: value >= 0.5 ? 1 : 0,
        });
      }
    }
    return tiles;
    // probability / current close over the fitted snapshot and the color slice
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm, current.id, encoding, loss, snapshot]);

  const selectedP = probability(currentX);
  const selectedLoss = pointLoss(selectedP, current.price, loss);
  const selectedZ = linear.bias + linear.weights.reduce((sum, weight, index) => sum + weight * (currentX[index] ?? 0), 0);
  const gap = snapshot.holdoutLoss - snapshot.trainLoss;
  const trainRows = apartments.filter((row) => row.split === "train");
  const holdRows = apartments.filter((row) => row.split === "holdout");

  const hypothesisLine =
    algorithm === "tree"
      ? loss === "logloss"
        ? `axis-aligned tree, entropy splits, depth ${depth}`
        : `axis-aligned tree, Gini splits, depth ${depth}`
      : algorithm === "logistic"
        ? "P(y=1|x) = σ(w·x+b)"
        : loss === "logloss"
          ? "ŷ = w·x+b, log-loss uses σ as the link"
          : "ŷ = w·x+b";

  const stageDetails = [
    `${current.name}  x = [${currentX.map((value) => fmt(value, 2)).join(", ")}]`,
    `${current.name} is ${current.split} · ${trainRows.length} train, ${holdRows.length} holdout`,
    hypothesisLine,
    algorithm === "tree"
      ? `grown depth ${step} · L_train ${fmt(snapshot.trainLoss)}`
      : `step ${step} · ‖∇‖ ${fmt(snapshot.gradNorm)} · L_train ${fmt(snapshot.trainLoss)}`,
    `L_hold ${fmt(snapshot.holdoutLoss)} · train ${fmt(snapshot.trainLoss)} · gap ${fmt(gap)}`,
  ];

  const knobNote =
    lastKnob === "encoding"
      ? "You last changed encoding. Algorithm and loss stayed put. If holdout moved, the representation moved the search."
      : lastKnob === "loss"
        ? "You last changed the loss. Same H, same x, a different number being driven down."
        : lastKnob === "depth"
          ? "You last changed tree depth. Extra capacity can isolate Rowan 9 on train only."
          : lastKnob === "holdout"
            ? "You last changed where the holdout comes from. The fit and the train loss did not move; only the four rows it is scored on did."
            : "You last changed the hypothesis class. Same table, same split, a different family of functions.";

  const sliceNote =
    encoding === "numeric"
      ? "The model sees only rooms and park, so this plane is the whole feature space. Color does not enter x."
      : `Color is taken from ${current.name} (${current.color}). The surface is the slice of the fitted rule at that color. Click another row to change the slice.`;

  const formula =
    algorithm === "tree"
      ? {
          label: `${current.name} leaf probability`,
          expression: `p = ${fmt(selectedP)}  in the leaf that contains this x`,
          result: selectedP >= 0.5 ? "class 1" : "class 0",
          detail:
            current.split === "holdout"
              ? "This row was not used to place a split. The leaf fraction is a train count."
              : loss === "logloss"
                ? "Splits maximize entropy gain on train. p is the train fraction of high-price rows in the leaf."
                : "Splits maximize Gini gain on train. On labels in {0,1}, Gini is twice the leaf variance, so the split matches an MSE tree.",
        }
      : algorithm === "logistic" || loss === "logloss"
        ? {
            label: `${current.name} logistic readout`,
            expression: `P(y=1|x) = σ(${fmt(selectedZ)}) = ${fmt(selectedP)}`,
            result: `ℓ = ${fmt(selectedLoss)}`,
            detail:
              current.split === "holdout"
                ? "Holdout contribution is scored after the fit. It never entered ∇L."
                : "ℓ = −[y ln p + (1−y) ln(1−p)] for this row. Mean log-loss averages that number over the twelve train rows.",
          }
        : {
            label: `${current.name} linear readout`,
            expression: `ŷ = w·x+b = ${fmt(selectedZ)}`,
            result: `ℓ = ${fmt(selectedLoss)}`,
            detail:
              current.split === "holdout"
                ? "Holdout residual is scored after the fit. The normal equation and the gradient used train rows only."
                : "ℓ = (ŷ − y)² for this row. The 0.5 contour is the drawn hyperplane.",
          };

  const setKnob = (patch: Record<string, string | number>) => setState(patch);

  return (
    <div className="gw-shell">
    <div className="tg-lab tg-lab--hero gw-lab sll-lab">
      <LabSurface label="Supervised loop" className="loop-pipeline-card">
        <SurfaceHeading
          kicker="One table, five stages"
          title="Encode, split, choose H, fit, then read the holdout"
          aside={<span className="tg-badge">{algorithm}</span>}
        />
        <StageFlow
          label="Supervised learning loop"
          stages={STAGES.map((item, index) => ({ ...item, detail: stageDetails[index] }))}
          current={stage}
          onSelect={(index) => setKnob({ stage: index })}
          loops
        />
        <div
          className="loop-stage-panel"
          data-loop-train-loss={fmt(snapshot.trainLoss)}
          data-loop-holdout-loss={fmt(snapshot.holdoutLoss)}
          data-loop-bias={fmt(snapshot.bias)}
          data-loop-step={String(step)}
          data-loop-split={current.split}
        >
          {stage === 0 && (
            <>
              <VectorChip label={`${current.name} x`} values={currentX} tone="forward" />
              <p className="lab-note">
                {encoding === "numeric"
                  ? "Rooms + park drops color. The vector is two numbers, and the later plane is not a projection."
                  : encoding === "onehot"
                    ? "One-hot keeps rooms and park and gives each color its own 0/1 slot. A tiny ridge (10⁻⁴) is added only when the normal equation inverts [1 | X]."
                    : "Integer codes use red = 1, blue = 2, green = 3. That order is authored, not measured."}
              </p>
            </>
          )}
          {stage === 1 && (
            <>
              <div className="metric-row">
                <Metric label="This row" value={`${current.name} · ${current.split}`} tone={current.split === "holdout" ? "loss" : "forward"} />
                <Metric label="In the fit?" value={current.split === "train" ? "yes" : "no"} tone={current.split === "train" ? "forward" : "loss"} />
                <Metric label="Train / holdout" value={`${trainRows.length} / ${holdRows.length}`} />
              </div>
              <p className="lab-note">
                Rowan 9 is an authored exception on train (3 rooms, park, labeled low).{" "}
                {holdoutSource === "same"
                  ? "Laurel 2 follows the usual rule on holdout."
                  : "The four holdout rows are from a different process: small flats are the expensive ones."}{" "}
                Neither gradient nor split may read a holdout row.
              </p>
            </>
          )}
          {stage === 2 && (
            <>
              <p className="loop-formula">{hypothesisLine}</p>
              {algorithm !== "tree" && (
                <VectorChip label="w" values={linear.weights} tone="attention" />
              )}
              {algorithm !== "tree" && (
                <div className="metric-row">
                  <Metric label="b" value={fmt(snapshot.bias)} />
                  <Metric label="dim(x)" value={`${names.length}`} />
                </div>
              )}
              {algorithm === "tree" && (
                <p className="lab-note">
                  Each split asks one feature &lt; t. The criterion is {loss === "logloss" ? "entropy" : "Gini"} on
                  the training slice only.
                </p>
              )}
            </>
          )}
          {stage === 3 && (
            <>
              <div className="metric-row">
                <Metric label={algorithm === "tree" ? "Grown depth" : "Step"} value={`${step}`} tone="forward" />
                <Metric label="L train" value={fmt(snapshot.trainLoss)} tone="loss" />
                <Metric label="‖∇‖" value={algorithm === "tree" ? "—" : fmt(snapshot.gradNorm)} />
              </div>
              <p className="lab-note">
                {algorithm === "tree"
                  ? "A tree is not gradient descent. The curve is the real train and holdout loss of the greedy tree at each depth."
                  : `Each step is θ ← θ − η ∇L_train with η = ${stepSize(algorithm, loss)}, where ∇L_train is the exact gradient of the mean train ${loss === "mse" ? "MSE" : "log-loss"}. Holdout loss is computed after the update and never enters ∇L.`}
              </p>
            </>
          )}
          {stage === 4 && (
            <>
              <div className="metric-row">
                <Metric label="L holdout" value={fmt(snapshot.holdoutLoss)} tone="loss" />
                <Metric label="L train" value={fmt(snapshot.trainLoss)} />
                <Metric
                  label="Gap"
                  value={fmt(gap)}
                  tone={gap > 0.08 ? "loss" : "forward"}
                />
              </div>
              <p className="lab-note">
                Gap is L_hold − L_train. A large gap after extra depth is memorization on this table, not a published
                metric. How we know a model works will name accuracy, precision, and leakage.
              </p>
            </>
          )}
        </div>

        <div className="swap-control">
          <div className="encoding-control">
            <SegmentedControl
              label="Encoding"
              value={encoding}
              options={[
                { value: "numeric", label: "rooms + park" },
                { value: "integer", label: "integer codes" },
                { value: "onehot", label: "one-hot" },
              ]}
              onChange={(value) => setKnob({ encoding: value as Encoding, lastKnob: "encoding" })}
            />
          </div>
          <div className="algorithm-control">
            <SegmentedControl
              label="Algorithm"
              value={algorithm}
              options={ALGORITHMS.map((item) => ({ value: item, label: item }))}
              onChange={(value) => {
                const next = value as Algorithm;
                setKnob({
                  algorithm: next,
                  lastKnob: "algorithm",
                  step: next === "tree" ? depth : GD_STEPS,
                });
              }}
            />
          </div>
          <div className="loss-control">
            <SegmentedControl
              label="Loss"
              value={loss}
              options={[
                { value: "logloss", label: "log-loss" },
                { value: "mse", label: "MSE" },
              ]}
              onChange={(value) => setKnob({ loss: value as LossKind, lastKnob: "loss" })}
            />
          </div>
        </div>
        <div className="loop-fit-controls">
          <div className="fit-step-control">
            <RangeControl
              label={algorithm === "tree" ? "Grown depth" : "Fit step"}
              min={0}
              max={maxStep}
              step={1}
              value={step}
              onChange={(value) =>
                setKnob(algorithm === "tree" ? { step: value, depth: value, lastKnob: "depth" } : { step: value })
              }
            />
          </div>
          <div className="capacity-control">
            <RangeControl
              label="Tree depth"
              min={0}
              max={TREE_MAX_DEPTH}
              step={1}
              value={depth}
              onChange={(value) =>
                setKnob({
                  depth: value,
                  lastKnob: "depth",
                  ...(algorithm === "tree" ? { step: value } : {}),
                })
              }
            />
          </div>
        </div>
        <p className="lab-note">{knobNote}</p>
      </LabSurface>

      <LabSurface label="Decision view" className="loop-decision-card">
        <SurfaceHeading
          kicker={
            usesProbability
              ? `P(y=1|x) on rooms × park${encoding === "numeric" ? "" : `, color = ${current.color}`}`
              : `ŷ = w·x+b on rooms × park${encoding === "numeric" ? "" : `, color = ${current.color}`}`
          }
          title="The boundary is the fitted rule, not a drawing"
        />
        <svg
          className="decision-canvas loop-canvas"
          viewBox={`0 0 ${GRID} ${GRID}`}
          role="img"
          aria-label={`${algorithm} decision view at step ${step}. Train loss ${fmt(snapshot.trainLoss)}, holdout loss ${fmt(snapshot.holdoutLoss)}.`}
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
            const y = clamp(1 - (event.clientY - rect.top) / rect.height, 0, 1);
            const rooms = fromPlotX(x);
            const park = fromPlotY(y);
            let best = selected;
            let bestDist = Infinity;
            apartments.forEach((row) => {
              const nudge = displayNudge(row.id);
              const dx = row.rooms + nudge.dx - rooms;
              const dy = row.park + nudge.dy - park;
              const dist = dx * dx + dy * dy;
              if (dist < bestDist) {
                best = row.id;
                bestDist = dist;
              }
            });
            setKnob({ selected: best });
          }}
        >
          {cells.map((cell, index) => (
            <rect
              key={index}
              x={cell.x * GRID - 0.5}
              y={(1 - cell.y) * GRID - 0.5}
              width="1"
              height="1"
              className={`is-class-${cell.klass}`}
              opacity={
                usesProbability
                  ? 0.42 + Math.min(1, Math.max(0, cell.value)) * 0.58
                  : cell.klass
                    ? 0.92
                    : 0.55
              }
            />
          ))}
          {[1, 2, 3, 4].map((rooms) => (
            <line
              key={`vr-${rooms}`}
              className="is-gridline"
              x1={toPlotX(rooms) * GRID}
              x2={toPlotX(rooms) * GRID}
              y1="0"
              y2={GRID}
            />
          ))}
          {[0, 1].map((park) => (
            <line
              key={`hr-${park}`}
              className="is-gridline"
              x1="0"
              x2={GRID}
              y1={(1 - toPlotY(park)) * GRID}
              y2={(1 - toPlotY(park)) * GRID}
            />
          ))}
          {boundary && (
            <line
              className="is-boundary"
              x1={toPlotX(boundary.x1) * GRID}
              y1={(1 - toPlotY(boundary.y1)) * GRID}
              x2={toPlotX(boundary.x2) * GRID}
              y2={(1 - toPlotY(boundary.y2)) * GRID}
            />
          )}
          {segments.map((segment, index) => {
            if (segment.axis === "rooms") {
              const x = toPlotX(segment.threshold) * GRID;
              return (
                <line
                  key={`s-${index}`}
                  className="is-split"
                  x1={x}
                  x2={x}
                  y1={(1 - toPlotY(segment.to)) * GRID}
                  y2={(1 - toPlotY(segment.from)) * GRID}
                />
              );
            }
            const y = (1 - toPlotY(segment.threshold)) * GRID;
            return (
              <line
                key={`s-${index}`}
                className="is-split"
                x1={toPlotX(segment.from) * GRID}
                x2={toPlotX(segment.to) * GRID}
                y1={y}
                y2={y}
              />
            );
          })}
          {apartments.map((row) => {
            const nudge = displayNudge(row.id);
            const cx = toPlotX(row.rooms + nudge.dx) * GRID;
            const cy = (1 - toPlotY(row.park + nudge.dy)) * GRID;
            return (
              <circle
                key={row.id}
                cx={cx}
                cy={cy}
                r={row.id === selected ? 0.42 : 0.32}
                className={`point-label-${row.price}${row.split === "holdout" ? " is-holdout" : ""}${row.id === selected ? " is-selected-point" : ""}`}
              />
            );
          })}
        </svg>
        <div className="loop-axis-caption">
          <span>rooms 1 → 4</span>
          <span>park 0 → 1</span>
        </div>
        <p className="lab-note">
          Filled circles are train. Open circles are holdout. Marks that share (rooms, park) are nudged on the
          plot only; every gradient and split uses the exact vector. {sliceNote}{" "}
          {algorithm === "tree"
            ? encoding === "numeric"
              ? "Opacity is the leaf's train fraction of high-price rows. Dashed segments are the tree's cuts on rooms or park, each spanning only the rectangle its parent owns."
              : "Opacity is the leaf's train fraction of high-price rows. Dashed segments are the rooms or park cuts that apply at this color. A color split does not draw a line here: the slice simply follows the branch its color falls into."
            : algorithm === "logistic" || loss === "logloss"
              ? "Opacity is P(y=1|x). The solid line is w·x+b = 0, the 50% contour."
              : "The solid line is ŷ = 0.5. Opacity is not a probability unless the loss used the sigmoid link."}
        </p>
      </LabSurface>

      <LabSurface label="Loss over the fit" className="loop-loss-card">
        <SurfaceHeading
          kicker={algorithm === "tree" ? "Computed at each grown depth" : "Computed after each gradient step"}
          title={loss === "logloss" ? "Mean log-loss, train versus holdout" : "Mean squared error, train versus holdout"}
        />
        <LineChart
          label={algorithm === "tree" ? "Loss versus tree depth" : "Loss versus gradient step"}
          xLabel={algorithm === "tree" ? "depth" : "step"}
          yLabel={loss === "logloss" ? "mean log-loss" : "MSE"}
          marker={{ x: step, label: algorithm === "tree" ? "depth" : "step" }}
          yDomain={(() => {
            const lows = history.map((item) => Math.min(item.trainLoss, item.holdoutLoss));
            const highs = history.map((item) => Math.max(item.trainLoss, item.holdoutLoss));
            const low = Math.min(...lows);
            const high = Math.max(...highs);
            const pad = Math.max(0.04, (high - low) * 0.12);
            return [Math.max(0, low - pad), high + pad] as [number, number];
          })()}
          xDomain={[0, Math.max(1, history.length - 1)]}
          series={[
            {
              id: "train",
              name: "train, final",
              tone: "loss",
              dash: "solid",
              points: history.map((item) => ({ x: item.step, y: item.trainLoss })),
            },
            {
              id: "holdout",
              name: "holdout, final",
              tone: "forward",
              dash: "dashed",
              points: history.map((item) => ({ x: item.step, y: item.holdoutLoss })),
            },
            ...(leastSquares
              ? [
                  {
                    id: "ols",
                    name: "least squares (train)",
                    tone: "attention" as const,
                    dash: "dotted" as const,
                    points: [
                      { x: 0, y: leastSquares.trainLoss },
                      { x: GD_STEPS, y: leastSquares.trainLoss },
                    ],
                  },
                ]
              : []),
          ]}
          footnote={
            algorithm === "tree"
              ? snapshot.holdoutLoss > 2 && loss === "logloss"
                ? "Entropy tree on train only. A leaf that is sure and wrong costs about −ln(10⁻⁹) on that holdout row. The clamp is numerical, not a second loss."
                : `${loss === "logloss" ? "Entropy" : "Gini"} tree on the twelve train rows only. Holdout is scored after each grown depth.`
              : leastSquares
                ? `Dotted line is the closed-form least squares train loss after a ridge of 10⁻⁴ on X̃ᵀX̃. GD starts at w = 0, b = 0.`
                : "Holdout is scored after each train-only update. Nothing in ∇L reads a holdout row."
          }
        />
        <div className="metric-row">
          <Metric label="L train" value={fmt(snapshot.trainLoss)} tone="loss" />
          <Metric label="L holdout" value={fmt(snapshot.holdoutLoss)} tone="forward" />
          <Metric
            label="Train acc"
            value={`${(snapshot.trainAccuracy * 100).toFixed(0)}%`}
          />
          <Metric
            label="Holdout acc"
            value={`${(snapshot.holdoutAccuracy * 100).toFixed(0)}%`}
          />
        </div>
      </LabSurface>

      <LabSurface label="Train versus holdout" className="split-table-card">
        <SurfaceHeading
          kicker="Authored 16-row apartment table"
          title="Only the filled split enters X and y"
        />
        <div className="split-strip" aria-hidden="true">
          {apartments.map((row) => (
            <i
              key={row.id}
              className={`${row.split === "train" ? "is-train" : "is-test"}${row.price ? " is-positive" : ""}`}
              title={row.name}
            />
          ))}
        </div>
        <div className="split-legend">
          <span>Train · 12</span>
          <span>Holdout · 4</span>
          <span>Outline · high price</span>
        </div>
        <div className="holdout-source-control">
          <SegmentedControl
            label="Holdout drawn from"
            value={holdoutSource}
            options={[
              { value: "same", label: "the same process" },
              { value: "shifted", label: "a different process" },
            ]}
            onChange={(value) => {
              setKnob({ holdoutSource: value as HoldoutSource, lastKnob: "holdout" });
              narrate(
                value === "same"
                  ? "Holdout drawn from the same process as the train rows."
                  : "Holdout drawn from a different process: small flats are the expensive ones.",
              );
            }}
          />
        </div>
        <div className="source-table split-table">
          <table className="tg-board">
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Split</th>
                <th scope="col">Rooms</th>
                <th scope="col">Color</th>
                <th scope="col">Park</th>
                <th scope="col">y</th>
                <th scope="col">p or ŷ</th>
              </tr>
            </thead>
            <tbody>
              {apartments.map((row) => {
                const p = probability(encodeApartment(row, encoding));
                return (
                  <tr
                    key={row.id}
                    className={`${row.id === selected ? "is-leader" : ""}${row.split === "holdout" ? " is-holdout-row" : ""}`}
                  >
                    <th scope="row">
                      <button type="button" className="quiet-action" onClick={() => setKnob({ selected: row.id })}>
                        {row.name}
                      </button>
                    </th>
                    <td>{row.split}</td>
                    <td>{row.rooms}</td>
                    <td>{row.color}</td>
                    <td>{row.park ? "yes" : "no"}</td>
                    <td>{row.price}</td>
                    <td>{fmt(p)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <table className="tg-board fit-table sll-shift-table" aria-label="The same fit scored on two different holdouts">
          <thead>
            <tr>
              <th scope="col">Holdout from</th>
              <th scope="col">L holdout</th>
              <th scope="col">Holdout acc</th>
              <th scope="col">Rows right</th>
            </tr>
          </thead>
          <tbody>
            {[
              { id: "same", label: "the same process", snap: sameSnapshot },
              { id: "shifted", label: "a different process", snap: shiftedSnapshot },
            ].map((item) => (
              <tr key={item.id} className={item.id === holdoutSource ? "is-leader" : ""}>
                <th scope="row">{item.label}</th>
                <td>{fmt(item.snap.holdoutLoss)}</td>
                <td>{(item.snap.holdoutAccuracy * 100).toFixed(0)}%</td>
                <td>{Math.round(item.snap.holdoutAccuracy * 4)} of 4</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric label="L train (same in both)" value={fmt(snapshot.trainLoss)} />
          <Metric
            label="Holdout loss, different ÷ same"
            value={sameSnapshot.holdoutLoss > 1e-9 ? `${(shiftedSnapshot.holdoutLoss / sameSnapshot.holdoutLoss).toFixed(1)}×` : "n/a"}
            tone={shiftedSnapshot.holdoutLoss > sameSnapshot.holdoutLoss ? "loss" : undefined}
          />
          <Metric
            label="Holdout accuracy change"
            value={`${((shiftedSnapshot.holdoutAccuracy - sameSnapshot.holdoutAccuracy) * 100).toFixed(0)} pts`}
            tone={shiftedSnapshot.holdoutAccuracy < sameSnapshot.holdoutAccuracy ? "loss" : undefined}
          />
        </div>
        <p className="lab-note">
          {holdoutSource === "same"
            ? "The first twelve names are the original six apartments plus six new train rows, including Rowan 9, the authored exception. The last four are holdout and follow the usual rooms-and-park rule. Color looks predictive on train (four of five reds are high) and is not on holdout."
            : "The last four rows come from a different process: a market where small flats are the expensive ones (high price when rooms ≤ 2). The inputs look like the train rows, but the rule that produced the labels changed."}{" "}
          The fit and its train loss are identical in both rows of the table; only the four rows it is scored on differ.
          That is distribution shift: a holdout estimates performance on rows from the process that made the train rows, and
          says nothing about a process that has moved.
        </p>
      </LabSurface>

      <LabSurface label="Worked example" className="loop-example-card">
        <SurfaceHeading
          kicker={`${current.name} · y = ${current.price} · ${current.split}`}
          title="One row, the same arithmetic as the curve"
        />
        {algorithm !== "tree" && (
          <div className="metric-row">
            <Metric label="b" value={fmt(linear.bias)} />
            {names.map((name, index) => (
              <Metric key={name} label={`w_${name}`} value={fmt(linear.weights[index] ?? 0)} />
            ))}
          </div>
        )}
        <VectorChip label="x" values={currentX} tone="forward" />
        <FormulaWithValues
          label={formula.label}
          expression={formula.expression}
          result={formula.result}
          detail={formula.detail}
          tone={current.split === "holdout" ? "loss" : "attention"}
        />
        {leastSquares && (
          <p className="lab-note">
            Closed form on train: b = {fmt(leastSquares.bias)}, w = [{leastSquares.weights.map((value) => fmt(value)).join(", ")}].
            That is (X̃ᵀX̃ + λI)⁻¹ X̃ᵀy with λ = 10⁻⁴. The scrubbed weights above are the GD path, not that solve,
            until the curve meets the dotted line.
          </p>
        )}
        {algorithm === "tree" && encoding !== "numeric" && splits.some((split) => split.axis !== roomsAxis && split.axis !== parkAxis) && (
          <p className="lab-note">
            At least one split is on color, so it does not appear as a rooms or park tick. The leaf probability still
            uses that cut.
          </p>
        )}
      </LabSurface>
    </div>
    </div>
  );
}
