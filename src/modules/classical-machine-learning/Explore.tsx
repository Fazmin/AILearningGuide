import { useMemo } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  ALGORITHMS,
  FOREST_SIZE,
  POINTS,
  countLeaves,
  forestVotes,
  knnRanked,
  lineInSquare,
  naiveFactors,
  marginWidthOf,
  score,
  scoreSplit,
  storedCount,
  supportVectors,
  treePredict,
  treeSegments,
  type Algorithm,
} from "./fit";
import { buildModel, classify, probabilityOf, treePath } from "./model";
import { Ballots, MethodGallery, ScoreLine, ShareBar, TreeWalk, type GalleryItem } from "./pictures";

const GRID = 28;
const THUMB = 14;
const TICKS = [0, 0.5, 1];

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

const sx = (x: number) => x * GRID;
const sy = (y: number) => (1 - y) * GRID;

export default function Explore({ state, setState }: ModuleContext) {
  const algorithm: Algorithm = (ALGORITHMS as readonly string[]).includes(asString(state, "algorithm", "linear"))
    ? (asString(state, "algorithm", "linear") as Algorithm)
    : "linear";
  const k = clamp(Math.round(asNumber(state, "k", 3)), 1, 7);
  const depth = clamp(Math.round(asNumber(state, "depth", 2)), 1, 5);
  const probeX = clamp(asNumber(state, "probeX", 0.5), 0, 1);
  const probeY = clamp(asNumber(state, "probeY", 0.5), 0, 1);

  const model = useMemo(() => buildModel(algorithm, k, depth), [algorithm, depth, k]);

  const probabilityAt = (x: number, y: number) => probabilityOf(model, x, y);

  const predict = (x: number, y: number): number => classify(model, x, y);

  /** Every method fitted at the current k and depth, sampled coarsely for the side-by-side strip. */
  const gallery = useMemo<GalleryItem[]>(
    () =>
      ALGORITHMS.map((name) => {
        const fitted = buildModel(name, k, depth);
        const tiles: number[] = [];
        for (let row = 0; row < THUMB; row += 1) {
          for (let column = 0; column < THUMB; column += 1) {
            tiles.push(classify(fitted, (column + 0.5) / THUMB, 1 - (row + 0.5) / THUMB));
          }
        }
        const misses = POINTS.map((point) => name !== "kmeans" && classify(fitted, point.x, point.y) !== point.label);
        return {
          id: name,
          cells: tiles,
          size: THUMB,
          hits: name === "kmeans" ? null : misses.filter((miss) => !miss).length,
          misses,
        };
      }),
    [depth, k],
  );

  const cells = useMemo(() => {
    const tiles: { column: number; row: number; value: number; probability: number | null }[] = [];
    for (let row = 0; row < GRID; row += 1) {
      for (let column = 0; column < GRID; column += 1) {
        const x = (column + 0.5) / GRID;
        const y = 1 - (row + 0.5) / GRID;
        tiles.push({ column, row, value: predict(x, y), probability: probabilityAt(x, y) });
      }
    }
    return tiles;
    // predict / probabilityAt close over model
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  /** Cell edges where the predicted class (or cluster id) changes: the drawn boundary for every method. */
  const regionEdges = useMemo(() => {
    const at = (column: number, row: number) => cells[row * GRID + column]?.value;
    const edges: string[] = [];
    for (let row = 0; row < GRID; row += 1) {
      for (let column = 0; column < GRID; column += 1) {
        if (column + 1 < GRID && at(column, row) !== at(column + 1, row)) {
          edges.push(`M${column + 1} ${row}V${row + 1}`);
        }
        if (row + 1 < GRID && at(column, row) !== at(column, row + 1)) {
          edges.push(`M${column} ${row + 1}H${column + 1}`);
        }
      }
    }
    return edges.join("");
  }, [cells]);

  const predictions = POINTS.map((point) => predict(point.x, point.y));
  const hits = algorithm === "kmeans" ? null : POINTS.filter((point, index) => predictions[index] === point.label).length;
  const classOneRegion = cells.filter((cell) => cell.value === 1).length / cells.length;

  const probeNeighbors = knnRanked(probeX, probeY).slice(0, k);
  const probeClass = predict(probeX, probeY);
  const probeProbability = probabilityAt(probeX, probeY);
  const linearScore =
    model.kind === "linear" || model.kind === "logistic" || model.kind === "svm"
      ? score(model.weights, probeX, probeY)
      : null;
  const votes = model.kind === "forest" ? forestVotes(model.trees, probeX, probeY) : null;

  const hyperplane =
    model.kind === "linear"
      ? lineInSquare(model.weights, 0.5)
      : model.kind === "logistic" || model.kind === "svm"
        ? lineInSquare(model.weights, 0)
        : null;
  const marginMinus = model.kind === "svm" ? lineInSquare(model.weights, -1) : null;
  const marginPlus = model.kind === "svm" ? lineInSquare(model.weights, 1) : null;
  const support = model.kind === "svm" ? supportVectors(model.weights, POINTS) : [];
  const marginWidth = model.kind === "svm" ? marginWidthOf(model.weights) : null;
  const rootScore =
    model.kind === "tree" && model.tree.kind === "split"
      ? { axis: model.tree.axis, threshold: model.tree.threshold, ...scoreSplit(POINTS, model.tree.axis, model.tree.threshold) }
      : null;
  const factors = model.kind === "naivebayes" ? naiveFactors(model.classes, probeX, probeY) : null;
  const segments =
    model.kind === "tree" ? treeSegments(model.tree) : model.kind === "forest" ? treeSegments(model.trees[0]) : [];
  const leaves =
    model.kind === "tree" ? countLeaves(model.tree) : model.kind === "forest" ? countLeaves(model.trees[0]) : null;
  const stored = storedCount(algorithm, k, depth);
  const storedAll = useMemo(() => ALGORITHMS.map((name) => ({ name, count: storedCount(name, k, depth) })), [depth, k]);
  const usesK = algorithm === "knn" || algorithm === "kmeans";
  const usesDepth = algorithm === "tree" || algorithm === "forest";
  const walk = model.kind === "tree" ? treePath(model.tree, probeX, probeY) : null;
  const treeVotes = model.kind === "forest" ? model.trees.map((tree) => treePredict(tree, probeX, probeY)) : [];
  const centerDistances =
    model.kind === "kmeans"
      ? model.centers.map((center, index) => ({ index, distance: Math.hypot(center.x - probeX, center.y - probeY) }))
      : [];

  const keeps =
    algorithm === "knn"
      ? `${POINTS.length} points`
      : algorithm === "kmeans"
        ? `${k} centers`
        : algorithm === "tree"
          ? `${leaves} leaves`
          : algorithm === "forest"
            ? `${FOREST_SIZE} trees`
            : algorithm === "naivebayes"
              ? "2 Gaussians"
              : "3 weights";

  const bias = (() => {
    if (algorithm === "linear") {
      return {
        title: "Inductive bias: a hyperplane",
        body: "Least squares on labels in {0,1} gives ŷ = w₀ + w₁x + w₂y. The cut ŷ = 0.5 is a straight line. It takes the four-point top-left corner and has to give up the two-point bottom-right corner: no line holds both.",
      };
    }
    if (algorithm === "logistic") {
      return {
        title: "Inductive bias: a hyperplane, read as a probability",
        body: "P(y=1|x) = σ(w·x+b). The sigmoid maps a score into (0, 1); it does not bend the boundary. The 50% contour is the line w·x+b = 0, and it misses the same bottom-right pair as least squares.",
      };
    }
    if (algorithm === "svm") {
      return {
        title: "Inductive bias: a wide street around a hyperplane",
        body: "Soft-margin SVM minimizes ½‖w‖² plus C times the hinge violations. The dashed lines are w·x+b = ±1, a street 2/‖w‖ wide. The set is not linearly separable, so points sit inside the street or on the wrong side, and each one is a support vector.",
      };
    }
    if (algorithm === "knn") {
      return {
        title: "Inductive bias: nearby points share a label",
        body: "The model is the dataset. k sets how local the vote is. With k ≤ 4 the two-point bottom-right group wins its own corner. From k = 5 up it is always outvoted, so a larger k smooths a small group away.",
      };
    }
    if (algorithm === "tree") {
      return {
        title: "Inductive bias: axis-aligned rectangles",
        body: "Each split asks x < t or y < t, and a deeper split only cuts the rectangle its parent made. Depth 1 is one cut (high bias). Depth 5 grows 7 leaves and classifies all 18 points, isolating single points (low bias, high variance on a tiny set).",
      };
    }
    if (algorithm === "forest") {
      return {
        title: "Inductive bias: average of axis-aligned trees",
        body: `Each of ${FOREST_SIZE} trees is grown on a bootstrap sample of the 18 points. The vote is an unweighted majority. Averaging usually lowers variance relative to one deep tree; it does not remove the axis-aligned bias.`,
      };
    }
    if (algorithm === "naivebayes") {
      return {
        title: "Inductive bias: coordinates independent given the class",
        body: "P(x,y|c) = P(x|c)P(y|c), each a 1-D Gaussian. Class 1 lives in two opposite corners, so its fitted Gaussian is one wide blob near the middle; the independence assumption cannot represent the anti-diagonal. The posterior crosses 0.5 only in the far top-left, along a curve, because the two classes have different variances.",
      };
    }
    return {
      title: "Inductive bias: Voronoi cells around k centers",
      body: "k-means never reads a label. Region color is a cluster id. Matching a label mark is an accident. k is a choice you bring, and Lloyd's algorithm only finds a local optimum from its starting centers.",
    };
  })();

  const formula =
    algorithm === "logistic" && linearScore !== null && probeProbability !== null
      ? {
          label: "Logistic at the probe",
          expression: `P(y=1|x) = σ(${linearScore.toFixed(3)})`,
          result: probeProbability.toFixed(3),
          detail: "σ(z) = 1/(1+e^{−z}). The cell shade uses this number; the hard class is 1[p ≥ 0.5].",
        }
      : algorithm === "svm" && linearScore !== null
        ? {
            label: "Signed margin score",
            expression: `w·x + b = ${linearScore.toFixed(3)}`,
            result: linearScore >= 0 ? "class 1" : "class 0",
            detail: `Decision at 0. The street is |w·x+b| < 1, width 2/‖w‖ = ${marginWidth?.toFixed(3)}. Support vectors are points with t(w·x+b) ≤ 1: ${support.length} of ${POINTS.length} here.`,
          }
        : algorithm === "naivebayes" && probeProbability !== null
          ? {
              label: "Naive Bayes posterior",
              expression: "P(c=1|x,y) ∝ P(c=1) N(x|c=1) N(y|c=1)",
              result: probeProbability.toFixed(3),
              detail: factors
                ? `The product of two 1-D Gaussians is the conditional-independence assumption, named on purpose. At the probe, class 0 scores ${factors[0].prior.toFixed(3)} × ${factors[0].densityX.toFixed(3)} × ${factors[0].densityY.toFixed(3)} = ${factors[0].score.toFixed(3)} and class 1 scores ${factors[1].prior.toFixed(3)} × ${factors[1].densityX.toFixed(3)} × ${factors[1].densityY.toFixed(3)} = ${factors[1].score.toFixed(3)}; the posterior is the class 1 share of the total.`
                : "The product of two 1-D Gaussians is the conditional-independence assumption, named on purpose.",
            }
          : algorithm === "knn"
            ? {
                label: "Local majority",
                expression: `${probeNeighbors.reduce((sum, point) => sum + point.label, 0)} of ${k} nearest votes are class 1`,
                result: `class ${probeClass}`,
                detail: "An even split goes to class 1. Distance is ordinary Euclidean distance on these two features.",
              }
            : algorithm === "forest" && votes !== null
              ? {
                  label: "Forest vote",
                  expression: `${votes} of ${FOREST_SIZE} trees predict class 1`,
                  result: `class ${probeClass}`,
                  detail: "Each tree saw a bootstrap sample. The drawn splits are from tree 0 only, so you can see the axis-aligned pieces.",
                }
              : algorithm === "linear" && linearScore !== null
                ? {
                    label: "Linear score",
                    expression: `w₀ + w₁x + w₂y = ${linearScore.toFixed(3)}`,
                    result: linearScore >= 0.5 ? "class 1" : "class 0",
                    detail: "Least squares on {0,1} labels. The drawn line is the level set ŷ = 0.5.",
                  }
                : algorithm === "tree" && rootScore
                  ? {
                      label: "Leaf at the probe",
                      expression: `query (${probeX.toFixed(2)}, ${probeY.toFixed(2)})`,
                      result: `class ${probeClass}`,
                      detail: `The probe walks the split questions from the root to one leaf and takes that leaf's majority class. The root cut ${rootScore.axis} < ${rootScore.threshold.toFixed(2)} sends ${rootScore.leftCount} points left and ${rootScore.rightCount} right. Gini falls from ${rootScore.parent.toFixed(3)} to a size-weighted ${rootScore.weighted.toFixed(3)}, a gain of ${rootScore.gain.toFixed(3)}, the largest of any single cut.`,
                    }
                : {
                    label: algorithm === "kmeans" ? "Nearest center" : "Leaf at the probe",
                    expression: `query (${probeX.toFixed(2)}, ${probeY.toFixed(2)})`,
                    result: algorithm === "kmeans" ? `cluster ${probeClass}` : `class ${probeClass}`,
                    detail:
                      algorithm === "kmeans"
                        ? "The probe joins whichever center is closest. No label is involved."
                        : "The probe walks the split questions from the root to one leaf and takes that leaf's majority class.",
                  };

  const surfaceLabel = `${algorithm} decision surface on the 18 points. ${
    hits === null
      ? `${k} clusters; labels unused.`
      : `${hits} of ${POINTS.length} training points classified correctly; class 1 covers ${Math.round(classOneRegion * 100)} percent of the square.`
  }`;

  return (
    <div className="gw-shell">
    <div className="tg-lab tg-lab--hero gw-lab cml-lab">
      <LabSurface label="Algorithm" className="algorithm-card">
        <SurfaceHeading
          kicker="One dataset, eight methods"
          title="Switch the method; keep the points"
          aside={<span className="tg-badge">{algorithm}</span>}
        />
        <div className="algorithm-control">
          <SegmentedControl
            label="Algorithm"
            value={algorithm}
            options={ALGORITHMS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => setState({ algorithm: value })}
          />
        </div>
        <MethodGallery items={gallery} current={algorithm} onSelect={(value) => setState({ algorithm: value })} />
        <div className="algorithm-params cml-params">
          <div className={`cml-param${usesK ? "" : " is-idle"}`}>
            <RangeControl label="k" min={1} max={7} step={1} value={k} onChange={(value) => setState({ k: value })} />
            <small>{usesK ? `used by ${algorithm} now` : "used by knn and kmeans"}</small>
          </div>
          <div className={`cml-param${usesDepth ? "" : " is-idle"}`}>
            <RangeControl
              label="Tree depth"
              min={1}
              max={5}
              step={1}
              value={depth}
              onChange={(value) => setState({ depth: value })}
            />
            <small>{usesDepth ? `used by ${algorithm} now` : "used by tree and forest"}</small>
          </div>
        </div>
        <p className="lab-note">
          k is used by k-nearest neighbors and k-means. Tree depth is used by the decision tree and the forest.
          Linear, logistic, SVM, and naive Bayes ignore both sliders.
        </p>
      </LabSurface>

      <LabSurface label="Decision surface" className="decision-surface-card">
        <SurfaceHeading
          kicker={
            algorithm === "kmeans"
              ? "Clusters, no labels used"
              : algorithm === "logistic" || algorithm === "naivebayes"
                ? "P(y=1|x) on a 28×28 grid"
                : "Predicted class on a 28×28 grid"
          }
          title="The same 18 points, a new boundary"
        />
        <div className="cml-surface">
        <svg
          className="decision-canvas cml-canvas"
          viewBox="-3.4 -1.4 33.4 33.4"
          role="img"
          aria-label={surfaceLabel}
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            const unitsX = ((event.clientX - rect.left) / rect.width) * 33.4 - 3.4;
            const unitsY = ((event.clientY - rect.top) / rect.height) * 33.4 - 1.4;
            setState({ probeX: clamp(unitsX / GRID, 0, 1), probeY: clamp(1 - unitsY / GRID, 0, 1) });
          }}
        >
          {cells.map((cell) => (
            <rect
              key={`${cell.column}-${cell.row}`}
              x={cell.column}
              y={cell.row}
              width="1.02"
              height="1.02"
              className={algorithm === "kmeans" ? `cml-cluster-${cell.value % 4}` : `is-class-${cell.value}`}
              opacity={cell.probability === null ? 1 : 0.28 + Math.abs(cell.probability - 0.5) * 1.44}
            />
          ))}
          {!hyperplane && <path className="cml-region-edge" d={regionEdges} />}
          <rect className="cml-frame" x="0" y="0" width={GRID} height={GRID} />
          {TICKS.map((tick) => (
            <g key={`tick-${tick}`} className="cml-tick">
              <text x={sx(tick)} y={GRID + 1.9} textAnchor="middle">
                {tick}
              </text>
              <text x={-0.7} y={sy(tick) + 0.45} textAnchor="end">
                {tick}
              </text>
            </g>
          ))}
          <text className="cml-axis" x={GRID + 0.9} y={GRID + 1.9}>
            x
          </text>
          <text className="cml-axis" x={-0.7} y={-0.5} textAnchor="end">
            y
          </text>
          {hyperplane && (
            <line
              className="is-boundary"
              x1={sx(hyperplane.x1)}
              y1={sy(hyperplane.y1)}
              x2={sx(hyperplane.x2)}
              y2={sy(hyperplane.y2)}
            />
          )}
          {[marginMinus, marginPlus].map((margin, index) =>
            margin ? (
              <line
                key={`margin-${index}`}
                className="is-margin"
                x1={sx(margin.x1)}
                y1={sy(margin.y1)}
                x2={sx(margin.x2)}
                y2={sy(margin.y2)}
              />
            ) : null,
          )}
          {segments.map((segment, index) =>
            segment.axis === "x" ? (
              <line
                key={`split-${index}`}
                className="cml-split"
                x1={sx(segment.threshold)}
                x2={sx(segment.threshold)}
                y1={sy(segment.from)}
                y2={sy(segment.to)}
              />
            ) : (
              <line
                key={`split-${index}`}
                className="cml-split"
                x1={sx(segment.from)}
                x2={sx(segment.to)}
                y1={sy(segment.threshold)}
                y2={sy(segment.threshold)}
              />
            ),
          )}
          {model.kind === "kmeans" &&
            model.centers.map((center, index) => (
              <g key={`c-${index}`} className="is-center">
                <line x1={sx(center.x) - 0.55} x2={sx(center.x) + 0.55} y1={sy(center.y)} y2={sy(center.y)} />
                <line x1={sx(center.x)} x2={sx(center.x)} y1={sy(center.y) - 0.55} y2={sy(center.y) + 0.55} />
                <text className="cml-center-label" x={sx(center.x) + 0.8} y={sy(center.y) - 0.6}>
                  {index}
                </text>
              </g>
            ))}
          {algorithm === "knn" &&
            probeNeighbors.map((point) => (
              <line
                key={`n-${point.index}`}
                className="is-neighbor"
                x1={sx(probeX)}
                y1={sy(probeY)}
                x2={sx(point.x)}
                y2={sy(point.y)}
              />
            ))}
          {POINTS.map((point, index) => {
            const isSupport = support.includes(point);
            const wrong = hits !== null && predictions[index] !== point.label;
            return (
              <g key={index}>
                <circle
                  cx={sx(point.x)}
                  cy={sy(point.y)}
                  r={isSupport ? 0.46 : 0.36}
                  className={`point-label-${point.label}${isSupport ? " is-support" : ""}${
                    algorithm === "knn" && probeNeighbors.some((item) => item.index === index) ? " is-neighbor-point" : ""
                  }`}
                />
                {wrong && (
                  <path
                    className="cml-miss"
                    d={`M${sx(point.x) + 0.45} ${sy(point.y) - 1.25}l0.8 0.8m0 -0.8l-0.8 0.8`}
                  />
                )}
              </g>
            );
          })}
          <circle className="is-probe" cx={sx(probeX)} cy={sy(probeY)} r="0.48" />
          <circle className="cml-probe-dot" cx={sx(probeX)} cy={sy(probeY)} r="0.12" />
        </svg>
        <div className="cml-surface__key">
        <ul className="cml-legend" aria-label="Legend">
          <li>
            <i className="cml-swatch is-one" />
            {algorithm === "kmeans" ? "cluster id shading" : "predicted class 1"}
          </li>
          {algorithm !== "kmeans" && (
            <li>
              <i className="cml-swatch is-zero" />
              predicted class 0
            </li>
          )}
          <li>
            <i className="cml-dot is-one" />
            point, label 1
          </li>
          <li>
            <i className="cml-dot is-zero" />
            point, label 0
          </li>
          {hits !== null && (
            <li>
              <b className="cml-x">×</b>
              misclassified
            </li>
          )}
          <li>
            <i className="cml-ring" />
            probe
          </li>
        </ul>
        <p className="lab-note">
          Shaded cells are the model evaluated at each cell center; the dark edge is where its prediction flips.
          {algorithm === "logistic" || algorithm === "naivebayes"
            ? " Cells fade toward the 50% line: opacity grows with |P(y=1|x) − 0.5|."
            : ""}
          {algorithm === "svm" ? " Dashed lines are the margin w·x+b = ±1; ringed points are support vectors." : ""}
          {algorithm === "tree" || algorithm === "forest"
            ? " Green dashes are split segments, each bounded by the region its parent owns."
            : ""}{" "}
          Circles keep their true label even under k-means. Click the canvas to move the probe.
        </p>
        </div>
        </div>
      </LabSurface>

      <LabSurface label="Probe and inductive bias" className="probe-card">
        <SurfaceHeading kicker={bias.title} title="A query point, and what the method assumes" />
        <div className="probe-control">
          <RangeControl
            label="Probe x"
            min={0}
            max={1}
            step={0.01}
            value={probeX}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ probeX: value })}
          />
          <RangeControl
            label="Probe y"
            min={0}
            max={1}
            step={0.01}
            value={probeY}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ probeY: value })}
          />
        </div>
        <div className="cml-probe-picture">
          {model.kind === "linear" && linearScore !== null && (
            <ScoreLine
              value={linearScore}
              min={-0.5}
              max={1.5}
              cut={0.5}
              cutLabel="ŷ = 0.5"
              valueLabel={`ŷ = ${linearScore.toFixed(3)}`}
              axisLabel="linear score ŷ"
              ticks={[-0.5, 0, 0.5, 1, 1.5]}
            />
          )}
          {model.kind === "logistic" && probeProbability !== null && (
            <ScoreLine
              value={probeProbability}
              min={0}
              max={1}
              cut={0.5}
              cutLabel="P = 0.5"
              valueLabel={`P(y=1|x) = ${probeProbability.toFixed(3)}`}
              axisLabel="probability of class 1"
              ticks={[0, 0.25, 0.5, 0.75, 1]}
            />
          )}
          {model.kind === "svm" && linearScore !== null && (
            <ScoreLine
              value={linearScore}
              min={-3}
              max={3}
              cut={0}
              cutLabel="w·x + b = 0"
              valueLabel={`w·x + b = ${linearScore.toFixed(3)}`}
              axisLabel="signed margin score"
              band={[-1, 1]}
              bandLabel="the street, |w·x + b| < 1"
              ticks={[-3, -2, -1, 0, 1, 2, 3]}
            />
          )}
          {model.kind === "knn" && (
            <Ballots
              title={`The ${k} nearest points, nearest first`}
              votes={probeNeighbors.map((point) => ({ label: point.label, caption: `d ${point.distance.toFixed(2)}` }))}
              result={probeClass}
              rule="An even split goes to class 1."
            />
          )}
          {model.kind === "forest" && (
            <Ballots
              title={`The ${FOREST_SIZE} trees, one ballot each`}
              votes={treeVotes.map((vote, index) => ({ label: vote, caption: `tree ${index}` }))}
              result={probeClass}
              rule="Unweighted majority."
            />
          )}
          {walk && walk.leaf.kind === "leaf" && (
            <TreeWalk
              steps={walk.steps}
              leafLabel={walk.leaf.label}
              leafCount={walk.leaf.count}
              leafPositives={walk.leaf.positives}
            />
          )}
          {model.kind === "naivebayes" && factors && (
            <ShareBar
              title="Each class's score at the probe, as shares of their total"
              parts={[
                { label: "class 1", value: factors[1].score, tone: "one" },
                { label: "class 0", value: factors[0].score, tone: "zero" },
              ]}
            />
          )}
          {model.kind === "kmeans" && (
            <BarList
              label="Distance from the probe to each center"
              items={centerDistances.map((item) => ({
                id: `c${item.index}`,
                label: `center ${item.index}`,
                value: item.distance,
                display: item.distance.toFixed(3),
                detail: item.index === probeClass ? "nearest, so the probe joins it" : undefined,
                tone: item.index === probeClass ? "forward" : "muted",
                emphasis: item.index === probeClass,
              }))}
              max={1.5}
            />
          )}
        </div>
        <FormulaWithValues
          label={formula.label}
          expression={formula.expression}
          result={formula.result}
          detail={formula.detail}
          tone={algorithm === "logistic" || algorithm === "naivebayes" ? "attention" : "forward"}
        />
        <p className="lab-note">{bias.body}</p>
      </LabSurface>

      <LabSurface label="What it stored" className="stored-model-card">
        <SurfaceHeading kicker="Parameters versus examples" title="What this method keeps after 'fitting'" />
        <div className="metric-row">
          <Metric label="Keeps" value={keeps} tone="forward" />
          <Metric label="Numbers stored" value={`${stored}`} />
          <Metric
            label={algorithm === "kmeans" ? "Within-cluster SS" : "Train accuracy"}
            value={
              model.kind === "kmeans"
                ? model.inertia.toFixed(3)
                : `${hits} / ${POINTS.length} · ${Math.round(((hits ?? 0) / POINTS.length) * 100)}%`
            }
          />
        </div>
        <p className="cml-stored-title" aria-hidden="true">
          Numbers each method stores at k = {k} and depth {depth}, on one scale
        </p>
        <BarList
          label="Numbers stored by each of the eight methods"
          items={storedAll.map((item) => ({
            id: item.name,
            label: item.name,
            value: item.count,
            display: `${item.count}`,
            tone: item.name === algorithm ? "forward" : "muted",
            emphasis: item.name === algorithm,
          }))}
          selectedId={algorithm}
        />
        <p className="lab-note">
          {algorithm === "knn"
            ? "k-nearest neighbors keeps all 18 points (x, y, label) and votes at query time. There is almost no fitting step, so memory and query cost grow with the dataset."
            : algorithm === "kmeans"
              ? "k-means keeps k centers (two coordinates each). It never reads a label, so accuracy is not defined. Within-cluster SS is the sum of squared distances to the nearest center — the number Lloyd's algorithm lowers. It can even rise when k grows, because each run only finds a local optimum from its start."
              : algorithm === "tree"
                ? `A tree keeps each split (a feature and a threshold) and each leaf's class. At depth ${depth} it has ${leaves} leaves. Classifying every training point is not a test score: the last cuts isolate one or two points.`
                : algorithm === "forest"
                  ? "A forest keeps seven trees and averages their votes. That usually lowers variance versus one deep tree. Training accuracy on 18 points is still not a test score."
                  : algorithm === "svm"
                    ? "The SVM keeps three weights. With a linear kernel the support vectors are not needed at prediction time; they are the points that decided where the weights settled."
                    : algorithm === "naivebayes"
                      ? "Naive Bayes keeps a class prior and a mean and variance per coordinate per class: nine numbers. The independence assumption is the entire inductive bias."
                      : algorithm === "logistic"
                        ? "Logistic regression keeps three weights. The shading is σ(w·x+b), a probability, but the 50% contour stays a straight line in these two features."
                        : "Linear least squares compresses the 18 points into three weights. The boundary stays a straight line in these two features."}
        </p>
      </LabSurface>
    </div>
    </div>
  );
}
