import { countMatrix, roc, sweep, type Baseline, type Example, type Population } from "./metrics";

const scale = (d0: number, d1: number, r0: number, r1: number) => (value: number) =>
  r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);

/**
 * All 100 detector scores on one axis, stacked into a dot plot. The shaded side of the threshold is what
 * the rule calls positive. Rows outside the scored split are faded, not removed. Positives are diamonds and
 * negatives are circles, so the two classes differ by shape and not only by colour.
 */
export function ScoreStrip({
  examples,
  population,
  threshold,
  baseline,
  scored,
}: {
  examples: readonly Example[];
  population: Population;
  threshold: number;
  baseline: Baseline;
  scored: ReadonlySet<number>;
}) {
  const width = 460;
  const height = 160;
  const left = 16;
  const right = width - 16;
  const bottom = height - 34;
  const xs = scale(0, 1, left, right);
  const binWidth = 0.012;
  const stacks = new Map<number, number>();
  const placed = examples.map((example) => {
    const bin = Math.round(example.score / binWidth);
    const level = stacks.get(bin) ?? 0;
    stacks.set(bin, level + 1);
    return { example, x: xs(example.score), level };
  });
  const cut =
    baseline === "always-negative" ? right : baseline === "always-positive" ? left : xs(Math.min(1, Math.max(0, threshold)));
  const rare = population === "rare";
  const positive = placed.find((item) => item.example.label === 1);
  const alarm = placed.find((item) => item.example.id === 12);
  const dotY = (level: number) => bottom - 5 - level * 8.2;
  const positives = examples.filter((example) => example.label === 1).map((example) => example.score);
  const negatives = examples.filter((example) => example.label === 0).map((example) => example.score);
  const predicted =
    baseline === "learned"
      ? `Rows with score at or above ${threshold.toFixed(2)} are predicted positive.`
      : baseline === "always-negative"
        ? "Every row is predicted negative."
        : "Every row is predicted positive.";
  const description = rare
    ? "The only positive, row 99, scores 0.42. Row 12, a negative, scores 0.88. The other 98 score between 0.04 and 0.26."
    : `The ${positives.length} positives, drawn as diamonds, score between ${Math.min(...positives).toFixed(2)} and ${Math.max(
        ...positives,
      ).toFixed(2)}. The ${negatives.length} negatives, drawn as circles, score between ${Math.min(...negatives).toFixed(2)} and ${Math.max(
        ...negatives,
      ).toFixed(2)}${Math.min(...positives) < Math.max(...negatives) ? ", so the two ranges overlap" : ""}.`;
  return (
    <svg
      className="gw-plot eval-strip"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`All ${examples.length} scores on an axis from 0 to 1. ${predicted} ${description}`}
    >
      <rect className="eval-fire" x={cut} y={12} width={Math.max(0, right - cut)} height={bottom - 12} />
      <text className="gw-note" x={right - 4} y={24} textAnchor="end">
        predicted 1 →
      </text>
      <text className="gw-note" x={left + 4} y={24}>
        ← predicted 0
      </text>
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      {[0, 0.2, 0.4, 0.6, 0.8, 1].map((value) => (
        <g key={value}>
          <line className="gw-axis" x1={xs(value)} x2={xs(value)} y1={bottom} y2={bottom + 4} />
          <text className="gw-tick" x={xs(value)} y={bottom + 15} textAnchor="middle">
            {value.toFixed(1)}
          </text>
        </g>
      ))}
      <text className="gw-label" x={(left + right) / 2} y={bottom + 30} textAnchor="middle">
        detector score
      </text>
      {placed.map(({ example, x, level }) => {
        const faded = !scored.has(example.id);
        if (example.label === 1) {
          const y = dotY(level);
          return (
            <path
              key={example.id}
              className={`eval-positive${faded ? " is-faded" : ""}`}
              d={`M${x} ${y - 6} l6 6 l-6 6 l-6 -6 z`}
            />
          );
        }
        return (
          <circle
            key={example.id}
            className={`eval-negative${rare && example.id === 12 ? " is-alarm" : ""}${faded ? " is-faded" : ""}`}
            cx={x}
            cy={dotY(level)}
            r="3.4"
          />
        );
      })}
      {rare && positive && (
        <text className="gw-note" x={positive.x} y={dotY(positive.level) - 10} textAnchor="middle">
          row 99 · y = 1
        </text>
      )}
      {rare && alarm && (
        <text className="gw-note" x={alarm.x} y={dotY(alarm.level) - 9} textAnchor="middle">
          row 12 · y = 0
        </text>
      )}
      {baseline === "learned" && (
        <g>
          <line className="eval-threshold" x1={cut} x2={cut} y1={10} y2={bottom} />
          <text className="gw-label" x={cut + (threshold > 0.8 ? -4 : 4)} y={44} textAnchor={threshold > 0.8 ? "end" : "start"}>
            t = {threshold.toFixed(2)}
          </text>
        </g>
      )}
    </svg>
  );
}

const LINES = [
  { key: "accuracy", label: "accuracy", className: "gw-forward" },
  { key: "precision", label: "precision", className: "gw-attention is-dashed" },
  { key: "recall", label: "recall", className: "gw-loss is-dotted" },
  { key: "f1", label: "F1", className: "gw-positive is-dashdot" },
] as const;

/** Every reading of the matrix at every threshold on the scored rows, with the current t marked. */
export function SweepPlot({ rows, threshold }: { rows: readonly Example[]; threshold: number }) {
  const width = 460;
  const height = 210;
  const left = 38;
  const right = width - 12;
  const top = 14;
  const bottom = height - 30;
  const xs = scale(0, 1, left, right);
  const ys = scale(0, 1, bottom, top);
  const points = sweep(rows);
  const segments = (key: (typeof LINES)[number]["key"]) => {
    const runs: string[] = [];
    let current: string[] = [];
    for (const point of points) {
      const value = point[key];
      if (value === null) {
        if (current.length > 1) runs.push(current.join(" "));
        current = [];
        continue;
      }
      current.push(`${xs(point.threshold).toFixed(1)},${ys(value).toFixed(1)}`);
    }
    if (current.length > 1) runs.push(current.join(" "));
    return runs;
  };
  const at = points[Math.round(Math.min(1, Math.max(0, threshold)) * 100)];
  return (
    <svg
      className="gw-plot eval-sweep"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Accuracy, precision, recall, and F1 for every threshold from 0 to 1 on the ${rows.length} scored rows. At t = ${threshold.toFixed(
        2,
      )}: ${LINES.map((line) => {
        const value = at[line.key];
        return `${line.label} ${value === null ? "n/a (0/0)" : `${(value * 100).toFixed(1)} percent`}`;
      }).join(", ")}.`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {Math.round(value * 100)}%
          </text>
        </g>
      ))}
      {[0, 0.2, 0.4, 0.6, 0.8, 1].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value.toFixed(1)}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 26} textAnchor="end">
        threshold t
      </text>
      {LINES.map((line) =>
        segments(line.key).map((run, index) => (
          <polyline key={`${line.key}-${index}`} className={line.className} points={run} />
        )),
      )}
      <line className="eval-threshold" x1={xs(threshold)} x2={xs(threshold)} y1={top} y2={bottom} />
      {LINES.map((line) => {
        const value = at[line.key];
        if (value === null) return null;
        const tone = line.className.split(" ")[0].replace("gw-", "");
        return <circle key={`dot-${line.key}`} className={`gw-dot is-${tone}`} cx={xs(threshold)} cy={ys(value)} r="3.4" />;
      })}
      {(() => {
        // Direct labels beside the marker, nudged apart so equal values stay readable.
        const labels = LINES.map((line) => ({ line, value: at[line.key] }))
          .filter((item): item is { line: (typeof LINES)[number]; value: number } => item.value !== null)
          .sort((a, b) => b.value - a.value)
          .map((item) => ({ ...item, y: ys(item.value) + 3.5 }));
        for (let index = 1; index < labels.length; index += 1) {
          labels[index].y = Math.max(labels[index].y, labels[index - 1].y + 11);
        }
        const last = labels.length - 1;
        if (last >= 0 && labels[last].y > bottom - 4) {
          labels[last].y = bottom - 4;
          for (let index = last - 1; index >= 0; index -= 1) {
            labels[index].y = Math.min(labels[index].y, labels[index + 1].y - 11);
          }
        }
        const flip = threshold > 0.7;
        return labels.map((item) => (
          <text
            key={`label-${item.line.key}`}
            className={`gw-note gw-halo is-${item.line.className.split(" ")[0].replace("gw-", "")}`}
            x={xs(threshold) + (flip ? -7 : 7)}
            y={item.y}
            textAnchor={flip ? "end" : "start"}
          >
            {item.line.label} {Math.round(item.value * 1000) / 10}%
          </text>
        ));
      })()}
    </svg>
  );
}

/** ŷ = x on five points; each residual is drawn, and the outlier's label stays inside the frame. */
export function ResidualPlot({ points }: { points: ReadonlyArray<{ x: number; y: number }> }) {
  const width = 360;
  const height = 220;
  const left = 34;
  const right = width - 14;
  const top = 14;
  const bottom = height - 28;
  const xs = scale(-0.3, 4.3, left, right);
  const ys = scale(0, 16.5, bottom, top);
  return (
    <svg
      className="gw-plot eval-residuals"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Identity line y = x with five points. Residuals ${points
        .map((point) => (point.y - point.x).toFixed(2))
        .join(", ")}.`}
    >
      {[0, 4, 8, 12, 16].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
        </g>
      ))}
      {[0, 1, 2, 3, 4].map((value) => (
        <text key={value} className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
          {value}
        </text>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 25} textAnchor="end">
        x
      </text>
      <line className="gw-forward" x1={xs(-0.3)} y1={ys(0)} x2={xs(4.3)} y2={ys(4.3)} />
      <text className="gw-note is-forward" x={xs(4.25)} y={ys(4.3) + 14} textAnchor="end">
        ŷ = x
      </text>
      {points.map((point) => (
        <g key={point.x}>
          <line className="gw-loss" x1={xs(point.x)} y1={ys(point.x)} x2={xs(point.x)} y2={ys(point.y)} />
          <circle className="gw-dot is-filled" cx={xs(point.x)} cy={ys(point.y)} r="3.6" />
        </g>
      ))}
      {points.length > 4 && (
        <text className="gw-note" x={xs(points[4].x) - 8} y={ys(points[4].y) + 4} textAnchor="end">
          outlier (4, {points[4].y.toFixed(1)}), residual {(points[4].y - 4).toFixed(1)}
        </text>
      )}
    </svg>
  );
}

/** ROC on the scored rows, the chance diagonal, and where the current threshold sits on it. */
export function RocPlot({ rows, threshold }: { rows: readonly Example[]; threshold: number }) {
  const size = 220;
  const left = 44;
  const right = size - 10;
  const top = 12;
  const bottom = size - 30;
  const xs = scale(0, 1, left, right);
  const ys = scale(0, 1, bottom, top);
  const curve = roc(rows);
  const at = countMatrix(rows, "learned", threshold);
  const negatives = at.tn + at.fp;
  const positives = at.tp + at.fn;
  const here = negatives > 0 && positives > 0 ? { fpr: at.fp / negatives, tpr: at.tp / positives } : null;
  return (
    <svg
      className="gw-plot eval-roc"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={
        curve
          ? `ROC curve on the ${rows.length} scored rows. AUC ${curve.auc.toFixed(3)}. At t = ${threshold.toFixed(2)} the false positive rate is ${here?.fpr.toFixed(3)} and the true positive rate is ${here?.tpr.toFixed(3)}.`
          : "No ROC curve: the scored split has no positive, so the true positive rate is 0/0."
      }
    >
      {[0, 0.5, 1].map((value) => (
        <g key={value}>
          <line className="gw-grid" x1={left} x2={right} y1={ys(value)} y2={ys(value)} />
          <line className="gw-grid" x1={xs(value)} x2={xs(value)} y1={top} y2={bottom} />
          <text className="gw-tick" x={left - 5} y={ys(value) + 3.5} textAnchor="end">
            {value}
          </text>
          <text className="gw-tick" x={xs(value)} y={bottom + 13} textAnchor="middle">
            {value}
          </text>
        </g>
      ))}
      <line className="gw-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="gw-axis" x1={left} x2={left} y1={top} y2={bottom} />
      <text className="gw-label" x={right} y={bottom + 26} textAnchor="end">
        false positive rate
      </text>
      <text className="gw-label" x={12} y={(top + bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(top + bottom) / 2})`}>
        true positive rate
      </text>
      <line className="gw-grid eval-chance" x1={xs(0)} y1={ys(0)} x2={xs(1)} y2={ys(1)} />
      {curve ? (
        <>
          <polygon
            className="eval-auc"
            points={[
              ...curve.points.map((point) => `${xs(point.fpr)},${ys(point.tpr)}`),
              `${xs(1)},${ys(0)}`,
              `${xs(0)},${ys(0)}`,
            ].join(" ")}
          />
          <polyline className="gw-forward" points={curve.points.map((point) => `${xs(point.fpr)},${ys(point.tpr)}`).join(" ")} />
          {here && <circle className="gw-dot is-filled" cx={xs(here.fpr)} cy={ys(here.tpr)} r="4" />}
          <text className="gw-label is-forward" x={xs(0.5)} y={ys(0.22)} textAnchor="middle">
            AUC {curve.auc.toFixed(3)}
          </text>
        </>
      ) : (
        <text className="gw-note" x={(left + right) / 2} y={(top + bottom) / 2} textAnchor="middle">
          no positives: TPR is 0/0
        </text>
      )}
    </svg>
  );
}
