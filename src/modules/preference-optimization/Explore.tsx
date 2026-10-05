import { useMemo } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  TINY_CORPORA,
  trainTinyDpo,
  trainTinyModel,
  type ModuleContext,
  type TinyPreferencePair,
} from "@app/module-sdk";
import {
  dpoLoss,
  gradientWeight,
  HELD_OUT_PAIRS,
  heldOutGapTrace,
  isReferenceChoice,
  marginForLoss,
  MAX_FIELD_LENGTH,
  MAX_PAIRS,
  parsePairs,
  sanitizePairsText,
  scorePair,
  SFT_STEPS,
  trainSftReference,
  type PairScore,
  type ReferenceChoice,
} from "./dpo";

const REFERENCE_EPOCHS = 50;
const referenceText = `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`;

const DEFAULT_PAIRS = [
  "the | fog settles over the harbor. | fog fog fog fog fog.",
  "warm the | pan and add a spoon of oil. | pan and the pan and the pan.",
  "simmer the | sauce until it thickens. | sauce and simmer the sauce.",
];

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;

/* -------------------------------------------------------------------------- */
/* Loss curve with the pairs on it                                             */
/* -------------------------------------------------------------------------- */

function LossCurve({ beta, scores }: { beta: number; scores: PairScore[] }) {
  const width = 460;
  const height = 250;
  const left = 44;
  const right = width - 14;
  const top = 16;
  const bottom = height - 40;
  const largest = Math.max(0, ...scores.map((score) => score.margin));
  const xMin = -2;
  const xMax = Math.max(12, Math.ceil(largest + 1));
  const yMax = 1.2;
  const xAt = (value: number) => left + ((value - xMin) / (xMax - xMin)) * (right - left);
  const yAt = (value: number) => bottom - (Math.min(yMax, value) / yMax) * (bottom - top);
  const curve = (value: number) => {
    const points: string[] = [];
    for (let step = 0; step <= 120; step += 1) {
      const m = xMin + ((xMax - xMin) * step) / 120;
      points.push(`${step === 0 ? "M" : "L"}${xAt(m).toFixed(1)},${yAt(dpoLoss(value, m)).toFixed(1)}`);
    }
    return points.join(" ");
  };
  const betas = [
    { value: beta / 2, className: "is-half", name: `β ${(beta / 2).toFixed(2)}` },
    { value: beta, className: "is-current", name: `β ${beta.toFixed(2)} (current)` },
    { value: beta * 2, className: "is-double", name: `β ${(beta * 2).toFixed(2)}` },
  ];
  const tickStep = xMax <= 12 ? 2 : xMax <= 24 ? 4 : 10;
  const xTicks = Array.from({ length: Math.floor(xMax / tickStep) + 1 }, (_, index) => index * tickStep);
  return (
    <div className="pref-curve">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`DPO loss against the log-ratio margin for beta ${(beta / 2).toFixed(2)}, ${beta.toFixed(2)} and ${(beta * 2).toFixed(2)}. ${scores
          .map((score, index) => `Pair ${index + 1} at margin ${score.margin.toFixed(2)}, loss ${dpoLoss(beta, score.margin).toFixed(3)}`)
          .join("; ")}.`}
      >
        {[0, 0.4, 0.8, 1.2].map((value) => (
          <g key={`y-${value}`}>
            <line className="pref-curve__grid" x1={left} x2={right} y1={yAt(value)} y2={yAt(value)} />
            <text className="pref-curve__tick" x={left - 6} y={yAt(value) + 3} textAnchor="end">
              {value.toFixed(1)}
            </text>
          </g>
        ))}
        {xTicks.map((value) => (
          <text key={`x-${value}`} className="pref-curve__tick" x={xAt(value)} y={bottom + 14} textAnchor="middle">
            {value}
          </text>
        ))}
        <line className="pref-curve__zero" x1={xAt(0)} x2={xAt(0)} y1={top} y2={bottom} />
        <line className="pref-curve__axis" x1={left} x2={right} y1={bottom} y2={bottom} />
        <text className="pref-curve__label" x={(left + right) / 2} y={height - 8} textAnchor="middle">
          log-ratio margin m = Δ log p(chosen) − Δ log p(rejected), in nats
        </text>
        <text className="pref-curve__label" x={left} y={top - 4}>
          loss −log σ(β·m)
        </text>
        <text className="pref-curve__note" x={xAt(0) + 4} y={yAt(Math.LN2) - 6}>
          ln 2 at step 0
        </text>
        {betas.map((entry) => (
          <path key={entry.className} className={`pref-curve__line ${entry.className}`} d={curve(entry.value)} />
        ))}
        {scores.map((score, index) => {
          const x = xAt(score.margin);
          const y = yAt(dpoLoss(beta, score.margin));
          return (
            <g key={index}>
              <circle className="pref-curve__dot" cx={x} cy={y} r={6} />
              <text className="pref-curve__dot-label" x={x} y={y + 3.5} textAnchor="middle">
                {index + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="pref-curve__legend" aria-hidden="true">
        {betas.map((entry) => (
          <span key={entry.className}>
            <i className={`pref-curve__key ${entry.className}`} />
            {entry.name}
          </span>
        ))}
        <span>
          <i className="pref-curve__key is-dot" />
          numbered pairs, on the current curve
        </span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* One comparison, scored                                                      */
/* -------------------------------------------------------------------------- */

function PairBlock({
  pair,
  score,
  beta,
  scale,
  label,
}: {
  pair: TinyPreferencePair;
  score: PairScore;
  beta: number;
  scale: number;
  /** "1", "2" for training pairs; "H1", "H2" for held-out ones. */
  label: string;
}) {
  const chosen = beta * score.chosenShift;
  const rejected = beta * score.rejectedShift;
  const position = (value: number) => 50 + (value / scale) * 46;
  return (
    <div className={`pref-pair ${score.margin > 0 ? "is-ordered" : "is-unordered"}`}>
      <div className="pref-pair__head">
        <b>{label}</b>
        <span>{pair.prompt.trim() || "(no prompt)"}</span>
        <small>{score.margin > 0 ? "ordered correctly" : "not ordered"}</small>
      </div>
      <div className="pref-pair__text">
        <p><small>preferred</small>{pair.chosen}</p>
        <p><small>rejected</small>{pair.rejected}</p>
      </div>
      <div
        className="pref-dumbbell"
        role="img"
        aria-label={`Pair ${label}: chosen reward ${signed(chosen)}, rejected reward ${signed(rejected)}, margin ${signed(chosen - rejected)}.`}
      >
        <i className="pref-dumbbell__zero" />
        <i
          className="pref-dumbbell__span"
          style={{
            left: `${Math.min(position(chosen), position(rejected))}%`,
            width: `${Math.abs(position(chosen) - position(rejected))}%`,
          }}
        />
        <i className="pref-dumbbell__rejected" style={{ left: `${position(rejected)}%` }} />
        <i className="pref-dumbbell__chosen" style={{ left: `${position(chosen)}%` }} />
      </div>
      <div className="pref-pair__values">
        <span className="is-rejected">■ rejected {signed(rejected)}</span>
        <span className="is-chosen">● chosen {signed(chosen)}</span>
      </div>
      <i className="pref-pair__foot">
        margin {signed(chosen - rejected)} · reference {score.referenceChosen > score.referenceRejected ? "already preferred it" : "preferred the rejected one"}
        {pair.chosen.length !== pair.rejected.length ? ` (lengths ${pair.chosen.length} vs ${pair.rejected.length})` : ""}
      </i>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Explore                                                                     */
/* -------------------------------------------------------------------------- */

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const beta = Math.min(1.5, Math.max(0.05, asNumber(state, "beta", 0.4)));
  const steps = Math.min(120, Math.max(0, Math.round(asNumber(state, "steps", 40))));
  const learningRate = Math.min(1, Math.max(0.05, asNumber(state, "learningRate", 0.35)));
  const pairsText = sanitizePairsText(asString(state, "pairsText", DEFAULT_PAIRS.join("\n")));
  const referenceChoice: ReferenceChoice = isReferenceChoice(state.reference) ? state.reference : "base";

  const pairs = useMemo(() => parsePairs(pairsText), [pairsText]);

  const base = useMemo(() => trainTinyModel({ text: referenceText, epochs: REFERENCE_EPOCHS, seed: 1 }), []);
  /** The frozen model every reward is measured against: the base table, or the base fine-tuned on the preferred completions. */
  const referenceWeights = useMemo(
    () => (referenceChoice === "sft" ? trainSftReference(base.weights, pairs) : base.weights),
    [base.weights, pairs, referenceChoice],
  );

  const betaSweep = useMemo(
    () =>
      [beta / 2, beta, beta * 2].map((value) => ({
        beta: value,
        run: trainTinyDpo({ reference: referenceWeights, pairs, beta: value, steps, learningRate }),
      })),
    [beta, learningRate, pairs, referenceWeights, steps],
  );
  const run = betaSweep[1].run;

  const scores = useMemo(
    () => pairs.map((pair) => scorePair(run.weights, referenceWeights, pair)),
    [pairs, referenceWeights, run.weights],
  );

  /** The same policy scored on comparisons the optimizer never saw. */
  const heldOutScores = useMemo(
    () => HELD_OUT_PAIRS.map((pair) => scorePair(run.weights, referenceWeights, pair)),
    [referenceWeights, run.weights],
  );
  const heldOutOrdered = heldOutScores.filter((score) => score.margin > 0).length;
  const trainedOrdered = scores.filter((score) => score.margin > 0).length;

  /** Held-out margin at five evenly spaced step counts, read along one pass of the same DPO update. */
  const heldOutCurve = useMemo(() => {
    const marks = Array.from(new Set([0, 1, 2, 3, 4, 5].map((part) => Math.round((steps * part) / 5))));
    const gaps = heldOutGapTrace({
      reference: referenceWeights,
      pairs,
      heldOut: HELD_OUT_PAIRS,
      beta,
      learningRate,
      marks,
    });
    const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0);
    return marks.map((mark, index) => ({ x: mark, y: beta * mean(gaps[index]) }));
  }, [beta, learningRate, pairs, referenceWeights, steps]);

  const last = run.history[run.history.length - 1];
  const meanGap = scores.length ? scores.reduce((sum, score) => sum + score.margin, 0) / scores.length : 0;
  const chosenMove = last?.chosenShift ?? 0;
  const rejectedMove = last?.rejectedShift ?? 0;
  const lopsided =
    steps === 0 || pairs.length === 0
      ? "At step 0 the policy is the reference, so every shift and the margin are exactly zero."
      : Math.abs(rejectedMove) > Math.abs(chosenMove)
        ? `The loss only rewards the gap. Here the rejected side moved ${Math.abs(rejectedMove).toFixed(2)} while the chosen side moved ${Math.abs(chosenMove).toFixed(2)}, so most of the margin comes from making the rejected completion unlikely.`
        : `The loss only rewards the gap. Here the chosen side moved ${Math.abs(chosenMove).toFixed(2)} and the rejected side ${Math.abs(rejectedMove).toFixed(2)}, so most of the margin comes from raising the chosen completion.`;

  const rewardScale = Math.max(
    1e-6,
    ...[...scores, ...heldOutScores].flatMap((score) => [
      Math.abs(beta * score.chosenShift),
      Math.abs(beta * score.rejectedShift),
    ]),
  );
  const driftMax = Math.max(1e-4, ...betaSweep.flatMap((entry) => entry.run.history.map((point) => point.drift)));

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="Implicit reward margin during training" className="tg-margin-panel">
        <SurfaceHeading
          kicker={`${pairs.length} training pairs · ${HELD_OUT_PAIRS.length} held out · beta ${beta.toFixed(2)} · ${steps} steps`}
          title="The objective moves the gap, not the two sides independently"
          aside={
            <>
              <span className="tg-badge">{(run.accuracy * 100).toFixed(0)}% ordered correctly</span>
              <span className="tg-badge">
                held-out: {heldOutOrdered} of {HELD_OUT_PAIRS.length} ordered
              </span>
            </>
          }
        />
        <LineChart
          label="Implicit reward margin and each side's shift from the reference model"
          xLabel="DPO step"
          yLabel="implicit reward, β × log-probability shift (nats)"
          series={[
            {
              id: "margin",
              name: "margin (chosen − rejected)",
              tone: "forward",
              points: run.history.map((point) => ({ x: point.step, y: point.margin })),
            },
            {
              id: "chosen",
              name: "chosen reward",
              tone: "attention",
              dash: "dashed",
              points: run.history.map((point) => ({ x: point.step, y: point.chosenShift })),
            },
            {
              id: "rejected",
              name: "rejected reward",
              tone: "loss",
              dash: "dotted",
              points: run.history.map((point) => ({ x: point.step, y: point.rejectedShift })),
            },
            {
              id: "held-out",
              name: "held-out margin (never trained on)",
              tone: "gradient",
              points: heldOutCurve,
            },
          ]}
          footnote={`Averages over the pairs. ${lopsided} The held-out line is the same margin on three pairs the optimizer never saw, measured at five step counts.`}
        />
        <div className="metric-row pref-metrics">
          <Metric label="DPO loss" value={(last?.loss ?? 0).toFixed(4)} tone="loss" />
          <Metric label="Margin, β-scaled" value={(last?.margin ?? 0).toFixed(3)} tone="forward" />
          <Metric label="Log-ratio gap m = margin ÷ β" value={`${meanGap.toFixed(2)} nats`} />
          <Metric
            label="Held-out pairs ordered correctly"
            value={`${heldOutOrdered} of ${HELD_OUT_PAIRS.length}`}
            tone={heldOutOrdered < trainedOrdered ? "loss" : "forward"}
          />
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Where each pair sits on the loss" className="pref-curve-panel">
          <SurfaceHeading kicker="The same loss at three betas" title="Why a pair stops pulling" />
          <LossCurve beta={beta} scores={scores} />
          <ol className="pref-weights">
            {scores.map((score, index) => (
              <li key={index}>
                <b>{index + 1}</b>
                <span>m {score.margin.toFixed(2)}</span>
                <span>loss {dpoLoss(beta, score.margin).toFixed(3)}</span>
                <span>pull σ(−β·m) {(gradientWeight(beta, score.margin) * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ol>
          <p className="lab-note">
            Loss 0.1 needs m = {marginForLoss(beta / 2, 0.1).toFixed(2)} at β {(beta / 2).toFixed(2)},{" "}
            {marginForLoss(beta, 0.1).toFixed(2)} at β {beta.toFixed(2)} and {marginForLoss(beta * 2, 0.1).toFixed(2)} at β{" "}
            {(beta * 2).toFixed(2)}. A higher β reaches the flat part of the curve with a smaller change to
            the model, and a pair on the flat part barely moves the weights.
          </p>
        </LabSurface>

        <LabSurface label="Drift from the reference model" className="tg-drift-panel a11-narrow-chart">
          <SurfaceHeading
            kicker="Three betas, same pairs, same steps"
            title="How far the policy wandered from where it started"
            aside={<span className="tg-badge">KL {run.finalDrift.toFixed(4)}</span>}
          />
          <LineChart
            label="Average KL divergence from the reference model for three values of beta"
            xLabel="DPO step"
            yLabel="mean KL(policy ‖ reference) over 30 rows (nats)"
            yDomain={[0, driftMax * 1.15]}
            series={betaSweep.map((entry, index) => {
              const final = entry.run.history.at(-1);
              const gap = final && entry.beta > 0 ? final.margin / entry.beta : 0;
              return {
                id: `beta-${index}`,
                name: `β ${entry.beta.toFixed(2)}${index === 1 ? " (current)" : ""} · gap ${gap.toFixed(1)} · KL`,
                tone: index === 1 ? "forward" : "muted",
                dash: index === 0 ? "dotted" : index === 2 ? "dashed" : "solid",
                format: (value: number) => value.toFixed(4),
                points: entry.run.history.map((point) => ({ x: point.step, y: point.drift })),
              };
            })}
            footnote="An unweighted average over all 30 context rows, so it measures the whole table rather than the three prompts. Gap is each run's mean log-ratio margin in nats."
          />
          <div className="tg-diff">
            <div className="tg-sample">
              <span>Reference model</span>
              <p>{sampleTinyText(referenceWeights, { prompt: "the ", length: 76, temperature: 0.7, seed: 8 })}</p>
            </div>
            <div className="tg-sample">
              <span>After preference optimization</span>
              <p>{sampleTinyText(run.weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 8 })}</p>
            </div>
          </div>
        </LabSurface>
      </div>

      <div className="tg-column">
        <LabSurface label="Preference pair editor" className="tg-pairs-panel">
          <SurfaceHeading kicker="Your labels" title="Write the comparisons yourself" />
          <label className="tg-editor">
            <span>prompt | preferred | rejected</span>
            <textarea
              aria-label="Preference pairs"
              rows={5}
              value={pairsText}
              maxLength={MAX_PAIRS * (3 * MAX_FIELD_LENGTH + 9)}
              onChange={(event) => setState({ pairsText: sanitizePairsText(event.target.value) })}
            />
          </label>
          <p className="lab-note">
            Up to {MAX_PAIRS} lines, each part up to {MAX_FIELD_LENGTH} characters; anything beyond is
            ignored so training stays instant.
          </p>
          <SegmentedControl
            label="Reference model"
            value={referenceChoice}
            options={[
              { value: "base", label: "Base model" },
              { value: "sft", label: "SFT checkpoint" },
            ]}
            onChange={(value) => {
              setState({ reference: value });
              narrate(value === "sft" ? "Reference is the SFT checkpoint." : "Reference is the base model.");
            }}
          />
          <p className="lab-note">
            {referenceChoice === "sft"
              ? `The base table fine-tuned for ${SFT_STEPS} steps on your preferred completions, with the prompt read as context and never graded. It is trained here; Instruction tuning & templates' own weights are never loaded.`
              : "The base table: 50 epochs on Harbor weather notes and Recipe steps, with no instruction tuning."}
          </p>
          <RangeControl
            label="DPO steps"
            min={0}
            max={120}
            step={2}
            value={steps}
            format={(value) => (value === 0 ? "untrained" : `${value} steps`)}
            onChange={(value) => {
              setState({ steps: value });
              narrate(`${value} DPO steps.`);
            }}
          />
          <RangeControl
            label="Beta"
            min={0.05}
            max={1.5}
            step={0.05}
            value={beta}
            format={(value) => value.toFixed(2)}
            onChange={(value) => {
              setState({ beta: value });
              narrate(`Beta ${value.toFixed(2)}.`);
            }}
          />
          <RangeControl
            label="DPO learning rate"
            min={0.05}
            max={1}
            step={0.05}
            value={learningRate}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ learningRate: value })}
          />
        </LabSurface>

        <LabSurface label="Per-pair rewards" className="tg-pair-scores">
          <SurfaceHeading kicker="One row per comparison" title="Implicit reward, pair by pair" />
          <div className="pref-pairs">
            {pairs.map((pair, index) => (
              <PairBlock key={index} pair={pair} score={scores[index]} beta={beta} scale={rewardScale} label={`${index + 1}`} />
            ))}
            {pairs.length === 0 && (
              <p className="lab-note">
                Each line needs three parts separated by vertical bars: a prompt, the completion you
                prefer, and the one you do not.
              </p>
            )}
          </div>
          <h4 className="pref-group-label">
            Held-out pairs · never used by the optimizer · {heldOutOrdered} of {HELD_OUT_PAIRS.length} ordered
            correctly
          </h4>
          <div className="pref-pairs">
            {HELD_OUT_PAIRS.map((pair, index) => (
              <PairBlock
                key={`held-${index}`}
                pair={pair}
                score={heldOutScores[index]}
                beta={beta}
                scale={rewardScale}
                label={`H${index + 1}`}
              />
            ))}
          </div>
          <p className="lab-note">
            Each dumbbell spans one pair&rsquo;s margin on an axis centred at zero (the reference). A
            reward is β times how much more log-probability the policy gives a completion than the
            reference did. It is not a quality score, and the reference&rsquo;s raw log-probability
            favours the shorter completion whatever its quality. The H pairs are fixed comparisons of
            the same kind that no step ever trains on, so a pair ordered here has carried over from the
            training pairs and a pair not ordered has not.
          </p>
        </LabSurface>
      </div>
    </div>
  );
}
