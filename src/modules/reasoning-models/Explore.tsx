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
  GROUP_RANGE,
  ITERATIONS,
  K_RANGE,
  LENGTH_COST,
  LENGTHS,
  MAX_LENGTH,
  passProbability,
  percent,
  RATE_RANGE,
  SEED_RANGE,
  STEP_SUCCESS,
  trainGroupRelative,
  type GrpoPoint,
} from "./grpo";
import { sanitizeState } from "./state";
import { binomialPmf, votingCurves, winGivenCorrect, type WrongSpread } from "./voting";

const PROBLEMS = {
  sunday: {
    label: "Sunday hours",
    question: "When does the north dock close on Sunday?",
    answer: "16:00",
    trace: [
      { kind: "useful", text: "The question names Sunday and the north dock." },
      { kind: "useful", text: "Harbor notes store weekday 18:00 and Sunday 16:00 as different facts." },
      { kind: "useful", text: "Select the Sunday slot, not the weekday slot." },
      { kind: "padding", text: "Restate that docks are in a harbor and harbors have notes." },
      { kind: "unfaithful", text: "Mention flares in shed B, which this question does not use." },
      { kind: "useful", text: "Commit to 16:00." },
    ],
  },
  add: {
    label: "17 + 28",
    question: "What is 17 + 28?",
    answer: "45",
    trace: [
      { kind: "useful", text: "Add the ones: 7 + 8 = 15, write 5 carry 1." },
      { kind: "useful", text: "Add the tens: 1 + 2 + 1 = 4." },
      { kind: "useful", text: "The pair is 45." },
      { kind: "padding", text: "Note that addition is associative, which is unused here." },
      { kind: "unfaithful", text: "Check 17 × 28 as if the task were a product." },
      { kind: "useful", text: "Reconfirm 45 and stop." },
    ],
  },
  ferry: {
    label: "Dog on the ferry",
    question: "Can a dog ride the 15:40 ferry?",
    answer: "Yes, with a tag.",
    trace: [
      { kind: "useful", text: "The 15:40 sailing exists on the timetable." },
      { kind: "useful", text: "The same note says dogs need a tag." },
      { kind: "useful", text: "Permission is yes, conditional on the tag." },
      { kind: "padding", text: "Recall that tickets cost 12 tokens, unused for this yes/no." },
      { kind: "unfaithful", text: "Invent a leash rule the corpus never stated." },
      { kind: "useful", text: "Answer: yes, with a tag." },
    ],
  },
} as const;

const SPREADS: Record<string, { label: string; wrong: WrongSpread }> = {
  same: { label: "Always the same one", wrong: 1 },
  three: { label: "Split over 3", wrong: 3 },
  unique: { label: "All different", wrong: "unique" },
};
const MAX_SAMPLES = 32;

const pct = (value: number) =>
  value >= 0.99995 && value < 1 ? ">99.99%" : `${(value * 100).toFixed(value > 0.995 && value < 1 ? 2 : 1)}%`;

function VotePlot({
  curves,
  samples,
}: {
  curves: ReturnType<typeof votingCurves>;
  samples: number;
}) {
  const left = 46;
  const right = 700;
  const top = 18;
  const bottom = 196;
  const xAt = (n: number) => left + ((n - 1) / (MAX_SAMPLES - 1)) * (right - left);
  const yAt = (value: number) => bottom - value * (bottom - top);
  const path = (key: "single" | "majority" | "verifier") =>
    curves.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(point.n)},${yAt(point[key])}`).join(" ");
  const current = curves[samples - 1];
  return (
    <svg
      className="rm-vote-plot"
      viewBox="0 0 720 236"
      role="img"
      aria-label={`Accuracy against number of samples. At ${samples} samples: one sample ${pct(current.single)}, majority vote ${pct(current.majority)}, verifier picks ${pct(current.verifier)}. At 32 samples: majority vote ${pct(curves[MAX_SAMPLES - 1].majority)}, verifier ${pct(curves[MAX_SAMPLES - 1].verifier)}.`}
    >
      <text x={left} y={10}>
        accuracy on this problem
      </text>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line className="rm-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
          <text x={left - 6} y={yAt(tick) + 3} textAnchor="end">
            {(tick * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      {[1, 4, 8, 12, 16, 20, 24, 28, 32].map((tick) => (
        <text key={tick} x={xAt(tick)} y={bottom + 15} textAnchor="middle">
          {tick}
        </text>
      ))}
      <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="marker-guide" x1={xAt(samples)} x2={xAt(samples)} y1={top} y2={bottom} />
      <path className="rm-curve is-single" d={path("single")} />
      <path className="rm-curve is-verifier" d={path("verifier")} />
      <path className="rm-curve is-majority" d={path("majority")} />
      <circle className="rm-dot is-verifier" cx={xAt(samples)} cy={yAt(current.verifier)} r={4.5} />
      <rect className="rm-dot is-majority" x={xAt(samples) - 4} y={yAt(current.majority) - 4} width={8} height={8} />
      <circle className="rm-dot is-single" cx={xAt(samples)} cy={yAt(current.single)} r={3.5} />
      <text x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
        samples drawn per question (n)
      </text>
    </svg>
  );
}

function CountBars({ samples, accuracy, wrong }: { samples: number; accuracy: number; wrong: WrongSpread }) {
  const pmf = binomialPmf(samples, accuracy);
  const left = 46;
  const right = 700;
  const top = 16;
  const bottom = 130;
  const peak = Math.max(...pmf);
  const slot = (right - left) / (samples + 1);
  const width = Math.max(2, slot * 0.72);
  const xAt = (k: number) => left + slot * (k + 0.5);
  const hAt = (value: number) => (value / peak) * (bottom - top);
  const label = pmf
    .map((probability, k) => `${k} correct: ${pct(probability)}, vote wins ${pct(winGivenCorrect(samples, k, wrong))}`)
    .join("; ");
  const labelEvery = samples > 20 ? 4 : samples > 10 ? 2 : 1;
  return (
    <svg className="rm-vote-plot rm-count-plot" viewBox="0 0 720 166" role="img" aria-label={`Binomial distribution of correct samples out of ${samples}. ${label}.`}>
      <text x={left} y={10}>
        probability of k correct samples · solid part = the vote picks the right answer
      </text>
      <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      {pmf.map((probability, k) => {
        const win = winGivenCorrect(samples, k, wrong);
        const height = hAt(probability);
        return (
          <g key={k}>
            <rect className="rm-bar is-lose" x={xAt(k) - width / 2} y={bottom - height} width={width} height={height} />
            <rect
              className="rm-bar is-win"
              x={xAt(k) - width / 2}
              y={bottom - height * win}
              width={width}
              height={height * win}
            />
            {k % labelEvery === 0 && (
              <text x={xAt(k)} y={bottom + 14} textAnchor="middle">
                {k}
              </text>
            )}
          </g>
        );
      })}
      <text x={(left + right) / 2} y={bottom + 30} textAnchor="middle">
        correct samples out of {samples}
      </text>
    </svg>
  );
}

/**
 * The policy over trace length at one training update: solid bars are its
 * probabilities, outlined bars the starting policy, and the dashed line is the
 * checker's pass chance for a trace of that length.
 */
function PolicyBars({ point, initial, k }: { point: GrpoPoint; initial: GrpoPoint; k: number }) {
  const left = 46;
  const right = 664;
  const top = 24;
  const bottom = 138;
  const slot = (right - left) / MAX_LENGTH;
  const width = slot * 0.56;
  const xAt = (length: number) => left + slot * (length - 0.5);
  const maxMass = 0.6;
  const yMass = (value: number) => bottom - (Math.min(value, maxMass) / maxMass) * (bottom - top);
  const yPass = (value: number) => bottom - value * (bottom - top);
  const passes = LENGTHS.map((length) => passProbability(length, k));
  const label = `Policy over trace length after update ${point.iteration}: ${LENGTHS.map(
    (length, index) => `${length} steps ${percent(point.probabilities[index])}`,
  ).join(", ")}. Checker pass chance by length: ${LENGTHS.map(
    (length, index) => `${length} steps ${percent(passes[index])}`,
  ).join(", ")}. Traces shorter than ${k} steps never pass.`;
  return (
    <svg className="rm-vote-plot rm-policy-plot" viewBox="0 0 720 190" role="img" aria-label={label}>
      <text x={left} y={12}>
        policy over trace length · outline = starting policy · dashed line = checker pass chance
      </text>
      {[0, 0.3, 0.6].map((tick) => (
        <g key={tick}>
          <line className="rm-grid" x1={left} x2={right} y1={yMass(tick)} y2={yMass(tick)} />
          <text x={left - 6} y={yMass(tick) + 3} textAnchor="end">
            {percent(tick)}
          </text>
        </g>
      ))}
      {[0, 0.5, 1].map((tick) => (
        <text key={tick} x={right + 8} y={yPass(tick) + 3}>
          {percent(tick)}
        </text>
      ))}
      <line className="plot-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="marker-guide" x1={left + slot * (k - 1)} x2={left + slot * (k - 1)} y1={top} y2={bottom} />
      {LENGTHS.map((length, index) => {
        const height = bottom - yMass(point.probabilities[index]);
        const ghost = bottom - yMass(initial.probabilities[index]);
        return (
          <g key={length}>
            <rect
              className="rm-bar is-ghost"
              x={xAt(length) - width / 2 - 2}
              y={bottom - ghost}
              width={width + 4}
              height={ghost}
            />
            <rect
              className={`rm-bar ${length >= k ? "is-win" : "is-lose"}`}
              x={xAt(length) - width / 2}
              y={bottom - height}
              width={width}
              height={height}
            />
            {point.probabilities[index] >= 0.04 && (
              <text x={xAt(length)} y={bottom - height - 4} textAnchor="middle">
                {percent(point.probabilities[index])}
              </text>
            )}
            <text x={xAt(length)} y={bottom + 14} textAnchor="middle">
              {length}
            </text>
          </g>
        );
      })}
      <path
        className="rm-curve is-verifier"
        d={LENGTHS.map((length, index) => `${index === 0 ? "M" : "L"}${xAt(length)},${yPass(passes[index])}`).join(" ")}
      />
      {LENGTHS.map((length, index) => (
        <circle key={length} className="rm-dot is-verifier" cx={xAt(length)} cy={yPass(passes[index])} r={3.2} />
      ))}
      <text x={(left + right) / 2} y={bottom + 30} textAnchor="middle">
        {k > 1 ? `trace length L (steps) · lengths left of the rule are shorter than ${k} and never pass` : "trace length L (steps)"}
      </text>
    </svg>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const settings = sanitizeState(state);
  const keys = Object.keys(PROBLEMS) as Array<keyof typeof PROBLEMS>;
  const problem = settings.problem as keyof typeof PROBLEMS;
  const spec = PROBLEMS[problem];
  const budget = Math.min(settings.budget as number, spec.trace.length);
  const visible = spec.trace.slice(0, budget);
  const useful = visible.filter((step) => step.kind === "useful").length;
  const padding = visible.filter((step) => step.kind === "padding").length;
  const unfaithful = visible.filter((step) => step.kind === "unfaithful").length;

  const samples = settings.samples as number;
  const accuracy = settings.accuracy as number;
  const spreadKey = settings.spread as string;
  const spread = SPREADS[spreadKey];
  const curves = useMemo(() => votingCurves(MAX_SAMPLES, accuracy, spread.wrong), [accuracy, spread.wrong]);
  const current = curves[samples - 1];

  const grpoK = settings.grpoK as number;
  const grpoGroup = settings.grpoGroup as number;
  const grpoRate = settings.grpoRate as number;
  const grpoCost = settings.grpoCost as boolean;
  const grpoSeed = settings.grpoSeed as number;
  const grpoView = settings.grpoView as number;
  // The whole run is recomputed from the controls and the seed; no weights are stored in state.
  const run = useMemo(
    () => trainGroupRelative({ k: grpoK, group: grpoGroup, rate: grpoRate, cost: grpoCost, seed: grpoSeed }),
    [grpoK, grpoGroup, grpoRate, grpoCost, grpoSeed],
  );
  const history = run.history;
  const now = history[grpoView];
  const start = history[0];
  const passSeries = useMemo(
    () => [
      {
        id: "pass",
        name: "Checker pass rate",
        tone: "forward" as const,
        dash: "solid" as const,
        points: history.map((point) => ({ x: point.iteration, y: point.passRate })),
        format: (value: number) => value.toFixed(2),
      },
      ...(grpoCost
        ? [
            {
              id: "reward",
              name: "Mean reward after cost",
              tone: "gradient" as const,
              dash: "dashed" as const,
              points: history.map((point) => ({ x: point.iteration, y: point.meanReward })),
              format: (value: number) => value.toFixed(2),
            },
          ]
        : []),
    ],
    [history, grpoCost],
  );
  const lengthSeries = useMemo(
    () => [
      {
        id: "length",
        name: "Mean trace length",
        tone: "attention" as const,
        dash: "solid" as const,
        points: history.map((point) => ({ x: point.iteration, y: point.meanLength })),
        format: (value: number) => `${value.toFixed(1)} steps`,
      },
      {
        id: "needed",
        name: `Steps the task needs (k = ${grpoK})`,
        tone: "muted" as const,
        dash: "dotted" as const,
        points: [
          { x: 0, y: grpoK },
          { x: ITERATIONS, y: grpoK },
        ],
        format: (value: number) => `${value.toFixed(0)} steps`,
      },
    ],
    [history, grpoK],
  );

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="Problem" className="reason-problem-card">
        <SurfaceHeading kicker="Trained traces, not few-shot prompts" title="A question the policy already practiced" />
        <div className="reason-problem-control">
          <SegmentedControl
            label="Problem"
            value={problem}
            options={keys.map((key) => ({ value: key, label: PROBLEMS[key].label }))}
            onChange={(value) => setState({ problem: value, budget: 3 })}
          />
        </div>
        <p className="lab-note">{spec.question}</p>
        <p className="lab-note">
          In-context learning prompted a chain with worked examples: the prompt changes, the weights do not. A reasoning
          model is post-trained, usually with reinforcement learning against answers a program can check, so its weights
          make long intermediate traces likely. The next card trains a toy version of that recipe. The cards after it
          show authored traces and the sampling arithmetic behind spending more compute at answer time.
        </p>
      </LabSurface>

      <LabSurface label="Train against a checker" className="rm-grpo-card">
        <SurfaceHeading
          kicker={`k = ${grpoK} · group of ${grpoGroup} · seed ${grpoSeed} · update ${grpoView} of ${ITERATIONS}`}
          title="Reinforcement learning against a checker makes traces longer"
        />
        <p className="lab-note">
          This trains one number, the typical trace length, with a group-relative policy gradient (GRPO-style). It is not a
          language model, and the checker is a hand-written rule, not a learned reward model: a trace of L steps passes when
          at least {grpoK} of its steps land, and each step lands with probability {STEP_SUCCESS.toFixed(2)}. Every update
          samples {grpoGroup} traces, scores each against its own group, and moves the length toward the traces that beat
          the group average.
        </p>
        <div className="rm-grpo-controls">
          <RangeControl
            label="Steps the task needs (k)"
            min={K_RANGE.min}
            max={K_RANGE.max}
            step={1}
            value={grpoK}
            format={(value) => `${value} step${value === 1 ? "" : "s"}`}
            onChange={(value) => setState({ grpoK: value })}
          />
          <RangeControl
            label="Group size (G)"
            min={GROUP_RANGE.min}
            max={GROUP_RANGE.max}
            step={1}
            value={grpoGroup}
            format={(value) => `${value} traces`}
            onChange={(value) => setState({ grpoGroup: value })}
          />
          <RangeControl
            label="Learning rate"
            min={RATE_RANGE.min}
            max={RATE_RANGE.max}
            step={RATE_RANGE.step}
            value={grpoRate}
            format={(value) => value.toFixed(1)}
            onChange={(value) => setState({ grpoRate: Math.round(value * 10) / 10 })}
          />
          <RangeControl
            label="Seed"
            min={SEED_RANGE.min}
            max={SEED_RANGE.max}
            step={1}
            value={grpoSeed}
            format={(value) => `${value}`}
            onChange={(value) => setState({ grpoSeed: value })}
          />
          <RangeControl
            label="Training update"
            min={0}
            max={ITERATIONS}
            step={1}
            value={grpoView}
            format={(value) => `${value} of ${ITERATIONS}`}
            onChange={(value) => setState({ grpoView: value })}
          />
          <div className="rm-grpo-cost">
            <SegmentedControl
              label="Length cost"
              value={grpoCost ? "on" : "off"}
              options={[
                { value: "off", label: "Off" },
                { value: "on", label: `On · ${LENGTH_COST.toFixed(2)} per step` },
              ]}
              onChange={(value) => {
                setState({ grpoCost: value === "on" });
                narrate(
                  value === "on"
                    ? `Each step now costs ${LENGTH_COST.toFixed(2)} reward.`
                    : "Steps are free; only the checker's verdict is rewarded.",
                );
              }}
            />
          </div>
        </div>
        <div className="rm-grpo-charts">
          <LineChart
            label="Checker pass rate while training"
            xLabel={`update 0 → ${ITERATIONS}`}
            yLabel="chance a sampled trace passes the checker"
            xDomain={[0, ITERATIONS]}
            yDomain={[0, 1]}
            marker={{ x: grpoView, label: `update ${grpoView}` }}
            series={passSeries}
          />
          <LineChart
            label="Mean trace length while training"
            xLabel={`update 0 → ${ITERATIONS}`}
            yLabel="mean trace length (steps)"
            xDomain={[0, ITERATIONS]}
            yDomain={[1, MAX_LENGTH]}
            marker={{ x: grpoView, label: `update ${grpoView}` }}
            series={lengthSeries}
          />
        </div>
        <PolicyBars point={now} initial={start} k={grpoK} />
        <div className="metric-row">
          <Metric label="Checker pass rate" value={percent(now.passRate)} tone="forward" />
          <Metric label="Mean length" value={`${now.meanLength.toFixed(1)} steps`} />
          <Metric label="Steps past k" value={now.extraSteps.toFixed(1)} tone="loss" />
          <Metric label="Mean reward after cost" value={now.meanReward.toFixed(2)} />
          <Metric label="Groups with no signal" value={`${now.flatSoFar} of ${grpoView}`} />
        </div>
        <p className="lab-note">
          Curves and bars are exact expectations of the current policy; the run itself is {ITERATIONS} sampled updates,
          recomputed from the controls and the seed, so a different seed gives a different path. With the cost off nothing
          pushes length back down, and an extra step still adds a little pass chance, so length keeps creeping past k.
          With it on, each step costs {LENGTH_COST.toFixed(2)}, so length stops where a step no longer pays for itself. A
          group whose traces all pass, or all fail, has equal rewards and teaches nothing.
        </p>
        <p className="lab-note">
          Real RLVR and GRPO train a token-level policy over a language model, keep it close to a reference model with a KL
          penalty, clip the probability ratio, and average over many different prompts per batch. This card has none of
          those, and a real checker never lands each step with a fixed probability.
        </p>
      </LabSurface>

      <LabSurface label="Thinking budget" className="reason-budget-card">
        <SurfaceHeading
          kicker={`${budget} of ${spec.trace.length} authored steps`}
          title="Test-time compute is extra tokens, not a new mind"
        />
        <RangeControl
          label="Thinking budget"
          min={0}
          max={spec.trace.length}
          step={1}
          value={budget}
          format={(value) => `${value} step${value === 1 ? "" : "s"}`}
          onChange={(value) => setState({ budget: value })}
        />
        <div className="metric-row">
          <Metric label="Useful" value={`${useful}`} tone="forward" />
          <Metric label="Padding" value={`${padding}`} />
          <Metric label="Unfaithful" value={`${unfaithful}`} tone="loss" />
          <Metric label="Answer (authored)" value={spec.answer} />
        </div>
        <p className="lab-note">
          Every generated token is one more forward pass, and each step can read the results the earlier steps wrote
          down. That is why a trace helps on problems with serial dependencies, like carrying the 1 in 17 + 28 before
          adding the tens. The answer here is authored and prints at every budget; on hard multi-step problems, cutting
          a real model's trace short usually costs accuracy. Past the useful steps, extra tokens only add latency.
        </p>
      </LabSurface>

      <LabSurface label="Inspectable trace" className="reason-trace-card">
        <SurfaceHeading kicker={budget === 0 ? "no trace" : `${visible.length} steps shown`} title="Visible steps are generated tokens" />
        <ol className="reason-trace">
          {visible.length === 0 && (
            <li>No intermediate tokens. The answer slot still prints {spec.answer}, because this lab authored it.</li>
          )}
          {visible.map((step, index) => (
            <li key={index} className={`is-${step.kind}`}>
              <strong>{step.kind}</strong>
              <span>{step.text}</span>
            </li>
          ))}
        </ol>
        <p className="lab-note">
          useful / padding / unfaithful are labels this lab assigns. They are not a window into a separate inner
          algorithm. A process reward model would score each step; this page only shows them.
        </p>
      </LabSurface>

      <LabSurface label="Sampling and voting" className="rm-vote-card">
        <SurfaceHeading
          kicker={`n = ${samples} samples · each right with p = ${accuracy.toFixed(2)} · wrong answers: ${spread.label.toLowerCase()}`}
          title="The other test-time dial: sample several traces, then pick one"
        />
        <div className="rm-vote-controls">
          <RangeControl
            label="Samples"
            min={1}
            max={MAX_SAMPLES}
            step={1}
            value={samples}
            format={(value) => `${value} per question`}
            onChange={(value) => setState({ samples: value })}
          />
          <RangeControl
            label="Single-sample accuracy"
            min={0.05}
            max={0.95}
            step={0.05}
            value={accuracy}
            format={(value) => `p = ${value.toFixed(2)}`}
            onChange={(value) => {
              setState({ accuracy: value });
              narrate(`Each sample is right with probability ${value.toFixed(2)}.`);
            }}
          />
          <SegmentedControl
            label="Wrong answers"
            value={spreadKey}
            options={Object.entries(SPREADS).map(([value, entry]) => ({ value, label: entry.label }))}
            onChange={(value) => setState({ spread: value })}
          />
        </div>
        <VotePlot curves={curves} samples={samples} />
        <div className="rm-legend" aria-hidden="true">
          <span className="is-verifier">verifier picks · 1 − (1 − p)ⁿ</span>
          <span className="is-majority">majority vote</span>
          <span className="is-single">one sample · p</span>
        </div>
        <CountBars samples={samples} accuracy={accuracy} wrong={spread.wrong} />
        <div className="metric-row">
          <Metric label="One sample" value={pct(current.single)} />
          <Metric label="Majority vote" value={pct(current.majority)} tone={current.majority < accuracy ? "loss" : "gradient"} />
          <Metric label="Verifier picks" value={pct(current.verifier)} tone="forward" />
          <Metric label="Decode cost" value={`${samples}× tokens`} />
        </div>
        <p className="lab-note">
          Both curves are exact binomial arithmetic, not a simulation, and p is your assumption rather than a measured
          model. A perfect checker (unit tests, a proof checker, an exact-match answer key) succeeds if any sample is
          right, which is pass@n. Voting needs no checker but amplifies whatever answer is most common: when any single wrong
          answer is more likely than the right one, adding samples makes the vote worse.
        </p>
      </LabSurface>
    </div>
  );
}
