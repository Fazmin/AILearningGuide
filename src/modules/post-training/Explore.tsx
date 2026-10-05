import { useMemo } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  StageFlow,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  BETA_MAX,
  BETA_MIN,
  CANDIDATES,
  DEFAULT_LABELS,
  FEATURE_NAMES,
  MAX_SFT_STEPS,
  PAIRS,
  PROMPT,
  RM_L2,
  RM_STEPS,
  SFT_DEMOS,
  betaSweep,
  comparisonsFrom,
  expectation,
  encodeLabels,
  fitRewardModel,
  goldAgreement,
  goldLabels,
  indexOf,
  klDivergence,
  opensWithPraise,
  parseLabels,
  rewardOf,
  sftLogits,
  sigmoid,
  softmax,
  tiltedPolicy,
  wordCount,
} from "./prefs";

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pct = (value: number) => `${Math.round(value * 100)}%`;
const fmt = (value: number, digits = 2) => value.toFixed(digits);
const formatBeta = (beta: number) => (beta >= 1 ? beta.toFixed(1) : beta >= 0.1 ? beta.toFixed(2) : beta.toFixed(3));

const STAGES = [
  { id: "pretrain", name: "Pretrained" },
  { id: "sft", name: "SFT" },
  { id: "prefs", name: "Preferences" },
  { id: "rm", name: "Reward model" },
  { id: "rl", name: "Policy + KL" },
];
const STAGE_FOR_STEP = [1, 3, 4, 2];

export default function Explore({ state, setState, currentStep, narrate }: ModuleContext) {
  const sftSteps = clamp(Math.round(asNumber(state, "sftSteps", 3)), 0, MAX_SFT_STEPS);
  const beta = clamp(asNumber(state, "beta", 2), BETA_MIN, BETA_MAX);
  const labels = parseLabels(state.labels);

  const base = useMemo(() => softmax(CANDIDATES.map((candidate) => candidate.baseLogit)), []);
  const sft = useMemo(() => sftLogits(sftSteps), [sftSteps]);
  const reference = useMemo(() => softmax(sft.logits), [sft]);
  const labelKey = encodeLabels(labels);
  const comparisons = useMemo(() => comparisonsFrom(parseLabels(labelKey)), [labelKey]);
  const fit = useMemo(() => fitRewardModel(comparisons), [comparisons]);
  const rewards = useMemo(() => CANDIDATES.map((_, index) => rewardOf(fit.weights, index)), [fit]);
  const policy = tiltedPolicy(reference, rewards, beta);
  const gold = CANDIDATES.map((candidate) => candidate.gold);
  const kl = klDivergence(policy, reference);
  const expectedReward = expectation(policy, rewards);
  const expectedGold = expectation(policy, gold);
  const referenceGold = expectation(reference, gold);
  const sweep = useMemo(() => betaSweep(reference, rewards), [reference, rewards]);
  const agreement = goldAgreement(rewards);
  const trainCorrect = comparisons.filter((c) => rewards[c.winner] > rewards[c.loser]).length;
  const rMin = Math.min(...rewards);
  const rMax = Math.max(...rewards);
  const rescale = (value: number) => (rMax - rMin < 1e-9 ? 0.5 : (value - rMin) / (rMax - rMin));
  const peak = sweep.reduce((best, point) => (point.gold > best.gold ? point : best), {
    beta: Infinity,
    kl: 0,
    reward: 0,
    gold: referenceGold,
  });
  const top = policy.indexOf(Math.max(...policy));
  const labeled = comparisons.length;
  const stage = STAGE_FOR_STEP[clamp(currentStep, 0, STAGE_FOR_STEP.length - 1)];

  const setLabel = (index: number, choice: "a" | "b") => {
    const next = [...labels];
    next[index] = next[index] === choice ? "none" : choice;
    setState({ labels: encodeLabels(next) });
    const pair = PAIRS[index];
    narrate(
      next[index] === "none"
        ? `Pair ${index + 1} cleared.`
        : `Pair ${index + 1}: ${next[index] === "a" ? pair.a : pair.b} preferred over ${next[index] === "a" ? pair.b : pair.a}.`,
    );
  };

  const stageDetails = [
    `${CANDIDATES[base.indexOf(Math.max(...base))].id} gets ${pct(Math.max(...base))}`,
    `${sftSteps} step${sftSteps === 1 ? "" : "s"} on ${SFT_DEMOS.join(", ")} · loss ${fmt(sft.losses[sft.losses.length - 1])}`,
    `${labeled} of ${PAIRS.length} pairs labeled`,
    `w = [${fit.weights.map((value) => fmt(value)).join(", ")}]`,
    `β ${formatBeta(beta)} · KL ${fmt(kl)} nats`,
  ];

  return (
    <div className="tg-lab tg-lab--hero pt-lab a08-lab">
      <LabSurface label="Post-training pipeline" className="pt-pipeline-card">
        <SurfaceHeading
          kicker={`prompt: “${PROMPT}” · six whole responses are the only actions`}
          title="Where the probability goes at each stage"
          aside={<span className="tg-badge">{`gold ${fmt(expectedGold)} after tuning`}</span>}
        />
        <StageFlow
          label="Post-training pipeline"
          stages={STAGES.map((item, index) => ({ ...item, detail: stageDetails[index] }))}
          current={stage}
        />
        <div className="pt-sft-control">
          <RangeControl
            label="SFT steps"
            min={0}
            max={MAX_SFT_STEPS}
            step={1}
            value={sftSteps}
            onChange={(value) => setState({ sftSteps: value })}
          />
        </div>
        <div className="pt-candidates" role="list" aria-label="Six candidate responses with their probabilities">
          {CANDIDATES.map((candidate, index) => {
            const words = wordCount(candidate.text);
            const praise = opensWithPraise(candidate.text);
            const demo = (SFT_DEMOS as readonly string[]).includes(candidate.id);
            return (
              <article
                key={candidate.id}
                role="listitem"
                className={`pt-candidate ${index === top ? "is-top" : ""}`.trim()}
                aria-label={`Response ${candidate.id}: base ${pct(base[index])}, after SFT ${pct(reference[index])}, after tuning ${pct(policy[index])}, reward ${fmt(rewards[index])}, gold ${fmt(candidate.gold)}`}
              >
                <header>
                  <strong>{candidate.id}</strong>
                  <span>{candidate.tag}</span>
                  {demo && <em>SFT demo</em>}
                </header>
                <p>{candidate.text}</p>
                <div className="pt-chips" aria-hidden="true">
                  <span>{words} words</span>
                  <span>{candidate.mechanism ? "names mechanism" : "no mechanism"}</span>
                  <span>{praise ? "opens with praise" : "no praise"}</span>
                </div>
                <div className="pt-dist" aria-hidden="true">
                  {[
                    { name: "base", value: base[index], tone: "muted" },
                    { name: "SFT", value: reference[index], tone: "forward" },
                    { name: "tuned", value: policy[index], tone: "attention" },
                  ].map((row) => (
                    <div key={row.name} className={`pt-dist__row pt-dist__row--${row.tone}`}>
                      <span>{row.name}</span>
                      <i>
                        <b style={{ width: `${Math.max(0.5, row.value * 100)}%` }} />
                      </i>
                      <strong>{pct(row.value)}</strong>
                    </div>
                  ))}
                </div>
                <footer>
                  <span>
                    r <b>{fmt(rewards[index])}</b>
                  </span>
                  <span>
                    gold <b>{fmt(candidate.gold)}</b>
                  </span>
                </footer>
              </article>
            );
          })}
        </div>
        <p className="lab-note">
          Base probabilities are authored logits for a raw text predictor, which favors continuing the page with more
          exam questions. SFT takes real cross-entropy steps toward the two demonstrations. “Tuned” is the exact optimum
          for the reward model and β below. Gold is an authored careful-rater score used only to grade the result.
        </p>
      </LabSurface>

      <LabSurface label="Preference pairs" className="pt-pairs-card">
        <SurfaceHeading kicker={`${labeled} labeled · click again to clear`} title="Which response is better?" />
        <ol className="pt-pairs">
          {PAIRS.map((pair, index) => {
            const a = indexOf(pair.a);
            const b = indexOf(pair.b);
            const label = labels[index];
            const pA = sigmoid(rewards[a] - rewards[b]);
            return (
              <li
                key={index}
                className={`pt-pair pt-pair-${index + 1} ${label === "none" ? "is-unlabelled" : ""}`.trim()}
              >
                <span className="pt-pair__name">Pair {index + 1}</span>
                {(["a", "b"] as const).map((side) => {
                  const candidate = CANDIDATES[side === "a" ? a : b];
                  return (
                    <button
                      key={side}
                      type="button"
                      aria-pressed={label === side}
                      aria-label={`Pair ${index + 1}: prefer ${candidate.id}, ${candidate.tag}`}
                      onClick={() => setLabel(index, side)}
                    >
                      <strong>{candidate.id}</strong> {candidate.tag}
                    </button>
                  );
                })}
                <span className="pt-pair__prob">
                  {label === "none" ? "unlabelled · " : ""}P({pair.a} ≻ {pair.b}) = {fmt(pA)}
                </span>
              </li>
            );
          })}
        </ol>
        <div className="pt-pair-actions">
          <button
            type="button"
            className="quiet-action"
            onClick={() => setState({ labels: encodeLabels(DEFAULT_LABELS) })}
          >
            Restore default labels
          </button>
          <button
            type="button"
            className="quiet-action"
            onClick={() => setState({ labels: encodeLabels(goldLabels()) })}
          >
            Label like the gold rater
          </button>
          <button
            type="button"
            className="quiet-action"
            onClick={() => setState({ labels: encodeLabels(PAIRS.map(() => "none")) })}
          >
            Clear all
          </button>
        </div>
        <p className="lab-note">
          P(A ≻ B) = σ(r_A − r_B) from the fitted reward model. In every default label the preferred response is
          also the longer one. Pair 5, the only pair where the answer that names the mechanism is the shorter one, starts
          unlabelled.
        </p>
      </LabSurface>

      <LabSurface label="Reward model fit" className="pt-rm-card">
        <SurfaceHeading
          kicker={`Bradley–Terry · ${RM_STEPS} gradient steps · L2 ${RM_L2}`}
          title="What the scorer learned from your labels"
        />
        <BarList
          label="Reward model weights"
          max={Math.max(0.5, ...fit.weights.map((value) => Math.abs(value)))}
          items={FEATURE_NAMES.map((name, index) => ({
            id: name,
            label: name,
            value: fit.weights[index],
            display: `${fit.weights[index] >= 0 ? "+" : "−"}${Math.abs(fit.weights[index]).toFixed(3)}`,
            tone: fit.weights[index] >= 0 ? "forward" : "loss",
            detail: fit.weights[index] >= 0 ? "raises reward" : "lowers reward",
          }))}
        />
        <LineChart
          label="Bradley–Terry training loss"
          xLabel={`gradient step, 0 → ${RM_STEPS}`}
          yLabel="mean −log σ(r_win − r_lose)"
          xDomain={[0, RM_STEPS]}
          yDomain={[0, 0.75]}
          series={[
            {
              id: "loss",
              name: "training loss",
              tone: "loss",
              points: fit.history
                .filter((point) => point.step % 5 === 0)
                .map((point) => ({ x: point.step, y: point.loss })),
            },
          ]}
        />
        <div className="metric-row">
          <Metric label="Pairs used" value={`${labeled}`} />
          <Metric label="Labels fitted" value={labeled ? `${trainCorrect} of ${labeled}` : "—"} tone="forward" />
          <Metric
            label="Agrees with gold"
            value={`${agreement.agree} of ${agreement.total}`}
            tone={agreement.agree >= 12 ? "forward" : "loss"}
          />
        </div>
        <p className="lab-note">
          r(y) = w·φ(y) with φ = [words ÷ 10, names mechanism, opens with praise]. The scorer never reads the words
          themselves, only these three numbers. “Agrees with gold” checks all 15 pairs of the six responses, most of
          which you never labeled.
        </p>
      </LabSurface>

      <LabSurface label="Optimize against the reward model" className="pt-rl-card">
        <SurfaceHeading
          kicker={`maximize E[r] − β·KL(π‖π_SFT) · β = ${formatBeta(beta)}`}
          title="The proxy keeps rising; the gold score does not"
        />
        <div className="pt-rl-layout">
          <div>
            <div className="pt-beta-control">
              <RangeControl
                label="KL coefficient β"
                min={Math.log10(BETA_MIN)}
                max={Math.log10(BETA_MAX)}
                step={0.05}
                value={Math.log10(beta)}
                format={(value) => formatBeta(10 ** value)}
                onChange={(value) => setState({ beta: Number((10 ** value).toPrecision(3)) })}
              />
            </div>
            <div className="metric-row">
              <Metric label="Reward-model score" value={fmt(expectedReward)} tone="forward" />
              <Metric
                label="Gold score"
                value={fmt(expectedGold)}
                tone={expectedGold < referenceGold ? "loss" : "forward"}
              />
              <Metric label="KL from SFT" value={`${fmt(kl)} nats`} />
            </div>
            <FormulaWithValues
              label={`π(${CANDIDATES[top].id}) ∝ π_SFT(${CANDIDATES[top].id}) · exp(r / β)`}
              expression={`${fmt(reference[top], 3)} · exp(${fmt(rewards[top])} / ${formatBeta(beta)}), then normalize over all six`}
              result={pct(policy[top])}
              tone="attention"
              detail={`Smaller β lets the reward term dominate the KL leash. Gold peaks at ${fmt(peak.gold)} near β ${formatBeta(peak.beta)} (KL ${fmt(peak.kl)}) for these labels, then falls toward the reward model's favorite.`}
            />
          </div>
          <LineChart
            label="Reward-model score and gold score against KL from the SFT model"
            xLabel={`KL(π‖π_SFT) in nats, 0 → ${fmt(sweep[sweep.length - 1].kl)}`}
            yLabel="score, 0 = worst response, 1 = best"
            yDomain={[0, 1]}
            marker={{ x: kl, label: "β" }}
            series={[
              {
                id: "rm",
                name: "reward-model score (rescaled)",
                tone: "forward",
                dash: "solid",
                points: [
                  { x: 0, y: rescale(expectation(reference, rewards)) },
                  ...sweep.map((point) => ({ x: point.kl, y: rescale(point.reward) })),
                ],
              },
              {
                id: "gold",
                name: "gold score",
                tone: "loss",
                dash: "dashed",
                points: [{ x: 0, y: referenceGold }, ...sweep.map((point) => ({ x: point.kl, y: point.gold }))],
              },
            ]}
            footnote={`Each point is the exact tilted policy for one β from ${BETA_MAX} down to ${BETA_MIN}. The reward-model line is rescaled so its worst response is 0 and its best is 1; legend values are the far-right end.`}
          />
        </div>
      </LabSurface>
    </div>
  );
}
