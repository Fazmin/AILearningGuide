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
  DEPLOYMENTS,
  GROUP_A,
  GROUP_B,
  HOLDOUT_SHARE_B,
  PARAMETER_OPTIONS,
  accuracyOf,
  breakEvenDays,
  equalOpportunityGap,
  falsePositiveGap,
  falsePositiveRate,
  truePositiveRate,
  megawattHours,
  mixOutcome,
  servingFlopsPerDay,
  trainingFlops,
  type Measured,
} from "./impact";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

const MEASURED_TEXT: Record<Measured, string> = {
  yes: "a benchmark number exists for this",
  partly: "only if the benchmark is split by group, matches who will use it, or is built to test this case",
  no: "no leaderboard measures this",
};

const formatParams = (value: number) => (value >= 1e9 ? `${Math.round(value / 1e9)}B` : `${Math.round(value / 1e6)}M`);
const formatTokens = (value: number) =>
  value >= 1e12
    ? `${+(value / 1e12).toFixed(1)}T`
    : value >= 1e9
      ? `${+(value / 1e9).toFixed(1)}B`
      : `${+(value / 1e6).toFixed(1)}M`;
const formatFlops = (value: number) => {
  const exponent = Math.floor(Math.log10(value));
  return `${(value / 10 ** exponent).toFixed(1)}×10${String(exponent).replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)])}`;
};
const formatDays = (days: number) =>
  days >= 730
    ? `${(days / 365).toFixed(1)} years`
    : days >= 1
      ? `${Math.round(days)} days`
      : `${(days * 24).toFixed(1)} hours`;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const deployment =
    DEPLOYMENTS.find((item) => item.value === asString(state, "deployment", "school")) ?? DEPLOYMENTS[0];
  const claimIndex = clamp(Math.round(asNumber(state, "claim", 0)), 0, deployment.claims.length - 1);
  const claim = deployment.claims[claimIndex];
  const forCount = deployment.claims.filter((item) => item.side === "For").length;
  const againstCount = deployment.claims.filter((item) => item.side === "Against").length;
  const measuredYes = deployment.claims.filter((item) => item.measured === "yes").length;
  const measuredPartly = deployment.claims.filter((item) => item.measured === "partly").length;

  const shareB = clamp(asNumber(state, "shareB", 0.3), 0, 0.6);
  const holdout = mixOutcome(HOLDOUT_SHARE_B);
  const tomorrow = mixOutcome(shareB);

  const paramIndex = clamp(Math.round(asNumber(state, "paramIndex", 2)), 0, PARAMETER_OPTIONS.length - 1);
  const parameters = PARAMETER_OPTIONS[paramIndex];
  const trainingTokens = clamp(asNumber(state, "trainTokens", 2), 0.5, 15) * 1e12;
  const servedLog = clamp(asNumber(state, "servedLog", 10), 6, 12);
  const tokensPerDay = 10 ** servedLog;
  const train = trainingFlops(parameters, trainingTokens);
  const servePerDay = servingFlopsPerDay(parameters, tokensPerDay);
  const breakEven = breakEvenDays(trainingTokens, tokensPerDay);
  const horizon = clamp(breakEven * 2, 10, 3650);
  const unit = 10 ** Math.floor(Math.log10(train));
  const chartDays = Array.from({ length: 41 }, (_, index) => (horizon * index) / 40);

  return (
    <div className="tg-lab tg-lab--hero ethics-lab a08-lab">
      <LabSurface label="Deployment" className="ethics-deployment-card">
        <SurfaceHeading
          kicker="No single right score"
          title="Pick a place the system would land"
          aside={<span className="tg-badge">Score: none</span>}
        />
        <div className="deployment-control">
          <SegmentedControl
            label="Deployment"
            value={deployment.value}
            options={DEPLOYMENTS.map((item) => ({ value: item.value, label: item.label }))}
            onChange={(value) => {
              setState({ deployment: value, claim: 0 });
              narrate(`Deployment ${DEPLOYMENTS.find((item) => item.value === value)?.label}.`);
            }}
          />
        </div>
        <div className="ethics-ledger">
          {(["For", "Against"] as const).map((side) => (
            <div key={side} className={`ethics-ledger__col is-${side.toLowerCase()}`}>
              <h4>{side}</h4>
              <ul>
                {deployment.claims.map((item, index) =>
                  item.side === side ? (
                    <li key={index}>
                      <button
                        type="button"
                        aria-pressed={index === claimIndex}
                        onClick={() => setState({ claim: index })}
                      >
                        <b>{index + 1}</b>
                        <span>{item.theme}</span>
                        <small>borne by {item.bearer}</small>
                        <em className={`is-${item.measured}`}>
                          {item.measured === "yes"
                            ? "on a leaderboard"
                            : item.measured === "partly"
                              ? "partly measurable"
                              : "not on a leaderboard"}
                        </em>
                      </button>
                    </li>
                  ) : null,
                )}
              </ul>
            </div>
          ))}
        </div>
        <div className="metric-row">
          <Metric label="For claims" value={`${forCount}`} tone="forward" />
          <Metric label="Against claims" value={`${againstCount}`} tone="loss" />
          <Metric
            label="A benchmark speaks to"
            value={`${measuredYes} of ${deployment.claims.length}, ${measuredPartly} partly`}
          />
          <Metric label="Score" value="none" />
        </div>
        <p className="lab-note">
          The counts are how many authored claims exist, not votes, and no weighting turns them into a score. The tags
          say who bears each claim and whether any benchmark number could speak to it.
        </p>
      </LabSurface>

      <LabSurface label="Tension explorer" className="ethics-tension-card">
        <SurfaceHeading
          kicker={`${claim.side} · ${claimIndex + 1} of ${deployment.claims.length}`}
          title="Competing claims, left unresolved"
        />
        <SegmentedControl
          label="Claim"
          value={`${claimIndex}`}
          options={deployment.claims.map((item, index) => ({ value: `${index}`, label: `${index + 1} ${item.side}` }))}
          onChange={(value) => setState({ claim: Number(value) })}
        />
        <blockquote className={`ethics-claim is-${claim.side.toLowerCase()}`}>{claim.text}</blockquote>
        <dl className="ethics-tags">
          <div>
            <dt>Theme</dt>
            <dd>{claim.theme}</dd>
          </div>
          <div>
            <dt>Who bears it</dt>
            <dd>{claim.bearer}</dd>
          </div>
          <div>
            <dt>Can a benchmark measure it?</dt>
            <dd className={`is-${claim.measured}`}>
              {claim.measured}: {MEASURED_TEXT[claim.measured]}
            </dd>
          </div>
        </dl>
      </LabSurface>

      <LabSurface label="Who the benchmark represents" className="ethics-bench-card">
        <SurfaceHeading
          kicker="Clinic intake · synthetic triage model · per 1,000 patients"
          title="The same model, a different population"
        />
        <div className="ethics-share-control">
          <RangeControl
            label="Group B share of patients"
            min={0}
            max={0.6}
            step={0.05}
            value={shareB}
            format={(value) => `${Math.round(value * 100)}%`}
            onChange={(value) => setState({ shareB: Number(value.toFixed(2)) })}
          />
        </div>
        <MixBars holdout={holdout} tomorrow={tomorrow} shareB={shareB} />
        <div className="metric-row">
          <Metric label={`Accuracy, holdout (${Math.round(HOLDOUT_SHARE_B * 100)}% B)`} value={pct(holdout.accuracy)} />
          <Metric label={`Accuracy at ${Math.round(shareB * 100)}% B`} value={pct(tomorrow.accuracy)} />
          <Metric
            label="Missed urgent per 1,000"
            value={`${Math.round(holdout.misses)} → ${Math.round(tomorrow.misses)}`}
            tone="loss"
          />
        </div>
        <div className="metric-row">
          <Metric
            label="True-positive rate, A / B"
            value={`${Math.round(truePositiveRate(GROUP_A) * 100)}% / ${Math.round(truePositiveRate(GROUP_B) * 100)}%`}
          />
          <Metric label="Equal-opportunity gap" value={`${Math.round(equalOpportunityGap() * 100)} points`} tone="loss" />
          <Metric
            label="False-positive rate, A / B"
            value={`${Math.round(falsePositiveRate(GROUP_A) * 100)}% / ${Math.round(falsePositiveRate(GROUP_B) * 100)}%`}
          />
          <Metric label="False-positive gap" value={`${Math.round(falsePositiveGap() * 100)} points`} />
        </div>
        <p className="lab-note">
          Group A: {pct(accuracyOf(GROUP_A))} accurate, misses {Math.round((1 - GROUP_A.sensitivity) * 100)}% of urgent
          cases. Group B: {pct(accuracyOf(GROUP_B))} accurate, misses {Math.round((1 - GROUP_B.sensitivity) * 100)}%.
          Accuracy is a weighted average of the two, so it barely moves; the misses concentrate in group B, which is{" "}
          {Math.round(shareB * 100)}% of patients and {Math.round(tomorrow.shareOfMissesB * 100)}% of missed urgent
          cases. Equal opportunity asks for equal true-positive rates; equalized odds asks for equal false-positive
          rates as well. The two gaps are properties of the model, so they do not move with the share.
        </p>
      </LabSurface>

      <LabSurface label="Training versus serving" className="ethics-energy-card">
        <SurfaceHeading
          kicker={`${formatParams(parameters)} parameters · ${formatTokens(trainingTokens)} training tokens · ${formatTokens(tokensPerDay)} tokens served a day`}
          title="Training is paid once; serving is paid every day"
        />
        <div className="ethics-energy-layout">
          <div>
            <RangeControl
              label="Parameters"
              min={0}
              max={PARAMETER_OPTIONS.length - 1}
              step={1}
              value={paramIndex}
              format={(value) => formatParams(PARAMETER_OPTIONS[Math.round(value)])}
              onChange={(value) => setState({ paramIndex: value })}
            />
            <RangeControl
              label="Training tokens"
              min={0.5}
              max={15}
              step={0.5}
              value={trainingTokens / 1e12}
              format={(value) => `${value}T`}
              onChange={(value) => setState({ trainTokens: value })}
            />
            <div className="ethics-serve-control">
              <RangeControl
                label="Tokens served per day"
                min={6}
                max={12}
                step={0.25}
                value={servedLog}
                format={(value) => formatTokens(10 ** value)}
                onChange={(value) => setState({ servedLog: value })}
              />
            </div>
            <div className="metric-row">
              <Metric label="Training, once" value={`${formatFlops(train)} FLOP`} />
              <Metric label="Serving, per day" value={`${formatFlops(servePerDay)} FLOP`} />
              <Metric label="Days to break even" value={formatDays(breakEven)} tone="loss" />
            </div>
            <p className="lab-note">
              Training ≈ 6·N·D. Serving ≈ 2·N per token. They are equal after 3·D ÷ tokens per day, so model size
              cancels out. At 40% of an A100's peak and 400 W, this training run is about{" "}
              {Math.round(megawattHours(train)).toLocaleString("en-US")} MWh of GPU power; serving at the same
              efficiency is {megawattHours(servePerDay).toFixed(2)} MWh a day, and real decoding usually runs less
              efficiently than that.
            </p>
          </div>
          <LineChart
            label="Cumulative compute of serving against one training run"
            xLabel={`days of serving, 0 → ${Math.round(horizon).toLocaleString("en-US")}`}
            yLabel={`cumulative FLOP, ×${formatFlops(unit).replace("1.0×", "")}`}
            xDomain={[0, horizon]}
            yDomain={[0, (Math.max(train, servePerDay * horizon) / unit) * 1.05]}
            marker={breakEven <= horizon ? { x: breakEven, label: "break-even" } : undefined}
            series={[
              {
                id: "train",
                name: "one training run",
                tone: "attention",
                dash: "dashed",
                points: [
                  { x: 0, y: train / unit },
                  { x: horizon, y: train / unit },
                ],
                format: (value) => value.toFixed(1),
              },
              {
                id: "serve",
                name: "serving, cumulative",
                tone: "loss",
                dash: "solid",
                points: chartDays.map((day) => ({ x: day, y: (servePerDay * day) / unit })),
                format: (value) => value.toFixed(1),
              },
            ]}
            footnote={
              breakEven <= horizon
                ? `The guide marks day ${Math.round(breakEven).toLocaleString("en-US")}, where serving has used as much compute as training.`
                : "Serving does not reach one training run's compute within ten years at this volume."
            }
          />
        </div>
      </LabSurface>
    </div>
  );
}

function MixBars({
  holdout,
  tomorrow,
  shareB,
}: {
  holdout: ReturnType<typeof mixOutcome>;
  tomorrow: ReturnType<typeof mixOutcome>;
  shareB: number;
}) {
  const maxMisses = Math.max(holdout.misses, tomorrow.misses, 60);
  const groups = [
    {
      title: "Patients per 1,000",
      scale: 1000,
      rows: [
        { label: `Holdout (${Math.round(HOLDOUT_SHARE_B * 100)}% B)`, a: holdout.patientsA, b: holdout.patientsB },
        { label: `At ${Math.round(shareB * 100)}% B`, a: tomorrow.patientsA, b: tomorrow.patientsB },
      ],
    },
    {
      title: "Missed urgent cases per 1,000",
      scale: maxMisses,
      rows: [
        { label: `Holdout (${Math.round(HOLDOUT_SHARE_B * 100)}% B)`, a: holdout.missesA, b: holdout.missesB },
        { label: `At ${Math.round(shareB * 100)}% B`, a: tomorrow.missesA, b: tomorrow.missesB },
      ],
    },
  ];
  return (
    <div
      className="ethics-mix"
      role="img"
      aria-label={`Per 1,000 patients. Holdout: ${Math.round(holdout.misses)} missed urgent cases. At ${Math.round(shareB * 100)}% group B: ${Math.round(tomorrow.misses)}, of which ${Math.round(tomorrow.missesB)} are in group B.`}
    >
      <p className="ethics-mix__legend" aria-hidden="true">
        <span className="ethics-mix__key ethics-mix__key--a" /> group A, solid
        <span className="ethics-mix__key ethics-mix__key--b" /> group B, hatched
      </p>
      {groups.map((group) => (
        <div key={group.title} className="ethics-mix__group">
          <h4>{group.title}</h4>
          {group.rows.map((row) => (
            <div key={row.label} className="ethics-mix__row">
              <span className="ethics-mix__label">{row.label}</span>
              <span className="ethics-mix__track">
                <i className="ethics-mix__a" style={{ width: `${(row.a / group.scale) * 100}%` }} />
                <i className="ethics-mix__b" style={{ width: `${(row.b / group.scale) * 100}%` }} />
              </span>
              <span className="ethics-mix__value">
                {Math.round(row.a + row.b)}
                <small>
                  {" "}
                  A {Math.round(row.a)} · B {Math.round(row.b)}
                </small>
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
