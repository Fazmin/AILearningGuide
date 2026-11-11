import type { CSSProperties } from "react";
import {
  DEPLOYMENTS,
  GROUP_A,
  GROUP_B,
  HOLDOUT_SHARE_B,
  accuracyOf,
  equalOpportunityGap,
  falsePositiveGap,
  falsePositiveRate,
  truePositiveRate,
  type Deployment,
  type Measured,
  type mixOutcome,
} from "./impact";

type Outcome = ReturnType<typeof mixOutcome>;

export const MEASURED_TAG: Record<Measured, string> = {
  yes: "on a leaderboard",
  partly: "partly measurable",
  no: "not on a leaderboard",
};

const whole = (value: number) => Math.round(value);
const percent = (value: number) => `${Math.round(value * 100)}%`;

/** A small drawn mark for the three measurability levels: filled, half filled, hollow. */
export function MeasureMark({ measured }: { measured: Measured }) {
  return <i className={`ethics-mark is-${measured}`} aria-hidden="true" />;
}

/**
 * Every deployment's five claims as one row of marks, so the pattern "a leaderboard reaches at most one claim in
 * five" can be read without switching deployments.
 */
export function CoverageMap({ current }: { current: Deployment["value"] }) {
  return (
    <figure className="ethics-cover">
      <figcaption>
        <strong>What a benchmark number can reach</strong>
        <span>one mark per claim, in claim order · all four deployments</span>
      </figcaption>
      <div
        className="ethics-cover__grid"
        role="img"
        aria-label={DEPLOYMENTS.map((deployment) => {
          const yes = deployment.claims.filter((claim) => claim.measured === "yes").length;
          const partly = deployment.claims.filter((claim) => claim.measured === "partly").length;
          return `${deployment.label}: ${yes} of ${deployment.claims.length} claims on a leaderboard, ${partly} partly measurable`;
        }).join(". ")}
      >
        {DEPLOYMENTS.map((deployment) => (
          <div
            key={deployment.value}
            className={`ethics-cover__row${deployment.value === current ? " is-current" : ""}`}
            aria-hidden="true"
          >
            <span className="ethics-cover__name">{deployment.label}</span>
            {deployment.claims.map((claim, index) => (
              <span key={index} className={`ethics-cover__cell is-${claim.measured} is-${claim.side.toLowerCase()}`}>
                <b>{index + 1}</b>
                <MeasureMark measured={claim.measured} />
                <small>{claim.side}</small>
              </span>
            ))}
          </div>
        ))}
      </div>
      <p className="ethics-cover__key" aria-hidden="true">
        <span>
          <MeasureMark measured="yes" /> on a leaderboard
        </span>
        <span>
          <MeasureMark measured="partly" /> partly measurable
        </span>
        <span>
          <MeasureMark measured="no" /> not on a leaderboard
        </span>
      </p>
    </figure>
  );
}

/** The people this deployment's claims land on, each with the numbered claims they bear. */
export function BearerMap({ deployment, claimIndex }: { deployment: Deployment; claimIndex: number }) {
  // "reporter" and "reporters" are one party; key on the singular and show the first spelling seen.
  const parties = new Map<string, { name: string; claims: number[] }>();
  deployment.claims.forEach((claim, index) => {
    for (const raw of claim.bearer.split(",")) {
      const name = raw.trim();
      const key = name.replace(/s$/, "");
      const party = parties.get(key) ?? { name, claims: [] };
      party.claims.push(index);
      parties.set(key, party);
    }
  });
  const rows = [...parties.values()];
  return (
    <figure className="ethics-bearers">
      <figcaption>
        <strong>Where the five claims land</strong>
        <span>each row is a party; each badge is a claim it bears</span>
      </figcaption>
      <ul
        aria-label={`Who bears each claim in ${deployment.label}: ${rows
          .map((row) => `${row.name}, ${row.claims.map((index) => `claim ${index + 1} ${deployment.claims[index].side}`).join(" and ")}`)
          .join("; ")}`}
      >
        {rows.map((row) => (
          <li key={row.name} className={row.claims.includes(claimIndex) ? "is-current" : ""} aria-hidden="true">
            <span>{row.name}</span>
            <span className="ethics-bearers__claims">
              {row.claims.map((index) => (
                <b
                  key={index}
                  className={`is-${deployment.claims[index].side.toLowerCase()}${index === claimIndex ? " is-current" : ""}`}
                >
                  {index + 1} {deployment.claims[index].side}
                </b>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

/** no · partly · yes as three steps, with the claim's level filled in. */
export function MeasureScale({ measured }: { measured: Measured }) {
  return (
    <span className="ethics-scale" aria-hidden="true">
      {(["no", "partly", "yes"] as const).map((level) => (
        <i key={level} className={`is-${level}${level === measured ? " is-on" : ""}`}>
          <MeasureMark measured={level} />
          {level}
        </i>
      ))}
    </span>
  );
}

/* ------------------------------------------------ the triage model, drawn */

function UrgentPanel({ title, note, outcome, live }: { title: string; note: string; outcome: Outcome; live?: boolean }) {
  const groups = [
    { id: "a", rates: GROUP_A, patients: outcome.patientsA, misses: outcome.missesA },
    { id: "b", rates: GROUP_B, patients: outcome.patientsB, misses: outcome.missesB },
  ];
  const total = outcome.patientsA + outcome.patientsB;
  return (
    <div className={`ethics-urgent__panel${live ? " is-live" : ""}`}>
      <div className="ethics-urgent__head">
        <span>
          <strong>{title}</strong>
          <small>{note}</small>
        </span>
        <b>
          {whole(outcome.misses)}
          <small> missed</small>
        </b>
      </div>
      <div
        className="ethics-urgent__plot"
        role="img"
        aria-label={`${title}. Of every 1,000 patients, ${whole(outcome.patientsA)} are in group A and ${whole(
          outcome.patientsB,
        )} in group B. The model misses ${whole(outcome.missesA)} urgent cases in group A and ${whole(
          outcome.missesB,
        )} in group B, ${whole(outcome.misses)} in all.`}
      >
        {groups.map((group) => (
          <div
            key={group.id}
            className={`ethics-urgent__col is-${group.id}`}
            style={{ width: `${(group.patients / total) * 100}%` }}
            aria-hidden="true"
          >
            <span className="ethics-urgent__missed" style={{ height: `${(1 - group.rates.sensitivity) * 100}%` }}>
              <em>{percent(1 - group.rates.sensitivity)} missed</em>
            </span>
            <span className="ethics-urgent__caught">
              <em>
                {percent(group.rates.sensitivity)} flagged
              </em>
            </span>
          </div>
        ))}
      </div>
      <div className="ethics-urgent__counts" aria-hidden="true">
        {groups.map((group) => (
          <span key={group.id} className={`is-${group.id}`}>
            {group.rates.name}: {whole(group.patients * group.rates.prevalence)} urgent, <b>{whole(group.misses)} missed</b>
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * The urgent patients in every 1,000, as one box per population. Column width is the group's share of patients and
 * the full-colour top slice is the share of that group's urgent cases the model misses, so that area is the misses.
 */
export function UrgentPanels({ holdout, tomorrow, shareB }: { holdout: Outcome; tomorrow: Outcome; shareB: number }) {
  return (
    <div className="ethics-urgent">
      <p className="ethics-urgent__lead">
        Each box is the <b>200 urgent patients</b> in every 1,000. A column&apos;s <b>width</b> is that group&apos;s share of
        patients. The <b>full-colour slice</b> on top is the urgent cases the model misses.
      </p>
      <div className="ethics-urgent__panels">
        <UrgentPanel title={`Holdout (${percent(HOLDOUT_SHARE_B)} B)`} note="the test the model passed" outcome={holdout} />
        <UrgentPanel title={`At ${percent(shareB)} B`} note="the clinic your slider describes" outcome={tomorrow} live />
      </div>
      <p className="ethics-urgent__call">
        Same model in both boxes, so the slice heights never change. Group B is {percent(shareB)} of patients and{" "}
        <b>{percent(tomorrow.shareOfMissesB)} of the missed urgent cases</b>.
      </p>
    </div>
  );
}

export function MixBars({ holdout, tomorrow, shareB }: { holdout: Outcome; tomorrow: Outcome; shareB: number }) {
  const maxMisses = Math.max(holdout.misses, tomorrow.misses, 60);
  const holdoutLabel = `Holdout (${percent(HOLDOUT_SHARE_B)} B)`;
  const liveLabel = `At ${percent(shareB)} B`;
  const correct = (outcome: Outcome) => ({
    a: (outcome.patientsA / 10) * accuracyOf(GROUP_A),
    b: (outcome.patientsB / 10) * accuracyOf(GROUP_B),
  });
  const show = (value: number, digits: number) => (digits === 0 ? `${Math.round(value)}` : value.toFixed(digits));
  const groups = [
    {
      title: "Patients per 1,000",
      scale: 1000,
      digits: 0,
      unit: "",
      rows: [
        { label: holdoutLabel, a: holdout.patientsA, b: holdout.patientsB },
        { label: liveLabel, a: tomorrow.patientsA, b: tomorrow.patientsB },
      ],
    },
    {
      title: "Accuracy, correct calls per 100 patients",
      scale: 100,
      digits: 1,
      unit: "%",
      rows: [
        { label: holdoutLabel, ...correct(holdout) },
        { label: liveLabel, ...correct(tomorrow) },
      ],
    },
    {
      title: "Missed urgent cases per 1,000",
      scale: maxMisses,
      digits: 0,
      unit: "",
      rows: [
        { label: holdoutLabel, a: holdout.missesA, b: holdout.missesB },
        { label: liveLabel, a: tomorrow.missesA, b: tomorrow.missesB },
      ],
    },
  ];
  return (
    <div
      className="ethics-mix"
      role="img"
      aria-label={`Per 1,000 patients. Holdout: ${(holdout.accuracy * 100).toFixed(1)}% accurate, ${whole(
        holdout.misses,
      )} missed urgent cases. At ${percent(shareB)} group B: ${(tomorrow.accuracy * 100).toFixed(1)}% accurate, ${whole(
        tomorrow.misses,
      )} missed, of which ${whole(tomorrow.missesB)} are in group B.`}
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
                {show(row.a + row.b, group.digits)}
                {group.unit}
                <small>
                  {" "}
                  A {show(row.a, group.digits)} · B {show(row.b, group.digits)}
                </small>
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/** The two per-group rates on one 0 to 100% scale, with the gap between the groups drawn as a bar. */
export function RatesChart() {
  const rows = [
    {
      label: "True-positive rate, A / B",
      hint: "urgent cases the model flags · higher is better",
      a: truePositiveRate(GROUP_A),
      b: truePositiveRate(GROUP_B),
      gapLabel: "Equal-opportunity gap",
      gap: equalOpportunityGap(),
      wide: true,
    },
    {
      label: "False-positive rate, A / B",
      hint: "non-urgent cases flagged anyway · lower is better",
      a: falsePositiveRate(GROUP_A),
      b: falsePositiveRate(GROUP_B),
      gapLabel: "False-positive gap",
      gap: falsePositiveGap(),
      wide: false,
    },
  ];
  return (
    <div className="ethics-rates">
      <h4>
        The model&apos;s rates in each group <span>fixed · the slider does not move them</span>
      </h4>
      <div className="ethics-rates__rows">
      {rows.map((row) => {
        const low = Math.min(row.a, row.b);
        return (
          <div key={row.label} className={`ethics-rates__row${row.wide ? " is-wide" : ""}`}>
            <div className="ethics-rates__head">
              <span>
                {row.label}
                <small>{row.hint}</small>
              </span>
              <b>
                {percent(row.a)} / {percent(row.b)}
              </b>
            </div>
            <div className="ethics-rates__scale">
              <span aria-hidden="true">0%</span>
              <div
                className="ethics-rates__track"
                role="img"
                aria-label={`${row.label}: ${percent(row.a)} for group A, ${percent(row.b)} for group B, a gap of ${Math.round(
                  row.gap * 100,
                )} points.`}
              >
                <i className="ethics-rates__gap" style={{ left: `${low * 100}%`, width: `${row.gap * 100}%` }} />
                <i className="ethics-rates__dot is-a" style={{ left: `${row.a * 100}%` }}>
                  <em>A {percent(row.a)}</em>
                </i>
                <i className="ethics-rates__dot is-b" style={{ left: `${row.b * 100}%` }}>
                  <em>B {percent(row.b)}</em>
                </i>
              </div>
              <span aria-hidden="true">100%</span>
            </div>
            <p className="ethics-rates__gap-label">
              {row.gapLabel} <b>{Math.round(row.gap * 100)} points</b>
            </p>
          </div>
        );
      })}
      </div>
    </div>
  );
}

/* ------------------------------------------------ break-even on a calendar */

const CALENDAR_TICKS = [
  { days: 1, label: "1 day" },
  { days: 7, label: "1 week" },
  { days: 30, label: "1 month" },
  { days: 365, label: "1 year" },
  { days: 3650, label: "10 years · chart limit" },
  { days: 36500, label: "100 years" },
  { days: 365000, label: "1,000 years" },
];
const CALENDAR_LOW = Math.log10(0.6);
const CALENDAR_HIGH = Math.log10(365000 * 2.2);
const TEN_YEARS = 3650;

/**
 * The break-even day on a logarithmic calendar. The line chart rescales itself around the break-even day, so its
 * picture barely changes; this is the strip where moving a slider visibly moves the mark.
 */
export function BreakEvenScale({ days, label }: { days: number; label: string }) {
  const at = (value: number) =>
    Math.min(1, Math.max(0, (Math.log10(value) - CALENDAR_LOW) / (CALENDAR_HIGH - CALENDAR_LOW))) * 100;
  const off = days > 365000 * 2.2;
  const side = at(days) > 62 ? "is-left" : "is-right";
  return (
    <figure className="ethics-calendar">
      <figcaption>
        <strong>When serving has cost as much as training</strong>
        <span>time on a log scale, from 1 day to 1,000 years</span>
      </figcaption>
      <div
        className="ethics-calendar__track"
        role="img"
        aria-label={`Serving matches one training run after ${label}${
          days > TEN_YEARS ? ", which is past the ten years the chart covers" : ""
        }.`}
        style={{ "--at": `${at(days)}%`, "--limit": `${at(TEN_YEARS)}%` } as CSSProperties}
      >
        <i className="ethics-calendar__before">
          <em>training has cost more</em>
        </i>
        <i className="ethics-calendar__after">
          <em>serving has cost more</em>
        </i>
        <i className="ethics-calendar__late" />
        {CALENDAR_TICKS.map((tick) => (
          <span key={tick.days} className="ethics-calendar__tick" style={{ left: `${at(tick.days)}%` }}>
            {tick.label}
          </span>
        ))}
        <span className={`ethics-calendar__pin ${side}`}>
          <b>
            break-even{off ? " is off this scale" : ""} · {label}
          </b>
        </span>
      </div>
      <p className="ethics-calendar__note">
        Serving costs less than training up to the mark and more after it. More tokens served a day pull the mark left;
        more training tokens push it right; Parameters leaves it where it is.
      </p>
    </figure>
  );
}
