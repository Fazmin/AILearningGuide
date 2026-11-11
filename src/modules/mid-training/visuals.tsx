import type { CSSProperties } from "react";

/**
 * Pictures for the mid-training lab. Each one draws numbers the lab has already computed; nothing here
 * trains or scores anything itself.
 */

const percent = (share: number) => `${Math.round(share * 1000) / 10}%`;

export interface MixDomain {
  id: string;
  name: string;
  general: boolean;
}

/* -------------------------------------------------------------------------- */
/* What one batch is made of, before and after the late switch                 */
/* -------------------------------------------------------------------------- */

function MixBar({ domains, shares }: { domains: ReadonlyArray<MixDomain>; shares: ReadonlyArray<number> }) {
  return (
    <span className="mt-budget-bar">
      {domains.map((domain, index) =>
        shares[index] > 0 ? (
          <i
            key={domain.id}
            className={`mt-budget-bar__${domain.general ? "general" : "target"} mt-budget-bar__${domain.id}`}
            style={{ flexGrow: shares[index] }}
            title={`${domain.name}: ${percent(shares[index])}`}
          >
            {shares[index] >= 0.095 && <span>{domain.name}</span>}
            {shares[index] >= 0.05 && <b>{percent(shares[index])}</b>}
          </i>
        ) : null,
      )}
    </span>
  );
}

export function BudgetBars({
  domains,
  shares,
  late,
  lateNote,
  earlyNote,
  spoken,
}: {
  domains: ReadonlyArray<MixDomain>;
  shares: ReadonlyArray<number>;
  /** The shares of every late batch, or null when the mixture never switches. */
  late: ReadonlyArray<number> | null;
  earlyNote: string;
  lateNote: string;
  spoken: string;
}) {
  return (
    <div className="mt-budget" role="img" aria-label={spoken}>
      <span className="mt-budget__label">{earlyNote}</span>
      <MixBar domains={domains} shares={shares} />
      {late && (
        <>
          <span className="mt-budget__label mt-budget__label--late">{lateNote}</span>
          <MixBar domains={domains} shares={late} />
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Passes over each domain's training text                                     */
/* -------------------------------------------------------------------------- */

export function PassStrips({
  rows,
}: {
  rows: ReadonlyArray<{ id: string; name: string; general: boolean; epochs: number; seen: string; pairs: string }>;
}) {
  const most = Math.max(1, ...rows.map((row) => row.epochs));
  const every = most > 24 ? 5 : 1;
  return (
    <div
      className="mt-passes"
      role="img"
      aria-label={`Full passes over each domain's training pairs: ${rows.map((row) => `${row.name} ${row.epochs.toFixed(2)}`).join(", ")}.`}
    >
      <span className="mt-subhead">Passes over each domain&apos;s own text: pairs drawn ÷ training pairs</span>
      {rows.map((row) => (
        <div className="mt-passes__row" key={row.id}>
          <b>{row.name}</b>
          <span className="mt-passes__track" style={{ "--mt-pass": `${(every / most) * 100}%` } as CSSProperties}>
            <i
              className={row.general ? "is-general" : "is-target"}
              style={{ width: `${Math.max(row.epochs > 0 ? 0.6 : 0, (row.epochs / most) * 100)}%` }}
            />
          </span>
          <code>{row.epochs.toFixed(2)}×</code>
          <small>
            {row.seen} drawn ÷ {row.pairs} training pairs
          </small>
        </div>
      ))}
      <p className="mt-passes__key">
        Each block is {every === 1 ? "one full pass" : "five full passes"}. A domain with little text is read many times;
        one with a lot may not be read once.
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* A number in a table cell with a bar under it                                */
/* -------------------------------------------------------------------------- */

export function CellBar({ value, max, text }: { value: number | null; max: number; text: string }) {
  return (
    <span className="mt-cell">
      {text}
      <i className="mt-cell__track">
        {value !== null && <i style={{ width: `${Math.min(100, Math.max(0, (value / max) * 100))}%` }} />}
      </i>
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Every held-out fact as one mark                                             */
/* -------------------------------------------------------------------------- */

export function FactGrid({ facts }: { facts: ReadonlyArray<{ question: string; truth: string; pick: string }> }) {
  const right = facts.filter((fact) => fact.truth === fact.pick).length;
  return (
    <div className="mt-facts">
      <span className="mt-subhead">Every held-out fact, one mark each</span>
      <div
        className="mt-facts__grid"
        role="img"
        aria-label={`${right} of ${facts.length} held-out facts answered correctly, ${facts.length - right} wrong.`}
      >
        {facts.map((fact) => {
          const ok = fact.truth === fact.pick;
          return (
            <i
              key={fact.question}
              className={ok ? "is-right" : "is-wrong"}
              title={`${fact.question} … model: ${fact.pick}, true: ${fact.truth}`}
            >
              {ok ? "✓" : "×"}
            </i>
          );
        })}
      </div>
      <p className="mt-facts__key">
        <i className="is-right">✓</i> right <i className="is-wrong">×</i> wrong
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* End-of-run losses on one shared axis                                        */
/* -------------------------------------------------------------------------- */

export interface EndGroup {
  title: string;
  tone: "attention" | "forward";
  rows: ReadonlyArray<{ id: string; label: string; value: number; current: boolean }>;
}

export function EndDots({ groups, format }: { groups: ReadonlyArray<EndGroup>; format: (value: number) => string }) {
  const values = groups.flatMap((group) => group.rows.map((row) => row.value));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = Math.max(0.03, (high - low) * 0.12);
  const from = Math.floor((low - pad) * 20) / 20;
  const span = Math.ceil((high + pad) * 20) / 20 - from;
  const at = (value: number) => ((value - from) / span) * 100;
  return (
    <div
      className="mt-ends"
      role="img"
      aria-label={groups
        .map((group) => `${group.title}: ${group.rows.map((row) => `${row.label} ${format(row.value)}`).join(", ")}.`)
        .join(" ")}
    >
      <div className="mt-ends__head">
        <span className="mt-subhead">Held-out loss at the end of the run, on one scale</span>
        <span className="mt-ends__better">← lower is better</span>
      </div>
      {groups.map((group) => (
        <div className={`mt-ends__group mt-ends__group--${group.tone}`} key={group.title}>
          <b>{group.title}</b>
          {group.rows.map((row) => (
            <div className={`mt-ends__row${row.current ? " is-current" : ""}`} key={row.id}>
              <span>{row.label}</span>
              <span className="mt-ends__track">
                <i className={`mt-ends__dot mt-ends__dot--${row.id}`} style={{ left: `${at(row.value)}%` }} />
              </span>
              <code>{format(row.value)}</code>
            </div>
          ))}
        </div>
      ))}
      <div className="mt-ends__axis" aria-hidden="true">
        <span />
        <span className="mt-ends__ticks">
          <i>{from.toFixed(2)}</i>
          <i>{(from + span).toFixed(2)}</i>
        </span>
        <span />
      </div>
    </div>
  );
}
