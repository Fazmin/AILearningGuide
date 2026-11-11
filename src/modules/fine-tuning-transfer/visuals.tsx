import type { ReactNode } from "react";

/**
 * Pictures for the fine-tuning lab. Each one draws numbers the lab has already measured; nothing here
 * trains or scores anything itself.
 */

const signed = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;

/* -------------------------------------------------------------------------- */
/* The head start at every checkpoint                                          */
/* -------------------------------------------------------------------------- */

export function HeadStart({ gaps }: { gaps: ReadonlyArray<number> }) {
  const reach = Math.max(1e-6, ...gaps.map(Math.abs));
  const largest = gaps.indexOf(Math.max(...gaps));
  return (
    <div
      className="ftt-head"
      role="img"
      aria-label={`Head start at each of ${gaps.length} checkpoints, scratch loss minus transfer loss: ${gaps.map((gap) => signed(gap)).join(", ")}. Largest at checkpoint ${largest + 1}.`}
    >
      <span className="ftt-rows__title">
        Head start at each checkpoint: scratch loss − transfer loss, in nats. A taller bar is a bigger lead for the
        pretrained start.
      </span>
      <div className="ftt-head__bars">
        {gaps.map((gap, index) => (
          <span
            key={index}
            className={`ftt-head__col${gap < 0 ? " is-behind" : ""}${index === largest && gap > 0 ? " is-largest" : ""}`}
          >
            <code>{signed(gap)}</code>
            <i style={{ height: `${(Math.abs(gap) / reach) * 100}%` }} />
            <small>{index === 0 ? "start" : index === gaps.length - 1 ? "end" : `${Math.round((index / (gaps.length - 1)) * 100)}%`}</small>
          </span>
        ))}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* The trade: what the target gains, what the original loses                   */
/* -------------------------------------------------------------------------- */

export interface TradeRow {
  id: "target" | "source";
  label: string;
  before: number;
  after: number;
}

export function TradeArrows({ rows }: { rows: ReadonlyArray<TradeRow> }) {
  const values = rows.flatMap((row) => [row.before, row.after]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = Math.max(0.05, (high - low) * 0.1);
  const from = Math.floor((low - pad) * 10) / 10;
  const span = Math.ceil((high + pad) * 10) / 10 - from;
  const at = (value: number) => ((value - from) / span) * 100;
  return (
    <div
      className="ftt-trade"
      role="img"
      aria-label={rows
        .map((row) => `${row.label}: loss ${row.before.toFixed(3)} before, ${row.after.toFixed(3)} after, ${signed(row.after - row.before)}.`)
        .join(" ")}
    >
      <div className="ftt-trade__head">
        <span className="ftt-rows__title">Loss before → after this fine-tune, both corpora on one scale</span>
        <span className="ftt-trade__better">← lower is better</span>
      </div>
      {rows.map((row) => {
        const change = row.after - row.before;
        const left = Math.min(at(row.before), at(row.after));
        const width = Math.abs(at(row.after) - at(row.before));
        return (
          <div className={`ftt-trade__row ftt-trade__row--${row.id}`} key={row.id}>
            <b>{row.label}</b>
            <span className="ftt-trade__track">
              <i className="ftt-trade__move" style={{ left: `${left}%`, width: `${width}%` }} />
              <i className="ftt-trade__from" style={{ left: `${at(row.before)}%` }} />
              <i
                className={`ftt-trade__to ${change < 0 ? "is-down" : "is-up"}`}
                style={{ left: `${at(row.after)}%` }}
              />
            </span>
            <code>
              {row.before.toFixed(3)} → {row.after.toFixed(3)}
              <small>
                {signed(change)} {Math.abs(change) < 0.0005 ? "no change" : change < 0 ? "better" : "worse"}
              </small>
            </code>
          </div>
        );
      })}
      <div className="ftt-trade__axis" aria-hidden="true">
        <span />
        <span>
          <i>{from.toFixed(1)}</i>
          <i>{(from + span).toFixed(1)}</i>
        </span>
        <span />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Which sentences the fine-tune sees                                          */
/* -------------------------------------------------------------------------- */

function TileRow({
  label,
  kind,
  total,
  used,
  usedNote,
  restNote,
  sentences,
}: {
  label: string;
  kind: "target" | "source";
  total: number;
  used: number;
  usedNote: string;
  restNote: string;
  sentences: ReadonlyArray<string>;
}) {
  return (
    <div className={`ftt-tiles__row ftt-tiles__row--${kind}`}>
      <b>{label}</b>
      <span className="ftt-tiles__set">
        {Array.from({ length: total }, (_, index) => (
          <i key={index} className={index < used ? "is-used" : "is-rest"} title={sentences[index]}>
            {index + 1}
          </i>
        ))}
      </span>
      <small>
        {used} {usedNote}
        {total - used > 0 ? ` · ${total - used} ${restNote}` : ""}
      </small>
    </div>
  );
}

export function SentenceTiles({
  target,
  source,
  spoken,
}: {
  target: { label: string; sentences: ReadonlyArray<string>; trained: number };
  source: { label: string; sentences: ReadonlyArray<string>; replayed: number };
  spoken: string;
}) {
  return (
    <div className="ftt-tiles" role="img" aria-label={spoken}>
      <span className="ftt-rows__title">
        One tile per sentence. Filled tiles are the fine-tuning text; outlined tiles are never trained on, only scored.
      </span>
      <TileRow
        label={target.label}
        kind="target"
        total={target.sentences.length}
        used={target.trained}
        usedNote="trained on"
        restNote="held out"
        sentences={target.sentences}
      />
      <TileRow
        label={source.label}
        kind="source"
        total={source.sentences.length}
        used={source.replayed}
        usedNote="replayed"
        restNote="not replayed"
        sentences={source.sentences}
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* A forgetting value with a bar that grows away from zero                     */
/* -------------------------------------------------------------------------- */

export function ForgetCell({ value, reach, text }: { value: number; reach: number; text: string }) {
  if (!Number.isFinite(value)) return <>{text}</>;
  const size = Math.min(1, Math.abs(value) / reach) * 50;
  return (
    <span className={`ftt-cell ftt-cell--${value > 0.0005 ? "worse" : value < -0.0005 ? "better" : "level"}`}>
      {text}
      <i className="ftt-cell__track">
        <i style={{ width: `${size}%`, ...(value > 0 ? { left: "50%" } : { right: "50%" }) }} />
      </i>
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* The rule table as a matrix of marks                                         */
/* -------------------------------------------------------------------------- */

export interface RuleRow {
  when: string;
  route: string;
  /** For each of the three conditions: does this rule need it? */
  needs: readonly [boolean, boolean, boolean];
}

export function RuleMatrix({
  rules,
  facts,
  matched,
}: {
  rules: ReadonlyArray<RuleRow>;
  /** The learner's answers folded into the rule table's three conditions, with the words to print. */
  facts: ReadonlyArray<{ title: string; value: boolean; detail: string }>;
  matched: number;
}) {
  const mark = (needed: boolean, value: boolean): ReactNode =>
    !needed ? (
      <span className="ftt-rules__mark is-any">
        <span aria-hidden="true">·</span>
        <span className="sr-only">either answer</span>
      </span>
    ) : (
      <span className={`ftt-rules__mark ${value ? "is-met" : "is-unmet"}`}>
        <span aria-hidden="true">{value ? "✓" : "×"}</span> <span className="ftt-rules__needs">needs </span>Yes
        <span className="sr-only">{value ? ", met" : ", not met"}</span>
      </span>
    );
  return (
    <div className="ftt-rules__wrap">
      <table className="ftt-rules" aria-label="Rule table, first match wins">
        <thead>
          <tr>
            <th scope="col">Rule</th>
            {facts.map((fact) => (
              <th scope="col" key={fact.title}>
                {fact.title}
              </th>
            ))}
            <th scope="col">Route</th>
          </tr>
          <tr className="ftt-rules__you">
            <th scope="row">You</th>
            {facts.map((fact) => (
              <td key={fact.title}>
                <b>{fact.value ? "Yes" : "No"}</b>
                <small>{fact.detail}</small>
              </td>
            ))}
            <td />
          </tr>
        </thead>
        <tbody>
          {rules.map((rule, index) => (
            <tr
              key={rule.when}
              className={index === matched ? "is-matched" : index < matched ? "is-failed" : "is-after"}
              aria-current={index === matched ? "true" : undefined}
            >
              <th scope="row">
                {index + 1}
                <span className="sr-only">. {rule.when}</span>
              </th>
              {rule.needs.map((needed, column) => (
                <td key={column}>{mark(needed, facts[column].value)}</td>
              ))}
              <td>
                <span aria-hidden="true">→</span> <strong>{rule.route}</strong>
                {index === matched ? <em> (matched)</em> : null}
                {index > matched ? <small> not reached</small> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
