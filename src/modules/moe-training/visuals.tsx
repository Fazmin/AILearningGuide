import type { CSSProperties } from "react";

/**
 * Pictures for the mixture-of-experts lab. Each one draws numbers the lab has already computed;
 * nothing here trains, routes or counts anything itself.
 */

/** Expert colours follow the load chart: four tones, then the same four hatched for experts 5 to 8. */
const EXPERT_TONES = ["forward", "attention", "gradient", "loss"] as const;
export const expertTone = (expert: number) => EXPERT_TONES[expert % EXPERT_TONES.length];
const expertClass = (expert: number) =>
  `moe-tone--${expertTone(expert)}${expert >= EXPERT_TONES.length ? " is-hatched" : ""}`;

const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) => (value >= 0 ? value.toFixed(2) : `−${Math.abs(value).toFixed(2)}`);
const count = (value: number) => value.toLocaleString("en-US");
const share = (value: number): CSSProperties => ({ width: `${Math.min(100, Math.max(0, value * 100))}%` });

/* -------------------------------------------------------------------------- */
/* One token through the router                                                */
/* -------------------------------------------------------------------------- */

const ROW = 38;

/** Curves from one point to one point per expert row, stretched to whatever width the column has. */
function Fan({
  rows,
  side,
  chosen,
  gates,
}: {
  rows: number;
  side: "in" | "out";
  chosen: ReadonlyArray<number>;
  gates: ReadonlyArray<number>;
}) {
  const height = rows * ROW;
  const middle = height / 2;
  return (
    <svg
      className={`moe-route__fan moe-route__fan--${side}`}
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      style={{ height }}
      aria-hidden="true"
    >
      {Array.from({ length: rows }, (_, e) => {
        const runs = chosen.includes(e);
        if (side === "out" && !runs) return null;
        const y = e * ROW + ROW / 2;
        const d =
          side === "in"
            ? `M0 ${middle} C55 ${middle} 45 ${y} 100 ${y}`
            : `M0 ${y} C55 ${y} 45 ${middle} 100 ${middle}`;
        return (
          <path
            key={e}
            d={d}
            className={runs ? "is-chosen" : undefined}
            style={side === "out" ? { strokeWidth: 1.5 + 6.5 * gates[e] } : undefined}
          />
        );
      })}
    </svg>
  );
}

export function RouteDiagram({
  token,
  tokenNote,
  spoken,
  z,
  bias,
  gates,
  chosen,
  next,
  topK,
}: {
  /** The character as drawn in the token chip. */
  token: string;
  tokenNote: string;
  /** The reading of the whole picture, for a screen reader. */
  spoken: string;
  z: ReadonlyArray<number>;
  bias: ReadonlyArray<number>;
  gates: ReadonlyArray<number>;
  chosen: ReadonlyArray<number>;
  /** The likeliest next characters, best first. */
  next: ReadonlyArray<{ label: string; share: number }>;
  topK: number;
}) {
  const rows = z.length;
  const picks = z.map((score, e) => score + bias[e]);
  const low = Math.min(0, ...picks, ...z);
  const high = Math.max(0, ...picks, ...z);
  const span = Math.max(1e-6, high - low);
  const at = (value: number) => ((value - low) / span) * 100;
  const biased = bias.some((value) => Math.abs(value) > 0.005);
  return (
    <div className="moe-route" role="img" aria-label={spoken}>
      <div className="moe-route__heads" aria-hidden="true">
        <span>Token</span>
        <span>
          The router scores every expert{biased ? " and adds its bias" : ""}; the top {topK} run
        </span>
        <span>Their answers, each × its gate</span>
      </div>
      <div className="moe-route__body">
        <div className="moe-route__token">
          <b>{token}</b>
          <small>{tokenNote}</small>
        </div>
        <Fan rows={rows} side="in" chosen={chosen} gates={gates} />
        <div className="moe-route__experts">
          <div className="moe-route__legend" aria-hidden="true">
            <span />
            <span>pick score z{biased ? " + b" : ""}</span>
            <span>gate p</span>
            <span />
          </div>
          {z.map((score, e) => {
            const runs = chosen.includes(e);
            const pick = picks[e];
            const from = Math.min(at(0), at(pick));
            const to = Math.max(at(0), at(pick));
            return (
              <div key={e} className={`moe-route__expert${runs ? " is-chosen" : ""}`} style={{ height: ROW - 6 }}>
                <b>E{e + 1}</b>
                <span className="moe-route__score">
                  <i className="moe-route__track">
                    <i className="moe-route__zero" style={{ left: `${at(0)}%` }} />
                    <i className="moe-route__bar" style={{ left: `${from}%`, width: `${to - from}%` }} />
                    {biased && <i className="moe-route__alone" style={{ left: `${at(score)}%` }} />}
                  </i>
                  <code>{signed(pick)}</code>
                </span>
                <span className="moe-route__gate">
                  <i className="moe-route__track">
                    <i className="moe-route__bar" style={share(gates[e])} />
                  </i>
                  <code>{gates[e].toFixed(2)}</code>
                </span>
                <em>{runs ? "runs" : "skipped"}</em>
              </div>
            );
          })}
        </div>
        <Fan rows={rows} side="out" chosen={chosen} gates={gates} />
        <div className="moe-route__guess">
          {next.map((entry, index) => (
            <div key={entry.label} className={index === 0 ? "is-top" : undefined}>
              <b>{entry.label}</b>
              <i className="moe-route__track">
                <i className="moe-route__bar" style={share(entry.share)} />
              </i>
              <code>{percent(entry.share)}</code>
            </div>
          ))}
          <small>likeliest next characters</small>
        </div>
      </div>
      {biased && (
        <p className="moe-route__key" aria-hidden="true">
          <i className="moe-route__alone" /> marks the score z alone. The bias b moves the bar, and with it who is
          picked. The gate ignores b.
        </p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* The split before and after training                                         */
/* -------------------------------------------------------------------------- */

function SplitBar({ shares }: { shares: ReadonlyArray<number> }) {
  return (
    <span className="moe-split__bar">
      <span className="moe-split__fill">
      {shares.map((value, e) =>
        value > 0.0005 ? (
          <i key={e} className={expertClass(e)} style={{ flexGrow: value }} title={`E${e + 1}: ${percent(value)}`}>
            {value >= 0.085 && (
              <span>
                E{e + 1} {percent(value)}
              </span>
            )}
          </i>
        ) : null,
      )}
      </span>
      {shares.slice(1).map((_, index) => (
        <u key={index} style={{ left: `${((index + 1) / shares.length) * 100}%` }} />
      ))}
    </span>
  );
}

export function SplitBars({ rows }: { rows: ReadonlyArray<{ label: string; shares: ReadonlyArray<number>; note: string }> }) {
  const experts = rows[0]?.shares.length ?? 1;
  return (
    <div
      className="moe-split"
      role="img"
      aria-label={rows
        .map((row) => `${row.label}: ${row.shares.map((value, e) => `E${e + 1} ${percent(value)}`).join(", ")}; ${row.note}.`)
        .join(" ")}
    >
      {rows.map((row) => (
        <div className="moe-split__row" key={row.label}>
          <span className="moe-split__label">{row.label}</span>
          <SplitBar shares={row.shares} />
          <code>{row.note}</code>
        </div>
      ))}
      {experts > 1 && (
        <p className="moe-split__key">
          <u /> an even split would cut the bar here, every {percent(1 / experts)}
        </p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Small bars inside table cells                                               */
/* -------------------------------------------------------------------------- */

/** A number with a bar under it, measured against `max`. */
export function CellBar({ value, max, text, tone }: { value: number; max: number; text: string; tone?: "forward" | "muted" }) {
  return (
    <span className={`moe-cell moe-cell--${tone ?? "forward"}`}>
      {text}
      <i className="moe-cell__track">
        <i style={share(max > 0 ? value / max : 0)} />
      </i>
    </span>
  );
}

/** A loss with a bar that grows left of the dense baseline when it is lower (better), right when higher. */
export function LossCell({ value, baseline, reach, text }: { value: number; baseline: number; reach: number; text: string }) {
  const gap = value - baseline;
  const size = Math.min(1, Math.abs(gap) / reach) * 50;
  const clipped = Math.abs(gap) > reach;
  return (
    <span className={`moe-cell moe-cell--${gap > 0.0005 ? "worse" : gap < -0.0005 ? "better" : "level"}`}>
      {text}
      <i className="moe-cell__track moe-cell__track--centre">
        <i style={{ width: `${size}%`, ...(gap > 0 ? { left: "50%" } : { right: "50%" }) }} />
        {clipped && <b style={gap > 0 ? { right: 0 } : { left: 0 }}>{gap > 0 ? "›" : "‹"}</b>}
      </i>
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Capacity: what each expert is sent against what it can seat                 */
/* -------------------------------------------------------------------------- */

export function CapacityBars({
  shares,
  room,
  dropped,
}: {
  shares: ReadonlyArray<number>;
  /** The share of all choices one expert's seats can serve, or null with the cap off. */
  room: number | null;
  dropped: number;
}) {
  return (
    <div className="moe-cap">
      {room !== null && (
        <div
          className="moe-cap__fate"
          role="img"
          aria-label={`Of all the router's choices, ${percent(1 - dropped)} were served and ${percent(dropped)} were dropped.`}
        >
          <span className="moe-cap__fate-bar">
            {dropped < 0.995 && (
              <i className="is-served" style={{ flexGrow: 1 - dropped }}>
                {1 - dropped >= 0.2 && <span>served {percent(1 - dropped)}</span>}
              </i>
            )}
            {dropped > 0.005 && (
              <i className="is-dropped" style={{ flexGrow: dropped }}>
                {dropped >= 0.2 && <span>dropped {percent(dropped)}</span>}
              </i>
            )}
          </span>
        </div>
      )}
      <ul
        className={`moe-cap__list${room !== null && room > 0.55 ? " is-wide-cap" : ""}`}
        aria-label="Each expert's share of the router's choices, with the share its seats can serve"
        style={room !== null ? ({ "--moe-cap": `${Math.min(100, room * 100)}%` } as CSSProperties) : undefined}
      >
        {room !== null && (
          <li className="moe-cap__axis" aria-hidden="true">
            <span />
            <span className="moe-cap__track-head">
              <em>seats for {percent(room)}</em>
            </span>
            <span />
          </li>
        )}
        {shares.map((value, e) => {
          const over = room !== null && value > room + 0.005;
          const served = room === null ? value : Math.min(value, room);
          return (
            <li key={e} className={over ? "is-over" : undefined}>
              <b>E{e + 1}</b>
              <span className="moe-cap__track">
                <i className="moe-cap__served" style={share(served)} />
                {over && room !== null && (
                  <i className="moe-cap__over" style={{ left: `${room * 100}%`, width: `${(value - room) * 100}%` }} />
                )}
                {room !== null && <i className="moe-cap__line" />}
              </span>
              <code>
                {percent(value)}
                {room !== null && <small>{over ? "over seats" : "fits"}</small>}
              </code>
            </li>
          );
        })}
      </ul>
      {room !== null && (
        <p className="moe-cap__key">
          <i className="moe-cap__served" /> sent, within its seats <i className="moe-cap__over" /> sent, past its seats
        </p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Stored weights against the weights one token runs                           */
/* -------------------------------------------------------------------------- */

export interface ParamRow {
  label: string;
  total: number;
  segments: ReadonlyArray<{ id: string; label: string; size: number; kind: "shared" | "active" | "idle" }>;
}

export function ParamBars({ rows, scale, spoken }: { rows: ReadonlyArray<ParamRow>; scale: number; spoken: string }) {
  return (
    <div className="moe-params" role="img" aria-label={spoken}>
      {rows.map((row) => (
        <div className="moe-params__row" key={row.label}>
          <span className="moe-params__label">{row.label}</span>
          <span className="moe-params__lane">
            <span className="moe-params__bar" style={share(row.total / scale)}>
              {row.segments.map((segment) => (
                <i
                  key={segment.id}
                  className={`is-${segment.kind}`}
                  style={{ flexGrow: segment.size }}
                  title={`${segment.label}: ${count(segment.size)} weights`}
                >
                  <span>{segment.label}</span>
                </i>
              ))}
            </span>
            <code>{count(row.total)}</code>
          </span>
        </div>
      ))}
      <p className="moe-params__key">
        <i className="is-shared" /> embedding and router, used by every token <i className="is-active" /> an expert
        that runs <i className="is-idle" /> an expert stored but skipped
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Which characters each expert receives                                       */
/* -------------------------------------------------------------------------- */

export function CharacterGroups({
  groups,
  heaviest,
}: {
  groups: ReadonlyArray<ReadonlyArray<{ name: string; pairs: number }>>;
  heaviest: number;
}) {
  return (
    <ul className="moe-groups" aria-label="The characters each expert receives">
      {groups.map((members, e) => (
        <li key={e}>
          <b>E{e + 1}</b>
          {members.length ? (
            <span className="moe-groups__tiles">
              {members.map((member) => (
                <i
                  key={member.name}
                  title={`${member.name}: ${member.pairs} pairs`}
                  style={{ "--moe-weight": Math.sqrt(member.pairs / heaviest) } as CSSProperties}
                >
                  {member.name}
                </i>
              ))}
            </span>
          ) : (
            <span className="moe-groups__none">no characters</span>
          )}
          <code>{count(members.reduce((sum, member) => sum + member.pairs, 0))} pairs</code>
        </li>
      ))}
    </ul>
  );
}
