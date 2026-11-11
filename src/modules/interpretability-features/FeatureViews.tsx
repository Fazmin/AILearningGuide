import type { CSSProperties } from "react";

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

export const showChar = (character: string) => (character === " " ? "␣" : character === "\n" ? "↵" : character);

/**
 * The 256 numbers of one residual vector as signed bars, with the SAE's
 * reconstruction drawn as a dot on each bar so the two can be compared
 * dimension by dimension.
 */
export function ResidualBars({
  x,
  reconstruction,
  label,
}: {
  x: ArrayLike<number> | null;
  reconstruction: ArrayLike<number> | null;
  label: string;
}) {
  const width = 512;
  const height = 110;
  const middle = height / 2;
  const values = x ? Array.from(x) : [];
  const peak = Math.max(1e-6, ...values.map(Math.abs), ...(reconstruction ? Array.from(reconstruction).map(Math.abs) : []));
  const scale = (middle - 6) / peak;
  const step = width / 256;
  return (
    <svg className="sf-residual" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
      <line x1={0} x2={width} y1={middle} y2={middle} className="sf-residual__axis" />
      {values.map((value, index) => (
        <rect
          key={index}
          x={index * step}
          y={value >= 0 ? middle - value * scale : middle}
          width={Math.max(0.6, step - 0.4)}
          height={Math.abs(value) * scale}
          className="sf-residual__bar"
        />
      ))}
      {reconstruction &&
        Array.from(reconstruction).map((value, index) => (
          <rect
            key={`r-${index}`}
            x={index * step}
            y={middle - value * scale - 0.9}
            width={Math.max(0.6, step - 0.4)}
            height={1.8}
            className="sf-residual__recon"
          />
        ))}
    </svg>
  );
}

/** 1,024 dictionary slots with a spike for every active feature. */
export function SparseCode({
  active,
  total,
  selected,
  label,
}: {
  active: ReadonlyArray<{ feature: number; value: number }>;
  total: number;
  selected: number;
  label: string;
}) {
  const width = 1024;
  const height = 120;
  const base = 100;
  const peak = Math.max(1e-6, ...active.map((entry) => entry.value));
  // Spikes stop short of the top so the tallest ones can carry their feature number.
  const rise = (value: number) => (value / peak) * (base - 26);
  // Name the three tallest spikes and the selected one, skipping a name that would sit on a taller neighbour's.
  const named: { feature: number; value: number }[] = [];
  const wanted = [...active].sort((left, right) => right.value - left.value).slice(0, 3);
  const chosen = active.find((entry) => entry.feature === selected);
  if (chosen && !wanted.includes(chosen)) wanted.push(chosen);
  for (const entry of wanted) {
    if (named.every((other) => Math.abs(other.feature - entry.feature) / total > 0.07)) named.push(entry);
  }
  return (
    <div className="sf-code-plot">
      <svg className="sf-code-strip" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
        <rect x={0} y={base} width={width} height={4} className="sf-code-strip__track" />
        {[0, 256, 512, 768, 1024].map((tick) => (
          <line key={tick} x1={tick} x2={tick} y1={base} y2={base + 10} className="sf-code-strip__tick" />
        ))}
        {active.map(({ feature, value }) => {
          const h = rise(value);
          return (
            <g key={feature} className={feature === selected ? "is-selected" : ""}>
              <rect x={(feature / total) * width - 3} y={base - h} width={6} height={h} className="sf-code-strip__spike" />
              <title>{`feature ${feature}: ${value.toFixed(2)}`}</title>
            </g>
          );
        })}
      </svg>
      {named.map((entry) => (
        <span
          key={entry.feature}
          aria-hidden="true"
          className={`sf-code-tag${entry.feature === selected ? " is-selected" : ""}`}
          style={{
            left: `${clamp((entry.feature / total) * 100, 3, 97)}%`,
            bottom: `${((height - base + rise(entry.value)) / height) * 100}%`,
          }}
        >
          #{entry.feature}
        </span>
      ))}
    </div>
  );
}

/** Characters of the probe text, shaded by one feature's activation, marking which fire and which match a label. */
export function ActivationText({
  text,
  activations,
  threshold,
  matches,
  selected,
  onSelect,
  maximum,
  earlyPositions = 0,
}: {
  text: string;
  activations: ReadonlyArray<number>;
  threshold: number;
  matches: (character: string, position: number) => boolean;
  selected?: number;
  onSelect?: (position: number) => void;
  maximum: number;
  /** Positions below this are the start of the model's window and get a marked top edge. */
  earlyPositions?: number;
}) {
  return (
    <ol className="sf-text" aria-label="Probe text with the feature's activation on each character">
      {Array.from(text).map((character, position) => {
        const value = activations[position] ?? 0;
        const fires = value > threshold;
        const match = matches(character, position);
        const early = position < earlyPositions;
        const spoken = character === " " ? "space" : character === "\n" ? "line break" : character;
        const body = (
          <>
            <span>{showChar(character)}</span>
            <small>{value > 0 ? value.toFixed(1) : "·"}</small>
          </>
        );
        return (
          <li
            key={position}
            className={[fires ? "is-firing" : "", match ? "is-match" : "", early ? "is-early" : "", selected === position ? "is-selected" : ""]
              .filter(Boolean)
              .join(" ")}
            style={{ "--a": clamp(value / Math.max(1e-6, maximum), 0, 1) } as CSSProperties}
          >
            {onSelect ? (
              <button
                type="button"
                aria-pressed={selected === position}
                aria-label={`position ${position}, ${spoken}, activation ${value.toFixed(2)}${fires ? ", fires" : ""}${match ? ", matches the label" : ""}${early ? ", window start" : ""}`}
                onClick={() => onSelect(position)}
              >
                {body}
              </button>
            ) : (
              <div
                role="group"
                aria-label={`position ${position}, ${spoken}, activation ${value.toFixed(2)}${fires ? ", fires" : ""}${match ? ", matches the label" : ""}${early ? ", window start" : ""}`}
              >
                {body}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** How much of the residual the active features rebuild, as one bar split into rebuilt and left over. */
export function ReconBar({ unexplained, count }: { unexplained: number | null; count: number }) {
  if (unexplained === null) return null;
  const rest = clamp(unexplained, 0, 1);
  const show = (value: number) => `${(value * 100).toFixed(1)}%`;
  return (
    <div
      className="sf-recon"
      role="img"
      aria-label={`The ${count} active features rebuild ${show(1 - rest)} of the residual's squared length; ${show(rest)} is left unexplained.`}
    >
      <div className="sf-recon__labels" aria-hidden="true">
        <span>
          <b>{show(1 - rest)}</b> of x rebuilt by the {count} active features
        </span>
        <span>
          left unexplained <b>{show(rest)}</b>
        </span>
      </div>
      <div className="sf-recon__bar" aria-hidden="true">
        <i style={{ width: `${(1 - rest) * 100}%` }} />
        <i className="is-rest" style={{ width: `${rest * 100}%` }} />
      </div>
    </div>
  );
}

/** Where one cached example sat in the 64-character window the model read: a dot on a 0 to 63 track. */
export function WindowTrack({ position, early, length = 64 }: { position: number; early: number; length?: number }) {
  return (
    <span
      className={`sf-win${position < early ? " is-early" : ""}`}
      aria-hidden="true"
      style={{ "--p": clamp(position / (length - 1), 0, 1), "--early": early / length } as CSSProperties}
    >
      <i />
    </span>
  );
}

/**
 * Precision and recall as rows of the characters themselves. The first row is every character where the
 * feature fires; the second is every character the label matches. A filled chip is a hit in both rows.
 */
export function LabelScore({
  text,
  activations,
  threshold,
  matches,
}: {
  text: string;
  activations: ReadonlyArray<number>;
  threshold: number;
  matches: (character: string, position: number) => boolean;
}) {
  const cells = Array.from(text).map((character, position) => ({
    character,
    position,
    fires: (activations[position] ?? 0) > threshold,
    match: matches(character, position),
  }));
  type Cell = (typeof cells)[number];
  const hits = cells.filter((cell) => cell.fires && cell.match).length;
  const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : word === "miss" ? "es" : "s"}`;
  const rows = [
    {
      name: "Precision",
      ask: "of the characters where it fires, how many match the label?",
      cells: cells.filter((cell) => cell.fires),
      good: (cell: Cell) => cell.match,
      badClass: "is-false",
      bad: "false alarm",
      none: "it fires nowhere on this text",
    },
    {
      name: "Recall",
      ask: "of the characters that match the label, on how many does it fire?",
      cells: cells.filter((cell) => cell.match),
      good: (cell: Cell) => cell.fires,
      badClass: "is-miss",
      bad: "miss",
      none: "no character matches the label",
    },
  ];
  return (
    <div className="sf-score">
      {rows.map((row) => {
        const missed = row.cells.length - hits;
        return (
          <div className="sf-score__row" key={row.name}>
            <span className="sf-score__name">
              <b>{row.name}</b> {row.ask}
            </span>
            <span
              className="sf-score__cells"
              role="img"
              aria-label={`${row.name}: ${plural(hits, "hit")} and ${plural(missed, row.bad)} among ${row.cells.length} characters.`}
            >
              {row.cells.length === 0 && <em>{row.none}</em>}
              {row.cells.map((cell) => (
                <i key={cell.position} className={row.good(cell) ? "is-hit" : row.badClass}>
                  {showChar(cell.character)}
                </i>
              ))}
            </span>
            <span className="sf-score__frac" aria-hidden="true">
              <b>
                {hits} of {row.cells.length}
              </b>
              {missed > 0 ? ` · ${plural(missed, row.bad)}` : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Two decoder directions drawn at their true angle, on a half circle from "same way" to "opposite",
 * with the sliver around a right angle where a typical pair of directions sits.
 */
export function AngleDial({ a, b, cosine, meanAbs }: { a: number; b: number; cosine: number; meanAbs: number }) {
  const cx = 150;
  const cy = 146;
  const R = 104;
  const at = (degrees: number, radius = R) => ({
    x: cx + radius * Math.cos((degrees * Math.PI) / 180),
    y: cy - radius * Math.sin((degrees * Math.PI) / 180),
  });
  const degreesOf = (value: number) => (Math.acos(clamp(value, -1, 1)) * 180) / Math.PI;
  const angle = degreesOf(cosine);
  const typical = degreesOf(meanAbs);
  const wedgeFrom = at(typical);
  const wedgeTo = at(180 - typical);
  const tip = at(angle, R - 4);
  const sweep = at(angle, 30);
  const labelOnLeft = angle > 90;
  return (
    <svg
      className="sf-dial"
      viewBox="0 0 300 172"
      role="img"
      aria-label={`Feature ${a} and feature ${b} point ${angle.toFixed(0)} degrees apart, cosine ${cosine.toFixed(3)}, which is nearly opposite. A typical pair of the 1,024 directions has absolute cosine ${meanAbs.toFixed(3)}, about ${typical.toFixed(0)} degrees, close to a right angle.`}
    >
      <path className="sf-dial__arc" d={`M${cx + R},${cy} A${R},${R} 0 0 0 ${cx - R},${cy}`} />
      <path
        className="sf-dial__typical"
        d={`M${cx},${cy} L${wedgeFrom.x.toFixed(1)},${wedgeFrom.y.toFixed(1)} A${R},${R} 0 0 0 ${wedgeTo.x.toFixed(1)},${wedgeTo.y.toFixed(1)} Z`}
      />
      <line className="sf-dial__base" x1={cx - R - 8} x2={cx + R + 8} y1={cy} y2={cy} />
      <text className="sf-dial__note" x={cx} y={12} textAnchor="middle">
        a typical pair: |cos| {meanAbs.toFixed(3)}, about {typical.toFixed(0)}°
      </text>
      <text className="sf-dial__tick" x={cx} y={26} textAnchor="middle">
        right angle · cos 0
      </text>
      <text className="sf-dial__tick" x={cx + R + 8} y={cy + 16} textAnchor="end">
        same way · +1
      </text>
      <text className="sf-dial__tick" x={cx - R - 8} y={cy + 16} textAnchor="start">
        opposite · −1
      </text>
      <path className="sf-dial__sweep" d={`M${cx + 30},${cy} A30,30 0 0 0 ${sweep.x.toFixed(1)},${sweep.y.toFixed(1)}`} />
      <text className="sf-dial__angle" x={cx + 8} y={cy - 38} textAnchor="start">
        {angle.toFixed(0)}° apart
      </text>
      <line className="sf-dial__arrow is-a" x1={cx} y1={cy} x2={cx + R - 4} y2={cy} />
      <circle className="sf-dial__head is-a" cx={cx + R - 4} cy={cy} r={4.5} />
      <line className="sf-dial__arrow is-b" x1={cx} y1={cy} x2={tip.x} y2={tip.y} />
      <rect
        className="sf-dial__head is-b"
        x={tip.x - 4.5}
        y={tip.y - 4.5}
        width={9}
        height={9}
        transform={`rotate(45 ${tip.x} ${tip.y})`}
      />
      <text className="sf-dial__name" x={cx + R - 4} y={cy - 10} textAnchor="end">
        #{a}
      </text>
      <text className="sf-dial__name" x={tip.x + (labelOnLeft ? 2 : -2)} y={tip.y - 10} textAnchor={labelOnLeft ? "start" : "end"}>
        #{b}
      </text>
    </svg>
  );
}

/** The probe text with a mark above each character where feature a is active and below where feature b is. */
export function PairStrip({
  text,
  rows,
  a,
  b,
}: {
  text: string;
  rows: ReadonlyArray<ArrayLike<number>>;
  a: number;
  b: number;
}) {
  const cells = Array.from(text).map((character, position) => ({
    character,
    first: (rows[position]?.[a] ?? 0) > 0,
    second: (rows[position]?.[b] ?? 0) > 0,
  }));
  type Cell = (typeof cells)[number];
  const count = (pick: (cell: Cell) => boolean) => cells.filter(pick).length;
  return (
    <div className="sf-pair">
      <ol
        className="sf-pair__cells"
        aria-label={`On the probe text, feature ${a} is active on ${count((cell) => cell.first)} characters, feature ${b} on ${count(
          (cell) => cell.second,
        )}, and both on ${count((cell) => cell.first && cell.second)}.`}
      >
        {cells.map((cell, position) => (
          <li
            key={position}
            aria-hidden="true"
            className={[cell.first ? "is-a" : "", cell.second ? "is-b" : ""].filter(Boolean).join(" ")}
          >
            {showChar(cell.character)}
          </li>
        ))}
      </ol>
      <ul className="sf-pair__key" aria-hidden="true">
        <li>
          <i className="is-a" /> bar on top: #{a} is active
        </li>
        <li>
          <i className="is-b" /> bar underneath: #{b} is active
        </li>
      </ul>
    </div>
  );
}

export interface SteerRow {
  id: string;
  /** The character as shown, with "(label)" appended for the label character. */
  label: string;
  target: boolean;
  value: number;
  display: string;
  /** The same character's probability before steering; drawn as a tick on the bar. */
  baseline?: number;
}

/** A ranked list of next characters. Each bar can carry a tick at the character's probability before steering. */
export function SteerList({
  label,
  rows,
  max,
  tone,
}: {
  label: string;
  rows: ReadonlyArray<SteerRow>;
  max: number;
  tone: "forward" | "gradient" | "muted";
}) {
  return (
    <ul className={`sf-steer-list is-${tone}`} aria-label={label}>
      {rows.map((row) => (
        <li key={row.id} className={row.target ? "is-target" : ""}>
          <span>{row.label}</span>
          <i>
            <b style={{ width: `${clamp(row.value / max, 0, 1) * 100}%` }} />
            {row.baseline !== undefined && <u style={{ left: `${clamp(row.baseline / max, 0, 1) * 100}%` }} />}
          </i>
          <strong>{row.display}</strong>
        </li>
      ))}
    </ul>
  );
}

/**
 * The feature's change in log probability against the spread of the random directions at one strength,
 * all on one axis, so "inside or outside the random range" is something you can see.
 */
export function NullStrip({
  name,
  feature,
  low,
  high,
  count,
  format,
  verdict,
}: {
  name: string;
  feature: number;
  low: number;
  high: number;
  count: number;
  format: (value: number) => string;
  verdict: string;
}) {
  const floor = Math.min(0, low, feature);
  const ceiling = Math.max(0, high, feature);
  const pad = Math.max(0.2, (ceiling - floor) * 0.08);
  const from = floor - pad;
  const span = ceiling + pad - from;
  const share = (value: number) => (value - from) / span;
  const at = (value: number) => `${share(value) * 100}%`;
  const inside = feature >= low && feature <= high;
  return (
    <div className="sf-null">
      <div className="sf-null__head">
        <strong>Is the change outside what any push this long does?</strong>
        <span>{verdict}</span>
      </div>
      <div
        className="sf-null__track"
        role="img"
        aria-label={`${name} changed the label character's log probability by ${format(feature)}. The ${count} random directions changed it by ${format(low)} to ${format(high)}, so the feature's change is ${inside ? "inside" : "outside"} the random range.`}
      >
        <i className="sf-null__zero" style={{ left: at(0) }}>
          <em>0</em>
        </i>
        <i className="sf-null__range" style={{ left: at(low), width: `${((high - low) / span) * 100}%` }}>
          <em className={share(low) > 0.5 ? "is-left" : ""}>
            {count} random directions: {format(low)} to {format(high)}
          </em>
        </i>
        <i className={`sf-null__feature${inside ? " is-inside" : ""}`} style={{ left: at(feature) }}>
          <em className={share(feature) > 0.5 ? "is-left" : ""}>
            {name}: {format(feature)}
          </em>
        </i>
      </div>
      <div className="sf-null__axis" aria-hidden="true">
        <span>{format(from)}</span>
        <span>change in log p of the label character</span>
        <span>{format(ceiling + pad)}</span>
      </div>
    </div>
  );
}
