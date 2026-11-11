import type { Era } from "./eras";
import type { SignalId } from "./signals";

/**
 * Which of the card's two threads a stop belongs to. The symbolic thread is the Dartmouth proposal, expert
 * systems and the two winters; the other seven stops follow neural nets. This is a drawing choice only.
 */
const SYMBOLIC_YEARS = new Set([1956, 1974, 1980, 1987]);
export const eraThread = (year: number) => (SYMBOLIC_YEARS.has(year) ? "symbolic" : "neural");

/** Stops whose name would run into a neighbour's are lifted to a second row. */
const RAISED_YEARS = new Set([1980, 2017]);
/** In a half-width card the names need three rows; this is each stop's row there. */
const NARROW_ROW: Record<number, number> = {
  1956: 1, 1958: 1, 1969: 2, 1974: 2, 1980: 3, 1986: 1, 1987: 1, 1997: 2, 2012: 1, 2017: 2, 2022: 3,
};

const AXIS_START = 1954;
const AXIS_END = 2024;
const DECADES = [1960, 1970, 1980, 1990, 2000, 2010, 2020];
const at = (year: number) => `${((year - AXIS_START) / (AXIS_END - AXIS_START)) * 100}%`;
const span = (from: number, to: number) => `${((to - from) / (AXIS_END - AXIS_START)) * 100}%`;

/**
 * The eleven stops drawn to scale on a year axis, in two lanes: symbolic AI below, neural nets above. The two
 * winters are shaded spans, so the gaps between stops read as time and not as equal slider notches.
 */
export function EraTimeline({
  eras,
  current,
  onSelect,
}: {
  eras: readonly Era[];
  current: number;
  onSelect: (index: number) => void;
}) {
  const now = eras[current];
  return (
    <div className="wai-era" role="group" aria-label="Timeline of the eleven stops, drawn to scale by year">
      <div className="wai-era__track">
        <span className="wai-era__lane is-neural" aria-hidden="true">
          neural nets · weights fitted from examples
        </span>
        <span className="wai-era__lane is-symbolic" aria-hidden="true">
          symbolic AI · rules people wrote
        </span>
        <i className="wai-era__winter" style={{ left: at(1974), width: span(1974, 1980) }} aria-hidden="true" />
        <i className="wai-era__winter is-open" style={{ left: at(1987), width: span(1987, 1995) }} aria-hidden="true" />
        <b className="wai-era__winter-name" style={{ left: at(1974) }} aria-hidden="true">
          <span>winter </span>≈1974–80
        </b>
        <b className="wai-era__winter-name" style={{ left: at(1987) }} aria-hidden="true">
          <span>winter </span>from ≈1987
        </b>
        <i className="wai-era__rail is-neural" aria-hidden="true" />
        <i className="wai-era__rail is-symbolic" aria-hidden="true" />
        <div className="wai-era__axis" aria-hidden="true">
          {DECADES.map((year) => (
            <span key={year} style={{ left: at(year) }}>
              {year}
            </span>
          ))}
        </div>
        <i className="wai-era__now" style={{ left: at(now.year) }} aria-hidden="true" />
        {eras.map((era, index) => {
          const thread = eraThread(era.year);
          const edge = index === 0 ? " is-first" : index === eras.length - 1 ? " is-last" : "";
          return (
            <button
              key={era.year}
              type="button"
              className={`wai-era__stop is-${thread}${index === current ? " is-current" : ""}${
                RAISED_YEARS.has(era.year) ? " is-raised" : ""
              } is-n${NARROW_ROW[era.year] ?? 1}${edge}`}
              style={{ left: at(era.year) }}
              aria-pressed={index === current}
              aria-label={`${era.year} ${era.title}, ${thread === "symbolic" ? "symbolic AI" : "neural nets"} thread`}
              onClick={() => onSelect(index)}
            >
              <i aria-hidden="true" />
              <span>
                <b>{era.year}</b>
                {era.title}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

const RING_ORDER = ["ai", "ml", "dl", "gen", "llm"] as const;
const RING_RADII: Record<string, number> = { ai: 150, ml: 121, dl: 93, gen: 66, llm: 39 };
const RING_NAMES: Record<string, string> = { ai: "AI", ml: "ML", dl: "DL", gen: "Gen", llm: "LLM" };
/** Mid-radius of the band a term sits in when that ring is its tightest one. */
const BAND_RADIUS: Record<string, number> = { ai: 135.5, ml: 107, dl: 79.5, gen: 52.5, llm: 14 };
/** One fixed bearing per term (degrees clockwise from straight down) so markers never stack. */
const TERM_ANGLE = [-62, -31, 0, 31, 62];

export type BoardTerm = { id: string; label: string; ring: string; guess?: string };

/**
 * The nested rings as a sorting board. A placed term shows as its list number inside its tightest ring;
 * a wrong guess leaves a dashed ghost in the guessed ring with an arrow to where it belongs.
 */
export function RingBoard({
  terms,
  selected,
  destination,
  onPickRing,
}: {
  terms: readonly BoardTerm[];
  selected: string;
  destination: string;
  onPickRing: (ring: string) => void;
}) {
  const cx = 160;
  const cy = 160;
  const spot = (ring: string, index: number) => {
    const angle = ((TERM_ANGLE[index] ?? 0) * Math.PI) / 180;
    const radius = BAND_RADIUS[ring] ?? 0;
    return { x: cx + radius * Math.sin(angle), y: cy + radius * Math.cos(angle) };
  };
  const placed = terms.filter((term) => term.guess);
  const summary = placed.length
    ? placed
        .map((term) =>
          term.guess === term.ring
            ? `${term.label} sits in ${RING_NAMES[term.ring]}`
            : `${term.label} was placed in ${RING_NAMES[term.guess ?? ""]} and belongs in ${RING_NAMES[term.ring]}`,
        )
        .join(". ")
    : "No term is placed yet";
  return (
    <svg
      className="wai-board"
      viewBox="0 0 320 320"
      role="img"
      aria-label={`The five nested rings as a sorting board. ${RING_NAMES[destination]} is the ring chosen under Place in ring. ${summary}.`}
    >
      <defs>
        <marker id="wai-board-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" className="wai-board__head" />
        </marker>
      </defs>
      {RING_ORDER.map((ring, index) => {
        const inner = RING_ORDER[index + 1];
        const outer = RING_RADII[ring];
        // The band between this ring and the next one in: the region where this ring is the tightest.
        const band = inner
          ? `M${cx - outer} ${cy}a${outer} ${outer} 0 1 0 ${outer * 2} 0a${outer} ${outer} 0 1 0 ${-outer * 2} 0zM${
              cx - RING_RADII[inner]
            } ${cy}a${RING_RADII[inner]} ${RING_RADII[inner]} 0 1 1 ${RING_RADII[inner] * 2} 0a${RING_RADII[inner]} ${
              RING_RADII[inner]
            } 0 1 1 ${-RING_RADII[inner] * 2} 0z`
          : `M${cx - outer} ${cy}a${outer} ${outer} 0 1 0 ${outer * 2} 0a${outer} ${outer} 0 1 0 ${-outer * 2} 0z`;
        return (
          <g key={ring}>
            <path
              className={`wai-board__band${ring === destination ? " is-target" : ""}`}
              d={band}
              fillRule="evenodd"
              onClick={() => onPickRing(ring)}
            />
            <text className="wai-board__name" x={cx} y={cy - outer + 15} textAnchor="middle">
              {RING_NAMES[ring]}
            </text>
          </g>
        );
      })}
      {terms.map((term, index) => {
        if (!term.guess) return null;
        const home = spot(term.ring, index);
        const wrong = term.guess !== term.ring;
        const ghost = wrong ? spot(term.guess, index) : null;
        const reach = ghost ? Math.hypot(home.x - ghost.x, home.y - ghost.y) : 0;
        return (
          <g key={term.id} className={`wai-board__term${term.id === selected ? " is-selected" : ""}`}>
            {ghost && reach > 24 && (
              <line
                className="wai-board__move"
                x1={ghost.x + ((home.x - ghost.x) / reach) * 11}
                y1={ghost.y + ((home.y - ghost.y) / reach) * 11}
                x2={home.x - ((home.x - ghost.x) / reach) * 12}
                y2={home.y - ((home.y - ghost.y) / reach) * 12}
                markerEnd="url(#wai-board-head)"
              />
            )}
            {ghost && <circle className="wai-board__ghost" cx={ghost.x} cy={ghost.y} r="9.5" />}
            <circle className={`wai-board__chip${wrong ? " is-corrected" : ""}`} cx={home.x} cy={home.y} r="10" />
            <text className="wai-board__num" x={home.x} y={home.y + 3.8} textAnchor="middle">
              {index + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** A small schematic of where each signal's training target comes from. Decorative: the tile text says the same. */
export function SignalGlyph({ signal }: { signal: SignalId }) {
  if (signal === "supervised") {
    return (
      <svg className="wai-glyph" viewBox="0 0 176 48" aria-hidden="true">
        {[0, 1, 2].map((row) => (
          <g key={row}>
            <rect className="wai-glyph__x" x="4" y={3 + row * 11} width="50" height="8" rx="2" />
            <path className="wai-glyph__link" d={`M56 ${7 + row * 11}H74`} />
            <rect className="wai-glyph__y" x="76" y={3 + row * 11} width="22" height="8" rx="2" />
          </g>
        ))}
        <text className="wai-glyph__cap" x="29" y="45" textAnchor="middle">
          input
        </text>
        <text className="wai-glyph__cap is-y" x="76" y="45">
          answer, from a person
        </text>
      </svg>
    );
  }
  if (signal === "unsupervised") {
    const dots = [
      [16, 10], [27, 18], [14, 23], [28, 7],
      [64, 20], [75, 12], [77, 24], [62, 9],
      [112, 14], [121, 22], [108, 24],
    ];
    return (
      <svg className="wai-glyph" viewBox="0 0 176 48" aria-hidden="true">
        <ellipse className="wai-glyph__group" cx="21" cy="15" rx="15" ry="13" />
        <ellipse className="wai-glyph__group" cx="69" cy="16" rx="15" ry="13" />
        <ellipse className="wai-glyph__group" cx="114" cy="20" rx="14" ry="11" />
        {dots.map(([x, y], index) => (
          <circle key={index} className="wai-glyph__dot" cx={x} cy={y} r="2.7" />
        ))}
        <text className="wai-glyph__cap" x="4" y="45">
          inputs only
        </text>
        <text className="wai-glyph__cap is-y" x="58" y="45">
          groups found, not given
        </text>
      </svg>
    );
  }
  if (signal === "self") {
    return (
      <svg className="wai-glyph" viewBox="0 0 176 48" aria-hidden="true">
        {[0, 1, 2].map((cell) => (
          <rect key={cell} className="wai-glyph__x" x={4 + cell * 25} y="6" width="22" height="14" rx="2" />
        ))}
        <rect className="wai-glyph__y is-hidden" x="79" y="6" width="22" height="14" rx="2" />
        <text className="wai-glyph__q" x="90" y="17" textAnchor="middle">
          ?
        </text>
        <path className="wai-glyph__brace" d="M4 24v3h72v-3" />
        <path className="wai-glyph__brace is-y" d="M79 24v3h22v-3" />
        <text className="wai-glyph__cap" x="40" y="38" textAnchor="middle">
          what the model sees
        </text>
        <text className="wai-glyph__cap is-y" x="82" y="38">
          hidden part = the answer
        </text>
      </svg>
    );
  }
  return (
    <svg className="wai-glyph" viewBox="0 0 176 48" aria-hidden="true">
      <circle className="wai-glyph__agent" cx="13" cy="15" r="8" />
      <path className="wai-glyph__link" d="M25 15H55M50 11l5 4l-5 4" />
      <rect className="wai-glyph__x" x="59" y="6" width="30" height="18" rx="2" />
      <path className="wai-glyph__link" d="M92 15H112M107 11l5 4l-5 4" />
      <text className="wai-glyph__reward" x="128" y="19.5" textAnchor="middle">
        +1
      </text>
      <text className="wai-glyph__cap" x="13" y="38" textAnchor="middle">
        act
      </text>
      <text className="wai-glyph__cap" x="74" y="38" textAnchor="middle">
        world
      </text>
      <text className="wai-glyph__cap is-y" x="128" y="38" textAnchor="middle">
        reward, later
      </text>
    </svg>
  );
}
