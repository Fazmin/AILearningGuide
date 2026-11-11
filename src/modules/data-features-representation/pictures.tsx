import { VOCABULARY, idOf } from "./preview";

/**
 * Rows for marks that share one number line: each value takes the lowest row in which the mark before it is
 * at least `gap` away, so equal or crowded values stack instead of covering each other.
 */
export const swarmRows = (values: readonly number[], gap: number): number[] => {
  const order = values.map((value, index) => ({ value, index })).sort((left, right) => left.value - right.value || left.index - right.index);
  const lastInRow: number[] = [];
  const rows = new Array<number>(values.length).fill(0);
  for (const item of order) {
    let row = lastInRow.findIndex((last) => item.value - last >= gap - 1e-9);
    if (row < 0) {
      row = lastInRow.length;
      lastInRow.push(item.value);
    } else {
      lastInRow[row] = item.value;
    }
    rows[item.index] = row;
  }
  return rows;
};

/** A 3×3 picture drawn small. Cells that differ from `against` are ringed, so the distance can be counted. */
export function MiniGrid({ pixels, against, label }: { pixels: string; against: string; label: string }) {
  return (
    <span className="dfr-mini" role="img" aria-label={label}>
      {Array.from(pixels).map((char, index) => (
        <i
          key={index}
          className={`${char === "1" ? "is-on" : ""}${against[index] !== char ? " is-diff" : ""}`.trim() || undefined}
        />
      ))}
    </span>
  );
}

/**
 * The eight vocabulary IDs as a number line. Words in the current sentence are filled. Two arcs from cat show
 * the gaps subtraction would report: 3 to mat and 4 to dog, an order the numbering invented.
 */
export function VocabLine({ used }: { used: readonly number[] }) {
  const width = 640;
  const height = 132;
  const left = 34;
  const right = width - 34;
  const axis = 84;
  const x = (id: number) => left + (id / (VOCABULARY.length - 1)) * (right - left);
  const cat = idOf("cat");
  const mat = idOf("mat");
  const dog = idOf("dog");
  const arc = (from: number, to: number, lift: number) =>
    `M${x(from)} ${axis - 9}C${x(from)} ${axis - lift} ${x(to)} ${axis - lift} ${x(to)} ${axis - 9}`;
  const usedWords = VOCABULARY.filter((_, id) => used.includes(id));
  return (
    <svg
      className="dfr-vocab"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`The eight vocabulary IDs on a number line, 0 to 7: ${VOCABULARY.map((word, id) => `${word} ${id}`).join(
        ", ",
      )}. This sentence uses ${usedWords.join(", ")}. On the line, cat is 3 steps from mat and 4 steps from dog.`}
    >
      <line className="dfr-vocab__axis" x1={left - 14} x2={right + 14} y1={axis} y2={axis} />
      <path className="dfr-vocab__arc is-near" d={arc(cat, mat, 46)} />
      <path className="dfr-vocab__arc is-far" d={arc(cat, dog, 76)} />
      <text className="dfr-vocab__gap is-near" x={(x(cat) + x(mat)) / 2} y={axis - 40} textAnchor="middle">
        cat → mat: gap 3
      </text>
      <text className="dfr-vocab__gap is-far" x={(x(cat) + x(dog)) / 2} y={axis - 66} textAnchor="middle">
        cat → dog: gap 4
      </text>
      {VOCABULARY.map((word, id) => {
        const on = used.includes(id);
        return (
          <g key={word} className={on ? "is-used" : undefined}>
            <circle className="dfr-vocab__dot" cx={x(id)} cy={axis} r={on ? 7 : 5} />
            <text className="dfr-vocab__id" x={x(id)} y={axis + 22} textAnchor="middle">
              {id}
            </text>
            <text className="dfr-vocab__word" x={x(id)} y={axis + 38} textAnchor="middle">
              {word}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export type VoteMark = {
  key: string;
  name: string;
  distance: number;
  high: boolean;
  voter: boolean;
  dropped: boolean;
  /** Distance on the clean table, when the mess moved this row. */
  cleanDistance: number | null;
  note: string;
};

const VOTE_REACH = 3.4;

/**
 * Every listing placed by its distance from the new listing. The band covers the three nearest, the ones that
 * vote; a row the mess moved keeps a ghost at its clean distance with an arrow to where it went.
 */
export function VoteLine({ marks, kth, label }: { marks: readonly VoteMark[]; kth: number; label: string }) {
  const rows = swarmRows(
    marks.map((mark) => mark.distance),
    0.8,
  );
  const depth = Math.max(0, ...rows) + 1;
  const at = (distance: number) => `${(Math.min(VOTE_REACH, Math.max(0, distance)) / VOTE_REACH) * 100}%`;
  return (
    <div className="dfr-vote" role="img" aria-label={label}>
      <div className="dfr-vote__field" style={{ height: `${depth * 2.9 + 1.5}rem` }}>
        <i className="dfr-vote__reach" style={{ width: at(kth) }}>
          <b>the 3 nearest vote</b>
        </i>
        <span className="dfr-vote__origin">new listing</span>
        {marks.map((mark, index) => {
          const moved = mark.cleanDistance !== null && Math.abs(mark.cleanDistance - mark.distance) > 0.005;
          const from = moved ? Math.min(mark.cleanDistance ?? 0, mark.distance) : 0;
          const to = moved ? Math.max(mark.cleanDistance ?? 0, mark.distance) : 0;
          const top = `${rows[index] * 2.9 + 1.5}rem`;
          return (
            <div key={mark.key}>
              {moved && (
                <>
                  <i
                    className={`dfr-vote__move${(mark.cleanDistance ?? 0) > mark.distance ? " is-left" : ""}`}
                    style={{ left: at(from), width: `calc(${at(to)} - ${at(from)})`, top }}
                  />
                  <i className="dfr-vote__ghost" style={{ left: at(mark.cleanDistance ?? 0), top }} />
                </>
              )}
              <div
                className={`dfr-vote__pin ${mark.high ? "is-high" : "is-low"}${mark.voter ? " is-voter" : ""}${
                  mark.dropped ? " is-dropped" : ""
                }${mark.note ? " is-spoiled" : ""}`}
                style={{ left: at(mark.distance), top }}
              >
                <i />
                <span>
                  <b>{mark.name}</b>
                  <small>
                    {mark.dropped ? "dropped" : `${mark.distance.toFixed(2)} · ${mark.high ? "high" : "low"}`}
                  </small>
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="dfr-vote__axis" aria-hidden="true">
        {[0, 1, 2, 3].map((tick) => (
          <span key={tick} style={{ left: at(tick) }}>
            {tick}
          </span>
        ))}
        <em>distance from the new listing →</em>
      </div>
    </div>
  );
}
