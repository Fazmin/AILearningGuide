import type { ReactNode } from "react";
import { GoalMeter, Metric, RangeControl, SegmentedControl, type ModuleContext } from "@app/module-sdk";
import {
  FULL_TABLE,
  IDENTITY,
  PIECES,
  PRESETS,
  ROUNDS,
  SEQUENCE_IDS,
  TABLE_SIZE,
  WORDS,
  add,
  aimMiss,
  aimSolved,
  apply,
  closestPair,
  columns,
  compose,
  determinant,
  fmt,
  matchingPreset,
  orderGap,
  pair,
  pieceTable,
  presetById,
  rebuildTable,
  scale,
  squeezeMatrix,
  storedNumbers,
  tableMiss,
  undoSqueeze,
  vecOf,
  wordById,
  type Mat,
  type Vec,
} from "./math";
import { Arrow, MovedGrid, MovedSquare, Plane, PlaneLabel, WordMark, labelSpot, useGlide, type Geometry } from "./plane";
import { RANGES, type LabState } from "./state";
import { WordPicker, followWord } from "./vector-cards";

type SetState = ModuleContext["setState"];

const CELL_NAMES = ["Top left", "Top right", "Bottom left", "Bottom right"] as const;

/**
 * The four numbers of a matrix as four sliders, laid out the way the matrix is written. The left
 * pair is the first column (where the size arrow lands) and the right pair is the second.
 */
export function MatrixSliders({ m, onChange }: { m: Mat; onChange: (index: number, value: number) => void }) {
  const name = (index: number) => CELL_NAMES[index];
  return (
    <div className="vm-matrix">
      <p className="vm-matrix__head is-size">
        <b>First column</b>
        <span>where the size arrow lands</span>
      </p>
      <p className="vm-matrix__head is-alive">
        <b>Second column</b>
        <span>where the alive arrow lands</span>
      </p>
      {[0, 1, 2, 3].map((index) => (
        <div key={index} className={`vm-matrix__cell ${index % 2 === 0 ? "is-size" : "is-alive"}`}>
          <RangeControl
            label={name(index)}
            min={RANGES.matrix[0]}
            max={RANGES.matrix[1]}
            step={0.1}
            value={m[index]}
            format={(v) => fmt(v)}
            onChange={(value) => onChange(index, value)}
          />
        </div>
      ))}
    </div>
  );
}

/** A matrix written out as numbers, with its two columns tinted like the arrows they describe. */
export function MatrixBox({ m, caption, tone = "" }: { m: Mat; caption: ReactNode; tone?: string }) {
  // A product such as 1.5 × 0.5 = 0.75 needs its second decimal: rounding it to 0.8 would show a different matrix.
  const digits = m.some((value) => Math.abs(value * 10 - Math.round(value * 10)) > 1e-9) ? 2 : 1;
  return (
    <figure
      className={`vm-box ${tone}`.trim()}
      aria-label={`Matrix: top row ${fmt(m[0], digits)} and ${fmt(m[1], digits)}, bottom row ${fmt(m[2], digits)} and ${fmt(
        m[3],
        digits,
      )}`}
    >
      <div className="vm-box__grid" aria-hidden="true">
        <span className="is-size">{fmt(m[0], digits)}</span>
        <span className="is-alive">{fmt(m[1], digits)}</span>
        <span className="is-size">{fmt(m[2], digits)}</span>
        <span className="is-alive">{fmt(m[3], digits)}</span>
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

/**
 * Thin lines from a word's home to where it has landed, with a hollow mark left at home. Pass
 * `only` to draw one word's trail: after a big move, eight crossing trails are a tangle.
 */
function Trails({ geo, m, only }: { geo: Geometry; m: Mat; only?: string }) {
  return (
    <g>
      {WORDS.filter((word) => !only || word.id === only).map((word) => {
        const home = geo.at(vecOf(word));
        const now = geo.at(apply(m, vecOf(word)));
        return (
          <g key={word.id}>
            <line className="vm-trail" x1={home.x} y1={home.y} x2={now.x} y2={now.y} />
            <WordMark geo={geo} word={word} at={vecOf(word)} tone="is-ghost" showLabel={false} />
          </g>
        );
      })}
    </g>
  );
}

/** A label for a column arrow, placed just past its tip. */
function TipLabel({ geo, to, className, children }: { geo: Geometry; to: Vec; className: string; children: ReactNode }) {
  const p = geo.at(to);
  const o = geo.at([0, 0]);
  const span = Math.hypot(p.x - o.x, p.y - o.y) || 1;
  const ux = (p.x - o.x) / span;
  const uy = (p.y - o.y) / span;
  // A sideways arrow runs along an axis line, so its label is lifted clear of the line.
  const spot = labelSpot(geo, p.x + ux * 11, p.y + uy * 11 + (Math.abs(uy) < 0.3 ? -6 : uy > 0 ? 10 : 2));
  const anchor = ux > 0.7 ? "start" : ux < -0.7 ? "end" : "middle";
  if (span < 6) return null;
  return (
    <PlaneLabel geo={geo} className={className} x={spot.x} y={spot.y} anchor={spot.anchor === "middle" ? anchor : spot.anchor}>
      {children}
    </PlaneLabel>
  );
}

/* ── 4 · A matrix moves every arrow at once ── */

export function MatrixCard({ s, setState, narrate }: { s: LabState; setState: SetState; narrate: (text: string) => void }) {
  const m = s.matrix;
  const shown = useGlide(m);
  const word = wordById(s.word);
  const v = vecOf(word);
  const landed = apply(m, v);
  const [first, second] = columns(m);
  const preset = matchingPreset(m);
  const keys = ["ma", "mb", "mc", "md"] as const;
  return (
    <>
      <WordPicker label="Word to follow" value={s.word} onChange={(id) => followWord(setState, id)} />
      <div className="vm-stage">
        <Plane
          quiet
          label={`A matrix with top row ${fmt(m[0])}, ${fmt(m[1])} and bottom row ${fmt(m[2])}, ${fmt(
            m[3],
          )} moves the whole map. The size arrow lands at ${pair(first)} and the alive arrow at ${pair(second)}. ${
            word.id
          } moves from ${pair(v)} to ${pair(landed)}.`}
        >
          {(geo) => {
            const [c1, c2] = columns(shown);
            const leg = scale(c1, v[0]);
            const end = add(leg, scale(c2, v[1]));
            return (
              <>
                <MovedGrid geo={geo} m={shown} />
                <MovedSquare geo={geo} m={shown} />
                <Arrow geo={geo} to={[1, 0]} className="is-ghost" />
                <Arrow geo={geo} to={[0, 1]} className="is-ghost" />
                <Trails geo={geo} m={shown} only={word.id} />
                {WORDS.map((item) => (
                  <WordMark
                    key={item.id}
                    geo={geo}
                    word={item}
                    at={apply(shown, vecOf(item))}
                    tone={item.id === word.id ? "is-picked" : ""}
                  />
                ))}
                <Arrow geo={geo} to={c1} className="is-size is-bold" head={11} />
                <Arrow geo={geo} to={c2} className="is-alive is-bold" head={11} />
                <Arrow geo={geo} to={leg} className="is-path" head={7} />
                <Arrow geo={geo} from={leg} to={end} className="is-path" head={7} />
                <TipLabel geo={geo} to={c1} className="is-size">
                  1st column
                </TipLabel>
                <TipLabel geo={geo} to={c2} className="is-alive">
                  2nd column
                </TipLabel>
              </>
            );
          }}
        </Plane>
        <div className="vm-side">
          <div className="vm-wrap vm-wrap--three">
            <SegmentedControl
              label="Ready-made matrices"
              value={preset?.id ?? ""}
              options={PRESETS.map((item) => ({ value: item.id, label: item.label }))}
              onChange={(value) => {
                const next = presetById(value);
                setState({ ma: next.matrix[0], mb: next.matrix[1], mc: next.matrix[2], md: next.matrix[3] });
                narrate(`Matrix set to ${next.label}: ${next.says}.`);
              }}
            />
          </div>
          <div className="vm-target">
            <MatrixSliders m={m} onChange={(index, value) => setState({ [keys[index]]: value })} />
          </div>
          <p className="vm-effect" role="status">
            {preset ? `This is “${preset.label}”: ${preset.says}.` : "A matrix of your own."} The shaded square started
            as one grid square; its area is now {fmt(Math.abs(determinant(m)), 2)}.
          </p>
        </div>
      </div>
      <table className="vm-worked">
        <caption>
          Where {word.id} {pair(v)} lands, one output number at a time
        </caption>
        <tbody>
          <tr>
            <th scope="row">new size</th>
            <td>
              top row · {word.id} = <b className="is-size">{fmt(m[0])}</b> × {fmt(v[0])} +{" "}
              <b className="is-alive">{fmt(m[1])}</b> × {fmt(v[1])}
            </td>
            <td className="vm-worked__result">= {fmt(landed[0], 2)}</td>
          </tr>
          <tr>
            <th scope="row">new alive</th>
            <td>
              bottom row · {word.id} = <b className="is-size">{fmt(m[2])}</b> × {fmt(v[0])} +{" "}
              <b className="is-alive">{fmt(m[3])}</b> × {fmt(v[1])}
            </td>
            <td className="vm-worked__result">= {fmt(landed[1], 2)}</td>
          </tr>
          <tr className="is-columns">
            <th scope="row">by columns</th>
            <td>
              {fmt(v[0])} × <b className="is-size">1st column {pair(first)}</b> + {fmt(v[1])} ×{" "}
              <b className="is-alive">2nd column {pair(second)}</b>
            </td>
            <td className="vm-worked__result">= {pair(landed, 2)}</td>
          </tr>
        </tbody>
      </table>
      <p className="vm-note">
        The two thick arrows are the matrix’s columns; the grey dashed arrows show where they started. The hollow
        mark is where {word.id} started. The black dashed path walks {fmt(v[0])} of the first column, then{" "}
        {fmt(v[1])} of the second, and ends on {word.id}.
      </p>
    </>
  );
}

/* ── 5 · One after another ── */

export function OrderCard({ s, setState, narrate }: { s: LabState; setState: SetState; narrate: (text: string) => void }) {
  const first = presetById(s.first);
  const second = presetById(s.second);
  const both = compose(second.matrix, first.matrix);
  const swapped = compose(first.matrix, second.matrix);
  const target = s.stage === 0 ? IDENTITY : s.stage === 1 ? first.matrix : both;
  const shown = useGlide(target);
  const gap = orderGap(both, swapped);
  const sequence = SEQUENCE_IDS.map((id) => presetById(id));
  const [bothFirst] = columns(both);
  const midway = apply(first.matrix, [1, 0]);
  return (
    <>
      <div className="vm-stage">
        <Plane
          quiet
          label={`Step 1 is ${first.label} and step 2 is ${second.label}. ${
            s.stage === 0 ? "The map is at its start." : s.stage === 1 ? "The map is shown after step 1." : "The map is shown after both steps."
          } Swapping the order moves ${gap.differ} of 8 words to a different place.`}
        >
          {(geo) => (
            <>
              <MovedGrid geo={geo} m={shown} />
              {s.stage === 2 && <MovedSquare geo={geo} m={swapped} className="is-other" />}
              <MovedSquare geo={geo} m={shown} />
              {s.stage === 2 &&
                WORDS.map((item) => {
                  const here = geo.at(apply(both, vecOf(item)));
                  const there = geo.at(apply(swapped, vecOf(item)));
                  return (
                    <g key={item.id}>
                      <line className="vm-trail" x1={here.x} y1={here.y} x2={there.x} y2={there.y} />
                      <WordMark geo={geo} word={item} at={apply(swapped, vecOf(item))} tone="is-ghost" showLabel={false} />
                    </g>
                  );
                })}
              {WORDS.map((item) => (
                <WordMark key={item.id} geo={geo} word={item} at={apply(shown, vecOf(item))} />
              ))}
              <Arrow geo={geo} to={columns(shown)[0]} className="is-size is-bold" head={11} />
              <Arrow geo={geo} to={columns(shown)[1]} className="is-alive is-bold" head={11} />
            </>
          )}
        </Plane>
        <div className="vm-side">
          <div className="vm-target">
            <div className="vm-wrap vm-wrap--two">
              <SegmentedControl
                label="Step 1"
                value={first.id}
                options={sequence.map((item) => ({ value: item.id, label: item.label }))}
                onChange={(value) => setState({ first: value, stage: 2 })}
              />
            </div>
            <div className="vm-wrap vm-wrap--two">
              <SegmentedControl
                label="Step 2"
                value={second.id}
                options={sequence.map((item) => ({ value: item.id, label: item.label }))}
                onChange={(value) => setState({ second: value, stage: 2 })}
              />
            </div>
            <button
              type="button"
              className="vm-button"
              onClick={() => {
                setState({ first: second.id, second: first.id, stage: 2 });
                narrate(`Order swapped: ${second.label} first, then ${first.label}.`);
              }}
            >
              ⇄ Swap the order
            </button>
          </div>
          <SegmentedControl
            label="Show the map"
            value={String(s.stage)}
            options={[
              { value: "0", label: "at the start" },
              { value: "1", label: "after step 1" },
              { value: "2", label: "after both" },
            ]}
            onChange={(value) => setState({ stage: Number(value) })}
          />
          <div className="metric-row vm-metrics">
            <Metric label="Words that land elsewhere if you swap" value={`${gap.differ} of 8`} />
            <Metric
              label="Widest gap between the two orders"
              value={gap.differ === 0 ? "0.00" : `${gap.widestWord.id}, ${fmt(gap.widest, 2)}`}
            />
          </div>
        </div>
      </div>
      <div className="vm-chain">
        <MatrixBox m={first.matrix} caption={<>step 1: {first.label}</>} />
        <span className="vm-chain__op">then</span>
        <MatrixBox m={second.matrix} caption={<>step 2: {second.label}</>} />
        <span className="vm-chain__op">=</span>
        <MatrixBox m={both} caption="one matrix that does both" tone="is-result" />
        <span className="vm-chain__op vm-chain__op--gap">swapped order gives</span>
        <MatrixBox m={swapped} caption={<>{second.label}, then {first.label}</>} tone="is-other" />
      </div>
      <p className="vm-note">
        The first column of the single matrix is where the size arrow ends up: (1, 0) goes to {pair(midway)} in
        step 1, then to {pair(bothFirst)} in step 2. The dashed outline and hollow marks show the swapped order.
        {gap.differ === 0 ? " For this pair the order makes no difference." : ""}
      </p>
    </>
  );
}

/* ── 6 · When a matrix forgets ── */

export function ForgetCard({ s, setState }: { s: LabState; setState: SetState }) {
  const squeeze = squeezeMatrix("alive", s.keep);
  const undo = undoSqueeze("alive", s.keep);
  const trying = s.undo === "stretch";
  const restored = trying && undo !== null;
  const final: Mat = restored ? IDENTITY : squeeze;
  const shown = useGlide(final);
  const closest = closestPair(final);
  const [one, two] = closest.pair;
  const spot = apply(final, vecOf(one));
  const area = Math.abs(determinant(final));
  const message = restored
    ? `Multiplying alive by 1 ÷ ${fmt(s.keep, 2)} = ${fmt(1 / s.keep, 2)} puts every word back. Nothing was lost.`
    : trying
      ? "Stretching back would need 1 ÷ 0, and there is no such number: 0 times anything is still 0."
      : closest.merged
        ? `${one.id} and ${two.id} now have the same list, ${pair(spot)}. What told them apart is gone.`
        : s.keep < 1
          ? `Every word still has its own spot, so a stretch could undo this. ${one.id} and ${two.id} are closest.`
          : "Nothing has moved yet. Slide left to squeeze.";
  return (
    <>
      <div className="vm-stage">
        <Plane
          quiet
          label={`Every alive number is multiplied by ${fmt(s.keep, 2)}${
            restored ? " and then stretched back" : ""
          }. The unit square has area ${fmt(area, 2)}. ${closest.distinct} of 8 words have their own spot. ${message}`}
        >
          {(geo) => {
            const flag = geo.at(spot);
            // The area label sits inside the square while it is tall enough, and under the axis once it is flat.
            const tall = Math.abs(shown[3]) >= 0.35;
            const areaAt = tall ? geo.at([0.5, shown[3] / 2]) : geo.at([0.5, 0]);
            return (
              <>
                <MovedGrid geo={geo} m={shown} />
                <MovedSquare geo={geo} m={shown} />
                <Trails geo={geo} m={shown} />
                {WORDS.map((item) => (
                  <WordMark
                    key={item.id}
                    geo={geo}
                    word={item}
                    at={apply(shown, vecOf(item))}
                    tone={closest.merged && (item.id === one.id || item.id === two.id) ? "is-flag" : ""}
                  />
                ))}
                {closest.merged && <circle className="vm-ring is-flag" cx={flag.x} cy={flag.y} r="13" />}
                <PlaneLabel geo={geo} className="is-accent" x={areaAt.x} y={areaAt.y + (tall ? 4 : 31)}>
                  area {fmt(area, 2)}
                </PlaneLabel>
              </>
            );
          }}
        </Plane>
        <div className="vm-side">
          <div className="vm-target">
            <RangeControl
              label="Multiply every alive number by"
              min={RANGES.keep[0]}
              max={RANGES.keep[1]}
              step={0.05}
              value={s.keep}
              format={(v) => `× ${fmt(v, 2)}`}
              onChange={(value) => setState({ keep: value })}
            />
          </div>
          <MatrixBox m={squeeze} caption="the squeezing matrix" />
          <SegmentedControl
            label="Then"
            value={s.undo}
            options={[
              { value: "leave", label: "leave it" },
              { value: "stretch", label: "stretch it back" },
            ]}
            onChange={(value) => setState({ undo: value })}
          />
          <div className="metric-row vm-metrics">
            <Metric label="Area of the shaded square" value={fmt(area, 2)} tone={area < 1e-9 ? "loss" : undefined} />
            <Metric
              label="Words with their own spot"
              value={`${closest.distinct} of 8`}
              tone={closest.distinct < 8 ? "loss" : undefined}
            />
          </div>
          <div className="metric-row vm-metrics">
            <Metric
              label="Closest pair"
              value={
                closest.merged ? `${one.id} = ${two.id}: same spot` : `${one.id} and ${two.id}, ${fmt(closest.gap, 2)} apart`
              }
              tone={closest.merged ? "loss" : undefined}
            />
          </div>
        </div>
      </div>
      <p className={`vm-callout ${closest.merged ? "is-loss" : ""}`.trim()} role="status">
        {message}
      </p>
    </>
  );
}

/* ── 7 · A table from a few pieces ── */

const cellTone = (value: number, max: number) => {
  const level = Math.min(4, Math.ceil((Math.abs(value) / max) * 4));
  return value === 0 ? "is-zero" : `${value > 0 ? "is-plus" : "is-minus"} is-l${level}`;
};

function NumberTable({ cells, caption, max }: { cells: number[][]; caption: ReactNode; max: number }) {
  return (
    <figure className="vm-table">
      <figcaption>{caption}</figcaption>
      <table>
        <tbody>
          {cells.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j} className={cellTone(value, max)}>
                  {value < 0 ? `−${-value}` : value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

function PieceFigure({ index, kept }: { index: number; kept: boolean }) {
  const part = PIECES[index];
  const cells = pieceTable(part);
  return (
    <figure
      className={`vm-piece ${kept ? "is-kept" : "is-dropped"}`}
      aria-label={`Piece ${index + 1}: one column of eight numbers, each plus or minus ${part.strength}, times one row of eight plus and minus ones. ${
        kept ? "Kept." : "Left out."
      }`}
    >
      <div className="vm-piece__grid" aria-hidden="true">
        <span className="vm-piece__corner">×</span>
        {part.row.map((value, j) => (
          <span key={`r${j}`} className="vm-piece__row">
            {value > 0 ? "+" : "−"}
          </span>
        ))}
        {cells.map((row, i) => [
          <span key={`c${i}`} className="vm-piece__col">
            {part.column[i] > 0 ? part.column[i] : `−${-part.column[i]}`}
          </span>,
          ...row.map((value, j) => <i key={`${i}-${j}`} className={value > 0 ? "is-plus" : "is-minus"} />),
        ])}
      </div>
      <figcaption>
        <b>piece {index + 1}</b> · cells are ±{part.strength}
        <span>{kept ? "kept" : "left out"}</span>
      </figcaption>
    </figure>
  );
}

export function PiecesCard({ s, setState }: { s: LabState; setState: SetState }) {
  const kept = s.pieces;
  const rebuilt = rebuildTable(kept);
  const miss = tableMiss(kept);
  const stored = storedNumbers(kept);
  const full = TABLE_SIZE * TABLE_SIZE;
  return (
    <>
      <div
        className="vm-pieces"
        role="group"
        aria-label="The four pieces the table is built from. Each is one column times one row."
      >
        {PIECES.map((_, index) => (
          <PieceFigure key={index} index={index} kept={index < kept} />
        ))}
      </div>
      <div className="vm-target vm-pieces__control">
        <RangeControl
          label="Pieces kept"
          min={1}
          max={PIECES.length}
          step={1}
          value={kept}
          format={(v) => `${v} of ${PIECES.length}`}
          onChange={(value) => setState({ pieces: value })}
        />
      </div>
      <div className="vm-tables">
        <NumberTable cells={FULL_TABLE} max={15} caption={<>The full table: {full} numbers</>} />
        <NumberTable
          cells={rebuilt}
          max={15}
          caption={
            <>
              Rebuilt from {kept} {kept === 1 ? "piece" : "pieces"}: {stored} numbers
            </>
          }
        />
      </div>
      <div className="metric-row vm-metrics">
        <Metric label="Numbers stored" value={`${kept} × (8 + 8) = ${stored}`} tone="forward" />
        <Metric label="Against the full table" value={`${full} (${Math.round((stored / full) * 100)}%)`} />
        <Metric label="Average miss per cell" value={fmt(miss.average, 2)} tone={miss.average > 0 ? "loss" : undefined} />
        <Metric label="Largest miss" value={fmt(miss.largest, 0)} />
      </div>
      <p className="vm-note">
        Each piece is a multiplication table: a cell is its column number times its row sign. Blue cells are
        positive; orange hatched cells are negative. The full table is the four pieces added together, cell by
        cell. The pieces were written by hand for this lab so the sums come out as whole numbers.
      </p>
    </>
  );
}

/* ── 8 · Challenge: aim the map ── */

export function AimCard({ s, setState, narrate }: { s: LabState; setState: SetState; narrate: (text: string) => void }) {
  const round = ROUNDS[s.round];
  const m = s.aim;
  const shown = useGlide(m);
  const miss = aimMiss(m, round.target);
  const start = aimMiss(IDENTITY, round.target).average;
  const done = aimSolved(m, round.target);
  const solvedCount = ROUNDS.filter((_, index) => (s.solved >> index) & 1).length;
  const allDone = solvedCount === ROUNDS.length;
  const keys = ["ca", "cb", "cc", "cd"] as const;
  const [first, second] = columns(m);
  const [goalFirst, goalSecond] = columns(round.target);
  const change = (index: number, value: number) => {
    const next = m.map((entry, i) => (i === index ? value : entry)) as unknown as Mat;
    const patch: Record<string, number> = { [keys[index]]: value };
    if (aimSolved(next, round.target) && !((s.solved >> s.round) & 1)) {
      patch.solved = s.solved | (1 << s.round);
      narrate(`${round.title} solved: all eight words are on their targets.`);
    }
    setState(patch);
  };
  const goTo = (index: number) => setState({ round: index, ca: 1, cb: 0, cc: 0, cd: 1, hint: 0 });
  return (
    <>
      <div className="vm-wrap vm-wrap--rounds">
        <SegmentedControl
          label={`Round (${solvedCount} of ${ROUNDS.length} solved)`}
          value={String(s.round)}
          options={ROUNDS.map((item, index) => ({
            value: String(index),
            label: `${item.title.replace("Round ", "")}${(s.solved >> index) & 1 ? " ✓" : ""}`,
          }))}
          onChange={(value) => goTo(Number(value))}
        />
      </div>
      <div className="vm-stage">
        <Plane
          quiet
          label={`${round.title}. Dashed hollow marks are the targets. Your matrix has top row ${fmt(m[0])}, ${fmt(
            m[1],
          )} and bottom row ${fmt(m[2])}, ${fmt(m[3])}. The words are on average ${fmt(miss.average, 2)} from their targets.`}
        >
          {(geo) => (
            <>
              <MovedGrid geo={geo} m={shown} />
              <MovedSquare geo={geo} m={shown} />
              {s.hint && (
                <>
                  <Arrow geo={geo} to={goalFirst} className="is-size is-hint" />
                  <Arrow geo={geo} to={goalSecond} className="is-alive is-hint" />
                </>
              )}
              {WORDS.map((item) => {
                const goal = apply(round.target, vecOf(item));
                const here = geo.at(apply(shown, vecOf(item)));
                const there = geo.at(goal);
                return (
                  <g key={item.id}>
                    <line className="vm-miss" x1={here.x} y1={here.y} x2={there.x} y2={there.y} />
                    <WordMark geo={geo} word={item} at={goal} tone="is-target" showLabel={false} />
                  </g>
                );
              })}
              {WORDS.map((item) => (
                <WordMark key={item.id} geo={geo} word={item} at={apply(shown, vecOf(item))} tone={done ? "is-picked" : ""} />
              ))}
              <Arrow geo={geo} to={columns(shown)[0]} className="is-size is-bold" head={11} />
              <Arrow geo={geo} to={columns(shown)[1]} className="is-alive is-bold" head={11} />
            </>
          )}
        </Plane>
        <div className="vm-side">
          <div className="vm-target">
            <MatrixSliders m={m} onChange={change} />
          </div>
          <p className="vm-effect">
            Your columns: {pair(first)} and {pair(second)}.
            {s.hint ? ` Dashed arrows show where they should land: ${pair(goalFirst)} and ${pair(goalSecond)}.` : ""}
          </p>
          <div className="vm-actions">
            <button type="button" className="vm-button" onClick={() => setState({ hint: s.hint ? 0 : 1 })} aria-pressed={s.hint}>
              {s.hint ? "Hide the hint" : "Hint: show where the two axis arrows should land"}
            </button>
            <button type="button" className="vm-button" onClick={() => setState({ ca: 1, cb: 0, cc: 0, cd: 1 })}>
              Start this round again
            </button>
          </div>
        </div>
      </div>
      <GoalMeter
        label="Average distance from the targets"
        value={start > 0 ? 1 - miss.average / start : 1}
        target={1}
        display={`${fmt(miss.average, 2)} to go (started at ${fmt(start, 2)})`}
        done={done}
        doneText="All eight words on target"
        hint="Move one slider and watch this number. If it falls, keep going; if it rises, go back the other way."
      />
      {done && (
        <div className="vm-win" role="status">
          <p>
            <strong>{round.solvedText}</strong>{" "}
            {allDone
              ? "That is all three. You found each matrix by nudging numbers until inputs landed on targets, which is what training a layer does, with millions of numbers and a rule that picks every nudge for you."
              : "You found a matrix that moves inputs onto targets. Training a layer is this same search."}
          </p>
          {s.round < ROUNDS.length - 1 && (
            <button type="button" className="vm-button is-primary" onClick={() => goTo(s.round + 1)}>
              Next round →
            </button>
          )}
        </div>
      )}
    </>
  );
}
