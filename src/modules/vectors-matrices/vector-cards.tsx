import type { CSSProperties } from "react";
import { FormulaWithValues, Metric, RangeControl, SegmentedControl, type ModuleContext } from "@app/module-sdk";
import {
  PUSHES,
  WORDS,
  add,
  angleBetween,
  cosine,
  distance,
  dot,
  fmt,
  kindOf,
  length,
  nearestWord,
  pair,
  pushById,
  rotate,
  scale,
  shadowLength,
  shadowPoint,
  signed,
  vecOf,
  wordById,
  type Vec,
} from "./math";
import { Arrow, DragHandle, Plane, PlaneLabel, WordMark, labelSpot, type Geometry } from "./plane";
import { RANGES, type LabState } from "./state";

type SetState = ModuleContext["setState"];

const snap = (value: number, low: number, high: number) => Math.min(high, Math.max(low, Math.round(value * 10) / 10));

/** A number squared, written with brackets round a negative so the minus sign is clearly inside the square. */
const squared = (value: number) => (Number(value.toFixed(1)) < 0 ? `(${fmt(value)})²` : `${fmt(value)}²`);

/** One button per word. Picking a word here follows it on every card that uses "the word you follow". */
export function WordPicker({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="vm-picker">
      <SegmentedControl
        label={label}
        value={value}
        options={WORDS.map((word) => ({ value: word.id, label: word.id }))}
        onChange={onChange}
      />
    </div>
  );
}

/** Follow a word: every card that reads `word` moves to it, and the first card's arrow snaps back onto it. */
export const followWord = (setState: SetState, id: string) => {
  const word = wordById(id);
  setState({ word: word.id, tipX: word.x, tipY: word.y });
};

/**
 * Where to write the name of an arrow from the centre: a little past its tip, starting beside the
 * tip when the arrow runs mostly sideways and centred over it when the arrow runs mostly up or down.
 */
const beyond = (geo: Geometry, v: Vec, by = 14) => {
  const p = geo.at(v);
  const o = geo.at([0, 0]);
  const span = Math.hypot(p.x - o.x, p.y - o.y) || 1;
  const ux = (p.x - o.x) / span;
  const uy = (p.y - o.y) / span;
  const spot = labelSpot(geo, p.x + ux * by, p.y + uy * by + (uy > 0.4 ? 9 : 4));
  const anchor: "start" | "middle" | "end" = ux > 0.7 ? "start" : ux < -0.7 ? "end" : "middle";
  return { x: spot.x, y: spot.y, anchor: spot.anchor === "middle" ? anchor : spot.anchor };
};

/* ── 1 · A word as two numbers ── */

export function WordCard({ s, setState }: { s: LabState; setState: SetState }) {
  const word = wordById(s.word);
  const tip = s.tip;
  const home = vecOf(word);
  const moved = distance(tip, home) > 1e-9;
  const near = nearestWord(tip);
  const len = length(tip);
  const setTip = (point: Vec) =>
    setState({ tipX: snap(point[0], ...RANGES.tipX), tipY: snap(point[1], ...RANGES.tipY) });
  return (
    <>
      <WordPicker label="Word to read" value={s.word} onChange={(id) => followWord(setState, id)} />
      <div className="vm-stage">
        <Plane
          label={`A map of eight words. The arrow runs from the centre to (${fmt(tip[0])}, ${fmt(tip[1])}): ${fmt(
            Math.abs(tip[0]),
          )} ${tip[0] < 0 ? "left" : "right"} for size and ${fmt(Math.abs(tip[1]))} ${
            tip[1] < 0 ? "down" : "up"
          } for alive. Its length is ${fmt(len, 2)}. The nearest word is ${near.word.id}.`}
        >
          {(geo) => {
            const corner = geo.at([tip[0], 0]);
            const end = geo.at(tip);
            const origin = geo.at([0, 0]);
            const mid = geo.at(scale(tip, 0.5));
            // Screen-space normal to the arrow, pointing away from the triangle's right-angle corner.
            const away = tip[0] * tip[1] >= 0 ? 1 : -1;
            const normal = len > 1e-6 ? [(-tip[1] / len) * away, (-tip[0] / len) * away] : [0, -1];
            const lengthSpot = labelSpot(geo, mid.x + normal[0] * 12, mid.y + normal[1] * 14 + 4);
            const lengthAnchor = normal[0] < -0.2 ? "end" : normal[0] > 0.2 ? "start" : "middle";
            return (
              <>
                {WORDS.map((item) => (
                  <WordMark
                    key={item.id}
                    geo={geo}
                    word={item}
                    at={vecOf(item)}
                    tone={item.id === word.id ? "is-picked" : "is-dim"}
                    lift={item.id === word.id && !moved ? 7 : 0}
                    onPick={() => followWord(setState, item.id)}
                  />
                ))}
                <line className="vm-leg is-size" x1={origin.x} y1={origin.y} x2={corner.x} y2={corner.y} />
                <line className="vm-leg is-alive" x1={corner.x} y1={corner.y} x2={end.x} y2={end.y} />
                {Math.abs(tip[0]) > 0.25 && (
                  <PlaneLabel geo={geo}
                    className="is-size"
                    x={(origin.x + corner.x) / 2}
                    y={origin.y + (tip[1] >= 0 ? 27 : -9)}
                  >
                    size {fmt(tip[0])}
                  </PlaneLabel>
                )}
                {Math.abs(tip[1]) > 0.25 && (
                  <PlaneLabel geo={geo}
                    className="is-alive"
                    x={corner.x + (tip[0] >= 0 ? 8 : -8)}
                    y={(corner.y + end.y) / 2 + 4}
                    anchor={tip[0] >= 0 ? "start" : "end"}
                  >
                    alive {fmt(tip[1])}
                  </PlaneLabel>
                )}
                <Arrow geo={geo} to={tip} className="is-accent is-bold" head={12} />
                {len > 0.5 && (
                  <PlaneLabel geo={geo} className="is-accent" x={lengthSpot.x} y={lengthSpot.y} anchor={lengthAnchor}>
                    length {fmt(len, 2)}
                  </PlaneLabel>
                )}
                <DragHandle geo={geo} at={tip} onMove={setTip} title="Drag the arrow tip" />
              </>
            );
          }}
        </Plane>
        <div className="vm-side">
          <div className="vm-list" aria-label={`The list of numbers for this arrow: ${fmt(tip[0])} and ${fmt(tip[1])}`}>
            <span className="vm-list__name">{moved ? "your arrow" : word.id}</span>
            <span className="vm-list__bracket" aria-hidden="true">
              (
            </span>
            <span className="vm-list__cell is-size">
              <b>{fmt(tip[0])}</b>
              <small>size</small>
            </span>
            <span className="vm-list__comma" aria-hidden="true">
              ,
            </span>
            <span className="vm-list__cell is-alive">
              <b>{fmt(tip[1])}</b>
              <small>alive</small>
            </span>
            <span className="vm-list__bracket" aria-hidden="true">
              )
            </span>
          </div>
          <div className="vm-target">
            <RangeControl
              label="Size (first number)"
              min={RANGES.tipX[0]}
              max={RANGES.tipX[1]}
              step={0.1}
              value={tip[0]}
              format={(v) => fmt(v)}
              onChange={(value) => setState({ tipX: value })}
            />
            <RangeControl
              label="Alive (second number)"
              min={RANGES.tipY[0]}
              max={RANGES.tipY[1]}
              step={0.1}
              value={tip[1]}
              format={(v) => fmt(v)}
              onChange={(value) => setState({ tipY: value })}
            />
          </div>
          <div className="metric-row vm-metrics">
            <Metric label="How much: length" value={fmt(len, 2)} />
            <Metric label="What kind: direction" value={len < 0.05 ? "no direction" : kindOf(tip)} />
          </div>
          <div className="metric-row vm-metrics">
            <Metric
              label="Nearest word on the map"
              value={near.gap < 0.005 ? `${near.word.id} (right on it)` : `${near.word.id}, ${fmt(near.gap, 2)} away`}
            />
          </div>
          {moved && (
            <button type="button" className="vm-button" onClick={() => followWord(setState, word.id)}>
              Put the arrow back on {word.id}
            </button>
          )}
        </div>
      </div>
      <FormulaWithValues
        label="Length, from the two legs of the triangle"
        expression={`√(${squared(tip[0])} + ${squared(tip[1])}) = √(${fmt(tip[0] ** 2, 2)} + ${fmt(tip[1] ** 2, 2)}) = √${fmt(
          tip[0] ** 2 + tip[1] ** 2,
          2,
        )}`}
        result={fmt(len, 2)}
        tone="forward"
        detail="Square each number, add the squares, take the square root. A minus sign disappears when you square, so length is never negative."
      />
      <p className="vm-note">
        Drag the round handle on the arrow tip, or use the two sliders. Round dots are living things and square
        dots are machines. The two numbers per word were made up for this lab; a real model learns hundreds per
        word.
      </p>
    </>
  );
}

/* ── 2 · Add and stretch ── */

export function AddCard({ s, setState }: { s: LabState; setState: SetState }) {
  const word = wordById(s.word);
  const start = vecOf(word);
  const push = pushById(s.push);
  const step = scale(push.vec, s.times);
  const result = add(start, step);
  const near = nearestWord(result);
  const stepName = `${fmt(s.times)} × ${push.label}`;
  // In a sum, a negative multiple reads as a subtraction: "− 1.7 × bigger", not "+ −1.7 × bigger".
  const stepTerm = `${s.times < -0.05 ? "−" : "+"} ${fmt(Math.abs(s.times))} × ${push.label}`;
  const effect =
    Math.abs(s.times) < 0.05
      ? "times 0: the step vanishes"
      : s.times < 0
        ? Math.abs(s.times + 1) < 0.05
          ? "flipped, same length"
          : Math.abs(s.times) < 1
            ? "flipped and shrunk"
            : "flipped and stretched"
        : Math.abs(s.times - 1) < 0.05
          ? "the step as it is"
          : s.times < 1
            ? "shrunk"
            : "stretched";
  return (
    <>
      <WordPicker label="Start from" value={s.word} onChange={(id) => followWord(setState, id)} />
      <div className="vm-stage">
        <Plane
          label={`${word.id} at ${pair(start)} plus ${stepName}, which is ${pair(step)}, lands at ${pair(
            result,
          )}. The nearest word is ${near.word.id}, ${fmt(near.gap, 2)} away.`}
        >
          {(geo) => {
            const o = geo.at([0, 0]);
            const a = geo.at(start);
            const b = geo.at(result);
            const nearAt = geo.at(vecOf(near.word));
            // Screen direction of the step; when the step has no length, carry on along the word's own arrow.
            const stepSpan = Math.hypot(b.x - a.x, b.y - a.y);
            const wordSpan = Math.hypot(a.x - o.x, a.y - o.y) || 1;
            const dir =
              stepSpan > 1
                ? [(b.x - a.x) / stepSpan, (b.y - a.y) / stepSpan]
                : [(a.x - o.x) / wordSpan, (a.y - o.y) / wordSpan];
            const side = (x: number) => (x > 0.4 ? "start" : x < -0.4 ? "end" : "middle") as "start" | "middle" | "end";
            const drop = (y: number) => 4 + (y > 0.4 ? 8 : y < -0.4 ? -3 : 0);
            // The step's label sits beside the step, a third of the way along, on the side away from the centre.
            const stepMid = { x: a.x + (b.x - a.x) * 0.36, y: a.y + (b.y - a.y) * 0.36 };
            let out = [-dir[1], dir[0]];
            if (out[0] * (o.x - stepMid.x) + out[1] * (o.y - stepMid.y) > 0) out = [-out[0], -out[1]];
            const stepSpot = labelSpot(geo, stepMid.x + out[0] * 12, stepMid.y + out[1] * 12 + drop(out[1]));
            // The result's label sits just past the tip, on the side the nearest word's name does not use.
            const resultAbove = b.y <= nearAt.y;
            const resultSpot = labelSpot(geo, b.x + dir[0] * 8, b.y + (resultAbove ? -11 : 21));
            // The word's own name sits beside its arrow, on the side away from the result.
            const wordMid = { x: (o.x + a.x) / 2, y: (o.y + a.y) / 2 };
            let off = [-(a.y - o.y) / wordSpan, (a.x - o.x) / wordSpan];
            if (off[0] * (b.x - wordMid.x) + off[1] * (b.y - wordMid.y) > 0) off = [-off[0], -off[1]];
            const wordSpot = labelSpot(geo, wordMid.x + off[0] * 10, wordMid.y + off[1] * 10 + drop(off[1]));
            return (
              <>
                {WORDS.map((item) => (
                  <WordMark
                    key={item.id}
                    geo={geo}
                    word={item}
                    at={vecOf(item)}
                    tone={item.id === word.id ? "is-picked" : item.id === near.word.id ? "" : "is-dim"}
                    showLabel={item.id !== word.id}
                    labelSide={item.id === near.word.id ? (resultAbove ? "below" : "above") : undefined}
                    lift={item.id === near.word.id ? 6 : 0}
                  />
                ))}
                <circle className="vm-ring" cx={nearAt.x} cy={nearAt.y} r="12" />
                <Arrow geo={geo} from={start} to={add(start, push.vec)} className="is-ghost" />
                <Arrow geo={geo} to={start} className="is-ink is-bold" head={11} />
                <Arrow geo={geo} from={start} to={result} className="is-gradient is-bold" head={11} />
                <Arrow geo={geo} to={result} className="is-accent is-bold" head={12} />
                <PlaneLabel geo={geo} className="is-ink" x={wordSpot.x} y={wordSpot.y} anchor={side(off[0])}>
                  {word.id}
                </PlaneLabel>
                {Math.abs(s.times) > 0.25 && (
                  <PlaneLabel geo={geo} className="is-gradient" x={stepSpot.x} y={stepSpot.y} anchor={side(out[0])}>
                    × {fmt(s.times)}
                  </PlaneLabel>
                )}
                <PlaneLabel geo={geo} className="is-accent" x={resultSpot.x} y={resultSpot.y} anchor={side(dir[0])}>
                  result
                </PlaneLabel>
              </>
            );
          }}
        </Plane>
        <div className="vm-side">
          <div className="vm-wrap">
            <SegmentedControl
              label="Step to add"
              value={push.id}
              options={PUSHES.map((item) => ({ value: item.id, label: `${item.label} ${pair(item.vec, 0)}` }))}
              onChange={(value) => setState({ push: value })}
            />
          </div>
          <div className="vm-target">
            <RangeControl
              label="Times (multiply the step by)"
              min={RANGES.times[0]}
              max={RANGES.times[1]}
              step={0.1}
              value={s.times}
              format={(v) => `× ${fmt(v)}`}
              onChange={(value) => setState({ times: value })}
            />
          </div>
          <p className="vm-effect" role="status">
            {stepName} = {pair(step)}: {effect}.
          </p>
          <table className="vm-sum">
            <caption className="sr-only">The addition, one number at a time</caption>
            <thead>
              <tr>
                <td />
                <th scope="col" className="is-size">
                  size
                </th>
                <th scope="col" className="is-alive">
                  alive
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">{word.id}</th>
                <td>{fmt(start[0])}</td>
                <td>{fmt(start[1])}</td>
              </tr>
              <tr className="is-step">
                <th scope="row">{stepTerm}</th>
                <td>{fmt(step[0])}</td>
                <td>{fmt(step[1])}</td>
              </tr>
              <tr className="is-result">
                <th scope="row">= result</th>
                <td>{fmt(result[0])}</td>
                <td>{fmt(result[1])}</td>
              </tr>
            </tbody>
          </table>
          <div className="metric-row vm-metrics">
            <Metric
              label="The result lands nearest"
              value={near.gap < 0.005 ? `${near.word.id} (right on it)` : `${near.word.id}, ${fmt(near.gap, 2)} away`}
            />
          </div>
        </div>
      </div>
      <p className="vm-note">
        The magenta arrow is the step after multiplying, placed with its tail on the tip of {word.id}; the label
        beside it is the multiplier. The dashed arrow is the same step at × 1, for comparison. The ring marks the
        word nearest the result.
      </p>
    </>
  );
}

/* ── 3 · How alike are two arrows? ── */

/** One signed bar: it grows right of the centre line for a positive number and left for a negative one. */
function SignedBar({
  name,
  working,
  value,
  scaleTo,
  tone,
}: {
  name: string;
  working: string;
  value: number;
  scaleTo: number;
  tone: string;
}) {
  const share = Math.min(1, Math.abs(value) / Math.max(1e-9, scaleTo)) * 50;
  const style: CSSProperties = value >= 0 ? { left: "50%", width: `${share}%` } : { right: "50%", width: `${share}%` };
  return (
    <div className={`vm-bar ${tone}`}>
      <span className="vm-bar__name">{name}</span>
      <span className="vm-bar__working">{working}</span>
      <span className="vm-bar__track" aria-hidden="true">
        <i style={style} className={value < 0 ? "is-negative" : ""} />
      </span>
      <strong className="vm-bar__value">{signed(value)}</strong>
    </div>
  );
}

export function AlikeCard({ s, setState }: { s: LabState; setState: SetState }) {
  const wordA = wordById(s.word);
  const wordB = wordById(s.other);
  const a = vecOf(wordA);
  const homeB = vecOf(wordB);
  const b = scale(rotate(homeB, s.turn), s.stretch);
  const changed = s.turn % 360 !== 0 || Math.abs(s.stretch - 1) > 1e-9;
  const product = dot(a, b);
  const cos = cosine(a, b);
  const angle = angleBetween(a, b);
  const shadow = shadowLength(a, b);
  const foot = shadowPoint(a, b);
  const lenA = length(a);
  const lenB = length(b);
  const scaleTo = lenA * length(homeB) * RANGES.stretch[1];
  const verdict =
    cos > 0.95
      ? "almost the same direction"
      : cos > 0.3
        ? "mostly the same direction"
        : cos >= -0.3
          ? Math.abs(cos) < 0.05
            ? "at right angles: no agreement either way"
            : "close to right angles"
          : cos >= -0.95
            ? "mostly opposite"
            : "almost exactly opposite";
  const nameB = changed ? `${wordB.id}, changed` : wordB.id;
  return (
    <>
      <div className="vm-pickers">
        <WordPicker label="First arrow" value={s.word} onChange={(id) => followWord(setState, id)} />
        <WordPicker
          label="Second arrow"
          value={s.other}
          onChange={(id) => setState({ other: id, turn: 0, stretch: 1 })}
        />
      </div>
      <div className="vm-stage">
        <Plane
          label={`${wordA.id} at ${pair(a)} and the second arrow at ${pair(b, 2)}. The angle between them is ${Math.round(
            angle,
          )} degrees. The shadow of ${wordA.id} on the second arrow is ${fmt(shadow, 2)} long. Dot product ${signed(
            product,
          )}, cosine ${signed(cos)}: ${verdict}.`}
        >
          {(geo) => {
            const far = scale(b, lenB > 1e-6 ? 6 / lenB : 0);
            const lineA = geo.at(scale(far, -1));
            const lineB = geo.at(far);
            const tipA = geo.at(a);
            const footAt = geo.at(foot);
            const origin = geo.at([0, 0]);
            const spotA = beyond(geo, a, 14);
            const spotB = beyond(geo, b, 14);
            const angleA = Math.atan2(a[1], a[0]);
            let sweep = Math.atan2(b[1], b[0]) - angleA;
            while (sweep > Math.PI) sweep -= 2 * Math.PI;
            while (sweep < -Math.PI) sweep += 2 * Math.PI;
            const radius = 0.5;
            const arcEnd = angleA + sweep;
            const r = radius * geo.unit;
            const arc = `M${geo.px(radius * Math.cos(angleA))} ${geo.py(radius * Math.sin(angleA))} A${r} ${r} 0 0 ${
              sweep > 0 ? 0 : 1
            } ${geo.px(radius * Math.cos(arcEnd))} ${geo.py(radius * Math.sin(arcEnd))}`;
            const midAngle = angleA + sweep / 2;
            const shadowMid = geo.at(scale(foot, 0.5));
            const unitB: Vec = lenB > 1e-6 ? [b[0] / lenB, b[1] / lenB] : [1, 0];
            // Put the shadow's label on the far side of the line from the first arrow.
            const away = a[0] * -unitB[1] + a[1] * unitB[0] >= 0 ? -1 : 1;
            const shadowSpot = labelSpot(
              geo,
              shadowMid.x + -unitB[1] * 16 * away,
              shadowMid.y - unitB[0] * 16 * away + 4,
            );
            return (
              <>
                {WORDS.map((item) => (
                  <WordMark
                    key={item.id}
                    geo={geo}
                    word={item}
                    at={vecOf(item)}
                    tone="is-dim"
                    showLabel={item.id !== wordA.id && item.id !== wordB.id}
                  />
                ))}
                <line className="vm-line" x1={lineA.x} y1={lineA.y} x2={lineB.x} y2={lineB.y} />
                {changed && <Arrow geo={geo} to={homeB} className="is-ghost" />}
                <line className="vm-drop" x1={tipA.x} y1={tipA.y} x2={footAt.x} y2={footAt.y} />
                <line className="vm-shadow" x1={origin.x} y1={origin.y} x2={footAt.x} y2={footAt.y} />
                <circle className="vm-shadow__foot" cx={footAt.x} cy={footAt.y} r="3.5" />
                {angle > 6 && lenB > 0.2 && <path className="vm-arc" d={arc} />}
                {angle > 6 && lenB > 0.2 && (
                  <PlaneLabel geo={geo}
                    className="is-quiet"
                    x={geo.px(0.82 * Math.cos(midAngle))}
                    y={geo.py(0.82 * Math.sin(midAngle)) + 4}
                  >
                    {Math.round(angle)}°
                  </PlaneLabel>
                )}
                <Arrow geo={geo} to={b} className="is-ink is-bold" head={11} />
                <Arrow geo={geo} to={a} className="is-accent is-bold" head={11} />
                <PlaneLabel geo={geo} className="is-accent" x={spotA.x} y={spotA.y} anchor={spotA.anchor}>
                  {wordA.id}
                </PlaneLabel>
                <PlaneLabel geo={geo} className="is-ink" x={spotB.x} y={spotB.y} anchor={spotB.anchor}>
                  {nameB}
                </PlaneLabel>
                {Math.abs(shadow) > 0.3 && (
                  <PlaneLabel geo={geo} className="is-gradient" x={shadowSpot.x} y={shadowSpot.y} anchor={shadowSpot.anchor}>
                    shadow {fmt(shadow, 2)}
                  </PlaneLabel>
                )}
              </>
            );
          }}
        </Plane>
        <div className="vm-side">
          <div className="vm-target">
            <RangeControl
              label="Turn the second arrow"
              min={RANGES.turn[0]}
              max={RANGES.turn[1]}
              step={15}
              value={s.turn}
              format={(v) => `${Math.round(v)}°`}
              onChange={(value) => setState({ turn: value })}
            />
          </div>
          <RangeControl
            label="Stretch the second arrow"
            min={RANGES.stretch[0]}
            max={RANGES.stretch[1]}
            step={0.1}
            value={s.stretch}
            format={(v) => `× ${fmt(v)}`}
            onChange={(value) => setState({ stretch: value })}
          />
          <div className="metric-row vm-metrics">
            <Metric label={`Length of ${wordA.id}`} value={fmt(lenA, 2)} />
            <Metric label="Length of second" value={fmt(lenB, 2)} />
          </div>
          <p className="vm-effect" role="status">
            The shadow is how far {wordA.id} reaches along the second arrow. Shadow × length of second ={" "}
            {fmt(shadow, 2)} × {fmt(lenB, 2)} = {signed(shadow * lenB)}: the dot product again.
          </p>
          <div
            className="vm-cosine"
            role="img"
            aria-label={`Cosine similarity ${signed(cos)} on a scale from minus one, opposite, through zero, at right angles, to plus one, same direction: ${verdict}.`}
          >
            <p className="vm-products__title">Cosine: the dot product with both lengths divided out</p>
            <div className="vm-cosine__track" aria-hidden="true">
              <i className="vm-cosine__zero" />
              <b style={{ left: `${((cos + 1) / 2) * 100}%` }}>
                <span>{signed(cos)}</span>
              </b>
            </div>
            <div className="vm-cosine__ends" aria-hidden="true">
              <span>−1 opposite</span>
              <span>0</span>
              <span>+1 same way</span>
            </div>
            <p className="vm-cosine__verdict">{verdict.charAt(0).toUpperCase() + verdict.slice(1)}</p>
          </div>
        </div>
      </div>
      <div
        className="vm-products"
        role="img"
        aria-label={`Size times size is ${signed(a[0] * b[0])}. Alive times alive is ${signed(
          a[1] * b[1],
        )}. They add to the dot product, ${signed(product)}.`}
      >
        <p className="vm-products__title">Multiply the matching numbers, then add: the dot product</p>
        <SignedBar
          name="size × size"
          working={`${fmt(a[0])} × ${fmt(b[0], 2)}`}
          value={a[0] * b[0]}
          scaleTo={scaleTo}
          tone="is-size"
        />
        <SignedBar
          name="alive × alive"
          working={`${fmt(a[1])} × ${fmt(b[1], 2)}`}
          value={a[1] * b[1]}
          scaleTo={scaleTo}
          tone="is-alive"
        />
        <SignedBar name="dot product" working="add the two" value={product} scaleTo={scaleTo} tone="is-sum" />
        <div className="vm-bar__axis" aria-hidden="true">
          <span>← negative: they disagree</span>
          <span>0</span>
          <span>positive: they agree →</span>
        </div>
      </div>
      <FormulaWithValues
        label="Cosine similarity"
        expression={`dot product ÷ (length × length) = ${signed(product)} ÷ (${fmt(lenA, 2)} × ${fmt(lenB, 2)})`}
        result={signed(cos)}
        tone="gradient"
        detail={`${wordA.id} and ${nameB}: ${verdict}. Stretch the second arrow and the dot product changes; the cosine does not.`}
      />
    </>
  );
}
