import type { CSSProperties } from "react";
import { TINY_VOCAB } from "@app/module-sdk";
import { pairLabel } from "./step";

const glyph = (id: number) => (TINY_VOCAB[id] === " " ? "␣" : TINY_VOCAB[id]);
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Dataset stage: the first characters of the text, and the overlapping pairs they are cut into. */
export function DatasetFigure({ ids, pairs }: { ids: ReadonlyArray<number>; pairs: number }) {
  const shown = ids.slice(0, 12);
  return (
    <div
      className="tr-fig tr-fig--dataset"
      role="img"
      aria-label={`The text starts ${shown.map(glyph).join(" ")}. Each character and the one after it make a training pair: ${shown
        .slice(0, 4)
        .map((id, index) => pairLabel(id, shown[index + 1]))
        .join(", ")}, and so on, ${pairs} pairs in all.`}
    >
      <div className="tr-fig__row" aria-hidden="true">
        <span className="tr-fig__tag">text</span>
        <ol className="tr-chars">
          {shown.map((id, index) => (
            <li key={index}>{glyph(id)}</li>
          ))}
          <li className="is-more">…</li>
        </ol>
      </div>
      <div className="tr-fig__row" aria-hidden="true">
        <span className="tr-fig__tag">pairs</span>
        <ol className="tr-chars tr-chars--pairs">
          {shown.slice(0, -1).map((id, index) => (
            <li key={index}>{pairLabel(id, shown[index + 1])}</li>
          ))}
          <li className="is-more">… {pairs} in all</li>
        </ol>
      </div>
      <p aria-hidden="true">Each pair sits under the two characters it joins: context first, then the character to predict.</p>
    </div>
  );
}

/** Batch stage: one epoch as a bar cut into batches, and the epochs as repeated passes. */
export function BatchFigure({
  pairs,
  batchSize,
  batchesPerEpoch,
  epochs,
  steps,
}: {
  pairs: number;
  batchSize: number;
  batchesPerEpoch: number;
  epochs: number;
  steps: number;
}) {
  const slice = (batchSize / pairs) * 100;
  return (
    <div
      className="tr-fig tr-fig--batch"
      role="img"
      aria-label={`One epoch is ${pairs} pairs, shuffled and cut into ${batchesPerEpoch} batches of ${batchSize}. ${epochs} epochs make ${steps} optimizer steps.`}
    >
      <div className="tr-fig__ends" aria-hidden="true">
        <span>
          <i className="tr-fig__swatch" /> one batch = {batchSize} pairs = one optimizer step
        </span>
        <span>
          one epoch = {pairs} pairs → {batchesPerEpoch} batches
        </span>
      </div>
      <div className="tr-epochbar" aria-hidden="true" style={{ "--slice": `${slice}%` } as CSSProperties}>
        <i />
      </div>
      <div className="tr-fig__ends" aria-hidden="true">
        <span>
          × {epochs} {epochs === 1 ? "epoch" : "epochs"}, reshuffled each time
        </span>
        <span>= {steps} steps</span>
      </div>
      <div className="tr-epochs" aria-hidden="true">
        {Array.from({ length: epochs }, (_, index) => (
          <i key={index} />
        ))}
      </div>
    </div>
  );
}

/** Forward stage: the whole 30-way distribution for one context, with the favourite and the true next character marked. */
export function ForwardFigure({
  probabilities,
  target,
  context,
}: {
  probabilities: ReadonlyArray<number>;
  target: number;
  context: string;
}) {
  const W = 600;
  const H = 126;
  const base = 96;
  const slot = W / probabilities.length;
  const peak = Math.max(1e-6, ...probabilities);
  const top = probabilities.indexOf(peak);
  const show = (value: number) => `${(value * 100).toFixed(1)}%`;
  const label = (id: number, text: string) => {
    const x = id * slot + slot / 2;
    const anchor = x < 70 ? "start" : x > W - 70 ? "end" : "middle";
    return (
      <text
        className="tr-prob-value"
        x={anchor === "middle" ? x : anchor === "start" ? x - slot / 2 + 3 : x + slot / 2 - 3}
        y={Math.max(10, base - (probabilities[id] / peak) * 70 - 5)}
        textAnchor={anchor}
      >
        {text}
      </text>
    );
  };
  return (
    <figure className="tr-gradient tr-fig--forward">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`The model's 30 next-character probabilities after "${context}". The favourite is ${glyph(top)} at ${show(peak)}; the character that really comes next, ${glyph(target)}, gets ${show(probabilities[target])}.`}
      >
        <line className="tr-grid" x1={0} x2={W} y1={base} y2={base} />
        {probabilities.map((value, id) => {
          const height = Math.max(0.8, (value / peak) * 70);
          return (
            <g key={id}>
              <rect
                className={`tr-prob-bar${id === top ? " is-top" : ""}${id === target ? " is-target" : ""}`}
                x={id * slot + 3}
                y={base - height}
                width={slot - 6}
                height={height}
              />
              <text className={`tr-grad-label${id === target ? " is-target" : ""}`} x={id * slot + slot / 2} y={H - 14} textAnchor="middle">
                {glyph(id)}
              </text>
            </g>
          );
        })}
        {label(top, `favourite ${show(peak)}`)}
        {top !== target && label(target, `real next: ${show(probabilities[target])}`)}
        <path
          className="tr-prob-mark"
          d={`M${target * slot + slot / 2 - 5},${H - 2} L${target * slot + slot / 2},${H - 9} L${target * slot + slot / 2 + 5},${H - 2} Z`}
        />
      </svg>
      <figcaption>
        All 30 probabilities after <code>{context}</code>, one bar per character; they add up to 1. The outlined bar and the
        marker under it are <code>{glyph(target)}</code>, the character the text really continues with.
      </figcaption>
    </figure>
  );
}

/** Loss stage: this checkpoint against the knows-nothing baseline on one scale, with a tick at every halving. */
export function LossFigure({ loss, baseline }: { loss: number; baseline: number }) {
  const max = baseline * 1.04;
  const at = (value: number) => `${clamp(value / max, 0, 1) * 100}%`;
  const doublings = [1, 2, 3, 4].map((count) => ({ count, value: baseline - count * Math.LN2 })).filter((tick) => tick.value > 0);
  return (
    <div
      className="tr-fig tr-fig--loss"
      role="img"
      aria-label={`Loss at this checkpoint is ${loss.toFixed(3)} nats per token against ${baseline.toFixed(3)} for a model that knows nothing. Every ${Math.LN2.toFixed(2)} nats lower means the right character got twice the probability.`}
    >
      <div className="tr-lossbar" aria-hidden="true">
        <span>knows nothing</span>
        <i>
          <b className="is-baseline" style={{ width: at(baseline) }} />
        </i>
        <strong>{baseline.toFixed(3)}</strong>
      </div>
      <div className="tr-lossbar" aria-hidden="true">
        <span>this checkpoint</span>
        <i>
          <b style={{ width: at(loss) }} />
        </i>
        <strong>{loss.toFixed(3)}</strong>
      </div>
      <div className="tr-lossbar tr-lossbar--ticks" aria-hidden="true">
        <span />
        <i>
          {doublings.map((tick) => (
            <em key={tick.count} style={{ left: at(tick.value) }}>
              {2 ** tick.count}×
            </em>
          ))}
          <em className="is-zero" style={{ left: "0%" }}>
            0
          </em>
        </i>
        <strong />
      </div>
      <p aria-hidden="true">
        Shorter is better. Each tick is 0.69 nats below the baseline: the right character is 2×, 4×, 8× as likely as a blind
        guess.
      </p>
    </div>
  );
}

/** The 30 rows of the weight table, shaded where this batch touched them. */
export function RowsStrip({
  rows,
  size,
}: {
  rows: ReadonlyArray<{ row: number; pairs: number; gradientNorm: number }>;
  size: number;
}) {
  const peak = Math.max(1e-9, ...rows.map((row) => row.gradientNorm));
  const byRow = new Map(rows.map((row) => [row.row, row]));
  return (
    <figure className="tr-rows">
      <figcaption>
        The {size} rows of the weight table, one per previous character. This batch moved <b>{rows.length}</b>; the blank ones
        got no gradient.
      </figcaption>
      <ol
        aria-label={`Rows moved by this batch: ${rows
          .map((row) => `${glyph(row.row)} from ${row.pairs} ${row.pairs === 1 ? "pair" : "pairs"}`)
          .join(", ")}. The other ${size - rows.length} rows did not move.`}
      >
        {Array.from({ length: size }, (_, id) => {
          const hit = byRow.get(id);
          return (
            <li
              key={id}
              aria-hidden="true"
              className={hit ? "is-hit" : ""}
              style={hit ? ({ "--g": clamp(hit.gradientNorm / peak, 0, 1) } as CSSProperties) : undefined}
            >
              <span>{glyph(id)}</span>
              <small>{hit ? hit.pairs : ""}</small>
            </li>
          );
        })}
      </ol>
      <p aria-hidden="true">
        <i /> stronger colour = larger row gradient · the small number is how many of the batch&apos;s pairs start with that character
      </p>
    </figure>
  );
}

/** The model state poured into reference devices: full boxes, then one partly filled. */
export function DeviceBoxes({ total, device, count, deviceLabel }: { total: number; device: number; count: number; deviceLabel: string }) {
  const shown = Math.min(count, 48);
  const full = Math.floor(total / device);
  const rest = total / device - full;
  return (
    <div
      className="tr-devices"
      role="img"
      aria-label={`The model state fills ${count} ${deviceLabel} ${count === 1 ? "device" : "devices"}${
        rest > 0.001 && count > 0 ? `; the last one is ${Math.max(1, Math.round(rest * 100))}% full` : ""
      }.`}
    >
      <div aria-hidden="true">
        {Array.from({ length: shown }, (_, index) => (
          <i key={index} style={{ "--fill": index < full ? 1 : clamp(rest, 0.03, 1) } as CSSProperties} />
        ))}
        {count > shown && <em>+{count - shown} more</em>}
      </div>
      <p aria-hidden="true">
        Each box is one {deviceLabel} device, filled with model state only. Activations would need room on top of this.
      </p>
    </div>
  );
}
