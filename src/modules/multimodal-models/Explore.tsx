import { useMemo, type CSSProperties } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  boundedTemperature,
  CAPTIONS,
  CHANCE_LOSS,
  contrast,
  IMAGES,
  pairingOf,
  SIMILARITY,
  TEMPERATURE_RANGE,
  TOPICS,
} from "./contrastive";
import {
  CONTEXT,
  DEFAULT_PIXELS,
  FILTER_NAMES,
  filters,
  LLM_WIDTH,
  parsePixels,
  patchify,
  PATCH_SIZES,
  realScale,
  SIDES,
  SIZE,
  VIT_PATCHES,
  type Connector,
  type Patch,
  type Scale,
} from "./patches";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const f2 = (value: number) => (Math.abs(value) < 0.005 ? "0.00" : value.toFixed(2));

const TEXT_BEFORE = ["USER:"];
const TEXT_AFTER = ["Which", "digit", "is", "this", "?", "ASSISTANT:"];

function MiniPatch({ patch, p, className = "" }: { patch: Patch; p: number; className?: string }) {
  return (
    <span className={`mm-mini ${className}`.trim()} style={{ gridTemplateColumns: `repeat(${p}, 1fr)` }} aria-hidden="true">
      {patch.x.map((bit, index) => (
        <i key={index} className={bit ? "is-on" : ""} />
      ))}
    </span>
  );
}

function FilterTile({ weights, p, name, value }: { weights: number[]; p: number; name: string; value: number }) {
  const max = Math.max(...weights.map((weight) => Math.abs(weight)), 1e-9);
  return (
    <figure className="mm-filter">
      <span className="mm-filter-grid" style={{ gridTemplateColumns: `repeat(${p}, 1fr)` }} aria-hidden="true">
        {weights.map((weight, index) => (
          <i
            key={index}
            className={weight > 0 ? "is-pos" : weight < 0 ? "is-neg" : ""}
            style={{ opacity: weight === 0 ? 1 : 0.35 + 0.65 * (Math.abs(weight) / max) }}
          >
            {p <= 4 ? (weight > 0 ? "+" : weight < 0 ? "−" : "") : ""}
          </i>
        ))}
      </span>
      <figcaption>
        <span>{name}</span>
        <code>{f2(value)}</code>
      </figcaption>
    </figure>
  );
}

/** The image square to scale against a 448px outline, with its patch grid, plus the token count it costs. */
function ScalePicture({
  side,
  vitPatch,
  cls,
  connector,
  scale,
}: {
  side: number;
  vitPatch: number;
  cls: boolean;
  connector: Connector;
  scale: Scale;
}) {
  const full = 150;
  const unit = full / SIDES[SIDES.length - 1];
  const size = Math.min(full + 8, scale.padded * unit);
  const cell = size / scale.perSide;
  const lines = Array.from({ length: scale.perSide - 1 }, (_, index) => (index + 1) * cell);
  const bySide = SIDES.map((option) => ({ option, tokens: realScale(option, vitPatch, cls, connector).patchTokens }));
  const most = Math.max(...bySide.map((entry) => entry.tokens));
  const share = scale.contextShare;
  const fmt = (value: number) => value.toLocaleString("en-US");
  return (
    <div className="mm-scale-picture">
      <figure className="mm-scale-figure">
        <svg
          className="mm-scale-grid"
          viewBox="0 0 166 166"
          role="img"
          aria-label={`A ${side} pixel image cut into ${scale.perSide} by ${scale.perSide} patches of ${vitPatch} pixels: ${fmt(scale.patchTokens)} patch tokens. The dashed outline is a 448 pixel image at the same scale.`}
        >
          <rect className="mm-scale-ghost" x="4" y="4" width={full} height={full} />
          <g transform={`translate(4 ${4 + full - Math.min(size, full)})`}>
            <rect className="mm-scale-image" width={size} height={size} />
            {lines.map((at) => (
              <path key={at} className="mm-scale-line" d={`M ${at} 0 V ${size} M 0 ${at} H ${size}`} />
            ))}
            <rect className="mm-scale-one" width={cell} height={cell} />
          </g>
        </svg>
        <figcaption>
          <b>
            {scale.perSide} × {scale.perSide} tiles = {fmt(scale.patchTokens)} patch tokens
          </b>
          <span>
            drawn to scale; the dashed square is a 448px image, and the filled tile is one {vitPatch}×{vitPatch} patch
          </span>
        </figcaption>
      </figure>
      <div className="mm-scale-side">
        <div
          className="mm-side-bars"
          role="img"
          aria-label={`Patch tokens by image side at ${vitPatch} pixel patches: ${bySide
            .map((entry) => `${entry.option} pixels ${fmt(entry.tokens)}`)
            .join(", ")}.`}
        >
          <span className="mm-mini-title">
            Patch tokens by image side, at {vitPatch}×{vitPatch}
          </span>
          {bySide.map((entry) => (
            <div key={entry.option} className={entry.option === side ? "is-current" : ""} aria-hidden="true">
              <span>{entry.option}px</span>
              <i>
                <b style={{ width: `${(entry.tokens / most) * 100}%` }} />
              </i>
              <code>{fmt(entry.tokens)}</code>
            </div>
          ))}
        </div>
        <div className="mm-context">
          <span className="mm-mini-title">Room taken in a {fmt(CONTEXT)}-token context</span>
          <div
            className="mm-context-bar"
            role="img"
            aria-label={`${fmt(scale.llmImageTokens)} of ${fmt(CONTEXT)} context tokens are image tokens, ${(share * 100).toFixed(1)} percent.`}
          >
            <i style={{ width: `${Math.min(1, share) * 100}%` }} />
          </div>
          <div className="mm-context-labels" aria-hidden="true">
            <b>
              {fmt(scale.llmImageTokens)} image tokens · {(share * 100).toFixed(1)}%
            </b>
            <span>{fmt(CONTEXT - scale.llmImageTokens)} left for text</span>
          </div>
          {connector !== "llava" && (
            <small>
              {connector === "clip"
                ? "CLIP ends in one vector per image, scored against captions. Nothing enters a language model."
                : "The 64 resampled vectors are read through cross-attention, so they take no places in the sequence."}
            </small>
          )}
        </div>
      </div>
    </div>
  );
}

/** Where the contrastive loss sits between 0 and 3, against the loss an even guess would score. */
function LossMeter({ loss, chance }: { loss: number; chance: number }) {
  const top = 3;
  const at = (value: number) => Math.min(1, Math.max(0, value / top)) * 100;
  const side = (value: number) => (at(value) < 18 ? " is-start" : at(value) > 82 ? " is-end" : "");
  return (
    <div
      className="mm-loss-meter"
      role="img"
      aria-label={`Contrastive loss ${loss.toFixed(3)}, ${loss > chance ? "above" : "below"} the chance loss of ${chance.toFixed(3)}. Lower is better.`}
    >
      <div className="mm-loss-track" aria-hidden="true">
        <i className="mm-loss-good" style={{ width: `${at(chance)}%` }} />
        <b className={`mm-loss-now${loss > chance ? " is-bad" : ""}${side(loss)}`} style={{ left: `${at(loss)}%` }}>
          <span>
            loss {loss.toFixed(3)}
            {loss > top ? " →" : ""}
          </span>
        </b>
        <b className="mm-loss-chance" style={{ left: `${at(chance)}%` }}>
          <span>an even guess · ln 3</span>
        </b>
      </div>
      <div className="mm-loss-ends" aria-hidden="true">
        <span>0 · labels and scores agree</span>
        <span>3</span>
      </div>
    </div>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const pixels = parsePixels(asString(state, "pixels", DEFAULT_PIXELS));
  const rawPatch = asNumber(state, "patchSize", 4);
  const p = (PATCH_SIZES as readonly number[]).includes(rawPatch) ? rawPatch : 4;
  const perSide = SIZE / p;
  const patchCount = perSide * perSide;
  const selected = clamp(Math.round(asNumber(state, "selected", 1)), 0, patchCount - 1);
  const pixel = clamp(Math.round(asNumber(state, "pixel", 0)), 0, SIZE * SIZE - 1);
  const side = (SIDES as readonly number[]).includes(asNumber(state, "side", 336)) ? asNumber(state, "side", 336) : 336;
  const vitPatch = (VIT_PATCHES as readonly number[]).includes(asNumber(state, "vitPatch", 14))
    ? asNumber(state, "vitPatch", 14)
    : 14;
  const cls = asString(state, "cls", "on") !== "off";
  const connectorRaw = asString(state, "connector", "llava");
  const connector: Connector = connectorRaw === "clip" || connectorRaw === "flamingo" ? connectorRaw : "llava";

  const temperature = boundedTemperature(state.temperature);
  const pairing = pairingOf(state.pairing);
  const pairs = useMemo(() => contrast(temperature, pairing), [pairing, temperature]);

  const patches = useMemo(() => patchify(pixels, p), [p, pixels]);
  const bank = useMemo(() => filters(p), [p]);
  const current = patches[selected] ?? patches[0];
  const scale = useMemo(() => realScale(side, vitPatch, cls, connector), [cls, connector, side, vitPatch]);

  const flip = (index: number) => {
    const next = pixels.slice();
    next[index] = next[index] ? 0 : 1;
    setState({ pixels: next.join("") });
    narrate(`Pixel row ${Math.floor(index / SIZE) + 1}, column ${(index % SIZE) + 1} ${next[index] ? "on" : "off"}.`);
  };

  const patchOf = (index: number) => Math.floor(Math.floor(index / SIZE) / p) * perSide + Math.floor((index % SIZE) / p);
  const textTokens = TEXT_BEFORE.length + TEXT_AFTER.length;

  return (
    <div className="tg-lab tg-lab--hero mm-lab">
      <LabSurface label="Image to patches" className="mm-image-card">
        <SurfaceHeading
          kicker={`8×8 teaching image · ${patchCount} patch${patchCount === 1 ? "" : "es"} of ${p}×${p}`}
          title="Cut the image into tiles; each tile becomes one token"
        />
        <div className="mm-image-grid">
          <div>
            <div
              className="pixel-pad mm-pad"
              role="grid"
              aria-label={`Eight by eight teaching image, cut into ${patchCount} patches`}
              style={{ ["--mm-p" as string]: p }}
            >
              {pixels.map((bit, index) => {
                const row = Math.floor(index / SIZE);
                const column = index % SIZE;
                const inSelected = patchOf(index) === selected;
                const edges = [
                  row % p === 0 ? "is-top" : "",
                  column % p === 0 ? "is-left" : "",
                  inSelected ? "is-in-patch" : "",
                  index === pixel ? "is-cursor" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <button
                    key={index}
                    type="button"
                    role="gridcell"
                    aria-pressed={bit === 1}
                    aria-label={`Row ${row + 1}, column ${column + 1}, ${bit ? "ink" : "blank"}, patch ${patchOf(index) + 1}`}
                    className={`${bit ? "is-on" : ""} ${edges}`.trim()}
                    onClick={() => {
                      flip(index);
                      setState({ pixel: index, selected: patchOf(index) });
                    }}
                  />
                );
              })}
            </div>
            <RangeControl
              label="Selected pixel"
              min={0}
              max={SIZE * SIZE - 1}
              step={1}
              value={pixel}
              format={(value) => `row ${Math.floor(value / SIZE) + 1}, col ${(value % SIZE) + 1}`}
              onChange={(value) => setState({ pixel: value })}
            />
            <button type="button" className="quiet-action mm-flip" onClick={() => flip(pixel)}>
              Flip selected pixel
            </button>
          </div>
          <div className="mm-image-side">
            <div className="mm-patch-control">
              <SegmentedControl
                label="Patch size"
                value={String(p)}
                options={PATCH_SIZES.map((size) => ({ value: String(size), label: `${size}×${size}` }))}
                onChange={(value) => setState({ patchSize: Number(value), selected: 0 })}
              />
            </div>
            {patchCount > 1 ? (
              <RangeControl
                label="Selected patch"
                min={0}
                max={patchCount - 1}
                step={1}
                value={selected}
                format={(value) => `#${value + 1} of ${patchCount}`}
                onChange={(value) => setState({ selected: value })}
              />
            ) : (
              <p className="mm-one-patch">Selected patch: #1 of 1 — the whole image is one token.</p>
            )}
            <div className="metric-row">
              <Metric label="Patch tokens" value={`(8 / ${p})² = ${patchCount}`} tone="forward" />
              <Metric label="Pixels per patch" value={`${p}×${p} = ${p * p}`} />
            </div>
            <div className="mm-flat">
              <span>patch #{selected + 1} flattened, row by row: x</span>
              <div className="mm-unroll" role="img" aria-label={`Patch ${selected + 1} pixels: ${current.x.join(" ")}`}>
                <span className="mm-unroll-grid" aria-hidden="true">
                  {Array.from({ length: p }, (_, row) => (
                    <span key={row}>
                      {current.x.slice(row * p, row * p + p).map((bit, index) => (
                        <i key={index} className={bit ? "is-on" : ""} />
                      ))}
                    </span>
                  ))}
                </span>
                <span className="mm-unroll-arrow" aria-hidden="true">
                  →
                </span>
                <span className={`mm-unroll-strip is-p${p}`} aria-hidden="true">
                  {Array.from({ length: p }, (_, row) => (
                    <span key={row}>
                      <small>row {row + 1}</small>
                      <span>
                        {current.x.slice(row * p, row * p + p).map((bit, index) => (
                          <i key={index} className={bit ? "is-on" : ""} />
                        ))}
                      </span>
                    </span>
                  ))}
                </span>
              </div>
              <code>
                {p} rows of {p} = {current.x.length} numbers
              </code>
            </div>
          </div>
        </div>
        <p className="lab-note">
          Heavy lines are patch borders; the outlined tiles are the selected patch. A stroke that crosses a border is
          split between two tokens. Nothing is lost — attention can combine the pieces — but no single token holds the
          whole stroke.
        </p>
      </LabSurface>

      <LabSurface label="Patch embedding" className="mm-embed-card">
        <SurfaceHeading kicker={`z = W·x + position · W is 4 × ${p * p}`} title="A linear map from pixels to a token vector" />
        <div className="mm-embed-row">
          <figure className="mm-filter is-input">
            <MiniPatch patch={current} p={p} className="mm-filter-grid" />
            <figcaption>
              <span>x · patch #{selected + 1}</span>
              <code>{current.x.filter(Boolean).length} of {p * p} inked</code>
            </figcaption>
          </figure>
          <span className="mm-times" aria-hidden="true">
            →
          </span>
          {bank.map((weights, index) => (
            <FilterTile key={FILTER_NAMES[index]} weights={weights} p={p} name={FILTER_NAMES[index]} value={current.wx[index]} />
          ))}
        </div>
        <div className="source-table mm-embed-table">
          <table className="tg-board">
            <thead>
              <tr>
                <th scope="col">row of W</th>
                <th scope="col">W·x</th>
                <th scope="col">+ position</th>
                <th scope="col">= token</th>
                <th scope="col" className="mm-token-bar-head">
                  −1 · token · +1
                </th>
              </tr>
            </thead>
            <tbody>
              {FILTER_NAMES.map((name, index) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{f2(current.wx[index])}</td>
                  <td>{f2(current.position[index])}</td>
                  <td>{f2(current.token[index])}</td>
                  <td className="mm-token-bar" aria-hidden="true">
                    <span>
                      <i
                        style={{
                          left: `${current.token[index] >= 0 ? 50 : 50 - Math.min(50, Math.abs(current.token[index]) * 45)}%`,
                          width: `${Math.min(50, Math.abs(current.token[index]) * 45)}%`,
                        }}
                      />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Each row of W is a {p}×{p} filter (+ and − cells); its output is the filter dotted with the patch. The
          position code is a fixed sine pattern here; ViT learns one. A real ViT-B/16 uses 768 learned filters of
          16×16×3.
        </p>
      </LabSurface>

      <LabSurface label="Token sequence" className="mm-sequence-card">
        <SurfaceHeading
          kicker={`${patchCount} image + ${textTokens} text tokens`}
          title="Image tokens sit in the same sequence as words"
        />
        <ol className="mm-sequence" aria-label="Model input sequence">
          {TEXT_BEFORE.map((token) => (
            <li key={token} className="is-text">
              {token}
            </li>
          ))}
          {patches.map((patch) => (
            <li key={`img-${patch.id}`} className={`is-image${patch.id === selected ? " is-selected" : ""}`}>
              <button
                type="button"
                aria-label={`Image token ${patch.id + 1}`}
                aria-pressed={patch.id === selected}
                onClick={() => setState({ selected: patch.id })}
              >
                <MiniPatch patch={patch} p={p} />
                <small>{patch.id + 1}</small>
              </button>
            </li>
          ))}
          {TEXT_AFTER.map((token, index) => (
            <li key={`${token}-${index}`} className="is-text">
              {token}
            </li>
          ))}
        </ol>
        <p className="lab-note">
          In a LLaVA-style model a projector maps each patch vector to the language model's embedding width, and the
          vectors replace an image placeholder in the prompt. From there, self-attention treats them like any other
          positions. Here the token is still 4 numbers; LLaVA-1.5 projects 1,024 to {LLM_WIDTH.toLocaleString("en-US")}.
        </p>
      </LabSurface>

      <LabSurface label="Real-scale arithmetic" className="mm-scale-card">
        <SurfaceHeading
          kicker={`${scale.encoder.name} at ${side}px · ${scale.patchTokens.toLocaleString("en-US")} patch tokens`}
          title="The same cut at the sizes real models use"
        />
        <div className="mm-scale-controls">
          <SegmentedControl
            label="Image side"
            value={String(side)}
            options={SIDES.map((value) => ({ value: String(value), label: `${value}px` }))}
            onChange={(value) => setState({ side: Number(value) })}
          />
          <SegmentedControl
            label="Encoder patch"
            value={String(vitPatch)}
            options={VIT_PATCHES.map((value) => ({ value: String(value), label: `${value}×${value}` }))}
            onChange={(value) => setState({ vitPatch: Number(value) })}
          />
          <SegmentedControl
            label="CLS token"
            value={cls ? "on" : "off"}
            options={[
              { value: "on", label: "+1 CLS" },
              { value: "off", label: "none" },
            ]}
            onChange={(value) => setState({ cls: value })}
          />
          <div className="mm-connector-control">
            <SegmentedControl
              label="Connector"
              value={connector}
              options={[
                { value: "clip", label: "CLIP (contrastive)" },
                { value: "llava", label: "LLaVA (projector)" },
                { value: "flamingo", label: "Flamingo-style (cross-attn)" },
              ]}
              onChange={(value) => setState({ connector: value })}
            />
          </div>
        </div>
        <ScalePicture side={side} vitPatch={vitPatch} cls={cls} connector={connector} scale={scale} />
        <ol className="mm-stages">
          {scale.stages.map((stage, index) => (
            <li key={stage.name}>
              <span className="mm-stage-no">{index + 1}</span>
              <strong>{stage.name}</strong>
              <code>{stage.shape}</code>
              <small>{stage.note}</small>
            </li>
          ))}
        </ol>
        <div className="metric-row">
          <Metric
            label="Patch tokens"
            value={`(${scale.padded} / ${vitPatch})² = ${scale.patchTokens.toLocaleString("en-US")}`}
            tone="forward"
          />
          <Metric label="Encoder sequence" value={scale.encoderSequence.toLocaleString("en-US")} />
          <Metric label="Values per patch" value={`${vitPatch}·${vitPatch}·3 = ${scale.rawPerPatch.toLocaleString("en-US")}`} />
          <Metric
            label={`Context share of ${CONTEXT.toLocaleString("en-US")}`}
            value={connector === "llava" ? `${(scale.contextShare * 100).toFixed(1)}%` : "0% (not in the sequence)"}
            tone={scale.contextShare > 0.25 ? "loss" : undefined}
          />
        </div>
        <p className="lab-note">
          Doubling the image side quadruples the patch tokens, and self-attention cost grows with the square of
          sequence length. The widths are the common CLIP ViT sizes and a 7B language model; other models differ.
        </p>
      </LabSurface>

      <LabSurface label="Contrastive pairs" className="mm-pairs-card">
        <SurfaceHeading
          kicker={`3 images · 3 captions · τ = ${temperature.toFixed(2)} · ${
            pairing === "true" ? "true pairs" : "shuffled captions"
          }`}
          title="Matched pairs win because the loss pulls them together"
          aside={<span className="tg-badge">hand-authored vectors</span>}
        />
        <div className="mm-pair-controls">
          <RangeControl
            label="Temperature τ"
            min={TEMPERATURE_RANGE.min}
            max={TEMPERATURE_RANGE.max}
            step={0.01}
            value={temperature}
            format={(value) => `τ = ${value.toFixed(2)}`}
            onChange={(value) => setState({ temperature: value })}
          />
          <SegmentedControl
            label="Which pairs count as matches"
            value={pairing}
            options={[
              { value: "true", label: "true pairs" },
              { value: "shuffled", label: "shuffled captions" },
            ]}
            onChange={(value) => setState({ pairing: value })}
          />
        </div>
        <div className="source-table mm-pair-table">
          <table
            className="tg-board"
            aria-label="Cosine similarity of each image with each caption, with the share each caption gets in its row and column"
          >
            <thead>
              <tr>
                <th scope="col">image ↓ · caption →</th>
                {CAPTIONS.map((caption) => (
                  <th scope="col" key={caption.id}>
                    {caption.id} · “{caption.label}”
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {IMAGES.map((image, row) => (
                <tr key={image.id}>
                  <th scope="row">
                    {image.id} · {image.label}
                  </th>
                  {CAPTIONS.map((caption, column) => {
                    const isMatch = pairs.positives[row] === column;
                    return (
                      <td
                        key={caption.id}
                        className={isMatch ? "is-match" : ""}
                        style={{ "--pair-share": pairs.rowShares[row][column] } as CSSProperties}
                      >
                        <b>{f2(SIMILARITY[row][column])}</b>
                        <small>
                          row {Math.round(pairs.rowShares[row][column] * 100)}% · column{" "}
                          {Math.round(pairs.columnShares[row][column] * 100)}%
                        </small>
                        <span className="mm-pair-share" aria-hidden="true">
                          <i style={{ width: `${pairs.rowShares[row][column] * 100}%` }} />
                        </span>
                        {isMatch && <em>labelled match</em>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <LossMeter loss={pairs.loss} chance={CHANCE_LOSS} />
        <div className="metric-row">
          <Metric
            label="Contrastive loss"
            value={pairs.loss.toFixed(3)}
            tone={pairs.loss > CHANCE_LOSS ? "loss" : "forward"}
          />
          <Metric label="Image → caption" value={pairs.imageToCaption.toFixed(3)} />
          <Metric label="Caption → image" value={pairs.captionToImage.toFixed(3)} />
          <Metric label="Chance loss, ln 3" value={CHANCE_LOSS.toFixed(3)} />
        </div>
        <div className="metric-row">
          <Metric label="Images that rank their match first" value={`${pairs.imagesRight} of ${IMAGES.length}`} />
          <Metric label="Captions that rank their match first" value={`${pairs.captionsRight} of ${CAPTIONS.length}`} />
        </div>
        <div className="source-table mm-vector-table">
          <table className="tg-board" aria-label="The hand-authored vectors behind the similarity matrix">
            <thead>
              <tr>
                <th scope="col">vector</th>
                {TOPICS.map((topic) => (
                  <th scope="col" key={topic}>
                    {topic}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...IMAGES.map((item) => ({ key: `image-${item.id}`, name: `image ${item.id}`, item })),
                ...CAPTIONS.map((item) => ({ key: `caption-${item.id}`, name: `caption ${item.id}`, item }))].map(
                ({ key, name, item }) => (
                  <tr key={key}>
                    <th scope="row">{name}</th>
                    {item.vector.map((value, index) => (
                      <td key={TOPICS[index]} style={{ "--mm-v": Math.min(1, Math.max(0, value)) } as CSSProperties}>
                        {f2(value)}
                      </td>
                    ))}
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Each cell is the cosine similarity of an image vector with a caption vector. “Row” is that caption's share
          among the three captions for that image after dividing by τ and applying a softmax; “column” is the image's
          share among the three images for that caption. The loss averages −log of the labelled match's share over both
          directions. The vectors are written by hand, with named topics for readability; no encoder ran and nothing
          was trained.
        </p>
      </LabSurface>
    </div>
  );
}
