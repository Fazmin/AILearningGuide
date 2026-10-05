import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import metadata from "./assets/language-models.metadata.json";
import vocabulary from "./assets/transformer-vocab.json";
import { HELD_OUT_PERCENT, LOSS_BITS, TRAINING_STEPS } from "./evaluation";
import { LENS_STAGES, type LensStage } from "./lens";
import { DEFAULT_PROMPT, provenanceOf, TRAINING_LINE, TRAINING_LINE_OFFSET, type Provenance } from "./prompts";
import {
  describeToken,
  displayToken,
  drawCandidate,
  encode,
  entropyBits,
  formatPercent,
  decodeDraw,
  encodeDraw,
  isModelId,
  logSoftmax,
  MAX_GENERATED,
  MAX_PROMPT,
  MIN_P_MAX,
  MODELS,
  seededUniform,
  shapeDistribution,
  softmax,
  surprisalBits,
  tokenText,
  TRANSFORMER_CONTEXT,
  VOCAB_SIZE,
  type DrawRecord,
  type ModelId,
} from "./sampling";
import { useCharModels, type ModelView } from "./useCharModels";
import { useLogitLens } from "./useLogitLens";

export { DEFAULT_PROMPT };
const TOP_ROWS = 10;
const SEED_MAX = 64;
const BURST = 10;

const MODEL_INFO: Record<
  ModelId,
  { name: string; params: string; heldOut: number; heldOutNote: string; wholeCorpus: number; wholeNote: string; budget: string }
> = {
  bigram: {
    name: "Bigram table",
    params: `${(VOCAB_SIZE * VOCAB_SIZE).toLocaleString("en-US")} probabilities`,
    heldOut: LOSS_BITS.bigram.heldOut,
    heldOutNote: `table fitted on the first ${100 - HELD_OUT_PERCENT}% only`,
    wholeCorpus: LOSS_BITS.bigram.wholeCorpus,
    wholeNote: "the shipped table, fitted on all of it",
    budget: "counted once, no training steps",
  },
  rnn: {
    name: "Character RNN (GRU)",
    params: `${metadata.rnn.parameters.toLocaleString("en-US")} weights`,
    heldOut: LOSS_BITS.rnn.heldOut,
    heldOutNote: "nearly held out: it trained on random windows of the whole corpus",
    wholeCorpus: LOSS_BITS.rnn.wholeCorpus,
    wholeNote: "about a third of an epoch of training",
    budget: `${TRAINING_STEPS.rnn.toLocaleString("en-US")} training steps`,
  },
  transformer: {
    name: "Tiny transformer",
    params: `${metadata.transformer.parameters.toLocaleString("en-US")} weights`,
    heldOut: LOSS_BITS.transformer.heldOut,
    heldOutNote: `held out: it never trained on the last ${HELD_OUT_PERCENT}%`,
    wholeCorpus: LOSS_BITS.transformer.wholeCorpus,
    wholeNote: `${100 - HELD_OUT_PERCENT}% of this is its own training text`,
    budget: `${TRAINING_STEPS.transformer.toLocaleString("en-US")} training steps`,
  },
};

const ordinal = (value: number) => {
  const tens = value % 100;
  if (tens >= 11 && tens <= 13) return `${value}th`;
  return `${value}${["th", "st", "nd", "rd"][value % 10 < 4 ? value % 10 : 0]}`;
};

const { model_width: WIDTH, heads: HEADS, layers: LAYERS, feed_forward_width: MLP_WIDTH } = metadata.transformer.config;

/** The path from characters to one sampled character, with the three places the logit lens reads. */
const PIPELINE: ReadonlyArray<{ name: string; detail: string; tap?: number }> = [
  { name: "Characters", detail: `${VOCAB_SIZE} symbols, one id each` },
  { name: "Embeddings", detail: `token + position: ${WIDTH} numbers per character`, tap: 0 },
  { name: "Layer 1", detail: `${HEADS} attention heads, then an MLP ${MLP_WIDTH} wide`, tap: 1 },
  { name: "Layer 2", detail: "same shape, its own weights", tap: 2 },
  { name: "Final norm + unembedding", detail: `${WIDTH} numbers → ${VOCAB_SIZE} logits` },
  { name: "Softmax", detail: "logits → probabilities" },
  { name: "Sample", detail: "draw one, append, run again" },
];

const trainingLinePercent = Math.round((TRAINING_LINE_OFFSET / metadata.corpus_characters) * 100);

function provenanceText(provenance: Provenance) {
  if (provenance === "unseen") {
    return "No. This exact text is not in Tiny Shakespeare (checked against the corpus offline), so no model trained on it.";
  }
  if (provenance === "training") {
    return `Yes. This line is in Tiny Shakespeare, ${trainingLinePercent}% of the way through, inside the first ${100 - HELD_OUT_PERCENT}% the transformer trained on. The GRU and the bigram table saw the whole corpus.`;
  }
  return "Unknown. The lab does not ship the corpus, so it cannot check your text; only a line copied from Tiny Shakespeare can be in the models' training data.";
}

const asNumber = (value: unknown, fallback: number) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

const contextStart = (model: ModelId, length: number) =>
  model === "bigram" ? Math.max(0, length - 1) : model === "transformer" ? Math.max(0, length - TRANSFORMER_CONTEXT) : 0;

const seesText = (model: ModelId, length: number) =>
  model === "bigram"
    ? "only the last character"
    : model === "transformer"
      ? `the last ${Math.min(length, TRANSFORMER_CONTEXT)} characters, each directly`
      : `all ${length} characters, through a 2-layer, 192-wide state`;

function statusText(model: ModelId, view: ModelView, empty: boolean) {
  const label = MODELS.find((item) => item.id === model)?.label ?? model;
  if (empty) return "Type at least one character: a character model needs something to condition on.";
  if (view.status === "error") {
    return `The ${label} could not run (${view.error ?? "unknown error"}). ${model === "bigram" ? "Try reloading the lab." : "The bigram table still works."}`;
  }
  if (view.status === "loading") {
    if (model === "bigram") return "Loading the bigram table (95 kB)…";
    if (!view.rows) return `Loading the ${label} (${model === "rnn" ? "1.9 MB" : "5.5 MB"} ONNX) in a background worker…`;
    return `Running the ${label} on the new text…`;
  }
  if (model === "bigram") return "Table lookup on the last character.";
  return `${label} forward pass ran with ${view.provider === "webgpu" ? "WebGPU" : "WebAssembly"}.`;
}

function meanSurprisal(view: ModelView, ids: ReadonlyArray<number>) {
  if (view.status !== "ready" || view.stale || !view.rows) return undefined;
  let total = 0;
  let count = 0;
  for (let position = Math.max(1, view.offset + 1); position < ids.length; position += 1) {
    const row = view.rows[position - 1 - view.offset];
    if (!row) continue;
    total += -logSoftmax(row)[ids[position]] * Math.LOG2E;
    count += 1;
  }
  return count ? { bits: total / count, count } : undefined;
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const prompt = (typeof state.prompt === "string" ? state.prompt : DEFAULT_PROMPT).slice(0, MAX_PROMPT);
  const generatedChars = typeof state.generated === "string" ? Array.from(state.generated).slice(0, MAX_GENERATED) : [];
  const generated = generatedChars.join("");
  // One stored draw per generated character, aligned from the end.
  const storedDraws = generatedChars.length === 0 || !Array.isArray(state.draws) ? [] : state.draws.slice(-generatedChars.length);
  const drawOffset = generatedChars.length - storedDraws.length;
  const draws = storedDraws
    .map((value, index) => decodeDraw(value, generatedChars[drawOffset + index]))
    .filter((record): record is DrawRecord => record !== undefined);
  const model: ModelId = isModelId(state.model) ? state.model : "rnn";
  const temperature = clamp(asNumber(state.temperature, 0.8), 0.1, 2);
  const topK = Math.round(clamp(asNumber(state.topK, VOCAB_SIZE), 1, VOCAB_SIZE));
  const topP = clamp(asNumber(state.topP, 1), 0.05, 1);
  const minP = clamp(asNumber(state.minP, 0), 0, MIN_P_MAX);
  const seed = Math.round(clamp(asNumber(state.sample, 2), 1, SEED_MAX));

  const provenance = provenanceOf(prompt);
  const text = prompt + generated;
  const chars = Array.from(text);
  const promptLength = chars.length - generatedChars.length;
  const ids = useMemo(() => encode(text, vocabulary), [text]);
  const models = useCharModels(ids);
  const lens = useLogitLens(ids);
  const view = models[model];
  const empty = ids.length === 0;
  const modelLabel = MODELS.find((item) => item.id === model)?.label ?? model;

  const nextLogits = !empty && view.rows?.length ? view.rows[view.rows.length - 1] : undefined;
  const shaped = useMemo(
    () => (nextLogits ? shapeDistribution(nextLogits, { temperature, topK, topP, minP }) : undefined),
    [nextLogits, temperature, topK, topP, minP],
  );
  const fresh = !empty && view.status === "ready" && !view.stale;
  const canSample = fresh && Boolean(shaped) && generatedChars.length < MAX_GENERATED;
  const u = seededUniform(seed, generatedChars.length);
  const preview = shaped ? drawCandidate(shaped, u) : undefined;

  const freshSurprisal = useMemo(() => {
    if (!fresh || !view.rows) return undefined;
    const out: Array<number | null> = ids.map(() => null);
    for (let position = 1; position < ids.length; position += 1) {
      const row = view.rows[position - 1 - view.offset];
      if (row) out[position] = -logSoftmax(row)[ids[position]] * Math.LOG2E;
    }
    return out;
  }, [fresh, ids, view.offset, view.rows]);

  // While a new forward pass is pending, keep showing the bars already measured for
  // the unchanged prefix instead of blanking the tape between draws.
  const lastSurprisal = useRef<{ text: string; model: ModelId; values: Array<number | null> } | undefined>(undefined);
  if (freshSurprisal) lastSurprisal.current = { text, model, values: freshSurprisal };
  const previous = lastSurprisal.current;
  const surprisal: Array<number | null> =
    freshSurprisal ??
    (previous && previous.model === model && text.startsWith(previous.text)
      ? ids.map((_, position) => previous.values[position] ?? null)
      : ids.map(() => null));

  const visibleFrom = contextStart(model, chars.length);
  const scored = surprisal.filter((value): value is number => value !== null);
  const drawBits = draws.map((record) => surprisalBits(record.p));
  const meanDrawBits = drawBits.length ? drawBits.reduce((a, b) => a + b, 0) / drawBits.length : undefined;
  const lastDraw: DrawRecord | undefined = draws[draws.length - 1];

  const [pending, setPending] = useState(0);

  const sampleOnce = () => {
    if (!canSample || !shaped) return;
    const pick = drawCandidate(shaped, u);
    const character = tokenText(pick.index, vocabulary);
    setState({
      generated: generated + character,
      draws: [...storedDraws, encodeDraw({ p: pick.p, q: pick.q, u, m: model })],
    });
    narrate(
      `Sampled ${describeToken(character)}: sampling probability ${formatPercent(pick.q)}, model probability ${formatPercent(pick.p)}, surprisal ${surprisalBits(pick.p).toFixed(2)} bits.`,
    );
  };

  useEffect(() => {
    if (pending <= 0) return;
    if (view.status === "error" || empty || generatedChars.length >= MAX_GENERATED) {
      setPending(0);
      return;
    }
    if (!canSample) return;
    sampleOnce();
    setPending((count) => count - 1);
    // sampleOnce closes over this render's state; text and canSample gate re-runs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending, canSample, text]);

  const shownRows = shaped ? shaped.candidates.slice(0, TOP_ROWS) : [];
  const tail = shaped ? shaped.candidates.slice(TOP_ROWS) : [];
  const tailP = tail.reduce((sum, item) => sum + item.p, 0);
  const tailQ = tail.reduce((sum, item) => sum + item.q, 0);
  const scale = Math.max(1e-9, ...shownRows.flatMap((item) => [item.p, item.q]));
  const kept = shaped ? shaped.candidates.filter((item) => item.kept) : [];
  const top = shaped?.candidates[0];

  const comparison = MODELS.map((item) => {
    const itemView = models[item.id];
    const logits = !empty && itemView.rows?.length ? itemView.rows[itemView.rows.length - 1] : undefined;
    const probabilities = logits ? softmax(logits) : undefined;
    const ranked = probabilities
      ? probabilities
          .map((value, index) => ({ value, index }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 3)
      : [];
    return {
      ...item,
      view: itemView,
      ranked,
      entropy: probabilities ? entropyBits(probabilities) : undefined,
      text: meanSurprisal(itemView, ids),
    };
  });

  const tapeSummary = `${promptLength} prompt characters and ${generatedChars.length} generated. ${
    scored.length
      ? `Mean surprisal under the ${modelLabel}: ${(scored.reduce((a, b) => a + b, 0) / scored.length).toFixed(2)} bits per character.`
      : ""
  } The ${modelLabel} sees ${seesText(model, chars.length)}.`;

  return (
    <div className="tg-lab ntp-lab">
      <LabSurface label="Generation loop" className="ntp-loop-card">
        <SurfaceHeading
          kicker={`${chars.length} characters · ${generatedChars.length} generated · 1 token = 1 character`}
          title="Predict, draw one character, append it, repeat"
        />
        <label className="prompt-input ntp-prompt">
          <span>Prompt</span>
          <textarea
            value={prompt}
            rows={2}
            maxLength={MAX_PROMPT}
            spellCheck={false}
            onChange={(event) => {
              setPending(0);
              setState({ prompt: event.target.value.slice(0, MAX_PROMPT), generated: "", draws: [] });
            }}
          />
          <small>Editing the prompt clears the continuation. Characters outside the 65-symbol alphabet become &lt;unk&gt;.</small>
        </label>
        <div className="ntp-presets">
          <button
            type="button"
            className="quiet-action"
            aria-pressed={provenance === "unseen"}
            onClick={() => {
              setPending(0);
              setState({ prompt: DEFAULT_PROMPT, generated: "", draws: [] });
            }}
          >
            Unseen line (default)
          </button>
          <button
            type="button"
            className="quiet-action"
            aria-pressed={provenance === "training"}
            onClick={() => {
              setPending(0);
              setState({ prompt: TRAINING_LINE, generated: "", draws: [] });
            }}
          >
            Line from the training text
          </button>
          <p className={`ntp-seen ntp-seen--${provenance}`} role="status" aria-live="polite">
            <strong>Seen in training?</strong> {provenanceText(provenance)}
          </p>
        </div>
        <div className="ntp-tape" role="img" aria-label={tapeSummary}>
          {chars.map((character, position) => {
            const bits = surprisal[position];
            const classes = [
              "ntp-tape__char",
              position >= promptLength ? "is-generated" : "",
              position < visibleFrom ? "is-outside" : "",
              ids[position] === vocabulary.unk ? "is-unknown" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <Fragment key={position}>
                <span
                  className={classes}
                  title={`${describeToken(character)}${bits === null ? "" : ` · ${bits.toFixed(2)} bits`}`}
                  style={{ "--ntp-bits": bits === null ? 0 : Math.min(1, bits / 10) } as CSSProperties}
                  aria-hidden="true"
                >
                  <b>{displayToken(character)}</b>
                  <i />
                </span>
                {character === "\n" && <span className="ntp-tape__break" aria-hidden="true" />}
              </Fragment>
            );
          })}
          <span className="ntp-tape__next" aria-hidden="true">
            {preview && fresh ? displayToken(tokenText(preview.index, vocabulary)) : "?"}
          </span>
        </div>
        <ul className="ntp-tape-legend">
          <li><i className="ntp-key ntp-key--prompt" /> prompt</li>
          <li><i className="ntp-key ntp-key--generated" /> sampled and appended</li>
          <li><i className="ntp-key ntp-key--bits" /> bar height = surprisal under the {modelLabel}, 0–10 bits</li>
          {visibleFrom > 0 && <li><i className="ntp-key ntp-key--outside" /> faded = outside what the {modelLabel} sees</li>}
          <li><i className="ntp-key ntp-key--next" /> dashed = where the next draw (u = {u.toFixed(3)}) would land</li>
        </ul>
        <div className="ntp-actions">
          <button type="button" className="primary-action" disabled={!canSample} onClick={sampleOnce}>
            Sample next token
          </button>
          <button
            type="button"
            className="quiet-action"
            disabled={!canSample || pending > 0}
            onClick={() => setPending(Math.min(BURST, MAX_GENERATED - generatedChars.length))}
          >
            {pending > 0 ? `Sampling… ${pending} left` : `Sample ${BURST} more`}
          </button>
          <button
            type="button"
            className="quiet-action"
            disabled={generatedChars.length === 0}
            onClick={() => {
              setPending(0);
              setState({ generated: generatedChars.slice(0, -1).join(""), draws: storedDraws.slice(0, -1) });
            }}
          >
            Undo
          </button>
          <button
            type="button"
            className="quiet-action"
            disabled={generatedChars.length === 0}
            onClick={() => {
              setPending(0);
              setState({ generated: "", draws: [] });
            }}
          >
            Clear continuation
          </button>
        </div>
        <p className="lab-note">
          Nothing here ends the text: the 66-symbol vocabulary has no end-of-sequence token, so the model never signals
          that it is done, and the lab simply stops accepting characters at {MAX_GENERATED}.
        </p>
        <div className="metric-row">
          <Metric
            label="Last draw"
            value={lastDraw ? `“${displayToken(lastDraw.c)}” · u ${lastDraw.u.toFixed(3)}` : "none yet"}
          />
          <Metric
            label="p model → q sampler"
            value={lastDraw ? `${formatPercent(lastDraw.p)} → ${formatPercent(lastDraw.q)}` : "—"}
          />
          <Metric
            label="Surprisal −log2 p"
            value={lastDraw ? `${surprisalBits(lastDraw.p).toFixed(2)} bits` : "—"}
            tone="loss"
          />
          <Metric
            label="Mean over draws"
            value={meanDrawBits === undefined ? "—" : `${meanDrawBits.toFixed(2)} bits · ppl ${(2 ** meanDrawBits).toFixed(1)}`}
            tone="loss"
          />
        </div>
        {draws.length > 0 && (
          <div className="ntp-log" role="table" aria-label="Most recent draws">
            <div role="row" className="ntp-log__head">
              <span role="columnheader">#</span>
              <span role="columnheader">token</span>
              <span role="columnheader">u</span>
              <span role="columnheader">q sampler</span>
              <span role="columnheader">p model</span>
              <span role="columnheader">surprisal</span>
              <span role="columnheader">model</span>
            </div>
            {draws
              .map((record, index) => ({ record, index }))
              .slice(-5)
              .reverse()
              .map(({ record, index }) => (
                <div role="row" key={index}>
                  <span role="cell">{index + 1}</span>
                  <span role="cell"><code>{displayToken(record.c)}</code></span>
                  <span role="cell">{record.u.toFixed(3)}</span>
                  <span role="cell">{formatPercent(record.q)}</span>
                  <span role="cell">{formatPercent(record.p)}</span>
                  <span role="cell">{surprisalBits(record.p).toFixed(2)} bits</span>
                  <span role="cell">{MODELS.find((item) => item.id === record.m)?.label}</span>
                </div>
              ))}
          </div>
        )}
      </LabSurface>

      <LabSurface label="Next token distribution" className="ntp-dist-card">
        <SurfaceHeading
          kicker={`${modelLabel} · sees ${seesText(model, chars.length)}`}
          title="Odds for the next character"
        />
        <p className={`ntp-status ntp-status--${empty ? "error" : view.status}`} role="status" aria-live="polite">
          {statusText(model, view, empty)}
        </p>
        {shaped ? (
          <div className={`ntp-dist ${fresh ? "" : "is-stale"}`} aria-busy={!fresh}>
            <div className="ntp-dist__head" aria-hidden="true">
              <span>next</span>
              <span>
                <i className="ntp-key ntp-key--p" /> p model, T = 1 <i className="ntp-key ntp-key--q" /> q sampler
              </span>
              <span>p → q</span>
            </div>
            <ol className="ntp-dist__rows" aria-label={`Top ${TOP_ROWS} next characters under the ${modelLabel}`}>
              {shownRows.map((item) => {
                const character = tokenText(item.index, vocabulary);
                const isPreview = preview?.index === item.index;
                return (
                  <li
                    key={item.index}
                    className={[item.kept ? "" : "is-cut", isPreview ? "is-preview" : ""].filter(Boolean).join(" ")}
                  >
                    <span className="sr-only">
                      {`${describeToken(character)}: model ${formatPercent(item.p)}, sampler ${item.kept ? formatPercent(item.q) : `0%, cut by ${item.cutBy}`}${isPreview ? ", the next draw lands here" : ""}.`}
                    </span>
                    <code aria-hidden="true">{displayToken(character)}</code>
                    <span className="ntp-bar" aria-hidden="true">
                      <i className="ntp-bar__p" style={{ width: `${(item.p / scale) * 100}%` }} />
                      <b className="ntp-bar__q" style={{ width: `${(item.q / scale) * 100}%` }} />
                    </span>
                    <span className="ntp-dist__values" aria-hidden="true">
                      {formatPercent(item.p)} → {item.kept ? formatPercent(item.q) : <em>cut · {item.cutBy}</em>}
                    </span>
                  </li>
                );
              })}
              <li className="ntp-dist__tail">
                <span className="sr-only">{`${tail.length} other characters: model ${formatPercent(tailP)}, sampler ${formatPercent(tailQ)}.`}</span>
                <code aria-hidden="true">+{tail.length}</code>
                <span aria-hidden="true">other characters</span>
                <span className="ntp-dist__values" aria-hidden="true">
                  {formatPercent(tailP)} → {formatPercent(tailQ)}
                </span>
              </li>
            </ol>
            <p className="ntp-dist__scale">Bars are scaled so the longest shown value ({formatPercent(scale)}) fills the track.</p>
            <div className="ntp-cdf">
              <span className="ntp-cdf__label">Inverse-CDF draw over the {shaped.keptCount} kept characters</span>
              <div className="ntp-cdf__strip" role="img" aria-label={`Kept characters laid end to end by sampling probability. The next uniform number, ${u.toFixed(3)}, falls in ${preview ? describeToken(tokenText(preview.index, vocabulary)) : "none"}.`}>
                {kept.map((item) => (
                  <span
                    key={item.index}
                    className={preview?.index === item.index ? "is-preview" : ""}
                    style={{ width: `${item.q * 100}%` }}
                  >
                    {item.q >= 0.06 ? displayToken(tokenText(item.index, vocabulary)) : ""}
                  </span>
                ))}
                <i className="ntp-cdf__marker" style={{ left: `${u * 100}%` }} />
              </div>
              <span className="ntp-cdf__axis" aria-hidden="true">
                <span>0</span>
                <span>u = {u.toFixed(3)} (seed {seed}, draw {generatedChars.length + 1})</span>
                <span>1</span>
              </span>
            </div>
            <div className="metric-row">
              <Metric label="Model entropy" value={`${entropyBits(shaped.p).toFixed(2)} bits`} />
              <Metric label="Sampler entropy" value={`${entropyBits(shaped.q).toFixed(2)} bits`} tone="forward" />
              <Metric label="Kept" value={`${shaped.keptCount} of ${VOCAB_SIZE}`} />
              <Metric label="Kept mass" value={formatPercent(shaped.keptMass)} />
            </div>
          </div>
        ) : (
          <div className="ntp-dist ntp-dist--empty" aria-hidden="true">
            {Array.from({ length: 6 }, (_, index) => (
              <span key={index} />
            ))}
          </div>
        )}
      </LabSurface>

      <LabSurface label="Sampler controls" className="ntp-controls-card">
        <SurfaceHeading kicker="Same logits, reshaped before the draw" title="Model, temperature, top-k, top-p, min-p, seed" />
        <div className="ntp-model-control">
          <SegmentedControl
            label="Model"
            value={model}
            options={MODELS.map((item) => ({ value: item.id, label: item.label }))}
            onChange={(value) => setState({ model: value })}
          />
        </div>
        <RangeControl
          label="Temperature"
          min={0.1}
          max={2}
          step={0.05}
          value={temperature}
          format={(value) => value.toFixed(2)}
          onChange={(value) => setState({ temperature: Math.round(value * 100) / 100 })}
        />
        <RangeControl
          label="Top-k"
          min={1}
          max={VOCAB_SIZE}
          step={1}
          value={topK}
          format={(value) => (value >= VOCAB_SIZE ? `off · all ${VOCAB_SIZE}` : value === 1 ? "1 · greedy" : `${value}`)}
          onChange={(value) => setState({ topK: value })}
        />
        <RangeControl
          label="Top-p"
          min={0.05}
          max={1}
          step={0.05}
          value={topP}
          format={(value) => (value >= 1 ? "off · 1.00" : value.toFixed(2))}
          onChange={(value) => setState({ topP: Math.round(value * 100) / 100 })}
        />
        <RangeControl
          label="Min-p"
          min={0}
          max={MIN_P_MAX}
          step={0.01}
          value={minP}
          format={(value) => (value <= 0 ? "off · 0.00" : value.toFixed(2))}
          onChange={(value) => setState({ minP: Math.round(value * 100) / 100 })}
        />
        <RangeControl
          label="Seed"
          min={1}
          max={SEED_MAX}
          step={1}
          value={seed}
          format={(value) => `${value}`}
          onChange={(value) => setState({ sample: value })}
        />
        {top && shaped && (
          <FormulaWithValues
            label={`Top candidate ${displayToken(tokenText(top.index, vocabulary))}: q = exp(z/T) / Σ over kept of exp(z/T)`}
            expression={`z = ${top.logit.toFixed(2)} → z/T = ${(top.logit / temperature).toFixed(2)}`}
            result={`q ${formatPercent(top.q)}`}
            detail={
              model === "bigram"
                ? "For the bigram, z is the natural log of the table entry, so T = 1 with nothing cut gives back the table."
                : `z is the ${modelLabel}'s raw output logit for this character.`
            }
            tone="forward"
          />
        )}
        {shaped && minP > 0 && (
          <FormulaWithValues
            label="Min-p cut-off: keep a character if its probability ≥ p_min × p_max"
            expression={`${minP.toFixed(2)} × ${formatPercent(shaped.candidates[0].tempered)} = ${formatPercent(shaped.minPThreshold)}`}
            result={`${shaped.keptCount} kept`}
            detail="p_max is the top character's probability after temperature, so the bar rises when the model is confident and falls when it is not."
            tone="forward"
          />
        )}
        <p className="lab-note">
          Order of operations: divide logits by T, keep the top k, keep the smallest top-p nucleus of what is left, drop what falls below the min-p cut-off, renormalise, draw. None of these can promote a lower-ranked character above a higher one.
        </p>
      </LabSurface>

      <LabSurface label="Three models, one context" className="ntp-compare-card">
        <SurfaceHeading
          kicker="Same 66-symbol alphabet, same Tiny Shakespeare text"
          title="How much of the context each model can use"
        />
        <div className="ntp-compare">
          {comparison.map((item) => (
            <article key={item.id} className={item.id === model ? "is-current" : ""} aria-label={`${MODEL_INFO[item.id].name}`}>
              <header>
                <strong>{MODEL_INFO[item.id].name}</strong>
                <small>sees {seesText(item.id, chars.length)}</small>
              </header>
              {item.view.status === "error" ? (
                <p className="ntp-status ntp-status--error">Could not run: {item.view.error}</p>
              ) : item.ranked.length ? (
                <ol className={item.view.stale ? "is-stale" : ""} aria-label={`Top three next characters under ${MODEL_INFO[item.id].name}`}>
                  {item.ranked.map((entry) => {
                    const character = tokenText(entry.index, vocabulary);
                    return (
                      <li key={entry.index}>
                        <span className="sr-only">{`${describeToken(character)} ${formatPercent(entry.value)}`}</span>
                        <code aria-hidden="true">{displayToken(character)}</code>
                        <span aria-hidden="true"><i style={{ width: `${entry.value * 100}%` }} /></span>
                        <b aria-hidden="true">{formatPercent(entry.value)}</b>
                      </li>
                    );
                  })}
                </ol>
              ) : (
                <p className="ntp-status ntp-status--loading">{empty ? "No context yet." : "Loading…"}</p>
              )}
              <dl>
                <div>
                  <dt>Next-character entropy</dt>
                  <dd>{item.entropy === undefined ? "—" : `${item.entropy.toFixed(2)} bits`}</dd>
                </div>
                <div>
                  <dt>Mean surprisal on this text</dt>
                  <dd>{item.text ? `${item.text.bits.toFixed(2)} bits · ${item.text.count} chars` : "—"}</dd>
                </div>
                <div>
                  <dt>Held-out loss</dt>
                  <dd>
                    {MODEL_INFO[item.id].heldOut.toFixed(2)} bits <small>{MODEL_INFO[item.id].heldOutNote}</small>
                  </dd>
                </div>
                <div>
                  <dt>Whole-corpus loss</dt>
                  <dd>
                    {MODEL_INFO[item.id].wholeCorpus.toFixed(2)} bits <small>{MODEL_INFO[item.id].wholeNote}</small>
                  </dd>
                </div>
                <div>
                  <dt>Size and training</dt>
                  <dd>
                    {MODEL_INFO[item.id].params} <small>{MODEL_INFO[item.id].budget}</small>
                  </dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
        <p className="lab-note">
          Lower bits mean better prediction. Scored on the held-out {HELD_OUT_PERCENT}% of the corpus, the transformer is ahead of the
          GRU, which is ahead of the bigram table. That is not a fair test of the architectures: the transformer trained
          for {TRAINING_STEPS.transformer.toLocaleString("en-US")} steps and the shipped GRU for {TRAINING_STEPS.rnn}, so
          the lab cannot say how much of the gap is wiring and how much is training. The Mean surprisal row scores the
          text on the tape, and is only a fair comparison when none of the models trained on that text.
        </p>
      </LabSurface>

      <LabSurface label="Layer by layer" className="ntp-lens-card">
        <SurfaceHeading
          kicker={`Logit lens · the last character of the text · the ${LAYERS}-layer transformer`}
          title="What each stage of the model would predict"
        />
        <ol className="ntp-lens__pipe" aria-label="The path from characters to one sampled character">
          {PIPELINE.map((stage, index) => (
            <li key={stage.name} className={stage.tap === undefined ? "" : "is-tap"}>
              <span aria-hidden="true">{index + 1}</span>
              <strong>{stage.name}</strong>
              <small>{stage.detail}</small>
              {stage.tap !== undefined && <em>lens reads here</em>}
            </li>
          ))}
        </ol>
        <p className={`ntp-status ntp-status--${empty ? "error" : lens.status}`} role="status" aria-live="polite">
          {lensStatusText(lens.status, lens.error, empty)}
        </p>
        {lens.result && !empty ? (
          <div className={lens.stale ? "is-stale" : ""} aria-busy={lens.stale}>
            <p className="ntp-lens__summary">
              {lensSummary(lens.result.stages, lens.result.firstStage, lens.result.finalIndex)}
            </p>
            <div className="ntp-lens__panels">
              {lens.result.stages.map((stage, index) => (
                <LensPanel key={stage.id} stage={stage} index={index} finalIndex={lens.result!.finalIndex} first={index === lens.result!.firstStage} />
              ))}
            </div>
            <div className="metric-row">
              <Metric label="Finished model's first choice" value={`“${displayToken(tokenText(lens.result.finalIndex, vocabulary))}” · ${formatPercent(lens.result.stages[LENS_STAGES.length - 1].finalProbability)}`} />
              <Metric label="First on top after" value={LENS_STAGES[lens.result.firstStage].name.replace("After ", "")} tone="forward" />
              <Metric
                label="Entropy, stage by stage"
                value={`${lens.result.stages.map((stage) => stage.entropyBits.toFixed(2)).join(" → ")} bits`}
                tone="loss"
              />
              <Metric label="Characters read" value={`${lens.result.length} of at most ${metadata.transformer.config.block_size}`} />
            </div>
          </div>
        ) : (
          <div className="ntp-dist ntp-dist--empty" aria-hidden="true">
            {Array.from({ length: 4 }, (_, index) => (
              <span key={index} />
            ))}
          </div>
        )}
      </LabSurface>
    </div>
  );
}

function lensStatusText(status: "loading" | "ready" | "error", error: string | undefined, empty: boolean) {
  if (empty) return "Type at least one character: the lens reads the residual stream at the last one.";
  if (status === "error") return `The layer-by-layer view could not run (${error ?? "unknown error"}). The other cards still work.`;
  if (status === "loading") return "Reading the transformer's weights (5.5 MB) and running both layers one at a time…";
  return "Both layers ran in plain TypeScript on the weights inside tiny-transformer.onnx; the last row matches the model's own logits.";
}

function lensSummary(stages: ReadonlyArray<LensStage>, firstStage: number, finalIndex: number) {
  const answer = `“${displayToken(tokenText(finalIndex, vocabulary))}”`;
  const ranks = stages.map((stage) => `${ordinal(stage.finalRank)} ${stage.id === "embeddings" ? "after the embeddings" : `after ${stage.id.replace("-", " ")}`}`);
  return `The finished model's first choice, ${answer}, first heads the list ${LENS_STAGES[firstStage].name.toLowerCase()}. Where it ranks: ${ranks.join(", ")}.`;
}

function LensPanel({ stage, index, finalIndex, first }: { stage: LensStage; index: number; finalIndex: number; first: boolean }) {
  return (
    <article className={first ? "is-first" : ""} aria-label={`${stage.name}: next-character readout`}>
      <header>
        <strong>{stage.name}</strong>
        <small>{LENS_STAGES[index].detail}</small>
        {first && <em>final answer first on top here</em>}
      </header>
      <ol aria-label={`Top ${stage.top.length} next characters ${stage.name.toLowerCase()}`}>
        {stage.top.map((entry) => {
          const character = tokenText(entry.index, vocabulary);
          return (
            <li key={entry.index} className={entry.index === finalIndex ? "is-final" : ""}>
              <span className="sr-only">{`${describeToken(character)} ${formatPercent(entry.p)}${entry.index === finalIndex ? ", the finished model's choice" : ""}`}</span>
              <code aria-hidden="true">{displayToken(character)}</code>
              <span aria-hidden="true"><i style={{ width: `${entry.p * 100}%` }} /></span>
              <b aria-hidden="true">{formatPercent(entry.p)}</b>
            </li>
          );
        })}
      </ol>
      <dl>
        <div>
          <dt>Entropy of this readout</dt>
          <dd>{stage.entropyBits.toFixed(2)} bits</dd>
        </div>
        <div>
          <dt>{`Finished model's choice “${displayToken(tokenText(finalIndex, vocabulary))}”`}</dt>
          <dd>{`${ordinal(stage.finalRank)} of ${VOCAB_SIZE} · ${formatPercent(stage.finalProbability)}`}</dd>
        </div>
      </dl>
    </article>
  );
}
