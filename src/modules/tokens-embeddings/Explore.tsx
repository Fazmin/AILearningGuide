import { useEffect, useMemo, useState } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  VectorChip,
  type ModuleContext,
} from "@app/module-sdk";
import tokenizerJson from "./assets/bpe-tokenizer.json";
import { buildModel, displayToken, encode, symbolHex, type TokenizerJson } from "./bpe";
import { analogy, neighbourOverlap, neighboursOf, type ProjectionKind } from "./embeddings";
import { ProjectionMap, type MapMark } from "./ProjectionMap";
import { BASE_VOCAB, DEFAULT_TEXT, FULL_VOCAB } from "./state";
import { useEmbeddingSpace } from "./useEmbeddingSpace";

const MODEL = buildModel(tokenizerJson as unknown as TokenizerJson);

const SAMPLE_TEXTS = [
  DEFAULT_TEXT,
  "the The  the",
  "2024 was 33 years",
  "naïve café",
  "the end<eos>",
] as const;

const ANALOGY_PRESETS: ReadonlyArray<readonly [string, string, string]> = [
  ["king", "man", "woman"],
  ["paris", "france", "italy"],
  ["bigger", "big", "small"],
  ["tokyo", "japan", "germany"],
];

const NEIGHBOURS = 10;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const fmt = (value: number, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const count = (value: number) => value.toLocaleString("en-US");

function describeToken(token: { token: string; id: number; unknown: boolean; special: boolean; source: string }) {
  if (token.unknown) {
    const shown = displayToken(token.source);
    const hex = symbolHex(token.source);
    return `<unk>: byte ${hex ?? shown}${hex && shown !== hex ? ` (“${shown}”)` : ""} is not one of the 65 byte symbols this vocabulary kept`;
  }
  if (token.special) return `a reserved special token, matched as a literal string before any merge`;
  if (token.id < BASE_VOCAB) return `single byte symbol, part of the starting alphabet (IDs 4 to 68)`;
  return `created by merge rank ${token.id - BASE_VOCAB}, so its ID is 69 + ${token.id - BASE_VOCAB}`;
}

function WordField({
  label,
  value,
  known,
  onCommit,
  className = "",
}: {
  label: string;
  value: string;
  known: (word: string) => boolean | null;
  onCommit: (word: string) => void;
  className?: string;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const cleaned = draft.trim().toLowerCase();
  const status = cleaned === value ? true : known(cleaned);
  return (
    <label className={`te-field ${status === false ? "is-invalid" : ""} ${className}`.trim()}>
      <span>{label}</span>
      <input
        value={draft}
        spellCheck={false}
        autoCapitalize="off"
        aria-invalid={status === false}
        onChange={(event) => {
          const next = event.target.value;
          setDraft(next);
          const word = next.trim().toLowerCase();
          if (word && known(word)) onCommit(word);
        }}
        onBlur={() => setDraft(value)}
      />
      {status === false && <small role="alert">Not in the 10,000-word table</small>}
    </label>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const text = asString(state, "text", DEFAULT_TEXT);
  const vocabSize = clamp(Math.round(asNumber(state, "vocabSize", FULL_VOCAB)), BASE_VOCAB, FULL_VOCAB);
  const mergeLimit = vocabSize - BASE_VOCAB;
  const word = asString(state, "word", "queen");
  const analogyA = asString(state, "analogyA", "king");
  const analogyB = asString(state, "analogyB", "man");
  const analogyC = asString(state, "analogyC", "woman");
  const exclude = state.exclude !== false;
  const projection = (asString(state, "projection", "umap") === "pca" ? "pca" : "umap") as ProjectionKind;
  const mapShow = asString(state, "mapShow", "neighbours") === "analogy" ? "analogy" : "neighbours";
  const mapFrame = asString(state, "mapFrame", "fit") === "all" ? "all" : "fit";

  const encoding = useMemo(() => encode(MODEL, text, mergeLimit), [text, mergeLimit]);
  const fullEncoding = useMemo(() => encode(MODEL, text), [text]);
  const tokens = encoding.tokens;
  const focus = tokens.length ? clamp(Math.round(asNumber(state, "focus", 1)), 0, tokens.length - 1) : 0;
  const focused = tokens[focus];
  const wordIndex = focused?.wordIndex ?? 0;
  const trace = fullEncoding.words[wordIndex];
  const characters = Array.from(encoding.normalized).length;
  const unknownCount = tokens.filter((token) => token.unknown).length;
  const outsideSteps = trace ? trace.steps.filter((step) => step.rank >= mergeLimit).length : 0;

  const embeddings = useEmbeddingSpace();
  const space = embeddings.space;
  const known = (candidate: string) => (space ? space.index.has(candidate) : null);
  const queryRow = space?.index.get(word);
  const neighbours = useMemo(() => (space ? neighboursOf(space, word, NEIGHBOURS) : []), [space, word]);
  const arithmetic = useMemo(
    () => (space ? analogy(space, analogyA, analogyB, analogyC, 5) : null),
    [space, analogyA, analogyB, analogyC],
  );
  const shownResults = arithmetic ? (exclude ? arithmetic.results : arithmetic.unfiltered) : [];
  const overlap = useMemo(
    () => (space && queryRow !== undefined ? neighbourOverlap(space, projection, queryRow, NEIGHBOURS) : null),
    [space, queryRow, projection],
  );

  const marks: MapMark[] = useMemo(() => {
    if (!space) return [];
    if (mapShow === "analogy") {
      const list: MapMark[] = [];
      const push = (candidate: string | undefined, role: MapMark["role"]) => {
        const index = candidate ? space.index.get(candidate) : undefined;
        if (index !== undefined && !list.some((mark) => mark.index === index)) list.push({ index, word: candidate!, role });
      };
      push(analogyA, "a");
      push(analogyB, "b");
      push(analogyC, "c");
      push(arithmetic?.results[0]?.word, "result");
      return list;
    }
    if (queryRow === undefined) return [];
    return [
      { index: queryRow, word, role: "query" },
      ...neighbours.map((entry) => ({ index: entry.index, word: entry.word, role: "neighbour" as const })),
    ];
  }, [space, mapShow, analogyA, analogyB, analogyC, arithmetic, queryRow, word, neighbours]);

  const statusText =
    embeddings.status === "ready" ? "Loaded" : embeddings.status === "error" ? "Unavailable" : "Loading…";
  const statusBadge = (live: boolean) => (
    <span
      className={`te-status te-status--${embeddings.status}`}
      role={live ? "status" : undefined}
      title={embeddings.error || undefined}
      aria-label={
        embeddings.status === "ready"
          ? `Vectors loaded: ${count(space?.words.length ?? 0)} words, ${space?.dims ?? 0} dimensions each`
          : embeddings.status === "error"
            ? `Vectors unavailable: ${embeddings.error}`
            : "Loading 2 megabytes of word vectors"
      }
    >
      <i aria-hidden="true" />
      {statusText}
    </span>
  );
  const errorNote =
    embeddings.status === "error" ? (
      <p className="lab-note">The word vectors did not load. Use Try loading again on Nearest neighbours.</p>
    ) : null;
  const errorPanel =
    embeddings.status === "error" ? (
      <div className="te-error">
        <p>
          The embedding files could not be read ({embeddings.error}). The tokenizer above does not need them and still
          works.
        </p>
        <button type="button" className="te-button" onClick={embeddings.retry}>
          Try loading again
        </button>
      </div>
    ) : null;

  const setText = (next: string) => {
    const nextCount = encode(MODEL, next, mergeLimit).tokens.length;
    setState({ text: next });
    narrate(`${nextCount} tokens at a ${vocabSize}-entry vocabulary.`);
  };

  return (
    <div className="te-lab">
      <LabSurface label="Tokenizer workbench" className="te-card te-card--wide">
        <SurfaceHeading
          kicker="Byte-level BPE · trained on Tiny Shakespeare"
          title="Text becomes a sequence of vocabulary IDs"
          aside={<span className="te-badge">{vocabSize} entries</span>}
        />
        <label className="te-field te-field--text">
          <span>Text to tokenize</span>
          <input value={text} maxLength={200} spellCheck={false} onChange={(event) => setText(event.target.value)} />
        </label>
        <div className="te-samples" role="group" aria-label="Sample texts">
          <span>Try</span>
          {SAMPLE_TEXTS.map((sample) => (
            <button
              type="button"
              key={sample}
              aria-pressed={text === sample}
              onClick={() => setText(sample)}
            >
              {sample.replace(/ {2}/g, " ␣")}
            </button>
          ))}
        </div>
        <RangeControl
          label="Vocabulary size"
          min={BASE_VOCAB}
          max={FULL_VOCAB}
          step={1}
          value={vocabSize}
          format={(value) => `${value} · ${value - BASE_VOCAB} merges`}
          onChange={(value) => {
            // Keep the same word selected while its tokens split or merge.
            const next = encode(MODEL, text, value - BASE_VOCAB).tokens.findIndex(
              (token) => token.wordIndex === wordIndex,
            );
            setState({ vocabSize: value, focus: next >= 0 ? next : 0 });
          }}
        />
        <div className="te-stream" role="group" aria-label={`${tokens.length} tokens. Select one to inspect its ID and merges.`}>
          {encoding.words.map((entry, index) => (
            <span
              className={`te-word ${index === wordIndex ? "is-focus-word" : ""}`.trim()}
              key={`${index}-${entry.text}`}
              data-word={index}
            >
              {entry.tokens.map((token, offset) => {
                const tokenIndex = tokens.indexOf(token);
                const shown = token.unknown ? "<unk>" : displayToken(token.token);
                return (
                  <button
                    type="button"
                    key={offset}
                    className={`te-chip ${token.unknown ? "te-chip--unk" : ""} ${token.special ? "te-chip--special" : ""}`.trim()}
                    aria-pressed={tokenIndex === focus}
                    aria-label={`Token ${tokenIndex + 1} of ${tokens.length}: ${shown}, ID ${token.id}`}
                    onClick={() => setState({ focus: tokenIndex })}
                  >
                    <span className="te-chip__id">{token.id}</span>
                    <span className="te-chip__text">{shown}</span>
                  </button>
                );
              })}
            </span>
          ))}
          {tokens.length === 0 && <p className="lab-note">Type some text to tokenize it.</p>}
        </div>
        <div className="metric-row te-metrics">
          <Metric label="Tokens" value={count(tokens.length)} tone="forward" />
          <Metric label="Characters" value={count(characters)} />
          <Metric label="Characters per token" value={tokens.length ? (characters / tokens.length).toFixed(2) : "—"} />
          <Metric label="Unknown bytes" value={count(unknownCount)} tone={unknownCount ? "loss" : undefined} />
        </div>
        {focused && (
          <p className="te-focus-line">
            <strong>
              {focused.unknown ? "<unk>" : displayToken(focused.token)} → ID {focused.id}
            </strong>
            <span>{describeToken(focused)}. The model would read row {focused.id} of a {vocabSize}-row embedding matrix.</span>
          </p>
        )}
      </LabSurface>

      <LabSurface label="Merge trace" className="te-card te-card--wide">
        <SurfaceHeading
          kicker={
            trace
              ? `Pre-token ${wordIndex + 1} of ${encoding.words.length} · “${displayToken(trace.symbolsText)}”`
              : "No text"
          }
          title="Replay the merge table on the selected word"
          aside={
            trace ? (
              <span className="te-badge">
                {trace.steps.length - outsideSteps} of {trace.steps.length} merges in budget
              </span>
            ) : undefined
          }
        />
        {trace && (
          <ol className="te-ladder" aria-label={`Merge sequence for ${displayToken(trace.symbolsText)}`}>
            <li className="te-rung te-rung--start">
              <span className="te-rung__rank">bytes</span>
              <span className="te-rung__rule">
                {trace.initial.length} byte symbol{trace.initial.length === 1 ? "" : "s"}
              </span>
              <span className="te-rung__symbols">
                {trace.initial.map((symbol, index) => (
                  <i key={index} className={MODEL.vocab.has(symbol) ? "" : "is-unknown"}>
                    {MODEL.vocab.has(symbol) ? displayToken(symbol) : "<unk>"}
                  </i>
                ))}
              </span>
            </li>
            {trace.steps.map((step) => {
              const inBudget = step.rank < mergeLimit;
              return (
                <li key={step.rank} className={`te-rung ${inBudget ? "" : "is-outside"}`.trim()}>
                  <span className="te-rung__rank">#{step.rank}</span>
                  <span className="te-rung__rule">
                    <code>{displayToken(step.left)}</code> + <code>{displayToken(step.right)}</code> →{" "}
                    <code>{displayToken(step.merged)}</code>
                    <small>
                      ID {step.id}
                      {step.count > 1 ? ` · ×${step.count}` : ""}
                      {inBudget ? "" : ` · needs vocabulary ≥ ${step.id + 1}`}
                    </small>
                  </span>
                  <span className="te-rung__symbols">
                    {step.symbols.map((symbol, index) => (
                      <i key={index} className={symbol === step.merged ? "is-new" : ""}>
                        {displayToken(symbol)}
                      </i>
                    ))}
                  </span>
                </li>
              );
            })}
            {trace.steps.length === 0 && (
              <li className="te-rung te-rung--none">
                <span className="te-rung__rank">—</span>
                <span className="te-rung__rule">No adjacent pair in this word appears in the merge table.</span>
              </li>
            )}
          </ol>
        )}
        <p className="lab-note">
          At each rung the adjacent pair with the lowest merge rank joins, everywhere it occurs in the word. Ranks only
          rise, so shrinking the vocabulary cuts the list from the bottom: faded rungs are merges this {vocabSize}-entry
          vocabulary never learned.
        </p>
      </LabSurface>

      <LabSurface label="Nearest neighbours" className="te-card">
        <SurfaceHeading kicker="GloVe · 10,000 words × 50 dimensions" title="Which rows point the same way?" aside={statusBadge(true)} />
        <WordField label="Query word" value={word} known={known} onCommit={(next) => setState({ word: next })} />
        {errorPanel}
        {space && queryRow === undefined && <p className="lab-note">“{word}” is not among the 10,000 words.</p>}
        {space && queryRow !== undefined && (
          <>
            <BarList
              label={`Ten nearest words to ${word} by cosine similarity`}
              items={neighbours.map((entry) => ({
                id: entry.word,
                label: entry.word,
                value: Math.max(0, entry.cosine),
                display: fmt(entry.cosine),
                tone: "attention" as const,
              }))}
              max={1}
              onSelect={(next) => setState({ word: next })}
            />
            <VectorChip
              label={`${word} · first 6 of ${space.dims} numbers`}
              values={Array.from(space.vectors.subarray(queryRow * space.dims, queryRow * space.dims + 6))}
              tone="forward"
            />
            <div className="metric-row">
              <Metric label="Frequency rank" value={`${count(queryRow + 1)} of ${count(space.words.length)}`} />
              <Metric label="Vector length ‖v‖" value={fmt(space.norms[queryRow], 2)} />
            </div>
          </>
        )}
        {!space && embeddings.status === "loading" && <div className="te-skeleton" aria-hidden="true" />}
      </LabSurface>

      <LabSurface label="Vector arithmetic" className="te-card">
        <SurfaceHeading kicker="3CosAdd on unit-length vectors" title="Add a direction, then find the nearest word" aside={statusBadge(false)} />
        <div className="te-equation">
          <WordField label="A" value={analogyA} known={known} onCommit={(next) => setState({ analogyA: next })} />
          <span aria-hidden="true">−</span>
          <WordField label="B" value={analogyB} known={known} onCommit={(next) => setState({ analogyB: next })} />
          <span aria-hidden="true">+</span>
          <WordField label="C" value={analogyC} known={known} onCommit={(next) => setState({ analogyC: next })} />
        </div>
        <div className="te-samples" role="group" aria-label="Analogy presets">
          <span>Try</span>
          {ANALOGY_PRESETS.map(([a, b, c]) => (
            <button
              type="button"
              key={a}
              aria-pressed={analogyA === a && analogyB === b && analogyC === c}
              onClick={() => setState({ analogyA: a, analogyB: b, analogyC: c })}
            >
              {a} − {b} + {c}
            </button>
          ))}
        </div>
        <SegmentedControl
          label="Input words"
          value={exclude ? "exclude" : "allow"}
          options={[
            { value: "exclude", label: "Excluded" },
            { value: "allow", label: "Allowed to win" },
          ]}
          onChange={(value) => setState({ exclude: value === "exclude" })}
        />
        {errorNote}
        {space && !arithmetic && <p className="lab-note">All three words must be in the 10,000-word table.</p>}
        {arithmetic && (
          <>
            <BarList
              label={`Nearest words to ${analogyA} minus ${analogyB} plus ${analogyC}`}
              items={shownResults.map((entry, index) => ({
                id: entry.word,
                label: entry.word,
                value: Math.max(0, entry.cosine),
                display: fmt(entry.cosine),
                tone: [analogyA, analogyB, analogyC].includes(entry.word) ? ("muted" as const) : ("gradient" as const),
                emphasis: index === 0,
                detail: [analogyA, analogyB, analogyC].includes(entry.word) ? "input word" : undefined,
              }))}
              max={1}
            />
            <p className="te-result-line">
              Top answer: <strong>{shownResults[0]?.word}</strong> at cosine {fmt(shownResults[0]?.cosine ?? NaN)}.{" "}
              {exclude
                ? `If the inputs may win, the top answer is ${arithmetic.unfiltered[0]?.word} (${fmt(arithmetic.unfiltered[0]?.cosine ?? NaN)}).`
                : "Most published analogy scores exclude the three inputs, which changes the answer here."}
            </p>
          </>
        )}
        {!space && embeddings.status === "loading" && <div className="te-skeleton" aria-hidden="true" />}
      </LabSurface>

      <LabSurface label="Embedding projection" className="te-card te-card--wide">
        <SurfaceHeading
          kicker={`${projection === "umap" ? "UMAP" : "PCA"} projection of all 10,000 words · 50 → 2 dimensions`}
          title="A flat map that keeps some neighbourhoods and breaks others"
          aside={
            overlap && mapShow === "neighbours" ? (
              <span className={`te-badge ${overlap.kept <= 2 ? "is-warning" : ""}`.trim()}>
                keeps {overlap.kept} of {overlap.of} neighbours
              </span>
            ) : undefined
          }
        />
        <div className="te-map-controls">
          <SegmentedControl
            label="Projection"
            value={projection}
            options={[
              { value: "umap", label: "UMAP" },
              { value: "pca", label: "PCA" },
            ]}
            onChange={(value) => setState({ projection: value })}
          />
          <SegmentedControl
            label="Highlight"
            value={mapShow}
            options={[
              { value: "neighbours", label: `${word} + neighbours` },
              { value: "analogy", label: "Analogy words" },
            ]}
            onChange={(value) => setState({ mapShow: value })}
          />
          <SegmentedControl
            label="Frame"
            value={mapFrame}
            options={[
              { value: "fit", label: "Fit highlights" },
              { value: "all", label: "All words" },
            ]}
            onChange={(value) => setState({ mapFrame: value })}
          />
        </div>
        {errorNote}
        <ProjectionMap
          space={space}
          kind={projection}
          marks={marks}
          frame={mapFrame}
          mode={mapShow}
          label={
            mapShow === "neighbours"
              ? `${projection.toUpperCase()} map: ${word} with its ten nearest 50-dimensional neighbours. ${
                  overlap ? `${overlap.kept} of ${overlap.of} are also among its ten nearest points on this map.` : ""
                }`
              : `${projection.toUpperCase()} map of ${analogyA}, ${analogyB}, ${analogyC}${
                  arithmetic?.results[0] ? ` and ${arithmetic.results[0].word}` : ""
                } with the two offset arrows.`
          }
        />
        <p className="lab-note">
          {mapShow === "neighbours"
            ? overlap
              ? `Dashed spokes join ${word} to its ten nearest words by 50-D cosine. On this map only ${overlap.kept} of those ten are also among its ten nearest dots; long spokes are neighbours the projection pulled apart.`
              : "Dashed spokes join the query word to its ten nearest words by 50-D cosine."
            : `Solid arrows run ${analogyB} → ${analogyA} and ${analogyC} → ${arithmetic?.results[0]?.word ?? "the answer"}. The dashed arrow copies the first arrow's 2-D offset onto ${analogyC}; if the map preserved the analogy, it would end on the answer.`}
        </p>
      </LabSurface>
    </div>
  );
}
