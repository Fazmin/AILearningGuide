import vocabulary from "./assets/transformer-vocab.json";
import { useEffect, useMemo } from "react";
import {
  ArcDiagram,
  FormulaWithValues,
  Heatmap,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  StepThroughController,
  SurfaceHeading,
  teachingAssetUrl,
  useTeachingTransformer,
  VectorChip,
  type ModuleContext,
} from "@app/module-sdk";
import {
  aggregateByWord,
  attentionMatrix,
  attentionRow,
  causalRow,
  characterLabel,
  fallbackVectors,
  headMatrix,
  largestAbsoluteDifference,
  laterShare,
  mixRow,
  saturatedRows,
  unownedShare,
  vectorLength,
  whitespaceShare,
  wordSpans,
  zeroValues,
} from "./attention-math";
import { CharacterHeatmap, SoftmaxStrip } from "./AttentionCharts";
import {
  DEFAULT_TEXT,
  EDIT_DIMENSIONS,
  MAX_CHARACTERS,
  MODEL_HEADS,
  readState,
} from "./state";

const modelUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");
const stages = ["Query", "Keys", "Score", "Softmax", "Mix"] as const;
/** Shell step -> inspector stage. Steps 4 (mask) and 5 (test a route) follow the softmax and query views. */
const STEP_STAGES = [0, 1, 3, 4, 3, 0] as const;

/**
 * Prepared inputs. `query` and `key` name the words selected when one is pressed; a word that
 * appears twice resolves to its last occurrence for the query and its first for the key.
 */
const EXAMPLES = [
  { id: "pronoun", label: "Pronoun sentence", text: DEFAULT_TEXT, view: "words", query: "it", key: "animal" },
  {
    id: "name",
    label: "Repeated name",
    text: "Mark Jones met Anna Blake and then Mark",
    view: "words",
    query: "Mark",
    key: "Jones",
  },
  {
    id: "moved",
    label: "Name moved earlier",
    text: "Anna Blake met Mark Jones and then Mark",
    view: "words",
    query: "Mark",
    key: "Jones",
  },
  { id: "repeat8", label: "Repeats every 8", text: "zqxjkvbwzqxjkvbwzqxjkvbwzqxjkvbw", view: "characters", query: "", key: "" },
  { id: "repeat9", label: "Repeats every 9", text: "zqxjkvbwmzqxjkvbwmzqxjkvbwmzqxjk", view: "characters", query: "", key: "" },
] as const;

const clamp = (value: number, low: number, high: number) =>
  Math.max(low, Math.min(high, value));

const percent = (value: number, digits = 1) =>
  `${((Number.isFinite(value) ? value : 0) * 100).toFixed(digits)}%`;

export default function Explore({
  state,
  setState,
  currentStep,
  narrate,
}: ModuleContext) {
  const controls = readState(state);
  const { text, layer, head, stage, editVectors, edits } = controls;
  const view = controls.matrixView;
  const causal = controls.mask === "causal";
  const useScale = controls.scaling === "scaled";
  const valueMode = controls.values;
  const outerStage = STEP_STAGES[currentStep] ?? 0;

  useEffect(() => {
    if (stage !== outerStage) setState({ stage: outerStage });
    // Only the shell's step should move the walkthrough here.
  }, [currentStep]);

  const characters = useMemo(() => Array.from(text).slice(0, MAX_CHARACTERS), [text]);
  const inputIds = useMemo(() => {
    const stoi = vocabulary.stoi as Record<string, number>;
    return characters.map((character) => stoi[character] ?? vocabulary.unk);
  }, [characters]);
  const spans = useMemo(() => wordSpans(characters), [characters]);
  const words = spans.map((span) => span.text);
  const runtime = useTeachingTransformer({ modelUrl, inputIds });

  const selected = clamp(controls.selected, 0, Math.max(0, spans.length - 1));
  const inspected = clamp(controls.inspected, 0, Math.max(0, spans.length - 1));

  // Model tensors are used only when they belong to the current text.
  const result = runtime.result;
  const modelReady =
    runtime.status === "ready" &&
    result !== undefined &&
    result.qkvShape[3] === characters.length &&
    result.attentionShape[3] === characters.length;

  const headData = useMemo(() => {
    const fromModel = (tensor: Float32Array | undefined, shape: readonly number[]) =>
      modelReady ? headMatrix(tensor, shape, layer, head) : undefined;
    const q = fromModel(result?.q, result?.qkvShape ?? []) ?? fallbackVectors(characters, layer, head, 0);
    const k = fromModel(result?.k, result?.qkvShape ?? []) ?? fallbackVectors(characters, layer, head, 1);
    const v = fromModel(result?.v, result?.qkvShape ?? []) ?? fallbackVectors(characters, layer, head, 2);
    const attention = fromModel(result?.attention, result?.attentionShape ?? []) ?? attentionMatrix(q, k);
    return { q, k, v, attention };
  }, [characters, head, layer, modelReady, result]);

  // The grid the lab draws: the model's own causal tensor, or the same head's weights recomputed
  // from the model's own Q and K once the mask is lifted or the scaling is switched off.
  const baseMatrix = useMemo(
    () =>
      causal && useScale
        ? headData.attention
        : attentionMatrix(headData.q, headData.k, { causal, scaled: useScale }),
    [causal, headData, useScale],
  );

  const querySpan = spans[selected];
  const keySpan = spans[inspected];
  const qPos = querySpan?.representative ?? 0;
  const kPos = keySpan?.representative ?? 0;
  const headWidth = headData.q[0]?.length ?? 64;
  const visibleCount = causal ? qPos + 1 : characters.length;

  const baseQuery = headData.q[qPos] ?? [];
  const query = baseQuery.map((value, index) => {
    const override = edits[`q${index}`];
    return editVectors && index < EDIT_DIMENSIONS && override !== undefined ? override : value;
  });
  const keys = headData.k.map((key, position) =>
    position !== kPos
      ? key
      : key.map((value, index) => {
          const override = edits[`k${index}`];
          return editVectors && index < EDIT_DIMENSIONS && override !== undefined ? override : value;
        }),
  );
  const recomputed = attentionRow(query, keys, qPos, { causal, scaled: useScale });
  const modelCausalRow = headData.attention[qPos] ?? [];
  const modelRow = baseMatrix[qPos] ?? recomputed.weights;
  const unedited = causalRow(baseQuery, headData.k, qPos).weights;
  const recomputeGap = largestAbsoluteDifference(
    unedited.slice(0, qPos + 1),
    modelCausalRow.slice(0, qPos + 1),
  );
  const activeRow = editVectors ? recomputed.weights : modelRow;

  const wordMatrix = spans.map((span, index) =>
    index === selected && editVectors
      ? aggregateByWord(activeRow, spans)
      : aggregateByWord(baseMatrix[span.representative] ?? [], spans),
  );
  const wordWeights = wordMatrix[selected] ?? [];
  const droppedShare = unownedShare(activeRow, spans);

  // The value vectors the mix reads. Zeroing changes the payload and nothing else: the weights above
  // come from Q and K and are computed before V is touched.
  const values = useMemo(() => {
    if (valueMode === "all") return zeroValues(headData.v, 0, headData.v.length);
    if (valueMode === "inspected" && keySpan) return zeroValues(headData.v, keySpan.start, keySpan.end);
    return headData.v;
  }, [headData.v, keySpan, valueMode]);
  const mixed = mixRow(activeRow, values);
  const modelMixed = mixRow(activeRow, headData.v);
  const mixLength = vectorLength(mixed);

  const spaces = whitespaceShare(activeRow.slice(0, visibleCount), characters);
  const later = causal ? 0 : laterShare(activeRow, qPos);
  const nextWeight = !causal && qPos + 1 < characters.length ? activeRow[qPos + 1] : undefined;
  const strongest = wordWeights.length
    ? wordWeights.indexOf(Math.max(...wordWeights))
    : 0;
  const keyMasked = causal && kPos > qPos;
  const score = recomputed.scores[kPos] ?? 0;
  const scaled = recomputed.scaledScores[kPos] ?? 0;
  const characterWeight = recomputed.weights[kPos] ?? 0;
  const modelCharacterWeight = modelCausalRow[kPos] ?? 0;
  const wordWeight = wordWeights[inspected] ?? 0;
  const keyCharacters = keySpan ? keySpan.end - keySpan.start : 0;
  const largestWeight = Math.max(0, ...activeRow.slice(0, visibleCount));
  const saturation = saturatedRows(baseMatrix, causal);
  const scaleFactor = Number(Math.sqrt(headWidth).toFixed(2));

  const statusText =
    runtime.status === "ready"
      ? `ONNX model · ${runtime.result?.provider.toUpperCase()}`
      : runtime.status === "loading"
        ? "Loading teaching model…"
        : runtime.status === "error"
          ? "Model unavailable · inspectable fallback"
          : "Waiting for input";
  const sourceNote = modelReady || characters.length === 0
    ? undefined
    : runtime.status === "error"
      ? "Stand-in values: the model failed to load, so every weight here comes from a deterministic sine pattern with no learned meaning."
      : "Stand-in values while the model runs: these weights come from a deterministic sine pattern, not from training.";
  const maskNote = causal
    ? undefined
    : layer === 0
      ? "Mask lifted for this head only. The shipped model was trained causal, so these are the weights Layer 1 head " +
        `${head + 1} gives if it may read the characters after each query, not what a trained bidirectional model would learn. ` +
        "Layer 1 reads only the embeddings, so for this head the numbers are exact."
      : "Mask lifted for this head only. The shipped model was trained causal, so these are the weights Layer 2 head " +
        `${head + 1} gives if it may read the characters after each query, not what a trained bidirectional model would learn. ` +
        "Its queries and keys were still built from a Layer 1 that stayed causal.";

  const queryLabel = querySpan
    ? `“${characterLabel(characters[qPos] ?? "")}” ending ${querySpan.text}`
    : "query";
  const keyLabel = keySpan
    ? `“${characterLabel(characters[kPos] ?? "")}” ending ${keySpan.text}`
    : "key";

  const characterSummary = querySpan
    ? `Character attention matrix for layer ${layer + 1}, head ${head + 1}, ${characters.length} characters${causal ? "" : ", causal mask lifted"}${useScale ? "" : ", scaling off"}. Highlighted row: the last character of ${querySpan.text}, which puts ${percent(spaces)} of its weight on spaces. Its top keys: ${activeRow
        .slice(0, visibleCount)
        .map((weight, position) => ({ weight, position }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 3)
        .map(({ weight, position }) => `“${characterLabel(characters[position] ?? "")}” at position ${position}, ${percent(weight)}`)
        .join("; ")}.`
    : "No characters to show.";

  const valueNote =
    valueMode === "all"
      ? "Every value is zero. The weights above are unchanged because they come from Q and K, and the mix is the zero vector."
      : valueMode === "inspected"
        ? `The ${keyCharacters} characters of “${keySpan?.text ?? "the key"}” (the ones its Word total sums over) now carry zero values. The weights are unchanged; the mix loses that word's share.`
        : "The model's own values. Zero them to see that the weights stay put and only the payload changes.";

  return (
    <div className="phase4-attention">
      <LabSurface
        label="Model attention routing diagram"
        className="attention-scope"
      >
        <SurfaceHeading
          kicker={`Layer ${layer + 1} · head ${head + 1}`}
          title="Trace where one token retrieves context"
          aside={
            <span
              className={`model-runtime-status model-runtime-status--${runtime.status}`}
              role="status"
              title={runtime.error || undefined}
            >
              <i />
              {statusText}
            </span>
          }
        />

        <label className="attention-sentence-input">
          <span>Sentence</span>
          <input
            value={text}
            maxLength={MAX_CHARACTERS}
            onChange={(event) =>
              setState({
                text: event.target.value,
                selected: 0,
                inspected: 0,
                stage: 0,
              })
            }
          />
          <small>{characters.length}/{MAX_CHARACTERS} characters</small>
        </label>
        <div className="atn-examples" role="group" aria-label="Example inputs">
          <span>Examples</span>
          {EXAMPLES.map((example) => {
            const exampleSpans = wordSpans(Array.from(example.text));
            const exampleWords = exampleSpans.map((span) => span.text);
            const queryIndex = example.query ? exampleWords.lastIndexOf(example.query) : -1;
            const keyIndex = example.key ? exampleWords.indexOf(example.key) : -1;
            return (
              <button
                type="button"
                key={example.id}
                aria-pressed={text === example.text}
                onClick={() => {
                  // An example is an experiment with the model as shipped, so the mask, the scaling
                  // and the value vectors go back to the model's own settings with it.
                  setState({
                    text: example.text,
                    selected: queryIndex >= 0 ? queryIndex : Math.max(0, exampleSpans.length - 1),
                    inspected: keyIndex >= 0 ? keyIndex : 0,
                    matrixView: example.view,
                    editVectors: false,
                    mask: "causal",
                    scaling: "scaled",
                    values: "model",
                    stage: 0,
                  });
                  narrate(`Loaded the ${example.label} example with the model's own mask, scaling and values.`);
                }}
              >
                {example.label}
              </button>
            );
          })}
        </div>

        {words.length > 0 ? (
          <div className={`attention-arc-scroll ${modelReady ? "" : "is-pending"}`.trim()}>
            <ArcDiagram
              tokens={words}
              weights={wordWeights}
              queryIndex={selected}
              inspectedIndex={inspected}
              label="Choose the query token"
              onQueryChange={(index) => {
                setState({ selected: index, stage: 0 });
                const row = wordMatrix[index] ?? [];
                narrate(
                  `${words[index]} is now the query. Its strongest route is to ${words[row.indexOf(Math.max(...row))] ?? "no token"}.`,
                );
              }}
            />
          </div>
        ) : (
          <div className="attention-empty">Enter a few words to run the model.</div>
        )}

        <div className="attention-scope__readout">
          <span>query <strong>{words[selected] ?? "—"}</strong></span>
          <span>inspected key <strong>{words[inspected] ?? "—"}</strong></span>
          <span>strongest route <strong>{words[strongest] ?? "—"}</strong></span>
          <span>weight <strong>{percent(wordWeights[strongest] ?? 0)}</strong></span>
          <span>on spaces <strong>{percent(spaces)}</strong></span>
          {!causal && (
            <>
              <span>on later characters <strong>{percent(later)}</strong></span>
              <span>
                on the next character{" "}
                <strong>{nextWeight === undefined ? "none" : percent(nextWeight)}</strong>
              </span>
            </>
          )}
        </div>

        <StepThroughController
          label="Attention calculation stages"
          steps={stages}
          current={stage}
          onChange={(next) => {
            setState({ stage: next });
            narrate(`Attention stage ${next + 1}: ${stages[next]}.`);
          }}
        />
      </LabSurface>

      <div className="attention-inspector">
        <LabSurface
          label="Attention matrix"
          className="attention-heatmap-panel"
        >
          <SurfaceHeading
            kicker={
              view === "words"
                ? `Words · ${spans.length} × ${spans.length}${editVectors ? " · edited row" : ""}${causal ? "" : " · mask lifted"}`
                : `Characters · ${characters.length} × ${characters.length}${causal ? "" : " · mask lifted"}`
            }
            title={view === "words" ? "Select a cell to inspect its vectors" : "What the model actually computes"}
          />
          <div className="atn-view-control">
            <SegmentedControl
              label="Matrix view"
              value={view}
              options={[
                { value: "words", label: "Words" },
                { value: "characters", label: "Characters" },
              ]}
              onChange={(value) => setState({ matrixView: value })}
            />
          </div>
          <div className="atn-view-control atn-mask-control">
            <SegmentedControl
              label="Mask"
              value={controls.mask}
              options={[
                { value: "causal", label: "Causal" },
                { value: "bidirectional", label: "Bidirectional" },
              ]}
              onChange={(value) => {
                setState({ mask: value });
                narrate(
                  value === "causal"
                    ? "Causal mask restored: each character reads itself and earlier characters only."
                    : "Causal mask lifted for this head: each character can now read every character, later ones included.",
                );
              }}
            />
          </div>
          {sourceNote && <p className="atn-source-note" role="note">{sourceNote}</p>}
          {maskNote && <p className="atn-source-note" role="note">{maskNote}</p>}
          <div className={modelReady ? "" : "is-pending"}>
            {view === "words" ? (
              <Heatmap
                rows={words}
                columns={words}
                values={wordMatrix}
                selected={{ row: selected, column: inspected }}
                label={`Attention matrix for layer ${layer + 1}, head ${head + 1}`}
                onSelect={(row, column) => {
                  setState({ selected: row, inspected: column, stage: 2 });
                  narrate(
                    `${words[row]} to ${words[column]} has attention weight ${percent(wordMatrix[row]?.[column] ?? 0)}.`,
                  );
                }}
              />
            ) : characters.length > 0 ? (
              <CharacterHeatmap
                characters={characters}
                matrix={baseMatrix}
                queryRow={qPos}
                keySpan={keySpan}
                label={characterSummary}
                causal={causal}
              />
            ) : null}
          </div>
          <p className="atn-matrix-note">
            {view === "words"
              ? causal
                ? "Each word owns the space in front of it, so a row's word cells add up to exactly 100%."
                : `Each word owns the space in front of it. With the mask lifted a row also reads later words, so its cells add up to 100%${droppedShare > 0.0005 ? ` less ${percent(droppedShare)} that falls on trailing spaces or past the first 12 words and is not drawn` : ""}.`
              : causal
                ? `Rows are query characters, columns are key characters, darker means more weight. Hatched cells are masked. Outlined: the row for ${querySpan?.text ?? "the query"} and the columns of ${keySpan?.text ?? "the key"}. Axis ticks every 8 positions.`
                : `Rows are query characters, columns are key characters, darker means more weight. Nothing is hatched: the mask is lifted, so the cells above the diagonal can carry weight. Outlined: the row for ${querySpan?.text ?? "the query"} and the columns of ${keySpan?.text ?? "the key"}. Axis ticks every 8 positions.`}
          </p>
          <details className="attention-data-table">
            <summary>View weights as text</summary>
            {view === "words" ? (
              <table>
                <thead>
                  <tr><th>Key word</th><th>Weight from {words[selected] ?? "query"}</th></tr>
                </thead>
                <tbody>
                  {words.map((word, index) => (
                    <tr key={`${word}-${index}`}>
                      <th>{word}</th>
                      <td>{percent(wordWeights[index] ?? 0, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <table>
                <thead>
                  <tr><th>Position</th><th>Key character</th><th>Weight from {queryLabel}</th></tr>
                </thead>
                <tbody>
                  {characters.slice(0, visibleCount).map((character, index) => (
                    <tr key={index}>
                      <th>{index}</th>
                      <td>{characterLabel(character)}</td>
                      <td>{percent(activeRow[index] ?? 0, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </details>
        </LabSurface>

        <LabSurface
          label="Query key value inspector"
          className="qkv-inspector"
        >
          <SurfaceHeading
            kicker={`Stage ${stage + 1} of ${stages.length}`}
            title={stages[stage]}
          />
          <div className="atn-stage" data-stage={stage}>
            <div className="attention-layer-controls">
              <SegmentedControl
                label="Layer"
                value={String(layer)}
                options={[
                  { value: "0", label: "Layer 1" },
                  { value: "1", label: "Layer 2" },
                ]}
                onChange={(value) => setState({ layer: Number(value), stage: 0 })}
              />
              <RangeControl
                label="Attention head"
                min={0}
                max={MODEL_HEADS - 1}
                step={1}
                value={head}
                format={(value) => `${value + 1} / ${MODEL_HEADS}`}
                onChange={(value) => setState({ head: value, stage: 0 })}
              />
            </div>

            <div className="atn-stage__grid">
              <div className="atn-stage__col">
                <div className="vector-workbench">
                  <div className="atn-part atn-part--query">
                    <VectorChip
                      label={`Q · ${queryLabel}`}
                      values={query.slice(0, EDIT_DIMENSIONS)}
                      editable
                      onChange={(index, value) =>
                        setState({ [`q${index}`]: value, editVectors: true })
                      }
                    />
                  </div>
                  <span className="vector-operator">·</span>
                  <div className="atn-part atn-part--key">
                    <VectorChip
                      label={`K · ${keyLabel}`}
                      values={keys[kPos]?.slice(0, EDIT_DIMENSIONS) ?? []}
                      editable
                      onChange={(index, value) =>
                        setState({ [`k${index}`]: value, editVectors: true })
                      }
                    />
                  </div>
                </div>
                <p className="vector-edit-note">
                  Edit the first {EDIT_DIMENSIONS} of {headWidth} dimensions. The other {headWidth - EDIT_DIMENSIONS} learned dimensions stay fixed.
                </p>
                {editVectors && (
                  <button
                    type="button"
                    className="quiet-action"
                    onClick={() => {
                      setState({ editVectors: false });
                    }}
                  >
                    Match the model vectors
                  </button>
                )}

                <div className="atn-part atn-part--strip">
                  <SoftmaxStrip
                    characters={characters}
                    scaledScores={recomputed.scaledScores}
                    weights={recomputed.weights}
                    queryPosition={qPos}
                    keyPosition={kPos}
                    keySpan={keySpan}
                    causal={causal}
                    scaled={useScale}
                  />
                </div>
              </div>
              <div className="atn-stage__col">
                <div className="atn-view-control atn-scale-control">
                  <SegmentedControl
                    label="Score scaling"
                    value={controls.scaling}
                    options={[
                      { value: "scaled", label: `÷ √${headWidth}` },
                      { value: "raw", label: "No scaling" },
                    ]}
                    onChange={(value) => {
                      setState({ scaling: value });
                      narrate(
                        value === "raw"
                          ? `Scaling switched off: softmax now sees raw scores, ${scaleFactor} times larger.`
                          : "Scaling restored: scores are divided by the square root of the head width.",
                      );
                    }}
                  />
                </div>
                <div className="attention-formula-stack">
                  <div className="atn-part atn-part--score">
                    <FormulaWithValues
                      label="Score"
                      expression="q · k"
                      result={score.toFixed(4)}
                      detail={`${headWidth}-dimensional dot product of the two vectors above`}
                    />
                  </div>
                  <div className="atn-part atn-part--score">
                    <FormulaWithValues
                      label="Scale"
                      expression={useScale ? `score ÷ √${headWidth}` : "score, not divided"}
                      result={keyMasked ? "masked" : scaled.toFixed(4)}
                      detail={
                        keyMasked
                          ? "This key comes after the query, so the causal mask sets its score to −∞"
                          : useScale
                            ? "Keeps wide dot products from saturating softmax"
                            : `Scaling is off: softmax sees the raw score, ${scaleFactor} times larger than the scaled one`
                      }
                    />
                  </div>
                  <div className="atn-part atn-part--softmax">
                    <FormulaWithValues
                      label="Softmax"
                      expression="exp(s) ÷ Σ exp(s)"
                      result={percent(characterWeight, 2)}
                      detail={
                        !modelReady
                          ? `Over ${visibleCount} visible characters, from stand-in vectors`
                          : editVectors
                            ? `Over ${visibleCount} visible characters, from your edited vectors. The model's own weight was ${percent(modelCharacterWeight, 2)}`
                            : causal && useScale
                              ? `Over ${visibleCount} visible characters. Model reports ${percent(modelCharacterWeight, 2)} (recomputed row matches within ${recomputeGap.toExponential(1)})`
                              : `Over ${visibleCount} visible characters${causal ? "" : " with the mask lifted"}${useScale ? "" : " and no scaling"}, recomputed from the model's own Q and K. The model's own causal, scaled weight was ${percent(modelCharacterWeight, 2)}`
                      }
                    />
                  </div>
                  <div className="atn-part atn-part--softmax">
                    <FormulaWithValues
                      label="Word total"
                      expression={`Σ over ${keySpan?.text ?? "key"} (${keyCharacters} chars)`}
                      result={percent(wordWeight, 2)}
                      detail="The heatmap cell: this query's weights on every character of the word, plus the space before it"
                    />
                  </div>
                </div>

                <div className="atn-view-control atn-value-control">
                  <SegmentedControl
                    label="Value vectors (V)"
                    value={valueMode}
                    options={[
                      { value: "model", label: "Model" },
                      { value: "inspected", label: "Zero word" },
                      { value: "all", label: "Zero all" },
                    ]}
                    onChange={(value) => {
                      setState({ values: value });
                      narrate(
                        value === "model"
                          ? "Value vectors restored to the model's own."
                          : value === "all"
                            ? "Every value vector is zero. The weights are unchanged and the mixed value is zero."
                            : `Values of ${keySpan?.text ?? "the inspected word"} set to zero. The weights are unchanged.`,
                      );
                    }}
                  />
                </div>
                <div className="atn-part atn-part--mix">
                  <VectorChip
                    label={`V · ${keyLabel}`}
                    values={(values[kPos] ?? []).slice(0, EDIT_DIMENSIONS)}
                    tone="forward"
                  />
                </div>
                <div className="atn-part atn-part--mix">
                  <VectorChip
                    label={`Mixed value · first ${EDIT_DIMENSIONS} of ${mixed.length} dims`}
                    values={mixed.slice(0, EDIT_DIMENSIONS)}
                    tone="forward"
                  />
                </div>
                <p className="vector-edit-note">
                  {valueNote}
                  {valueMode !== "model" && ` With the model's own values the mix has length ${vectorLength(modelMixed).toFixed(3)}.`}
                </p>
              </div>
            </div>
            <div className="metric-row">
              <Metric
                label="Selected score"
                value={keyMasked ? "masked" : scaled.toFixed(3)}
                tone="gradient"
              />
              <Metric
                label="Selected weight"
                value={percent(wordWeight)}
              />
              <Metric
                label="On spaces"
                value={percent(spaces)}
                tone="forward"
              />
            </div>
            <div className="metric-row">
              <Metric
                label="Largest weight"
                value={percent(largestWeight)}
              />
              <Metric
                label="Saturated rows"
                value={`${saturation.saturated} of ${saturation.total}`}
              />
              <Metric
                label="Mix length"
                value={mixLength.toFixed(3)}
                tone="forward"
              />
            </div>
          </div>
        </LabSurface>
      </div>
    </div>
  );
}
