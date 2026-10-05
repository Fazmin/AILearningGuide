import { useEffect, useMemo, useState } from "react";
import {
  BarList,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import cache from "./assets/sae-top-activations.json";
import { ActivationText, ResidualBars, showChar, SparseCode } from "./FeatureViews";
import { pct } from "./format";
import { loadDecoder, runSae, type SaeRun, type Status } from "./runtime";
import {
  ACTIVE,
  activeFeatures,
  decoderOverlap,
  dominantFocus,
  EARLY_POSITIONS,
  examplesFor,
  featureFlags,
  FEATURES,
  HYPOTHESES,
  isStartArtefact,
  L1_COEFFICIENT,
  matchesHypothesis,
  pairActivity,
  seenBefore,
  testLabel,
  topKCurve,
  unexplainedFraction,
  WIDTH,
  type CachedExample,
  type CachedRecord,
  type Hypothesis,
  type OverlapStats,
  type SaeWeights,
} from "./sae";
import { SteerView } from "./SteerView";
import { LABELLED_FEATURES, labelledFeature, MAX_STRENGTH, NULL_DIRECTIONS } from "./steering";

const MAX_CHARACTERS = 64;
const DEFAULT_TEXT = "hath the king a good horse? KING HENRY: aye, he hath.";
const DEFAULT_FEATURE = 876;
const DEFAULT_STEER_FEATURE = LABELLED_FEATURES[0].feature;
const records = cache.top_activations as unknown as CachedRecord[];
const recordByFeature = new Map(records.map((record) => [record.feature, record]));
const training = cache.training;
const lastTraining = training[training.length - 1];
const scanned = cache.position_summary.scanned_tokens;

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const spoken = (character: string) => (character === " " ? "space" : character === "\n" ? "line break" : character);

/** How a feature appears in a menu or a ranked list: its number, plus a mark when the cache flags it. */
const featureTag = (feature: number) => {
  const record = recordByFeature.get(feature);
  return `#${feature}${record && isStartArtefact(record) ? " · window start" : ""}`;
};

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const text = asString(state, "text", DEFAULT_TEXT).slice(0, MAX_CHARACTERS);
  const characters = Array.from(text);
  const lastIndex = Math.max(0, characters.length - 1);
  const position = clamp(Math.round(asNumber(state, "token", 4)), 0, lastIndex);
  const feature = clamp(Math.round(asNumber(state, "feature", DEFAULT_FEATURE)), 0, FEATURES - 1);
  const threshold = clamp(asNumber(state, "threshold", 5), 0, 10);
  const hypothesisRaw = asString(state, "hypothesis", "start");
  const hypothesis: Hypothesis = HYPOTHESES.some((entry) => entry.value === hypothesisRaw)
    ? (hypothesisRaw as Hypothesis)
    : "start";
  const exampleView = asString(state, "examples", "top") === "context" ? "context" : "top";
  const keep = clamp(Math.round(asNumber(state, "keep", 8)), 1, ACTIVE);
  const steerFeature = labelledFeature(asNumber(state, "steerFeature", DEFAULT_STEER_FEATURE))?.feature ?? DEFAULT_STEER_FEATURE;
  const steerAt = clamp(Math.round(asNumber(state, "steerAt", 43)), 0, lastIndex);
  const strength = clamp(Math.round(asNumber(state, "strength", 5)), 0, MAX_STRENGTH);
  const seed = clamp(Math.round(asNumber(state, "seed", 1)), 1, NULL_DIRECTIONS);

  // ---- Real models: transformer residual (ONNX), then the SAE (ONNX), in the worker.
  const [run, setRun] = useState<{ status: Status; value?: SaeRun; error?: string }>({ status: "loading" });
  useEffect(() => {
    if (!text) return;
    let active = true;
    setRun((previous) => ({ ...previous, status: "loading" }));
    const timer = window.setTimeout(() => {
      runSae(text)
        .then((value) => active && setRun({ status: "ready", value }))
        .catch((error: unknown) =>
          active && setRun((previous) => ({ ...previous, status: "error", error: error instanceof Error ? error.message : String(error) })),
        );
    }, 160);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [text]);

  // ---- Decoder directions, read from the same ONNX file, for top-k and overlap.
  const [decoder, setDecoder] = useState<{ status: Status; sae?: SaeWeights; overlap?: OverlapStats }>({ status: "loading" });
  useEffect(() => {
    let active = true;
    loadDecoder()
      .then((sae) => {
        if (!active) return;
        setDecoder({ status: "ready", sae });
        window.setTimeout(() => active && setDecoder({ status: "ready", sae, overlap: decoderOverlap(sae) }), 50);
      })
      .catch(() => active && setDecoder({ status: "error" }));
    return () => {
      active = false;
    };
  }, []);

  const current = run.value;
  const aligned = current !== undefined && current.text === text;
  const features = aligned ? current.features[position] ?? null : null;
  const x = aligned ? current.x[position] ?? null : null;
  const reconstruction = aligned ? current.reconstruction[position] ?? null : null;
  const active = useMemo(() => (features ? activeFeatures(features) : []), [features]);
  const unexplained = x && reconstruction ? unexplainedFraction(x, reconstruction) : null;
  const similarity = useMemo(() => {
    if (!x || !reconstruction) return null;
    let dot = 0;
    let a = 0;
    let b = 0;
    for (let index = 0; index < WIDTH; index += 1) {
      dot += x[index] * reconstruction[index];
      a += x[index] ** 2;
      b += reconstruction[index] ** 2;
    }
    return dot / Math.sqrt(a * b);
  }, [x, reconstruction]);
  const curve = useMemo(
    () => (decoder.sae && x && active.length ? topKCurve(decoder.sae, x, active) : null),
    [decoder.sae, x, active],
  );
  const localOverlap = useMemo(
    () => (decoder.sae && active.length > 1 ? decoderOverlap(decoder.sae, active.map((entry) => entry.feature)) : null),
    [decoder.sae, active],
  );
  const strongest = decoder.overlap?.strongest ?? null;
  const pair = useMemo(
    () => (aligned && strongest ? pairActivity(current.features, strongest.a, strongest.b) : null),
    [aligned, current, strongest],
  );

  // ---- The selected feature across the probe text, and its cached evidence.
  const featureActivations = aligned ? current.features.map((row) => row[feature] ?? 0) : [];
  const featureMaximum = Math.max(1e-6, ...featureActivations);
  const record = recordByFeature.get(feature);
  const hasContext = Boolean(record?.examples_in_context);
  const shown: CachedExample[] = record ? examplesFor(record, hasContext ? exampleView : "top") : [];
  const shownMaximum = Math.max(1e-6, ...shown.map((example) => example.activation));
  const flags = record ? featureFlags(record) : [];
  const focus = shown.length ? dominantFocus(shown) : null;
  const focusCharacter = focus?.character ?? characters[position] ?? "";
  const labelTest = aligned ? testLabel(text, featureActivations, threshold, hypothesis, focusCharacter) : null;
  const cachedMatches =
    shown.length && hypothesis !== "newline"
      ? shown.filter((example) =>
          matchesHypothesis(hypothesis, example.text[example.focus_offset] ?? "", example.window_position, focusCharacter),
        ).length
      : null;
  const hypothesisText =
    hypothesis === "char"
      ? `fires on “${showChar(focusCharacter)}”`
      : hypothesis === "upper"
        ? "fires on capital letters"
        : hypothesis === "space"
          ? "fires on spaces"
          : hypothesis === "newline"
            ? "fires on line breaks"
            : `fires on the first ${EARLY_POSITIONS} characters of a window`;

  const statusChip = (
    <span className={`model-runtime-status model-runtime-status--${run.status}`} role="status" title={run.error || undefined}>
      <i />
      {run.status === "ready"
        ? `ONNX · ${(current?.provider ?? "wasm").toUpperCase()}`
        : run.status === "error"
          ? "Model unavailable · cached examples only"
          : "Running transformer + SAE…"}
    </span>
  );

  const featureOptions = records.some((entry) => entry.feature === feature)
    ? records
    : [{ feature, maximum: 0, examples: [] as CachedExample[] } as unknown as CachedRecord, ...records];

  return (
    <div className="sf-lab">
      <LabSurface label="Residual to sparse code" className="sf-surface">
        <SurfaceHeading
          kicker="Transformer residual after block 2 → 1,024-feature sparse autoencoder"
          title="One dense vector, rewritten as 32 active features"
          aside={statusChip}
        />
        <label className="sf-input">
          <span>Probe text</span>
          <textarea
            value={text}
            rows={2}
            maxLength={MAX_CHARACTERS}
            spellCheck={false}
            onChange={(event) => setState({ text: event.target.value, token: 0 })}
          />
          <small>{characters.length}/{MAX_CHARACTERS}</small>
        </label>
        <div className="sf-pick" role="group" aria-label="Choose a character to inspect">
          {characters.map((character, index) => (
            <button
              type="button"
              key={index}
              aria-pressed={index === position}
              aria-label={`position ${index}: ${spoken(character)}`}
              onClick={() => setState({ token: index })}
            >
              {showChar(character)}
            </button>
          ))}
        </div>
        <p className="sf-small">
          The text is read as one window, so position {position} has seen {position + 1} character{position === 0 ? "" : "s"}. Positions 0 to{" "}
          {EARLY_POSITIONS - 1} are the start of a window, where the residual is unlike running text.
        </p>

        <div className="sf-panels">
          <div className="sf-panel">
            <div className="sf-panel__head">
              <strong>Residual x at “{showChar(characters[position] ?? "")}” (position {position})</strong>
              <span>256 numbers, nearly all nonzero · bars = x, dashes = SAE reconstruction x̂</span>
            </div>
            <ResidualBars
              x={x}
              reconstruction={reconstruction}
              label={`Residual vector at position ${position}: 256 values, with the reconstruction overlaid. Unexplained share ${pct(unexplained, 1)}.`}
            />
          </div>
          <div className="sf-panel sf-code">
            <div className="sf-panel__head">
              <strong>SAE code f: {active.length} of 1,024 features active</strong>
              <span>each spike is one active feature; its height is the activation</span>
            </div>
            <SparseCode
              active={active}
              total={FEATURES}
              selected={feature}
              label={`Sparse code at position ${position}: ${active.length} active features out of 1,024; largest ${active
                .slice(0, 3)
                .map((entry) => `feature ${entry.feature} at ${entry.value.toFixed(2)}`)
                .join(", ")}.`}
            />
            <div className="sf-code__ticks" aria-hidden="true">
              <span>0</span>
              <span>256</span>
              <span>512</span>
              <span>768</span>
              <span>1,024</span>
            </div>
          </div>
        </div>

        <div className="metric-row">
          <Metric label="Active features (L0)" value={features ? `${active.length} / ${FEATURES.toLocaleString("en-US")}` : "—"} tone="forward" />
          <Metric label="‖x − x̂‖² / ‖x‖²" value={pct(unexplained, 1)} tone="loss" />
          <Metric label="cos(x, x̂)" value={similarity === null ? "—" : similarity.toFixed(3)} />
        </div>

        <div className="sf-top">
          <span>Largest active features here; choose one to inspect it below. “window start” marks a feature whose cached top examples come from the first four positions of a window.</span>
          {active.length ? (
            <BarList
              label="Largest active features at this character"
              items={active.slice(0, 8).map((entry) => ({
                id: String(entry.feature),
                label: `${featureTag(entry.feature)}${recordByFeature.has(entry.feature) ? "" : " · no cached examples"}`,
                value: entry.value,
                display: entry.value.toFixed(2),
                tone: entry.feature === feature ? "attention" : "forward",
              }))}
              selectedId={String(feature)}
              onSelect={(id) => {
                setState({ feature: Number(id) });
                narrate(`Feature ${id} selected.`);
              }}
            />
          ) : (
            <p className="sf-empty" role="status">
              {!text
                ? "Type some probe text to run the models."
                : run.status === "error"
                  ? "The models did not load, so no features can be computed."
                  : "Waiting for the models…"}
            </p>
          )}
        </div>
      </LabSurface>

      <LabSurface label="Feature evidence" className="sf-surface">
        <SurfaceHeading
          kicker={`Feature #${feature} · cached top examples from training text`}
          title="Propose a label, then try to break it"
        />
        <label className="sf-select">
          <span>Feature</span>
          <select value={feature} onChange={(event) => setState({ feature: Number(event.target.value) })}>
            {featureOptions.map((entry) => {
              const summary = entry.examples.length ? dominantFocus(entry.examples) : null;
              return (
                <option key={entry.feature} value={entry.feature}>
                  {`${featureTag(entry.feature)}${summary ? ` · top examples: “${showChar(summary.character)}” × ${summary.count}` : " · no cached examples"}`}
                </option>
              );
            })}
          </select>
        </label>

        {record ? (
          <>
            {flags.length > 0 && (
              <ul className="sf-flags" aria-label={`Flags on feature ${feature}`}>
                {flags.map((flag) => (
                  <li key={flag}>{flag}</li>
                ))}
              </ul>
            )}
            <p className="sf-small">
              Active on {record.fires.toLocaleString("en-US")} of {scanned.toLocaleString("en-US")} scanned characters ({pct(record.fires / scanned, 1)}),
              {" "}{record.fires_at_position_zero.toLocaleString("en-US")} of them at window position 0. Largest activation {record.maximum.toFixed(1)}
              {record.maximum_after_position_zero < record.maximum ? `; from position 1 on, ${record.maximum_after_position_zero.toFixed(1)}` : ""}.
            </p>
            {hasContext ? (
              <SegmentedControl
                label="Examples shown"
                value={exampleView}
                options={[
                  { value: "top", label: "Top 8 overall" },
                  { value: "context", label: `Top 8 from position ${EARLY_POSITIONS} on` },
                ]}
                onChange={(value) => setState({ examples: value })}
              />
            ) : (
              <p className="sf-small">None of this feature&apos;s top examples are in the first {EARLY_POSITIONS} positions, so there is no separate in-context list.</p>
            )}
            <ol className="sf-examples" aria-label={`Eight cached top-activating examples for feature ${feature}`}>
              {shown.map((example, index) => {
                const before = example.text.slice(Math.max(0, example.focus_offset - 22), example.focus_offset);
                const seen = Math.min(seenBefore(example), before.length);
                const at = example.text[example.focus_offset] ?? "";
                const after = example.text.slice(example.focus_offset + 1, example.focus_offset + 14);
                return (
                  <li key={index}>
                    <code>
                      <span className="sf-unseen">{before.slice(0, before.length - seen)}</span>
                      <span>{before.slice(before.length - seen)}</span>
                      <mark>{showChar(at)}</mark>
                      <span className="sf-unseen">{after}</span>
                    </code>
                    <em>{example.window_position === 0 ? "first character" : `position ${example.window_position}`}</em>
                    <i style={{ width: `${(example.activation / shownMaximum) * 100}%` }} />
                    <b>{example.activation.toFixed(2)}</b>
                  </li>
                );
              })}
            </ol>
          </>
        ) : (
          <p className="sf-empty">
            Feature #{feature} is not among the 128 features whose top examples were cached, so there is no training-text
            evidence to read; only its activations on your text below.
          </p>
        )}
        {focus && (
          <p className="sf-small">
            Focus characters of these examples: “{showChar(focus.character)}” × {focus.count}
            {focus.distinct > 1 ? `, other characters × ${shown.length - focus.count}` : ""}. Dimmed text is outside what the model
            had read when it computed the activation: it came before its 64-character window or after the marked character.
          </p>
        )}

        <div className="sf-test-controls">
          <SegmentedControl
            label="Label to test"
            value={hypothesis}
            options={HYPOTHESES.map((entry) => ({ value: entry.value, label: entry.label }))}
            onChange={(value) => setState({ hypothesis: value })}
          />
          <RangeControl
            label="Activation threshold"
            min={0}
            max={10}
            step={0.1}
            value={threshold}
            format={(value) => value.toFixed(1)}
            onChange={(value) => setState({ threshold: value })}
          />
        </div>
        <p className="sf-hypothesis">
          Hypothesis: feature #{feature} {hypothesisText}. It “fires” where its activation is above {threshold.toFixed(1)}.
        </p>
        {aligned ? (
          <ActivationText
            text={text}
            activations={featureActivations}
            threshold={threshold}
            matches={(character, index) => matchesHypothesis(hypothesis, character, index, focusCharacter)}
            maximum={featureMaximum}
            earlyPositions={EARLY_POSITIONS}
          />
        ) : (
          <p className="sf-empty" role="status">
            {!text ? "Type some probe text above." : run.status === "error" ? "Activations need the models." : "Computing activations…"}
          </p>
        )}
        <ul className="sf-key">
          <li><i className="sf-key__fire" /> fires (above the threshold)</li>
          <li><i className="sf-key__match" /> matches the label</li>
          <li><i className="sf-key__heat" /> activation, darker = higher</li>
          <li><i className="sf-key__early" /> dotted top edge: window position 0 to {EARLY_POSITIONS - 1}</li>
        </ul>
        <div className="metric-row">
          <Metric label="Precision" value={labelTest ? pct(labelTest.precision) : "—"} tone="forward" />
          <Metric label="Recall" value={labelTest ? pct(labelTest.recall) : "—"} tone="gradient" />
          <Metric
            label="Fires on"
            value={labelTest ? `${labelTest.truePositive + labelTest.falsePositive} of ${characters.length}` : "—"}
          />
        </div>
        <p className="sf-small">
          {labelTest
            ? `${labelTest.truePositive} hit${labelTest.truePositive === 1 ? "" : "s"}, ${labelTest.falseNegative} miss${labelTest.falseNegative === 1 ? "" : "es"}, ${labelTest.falsePositive} false alarm${labelTest.falsePositive === 1 ? "" : "s"}. `
            : ""}
          Precision: of the characters where it fires, the share that match the label. Recall: of the characters that
          match, the share where it fires.{" "}
          {shown.length
            ? hypothesis === "newline"
              ? "The cache writes every line break as a space, so it cannot check a line-break label against its examples."
              : `On these cached examples the label matches ${cachedMatches} of ${shown.length}${hypothesis === "space" ? "; an excerpt shows a line break as a space, so this count can include line breaks" : ""}.`
            : ""}
        </p>
      </LabSurface>

      <LabSurface label="Sparsity and superposition" className="sf-surface">
        <SurfaceHeading
          kicker="TopK sparsity · 1,024 directions in 256 dimensions"
          title="How many features does a good reconstruction need?"
        />
        <div className="sf-topk">
          <RangeControl
            label="Keep top k features"
            min={1}
            max={ACTIVE}
            step={1}
            value={keep}
            format={(value) => `${value} of ${ACTIVE}`}
            onChange={(value) => setState({ keep: value })}
          />
        </div>
        <div className="sf-charts">
          {curve ? (
            <LineChart
              label="Unexplained share against the number of features kept"
              series={[
                {
                  id: "curve",
                  name: `‖x − x̂_k‖² / ‖x‖² at “${showChar(characters[position] ?? "")}”`,
                  tone: "loss",
                  points: curve.map((value, index) => ({ x: index + 1, y: value })),
                  format: (value) => pct(value, 1),
                },
              ]}
              xLabel="features kept, largest first"
              yLabel="unexplained share"
              xDomain={[1, ACTIVE]}
              yDomain={[0, 1]}
              marker={{ x: keep, label: `k = ${keep}` }}
              footnote={`At position ${position}: keeping the ${keep} largest leaves ${pct(curve[keep - 1] ?? null, 1)} unexplained; the single largest leaves ${pct(curve[0] ?? null, 1)}.`}
            />
          ) : (
            <p className="sf-empty" role="status">
              {decoder.status === "error" ? "The decoder weights did not load." : "Waiting for the decoder weights and a residual…"}
            </p>
          )}
          <LineChart
            label="SAE training reconstruction error"
            series={[
              {
                id: "reconstruction",
                name: "Training reconstruction MSE",
                tone: "forward",
                points: training.map((entry) => ({ x: entry.step, y: entry.reconstruction })),
                format: (value) => value.toFixed(4),
              },
            ]}
            xLabel="training step"
            yLabel="mean squared error (normalized units)"
            xDomain={[0, lastTraining.step]}
            footnote={`${lastTraining.step.toLocaleString("en-US")} steps, logged in sae-top-activations.json. ${pct(lastTraining.active_fraction, 3)} of features were active at every logged step, because TopK keeps 32 of 1,024; the L1 term (× ${L1_COEFFICIENT}) added only ${(lastTraining.loss - lastTraining.reconstruction).toExponential(1)} to the final loss.`}
          />
        </div>
        <div className="metric-row">
          <Metric label={`Keep ${keep}`} value={curve ? pct(curve[keep - 1] ?? null, 1) : "—"} tone="loss" />
          <Metric label="Keep 1" value={curve ? pct(curve[0] ?? null, 1) : "—"} />
          <Metric label={`Keep all ${ACTIVE}`} value={curve ? pct(curve[curve.length - 1] ?? null, 1) : "—"} tone="forward" />
        </div>

        <div className="sf-overlap">
          <div>
            <strong>Overlap between decoder directions</strong>
            <p className="sf-small">
              Each feature writes along one unit-length direction. 1,024 of them cannot be orthogonal in 256 dimensions, so
              they overlap a little: that is superposition, measured. The largest overlap is a surprise: it is not between two
              features that mean nearly the same thing.
            </p>
          </div>
          <div className="metric-row">
            <Metric
              label="Active: mean |cos|"
              value={localOverlap ? localOverlap.meanAbs.toFixed(3) : "—"}
              tone="gradient"
            />
            <Metric label="Active: max |cos|" value={localOverlap ? localOverlap.maxAbs.toFixed(2) : "—"} />
            <Metric
              label="All: mean · max"
              value={decoder.overlap ? `${decoder.overlap.meanAbs.toFixed(3)} · ${decoder.overlap.maxAbs.toFixed(2)}` : "—"}
            />
          </div>
          <div className="metric-row">
            <Metric
              label="Most similar pair"
              value={strongest ? `#${strongest.a} · #${strongest.b}` : "—"}
              tone="loss"
            />
            <Metric label="Their cosine (signed)" value={strongest ? strongest.cosine.toFixed(3) : "—"} />
            <Metric label="Pairs with |cos| above 0.9" value={decoder.overlap ? String(decoder.overlap.nearIdentical) : "—"} />
          </div>
          <p className="sf-small">
            {strongest
              ? `The sign is negative: #${strongest.a} and #${strongest.b} write in nearly opposite directions. ${
                  pair
                    ? `On your probe text they are active together on ${pair.both} of ${pair.total} characters, and at least one of them on ${pair.either}.`
                    : ""
                } A ReLU feature cannot go negative, so one signed direction in the residual needs two features, one for each side.`
              : "Waiting for the decoder weights…"}
          </p>
        </div>
      </LabSurface>

      <LabSurface label="Steer with a feature" className="sf-surface">
        <SurfaceHeading
          kicker="Add a decoder direction to the layer-2 residual · read the next-character distribution"
          title="Does turning a feature up do what its label says?"
          aside={statusChip}
        />
        <SteerView
          text={text}
          position={steerAt}
          feature={steerFeature}
          strength={strength}
          seed={seed}
          run={current}
          aligned={aligned}
          runStatus={run.status}
          sae={decoder.sae}
          setState={setState}
          narrate={narrate}
        />
      </LabSurface>
    </div>
  );
}
