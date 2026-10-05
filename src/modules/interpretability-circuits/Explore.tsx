import { useEffect, useMemo, useState } from "react";
import {
  LabSurface,
  Metric,
  runTeachingModel,
  SegmentedControl,
  SurfaceHeading,
  teachingAssetUrl,
  type ModuleContext,
} from "@app/module-sdk";
import cache from "./assets/interpretability-cache.json";
import { AttentionGrid, PatchGrid, PositionBars, TokenStrip } from "./CircuitViews";
import { runTransformer, weightsFromOnnx, type TransformerWeights } from "./engine";
import {
  cachedPatchSweep,
  controlPosition,
  copyPatchSweep,
  copyPositions,
  copyPrompt,
  copyScore,
  CORRUPT_LETTER,
  DEFAULT_PERIOD,
  headId,
  headScores,
  HEADS,
  HELD_OUT_PERIODS,
  inductionStripe,
  LAYERS,
  PERIODS,
  referenceMeans,
  runWithAblation,
  singleHeadEffects,
  STABLE_MARGIN,
  type AblationMode,
  type CachedPatchRecord,
  type CopyPrompt,
  type Corruption,
  type PatchSweep,
} from "./experiments";
import { REFERENCE_STRINGS, type HeadMeans } from "./mean-ablation";
import { showToken } from "./vocabulary";

const modelUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");
const ALL_HEADS = Array.from({ length: LAYERS * HEADS }, (_, index) => headId(Math.floor(index / HEADS), index % HEADS));
const cachedSweeps = (cache.activation_patching as CachedPatchRecord[]).map(cachedPatchSweep);
/** The canned attention example used when the live weights cannot load: a random-letter repeat of period 13. */
const cachedInduction = cache.attention.find((record) => record.id === "induction-random") ?? cache.attention[0];
const CACHED_PERIOD = 13;

type LoadState =
  | { status: "loading" }
  | { status: "ready"; weights: TransformerWeights }
  | { status: "error"; error: string };

let weightsPromise: Promise<TransformerWeights> | null = null;
function loadWeights() {
  weightsPromise ??= fetch(modelUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} for tiny-transformer.onnx`);
      return response.arrayBuffer();
    })
    .then((buffer) => weightsFromOnnx(buffer))
    .catch((error: unknown) => {
      weightsPromise = null;
      throw error;
    });
  return weightsPromise;
}

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number, digits = 2) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
const isHeldOut = (period: number) => HELD_OUT_PERIODS.includes(period);

/** The cell worth clicking first in a live sweep: the corrupted letter's own position. */
const livePatchStart = (prompt: CopyPrompt, corruption: Corruption) =>
  corruption === "source" ? prompt.source : controlPosition(prompt);

/** Where a head's attention goes, in words, for the one offset that takes the most of it. */
function offsetMeaning(offset: number, period: number) {
  if (offset === period - 1) return "the key after the previous occurrence";
  if (offset === 1) return "the previous letter";
  if (offset === 0) return "the letter itself";
  return "";
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const periodRaw = Math.round(asNumber(state, "period", DEFAULT_PERIOD));
  const period = (PERIODS as readonly number[]).includes(periodRaw) ? periodRaw : DEFAULT_PERIOD;
  const attnLayer = clamp(Math.round(asNumber(state, "attnLayer", 1)), 0, LAYERS - 1);
  const attnHead = clamp(Math.round(asNumber(state, "attnHead", 2)), 0, HEADS - 1);
  const ablated = Array.isArray(state.ablated) ? state.ablated.filter((id) => ALL_HEADS.includes(id)) : [];
  const ablationMode: AblationMode = asString(state, "ablation", "zero") === "mean" ? "mean" : "zero";
  const sweepId = asString(state, "sweep", "live");
  const corruption: Corruption = asString(state, "corruption", "source") === "control" ? "control" : "source";
  const patchLayer = clamp(Math.round(asNumber(state, "layer", 0)), 0, LAYERS - 1);

  const [load, setLoad] = useState<LoadState>({ status: "loading" });
  useEffect(() => {
    let active = true;
    loadWeights()
      .then((weights) => active && setLoad({ status: "ready", weights }))
      .catch((error: unknown) => active && setLoad({ status: "error", error: error instanceof Error ? error.message : String(error) }));
    return () => {
      active = false;
    };
  }, []);
  const weights = load.status === "ready" ? load.weights : null;

  const prompt = useMemo(() => copyPrompt(period), [period]);
  const tokens = Array.from(prompt.text);
  const run = useMemo(() => (weights ? runTransformer(weights, prompt.ids) : null), [weights, prompt]);
  const score = useMemo(() => (run ? copyScore(run, prompt) : null), [run, prompt]);
  const heads = useMemo(() => (run ? headScores(run, prompt) : null), [run, prompt]);

  // The slower work runs after paint so a click responds at once: the single-head table (and, for
  // mean ablation, the reference averages) and the patching sweep.
  const ablationKey = `${period}:${ablationMode}`;
  const [ablationWork, setAblationWork] = useState<{
    key: string;
    single: { id: string; mean: number }[];
    means: HeadMeans | null;
  } | null>(null);
  useEffect(() => {
    if (!weights) return;
    const timer = window.setTimeout(() => {
      const means = ablationMode === "mean" ? referenceMeans(weights, prompt) : null;
      setAblationWork({
        key: ablationKey,
        means,
        single: singleHeadEffects(weights, prompt, ablationMode, means ?? undefined),
      });
    }, 30);
    return () => window.clearTimeout(timer);
  }, [weights, prompt, ablationMode, ablationKey]);
  const freshAblation = ablationWork && ablationWork.key === ablationKey ? ablationWork : null;

  const ablatedKey = ablated.join(",");
  const ablatedRun = useMemo(() => {
    if (!weights || !run) return null;
    if (ablated.length === 0) return run;
    if (ablationMode === "mean") {
      return freshAblation?.means ? runWithAblation(weights, prompt, ablated, "mean", freshAblation.means) : null;
    }
    return runWithAblation(weights, prompt, ablated, "zero");
    // ablatedKey stands in for the array identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weights, prompt, ablatedKey, ablationMode, run, freshAblation]);
  const ablatedScore = useMemo(() => (ablatedRun ? copyScore(ablatedRun, prompt) : null), [ablatedRun, prompt]);
  const stripeIntact = useMemo(() => (run ? inductionStripe(run, prompt) : null), [run, prompt]);
  const stripeAblated = useMemo(() => (ablatedRun ? inductionStripe(ablatedRun, prompt) : null), [ablatedRun, prompt]);

  const sweepKey = `${period}:${corruption}`;
  const [sweepWork, setSweepWork] = useState<{ key: string; sweep: PatchSweep } | null>(null);
  useEffect(() => {
    if (!weights || sweepId !== "live") return;
    const timer = window.setTimeout(() => {
      setSweepWork({ key: sweepKey, sweep: copyPatchSweep(weights, prompt, corruption) });
    }, 30);
    return () => window.clearTimeout(timer);
  }, [weights, prompt, corruption, sweepKey, sweepId]);
  const liveSweep = sweepWork && sweepWork.key === sweepKey ? sweepWork.sweep : null;

  // Cross-check the TypeScript forward pass against ONNX Runtime on this prompt.
  const [parity, setParity] = useState<{ period: number; delta: number; provider: string } | null>(null);
  useEffect(() => {
    if (!run) return;
    let active = true;
    runTeachingModel(modelUrl, { input_ids: { type: "int64", dims: [1, prompt.ids.length], data: prompt.ids } })
      .then((result) => {
        const logits = result.outputs.logits?.data ?? [];
        let delta = 0;
        for (let index = 0; index < run.logits.length; index += 1) {
          delta = Math.max(delta, Math.abs((logits[index] ?? Infinity) - run.logits[index]));
        }
        if (active) setParity({ period, delta, provider: result.provider });
      })
      .catch(() => active && setParity(null));
    return () => {
      active = false;
    };
  }, [run, prompt, period]);

  const sweep: PatchSweep | null =
    sweepId === "live" ? liveSweep : cachedSweeps.find((entry) => entry.id === sweepId) ?? null;
  const sweepTokens = sweep ? Array.from(sweep.clean) : [];
  const patchPosition = clamp(
    Math.round(asNumber(state, "patch", sweep ? sweepTokens.length - 1 : 0)),
    0,
    Math.max(0, sweepTokens.length - 1),
  );
  const gap = sweep ? sweep.cleanScore - sweep.corruptScore : 0;
  const selectedRecovery = sweep?.recovery[patchLayer]?.[patchPosition] ?? 0;

  const statusChip =
    load.status === "ready" ? (
      <span className="model-runtime-status model-runtime-status--ready" role="status">
        <i />
        {parity && parity.period === period ? `Matches ONNX · Δ ${parity.delta.toExponential(0)}` : "Weights loaded"}
      </span>
    ) : load.status === "error" ? (
      <span className="model-runtime-status model-runtime-status--error" role="status" title={load.error}>
        <i />
        Weights unavailable · cached view
      </span>
    ) : (
      <span className="model-runtime-status model-runtime-status--loading" role="status">
        <i />
        Loading weights…
      </span>
    );

  const selectedHead = heads?.find((entry) => entry.layer === attnLayer && entry.head === attnHead);
  const cachedTokens = cachedInduction.tokens;
  const attentionWeights = run
    ? run.attention[attnLayer].subarray(attnHead * tokens.length * tokens.length, (attnHead + 1) * tokens.length * tokens.length)
    : load.status === "error"
      ? Float32Array.from((cachedInduction.attention as number[][][][])[attnLayer][attnHead].flat())
      : null;
  const gridTokens = run || load.status !== "error" ? tokens : cachedTokens;
  const gridPeriod = run ? period : load.status === "error" ? CACHED_PERIOD : null;
  const topPrediction = score?.top[0];

  const marks: Record<number, string> = { [prompt.source]: "source" };

  const headCaption = selectedHead
    ? (() => {
        const meaning = offsetMeaning(selectedHead.strongestOffset, period);
        const where =
          selectedHead.strongestOffset === 0 ? "on the letter itself" : `exactly ${selectedHead.strongestOffset} back`;
        return `L${attnLayer + 1} H${attnHead + 1}: its largest single share of attention, ${pct(selectedHead.strongestShare, 0)}, goes ${where}${meaning ? ` (${meaning})` : ""}.`;
      })()
    : null;

  const modeKicker =
    ablationMode === "mean"
      ? `Mean ablation · head output replaced by its average on ${REFERENCE_STRINGS} random repeats, before the output projection`
      : "Zero-ablation · head output set to 0 before the output projection";

  return (
    <div className="ci-lab">
      <LabSurface label="Copy task and attention" className="ci-surface">
        <SurfaceHeading
          kicker="Shipped 2-layer transformer · forward pass in your browser"
          title="Does the model copy a repeating sequence?"
          aside={statusChip}
        />
        <div className="ci-period">
          <SegmentedControl
            label="Repeat period"
            value={String(period)}
            options={PERIODS.map((value) => ({
              value: String(value),
              label: isHeldOut(value) ? `${value} · unseen` : String(value),
            }))}
            onChange={(value) => {
              const nextPeriod = Number(value);
              setState({
                period: nextPeriod,
                patch: sweepId === "live" ? livePatchStart(copyPrompt(nextPeriod), corruption) : patchPosition,
              });
              narrate(`Repeat period ${value}${isHeldOut(nextPeriod) ? ", a period the model never saw in training" : ""}.`);
            }}
          />
          <p className="ci-small">
            {isHeldOut(period)
              ? `Period ${period} is unseen: the model's synthetic training repeats never used it.`
              : `Period ${period} was one of the periods the model's synthetic training repeats used. Periods marked unseen were not.`}
          </p>
        </div>
        <div className="ci-prompt">
          <TokenStrip
            tokens={tokens}
            period={period}
            marks={marks}
            label={`Prompt: ${prompt.text}, repeating every ${period} letters`}
          />
          <div className="ci-next" aria-label="Next-letter prediction">
            <span>next</span>
            <strong>{prompt.target}?</strong>
            <small>
              {topPrediction
                ? `model: ${showToken(topPrediction.token)} ${pct(topPrediction.probability)}`
                : "model: —"}
            </small>
          </div>
        </div>
        <ul className="ci-key">
          <li>
            <i className="ci-key__source" /> previous occurrence of the next letter (position {prompt.source})
          </li>
        </ul>
        <div className="metric-row">
          <Metric
            label="Copy accuracy"
            value={score ? pct(score.mean) : "—"}
            tone="forward"
          />
          <Metric label={`P(${prompt.target}) at the end`} value={score ? pct(score.finalTarget) : "—"} />
          <Metric
            label="Top 3 next letters"
            value={score ? score.top.map((entry) => `${showToken(entry.token)} ${pct(entry.probability, entry.probability < 0.1 ? 1 : 0)}`).join(" · ") : "—"}
          />
        </div>
        <p className="ci-small">
          Copy accuracy is the mean probability of the correct next letter over positions {prompt.period} to{" "}
          {tokens.length - 1}: the second repeat onward, where the current letter has already appeared once.
        </p>

        <div className="ci-attn-layout">
          <div className="ci-attention">
            <div className="ci-head-picker" role="group" aria-label="Choose the head to show">
              {ALL_HEADS.map((id, index) => {
                const layer = Math.floor(index / HEADS);
                const head = index % HEADS;
                return (
                  <button
                    type="button"
                    key={id}
                    aria-pressed={layer === attnLayer && head === attnHead}
                    onClick={() => setState({ attnLayer: layer, attnHead: head })}
                  >
                    L{layer + 1} H{head + 1}
                  </button>
                );
              })}
            </div>
            {attentionWeights ? (
              <AttentionGrid
                tokens={gridTokens}
                weights={attentionWeights}
                period={gridPeriod}
                label={
                  selectedHead
                    ? `Layer ${attnLayer + 1} head ${attnHead + 1} attention. Mean weight ${pct(selectedHead.induction)} on the token after the previous occurrence, ${pct(selectedHead.previous)} on the previous letter.`
                    : `Cached layer ${attnLayer + 1} head ${attnHead + 1} attention for ${cachedInduction.text}`
                }
              />
            ) : (
              <div className="ci-empty" role="status">Loading the model weights…</div>
            )}
            <ul className="ci-key">
              <li><i className="ci-key__cell" /> attention weight (darker = more)</li>
              <li><i className="ci-key__induction" /> key an induction head would read: i − {gridPeriod ?? period} + 1</li>
              <li><i className="ci-key__previous-cell" /> previous letter: i − 1</li>
            </ul>
            {headCaption && <p className="ci-small">{headCaption}</p>}
            {!run && load.status === "error" && (
              <p className="ci-small">Cached attention for “{cachedInduction.text}”, precomputed offline; the live model did not load.</p>
            )}
          </div>

          <div className="ci-head-table" role="group" aria-label="Mean attention per head over the second repeat; choose a head to show it">
            <div className="ci-head-table__header" aria-hidden="true">
              <span>Head</span>
              <span>previous letter</span>
              <span>after previous occurrence</span>
            </div>
            {(heads ?? []).map((entry) => (
              <button
                type="button"
                key={headId(entry.layer, entry.head)}
                aria-pressed={entry.layer === attnLayer && entry.head === attnHead}
                aria-label={`Layer ${entry.layer + 1} head ${entry.head + 1}: ${entry.previous.toFixed(2)} on the previous letter, ${entry.induction.toFixed(2)} on the token after the previous occurrence`}
                onClick={() => setState({ attnLayer: entry.layer, attnHead: entry.head })}
              >
                <span>L{entry.layer + 1} H{entry.head + 1}</span>
                <span className="ci-bar ci-bar--previous">
                  <i style={{ width: `${entry.previous * 100}%` }} />
                  <b>{entry.previous.toFixed(2)}</b>
                </span>
                <span className="ci-bar">
                  <i style={{ width: `${entry.induction * 100}%` }} />
                  <b>{entry.induction.toFixed(2)}</b>
                </span>
              </button>
            ))}
            {!heads && <p className="ci-small">Per-head scores appear once the weights load.</p>}
            <p className="ci-small">
              Mean attention from each query i ≥ {period} to the two keys. Read the columns against the key outlines in the grid.
            </p>
          </div>
        </div>
      </LabSurface>

      <LabSurface label="Head ablation" className="ci-surface">
        <SurfaceHeading kicker={modeKicker} title="Which heads is the copy behaviour relying on?" />
        {weights && score ? (
          <>
            <SegmentedControl
              label="Ablation"
              value={ablationMode}
              options={[
                { value: "zero", label: "Zero" },
                { value: "mean", label: "Mean" },
              ]}
              onChange={(value) => {
                setState({ ablation: value });
                narrate(`${value === "mean" ? "Mean" : "Zero"} ablation.`);
              }}
            />
            <div className="ci-heads" role="group" aria-label="Toggle heads off">
              {ALL_HEADS.map((id) => {
                const single = freshAblation?.single.find((entry) => entry.id === id);
                const off = ablated.includes(id);
                return (
                  <button
                    type="button"
                    key={id}
                    aria-pressed={off}
                    aria-label={`${id}${off ? ", ablated" : ", active"}${single ? `. Alone it moves copy accuracy by ${((single.mean - score.mean) * 100).toFixed(1)} points` : ""}`}
                    onClick={() => {
                      const next = off ? ablated.filter((entry) => entry !== id) : [...ablated, id];
                      setState({ ablated: next });
                      narrate(`${id} ${off ? "restored" : "ablated"}.`);
                    }}
                  >
                    <strong>{id.replace("H", " H")}</strong>
                    <span>{off ? "off" : "on"}</span>
                    <small>alone {single ? signed((single.mean - score.mean) * 100, 1) : "…"} pts</small>
                  </button>
                );
              })}
            </div>
            <div className="ci-heads__actions">
              <button type="button" className="ci-chip-button" onClick={() => setState({ ablated: ALL_HEADS.slice(0, HEADS) })}>
                Ablate layer 1
              </button>
              <button type="button" className="ci-chip-button" onClick={() => setState({ ablated: ALL_HEADS.slice(HEADS) })}>
                Ablate layer 2
              </button>
              <button type="button" className="ci-chip-button" onClick={() => setState({ ablated: [] })}>
                Restore all
              </button>
            </div>
            {ablatedScore && stripeIntact && stripeAblated ? (
              <>
                <p className="ci-small">Copy accuracy with every head on, then with the heads above switched off:</p>
                <div className="metric-row">
                  <Metric label="Intact" value={pct(score.mean)} tone="forward" />
                  <Metric label={`${ablated.length} off`} value={pct(ablatedScore.mean)} tone="loss" />
                  <Metric label="Change" value={`${signed((ablatedScore.mean - score.mean) * 100, 1)} pts`} />
                  <Metric
                    label="Top next letter"
                    value={`${showToken(ablatedScore.top[0]?.token ?? "")} ${pct(ablatedScore.top[0]?.probability ?? 0, 0)}`}
                  />
                </div>
                <div className="metric-row">
                  <Metric
                    label="Attention on the induction key, L2 H2 · L2 H3"
                    value={`${stripeIntact.map((entry) => entry.induction.toFixed(2)).join(" · ")} → ${stripeAblated.map((entry) => entry.induction.toFixed(2)).join(" · ")}`}
                  />
                </div>
                <PositionBars
                  positions={copyPositions(prompt)}
                  tokens={tokens}
                  intact={score.perPosition}
                  ablated={ablatedScore.perPosition}
                  label={`Probability of the correct next letter at each position: intact mean ${pct(score.mean)}, with ablations ${pct(ablatedScore.mean)}.`}
                />
                <ul className="ci-key">
                  <li><i className="ci-key__intact" /> intact model</li>
                  <li><i className="ci-key__ablated" /> with the heads above switched off</li>
                </ul>
              </>
            ) : (
              <div className="ci-empty" role="status">Averaging each head's output over {REFERENCE_STRINGS} random repeats…</div>
            )}
            <p className="ci-small">
              {ablationMode === "mean"
                ? `Mean ablation replaces a head's output with its average over the second repeat of ${REFERENCE_STRINGS} seeded random-letter strings of period ${period}, not this prompt, so the replacement is a typical value rather than zero.`
                : "Zero-ablation sets the head's output to 0, a value the next layer never saw in training; compare it with Mean."}
            </p>
          </>
        ) : (
          <div className="ci-empty" role="status">
            {load.status === "error"
              ? "Head ablation needs the model weights, which did not load. The cached patching sweeps below still work."
              : "Loading the model weights…"}
          </div>
        )}
      </LabSurface>

      <LabSurface label="Activation patching" className="ci-surface">
        <SurfaceHeading
          kicker="Clean run → corrupted run · one residual vector at a time"
          title="Where does the information the answer needs live?"
        />
        <SegmentedControl
          label="Sweep"
          value={sweepId}
          options={[
            { value: "live", label: "Live · copy task" },
            { value: "speaker", label: "Cached · speaker" },
            { value: "copy", label: "Cached · letters" },
          ]}
          onChange={(value) => {
            const next = value === "live" ? null : cachedSweeps.find((entry) => entry.id === value);
            const start = next
              ? next.recovery[0].reduce((best, entry, index) => (Math.abs(entry) > Math.abs(next.recovery[0][best]) ? index : best), 0)
              : livePatchStart(prompt, corruption);
            setState({ sweep: value, patch: start, layer: 0 });
          }}
        />
        {sweepId === "live" && (
          <SegmentedControl
            label="Corrupt"
            value={corruption}
            options={[
              { value: "source", label: "The letter to copy" },
              { value: "control", label: "An unrelated letter" },
            ]}
            onChange={(value) => {
              const next: Corruption = value === "control" ? "control" : "source";
              setState({ corruption: next, patch: livePatchStart(prompt, next), layer: 0 });
              narrate(next === "control" ? "Corrupting an unrelated letter." : "Corrupting the letter to copy.");
            }}
          />
        )}
        {sweep ? (
          <>
            <div className="ci-pair">
              <div>
                <span>clean</span>
                <TokenStrip
                  tokens={sweepTokens}
                  marks={Object.fromEntries(sweep.differing.map((index) => [index, "differs"]))}
                  label={`Clean prompt: ${sweep.clean}`}
                />
              </div>
              <div>
                <span>corrupted</span>
                <TokenStrip
                  tokens={Array.from(sweep.corrupt)}
                  marks={Object.fromEntries(sweep.differing.map((index) => [index, "differs"]))}
                  label={`Corrupted prompt: ${sweep.corrupt}`}
                />
              </div>
            </div>
            <p className="ci-small">
              Metric, read at the last position: <code>{sweep.metric}</code>
            </p>
            <div className="metric-row">
              <Metric label="Clean" value={sweep.cleanScore.toFixed(2)} tone="forward" />
              <Metric label="Corrupted" value={sweep.corruptScore.toFixed(2)} tone="loss" />
              <Metric label="Gap" value={Math.abs(gap) < 0.01 ? gap.toExponential(1) : gap.toFixed(2)} />
              {sweep.cleanTop && sweep.corruptTop && (
                <Metric
                  label="Top letter, clean → corrupted"
                  value={`${showToken(sweep.cleanTop.token)} ${pct(sweep.cleanTop.probability, 0)} → ${showToken(sweep.corruptTop.token)} ${pct(sweep.corruptTop.probability, 0)}`}
                />
              )}
            </div>
            {Math.abs(gap) < STABLE_MARGIN && (
              <p className="ci-warning" role="note">
                The clean and corrupted runs differ by only {Math.abs(gap) < 0.01 ? gap.toExponential(1) : gap.toFixed(2)} on this
                metric, so every recovery below divides by a tiny number and is not a stable measurement.
              </p>
            )}
            <PatchGrid
              tokens={sweepTokens}
              recovery={sweep.recovery}
              selected={{ layer: patchLayer, position: patchPosition }}
              onSelect={(layer, position) => {
                setState({ layer, patch: position });
                const value = sweep.recovery[layer]?.[position] ?? 0;
                narrate(`After block ${layer + 1}, position ${position}: recovery ${(value * 100).toFixed(0)} percent.`);
              }}
              rowLabels={["After block 1", "After block 2"]}
              differing={sweep.differing}
            />
            <div className="ci-legend-scale" aria-hidden="true">
              <span>−100%</span>
              <i />
              <span>0</span>
              <i className="is-positive" />
              <span>+100% recovered</span>
            </div>
            <p className="ci-readout" role="status">
              Patching the clean residual after block {patchLayer + 1} at position {patchPosition} (
              {showToken(sweepTokens[patchPosition] ?? "")}) into the corrupted run recovers{" "}
              <strong>{Math.abs(selectedRecovery) >= 10 ? selectedRecovery.toFixed(1) + "×" : pct(selectedRecovery, 1)}</strong> of the gap
              {sweep.patched ? ` (metric ${sweep.patched[patchLayer]?.[patchPosition]?.toFixed(2) ?? "—"})` : ""}.
            </p>
            <p className="ci-small">
              {sweep.id === "live"
                ? corruption === "source"
                  ? `Clean: the period-${period} prompt. Corrupted: position ${prompt.source}, the previous ${prompt.target}, becomes ${CORRUPT_LETTER}. Each cell is one extra forward pass from that layer on.`
                  : `Clean: the period-${period} prompt. Corrupted: position ${controlPosition(prompt)}, ${prompt.text[controlPosition(prompt)]}, a letter the answer does not depend on, becomes ${CORRUPT_LETTER}. A control: the runs barely differ, so recovery is not defined well.`
                : sweep.id === "copy"
                  ? "Precomputed offline by precompute_interpretability.py: the same period-13 prompts as Live at period 13 with the first corruption, but the metric is the target logit alone, so the gap differs. Stored in interpretability-cache.json."
                  : "Precomputed offline by precompute_interpretability.py with the target logit as the metric; the corrupted prompt is cut to the clean prompt's length, so it ends at JULIET with no colon and the last character differs. Stored in interpretability-cache.json."}{" "}
              After block 2 only the last position can matter, because nothing after the last block reads other positions.
            </p>
          </>
        ) : (
          <div className="ci-empty" role="status">
            {load.status === "error" ? "The live sweep needs the model weights. Choose a cached sweep." : "Running the sweep…"}
          </div>
        )}
      </LabSurface>
    </div>
  );
}
