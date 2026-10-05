import { useMemo, type CSSProperties } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  decodeStepCost,
  formatBytes,
  formatFlopCount,
  formatSeconds,
  H100_SXM,
  kvBytesPerToken,
  LLAMA3_8B,
  memoryBudget,
  prefillCost,
  ridgePoint,
  toyPass,
  type PassCost,
} from "./kv";
import { CONTEXT_RANGE, GENERATED_RANGE, PROMPT_RANGE, SEQUENCE_RANGE } from "./ranges";
import {
  clampQuality,
  closedForm,
  draftRows,
  GAMMA_RANGE,
  meanAcceptance,
  PROMPT,
  QUALITY_RANGE,
  SEED_RANGE,
  simulate,
  SIM_PASSES,
  trainTargetRows,
  type PassRecord,
} from "./speculative";

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clampInt = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, Math.round(value)));

/* ---------- causal triangle ---------- */

const CELL = 18;
const LEFT = 34;
const TOP = 36;
/** Both SVGs share one width so key columns line up with cache slots; 1.35 px per unit at most. */
const figureWidth = (positions: number) => LEFT + positions * CELL + 16;
const figureStyle = (positions: number) => ({ maxWidth: `${figureWidth(positions) * 1.35}px` });

const tokenLabel = (index: number, prompt: number) =>
  index < prompt ? `t${index + 1}` : `+${index - prompt + 1}`;

function CausalTriangle({
  prompt,
  generated,
  useCache,
}: {
  prompt: number;
  generated: number;
  useCache: boolean;
}) {
  const pass = toyPass(prompt, generated, useCache);
  const n = pass.length;
  const height = TOP + n * CELL + 4;
  const current = n - 1;
  const cellClass = (row: number, column: number) => {
    if (column > row) return "kv-cell kv-cell--masked";
    if (pass.isPrefill) return "kv-cell kv-cell--now";
    if (row === current) return "kv-cell kv-cell--now";
    return useCache ? "kv-cell kv-cell--earlier" : "kv-cell kv-cell--repeat";
  };
  const summary = pass.isPrefill
    ? `Prefill pass over ${prompt} prompt positions: all ${pass.dotProducts} query-key products of the causal triangle are computed at once.`
    : useCache
      ? `Decode step ${generated} with the cache: only the last row, ${pass.dotProducts} query-key products, is computed; the other rows were computed by earlier passes.`
      : `Decode step ${generated} without a cache: all ${pass.dotProducts} products of the triangle are computed again, ${pass.repeatedDotProducts} of them repeats of the previous pass.`;

  return (
    <svg
      className="kv-triangle"
      viewBox={`0 0 ${figureWidth(n)} ${height}`}
      style={figureStyle(n)}
      role="img"
      aria-label={summary}
    >
      <text className="kv-axis-note" x={2} y={TOP - 4}>
        q ↓ k →
      </text>
      {Array.from({ length: n }, (_, column) => {
        const x = LEFT + column * CELL + CELL / 2 + 3;
        return (
          <text
            key={`c-${column}`}
            className={`kv-label ${column >= prompt ? "is-generated" : ""}`}
            x={x}
            y={TOP - 4}
            transform={`rotate(-55 ${x} ${TOP - 4})`}
          >
            {tokenLabel(column, prompt)}
          </text>
        );
      })}
      {Array.from({ length: n }, (_, row) => (
        <g key={`r-${row}`}>
          <text
            className={`kv-label ${row >= prompt ? "is-generated" : ""} ${row === current && !pass.isPrefill ? "is-current" : ""}`}
            x={LEFT - 5}
            y={TOP + row * CELL + CELL / 2 + 3.5}
            textAnchor="end"
          >
            {tokenLabel(row, prompt)}
          </text>
          {Array.from({ length: n }, (_, column) => (
            <rect
              key={column}
              className={cellClass(row, column)}
              x={LEFT + column * CELL + 1}
              y={TOP + row * CELL + 1}
              width={CELL - 2}
              height={CELL - 2}
              rx={2}
            />
          ))}
        </g>
      ))}
      {!pass.isPrefill && (
        <rect
          className="kv-row-outline"
          x={LEFT - 1}
          y={TOP + current * CELL}
          width={n * CELL + 2}
          height={CELL}
          rx={3}
        />
      )}
      {prompt < n && (
        <line
          className="kv-split"
          x1={LEFT + prompt * CELL}
          x2={LEFT + prompt * CELL}
          y1={TOP - 16}
          y2={TOP + n * CELL}
        />
      )}
    </svg>
  );
}

function CacheRows({
  prompt,
  generated,
  useCache,
}: {
  prompt: number;
  generated: number;
  useCache: boolean;
}) {
  const n = prompt + generated;
  const isPrefill = generated === 0;
  const rowHeight = 16;
  const height = 2 * rowHeight + 10;
  const state = (column: number) => {
    if (!useCache) return "kv-slot kv-slot--recomputed";
    if (isPrefill || column === n - 1) return "kv-slot kv-slot--written";
    return "kv-slot kv-slot--read";
  };
  return (
    <div className="kv-cache-rows">
      <svg
        viewBox={`0 0 ${figureWidth(n)} ${height}`}
        style={figureStyle(n)}
        role="img"
        aria-label={
          useCache
            ? isPrefill
              ? `Key and value cache: prefill writes ${n} positions.`
              : `Key and value cache: ${n - 1} positions read, 1 written this step, ${n} held afterwards.`
            : `No cache: keys and values for all ${n} positions are recomputed and nothing is kept.`
        }
      >
        {["K", "V"].map((name, row) => (
          <g key={name}>
            <text
              className="kv-label"
              x={LEFT - 5}
              y={row * (rowHeight + 4) + rowHeight / 2 + 4}
              textAnchor="end"
            >
              {name}
            </text>
            {Array.from({ length: n }, (_, column) => (
              <rect
                key={column}
                className={state(column)}
                x={LEFT + column * CELL + 1}
                y={row * (rowHeight + 4)}
                width={CELL - 2}
                height={rowHeight}
                rx={2}
              />
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}

/* ---------- time bars ---------- */

const TIME_MIN = -5; // 10 µs
const TIME_MAX = 1; // 10 s
const timePct = (seconds: number) =>
  `${Math.min(100, Math.max(0, ((Math.log10(Math.max(seconds, 1e-9)) - TIME_MIN) / (TIME_MAX - TIME_MIN)) * 100))}%`;

function PhaseBars({ label, cost, detail }: { label: string; cost: PassCost; detail: string }) {
  return (
    <div className="kv-phase-row">
      <div className="kv-phase-row__head">
        <strong>{label}</strong>
        <span className={`kv-bound kv-bound--${cost.bound}`}>{cost.bound}-bound</span>
        <b>{formatSeconds(cost.seconds)}</b>
      </div>
      <div
        className="kv-phase-bars"
        role="img"
        aria-label={`${label}: arithmetic alone would take ${formatSeconds(cost.computeSeconds)}, moving memory alone ${formatSeconds(cost.memorySeconds)}. Modeled time ${formatSeconds(cost.seconds)}, ${cost.bound}-bound.`}
      >
        <span>arithmetic</span>
        <i>
          <b
            className={`kv-time kv-time--compute ${cost.bound === "compute" ? "is-binding" : ""}`}
            style={{ width: timePct(cost.computeSeconds) }}
          />
        </i>
        <em>{formatSeconds(cost.computeSeconds)}</em>
        <span>memory</span>
        <i>
          <b
            className={`kv-time kv-time--memory ${cost.bound === "memory" ? "is-binding" : ""}`}
            style={{ width: timePct(cost.memorySeconds) }}
          />
        </i>
        <em>{formatSeconds(cost.memorySeconds)}</em>
      </div>
      <small>{detail}</small>
    </div>
  );
}

/* ---------- speculative decoding ---------- */

/** Spaces are invisible in a transcript, so they are drawn as a visible mark. */
const visible = (text: string) => text.replace(/ /g, "␣");

function PassLine({ index, pass, gamma }: { index: number; pass: PassRecord; gamma: number }) {
  const kept = pass.proposed.slice(0, pass.accepted);
  return (
    <li>
      <b>Pass {index + 1}</b>
      <span>
        draft proposed <code>{visible(pass.proposed)}</code>
      </span>
      <span>
        target kept {pass.accepted} of {gamma}
        {pass.accepted > 0 ? (
          <>
            : <code>{visible(kept)}</code>
          </>
        ) : null}
        {pass.rejected !== null ? (
          <>
            ; refused <code>{visible(pass.rejected)}</code> and wrote <code>{visible(pass.fromTarget)}</code> instead
          </>
        ) : (
          <>
            ; all kept, so it wrote one more: <code>{visible(pass.fromTarget)}</code>
          </>
        )}
      </span>
      <strong>+{pass.produced} token{pass.produced === 1 ? "" : "s"}</strong>
    </li>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const prompt = clampInt(asNumber(state, "context", 7), PROMPT_RANGE.min, PROMPT_RANGE.max);
  const generated = clampInt(asNumber(state, "generated", 4), GENERATED_RANGE.min, GENERATED_RANGE.max);
  const useCache = typeof state.useCache === "boolean" ? state.useCache : true;
  const realContext = clampInt(asNumber(state, "realContext", 8192), CONTEXT_RANGE.min, CONTEXT_RANGE.max);
  const sequences = clampInt(asNumber(state, "sequences", 16), SEQUENCE_RANGE.min, SEQUENCE_RANGE.max);
  const specGamma = clampInt(asNumber(state, "specGamma", 4), GAMMA_RANGE.min, GAMMA_RANGE.max);
  const specQuality = clampQuality(asNumber(state, "specQuality", 0.7));
  const specSeed = clampInt(asNumber(state, "specSeed", 7), SEED_RANGE.min, SEED_RANGE.max);

  // The target bigram is trained once. The draft, the acceptance rate and one seeded simulation per
  // draft length are recomputed when the quality or the seed moves; the draft length only picks a row.
  const target = useMemo(() => trainTargetRows(), []);
  const speculation = useMemo(() => {
    const draft = draftRows(target, specQuality);
    const alpha = meanAcceptance(target, draft);
    const sweep = Array.from({ length: GAMMA_RANGE.max }, (_, index) => {
      const gamma = index + 1;
      return { gamma, closed: closedForm(alpha, gamma), run: simulate({ p: target, q: draft, gamma, seed: specSeed }) };
    });
    return { alpha, sweep };
  }, [target, specQuality, specSeed]);
  const spec = speculation.sweep[specGamma - 1];
  const specGap = spec.run.meanTokens - spec.closed;
  const specZ = spec.run.standardError > 0 ? Math.abs(specGap) / spec.run.standardError : 0;
  const signedTokens = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;

  const pass = toyPass(prompt, generated, useCache);
  const n = pass.length;
  const model = LLAMA3_8B;
  const perToken = kvBytesPerToken(model);
  const perTokenMha = kvBytesPerToken(model, model.queryHeads);
  const budget = memoryBudget(model, realContext, sequences);
  const prefill = prefillCost(model, realContext);
  const cached = decodeStepCost(model, realContext, sequences, true);
  const recompute = decodeStepCost(model, realContext, sequences, false);
  const capacity = H100_SXM.memoryBytes;
  const barMax = Math.max(capacity, budget.weights + budget.cache) * 1.02;
  const pct = (bytes: number) => `${(bytes / barMax) * 100}%`;

  const caption = pass.isPrefill
    ? `Prefill: t1…t${prompt} go through the model together in one pass, and its last row predicts +1.`
    : `Decode step ${generated}: this pass feeds ${tokenLabel(n - 1, prompt)} at position ${n} and predicts +${generated + 1}.`;

  return (
    <div className="kv-lab">
      <LabSurface label="Causal attention work" className="kv-work-card">
        <SurfaceHeading
          kicker={
            pass.isPrefill
              ? "Prefill pass · all rows at once"
              : `Decode step ${generated} · ${useCache ? "with cache" : "recompute"}`
          }
          title="Rows are queries, columns are keys; the mask keeps only the lower triangle"
        />
        <p className="kv-caption">{caption}</p>
        <div className="kv-figure">
          <CausalTriangle prompt={prompt} generated={generated} useCache={useCache} />
          <CacheRows prompt={prompt} generated={generated} useCache={useCache} />
        </div>
        <ul className="kv-legend">
          <li>
            <i className="kv-swatch kv-cell--now" />
            computed in this pass
          </li>
          {!pass.isPrefill && useCache && (
            <li>
              <i className="kv-swatch kv-cell--earlier" />
              computed by an earlier pass (scores are not stored)
            </li>
          )}
          {!pass.isPrefill && !useCache && (
            <li>
              <i className="kv-swatch kv-cell--repeat" />
              computed again: same values as the last pass
            </li>
          )}
          {useCache && !pass.isPrefill && (
            <li>
              <i className="kv-swatch kv-slot--read" />
              K/V read from cache
            </li>
          )}
          {useCache && (
            <li>
              <i className="kv-swatch kv-slot--written" />
              K/V written this pass
            </li>
          )}
          {!useCache && (
            <li>
              <i className="kv-swatch kv-slot--recomputed" />
              K/V recomputed, then dropped
            </li>
          )}
        </ul>
        <div className="metric-row kv-work-metrics">
          <Metric
            label="New Q, K, V"
            value={`${pass.projectedPositions}`}
            tone={pass.projectedPositions > 1 ? "loss" : "forward"}
          />
          <Metric
            label="Q·K this pass"
            value={`${pass.dotProducts}`}
            tone={pass.repeatedDotProducts > 0 ? "loss" : undefined}
          />
          <Metric label="K/V read" value={`${pass.readFromCache}`} />
          <Metric
            label="Q·K whole run"
            value={`${useCache ? pass.runTotalCached : pass.runTotalRecompute}`}
          />
        </div>
        <p className="lab-note">
          Counts are per head per layer; {model.name} repeats them across {model.queryHeads} query heads and{" "}
          {model.layers} layers. Over prefill and {generated} decode step{generated === 1 ? "" : "s"}, the
          cache needs {pass.runTotalCached} products in total and recomputing needs {pass.runTotalRecompute}.
        </p>
      </LabSurface>

      <LabSurface label="KV cache controls" className="kv-controls-card">
        <SurfaceHeading kicker="Toy sequence" title="Prompt, output, and whether to keep K and V" />
        <SegmentedControl
          label="KV cache"
          value={useCache ? "on" : "off"}
          options={[
            { value: "on", label: "Use cache" },
            { value: "off", label: "Recompute" },
          ]}
          onChange={(value) => setState({ useCache: value === "on" })}
        />
        <RangeControl
          label="Prompt tokens"
          min={PROMPT_RANGE.min}
          max={PROMPT_RANGE.max}
          step={1}
          value={prompt}
          onChange={(value) => setState({ context: value })}
        />
        <RangeControl
          label="Generated tokens"
          min={GENERATED_RANGE.min}
          max={GENERATED_RANGE.max}
          step={1}
          value={generated}
          onChange={(value) => setState({ generated: value })}
          format={(value) => (value === 0 ? "0 · prefill" : String(value))}
        />
        <p className="lab-note">
          Generated tokens 0 shows the prefill pass. Each step after that feeds the newest token. With the
          cache, the step projects one position and reads{" "}
          {n - 1 > 0 && !pass.isPrefill ? n - 1 : "the earlier"} stored keys and values; without it, the whole
          prefix is run again.
        </p>
      </LabSurface>

      <LabSurface label="Memory budget" className="kv-memory-card">
        <SurfaceHeading
          kicker={`${model.name} · bf16 · ${model.kvHeads} KV heads × ${model.headDim} dims × ${model.layers} layers`}
          title="Weights are fixed; the cache grows with every token of every sequence"
        />
        <div className="kv-memory-controls">
          <label className="range-control">
            <span>
              Context length
              <output>{realContext.toLocaleString("en-US")} tokens</output>
            </span>
            <input
              type="range"
              aria-label="Context length"
              aria-valuetext={`${realContext.toLocaleString("en-US")} tokens`}
              min={Math.log2(CONTEXT_RANGE.min)}
              max={Math.log2(CONTEXT_RANGE.max)}
              step={1}
              value={Math.log2(realContext)}
              style={
                {
                  "--range-progress": `${((Math.log2(realContext) - Math.log2(CONTEXT_RANGE.min)) / (Math.log2(CONTEXT_RANGE.max) - Math.log2(CONTEXT_RANGE.min))) * 100}%`,
                } as CSSProperties
              }
              onChange={(event) =>
                setState({ realContext: Math.round(Math.pow(2, Number(event.target.value))) })
              }
            />
          </label>
          <RangeControl
            label="Concurrent sequences"
            min={SEQUENCE_RANGE.min}
            max={SEQUENCE_RANGE.max}
            step={1}
            value={sequences}
            onChange={(value) => setState({ sequences: value })}
          />
        </div>
        <div
          className="kv-memory-bar"
          role="img"
          aria-label={`Weights ${formatBytes(budget.weights)} plus KV cache ${formatBytes(budget.cache)} against ${formatBytes(capacity)} of accelerator memory. ${budget.fits ? `${formatBytes(budget.free)} left` : `Over by ${formatBytes(-budget.free)}`}.`}
        >
          <b className="kv-mem kv-mem--weights" style={{ width: pct(budget.weights) }}>
            <span>weights {formatBytes(budget.weights)}</span>
          </b>
          <b
            className={`kv-mem kv-mem--cache ${budget.fits ? "" : "is-over"}`}
            style={{ width: pct(budget.cache) }}
          >
            <span>KV {formatBytes(budget.cache)}</span>
          </b>
          <i className="kv-capacity" style={{ left: pct(capacity) }}>
            <span>80 GB</span>
          </i>
        </div>
        <div className="metric-row">
          <Metric
            label="KV per token"
            value={`${formatBytes(perToken)} (${perToken / 1024} KiB)`}
            tone="forward"
          />
          <Metric
            label="KV cache total"
            value={formatBytes(budget.cache)}
            tone={budget.cache > budget.weights ? "loss" : undefined}
          />
          <Metric
            label={budget.fits ? "Memory left" : "Over by"}
            value={formatBytes(Math.abs(budget.free))}
            tone={budget.fits ? undefined : "loss"}
          />
          <Metric label="Sequences that fit" value={`${budget.maxSequences}`} />
        </div>
        <p className="lab-note">
          2 × {model.layers} layers × {model.kvHeads} KV heads × {model.headDim} × 2 bytes ={" "}
          {perToken.toLocaleString("en-US")} bytes per token. Grouped-query attention is why it is only{" "}
          {model.kvHeads} heads: with one KV head per query head ({model.queryHeads}) it would be{" "}
          {formatBytes(perTokenMha)} per token and {formatBytes(budget.cache * 4)} here. Your toy sequence of{" "}
          {n} tokens would hold {formatBytes(n * perToken)}.
        </p>
      </LabSurface>

      <LabSurface label="Prefill versus decode" className="kv-phase-card">
        <SurfaceHeading
          kicker={`Roofline model · ${H100_SXM.name} peaks: ${H100_SXM.peakFlops / 1e12} TFLOP/s, ${H100_SXM.bandwidth / 1e12} TB/s`}
          title="Each pass is limited by arithmetic or by memory traffic, whichever takes longer"
        />
        <div className="kv-phases">
          <PhaseBars
            label={`Prefill, one ${realContext.toLocaleString("en-US")}-token prompt`}
            cost={prefill}
            detail={`${formatFlopCount(prefill.flops)} over ${formatBytes(prefill.bytes)} of weights and cache writes · ${prefill.intensity.toFixed(0)} FLOP/byte. This is time to first token.`}
          />
          <PhaseBars
            label={`Decode step, cache on, ${sequences} sequence${sequences === 1 ? "" : "s"}`}
            cost={cached}
            detail={`${formatFlopCount(cached.flops)} over ${formatBytes(cached.bytes)} (weights once + every cache) · ${cached.intensity.toFixed(1)} FLOP/byte · ${Math.round(sequences / cached.seconds).toLocaleString("en-US")} tokens/s in total, ${Math.round(1 / cached.seconds)} per sequence.`}
          />
          <PhaseBars
            label={`Decode step, recompute, ${sequences} sequence${sequences === 1 ? "" : "s"}`}
            cost={recompute}
            detail={`${formatFlopCount(recompute.flops)}: every sequence reruns its ${realContext.toLocaleString("en-US")}-token prefix, a full prefill per new token.`}
          />
        </div>
        <p className="lab-note">
          Time = max(FLOPs ÷ peak FLOP/s, bytes ÷ bandwidth), at 100% of spec-sheet peak. Real kernels reach
          less, and activations, sampling and scheduling are left out, so read these as lower bounds from a
          model, not measurements. The switch point is {ridgePoint().toFixed(0)} FLOP per byte.
        </p>
      </LabSurface>

      <LabSurface label="Speculative decoding" className="kv-spec-card">
        <SurfaceHeading
          kicker={`Draft proposes ${specGamma} · draft quality ${specQuality.toFixed(2)} · seed ${specSeed} · ${SIM_PASSES.toLocaleString("en-US")} simulated passes`}
          title="A cheap guess is kept only where the target would have said the same"
        />
        <div className="kv-spec-controls">
          <RangeControl
            label="Draft length"
            min={GAMMA_RANGE.min}
            max={GAMMA_RANGE.max}
            step={1}
            value={specGamma}
            format={(value) => `${value} proposed per pass`}
            onChange={(value) => setState({ specGamma: value })}
          />
          <RangeControl
            label="Draft quality"
            min={QUALITY_RANGE.min}
            max={QUALITY_RANGE.max}
            step={0.05}
            value={specQuality}
            format={(value) => (value >= 1 ? "1.00 · the target itself" : value.toFixed(2))}
            onChange={(value) => setState({ specQuality: clampQuality(value) })}
          />
          <RangeControl
            label="Seed"
            min={SEED_RANGE.min}
            max={SEED_RANGE.max}
            step={1}
            value={specSeed}
            onChange={(value) => setState({ specSeed: value })}
          />
        </div>
        <ol className="kv-spec-passes" aria-label={`The first ${spec.run.transcript.length} verification passes for this seed`}>
          {spec.run.transcript.map((entry, index) => (
            <PassLine key={index} index={index} pass={entry} gamma={specGamma} />
          ))}
        </ol>
        <p className="kv-spec-output">
          <span>Text those passes wrote, after the prompt “{visible(PROMPT)}”</span>
          <code>{visible(spec.run.text)}</code>
        </p>
        <div className="metric-row">
          <Metric label="Acceptance rate α" value={speculation.alpha.toFixed(3)} />
          <Metric label="Closed form, tokens per pass" value={spec.closed.toFixed(2)} tone="forward" />
          <Metric
            label="Simulated, tokens per pass"
            value={`${spec.run.meanTokens.toFixed(2)} ± ${spec.run.standardError.toFixed(2)}`}
            tone="forward"
          />
          <Metric label="Without drafting" value="1 token per pass" />
        </div>
        <p className="lab-note">
          {spec.run.standardError > 0
            ? `Simulation minus closed form: ${signedTokens(specGap)} tokens, ${specZ.toFixed(1)} standard errors over ${SIM_PASSES.toLocaleString("en-US")} passes.`
            : `Every proposal is kept in every pass, so the simulation equals the closed form exactly: ${spec.closed.toFixed(2)} tokens.`}{" "}
          The closed form is (1 − α^(γ+1)) ÷ (1 − α) with γ = {specGamma} and α the acceptance rate averaged over
          the contexts the target visits. It assumes each proposal is kept independently with the same chance, which
          the simulation does not.
        </p>
        <LineChart
          label="Tokens per target pass against draft length at the current draft quality"
          xLabel="draft length γ"
          yLabel="tokens per verification pass"
          xDomain={[GAMMA_RANGE.min, GAMMA_RANGE.max]}
          yDomain={[0, GAMMA_RANGE.max + 1]}
          series={[
            {
              id: "closed",
              name: "closed form",
              tone: "forward",
              points: speculation.sweep.map((entry) => ({ x: entry.gamma, y: entry.closed })),
              format: (value: number) => value.toFixed(2),
            },
            {
              id: "simulated",
              name: "simulated",
              tone: "attention",
              dash: "dashed",
              points: speculation.sweep.map((entry) => ({ x: entry.gamma, y: entry.run.meanTokens })),
              format: (value: number) => value.toFixed(2),
            },
            {
              id: "plain",
              name: "no drafting",
              tone: "muted",
              dash: "dotted",
              points: speculation.sweep.map((entry) => ({ x: entry.gamma, y: 1 })),
              format: (value: number) => value.toFixed(2),
            },
          ]}
          footnote={`Each simulated point is its own seeded run of ${SIM_PASSES.toLocaleString("en-US")} passes. The closed form can never pass 1 ÷ (1 − α) = ${(1 / Math.max(1e-9, 1 - speculation.alpha)).toFixed(2)}, however long the draft.`}
        />
      </LabSurface>
    </div>
  );
}
