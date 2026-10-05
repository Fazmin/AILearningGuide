import { useMemo } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  tinyNextDistribution,
  trainTinyLora,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  adapterBytes,
  adapterParameters,
  architectureFor,
  baseParameters,
  BITS,
  calculatorState,
  CUSTOM,
  D_MAX,
  D_MIN,
  D_STEP,
  formatBillions,
  formatCount,
  formatMegabytes,
  formatShare,
  LAYERS_MAX,
  LAYERS_MIN,
  PRESETS,
  RANK_MAX,
  RANK_MIN,
  readCalculator,
  TARGETS,
  toyShare,
  type TargetId,
} from "./calculator";
import { centerRows, checkMerge, energyCaptured, numericalRank, singularValues } from "./lowrank";

const source = TINY_CORPORA.harbor;
const target = TINY_CORPORA.recipes;
const PRETRAIN_EPOCHS = 60;
const V = TINY_VOCAB_SIZE;
const MAX_RANK = 12;
const PROBES = [" ", "e", "s", "l", "a"];

const glyph = (character: string) => (character === " " ? "␣" : character);

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" ? state[key] : fallback;
const asBoolean = (state: ModuleContext["state"], key: string, fallback: boolean) =>
  typeof state[key] === "boolean" ? state[key] : fallback;

/** Shared state can arrive from a link or an old snapshot, so clamp to the controls. */
const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, Number.isFinite(value) ? value : low));

/* -------------------------------------------------------------------------- */
/* Matrix drawing                                                              */
/* -------------------------------------------------------------------------- */

function maxAbs(values: ArrayLike<number>) {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) maximum = Math.max(maximum, Math.abs(values[index]));
  return maximum;
}

/** Signed cells: blue raises, red lowers, depth is |value| over this matrix's own maximum. */
function MatrixCells({
  values,
  rows,
  columns,
  x,
  y,
  cell,
}: {
  values: ArrayLike<number>;
  rows: number;
  columns: number;
  x: number;
  y: number;
  cell: number;
}) {
  const scale = maxAbs(values);
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = values[row * columns + column];
      const magnitude = scale > 0 ? Math.abs(value) / scale : 0;
      cells.push(
        <rect
          key={`${row}-${column}`}
          x={x + column * cell}
          y={y + row * cell}
          width={cell}
          height={cell}
          style={{
            fill:
              magnitude < 0.004
                ? "var(--panel-muted)"
                : `color-mix(in srgb, ${value >= 0 ? "var(--forward)" : "var(--loss)"} ${Math.round(4 + magnitude * 96)}%, var(--panel-solid))`,
          }}
        />,
      );
    }
  }
  return (
    <g>
      {cells}
      <rect className="lra-matrix__frame" x={x} y={y} width={columns * cell} height={rows * cell} />
    </g>
  );
}

function ProductDiagram({
  rank,
  up,
  down,
  delta,
  scaleText,
}: {
  rank: number;
  up: Float32Array;
  down: Float32Array;
  delta: Float64Array;
  scaleText: string;
}) {
  const cell = 7;
  const size = V * cell;
  const bandMax = MAX_RANK * cell;
  const gap = 16;
  const productX = 24 + bandMax + gap;
  const productY = 30 + bandMax + gap;
  const bX = productX - gap - rank * cell;
  const aY = productY - gap - rank * cell;
  const width = productX + size + 12;
  const height = productY + size + 30;
  const allZero = maxAbs(up) === 0;
  return (
    <svg
      className="lra-product"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`B is ${V} by ${rank}, A is ${rank} by ${V}, and their scaled product is the ${V} by ${V} update. ${allZero ? "B is still all zeros, so the update is exactly zero." : `Largest update entry ${maxAbs(delta).toFixed(2)} after removing each row's mean.`}`}
    >
      <text className="lra-product__title" x={productX} y={aY - 8}>
        A · {rank} × {V} · random start
      </text>
      <MatrixCells values={down} rows={rank} columns={V} x={productX} y={aY} cell={cell} />
      <text className="lra-product__title" x={bX - 6} y={productY + size + 16} textAnchor="start">
        B · {V} × {rank}
      </text>
      <text className="lra-product__note" x={bX - 6} y={productY + size + 27} textAnchor="start">
        zero start
      </text>
      <MatrixCells values={up} rows={V} columns={rank} x={bX} y={productY} cell={cell} />
      <MatrixCells values={delta} rows={V} columns={V} x={productX} y={productY} cell={cell} />
      <text className="lra-product__title" x={productX + size} y={productY + size + 16} textAnchor="end">
        ΔW = {scaleText} · B · A · 30 × 30
      </text>
      <text className="lra-product__note" x={productX + size} y={productY + size + 27} textAnchor="end">
        each row&rsquo;s mean removed
      </text>

    </svg>
  );
}

function FullDeltaMap({ delta }: { delta: Float64Array }) {
  const cell = 7;
  const size = V * cell;
  return (
    <svg
      className="lra-full"
      viewBox={`0 0 ${size + 4} ${size + 32}`}
      role="img"
      aria-label={`Full fine-tune update, ${V} by ${V}, row means removed. Largest entry ${maxAbs(delta).toFixed(2)}.`}
    >
      <MatrixCells values={delta} rows={V} columns={V} x={2} y={2} cell={cell} />
      <text className="lra-product__title" x={2} y={size + 18}>
        full fine-tune ΔW · 900 trained
      </text>
      <text className="lra-product__note" x={2} y={size + 29}>
        same epochs, each row&rsquo;s mean removed
      </text>
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Explore                                                                     */
/* -------------------------------------------------------------------------- */

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const rank = clamp(Math.round(asNumber(state, "rank", 4)), 1, MAX_RANK);
  const alpha = clamp(Math.round(asNumber(state, "alpha", 8)), 1, 24);
  const epochs = clamp(Math.round(asNumber(state, "epochs", 24)), 0, 48);
  const adapterOn = asBoolean(state, "adapterOn", true);
  const merged = asBoolean(state, "merged", false);
  const probe = typeof state.probe === "string" && PROBES.includes(state.probe) ? state.probe : " ";

  const base = useMemo(
    () => trainTinyModel({ text: source.text, epochs: PRETRAIN_EPOCHS, seed: 1 }),
    [],
  );

  const lora = useMemo(
    () =>
      trainTinyLora({
        base: base.weights,
        text: target.text,
        rank,
        alpha,
        epochs,
        batchSize: 16,
        learningRate: 0.6,
        seed: 5,
      }),
    [alpha, base.weights, epochs, rank],
  );

  /** Full fine-tuning at the same budget, for the parameter and spectrum comparisons. */
  const fullFineTune = useMemo(
    () =>
      trainTinyModel({
        text: target.text,
        epochs,
        learningRate: 0.6,
        init: base.weights,
        seed: 5,
      }),
    [base.weights, epochs],
  );

  const spectra = useMemo(() => {
    const loraDelta = centerRows(lora.delta, V);
    const fullRaw = new Float64Array(V * V);
    for (let index = 0; index < fullRaw.length; index += 1) {
      fullRaw[index] = fullFineTune.weights[index] - base.weights[index];
    }
    const fullDelta = centerRows(fullRaw, V);
    const loraValues = singularValues(loraDelta, V, V);
    const fullValues = singularValues(fullDelta, V, V);
    return { loraDelta, fullDelta, loraValues, fullValues };
  }, [base.weights, fullFineTune.weights, lora.delta]);

  const mergeCheck = useMemo(
    () => checkMerge(lora.merged, base.weights, lora.up, lora.down, rank, lora.scale, V, target.text),
    [base.weights, lora, rank],
  );

  const active = adapterOn || merged;
  const serving = active ? lora.merged : base.weights;
  const targetBase = tinyCrossEntropy(base.weights, target.text);
  const targetAdapted = tinyCrossEntropy(lora.merged, target.text);
  const sourceBase = tinyCrossEntropy(base.weights, source.text);
  const sourceAdapted = tinyCrossEntropy(lora.merged, source.text);
  const fullTargetLoss = tinyCrossEntropy(fullFineTune.weights, target.text);
  /** Merged, the adapter is gone. Attached, both matrices have to be served. */
  const servingWeights = merged
    ? lora.fullParameters
    : lora.fullParameters + (adapterOn ? lora.trainableParameters : 0);

  const loraRank = numericalRank(spectra.loraValues);
  const fullEnergyAtRank = energyCaptured(spectra.fullValues, rank);
  const untrained = epochs === 0;

  const probeRows = useMemo(() => {
    const before = tinyNextDistribution(base.weights, probe);
    const after = tinyNextDistribution(lora.merged, probe);
    return Array.from({ length: V }, (_, index) => ({ index, before: before[index], after: after[index] }))
      .sort((left, right) => Math.max(right.before, right.after) - Math.max(left.before, left.after))
      .slice(0, 7);
  }, [base.weights, lora.merged, probe]);

  /**
   * The calculator is closed-form arithmetic on a published architecture, so nothing on it is trained
   * and it costs no render time, but its numbers still come from clamped settings.
   */
  const calc = readCalculator(state);
  const architecture = architectureFor(calc);
  const counts = adapterParameters(architecture, calc.targets, calc.rank);
  const baseCount = baseParameters(architecture);
  const realShare = counts.total / baseCount;
  const realAtToyRank = adapterParameters(architecture, calc.targets, rank).total / baseCount;
  const toyFraction = toyShare(rank);
  const setCalculator = (patch: Partial<ReturnType<typeof readCalculator>>) => setState(calculatorState({ ...calc, ...patch }));
  const toggleTarget = (id: TargetId) => {
    const next = calc.targets.includes(id) ? calc.targets.filter((entry) => entry !== id) : [...calc.targets, id];
    setCalculator({ targets: next });
    narrate(`${id === "mlp" ? "MLP" : id} ${calc.targets.includes(id) ? "removed from" : "added to"} the targeted matrices.`);
  };

  const energySeries = (values: number[]) =>
    Array.from({ length: 16 }, (_, index) => ({ x: index + 1, y: energyCaptured(values, index + 1) }));

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="The low-rank update" className="tg-matrix-panel lra-hero">
        <SurfaceHeading
          kicker={`rank ${rank} · alpha ${alpha} · scale ${(alpha / rank).toFixed(2)}`}
          title="One frozen table, plus two thin trainable ones"
          aside={<span className="tg-badge">{lora.trainableParameters} of {lora.fullParameters} weights train</span>}
        />
        <div className="lra-hero__top">
          <div className="lra-hero__dials">
            <RangeControl
              label="Rank"
              min={1}
              max={MAX_RANK}
              step={1}
              value={rank}
              format={(value) => `${value} · ${value * (V + V)} weights`}
              onChange={(value) => {
                setState({ rank: value, merged: false });
                narrate(`Rank ${value}. ${value * (V + V)} trainable weights.`);
              }}
            />
            <RangeControl
              label="Alpha"
              min={1}
              max={24}
              step={1}
              value={alpha}
              format={(value) => `${value} · scale ${(value / rank).toFixed(2)}`}
              onChange={(value) => setState({ alpha: value, merged: false })}
            />
            {alpha !== rank && (
              <div className="tg-callout is-warning">
                <strong>Changing rank at a fixed alpha also changes the scale, now {(alpha / rank).toFixed(2)}.</strong>
                <span>
                  A rank sweep at fixed alpha varies capacity and update size together, so a worse
                  result at higher rank has two possible causes.
                </span>
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => {
                    setState({ alpha: rank, merged: false });
                    narrate(`Alpha set to ${rank}, so the scale is exactly 1.`);
                  }}
                >
                  Set alpha = rank, for a scale of 1
                </button>
              </div>
            )}
            <BarList
              label="Trainable parameter count"
              items={[
                {
                  id: "full",
                  label: "Full fine-tuning",
                  value: lora.fullParameters,
                  display: `${lora.fullParameters}`,
                  tone: "loss",
                  detail: `d · k = 30 · 30 · target loss ${fullTargetLoss.toFixed(3)}`,
                },
                {
                  id: "lora",
                  label: `LoRA rank ${rank}`,
                  value: lora.trainableParameters,
                  display: `${lora.trainableParameters}`,
                  tone: "forward",
                  emphasis: true,
                  detail: `r · (d + k) = ${rank} · 60 · ${Math.round((lora.trainableParameters / lora.fullParameters) * 100)}% · target loss ${lora.finalLoss.toFixed(3)}`,
                },
              ]}
              max={lora.fullParameters}
            />
          </div>
          <div className="lra-hero__diagram">
            <ProductDiagram
              rank={rank}
              up={lora.up}
              down={lora.down}
              delta={spectra.loraDelta}
              scaleText={`(${alpha}/${rank})`}
            />
          </div>
        </div>
        <div className="lra-hero__bottom">
          <div>
            <LineChart
              label="Share of each update's squared size captured by its first k directions"
              xLabel="directions kept (k), from the largest singular value"
              yLabel="share of update energy"
              xDomain={[1, 16]}
              yDomain={[0, 1]}
              marker={{ x: rank, label: `rank ${rank}` }}
              series={[
                {
                  id: "full",
                  name: "full fine-tune ΔW",
                  tone: "loss",
                  dash: "dashed",
                  // The legend reports the value at the current rank, not at the right edge.
                  format: () => `${Math.round(fullEnergyAtRank * 100)}% at k = ${rank}`,
                  points: untrained ? [] : energySeries(spectra.fullValues),
                },
                {
                  id: "lora",
                  name: `LoRA ΔW, rank ${rank}`,
                  tone: "forward",
                  format: () => `${Math.round(energyCaptured(spectra.loraValues, rank) * 100)}% at k = ${rank}`,
                  points: untrained ? [] : energySeries(spectra.loraValues),
                },
              ]}
              footnote={
                untrained
                  ? "Both updates are zero before training, so neither has any directions to count."
                  : `Singular values of each row-centred update. LoRA's ΔW has exactly ${loraRank} non-zero directions, so its curve reaches 100% at k = ${loraRank}. The full fine-tune's first ${rank} directions hold ${Math.round(fullEnergyAtRank * 100)}% of its update.`
              }
            />
          </div>
          <FullDeltaMap delta={spectra.fullDelta} />
        </div>
        <FormulaWithValues
          label="Effective weights"
          expression={`W + (${alpha} / ${rank}) · B · A`}
          result={`${lora.trainableParameters} trainable of ${lora.fullParameters}`}
          detail={`B starts at zero, so an untrained adapter changes nothing. Full fine-tuning puts all ${lora.fullParameters} weights in the optimizer; ${fullFineTune.updatedParameters} of them actually move, because rows for characters Recipe steps never uses as context get no gradient.`}
          tone="gradient"
        />
      </LabSurface>

      <LabSurface label="Adapter training and behaviour" className="tg-loss-panel lra-behaviour a11-narrow-chart">
        <SurfaceHeading kicker="Measured on the target corpus" title="Only B and A receive gradients" />
        <RangeControl
          label="Adapter epochs"
          min={0}
          max={48}
          step={1}
          value={epochs}
          format={(value) => (value === 0 ? "untrained" : `${value} passes`)}
          onChange={(value) => setState({ epochs: value, merged: false })}
        />
        <LineChart
          label="Adapter training loss"
          xLabel="optimizer step"
          yLabel="batch loss on recipes (nats/char)"
          yDomain={[1.7, UNIFORM_CROSS_ENTROPY + 0.1]}
          series={[
            {
              id: "lora",
              name: `LoRA rank ${rank}, per-batch`,
              tone: "forward",
              // With zero epochs there is no history, so show where an untrained
              // adapter sits rather than dropping the series off the chart.
              points: lora.history.length
                ? lora.history.map((point) => ({ x: point.step, y: point.loss }))
                : [
                    { x: 0, y: targetBase },
                    { x: 1, y: targetBase },
                  ],
            },
            {
              id: "frozen",
              name: "frozen base, full corpus",
              tone: "muted",
              dash: "dotted",
              points: [
                { x: 0, y: targetBase },
                { x: Math.max(1, lora.history.at(-1)?.step ?? 1), y: targetBase },
              ],
            },
          ]}
          footnote={`The solid line averages minibatch losses between history points, so it is noisy. After training, the full-corpus recipe loss with the adapter is ${targetAdapted.toFixed(3)}.`}
        />
        <SegmentedControl
          label="Next character after"
          value={probe}
          options={PROBES.map((character) => ({ value: character, label: `“${glyph(character)}”` }))}
          onChange={(value) => {
            setState({ probe: value });
            narrate(`Showing what follows ${value === " " ? "a space" : value}.`);
          }}
        />
        <div className="lra-probe" role="list" aria-label={`Next-character probabilities after ${glyph(probe)}, base model then base plus adapter`}>
          {probeRows.map((row) => (
            <div className="lra-probe__row" role="listitem" key={row.index}>
              <span className="lra-probe__char">{glyph(TINY_VOCAB[row.index])}</span>
              <div className="lra-probe__bars">
                <i className="lra-probe__bar is-base" style={{ width: `${row.before * 100}%` }} />
                <i className="lra-probe__bar is-adapted" style={{ width: `${row.after * 100}%` }} />
              </div>
              <span className="lra-probe__values">
                {(row.before * 100).toFixed(0)}% → <b>{(row.after * 100).toFixed(0)}%</b>
              </span>
            </div>
          ))}
        </div>
        <p className="lra-probe__key">
          <span><i className="lra-probe__swatch is-base" />base model (upper, hatched)</span>
          <span><i className="lra-probe__swatch is-adapted" />base + adapter (lower, solid)</span>
        </p>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Hot-swap and merge" className="tg-merge-panel">
          <SurfaceHeading kicker="Serving" title="Swap it out, or fold it in" />
          <div className="tg-swap">
            <SegmentedControl
              label="Serving"
              value={merged ? "merged" : adapterOn ? "adapted" : "base"}
              options={
                merged
                  ? [{ value: "merged", label: "Merged — nothing left to swap" }]
                  : [
                      { value: "base", label: "Base only" },
                      { value: "adapted", label: "Base + adapter" },
                    ]
              }
              onChange={(value) => {
                if (merged) return;
                setState({ adapterOn: value === "adapted" });
                narrate(value === "adapted" ? "Adapter attached." : "Adapter detached.");
              }}
            />
          </div>
          <div className="metric-row lra-metrics">
            <Metric label="Target loss" value={(active ? targetAdapted : targetBase).toFixed(3)} tone="forward" />
            <Metric
              label="Source loss"
              value={(active ? sourceAdapted : sourceBase).toFixed(3)}
              tone={active && sourceAdapted - sourceBase > 0.05 ? "loss" : undefined}
            />
            <Metric
              label="Serving weights"
              value={`${servingWeights}`}
              tone={servingWeights > lora.fullParameters ? "loss" : "forward"}
            />
          </div>
          <div className="tg-sample">
            <span>{active ? "With the adapter" : "Base model only"}, temperature 0.7</span>
            <p>{sampleTinyText(serving, { prompt: "the ", length: 76, temperature: 0.7, seed: 12 })}</p>
          </div>
          <button
            type="button"
            className="primary-action"
            disabled={merged}
            onClick={() => {
              setState({ merged: true, adapterOn: true });
              narrate(
                `Merged. The largest logit difference from the unmerged path is ${mergeCheck.maxLogitGap.toExponential(1)}.`,
              );
            }}
          >
            {merged ? "Adapter merged into W" : "Merge the adapter into W"}
          </button>
          <div className={`tg-callout ${merged ? "" : "is-quiet"}`.trim()}>
            <strong>
              {merged
                ? `Merged. Largest logit gap from the unmerged path: ${mergeCheck.maxLogitGap.toExponential(1)}.`
                : "Merging adds (α/r)·B·A into W once, before serving."}
            </strong>
            <span>
              {merged
                ? `The unmerged path computes W·x + (α/r)·B·(A·x) in float64; the merged table is one float32 matrix. They differ only by rounding, and recipe loss is ${mergeCheck.unmergedLoss.toFixed(4)} either way. There is no adapter left to detach.`
                : "Unmerged, the adapter is a separate file you can attach per request, at the cost of an extra multiply. Merged, it costs nothing at inference but cannot be removed."}
            </span>
          </div>
        </LabSurface>
      </div>

      <LabSurface label="Adapter size calculator" className="lra-calc-panel">
        <SurfaceHeading
          kicker={`${architecture.name} · d ${formatCount(architecture.d)} · ${architecture.layers} layers · closed-form arithmetic`}
          title="What the same idea costs on a real model"
          aside={<span className="tg-badge">{formatBillions(baseCount)} base parameters</span>}
        />
        <div className="lra-calc__controls">
          <SegmentedControl
            label="Architecture"
            value={calc.preset}
            options={[
              ...PRESETS.map((entry) => ({ value: entry.id, label: entry.name })),
              { value: CUSTOM, label: "Custom" },
            ]}
            onChange={(value) => {
              const named = PRESETS.find((entry) => entry.id === value);
              setCalculator(named ? { preset: named.id, d: named.d, layers: named.layers } : { preset: CUSTOM });
              narrate(named ? `${named.name}.` : "Custom architecture, using Llama 3's ratios.");
            }}
          />
          <fieldset className="segmented">
            <legend>Target matrices</legend>
            <div>
              {TARGETS.map((target) => (
                <button
                  type="button"
                  key={target.id}
                  aria-pressed={calc.targets.includes(target.id)}
                  title={target.detail}
                  onClick={() => toggleTarget(target.id)}
                >
                  {target.label}
                </button>
              ))}
            </div>
          </fieldset>
          <RangeControl
            label="Model dimension d"
            min={D_MIN}
            max={D_MAX}
            step={D_STEP}
            value={calc.d}
            format={(value) => formatCount(value)}
            onChange={(value) => setCalculator({ preset: CUSTOM, d: value })}
          />
          <RangeControl
            label="Layers"
            min={LAYERS_MIN}
            max={LAYERS_MAX}
            step={1}
            value={calc.layers}
            format={(value) => `${value}`}
            onChange={(value) => setCalculator({ preset: CUSTOM, layers: value })}
          />
          <RangeControl
            label="Adapter rank r"
            min={RANK_MIN}
            max={RANK_MAX}
            step={1}
            value={calc.rank}
            format={(value) => `r = ${value}`}
            onChange={(value) => {
              setCalculator({ rank: value });
              narrate(`Rank ${value}.`);
            }}
          />
          <SegmentedControl
            label="Adapter precision"
            value={`${calc.bits}`}
            options={BITS.map((bits) => ({ value: `${bits}`, label: `${bits}-bit` }))}
            onChange={(value) => setCalculator({ bits: Number(value) as (typeof BITS)[number] })}
          />
        </div>
        <div className="metric-row lra-calc__metrics">
          <Metric label="Trainable parameters" value={formatCount(counts.total)} tone="forward" />
          <Metric label="Share of the base model" value={formatShare(realShare)} tone="forward" />
          <Metric label={`Adapter size at ${calc.bits}-bit`} value={formatMegabytes(adapterBytes(counts.total, calc.bits))} />
        </div>
        <BarList
          label="Trainable parameters by targeted matrix, across every layer"
          items={counts.byTarget.map((entry) => ({
            id: entry.id,
            label: entry.label,
            value: entry.parameters,
            display: formatCount(entry.parameters),
            detail: `${formatShare(entry.parameters / Math.max(1, counts.total))} of the adapter`,
            tone: "forward" as const,
          }))}
          max={Math.max(1, counts.total)}
        />
        <FormulaWithValues
          label="Trainable parameters"
          expression="Σ r · (inputs + outputs), over targeted matrices and layers"
          result={calc.targets.length ? `${formatCount(counts.total)} in all` : "nothing targeted"}
          detail={
            calc.targets.length
              ? `B is outputs × r and A is r × inputs. Summed over the targeted matrices and ${architecture.layers} layers at r = ${calc.rank}, that is ${formatCount(counts.total)} numbers, against ${formatCount(baseCount)} in the base model.`
              : "Choose at least one matrix to adapt; with none, no parameter trains."
          }
          tone="gradient"
        />
        <div className="tg-callout">
          <strong>
            The toy at rank {rank} trains {formatShare(toyFraction)} of its table. This model, at the same rank and
            targets, trains {calc.targets.length ? formatShare(realAtToyRank) : "nothing"}.
          </strong>
          <span>
            {calc.targets.length
              ? `A pair costs r × (inputs + outputs) while the matrix it adapts holds inputs × outputs, so the share falls as the matrix grows: with 30 rows and columns it is large, with thousands it is a fraction of a percent. The toy's share is about ${formatCount(Math.round(toyFraction / realAtToyRank))} times larger.`
              : "With no matrix targeted there is nothing to compare."}
          </span>
        </div>
        <p className="lab-note">
          Nothing on this card is trained: every number is arithmetic on the architecture named above. A custom
          model keeps Llama 3&rsquo;s ratios (key and value width a quarter of d, feed-forward width 3.5 · d), and
          sizes count the weights only, not file metadata.
        </p>
      </LabSurface>
    </div>
  );
}
