import { useMemo } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyPerplexity,
  trainTinyModel,
  type ModuleContext,
} from "@app/module-sdk";
import { FrontierPlot, GridAndErrorPlot, type FrontierSeries } from "./QuantCharts";
import {
  bitsPerWeightOf,
  codesUsed,
  DOWNLOADED_FILE,
  errorStats,
  fileBitsPerWeight,
  fileEffectiveBitsPerWeight,
  fileSlices,
  FLOAT_FORMATS,
  floatFormatStats,
  GGUF_BLOCK_FORMATS,
  quantizeWeights,
  roundTableToFloatFormat,
  roundToFloatFormat,
  type QuantScope,
  type ZeroPoint,
} from "./quant";

const corpus = TINY_CORPORA.harbor;
const PRETRAIN_EPOCHS = 80;
const SWEEP_BITS = [2, 3, 4, 5, 6, 8];
const MAX_PLOTTED_BPW = 17;
const SCOPES: QuantScope[] = ["tensor", "row", "group"];

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const glyph = (id: number) => (TINY_VOCAB[id] === " " ? "␣" : TINY_VOCAB[id]);
const percent = (value: number) => {
  const rounded = Math.abs(value) < 0.005 ? 0 : value;
  return `${rounded >= 0 ? "+" : "−"}${Math.abs(rounded).toFixed(2)}%`;
};

const GIB = 1024 ** 3;
const FLOAT_EXAMPLES = [0.001, 0.02, 300, 100000];
const FLOAT_MIN = 1e-6;
const FLOAT_MAX = 1e6;
const clampFloatValue = (value: number) => Math.min(FLOAT_MAX, Math.max(FLOAT_MIN, value));
/** Five significant digits in a plain number, exponent form for the very small, so no digit string runs past five decimals. */
const formatFloat = (value: number) => {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude >= 0.1 && magnitude < 1e5 ? String(Number(value.toPrecision(5))) : value.toExponential(4);
};
const withCommas = (value: number) => value.toLocaleString("en-US");
/** The value a learner typed or slid to: plain digits where that is short, a power of ten below a thousandth. */
const formatInput = (value: number) =>
  value >= 1e-3 && value < 1e5 ? withCommas(Number(value.toPrecision(4))) : value >= 1e5 ? withCommas(Math.round(value)) : value.toExponential(3);
/** A signed percent error to two significant digits, so 0.0214 reads 0.021 and 2.34 reads 2.3. */
const signedPrecise = (percentValue: number) => {
  if (percentValue === 0) return "0%";
  const magnitude = Math.abs(percentValue);
  return `${percentValue > 0 ? "+" : "−"}${magnitude >= 0.01 ? magnitude.toPrecision(2) : magnitude.toExponential(1)}%`;
};
/** A fraction as a percent that never prints more than five decimals. */
const relativePercent = (fraction: number) => {
  const value = fraction * 100;
  return `${value >= 0.01 ? value.toPrecision(3) : value.toExponential(1)}%`;
};

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const bits = clamp(Math.round(asNumber(state, "bits", 4)), 2, 8);
  const scopeRaw = asString(state, "scope", "group");
  const scope: QuantScope = (SCOPES as string[]).includes(scopeRaw) ? (scopeRaw as QuantScope) : "group";
  const groupSize = clamp(Math.round(asNumber(state, "groupSize", 8) / 2) * 2, 2, 30);
  const zeroPoint: ZeroPoint = asString(state, "zeroPoint", "asymmetric") === "symmetric" ? "symmetric" : "asymmetric";
  const fileGroup: "type" | "role" = asString(state, "fileGroup", "type") === "role" ? "role" : "type";
  const floatValue = clampFloatValue(asNumber(state, "floatValue", 0.02));

  const base = useMemo(() => trainTinyModel({ text: corpus.text, epochs: PRETRAIN_EPOCHS, seed: 1 }), []);
  const basePerplexity = useMemo(() => tinyPerplexity(base.weights, corpus.text), [base.weights]);
  const outlier = useMemo(() => {
    let index = 0;
    for (let k = 1; k < base.weights.length; k += 1) {
      if (Math.abs(base.weights[k]) > Math.abs(base.weights[index])) index = k;
    }
    const row = Math.floor(index / TINY_VOCAB_SIZE);
    const column = index % TINY_VOCAB_SIZE;
    return { index, value: base.weights[index], label: `“${glyph(row)}” → “${glyph(column)}”` };
  }, [base.weights]);

  const quantized = useMemo(
    () => quantizeWeights(base.weights, { bits, scope, groupSize, zeroPoint, rowLength: TINY_VOCAB_SIZE }),
    [base.weights, bits, groupSize, scope, zeroPoint],
  );
  const quantizedPerplexity = useMemo(() => tinyPerplexity(quantized.weights, corpus.text), [quantized.weights]);

  const outlierBlock = quantized.blocks.findIndex((block) => outlier.index >= block.start && outlier.index < block.end);
  const inspectRaw = Math.round(asNumber(state, "inspect", -1));
  const inspect = inspectRaw < 0 ? outlierBlock : clamp(inspectRaw, 0, quantized.blocks.length - 1);
  const block = quantized.blocks[inspect];
  const used = codesUsed(quantized, block);

  const cost = (perplexity: number) => (perplexity / basePerplexity - 1) * 100;

  const frontier = useMemo<FrontierSeries[]>(
    () =>
      SCOPES.map((candidate) => ({
        id: candidate,
        name: candidate === "tensor" ? "per tensor" : candidate === "row" ? "per row" : `per group of ${groupSize}`,
        shape: candidate === "tensor" ? "circle" : candidate === "row" ? "square" : "triangle",
        dash: candidate === "tensor" ? "solid" : candidate === "row" ? "dashed" : "dotted",
        points: SWEEP_BITS.map((sweepBits) => {
          const result = quantizeWeights(base.weights, {
            bits: sweepBits,
            scope: candidate,
            groupSize,
            zeroPoint,
            rowLength: TINY_VOCAB_SIZE,
          });
          return {
            bits: sweepBits,
            bpw: result.bitsPerWeight,
            cost: (tinyPerplexity(result.weights, corpus.text) / basePerplexity - 1) * 100,
          };
        }),
      })),
    [base.weights, basePerplexity, groupSize, zeroPoint],
  );

  const formats = useMemo(
    () =>
      GGUF_BLOCK_FORMATS.map((format) => {
        const rounded = format.quantize(base.weights);
        const perplexity = tinyPerplexity(rounded, corpus.text);
        return { ...format, perplexity, rmsError: errorStats(base.weights, rounded).rmsError };
      }),
    [base.weights],
  );

  const samples = useMemo(
    () => ({
      base: sampleTinyText(base.weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 5 }),
      quantized: sampleTinyText(quantized.weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 5 }),
    }),
    [base.weights, quantized.weights],
  );

  const slices = useMemo(() => fileSlices(fileGroup), [fileGroup]);
  const effective = fileEffectiveBitsPerWeight();
  const embedding = DOWNLOADED_FILE.groups.find((group) => group.role === "embedding") ?? DOWNLOADED_FILE.groups[0];
  const floatTable = useMemo(
    () =>
      FLOAT_FORMATS.map((format) => {
        const rounded = roundTableToFloatFormat(base.weights, format);
        const perplexity = tinyPerplexity(rounded, corpus.text);
        return {
          format,
          stats: floatFormatStats(format),
          rmsError: errorStats(base.weights, rounded).rmsError,
          cost: (perplexity / basePerplexity - 1) * 100,
        };
      }),
    [base.weights, basePerplexity],
  );

  const blockSize = block.end - block.start;
  const scopeDetail =
    scope === "tensor"
      ? `one grid for all ${base.weights.length} weights`
      : scope === "row"
        ? `one grid per context row, ${quantized.blocks.length} of them`
        : `one grid per ${groupSize} weights, ${quantized.blocks.length} of them`;
  const metadata = zeroPoint === "symmetric" ? "one fp16 scale" : "an fp16 scale and an fp16 offset";
  const blockLabel = (index: number) => {
    const target = quantized.blocks[index];
    const first = Math.floor(target.start / TINY_VOCAB_SIZE);
    const last = Math.floor((target.end - 1) / TINY_VOCAB_SIZE);
    const rows = first === last ? `row “${glyph(first)}”` : `rows “${glyph(first)}”…“${glyph(last)}”`;
    return `block ${index + 1} of ${quantized.blocks.length} · ${rows}${index === outlierBlock ? " · holds the largest weight" : ""}`;
  };

  return (
    <div className="tg-lab tg-lab--hero qz-lab">
      <LabSurface label="Quantization controls" className="tg-quant-panel">
        <SurfaceHeading kicker="Precision, granularity, range" title="Three dials, one storage bill" />
        <div className="qz-controls">
          <div className="qz-controls__column">
            <RangeControl
              label="Bit width"
              min={2}
              max={8}
              step={1}
              value={bits}
              format={(value) => `${value} bits · ${zeroPoint === "symmetric" ? 2 ** value - 1 : 2 ** value} levels`}
              onChange={(value) => {
                setState({ bits: value });
                narrate(`${value} bits per weight code.`);
              }}
            />
            {scope === "group" && (
              <RangeControl
                label="Group size"
                min={2}
                max={30}
                step={2}
                value={groupSize}
                format={(value) => `${value} weights`}
                onChange={(value) => setState({ groupSize: value, inspect: -1 })}
              />
            )}
          </div>
          <div className="qz-controls__column">
            <div className="tg-scope">
              <SegmentedControl
                label="Scaling granularity"
                value={scope}
                options={[
                  { value: "tensor", label: "Per tensor" },
                  { value: "row", label: "Per row" },
                  { value: "group", label: "Per group" },
                ]}
                onChange={(value) => {
                  setState({ scope: value, inspect: -1 });
                  narrate(
                    `Scaling ${value === "tensor" ? "once for the whole tensor" : value === "row" ? "per context row" : `per ${groupSize} weights`}.`,
                  );
                }}
              />
            </div>
            <div className="qz-range-mapping">
              <SegmentedControl
                label="Range mapping"
                value={zeroPoint}
                options={[
                  { value: "asymmetric", label: "Asymmetric · min/max" },
                  { value: "symmetric", label: "Symmetric · absmax" },
                ]}
                onChange={(value) => {
                  setState({ zeroPoint: value });
                  narrate(
                    value === "symmetric"
                      ? "Symmetric: one scale per block, zero is always a level."
                      : "Asymmetric: a scale and an offset per block, the grid spans the block's own range.",
                  );
                }}
              />
            </div>
          </div>
        </div>
        <div className="metric-row">
          <Metric label="Bits per weight" value={quantized.bitsPerWeight.toFixed(3)} />
          <Metric label="Model size" value={`${quantized.totalBytes} B`} tone="forward" />
          <Metric
            label="Against 32-bit"
            value={`${((quantized.totalBytes / (base.weights.length * 4)) * 100).toFixed(1)}%`}
          />
        </div>
        <p className="lab-note">
          Each block stores {metadata} ({quantized.metadataBitsPerBlock} bits), so {quantized.blocks.length} block
          {quantized.blocks.length === 1 ? " adds" : "s add"} {(quantized.bitsPerWeight - bits).toFixed(3)} bits to every
          weight's {bits}-bit code. At a group of 2 the scales cost at least as much as the codes.
        </p>
      </LabSurface>

      <LabSurface label="Rounding grid and per-weight error" className="qz-grid-card">
        <SurfaceHeading
          kicker={`${bits}-bit · ${zeroPoint} · ${scopeDetail}`}
          title="Every weight snaps to the nearest level of its block's grid"
          aside={<span className="tg-badge">rms error {quantized.rmsError.toFixed(4)}</span>}
        />
        <GridAndErrorPlot
          source={base.weights}
          result={quantized}
          block={block}
          blockIndex={inspect}
          outlier={outlier}
        />
        {quantized.blocks.length > 1 && (
          <div className="qz-inspect">
            <RangeControl
              label="Inspect block"
              min={1}
              max={quantized.blocks.length}
              step={1}
              value={inspect + 1}
              format={(value) => blockLabel(value - 1)}
              onChange={(value) => setState({ inspect: value - 1 })}
            />
          </div>
        )}
        <div className="metric-row">
          <Metric label="RMS rounding error" value={quantized.rmsError.toFixed(4)} tone="gradient" />
          <Metric label="Largest error" value={quantized.maxError.toFixed(4)} tone="gradient" />
          <Metric label="Step in this block" value={block.scale.toFixed(4)} />
          <Metric
            label="Levels used in this block"
            value={`${used} / ${quantized.levels}`}
            tone={used < quantized.levels * 0.7 ? "loss" : undefined}
          />
        </div>
        <p className="lab-note">
          {zeroPoint === "symmetric"
            ? `Symmetric: scale = max|w| / ${2 ** (bits - 1) - 1}, codes −${2 ** (bits - 1) - 1}…${2 ** (bits - 1) - 1}, and 0 is always a level. This table is lopsided (−1.71 to +5.93), so the negative half of the grid is mostly empty — dashed lines are levels no weight in the block landed on.`
            : `Asymmetric: scale = (max − min) / ${2 ** bits - 1} and code 0 decodes to the block minimum, so the grid covers exactly the block's range. The block containing ${outlier.label} is stretched by that one weight; every other weight in it shares the wide steps.`}{" "}
          {blockSize} weights share this grid.
        </p>
      </LabSurface>

      <LabSurface label="Quality against bits per weight" className="qz-frontier-card">
        <SurfaceHeading
          kicker={`Measured perplexity, ${zeroPoint} grids, same trained weights`}
          title="What each bit of storage buys back"
          aside={<span className="tg-badge">{quantized.bitsPerWeight.toFixed(2)} bits/weight</span>}
        />
        <FrontierPlot
          series={frontier}
          formats={formats
            .filter((format) => format.name !== "F16")
            .map((format) => ({ name: format.name, bpw: format.bitsPerWeight, cost: cost(format.perplexity) }))}
          current={{ bpw: quantized.bitsPerWeight, cost: cost(quantizedPerplexity) }}
          maxBpw={MAX_PLOTTED_BPW}
        />
        <div className="metric-row">
          <Metric label="Full precision" value={basePerplexity.toFixed(3)} />
          <Metric label="Quantized" value={quantizedPerplexity.toFixed(3)} tone="loss" />
          <Metric
            label="Quality cost"
            value={percent(cost(quantizedPerplexity))}
            tone={cost(quantizedPerplexity) > 2 ? "loss" : "forward"}
          />
        </div>
        <div className="tg-diff">
          <div className="tg-sample">
            <span>Full precision, temperature 0.7</span>
            <p>{samples.base}</p>
          </div>
          <div className="tg-sample">
            <span>
              {bits}-bit {scope === "tensor" ? "per tensor" : scope === "row" ? "per row" : `groups of ${groupSize}`},
              temperature 0.7
            </span>
            <p>{samples.quantized}</p>
          </div>
        </div>
      </LabSurface>

      <LabSurface label="Published quantization formats" className="qz-formats-card">
        <SurfaceHeading
          kicker="llama.cpp block layouts, run on this table"
          title="What the letters in a GGUF name stand for"
        />
        <div className="qz-table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Block type</th>
                <th scope="col">Bits per weight</th>
                <th scope="col">Toy perplexity</th>
                <th scope="col">RMS error</th>
                <th scope="col">llama.cpp file, Llama-3-8B</th>
              </tr>
            </thead>
            <tbody>
              {formats.map((format) => (
                <tr key={format.name} className={format.codeBits === bits ? "is-current" : ""}>
                  <th scope="row">
                    {format.name}
                    <small>{format.layout}</small>
                    <small>{format.note}</small>
                  </th>
                  <td>
                    <span className="qz-bpw">
                      <i style={{ width: `${(format.bitsPerWeight / 16) * 100}%` }} aria-hidden="true" />
                      {String(format.bitsPerWeight)}
                    </span>
                  </td>
                  <td>
                    {format.perplexity.toFixed(3)}
                    <small>{percent(cost(format.perplexity))}</small>
                  </td>
                  <td>{format.rmsError.toFixed(4)}</td>
                  <td className="qz-files">{format.files}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          A name like Q4_K_M is a file recipe, not a block: mostly Q4_K blocks, with some tensors (parts of the
          attention value and feed-forward down projections, and the output layer) kept in Q6_K. That is why llama.cpp's
          own Q4_K_M file for Llama-3-8B is 4.58 GiB, about {fileBitsPerWeight(4.58).toFixed(2)} bits per weight, not
          4.5. The toy column quantizes this 900-weight table in storage order with each block layout; it is one tensor,
          so the _S / _M / _L mixes cannot be shown here.
        </p>
      </LabSurface>

      <LabSurface label="The file you downloaded" className="qz-formats-card qz-file-card">
        <SurfaceHeading
          kicker={`${DOWNLOADED_FILE.name} · ${DOWNLOADED_FILE.repository}, read ${DOWNLOADED_FILE.checked}`}
          title="What a 4-bit file costs per weight"
          aside={<span className="tg-badge">{effective.toFixed(3)} bits/weight</span>}
        />
        <div className="metric-row">
          <Metric label="File size" value={`${(DOWNLOADED_FILE.bytes / GIB).toFixed(2)} GiB`} tone="forward" />
          <Metric label="Parameters in the file" value={`${(DOWNLOADED_FILE.parameters / 1e9).toFixed(2)} B`} />
          <Metric label="Effective bits per weight" value={effective.toFixed(3)} tone="loss" />
          <Metric label="Share of the F16 size" value={`${((effective / 16) * 100).toFixed(1)}%`} />
        </div>
        <div className="qz-inspect">
          <SegmentedControl
            label="Group tensors by"
            value={fileGroup}
            options={[
              { value: "type", label: "Block type" },
              { value: "role", label: "Role in the network" },
            ]}
            onChange={(value) => {
              setState({ fileGroup: value });
              narrate(value === "type" ? "Tensors grouped by block type." : "Tensors grouped by role in the network.");
            }}
          />
        </div>
        <BarList
          label="Share of the file's tensor bytes by group, with each group's bits per weight"
          items={slices.map((slice) => ({
            id: slice.key,
            label: slice.label,
            value: slice.byteShare,
            display: `${(slice.byteShare * 100).toFixed(1)}% of bytes`,
            tone: slice.bitsPerWeight > 5 ? "loss" : "forward",
            detail: `${slice.tensors} tensor${slice.tensors === 1 ? "" : "s"} · ${(slice.parameterShare * 100).toFixed(1)}% of parameters · ${slice.bitsPerWeight.toFixed(2)} bits per weight`,
          }))}
          max={Math.max(...slices.map((slice) => slice.byteShare))}
        />
        <p className="lab-note">
          {withCommas(DOWNLOADED_FILE.bytes)} bytes divided over {withCommas(DOWNLOADED_FILE.parameters)} parameters is{" "}
          {effective.toFixed(3)} bits per weight: {(effective - 4.5).toFixed(3)} above the 4.5 of a Q4_K block, and{" "}
          {(((DOWNLOADED_FILE.bytes * 8) / (DOWNLOADED_FILE.parameters * 4.5) - 1) * 100).toFixed(1)}% more bytes than a file
          that stored every tensor as Q4_K. The {withCommas(DOWNLOADED_FILE.vocabulary)}-token embedding table holds{" "}
          {((embedding.parameters / DOWNLOADED_FILE.parameters) * 100).toFixed(1)}% of the parameters and the file keeps it at{" "}
          {embedding.type}, {bitsPerWeightOf(embedding.bytes, embedding.parameters).toFixed(2)} bits per weight. The header, with metadata, the
          vocabulary, and the chat template, adds {withCommas(DOWNLOADED_FILE.headerBytes)} bytes, or{" "}
          {bitsPerWeightOf(DOWNLOADED_FILE.headerBytes, DOWNLOADED_FILE.parameters).toFixed(3)} bits per weight. The file's
          metadata also records an importance matrix computed on {DOWNLOADED_FILE.calibration.dataset} ({DOWNLOADED_FILE.calibration.chunks} chunks,{" "}
          {DOWNLOADED_FILE.calibration.entries} entries), so this one is a calibrated quantization.
        </p>
      </LabSurface>

      <LabSurface label="Float formats" className="qz-formats-card qz-float-card">
        <SurfaceHeading
          kicker="Sign, exponent, mantissa: the range and precision of each, computed from the bit layout"
          title="The float ladder from fp32 down to fp8"
          aside={<span className="tg-badge">{formatInput(floatValue)}</span>}
        />
        <div className="qz-controls">
          <SegmentedControl
            label="Example value"
            value={String(floatValue)}
            options={FLOAT_EXAMPLES.map((example) => ({ value: String(example), label: withCommas(example) }))}
            onChange={(value) => {
              setState({ floatValue: Number(value) });
              narrate(`Storing ${value} in each float format.`);
            }}
          />
          <RangeControl
            label="Value to store"
            min={-120}
            max={120}
            step={1}
            value={Math.round(20 * Math.log10(floatValue))}
            format={() => formatInput(floatValue)}
            onChange={(value) => setState({ floatValue: clampFloatValue(10 ** (value / 20)) })}
          />
        </div>
        <div className="qz-table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Format</th>
                <th scope="col">Smallest normal to largest</th>
                <th scope="col">Spacing at 1.0</th>
                <th scope="col">{formatInput(floatValue)} is stored as</th>
                <th scope="col">Error</th>
                <th scope="col">Toy table RMS error</th>
                <th scope="col">Toy perplexity</th>
              </tr>
            </thead>
            <tbody>
              {floatTable.map(({ format, stats, rmsError, cost: toyCost }) => {
                const stored = roundToFloatFormat(floatValue, format);
                return (
                  <tr key={format.id}>
                    <th scope="row">
                      {format.name}
                      <small>
                        1 + {format.exponentBits} + {format.mantissaBits} bits, bias {format.bias}
                      </small>
                    </th>
                    <td>
                      {formatFloat(stats.minNormal)} to {formatFloat(stats.maxFinite)}
                      <small>{withCommas(stats.finiteValues)} finite values</small>
                    </td>
                    <td>
                      2^−{format.mantissaBits}
                      <small>
                        {formatFloat(stats.spacingAtOne)}, at most {relativePercent(stats.maxRelativeError)} off
                      </small>
                    </td>
                    <td>
                      {stored.overflow ? "overflow" : stored.value === 0 ? "0" : formatFloat(stored.value)}
                      <small>
                        {stored.overflow
                          ? `above ${formatFloat(stats.maxFinite)}`
                          : stored.value === 0
                            ? "underflows to zero"
                            : stored.subnormal
                              ? "subnormal: spacing stops shrinking"
                              : "normal range"}
                      </small>
                    </td>
                    <td>
                      {stored.overflow || stored.value === 0 ? "—" : signedPrecise(((stored.value - floatValue) / floatValue) * 100)}
                    </td>
                    <td>{rmsError.toFixed(4)}</td>
                    <td>
                      {percent(toyCost)}
                      <small>against {basePerplexity.toFixed(3)}</small>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Range comes from the exponent bits and precision from the mantissa bits: bf16 keeps fp32's eight exponent bits
          and only seven mantissa bits, so it reaches {formatFloat(floatFormatStats(FLOAT_FORMATS[2]).maxFinite)} but
          is eight times coarser than fp16 near 1.0. The toy columns round all 900 trained weights through each format
          with no per-tensor scale; the table's weights sit between −1.71 and +5.93, inside every format's range. Real
          fp8 pipelines scale each tensor first so its largest value lands near the format's maximum. Numbers below 0.1
          are written as powers of ten so every stored digit shows.
        </p>
      </LabSurface>
    </div>
  );
}
