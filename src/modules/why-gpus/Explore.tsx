import { useMemo } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { RooflinePlot } from "./RooflinePlot";
import {
  BANDWIDTH_MAX,
  BANDWIDTH_MIN,
  BATCH_MAX,
  BATCH_MIN,
  batchAtRidge,
  evaluate,
  formatCount,
  formatIntensity,
  formatSeconds,
  OPERATIONS,
  PRECISIONS,
  precisionById,
  whatIf,
  WIDTH_MAX,
  WIDTH_MIN,
  type OperationId,
  type Scenario,
} from "./roofline";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const powerOfTwo = (value: number, low: number, high: number) =>
  2 ** clamp(Math.round(Math.log2(Math.max(1, value))), Math.log2(low), Math.log2(high));

const speedup = (base: number, next: number) => {
  const ratio = base / next;
  if (Math.abs(ratio - 1) < 0.005) return "no change";
  return ratio > 1 ? `${ratio.toFixed(2)}× faster` : `${(1 / ratio).toFixed(2)}× slower`;
};

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const operation: OperationId = asString(state, "operation", "matmul") === "add" ? "add" : "matmul";
  const batch = powerOfTwo(asNumber(state, "batch", 1), BATCH_MIN, BATCH_MAX);
  const width = powerOfTwo(asNumber(state, "layerWidth", 4096), WIDTH_MIN, WIDTH_MAX);
  const precision = precisionById(asString(state, "precision", "fp16"));
  const bandwidth = clamp(Math.round(asNumber(state, "bandwidth", 2000)), BANDWIDTH_MIN, BANDWIDTH_MAX);

  const scenario = useMemo(
    () => ({ operation, batch, width, precision, bandwidth }),
    [bandwidth, batch, operation, precision, width],
  );
  const result = useMemo(() => evaluate(scenario), [scenario]);
  const options = useMemo(() => whatIf(scenario), [scenario]);
  const { work, time, ridge } = result;
  const ridgeRows = batchAtRidge(width, precision.bytes, ridge);
  const memoryBound = time.bound === "memory";
  const opUnit = precision.id === "int8" ? "OP" : "FLOP";

  /** Screen-reader summary of the workload after a control change, computed from the same roofline math. */
  const announce = (lead: string, change: Partial<Scenario>) => {
    const next = evaluate({ ...scenario, ...change });
    const unit = (change.precision ?? precision).id === "int8" ? "OP" : "FLOP";
    const used =
      next.time.utilization >= 0.9995 ? "100" : (next.time.utilization * 100).toFixed(next.time.utilization < 0.1 ? 2 : 1);
    narrate(
      `${lead} Now ${next.time.bound}-bound: intensity ${formatIntensity(next.work.intensity)} ${unit}/B against a ridge of ${formatIntensity(next.ridge)}, ${used}% of peak arithmetic in use.`,
    );
  };

  const expression =
    operation === "matmul"
      ? `ops = 2·B·d² = 2·${batch}·${width}²   bytes = (d² + 2·B·d)·${precision.bytes}`
      : `ops = B·d = ${batch}·${width}   bytes = 3·B·d·${precision.bytes}`;

  return (
    <div className="tg-lab wg-lab">
      <LabSurface label="Workload knobs" className="wg-knobs-card">
        <SurfaceHeading
          kicker={`${operation === "matmul" ? `X ${batch}×${width} · W ${width}×${width}` : `X + R, ${batch}×${width}`} · ${precision.label} · ${bandwidth} GB/s`}
          title="One layer operation on an illustrative GPU"
        />
        <div className="wg-knob-grid">
          <div className="wg-operation-control">
            <SegmentedControl
              label="Operation"
              value={operation}
              options={OPERATIONS.map((item) => ({ value: item.id, label: item.label }))}
              onChange={(value) => {
                const next: OperationId = value === "add" ? "add" : "matmul";
                setState({ operation: next });
                announce(`Operation ${next === "add" ? "elementwise add" : "matmul"}.`, { operation: next });
              }}
            />
          </div>
          <div className="precision-control">
            <SegmentedControl
              label="Precision"
              value={precision.id}
              options={PRECISIONS.map((item) => ({ value: item.id, label: item.label }))}
              onChange={(value) => {
                const next = precisionById(value);
                setState({ precision: next.id });
                announce(`Precision ${next.label}, ${next.bytes} byte${next.bytes === 1 ? "" : "s"} per value.`, { precision: next });
              }}
            />
          </div>
          <RangeControl
            label="Batch rows"
            min={0}
            max={Math.log2(BATCH_MAX)}
            step={1}
            value={Math.log2(batch)}
            format={(value) => `${2 ** value} row${value === 0 ? "" : "s"}`}
            onChange={(value) => {
              setState({ batch: 2 ** value });
              announce(`Batch ${2 ** value} row${value === 0 ? "" : "s"}.`, { batch: 2 ** value });
            }}
          />
          <RangeControl
            label="Layer width"
            min={Math.log2(WIDTH_MIN)}
            max={Math.log2(WIDTH_MAX)}
            step={1}
            value={Math.log2(width)}
            format={(value) => `d = ${2 ** value}`}
            onChange={(value) => {
              setState({ layerWidth: 2 ** value });
              announce(`Layer width ${2 ** value}.`, { width: 2 ** value });
            }}
          />
          <RangeControl
            label="Memory bandwidth"
            min={BANDWIDTH_MIN}
            max={BANDWIDTH_MAX}
            step={100}
            value={bandwidth}
            format={(value) => `${value.toFixed(0)} GB/s`}
            onChange={(value) => {
              setState({ bandwidth: value });
              announce(`Memory bandwidth ${value.toFixed(0)} gigabytes per second.`, { bandwidth: value });
            }}
          />
        </div>
        <FormulaWithValues
          label="Work versus traffic"
          expression={expression}
          result={`${formatCount(work.flops, opUnit)} · ${formatCount(work.bytes, "B")}`}
          detail={
            operation === "matmul"
              ? `Each weight is fetched once and used by all ${batch} row${batch === 1 ? "" : "s"}, so intensity ≈ 2·B / bytes when B ≪ d.`
              : "Each output needs one add and three memory touches (two reads, one write). No reuse, at any batch."
          }
          tone={memoryBound ? "loss" : "forward"}
        />
        <p className="lab-note wg-device-note">
          Illustrative device, round numbers: peak {PRECISIONS.map((item) => `${item.label} ${item.peakTflops} ${item.unit}`).join(", ")}.
          Bandwidth is yours to set. This is arithmetic on shapes, not a kernel run on a GPU.
        </p>
      </LabSurface>

      <LabSurface label="Roofline" className="wg-roofline-card">
        <SurfaceHeading
          kicker={memoryBound ? "Left of the ridge: bandwidth sets the ceiling" : "Right of the ridge: peak arithmetic sets the ceiling"}
          title="How fast can this operation possibly run?"
        />
        <RooflinePlot
          peakTflops={precision.peakTflops}
          bandwidth={bandwidth}
          bytesPerElement={precision.bytes}
          operation={operation}
          batch={batch}
          width={width}
          precisionLabel={precision.label}
          unit={precision.unit}
        />
        <div className="metric-row">
          <Metric label="Intensity" value={`${formatIntensity(work.intensity)} ${opUnit}/B`} />
          <Metric label="Ridge" value={`${formatIntensity(ridge)} ${opUnit}/B`} />
          <Metric label="Attainable" value={`${(time.attainableFlops / 1e12).toPrecision(3)} ${precision.unit}`} tone={memoryBound ? "loss" : "forward"} />
          <Metric label="Bound" value={time.bound} tone={memoryBound ? "loss" : "forward"} />
        </div>
      </LabSurface>

      <LabSurface label="Where the time goes" className="wg-time-card">
        <SurfaceHeading
          kicker={
            time.utilization >= 0.9995
              ? "100% of peak arithmetic in use (the ideal roofline)"
              : `${(time.utilization * 100).toFixed(time.utilization < 0.1 ? 2 : 1)}% of peak arithmetic in use`
          }
          title="Two clocks; the slower one is the wall-clock"
        />
        <BarList
          label="Arithmetic time versus memory time"
          items={[
            {
              id: "compute",
              label: "Arithmetic at peak",
              value: time.computeSeconds,
              display: formatSeconds(time.computeSeconds),
              detail: `${formatCount(work.flops, opUnit)} ÷ ${precision.peakTflops} ${precision.unit}${memoryBound ? "" : " · sets the time"}`,
              tone: "forward",
              emphasis: !memoryBound,
            },
            {
              id: "memory",
              label: "Memory traffic",
              value: time.memorySeconds,
              display: formatSeconds(time.memorySeconds),
              detail: `${formatCount(work.bytes, "B")} ÷ ${bandwidth} GB/s${memoryBound ? " · sets the time" : ""}`,
              tone: "loss",
              emphasis: memoryBound,
            },
          ]}
        />
        <div className="wg-whatif" role="table" aria-label="Wall-clock after one change">
          <div role="row" className="wg-whatif__head">
            <span role="columnheader">Change one thing</span>
            <span role="columnheader">Wall-clock</span>
            <span role="columnheader">Effect</span>
          </div>
          <div role="row">
            <span role="cell">Nothing (current)</span>
            <span role="cell">{formatSeconds(options.base)}</span>
            <span role="cell">—</span>
          </div>
          <div role="row">
            <span role="cell">2× peak arithmetic</span>
            <span role="cell">{formatSeconds(options.doublePeak)}</span>
            <span role="cell">{speedup(options.base, options.doublePeak)}</span>
          </div>
          <div role="row">
            <span role="cell">2× memory bandwidth</span>
            <span role="cell">{formatSeconds(options.doubleBandwidth)}</span>
            <span role="cell">{speedup(options.base, options.doubleBandwidth)}</span>
          </div>
          {options.smallerPrecision !== undefined && (
            <div role="row">
              <span role="cell">{precision.label} → {options.smallerLabel}</span>
              <span role="cell">{formatSeconds(options.smallerPrecision)}</span>
              <span role="cell">{speedup(options.base, options.smallerPrecision)}</span>
            </div>
          )}
        </div>
        <div className="metric-row">
          <Metric
            label="Parallel outputs"
            value={operation === "matmul" ? `${work.outputs.toLocaleString("en-US")} dots of ${work.dotLength}` : `${work.outputs.toLocaleString("en-US")} adds`}
          />
          <Metric
            label="Rows to reach ridge"
            value={operation === "add" ? "never" : Number.isFinite(ridgeRows) ? `about ${Math.ceil(ridgeRows)}` : "never at this d"}
          />
        </div>
        <p className="lab-note">
          {operation === "add"
            ? "An elementwise add never reuses a byte, so it stays far left of the ridge at every batch. Real frameworks fuse such ops into the matmul that precedes them so the tensor is not written and re-read."
            : memoryBound
              ? `Plenty of parallel work, yet the chip idles: every weight byte crosses the memory bus to be used by only ${batch} row${batch === 1 ? "" : "s"}. More rows per weight fetch, or fewer bytes per weight, move the dot up the slanted roof. Extra peak arithmetic does not.`
              : "Each weight fetch now feeds enough rows that arithmetic is the limit. This is where training batches and long prompts sit, and where a GPU's thousands of multiply units pay off."}
        </p>
      </LabSurface>
    </div>
  );
}
