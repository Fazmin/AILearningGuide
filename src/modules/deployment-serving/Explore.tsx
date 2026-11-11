import { useMemo } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  TINY_VOCAB_SIZE,
  type ModuleContext,
} from "@app/module-sdk";
import {
  activePreset,
  formatsFor,
  HARDWARE_PRESETS,
  hardwareFrom,
  KV_TYPES,
  LIMITS,
  leversFrom,
  NO_LEVERS,
  normalizeConfig,
  OPERATING_MODELS,
  operatingPoint,
  PAGE_TOKENS,
  type Levers,
  type OperatingModel,
  type OperatingPoint,
} from "./operate";
import { BLOCK_TYPE, formatBytes, HARDWARE } from "./serving";
import { DecodeTimeline, LeverTable, MemoryBar, ThroughputPlot, timelineSpan } from "./visuals";

const MAX_USERS = LIMITS.users.max;
const TOY_WEIGHTS = TINY_VOCAB_SIZE * TINY_VOCAB_SIZE;
/** The cards that existed before the hardware selector keep the lab's original, stated hardware. */
const LAB_HARDWARE = hardwareFrom(HARDWARE_PRESETS[0]);

const tokens = (value: number) => (value >= 100 ? value.toFixed(0) : value.toFixed(1));
const ms = (seconds: number) => (seconds >= 10 ? `${seconds.toFixed(1)} s` : `${(seconds * 1000).toFixed(0)} ms`);
const dollars = (value: number) => `$${value >= 0.1 ? value.toFixed(2) : value.toFixed(3)}`;
const costLabel = (point: OperatingPoint) =>
  !point.fits ? "over memory" : Number.isFinite(point.costPerMillionTokens) ? `${dollars(point.costPerMillionTokens)} per M tokens` : "no output";

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const config = normalizeConfig(state);
  const { users, context, batching } = config;

  const model: OperatingModel = OPERATING_MODELS.find((entry) => entry.id === config.modelId) ?? OPERATING_MODELS[2];
  const formats = formatsFor(model);
  const format = formats.find((entry) => entry.name === config.formatName) ?? formats.find((entry) => entry.name === "Q4_K_M") ?? formats[0];
  const hybrid = model.stateBytesPerStream > 0;
  const serving = useMemo(
    () => operatingPoint(model, format, users, context, batching, LAB_HARDWARE),
    [batching, context, format, model, users],
  );
  const sweep = useMemo(
    () =>
      Array.from({ length: MAX_USERS }, (_, index) => {
        const point = operatingPoint(model, format, index + 1, context, batching, LAB_HARDWARE);
        return { users: index + 1, total: point.totalTokensPerSecond, perUser: point.perUserTokensPerSecond };
      }),
    [batching, context, format, model],
  );

  const block = BLOCK_TYPE[format.name] ?? BLOCK_TYPE.Q4_K_M;
  const toyBytes = Math.ceil((TOY_WEIGHTS * block.bitsPerWeight) / 8);
  const headroom = HARDWARE.vramBytes - serving.totalBytes;
  // With one stream the total is one token per step, so its inverse is the single-stream step time.
  const soloStep = 1 / sweep[0].total;
  const span = timelineSpan(soloStep);

  // The hardware, lever, and cost card: same model, format, load, and scheduler, other hardware and levers.
  const hardware = hardwareFrom(config, config.utilization);
  const preset = activePreset(config);
  const levers = leversFrom(config);
  const kvType = KV_TYPES.find((entry) => entry.id === config.kvType) ?? KV_TYPES[0];
  const leverRows: { id: string; label: string; point: OperatingPoint; levers: Levers }[] = [
    { id: "none", label: "No levers", levers: NO_LEVERS },
    { id: "paged", label: "Paged attention alone", levers: { ...NO_LEVERS, paged: true, pagedFill: config.pagedFill } },
    {
      id: "kv",
      label: `${(config.kvType === "f16" ? KV_TYPES[1] : kvType).name} KV cache alone`,
      levers: { ...NO_LEVERS, kvBits: (config.kvType === "f16" ? KV_TYPES[1] : kvType).bitsPerValue },
    },
    {
      id: "spec",
      label: "Speculative decoding alone",
      levers: {
        ...NO_LEVERS,
        speculative: true,
        acceptance: config.acceptance,
        draftLength: config.draftLength,
        draftCost: config.draftCost,
      },
    },
    { id: "selected", label: config.paged || config.speculative || config.kvType !== "f16" ? "Your selection" : "Your selection (none on)", levers },
  ].map((row) => ({ ...row, point: operatingPoint(model, format, users, context, batching, hardware, row.levers) }));
  const baseline = leverRows[0].point;
  const chosen = leverRows[leverRows.length - 1].point;

  return (
    <div className="tg-lab ds-lab">
      <LabSurface label="Serving controls" className="tg-serving-controls">
        <SurfaceHeading kicker="Load" title="Model, format, and load" />
        <div className="ds-controls">
          <SegmentedControl
            label="Model"
            value={model.id}
            options={OPERATING_MODELS.map((entry) => ({ value: entry.id, label: entry.name }))}
            onChange={(value) => setState({ modelId: value })}
          />
          <div className="ds-format-control">
            <SegmentedControl
              label="Weight format"
              value={format.name}
              options={formats.map((entry) => ({ value: entry.name, label: entry.name }))}
              onChange={(value) => {
                setState({ formatName: value });
                const next = formats.find((entry) => entry.name === value);
                narrate(`${value}. About ${(next?.bitsPerWeight ?? 4.9).toFixed(2)} bits per weight.`);
              }}
            />
          </div>
          <RangeControl
            label="Concurrent users"
            min={LIMITS.users.min}
            max={MAX_USERS}
            step={1}
            value={users}
            format={(value) => `${value} streams`}
            onChange={(value) => {
              setState({ users: value });
              narrate(`${value} concurrent streams.`);
            }}
          />
          <RangeControl
            label="Context length"
            min={LIMITS.context.min}
            max={LIMITS.context.max}
            step={512}
            value={context}
            format={(value) => `${value.toLocaleString()} tokens`}
            onChange={(value) => {
              setState({ context: value });
              narrate(
                `Context ${value} tokens. KV cache now ${formatBytes(serving.kvBytesPerToken * value * users)} for ${users} streams.`,
              );
            }}
          />
          <SegmentedControl
            label="Scheduler"
            value={batching ? "batched" : "serial"}
            options={[
              { value: "batched", label: "Continuous batching" },
              { value: "serial", label: "One at a time" },
            ]}
            onChange={(value) => setState({ batching: value === "batched" })}
          />
        </div>
        <p className="lab-note">
          {model.name} uses the shape of {model.shape}: {model.basis} {(model.params / 1e9).toFixed(2)} B parameters.{" "}
          {format.name} is {format.bitsPerWeight.toFixed(hybrid && format.name !== "F16" ? 3 : 2)} bits per weight ({format.basis}).
          {hybrid &&
            ` Only the cache-holding layers are charged per token: ${formatBytes(serving.kvBytesPerToken)} a token here, against ${formatBytes(serving.kvBytesPerToken * (model.totalLayers / model.layers))} if all ${model.totalLayers} layers cached.`}
        </p>
      </LabSurface>

      <LabSurface label="Memory budget" className="tg-memory-panel">
        <SurfaceHeading
          kicker={`${model.name} at ${format.name} · ${context.toLocaleString()} tokens per stream · ${HARDWARE.name}`}
          title="Where the memory goes"
          aside={
            <span className={`tg-badge ${serving.fits ? "" : "is-warning"}`.trim()}>
              {formatBytes(serving.totalBytes)} of {formatBytes(HARDWARE.vramBytes)}
            </span>
          }
        />
        <div className="tg-budget ds-budget">
          <MemoryBar point={serving} users={users} />
        </div>
        <div className="metric-row">
          <Metric label="Per token, per stream" value={formatBytes(serving.kvBytesPerToken)} />
          <Metric label="Per stream at this context" value={formatBytes(serving.kvBytesPerStream)} />
          {hybrid && <Metric label="Fixed state per stream" value={`${(serving.stateBytesPerStream / 1024 ** 2).toFixed(2)} MiB`} />}
          <Metric label="Streams that fit" value={`${serving.maxStreams}`} tone={serving.maxStreams >= users ? "forward" : "loss"} />
          <Metric label="Headroom" value={formatBytes(Math.max(0, headroom))} tone={serving.fits ? "forward" : "loss"} />
        </div>
        <div className={`tg-callout ${serving.fits ? "is-quiet" : "is-warning"}`}>
          <strong>
            {serving.fits
              ? `This configuration fits, with ${formatBytes(headroom)} spare.`
              : `Over budget by ${formatBytes(-headroom)}: at ${context.toLocaleString()} tokens each, only ${serving.maxStreams} streams fit.`}
          </strong>
          <span>
            Weights are paid once. The KV cache is paid per token per stream: the thin dividers in the red segment are
            one stream each, and the dashed slots after it are streams that would still fit. This budget reserves the full context for every stream, the worst case; a paged cache
            allocates in small blocks as a conversation grows, so real streams usually hold less.
            {hybrid &&
              " The recurrent layers add a fixed state per stream that does not grow with context, which is why this model's cache stays small at long context."}
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Throughput and latency" className="tg-throughput-panel">
        <SurfaceHeading
          kicker={batching ? "Continuous batching on" : "One request at a time"}
          title="Total throughput rises while each user waits longer"
          aside={<span className="tg-badge">{serving.computeBound ? "compute bound" : "bandwidth bound"}</span>}
        />
        <p className="ds-figure-title">
          {batching ? "Every decode step feeds every stream" : "Each decode step feeds one stream"} · {ms(span)} drawn to scale
        </p>
        <DecodeTimeline
          point={serving}
          users={users}
          batching={batching}
          soloStep={soloStep}
          bandwidth={LAB_HARDWARE.bandwidthBytesPerSecond}
        />
        <p className="ds-figure-title">The same arithmetic at every user count from 1 to {MAX_USERS}</p>
        <ThroughputPlot sweep={sweep} users={users} maxStreams={serving.maxStreams} ceiling={serving.computeCeiling} />
        <div className="metric-row">
          <Metric label="Total" value={`${tokens(serving.totalTokensPerSecond)} tok/s`} tone="forward" />
          <Metric label="Per user" value={`${tokens(serving.perUserTokensPerSecond)} tok/s`} tone="loss" />
          <Metric label="Time per output token" value={ms(serving.timePerOutputToken)} />
          <Metric label="Time to first token" value={ms(serving.timeToFirstToken)} />
        </div>
        <p className="lab-note">
          Assumptions, stated so you can disagree with them: {(HARDWARE.bandwidthBytesPerSecond / 1e9).toFixed(0)} GB/s
          of memory bandwidth, {(HARDWARE.computeFlops / 1e12).toFixed(0)} TFLOP/s of compute at full utilization,{" "}
          {formatBytes(HARDWARE.vramBytes)} of memory, a flat {formatBytes(HARDWARE.overheadBytes)} for activations
          and runtime buffers, two FLOPs per parameter per token, an fp16 KV cache, attention arithmetic ignored, and
          no speculative decoding or prefix sharing. A batched decode step reads the weights once and every stream's
          cache once, then emits one token per stream. The compute ceiling here is {serving.computeCeiling.toFixed(0)}{" "}
          tok/s, {serving.computeCeiling > serving.totalTokensPerSecond * 5 ? "far above every point on the chart, so decode stays bandwidth bound" : "close enough to the chart that compute can bind"}.
          Time to first token prices a prompt that fills the context. The next card changes the hardware and turns on
          the levers this one leaves off. This is arithmetic, not a benchmark.
        </p>
      </LabSurface>

      <LabSurface label="Hardware, levers, and cost" className="tg-levers-panel">
        <SurfaceHeading
          kicker={`${model.name} at ${format.name} · ${users} stream${users === 1 ? "" : "s"} · ${context.toLocaleString()} tokens · ${preset?.name ?? "your own hardware numbers"}`}
          title="Same load, other hardware, and the levers that move the bill"
          aside={<span className={`tg-badge ${chosen.fits ? "" : "is-warning"}`.trim()}>{costLabel(chosen)}</span>}
        />
        <div className="ds-hardware">
          <SegmentedControl
            label="Hardware"
            value={preset?.id ?? ""}
            options={HARDWARE_PRESETS.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => {
              const next = HARDWARE_PRESETS.find((entry) => entry.id === value);
              if (!next) return;
              setState({
                memoryGiB: next.memoryGiB,
                bandwidthGBs: next.bandwidthGBs,
                computeTflops: next.computeTflops,
                pricePerHour: next.pricePerHour,
              });
              narrate(`${next.name}. ${next.memoryGiB} gibibytes, ${next.bandwidthGBs} gigabytes per second, ${next.computeTflops} teraflops.`);
            }}
          />
          <div className="ds-controls">
            <RangeControl
              label="Memory"
              min={LIMITS.memoryGiB.min}
              max={LIMITS.memoryGiB.max}
              step={1}
              value={config.memoryGiB}
              format={(value) => `${value} GiB`}
              onChange={(value) => setState({ memoryGiB: value })}
            />
            <RangeControl
              label="Memory bandwidth"
              min={LIMITS.bandwidthGBs.min}
              max={LIMITS.bandwidthGBs.max}
              step={1}
              value={config.bandwidthGBs}
              format={() => `${config.bandwidthGBs} GB/s`}
              onChange={(value) => setState({ bandwidthGBs: value })}
            />
            <RangeControl
              label="Compute"
              min={LIMITS.computeTflops.min}
              max={LIMITS.computeTflops.max}
              step={0.5}
              value={config.computeTflops}
              format={() => `${config.computeTflops} TFLOP/s`}
              onChange={(value) => setState({ computeTflops: value })}
            />
            <RangeControl
              label="Price"
              min={LIMITS.pricePerHour.min}
              max={LIMITS.pricePerHour.max}
              step={0.01}
              value={config.pricePerHour}
              format={() => `$${config.pricePerHour.toFixed(2)} an hour`}
              onChange={(value) => setState({ pricePerHour: value })}
            />
            <RangeControl
              label="Busy share of each hour"
              min={LIMITS.utilization.min}
              max={LIMITS.utilization.max}
              step={0.05}
              value={config.utilization}
              format={() => `${Math.round(config.utilization * 100)}% generating`}
              onChange={(value) => setState({ utilization: value })}
            />
          </div>
          <p className="lab-note">
            {preset
              ? `${preset.basis} Every figure is an input you can edit: specs change, and prices change faster.`
              : "Custom hardware: these four numbers are your own assumptions. Specs change and prices change faster, so treat each one as an input rather than a fact."}
          </p>
        </div>

        <div className="ds-levers">
          <div className="tg-switch-list">
            <button
              type="button"
              className="tg-switch"
              aria-pressed={config.paged}
              onClick={() => {
                setState({ paged: !config.paged });
                narrate(config.paged ? "Paged attention off." : "Paged attention on.");
              }}
            >
              <strong>Paged attention</strong>
              <small>
                Allocate the cache in {PAGE_TOKENS}-token pages as each stream fills, instead of reserving the whole context for every stream.
              </small>
              <b>{config.paged ? "on" : "off"}</b>
            </button>
            <button
              type="button"
              className="tg-switch"
              aria-pressed={config.speculative}
              onClick={() => {
                setState({ speculative: !config.speculative });
                narrate(config.speculative ? "Speculative decoding off." : "Speculative decoding on.");
              }}
            >
              <strong>Speculative decoding</strong>
              <small>
                A small draft model proposes tokens and the target model verifies them in one pass, with the same output distribution.
              </small>
              <b>{config.speculative ? "on" : "off"}</b>
            </button>
          </div>
          <SegmentedControl
            label="KV cache type"
            value={config.kvType}
            options={KV_TYPES.map((entry) => ({ value: entry.id, label: `${entry.name} · ${entry.bitsPerValue / 8} B` }))}
            onChange={(value) => {
              setState({ kvType: value });
              narrate(`Cache stored as ${value}.`);
            }}
          />
          <div className="ds-controls">
            <RangeControl
              label="Average fill of the reserved context"
              min={LIMITS.pagedFill.min}
              max={LIMITS.pagedFill.max}
              step={0.05}
              value={config.pagedFill}
              format={() => `${Math.round(config.pagedFill * 100)}% of the context in use`}
              onChange={(value) => setState({ pagedFill: value })}
            />
            <RangeControl
              label="Acceptance rate"
              min={LIMITS.acceptance.min}
              max={LIMITS.acceptance.max}
              step={0.05}
              value={config.acceptance}
              format={() => `α = ${config.acceptance.toFixed(2)}`}
              onChange={(value) => setState({ acceptance: value })}
            />
            <RangeControl
              label="Draft length"
              min={LIMITS.draftLength.min}
              max={LIMITS.draftLength.max}
              step={1}
              value={config.draftLength}
              format={() => `γ = ${config.draftLength} tokens`}
              onChange={(value) => setState({ draftLength: value })}
            />
            <RangeControl
              label="Draft cost"
              min={LIMITS.draftCost.min}
              max={LIMITS.draftCost.max}
              step={0.01}
              value={config.draftCost}
              format={() => `c = ${config.draftCost.toFixed(2)} of a target step`}
              onChange={(value) => setState({ draftCost: value })}
            />
          </div>
        </div>

        <div className="metric-row">
          <Metric
            label="Streams that fit"
            value={`${chosen.maxStreams}`}
            tone={chosen.maxStreams >= users ? "forward" : "loss"}
          />
          <Metric label="Per user" value={`${tokens(chosen.perUserTokensPerSecond)} tok/s`} tone="loss" />
          <Metric label="Total" value={`${tokens(chosen.totalTokensPerSecond)} tok/s`} tone="forward" />
          <Metric label="Cost per million tokens" value={chosen.fits ? dollars(chosen.costPerMillionTokens) : "over memory"} />
        </div>
        <LeverTable
          label="Streams that fit, tokens per second for each user, and total tokens per second for each lever on its own and for your selection, with each one's cost per million tokens"
          rows={leverRows.map((row) => ({ id: row.id, label: row.label, point: row.point, cost: costLabel(row.point).replace(" per M tokens", "") }))}
        />
        <p className="lab-note">
          Arithmetic, not measurement. Decode runs at 100% of the memory bandwidth above for the share of each hour you
          set; cost per million tokens is the price an hour over the output tokens generated in an hour, and prefill is
          not billed. Paged attention charges {PAGE_TOKENS}-token pages for the filled share of the context (
          {chosen.tokensHeldPerStream.toLocaleString()} tokens a stream here); the cache type changes bytes per value;
          a speculative pass costs one verification step plus γ draft steps and yields{" "}
          {leverRows[3].point.tokensPerPass.toFixed(2)} tokens at these settings
          {hybrid ? ". The recurrent state of the 2B model stays float32 whatever the cache type" : ""}. On this load the
          baseline is {costLabel(baseline)}; with your selection it is {costLabel(chosen)}.
        </p>
      </LabSurface>

      <LabSurface label="Export the model you trained" className="tg-export-panel">
        <SurfaceHeading
          kicker={`${TOY_WEIGHTS} weights → ${format.name}`}
          title="Export checklist"
          aside={<span className="tg-badge">{formatBytes(toyBytes)}</span>}
        />
        <ol className="tg-export-steps">
          <li>
            <strong>Merge the adapter</strong>
            <small>One matrix instead of a base plus an adapter, so inference costs what the base cost.</small>
            <b>{TOY_WEIGHTS} weights</b>
          </li>
          <li>
            <strong>Quantize to {format.name}</strong>
            <small>
              {format.name === "F16"
                ? "Half precision, 2 bytes per weight."
                : `${block.block} blocks at ${block.bitsPerWeight} bits per weight${format.name.endsWith("_M") ? `; the _M mix keeps some tensors in a larger type, but this toy has only one tensor` : ""}. The quantization lab measures what that rounding costs.`}
            </small>
            <b>{block.bitsPerWeight} bits/weight</b>
          </li>
          <li>
            <strong>Write the GGUF file</strong>
            <small>Tensors, quantization metadata, and the tokenizer in one file with a checkable header.</small>
            <b>{formatBytes(toyBytes)}</b>
          </li>
          <li>
            <strong>Verify before running</strong>
            <small>Check the header and the file hash, exactly as this app does for its own bundled engine and model.</small>
            <b>sha-256</b>
          </li>
          <li>
            <strong>Serve it</strong>
            <small>
              Bind to loopback, set the context length you budgeted above, and cap concurrency at the streams that
              fit, so an overload queues instead of running out of memory.
            </small>
            <b>≤ {Math.min(serving.maxStreams, users)} streams</b>
          </li>
        </ol>
        <p className="lab-note">
          At production scale the same five steps apply to {model.name} parameters: {formatBytes(serving.weightBytes)}{" "}
          of weights at {format.bitsPerWeight.toFixed(2)} bits each, plus {formatBytes(serving.kvBytes + serving.stateBytes)} of
          cache{hybrid ? " and state" : ""} for the load you chose.
        </p>
      </LabSurface>
    </div>
  );
}
