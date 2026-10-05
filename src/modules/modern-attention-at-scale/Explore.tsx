import { useMemo } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  CONFIG,
  crossoverTokens,
  fullSoftmaxAttention,
  GIB,
  KV_HEAD_OPTIONS,
  kvBytesPerToken,
  kvCacheBytes,
  kvLabel,
  onlineSoftmax,
  prefillFlops,
  ropeFrequency,
  ropeRotate,
  ropeScore,
  rowStatisticBytes,
  scoreMatrixBytes,
} from "./scale-math";

const TILE_SCORES = [0.3, -1.2, 2.4, 0.9, -0.5, 3.1, 1.7, -2.2, 0.1, 2.9, -0.8, 1.1, 0.4, -1.6, 2.2, 0.6];
const TILE_VALUES = TILE_SCORES.map((_, index) => [Math.cos(index * 0.7), Math.sin(index * 0.7)]);
const TILE_SIZES = [2, 4, 8, 16] as const;
const ROPE_Q = [1, 0.2, 0.6, -0.4, 0.8, 0.5, -0.3, 0.9];
const ROPE_K = [0.9, 0.1, 0.3, -0.7, 0.6, 0.6, 0.2, 0.8];
const ROPE_MAX = 96;
const SHIFT = 16;

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const grouped = (value: number) => Math.round(value).toLocaleString("en-US");

function bytes(value: number) {
  if (value >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(2)} TiB`;
  if (value >= GIB) return `${(value / GIB).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(value >= 10 * 1024 ** 2 ? 0 : 1)} MiB`;
  return `${(value / 1024).toFixed(0)} KiB`;
}

function tokensLabel(value: number) {
  return value >= 1024 ? `${grouped(value)} (${value / 1024}K)` : grouped(value);
}

function HeadSharing({ kvHeads }: { kvHeads: number }) {
  const width = 640;
  const cell = 16;
  const gap = (width - 20 - CONFIG.queryHeads * cell) / (CONFIG.queryHeads - 1);
  const group = CONFIG.queryHeads / kvHeads;
  const xAt = (index: number) => 10 + index * (cell + gap);
  return (
    <figure className="mas-heads">
      <figcaption>{CONFIG.queryHeads} query heads</figcaption>
      <svg
        viewBox={`0 0 ${width} 62`}
        role="img"
        aria-label={`${CONFIG.queryHeads} query heads share ${kvHeads} key-value heads, ${group} query ${group === 1 ? "head" : "heads"} per KV head.`}
      >
        {Array.from({ length: CONFIG.queryHeads }, (_, index) => {
          const kv = Math.floor(index / group);
          const groupStart = xAt(kv * group);
          const groupEnd = xAt((kv + 1) * group - 1) + cell;
          return (
            <g key={index}>
              <rect className="mas-heads__q" x={xAt(index)} y={2} width={cell} height={cell} rx={3} />
              <line
                className="mas-heads__link"
                x1={xAt(index) + cell / 2}
                y1={18}
                x2={(groupStart + groupEnd) / 2}
                y2={42}
              />
            </g>
          );
        })}
        {Array.from({ length: kvHeads }, (_, kv) => {
          const start = xAt(kv * group);
          const end = xAt((kv + 1) * group - 1) + cell;
          return <rect key={kv} className="mas-heads__kv" x={start} y={42} width={end - start} height={16} rx={3} />;
        })}
      </svg>
      <figcaption>
        {kvHeads} KV {kvHeads === 1 ? "head" : "heads"} cached · {group} query {group === 1 ? "head" : "heads"} each
      </figcaption>
    </figure>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const kvHeadsRaw = Math.round(asNumber(state, "kvHeads", 8));
  const kvHeads = (KV_HEAD_OPTIONS as readonly number[]).includes(kvHeadsRaw) ? kvHeadsRaw : 8;
  const windowed = asString(state, "span", "full") === "window";
  const windowSize = windowed ? CONFIG.window : undefined;
  const seqLen = 2 ** clamp(Math.round(Math.log2(asNumber(state, "seqLen", 8192))), 10, 17);
  const tileSizeRaw = Math.round(asNumber(state, "tileSize", 4));
  const tileSize = (TILE_SIZES as readonly number[]).includes(tileSizeRaw) ? tileSizeRaw : 4;
  const tileCount = TILE_SCORES.length / tileSize;
  const tiles = clamp(Math.round(asNumber(state, "tiles", tileCount)), 0, tileCount);
  const queryPos = clamp(Math.round(asNumber(state, "queryPos", 20)), 0, ROPE_MAX);
  const keyPos = clamp(Math.round(asNumber(state, "keyPos", 7)), 0, ROPE_MAX);
  const pair = clamp(Math.round(asNumber(state, "pair", 1)), 0, 3);
  const ropeBase = asNumber(state, "ropeBase", 10000) === 500000 ? 500000 : 10000;

  /* KV cache */
  const perToken = kvBytesPerToken(kvHeads);
  const cache = kvCacheBytes(kvHeads, seqLen, windowSize);
  const mhaCache = kvCacheBytes(CONFIG.queryHeads, seqLen, windowSize);
  const kept = windowSize ? Math.min(seqLen, windowSize) : seqLen;

  /* Cost */
  const flops = prefillFlops(seqLen, kvHeads, windowSize);
  const attentionShare = flops.attention / (flops.attention + flops.weights);
  const crossover = crossoverTokens(kvHeads);
  const costSeries = useMemo(() => {
    const samples = Array.from({ length: 49 }, (_, index) => 1024 + (index / 48) * (131072 - 1024));
    const at = (fn: (n: number) => number) => samples.map((n) => ({ x: n / 1024, y: fn(n) / 1e12 }));
    return [
      {
        id: "weights",
        name: "Weight matmuls",
        tone: "forward" as const,
        dash: "solid" as const,
        points: at((n) => prefillFlops(n, kvHeads).weights),
        format: (value: number) => `${value.toFixed(1)} TFLOP`,
      },
      {
        id: "full",
        name: "Attention, full span",
        tone: "loss" as const,
        dash: "dashed" as const,
        points: at((n) => prefillFlops(n, kvHeads).attention),
        format: (value: number) => `${value.toFixed(1)} TFLOP`,
      },
      {
        id: "window",
        name: `Attention, window ${grouped(CONFIG.window)}`,
        tone: "attention" as const,
        dash: "dotted" as const,
        points: at((n) => prefillFlops(n, kvHeads, CONFIG.window).attention),
        format: (value: number) => `${value.toFixed(1)} TFLOP`,
      },
    ];
  }, [kvHeads]);

  /* Online softmax */
  const online = useMemo(() => onlineSoftmax(TILE_SCORES, TILE_VALUES, tileSize), [tileSize]);
  const full = useMemo(() => fullSoftmaxAttention(TILE_SCORES, TILE_VALUES), []);
  const current = tiles > 0 ? online.steps[tiles - 1] : undefined;
  const seen = current?.to ?? 0;
  const partialWeights = TILE_SCORES.map((score, index) =>
    current && index < seen ? Math.exp(score - current.runningMax) / current.runningSum : 0,
  );
  const partialOutput = current ? current.accumulator.map((value) => value / current.runningSum) : [0, 0];
  const finalGap = Math.max(...online.output.map((value, index) => Math.abs(value - full.output[index])));
  const maxWeight = Math.max(...full.weights, ...partialWeights);

  /* RoPE */
  const theta = ropeFrequency(pair, ROPE_Q.length, ropeBase);
  const qRot = ropeRotate(ROPE_Q, queryPos, ropeBase);
  const kRot = ropeRotate(ROPE_K, keyPos, ropeBase);
  const score = ropeScore(ROPE_Q, ROPE_K, queryPos, keyPos, ropeBase);
  const relative = queryPos - keyPos;
  const shiftUp = Math.max(queryPos, keyPos) + SHIFT <= ROPE_MAX;
  const shiftedScore = shiftUp
    ? ropeScore(ROPE_Q, ROPE_K, queryPos + SHIFT, keyPos + SHIFT, ropeBase)
    : ropeScore(ROPE_Q, ROPE_K, queryPos - SHIFT, keyPos - SHIFT, ropeBase);
  const canShiftDown = Math.min(queryPos, keyPos) - SHIFT >= 0;
  const ropeSeries = useMemo(() => {
    const distances = Array.from({ length: ROPE_MAX + 1 }, (_, index) => index);
    const pairScore = (distance: number, base: number) => {
      const q = ropeRotate(ROPE_Q, distance, base);
      const k = ROPE_K;
      return q[2 * pair] * k[2 * pair] + q[2 * pair + 1] * k[2 * pair + 1];
    };
    return [
      {
        id: "base-10k",
        name: "base 10,000",
        tone: "attention" as const,
        dash: "solid" as const,
        points: distances.map((distance) => ({ x: distance, y: pairScore(distance, 10000) })),
        format: (value: number) => value.toFixed(3),
      },
      {
        id: "base-500k",
        name: "base 500,000",
        tone: "forward" as const,
        dash: "dashed" as const,
        points: distances.map((distance) => ({ x: distance, y: pairScore(distance, 500000) })),
        format: (value: number) => value.toFixed(3),
      },
    ];
  }, [pair]);
  const pairAngle = (vector: number[]) => Math.atan2(vector[2 * pair + 1], vector[2 * pair]);
  const ray = (vector: number[]) => {
    const length = Math.hypot(vector[2 * pair], vector[2 * pair + 1]) || 1;
    return { x: vector[2 * pair] / length, y: vector[2 * pair + 1] / length };
  };
  const qRay = ray(qRot);
  const kRay = ray(kRot);
  const q0 = ray(ROPE_Q);
  const k0 = ray(ROPE_K);
  const between = (((pairAngle(qRot) - pairAngle(kRot)) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;

  return (
    <div className="mas-lab">
      <LabSurface label="KV cache budget" className="mas-kv-card">
        <SurfaceHeading
          kicker={`${CONFIG.name} · ${CONFIG.layers} layers · ${CONFIG.queryHeads} query heads · head dim ${CONFIG.headDim} · 16-bit cache`}
          title="Fewer KV heads, smaller cache"
        />
        <div className="mas-controls mas-controls--stacked">
          <div className="mechanism-control">
            <SegmentedControl
              label="KV heads"
              value={String(kvHeads)}
              options={KV_HEAD_OPTIONS.map((option) => ({ value: String(option), label: `${option} · ${kvLabel(option)}` }))}
              onChange={(value) => {
                setState({ kvHeads: Number(value) });
                narrate(`${value} key-value heads: ${bytes(kvCacheBytes(Number(value), seqLen, windowSize))} of cache.`);
              }}
            />
          </div>
          <div className="mas-span-control">
            <SegmentedControl
              label="Attention span"
              value={windowed ? "window" : "full"}
              options={[
                { value: "full", label: "Full" },
                { value: "window", label: `Sliding window ${grouped(CONFIG.window)}` },
              ]}
              onChange={(value) => setState({ span: value })}
            />
          </div>
        </div>
        <RangeControl
          label="Sequence length"
          min={10}
          max={17}
          step={1}
          value={Math.log2(seqLen)}
          format={(value) => `${tokensLabel(2 ** value)} tokens`}
          onChange={(value) => setState({ seqLen: 2 ** value })}
        />
        <HeadSharing kvHeads={kvHeads} />
        <BarList
          label={`KV cache at ${grouped(seqLen)} tokens`}
          items={KV_HEAD_OPTIONS.map((option) => ({
            id: String(option),
            label: `${option} KV · ${kvLabel(option)}`,
            value: kvCacheBytes(option, seqLen, windowSize),
            display: bytes(kvCacheBytes(option, seqLen, windowSize)),
            tone: option === kvHeads ? ("attention" as const) : ("muted" as const),
            emphasis: option === kvHeads,
          }))}
        />
        <FormulaWithValues
          label="KV cache"
          expression={`2 × ${CONFIG.layers} × ${kvHeads} × ${CONFIG.headDim} × 2 B × ${grouped(kept)}`}
          result={bytes(cache)}
          detail={
            windowSize && seqLen > windowSize
              ? `K and V × layers × KV heads × head dim × bytes × tokens. Only the last ${grouped(windowSize)} tokens are kept, a rolling buffer as in Mistral 7B. A token reaches ${grouped(windowSize)} back directly and up to ${grouped(windowSize * CONFIG.layers)} back by hopping through ${CONFIG.layers} layers.`
              : "K and V × layers × KV heads × head dim × bytes × tokens. Grows linearly with tokens and with KV heads; FlashAttention does not change it."
          }
        />
        <div className="metric-row">
          <Metric label="Per token" value={bytes(perToken)} />
          <Metric label="Cache" value={bytes(cache)} tone="gradient" />
          <Metric label="vs MHA" value={`${((cache / mhaCache) * 100).toFixed(1)}%`} tone="forward" />
        </div>
      </LabSurface>

      <LabSurface label="Quadratic cost" className="mas-cost-card">
        <SurfaceHeading kicker="Prefill · one layer · causal mask" title="The n² term catches up" />
        <LineChart
          label="Prefill FLOPs for one layer against sequence length"
          series={costSeries}
          xLabel="sequence length (K tokens)"
          yLabel="TFLOP per layer"
          marker={{ x: seqLen / 1024, label: `${grouped(seqLen)} tokens` }}
          footnote={`Weight matmuls: 2 × n × ${grouped(prefillFlops(1, kvHeads).weights / 2)} weights. Attention: 4 × ${CONFIG.queryHeads} heads × ${CONFIG.headDim} × causal query–key pairs.`}
        />
        <div className="metric-row">
          <Metric label="Attention share" value={`${(attentionShare * 100).toFixed(1)}%`} tone="loss" />
          <Metric label="Crossover" value={`${grouped(crossover)} tokens`} />
        </div>
        <div className="metric-row">
          <Metric label="Score matrix if stored" value={bytes(scoreMatrixBytes(seqLen))} tone="gradient" />
          <Metric label="FlashAttention keeps" value={bytes(rowStatisticBytes(seqLen))} tone="forward" />
        </div>
        <p className="mas-note">
          KV heads barely moves these curves: every one of the {CONFIG.queryHeads} query heads still scores every
          visible key. Fewer KV heads only shrink the K and V projections. A window caps each query at{" "}
          {grouped(CONFIG.window)} keys, so the dotted line grows linearly past {grouped(CONFIG.window)} tokens.
        </p>
      </LabSurface>

      <LabSurface label="FlashAttention tiles" className="mas-flash-card">
        <SurfaceHeading
          kicker={`One query row · ${TILE_SCORES.length} keys · 2-d values · ${tileCount} tiles`}
          title="Exact softmax, one tile at a time"
        />
        <div className="mas-controls">
          <SegmentedControl
            label="Tile size"
            value={String(tileSize)}
            options={TILE_SIZES.map((size) => ({ value: String(size), label: `${size} keys` }))}
            onChange={(value) => setState({ tileSize: Number(value), tiles: TILE_SCORES.length / Number(value) })}
          />
        </div>
        <RangeControl
          label="Tiles processed"
          min={0}
          max={tileCount}
          step={1}
          value={tiles}
          format={(value) => `${value} of ${tileCount}`}
          onChange={(value) => setState({ tiles: value })}
        />
        <p className="mas-caption">
          Keys wait in HBM, one box each with its score s = q·k. The outlined tile is the one loaded into SRAM; shaded
          tiles are done.
        </p>
        <svg
          className="mas-tiles"
          viewBox="0 0 640 176"
          role="img"
          aria-label={`${seen} of ${TILE_SCORES.length} keys processed. Running max ${current ? current.runningMax.toFixed(2) : "none"}, running sum ${current ? current.runningSum.toFixed(3) : "0"}.`}
        >
          {TILE_SCORES.map((value, index) => {
            const x = 6 + index * 39.2;
            const tile = Math.floor(index / tileSize);
            const status = tile < tiles - 1 ? "done" : tile === tiles - 1 ? "sram" : "waiting";
            return (
              <g key={index} className={`mas-tiles__key is-${status}`}>
                <rect x={x} y={8} width={36} height={26} rx={4} />
                <text x={x + 18} y={25.5} textAnchor="middle">{value.toFixed(1)}</text>
              </g>
            );
          })}
          {Array.from({ length: tileCount }, (_, tile) => (
            <rect
              key={tile}
              className={`mas-tiles__group ${tile === tiles - 1 ? "is-current" : ""}`}
              x={6 + tile * tileSize * 39.2 - 2}
              y={4}
              width={tileSize * 39.2 - 0.2}
              height={34}
              rx={6}
            />
          ))}
          {TILE_SCORES.map((_, index) => {
            const x = 6 + index * 39.2;
            const trueHeight = (full.weights[index] / maxWeight) * 116;
            const partialHeight = (partialWeights[index] / maxWeight) * 116;
            return (
              <g key={`w-${index}`}>
                <rect className="mas-tiles__partial" x={x + 7} y={166 - partialHeight} width={22} height={Math.max(0, partialHeight)} />
                <rect className="mas-tiles__true" x={x + 5} y={166 - trueHeight} width={26} height={Math.max(0.5, trueHeight)} />
              </g>
            );
          })}
          <line className="mas-tiles__base" x1={6} x2={634} y1={166} y2={166} />
        </svg>
        <div className="mas-legend" aria-hidden="true">
          <span><i className="mas-legend__true" />full softmax weight</span>
          <span><i className="mas-legend__partial" />exp(s − m) ÷ ℓ with the tiles so far</span>
        </div>
        <div className="mas-table-wrap">
          <table className="mas-table">
            <caption className="sr-only">Running statistics after each tile</caption>
            <thead>
              <tr>
                <th scope="col">Tile</th>
                <th scope="col">Tile max</th>
                <th scope="col">Running max m</th>
                <th scope="col">Rescale e^(m_old − m)</th>
                <th scope="col">Running sum ℓ</th>
                <th scope="col">Output o ÷ ℓ</th>
              </tr>
            </thead>
            <tbody>
              {online.steps.slice(0, tiles).map((step) => (
                <tr key={step.tile} className={step.tile === tiles - 1 ? "is-current" : ""}>
                  <th scope="row">{step.tile + 1} · keys {step.from}–{step.to - 1}</th>
                  <td>{step.tileMax.toFixed(2)}</td>
                  <td>{step.runningMax.toFixed(2)}</td>
                  <td>{step.tile === 0 ? "—" : step.rescale.toFixed(4)}</td>
                  <td>{step.runningSum.toFixed(4)}</td>
                  <td>[{(step.accumulator[0] / step.runningSum).toFixed(4)}, {(step.accumulator[1] / step.runningSum).toFixed(4)}]</td>
                </tr>
              ))}
              {tiles === 0 && (
                <tr>
                  <td colSpan={6}>No tiles yet: m = −∞, ℓ = 0.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <FormulaWithValues
          label="Online vs full"
          expression={tiles === tileCount ? "max |o_online − o_full|" : `${seen} of ${TILE_SCORES.length} keys seen`}
          result={tiles === tileCount ? finalGap.toExponential(1) : `[${partialOutput.map((value) => value.toFixed(3)).join(", ")}]`}
          tone="forward"
          detail={
            tiles === tileCount
              ? `Full softmax output [${full.output.map((value) => value.toFixed(4)).join(", ")}]. The only difference is floating-point rounding.`
              : `Partial output so far. The full answer is [${full.output.map((value) => value.toFixed(3)).join(", ")}].`
          }
        />
      </LabSurface>

      <LabSurface label="RoPE rotation" className="mas-rope-card">
        <SurfaceHeading
          kicker={`Toy head: 8 dims, 4 pairs · θᵢ = ${grouped(ropeBase)}^(−2i/8)`}
          title="Position as a rotation"
        />
        <div className="mas-rope-grid">
          <div>
            <RangeControl
              label="Query position"
              min={0}
              max={ROPE_MAX}
              step={1}
              value={queryPos}
              format={(value) => `m = ${value}`}
              onChange={(value) => setState({ queryPos: value })}
            />
            <RangeControl
              label="Key position"
              min={0}
              max={ROPE_MAX}
              step={1}
              value={keyPos}
              format={(value) => `n = ${value}`}
              onChange={(value) => setState({ keyPos: value })}
            />
            <div className="mas-shift">
              <button
                type="button"
                disabled={!canShiftDown}
                onClick={() => setState({ queryPos: queryPos - SHIFT, keyPos: keyPos - SHIFT })}
              >
                Shift both −{SHIFT}
              </button>
              <button
                type="button"
                disabled={!shiftUp}
                onClick={() => setState({ queryPos: queryPos + SHIFT, keyPos: keyPos + SHIFT })}
              >
                Shift both +{SHIFT}
              </button>
            </div>
            <RangeControl
              label="Dimension pair"
              min={0}
              max={3}
              step={1}
              value={pair}
              format={(value) => `dims ${value * 2},${value * 2 + 1} · θ = ${ropeFrequency(value, 8, ropeBase).toPrecision(2)}`}
              onChange={(value) => setState({ pair: value })}
            />
            <SegmentedControl
              label="RoPE base"
              value={String(ropeBase)}
              options={[
                { value: "10000", label: "10,000" },
                { value: "500000", label: "500,000" },
              ]}
              onChange={(value) => setState({ ropeBase: Number(value) })}
            />
          </div>
          <svg
            className="mas-rope-plane"
            viewBox="-1.35 -1.35 2.7 2.7"
            role="img"
            aria-label={`Pair ${pair}: the query is rotated by ${(queryPos * theta).toFixed(2)} radians and the key by ${(keyPos * theta).toFixed(2)} radians; the angle between them is ${between.toFixed(2)} radians.`}
          >
            <circle className="mas-rope-plane__circle" r="1" />
            <line className="mas-rope-plane__axis" x1={-1.25} x2={1.25} y1={0} y2={0} />
            <line className="mas-rope-plane__axis" x1={0} x2={0} y1={-1.25} y2={1.25} />
            <line className="mas-rope-plane__ghost" x1={0} y1={0} x2={q0.x} y2={-q0.y} />
            <line className="mas-rope-plane__ghost" x1={0} y1={0} x2={k0.x} y2={-k0.y} />
            <line className="mas-rope-plane__q" x1={0} y1={0} x2={qRay.x} y2={-qRay.y} />
            <line className="mas-rope-plane__k" x1={0} y1={0} x2={kRay.x} y2={-kRay.y} />
            <circle className="mas-rope-plane__q-dot" cx={qRay.x} cy={-qRay.y} r="0.07" />
            <rect className="mas-rope-plane__k-dot" x={kRay.x - 0.06} y={-kRay.y - 0.06} width="0.12" height="0.12" />
            <text className="mas-rope-plane__text" x={qRay.x * 1.18} y={-qRay.y * 1.18 + 0.04} textAnchor="middle">q</text>
            <text className="mas-rope-plane__text" x={kRay.x * 1.18} y={-kRay.y * 1.18 + 0.04} textAnchor="middle">k</text>
          </svg>
        </div>
        <FormulaWithValues
          label="Score"
          expression={`rot(q, ${queryPos}) · rot(k, ${keyPos})`}
          result={score.toFixed(4)}
          detail={`Shift both by ${shiftUp ? "+" : "−"}${SHIFT}: ${shiftedScore.toFixed(4)}. Only m − n = ${relative} enters the dot product.${relative < 0 ? " The key is after the query, so a causal model would mask it." : ""}`}
        />
        <div className="metric-row">
          <Metric label="m − n" value={String(relative)} />
          <Metric label="Pair angle" value={`${between.toFixed(2)} rad`} tone="gradient" />
          <Metric label="Wavelength 2π/θ" value={`${grouped((2 * Math.PI) / theta)} tokens`} tone="forward" />
        </div>
        <LineChart
          label={`Contribution of dims ${pair * 2},${pair * 2 + 1} to the score against relative distance`}
          series={ropeSeries}
          xLabel="relative distance m − n (tokens)"
          yLabel={`dims ${pair * 2},${pair * 2 + 1} share of q·k`}
          marker={relative >= 0 ? { x: relative, label: `m − n = ${relative}` } : undefined}
          footnote="This pair's term |q||k|·cos(φ + (m − n)θ): a cosine in the distance whose wavelength is 2π/θ. The full score adds four such terms. Legend values are at distance 96."
        />
        <p className="mas-note">
          Llama models use 128-dimensional heads, so 64 pairs with θ from 1 down to about base^(−1). Llama 3 raised
          the base from 10,000 to 500,000, which slows the slow pairs further; context-extension methods such as
          position interpolation, NTK-aware scaling and YaRN also work by rescaling these frequencies.
        </p>
      </LabSurface>
    </div>
  );
}
