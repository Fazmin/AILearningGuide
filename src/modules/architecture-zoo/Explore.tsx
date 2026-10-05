import { useMemo } from "react";
import type { CSSProperties } from "react";
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
import {
  columnMeans,
  columnVectors,
  convolve,
  DEFAULT_PIXELS,
  dependencyMatrix,
  FILTERS,
  MAX_EXPERTS,
  moeParameters,
  parsePixels,
  rnnForward,
  routeTokens,
  SIZE,
  windowAt,
  WIRINGS,
  wiringStats,
  type WiringId,
} from "./zoo";

const WIDTH = 64;
const FFN_HIDDEN = 4 * WIDTH;
const LONG_SEQUENCE = 4096;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const signed = (value: number) => (value > 0 ? `+${value}` : value < 0 ? `−${Math.abs(value)}` : "0");
const small = (value: number) =>
  value === 0 ? "0" : Math.abs(value) >= 0.001 ? value.toFixed(3) : value.toExponential(1).replace("-", "−");
const count = (value: number) => value.toLocaleString("en-US");
const rowCol = (index: number) => `r${Math.floor(index / SIZE) + 1} c${(index % SIZE) + 1}`;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const pixels = parsePixels(asString(state, "pixels", DEFAULT_PIXELS));
  const filterId = FILTERS.some((item) => item.id === asString(state, "filter", "edge")) ? asString(state, "filter", "edge") : "edge";
  const filter = FILTERS.find((item) => item.id === filterId) ?? FILTERS[0];
  const cell = clamp(Math.round(asNumber(state, "cell", 0)), 0, SIZE * SIZE - 1);
  const rnnStep = clamp(Math.round(asNumber(state, "rnnStep", 0)), 0, SIZE - 1);
  const recurrentWeight = clamp(asNumber(state, "recurrentWeight", 0.9), 0, 2);
  const wiring = (WIRINGS.some((item) => item.id === asString(state, "wiring", "conv")) ? asString(state, "wiring", "conv") : "conv") as WiringId;
  const convLayers = clamp(Math.round(asNumber(state, "convLayers", 1)), 1, SIZE - 1);
  const experts = clamp(Math.round(asNumber(state, "experts", 8)), 2, MAX_EXPERTS);
  const perToken = clamp(Math.round(asNumber(state, "expertsPerToken", 2)), 1, Math.min(4, experts));

  const selectedRow = Math.floor(cell / SIZE);
  const selectedColumn = cell % SIZE;
  const response = useMemo(() => convolve(pixels, filter.kernel), [filter.kernel, pixels]);
  const neighbourhood = windowAt(pixels, selectedRow, selectedColumn);
  const products = neighbourhood
    .map((value, index) => ({ value, weight: filter.kernel[index] }))
    .filter((item) => item.value !== 0 && item.weight !== 0);
  const windowCells = new Set<number>();
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const y = selectedRow + dy;
      const x = selectedColumn + dx;
      if (y >= 0 && y < SIZE && x >= 0 && x < SIZE) windowCells.add(y * SIZE + x);
    }
  }
  const maxAbs = Math.max(1, ...response.flat().map(Math.abs));

  const inputs = columnMeans(pixels);
  const rnn = useMemo(() => rnnForward(inputs, recurrentWeight), [inputs.join(","), recurrentWeight]); // eslint-disable-line react-hooks/exhaustive-deps
  const laterFactors = rnn.jacobian.slice(rnnStep + 1);

  const matrix = dependencyMatrix(wiring, SIZE, convLayers);
  const stats = WIRINGS.map((item) => ({ ...item, stats: wiringStats(item.id, SIZE, WIDTH) }));
  const current = stats.find((item) => item.id === wiring) ?? stats[0];

  const routes = useMemo(() => routeTokens(columnVectors(pixels), experts, perToken), [experts, perToken, pixels.join("")]); // eslint-disable-line react-hooks/exhaustive-deps
  const load = Array.from({ length: experts }, (_, expert) => routes.filter((route) => route.chosen.includes(expert)).length);
  const moe = moeParameters(WIDTH, FFN_HIDDEN, experts, perToken);
  const idle = load.map((tokens, expert) => (tokens === 0 ? `E${expert + 1}` : "")).filter(Boolean);

  const toggle = (index: number) => {
    const next = pixels.slice();
    next[index] = next[index] ? 0 : 1;
    setState({ pixels: next.join("") });
    narrate(`Pixel ${rowCol(index)} is now ${next[index] ? "on" : "off"}.`);
  };

  return (
    <div className="tg-lab az-lab">
      <LabSurface label="Drawn digit" className="drawn-digit-card">
        <SurfaceHeading kicker="8×8 input · click to ink" title="The image every card below reads" />
        <div className="pixel-pad az-pad" role="grid" aria-label={`Drawn digit, 8 by 8. The outlined 3 by 3 window around ${rowCol(cell)} is what that output of the filter can see.`}>
          {pixels.map((bit, index) => (
            <button
              key={index}
              type="button"
              role="gridcell"
              aria-pressed={bit === 1}
              className={[bit ? "is-on" : "", windowCells.has(index) ? "is-window" : "", index === cell ? "is-selected" : ""].filter(Boolean).join(" ")}
              onClick={() => toggle(index)}
            >
              <span className="sr-only">
                Row {Math.floor(index / SIZE) + 1}, column {(index % SIZE) + 1}, {bit ? "on" : "off"}
                {windowCells.has(index) ? ", inside the selected window" : ""}
              </span>
            </button>
          ))}
        </div>
        <RangeControl
          label="Selected cell"
          min={0}
          max={SIZE * SIZE - 1}
          step={1}
          value={cell}
          format={rowCol}
          onChange={(value) => setState({ cell: value })}
        />
        <button type="button" className="primary-action" onClick={() => toggle(cell)}>
          Flip selected cell
        </button>
        <p className="lab-note">
          The outlined 3×3 window is the receptive field of the selected output cell: the only
          pixels that output can see after one layer.
        </p>
      </LabSurface>

      <LabSurface label="Live filter" className="live-filter-card">
        <SurfaceHeading kicker={`${filter.label} · 9 weights reused at 64 positions`} title="One kernel, slid over every position" />
        <div className="filter-control">
          <SegmentedControl
            label="Filter"
            value={filterId}
            options={FILTERS.map((item) => ({ value: item.id, label: item.label }))}
            onChange={(value) => setState({ filter: value })}
          />
        </div>
        <div className="az-filter-row">
          <div className="az-kernel" role="img" aria-label={`Kernel weights, row by row: ${filter.kernel.join(", ")}`}>
            {filter.kernel.map((weight, index) => (
              <span key={index} className={weight > 0 ? "is-positive" : weight < 0 ? "is-negative" : ""}>
                {signed(weight)}
              </span>
            ))}
            <small>kernel</small>
          </div>
          <div className="az-response" role="grid" aria-label="Filter response: the kernel's dot product with each 3 by 3 window. Select a cell to inspect it.">
            {response.flat().map((value, index) => (
              <button
                key={index}
                type="button"
                role="gridcell"
                aria-label={`Output ${rowCol(index)}: ${signed(value)}`}
                aria-selected={index === cell}
                className={[value > 0 ? "is-positive" : value < 0 ? "is-negative" : "", index === cell ? "is-selected" : ""].filter(Boolean).join(" ")}
                style={{ "--az-heat": Math.abs(value) / maxAbs } as CSSProperties}
                onClick={() => setState({ cell: index })}
              >
                {signed(value)}
              </button>
            ))}
          </div>
        </div>
        <FormulaWithValues
          label={`Output ${rowCol(cell)}`}
          expression={
            products.length
              ? `Σ pixel × weight = ${products.map((item) => `${item.value}·(${signed(item.weight)})`).join(" + ")}`
              : "every inked pixel in this window meets a zero weight"
          }
          result={signed(response[selectedRow][selectedColumn])}
          tone="forward"
        />
        <div className="metric-row">
          <Metric label="Filter weights" value="9, shared" />
          <Metric label="Dense layer, 64 → 64" value={count(64 * 64)} />
          <Metric label="Receptive field" value="3×3 per layer" />
        </div>
      </LabSurface>

      <LabSurface label="Unrolled RNN" className="rnn-card">
        <SurfaceHeading
          kicker={`h7 = ${rnn.hidden[SIZE - 1].toFixed(3)} · column ${rnnStep + 1} reaches it with gradient ${small(rnn.influence[rnnStep])}`}
          title="Read the columns in order; watch early ones fade"
        />
        <div className="az-rnn-controls">
          <RangeControl
            label="Recurrent weight"
            min={0}
            max={2}
            step={0.05}
            value={recurrentWeight}
            format={(value) => `w = ${value.toFixed(2)}`}
            onChange={(value) => setState({ recurrentWeight: Math.round(value * 100) / 100 })}
          />
          <RangeControl
            label="RNN step"
            min={0}
            max={SIZE - 1}
            step={1}
            value={rnnStep}
            format={(value) => `t=${value}`}
            onChange={(value) => setState({ rnnStep: value })}
          />
        </div>
        <ol className="az-chain" aria-label="Unrolled recurrent steps. Select a step to trace its gradient to h7.">
          {rnn.hidden.map((value, index) => {
            const onPath = index > rnnStep;
            return (
              <li
                key={index}
                className={[index === rnnStep ? "is-current" : "", onPath ? "is-path" : ""].filter(Boolean).join(" ")}
                style={{ "--az-h": Math.abs(value) } as CSSProperties}
              >
                <button
                  type="button"
                  aria-current={index === rnnStep ? "step" : undefined}
                  aria-label={`Step t=${index}: input ${inputs[index].toFixed(2)}, hidden ${value.toFixed(3)}, hop factor ${rnn.jacobian[index].toFixed(2)}`}
                  onClick={() => setState({ rnnStep: index })}
                >
                  <span className="az-chain__name">h{index}</span>
                  <strong>{value.toFixed(2)}</strong>
                  <small>x {inputs[index].toFixed(2)}</small>
                </button>
                {index > 0 && <span className="az-chain__hop" aria-hidden="true">×{rnn.jacobian[index].toFixed(2)}</span>}
              </li>
            );
          })}
        </ol>
        <BarList
          label="Gradient of the final hidden state with respect to each column's input"
          selectedId={`t${rnnStep}`}
          onSelect={(id) => setState({ rnnStep: Number(id.slice(1)) })}
          items={rnn.influence.map((value, index) => ({
            id: `t${index}`,
            label: `∂h7/∂x${index} · col ${index + 1}`,
            value: Math.abs(value),
            display: small(value),
            tone: index === rnnStep ? "loss" : "muted",
          }))}
        />
        <FormulaWithValues
          label={`Column ${rnnStep + 1} → h7`}
          expression={`(1 − h${rnnStep}²)${laterFactors.map((factor) => ` × ${factor.toFixed(2)}`).join("")}`}
          result={small(rnn.influence[rnnStep])}
          detail={`Each later step multiplies by its own w·(1 − h²). ${laterFactors.length} hop${laterFactors.length === 1 ? "" : "s"} from column ${rnnStep + 1} to the end.`}
          tone="loss"
        />
        <div className="metric-row">
          <Metric label={`Hidden h${rnnStep}`} value={rnn.hidden[rnnStep].toFixed(3)} />
          <Metric label={`∂h7/∂x${rnnStep}`} value={small(rnn.influence[rnnStep])} tone="loss" />
          <Metric label="Hops to h7" value={`${SIZE - 1 - rnnStep}`} />
          <Metric label="Column mean" value={inputs[rnnStep].toFixed(3)} />
        </div>
      </LabSurface>

      <LabSurface label="Who sees whom" className="az-wiring-card">
        <SurfaceHeading
          kicker={`${current.label} · ${current.stats.seesFuture ? "can see later positions" : wiring === "ffn" ? "sees only its own position" : "causal: sees only earlier positions"}`}
          title="Which inputs can each output use?"
        />
        <div className="az-wiring-control">
          <SegmentedControl
            label="Wiring"
            value={wiring}
            options={WIRINGS.map((item) => ({ value: item.id, label: item.label }))}
            onChange={(value) => setState({ wiring: value })}
          />
        </div>
        {wiring === "conv" && (
          <RangeControl
            label="Conv layers"
            min={1}
            max={SIZE - 1}
            step={1}
            value={convLayers}
            format={(value) => `${value} layer${value === 1 ? "" : "s"} · sees ±${value}`}
            onChange={(value) => setState({ convLayers: value })}
          />
        )}
        <div className="az-matrix-wrap">
          <div
            className="az-matrix"
            role="img"
            aria-label={`${current.label}: rows are output positions 1 to 8, columns are input positions 1 to 8. ${matrix
              .map((row, t) => `Output ${t + 1} sees ${row.filter((hop) => hop !== null).length} inputs`)
              .join(". ")}.`}
          >
            <span className="az-matrix__corner" aria-hidden="true">out ↓ in →</span>
            {Array.from({ length: SIZE }, (_, s) => (
              <span key={`h${s}`} className="az-matrix__head" aria-hidden="true">{s + 1}</span>
            ))}
            {matrix.map((row, t) => [
              <span key={`r${t}`} className="az-matrix__head" aria-hidden="true">{t + 1}</span>,
              ...row.map((hop, s) => (
                <span
                  key={`${t}-${s}`}
                  aria-hidden="true"
                  className={hop === null ? "az-matrix__cell is-blocked" : hop === 1 ? "az-matrix__cell is-direct" : "az-matrix__cell is-chain"}
                  style={hop === null ? undefined : ({ "--az-hop": Math.min(1, (hop - 1) / (SIZE - 1)) } as CSSProperties)}
                >
                  {hop ?? "·"}
                </span>
              )),
            ])}
          </div>
          <ul className="az-matrix-legend">
            <li><i className="az-key az-key--direct" /> 1 = one layer or step between input and output</li>
            <li><i className="az-key az-key--chain" /> 2–8 = hops through a chain of steps or layers</li>
            <li><i className="az-key az-key--blocked" /> · = cannot see (future, or out of reach)</li>
          </ul>
        </div>
        <div className="az-stats" role="table" aria-label={`Per-layer figures at width d = ${WIDTH}, kernel k = 3, state N = 16, n = ${SIZE} positions`}>
          <div role="row" className="az-stats__head">
            <span role="columnheader">Wiring</span>
            <span role="columnheader">Weights / layer</span>
            <span role="columnheader">Sequential steps</span>
            <span role="columnheader">Longest path</span>
            <span role="columnheader">Kept to extend (n = 8 → {count(LONG_SEQUENCE)})</span>
          </div>
          {stats.map((item) => (
            <div role="row" key={item.id} className={item.id === wiring ? "is-current" : ""}>
              <span role="cell">{item.label}</span>
              <span role="cell">{count(item.stats.weights)} <small>{item.stats.weightsFormula}</small></span>
              <span role="cell">{item.stats.sequential}</span>
              <span role="cell">{Number.isFinite(item.stats.longestPath) ? item.stats.longestPath : "never mixes"}</span>
              <span role="cell">
                {item.id === "mlp"
                  ? `${count(item.stats.carried(SIZE))} (fixed window)`
                  : `${count(item.stats.carried(SIZE))} → ${count(item.stats.carried(LONG_SEQUENCE))}`}{" "}
                <small>{item.stats.carriedFormula}</small>
              </span>
            </div>
          ))}
        </div>
        <p className="lab-note">
          {wiring === "mlp" && "Every output sees every input through its own weights: no sharing across positions, so weights grow with (n·d)² and the input length is fixed."}
          {wiring === "ffn" && "The MLP inside a transformer block runs on each position separately. It mixes features within a position and never across positions; attention does that part."}
          {wiring === "conv" && `One shared kernel reaches ±1 position per layer, so ${convLayers} layer${convLayers === 1 ? "" : "s"} see ±${convLayers}. Connecting positions 1 and 8 takes 7 layers of kernel 3 (dilated kernels reach farther).`}
          {wiring === "rnn" && "Causal, and the path from input s to output t runs through t − s + 1 steps that must be computed in order. Everything the past says is squeezed into d numbers."}
          {wiring === "attn" && "Causal self-attention: every output reads every earlier input in one hop, in parallel. The price is n² scores per layer and a KV cache that grows with n."}
          {wiring === "ssm" && "Same causal chain as recurrence, but the state update is linear, so training can use a parallel scan, and generation keeps a fixed d·N state. Mamba makes the update depend on the input."}
        </p>
      </LabSurface>

      <LabSurface label="Sparse experts" className="az-moe-card">
        <SurfaceHeading
          kicker={`${experts} experts · top-${perToken} · ${((perToken / experts) * 100).toFixed(0)}% of expert weights run per token`}
          title="Mixture of experts: many weights stored, few used"
        />
        <div className="az-moe-controls">
          <RangeControl
            label="Experts"
            min={2}
            max={MAX_EXPERTS}
            step={1}
            value={experts}
            format={(value) => `E = ${value}`}
            onChange={(value) => setState({ experts: value, expertsPerToken: Math.min(perToken, value) })}
          />
          <RangeControl
            label="Experts per token"
            min={1}
            max={4}
            step={1}
            value={perToken}
            format={(value) => `k = ${value}`}
            onChange={(value) => setState({ expertsPerToken: Math.min(value, experts) })}
          />
        </div>
        <div className="az-routing-wrap">
          <div
            className="az-routing"
            role="table"
            aria-label="Router gates: rows are tokens (the pad's columns), columns are experts. A percentage means the token runs through that expert with that weight."
            style={{ "--az-experts": experts } as CSSProperties}
          >
            <div role="row" className="az-routing__row az-routing__row--head">
              <span role="columnheader">token</span>
              {Array.from({ length: experts }, (_, expert) => (
                <span role="columnheader" key={expert}>E{expert + 1}</span>
              ))}
            </div>
            {routes.map((route, token) => (
              <div role="row" className="az-routing__row" key={token}>
                <span role="rowheader">col {token + 1}</span>
                {route.gates.map((gate, expert) => (
                  <span
                    role="cell"
                    key={expert}
                    className={gate > 0 ? "is-routed" : ""}
                    style={gate > 0 ? ({ "--az-gate": gate } as CSSProperties) : undefined}
                  >
                    {gate > 0 ? `${Math.round(gate * 100)}%` : ""}
                  </span>
                ))}
              </div>
            ))}
            <div role="row" className="az-routing__row az-routing__row--foot">
              <span role="rowheader">load</span>
              {load.map((tokens, expert) => (
                <span role="cell" key={expert} className={tokens === 0 ? "is-idle" : ""}>{tokens}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="metric-row">
          <Metric label="Total expert weights" value={count(moe.total)} />
          <Metric label="Active per token" value={count(moe.active)} tone="forward" />
          <Metric label="Router weights" value={count(moe.router)} />
        </div>
        <p className="lab-note">
          Each expert is a d → 4d → d MLP with {count(moe.perExpert)} weights. Adding experts grows
          what must be stored; k sets what each token computes. The router is an untrained linear
          layer with fixed seeded weights.{" "}
          {idle.length
            ? `${idle.join(", ")} ${idle.length === 1 ? "gets" : "get"} no tokens: stored, never used. Trained routers add a load-balancing loss to prevent that.`
            : "Every expert gets at least one token here, but the loads are uneven; trained routers add a load-balancing loss."}
        </p>
      </LabSurface>
    </div>
  );
}
