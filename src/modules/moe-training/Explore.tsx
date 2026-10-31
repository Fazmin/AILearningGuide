import { useMemo, useState } from "react";
import {
  BarList,
  FormulaWithValues,
  Heatmap,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  TINY_VOCAB,
  type LineSeries,
  type ModuleContext,
  type SeriesTone,
} from "@app/module-sdk";
import {
  BATCH,
  capacitySeats,
  CORPUS_IDS,
  denseSettings,
  EMBED,
  INPUT,
  MIXTRAL_8X7B,
  moeData,
  parameterCounts,
  routeCharacter,
  trainMoe,
  VOCAB,
  type Balancing,
  type MoeRun,
} from "./moe";
import {
  AUX_LADDER,
  BIAS_LADDER,
  CAPACITY_LADDER,
  EPOCHS_MAX,
  EPOCHS_MIN,
  EXPERTS_MAX,
  EXPERTS_MIN,
  readState,
  SEED_MAX,
  SEED_MIN,
  TOP_K_MAX,
} from "./state";

const BALANCING_LABELS: Record<Balancing, string> = {
  none: "None",
  aux: "Auxiliary loss",
  bias: "Bias update",
};
const CORPUS_LABELS = ["Harbor", "Recipes", "Proverbs"] as const;
/** Line styles for up to eight experts: four colours, then the same four dashed, so no line relies on colour alone. */
const TONES: ReadonlyArray<SeriesTone> = ["forward", "attention", "gradient", "loss"];

const percent = (value: number) => `${Math.round(value * 100)}%`;
const loss = (value: number) => value.toFixed(3);
const count = (value: number) => value.toLocaleString("en-US");
const signed = (value: number) => (value > 0.0005 ? `+${value.toFixed(2)}` : value < -0.0005 ? `−${Math.abs(value).toFixed(2)}` : "0.00");
const signedSmall = (value: number) => (value >= 0 ? value.toFixed(2) : `−${Math.abs(value).toFixed(2)}`);
const billions = (value: number) => `${(value / 1e9).toFixed(1)}B`;
const charName = (character: number) => (TINY_VOCAB[character] === " " ? "space" : TINY_VOCAB[character]);
const quoted = (character: number) => (TINY_VOCAB[character] === " " ? "space" : `“${TINY_VOCAB[character]}”`);
const expertName = (expert: number) => `E${expert + 1}`;
const listNames = (names: string[]) =>
  names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

/** Every character that starts at least one pair, in vocabulary order. */
const PRESENT = moeData()
  .characterCounts.map((total, character) => ({ character, total }))
  .filter((entry) => entry.total > 0);
const DEFAULT_EXAMPLE = Math.max(0, PRESENT.findIndex((entry) => TINY_VOCAB[entry.character] === "t"));

export default function Explore({ state, setState, narrate }: ModuleContext) {
  // Defence in depth: hydrateState already validates, but a live state can also arrive from a snapshot
  // or a patch, and up to three runs train synchronously below.
  const { balancing, auxWeight, biasStep, experts, topK, capacity, epochs, seed, specialView } = readState(state);
  const [exampleIndex, setExampleIndex] = useState(DEFAULT_EXAMPLE);

  const dense = useMemo(() => trainMoe(denseSettings({ epochs, seed })), [epochs, seed]);
  const unbalanced = useMemo(
    () => trainMoe({ experts, topK, balancing: "none", auxWeight: 0, biasStep: 0, capacity, epochs, seed }),
    [capacity, epochs, experts, seed, topK],
  );
  const run: MoeRun = useMemo(
    () =>
      balancing === "none"
        ? unbalanced
        : trainMoe({ experts, topK, balancing, auxWeight, biasStep, capacity, epochs, seed }),
    [auxWeight, balancing, biasStep, capacity, epochs, experts, seed, topK, unbalanced],
  );

  const { final, history, params } = run;
  const shares = final.shares;
  const busiest = shares.indexOf(Math.max(...shares));
  const idle = shares.map((share, e) => (share < 0.005 ? expertName(e) : "")).filter(Boolean);
  const collapsed = experts > topK && final.effective < topK + 0.05;
  const counts = parameterCounts(experts, topK);
  const denseCounts = parameterCounts(1, 1);
  const seats = capacitySeats(capacity, BATCH, topK, experts);
  const seatShare = Number.isFinite(seats) ? Math.min(1, seats / (BATCH * topK)) : 1;
  const balancingLabel = BALANCING_LABELS[balancing];
  const settingLabel =
    balancing === "aux" ? `Aux weight ${auxWeight}` : balancing === "bias" ? `Bias step ${biasStep}` : "No balancing";

  const example = PRESENT[Math.min(exampleIndex, PRESENT.length - 1)];
  const route = routeCharacter(params, example.character);
  const best = route.next.indexOf(Math.max(...route.next));

  const loadSeries: LineSeries[] = [
    ...shares.map((_, e) => ({
      id: `e${e}`,
      name: expertName(e),
      points: history.map((point, epoch) => ({ x: epoch, y: point.shares[e] })),
      tone: TONES[e % TONES.length],
      dash: e < TONES.length ? ("solid" as const) : ("dashed" as const),
      format: percent,
    })),
    {
      id: "even",
      name: "Even share",
      points: [
        { x: 0, y: 1 / experts },
        { x: epochs, y: 1 / experts },
      ],
      tone: "muted",
      dash: "dotted",
      format: percent,
    },
  ];

  /** Columns of the specialisation grid: characters or corpora, each split across the experts. */
  const grid = useMemo(() => {
    const normalise = (row: number[]) => {
      const total = row.reduce((sum, value) => sum + value, 0);
      return row.map((value) => (total > 0 ? value / total : 0));
    };
    if (specialView === "corpus") {
      const rows = final.byCorpus.map(normalise);
      const gap = Math.max(...rows.flatMap((row) => row.map((share, e) => Math.abs(share - shares[e]))));
      return { labels: CORPUS_IDS.map((_, index) => CORPUS_LABELS[index]), rows, gap, purity: 0 };
    }
    const rows = PRESENT.map((entry) => normalise(final.byCharacter[entry.character]));
    const totalPairs = PRESENT.reduce((sum, entry) => sum + entry.total, 0);
    const purity = PRESENT.reduce((sum, entry, index) => sum + entry.total * Math.max(...rows[index]), 0) / totalPairs;
    return { labels: PRESENT.map((entry) => `${charName(entry.character)} (${entry.total})`), rows, gap: 0, purity };
  }, [final, shares, specialView]);
  /** Every character an expert receives any choices of. At top-1 each character appears under exactly one expert. */
  const groups = shares.map((_, e) =>
    PRESENT.filter((entry) => final.byCharacter[entry.character][e] > 0).map((entry) => charName(entry.character)),
  );

  const auxIndex = Math.max(0, AUX_LADDER.indexOf(auxWeight as (typeof AUX_LADDER)[number]));
  const biasIndex = Math.max(0, BIAS_LADDER.indexOf(biasStep as (typeof BIAS_LADDER)[number]));
  const capacityIndex = Math.max(0, CAPACITY_LADDER.indexOf(capacity as (typeof CAPACITY_LADDER)[number]));
  const compareRows = [
    { name: "No balancing", run: unbalanced, selected: balancing === "none" },
    ...(balancing === "none" ? [] : [{ name: settingLabel, run, selected: true }]),
  ];

  return (
    <div className="tg-lab moe-lab">
      <LabSurface label="Router and experts" className="moe-router-card">
        <SurfaceHeading
          kicker={`${experts} expert${experts === 1 ? "" : "s"} · top-${topK} · ${epochs} epoch${epochs === 1 ? "" : "s"} · seed ${seed}`}
          title="A router sends each character to k of N small experts"
          aside={<span className="tg-badge">{count(run.steps)} steps</span>}
        />
        <div className="moe-controls">
          <RangeControl
            label="Experts"
            min={EXPERTS_MIN}
            max={EXPERTS_MAX}
            step={1}
            value={experts}
            format={(value) => (value === 1 ? "1 · dense" : `${value} experts`)}
            onChange={(value) => {
              setState({ experts: value, topK: Math.min(topK, value) });
              narrate(`${value} expert${value === 1 ? "" : "s"}.`);
            }}
          />
          <RangeControl
            label="Top-k"
            min={1}
            max={TOP_K_MAX}
            step={1}
            value={topK}
            format={(value) => `${value} per token`}
            onChange={(value) => {
              setState({ topK: Math.min(value, experts) });
              narrate(`Each token runs ${Math.min(value, experts)} expert${Math.min(value, experts) === 1 ? "" : "s"}.`);
            }}
          />
          <RangeControl
            label="Epochs"
            min={EPOCHS_MIN}
            max={EPOCHS_MAX}
            step={1}
            value={epochs}
            format={(value) => `${value} passes`}
            onChange={(value) => setState({ epochs: value })}
          />
          <RangeControl
            label="Seed"
            min={SEED_MIN}
            max={SEED_MAX}
            step={1}
            value={seed}
            format={(value) => `seed ${value}`}
            onChange={(value) => setState({ seed: value })}
          />
        </div>
        <div className="moe-example">
          <RangeControl
            label="Example character"
            min={0}
            max={PRESENT.length - 1}
            step={1}
            value={exampleIndex}
            format={(index) => `${charName(PRESENT[index].character)} · ${PRESENT[index].total} pairs`}
            onChange={(index) => setExampleIndex(index)}
          />
          <div className="moe-table-wrap">
            <table className="moe-table">
              <caption>
                The trained router on {quoted(example.character)}: h = [{route.h.slice(0, EMBED).map(signedSmall).join(", ")}, 1],
                four learned numbers and the constant 1. Experts are picked by z + b; the gate p is the softmax of z alone.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Expert</th>
                  <th scope="col">Score z</th>
                  <th scope="col">Bias b</th>
                  <th scope="col">z + b</th>
                  <th scope="col">Gate p</th>
                  <th scope="col">Runs</th>
                </tr>
              </thead>
              <tbody>
                {route.z.map((score, e) => {
                  const chosen = route.chosen.includes(e);
                  return (
                    <tr key={e} className={chosen ? "is-selected" : undefined}>
                      <th scope="row">{expertName(e)}</th>
                      <td>{signedSmall(score)}</td>
                      <td>{signedSmall(route.bias[e])}</td>
                      <td>{signedSmall(score + route.bias[e])}</td>
                      <td>{route.gates[e].toFixed(2)}</td>
                      <td>{chosen ? `yes, ×${route.gates[e].toFixed(2)}` : "no"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <FormulaWithValues
            label={`Next-character scores after ${quoted(example.character)}`}
            expression={route.chosen.map((e) => `${route.gates[e].toFixed(2)} × ${expertName(e)}(h)`).join(" + ")}
            result={`top guess ${quoted(best)} · ${percent(route.next[best])}`}
            detail={`Each expert is a ${INPUT} × ${VOCAB} map from h to ${VOCAB} next-character scores. Only the chosen experts run, and each is scaled by its gate.`}
            tone="forward"
          />
        </div>
        <p className="lab-note">
          Every number here is measured. The lab trains this layer on all {count(moeData().count)} character pairs of the three
          tiny corpora with plain SGD (learning rate 0.5, batches of {BATCH}), from seeded starting weights, every time a
          control moves. The router sees only the previous character, so every occurrence of a character takes the same
          route.
        </p>
      </LabSurface>

      <LabSurface label="Expert load over training" className="moe-load-card">
        <SurfaceHeading
          kicker={`Share of the router's choices after each epoch · ${balancing === "none" ? "no balancing" : balancingLabel.toLowerCase()}`}
          title={experts === 1 ? "One expert does all the work" : collapsed ? "One expert has taken over" : "How the work is shared out"}
          aside={<span className={`tg-badge${collapsed ? " is-warning" : ""}`}>{final.effective.toFixed(2)} effective</span>}
        />
        <LineChart
          label="Each expert's share of the router's choices at every epoch boundary, measured on the saved weights over all pairs"
          series={loadSeries}
          xLabel="epoch"
          yLabel="share of choices"
          xDomain={[0, epochs]}
          yDomain={[0, 1]}
        />
        <div className="metric-row">
          <Metric label="Effective experts" value={final.effective.toFixed(2)} tone="forward" />
          <Metric label="Loss" value={loss(final.loss)} tone="loss" />
          <Metric label="Dense baseline" value={loss(dense.final.loss)} />
        </div>
        <div className={`tg-callout ${collapsed || idle.length > 0 ? "is-warning" : "is-quiet"}`}>
          <strong>
            {experts === 1
              ? "With one expert this is the dense baseline: every token runs the same map with a gate of exactly 1."
              : collapsed
                ? topK === 1
                  ? `Collapsed: ${expertName(busiest)} takes ${percent(shares[busiest])} of the choices and ${listNames(idle)} ${idle.length === 1 ? "gets" : "get"} none.`
                  : `Collapsed onto ${topK} experts: the rest get no tokens.`
                : idle.length > 0
                  ? `${listNames(idle)} ${idle.length === 1 ? "gets" : "get"} no tokens, and ${expertName(busiest)} takes ${percent(shares[busiest])}.`
                  : `Every expert gets work: the busiest takes ${percent(Math.max(...shares))} and the quietest ${percent(Math.min(...shares))}.`}
          </strong>
          <span>
            Effective experts is exp(−Σ s·ln s) over the shares s: 1.00 when one expert takes every choice and {experts}.00
            when all {experts} share evenly. Loss is cross-entropy in nats on the same text the run trained on, so it
            measures fit, not generalisation.
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Balancing" className="moe-balancing-card">
        <SurfaceHeading
          kicker={balancing === "none" ? "Trained by the language-model loss alone" : `${balancingLabel} · ${settingLabel.toLowerCase()}`}
          title="Push the router back toward an even split"
        />
        <div className="moe-balancing-controls">
          <SegmentedControl
            label="Balancing"
            value={balancing}
            options={(["none", "aux", "bias"] as const).map((value) => ({ value, label: BALANCING_LABELS[value] }))}
            onChange={(value) => {
              setState({ balancing: value });
              narrate(`Balancing: ${BALANCING_LABELS[value as Balancing]}.`);
            }}
          />
          {balancing === "aux" && (
            <RangeControl
              label="Aux weight"
              min={0}
              max={AUX_LADDER.length - 1}
              step={1}
              value={auxIndex}
              format={(index) => `α = ${AUX_LADDER[index]}`}
              onChange={(index) => {
                setState({ auxWeight: AUX_LADDER[index] });
                narrate(`Aux weight ${AUX_LADDER[index]}.`);
              }}
            />
          )}
          {balancing === "bias" && (
            <RangeControl
              label="Bias step"
              min={0}
              max={BIAS_LADDER.length - 1}
              step={1}
              value={biasIndex}
              format={(index) => `γ = ${BIAS_LADDER[index]}`}
              onChange={(index) => {
                setState({ biasStep: BIAS_LADDER[index] });
                narrate(`Bias step ${BIAS_LADDER[index]}.`);
              }}
            />
          )}
        </div>
        <p className="lab-note">
          {balancing === "aux"
            ? `Each batch adds α · N · Σ f·P to the loss: f is each expert's share of first choices in the batch, held fixed, and P its mean gate. The term is 1 when the split is even and N when one expert takes everything, and its gradient pulls on the router's scores.`
            : balancing === "bias"
              ? `After each batch every expert's bias moves by γ: up if it got fewer than its even share of ${BATCH} × k ÷ N choices, down if it got more. The bias is added to the scores only to pick experts; the gates still come from the scores alone.`
              : "Nothing pushes back: the router learns only from the next-character loss. An expert that wins a few more tokens trains faster, which wins it more tokens."}
        </p>
        <div className="moe-table-wrap">
          <table className="moe-table">
            <caption>Same seed, experts, top-k, capacity and epochs, with and without balancing, beside the one-expert model.</caption>
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Effective experts</th>
                <th scope="col">Loss</th>
                <th scope="col">Mean gate</th>
              </tr>
            </thead>
            <tbody>
              {compareRows.map((row) => (
                <tr key={row.name} className={row.selected ? "is-selected" : undefined}>
                  <th scope="row">{row.name}</th>
                  <td>{row.run.final.effective.toFixed(2)}</td>
                  <td>{loss(row.run.final.loss)}</td>
                  <td>{row.run.final.meanGate.toFixed(2)}</td>
                </tr>
              ))}
              <tr>
                <th scope="row">Dense baseline</th>
                <td>1.00</td>
                <td>{loss(dense.final.loss)}</td>
                <td>1.00</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric label="Balance term N·Σ f·P" value={final.balance.toFixed(2)} />
          <Metric label="Aux loss" value={balancing === "aux" ? (auxWeight * final.balance).toFixed(3) : "off"} tone="gradient" />
        </div>
        <div className="moe-bias-bars">
          <h4 className="tg-subhead">Bias per expert, used only to pick experts</h4>
          <BarList
            label="Balancing bias per expert after training"
            items={Array.from(params.bias, (value, e) => ({
              id: `b${e}`,
              label: expertName(e),
              value,
              display: signed(value),
              tone: value >= 0 ? ("forward" as const) : ("loss" as const),
            }))}
            max={Math.max(0.05, ...Array.from(params.bias, Math.abs))}
          />
          <p className="lab-note">
            {balancing === "bias"
              ? `Starved experts climb and crowded ones sink, by ${biasStep} per batch, so in ${count(run.steps)} steps no bias can move further than ${(biasStep * run.steps).toFixed(1)}. Only the gaps between the bars change a choice.`
              : "The bias stays at 0 unless Balancing is Bias update."}
          </p>
        </div>
      </LabSurface>

      <LabSurface label="Capacity and dropped tokens" className="moe-capacity-card">
        <SurfaceHeading
          kicker={Number.isFinite(seats) ? `${seats} seats per expert in each batch of ${BATCH}` : "No cap: every choice is served"}
          title="Cap each expert, and the overflow is dropped"
        />
        <RangeControl
          label="Capacity factor"
          min={0}
          max={CAPACITY_LADDER.length - 1}
          step={1}
          value={capacityIndex}
          format={(index) => (CAPACITY_LADDER[index] === 0 ? "off" : `${CAPACITY_LADDER[index].toFixed(2)}`)}
          onChange={(index) => {
            setState({ capacity: CAPACITY_LADDER[index] });
            narrate(CAPACITY_LADDER[index] === 0 ? "Capacity cap off." : `Capacity factor ${CAPACITY_LADDER[index]}.`);
          }}
        />
        {Number.isFinite(seats) ? (
          <FormulaWithValues
            label="Seats per expert"
            expression={`⌈${capacity} × ${BATCH} × ${topK} ÷ ${experts}⌉`}
            result={`${seats} of ${BATCH * topK} choices`}
            detail="First choices are seated before second choices, each in batch order. A choice that finds its expert full is dropped."
            tone="loss"
          />
        ) : (
          <p className="lab-note">With the cap off, an expert takes every token routed to it, however many that is.</p>
        )}
        <div className="metric-row">
          <Metric label="Dropped tokens" value={percent(final.dropped)} tone="loss" />
          <Metric label="Unserved tokens" value={percent(final.unserved)} />
          <Metric label="Loss" value={loss(final.loss)} />
        </div>
        <BarList
          label="Each expert's share of the router's choices, with the share its seats can serve"
          items={shares.map((share, e) => ({
            id: `c${e}`,
            label: expertName(e),
            value: share,
            display: percent(share),
            detail: Number.isFinite(seats) ? `room for ${percent(seatShare)}` : undefined,
            tone: share > seatShare + 0.005 ? ("loss" as const) : ("forward" as const),
          }))}
          max={1}
        />
        <p className="lab-note">
          A token that loses every expert predicts all 30 characters equally, a loss of ln 30 = 3.401, because this toy has no
          residual stream to carry it past the layer. The cap is applied the same way when the final weights are scored.
        </p>
      </LabSurface>

      <LabSurface label="Specialisation" className="moe-special-card">
        <SurfaceHeading
          kicker={specialView === "chars" ? "Rows: the previous character, with its pair count" : "Rows: the corpus the pair came from"}
          title="Which tokens each expert gets"
        />
        <SegmentedControl
          label="Grid"
          value={specialView}
          options={[
            { value: "chars", label: "By previous character" },
            { value: "corpus", label: "By corpus" },
          ]}
          onChange={(value) => {
            setState({ specialView: value });
            narrate(value === "chars" ? "Grid by previous character." : "Grid by corpus.");
          }}
        />
        <div className={`moe-heatmap moe-heatmap--${specialView}`}>
          <Heatmap
            label={`Share of each ${specialView === "chars" ? "character's" : "corpus's"} choices that goes to each expert. Each row sums to 100%.`}
            rows={grid.labels}
            columns={shares.map((_, e) => expertName(e))}
            values={grid.rows}
          />
        </div>
        {specialView === "chars" ? (
          <>
            <div className="metric-row">
              <Metric label="Top expert's share per character" value={percent(grid.purity)} tone="forward" />
            </div>
            <ul className="moe-groups" aria-label="The characters each expert receives">
              {groups.map((names, e) => (
                <li key={e}>
                  <b>{expertName(e)}</b> {names.length ? names.join(" ") : "no characters"}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <div className="metric-row">
            <Metric label="Largest gap from the overall split" value={`±${grid.gap.toFixed(2)}`} tone="forward" />
          </div>
        )}
        <p className="lab-note">
          {specialView === "chars"
            ? "Clean rows are guaranteed here, not discovered: the router's only input is the previous character, so every occurrence of a character takes the same route. What training decides is which characters share an expert."
            : "The router never sees which text a pair came from, so each corpus can only inherit the mix of its characters. A close match to the overall split is forced by the design, not a finding about the experts."}
        </p>
      </LabSurface>

      <LabSurface label="Total vs active parameters" className="moe-params-card">
        <SurfaceHeading
          kicker={`${experts} expert${experts === 1 ? "" : "s"} stored · ${topK} run per token`}
          title="Every expert is stored; only k run for each token"
        />
        <div className="metric-row">
          <Metric label="Total parameters" value={count(counts.total)} />
          <Metric label="Active per token" value={count(counts.active)} tone="forward" />
          <Metric label="Loss" value={loss(final.loss)} tone="loss" />
        </div>
        <div className="moe-formulas">
          <FormulaWithValues
            label="Total"
            expression={`${VOCAB}×${EMBED} embedding + ${INPUT}×${experts} router + ${experts}×${INPUT}×${VOCAB} experts`}
            result={count(counts.total)}
            tone="attention"
          />
          <FormulaWithValues
            label="Active per token"
            expression={`${VOCAB}×${EMBED} + ${INPUT}×${experts} + ${topK}×${INPUT}×${VOCAB}`}
            result={count(counts.active)}
            tone="forward"
          />
        </div>
        <div className="moe-table-wrap">
          <table className="moe-table moe-params-table">
            <caption>
              The lab's counts come from the formulas above. The Mixtral row is published, not computed: 46.7B in total and
              12.9B used per token, with 2 of 8 experts in every layer.
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Total</th>
                <th scope="col">Active per token</th>
                <th scope="col">Active share</th>
                <th scope="col">Loss</th>
              </tr>
            </thead>
            <tbody>
              <tr className="is-selected">
                <th scope="row">This lab, {experts} expert{experts === 1 ? "" : "s"}, top-{topK}</th>
                <td>{count(counts.total)}</td>
                <td>{count(counts.active)}</td>
                <td>{percent(counts.active / counts.total)}</td>
                <td>{loss(final.loss)}</td>
              </tr>
              <tr>
                <th scope="row">Dense baseline, 1 expert</th>
                <td>{count(denseCounts.total)}</td>
                <td>{count(denseCounts.active)}</td>
                <td>100%</td>
                <td>{loss(dense.final.loss)}</td>
              </tr>
              <tr>
                <th scope="row">{MIXTRAL_8X7B.name}, published</th>
                <td>{billions(MIXTRAL_8X7B.total)}</td>
                <td>{billions(MIXTRAL_8X7B.active)}</td>
                <td>{percent(MIXTRAL_8X7B.active / MIXTRAL_8X7B.total)}</td>
                <td>not comparable</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Active counts the whole embedding and router, which every token needs, plus the {topK} expert{topK === 1 ? "" : "s"} it
          runs. Memory has to hold all {experts} experts anyway. At real scale they do not fit on one GPU, so each GPU holds
          some of the experts and tokens travel to the ones they chose: expert parallelism, the subject of Splitting
          training across GPUs.
        </p>
      </LabSurface>
    </div>
  );
}
