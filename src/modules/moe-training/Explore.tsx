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
import { CapacityBars, CellBar, CharacterGroups, LossCell, ParamBars, RouteDiagram, SplitBars, type ParamRow } from "./visuals";
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

  const skipped = route.z.map((_, e) => e).filter((e) => !route.chosen.includes(e));
  const routeSpoken = `${quoted(example.character)} runs ${listNames(
    route.chosen.map((e) => `${expertName(e)} with gate ${route.gates[e].toFixed(2)}`),
  )}${skipped.length ? `; ${listNames(skipped.map(expertName))} ${skipped.length === 1 ? "is" : "are"} skipped` : ""}. Top guess for the next character: ${quoted(best)}, ${percent(route.next[best])}.`;
  const nextRanked = route.next
    .map((share, character) => ({ label: charName(character) === "space" ? "␣" : charName(character), share }))
    .sort((left, right) => right.share - left.share)
    .slice(0, 4);

  /** The first epoch boundary at which the router is down to k experts, if it ever is. */
  const collapseEpoch = experts > topK ? history.findIndex((point, epoch) => epoch > 0 && point.effective < topK + 0.05) : -1;

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
      return { labels: [...CORPUS_IDS.map((_, index) => CORPUS_LABELS[index]), "All three"], rows: [...rows, [...shares]], gap, purity: 0 };
    }
    const rows = PRESENT.map((entry) => normalise(final.byCharacter[entry.character]));
    const totalPairs = PRESENT.reduce((sum, entry) => sum + entry.total, 0);
    const purity = PRESENT.reduce((sum, entry, index) => sum + entry.total * Math.max(...rows[index]), 0) / totalPairs;
    return { labels: PRESENT.map((entry) => `${charName(entry.character)} (${entry.total})`), rows, gap: 0, purity };
  }, [final, shares, specialView]);
  /** Every character an expert receives any choices of. At top-1 each character appears under exactly one expert. */
  const groups = shares.map((_, e) =>
    PRESENT.filter((entry) => final.byCharacter[entry.character][e] > 0).map((entry) => ({
      name: charName(entry.character),
      pairs: entry.total,
    })),
  );

  const sharedSegments = [
    { id: "embedding", label: "embed", size: counts.embedding, kind: "shared" as const },
    { id: "router", label: "", size: counts.router, kind: "shared" as const },
  ];
  const paramRows: ParamRow[] = [
    {
      label: `Stored: all ${experts} expert${experts === 1 ? "" : "s"}`,
      total: counts.total,
      segments: [
        ...sharedSegments,
        ...shares.map((_, e) => ({
          id: `e${e}`,
          label: expertName(e),
          size: counts.perExpert,
          kind: route.chosen.includes(e) ? ("active" as const) : ("idle" as const),
        })),
      ],
    },
    {
      label: `Run for one token, here ${quoted(example.character)}`,
      total: counts.active,
      segments: [
        ...sharedSegments,
        ...route.chosen
          .slice()
          .sort((left, right) => left - right)
          .map((e) => ({ id: `e${e}`, label: expertName(e), size: counts.perExpert, kind: "active" as const })),
      ],
    },
    {
      label: "Dense baseline: 1 expert",
      total: denseCounts.total,
      segments: [
        { id: "embedding", label: "embed", size: denseCounts.embedding, kind: "shared" as const },
        { id: "router", label: "", size: denseCounts.router, kind: "shared" as const },
        { id: "e0", label: "E1", size: denseCounts.perExpert, kind: "active" as const },
      ],
    },
  ];
  const paramSpoken = `The lab stores ${count(counts.total)} weights: an embedding of ${counts.embedding}, a router of ${counts.router} and ${experts} expert${experts === 1 ? "" : "s"} of ${counts.perExpert} each. One token runs ${count(counts.active)} of them. The one-expert dense baseline stores and runs ${count(denseCounts.total)}.`;

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
          <RouteDiagram
            token={charName(example.character) === "space" ? "␣" : charName(example.character)}
            tokenNote={`${charName(example.character) === "space" ? "space · " : ""}${example.total} pairs`}
            spoken={routeSpoken}
            z={route.z}
            bias={route.bias}
            gates={route.gates}
            chosen={route.chosen}
            next={nextRanked}
            topK={topK}
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
          marker={collapseEpoch > 0 ? { x: collapseEpoch, label: `down to ${topK} expert${topK === 1 ? "" : "s"} by epoch ${collapseEpoch}` } : undefined}
        />
        <SplitBars
          rows={[
            { label: "Before training", shares: history[0].shares, note: `${history[0].effective.toFixed(2)} effective` },
            { label: `After ${epochs} epoch${epochs === 1 ? "" : "s"}`, shares, note: `${final.effective.toFixed(2)} effective` },
          ]}
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
            <caption>
              Same seed, experts, top-k, capacity and epochs, with and without balancing, beside the one-expert model. The
              bar under each Loss starts at the dense baseline: left is lower (better), right is higher, and a full bar is
              0.1.
            </caption>
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
                  <td>
                    <CellBar value={row.run.final.effective} max={experts} text={row.run.final.effective.toFixed(2)} />
                  </td>
                  <td>
                    <LossCell value={row.run.final.loss} baseline={dense.final.loss} reach={0.1} text={loss(row.run.final.loss)} />
                  </td>
                  <td>
                    <CellBar value={row.run.final.meanGate} max={1} text={row.run.final.meanGate.toFixed(2)} />
                  </td>
                </tr>
              ))}
              <tr>
                <th scope="row">Dense baseline</th>
                <td>
                  <CellBar value={1} max={experts} text="1.00" tone="muted" />
                </td>
                <td>
                  <LossCell value={dense.final.loss} baseline={dense.final.loss} reach={0.1} text={loss(dense.final.loss)} />
                </td>
                <td>
                  <CellBar value={1} max={1} text="1.00" tone="muted" />
                </td>
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
        <CapacityBars shares={shares} room={Number.isFinite(seats) ? seatShare : null} dropped={final.dropped} />
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
          <div className="moe-heatmap__blocks">
            {(specialView === "chars" ? [0, 1] : [0]).map((half) => {
              const cut = specialView === "chars" ? Math.ceil(grid.labels.length / 2) : grid.labels.length;
              return (
                <Heatmap
                  key={half}
                  label={`Share of each ${specialView === "chars" ? "character's" : "corpus's"} choices that goes to each expert${specialView === "chars" ? (half === 0 ? ", first half of the characters" : ", second half of the characters") : ""}. Each row sums to 100%.`}
                  rows={grid.labels.slice(half * cut, half * cut + cut)}
                  columns={shares.map((_, e) => expertName(e))}
                  values={grid.rows.slice(half * cut, half * cut + cut)}
                />
              );
            })}
          </div>
        </div>
        {specialView === "chars" ? (
          <>
            <div className="metric-row">
              <Metric label="Top expert's share per character" value={percent(grid.purity)} tone="forward" />
            </div>
            <h4 className="tg-subhead moe-groups__title">The same grid, sorted: the characters each expert receives</h4>
            <CharacterGroups groups={groups} heaviest={Math.max(...PRESENT.map((entry) => entry.total))} />
            <p className="moe-groups__key">A darker tile is a character that starts more pairs.</p>
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
        <ParamBars rows={paramRows} scale={parameterCounts(EXPERTS_MAX, 1).total} spoken={paramSpoken} />
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
                <td>
                  <CellBar value={counts.active / counts.total} max={1} text={percent(counts.active / counts.total)} />
                </td>
                <td>{loss(final.loss)}</td>
              </tr>
              <tr>
                <th scope="row">Dense baseline, 1 expert</th>
                <td>{count(denseCounts.total)}</td>
                <td>{count(denseCounts.active)}</td>
                <td>
                  <CellBar value={1} max={1} text="100%" tone="muted" />
                </td>
                <td>{loss(dense.final.loss)}</td>
              </tr>
              <tr>
                <th scope="row">{MIXTRAL_8X7B.name}, published</th>
                <td>{billions(MIXTRAL_8X7B.total)}</td>
                <td>{billions(MIXTRAL_8X7B.active)}</td>
                <td>
                  <CellBar value={MIXTRAL_8X7B.active / MIXTRAL_8X7B.total} max={1} text={percent(MIXTRAL_8X7B.active / MIXTRAL_8X7B.total)} tone="muted" />
                </td>
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
