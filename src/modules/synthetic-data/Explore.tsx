import { useMemo } from "react";
import {
  encodeTinyText,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  acceptanceOf,
  GENERATIONS,
  glyph,
  GRID_IDS,
  realWordShare,
  runFamily,
  SEED_COUNTS,
  SEED_IDS,
  SEED_WORDS,
  tailLoss,
  teacherUnseenMass,
  verdictTokens,
  TEACHER_EXTRA_TEXT,
  type FilterKind,
  type GenerationStats,
  type Generator,
  type Regime,
} from "./collapse";

const V = 30;
const REGIMES: ReadonlyArray<{ value: Regime; label: string; short: string }> = [
  { value: "replace", label: "Synthetic only", short: "synthetic only" },
  { value: "accumulate", label: "Real + synthetic", short: "real + synthetic" },
];
const MEASURES = [
  {
    value: "support",
    label: "Transitions",
    axis: "transitions with evidence",
    domain: [0, 160] as [number, number],
    ticks: [0, 40, 80, 120, 160],
    format: (value: number) => value.toFixed(0),
  },
  {
    value: "coverage",
    label: "Coverage",
    axis: "seed text still producible (%)",
    domain: [0, 1] as [number, number],
    ticks: [0, 0.25, 0.5, 0.75, 1],
    format: (value: number) => `${(value * 100).toFixed(1)}%`,
  },
  {
    value: "entropy",
    label: "Entropy",
    axis: "next-character entropy (nats)",
    domain: [0, 2] as [number, number],
    ticks: [0, 0.5, 1, 1.5, 2],
    format: (value: number) => value.toFixed(3),
  },
  {
    value: "realWords",
    label: "Real words",
    axis: "real words in its writing (%)",
    domain: [0, 1] as [number, number],
    ticks: [0, 0.25, 0.5, 0.75, 1],
    format: (value: number) => `${(value * 100).toFixed(1)}%`,
  },
  {
    value: "heldOut",
    label: "Unseen text",
    axis: "unseen harbor text it can produce (%)",
    domain: [0, 1] as [number, number],
    ticks: [0, 0.25, 0.5, 0.75, 1],
    format: (value: number) => `${(value * 100).toFixed(1)}%`,
  },
] as const;
type MeasureKey = (typeof MEASURES)[number]["value"];

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pick = <T extends string>(value: string, allowed: readonly T[], fallback: T): T =>
  (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

/* Chart geometry in viewBox units. */
const W = 560;
const H = 246;
const L = 50;
const R = 540;
const T = 24;
const B = 204;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const regime = pick(asString(state, "regime", "replace"), ["replace", "accumulate"], "replace");
  const measure = pick<MeasureKey>(
    asString(state, "measure", "support"),
    ["support", "coverage", "entropy", "realWords", "heldOut"],
    "support",
  );
  const generator = pick<Generator>(asString(state, "generator", "self"), ["self", "teacher"], "self");
  const filter = pick<FilterKind>(asString(state, "filter", "none"), ["none", "words"], "none");
  const temperature = Math.round(clamp(asNumber(state, "temperature", 1), 0.5, 1.2) * 10) / 10;
  const sampleSize = [250, 500, 2000].includes(asNumber(state, "sampleSize", 500))
    ? asNumber(state, "sampleSize", 500)
    : 500;
  const generation = clamp(Math.round(asNumber(state, "generation", 5)), 0, GENERATIONS);

  const families = useMemo(
    () => ({
      replace: runFamily({ regime: "replace", temperature, filter, sampleSize, generator }),
      accumulate: runFamily({ regime: "accumulate", temperature, filter, sampleSize, generator }),
    }),
    [filter, generator, sampleSize, temperature],
  );
  const teacherMass = useMemo(() => (generator === "teacher" ? teacherUnseenMass() : 0), [generator]);

  const baseSpec = MEASURES.find((entry) => entry.value === measure) ?? MEASURES[0];
  // A stronger teacher can lift the transition count past 160, so that axis grows to fit the data.
  const supportPeak = Math.max(
    ...[families.replace, families.accumulate].flatMap((family) =>
      family.runs.flatMap((run) => run.stats.map((point) => point.support)),
    ),
  );
  const supportTop = Math.max(160, Math.ceil(supportPeak / 40) * 40);
  const spec =
    baseSpec.value === "support"
      ? {
          ...baseSpec,
          domain: [0, supportTop] as [number, number],
          ticks: [0, supportTop / 4, supportTop / 2, (supportTop * 3) / 4, supportTop],
        }
      : baseSpec;
  const isShare = measure === "coverage" || measure === "realWords" || measure === "heldOut";
  const value = (stats: GenerationStats) => stats[measure];
  const selected = families[regime];
  const firstRun = selected.runs[0];
  const probe = firstRun.probes[generation].slice(0, 240);
  const tokens = verdictTokens(probe);
  const counts = firstRun.counts[generation];
  const tails = tailLoss(counts);
  const runStats = firstRun.stats[generation];
  const seedStats = selected.mean[0];

  const xAt = (g: number) => L + (g / GENERATIONS) * (R - L);
  const yAt = (v: number) => {
    const [low, high] = spec.domain;
    return B - ((clamp(v, low, high) - low) / (high - low)) * (B - T);
  };
  const path = (stats: ReadonlyArray<GenerationStats>) =>
    stats.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(index).toFixed(1)},${yAt(value(point)).toFixed(1)}`).join(" ");

  const meanReplace = families.replace.mean[generation];
  const meanAccumulate = families.accumulate.mean[generation];
  const chartLabel = `${spec.axis} over ${GENERATIONS} generations at temperature ${temperature.toFixed(1)}, ${
    filter === "words" ? "real-words filter" : "no filter"
  }, ${sampleSize} characters per generation. Mean of eight runs at generation ${generation}: synthetic only ${spec.format(
    value(meanReplace),
  )}, real plus synthetic ${spec.format(value(meanAccumulate))}. Generation 0 is ${spec.format(value(seedStats))}.`;

  const seedTotal = tails.reduce((sum, bucket) => sum + bucket.total, 0);
  const lostTotal = tails.reduce((sum, bucket) => sum + bucket.lost, 0);
  const sampleAcceptance = acceptanceOf(probe);

  return (
    <div className="tg-lab tg-lab--hero a10-lab syn-lab">
      <LabSurface label="Recursive training" className="syn-chart-card">
        <SurfaceHeading
          kicker={`Each generation drafts the next one's training text · ${sampleSize} chars per generation`}
          title="What survives ten rounds of training on model output"
          aside={<span className="tg-badge">8 runs per line</span>}
        />
        <div className="syn-controls-row">
          <div className="syn-regime-control">
            <SegmentedControl
              label="Training data"
              value={regime}
              options={REGIMES.map((entry) => ({ value: entry.value, label: entry.label }))}
              onChange={(next) => {
                setState({ regime: next });
                narrate(
                  next === "replace"
                    ? "Each generation now trains only on the previous generation's text."
                    : "Each generation now trains on the human seed plus every synthetic batch so far.",
                );
              }}
            />
          </div>
          <SegmentedControl
            label="Measure"
            value={measure}
            options={MEASURES.map((entry) => ({ value: entry.value, label: entry.label }))}
            onChange={(next) => setState({ measure: next })}
          />
        </div>
        <svg className="syn-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chartLabel}>
          {spec.ticks.map((tick) => (
            <g key={tick}>
              <line className="syn-grid" x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
              <text className="syn-tick" x={L - 6} y={yAt(tick) + 3} textAnchor="end">
                {isShare ? `${tick * 100}%` : tick}
              </text>
            </g>
          ))}
          {Array.from({ length: GENERATIONS + 1 }, (_, g) => (
            <text key={g} className="syn-tick" x={xAt(g)} y={B + 13} textAnchor="middle">
              {g}
            </text>
          ))}
          <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
          <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
          <line className="syn-marker" x1={xAt(generation)} x2={xAt(generation)} y1={T} y2={B} />
          {(["accumulate", "replace"] as const).map((key) => (
            <g key={key} className={`syn-family syn-family--${key}${key === regime ? " is-selected" : ""}`}>
              {families[key].runs.map((run, index) => (
                <path key={index} className="syn-run" d={path(run.stats)} />
              ))}
              <path className="syn-mean" d={path(families[key].mean)} />
              {families[key].mean.map((point, g) => (
                <circle
                  key={g}
                  className={`syn-dot${g === generation ? " is-current" : ""}`}
                  cx={xAt(g)}
                  cy={yAt(value(point))}
                  r={g === generation ? 4.2 : 2.2}
                />
              ))}
            </g>
          ))}
          <text className="syn-axis-label" x={L} y={H - 4}>
            generation (0 = fitted to the human seed)
          </text>
          <text className="syn-axis-label" x={L - 40} y={11}>
            {spec.axis}
          </text>
        </svg>
        <div className="syn-legend">
          <span className="syn-legend__item syn-legend__item--replace">
            <i />
            Synthetic only <b>{spec.format(value(meanReplace))}</b>
          </span>
          <span className="syn-legend__item syn-legend__item--accumulate">
            <i />
            Real + synthetic <b>{spec.format(value(meanAccumulate))}</b>
          </span>
          <span className="syn-legend__note">
            Thick line: mean of 8 sampling seeds. Thin lines: each seed. Values at generation {generation}.
          </span>
        </div>
        <div className="syn-generation-control">
          <RangeControl
            label="Generation"
            min={0}
            max={GENERATIONS}
            step={1}
            value={generation}
            format={(g) => (g === 0 ? "0 · fitted to the seed" : `${g} of ${GENERATIONS}`)}
            onChange={(next) => {
              setState({ generation: next });
              narrate(
                `Generation ${next}. ${spec.label}: synthetic only ${spec.format(
                  value(families.replace.mean[next]),
                )}, real plus synthetic ${spec.format(value(families.accumulate.mean[next]))}.`,
              );
            }}
          />
        </div>
        <div className="metric-row">
          <Metric label="Seed (generation 0)" value={spec.format(value(seedStats))} />
          <Metric
            label={`Gen ${generation} · synthetic only`}
            value={spec.format(value(meanReplace))}
            tone="loss"
          />
          <Metric
            label={`Gen ${generation} · real + synthetic`}
            value={spec.format(value(meanAccumulate))}
            tone="forward"
          />
        </div>
      </LabSurface>

      <LabSurface label="Which transitions survive" className="syn-grid-card">
        <SurfaceHeading
          kicker={`Run 1 · generation ${generation} · ${REGIMES.find((entry) => entry.value === regime)?.short}`}
          title={`${seedTotal - lostTotal} of the seed's ${seedTotal} transitions remain`}
        />
        <TransitionGrid counts={counts} generation={generation} />
        <div className="syn-tails">
          {tails.map((bucket) => (
            <div key={bucket.id}>
              <span>{bucket.label}</span>
              <div className="syn-tails__bar" aria-hidden="true">
                <i style={{ width: `${bucket.total === 0 ? 0 : (bucket.lost / bucket.total) * 100}%` }} />
              </div>
              <b>
                {bucket.lost} / {bucket.total} lost
              </b>
            </div>
          ))}
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Generator and filter" className="syn-generator-card">
          <SurfaceHeading kicker="How each generation drafts its text" title="Choose the writer, sharpen, verify, and size" />
          <div className="syn-generator-control">
            <SegmentedControl
              label="Generator"
              value={generator}
              options={[
                { value: "self", label: "The model itself" },
                { value: "teacher", label: "A stronger teacher" },
              ]}
              onChange={(next) => {
                setState({ generator: next });
                narrate(
                  next === "teacher"
                    ? "A fixed stronger teacher, trained on about four times as much harbor text, now writes every generation's data."
                    : "Each generation's data is written by the model fitted in the previous round.",
                );
              }}
            />
          </div>
          <RangeControl
            label="Temperature"
            min={0.5}
            max={1.2}
            step={0.1}
            value={temperature}
            format={(t) => `${t.toFixed(1)}${t < 1 ? " · sharper" : t > 1 ? " · flatter" : " · as fitted"}`}
            onChange={(next) => setState({ temperature: Math.round(next * 10) / 10 })}
          />
          <div className="syn-filter-control">
            <SegmentedControl
              label="Filter"
              value={filter}
              options={[
                { value: "none", label: "No filter" },
                { value: "words", label: "Real words only" },
              ]}
              onChange={(next) => {
                setState({ filter: next });
                narrate(
                  next === "words"
                    ? `Verifier on: only tokens that are one of the ${SEED_WORDS.size} words in the seed are kept.`
                    : "Filter off: every drafted character is kept.",
                );
              }}
            />
          </div>
          <SegmentedControl
            label="Sample size"
            value={String(sampleSize)}
            options={[
              { value: "250", label: "250 chars" },
              { value: "500", label: "500 chars" },
              { value: "2000", label: "2000 chars" },
            ]}
            onChange={(next) => setState({ sampleSize: Number(next) })}
          />
          <p className="lab-note">
            Each model is the 30 × 30 bigram table fitted exactly, by counting what follows each
            character. Sampling raises every probability to the power 1/T and renormalizes. The
            verifier keeps only tokens that are one of the {SEED_WORDS.size} words in the seed.
          </p>
          {generator === "teacher" && (
            <p className="lab-note">
              The teacher is a bigram trained with the track's SGD trainer on the {SEED_IDS.length} seed characters plus{" "}
              {encodeTinyText(TEACHER_EXTRA_TEXT).length.toLocaleString("en-US")} characters of extra harbor notes that the student never sees. It is
              fixed, so nothing a student gets wrong can reach the next round. SGD leaves about {(teacherMass * 100).toFixed(1)}%
              of its probability on pairs its data never contained, so a student can list more transitions than the teacher's
              data held.
            </p>
          )}
        </LabSurface>

        <LabSurface label="What a generation writes" className="syn-sample-card">
          <SurfaceHeading
            kicker={`Run 1 of 8 · ${REGIMES.find((entry) => entry.value === regime)?.short} · T = ${temperature.toFixed(1)}`}
            title={generation === 0 ? "The model fitted to the human seed" : `Generation ${generation}'s own writing`}
          />
          <p className="syn-sample" aria-label={`Sample from generation ${generation}: ${probe}`}>
            {filter === "words"
              ? tokens.map((entry, index) => (
                  <span key={index} className={entry.real ? "is-real" : "is-rejected"}>
                    {entry.token}{" "}
                  </span>
                ))
              : probe}
          </p>
          <p className="lab-note">
            {filter === "words"
              ? "Struck-through tokens are what the real-words verifier would throw away before this text trains the next generation."
              : "With no filter, every character of text like this becomes the next generation's training data."}
          </p>
          <div className="metric-row">
            <Metric label="Real words here" value={`${(realWordShare(probe) * 100).toFixed(0)}%`} tone="forward" />
            <Metric
              label="Filter keeps"
              value={filter === "words" ? `${(sampleAcceptance * 100).toFixed(0)}% of chars` : "all · no filter"}
            />
            <Metric
              label="Trained on"
              value={
                generation === 0
                  ? `${SEED_IDS.length} human chars`
                  : regime === "replace"
                    ? `${runStats.trainedChars} synthetic chars`
                    : `${SEED_IDS.length} human + ${firstRun.stats
                        .slice(1, generation + 1)
                        .reduce((sum, point) => sum + point.trainedChars, 0)} synthetic`
              }
            />
          </div>
        </LabSurface>
      </div>
    </div>
  );
}

function TransitionGrid({ counts, generation }: { counts: Float64Array; generation: number }) {
  const cell = 9;
  const pad = 12;
  const size = GRID_IDS.length * cell;
  let kept = 0;
  let lost = 0;
  const cells = [];
  for (let r = 0; r < GRID_IDS.length; r += 1) {
    for (let c = 0; c < GRID_IDS.length; c += 1) {
      const index = GRID_IDS[r] * V + GRID_IDS[c];
      const seed = SEED_COUNTS[index];
      if (seed === 0) continue;
      const alive = counts[index] > 0;
      if (alive) kept += 1;
      else lost += 1;
      const x = pad + c * cell;
      const y = pad + r * cell;
      cells.push(
        alive ? (
          <rect
            key={index}
            className="syn-cell is-kept"
            x={x + 0.5}
            y={y + 0.5}
            width={cell - 1}
            height={cell - 1}
            style={{ opacity: 0.35 + Math.min(1, Math.log(1 + seed) / Math.log(40)) * 0.65 }}
          >
            <title>{`${glyph(GRID_IDS[r])} → ${glyph(GRID_IDS[c])}: seen ${seed}× in the seed, still present`}</title>
          </rect>
        ) : (
          <g key={index} className="syn-cell is-lost">
            <rect x={x + 0.5} y={y + 0.5} width={cell - 1} height={cell - 1} />
            <line x1={x + 1.5} y1={y + cell - 1.5} x2={x + cell - 1.5} y2={y + 1.5} />
            <title>{`${glyph(GRID_IDS[r])} → ${glyph(GRID_IDS[c])}: seen ${seed}× in the seed, lost`}</title>
          </g>
        ),
      );
    }
  }
  return (
    <figure className="syn-grid-figure">
      <svg
        viewBox={`0 0 ${pad + size + 2} ${pad + size + 2}`}
        role="img"
        aria-label={`Transition grid at generation ${generation}: rows are the current character, columns the next. ${kept} seed transitions still have evidence, ${lost} are lost.`}
      >
        {GRID_IDS.map((id, index) => (
          <g key={id}>
            <text className="syn-grid-label" x={pad + index * cell + cell / 2} y={pad - 3} textAnchor="middle">
              {glyph(id)}
            </text>
            <text className="syn-grid-label" x={pad - 3} y={pad + index * cell + cell - 2} textAnchor="end">
              {glyph(id)}
            </text>
          </g>
        ))}
        <rect className="syn-grid-frame" x={pad} y={pad} width={size} height={size} />
        {cells}
      </svg>
      <figcaption>
        <span>
          <i className="syn-key syn-key--kept" /> still present (darker = more common in the seed)
        </span>
        <span>
          <i className="syn-key syn-key--lost" /> lost (crossed)
        </span>
      </figcaption>
    </figure>
  );
}
