import { useMemo } from "react";
import {
  encodeTinyText,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  tinyCrossEntropy,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  COPY_WINDOW,
  CONTEXT_MAX,
  CONTEXT_MIN,
  copiedShare,
  footerShare,
  recitationReport,
  SAMPLE_LENGTH,
  SAMPLE_SEED,
  sampleRecitation,
} from "./recite";
import {
  CONTAMINATION_N,
  funnel,
  jaccard,
  longestSharedSpan,
  MINHASH_SIZE,
  minhashEstimate,
  NEAR_THRESHOLD,
  runPipeline,
  scrapeLines,
  trigramStats,
  UNLEAKED_TEXT,
  VALIDATION_TEXT,
  VARIANT_PAIR,
  type FunnelRow,
  type LineVerdict,
  type PipelineOptions,
} from "./clean";

const verdictLabel: Record<LineVerdict, string> = {
  kept: "kept",
  boilerplate: "boilerplate",
  short: "too short",
  duplicate: "exact repeat",
  near: "near-duplicate",
  contaminated: "held-out overlap",
};

const TRAINING = { epochs: 25, batchSize: 16, learningRate: 0.6, seed: 4, checkpoints: 7 } as const;

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asBoolean = (state: ModuleContext["state"], key: string, fallback: boolean) =>
  typeof state[key] === "boolean" ? state[key] : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const options: PipelineOptions = {
    dropBoilerplate: asBoolean(state, "dropBoilerplate", true),
    dropExact: asBoolean(state, "dropExact", true),
    dropNear: asBoolean(state, "dropNear", false),
    decontaminate: asBoolean(state, "decontaminate", false),
    leak: asBoolean(state, "leak", false),
    minLength: Math.max(0, Math.min(60, Math.round(asNumber(state, "minLength", 0)))),
  };
  const { dropBoilerplate, dropExact, dropNear, decontaminate, leak, minLength } = options;
  const view = asString(state, "view", "raw") === "clean" ? "clean" : "raw";
  const context = Math.max(CONTEXT_MIN, Math.min(CONTEXT_MAX, Math.round(asNumber(state, "context", 3))));

  const results = useMemo(
    () => runPipeline({ dropBoilerplate, dropExact, dropNear, decontaminate, leak, minLength }),
    [decontaminate, dropBoilerplate, dropExact, dropNear, leak, minLength],
  );
  const rows = funnel(results, options);
  const scrape = scrapeLines(leak);
  const cleanedLines = results.filter((entry) => entry.verdict === "kept").map((entry) => entry.line);
  const cleanedText = cleanedLines.join(" ");
  const rawText = scrape.map((entry) => entry.line).join(" ");
  const shownLines = results
    .map((entry, index) => ({ ...entry, index }))
    .filter((entry) => view === "raw" || entry.verdict === "kept");

  const quality = useMemo(
    () => ({ raw: trigramStats(rawText), clean: trigramStats(cleanedText) }),
    [cleanedText, rawText],
  );

  const runs = useMemo(
    () => ({
      raw: trainTinyModel({ text: rawText, ...TRAINING }),
      clean: trainTinyModel({ text: cleanedText || rawText, ...TRAINING }),
    }),
    [cleanedText, rawText],
  );

  // A count-based n-gram on each training text: it is a table of counts, not a trained model, so it is cheap.
  const recital = useMemo(
    () => ({ raw: recitationReport(rawText, context), clean: recitationReport(cleanedText || rawText, context) }),
    [cleanedText, context, rawText],
  );
  const recitalSweep = useMemo(
    () =>
      [1, 2, 3, 4, 5, 6, 8, 12].map((length) => {
        const raw = sampleRecitation(rawText, length);
        const clean = sampleRecitation(cleanedText || rawText, length);
        return {
          length,
          rawCopied: copiedShare(raw.text, rawText),
          cleanCopied: copiedShare(clean.text, cleanedText || rawText),
          rawFooter: footerShare(raw.text),
        };
      }),
    [cleanedText, rawText],
  );

  const validationSeries = (checkpoints: typeof runs.raw.checkpoints) =>
    checkpoints.map((point, index) => ({
      x: index / Math.max(1, checkpoints.length - 1),
      y: tinyCrossEntropy(point.weights, VALIDATION_TEXT),
    }));

  const rawValidation = tinyCrossEntropy(runs.raw.weights, VALIDATION_TEXT);
  const cleanValidation = tinyCrossEntropy(runs.clean.weights, VALIDATION_TEXT);
  const cleanUntouched = tinyCrossEntropy(runs.clean.weights, UNLEAKED_TEXT);
  const shared = longestSharedSpan(cleanedText, VALIDATION_TEXT);
  const contaminated = shared.length >= CONTAMINATION_N;
  const trainShare = Math.round(cleanedLines.length * 0.7);
  const validationShare = Math.max(1, Math.round(cleanedLines.length * 0.15));
  const testShare = Math.max(0, cleanedLines.length - trainShare - validationShare);
  const variantJaccard = jaccard(VARIANT_PAIR[0], VARIANT_PAIR[1]);
  const variantMinhash = minhashEstimate(VARIANT_PAIR[0], VARIANT_PAIR[1], MINHASH_SIZE);

  const removedBy = (verdict: LineVerdict) => results.filter((entry) => entry.verdict === verdict).length;
  const stages = [
    {
      id: "boilerplate",
      name: "Strip boilerplate",
      detail: "Removes lines containing site furniture: copyright, subscribe, share, click here.",
      active: dropBoilerplate,
      removed: removedBy("boilerplate"),
      toggle: () => setState({ dropBoilerplate: !dropBoilerplate }),
    },
    {
      id: "exact",
      name: "Drop exact duplicates",
      detail: "Keeps only the first copy of a byte-identical line.",
      active: dropExact,
      removed: removedBy("duplicate"),
      toggle: () => setState({ dropExact: !dropExact }),
    },
    {
      id: "near",
      name: "Drop near-duplicates",
      detail: `Removes lines whose character-trigram Jaccard with a kept line is at least ${NEAR_THRESHOLD}.`,
      active: dropNear,
      removed: removedBy("near"),
      toggle: () => setState({ dropNear: !dropNear }),
    },
    {
      id: "decontaminate",
      name: "Decontaminate",
      detail: `Removes lines sharing any ${CONTAMINATION_N}-word run with the held-out text.`,
      active: decontaminate,
      removed: removedBy("contaminated"),
      toggle: () => setState({ decontaminate: !decontaminate }),
    },
  ];

  return (
    <div className="tg-lab tg-lab--hero a10-lab">
      <LabSurface label="Source documents and cleaning verdicts" className="tg-source-panel">
        <SurfaceHeading
          kicker={`${scrape.length} scraped lines · ${results.length - cleanedLines.length} removed`}
          title="Every line, and the stage that removed it"
          aside={
            <SegmentedControl
              label="Show"
              value={view}
              options={[
                { value: "raw", label: "All lines" },
                { value: "clean", label: "Kept only" },
              ]}
              onChange={(value) => setState({ view: value })}
            />
          }
        />
        <button
          type="button"
          className="db-leak-toggle"
          aria-pressed={leak}
          onClick={() => {
            setState({ leak: !leak });
            narrate(
              leak
                ? "The copied held-out sentence is out of the scrape."
                : "One held-out sentence is now copied into the scrape, the way a quoted benchmark item lands in a crawl.",
            );
          }}
        >
          <strong>Leak a held-out sentence into the scrape</strong>
          <small>
            Copies “the fog returns in the morning and the pattern repeats.” into the page, as a quoted test
            item ends up in a crawl.
          </small>
          <b>{leak ? "leaked" : "off"}</b>
        </button>
        <ol className="tg-lines">
          {shownLines.map((entry) => (
            <li
              key={`${entry.line}-${entry.index}`}
              className={`is-${entry.verdict}${entry.leaked ? " is-leaked" : ""}`}
            >
              <span>{entry.line}</span>
              <i>
                {verdictLabel[entry.verdict]}
                {entry.match ? ` · J ${entry.match.similarity.toFixed(2)}` : ""}
                {entry.leaked && entry.verdict === "kept" ? " · leaked" : ""}
              </i>
              <b>{entry.line.length}</b>
            </li>
          ))}
        </ol>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Validation loss for both datasets" className="tg-compare-panel">
          <SurfaceHeading
            kicker="Same model, same epochs, same held-out text"
            title="What the cleaning actually bought"
          />
          <LineChart
            label="Held-out validation loss during training"
            xLabel="training progress"
            yLabel="held-out loss (nats/token)"
            yDomain={[1.5, UNIFORM_CROSS_ENTROPY + 0.1]}
            series={[
              {
                id: "raw",
                name: `raw (${runs.raw.tokens} pairs)`,
                tone: "loss",
                dash: "dashed",
                points: validationSeries(runs.raw.checkpoints),
              },
              {
                id: "clean",
                name: `cleaned (${runs.clean.tokens} pairs)`,
                tone: "forward",
                points: validationSeries(runs.clean.checkpoints),
              },
            ]}
            footnote="Both runs use 25 epochs on their own dataset and are scored on the same three held-out sentences, which are in neither training set unless you leak one."
          />
          <div className="metric-row">
            <Metric label="Raw held-out" value={`${rawValidation.toFixed(3)} nats`} tone="loss" />
            <Metric label="Cleaned held-out" value={`${cleanValidation.toFixed(3)} nats`} tone="forward" />
            <Metric
              label="Difference"
              value={`${cleanValidation < rawValidation ? "−" : "+"}${Math.abs(rawValidation - cleanValidation).toFixed(3)}`}
            />
          </div>
          <div className={`db-contamination${contaminated ? " is-flagged" : ""}`}>
            <strong>
              Contamination check · longest word run shared with the held-out text: {shared.length}{" "}
              {shared.length === 1 ? "word" : "words"}
              {contaminated ? ` · flagged at ${CONTAMINATION_N}+` : ` · below the ${CONTAMINATION_N}-word flag`}
            </strong>
            <span>“{shared.span}”</span>
            {leak && (
              <span>
                Cleaned model on the two sentences the leak did not copy: {cleanUntouched.toFixed(3)} nats.
              </span>
            )}
          </div>
        </LabSurface>

        <LabSurface label="Survival funnel" className="db-funnel-card">
          <SurfaceHeading
            kicker={`${rows[0].lines} lines in · ${rows[rows.length - 1].lines} lines out`}
            title="How much of the scrape survives each stage"
          />
          <Funnel rows={rows} />
          <p className="lab-note">
            Bars are lines (documents) left after each stage, in the order the pipeline runs them; the hatched
            ghost is what that stage removed. Characters are counted after encoding into the model's 30-symbol
            vocabulary, which is what the trainer sees.
          </p>
        </LabSurface>
      </div>

      <div className="tg-column">
        <LabSurface label="Cleaning stages" className="tg-clean-panel">
          <SurfaceHeading kicker="Pipeline" title="Toggle a stage and watch the verdicts move" />
          <div className="tg-switch-list">
            {stages.map((stage) => (
              <button
                type="button"
                key={stage.id}
                className="tg-switch"
                aria-pressed={stage.active}
                onClick={() => {
                  stage.toggle();
                  narrate(`${stage.name} ${stage.active ? "off" : "on"}.`);
                }}
              >
                <strong>{stage.name}</strong>
                <small>{stage.detail}</small>
                <b>{stage.active ? `−${stage.removed} lines` : "off"}</b>
              </button>
            ))}
          </div>
          <RangeControl
            label="Minimum line length"
            min={0}
            max={60}
            step={1}
            value={minLength}
            format={(value) => (value === 0 ? "off" : `${value} chars`)}
            onChange={(value) => setState({ minLength: value })}
          />
          <p className="lab-note">
            harbour vs harbor line: exact trigram Jaccard {variantJaccard.toFixed(3)}. A {MINHASH_SIZE}-hash MinHash
            signature estimates {variantMinhash.toFixed(3)} without comparing the sets directly, which is how
            near-duplicate search scales to billions of documents.
          </p>
        </LabSurface>

        <LabSurface label="Dataset statistics" className="tg-quality-panel">
          <SurfaceHeading kicker="Counts" title="Size, diversity, and the split" />
          <div className="metric-row db-metric-row">
            <Metric label="Lines kept" value={`${cleanedLines.length} / ${scrape.length}`} />
            <Metric label="Characters" value={`${encodeTinyText(cleanedText).length}`} />
            <Metric label="Training pairs" value={`${runs.clean.tokens}`} />
          </div>
          <div className="metric-row db-metric-row">
            <Metric label="Distinct trigrams" value={`${quality.clean.distinct}`} tone="forward" />
            <Metric
              label="Repetition"
              value={`${((1 - quality.clean.ratio) * 100).toFixed(0)}%`}
              tone={quality.clean.ratio < quality.raw.ratio ? "loss" : "forward"}
            />
            <Metric label="Raw repetition" value={`${((1 - quality.raw.ratio) * 100).toFixed(0)}%`} />
          </div>
          <div
            className="tg-split-bar"
            role="img"
            aria-label={`Split: ${trainShare} train, ${validationShare} validation, ${testShare} test lines`}
          >
            <i className="is-train" style={{ flexGrow: Math.max(1, trainShare) }}>
              train {trainShare}
            </i>
            <i className="is-validation" style={{ flexGrow: Math.max(1, validationShare) }}>
              val {validationShare}
            </i>
            <i className="is-test" style={{ flexGrow: Math.max(1, testShare) }}>
              test {testShare}
            </i>
          </div>
          <p className="lab-note">
            The split is drawn by line, so no sentence appears in two places. The held-out text scored on the
            chart is separate again: it never enters either training set unless you leak it, which is the only
            reason the comparison means anything.
          </p>
        </LabSurface>
      </div>

      <LabSurface label="Regurgitation probe" className="db-probe-card">
        <SurfaceHeading
          kicker={`A count-based character model · ${SAMPLE_LENGTH} characters sampled · seed ${SAMPLE_SEED}`}
          title="Give a model more context and it starts reciting its training text"
          aside={<span className="tg-badge">{context === 1 ? "1 character of context" : `${context} characters of context`}</span>}
        />
        <div className="db-probe-control">
          <RangeControl
            label="Context length"
            min={CONTEXT_MIN}
            max={CONTEXT_MAX}
            step={1}
            value={context}
            format={(value) => (value === 1 ? "1 character · the bigram" : `${value} characters`)}
            onChange={(value) => {
              setState({ context: value });
              narrate(
                `Context length ${value}. Copied verbatim: raw ${(copiedShare(sampleRecitation(rawText, value).text, rawText) * 100).toFixed(
                  0,
                )} percent, cleaned ${(
                  copiedShare(sampleRecitation(cleanedText || rawText, value).text, cleanedText || rawText) * 100
                ).toFixed(0)} percent.`,
              );
            }}
          />
        </div>
        <div className="db-probe-grid">
          {(
            [
              { id: "raw", title: "Trained on the raw scrape", report: recital.raw, tone: "loss" },
              { id: "clean", title: "Trained on the cleaned set", report: recital.clean, tone: "forward" },
            ] as const
          ).map((entry) => (
            <div key={entry.id} className="db-probe-column">
              <h4>{entry.title}</h4>
              <p className="db-probe-sample" aria-label={`Sample from the model trained on the ${entry.id === "raw" ? "raw scrape" : "cleaned set"}`}>
                {entry.report.text.slice(0, 170)}
              </p>
              <div className="metric-row">
                <Metric label={`Copied verbatim (${COPY_WINDOW}+ chars)`} value={`${(entry.report.copied * 100).toFixed(0)}%`} tone={entry.tone} />
                <Metric label="Longest copied run" value={`${entry.report.longest} chars`} />
                <Metric label="Footer text in the sample" value={`${(entry.report.footer * 100).toFixed(0)}%`} />
              </div>
              <p className="lab-note">
                {entry.report.forcedContexts} of {entry.report.contexts} contexts have only one possible next character, so
                there the model has no choice but to continue the text it saw.
              </p>
            </div>
          ))}
        </div>
        <div className="db-probe-table-wrap">
          <table className="db-probe-table">
            <caption>
              The same sweep at eight context lengths: the share of each sample copied verbatim from its own training text
              (runs of {COPY_WINDOW} or more characters), and how much of the raw model's sample is footer text.
            </caption>
            <thead>
              <tr>
                <th scope="col">Context length</th>
                <th scope="col">Raw copied</th>
                <th scope="col">Cleaned copied</th>
                <th scope="col">Raw footer text</th>
              </tr>
            </thead>
            <tbody>
              {recitalSweep.map((row) => (
                <tr key={row.length} className={row.length === context ? "is-selected" : undefined}>
                  <th scope="row">{row.length}</th>
                  <td>{(row.rawCopied * 100).toFixed(0)}%</td>
                  <td>{(row.cleanCopied * 100).toFixed(0)}%</td>
                  <td>{(row.rawFooter * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          The model is a table of counts: every run of {context} {context === 1 ? "character" : "characters"} in the
          training text records which character followed it, and sampling draws in proportion. It starts from the first
          characters of its own training text, the way an extraction attempt starts from a prefix. At one character of
          context it is the track's bigram and recites nothing.
        </p>
      </LabSurface>
    </div>
  );
}

function Funnel({ rows }: { rows: ReadonlyArray<FunnelRow> }) {
  const rowHeight = 32;
  const top = 4;
  const barLeft = 96;
  const barWidth = 132;
  const center = barLeft + barWidth / 2;
  const max = Math.max(1, rows[0].lines);
  const height = top + rows.length * rowHeight + 4;
  const summary = rows
    .map((row) =>
      row.id === "scraped"
        ? `${row.lines} lines scraped`
        : row.active
          ? `${row.label} removes ${row.removedLines}, leaving ${row.lines}`
          : `${row.label} off`,
    )
    .join("; ");
  return (
    <svg className="db-funnel" viewBox={`0 0 380 ${height}`} role="img" aria-label={`Survival funnel: ${summary}.`}>
      <defs>
        <pattern id="db-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" className="db-hatch-line" />
        </pattern>
      </defs>
      {rows.map((row, index) => {
        const y = top + index * rowHeight;
        const previous = index === 0 ? row.lines : rows[index - 1].lines;
        const before = (previous / max) * barWidth;
        const after = (row.lines / max) * barWidth;
        return (
          <g key={row.id} className={`db-funnel-row${row.active ? "" : " is-off"}`}>
            <text className="db-funnel-label" x={0} y={y + 20}>
              {row.label}
            </text>
            {row.active && row.removedLines > 0 && (
              <rect
                className="db-funnel-ghost"
                x={center - before / 2}
                y={y + 5}
                width={before}
                height={rowHeight - 10}
                rx={3}
              />
            )}
            <rect
              className={`db-funnel-bar${row.id === "scraped" ? " is-source" : ""}`}
              x={center - after / 2}
              y={y + 5}
              width={Math.max(1, after)}
              height={rowHeight - 10}
              rx={3}
            />
            <text className="db-funnel-value" x={380} y={y + 14} textAnchor="end">
              {row.lines} lines · {row.chars} chars
            </text>
            <text
              className={`db-funnel-delta${row.active && row.removedLines > 0 ? " is-removed" : ""}`}
              x={380}
              y={y + 27}
              textAnchor="end"
            >
              {row.id === "scraped"
                ? "the raw page"
                : !row.active
                  ? "stage off"
                  : row.removedLines === 0
                    ? "removes nothing here"
                    : `−${row.removedLines} lines · −${row.removedChars} chars`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
