import { useMemo, useState } from "react";
import {
  encodeTinyText,
  LabSurface,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  StageFlow,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_SCHEDULES,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  tinyPerplexity,
  trainTinyFactored,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import { BOILERPLATE_LINES, buildScrape, cleanScrape, encodedLength, scrapeText, splitCorpus } from "./data";
import { describeRecipe, formatRecipe, parseRecipe, RECIPE_MAX_LENGTH } from "./recipe";
import {
  BATCH_LADDER,
  CLIP_MAX,
  CLIP_STEP,
  EPOCHS_MAX,
  RATE_LADDER,
  readSettings,
  SEED,
} from "./state";

const corpora = [TINY_CORPORA.harbor, TINY_CORPORA.recipes, TINY_CORPORA.proverbs];

const wizardStages = [
  { id: "corpus", name: "Corpus", detail: "what it reads" },
  { id: "size", name: "Size", detail: "how many weights" },
  { id: "recipe", name: "Recipe", detail: "how it trains" },
  { id: "card", name: "Model card", detail: "what you built" },
];

/** A loss that is not a number means the run diverged, and the page says so instead of printing it. */
const showLoss = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "diverged");
/** Gradient norms can explode by many orders of magnitude, so large ones switch to exponent form. */
const showNorm = (value: number) =>
  !Number.isFinite(value) ? "diverged" : value >= 1000 ? value.toExponential(1) : value.toFixed(2);
/** The losses a curve can draw: everything up to the first value that is not a number. */
const finitePrefix = (values: number[]) => {
  const kept: number[] = [];
  for (const value of values) {
    if (!Number.isFinite(value)) break;
    kept.push(value);
  }
  return kept;
};

export default function Explore({ state, setState, narrate }: ModuleContext) {
  // Defence in depth: hydrateState already validates, but a live state can also arrive from a
  // snapshot or a patch, and both runs below train synchronously.
  const settings = readSettings(state);
  const { wizard, corpusId, rank, epochs, learningRate, schedule, batchSize, clipNorm, data } = settings;
  const [pasted, setPasted] = useState("");
  const [shareNote, setShareNote] = useState("");

  const corpus = corpora.find((entry) => entry.id === corpusId) ?? corpora[0];
  const split = useMemo(() => splitCorpus(corpus.text), [corpus]);
  /** The scrape this lab builds around the training sentences, and what cleaning keeps of it. */
  const scrape = useMemo(() => buildScrape(split.sentences), [split]);
  const cleaning = useMemo(() => cleanScrape(scrape), [scrape]);
  const rawText = useMemo(() => scrapeText(scrape), [scrape]);
  const trainText = data === "raw" ? rawText : cleaning.text;

  const run = useMemo(
    () =>
      trainTinyFactored({
        text: trainText,
        rank,
        epochs,
        batchSize,
        learningRate,
        schedule,
        clipNorm,
        seed: SEED,
        // One checkpoint per epoch boundary, so both curves are exact per-epoch losses.
        checkpoints: epochs + 1,
      }),
    [batchSize, clipNorm, epochs, learningRate, rank, schedule, trainText],
  );

  /**
   * The full weight table from lab 17, trained on identical settings so the
   * only difference between the two runs is how the weights are stored.
   */
  const fullTable = useMemo(
    () =>
      trainTinyModel({
        text: trainText,
        epochs,
        batchSize,
        learningRate,
        schedule,
        clipNorm,
        seed: SEED,
        checkpoints: epochs + 1,
      }),
    [batchSize, clipNorm, epochs, learningRate, schedule, trainText],
  );

  const curves = useMemo(
    () => ({
      factoredTrain: finitePrefix(run.checkpoints.map((point) => point.loss)),
      factoredHeld: finitePrefix(run.checkpoints.map((point) => tinyCrossEntropy(point.weights, split.held))),
      fullTrain: finitePrefix(fullTable.checkpoints.map((point) => point.loss)),
      fullHeld: finitePrefix(fullTable.checkpoints.map((point) => tinyCrossEntropy(point.weights, split.held))),
    }),
    [fullTable, run, split.held],
  );

  const failed = run.diverged;
  const fullFailed = fullTable.diverged;
  const heldLoss = failed ? Number.NaN : tinyCrossEntropy(run.weights, split.held);
  /** How well the model predicts the three footer lines: low means it has learned page furniture. */
  const footerLoss = failed ? Number.NaN : tinyCrossEntropy(run.weights, BOILERPLATE_LINES.join(" "));
  const fullHeldLoss = fullFailed ? Number.NaN : tinyCrossEntropy(fullTable.weights, split.held);
  const factoredAhead = !failed && !fullFailed && run.finalLoss < fullTable.finalLoss;
  /** Samples come from the factored model the wizard is configuring. */
  const ladder = [...new Set([0, Math.floor(epochs / 2), epochs])]
    .map((epoch) => ({ epoch, point: run.checkpoints[epoch] }))
    .filter((entry) => Boolean(entry.point));
  const markerEpochs = ladder.map((entry) => entry.epoch);

  const recipeText = formatRecipe(settings);
  const rateIndex = Math.max(0, RATE_LADDER.indexOf(learningRate as (typeof RATE_LADDER)[number]));
  const batchIndex = Math.max(0, BATCH_LADDER.indexOf(batchSize as (typeof BATCH_LADDER)[number]));
  const pastedRecipe = useMemo(() => parseRecipe(pasted), [pasted]);

  const copyRecipe = async () => {
    try {
      await navigator.clipboard.writeText(recipeText);
      setShareNote("Recipe copied.");
      narrate("Recipe copied to the clipboard.");
    } catch {
      setShareNote("Copying is unavailable in this window. Select the text above and copy it yourself.");
    }
  };

  return (
    <div className="tg-lab tg-lab--hero a10-lab">
      <LabSurface label="Build a model in four choices" className="tg-wizard-panel">
        <SurfaceHeading
          kicker={`Choice ${wizard + 1} of 4`}
          title="Every choice below retrains the model immediately"
          aside={<span className="tg-badge">{run.parameters} trained weights</span>}
        />
        <StageFlow
          label="Wizard stages"
          stages={wizardStages}
          current={wizard}
          onSelect={(index) => setState({ wizard: index })}
        />
        <div className="tg-wizard-choices">
          {corpora.map((entry) => {
            const characters = encodeTinyText(entry.text).length;
            return (
              <button
                type="button"
                key={entry.id}
                aria-pressed={entry.id === corpusId}
                onClick={() => {
                  setState({ corpusId: entry.id, wizard: Math.max(1, wizard) });
                  narrate(`${entry.name}. ${characters} characters of training text.`);
                }}
              >
                <strong>{entry.name}</strong>
                <small>{entry.detail}</small>
                <b>{characters} chars</b>
              </button>
            );
          })}
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Training progress and samples" className="tg-loss-panel">
          <SurfaceHeading
            kicker={`${run.parameters} weights against ${fullTable.tokens} pairs`}
            title="Loss falling, and what the model writes as it falls"
          />
          <CurvePair curves={curves} epochs={epochs} rank={rank} markers={markerEpochs} />
          <div className="tl-legend">
            <span className="tl-legend__item tl-legend__item--train">
              <i /> factored rank {rank}, training split <b>{showLoss(run.finalLoss)}</b>
            </span>
            <span className="tl-legend__item tl-legend__item--held">
              <i /> factored rank {rank}, held out <b>{showLoss(heldLoss)}</b>
            </span>
            <span className="tl-legend__item tl-legend__item--full-train">
              <i /> full table, training split <b>{showLoss(fullTable.finalLoss)}</b>
            </span>
            <span className="tl-legend__item tl-legend__item--full-held">
              <i /> full table, held out <b>{showLoss(fullHeldLoss)}</b>
            </span>
          </div>
          <div className="tg-sample-ladder">
            {ladder.map(({ epoch, point }, index) => (
              <div className="tg-sample" key={epoch}>
                <span>
                  {String.fromCharCode(65 + index)} · {epoch === 0 ? "before training" : `epoch ${epoch}, step ${point.step}`} ·
                  loss {point.loss.toFixed(2)}
                </span>
                <p>
                  {sampleTinyText(point.weights, {
                    prompt: "the ",
                    length: 78,
                    temperature: 0.7,
                    seed: 31,
                  })}
                </p>
              </div>
            ))}
          </div>
        </LabSurface>

        <LabSurface label="Training recipe" className="tl-recipe-panel">
          <SurfaceHeading
            kicker="Text, rate, schedule, batch, clip"
            title="The recipe is part of this lab's state"
            aside={<span className="tg-badge">{run.steps} steps</span>}
          />
          <div className="tl-recipe-controls">
            <SegmentedControl
              label="Training text"
              value={data}
              options={[
                { value: "cleaned", label: "Cleaned scrape" },
                { value: "raw", label: "Raw scrape" },
              ]}
              onChange={(value) => {
                setState({ data: value, wizard: Math.max(2, wizard) });
                narrate(
                  value === "raw"
                    ? `Raw scrape: ${scrape.length} lines, ${encodedLength(rawText)} characters, boilerplate and repeats included.`
                    : `Cleaned scrape: ${cleaning.kept.length} lines, ${encodedLength(cleaning.text)} characters.`,
                );
              }}
            />
            <p className="lab-note tl-recipe-funnel">
              The scrape is {scrape.length} lines and {encodedLength(rawText)} characters. Cleaning removes{" "}
              {cleaning.removed.boilerplate} boilerplate, {cleaning.removed.short} too-short,{" "}
              {cleaning.removed.duplicate} repeated and {cleaning.removed.near} near-repeated lines, which keeps{" "}
              {cleaning.kept.length} lines and {encodedLength(cleaning.text)} characters.
            </p>
            <RangeControl
              label="Learning rate"
              min={0}
              max={RATE_LADDER.length - 1}
              step={1}
              value={rateIndex}
              format={(value) => `${RATE_LADDER[value]}`}
              onChange={(value) => {
                setState({ learningRate: RATE_LADDER[value], wizard: Math.max(2, wizard) });
                narrate(`Learning rate ${RATE_LADDER[value]}.`);
              }}
            />
            <SegmentedControl
              label="Schedule"
              value={schedule}
              options={TINY_SCHEDULES.map((entry) => ({ value: entry.value, label: entry.label }))}
              onChange={(value) => {
                setState({ schedule: value, wizard: Math.max(2, wizard) });
                narrate(`${TINY_SCHEDULES.find((entry) => entry.value === value)?.detail ?? ""}`);
              }}
            />
            <RangeControl
              label="Batch size"
              min={0}
              max={BATCH_LADDER.length - 1}
              step={1}
              value={batchIndex}
              format={(value) => `${BATCH_LADDER[value]} pairs`}
              onChange={(value) => {
                setState({ batchSize: BATCH_LADDER[value], wizard: Math.max(2, wizard) });
                narrate(`Batch size ${BATCH_LADDER[value]}.`);
              }}
            />
            <RangeControl
              label="Gradient clip"
              min={0}
              max={CLIP_MAX}
              step={CLIP_STEP}
              value={clipNorm}
              format={(value) => (value > 0 ? value.toFixed(2) : "off")}
              onChange={(value) => {
                setState({ clipNorm: Number(value.toFixed(2)), wizard: Math.max(2, wizard) });
                narrate(value > 0 ? `Gradient clip ${value.toFixed(2)}.` : "Gradient clipping off.");
              }}
            />
          </div>
          <div className="metric-row">
            <Metric label="Held out" value={showLoss(heldLoss)} tone="loss" />
            <Metric label="Footer lines" value={showLoss(footerLoss)} />
            <Metric label="Steps" value={`${run.steps}`} />
          </div>
          <div className="metric-row">
            <Metric label="Peak gradient norm" value={showNorm(run.peakGradientNorm)} tone="gradient" />
            <Metric
              label="Clipped steps"
              value={clipNorm > 0 ? `${run.clippedSteps} of ${run.steps}` : "clip off"}
            />
          </div>
          {failed && (
            <div className="tg-callout is-warning" role="status">
              <strong>
                This run diverged at step {run.steps}: the loss stopped being a number, so training halted.
              </strong>
              <span>
                A factored model multiplies two matrices, so each one&rsquo;s gradient grows with the other and a
                large step can feed on itself. Set Gradient clip to 0.50, or lower the learning rate, and the same run
                finishes.
              </span>
            </div>
          )}
          <div className="tl-recipe-share">
            <label className="tg-editor">
              <span>My recipe, as text</span>
              <textarea
                readOnly
                rows={3}
                value={recipeText}
                aria-label="My recipe, as text"
                onFocus={(event) => event.currentTarget.select()}
              />
            </label>
            <button type="button" className="quiet-action" onClick={() => void copyRecipe()}>
              Copy my recipe
            </button>
            <label className="tg-editor">
              <span>Load a recipe someone sent you</span>
              <textarea
                rows={2}
                maxLength={RECIPE_MAX_LENGTH}
                value={pasted}
                aria-label="Paste a recipe to load"
                onChange={(event) => setPasted(event.target.value.slice(0, RECIPE_MAX_LENGTH))}
              />
            </label>
            <button
              type="button"
              className="quiet-action"
              disabled={!pastedRecipe}
              onClick={() => {
                if (!pastedRecipe) return;
                setState({ ...pastedRecipe, wizard: 2 });
                setShareNote(`Loaded ${Object.keys(pastedRecipe).length} settings from the pasted recipe.`);
                narrate("Recipe loaded.");
              }}
            >
              Load this recipe
            </button>
            <p className="lab-note" role="status">
              {shareNote ||
                "Every setting above is saved in this lab's state, so the Copy a link to this state button in the header shares the whole recipe along with the rest of your work."}
            </p>
          </div>
        </LabSurface>
      </div>

      <div className="tg-column">
        <LabSurface label="Size and epochs" className="tg-size-panel">
          <SurfaceHeading kicker="Size and budget" title="Capacity, then how long" />
          <RangeControl
            label="Model size"
            min={1}
            max={12}
            step={1}
            value={rank}
            format={(value) => `rank ${value} · ${2 * value * TINY_VOCAB_SIZE} weights`}
            onChange={(value) => {
              setState({ rank: value, wizard: Math.max(1, wizard) });
              narrate(`Rank ${value}, which is ${2 * value * TINY_VOCAB_SIZE} trainable weights.`);
            }}
          />
          <RangeControl
            label="Epochs"
            min={1}
            max={EPOCHS_MAX}
            step={1}
            value={epochs}
            format={(value) => `${value} passes`}
            onChange={(value) => setState({ epochs: value, wizard: Math.max(2, wizard) })}
          />
          <div className={`tg-callout ${factoredAhead ? "is-quiet" : ""}`}>
            <strong>
              {failed
                ? "The factored run diverged, so there is nothing to compare yet."
                : fullFailed
                  ? "The full-table run diverged, so there is nothing to compare yet."
                  : factoredAhead
                    ? `The ${run.parameters}-weight model is currently fitting better than the ${TINY_VOCAB_SIZE * TINY_VOCAB_SIZE}-weight one.`
                    : `The full table is ahead by ${(run.finalLoss - fullTable.finalLoss).toFixed(3)} nats.`}
            </strong>
            <span>
              {failed || fullFailed
                ? "A run that diverges stops at the last finite step. Lower the learning rate or turn on Gradient clip on the recipe card and the comparison returns."
                : factoredAhead
                  ? `Both runs use the same data, batch size, learning rate, schedule, and seed. One example moves only its own 30-weight row of the full table, but in the factored model it moves its row of the input factor plus the whole ${rank} × 30 output factor that every character shares: ${31 * rank} weights. At this data scale that sharing gets more out of each example. Parameter count and capacity are not the same thing.`
                  : `Both runs use the same data, batch size, learning rate, schedule, and seed. At rank ${rank} every row of scores is a mix of only ${rank} shared pattern${rank === 1 ? "" : "s"}, so the factored model cannot express the different rows the data needs and its loss flattens out. Here the missing thing is capacity, not steps.`}
            </span>
          </div>
          <p className="lab-note">
            Rank {rank} keeps {2 * rank * TINY_VOCAB_SIZE} weights where the full table keeps{" "}
            {TINY_VOCAB_SIZE * TINY_VOCAB_SIZE}. Past rank {Math.floor(TINY_VOCAB_SIZE / 2)} the
            factorization stops saving anything at all.
          </p>
        </LabSurface>

        <LabSurface label="Model card" className="tg-model-card">
          <SurfaceHeading kicker="What you built" title="Model card" />
          <dl>
            <div>
              <dt>Corpus</dt>
              <dd>
                {corpus.name} · {encodedLength(trainText)} training characters
                {data === "raw" ? " (raw scrape)" : ""}, {encodedLength(split.held)} held out
              </dd>
            </div>
            <div>
              <dt>Architecture</dt>
              <dd>
                Character bigram, rank {rank} factorization, {run.parameters} weights
              </dd>
            </div>
            <div>
              <dt>Recipe</dt>
              <dd>
                {describeRecipe(settings)} · {fullTable.steps} steps
              </dd>
            </div>
            <div>
              <dt>Training loss</dt>
              <dd>{failed ? "diverged" : `${showLoss(run.finalLoss)} nats/token`}</dd>
            </div>
            <div>
              <dt>Held-out loss</dt>
              <dd>
                {failed
                  ? "diverged"
                  : `${showLoss(heldLoss)} nats/token · perplexity ${tinyPerplexity(run.weights, split.held).toFixed(2)}`}
              </dd>
            </div>
            <div>
              <dt>Full table for comparison</dt>
              <dd>
                {showLoss(fullTable.finalLoss)} training, {showLoss(fullHeldLoss)} held out
              </dd>
            </div>
          </dl>
          <div className="metric-row">
            <Metric label="Train" value={showLoss(run.finalLoss)} tone="forward" />
            <Metric label="Held out" value={showLoss(heldLoss)} tone="loss" />
            <Metric label="Gap" value={failed ? "diverged" : (heldLoss - run.finalLoss).toFixed(3)} />
          </div>
        </LabSurface>
      </div>
    </div>
  );
}

type Curves = {
  factoredTrain: number[];
  factoredHeld: number[];
  fullTrain: number[];
  fullHeld: number[];
};

function CurvePair({
  curves,
  epochs,
  rank,
  markers,
}: {
  curves: Curves;
  epochs: number;
  rank: number;
  markers: number[];
}) {
  const W = 440;
  const H = 250;
  const L = 40;
  const R = 432;
  const T = 22;
  const B = 222;
  const values = [...curves.factoredTrain, ...curves.factoredHeld, ...curves.fullTrain, ...curves.fullHeld];
  const low = Math.max(0, Math.floor((Math.min(...values) - 0.1) * 5) / 5);
  const high = UNIFORM_CROSS_ENTROPY + 0.1;
  const xAt = (epoch: number) => L + (epoch / Math.max(1, epochs)) * (R - L);
  const yAt = (loss: number) => B - ((Math.min(high, Math.max(low, loss)) - low) / (high - low)) * (B - T);
  const path = (series: number[]) =>
    series.map((loss, epoch) => `${epoch === 0 ? "M" : "L"}${xAt(epoch).toFixed(1)},${yAt(loss).toFixed(1)}`).join(" ");
  const gapArea = `${path(curves.factoredTrain)} ${curves.factoredHeld
    .map((loss, epoch) => ({ loss, epoch }))
    .reverse()
    .map(({ loss, epoch }) => `L${xAt(epoch).toFixed(1)},${yAt(loss).toFixed(1)}`)
    .join(" ")} Z`;
  const ticks = [low, (low + high) / 2, high];
  const epochTicks = Array.from(new Set([0, Math.round(epochs / 4), Math.round(epochs / 2), Math.round((3 * epochs) / 4), epochs]));
  /** A run that diverged has fewer points than epochs; each curve ends at its own last finite one. */
  const final = (series: number[]) => series[series.length - 1];
  const lastEpoch = curves.factoredTrain.length - 1;
  const cut = (series: number[]) =>
    series.length <= epochs ? ` The curve stops at epoch ${series.length - 1}, where the run diverged.` : "";
  return (
    <svg
      className="tl-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Loss per epoch on the training split and on the held-out sentences. Factored rank ${rank}: ${final(
        curves.factoredTrain,
      ).toFixed(3)} training, ${final(curves.factoredHeld).toFixed(3)} held out.${cut(curves.factoredTrain)} Full table: ${final(
        curves.fullTrain,
      ).toFixed(3)} training, ${final(curves.fullHeld).toFixed(3)} held out.${cut(curves.fullTrain)} Uniform baseline ${UNIFORM_CROSS_ENTROPY.toFixed(3)}.`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="tl-grid" x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
          <text className="tl-tick" x={L - 5} y={yAt(tick) + 3} textAnchor="end">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {epochTicks.map((epoch) => (
        <text key={epoch} className="tl-tick" x={xAt(epoch)} y={B + 13} textAnchor="middle">
          {epoch}
        </text>
      ))}
      <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
      <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
      <line className="tl-baseline" x1={L} x2={R} y1={yAt(UNIFORM_CROSS_ENTROPY)} y2={yAt(UNIFORM_CROSS_ENTROPY)} />
      <text className="tl-note" x={R - 2} y={yAt(UNIFORM_CROSS_ENTROPY) - 4} textAnchor="end">
        knows nothing · ln 30 = {UNIFORM_CROSS_ENTROPY.toFixed(3)}
      </text>
      <path className="tl-gap" d={gapArea} />
      <path className="tl-line tl-line--full-train" d={path(curves.fullTrain)} />
      <path className="tl-line tl-line--full-held" d={path(curves.fullHeld)} />
      <path className="tl-line tl-line--held" d={path(curves.factoredHeld)} />
      <path className="tl-line tl-line--train" d={path(curves.factoredTrain)} />
      {markers.filter((epoch) => epoch < curves.factoredTrain.length).map((epoch, index) => (
        <g key={epoch} className="tl-marker">
          <circle cx={xAt(epoch)} cy={yAt(curves.factoredTrain[epoch])} r={7} />
          <text x={xAt(epoch)} y={yAt(curves.factoredTrain[epoch]) + 3} textAnchor="middle">
            {String.fromCharCode(65 + index)}
          </text>
        </g>
      ))}
      <text
        className="tl-note tl-note--gap"
        x={xAt(lastEpoch) - 12}
        y={Math.min(yAt(final(curves.factoredHeld)), yAt(final(curves.factoredTrain))) - 9}
        textAnchor="end"
      >
        gap {(final(curves.factoredHeld) - final(curves.factoredTrain)).toFixed(3)}
      </text>
      <text className="tl-tick" x={L} y={H - 4}>
        epoch
      </text>
      <text className="tl-tick" x={4} y={11}>
        loss (nats/token)
      </text>
    </svg>
  );
}
