import { useMemo } from "react";
import {
  BarList,
  encodeTinyText,
  LabSurface,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  StageFlow,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_VOCAB_SIZE,
  TINY_VOCAB,
  tinyCrossEntropy,
  tinyNextDistribution,
  tinyPerplexity,
  tinyTopTokens,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  devicesNeeded,
  formatBytes,
  MODEL_SIZES,
  memoryBreakdown,
  OPTIMIZERS,
  PRECISIONS,
  REFERENCE_DEVICE_BYTES,
} from "./memory";
import { readState, STEPS_TAKEN_MAX } from "./state";
import { pairLabel, STEP_LEARNING_RATE, stepFromCheckpoint } from "./step";

const corpus = TINY_CORPORA.harbor;
const PROBE_CONTEXT = "the fo";
/** In the corpus, "the fo" continues as "the fog", so this is the pair the Backward stage differentiates. */
const PROBE_TARGET = "g";

const stages = [
  { id: "dataset", name: "Dataset", detail: "text → token pairs" },
  { id: "batch", name: "Batch", detail: "shuffle and slice" },
  { id: "forward", name: "Forward", detail: "logits → probabilities" },
  { id: "loss", name: "Loss", detail: "cross-entropy" },
  { id: "backward", name: "Backward", detail: "one gradient per weight" },
  { id: "step", name: "Optimizer step", detail: "subtract the gradient" },
  { id: "checkpoint", name: "Checkpoint", detail: "save the weights" },
];

/** The run's shuffle seed. The Step one batch card replays the batches this seed produces. */
const RUN_SEED = 1;
/** Rows of the before-and-after table; the batch can touch more, and the card says how many. */
const TABLE_ROWS = 8;

export default function Explore({ state, setState, narrate }: ModuleContext) {
  // Clamp to the slider ranges so a zeroed or hand-edited state still trains a real run.
  const { stage, epochs, batchSize, stepsTaken, memModel, memPrecision, memOptimizer } = readState(state);

  const run = useMemo(
    () =>
      trainTinyModel({
        text: corpus.text,
        epochs,
        batchSize,
        learningRate: 0.6,
        seed: RUN_SEED,
        checkpoints: 9,
      }),
    [epochs, batchSize],
  );

  const tokens = useMemo(() => encodeTinyText(corpus.text).length, []);
  const checkpointIndex = Math.max(
    0,
    Math.min(run.checkpoints.length - 1, readState(state).checkpoint),
  );
  const checkpoint = run.checkpoints[checkpointIndex];
  const batchesPerEpoch = Math.ceil(run.tokens / batchSize);
  const lastPoint = run.history[run.history.length - 1];
  const probe = tinyTopTokens(checkpoint.weights, PROBE_CONTEXT, 5);
  const probeRow = tinyNextDistribution(checkpoint.weights, PROBE_CONTEXT);
  const targetId = TINY_VOCAB.indexOf(PROBE_TARGET);
  const pairGradient = probeRow.map((probability, id) => probability - (id === targetId ? 1 : 0));
  const pairNorm = Math.sqrt(pairGradient.reduce((sum, value) => sum + value * value, 0));

  // One real update, replayed from the selected checkpoint: the batch the run drew at that step.
  const stepResult = useMemo(
    () => stepFromCheckpoint(checkpoint.weights, checkpoint.step, stepsTaken, corpus.text, batchSize, RUN_SEED),
    [checkpoint, stepsTaken, batchSize],
  );
  const stepCorpusLoss = useMemo(
    () =>
      stepResult
        ? {
            before: tinyCrossEntropy(stepResult.before, corpus.text),
            after: tinyCrossEntropy(stepResult.after, corpus.text),
          }
        : null,
    [stepResult],
  );

  const modelSize = MODEL_SIZES.find((entry) => entry.id === memModel) ?? MODEL_SIZES[3];
  const memory = memoryBreakdown(modelSize.parameters, memPrecision, memOptimizer);
  const precisionChoice = PRECISIONS.find((entry) => entry.id === memPrecision) ?? PRECISIONS[1];
  const optimizerChoice = OPTIMIZERS.find((entry) => entry.id === memOptimizer) ?? OPTIMIZERS[2];
  const stateDescription =
    memory.stateParts.length === 0
      ? "none"
      : memory.stateParts.map((part) => `${part.name} ${part.bytesPerParameter}`).join(" + ");
  const memorySummary = `${modelSize.label} parameters, ${precisionChoice.label}, ${optimizerChoice.label}: ${formatBytes(
    memory.total,
  )} in total. Weights ${formatBytes(memory.weights)}, gradients ${formatBytes(memory.gradients)}, optimizer state ${formatBytes(
    memory.optimizerState,
  )}.`;

  const stageReadout = [
    {
      title: `${tokens} characters become ${run.tokens} training pairs`,
      note: `Every pair is one context character and the character that follows it. The vocabulary is ${TINY_VOCAB_SIZE} symbols, so a model that has learned nothing scores ${UNIFORM_CROSS_ENTROPY.toFixed(2)} nats per token.`,
      metrics: [
        { label: "Characters", value: `${tokens}` },
        { label: "Training pairs", value: `${run.tokens}` },
        { label: "Vocabulary", value: `${TINY_VOCAB_SIZE}` },
      ],
    },
    {
      title: `${batchesPerEpoch} batches per epoch, ${run.steps} optimizer steps in total`,
      note: "The pairs are shuffled once per epoch with a seeded generator, then cut into batches. Shuffling makes each batch a fair sample of the whole text, so consecutive updates do not all pull toward one passage and the weights at the end of an epoch are not biased toward whatever came last.",
      metrics: [
        { label: "Batch size", value: `${batchSize} pairs` },
        { label: "Batches / epoch", value: `${batchesPerEpoch}` },
        { label: "Total steps", value: `${run.steps}` },
      ],
    },
    {
      title: `The forward pass turns one row of weights into ${TINY_VOCAB_SIZE} probabilities`,
      note: `After "${PROBE_CONTEXT}" the model reads the row for “o” and softmaxes it. The ranked list on the right is that distribution, taken from the checkpoint you have selected.`,
      metrics: [
        { label: "Top prediction", value: `“${probe[0].label}” ${(probe[0].probability * 100).toFixed(1)}%` },
        { label: "Runner-up", value: `“${probe[1].label}” ${(probe[1].probability * 100).toFixed(1)}%` },
        { label: "Row width", value: `${TINY_VOCAB_SIZE} logits` },
      ],
    },
    {
      title: `Loss is the negative log-probability of the character that actually came next`,
      note: "Averaged over the pairs, in nats per token. A fall of ln 2 ≈ 0.69 nats means the probability given to the right character doubled, as a geometric mean. Like the forward stage, this reads the checkpoint selected on the right, scored on the whole corpus.",
      metrics: [
        { label: "Loss at this checkpoint", value: `${checkpoint.loss.toFixed(3)} nats`, tone: "loss" as const },
        { label: "Its perplexity", value: tinyPerplexity(checkpoint.weights, corpus.text).toFixed(2) },
        { label: "Uniform baseline", value: `${UNIFORM_CROSS_ENTROPY.toFixed(3)} nats` },
      ],
    },
    {
      title: "The backward pass produces one number per weight",
      note: `For softmax cross-entropy the gradient of a row is the predicted distribution minus a one at the correct character. The strip below is that gradient for one real pair, "o" followed by "g" in "the fog", at the selected checkpoint. A batch gradient averages such rows over its pairs; only rows that appeared in the batch receive any.`,
      metrics: [
        { label: "This pair's gradient norm", value: pairNorm.toFixed(4), tone: "gradient" as const },
        { label: "Last batch gradient norm", value: (lastPoint?.gradientNorm ?? 0).toFixed(4), tone: "gradient" as const },
        { label: "Peak batch norm in the run", value: run.peakGradientNorm.toFixed(4), tone: "gradient" as const },
      ],
    },
    {
      title: "The optimizer step subtracts the gradient, scaled by the learning rate",
      note: "This run uses plain minibatch gradient descent at a fixed rate of 0.6. Nothing else about the model changes during a step. The Step one batch card below replays one real update and shows the rows it changed.",
      metrics: [
        { label: "Learning rate", value: (lastPoint?.learningRate ?? 0.6).toFixed(3) },
        { label: "Rows updated", value: `${run.updatedParameters / TINY_VOCAB_SIZE} / ${TINY_VOCAB_SIZE}` },
        { label: "Steps taken", value: `${run.steps}` },
      ],
    },
    {
      title: `${run.checkpoints.length} checkpoints saved, including the untrained one`,
      note: "A checkpoint is just a copy of the weights. Scrub the timeline on the right to load any of them and sample from that exact model.",
      metrics: [
        { label: "Checkpoints", value: `${run.checkpoints.length}` },
        { label: "Selected step", value: `${checkpoint.step}` },
        { label: "Its loss", value: `${checkpoint.loss.toFixed(3)} nats`, tone: "loss" as const },
      ],
    },
  ][stage];

  return (
    <div className="tg-lab tg-lab--hero a10-lab">
      <LabSurface label="One pass through the training loop" className="tg-loop-panel">
        <SurfaceHeading
          kicker={`Stage ${stage + 1} of ${stages.length}`}
          title="Every stage below is running on the bundled character model"
          aside={<span className="tg-badge">{corpus.name}</span>}
        />
        <StageFlow
          label="Training loop stages"
          stages={stages}
          current={stage}
          loops
          onSelect={(index) => {
            setState({ stage: index });
            narrate(`${stages[index].name}. ${stageReadout.title}`);
          }}
        />
        <div className="tg-stage-detail">
          <strong>{stageReadout.title}</strong>
          <p>{stageReadout.note}</p>
        </div>
        <div className="metric-row">
          {stageReadout.metrics.map((metric) => (
            <Metric key={metric.label} label={metric.label} value={metric.value} tone={metric.tone} />
          ))}
        </div>
        {stage === 4 && <GradientStrip gradient={pairGradient} target={targetId} step={checkpoint.step} />}
      </LabSurface>

      <LabSurface label="Measured training loss" className="tg-loss-panel">
        <SurfaceHeading
          kicker="Measured, not illustrated"
          title="Loss over the whole run"
          aside={<span className="tg-badge">{run.steps} steps</span>}
        />
        <LossTimeline
          history={run.history.map((point) => ({ step: point.step, loss: point.loss }))}
          checkpoints={run.checkpoints.map((point) => ({ step: point.step, loss: point.loss }))}
          selected={checkpointIndex}
          steps={run.steps}
          epochs={epochs}
          batchesPerEpoch={batchesPerEpoch}
        />
        <div className="tr-legend">
          <span className="tr-legend__item tr-legend__item--batch">
            <i /> batch loss, averaged per logging window <b>{(lastPoint?.loss ?? UNIFORM_CROSS_ENTROPY).toFixed(3)}</b>
          </span>
          <span className="tr-legend__item tr-legend__item--corpus">
            <i /> whole-corpus loss at each checkpoint <b>{run.finalLoss.toFixed(3)}</b>
          </span>
          <span className="tr-legend__item tr-legend__item--uniform">
            <i /> knows nothing, ln 30 <b>{UNIFORM_CROSS_ENTROPY.toFixed(3)}</b>
          </span>
        </div>
        <p className="lab-note">
          The large dot is the checkpoint selected on the right, scored on all {run.tokens} pairs. Ticks under the
          axis mark epoch boundaries, every {batchesPerEpoch} steps. The thin line is noisier because each point
          averages only a few batches, and each batch is a different sample of the text.
        </p>
        <div className="metric-row">
          <Metric label="Final loss" value={`${run.finalLoss.toFixed(3)} nats`} tone="loss" />
          <Metric label="Perplexity" value={tinyPerplexity(run.weights, corpus.text).toFixed(2)} />
          <Metric
            label="Better than chance by"
            value={`${(((UNIFORM_CROSS_ENTROPY - run.finalLoss) / UNIFORM_CROSS_ENTROPY) * 100).toFixed(0)}%`}
            tone="forward"
          />
        </div>
      </LabSurface>

      <LabSurface label="Run controls and checkpoint timeline" className="tg-checkpoint-panel">
        <SurfaceHeading kicker="Loop settings" title="Change the loop, then reload a checkpoint" />
        <RangeControl
          label="Batch size"
          min={2}
          max={64}
          step={2}
          value={batchSize}
          format={(value) => `${value} pairs`}
          onChange={(value) => {
            // Jump to the last snapshot of the new run, however many it saved.
            setState({ batchSize: value, checkpoint: 99, stepsTaken: 0 });
            narrate(`Batch size ${value}. That is ${Math.ceil(run.tokens / value)} batches per epoch.`);
          }}
        />
        <RangeControl
          label="Epochs"
          min={1}
          max={60}
          step={1}
          value={epochs}
          format={(value) => `${value} passes`}
          onChange={(value) => setState({ epochs: value, checkpoint: 99, stepsTaken: 0 })}
        />
        <RangeControl
          label="Checkpoint"
          min={0}
          max={Math.max(1, run.checkpoints.length - 1)}
          step={1}
          value={checkpointIndex}
          format={(value) => `step ${run.checkpoints[value]?.step ?? 0}`}
          onChange={(value) => {
            setState({ checkpoint: value, stepsTaken: 0 });
            narrate(
              `Checkpoint at step ${run.checkpoints[value].step}. Loss ${run.checkpoints[value].loss.toFixed(3)} nats per token.`,
            );
          }}
        />
        <div className="tg-sample">
          <span>Sampled from this checkpoint at temperature 0.8</span>
          <p>
            {sampleTinyText(checkpoint.weights, {
              prompt: "the ",
              length: 96,
              temperature: 0.8,
              seed: 21,
            })}
          </p>
        </div>
        <BarList
          label={`Next character after "${PROBE_CONTEXT}" at this checkpoint`}
          items={probe.map((entry, index) => ({
            id: `${entry.token}-${index}`,
            label: entry.label === "␣" ? "space" : `“${entry.label}”`,
            value: entry.probability,
            display: `${(entry.probability * 100).toFixed(1)}%`,
            tone: index === 0 ? "forward" : "muted",
            emphasis: index === 0,
          }))}
          max={1}
        />
      </LabSurface>

      <LabSurface label="Step one batch" className="tr-step-card">
        <SurfaceHeading
          kicker={`One real update · learning rate ${STEP_LEARNING_RATE} · seed ${RUN_SEED}`}
          title="Take the next minibatch from the selected checkpoint"
          aside={
            <span className="tg-badge">
              {stepsTaken} / {STEPS_TAKEN_MAX} steps taken
            </span>
          }
        />
        <div className="tr-step-actions">
          <button
            type="button"
            className="primary-action"
            disabled={stepsTaken >= STEPS_TAKEN_MAX}
            onClick={() => {
              const next = Math.min(STEPS_TAKEN_MAX, stepsTaken + 1);
              setState({ stepsTaken: next });
              const result = stepFromCheckpoint(
                checkpoint.weights,
                checkpoint.step,
                next,
                corpus.text,
                batchSize,
                RUN_SEED,
              );
              if (result) {
                narrate(
                  `Took step ${result.batch.step + 1} from the checkpoint at step ${checkpoint.step}. Batch loss ${result.lossBefore.toFixed(
                    3,
                  )} before, ${result.lossAfter.toFixed(3)} after. Gradient norm ${result.gradientNorm.toFixed(3)}. ${
                    result.rows.length
                  } rows changed.`,
                );
              }
            }}
          >
            Step one batch
          </button>
          <button
            type="button"
            className="quiet-action"
            disabled={stepsTaken === 0}
            onClick={() => {
              setState({ stepsTaken: 0 });
              narrate(`Back to the checkpoint at step ${checkpoint.step}.`);
            }}
          >
            Back to the checkpoint
          </button>
        </div>
        {stepResult && stepCorpusLoss ? (
          <>
            <p className="tr-step-lead" aria-live="polite">
              <strong>
                Step {stepResult.batch.step} → {stepResult.batch.step + 1}
              </strong>{" "}
              · epoch {stepResult.batch.epoch + 1}, batch {stepResult.batch.batchInEpoch + 1} of {batchesPerEpoch} ·{" "}
              {stepResult.batch.inputs.length} pairs
              {stepResult.batch.step >= run.steps ? " · past the end of this run, so it is the next epoch's batch" : ""}
            </p>
            <ul className="tr-pairs" aria-label="The pairs in this batch, previous character then next character">
              {stepResult.batch.inputs.map((input, index) => (
                <li key={index}>{pairLabel(input, stepResult.batch.targets[index])}</li>
              ))}
            </ul>
            <div className="metric-row">
              <Metric
                label="Batch loss, before → after"
                value={`${stepResult.lossBefore.toFixed(3)} → ${stepResult.lossAfter.toFixed(3)}`}
                tone="loss"
              />
              <Metric label="Batch gradient norm" value={stepResult.gradientNorm.toFixed(4)} tone="gradient" />
              <Metric
                label="Whole-corpus loss, before → after"
                value={`${stepCorpusLoss.before.toFixed(3)} → ${stepCorpusLoss.after.toFixed(3)}`}
                tone="loss"
              />
              <Metric
                label="Weights changed"
                value={`${stepResult.weightsChanged} of ${TINY_VOCAB_SIZE * TINY_VOCAB_SIZE} · ${stepResult.rows.length} rows`}
              />
            </div>
            <div className="tr-table-wrap">
              <table className="tr-table">
                <caption>
                  The {Math.min(TABLE_ROWS, stepResult.rows.length)} rows with the largest gradient
                  {stepResult.rows.length > TABLE_ROWS
                    ? ` (${stepResult.rows.length - TABLE_ROWS} more ${stepResult.rows.length - TABLE_ROWS === 1 ? "row" : "rows"} changed less)`
                    : ""}.
                  Each row is the previous character; “target” is the next character that row was asked for most often
                  in this batch.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Row</th>
                    <th scope="col">Pairs</th>
                    <th scope="col">Row gradient</th>
                    <th scope="col">Target</th>
                    <th scope="col">Target logit</th>
                    <th scope="col">Target probability</th>
                  </tr>
                </thead>
                <tbody>
                  {stepResult.rows.slice(0, TABLE_ROWS).map((row) => (
                    <tr key={row.row}>
                      <th scope="row">{TINY_VOCAB[row.row] === " " ? "␣" : TINY_VOCAB[row.row]}</th>
                      <td>{row.pairs}</td>
                      <td>{row.gradientNorm.toFixed(3)}</td>
                      <td>{TINY_VOCAB[row.target] === " " ? "␣" : TINY_VOCAB[row.target]}</td>
                      <td>
                        {row.logitBefore.toFixed(3)} → {row.logitAfter.toFixed(3)}
                      </td>
                      <td>
                        {(row.probabilityBefore * 100).toFixed(1)}% → {(row.probabilityAfter * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="lab-note">
            Press <strong>Step one batch</strong> to apply the very next update this run took after the checkpoint
            selected above, using the same shuffled batch. It shows which rows of the 900-weight table moved, how far, and
            what the batch and the whole corpus thought of the change. Pressing again chains the next update.
          </p>
        )}
        <p className="lab-note">
          This is a replay, not a second run: the batch is drawn from the same seeded shuffle the full run used, and the
          update is the same arithmetic, so repeating a press always gives the same numbers. Rows that did not appear in
          the batch receive no gradient and do not move.
        </p>
      </LabSurface>

      <LabSurface label="Where the memory goes" className="tr-memory-card">
        <SurfaceHeading
          kicker="Arithmetic on a formula, not a measurement of any run"
          title="Model state costs a fixed number of bytes per parameter"
          aside={<span className="tg-badge">{formatBytes(memory.total)}</span>}
        />
        <div className="tr-memory-controls">
          <SegmentedControl
            label="Model size"
            value={modelSize.id}
            options={MODEL_SIZES.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => {
              setState({ memModel: value });
              const next = MODEL_SIZES.find((entry) => entry.id === value);
              if (next) {
                narrate(
                  `${next.label} parameters: ${formatBytes(memoryBreakdown(next.parameters, memPrecision, memOptimizer).total)} of model state.`,
                );
              }
            }}
          />
          <SegmentedControl
            label="Precision"
            value={precisionChoice.id}
            options={PRECISIONS.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => {
              setState({ memPrecision: value });
              narrate(PRECISIONS.find((entry) => entry.id === value)?.detail ?? "");
            }}
          />
          <SegmentedControl
            label="Optimizer"
            value={optimizerChoice.id}
            options={OPTIMIZERS.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => {
              setState({ memOptimizer: value });
              narrate(OPTIMIZERS.find((entry) => entry.id === value)?.detail ?? "");
            }}
          />
        </div>
        <div className="tr-membar" role="img" aria-label={memorySummary}>
          <i className="is-weights" style={{ flexGrow: memory.weights }}>
            <b>Weights</b>
            <span>{formatBytes(memory.weights)}</span>
          </i>
          <i className="is-gradients" style={{ flexGrow: memory.gradients }}>
            <b>Gradients</b>
            <span>{formatBytes(memory.gradients)}</span>
          </i>
          {memory.optimizerState > 0 && (
            <i className="is-state" style={{ flexGrow: memory.optimizerState }}>
              <b>Optimizer state</b>
              <span>{formatBytes(memory.optimizerState)}</span>
            </i>
          )}
        </div>
        <div className="tr-table-wrap">
          <table className="tr-table">
            <caption>
              {modelSize.label} parameters ({modelSize.parameters.toLocaleString("en-US")}), {precisionChoice.label},{" "}
              {optimizerChoice.label}. Units are decimal: 1 GB is 10⁹ bytes.
            </caption>
            <thead>
              <tr>
                <th scope="col">Part</th>
                <th scope="col">Bytes per parameter</th>
                <th scope="col">Total</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Weights</th>
                <td>{memory.weightBytesPerParameter}</td>
                <td>{formatBytes(memory.weights)}</td>
              </tr>
              <tr>
                <th scope="row">Gradients</th>
                <td>{memory.gradientBytesPerParameter}</td>
                <td>{formatBytes(memory.gradients)}</td>
              </tr>
              <tr>
                <th scope="row">Optimizer state</th>
                <td>
                  {memory.stateBytesPerParameter}
                  <small> ({stateDescription})</small>
                </td>
                <td>{formatBytes(memory.optimizerState)}</td>
              </tr>
              <tr className="is-total">
                <th scope="row">Total</th>
                <td>{memory.totalBytesPerParameter}</td>
                <td>{formatBytes(memory.total)}</td>
              </tr>
              <tr className="is-omitted">
                <th scope="row">Activations</th>
                <td colSpan={2}>not included: they depend on batch size, sequence length, and recomputation</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric label="Bytes per parameter" value={`${memory.totalBytesPerParameter}`} tone="forward" />
          <Metric label="Optimizer state ÷ weights" value={`${(memory.optimizerState / memory.weights).toFixed(1)}×`} tone="gradient" />
          <Metric
            label={`${formatBytes(REFERENCE_DEVICE_BYTES)} devices for state alone`}
            value={`${devicesNeeded(memory.total)}`}
          />
        </div>
        <p className="lab-note">
          {optimizerChoice.detail} {precisionChoice.detail} Totals this large are why big runs split the state across
          devices: data, tensor, and pipeline parallelism divide the work, and ZeRO and FSDP shard the state itself so
          no device holds all of it. Only parameters that are trained need a gradient and optimizer state, which is the
          case for the LoRA & adapters lab.
        </p>
      </LabSurface>
    </div>
  );
}

function LossTimeline({
  history,
  checkpoints,
  selected,
  steps,
  epochs,
  batchesPerEpoch,
}: {
  history: ReadonlyArray<{ step: number; loss: number }>;
  checkpoints: ReadonlyArray<{ step: number; loss: number }>;
  selected: number;
  steps: number;
  epochs: number;
  batchesPerEpoch: number;
}) {
  const W = 440;
  const H = 236;
  const L = 40;
  const R = 432;
  const T = 20;
  const B = 204;
  const low = 1.5;
  const high = UNIFORM_CROSS_ENTROPY + 0.15;
  const xAt = (step: number) => L + (step / Math.max(1, steps)) * (R - L);
  const yAt = (loss: number) => B - ((Math.min(high, Math.max(low, loss)) - low) / (high - low)) * (B - T);
  const line = (points: ReadonlyArray<{ step: number; loss: number }>) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"}${xAt(point.step).toFixed(1)},${yAt(point.loss).toFixed(1)}`).join(" ");
  const current = checkpoints[selected] ?? checkpoints[checkpoints.length - 1];
  const labelLeft = xAt(current.step) > (L + R) / 2;
  return (
    <svg
      className="tr-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Training loss over ${steps} optimizer steps and ${epochs} epochs. Whole-corpus loss at the ${
        checkpoints.length
      } checkpoints: ${checkpoints.map((point) => point.loss.toFixed(3)).join(", ")}. Selected checkpoint at step ${
        current.step
      }, loss ${current.loss.toFixed(3)}.`}
    >
      {[low, 2.5, UNIFORM_CROSS_ENTROPY].map((tick) => (
        <g key={tick}>
          <line className="tr-grid" x1={L} x2={R} y1={yAt(tick)} y2={yAt(tick)} />
          <text className="tr-tick" x={L - 5} y={yAt(tick) + 3} textAnchor="end">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {Array.from({ length: epochs + 1 }, (_, epoch) => (
        <line
          key={epoch}
          className="tr-epoch-tick"
          x1={xAt(Math.min(steps, epoch * batchesPerEpoch))}
          x2={xAt(Math.min(steps, epoch * batchesPerEpoch))}
          y1={B}
          y2={B + 5}
        />
      ))}
      <text className="tr-tick" x={L} y={B + 16}>
        0
      </text>
      <text className="tr-tick" x={R} y={B + 16} textAnchor="end">
        {steps} steps · {epochs} {epochs === 1 ? "epoch" : "epochs"}
      </text>
      <line className="plot-axis" x1={L} x2={R} y1={B} y2={B} />
      <line className="plot-axis" x1={L} x2={L} y1={T} y2={B} />
      <line className="tr-baseline" x1={L} x2={R} y1={yAt(UNIFORM_CROSS_ENTROPY)} y2={yAt(UNIFORM_CROSS_ENTROPY)} />
      <path className="tr-batch" d={line([{ step: 0, loss: UNIFORM_CROSS_ENTROPY }, ...history])} />
      <path className="tr-corpus" d={line(checkpoints)} />
      {checkpoints.map((point, index) => (
        <circle
          key={point.step}
          className={`tr-checkpoint${index === selected ? " is-selected" : ""}`}
          cx={xAt(point.step)}
          cy={yAt(point.loss)}
          r={index === selected ? 5.5 : 2.8}
        />
      ))}
      <line className="tr-guide" x1={xAt(current.step)} x2={xAt(current.step)} y1={T} y2={B} />
      <text
        className="tr-dot-label"
        x={xAt(current.step) + (labelLeft ? -9 : 9)}
        y={Math.max(T + 10, yAt(current.loss) - 14)}
        textAnchor={labelLeft ? "end" : "start"}
      >
        step {current.step} · {current.loss.toFixed(3)} nats
      </text>
      <text className="tr-tick" x={4} y={11}>
        loss (nats/token)
      </text>
    </svg>
  );
}

function GradientStrip({ gradient, target, step }: { gradient: ReadonlyArray<number>; target: number; step: number }) {
  const W = 600;
  const H = 128;
  const top = 12;
  const zero = 62;
  const scale = 46;
  const slot = W / gradient.length;
  const glyph = (id: number) => (TINY_VOCAB[id] === " " ? "␣" : TINY_VOCAB[id]);
  const largest = gradient
    .map((value, id) => ({ value, id }))
    .filter((entry) => entry.id !== target)
    .sort((left, right) => right.value - left.value)[0];
  return (
    <figure className="tr-gradient">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Gradient of the loss with respect to the 30 logits in row o, for the pair o followed by g, at step ${step}. The target g gets ${gradient[
          target
        ].toFixed(3)}; the largest push down is on ${glyph(largest.id)} at +${largest.value.toFixed(3)}; every other entry equals its predicted probability.`}
      >
        <line className="tr-grid" x1={0} x2={W} y1={zero} y2={zero} />
        {gradient.map((value, id) => {
          const height = Math.abs(value) * scale;
          return (
            <g key={id}>
              <rect
                className={id === target ? "tr-grad-bar is-target" : "tr-grad-bar"}
                x={id * slot + 3}
                y={value >= 0 ? zero - height : zero}
                width={slot - 6}
                height={Math.max(0.8, height)}
              />
              <text className="tr-grad-label" x={id * slot + slot / 2} y={H - 4} textAnchor="middle">
                {glyph(id)}
              </text>
            </g>
          );
        })}
        <text className="tr-tick" x={4} y={top}>
          above the line: that logit is pushed down · below: pushed up
        </text>
        <text className="tr-grad-value" x={target * slot + slot + 1} y={zero - (gradient[target] * scale) / 2 + 3}>
          {gradient[target].toFixed(3)}
        </text>
      </svg>
      <figcaption>
        ∂loss/∂logit for the row of <code>o</code> on the pair <code>o → g</code>: every character's predicted probability,
        minus 1 at <code>g</code>. The update subtracts learning rate × this, so <code>g</code> rises and everything
        else falls, most of all the current favourite.
      </figcaption>
    </figure>
  );
}
