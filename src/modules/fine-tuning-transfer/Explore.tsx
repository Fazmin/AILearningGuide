import { useMemo } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  DECISION_KEYS,
  DECISION_LABELS,
  DECISION_QUESTIONS,
  DECISION_RULES,
  recommend,
  ROUTES,
  type DecisionKey,
} from "./decision";
import { buildFineTuneData, SOURCE_SENTENCES, TARGET_SENTENCES } from "./holdout";
import {
  BLOCK_SIZE,
  boundedDecision,
  boundedEpochs,
  boundedFrozen,
  boundedHeldOut,
  boundedLearningRate,
  boundedReplay,
  EPOCHS_MAX,
  EPOCHS_MIN,
  HELD_OUT_MAX,
  HELD_OUT_MIN,
  LEARNING_RATE_MAX,
  LEARNING_RATE_MIN,
  REPLAY_MAX,
  REPLAY_STEP,
} from "./state";

const source = TINY_CORPORA.harbor;
const targetCorpus = TINY_CORPORA.recipes;
const PRETRAIN_EPOCHS = 60;

const blocks = Array.from({ length: Math.ceil(TINY_VOCAB_SIZE / BLOCK_SIZE) }, (_, index) => ({
  id: `block-${index}`,
  label: TINY_VOCAB.slice(index * BLOCK_SIZE, index * BLOCK_SIZE + BLOCK_SIZE).replace(" ", "␣"),
  rows: Array.from({ length: BLOCK_SIZE }, (_, offset) => index * BLOCK_SIZE + offset).filter(
    (row) => row < TINY_VOCAB_SIZE,
  ),
}));

export default function Explore({ state, setState, narrate }: ModuleContext) {
  // Defence in depth: hydrateState already clamps, but a live state can also arrive from a
  // snapshot or a patch, and training below runs synchronously.
  const fineTuneEpochs = boundedEpochs(state.fineTuneEpochs);
  const learningRate = boundedLearningRate(state.learningRate);
  const frozenKey = boundedFrozen(state.frozen).join("|");
  const frozen = useMemo(() => (frozenKey ? frozenKey.split("|") : []), [frozenKey]);

  const base = useMemo(
    () => trainTinyModel({ text: source.text, epochs: PRETRAIN_EPOCHS, seed: 1 }),
    [],
  );

  const trainableRows = useMemo(() => {
    const rows = Array.from({ length: TINY_VOCAB_SIZE }, () => true);
    for (const block of blocks) {
      if (frozen.includes(block.id)) for (const row of block.rows) rows[row] = false;
    }
    return rows;
  }, [frozen]);

  const transfer = useMemo(
    () =>
      trainTinyModel({
        text: targetCorpus.text,
        epochs: fineTuneEpochs,
        learningRate,
        init: base.weights,
        trainableRows,
        seed: 2,
        checkpoints: 9,
      }),
    [base.weights, fineTuneEpochs, learningRate, trainableRows],
  );

  const scratch = useMemo(
    () =>
      trainTinyModel({
        text: targetCorpus.text,
        epochs: fineTuneEpochs,
        learningRate,
        seed: 2,
        checkpoints: 9,
      }),
    [fineTuneEpochs, learningRate],
  );

  const sourceBefore = tinyCrossEntropy(base.weights, source.text);
  const sourceAfter = tinyCrossEntropy(transfer.weights, source.text);
  const targetBefore = tinyCrossEntropy(base.weights, targetCorpus.text);
  const targetAfter = tinyCrossEntropy(transfer.weights, targetCorpus.text);
  const forgetting = sourceAfter - sourceBefore;
  const frozenRows = trainableRows.filter((value) => !value).length;

  const progressAt = (index: number, total: number) => index / Math.max(1, total - 1);

  /** Both corpora scored at every checkpoint of the transfer run. */
  const meter = useMemo(() => {
    const target = transfer.checkpoints.map((point) => tinyCrossEntropy(point.weights, targetCorpus.text));
    const original = transfer.checkpoints.map((point) => tinyCrossEntropy(point.weights, source.text));
    const all = [...target, ...original];
    const low = Math.min(...all);
    const high = Math.max(...all);
    const pad = Math.max(0.03, (high - low) * 0.12);
    return { target, original, domain: [low - pad, high + pad] as [number, number] };
  }, [transfer]);

  /** How far each context row moved: the L2 norm of that row of W_after − W_before. */
  const rowMovement = useMemo(
    () =>
      Array.from({ length: TINY_VOCAB_SIZE }, (_, row) => {
        let squared = 0;
        for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
          const index = row * TINY_VOCAB_SIZE + k;
          squared += (transfer.weights[index] - base.weights[index]) ** 2;
        }
        return Math.sqrt(squared);
      }),
    [base.weights, transfer.weights],
  );
  const maxMovement = Math.max(1e-6, ...rowMovement);
  const unusedTrainable = rowMovement.filter((value, row) => trainableRows[row] && value === 0).length;

  /**
   * The replay and held-out card trains its own fine-tune on a split of the target corpus, so it
   * never changes what the other cards compute. It shares the epoch count and learning rate above.
   */
  const heldOutSentences = boundedHeldOut(state.heldOutSentences);
  const replayShare = boundedReplay(state.replayShare);
  const data = useMemo(() => buildFineTuneData(heldOutSentences, replayShare), [heldOutSentences, replayShare]);
  const plainData = useMemo(() => buildFineTuneData(heldOutSentences, 0), [heldOutSentences]);
  const withoutReplay = useMemo(
    () =>
      trainTinyModel({
        text: plainData.text,
        epochs: fineTuneEpochs,
        learningRate,
        init: base.weights,
        seed: 2,
        checkpoints: 9,
      }),
    [base.weights, fineTuneEpochs, learningRate, plainData.text],
  );
  const withReplay = useMemo(
    () =>
      data.replaySentences === 0
        ? withoutReplay
        : trainTinyModel({
            text: data.text,
            epochs: fineTuneEpochs,
            learningRate,
            init: base.weights,
            seed: 2,
            checkpoints: 9,
          }),
    [base.weights, data.replaySentences, data.text, fineTuneEpochs, learningRate, withoutReplay],
  );
  const sourceLoss = (weights: Float32Array, text: string) =>
    text ? tinyCrossEntropy(weights, text) - tinyCrossEntropy(base.weights, text) : Number.NaN;
  /** The same readouts for any run; the two original-corpus groups are always the chosen replay's. */
  const scoreRun = (run: typeof withReplay) => ({
    trained: tinyCrossEntropy(run.weights, data.trainTarget),
    held: data.heldSentences > 0 ? tinyCrossEntropy(run.weights, data.heldTarget) : Number.NaN,
    forgetting: tinyCrossEntropy(run.weights, source.text) - sourceBefore,
    replayed: sourceLoss(run.weights, data.replay),
    notReplayed: sourceLoss(run.weights, data.notReplayed),
    steps: run.steps,
  });
  const plainScore = scoreRun(withoutReplay);
  const replayScore = scoreRun(withReplay);
  const replayChart = useMemo(() => {
    const trained = withReplay.checkpoints.map((point) => tinyCrossEntropy(point.weights, data.trainTarget));
    const held = withReplay.checkpoints.map((point) =>
      data.heldSentences > 0 ? tinyCrossEntropy(point.weights, data.heldTarget) : Number.NaN,
    );
    const original = withReplay.checkpoints.map((point) => tinyCrossEntropy(point.weights, source.text));
    const all = [...trained, ...held, ...original].filter(Number.isFinite);
    const low = Math.min(...all);
    const high = Math.max(...all);
    const pad = Math.max(0.03, (high - low) * 0.12);
    return { trained, held, original, domain: [low - pad, high + pad] as [number, number] };
  }, [data.heldSentences, data.heldTarget, data.trainTarget, withReplay]);
  const showLoss = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "none held out");
  const showForgetting = (value: number, empty: string) =>
    Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}` : empty;
  const replayRows: Array<{ label: string; plain: string; replay: string }> = [
    { label: "Trained-on target", plain: showLoss(plainScore.trained), replay: showLoss(replayScore.trained) },
    { label: "Held-out target", plain: showLoss(plainScore.held), replay: showLoss(replayScore.held) },
    {
      label: "Forgetting, all original sentences",
      plain: showForgetting(plainScore.forgetting, "none"),
      replay: showForgetting(replayScore.forgetting, "none"),
    },
    ...(data.replaySentences > 0
      ? [
          {
            label: "Forgetting, the replayed sentences",
            plain: showForgetting(plainScore.replayed, "none"),
            replay: showForgetting(replayScore.replayed, "none"),
          },
          {
            label: "Forgetting, the sentences not replayed",
            plain: showForgetting(plainScore.notReplayed, "none left"),
            replay: showForgetting(replayScore.notReplayed, "none left"),
          },
        ]
      : []),
    { label: "Optimizer steps", plain: `${plainScore.steps}`, replay: `${replayScore.steps}` },
  ];

  // The decision card is a hand-written table, not a training run: no memo is needed.
  const answers = boundedDecision(state);
  const recommendation = recommend(answers);
  const answerDecision = (key: DecisionKey, yes: boolean) => {
    setState({ [key]: yes });
    narrate(
      `${DECISION_QUESTIONS[key]} ${yes ? "Yes" : "No"}. Recommended route: ${recommend({ ...answers, [key]: yes }).route.title}.`,
    );
  };

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="Transfer against training from scratch" className="tg-compare-panel ftt-panel">
        <SurfaceHeading
          kicker={`Pretrained ${PRETRAIN_EPOCHS} epochs on ${source.name.toLowerCase()}`}
          title={`Both runs get ${fineTuneEpochs} epochs on ${targetCorpus.name.toLowerCase()}`}
          aside={<span className="tg-badge">{transfer.steps} steps each</span>}
        />
        <LineChart
          label="Loss on the target corpus for transfer and scratch"
          xLabel="fine-tuning progress"
          yLabel="target loss (nats/token)"
          yDomain={[1.7, UNIFORM_CROSS_ENTROPY + 0.1]}
          series={[
            {
              id: "transfer",
              name: "started from pretrained",
              tone: "forward",
              points: transfer.checkpoints.map((point, index) => ({
                x: progressAt(index, transfer.checkpoints.length),
                y: tinyCrossEntropy(point.weights, targetCorpus.text),
              })),
            },
            {
              id: "scratch",
              name: "started from zero",
              tone: "loss",
              dash: "dashed",
              points: scratch.checkpoints.map((point, index) => ({
                x: progressAt(index, scratch.checkpoints.length),
                y: tinyCrossEntropy(point.weights, targetCorpus.text),
              })),
            },
          ]}
          footnote="Both curves are measured on the full target corpus at each saved checkpoint. Same data, same seed, same budget — only the starting weights differ."
        />
        <div className="metric-row">
          <Metric label="Transfer, target loss" value={`${targetAfter.toFixed(3)} nats`} tone="forward" />
          <Metric label="Scratch, target loss" value={`${scratch.finalLoss.toFixed(3)} nats`} tone="loss" />
          <Metric
            label={
              scratch.finalLoss - targetAfter >= 0
                ? "Transfer is lower by"
                : "Transfer is higher by"
            }
            value={`${Math.abs(scratch.finalLoss - targetAfter).toFixed(3)} nats`}
            tone={scratch.finalLoss - targetAfter >= 0 ? "forward" : "loss"}
          />
        </div>
      </LabSurface>

      <LabSurface label="Forgetting meter" className="tg-forget-panel ftt-panel a11-narrow-chart">
        <SurfaceHeading
          kicker="Two corpora, one set of weights"
          title="What adaptation costs on the original data"
          aside={<span className="tg-badge">{forgetting >= 0 ? "+" : ""}{forgetting.toFixed(3)} nats</span>}
        />
        <LineChart
          label="Target and source loss during fine-tuning"
          xLabel="fine-tuning progress"
          yLabel="loss (nats/char)"
          yDomain={meter.domain}
          series={[
            {
              id: "target",
              name: `${targetCorpus.name} (adapting)`,
              tone: "forward",
              points: meter.target.map((value, index) => ({
                x: progressAt(index, meter.target.length),
                y: value,
              })),
            },
            {
              id: "source",
              name: `${source.name} (forgetting)`,
              tone: "loss",
              dash: "dashed",
              points: meter.original.map((value, index) => ({
                x: progressAt(index, meter.original.length),
                y: value,
              })),
            },
          ]}
          footnote="One model, scored on both corpora at every checkpoint. Nothing is being averaged or smoothed."
        />
        <div className="tg-diff">
          <div className="tg-sample">
            <span>Before adaptation</span>
            <p>{sampleTinyText(base.weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 12 })}</p>
          </div>
          <div className="tg-sample">
            <span>After adaptation</span>
            <p>{sampleTinyText(transfer.weights, { prompt: "the ", length: 76, temperature: 0.7, seed: 12 })}</p>
          </div>
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Fine-tuning controls" className="tg-finetune-panel ftt-panel">
          <SurfaceHeading kicker="Adaptation budget" title="How hard to push" />
          <RangeControl
            label="Fine-tune epochs"
            min={EPOCHS_MIN}
            max={EPOCHS_MAX}
            step={1}
            value={fineTuneEpochs}
            format={(value) => (value === 0 ? "no adaptation" : `${value} passes`)}
            onChange={(value) => {
              setState({ fineTuneEpochs: value });
              narrate(`${value} fine-tuning epochs.`);
            }}
          />
          <RangeControl
            label="Fine-tune learning rate"
            min={LEARNING_RATE_MIN}
            max={LEARNING_RATE_MAX}
            step={0.05}
            value={learningRate}
            format={(value) => value.toFixed(2)}
            onChange={(value) => setState({ learningRate: value })}
          />
          <div className="metric-row">
            <Metric label="Source before" value={sourceBefore.toFixed(3)} />
            <Metric label="Source after" value={sourceAfter.toFixed(3)} tone={forgetting > 0.05 ? "loss" : undefined} />
            <Metric label="Target gained" value={(targetBefore - targetAfter).toFixed(3)} tone="forward" />
          </div>
        </LabSurface>

        <LabSurface label="Layer freezing" className="tg-freeze-wrap ftt-panel">
          <SurfaceHeading kicker="Freeze grid" title="Choose which weights may move" />
          <div className="tg-freeze-grid">
            {blocks.map((block) => {
              const isFrozen = frozen.includes(block.id);
              return (
                <button
                  type="button"
                  key={block.id}
                  aria-pressed={isFrozen}
                  className={isFrozen ? "is-frozen" : ""}
                  onClick={() => {
                    setState({
                      frozen: isFrozen
                        ? frozen.filter((id) => id !== block.id)
                        : [...frozen, block.id],
                    });
                    narrate(`${block.label} ${isFrozen ? "unfrozen" : "frozen"}.`);
                  }}
                >
                  <span>{block.label}</span>
                  <small>{isFrozen ? "frozen" : "trainable"}</small>
                </button>
              );
            })}
          </div>
          <div className="metric-row">
            <Metric label="Frozen rows" value={`${frozenRows} / ${TINY_VOCAB_SIZE}`} />
            <Metric
              label="Trainable weights"
              value={`${(TINY_VOCAB_SIZE - frozenRows) * TINY_VOCAB_SIZE}`}
              tone="forward"
            />
            <Metric label="Forgetting" value={`${forgetting >= 0 ? "+" : ""}${forgetting.toFixed(3)}`} tone={forgetting > 0.05 ? "loss" : "forward"} />
          </div>
          <div className="ftt-rows">
            <span className="ftt-rows__title">How far each row moved, ‖W_after − W_before‖ per context character</span>
            <div
              className="ftt-rows__bars"
              role="img"
              aria-label={`Row movement after fine-tuning. Largest ${maxMovement.toFixed(2)}. ${frozenRows} frozen rows and ${unusedTrainable} trainable rows that recipes never use as context did not move.`}
            >
              {rowMovement.map((value, row) => {
                const frozenRow = !trainableRows[row];
                return (
                  <span
                    key={row}
                    className={`ftt-rows__col ${frozenRow ? "is-frozen" : value === 0 ? "is-unused" : ""}`.trim()}
                    title={`${TINY_VOCAB[row] === " " ? "space" : TINY_VOCAB[row]}: ${frozenRow ? "frozen" : value === 0 ? "no gradient" : value.toFixed(2)}`}
                  >
                    <i style={{ height: `${(value / maxMovement) * 100}%` }} />
                    <b>{TINY_VOCAB[row] === " " ? "␣" : TINY_VOCAB[row]}</b>
                  </span>
                );
              })}
            </div>
            <span className="ftt-rows__key">
              Bar height out of {maxMovement.toFixed(2)}. Hatched: frozen. Empty with a dot: trainable
              but never a context in recipes, so no gradient.
            </span>
          </div>
          <p className="lab-note">
            This model has no layers, so the grid freezes blocks of the weight table by the character
            each row predicts from. Here the gradient for a frozen row is computed and then discarded;
            frameworks usually skip computing it at all. A real freeze grid would name layers, not
            letters.
          </p>
        </LabSurface>
      </div>

      <LabSurface label="Replay and held-out target" className="ftt-replay-panel ftt-panel">
        <SurfaceHeading
          kicker={`${data.trainSentences} target sentences trained on, ${data.heldSentences} held out, ${data.replaySentences} of ${SOURCE_SENTENCES.length} ${source.name.toLowerCase()} replayed`}
          title="Does the fine-tune generalize, and what does replay cost?"
          aside={<span className="tg-badge">{withReplay.steps} steps</span>}
        />
        <div className="ftt-replay__controls">
          <RangeControl
            label="Replay share"
            min={0}
            max={REPLAY_MAX}
            step={REPLAY_STEP}
            value={replayShare}
            format={(value) => {
              const sentences = Math.round((value / 100) * SOURCE_SENTENCES.length);
              return value === 0 ? "no replay" : `${value}% of the original · ${sentences} sentences`;
            }}
            onChange={(value) => {
              setState({ replayShare: value });
              narrate(`Replay ${value} percent of ${source.name}.`);
            }}
          />
          <RangeControl
            label="Held-out target sentences"
            min={HELD_OUT_MIN}
            max={HELD_OUT_MAX}
            step={1}
            value={heldOutSentences}
            format={(value) =>
              value === 0
                ? "none held out · all 8 trained on"
                : `${value} held out · ${TARGET_SENTENCES.length - value} trained on`
            }
            onChange={(value) => {
              setState({ heldOutSentences: value });
              narrate(`${value} target sentences held out.`);
            }}
          />
        </div>
        <LineChart
          label="Trained-on target, held-out target, and original loss during fine-tuning"
          xLabel="fine-tuning progress"
          yLabel="loss (nats/char)"
          yDomain={replayChart.domain}
          series={[
            {
              id: "trained",
              name: `${targetCorpus.name}, trained on`,
              tone: "forward",
              points: replayChart.trained.map((value, index) => ({
                x: progressAt(index, replayChart.trained.length),
                y: value,
              })),
            },
            {
              id: "held",
              name: `${targetCorpus.name}, held out`,
              tone: "attention",
              dash: "dotted",
              points: data.heldSentences
                ? replayChart.held.map((value, index) => ({
                    x: progressAt(index, replayChart.held.length),
                    y: value,
                  }))
                : [],
            },
            {
              id: "original",
              name: `${source.name} (forgetting)`,
              tone: "loss",
              dash: "dashed",
              points: replayChart.original.map((value, index) => ({
                x: progressAt(index, replayChart.original.length),
                y: value,
              })),
            },
          ]}
          footnote={
            data.heldSentences
              ? "One model per setting, scored at every saved checkpoint. The trained-on and held-out lines score different sentences of the same corpus; only the first was ever in the loss."
              : "Nothing is held out, so the trained-on line scores every target sentence it trained on and flatters the fit."
          }
        />
        <table className="tg-board ftt-replay-table">
          <caption className="ftt-rows__title">
            {data.replaySentences === 0
              ? "No replay: the loss mentions only the target sentences. Lower is better; forgetting is the change from the base model."
              : `With ${replayShare}% replay: ${data.replaySentences} ${source.name.toLowerCase()} sentences, ${Math.round(data.replayFraction * 100)}% of the fine-tuning sentences. Lower is better; forgetting is the change from the base model.`}
          </caption>
          <thead>
            <tr>
              <th scope="col">Loss in nats per character</th>
              <th scope="col">No replay</th>
              {data.replaySentences > 0 && <th scope="col">With {replayShare}% replay</th>}
            </tr>
          </thead>
          <tbody>
            {replayRows.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{row.plain}</td>
                {data.replaySentences > 0 && <td>{row.replay}</td>}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="lab-note">
          This card runs its own fine-tune on the settings above plus its two sliders, with every row free,
          so Layer freezing does not apply here and nothing on the other cards changes. A fine-tune with
          replay also takes more steps per epoch, because the replayed sentences are extra training pairs.
        </p>
      </LabSurface>

      <LabSurface label="Should you fine-tune? Rule of thumb" className="ftt-decision-panel ftt-panel">
        <SurfaceHeading
          kicker="A hand-written rule of thumb, not a measurement"
          title="Four questions, one recommended route"
          aside={
            <span className="tg-badge">
              rule {recommendation.rule + 1} of {DECISION_RULES.length} matched
            </span>
          }
        />
        <div className="ftt-decision__controls">
          {DECISION_KEYS.map((key) => (
            <SegmentedControl
              key={key}
              label={DECISION_QUESTIONS[key]}
              value={answers[key] ? "yes" : "no"}
              options={[
                { value: "yes", label: "Yes" },
                { value: "no", label: "No" },
              ]}
              onChange={(value) => answerDecision(key, value === "yes")}
            />
          ))}
        </div>
        <div className="ftt-decision__result">
          <span className="ftt-decision__kicker">Recommended route</span>
          <strong className="ftt-decision__route" role="status">
            {recommendation.route.title}
          </strong>
          <p>{recommendation.because}</p>
          <p>
            <b>Next:</b> {recommendation.route.next}
          </p>
        </div>
        <ul className="ftt-decision__lines" aria-label="What each of your four answers says">
          {recommendation.lines.map((line) => (
            <li key={line.key}>
              <b>
                {DECISION_LABELS[line.key]}: {line.value ? "Yes" : "No"}.
              </b>{" "}
              {line.text}
            </li>
          ))}
        </ul>
        <div className="ftt-decision__rules">
          <span className="ftt-rows__title">The whole rule table. The first row that matches wins.</span>
          <ol aria-label="Rule table, first match wins">
            {DECISION_RULES.map((rule, index) => (
              <li
                key={rule.when}
                className={index === recommendation.rule ? "is-matched" : ""}
                aria-current={index === recommendation.rule ? "true" : undefined}
              >
                <span>{rule.when}</span> <span aria-hidden="true">→</span> <strong>{ROUTES[rule.route].title}</strong>
                {index === recommendation.rule ? <em> (matched)</em> : null}
              </li>
            ))}
          </ol>
        </div>
        <p className="lab-note">
          This card is a rule of thumb that someone wrote by hand. Nothing on it was measured, and
          the loss curves above are the only measurements in this lab. Real decisions also weigh
          cost, latency, privacy and risk, which these four questions leave out.
        </p>
      </LabSurface>
    </div>
  );
}
