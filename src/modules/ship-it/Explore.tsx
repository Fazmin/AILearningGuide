import { useMemo } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  teachingServing,
  type ModuleContext,
} from "@app/module-sdk";
import { FORMAT_NAMES, RECIPE_BOUNDS, STAGE_LABELS, runPipeline } from "./pipeline";
import { normalizeRecipe } from "./state";

const { formatBytes, GIB, HARDWARE } = teachingServing;

const nats = (value: number) => value.toFixed(3);
const signed = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
const tokens = (value: number) => (value >= 100 ? value.toFixed(0) : value.toFixed(1));
/** A stage-to-stage change smaller than this is noise in a table of three decimals. */
const MOVED = 0.02;

function Change({ value }: { value: number }) {
  const worse = value > MOVED;
  const better = value < -MOVED;
  return (
    <small className={worse ? "is-worse" : better ? "is-better" : ""}>
      {signed(value)}
      {worse ? " worse" : better ? " better" : ""}
    </small>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const recipe = normalizeRecipe(state);
  const key = JSON.stringify(recipe);
  // The whole pipeline is a function of the recipe, so it is recomputed rather than stored.
  const release = useMemo(() => runPipeline(recipe), [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const { stages, gate, shipping, alignment } = release;
  const serving = shipping.serving;
  const first = stages[0];
  const last = stages[stages.length - 1];
  const headroom = HARDWARE.vramBytes - serving.totalBytes;
  const [targetLow, targetHigh] = RECIPE_BOUNDS.gateTarget;
  const [controlLow, controlHigh] = RECIPE_BOUNDS.gateControl;

  const set = (patch: Record<string, string | number>, message: string) => {
    setState(patch);
    narrate(message);
  };

  const advice: string[] = [];
  if (gate.checks.find((check) => check.id === "control" && !check.passed)) {
    advice.push(
      "The harbor check failed: raise Replay sentences, or lower Adapter epochs or LoRA rank, so the adapter has less room to move away from harbor text.",
    );
  }
  if (gate.checks.find((check) => check.id === "target" && !check.passed)) {
    advice.push(
      "The recipe check failed: adapt for longer or at a higher rank, or prune less. A change that only trims the harbor loss has not learned the task.",
    );
  }
  if (!serving.fits) {
    advice.push("The budget does not fit: lower Concurrent users or Context length, or choose a smaller Weight format.");
  }

  return (
    <div className="tg-lab ship-lab">
      <LabSurface label="Training recipe" className="ship-recipe">
        <SurfaceHeading kicker="Stages 1 to 3" title="Pretrain, adapt, align" />
        <div className="ship-controls">
          <RangeControl
            label="Base training epochs"
            min={RECIPE_BOUNDS.baseEpochs[0]}
            max={RECIPE_BOUNDS.baseEpochs[1]}
            step={1}
            value={recipe.baseEpochs}
            format={(value) => `${value} epochs`}
            onChange={(value) => set({ baseEpochs: value }, `Base model trains for ${value} epochs on harbor text.`)}
          />
          <RangeControl
            label="LoRA rank"
            min={RECIPE_BOUNDS.loraRank[0]}
            max={RECIPE_BOUNDS.loraRank[1]}
            step={1}
            value={recipe.loraRank}
            format={(value) => `rank ${value}`}
            onChange={(value) => set({ loraRank: value }, `Adapter rank ${value}.`)}
          />
          <RangeControl
            label="Adapter epochs"
            min={RECIPE_BOUNDS.loraEpochs[0]}
            max={RECIPE_BOUNDS.loraEpochs[1]}
            step={1}
            value={recipe.loraEpochs}
            format={(value) => `${value} epochs`}
            onChange={(value) => set({ loraEpochs: value }, `Adapter trains for ${value} epochs on recipe text.`)}
          />
          <RangeControl
            label="Replay sentences"
            min={RECIPE_BOUNDS.replay[0]}
            max={RECIPE_BOUNDS.replay[1]}
            step={1}
            value={recipe.replay}
            format={(value) => (value === 0 ? "none" : `${value} of 6 harbor sentences`)}
            onChange={(value) =>
              set({ replay: value }, value === 0 ? "No harbor text mixed into adaptation." : `${value} harbor sentences mixed into adaptation.`)
            }
          />
          <RangeControl
            label="DPO steps"
            min={RECIPE_BOUNDS.dpoSteps[0]}
            max={RECIPE_BOUNDS.dpoSteps[1]}
            step={1}
            value={recipe.dpoSteps}
            format={(value) => (value === 0 ? "off" : `${value} steps`)}
            onChange={(value) => set({ dpoSteps: value }, value === 0 ? "Preference training off." : `DPO runs ${value} steps.`)}
          />
          <RangeControl
            label="DPO beta"
            min={RECIPE_BOUNDS.beta[0]}
            max={RECIPE_BOUNDS.beta[1]}
            step={0.05}
            value={recipe.beta}
            format={(value) => value.toFixed(2)}
            onChange={(value) => set({ beta: value }, `DPO beta ${value.toFixed(2)}.`)}
          />
        </div>
        <p className="lab-note">
          Stage 1 trains the 900-weight character table on six harbor sentences. Stage 2 trains a low-rank adapter on the
          first six recipe sentences, plus any replayed harbor sentences, and merges it. Stage 3 runs DPO on three
          preference pairs, with the adapted model as its reference. The last two sentences of each corpus are never
          trained on; every loss below is read from them.
        </p>
      </LabSurface>

      <LabSurface label="Stage by stage" className="ship-stages">
        <SurfaceHeading kicker="Held-out text" title="What each stage did to the model" />
        <table className="ship-table">
          <caption>Held-out loss in nats per character, and benchmark items answered correctly out of six</caption>
          <thead>
            <tr>
              <th scope="col">Stage</th>
              <th scope="col">Recipe text</th>
              <th scope="col">Harbor text</th>
              <th scope="col">Recipe items</th>
              <th scope="col">Harbor items</th>
            </tr>
          </thead>
          <tbody>
            {stages.map((stage, index) => {
              const previous = stages[index - 1];
              return (
                <tr key={stage.id}>
                  <th scope="row">{STAGE_LABELS[stage.id]}</th>
                  <td>
                    {nats(stage.metrics.targetLoss)}{" "}
                    {previous && <Change value={stage.metrics.targetLoss - previous.metrics.targetLoss} />}
                  </td>
                  <td>
                    {nats(stage.metrics.controlLoss)}{" "}
                    {previous && <Change value={stage.metrics.controlLoss - previous.metrics.controlLoss} />}
                  </td>
                  <td>{stage.metrics.target.correct} of {stage.metrics.target.total}</td>
                  <td>{stage.metrics.control.correct} of {stage.metrics.control.total}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric
            label="Recipe loss, base to shipped"
            value={signed(last.metrics.targetLoss - first.metrics.targetLoss)}
            tone={last.metrics.targetLoss <= first.metrics.targetLoss ? "forward" : "loss"}
          />
          <Metric
            label="Harbor loss, base to shipped"
            value={signed(last.metrics.controlLoss - first.metrics.controlLoss)}
            tone={last.metrics.controlLoss <= first.metrics.controlLoss ? "forward" : "loss"}
          />
          <Metric
            label="Preference pairs ordered"
            value={alignment ? `${Math.round(alignment.accuracy * 100)}%` : "DPO off"}
          />
          <Metric label="Reward margin" value={alignment ? alignment.margin.toFixed(2) : "n/a"} />
        </div>
        <div className={`tg-callout ${release.regressions.length ? "is-warning" : "is-quiet"}`}>
          <strong>
            {release.regressions.length
              ? `${release.regressions.length} stage${release.regressions.length === 1 ? "" : "s"} made something worse.`
              : "No stage made either text worse by more than 0.02 nats."}
          </strong>
          {release.regressions.length > 0 && (
            <ul className="ship-regressions">
              {release.regressions.map((regression) => (
                <li key={regression}>{regression}</li>
              ))}
            </ul>
          )}
        </div>
      </LabSurface>

      <LabSurface label="Eval gate" className="ship-gate">
        <SurfaceHeading
          kicker="Run on the artifact you would ship"
          title="Does it clear the bar?"
          aside={<span className={`tg-badge ${gate.passed ? "" : "is-warning"}`.trim()}>{gate.passed ? "Gate: pass" : "Gate: fail"}</span>}
        />
        <div className="ship-controls">
          <RangeControl
            label="Required recipe-loss gain"
            min={targetLow}
            max={targetHigh}
            step={0.005}
            value={recipe.gateTarget}
            format={(value) => `${nats(value)} nats`}
            onChange={(value) => set({ gateTarget: value }, `Recipe loss must fall by at least ${nats(value)} nats.`)}
          />
          <RangeControl
            label="Allowed harbor-loss rise"
            min={controlLow}
            max={controlHigh}
            step={0.01}
            value={recipe.gateControl}
            format={(value) => `${nats(value)} nats`}
            onChange={(value) => set({ gateControl: value }, `Harbor loss may rise by at most ${nats(value)} nats.`)}
          />
        </div>
        <ul className="ship-gate-checks">
          {gate.checks.map((check) => (
            <li key={check.id} className={check.passed ? "is-pass" : "is-fail"}>
              <b>{check.passed ? "Pass" : "Fail"}</b>
              <strong>{check.id === "target" ? "Gains on the recipe text" : "Keeps the harbor text"}</strong>
              <span>{check.detail}</span>
            </li>
          ))}
        </ul>
        <p className="lab-note">{gate.advisory}</p>
      </LabSurface>

      <LabSurface label="Compress and serve" className="ship-compress">
        <SurfaceHeading kicker="Stages 4 and 5" title="Shrink it, then fit it" />
        <div className="ship-controls">
          <SegmentedControl
            label="Weight format"
            value={recipe.format}
            options={FORMAT_NAMES.map((name) => ({ value: name, label: name }))}
            onChange={(value) => set({ format: value }, `Weight format ${value}.`)}
          />
          <RangeControl
            label="Prune"
            min={RECIPE_BOUNDS.prune[0]}
            max={RECIPE_BOUNDS.prune[1]}
            step={0.05}
            value={recipe.prune}
            format={(value) => `${Math.round(value * 100)}% of weights`}
            onChange={(value) => set({ prune: value }, `${Math.round(value * 100)} percent of weights pruned.`)}
          />
          <RangeControl
            label="Concurrent users"
            min={RECIPE_BOUNDS.users[0]}
            max={RECIPE_BOUNDS.users[1]}
            step={1}
            value={recipe.users}
            format={(value) => `${value} streams`}
            onChange={(value) => set({ users: value }, `${value} concurrent streams.`)}
          />
          <RangeControl
            label="Context length"
            min={RECIPE_BOUNDS.context[0]}
            max={RECIPE_BOUNDS.context[1]}
            step={512}
            value={recipe.context}
            format={(value) => `${value.toLocaleString("en-US")} tokens`}
            onChange={(value) => set({ context: value }, `Context ${value} tokens.`)}
          />
        </div>
        <div className="metric-row">
          <Metric label="Toy file at this format" value={formatBytes(shipping.toyBytes)} />
          <Metric label="Toy bits per weight" value={shipping.toyBitsPerWeight.toFixed(2)} />
          <Metric
            label="Recipe loss, as shipped"
            value={`${nats(last.metrics.targetLoss)}`}
            tone={last.metrics.targetLoss - stages[2].metrics.targetLoss > MOVED ? "loss" : "forward"}
          />
          <Metric
            label="Harbor loss, as shipped"
            value={`${nats(last.metrics.controlLoss)}`}
            tone={last.metrics.controlLoss - stages[2].metrics.controlLoss > MOVED ? "loss" : "forward"}
          />
        </div>
        <div className="metric-row">
          <Metric label="8B-class weights" value={formatBytes(serving.weightBytes)} />
          <Metric label="KV cache for all streams" value={formatBytes(serving.kvBytes)} />
          <Metric label="Streams that fit" value={`${serving.maxStreams}`} tone={serving.maxStreams >= recipe.users ? "forward" : "loss"} />
          <Metric label="Per user" value={`${tokens(serving.perUserTokensPerSecond)} tok/s`} />
        </div>
        <div className={`tg-callout ${serving.fits ? "is-quiet" : "is-warning"}`}>
          <strong>
            {serving.fits
              ? `The imagined 8B-class budget fits: ${formatBytes(serving.totalBytes)} of ${formatBytes(HARDWARE.vramBytes)}, ${formatBytes(Math.max(0, headroom))} spare.`
              : `Over budget by ${formatBytes(-headroom)}: ${formatBytes(serving.totalBytes)} against ${formatBytes(HARDWARE.vramBytes)}.`}
          </strong>
          <span>
            The quality columns come from the 900-weight toy you just trained. The budget is first-order arithmetic for
            an 8B-class model (Llama-3.1-8B shape) on {HARDWARE.name}, with the same stated assumptions as the serving
            lab: bandwidth-bound decode, an fp16 cache sized for the full context of every stream, and a flat{" "}
            {formatBytes(HARDWARE.overheadBytes)} of overhead. Nothing here was measured on real hardware.
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Release report" className="ship-report">
        <SurfaceHeading
          kicker="Gate and budget together"
          title={release.verdict === "ship" ? "Ship it" : "Hold the release"}
          aside={<span className={`tg-badge ${release.verdict === "ship" ? "" : "is-warning"}`.trim()}>{release.verdict === "ship" ? "Verdict: ship" : "Verdict: hold"}</span>}
        />
        <dl className="ship-report-list">
          <div>
            <dt>Artifact</dt>
            <dd>
              {shipping.format}, {recipe.prune > 0 ? `${Math.round(recipe.prune * 100)}% pruned` : "not pruned"}, merged adapter of rank {recipe.loraRank}
            </dd>
          </div>
          <div>
            <dt>Quality gate</dt>
            <dd>{gate.passed ? "Both checks passed" : `${gate.checks.filter((check) => !check.passed).length} of 2 checks failed`}</dd>
          </div>
          <div>
            <dt>Serving budget</dt>
            <dd>
              {serving.fits ? "Fits" : "Does not fit"} at {recipe.users} users and {recipe.context.toLocaleString("en-US")} tokens ({(serving.totalBytes / GIB).toFixed(2)} GiB)
            </dd>
          </div>
          <div>
            <dt>Cost of getting here</dt>
            <dd>
              {release.regressions.length
                ? release.regressions.join("; ")
                : "no stage made either text worse by more than 0.02 nats"}
            </dd>
          </div>
        </dl>
        {release.verdict === "hold" ? (
          <div className="tg-callout is-warning">
            <strong>Why it is on hold</strong>
            <ul className="ship-regressions">
              {release.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
            {advice.length > 0 && <span>{advice.join(" ")}</span>}
          </div>
        ) : (
          <div className="tg-callout is-quiet">
            <strong>Every condition holds.</strong>
            <span>
              Passing is a statement about two held-out texts and one imagined machine, not about a real model. The
              lessons below name what a real release adds: a larger and fresher evaluation set, safety and regression
              suites for the application, a staged rollout, and monitoring after it.
            </span>
          </div>
        )}
      </LabSurface>
    </div>
  );
}
