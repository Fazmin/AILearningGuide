import { useEffect, useMemo, useRef, useState } from "react";
import { BarList, LineChart, Metric, RangeControl, type BarListItem, type ModuleContext } from "@app/module-sdk";
import cache from "./assets/sae-top-activations.json";
import { showChar } from "./FeatureViews";
import { formatChange, formatProbability, formatScore, pct } from "./format";
import { layerTwoResidual, loadTransformerWeights, type SaeRun, type Status } from "./runtime";
import { WIDTH, testLabel, type SaeWeights } from "./sae";
import {
  decoderDirection,
  distributionOf,
  greedyContinuation,
  labelText,
  LABELLED_FEATURES,
  labelledFeature,
  largestShifts,
  MAX_STRENGTH,
  nextLogits,
  NULL_DIRECTIONS,
  nullDirections,
  prefixRows,
  steeredResidual,
  strengthSweep,
  topTokens,
  type TransformerWeights,
} from "./steering";
import { showToken, tokenId } from "./vocabulary";

const STRENGTHS = Array.from({ length: MAX_STRENGTH + 1 }, (_, index) => index);
const CONTINUATION_STEPS = 12;
const TOP = 6;
const scale = cache.normalization_scale;

interface Chain {
  name: string;
  text: string;
  /** The smallest logit gap between the chosen character and the runner-up over the steps. */
  margin: number;
}

type Continuation = { key: string; chains: Chain[] } | { key: string; error: string };

export interface SteerViewProps {
  text: string;
  position: number;
  feature: number;
  strength: number;
  seed: number;
  run: SaeRun | undefined;
  /** Whether `run` was computed for the current text. */
  aligned: boolean;
  runStatus: Status;
  sae: SaeWeights | undefined;
  setState: ModuleContext["setState"];
  narrate: ModuleContext["narrate"];
}

const characterItems = (
  distribution: ReturnType<typeof distributionOf>,
  target: number,
  ceiling: number,
  tone: BarListItem["tone"],
): BarListItem[] => {
  const top = topTokens(distribution, TOP);
  const items: BarListItem[] = top.map((entry) => ({
    id: String(entry.id),
    label: `${showToken(entry.id)}${entry.id === target ? " (label)" : ""}`,
    value: entry.probability,
    display: formatProbability(entry.probability),
    tone: entry.id === target ? "attention" : tone,
  }));
  if (!top.some((entry) => entry.id === target)) {
    const probability = distribution.probabilities[target] ?? 0;
    items.push({
      id: String(target),
      label: `${showToken(target)} (label)`,
      value: Math.min(probability, ceiling),
      display: formatProbability(probability),
      tone: "attention",
    });
  }
  return items;
};

export function SteerView({ text, position, feature, strength, seed, run, aligned, runStatus, sae, setState, narrate }: SteerViewProps) {
  const characters = useMemo(() => Array.from(text), [text]);
  const entry = labelledFeature(feature) ?? LABELLED_FEATURES[0];
  const target = tokenId(entry.character);

  const [weights, setWeights] = useState<{ status: Status; value?: TransformerWeights }>({ status: "loading" });
  useEffect(() => {
    let active = true;
    loadTransformerWeights()
      .then((value) => active && setWeights({ status: "ready", value }))
      .catch(() => active && setWeights({ status: "error" }));
    return () => {
      active = false;
    };
  }, []);

  const direction = useMemo(() => (sae ? decoderDirection(sae, entry.feature) : null), [sae, entry.feature]);
  const prefix = useMemo(
    () => (aligned && run && (position + 1) * WIDTH <= run.residual.length ? prefixRows(run.residual, position) : null),
    [aligned, run, position],
  );
  const model = weights.value;

  const sweep = useMemo(
    () => (model && prefix && direction ? strengthSweep(model, prefix, direction, target, scale, STRENGTHS) : null),
    [model, prefix, direction, target],
  );

  const result = useMemo(() => {
    if (!model || !prefix || !direction) return null;
    const raw = prefix.subarray(position * WIDTH, (position + 1) * WIDTH);
    const before = distributionOf(nextLogits(model, prefix));
    const after = distributionOf(nextLogits(model, prefix, steeredResidual(raw, direction, strength, scale)));
    const control = distributionOf(
      nextLogits(model, prefix, steeredResidual(raw, nullDirections()[seed - 1] ?? nullDirections()[0], strength, scale)),
    );
    return { before, after, control, shifts: largestShifts(before, after, 3) };
  }, [model, prefix, direction, position, strength, seed]);

  const row = sweep?.[strength] ?? null;
  const delta = result ? result.after.logProbabilities[target] - result.before.logProbabilities[target] : null;
  const controlDelta = result ? result.control.logProbabilities[target] - result.before.logProbabilities[target] : null;
  const labelTest = useMemo(
    () =>
      aligned && run
        ? testLabel(text, run.features.map((features) => features[entry.feature] ?? 0), entry.threshold, "char", entry.character)
        : null,
    [aligned, run, text, entry],
  );

  const ceiling = result
    ? Math.max(1e-9, topTokens(result.before, 1)[0].probability, topTokens(result.after, 1)[0].probability)
    : 1;
  const residualLength = useMemo(() => {
    if (!prefix) return null;
    let total = 0;
    for (let index = position * WIDTH; index < (position + 1) * WIDTH; index += 1) total += prefix[index] ** 2;
    return Math.sqrt(total);
  }, [prefix, position]);
  const heard = characters.slice(0, position + 1).join("");
  const shownPrefix = Array.from(heard).slice(-24).map(showChar).join("");
  const next = characters[position + 1];

  // ---- A short greedy continuation, run on demand: three chains of ONNX calls.
  const [continuation, setContinuation] = useState<Continuation | null>(null);
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const key = [text, position, feature, strength, seed].join("|");
  const room = model ? Math.min(CONTINUATION_STEPS, model.context - (position + 1)) : 0;
  const generate = async () => {
    if (!model || !direction || busy || room <= 0) return;
    setBusy(true);
    narrate("Generating three short continuations.");
    try {
      const chains: Chain[] = [];
      const plans = [
        { name: "No steering", steer: undefined },
        { name: `#${entry.feature} added`, steer: { direction, strength, scale } },
        { name: `Random direction ${seed}`, steer: { direction: nullDirections()[seed - 1] ?? nullDirections()[0], strength, scale } },
      ];
      for (const plan of plans) {
        const made = await greedyContinuation({ weights: model, prompt: heard, steps: room, residuals: layerTwoResidual, steer: plan.steer });
        chains.push({ name: plan.name, text: Array.from(made.text).map(showChar).join(""), margin: Math.min(...made.margins) });
      }
      if (mounted.current) setContinuation({ key, chains });
      narrate("Continuations ready.");
    } catch (error) {
      if (mounted.current) setContinuation({ key, error: error instanceof Error ? error.message : String(error) });
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  const yValues = sweep ? sweep.flatMap((entryRow) => [entryRow.feature, entryRow.nullMin, entryRow.nullMax]) : [0];
  const yDomain: [number, number] = [Math.floor(Math.min(0, ...yValues)), Math.ceil(Math.max(1, ...yValues))];

  const verdict =
    result && delta !== null && row
      ? strength === 0
        ? "Strength 0 adds nothing, so the before and after lists are identical. Raise the strength."
        : `Adding feature #${entry.feature} at strength ${strength} changed log p(“${showChar(entry.character)}”) by ${formatChange(delta)}. ` +
          `${NULL_DIRECTIONS} random directions of the same length changed it by ${formatChange(row.nullMin)} to ${formatChange(row.nullMax)} ` +
          `(mean ${formatChange(row.nullMean)}, standard deviation ${row.nullSd.toFixed(2)}), so this change is ${Math.abs(row.score).toFixed(1)} standard deviations ${row.score < 0 ? "below" : "above"} their mean: ` +
          `${
            Math.abs(row.score) >= 3
              ? "far outside the random spread, so it belongs to this direction."
              : Math.abs(row.score) >= 2
                ? "outside the random spread, but not by much, which is weak evidence."
                : "inside the random spread, so it cannot be told apart from the side effects of adding any vector this long."
          } ` +
          `${
            Math.abs(row.score) < 2
              ? ""
              : delta > 0
                ? "The label character became more likely."
                : "The label character became less likely, not more."
          }`
      : weights.status === "error"
        ? "The transformer weights did not load, so nothing can be steered."
        : runStatus === "error"
          ? "The models did not load, so there is no residual to steer."
          : "Waiting for the models and the decoder directions…";

  return (
    <>
      <div className="sf-steer-controls">
        <label className="sf-select">
          <span>Feature to steer</span>
          <select
            value={entry.feature}
            onChange={(event) => {
              setState({ steerFeature: Number(event.target.value) });
              narrate(`Steering feature ${event.target.value}.`);
            }}
          >
            {LABELLED_FEATURES.map((option) => (
              <option key={option.feature} value={option.feature}>
                {`#${option.feature} · ${labelText(option)}`}
              </option>
            ))}
          </select>
        </label>
        <RangeControl
          label="Steering position"
          min={0}
          max={Math.max(0, characters.length - 1)}
          step={1}
          value={position}
          format={(value) => `${value}: “${showChar(characters[value] ?? "")}”`}
          onChange={(value) => setState({ steerAt: value })}
        />
        <RangeControl
          label="Steering strength"
          min={0}
          max={MAX_STRENGTH}
          step={1}
          value={strength}
          format={(value) => `+${value}`}
          onChange={(value) => setState({ strength: value })}
        />
        <RangeControl
          label="Random control direction"
          min={1}
          max={NULL_DIRECTIONS}
          step={1}
          value={seed}
          format={(value) => `seed ${value}`}
          onChange={(value) => setState({ seed: value })}
        />
      </div>

      <p className="sf-hypothesis">
        The model has read “{Array.from(heard).length > 24 ? "…" : ""}{shownPrefix}” and is predicting the next character
        {next === undefined ? "" : `; your text continues with “${showChar(next)}”`}. Strength s adds s × the decoder direction to the
        layer-2 residual at position {position}, which raises the feature&apos;s activation by about s
        {residualLength ? `; at strength ${strength} that is a vector ${pct((strength * scale) / residualLength)} as long as the residual there` : ""}.{" "}
        {labelTest
          ? labelTest.truePositive + labelTest.falsePositive + labelTest.falseNegative === 0
            ? `On your text the label “${labelText(entry)}” has nothing to score: no matching character and no firing. Type some into the probe text to see it.`
            : `On your text, the label “${labelText(entry)}” has ${labelTest.truePositive} hit${labelTest.truePositive === 1 ? "" : "s"}, ${labelTest.falseNegative} miss${labelTest.falseNegative === 1 ? "" : "es"} and ${labelTest.falsePositive} false alarm${labelTest.falsePositive === 1 ? "" : "s"} (precision ${pct(labelTest.precision)}, recall ${pct(labelTest.recall)}).`
          : ""}
      </p>

      {result ? (
        <div className="sf-steer-lists">
          <div>
            <strong>Before: unsteered</strong>
            <BarList
              label="Most likely next characters before steering"
              max={ceiling}
              items={characterItems(result.before, target, ceiling, "forward")}
            />
          </div>
          <div>
            <strong>After: +{strength} along #{entry.feature}</strong>
            <BarList
              label="Most likely next characters after steering along the feature direction"
              max={ceiling}
              items={characterItems(result.after, target, ceiling, "gradient")}
            />
          </div>
          <div>
            <strong>
              Control: +{strength} along random direction {seed}
              {controlDelta === null ? "" : ` (log p ${formatChange(controlDelta)})`}
            </strong>
            <BarList
              label="Most likely next characters after adding a random direction of the same length"
              max={ceiling}
              items={characterItems(result.control, target, ceiling, "muted")}
            />
          </div>
        </div>
      ) : (
        <p className="sf-empty" role="status">
          {weights.status === "error" ? "The transformer weights did not load." : "Waiting for the models and the decoder directions…"}
        </p>
      )}

      <div className="metric-row">
        <Metric
          label={`p(“${showChar(entry.character)}”) before → after`}
          value={result ? `${formatProbability(result.before.probabilities[target])} → ${formatProbability(result.after.probabilities[target])}` : "—"}
          tone="forward"
        />
        <Metric label="Change in log p" value={delta === null ? "—" : formatChange(delta)} tone="gradient" />
        <Metric
          label={`${NULL_DIRECTIONS} random: min · median · max`}
          value={row ? `${formatChange(row.nullMin)} · ${formatChange(row.nullMedian)} · ${formatChange(row.nullMax)}` : "—"}
        />
        <Metric label="Feature vs random, in SDs" value={row ? (strength === 0 ? "—" : formatScore(row.score)) : "—"} tone="loss" />
      </div>
      <p className="sf-small" role="status">{verdict}</p>
      {result && strength > 0 && (
        <p className="sf-small">
          Largest rises in log probability:{" "}
          {result.shifts.gains.map((shift) => `“${showToken(shift.id)}” ${formatChange(shift.delta)}`).join(", ")}. Largest falls:{" "}
          {result.shifts.losses.map((shift) => `“${showToken(shift.id)}” ${formatChange(shift.delta)}`).join(", ")}.
        </p>
      )}

      {sweep && (
        <>
          <LineChart
            label={`Change in log probability of “${entry.character}” against steering strength`}
            series={[
              { id: "feature", name: `#${entry.feature} direction`, tone: "attention", points: sweep.map((r) => ({ x: r.strength, y: r.feature })), format: formatChange },
              { id: "median", name: "Random median", tone: "muted", dash: "dashed", points: sweep.map((r) => ({ x: r.strength, y: r.nullMedian })), format: formatChange },
              { id: "max", name: "Random max", tone: "loss", dash: "dotted", points: sweep.map((r) => ({ x: r.strength, y: r.nullMax })), format: formatChange },
              { id: "min", name: "Random min", tone: "loss", dash: "dotted", points: sweep.map((r) => ({ x: r.strength, y: r.nullMin })), format: formatChange },
            ]}
            xLabel="steering strength"
            yLabel="change in log p of the label character"
            xDomain={[0, MAX_STRENGTH]}
            yDomain={yDomain}
            marker={{ x: strength, label: `+${strength}` }}
            footnote={`At position ${position}. The dotted lines bound ${NULL_DIRECTIONS} fixed random directions of the same length; a change inside them, or within about two standard deviations of their mean, is not evidence about this feature.`}
          />
          <details className="sf-table-wrap">
            <summary>Strength sweep as a table</summary>
            <table className="sf-table">
              <caption>
                Change in log probability of “{showChar(entry.character)}” at position {position}, by strength
              </caption>
              <thead>
                <tr>
                  <th scope="col">Strength</th>
                  <th scope="col">#{entry.feature} direction</th>
                  <th scope="col">Random min</th>
                  <th scope="col">Random median</th>
                  <th scope="col">Random max</th>
                </tr>
              </thead>
              <tbody>
                {sweep.map((r) => (
                  <tr key={r.strength} aria-current={r.strength === strength ? "true" : undefined}>
                    <th scope="row">+{r.strength}</th>
                    <td>{formatChange(r.feature)}</td>
                    <td>{formatChange(r.nullMin)}</td>
                    <td>{formatChange(r.nullMedian)}</td>
                    <td>{formatChange(r.nullMax)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      )}

      <div className="sf-continue">
        <button type="button" className="sf-button" onClick={generate} disabled={!model || !direction || busy || room <= 0 || !aligned}>
          {busy ? "Generating…" : room > 0 ? `Continue ${room} characters` : "No room left in the window"}
        </button>
        <span className="sf-small">
          Greedy: each step takes the single most likely character, with the same push added at every step. Runs three chains of {room > 0 ? room : "up to 12"} steps.
        </span>
      </div>
      {continuation && continuation.key === key && "chains" in continuation && (
        <ul className="sf-chains" aria-label="Greedy continuations">
          {continuation.chains.map((chain) => (
            <li key={chain.name}>
              <b>{chain.name}</b>
              <code>
                {Array.from(heard).slice(-12).map(showChar).join("")}
                <mark>{chain.text}</mark>
              </code>
            </li>
          ))}
        </ul>
      )}
      {continuation && continuation.key === key && "error" in continuation && (
        <p className="sf-empty" role="status">{continuation.error}</p>
      )}
      {continuation && continuation.key !== key && (
        <p className="sf-small">The settings changed since the last continuation; press the button to run it again.</p>
      )}
    </>
  );
}
