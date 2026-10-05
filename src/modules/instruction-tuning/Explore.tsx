import { useMemo } from "react";
import {
  BarList,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  tinyTopTokens,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  encodeSegments,
  firstMarkerAt,
  HELD_OUT_REPLY,
  inLoss,
  isServeMode,
  MAX_DATASET_LINE_LENGTH,
  MAX_DATASET_LINES,
  probeServing,
  renderExample,
  renderPlain,
  responseLogProb,
  sanitizeDataset,
  SERVE_MODES,
  SYSTEM_TEXT,
  surprisals,
  trainMasked,
  transitionsFor,
  type EncodedCharacter,
  type LossMask,
  type ServeMode,
} from "./sft";

const DEFAULT_PAIRS = [
  "when does the fog arrive | the fog settles over the harbor in the morning.",
  "how do i start the sauce | warm the pan and add a spoon of oil.",
  "what happens by noon | the clouds break apart and the sun warms the streets.",
  "how long should it simmer | simmer the sauce until it thickens.",
];

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asBoolean = (state: ModuleContext["state"], key: string, fallback: boolean) =>
  typeof state[key] === "boolean" ? state[key] : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

const MAX_SURPRISAL = 5;

function LossMaskStrip({
  sequence,
  values,
  mask,
}: {
  sequence: EncodedCharacter[];
  values: number[];
  mask: LossMask;
}) {
  let gradedTotal = 0;
  let gradedCount = 0;
  let contextTotal = 0;
  let contextCount = 0;
  sequence.forEach((_, index) => {
    if (index === 0) return;
    if (inLoss(sequence, index, mask)) {
      gradedTotal += values[index];
      gradedCount += 1;
    } else {
      contextTotal += values[index];
      contextCount += 1;
    }
  });
  // Group cells into words and markers so a line never breaks inside one.
  const groups: number[][] = [];
  sequence.forEach((entry, index) => {
    const previous = sequence[index - 1];
    const startsNew =
      index === 0 ||
      previous.character === " " ||
      entry.marker !== previous.marker ||
      (entry.marker && entry.character === "<");
    if (startsNew) groups.push([index]);
    else groups[groups.length - 1].push(index);
  });
  const label = `Loss mask over the first example, ${sequence.length} characters. ${gradedCount} are graded, averaging ${(gradedTotal / Math.max(1, gradedCount)).toFixed(2)} nats of surprise; ${contextCount} are context only, averaging ${(contextTotal / Math.max(1, contextCount)).toFixed(2)} nats.`;
  return (
    <div className="sft-strip">
      <div className="sft-strip__cells" role="img" aria-label={label}>
        {groups.map((group) => (
          <span className="sft-word" key={group[0]}>
            {group.map((index) => {
              const entry = sequence[index];
              const graded = inLoss(sequence, index, mask);
              const value = values[index];
              const height = Number.isFinite(value) ? Math.min(1, value / MAX_SURPRISAL) : 0;
              return (
                <span
                  key={index}
                  className={[
                    "sft-cell",
                    `sft-cell--${entry.role}`,
                    entry.marker ? "is-marker" : "",
                    graded ? "is-graded" : "is-context",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  title={
                    index === 0
                      ? "first character: nothing predicts it"
                      : `${graded ? "graded" : "not graded"} · −log p = ${value.toFixed(2)} nats`
                  }
                >
                  <i style={{ height: `${height * 100}%` }} />
                  <b>{entry.character === " " ? "␣" : entry.character}</b>
                </span>
              );
            })}
          </span>
        ))}
      </div>
      <div className="sft-strip__legend" aria-hidden="true">
        <span><i className="sft-key sft-key--graded" />graded: in the loss</span>
        <span><i className="sft-key sft-key--context" />context only: read, never graded</span>
        <span><i className="sft-key sft-key--bar" />bar height = −log p of that character, 0 to 5 nats</span>
        <span><i className="sft-key sft-key--system" />system</span>
        <span><i className="sft-key sft-key--user" />user</span>
        <span><i className="sft-key sft-key--assistant" />assistant</span>
      </div>
      <div className="metric-row sft-metrics">
        <Metric label="Graded characters" value={`${gradedCount} of ${Math.max(0, sequence.length - 1)}`} tone="forward" />
        <Metric label="Mean −log p, graded" value={`${(gradedTotal / Math.max(1, gradedCount)).toFixed(2)} nats`} tone="forward" />
        <Metric
          label="Mean −log p, context"
          value={contextCount ? `${(contextTotal / contextCount).toFixed(2)} nats` : "none"}
          tone="loss"
        />
      </div>
    </div>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const useSystem = asBoolean(state, "useSystem", true);
  const useRoles = asBoolean(state, "useRoles", true);
  const useEnd = asBoolean(state, "useEnd", true);
  const lossMask: LossMask = state.lossMask === "all" ? "all" : "assistant";
  const epochs = Math.min(80, Math.max(1, Math.round(asNumber(state, "epochs", 30))));
  const dataset = sanitizeDataset(asString(state, "dataset", DEFAULT_PAIRS.join("\n")));
  const serveWith: ServeMode = isServeMode(state.serveWith) ? state.serveWith : "template";

  const pairs = useMemo(
    () =>
      dataset
        .split("\n")
        .map((line) => line.split("|"))
        .filter((parts) => parts.length >= 2)
        .map((parts) => ({
          instruction: parts[0].trim().toLowerCase(),
          response: parts.slice(1).join("|").trim().toLowerCase(),
        }))
        .filter((pair) => pair.instruction && pair.response),
    [dataset],
  );
  const sequences = useMemo(() => {
    // An empty or unparseable editor still trains on something, so no panel divides by zero.
    const trainingPairs = pairs.length ? pairs : [{ instruction: "the", response: "the." }];
    const options = { system: useSystem, roles: useRoles, end: useEnd };
    return {
      trainingPairs,
      templated: trainingPairs.map((pair) => encodeSegments(renderExample(pair.instruction, pair.response, options))),
      plain: trainingPairs.map((pair) => encodeSegments(renderPlain(pair.instruction, pair.response))),
      firstSegments: renderExample(trainingPairs[0].instruction, trainingPairs[0].response, options),
    };
  }, [pairs, useEnd, useRoles, useSystem]);
  const { trainingPairs } = sequences;

  const runs = useMemo(() => {
    const templatedTransitions = transitionsFor(sequences.templated, lossMask);
    const plainTransitions = transitionsFor(sequences.plain, lossMask);
    return {
      templated: trainMasked({ transitions: templatedTransitions, epochs }),
      plain: trainMasked({ transitions: plainTransitions, epochs }),
    };
  }, [epochs, lossMask, sequences]);

  const first = sequences.templated[0];
  const firstPlain = sequences.plain[0];
  const firstValues = useMemo(() => surprisals(runs.templated.weights, first), [first, runs.templated.weights]);
  const templatedChars = sequences.templated.reduce((total, sequence) => total + sequence.length, 0);
  const plainChars = sequences.plain.reduce((total, sequence) => total + sequence.length, 0);

  /** Everything the model is given before it has to produce the response. */
  const responseStart = sequences.firstSegments.findIndex((segment) => segment.graded && !segment.marker);
  const promptOnly = sequences.firstSegments
    .slice(0, Math.max(1, responseStart))
    .map((segment) => segment.text)
    .join("");
  const plainPrompt = `${trainingPairs[0].instruction} `;

  const templatedResponseLogProb = responseLogProb(runs.templated.weights, first);
  const plainResponseLogProb = responseLogProb(runs.plain.weights, firstPlain);

  /** Responses usually end in a full stop, so that is where a turn boundary lands. */
  const turnBoundary = trainingPairs[0].response.slice(-1) || ".";
  const templatedNext = tinyTopTokens(runs.templated.weights, turnBoundary, 5);
  const plainNext = tinyTopTokens(runs.plain.weights, turnBoundary, 5);
  const markerProbability = (weights: Float32Array) =>
    tinyTopTokens(weights, turnBoundary, 30).find((entry) => entry.token === "<")?.probability ?? 0;

  const switches = [
    {
      id: "system",
      name: "System block",
      detail: `Prepends <system>${SYSTEM_TEXT}`,
      active: useSystem,
      toggle: () => setState({ useSystem: !useSystem }),
    },
    {
      id: "roles",
      name: "Role markers",
      detail: "Wraps each turn in <user> and <assistant>",
      active: useRoles,
      toggle: () => setState({ useRoles: !useRoles }),
    },
    {
      id: "end",
      name: "End-of-turn marker",
      detail: "Closes every turn with <end>",
      active: useEnd,
      toggle: () => setState({ useEnd: !useEnd }),
    },
  ];

  const plainUntrained = Math.abs((plainNext[0]?.probability ?? 0) - 1 / 30) < 1e-6;
  const boundaryLabel = turnBoundary === "." ? "full stop" : `“${turnBoundary}”`;

  /** Both trained tables, probed with the same serving string. Only the string changes with `Serve with`. */
  const serve = useMemo(() => {
    const options = { system: useSystem, roles: useRoles, end: useEnd };
    const instruction = trainingPairs[0].instruction;
    const reply = trainingPairs[0].response;
    const inputs = { instruction, reply, options, mode: serveWith };
    const templated = probeServing({ weights: runs.templated.weights, ...inputs });
    const plain = probeServing({ weights: runs.plain.weights, ...inputs });
    const sample = (weights: Float32Array) =>
      sampleTinyText(weights, { prompt: templated.served, length: 90, temperature: 0.6, seed: 17, allowReserved: true });
    return {
      templated,
      plain,
      served: templated.served,
      lastCharacter: templated.lastCharacter,
      templatedSample: sample(runs.templated.weights),
      plainSample: sample(runs.plain.weights),
      templatedNext: tinyTopTokens(runs.templated.weights, templated.served, 5),
      plainNext: tinyTopTokens(runs.plain.weights, templated.served, 5),
    };
  }, [runs, serveWith, trainingPairs, useEnd, useRoles, useSystem]);
  const firstReplyCharacter = trainingPairs[0].response.trim().slice(0, 1) || "t";
  const stopNote = (sample: string) => {
    const at = firstMarkerAt(sample);
    if (at < 0) return `No marker in ${sample.length} characters: a decoder watching for one would keep going.`;
    if (at === 0) return "The first character is already a marker, so a decoder watching for one would stop on an empty reply.";
    return `The first marker comes after ${at} character${at === 1 ? "" : "s"}. A decoder watching for it would stop there and keep “${sample.slice(0, at).trim()}”.`;
  };
  const servedHead = serve.served.slice(0, -1);
  const servedTail = serve.served.slice(-1);
  const percent = (value: number) => `${(value * 100).toFixed(0)}%`;

  return (
    <div className="tg-lab tg-lab--hero">
      <LabSurface label="Chat template editor" className="tg-template-panel">
        <SurfaceHeading
          kicker={`${templatedChars} characters of templated training text`}
          title="Exactly what the model is shown, and what it is graded on"
          aside={<span className="tg-badge">{pairs.length} examples</span>}
        />
        <div className="sft-editor-grid">
          <div className="tg-switch-list">
            {switches.map((entry) => (
              <button
                type="button"
                key={entry.id}
                className="tg-switch"
                aria-pressed={entry.active}
                onClick={() => {
                  entry.toggle();
                  narrate(`${entry.name} ${entry.active ? "removed" : "added"}.`);
                }}
              >
                <strong>{entry.name}</strong>
                <small>{entry.detail}</small>
                <b>{entry.active ? "in template" : "off"}</b>
              </button>
            ))}
          </div>
          <div className="sft-mask-control">
            <SegmentedControl
              label="Loss on"
              value={lossMask}
              options={[
                { value: "assistant", label: "Assistant reply only" },
                { value: "all", label: "Every character" },
              ]}
              onChange={(value) => {
                setState({ lossMask: value });
                narrate(
                  value === "all"
                    ? "Loss on every character, as in pretraining."
                    : "Loss on the assistant reply and its end marker only.",
                );
              }}
            />
            <p>
              {lossMask === "assistant"
                ? "Standard SFT: only the reply and the marker that closes it are graded. Everything before is read as context."
                : "Pretraining-style: every next character is graded, the system text, the user text and every marker included."}
            </p>
          </div>
        </div>
        <LossMaskStrip sequence={first} values={firstValues} mask={lossMask} />
        <p className="lab-note">
          The first example, one cell per character, scored by the templated model after training.
          The angle brackets are reserved: they appear in no corpus and are stripped from anything
          you type. A real tokenizer gives each marker its own token id instead of spelling it in
          letters, so a marker never shares weights with ordinary text the way <code>&lt;end&gt;</code>{" "}
          shares the rows for e, n and d here.
        </p>
      </LabSurface>

      <LabSurface label="Two models, two formats" className="tg-loss-panel a11-narrow-chart">
        <SurfaceHeading kicker="Same pairs, same budget, same loss rule" title="Templated against plain concatenation" />
        <LineChart
          label="Training loss on graded characters for the templated and plain datasets"
          xLabel="optimizer step"
          yLabel="loss on graded characters (nats/char)"
          yDomain={[0.6, UNIFORM_CROSS_ENTROPY + 0.1]}
          series={[
            {
              id: "templated",
              name: "with template",
              tone: "forward",
              points: runs.templated.history.map((point) => ({ x: point.step, y: point.loss })),
            },
            {
              id: "plain",
              name: "no template",
              tone: "loss",
              dash: "dashed",
              points: runs.plain.history.map((point) => ({ x: point.step, y: point.loss })),
            },
          ]}
          footnote={`Batch loss over the graded targets only: ${runs.templated.gradedTargets} for the templated run and ${runs.plain.gradedTargets} for the plain one. Different targets, so the two curves are not a quality ranking.`}
        />
        <div className="metric-row sft-metrics">
          <Metric label="Reply log-prob per char, templated" value={templatedResponseLogProb.toFixed(3)} tone="forward" />
          <Metric label="Reply log-prob per char, plain" value={plainResponseLogProb.toFixed(3)} tone="loss" />
          <Metric
            label={`“<” after the ${boundaryLabel}`}
            value={`${(markerProbability(runs.templated.weights) * 100).toFixed(0)}% vs ${(markerProbability(runs.plain.weights) * 100).toFixed(0)}%`}
          />
        </div>
        <p className="lab-note">
          Both scores cover the same reply characters of the first example, each conditioned on
          whatever its own format puts in front. The templated model usually scores the reply a
          little lower: in a character model the letters of <code>&lt;end&gt;</code> and{" "}
          <code>&lt;assistant&gt;</code> share rows with the reply&rsquo;s letters. What the template buys is a known place where a
          turn starts and ends.
        </p>
        <div className="tg-diff">
          <div className="tg-sample">
            <span>Templated model continues the prompt · markers shown</span>
            <p>
              {sampleTinyText(runs.templated.weights, {
                prompt: promptOnly,
                length: 70,
                temperature: 0.6,
                seed: 17,
                allowReserved: true,
              })}
            </p>
          </div>
          <div className="tg-sample">
            <span>Plain model continues the same words</span>
            <p>
              {sampleTinyText(runs.plain.weights, {
                prompt: plainPrompt,
                length: 70,
                temperature: 0.6,
                seed: 17,
              })}
            </p>
          </div>
        </div>
      </LabSurface>

      <div className="tg-column">
        <LabSurface label="Supervised dataset editor" className="tg-sft-panel">
          <SurfaceHeading kicker="Your SFT data" title="One example per line" />
          <label className="tg-editor">
            <span>instruction | response</span>
            <textarea
              aria-label="Instruction and response pairs"
              rows={6}
              value={dataset}
              maxLength={MAX_DATASET_LINES * (MAX_DATASET_LINE_LENGTH + 1)}
              onChange={(event) => setState({ dataset: sanitizeDataset(event.target.value) })}
            />
          </label>
          <p className="lab-note">
            Up to {MAX_DATASET_LINES} lines of {MAX_DATASET_LINE_LENGTH} characters each; anything beyond is
            ignored so training stays instant.
          </p>
          <RangeControl
            label="SFT epochs"
            min={1}
            max={80}
            step={1}
            value={epochs}
            format={(value) => `${value} passes`}
            onChange={(value) => {
              setState({ epochs: value });
              narrate(`${value} epochs over ${pairs.length} examples.`);
            }}
          />
          <div className="metric-row sft-metrics">
            <Metric label="Examples" value={`${pairs.length}`} />
            <Metric label="Templated chars" value={`${templatedChars}`} />
            <Metric label="Graded targets" value={`${runs.templated.gradedTargets}`} tone="forward" />
          </div>
        </LabSurface>

        <LabSurface label="What follows the end of a turn" className="tg-followup-panel">
          <SurfaceHeading
            kicker={`After the ${boundaryLabel} that ends a reply`}
            title="Does the model know a turn just ended?"
          />
          <BarList
            label="Templated model, next character after the turn boundary"
            items={templatedNext.map((entry, index) => ({
              id: `t-${entry.token}-${index}`,
              label: entry.label === "␣" ? "space" : `“${entry.label}”`,
              value: entry.probability,
              display: `${(entry.probability * 100).toFixed(1)}%`,
              tone: index === 0 ? "forward" : "muted",
              emphasis: index === 0,
            }))}
            max={1}
          />
          <BarList
            label="Plain model, next character after the same boundary"
            items={plainNext.map((entry, index) => ({
              id: `p-${entry.token}-${index}`,
              label: entry.label === "␣" ? "space" : `“${entry.label}”`,
              value: entry.probability,
              display: `${(entry.probability * 100).toFixed(1)}%`,
              tone: index === 0 ? "loss" : "muted",
            }))}
            max={1}
          />
          <p className="lab-note">
            The upper list is the templated model, the lower the plain one.{" "}
            {useEnd
              ? "With <end> in the template, the templated model learns that a marker follows a finished reply."
              : "With <end> switched off, nothing follows a finished reply in the templated data either."}{" "}
            {plainUntrained
              ? "The plain model never saw anything follow that character, so its row is untrained and every character gets 1/30. Nothing in its data says when to stop."
              : "The plain model has seen that character mid-reply, so it predicts ordinary text after it. Nothing in its data says when to stop."}
          </p>
        </LabSurface>
      </div>

      <LabSurface label="Serve the trained models" className="tg-serve-panel">
        <SurfaceHeading
          kicker="Same trained weights, a different string at serving time"
          title="What each model does when the prompt is not the one it trained on"
        />
        <SegmentedControl
          label="Serve with"
          value={serveWith}
          options={SERVE_MODES}
          onChange={(value) => {
            setState({ serveWith: value });
            narrate(`Serving with ${SERVE_MODES.find((mode) => mode.value === value)?.label ?? value}.`);
          }}
        />
        <p className="sft-served">
          <span>Sent to both models, up to the first reply character</span>
          <code>
            {servedHead}
            <mark>{servedTail === " " ? "␣" : servedTail}</mark>
          </code>
          <span>
            Each model reads only the last character, “{serve.lastCharacter === " " ? "space" : serve.lastCharacter}”,
            to choose what comes next.
          </span>
        </p>
        <div className="sft-serve-grid">
          <div className="tg-sample">
            <span>Templated model continues · markers shown</span>
            <p>{serve.templatedSample}</p>
            <small>{stopNote(serve.templatedSample)}</small>
          </div>
          <div className="tg-sample">
            <span>Plain model continues · markers shown</span>
            <p>{serve.plainSample}</p>
            <small>{stopNote(serve.plainSample)}</small>
          </div>
        </div>
        <p className="sft-list-label">Templated model · next character after the served string</p>
        <BarList
          label="Templated model, next character after the served string"
          items={serve.templatedNext.map((entry, index) => ({
            id: `st-${entry.token}-${index}`,
            label: entry.label === "␣" ? "space" : `“${entry.label}”`,
            value: entry.probability,
            display: `${(entry.probability * 100).toFixed(1)}%`,
            tone: index === 0 ? "forward" : "muted",
            emphasis: index === 0,
          }))}
          max={1}
        />
        <p className="sft-list-label">Plain model · next character after the served string</p>
        <BarList
          label="Plain model, next character after the served string"
          items={serve.plainNext.map((entry, index) => ({
            id: `sp-${entry.token}-${index}`,
            label: entry.label === "␣" ? "space" : `“${entry.label}”`,
            value: entry.probability,
            display: `${(entry.probability * 100).toFixed(1)}%`,
            tone: index === 0 ? "loss" : "muted",
          }))}
          max={1}
        />
        <div className="metric-row sft-metrics">
          <Metric
            label={`First reply character “${firstReplyCharacter}”, templated vs plain`}
            value={`${percent(serve.templated.firstReplyProbability)} vs ${percent(serve.plain.firstReplyProbability)}`}
            tone="forward"
          />
          <Metric
            label="Held-out reply, log-prob per char, templated vs plain"
            value={`${serve.templated.heldOutLogProb.toFixed(2)} vs ${serve.plain.heldOutLogProb.toFixed(2)}`}
            tone="loss"
          />
        </div>
        <p className="lab-note">
          The held-out reply is “{HELD_OUT_REPLY}”, which appears in no default pair. Because each model reads
          one character, a served string matters only through its last character. Any two serves that end in
          the same character, such as the chat template and the wrong role marker, give identical results here;
          a model with real context would read the whole marker. A model that never saw a character as
          context has an untrained row for it and spreads probability evenly.
        </p>
      </LabSurface>
    </div>
  );
}
