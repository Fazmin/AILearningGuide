import { useMemo } from "react";
import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import {
  assembleContext,
  benchResults,
  clampProbe,
  evaluate,
  FILTER_PATTERN,
  PROBE_MAX,
  probeResult,
  scenarioById,
  SCENARIOS,
  worstCaseReply,
  type Author,
  type Controls,
  type EmailMode,
  type ProbeVerdict,
  type RemovedBy,
  type Risk,
} from "./safety";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

const AUTHOR_LABEL: Record<Author, string> = {
  developer: "written by the developer",
  host: "written by the host program",
  user: "typed by the user",
  document: "written by a document author",
  attacker: "written by an attacker",
};

const RISK_LABEL: Record<Risk, string> = {
  open: "possible",
  blocked: "blocked",
  "needs a person": "only if a person approves",
  "not applicable": "no attacker",
  "not used": "not used by this attack",
};

const REMOVED_LABEL: Record<RemovedBy, string> = {
  filter: "removed by the keyword filter",
  allowlist: "left out: its source is not on the allowlist",
};

const PROBE_LABEL: Record<ProbeVerdict, string> = {
  caught: "caught",
  bypass: "bypass",
  "over-blocked": "over-blocked",
  passes: "passes",
};

const PROBE_NOTE: Record<ProbeVerdict, string> = {
  caught: "The filter removed text the simulated assistant would have obeyed.",
  bypass: "The filter kept text the simulated assistant would obey: the pattern missed it.",
  "over-blocked": "The filter removed text the simulated assistant would not have acted on.",
  passes: "The filter kept text that asks for nothing the lab protects.",
};

/** CSS class for a probe verdict, reusing the bench row styles: a bypass is a miss. */
const PROBE_CLASS: Record<ProbeVerdict, string> = {
  caught: "is-caught",
  bypass: "is-missed",
  "over-blocked": "is-over-blocked",
  passes: "is-passed",
};

export default function Explore({ state, setState }: ModuleContext) {
  const scenario = scenarioById(asString(state, "scenario", "indirect"));
  const view = asString(state, "view", "authors") === "model" ? "model" : "authors";
  const filter = asString(state, "filter", "off") === "on";
  const secretInPrompt = asString(state, "secret", "in") !== "out";
  const emailRaw = asString(state, "email", "auto");
  const email: EmailMode = emailRaw === "none" || emailRaw === "approval" ? emailRaw : "auto";
  const sources = asString(state, "sources", "any") === "allowlist" ? "allowlist" : "any";
  const images = asString(state, "images", "render") === "off" ? "off" : "render";
  const probe = clampProbe(state.probe);
  const controls: Controls = { filter, secretInPrompt, email, sources, images };

  const segments = assembleContext(scenario, controls);
  const outcome = evaluate(scenario, controls);
  const reply = worstCaseReply(scenario, controls);
  const probed = probeResult(probe);
  const bench = useMemo(() => benchResults(), []);
  const channelName = scenario.channel === "image" ? "a rendered image" : "email";
  const visible = segments.filter((segment) => !segment.filtered);
  const modelView = visible.map((segment) => `[${segment.role}] ${segment.text}`).join("\n");

  return (
    <div className="tg-lab tg-lab--hero fms-lab">
      <LabSurface label="One context window" className="fms-context-card">
        <SurfaceHeading
          kicker={`${scenario.failure} · ${scenario.kind === "security" ? "a security failure" : "an accuracy failure"}`}
          title="Instructions and data arrive in the same stream of tokens"
        />
        <div className="attack-control fms-scenarios" role="group" aria-label="Scenario">
          <span>Scenario</span>
          {SCENARIOS.map((item) => (
            <button key={item.id} type="button" aria-pressed={item.id === scenario.id} onClick={() => setState({ scenario: item.id })}>
              {item.label}
            </button>
          ))}
        </div>
        <div className="fms-view-control">
          <SegmentedControl
            label="View"
            value={view}
            options={[
              { value: "authors", label: "who wrote each part" },
              { value: "model", label: "what the model receives" },
            ]}
            onChange={(value) => setState({ view: value })}
          />
        </div>
        {view === "authors" ? (
          <ol className="fms-stream" aria-label="Context window, labelled by author">
            {segments.map((segment) => (
              <li
                key={segment.role}
                className={`is-${segment.author}${segment.filtered ? " is-filtered" : ""}${segment.hostile ? " is-hostile" : ""}`}
              >
                <header>
                  <b>{segment.role}</b>
                  <span>
                    {AUTHOR_LABEL[segment.author]}
                    {segment.source ? ` · from the ${segment.source}` : ""}
                    {segment.hostile ? " · carries instructions" : ""}
                    {segment.removedBy ? ` · ${REMOVED_LABEL[segment.removedBy]}` : ""}
                  </span>
                </header>
                <p>{segment.text}</p>
              </li>
            ))}
          </ol>
        ) : (
          <pre className="fms-flat" aria-label="Context window as the model receives it">
            {modelView}
          </pre>
        )}
        <p className="lab-note">
          {view === "authors"
            ? `${scenario.note} The colours and labels are host knowledge; they are not in what the model reads.${scenario.pointer ? ` ${scenario.pointer}` : ""}`
            : "This is the whole input: one sequence of tokens. Role markers are more tokens. Nothing in the architecture marks a retrieved sentence as less authoritative than the system prompt."}
        </p>
      </LabSurface>

      <LabSurface label="Host controls" className="fms-controls-card">
        <SurfaceHeading kicker="Assume the model obeys any instruction it reads" title="What host design can guarantee" />
        <div className="fms-filter-toggle">
          <SegmentedControl
            label="Keyword filter on retrieved text"
            value={filter ? "on" : "off"}
            options={[
              { value: "off", label: "off" },
              { value: "on", label: "on" },
            ]}
            onChange={(value) => setState({ filter: value })}
          />
        </div>
        <div className="fms-sources-toggle">
          <SegmentedControl
            label="Retrieval sources"
            value={sources}
            options={[
              { value: "any", label: "any page" },
              { value: "allowlist", label: "allowlist only" },
            ]}
            onChange={(value) => setState({ sources: value })}
          />
        </div>
        <div className="fms-secret-toggle">
          <SegmentedControl
            label="Password in the system prompt"
            value={secretInPrompt ? "in" : "out"}
            options={[
              { value: "in", label: "in the prompt" },
              { value: "out", label: "kept out" },
            ]}
            onChange={(value) => setState({ secret: value })}
          />
        </div>
        <div className="fms-email-toggle">
          <SegmentedControl
            label="send_email tool"
            value={email}
            options={[
              { value: "auto", label: "runs automatically" },
              { value: "approval", label: "needs approval" },
              { value: "none", label: "not offered" },
            ]}
            onChange={(value) => setState({ email: value })}
          />
        </div>
        <div className="fms-images-toggle">
          <SegmentedControl
            label="Images in replies"
            value={images}
            options={[
              { value: "render", label: "rendered automatically" },
              { value: "off", label: "not fetched" },
            ]}
            onChange={(value) => setState({ images: value })}
          />
        </div>
        <ul className="fms-trifecta" aria-label="Exfiltration conditions">
          <li className={outcome.privateDataInContext ? "is-on" : ""}>
            <b>{outcome.privateDataInContext ? "yes" : "no"}</b> private data in context
          </li>
          <li className={outcome.untrustedInContext ? "is-on" : ""}>
            <b>{outcome.untrustedInContext ? "yes" : "no"}</b> attacker text reaches the model
          </li>
          <li className={outcome.outboundChannel ? "is-on" : ""}>
            <b>{outcome.outboundChannel ? "yes" : "no"}</b> unattended way to send data out (email, or an image the chat window loads)
          </li>
        </ul>
        <div className="metric-row">
          <Metric
            label="Password in the reply"
            value={RISK_LABEL[outcome.replyLeak]}
            tone={outcome.replyLeak === "open" ? "loss" : "forward"}
          />
          <Metric
            label="Password sent by email"
            value={RISK_LABEL[outcome.emailLeak]}
            tone={outcome.emailLeak === "open" ? "loss" : "forward"}
          />
          <Metric
            label="Password in a rendered image"
            value={RISK_LABEL[outcome.imageLeak]}
            tone={outcome.imageLeak === "open" ? "loss" : "forward"}
          />
        </div>
        {reply && (
          <>
            <pre className="fms-flat" aria-label="Worst-case reply">
              {reply.text}
            </pre>
            <p className="lab-note">
              {reply.request
                ? `The chat window renders the image by itself, so the browser sends ${reply.request} and the password reaches maps.drop.example. No tool was called and no person was asked.`
                : "The host shows this as plain text and does not fetch the image, so no request is made. The password is still in the reply text, because the model obeyed."}
            </p>
          </>
        )}
        <p className="lab-note">
          {scenario.kind === "accuracy"
            ? `No attacker here, so these controls change nothing. The risk is a wrong answer, reduced by retrieval checks, abstention and training — not by filters.${scenario.pointer ? ` ${scenario.pointer}` : ""}`
            : outcome.trifecta
              ? `All three conditions hold, so exfiltration through ${channelName} is open. Removing any one of them closes it, whatever the attack says.`
              : `At least one condition is missing, so an unattended leak through ${channelName} is impossible here — regardless of the model's behaviour.${
                  scenario.id === "direct" && sources === "allowlist"
                    ? " An allowlist limits retrieved pages only; here the attacker is the user."
                    : ""
                }`}
        </p>
      </LabSurface>

      <LabSurface label="Filter test bench" className="fms-bench-card">
        <SurfaceHeading kicker="Five fixed texts, then yours, through the same regular expression" title="A keyword filter, tested for real" />
        <code className="fms-pattern">{`/${FILTER_PATTERN.source}/i`}</code>
        <ul className="fms-bench fms-bench-fixed">
          {bench.rows.map((row) => (
            <li key={row.id} className={`is-${row.verdict.replace(" ", "-")}`}>
              <header>
                <strong>{row.label}</strong>
                <em>{row.verdict}</em>
              </header>
              <p>{row.text}</p>
              <small>
                {row.hostile ? "hostile" : "benign"} · {row.blocked ? "matches the pattern, removed" : "no match, kept"}
              </small>
            </li>
          ))}
        </ul>
        <div className="fms-probe">
          <label className="prompt-input">
            <span>Your text: type an attack, or a harmless note</span>
            <input
              aria-label="Your text"
              value={probe}
              maxLength={PROBE_MAX}
              spellCheck={false}
              onChange={(event) => setState({ probe: clampProbe(event.target.value) })}
            />
          </label>
          {probed ? (
            <ul className="fms-bench" aria-label="Your text through the filter">
              <li className={PROBE_CLASS[probed.verdict]}>
                <header>
                  <strong>Your text</strong>
                  <em>{PROBE_LABEL[probed.verdict]}</em>
                </header>
                <p>{probed.text}</p>
                <small>
                  filter: {probed.blocked ? "matches the pattern, removed" : "no match, kept"} · simulated assistant:{" "}
                  {probed.reading.acts ? "would act on it" : "would not act on it"}, because {probed.reading.because}.{" "}
                  {PROBE_NOTE[probed.verdict]}
                </small>
              </li>
            </ul>
          ) : (
            <p className="lab-note">Type a text to see the filter’s verdict and whether the simulated assistant would act.</p>
          )}
        </div>
        <div className="metric-row">
          <Metric label="Attacks caught" value={`${bench.caught} of ${bench.attacks}`} tone={bench.caught < bench.attacks ? "loss" : "forward"} />
          <Metric label="Benign blocked" value={`${bench.overBlocked} of ${bench.benign}`} tone={bench.overBlocked ? "loss" : "forward"} />
        </div>
      </LabSurface>
    </div>
  );
}
