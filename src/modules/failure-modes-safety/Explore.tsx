import { useMemo } from "react";
import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import {
  assembleContext,
  benchResults,
  clampProbe,
  evaluate,
  FILTER_PATTERN,
  keywordFilterBlocks,
  PROBE_MAX,
  probeResult,
  scenarioById,
  SCENARIOS,
  worstCaseReply,
  type Author,
  type Controls,
  type EmailMode,
  type Outcome,
  type ProbeVerdict,
  type RemovedBy,
  type Risk,
  type Scenario,
  type Segment,
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

/** The 2×2 the bench sorts texts into: is the text hostile, and did the filter remove it? */
const QUADRANTS = [
  { verdict: "caught", hostile: true, blocked: true, right: true },
  { verdict: "missed", hostile: true, blocked: false, right: false },
  { verdict: "over-blocked", hostile: false, blocked: true, right: false },
  { verdict: "passed", hostile: false, blocked: false, right: true },
] as const;

/** Which bench quadrant a typed text lands in: a bypass is a miss, and "passes" is the passed corner. */
const PROBE_QUADRANT: Record<ProbeVerdict, (typeof QUADRANTS)[number]["verdict"]> = {
  caught: "caught",
  bypass: "missed",
  "over-blocked": "over-blocked",
  passes: "passed",
};

type GateState = "open" | "leaky" | "closed" | "person";

/** A check drawn across the path: two posts with a gap when text gets through, one solid bar when it does not. */
function PathGate({ x, y, state, live }: { x: number; y: number; state: GateState; live: boolean }) {
  const half = 20;
  const cls = `fms-gate is-${state}${live ? " is-live" : ""}`;
  if (state === "closed") return <line className={cls} x1={x} x2={x} y1={y - half} y2={y + half} />;
  if (state === "person") {
    return (
      <g className={cls}>
        <line x1={x} x2={x} y1={y - half} y2={y - 11} />
        <line x1={x} x2={x} y1={y + 11} y2={y + half} />
        <circle className="fms-gate-person" cx={x} cy={y} r="9" />
        <circle className="fms-gate-head" cx={x} cy={y - 2.6} r="2.4" />
        <path className="fms-gate-head" d={`M ${x - 4.4} ${y + 5} a 4.4 4.4 0 0 1 8.8 0 z`} />
      </g>
    );
  }
  return (
    <g className={cls}>
      <line x1={x} x2={x} y1={y - half} y2={y - 8} />
      <line x1={x} x2={x} y1={y + 8} y2={y + half} />
    </g>
  );
}

/** Where a control stopped the attack. */
function PathStop({ x, y }: { x: number; y: number }) {
  return (
    <g className="fms-stop" transform={`translate(${x} ${y})`}>
      <circle r="8.5" />
      <path d="M -3.4 -3.4 L 3.4 3.4 M 3.4 -3.4 L -3.4 3.4" />
    </g>
  );
}

/**
 * The route the attacker's text would take, drawn from the outcome the lab already computed:
 * retrieval checks, then the model's context, then the three ways a password could leave.
 */
function AttackPath({
  scenario,
  controls,
  outcome,
  segments,
}: {
  scenario: Scenario;
  controls: Controls;
  outcome: Outcome;
  segments: Segment[];
}) {
  const security = scenario.kind === "security";
  const direct = scenario.id === "direct";
  const retrievedAttack = security && !direct;
  const hostile = segments.find((segment) => segment.hostile && segment.author === "attacker");
  const hostilePassage = scenario.retrieved.find((passage) => passage.hostile);
  const stoppedAtSources = retrievedAttack && hostile?.removedBy === "allowlist";
  const stoppedAtFilter = retrievedAttack && hostile?.removedBy === "filter";
  const reaches = security && outcome.attackerReachesModel;
  const reachable = reaches && outcome.privateDataInContext;
  const filterMatches = hostilePassage ? keywordFilterBlocks(hostilePassage.text) : false;

  const sourcesGate: GateState = controls.sources === "allowlist" ? "closed" : "open";
  const filterGate: GateState = !controls.filter ? "open" : retrievedAttack && !filterMatches ? "leaky" : "closed";
  const emailGate: GateState = controls.email === "auto" ? "open" : controls.email === "approval" ? "person" : "closed";
  const imageGate: GateState = controls.images === "render" ? "open" : "closed";

  const emailToGate = reachable && scenario.channel === "email";
  const imageToGate = reachable && scenario.channel === "image";

  const Y = 114;
  const edge = (live: boolean, extra = "") => `fms-edge${live ? " is-live" : ""}${extra}`;
  const head = (live: boolean) => (live ? "url(#fms-arrow-live)" : "url(#fms-arrow-idle)");
  const exitClass = (risk: Risk) =>
    `fms-exit is-${risk === "open" ? "open" : risk === "needs a person" ? "person" : risk === "blocked" ? "blocked" : "idle"}`;

  const sourceTitle = !security ? "No attacker" : direct ? "Attacker is the user" : "Attacker’s notice";
  const sourceSub = !security ? "nothing hostile here" : direct ? "typed into the chat" : "on the visitor board";
  const sourcesText = controls.sources === "allowlist" ? "allowlist only" : "any page";
  const filterText = !controls.filter ? "off" : !retrievedAttack ? "on" : filterMatches ? "on · phrase matched" : "on · no match";

  const reading = !security
    ? "No attacker in this scenario, so no path is lit."
    : `${
        direct
          ? "The attack is the user's own message, which neither retrieval check reads, so it reaches the model."
          : stoppedAtSources
            ? "The notice is stopped at Retrieval sources: its source is not on the allowlist."
            : stoppedAtFilter
              ? "The notice passes Retrieval sources and is stopped by the keyword filter."
              : `The notice passes Retrieval sources (${sourcesText}) and the keyword filter (${filterText}) and reaches the model.`
      } ${
        reaches
          ? outcome.privateDataInContext
            ? "The password is in the context."
            : "The password is not in the context, so there is nothing to take."
          : ""
      } Reply: ${RISK_LABEL[outcome.replyLeak]}. Email: ${RISK_LABEL[outcome.emailLeak]}. Image: ${RISK_LABEL[outcome.imageLeak]}.`;

  return (
    <svg className="fms-path" viewBox="0 0 664 226" role="img" aria-label={`Attack path. ${reading}`}>
      <defs>
        <marker id="fms-arrow-live" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" className="fms-arrowhead is-live" />
        </marker>
        <marker id="fms-arrow-idle" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" className="fms-arrowhead" />
        </marker>
        <marker id="fms-arrow-person" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" className="fms-arrowhead is-person" />
        </marker>
      </defs>

      {/* retrieval route: source → sources check → keyword filter → model */}
      <path className={edge(retrievedAttack)} d={`M 130 ${Y} L 186 ${Y}`} />
      <path className={edge(retrievedAttack && !stoppedAtSources)} d={`M 186 ${Y} L 258 ${Y}`} />
      <path
        className={edge(retrievedAttack && reaches)}
        d={`M 258 ${Y} L 309 ${Y}`}
        markerEnd={head(retrievedAttack && reaches)}
      />
      {direct && (
        <>
          <path className={edge(true)} d="M 68 88 C 68 26, 387 14, 387 63" markerEnd="url(#fms-arrow-live)" />
          <text className="fms-path-note" x="228" y="17" textAnchor="middle">
            the user’s message skips both checks
          </text>
        </>
      )}

      <g className={`fms-node is-source${security ? " is-hostile" : ""}`}>
        <rect x="6" y="88" width="124" height="52" rx="9" />
        <text x="68" y="110" textAnchor="middle">
          {sourceTitle}
        </text>
        <text className="fms-node-sub" x="68" y="125" textAnchor="middle">
          {sourceSub}
        </text>
      </g>

      <PathGate x={186} y={Y} state={sourcesGate} live={retrievedAttack} />
      <text className="fms-gate-label" x="186" y="62" textAnchor="middle">
        Retrieval sources
      </text>
      <text className="fms-gate-state" x="186" y="76" textAnchor="middle">
        {sourcesText}
      </text>
      {stoppedAtSources && <PathStop x={172} y={Y} />}

      <PathGate x={258} y={Y} state={filterGate} live={retrievedAttack && !stoppedAtSources} />
      <text className="fms-gate-label" x="258" y="158" textAnchor="middle">
        Keyword filter
      </text>
      <text className="fms-gate-state" x="258" y="172" textAnchor="middle">
        {filterText}
      </text>
      {stoppedAtFilter && <PathStop x={244} y={Y} />}

      {/* the three ways out */}
      <path
        className={edge(outcome.replyLeak === "open")}
        d="M 462 90 C 500 90, 492 33, 533 33"
        markerEnd={head(outcome.replyLeak === "open")}
      />
      <text className="fms-path-note" x="470" y="50" textAnchor="middle">
        no check
      </text>
      <path className={edge(emailToGate)} d={`M 462 ${Y} L 510 ${Y}`} />
      <path
        className={edge(outcome.emailLeak === "open", outcome.emailLeak === "needs a person" ? " is-person" : "")}
        d={`M 510 ${Y} L 533 ${Y}`}
        markerEnd={
          outcome.emailLeak === "needs a person" ? "url(#fms-arrow-person)" : head(outcome.emailLeak === "open")
        }
      />
      <path className={edge(imageToGate)} d="M 462 138 C 488 138, 478 195, 510 195" />
      <path
        className={edge(outcome.imageLeak === "open")}
        d="M 510 195 L 533 195"
        markerEnd={head(outcome.imageLeak === "open")}
      />
      <PathGate x={510} y={Y} state={emailGate} live={emailToGate} />
      <PathGate x={510} y={195} state={imageGate} live={imageToGate} />
      {emailToGate && controls.email === "none" && <PathStop x={496} y={Y} />}
      {imageToGate && controls.images === "off" && <PathStop x={496} y={195} />}

      <g className={`fms-node is-model${reaches ? " is-reached" : ""}`}>
        <rect x="312" y="66" width="150" height="96" rx="10" />
        <text x="387" y="86" textAnchor="middle">
          Model’s context
        </text>
        <text className="fms-node-sub" x="387" y="100" textAnchor="middle">
          assumed to obey what it reads
        </text>
        <g className={`fms-secret${outcome.privateDataInContext ? " is-present" : ""}`}>
          <rect x="324" y="112" width="126" height="40" rx="6" />
          <text className="fms-node-sub" x="387" y="128" textAnchor="middle">
            {outcome.privateDataInContext ? "private data" : "no private data"}
          </text>
          <text className="fms-secret-value" x="387" y="143" textAnchor="middle">
            {outcome.privateDataInContext ? "password: bluebell" : "password kept out"}
          </text>
        </g>
      </g>

      <g className={exitClass(outcome.replyLeak)}>
        <rect x="536" y="10" width="124" height="46" rx="9" />
        <text x="598" y="29" textAnchor="middle">
          Reply on screen
        </text>
        <text className="fms-node-sub" x="598" y="44" textAnchor="middle">
          {RISK_LABEL[outcome.replyLeak]}
        </text>
      </g>
      <g className={exitClass(outcome.emailLeak)}>
        <rect x="536" y="91" width="124" height="46" rx="9" />
        <text x="598" y="110" textAnchor="middle">
          Email sent out
        </text>
        <text className="fms-node-sub" x="598" y="125" textAnchor="middle">
          {RISK_LABEL[outcome.emailLeak]}
        </text>
      </g>
      <g className={exitClass(outcome.imageLeak)}>
        <rect x="536" y="172" width="124" height="46" rx="9" />
        <text x="598" y="191" textAnchor="middle">
          Image request
        </text>
        <text className="fms-node-sub" x="598" y="206" textAnchor="middle">
          {RISK_LABEL[outcome.imageLeak]}
        </text>
      </g>
    </svg>
  );
}

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
        <AttackPath scenario={scenario} controls={controls} outcome={outcome} segments={segments} />
        <p className="fms-path-key" aria-hidden="true">
          <span className="is-live">
            <i /> the attacker’s text, then the password
          </span>
          <span className="is-idle">
            <i /> nothing hostile travels here
          </span>
          <span className="is-stop">
            <i>✕</i> stopped by a control
          </span>
        </p>
        <div className="fms-toggles">
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
        <div className="fms-bench-fixed">
          <table
            className="fms-matrix"
            aria-label="The five fixed texts, sorted by whether they are hostile and whether the filter removed them"
          >
            <thead>
              <tr>
                <td />
                <th scope="col">Filter removes it</th>
                <th scope="col">Filter keeps it</th>
              </tr>
            </thead>
            <tbody>
              {[true, false].map((hostile) => (
                <tr key={String(hostile)}>
                  <th scope="row">
                    <span>{hostile ? "Hostile text" : "Benign text"}</span>
                  </th>
                  {QUADRANTS.filter((quadrant) => quadrant.hostile === hostile).map((quadrant) => {
                    const rows = bench.rows.filter((row) => row.verdict === quadrant.verdict);
                    const yours = probed && PROBE_QUADRANT[probed.verdict] === quadrant.verdict;
                    return (
                      <td key={quadrant.verdict} className={quadrant.right ? "is-right" : "is-wrong"}>
                        <header>
                          <em>{quadrant.verdict}</em>
                          <span>
                            {quadrant.right ? "✓" : "✗"} {rows.length} of {hostile ? bench.attacks : bench.benign}
                          </span>
                        </header>
                        <ul className="fms-bench">
                          {rows.map((row) => (
                            <li key={row.id} className={`is-${row.verdict.replace(" ", "-")}`}>
                              <strong>{row.label}</strong>
                              <p>{row.text}</p>
                            </li>
                          ))}
                          {yours && probed && (
                            <li className="is-yours">
                              <strong>Your text</strong>
                              <small>lands here: {PROBE_LABEL[probed.verdict]}</small>
                            </li>
                          )}
                        </ul>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric label="Attacks caught" value={`${bench.caught} of ${bench.attacks}`} tone={bench.caught < bench.attacks ? "loss" : "forward"} />
          <Metric label="Benign blocked" value={`${bench.overBlocked} of ${bench.benign}`} tone={bench.overBlocked ? "loss" : "forward"} />
        </div>
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
      </LabSurface>
    </div>
  );
}
