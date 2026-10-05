import { useMemo, type ReactNode } from "react";
import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import { presetById, PRESETS, REPAIR_IDS, runConversation, TOOLS, type StageStatus } from "./host";
import { cappedText, DRAFT_MAX } from "./state";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

const WELL_FORMED = ["valid", "note", "email"];
const MALFORMED = ["missing", "enum", "type", "extra", "broken", "unknown"];
const COUNT_WORDS: Record<number, string> = { 4: "four", 6: "six" };

const STATUS_MARK: Record<StageStatus, string> = { pass: "✓ pass", fail: "✕ fail", skipped: "– skipped" };

function Stage({
  number,
  title,
  status,
  children,
}: {
  number: number;
  title: string;
  status?: StageStatus;
  children?: ReactNode;
}) {
  return (
    <div className={`tool-stage${status ? ` is-${status}` : ""}`}>
      <header>
        <span className="tool-stage-no">{number}</span>
        <strong>{title}</strong>
        {status && <em>{STATUS_MARK[status]}</em>}
      </header>
      {children && <div className="tool-stage-body">{children}</div>}
    </div>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const presetId = PRESETS.some((item) => item.id === asString(state, "preset", "valid"))
    ? asString(state, "preset", "valid")
    : "valid";
  const preset = presetById(presetId);
  const isRepair = preset.retry !== undefined;
  const draft = cappedText(state.draft, preset.output);
  const retry = isRepair ? cappedText(state.retry, preset.retry ?? "") : "";
  const approved = asString(state, "approve", "no") === "yes";
  const conversation = useMemo(() => runConversation(draft, retry, approved), [approved, draft, retry]);
  const turn = conversation.second && asString(state, "turn", "1") === "2" ? 2 : 1;
  const run = turn === 2 && conversation.second ? conversation.second : conversation.first;
  const finalRun = conversation.second ?? conversation.first;
  const shownCall = turn === 2 ? retry : draft;
  const edited = draft !== preset.output || (isRepair && retry !== preset.retry);
  const { stages } = run;
  const toolName = run.parsed?.name ?? "—";
  const shownTool = run.tool ?? TOOLS.find((item) => item.name === run.parsed?.name) ?? null;
  const executed = stages.execute === "pass";
  const failedAt = (Object.entries(stages).find(([, value]) => value === "fail")?.[0] ?? null) as
    | keyof typeof stages
    | null;
  const failLabel: Record<keyof typeof stages, string> = {
    parse: "parse",
    lookup: "tool lookup",
    validate: "validation",
    policy: "policy",
    execute: "execution",
  };

  const choose = (id: string) =>
    setState({ preset: id, draft: presetById(id).output, retry: presetById(id).retry ?? "", turn: "1" });

  const presetButtons = (ids: string[]) =>
    ids.map((id) => {
      const item = presetById(id);
      return (
        <button key={id} type="button" aria-pressed={presetId === id && !edited} onClick={() => choose(id)}>
          {item.label}
        </button>
      );
    });

  return (
    <div className="tg-lab tg-lab--hero tool-lab">
      <LabSurface label="Model output" className="tool-output-card">
        <SurfaceHeading
          kicker="The model's whole contribution is this string"
          title="A tool call is text the model writes"
          aside={<span className="tg-badge">authored outputs</span>}
        />
        <p className="tool-user">
          <span>User</span> {preset.user}
        </p>
        <div className="tool-preset-groups">
          <div role="group" aria-label="Well-formed model outputs" className="tool-presets">
            <span>well-formed</span>
            {presetButtons(WELL_FORMED)}
          </div>
          <div role="group" aria-label="Malformed model outputs" className="tool-presets">
            <span>malformed</span>
            {presetButtons(MALFORMED)}
          </div>
          <div role="group" aria-label="Error, then retry model outputs" className="tool-presets tool-repair-presets">
            <span>error, then retry</span>
            {presetButtons(REPAIR_IDS as string[])}
          </div>
        </div>
        <label className="tg-editor tool-draft">
          <span>
            {isRepair ? "Model output, turn 1" : "Model output"} (raw text{edited ? ", edited by you" : ""})
          </span>
          <textarea
            aria-label="Model output"
            rows={3}
            maxLength={DRAFT_MAX}
            spellCheck={false}
            value={draft}
            onChange={(event) => setState({ draft: event.target.value.slice(0, DRAFT_MAX) })}
          />
        </label>
        {isRepair && (
          <label className="tg-editor tool-retry">
            <span>Model output, turn 2 (authored retry; sent only if turn 1 fails)</span>
            <textarea
              aria-label="Model output, turn 2"
              rows={3}
              maxLength={DRAFT_MAX}
              spellCheck={false}
              value={retry}
              onChange={(event) => setState({ retry: event.target.value.slice(0, DRAFT_MAX) })}
            />
          </label>
        )}
        <p className="lab-note">
          The model cannot run anything. It emits characters; this one happens to be JSON naming a tool. Edit the text
          and every host check below reruns on exactly what you typed.
          {isRepair
            ? " The retry is written as if the model had read the first error. Nothing here generates it, and the host checks it from scratch."
            : ""}
        </p>
      </LabSurface>

      <LabSurface label="Host pipeline" className="tool-wall-card">
        <SurfaceHeading
          kicker={`${conversation.second ? `Turn ${turn} of 2 · ` : ""}${
            executed ? `${toolName} ran on the host` : failedAt ? `stopped at ${failLabel[failedAt]}` : "waiting"
          }`}
          title="The host parses, checks, decides, and runs"
          aside={
            <span className={`tg-badge${executed ? "" : " is-warning"}`}>{executed ? "executed" : "not executed"}</span>
          }
        />
        <div className="tool-wall" role="group" aria-label="Model side and host side of one tool call">
          <div className="tool-lane-head is-model">Model — writes text</div>
          <div className="tool-lane-head is-wall" aria-hidden="true" />
          <div className="tool-lane-head is-host">Host program — runs code</div>

          <Stage number={2} title="Reads the tool list in its prompt" />
          <div className="tool-arrow is-left" aria-hidden="true">←</div>
          <Stage number={1} title={`Sends ${TOOLS.length} tool schemas with the request`}>
            {TOOLS.map((tool) => tool.name).join(" · ")}
          </Stage>

          <Stage number={3} title="Writes a call as text">
            <code className="tool-snippet">{shownCall || "(empty)"}</code>
          </Stage>
          <div className="tool-arrow" aria-hidden="true">→</div>
          <Stage number={4} title="Parse the text as JSON" status={stages.parse}>
            {run.parseError ? run.parseError : run.parsed ? `name = ${run.parsed.name}` : null}
          </Stage>

          <div className="tool-lane-gap" />
          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={5} title="Look up the tool name" status={stages.lookup}>
            {stages.lookup === "fail" ? `no tool named ${toolName}` : run.tool ? `${run.tool.name}: ${run.tool.effect}` : null}
          </Stage>

          <div className="tool-lane-gap" />
          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={6} title="Validate arguments against the schema" status={stages.validate}>
            {stages.validate === "fail"
              ? `${run.errors.length} error${run.errors.length === 1 ? "" : "s"}: ${run.errors.map((item) => item.path).join(", ")}`
              : stages.validate === "pass"
                ? "types, enums, required and extra fields all check out"
                : null}
          </Stage>

          <div className="tool-lane-gap" />
          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={7} title="Apply host policy" status={stages.policy}>
            {run.tool
              ? `${stages.policy === "skipped" ? "not reached · " : ""}${
                  run.tool.policy === "approval"
                    ? approved
                      ? "send_email needs a human approval — given"
                      : "send_email needs a human approval — none given"
                    : "allowed without approval"
                }`
              : null}
          </Stage>

          <div className="tool-lane-gap" />
          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={8} title="Execute" status={stages.execute}>
            {run.sideEffect ?? (executed ? "read-only lookup; nothing changed" : null)}
          </Stage>

          <Stage number={10} title="Next turn reads the tool message">
            and nothing else about what the host did
          </Stage>
          <div className="tool-arrow is-left" aria-hidden="true">←</div>
          <Stage number={9} title="Append the result as a tool message">
            <code className="tool-snippet">{run.result}</code>
          </Stage>
        </div>
        <div className="tool-approve-control">
          <SegmentedControl
            label="Approve send_email"
            value={approved ? "yes" : "no"}
            options={[
              { value: "no", label: "no approval" },
              { value: "yes", label: "human approves" },
            ]}
            onChange={(value) => setState({ approve: value })}
          />
        </div>
        {conversation.second && (
          <div className="tool-turn-control">
            <SegmentedControl
              label="Turn"
              value={String(turn)}
              options={[
                { value: "1", label: "1 · first call" },
                { value: "2", label: "2 · retry" },
              ]}
              onChange={(value) => setState({ turn: value })}
            />
          </div>
        )}
      </LabSurface>

      <LabSurface label="Schema check" className="tool-schema-card">
        <SurfaceHeading
          kicker={shownTool ? `${shownTool.name} · policy ${shownTool.policy === "approval" ? "needs approval" : "auto"}` : "No matching tool"}
          title="The contract the arguments must meet"
        />
        {shownTool ? (
          <pre className="tool-schema" aria-label={`${shownTool.name} JSON schema`}>
            {JSON.stringify({ name: shownTool.name, description: shownTool.description, parameters: shownTool.parameters }, null, 2)}
          </pre>
        ) : (
          <p className="lab-note">
            {run.parseError
              ? `The host could not read a tool name: ${run.parseError}.`
              : `The host advertises only ${TOOLS.map((tool) => tool.name).join(", ")}. “${toolName}” is not one of them.`}
          </p>
        )}
        <ul className="tool-errors">
          {run.errors.length === 0 && (
            <li className="is-empty">
              {stages.validate === "pass"
                ? "No validation errors."
                : stages.validate === "skipped"
                  ? "Validation did not run because an earlier check failed."
                  : ""}
            </li>
          )}
          {run.errors.map((error) => (
            <li key={`${error.path}-${error.message}`}>
              <code>{error.path}</code> {error.message}
            </li>
          ))}
        </ul>
      </LabSurface>

      <LabSurface label="Transcript" className="tool-transcript-card">
        <SurfaceHeading
          kicker="What the next model call receives"
          title={conversation.second ? "Messages, in order, across both turns" : "Messages, in order"}
        />
        <ol className="tool-transcript">
          <li>
            <b>system</b>
            <span>Tools available: {TOOLS.map((tool) => tool.name).join(", ")} (schemas attached)</span>
          </li>
          <li>
            <b>user</b>
            <span>{preset.user}</span>
          </li>
          <li>
            <b>assistant</b>
            <code>{draft || "(empty)"}</code>
          </li>
          <li className={conversation.first.stages.execute === "pass" ? "" : "is-error"}>
            <b>tool</b>
            <code>{conversation.first.result}</code>
          </li>
          {conversation.second && (
            <>
              <li>
                <b>assistant</b>
                <code>{retry || "(empty)"}</code>
              </li>
              <li className={conversation.second.stages.execute === "pass" ? "" : "is-error"}>
                <b>tool</b>
                <code>{conversation.second.result}</code>
              </li>
            </>
          )}
          <li className="is-next">
            <b>assistant</b>
            <span>next call: generated from the {COUNT_WORDS[conversation.second ? 6 : 4]} messages above</span>
          </li>
        </ol>
        <div className="metric-row">
          <Metric
            label="Executed"
            value={finalRun.stages.execute === "pass" ? "yes" : "no"}
            tone={finalRun.stages.execute === "pass" ? "forward" : "loss"}
          />
          <Metric
            label="Grammar blocks it"
            value={
              conversation.first.grammarWouldPrevent === null
                ? "n/a"
                : conversation.first.grammarWouldPrevent
                  ? "yes"
                  : "no"
            }
            tone={conversation.first.grammarWouldPrevent === false ? "loss" : undefined}
          />
          <Metric label="Calls made" value={conversation.second ? "2" : "1"} />
        </div>
        <p className="lab-note">
          “Grammar blocks it” asks whether constrained decoding with these schemas could have stopped the first call
          from being generated. Shape errors, yes. A valid call to a dangerous tool, no — only host policy stops that.
        </p>
      </LabSurface>
    </div>
  );
}
