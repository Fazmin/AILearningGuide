import { useMemo, type ReactNode } from "react";
import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import {
  presetById,
  PRESETS,
  REPAIR_IDS,
  runConversation,
  TOOLS,
  type HostRun,
  type Schema,
  type StageStatus,
  type ToolSpec,
  type ValidationError,
} from "./host";
import { cappedText, DRAFT_MAX } from "./state";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

const WELL_FORMED = ["valid", "note", "email"];
const MALFORMED = ["missing", "enum", "type", "extra", "broken", "unknown"];
const COUNT_WORDS: Record<number, string> = { 4: "four", 6: "six" };

const STATUS_MARK: Record<StageStatus, string> = { pass: "✓ pass", fail: "✕ fail", skipped: "– skipped" };

const HOST_STAGE_NO = { parse: 4, lookup: 5, validate: 6, policy: 7, execute: 8 } as const;

function Stage({
  number,
  title,
  status,
  note,
  noteTone,
  children,
}: {
  number: number;
  title: string;
  status?: StageStatus;
  /** A short tag at the right of the header, for a stage that carries no pass or fail mark. */
  note?: string;
  noteTone?: "error" | "ok";
  children?: ReactNode;
}) {
  return (
    <div className={`tool-stage${status ? ` is-gate is-${status}` : ""}${noteTone ? ` is-${noteTone}` : ""}`}>
      <header>
        <span className="tool-stage-no">{number}</span>
        <strong>{title}</strong>
        {status && <em>{STATUS_MARK[status]}</em>}
        {!status && note && <em>{note}</em>}
      </header>
      {children && <div className="tool-stage-body">{children}</div>}
    </div>
  );
}

/** One crossing of the wall: a slot with an arrow and the name of the text that goes through it. */
function Crossing({ direction, carries }: { direction: "to-model" | "to-host"; carries: string }) {
  return (
    <div className={`tool-arrow${direction === "to-model" ? " is-left" : ""}`} aria-hidden="true">
      <b>{direction === "to-model" ? "←" : "→"}</b>
      <span>{carries}</span>
    </div>
  );
}

/** Which line of the printed schema a validation error breaks. */
const ruleOf = (error: ValidationError) => {
  if (error.message.startsWith("required")) return "required";
  if (error.message.startsWith("property not allowed")) return "additional";
  if (error.message.startsWith("expected")) return `type:${error.path}`;
  if (error.message.startsWith("must be one of")) return `enum:${error.path}`;
  if (error.message.startsWith("longer than")) return `maxLength:${error.path}`;
  if (error.message.startsWith("does not match")) return `pattern:${error.path}`;
  return "";
};

interface SchemaLine {
  text: string;
  rule?: string;
}

/** The schema as sent to the model, one rule per line so a broken rule can be marked on its own line. */
function schemaLines(tool: ToolSpec): SchemaLine[] {
  const lines: SchemaLine[] = [];
  const pad = (depth: number) => "  ".repeat(depth);
  const inline = (value: unknown) =>
    Array.isArray(value) ? `[${value.map((item) => JSON.stringify(item)).join(", ")}]` : JSON.stringify(value);
  lines.push({ text: "{" });
  lines.push({ text: `${pad(1)}"name": ${JSON.stringify(tool.name)},` });
  lines.push({ text: `${pad(1)}"description": ${JSON.stringify(tool.description)},` });
  lines.push({ text: `${pad(1)}"parameters": {` });
  lines.push({ text: `${pad(2)}"type": "object",` });
  lines.push({ text: `${pad(2)}"properties": {` });
  const properties = Object.entries(tool.parameters.properties ?? {});
  properties.forEach(([key, schema], index) => {
    lines.push({ text: `${pad(3)}${JSON.stringify(key)}: {` });
    const rules = Object.entries(schema).filter(([, value]) => value !== undefined);
    rules.forEach(([rule, value], position) => {
      lines.push({
        text: `${pad(4)}${JSON.stringify(rule)}: ${inline(value)}${position < rules.length - 1 ? "," : ""}`,
        rule: `${rule}:/${key}`,
      });
    });
    lines.push({ text: `${pad(3)}}${index < properties.length - 1 ? "," : ""}` });
  });
  lines.push({ text: `${pad(2)}},` });
  lines.push({ text: `${pad(2)}"required": ${inline(tool.parameters.required ?? [])},`, rule: "required" });
  lines.push({
    text: `${pad(2)}"additionalProperties": ${JSON.stringify(tool.parameters.additionalProperties ?? true)}`,
    rule: "additional",
  });
  lines.push({ text: `${pad(1)}}` });
  lines.push({ text: "}" });
  return lines;
}

const ruleWords = (schema: Schema) =>
  [
    schema.type,
    schema.maxLength !== undefined ? `at most ${schema.maxLength} characters` : "",
    schema.pattern ? "must match the pattern" : "",
  ]
    .filter(Boolean)
    .join(" · ");

/** The call's arguments laid against the schema: one row per field the schema names or the call sent. */
function ArgumentCheck({ tool, run }: { tool: ToolSpec; run: HostRun }) {
  const sent = run.parsed?.arguments ?? {};
  const properties = tool.parameters.properties ?? {};
  const required = tool.parameters.required ?? [];
  const errorsAt = (key: string) => run.errors.filter((error) => error.path === `/${key}`);
  const known = Object.keys(properties);
  const extra = Object.keys(sent).filter((key) => !(key in properties));
  const placed = new Set([...known, ...extra].map((key) => `/${key}`));
  const loose = run.errors.filter((error) => !placed.has(error.path));
  const show = (value: unknown) => JSON.stringify(value);

  return (
    <ul className="tool-errors" aria-label="Each argument checked against the schema">
      {known.map((key) => {
        const schema = properties[key];
        const present = key in sent;
        const errors = errorsAt(key);
        const ok = errors.length === 0;
        return (
          <li key={key} className={ok ? "is-ok" : "is-bad"}>
            <span className="tool-check-mark">{ok ? "✓" : "✕"}</span>
            <span className="tool-check-field">
              <code>{key}</code>
              {required.includes(key) && <small>required</small>}
            </span>
            <span className={`tool-check-sent${present ? "" : " is-absent"}`}>
              {present ? show(sent[key]) : "not sent"}
            </span>
            <span className="tool-check-rule">
              {schema.enum ? (
                <>
                  {schema.type} · one of{" "}
                  {schema.enum.map((option) => (
                    <i key={String(option)} className={present && sent[key] === option ? "is-match" : ""}>
                      {String(option)}
                    </i>
                  ))}
                </>
              ) : (
                ruleWords(schema)
              )}
            </span>
            {errors.map((error) => (
              <span key={error.message} className="tool-check-error">
                <code>{error.path}</code> {error.message}
              </span>
            ))}
          </li>
        );
      })}
      {extra.map((key) => (
        <li key={key} className="is-bad">
          <span className="tool-check-mark">✕</span>
          <span className="tool-check-field">
            <code>{key}</code>
            <small>not in the schema</small>
          </span>
          <span className="tool-check-sent">{show(sent[key])}</span>
          {errorsAt(key).map((error) => (
            <span key={error.message} className="tool-check-error">
              <code>{error.path}</code> {error.message}
            </span>
          ))}
        </li>
      ))}
      {loose.map((error) => (
        <li key={`${error.path}-${error.message}`} className="is-bad">
          <span className="tool-check-mark">✕</span>
          <span className="tool-check-error">
            <code>{error.path}</code> {error.message}
          </span>
        </li>
      ))}
      {run.errors.length === 0 && (
        <li className="is-empty">
          {run.stages.validate === "pass"
            ? "No validation errors."
            : run.stages.validate === "skipped"
              ? "Validation did not run because an earlier check failed."
              : ""}
        </li>
      )}
    </ul>
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
  const brokenRules = new Set(run.errors.map(ruleOf));
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
          <Crossing direction="to-model" carries="tool list" />
          <Stage number={1} title={`Sends ${TOOLS.length} tool schemas with the request`}>
            {TOOLS.map((tool) => tool.name).join(" · ")}
          </Stage>

          <Stage number={3} title="Writes a call as text">
            <code className="tool-snippet">{shownCall || "(empty)"}</code>
          </Stage>
          <Crossing direction="to-host" carries="call text" />
          <Stage number={4} title="Parse the text as JSON" status={stages.parse}>
            {run.parseError ? run.parseError : run.parsed ? `name = ${run.parsed.name}` : null}
          </Stage>

          <div className="tool-lane-wait">
            <strong>The model is not running</strong>
            <span>
              Stages 4–8 are host code. The model cannot see them or change them. It learns the outcome only from the
              tool message.
            </span>
          </div>
          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={5} title="Look up the tool name" status={stages.lookup}>
            {stages.lookup === "fail" ? `no tool named ${toolName}` : run.tool ? `${run.tool.name}: ${run.tool.effect}` : null}
          </Stage>

          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={6}title="Validate arguments against the schema" status={stages.validate}>
            {stages.validate === "fail"
              ? `${run.errors.length} error${run.errors.length === 1 ? "" : "s"}: ${run.errors.map((item) => item.path).join(", ")}`
              : stages.validate === "pass"
                ? "types, enums, required and extra fields all check out"
                : null}
          </Stage>

          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={7}title="Apply host policy" status={stages.policy}>
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

          <div className="tool-wall-line" aria-hidden="true" />
          <Stage number={8} title="Execute" status={stages.execute}>
            {run.sideEffect ?? (executed ? "read-only lookup; nothing changed" : null)}
          </Stage>

          <Stage number={10} title="Next turn reads the tool message">
            and nothing else about what the host did
          </Stage>
          <Crossing direction="to-model" carries="tool message" />
          <Stage
            number={9}
            title="Append the result as a tool message"
            note={failedAt ? `error from stage ${HOST_STAGE_NO[failedAt]}` : executed ? "result of stage 8" : undefined}
            noteTone={failedAt ? "error" : executed ? "ok" : undefined}
          >
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
          <div className="tool-contract">
            <figure className="tool-contract-side">
              <figcaption>The schema, as sent to the model</figcaption>
              <pre className="tool-schema" aria-label={`${shownTool.name} JSON schema`}>
                {schemaLines(shownTool).map((line, index) => {
                  const broken = line.rule !== undefined && brokenRules.has(line.rule);
                  return (
                    <span key={index} className={`tool-schema-line${broken ? " is-broken" : ""}`}>
                      {line.text}
                      {broken && <b> ✕ broken</b>}
                      {"\n"}
                    </span>
                  );
                })}
              </pre>
            </figure>
            <figure className="tool-contract-side">
              <figcaption>
                The call’s arguments, checked against it
                {conversation.second ? ` (turn ${turn})` : ""}
              </figcaption>
              <ArgumentCheck tool={shownTool} run={run} />
            </figure>
          </div>
        ) : (
          <>
            <ul className="tool-offered" aria-label="Tools the host advertises">
              {TOOLS.map((tool) => (
                <li key={tool.name}>
                  <code>{tool.name}</code>
                </li>
              ))}
              {run.parsed && (
                <li className="is-unknown">
                  <code>{toolName}</code> ✕ not offered
                </li>
              )}
            </ul>
            <p className="lab-note">
              {run.parseError
                ? `The host could not read a tool name: ${run.parseError}.`
                : `The host advertises only ${TOOLS.map((tool) => tool.name).join(", ")}. “${toolName}” is not one of them.`}
            </p>
            <ul className="tool-errors">
              <li className="is-empty">Validation did not run because an earlier check failed.</li>
            </ul>
          </>
        )}
      </LabSurface>

      <LabSurface label="Transcript" className="tool-transcript-card">
        <SurfaceHeading
          kicker="What the next model call receives"
          title={conversation.second ? "Messages, in order, across both turns" : "Messages, in order"}
        />
        <ol className="tool-transcript">
          <li className="is-host">
            <b>
              system<small>by the host</small>
            </b>
            <span>Tools available: {TOOLS.map((tool) => tool.name).join(", ")} (schemas attached)</span>
          </li>
          <li className="is-person">
            <b>
              user<small>by a person</small>
            </b>
            <span>{preset.user}</span>
          </li>
          <li className="is-model">
            <b>
              assistant<small>by the model{conversation.second ? " · turn 1" : ""}</small>
            </b>
            <code>{draft || "(empty)"}</code>
          </li>
          <li className={`is-host${conversation.first.stages.execute === "pass" ? "" : " is-error"}`}>
            <b>
              tool
              <small>
                by the host{conversation.first.stages.execute === "pass" ? "" : " · error"}
              </small>
            </b>
            <code>{conversation.first.result}</code>
          </li>
          {conversation.second && (
            <>
              <li className="is-model">
                <b>
                  assistant<small>by the model · turn 2</small>
                </b>
                <code>{retry || "(empty)"}</code>
              </li>
              <li className={`is-host${conversation.second.stages.execute === "pass" ? "" : " is-error"}`}>
                <b>
                  tool
                  <small>
                    by the host{conversation.second.stages.execute === "pass" ? "" : " · error"}
                  </small>
                </b>
                <code>{conversation.second.result}</code>
              </li>
            </>
          )}
          <li className="is-next">
            <b>
              assistant<small>not written yet</small>
            </b>
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
