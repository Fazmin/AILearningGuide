import { useMemo } from "react";
import {
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { CAP_MAX, runAgent, TASKS, type AgentEvent, type TaskId } from "./loop";
import {
  CATCH_RANGE,
  completionCurve,
  completionProbability,
  expectedStepsRun,
  GATE_RANGE,
  gatePosition,
  halfLifeSteps,
  percent,
  RUN_STEPS_RANGE,
  STEP_SUCCESS_RANGE,
  TABLE_STEPS,
  type Gate,
} from "./reliability";
import { asTask, boundedCatch, boundedInteger } from "./state";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

type Node = "context" | "model" | "decide" | "tool" | "gate" | "answer" | "halt";

function activeNodes(event: AgentEvent | undefined): Node[] {
  if (!event) return [];
  if (event.kind === "model") return ["context", "model", "decide"];
  if (event.kind === "tool") return ["tool", "gate"];
  return event.reason === "final answer" ? ["answer"] : ["gate", "halt"];
}

function LoopDiagram({ event, cap, guard, contextTokens }: { event?: AgentEvent; cap: number; guard: boolean; contextTokens: number }) {
  const on = activeNodes(event);
  const cls = (node: Node, extra = "") => `agent-node${on.includes(node) ? " is-active" : ""}${extra}`;
  const haltLabel = event?.kind === "stop" && event.reason === "host stop rule" ? "Stop · host rule" : "Stop · step cap";
  const describe =
    event?.kind === "model"
      ? `Model call ${event.call} reads ${event.inputTokens} tokens and ${event.final ? "writes a final answer" : "writes a tool call"}.`
      : event?.kind === "tool"
        ? `The host runs the tool and appends the result${event.failed ? ", an error" : ""}. Then it checks the cap${guard ? " and the stop rule" : ""}.`
        : event?.kind === "stop"
          ? `The loop stops: ${event.reason}.`
          : "";
  return (
    <svg className="agent-loop" viewBox="0 0 660 214" role="img" aria-label={`Agent loop diagram. ${describe}`}>
      <defs>
        <marker id="agent-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" className="agent-arrowhead" />
        </marker>
      </defs>
      <g className={cls("context")}>
        <rect x="10" y="62" width="120" height="58" rx="10" />
        <text x="70" y="86" textAnchor="middle">Context</text>
        <text x="70" y="104" textAnchor="middle" className="agent-node-sub">{contextTokens} tokens</text>
      </g>
      <g className={cls("model", " is-model")}>
        <rect x="170" y="62" width="126" height="58" rx="10" />
        <text x="233" y="86" textAnchor="middle">Model call</text>
        <text x="233" y="104" textAnchor="middle" className="agent-node-sub">writes text only</text>
      </g>
      <g className={cls("decide")}>
        <path d="M 386 55 L 440 91 L 386 127 L 332 91 Z" />
        <text x="386" y="88" textAnchor="middle">final</text>
        <text x="386" y="101" textAnchor="middle">answer?</text>
      </g>
      <g className={cls("answer", " is-exit")}>
        <rect x="486" y="10" width="160" height="44" rx="10" />
        <text x="566" y="30" textAnchor="middle">Stop · answer</text>
        <text x="566" y="45" textAnchor="middle" className="agent-node-sub">terminal action</text>
      </g>
      <g className={cls("tool", " is-host")}>
        <rect x="486" y="70" width="160" height="44" rx="10" />
        <text x="566" y="90" textAnchor="middle">Host runs the tool</text>
        <text x="566" y="105" textAnchor="middle" className="agent-node-sub">appends the result</text>
      </g>
      <g className={cls("gate", " is-host")}>
        <rect x="300" y="146" width="190" height="44" rx="10" />
        <text x="395" y="165" textAnchor="middle">Host check</text>
        <text x="395" y="180" textAnchor="middle" className="agent-node-sub">
          calls &lt; {cap}{guard ? " · no repeated failure" : ""}
        </text>
      </g>
      <g className={cls("halt", " is-exit")}>
        <rect x="10" y="150" width="150" height="40" rx="10" />
        <text x="85" y="174" textAnchor="middle">{haltLabel}</text>
      </g>
      <path className="agent-edge" d="M 130 91 L 168 91" markerEnd="url(#agent-arrow)" />
      <path className="agent-edge" d="M 296 91 L 330 91" markerEnd="url(#agent-arrow)" />
      <path className="agent-edge" d="M 418 76 C 440 40 460 32 484 32" markerEnd="url(#agent-arrow)" />
      <text className="agent-edge-label" x="424" y="50">yes</text>
      <path className="agent-edge" d="M 440 91 L 484 91" markerEnd="url(#agent-arrow)" />
      <text className="agent-edge-label" x="452" y="84">no</text>
      <path className="agent-edge" d="M 566 114 C 566 150 540 168 492 168" markerEnd="url(#agent-arrow)" />
      <path className="agent-edge" d="M 300 160 C 200 150 90 150 70 122" markerEnd="url(#agent-arrow)" />
      <text className="agent-edge-label" x="190" y="146">pass: next call</text>
      <path className="agent-edge is-dashed" d="M 300 180 L 162 175" markerEnd="url(#agent-arrow)" />
      <text className="agent-edge-label" x="206" y="194">fail</text>
    </svg>
  );
}

function ContextBars({ events, cap, tick }: { events: AgentEvent[]; cap: number; tick: number }) {
  const calls = events.filter((event): event is Extract<AgentEvent, { kind: "model" }> => event.kind === "model");
  const maxInput = Math.max(1, ...calls.map((call) => call.inputTokens));
  const slots = Math.max(cap, calls.length);
  const left = 40;
  const right = 640;
  const top = 36;
  const bottom = 150;
  const band = (right - left) / Math.max(1, slots);
  const barWidth = Math.min(44, band * 0.66);
  const y = (value: number) => bottom - (value / maxInput) * (bottom - top);
  const visibleCall = (() => {
    const upTo = events.slice(0, tick + 1).filter((event) => event.kind === "model");
    return upTo.length;
  })();
  return (
    <svg
      className="agent-bars"
      viewBox="0 0 660 188"
      role="img"
      aria-label={`Tokens read per model call: ${calls.map((call) => `call ${call.call} ${call.inputTokens}`).join(", ")}.`}
    >
      <line className="agent-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
      <line className="agent-axis" x1={left} x2={left} y1={top - 6} y2={bottom} />
      <text className="agent-tick" x={left - 6} y={top + 3} textAnchor="end">{maxInput}</text>
      <text className="agent-tick" x={left - 6} y={bottom + 3} textAnchor="end">0</text>
      {Array.from({ length: slots }, (_, index) => {
        const call = calls[index];
        const x = left + band * index + (band - barWidth) / 2;
        if (!call) {
          return (
            <g key={index} className="agent-bar is-empty">
              <rect x={x} y={bottom - 4} width={barWidth} height={4} rx={1.5} />
              <text className="agent-tick" x={x + barWidth / 2} y={bottom + 14} textAnchor="middle">{index + 1}</text>
            </g>
          );
        }
        const shown = index < visibleCall;
        return (
          <g key={index} className={`agent-bar${shown ? "" : " is-future"}`}>
            <rect className="is-base" x={x} y={y(call.baseTokens)} width={barWidth} height={bottom - y(call.baseTokens)} rx={1.5} />
            <rect
              className="is-history"
              x={x}
              y={y(call.inputTokens)}
              width={barWidth}
              height={Math.max(0, y(call.baseTokens) - y(call.inputTokens))}
              rx={1.5}
            />
            <text className="agent-bar-value" x={x + barWidth / 2} y={y(call.inputTokens) - 4} textAnchor="middle">
              {call.inputTokens}
            </text>
            <text className="agent-tick" x={x + barWidth / 2} y={bottom + 14} textAnchor="middle">{index + 1}</text>
          </g>
        );
      })}
      <text className="agent-tick" x={left} y={12}>tokens read per model call</text>
      <text className="agent-tick" x={right} y={bottom + 32} textAnchor="end">model call</text>
    </svg>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const task = asTask(state.task);
  const stepCap = clamp(Math.round(asNumber(state, "stepCap", 4)), 1, CAP_MAX);
  const guard = asString(state, "guard", "off") === "on";
  const run = useMemo(() => runAgent(task, stepCap, guard), [guard, stepCap, task]);
  const lastIndex = run.events.length - 1;
  const tick = clamp(Math.round(asNumber(state, "tick", lastIndex)), 0, lastIndex);
  const event = run.events[tick];
  const eventsSoFar = run.events.slice(0, tick + 1);
  const contextNow = (() => {
    const lastModel = [...eventsSoFar].reverse().find((item) => item.kind === "model");
    return lastModel && lastModel.kind === "model" ? lastModel.inputTokens : run.events[0]?.kind === "model" ? run.events[0].baseTokens : 0;
  })();
  const toolRuns = run.events.filter((item) => item.kind === "tool").length;
  const failures = run.events.filter((item) => item.kind === "tool" && item.failed).length;

  const stepSuccess = boundedInteger(state.stepSuccess, STEP_SUCCESS_RANGE.min, STEP_SUCCESS_RANGE.max, 95);
  const runSteps = boundedInteger(state.runSteps, RUN_STEPS_RANGE.min, RUN_STEPS_RANGE.max, 20);
  const gateAfter = boundedInteger(state.gateAfter, GATE_RANGE.min, GATE_RANGE.max, 0);
  const catchRate = boundedCatch(state.catchRate);
  const successShare = stepSuccess / 100;
  const gate: Gate | null = gateAfter > 0 ? { after: gateAfter, catchRate: catchRate / 100 } : null;
  const gateAt = gatePosition(runSteps, gate);
  const plainChance = completionProbability(successShare, runSteps);
  const gatedChance = gate ? completionProbability(successShare, runSteps, gate) : null;
  const stepsRun = expectedStepsRun(successShare, runSteps, gate);
  const half = halfLifeSteps(successShare);
  const curve = useMemo(
    () => completionCurve(successShare, gateAfter > 0 ? { after: gateAfter, catchRate: catchRate / 100 } : null),
    [catchRate, gateAfter, successShare],
  );
  const tableSteps = [...new Set<number>([...TABLE_STEPS, runSteps])].sort((a, b) => a - b);

  const eventLabel = (item: AgentEvent) =>
    item.kind === "model"
      ? `Model call ${item.call}`
      : item.kind === "tool"
        ? `Host · tool result`
        : `Stop`;

  return (
    <div className="tg-lab tg-lab--hero agent-lab">
      <LabSurface label="Agent loop" className="agent-loop-card">
        <SurfaceHeading
          kicker={`${TASKS[task].label} · ${run.calls} model call${run.calls === 1 ? "" : "s"} · stopped by ${run.stop}`}
          title="A loop the host runs around a model"
          aside={<span className="tg-badge">scripted policy</span>}
        />
        <div className="agent-task-control">
          <SegmentedControl
            label="Task"
            value={task}
            options={(Object.keys(TASKS) as TaskId[]).map((key) => ({ value: key, label: TASKS[key].label }))}
            onChange={(value) => setState({ task: value, tick: 99 })}
          />
        </div>
        <LoopDiagram event={event} cap={stepCap} guard={guard} contextTokens={contextNow} />
        <RangeControl
          label="Replay step"
          min={0}
          max={lastIndex}
          step={1}
          value={tick}
          format={(value) => {
            const item = run.events[value];
            return item ? `${value + 1}/${lastIndex + 1} · ${eventLabel(item)}` : `${value + 1}`;
          }}
          onChange={(value) => setState({ tick: value })}
        />
        <p className="agent-now">
          {event?.kind === "model"
            ? `Model call ${event.call} reads the whole context — ${event.inputTokens} tokens — and writes ${event.final ? "a final answer" : "a tool call"} of ${event.outputTokens} tokens.`
            : event?.kind === "tool"
              ? `The host runs the call and appends ${event.failed ? "an error" : "the result"}. Then it checks calls < ${stepCap}${guard ? " and the repeat rule" : ""}.`
              : event?.kind === "stop"
                ? `Stopped by ${event.reason} after ${event.call} model call${event.call === 1 ? "" : "s"}.`
                : ""}
        </p>
      </LabSurface>

      <LabSurface label="Trace" className="agent-trace-card">
        <SurfaceHeading kicker={`User: ${TASKS[task].user}`} title="Every message, in the order the host wrote it" />
        <ol className="agent-trace">
          {run.events.map((item, index) => (
            <li
              key={index}
              className={`is-${item.kind}${index === tick ? " is-current" : ""}${index > tick ? " is-future" : ""}${
                item.kind === "tool" && item.failed ? " is-failed" : ""
              }`}
            >
              <button type="button" aria-current={index === tick ? "step" : undefined} onClick={() => setState({ tick: index })}>
                <b>{eventLabel(item)}</b>
                {item.kind === "model" && (
                  <>
                    <code>{item.text}</code>
                    <small>
                      read {item.inputTokens} · wrote {item.outputTokens} tokens{item.final ? " · final answer" : ""}
                    </small>
                  </>
                )}
                {item.kind === "tool" && <code>{item.text}</code>}
                {item.kind === "stop" && <span>{item.reason}</span>}
              </button>
            </li>
          ))}
        </ol>
      </LabSurface>

      <LabSurface label="Budgets" className="agent-budget-card">
        <SurfaceHeading kicker="Horizon and stop rules" title="What ends the loop" />
        <RangeControl
          label="Step cap"
          min={1}
          max={CAP_MAX}
          step={1}
          value={stepCap}
          format={(value) => `${value} model call${value === 1 ? "" : "s"}`}
          onChange={(value) => setState({ stepCap: value, tick: 99 })}
        />
        <div className="agent-guard-control">
          <SegmentedControl
            label="Stop rule"
            value={guard ? "on" : "off"}
            options={[
              { value: "off", label: "none" },
              { value: "on", label: "stop after 2 identical failures" },
            ]}
            onChange={(value) => setState({ guard: value, tick: 99 })}
          />
        </div>
        <div className="metric-row agent-metrics">
          <Metric label="Model calls" value={`${run.calls}`} />
          <Metric label="Tool runs" value={`${toolRuns} (${failures} failed)`} tone={failures ? "loss" : undefined} />
          <Metric label="Stop reason" value={run.stop} tone={run.stop === "final answer" ? "forward" : "loss"} />
        </div>
        <div className="metric-row agent-metrics">
          <Metric label="Tokens read" value={`${run.inputTokens}`} />
          <Metric label="Tokens written" value={`${run.outputTokens}`} />
        </div>
      </LabSurface>

      <LabSurface label="Context growth" className="agent-growth-card">
        <SurfaceHeading
          kicker={`${run.calls} of ${stepCap} allowed calls used · ${run.inputTokens.toLocaleString("en-US")} tokens read in total`}
          title="Every call re-reads everything before it"
        />
        <ContextBars events={run.events} cap={stepCap} tick={tick} />
        <div className="agent-bar-legend" aria-hidden="true">
          <span>
            <i className="is-base" /> system prompt + user message
          </span>
          <span>
            <i className="is-history" /> earlier calls and tool results
          </span>
          <span>
            <i className="is-empty" /> allowed but unused
          </span>
        </div>
        <p className="lab-note">
          Each bar is one model call's input. The loop appends a call and a result every turn, so input grows each
          time and the total in the heading grows faster still. Token counts are a toy: runs of letters, runs of digits,
          and single symbols. Bars past the replay point are faded.
        </p>
      </LabSurface>

      <LabSurface label="Reliability" className="agent-reliability-card">
        <SurfaceHeading
          kicker={`${stepSuccess}% right per step · ${runSteps} step${runSteps === 1 ? "" : "s"} · ${
            gate ? `approval gate after step ${gateAt}` : "no gate"
          }`}
          title="Small errors compound across a long run"
          aside={<span className="tg-badge">closed-form formula</span>}
        />
        <div className="agent-rel-controls">
          <RangeControl
            label="Per-step success"
            min={STEP_SUCCESS_RANGE.min}
            max={STEP_SUCCESS_RANGE.max}
            step={1}
            value={stepSuccess}
            format={(value) => `${value}%`}
            onChange={(value) => setState({ stepSuccess: value })}
          />
          <RangeControl
            label="Steps in the run"
            min={RUN_STEPS_RANGE.min}
            max={RUN_STEPS_RANGE.max}
            step={1}
            value={runSteps}
            format={(value) => `${value} step${value === 1 ? "" : "s"}`}
            onChange={(value) => setState({ runSteps: value })}
          />
          <RangeControl
            label="Approval gate after step"
            min={GATE_RANGE.min}
            max={GATE_RANGE.max}
            step={1}
            value={gateAfter}
            format={(value) => (value === 0 ? "no gate" : value > runSteps ? `${runSteps} (end of run)` : String(value))}
            onChange={(value) => setState({ gateAfter: value })}
          />
          <RangeControl
            label="Reviewer catches"
            min={CATCH_RANGE.min}
            max={CATCH_RANGE.max}
            step={5}
            value={catchRate}
            format={(value) => `${value}% of wrong runs`}
            onChange={(value) => setState({ catchRate: value })}
          />
        </div>
        <div className="metric-row agent-metrics">
          <Metric label="Completes end to end" value={percent(plainChance)} tone={plainChance < 0.5 ? "loss" : "forward"} />
          <Metric
            label="With the gate"
            value={gatedChance === null ? "no gate" : percent(gatedChance)}
            tone={gatedChance === null ? undefined : "forward"}
          />
          <Metric
            label="Longest run that finishes half the time"
            value={half === null ? "no limit" : `${half} step${half === 1 ? "" : "s"}`}
          />
          <Metric label="Expected steps run" value={stepsRun.toFixed(1)} />
        </div>
        <LineChart
          label="Chance that a run completes every step, by run length"
          xLabel="steps in the run"
          yLabel="probability every step is right"
          xDomain={[1, RUN_STEPS_RANGE.max]}
          yDomain={[0, 1]}
          marker={{ x: runSteps, label: `${runSteps} step${runSteps === 1 ? "" : "s"}` }}
          series={[
            {
              id: "plain",
              name: "No gate",
              tone: "loss",
              points: curve.map((point) => ({ x: point.steps, y: point.plain })),
              format: (value) => percent(value),
            },
            ...(gate
              ? [
                  {
                    id: "gated",
                    name: `Gate after step ${gateAfter}`,
                    tone: "forward" as const,
                    dash: "dashed" as const,
                    points: curve.map((point) => ({ x: point.steps, y: point.gated ?? 0 })),
                    format: (value: number) => percent(value),
                  },
                ]
              : []),
          ]}
        />
        <div className="source-table agent-rel-table">
          <table className="tg-board" aria-label="Chance a run completes, by number of steps">
            <thead>
              <tr>
                <th scope="col">Steps</th>
                <th scope="col">No gate</th>
                <th scope="col">{gate ? `Gate after step ${gateAfter}` : "With a gate"}</th>
              </tr>
            </thead>
            <tbody>
              {tableSteps.map((count) => (
                <tr key={count} aria-current={count === runSteps ? "true" : undefined}>
                  <th scope="row">{count === runSteps ? `${count} (current)` : count}</th>
                  <td>{percent(completionProbability(successShare, count))}</td>
                  <td>{gate ? percent(completionProbability(successShare, count, gate)) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Every figure is a formula, not a sample. It assumes each step succeeds independently with the same chance, that
          the agent cannot see its own wrong steps, and that the reviewer is the only check. A caught run is redone once,
          and the redo is not checked again. Real steps differ in difficulty and their errors are often linked.
        </p>
      </LabSurface>
    </div>
  );
}
