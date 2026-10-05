/**
 * A host-side agent loop, run for real over a scripted policy.
 *
 * The "model" is a fixed rule that maps the transcript to its next message, so
 * it stands in for sampling. Everything else is computed: the loop, the step
 * cap, the host stop rule, the tool executions, the context that every model
 * call re-reads, and toy token counts.
 */

export type TaskId = "hours" | "recover" | "loop" | "repair";

export const CAP_MAX = 10;

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  text: string;
}

export const SYSTEM_PROMPT =
  "You are the harbor assistant. Tools: get_dock_hours(dock, day), get_ferry_time(sailing). Call one tool at a time as JSON. When you can answer, reply in plain text.";

export const TASKS: Record<TaskId, { label: string; user: string }> = {
  hours: { label: "Look up hours", user: "When does the north dock close on Sunday?" },
  recover: { label: "Recover from error", user: "What time does the afternoon ferry leave?" },
  loop: { label: "Retry forever", user: "What is the wind speed at the harbor right now?" },
  repair: { label: "Fix an argument", user: "When does the south dock close on a weekday?" },
};

/** Toy token count: runs of letters, runs of digits, and single symbols. */
export const countTokens = (text: string) => (text.match(/[A-Za-z]+|\d+|[^\sA-Za-z\d]/g) ?? []).length;

const CALLS: Record<TaskId, string> = {
  hours: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}',
  recover: '{"name": "get_ferry_time", "arguments": {"sailing": "afternoon"}}',
  loop: '{"name": "get_wind", "arguments": {}}',
  // The first call has the right dock and day, but "Weekday" is not one of the schema's values.
  repair: '{"name": "get_dock_hours", "arguments": {"dock": "south", "day": "Weekday"}}',
};

const FINALS: Record<TaskId, string> = {
  hours: "The north dock closes at 16:00 on Sundays.",
  recover: "The afternoon ferry leaves at 15:40.",
  loop: "",
  repair: "The south dock closes at 21:00 on weekdays.",
};

/** Writes a call the way the scripted calls above are written, so token counts stay comparable. */
const formatCall = (name: string, args: Record<string, unknown>) =>
  `{"name": "${name}", "arguments": {${Object.entries(args)
    .map(([key, value]) => `"${key}": ${JSON.stringify(value)}`)
    .join(", ")}}}`;

/**
 * The repair rule: read the host's error text and fix what it names. For each detail of the form
 * `/arg: must be one of "a", "b"; got "x"` it swaps in the listed value that matches x apart from case
 * and spacing. Anything it cannot read leaves the call unchanged. This is a fixed string rule standing
 * in for a model that would infer the fix from language; it does not understand the message.
 */
export function correctArguments(callText: string, errorText: string): string {
  try {
    const call = JSON.parse(callText) as { name: string; arguments: Record<string, unknown> };
    const error = JSON.parse(errorText) as { details?: unknown };
    if (!Array.isArray(error.details)) return callText;
    const fixed: Record<string, unknown> = { ...call.arguments };
    let changed = false;
    for (const detail of error.details) {
      const found = /^\/(\w+): must be one of (.+); got (".*")$/.exec(String(detail));
      if (!found) continue;
      const allowed = JSON.parse(`[${found[2]}]`) as string[];
      const got = String(JSON.parse(found[3])).trim().toLowerCase();
      const match = allowed.find((value) => value.toLowerCase() === got);
      if (match !== undefined) {
        fixed[found[1]] = match;
        changed = true;
      }
    }
    return changed ? formatCall(call.name, fixed) : callText;
  } catch {
    return callText;
  }
}

/**
 * The scripted policy. It reads only the transcript: answer if the last tool result succeeded,
 * otherwise issue the next call. For most tasks that is the same call again. For the repair task
 * it is the previous call with the arguments the error text names corrected. It has no stop rule
 * of its own.
 */
export function policy(task: TaskId, transcript: readonly Message[]): { text: string; final: boolean } {
  const last = transcript[transcript.length - 1];
  if (last?.role === "tool" && !last.text.includes('"error"')) return { text: FINALS[task], final: true };
  if (task === "repair" && last?.role === "tool") {
    const previous = [...transcript].reverse().find((message) => message.role === "assistant");
    if (previous) return { text: correctArguments(previous.text, last.text), final: false };
  }
  return { text: CALLS[task], final: false };
}

const DOCK_CLOSES: Record<string, Record<string, string>> = {
  north: { weekday: "18:00", sunday: "16:00" },
  south: { weekday: "21:00", sunday: "21:00" },
};
const DOCK_ENUMS: Record<string, readonly string[]> = { dock: ["north", "south"], day: ["weekday", "sunday"] };

/** Checks the two enum arguments the way the previous lab's validator does, with the same message wording. */
function dockHoursErrors(args: Record<string, unknown>): string[] {
  const errors: string[] = [];
  for (const [key, allowed] of Object.entries(DOCK_ENUMS)) {
    const value = args[key];
    if (value === undefined) errors.push(`/${key}: required property is missing`);
    else if (!allowed.includes(value as string)) {
      errors.push(`/${key}: must be one of ${allowed.map((item) => JSON.stringify(item)).join(", ")}; got ${JSON.stringify(value)}`);
    }
  }
  return errors;
}

/** The host's tools. get_wind was never advertised, so the host rejects it every time. */
export function runTool(callText: string, attempt: number): string {
  const call = JSON.parse(callText) as { name: string; arguments: Record<string, string> };
  if (call.name === "get_dock_hours") {
    const errors = dockHoursErrors(call.arguments);
    if (errors.length) return `{"error": "invalid arguments", "details": ${JSON.stringify(errors)}}`;
    const { dock, day } = call.arguments;
    return `{"dock": "${dock}", "day": "${day}", "closes": "${DOCK_CLOSES[dock][day]}"}`;
  }
  if (call.name === "get_ferry_time") {
    return attempt === 1 ? '{"error": "timeout after 5 s"}' : '{"sailing": "afternoon", "departs": "15:40"}';
  }
  return `{"error": "unknown tool ${call.name}"}`;
}

export type StopReason = "final answer" | "step cap" | "host stop rule";

export type AgentEvent =
  | {
      kind: "model";
      call: number;
      text: string;
      final: boolean;
      inputTokens: number;
      baseTokens: number;
      outputTokens: number;
    }
  | { kind: "tool"; call: number; text: string; failed: boolean }
  | { kind: "stop"; reason: StopReason; call: number };

export interface AgentRun {
  events: AgentEvent[];
  calls: number;
  stop: StopReason;
  inputTokens: number;
  outputTokens: number;
  transcript: Message[];
}

/**
 * while calls < cap:
 *   output = model(transcript)            // reads the whole transcript
 *   if output is final: stop
 *   result = host.run(output); append both
 *   if guard and the same call failed twice in a row: stop
 * stop at the cap
 */
export function runAgent(task: TaskId, cap: number, guard: boolean): AgentRun {
  const transcript: Message[] = [
    { role: "system", text: SYSTEM_PROMPT },
    { role: "user", text: TASKS[task].user },
  ];
  const baseTokens = transcript.reduce((sum, message) => sum + countTokens(message.text), 0);
  const events: AgentEvent[] = [];
  let calls = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopReason = "step cap";
  let attempts = 0;
  let sameFailures = 0;
  let lastFailedCall = "";

  while (calls < cap) {
    calls += 1;
    const input = transcript.reduce((sum, message) => sum + countTokens(message.text), 0);
    const output = policy(task, transcript);
    const produced = countTokens(output.text);
    inputTokens += input;
    outputTokens += produced;
    events.push({
      kind: "model",
      call: calls,
      text: output.text,
      final: output.final,
      inputTokens: input,
      baseTokens,
      outputTokens: produced,
    });
    transcript.push({ role: "assistant", text: output.text });
    if (output.final) {
      stop = "final answer";
      break;
    }
    attempts += 1;
    const result = runTool(output.text, attempts);
    const failed = result.includes('"error"');
    events.push({ kind: "tool", call: calls, text: result, failed });
    transcript.push({ role: "tool", text: result });
    sameFailures = failed && output.text === lastFailedCall ? sameFailures + 1 : failed ? 1 : 0;
    lastFailedCall = failed ? output.text : "";
    if (guard && sameFailures >= 2) {
      stop = "host stop rule";
      break;
    }
  }
  events.push({ kind: "stop", reason: stop, call: calls });
  return { events, calls, stop, inputTokens, outputTokens, transcript };
}
