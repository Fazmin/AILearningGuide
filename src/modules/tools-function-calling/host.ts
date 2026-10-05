/**
 * The host side of a tool call, computed for real: parse the model's text as
 * JSON, validate the arguments against a JSON-Schema subset, apply a policy,
 * run the tool, and serialize the result as the next message.
 * The model side is only text — the presets below are authored model outputs.
 */

export type Schema = {
  type?: "object" | "string" | "integer" | "number" | "boolean";
  description?: string;
  properties?: Record<string, Schema>;
  required?: string[];
  additionalProperties?: boolean;
  enum?: ReadonlyArray<string | number>;
  maxLength?: number;
  pattern?: string;
  minimum?: number;
  maximum?: number;
};

export type Policy = "auto" | "approval";

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Schema;
  policy: Policy;
  effect: "read-only" | "writes host memory" | "sends a message outside";
}

export const TOOLS: readonly ToolSpec[] = [
  {
    name: "get_dock_hours",
    description: "Closing time for one dock on one kind of day.",
    parameters: {
      type: "object",
      properties: {
        dock: { type: "string", enum: ["north", "south"] },
        day: { type: "string", enum: ["weekday", "sunday"] },
      },
      required: ["dock", "day"],
      additionalProperties: false,
    },
    policy: "auto",
    effect: "read-only",
  },
  {
    name: "add_note",
    description: "Save a short reminder in the host's notes.",
    parameters: {
      type: "object",
      properties: { text: { type: "string", maxLength: 60 } },
      required: ["text"],
      additionalProperties: false,
    },
    policy: "auto",
    effect: "writes host memory",
  },
  {
    name: "send_email",
    description: "Send an email from the harbor office account.",
    parameters: {
      type: "object",
      properties: {
        to: { type: "string", pattern: "^[^@\\s]+@[^@\\s]+$" },
        body: { type: "string", maxLength: 200 },
      },
      required: ["to", "body"],
      additionalProperties: false,
    },
    policy: "approval",
    effect: "sends a message outside",
  },
];

export interface ValidationError {
  path: string;
  message: string;
  /** A grammar compiled from the schema can rule this out while decoding. */
  grammarPreventable: boolean;
}

const typeOf = (value: unknown) => {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value;
};

const matchesType = (expected: Schema["type"], value: unknown) => {
  const actual = typeOf(value);
  if (!expected) return true;
  if (expected === "number") return actual === "number" || actual === "integer";
  return actual === expected;
};

/** Validate against the subset of JSON Schema used above. Paths use JSON-pointer style. */
export function validate(schema: Schema, value: unknown, path = ""): ValidationError[] {
  const where = path || "/";
  if (!matchesType(schema.type, value)) {
    return [{ path: where, message: `expected ${schema.type}, got ${typeOf(value)}`, grammarPreventable: true }];
  }
  const errors: ValidationError[] = [];
  if (schema.enum && !schema.enum.includes(value as string | number)) {
    errors.push({
      path: where,
      message: `must be one of ${schema.enum.map((item) => JSON.stringify(item)).join(", ")}; got ${JSON.stringify(value)}`,
      grammarPreventable: true,
    });
  }
  if (typeof value === "string") {
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      errors.push({
        path: where,
        message: `longer than ${schema.maxLength} characters (${value.length})`,
        grammarPreventable: true,
      });
    }
    if (schema.pattern && !new RegExp(schema.pattern).test(value)) {
      errors.push({ path: where, message: `does not match ${schema.pattern}`, grammarPreventable: true });
    }
  }
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) {
      errors.push({ path: where, message: `below minimum ${schema.minimum}`, grammarPreventable: true });
    }
    if (schema.maximum !== undefined && value > schema.maximum) {
      errors.push({ path: where, message: `above maximum ${schema.maximum}`, grammarPreventable: true });
    }
  }
  if (schema.type === "object" && value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (!(key in record)) {
        errors.push({ path: `${path}/${key}`, message: "required property is missing", grammarPreventable: true });
      }
    }
    for (const [key, child] of Object.entries(record)) {
      const childSchema = schema.properties?.[key];
      if (!childSchema) {
        if (schema.additionalProperties === false) {
          errors.push({ path: `${path}/${key}`, message: "property not allowed by the schema", grammarPreventable: true });
        }
        continue;
      }
      errors.push(...validate(childSchema, child, `${path}/${key}`));
    }
  }
  return errors;
}

export type StageStatus = "pass" | "fail" | "skipped";

export interface HostRun {
  parsed: { name: string; arguments: Record<string, unknown> } | null;
  parseError: string | null;
  tool: ToolSpec | null;
  errors: ValidationError[];
  stages: { parse: StageStatus; lookup: StageStatus; validate: StageStatus; policy: StageStatus; execute: StageStatus };
  /** The JSON string the host appends as the tool message. */
  result: string;
  sideEffect: string | null;
  /** Could constrained decoding against these schemas have prevented the failure? */
  grammarWouldPrevent: boolean | null;
}

const DOCK_HOURS: Record<string, Record<string, string>> = {
  north: { weekday: "18:00", sunday: "16:00" },
  south: { weekday: "21:00", sunday: "21:00" },
};

function execute(tool: ToolSpec, args: Record<string, unknown>, notesBefore: number) {
  if (tool.name === "get_dock_hours") {
    const closes = DOCK_HOURS[String(args.dock)]?.[String(args.day)] ?? "unknown";
    return { result: { dock: args.dock, day: args.day, closes }, sideEffect: null };
  }
  if (tool.name === "add_note") {
    return {
      result: { saved: true, note_id: notesBefore + 1 },
      sideEffect: `Host saved note #${notesBefore + 1}: “${String(args.text)}”`,
    };
  }
  return {
    result: { queued: true, to: args.to },
    sideEffect: `Host queued an email to ${String(args.to)} (simulated; nothing leaves this app)`,
  };
}

export function runHost(text: string, approved: boolean, notesBefore = 0): HostRun {
  const skipped = { parse: "skipped", lookup: "skipped", validate: "skipped", policy: "skipped", execute: "skipped" } as const;
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : "invalid JSON";
    return {
      parsed: null,
      parseError: message,
      tool: null,
      errors: [],
      stages: { ...skipped, parse: "fail" },
      result: JSON.stringify({ error: "could not parse the tool call as JSON" }),
      sideEffect: null,
      grammarWouldPrevent: true,
    };
  }
  const envelope = raw as Record<string, unknown> | null;
  if (
    !envelope ||
    typeof envelope !== "object" ||
    Array.isArray(envelope) ||
    typeof envelope.name !== "string" ||
    !envelope.arguments ||
    typeof envelope.arguments !== "object" ||
    Array.isArray(envelope.arguments)
  ) {
    return {
      parsed: null,
      parseError: "expected an object with a string \"name\" and an object \"arguments\"",
      tool: null,
      errors: [],
      stages: { ...skipped, parse: "fail" },
      result: JSON.stringify({ error: "tool call must be {\"name\": string, \"arguments\": object}" }),
      sideEffect: null,
      grammarWouldPrevent: true,
    };
  }
  const parsed = { name: envelope.name, arguments: envelope.arguments as Record<string, unknown> };
  const tool = TOOLS.find((item) => item.name === parsed.name) ?? null;
  if (!tool) {
    return {
      parsed,
      parseError: null,
      tool: null,
      errors: [],
      stages: { ...skipped, parse: "pass", lookup: "fail" },
      result: JSON.stringify({ error: `unknown tool ${parsed.name}`, available: TOOLS.map((item) => item.name) }),
      sideEffect: null,
      grammarWouldPrevent: true,
    };
  }
  const errors = validate(tool.parameters, parsed.arguments);
  if (errors.length) {
    return {
      parsed,
      parseError: null,
      tool,
      errors,
      stages: { ...skipped, parse: "pass", lookup: "pass", validate: "fail" },
      result: JSON.stringify({ error: "invalid arguments", details: errors.map((item) => `${item.path}: ${item.message}`) }),
      sideEffect: null,
      grammarWouldPrevent: errors.every((item) => item.grammarPreventable),
    };
  }
  if (tool.policy === "approval" && !approved) {
    return {
      parsed,
      parseError: null,
      tool,
      errors,
      stages: { ...skipped, parse: "pass", lookup: "pass", validate: "pass", policy: "fail" },
      result: JSON.stringify({ error: `${tool.name} needs human approval; none was given` }),
      sideEffect: null,
      grammarWouldPrevent: false,
    };
  }
  const done = execute(tool, parsed.arguments, notesBefore);
  return {
    parsed,
    parseError: null,
    tool,
    errors,
    stages: { parse: "pass", lookup: "pass", validate: "pass", policy: "pass", execute: "pass" },
    result: JSON.stringify(done.result),
    sideEffect: done.sideEffect,
    grammarWouldPrevent: null,
  };
}

export interface Preset {
  id: string;
  label: string;
  user: string;
  output: string;
  /**
   * An authored second call, written as if the model had read the first call's error message.
   * Only the "error, then retry" presets carry one; the host still checks it for real.
   */
  retry?: string;
}

export const PRESETS: readonly Preset[] = [
  {
    id: "valid",
    label: "valid call",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}',
  },
  {
    id: "missing",
    label: "missing field",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"day": "sunday"}}',
  },
  {
    id: "enum",
    label: "bad enum",
    user: "When does the east dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "east", "day": "sunday"}}',
  },
  {
    id: "type",
    label: "wrong type",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": 7}}',
  },
  {
    id: "extra",
    label: "extra field",
    user: "When does the north dock close on Sunday? It's urgent.",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday", "urgent": true}}',
  },
  {
    id: "broken",
    label: "not JSON",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"',
  },
  {
    id: "unknown",
    label: "unknown tool",
    user: "Is it windy at the harbor?",
    output: '{"name": "get_weather", "arguments": {"port": "Gull Harbor"}}',
  },
  {
    id: "note",
    label: "add a note",
    user: "Remind me to bring the dog's tag.",
    output: '{"name": "add_note", "arguments": {"text": "Bring the dog\'s tag"}}',
  },
  {
    id: "email",
    label: "email the log",
    user: "Email the visitor log to everyone at the harbor.",
    output: '{"name": "send_email", "arguments": {"to": "all@harbor.example", "body": "Full visitor log attached."}}',
  },
  {
    id: "retry-type",
    label: "wrong type, then fixed",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": 7}}',
    retry: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}',
  },
  {
    id: "retry-missing",
    label: "missing field, then fixed",
    user: "When does the north dock close on Sunday?",
    output: '{"name": "get_dock_hours", "arguments": {"day": "sunday"}}',
    retry: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}',
  },
];

export const presetById = (id: string) => PRESETS.find((item) => item.id === id) ?? PRESETS[0];

/** The presets that carry an authored retry. */
export const REPAIR_IDS: readonly string[] = PRESETS.filter((item) => item.retry !== undefined).map((item) => item.id);

export interface Conversation {
  first: HostRun;
  /** Runs only when the first call did not execute and there is a retry to run. */
  second: HostRun | null;
}

/**
 * Two turns of the same loop. The host reports the first call's failure as an ordinary tool message;
 * the retry text is whatever the model wrote next (authored here, or typed by the learner), and the
 * host checks it from scratch. Nothing in the first run changes what the retry is allowed to be.
 */
export function runConversation(first: string, retry: string, approved: boolean, notesBefore = 0): Conversation {
  const firstRun = runHost(first, approved, notesBefore);
  const needsRetry = firstRun.stages.execute !== "pass" && retry.trim() !== "";
  return { first: firstRun, second: needsRetry ? runHost(retry, approved, notesBefore) : null };
}
