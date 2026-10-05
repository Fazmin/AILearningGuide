import { describe, expect, it } from "vitest";
import { hydrateToolsState } from "./module";
import { presetById, PRESETS, REPAIR_IDS, runConversation, runHost, TOOLS, validate } from "./host";
import { DRAFT_MAX, initialState } from "./state";

const run = (id: string, approved = false) => runHost(presetById(id).output, approved);

describe("schema validation", () => {
  const schema = TOOLS[0].parameters;

  it("accepts a valid argument object", () => {
    expect(validate(schema, { dock: "north", day: "sunday" })).toEqual([]);
  });

  it("reports missing, enum, type and extra-field errors with paths", () => {
    expect(validate(schema, { day: "sunday" }).map((error) => error.path)).toEqual(["/dock"]);
    expect(validate(schema, { dock: "east", day: "sunday" })[0].message).toMatch(/must be one of/);
    expect(validate(schema, { dock: "north", day: 7 })[0].message).toBe("expected string, got integer");
    expect(validate(schema, { dock: "north", day: "sunday", urgent: true })[0].path).toBe("/urgent");
    expect(validate(schema, { dock: "North", day: "sunday" })).toHaveLength(1);
  });

  it("checks maxLength and pattern", () => {
    expect(validate(TOOLS[1].parameters, { text: "x".repeat(61) })[0].message).toMatch(/longer than 60/);
    expect(validate(TOOLS[2].parameters, { to: "nobody", body: "hi" })[0].path).toBe("/to");
  });
});

describe("the host pipeline", () => {
  it("runs a valid lookup and returns the closing time", () => {
    const result = run("valid");
    expect(result.stages.execute).toBe("pass");
    expect(JSON.parse(result.result)).toEqual({ dock: "north", day: "sunday", closes: "16:00" });
  });

  it("stops each malformed preset at the stage the lesson names", () => {
    expect(run("broken").stages.parse).toBe("fail");
    expect(run("unknown").stages.lookup).toBe("fail");
    for (const id of ["missing", "enum", "type", "extra"]) {
      const result = run(id);
      expect(result.stages.parse).toBe("pass");
      expect(result.stages.validate).toBe("fail");
      expect(result.stages.execute).toBe("skipped");
      expect(result.grammarWouldPrevent).toBe(true);
    }
  });

  it("refuses a valid send_email without approval and runs it with approval", () => {
    const refused = run("email");
    expect(refused.stages.validate).toBe("pass");
    expect(refused.stages.policy).toBe("fail");
    expect(refused.grammarWouldPrevent).toBe(false);
    expect(refused.sideEffect).toBeNull();
    const approved = run("email", true);
    expect(approved.stages.execute).toBe("pass");
    expect(approved.sideEffect).toMatch(/simulated/);
  });

  it("returns every failure as a JSON tool message", () => {
    for (const preset of PRESETS) {
      expect(() => JSON.parse(runHost(preset.output, false).result)).not.toThrow();
    }
  });

  it("rejects JSON that is not a call envelope", () => {
    expect(runHost("[1, 2]", false).stages.parse).toBe("fail");
    expect(runHost('{"name": "add_note"}', false).stages.parse).toBe("fail");
  });
});

describe("the error-then-retry presets", () => {
  const converse = (id: string, approved = false) => {
    const preset = presetById(id);
    return runConversation(preset.output, preset.retry ?? "", approved);
  };

  it("counts eleven authored first calls, two of them with a retry", () => {
    expect(PRESETS).toHaveLength(11);
    expect(REPAIR_IDS).toEqual(["retry-type", "retry-missing"]);
    expect(PRESETS.filter((preset) => preset.retry === undefined)).toHaveLength(9);
  });

  it("returns a wrong type as a structured error and runs the corrected retry", () => {
    const { first, second } = converse("retry-type");
    expect(first.stages.validate).toBe("fail");
    expect(first.stages.execute).toBe("skipped");
    expect(first.result).toBe('{"error":"invalid arguments","details":["/day: expected string, got integer"]}');
    expect(first.grammarWouldPrevent).toBe(true);
    expect(second).not.toBeNull();
    expect(second!.stages.execute).toBe("pass");
    expect(second!.result).toBe('{"dock":"north","day":"sunday","closes":"16:00"}');
  });

  it("returns a missing field as a structured error and runs the corrected retry", () => {
    const { first, second } = converse("retry-missing");
    expect(first.result).toBe('{"error":"invalid arguments","details":["/dock: required property is missing"]}');
    expect(second!.result).toBe('{"dock":"north","day":"sunday","closes":"16:00"}');
  });

  it("checks the retry from scratch: a repeated mistake fails with the same error", () => {
    const preset = presetById("retry-type");
    const { first, second } = runConversation(preset.output, preset.output, false);
    expect(second!.stages.validate).toBe("fail");
    expect(second!.result).toBe(first.result);
  });

  it("sends no retry when the first call executes or when there is no retry text", () => {
    expect(runConversation(presetById("valid").output, presetById("retry-type").retry!, false).second).toBeNull();
    expect(runConversation(presetById("missing").output, "", false).second).toBeNull();
    expect(runConversation(presetById("missing").output, "   ", false).second).toBeNull();
  });

  it("lets the learner type a different retry and still checks it for real", () => {
    const badEnum = '{"name": "get_dock_hours", "arguments": {"dock": "east", "day": "sunday"}}';
    const { second } = runConversation(presetById("retry-type").output, badEnum, false);
    expect(second!.stages.validate).toBe("fail");
    expect(second!.errors[0].path).toBe("/dock");
  });
});

describe("state", () => {
  it("migrates version 1 tasks to presets and drops step", () => {
    const state = hydrateToolsState(JSON.stringify({ task: "mail", step: 3 }));
    expect(state.preset).toBe("email");
    expect(state.draft).toBe(presetById("email").output);
    expect(state).not.toHaveProperty("step");
    expect(state).not.toHaveProperty("task");
  });

  it("keeps an edited draft", () => {
    const state = hydrateToolsState(JSON.stringify({ preset: "valid", draft: "{}", approve: "yes" }));
    expect(state.draft).toBe("{}");
    expect(state.approve).toBe("yes");
  });
});

describe("state versions", () => {
  it("reads a version 2 payload and fills the retry and turn", () => {
    const state = hydrateToolsState(JSON.stringify({ preset: "email", draft: presetById("email").output, approve: "yes" }));
    expect(state).toEqual({ preset: "email", draft: presetById("email").output, retry: "", approve: "yes", turn: "1" });
  });

  it("fills a repair preset's authored retry when an older payload has none", () => {
    const state = hydrateToolsState(JSON.stringify({ preset: "retry-type", draft: presetById("retry-type").output }));
    expect(state.retry).toBe(presetById("retry-type").retry);
  });

  it("keeps an emptied retry, drops a stray retry on a single-turn preset and validates the turn", () => {
    expect(hydrateToolsState(JSON.stringify({ preset: "retry-type", retry: "" })).retry).toBe("");
    expect(hydrateToolsState(JSON.stringify({ preset: "valid", retry: "{}" })).retry).toBe("");
    expect(hydrateToolsState(JSON.stringify({ turn: "2" })).turn).toBe("2");
    expect(hydrateToolsState(JSON.stringify({ turn: 2 })).turn).toBe("1");
  });

  it("caps free text and drops unknown keys", () => {
    const state = hydrateToolsState(JSON.stringify({ draft: "x".repeat(10 * DRAFT_MAX), retry: "y", mystery: 1 }));
    expect((state.draft as string).length).toBe(DRAFT_MAX);
    expect(state).not.toHaveProperty("mystery");
    expect(hydrateToolsState("[1, 2]")).toEqual(initialState);
    expect(hydrateToolsState("not json")).toEqual(initialState);
  });
});
