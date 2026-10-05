import { describe, expect, it } from "vitest";
import { hydrateAgentsState } from "./module";
import { correctArguments, countTokens, runAgent, runTool } from "./loop";

describe("the agent loop", () => {
  it("answers the hours task in two calls with one tool run", () => {
    const run = runAgent("hours", 4, false);
    expect(run.calls).toBe(2);
    expect(run.stop).toBe("final answer");
    expect(run.events.filter((event) => event.kind === "tool")).toHaveLength(1);
  });

  it("runs the requested tool even when the cap forbids a second call", () => {
    const run = runAgent("hours", 1, false);
    expect(run.stop).toBe("step cap");
    expect(run.events.map((event) => event.kind)).toEqual(["model", "tool", "stop"]);
  });

  it("recovers from one timeout and is not stopped by the repeat rule", () => {
    for (const guard of [false, true]) {
      const run = runAgent("recover", 4, guard);
      expect(run.calls).toBe(3);
      expect(run.stop).toBe("final answer");
      const tools = run.events.filter((event) => event.kind === "tool");
      expect(tools.map((event) => event.kind === "tool" && event.failed)).toEqual([true, false]);
    }
  });

  it("retries forever until the cap, and the stop rule ends it after two identical failures", () => {
    expect(runAgent("loop", 10, false).calls).toBe(10);
    expect(runAgent("loop", 10, false).stop).toBe("step cap");
    const guarded = runAgent("loop", 10, true);
    expect(guarded.calls).toBe(2);
    expect(guarded.stop).toBe("host stop rule");
  });

  it("re-reads a growing context, so tokens grow faster than calls", () => {
    const four = runAgent("loop", 4, false);
    const ten = runAgent("loop", 10, false);
    const inputs = ten.events.flatMap((event) => (event.kind === "model" ? [event.inputTokens] : []));
    const steps = inputs.slice(1).map((value, index) => value - inputs[index]);
    expect(new Set(steps).size).toBe(1);
    expect(steps[0]).toBe(31);
    expect(inputs[0]).toBe(58);
    expect(inputs[9]).toBe(337);
    expect(four.inputTokens).toBe(418);
    expect(ten.inputTokens).toBe(1975);
    expect(ten.inputTokens / four.inputTokens).toBeGreaterThan(10 / 4);
  });

  it("counts toy tokens as letter runs, digit runs and symbols", () => {
    expect(countTokens('{"a": 12}')).toBe(7);
  });
});

describe("the repair task", () => {
  const calls = (run: ReturnType<typeof runAgent>) =>
    run.events.flatMap((event) => (event.kind === "model" ? [event.text] : []));

  it("reads the error, changes the argument, and answers on the third call", () => {
    const run = runAgent("repair", 4, false);
    expect(run.calls).toBe(3);
    expect(run.stop).toBe("final answer");
    const tools = run.events.flatMap((event) => (event.kind === "tool" ? [event] : []));
    expect(tools.map((event) => event.failed)).toEqual([true, false]);
    expect(tools[0].text).toBe(
      '{"error": "invalid arguments", "details": ["/day: must be one of \\"weekday\\", \\"sunday\\"; got \\"Weekday\\""]}',
    );
    expect(tools[1].text).toBe('{"dock": "south", "day": "weekday", "closes": "21:00"}');
    expect(calls(run)).toEqual([
      '{"name": "get_dock_hours", "arguments": {"dock": "south", "day": "Weekday"}}',
      '{"name": "get_dock_hours", "arguments": {"dock": "south", "day": "weekday"}}',
      "The south dock closes at 21:00 on weekdays.",
    ]);
  });

  it("is not ended by the identical-failure rule, because the second call differs", () => {
    const run = runAgent("repair", 4, true);
    expect(run.stop).toBe("final answer");
    expect(run.calls).toBe(3);
    expect(new Set(calls(run).slice(0, 2)).size).toBe(2);
    expect(runAgent("loop", 10, true).stop).toBe("host stop rule");
  });

  it("is cut off by the cap before it can answer, like any other task", () => {
    const run = runAgent("repair", 2, false);
    expect(run.stop).toBe("step cap");
    expect(run.events.map((event) => event.kind)).toEqual(["model", "tool", "model", "tool", "stop"]);
  });

  it("pins the figures the lesson quotes", () => {
    const run = runAgent("repair", 4, false);
    expect(run.calls).toBe(3);
    expect(run.inputTokens).toBe(391);
    expect(run.outputTokens).toBe(81);
    expect(run.events.flatMap((event) => (event.kind === "model" ? [event.inputTokens] : []))).toEqual([57, 136, 198]);
    expect(run.events.filter((event) => event.kind === "tool")).toHaveLength(2);
  });

  it("corrects only what the error text names, and leaves anything unreadable alone", () => {
    const call = '{"name": "get_dock_hours", "arguments": {"dock": "South ", "day": "Sunday"}}';
    const error = runTool(call, 1);
    expect(correctArguments(call, error)).toBe('{"name": "get_dock_hours", "arguments": {"dock": "south", "day": "sunday"}}');
    expect(correctArguments(call, '{"error": "timeout after 5 s"}')).toBe(call);
    expect(correctArguments(call, "not json")).toBe(call);
    expect(correctArguments("not json", error)).toBe("not json");
    const missing = '{"name": "get_dock_hours", "arguments": {"day": "sunday"}}';
    expect(correctArguments(missing, runTool(missing, 1))).toBe(missing);
  });

  it("keeps the other tasks' numbers and the dock lookup unchanged", () => {
    expect(runTool('{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}', 1)).toBe(
      '{"dock": "north", "day": "sunday", "closes": "16:00"}',
    );
    const hours = runAgent("hours", 4, false);
    expect(hours.events.flatMap((event) => (event.kind === "model" ? [event.inputTokens] : []))).toEqual([56, 118]);
    expect(runAgent("loop", 10, true).inputTokens).toBe(147);
    const recover = runAgent("recover", 4, false);
    expect(recover.events.filter((event) => event.kind === "tool" && event.failed)).toHaveLength(1);
  });
});

describe("state", () => {
  it("migrates version 1 and resets its tick", () => {
    const state = hydrateAgentsState(JSON.stringify({ task: "loop", stepCap: 8, tick: 3 }));
    expect(state.tick).toBe(99);
    expect(state.stepCap).toBe(8);
    expect(state.guard).toBe("off");
  });

  it("reads a version 2 payload unchanged and fills the Reliability controls", () => {
    const state = hydrateAgentsState(JSON.stringify({ task: "recover", stepCap: 6, tick: 2, guard: "on" }));
    expect(state).toEqual({
      task: "recover",
      stepCap: 6,
      tick: 2,
      guard: "on",
      stepSuccess: 95,
      runSteps: 20,
      gateAfter: 0,
      catchRate: 100,
    });
  });

  it("clamps every number and drops unknown keys", () => {
    for (const value of [1e9, -1e9]) {
      const state = hydrateAgentsState(
        JSON.stringify({ task: "repair", guard: "on", stepCap: value, tick: value, stepSuccess: value, runSteps: value, gateAfter: value, catchRate: value, extra: 1 }),
      );
      expect(state.stepCap).toBe(value > 0 ? 10 : 1);
      expect(state.tick).toBe(value > 0 ? 99 : 0);
      expect(state.stepSuccess).toBe(value > 0 ? 100 : 50);
      expect(state.runSteps).toBe(value > 0 ? 30 : 1);
      expect(state.gateAfter).toBe(value > 0 ? 30 : 0);
      expect(state.catchRate).toBe(value > 0 ? 100 : 0);
      expect(state).not.toHaveProperty("extra");
    }
    const infinite = hydrateAgentsState('{"guard":"off","stepSuccess":1e999,"runSteps":-1e999,"catchRate":47}');
    expect([infinite.stepSuccess, infinite.runSteps, infinite.catchRate]).toEqual([95, 20, 45]);
    expect(hydrateAgentsState(JSON.stringify({ task: "nonsense", guard: "off" })).task).toBe("hours");
    expect(hydrateAgentsState("[1]")).toEqual(hydrateAgentsState("{}"));
  });
});
