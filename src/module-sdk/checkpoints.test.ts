import { describe, expect, it } from "vitest";
import { checkpointQuestions, shuffleCheckpoint } from "./checkpoints";
import type { Checkpoint } from "./types";

const question: Checkpoint = {
  prompt: "Which one?",
  options: ["alpha", "beta", "gamma", "delta"],
  answer: 2,
  explanation: "Because gamma is the third letter, which is the answer here.",
};

describe("checkpointQuestions", () => {
  it("wraps a single legacy question and passes a list through", () => {
    expect(checkpointQuestions({ checkpoint: question })).toEqual([question]);
    expect(checkpointQuestions({ checkpoint: [question, question] })).toHaveLength(2);
  });
});

describe("shuffleCheckpoint", () => {
  it("keeps the same options and points `answer` at the same text", () => {
    for (let seed = 0; seed < 50; seed += 1) {
      const shuffled = shuffleCheckpoint(question, `module:${seed}`);
      expect([...shuffled.options].sort()).toEqual([...question.options].sort());
      expect(shuffled.options[shuffled.answer]).toBe("gamma");
    }
  });

  it("is deterministic for a seed, so reopening a module shows the same order", () => {
    expect(shuffleCheckpoint(question, "module-x:0")).toEqual(shuffleCheckpoint(question, "module-x:0"));
  });

  it("does not leave the correct option in the position the author wrote it", () => {
    const positions = new Set<number>();
    for (let seed = 0; seed < 60; seed += 1) positions.add(shuffleCheckpoint(question, `m:${seed}`).answer);
    expect(positions.size).toBe(question.options.length);
  });

  it("never mutates the source question", () => {
    const before = JSON.stringify(question);
    shuffleCheckpoint(question, "anything");
    expect(JSON.stringify(question)).toBe(before);
  });
});
