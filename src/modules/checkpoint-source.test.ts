import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM build script, intentionally untyped
import { parseModuleCheckpoints } from "../../scripts/lib/search-documents.mjs";

type Parsed = {
  isList: boolean;
  questions: Array<{ prompt: string; options: string[]; answer?: number; explanation: string; objective?: number }>;
};
const parse = parseModuleCheckpoints as (source: string) => Parsed;

const single = `
  glossary: [],
  checkpoint: {
    prompt: "What happens to the loss?",
    options: ["It rises", "It falls", "It stays flat"],
    answer: 1,
    explanation: "Gradient descent moves downhill, so the loss falls."
  },
  initialState,
`;

const list = `
  glossary: [],
  checkpoint: [
    {
      prompt: "First, what is “x”?",
      options: ["One", "Two", "Three"],
      answer: 0,
      explanation: "Because x is one, the very first option is right.",
      objective: 0,
    },
    {
      prompt:
        "Second, wrapped onto the next line?",
      options: ["A", "B", "C", "D"],
      answer: 3,
      explanation: "The wrapped prompt still parses, so D is right.",
      objective: 1,
    },
    {
      prompt: "Third?",
      options: ["Yes", "No"],
      answer: 0,
      explanation: "A two-option question is allowed to parse.",
      objective: 1,
    },
  ],
  initialState,
`;

describe("parseModuleCheckpoints", () => {
  it("reads the legacy single-question object", () => {
    const parsed = parse(single);
    expect(parsed.isList).toBe(false);
    expect(parsed.questions).toHaveLength(1);
    expect(parsed.questions[0]).toMatchObject({
      prompt: "What happens to the loss?",
      options: ["It rises", "It falls", "It stays flat"],
      answer: 1,
    });
  });

  it("reads a list of questions with their answers and objectives", () => {
    const parsed = parse(list);
    expect(parsed.isList).toBe(true);
    expect(parsed.questions.map((question) => question.answer)).toEqual([0, 3, 0]);
    expect(parsed.questions.map((question) => question.objective)).toEqual([0, 1, 1]);
    expect(parsed.questions[1].prompt).toBe("Second, wrapped onto the next line?");
    expect(parsed.questions[1].options).toEqual(["A", "B", "C", "D"]);
    expect(parsed.questions[0].prompt).toBe("First, what is “x”?");
  });

  it("does not split a list on option arrays that end in a comma", () => {
    expect(parse(list).questions).toHaveLength(3);
  });

  it("reports no questions when there is no checkpoint", () => {
    expect(parse("const nothing = 1;").questions).toEqual([]);
  });
});
