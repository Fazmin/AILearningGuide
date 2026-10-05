import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { shuffleCheckpoint } from "@app/module-sdk";
import type { Checkpoint as CheckpointQuestion } from "@app/module-sdk";
import type { RegisteredModule } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { Checkpoint } from "./Checkpoint";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const questions: CheckpointQuestion[] = [
  { prompt: "First question?", options: ["a1", "a2", "a3"], answer: 0, explanation: "Because the first option is right.", objective: 0 },
  { prompt: "Second question?", options: ["b1", "b2", "b3"], answer: 1, explanation: "Because the second option is right.", objective: 1 },
  { prompt: "Third question?", options: ["c1", "c2", "c3"], answer: 2, explanation: "Because the third option is right.", objective: 2 },
];

const fake = (checkpoint: CheckpointQuestion | CheckpointQuestion[]) =>
  ({ id: "module-test", checkpoint, objectives: ["x", "y", "z"] }) as unknown as RegisteredModule;

let container: HTMLDivElement;
let root: Root;

const mount = (module: RegisteredModule) =>
  act(() => {
    root.render(createElement(Checkpoint, { module }));
  });

/** Clicks the option showing `text` inside the nth question. */
const choose = (question: number, text: string) =>
  act(() => {
    const group = container.querySelectorAll(".checkpoint__question")[question];
    const button = Array.from(group.querySelectorAll("button")).find((item) => item.textContent?.includes(text));
    button!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

const correctText = (index: number) => {
  const shuffled = shuffleCheckpoint(questions[index], `module-test:${index}`);
  return shuffled.options[shuffled.answer];
};

beforeEach(() => {
  useAppStore.getState().resetLearningData();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("Checkpoint", () => {
  it("shows every question with its position", () => {
    mount(fake(questions));
    expect(container.querySelectorAll(".checkpoint__question")).toHaveLength(3);
    expect(container.textContent).toContain("Question 2 of 3");
    expect(container.textContent).toContain("3 questions before you move on");
  });

  it("keeps the old single-question presentation for a legacy module", () => {
    mount(fake(questions[0]));
    expect(container.querySelectorAll(".checkpoint__question")).toHaveLength(1);
    expect(container.textContent).toContain("One question before you move on");
    expect(container.textContent).not.toContain("Question 1 of");
    expect(container.querySelector(".checkpoint__summary")).toBeNull();
  });

  it("reveals the explanation and the right option after an answer", () => {
    mount(fake(questions));
    choose(0, correctText(0));
    expect(container.textContent).toContain("That’s it.");
    expect(container.textContent).toContain("Because the first option is right.");
    expect(container.querySelectorAll(".checkpoint__question")[0].querySelector(".is-correct")?.textContent).toContain(correctText(0));
  });

  it("scores the first answer only and summarises it", () => {
    mount(fake(questions));
    const wrong = shuffleCheckpoint(questions[0], "module-test:0").options.find((option) => option !== correctText(0))!;
    choose(0, wrong);
    choose(0, correctText(0));
    choose(1, correctText(1));
    expect(useAppStore.getState().progress["module-test"].checkpointResults).toEqual([false, true, null]);
    expect(container.querySelector(".checkpoint__summary")?.textContent).toContain("1 of 3 correct on the first try");
    expect(container.querySelector(".checkpoint__summary button")).toBeNull();
  });

  it("offers a retake once every question is answered and clears the results", () => {
    mount(fake(questions));
    choose(0, correctText(0));
    choose(1, correctText(1));
    choose(2, correctText(2));
    const retake = container.querySelector<HTMLButtonElement>(".checkpoint__summary button");
    expect(retake?.textContent).toContain("Try again");
    act(() => retake!.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(useAppStore.getState().progress["module-test"].checkpointResults).toEqual([]);
    expect(container.querySelector(".checkpoint__feedback")).toBeNull();
  });

  it("restores earlier results when the module is reopened", () => {
    useAppStore.getState().markCheckpoint("module-test", 0, true, 3);
    useAppStore.getState().markCheckpoint("module-test", 1, false, 3);
    mount(fake(questions));
    expect(container.textContent).toContain("You got this one on the first try.");
    expect(container.textContent).toContain("You missed this one the first time.");
    expect(container.querySelector(".checkpoint__summary")?.textContent).toContain("1 of 3 correct on the first try");
  });
});
