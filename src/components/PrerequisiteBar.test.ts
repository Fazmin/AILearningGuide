import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { PrerequisiteBar } from "./PrerequisiteBar";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const mount = (slug: string) => {
  const module = modulesBySlug.get(slug)!;
  act(() => {
    root.render(createElement(PrerequisiteBar, { module }));
  });
  return module;
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

describe("PrerequisiteBar", () => {
  it("renders nothing for a lab that builds on nothing", () => {
    mount("what-ai-is");
    expect(container.innerHTML).toBe("");
  });

  it("names every lab this one builds on", () => {
    const module = mount("backpropagation");
    const titles = Array.from(container.querySelectorAll(".prereq-chip")).map((chip) => chip.textContent ?? "");
    expect(titles).toHaveLength(module.prerequisites.length);
    expect(titles[0]).toContain("Loss & gradient descent");
  });

  it("says in text, not only colour, how far the learner has got with each", () => {
    mount("backpropagation");
    expect(container.textContent).toContain("not started");
    act(() => {
      useAppStore.getState().openModule(modulesBySlug.get("loss-gradient-descent")!.id);
    });
    expect(container.textContent).toContain("started, not completed");
    act(() => {
      useAppStore.getState().toggleModuleComplete(modulesBySlug.get("loss-gradient-descent")!.id);
    });
    expect(container.textContent).toContain("completed");
    expect(container.querySelector(".prereq-chip.is-done")).not.toBeNull();
  });

  it("opens the prerequisite when its chip is pressed", () => {
    mount("backpropagation");
    const chip = container.querySelector<HTMLButtonElement>(".prereq-chip")!;
    act(() => chip.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(useAppStore.getState().view).toBe("module");
    expect(useAppStore.getState().activeModuleId).toBe(modulesBySlug.get("loss-gradient-descent")!.id);
  });
});
