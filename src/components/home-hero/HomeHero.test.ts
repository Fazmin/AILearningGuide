import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { HomeHero } from "./HomeHero";
import { heroForThisLaunch, heroVariants, pickHeroVariant, resetLaunchHero, type HeroVariant } from "./pick-hero";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("picking the home hero", () => {
  it("never shows the previous launch's hero again", () => {
    for (const previous of heroVariants) {
      for (let i = 0; i < 20; i += 1) expect(pickHeroVariant(previous, () => i / 20)).not.toBe(previous);
    }
  });

  it("can land on each of the other three", () => {
    const picks = new Set([0, 0.34, 0.67, 0.999].map((r) => pickHeroVariant("descent", () => r)));
    expect(picks).toEqual(new Set(["next-word", "attention", "meaning-map"]));
  });

  it("chooses from all four on a first launch, even if random() returns 1", () => {
    expect(pickHeroVariant(null, () => 0)).toBe("next-word");
    expect(pickHeroVariant(null, () => 1)).toBe("descent");
  });

  it("keeps one pick for the whole launch", () => {
    resetLaunchHero();
    window.history.replaceState(null, "", "/");
    useAppStore.setState({ lastHomeHero: "attention" });
    const first = heroForThisLaunch();
    expect(first).not.toBe("attention");
    useAppStore.setState({ lastHomeHero: first });
    expect(heroForThisLaunch()).toBe(first);
  });
});

const labFor: Record<HeroVariant, string> = {
  "next-word": "next-token-prediction",
  attention: "attention",
  "meaning-map": "tokens-embeddings",
  descent: "loss-gradient-descent",
};

describe.each(heroVariants)("the %s hero", (variant) => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    resetLaunchHero();
    window.history.replaceState(null, "", `/?hero=${variant}`);
    useAppStore.setState({ view: "home", activeModuleId: null, lastHomeHero: null });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => root.render(createElement(HomeHero)));
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    resetLaunchHero();
    window.history.replaceState(null, "", "/");
  });

  it("renders its still frame and remembers it was shown", () => {
    expect(container.querySelector(`.home-hero--${variant}`)).not.toBeNull();
    expect(container.querySelector(".home-hero__stage")?.childElementCount).toBeGreaterThan(0);
    expect(useAppStore.getState().lastHomeHero).toBe(variant);
  });

  it("opens the lab it previews", () => {
    const lab = modulesBySlug.get(labFor[variant])!;
    const button = container.querySelector<HTMLButtonElement>(".home-hero__lab")!;
    expect(button.textContent).toContain(lab.title);
    act(() => button.click());
    expect(useAppStore.getState()).toMatchObject({ view: "module", activeModuleId: lab.id });
  });
});
