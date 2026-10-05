import { describe, expect, it } from "vitest";
import definition, { hydrateDiffusionState } from "./module";

describe("diffusion-vaes state", () => {
  it("reads a version-1 payload, keeping its step, latent point and chosen view", () => {
    const v1 = JSON.stringify({ step: 62, latentX: 0.3, latentY: -0.25, mode: "vae" });
    const state = hydrateDiffusionState(v1);
    expect(state.step).toBe(62);
    expect(state.latentX).toBe(0.3);
    expect(state.latentY).toBe(-0.25);
    expect(state.mode).toBe("vae");
    expect(state.denoiser).toBe("network");
    expect(state.start).toBe("noise");
  });

  it("clamps out-of-range values and rejects unknown options", () => {
    const state = hydrateDiffusionState(
      JSON.stringify({ step: 400, latentX: -9, seed: 0.4, denoiser: "magic", sample: "eight" }),
    );
    expect(state.step).toBe(100);
    expect(state.latentX).toBe(-4);
    expect(state.seed).toBe(1);
    expect(state.denoiser).toBe("network");
    expect(state.sample).toBe("three");
  });

  it("keeps every stored choice that is valid, so a saved Ideal denoiser or Noised x₀ run still opens", () => {
    const state = hydrateDiffusionState(JSON.stringify({ denoiser: "ideal", start: "digit", seed: 12, sample: "seven" }));
    expect(state.denoiser).toBe("ideal");
    expect(state.start).toBe("digit");
    expect(state.seed).toBe(12);
    expect(state.sample).toBe("seven");
  });

  it("lets the lesson step choose the view until the learner picks one", () => {
    expect(definition.initialState.mode).toBeUndefined();
    expect(hydrateDiffusionState("not json")).toEqual(definition.initialState);
    expect(definition.stateVersion).toBe(2);
  });
});
