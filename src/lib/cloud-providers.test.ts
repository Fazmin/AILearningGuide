import { describe, expect, it } from "vitest";
import {
  catalogModels,
  compatibleModelsUrl,
  CUSTOM_MODEL_ID,
  interpretProviderStatus,
  parseProviderModels,
  providerModelsUrl,
  redactSecret,
  resolveModelId,
  selectionForModel,
} from "./cloud-providers";

describe("cloud provider helpers", () => {
  it("builds models URLs without double-appending /models", () => {
    expect(compatibleModelsUrl("https://api.together.xyz/v1/")).toBe("https://api.together.xyz/v1/models");
    expect(compatibleModelsUrl("http://127.0.0.1:11434/v1/models")).toBe("http://127.0.0.1:11434/v1/models");
    expect(providerModelsUrl("openai")).toBe("https://api.openai.com/v1/models");
    expect(providerModelsUrl("google")).toContain("generativelanguage.googleapis.com");
  });

  it("treats unknown model IDs as a custom selection", () => {
    expect(selectionForModel("openai", "gpt-5.6-terra")).toEqual({ choice: "gpt-5.6-terra", custom: "" });
    expect(selectionForModel("openai", "my-fine-tune")).toEqual({
      choice: CUSTOM_MODEL_ID,
      custom: "my-fine-tune",
    });
    expect(resolveModelId(CUSTOM_MODEL_ID, " my-fine-tune ", "openai")).toBe("my-fine-tune");
  });

  it("parses provider model lists and drops non-chat endpoints", () => {
    expect(
      parseProviderModels("openai", {
        data: [{ id: "gpt-5.6-terra" }, { id: "text-embedding-3-large" }, { id: "gpt-image-1" }],
      }),
    ).toEqual(["gpt-5.6-terra"]);
    expect(
      parseProviderModels("google", {
        models: [{ name: "models/gemini-3.8-flash" }, { name: "models/gemini-3.1-flash-image" }],
      }),
    ).toEqual(["gemini-3.8-flash"]);
  });

  it("keeps curated models first and appends a custom option", () => {
    const models = catalogModels("anthropic", ["claude-sonnet-5", "office-bot"]);
    expect(models[0]?.id).toBe("claude-fable-5-1");
    expect(models.some((model) => model.id === "office-bot" && model.note === "From your account")).toBe(true);
    expect(models.at(-1)?.id).toBe(CUSTOM_MODEL_ID);
  });

  it("interprets key tests without echoing secrets", () => {
    expect(interpretProviderStatus(401, [], "gpt-5.6-terra").ok).toBe(false);
    expect(interpretProviderStatus(429, [], "gpt-5.6-terra").ok).toBe(true);
    expect(interpretProviderStatus(200, ["gpt-5.6-luna"], "gpt-5.6-terra").message).toMatch(/cannot see/);
    expect(redactSecret("Bearer sk-secret exploded", "sk-secret")).toBe("Bearer ••• exploded");
  });
});
