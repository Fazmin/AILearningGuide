export type CloudProviderId = "openai" | "anthropic" | "google" | "compatible";

export const CUSTOM_MODEL_ID = "custom";

export interface CloudModelOption {
  id: string;
  name: string;
  note: string;
}

export interface CloudProviderOption {
  id: CloudProviderId;
  name: string;
  note: string;
  keyPlaceholder: string;
  needsBaseUrl: boolean;
  defaultModel: string;
  models: CloudModelOption[];
}

export interface ProviderTestRequest {
  provider: CloudProviderId;
  secret?: string | null;
  model?: string | null;
  baseUrl?: string | null;
}

export interface ProviderTestResult {
  ok: boolean;
  message: string;
  model: string | null;
  models: string[];
}

export interface ProviderConfig {
  provider: string;
  baseUrl: string | null;
  model: string | null;
  enabled: boolean;
  updatedAt?: string | null;
}

const SKIP_MODEL = /embedding|whisper|tts|audio|realtime|transcribe|image|dall-e|davinci|babbage|moderation|search|omni/i;

export const cloudProviders: CloudProviderOption[] = [
  {
    id: "openai",
    name: "OpenAI",
    note: "GPT models through api.openai.com",
    keyPlaceholder: "sk-…",
    needsBaseUrl: false,
    defaultModel: "gpt-5.6-terra",
    models: [
      { id: "gpt-6-astra", name: "GPT-6 Astra", note: "Flagship reasoning and coding" },
      { id: "gpt-5.6-sol", name: "GPT-5.6 Sol", note: "Complex professional work" },
      { id: "gpt-5.6-terra", name: "GPT-5.6 Terra", note: "Balanced intelligence and cost" },
      { id: "gpt-5.6-luna", name: "GPT-5.6 Luna", note: "Fast, cost-sensitive work" },
    ],
  },
  {
    id: "anthropic",
    name: "Anthropic",
    note: "Claude through the Anthropic API",
    keyPlaceholder: "sk-ant-…",
    needsBaseUrl: false,
    defaultModel: "claude-sonnet-5",
    models: [
      { id: "claude-fable-5-1", name: "Claude Fable 5.1", note: "Long-horizon reasoning" },
      { id: "claude-opus-5", name: "Claude Opus 5", note: "Complex agentic work" },
      { id: "claude-sonnet-5", name: "Claude Sonnet 5", note: "Speed and intelligence" },
      { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", note: "Fastest Claude" },
    ],
  },
  {
    id: "google",
    name: "Google",
    note: "Gemini through Google AI Studio",
    keyPlaceholder: "AIza…",
    needsBaseUrl: false,
    defaultModel: "gemini-3.8-flash",
    models: [
      { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", note: "Current Flash production model" },
      { id: "gemini-3.7-flash", name: "Gemini 3.7 Flash", note: "Previous Flash generation" },
      { id: "gemini-3.1-pro-preview", name: "Gemini 3.1 Pro", note: "Deeper reasoning preview" },
      { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", note: "Stable 2.5 reasoning model" },
    ],
  },
  {
    id: "compatible",
    name: "OpenAI-compatible",
    note: "Any /v1 endpoint: Groq, Together, Ollama, or a local server",
    keyPlaceholder: "Provider token",
    needsBaseUrl: true,
    defaultModel: "",
    models: [],
  },
];

export const customModelOption: CloudModelOption = {
  id: CUSTOM_MODEL_ID,
  name: "Custom model ID",
  note: "Paste the exact API model string",
};

export function isCloudProvider(value: string): value is CloudProviderId {
  return cloudProviders.some((provider) => provider.id === value);
}

export function providerMeta(id: CloudProviderId) {
  return cloudProviders.find((provider) => provider.id === id) ?? cloudProviders[0];
}

export function defaultModelFor(id: CloudProviderId) {
  return providerMeta(id).defaultModel;
}

export function catalogModels(id: CloudProviderId, extraIds: string[] = []): CloudModelOption[] {
  const provider = providerMeta(id);
  const extras = extraIds
    .filter((modelId) => keepChatModel(modelId))
    .filter((modelId) => !provider.models.some((model) => model.id === modelId))
    .slice(0, 24)
    .map((modelId) => ({ id: modelId, name: modelId, note: "From your account" }));
  return [...provider.models, ...extras, customModelOption];
}

export function keepChatModel(id: string) {
  return Boolean(id.trim()) && !SKIP_MODEL.test(id);
}

export function resolveModelId(choice: string, customId: string, provider: CloudProviderId) {
  if (choice === CUSTOM_MODEL_ID) return customId.trim();
  return choice.trim() || defaultModelFor(provider);
}

export function selectionForModel(provider: CloudProviderId, modelId: string) {
  const catalog = providerMeta(provider).models;
  if (!modelId.trim()) {
    return { choice: defaultModelFor(provider) || CUSTOM_MODEL_ID, custom: "" };
  }
  if (catalog.some((model) => model.id === modelId)) {
    return { choice: modelId, custom: "" };
  }
  return { choice: CUSTOM_MODEL_ID, custom: modelId };
}

export function compatibleModelsUrl(baseUrl: string) {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  return trimmed.endsWith("/models") ? trimmed : `${trimmed}/models`;
}

export function providerModelsUrl(provider: CloudProviderId, baseUrl?: string | null) {
  switch (provider) {
    case "openai":
      return "https://api.openai.com/v1/models";
    case "anthropic":
      return "https://api.anthropic.com/v1/models";
    case "google":
      return "https://generativelanguage.googleapis.com/v1beta/models";
    case "compatible":
      return compatibleModelsUrl(baseUrl ?? "");
  }
}

export function providerAuthHeaders(provider: CloudProviderId, secret: string): Record<string, string> {
  switch (provider) {
    case "openai":
    case "compatible":
      return { Authorization: `Bearer ${secret}` };
    case "anthropic":
      return { "x-api-key": secret, "anthropic-version": "2023-06-01" };
    case "google":
      return { "x-goog-api-key": secret };
  }
}

export function parseProviderModels(provider: CloudProviderId, payload: unknown) {
  if (!payload || typeof payload !== "object") return [];
  const record = payload as Record<string, unknown>;
  const rows = Array.isArray(record.data)
    ? record.data
    : Array.isArray(record.models)
      ? record.models
      : [];
  return rows.flatMap((row) => {
    if (!row || typeof row !== "object") return [];
    const item = row as Record<string, unknown>;
    const raw = typeof item.id === "string" ? item.id : typeof item.name === "string" ? item.name : "";
    const id = raw.replace(/^models\//, "").trim();
    return keepChatModel(id) ? [id] : [];
  });
}

export function redactSecret(text: string, secret: string) {
  if (!secret) return text;
  return text.split(secret).join("•••");
}

export function interpretProviderStatus(
  status: number,
  models: string[],
  selectedModel?: string | null,
): ProviderTestResult {
  const model = selectedModel?.trim() || null;
  if (status === 401 || status === 403) {
    return { ok: false, message: "The provider rejected this key.", model, models: [] };
  }
  if (status === 429) {
    return {
      ok: true,
      message: "The key was accepted, but the provider is rate-limiting right now.",
      model,
      models,
    };
  }
  if (status === 404) {
    return { ok: false, message: "The models endpoint was not found. Check the base URL.", model, models: [] };
  }
  if (status < 200 || status >= 300) {
    return { ok: false, message: `The provider returned HTTP ${status}.`, model, models: [] };
  }
  if (model && models.length > 0 && !models.includes(model)) {
    return {
      ok: true,
      message: `The key works, but this account cannot see ${model}. Choose another model.`,
      model,
      models,
    };
  }
  if (model) {
    return { ok: true, message: `Connection ready · ${model}`, model, models };
  }
  return { ok: true, message: "Connection ready. The key can list models.", model, models };
}

export async function testProviderInBrowser(request: ProviderTestRequest): Promise<ProviderTestResult> {
  const secret = request.secret?.trim() ?? "";
  if (!secret) {
    return { ok: false, message: "Enter a key, or store one in the desktop app first.", model: null, models: [] };
  }
  if (request.provider === "compatible" && !request.baseUrl?.trim()) {
    return { ok: false, message: "Enter the OpenAI-compatible base URL.", model: null, models: [] };
  }
  const url = providerModelsUrl(request.provider, request.baseUrl);
  try {
    const response = await fetch(url, { headers: providerAuthHeaders(request.provider, secret) });
    const payload = await response.json().catch(() => null);
    const models = parseProviderModels(request.provider, payload);
    return interpretProviderStatus(response.status, models, request.model);
  } catch (error) {
    const raw = redactSecret(error instanceof Error ? error.message : String(error), secret);
    if (/failed to fetch|networkerror|cors|load failed/i.test(raw)) {
      return {
        ok: false,
        message: "The browser preview cannot reach this provider. Open the desktop app to test the key.",
        model: request.model ?? null,
        models: [],
      };
    }
    return { ok: false, message: raw, model: request.model ?? null, models: [] };
  }
}
