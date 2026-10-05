import { invoke } from "@tauri-apps/api/core";
import {
  AlertTriangle,
  Globe,
  KeyRound,
  LoaderCircle,
  PlugZap,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  catalogModels,
  cloudProviders,
  CUSTOM_MODEL_ID,
  defaultModelFor,
  isCloudProvider,
  providerMeta,
  resolveModelId,
  selectionForModel,
  testProviderInBrowser,
  type CloudProviderId,
  type ProviderConfig,
  type ProviderTestResult,
} from "@app/lib/cloud-providers";
import { useAppStore } from "@app/store/app-store";
import { SettingsSelect } from "./SettingsSelect";

const desktop = () => Boolean(window.__TAURI_INTERNALS__);

export function CloudProviderSettings() {
  const store = useAppStore();
  const initialProvider = store.provider !== "none" && isCloudProvider(store.provider) ? store.provider : "openai";
  const initialSelection = selectionForModel(initialProvider, store.providerModel);
  const [providerChoice, setProviderChoice] = useState<CloudProviderId>(initialProvider);
  const [modelChoice, setModelChoice] = useState(initialSelection.choice || CUSTOM_MODEL_ID);
  const [customModel, setCustomModel] = useState(initialSelection.custom);
  const [baseUrl, setBaseUrl] = useState(store.providerBaseUrl);
  const [apiKey, setApiKey] = useState("");
  const [keyStored, setKeyStored] = useState(false);
  const [liveModels, setLiveModels] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "error">("idle");
  const [busy, setBusy] = useState<"test" | "save" | "remove" | null>(null);

  const provider = providerMeta(providerChoice);
  const modelId = resolveModelId(modelChoice, customModel, providerChoice);
  const models = useMemo(
    () => catalogModels(providerChoice, liveModels),
    [providerChoice, liveModels],
  );
  const connected = store.provider === providerChoice && keyStored;

  const applyConfig = (next: CloudProviderId, model: string, nextBaseUrl: string) => {
    const selection = selectionForModel(next, model);
    setProviderChoice(next);
    setModelChoice(selection.choice || CUSTOM_MODEL_ID);
    setCustomModel(selection.custom);
    setBaseUrl(nextBaseUrl);
  };

  useEffect(() => {
    if (!desktop()) return;
    let cancelled = false;
    void Promise.all([
      invoke<boolean>("has_provider_secret", { provider: providerChoice }).catch(() => false),
      invoke<ProviderConfig | null>("get_provider_config", { provider: providerChoice }).catch(() => null),
    ]).then(([stored, config]) => {
      if (cancelled) return;
      setKeyStored(Boolean(stored));
      if (config?.model || config?.baseUrl) {
        applyConfig(providerChoice, config.model ?? store.providerModel, config.baseUrl ?? baseUrl);
      }
    });
    return () => {
      cancelled = true;
    };
    // Reload stored config when the selected provider changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerChoice]);

  const report = (result: ProviderTestResult | { ok: boolean; message: string }) => {
    setStatus(result.message);
    setStatusKind(result.ok ? "ok" : "error");
    if ("models" in result && result.models.length) setLiveModels(result.models);
  };

  const persistPreference = (enabled: boolean) => {
    store.setProvider(enabled ? providerChoice : "none");
    store.setProviderModel(enabled ? modelId : "");
    store.setProviderBaseUrl(providerChoice === "compatible" ? baseUrl.trim() : "");
    if (!desktop()) return;
    void invoke("upsert_provider_config", {
      config: {
        provider: providerChoice,
        baseUrl: providerChoice === "compatible" ? baseUrl.trim() || null : null,
        model: modelId || null,
        enabled,
      },
    }).catch(() => undefined);
  };

  const changeProvider = (next: CloudProviderId) => {
    setLiveModels([]);
    setStatus("");
    setStatusKind("idle");
    if (next === providerChoice) return;
    applyConfig(next, defaultModelFor(next), next === "compatible" ? baseUrl : "");
  };

  const testConnection = async () => {
    if (provider.needsBaseUrl && !baseUrl.trim()) {
      report({ ok: false, message: "Enter the OpenAI-compatible base URL." });
      return;
    }
    if (!apiKey.trim() && !keyStored) {
      report({ ok: false, message: "Enter a key first." });
      return;
    }
    if (modelChoice === CUSTOM_MODEL_ID && !customModel.trim() && providerChoice !== "compatible") {
      report({ ok: false, message: "Enter a model ID, or pick one from the list." });
      return;
    }
    setBusy("test");
    setStatus("Checking the key with a models request…");
    setStatusKind("idle");
    const request = {
      provider: providerChoice,
      secret: apiKey.trim() || null,
      model: modelId || null,
      baseUrl: provider.needsBaseUrl ? baseUrl.trim() : null,
    };
    try {
      const result = desktop()
        ? await invoke<ProviderTestResult>("test_provider_connection", { request })
        : await testProviderInBrowser(request);
      report(result);
    } catch (error) {
      report({ ok: false, message: String(error) });
    } finally {
      setBusy(null);
    }
  };

  const saveProvider = async () => {
    if (!apiKey.trim() && !keyStored) {
      report({ ok: false, message: "Enter a key first." });
      return;
    }
    if (provider.needsBaseUrl && !baseUrl.trim()) {
      report({ ok: false, message: "Enter the OpenAI-compatible base URL." });
      return;
    }
    if (!modelId) {
      report({ ok: false, message: "Choose a model or enter a model ID." });
      return;
    }
    if (!desktop()) {
      store.setProviderModel(modelId);
      store.setProviderBaseUrl(provider.needsBaseUrl ? baseUrl.trim() : "");
      report({ ok: false, message: "The model is saved here. Keys can only be stored in the signed desktop app." });
      return;
    }
    setBusy("save");
    setStatus("Writing the key to the OS keychain…");
    setStatusKind("idle");
    try {
      if (apiKey.trim()) {
        await invoke("set_provider_secret", { provider: providerChoice, secret: apiKey.trim() });
        setApiKey("");
        setKeyStored(true);
      }
      persistPreference(true);
      setStatus("Stored in the OS keychain. Connection ready.");
      setStatusKind("ok");
    } catch (error) {
      report({ ok: false, message: `Could not store the key: ${String(error)}` });
    } finally {
      setBusy(null);
    }
  };

  const removeProvider = async () => {
    if (!window.confirm(`Remove the stored ${provider.name} key from this device?`)) return;
    setBusy("remove");
    try {
      if (desktop()) {
        await invoke("delete_provider_secret", { provider: providerChoice });
      }
      setKeyStored(false);
      persistPreference(false);
      setStatus("Cloud provider disconnected.");
      setStatusKind("ok");
    } catch (error) {
      report({ ok: false, message: String(error) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="cloud-provider-settings">
      <div className={`provider-connection ${connected ? "is-ready" : ""}`}>
        <i className={connected ? "is-ready" : ""} />
        <div>
          <strong>{connected ? "Cloud guide ready" : "No cloud provider connected"}</strong>
          <span>
            {connected
              ? `${provider.name} · ${modelId || "model unset"}`
              : "Pick a provider, choose a model, then test the key before storing it."}
          </span>
        </div>
        {connected && <b>keychain</b>}
      </div>

      <div className="provider-form">
        <div className="settings-field">
          <span id="cloud-provider-label">Provider</span>
          <SettingsSelect
            labelledBy="cloud-provider-label"
            value={providerChoice}
            onChange={changeProvider}
            options={cloudProviders.map((item) => ({
              value: item.id,
              label: item.name,
              description: item.note,
            }))}
          />
        </div>

        {provider.models.length > 0 && (
          <div className="settings-field">
            <span id="cloud-model-label">Model</span>
            <SettingsSelect
              labelledBy="cloud-model-label"
              value={models.some((model) => model.id === modelChoice) ? modelChoice : CUSTOM_MODEL_ID}
              onChange={setModelChoice}
              options={models.map((model) => ({
                value: model.id,
                label: model.name,
                description: model.note,
              }))}
            />
          </div>
        )}

        {(modelChoice === CUSTOM_MODEL_ID || provider.models.length === 0) && (
          <label className="settings-field">
            <span>Model ID</span>
            <div>
              <Sparkles />
              <input
                value={customModel}
                autoComplete="off"
                spellCheck={false}
                placeholder={providerChoice === "compatible" ? "llama3.1" : provider.defaultModel || "model-id"}
                onChange={(event) => setCustomModel(event.target.value)}
              />
            </div>
          </label>
        )}

        {provider.needsBaseUrl && (
          <label className="settings-field">
            <span>Base URL</span>
            <div>
              <Globe />
              <input
                value={baseUrl}
                autoComplete="off"
                spellCheck={false}
                placeholder="https://api.together.xyz/v1"
                onChange={(event) => setBaseUrl(event.target.value)}
              />
            </div>
          </label>
        )}

        <label className="settings-field">
          <span>API key</span>
          <div>
            <KeyRound />
            <input
              type="password"
              value={apiKey}
              autoComplete="off"
              placeholder={keyStored ? "Key stored · enter a new one to replace it" : provider.keyPlaceholder}
              onChange={(event) => setApiKey(event.target.value)}
            />
          </div>
        </label>

        <div className="provider-actions">
          <button type="button" className="settings-action settings-action--secondary" disabled={busy !== null} onClick={() => void testConnection()}>
            {busy === "test" ? <LoaderCircle className="is-spinning" /> : <PlugZap />}
            Test connection
          </button>
          <button type="button" className="settings-action" disabled={busy !== null} onClick={() => void saveProvider()}>
            {busy === "save" ? <LoaderCircle className="is-spinning" /> : <ShieldCheck />}
            Store securely
          </button>
          {keyStored && (
            <button type="button" className="settings-action settings-action--danger" disabled={busy !== null} onClick={() => void removeProvider()}>
              <Trash2 /> Remove key
            </button>
          )}
        </div>

        {status && (
          <div className={`provider-status ${statusKind === "error" ? "is-error" : statusKind === "ok" ? "is-ok" : ""}`} role="status">
            {statusKind === "error" ? <AlertTriangle /> : <ShieldCheck />}
            <span>{status}</span>
          </div>
        )}
      </div>
    </div>
  );
}
