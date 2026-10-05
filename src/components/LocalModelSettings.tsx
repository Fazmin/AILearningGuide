import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  Check,
  CircleStop,
  Cpu,
  Download,
  HardDrive,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  formatBytes,
  getLocalAiStatus,
  localAiAvailable,
  type LocalAiProgress,
  type LocalAiStatus,
} from "@app/lib/local-ai";
import { useAppStore } from "@app/store/app-store";

const MODEL_BYTES = 1_280_835_840;
const MODEL_HASH = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223";

const emptyStatus: LocalAiStatus = {
  state: "notInstalled",
  engine: "llama.cpp",
  engineVersion: "b10991",
  platform: `${navigator.platform || "desktop"}`,
  modelId: "qwen3.5-2b-q4-k-m",
  modelName: "Qwen3.5 2B · Q4_K_M",
  modelBytes: MODEL_BYTES,
  modelSha256: MODEL_HASH,
  engineInstalled: false,
  modelInstalled: false,
  serverRunning: false,
  port: null,
  error: null,
};

function statusLabel(status: LocalAiStatus) {
  if (status.serverRunning) return "Running locally";
  if (status.engineInstalled && status.modelInstalled) return "Installed and ready";
  if (status.state === "installing") return "Setting up";
  if (status.state === "error") return "Setup needs attention";
  return "Not installed";
}

export function LocalModelSettings() {
  const desktop = localAiAvailable();
  const setLocalModelReady = useAppStore((state) => state.setLocalModelReady);
  const [status, setStatus] = useState<LocalAiStatus>(emptyStatus);
  const [progress, setProgress] = useState<LocalAiProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const applyStatus = useCallback(
    (next: LocalAiStatus) => {
      setStatus(next);
      setLocalModelReady(next.engineInstalled && next.modelInstalled);
    },
    [setLocalModelReady],
  );

  const refresh = useCallback(async () => {
    if (!desktop) return;
    try {
      applyStatus(await getLocalAiStatus());
    } catch (error) {
      setMessage(`Could not read local model status: ${String(error)}`);
    }
  }, [applyStatus, desktop]);

  useEffect(() => {
    void refresh();
    if (!desktop) return;
    let disposed = false;
    let removeListener: (() => void) | undefined;
    void listen<LocalAiProgress>("local-ai-progress", ({ payload }) => {
      setProgress(payload);
      if (payload.stage === "error") setMessage(payload.message);
      if (payload.stage === "ready") setMessage("Private topic chat is ready.");
    }).then((unlisten) => {
      if (disposed) unlisten();
      else removeListener = unlisten;
    });
    const timer = window.setInterval(() => void refresh(), 4_000);
    return () => {
      disposed = true;
      removeListener?.();
      window.clearInterval(timer);
    };
  }, [desktop, refresh]);

  const runStatusCommand = async (command: "start_local_ai" | "stop_local_ai") => {
    if (!desktop) return;
    setBusy(true);
    setMessage("");
    try {
      applyStatus(await invoke<LocalAiStatus>(command));
    } catch (error) {
      setMessage(String(error));
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const install = async () => {
    if (!desktop) return;
    setBusy(true);
    setMessage("");
    setProgress({
      stage: "prepare",
      downloadedBytes: 0,
      totalBytes: 1,
      percent: 0,
      message: "Checking disk space and platform",
    });
    setStatus((current) => ({ ...current, state: "installing", error: null }));
    try {
      applyStatus(await invoke<LocalAiStatus>("install_local_ai"));
    } catch (error) {
      setMessage(String(error));
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (!desktop) return;
    setMessage("Cancelling safely…");
    try {
      await invoke("cancel_local_ai_install");
    } catch (error) {
      setMessage(String(error));
    }
  };

  const remove = async () => {
    if (!desktop || !window.confirm("Remove Qwen and the downloaded llama.cpp engine from this device?")) return;
    setBusy(true);
    setMessage("");
    try {
      applyStatus(await invoke<LocalAiStatus>("remove_local_ai"));
      setProgress(null);
      setMessage("Local engine and model removed.");
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const installed = status.engineInstalled && status.modelInstalled;
  const installing = status.state === "installing" || (busy && progress !== null && !installed);
  const progressTotal = progress?.totalBytes ?? 0;
  const progressDownloaded = progress?.downloadedBytes ?? 0;

  return (
    <div className="local-model-settings">
      <div className={`local-runtime-status local-runtime-status--${status.state}`}>
        <i className={status.serverRunning ? "is-running" : installed ? "is-ready" : ""} />
        <div>
          <strong>{statusLabel(status)}</strong>
          <span>{status.engine} {status.engineVersion} · {status.platform}</span>
        </div>
        {status.serverRunning && <b>127.0.0.1:{status.port}</b>}
      </div>

      <section className="local-model-product">
        <div className="local-model-product__heading">
          <div className="local-model-glyph"><Cpu /><i /><i /></div>
          <div>
            <span>Recommended local guide</span>
            <h3>Qwen3.5 2B</h3>
            <p>Q4_K_M quantization · {formatBytes(MODEL_BYTES)} · Apache 2.0</p>
          </div>
          {installed && <div className="verified-badge"><ShieldCheck /> verified</div>}
        </div>

        <div className="local-model-details">
          <div><Server /><span><strong>Engine</strong>llama.cpp {status.engineVersion}</span>{status.engineInstalled && <Check />}</div>
          <div><HardDrive /><span><strong>Storage</strong>About 1.5 GB during setup</span>{status.modelInstalled && <Check />}</div>
          <div><ShieldCheck /><span><strong>Integrity</strong><code>{MODEL_HASH.slice(0, 12)}…{MODEL_HASH.slice(-8)}</code></span>{status.modelInstalled && <Check />}</div>
        </div>

        {installing && progress && (
          <div className="local-download-progress" aria-live="polite">
            <div>
              <span>{progress.message}</span>
              <strong>{progress.percent.toFixed(0)}%</strong>
            </div>
            <i><b style={{ width: `${progress.percent}%` }} /></i>
            <div>
              <small>
                {progressTotal > 1
                  ? `${formatBytes(progressDownloaded)} of ${formatBytes(progressTotal)}`
                  : "Preparing secure download"}
              </small>
              <button type="button" onClick={() => void cancel()}><X /> Cancel</button>
            </div>
          </div>
        )}

        {!installed && !installing && (
          <div className="local-setup-action">
            <button type="button" disabled={!desktop || busy} onClick={() => void install()}>
              <Download /> Download and set up
            </button>
            <span>
              {desktop
                ? "One click installs the pinned engine and model, verifies both, then starts the private server."
                : "Open the Tauri desktop app to install. Browser previews cannot write or launch local binaries."}
            </span>
          </div>
        )}

        {installed && !installing && (
          <div className="local-runtime-actions">
            <button
              type="button"
              className="local-primary-action"
              disabled={busy}
              onClick={() => void runStatusCommand(status.serverRunning ? "stop_local_ai" : "start_local_ai")}
            >
              {status.serverRunning ? <CircleStop /> : <Play />}
              {status.serverRunning ? "Stop engine" : "Start engine"}
            </button>
            <button type="button" disabled={busy} onClick={() => void refresh()}><RefreshCw /> Check status</button>
            <button type="button" className="local-remove-action" disabled={busy} onClick={() => void remove()}><Trash2 /> Remove</button>
          </div>
        )}
      </section>

      {(message || status.error) && (
        <div className={status.state === "error" ? "local-model-message is-error" : "local-model-message"} role="status">
          {status.state === "error" ? <AlertTriangle /> : <ShieldCheck />}
          <span>{message || status.error}</span>
        </div>
      )}

      <div className="local-privacy-note">
        <ShieldCheck />
        <div><strong>Private by default</strong><span>Prompts stay on this device. The server binds only to 127.0.0.1 and stops when the app closes.</span></div>
      </div>
    </div>
  );
}
