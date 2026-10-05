import { invoke } from "@tauri-apps/api/core";

export type LocalAiState =
  | "notInstalled"
  | "installing"
  | "ready"
  | "running"
  | "error";

export interface LocalAiStatus {
  state: LocalAiState;
  engine: string;
  engineVersion: string;
  platform: string;
  modelId: string;
  modelName: string;
  modelBytes: number;
  modelSha256: string;
  engineInstalled: boolean;
  modelInstalled: boolean;
  serverRunning: boolean;
  port: number | null;
  error: string | null;
}

export interface LocalAiProgress {
  stage:
    | "prepare"
    | "engine"
    | "model"
    | "verify"
    | "extract"
    | "launch"
    | "ready"
    | "cancel"
    | "error";
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  message: string;
}

export interface LocalChatChunk {
  requestId: string;
  content: string;
  done: boolean;
}

export const localAiAvailable = () => Boolean(window.__TAURI_INTERNALS__);

export async function getLocalAiStatus() {
  return invoke<LocalAiStatus>("local_ai_status");
}

/**
 * Binary units: the divisors are powers of 1024, so the labels are GiB, MiB and
 * KiB. (They used to read GB, which is 1,000,000,000 bytes and a different number.)
 */
export function formatBytes(bytes: number) {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(2)} GiB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MiB`;
  if (bytes >= 1_024) return `${(bytes / 1_024).toFixed(1)} KiB`;
  return `${bytes} B`;
}
