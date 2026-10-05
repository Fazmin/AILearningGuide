export interface TransformerRunResult {
  provider: "webgpu" | "wasm";
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  attention: Float32Array;
  qkvShape: readonly number[];
  attentionShape: readonly number[];
}

export interface TransformerRunRequest {
  id: number;
  type: "run-transformer";
  modelUrl: string;
  inputIds: number[];
}

export interface WireTensor {
  type: "float32" | "int64";
  data: number[];
  dims: number[];
}

export interface ModelRunRequest {
  id: number;
  type: "run-model";
  modelUrl: string;
  feeds: Record<string, WireTensor>;
}

export interface ModelRunResult {
  provider: "webgpu" | "wasm";
  outputs: Record<string, WireTensor>;
}

export interface ModelRunSuccess extends ModelRunResult {
  id: number;
  ok: true;
  type: "model-result";
}

export interface TransformerRunSuccess extends TransformerRunResult {
  id: number;
  ok: true;
  type: "transformer-result";
}

export interface TransformerRunFailure {
  id: number;
  ok: false;
  type: "error";
  error: string;
}

export type TransformerWorkerResponse =
  | TransformerRunSuccess
  | ModelRunSuccess
  | TransformerRunFailure;

export type TeachingModelRequest = TransformerRunRequest | ModelRunRequest;
