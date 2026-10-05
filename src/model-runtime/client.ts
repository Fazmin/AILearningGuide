import type {
  ModelRunRequest,
  ModelRunResult,
  TransformerRunRequest,
  TransformerRunResult,
  TransformerWorkerResponse,
  WireTensor,
} from "./types";

type PendingRequest =
  | {
      kind: "transformer";
      resolve: (result: TransformerRunResult) => void;
      reject: (error: Error) => void;
    }
  | {
      kind: "model";
      resolve: (result: ModelRunResult) => void;
      reject: (error: Error) => void;
    };

let runtimeWorker: Worker | undefined;
let nextRequestId = 1;
const pendingRequests = new Map<number, PendingRequest>();

function getWorker() {
  if (runtimeWorker) return runtimeWorker;
  if (typeof Worker === "undefined") {
    throw new Error("Web Workers are unavailable in this environment.");
  }

  runtimeWorker = new Worker(new URL("./model.worker.ts", import.meta.url), {
    type: "module",
    name: "teaching-model-runtime",
  });
  runtimeWorker.onmessage = (
    event: MessageEvent<TransformerWorkerResponse>,
  ) => {
    const response = event.data;
    const pending = pendingRequests.get(response.id);
    if (!pending) return;
    pendingRequests.delete(response.id);

    if (!response.ok) {
      pending.reject(new Error(response.error));
      return;
    }

    if (
      response.type === "transformer-result" &&
      pending.kind === "transformer"
    ) {
      const { id: _id, ok: _ok, type: _type, ...result } = response;
      pending.resolve(result);
      return;
    }
    if (response.type === "model-result" && pending.kind === "model") {
      const { id: _id, ok: _ok, type: _type, ...result } = response;
      pending.resolve(result);
      return;
    }
    pending.reject(new Error("The model worker returned an unexpected result."));
  };
  runtimeWorker.onerror = (event) => {
    const error = new Error(event.message || "The model worker stopped.");
    pendingRequests.forEach(({ reject }) => reject(error));
    pendingRequests.clear();
    runtimeWorker?.terminate();
    runtimeWorker = undefined;
  };

  return runtimeWorker;
}

export function runTeachingTransformer(
  modelUrl: string,
  inputIds: number[],
): Promise<TransformerRunResult> {
  if (inputIds.length === 0) {
    return Promise.reject(new Error("The model needs at least one token."));
  }

  const id = nextRequestId++;
  const request: TransformerRunRequest = {
    id,
    type: "run-transformer",
    modelUrl,
    inputIds,
  };

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { kind: "transformer", resolve, reject });
    try {
      getWorker().postMessage(request);
    } catch (error) {
      pendingRequests.delete(id);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

export function runTeachingModel(
  modelUrl: string,
  feeds: Record<string, WireTensor>,
): Promise<ModelRunResult> {
  const id = nextRequestId++;
  const request: ModelRunRequest = {
    id,
    type: "run-model",
    modelUrl,
    feeds,
  };

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { kind: "model", resolve, reject });
    try {
      getWorker().postMessage(request);
    } catch (error) {
      pendingRequests.delete(id);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
