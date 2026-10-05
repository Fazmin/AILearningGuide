/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/webgpu";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import type {
  ModelRunSuccess,
  TeachingModelRequest,
  TransformerRunSuccess,
  TransformerWorkerResponse,
  WireTensor,
} from "./types";

type RuntimeProvider = "webgpu" | "wasm";

const workerScope: DedicatedWorkerGlobalScope =
  self as unknown as DedicatedWorkerGlobalScope;
const sessions = new Map<string, Promise<ort.InferenceSession>>();
let providerOrderPromise: Promise<RuntimeProvider[]> | undefined;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = {
  wasm: new URL(wasmUrl, workerScope.location.href).href,
  mjs: new URL(wasmModuleUrl, workerScope.location.href).href,
};

async function providerOrder(): Promise<RuntimeProvider[]> {
  if (providerOrderPromise) return providerOrderPromise;
  providerOrderPromise = (async () => {
    const webGpu = (
      navigator as Navigator & {
        gpu?: { requestAdapter: () => Promise<unknown | null> };
      }
    ).gpu;
    let canUseWebGpu = false;
    if (webGpu) {
      try {
        canUseWebGpu = Boolean(await webGpu.requestAdapter());
      } catch {
        canUseWebGpu = false;
      }
    }
    return canUseWebGpu ? ["webgpu", "wasm"] : ["wasm"];
  })();
  return providerOrderPromise;
}

async function createSession(
  modelUrl: string,
  provider: RuntimeProvider,
): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: [provider],
    graphOptimizationLevel: "all",
  });
}

function sessionFor(modelUrl: string, provider: RuntimeProvider) {
  const key = `${provider}:${modelUrl}`;
  const existing = sessions.get(key);
  if (existing) return existing;

  const pending = createSession(modelUrl, provider);
  sessions.set(key, pending);
  pending.catch(() => sessions.delete(key));
  return pending;
}

async function runWithFallback(
  modelUrl: string,
  feeds: Record<string, ort.Tensor>,
) {
  const errors: string[] = [];

  for (const provider of await providerOrder()) {
    try {
      const session = await sessionFor(modelUrl, provider);
      const output = await session.run(feeds);
      return { output, provider };
    } catch (error) {
      errors.push(
        `${provider}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  throw new Error(
    `No ONNX execution provider was available (${errors.join("; ")}).`,
  );
}

function floatData(tensor: ort.Tensor | undefined, name: string) {
  if (!tensor) throw new Error(`The teaching model did not return "${name}".`);
  return Float32Array.from(tensor.data as ArrayLike<number>);
}

function tensorFromWire(tensor: WireTensor) {
  if (tensor.type === "int64") {
    return new ort.Tensor(
      "int64",
      BigInt64Array.from(tensor.data, BigInt),
      tensor.dims,
    );
  }
  return new ort.Tensor("float32", Float32Array.from(tensor.data), tensor.dims);
}

function tensorToWire(tensor: ort.Tensor): WireTensor {
  const type = tensor.type === "int64" ? "int64" : "float32";
  return {
    type,
    dims: Array.from(tensor.dims),
    data: Array.from(tensor.data as ArrayLike<number | bigint>, Number),
  };
}

workerScope.onmessage = async (event: MessageEvent<TeachingModelRequest>) => {
  const request = event.data;

  try {
    if (request.type === "run-transformer") {
      const input = new ort.Tensor(
        "int64",
        BigInt64Array.from(request.inputIds, BigInt),
        [1, request.inputIds.length],
      );
      const { output, provider } = await runWithFallback(request.modelUrl, {
        input_ids: input,
      });
      const q = floatData(output.q, "q");
      const k = floatData(output.k, "k");
      const v = floatData(output.v, "v");
      const attention = floatData(output.attention, "attention");

      const response: TransformerRunSuccess = {
        id: request.id,
        ok: true,
        type: "transformer-result",
        provider,
        q,
        k,
        v,
        attention,
        qkvShape: output.q.dims,
        attentionShape: output.attention.dims,
      };

      workerScope.postMessage(response, [
        q.buffer,
        k.buffer,
        v.buffer,
        attention.buffer,
      ]);
      return;
    }

    const feeds = Object.fromEntries(
      Object.entries(request.feeds).map(([name, tensor]) => [
        name,
        tensorFromWire(tensor),
      ]),
    );
    const { output, provider } = await runWithFallback(
      request.modelUrl,
      feeds,
    );
    const response: ModelRunSuccess = {
      id: request.id,
      ok: true,
      type: "model-result",
      provider,
      outputs: Object.fromEntries(
        Object.entries(output).map(([name, tensor]) => [
          name,
          tensorToWire(tensor),
        ]),
      ),
    };
    workerScope.postMessage(response);
  } catch (error) {
    const response: TransformerWorkerResponse = {
      id: request.id,
      ok: false,
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    };
    workerScope.postMessage(response);
  }
};

export {};
