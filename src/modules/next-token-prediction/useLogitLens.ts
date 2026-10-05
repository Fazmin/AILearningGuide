import { useEffect, useState } from "react";
import { teachingAssetUrl, teachingTransformer } from "@app/module-sdk";
import { logitLens, type LensResult } from "./lens";

// The same file the Attention lab ships and the "Three models" card runs through ONNX Runtime.
const modelUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");

export type LensLoad =
  | { status: "loading" }
  | { status: "ready"; weights: teachingTransformer.TransformerWeights }
  | { status: "error"; error: string };

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

let weightsPromise: Promise<teachingTransformer.TransformerWeights> | undefined;

/** Reads the weights straight out of the ONNX file once, so the layers can be read one at a time. */
function loadWeights() {
  weightsPromise ??= fetch(modelUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`tiny-transformer.onnx returned HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then((buffer) => teachingTransformer.weightsFromOnnx(buffer))
    .catch((error: unknown) => {
      weightsPromise = undefined;
      throw error;
    });
  return weightsPromise;
}

export interface LensView {
  status: "loading" | "ready" | "error";
  error?: string;
  result?: LensResult;
  /** True while the result belongs to earlier text and a fresh read is pending. */
  stale: boolean;
}

/**
 * The logit lens for the last character of `ids`, recomputed (after a short pause, so typing stays
 * responsive) whenever the text changes. The forward pass is a plain TypeScript loop over the
 * transformer's own weights, which is what lets it stop and read the stream after each layer.
 */
export function useLogitLens(ids: ReadonlyArray<number>, debounceMs = 120): LensView {
  const key = ids.join(",");
  const [load, setLoad] = useState<LensLoad>({ status: "loading" });
  const [run, setRun] = useState<{ key: string; result?: LensResult; error?: string } | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    loadWeights()
      .then((weights) => alive && setLoad({ status: "ready", weights }))
      .catch((error: unknown) => alive && setLoad({ status: "error", error: message(error) }));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (load.status !== "ready" || ids.length === 0) return;
    const timer = window.setTimeout(() => {
      try {
        setRun({ key, result: logitLens(load.weights, ids) });
      } catch (error) {
        setRun({ key, error: message(error) });
      }
    }, debounceMs);
    return () => window.clearTimeout(timer);
    // key stands in for the id sequence.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, key, debounceMs]);

  if (load.status === "error") return { status: "error", error: load.error, stale: false };
  if (run && run.error && run.key === key) return { status: "error", error: run.error, stale: false };
  if (run?.result && run.key === key) return { status: "ready", result: run.result, stale: false };
  return { status: "loading", result: run?.result, stale: Boolean(run?.result) };
}
