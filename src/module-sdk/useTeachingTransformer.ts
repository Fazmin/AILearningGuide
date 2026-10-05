import { useEffect, useState } from "react";
import { runTeachingTransformer } from "@app/model-runtime/client";
import type { TransformerRunResult } from "@app/model-runtime/types";

export type TeachingModelStatus = "idle" | "loading" | "ready" | "error";

export function useTeachingTransformer({
  modelUrl,
  inputIds,
  debounceMs = 140,
}: {
  modelUrl: string;
  inputIds: number[];
  debounceMs?: number;
}) {
  const [status, setStatus] = useState<TeachingModelStatus>("idle");
  const [result, setResult] = useState<TransformerRunResult>();
  const [error, setError] = useState("");
  const inputKey = inputIds.join(",");

  useEffect(() => {
    if (!modelUrl || inputIds.length === 0) {
      setStatus("idle");
      setResult(undefined);
      return;
    }

    let disposed = false;
    setStatus("loading");
    setError("");
    const timer = window.setTimeout(() => {
      void runTeachingTransformer(modelUrl, inputIds)
        .then((next) => {
          if (disposed) return;
          setResult(next);
          setStatus("ready");
        })
        .catch((reason) => {
          if (disposed) return;
          setError(reason instanceof Error ? reason.message : String(reason));
          setStatus("error");
        });
    }, debounceMs);

    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
    // inputKey is a stable dependency for the token sequence.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debounceMs, inputKey, modelUrl]);

  return { status, result, error };
}
