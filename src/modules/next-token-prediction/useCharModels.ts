import { useEffect, useState } from "react";
import { runTeachingModel, teachingAssetUrl } from "@app/module-sdk";
import { TRANSFORMER_CONTEXT, VOCAB_SIZE, type ModelId } from "./sampling";

const bigramUrl = teachingAssetUrl("next-token-prediction", "bigram.json");
const rnnUrl = teachingAssetUrl("next-token-prediction", "character-rnn.onnx");
// The two-layer transformer trained on the same corpus and vocabulary ships with
// the attention module; its vocabulary and training history ship here too.
const transformerUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");

export type LoadStatus = "loading" | "ready" | "error";

export interface ModelView {
  status: LoadStatus;
  error?: string;
  /**
   * Next-token logits for each scored position: row j is the model's output after
   * reading character offset + j, so it predicts character offset + j + 1.
   * The bigram's logits are the logarithms of its table row.
   */
  rows?: number[][];
  offset: number;
  provider?: string;
  /** True while rows belong to an earlier text and a fresh run is pending. */
  stale: boolean;
}

interface RunRecord {
  key: string;
  status: LoadStatus;
  rows?: number[][];
  offset: number;
  error?: string;
  provider?: string;
}

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

let bigramPromise: Promise<number[][]> | undefined;

function loadBigram() {
  if (!bigramPromise) {
    bigramPromise = fetch(bigramUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`bigram.json returned HTTP ${response.status}`);
        return response.json() as Promise<{ probabilities?: unknown }>;
      })
      .then(({ probabilities }) => {
        if (
          !Array.isArray(probabilities) ||
          probabilities.length !== VOCAB_SIZE ||
          probabilities.some((row) => !Array.isArray(row) || row.length !== VOCAB_SIZE)
        ) {
          throw new Error("bigram.json is not a 66 × 66 probability table.");
        }
        return (probabilities as number[][]).map((row) => row.map((value) => Math.log(value)));
      });
    bigramPromise.catch(() => {
      bigramPromise = undefined;
    });
  }
  return bigramPromise;
}

function rowsFrom(data: ReadonlyArray<number>, dims: ReadonlyArray<number>, expected: number) {
  if (dims.length !== 3 || dims[1] !== expected || dims[2] !== VOCAB_SIZE) {
    throw new Error(`Unexpected logits shape [${dims.join(", ")}].`);
  }
  return Array.from({ length: expected }, (_, row) => Array.from(data.slice(row * VOCAB_SIZE, (row + 1) * VOCAB_SIZE)));
}

const emptyRun: RunRecord = { key: "", status: "loading", offset: 0 };

/**
 * Runs all three character models on the same token ids: the bigram table
 * synchronously once it has loaded, the GRU on every id, and the transformer on
 * the last 64 (its trained block size). ONNX runs happen in the shared worker.
 */
export function useCharModels(ids: ReadonlyArray<number>, debounceMs = 80): Record<ModelId, ModelView> {
  const key = ids.join(",");
  const [bigram, setBigram] = useState<{ status: LoadStatus; table?: number[][]; error?: string }>({ status: "loading" });
  const [runs, setRuns] = useState<Record<"rnn" | "transformer", RunRecord>>({ rnn: emptyRun, transformer: emptyRun });

  useEffect(() => {
    let alive = true;
    loadBigram()
      .then((table) => alive && setBigram({ status: "ready", table }))
      .catch((error) => alive && setBigram({ status: "error", error: message(error) }));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (ids.length === 0) return;
    let disposed = false;
    const timer = window.setTimeout(() => {
      for (const model of ["rnn", "transformer"] as const) {
        const windowIds = model === "transformer" ? ids.slice(-TRANSFORMER_CONTEXT) : ids.slice();
        const offset = ids.length - windowIds.length;
        runTeachingModel(model === "rnn" ? rnnUrl : transformerUrl, {
          input_ids: { type: "int64", data: windowIds, dims: [1, windowIds.length] },
        })
          .then((result) => {
            if (disposed) return;
            const logits = result.outputs.logits;
            if (!logits) throw new Error("The model returned no logits.");
            const rows = rowsFrom(logits.data, logits.dims, windowIds.length);
            setRuns((previous) => ({
              ...previous,
              [model]: { key, status: "ready", rows, offset, provider: result.provider },
            }));
          })
          .catch((error) => {
            if (disposed) return;
            setRuns((previous) => ({
              ...previous,
              [model]: { key, status: "error", offset, error: message(error) },
            }));
          });
      }
    }, debounceMs);
    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
    // key is a stable dependency for the id sequence.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, debounceMs]);

  const bigramView: ModelView =
    bigram.status === "ready" && bigram.table
      ? { status: "ready", rows: ids.map((id) => bigram.table![id]), offset: 0, stale: false }
      : { status: bigram.status, error: bigram.error, offset: 0, stale: false };

  const view = (record: RunRecord): ModelView =>
    record.key === key
      ? { status: record.status, rows: record.rows, offset: record.offset, error: record.error, provider: record.provider, stale: false }
      : { status: "loading", rows: record.rows, offset: record.offset, provider: record.provider, stale: true };

  return { bigram: bigramView, rnn: view(runs.rnn), transformer: view(runs.transformer) };
}
