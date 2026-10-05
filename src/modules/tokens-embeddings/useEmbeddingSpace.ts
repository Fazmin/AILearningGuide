import { useCallback, useEffect, useState } from "react";
import { teachingAssetUrl } from "@app/module-sdk";
import { buildSpace, readFloat32LE, type EmbeddingSpace, type ProjectionKind } from "./embeddings";

export type EmbeddingStatus = "loading" | "ready" | "error";

interface IndexFile {
  words: string[];
  count: number;
  dimensions: number;
}

type ProjectionFile = Record<ProjectionKind, Array<[number, number]>> & { words: string[] };

let pending: Promise<EmbeddingSpace> | null = null;

async function fetchAsset(file: string) {
  const response = await fetch(teachingAssetUrl("tokens-embeddings", file));
  if (!response.ok) throw new Error(`${file} returned HTTP ${response.status}`);
  return response;
}

/**
 * Fetch the 2 MB vector file, its word index, and the two projections once per
 * session. The vectors are read as an ArrayBuffer rather than bundled.
 */
export function loadEmbeddingSpace() {
  if (!pending) {
    pending = (async () => {
      const [indexResponse, vectorResponse, projectionResponse] = await Promise.all([
        fetchAsset("embedding-10k.index.json"),
        fetchAsset("embedding-10k.f32"),
        fetchAsset("embedding-10k.projections.json"),
      ]);
      const index = (await indexResponse.json()) as IndexFile;
      const buffer = await vectorResponse.arrayBuffer();
      const projections = (await projectionResponse.json()) as ProjectionFile;
      if (projections.words.length !== index.words.length) {
        throw new Error("Projection file and word index disagree on vocabulary size.");
      }
      return buildSpace(index.words, index.dimensions, readFloat32LE(buffer, index.count * index.dimensions), {
        umap: projections.umap,
        pca: projections.pca,
      });
    })();
    pending.catch(() => {
      pending = null;
    });
  }
  return pending;
}

export function useEmbeddingSpace() {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    status: EmbeddingStatus;
    space: EmbeddingSpace | null;
    error: string;
  }>({ status: "loading", space: null, error: "" });

  useEffect(() => {
    let active = true;
    setResult((current) => (current.space ? current : { status: "loading", space: null, error: "" }));
    loadEmbeddingSpace().then(
      (space) => {
        if (active) setResult({ status: "ready", space, error: "" });
      },
      (error: unknown) => {
        if (active) {
          setResult({
            status: "error",
            space: null,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },
    );
    return () => {
      active = false;
    };
  }, [attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  return { ...result, retry };
}
