/**
 * Loading and running the shipped models for the lab: the transformer's residual
 * stream through ONNX Runtime, the SAE through ONNX Runtime, and the same
 * transformer's weights in TypeScript (for the steering card's patch intervention).
 */

import { runTeachingModel, teachingAssetUrl, teachingTransformer } from "@app/module-sdk";
import cache from "./assets/sae-top-activations.json";
import { FEATURES, normalizeResidual, saeFromOnnx, WIDTH, type SaeWeights } from "./sae";
import { encodeText } from "./vocabulary";

export const transformerUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");
export const saeUrl = teachingAssetUrl("interpretability-features", "residual-sae.onnx");

export type Status = "loading" | "ready" | "error";

export interface SaeRun {
  text: string;
  /** Normalized residual per character. */
  x: Float32Array[];
  /** The raw layer-2 residual of every character, [characters x WIDTH] flattened. */
  residual: Float32Array;
  features: Float32Array[];
  reconstruction: Float32Array[];
  provider: string;
}

let decoderPromise: Promise<SaeWeights> | null = null;
export function loadDecoder() {
  decoderPromise ??= fetch(saeUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} for residual-sae.onnx`);
      return response.arrayBuffer();
    })
    .then((buffer) => saeFromOnnx(buffer))
    .catch((error: unknown) => {
      decoderPromise = null;
      throw error;
    });
  return decoderPromise;
}

let weightsPromise: Promise<teachingTransformer.TransformerWeights> | null = null;
/** The transformer's weights for the TypeScript forward pass, read from the same ONNX file. */
export function loadTransformerWeights() {
  weightsPromise ??= fetch(transformerUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} for tiny-transformer.onnx`);
      return response.arrayBuffer();
    })
    .then((buffer) => teachingTransformer.weightsFromOnnx(new Uint8Array(buffer)))
    .catch((error: unknown) => {
      weightsPromise = null;
      throw error;
    });
  return weightsPromise;
}

function rows(data: ReadonlyArray<number>, width: number) {
  const out: Float32Array[] = [];
  for (let row = 0; (row + 1) * width <= data.length; row += 1) {
    out.push(Float32Array.from(data.slice(row * width, (row + 1) * width)));
  }
  return out;
}

/** The raw residual stream after block 2 for a list of character ids, [ids.length x WIDTH] flattened. */
export async function layerTwoResidual(ids: ReadonlyArray<number>) {
  const transformer = await runTeachingModel(transformerUrl, {
    input_ids: { type: "int64", dims: [1, ids.length], data: [...ids] },
  });
  const residual = transformer.outputs.residual;
  if (!residual) throw new Error("The transformer returned no residual stream.");
  // residual is [layers, batch, sequence, width]; the SAE reads the layer it was trained on.
  const sequence = residual.dims[2];
  const offset = cache.layer * residual.dims[1] * sequence * WIDTH;
  return Float32Array.from(residual.data.slice(offset, offset + sequence * WIDTH));
}

export async function runSae(text: string): Promise<SaeRun> {
  const residual = await layerTwoResidual(encodeText(text));
  const x = Array.from({ length: residual.length / WIDTH }, (_, index) =>
    normalizeResidual(residual.subarray(index * WIDTH, (index + 1) * WIDTH), cache.normalization_mean, cache.normalization_scale),
  );
  const flat: number[] = [];
  x.forEach((row) => row.forEach((value) => flat.push(value)));
  const sae = await runTeachingModel(saeUrl, {
    residual: { type: "float32", dims: [x.length, WIDTH], data: flat },
  });
  const features = sae.outputs.features;
  const reconstruction = sae.outputs.reconstruction;
  if (!features || !reconstruction) throw new Error("The sparse autoencoder returned no features.");
  return {
    text,
    x,
    residual,
    features: rows(features.data, FEATURES),
    reconstruction: rows(reconstruction.data, WIDTH),
    provider: sae.provider,
  };
}
