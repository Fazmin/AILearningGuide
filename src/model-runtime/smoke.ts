import { teachingAssetUrl } from "./asset-registry";
import { runTeachingModel } from "./client";
import type { WireTensor } from "./types";

const transformerUrl = teachingAssetUrl("attention", "tiny-transformer.onnx");
const rnnUrl = teachingAssetUrl(
  "next-token-prediction",
  "character-rnn.onnx",
);
const saeUrl = teachingAssetUrl(
  "interpretability-features",
  "residual-sae.onnx",
);
const diffusionUrl = teachingAssetUrl(
  "diffusion-vaes",
  "mnist-diffusion-denoiser.onnx",
);
const vaeEncoderUrl = teachingAssetUrl(
  "diffusion-vaes",
  "mnist-vae-encoder.onnx",
);
const vaeDecoderUrl = teachingAssetUrl(
  "diffusion-vaes",
  "mnist-vae-decoder.onnx",
);

interface SmokeSpec {
  id: string;
  url: string;
  feeds: Record<string, WireTensor>;
  expectedOutputs: string[];
}

const floatTensor = (dims: number[], length: number): WireTensor => ({
  type: "float32",
  dims,
  data: Array.from({ length }, () => 0),
});

const tokenTensor: WireTensor = {
  type: "int64",
  dims: [1, 8],
  data: [1, 32, 46, 43, 1, 39, 47, 1],
};

const specs: SmokeSpec[] = [
  {
    id: "tiny-transformer",
    url: transformerUrl,
    feeds: { input_ids: tokenTensor },
    expectedOutputs: ["logits", "q", "k", "v", "attention", "residual"],
  },
  {
    id: "character-rnn",
    url: rnnUrl,
    feeds: { input_ids: tokenTensor },
    expectedOutputs: ["logits"],
  },
  {
    id: "residual-sae",
    url: saeUrl,
    feeds: { residual: floatTensor([1, 256], 256) },
    expectedOutputs: ["features", "reconstruction"],
  },
  {
    id: "mnist-vae-encoder",
    url: vaeEncoderUrl,
    feeds: { image: floatTensor([1, 1, 28, 28], 28 * 28) },
    expectedOutputs: ["mean", "log_variance"],
  },
  {
    id: "mnist-vae-decoder",
    url: vaeDecoderUrl,
    feeds: { latent: floatTensor([1, 2], 2) },
    expectedOutputs: ["image"],
  },
  {
    id: "mnist-diffusion-denoiser",
    url: diffusionUrl,
    feeds: {
      image: floatTensor([1, 1, 28, 28], 28 * 28),
      timestep: { type: "int64", dims: [1], data: [50] },
    },
    expectedOutputs: ["noise"],
  },
];

export interface ModelSmokeResult {
  id: string;
  provider: "webgpu" | "wasm";
  milliseconds: number;
  outputs: string[];
}

export async function smokeTestTeachingModels(): Promise<ModelSmokeResult[]> {
  const results: ModelSmokeResult[] = [];

  for (const spec of specs) {
    const started = performance.now();
    const result = await runTeachingModel(spec.url, spec.feeds);
    const outputs = Object.keys(result.outputs);
    for (const expected of spec.expectedOutputs) {
      const tensor = result.outputs[expected];
      if (!tensor) {
        throw new Error(`${spec.id} did not return ${expected}.`);
      }
      if (tensor.data.some((value) => !Number.isFinite(value))) {
        throw new Error(`${spec.id}/${expected} returned a non-finite value.`);
      }
    }
    results.push({
      id: spec.id,
      provider: result.provider,
      milliseconds: performance.now() - started,
      outputs,
    });
  }

  return results;
}
