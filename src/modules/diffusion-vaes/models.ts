import { runTeachingModel, teachingAssetUrl } from "@app/module-sdk";
import { PIXELS } from "./vae";

const decoderUrl = teachingAssetUrl("diffusion-vaes", "mnist-vae-decoder.onnx");
const encoderUrl = teachingAssetUrl("diffusion-vaes", "mnist-vae-encoder.onnx");
const denoiserUrl = teachingAssetUrl("diffusion-vaes", "mnist-diffusion-denoiser.onnx");

export type Provider = "webgpu" | "wasm";

function split(data: ReadonlyArray<number>, width: number) {
  const out: Float32Array[] = [];
  for (let offset = 0; offset + width <= data.length; offset += width) {
    out.push(Float32Array.from(data.slice(offset, offset + width)));
  }
  return out;
}

/** Run the real VAE decoder on a batch of 2-D latents; returns one [0, 1] image per latent. */
export async function decodeLatents(latents: ReadonlyArray<number>) {
  const count = latents.length / 2;
  const result = await runTeachingModel(decoderUrl, {
    latent: { type: "float32", dims: [count, 2], data: Array.from(latents) },
  });
  const image = result.outputs.image;
  if (!image) throw new Error("The VAE decoder returned no image.");
  return { images: split(image.data, PIXELS), provider: result.provider as Provider };
}

/** Run the real VAE encoder on one [0, 1] image; returns the posterior mean and log-variance. */
export async function encodeImage(image: ArrayLike<number>) {
  const result = await runTeachingModel(encoderUrl, {
    image: { type: "float32", dims: [1, 1, 28, 28], data: Array.from(image) },
  });
  const mean = result.outputs.mean;
  const logVariance = result.outputs.log_variance;
  if (!mean || !logVariance) throw new Error("The VAE encoder returned no latent.");
  return {
    mean: [mean.data[0], mean.data[1]] as [number, number],
    logVariance: [logVariance.data[0], logVariance.data[1]] as [number, number],
    provider: result.provider as Provider,
  };
}

/**
 * Run the real denoiser. Each entry is one image in [-1, 1] and its integer
 * timestep; the whole batch is one ONNX call.
 */
export async function predictNoise(images: ReadonlyArray<ArrayLike<number>>, timesteps: ReadonlyArray<number>) {
  const data: number[] = [];
  images.forEach((image) => {
    for (let index = 0; index < PIXELS; index += 1) data.push(image[index]);
  });
  const result = await runTeachingModel(denoiserUrl, {
    image: { type: "float32", dims: [images.length, 1, 28, 28], data },
    timestep: { type: "int64", dims: [timesteps.length], data: Array.from(timesteps) },
  });
  const noise = result.outputs.noise;
  if (!noise) throw new Error("The denoiser returned no noise prediction.");
  return { noise: split(noise.data, PIXELS), provider: result.provider as Provider };
}
