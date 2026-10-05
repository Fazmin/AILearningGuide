// @vitest-environment node
/// <reference types="node" />
/**
 * Pins the figures that the Diffusion & VAEs lesson (content/standard.mdx, content/plain.mdx),
 * card-info.ts and module.ts quote from the shipped MNIST models, so a retrain cannot silently
 * stale them. Pattern: attention/quoted.test.ts and train-tiny-lm/quoted.test.ts.
 *
 * Where each figure comes from:
 *  - Offline measurements (training curves, the independent digit classifier's readings of
 *    generated samples and of the latent map, where the 10,000 MNIST test digits land) are written
 *    to mnist-models.metadata.json by models/train_mnist_models.py. The lab ships no MNIST images
 *    and no classifier, so it quotes these rather than recomputing them; the tests below read the
 *    same file and check the lesson's wording against it.
 *  - Everything the lab itself computes is reproduced here with the lab's own helpers
 *    (diffusion.ts, vae.ts) on the real ONNX files, run through onnxruntime-web's WebAssembly CPU
 *    provider: the default digit's error curve, the seed-7 reverse runs, the ideal denoiser's
 *    weights, the seeds 1 to 24 sweep, the preset encodings. The lab prefers WebGPU when the
 *    browser has it; that is browser-only and not exercised here. Figures are quoted to two or
 *    three digits, far from a rounding boundary.
 *  - "Which digit is where" claims use the classifier's label for each of the 256 mosaic tiles
 *    (metadata vae.test_digits.map), which is also how the preset reconstructions are read.
 *
 * If this file fails after a retrain: re-measure, then update the lesson, card-info and module.ts
 * together. Do not just edit the numbers here.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import metadata from "./assets/mnist-models.metadata.json";
import shipped from "./assets/mnist-diffusion-schedule.json";
import cardInfo from "./card-info";
import {
  backgroundShare,
  ddpmStep,
  idealDenoise,
  meanSquaredError,
  noiseErrorByStep,
  normalStream,
  reverseStart,
  runIdealReverse,
  runNetworkReverse,
  scheduleFromAlphaBar,
  STEPS,
  timestepForCall,
  toSigned,
} from "./diffusion";
import definition from "./module";
import {
  binaryCrossEntropy,
  gaussianKl,
  latentGrid,
  MAP_GRID,
  MAP_MAX,
  MAP_MIN,
  PIXELS,
  PRESET_STROKES,
  rasterizeStrokes,
  referenceLatents,
} from "./vae";

const root = process.cwd();
const folder = "src/modules/diffusion-vaes";
const read = (path: string) => readFileSync(resolve(root, path));
const text = (path: string) => read(path).toString("utf8");
const onnx = (name: string) => new Uint8Array(read(`${folder}/assets/${name}`));

/** Lesson files wrap lines at 100 columns; compare them with whitespace collapsed. */
const flat = (value: string) => value.replace(/\s+/g, " ");
const standard = flat(text(`${folder}/content/standard.mdx`));
const plain = flat(text(`${folder}/content/plain.mdx`));
const explore = text(`${folder}/Explore.tsx`);
const cards = Object.values(cardInfo)
  .flatMap((card) => [card.title, card.summary, ...card.whatYouSee, ...card.howItWorks, ...(card.controls ?? []), ...(card.notice ?? []), ...card.limits])
  .join("\n");
const moduleText = [
  ...definition.stepInstructions,
  ...definition.glossary.map((entry) => entry.definition),
  ...checkpointQuestions(definition).flatMap((question) => [question.prompt, ...question.options, question.explanation]),
].join("\n");
const everything = [standard, plain, cards, moduleText, explore].join("\n");

const schedule = scheduleFromAlphaBar(shipped.alpha_cumulative);
const diffusion = metadata.diffusion;
const vae = metadata.vae;
const testDigits = vae.test_digits;
const sampleCheck = diffusion.sample_check;
const mapLabels = testDigits.map.labels;
const mapConfidence = testDigits.map.confidence;
const percent = (value: number) => `${Math.round(value * 100)}%`;
const fixed = (value: number, digits: number) => value.toFixed(digits);

/** The classifier's label for the mosaic tile that contains a latent point. */
function tileLabel(z1: number, z2: number) {
  const column = Math.min(MAP_GRID - 1, Math.floor(((z1 - MAP_MIN) / (MAP_MAX - MAP_MIN)) * MAP_GRID));
  const row = Math.min(MAP_GRID - 1, Math.floor(((MAP_MAX - z2) / (MAP_MAX - MAP_MIN)) * MAP_GRID));
  return mapLabels[row * MAP_GRID + column];
}
/** Centre of mosaic tile i, in latent coordinates. */
function tileCentre(index: number) {
  const row = Math.floor(index / MAP_GRID);
  const column = index % MAP_GRID;
  const size = (MAP_MAX - MAP_MIN) / MAP_GRID;
  return [MAP_MIN + (column + 0.5) * size, MAP_MAX - (row + 0.5) * size] as const;
}
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

describe("what the metadata records, as the lesson quotes it", () => {
  it("describes the denoiser: a 1,181,217-parameter U-Net trained 45 epochs on all 60,000 digits", () => {
    expect(metadata.samples).toBe(60000);
    expect(diffusion.parameters).toBe(1181217);
    expect(diffusion.epochs).toBe(45);
    expect(diffusion.history).toHaveLength(diffusion.epochs);
    expect(diffusion.architecture.kind).toBe("convolutional U-Net");
    expect(diffusion.architecture.base_channels).toBe(32);
    expect(diffusion.architecture.residual_blocks).toBe(6);
    expect(diffusion.architecture.resolutions).toEqual([28, 14, 7]);
    expect(diffusion.architecture.conditioning).toBe("none (unconditional)");
    expect(diffusion.steps).toBe(STEPS);
  });

  it("ends training at a noise MSE of 0.036 against 1.0 for predicting zero", () => {
    const last = diffusion.history[diffusion.history.length - 1].noise_mse;
    expect(last).toBeCloseTo(0.036, 3);
    expect(diffusion.history[0].noise_mse).toBeGreaterThan(0.1);
  });

  it("measures the network on 10,000 test digits: error 0.13 at t = 0, 0.028 at t = 49, under 0.001 at t = 99", () => {
    const held = diffusion.held_out_noise_mse;
    expect(held.t0).toBeCloseTo(0.13, 2);
    expect(held.t49).toBeCloseTo(0.028, 3);
    // t = 99 is tiny and moves between MPS runs (3e-4 and 4e-4 were both seen), so only its bound is quoted.
    expect(held.t99).toBeLessThan(0.001);
    expect(held.uniform_t).toBeCloseTo(0.036, 3);
  });

  it("records the pure-noise sample check that gated publication", () => {
    expect(sampleCheck.count).toBe(256);
    expect(sampleCheck.start).toBe("pure noise");
    expect(sampleCheck.mean_confidence).toBeCloseTo(0.95, 2);
    expect(sampleCheck.fraction_above_0_9).toBeCloseTo(0.86, 2);
    expect(sampleCheck.real_test_digits_mean_confidence).toBeCloseTo(0.99, 2);
    expect(sampleCheck.classifier_test_accuracy).toBeGreaterThan(0.98);
    // All ten digits are generated; none is missing or dominant.
    expect(sampleCheck.class_counts).toHaveLength(10);
    expect(Math.min(...sampleCheck.class_counts)).toBeGreaterThanOrEqual(10);
    expect(Math.max(...sampleCheck.class_counts)).toBeLessThanOrEqual(40);
    // Novel, not memorized: as far from the training set as real held-out digits are.
    expect(sampleCheck.mean_nearest_training_distance).toBeCloseTo(4.45, 2);
    expect(sampleCheck.real_test_digits_mean_nearest_training_distance).toBeCloseTo(4.11, 2);
    expect(sampleCheck.mean_nearest_training_distance).toBeGreaterThan(
      0.8 * sampleCheck.real_test_digits_mean_nearest_training_distance,
    );
    expect(sampleCheck.near_copy_distance).toBe(1.5);
    expect(sampleCheck.fraction_within_near_copy_distance).toBeCloseTo(0.02, 2);
    expect(sampleCheck.real_test_digits_fraction_within_near_copy_distance).toBeCloseTo(0.04, 2);
    // The few samples that sit close to a training digit are thin 1s, which look alike in any set.
    expect(sampleCheck.closest_eight_nearest_training_labels).toEqual(Array(8).fill(1));
    expect(sampleCheck.fraction_within_near_copy_distance).toBeLessThan(
      sampleCheck.real_test_digits_fraction_within_near_copy_distance,
    );
  });

  it("describes the VAE: 40 epochs, KL weight 0.35, last epoch 133.3 nats of BCE and 8.03 of KL", () => {
    expect(vae.epochs).toBe(40);
    expect(vae.history).toHaveLength(vae.epochs);
    expect(vae.kl_weight).toBe(0.35);
    const last = vae.history[vae.history.length - 1];
    expect(last.reconstruction).toBeCloseTo(133.3, 1);
    expect(last.kl).toBeCloseTo(8.03, 2);
  });

  it("spreads the test digits wider than the prior: sd 1.41 and 1.55, 15% beyond radius 3 (prior 1.1%)", () => {
    expect(testDigits.test_images).toBe(10000);
    expect(testDigits.encoded_mean_std[0]).toBeCloseTo(1.41, 2);
    expect(testDigits.encoded_mean_std[1]).toBeCloseTo(1.55, 2);
    expect(testDigits.fraction_beyond_radius_3).toBeCloseTo(0.15, 2);
    // A standard normal in two dimensions has probability exp(-r^2 / 2) outside radius r.
    expect(Math.exp(-4.5)).toBeCloseTo(0.011, 3);
    expect(testDigits.fraction_within_radius_2).toBeCloseTo(0.61, 2);
    expect(testDigits.posterior_sigma_average[0]).toBeCloseTo(0.036, 3);
    expect(testDigits.posterior_sigma_average[1]).toBeCloseTo(0.036, 3);
  });

  it("reads reconstructions and prior draws with the independent classifier", () => {
    expect(testDigits.reconstruction.bce).toBeCloseTo(135.6, 1);
    expect(testDigits.reconstruction.label_agreement).toBeCloseTo(0.745, 3);
    expect(testDigits.prior_samples.count).toBe(10000);
    expect(testDigits.prior_samples.fraction_above_0_9).toBeCloseTo(0.483, 3);
    // The comparison the lesson draws: prior draws decode to readable digits far less often than
    // the diffusion network's pure-noise samples do.
    expect(testDigits.prior_samples.fraction_above_0_9).toBeLessThan(sampleCheck.fraction_above_0_9 - 0.2);
  });

  it("keeps the map grid the lab draws in step with the metadata", () => {
    expect(testDigits.map.grid).toBe(MAP_GRID);
    expect(testDigits.map.limit).toBe(MAP_MAX);
    expect(MAP_MIN).toBe(-MAP_MAX);
    expect(mapLabels).toHaveLength(MAP_GRID * MAP_GRID);
    expect(mapConfidence).toHaveLength(MAP_GRID * MAP_GRID);
  });
});

describe("the latent map: where each digit lives (classifier labels of the 256 tiles)", () => {
  it("puts 0s along the bottom, 2s on the right, 1s top right, 7s top left and 5s on the left", () => {
    expect(tileLabel(0, -3.75)).toBe(0);
    expect(tileLabel(3.75, -1)).toBe(2);
    expect(tileLabel(3.75, 3.75)).toBe(1);
    expect(tileLabel(-3.75, 3.75)).toBe(7);
    expect(tileLabel(-3.75, -1.25)).toBe(5);
  });

  it("puts the default point (-2, -0.8) among the 5s", () => {
    expect(tileLabel(-2, -0.8)).toBe(5);
  });

  it("covers all ten digits, and the 64 references (tile centres) cover all ten too", () => {
    expect(new Set(mapLabels).size).toBe(10);
    const references = referenceLatents();
    expect(references).toHaveLength(2 * 64);
    const referenceLabels = new Set<number>();
    for (let index = 0; index < 64; index += 1) {
      const [z1, z2] = [references[2 * index], references[2 * index + 1]];
      // Each reference is the exact centre of a tile.
      const row = Math.floor(((MAP_MAX - z2) / (MAP_MAX - MAP_MIN)) * MAP_GRID);
      const column = Math.floor(((z1 - MAP_MIN) / (MAP_MAX - MAP_MIN)) * MAP_GRID);
      const [cz1, cz2] = tileCentre(row * MAP_GRID + column);
      expect(z1).toBeCloseTo(cz1, 10);
      expect(z2).toBeCloseTo(cz2, 10);
      referenceLabels.add(tileLabel(z1, z2));
    }
    expect(referenceLabels.size).toBe(10);
  });

  it("follows the card's walk: 0s along the bottom, up the right side through the 2s to the 1s, along the top to the 7s", () => {
    const at = (row: number, column: number) => mapLabels[row * MAP_GRID + column];
    const last = MAP_GRID - 1;
    const bottomRow = Array.from({ length: MAP_GRID }, (_, column) => at(last, column));
    expect(bottomRow.slice(0, 14).every((label) => label === 0)).toBe(true);
    expect(bottomRow.slice(14)).toEqual([2, 2]);
    const rightSide = Array.from({ length: MAP_GRID }, (_, row) => at(last - row, last));
    expect(new Set(rightSide)).toEqual(new Set([1, 2]));
    expect(rightSide.indexOf(1)).toBeGreaterThan(rightSide.lastIndexOf(2));
    const topRow = Array.from({ length: MAP_GRID }, (_, column) => at(0, column));
    expect(new Set(topRow)).toEqual(new Set([1, 7]));
    expect(topRow.indexOf(7)).toBe(0);
  });

  it("crowds eight of the ten digits into the 12 tiles within radius 1", () => {
    const inside = mapLabels.filter((_, index) => Math.hypot(...tileCentre(index)) < 1);
    expect(inside).toHaveLength(12);
    expect(new Set(inside).size).toBe(8);
  });

  it("blurs in the middle: mean classifier confidence 0.70 within radius 1 and 0.96 beyond radius 3", () => {
    const inside: number[] = [];
    const outside: number[] = [];
    mapConfidence.forEach((confidence, index) => {
      const radius = Math.hypot(...tileCentre(index));
      if (radius < 1) inside.push(confidence);
      if (radius > 3) outside.push(confidence);
    });
    expect(inside.length).toBe(12);
    expect(mean(inside)).toBeCloseTo(0.7, 2);
    expect(mean(outside)).toBeCloseTo(0.96, 2);
  });

  it("matches the mosaic layout the lab uses (row-major from the top)", () => {
    const grid = latentGrid(MAP_GRID, MAP_MIN, MAP_MAX);
    for (const index of [0, 15, 120, 255]) {
      const [z1, z2] = tileCentre(index);
      expect(grid[2 * index]).toBeCloseTo(z1, 10);
      expect(grid[2 * index + 1]).toBeCloseTo(z2, 10);
    }
  });
});

describe("the lab's own computations on the shipped ONNX files (onnxruntime-web, WASM CPU)", () => {
  let decoder: ort.InferenceSession;
  let encoder: ort.InferenceSession;
  let denoiser: ort.InferenceSession;

  const split = (data: Float32Array) => {
    const out: Float32Array[] = [];
    for (let offset = 0; offset + PIXELS <= data.length; offset += PIXELS) out.push(data.slice(offset, offset + PIXELS));
    return out;
  };
  const decode = async (latents: number[]) => {
    const result = await decoder.run({ latent: new ort.Tensor("float32", Float32Array.from(latents), [latents.length / 2, 2]) });
    return split(result.image.data as Float32Array);
  };
  const predictBatch = async (images: ReadonlyArray<Float32Array>, timesteps: ReadonlyArray<number>) => {
    const data = new Float32Array(images.length * PIXELS);
    images.forEach((image, index) => data.set(image, index * PIXELS));
    const result = await denoiser.run({
      image: new ort.Tensor("float32", data, [images.length, 1, 28, 28]),
      timestep: new ort.Tensor("int64", BigInt64Array.from(timesteps, BigInt), [timesteps.length]),
    });
    return split(result.noise.data as Float32Array);
  };
  const rms = (image: ArrayLike<number>) => Math.sqrt(image.length ? Array.from(image).reduce((sum, value) => sum + value * value, 0) / image.length : 0);

  let x0: Float32Array;
  let references: Float32Array[];
  const nearest = (image: Float32Array) => {
    let best = { index: -1, distance: Infinity };
    references.forEach((reference, index) => {
      const distance = meanSquaredError(image, reference);
      if (distance < best.distance) best = { index, distance };
    });
    return best;
  };

  beforeAll(async () => {
    ort.env.wasm.numThreads = 1;
    const make = (name: string) => ort.InferenceSession.create(onnx(name), { executionProviders: ["wasm"] });
    [decoder, encoder, denoiser] = await Promise.all([
      make("mnist-vae-decoder.onnx"),
      make("mnist-vae-encoder.onnx"),
      make("mnist-diffusion-denoiser.onnx"),
    ]);
    // The lab's default digit: the decoder at z = (-2, -0.8), scaled to [-1, 1].
    x0 = toSigned((await decode([-2, -0.8]))[0]);
    references = (await decode(referenceLatents())).map(toSigned);
  }, 120_000);

  it("keeps the ONNX contract: names and shapes", () => {
    expect(denoiser.inputNames).toEqual(["image", "timestep"]);
    expect(denoiser.outputNames).toEqual(["noise"]);
    expect(encoder.inputNames).toEqual(["image"]);
    expect(encoder.outputNames).toEqual(["mean", "log_variance"]);
    expect(decoder.inputNames).toEqual(["latent"]);
    expect(decoder.outputNames).toEqual(["image"]);
  });

  it("scores the network's noise prediction at every step: 0.60 at t = 0, under 0.001 at t = 99, against 1.03 for zero", async () => {
    const epsilon = normalStream(7 + 1000).fill();
    const curve = await noiseErrorByStep(x0, epsilon, schedule, predictBatch);
    expect(curve.network).toHaveLength(STEPS);
    expect(curve.baseline).toBeCloseTo(1.03, 2);
    expect(curve.network[0]).toBeCloseTo(0.60, 2);
    expect(curve.network[STEPS - 1]).toBeLessThan(0.001);
    // Hardest where there is least noise; never worse than guessing zero.
    expect(curve.network.indexOf(Math.max(...curve.network))).toBe(0);
    curve.network.forEach((value) => expect(value).toBeLessThan(curve.baseline));
    // The legend prints the last value to three decimals.
    expect(curve.network[STEPS - 1].toFixed(3)).toBe("0.000");
    expect(Math.sqrt(schedule.alphaBar[STEPS - 1]).toFixed(3)).toBe("0.031");
    expect(Math.sqrt(1 - schedule.alphaBar[STEPS - 1]).toFixed(3)).toBe("1.000");
  }, 120_000);

  it("reads faint noise easily off a flat background: 0.07 to 0.10 at t = 0 on the sharp presets, against 0.60 on the decoder's soft x0", async () => {
    const errors: number[] = [];
    for (const strokes of Object.values(PRESET_STROKES)) {
      const epsilon = normalStream(1007).fill();
      const curve = await noiseErrorByStep(toSigned(rasterizeStrokes(strokes)), epsilon, schedule, predictBatch);
      errors.push(curve.network[0]);
      expect(curve.network[STEPS - 1]).toBeLessThan(0.001);
    }
    expect(Math.min(...errors).toFixed(2)).toBe("0.07");
    expect(Math.max(...errors).toFixed(2)).toBe("0.10");
  }, 120_000);

  describe("seed 7, the lab's default", () => {
    const predictOne = async (x: Float32Array, t: number) => (await predictBatch([x], [t]))[0];
    let start: ReturnType<typeof reverseStart>;
    let network: Awaited<ReturnType<typeof runNetworkReverse>>;
    let ideal: ReturnType<typeof runIdealReverse>;

    beforeAll(async () => {
      start = reverseStart(7, "noise", x0, schedule);
      network = await runNetworkReverse(start, predictOne, schedule);
      ideal = runIdealReverse(start, references, schedule);
    }, 240_000);

    it("turns static into a digit: Background pixels climb from 0.17 to 0.82", () => {
      expect(network.states).toHaveLength(STEPS + 1);
      expect(backgroundShare(network.states[0])).toBeCloseTo(0.17, 2);
      expect(backgroundShare(network.states[STEPS])).toBeCloseTo(0.82, 2);
      // Standard-normal noise has about 18% of its pixels at or below -0.9.
      expect(backgroundShare(normalStream(99).fill(200_000))).toBeCloseTo(0.184, 2);
      expect(rms(network.states[0])).toBeCloseTo(1.04, 1);
      expect(rms(network.states[STEPS])).toBeCloseTo(0.96, 1);
    });

    it("ends the network on a new image, nearest to reference #28 at squared distance 0.26", () => {
      const best = nearest(network.states[STEPS]);
      expect(best.index + 1).toBe(28);
      expect(best.distance).toBeCloseTo(0.26, 2);
      expect(fixed(best.distance, 2)).toBe("0.26");
    });

    it("ends the ideal denoiser exactly on reference #64, with distance 0", () => {
      const best = nearest(ideal.states[STEPS]);
      expect(best.index + 1).toBe(64);
      expect(best.distance).toBeLessThan(1e-12);
      expect(backgroundShare(ideal.states[STEPS])).toBeCloseTo(0.75, 2);
    });

    it("gives one reference half the weight after 15 calls and 99% after 32", () => {
      const tops = (ideal.weights ?? []).map((weights) => Math.max(...Array.from(weights)));
      expect(tops[0]).toBeLessThan(0.05);
      expect(tops.findIndex((weight) => weight > 0.5)).toBe(15);
      expect(tops.findIndex((weight) => weight > 0.99)).toBe(32);
    });

    it("starts from q(x_99 | x0) or from pure noise and reaches almost the same sample", async () => {
      const noised = reverseStart(7, "digit", x0, schedule);
      expect(Math.sqrt(schedule.alphaBar[STEPS - 1])).toBeCloseTo(0.031, 3);
      const run = await runNetworkReverse(noised, predictOne, schedule);
      expect(meanSquaredError(run.states[STEPS], network.states[STEPS])).toBeLessThan(0.02);
      expect(nearest(run.states[STEPS]).index).toBe(nearest(network.states[STEPS]).index);
    }, 240_000);

    it("shows the sample the lab prints: nearest reference and its distance", () => {
      const final = network.states[STEPS];
      const idealFinal = ideal.states[STEPS];
      expect(nearest(final).distance).toBeGreaterThan(0.2);
      expect(nearest(idealFinal).distance.toFixed(4)).toBe("0.0000");
    });
  });

  describe("seeds 1 to 24 from pure noise", () => {
    const seeds = Array.from({ length: 24 }, (_, index) => index + 1);

    it("makes the ideal denoiser end on 20 different references", () => {
      const ends = seeds.map((seed) => {
        const run = runIdealReverse(reverseStart(seed, "noise", x0, schedule), references, schedule);
        const final = run.states[STEPS];
        const best = nearest(final);
        expect(best.distance).toBeLessThan(1e-12);
        return best.index;
      });
      expect(new Set(ends).size).toBe(20);
    });

    it("never makes the network copy a reference: nearest distance stays above 0.07, and every sample is digit-like", async () => {
      const starts = seeds.map((seed) => reverseStart(seed, "noise", x0, schedule));
      let states = starts.map((item) => item.x);
      for (let call = 0; call < STEPS; call += 1) {
        const t = timestepForCall(call);
        const predictions = await predictBatch(states, states.map(() => t));
        states = states.map((state, index) => ddpmStep(state, predictions[index], t, schedule, t > 0 ? starts[index].z[call] : null));
      }
      const distances = states.map((state) => nearest(state).distance);
      expect(Math.min(...distances)).toBeGreaterThan(0.07);
      expect(Math.max(...distances)).toBeLessThan(0.5);
      states.forEach((state) => {
        expect(backgroundShare(state)).toBeGreaterThan(0.65);
        expect(Math.max(...state)).toBeLessThan(1.5);
        expect(Math.min(...state)).toBeGreaterThan(-1.5);
      });
    }, 300_000);
  });

  describe("the presets through the encoder and decoder", () => {
    const encodePreset = async (name: string) => {
      const drawing = rasterizeStrokes(PRESET_STROKES[name]);
      const result = await encoder.run({ image: new ort.Tensor("float32", Float32Array.from(drawing), [1, 1, 28, 28]) });
      const mu = Array.from(result.mean.data as Float32Array);
      const logVariance = Array.from(result.log_variance.data as Float32Array);
      const [reconstruction] = await decode(mu);
      return {
        mu,
        sigma: logVariance.map((value) => Math.exp(0.5 * value)),
        bce: binaryCrossEntropy(drawing, reconstruction),
        kl: gaussianKl(mu, logVariance),
        read: tileLabel(mu[0], mu[1]),
      };
    };

    it("brings 0, 1 and 3 back as themselves and the heavy-barred 7 back as a 3", async () => {
      const results = Object.fromEntries(await Promise.all(Object.keys(PRESET_STROKES).map(async (name) => [name, await encodePreset(name)] as const)));
      expect(results.zero.read).toBe(0);
      expect(results.one.read).toBe(1);
      expect(results.three.read).toBe(3);
      expect(results.seven.read).toBe(3);
    }, 60_000);

    it("encodes 3 to mu = (1.46, -0.32) with BCE 121.1 and KL 7.12, and 7 to (1.55, 0.02) with BCE 211.6 and KL 6.26", async () => {
      const three = await encodePreset("three");
      expect(three.mu[0]).toBeCloseTo(1.46, 2);
      expect(three.mu[1]).toBeCloseTo(-0.32, 2);
      expect(three.bce).toBeCloseTo(121.1, 1);
      expect(three.kl).toBeCloseTo(7.12, 2);
      const seven = await encodePreset("seven");
      expect(seven.mu[0]).toBeCloseTo(1.55, 2);
      expect(seven.mu[1]).toBeCloseTo(0.02, 2);
      expect(seven.bce).toBeCloseTo(211.6, 1);
      expect(seven.kl).toBeCloseTo(6.26, 2);
      // "sigma is about 0.03 to 0.05 while the prior's is 1"
      for (const result of [three, seven]) result.sigma.forEach((value) => {
        expect(value).toBeGreaterThan(0.025);
        expect(value).toBeLessThan(0.055);
      });
    }, 60_000);
  });
});

describe("the ideal denoiser's defining property, on the shipped schedule", () => {
  it("returns weights that sum to one and a clean-image estimate inside the span of its references", () => {
    const references = [Float32Array.from({ length: PIXELS }, () => -1), Float32Array.from({ length: PIXELS }, () => 1)];
    const x = normalStream(5).fill();
    const { weights, x0: estimate } = idealDenoise(x, STEPS - 1, references, schedule);
    expect(weights[0] + weights[1]).toBeCloseTo(1, 12);
    expect(Math.min(...estimate)).toBeGreaterThanOrEqual(-1 - 1e-6);
    expect(Math.max(...estimate)).toBeLessThanOrEqual(1 + 1e-6);
  });
});

describe("the lesson text agrees with the measurements", () => {
  const needles = (source: string, name: string, values: string[]) =>
    values.forEach((needle) => expect(source, `${name} should quote "${needle}"`).toContain(needle));

  it("states the schedule's last step as 0.031 of the signal, and the 3% the plain lesson rounds it to", () => {
    needles(standard, "standard.mdx", ["√ᾱ = 0.031", "falls to 0.031"]);
    needles(plain, "plain.mdx", ["just 3% of the digit", "Its last step keeps 3% of the digit"]);
    needles(cards, "card-info.ts", ["0.031", "0.00097", "β = 0.0006 at t = 0", "β = 0.55 at t = 99"]);
    needles(moduleText, "module.ts", ["falls to 0.03 at the last step", "keeps only 0.03 of the signal"]);
    expect(schedule.beta[0].toFixed(4)).toBe("0.0006");
    expect(schedule.beta[STEPS - 1].toFixed(2)).toBe("0.55");
    expect(schedule.alphaBar[STEPS - 1].toFixed(5)).toBe("0.00097");
  });

  it("quotes the denoiser's size, epochs and training data in both modes and the glossary", () => {
    const parameters = diffusion.parameters.toLocaleString("en-US");
    needles(standard, "standard.mdx", [parameters, `${diffusion.epochs} epochs`, "60,000"]);
    needles(plain, "plain.mdx", [parameters, `${diffusion.epochs} passes`, "60,000"]);
    needles(moduleText, "module.ts", [parameters, `${diffusion.epochs} epochs`]);
    needles(cards, "card-info.ts", [parameters, `${diffusion.epochs} epochs`]);
    // The lab's own note builds this sentence from the metadata, so it cannot go stale.
    needles(explore, "Explore.tsx", ["metadata.diffusion.parameters", "diffusionHistory.length", "metadata.samples"]);
  });

  it("quotes the offline classifier readings of generated samples", () => {
    const confidence = fixed(sampleCheck.mean_confidence, 2);
    const above = percent(sampleCheck.fraction_above_0_9);
    const real = fixed(sampleCheck.real_test_digits_mean_confidence, 2);
    needles(standard, "standard.mdx", [`mean confidence of ${confidence}`, `${above} of them`, `against ${real} for real test digits`]);
    needles(plain, "plain.mdx", [`${confidence} average confidence`, `${above}`, `against ${real} for real digits`]);
    needles(cards, "card-info.ts", [`mean confidence of ${confidence}`, `${above} of them`, `against ${real} for real test digits`]);
    const near = percent(sampleCheck.fraction_within_near_copy_distance);
    const realNear = percent(sampleCheck.real_test_digits_fraction_within_near_copy_distance);
    needles(standard, "standard.mdx", [
      fixed(sampleCheck.mean_nearest_training_distance, 2),
      fixed(sampleCheck.real_test_digits_mean_nearest_training_distance, 2),
      `${near} of them sit within 1.5`,
      `against ${realNear} of real test`,
    ]);
    needles(plain, "plain.mdx", [fixed(sampleCheck.mean_nearest_training_distance, 2), fixed(sampleCheck.real_test_digits_mean_nearest_training_distance, 2)]);
  });

  it("quotes the held-out noise errors in the Forward process card", () => {
    const held = diffusion.held_out_noise_mse;
    needles(cards, "card-info.ts", [
      `${fixed(held.t0, 2)} at t = 0`,
      `${fixed(held.t49, 3)} at t = 49`,
      "under 0.001 at t = 99",
      "0.07 to 0.10",
    ]);
  });

  it("quotes the VAE's training and test-digit figures", () => {
    const last = vae.history[vae.history.length - 1];
    needles(cards, "card-info.ts", [
      `${vae.epochs} epochs on all 60,000`,
      `${fixed(last.reconstruction, 1)} nats of BCE and ${fixed(last.kl, 2)} of KL`,
      `${fixed(testDigits.encoded_mean_std[0], 2)} and ${fixed(testDigits.encoded_mean_std[1], 2)}`,
      `${percent(testDigits.fraction_beyond_radius_3)} land beyond radius 3`,
      `${percent(testDigits.reconstruction.label_agreement)} of reconstructions`,
      `${percent(testDigits.prior_samples.fraction_above_0_9)} of 10,000 prior draws`,
    ]);
    needles(standard, "standard.mdx", [
      `${fixed(testDigits.encoded_mean_std[0], 2)} and ${fixed(testDigits.encoded_mean_std[1], 2)}`,
      `${percent(testDigits.fraction_beyond_radius_3)} of test digits land beyond radius 3`,
      `${percent(testDigits.prior_samples.fraction_above_0_9)} of 10,000 prior draws`,
    ]);
    needles(plain, "plain.mdx", [
      `${percent(testDigits.fraction_beyond_radius_3)} of real digits land far out`,
      `only ${percent(testDigits.prior_samples.fraction_above_0_9)} of 10,000 picks`,
    ]);
    needles(cards, "card-info.ts", ["0.70", "0.96", "eight of the ten digits appear among the 12 tiles within radius 1"]);
    for (const source of [standard, plain, cards]) expect(source).not.toContain("@@");
  });

  it("quotes the seed-7 and seeds 1 to 24 results measured above", () => {
    // Reverse diffusion card and both lessons: the ideal run lands on #64, the network is nearest #28 at 0.26.
    needles(cards, "card-info.ts", ["reference #64", "reference #28", "squared distance of 0.26", "after 15 calls and 99% after 32", "20 different references", "never closer than 0.07"]);
    needles(standard, "standard.mdx", ["reference #64", "reference #28", "squared distance of 0.26", "after 15 calls and 99% after 32", "20 different references", "0.17 to 0.82"]);
    needles(plain, "plain.mdx", ["reference #64", "reference #28", "at 0.26", "after 15 calls and 99% after 32", "20 different ones", "0.17 to 0.82"]);
    needles(cards, "card-info.ts", ["0.17 to 0.82", "about 0.17 for the seed-7 starting noise and about 0.82", "0.60 at t = 0", "under 0.001 at t = 99", "1.03"]);
    needles(standard, "standard.mdx", ["0.60 at `t = 0`", "under 0.001 at `t = 99`"]);
    needles(plain, "plain.mdx", ["error 0.60", "under 0.001"]);
  });

  it("quotes the preset encodings: 7 and 3", () => {
    needles(cards, "card-info.ts", ["μ = (1.55, 0.02)", "BCE 211.6 nats and KL 6.26", "μ = (1.46, −0.32)", "BCE 121.1 and KL 7.12"]);
    needles(standard, "standard.mdx", ["μ = (1.55, 0.02)", "BCE is 211.6 nats and its KL 6.26", "BCE 121.1"]);
  });

  it("explains why Background pixels replaced Pixel RMS: noise and digits both sit near 1", () => {
    needles(cards, "card-info.ts", ["Pixel RMS would not do here"]);
    expect(explore).toContain("Background pixels");
    expect(explore).not.toContain("Pixel RMS");
    expect(moduleText).not.toContain("Pixel RMS");
  });

  it("no longer carries the old model's claims", () => {
    for (const stale of ["0.603", "0.364", "0.858", "1.82", "1.04 at", "772,192", "3 epochs", "#56", "inverted smudges", "off-schedule", "lower right", "lower-right", "0.07%"]) {
      expect(everything, `"${stale}" is a claim about the old models`).not.toContain(stale);
    }
  });
});
