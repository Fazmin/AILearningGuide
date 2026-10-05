import { describe, expect, it } from "vitest";
import shipped from "./assets/mnist-diffusion-schedule.json";
import metadata from "./assets/mnist-models.metadata.json";
import {
  backgroundShare,
  cosineSchedule,
  ddpmStep,
  forwardNoise,
  idealDenoise,
  meanSquare,
  meanSquaredError,
  normalStream,
  PIXELS,
  predictCleanImage,
  reverseStart,
  runIdealReverse,
  scheduleFromAlphaBar,
  stepCoefficients,
  STEPS,
  timestepForCall,
} from "./diffusion";

const schedule = scheduleFromAlphaBar(shipped.alpha_cumulative);

const blob = (centre: number) =>
  Float32Array.from({ length: PIXELS }, (_, index) => {
    const x = index % 28;
    const y = Math.floor(index / 28);
    return Math.hypot(x - centre, y - 14) < 6 ? 1 : -1;
  });

describe("shipped diffusion schedule", () => {
  it("is the 100-step cosine schedule that the metadata describes", () => {
    const { steps, schedule: recipe } = metadata.diffusion;
    expect(shipped.steps).toBe(STEPS);
    expect(steps).toBe(STEPS);
    expect(shipped.alpha_cumulative).toHaveLength(STEPS);
    expect(recipe.kind).toBe("cosine");
    const cosine = cosineSchedule(STEPS, recipe.offset, recipe.curve_fraction);
    for (let t = 0; t < STEPS; t += 1) {
      // The JSON stores float32 values, hence the loose precision on beta (about 1e-4 early on).
      expect(schedule.alphaBar[t]).toBeCloseTo(cosine.alphaBar[t], 6);
      expect(schedule.beta[t]).toBeCloseTo(cosine.beta[t], 4);
    }
  });

  it("falls monotonically and ends at nearly pure noise: sqrt(alpha-bar) is 0.031 at the last step", () => {
    for (let t = 1; t < STEPS; t += 1) expect(schedule.alphaBar[t]).toBeLessThan(schedule.alphaBar[t - 1]);
    expect(schedule.alphaBar[0]).toBeGreaterThan(0.999);
    const last = schedule.alphaBar[STEPS - 1];
    expect(last).toBeCloseTo(0.000971, 6);
    expect(Math.sqrt(last)).toBeCloseTo(0.0312, 4);
    expect(Math.sqrt(1 - last)).toBeCloseTo(0.9995, 4);
    expect(metadata.diffusion.schedule.final_signal).toBeCloseTo(Math.sqrt(last), 6);
    expect(metadata.diffusion.schedule.final_noise).toBeCloseTo(Math.sqrt(1 - last), 6);
    // The old linear schedule left sqrt(alpha-bar) = 0.603 here.
    expect(Math.sqrt(last)).toBeLessThan(0.05);
  });

  it("takes small steps early and large ones late, up to beta = 0.55", () => {
    expect(schedule.beta[0]).toBeCloseTo(0.0006, 4);
    expect(schedule.beta[STEPS - 1]).toBeCloseTo(0.549, 3);
    expect(Math.max(...schedule.beta)).toBe(schedule.beta[STEPS - 1]);
  });
});

describe("background share", () => {
  it("separates noise (about 0.18) from a digit-like image, where pixel RMS (about 1.0 for both) does not", () => {
    const noise = normalStream(21).fill(100_000);
    expect(backgroundShare(noise)).toBeCloseTo(0.184, 2);
    const digit = blob(14);
    expect(backgroundShare(digit)).toBeGreaterThan(0.8);
    expect(Math.sqrt(meanSquare(noise))).toBeCloseTo(1, 1);
    expect(Math.sqrt(meanSquare(digit))).toBeCloseTo(1, 1);
    expect(backgroundShare([-1, -0.9, -0.89, 0])).toBe(0.5);
  });
});

describe("forward process and the DDPM update", () => {
  it("inverts the closed form exactly when the true noise is supplied", () => {
    const x0 = blob(14);
    const epsilon = normalStream(3).fill();
    for (const t of [0, 40, 99]) {
      const xt = forwardNoise(x0, epsilon, schedule.alphaBar[t]);
      const recovered = predictCleanImage(xt, epsilon, schedule.alphaBar[t]);
      expect(meanSquaredError(recovered, x0)).toBeLessThan(1e-9);
    }
  });

  it("lands on x0 at the final step when the noise prediction is perfect", () => {
    const x0 = blob(12);
    const epsilon = normalStream(8).fill();
    const x = forwardNoise(x0, epsilon, schedule.alphaBar[0]);
    const out = ddpmStep(x, epsilon, 0, schedule, null);
    expect(meanSquaredError(out, x0)).toBeLessThan(1e-9);
    expect(stepCoefficients(schedule, 0).sigma).toBe(0);
    expect(stepCoefficients(schedule, 50).sigma).toBeCloseTo(Math.sqrt(schedule.beta[50]), 12);
  });

  it("numbers denoiser calls from t = 99 down to t = 0", () => {
    expect(timestepForCall(0)).toBe(99);
    expect(timestepForCall(99)).toBe(0);
  });

  it("draws reproducible standard-normal noise", () => {
    const a = normalStream(11).fill(20000);
    const b = normalStream(11).fill(20000);
    expect(Array.from(a.slice(0, 5))).toEqual(Array.from(b.slice(0, 5)));
    const mean = a.reduce((sum, value) => sum + value, 0) / a.length;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(meanSquare(a)).toBeGreaterThan(0.96);
    expect(meanSquare(a)).toBeLessThan(1.04);
    expect(normalStream(12).fill(3)[0]).not.toBe(a[0]);
  });
});

describe("ideal (posterior-mean) denoiser", () => {
  const references = [blob(9), blob(14), blob(19)];

  it("returns weights that sum to one and concentrate on the source at low noise", () => {
    const epsilon = normalStream(4).fill();
    const xt = forwardNoise(references[2], epsilon, schedule.alphaBar[5]);
    const { weights, x0 } = idealDenoise(xt, 5, references, schedule);
    expect(weights.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(weights[2]).toBeGreaterThan(0.999);
    expect(meanSquaredError(x0, references[2])).toBeLessThan(1e-4);
  });

  it("can only reproduce one of its references when run through all 100 steps", () => {
    const start = reverseStart(5, "digit", references[0], schedule);
    const run = runIdealReverse(start, references, schedule);
    expect(run.states).toHaveLength(STEPS + 1);
    expect(run.cleanPredictions).toHaveLength(STEPS);
    const final = run.states[STEPS];
    const distances = references.map((reference) => meanSquaredError(final, reference));
    expect(Math.min(...distances)).toBeLessThan(1e-3);
  });

  it("picks its reference from the noise, not from x0: the schedule leaves only 3% of x0 to start from", () => {
    // With a schedule that ends at almost pure noise, starting from a noised copy of references[0]
    // does not pull the run toward it; the seed decides. (A schedule that kept 60% would.)
    const ends = [1, 2, 3, 4, 5, 6].map((seed) => {
      const run = runIdealReverse(reverseStart(seed, "digit", references[0], schedule), references, schedule);
      const distances = references.map((reference) => meanSquaredError(run.states[STEPS], reference));
      return distances.indexOf(Math.min(...distances));
    });
    expect(new Set(ends).size).toBeGreaterThan(1);
    const pure = reverseStart(3, "noise", references[0], schedule);
    const noised = reverseStart(3, "digit", references[0], schedule);
    // The two starts differ only by sqrt(alpha-bar) x0, a few percent of one pixel's range.
    expect(meanSquaredError(pure.x, noised.x)).toBeLessThan(0.001 * 1.1);
  });

  it("starts pure-noise runs from the seed alone and digit runs from q(x_99 | x0)", () => {
    const noise = reverseStart(2, "noise", references[1], schedule);
    const digit = reverseStart(2, "digit", references[1], schedule);
    const expected = forwardNoise(references[1], noise.x, schedule.alphaBar[STEPS - 1]);
    expect(meanSquaredError(digit.x, expected)).toBeLessThan(1e-12);
    expect(meanSquaredError(noise.z[0], digit.z[0])).toBe(0);
  });
});
