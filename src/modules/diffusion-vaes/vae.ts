/**
 * VAE bookkeeping for the shipped two-dimensional MNIST VAE.
 *
 * Training (models/train_mnist_models.py) minimizes, per image,
 *   binary cross-entropy summed over 784 pixels + 0.35 x KL(q(z|x) || N(0, I)).
 * A KL weight below 1 makes this a beta-VAE objective rather than the exact
 * negative ELBO, which is why the lab prints both terms separately.
 */

export const IMAGE_SIDE = 28;
export const PIXELS = IMAGE_SIDE * IMAGE_SIDE;
export const KL_WEIGHT = 0.35;

/** The latent map: MAP_GRID x MAP_GRID tiles over [MAP_MIN, MAP_MAX] on each axis. */
export const MAP_GRID = 16;
export const MAP_MIN = -4;
export const MAP_MAX = 4;

/** PyTorch clamps each log term of binary_cross_entropy at -100. */
const safeLog = (value: number) => Math.max(-100, Math.log(value));

/** -sum [x log x-hat + (1 - x) log(1 - x-hat)], in nats per image. */
export function binaryCrossEntropy(target: ArrayLike<number>, reconstruction: ArrayLike<number>) {
  let total = 0;
  for (let index = 0; index < target.length; index += 1) {
    const x = target[index];
    const p = reconstruction[index];
    total -= x * safeLog(p) + (1 - x) * safeLog(1 - p);
  }
  return total;
}

/** KL(N(mu, diag(exp(logVar))) || N(0, I)) = -1/2 sum (1 + logVar - mu^2 - exp(logVar)). */
export function gaussianKl(mean: ArrayLike<number>, logVariance: ArrayLike<number>) {
  let total = 0;
  for (let index = 0; index < mean.length; index += 1) {
    total += 1 + logVariance[index] - mean[index] ** 2 - Math.exp(logVariance[index]);
  }
  return -0.5 * total;
}

/** log N(z; 0, I) for a 2-D latent, in nats. */
export function priorLogDensity(z: ReadonlyArray<number>) {
  const squared = z.reduce((sum, value) => sum + value * value, 0);
  return -0.5 * squared - (z.length / 2) * Math.log(2 * Math.PI);
}

/** Latent coordinates for an n x n mosaic, row-major from the top (largest z2) down. */
export function latentGrid(count: number, minimum: number, maximum: number) {
  const values: number[] = [];
  const span = maximum - minimum;
  for (let row = 0; row < count; row += 1) {
    const z2 = maximum - ((row + 0.5) / count) * span;
    for (let column = 0; column < count; column += 1) {
      const z1 = minimum + ((column + 0.5) / count) * span;
      values.push(z1, z2);
    }
  }
  return values;
}

/**
 * Latents for the ideal denoiser's reference set: an 8 x 8 grid at +-0.25,
 * +-1.25, +-2.25 and +-3.25 on each axis, row-major from the top. Every one is
 * the centre of a tile of the 16 x 16 latent map, so which digit the
 * classifier read there (mnist-models.metadata.json, vae.test_digits.map) is
 * known for all 64, and together they cover all ten digits.
 */
export function referenceLatents() {
  const offsets = [-3.25, -2.25, -1.25, -0.25, 0.25, 1.25, 2.25, 3.25];
  const values: number[] = [];
  for (let row = 0; row < offsets.length; row += 1) {
    const z2 = offsets[offsets.length - 1 - row];
    for (const z1 of offsets) values.push(z1, z2);
  }
  return values;
}

export type Point = readonly [number, number];
export type Stroke = ReadonlyArray<Point>;

function segmentDistance(px: number, py: number, a: Point, b: Point) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  const t =
    lengthSquared === 0
      ? 0
      : Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / lengthSquared));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

/**
 * Rasterize pen strokes (in 28 x 28 pixel coordinates) into a soft-edged,
 * MNIST-like image: white ink on black, values in [0, 1].
 */
export function rasterizeStrokes(strokes: ReadonlyArray<Stroke>, radius = 1.5, softness = 0.8) {
  const image = new Float32Array(PIXELS);
  for (let y = 0; y < IMAGE_SIDE; y += 1) {
    for (let x = 0; x < IMAGE_SIDE; x += 1) {
      const px = x + 0.5;
      const py = y + 0.5;
      let nearest = Infinity;
      for (const stroke of strokes) {
        if (stroke.length === 1) {
          nearest = Math.min(nearest, Math.hypot(px - stroke[0][0], py - stroke[0][1]));
        }
        for (let index = 0; index + 1 < stroke.length; index += 1) {
          nearest = Math.min(nearest, segmentDistance(px, py, stroke[index], stroke[index + 1]));
        }
      }
      image[y * IMAGE_SIDE + x] = Math.max(0, Math.min(1, (radius - nearest) / softness + 0.5));
    }
  }
  return image;
}

const ellipse = (cx: number, cy: number, rx: number, ry: number, count = 28): Point[] =>
  Array.from({ length: count + 1 }, (_, index) => {
    const angle = (2 * Math.PI * index) / count;
    return [cx + rx * Math.sin(angle), cy - ry * Math.cos(angle)] as const;
  });

/** Hand-authored pen strokes, used as keyboard-reachable inputs to the encoder. */
export const PRESET_STROKES: Record<string, ReadonlyArray<Stroke>> = {
  zero: [ellipse(14, 14, 5.5, 8.5)],
  one: [[[15.5, 5], [12.5, 23]]],
  three: [
    [
      [9, 7], [13, 5], [18, 6.5], [18, 11], [13.5, 13.5],
      [18.5, 16], [18.5, 20.5], [13, 23], [8.5, 21],
    ],
  ],
  seven: [[[8, 6.5], [20, 6.5], [12.5, 23]]],
};

export const PRESET_LABELS: Record<string, string> = {
  zero: "0",
  one: "1",
  three: "3",
  seven: "7",
};

export function inkTotal(image: ArrayLike<number>) {
  let total = 0;
  for (let index = 0; index < image.length; index += 1) total += image[index];
  return total;
}
