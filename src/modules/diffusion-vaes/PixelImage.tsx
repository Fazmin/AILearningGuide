import { useEffect, useRef } from "react";

const SIDE = 28;

/** Paint grayscale tiles (each 28 x 28, row-major) onto a canvas as an n x m mosaic. */
function paint(
  canvas: HTMLCanvasElement,
  tiles: ReadonlyArray<ArrayLike<number>>,
  columns: number,
  low: number,
  high: number,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const rows = Math.ceil(tiles.length / columns);
  const width = columns * SIDE;
  const height = rows * SIDE;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const image = context.createImageData(width, height);
  const span = high - low || 1;
  tiles.forEach((tile, tileIndex) => {
    const originX = (tileIndex % columns) * SIDE;
    const originY = Math.floor(tileIndex / columns) * SIDE;
    for (let pixel = 0; pixel < SIDE * SIDE; pixel += 1) {
      const x = originX + (pixel % SIDE);
      const y = originY + Math.floor(pixel / SIDE);
      const value = Math.max(0, Math.min(1, ((tile[pixel] ?? low) - low) / span));
      const offset = (y * width + x) * 4;
      const grey = Math.round(value * 255);
      image.data[offset] = grey;
      image.data[offset + 1] = grey;
      image.data[offset + 2] = grey;
      image.data[offset + 3] = 255;
    }
  });
  context.putImageData(image, 0, 0);
}

/**
 * One 28 x 28 image drawn at integer-pixel scale. `range` maps values to
 * black and white: [0, 1] for decoder output, [-1, 1] for diffusion pixels.
 */
export function PixelImage({
  data,
  range = [0, 1],
  size = 112,
  label,
  className = "",
}: {
  data: ArrayLike<number> | null | undefined;
  range?: readonly [number, number];
  size?: number;
  label: string;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [low, high] = range;

  useEffect(() => {
    if (!ref.current) return;
    if (!data) {
      const context = ref.current.getContext("2d");
      context?.clearRect(0, 0, SIDE, SIDE);
      return;
    }
    paint(ref.current, [data], 1, low, high);
  }, [data, low, high]);

  return (
    <canvas
      ref={ref}
      width={SIDE}
      height={SIDE}
      role="img"
      aria-label={label}
      className={`dv-pixels ${data ? "" : "is-empty"} ${className}`.trim()}
      style={{ width: size, height: size }}
    />
  );
}

/** Many tiles in one canvas, used for the latent mosaic and the reference set. */
export function PixelMosaic({
  tiles,
  columns,
  range = [0, 1],
  label,
  className = "",
}: {
  tiles: ReadonlyArray<ArrayLike<number>> | null | undefined;
  columns: number;
  range?: readonly [number, number];
  label: string;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [low, high] = range;

  useEffect(() => {
    if (!ref.current || !tiles || tiles.length === 0) return;
    paint(ref.current, tiles, columns, low, high);
  }, [tiles, columns, low, high]);

  return (
    <canvas
      ref={ref}
      width={columns * SIDE}
      height={columns * SIDE}
      role="img"
      aria-label={label}
      className={`dv-pixels dv-mosaic ${tiles ? "" : "is-empty"} ${className}`.trim()}
    />
  );
}
