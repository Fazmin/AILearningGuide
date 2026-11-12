/**
 * Marching squares over a rectangle: the line segments along which `value(x, y)` equals `level`.
 * Returns a flat list, four numbers per segment: x1, y1, x2, y2. Each crossing is placed by linear
 * interpolation along a cell edge.
 */
export function contourLines(
  value: (x: number, y: number) => number,
  level: number,
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  cells = 56,
): number[] {
  const stepX = (xRange[1] - xRange[0]) / cells;
  const stepY = (yRange[1] - yRange[0]) / cells;
  const grid: number[][] = [];
  for (let row = 0; row <= cells; row += 1) {
    const line: number[] = [];
    for (let column = 0; column <= cells; column += 1) {
      line.push(value(xRange[0] + column * stepX, yRange[0] + row * stepY) - level);
    }
    grid.push(line);
  }
  const out: number[] = [];
  const cross = (x1: number, y1: number, v1: number, x2: number, y2: number, v2: number): [number, number] => {
    const share = v1 / (v1 - v2);
    return [x1 + (x2 - x1) * share, y1 + (y2 - y1) * share];
  };
  for (let row = 0; row < cells; row += 1) {
    for (let column = 0; column < cells; column += 1) {
      const x0 = xRange[0] + column * stepX;
      const y0 = yRange[0] + row * stepY;
      const x1 = x0 + stepX;
      const y1 = y0 + stepY;
      const corners = [grid[row][column], grid[row][column + 1], grid[row + 1][column + 1], grid[row + 1][column]];
      const edges: [number, number][] = [];
      if (corners[0] > 0 !== corners[1] > 0) edges.push(cross(x0, y0, corners[0], x1, y0, corners[1]));
      if (corners[1] > 0 !== corners[2] > 0) edges.push(cross(x1, y0, corners[1], x1, y1, corners[2]));
      if (corners[2] > 0 !== corners[3] > 0) edges.push(cross(x1, y1, corners[2], x0, y1, corners[3]));
      if (corners[3] > 0 !== corners[0] > 0) edges.push(cross(x0, y1, corners[3], x0, y0, corners[0]));
      if (edges.length === 2) {
        out.push(edges[0][0], edges[0][1], edges[1][0], edges[1][1]);
      } else if (edges.length === 4) {
        // A saddle cell: pair the crossings so the two lines do not cross inside the cell.
        const centre = (corners[0] + corners[1] + corners[2] + corners[3]) / 4;
        const pairs = centre > 0 === corners[0] > 0 ? [[0, 1], [2, 3]] : [[0, 3], [1, 2]];
        for (const [from, to] of pairs) out.push(edges[from][0], edges[from][1], edges[to][0], edges[to][1]);
      }
    }
  }
  return out;
}
