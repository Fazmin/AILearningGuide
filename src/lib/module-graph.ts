/**
 * The prerequisite links between labs, as a graph: what a lab builds on, what builds on it, and a
 * top-to-bottom layout in which every link points down. These are pure functions over
 * `{ id, order, prerequisites }` so they stay testable without the registry.
 */
export interface GraphModule {
  id: string;
  order: number;
  prerequisites: string[];
}

export interface ModuleGraph {
  /** Every lab id, in learning-map order. */
  ids: string[];
  order: ReadonlyMap<string, number>;
  /** Direct prerequisites of each lab in learning-map order. Unknown ids are dropped. */
  prerequisites: ReadonlyMap<string, string[]>;
  /** The labs that list each lab as a direct prerequisite, in learning-map order. */
  dependents: ReadonlyMap<string, string[]>;
}

export function buildGraph(modules: readonly GraphModule[]): ModuleGraph {
  const sorted = [...modules].sort((a, b) => a.order - b.order);
  const order = new Map(sorted.map((module) => [module.id, module.order]));
  const prerequisites = new Map<string, string[]>();
  const dependents = new Map<string, string[]>(sorted.map((module) => [module.id, []]));

  for (const module of sorted) {
    const known = [...new Set(module.prerequisites)]
      .filter((id) => id !== module.id && order.has(id))
      .sort((a, b) => order.get(a)! - order.get(b)!);
    prerequisites.set(module.id, known);
    for (const id of known) dependents.get(id)!.push(module.id);
  }

  return { ids: sorted.map((module) => module.id), order, prerequisites, dependents };
}

function reach(start: string, next: ReadonlyMap<string, string[]>, order: ReadonlyMap<string, number>) {
  const seen = new Set<string>();
  const stack = [...(next.get(start) ?? [])];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (id === start || seen.has(id)) continue;
    seen.add(id);
    stack.push(...(next.get(id) ?? []));
  }
  return [...seen].sort((a, b) => order.get(a)! - order.get(b)!);
}

export interface Focus {
  id: string;
  /** Everything the lab builds on, directly or not, in learning-map order. */
  before: string[];
  /** Everything that builds on the lab, directly or not, in learning-map order. */
  after: string[];
  directBefore: string[];
  directAfter: string[];
  beforeSet: ReadonlySet<string>;
  afterSet: ReadonlySet<string>;
}

/** How one lab connects to the rest of the guide. */
export function focusOf(graph: ModuleGraph, id: string): Focus {
  const before = reach(id, graph.prerequisites, graph.order);
  const after = reach(id, graph.dependents, graph.order);
  return {
    id,
    before,
    after,
    directBefore: graph.prerequisites.get(id) ?? [],
    directAfter: graph.dependents.get(id) ?? [],
    beforeSet: new Set(before),
    afterSet: new Set(after),
  };
}

export interface LayoutOptions {
  nodeWidth: number;
  nodeHeight: number;
  /** Space between neighbouring labs on a row. */
  nodeGap: number;
  /** Space between rows, where links cross over. */
  rowGap: number;
  /** Width reserved on a row for a link passing through it, and the space around that lane. */
  laneWidth: number;
  laneGap: number;
  margin: number;
}

export const defaultLayoutOptions: LayoutOptions = {
  nodeWidth: 10.5,
  nodeHeight: 3.4,
  nodeGap: 1.1,
  rowGap: 2.6,
  laneWidth: 0.5,
  laneGap: 0.7,
  margin: 1.5,
};

export interface Point {
  x: number;
  y: number;
}

export interface PlacedNode {
  id: string;
  row: number;
  /** Top-left corner. */
  x: number;
  y: number;
}

export interface PlacedLink {
  /** The prerequisite the link leaves. */
  from: string;
  /** The lab that builds on it. */
  to: string;
  /** Leaves the bottom of `from`, passes straight down any lane it crosses, enters the top of `to`. */
  points: Point[];
}

export interface GraphLayout {
  width: number;
  height: number;
  rows: number;
  nodes: PlacedNode[];
  links: PlacedLink[];
}

/** Longest chain of prerequisites above each lab, so every lab sits below everything it builds on. */
function rowsOf(graph: ModuleGraph) {
  const rows = new Map<string, number>();
  const visiting = new Set<string>();
  const rowOf = (id: string): number => {
    const known = rows.get(id);
    if (known !== undefined) return known;
    visiting.add(id);
    let row = 0;
    for (const prerequisite of graph.prerequisites.get(id) ?? []) {
      if (!visiting.has(prerequisite)) row = Math.max(row, rowOf(prerequisite) + 1);
    }
    visiting.delete(id);
    rows.set(id, row);
    return row;
  };
  graph.ids.forEach(rowOf);

  // A lab that feeds more labs than it draws on moves down to sit just above the first of them,
  // which shortens more links than it lengthens.
  for (const id of [...graph.ids].reverse()) {
    const dependents = graph.dependents.get(id) ?? [];
    if (dependents.length <= (graph.prerequisites.get(id) ?? []).length) continue;
    const latest = Math.min(...dependents.map((dependent) => rows.get(dependent)!)) - 1;
    if (latest > rows.get(id)!) rows.set(id, latest);
  }
  return rows;
}

interface Item {
  /** The lab this item stands for, or `null` for a lane a long link passes through. */
  id: string | null;
  row: number;
  width: number;
  sortKey: number;
  up: number[];
  down: number[];
}

/**
 * Lay the graph out in rows, top to bottom. Labs sit on the row below their deepest prerequisite;
 * a link that skips rows reserves a narrow lane on each row it crosses, so it never runs behind an
 * unrelated lab. Neighbours are then ordered to cut crossings and nudged to straighten links.
 */
export function layoutGraph(
  graph: ModuleGraph,
  options: LayoutOptions = defaultLayoutOptions,
): GraphLayout {
  if (graph.ids.length === 0) {
    return { width: options.margin * 2, height: options.margin * 2, rows: 0, nodes: [], links: [] };
  }
  const rowOf = rowsOf(graph);
  const items: Item[] = [];
  const itemOfModule = new Map<string, number>();

  for (const id of graph.ids) {
    itemOfModule.set(id, items.length);
    items.push({
      id,
      row: rowOf.get(id)!,
      width: options.nodeWidth,
      sortKey: graph.order.get(id)!,
      up: [],
      down: [],
    });
  }

  const connect = (upper: number, lower: number) => {
    items[upper].down.push(lower);
    items[lower].up.push(upper);
  };

  const chains: { from: string; to: string; chain: number[] }[] = [];
  for (const to of graph.ids) {
    for (const from of graph.prerequisites.get(to) ?? []) {
      const first = rowOf.get(from)!;
      const last = rowOf.get(to)!;
      if (last <= first) continue;
      const chain = [itemOfModule.get(from)!];
      for (let row = first + 1; row < last; row += 1) {
        chain.push(items.length);
        items.push({
          id: null,
          row,
          width: options.laneWidth,
          sortKey: graph.order.get(from)! + 0.5,
          up: [],
          down: [],
        });
      }
      chain.push(itemOfModule.get(to)!);
      for (let i = 0; i < chain.length - 1; i += 1) connect(chain[i], chain[i + 1]);
      chains.push({ from, to, chain });
    }
  }

  const rowCount = Math.max(0, ...items.map((item) => item.row)) + 1;
  const rows: number[][] = Array.from({ length: rowCount }, () => []);
  items.forEach((item, index) => rows[item.row].push(index));
  rows.forEach((row) => row.sort((a, b) => items[a].sortKey - items[b].sortKey || a - b));

  orderRows(rows, items);
  const centers = placeRows(rows, items, options);

  const left = Math.min(...items.map((item, index) => centers[index] - item.width / 2));
  const shift = options.margin - left;
  const rowTop = (row: number) => options.margin + row * (options.nodeHeight + options.rowGap);
  const centerOf = (index: number) => centers[index] + shift;

  const nodes: PlacedNode[] = graph.ids.map((id) => {
    const index = itemOfModule.get(id)!;
    return { id, row: items[index].row, x: centerOf(index) - options.nodeWidth / 2, y: rowTop(items[index].row) };
  });

  const links: PlacedLink[] = chains.map(({ from, to, chain }) => {
    const points: Point[] = [];
    chain.forEach((index, position) => {
      const x = centerOf(index);
      const top = rowTop(items[index].row);
      if (position === 0) points.push({ x, y: top + options.nodeHeight });
      else if (position === chain.length - 1) points.push({ x, y: top });
      else points.push({ x, y: top }, { x, y: top + options.nodeHeight });
    });
    return { from, to, points };
  });

  const right = Math.max(...items.map((item, index) => centerOf(index) + item.width / 2));
  return {
    width: right + options.margin,
    height: rowTop(rowCount - 1) + options.nodeHeight + options.margin,
    rows: rowCount,
    nodes,
    links,
  };
}

/** Reorder each row so fewer links cross, sweeping down and up by neighbour position. */
function orderRows(rows: number[][], items: Item[]) {
  const position = new Array<number>(items.length).fill(0);
  const reindex = () => rows.forEach((row) => row.forEach((index, at) => (position[index] = at)));
  reindex();

  const sweep = (rowIndexes: number[], neighbours: (item: Item) => number[]) => {
    for (const r of rowIndexes) {
      const row = rows[r];
      const mean = (index: number) => {
        const adjacent = neighbours(items[index]);
        if (adjacent.length === 0) return position[index];
        return adjacent.reduce((sum, other) => sum + position[other], 0) / adjacent.length;
      };
      const keyed = row.map((index) => ({ index, key: mean(index), was: position[index] }));
      keyed.sort((a, b) => a.key - b.key || a.was - b.was);
      rows[r] = keyed.map((entry) => entry.index);
      rows[r].forEach((index, at) => (position[index] = at));
    }
  };

  const crossings = () => {
    let total = 0;
    for (let r = 0; r < rows.length - 1; r += 1) {
      const segments = rows[r].flatMap((upper) => items[upper].down.map((lower) => [position[upper], position[lower]]));
      for (let i = 0; i < segments.length; i += 1) {
        for (let j = i + 1; j < segments.length; j += 1) {
          if ((segments[i][0] - segments[j][0]) * (segments[i][1] - segments[j][1]) < 0) total += 1;
        }
      }
    }
    return total;
  };

  /** Crossings between the links of `left` and `right`, which are neighbours on a row. */
  const crossingsBetween = (left: number, right: number) => {
    let total = 0;
    for (const side of ["up", "down"] as const) {
      for (const a of items[left][side]) {
        for (const b of items[right][side]) if (position[a] > position[b]) total += 1;
      }
    }
    return total;
  };

  const transpose = () => {
    let improved = true;
    while (improved) {
      improved = false;
      for (const row of rows) {
        for (let at = 0; at < row.length - 1; at += 1) {
          const [left, right] = [row[at], row[at + 1]];
          const asIs = crossingsBetween(left, right);
          [position[left], position[right]] = [position[right], position[left]];
          const swapped = crossingsBetween(right, left);
          if (swapped < asIs) {
            [row[at], row[at + 1]] = [right, left];
            improved = true;
          } else {
            [position[left], position[right]] = [position[right], position[left]];
          }
        }
      }
    }
  };

  const ascending = rows.map((_, r) => r);
  const descending = [...ascending].reverse();
  let best = rows.map((row) => [...row]);
  let bestCrossings = crossings();
  let stale = 0;

  for (let pass = 0; pass < 40 && stale < 8 && bestCrossings > 0; pass += 1) {
    if (pass % 2 === 0) sweep(ascending.slice(1), (item) => item.up);
    else sweep(descending.slice(1), (item) => item.down);
    transpose();
    const total = crossings();
    if (total < bestCrossings) {
      best = rows.map((row) => [...row]);
      bestCrossings = total;
      stale = 0;
    } else {
      stale += 1;
    }
  }
  best.forEach((row, r) => (rows[r] = row));
}

/** How hard a link pulls its two ends into line; lane-to-lane links hold a long link upright. */
function pull(a: Item, b: Item) {
  if (a.id === null && b.id === null) return 8;
  return a.id === null || b.id === null ? 3 : 1;
}

/** The centre x of every item: close to its neighbours, in order, and never overlapping. */
function placeRows(rows: number[][], items: Item[], options: LayoutOptions) {
  const centers = new Array<number>(items.length).fill(0);
  const spacing = (a: Item, b: Item) =>
    (a.width + b.width) / 2 + (a.id === null || b.id === null ? options.laneGap : options.nodeGap);

  const offsetsOf = (row: number[]) => {
    const offsets: number[] = [];
    row.forEach((index, at) => {
      offsets.push(at === 0 ? 0 : offsets[at - 1] + spacing(items[row[at - 1]], items[index]));
    });
    return offsets;
  };

  for (const row of rows) {
    const offsets = offsetsOf(row);
    const middle = (offsets[offsets.length - 1] ?? 0) / 2;
    row.forEach((index, at) => (centers[index] = offsets[at] - middle));
  }

  const relax = (r: number) => {
    const row = rows[r];
    const offsets = offsetsOf(row);
    const targets: number[] = [];
    const weights: number[] = [];
    for (const index of row) {
      const item = items[index];
      let sum = 0;
      let weight = 0;
      for (const other of [...item.up, ...item.down]) {
        const w = pull(item, items[other]);
        sum += w * centers[other];
        weight += w;
      }
      targets.push(weight > 0 ? sum / weight : centers[index]);
      weights.push(weight > 0 ? weight : 1e-6);
    }
    placeInOrder(targets, weights, offsets).forEach((x, at) => (centers[row[at]] = x));
  };

  for (let pass = 0; pass < 30; pass += 1) {
    if (pass % 2 === 0) for (let r = 0; r < rows.length; r += 1) relax(r);
    else for (let r = rows.length - 1; r >= 0; r -= 1) relax(r);
  }
  return centers;
}

/**
 * Weighted least-squares placement of items that must stay in order and keep their minimum
 * spacing: shift each target by its cumulative offset, then pool neighbours that came out reversed.
 */
function placeInOrder(targets: number[], weights: number[], offsets: number[]) {
  const blocks: { weight: number; total: number; count: number }[] = [];
  targets.forEach((target, at) => {
    blocks.push({ weight: weights[at], total: weights[at] * (target - offsets[at]), count: 1 });
    while (blocks.length > 1) {
      const last = blocks[blocks.length - 1];
      const previous = blocks[blocks.length - 2];
      if (previous.total / previous.weight <= last.total / last.weight) break;
      blocks.splice(blocks.length - 2, 2, {
        weight: previous.weight + last.weight,
        total: previous.total + last.total,
        count: previous.count + last.count,
      });
    }
  });
  const placed: number[] = [];
  for (const block of blocks) {
    for (let i = 0; i < block.count; i += 1) placed.push(block.total / block.weight + offsets[placed.length]);
  }
  return placed;
}
