import { useId } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import { HAND_GRID, HAND_LIMIT, SMALL_NAMES, SMALL_ROLES, X_RANGE, Y_RANGE } from "./model";
import { knobLayout, type Dots } from "./net";
import { Knob, MiniKnob, count, curvePath, fmt, linear, signed } from "./kit";

const clampY = (value: number) => Math.max(Y_RANGE[0] - 1, Math.min(Y_RANGE[1] + 1, Number.isFinite(value) ? value : 0));

/* ------------------------------------------------------------------ */
/* The dots and the machine's curve                                    */
/* ------------------------------------------------------------------ */

export interface Piece {
  name: string;
  rule: (x: number) => number;
}

/**
 * The picture every card shares: the dots to match, the curve the machine draws at its current knob
 * settings, and a thin line from each dot to the curve showing how far off the machine is there.
 */
export function FitPlot({
  dots,
  curve,
  pieces = [],
  probe = null,
  label,
  curveName = "machine's output",
  tall = false,
}: {
  dots: Dots;
  curve: (x: number) => number;
  /** Each hidden neuron's share of the output, drawn thin underneath the curve. */
  pieces?: ReadonlyArray<Piece>;
  /** An input to trace: a marker rides the curve there. */
  probe?: number | null;
  /** The full reading of the picture, for a screen reader. */
  label: string;
  curveName?: string;
  tall?: boolean;
}) {
  const id = useId();
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(360);
  const width = Math.max(240, measured);
  const height = Math.round(Math.max(210, Math.min(tall ? 340 : 300, width * (tall ? 0.6 : 0.7))));
  const left = 30;
  const right = width - 10;
  const top = 12;
  const bottom = height - 26;
  const x = linear(X_RANGE[0], X_RANGE[1], left, right);
  const y = (value: number) => linear(Y_RANGE[0], Y_RANGE[1], bottom, top)(clampY(value));
  const ticks = [-2, -1, 0, 1, 2];
  return (
    <div className="nk-frame" ref={ref}>
      <svg className="nk-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        <defs>
          <clipPath id={`${id}-clip`}>
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {ticks.map((tick) => (
          <g key={`x${tick}`}>
            <line className={tick === 0 ? "nk-axis" : "nk-grid"} x1={x(tick)} x2={x(tick)} y1={top} y2={bottom} />
            <text className="nk-tick" x={x(tick)} y={bottom + 14} textAnchor="middle">
              {fmt(tick, 0)}
            </text>
          </g>
        ))}
        {ticks.map((tick) => (
          <g key={`y${tick}`}>
            <line className={tick === 0 ? "nk-axis" : "nk-grid"} x1={left} x2={right} y1={y(tick)} y2={y(tick)} />
            <text className="nk-tick" x={left - 6} y={y(tick) + 3.5} textAnchor="end">
              {fmt(tick, 0)}
            </text>
          </g>
        ))}
        <text className="nk-axis-label" x={right} y={height - 2} textAnchor="end">
          input x →
        </text>
        <g clipPath={`url(#${id}-clip)`}>
          {pieces.map((piece, index) => (
            <path
              key={piece.name}
              className={`nk-piece nk-piece--${index % 6}`}
              d={curvePath(X_RANGE[0], X_RANGE[1], 80, x, y, piece.rule)}
            />
          ))}
          {dots.xs.map((dotX, index) => (
            <line
              key={`gap${dotX}`}
              className="nk-gap"
              x1={x(dotX)}
              x2={x(dotX)}
              y1={y(dots.ys[index])}
              y2={y(curve(dotX))}
            />
          ))}
          <path className="nk-curve" d={curvePath(X_RANGE[0], X_RANGE[1], 96, x, y, curve)} />
        </g>
        {dots.xs.map((dotX, index) => (
          <circle key={`dot${dotX}`} className="nk-dot" cx={x(dotX)} cy={y(dots.ys[index])} r={4} />
        ))}
        {probe !== null && (
          <g>
            <line className="nk-probe-line" x1={x(probe)} x2={x(probe)} y1={bottom} y2={y(curve(probe))} />
            <circle className="nk-probe" cx={x(probe)} cy={y(curve(probe))} r={5.5} />
          </g>
        )}
      </svg>
      <ul className="nk-legend">
        <li>
          <i className="nk-key nk-key--dot" aria-hidden="true" />
          dots: the right answers
        </li>
        <li>
          <i className="nk-key nk-key--curve" aria-hidden="true" />
          {curveName}
        </li>
        <li>
          <i className="nk-key nk-key--gap" aria-hidden="true" />
          gap at each dot
        </li>
        {pieces.length > 0 && (
          <li>
            <i className="nk-key nk-key--piece" aria-hidden="true" />
            thin lines: one hidden neuron&apos;s share each
          </li>
        )}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card 1: the two-knob machine                                        */
/* ------------------------------------------------------------------ */

/** The number's path through the smallest machine: in, times w, plus b, out. */
export function LineMachine({
  w,
  b,
  x,
  onW,
  onB,
}: {
  w: number;
  b: number;
  x: number;
  onW: (value: number) => void;
  onB: (value: number) => void;
}) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(340);
  const width = Math.max(280, measured);
  const height = 190;
  const mid = 92;
  const radius = 24;
  const chip = { w: Math.min(74, width * 0.2), h: 40 };
  const at = (share: number) => 6 + chip.w / 2 + (width - 12 - chip.w) * share;
  const stops = [at(0), at(0.25), at(0.5), at(0.75), at(1)];
  const scaled = w * x;
  const out = scaled + b;
  const chips = [
    { cx: stops[0], title: "input x", value: fmt(x) },
    { cx: stops[2], title: "w × x", value: fmt(scaled) },
    { cx: stops[4], title: "output", value: fmt(out) },
  ];
  return (
    <div className="nk-frame" ref={ref}>
      <svg
        className="nk-plot nk-machine"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label={`The machine. The input ${fmt(x)} is multiplied by knob w, ${signed(w)}, giving ${fmt(scaled)}. Then knob b, ${signed(
          b,
        )}, is added, giving the output ${fmt(out)}.`}
      >
        <line className="nk-wire" x1={stops[0]} x2={stops[4]} y1={mid} y2={mid} />
        {chips.map((item) => (
          <g key={item.title}>
            <rect className="nk-chip" x={item.cx - chip.w / 2} y={mid - chip.h / 2} width={chip.w} height={chip.h} rx={7} />
            <text className="nk-chip__title" x={item.cx} y={mid - 4} textAnchor="middle">
              {item.title}
            </text>
            <text className="nk-chip__value" x={item.cx} y={mid + 13} textAnchor="middle">
              {item.value}
            </text>
          </g>
        ))}
        <text className="nk-op" x={stops[1]} y={mid - radius - 26} textAnchor="middle">
          multiply by
        </text>
        <text className="nk-op" x={stops[3]} y={mid - radius - 26} textAnchor="middle">
          then add
        </text>
        <Knob
          cx={stops[1]}
          cy={mid}
          radius={radius}
          value={w}
          limit={HAND_LIMIT}
          grid={HAND_GRID}
          name="w"
          label="Knob w, on the machine: the input is multiplied by it"
          viewWidth={width}
          onChange={onW}
        />
        <Knob
          cx={stops[3]}
          cy={mid}
          radius={radius}
          value={b}
          limit={HAND_LIMIT}
          grid={HAND_GRID}
          name="b"
          label="Knob b, on the machine: it is added at the end"
          viewWidth={width}
          onChange={onB}
        />
        <text className="nk-tick" x={width / 2} y={height - 8} textAnchor="middle">
          drag around a knob to turn it · straight up is 0
        </text>
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Cards 2 and 3: seven knobs wired into a network                     */
/* ------------------------------------------------------------------ */

/**
 * One input, two hidden neurons, one output, with a knob on every wire and one on every neuron. Each
 * hidden neuron's box draws the S-shaped curve it currently sends on.
 */
export function SmallNetDiagram({
  knobs,
  pick,
  slopes,
  slopeScale = 1,
  onPick,
  onTurn,
}: {
  knobs: readonly number[];
  pick: number;
  /** The loss's slope along each knob; draws the turn hints. */
  slopes?: readonly number[];
  slopeScale?: number;
  onPick: (index: number) => void;
  onTurn: (index: number, value: number) => void;
}) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(340);
  const width = Math.max(290, Math.min(460, measured));
  const height = 344;
  const radius = 15;
  const xIn = 32;
  const xHid = width / 2;
  const xOut = width - 44;
  const yTop = 112;
  const yBot = 236;
  const yMid = (yTop + yBot) / 2;
  const box = { w: 56, h: 40 };
  const wire = (x0: number, y0: number, x1: number, y1: number) => ({ x0, y0, x1, y1, mx: (x0 + x1) / 2, my: (y0 + y1) / 2 });
  const wires = [
    wire(xIn + 15, yMid, xHid - box.w / 2, yTop),
    wire(xIn + 15, yMid, xHid - box.w / 2, yBot),
    wire(xHid + box.w / 2, yTop, xOut - 15, yMid),
    wire(xHid + box.w / 2, yBot, xOut - 15, yMid),
  ];
  // Where each of the seven knobs sits, and where its label goes so no wire runs through it.
  const places = [
    { x: wires[0].mx, y: wires[0].my, text: { x: wires[0].mx + 4, y: wires[0].my - radius - 13, anchor: "end" as const } },
    { x: xHid, y: yTop - 58, text: { x: xHid + radius + 12, y: yTop - 54, anchor: "start" as const } },
    { x: wires[1].mx, y: wires[1].my, text: { x: wires[1].mx + 4, y: wires[1].my + radius + 22, anchor: "end" as const } },
    { x: xHid, y: yBot + 58, text: { x: xHid + radius + 12, y: yBot + 62, anchor: "start" as const } },
    { x: wires[2].mx, y: wires[2].my, text: { x: xOut - 9, y: wires[2].my - radius - 13, anchor: "end" as const } },
    { x: wires[3].mx, y: wires[3].my, text: { x: wires[3].mx - 4, y: wires[3].my + radius + 22, anchor: "start" as const } },
    { x: xOut, y: yMid - 86, text: { x: xOut, y: yMid - 86 - radius - 13, anchor: "middle" as const } },
  ];
  const neurons = [
    { cy: yTop, w: knobs[0], b: knobs[1], name: "neuron 1", nameY: yTop + box.h / 2 + 14 },
    { cy: yBot, w: knobs[2], b: knobs[3], name: "neuron 2", nameY: yBot - box.h / 2 - 7 },
  ];
  const sx = linear(X_RANGE[0], X_RANGE[1], -box.w / 2 + 5, box.w / 2 - 5);
  const sy = linear(-1, 1, box.h / 2 - 6, -box.h / 2 + 6);
  return (
    <div className="nk-frame" ref={ref}>
      <svg
        className="nk-plot nk-net"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label={`The network: one input, two hidden neurons, one output, and seven knobs. ${SMALL_NAMES.map(
          (name, index) => `${name} is ${signed(knobs[index])}`,
        ).join(", ")}.`}
      >
        {wires.map((line, index) => (
          <line key={index} className="nk-wire" x1={line.x0} y1={line.y0} x2={line.x1} y2={line.y1} />
        ))}
        <line className="nk-wire nk-wire--stub" x1={xHid} y1={yTop - 58 + radius} x2={xHid} y2={yTop - box.h / 2} />
        <line className="nk-wire nk-wire--stub" x1={xHid} y1={yBot + 58 - radius} x2={xHid} y2={yBot + box.h / 2} />
        <line className="nk-wire nk-wire--stub" x1={xOut} y1={yMid - 86 + radius} x2={xOut} y2={yMid - 15} />

        <circle className="nk-node" cx={xIn} cy={yMid} r={15} />
        <text className="nk-node__text" x={xIn} y={yMid + 4} textAnchor="middle">
          x
        </text>
        <text className="nk-tick" x={xIn} y={yMid + 31} textAnchor="middle">
          input
        </text>

        {neurons.map((neuron) => (
          <g key={neuron.name} transform={`translate(${xHid} ${neuron.cy})`}>
            <rect className="nk-node" x={-box.w / 2} y={-box.h / 2} width={box.w} height={box.h} rx={8} />
            <line className="nk-grid" x1={-box.w / 2 + 5} x2={box.w / 2 - 5} y1={0} y2={0} />
            <path className="nk-spark" d={curvePath(X_RANGE[0], X_RANGE[1], 32, sx, sy, (x) => Math.tanh(neuron.w * x + neuron.b))} />
          </g>
        ))}
        {neurons.map((neuron) => (
          <text key={`${neuron.name}-name`} className="nk-tick" x={xHid} y={neuron.nameY} textAnchor="middle">
            {neuron.name}
          </text>
        ))}

        <circle className="nk-node" cx={xOut} cy={yMid} r={15} />
        <text className="nk-node__text" x={xOut} y={yMid + 4} textAnchor="middle">
          Σ
        </text>
        <text className="nk-tick" x={xOut} y={yMid + 31} textAnchor="middle">
          output
        </text>

        {places.map((place, index) => (
          <Knob
            key={SMALL_NAMES[index]}
            cx={place.x}
            cy={place.y}
            radius={radius}
            value={knobs[index]}
            limit={HAND_LIMIT}
            grid={HAND_GRID}
            name={SMALL_NAMES[index]}
            label={`Knob ${SMALL_NAMES[index]}, ${SMALL_ROLES[index]}`}
            textAt={place.text}
            selected={pick === index}
            slope={slopes?.[index]}
            slopeScale={slopeScale}
            viewWidth={width}
            onSelect={() => onPick(index)}
            onChange={(value) => onTurn(index, value)}
          />
        ))}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Cards 4 to 6: sketches of bigger networks                           */
/* ------------------------------------------------------------------ */

/**
 * Layers as columns of circles with every wire drawn. A layer too tall to draw shows a few circles
 * around a gap, and the true count is printed under every column.
 */
export function NetSketch({ sizes, label, cap = 7 }: { sizes: readonly number[]; label: string; cap?: number }) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(340);
  const width = Math.max(260, measured);
  const height = 178;
  const top = 14;
  const bottom = height - 30;
  const columns = sizes.map((size, index) => {
    const cx = sizes.length === 1 ? width / 2 : 22 + ((width - 44) * index) / (sizes.length - 1);
    const shown = Math.min(size, cap);
    const folded = size > cap;
    const pitch = Math.min(22, (bottom - top) / Math.max(1, shown));
    const start = (top + bottom) / 2 - (pitch * (shown - 1)) / 2;
    const nodes = Array.from({ length: shown }, (_, node) => ({ cy: start + pitch * node, gap: folded && node === Math.floor(shown / 2) }));
    return { cx, size, nodes, radius: Math.min(6.5, pitch * 0.36) };
  });
  return (
    <div className="nk-frame" ref={ref}>
      <svg className="nk-plot nk-sketch" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
        {columns.slice(1).map((column, index) =>
          columns[index].nodes
            .filter((from) => !from.gap)
            .flatMap((from, a) =>
              column.nodes
                .filter((to) => !to.gap)
                .map((to, b) => (
                  <line key={`${index}-${a}-${b}`} className="nk-thread" x1={columns[index].cx} y1={from.cy} x2={column.cx} y2={to.cy} />
                )),
            ),
        )}
        {columns.map((column, index) => (
          <g key={index}>
            {column.nodes.map((node, at) =>
              node.gap ? (
                <text key={at} className="nk-fold" x={column.cx} y={node.cy + 4} textAnchor="middle">
                  ⋮
                </text>
              ) : (
                <circle key={at} className="nk-bead" cx={column.cx} cy={node.cy} r={column.radius} />
              ),
            )}
            <text className="nk-tick" x={column.cx} y={height - 8} textAnchor={index === 0 ? "start" : index === columns.length - 1 ? "end" : "middle"} dx={index === 0 ? -14 : index === columns.length - 1 ? 14 : 0}>
              {index === 0 ? `${count(column.size)} in` : index === columns.length - 1 ? `${count(column.size)} out` : count(column.size)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/** Every knob of a network as a small dial, grouped by the layer of wires it belongs to. */
export function KnobWall({ sizes, knobs, label }: { sizes: readonly number[]; knobs: readonly number[]; label: string }) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(340);
  const width = Math.max(240, measured);
  const layout = knobLayout(sizes);
  // A small network gets larger dials; a wall of sixty keeps them compact.
  const radius = layout.length <= 19 ? 12 : layout.length <= 33 ? 10 : 8;
  const pitch = radius * 2 + 6;
  const perRow = Math.max(4, Math.floor((width - 12) / pitch));
  const groups: Array<{ title: string; y: number; dials: Array<{ cx: number; cy: number; value: number }> }> = [];
  let cursor = 0;
  let y = 4;
  for (let layer = 1; layer < sizes.length; layer += 1) {
    const members = layout.map((place, index) => ({ place, index })).filter((entry) => entry.place.layer === layer);
    const title =
      layer === sizes.length - 1
        ? `into the output · ${members.length} knobs`
        : `into hidden layer ${layer} · ${members.length} knobs`;
    const titleY = y + 11;
    y += 18;
    const dials = members.map((entry, position) => ({
      cx: 6 + radius + (position % perRow) * pitch,
      cy: y + radius + Math.floor(position / perRow) * pitch,
      value: knobs[entry.index] ?? 0,
    }));
    y += Math.ceil(members.length / perRow) * pitch + 4;
    groups.push({ title, y: titleY, dials });
    cursor += members.length;
  }
  return (
    <div className="nk-frame" ref={ref}>
      <svg className="nk-plot nk-wall" viewBox={`0 0 ${width} ${y + 2}`} role="img" aria-label={`${label} ${count(cursor)} knobs in all.`}>
        {groups.map((group) => (
          <g key={group.title}>
            <text className="nk-tick" x={6} y={group.y}>
              {group.title}
            </text>
            {group.dials.map((dial, index) => (
              <MiniKnob key={index} cx={dial.cx} cy={dial.cy} radius={radius} value={dial.value} soften={3} />
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}
