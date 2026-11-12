import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  SurfaceStage,
  type StageView,
  type SurfacePath,
  type SurfacePoint,
} from "@app/module-sdk";
import {
  RAVINE_GENTLE,
  RAVINE_OPTIMIZERS,
  RAVINE_RANGE,
  RAVINE_SETTLED,
  RAVINE_STEEP,
  RAVINE_STEPS,
  RAVINE_TOP,
  SGD_LIMIT,
  ravineAt,
  ravineFloor,
  ravineLoss,
  ravineLost,
  ravineRate,
  ravineSettled,
  ravineStart,
  ravineWalk,
  type RavineOptimizer,
  type RavineRates,
  type RavineTurn,
} from "./ravine";

const TICKS = [-1, -0.5, 0, 0.5, 1];
const LOSS_TICKS = [0, 0.3, 0.6, 0.9];
const LEVELS = [0.005, 0.03, 0.1, 0.2, 0.35, 0.55, 0.8];
const HEIGHTS = [0, RAVINE_TOP] as const;
const RATIO = Math.round(RAVINE_STEEP / RAVINE_GENTLE);
const AXES = { x: "weight 1 →", y: "weight 2 →", z: "loss" };
/** From the low end of the floor and a little to one side, so both walls and the whole floor show. */
const VIEWS: Record<RavineTurn, StageView> = {
  axis: { azimuth: -58, elevation: 30, distance: 4.9, target: [0, 0.2, 0] },
  diagonal: { azimuth: -18, elevation: 32, distance: 4.9, target: [0, 0.2, 0] },
};

const lossText = (value: number) =>
  !Number.isFinite(value) || value > 999 ? "flew out" : value >= 1 ? value.toFixed(2) : value >= 0.00005 ? value.toFixed(4) : "under 0.0001";
const rateText = (value: number) => String(Number(value.toPrecision(3)));
const key = (tone: string): CSSProperties => ({ "--key": `var(${tone})` }) as CSSProperties;
const reducedMotion = () =>
  typeof document !== "undefined" &&
  (document.documentElement.dataset.motion === "reduced" ||
    (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches));

/**
 * The inside of the "A ravine in two weights" card (Explore.tsx owns the card itself, so the card
 * list stays in one file). One steep-sided ravine in two weights, and SGD, momentum and Adam
 * walking it from the same start. Turning the ravine so it runs between the weights leaves SGD and
 * momentum unchanged and takes Adam's advantage away.
 */
export function RavineCard({
  turn,
  rates,
  step,
  onChange,
  narrate,
}: {
  turn: RavineTurn;
  rates: RavineRates;
  step: number;
  onChange: (patch: { ravineTurn?: RavineTurn; ravineRates?: RavineRates; ravineStep?: number }) => void;
  narrate: (text: string) => void;
}) {
  // While the walk plays, the step lives here; it is written back to the lab's state when it stops.
  const [live, setLive] = useState<number | null>(null);
  const shown = live ?? step;
  useEffect(() => {
    if (live === null) return;
    if (live >= RAVINE_STEPS) {
      onChange({ ravineStep: RAVINE_STEPS });
      setLive(null);
      return;
    }
    const timer = window.setTimeout(() => setLive((value) => (value === null ? null : value + 1)), 70);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live]);

  const walks = useMemo(
    () =>
      RAVINE_OPTIMIZERS.map((entry) => {
        const rate = ravineRate(entry.id, rates);
        const walk = ravineWalk(entry.id, turn, rate);
        return { ...entry, rate, walk, settled: ravineSettled(walk), lost: ravineLost(walk) };
      }),
    [turn, rates],
  );
  const start = ravineStart(turn);
  const floor = ravineFloor(turn);
  const inSquare = (point: { x: number; y: number }) =>
    Math.abs(point.x) <= RAVINE_RANGE[1] && Math.abs(point.y) <= RAVINE_RANGE[1];

  const paths: SurfacePath[] = [
    { id: "floor", points: floor, tone: "--ink", width: 1.3, opacity: 0.5, dashed: true },
    ...walks.map((entry) => ({
      id: entry.id,
      points: entry.walk.slice(0, shown + 1).map((point) => ({ x: point.x, y: point.y })),
      tone: entry.tone,
      width: 2.6,
    })),
  ];
  const points: SurfacePoint[] = [
    { id: "minimum", x: 0, y: 0, kind: "cross", tone: "--ink", label: "lowest", labelTone: "is-ink" },
    { id: "start", x: start.x, y: start.y, kind: "ring", tone: "--ink", label: shown === 0 ? "start" : undefined, labelTone: "is-ink" },
    ...walks
      .map((entry) => ({ entry, at: ravineAt(entry.walk, shown) }))
      .filter(({ at }) => inSquare(at) && at.loss <= RAVINE_TOP)
      .map(({ entry, at }) => ({
        id: entry.id,
        x: at.x,
        y: at.y,
        kind: "ball" as const,
        tone: entry.tone,
        label: entry.label,
        labelTone: entry.id === "sgd" ? "is-forward" : entry.id === "momentum" ? "is-attention" : "is-gradient",
        size: 0.85,
        labelLift: entry.id === "sgd" ? 0 : entry.id === "momentum" ? 14 : 28,
      })),
  ];

  const by = (id: RavineOptimizer) => walks.find((entry) => entry.id === id) ?? walks[0];
  const steps = (value: number | null) => (value === null ? `not in ${RAVINE_STEPS} steps` : `${value} steps`);
  const note = (() => {
    const sgd = by("sgd");
    const momentum = by("momentum");
    const adam = by("adam");
    if (sgd.lost) {
      return `SGD's rate of ${rateText(sgd.rate)} is past ${rateText(SGD_LIMIT)}, the most this ravine's walls allow (2 ÷ steepness ${RAVINE_STEEP}), so each bounce across lands higher than the last and it flies out. Momentum and Adam still settle, in ${steps(momentum.settled)} and ${steps(adam.settled)}: doubling a rate is not the same risk for every optimizer.`;
    }
    if (turn === "diagonal") {
      return `SGD and momentum do exactly what they did with the ravine along weight 1, only turned: ${steps(sgd.settled)} and ${steps(momentum.settled)}. Adam ${
        adam.settled === null ? `has not settled after ${RAVINE_STEPS} steps` : `needs ${steps(adam.settled)}`
      }. It sizes its step one weight at a time, and here both weights see the same mix of steep wall and gentle floor, so there is nothing for it to tell apart.`;
    }
    if (rates === "half") {
      return `At half the rate SGD no longer bounces: it drops to the floor and then creeps along it at half the pace, ${steps(
        sgd.settled,
      )}. Momentum is slowed the same way: ${steps(momentum.settled)}. Adam's steps are about the size of its rate, so its walk is longer too: ${steps(
        adam.settled,
      )}.`;
    }
    return `SGD has one rate for both weights. It has to stay small enough for the steep walls, so after bouncing between them it creeps along the floor: ${steps(
      sgd.settled,
    )}. Momentum, at a tenth of SGD's rate, builds up to about the same step along the floor and swings from wall to wall for longer: ${steps(
      momentum.settled,
    )}. Adam divides each weight's step by the usual size of that weight's own slopes, so it moves both weights at the same steady pace, down the wall and along the floor alike: ${steps(
      adam.settled,
    )}.`;
  })();

  const play = () => {
    if (live !== null) {
      onChange({ ravineStep: live });
      setLive(null);
      narrate(`Paused at step ${live}.`);
      return;
    }
    if (reducedMotion()) {
      onChange({ ravineStep: RAVINE_STEPS });
      narrate(`All ${RAVINE_STEPS} steps taken.`);
      return;
    }
    setLive(step >= RAVINE_STEPS ? 0 : step);
    narrate("Walking the ravine.");
  };

  return (
    <>
      <SurfaceHeading
        kicker={`A picture, not a run · two weights · walls ${RATIO}× steeper than the floor`}
        title="Why one step size is hard to choose"
        aside={<span className="tg-badge">step {shown} of {RAVINE_STEPS}</span>}
      />
      <p className="op-ravine__lede">
        The race trains 900 weights, which no picture can show. This ground has two. Its loss rises steeply across and
        barely at all along the floor, the shape that makes one step size hard to choose. Three of the optimizers walk it
        from the same start, each with the update rule and the default rate it has in the race.
      </p>
      <div className="op-ravine__controls">
        <SegmentedControl
          label="Ravine runs"
          value={turn}
          options={[
            { value: "axis", label: "Along weight 1" },
            { value: "diagonal", label: "Between the weights" },
          ]}
          onChange={(picked) => {
            const value = picked as RavineTurn;
            setLive(null);
            onChange({ ravineTurn: value, ravineStep: 0 });
            narrate(value === "axis" ? "The ravine runs along weight 1. Step reset to 0." : "The ravine runs diagonally, between the two weights. Step reset to 0.");
          }}
        />
        <SegmentedControl
          label="Rates"
          value={rates}
          options={[
            { value: "half", label: "Half" },
            { value: "default", label: "Each one's own" },
            { value: "double", label: "Double" },
          ]}
          onChange={(picked) => {
            const value = picked as RavineRates;
            setLive(null);
            onChange({ ravineRates: value, ravineStep: 0 });
            narrate(`Rates: ${RAVINE_OPTIMIZERS.map((entry) => `${entry.label} ${rateText(ravineRate(entry.id, value))}`).join(", ")}. Step reset to 0.`);
          }}
        />
      </div>
      {/* Turning the ravine swaps the whole surface, so the picture is rebuilt with a view that looks up the new floor. */}
      <SurfaceStage
        key={turn}
        surfaceKey={`ravine-${turn}`}
        value={(x, y) => ravineLoss(turn, x, y)}
        xRange={RAVINE_RANGE}
        yRange={RAVINE_RANGE}
        zRange={HEIGHTS}
        levels={LEVELS}
        axes={AXES}
        xTicks={TICKS}
        yTicks={TICKS}
        zTicks={LOSS_TICKS}
        paths={paths}
        points={points}
        view={VIEWS[turn]}
        aspect={0.6}
        label={`A loss over two weights drawn as ground: a ravine ${RATIO} times steeper across than along its floor, running ${
          turn === "axis" ? "along weight 1" : "diagonally between the two weights"
        }, cut off above a loss of ${RAVINE_TOP}. Three walks start together part-way up one wall. At step ${shown}: ${walks
          .map((entry) => `${entry.label} at rate ${rateText(entry.rate)} has loss ${lossText(ravineAt(entry.walk, shown).loss)}`)
          .join("; ")}.`}
      />
      <ul className="stage3d-legend" aria-hidden="true">
        {walks.map((entry) => (
          <li key={entry.id}><i style={key(entry.tone)} />{entry.label}</li>
        ))}
        <li><i className="is-dashed" style={key("--ink")} />ravine floor</li>
        <li><i className="is-shade" />higher loss, stronger colour</li>
      </ul>
      <div className="op-ravine__transport">
        <button type="button" className="primary-action" onClick={play}>
          {live !== null ? "Pause" : shown >= RAVINE_STEPS ? "Play again" : "Play"}
        </button>
        <RangeControl
          label="Step"
          min={0}
          max={RAVINE_STEPS}
          step={1}
          value={shown}
          format={(value) => `${value} of ${RAVINE_STEPS}`}
          onChange={(value) => {
            setLive(null);
            onChange({ ravineStep: Math.round(value) });
          }}
        />
      </div>
      <div className="op-table-wrap">
        <table className="op-table">
          <caption>
            Each walk on this ground. A walk has settled once its loss stays under {RAVINE_SETTLED} for the rest of the{" "}
            {RAVINE_STEPS} steps.
          </caption>
          <thead>
            <tr>
              <th scope="col">Optimizer</th>
              <th scope="col">Rate</th>
              <th scope="col">Loss at step {shown}</th>
              <th scope="col">Settles at step</th>
            </tr>
          </thead>
          <tbody>
            {walks.map((entry) => (
              <tr key={entry.id}>
                <th scope="row">
                  <span className="op-ravine__swatch" style={key(entry.tone)} aria-hidden="true" />
                  {entry.label}
                </th>
                <td>{rateText(entry.rate)}</td>
                <td>{entry.lost && shown >= entry.walk.length - 1 ? "flew out" : lossText(ravineAt(entry.walk, shown).loss)}</td>
                <td>{entry.lost ? "never: it flies out" : entry.settled === null ? `not in ${RAVINE_STEPS}` : entry.settled}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lab-note" role="status">
        {note}
      </p>
    </>
  );
}
