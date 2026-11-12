import { useMemo } from "react";
import {
  FormulaWithValues,
  GoalMeter,
  LabSurface,
  Metric,
  PredictCheck,
  RangeControl,
  SegmentedControl,
  StepPager,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  CHAIN_NUDGE,
  MAX_WALK_STEPS,
  NUDGES,
  RACE_BUDGET,
  RACE_FLOOR,
  RACE_START,
  RACE_TARGET,
  VALLEY_WALL,
  chainReading,
  curveSlope,
  curveValue,
  flatSpots,
  nudgeRatio,
  stepDown,
  surfaceGradient,
  surfaceValue,
  walk,
  walkSurface,
} from "./curves";
import { ChainPlot } from "./ChainPlot";
import { GradientMap, RaceMap } from "./ContourMaps";
import { GradientSurface, RaceSurface } from "./SurfaceViews";
import { NudgePlot } from "./NudgePlot";
import { SlopePlot } from "./SlopePlot";
import { WalkPlot } from "./WalkPlot";
import { fmt, fmtChange, fmtExact, signed } from "./kit";
import { readState } from "./state";

/** The nudge slider runs from the smallest nudge on the left to the largest on the right. */
const NUDGE_STOPS = [...NUDGES].reverse();

const FLAT_WORDS = {
  bottom: "a bottom: the meter rises on both sides",
  top: "a top: the meter falls on both sides",
  shelf: "a shelf: the meter keeps rising on the far side",
} as const;

/** How far along the challenge is, 0 at the start and 1 on the valley floor. The square root keeps the last stretch visible. */
const raceProgress = (reading: number) => {
  const startReading = surfaceValue("valley", RACE_START.x, RACE_START.y);
  return 1 - Math.sqrt(Math.max(0, reading - RACE_FLOOR) / (startReading - RACE_FLOOR));
};

/** The lab shows one card per step, like pages. */
const CARD_COUNT = 6;

export default function Explore({ state, setState, currentStep, narrate, steps, goToStep }: ModuleContext) {
  const lab = readState(state);
  const page = Math.min(CARD_COUNT - 1, Math.max(0, Math.floor(Number.isFinite(currentStep) ? currentStep : 0)));
  const stepNames = steps ?? Array.from({ length: CARD_COUNT }, (_, index) => `step ${index + 1}`);

  /* ---------------- card 1 ---------------- */
  const meterAt = curveValue("bowl", lab.nudgeW);
  const meterNudged = curveValue("bowl", lab.nudgeW + lab.nudge);
  const meterChange = meterNudged - meterAt;
  const ratio = nudgeRatio("bowl", lab.nudgeW, lab.nudge);
  const slopeHere = curveSlope("bowl", lab.nudgeW);

  /* ---------------- card 2 ---------------- */
  const slopeMeter = curveValue(lab.slopeCurve, lab.slopeW);
  const slopeAt = curveSlope(lab.slopeCurve, lab.slopeW);
  const slopeSpots = useMemo(() => flatSpots(lab.slopeCurve), [lab.slopeCurve]);
  const onFlat = slopeSpots.find((spot) => Math.abs(spot.w - lab.slopeW) < 0.011) ?? null;
  const slopeFlat = Math.abs(slopeAt) < 0.03 || onFlat !== null;
  /** Flat spots pull the dial onto them, so the reader can park exactly where the slope is zero. */
  const moveSlopeDial = (w: number) => {
    const near = slopeSpots.find((spot) => Math.abs(spot.w - w) <= 0.026);
    setState({ slopeW: near ? Number(near.w.toFixed(2)) : w });
  };
  const slopeReading = slopeFlat
    ? `At w = ${fmt(lab.slopeW)} the slope is 0: the ground is level, so a small nudge either way barely moves the meter. ${
        onFlat ? `This flat spot is ${FLAT_WORDS[onFlat.kind]}.` : ""
      }`
    : slopeAt > 0
      ? `At w = ${fmt(lab.slopeW)} the slope is ${signed(slopeAt)}: turn the dial up a little and the meter rises about ${fmt(
          slopeAt,
        )} times as far. Uphill is to the right, so downhill is to the left.`
      : `At w = ${fmt(lab.slopeW)} the slope is ${signed(slopeAt)}: turn the dial up a little and the meter falls about ${fmt(
          -slopeAt,
        )} times as far. Downhill is to the right.`;

  /* ---------------- card 3 ---------------- */
  const mapMeter = surfaceValue(lab.surface, lab.mapW, lab.mapB);
  const gradient = surfaceGradient(lab.surface, lab.mapW, lab.mapB);
  const steepness = Math.hypot(gradient.x, gradient.y);
  const onSurface = lab.mapView === "surface";
  const legsNote =
    onSurface && lab.legs !== "both"
      ? `Only ${lab.legs} turns, so the ball can only travel along the coloured cut. Its top edge is a one-dial curve like the ones in cards 1 and 2, and its slope under the ball is ${signed(
          lab.legs === "w" ? gradient.x : gradient.y,
        )}. Move the other dial and the cut moves to a different curve.`
      : lab.legs === "w"
      ? `Only w turns, so the point can only travel along the dashed line. The slope along that line is ${signed(
          gradient.x,
        )}: the blue leg points the way w must turn to raise the meter, and it is longer where that climb is steeper.`
      : lab.legs === "b"
        ? `Only b turns, so the point travels along the dashed line. The slope along it is ${signed(
            gradient.y,
          )}, drawn as the purple leg. Each dial has its own slope at the same spot.`
        : steepness < 0.02
          ? "Both slopes are 0 here, so there is no arrow: the ground is level in every direction. Level does not have to mean lowest; check the Saddle."
          : `The two legs, tip to tail, make one arrow: the gradient. It points straight uphill and crosses the contour line at a right angle; its length, ${fmt(
              steepness,
            )}, is how steep the ground is. The dashed arrow the other way is straight downhill.`;

  /* ---------------- card 4 ---------------- */
  const chain = chainReading(lab.chainW);
  const chainBarScale = 0.7;
  const bar = (change: number) => `${Math.min(100, (Math.abs(change) / chainBarScale) * 100).toFixed(1)}%`;
  const chainNote =
    chain.slopeB < 0.2
      ? `Machine B is almost flat here (slope ${fmt(chain.slopeB, 3)}), so almost nothing of the nudge gets through, however steep machine A is. Notice what the meter reads: ${fmt(
          chain.meter,
        )}. A flat link means the dial has lost its say, not that the meter is low.`
      : "Each machine multiplies the change passing through it by its own slope. Slide the dial up to 5: A gets steeper, B goes flat, and watch what reaches the meter.";

  /* ---------------- card 5 ---------------- */
  const path = useMemo(
    () => walk(lab.walkCurve, lab.walkStart, lab.walkStep, lab.walkSteps),
    [lab.walkCurve, lab.walkStart, lab.walkStep, lab.walkSteps],
  );
  const walkW = path[path.length - 1];
  const walkMeter = curveValue(lab.walkCurve, walkW);
  const walkSlope = curveSlope(lab.walkCurve, walkW);
  const walkNext = stepDown(lab.walkCurve, walkW, lab.walkStep);
  const walkSpots = useMemo(() => flatSpots(lab.walkCurve), [lab.walkCurve]);
  const walkBottoms = walkSpots.filter((spot) => spot.kind === "bottom");
  const deepest = walkBottoms.reduce((best, spot) => (spot.value < best.value ? spot : best), walkBottoms[0]);
  const settledAt = walkBottoms.find((spot) => Math.abs(spot.w - walkW) < 0.03 && Math.abs(walkSlope) < 0.06) ?? null;
  const walkClimbs = curveValue(lab.walkCurve, walkNext) > walkMeter + 1e-9 && Math.abs(walkNext - walkW) > 0.004;
  const walkNote = (() => {
    if (lab.walkSteps === 0) {
      return "Press Take a step. The dial moves against the slope, by step size × slope, so a steeper spot gives a longer step and a level spot gives none.";
    }
    if (settledAt) {
      return walkBottoms.length > 1 && settledAt !== deepest
        ? `Settled at w = ${fmt(settledAt.w)}, where the meter reads ${fmt(settledAt.value)}. The slope here is 0, so the steps have shrunk to nothing, even though the other valley is deeper (${fmt(
            deepest.value,
          )}). The walk cannot see it from here.`
        : `Settled at w = ${fmt(settledAt.w)}, where the meter reads ${fmt(settledAt.value)}. The slope is 0, so the steps have shrunk to nothing.`;
    }
    if (walkClimbs) {
      return "This step is so long that it lands higher on the far side than where it started. The slope was right about the direction; the step size trusted it for too far.";
    }
    if (lab.walkSteps >= MAX_WALK_STEPS) return `That is ${MAX_WALK_STEPS} steps, the most this card draws. Press Start over to walk again.`;
    return "Keep stepping. Watch the steps get shorter as the ground levels out, with no change to the step size.";
  })();
  const takeSteps = (count: number) => {
    const steps = Math.min(MAX_WALK_STEPS, lab.walkSteps + count);
    setState({ walkSteps: steps });
    const end = walk(lab.walkCurve, lab.walkStart, lab.walkStep, steps).pop() ?? lab.walkStart;
    narrate(`After ${steps} steps the dial is at ${fmt(end, 3)} and the meter reads ${fmt(curveValue(lab.walkCurve, end), 3)}.`);
  };

  /* ---------------- card 6 ---------------- */
  const racePath = useMemo(() => walkSurface("valley", RACE_START, lab.raceStep, lab.raceSteps), [lab.raceStep, lab.raceSteps]);
  const raceGhost = useMemo(
    () => (lab.racePrev > 0 ? walkSurface("valley", RACE_START, lab.racePrev, RACE_BUDGET) : null),
    [lab.racePrev],
  );
  const racePoint = racePath[racePath.length - 1];
  const raceMeter = surfaceValue("valley", racePoint.x, racePoint.y);
  const raceGradient = surfaceGradient("valley", racePoint.x, racePoint.y);
  const raceDone = raceMeter < RACE_TARGET;
  const stepsLeft = RACE_BUDGET - lab.raceSteps;
  const bouncing = Math.abs(1 - lab.raceStep * VALLEY_WALL) >= 0.85 && lab.raceStep * VALLEY_WALL > 1;
  const raceHint = (() => {
    if (lab.raceSteps === 0) return `Pick a step size, then press Run all ${RACE_BUDGET} steps.`;
    if (stepsLeft > 0) return `${stepsLeft} ${stepsLeft === 1 ? "step" : "steps"} left.`;
    if (bouncing) {
      return "Out of steps. Each step was too long for the steep walls: the walk jumped clean across the valley and back. Try a smaller step size.";
    }
    return "Out of steps. The walk dropped down the steep wall quickly, then crawled along the gentle floor. Try a bigger step size.";
  })();
  const restartRace = (stepSize: number) => {
    setState({ raceStep: stepSize, raceSteps: 0, racePrev: lab.raceSteps > 0 ? lab.raceStep : lab.racePrev });
  };
  const raceTo = (steps: number) => {
    const count = Math.min(RACE_BUDGET, steps);
    setState({ raceSteps: count });
    const end = walkSurface("valley", RACE_START, lab.raceStep, count).pop() ?? RACE_START;
    const reading = surfaceValue("valley", end.x, end.y);
    narrate(
      `After ${count} steps at step size ${fmt(lab.raceStep)} the meter reads ${fmt(reading, 3)}. ${
        reading < RACE_TARGET ? "Goal reached." : `The goal is under ${fmt(RACE_TARGET)}.`
      }`,
    );
  };

  return (
    <div className="sg-lab">
      {/* ============================ 1 ============================ */}
      {page === 0 && (
      <LabSurface label="Nudge the dial" className={`sg-card sg-card--nudge`}>
        <SurfaceHeading kicker="1 of 6 · the derivative" title="Nudge the dial and see how far the meter moves" />
        <p className="sg-lede">
          This machine has one dial, <b>w</b>, and one meter, <b>L</b>, that reads how wrong the machine is. The curve is the
          meter&apos;s reading at every dial setting. Give the dial a small nudge and compare the two changes.
        </p>
        <PredictCheck
          prompt="The nudge is 1 to begin with. If you make the nudge half as big, what happens to the ratio: change in the meter ÷ change in the dial?"
          options={["It halves as well", "It doubles", "It changes only a little"]}
          answer={2}
          picked={lab.nudgeGuess}
          onPick={(index) => setState({ nudgeGuess: index })}
          explanation={
            <>
              A smaller nudge causes a smaller change in the meter, so the top and the bottom of the ratio shrink together.
              With the dial at {fmt(lab.nudgeW)}, drag <b>Nudge size</b> from 1 down to 0.5: the ratio only slides from{" "}
              {fmt(nudgeRatio("bowl", lab.nudgeW, 1))} to {fmt(nudgeRatio("bowl", lab.nudgeW, 0.5))}. Keep shrinking and it
              settles on {fmt(slopeHere)}.
            </>
          }
        />
        <NudgePlot w={lab.nudgeW} nudge={lab.nudge} onDial={(w) => setState({ nudgeW: w })} />
        <div className="sg-controls sg-controls--two">
          <RangeControl
            label="Dial w"
            min={0}
            max={5}
            step={0.05}
            value={lab.nudgeW}
            format={(value) => fmt(value)}
            onChange={(value) => setState({ nudgeW: value })}
          />
          <RangeControl
            label="Nudge size"
            min={0}
            max={NUDGE_STOPS.length - 1}
            step={1}
            value={Math.max(0, NUDGE_STOPS.indexOf(lab.nudge))}
            format={(index) => String(NUDGE_STOPS[index] ?? lab.nudge)}
            onChange={(index) => setState({ nudge: NUDGE_STOPS[index] ?? 1 })}
          />
        </div>
        <FormulaWithValues
          label="Ratio"
          tone="gradient"
          expression={`change in meter ÷ change in dial = ${fmtChange(meterChange)} ÷ ${signed(lab.nudge)}`}
          result={fmt(ratio, 3)}
          detail={`Shrink the nudge and this settles on ${fmt(slopeHere)}: the slope of the meter at w = ${fmt(
            lab.nudgeW,
          )}. That settled number is the derivative.`}
        />
        <div className="metric-row">
          <Metric label={`Meter at w = ${fmt(lab.nudgeW)}`} value={fmtExact(meterAt)} tone="loss" />
          <Metric label={`Meter at w = ${fmt(lab.nudgeW + lab.nudge)}`} value={fmtExact(meterNudged)} tone="loss" />
          <Metric label="Ratio at this nudge" value={fmt(ratio, 3)} tone="gradient" />
          <Metric label="Slope it settles on" value={fmt(slopeHere)} tone="gradient" />
        </div>
        <p className="lab-note">
          Read the slope as an exchange rate: near this setting, one small unit of dial buys about {fmt(Math.abs(slopeHere))}{" "}
          units of meter{slopeHere < 0 ? ", downward, because the slope is negative" : slopeHere > 0 ? ", upward" : ""}. The
          rate only holds for small changes, which is why a nudge of 1 misses it.
        </p>
      </LabSurface>
      )}

      {/* ============================ 2 ============================ */}
      {page === 1 && (
      <LabSurface label="A slope at every point" className={`sg-card sg-card--slope`}>
        <SurfaceHeading kicker="2 of 6 · the slope is a curve too" title="Read the slope at every dial setting" />
        <p className="sg-lede">
          A bumpier meter. The lower picture plots the slope you measured in card 1, at every dial setting at once. Drag the
          dial across and read both pictures together.
        </p>
        <div className="sg-pick">
          <SegmentedControl
            label="Meter curve"
            value={lab.slopeCurve}
            options={[
              { value: "valleys", label: "Two valleys" },
              { value: "shelf", label: "A shelf" },
            ]}
            onChange={(value) => setState({ slopeCurve: value })}
          />
        </div>
        <SlopePlot curve={lab.slopeCurve} w={lab.slopeW} onDial={moveSlopeDial} />
        <div className="sg-controls">
          <RangeControl
            label="Dial w"
            min={0}
            max={6}
            step={0.05}
            value={lab.slopeW}
            format={(value) => fmt(value)}
            onChange={moveSlopeDial}
          />
        </div>
        <p className="sg-reading" role="status">
          {slopeReading}
        </p>
        <div className="metric-row">
          <Metric label="Meter L" value={fmt(slopeMeter)} tone="loss" />
          <Metric label="Slope here" value={slopeFlat ? "0.00" : signed(slopeAt)} tone="gradient" />
          <Metric label="The meter is" value={slopeFlat ? "level" : slopeAt > 0 ? "rising ↗" : "falling ↘"} />
          <Metric label="Downhill is" value={slopeFlat ? "neither way" : slopeAt > 0 ? "← to the left" : "to the right →"} />
        </div>
        <p className="lab-note">
          {lab.slopeCurve === "valleys"
            ? "The slope is 0 in three places on this curve, and only two of them are bottoms. Park the dial on each dotted line, then switch to A shelf for a level spot that is neither a bottom nor a top."
            : "At the shelf the slope touches 0 without changing sign: the meter rises, levels off, then rises again. A slope of 0 says the ground is level, never that you have reached the lowest point."}
        </p>
      </LabSurface>
      )}

      {/* ============================ 3 ============================ */}
      {page === 2 && (
      <LabSurface label="Two dials, one slope each" className={`sg-card sg-card--map`}>
        <SurfaceHeading kicker="3 of 6 · the gradient" title="Two dials: one slope each, then one arrow" />
        <p className="sg-lede">
          Now the machine has a second dial, <b>b</b>, beside <b>w</b>. You are looking down on it from above: each line joins dial
          settings where the meter reads the same, like height lines on a hiking map. Switch <b>Picture</b> to 3D surface to see
          the ground those lines describe.
        </p>
        <PredictCheck
          prompt="Hold one dial still and you get one slope, drawn as a leg. When the two legs are joined into a single arrow from the point, which way will it point?"
          options={["Downhill, toward lower ground", "Uphill, toward higher ground", "Level, along the contour line"]}
          answer={1}
          picked={lab.arrowGuess}
          onPick={(index) => setState({ arrowGuess: index })}
          explanation={
            <>
              Each leg points the way its dial must turn to raise the meter, so the arrow built from them points to where the
              meter rises fastest: straight uphill. Press <b>Both together</b> to see it cross the contour line at a right
              angle. Downhill is exactly the opposite way, which is why card 5 moves <i>against</i> the slope.
            </>
          }
        />
        <div className={`sg-split${onSurface ? " is-surface" : ""}`}>
          <div className="stage3d-switch sg-picture">
            <SegmentedControl
              label="Picture"
              value={lab.mapView}
              options={[
                { value: "map", label: "Flat map" },
                { value: "surface", label: "3D surface" },
              ]}
              onChange={(value) => {
                setState({ mapView: value });
                narrate(
                  value === "surface"
                    ? "3D surface: the same meter drawn as ground. Height is the meter reading, and each contour line sits at its own height."
                    : "Flat map: the ground seen from straight above.",
                );
              }}
            />
            {lab.mapView === "surface" ? (
              <GradientSurface
                surface={lab.surface}
                w1={lab.mapW}
                w2={lab.mapB}
                legs={lab.legs}
                onMove={(w1, w2) => setState({ mapW: w1, mapB: w2 })}
              />
            ) : (
              <GradientMap
                surface={lab.surface}
                w1={lab.mapW}
                w2={lab.mapB}
                legs={lab.legs}
                onMove={(w1, w2) => setState({ mapW: w1, mapB: w2 })}
              />
            )}
          </div>
          <div className="sg-side">
            <SegmentedControl
              label="Surface"
              value={lab.surface}
              options={[
                { value: "bowl", label: "Round bowl" },
                { value: "valley", label: "Long valley" },
                { value: "saddle", label: "Saddle" },
              ]}
              onChange={(value) => setState({ surface: value })}
            />
            <div className="sg-pick sg-pick--legs">
              <SegmentedControl
                label="Dials turning"
                value={lab.legs}
                options={[
                  { value: "w", label: "w only" },
                  { value: "b", label: "b only" },
                  { value: "both", label: "Both together" },
                ]}
                onChange={(value) => setState({ legs: value })}
              />
            </div>
            <RangeControl
              label="Dial w"
              min={0}
              max={6}
              step={0.05}
              value={lab.mapW}
              format={(value) => fmt(value)}
              onChange={(value) => setState({ mapW: value })}
            />
            <RangeControl
              label="Dial b"
              min={0}
              max={6}
              step={0.05}
              value={lab.mapB}
              format={(value) => fmt(value)}
              onChange={(value) => setState({ mapB: value })}
            />
            <div className="metric-row sg-metrics--grid">
              <Metric label="Meter L" value={fmt(mapMeter)} tone="loss" />
              <Metric label="Steepness (arrow length)" value={fmt(steepness)} tone="gradient" />
              <Metric label="Slope along w" value={signed(gradient.x)} tone="forward" />
              <Metric label="Slope along b" value={signed(gradient.y)} />
            </div>
          </div>
        </div>
        <FormulaWithValues
          label="Gradient"
          tone="gradient"
          expression={`(slope along w, slope along b) = (${signed(gradient.x)}, ${signed(gradient.y)})`}
          result={`length ${fmt(steepness)}`}
          detail="One slope per dial, written as a list. A list of numbers is a vector, so the gradient is an arrow: its direction is straight uphill and its length is how steep."
        />
        <p className="lab-note">{legsNote}</p>
      </LabSurface>
      )}

      {/* ============================ 4 ============================ */}
      {page === 3 && (
      <LabSurface label="Dials that drive dials" className={`sg-card sg-card--chain`}>
        <SurfaceHeading kicker="4 of 6 · the chain rule" title="A dial that reaches the meter through another machine" />
        <p className="sg-lede">
          The dial no longer reaches the meter directly. Machine A turns the dial <b>w</b> into a middle reading <b>u</b>, and
          machine B turns <b>u</b> into the meter <b>L</b>. Nudge the dial and follow the change down the chain.
        </p>
        <PredictCheck
          prompt="With the dial at 3, machine A's slope is 3 and machine B's slope is 2. If you nudge the dial by 0.1, about how far does the meter move?"
          options={["About 0.2", "About 0.3", "About 0.5", "About 0.6"]}
          answer={3}
          picked={lab.chainGuess}
          onPick={(index) => setState({ chainGuess: index })}
          explanation={
            <>
              Machine A turns the 0.1 into about 0.3 of u (0.1 × 3). Machine B turns that 0.3 into about 0.6 of meter (0.3 ×
              2). The slopes multiply: 3 × 2 = 6 units of meter for each unit of dial. Adding them, which gives 0.5, is the
              usual slip.
            </>
          }
        />
        <ChainPlot w={lab.chainW} onDial={(w) => setState({ chainW: w })} />
        <div
          className="sg-chain"
          role="img"
          aria-label={`The chain. A nudge of ${fmt(CHAIN_NUDGE)} in the dial w, times machine A's slope ${fmt(
            chain.slopeA,
          )}, moves the middle reading u by about ${signed(chain.predictedU, 3)}. That, times machine B's slope ${fmt(
            chain.slopeB,
            3,
          )}, moves the meter L by about ${signed(chain.predictedMeter, 3)}.`}
        >
          <div className="sg-chain__cell sg-chain__cell--dial">
            <span>dial w</span>
            <strong>{fmt(lab.chainW)}</strong>
            <i><b style={{ width: bar(CHAIN_NUDGE) }} /></i>
            <small>nudge {signed(CHAIN_NUDGE)}</small>
          </div>
          <div className="sg-chain__link">
            <span>machine A</span>
            <strong>× {fmt(chain.slopeA)}</strong>
            <em aria-hidden="true">→</em>
          </div>
          <div className="sg-chain__cell sg-chain__cell--mid">
            <span>middle reading u</span>
            <strong>{fmt(chain.u)}</strong>
            <i><b style={{ width: bar(chain.predictedU) }} /></i>
            <small>moves about {signed(chain.predictedU, Math.abs(chain.predictedU) < 0.1 ? 3 : 2)}</small>
          </div>
          <div className="sg-chain__link">
            <span>machine B</span>
            <strong>× {fmt(chain.slopeB, chain.slopeB < 0.1 ? 3 : 2)}</strong>
            <em aria-hidden="true">→</em>
          </div>
          <div className="sg-chain__cell sg-chain__cell--meter">
            <span>meter L</span>
            <strong>{fmt(chain.meter)}</strong>
            <i><b style={{ width: bar(chain.predictedMeter) }} /></i>
            <small>moves about {signed(chain.predictedMeter, Math.abs(chain.predictedMeter) < 0.1 ? 3 : 2)}</small>
          </div>
        </div>
        <p className="sg-caption">The three bars are drawn to one scale: the nudge going in, and what it has become after each machine.</p>
        <div className="sg-controls">
          <RangeControl
            label="Dial w"
            min={0}
            max={6}
            step={0.05}
            value={lab.chainW}
            format={(value) => fmt(value)}
            onChange={(value) => setState({ chainW: value })}
          />
        </div>
        <FormulaWithValues
          label="Overall slope"
          tone="gradient"
          expression={`slope of A × slope of B = ${fmt(chain.slopeA)} × ${fmt(chain.slopeB, chain.slopeB < 0.1 ? 3 : 2)}`}
          result={fmt(chain.overall, chain.overall < 0.1 ? 3 : 2)}
          detail={`So a nudge of ${fmt(CHAIN_NUDGE)} should move the meter by about ${signed(
            chain.predictedMeter,
            3,
          )}. Run through both machines, the same nudge really moves it by ${signed(
            chain.actualMeter,
            3,
          )}: slopes are exact only for a vanishingly small nudge.`}
        />
        <p className="lab-note">{chainNote}</p>
      </LabSurface>
      )}

      {/* ============================ 5 ============================ */}
      {page === 4 && (
      <LabSurface label="Walk downhill" className={`sg-card sg-card--walk`}>
        <SurfaceHeading kicker="5 of 6 · gradient descent" title="Walk downhill, one step at a time" />
        <p className="sg-lede">
          Back to one dial. Pretend you cannot see the curve, only the slope under your feet. The rule: read the slope, then
          move the dial the other way, by <b>step size × slope</b>.
        </p>
        <PredictCheck
          prompt="On Two valleys, the walk starts just left of the hump, and the deeper valley is on the right. Where will the walk end up?"
          options={["In the left valley, the shallower one", "In the right valley, the deeper one", "Stuck on the hump"]}
          answer={0}
          picked={lab.walkGuess}
          onPick={(index) => setState({ walkGuess: index })}
          explanation={
            <>
              A step only reads the slope where it stands. Just left of the hump the ground tilts down to the left, so every
              step goes left, into the shallower valley, and the walk stops there because the slope at the floor is 0. Press{" "}
              <b>Take 10 steps</b> and watch. Gradient descent finds <i>a</i> low point, not always the lowest.
            </>
          }
        />
        <div className="sg-pick">
          <SegmentedControl
            label="Meter curve"
            value={lab.walkCurve}
            options={[
              { value: "valleys", label: "Two valleys" },
              { value: "bowl", label: "Bowl" },
            ]}
            onChange={(value) => setState({ walkCurve: value, walkSteps: 0 })}
          />
        </div>
        <WalkPlot
          curve={lab.walkCurve}
          path={path}
          stepSize={lab.walkStep}
          onStart={(w) => setState({ walkStart: w, walkSteps: 0 })}
        />
        <div className="sg-controls sg-controls--two">
          <RangeControl
            label="Start w"
            min={0}
            max={6}
            step={0.05}
            value={lab.walkStart}
            format={(value) => fmt(value)}
            onChange={(value) => setState({ walkStart: value, walkSteps: 0 })}
          />
          <RangeControl
            label="Step size"
            min={0.05}
            max={1.2}
            step={0.05}
            value={lab.walkStep}
            format={(value) => fmt(value)}
            onChange={(value) => setState({ walkStep: value, walkSteps: 0 })}
          />
        </div>
        <div className="sg-actions">
          <button type="button" className="primary-action" disabled={lab.walkSteps >= MAX_WALK_STEPS} onClick={() => takeSteps(1)}>
            Take a step
          </button>
          <button type="button" className="quiet-action" disabled={lab.walkSteps >= MAX_WALK_STEPS} onClick={() => takeSteps(10)}>
            Take 10 steps
          </button>
          <button type="button" className="quiet-action" disabled={lab.walkSteps === 0} onClick={() => setState({ walkSteps: 0 })}>
            Start over
          </button>
        </div>
        <FormulaWithValues
          label="Next step"
          tone="forward"
          expression={`new w = w − step size × slope = ${fmt(walkW, 3)} − ${fmt(lab.walkStep)} × ${
            walkSlope < 0 ? `(${fmt(walkSlope, 3)})` : fmt(walkSlope, 3)
          }`}
          result={fmt(walkNext, 3)}
          detail={
            Math.abs(walkSlope) < 0.005
              ? "The slope is 0 here, so the step is 0: the walk stays where it is."
              : walkSlope > 0
                ? "The slope is positive, so uphill is to the right and the step goes left."
                : "The slope is negative, so uphill is to the left and the step goes right."
          }
        />
        <div className="metric-row">
          <Metric label="Steps taken" value={String(lab.walkSteps)} />
          <Metric label="Dial w" value={fmt(walkW, 3)} />
          <Metric label="Meter L" value={fmt(walkMeter, 3)} tone="loss" />
          <Metric label="Slope here" value={signed(walkSlope, 3)} tone="gradient" />
        </div>
        <p className="lab-note" role="status">
          {walkNote}
        </p>
      </LabSurface>
      )}

      {/* ============================ 6 ============================ */}
      {page === 5 && (
      <LabSurface label="Challenge: reach the valley floor" className={`sg-card sg-card--race`}>
        <SurfaceHeading kicker="6 of 6 · challenge" title={`Reach the valley floor in ${RACE_BUDGET} steps`} />
        <p className="sg-lede">
          The Long valley from card 3, a fixed start, and {RACE_BUDGET} steps. Every step moves both dials against their own
          slopes, by the same step size. Choose that one number so the meter ends under {fmt(RACE_TARGET)}; the valley floor
          reads {fmt(RACE_FLOOR)}.
        </p>
        <div className={`sg-split${lab.raceView === "surface" ? " is-surface" : ""}`}>
          <div className="stage3d-switch sg-picture">
            <SegmentedControl
              label="Picture"
              value={lab.raceView}
              options={[
                { value: "map", label: "Flat map" },
                { value: "surface", label: "3D surface" },
              ]}
              onChange={(value) => {
                setState({ raceView: value });
                narrate(value === "surface" ? "3D surface: the long valley drawn as ground, with the walk on it." : "Flat map: the valley seen from straight above.");
              }}
            />
            {lab.raceView === "surface" ? (
              <RaceSurface path={racePath} ghost={raceGhost} ghostStep={lab.racePrev} />
            ) : (
              <RaceMap path={racePath} ghost={raceGhost} ghostStep={lab.racePrev} />
            )}
          </div>
          <div className="sg-side">
            <RangeControl
              label="Step size"
              min={0.05}
              max={1.1}
              step={0.05}
              value={lab.raceStep}
              format={(value) => fmt(value)}
              onChange={restartRace}
            />
            <div className="sg-actions sg-actions--stack">
              <button type="button" className="primary-action" disabled={stepsLeft === 0} onClick={() => raceTo(RACE_BUDGET)}>
                Run all {RACE_BUDGET} steps
              </button>
              <button type="button" className="quiet-action" disabled={stepsLeft === 0} onClick={() => raceTo(lab.raceSteps + 1)}>
                Take one step
              </button>
              <button type="button" className="quiet-action" disabled={lab.raceSteps === 0} onClick={() => restartRace(lab.raceStep)}>
                Back to the start
              </button>
            </div>
            <GoalMeter
              label="Meter L"
              value={raceProgress(raceMeter)}
              target={raceProgress(RACE_TARGET)}
              display={`${fmt(raceMeter, raceMeter < 2 ? 3 : 2)} · goal under ${fmt(RACE_TARGET)}`}
              done={raceDone}
              doneText={`${fmt(raceMeter, 3)} · goal met`}
              hint={raceHint}
            />
            <div className="metric-row">
              <Metric label="Steps left" value={String(stepsLeft)} />
              <Metric label="Meter L" value={fmt(raceMeter, raceMeter < 2 ? 3 : 2)} tone="loss" />
            </div>
          </div>
        </div>
        <FormulaWithValues
          label="Next step"
          tone="forward"
          expression={`−step size × gradient = −${fmt(lab.raceStep)} × (${signed(raceGradient.x)}, ${signed(raceGradient.y)})`}
          result={`(${signed(-lab.raceStep * raceGradient.x)}, ${signed(-lab.raceStep * raceGradient.y)})`}
          detail="The first number moves w along the gentle floor; the second moves b across the steep walls. One step size has to suit both."
        />
        <p className="lab-note" role="status">
          {raceDone
            ? "You just chose a learning rate. Training a real model is this walk with millions of dials instead of two, and the step size you tuned here is the first number anyone sets: small enough not to bounce off the steep directions, large enough to make progress along the gentle ones."
            : "Too small a step runs out of steps on the gentle floor. Too large a step overshoots the steep walls and bounces between them. The dashed path is your previous try, kept for comparison."}
        </p>
      </LabSurface>
      )}
      <StepPager steps={stepNames} current={page} goToStep={goToStep} />
    </div>
  );
}
