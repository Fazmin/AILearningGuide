import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarList,
  FormulaWithValues,
  GoalMeter,
  LabSurface,
  LineChart,
  Metric,
  PredictCheck,
  RangeControl,
  SegmentedControl,
  StepPager,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  AUTO_GOAL,
  AUTO_MAX,
  AUTO_STEP,
  DOTS_KINDS,
  DOTS_NAMES,
  HAND_GOAL,
  HAND_GRID,
  HAND_LIMIT,
  HAND_START,
  HILL_DOTS,
  HILL_MADE_WITH,
  INPUT_STOPS,
  LANDMARKS,
  LINE_DOTS,
  LINE_GOAL,
  LINE_MADE_WITH,
  LINE_START,
  MAX_LAYERS,
  MAX_UNITS,
  MOMENTUM,
  OUTPUT_STOPS,
  PLAY_DOTS,
  PLAY_GOAL,
  RATE_STOPS,
  SMALL_NAMES,
  SMALL_NET,
  SMALL_ROLES,
  TRAIN_STEPS,
  WAVE_DOTS,
  WIDTH_STOPS,
  autoRun,
  builtSizes,
  firstUnder,
  lineLoss,
  lineOutput,
  playNet,
  playRun,
  scrambleSmall,
  smallPieces,
  swapNeurons,
  wideNet,
  widePiece,
  wideRun,
} from "./model";
import { knobCount, knobsPerLayer, loss as netLoss, lossAndGradient, output } from "./net";
import { FitPlot, KnobWall, LineMachine, NetSketch, SmallNetDiagram } from "./pictures";
import { count, countSaid, duration, fmt, fmtLoss, signed } from "./kit";
import { MAX_SEED, PLAY_MAX_WIDTH, handPatch, readState } from "./state";

/** The lab shows one card per step, like pages. */
const CARD_COUNT = 6;

/** How far a loss has come from `start` toward 0, on a scale that keeps the last stretch visible. */
const progress = (value: number, start: number) => 1 - Math.sqrt(Math.min(Math.max(0, value), start) / start);

/** The playground's scale: each third of the bar is a loss ten times smaller, from 1 down to 0.001. */
const logProgress = (value: number) => 1 - (Math.log10(Math.min(1, Math.max(1e-3, value))) + 3) / 3;

const reducedMotion = () => typeof document !== "undefined" && document.documentElement.dataset.motion === "reduced";

/** Call `tick` every `ms` while `active`. The newest `tick` is always the one that runs. */
function useTicker(active: boolean, tick: () => void, ms: number) {
  const saved = useRef(tick);
  saved.current = tick;
  useEffect(() => {
    if (!active || typeof window === "undefined") return undefined;
    const id = window.setInterval(() => saved.current(), ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
}

const LINE_START_LOSS = lineLoss(LINE_START.w, LINE_START.b);
const HAND_START_LOSS = netLoss(SMALL_NET, HAND_START, HILL_DOTS);
const GPT3 = LANDMARKS[LANDMARKS.length - 1];

export default function Explore({ state, setState, currentStep, narrate, steps, goToStep }: ModuleContext) {
  const lab = readState(state);
  const page = Math.min(CARD_COUNT - 1, Math.max(0, Math.floor(Number.isFinite(currentStep) ? currentStep : 0)));
  const stepNames = steps ?? Array.from({ length: CARD_COUNT }, (_, index) => `step ${index + 1}`);
  const [autoRunning, setAutoRunning] = useState(false);
  const [playing, setPlaying] = useState(false);

  /* ---------------- card 1: two knobs ---------------- */
  const lineNow = lineLoss(lab.lineW, lab.lineB);
  const lineOut = lineOutput(lab.lineW, lab.lineB, lab.lineX);
  const lineDone = lineNow < LINE_GOAL;
  const lineHint =
    Math.abs(lab.lineW - LINE_MADE_WITH.w) > 0.05
      ? "The gaps change in length from one end of the line to the other, so the tilt is wrong. Knob w sets the tilt."
      : "The tilt matches: every gap is the same length. Knob b slides the whole line up or down.";

  /* ---------------- cards 2 and 3: seven knobs ---------------- */
  const handKey = lab.hand.join(",");
  const auto = useMemo(() => autoRun(lab.hand), [handKey]);
  const knobs = auto.knobs[lab.auto];
  const smallLoss = auto.losses[lab.auto];
  const slopes = useMemo(() => lossAndGradient(SMALL_NET, knobs, HILL_DOTS).gradient, [knobs]);
  const handDone = smallLoss < HAND_GOAL;
  const autoDone = smallLoss < AUTO_GOAL;
  const picked = SMALL_NAMES[lab.pick];
  const strongest = slopes.reduce((best, slope, index) => (Math.abs(slope) > Math.abs(slopes[best]) ? index : best), 0);
  const autoGoalStep = firstUnder(auto, AUTO_GOAL);
  const sameAsMade = knobs.every((knob, index) => Math.abs(knob - HILL_MADE_WITH[index]) < 0.3);

  /** A hand turn keeps whatever the machine has done so far and starts a fresh count of steps from there. */
  const turnKnob = (index: number, value: number) => {
    setAutoRunning(false);
    const next = [...knobs];
    next[index] = value;
    setState({ ...handPatch(next), pick: index });
  };
  const scrambleHand = () => {
    setAutoRunning(false);
    const seed = (lab.scramble % MAX_SEED) + 1;
    const next = scrambleSmall(seed);
    setState({ ...handPatch(next), scramble: seed });
    narrate(`Knobs scrambled. The loss is now ${fmtLoss(netLoss(SMALL_NET, next, HILL_DOTS))}.`);
  };
  const swapHand = () => {
    setAutoRunning(false);
    setState(handPatch(swapNeurons(knobs)));
    narrate(`The two neurons traded knobs. Six knobs changed and the loss is still ${fmtLoss(smallLoss)}.`);
  };
  const stepsTo = (target: number) => {
    const next = Math.min(AUTO_MAX, Math.max(0, target));
    setState({ auto: next });
    narrate(`After ${next} steps the loss is ${fmtLoss(auto.losses[next])}.`);
  };
  const runToGoal = () => {
    if (autoRunning) {
      setAutoRunning(false);
      return;
    }
    if (reducedMotion()) {
      stepsTo(autoGoalStep >= 0 ? autoGoalStep : AUTO_MAX);
      return;
    }
    setAutoRunning(true);
  };
  useTicker(
    autoRunning && page === 2,
    () => {
      // Two steps a tick, but never past the step at which the goal is first met.
      const next = autoGoalStep > lab.auto ? Math.min(lab.auto + 2, autoGoalStep) : Math.min(AUTO_MAX, lab.auto + 2);
      setState({ auto: next });
      if (next >= AUTO_MAX || auto.losses[next] < AUTO_GOAL) {
        setAutoRunning(false);
        narrate(`Stopped after ${next} steps with the loss at ${fmtLoss(auto.losses[next])}.`);
      }
    },
    45,
  );
  const autoNote = (() => {
    if (lab.auto === 0) {
      return "Each pink arrow is a hint for one knob: which way to turn it and how hard. Press Take one step and every knob moves a little along its own arrow at the same moment.";
    }
    if (autoDone) {
      return sameAsMade
        ? "The loss is under 0.01, and here the knobs have landed close to the numbers the dots were made with. Press Scramble the knobs and run again: another start can end somewhere else and fit the dots as well."
        : "The loss is under 0.01, yet the knobs do not show the numbers the dots were made with. Several different settings draw this same curve, and the machine stopped at the one that was downhill from where it started.";
    }
    if (lab.auto >= AUTO_MAX) {
      return "That is all 300 steps, and the loss is still above 0.01. The hints have shrunk to almost nothing, so the machine has stopped improving where it stands. Press Scramble the knobs to start somewhere else.";
    }
    return "Keep stepping. No knob is told its final value: each one only gets a direction and a size for this one small turn, and all seven turn together.";
  })();

  /* ---------------- card 4: more neurons ---------------- */
  const wideRuns = useMemo(
    () => Array.from({ length: MAX_UNITS }, (_, index) => wideRun(index + 1, lab.unitsSeed)),
    [lab.unitsSeed],
  );
  const wide = wideRuns[lab.units - 1];
  const wideKnobs = wide.knobs[TRAIN_STEPS];
  const wideLoss = wide.losses[TRAIN_STEPS];
  const wideSizes = wideNet(lab.units).sizes;
  const wideFits = wideLoss < PLAY_GOAL;
  const fewestThatFit = wideRuns.findIndex((run) => run.losses[TRAIN_STEPS] < PLAY_GOAL) + 1;
  const wideNote = wideFits
    ? `With ${lab.units} hidden neurons the trained curve passes through the dots. Each thin line is one neuron's share, a single S-shaped step; the thick curve is those steps added together. No neuron draws the wave. The sum does.`
    : lab.units === 1
      ? "One hidden neuron can only send on one S-shaped step, so the best its four knobs can do is a gentle tilt through the middle of the dots. More training steps would not help: the shape is not among the ones these knobs can make."
      : `With ${lab.units} hidden neurons the trained curve catches part of the wave and misses the rest. ${
          fewestThatFit > lab.units
            ? `From this start, the smallest network on this card that gets the loss under 0.01 has ${fewestThatFit} neurons. Whether this size could do it from a better start is a separate question: press Scramble the knobs and see.`
            : fewestThatFit === 0
              ? "From this start none of the six sizes gets the loss under 0.01. Press Scramble the knobs for another start."
              : `A network with only ${fewestThatFit} neurons got under 0.01 from this same start, so this one had the knobs it needed and got stuck on the way. More knobs make a fit possible; they do not guarantee that the steps find it. Press Scramble the knobs.`
        }`;

  /* ---------------- card 5: count the knobs ---------------- */
  const sizes = builtSizes(lab.inputs, lab.layers, lab.width, lab.outputs);
  const total = knobCount(sizes);
  const perLayer = knobsPerLayer(sizes);
  const ladder = [
    { id: "card2", label: "Card 2's network", detail: "the seven knobs you turned by hand", knobs: 7, display: "7", mine: false },
    { id: "mine", label: "Your network", detail: "built with the sliders above", knobs: total, display: countSaid(total), mine: true },
    ...LANDMARKS.map((mark) => ({ id: mark.id, label: mark.name, detail: mark.what, knobs: mark.knobs, display: mark.said, mine: false })),
  ].sort((a, b) => a.knobs - b.knobs || (a.mine ? 1 : -1));

  /* ---------------- card 6: the playground ---------------- */
  const setup = {
    dots: lab.pgDots,
    layers: lab.pgLayers,
    width: lab.pgWidth,
    rate: lab.pgRate,
    momentum: lab.pgMomentum,
    seed: lab.pgSeed,
  };
  const play = useMemo(
    () => playRun(setup),
    [setup.dots, setup.layers, setup.width, setup.rate, setup.momentum, setup.seed],
  );
  const playSizes = playNet(setup).sizes;
  const playKnobs = play.knobs[lab.pgSteps];
  const playLoss = play.losses[lab.pgSteps];
  const playDone = playLoss < PLAY_GOAL;
  const playGoalStep = firstUnder(play, PLAY_GOAL);
  const playWorst = Math.max(...play.losses.slice(0, lab.pgSteps + 1));
  const playBlewUp = playLoss > Math.max(4, play.losses[0] * 2);
  const playNetTotal = knobCount(playSizes);
  const resetPlay = (patch: Record<string, string | number>) => {
    setPlaying(false);
    setState({ ...patch, pgSteps: 0 });
  };
  const playTo = (target: number) => {
    const next = Math.min(TRAIN_STEPS, Math.max(0, target));
    setState({ pgSteps: next });
    narrate(`After ${next} training steps the loss is ${fmtLoss(play.losses[next])}.`);
  };
  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (reducedMotion()) {
      playTo(TRAIN_STEPS);
      return;
    }
    if (lab.pgSteps >= TRAIN_STEPS) setState({ pgSteps: 0 });
    setPlaying(true);
  };
  useTicker(
    playing && page === 5,
    () => {
      const next = Math.min(TRAIN_STEPS, lab.pgSteps + 4);
      setState({ pgSteps: next });
      if (next >= TRAIN_STEPS) {
        setPlaying(false);
        narrate(`Training finished. After ${TRAIN_STEPS} steps the loss is ${fmtLoss(play.losses[TRAIN_STEPS])}.`);
      }
    },
    40,
  );
  const lossSeries = (losses: readonly number[], upTo: number) => {
    const stride = Math.max(1, Math.ceil(upTo / 150));
    const points = [];
    for (let step = 0; step <= upTo; step += stride) points.push({ x: step, y: losses[step] });
    if (points[points.length - 1].x !== upTo) points.push({ x: upTo, y: losses[upTo] });
    return points;
  };
  const playHint = (() => {
    if (lab.pgSteps === 0) return "Press Train and watch the curve, the knobs and the loss move together.";
    if (playBlewUp) return "The loss has shot up instead of down. Each step is so long that it overshoots: try a smaller Step size.";
    if (lab.pgSteps < TRAIN_STEPS) return `${TRAIN_STEPS - lab.pgSteps} steps left in this run.`;
    return "All 600 steps are used and the loss is still above the goal. Change the network or the stepping, or press Scramble the knobs for another start.";
  })();
  const playNote = (() => {
    if (playDone) {
      return `Goal met with ${count(playNetTotal)} knobs${
        playGoalStep >= 0 ? `: this run first dips under 0.01 at step ${playGoalStep}` : ""
      }. Nobody chose those ${count(playNetTotal)} numbers. You chose the dots, the size of the network and the stepping rule, and the steps found the settings. Now try for the same goal with fewer knobs.`;
    }
    if (lab.pgDots === "wave" && lab.pgLayers === 1 && lab.pgWidth <= 2) {
      return "Challenge: get the loss on Wave under 0.01. This small network gets stuck far above that, however long it trains. Give it more neurons, then see how few knobs you can get away with.";
    }
    return "Change one thing at a time and press Train again: the dots, the number of neurons, the step size, momentum, or the start. Every run begins from scrambled knobs and takes at most 600 steps.";
  })();

  return (
    <div className="nk-lab">
      {/* ============================ 1 ============================ */}
      {page === 0 && (
        <LabSurface label="A machine with two knobs" className="nk-card nk-card--line">
          <SurfaceHeading kicker="1 of 6 · parameters" title="Two knobs decide what this machine does" />
          <p className="nk-lede">
            A number goes in on the left, is multiplied by knob <b>w</b>, has knob <b>b</b> added, and comes out on the right.
            The dots are the right answers: for each input, the output we want. Turn the two knobs until the machine&apos;s line
            runs through every dot.
          </p>
          <PredictCheck
            prompt="Suppose the knobs were set to w = 0.5 and b = 1, and you fed in x = 2. What would come out?"
            options={["2", "3", "3.5"]}
            answer={0}
            picked={lab.lineGuess}
            onPick={(index) => setState({ lineGuess: index })}
            explanation={
              <>
                Multiply first, then add: 2 × 0.5 = 1, and 1 + 1 = 2. The machine does nothing else. Set <b>Knob w</b> to 0.50,{" "}
                <b>Knob b</b> to 1.00 and <b>Input x</b> to 2.00, and read the three boxes on the machine. Everything this machine
                will ever do is fixed by those two knob settings.
              </>
            }
          />
          <div className="nk-split">
            <div className="nk-side">
              <LineMachine
                w={lab.lineW}
                b={lab.lineB}
                x={lab.lineX}
                onW={(value) => setState({ lineW: value })}
                onB={(value) => setState({ lineB: value })}
              />
              <RangeControl
                label="Knob w"
                min={-HAND_LIMIT}
                max={HAND_LIMIT}
                step={HAND_GRID}
                value={lab.lineW}
                format={(value) => signed(value)}
                onChange={(value) => setState({ lineW: value })}
              />
              <RangeControl
                label="Knob b"
                min={-HAND_LIMIT}
                max={HAND_LIMIT}
                step={HAND_GRID}
                value={lab.lineB}
                format={(value) => signed(value)}
                onChange={(value) => setState({ lineB: value })}
              />
              <RangeControl
                label="Input x"
                min={-2}
                max={2}
                step={0.25}
                value={lab.lineX}
                format={(value) => fmt(value)}
                onChange={(value) => setState({ lineX: value })}
              />
            </div>
            <div className="nk-picture">
              <FitPlot
                dots={LINE_DOTS}
                curve={(x) => lineOutput(lab.lineW, lab.lineB, x)}
                probe={lab.lineX}
                curveName="the machine's output at every input"
                label={`Seventeen dots on a straight line, and the machine's line for knob w at ${signed(lab.lineW)} and knob b at ${signed(
                  lab.lineB,
                )}. The loss, the average squared gap between the line and the dots, is ${fmtLoss(lineNow)}. At input ${fmt(
                  lab.lineX,
                )} the machine's output is ${fmt(lineOut)}.`}
              />
              <GoalMeter
                label="Loss"
                value={progress(lineNow, LINE_START_LOSS)}
                target={progress(LINE_GOAL, LINE_START_LOSS)}
                display={`${fmtLoss(lineNow)} · goal 0.000`}
                done={lineDone}
                doneText={`${fmtLoss(lineNow)} · every dot is on the line`}
                hint={lineHint}
              />
            </div>
          </div>
          <FormulaWithValues
            label="Output"
            tone="forward"
            expression={`w × x + b = ${lab.lineW < 0 ? `(${fmt(lab.lineW)})` : fmt(lab.lineW)} × ${
              lab.lineX < 0 ? `(${fmt(lab.lineX)})` : fmt(lab.lineX)
            } ${lab.lineB < 0 ? "−" : "+"} ${fmt(Math.abs(lab.lineB))}`}
            result={fmt(lineOut)}
            detail="The input changes from one question to the next. The knobs stay where you left them: they are the machine's parameters."
          />
          <div className="metric-row">
            <Metric label="Knob w" value={signed(lab.lineW)} tone="forward" />
            <Metric label="Knob b" value={signed(lab.lineB)} tone="forward" />
            <Metric label="Loss" value={fmtLoss(lineNow)} tone="loss" />
            <Metric label="Knobs" value="2" />
          </div>
          <p className="lab-note" role="status">
            {lineDone
              ? "You just trained a model by hand. The dots never moved and the machine's wiring never changed: all of the learning is stored in two numbers. The loss is how the machine is told it is wrong, as one number: the average of the squared gaps."
              : "Each thin line is a gap between a dot and the machine's answer there. The loss squares every gap and takes the average, so one number says how wrong the whole line is. Lower is better, and 0 means every dot is hit."}
          </p>
        </LabSurface>
      )}

      {/* ============================ 2 ============================ */}
      {page === 1 && (
        <LabSurface label="Seven knobs by hand" className="nk-card nk-card--hand">
          <SurfaceHeading kicker="2 of 6 · a network" title="Wire three small machines together: seven knobs" />
          <p className="nk-lede">
            These dots form a hill, and no setting of two knobs bends a straight line into a hill. So the input now feeds two{" "}
            <b>hidden neurons</b>. Each is a small machine with its own knobs that sends on an S-shaped curve, and the output adds
            the two curves up. That is a neural network. Turn its knobs until the curve follows the dots.
          </p>
          <PredictCheck
            prompt="Say you tried only 10 settings for each knob. The two-knob machine would have 100 combinations to try. How many does this seven-knob network have?"
            options={["700", "70,000", "10,000,000"]}
            answer={2}
            picked={lab.comboGuess}
            onPick={(index) => setState({ comboGuess: index })}
            explanation={
              <>
                Every extra knob multiplies the count by 10, so seven knobs give 10 × 10 × 10 × 10 × 10 × 10 × 10 = 10,000,000
                combinations. At one try per second that is about 116 days, for a network small enough to draw on a card. Trying
                settings one by one is finished as an idea. Tune it by feel instead, and notice how each knob&apos;s best spot
                moves when you turn another.
              </>
            }
          />
          <div className="nk-split">
            <div className="nk-side">
              <SmallNetDiagram knobs={knobs} pick={lab.pick} onPick={(index) => setState({ pick: index })} onTurn={turnKnob} />
              <div className="nk-pick">
                <SegmentedControl
                  label="Selected knob"
                  value={String(lab.pick)}
                  options={SMALL_NAMES.map((name, index) => ({ value: String(index), label: name }))}
                  onChange={(value) => setState({ pick: Number(value) })}
                />
              </div>
              <RangeControl
                label="Turn the selected knob"
                min={-HAND_LIMIT}
                max={HAND_LIMIT}
                step={HAND_GRID}
                value={Number(knobs[lab.pick].toFixed(2))}
                format={(value) => `${picked} ${signed(value)}`}
                onChange={(value) => turnKnob(lab.pick, value)}
              />
              <p className="nk-caption">
                <b className="nk-name">{picked}</b> is {SMALL_ROLES[lab.pick]}. Knobs on wires are called weights; a
                neuron&apos;s own knob is its bias.
              </p>
            </div>
            <div className="nk-picture">
              <FitPlot
                dots={HILL_DOTS}
                curve={(x) => output(SMALL_NET, knobs, x)}
                pieces={[
                  { name: "neuron 1", rule: (x) => smallPieces(knobs, x)[0] },
                  { name: "neuron 2", rule: (x) => smallPieces(knobs, x)[1] },
                ]}
                curveName="the network's output"
                label={`Seventeen dots forming a hill, and the network's curve at the current seven knob settings. The loss is ${fmtLoss(
                  smallLoss,
                )}. Two thin lines show each hidden neuron's share of the output.`}
              />
              <GoalMeter
                label="Loss"
                value={progress(smallLoss, HAND_START_LOSS)}
                target={progress(HAND_GOAL, HAND_START_LOSS)}
                display={`${fmtLoss(smallLoss)} · goal under ${fmt(HAND_GOAL)}`}
                done={handDone}
                doneText={`${fmtLoss(smallLoss)} · the curve follows the dots`}
                hint="Start with the two neurons: each needs to become one side of the hill. It can be done by hand, and it takes patience."
              />
              <div className="nk-actions">
                <button type="button" className="quiet-action" onClick={swapHand}>
                  Swap the two neurons
                </button>
                <button type="button" className="quiet-action" onClick={scrambleHand}>
                  Scramble the knobs
                </button>
              </div>
            </div>
          </div>
          <div className="metric-row">
            <Metric label="Knobs" value="7" />
            <Metric label="Wire knobs (weights)" value="4" tone="forward" />
            <Metric label="Neuron knobs (biases)" value="3" tone="forward" />
            <Metric label="Loss" value={fmtLoss(smallLoss)} tone="loss" />
          </div>
          <p className="lab-note" role="status">
            {handDone
              ? "The curve follows the dots. Now press Swap the two neurons: six of the seven knobs change and the curve does not move at all. Two different settings, one machine. A knob has no meaning on its own, only together with the others."
              : "Press Swap the two neurons at any time: neuron 1's three knobs trade places with neuron 2's. Six knobs change and the curve and the loss stay exactly where they were, because the output only adds the two neurons up. A knob has no meaning on its own, only together with the others."}
          </p>
        </LabSurface>
      )}

      {/* ============================ 3 ============================ */}
      {page === 2 && (
        <LabSurface label="Let the machine turn the knobs" className="nk-card nk-card--auto">
          <SurfaceHeading kicker="3 of 6 · training" title="Every knob gets a hint, and all of them turn at once" />
          <p className="nk-lede">
            The same seven knobs, exactly as you left them on card 2. For each knob the machine now works out the slope of the
            loss: which way a small turn would move the loss, and by how much. The pink arrow on a knob points the way that
            lowers the loss. One <b>step</b> turns every knob a little along its own arrow.
          </p>
          <PredictCheck
            prompt="The machine will keep stepping until the loss is close to 0. Will its seven knobs then show the numbers the dots were made with?"
            options={[
              "Yes: only one setting fits the dots",
              "Not necessarily: several settings draw the same curve",
              "No: stepping cannot get the loss close to 0",
            ]}
            answer={1}
            picked={lab.sameGuess}
            onPick={(index) => setState({ sameGuess: index })}
            explanation={
              <>
                Card 2&apos;s <b>Swap the two neurons</b> already showed two settings that draw one curve, and there are more:
                flip the sign of all three knobs that belong to one neuron and the curve is unchanged again. The steps only walk
                downhill from wherever the knobs start, so they stop at whichever of those settings is nearest that path. Press{" "}
                <b>Run to the goal</b>, then compare the two rows of the table under the picture.
              </>
            }
          />
          <div className="nk-split">
            <div className="nk-side">
              <SmallNetDiagram
                knobs={knobs}
                pick={lab.pick}
                slopes={slopes}
                slopeScale={Math.max(0.4, ...slopes.map((slope) => Math.abs(slope)))}
                onPick={(index) => setState({ pick: index })}
                onTurn={turnKnob}
              />
              <div className="nk-pick">
                <SegmentedControl
                  label="Selected knob"
                  value={String(lab.pick)}
                  options={SMALL_NAMES.map((name, index) => ({ value: String(index), label: name }))}
                  onChange={(value) => setState({ pick: Number(value) })}
                />
              </div>
              <p className="nk-caption">
                Pink arrows are the hints: the way each knob will turn on the next step, longer where the loss is more sensitive
                to that knob. You can still turn any knob by hand; the count of steps then starts again from there.
              </p>
            </div>
            <div className="nk-picture">
              <FitPlot
                dots={HILL_DOTS}
                curve={(x) => output(SMALL_NET, knobs, x)}
                curveName="the network's output"
                label={`The hill of dots and the network's curve after ${lab.auto} steps. The loss is ${fmtLoss(smallLoss)}.`}
              />
              <div className="nk-actions">
                <button type="button" className="primary-action" disabled={lab.auto >= AUTO_MAX} onClick={() => stepsTo(lab.auto + 1)}>
                  Take one step
                </button>
                <button type="button" className="quiet-action" disabled={lab.auto >= AUTO_MAX} onClick={() => stepsTo(lab.auto + 20)}>
                  Take 20 steps
                </button>
                <button
                  type="button"
                  className="quiet-action"
                  aria-pressed={autoRunning}
                  disabled={!autoRunning && (lab.auto >= AUTO_MAX || autoDone)}
                  onClick={runToGoal}
                >
                  {autoRunning ? "Pause" : "Run to the goal"}
                </button>
                <button type="button" className="quiet-action" disabled={lab.auto === 0} onClick={() => stepsTo(0)}>
                  Back to step 0
                </button>
                <button type="button" className="quiet-action" onClick={scrambleHand}>
                  Scramble the knobs
                </button>
              </div>
              <GoalMeter
                label="Loss"
                value={progress(smallLoss, Math.max(auto.losses[0], AUTO_GOAL * 4))}
                target={progress(AUTO_GOAL, Math.max(auto.losses[0], AUTO_GOAL * 4))}
                display={`${fmtLoss(smallLoss)} · goal under ${fmt(AUTO_GOAL)}`}
                done={autoDone}
                doneText={`${fmtLoss(smallLoss)} · goal met after ${lab.auto} steps`}
                hint={`${AUTO_MAX - lab.auto} of ${AUTO_MAX} steps left.`}
              />
            </div>
          </div>
          <div className="nk-table-wrap">
            <table className="nk-table">
              <caption>The seven knobs now, beside the setting the dots were made with</caption>
              <thead>
                <tr>
                  <th scope="col">Knob</th>
                  {SMALL_NAMES.map((name) => (
                    <th scope="col" key={name}>
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Now</th>
                  {knobs.map((knob, index) => (
                    <td key={SMALL_NAMES[index]}>{signed(knob)}</td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Slope of the loss</th>
                  {slopes.map((slope, index) => (
                    <td key={SMALL_NAMES[index]}>{signed(slope)}</td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Made with</th>
                  {HILL_MADE_WITH.map((knob, index) => (
                    <td key={SMALL_NAMES[index]}>{signed(knob)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <FormulaWithValues
            label="Next turn"
            tone="gradient"
            expression={`new ${picked} = ${picked} − step size × slope = ${fmt(knobs[lab.pick])} − ${fmt(AUTO_STEP, 1)} × ${
              slopes[lab.pick] < 0 ? `(${fmt(slopes[lab.pick])})` : fmt(slopes[lab.pick])
            }`}
            result={fmt(Math.max(-HAND_LIMIT, Math.min(HAND_LIMIT, knobs[lab.pick] - AUTO_STEP * slopes[lab.pick])))}
            detail="The same rule runs for all seven knobs in the same step, each with its own slope. The list of seven slopes is the gradient."
          />
          <div className="metric-row">
            <Metric label="Steps taken" value={String(lab.auto)} />
            <Metric label="Loss" value={fmtLoss(smallLoss)} tone="loss" />
            <Metric
              label="Strongest hint"
              value={
                Math.abs(slopes[strongest]) < 0.005
                  ? "none: all near 0"
                  : `${SMALL_NAMES[strongest]}: turn ${slopes[strongest] < 0 ? "up" : "down"}`
              }
              tone="gradient"
            />
            <Metric label="Step size" value={fmt(AUTO_STEP, 1)} />
          </div>
          <p className="lab-note" role="status">
            {autoNote}
          </p>
        </LabSurface>
      )}

      {/* ============================ 4 ============================ */}
      {page === 3 && (
        <LabSurface label="More neurons, more shapes" className="nk-card nk-card--wide">
          <SurfaceHeading kicker="4 of 6 · capacity" title="What a network can draw depends on how many knobs it has" />
          <p className="nk-lede">
            New dots: a wave that rises, falls, rises and falls again. Each network on this card has one row of hidden neurons
            and was trained the same way, with {TRAIN_STEPS} steps from scrambled knobs. Only the number of hidden neurons
            changes. Drag <b>Hidden neurons</b> and compare what each size ends up drawing.
          </p>
          <PredictCheck
            prompt="A network with one hidden neuron can only draw a single S-shaped step. If you trained that network on these dots for ten times as long, would its curve follow the wave?"
            options={["Yes, given enough steps", "Yes, if the steps were larger", "No: none of its settings makes that shape"]}
            answer={2}
            picked={lab.shapeGuess}
            onPick={(index) => setState({ shapeGuess: index })}
            explanation={
              <>
                Training only chooses among the shapes the knobs can make. One hidden neuron&apos;s four knobs can move its
                S-shaped step, stretch it and flip it, and that is all. The wave is not on that list, so no amount of stepping
                reaches it. Set <b>Hidden neurons</b> to 1 and look, then add neurons: every one adds a new S-shaped step for
                the output to combine.
              </>
            }
          />
          <div className="nk-split nk-split--plot">
            <div className="nk-picture">
              <FitPlot
                tall
                dots={WAVE_DOTS}
                curve={(x) => output(wideNet(lab.units), wideKnobs, x)}
                pieces={Array.from({ length: lab.units }, (_, unit) => ({
                  name: `neuron ${unit + 1}`,
                  rule: (x: number) => widePiece(lab.units, wideKnobs, unit, x),
                }))}
                curveName="the trained network's output"
                label={`Seventeen dots forming a wave with two crests, and the curve drawn by a network with ${lab.units} hidden ${
                  lab.units === 1 ? "neuron" : "neurons"
                } after ${TRAIN_STEPS} training steps. Its loss is ${fmtLoss(wideLoss)}. A thin line shows each hidden neuron's share.`}
              />
            </div>
            <div className="nk-side">
              <RangeControl
                label="Hidden neurons"
                min={1}
                max={MAX_UNITS}
                step={1}
                value={lab.units}
                format={(value) => `${value} · ${knobCount(wideNet(value).sizes)} knobs`}
                onChange={(value) => setState({ units: value })}
              />
              <BarList
                label={`Loss after ${TRAIN_STEPS} training steps, for each number of hidden neurons`}
                items={wideRuns.map((run, index) => ({
                  id: String(index + 1),
                  label: `${index + 1} ${index === 0 ? "neuron" : "neurons"}`,
                  value: run.losses[TRAIN_STEPS],
                  display: fmtLoss(run.losses[TRAIN_STEPS]),
                  tone: "loss" as const,
                  emphasis: index + 1 === lab.units,
                }))}
                selectedId={String(lab.units)}
                onSelect={(id) => setState({ units: Number(id) })}
              />
              <p className="nk-caption">Bars: the loss each size is left with. Shorter is better.</p>
              <KnobWall
                sizes={wideSizes}
                knobs={wideKnobs}
                label={`Every knob of the trained network with ${lab.units} hidden ${lab.units === 1 ? "neuron" : "neurons"}, drawn as a small dial.`}
              />
              <div className="nk-actions">
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => {
                    const seed = (lab.unitsSeed % MAX_SEED) + 1;
                    setState({ unitsSeed: seed });
                    narrate(
                      `Knobs scrambled and all six networks retrained. With ${lab.units} hidden neurons the loss is now ${fmtLoss(
                        wideRun(lab.units, seed).losses[TRAIN_STEPS],
                      )}.`,
                    );
                  }}
                >
                  Scramble the knobs
                </button>
              </div>
            </div>
          </div>
          <div className="metric-row">
            <Metric label="Hidden neurons" value={String(lab.units)} />
            <Metric label="Knobs" value={String(knobCount(wideSizes))} tone="forward" />
            <Metric label="Loss before training" value={fmtLoss(wide.losses[0])} tone="loss" />
            <Metric label={`Loss after ${TRAIN_STEPS} steps`} value={fmtLoss(wideLoss)} tone="loss" />
          </div>
          <p className="lab-note" role="status">
            {wideNote}
          </p>
        </LabSurface>
      )}

      {/* ============================ 5 ============================ */}
      {page === 4 && (
        <LabSurface label="Count the knobs" className="nk-card nk-card--count">
          <SurfaceHeading kicker="5 of 6 · scale" title="From seven knobs to hundreds of billions" />
          <p className="nk-lede">
            Bigger networks are the same thing with more of it: more inputs, more neurons in a row, more rows. Every wire carries
            a knob and every neuron has one of its own. Build a network with the sliders and watch the count. When people say a
            model has 175 billion <b>parameters</b>, this is the number they mean.
          </p>
          <PredictCheck
            prompt="One row of 100 neurons feeds a second row of 100 neurons, and every neuron in the first is wired to every neuron in the second. How many knobs does the second row add?"
            options={["200", "10,100", "1,000,000"]}
            answer={1}
            picked={lab.wiresGuess}
            onPick={(index) => setState({ wiresGuess: index })}
            explanation={
              <>
                Each of the 100 neurons in the second row has 100 wires coming in, which is 100 × 100 = 10,000 wire knobs, plus
                one knob of its own for each of the 100 neurons: 10,100. The wires are nearly all of it. Double both rows to 200
                and the count becomes 40,200, about four times as many. Raise <b>Neurons per layer</b> with two or more{" "}
                <b>Hidden layers</b> and watch the count grow that way.
              </>
            }
          />
          <div className="nk-split">
            <div className="nk-side">
              <RangeControl
                label="Inputs"
                min={0}
                max={INPUT_STOPS.length - 1}
                step={1}
                value={Math.max(0, INPUT_STOPS.indexOf(lab.inputs as (typeof INPUT_STOPS)[number]))}
                format={(index) => count(INPUT_STOPS[index] ?? lab.inputs)}
                onChange={(index) => setState({ inputs: INPUT_STOPS[index] ?? 1 })}
              />
              <RangeControl
                label="Hidden layers"
                min={1}
                max={MAX_LAYERS}
                step={1}
                value={lab.layers}
                onChange={(value) => setState({ layers: value })}
              />
              <RangeControl
                label="Neurons per layer"
                min={0}
                max={WIDTH_STOPS.length - 1}
                step={1}
                value={Math.max(0, WIDTH_STOPS.indexOf(lab.width as (typeof WIDTH_STOPS)[number]))}
                format={(index) => count(WIDTH_STOPS[index] ?? lab.width)}
                onChange={(index) => setState({ width: WIDTH_STOPS[index] ?? 2 })}
              />
              <SegmentedControl
                label="Outputs"
                value={String(lab.outputs)}
                options={OUTPUT_STOPS.map((stop) => ({ value: String(stop), label: String(stop) }))}
                onChange={(value) => setState({ outputs: Number(value) })}
              />
              <div className="nk-actions">
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => setState({ inputs: 1, layers: 1, width: 2, outputs: 1 })}
                >
                  Card 2&apos;s network
                </button>
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => setState({ inputs: 784, layers: 2, width: 16, outputs: 10 })}
                >
                  A small digit reader
                </button>
              </div>
            </div>
            <div className="nk-picture">
              <NetSketch
                sizes={sizes}
                label={`A network with ${count(lab.inputs)} ${lab.inputs === 1 ? "input" : "inputs"}, ${lab.layers} hidden ${
                  lab.layers === 1 ? "layer" : "layers"
                } of ${count(lab.width)} neurons, and ${lab.outputs} ${lab.outputs === 1 ? "output" : "outputs"}: ${count(
                  total,
                )} knobs. Layers too tall to draw show a few circles around a gap.`}
              />
              <ol className="nk-tally" aria-label="Knobs added by each layer">
                {perLayer.map((knobsHere, index) => (
                  <li key={index}>
                    <span>{index === perLayer.length - 1 ? "into the output" : `into hidden layer ${index + 1}`}</span>
                    <code>
                      {count(sizes[index])} × {count(sizes[index + 1])} wires + {count(sizes[index + 1])}
                    </code>
                    <strong>{count(knobsHere)}</strong>
                  </li>
                ))}
              </ol>
            </div>
          </div>
          <FormulaWithValues
            label="Knobs"
            tone="forward"
            expression={perLayer.map((value) => count(value)).join(" + ")}
            result={count(total)}
            detail="For each layer: the neurons in the layer before × the neurons in this one, for the wires, plus one more for each neuron in this one."
          />
          <BarList
            label="Knob counts of this network and four published ones. Each equal stretch of bar is ten times more knobs."
            max={12}
            items={ladder.map((row) => ({
              id: row.id,
              label: row.label,
              detail: row.detail,
              value: Math.log10(Math.max(1, row.knobs)),
              display: row.display,
              tone: row.mine ? ("forward" as const) : ("muted" as const),
              emphasis: row.mine,
            }))}
          />
          <p className="nk-caption">
            The bars are on a times-ten scale: each equal stretch of bar is ten times more knobs. On an ordinary scale, with
            card 2&apos;s bar one millimetre long, GPT-3&apos;s would be 25,000 kilometres long. The four published counts are
            rounded as their papers give them.
          </p>
          <div className="metric-row">
            <Metric label="Knobs in your network" value={countSaid(total)} tone="forward" />
            <Metric label="Times card 2's network" value={`× ${countSaid(Math.round(total / 7))}`} />
            <Metric label="One second per knob" value={duration(total)} />
            <Metric label={`${GPT3.name.replace(/ \(.*/, "")} is larger by`} value={`× ${countSaid(Math.round(GPT3.knobs / total))}`} />
          </div>
          <p className="lab-note">
            Nothing new was added on the way up: every one of those knobs is a number that training nudges along its own hint,
            exactly as on card 3. Touching each of GPT-3&apos;s 175 billion knobs once, at one per second, would take about{" "}
            {duration(GPT3.knobs)}. That is why the hints have to be worked out by arithmetic, for all the knobs at once.
          </p>
        </LabSurface>
      )}

      {/* ============================ 6 ============================ */}
      {page === 5 && (
        <LabSurface label="Playground: train a network" className="nk-card nk-card--play">
          <SurfaceHeading kicker="6 of 6 · your turn" title="Pick the dots, shape the network, and train it" />
          <p className="nk-lede">
            Everything from the first five cards in one place. Choose what the network should learn and how big it is, then press{" "}
            <b>Train</b> and watch three things move together: the curve, the wall of knobs, and the loss.
          </p>
          <div className="nk-controls">
            <SegmentedControl
              label="Dots"
              value={lab.pgDots}
              options={DOTS_KINDS.map((kind) => ({ value: kind, label: DOTS_NAMES[kind] }))}
              onChange={(value) => resetPlay({ pgDots: value })}
            />
            <SegmentedControl
              label="Hidden layers"
              value={String(lab.pgLayers)}
              options={[
                { value: "1", label: "1" },
                { value: "2", label: "2" },
              ]}
              onChange={(value) => resetPlay({ pgLayers: Number(value) })}
            />
            <SegmentedControl
              label="Momentum"
              value={lab.pgMomentum ? "on" : "off"}
              options={[
                { value: "off", label: "Off" },
                { value: "on", label: "On" },
              ]}
              onChange={(value) => resetPlay({ pgMomentum: value })}
            />
            <RangeControl
              label="Neurons per layer"
              min={1}
              max={PLAY_MAX_WIDTH}
              step={1}
              value={lab.pgWidth}
              onChange={(value) => resetPlay({ pgWidth: value })}
            />
            <RangeControl
              label="Step size"
              min={0}
              max={RATE_STOPS.length - 1}
              step={1}
              value={Math.max(0, RATE_STOPS.indexOf(lab.pgRate as (typeof RATE_STOPS)[number]))}
              format={(index) => String(RATE_STOPS[index] ?? lab.pgRate)}
              onChange={(index) => resetPlay({ pgRate: RATE_STOPS[index] ?? 0.05 })}
            />
          </div>
          <div className="nk-split nk-split--plot">
            <div className="nk-picture">
              <FitPlot
                tall
                dots={PLAY_DOTS[lab.pgDots]}
                curve={(x) => output(playNet(setup), playKnobs, x)}
                curveName="the network's output"
                label={`The ${DOTS_NAMES[lab.pgDots]} dots and the curve of a network with ${lab.pgLayers} hidden ${
                  lab.pgLayers === 1 ? "layer" : "layers"
                } of ${lab.pgWidth} ${lab.pgWidth === 1 ? "neuron" : "neurons"}, after ${lab.pgSteps} of ${TRAIN_STEPS} training steps. The loss is ${fmtLoss(
                  playLoss,
                )}.`}
              />
              <div className="nk-actions">
                <button type="button" className="primary-action" aria-pressed={playing} onClick={togglePlay}>
                  {playing ? "Pause" : lab.pgSteps >= TRAIN_STEPS ? "Train again" : lab.pgSteps > 0 ? "Keep training" : "Train"}
                </button>
                <button type="button" className="quiet-action" disabled={lab.pgSteps >= TRAIN_STEPS} onClick={() => playTo(lab.pgSteps + 10)}>
                  Take 10 steps
                </button>
                <button type="button" className="quiet-action" disabled={lab.pgSteps === 0} onClick={() => resetPlay({})}>
                  Back to step 0
                </button>
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => {
                    resetPlay({ pgSeed: (lab.pgSeed % MAX_SEED) + 1 });
                    narrate("Knobs scrambled: the network starts from a new random setting.");
                  }}
                >
                  Scramble the knobs
                </button>
              </div>
              <RangeControl
                label="Training steps"
                min={0}
                max={TRAIN_STEPS}
                step={1}
                value={lab.pgSteps}
                format={(value) => `${value} of ${TRAIN_STEPS}`}
                onChange={(value) => {
                  setPlaying(false);
                  setState({ pgSteps: value });
                }}
              />
            </div>
            <div className="nk-side">
              <KnobWall
                sizes={playSizes}
                knobs={playKnobs}
                label={`Every knob of the network after ${lab.pgSteps} training steps, drawn as a small dial.`}
              />
              <p className="nk-caption">
                One dial per knob. Straight up is 0, clockwise is positive, and a dial near its stop is a knob that has travelled
                far.
              </p>
              <GoalMeter
                label="Loss"
                value={logProgress(playLoss)}
                target={logProgress(PLAY_GOAL)}
                display={`${fmtLoss(playLoss)} · goal under ${fmt(PLAY_GOAL)}`}
                done={playDone}
                doneText={`${fmtLoss(playLoss)} · goal met with ${count(playNetTotal)} knobs`}
                hint={playHint}
              />
              <div className="metric-row nk-metrics--grid">
                <Metric label="Knobs" value={count(playNetTotal)} tone="forward" />
                <Metric label="Steps taken" value={String(lab.pgSteps)} />
                <Metric label="Loss" value={fmtLoss(playLoss)} tone="loss" />
                <Metric label="Highest loss so far" value={fmtLoss(playWorst)} tone="loss" />
              </div>
            </div>
          </div>
          <LineChart
            label="Loss after each training step of this run"
            series={[{ id: "loss", name: "Loss", tone: "loss", points: lossSeries(play.losses, lab.pgSteps), format: fmtLoss }]}
            xLabel="training steps"
            yLabel="loss"
            xDomain={[0, TRAIN_STEPS]}
            yDomain={[0, Math.min(4, Math.max(1, Math.ceil(play.losses[0] * 2) / 2))]}
            footnote={
              lab.pgMomentum
                ? `Each step: every knob moves by ${MOMENTUM} × its last move − step size × its slope.`
                : "Each step: every knob moves by − step size × its slope, the rule from card 3."
            }
          />
          <p className="lab-note" role="status">
            {playNote}
          </p>
        </LabSurface>
      )}
      <StepPager steps={stepNames} current={page} goToStep={goToStep} />
    </div>
  );
}
