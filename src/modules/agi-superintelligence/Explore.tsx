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
  AHEAD_MAX,
  BEYOND,
  BOTTLENECK_STOPS,
  CLAIMS,
  DOMAINS,
  DOUBLING_STOPS,
  HORIZON,
  HORIZON_START_MINUTES,
  HORIZON_START_YEAR,
  LEVELS,
  MEASURED_DOUBLING_MONTHS,
  MULTIPLIER_STOPS,
  PROFILES,
  PROFILE_NAMES,
  PROJECT_STOPS,
  RETURN_STOPS,
  SHARE_STOPS,
  SURVEY,
  SYSTEMS,
  ASI_CELL,
  columnOfCell,
  encodePlacement,
  encodeTags,
  horizonAfter,
  levelOfCell,
  projectTime,
  researchEffort,
  takeoff,
  takeoffGap,
  verdict,
  type ClaimKind,
  type Verdict,
} from "./model";
import { LevelsGrid, ProfileChart, ProjectClock, SurveyTimeline, TakeoffPlot, HorizonPlot, ClaimList, abilityAt } from "./pictures";
import { fmt, pct, span, times, workSpan } from "./kit";
import { readState } from "./state";

/** The lab shows one card per step, like pages. */
const CARD_COUNT = 6;

const RULE_NAMES: Record<Verdict, string> = {
  sum: "Add up the domains",
  weakest: "Weakest domain decides",
  count: "Count domains at adult level",
};

const stopIndex = (stops: readonly number[], value: number) => Math.max(0, stops.indexOf(value));

export default function Explore({ state, setState, currentStep, narrate, steps, goToStep }: ModuleContext) {
  const lab = readState(state);
  const page = Math.min(CARD_COUNT - 1, Math.max(0, Math.floor(Number.isFinite(currentStep) ? currentStep : 0)));
  const stepNames = steps ?? Array.from({ length: CARD_COUNT }, (_, index) => `step ${index + 1}`);

  /* ---------------- card 1: the levels ---------------- */
  const placedCount = lab.placement.filter((cell) => cell >= 0).length;
  const placedRight = lab.placement.filter((cell, index) => cell === SYSTEMS[index].cell).length;
  const allPlaced = placedCount === SYSTEMS.length;
  const selectedLevel = LEVELS[levelOfCell(lab.cell)];
  const selectedColumn = columnOfCell(lab.cell);
  const chip = SYSTEMS[lab.chip];
  const placeChip = (cell: number) => {
    const next = [...lab.placement];
    next[lab.chip] = cell;
    // Move on to the next system that has not been placed yet.
    const following = SYSTEMS.findIndex((_, index) => index > lab.chip && next[index] < 0);
    const anyLeft = next.findIndex((value) => value < 0);
    setState({ place: encodePlacement(next), cell, chip: following >= 0 ? following : anyLeft >= 0 ? anyLeft : lab.chip });
    const level = LEVELS[levelOfCell(cell)];
    narrate(`${chip.name} placed at level ${level.level}, ${level.name}, ${columnOfCell(cell)}.`);
  };

  /* ---------------- card 2: the profile ---------------- */
  const scores = PROFILES[lab.profile];
  const otherModel = lab.profile === "gpt5" ? "gpt4" : "gpt5";
  const readings = {
    sum: verdict(scores, "sum"),
    weakest: verdict(scores, "weakest"),
    count: verdict(scores, "count"),
  };
  const weakestIndex = scores.indexOf(Math.min(...scores));
  const strongCount = scores.filter((score) => score >= 10).length;
  const zeroDomains = DOMAINS.filter((_, index) => scores[index] === 0).map((domain) => domain.name.toLowerCase());

  /* ---------------- card 3: faster, more ---------------- */
  const workload = { work: lab.work, speed: lab.speed, copies: lab.copies, inOrder: lab.inOrder, world: lab.world };
  const time = projectTime(workload);
  const alone = projectTime({ ...workload, speed: 1, copies: 1 }).total;
  const unbounded = projectTime({ ...workload, speed: 1e12, copies: 1e12 }).total;
  const largest = (["world", "inOrder", "parallel"] as const).reduce((best, key) => (time[key] > time[best] ? key : best), "world" as "world" | "inOrder" | "parallel");
  const largestWords = { world: "waiting on the outside world", inOrder: "thinking that has to happen in order", parallel: "thinking split across the copies" }[largest];

  /* ---------------- card 4: the loop ---------------- */
  const run = useMemo(() => takeoff(lab.returns, lab.bottleneck), [lab.returns, lab.bottleneck]);
  const ghost = useMemo(
    () => (lab.prevReturns > 0 ? takeoff(lab.prevReturns, lab.prevBottleneck) : null),
    [lab.prevReturns, lab.prevBottleneck],
  );
  const gap = takeoffGap(run);
  const abilityNow = abilityAt(run, lab.loopYear);
  const effortNow = researchEffort(abilityNow, lab.bottleneck);
  const effortCap = lab.bottleneck > 0 ? 1 / lab.bottleneck : Number.POSITIVE_INFINITY;
  const changeLoop = (patch: { returns?: number; bottleneck?: number }) =>
    setState({ ...patch, prevReturns: lab.returns, prevBottleneck: lab.bottleneck });
  const loopNote = (() => {
    if (run.offChartYear >= 0) {
      return `Each gain in ability speeds up the research that makes the next gain, and with returns above 1 every doubling arrives sooner than the last. The curve leaves the chart, past a million times, in year ${fmt(
        run.offChartYear,
        1,
      )}. This runaway is what people mean by an intelligence explosion.`;
    }
    if (gap >= 0) {
      return lab.bottleneck > 0
        ? `The human-only share caps research speed at ${times(effortCap)} today's, so the loop cannot run away forever. It still gets to 100× in year ${fmt(run.beyondYear, 1)}, because the cap is far above where it started.`
        : `With returns at exactly 1, each doubling takes the same time: steady exponential growth, with no runaway. 100× arrives in year ${fmt(run.beyondYear, 1)}.`;
    }
    return lab.returns < 1
      ? "With returns below 1, each doubling of research effort buys less than a doubling of ability. AI keeps improving after it matches people, but more and more slowly."
      : `The human-only share caps research speed at ${times(effortCap)} today's. The loop speeds up for a while, then settles into slow, steady growth, and never reaches 100× within ${HORIZON} years.`;
  })();

  /* ---------------- card 5: forecasts ---------------- */
  const horizon = horizonAfter(lab.ahead, lab.doubling);
  const measuredHorizon = horizonAfter(lab.ahead, MEASURED_DOUBLING_MONTHS);

  /* ---------------- card 6: claims ---------------- */
  const taggedRight = lab.tags.filter((tag, index) => tag === CLAIMS[index].kind).length;
  const taggedCount = lab.tags.filter((tag) => tag !== null).length;
  const tagClaim = (index: number, kind: ClaimKind) => {
    const next = [...lab.tags];
    next[index] = kind;
    setState({ tags: encodeTags(next), claim: index });
    narrate(kind === CLAIMS[index].kind ? "Right." : "Not quite: read why under the claim.");
  };

  return (
    <div className="asi-lab">
      {/* ============================ 1 ============================ */}
      {page === 0 && (
        <LabSurface label="Narrow, general, and how good" className="asi-card asi-card--levels">
          <SurfaceHeading kicker="1 of 6 · two axes" title="AGI and ASI are corners of one table" />
          <p className="asi-lede">
            Researchers at Google DeepMind proposed grading AI on two separate questions. <b>How good</b> is it, compared with
            people? And <b>how general</b> is it: one task, or a wide range? Their table, drawn here, gives AGI and ASI exact places.
            Pick a system below, then click the cell where you think the table puts it.
          </p>
          <PredictCheck
            prompt="Has any AI system already beaten every human at something?"
            options={["No: that would be superintelligence", "Yes, at a few narrow tasks", "Yes, at most tasks"]}
            answer={1}
            picked={lab.levelGuess}
            onPick={(index) => setState({ levelGuess: index })}
            explanation={
              <>
                The table&apos;s top row, <b>Superhuman</b>, already has AlphaFold, AlphaZero and Stockfish in the narrow column:
                each outperforms every person at one task. The general column above level 1 says &ldquo;not yet achieved&rdquo;.
                ASI is not &ldquo;superhuman at something&rdquo;; it is the top-right corner, superhuman across a wide range of
                tasks.
              </>
            }
          />
          <div className="asi-chips" role="group" aria-label="Systems to place">
            {SYSTEMS.map((system, index) => {
              const placed = lab.placement[index] >= 0;
              return (
                <button
                  type="button"
                  key={system.id}
                  aria-pressed={lab.chip === index}
                  className={`${placed ? "is-placed" : ""}${allPlaced ? (lab.placement[index] === system.cell ? " is-right" : " is-wrong") : ""}`}
                  onClick={() => setState({ chip: index })}
                >
                  {system.name}
                </button>
              );
            })}
          </div>
          <p className="asi-caption">
            Placing <b>{chip.name}</b>: {chip.what}. Click a cell in the table.
          </p>
          <div className="asi-split asi-split--grid">
            <LevelsGrid selected={lab.cell} placement={lab.placement} reveal={allPlaced} onCell={placeChip} />
            <div className="asi-side">
              <div className={`asi-cellinfo${lab.cell === ASI_CELL ? " is-asi" : ""}`} role="status">
                <span>
                  Level {selectedLevel.level} · {selectedLevel.name} · {selectedColumn}
                </span>
                <strong>{selectedLevel.bar}</strong>
                <p>
                  {selectedColumn === "narrow" ? "Narrow" : "General"} examples in the table:{" "}
                  {selectedColumn === "narrow" ? selectedLevel.narrow : selectedLevel.general}.
                </p>
              </div>
              <GoalMeter
                label="Placed where the table puts them"
                value={placedRight / SYSTEMS.length}
                target={1}
                display={allPlaced ? `${placedRight} of ${SYSTEMS.length}` : `${placedCount} of ${SYSTEMS.length} placed`}
                done={placedRight === SYSTEMS.length}
                doneText={`${SYSTEMS.length} of ${SYSTEMS.length} · every system where the table puts it`}
                hint={allPlaced ? "Ticks and crosses show the table's answers. Move any system by selecting it and clicking another cell." : "Answers appear once all eight are placed."}
              />
              <div className="asi-actions">
                <button
                  type="button"
                  className="quiet-action"
                  disabled={placedCount === 0}
                  onClick={() => setState({ place: encodePlacement(SYSTEMS.map(() => -1)), chip: 0 })}
                >
                  Clear the table
                </button>
                <button
                  type="button"
                  className="quiet-action"
                  onClick={() => setState({ place: encodePlacement(SYSTEMS.map((system) => system.cell)), cell: ASI_CELL })}
                >
                  Show the table&apos;s answers
                </button>
              </div>
            </div>
          </div>
          <div className="metric-row">
            <Metric label="Narrow AI" value="left column" />
            <Metric label="AGI in most definitions" value="general, level 2" tone="forward" />
            <Metric label="ASI" value="general, level 5" tone="gradient" />
            <Metric label="Filled general cells" value="levels 0 and 1" />
          </div>
          <p className="lab-note">
            The paper calls level 2 in the general column &ldquo;Competent AGI&rdquo; and says it is probably the best catch-all for
            many existing definitions of AGI. The general column above level 1 reads &ldquo;not yet
            achieved&rdquo; as of the paper, which was published in 2024. &ldquo;Superhuman&rdquo; already exists, one task at a
            time; what nobody has built is the top of the general column.
          </p>
        </LabSurface>
      )}

      {/* ============================ 2 ============================ */}
      {page === 1 && (
        <LabSurface label="One profile, three verdicts" className="asi-card asi-card--profile">
          <SurfaceHeading kicker="2 of 6 · definitions" title="The same scores, three different answers to “is it AGI?”" />
          <p className="asi-lede">
            A 2025 paper by Dan Hendrycks and colleagues defines AGI as matching a well-educated adult across ten areas of thinking,
            each worth 10 points. Here are the scores its authors gave two models. The shape is lopsided, or <b>jagged</b>: strong
            in some areas, at zero in others. Pick a rule and see how far from AGI that same shape is.
          </p>
          <PredictCheck
            prompt="GPT-5 scores 57 out of 100 on this test. If AGI instead meant “adult level in every one of the ten areas”, how close would GPT-5 be?"
            options={["About 57%", "About 30%", "0%"]}
            answer={2}
            picked={lab.profileGuess}
            onPick={(index) => setState({ profileGuess: index })}
            explanation={
              <>
                Its long-term memory storage scores 0 of 10, so under a rule where the weakest area decides, it is at 0%. Choose{" "}
                <b>Weakest domain decides</b> and watch the reading drop from 57% to 0% without a single score changing. Which
                answer you get depends on the definition, not on the model.
              </>
            }
          />
          <div className="asi-pick">
            <SegmentedControl
              label="Model"
              value={lab.profile}
              options={[
                { value: "gpt4", label: "GPT-4" },
                { value: "gpt5", label: "GPT-5" },
              ]}
              onChange={(value) => setState({ profile: value })}
            />
            <SegmentedControl
              label="Rule for “how close to AGI”"
              value={lab.rule}
              options={(["sum", "weakest", "count"] as const).map((rule) => ({ value: rule, label: RULE_NAMES[rule] }))}
              onChange={(value) => setState({ rule: value })}
            />
          </div>
          <div className="asi-split asi-split--plot">
            <ProfileChart scores={scores} other={PROFILES[otherModel]} model={lab.profile} otherModel={otherModel} />
            <div className="asi-side">
              <div className="asi-verdicts" role="list" aria-label="How close to AGI under each rule">
                {(["sum", "weakest", "count"] as const).map((rule) => (
                  <div key={rule} role="listitem" className={`asi-verdict${lab.rule === rule ? " is-active" : ""}`}>
                    <span>{RULE_NAMES[rule]}</span>
                    <i aria-hidden="true">
                      <b style={{ width: `${readings[rule] * 100}%` }} />
                    </i>
                    <strong>{pct(readings[rule])}</strong>
                  </div>
                ))}
              </div>
              <p className="asi-caption">Bars: how close to AGI the same profile is under each rule. A full bar would be AGI.</p>
            </div>
          </div>
          <FormulaWithValues
            label={RULE_NAMES[lab.rule]}
            tone="forward"
            expression={
              lab.rule === "sum"
                ? `${scores.join(" + ")} out of 100`
                : lab.rule === "weakest"
                  ? `lowest of the ten = ${DOMAINS[weakestIndex].name.toLowerCase()}, ${scores[weakestIndex]} of 10`
                  : `areas scoring 10 of 10 = ${strongCount} of 10`
            }
            result={pct(readings[lab.rule])}
            detail={
              lab.rule === "sum"
                ? "This is the paper's own AGI score: every area counts equally, so strength in one area makes up for weakness in another."
                : lab.rule === "weakest"
                  ? "A system is only as general as its weakest area. This reading is stricter than the paper's own."
                  : "Count only the areas where the model matches the adult in full. This reading is also stricter than the paper's."
            }
          />
          <div className="metric-row">
            <Metric label={`${PROFILE_NAMES[lab.profile]}'s AGI score`} value={pct(readings.sum)} tone="forward" />
            <Metric label="Weakest area" value={`${DOMAINS[weakestIndex].short}: ${scores[weakestIndex]}`} tone="loss" />
            <Metric label="Areas at adult level" value={`${strongCount} of 10`} />
            <Metric label="Highest score this test allows" value="100%: one adult" />
          </div>
          <p className="lab-note">
            {zeroDomains.length > 0
              ? `${PROFILE_NAMES[lab.profile]} scores 0 in ${zeroDomains.join(", ")}. `
              : ""}
            Notice what the chart cannot show: its outer edge is a well-educated adult, so a system far beyond every person would
            score the same 100% as one that only matches the adult. A test built to define AGI has no room for ASI.
          </p>
        </LabSurface>
      )}

      {/* ============================ 3 ============================ */}
      {page === 2 && (
        <LabSurface label="Faster, more, or better" className="asi-card asi-card--speed">
          <SurfaceHeading kicker="3 of 6 · what “super” could mean" title="Think faster, run more copies, or think better" />
          <p className="asi-lede">
            Picture a research project that would take one expert many years. A mind could be beyond human in three different
            ways: it could think <b>faster</b>, it could run as many <b>copies</b> at once, or it could think <b>better</b>. The
            first two are arithmetic, so this card can compute them. Then it adds the parts no extra thinking can speed up.
          </p>
          <PredictCheck
            prompt="A million copies, each thinking 1,000× faster than a person, take on a 100 expert-year project. If 10% of the work is waiting on the outside world, such as experiments and building things, about how long does it take?"
            options={["A few seconds", "About a day", "About 10 years"]}
            answer={2}
            picked={lab.speedGuess}
            onPick={(index) => setState({ speedGuess: index })}
            explanation={
              <>
                The thinking finishes in seconds, but 10% of 100 expert-years is 10 years of experiments, and no amount of thinking
                makes a cell grow faster. Set <b>Speed</b> to 1,000×, <b>Copies</b> to 1 million× and <b>Waits on the world</b> to
                10%, then set the world share back to 0% and see the time drop to about 3 seconds.
              </>
            }
          />
          <div className="asi-controls">
            <RangeControl
              label="Project"
              min={0}
              max={PROJECT_STOPS.length - 1}
              step={1}
              value={stopIndex(PROJECT_STOPS, lab.work)}
              format={(index) => `${PROJECT_STOPS[index] ?? lab.work} expert-years`}
              onChange={(index) => setState({ work: PROJECT_STOPS[index] ?? 100 })}
            />
            <RangeControl
              label="Speed"
              min={0}
              max={MULTIPLIER_STOPS.length - 1}
              step={1}
              value={stopIndex(MULTIPLIER_STOPS, lab.speed)}
              format={(index) => times(MULTIPLIER_STOPS[index] ?? lab.speed)}
              onChange={(index) => setState({ speed: MULTIPLIER_STOPS[index] ?? 1 })}
            />
            <RangeControl
              label="Copies"
              min={0}
              max={MULTIPLIER_STOPS.length - 1}
              step={1}
              value={stopIndex(MULTIPLIER_STOPS, lab.copies)}
              format={(index) => times(MULTIPLIER_STOPS[index] ?? lab.copies)}
              onChange={(index) => setState({ copies: MULTIPLIER_STOPS[index] ?? 1 })}
            />
            <RangeControl
              label="Must happen in order"
              min={0}
              max={SHARE_STOPS.length - 1}
              step={1}
              value={stopIndex(SHARE_STOPS, lab.inOrder)}
              format={(index) => pct(SHARE_STOPS[index] ?? lab.inOrder)}
              onChange={(index) => setState({ inOrder: SHARE_STOPS[index] ?? 0 })}
            />
            <RangeControl
              label="Waits on the world"
              min={0}
              max={SHARE_STOPS.length - 1}
              step={1}
              value={stopIndex(SHARE_STOPS, lab.world)}
              format={(index) => pct(SHARE_STOPS[index] ?? lab.world)}
              onChange={(index) => setState({ world: SHARE_STOPS[index] ?? 0 })}
            />
          </div>
          <ProjectClock alone={alone} now={time} />
          <FormulaWithValues
            label="Calendar time"
            tone="forward"
            expression={`${lab.work} × [${pct(lab.world)} + ${pct(1 - lab.world)} × (${pct(lab.inOrder)} ÷ ${times(lab.speed)} + ${pct(
              1 - lab.inOrder,
            )} ÷ (${times(lab.speed)} × ${times(lab.copies)}))] years`}
            result={span(time.total)}
            detail="The world share takes as long as it always did. Work that must happen in order runs at one copy's speed. Only the rest is split across every copy."
          />
          <div className="metric-row">
            <Metric label="One expert alone" value={span(alone)} />
            <Metric label="With these settings" value={span(time.total)} tone="forward" />
            <Metric label="Most of the time goes to" value={largest === "world" ? "the world" : largest === "inOrder" ? "in-order thinking" : "split thinking"} tone="loss" />
            <Metric label="Fastest possible here" value={span(unbounded)} />
          </div>
          <p className="lab-note">
            Most of the time now goes to {largestWords}. {lab.world > 0 || lab.inOrder > 0
              ? `However fast and however many, this project cannot finish in under ${span(unbounded)}: that floor is set by the parts no copy can split and no speed can shrink.`
              : "With nothing waiting on the world and nothing forced into order, speed and copies multiply without limit, which real projects never allow."}{" "}
            The third way, thinking better, has no slider: a better idea is not a faster one, and nobody knows how to put it on a
            scale.
          </p>
        </LabSurface>
      )}

      {/* ============================ 4 ============================ */}
      {page === 3 && (
        <LabSurface label="A loop that feeds itself" className="asi-card asi-card--loop">
          <SurfaceHeading kicker="4 of 6 · the intelligence explosion" title="What happens when AI does AI research?" />
          <p className="asi-lede">
            The best-known argument for ASI is a loop: once AI can do AI research, a better AI makes research faster, and faster
            research makes a better AI. This card runs a small model of that loop. Two numbers nobody has measured decide its shape,
            and you control both.
          </p>
          <PredictCheck
            prompt="With returns at 1.5 and nothing that only people can do, AI goes from matching people to 100× beyond them in 12 years. What happens if 20% of AI research can only be done by people?"
            options={["It still reaches 100×, a few years later", "It never reaches 100× in 40 years", "Progress stops once AI matches people"]}
            answer={1}
            picked={lab.loopGuess}
            onPick={(index) => setState({ loopGuess: index })}
            explanation={
              <>
                If a fifth of the work runs at human speed, research can never get more than 5× faster than today, however clever
                the AI becomes. Set <b>Only people can do</b> to 20%: the loop speeds up for a while, then settles into slow growth
                and ends below 100×. Economists call this a bottleneck: growth is held back by what is essential and hard to
                improve.
              </>
            }
          />
          <div className="asi-loop" aria-label="The loop at the year you have picked">
            <div>
              <span>AI research ability</span>
              <strong>{times(Number(abilityNow.toPrecision(3)))}</strong>
              <small>1× = today&apos;s researchers</small>
            </div>
            <em aria-hidden="true">→</em>
            <div>
              <span>Research speed</span>
              <strong>{times(Number(effortNow.toPrecision(3)))}</strong>
              <small>{lab.bottleneck > 0 ? `capped at ${times(effortCap)}` : "no cap"}</small>
            </div>
            <em aria-hidden="true">→</em>
            <div>
              <span>Next improvement</span>
              <strong>returns {fmt(lab.returns, 2)}</strong>
              <small>{lab.returns > 1 ? "comes sooner" : lab.returns === 1 ? "same pace" : "comes later"}</small>
            </div>
            <em className="asi-loop__back" aria-hidden="true">↺</em>
          </div>
          <TakeoffPlot run={run} ghost={ghost} year={lab.loopYear} />
          <div className="asi-controls asi-controls--three">
            <RangeControl
              label="Returns to research"
              min={0}
              max={RETURN_STOPS.length - 1}
              step={1}
              value={stopIndex(RETURN_STOPS, lab.returns)}
              format={(index) => fmt(RETURN_STOPS[index] ?? lab.returns, 2)}
              onChange={(index) => changeLoop({ returns: RETURN_STOPS[index] ?? 1.5 })}
            />
            <RangeControl
              label="Only people can do"
              min={0}
              max={BOTTLENECK_STOPS.length - 1}
              step={1}
              value={stopIndex(BOTTLENECK_STOPS, lab.bottleneck)}
              format={(index) => pct(BOTTLENECK_STOPS[index] ?? lab.bottleneck)}
              onChange={(index) => changeLoop({ bottleneck: BOTTLENECK_STOPS[index] ?? 0 })}
            />
            <RangeControl
              label="Year"
              min={0}
              max={HORIZON}
              step={0.5}
              value={lab.loopYear}
              format={(value) => fmt(value, 1)}
              onChange={(value) => setState({ loopYear: value })}
            />
          </div>
          <div className="metric-row">
            <Metric label="Matches people" value={`year ${fmt(run.matchYear, 1)}`} />
            <Metric label={`Reaches ${BEYOND}×`} value={run.beyondYear >= 0 ? `year ${fmt(run.beyondYear, 1)}` : `not by year ${HORIZON}`} tone="gradient" />
            <Metric label="The climb from 1× to 100×" value={gap >= 0 ? `${fmt(gap, 1)} years` : "never, here"} tone="forward" />
            <Metric label="Leaves the chart" value={run.offChartYear >= 0 ? `year ${fmt(run.offChartYear, 1)}` : "no"} />
          </div>
          <p className="lab-note" role="status">
            {loopNote} Getting from 0.1× to 1× took 6 years in every run; the climb after that is the part people argue about, and
            they call a short one a <b>fast takeoff</b> and a long one a <b>slow takeoff</b>. The clock itself is made up.
          </p>
        </LabSurface>
      )}

      {/* ============================ 5 ============================ */}
      {page === 4 && (
        <LabSurface label="Reading forecasts" className="asi-card asi-card--forecast">
          <SurfaceHeading kicker="5 of 6 · forecasts" title="When? The answer depends on the question and the assumption" />
          <p className="asi-lede">
            Two kinds of evidence come up whenever people argue about when AGI will arrive. One is asking experts. The other is
            measuring a trend and carrying it forward. This card shows one of each, and how much each depends on how it is set up.
          </p>
          <PredictCheck
            prompt={`In late 2023, ${SURVEY.respondents.toLocaleString("en-US")} AI researchers were asked about “machines beating people at every task” and about “every occupation being fully automatable”. How far apart were their 50% years?`}
            options={["About the same year", "About 20 years apart", "About 70 years apart"]}
            answer={2}
            picked={lab.surveyGuess}
            onPick={(index) => setState({ surveyGuess: index })}
            explanation={
              <>
                {SURVEY.hlmi.p50} for every task, {SURVEY.faol.p50} for every occupation: {SURVEY.faol.p50 - SURVEY.hlmi.p50} years
                apart, from the same people. The two questions describe nearly the same world, so the gap says more about how the
                question is asked than about machines.
              </>
            }
          />
          <div className="asi-pick">
            <SegmentedControl
              label="Survey"
              value={lab.survey}
              options={[
                { value: "2023", label: "Late 2023" },
                { value: "2022", label: "2022" },
              ]}
              onChange={(value) => setState({ survey: value })}
            />
          </div>
          <SurveyTimeline survey={lab.survey} />
          <h4 className="asi-subhead">A measured trend, carried forward</h4>
          <p className="asi-caption asi-caption--lead">
            METR timed software tasks by how long they take people, then found the longest tasks AI finishes half the time. That
            length was about {HORIZON_START_MINUTES} minutes in early {HORIZON_START_YEAR}, after doubling about every{" "}
            {MEASURED_DOUBLING_MONTHS} months since 2019.
          </p>
          <HorizonPlot doubling={lab.doubling} ahead={lab.ahead} />
          <div className="asi-controls asi-controls--two">
            <RangeControl
              label="Years after early 2025"
              min={0}
              max={AHEAD_MAX}
              step={0.5}
              value={lab.ahead}
              format={(value) => fmt(value, 1)}
              onChange={(value) => setState({ ahead: value })}
            />
            <SegmentedControl
              label="Assumed doubling time"
              value={String(lab.doubling)}
              options={DOUBLING_STOPS.map((months) => ({ value: String(months), label: `${months} mo` }))}
              onChange={(value) => setState({ doubling: Number(value) })}
            />
          </div>
          <div className="metric-row">
            <Metric label="Carried forward" value={workSpan(horizon)} tone="forward" />
            <Metric label="At the measured 7 months" value={workSpan(measuredHorizon)} />
            <Metric label="Year" value={fmt(HORIZON_START_YEAR + lab.ahead, 1).replace(/\.0$/, "")} />
            <Metric label="Survey's 50% year, every task" value={String(lab.survey === "2023" ? SURVEY.hlmi.p50 : SURVEY.hlmi.p50Before)} />
          </div>
          <p className="lab-note">
            {lab.doubling === MEASURED_DOUBLING_MONTHS
              ? `At the measured pace, tasks of ${workSpan(horizonAfter(5, MEASURED_DOUBLING_MONTHS))} come within reach 5 years on, which is the paper's own projection, and it adds "if these results generalize to real-world software tasks". `
              : `At ${lab.doubling} months instead of ${MEASURED_DOUBLING_MONTHS}, the same start reaches ${workSpan(
                  horizonAfter(5, lab.doubling),
                )} after 5 years instead of ${workSpan(horizonAfter(5, MEASURED_DOUBLING_MONTHS))}. `}
            Everything right of the shaded band is an assumption that the trend continues. The survey moved too: the 2022 version
            put machines beating people at every task in {SURVEY.hlmi.p50Before}, 13 years later than the 2023 one.
          </p>
        </LabSurface>
      )}

      {/* ============================ 6 ============================ */}
      {page === 5 && (
        <LabSurface label="Challenge: sort the claims" className="asi-card asi-card--claims">
          <SurfaceHeading kicker="6 of 6 · challenge" title="Measured, defined, argued, or guessed?" />
          <p className="asi-lede">
            Talk about AGI and ASI mixes four kinds of statement, and they deserve different amounts of trust. A <b>measurement</b>{" "}
            reports something observed. A <b>definition</b> chooses how to use a word. An <b>argument</b> follows from assumptions.
            A <b>forecast</b> guesses at the future. Sort each claim; every one comes from an earlier card.
          </p>
          <ClaimList tags={lab.tags} onTag={tagClaim} />
          <GoalMeter
            label="Sorted correctly"
            value={taggedRight / CLAIMS.length}
            target={1}
            display={`${taggedRight} of ${CLAIMS.length} right · ${taggedCount} sorted`}
            done={taggedRight === CLAIMS.length}
            doneText={`${CLAIMS.length} of ${CLAIMS.length} · every claim sorted`}
            hint="You can change an answer at any time. The reason appears under each claim once you choose."
          />
          <div className="asi-actions">
            <button type="button" className="quiet-action" disabled={taggedCount === 0} onClick={() => setState({ tags: encodeTags(CLAIMS.map(() => null)) })}>
              Start over
            </button>
          </div>
          <p className="lab-note" role="status">
            {taggedRight === CLAIMS.length
              ? "All sorted. Next time someone says AGI is close, or ASI is impossible, ask which of the four kinds of claim they are making. Most strong statements about superintelligence are arguments or forecasts, resting on definitions, with a few measurements underneath."
              : "Some claims mix kinds. A survey result is measured, but what it reports is a forecast; a trend is measured, but carrying it forward is not. Sort each claim by what it asserts."}
          </p>
        </LabSurface>
      )}
      <StepPager steps={stepNames} current={page} goToStep={goToStep} />
    </div>
  );
}
