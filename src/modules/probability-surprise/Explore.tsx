import { useMemo, type ReactNode } from "react";
import {
  FormulaWithValues,
  GoalMeter,
  LabSurface,
  LineChart,
  Metric,
  PredictCheck,
  StepPager,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  CHALLENGE_DRAWS,
  CHALLENGE_GOAL,
  CHALLENGE_ROUNDS,
  EQUAL_BARS,
  MAX_BAR,
  MIN_BAR,
  PROMPTS,
  SKY_BARS,
  SKY_SCORES,
  TEST_COUNT,
  TEST_SIZES,
  WORDS,
  biggestGap,
  challengeScore,
  challengeTally,
  conditional,
  crossEntropy,
  drawSamples,
  entropy,
  expectedValue,
  fairChoices,
  klDivergence,
  likelihoodOf,
  pileStats,
  posterior as bayesPosterior,
  priorFor,
  runningShare,
  setBar,
  shapeShares,
  sigmoid,
  simulateTests,
  softmax,
  sum,
  surprise,
  tally,
  variance,
} from "./chance";
import { BayesArea, Columns, FairBoxes, Legend, PairPicture, SigmoidCurve, SurpriseBars, WholeStrip, bits, pct } from "./pictures";
import { BellPile, Skyline } from "./plots";
import {
  MAX_DRAWS,
  PRIOR_MAX,
  PRIOR_MIN,
  SCORE_LIMIT,
  TEMPERATURE_MAX,
  TEMPERATURE_MIN,
  formatList,
  readState,
  type LabState,
} from "./state";
import { STEPS } from "./steps";

type Patch = Parameters<ModuleContext["setState"]>[0];
interface CardProps {
  s: LabState;
  set: (patch: Patch) => void;
  narrate: (message: string) => void;
}

const WORD_OPTIONS = WORDS.map((word, index) => ({ value: String(index), label: word }));
const toShares = (bars: ReadonlyArray<number>) => bars.map((bar) => bar / 100);
const SKY = toShares(SKY_BARS);
const sameList = (a: ReadonlyArray<number>, b: ReadonlyArray<number>) => a.every((value, index) => value === b[index]);
const two = (value: number) => value.toFixed(2);

/** The prompt with its blank, so every card says which sentence the bars belong to. */
function Prompt({ text, children }: { text: string; children?: ReactNode }) {
  return (
    <p className="ps-prompt">
      <span className="ps-prompt__text">
        “{text} <b>___</b>”
      </span>
      {children && <span className="ps-prompt__aside">{children}</span>}
    </p>
  );
}

/** Five sliders in a row, lined up under the five columns they control. */
function SliderRow({
  name,
  values,
  onChange,
  min = MIN_BAR,
  max = MAX_BAR,
  step = 1,
  format = (value: number) => `${value}%`,
}: {
  name: (word: string) => string;
  values: ReadonlyArray<number>;
  onChange: (index: number, value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (value: number) => string;
}) {
  return (
    <div className="ps-sliders">
      {WORDS.map((word, index) => (
        <RangeControl
          key={word}
          label={name(word)}
          min={min}
          max={max}
          step={step}
          value={values[index]}
          format={format}
          onChange={(value) => onChange(index, value)}
        />
      ))}
    </div>
  );
}

function BarsCard({ s, set }: CardProps) {
  const shares = toShares(s.bars);
  const counts = s.view === "counts";
  const changed = !sameList(s.bars, SKY_BARS);
  const change = (index: number, value: number) => set({ bars: formatList(setBar(s.bars, index, value)) });
  return (
    <LabSurface label="Bars that add to one" className="ps-card">
      <SurfaceHeading kicker="Guess the next word" title="Bars that add to one" />
      <Prompt text={PROMPTS[0].text}>Five words could come next. The numbers are made up for this lab.</Prompt>
      <WholeStrip
        label={`One whole shared among five words: ${WORDS.map((word, index) => `${word} ${s.bars[index]}${counts ? " of 100 stories" : "%"}`).join(", ")}.`}
        parts={WORDS.map((word, index) => ({ name: word, share: shares[index], text: counts ? String(s.bars[index]) : `${s.bars[index]}%` }))}
        cells={counts}
        caption={counts ? "100 stories that begin “the sky is”, lined up" : "one whole: all the chance there is"}
      />
      <Columns
        label="Chance of each next word. Drag a bar, or focus it and press the arrow keys."
        names={WORDS}
        series={[
          {
            id: "truth",
            name: counts ? "stories out of 100" : "chance",
            shares,
            labels: s.bars.map((bar) => (counts ? `${bar} of 100` : `${bar}%`)),
          },
        ]}
        ghost={changed ? SKY : undefined}
        editable={{ series: 0, name: (word) => `Bar for ${word}`, onChange: change }}
      />
      <p className="ps-hint">
        <span aria-hidden="true">↕</span> Drag any bar up or down. The other four shrink or grow to keep the total at{" "}
        {counts ? "100 stories" : "100%"}.{changed ? " Dashed outlines show where the bars started." : ""}
      </p>
      <SliderRow name={(word) => `Chance of ${word}`} values={s.bars} onChange={change} />
      <div className="ps-controls">
        <SegmentedControl
          label="Show as"
          value={s.view}
          options={[
            { value: "percent", label: "percent" },
            { value: "counts", label: "counts out of 100 stories" },
          ]}
          onChange={(value) => set({ view: value })}
        />
        <button type="button" className="ps-button" disabled={!changed} onClick={() => set({ bars: formatList(SKY_BARS) })}>
          Reset the bars
        </button>
      </div>
      <FormulaWithValues
        label="The five shares always add to one whole"
        expression={shares.map((share) => two(share)).join(" + ")}
        result={`= ${two(sum(shares))}`}
        detail={`P(${WORDS[0]}) = ${two(shares[0])} is read “the probability of ${WORDS[0]} is ${two(shares[0])}”: the same thing as ${s.bars[0]}%, or ${s.bars[0]} stories in 100. A full set of bars like this is called a probability distribution.`}
      />
    </LabSurface>
  );
}

function DrawCard({ s, set, narrate }: CardProps) {
  const shares = toShares(s.bars);
  const barsKey = formatList(s.bars);
  const stream = useMemo(() => drawSamples(toShares(barsKey.split(",").map(Number)), MAX_DRAWS, s.luck), [barsKey, s.luck]);
  const draws = stream.slice(0, s.draws);
  const counts = tally(draws);
  const total = draws.length;
  const gap = biggestGap(counts, shares);
  const follow = WORDS[s.follow];
  const line = useMemo(() => {
    const running = runningShare(draws, s.follow);
    const stride = Math.max(1, Math.ceil(running.length / 400));
    return running
      .map((share, index) => ({ x: index + 1, y: share * 100 }))
      .filter((_, index) => index % stride === stride - 1 || index === running.length - 1 || index < 12);
    // `draws` is a slice of the memoised stream, so its length and the stream identify it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream, s.draws, s.follow]);
  const xMax = Math.max(10, total);
  const recent = draws.slice(-12);
  const setDraws = (next: number) => {
    const clamped = Math.min(MAX_DRAWS, Math.max(0, next));
    set({ draws: clamped });
    if (clamped > 0) {
      const made = tally(stream.slice(0, clamped));
      narrate(`${clamped} draws: ${WORDS.map((word, index) => `${word} ${made[index]}`).join(", ")}.`);
    }
  };
  return (
    <LabSurface label="Draw from the bars" className="ps-card">
      <SurfaceHeading kicker="Sampling" title="Draw from the bars" />
      <PredictCheck
        prompt="You are about to draw 10 words from these bars. Will the tally of the 10 match the bars?"
        options={["Yes, it will match", "No, but more draws bring it closer", "No, and more draws will not help"]}
        answer={1}
        picked={s.predictDraws}
        onPick={(index) => set({ predictDraws: index })}
        explanation="Ten draws can only make shares in steps of 10%, and luck pushes them around on top of that. Press 10, then 100, then 1,000 and watch Biggest gap shrink: the tally settles onto the bars as draws pile up."
      />
      <div className="ps-controls">
        <SegmentedControl
          label="Number of draws"
          value={String(s.draws)}
          options={[
            { value: "0", label: "none" },
            { value: "1", label: "1" },
            { value: "10", label: "10" },
            { value: "100", label: "100" },
            { value: "1000", label: "1,000" },
          ]}
          onChange={(value) => setDraws(Number(value))}
        />
        <button type="button" className="ps-button" disabled={s.draws >= MAX_DRAWS} onClick={() => setDraws(s.draws + 1)}>
          Draw one more
        </button>
        <button
          type="button"
          className="ps-button"
          onClick={() => {
            set({ luck: (s.luck % 999) + 1 });
            narrate("New luck: the same number of draws, drawn again.");
          }}
        >
          New luck
        </button>
      </div>
      <p className="ps-tape" aria-live="polite">
        <span>{total === 0 ? "No draws yet." : total > recent.length ? `Last ${recent.length} of ${total.toLocaleString("en-US")} draws:` : total === 1 ? "The draw:" : `The ${total} draws, in order:`}</span>
        {recent.map((draw, index) => (
          <code key={index} className={draw === s.follow ? "is-follow" : ""}>
            {WORDS[draw]}
          </code>
        ))}
      </p>
      <Legend
        items={[
          { id: "truth", text: "the bars: each word’s chance" },
          { id: "tally", text: "the tally: its share of the draws so far (count on top)" },
        ]}
      />
      <Columns
        label={`The bars beside the tally of ${total} draws. ${WORDS.map((word, index) => `${word}: chance ${pct(shares[index])}, drawn ${counts[index]} times`).join("; ")}.`}
        names={WORDS}
        series={[
          { id: "truth", name: "chance", shares },
          {
            id: "tally",
            name: "share of the draws",
            shares: counts.map((count) => (total > 0 ? count / total : 0)),
            labels: counts.map((count) => (total > 0 ? String(count) : "")),
          },
        ]}
        selected={s.follow}
        onSelect={(index) => set({ follow: index })}
        footers={counts.map((count) => (total > 0 ? `${count} of ${total.toLocaleString("en-US")}\n= ${pct(count / total, total >= 100 ? 1 : 0)}` : "not drawn yet"))}
      />
      <div className="metric-row">
        <Metric label="Draws so far" value={total.toLocaleString("en-US")} />
        <Metric label={`Share that were ${follow}`} value={total > 0 ? pct(counts[s.follow] / total, 1) : "no draws"} tone="forward" />
        <Metric label="Biggest gap, tally to bars" value={total > 0 ? `${(gap * 100).toFixed(1)} points` : "no draws"} tone="loss" />
      </div>
      <h4 className="ps-subhead">Follow one word as the draws pile up</h4>
      <p className="ps-hint">Press a word’s name under the bars to follow it. The line is its share of all the draws so far.</p>
      {total > 0 ? (
        <LineChart
          label={`Share of the draws that were ${follow}, after each draw`}
          series={[
            { id: "share", name: `share of draws that were ${follow}`, points: line, tone: "forward", format: (value) => `${value.toFixed(1)}%` },
            {
              id: "chance",
              name: `${follow}’s chance (its bar)`,
              points: [
                { x: 0, y: shares[s.follow] * 100 },
                { x: xMax, y: shares[s.follow] * 100 },
              ],
              tone: "gradient",
              dash: "dashed",
              format: (value) => `${value.toFixed(0)}%`,
            },
          ]}
          xLabel="draws so far"
          yLabel={`share that were ${follow}, %`}
          xDomain={[0, xMax]}
          yDomain={[0, 100]}
        />
      ) : (
        <p className="ps-empty">Choose a number of draws above and the line appears here.</p>
      )}
    </LabSurface>
  );
}

function BellCard({ s, set }: CardProps) {
  const shares = toShares(s.bars);
  const top = shares.indexOf(Math.max(...shares));
  const points = WORDS.map((_, index) => (index === top ? 1 : 0));
  const mean = expectedValue(shares, points);
  const spread = variance(shares, points);
  const pile = useMemo(() => simulateTests(mean, s.testSize), [mean, s.testSize]);
  const stats = pileStats(pile);
  const predictedSd = Math.sqrt(spread / s.testSize);
  return (
    <LabSurface label="Averages pile up in a bell" className="ps-card">
      <SurfaceHeading kicker="Mean, variance and the bell curve" title="Averages pile up in a bell" />
      <p className="ps-lead">
        A guesser always answers with the tallest bar, <b>{WORDS[top]}</b>. Give it 1 point for a hit and 0 for a miss.
        One answer is all or nothing, but a test averages many answers.
      </p>
      <div className="ps-table-wrap">
        <table className="ps-table ps-table--sum">
          <caption>One answer: a number attached to each outcome</caption>
          <thead>
            <tr>
              <th scope="col">next word</th>
              <th scope="col">chance</th>
              <th scope="col">points</th>
              <th scope="col">chance × points</th>
              <th scope="col">chance × (points − mean)²</th>
            </tr>
          </thead>
          <tbody>
            {WORDS.map((word, index) => (
              <tr key={word} className={index === top ? "is-hit" : ""}>
                <th scope="row">{word}</th>
                <td>{two(shares[index])}</td>
                <td>{points[index]}</td>
                <td>{two(shares[index] * points[index])}</td>
                <td>{(shares[index] * (points[index] - mean) ** 2).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row" colSpan={3}>
                add them up
              </th>
              <td>
                <b>{two(mean)}</b> <small>the mean (expected value)</small>
              </td>
              <td>
                <b>{spread.toFixed(4)}</b> <small>the variance</small>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      <SegmentedControl
        label="Questions per test"
        value={String(s.testSize)}
        options={TEST_SIZES.map((size) => ({ value: String(size), label: String(size) }))}
        onChange={(value) => set({ testSize: Number(value) })}
      />
      <BellPile
        pile={pile}
        mean={stats.mean}
        sd={stats.sd}
        label={`${TEST_COUNT.toLocaleString("en-US")} simulated tests of ${s.testSize} questions each. Their scores pile up around a mean of ${two(stats.mean)} with a standard deviation of ${stats.sd.toFixed(3)}. A bell curve drawn from that mean and variance is laid over the pile.`}
      />
      <Legend
        items={[
          { id: "pile", text: `bars: the share of the ${TEST_COUNT.toLocaleString("en-US")} tests that got each score` },
          { id: "inside", text: "tinted bars: within 1 standard deviation of the mean" },
          { id: "bell", text: "curve: the bell (Gaussian) set by the pile’s mean and variance" },
        ]}
      />
      <div className="metric-row">
        <Metric label="Mean of the pile" value={two(stats.mean)} tone="gradient" />
        <Metric label="Variance of the pile" value={stats.variance.toFixed(4)} />
        <Metric label="Standard deviation" value={stats.sd.toFixed(3)} tone="forward" />
      </div>
      <FormulaWithValues
        label="What the bars predict for the pile’s width"
        expression={`variance of one answer ÷ questions = ${spread.toFixed(4)} ÷ ${s.testSize} = ${(spread / s.testSize).toFixed(4)}, and its square root is`}
        result={predictedSd.toFixed(3)}
        tone="forward"
        detail={`The standard deviation is the square root of the variance, back in points. It is the typical distance of one test’s score from the mean, so a ${s.testSize}-question test usually lands within about ${predictedSd.toFixed(3)} of ${two(mean)}. Four times the questions halves it.`}
      />
    </LabSurface>
  );
}

function CountTable({ selectedRow, selectedColumn }: { selectedRow?: number; selectedColumn?: number }) {
  const largest = Math.max(...PROMPTS.flatMap((prompt) => prompt.counts));
  return (
    <table className="ps-table ps-table--counts">
      <caption>200 made-up stories, counted by how they begin and which word comes next</caption>
      <thead>
        <tr>
          <th scope="col">the story begins</th>
          {WORDS.map((word, index) => (
            <th scope="col" key={word} className={selectedColumn === index ? "is-picked" : ""}>
              {word}
            </th>
          ))}
          <th scope="col">stories</th>
        </tr>
      </thead>
      <tbody>
        {PROMPTS.map((prompt, row) => (
          <tr key={prompt.id} className={selectedRow === row ? "is-picked" : ""}>
            <th scope="row">{prompt.text}</th>
            {prompt.counts.map((count, column) => (
              <td key={column} className={selectedColumn === column ? "is-picked" : ""}>
                <span>{count}</span>
                <i style={{ width: `${(count / largest) * 100}%` }} aria-hidden="true" />
              </td>
            ))}
            <td className="ps-table__total">{sum(prompt.counts)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ContextCard({ s, set }: CardProps) {
  const prompt = PROMPTS[s.prompt];
  const row = conditional(s.prompt);
  const total = sum(prompt.counts);
  const top = row.indexOf(Math.max(...row));
  return (
    <LabSurface label="Context changes the bars" className="ps-card">
      <SurfaceHeading kicker="Conditional probability" title="Context changes the bars" />
      <SegmentedControl
        label="Prompt"
        value={String(s.prompt)}
        options={PROMPTS.map((entry, index) => ({ value: String(index), label: `${entry.text} ___` }))}
        onChange={(value) => set({ prompt: Number(value) })}
      />
      <div className="ps-aligned">
        <div className="ps-table-wrap">
          <CountTable selectedRow={s.prompt} />
        </div>
        <p className="ps-arrow">
          <span aria-hidden="true">↓</span> Keep only the “{prompt.text}” row and divide each count by the row’s total, {total}
        </p>
        <Columns
          label={`Chance of each next word after “${prompt.text}”: ${WORDS.map((word, index) => `${word} ${pct(row[index])}`).join(", ")}.`}
          names={WORDS}
          series={[{ id: "truth", name: `chance after “${prompt.text}”`, shares: row }]}
          ghost={s.prompt === 0 ? undefined : SKY}
          footers={prompt.counts.map((count) => `${count} ÷ ${total}`)}
        />
      </div>
      {s.prompt !== 0 && <p className="ps-hint">Dashed outlines: the bars for “{PROMPTS[0].text}”, for comparison.</p>}
      <FormulaWithValues
        label="One row, rescaled to add to one"
        expression={`P(${WORDS[top]} | ${prompt.text}) = ${prompt.counts[top]} ÷ ${total}`}
        result={`= ${two(row[top])}`}
        detail={`Read the upright bar as “given”: the probability of ${WORDS[top]}, given that the story so far is “${prompt.text}”. This is a conditional probability, and it is the one thing a language model outputs: a full set of bars for the next word, given everything before it.`}
      />
    </LabSurface>
  );
}

function BayesCard({ s, set }: CardProps) {
  const prior = priorFor(s.prior / 100);
  const likelihood = likelihoodOf(s.clue);
  const result = bayesPosterior(prior, likelihood);
  const word = WORDS[s.clue];
  const names = PROMPTS.map((prompt) => prompt.text);
  return (
    <LabSurface label="Read the table backwards" className="ps-card">
      <SurfaceHeading kicker="Bayes’ rule" title="Read the table backwards" />
      <p className="ps-lead">
        Now the prompt is hidden and you only see the next word. Which prompt was it? Start from how common each prompt
        is, then let the word shift that belief.
      </p>
      <PredictCheck
        prompt="Suppose “the sky is” stories become rare: 1 story in 10 instead of 1 in 2. The next word is still “blue”. What happens to the chance that the story began “the sky is”?"
        options={["It stays the same", "It falls", "It rises"]}
        answer={1}
        picked={s.predictPrior}
        onPick={(index) => set({ predictPrior: index })}
        explanation="“Blue” is just as typical of sky stories as before, but there are far fewer sky stories to say it, so most “blue”s now come from elsewhere. Set the word to blue and drag the prior down to 10%: the answer falls from 67% to 18%."
      />
      <div className="ps-table-wrap">
        <CountTable selectedColumn={s.clue} />
      </div>
      <div className="ps-controls">
        <SegmentedControl label="The next word was" value={String(s.clue)} options={WORD_OPTIONS} onChange={(value) => set({ clue: Number(value) })} />
      </div>
      <RangeControl
        label="Prior: share of all stories that begin “the sky is”"
        min={PRIOR_MIN}
        max={PRIOR_MAX}
        step={5}
        value={s.prior}
        format={(value) => `${value}%`}
        onChange={(value) => set({ prior: value })}
      />
      <BayesArea
        label={`Bayes’ rule as areas for the word ${word}. ${names.map((name, index) => `${name}: prior ${pct(prior[index], 1)}, chance of ${word} ${pct(likelihood[index])}, product ${pct(result.joint[index], 1)}, posterior ${pct(result.posterior[index], 1)}`).join(". ")}.`}
        names={names}
        short={PROMPTS.map((prompt) => prompt.id)}
        prior={prior}
        likelihood={likelihood}
        joint={result.joint}
        posterior={result.posterior}
        word={word}
      />
      <div className="ps-table-wrap">
        <table className="ps-table ps-table--sum">
          <caption>Prior × likelihood, then rescale so the answers add to one</caption>
          <thead>
            <tr>
              <th scope="col">the story began</th>
              <th scope="col">prior</th>
              <th scope="col">chance of “{word}” (likelihood)</th>
              <th scope="col">prior × likelihood</th>
              <th scope="col">posterior</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name, index) => (
              <tr key={name} className={index === 0 ? "is-hit" : ""}>
                <th scope="row">{name}</th>
                <td>{prior[index].toFixed(3)}</td>
                <td>{two(likelihood[index])}</td>
                <td>{result.joint[index].toFixed(4)}</td>
                <td>
                  <b>{pct(result.posterior[index], 1)}</b>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th scope="row" colSpan={3}>
                add them up
              </th>
              <td>
                <b>{result.evidence.toFixed(4)}</b> <small>all stories that say “{word}”</small>
              </td>
              <td>
                <b>100%</b>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      <FormulaWithValues
        label="Bayes’ rule with the live numbers"
        expression={`P(${names[0]} | ${word}) = ${prior[0].toFixed(3)} × ${two(likelihood[0])} ÷ ${result.evidence.toFixed(4)}`}
        result={`= ${result.posterior[0].toFixed(3)}`}
        detail="The prior is what you believed before the clue. The likelihood is how well each explanation predicts the clue. The posterior is the updated belief: prior × likelihood, divided by the total so the three answers add to one."
      />
    </LabSurface>
  );
}

function SurpriseCard({ s, set }: CardProps) {
  const shares = toShares(s.bars);
  const surprises = shares.map(surprise);
  const word = WORDS[s.word];
  const change = (index: number, value: number) => set({ bars: formatList(setBar(s.bars, index, value)), word: index });
  const pFirst = shares[s.first];
  const pSecond = shares[s.second];
  return (
    <LabSurface label="Surprise in bits" className="ps-card">
      <SurfaceHeading kicker="Logarithms" title="Surprise in bits" />
      <Prompt text={PROMPTS[0].text}>Bars above: each word’s chance. Bars below: how surprised you are if it happens.</Prompt>
      <div className="ps-mirror">
        <Columns
          label="Chance of each next word. Drag a bar, or focus it and press the arrow keys; its surprise bar underneath follows."
          names={WORDS}
          series={[{ id: "truth", name: "chance", shares }]}
          editable={{ series: 0, name: (name) => `Bar for ${name}`, onChange: change }}
          selected={s.word}
          onSelect={(index) => set({ word: index })}
          track={132}
        />
        <SurpriseBars names={WORDS} values={surprises} selected={s.word} />
      </div>
      <RangeControl
        label={`Chance of ${word}`}
        min={MIN_BAR}
        max={MAX_BAR}
        step={1}
        value={s.bars[s.word]}
        format={(value) => `${value}%`}
        onChange={(value) => change(s.word, value)}
      />
      <div className="ps-controls">
        <p className="ps-hint">
          <span aria-hidden="true">↕</span> Drag a bar above the line, or press a word’s name and use the slider. These are the same bars as
          on the first card.
        </p>
        <button type="button" className="ps-button" disabled={sameList(s.bars, SKY_BARS)} onClick={() => set({ bars: formatList(SKY_BARS) })}>
          Reset the bars
        </button>
      </div>
      <FormulaWithValues
        label={`Surprise of ${word}: how many halvings from 100% down to ${s.bars[s.word]}%`}
        expression={`−log₂ ${two(shares[s.word])}`}
        result={`= ${bits(surprises[s.word])} bits`}
        detail="Read −log₂ as “minus log base two”. It counts halvings: 50% is one halving (1 bit), 25% is two (2 bits), 12.5% is three (3 bits). Every time a chance halves, its surprise goes up by exactly one bit."
      />
      <h4 className="ps-subhead">Two words in a row</h4>
      <div className="ps-controls">
        <SegmentedControl label="First word" value={String(s.first)} options={WORD_OPTIONS} onChange={(value) => set({ first: Number(value) })} />
        <SegmentedControl label="Second word" value={String(s.second)} options={WORD_OPTIONS} onChange={(value) => set({ second: Number(value) })} />
      </div>
      <PairPicture first={WORDS[s.first]} second={WORDS[s.second]} pFirst={pFirst} pSecond={pSecond} />
      <FormulaWithValues
        label="Multiply the chances, add the surprises"
        expression={`${two(pFirst)} × ${two(pSecond)} = ${(pFirst * pSecond).toFixed(4)}   ·   ${bits(surprise(pFirst))} + ${bits(surprise(pSecond))}`}
        result={`= ${bits(surprise(pFirst) + surprise(pSecond))} bits`}
        detail="Two separate stories, each drawn from the bars above. The logarithm is what turns “multiply the chances” into “add the surprises”, and sums are far easier to keep track of than products of small numbers."
      />
    </LabSurface>
  );
}

const shapeName = (value: number) =>
  value === 0 ? "one sure thing" : value === 50 ? "the sky bars" : value === 100 ? "all equal" : value < 50 ? `${value} · towards the sky bars` : `${value} · towards all equal`;

function EntropyCard({ s, set }: CardProps) {
  const shares = shapeShares(s.shape);
  const surprises = shares.map(surprise);
  const average = entropy(shares);
  const choices = fairChoices(average);
  const terms = shares
    .map((share, index) => ({ share, index }))
    .filter((term) => term.share > 0)
    .map((term) => `${two(term.share)}×${bits(surprises[term.index])}`);
  return (
    <LabSurface label="Average surprise" className="ps-card">
      <SurfaceHeading kicker="Entropy" title="Average surprise" />
      <PredictCheck
        prompt="Which set of bars gives the highest average surprise?"
        options={["One sure thing", "The lopsided sky bars", "All five equal"]}
        answer={2}
        picked={s.predictShape}
        onPick={(index) => set({ predictShape: index })}
        explanation="The sky bars do have the tallest surprises, but those belong to rare words, so they are narrow and add little. With five equal bars every outcome is a real surprise. Drag the slider to each end and watch the dashed average line."
      />
      <RangeControl
        label="Shape of the bars"
        min={0}
        max={100}
        step={5}
        value={s.shape}
        format={shapeName}
        onChange={(value) => set({ shape: value })}
      />
      <div className="ps-range-ends" aria-hidden="true">
        <span>← one sure thing</span>
        <span>the sky bars</span>
        <span>all equal →</span>
      </div>
      <Columns
        label={`The bars at this shape: ${WORDS.map((word, index) => `${word} ${pct(shares[index], 1)}`).join(", ")}.`}
        names={WORDS}
        series={[{ id: "truth", name: "chance", shares, labels: shares.map((share) => pct(share, Number.isInteger(+(share * 100).toFixed(4)) ? 0 : 1)) }]}
        ghost={s.shape === 50 ? undefined : SKY}
        track={104}
      />
      <Skyline
        label={`Each word is a block as wide as its chance and as tall as its surprise. ${WORDS.map((word, index) => (shares[index] > 0 ? `${word}: ${pct(shares[index], 1)} wide, ${bits(surprises[index])} bits tall` : `${word}: never happens`)).join("; ")}. The average height is ${bits(average)} bits.`}
        names={WORDS}
        widths={shares}
        heights={surprises}
        average={average}
        averageName="average surprise"
      />
      <div className="ps-fair-row">
        <div className="metric-row">
          <Metric label="Average surprise (entropy)" value={`${bits(average)} bits`} />
          <Metric label="As uncertain as a fair choice among" value={`${choices.toFixed(2)} words`} />
        </div>
        <FairBoxes count={choices} total={WORDS.length} />
      </div>
      <FormulaWithValues
        label="Chance × surprise for each word, added up"
        expression={terms.join(" + ")}
        result={`= ${bits(average)} bits`}
        detail={`This is the area of the blocks. Two to the power ${bits(average)} is ${choices.toFixed(2)}: these bars leave you as unsure as a fair pick among that many words. Later labs call that number perplexity.`}
      />
    </LabSurface>
  );
}

const WRONG_BARS: readonly number[] = [1, 1, 1, 1, 96];
/** The bill bar's fixed scale in bits: the largest bill the 1% floor allows is about 6.6. */
const BILL_SCALE = 7;
const BILL_PRESETS = [
  { value: "equal", label: "knows nothing: all equal", bars: EQUAL_BARS },
  { value: "wrong", label: "confidently wrong: 96% green", bars: WRONG_BARS },
  { value: "truth", label: "matches what happens", bars: SKY_BARS },
] as const;

function BillCard({ s, set }: CardProps) {
  const q = toShares(s.guess);
  const best = entropy(SKY);
  const bill = crossEntropy(SKY, q);
  const wasted = klDivergence(SKY, q);
  const preset = BILL_PRESETS.find((entry) => sameList(entry.bars, s.guess))?.value ?? "";
  const change = (index: number, value: number) => set({ guess: formatList(setBar(s.guess, index, value)) });
  return (
    <LabSurface label="Your guess against what happens" className="ps-card">
      <SurfaceHeading kicker="Cross-entropy and KL divergence" title="Your guess against what happens" />
      <PredictCheck
        prompt="The truth’s own average surprise is 1.87 bits. Can a model’s bill, with any set of bars, come in lower than that?"
        options={["Yes, by a lucky guess", "Yes, by being more confident", "No, never"]}
        answer={2}
        picked={s.predictBill}
        onPick={(index) => set({ predictBill: index })}
        explanation="Being more confident makes the words you favour cheaper, but the others cost more, and the truth still sends you those others just as often. Drag the model’s bars onto the truth: the wasted bits shrink to zero and stop there."
      />
      <Prompt text={PROMPTS[0].text}>Words arrive with the truth’s chances, p. The model pays surprise by its own bars, q.</Prompt>
      <SegmentedControl
        label="Set the model to"
        value={preset}
        options={BILL_PRESETS.map((entry) => ({ value: entry.value, label: entry.label }))}
        onChange={(value) => {
          const entry = BILL_PRESETS.find((item) => item.value === value);
          if (entry) set({ guess: formatList(entry.bars) });
        }}
      />
      <Legend
        items={[
          { id: "truth", text: "p: what really happens (fixed)" },
          { id: "guess", text: "q: the model’s guess (drag these)" },
        ]}
      />
      <Columns
        label={`Truth beside the model’s guess. ${WORDS.map((word, index) => `${word}: happens ${pct(SKY[index])}, model says ${pct(q[index])}`).join("; ")}. Drag a hatched bar, or focus it and press the arrow keys.`}
        names={WORDS}
        series={[
          { id: "truth", name: "what really happens, p", shares: SKY },
          { id: "guess", name: "the model’s guess, q", shares: q },
        ]}
        editable={{ series: 1, name: (word) => `Model’s bar for ${word}`, onChange: change }}
        footers={q.map((share) => `costs ${bits(surprise(share))} bits`)}
      />
      <SliderRow name={(word) => `Model: ${word}`} values={s.guess} onChange={change} />
      <h4 className="ps-subhead">The bill: how often each word happens × what the model pays for it</h4>
      <Skyline
        label={`Each block is as wide as how often the word really happens and as tall as the surprise the model pays for it. ${WORDS.map((word, index) => `${word}: ${pct(SKY[index])} wide, ${bits(surprise(q[index]))} bits tall`).join("; ")}. The bill is ${bits(bill)} bits; the best possible is ${bits(best)}; ${bits(wasted)} bits are wasted.`}
        names={WORDS}
        widths={SKY}
        heights={q.map(surprise)}
        average={bill}
        averageName="the bill"
        ghost={{ heights: SKY.map(surprise), average: best, name: "best possible" }}
        tone="guess"
      />
      <Legend
        items={[
          { id: "block", text: "blocks: surprise by the model’s bars, q" },
          { id: "ghost", text: "dashed outline: surprise by the truth’s own bars, p" },
        ]}
      />
      <div
        className="ps-bill"
        role="img"
        aria-label={`The bill of ${bits(bill)} bits is the best possible ${bits(best)} bits plus ${bits(wasted)} bits wasted by believing q.`}
      >
        <div className="ps-bill__bar">
          <span className="ps-bill__best" style={{ width: `${(best / BILL_SCALE) * 100}%` }} />
          <span className="ps-bill__waste" style={{ width: `${(wasted / BILL_SCALE) * 100}%` }} />
        </div>
        <p className="ps-bill__key">
          <span>
            <i className="ps-bill__best" />
            best possible {bits(best)} bits
          </span>
          <span className="is-waste">
            <i className="ps-bill__waste" />+ {bits(wasted)} bits wasted by believing q
          </span>
          <span>= the bill, {bits(bill)} bits</span>
        </p>
      </div>
      <div className="metric-row">
        <Metric label="The bill (cross-entropy)" value={`${bits(bill)} bits`} tone="forward" />
        <Metric label="Best possible (entropy)" value={`${bits(best)} bits`} />
        <Metric label="Wasted (KL divergence)" value={`${bits(wasted)} bits`} tone="loss" />
      </div>
      <FormulaWithValues
        label="The bill: p × (surprise under q) for each word, added up"
        expression={WORDS.map((_, index) => `${two(SKY[index])}×${bits(surprise(q[index]))}`).join(" + ")}
        result={`= ${bits(bill)} bits`}
        tone="forward"
        detail="Training a language model means changing q to make this bill smaller. It is the loss the later labs call cross-entropy. It can fall to the entropy of the truth and no further."
      />
    </LabSurface>
  );
}

function SoftmaxCard({ s, set }: CardProps) {
  const result = softmax(s.scores, s.temperature);
  const plain = softmax(s.scores, 1);
  const largest = Math.max(...result.powers);
  const canAdd = s.scores.every((score) => score + 1 <= SCORE_LIMIT);
  const canTake = s.scores.every((score) => score - 1 >= -SCORE_LIMIT);
  const shift = (amount: number) => set({ scores: formatList(s.scores.map((score) => Math.round((score + amount) * 10) / 10)) });
  const gap = (s.scores[0] - s.scores[1]) / s.temperature;
  const pair = softmax([s.scores[0], s.scores[1]], s.temperature);
  const power = (value: number) => (value >= 1000 ? value.toExponential(1) : value >= 100 ? value.toFixed(0) : value.toFixed(2));
  const changed = !sameList(s.scores, SKY_SCORES) || s.temperature !== 1;
  return (
    <LabSurface label="From scores to bars" className="ps-card">
      <SurfaceHeading kicker="Softmax, temperature and the sigmoid" title="From scores to bars" />
      <p className="ps-lead">
        A model does not produce chances directly. It produces one raw <b>score</b> per word, of any size and sign (these are
        called logits). Three moves turn the scores into bars that add to one.
      </p>
      <div className="ps-aligned ps-aligned--flow">
        <div className="ps-flow">
          <div className="ps-flow__row ps-flow__row--scores">
            <span className="ps-flow__name">
              <b>Scores{s.temperature !== 1 ? ` ÷ ${s.temperature.toFixed(1)}` : ""}</b>
              the line is 0
            </span>
            {WORDS.map((word, index) => {
              const value = result.scaled[index];
              const size = Math.min(38, (Math.abs(value) / SCORE_LIMIT) * 38);
              return (
                <span key={word} className="ps-flow__cell ps-flow__cell--score" role="img" aria-label={`${word}: score ${s.scores[index].toFixed(1)}${s.temperature !== 1 ? `, divided by temperature ${value.toFixed(2)}` : ""}`}>
                  <i className={value < 0 ? "is-negative" : ""} style={value < 0 ? { top: "50%", height: `${size}%` } : { bottom: "50%", height: `${size}%` }} />
                  <b style={value < 0 ? { top: `calc(50% + ${size}% + 1px)` } : { bottom: `calc(50% + ${size}% + 1px)` }}>{value.toFixed(s.temperature !== 1 ? 2 : 1)}</b>
                </span>
              );
            })}
          </div>
          <p className="ps-flow__step">
            <span aria-hidden="true">↓</span> <b>1.</b> Raise e to each score. e is a fixed number, about 2.718. The result is always positive, and a
            higher score gives a much bigger one.
          </p>
          <div className="ps-flow__row ps-flow__row--powers">
            <span className="ps-flow__name">
              <b>e to the score</b>
              drawn against the tallest
            </span>
            {WORDS.map((word, index) => (
              <span key={word} className="ps-flow__cell ps-flow__cell--power" role="img" aria-label={`${word}: e to the score is ${power(result.powers[index])}`}>
                <i style={{ height: `${(result.powers[index] / largest) * 100}%` }}>
                  <b>{power(result.powers[index])}</b>
                </i>
              </span>
            ))}
          </div>
          <p className="ps-flow__step">
            <span aria-hidden="true">↓</span> <b>2.</b> Add them up: {result.powers.map(power).join(" + ")} = <b>{power(result.total)}</b>.{" "}
            <b>3.</b> Divide each by that total.
          </p>
        </div>
        <Columns
          label={`The bars softmax makes from the scores: ${WORDS.map((word, index) => `${word} ${pct(result.shares[index], 1)}`).join(", ")}.`}
          names={WORDS}
          series={[{ id: "truth", name: "chance", shares: result.shares, labels: result.shares.map((share) => pct(share, share < 0.1 ? 1 : 0)) }]}
          ghost={s.temperature !== 1 ? plain.shares : undefined}
          track={132}
        />
        <div className="ps-aligned__sliders">
          <SliderRow
            name={(word) => `Score: ${word}`}
            values={s.scores}
            min={-SCORE_LIMIT}
            max={SCORE_LIMIT}
            step={0.1}
            format={(value) => value.toFixed(1)}
            onChange={(index, value) => set({ scores: formatList(s.scores.map((score, at) => (at === index ? Math.round(value * 10) / 10 : score))) })}
          />
        </div>
      </div>
      {s.temperature !== 1 && <p className="ps-hint">Dashed outlines: the bars at temperature 1, for comparison.</p>}
      <div className="ps-controls">
        <button type="button" className="ps-button" disabled={!canAdd} onClick={() => shift(1)}>
          Add 1 to every score
        </button>
        <button type="button" className="ps-button" disabled={!canTake} onClick={() => shift(-1)}>
          Take 1 from every score
        </button>
        <button type="button" className="ps-button" disabled={!changed} onClick={() => set({ scores: formatList(SKY_SCORES), temperature: 1 })}>
          Reset scores
        </button>
      </div>
      <RangeControl
        label="Temperature (every score is divided by this first)"
        min={TEMPERATURE_MIN}
        max={TEMPERATURE_MAX}
        step={0.1}
        value={s.temperature}
        format={(value) => (value === 1 ? "1.0 · scores as they are" : value < 1 ? `${value.toFixed(1)} · sharper` : `${value.toFixed(1)} · flatter`)}
        onChange={(value) => set({ temperature: Math.round(value * 10) / 10 })}
      />
      <FormulaWithValues
        label={`Softmax for ${WORDS[0]}`}
        expression={`e^${result.scaled[0].toFixed(2)} ÷ total = ${power(result.powers[0])} ÷ ${power(result.total)}`}
        result={`= ${result.shares[0].toFixed(3)}`}
        detail="Adding the same amount to every score multiplies every “e to the score” by the same factor, so the shares do not move. Only the gaps between scores matter. Temperature stretches or shrinks those gaps: it reshapes the bars and adds no knowledge."
      />
      <h4 className="ps-subhead">With only two words, softmax is the sigmoid</h4>
      <p className="ps-hint">
        Keep only {WORDS[0]} and {WORDS[1]}. {WORDS[0]}’s share of the two depends on one number: the gap between their scores.
      </p>
      <SigmoidCurve gap={gap} first={WORDS[0]} second={WORDS[1]} />
      <div className="metric-row">
        <Metric
          label={`Score gap, ${WORDS[0]} − ${WORDS[1]}${s.temperature !== 1 ? ", ÷ temperature" : ""}`}
          value={`${gap >= 0 ? "+" : ""}${gap.toFixed(2)}`}
        />
        <Metric label="Softmax on the two scores" value={pct(pair.shares[0], 1)} tone="forward" />
        <Metric label="Sigmoid of the gap" value={pct(sigmoid(gap), 1)} tone="forward" />
      </div>
    </LabSurface>
  );
}

/** The goal meter is empty at this many wasted bits or more, and full at zero. */
const GOAL_SCALE = 0.6;

function ChallengeCard({ s, set, narrate }: CardProps) {
  const round = CHALLENGE_ROUNDS[s.round];
  const counts = challengeTally(s.round);
  const q = toShares(s.answer);
  const score = challengeScore(s.round, s.answer);
  const scored = s.scored === 1;
  const done = scored && score.gap <= CHALLENGE_GOAL;
  const revealed = s.revealed === 1;
  const truth = toShares(round.truth);
  const change = (index: number, value: number) => set({ answer: formatList(setBar(s.answer, index, value)), scored: 0 });
  const startOver = (next: number) => set({ round: next, answer: formatList(EQUAL_BARS), scored: 0, tries: 0, revealed: 0 });
  return (
    <LabSurface label="Challenge: learn it from samples" className="ps-card ps-card--challenge">
      <SurfaceHeading kicker="Challenge" title="Learn the bars from 40 samples" />
      <SegmentedControl
        label="Round"
        value={String(s.round)}
        options={CHALLENGE_ROUNDS.map((entry, index) => ({ value: String(index), label: `${index + 1} · ${entry.text} ___` }))}
        onChange={(value) => startOver(Number(value))}
      />
      <Prompt text={round.text}>
        The true bars are hidden. You get {CHALLENGE_DRAWS} draws from them. Set your bars so the bill comes within{" "}
        {CHALLENGE_GOAL.toFixed(2)} bits of the best possible.
      </Prompt>
      <Legend
        items={[
          { id: "tally", text: `the tally: each word’s share of the ${CHALLENGE_DRAWS} draws (count on top)` },
          { id: "guess", text: "q: your bars (drag these)" },
          ...(revealed ? [{ id: "ghost", text: "dashed outline: the true bars" }] : []),
        ]}
      />
      <Columns
        label={`Tally of ${CHALLENGE_DRAWS} draws beside your bars. ${WORDS.map((word, index) => `${word}: drawn ${counts[index]} times, your bar ${s.answer[index]}%`).join("; ")}. Drag a hatched bar, or focus it and press the arrow keys.`}
        names={WORDS}
        series={[
          { id: "tally", name: `share of the ${CHALLENGE_DRAWS} draws`, shares: counts.map((count) => count / CHALLENGE_DRAWS), labels: counts.map(String) },
          { id: "guess", name: "your bars, q", shares: q },
        ]}
        ghost={revealed ? truth : undefined}
        editable={{ series: 1, name: (word) => `Your bar for ${word}`, onChange: change }}
        footers={counts.map((count) => `${count} of ${CHALLENGE_DRAWS} draws`)}
      />
      <SliderRow name={(word) => `Guess: ${word}`} values={s.answer} onChange={change} />
      <div className="ps-controls">
        <button
          type="button"
          className="primary-action"
          onClick={() => {
            set({ scored: 1, tries: Math.min(999, s.tries + 1) });
            narrate(
              score.gap <= CHALLENGE_GOAL
                ? `Scored: ${score.gap.toFixed(3)} bits wasted. Goal met.`
                : `Scored: ${score.gap.toFixed(3)} bits wasted. The goal is ${CHALLENGE_GOAL.toFixed(2)} or less.`,
            );
          }}
        >
          Score it
        </button>
        <button type="button" className="ps-button" onClick={() => startOver(s.round)}>
          Start this round again
        </button>
        <button type="button" className="ps-button" disabled={s.tries === 0 || revealed} onClick={() => set({ revealed: 1 })}>
          Reveal the true bars
        </button>
      </div>
      <GoalMeter
        label={scored ? `Bits wasted, after ${s.tries} ${s.tries === 1 ? "scoring" : "scorings"}` : "Bits wasted"}
        value={scored ? 1 - Math.min(1, score.gap / GOAL_SCALE) : 0}
        target={1 - CHALLENGE_GOAL / GOAL_SCALE}
        display={scored ? `${score.gap.toFixed(3)} bits wasted · goal ${CHALLENGE_GOAL.toFixed(2)} or less` : "press Score it"}
        done={done}
        doneText={`${score.gap.toFixed(3)} bits wasted: within ${CHALLENGE_GOAL.toFixed(2)}`}
        hint={
          !scored
            ? "Move your bars, then press Score it. You can score as often as you like."
            : s.tries < 2
              ? "The tally is your only evidence about the hidden bars. Which words came up often, and which hardly at all?"
              : `What share of the ${CHALLENGE_DRAWS} draws was each word? ${Math.max(...counts)} of ${CHALLENGE_DRAWS} is ${pct(Math.max(...counts) / CHALLENGE_DRAWS, 1)}. Make each of your bars as tall as its tally bar.`
        }
      />
      {scored && (
        <div className="metric-row">
          <Metric label="Your bill (cross-entropy)" value={`${bits(score.bill, 3)} bits`} tone="forward" />
          <Metric label="Best possible (entropy)" value={`${bits(score.best, 3)} bits`} />
          <Metric label="Wasted (KL divergence)" value={`${bits(score.gap, 3)} bits`} tone="loss" />
        </div>
      )}
      {done && (
        <p className="ps-win">
          <b>You just trained a model.</b> Counting what follows a prompt and dividing by the total is how the smallest
          language model in this guide, the bigram table in <i>Next-token prediction</i>, is built. Your bill is still a
          little above the best possible, because {CHALLENGE_DRAWS} draws are a noisy picture of the truth.
        </p>
      )}
    </LabSurface>
  );
}

const CARDS: ReadonlyArray<ReadonlyArray<(props: CardProps) => ReactNode>> = [
  [BarsCard],
  [DrawCard],
  [BellCard],
  [ContextCard, BayesCard],
  [SurpriseCard, EntropyCard],
  [BillCard],
  [SoftmaxCard],
  [ChallengeCard],
];

export default function Explore({ state, setState, currentStep, narrate, goToStep }: ModuleContext) {
  const s = readState(state);
  const step = Math.min(STEPS.length - 1, Math.max(0, Math.floor(Number.isFinite(currentStep) ? currentStep : 0)));
  const props: CardProps = { s, set: setState, narrate };
  return (
    <div className="ps-shell">
      <div className="ps-lab" data-ps-step={step}>
        {CARDS[step].map((Card) => (
          <Card key={Card.name} {...props} />
        ))}
        <StepPager steps={STEPS} current={step} goToStep={goToStep} />
      </div>
    </div>
  );
}
