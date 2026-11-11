import { useEffect, useMemo, useState } from "react";
import {
  BarList,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  sampleTinyText,
  SegmentedControl,
  SurfaceHeading,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  tinyPerplexity,
  trainTinyFactored,
  trainTinyModel,
  UNIFORM_CROSS_ENTROPY,
  type ModuleContext,
} from "@app/module-sdk";
import {
  composePipeline,
  COMPOSE_GROUP,
  corpusCounts,
  corpusKl,
  IMPORTANCE_SCORES,
  labelFrequencies,
  prune,
  rowDistribution,
  STORED_BITS,
  type ImportanceScore,
  type PruneMode,
} from "./compress";
import { MaskGrid, PruneChart, ShapeKey, SoftTargetBars, StageLadder, TailMeter, TemperatureSweep } from "./visuals";

const corpus = TINY_CORPORA.harbor;
const TEACHER_EPOCHS = 80;
const V = TINY_VOCAB_SIZE;
const CONTEXTS = [" ", "h", "t", "e", "o", "s"];
const MODES: PruneMode[] = ["unstructured", "semi", "structured"];
const SWEEP = Array.from({ length: 10 }, (_, index) => (index / 9) * 0.9);
const IMPORTANCE_IDS = IMPORTANCE_SCORES.map((entry) => entry.id) as string[];
/** Every stop of the Distillation temperature slider, for the chart that shows KL at each one. */
const TEMPERATURES = Array.from({ length: 12 }, (_, index) => 0.5 + index * 0.5);
type SweepPoint = { t: number; kl: number };
/** Finished temperature sweeps by student size and budget, so returning to a setting redraws at once. */
const sweepCache = new Map<string, SweepPoint[]>();
const withCommas = (value: number) => value.toLocaleString("en-US");

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const glyph = (character: string) => (character === " " ? "␣" : character);

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const rank = clamp(Math.round(asNumber(state, "rank", 6)), 1, 12);
  const temperature = clamp(Math.round(asNumber(state, "temperature", 1) * 2) / 2, 0.5, 6);
  const epochs = clamp(Math.round(asNumber(state, "epochs", 18)), 1, 40);
  const labels = asString(state, "labels", "soft") === "hard" ? "hard" : "soft";
  const pruneFraction = clamp(Math.round(asNumber(state, "pruneFraction", 0.3) * 20) / 20, 0, 0.9);
  const modeRaw = asString(state, "pruneMode", "unstructured");
  const pruneMode: PruneMode = (MODES as string[]).includes(modeRaw) ? (modeRaw as PruneMode) : "unstructured";
  const contextRaw = asString(state, "context", "h");
  const context = CONTEXTS.includes(contextRaw) ? contextRaw : "h";
  const importanceRaw = asString(state, "importance", "magnitude");
  const importance: ImportanceScore = IMPORTANCE_IDS.includes(importanceRaw) ? (importanceRaw as ImportanceScore) : "magnitude";
  const composeShare = clamp(Math.round(asNumber(state, "composeShare", 0.5) * 20) / 20, 0, 0.9);
  const composeBits = clamp(Math.round(asNumber(state, "composeBits", 4)), 2, 8);

  const teacher = useMemo(() => trainTinyModel({ text: corpus.text, epochs: TEACHER_EPOCHS, seed: 1 }), []);
  const teacherPerplexity = useMemo(() => tinyPerplexity(teacher.weights, corpus.text), [teacher.weights]);

  const students = useMemo(() => {
    const shared = {
      text: corpus.text,
      rank,
      epochs,
      batchSize: 16,
      learningRate: 0.5,
      seed: 3,
      teacher: teacher.weights,
    };
    const soft = trainTinyFactored({ ...shared, softLabels: true, temperature });
    const hard = trainTinyFactored({ ...shared, softLabels: false });
    return {
      soft,
      hard,
      softKl: corpusKl(teacher.weights, soft.weights, corpus.text),
      hardKl: corpusKl(teacher.weights, hard.weights, corpus.text),
    };
  }, [epochs, rank, teacher.weights, temperature]);

  // The same soft-label run at every other temperature, for the sweep chart only. It is measured a point at a time
  // after the controls settle, so dragging a slider never waits on it.
  const sweepKey = `${rank}:${epochs}`;
  const [sweep, setSweep] = useState<{ key: string; points: SweepPoint[] }>({ key: "", points: [] });
  useEffect(() => {
    const cached = sweepCache.get(sweepKey);
    if (cached) {
      setSweep({ key: sweepKey, points: cached });
      return;
    }
    const points: SweepPoint[] = [];
    let timer = window.setTimeout(function measure() {
      const t = TEMPERATURES[points.length];
      const run = trainTinyFactored({
        text: corpus.text,
        rank,
        epochs,
        batchSize: 16,
        learningRate: 0.5,
        seed: 3,
        teacher: teacher.weights,
        softLabels: true,
        temperature: t,
      });
      points.push({ t, kl: corpusKl(teacher.weights, run.weights, corpus.text) });
      setSweep({ key: sweepKey, points: [...points] });
      if (points.length < TEMPERATURES.length) timer = window.setTimeout(measure, 0);
      else sweepCache.set(sweepKey, points);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [epochs, rank, sweepKey, teacher.weights]);
  const sweepPoints = sweep.key === sweepKey ? sweep.points : [];

  const student = labels === "soft" ? students.soft : students.hard;
  const studentKl = labels === "soft" ? students.softKl : students.hardKl;

  const row = TINY_VOCAB.indexOf(context);
  const soft = useMemo(() => {
    const frequencies = labelFrequencies(corpus.text, row);
    const teacherAtT = rowDistribution(teacher.weights, row, temperature);
    const teacherAt1 = rowDistribution(teacher.weights, row, 1);
    const unseenMass = (values: number[]) =>
      values.reduce((sum, value, k) => (frequencies[k] === 0 && TINY_VOCAB[k] !== "<" && TINY_VOCAB[k] !== ">" ? sum + value : sum), 0);
    return {
      frequencies,
      teacherAtT,
      student: rowDistribution(student.weights, row, 1),
      unseenAt1: unseenMass(teacherAt1),
      unseenAtT: unseenMass(teacherAtT),
      observed: frequencies.filter((value) => value > 0).length,
    };
  }, [row, student.weights, teacher.weights, temperature]);

  const pruned = useMemo(
    () => prune(teacher.weights, pruneMode, pruneFraction),
    [pruneFraction, pruneMode, teacher.weights],
  );
  const prunedPerplexity = tinyPerplexity(pruned.weights, corpus.text);
  const sampleOptions = { prompt: "the ", length: 68, temperature: 0.7, seed: 4 } as const;
  const teacherSample = useMemo(() => sampleTinyText(teacher.weights, sampleOptions), [teacher.weights]);
  const prunedSample = sampleTinyText(pruned.weights, sampleOptions);

  const pruneSweep = useMemo(() => {
    const measure = (mode: "unstructured" | "structured") =>
      SWEEP.map((fraction) => ({
        x: fraction,
        y: tinyPerplexity(prune(teacher.weights, mode, fraction).weights, corpus.text),
      }));
    return {
      unstructured: measure("unstructured"),
      structured: measure("structured"),
      semi: tinyPerplexity(prune(teacher.weights, "semi", 0.5).weights, corpus.text),
    };
  }, [teacher.weights]);

  // The compose card: the student selected above goes through prune and quantize, scored three ways so the rows compare.
  const counts = useMemo(() => corpusCounts(corpus.text), []);
  const composed = useMemo(
    () =>
      Object.fromEntries(
        IMPORTANCE_SCORES.map((entry) => [
          entry.id,
          composePipeline({
            teacher: teacher.weights,
            student: student.weights,
            rank,
            text: corpus.text,
            importance: entry.id,
            share: composeShare,
            bits: composeBits,
            counts,
          }),
        ]),
      ) as Record<ImportanceScore, ReturnType<typeof composePipeline>>,
    [composeBits, composeShare, counts, rank, student.weights, teacher.weights],
  );
  const pipeline = composed[importance];
  const [teacherStage, studentStage, prunedStage, finalStage] = pipeline.stages;

  const studentName = labels === "soft" ? "soft-label student" : "hard-label student";
  const softWins = students.softKl < students.hardKl;

  return (
    <div className="tg-lab dp-lab">
      <LabSurface label="Student controls" className="tg-student-panel">
        <SurfaceHeading kicker="Student" title="Size, target, temperature, budget" />
        <div className="dp-controls">
          <div className="tg-swap">
            <SegmentedControl
              label="Training target"
              value={labels}
              options={[
                { value: "soft", label: "Teacher distribution" },
                { value: "hard", label: "One-hot labels" },
              ]}
              onChange={(value) => {
                setState({ labels: value });
                narrate(value === "soft" ? "Training on the teacher's full distribution." : "Training on one-hot labels only.");
              }}
            />
          </div>
          <RangeControl
            label="Student size"
            min={1}
            max={12}
            step={1}
            value={rank}
            format={(value) => `rank ${value} · ${2 * value * V} weights`}
            onChange={(value) => setState({ rank: value })}
          />
          <RangeControl
            label="Distillation temperature"
            min={0.5}
            max={6}
            step={0.5}
            value={temperature}
            format={(value) => (value === 1 ? "T = 1 (no tempering)" : `T = ${value.toFixed(1)}`)}
            onChange={(value) => {
              setState({ temperature: value });
              narrate(`Temperature ${value.toFixed(1)}. Higher values flatten the teacher's distribution.`);
            }}
          />
          <RangeControl
            label="Student epochs"
            min={1}
            max={40}
            step={1}
            value={epochs}
            format={(value) => `${value} passes`}
            onChange={(value) => setState({ epochs: value })}
          />
        </div>
        <div className="metric-row">
          <Metric label="Student loss" value={student.finalLoss.toFixed(3)} tone="forward" />
          <Metric label="Teacher loss" value={teacher.finalLoss.toFixed(3)} />
          <Metric label="KL to teacher" value={studentKl.toFixed(4)} tone="gradient" />
        </div>
        <p className="tg-subhead">
          The soft-label student at every temperature · rank {rank}, {epochs} passes
        </p>
        <TemperatureSweep
          points={sweepPoints}
          hardKl={students.hardKl}
          temperature={temperature}
          softKl={students.softKl}
          complete={sweepPoints.length === TEMPERATURES.length}
        />
        <div className={`tg-callout ${softWins ? "is-quiet" : "is-warning"}`}>
          <strong>
            {softWins
              ? `Soft labels are closer to the teacher by ${(students.hardKl - students.softKl).toFixed(4)} nats per position.`
              : `Soft labels are currently ${(students.softKl - students.hardKl).toFixed(4)} nats per position further from the teacher than one-hot labels.`}
          </strong>
          <span>
            Each soft-label update carries the teacher's whole 30-way distribution; each one-hot update carries one
            sampled character, so the hard-label student has to average many noisy updates to reach the same place.
            Temperature changes which part of the distribution the loss weighs, not where its optimum is: a
            full-capacity student matching the teacher at any T matches it at T = 1 too.
          </span>
        </div>
      </LabSurface>

      <LabSurface label="Distillation" className="tg-distill-panel">
        <SurfaceHeading
          kicker={`Teacher ${teacher.weights.length} weights · student ${student.parameters}`}
          title="Same student size, two different targets"
          aside={<span className="tg-badge">{((student.parameters / student.fullParameters) * 100).toFixed(0)}% of the teacher</span>}
        />
        <LineChart
          label="Student training loss from soft and hard labels"
          xLabel="optimizer step"
          yLabel="loss on hard labels (nats/token)"
          yDomain={[1.6, UNIFORM_CROSS_ENTROPY + 0.1]}
          series={[
            {
              id: "soft",
              name: `soft labels, T=${temperature.toFixed(1)}`,
              tone: "forward",
              points: students.soft.history.map((point) => ({ x: point.step, y: point.loss })),
            },
            {
              id: "hard",
              name: "hard labels",
              tone: "loss",
              dash: "dashed",
              points: students.hard.history.map((point) => ({ x: point.step, y: point.loss })),
            },
            {
              id: "teacher",
              name: "teacher",
              tone: "muted",
              dash: "dotted",
              points: [
                { x: 0, y: teacher.finalLoss },
                { x: Math.max(1, students.soft.history.at(-1)?.step ?? 1), y: teacher.finalLoss },
              ],
            },
          ]}
          footnote="Both curves are minibatch hard-label cross-entropy, whichever target the student trained against. The dotted line is the teacher's loss on the whole corpus."
        />
        <p className="tg-subhead">KL(teacher ‖ student) over the corpus's own contexts · lower is closer</p>
        <BarList
          label="KL from the teacher to each student, weighted by how often each context occurs, lower is closer"
          items={[
            {
              id: "soft",
              label: "Soft labels",
              value: students.softKl,
              display: students.softKl.toFixed(4),
              tone: "forward",
              emphasis: labels === "soft",
              detail: `trained on the teacher's distribution at T = ${temperature.toFixed(1)}`,
            },
            {
              id: "hard",
              label: "Hard labels",
              value: students.hardKl,
              display: students.hardKl.toFixed(4),
              tone: "loss",
              emphasis: labels === "hard",
              detail: "trained on one-hot targets, teacher never consulted",
            },
          ]}
          max={Math.max(students.softKl, students.hardKl)}
        />
      </LabSurface>

      <LabSurface label="Soft targets" className="dp-soft-card">
        <SurfaceHeading
          kicker={`After “${glyph(context)}” · ${soft.observed} different characters follow it in the corpus`}
          title="What a soft label carries that a one-hot label does not"
        />
        <div className="dp-context-control">
          <SegmentedControl
            label="Context character"
            value={context}
            options={CONTEXTS.map((character) => ({ value: character, label: glyph(character) }))}
            onChange={(value) => setState({ context: value })}
          />
        </div>
        <SoftTargetBars
          labels={soft.frequencies}
          teacher={soft.teacherAtT}
          student={soft.student}
          temperature={temperature}
          studentName={studentName}
          context={context}
        />
        <div className="dp-legend" aria-hidden="true">
          <span className="is-label">top bar: one-hot labels, averaged</span>
          <span className="is-teacher">middle: teacher at T = {temperature.toFixed(1)}</span>
          <span className="is-student">bottom: {studentName} at T = 1</span>
        </div>
        <TailMeter atOne={soft.unseenAt1} atT={soft.unseenAtT} temperature={temperature} />
        <p className="lab-note">
          A one-hot label is a single character; averaged over the corpus it becomes the grey frequencies. This bigram
          teacher learned almost exactly those frequencies plus a thin tail on characters it never saw here, so its
          “dark knowledge” is small. Raising T lifts that tail and flattens the head.
        </p>
      </LabSurface>

      <LabSurface label="Pruning controls" className="tg-prune-controls">
        <SurfaceHeading kicker="Pruning" title="How much, and in what shape" />
        <div className="dp-controls">
          <RangeControl
            label="Pruned share"
            min={0}
            max={0.9}
            step={0.05}
            value={pruneFraction}
            format={(value) => (pruneMode === "semi" ? "fixed at 50% by 2:4" : `${Math.round(value * 100)}% removed`)}
            onChange={(value) => {
              setState({ pruneFraction: value });
              narrate(`${Math.round(value * 100)} percent of weights removed.`);
            }}
          />
          <SegmentedControl
            label="Pruning mode"
            value={pruneMode}
            options={[
              { value: "unstructured", label: "Unstructured" },
              { value: "semi", label: "2:4 semi-structured" },
              { value: "structured", label: "Structured rows" },
            ]}
            onChange={(value) => setState({ pruneMode: value })}
          />
        </div>
        <div className="tg-diff">
          <div className="tg-sample">
            <span>Teacher</span>
            <p>{teacherSample}</p>
          </div>
          <div className="tg-sample">
            <span>
              Pruned {Math.round(pruned.removedFraction * 100)}% ·{" "}
              {prunedSample === teacherSample
                ? "identical to the teacher"
                : `${[...prunedSample].filter((character, index) => character !== teacherSample[index]).length} of ${prunedSample.length} characters differ`}
            </span>
            <p>
              {[...prunedSample].map((character, index) =>
                character !== teacherSample[index] ? (
                  <mark key={index} className="dp-diff">
                    {character}
                  </mark>
                ) : (
                  character
                ),
              )}
            </p>
          </div>
        </div>
        <p className="tg-subhead">What each shape of zeros saves</p>
        <ShapeKey mode={pruneMode} />
      </LabSurface>

      <LabSurface label="Pruning" className="tg-prune-panel">
        <SurfaceHeading
          kicker={`${pruned.removed} of ${teacher.weights.length} weights zeroed`}
          title="Removing weights by magnitude, in three shapes"
          aside={<span className="tg-badge">perplexity {prunedPerplexity.toFixed(2)}</span>}
        />
        <PruneChart
          sweep={pruneSweep}
          semi={pruneSweep.semi}
          base={teacherPerplexity}
          share={pruneFraction}
          mode={pruneMode}
          current={prunedPerplexity}
        />
        <div className="dp-prune-layout">
          <figure className="dp-mask-figure">
            <MaskGrid mask={pruned.mask} source={teacher.weights} removedRows={pruned.removedRows} />
          </figure>
          <div className="dp-mask-side">
            <p className="tg-subhead">The current mask</p>
            <ul className="dp-mask-key" aria-hidden="true">
              <li className="is-kept">kept · darker means a larger weight</li>
              <li className="is-pruned">
                set to zero · {pruned.removed} of {teacher.weights.length}
              </li>
            </ul>
            <p className="lab-note">
              Rows are the context character, columns the next character. Filled cells are kept, shaded by
              magnitude; empty outlined cells are zero. Unstructured zeros scatter wherever weights are small, 2:4
              leaves exactly two of every four cells down each column, and structured pruning blanks whole rows.
            </p>
          </div>
        </div>
        <div className="metric-row">
          <Metric label="Unpruned" value={teacherPerplexity.toFixed(3)} />
          <Metric label="Pruned" value={prunedPerplexity.toFixed(3)} tone="loss" />
          <Metric label="Rows removed" value={pruneMode === "structured" ? `${pruned.removedRows.length} / ${V}` : "0"} />
        </div>
      </LabSurface>

      <LabSurface label="Compose the three steps" className="dp-compose-card">
        <SurfaceHeading
          kicker={`${studentName}, rank ${rank} → ${Math.round(composeShare * 100)}% pruned by ${IMPORTANCE_SCORES.find((entry) => entry.id === importance)?.label.toLowerCase()} → ${composeBits}-bit groups of ${COMPOSE_GROUP}`}
          title="Perplexity and stored size after each stage"
          aside={
            <span className="tg-badge">{(teacherStage.bytes / finalStage.bytes).toFixed(1)}× smaller than the teacher</span>
          }
        />
        <div className="dp-controls dp-compose-controls">
          <SegmentedControl
            label="Importance score"
            value={importance}
            options={IMPORTANCE_SCORES.map((entry) => ({ value: entry.id, label: entry.label }))}
            onChange={(value) => {
              setState({ importance: value });
              narrate(`Pruning by ${value} importance.`);
            }}
          />
          <RangeControl
            label="Share pruned from the student"
            min={0}
            max={0.9}
            step={0.05}
            value={composeShare}
            format={(value) => `${Math.round(value * 100)}% of its ${studentStage.values} values`}
            onChange={(value) => setState({ composeShare: value })}
          />
          <RangeControl
            label="Final bit width"
            min={2}
            max={8}
            step={1}
            value={composeBits}
            format={(value) => `${value} bits per kept value`}
            onChange={(value) => setState({ composeBits: value })}
          />
        </div>
        <StageLadder stages={pipeline.stages} rank={rank} share={composeShare} bits={composeBits} />
        <p className="tg-subhead">Perplexity after the prune stage at {Math.round(composeShare * 100)}% · same student, three importance scores · lower is better</p>
        <BarList
          label="Perplexity after pruning for each importance score, drawn as the rise over the unpruned student, lower is better"
          items={IMPORTANCE_SCORES.map((entry) => {
            const stage = composed[entry.id].stages[2];
            return {
              id: entry.id,
              label: entry.label,
              value: Math.max(0, stage.perplexity - studentStage.perplexity),
              display: stage.perplexity.toFixed(3),
              tone: entry.id === importance ? "forward" : "gradient",
              emphasis: entry.id === importance,
              detail:
                entry.id === "magnitude"
                  ? "0 extra passes: read straight off the weights"
                  : entry.id === "activation"
                    ? "1 pass over the corpus to count how often each input fires"
                    : `${composed[entry.id].scoringPasses} passes: one corpus loss per factor entry, each with that entry set to zero`,
            };
          })}
          max={Math.max(1e-9, ...IMPORTANCE_SCORES.map((entry) => composed[entry.id].stages[2].perplexity - studentStage.perplexity))}
        />
        <div className="metric-row">
          <Metric label="Final perplexity" value={finalStage.perplexity.toFixed(3)} tone="loss" />
          <Metric label="Final size" value={`${withCommas(finalStage.bytes)} B`} tone="forward" />
          <Metric label="Passes to score" value={withCommas(pipeline.scoringPasses)} />
          <Metric label="Values kept" value={`${pipeline.kept} of ${studentStage.values}`} />
        </div>
        <p className="lab-note">
          The student is stored as its two factors ({studentStage.values} {STORED_BITS}-bit values). Pruning removes the
          same share of each factor and adds a one-bit keep-mask per position; quantizing rounds the {pipeline.kept} kept values
          in {pipeline.blocks} blocks of {COMPOSE_GROUP}, each with two 16-bit numbers, and the pruned positions stay zero.
          Every perplexity is measured on the training corpus, which is also the text the activation-weighted score and
          the ablation are calibrated on.
        </p>
      </LabSurface>
    </div>
  );
}
