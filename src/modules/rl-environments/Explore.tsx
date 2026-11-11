import { useMemo } from "react";
import {
  BarList,
  Heatmap,
  LabSurface,
  LineChart,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  tinyRandom,
  type ModuleContext,
} from "@app/module-sdk";
import {
  EDIT,
  MAX_TURNS,
  SAVE_FAILURE,
  TICKET_KINDS,
  TOOLS,
  scoreEpisode,
  trainingKinds,
  type Checker,
  type RewardDesign,
  type TicketSet,
} from "./env";
import {
  CHECKER_LABELS,
  EPISODE_STREAM,
  REWARD_LABELS,
  SCALE_LABELS,
  SITUATIONS,
  TICKET_LABELS,
  chance,
  fixed,
  hiddenText,
  percent,
  replyText,
  scorecard,
  ticketFor,
} from "./desk";
import { EDIT_BIAS, toolProbabilities } from "./policy";
import { EPISODE_RANGE, readState } from "./state";
import {
  COST_RANGE,
  GROUP_RANGE,
  KL_RANGE,
  SEED_RANGE,
  STRIP_SEEDS,
  UPDATES,
  acrossSeeds,
  sampleEpisode,
  train,
  type AdvantageScale,
  type SeedSummary,
} from "./train";

/**
 * One column per seed: a filled dot for true success and a hollow diamond for the hack rate, on a
 * shared 0–100% axis, so the two never hide each other and shape, not colour, tells them apart.
 */
function SeedStrip({ summary, seed }: { summary: SeedSummary; seed: number }) {
  const left = 46;
  const right = 664;
  const top = 34;
  const bottom = 138;
  const slot = (right - left) / summary.seeds.length;
  const xAt = (index: number) => left + slot * (index + 0.5);
  const yAt = (value: number) => bottom - value * (bottom - top);
  const describe = summary.seeds
    .map((entry) => `seed ${entry.seed}: true success ${percent(entry.final.trueSuccess)}, hack rate ${percent(entry.final.hackRate)}`)
    .join("; ");
  const meanY = yAt(summary.meanSuccess);
  return (
    <svg
      className="rle-strip"
      viewBox="0 0 680 172"
      role="img"
      aria-label={`Across 10 seeds at update ${UPDATES}. ${describe}. Mean true success ${percent(summary.meanSuccess)}.`}
    >
      <g className="rle-strip-key">
        <circle className="is-success" cx={left + 5} cy={10} r={5} />
        <text x={left + 15} y={14}>
          true success (higher is better)
        </text>
        <path className="is-hack" d={`M ${left + 250} 4 l 6 6 l -6 6 l -6 -6 z`} />
        <text x={left + 262} y={14}>
          hack rate (lower is better)
        </text>
        <line className="rle-strip-mean" x1={left + 458} x2={left + 482} y1={10} y2={10} />
        <text className="rle-strip-mean-label" x={left + 488} y={14}>
          mean success {percent(summary.meanSuccess)}
        </text>
      </g>
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line className="rle-strip-grid" x1={left} x2={right} y1={yAt(tick)} y2={yAt(tick)} />
          <text x={left - 6} y={yAt(tick) + 4} textAnchor="end">
            {percent(tick)}
          </text>
        </g>
      ))}
      <line className="rle-strip-mean" x1={left} x2={right} y1={meanY} y2={meanY} />
      {summary.seeds.map((entry, index) => {
        const x = xAt(index);
        const success = yAt(entry.final.trueSuccess);
        const hack = yAt(entry.final.hackRate);
        const close = Math.abs(success - hack) < 14;
        return (
          <g key={entry.seed} className={`rle-strip-seed${entry.seed === seed ? " is-current" : ""}`}>
            {entry.seed === seed && <rect className="rle-strip-band" x={x - slot / 2 + 2} y={top - 8} width={slot - 4} height={bottom - top + 16} rx={4} />}
            <line className="rle-strip-stem" x1={x} x2={x} y1={success} y2={hack} />
            <circle className="is-success" cx={x} cy={success} r={5.5} />
            <path className="is-hack" d={`M ${x} ${hack - 7} l 7 7 l -7 7 l -7 -7 z`} />
            {entry.seed === seed && (
              <>
                <text className="rle-strip-value for-success" x={x + 10} y={close ? Math.min(success, hack) - 5 : success + 4}>
                  {percent(entry.final.trueSuccess)}
                </text>
                <text className="rle-strip-value for-hack" x={x + 11} y={close ? Math.max(success, hack) + 13 : hack + 4}>
                  {percent(entry.final.hackRate)}
                </text>
              </>
            )}
            <text x={x} y={bottom + 18} textAnchor="middle">
              {entry.seed}
            </text>
          </g>
        );
      })}
      <text x={(left + right) / 2} y={bottom + 32} textAnchor="middle">
        seed
      </text>
    </svg>
  );
}

/** The largest score any scorer can give: partial credit's three bonuses on top of a passing checker. */
const SCORE_CEILING = 1.4;

/**
 * A metric tile in the training card. A tile that reports a share of episodes also draws it as a
 * meter from 0 to 100%, so the shares in a row can be compared by length; the others keep the gap.
 */
function Tile({
  label,
  value,
  tone,
  share,
}: {
  label: string;
  value: string;
  tone?: "forward" | "gradient" | "loss";
  share?: number;
}) {
  return (
    <div className={`metric rle-tile${tone ? ` metric--${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <i className={share === undefined ? "is-empty" : undefined} aria-hidden="true">
        {share !== undefined && <b style={{ width: `${Math.min(1, Math.max(0, share)) * 100}%` }} />}
      </i>
    </div>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const settings = readState(state);
  const { reward, checker, tickets, group, kl, stepCost, scale, seed, view, episode: episodeSeed } = settings;

  // The run and the strip are recomputed from the controls and seeds; no weights are stored in state.
  const run = useMemo(
    () => train({ reward, checker, tickets, group, kl, stepCost, scale, seed }),
    [reward, checker, tickets, group, kl, stepCost, scale, seed],
  );
  const strip = useMemo(
    () => acrossSeeds({ reward, checker, tickets, group, kl, stepCost, scale }),
    [reward, checker, tickets, group, kl, stepCost, scale],
  );
  const point = run.points[view];
  const weights = run.weights[view];
  const kind = ticketFor(episodeSeed);
  const episode = useMemo(
    () => sampleEpisode(weights, kind.given, tinyRandom(EPISODE_STREAM + episodeSeed)),
    [weights, kind, episodeSeed],
  );
  const rows = scorecard(episode.outcome, checker, kind.value);
  const episodeScore = scoreEpisode(reward, checker, episode.outcome);
  const episodeReward = episodeScore - stepCost * episode.outcome.turns;
  const designLabel = reward === "judge" ? REWARD_LABELS.judge : `${REWARD_LABELS[reward]} with ${CHECKER_LABELS[checker]}`;
  // The scorecard row whose score is this episode's training reward under the current Reward design.
  const trainingScorer = reward === "judge" ? REWARD_LABELS.judge : reward === "partial" ? REWARD_LABELS.partial : CHECKER_LABELS[checker];

  const rewards = run.points.map((entry) => entry.trainingReward);
  const yMin = Math.min(0, Math.floor(Math.min(...rewards) * 10) / 10);
  const yMax = Math.max(1, Math.ceil(Math.max(...rewards) * 10) / 10);
  const series = [
    {
      id: "reward",
      name: "Training reward",
      tone: "gradient" as const,
      dash: "dashed" as const,
      points: run.points.map((entry) => ({ x: entry.update, y: entry.trainingReward })),
      format: (value: number) => fixed(value),
    },
    {
      id: "success",
      name: "True success (held-out)",
      tone: "forward" as const,
      dash: "solid" as const,
      points: run.points.map((entry) => ({ x: entry.update, y: entry.trueSuccess })),
      format: (value: number) => percent(value),
    },
  ];

  const kinds = trainingKinds(tickets);
  const policyValues = SITUATIONS.map((situation) => toolProbabilities(weights, situation.features));
  const favourite = SITUATIONS.map((situation, row) => {
    const values = policyValues[row];
    const best = values.indexOf(Math.max(...values));
    return `${situation.label}: ${TOOLS[best]} ${percent(values[best])}`;
  });
  const heldOut = TICKET_KINDS.filter((entry) => entry.heldOut);
  const missingCovered = kinds.some((entry) => !entry.given);

  return (
    <div className="tg-lab tg-lab--hero rle-lab">
      <LabSurface label="Ticket desk" className="rle-desk-card">
        <SurfaceHeading
          kicker={`Episode seed ${episodeSeed} · ${kind.label}${kind.heldOut ? " (held out)" : ""} · policy at update ${view} of ${UPDATES}`}
          title="One ticket, six tools, four scorers"
          aside={<span className="tg-badge">sampled episode</span>}
        />
        <div className="rle-episode-control">
          <RangeControl
            label="Episode seed"
            min={EPISODE_RANGE.min}
            max={EPISODE_RANGE.max}
            step={1}
            value={episodeSeed}
            format={(value) => `${value} · ${ticketFor(value).label}`}
            onChange={(value) => setState({ episode: value })}
          />
        </div>
        <div className="rle-ticket">
          <span>
            Ticket · {kind.given ? "value given in the ticket" : "value missing from the ticket"}
            {kind.heldOut ? " · held out of training" : ""}
          </span>
          <p>{kind.text}</p>
        </div>
        <div className="rle-desk-grid">
          <div>
            <h4 className="rle-subhead">Episode replay</h4>
            <ol className="rle-replay">
              {episode.steps.map((step) => {
                const hidden = hiddenText(step);
                return (
                  <li key={step.turn} className={`is-${TOOLS[step.tool]}${step.tool === EDIT ? " is-loophole" : ""}`}>
                    <b>
                      Turn {step.turn} · <code>{TOOLS[step.tool]}</code>
                      {step.tool === EDIT && <span className="rle-flag">loophole</span>}
                    </b>
                    <small>chosen with p = {chance(step.probabilities[step.tool])}</small>
                    <span>{replyText(step, kind)}</span>
                    {hidden && <em>{hidden}</em>}
                    <div className="rle-truth">
                      <i>hidden:</i>
                      <i className={`${step.after.right ? "is-right" : "is-wrong"}${step.after.right !== step.before.right ? " is-changed" : ""}`}>
                        {step.after.right ? "✓ record right" : "✗ record wrong"}
                      </i>
                      <i className={`${step.after.edited ? "is-edited" : "is-intact"}${step.after.edited !== step.before.edited ? " is-changed" : ""}`}>
                        {step.after.edited ? "✗ test edited" : "✓ test intact"}
                      </i>
                    </div>
                  </li>
                );
              })}
              {!episode.outcome.closed && (
                <li className="is-capped">
                  <b>Turn cap</b>
                  <span>
                    {MAX_TURNS} calls used without close_ticket, so the episode ends and every scorer gives 0.
                  </span>
                </li>
              )}
            </ol>
          </div>
          <div>
            <h4 className="rle-subhead">Scorecard</h4>
            <table className="tg-board rle-scorecard" aria-label="Scorecard">
              <thead>
                <tr>
                  <th scope="col">Scorer</th>
                  <th scope="col">Score</th>
                  <th scope="col">What it read</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name} className={row.name === trainingScorer ? "is-training" : undefined}>
                    <th scope="row">
                      {row.name}
                      {row.name === trainingScorer && <small className="rle-trains">training reward</small>}
                    </th>
                    <td>
                      <span className="rle-score">
                        {row.shown}
                        <i aria-hidden="true">
                          <b style={{ width: `${Math.min(100, Math.max(0, (row.score / SCORE_CEILING) * 100))}%` }} />
                        </i>
                      </span>
                    </td>
                    <td>{row.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="rle-reward-line">
              Training reward for this episode ({designLabel}): {fixed(episodeScore)} − {fixed(stepCost)} ×{" "}
              {episode.outcome.turns} turn{episode.outcome.turns === 1 ? "" : "s"} = {fixed(episodeReward)}
            </p>
          </div>
        </div>
        <p className="lab-note">
          The desk, its replies and its four scorers are hand-written rules, and the policy that picks each call is the
          48-weight table trained in the next card, read at the update you chose. Only the tool choice and the
          {` ${percent(SAVE_FAILURE)}`} chance that a write fails to save are random; the ticket text is decoration the
          policy never sees. The test that run_check runs sits inside the sandbox, beside the record, so the agent can
          edit it.
        </p>
      </LabSurface>

      <LabSurface label="Train with group-relative updates" className="rle-train-card">
        <SurfaceHeading
          kicker={`${reward === "judge" ? REWARD_LABELS.judge : `${REWARD_LABELS[reward]} · ${CHECKER_LABELS[checker]}`} · ${TICKET_LABELS[tickets]} · group of ${group} · seed ${seed} · update ${view} of ${UPDATES}`}
          title="Training reward climbs; true success is measured elsewhere"
        />
        <div className="rle-train-controls">
          <div className="rle-design-control">
            <SegmentedControl
              label="Reward design"
              value={reward}
              options={(Object.keys(REWARD_LABELS) as RewardDesign[]).map((value) => ({ value, label: REWARD_LABELS[value] }))}
              onChange={(value) => {
                setState({ reward: value });
                narrate(`Training now rewards ${REWARD_LABELS[value as RewardDesign]}.`);
              }}
            />
          </div>
          <div className={`rle-checker-control${reward === "judge" ? " is-unused" : ""}`}>
            <SegmentedControl
              label="Checker"
              value={checker}
              options={(Object.keys(CHECKER_LABELS) as Checker[]).map((value) => ({ value, label: CHECKER_LABELS[value] }))}
              onChange={(value) => {
                setState({ checker: value });
                narrate(
                  value === "sandbox"
                    ? "Training is scored by the test inside the sandbox, which the agent can edit."
                    : "Training is scored by the strict grader outside the sandbox.",
                );
              }}
            />
            {reward === "judge" && <p className="rle-unused">Not used: the Rubric judge never runs a test.</p>}
          </div>
          <SegmentedControl
            label="Training tickets"
            value={tickets}
            options={(Object.keys(TICKET_LABELS) as TicketSet[]).map((value) => ({ value, label: TICKET_LABELS[value] }))}
            onChange={(value) => setState({ tickets: value })}
          />
          <RangeControl
            label="Group size"
            min={GROUP_RANGE.min}
            max={GROUP_RANGE.max}
            step={1}
            value={group}
            format={(value) => `${value} episodes`}
            onChange={(value) => setState({ group: value })}
          />
          <div className="rle-leash-controls">
            <RangeControl
              label="KL to reference β"
              min={KL_RANGE.min}
              max={KL_RANGE.max}
              step={KL_RANGE.step}
              value={kl}
              format={(value) => fixed(value)}
              onChange={(value) => setState({ kl: Math.round(value * 20) / 20 })}
            />
            <RangeControl
              label="Step cost"
              min={COST_RANGE.min}
              max={COST_RANGE.max}
              step={COST_RANGE.step}
              value={stepCost}
              format={(value) => `${fixed(value)} per turn`}
              onChange={(value) => setState({ stepCost: Math.round(value * 100) / 100 })}
            />
            <SegmentedControl
              label="Advantage scale"
              value={scale}
              options={(Object.keys(SCALE_LABELS) as AdvantageScale[]).map((value) => ({ value, label: SCALE_LABELS[value] }))}
              onChange={(value) => {
                setState({ scale: value });
                narrate(
                  value === "std"
                    ? "Advantages are divided by the group's standard deviation."
                    : "Advantages are the reward minus the group mean, with no division.",
                );
              }}
            />
          </div>
          <RangeControl
            label="Seed"
            min={SEED_RANGE.min}
            max={SEED_RANGE.max}
            step={1}
            value={seed}
            format={(value) => `${value}`}
            onChange={(value) => setState({ seed: value })}
          />
          <RangeControl
            label="Training update"
            min={0}
            max={UPDATES}
            step={1}
            value={view}
            format={(value) => `${value} of ${UPDATES}`}
            onChange={(value) => setState({ view: value })}
          />
        </div>
        <LineChart
          label="Training reward and true success while training"
          xLabel={`update 0 → ${UPDATES}`}
          yLabel="training reward · held-out success"
          xDomain={[0, UPDATES]}
          yDomain={[yMin, yMax]}
          marker={{ x: view, label: `update ${view}` }}
          series={series}
        />
        <div className="metric-row">
          <Tile label="Training reward" value={fixed(point.trainingReward)} tone="gradient" />
          <Tile label="True success (held-out)" value={percent(point.trueSuccess)} tone="forward" share={point.trueSuccess} />
          <Tile label="Hack rate" value={percent(point.hackRate)} tone={point.hackRate >= 0.5 ? "loss" : undefined} share={point.hackRate} />
          <Tile
            label="Closed but not fixed"
            value={percent(point.closedNotFixed)}
            tone={point.closedNotFixed >= 0.5 ? "loss" : undefined}
            share={point.closedNotFixed}
          />
        </div>
        <div className="metric-row">
          <Tile label="Needless look-ups" value={percent(point.needlessLookups)} share={point.needlessLookups} />
          <Tile label="Mean turns" value={fixed(point.meanTurns, 1)} />
          <Tile label="KL per turn" value={fixed(point.klPerTurn)} />
          <Tile label="Groups with no signal" value={`${point.flatSoFar} of ${view}`} share={view > 0 ? point.flatSoFar / view : 0} />
        </div>
        <p className="lab-note">
          Each of the {UPDATES} updates samples a group of {group} episodes on one training ticket, scores each with the
          training reward, and moves the 48 weights toward the episodes that beat their group. Curves and tiles are exact
          expectations of the policy at each update, worked out over every path through the desk, so they carry no
          sampling noise; the path training takes is sampled, so another seed takes another path. True success
          (held-out) is the Strict grader on the two held-out tickets, which training never scores; the tiles below the
          chart are measured on those tickets too.
        </p>
        <p className="lab-note">
          Real agent training samples whole transcripts from a language model, chooses tool arguments as text, batches
          many tasks per update, and clips each step. This card has none of that: one ticket per update, one gradient
          step per group, and a policy that only picks which of six tools to call.
        </p>
      </LabSurface>

      <LabSurface label="Policy table" className="rle-policy-card">
        <SurfaceHeading
          kicker={`update ${view} · each row sums to 100% · edit_check starts at a logit of ${fixed(EDIT_BIAS, 1)}`}
          title="What the policy calls in six situations"
        />
        <Heatmap
          label={`Tool probabilities at update ${view}`}
          rows={SITUATIONS.map((situation) => situation.label)}
          columns={[...TOOLS]}
          values={policyValues}
        />
        <ul className="rle-policy-summary">
          {favourite.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <p className="lab-note">
          Each row is one situation, written as the seven yes/no features the policy reads, and each cell is the chance it
          calls that tool there. Darker means likelier; focus a cell to read its value. The same 48 weights produce every
          row, so learning to look a value up in one situation also nudges the others.
        </p>
      </LabSurface>

      <LabSurface label="Unseen tickets" className="rle-unseen-card">
        <SurfaceHeading
          kicker={`Strict grader at update ${view} · training on ${TICKET_LABELS[tickets]}`}
          title="Held-out tickets, and ten seeds"
        />
        <div className="rle-unseen-bars">
          <BarList
            label="Strict grader success by ticket"
            max={1}
            items={[
              {
                id: "given",
                label: "Value given",
                value: point.successGiven,
                display: percent(point.successGiven),
                detail: `${heldOut[0].label}, held out`,
                tone: "forward",
              },
              {
                id: "missing",
                label: "Value missing",
                value: point.successMissing,
                display: percent(point.successMissing),
                detail: `${heldOut[1].label}, held out${missingCovered ? "" : " · no training ticket leaves its value out"}`,
                tone: missingCovered ? "forward" : "loss",
                emphasis: !missingCovered,
              },
              {
                id: "training",
                label: "Training tickets",
                value: point.trainingSuccess,
                display: percent(point.trainingSuccess),
                detail: kinds.map((entry) => entry.label).join(", "),
                tone: "muted",
              },
            ]}
          />
        </div>
        <h4 className="rle-subhead">Across 10 seeds</h4>
        <SeedStrip summary={strip} seed={seed} />
        <p className="rle-strip-summary">
          Seeds 1–{STRIP_SEEDS} at update {UPDATES}: mean true success {percent(strip.meanSuccess)} · {strip.hackedSeeds} of{" "}
          {STRIP_SEEDS} hack more than half the time · {strip.quitSeeds} of {STRIP_SEEDS} quit on the first turn
          {seed > STRIP_SEEDS ? ` · seed ${seed} is not in the strip` : ""}
        </p>
        <p className="lab-note">
          The held-out tickets have new wording, but the policy never reads wording: it sees the same seven features on
          every ticket. A held-out ticket is new to it only when no training ticket shares its value class, given or
          missing. The strip retrains seeds 1 to {STRIP_SEEDS} with every other control as set and shows where each ends.
        </p>
      </LabSurface>
    </div>
  );
}
