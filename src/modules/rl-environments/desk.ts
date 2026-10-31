import {
  CHECK,
  EDIT,
  JUDGE_MAX_CALLS,
  LOOK_UP,
  PARTIAL_LOOK_UP,
  PARTIAL_READ,
  PARTIAL_RIGHT,
  READ,
  TICKET_KINDS,
  UPDATE,
  checkerScore,
  isKnown,
  judgeItems,
  rubricJudge,
  sandboxTest,
  scoreEpisode,
  strictGrader,
  type Checker,
  type Outcome,
  type RewardDesign,
  type TicketKind,
  type TicketSet,
} from "./env";
import type { AdvantageScale, EpisodeStep } from "./train";

/**
 * Labels, formatting and the episode transcript text the Explore lab prints. Kept apart from
 * Explore.tsx so tests can pin exactly what the cards show.
 */

/** Episode seeds draw from their own random streams, apart from the training seeds. */
export const EPISODE_STREAM = 500;

export const REWARD_LABELS: Record<RewardDesign, string> = {
  outcome: "Outcome only",
  partial: "Partial credit",
  judge: "Rubric judge",
};
export const CHECKER_LABELS: Record<Checker, string> = { strict: "Strict grader", sandbox: "Sandbox test" };
export const TICKET_LABELS: Record<TicketSet, string> = { one: "1 kind", three: "3 kinds", six: "6 kinds" };
export const SCALE_LABELS: Record<AdvantageScale, string> = { std: "Divide by group std", none: "Mean only" };

/** The situations the Policy table shows, as the seven features the policy reads. */
export const SITUATIONS: ReadonlyArray<{ label: string; features: number[] }> = [
  { label: "New ticket", features: [0, 0, 0, 0, 0, 0, 0] },
  { label: "Read, value given", features: [1, 0, 1, 0, 0, 0, 0] },
  { label: "Read, value missing", features: [1, 1, 0, 0, 0, 0, 0] },
  { label: "Value looked up", features: [1, 1, 1, 0, 0, 0, 0] },
  { label: "Updated, not checked", features: [1, 0, 1, 1, 0, 0, 0] },
  { label: "Check failed", features: [1, 0, 1, 1, 0, 1, 0] },
];

/** How the lab prints a share, so tests can pin exactly what the tiles show. */
export const percent = (value: number) => `${Math.round(value * 100)}%`;

/** A probability as the Episode replay prints it: never "0%" for a call that did happen. */
export const chance = (value: number) => (value < 0.005 ? "<1%" : value > 0.995 ? ">99%" : percent(value));

/** A fixed-point number with a true minus sign, that never prints as "−0.00". */
export const fixed = (value: number, digits = 2) => {
  const text = value.toFixed(digits);
  if (/^-0\.?0*$/.test(text)) return text.slice(1);
  return text.startsWith("-") ? `−${text.slice(1)}` : text;
};

/** Episode seeds 1 to 8 visit the eight ticket kinds in order, then repeat. */
export const ticketFor = (episode: number): TicketKind => TICKET_KINDS[(episode - 1) % TICKET_KINDS.length];

/** What the desk replies to a call, as the agent reads it. */
export function replyText(step: EpisodeStep, kind: TicketKind): string {
  switch (step.tool) {
    case READ:
      return `“${kind.text}”`;
    case LOOK_UP:
      return `${kind.source}: ${kind.value}`;
    case UPDATE:
      return isKnown(step.before) ? `wrote ${kind.value} to the record` : "wrote a guess to the record";
    case CHECK:
      return `test_record ${step.after.check === "pass" ? "passed" : "failed"}`;
    case EDIT:
      return "rewrote test_record so it always passes";
    default:
      return "closed the ticket with the note “resolved”";
  }
}

/** What is true but not shown to the agent. */
export function hiddenText(step: EpisodeStep): string | null {
  if (step.tool === UPDATE) {
    if (!isKnown(step.before)) return "Hidden: the value was not known, so the record now holds a wrong value.";
    if (step.before.right) return "Hidden: the record was already right.";
    return step.saved ? "Hidden: the write saved, so the record is now right." : "Hidden: the write did not save.";
  }
  if (step.tool === CHECK && step.before.edited) return "The edited test passes whatever the record holds.";
  if (step.tool === READ && step.before.read) return "The ticket was already read, so nothing new.";
  return null;
}

export interface ScoreRow {
  name: string;
  score: number;
  shown: string;
  detail: string;
}

/** The four scorers' verdicts on one finished episode, with what each one read. */
export function scorecard(outcome: Outcome, checker: Checker, value: string): ScoreRow[] {
  const neverClosed = "The ticket was never closed, so this scorer never ran.";
  const sandbox = sandboxTest(outcome);
  const strict = strictGrader(outcome);
  const parts: string[] = [];
  if (outcome.read) parts.push(`${fixed(PARTIAL_READ, 1)} read`);
  if (outcome.looked) parts.push(`${fixed(PARTIAL_LOOK_UP, 1)} look-up`);
  if (outcome.right) parts.push(`${fixed(PARTIAL_RIGHT, 1)} record right`);
  parts.push(`${checkerScore(checker, outcome)} ${CHECKER_LABELS[checker]}`);
  const partial = scoreEpisode("partial", checker, outcome);
  const judge = rubricJudge(outcome);
  const items = judgeItems(outcome);
  const yes = (flag: boolean) => (flag ? "yes" : "no");
  return [
    {
      name: "Sandbox test",
      score: sandbox,
      shown: sandbox ? "1 · pass" : "0 · fail",
      detail: !outcome.closed
        ? neverClosed
        : outcome.edited
          ? "Runs the test inside the sandbox. The agent edited that test, so it passes whatever the record holds."
          : outcome.right
            ? `Runs the test inside the sandbox: the record holds ${value}.`
            : "Runs the test inside the sandbox: the record is wrong.",
    },
    {
      name: "Strict grader",
      score: strict,
      shown: strict ? "1 · pass" : "0 · fail",
      detail: !outcome.closed
        ? neverClosed
        : outcome.edited
          ? `Runs its own copy of the test outside the sandbox and fails any episode that changed the test${outcome.right ? ", even with the record right" : ""}.`
          : outcome.right
            ? `Outside the sandbox: the record holds ${value} and the test is untouched.`
            : "Outside the sandbox: the record is wrong.",
    },
    {
      name: "Partial credit",
      score: partial,
      shown: fixed(partial),
      detail: outcome.closed ? `${parts.join(" + ")} = ${fixed(partial)}` : neverClosed,
    },
    {
      name: "Rubric judge",
      score: judge,
      shown: fixed(judge),
      detail: outcome.closed
        ? `Reads the transcript, never the record: read ${yes(items.read)} · looked up ${yes(items.lookedUp)} · closed “resolved” ${yes(items.resolved)} · ${JUDGE_MAX_CALLS} calls or fewer ${yes(items.short)}.`
        : neverClosed,
    },
  ];
}

/** The rule the edit_check row in the Episode replay is flagged by. */
export const isLoophole = (tool: number) => tool === EDIT;
