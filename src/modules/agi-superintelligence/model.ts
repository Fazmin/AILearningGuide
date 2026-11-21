/**
 * Everything the six cards compute. Cards 1, 2 and 5 draw published tables and figures, cited in
 * module.ts; cards 3 and 4 are small models written for this lab, and every number they print comes
 * from the functions below.
 */

export const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, Number.isFinite(value) ? value : low));

/* ------------------------------------------------------------------ */
/* Card 1: the Levels of AGI table (Morris et al., ICML 2024, Table 1) */
/* ------------------------------------------------------------------ */

export interface Level {
  level: number;
  name: string;
  /** The performance bar, as the table words it. */
  bar: string;
  narrow: string;
  general: string;
}

export const LEVELS: readonly Level[] = [
  { level: 0, name: "No AI", bar: "No performance bar", narrow: "Calculator software; a compiler", general: "Human-in-the-loop computing, such as Amazon Mechanical Turk" },
  { level: 1, name: "Emerging", bar: "Equal to or somewhat better than an unskilled human", narrow: "Rule-based systems such as SHRDLU", general: "ChatGPT, Bard, Llama 2, Gemini" },
  { level: 2, name: "Competent", bar: "At least the 50th percentile of skilled adults", narrow: "Smart speakers such as Siri; the Jigsaw toxicity detector; Watson", general: "Not yet achieved" },
  { level: 3, name: "Expert", bar: "At least the 90th percentile of skilled adults", narrow: "Spelling and grammar checkers such as Grammarly; image generators such as DALL·E 2", general: "Not yet achieved" },
  { level: 4, name: "Exceptional", bar: "At least the 99th percentile of skilled adults", narrow: "Deep Blue; AlphaGo", general: "Not yet achieved" },
  { level: 5, name: "Superhuman", bar: "Outperforms 100% of humans", narrow: "AlphaFold; AlphaZero; Stockfish", general: "Artificial superintelligence (ASI): not yet achieved" },
];

export type Column = "narrow" | "general";
/** Cells are numbered level × 2 + column, narrow = 0 and general = 1, so the ASI corner is 11. */
export const cellOf = (level: number, column: Column) => level * 2 + (column === "general" ? 1 : 0);
export const levelOfCell = (cell: number) => Math.floor(cell / 2);
export const columnOfCell = (cell: number): Column => (cell % 2 === 1 ? "general" : "narrow");
export const ASI_CELL = cellOf(5, "general");
export const CELL_COUNT = 12;

export interface PlacedSystem {
  id: string;
  name: string;
  what: string;
  /** The cell Table 1 lists it in. */
  cell: number;
}

/** Systems the table names, each in the cell it lists them under. */
export const SYSTEMS: readonly PlacedSystem[] = [
  { id: "calculator", name: "Calculator", what: "does arithmetic", cell: cellOf(0, "narrow") },
  { id: "turk", name: "Mechanical Turk", what: "sends tasks to paid people", cell: cellOf(0, "general") },
  { id: "shrdlu", name: "SHRDLU", what: "a 1970 program that moves blocks on command", cell: cellOf(1, "narrow") },
  { id: "chatgpt", name: "ChatGPT (2023)", what: "a chat assistant", cell: cellOf(1, "general") },
  { id: "siri", name: "Siri", what: "a voice assistant", cell: cellOf(2, "narrow") },
  { id: "grammarly", name: "Grammarly", what: "checks spelling and grammar", cell: cellOf(3, "narrow") },
  { id: "deepblue", name: "Deep Blue", what: "beat the chess world champion in 1997", cell: cellOf(4, "narrow") },
  { id: "alphafold", name: "AlphaFold", what: "predicts protein shapes", cell: cellOf(5, "narrow") },
];

/** Placements are stored as one character per system: "a" to "l" for a cell, "-" for not placed. */
export const NOT_PLACED = "-";
export const encodeCell = (cell: number) => String.fromCharCode(97 + cell);
export const decodePlacement = (text: string): number[] =>
  SYSTEMS.map((_, index) => {
    const code = (text[index] ?? NOT_PLACED).charCodeAt(0) - 97;
    return code >= 0 && code < CELL_COUNT ? code : -1;
  });
export const encodePlacement = (cells: readonly number[]) =>
  SYSTEMS.map((_, index) => (cells[index] >= 0 && cells[index] < CELL_COUNT ? encodeCell(cells[index]) : NOT_PLACED)).join("");

/* ------------------------------------------------------------------ */
/* Card 2: a jagged profile (Hendrycks et al., 2025, Table 1)          */
/* ------------------------------------------------------------------ */

export interface Domain {
  id: string;
  name: string;
  short: string;
}

export const DOMAINS: readonly Domain[] = [
  { id: "K", name: "General knowledge", short: "Knowledge" },
  { id: "RW", name: "Reading and writing", short: "Read & write" },
  { id: "M", name: "Mathematics", short: "Maths" },
  { id: "R", name: "On-the-spot reasoning", short: "Reasoning" },
  { id: "WM", name: "Working memory", short: "Working memory" },
  { id: "MS", name: "Long-term memory storage", short: "Memory storage" },
  { id: "MR", name: "Long-term memory retrieval", short: "Memory retrieval" },
  { id: "V", name: "Visual processing", short: "Seeing" },
  { id: "A", name: "Auditory processing", short: "Hearing" },
  { id: "S", name: "Speed", short: "Speed" },
];

/** Each domain is worth 10 points; 10 means the level of a well-educated adult. */
export const DOMAIN_MAX = 10;
export type ProfileModel = "gpt4" | "gpt5";
export const PROFILE_MODELS: readonly ProfileModel[] = ["gpt4", "gpt5"];
export const PROFILE_NAMES: Record<ProfileModel, string> = { gpt4: "GPT-4", gpt5: "GPT-5" };
export const PROFILES: Record<ProfileModel, readonly number[]> = {
  gpt4: [8, 6, 4, 0, 2, 0, 4, 0, 0, 3],
  gpt5: [9, 10, 10, 7, 4, 0, 4, 4, 6, 3],
};

export type Verdict = "sum" | "weakest" | "count";
export const VERDICTS: readonly Verdict[] = ["sum", "weakest", "count"];

/** How far a profile is from AGI under each way of reading it, as a share from 0 to 1. */
export function verdict(scores: readonly number[], rule: Verdict): number {
  if (rule === "sum") return scores.reduce((total, score) => total + score, 0) / (DOMAIN_MAX * scores.length);
  if (rule === "weakest") return Math.min(...scores) / DOMAIN_MAX;
  return scores.filter((score) => score >= DOMAIN_MAX).length / scores.length;
}

/* ------------------------------------------------------------------ */
/* Card 3: faster, more, or better                                      */
/* ------------------------------------------------------------------ */

export const PROJECT_STOPS = [1, 10, 100, 1000] as const;
export const MULTIPLIER_STOPS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000] as const;
export const SHARE_STOPS = [0, 0.01, 0.1, 0.25, 0.5] as const;

export interface Workload {
  /** Expert-years of work in the project. */
  work: number;
  /** How many times faster than a person each copy thinks. */
  speed: number;
  /** How many copies work at once. */
  copies: number;
  /** Share of the thinking that must happen one step after another, so extra copies cannot split it. */
  inOrder: number;
  /** Share of the work that waits on the outside world (experiments, building, approvals), which no mind speeds up. */
  world: number;
}

export interface Breakdown {
  /** Years spent waiting on the world. */
  world: number;
  /** Years spent on thinking that must be done in order. */
  inOrder: number;
  /** Years spent on thinking that copies can split. */
  parallel: number;
  total: number;
}

/**
 * Calendar time for a project: the outside-world share takes as long as it always did, the in-order
 * thinking runs at one copy's speed, and the rest is split across every copy.
 */
export function projectTime({ work, speed, copies, inOrder, world }: Workload): Breakdown {
  const thinking = work * (1 - world);
  const parts = {
    world: work * world,
    inOrder: (thinking * inOrder) / speed,
    parallel: (thinking * (1 - inOrder)) / (speed * copies),
  };
  return { ...parts, total: parts.world + parts.inOrder + parts.parallel };
}

/* ------------------------------------------------------------------ */
/* Card 4: the feedback loop                                            */
/* ------------------------------------------------------------------ */

/**
 * A toy model of AI that does AI research. `A` is AI research ability, where 1 means it matches
 * today's community of human researchers. Research effort R is measured in the same unit: people
 * supply 1, and AI takes over the share of the work it can do at speed A once A passes 1. The share
 * only people can do (the bottleneck) still runs at human pace, as in Amdahl's law:
 *
 *   R(A) = 1 / (b + (1 − b) / max(1, A))
 *
 * Progress follows d(ln A)/dt = g · R^r / A. With no bottleneck and A above 1, that is g · A^(r − 1):
 * returns r above 1 make growth speed up by itself, r = 1 gives steady doubling, and r below 1 makes
 * it slow down. Below A = 1, R is 1 and A grows by g each year.
 */
export const START_ABILITY = 0.1;
/** Yearly gain in A before AI can help, chosen so AI matches people in year 6. */
export const PRE_GAIN = 0.15;
export const HORIZON = 40;
export const MATCH = 1;
/** The lab's stand-in for "far beyond every person". */
export const BEYOND = 100;
/** Past this the curve has left the chart. */
export const CEILING = 1e6;
export const RETURN_STOPS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3] as const;
export const BOTTLENECK_STOPS = [0, 0.001, 0.01, 0.05, 0.2] as const;

export const researchEffort = (ability: number, bottleneck: number) =>
  1 / (bottleneck + (1 - bottleneck) / Math.max(1, ability));

export interface Takeoff {
  /** Years, every `dt`. */
  years: number[];
  /** Research ability at each of those years, stopping at the ceiling. */
  ability: number[];
  /** First year A reaches 1 and 100, or -1 if it never does within the horizon. */
  matchYear: number;
  beyondYear: number;
  /** Year the curve leaves the chart, or -1. */
  offChartYear: number;
}

const DT = 0.002;
const RECORD_EVERY = 25;

/** Integrate the loop from year 0, recording every 0.05 years. */
export function takeoff(returns: number, bottleneck: number): Takeoff {
  const r = clamp(returns, 0.1, 5);
  const b = clamp(bottleneck, 0, 1);
  const slope = (logA: number) => {
    const ability = Math.exp(logA);
    return (PRE_GAIN * researchEffort(ability, b) ** r) / ability;
  };
  const years = [0];
  const ability = [START_ABILITY];
  let logA = Math.log(START_ABILITY);
  let matchYear = -1;
  let beyondYear = -1;
  let offChartYear = -1;
  const steps = Math.round(HORIZON / DT);
  for (let step = 1; step <= steps; step += 1) {
    const before = logA;
    // Classic fourth-order Runge–Kutta on ln A.
    const k1 = slope(logA);
    const k2 = slope(logA + (DT / 2) * k1);
    const k3 = slope(logA + (DT / 2) * k2);
    const k4 = slope(logA + DT * k3);
    logA += (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    const year = step * DT;
    const crossed = (level: number) => before < Math.log(level) && logA >= Math.log(level);
    const at = (level: number) => year - DT * ((logA - Math.log(level)) / (logA - before));
    if (matchYear < 0 && crossed(MATCH)) matchYear = at(MATCH);
    if (beyondYear < 0 && crossed(BEYOND)) beyondYear = at(BEYOND);
    if (!Number.isFinite(logA) || logA >= Math.log(CEILING)) {
      offChartYear = year;
      years.push(year);
      ability.push(CEILING);
      break;
    }
    if (step % RECORD_EVERY === 0) {
      years.push(year);
      ability.push(Math.exp(logA));
    }
  }
  return { years, ability, matchYear, beyondYear, offChartYear };
}

/** Years from matching people to 100× beyond them, or -1 if it does not happen within the horizon. */
export const takeoffGap = (run: Takeoff) => (run.matchYear >= 0 && run.beyondYear >= 0 ? run.beyondYear - run.matchYear : -1);

/* ------------------------------------------------------------------ */
/* Card 5: reading forecasts                                            */
/* ------------------------------------------------------------------ */

/** Grace et al. (2024): 2,778 published AI researchers, surveyed in late 2023. */
export const SURVEY = {
  respondents: 2778,
  hlmi: { p10: 2027, p50: 2047, p50Before: 2060 },
  faol: { p10: 2037, p50: 2116 },
} as const;

/** Kwa et al. (2025): a 50% time horizon of about 50 minutes for early-2025 frontier models, doubling about every 7 months. */
export const HORIZON_START_YEAR = 2025;
export const HORIZON_START_MINUTES = 50;
export const MEASURED_DOUBLING_MONTHS = 7;
export const DOUBLING_STOPS = [4, 7, 12, 24] as const;
export const AHEAD_MAX = 10;
/** A working month: about 167 hours. */
export const WORK_MONTH_MINUTES = 167 * 60;

export const horizonAfter = (yearsAhead: number, doublingMonths: number) =>
  HORIZON_START_MINUTES * 2 ** ((12 * yearsAhead) / doublingMonths);

/* ------------------------------------------------------------------ */
/* Card 6: sort the claims                                              */
/* ------------------------------------------------------------------ */

export type ClaimKind = "m" | "d" | "a" | "f";
export const CLAIM_KINDS: readonly ClaimKind[] = ["m", "d", "a", "f"];
export const KIND_NAMES: Record<ClaimKind, string> = { m: "Measured", d: "Definition", a: "Argument", f: "Forecast" };

export interface Claim {
  text: string;
  kind: ClaimKind;
  why: string;
}

export const CLAIMS: readonly Claim[] = [
  {
    text: "Stockfish beats every human chess player.",
    kind: "m",
    why: "Games against the strongest players can be played and counted. That is why the Levels of AGI table can put Stockfish in the superhuman row.",
  },
  {
    text: "ASI means the top row of the general column in the Levels of AGI table.",
    kind: "d",
    why: "This is a choice about how to use a word. It can be useful or not, but no experiment can prove it wrong.",
  },
  {
    text: "If AI does AI research and each doubling of effort brings more than one doubling of ability, progress speeds itself up.",
    kind: "a",
    why: "It follows from its assumptions, as card 4 shows. Whether the assumptions hold in the real world is a separate, open question.",
  },
  {
    text: "Machines will outperform people at every task by 2047.",
    kind: "f",
    why: "Nobody can measure 2047 yet. 2047 is the year surveyed researchers gave a 50% chance to, which is a forecast and not a fact about machines.",
  },
  {
    text: "AGI means highly autonomous systems that outperform humans at most economically valuable work.",
    kind: "d",
    why: "This is OpenAI's definition, from its charter. Other groups define AGI differently, which is why card 2 gets three verdicts from one profile.",
  },
  {
    text: "The length of software tasks frontier AI can finish has doubled about every seven months since 2019.",
    kind: "m",
    why: "METR measured it on a fixed set of tasks timed against people. The trend is measured; carrying it forward, as card 5 does, is a forecast.",
  },
  {
    text: "A mind could be extremely capable and still pursue almost any goal: intelligence does not settle what it wants.",
    kind: "a",
    why: "This is the orthogonality thesis, an argument from philosophy. It is reasoned from what goals and intelligence are, not observed in a system.",
  },
  {
    text: "Superintelligence will follow within ten years of AGI.",
    kind: "f",
    why: "It is a guess about the future. Card 4 shows the gap can be months or decades depending on assumptions nobody has measured.",
  },
];

export const NO_TAG = "-";
export const decodeTags = (text: string): Array<ClaimKind | null> =>
  CLAIMS.map((_, index) => {
    const tag = text[index];
    return (CLAIM_KINDS as readonly string[]).includes(tag ?? "") ? (tag as ClaimKind) : null;
  });
export const encodeTags = (tags: ReadonlyArray<ClaimKind | null>) => CLAIMS.map((_, index) => tags[index] ?? NO_TAG).join("");
