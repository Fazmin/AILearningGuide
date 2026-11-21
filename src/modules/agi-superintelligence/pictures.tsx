import { useId } from "react";
import { useMeasuredWidth } from "@app/module-sdk";
import {
  BEYOND,
  CEILING,
  CLAIM_KINDS,
  CLAIMS,
  DOMAINS,
  DOMAIN_MAX,
  DOUBLING_STOPS,
  HORIZON,
  HORIZON_START_YEAR,
  KIND_NAMES,
  LEVELS,
  MATCH,
  PROFILE_NAMES,
  START_ABILITY,
  SURVEY,
  SYSTEMS,
  ASI_CELL,
  cellOf,
  horizonAfter,
  type Breakdown,
  type ClaimKind,
  type ProfileModel,
  type Takeoff,
} from "./model";
import { linear, logRaw, logScale, span, workSpan } from "./kit";

/* ------------------------------------------------------------------ */
/* Card 1: the levels grid                                             */
/* ------------------------------------------------------------------ */

/**
 * The Levels of AGI table drawn as a ladder: superhuman at the top, no AI at the bottom, narrow on the
 * left and general on the right. Each cell is a button; the reader's placed systems sit inside it, and
 * once they ask, the table's own examples too.
 */
export function LevelsGrid({
  selected,
  placement,
  reveal,
  onCell,
}: {
  selected: number;
  placement: readonly number[];
  reveal: boolean;
  onCell: (cell: number) => void;
}) {
  const rows = [...LEVELS].reverse();
  return (
    <div className="asi-grid" role="group" aria-label="The Levels of AGI table. Rows are performance levels, from superhuman at the top to no AI at the bottom; columns are narrow and general.">
      <div className="asi-grid__corner" aria-hidden="true" />
      <div className="asi-grid__head">Narrow<small>one task or a few</small></div>
      <div className="asi-grid__head">General<small>a wide range of tasks</small></div>
      {rows.map((row) => (
        <div className="asi-grid__row" key={row.level}>
          <div className="asi-grid__level">
            <b>{row.level}</b>
            <span>{row.name}</span>
          </div>
          {(["narrow", "general"] as const).map((column) => {
            const cell = cellOf(row.level, column);
            const here = SYSTEMS.filter((_, index) => placement[index] === cell);
            const answers = SYSTEMS.filter((system) => system.cell === cell);
            const empty = column === "general" && row.level >= 2;
            return (
              <button
                type="button"
                key={column}
                className={`asi-cell${selected === cell ? " is-selected" : ""}${cell === ASI_CELL ? " is-asi" : ""}${empty ? " is-empty" : ""}`}
                aria-pressed={selected === cell}
                aria-label={`Level ${row.level}, ${row.name}, ${column}${cell === ASI_CELL ? ": artificial superintelligence" : ""}. ${
                  here.length ? `You placed: ${here.map((system) => system.name).join(", ")}.` : "Nothing placed."
                }`}
                onClick={() => onCell(cell)}
              >
                {cell === ASI_CELL && <span className="asi-cell__tag">ASI</span>}
                {column === "general" && row.level === 2 && <span className="asi-cell__tag asi-cell__tag--soft">Competent AGI</span>}
                <span className="asi-cell__chips">
                  {here.map((system) => {
                    const right = system.cell === cell;
                    return (
                      <i key={system.id} className={reveal ? (right ? "is-right" : "is-wrong") : ""}>
                        {system.name}
                        {reveal && <span className="sr-only">{right ? " (right)" : " (not where the table puts it)"}</span>}
                        {reveal && <span aria-hidden="true">{right ? " ✓" : " ✗"}</span>}
                      </i>
                    );
                  })}
                </span>
                {reveal && answers.length > 0 && (
                  <small className="asi-cell__answer">Table: {answers.map((system) => system.name).join(", ")}</small>
                )}
                {reveal && empty && <small className="asi-cell__answer">Table: not yet achieved</small>}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card 2: a jagged profile                                             */
/* ------------------------------------------------------------------ */

export function ProfileChart({ scores, other, model, otherModel }: { scores: readonly number[]; other: readonly number[]; model: ProfileModel; otherModel: ProfileModel }) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(360);
  const width = Math.max(280, Math.min(560, measured));
  const height = Math.round(width * 0.86);
  const cx = width / 2;
  const cy = height / 2 + 4;
  const adult = Math.min(width, height) * 0.3;
  const beyond = adult * 1.3;
  const angle = (index: number) => (index / DOMAINS.length) * Math.PI * 2 - Math.PI / 2;
  const point = (index: number, value: number, radius = adult) => {
    const r = (Math.max(0, value) / DOMAIN_MAX) * radius;
    return [cx + r * Math.cos(angle(index)), cy + r * Math.sin(angle(index))] as const;
  };
  const shape = (values: readonly number[]) =>
    values.map((value, index) => point(index, value).map((n) => n.toFixed(1)).join(",")).join(" ");
  const ring = (radius: number) =>
    DOMAINS.map((_, index) => [cx + radius * Math.cos(angle(index)), cy + radius * Math.sin(angle(index))].map((n) => n.toFixed(1)).join(",")).join(" ");
  return (
    <div className="asi-frame" ref={ref}>
      <svg
        className="asi-plot"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${PROFILE_NAMES[model]}'s scores in ten cognitive domains, each out of 10, where 10 is a well-educated adult: ${DOMAINS.map(
          (domain, index) => `${domain.name} ${scores[index]}`,
        ).join(", ")}. ${PROFILE_NAMES[otherModel]} is drawn dashed for comparison. The scale stops at the adult level, so it cannot show anything beyond a person.`}
      >
        <polygon className="asi-ring asi-ring--beyond" points={ring(beyond)} />
        {[0.25, 0.5, 0.75].map((share) => (
          <polygon key={share} className="asi-ring" points={ring(adult * share)} />
        ))}
        <polygon className="asi-ring asi-ring--adult" points={ring(adult)} />
        {DOMAINS.map((_, index) => {
          const [x, y] = point(index, DOMAIN_MAX, beyond);
          return <line key={index} className="asi-spoke" x1={cx} y1={cy} x2={x.toFixed(1)} y2={y.toFixed(1)} />;
        })}
        <polygon className="asi-shape asi-shape--other" points={shape(other)} />
        <polygon className="asi-shape" points={shape(scores)} />
        {scores.map((value, index) => {
          const [x, y] = point(index, value);
          return <circle key={index} className={`asi-vertex${value === 0 ? " is-zero" : ""}`} cx={x.toFixed(1)} cy={y.toFixed(1)} r={value === 0 ? 4.5 : 3.5} />;
        })}
        {DOMAINS.map((domain, index) => {
          const [x, y] = point(index, DOMAIN_MAX, beyond + 14);
          const cos = Math.cos(angle(index));
          const anchor = Math.abs(cos) < 0.2 ? "middle" : cos > 0 ? "start" : "end";
          return (
            <text key={domain.id} className="asi-label" x={x.toFixed(1)} y={(y + 4).toFixed(1)} textAnchor={anchor}>
              {domain.short} <tspan className="asi-label__value">{scores[index]}</tspan>
            </text>
          );
        })}
        <text className="asi-note" x={cx} y={(cy - adult - 4).toFixed(1)} textAnchor="middle">
          adult · 10
        </text>
      </svg>
      <ul className="asi-legend">
        <li>
          <i className="asi-key asi-key--shape" aria-hidden="true" />
          {PROFILE_NAMES[model]}
        </li>
        <li>
          <i className="asi-key asi-key--other" aria-hidden="true" />
          {PROFILE_NAMES[otherModel]}, dashed
        </li>
        <li>
          <i className="asi-key asi-key--adult" aria-hidden="true" />
          well-educated adult (10 of 10)
        </li>
        <li>
          <i className="asi-key asi-key--beyond" aria-hidden="true" />
          beyond any person: not on this scale
        </li>
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card 3: how long a project takes                                     */
/* ------------------------------------------------------------------ */

const TIME_MARKS: Array<[number, string]> = [
  [1 / (365.25 * 86400), "1 s"],
  [1 / (365.25 * 1440), "1 min"],
  [1 / (365.25 * 24), "1 h"],
  [1 / 365.25, "1 day"],
  [1, "1 yr"],
  [100, "100 yr"],
];

export function ProjectClock({ alone, now }: { alone: number; now: Breakdown }) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(560);
  const width = Math.max(280, measured);
  const height = 132;
  const left = 14;
  const right = width - 14;
  const low = 1e-9;
  const high = 1000;
  const x = logScale(low, high, left, right);
  const barY = 70;
  const parts = [
    { key: "world", value: now.world, label: "waiting on the world" },
    { key: "inorder", value: now.inOrder, label: "thinking in order" },
    { key: "parallel", value: now.parallel, label: "thinking split across copies" },
  ];
  const total = Math.max(now.total, 1e-30);
  let cursor = left;
  return (
    <div className="asi-frame" ref={ref}>
      <svg
        className="asi-plot asi-clock"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Calendar time on a scale where each step is a thousand times longer. One person alone: ${span(alone)}. With these settings: ${span(
          now.total,
        )}, of which ${parts.map((part) => `${span(part.value)} ${part.label}`).join(", ")}.`}
      >
        <line className="asi-axis" x1={left} x2={right} y1={34} y2={34} />
        {TIME_MARKS.map(([value, label]) => (
          <g key={label}>
            <line className="asi-tick-line" x1={x(value)} x2={x(value)} y1={29} y2={39} />
            <text className="asi-tick" x={x(value)} y={22} textAnchor="middle">
              {label}
            </text>
          </g>
        ))}
        <g>
          <path className="asi-pin asi-pin--alone" d={`M${x(alone)} 34 l-5 -9 h10 z`} />
          <text className="asi-tick" x={x(alone)} y={52} textAnchor={x(alone) > width - 90 ? "end" : "middle"}>
            one person
          </text>
        </g>
        <g>
          <path className="asi-pin" d={`M${x(now.total)} 34 l-6 -11 h12 z`} />
          {Math.abs(x(now.total) - x(alone)) > 70 && (
            <text className="asi-tick asi-tick--strong" x={x(now.total)} y={52} textAnchor={x(now.total) < 60 ? "start" : "middle"}>
              these settings
            </text>
          )}
        </g>
        {parts.map((part) => {
          const length = ((right - left) * part.value) / total;
          const start = cursor;
          cursor += length;
          return length > 0.2 ? <rect key={part.key} className={`asi-part asi-part--${part.key}`} x={start} y={barY} width={length} height={20} /> : null;
        })}
        <text className="asi-tick" x={left} y={barY + 36}>
          where the time goes, as shares of {span(now.total)}
        </text>
      </svg>
      <ul className="asi-legend">
        {parts.map((part) => (
          <li key={part.key}>
            <i className={`asi-key asi-key--${part.key}`} aria-hidden="true" />
            {part.label}: {span(part.value)}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card 4: the feedback loop                                            */
/* ------------------------------------------------------------------ */

const ABILITY_TICKS: Array<[number, string]> = [
  [0.1, "0.1×"],
  [1, "1×"],
  [10, "10×"],
  [100, "100×"],
  [1e4, "10,000×"],
  [1e6, "1 million×"],
];

export function TakeoffPlot({ run, ghost, year }: { run: Takeoff; ghost: Takeoff | null; year: number }) {
  const id = useId();
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(560);
  const width = Math.max(280, measured);
  const height = Math.round(Math.max(230, Math.min(320, width * 0.52)));
  const left = 84;
  const right = width - 12;
  const top = 14;
  const bottom = height - 30;
  const x = linear(0, HORIZON, left, right);
  const y = logScale(START_ABILITY, CEILING, bottom, top);
  const yRaw = logRaw(START_ABILITY, CEILING, bottom, top);
  // A curve that leaves the chart is drawn on past the top edge, where the clip trims it.
  const path = (series: Takeoff) =>
    series.years
      .map((value, index) => `${index === 0 ? "M" : "L"}${x(value).toFixed(1)} ${yRaw(series.ability[index] >= CEILING ? CEILING * 100 : series.ability[index]).toFixed(1)}`)
      .join("");
  const at = abilityAt(run, year);
  return (
    <div className="asi-frame" ref={ref}>
      <svg
        className="asi-plot"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`AI research ability against years, on a scale where each step up is ten times more. It matches people in year ${run.matchYear.toFixed(
          1,
        )}. ${
          run.beyondYear >= 0 ? `It reaches 100 times beyond people in year ${run.beyondYear.toFixed(1)}.` : `It does not reach 100 times beyond people within ${HORIZON} years.`
        } ${run.offChartYear >= 0 ? `It leaves the chart, past a million times, in year ${run.offChartYear.toFixed(1)}.` : ""}`}
      >
        <defs>
          <clipPath id={`${id}-clip`}>
            <rect x={left} y={top - 2} width={right - left} height={bottom - top + 4} />
          </clipPath>
        </defs>
        {ABILITY_TICKS.map(([value, label]) => (
          <g key={label}>
            <line className="asi-grid-line" x1={left} x2={right} y1={y(value)} y2={y(value)} />
            <text className="asi-tick" x={left - 6} y={y(value) + 3.5} textAnchor="end">
              {label}
            </text>
          </g>
        ))}
        {[0, 10, 20, 30, 40].map((tick) => (
          <text key={tick} className="asi-tick" x={x(tick)} y={bottom + 16} textAnchor="middle">
            {tick}
          </text>
        ))}
        <text className="asi-tick" x={right} y={height - 2} textAnchor="end">
          years in this toy →
        </text>
        {run.matchYear >= 0 && run.beyondYear >= 0 && (
          <rect className="asi-gap" x={x(run.matchYear)} y={y(BEYOND)} width={x(run.beyondYear) - x(run.matchYear)} height={y(MATCH) - y(BEYOND)} />
        )}
        <line className="asi-threshold" x1={left} x2={right} y1={y(MATCH)} y2={y(MATCH)} />
        <line className="asi-threshold asi-threshold--beyond" x1={left} x2={right} y1={y(BEYOND)} y2={y(BEYOND)} />
        <text className="asi-threshold-label" x={left + 6} y={y(MATCH) - 5}>
          matches people at AI research
        </text>
        <text className="asi-threshold-label" x={left + 6} y={y(BEYOND) - 5}>
          100× beyond people (this lab&apos;s stand-in for ASI)
        </text>
        <g clipPath={`url(#${id}-clip)`}>
          {ghost && <path className="asi-curve asi-curve--ghost" d={path(ghost)} />}
          <path className="asi-curve" d={path(run)} />
        </g>
        {run.offChartYear >= 0 && (
          <path className="asi-pin" d={`M${x(run.offChartYear)} ${top + 2} l-6 -9 h12 z`} transform={`rotate(180 ${x(run.offChartYear)} ${top - 2})`} />
        )}
        <line className="asi-cursor" x1={x(year)} x2={x(year)} y1={top} y2={bottom} />
        <circle className="asi-dot" cx={x(year)} cy={y(at)} r={5} />
      </svg>
      <ul className="asi-legend">
        <li>
          <i className="asi-key asi-key--curve" aria-hidden="true" />
          AI research ability (1× = today&apos;s human researchers)
        </li>
        {ghost && (
          <li>
            <i className="asi-key asi-key--ghost" aria-hidden="true" />
            your previous run
          </li>
        )}
        <li>
          <i className="asi-key asi-key--gap" aria-hidden="true" />
          the climb from 1× to 100×
        </li>
      </ul>
    </div>
  );
}

/** Research ability at a year, read from the recorded run (the last value once it has left the chart). */
export function abilityAt(run: Takeoff, year: number): number {
  if (year <= 0) return run.ability[0];
  for (let index = 1; index < run.years.length; index += 1) {
    if (run.years[index] >= year) {
      const t = (year - run.years[index - 1]) / (run.years[index] - run.years[index - 1] || 1);
      return Math.exp(Math.log(run.ability[index - 1]) * (1 - t) + Math.log(run.ability[index]) * t);
    }
  }
  return run.ability[run.ability.length - 1];
}

/* ------------------------------------------------------------------ */
/* Card 5: forecasts                                                    */
/* ------------------------------------------------------------------ */

export function SurveyTimeline({ survey }: { survey: "2022" | "2023" }) {
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(560);
  const width = Math.max(280, measured);
  const height = 150;
  const left = 14;
  const right = width - 14;
  const x = linear(2020, 2130, left, right);
  const hlmi50 = survey === "2023" ? SURVEY.hlmi.p50 : SURVEY.hlmi.p50Before;
  const rows = [
    { label: "Machines beat people at every task", y: 60, p10: survey === "2023" ? SURVEY.hlmi.p10 : null, p50: hlmi50 },
    { label: "Every occupation can be fully automated", y: 112, p10: survey === "2023" ? SURVEY.faol.p10 : null, p50: survey === "2023" ? SURVEY.faol.p50 : null },
  ];
  return (
    <div className="asi-frame" ref={ref}>
      <svg
        className="asi-plot"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={
          survey === "2023"
            ? `Survey of ${SURVEY.respondents} AI researchers, late 2023. Machines beating people at every task: 10% chance by ${SURVEY.hlmi.p10}, 50% by ${SURVEY.hlmi.p50}. Every occupation fully automatable: 10% by ${SURVEY.faol.p10}, 50% by ${SURVEY.faol.p50}.`
            : `The previous survey, in 2022, put the 50% year for machines beating people at every task at ${SURVEY.hlmi.p50Before}.`
        }
      >
        {[2020, 2040, 2060, 2080, 2100, 2120].map((tick) => (
          <g key={tick}>
            <line className="asi-grid-line" x1={x(tick)} x2={x(tick)} y1={20} y2={height - 6} />
            <text className="asi-tick" x={x(tick)} y={14} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        {rows.map((row) => (
          <g key={row.label}>
            <text className="asi-row-label" x={left} y={row.y - 16}>
              {row.label}
            </text>
            {row.p10 !== null && row.p50 !== null && (
              <line className="asi-range" x1={x(row.p10)} x2={x(row.p50)} y1={row.y} y2={row.y} />
            )}
            {row.p10 !== null && (
              <g>
                <circle className="asi-p10" cx={x(row.p10)} cy={row.y} r={5} />
                <text className="asi-tick" x={x(row.p10)} y={row.y + 18} textAnchor="middle">
                  10% · {row.p10}
                </text>
              </g>
            )}
            {row.p50 !== null ? (
              <g>
                <circle className="asi-p50" cx={x(row.p50)} cy={row.y} r={7} />
                <text className="asi-tick asi-tick--strong" x={x(row.p50)} y={row.y + 20} textAnchor={x(row.p50) > right - 40 ? "end" : "middle"}>
                  50% · {row.p50}
                </text>
              </g>
            ) : (
              <text className="asi-tick" x={x(2075)} y={row.y + 4} textAnchor="middle">
                not shown for the 2022 survey in this lab
              </text>
            )}
          </g>
        ))}
      </svg>
      <ul className="asi-legend">
        <li>
          <i className="asi-key asi-key--p10" aria-hidden="true" />
          year given a 10% chance
        </li>
        <li>
          <i className="asi-key asi-key--p50" aria-hidden="true" />
          year given a 50% chance
        </li>
      </ul>
    </div>
  );
}

const WORK_MARKS: Array<[number, string]> = [
  [1, "1 min"],
  [60, "1 hour"],
  [480, "1 day"],
  [2400, "1 week"],
  [10_020, "1 month"],
  [120_000, "1 year"],
];

/** The task-length trend: fitted to measurements from 2019 to early 2025, then carried forward. */
export function HorizonPlot({ doubling, ahead }: { doubling: number; ahead: number }) {
  const id = useId();
  const [ref, measured] = useMeasuredWidth<HTMLDivElement>(560);
  const width = Math.max(280, measured);
  const height = Math.round(Math.max(220, Math.min(300, width * 0.5)));
  const left = 58;
  const right = width - 12;
  const top = 12;
  const bottom = height - 30;
  const first = 2019;
  const last = HORIZON_START_YEAR + 10;
  const x = linear(first, last, left, right);
  const y = logScale(0.01, 1e6, bottom, top);
  const yRaw = logRaw(0.01, 1e6, bottom, top);
  const curve = (months: number, from: number, to: number) => {
    let d = "";
    for (let step = 0; step <= 60; step += 1) {
      const year = from + ((to - from) * step) / 60;
      d += `${step === 0 ? "M" : "L"}${x(year).toFixed(1)} ${Math.max(-1000, yRaw(horizonAfter(year - HORIZON_START_YEAR, months))).toFixed(1)}`;
    }
    return d;
  };
  const now = horizonAfter(ahead, doubling);
  return (
    <div className="asi-frame" ref={ref}>
      <svg
        className="asi-plot"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Length of software tasks, timed by how long they take a person, that frontier AI finishes half the time. The measured trend runs from ${first} to ${HORIZON_START_YEAR}, doubling every 7 months, to about 50 minutes. Carried forward ${ahead} years at a doubling every ${doubling} months, it reaches ${workSpan(
          now,
        )}.`}
      >
        <defs>
          <clipPath id={`${id}-clip`}>
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {WORK_MARKS.map(([minutes, label]) => (
          <g key={label}>
            <line className="asi-grid-line" x1={left} x2={right} y1={y(minutes)} y2={y(minutes)} />
            <text className="asi-tick" x={left - 6} y={y(minutes) + 3.5} textAnchor="end">
              {label}
            </text>
          </g>
        ))}
        {[2019, 2022, 2025, 2028, 2031, 2034].map((tick) => (
          <text key={tick} className="asi-tick" x={x(tick)} y={bottom + 16} textAnchor="middle">
            {tick}
          </text>
        ))}
        <rect className="asi-measured" x={x(first)} y={top} width={x(HORIZON_START_YEAR) - x(first)} height={bottom - top} />
        <text className="asi-threshold-label" x={x(first) + 6} y={top + 14}>
          measured
        </text>
        <text className="asi-threshold-label" x={x(HORIZON_START_YEAR) + 6} y={top + 14}>
          carried forward: an assumption
        </text>
        <g clipPath={`url(#${id}-clip)`}>
          {DOUBLING_STOPS.filter((months) => months !== doubling).map((months) => (
            <path key={months} className="asi-curve asi-curve--ghost" d={curve(months, HORIZON_START_YEAR, last)} />
          ))}
          <path className="asi-curve asi-curve--fit" d={curve(7, first, HORIZON_START_YEAR)} />
          <path className="asi-curve asi-curve--forward" d={curve(doubling, HORIZON_START_YEAR, last)} />
        </g>
        <line className="asi-cursor" x1={x(HORIZON_START_YEAR + ahead)} x2={x(HORIZON_START_YEAR + ahead)} y1={top} y2={bottom} />
        <circle className="asi-dot" cx={x(HORIZON_START_YEAR + ahead)} cy={y(Math.min(1e6, now))} r={5} />
      </svg>
      <ul className="asi-legend">
        <li>
          <i className="asi-key asi-key--fit" aria-hidden="true" />
          trend fitted to measurements, doubling every 7 months
        </li>
        <li>
          <i className="asi-key asi-key--forward" aria-hidden="true" />
          carried forward at your doubling time
        </li>
        <li>
          <i className="asi-key asi-key--ghost" aria-hidden="true" />
          the other doubling times
        </li>
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card 6: sort the claims                                              */
/* ------------------------------------------------------------------ */

export function ClaimList({
  tags,
  onTag,
}: {
  tags: ReadonlyArray<ClaimKind | null>;
  onTag: (index: number, kind: ClaimKind) => void;
}) {
  return (
    <ol className="asi-claims">
      {CLAIMS.map((claim, index) => {
        const tag = tags[index];
        const right = tag === claim.kind;
        return (
          <li key={index} className={tag ? (right ? "is-right" : "is-wrong") : ""}>
            <p className="asi-claims__text">{claim.text}</p>
            <div className="asi-claims__kinds" role="group" aria-label={`What kind of claim is this: ${claim.text}`}>
              {CLAIM_KINDS.map((kind) => (
                <button
                  type="button"
                  key={kind}
                  aria-pressed={tag === kind}
                  className={tag === kind ? (right ? "is-right" : "is-wrong") : tag && kind === claim.kind ? "is-answer" : ""}
                  onClick={() => onTag(index, kind)}
                >
                  {KIND_NAMES[kind]}
                </button>
              ))}
            </div>
            {tag && (
              <p className="asi-claims__why" role="status">
                <strong>{right ? "Right." : `It is a ${KIND_NAMES[claim.kind].toLowerCase()}.`}</strong> {claim.why}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}
