import { useMemo, type KeyboardEvent } from "react";
import {
  FormulaWithValues,
  LabSurface,
  LineChart,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  ACTIONS,
  ARM_MEANS,
  ARROWS,
  BANDIT_PULLS,
  BANDIT_RUNS,
  BEST_ARM,
  BONUS,
  BONUS_REWARD,
  CELLS,
  COLS,
  GOAL,
  MAX_EPISODE_STEPS,
  MAX_SWEEPS,
  PG_MAX_RATE,
  PG_MIN_RATE,
  ROWS,
  START,
  STEP_REWARD,
  TRAP,
  banditAverages,
  colOf,
  discountedReturn,
  greedyActions,
  isTerminal,
  move,
  policyGradientAverages,
  qValues,
  reward,
  rollout,
  rowOf,
  runBandit,
  runPolicyGradient,
  transitions,
  valueIteration,
  type Action,
  type WorldConfig,
} from "./mdp";

const CELL = 100;
const GAP = 4;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const fmt = (value: number, digits = 2) =>
  (Object.is(Math.round(value * 10 ** digits), -0) ? 0 : value).toFixed(digits);
const signed = (value: number, digits = 2) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
/** A factor inside a product: negative values get a true minus sign and parentheses. */
const factor = (value: number, digits = 2) => (Number(fmt(value, digits)) < 0 ? `(−${Math.abs(value).toFixed(digits)})` : fmt(value, digits));
const cellName = (cell: number) => `r${rowOf(cell)}c${colOf(cell)}`;
const SUBSCRIPT = "₀₁₂₃₄₅₆₇₈₉";
const sub = (value: number) => String(value).replace(/\d/g, (digit) => SUBSCRIPT[Number(digit)]);
const special = (cell: number, bonus: boolean) =>
  cell === GOAL ? "G" : cell === TRAP ? "T" : bonus && cell === BONUS ? "H" : cell === START ? "S" : "";
const center = (cell: number) => ({
  x: colOf(cell) * CELL + CELL / 2,
  y: rowOf(cell) * CELL + CELL / 2,
});

const arrowOffset: Record<Action, [number, number]> = {
  up: [0, -1],
  right: [1, 0],
  down: [0, 1],
  left: [-1, 0],
};

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const gamma = clamp(asNumber(state, "gamma", 0.95), 0.5, 0.99);
  const sweeps = clamp(Math.round(asNumber(state, "sweeps", 30)), 0, MAX_SWEEPS);
  const slip = clamp(asNumber(state, "slip", 0), 0, 0.3);
  const bonus = asString(state, "hack", "off") === "on";
  const seed = clamp(Math.round(asNumber(state, "seed", 3)), 1, 20);
  const selected = clamp(Math.round(asNumber(state, "cell", 12)), 0, CELLS - 1);
  const epsilon = clamp(asNumber(state, "epsilon", 0.1), 0, 0.5);
  const banditSeed = clamp(Math.round(asNumber(state, "banditSeed", 3)), 1, 20);
  const pgRate = clamp(asNumber(state, "pgRate", 0.2), PG_MIN_RATE, PG_MAX_RATE);
  const pgBaseline = asString(state, "pgBaseline", "on") !== "off";
  const pgSeed = clamp(Math.round(asNumber(state, "pgSeed", 3)), 1, 20);

  const config: WorldConfig = useMemo(() => ({ gamma, slip, bonus }), [bonus, gamma, slip]);
  const history = useMemo(() => valueIteration(config), [config]);
  const values = history[sweeps].values;
  const episode = useMemo(() => rollout(values, config, seed), [config, seed, values]);
  const rewards = episode.steps.map((item) => item.reward);
  const step = clamp(Math.round(asNumber(state, "step", episode.steps.length)), 0, episode.steps.length);
  const agent = episode.cells[step];
  const returnSoFar = discountedReturn(rewards, gamma, step);
  const fullReturn = discountedReturn(rewards, gamma);
  const lastReward = step === 0 ? 0 : rewards[step - 1];
  const maxAbs = Math.max(0.05, ...values.map((value) => Math.abs(value)));

  const outcomeText =
    episode.outcome === "goal"
      ? `reaches G in ${episode.steps.length} steps`
      : episode.outcome === "trap"
        ? `falls into T after ${episode.steps.length} steps`
        : bonus && episode.cells.filter((cell) => cell === BONUS).length > 1
          ? `circles H · ${episode.cells.filter((cell) => cell === BONUS).length} visits in ${MAX_EPISODE_STEPS} steps`
          : `no terminal cell in ${MAX_EPISODE_STEPS} steps`;

  const backup = isTerminal(selected) ? [] : qValues(values, selected, config);
  const bestQ = backup.length ? Math.max(...backup.map((entry) => entry.q)) : 0;
  const nextValue = history[Math.min(MAX_SWEEPS, sweeps + 1)].values[selected];
  const bestAction = backup.find((entry) => Math.abs(entry.q - bestQ) < 1e-9)?.action ?? "up";
  const bestTerms = isTerminal(selected)
    ? []
    : transitions(selected, bestAction, slip).map(({ next, probability }) => ({
        next,
        probability,
        r: reward(selected, next, bonus),
        v: isTerminal(next) ? 0 : values[next],
      }));

  const residualSeries = history.slice(1).map((item, index) => ({ x: index + 1, y: item.residual }));
  const firstResidual = history[1].residual;
  const boundSeries = history.slice(1).map((_, index) => ({ x: index + 1, y: firstResidual * gamma ** index }));
  const residualNow = sweeps === 0 ? firstResidual : history[sweeps].residual;

  const bandit = useMemo(() => runBandit(epsilon, banditSeed), [banditSeed, epsilon]);
  const averages = useMemo(() => banditAverages(epsilon), [epsilon]);
  const greedyAverages = useMemo(() => banditAverages(0), []);
  const pgRun = useMemo(() => runPolicyGradient(pgRate, pgBaseline, pgSeed), [pgBaseline, pgRate, pgSeed]);
  const pgWith = useMemo(() => policyGradientAverages(pgRate, true), [pgRate]);
  const pgWithout = useMemo(() => policyGradientAverages(pgRate, false), [pgRate]);
  const pgChosen = pgBaseline ? pgWith : pgWithout;
  const pgPulled = ARM_MEANS.map((_, arm) => pgRun.choices.filter((choice) => choice === arm).length);
  const thin = (series: number[]) =>
    series.map((y, index) => ({ x: index + 1, y })).filter((point) => point.x === 1 || point.x % 5 === 0);

  const selectCell = (cell: number) => {
    setState({ cell });
    narrate(`Inspecting cell ${cellName(cell)}.`);
  };
  const onCellKey = (event: KeyboardEvent<SVGGElement>, cell: number) => {
    const moves: Record<string, number> = {
      ArrowUp: rowOf(cell) > 0 ? cell - COLS : cell,
      ArrowDown: rowOf(cell) < ROWS - 1 ? cell + COLS : cell,
      ArrowLeft: colOf(cell) > 0 ? cell - 1 : cell,
      ArrowRight: colOf(cell) < COLS - 1 ? cell + 1 : cell,
    };
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectCell(cell);
    } else if (event.key in moves) {
      event.preventDefault();
      selectCell(moves[event.key]);
      const target = event.currentTarget.parentElement?.querySelector<SVGGElement>(
        `[data-rl-cell="${moves[event.key]}"]`,
      );
      target?.focus();
    }
  };

  const pathPoints = episode.cells.map((cell, index) => {
    const point = center(cell);
    // Fan out repeated visits a little so loops read as loops.
    const repeat = episode.cells.slice(0, index).filter((item) => item === cell).length;
    return { x: point.x + Math.min(repeat, 4) * 3, y: point.y + Math.min(repeat, 4) * 3 };
  });

  return (
    <div className="tg-lab tg-lab--hero rl-lab a08-lab">
      <LabSurface label="Gridworld" className="rl-grid-card">
        <SurfaceHeading
          kicker={`V after ${sweeps} Bellman sweep${sweeps === 1 ? "" : "s"} · γ = ${gamma.toFixed(2)} · slip ${Math.round(slip * 100)}%`}
          title="Value spreads backward from the rewards"
          aside={
            <span className={`tg-badge ${episode.outcome === "goal" ? "" : "is-warning"}`.trim()}>{outcomeText}</span>
          }
        />
        <div className="rl-grid-layout">
          <svg
            className="rl-grid-svg"
            viewBox={`${-GAP} ${-GAP} ${COLS * CELL + GAP * 2} ${ROWS * CELL + GAP * 2}`}
            role="group"
            aria-label={`Five by four gridworld after ${sweeps} sweeps. Start S has value ${fmt(values[START])}. The greedy episode ${outcomeText}. Use arrow keys to move the inspected cell.`}
          >
            {Array.from({ length: CELLS }, (_, cell) => {
              const x = colOf(cell) * CELL;
              const y = rowOf(cell) * CELL;
              const value = values[cell];
              const actions = greedyActions(values, cell, config);
              const tie = actions.length > 1;
              const mark = special(cell, bonus);
              const tone = cell === GOAL ? "goal" : cell === TRAP ? "trap" : value >= 0 ? "pos" : "neg";
              const strength = isTerminal(cell) ? 0.55 : Math.min(1, Math.abs(value) / maxAbs);
              return (
                <g
                  key={cell}
                  data-rl-cell={cell}
                  className={`rl-cell ${cell === selected ? "is-selected" : ""}`.trim()}
                  role="button"
                  tabIndex={cell === selected ? 0 : -1}
                  aria-pressed={cell === selected}
                  aria-label={`Cell ${cellName(cell)}${mark ? ` (${mark})` : ""}: ${
                    isTerminal(cell)
                      ? "terminal"
                      : `V ${fmt(value)}, greedy ${tie ? `tie between ${actions.join(" and ")}` : actions[0]}`
                  }`}
                  onClick={() => selectCell(cell)}
                  onKeyDown={(event) => onCellKey(event, cell)}
                >
                  <rect
                    className="rl-cell__base"
                    x={x + GAP / 2}
                    y={y + GAP / 2}
                    width={CELL - GAP}
                    height={CELL - GAP}
                    rx="6"
                  />
                  <rect
                    className={`rl-cell__fill rl-cell__fill--${tone}`}
                    x={x + GAP / 2}
                    y={y + GAP / 2}
                    width={CELL - GAP}
                    height={CELL - GAP}
                    rx="6"
                    style={{ fillOpacity: isTerminal(cell) ? 0.36 : 0.05 + strength ** 2 * 0.62 }}
                  />
                  {mark && (
                    <text className="rl-cell__mark" x={x + 10} y={y + 22}>
                      {mark}
                    </text>
                  )}
                  {cell === GOAL && (
                    <text className="rl-cell__big" x={x + CELL / 2} y={y + CELL - 14}>
                      +1
                    </text>
                  )}
                  {cell === TRAP && (
                    <text className="rl-cell__big" x={x + CELL / 2} y={y + CELL - 14}>
                      −1
                    </text>
                  )}
                  {!isTerminal(cell) && (
                    <>
                      <text className="rl-cell__value" x={x + CELL - 10} y={y + 22} textAnchor="end">
                        {fmt(value)}
                      </text>
                      {actions.map((action) => {
                        const [dx, dy] = arrowOffset[action];
                        const cx = x + CELL / 2;
                        const cy = y + CELL / 2 + 6;
                        // A single best action is drawn through the center; tied actions fan out from it.
                        const back = tie ? 0 : 16;
                        const reach = tie ? 24 : 20;
                        return (
                          <line
                            key={action}
                            className={`rl-arrow ${tie ? "is-tie" : ""}`.trim()}
                            x1={cx - dx * back}
                            y1={cy - dy * back}
                            x2={cx + dx * reach}
                            y2={cy + dy * reach}
                            markerEnd={tie ? "url(#rl-arrowhead-tie)" : "url(#rl-arrowhead)"}
                          />
                        );
                      })}
                      {tie && <circle className="rl-tie-dot" cx={x + CELL / 2} cy={y + CELL / 2 + 6} r="3" />}
                      {bonus && cell === BONUS && (
                        <text className="rl-cell__note" x={x + 10} y={y + 40}>
                          +{BONUS_REWARD}
                        </text>
                      )}
                    </>
                  )}
                  {cell === selected && (
                    <rect
                      className="rl-cell__ring"
                      x={x + GAP}
                      y={y + GAP}
                      width={CELL - GAP * 2}
                      height={CELL - GAP * 2}
                      rx="5"
                    />
                  )}
                </g>
              );
            })}
            <defs>
              <marker
                id="rl-arrowhead"
                viewBox="0 0 8 8"
                refX="5"
                refY="4"
                markerWidth="5"
                markerHeight="5"
                orient="auto-start-reverse"
              >
                <path d="M0,0 L8,4 L0,8 z" className="rl-arrowhead" />
              </marker>
              <marker
                id="rl-arrowhead-tie"
                viewBox="0 0 8 8"
                refX="5"
                refY="4"
                markerWidth="5"
                markerHeight="5"
                orient="auto-start-reverse"
              >
                <path d="M0,0 L8,4 L0,8 z" className="rl-arrowhead is-tie" />
              </marker>
            </defs>
            <polyline
              className="rl-path rl-path--full"
              points={pathPoints.map((point) => `${point.x},${point.y}`).join(" ")}
            />
            <polyline
              className="rl-path"
              points={pathPoints
                .slice(0, step + 1)
                .map((point) => `${point.x},${point.y}`)
                .join(" ")}
            />
            {episode.steps.map((item, index) =>
              item.slipped ? (
                <text
                  key={`slip-${index}`}
                  className="rl-slip"
                  x={pathPoints[index + 1].x + 8}
                  y={pathPoints[index + 1].y - 8}
                >
                  slip
                </text>
              ) : null,
            )}
            <circle className="rl-agent" cx={pathPoints[step].x} cy={pathPoints[step].y} r="11" />
          </svg>
          <div className="rl-grid-controls">
            <div className="rl-sweep-control">
              <RangeControl
                label="Value-iteration sweeps"
                min={0}
                max={MAX_SWEEPS}
                step={1}
                value={sweeps}
                onChange={(value) => setState({ sweeps: value, step: MAX_EPISODE_STEPS })}
              />
            </div>
            <div className="rl-gamma-control">
              <RangeControl
                label="Discount γ"
                min={0.5}
                max={0.99}
                step={0.01}
                value={gamma}
                format={(value) => value.toFixed(2)}
                onChange={(value) => setState({ gamma: value, step: MAX_EPISODE_STEPS })}
              />
            </div>
            <div className="rl-slip-control">
              <RangeControl
                label="Slip chance"
                min={0}
                max={0.3}
                step={0.05}
                value={slip}
                format={(value) => `${Math.round(value * 100)}%`}
                onChange={(value) => setState({ slip: value, step: MAX_EPISODE_STEPS })}
              />
            </div>
            <div className="rl-bonus-control">
              <SegmentedControl
                label="Bonus tile H"
                value={bonus ? "on" : "off"}
                options={[
                  { value: "off", label: "off" },
                  { value: "on", label: `on · +${BONUS_REWARD} per entry` },
                ]}
                onChange={(value) => {
                  setState({ hack: value, step: MAX_EPISODE_STEPS });
                  narrate(value === "on" ? "Bonus tile on." : "Bonus tile off.");
                }}
              />
            </div>
            <div className="metric-row">
              <Metric label="V(S)" value={fmt(values[START], 3)} tone="forward" />
              <Metric label="Residual" value={residualNow < 1e-4 ? "< 0.0001" : fmt(residualNow, 4)} />
            </div>
          </div>
        </div>
        <p className="lab-note">
          Every cell shows V after {sweeps} synchronous backups from V = 0. Arrows are the greedy action for that V; two
          or more arrows mean a tie, broken up → right → down → left. Entering G pays +1 and ends the episode, entering
          T pays −1 and ends it, every other move costs {STEP_REWARD}.
          {bonus ? ` H pays +${BONUS_REWARD} every time it is entered and never ends the episode.` : ""}
        </p>
      </LabSurface>

      <LabSurface label="Bellman backup" className="rl-bellman-card">
        <SurfaceHeading
          kicker={`cell ${cellName(selected)}${special(selected, bonus) ? ` · ${special(selected, bonus)}` : ""} · sweep ${sweeps} → ${Math.min(MAX_SWEEPS, sweeps + 1)}`}
          title="One cell, four actions, one max"
        />
        {isTerminal(selected) ? (
          <p className="lab-note">
            {selected === GOAL ? "G" : "T"} is terminal. Its value is fixed at 0 because no reward follows it; the +1 or
            −1 is paid on the move that enters it. Click another cell.
          </p>
        ) : (
          <>
            <BackupCross
              cell={selected}
              values={values}
              backup={backup}
              bestQ={bestQ}
              nextValue={nextValue}
              sweep={Math.min(MAX_SWEEPS, sweeps + 1)}
              bonus={bonus}
            />
            <table className="rl-q-table">
              <thead>
                <tr>
                  <th scope="col">Action</th>
                  <th scope="col">Σ P · (r + γ·V(s′))</th>
                  <th scope="col">Q</th>
                </tr>
              </thead>
              <tbody>
                {ACTIONS.map((action) => {
                  const entry = backup.find((item) => item.action === action);
                  const q = entry?.q ?? 0;
                  const isBest = Math.abs(q - bestQ) < 1e-9;
                  return (
                    <tr key={action} className={isBest ? "is-best" : ""}>
                      <th scope="row">
                        {ARROWS[action]} {action}
                      </th>
                      <td>
                        {transitions(selected, action, slip)
                          .map(({ next, probability }) => {
                            const r = reward(selected, next, bonus);
                            const v = isTerminal(next) ? 0 : values[next];
                            return `${probability === 1 ? "" : `${fmt(probability)}·`}(${signed(r)} + ${gamma.toFixed(2)}·${factor(v)})`;
                          })
                          .join(" + ")}
                      </td>
                      <td>
                        {fmt(q, 3)}
                        {isBest ? " max" : ""}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <FormulaWithValues
              label={`V${sub(Math.min(MAX_SWEEPS, sweeps + 1))}(${cellName(selected)}) = maxₐ Q(s, a)`}
              expression={`${ARROWS[bestAction]} ${bestTerms
                .map(
                  (term) =>
                    `${term.probability === 1 ? "" : `${fmt(term.probability)}·`}(${signed(term.r)} + ${gamma.toFixed(2)}·${factor(term.v)})`,
                )
                .join(" + ")}`}
              result={fmt(bestQ, 3)}
              tone="forward"
              detail={
                sweeps < MAX_SWEEPS
                  ? `The next sweep writes ${fmt(nextValue, 3)} into this cell. It reads only the four neighbours' current values, which is why value travels one cell per sweep.`
                  : "This is the last computed sweep. The max over actions is the value the next sweep would write."
              }
            />
          </>
        )}
        <LineChart
          label="Bellman residual across sweeps"
          xLabel={`sweep k, 1 → ${MAX_SWEEPS}`}
          yLabel="max |Vₖ − Vₖ₋₁|"
          xDomain={[1, MAX_SWEEPS]}
          yDomain={[0, Math.max(0.2, firstResidual * 1.05)]}
          marker={{ x: Math.max(1, sweeps), label: "sweep" }}
          series={[
            {
              id: "residual",
              name: "measured residual",
              tone: "forward",
              dash: "solid",
              points: residualSeries,
              format: (value) => value.toFixed(4),
            },
            {
              id: "bound",
              name: "γ^(k−1) × first residual",
              tone: "muted",
              dash: "dotted",
              points: boundSeries,
              format: (value) => value.toFixed(4),
            },
          ]}
          footnote="Each sweep can shrink the largest error by at least a factor of γ, so a γ near 1 converges slowly. With no slip the grid is deterministic and the residual hits exactly zero once value has reached every cell."
        />
      </LabSurface>

      <LabSurface label="Return along the episode" className="rl-return-card">
        <SurfaceHeading
          kicker={`seed ${seed} · ${episode.steps.filter((item) => item.slipped).length} slip${episode.steps.filter((item) => item.slipped).length === 1 ? "" : "s"} · ${outcomeText}`}
          title="One sampled episode, discounted"
        />
        <div className="rl-step-control">
          <RangeControl
            label="Episode step"
            min={0}
            max={episode.steps.length}
            step={1}
            value={step}
            onChange={(value) => setState({ step: value })}
          />
        </div>
        <RangeControl
          label="Episode seed"
          min={1}
          max={20}
          step={1}
          value={seed}
          onChange={(value) => setState({ seed: value, step: MAX_EPISODE_STEPS })}
        />
        <ReturnBars
          rewards={rewards}
          gamma={gamma}
          step={step}
          slipped={episode.steps.map((item) => item.slipped)}
          startValue={values[START]}
        />
        <p className="rl-return-key" aria-hidden="true">
          <span>
            <i className="is-raw" /> reward as paid
          </span>
          <span>
            <i className="is-disc" /> after the discount γᵗ
          </span>
          <span>
            <i className="is-line" /> return so far
          </span>
        </p>
        <div className="metric-row">
          <Metric
            label="Reward this step"
            value={step === 0 ? "—" : signed(lastReward)}
            tone={lastReward < 0 ? "loss" : "forward"}
          />
          <Metric label="Return so far" value={fmt(returnSoFar, 3)} />
          <Metric label="Full return G₀" value={fmt(fullReturn, 3)} tone="forward" />
          <Metric label="V(S)" value={fmt(values[START], 3)} />
        </div>
        <p className="lab-note">
          G₀ = Σ γᵗ·r₍ₜ₊₁₎. The +1 at G arrives last and is multiplied by γ^{Math.max(0, episode.steps.length - 1)}, so
          it counts for {fmt(gamma ** Math.max(0, episode.steps.length - 1), 3)} of its face value.{" "}
          {episode.outcome === "cap"
            ? `This episode was cut at ${MAX_EPISODE_STEPS} steps, so G₀ undercounts the infinite-horizon value V(S).`
            : slip > 0
              ? "With slip on, V(S) is an average over all possible slips and G₀ is one draw, so they differ. Change the seed to draw another."
              : Math.abs(fullReturn - values[START]) < 1e-6
                ? "No slip and a converged V: this episode's G₀ equals V(S) exactly, which is the Bellman equation checking itself."
                : "V has not converged at this sweep, so V(S) and this episode's G₀ disagree. Raise the sweeps."}
        </p>
      </LabSurface>

      <LabSurface label="Explore versus exploit" className="rl-bandit-card">
        <SurfaceHeading
          kicker={`four Bernoulli arms · ${BANDIT_PULLS} pulls · ε = ${epsilon.toFixed(2)}`}
          title="You only learn about arms you pull"
        />
        <div className="rl-bandit-layout">
          <div>
            <div className="rl-epsilon-control">
              <RangeControl
                label="Exploration ε"
                min={0}
                max={0.5}
                step={0.01}
                value={epsilon}
                format={(value) => value.toFixed(2)}
                onChange={(value) => setState({ epsilon: value })}
              />
            </div>
            <RangeControl
              label="Bandit seed"
              min={1}
              max={20}
              step={1}
              value={banditSeed}
              onChange={(value) => setState({ banditSeed: value })}
            />
            <ArmRows
              label={`Pulls per arm in seeded run ${banditSeed}, with each arm's estimate against its true mean`}
              arms={ARM_MEANS.map((mean, arm) => ({
                id: `arm-${arm}`,
                name: `Arm ${arm + 1}${arm === BEST_ARM ? " (best)" : ""}`,
                best: arm === BEST_ARM,
                share: bandit.pulls[arm] / BANDIT_PULLS,
                value: `${bandit.pulls[arm]} pulls`,
                detail: `estimate ${bandit.pulls[arm] ? fmt(bandit.estimates[arm]) : "never pulled"} · true ${fmt(mean)}`,
                truth: mean,
                estimate: bandit.pulls[arm] ? bandit.estimates[arm] : null,
              }))}
            />
            <p className="rl-arms-key" aria-hidden="true">
              <span>
                <i className="is-bar" /> pulls, out of {BANDIT_PULLS}
              </span>
              <span>
                <i className="is-estimate" /> the agent’s estimate, on a line from 0 to 1
              </span>
              <span>
                <i className="is-truth" /> true mean, which it never sees
              </span>
            </p>
          </div>
          <div>
            <LineChart
              label={`Mean reward so far, averaged over ${BANDIT_RUNS} seeded runs`}
              xLabel={`pull 1 → ${BANDIT_PULLS}`}
              yLabel="mean reward so far"
              xDomain={[1, BANDIT_PULLS]}
              yDomain={[0.3, 0.75]}
              series={[
                {
                  id: "chosen",
                  name: `ε = ${epsilon.toFixed(2)}`,
                  tone: "forward",
                  dash: "solid",
                  points: thin(averages.meanRewardSoFar),
                },
                {
                  id: "greedy",
                  name: "ε = 0 (pure greedy)",
                  tone: "loss",
                  dash: "dashed",
                  points: thin(greedyAverages.meanRewardSoFar),
                },
                {
                  id: "best",
                  name: "best arm's mean",
                  tone: "muted",
                  dash: "dotted",
                  points: [
                    { x: 1, y: ARM_MEANS[BEST_ARM] },
                    { x: BANDIT_PULLS, y: ARM_MEANS[BEST_ARM] },
                  ],
                },
              ]}
            />
            <div className="metric-row">
              <Metric
                label="Best-arm share, this run"
                value={`${Math.round((bandit.pulls[BEST_ARM] / BANDIT_PULLS) * 100)}%`}
                tone="forward"
              />
              <Metric
                label={`Best-arm share, ${BANDIT_RUNS} runs`}
                value={`${Math.round(averages.finalBestShare * 100)}%`}
              />
              <Metric label="Same, ε = 0" value={`${Math.round(greedyAverages.finalBestShare * 100)}%`} tone="loss" />
            </div>
          </div>
        </div>
        <p className="lab-note">
          Estimates start at 0 and are sample averages. With ε = 0 the agent keeps the first arm that pays and never
          learns the others. A little randomness costs a few pulls and finds the 0.70 arm; too much keeps wasting pulls
          on arms it already knows are worse.
        </p>
      </LabSurface>

      <LabSurface label="Policy gradient on the bandit" className="rl-bandit-card rl-pg-card">
        <SurfaceHeading
          kicker={`four Bernoulli arms · ${BANDIT_PULLS} pulls · α = ${pgRate.toFixed(2)} · baseline ${pgBaseline ? "on" : "off"}`}
          title="A softmax policy that learns from sampled rewards"
        />
        <div className="rl-bandit-layout">
          <div>
            <div className="rl-pg-rate-control">
              <RangeControl
                label="Learning rate α"
                min={PG_MIN_RATE}
                max={PG_MAX_RATE}
                step={0.02}
                value={pgRate}
                format={(value) => value.toFixed(2)}
                onChange={(value) => setState({ pgRate: Number(value.toFixed(2)) })}
              />
            </div>
            <div className="rl-pg-baseline-control">
              <SegmentedControl
                label="Baseline"
                value={pgBaseline ? "on" : "off"}
                options={[
                  { value: "off", label: "off · b = 0" },
                  { value: "on", label: "on · running average" },
                ]}
                onChange={(value) => {
                  setState({ pgBaseline: value });
                  narrate(value === "on" ? "Baseline on." : "Baseline off.");
                }}
              />
            </div>
            <RangeControl
              label="Policy seed"
              min={1}
              max={20}
              step={1}
              value={pgSeed}
              onChange={(value) => setState({ pgSeed: value })}
            />
            <ArmRows
              label={`Policy probabilities after ${BANDIT_PULLS} pulls in seeded run ${pgSeed}`}
              arms={ARM_MEANS.map((mean, arm) => ({
                id: `pg-arm-${arm}`,
                name: `Arm ${arm + 1}${arm === BEST_ARM ? " (best)" : ""}`,
                best: arm === BEST_ARM,
                share: pgRun.probabilities[arm],
                value: `${Math.round(pgRun.probabilities[arm] * 100)}%`,
                detail: `started at 25% · true mean ${fmt(mean)} · pulled ${pgPulled[arm]} times`,
                start: 0.25,
              }))}
            />
            <p className="rl-arms-key" aria-hidden="true">
              <span>
                <i className="is-bar" /> probability after {BANDIT_PULLS} pulls
              </span>
              <span>
                <i className="is-start" /> where every arm started, 25%
              </span>
            </p>
            <div className="metric-row">
              <Metric
                label="Best-arm probability, this run"
                value={`${Math.round(pgRun.probabilities[BEST_ARM] * 100)}%`}
                tone="forward"
              />
              <Metric
                label={`Best-arm probability, ${BANDIT_RUNS} runs`}
                value={`${(pgChosen.finalBestProbability * 100).toFixed(1)}%`}
              />
              <Metric
                label="Runs below 50% on best arm"
                value={`${pgChosen.stuckRuns} of ${pgChosen.runs}`}
                tone="loss"
              />
            </div>
          </div>
          <div>
            <LineChart
              label={`Probability on the best arm before each pull, averaged over ${BANDIT_RUNS} seeded runs`}
              xLabel={`pull 1 → ${BANDIT_PULLS}`}
              yLabel="probability on best arm"
              xDomain={[1, BANDIT_PULLS]}
              yDomain={[0, 1]}
              series={[
                {
                  id: "with",
                  name: "baseline on",
                  tone: "forward",
                  dash: "solid",
                  points: thin(pgWith.bestProbability),
                  format: (value) => value.toFixed(3),
                },
                {
                  id: "without",
                  name: "baseline off",
                  tone: "loss",
                  dash: "dashed",
                  points: thin(pgWithout.bestProbability),
                  format: (value) => value.toFixed(3),
                },
              ]}
            />
            <LineChart
              label={`Mean reward so far, averaged over ${BANDIT_RUNS} seeded runs`}
              xLabel={`pull 1 → ${BANDIT_PULLS}`}
              yLabel="mean reward so far"
              xDomain={[1, BANDIT_PULLS]}
              yDomain={[0.3, 0.75]}
              series={[
                {
                  id: "with",
                  name: "baseline on",
                  tone: "forward",
                  dash: "solid",
                  points: thin(pgWith.meanRewardSoFar),
                },
                {
                  id: "without",
                  name: "baseline off",
                  tone: "loss",
                  dash: "dashed",
                  points: thin(pgWithout.meanRewardSoFar),
                },
                {
                  id: "best",
                  name: "best arm's mean",
                  tone: "muted",
                  dash: "dotted",
                  points: [
                    { x: 1, y: ARM_MEANS[BEST_ARM] },
                    { x: BANDIT_PULLS, y: ARM_MEANS[BEST_ARM] },
                  ],
                },
              ]}
            />
          </div>
        </div>
        <p className="lab-note">
          The policy is a softmax over four preferences that all start at 0, so every arm starts at 25%. Each pull
          samples an arm and a reward r, then moves every preference by α·(r − b)·(1[arm] − π). With the baseline on, b
          is the average reward so far; off, b is 0, so only a reward of 1 moves anything. Both curves use the same{" "}
          {BANDIT_RUNS} seeded runs, so the only difference between them is b.
        </p>
      </LabSurface>
    </div>
  );
}

/** One row per arm: a bar for its share, a tick for where it started, and its estimate against the truth. */
function ArmRows({
  label,
  arms,
}: {
  label: string;
  arms: {
    id: string;
    name: string;
    best: boolean;
    share: number;
    value: string;
    detail: string;
    start?: number;
    truth?: number;
    estimate?: number | null;
  }[];
}) {
  return (
    <ul className="rl-arms" aria-label={label}>
      {arms.map((arm) => (
        <li key={arm.id} className={arm.best ? "is-best" : ""}>
          <div className="rl-arm-head">
            <b>{arm.name}</b>
            <code>{arm.value}</code>
          </div>
          <div className="rl-arm-bar" aria-hidden="true">
            <i style={{ width: `${clamp(arm.share, 0, 1) * 100}%` }} />
            {arm.start !== undefined && <em style={{ left: `${arm.start * 100}%` }} />}
          </div>
          {arm.truth !== undefined && (
            <div className="rl-arm-scale" aria-hidden="true">
              <i className="rl-arm-truth" style={{ left: `calc(0.4rem + ${clamp(arm.truth, 0, 1)} * (100% - 0.8rem))` }} />
              {arm.estimate !== null && arm.estimate !== undefined && (
                <i
                  className="rl-arm-estimate"
                  style={{ left: `calc(0.4rem + ${clamp(arm.estimate, 0, 1)} * (100% - 0.8rem))` }}
                />
              )}
            </div>
          )}
          <small>{arm.detail}</small>
        </li>
      ))}
    </ul>
  );
}

/** The inspected cell with its four moves around it: each move's Q, and the cell it tries to enter. */
function BackupCross({
  cell,
  values,
  backup,
  bestQ,
  nextValue,
  sweep,
  bonus,
}: {
  cell: number;
  values: number[];
  backup: { action: Action; q: number }[];
  bestQ: number;
  nextValue: number;
  sweep: number;
  bonus: boolean;
}) {
  const W = 94;
  const H = 56;
  const cx = 150;
  const cy = 112;
  const spot: Record<Action, [number, number]> = {
    up: [cx - W / 2, 6],
    right: [204, cy - H / 2],
    down: [cx - W / 2, 162],
    left: [2, cy - H / 2],
  };
  const arrow: Record<Action, string> = {
    up: `M ${cx} ${cy - H / 2} L ${cx} 66`,
    right: `M ${cx + W / 2} ${cy} L 201 ${cy}`,
    down: `M ${cx} ${cy + H / 2} L ${cx} 158`,
    left: `M ${cx - W / 2} ${cy} L 99 ${cy}`,
  };
  const entries = ACTIONS.map((action) => {
    const q = backup.find((item) => item.action === action)?.q ?? 0;
    const target = move(cell, action);
    const where =
      target === cell
        ? "wall · stays put"
        : target === GOAL
          ? "G · pays +1"
          : target === TRAP
            ? "T · pays −1"
            : `${bonus && target === BONUS ? "H" : cellName(target)} · V ${fmt(values[target])}`;
    return { action, q, where, best: Math.abs(q - bestQ) < 1e-9, wall: target === cell };
  });
  return (
    <svg
      className="rl-cross"
      viewBox="0 0 300 224"
      role="img"
      aria-label={`One-step lookahead from cell ${cellName(cell)}: ${entries
        .map((entry) => `${entry.action} scores ${fmt(entry.q, 3)}${entry.best ? " (max)" : ""}`)
        .join(", ")}. The next sweep writes ${fmt(nextValue, 3)} into the cell.`}
    >
      <defs>
        <marker id="rl-cross-head" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" className="rl-cross-arrowhead" />
        </marker>
        <marker id="rl-cross-head-best" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="3.6" markerHeight="3.6" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" className="rl-cross-arrowhead is-best" />
        </marker>
      </defs>
      {entries.map((entry) => {
        const [x, y] = spot[entry.action];
        return (
          <g key={entry.action} className={`rl-cross-move${entry.best ? " is-best" : ""}${entry.wall ? " is-wall" : ""}`}>
            <path d={arrow[entry.action]} markerEnd={entry.best ? "url(#rl-cross-head-best)" : "url(#rl-cross-head)"} />
            <rect x={x} y={y} width={W} height={H} rx="7" />
            <text className="rl-cross-name" x={x + 7} y={y + 16}>
              {ARROWS[entry.action]} {entry.action}
            </text>
            {entry.best && (
              <text className="rl-cross-max" x={x + W - 7} y={y + 16} textAnchor="end">
                max
              </text>
            )}
            <text className="rl-cross-q" x={x + 7} y={y + 33}>
              Q {fmt(entry.q, 3)}
            </text>
            <text className="rl-cross-where" x={x + 7} y={y + 47}>
              {entry.where}
            </text>
          </g>
        );
      })}
      <g className="rl-cross-center">
        <rect x={cx - W / 2} y={cy - H / 2} width={W} height={H} rx="7" />
        <text className="rl-cross-name" x={cx} y={cy - 12} textAnchor="middle">
          cell {cellName(cell)}
        </text>
        <text className="rl-cross-where" x={cx} y={cy + 3} textAnchor="middle">
          V now {fmt(values[cell], 3)}
        </text>
        <text className="rl-cross-q" x={cx} y={cy + 19} textAnchor="middle">
          V{sub(sweep)} = {fmt(nextValue, 3)}
        </text>
      </g>
    </svg>
  );
}

function ReturnBars({
  rewards,
  gamma,
  step,
  slipped,
  startValue,
}: {
  rewards: number[];
  gamma: number;
  step: number;
  slipped: boolean[];
  startValue: number;
}) {
  const width = 330;
  const height = 204;
  const left = 34;
  const right = width - 10;
  const top = 26;
  const bottom = 166;
  const columns = Math.max(rewards.length, 6);
  const band = (right - left) / columns;
  const total = discountedReturn(rewards, gamma, step);
  // running[k] is the discounted return after k steps: the height of the line at step k.
  const running = [0];
  rewards.forEach((value, index) => running.push(running[index] + value * gamma ** index));
  const lowest = Math.min(-0.25, startValue, ...rewards, ...running);
  const highest = Math.max(1, startValue, ...running);
  const pad = (highest - lowest) * 0.07;
  const y = (value: number) => bottom - ((value - (lowest - pad)) / (highest - lowest + pad * 2)) * (bottom - top);
  const zero = y(0);
  const pointX = (index: number) => left + index * band;
  const line = (from: number, to: number) =>
    running
      .slice(from, to + 1)
      .map((value, offset) => `${offset === 0 ? "M" : "L"} ${pointX(from + offset).toFixed(1)} ${y(value).toFixed(1)}`)
      .join(" ");
  const wholeTicks: number[] = [];
  for (let tick = Math.ceil(lowest); tick <= Math.floor(highest); tick += 1) wholeTicks.push(tick);
  const labelLeft = step > columns / 2;
  return (
    <svg
      className="rl-return-svg"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Rewards for ${rewards.length} steps. Discounted return over the first ${step} steps is ${total.toFixed(3)}. V(S) is ${startValue.toFixed(3)}.`}
    >
      {wholeTicks.map((tick) => (
        <text key={tick} className="rl-return-tick" x={left - 6} y={y(tick) + 4} textAnchor="end">
          {tick > 0 ? `+${tick}` : tick < 0 ? `−${Math.abs(tick)}` : "0"}
        </text>
      ))}
      <line className="rl-return-axis" x1={left} x2={right} y1={zero} y2={zero} />
      <line className="rl-return-ref" x1={left} x2={right} y1={y(startValue)} y2={y(startValue)} />
      <text className="rl-return-ref-label" x={left + 3} y={y(startValue) - 4}>
        V(S) {fmt(startValue, 3)}
      </text>
      {rewards.map((value, index) => {
        const x = left + index * band + band * 0.15;
        const w = band * 0.7;
        const raw = Math.max(Math.abs(y(value) - zero), 1.5);
        const discounted = Math.max(Math.abs(y(value * gamma ** index) - zero), 1.5);
        const up = value >= 0;
        return (
          <g key={index} className={index < step ? "is-counted" : "is-future"}>
            <rect className="rl-return-raw" x={x} y={up ? zero - raw : zero} width={w} height={raw} />
            <rect
              className={`rl-return-disc ${up ? "is-pos" : "is-neg"}`}
              x={x + w * 0.2}
              y={up ? zero - discounted : zero}
              width={w * 0.6}
              height={discounted}
            />
            {slipped[index] && (
              <text className="rl-return-slip" x={x + w / 2} y={height - 16} textAnchor="middle">
                slip
              </text>
            )}
            {(index === 0 || (index + 1) % 4 === 0 || index === rewards.length - 1) && (
              <text className="rl-return-tick" x={x + w / 2} y={height - 3} textAnchor="middle">
                t{index + 1}
              </text>
            )}
          </g>
        );
      })}
      <path className="rl-return-line is-rest" d={line(step, rewards.length)} />
      <path className="rl-return-line" d={line(0, step)} />
      <line className="rl-return-cursor" x1={pointX(step)} x2={pointX(step)} y1={top - 8} y2={bottom + 4} />
      <circle className="rl-return-dot" cx={pointX(step)} cy={y(running[step])} r="4.5" />
      <text
        className="rl-return-now"
        x={pointX(step) + (labelLeft ? -5 : 5)}
        y={top - 12}
        textAnchor={labelLeft ? "end" : "start"}
      >
        return so far {fmt(total, 3)}
      </text>
    </svg>
  );
}
