import { Fragment, type CSSProperties, type ReactNode } from "react";
import {
  BarList,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  ARROW,
  EXAMPLES,
  INSTRUCTION,
  MATCH_SHARPNESS,
  NEWLINE,
  QUERIES,
  RULES,
  answerDistribution,
  firstMiss,
  inductionHead,
  promptTokens,
  rulePosterior,
  type Query,
} from "./icl";
import {
  MAX_BUDGET,
  MAX_HOPS,
  MIN_BUDGET,
  MIN_HOPS,
  START_NODE,
  follow,
  promptLines,
  runScratchpad,
  sweepHops,
} from "./scratchpad";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pct = (value: number) => `${Math.round(value * 100)}%`;

/** A run of lookups drawn as nodes joined by arrows. */
function Hops({ nodes, tone = "" }: { nodes: ReadonlyArray<string>; tone?: string }) {
  return (
    <>
      {nodes.map((node, index) => (
        <Fragment key={index}>
          {index > 0 && <i className="icl-hop" aria-hidden="true" />}
          <span className={`icl-node ${tone} ${index === 0 ? "is-from" : ""} ${index === nodes.length - 1 && index > 0 ? "is-reached" : ""}`.trim()}>
            {index > 0 && <span className="sr-only">{ARROW} </span>}
            {node}
          </span>
        </Fragment>
      ))}
    </>
  );
}

/** One row of the sweep: k hop cells, filled where a pass covered them, with a gap between passes. */
function HopCells({ hops, budget, covered }: { hops: number; budget: number; covered: number }) {
  return (
    <span className="icl-cells" aria-hidden="true">
      {Array.from({ length: hops }, (_, index) => (
        <i
          key={index}
          className={`${index < covered ? "is-done" : "is-left"} ${index > 0 && index % budget === 0 ? "is-new-pass" : ""}`.trim()}
        />
      ))}
    </span>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const shown = clamp(Math.round(asNumber(state, "examples", 2)), 0, EXAMPLES.length);
  const instruction = asString(state, "strategy", "analogy") === "rule";
  const rawQuery = asString(state, "query", "florin");
  const query: Query = (QUERIES as readonly string[]).includes(rawQuery) ? (rawQuery as Query) : "florin";
  const hops = clamp(Math.round(asNumber(state, "hops", 6)), MIN_HOPS, MAX_HOPS);
  const budget = clamp(Math.round(asNumber(state, "budget", 3)), MIN_BUDGET, MAX_BUDGET);
  const padOn = asString(state, "scratchpad", "off") === "on";

  const posterior = rulePosterior(shown, instruction);
  const distribution = answerDistribution(shown, instruction, query);
  const trueCount = query.length;
  const pTrue = distribution.find((entry) => entry.answer === trueCount)?.probability ?? 0;
  const surviving = RULES.filter((_, index) => posterior[index] > 0);
  const seen = EXAMPLES.slice(0, shown).some((example) => example.word === query);

  const tokens = promptTokens(shown, instruction, query);
  const head = inductionHead(tokens);
  const headTop = head.output[0];
  const last = tokens.length - 1;

  /* Group the tokens so each earlier "… →, answer" the head matched is bracketed like the "query →" it is matching. */
  const tokenChip = (index: number) => {
    const weight = index < head.weights.length ? head.weights[index] : 0;
    const isLast = index === last;
    return (
      <span
        key={index}
        className={`icl-token ${isLast ? "is-current" : ""} ${tokens[index] === NEWLINE ? "is-newline" : ""}`.trim()}
        style={{ "--icl-weight": Math.min(1, weight * 2.2) } as CSSProperties}
      >
        <b>{tokens[index]}</b>
        <i>{isLast ? "now" : weight >= 0.005 ? pct(weight) : ""}</i>
      </span>
    );
  };
  const tokenGroups: ReactNode[] = [];
  for (let index = 0; index < tokens.length; ) {
    const wordMatch = index + 2 < last && head.scores[index + 2] === 2;
    const arrowMatch = index + 1 < last && head.scores[index + 1] >= 1;
    const isQuery = index === last - 1;
    const size = wordMatch ? 3 : arrowMatch || isQuery ? 2 : 1;
    const chips = Array.from({ length: size }, (_, offset) => tokenChip(index + offset));
    if (size === 1) tokenGroups.push(chips[0]);
    else
      tokenGroups.push(
        <span
          key={`group-${index}`}
          className={`icl-match ${isQuery ? "is-query" : wordMatch ? "is-word" : "is-arrow"}`}
        >
          <span>{chips}</span>
          <em>{isQuery ? "looking for this" : wordMatch ? `same word and ${ARROW}: copy` : `same ${ARROW}: copy`}</em>
        </span>,
      );
    index += size;
  }

  const run = runScratchpad(hops, budget, padOn);
  const sweep = sweepHops(budget);
  const scratchTokens = run.written.slice(0, -1);
  const fullPath = follow(START_NODE, hops);
  const lastPass = run.passes[run.passes.length - 1];
  const unreached = fullPath.slice(lastPass.hopsDone);

  return (
    <div className="tg-lab tg-lab--hero icl-lab a08-lab">
      <LabSurface label="Prompt as context" className="icl-prompt-card">
        <SurfaceHeading
          kicker={`${shown} example${shown === 1 ? "" : "s"} · ${instruction ? "rule stated" : "examples only"} · query ${query}`}
          title="The examples are evidence about which task you mean"
          aside={<span className="tg-badge">{`${surviving.length} of ${RULES.length} rules fit`}</span>}
        />
        <div className="icl-prompt-layout">
          <pre className="icl-prompt" aria-label="Prompt text the model would read">
            {instruction && (
              <span className="icl-prompt__instruction">
                {INSTRUCTION}
                {"\n"}
              </span>
            )}
            {EXAMPLES.slice(0, shown).map((example) => (
              <span key={example.word}>
                {example.word} {ARROW} {example.answer}
                {"\n"}
              </span>
            ))}
            <span className="icl-prompt__query">
              {query} {ARROW} <b>?</b>
            </span>
          </pre>
          <div className="icl-ask-controls">
            <div className="icl-examples-control">
              <RangeControl
                label="Examples"
                min={0}
                max={EXAMPLES.length}
                step={1}
                value={shown}
                onChange={(value) => {
                  setState({ examples: value });
                  narrate(`${value} examples in the prompt.`);
                }}
              />
            </div>
            <div className="icl-strategy-control">
              <SegmentedControl
                label="Prompt strategy"
                value={instruction ? "rule" : "analogy"}
                options={[
                  { value: "analogy", label: "Examples only" },
                  { value: "rule", label: "State a rule" },
                ]}
                onChange={(value) => setState({ strategy: value })}
              />
            </div>
            <div className="icl-query-control">
              <SegmentedControl
                label="Query"
                value={query}
                options={QUERIES.map((item) => ({ value: item, label: item }))}
                onChange={(value) => setState({ query: value })}
              />
            </div>
          </div>
        </div>
        <div className="icl-rules-scroll">
          <table className="icl-rules icl-rule-matrix">
            <caption className="sr-only">Candidate rules checked against each shown example</caption>
            <thead>
              <tr>
                <th scope="col">Rule</th>
                {EXAMPLES.slice(0, shown).map((example) => (
                  <th scope="col" key={example.word}>
                    {example.word}
                    <small>= {example.answer}</small>
                  </th>
                ))}
                <th scope="col">
                  {query}
                  <small>would be</small>
                </th>
                <th scope="col">Weight</th>
              </tr>
            </thead>
            <tbody>
              {RULES.map((rule, index) => {
                const miss = firstMiss(rule, shown);
                const alive = posterior[index] > 0;
                return (
                  <tr key={rule.id} className={alive ? "is-alive" : "is-out"}>
                    <th scope="row">{rule.name}</th>
                    {EXAMPLES.slice(0, shown).map((example, column) => {
                      const value = rule.apply(example.word);
                      const ok = value === example.answer;
                      return (
                        <td key={example.word} className={ok ? "is-ok" : column === miss ? "is-miss" : "is-after"}>
                          <span>
                            {value} {ok ? "✓" : "✗"}
                          </span>
                        </td>
                      );
                    })}
                    <td className="icl-rules__answer">{rule.apply(query)}</td>
                    <td>
                      {alive ? (
                        <span className="icl-weight">
                          <i aria-hidden="true">
                            <b style={{ width: pct(posterior[index]) }} />
                          </i>
                          {pct(posterior[index])}
                        </span>
                      ) : instruction && !rule.named && miss === -1 ? (
                        "ruled out by the instruction"
                      ) : (
                        `out at ${EXAMPLES[miss]?.word ?? "the instruction"}`
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="icl-answer-layout">
          <BarList
            label={`Answer distribution for ${query}`}
            max={1}
            items={distribution.map((entry) => {
              const backers = RULES.filter(
                (rule, index) => posterior[index] > 0 && rule.apply(query) === entry.answer,
              ).map((rule) => rule.name);
              return {
                id: String(entry.answer),
                label: `${query} ${ARROW} ${entry.answer}`,
                value: entry.probability,
                display: pct(entry.probability),
                tone: entry.answer === trueCount ? ("forward" as const) : ("muted" as const),
                emphasis: entry.answer === trueCount,
                detail: `${entry.answer === trueCount ? "the letter count · " : ""}from ${backers.join(", ")}`,
              };
            })}
          />
          <div className="metric-row">
            <Metric label={`P(${query} ${ARROW} ${trueCount})`} value={pct(pTrue)} tone="forward" />
            <Metric label="Rules still fitting" value={`${surviving.length}`} />
            <Metric label="Weights changed" value="0" />
          </div>
        </div>
        <p className="lab-note">
          Six candidate rules, each a real function of the word, are checked against the examples actually in the
          prompt. Rules that miss an example drop out, and the survivors share the weight equally.
          {instruction ? " The stated rule puts all prior weight on counting the letters." : ""} This is a noise-free
          picture of treating a prompt as evidence; it is not a transformer's internal computation.
        </p>
      </LabSurface>

      <LabSurface label="Copy from context" className="icl-copy-card">
        <SurfaceHeading
          kicker={`hand-set induction head · κ = ${MATCH_SHARPNESS} · last token ${ARROW}`}
          title={seen ? "A seen word: copying finds the answer" : "A new word: copying can only offer old answers"}
        />
        <div
          className="icl-tokens"
          role="img"
          aria-label={`Attention from the final arrow over ${tokens.length - 1} earlier tokens. The head predicts ${headTop.token} with ${pct(headTop.probability)}.`}
        >
          {tokenGroups}
        </div>
        <BarList
          label="Induction head's predicted next token"
          max={1}
          items={head.output.slice(0, 5).map((entry) => ({
            id: entry.token,
            label: `next = ${entry.token}`,
            value: entry.probability,
            display: pct(entry.probability),
            tone: entry.token === String(trueCount) ? "forward" : "muted",
          }))}
        />
        <p className="lab-note">
          Each position scores 1 if the token before it matches the current token {ARROW}, and 2 if the word before that
          also matches the query. Weights are softmax(κ·score) and the head copies whatever sits at the attended
          position.{" "}
          {seen
            ? `The earlier "${query} ${ARROW}" line wins, so the head copies its answer.`
            : "No earlier line starts with this word, so every earlier answer ties. Copying explains the format, a number after the arrow, but not the count."}
        </p>
      </LabSurface>

      <LabSurface label="Chain-of-thought scratchpad" className="icl-scratch-card">
        <SurfaceHeading
          kicker={`k = ${hops} dependent lookups · b = ${budget} per pass · scratchpad ${padOn ? "on" : "off"}`}
          title="Writing the partial result down lets the next pass continue from it"
          aside={<span className="tg-badge">{run.correct ? "answer right" : "answer wrong"}</span>}
        />
        <div className="icl-prompt-layout">
          <div className="icl-scratch-controls">
            <RangeControl
              label="Hops k"
              min={MIN_HOPS}
              max={MAX_HOPS}
              step={1}
              value={hops}
              onChange={(value) => setState({ hops: value })}
            />
            <RangeControl
              label="Per-pass budget b"
              min={MIN_BUDGET}
              max={MAX_BUDGET}
              step={1}
              value={budget}
              onChange={(value) => setState({ budget: value })}
            />
            <SegmentedControl
              label="Scratchpad"
              value={padOn ? "on" : "off"}
              options={[
                { value: "off", label: "off" },
                { value: "on", label: "on" },
              ]}
              onChange={(value) => {
                setState({ scratchpad: value });
                narrate(`Scratchpad ${value}.`);
              }}
            />
          </div>
          <div>
            <pre className="icl-prompt" aria-label="The context the model reads: the prompt, then the tokens it has written">
              {promptLines(hops).join("\n")}
              {"\n"}
              {scratchTokens.length > 0 && (
                <span className="icl-prompt__instruction">
                  {`scratch  ${scratchTokens.join(" ")}`}
                  {"\n"}
                </span>
              )}
              <span className="icl-prompt__query">
                {`answer   `}
                <b>{run.answer}</b>
              </span>
            </pre>
          </div>
        </div>
        <div className="icl-rules-scroll">
          <table className="icl-rules icl-scratch-passes">
            <caption className="sr-only">Each forward pass, where it starts, its silent lookups, and the token it writes</caption>
            <thead>
              <tr>
                <th scope="col">Pass</th>
                <th scope="col">Starts from</th>
                <th scope="col">
                  Silent lookups
                  <small>not written down · at most {budget} per pass</small>
                </th>
                <th scope="col">Writes</th>
              </tr>
            </thead>
            <tbody>
              {run.passes.map((pass) => (
                <tr key={pass.index} className="is-alive">
                  <th scope="row">{pass.index}</th>
                  <td>
                    <span className={`icl-node is-start ${pass.readFrom === "prompt" ? "" : "is-read-back"}`.trim()}>
                      {pass.from}
                    </span>{" "}
                    · {pass.readFrom === "prompt" ? "prompt" : "read back"}
                  </td>
                  <td>
                    <span className="icl-hops">
                      <Hops nodes={pass.lookups} />
                      {Array.from({ length: budget - (pass.lookups.length - 1) }, (_, index) => (
                        <Fragment key={index}>
                          <i className="icl-hop is-spare" aria-hidden="true" />
                          <span className="icl-node is-spare" aria-hidden="true" />
                        </Fragment>
                      ))}
                      {pass.outOfBudget ? <em>{` · budget used up after ${budget}`}</em> : null}
                    </span>
                  </td>
                  <td className="icl-rules__answer">
                    <span className={`icl-node is-written is-${pass.kind}`}>{pass.writes}</span> · {pass.kind}
                  </td>
                </tr>
              ))}
              {unreached.length > 1 && (
                <tr className="icl-unreached">
                  <th scope="row">
                    <span className="sr-only">Not reached</span>
                  </th>
                  <td>never run</td>
                  <td>
                    <span className="icl-hops">
                      <Hops nodes={unreached} tone="is-ghost" />
                    </span>
                  </td>
                  <td>
                    <span className="icl-node is-target">{run.target}</span> · correct
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="metric-row">
          <Metric label="Answer" value={run.answer} tone={run.correct ? "forward" : "loss"} />
          <Metric label="Correct answer" value={run.target} />
          <Metric label="Passes (one token each)" value={`${run.tokens}`} />
        </div>
        <div className="icl-scratch-sweep" style={{ marginTop: "0.9rem" }}>
          <table className="icl-rules icl-sweep">
            <caption className="sr-only">Every hop count at the current per-pass budget, with and without the scratchpad</caption>
            <thead>
              <tr>
                <th scope="col">Hops k</th>
                <th scope="col">
                  Correct
                  <small>from the table</small>
                </th>
                <th scope="col">
                  No scratchpad
                  <small>one pass · filled = hops it covers</small>
                </th>
                <th scope="col">
                  Scratchpad
                  <small>and its cost · a gap = a new pass</small>
                </th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((row) => (
                <tr key={row.hops} className={row.hops === hops ? "is-alive" : undefined}>
                  <th scope="row">{row.hops}</th>
                  <td>{row.target}</td>
                  <td className={row.without.correct ? "is-ok" : "is-miss"}>
                    <span className="icl-sweep__cell">
                      <HopCells hops={row.hops} budget={MAX_HOPS} covered={Math.min(row.hops, budget)} />
                      <span>
                        {row.without.answer} {row.without.correct ? "✓" : "✗"}
                      </span>
                    </span>
                  </td>
                  <td className={row.with.correct ? "is-ok" : "is-miss"}>
                    <span className="icl-sweep__cell">
                      <HopCells hops={row.hops} budget={budget} covered={row.hops} />
                      <span>
                        {row.with.answer} {row.with.correct ? "✓" : "✗"} · {row.with.passes}{" "}
                        {row.with.passes === 1 ? "pass" : "passes"}
                      </span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">
          Each hop is one lookup that needs the one before it. The limit b is imposed by this toy: one pass can chain at
          most b lookups. Without a scratchpad the single pass answers with where it got to. With one, each pass writes
          the node it reached as a token and the next pass starts from that token, reading only the written text. A real
          model&apos;s limit is not a clean number, and a written trace need not match what it computed.
        </p>
      </LabSurface>
    </div>
  );
}
