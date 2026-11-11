import type { ReactNode } from "react";
import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import {
  CARD,
  CLAUSES,
  CLAUSE_NAMES,
  LICENSES,
  SPECTRUM_ROWS,
  TIERS,
  USES,
  checkUse,
  missingForOsaid,
  standardError,
  tierOf,
  trainingEnergy,
  type Outcome,
} from "./terms";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;

const OUTCOME_TEXT: Record<Outcome, string> = {
  yes: "allowed",
  conditional: "allowed with conditions",
  no: "not allowed",
};
const OUTCOME_MARK: Record<Outcome, string> = { yes: "✓", conditional: "◐", no: "✗" };
const OUTCOME_RANK: Record<Outcome, number> = { yes: 0, conditional: 1, no: 2 };

const cellTone = (value: string) =>
  value === "yes" || value === "required" ? "is-yes" : value === "no" ? "is-no" : "is-partial";

/** The mark drawn in a spectrum cell: got it, partly, not promised, or no. */
const cellMark = (value: string) =>
  value === "yes" || value === "required"
    ? { kind: "yes", glyph: "✓" }
    : value === "no"
      ? { kind: "no", glyph: "✗" }
      : value.startsWith("not ")
        ? { kind: "open", glyph: "" }
        : { kind: "conditional", glyph: "" };

/** The spectrum rows fall into what you may do with the model and what it takes to rebuild it. */
const SPECTRUM_GROUPS = [
  { title: "Using it", rows: SPECTRUM_ROWS.slice(0, 3) },
  { title: "Rebuilding it", rows: SPECTRUM_ROWS.slice(3) },
];

function Mark({ kind, glyph }: { kind: string; glyph?: string }) {
  return (
    <span className={`eco-mark is-${kind}`} aria-hidden="true">
      {glyph}
    </span>
  );
}

function Part({ name, shipped, children }: { name: string; shipped: boolean; children?: ReactNode }) {
  return (
    <li className={`eco-part ${shipped ? "is-in" : "is-out"}`}>
      <Mark kind={shipped ? "yes" : "no"} glyph={shipped ? "✓" : "✗"} />
      {children ?? <strong>{name}</strong>}
      <em>{shipped ? "in the box" : "held back"}</em>
    </li>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const license = LICENSES.find((item) => item.id === asString(state, "license", "research")) ?? LICENSES[0];
  const trainingCode = asString(state, "code", "closed") === "released";
  const dataInformation = asString(state, "data", "undisclosed") === "published";
  const use = USES.find((item) => item.id === asString(state, "use", "research")) ?? USES[0];

  const release = { license, trainingCode, dataInformation };
  const tier = tierOf(release);
  const tierIndex = TIERS.findIndex((item) => item.id === tier);
  const tierName = TIERS.find((item) => item.id === tier)?.name ?? "";
  const missing = missingForOsaid(release);
  const check = checkUse(license, use);
  const energy = trainingEnergy(CARD.parameters, CARD.trainingTokens);
  const se = standardError(CARD.harnessScore, CARD.harnessItems);

  /* Evaluation harness, as a picture: the items, then the score with its error bars. */
  const scorePct = CARD.harnessScore * 100;
  const sePct = se * 100;
  const rightItems = Math.round(CARD.harnessScore * CARD.harnessItems);
  const intervalLow = Math.round(scorePct - 1.96 * sePct);
  const intervalHigh = Math.round(scorePct + 1.96 * sePct);

  /* What each stop on the spectrum asks for before the card may move to it. */
  const gates: { text: string; met: boolean }[][] = [
    [],
    [{ text: "weights to download", met: true }],
    [{ text: "any-purpose terms", met: license.anyPurpose }],
    [
      { text: "training code", met: trainingCode },
      { text: "data information", met: dataInformation },
    ],
  ];

  /* Every clause a license has, with the ones this use needs lit; the path carries the strictest mark so far. */
  let soFar: Outcome = "yes";
  const clauseRows = CLAUSES.map((clause) => {
    const needed = check.rows.find((row) => row.clause === clause);
    const incoming = soFar;
    if (needed && OUTCOME_RANK[needed.outcome] > OUTCOME_RANK[soFar]) soFar = needed.outcome;
    return { clause, needed, incoming, outgoing: soFar };
  });

  return (
    <div className="tg-lab tg-lab--hero eco-lab a08-lab">
      <LabSurface label="Sample model card" className="model-card">
        <SurfaceHeading
          kicker="harbor-lm-7b · a fictional release"
          title="Read the card before the download"
          aside={<span className={`tg-badge ${tier === "osaid" ? "" : "is-warning"}`.trim()}>{tierName}</span>}
        />
        <div className="eco-release-controls eco-box">
          <div className={`eco-box-terms ${license.anyPurpose ? "is-yes" : "is-no"}`}>
            <span className="eco-cap">Rules printed on the box</span>
            <div className="eco-license-control">
              <SegmentedControl
                label="License"
                value={license.id}
                options={LICENSES.map((item) => ({ value: item.id, label: item.name }))}
                onChange={(value) => {
                  setState({ license: value });
                  narrate(`License set to ${LICENSES.find((item) => item.id === value)?.name}.`);
                }}
              />
            </div>
            <p className="eco-box-terms-read">
              <Mark kind={license.anyPurpose ? "yes" : "no"} glyph={license.anyPurpose ? "✓" : "✗"} />
              {license.anyPurpose ? "Allows any purpose" : "Does not allow every purpose"}
            </p>
          </div>
          <span className="eco-cap">What comes in the box</span>
          <ul className="eco-parts" aria-label="Parts of this release">
            <Part name="Weights" shipped />
            <Part name="Inference code" shipped />
            <Part name="Training code" shipped={trainingCode}>
              <div className="eco-code-control">
                <SegmentedControl
                  label="Training code"
                  value={trainingCode ? "released" : "closed"}
                  options={[
                    { value: "closed", label: "not released" },
                    { value: "released", label: "released" },
                  ]}
                  onChange={(value) => setState({ code: value })}
                />
              </div>
            </Part>
            <Part name="Data information" shipped={dataInformation}>
              <div className="eco-data-control">
                <SegmentedControl
                  label="Data information"
                  value={dataInformation ? "published" : "undisclosed"}
                  options={[
                    { value: "undisclosed", label: "not published" },
                    { value: "published", label: "published" },
                  ]}
                  onChange={(value) => setState({ data: value })}
                />
              </div>
            </Part>
          </ul>
          <p className="lab-note">
            {tier === "osaid"
              ? "Weights, complete training code, and data information under any-purpose terms: the components OSAID 1.0 asks for. OSI's own review would also check the exact terms."
              : `Open weights, not open source. Still missing for OSAID 1.0: ${missing.join(", ")}.`}
          </p>
        </div>
        <dl className="model-card-fields eco-fields">
          <div>
            <dt>Base model</dt>
            <dd>7B parameters, next-token pretrained on 2T tokens. Not an assistant.</dd>
          </div>
          <div>
            <dt>Fine-tunes</dt>
            <dd>None from the authors. Community adapters exist, unreviewed.</dd>
          </div>
          <div>
            <dt>Weights</dt>
            <dd>Downloadable, with inference code.</dd>
          </div>
          <div>
            <dt>Training code</dt>
            <dd className={trainingCode ? "is-yes" : "is-no"}>{trainingCode ? "Released, complete." : "Not released."}</dd>
          </div>
          <div>
            <dt>Data information</dt>
            <dd className={dataInformation ? "is-yes" : "is-no"}>
              {dataInformation
                ? "Published: sources, filtering steps, and mix, detailed enough to rebuild a similar set. No consent log."
                : "“Web crawl and licensed books.” No sources, mix, or consent log."}
            </dd>
          </div>
          <div>
            <dt>License</dt>
            <dd>
              {license.name}: {license.clauses.commercial.text}.
            </dd>
          </div>
          <div>
            <dt>Evaluation harness</dt>
            <dd>
              harbor-bench, {CARD.harnessItems} items, score {Math.round(scorePct)}%. One standard error is ±
              {sePct.toFixed(1)} points, so a 95% interval spans about {intervalLow}–{intervalHigh}%.
              <div
                className="eco-harness"
                role="img"
                aria-label={`${rightItems} of ${CARD.harnessItems} items right, a score of ${Math.round(scorePct)}%. One standard error either side reaches from ${(scorePct - sePct).toFixed(1)}% to ${(scorePct + sePct).toFixed(1)}%, and the 95% interval from ${intervalLow}% to ${intervalHigh}%.`}
              >
                <div className="eco-items">
                  <span className="eco-items-dots">
                    {Array.from({ length: CARD.harnessItems }, (_, index) => (
                      <i key={index} className={index < rightItems ? "is-right" : ""} />
                    ))}
                  </span>
                  <span className="eco-items-label">
                    {rightItems} of {CARD.harnessItems} right
                  </span>
                </div>
                <div className="eco-interval">
                  <span className="eco-interval-axis" />
                  <span
                    className="eco-interval-whisker"
                    style={{ left: `${intervalLow}%`, width: `${intervalHigh - intervalLow}%` }}
                  />
                  <span
                    className="eco-interval-bar"
                    style={{ left: `${scorePct - sePct}%`, width: `${2 * sePct}%` }}
                  />
                  <span className="eco-interval-dot" style={{ left: `${scorePct}%` }} />
                  <span className="eco-interval-tag is-low" style={{ left: `${intervalLow}%` }}>
                    95% interval: {intervalLow}%
                  </span>
                  <span className="eco-interval-tag is-score" style={{ left: `${scorePct}%` }}>
                    score {Math.round(scorePct)}%
                  </span>
                  <span className="eco-interval-tag is-high" style={{ left: `${intervalHigh}%` }}>
                    {intervalHigh}%
                  </span>
                  <span className="eco-interval-tag is-below" style={{ left: `${scorePct}%` }}>
                    ±{sePct.toFixed(1)} points
                  </span>
                  <span className="eco-interval-tick" style={{ left: "0%" }}>
                    0%
                  </span>
                  <span className="eco-interval-tick is-mid" style={{ left: "50%" }}>
                    50%
                  </span>
                  <span className="eco-interval-tick is-max" style={{ left: "100%" }}>
                    100%
                  </span>
                </div>
              </div>
            </dd>
          </div>
          <div>
            <dt>Training energy (estimated)</dt>
            <dd>
              6 × 7B × 2T = {(energy.flops / 1e22).toFixed(1)}×10²² FLOP. At 40% of an A100's 312 TFLOP/s:{" "}
              {Math.round(energy.gpuHours / 1000)}k GPU-hours × 400 W ≈ {Math.round(energy.megawattHours)} MWh of GPU
              power, one run, not serving.
              <div
                className="eco-chain"
                role="img"
                aria-label={`Compute of ${(energy.flops / 1e22).toFixed(1)}×10²² FLOP, divided by the chips' speed, gives ${Math.round(energy.gpuHours / 1000)}k GPU-hours; times 400 watts gives about ${Math.round(energy.megawattHours)} megawatt-hours.`}
              >
                <span className="eco-chain-box">
                  <strong>{(energy.flops / 1e22).toFixed(1)}×10²² FLOP</strong>
                  <small>compute: 6 × weights × tokens</small>
                </span>
                <span className="eco-chain-op">
                  <small>÷ chip speed</small>
                </span>
                <span className="eco-chain-box">
                  <strong>{Math.round(energy.gpuHours / 1000)}k GPU-hours</strong>
                  <small>time on A100 chips</small>
                </span>
                <span className="eco-chain-op">
                  <small>× 400 W</small>
                </span>
                <span className="eco-chain-box is-result">
                  <strong>≈ {Math.round(energy.megawattHours)} MWh</strong>
                  <small>electricity, one run</small>
                </span>
              </div>
            </dd>
          </div>
        </dl>
      </LabSurface>

      <LabSurface label="Release spectrum" className="eco-spectrum-card">
        <SurfaceHeading kicker="what each kind of release gives you" title={`This card sits in: ${tierName}`} />
        <div className="eco-spectrum-scroll">
          <table className="eco-spectrum">
            <colgroup>
              <col className="eco-col-label" />
              {TIERS.map((item) => (
                <col key={item.id} />
              ))}
            </colgroup>
            <thead>
              <tr className="eco-track">
                <td>
                  <span className="eco-track-cap">less open → more open</span>
                  <span className="eco-track-hint">A gate opens when the card has what is named under it.</span>
                </td>
                {TIERS.map((item, index) => {
                  const reached = index <= tierIndex;
                  const gate = gates[index];
                  const open = gate.every((need) => need.met);
                  return (
                    <td
                      key={item.id}
                      className={`${reached ? "is-reached" : "is-ahead"} ${index === tierIndex ? "is-current" : ""} ${index + 1 <= tierIndex ? "is-leaving" : ""}`.trim()}
                    >
                      <span className="eco-stop" aria-hidden="true" />
                      {gate.length > 0 && (
                        <span className={`eco-gate ${open ? "is-open" : "is-shut"}`}>
                          <span className="eco-gate-bar" aria-hidden="true" />
                          {gate.map((need) => (
                            <span key={need.text} className={need.met ? "is-met" : "is-missing"}>
                              <b aria-hidden="true">{need.met ? "✓" : "✗"}</b> {need.text}
                              <span className="sr-only">{need.met ? " (met)" : " (missing)"}</span>
                            </span>
                          ))}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
              <tr>
                <th scope="col">You get</th>
                {TIERS.map((item) => (
                  <th
                    scope="col"
                    key={item.id}
                    className={item.id === tier ? "is-current" : ""}
                    aria-current={item.id === tier ? "true" : undefined}
                  >
                    {item.name}
                    {item.id === tier && <small>this card</small>}
                  </th>
                ))}
              </tr>
            </thead>
            {SPECTRUM_GROUPS.map((group) => (
              <tbody key={group.title}>
                <tr className="eco-group">
                  <th scope="rowgroup">{group.title}</th>
                  {TIERS.map((item) => (
                    <td key={item.id} className={item.id === tier ? "is-current" : ""} />
                  ))}
                </tr>
                {group.rows.map((row) => (
                  <tr key={row.label}>
                    <th scope="row">{row.label}</th>
                    {TIERS.map((item) => (
                      <td
                        key={item.id}
                        className={`${cellTone(row.cells[item.id])} ${item.id === tier ? "is-current" : ""}`.trim()}
                      >
                        <div className="eco-cell">
                          <Mark {...cellMark(row.cells[item.id])} />
                          <span>{row.cells[item.id]}</span>
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            ))}
          </table>
        </div>
        <ul className="eco-legend" aria-label="What the marks mean">
          <li>
            <Mark kind="yes" glyph="✓" /> you get it
          </li>
          <li>
            <Mark kind="conditional" /> partly, or on conditions
          </li>
          <li>
            <Mark kind="open" /> not promised
          </li>
          <li>
            <Mark kind="no" glyph="✗" /> no
          </li>
        </ul>
        <p className="lab-note">
          Columns are categories, not products. “Not required” means the category does not promise it; some releases in
          that column include it anyway. OSAID 1.0 (October 2024) requires detailed data information, not always the
          data.
        </p>
      </LabSurface>

      <LabSurface label="Is this use allowed" className="eco-use-card">
        <SurfaceHeading
          kicker={`${use.name} under ${license.name}`}
          title="Decide from the terms, not the download button"
          aside={
            <span className={`tg-badge ${check.verdict === "yes" ? "" : "is-warning"}`.trim()}>
              {OUTCOME_TEXT[check.verdict]}
            </span>
          }
        />
        <div className="eco-use-control">
          <SegmentedControl
            label="Intended use"
            value={use.id}
            options={USES.map((item) => ({ value: item.id, label: item.name }))}
            onChange={(value) => setState({ use: value })}
          />
        </div>
        <div className="eco-use-layout">
          <div className="eco-lane">
            <p className="eco-lane-head">
              <span className="eco-cap">Permission</span>
              The license answers this, one clause at a time.
            </p>
            <p className="eco-path-start">
              <strong>{use.name}</strong> needs {check.rows.length} of the license’s {CLAUSES.length} clauses
            </p>
            <ul className="eco-checks" aria-label={`The ${license.name} clauses, with the ones ${use.name} needs marked`}>
              {clauseRows.map(({ clause, needed, incoming, outgoing }) =>
                needed ? (
                  <li key={clause} className={`is-needed is-${needed.outcome} in-${incoming} out-${outgoing}`}>
                    <span className={`eco-mark is-${needed.outcome}`} aria-hidden="true">
                      {OUTCOME_MARK[needed.outcome]}
                    </span>
                    <strong>{CLAUSE_NAMES[clause]}</strong>
                    <em>{OUTCOME_TEXT[needed.outcome]}</em>
                    <small>{needed.text}</small>
                  </li>
                ) : (
                  <li key={clause} className={`is-skipped in-${incoming} out-${outgoing}`}>
                    <strong>{CLAUSE_NAMES[clause]}</strong>
                    <em>not needed for this use</em>
                  </li>
                ),
              )}
            </ul>
            <p className={`eco-path-end is-${check.verdict}`}>
              <span aria-hidden="true">▼</span> the strictest mark decides: <strong>{OUTCOME_TEXT[check.verdict]}</strong>
            </p>
          </div>
          <div className="eco-lane eco-lane--unknown">
            <p className="eco-lane-head">
              <span className="eco-cap">Provenance</span>
              No license can answer this.
            </p>
            <ul className="eco-unknowns" aria-label="What the license cannot answer">
              <li className={dataInformation ? "is-known" : "is-unknown"}>
                <span className={`eco-mark ${dataInformation ? "is-yes" : "is-unknown"}`} aria-hidden="true">
                  {dataInformation ? "✓" : "?"}
                </span>
                <strong>Data sources</strong>
                <em>{dataInformation ? "published on the card" : "unknown"}</em>
                <small>Set by Data information on the model card.</small>
              </li>
              <li className="is-unknown">
                <span className="eco-mark is-unknown" aria-hidden="true">
                  ?
                </span>
                <strong>Consent</strong>
                <em>unknown</em>
                <small>The card keeps no consent log, whatever you set.</small>
              </li>
            </ul>
            <p className="eco-lane-foot">Switching License never changes this column.</p>
          </div>
        </div>
        <div className="metric-row">
          <Metric
            label="License match"
            value={check.verdict === "yes" ? "yes" : check.verdict === "conditional" ? "if conditions met" : "no"}
            tone={check.verdict === "no" ? "loss" : "forward"}
          />
          <Metric label="Clauses checked" value={`${check.rows.length}`} />
          <Metric label="Still unknown" value={dataInformation ? "consent" : "data sources, consent"} tone="loss" />
        </div>
        <p className="lab-note">
          The verdict is the strictest clause the use needs. A license can only grant what its authors had the right to
          grant: it says nothing about whether the training data was collected with consent.
        </p>
      </LabSurface>
    </div>
  );
}
