import { LabSurface, Metric, SegmentedControl, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import {
  CARD,
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

const cellTone = (value: string) =>
  value === "yes" || value === "required" ? "is-yes" : value === "no" ? "is-no" : "is-partial";

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const license = LICENSES.find((item) => item.id === asString(state, "license", "research")) ?? LICENSES[0];
  const trainingCode = asString(state, "code", "closed") === "released";
  const dataInformation = asString(state, "data", "undisclosed") === "published";
  const use = USES.find((item) => item.id === asString(state, "use", "research")) ?? USES[0];

  const release = { license, trainingCode, dataInformation };
  const tier = tierOf(release);
  const tierName = TIERS.find((item) => item.id === tier)?.name ?? "";
  const missing = missingForOsaid(release);
  const check = checkUse(license, use);
  const energy = trainingEnergy(CARD.parameters, CARD.trainingTokens);
  const se = standardError(CARD.harnessScore, CARD.harnessItems);

  return (
    <div className="tg-lab tg-lab--hero eco-lab a08-lab">
      <LabSurface label="Sample model card" className="model-card">
        <SurfaceHeading
          kicker="harbor-lm-7b · a fictional release"
          title="Read the card before the download"
          aside={<span className={`tg-badge ${tier === "osaid" ? "" : "is-warning"}`.trim()}>{tierName}</span>}
        />
        <div className="eco-card-layout">
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
              <dd className={trainingCode ? "is-yes" : "is-no"}>
                {trainingCode ? "Released, complete." : "Not released."}
              </dd>
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
                harbor-bench, {CARD.harnessItems} items, score {Math.round(CARD.harnessScore * 100)}%. One standard
                error is ±{(se * 100).toFixed(1)} points, so a 95% interval spans about{" "}
                {Math.round(CARD.harnessScore * 100 - 196 * se)}–{Math.round(CARD.harnessScore * 100 + 196 * se)}%.
              </dd>
            </div>
            <div>
              <dt>Training energy (estimated)</dt>
              <dd>
                6 × 7B × 2T = {(energy.flops / 1e22).toFixed(1)}×10²² FLOP. At 40% of an A100's 312 TFLOP/s:{" "}
                {Math.round(energy.gpuHours / 1000)}k GPU-hours × 400 W ≈ {Math.round(energy.megawattHours)} MWh of GPU
                power, one run, not serving.
              </dd>
            </div>
          </dl>
          <div className="eco-release-controls">
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
            <p className="lab-note">
              {tier === "osaid"
                ? "Weights, complete training code, and data information under any-purpose terms: the components OSAID 1.0 asks for. OSI's own review would also check the exact terms."
                : `Open weights, not open source. Still missing for OSAID 1.0: ${missing.join(", ")}.`}
            </p>
          </div>
        </div>
      </LabSurface>

      <LabSurface label="Release spectrum" className="eco-spectrum-card">
        <SurfaceHeading kicker="what each kind of release gives you" title={`This card sits in: ${tierName}`} />
        <div className="eco-spectrum-scroll">
          <table className="eco-spectrum">
            <thead>
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
            <tbody>
              {SPECTRUM_ROWS.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  {TIERS.map((item) => (
                    <td
                      key={item.id}
                      className={`${cellTone(row.cells[item.id])} ${item.id === tier ? "is-current" : ""}`.trim()}
                    >
                      {row.cells[item.id]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
        <div className="eco-use-layout">
          <div>
            <div className="eco-use-control">
              <SegmentedControl
                label="Intended use"
                value={use.id}
                options={USES.map((item) => ({ value: item.id, label: item.name }))}
                onChange={(value) => setState({ use: value })}
              />
            </div>
            <ul className="eco-checks" aria-label={`Clauses ${use.name} needs`}>
              {check.rows.map((row) => (
                <li key={row.clause} className={`is-${row.outcome}`}>
                  <span aria-hidden="true">{OUTCOME_MARK[row.outcome]}</span>
                  <strong>{CLAUSE_NAMES[row.clause]}</strong>
                  <em>{OUTCOME_TEXT[row.outcome]}</em>
                  <small>{row.text}</small>
                </li>
              ))}
            </ul>
          </div>
          <div>
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
              The verdict is the strictest clause the use needs. A license can only grant what its authors had the right
              to grant: it says nothing about whether the training data was collected with consent.
            </p>
          </div>
        </div>
      </LabSurface>
    </div>
  );
}
