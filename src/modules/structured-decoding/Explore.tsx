import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  StepThroughController,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import {
  DETOUR,
  distortion,
  distributionAt,
  EOS,
  expectation,
  GRAMMARS,
  matchPrefix,
  VOCAB,
  type GrammarSpec,
  type TokenRow,
} from "./mask";

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const VISIBLE_ROWS = 10;

function TokenChip({ token }: { token: string }) {
  return <code className={`sd-chip ${token === EOS ? "is-eos" : ""}`}>{token}</code>;
}

function MaskTable({ grammar, rows, greedy }: { grammar: GrammarSpec; rows: TokenRow[]; greedy: TokenRow }) {
  const sorted = [...rows].sort((left, right) => right.p - left.p);
  // Show the tokens that matter: every legal one and any illegal one with at least 1% of the mass.
  const shown = sorted.filter((row) => row.legal || row.p >= 0.01).slice(0, VISIBLE_ROWS);
  const rest = sorted.filter((row) => !shown.includes(row));
  const restRaw = rest.reduce((sum, row) => sum + row.p, 0);
  const restLegal = rest.filter((row) => row.legal);
  const restMasked = restLegal.reduce((sum, row) => sum + row.masked, 0);
  const scale = Math.max(...rows.map((row) => Math.max(row.p, row.masked)));
  const width = (value: number) => `${(value / scale) * 100}%`;

  return (
    <div className="sd-mask-table" role="table" aria-label="Next-token probabilities before and after the grammar mask">
      <div className="sd-mask-row sd-mask-row--head" role="row">
        <span role="columnheader">token</span>
        <span role="columnheader">model p</span>
        <span role="columnheader">mask</span>
        <span role="columnheader">after mask</span>
      </div>
      {shown.map((row) => (
        <div
          key={row.token}
          role="row"
          className={`sd-mask-row ${row.legal ? "is-legal" : "is-masked"} ${row.token === greedy.token ? "is-pick" : ""}`}
        >
          <span role="cell" className="sd-mask-token">
            <TokenChip token={row.token} />
            {grammar.straddles[row.token] && row.legal && (
              <small title={`spans ${grammar.straddles[row.token]}`}>↔ {grammar.straddles[row.token]}</small>
            )}
          </span>
          <span role="cell" className="sd-mask-bar">
            <i>
              <b className="sd-bar--raw" style={{ width: width(row.p) }} />
            </i>
            <em>{pct(row.p)}</em>
          </span>
          <span role="cell" className="sd-mask-gate">
            {row.legal ? "✓ legal" : "✗ masked"}
          </span>
          <span role="cell" className="sd-mask-bar">
            <i>
              <b className="sd-bar--masked" style={{ width: width(row.masked) }} />
            </i>
            <em>{row.legal ? pct(row.masked) : "0"}</em>
          </span>
        </div>
      ))}
      {rest.length > 0 && (
        <div className="sd-mask-row sd-mask-row--rest" role="row">
          <span role="cell">+{rest.length} more tokens</span>
          <span role="cell">{pct(restRaw)} together</span>
          <span role="cell">
            {restLegal.length ? `${restLegal.length} legal` : "all masked"}
          </span>
          <span role="cell">{restLegal.length ? `${pct(restMasked)} together` : "0"}</span>
        </div>
      )}
    </div>
  );
}

interface RailCell {
  char: string;
  /** Index of the emitted token that wrote this character, or -1 when it is not written yet. */
  token: number;
  next: boolean;
}

/**
 * The grammar as a track of characters, with the text so far laid onto it. Which cells are "next" comes
 * from the matcher's own states, so the picture and the mask can never disagree.
 */
function GrammarRail({ grammar, emitted, expects }: { grammar: GrammarSpec; emitted: readonly string[]; expects: string }) {
  const prefix = emitted.join("");
  const owner: number[] = [];
  emitted.forEach((token, index) => {
    for (let i = 0; i < token.length; i += 1) owner.push(index);
  });
  const states = matchPrefix(grammar, prefix);
  const finished = states.some((state) => state.part === grammar.parts.length);
  let cursor = 0;

  const cells = (text: string, start: number, written: number, nextAt: number): RailCell[] =>
    [...text].map((char, index) => ({
      char,
      token: index < written ? owner[start + index] ?? -1 : -1,
      next: index === nextAt,
    }));

  const groups = (row: RailCell[]) => {
    const out: RailCell[][] = [];
    row.forEach((cell) => {
      const last = out[out.length - 1];
      if (last && cell.token >= 0 && last[0].token === cell.token) last.push(cell);
      else out.push([cell]);
    });
    return out;
  };

  const drawRow = (row: RailCell[]) =>
    groups(row).map((group, index) =>
      group[0].token >= 0 ? (
        <span key={index} className="sd-rail-tok" title={`token ${group[0].token + 1}: ${emitted[group[0].token]}`}>
          {group.map((cell, position) => (
            <i key={position}>{cell.char}</i>
          ))}
        </span>
      ) : (
        <i key={index} className={`sd-rail-cell${group[0].next ? " is-next" : ""}`}>
          {group[0].char}
        </i>
      ),
    );

  const parts = grammar.parts.map((part, partIndex) => {
    const here = states.filter((state) => state.part === partIndex);
    const rest = prefix.slice(cursor);
    const start = cursor;
    if (part.kind === "literal") {
      let written = 0;
      while (written < part.text.length && rest[written] === part.text[written]) written += 1;
      cursor += written;
      const nextAt = here.find((state) => state.offset < part.text.length)?.offset ?? -1;
      return (
        <div key={partIndex} className="sd-rail-part">
          <span className="sd-rail-row">{drawRow(cells(part.text, start, written, nextAt))}</span>
          <small>fixed text</small>
        </div>
      );
    }
    if (part.kind === "choice") {
      const taken = rest.length
        ? part.options.findIndex((option) => option.startsWith(rest) || rest.startsWith(option))
        : -1;
      const written = taken >= 0 ? Math.min(part.options[taken].length, rest.length) : 0;
      cursor += written;
      return (
        <div key={partIndex} className="sd-rail-part is-choice">
          {part.options.map((option, optionIndex) => {
            const state = here.find((item) => item.option === optionIndex || item.option === -1);
            const nextAt = state && state.offset < option.length ? state.offset : -1;
            const dead = taken >= 0 && taken !== optionIndex;
            return (
              <span key={option} className={`sd-rail-row${dead ? " is-dead" : ""}`}>
                {drawRow(cells(option, start, taken === optionIndex ? written : 0, dead ? -1 : nextAt))}
              </span>
            );
          })}
          <small>one of these</small>
        </div>
      );
    }
    let written = 0;
    while (written < rest.length && /[A-Za-z]/.test(rest[written])) written += 1;
    cursor += written;
    return (
      <div key={partIndex} className="sd-rail-part">
        <span className="sd-rail-row">
          {drawRow(cells(rest.slice(0, written), start, written, -1))}
          <i className={`sd-rail-cell is-loop${here.length ? " is-next" : ""}`}>a–z ↻</i>
        </span>
        <small>any letters, one or more</small>
      </div>
    );
  });

  return (
    <div
      className="sd-rail"
      role="img"
      aria-label={`The grammar as a track of characters. Written so far: ${prefix || "nothing"}. Accepted next: ${expects}.`}
    >
      <span className="sd-rail-end">start</span>
      {parts.map((part, index) => (
        <span key={index} className="sd-rail-step">
          <b aria-hidden="true">→</b>
          {part}
        </span>
      ))}
      <span className="sd-rail-step">
        <b aria-hidden="true">→</b>
        <span className={`sd-rail-end is-stop${finished ? " is-next" : ""}`}>end {EOS}</span>
      </span>
    </div>
  );
}

/** The whole next-token distribution as one bar, and the legal slice of it stretched back to 100%. */
function RenormBars({ rows, z, greedy }: { rows: TokenRow[]; z: number; greedy: TokenRow }) {
  const byMass = [...rows].sort((left, right) => right.p - left.p);
  const legal = byMass.filter((row) => row.legal);
  const masked = byMass.filter((row) => !row.legal && row.p >= 0.02);
  const dust = byMass.filter((row) => !row.legal && row.p < 0.02);
  const dustMass = dust.reduce((sum, row) => sum + row.p, 0);
  const tag = (row: TokenRow, value: number) =>
    value >= 0.085 ? (
      <span>
        {row.token} <em>{pct(value, 0)}</em>
      </span>
    ) : null;

  return (
    <figure
      className="sd-renorm"
      role="img"
      aria-label={`The model's probability as one bar: ${pct(z)} on ${legal.length} legal token${
        legal.length === 1 ? "" : "s"
      } and ${pct(1 - z)} on masked tokens. After the mask the legal part is stretched to 100%: ${legal
        .slice(0, 4)
        .map((row) => `${row.token} ${pct(row.masked)}`)
        .join(", ")}.`}
    >
      <div className="sd-renorm__caption">
        <strong>Before</strong> the model’s probability over all {rows.length} tokens
      </div>
      <div className="sd-renorm__bar">
        {legal.map((row) => (
          <b
            key={row.token}
            className={`is-legal${row.token === greedy.token ? " is-pick" : ""}`}
            style={{ width: pct(row.p, 3) }}
          >
            {tag(row, row.p)}
          </b>
        ))}
        {masked.map((row) => (
          <b key={row.token} className="is-masked" style={{ width: pct(row.p, 3) }}>
            {tag(row, row.p)}
          </b>
        ))}
        {dustMass > 0 && (
          <b className="is-masked is-dust" style={{ width: pct(dustMass, 3) }}>
            {dustMass >= 0.12 ? (
              <span>
                {dust.length} more <em>{pct(dustMass, 0)}</em>
              </span>
            ) : null}
          </b>
        )}
      </div>
      <div className="sd-renorm__split">
        <span className="is-legal" style={{ width: pct(z, 3) }} />
        <span className="is-masked" />
        <em className="is-legal">legal · Z = {z.toFixed(3)} kept</em>
        <em className="is-masked">✗ masked · {pct(1 - z, 0)} removed</em>
      </div>
      <svg className="sd-renorm__zoom" viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">
        <polygon points={`0,0 ${(z * 100).toFixed(2)},0 100,10 0,10`} />
      </svg>
      <div className="sd-renorm__bar is-after">
        {legal.map((row) => (
          <b
            key={row.token}
            className={`is-legal${row.token === greedy.token ? " is-pick" : ""}`}
            style={{ width: pct(row.masked, 3) }}
          >
            {tag(row, row.masked)}
          </b>
        ))}
      </div>
      <div className="sd-renorm__caption">
        <strong>After</strong> the legal part ÷ Z, stretched back to 100%. The ratios between legal tokens do not
        change.
      </div>
    </figure>
  );
}

function DetourTree({ legalAfterO }: { legalAfterO: number }) {
  const { O, B, P } = DETOUR.first;
  const afterB = DETOUR.afterB.legal;
  const nodes = [
    { id: "O", x: 150, y: 44, label: '"O', p: O, legal: true },
    { id: "B", x: 150, y: 132, label: '"B', p: B, legal: true },
    { id: "P", x: 150, y: 196, label: '"P', p: P, legal: false },
  ];
  const leaves = [
    { from: "O", x: 292, y: 22, label: 'slo"', city: "Oslo", p: legalAfterO, legal: true },
    { from: "O", x: 292, y: 70, label: 'ttawa"', city: "Ottawa", p: 1 - legalAfterO, legal: false },
    { from: "B", x: 292, y: 114, label: 'ergen"', city: "Bergen", p: afterB, legal: true },
    { from: "B", x: 292, y: 158, label: 'ern"', city: "Bern", p: 1 - afterB, legal: false },
  ];
  const nodeAt = (id: string) => nodes.find((node) => node.id === id)!;
  // A branch is as thick as the share of probability the model sends down it.
  const flow = (p: number) => 1 + p * 8;
  return (
    <svg
      className="sd-tree"
      viewBox="0 0 430 214"
      role="img"
      aria-label={`Token tree for the value of an enum field that allows Oslo or Bergen. First token: "O ${pct(O, 0)}, "B ${pct(B, 0)}, "P ${pct(P, 0)} (masked). After "O: slo" ${pct(legalAfterO, 0)}, ttawa" ${pct(1 - legalAfterO, 0)} (masked). After "B: ergen" ${pct(afterB, 0)}, ern" ${pct(1 - afterB, 0)} (masked).`}
    >
      <text className="sd-tree-root" x={0} y={118}>
        {'{"city":'}
      </text>
      {nodes.map((node) => (
        <g key={node.id} className={node.legal ? "is-legal" : "is-masked"}>
          <path d={`M66 114 C 96 114, 96 ${node.y}, ${node.x - 24} ${node.y}`} style={{ strokeWidth: flow(node.p) }} />
          <text className="sd-tree-p" x={node.x - 30} y={node.y < 114 ? node.y - 6 : node.y + 17} textAnchor="end">
            {pct(node.p, 0)}
          </text>
          <rect x={node.x - 24} y={node.y - 12} width={48} height={24} rx={5} />
          <text x={node.x} y={node.y + 4.5} textAnchor="middle">
            {node.label}
          </text>
        </g>
      ))}
      {leaves.map((leaf) => {
        const parent = nodeAt(leaf.from);
        return (
          <g key={leaf.label} className={leaf.legal ? "is-legal" : "is-masked"}>
            <path
              d={`M${parent.x + 24} ${parent.y} C 205 ${parent.y}, 205 ${leaf.y}, ${leaf.x - 36} ${leaf.y}`}
              style={{ strokeWidth: flow(leaf.p) }}
            />
            <text
              className="sd-tree-p"
              x={leaf.x - 42}
              y={leaf.y > parent.y ? leaf.y + 17 : leaf.y - 6}
              textAnchor="end"
            >
              {pct(leaf.p, 0)}
            </text>
            <rect x={leaf.x - 36} y={leaf.y - 11} width={72} height={22} rx={5} />
            <text x={leaf.x} y={leaf.y + 4.5} textAnchor="middle">
              {leaf.label}
            </text>
            <text className="sd-tree-city" x={leaf.x + 44} y={leaf.y + 4.5}>
              {leaf.legal ? `→ ${leaf.city}` : `${leaf.city} ✗`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function SplitBar({ label, oslo, detail }: { label: string; oslo: number; detail: string }) {
  return (
    <div className="sd-split">
      <span className="sd-split__label">
        {label}
        <b>
          Oslo {pct(oslo, 0)} · Bergen {pct(1 - oslo, 0)}
        </b>
      </span>
      <div
        className="sd-split__bar"
        role="img"
        aria-label={`${label}: Oslo ${pct(oslo, 0)}, Bergen ${pct(1 - oslo, 0)}.`}
      >
        <b className="sd-split__oslo" style={{ width: pct(oslo, 2) }}>
          {oslo > 0.25 && "Oslo"}
        </b>
        <b className="sd-split__bergen" style={{ width: pct(1 - oslo, 2) }}>
          {1 - oslo > 0.25 && "Bergen"}
        </b>
      </div>
      <small>{detail}</small>
    </div>
  );
}

export default function Explore({ state, setState }: ModuleContext) {
  const grammarId = state.grammar === "call" ? "call" : "json";
  const grammar = GRAMMARS[grammarId];
  const lastStep = grammar.walk.length - 1;
  const step = clamp(Math.round(asNumber(state, "step", 0)), 0, lastStep);
  const legalAfterO = clamp(asNumber(state, "detour", 0.1), 0.05, 1);

  const { prefix, rows, z, rawTop, greedy, legalCount } = distributionAt(grammar, step);
  const expects = expectation(grammar, prefix);
  const detour = distortion(legalAfterO);
  const emitted = grammar.walk.slice(0, step);

  return (
    <div className="sd-lab">
      <LabSurface label="Grammar" className="grammar-card sd-grammar-card">
        <SurfaceHeading kicker={grammar.label} title="A filter, not a knowledge source" />
        <div className="sd-grammar-body">
          <div className="sd-grammar-side">
            <div className="grammar-control">
              <SegmentedControl
                label="Grammar"
                value={grammarId}
                options={[
                  { value: "json", label: "JSON object" },
                  { value: "call", label: "Function call" },
                ]}
                onChange={(value) => setState({ grammar: value, step: 0 })}
              />
            </div>
            <p className="lab-note">
              The lab compiles the pattern into a character-level matcher. A token is legal when all of its
              characters can follow the text so far. Switching grammar resets Decode step.
            </p>
          </div>
          <dl className="sd-spec">
            <dt>Prompt</dt>
            <dd>{grammar.prompt}</dd>
            <dt>Allowed output</dt>
            <dd>
              <code>{grammar.pattern}</code>
            </dd>
            <dt>Rule</dt>
            <dd>{grammar.source}</dd>
            <dt>Vocabulary</dt>
            <dd>{VOCAB.length} tokens, some spanning two grammar symbols</dd>
          </dl>
        </div>
      </LabSurface>

      <LabSurface label="Prefix walk" className="prefix-card sd-prefix-card">
        <SurfaceHeading
          kicker={`Step ${step + 1} of ${grammar.walk.length} · greedy constrained decoding`}
          title="The mask is computed from the text emitted so far"
        />
        <div className="sd-prefix" aria-label={`Emitted so far: ${prefix || "nothing"}`}>
          {emitted.length === 0 && <span className="sd-prefix__empty">empty prefix</span>}
          {emitted.map((token, index) => (
            <TokenChip key={`${token}-${index}`} token={token} />
          ))}
          <code className="sd-chip sd-chip--next">?</code>
        </div>
        <GrammarRail grammar={grammar} emitted={emitted} expects={expects} />
        <p className="sd-expect">
          Grammar accepts next: <strong>{expects}</strong>
        </p>
        <StepThroughController
          label="Decode step"
          steps={grammar.walk.map((token) => token)}
          current={step}
          onChange={(value) => setState({ step: value })}
        />
        <div className="metric-row">
          <Metric
            label="Unconstrained pick"
            value={`${rawTop.token}${rawTop.legal ? "" : " ✗"}`}
            tone={rawTop.legal ? undefined : "loss"}
          />
          <Metric label="Pick after mask" value={greedy.token} tone="forward" />
          <Metric label="Legal tokens" value={`${legalCount} / ${VOCAB.length}`} />
        </div>
      </LabSurface>

      <LabSurface label="Masked distribution" className="mask-card sd-mask-card">
        <SurfaceHeading
          kicker={`Legal mass Z = ${z.toFixed(3)} · ${pct(1 - z, 0)} of the model's probability removed`}
          title="Zero the illegal tokens, then divide the survivors by what is left"
        />
        <RenormBars rows={rows} z={z} greedy={greedy} />
        <div className="token-mask">
          <MaskTable grammar={grammar} rows={rows} greedy={greedy} />
        </div>
        <div className="sd-formula" role="group" aria-label="Renormalization of the greedy pick">
          <code>
            p′({greedy.token}) = p({greedy.token}) ÷ Z = {greedy.p.toFixed(3)} ÷ {z.toFixed(3)} ={" "}
            <strong>{greedy.masked.toFixed(3)}</strong>
          </code>
          <small>
            Z sums the model&apos;s probability over the {legalCount} legal token{legalCount === 1 ? "" : "s"}. Same result
            as setting illegal logits to −∞ before the softmax.
          </small>
        </div>
      </LabSurface>

      <LabSurface label="Mask distortion" className="sd-distortion-card">
        <SurfaceHeading
          kicker={'Separate example · enum field: "Oslo" or "Bergen"'}
          title="Masking one token at a time is not the same as conditioning on a valid answer"
        />
        <div className="sd-distortion">
          <DetourTree legalAfterO={legalAfterO} />
          <div className="sd-distortion__side">
            <div className="sd-distortion-control">
              <RangeControl
                label={'P(slo" after "O)'}
                min={0.05}
                max={1}
                step={0.05}
                value={legalAfterO}
                onChange={(value) => setState({ detour: Number(value.toFixed(2)) })}
                format={(value) => pct(value, 0)}
              />
            </div>
            <SplitBar
              label="Per-step mask"
              oslo={detour.stepwise.oslo}
              detail={`"O keeps ${pct(DETOUR.first.O, 0)} ÷ ${pct(DETOUR.first.O + DETOUR.first.B, 0)} of the first step, then the mask forces slo".`}
            />
            <SplitBar
              label="Model, conditioned on a valid string"
              oslo={detour.conditioned.oslo}
              detail={`Oslo ${detour.joint.oslo.toFixed(3)} vs Bergen ${detour.joint.bergen.toFixed(3)} as whole strings, divided by ${detour.joint.valid.toFixed(3)}.`}
            />
          </div>
        </div>
        <p className="lab-note">
          The per-step mask cannot see that the model chose &quot;O on its way to Ottawa. It commits, then forces slo&quot;.
          Drag the slider to 90%: once both branches keep the same share of legal continuations, the two bars match.
        </p>
      </LabSurface>
    </div>
  );
}
