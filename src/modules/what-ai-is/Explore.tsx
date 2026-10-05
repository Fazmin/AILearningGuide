import { useEffect, useState } from "react";
import {
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  StageFlow,
  SurfaceHeading,
  type ModuleContext,
} from "@app/module-sdk";
import { ERAS, ERA_YEARS, eraYearFromState } from "./eras";
import { STEP_SIZE, downhill, gradient, leastSquares, mse, residuals } from "./fit";
import { DataPlot, ParamPlot } from "./plots";
import {
  OUTPUT_KINDS,
  SCENARIOS,
  SCENARIO_IDS,
  SIGNALS,
  SIGNAL_IDS,
  parseSignalPlacements,
  scenarioById,
  scoreSignalPlacements,
  signalById,
} from "./signals";

const RINGS = [
  {
    id: "ai",
    label: "AI",
    radius: 148,
    learns: "Any method that performs a task that used to need judgment. A rule list counts.",
    math: "No shared training story. Some methods have no parameters θ at all.",
  },
  {
    id: "ml",
    label: "ML",
    radius: 118,
    learns: "Fit parameters from examples instead of writing the decision rule by hand.",
    math: "Search θ in a hypothesis class so f_θ(x) ≈ y on data, then hope it holds on new x.",
  },
  {
    id: "dl",
    label: "DL",
    radius: 90,
    learns: "Stack differentiable layers so later features reuse earlier ones.",
    math: "f_θ is a composition of maps. Representation and predictor are fitted together.",
  },
  {
    id: "gen",
    label: "Gen",
    radius: 64,
    learns: "Neural generative models: deep networks trained to produce new samples from a distribution — text, images, or audio.",
    math: "The object is a distribution p_θ(x), or p_θ(x | context), not only a label ŷ. This ring holds the neural ones.",
  },
  {
    id: "llm",
    label: "LLM",
    radius: 40,
    learns: "Predict the next token in text. Assistant behavior is added later.",
    math: "Training job is p_θ(token_t | token_<t). Chat style is a later wrap, not the loss.",
  },
] as const;

const TERMS = [
  {
    id: "chess",
    label: "Hand-written chess rules",
    ring: "ai",
    why: "Someone wrote the rule. Nothing was fitted from examples, so it is AI without machine learning.",
  },
  {
    id: "linreg",
    label: "Linear regression on house prices",
    ring: "ml",
    why: "Parameters come from labeled examples, but there is no stack of learned layers.",
  },
  {
    id: "convnet",
    label: "Convolutional image classifier",
    ring: "dl",
    why: "Stacked filters are deep learning. A classifier need not generate new images.",
  },
  {
    id: "gan",
    label: "A GAN that samples faces",
    ring: "gen",
    why: "It draws new samples from a distribution, but it is not a next-token language model.",
  },
  {
    id: "chat",
    label: "A next-token chat assistant",
    ring: "llm",
    why: "Token prediction is the training job; the assistant wrapper sits on an LLM.",
  },
] as const;

const STAGE_DETAILS = ["3 labeled pairs", "ŷ = w x + b", "L = mean (ŷ − y)²", "θ ← θ − η ∇L", "error on a new x"] as const;

const STAGES = [
  {
    id: "data",
    name: "Data",
    title: "Three labeled pairs, already numbers",
    body: "Each row is (x, y). y is the label someone chose. The model never sees a meaning for x beyond the number you plot.",
  },
  {
    id: "h",
    name: "Hypothesis class",
    title: "H = { x ↦ w x + b }",
    body: "Every pair (w, b) is one hypothesis. The class is every straight line, not “intelligence.” A different class (a quadratic, a tree) could fit these three points differently.",
  },
  {
    id: "loss",
    name: "Loss",
    title: "L = (1/3) Σ (ŷ − y)²",
    body: "Mean squared error on these three points only; each shaded square is one (ŷ − y)², and L is their average area. A smaller L means a better fit here. It is the training objective, not a claim about new x.",
  },
  {
    id: "opt",
    name: "Optimization",
    title: "Search θ by moving downhill on L",
    body: "The sliders are you choosing θ, a point on the contour map. Take a downhill step moves that point by −0.12 × the closed-form gradient of this MSE. That is the whole “learning” step: change parameters so L shrinks.",
  },
  {
    id: "gen",
    name: "Generalization",
    title: "Low L on these 3 points is not the test",
    body: "A new x drawn from the same process can still be wrong. Memorizing the three residuals is allowed by a rich enough H. Later labs score a held-out slice for that reason.",
  },
] as const;

const DISTINCTIONS = [
  {
    id: "family",
    label: "family",
    title: "Symbolic, statistical, deep — different objects",
    tiles: [
      {
        title: "Symbolic",
        body: "Rules, search, logic. The decision is written. No θ is fitted from a table.",
      },
      {
        title: "Statistical ML",
        body: "Choose features, then fit θ so f_θ(x) matches labels. Linear regression lives here.",
      },
      {
        title: "Deep learning",
        body: "Stack differentiable maps. Features and predictor are fitted together from examples.",
      },
    ],
    note: "The rings nest modern methods. They do not say symbolic AI disappeared. A checker or a planner can still be AI with zero learned weights.",
  },
  {
    id: "scope",
    label: "scope",
    title: "Narrow systems, not a general mind",
    tiles: [
      {
        title: "Narrow",
        body: "One task family: prices, spam, next token. The hypothesis class is built for that map.",
      },
      {
        title: "Broad tool",
        body: "One trained network used on many prompts. Still a predictor, still bounded by its data and loss.",
      },
      {
        title: "Not AGI here",
        body: "AGI, artificial general intelligence, names a system that could do any cognitive task a person can. People define it differently and there is no agreed test, so it is a goal, not a measured property. These labs never claim it, and fluency on text is not evidence of it.",
      },
    ],
    note: "A chat model looks general because language is a wide interface. The training job is still next-token prediction plus later wraps.",
  },
  {
    id: "job",
    label: "job",
    title: "Prediction is not the same as reasoning",
    tiles: [
      {
        title: "Predict",
        body: "Return ŷ or p_θ(y | x). That is the learning problem on this page.",
      },
      {
        title: "Reason / search",
        body: "Explore steps, proofs, or tool calls. A predictor can be a subroutine inside that search.",
      },
      {
        title: "Fluent ≠ proved",
        body: "A high-likelihood continuation can read like an argument and still be an interpolation.",
      },
    ],
    note: "Later reasoning labs add search on top of a predictor. They do not redefine the loss as “thinking.”",
  },
  {
    id: "memory",
    label: "memory",
    title: "Fitting the sample versus working on new draws",
    tiles: [
      {
        title: "Memorize",
        body: "Drive L to ~0 on the training points. A rich H can copy residuals instead of a rule.",
      },
      {
        title: "Generalize",
        body: "Small error on new x from the same process. That is the scientific target, and it is not visible in training L alone.",
      },
      {
        title: "This toy",
        body: "Three points and a line. You can make L small and still know nothing about a fourth x.",
      },
    ],
    note: "How we know a model works is the later lab that scores a slice the fit never saw.",
  },
  {
    id: "not",
    label: "not",
    title: "What the math does not require",
    tiles: [
      {
        title: "Not consciousness",
        body: "Nothing in f_θ, L, or gradient descent posits an inner experience. The objects are functions and numbers.",
      },
      {
        title: "Not folk understanding",
        body: "Matching y on data does not imply a human-like grasp of the domain. The test is a number on a distribution.",
      },
      {
        title: "Not a verdict on use",
        body: "A well-fit predictor can still be the wrong tool, the wrong label, or an unsafe deployment. That is a later civic question.",
      },
    ],
    note: "Refusing those extra claims is precision, not a slight. The course stays with what the trained map actually does.",
  },
] as const;

const RING_OPTIONS = RINGS.map((ring) => ({ value: ring.id, label: ring.label }));
const DISTINCTION_OPTIONS = DISTINCTIONS.map((item) => ({ value: item.id, label: item.label }));
const SIGNAL_OPTIONS = SIGNALS.map((item) => ({ value: item.id, label: item.label }));

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const asStringArray = (state: ModuleContext["state"], key: string) =>
  Array.isArray(state[key]) ? state[key].filter((item): item is string => typeof item === "string") : [];

const ringIndex = (id: string) => RINGS.findIndex((ring) => ring.id === id);

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const ring = RINGS.some((item) => item.id === asString(state, "ring", "ai"))
    ? asString(state, "ring", "ai")
    : "ai";
  const era = Math.max(0, ERA_YEARS.indexOf(eraYearFromState(state.eraYear, state.era)));
  const term = TERMS.some((item) => item.id === asString(state, "term", "chess"))
    ? asString(state, "term", "chess")
    : "chess";
  const destination = RINGS.some((item) => item.id === asString(state, "destination", "ai"))
    ? asString(state, "destination", "ai")
    : "ai";
  const placements = asStringArray(state, "placements");
  const stage = Math.min(STAGES.length - 1, Math.max(0, Math.round(asNumber(state, "stage", 1))));
  const weight = Math.min(2, Math.max(0, asNumber(state, "weight", 1)));
  const bias = Math.min(2.4, Math.max(0, asNumber(state, "bias", 0.9)));
  const distinction = DISTINCTIONS.some((item) => item.id === asString(state, "distinction", "family"))
    ? asString(state, "distinction", "family")
    : "family";

  const scenarioId = SCENARIO_IDS.includes(asString(state, "scenario", "house"))
    ? asString(state, "scenario", "house")
    : "house";
  const signalChoice = SIGNAL_IDS.includes(asString(state, "signal", "supervised"))
    ? asString(state, "signal", "supervised")
    : "supervised";
  const signalPlacements = parseSignalPlacements(state.signalPlacements);
  const signalPlaced = Object.fromEntries(signalPlacements.map((entry) => entry.split(":") as [string, string]));
  const signalScore = scoreSignalPlacements(signalPlacements);
  const selectedScenario = scenarioById(scenarioId);
  const currentSignalPlacement = signalPlaced[selectedScenario.id];
  const signalVerdict =
    currentSignalPlacement === undefined
      ? "unplaced"
      : currentSignalPlacement === selectedScenario.signal
        ? "correct"
        : "corrected";

  const selectedRing = RINGS.find((item) => item.id === ring) ?? RINGS[0];
  const selectedTerm = TERMS.find((item) => item.id === term) ?? TERMS[0];
  const selectedEra = ERAS[era];
  const selectedStage = STAGES[stage];
  const selectedDistinction = DISTINCTIONS.find((item) => item.id === distinction) ?? DISTINCTIONS[0];
  const placed = Object.fromEntries(
    placements.map((entry) => {
      const [id, placedRing] = entry.split(":");
      return [id, placedRing];
    }),
  );
  const placedCount = TERMS.filter((item) => placed[item.id]).length;
  const correctCount = TERMS.filter((item) => placed[item.id] === item.ring).length;
  const currentPlacement = placed[selectedTerm.id];
  const currentVerdict =
    currentPlacement === undefined
      ? "unplaced"
      : currentPlacement === selectedTerm.ring
        ? "correct"
        : "corrected";

  const fitted = residuals(weight, bias);
  const loss = mse(weight, bias);
  const { w: gradW, b: gradB } = gradient(weight, bias);
  const best = leastSquares();

  /** Path of downhill steps, kept as view state: dragging a slider starts a fresh path. */
  const [trail, setTrail] = useState<{ w: number; b: number }[]>([{ w: weight, b: bias }]);
  useEffect(() => {
    setTrail((previous) => {
      const last = previous[previous.length - 1];
      return last && last.w === weight && last.b === bias ? previous : [{ w: weight, b: bias }];
    });
  }, [weight, bias]);

  const assign = (termId: string, ringId: string) => {
    const next = placements.filter((entry) => !entry.startsWith(`${termId}:`));
    next.push(`${termId}:${ringId}`);
    const match = TERMS.find((item) => item.id === termId);
    setState({ placements: next, term: termId, destination: ringId });
    if (!match) return;
    narrate(
      ringId === match.ring
        ? `${match.label} sits in ${RINGS.find((item) => item.id === match.ring)?.label}.`
        : `${match.label} belongs in ${RINGS.find((item) => item.id === match.ring)?.label}, not ${RINGS.find((item) => item.id === ringId)?.label}.`,
    );
  };

  const assignSignal = (id: string, signalId: string) => {
    const next = signalPlacements.filter((entry) => !entry.startsWith(`${id}:`));
    next.push(`${id}:${signalId}`);
    const match = scenarioById(id);
    setState({ signalPlacements: next, scenario: id, signal: signalId });
    narrate(
      signalId === match.signal
        ? `${match.short} is trained by ${signalById(match.signal).label} learning.`
        : `${match.short} is ${signalById(match.signal).label} learning, not ${signalById(signalId).label}.`,
    );
  };

  const stepDownhill = () => {
    const { w: nextW, b: nextB } = downhill(weight, bias);
    setTrail((previous) => [...previous.slice(-30), { w: nextW, b: nextB }]);
    setState({ weight: nextW, bias: nextB });
    narrate(
      `One gradient step on this MSE moved (w, b) toward smaller L. New loss will recompute from ŷ = ${nextW.toFixed(2)} x + ${nextB.toFixed(2)}.`,
    );
  };

  return (
    <div className="gw-shell">
    <div className="tg-lab tg-lab--hero gw-lab wai-lab">
      <LabSurface label="Nested methods" className="nested-methods-card">
        <SurfaceHeading
          kicker="A common stack: AI ⊃ ML ⊃ DL ⊃ neural generative models ⊃ LLMs"
          title="Five rings, one nesting"
          aside={<span className="tg-badge">{selectedRing.label} selected</span>}
        />
        <div className="nested-method-control">
          <SegmentedControl
            label="Inspected method"
            value={ring}
            options={RING_OPTIONS}
            onChange={(value) => {
              setState({ ring: value });
              narrate(`${RINGS.find((item) => item.id === value)?.label ?? value} is now the inspected ring.`);
            }}
          />
        </div>
        <div className="ai-ring-layout">
          <svg
            className="nested-rings"
            viewBox="0 0 320 320"
            role="img"
            aria-label={`Nested circles. ${selectedRing.label} is selected. ${selectedRing.learns} A marker for an n-gram sits inside the ML circle and outside the deep-learning circle, because an n-gram is generative without being deep.`}
          >
            {RINGS.map((item) => {
              const active = item.id === ring;
              return (
                <g key={item.id}>
                  <circle
                    cx="160"
                    cy="160"
                    r={item.radius}
                    className={active ? "is-current" : ""}
                    onClick={() => setState({ ring: item.id })}
                  />
                  <text x="160" y={160 - item.radius + 16} textAnchor="middle">
                    {item.label}
                  </text>
                </g>
              );
            })}
            <g className="ai-ring-marker">
              <circle cx="138" cy="263" r="4" />
              <text x="146" y="267">
                n-gram
              </text>
            </g>
          </svg>
          <ol className="ai-ring-key" aria-label="What each ring claims">
            {RINGS.map((item) => (
              <li key={item.id} className={item.id === ring ? "is-current" : ""}>
                <button type="button" onClick={() => setState({ ring: item.id })}>
                  <strong>{item.label}</strong>
                  <span>{item.math}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
        <p className="lab-note">
          <strong>{selectedRing.label} learns:</strong> {selectedRing.learns} The rings are the
          common modern stack, not a law: every LLM is generative, but a GAN is generative without
          being an LLM.
        </p>
        <p className="lab-note">
          <strong>What the drawing cannot show:</strong> generative is a property that crosses the deep-learning
          boundary. The Gen ring holds the neural generative models only. An n-gram, which counts word pairs and
          samples from the counts, is generative without being deep, so it is marked inside ML and outside DL
          rather than inside Gen.
        </p>
      </LabSurface>

      <LabSurface label="Learning problem" className="learning-problem-card gw-wide">
        <SurfaceHeading
          kicker="Approximate y with f_θ(x) by searching θ"
          title="The objects: hypothesis, loss, a downhill step"
        />
        <StageFlow
          label="Learning object"
          stages={STAGES.map((item, index) => ({ id: item.id, name: item.name, detail: STAGE_DETAILS[index] }))}
          current={stage}
          onSelect={(index) => setState({ stage: index })}
        />
        <div className="tg-stage-detail">
          <strong>{selectedStage.title}</strong>
          <p>{selectedStage.body}</p>
        </div>
        <div className="learning-fit-control">
          <div className="wai-sliders">
            <RangeControl
              label="Weight w"
              min={0}
              max={2}
              step={0.05}
              value={weight}
              format={(value) => value.toFixed(2)}
              onChange={(value) => setState({ weight: value })}
            />
            <RangeControl
              label="Bias b"
              min={0}
              max={2.4}
              step={0.05}
              value={bias}
              format={(value) => value.toFixed(2)}
              onChange={(value) => setState({ bias: value })}
            />
          </div>
          <button type="button" className="primary-action" onClick={stepDownhill}>
            Take a downhill step
          </button>
        </div>
        <div className="wai-twin">
          <figure>
            <figcaption>Data space · one line, three squared misses</figcaption>
            <DataPlot w={weight} b={bias} />
          </figure>
          <figure>
            <figcaption>Parameter space · every point is a line</figcaption>
            <ParamPlot w={weight} b={bias} trail={trail} />
          </figure>
        </div>
        <ul className="gw-legend wai-legend" aria-hidden="true">
          <li><i className="is-forward" />current line / current (w, b)</li>
          <li><i className="is-loss" />residual ŷ − y, with its square</li>
          <li><i className="is-ink" />contours of equal L</li>
          <li><i className="is-gradient" />next step, −0.12 ∇L</li>
          <li><i className="is-ink is-dotted" />steps taken</li>
          <li><b className="wai-x">×</b>least-squares bottom ({best.w.toFixed(2)}, {best.b.toFixed(2)})</li>
        </ul>
        <table className="tg-board fit-table">
          <thead>
            <tr>
              <th scope="col">x</th>
              <th scope="col">y</th>
              <th scope="col">ŷ = wx+b</th>
              <th scope="col">ŷ − y</th>
              <th scope="col">(ŷ − y)²</th>
            </tr>
          </thead>
          <tbody>
            {fitted.map((point) => (
              <tr key={`${point.x}-row`}>
                <td>{point.x.toFixed(1)}</td>
                <td>{point.y.toFixed(1)}</td>
                <td>{point.prediction.toFixed(2)}</td>
                <td>{point.residual.toFixed(2)}</td>
                <td>{point.squared.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric label="L  MSE" value={loss.toFixed(3)} tone="loss" />
          <Metric label="∂L/∂w" value={gradW.toFixed(3)} tone="gradient" />
          <Metric label="∂L/∂b" value={gradB.toFixed(3)} tone="gradient" />
          <Metric label="Minimum L" value={best.loss.toFixed(4)} />
        </div>
        <FormulaWithValues
          label="Hypothesis and empirical loss"
          expression={`ŷ = ${weight.toFixed(2)} x + ${bias.toFixed(2)}   ·   L = (${fitted
            .map((point) => point.squared.toFixed(3))
            .join(" + ")}) / 3`}
          result={loss.toFixed(3)}
          detail={`The solid line is the current hypothesis. Vertical ticks are residuals and the shaded squares are their squares. Downhill moves (w, b) by −${STEP_SIZE} × (∂L/∂w, ∂L/∂b) of this same MSE, not a metaphor.`}
          tone="loss"
        />
        <p className="lab-note">
          You are searching a two-number parameter vector: the right-hand map is the whole search space.
          Consciousness is not an extra variable in that search. Even the bottom of the bowl, L ={" "}
          {best.loss.toFixed(4)}, is a fit to three authored points, not a test of a new x.
        </p>
      </LabSurface>

      <LabSurface label="What the word hides" className="distinction-card">
        <SurfaceHeading
          kicker="Same word, five different cuts"
          title="Family, scope, job, memory, and what is not claimed"
          aside={<span className="tg-badge">{selectedDistinction.label}</span>}
        />
        <div className="distinction-control">
          <SegmentedControl
            label="Distinction"
            value={distinction}
            options={DISTINCTION_OPTIONS}
            onChange={(value) => {
              setState({ distinction: value });
              narrate(`Distinction set to ${value}.`);
            }}
          />
        </div>
        <p className="lab-note">
          <strong>{selectedDistinction.title}</strong>
        </p>
        <div className="mixing-grid distinction-grid">
          {selectedDistinction.tiles.map((tile) => (
            <div className="mixing-tile" key={tile.title}>
              <strong>{tile.title}</strong>
              <small>{tile.body}</small>
            </div>
          ))}
        </div>
        <p className="lab-note">{selectedDistinction.note}</p>
      </LabSurface>

      <LabSurface label="Era timeline" className="era-timeline-card">
        <SurfaceHeading
          kicker={`${selectedEra.year} · ${selectedEra.title}`}
          title="What changed, and why it stalled"
        />
        <RangeControl
          label="Era"
          min={0}
          max={ERAS.length - 1}
          step={1}
          value={era}
          format={(value) => `${ERAS[value]?.year ?? ""} ${ERAS[value]?.title ?? ""}`}
          onChange={(value) => setState({ eraYear: ERAS[Math.min(ERAS.length - 1, Math.max(0, Math.round(value)))].year })}
        />
        <div className="metric-row">
          <Metric label="Year" value={`${selectedEra.year}`} />
          <Metric label="Key idea" value={selectedEra.title} tone="forward" />
        </div>
        <p className="lab-note">
          <strong>Idea:</strong> {selectedEra.idea}
        </p>
        <p className="lab-note">
          <strong>Why it stalled:</strong> {selectedEra.stalled}
        </p>
      </LabSurface>

      <LabSurface label="Sort leftover terms" className="term-sort-card">
        <SurfaceHeading
          kicker={`${correctCount} of ${TERMS.length} in the tightest ring`}
          title="Place each leftover in its tightest circle"
          aside={
            <span className={`tg-badge ${currentVerdict === "corrected" ? "is-warning" : ""}`.trim()}>
              {currentVerdict}
            </span>
          }
        />
        <div
          className="term-listbox"
          role="listbox"
          aria-label="Leftover terms"
          tabIndex={0}
          onKeyDown={(event) => {
            const index = TERMS.findIndex((item) => item.id === term);
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setState({ term: TERMS[(index + 1) % TERMS.length].id });
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              setState({ term: TERMS[(index - 1 + TERMS.length) % TERMS.length].id });
            }
            if (event.key === "Enter") assign(selectedTerm.id, destination);
          }}
        >
          {TERMS.map((item) => {
            const guess = placed[item.id];
            return (
              <button
                key={item.id}
                type="button"
                role="option"
                aria-selected={item.id === term}
                draggable
                className={item.id === term ? "is-current" : ""}
                onClick={() => setState({ term: item.id })}
                onDragStart={(event) => {
                  event.dataTransfer.setData("text/plain", item.id);
                  setState({ term: item.id });
                }}
              >
                <strong>{item.label}</strong>
                <small>
                  {guess
                    ? guess === item.ring
                      ? `Placed in ${RINGS.find((ringItem) => ringItem.id === item.ring)?.label}`
                      : `Corrected to ${RINGS.find((ringItem) => ringItem.id === item.ring)?.label}`
                    : "Unplaced"}
                </small>
              </button>
            );
          })}
        </div>
        <div
          className="place-ring-control"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            const dropped = event.dataTransfer.getData("text/plain");
            if (TERMS.some((item) => item.id === dropped)) assign(dropped, destination);
          }}
        >
          <SegmentedControl
            label="Place in ring"
            value={destination}
            options={RING_OPTIONS}
            onChange={(value) => setState({ destination: value })}
          />
          <button
            type="button"
            className="primary-action"
            onClick={() => assign(selectedTerm.id, destination)}
          >
            Place “{selectedTerm.label}”
          </button>
        </div>
        <div className="metric-row">
          <Metric label="Placed" value={`${placedCount} / ${TERMS.length}`} />
          <Metric
            label="Tightest-ring matches"
            value={`${correctCount}`}
            tone={correctCount === TERMS.length ? "forward" : "loss"}
          />
          <Metric
            label="Nesting depth"
            value={`${ringIndex(selectedTerm.ring) + 1} of ${RINGS.length}`}
          />
        </div>
        <p className="lab-note">
          {currentPlacement === undefined
            ? `Select a leftover, choose a ring, then press Place. Drag is optional; the listbox and Place in ring control are the keyboard path.`
            : currentPlacement === selectedTerm.ring
              ? selectedTerm.why
              : `You placed it in ${RINGS.find((item) => item.id === currentPlacement)?.label}. ${selectedTerm.why}`}
        </p>
      </LabSurface>

      <LabSurface label="What is the learning signal" className="signal-sort-card gw-wide">
        <SurfaceHeading
          kicker={`${signalScore.matched} of ${signalScore.total} sorted under the signal that trains them`}
          title="Where does the training target come from?"
          aside={
            <span className={`tg-badge ${signalVerdict === "corrected" ? "is-warning" : ""}`.trim()}>
              {signalVerdict}
            </span>
          }
        />
        <div className="mixing-grid distinction-grid signal-tiles">
          {SIGNALS.map((item) => (
            <div className="mixing-tile" key={item.id}>
              <strong>
                {item.label} · {item.source}
              </strong>
              <small>{item.body}</small>
            </div>
          ))}
        </div>
        <div className="signal-sort">
          <div
            className="term-listbox"
            role="listbox"
            aria-label="Learning scenarios"
            tabIndex={0}
            onKeyDown={(event) => {
              const index = SCENARIOS.findIndex((item) => item.id === scenarioId);
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setState({ scenario: SCENARIOS[(index + 1) % SCENARIOS.length].id });
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setState({ scenario: SCENARIOS[(index - 1 + SCENARIOS.length) % SCENARIOS.length].id });
              }
              if (event.key === "Enter") assignSignal(selectedScenario.id, signalChoice);
            }}
          >
            {SCENARIOS.map((item) => {
              const guess = signalPlaced[item.id];
              return (
                <button
                  key={item.id}
                  type="button"
                  role="option"
                  aria-selected={item.id === scenarioId}
                  draggable
                  className={item.id === scenarioId ? "is-current" : ""}
                  onClick={() => setState({ scenario: item.id })}
                  onDragStart={(event) => {
                    event.dataTransfer.setData("text/plain", item.id);
                    setState({ scenario: item.id });
                  }}
                >
                  <strong>{item.short}</strong>
                  <span>{item.text}</span>
                  <small>
                    {guess
                      ? guess === item.signal
                        ? `Placed under ${signalById(item.signal).label}`
                        : `Corrected to ${signalById(item.signal).label}`
                      : "Unplaced"}
                  </small>
                </button>
              );
            })}
          </div>
          <div
            className="place-signal-control"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              const dropped = event.dataTransfer.getData("text/plain");
              if (SCENARIO_IDS.includes(dropped)) assignSignal(dropped, signalChoice);
            }}
          >
            <SegmentedControl
              label="Learning signal"
              value={signalChoice}
              options={SIGNAL_OPTIONS}
              onChange={(value) => setState({ signal: value })}
            />
            <button
              type="button"
              className="primary-action"
              onClick={() => assignSignal(selectedScenario.id, signalChoice)}
            >
              Place “{selectedScenario.short}”
            </button>
          </div>
        </div>
        <div className="metric-row">
          <Metric label="Placed" value={`${signalScore.placed} / ${signalScore.total}`} />
          <Metric
            label="Signal matches"
            value={`${signalScore.matched}`}
            tone={signalScore.matched === signalScore.total ? "forward" : "loss"}
          />
          <Metric
            label="Output type"
            value={`${OUTPUT_KINDS[selectedScenario.output].split(" · ")[0]} · ${selectedScenario.outputNote}`}
          />
        </div>
        <p className="lab-note">
          {currentSignalPlacement === undefined
            ? "Select a scenario with the arrow keys or a click, choose a signal, then press Place. Drag is optional; the listbox and the Place button are the keyboard path."
            : currentSignalPlacement === selectedScenario.signal
              ? selectedScenario.why
              : `You placed it under ${signalById(currentSignalPlacement).label}. It belongs under ${signalById(selectedScenario.signal).label}. ${selectedScenario.why}`}
        </p>
        <p className="lab-note">
          <strong>Output type is a separate question.</strong> {OUTPUT_KINDS[selectedScenario.output]}. The signal says where the target
          comes from; classification and regression say what kind of answer the model gives. Real projects also blend
          signals: semi-supervised learning mixes a few labeled examples with many unlabeled ones, and a chat assistant is
          pretrained self-supervised, tuned on human-written answers, then trained on human-preference rewards.
        </p>
      </LabSurface>
    </div>
    </div>
  );
}
