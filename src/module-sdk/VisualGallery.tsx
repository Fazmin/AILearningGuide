import { useState, type CSSProperties } from "react";
import {
  smokeTestTeachingModels,
  type ModelSmokeResult,
} from "@app/model-runtime/smoke";
import {
  ArcDiagram,
  BarList,
  FormulaWithValues,
  Heatmap,
  LabSurface,
  LineChart,
  StageFlow,
  StepThroughController,
  SurfaceHeading,
  VectorChip,
} from "./visualizations";
import { TINY_CORPORA, trainTinyModel, UNIFORM_CROSS_ENTROPY } from "./tiny-lm";

const trainingStages = [
  { id: "batch", name: "Batch", detail: "sample pairs" },
  { id: "forward", name: "Forward", detail: "logits" },
  { id: "loss", name: "Loss", detail: "cross-entropy" },
  { id: "backward", name: "Backward", detail: "gradients" },
  { id: "step", name: "Step", detail: "update weights" },
];
const galleryRun = trainTinyModel({
  text: TINY_CORPORA.harbor.text,
  epochs: 14,
  seed: 3,
});
const galleryClipped = trainTinyModel({
  text: TINY_CORPORA.harbor.text,
  epochs: 14,
  learningRate: 4,
  clipNorm: 0.6,
  seed: 3,
});
const tokens = ["The", "animal", "crossed", "because", "it", "was", "tired"];
const weights = [0.04, 0.54, 0.08, 0.06, 0.12, 0.07, 0.09];
const matrix = tokens.map((_, row) => {
  const values = tokens.map((__, column) =>
    Math.exp(-Math.abs(row - column) * 0.8) +
    (row === 4 && column === 1 ? 2.5 : 0),
  );
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
});

export function VisualGallery() {
  const [step, setStep] = useState(2);
  const [query, setQuery] = useState(4);
  const [selected, setSelected] = useState({ row: 4, column: 1 });
  const [modelStatus, setModelStatus] = useState<
    "idle" | "running" | "passed" | "failed"
  >("idle");
  const [modelResults, setModelResults] = useState<ModelSmokeResult[]>([]);
  const [modelError, setModelError] = useState("");

  return (
    <main
      className="visual-gallery"
      style={{ "--module-accent": "var(--attention)" } as CSSProperties}
    >
      <header>
        <span>Module SDK</span>
        <h1>Visualization instruments</h1>
        <p>
          The shared, keyboard-operable vocabulary used by the anchor modules.
        </p>
      </header>

      <LabSurface label="Vector and formula examples">
        <SurfaceHeading kicker="Live values" title="Vectors and formulas" />
        <div className="visual-gallery__row">
          <VectorChip label="Query" values={[0.24, -0.71, 0.92, 0.18]} />
          <VectorChip
            label="Gradient"
            values={[-0.11, 0.08, -0.34, 0.57]}
            tone="gradient"
          />
        </div>
        <FormulaWithValues
          label="Scaled score"
          expression="Q · K ÷ √64"
          result="0.438"
          detail="Values update without replacing the mathematical structure."
        />
      </LabSurface>

      <LabSurface label="Teaching model worker diagnostics">
        <SurfaceHeading
          kicker="ONNX Runtime Web"
          title="Worker-isolated asset smoke test"
        />
        <button
          type="button"
          className="primary-action visual-gallery__model-test"
          disabled={modelStatus === "running"}
          onClick={() => {
            setModelStatus("running");
            setModelError("");
            void smokeTestTeachingModels()
              .then((results) => {
                setModelResults(results);
                setModelStatus("passed");
              })
              .catch((error) => {
                setModelError(
                  error instanceof Error ? error.message : String(error),
                );
                setModelStatus("failed");
              });
          }}
        >
          {modelStatus === "running"
            ? "Running six models…"
            : "Run all model assets"}
        </button>
        <div
          className={`visual-gallery__model-results is-${modelStatus}`}
          role="status"
        >
          {modelStatus === "idle" && (
            <p>Loads each exported ONNX graph and checks finite outputs.</p>
          )}
          {modelStatus === "failed" && <p>{modelError}</p>}
          {modelResults.map((result) => (
            <div key={result.id}>
              <strong>{result.id}</strong>
              <span>{result.provider}</span>
              <code>{result.milliseconds.toFixed(0)} ms</code>
              <small>{result.outputs.join(", ")}</small>
            </div>
          ))}
        </div>
      </LabSurface>

      <LabSurface label="Step controller example">
        <SurfaceHeading kicker="Sequence" title="Step-through controller" />
        <StepThroughController
          label="Attention stages"
          steps={["Query", "Keys", "Score", "Softmax", "Mix"]}
          current={step}
          onChange={setStep}
        />
      </LabSurface>

      <LabSurface label="Stage flow example">
        <SurfaceHeading kicker="Pipeline" title="Training-loop stage flow" />
        <StageFlow
          label="Training loop stages"
          stages={trainingStages}
          current={step % trainingStages.length}
          onSelect={(index) => setStep(index)}
          loops
        />
      </LabSurface>

      <LabSurface label="Line chart example">
        <SurfaceHeading
          kicker="Measured runs"
          title="Multi-series line chart"
          aside={<span>{galleryRun.steps} real optimizer steps</span>}
        />
        <LineChart
          label="Training loss for two learning rates"
          xLabel="step"
          yLabel="loss (nats/token)"
          yDomain={[0, UNIFORM_CROSS_ENTROPY]}
          series={[
            {
              id: "baseline",
              name: "lr 0.6",
              tone: "forward",
              points: galleryRun.history.map((point) => ({ x: point.step, y: point.loss })),
            },
            {
              id: "clipped",
              name: "lr 4.0, clipped",
              tone: "loss",
              dash: "dashed",
              points: galleryClipped.history.map((point) => ({ x: point.step, y: point.loss })),
            },
          ]}
          footnote="Both curves come from the bundled character model training in this tab."
        />
      </LabSurface>

      <LabSurface label="Bar list example">
        <SurfaceHeading kicker="Ranked values" title="Bar list" />
        <BarList
          label="Next-token probabilities"
          items={[
            { id: "the", label: "the", value: 0.42, display: "42%" },
            { id: "fog", label: "fog", value: 0.27, display: "27%", tone: "attention" },
            { id: "harbor", label: "harbor", value: 0.19, display: "19%", tone: "gradient" },
            { id: "other", label: "other", value: 0.12, display: "12%", tone: "muted" },
          ]}
          max={0.42}
        />
      </LabSurface>

      <LabSurface label="Arc diagram example">
        <SurfaceHeading kicker="Routing" title="Arc diagram" />
        <div className="visual-gallery__scroll">
          <ArcDiagram
            tokens={tokens}
            weights={weights}
            queryIndex={query}
            inspectedIndex={1}
            label="Attention route example"
            onQueryChange={setQuery}
          />
        </div>
      </LabSurface>

      <LabSurface label="Heatmap example">
        <SurfaceHeading kicker="Matrix" title="Accessible heatmap" />
        <Heatmap
          rows={tokens}
          columns={tokens}
          values={matrix}
          selected={selected}
          label="Example attention matrix"
          onSelect={(row, column) => setSelected({ row, column })}
        />
      </LabSurface>
    </main>
  );
}
