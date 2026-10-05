import { useEffect, useMemo, useRef, useState } from "react";
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
import shippedSchedule from "./assets/mnist-diffusion-schedule.json";
import metadata from "./assets/mnist-models.metadata.json";
import {
  backgroundShare,
  forwardNoise,
  meanSquaredError,
  noiseErrorByStep,
  normalStream,
  reverseStart,
  runIdealReverse,
  runNetworkReverse,
  scheduleFromAlphaBar,
  stepCoefficients,
  STEPS,
  timestepForCall,
  toSigned,
  type StartKind,
  type Trajectory,
} from "./diffusion";
import { DrawingPad, LatentMap } from "./LatentWidgets";
import { decodeLatents, encodeImage, predictNoise, type Provider } from "./models";
import { PixelImage, PixelMosaic } from "./PixelImage";
import { useAsyncValue, type AsyncStatus } from "./useAsyncValue";
import {
  binaryCrossEntropy,
  gaussianKl,
  inkTotal,
  KL_WEIGHT,
  latentGrid,
  MAP_GRID,
  MAP_MAX,
  MAP_MIN,
  PRESET_LABELS,
  PRESET_STROKES,
  priorLogDensity,
  rasterizeStrokes,
  referenceLatents,
  type Stroke,
} from "./vae";

const FILM_CALLS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
const FORWARD_STEPS = [0, 19, 39, 59, 79, 99];
const SEED_MAX = 24;
const schedule = scheduleFromAlphaBar(shippedSchedule.alpha_cumulative);
const LAST = STEPS - 1;
const PRESETS = Object.keys(PRESET_STROKES);
const mosaicLatents = latentGrid(MAP_GRID, MAP_MIN, MAP_MAX);
const referenceGrid = referenceLatents();
const fallbackReferences = PRESETS.map((name) => toSigned(rasterizeStrokes(PRESET_STROKES[name])));
const fallbackDigit = rasterizeStrokes(PRESET_STROKES.zero);
const diffusionHistory = metadata.diffusion.history;
const finalNoiseMse = diffusionHistory[diffusionHistory.length - 1]?.noise_mse ?? 0;

const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? (state[key] as number) : fallback;
const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? (state[key] as string) : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const fmt = (value: number, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const signed = (value: number, digits = 2) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
/** Fixed decimals with a typographic minus sign. */
const num = (value: number, digits = 2) => value.toFixed(digits).replace("-", "−");

type Mode = "diffusion" | "vae";
type Denoiser = "ideal" | "network";

function StatusChip({ status, text, error }: { status: AsyncStatus; text: string; error?: string }) {
  return (
    <span className={`model-runtime-status model-runtime-status--${status}`} role="status" title={error || undefined}>
      <i />
      {text}
    </span>
  );
}

const providerText = (provider: Provider | undefined) => `ONNX · ${(provider ?? "wasm").toUpperCase()}`;

export default function Explore({ state, setState, currentStep, narrate }: ModuleContext) {
  const stepMode: Mode = currentStep >= 2 ? "vae" : "diffusion";
  const storedMode = asString(state, "mode", "");
  const mode: Mode = storedMode === "vae" || storedMode === "diffusion" ? storedMode : stepMode;
  const calls = clamp(Math.round(asNumber(state, "step", 62)), 0, STEPS);
  const z1 = clamp(asNumber(state, "latentX", -2), MAP_MIN, MAP_MAX);
  const z2 = clamp(asNumber(state, "latentY", -0.8), MAP_MIN, MAP_MAX);
  const seed = clamp(Math.round(asNumber(state, "seed", 7)), 1, SEED_MAX);
  const denoiser: Denoiser = asString(state, "denoiser", "network") === "ideal" ? "ideal" : "network";
  const start: StartKind = asString(state, "start", "noise") === "digit" ? "digit" : "noise";
  const sample = asString(state, "sample", "three");

  // Follow the lesson's step into the matching model, but only when the step changes.
  const previousStep = useRef(currentStep);
  useEffect(() => {
    if (previousStep.current === currentStep) return;
    previousStep.current = currentStep;
    if (mode !== stepMode) setState({ mode: stepMode });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  // ---- Real decoder: the latent mosaic, the current digit, and the reference set.
  const mosaic = useAsyncValue(mode === "vae" ? "mosaic" : null, () => decodeLatents(mosaicLatents));
  const latentKey = `${z1.toFixed(2)},${z2.toFixed(2)}`;
  const digit = useAsyncValue(
    latentKey,
    async () => {
      const decoded = await decodeLatents([z1, z2]);
      return { image: decoded.images[0], key: latentKey, provider: decoded.provider };
    },
    40,
  );
  const referenceRun = useAsyncValue(mode === "diffusion" ? "references" : null, () => decodeLatents(referenceGrid));

  const x0Image = digit.value?.image ?? (digit.status === "error" ? fallbackDigit : null);
  const x0Key = digit.value?.key ?? (digit.status === "error" ? "fallback" : "pending");
  const x0Signed = useMemo(() => (x0Image ? toSigned(x0Image) : null), [x0Image]);
  const references = useMemo(
    () =>
      referenceRun.value
        ? referenceRun.value.images.map(toSigned)
        : referenceRun.status === "error"
          ? fallbackReferences
          : null,
    [referenceRun.value, referenceRun.status],
  );
  const referenceTiles = useMemo(
    () => (referenceRun.value ? referenceRun.value.images : references ? references.map((image) => Float32Array.from(image, (v) => (v + 1) / 2)) : null),
    [referenceRun.value, references],
  );

  // ---- Reverse process: identical seeded start and fresh noise for both denoisers.
  const startState = useMemo(
    () => (x0Signed || start === "noise" ? reverseStart(seed, start, x0Signed, schedule) : null),
    [seed, start, x0Signed],
  );
  const idealRun = useMemo<Trajectory | null>(
    () =>
      mode === "diffusion" && denoiser === "ideal" && startState && references
        ? runIdealReverse(startState, references, schedule)
        : null,
    [mode, denoiser, startState, references],
  );
  const networkKey =
    mode === "diffusion" && denoiser === "network" && startState ? `${seed}|${start}|${start === "digit" ? x0Key : "-"}` : null;
  const networkRun = useAsyncValue(networkKey, async (token) => {
    if (!startState) throw new Error("No start image yet.");
    let provider: Provider = "wasm";
    const trajectory = await runNetworkReverse(
      startState,
      async (x, t) => {
        const result = await predictNoise([x], [t]);
        provider = result.provider;
        return result.noise[0];
      },
      schedule,
      (call) => {
        if (token.cancelled) throw new Error("cancelled");
        if (call % 5 === 0 && call > 0) token.report(call / STEPS);
      },
    );
    return { trajectory, provider };
  });
  const trajectory = denoiser === "ideal" ? idealRun : networkRun.value?.trajectory ?? null;

  // ---- The network's noise-prediction error at every t, on this x0 with this seed's noise.
  const errorKey = mode === "diffusion" && x0Signed ? `${x0Key}|${seed}` : null;
  const errorRun = useAsyncValue(errorKey, async () => {
    if (!x0Signed) throw new Error("No clean image yet.");
    const epsilon = normalStream(seed + 1000).fill();
    let provider: Provider = "wasm";
    const curve = await noiseErrorByStep(x0Signed, epsilon, schedule, async (images, timesteps) => {
      const result = await predictNoise(images, timesteps);
      provider = result.provider;
      return result.noise;
    });
    return { ...curve, provider };
  });

  // ---- Forward strip: the closed form q(x_t | x0) with one noise draw.
  const forwardStrip = useMemo(() => {
    if (!x0Signed) return null;
    const epsilon = normalStream(seed + 1000).fill();
    return FORWARD_STEPS.map((t) => forwardNoise(x0Signed, epsilon, schedule.alphaBar[t]));
  }, [x0Signed, seed]);

  // ---- Encoder: a drawing (or preset) through the real encoder and decoder.
  const [strokes, setStrokes] = useState<Stroke[]>(() =>
    PRESET_STROKES[sample] ? [...PRESET_STROKES[sample]] : [],
  );
  const [revision, setRevision] = useState(0);
  const loadedSample = useRef(sample);
  useEffect(() => {
    // A reset or restored snapshot can change the preset from outside the pad.
    if (loadedSample.current === sample) return;
    loadedSample.current = sample;
    if (PRESET_STROKES[sample]) {
      setStrokes([...PRESET_STROKES[sample]]);
      setRevision((value) => value + 1);
    }
  }, [sample]);
  const drawing = useMemo(() => rasterizeStrokes(strokes), [strokes]);
  const ink = inkTotal(drawing);
  const encoded = useAsyncValue(mode === "vae" && ink > 1 ? `${sample}|${revision}|${seed}` : null, async () => {
    const posterior = await encodeImage(drawing);
    const sigma: [number, number] = [
      Math.exp(0.5 * posterior.logVariance[0]),
      Math.exp(0.5 * posterior.logVariance[1]),
    ];
    const noise = normalStream(seed + 2000).fill(2);
    const sampled: [number, number] = [
      posterior.mean[0] + sigma[0] * noise[0],
      posterior.mean[1] + sigma[1] * noise[1],
    ];
    const decoded = await decodeLatents([...posterior.mean, ...sampled]);
    return {
      ...posterior,
      sigma,
      sampled,
      meanImage: decoded.images[0],
      sampleImage: decoded.images[1],
      bce: binaryCrossEntropy(drawing, decoded.images[0]),
      bceSampled: binaryCrossEntropy(drawing, decoded.images[1]),
      kl: gaussianKl(posterior.mean, posterior.logVariance),
      provider: decoded.provider,
    };
  });
  const loadPreset = (name: string) => {
    loadedSample.current = name;
    setStrokes([...(PRESET_STROKES[name] ?? [])]);
    setRevision((value) => value + 1);
    setState({ sample: name });
    narrate(`Loaded a hand-drawn ${PRESET_LABELS[name] ?? name} and sent it to the encoder.`);
  };

  // ---- Readouts for the reverse surface.
  const nextCall = Math.min(calls, STEPS - 1);
  const nextT = timestepForCall(nextCall);
  const coefficients = stepCoefficients(schedule, nextT);
  const currentLevel = calls < STEPS ? timestepForCall(calls) : -1;
  const alphaBarNow = currentLevel >= 0 ? schedule.alphaBar[currentLevel] : 1;
  const stateAt = trajectory?.states[calls] ?? null;
  const cleanAt = calls < STEPS ? trajectory?.cleanPredictions[calls] ?? null : null;
  const noiseAt = calls < STEPS ? trajectory?.noisePredictions[calls] ?? null : null;
  const weightsAt = denoiser === "ideal" && idealRun?.weights ? idealRun.weights[nextCall] : null;
  const topReference = weightsAt ? Array.from(weightsAt).reduce((best, value, index, all) => (value > all[best] ? index : best), 0) : -1;
  const finalSample = trajectory?.states[STEPS] ?? null;
  const backgroundAt = stateAt ? backgroundShare(stateAt) : NaN;
  const cleanBackground = x0Signed ? backgroundShare(x0Signed) : NaN;
  const nearestReference =
    finalSample && references
      ? references.reduce(
          (best, reference, index) => {
            const distance = meanSquaredError(finalSample, reference);
            return distance < best.distance ? { index, distance } : best;
          },
          { index: -1, distance: Infinity },
        )
      : null;

  const reverseStatus: { status: AsyncStatus; text: string; error?: string } =
    denoiser === "network"
      ? networkRun.status === "loading"
        ? { status: "loading", text: `Denoiser call ${Math.round(networkRun.progress * STEPS)} / ${STEPS}` }
        : networkRun.status === "error"
          ? { status: "error", text: "Denoiser unavailable", error: networkRun.error }
          : networkRun.status === "ready"
            ? { status: "ready", text: providerText(networkRun.value?.provider) }
            : { status: "idle", text: "Waiting for x₀" }
      : referenceRun.status === "error"
        ? { status: "error", text: "Decoder unavailable · fallback set", error: referenceRun.error }
        : references
          ? { status: "ready", text: "Exact math · no network" }
          : { status: "loading", text: "Decoding references…" };

  const x0Caption =
    digit.status === "error" && !digit.value
      ? "Fallback x₀: a hand-drawn 0 (the VAE decoder did not load)"
      : `x₀ = VAE decoder output at z = (${num(z1)}, ${num(z2)})`;

  const scheduleSeries = [
    {
      id: "signal",
      name: "Signal √ᾱ_t",
      tone: "forward" as const,
      points: Array.from(schedule.alphaBar, (value, t) => ({ x: t, y: Math.sqrt(value) })),
    },
    {
      id: "noise",
      name: "Noise √(1−ᾱ_t)",
      tone: "gradient" as const,
      dash: "dashed" as const,
      points: Array.from(schedule.alphaBar, (value, t) => ({ x: t, y: Math.sqrt(1 - value) })),
    },
  ];
  const errorSeries = errorRun.value
    ? [
        {
          id: "network",
          name: "Trained network",
          tone: "loss" as const,
          points: errorRun.value.network.map((value, t) => ({ x: t, y: value })),
        },
        {
          id: "zero",
          name: "Predict ε̂ = 0",
          tone: "muted" as const,
          dash: "dotted" as const,
          points: [
            { x: 0, y: errorRun.value.baseline },
            { x: LAST, y: errorRun.value.baseline },
          ],
        },
      ]
    : [];
  const markerT = currentLevel >= 0 ? currentLevel : 0;
  const nearestForward = FORWARD_STEPS.reduce(
    (best, t, index) => (Math.abs(t - markerT) < Math.abs(FORWARD_STEPS[best] - markerT) ? index : best),
    0,
  );

  const modeBar = (
    <div className="dv-modebar">
      <SegmentedControl
        label="Model"
        value={mode}
        options={[
          { value: "diffusion", label: "Diffusion" },
          { value: "vae", label: "VAE" },
        ]}
        onChange={(value) => {
          setState({ mode: value });
          narrate(value === "vae" ? "VAE view: one decoder call per image." : "Diffusion view: one denoiser call per step.");
        }}
      />
      <p>
        {mode === "diffusion"
          ? "Diffusion makes an image with 100 sequential denoiser calls. Every image below is computed from the shipped schedule."
          : "The VAE makes an image with one decoder call on two numbers. Every image below is the shipped decoder's output."}
      </p>
    </div>
  );

  if (mode === "vae") {
    const logDensity = priorLogDensity([z1, z2]);
    const radius = Math.hypot(z1, z2);
    const encodedValue = encoded.value;
    const encodedStatus: { status: AsyncStatus; text: string; error?: string } =
      ink <= 1
        ? { status: "idle", text: "Draw a digit" }
        : encoded.status === "error"
          ? { status: "error", text: "Encoder unavailable", error: encoded.error }
          : encoded.status === "ready"
            ? { status: "ready", text: providerText(encodedValue?.provider) }
            : { status: "loading", text: "Encoding…" };
    const mapStatus: { status: AsyncStatus; text: string; error?: string } =
      mosaic.status === "error" || digit.status === "error"
        ? { status: "error", text: "Decoder unavailable", error: mosaic.error ?? digit.error }
        : mosaic.status === "ready" && digit.value
          ? { status: "ready", text: providerText(mosaic.value?.provider) }
          : { status: "loading", text: "Decoding…" };

    return (
      <div className="dv-lab">
        {modeBar}
        <LabSurface label="Latent map" className="dv-surface">
          <SurfaceHeading
            kicker={`Decoder · 2 numbers in, 784 pixels out`}
            title="Drag the point; one decoder call draws the digit"
            aside={<StatusChip {...mapStatus} />}
          />
          <div className="dv-map-layout">
            <LatentMap
              tiles={mosaic.value?.images ?? null}
              columns={MAP_GRID}
              minimum={MAP_MIN}
              maximum={MAP_MAX}
              point={[z1, z2]}
              onChange={([x, y]) => setState({ latentX: x, latentY: y })}
              encoded={encodedValue ? { mean: encodedValue.mean, sigma: encodedValue.sigma } : null}
              label="Latent map"
            />
            <div className="dv-map-side">
              <figure className="dv-figure dv-figure--hero">
                <PixelImage
                  data={digit.value?.image ?? null}
                  size={168}
                  label={`Decoder output at z1 ${z1.toFixed(2)}, z2 ${z2.toFixed(2)}`}
                />
                <figcaption>decoder(z) at ({num(z1)}, {num(z2)})</figcaption>
              </figure>
              <RangeControl
                label="Latent z₁"
                min={MAP_MIN}
                max={MAP_MAX}
                step={0.05}
                value={z1}
                format={(value) => signed(value)}
                onChange={(value) => setState({ latentX: value })}
              />
              <RangeControl
                label="Latent z₂"
                min={MAP_MIN}
                max={MAP_MAX}
                step={0.05}
                value={z2}
                format={(value) => signed(value)}
                onChange={(value) => setState({ latentY: value })}
              />
              <div className="metric-row">
                <Metric label="‖z‖" value={`${radius.toFixed(2)} σ`} tone="forward" />
                <Metric label="log p(z)" value={`${num(logDensity)} nats`} />
              </div>
            </div>
          </div>
          <ul className="dv-legend" aria-label="Map legend">
            <li><i className="dv-legend__ring" /> Prior N(0, I): dashed circles at 1σ and 2σ</li>
            <li><i className="dv-legend__point" /> Your point z</li>
            <li><i className="dv-legend__diamond" /> Encoded drawing: mean, with a 2σ ellipse</li>
          </ul>
          <p className="lab-note">
            The mosaic is the real decoder run on a {MAP_GRID} × {MAP_GRID} grid from {MAP_MIN} to {MAP_MAX}. Each digit
            owns a region and the decoder blends two digits where their regions meet, most of all in the crowded middle,
            where the prior puts most of its mass.
          </p>
        </LabSurface>

        <LabSurface label="Encode a drawing" className="dv-surface">
          <SurfaceHeading
            kicker="Encoder → (μ, σ) → decoder"
            title="Compress a drawing to two numbers and back"
            aside={<StatusChip {...encodedStatus} />}
          />
          <div className="dv-encode">
            <div className="dv-encode__input">
              <DrawingPad
                strokes={strokes}
                label="Drawing pad, 28 by 28 pixels"
                onCommit={(next) => {
                  setStrokes(next);
                  setRevision((value) => value + 1);
                  setState({ sample: "custom" });
                }}
              />
              <div className="dv-presets" role="group" aria-label="Load a hand-drawn digit">
                {PRESETS.map((name) => (
                  <button
                    type="button"
                    key={name}
                    aria-pressed={sample === name}
                    onClick={() => loadPreset(name)}
                  >
                    {PRESET_LABELS[name]}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    setStrokes([]);
                    setRevision((value) => value + 1);
                    setState({ sample: "custom" });
                  }}
                >
                  Clear
                </button>
              </div>
              <small>Draw with a mouse or pen, or load a preset.</small>
            </div>
            <div className="dv-encode__outputs">
              <figure className="dv-figure">
                <PixelImage
                  data={encodedValue?.meanImage ?? null}
                  size={112}
                  label="Reconstruction decoded from the posterior mean"
                />
                <figcaption>decoder(μ)</figcaption>
              </figure>
              <figure className="dv-figure">
                <PixelImage
                  data={encodedValue?.sampleImage ?? null}
                  size={112}
                  label="Reconstruction decoded from a sampled latent"
                />
                <figcaption>decoder(μ + σ·ε)</figcaption>
              </figure>
            </div>
            <div className="dv-encode__numbers">
              <dl className="dv-readout">
                <div>
                  <dt>Posterior mean μ</dt>
                  <dd className="is-forward">
                    {encodedValue ? `(${num(encodedValue.mean[0])}, ${num(encodedValue.mean[1])})` : "—"}
                  </dd>
                </div>
                <div>
                  <dt>Posterior σ (prior σ = 1)</dt>
                  <dd className="is-forward">
                    {encodedValue ? `(${fmt(encodedValue.sigma[0], 3)}, ${fmt(encodedValue.sigma[1], 3)})` : "—"}
                  </dd>
                </div>
                <div>
                  <dt>Reconstruction BCE, decoder(μ)</dt>
                  <dd className="is-loss">{encodedValue ? `${fmt(encodedValue.bce, 1)} nats` : "—"}</dd>
                </div>
                <div>
                  <dt>KL(q(z | x) ‖ N(0, I))</dt>
                  <dd className="is-gradient">{encodedValue ? `${fmt(encodedValue.kl, 2)} nats` : "—"}</dd>
                </div>
                <div className="is-total">
                  <dt>Training loss, BCE + {KL_WEIGHT} × KL</dt>
                  <dd>{encodedValue ? `${fmt(encodedValue.bce + KL_WEIGHT * encodedValue.kl, 1)} nats` : "—"}</dd>
                </div>
              </dl>
              <p className="dv-small">
                BCE sums over 784 pixels. Training scored the sampled latent instead, which gives{" "}
                {encodedValue ? `${fmt(encodedValue.bceSampled, 1)} nats` : "—"} here; the two agree closely because σ is tiny next to the prior's 1.
              </p>
            </div>
          </div>
        </LabSurface>
      </div>
    );
  }

  return (
    <div className="dv-lab">
      {modeBar}
      <LabSurface label="Forward process" className="dv-surface">
        <SurfaceHeading
          kicker="q(x_t | x₀) = N(√ᾱ_t · x₀, (1 − ᾱ_t) I)"
          title="Adding noise needs no network"
          aside={<StatusChip status={errorRun.status === "error" ? "error" : errorRun.value ? "ready" : "loading"} text={errorRun.status === "error" ? "Denoiser unavailable" : errorRun.value ? providerText(errorRun.value.provider) : "Scoring the denoiser…"} error={errorRun.error} />}
        />
        <div className="dv-strip" role="list" aria-label="The same digit at six noise levels">
          <figure className="dv-figure" role="listitem">
            <PixelImage data={x0Signed} range={[-1, 1]} size={72} label="Clean digit x0" />
            <figcaption>x₀</figcaption>
          </figure>
          {FORWARD_STEPS.map((t, index) => (
            <figure className={`dv-figure ${index === nearestForward ? "is-current" : ""}`} role="listitem" key={t}>
              <PixelImage
                data={forwardStrip?.[index] ?? null}
                range={[-1, 1]}
                size={72}
                label={`x at t ${t}: signal ${Math.sqrt(schedule.alphaBar[t]).toFixed(2)}, noise ${Math.sqrt(1 - schedule.alphaBar[t]).toFixed(2)}`}
              />
              <figcaption>
                t = {t}
                <small>√ᾱ {Math.sqrt(schedule.alphaBar[t]).toFixed(2)}</small>
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="dv-small">{x0Caption}. Pixels are scaled to [−1, 1]; one seeded ε is reused across the strip.</p>
        <div className="dv-charts">
          <LineChart
            label="Noise schedule"
            series={scheduleSeries}
            xLabel="step t"
            yLabel="coefficient"
            xDomain={[0, LAST]}
            yDomain={[0, 1]}
            marker={{ x: markerT, label: `t = ${markerT}` }}
            footnote={`Cosine schedule: β_t grows from ${schedule.beta[0].toFixed(4)} to ${schedule.beta[LAST].toFixed(2)} over 100 steps, so the last step keeps only √ᾱ = ${Math.sqrt(schedule.alphaBar[LAST]).toFixed(3)} of the digit.`}
          />
          {errorRun.value ? (
            <LineChart
              label="Noise-prediction error by step"
              series={errorSeries}
              xLabel="step t"
              yLabel="mean squared error of ε̂"
              xDomain={[0, LAST]}
              yDomain={[0, 1.2]}
              marker={{ x: markerT, label: `t = ${markerT}` }}
              footnote={`The shipped network on this x₀, one seeded ε, all 100 steps in one batched call. Training ended at a mean of ${finalNoiseMse.toFixed(3)} after ${diffusionHistory.length} epochs.`}
            />
          ) : (
            <div className="dv-chart-empty" role="status">
              {errorRun.status === "error"
                ? "The denoiser did not load, so its error curve cannot be measured here."
                : "Running the denoiser on 100 noisy copies of x₀…"}
            </div>
          )}
        </div>
      </LabSurface>

      <LabSurface label="Reverse diffusion" className="dv-surface">
        <SurfaceHeading
          kicker={denoiser === "ideal" ? "Ideal denoiser · exact posterior mean over 64 references" : "Trained network · shipped ONNX denoiser"}
          title="Each step calls the denoiser once"
          aside={<StatusChip {...reverseStatus} />}
        />
        <div className="dv-controls">
          <SegmentedControl
            label="Denoiser"
            value={denoiser}
            options={[
              { value: "ideal", label: "Ideal denoiser" },
              { value: "network", label: "Trained network" },
            ]}
            onChange={(value) => setState({ denoiser: value })}
          />
          <SegmentedControl
            label="Start from"
            value={start}
            options={[
              { value: "digit", label: "Noised x₀" },
              { value: "noise", label: "Pure noise" },
            ]}
            onChange={(value) => setState({ start: value })}
          />
          <RangeControl
            label="Noise seed"
            min={1}
            max={SEED_MAX}
            step={1}
            value={seed}
            onChange={(value) => setState({ seed: value })}
          />
        </div>

        <div className="dv-now">
          <figure className="dv-figure dv-figure--hero">
            <PixelImage
              data={stateAt}
              range={[-1, 1]}
              size={150}
              label={calls < STEPS ? `Image after ${calls} denoiser calls, noise level t ${currentLevel}` : "Final sample after 100 denoiser calls"}
            />
            <figcaption>{calls < STEPS ? `x after ${calls} calls · level t = ${currentLevel}` : "sample after 100 calls"}</figcaption>
          </figure>
          <span className="dv-now__op" aria-hidden="true">→</span>
          <figure className="dv-figure dv-figure--hero">
            <PixelImage
              data={cleanAt}
              range={[-1, 1]}
              size={150}
              label={calls < STEPS ? `Predicted clean image at t ${nextT}` : "No call left"}
            />
            <figcaption>{calls < STEPS ? `predicted x̂₀ (call ${calls + 1}, t = ${nextT})` : "no call left"}</figcaption>
          </figure>
          <figure className="dv-figure dv-figure--hero">
            <PixelImage
              data={noiseAt}
              range={[-2.5, 2.5]}
              size={150}
              label={calls < STEPS ? `Predicted noise at t ${nextT}` : "No call left"}
            />
            <figcaption>{calls < STEPS ? `predicted ε̂ (grey = 0)` : "no call left"}</figcaption>
          </figure>
        </div>

        <RangeControl
          label="Denoiser calls"
          min={0}
          max={STEPS}
          step={1}
          value={calls}
          format={(value) => `${value} / ${STEPS}`}
          onChange={(value) => setState({ step: value })}
        />
        <div className="dv-film" role="group" aria-label="Jump to a point in the reverse process">
          {FILM_CALLS.map((call) => (
            <button
              type="button"
              key={call}
              aria-pressed={calls === call}
              aria-label={`After ${call} denoiser calls`}
              onClick={() => setState({ step: call })}
            >
              <PixelImage data={trajectory?.states[call] ?? null} range={[-1, 1]} size={40} label={`After ${call} calls`} />
              <span>{call}</span>
            </button>
          ))}
        </div>

        {calls < STEPS ? (
          <FormulaWithValues
            label={`Call ${calls + 1}: DDPM update at t = ${nextT}`}
            expression={`${nextT > 0 ? `x_${nextT - 1}` : "sample"} = (x_${nextT} − ${fmt(coefficients.epsilonScale, 4)} · ε̂) × ${fmt(coefficients.rescale, 4)}${coefficients.sigma > 0 ? ` + ${fmt(coefficients.sigma, 4)} · z` : ""}`}
            result={`β = ${fmt(coefficients.beta, 5)}`}
            detail="ε̂ multiplier = β_t / √(1 − ᾱ_t); rescale = 1 / √α_t; fresh noise z has σ_t = √β_t and is skipped on the last call."
            tone="gradient"
          />
        ) : (
          <FormulaWithValues
            label="Sample"
            expression="all 100 calls done"
            result={nearestReference ? `nearest: reference #${nearestReference.index + 1}` : "—"}
            detail={
              denoiser === "ideal"
                ? `Mean squared distance from the sample to that reference: ${fmt(nearestReference?.distance ?? NaN, 4)}. The ideal denoiser can only return one of its references.`
                : `Mean squared distance to the nearest reference: ${fmt(nearestReference?.distance ?? NaN, 2)} (pixels span −1 to 1). Not zero: the network's sample is a new image, not one of the 64 references.`
            }
            tone="gradient"
          />
        )}

        <div className="metric-row">
          <Metric label="Denoiser calls" value={`${calls} / ${STEPS}`} tone="gradient" />
          <Metric label="Noise level" value={currentLevel >= 0 ? `t = ${currentLevel}` : "done"} />
          <Metric
            label="√ᾱ · √(1−ᾱ)"
            value={`${fmt(Math.sqrt(alphaBarNow), 2)} · ${fmt(Math.sqrt(1 - alphaBarNow), 2)}`}
            tone="forward"
          />
          <Metric
            label="Background pixels"
            value={Number.isFinite(backgroundAt) ? `${fmt(backgroundAt, 2)} (x₀ ${fmt(cleanBackground, 2)})` : "—"}
            tone="loss"
          />
        </div>

        {denoiser === "ideal" ? (
          <div className="dv-references">
            <div className="dv-references__grid" aria-hidden="true">
              <PixelMosaic tiles={referenceTiles} columns={referenceTiles && referenceTiles.length < 64 ? referenceTiles.length : 8} label="Reference set" />
              {weightsAt && (
                <div
                  className="dv-references__weights"
                  style={{ gridTemplateColumns: `repeat(${references && references.length < 64 ? references.length : 8}, 1fr)` }}
                >
                  {Array.from(weightsAt).map((weight, index) => (
                    <i key={index} style={{ opacity: Math.min(1, weight * 2) }} className={index === topReference ? "is-top" : ""} />
                  ))}
                </div>
              )}
            </div>
            <p className="dv-small">
              {references && references.length < 64
                ? "Fallback reference set: four hand-drawn digits, because the VAE decoder did not load. "
                : "Reference set: 64 digits drawn by the VAE decoder on an 8 × 8 latent grid. "}
              {weightsAt
                ? `At t = ${nextT} the posterior puts ${(weightsAt[topReference] * 100).toFixed(1)}% of its weight on reference #${topReference + 1} (outlined).`
                : "Weights appear once the references are decoded."}{" "}
              x̂₀ is the weighted average of the references, so it is blurry only while the weight is spread.
              {start === "digit" &&
                ` Noised x₀ keeps only ${fmt(Math.sqrt(schedule.alphaBar[LAST]), 2)} × x₀ at t = ${LAST}, so it is almost the same start as pure noise.`}
            </p>
          </div>
        ) : (
          <p className="lab-note">
            The shipped denoiser is a {metadata.diffusion.parameters.toLocaleString("en-US")}-parameter convolutional U-Net
            trained for {diffusionHistory.length} epochs on all {metadata.samples.toLocaleString("en-US")} MNIST training digits.
            Its training noise MSE ended at {finalNoiseMse.toFixed(3)}, against 1.0 for always predicting zero. It is
            unconditional, so the seed picks the digit and you cannot ask for one. Compare the Ideal denoiser with the same seed.
          </p>
        )}
      </LabSurface>
    </div>
  );
}
