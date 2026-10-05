import { useEffect, useMemo, useState } from "react";
import {
  BarList,
  FormulaWithValues,
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  VectorChip,
  type ModuleContext,
} from "@app/module-sdk";
import {
  LIKELIHOOD_NEGATIVE,
  LIKELIHOOD_POSITIVE,
  SAMPLE,
  bayes,
  convexGrad,
  convexLoss,
  crossEntropy as crossEntropyOf,
  descend,
  det2,
  entropy as entropyOf,
  gaussianMass,
  gaussianPdf,
  logit as logitOf,
  nonconvexGrad,
  nonconvexLoss,
  sampleMean,
  sampleVariance,
  sigmoid as sigmoidOf,
  softmax2,
  stationaryPoints,
} from "./curves";
import { DerivativePlot, GaussianPlot, LossPlot, MatrixPlane, SigmoidPlot, SurprisePlot, VectorPlane } from "./plots";

const TOPICS = ["vectors", "matrices", "derivatives", "optimization", "probability", "logarithms"] as const;
const PROB_IDEAS = ["counts", "bayes", "gaussian", "logit"] as const;
const PROB_IDEA_OPTIONS = [
  { value: "counts", label: "counts and softmax" },
  { value: "bayes", label: "Bayes' rule" },
  { value: "gaussian", label: "mean and variance" },
  { value: "logit", label: "logit to probability" },
];

const EMPIRICAL = [
  { x: 0, y: 0, n: 3 },
  { x: 0, y: 1, n: 1 },
  { x: 1, y: 0, n: 1 },
  { x: 1, y: 1, n: 3 },
] as const;

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) =>
  Math.min(high, Math.max(low, value));

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const topic = (TOPICS as readonly string[]).includes(asString(state, "topic", "vectors"))
    ? asString(state, "topic", "vectors")
    : "vectors";
  const ax = clamp(asNumber(state, "ax", 1), -2, 2);
  const ay = clamp(asNumber(state, "ay", 0.4), -2, 2);
  const bx = clamp(asNumber(state, "bx", 0.6), -2, 2);
  const by = clamp(asNumber(state, "by", 1.1), -2, 2);
  const m00 = clamp(asNumber(state, "m00", 1), -2, 2);
  const m01 = clamp(asNumber(state, "m01", 0.4), -2, 2);
  const m10 = clamp(asNumber(state, "m10", 0), -2, 2);
  const m11 = clamp(asNumber(state, "m11", 1), -2, 2);
  const vx = clamp(asNumber(state, "vx", 1), -2, 2);
  const vy = clamp(asNumber(state, "vy", 0.2), -2, 2);
  const b0 = clamp(asNumber(state, "b0", 0.2), -1, 1);
  const b1 = clamp(asNumber(state, "b1", 0.1), -1, 1);
  const x = clamp(asNumber(state, "x", 1.2), -2, 2);
  const nudge = clamp(asNumber(state, "nudge", 0.4), 0.05, 0.8);
  const pA = clamp(asNumber(state, "pA", 2.4), 0.1, 6);
  const pB = clamp(asNumber(state, "pB", 1.1), 0.1, 6);
  const value = clamp(asNumber(state, "value", 0.25), 0.02, 0.98);
  const modelQ = clamp(asNumber(state, "modelQ", 0.6), 0.02, 0.98);
  const surface = asString(state, "surface", "convex") === "nonconvex" ? "nonconvex" : "convex";
  const optW = clamp(asNumber(state, "optW", -1.4), -2, 2);
  const stepSize = clamp(asNumber(state, "stepSize", 0.15), 0.02, 0.8);
  const probIdea = (PROB_IDEAS as readonly string[]).includes(asString(state, "probIdea", "counts"))
    ? asString(state, "probIdea", "counts")
    : "counts";
  const prior = clamp(asNumber(state, "prior", 0.5), 0.02, 0.98);
  const gaussMu = clamp(asNumber(state, "gaussMu", 0), -2, 2);
  const gaussVar = clamp(asNumber(state, "gaussVar", 1), 0.1, 2);
  const gaussX = clamp(asNumber(state, "gaussX", 1), -4, 4);

  const dot = ax * bx + ay * by;
  const lengthA = Math.hypot(ax, ay);
  const lengthB = Math.hypot(bx, by);
  const distanceAB = Math.hypot(ax - bx, ay - by);
  const cosine = lengthA * lengthB === 0 ? 0 : dot / (lengthA * lengthB);
  const projScale = lengthB === 0 ? 0 : dot / (lengthB * lengthB);
  const projX = projScale * bx;
  const projY = projScale * by;
  const tx = m00 * vx + m01 * vy;
  const ty = m10 * vx + m11 * vy;
  const y0 = tx + b0;
  const y1 = ty + b1;
  const y = x * x;
  const slope = 2 * x;
  const nudged = x + nudge;
  const yNudge = nudged * nudged;
  const secant = (yNudge - y) / nudge;
  const [probA, probB] = softmax2(pA, pB);
  const logLoss = -Math.log(value);
  const entropy = entropyOf(value);
  const crossEntropy = crossEntropyOf(value, modelQ);
  const kl = crossEntropy - entropy;
  const determinant = det2(m00, m01, m10, m11);
  const nX0 = 4;
  const nX1 = 4;
  const pY1Given0 = 1 / nX0;
  const pY1Given1 = 3 / nX1;
  const likelihood = probA * probA * probB;
  const logLik = 2 * Math.log(probA) + Math.log(probB);
  const bay = bayes(prior, LIKELIHOOD_POSITIVE, LIKELIHOOD_NEGATIVE);
  const sampleAverage = sampleMean(SAMPLE);
  const sampleSpread = sampleVariance(SAMPLE);
  const sigma = Math.sqrt(gaussVar);
  const density = gaussianPdf(gaussX, gaussMu, gaussVar);
  const zScore = (gaussX - gaussMu) / sigma;
  const withinOneSigma = gaussianMass(gaussMu, gaussVar, gaussMu - sigma, gaussMu + sigma);
  const logitZ = pA - pB;
  const sigmoidZ = sigmoidOf(logitZ);
  const lossAt = surface === "convex" ? convexLoss(optW) : nonconvexLoss(optW);
  const gradAt = surface === "convex" ? convexGrad(optW) : nonconvexGrad(optW);
  const minima = useMemo(() => stationaryPoints(surface).filter((point) => point.kind === "minimum"), [surface]);
  const nearestMin = minima.reduce((best, point) =>
    Math.abs(point.w - optW) < Math.abs(best.w - optW) ? point : best,
  );

  /** The path walked with the downhill button. Local view state: moving the slider starts a new path. */
  const [trail, setTrail] = useState<number[]>([optW]);
  useEffect(() => {
    setTrail((previous) => (previous[previous.length - 1] === optW ? previous : [optW]));
  }, [optW]);
  useEffect(() => {
    setTrail([optW]);
    // a new surface starts a new path
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surface]);

  const formula = (() => {
    if (topic === "vectors") {
      return {
        label: "Dot product",
        expression: `${ax.toFixed(2)}·${bx.toFixed(2)} + ${ay.toFixed(2)}·${by.toFixed(2)}`,
        result: dot.toFixed(3),
        detail: `|a||b| cos θ = ${lengthA.toFixed(2)} × ${lengthB.toFixed(2)} × ${cosine.toFixed(2)}. If a is features and b is weights, this is the linear score before a bias.`,
      };
    }
    if (topic === "matrices") {
      return {
        label: "Affine layer  W x + b",
        expression: `[${m00.toFixed(1)} ${m01.toFixed(1)}; ${m10.toFixed(1)} ${m11.toFixed(1)}] [${vx.toFixed(2)}; ${vy.toFixed(2)}] + [${b0.toFixed(1)}; ${b1.toFixed(1)}]`,
        result: `[${y0.toFixed(2)}, ${y1.toFixed(2)}]`,
        detail: `Shapes (2×2)(2×1) + (2×1) = (2×1): the inner sizes must match. Each output coordinate is one row of W dotted with x, plus a bias. Wx+b is affine, not linear through the origin, unless b = 0. det W = ${determinant.toFixed(2)} scales every area.`,
      };
    }
    if (topic === "derivatives") {
      return {
        label: "Analytic slope versus a nudge",
        expression: `d/dx (x²) = 2x = ${slope.toFixed(3)}   ·   [f(x+ε)−f(x)]/ε = ${secant.toFixed(3)}`,
        result: `gap ${(secant - slope).toFixed(3)}`,
        detail: `For f(x)=x² the secant is exactly 2x+ε. As the nudge ε=${nudge.toFixed(2)} shrinks, the finite difference meets the derivative.`,
      };
    }
    if (topic === "optimization") {
      return {
        label: surface === "convex" ? "Convex bowl  (w−0.8)² + 0.25" : "Tilted double well  (w²−1)² + 0.35w + 0.6",
        expression: `L(${optW.toFixed(2)}) = ${lossAt.toFixed(3)}   ·   L'(w) = ${gradAt.toFixed(3)}`,
        result: `next w ${(optW - stepSize * gradAt).toFixed(3)}`,
        detail:
          surface === "convex"
            ? "One valley. The gradient is the direction of steepest increase, so a downhill step subtracts step-size × L'(w)."
            : "Two valleys of different depth. A start on the right can settle in the shallower basin. That local min is real. High-dimensional nets are not this picture.",
      };
    }
    if (topic === "probability" && probIdea === "bayes") {
      return {
        label: "Bayes' rule on the count table",
        expression: `P(y=1|x=1) = (${prior.toFixed(2)} × ${LIKELIHOOD_POSITIVE.toFixed(2)}) / (${prior.toFixed(2)} × ${LIKELIHOOD_POSITIVE.toFixed(2)} + ${(1 - prior).toFixed(2)} × ${LIKELIHOOD_NEGATIVE.toFixed(2)})`,
        result: bay.posterior.toFixed(3),
        detail: `The prior is the share of y=1 before looking at x. The likelihoods 3/4 and 1/4 are P(x=1|y) read from the count table. In odds form, posterior odds = prior odds × ${bay.likelihoodRatio.toFixed(0)}.`,
      };
    }
    if (topic === "probability" && probIdea === "gaussian") {
      return {
        label: "Gaussian density N(μ, σ²)",
        expression: `N(${gaussX.toFixed(1)}) = exp(−(${gaussX.toFixed(1)} − ${gaussMu.toFixed(2)})² / (2 × ${gaussVar.toFixed(2)})) / √(2π × ${gaussVar.toFixed(2)})`,
        result: density.toFixed(4),
        detail: "A density is a height, not a probability: it can exceed 1 when the variance is small. Probability is area, and about 68% of the area lies within one σ of μ. Naive Bayes fits one mean and one variance per class and feature.",
      };
    }
    if (topic === "probability" && probIdea === "logit") {
      return {
        label: "Two-class softmax is a sigmoid",
        expression: `P(A) = e^${pA.toFixed(1)} / (e^${pA.toFixed(1)} + e^${pB.toFixed(1)}) = 1 / (1 + e^−${logitZ.toFixed(1)}) = σ(${logitZ.toFixed(1)})`,
        result: sigmoidZ.toFixed(4),
        detail: "The logit z = A − B is the log-odds ln(P(A)/P(B)). Logistic regression computes z = w·x + b and reports σ(z) as P(y=1|x).",
      };
    }
    if (topic === "probability") {
      return {
        label: "Softmax, then a 3-draw likelihood",
        expression: `P(A) = e^${pA.toFixed(1)} / (e^${pA.toFixed(1)} + e^${pB.toFixed(1)}) = ${probA.toFixed(3)}   ·   P(A,A,B) = P(A)²P(B)`,
        result: likelihood.toFixed(4),
        detail: `Empirical P(y=1|x=1) = 3/4 from the count table. Softmax only turns scores into a distribution. Likelihood multiplies those probabilities for the observed draws.`,
      };
    }
    return {
      label: "Surprise, entropy, cross-entropy (nats)",
      expression: `H(p,q) = −${value.toFixed(2)}·ln ${modelQ.toFixed(2)} − ${(1 - value).toFixed(2)}·ln ${(1 - modelQ).toFixed(2)} = ${crossEntropy.toFixed(3)}   ·   H(p) = ${entropy.toFixed(3)}`,
      result: `KL ${kl.toFixed(3)}`,
      detail: "−ln q is the surprise of an event the model called q. Cross-entropy is the expected surprise when the world is p but you believed q. It equals entropy only when q = p. Natural log, so the unit is nats; log base 2 would give bits.",
    };
  })();

  const walkthrough = (() => {
    if (topic === "vectors") {
      const t1 = ax * bx;
      const t2 = ay * by;
      return {
        title: "Two features, one weighted sum",
        steps: [
          `Read a as a row: rooms-like ${ax.toFixed(2)}, park-like ${ay.toFixed(2)}.`,
          `Read b as weights: ${bx.toFixed(2)} and ${by.toFixed(2)}.`,
          `Pair and multiply: ${ax.toFixed(2)}×${bx.toFixed(2)} = ${t1.toFixed(3)}, and ${ay.toFixed(2)}×${by.toFixed(2)} = ${t2.toFixed(3)}.`,
          `Add: a · b = ${t1.toFixed(3)} + ${t2.toFixed(3)} = ${dot.toFixed(3)}.`,
          `Same number as |a||b|cos θ = ${lengthA.toFixed(2)}×${lengthB.toFixed(2)}×${cosine.toFixed(2)}.`,
        ],
      };
    }
    if (topic === "matrices") {
      return {
        title: "A 2-unit layer, written out",
        steps: [
          `Row 0 · x = ${m00.toFixed(2)}×${vx.toFixed(2)} + ${m01.toFixed(2)}×${vy.toFixed(2)} = ${tx.toFixed(3)}.`,
          `Row 1 · x = ${m10.toFixed(2)}×${vx.toFixed(2)} + ${m11.toFixed(2)}×${vy.toFixed(2)} = ${ty.toFixed(3)}.`,
          `Add bias: y = [${tx.toFixed(3)} + ${b0.toFixed(2)}, ${ty.toFixed(3)} + ${b1.toFixed(2)}] = [${y0.toFixed(2)}, ${y1.toFixed(2)}].`,
          `The shaded parallelogram is that same map applied to the corners (±1, ±1) of the dashed square, then shifted by b. Its area is |det W| = ${Math.abs(determinant).toFixed(2)} times the square's.`,
          `Shapes: W is 2×2 and x is 2×1, so Wx is 2×1. Stack n inputs as the rows of an n×2 matrix X and X Wᵀ + b scores all n at once — that stacked grid is a batch.`,
        ],
      };
    }
    if (topic === "derivatives") {
      return {
        title: "Nudge x and watch f change",
        steps: [
          `f(x) = x², so f(${x.toFixed(2)}) = ${y.toFixed(3)}.`,
          `Nudge ε = ${nudge.toFixed(2)} gives f(${nudged.toFixed(2)}) = ${yNudge.toFixed(3)}.`,
          `Change in output: ${yNudge.toFixed(3)} − ${y.toFixed(3)} = ${(yNudge - y).toFixed(3)}.`,
          `Per unit nudge: ${(yNudge - y).toFixed(3)} / ${nudge.toFixed(2)} = ${secant.toFixed(3)}.`,
          `The derivative 2x = ${slope.toFixed(3)} is that ratio in the limit ε → 0. For this f, the gap is exactly ε.`,
        ],
      };
    }
    if (topic === "optimization") {
      const next = optW - stepSize * gradAt;
      return {
        title: "The gradient points uphill",
        steps: [
          surface === "convex"
            ? `L(w) = (w − 0.8)² + 0.25 is a single bowl. L'(w) = 2(w − 0.8).`
            : `L(w) = (w² − 1)² + 0.35w + 0.6. The +0.35w tilts two wells so they are not equal.`,
          `At w = ${optW.toFixed(2)}, L = ${lossAt.toFixed(3)} and L' = ${gradAt.toFixed(3)}.`,
          `Steepest increase is the sign of L'. Downhill is the opposite: w ← w − η L'.`,
          `η = ${stepSize.toFixed(2)} gives the next w ≈ ${next.toFixed(3)} (then clipped to [−2, 2]).`,
          surface === "convex"
            ? `From either side, small η reaches w = 0.8. Here L'' = 2, so any η above 0.5 overshoots the floor each step, and only η above 1 would diverge.`
            : `Start near +1.4 and you settle in the shallower well at w ≈ ${minima[minima.length - 1]?.w.toFixed(2)}. That is a local minimum, not a myth. Deep nets live in thousands of dimensions; this curve does not describe them.`,
        ],
      };
    }
    if (topic === "probability" && probIdea === "bayes") {
      return {
        title: "Prior times likelihood, then normalize",
        steps: [
          `Prior: P(y=1) = ${prior.toFixed(2)} is the share of class 1 before looking at x. The count table has 4 of 8 rows in each class, so its own prior is 0.50.`,
          `Likelihoods from the counts: P(x=1|y=1) = 3/4 = ${LIKELIHOOD_POSITIVE.toFixed(2)} and P(x=1|y=0) = 1/4 = ${LIKELIHOOD_NEGATIVE.toFixed(2)}.`,
          `Weight each explanation of x = 1: ${prior.toFixed(2)} × ${LIKELIHOOD_POSITIVE.toFixed(2)} = ${bay.jointPositive.toFixed(3)} for y=1, and ${(1 - prior).toFixed(2)} × ${LIKELIHOOD_NEGATIVE.toFixed(2)} = ${bay.jointNegative.toFixed(3)} for y=0.`,
          `Evidence: P(x=1) = ${bay.jointPositive.toFixed(3)} + ${bay.jointNegative.toFixed(3)} = ${bay.evidence.toFixed(3)}, the total weight of the ways x = 1 can happen.`,
          `Posterior: P(y=1|x=1) = ${bay.jointPositive.toFixed(3)} / ${bay.evidence.toFixed(3)} = ${bay.posterior.toFixed(3)}. Odds form: ${bay.priorOdds.toFixed(2)} × ${bay.likelihoodRatio.toFixed(0)} = ${bay.posteriorOdds.toFixed(2)}.`,
        ],
      };
    }
    if (topic === "probability" && probIdea === "gaussian") {
      return {
        title: "From four numbers to a bell",
        steps: [
          `Sample: ${SAMPLE.join(", ")}. The mean is their sum over n: ${SAMPLE.join(" + ")} = ${SAMPLE.reduce((sum, item) => sum + item, 0)}, divided by ${SAMPLE.length} is ${sampleAverage.toFixed(2)}.`,
          `Deviations from the mean, squared: ${SAMPLE.map((item) => ((item - sampleAverage) ** 2).toFixed(2)).join(", ")}. Their average is the variance, ${sampleSpread.toFixed(2)}, and √${sampleSpread.toFixed(2)} = ${Math.sqrt(sampleSpread).toFixed(3)} is the standard deviation.`,
          `Your bell has μ = ${gaussMu.toFixed(2)} and σ² = ${gaussVar.toFixed(2)}, so σ = ${sigma.toFixed(3)}. The mean sets where it is centered and the variance sets how wide.`,
          `At x = ${gaussX.toFixed(1)} the height is exp(−(${gaussX.toFixed(1)} − ${gaussMu.toFixed(2)})² / (2 × ${gaussVar.toFixed(2)})) / √(2π × ${gaussVar.toFixed(2)}) = ${density.toFixed(4)}, which is ${zScore.toFixed(2)} standard deviations from the mean.`,
          `About ${(withinOneSigma * 100).toFixed(1)}% of the area lies between ${(gaussMu - sigma).toFixed(2)} and ${(gaussMu + sigma).toFixed(2)}. Height is not probability; area is.`,
        ],
      };
    }
    if (topic === "probability" && probIdea === "logit") {
      return {
        title: "Two logits, one gap, one sigmoid",
        steps: [
          `Two logits: A = ${pA.toFixed(1)} and B = ${pB.toFixed(1)}. Softmax only uses their gap z = A − B = ${logitZ.toFixed(1)}.`,
          `Softmax: P(A) = e^${pA.toFixed(1)} / (e^${pA.toFixed(1)} + e^${pB.toFixed(1)}) = ${probA.toFixed(4)}.`,
          `Divide top and bottom by e^${pA.toFixed(1)}: P(A) = 1 / (1 + e^−z) = σ(${logitZ.toFixed(1)}) = ${sigmoidZ.toFixed(4)}. It is the same number.`,
          `Run it backward: ln(P(A)/P(B)) = ln(${probA.toFixed(3)} / ${probB.toFixed(3)}) = ${logitOf(probA).toFixed(3)}. The log-odds of A is the logit gap.`,
          `At z = 0 the probability is 0.5, and it flattens toward 0 and 1 as |z| grows. A logistic regression computes z = w·x + b, and σ(z) is P(y=1|x).`,
        ],
      };
    }
    if (topic === "probability") {
      return {
        title: "Counts, then P(y|x), then a likelihood",
        steps: [
          `When x=0 the table has 3 zeros and 1 one, so P(y=1|x=0) = 1/4 = ${pY1Given0.toFixed(2)}.`,
          `When x=1 it has 1 zero and 3 ones, so P(y=1|x=1) = 3/4 = ${pY1Given1.toFixed(2)}.`,
          `Expectation E[y|x=1] = 0·(1/4) + 1·(3/4) = ${pY1Given1.toFixed(2)}. A conditional mean, not a destiny.`,
          `Softmax on the two logits gives a model distribution P(A)=${(probA * 100).toFixed(1)}%, P(B)=${(probB * 100).toFixed(1)}%.`,
          `Likelihood of the sequence A, A, B is P(A)²P(B) = ${likelihood.toFixed(4)}. Logs turn that product into a sum: 2 ln P(A) + ln P(B) = ${logLik.toFixed(3)}.`,
        ],
      };
    }
    return {
      title: "Surprise of p, then surprise under q",
      steps: [
        `An event with probability p = ${value.toFixed(2)} has surprise −ln p = ${logLoss.toFixed(3)} nats.`,
        `A fair coin (p = 0.5) has surprise ln 2 ≈ 0.693 nats, which is exactly 1 bit. Rare events surprise more.`,
        `Entropy H(p) = −p ln p − (1−p) ln(1−p) = ${entropy.toFixed(3)} is the expected surprise if the world is p.`,
        `The model says q = ${modelQ.toFixed(2)}. Cross-entropy H(p,q) = ${crossEntropy.toFixed(3)} is the expected surprise of p if you believed q.`,
        `KL(p‖q) = H(p,q) − H(p) = ${kl.toFixed(3)} ≥ 0, and is 0 only when q = p. A confident wrong q makes −ln q huge.`,
      ],
    };
  })();

  const takeOptStep = () => {
    const next = descend(surface, optW, stepSize);
    setTrail((previous) => [...previous.slice(-24), next]);
    setState({ optW: next });
    narrate(`Downhill step on ${surface} L moved w to ${next.toFixed(3)}.`);
  };

  return (
    <div className="gw-shell">
    <div className="tg-lab tg-lab--hero gw-lab myn-lab">
      <LabSurface label="Topic picker" className="math-topic-card">
        <SurfaceHeading
          kicker="Six refreshers, one focused widget"
          title="Choose the idea a later lab will deep-link"
          aside={<span className="tg-badge">{topic}</span>}
        />
        <div className="math-topic-control">
          <SegmentedControl
            label="Topic"
            value={topic}
            options={TOPICS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => setState({ topic: value })}
          />
        </div>
        <p className="lab-note">
          `topic` is stored as {topic}. Later labs can still open vectors, matrices, derivatives,
          probability, or logarithms. Optimization is the extra refresher for a loss surface.
        </p>
      </LabSurface>

      <LabSurface label="Active refresher" className="math-widget-card">
        <SurfaceHeading
          kicker={topic}
          title={
            topic === "vectors"
              ? "Two arrows, a weighted sum"
              : topic === "matrices"
                ? "W x + b on a square"
                : topic === "derivatives"
                  ? "A nudge on x²"
                  : topic === "optimization"
                    ? "Downhill on a 1-D loss"
                    : topic === "probability"
                      ? probIdea === "bayes"
                        ? "Prior × likelihood, then normalize"
                        : probIdea === "gaussian"
                          ? "A bell with two numbers"
                          : probIdea === "logit"
                            ? "Logit, sigmoid, probability"
                            : "P(y|x), then a likelihood"
                      : "Surprise, entropy, cross-entropy"
          }
        />
        <div className="math-widget">
          {topic === "vectors" && (
            <>
              <VectorChip
                label="a  features"
                values={[ax, ay]}
                editable
                tone="forward"
                onChange={(index, next) => setState(index === 0 ? { ax: next } : { ay: next })}
              />
              <VectorChip
                label="b  weights"
                values={[bx, by]}
                editable
                tone="attention"
                onChange={(index, next) => setState(index === 0 ? { bx: next } : { by: next })}
              />
              <VectorPlane ax={ax} ay={ay} bx={bx} by={by} />
              <ul className="gw-legend" aria-hidden="true">
                <li><i className="is-forward" />a</li>
                <li><i className="is-attention" />b</li>
                <li><i className="is-forward is-dashed" />projection of a on b</li>
                <li><i className="is-loss is-dotted" />perpendicular drop</li>
              </ul>
              <p className="lab-note">
                The dotted drop runs from the tip of a to the line of b, at a right angle. Its foot is the
                projection ((a·b)/|b|²) b, so a·b = |b| × (signed length of that projection). Alignment,
                not magic, is why the product is large; opposite arrows make it negative.
              </p>
              <table className="tg-board fit-table">
                <thead>
                  <tr>
                    <th scope="col">i</th>
                    <th scope="col">aᵢ</th>
                    <th scope="col">bᵢ</th>
                    <th scope="col">aᵢ bᵢ</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>1</td>
                    <td>{ax.toFixed(2)}</td>
                    <td>{bx.toFixed(2)}</td>
                    <td>{(ax * bx).toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>2</td>
                    <td>{ay.toFixed(2)}</td>
                    <td>{by.toFixed(2)}</td>
                    <td>{(ay * by).toFixed(3)}</td>
                  </tr>
                  <tr className="is-leader">
                    <th scope="row">sum</th>
                    <td colSpan={2}>a · b</td>
                    <td>{dot.toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="metric-row">
                <Metric label="a · b" value={dot.toFixed(3)} tone="forward" />
                <Metric label="cos θ" value={cosine.toFixed(3)} />
                <Metric label="|a|, |b|" value={`${lengthA.toFixed(2)}, ${lengthB.toFixed(2)}`} />
                <Metric label="Distance |a − b|" value={distanceAB.toFixed(2)} />
              </div>
            </>
          )}
          {topic === "matrices" && (
            <>
              <div className="myn-sliders">
              <RangeControl label="m00" min={-2} max={2} step={0.1} value={m00} format={(v) => v.toFixed(1)} onChange={(value) => setState({ m00: value })} />
              <RangeControl label="m01" min={-2} max={2} step={0.1} value={m01} format={(v) => v.toFixed(1)} onChange={(value) => setState({ m01: value })} />
              <RangeControl label="m10" min={-2} max={2} step={0.1} value={m10} format={(v) => v.toFixed(1)} onChange={(value) => setState({ m10: value })} />
              <RangeControl label="m11" min={-2} max={2} step={0.1} value={m11} format={(v) => v.toFixed(1)} onChange={(value) => setState({ m11: value })} />
              <RangeControl label="b0" min={-1} max={1} step={0.1} value={b0} format={(v) => v.toFixed(1)} onChange={(value) => setState({ b0: value })} />
              <RangeControl label="b1" min={-1} max={1} step={0.1} value={b1} format={(v) => v.toFixed(1)} onChange={(value) => setState({ b1: value })} />
              </div>
              <VectorChip
                label="x"
                values={[vx, vy]}
                editable
                onChange={(index, next) => setState(index === 0 ? { vx: next } : { vy: next })}
              />
              <MatrixPlane m={[m00, m01, m10, m11]} b={[b0, b1]} v={[vx, vy]} />
              <ul className="gw-legend" aria-hidden="true">
                <li><i className="is-ink is-dashed" />square, corners (±1, ±1)</li>
                <li><i className="is-forward" />its image W·corner + b</li>
                <li><i className="is-forward is-dashed" />Wx</li>
                <li><i className="is-loss is-dotted" />+ b shift</li>
              </ul>
              <table className="tg-board fit-table">
                <thead>
                  <tr>
                    <th scope="col">out</th>
                    <th scope="col">row · x</th>
                    <th scope="col">+ b</th>
                    <th scope="col">y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>y₀</td>
                    <td>{tx.toFixed(3)}</td>
                    <td>{b0.toFixed(2)}</td>
                    <td>{y0.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>y₁</td>
                    <td>{ty.toFixed(3)}</td>
                    <td>{b1.toFixed(2)}</td>
                    <td>{y1.toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
              <div className="metric-row">
                <Metric label="W x + b" value={`[${y0.toFixed(2)}, ${y1.toFixed(2)}]`} tone="forward" />
                <Metric label="Shapes" value="(2×2)(2×1)+(2×1)" />
                <Metric label="det W · area scale" value={determinant.toFixed(2)} />
              </div>
              <p className="lab-note">
                Column 1 of W is where (1, 0) lands and column 2 is where (0, 1) lands; every other
                point is a mix of those two. det W is the factor every area is multiplied by, and a
                negative det flips the square over. At det W = 0 the square collapses onto a line.
              </p>
            </>
          )}
          {topic === "derivatives" && (
            <>
              <RangeControl
                label="x"
                min={-2}
                max={2}
                step={0.05}
                value={x}
                format={(v) => v.toFixed(2)}
                onChange={(value) => setState({ x: value })}
              />
              <RangeControl
                label="Nudge ε"
                min={0.05}
                max={0.8}
                step={0.05}
                value={nudge}
                format={(v) => v.toFixed(2)}
                onChange={(value) => setState({ nudge: value })}
              />
              <DerivativePlot x={x} eps={nudge} />
              <ul className="gw-legend" aria-hidden="true">
                <li><i className="is-ink" />f(x) = x²</li>
                <li><i className="is-gradient" />tangent at x</li>
                <li><i className="is-loss is-dashed" />secant from x to x + ε</li>
              </ul>
              <div className="metric-row">
                <Metric label="x²" value={y.toFixed(3)} />
                <Metric label="slope 2x" value={slope.toFixed(3)} tone="gradient" />
                <Metric label="secant" value={secant.toFixed(3)} />
              </div>
              <p className="lab-note">
                The solid tangent has the true slope 2x. The dashed secant climbs Δf over a run of ε,
                and its slope is exactly 2x + ε on this curve, so the gap is the nudge itself. In
                training the thing you nudge is a parameter, not this x. That is the next topic.
              </p>
            </>
          )}
          {topic === "optimization" && (
            <>
              <div className="math-surface-control">
                <SegmentedControl
                  label="Loss surface"
                  value={surface}
                  options={[
                    { value: "convex", label: "convex bowl" },
                    { value: "nonconvex", label: "two wells" },
                  ]}
                  onChange={(value) => setState({ surface: value })}
                />
              </div>
              <RangeControl
                label="Parameter w"
                min={-2}
                max={2}
                step={0.05}
                value={optW}
                format={(v) => v.toFixed(2)}
                onChange={(value) => setState({ optW: value })}
              />
              <RangeControl
                label="Step size η"
                min={0.02}
                max={0.8}
                step={0.02}
                value={stepSize}
                format={(v) => v.toFixed(2)}
                onChange={(value) => setState({ stepSize: value })}
              />
              <button type="button" className="primary-action" onClick={takeOptStep}>
                Take a downhill step
              </button>
              <LossPlot surface={surface} w={optW} eta={stepSize} trail={trail} />
              <ul className="gw-legend" aria-hidden="true">
                <li><i className="is-loss" />L(w)</li>
                <li><i className="is-gradient" />tangent, slope L′(w)</li>
                <li><i className="is-forward is-dashed" />next step, w − η L′(w)</li>
                <li><i className="is-ink is-dotted" />steps taken</li>
              </ul>
              <div className="metric-row">
                <Metric label="L(w)" value={lossAt.toFixed(3)} tone="loss" />
                <Metric label="L'(w)" value={gradAt.toFixed(3)} tone="gradient" />
                <Metric label="η L'" value={(stepSize * gradAt).toFixed(3)} />
                <Metric label="Nearest minimum" value={`w ${nearestMin.w.toFixed(2)} · L ${nearestMin.loss.toFixed(3)}`} />
              </div>
              <p className="lab-note">
                {surface === "convex"
                  ? "One valley at w = 0.80. L'' = 2 everywhere, so a step size above 0.5 overshoots the floor on every step and one above 1 would diverge; this slider stops at 0.8."
                  : `Two valleys: the left one is deeper. From w = 1.4 the steps settle at w ≈ ${minima[minima.length - 1]?.w.toFixed(2)} and stay there, because L' = 0 at the bottom of the shallower well.`}
              </p>
            </>
          )}
          {topic === "probability" && (
            <>
              <div className="prob-idea-control">
                <SegmentedControl
                  label="Probability idea"
                  value={probIdea}
                  options={PROB_IDEA_OPTIONS}
                  onChange={(value) => setState({ probIdea: value })}
                />
              </div>
              {probIdea === "counts" && (
                <>
                  <table className="tg-board fit-table">
                    <thead>
                      <tr>
                        <th scope="col">x</th>
                        <th scope="col">y=0</th>
                        <th scope="col">y=1</th>
                        <th scope="col">P(y=1|x)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>0</td>
                        <td>3</td>
                        <td>1</td>
                        <td>{pY1Given0.toFixed(2)}</td>
                      </tr>
                      <tr>
                        <td>1</td>
                        <td>1</td>
                        <td>3</td>
                        <td>{pY1Given1.toFixed(2)}</td>
                      </tr>
                    </tbody>
                  </table>
                  <p className="lab-note">
                    Those four counts are authored. P(y=1|x) is a fraction of the matching row, the
                    definition of a conditional probability from a table. E[y|x=1] = 0.75.
                  </p>
                  <RangeControl label="Logit A" min={0.1} max={6} step={0.1} value={pA} format={(v) => v.toFixed(1)} onChange={(value) => setState({ pA: value })} />
                  <RangeControl label="Logit B" min={0.1} max={6} step={0.1} value={pB} format={(v) => v.toFixed(1)} onChange={(value) => setState({ pB: value })} />
                  <BarList
                    label="Model distribution after softmax"
                    items={[
                      { id: "a", label: "P(A)", value: probA, display: `${(probA * 100).toFixed(1)}%`, tone: "forward" },
                      { id: "b", label: "P(B)", value: probB, display: `${(probB * 100).toFixed(1)}%`, tone: "attention" },
                    ]}
                    max={1}
                  />
                  <div className="metric-row">
                    <Metric label="P(A,A,B)" value={likelihood.toFixed(4)} />
                    <Metric label="ln lik" value={logLik.toFixed(3)} tone="loss" />
                  </div>
                  <p className="lab-note">
                    A classifier would compute a fresh pair of logits for each x, then softmax to
                    P(y|x). The likelihood of independent draws is still the product of those
                    conditionals — that is why training talks in likelihoods.
                  </p>
                </>
              )}
              {probIdea === "bayes" && (
                <>
                  <table className="tg-board fit-table">
                    <thead>
                      <tr>
                        <th scope="col">class y</th>
                        <th scope="col">x=0</th>
                        <th scope="col">x=1</th>
                        <th scope="col">P(x=1|y)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>0</td>
                        <td>3</td>
                        <td>1</td>
                        <td>{LIKELIHOOD_NEGATIVE.toFixed(2)}</td>
                      </tr>
                      <tr>
                        <td>1</td>
                        <td>1</td>
                        <td>3</td>
                        <td>{LIKELIHOOD_POSITIVE.toFixed(2)}</td>
                      </tr>
                    </tbody>
                  </table>
                  <RangeControl
                    label="Prior P(y=1)"
                    min={0.02}
                    max={0.98}
                    step={0.01}
                    value={prior}
                    format={(v) => v.toFixed(2)}
                    onChange={(next) => setState({ prior: next })}
                  />
                  <BarList
                    label="Weight of each way to see x = 1"
                    items={[
                      {
                        id: "positive",
                        label: "y=1 and x=1",
                        value: bay.jointPositive,
                        display: bay.jointPositive.toFixed(3),
                        detail: "prior × P(x=1|y=1)",
                        tone: "forward",
                      },
                      {
                        id: "negative",
                        label: "y=0 and x=1",
                        value: bay.jointNegative,
                        display: bay.jointNegative.toFixed(3),
                        detail: "(1 − prior) × P(x=1|y=0)",
                        tone: "attention",
                      },
                    ]}
                    max={1}
                  />
                  <div className="metric-row">
                    <Metric label="Evidence P(x=1)" value={bay.evidence.toFixed(3)} />
                    <Metric label="Posterior P(y=1|x=1)" value={bay.posterior.toFixed(3)} tone="forward" />
                    <Metric
                      label="Odds: prior × ratio"
                      value={`${bay.priorOdds.toFixed(2)} × ${bay.likelihoodRatio.toFixed(0)} = ${bay.posteriorOdds.toFixed(2)}`}
                    />
                  </div>
                  <p className="lab-note">
                    The counts are authored: four rows per class, so the table's own prior is 0.50 and the posterior is
                    the 3/4 it gives directly. The likelihood ratio, 3 to 1, never changes here. Lower the prior and
                    the same evidence lands on a lower posterior: a rare class stays rare unless the evidence is
                    overwhelming.
                  </p>
                </>
              )}
              {probIdea === "gaussian" && (
                <>
                  <table className="tg-board fit-table">
                    <thead>
                      <tr>
                        <th scope="col">sample x</th>
                        <th scope="col">x − mean</th>
                        <th scope="col">(x − mean)²</th>
                      </tr>
                    </thead>
                    <tbody>
                      {SAMPLE.map((item) => (
                        <tr key={item}>
                          <td>{item}</td>
                          <td>{(item - sampleAverage).toFixed(2)}</td>
                          <td>{((item - sampleAverage) ** 2).toFixed(2)}</td>
                        </tr>
                      ))}
                      <tr className="is-leader">
                        <th scope="row">mean · variance</th>
                        <td>{sampleAverage.toFixed(2)}</td>
                        <td>{sampleSpread.toFixed(2)}</td>
                      </tr>
                    </tbody>
                  </table>
                  <div className="myn-sliders">
                    <RangeControl
                      label="Mean μ"
                      min={-2}
                      max={2}
                      step={0.05}
                      value={gaussMu}
                      format={(v) => v.toFixed(2)}
                      onChange={(next) => setState({ gaussMu: next })}
                    />
                    <RangeControl
                      label="Variance σ²"
                      min={0.1}
                      max={2}
                      step={0.05}
                      value={gaussVar}
                      format={(v) => v.toFixed(2)}
                      onChange={(next) => setState({ gaussVar: next })}
                    />
                    <RangeControl
                      label="Probe x"
                      min={-4}
                      max={4}
                      step={0.1}
                      value={gaussX}
                      format={(v) => v.toFixed(1)}
                      onChange={(next) => setState({ gaussX: next })}
                    />
                  </div>
                  <button
                    type="button"
                    className="primary-action"
                    onClick={() => {
                      setState({ gaussMu: sampleAverage, gaussVar: sampleSpread });
                      narrate(`Fitted the bell to the sample: mean ${sampleAverage.toFixed(2)}, variance ${sampleSpread.toFixed(2)}.`);
                    }}
                  >
                    Fit μ and σ² to the sample
                  </button>
                  <GaussianPlot mean={gaussMu} variance={gaussVar} x={gaussX} sample={SAMPLE} />
                  <ul className="gw-legend" aria-hidden="true">
                    <li><i className="is-ink" />density N(μ, σ²)</li>
                    <li><i className="is-gradient is-dashed" />mean μ</li>
                    <li><i className="is-forward" />band μ ± σ</li>
                    <li><i className="is-loss" />the four sample values, as ticks</li>
                  </ul>
                  <div className="metric-row">
                    <Metric label="Density at x" value={density.toFixed(4)} tone="loss" />
                    <Metric label="z = (x − μ)/σ" value={zScore.toFixed(2)} />
                    <Metric label="Area within μ ± σ" value={`${(withinOneSigma * 100).toFixed(1)}%`} />
                  </div>
                  <p className="lab-note">
                    The mean is the center of mass; the variance is the average squared distance from it, so σ = √σ² is
                    back in the units of x. That 68% share holds for every μ and σ². A density is a
                    height: shrink σ² to 0.10 and the peak rises above 1, which no probability can.
                  </p>
                </>
              )}
              {probIdea === "logit" && (
                <>
                  <RangeControl label="Logit A" min={0.1} max={6} step={0.1} value={pA} format={(v) => v.toFixed(1)} onChange={(value) => setState({ pA: value })} />
                  <RangeControl label="Logit B" min={0.1} max={6} step={0.1} value={pB} format={(v) => v.toFixed(1)} onChange={(value) => setState({ pB: value })} />
                  <SigmoidPlot z={logitZ} />
                  <div className="metric-row">
                    <Metric label="Logit gap z = A − B" value={logitZ.toFixed(1)} />
                    <Metric label="Softmax P(A)" value={probA.toFixed(4)} tone="forward" />
                    <Metric label="Sigmoid σ(z)" value={sigmoidZ.toFixed(4)} tone="forward" />
                    <Metric label="Log-odds ln(P(A)/P(B))" value={logitOf(probA).toFixed(3)} />
                  </div>
                  <p className="lab-note">
                    With two classes, softmax depends only on the gap between the logits, and that dependence is the
                    sigmoid. Move Logit A and Logit B together and P(A) does not change. A logistic regression builds
                    one logit z = w·x + b from its features and reports σ(z) as P(y=1|x).
                  </p>
                </>
              )}
            </>
          )}
          {topic === "logarithms" && (
            <>
              <RangeControl
                label="True p"
                min={0.02}
                max={0.98}
                step={0.01}
                value={value}
                format={(v) => v.toFixed(2)}
                onChange={(next) => setState({ value: next })}
              />
              <RangeControl
                label="Model q"
                min={0.02}
                max={0.98}
                step={0.01}
                value={modelQ}
                format={(v) => v.toFixed(2)}
                onChange={(next) => setState({ modelQ: next })}
              />
              <div className="metric-row">
                <Metric label="−ln p" value={logLoss.toFixed(3)} tone="loss" />
                <Metric label="H(p)" value={entropy.toFixed(3)} />
                <Metric label="H(p,q)" value={crossEntropy.toFixed(3)} tone="loss" />
                <Metric label="KL(p‖q)" value={kl.toFixed(3)} tone={kl < 0.02 ? "forward" : "loss"} />
              </div>
              <SurprisePlot p={value} q={modelQ} />
              <ul className="gw-legend" aria-hidden="true">
                <li><i className="is-ink" />surprise −ln u</li>
                <li><i className="is-loss" />model: −ln q, −ln(1−q)</li>
                <li><i className="is-ink" />hollow rings: truth −ln p, −ln(1−p)</li>
                <li><i className="is-loss" />bar gap = KL</li>
              </ul>
              <p className="lab-note">
                Each bar stacks two weighted surprises: p × (surprise of outcome 1) plus (1 − p) ×
                (surprise of outcome 0). H(p) uses the truth's own probabilities; H(p,q) uses the
                model's. Set q = p and the bars match. A product p·p·p becomes ln p + ln p + ln p, which
                is why losses add log-probabilities.
              </p>
            </>
          )}
        </div>
      </LabSurface>

      <LabSurface label="Worked arithmetic" className="math-walk-card">
        <SurfaceHeading kicker="Words, then the live numbers" title={walkthrough.title} />
        <ol className="math-walk">
          {walkthrough.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </LabSurface>

      <LabSurface label="Live formula" className="math-formula-card">
        <SurfaceHeading kicker="Closed form for the focused topic" title="The number the widget actually evaluates" />
        <div className="math-formula">
          <FormulaWithValues
            label={formula.label}
            expression={formula.expression}
            result={formula.result}
            detail={formula.detail}
            tone={
              topic === "derivatives" || topic === "optimization"
                ? "gradient"
                : topic === "logarithms"
                  ? "loss"
                  : "attention"
            }
          />
        </div>
      </LabSurface>
    </div>
    </div>
  );
}
