import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const TOPICS = ["vectors", "matrices", "derivatives", "optimization", "probability", "logarithms"] as const;
const SURFACES = ["convex", "nonconvex"] as const;
const PROB_IDEAS = ["counts", "bayes", "gaussian", "logit"] as const;

const initialState: ModuleState = {
  topic: "vectors",
  ax: 1,
  ay: 0.4,
  bx: 0.6,
  by: 1.1,
  m00: 1,
  m01: 0.4,
  m10: 0,
  m11: 1,
  vx: 1,
  vy: 0.2,
  b0: 0.2,
  b1: 0.1,
  x: 1.2,
  nudge: 0.4,
  pA: 2.4,
  pB: 1.1,
  value: 0.25,
  modelQ: 0.6,
  surface: "convex",
  optW: -1.4,
  stepSize: 0.15,
  probIdea: "counts",
  prior: 0.5,
  gaussMu: 0,
  gaussVar: 1,
  gaussX: 1,
};

const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;
const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-30-math-you-need",
  slug: "math-you-need",
  title: "The math you need",
  group: "groundwork",
  order: 2,
  icon: "FunctionSquare",
  accent: "#2a8f7a",
  prerequisites: ["module-29-what-ai-is"],
  estimatedMinutes: 16,
  steps: [
    "Choose a topic",
    "Align two vectors",
    "Shrink the nudge",
    "Compare two loss surfaces",
    "Update a belief",
    "Read a Gaussian",
    "Bridge a logit",
    "Move q onto p",
    "Check the arithmetic",
  ],
  stepInstructions: [
    "Set Topic to vectors, then to each of the other names, and watch only one refresher stay focused.",
    "On vectors, edit a and b until the arrows oppose and a · b turns negative; then switch Topic to matrices and move m01 to see where the square lands.",
    "Switch Topic to derivatives and shrink Nudge ε, watching the secant slope close in on the tangent slope 2x.",
    "On optimization, set Loss surface to two wells, put Parameter w near 1.4, press Take a downhill step, then repeat from −1.4 and on the convex bowl.",
    "On probability, set Probability idea to Bayes' rule and lower Prior P(y=1); watch the posterior fall while the likelihoods stay put.",
    "Set Probability idea to mean and variance, press Fit μ and σ² to the sample, then raise Variance σ² and watch the bell widen and its peak fall.",
    "Set Probability idea to logit to probability and raise Logit A and Logit B by the same amount; P(A) stays put because only the gap z matters.",
    "On logarithms, move Model q onto True p and watch H(p,q) fall to H(p) and KL reach zero.",
    "Read Worked arithmetic and recompute one line with the live numbers, then check that Live formula prints the closed form this page evaluates.",
  ],
  stateVersion: 3,
  tagline: "Meet each idea four ways — words, a picture, a tiny numeric walk-through, then the closed form — so later labs can point at one mechanism.",
  objectives: [
    "Read a dot product as a weighted sum of features and a matrix as the map behind Wx+b",
    "Say what a derivative, a gradient step, a conditional probability, and a logarithm each change in a calculation",
    "Distinguish a convex bowl from a tilted double well without treating local minima as mythology",
    "Use Bayes' rule to turn a prior and a likelihood into a posterior, read a Gaussian's mean and variance, and turn a logit into a probability with the sigmoid",
  ],
  glossary: [
    {
      term: "Vector",
      definition: "An ordered list of numbers treated as a point or an arrow. In these labs the list is the whole object; there is no extra hidden meaning in the coordinates. The distance between two vectors is the length of their difference, the straight-line gap between their tips.",
    },
    {
      term: "Dot product",
      definition: "The sum of paired coordinates, equal to the product of the two lengths times the cosine of the angle between them. Read as features times weights, it is the linear score before a bias.",
    },
    {
      term: "Matrix",
      definition: "A grid of numbers that transforms a vector by mixing its coordinates. Each output coordinate is a dot product of one row with the input. The same grid can also store a batch of examples as rows.",
    },
    {
      term: "Linear layer",
      definition: "The map y = Wx + b. W is a matrix of weights, x is the incoming vector, b is a bias. It is affine unless b = 0. A later nonlinearity is what lets a stack bend.",
    },
    {
      term: "Derivative",
      definition: "The local slope of a function: how much the output changes when the input moves a tiny step. For f(x)=x² the finite difference across ε is exactly 2x+ε, which meets 2x as ε shrinks.",
    },
    {
      term: "Gradient",
      definition: "The vector of partial derivatives of a scalar. It points in the direction of steepest increase. Gradient descent steps the opposite way.",
    },
    {
      term: "Gradient descent",
      definition: "The update θ ← θ − η ∇L(θ). η is a step size you choose, and too large a step can jump a valley. On a convex bowl every downhill path reaches the one minimum; on a double well a start can settle in a shallower local minimum. The 1-D toy on this page is honest about that and is not a picture of a deep net.",
    },
    {
      term: "Probability",
      definition: "A non-negative number assigned to an outcome so the outcomes in a complete set sum to one. Conditional probability P(y|x) is that rule after you restrict to a given x.",
    },
    {
      term: "Likelihood",
      definition: "The probability a model assigns to the data you actually observed, as a function of its parameters. Independent draws multiply. Logs turn that product into a sum.",
    },
    {
      term: "Bayes' rule",
      definition: "Posterior = prior × likelihood / evidence, written P(y|x) = P(y) P(x|y) / P(x). The prior is the belief about y before seeing x, the likelihood is how probable x is if y is true, and the posterior is the updated belief. In odds form, posterior odds = prior odds × likelihood ratio, so a rare class needs strong evidence. Naive Bayes multiplies such likelihoods feature by feature.",
    },
    {
      term: "Mean and variance",
      definition: "The mean is the average of a set of numbers, the center of a distribution. The variance is the average squared distance from the mean, so its square root, the standard deviation σ, is in the units of the data. A Gaussian is the bell curve fixed by these two numbers: μ sets where it is centered and σ² sets how wide. Its height is a density, not a probability; probability is area.",
    },
    {
      term: "Sigmoid and logit",
      definition: "The sigmoid σ(z) = 1/(1 + e^−z) maps any real number, called a logit, to a probability between 0 and 1, with σ(0) = 0.5. The logit is the log-odds ln(p/(1−p)), so the two undo each other. Softmax over two classes is a sigmoid of the logit gap. Logistic regression reports σ(w·x+b) as P(y=1|x).",
    },
    {
      term: "Logarithm",
      definition: "The exponent that produces a number from a fixed base. Logs turn products into sums, which is why loss functions add log probabilities instead of multiplying tiny fractions.",
    },
    {
      term: "Cross-entropy",
      definition: "The expected surprise of a true distribution p if you announced q: −Σ p ln q. It equals the entropy of p only when q = p. A confident wrong q is very expensive.",
    },
  ],
  references: [
    {
      authors: "Stephen Boyd and Lieven Vandenberghe",
      title: "Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares",
      source: "Cambridge University Press, free to read online",
      year: 2018,
      url: "https://web.stanford.edu/~boyd/vmls/",
      note: "Section 1.4 reads the inner product of a weight vector and a feature vector as a score, the same features-times-weights reading as this lab's dot product. Later chapters cover distance and angle, matrix-vector multiplication, and affine functions like Wx + b.",
    },
    {
      authors: "Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong",
      title: "Mathematics for Machine Learning",
      source: "Cambridge University Press, free to read online",
      year: 2020,
      url: "https://mml-book.github.io/",
      note: "One book for most of this page: dot products, lengths and angles in Chapter 3, the derivative as a shrinking difference quotient in Chapter 5, and the Gaussian in Chapter 6. Chapter 7 opens on a curve with a global and a local minimum, much like the two-well surface, and warns that too large a step can overshoot.",
    },
    {
      authors: "Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind",
      title: "Automatic Differentiation in Machine Learning: a Survey",
      source: "Journal of Machine Learning Research 18(153), 1–43",
      year: 2018,
      url: "https://jmlr.org/papers/v18/17-468.html",
      note: "Explains how training software computes exact gradients. Section 2.1 shows why the nudge-and-divide slope from the derivatives widget is a poor fit for training: it needs one extra run of the function per parameter, and rounding and the leftover nudge make it inexact.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 5 builds training losses from the likelihood and takes its log so a product of tiny probabilities becomes a sum; its binary example turns a network output into a probability with the logistic sigmoid. Chapter 6 covers gradient descent, local minima, and saddle points.",
    },
    {
      authors: "Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, et al.",
      title: "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization",
      source: "Advances in Neural Information Processing Systems 27 (NIPS 2014)",
      year: 2014,
      url: "https://proceedings.neurips.cc/paper/2014/hash/04192426585542c54b96ba14445be996-Abstract.html",
      note: "Argues that when there are many parameters, the main obstacle for gradient descent is flat saddle points, not a crowd of bad local minima. It backs the lesson's point that a two-well curve is not a picture of a deep network.",
    },
    {
      authors: "Hao Li, Zheng Xu, Gavin Taylor, et al.",
      title: "Visualizing the Loss Landscape of Neural Nets",
      source: "Advances in Neural Information Processing Systems 31 (NeurIPS 2018)",
      year: 2018,
      url: "https://proceedings.neurips.cc/paper_files/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html",
      note: "Draws slices through the loss of real image classifiers. As networks get deeper without skip connections, the surface turns from nearly convex to highly chaotic, which shows how far real landscapes are from this lab's one-parameter curves.",
    },
    {
      authors: "Joseph K. Blitzstein and Jessica Hwang",
      title: "Introduction to Probability, Second Edition",
      source: "Chapman and Hall/CRC, free to read online",
      year: 2019,
      url: "https://stat110.hsites.harvard.edu/",
      note: "The textbook for Harvard's Stat 110 course. Its conditional probability chapter works through Bayes' rule and its common pitfalls, and later chapters cover variance, probability density functions, and the Normal (Gaussian) distribution.",
    },
    {
      authors: "Kevin P. Murphy",
      title: "Probabilistic Machine Learning: An Introduction",
      source: "MIT Press",
      year: 2022,
      url: "https://mitpress.mit.edu/9780262046824/probabilistic-machine-learning/",
      note: "Section 2.3 applies Bayes' rule to a COVID-19 test and shows a low base rate keeping the posterior low, the same lesson as lowering the prior here. Chapter 2 also covers the sigmoid, softmax, and the Gaussian, and Chapter 6 defines entropy, cross-entropy, and KL divergence.",
    },
    {
      authors: "Michelle McDowell and Perke Jacobs",
      title: "Meta-analysis of the effect of natural frequencies on Bayesian reasoning",
      source: "Psychological Bulletin 143(12), 1273–1312",
      year: 2017,
      url: "https://www.semanticscholar.org/paper/ef69509ad2f141e7d045c2ced471b56dc9ae198d",
      note: "Reviews 20 years of studies finding that people solve Bayes' rule problems more often when the numbers come as counts than as conditional probabilities. That is why the lesson counts messages, such as 15 of 20, before it writes the formula.",
    },
    {
      authors: "NumPy Developers",
      title: "numpy.var",
      source: "NumPy reference manual",
      year: 2026,
      url: "https://numpy.org/doc/stable/reference/generated/numpy.var.html",
      note: "The official page for computing a variance. By default it divides by the full count N, as this lab does, and its notes explain the N − 1 version that some textbooks use for a sample.",
    },
  ],
  checkpoint: [
    {
      prompt: "On vectors you edit a and b until the arrows point in roughly opposite directions. What happens to a · b and cos θ?",
      options: [
        "a · b stays positive, because it multiplies two lengths that cannot be negative",
        "cos θ stays at 1 and only the lengths change, because the arrows share a line",
        "Both fall to zero, because opposite arrows cancel each other out completely",
        "Both turn negative, because the angle between the arrows is past 90 degrees",
      ],
      answer: 3,
      explanation: "The dot product is |a||b| cos θ. Opposing arrows have an angle past 90 degrees, so the cosine is negative and the lengths, which are always positive, only scale it. The sum of paired products turns negative, which is why a weight vector can score a feature vector against its direction.",
      objective: 0,
    },
    {
      prompt: "On derivatives you shrink Nudge ε toward 0.05 and watch the secant slope. What does it do?",
      options: [
        "It stays fixed at 2x + 1, because f(x) = x² has one slope per point",
        "It drifts away from 2x, because a smaller step is a rougher estimate",
        "It turns into the value of x², because the nudge cancels in the ratio",
        "It closes in on the tangent slope 2x, leaving a gap of exactly ε",
      ],
      answer: 3,
      explanation: "For f(x) = x² the secant slope is exactly 2x + ε, so shrinking the nudge removes the gap. The derivative is that ratio in the limit as ε goes to zero. A smaller nudge gives a better local estimate here, not a rougher one.",
      objective: 1,
    },
    {
      prompt: "Training losses add log-probabilities instead of multiplying probabilities. Why?",
      options: [
        "A log caps the cost of a confident wrong answer, so one miss cannot dominate the total",
        "A log turns a product of many tiny probabilities into a sum that stays easy to add",
        "A log shrinks the loss so a model looks better without any change to the model itself",
        "Probabilities cannot be multiplied until they are first written out as percentages",
      ],
      answer: 1,
      explanation: "The likelihood of independent draws is a product of probabilities, and a long product of fractions underflows toward zero. Logs make it a sum, which is stable and easy to differentiate. The log does not cap the cost of a confident wrong answer: minus ln q grows without bound as q falls toward zero.",
      objective: 1,
    },
    {
      prompt: "On optimization with two wells, the run from w = 1.4 settles at a higher loss than the run from w = −1.4. What does that show?",
      options: [
        "Descent follows the local slope, so each start settles in the nearest well",
        "Descent always finds the global minimum, so the right-hand run must contain a bug",
        "A surface with two wells cannot be trained, so neither run means anything",
      ],
      answer: 0,
      explanation: "Gradient descent only reads the slope where it stands, so each start rolls into the nearest valley. The right-hand valley is a real local minimum, shallower than the left. That is a fact about this tilted curve. On the convex bowl every start reaches one minimum, and deep networks are not two-well curves.",
      objective: 2,
    },
    {
      prompt: "On Bayes' rule you lower Prior P(y=1) from 0.50 toward 0.10 while the count table stays fixed. What happens to P(y=1|x=1)?",
      options: [
        "It stays at 0.75, because the count table fixes the likelihoods and nothing else",
        "It rises, because a rarer class makes the same evidence more surprising",
        "It falls, because the 3-to-1 likelihood ratio now multiplies smaller prior odds",
        "It falls to the prior itself, because x carries no information once the prior changes",
      ],
      answer: 2,
      explanation: "Posterior odds are prior odds times the likelihood ratio. The ratio stays 3 to 1, but the prior odds drop from 1 to about 0.11, so the posterior falls. The table's 3/4 is the posterior only when the prior is the table's own 0.50. Evidence moves a belief; it does not replace the base rate.",
      objective: 3,
    },
    {
      prompt: "On Probability idea logit to probability you raise Logit A and Logit B by the same amount. What happens to P(A)?",
      options: [
        "It rises toward 1, because e^A grows faster than e^B as both logits increase",
        "It rises, because larger logits mean the model is more confident in its choice",
        "It stays the same, because two-class softmax depends only on the gap A − B",
        "It falls toward 0.5, because both exponentials saturate together as they grow",
      ],
      answer: 2,
      explanation: "Dividing top and bottom of the softmax by e^A leaves 1/(1 + e^−(A − B)), the sigmoid of the gap. Shifting both logits leaves the gap unchanged, so P(A) does not move. Confidence comes from the size of the gap, not from how large the logits are.",
      objective: 3,
    },
    {
      prompt: "On mean and variance you raise Variance σ² while the mean μ stays put. What happens to the bell?",
      options: [
        "It moves right, because a larger variance raises the average of the bell",
        "It gets taller, because variance measures how much probability there is in total",
        "It widens and its peak falls, since the same area spreads over a wider range",
        "It only changes the sample table, because variance is a property of the data alone",
      ],
      answer: 2,
      explanation: "The mean sets where the bell is centered and the variance sets how wide it is. The total area under a density is always 1, so spreading it wider lowers the peak. A density is a height, not a probability: shrink the variance far enough and the peak rises above 1.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      return {
        topic: asMember(parsed.topic, TOPICS, initialState.topic as string),
        ax: clampNumber(parsed.ax, 1, -2, 2),
        ay: clampNumber(parsed.ay, 0.4, -2, 2),
        bx: clampNumber(parsed.bx, 0.6, -2, 2),
        by: clampNumber(parsed.by, 1.1, -2, 2),
        m00: clampNumber(parsed.m00, 1, -2, 2),
        m01: clampNumber(parsed.m01, 0.4, -2, 2),
        m10: clampNumber(parsed.m10, 0, -2, 2),
        m11: clampNumber(parsed.m11, 1, -2, 2),
        vx: clampNumber(parsed.vx, 1, -2, 2),
        vy: clampNumber(parsed.vy, 0.2, -2, 2),
        b0: clampNumber(parsed.b0, 0.2, -1, 1),
        b1: clampNumber(parsed.b1, 0.1, -1, 1),
        x: clampNumber(parsed.x, 1.2, -2, 2),
        nudge: clampNumber(parsed.nudge, 0.4, 0.05, 0.8),
        pA: clampNumber(parsed.pA, 2.4, 0.1, 6),
        pB: clampNumber(parsed.pB, 1.1, 0.1, 6),
        value: clampNumber(parsed.value, 0.25, 0.02, 0.98),
        modelQ: clampNumber(parsed.modelQ, 0.6, 0.02, 0.98),
        surface: asMember(parsed.surface, SURFACES, initialState.surface as string),
        optW: clampNumber(parsed.optW, -1.4, -2, 2),
        stepSize: clampNumber(parsed.stepSize, 0.15, 0.02, 0.8),
        probIdea: asMember(parsed.probIdea, PROB_IDEAS, initialState.probIdea as string),
        prior: clampNumber(parsed.prior, 0.5, 0.02, 0.98),
        gaussMu: clampNumber(parsed.gaussMu, 0, -2, 2),
        gaussVar: clampNumber(parsed.gaussVar, 1, 0.1, 2),
        gaussX: clampNumber(parsed.gaussX, 1, -4, 4),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
