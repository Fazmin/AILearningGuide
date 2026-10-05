import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Topic picker": {
    title: "One topic key, six widgets",
    summary:
      "Topic is a single string — vectors, matrices, derivatives, optimization, probability, or logarithms — stored so a later lab can deep-link to one refresher. The original five names still work. Optimization is the extra surface.",
    whatYouSee: [
      "A kicker Six refreshers, one focused widget and a badge that repeats the current topic string.",
      "Topic, a six-way segmented control whose values are the exact state strings.",
      "A note that later labs can open this page on the original five names.",
    ],
    howItWorks: [
      "Module state stores `topic` as one of six literals. hydrateState rejects any other string and falls back to vectors.",
      "The other cards read that same key. Unused widgets stay unmounted until selected.",
    ],
    controls: [
      "Topic is the only control on this card.",
      "Comparison worth running: leave the vector chips alone, switch to matrices, then back. The arrows are still there because their numbers live in state.",
    ],
    notice: [
      "The default is vectors, which is what a snapshot without a topic field becomes.",
      "Do not rename the original five strings. They are the deep-link contract.",
    ],
    limits: [
      "In this lab: six closed-form refreshers. This is not a linear-algebra course and it does not train anything.",
      "In general: later labs need these ideas, plus more numerical care than a 2-D widget can show.",
    ],
  },

  "Active refresher": {
    title: "The widget the topic key selected",
    summary:
      "Each topic mounts one experiment with live arithmetic: a weighted sum and a projection, an affine layer Wx+b, a tangent versus a secant on x², a 1-D loss with a downhill step, a count table plus softmax likelihood, or surprise versus cross-entropy.",
    whatYouSee: [
      "A kicker that repeats the topic and a title that names the widget.",
      "Vectors: chips a and b, a gridded plane with arrows a and b, the angle θ, the dashed projection of a onto b and the dotted perpendicular drop, a product table with a sum row, and metrics a · b, cos θ, the lengths |a| and |b|, and the distance |a − b| between the two arrow tips.",
      "Matrices: sliders for W and b, chip x, the dashed square with corners (±1, ±1) and its shaded image under Wx + b, the input x carried to Wx (dashed) and then shifted by b (dotted), a y₀/y₁ table, and metrics W x + b, Shapes, and det W.",
      "Derivatives: sliders x and Nudge ε, a plot of x² with the tangent at x, the dashed secant to x + ε, and the run ε and rise Δf between them, plus metrics x², slope 2x, and secant.",
      "Optimization: Loss surface, Parameter w, Step size η, a downhill button, and a plot of L(w) with each minimum marked, the current w as a ball, its tangent, a dashed arrow to the next w, and a dotted path of the steps taken. Metrics L, L', ηL', and the nearest minimum.",
      "Probability: a Probability idea control with four views. Counts and softmax shows a 2×2 count table, two logits, softmax bars, and metrics P(A,A,B) and ln lik. Bayes' rule shows the class-by-x count table with P(x=1|y), a Prior P(y=1) slider, two bars for the weight of each way to see x = 1, and metrics for the evidence, the posterior, and the odds. Mean and variance shows a four-number sample table, sliders Mean μ, Variance σ², and Probe x, a Fit button, a bell-curve plot with the mean line, a μ ± σ band, and the four sample values as ticks, and metrics for the density at x, z, and the area within one σ. Logit to probability shows Logit A and Logit B, a sigmoid curve with the current gap marked, and metrics that put softmax P(A) beside σ(z) and the log-odds.",
      "Logarithms: sliders True p and Model q, metrics −ln p, H(p), H(p,q), and KL, a surprise curve −ln u with the model's two surprises (filled) and the truth's (rings), and two stacked bars, H(p) and H(p,q), whose gap is KL.",
    ],
    howItWorks: [
      "Dot product is ax·bx + ay·by. The projection is ((a·b)/|b|²) b. Cosine divides by the two lengths. Distance is the length of a − b, √((ax−bx)² + (ay−by)²).",
      "y = Wx + b. The parallelogram applies that affine map to the corners (±1, ±1). Each output coordinate is one row dotted with x, plus a bias. det W = m00·m11 − m01·m10 is the area scale factor.",
      "The curve is y = x². The printed slope is 2x. The secant is [f(x+ε)−f(x)]/ε, which equals 2x+ε on this f.",
      "Convex L(w)=(w−0.8)²+0.25. Nonconvex L(w)=(w²−1)²+0.35w+0.6. A downhill step writes w ← clip(w − η L'(w), −2, 2). Minima are the roots of L'(w) found by bisection; on the tilted curve they sit near w = −1.04 (L ≈ 0.24) and w = 0.95 (L ≈ 0.94).",
      "Empirical P(y=1|x) uses authored counts 3/1 and 1/3. Softmax is e^{logit}/Σ e^{logit}. Likelihood of A,A,B is P(A)²P(B).",
      "Bayes' rule reads its likelihoods from the same counts, P(x=1|y=1) = 3/4 and P(x=1|y=0) = 1/4, and weighs them by your prior: posterior = prior × 0.75 / (prior × 0.75 + (1 − prior) × 0.25). At the table's own prior of 0.50 that is 0.75, the conditional the table gives directly. At a prior of 0.10 it is 0.25. In odds, posterior odds = prior odds × 3.",
      "The Gaussian view fits the sample −1, 0, 1, 2 by the maximum-likelihood rule: mean = sum / n = 0.50 and variance = mean squared deviation = 1.25 (divide by n, not n − 1). The plotted density is exp(−(x − μ)² / (2σ²)) / √(2πσ²), and the band μ ± σ holds about 68.3% of the area, found by numeric integration.",
      "The logit view uses the same two logits as softmax. The gap z = A − B feeds σ(z) = 1/(1 + e^−z), which equals softmax P(A) exactly, and ln(P(A)/P(B)) returns z.",
      "Entropy and cross-entropy use the natural log on Bernoulli p and q, so the unit is nats. H(p,q) = p(−ln q) + (1−p)(−ln(1−q)); the bars stack exactly those two terms. KL is H(p,q)−H(p).",
    ],
    controls: [
      "Edit the chips or sliders that belong to the current topic. The other topic's numbers stay in state.",
      "Comparison worth running: on derivatives, shrink Nudge ε and watch the secant meet 2x. On optimization, start the two-well surface at w = 1.4 versus w = −1.4. On probability, set Probability idea to Bayes' rule and compare Prior P(y=1) at 0.50 and 0.10.",
    ],
    notice: [
      "A large dot product can come from length or from alignment. Cosine isolates the angle.",
      "A derivative is local. A local minimum on the tilted well is a real stationary point, not a story about deep nets.",
    ],
    limits: [
      "In this lab: every readout is a closed-form 1-D or 2-D formula. There is no automatic differentiation and no learned parameter except the w you step by hand. The count table, its likelihoods, and the four-number Gaussian sample are authored. The path of steps is view-only history: moving Parameter w or switching surface starts a new one. The loss plot clips its y-axis and labels the ball when L runs off the top.",
      "In general: training uses the same ideas in thousands of dimensions, with care for overflow, batching, and discrete choices that have no slope. High-dimensional loss geometry is not a 1-D double well. A real Bayes update multiplies many estimated likelihoods rather than reading one from a table, and real features are often not bell-shaped.",
    ],
  },

  "Worked arithmetic": {
    title: "Recompute the live numbers in order",
    summary:
      "Five authored sentences for the focused topic, filled with the same numbers the widget just computed. The card is a walk-through, not a second algorithm.",
    whatYouSee: [
      "A kicker Words, then the live numbers and a title that names the walk-through.",
      "An ordered list of five steps whose digits update with the sliders.",
    ],
    howItWorks: [
      "The list formats values already used to draw the widget. It does not re-derive a different formula.",
      "Vectors multiply then add. Matrices write both output rows. Derivatives compute a secant. Optimization writes the downhill update. Probability follows the Probability idea: counts then a 3-draw likelihood, a prior times a likelihood normalized into a posterior, a sample reduced to a mean and variance, or a logit gap fed through a sigmoid. Logarithms compare H(p) to H(p,q).",
    ],
    controls: [
      "This card has no controls. Topic and the widget sliders rewrite it.",
      "Comparison worth running: on vectors, flip b to oppose a and recompute the last two products by hand.",
    ],
    notice: [
      "If a printed number disagrees with the widget, the walk-through is wrong. They share one render.",
      "The last sentence on each topic is the formal name for the arithmetic you just did.",
    ],
    limits: [
      "In this lab: five sentences and two-dimensional arithmetic. It is not a proof.",
      "In general: the same order — words, a picture, a numeric line, then notation — is how later labs expect you to read a formula.",
    ],
  },

  "Live formula": {
    title: "The expression this page actually evaluates",
    summary:
      "A FormulaWithValues block reprints the closed form for the focused topic, so the widget never hides behind a metaphor. The result string is the same number the Active refresher already computed.",
    whatYouSee: [
      "A kicker Closed form for the focused topic.",
      "A formula row whose label, expression, result, and detail change with Topic and, on probability, with Probability idea.",
    ],
    howItWorks: [
      "The card does not recompute a second algorithm. It formats the values already used to draw the widget.",
      "Tone follows the topic: gradient for derivatives and optimization, loss for logarithms, attention otherwise.",
    ],
    controls: [
      "This card has no controls. Topic and the widget sliders rewrite it.",
      "Comparison worth running: set True p to 0.02 and Model q to 0.90 and read how H(p,q) and KL move.",
    ],
    notice: [
      "The expression is written with the current numbers, not with symbols only, so you can check the arithmetic.",
      "Logs are here so products become sums and surprises become a loss, not so a number merely looks smaller.",
    ],
    limits: [
      "In this lab: the formula is a display of the toy calculation. It is not a proof and not a library autodiff tape.",
      "In general: production losses add many log-probabilities under a chosen base and a chosen reduction. The mechanism is the same; the scale is not.",
    ],
  },
};

export default cardInfo;
