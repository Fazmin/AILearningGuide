import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Loss surface": {
    title: "The whole loss, over both parameters",
    summary:
      "The mean squared error of the line ŷ = w·x + b on eight fixed points, drawn for every (w, b) in view as contour lines, with the optimizer's path, the current point, and the steepest-downhill arrow −∇L. Every value is computed from the data; the surface is not a sketch.",
    whatYouSee: [
      "The horizontal axis is the slope `w` from −1.6 to 3.6; the vertical axis is the intercept `b` from −2 to 2. Both use the same scale, so a right angle on screen is a right angle in (w, b).",
      "Contour lines at loss 0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, and 32, with 0.5, 2, and 8 labelled. Shading deepens toward higher loss. The green × is the least-squares minimum: loss 0.0205 at w = 1.443, b = −0.542.",
      "A dashed ring marks the start. The solid blue line and dots are the steps taken so far; the dashed blue line is the whole 80-step run at the current settings. The orange ball is the current step.",
      "The pink arrow is −∇L at the current point, drawn at a fixed length. Its direction is the claim; `∂L/∂w` and `∂L/∂b` below give its size.",
      "Metrics `Step`, `Loss L`, `∂L/∂w`, and `∂L/∂b`. At the default start (w −1, b 1.2) they read 0, 2.511, −3.039, and −1.403. The badge names the regime: one-sided, overshoots, or diverges.",
    ],
    howItWorks: [
      "`L(w, b) = (1/8) Σ (w·xᵢ + b − yᵢ)²`, with `∂L/∂w = (2/8) Σ rᵢxᵢ` and `∂L/∂b = (2/8) Σ rᵢ`, where `rᵢ = ŷᵢ − yᵢ`. A unit test checks both against finite differences.",
      "The loss is quadratic, so every contour is an exact ellipse around the minimum. Its long axis is the shallow direction (curvature λ_min = 0.307) and its short axis the steep one (λ_max = 4.363): the valley is about 14 times more curved across than along.",
      "Plain descent: `θ ← θ − η∇L`. Momentum: `v ← βv + ∇L`, then `θ ← θ − ηv`. The full 80-step run is recomputed whenever a control changes; `Step` only chooses which point to show.",
      "A point outside the window is reported as off the map, and a run stops updating once its loss passes 10⁶.",
    ],
    controls: [
      "Click anywhere on the surface to start there; `Start w` and `Start b` on `Optimizer controls` do the same from the keyboard. Moving the start returns to step 0.",
      "Comparison worth running: at step 0 the arrow crosses the nearest contour at a right angle. Now look where it points. It misses the minimum, because the gradient is the steepest local direction, not the direction to the bottom.",
    ],
    notice: [
      "The path first drops across the narrow valley, then crawls along its floor. The crawl is slow because the floor is nearly flat: curvature 0.307 there.",
      "When the dashed run zigzags from wall to wall, η is past 1/λ_max = 0.229. When the zigzags grow, it is past 2/λ_max = 0.458.",
      "Under Momentum the path swings past the floor and curls back, and it can overshoot the minimum along the valley too.",
    ],
    limits: [
      "In this lab: two parameters, eight authored points, and the exact full-data gradient. The surface is a perfect quadratic bowl with one minimum, so it has no saddle points, plateaus, or noisy minibatch steps.",
      "In this lab: the shading is a display layer of translucent ellipses, not a colour scale to read values from. Read values from the contour labels and the metrics.",
      "In general: a network has millions to billions of parameters and a loss that is not quadratic. The landscape cannot be drawn, curvature changes from place to place, and saddle points and flat regions are the realistic obstacles rather than bad local minima.",
      "In general: low training loss is a proxy. The Training dynamics lab shows it falling while held-out loss rises.",
    ],
  },

  "Optimizer controls": {
    title: "Direction from the gradient, distance from you",
    summary:
      "Everything that shapes the run: the update rule, the learning rate η with a ruler showing where this bowl changes regime, the momentum coefficient β, which step to show, and where to start. The gradient alone picks each step's direction; these controls scale and accumulate it.",
    whatYouSee: [
      "`Optimizer` with Plain descent and Momentum.",
      "`Learning rate` from η 0.01 to 0.50, and under it a ruler coloured by regime at every slider value (one-sided, overshoots, diverges), a black marker at the current η, and the limits printed below.",
      "`Momentum β` from 0 to 0.95, shown only under Momentum.",
      "`Step` from 0 to 80, `Take one gradient step`, `Back to step 0`, and the `Start w` and `Start b` sliders.",
    ],
    howItWorks: [
      "The ruler is computed, not drawn. For each η it finds how one step multiplies the error along the steep and shallow axes. Plain descent: `1 − ηλ`; negative means overshoot, and a size above 1 means divergence. Momentum: the eigenvalues of `[[1 + β − ηλ, −β], [1, 0]]`, where a negative or complex eigenvalue means overshoot.",
      "For plain descent the boundaries are 1/λ_max = 0.229 and 2/λ_max = 0.458. Heavy-ball momentum stays stable up to 2(1 + β)/λ_max, which is 0.871 at β = 0.9 and so beyond this slider.",
      "`Take one gradient step` adds one to `Step`. The run is deterministic, so dragging `Step` back and forth replays the same path.",
    ],
    controls: [
      "Comparison worth running: set `Learning rate` to 0.05, 0.20, 0.40, and 0.48 and read `Settles by` on `Loss per step`: not within 80 steps, step 39, step 19, and never (diverges).",
      "Second comparison: at η 0.05 switch to Momentum. β 0.5 settles by step 77, β 0.8 by step 26, β 0.9 by step 46, and β 0.95 not within 80. Plain descent at 0.05 has not settled by step 80.",
    ],
    notice: [
      "The learning rate changes the length of every step and the direction of none. Momentum's first step equals the plain step, because velocity starts at zero.",
      "Momentum speeds up the slow valley floor but swings across the steep direction. There is a best β for this bowl, and it is not the largest.",
      "Setting β to 0 reproduces plain descent exactly; a unit test checks this.",
    ],
    limits: [
      "In this lab: one fixed quadratic, so the regime boundaries are exact numbers. There is no schedule, no gradient clipping, no per-parameter scaling, and no weight decay.",
      "In general: production runs use AdamW, which divides each parameter's step by a running root-mean-square of its gradient, together with a warmup-then-decay schedule and gradient-norm clipping. Curvature changes during training, so a learning rate that is stable early can become unstable later.",
    ],
  },

  "Loss per step": {
    title: "The same run, one number per step",
    summary:
      "Loss against step on a log scale for the current run: solid up to the current step, dashed for the rest of the 80 steps, with the minimum loss drawn as a floor. The metrics say whether the run settles and why.",
    whatYouSee: [
      "The horizontal axis is step 0 to 80. The vertical axis is loss on a log scale from 0.01 to 100, so each gridline is ten times the one below. The green dashed line is the minimum, 0.0205.",
      "The solid orange line is the steps taken, the dashed line is the rest of the run, and the ball on the dotted marker is the current step. A red note appears when the loss passes 100; the curve is pinned to the top from there.",
      "`Regime`; `Settles by`, the first step after which the loss stays within 0.01 of the minimum; `Steep axis ×`, the per-step factor across the valley; and `Shallow axis ×`, the factor along it. The note below explains the regime in numbers.",
    ],
    howItWorks: [
      "Each point is `L(w, b)` at that step of the run on `Loss surface`. Nothing is smoothed.",
      "Near the minimum the error along each axis is multiplied by a fixed factor every step, so on a log scale the late curve is a straight line whose slope comes from the slower factor.",
      "`Settles by` requires the loss to stay within 0.01 of the minimum through step 80. An overshooting run that dips in and swings back out is not credited early.",
      "Under plain descent `Steep axis ×` is the signed factor `1 − ηλ_max`; under Momentum both factors are the size of the dominant eigenvalue, with `swing` added when it oscillates.",
    ],
    controls: [
      "This card has no controls. `Learning rate`, `Momentum β`, and `Step` on `Optimizer controls` drive it.",
      "Comparison worth running: at η 0.05 the curve drops fast and then bends into a long shallow slope. Read `Shallow axis ×`, 0.985: that is the slope of the long tail.",
    ],
    notice: [
      "Two phases in one curve: the fast drop is the steep axis finishing, and the slow tail is the shallow axis.",
      "Between 1/λ_max and 2/λ_max the point lands on the far wall every step, yet plain descent's loss still falls on every step, because the error along each axis shrinks in size. Above 2/λ_max it eventually rises without end, even when the first steps go down.",
      "Under Momentum the curve is not monotone: at η 0.05 and β 0.95 the loss climbs from 0.236 at step 10 to 0.643 at step 20 as the velocity carries the point up the far side.",
    ],
    limits: [
      "In this lab: the loss is exact and noise-free, so the curve is smooth, and the minimum is known, so `settled` can be measured against it.",
      "In general: real loss curves are jagged because each step uses a random minibatch, the minimum is unknown, and practitioners smooth the curve and judge progress on held-out data.",
    ],
  },

  "Data and fit": {
    title: "What the loss is measuring",
    summary:
      "The eight points, the line at the current step, the residual from each point to that line, and the squared residuals whose mean is the loss. This is the error that the surface summarizes for every (w, b) at once.",
    whatYouSee: [
      "Top: x from 0 to 2 and y from −1.5 to 3. Blue dots are the eight fixed points. The purple line is ŷ = w·x + b at the current step, and the green dotted line is the least-squares line at the minimum. Orange vertical segments are the residuals ŷᵢ − yᵢ.",
      "Bottom: one bar per point with height rᵢ² and its value printed beneath, plus a dashed line at their mean; the legend prints that mean as L. At the default start the bars read 2.04, 1.00, 0.02, 0.16, 1.37, 2.31, 5.06, and 8.12, with mean 2.511.",
    ],
    howItWorks: [
      "`rᵢ = w·xᵢ + b − yᵢ` and `L = (1/8) Σ rᵢ²`. Squaring counts misses above and below the same, and lets large misses dominate: one miss of 2 costs as much as four misses of 1.",
      "The bar scale resets to the tallest bar, so compare bars within one view, not across steps.",
      "For a classifier, cross-entropy would replace rᵢ²: `−log p`, where p is the probability given to the correct class. It is near 0 for a confident correct answer and grows without bound for a confident wrong one.",
    ],
    controls: [
      "`Start w` and `Start b`, or a click on the surface, move the line at step 0. `Step` and `Take one gradient step` move it along the run.",
    ],
    notice: [
      "As descent changes w and b, the line pivots and slides, and the tallest bars shrink first. Each residual enters the gradient in proportion to its size, so the largest misses dominate the early steps.",
      "At the minimum the residuals do not vanish. Eight points do not lie on one line, so the best achievable loss is 0.0205, not 0.",
    ],
    limits: [
      "In this lab: eight authored points and a straight-line model, so squared error gives an exact bowl-shaped surface.",
      "In general: language models are trained with cross-entropy on next-token probabilities, averaged per token, and their loss is not a bowl in the parameters.",
    ],
  },
};

export default cardInfo;
