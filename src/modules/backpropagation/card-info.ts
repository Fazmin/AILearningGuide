import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Interactive backpropagation computation graph": {
    title: "Values move right, gradients move left",
    summary:
      "A one-weight computation graph you walk one edge at a time: input and weight multiply, a sigmoid squashes the product, and squared error against a target gives the loss. The first three steps fill in values; the last three multiply one local derivative per edge onto the gradient, and the tape below writes every step down.",
    whatYouSee: [
      "Six nodes, `input x`, `weight w`, `multiply z`, `sigmoid ŷ`, `target y`, and `loss L`, each showing its forward value. A value not yet computed shows `?` inside a dashed ring.",
      "Once the backward pass reaches a node, a pink pill under it shows that node's gradient: `∂L/∂L 1`, then `∂L/∂ŷ`, `∂L/∂z`, and finally `∂L/∂w` and `∂L/∂x` together.",
      "Blue labels above each edge are the forward values it carries. Once the backward pass crosses an edge, a pink `× factor` below it shows the local derivative that edge multiplies in, and a dashed pink stroke with a reversed arrow marks the gradient's path.",
      "The tape: six rows, three forward (blue edge) and three backward (pink edge), each with its formula, the numbers substituted in, and the result. Rows ahead of the current step are dimmed with `?`.",
      "At the defaults (x 0.80, w 0.60, y 0.90) the tape reads z 0.480, ŷ 0.618, L 0.0797, then ∂L/∂ŷ −0.565, ∂L/∂z −0.1333, and ∂L/∂w −0.1066, with ∂L/∂x −0.0800.",
    ],
    howItWorks: [
      "Forward: `z = x · w`, `ŷ = σ(z) = 1 / (1 + e^−z)`, `L = (ŷ − y)²`. There is no ½ in the loss, so its derivative is `2(ŷ − y)`.",
      "Backward, one factor per edge: `∂L/∂ŷ = 2(ŷ − y)`, then `∂L/∂z = ∂L/∂ŷ · ŷ(1 − ŷ)`, then `∂L/∂w = ∂L/∂z · x` and `∂L/∂x = ∂L/∂z · w`. Each pill equals the pill to its right times the factor on the edge between them.",
      "The multiply node sends a gradient to both of its inputs in the same step, and each uses the other input's forward value: `∂z/∂w = x` and `∂z/∂x = w`. That is why forward values must be kept until the backward pass reaches them.",
      "All numbers come from the SDK's `backpropSnapshot`. A unit test checks them against finite differences for both w and x across the slider ranges.",
    ],
    controls: [
      "`Backpropagation trace`, `Advance one edge`, or a click on an edge chooses the step. During the backward pass, clicking the `target y` edge changes nothing and announces that the target gets no gradient.",
      "`Input x`, `Weight w`, `Target y`, and `Learning rate` live on `Backpropagation controls and chain rule`. Changing any of the first three restarts the trace at Multiply.",
      "Comparison worth running: at the defaults, walk all six steps and note each `× factor`. Then set `Input x` 1.50, `Weight w` 3.00, and `Target y` 0.10 and walk again: the sigmoid factor falls to 0.011 and `∂L/∂w` to 0.029, even though the loss, 0.790, is ten times larger.",
    ],
    notice: [
      "A large error does not guarantee a large gradient. When ŷ is near 0 or 1, `ŷ(1 − ŷ)` collapses and takes the whole product with it. That is sigmoid saturation, and it is one layer of the vanishing-gradient problem. The Gradient flow through depth card stacks twelve of them.",
      "`∂L/∂x` is computed even though x is data and never updated. In a deeper network the same branch is how the gradient reaches earlier layers.",
      "Nobody in the graph knows the whole formula. Each node only multiplies what arrives from its right by its own local derivative.",
    ],
    limits: [
      "In this lab: one scalar weight, one example, squared error, and hand-written local derivatives. No node feeds two others, so no gradients have to be summed.",
      "In this lab: the graph is a picture of a three-operation chain. A real network's graph has millions of such edges, and frameworks store its activations so the backward pass never recomputes the forward one.",
      "In general: when a value feeds several later operations, the gradients arriving from each path are added. Weights shared across positions or time steps collect a sum of contributions this way.",
      "In general: vanishing and exploding gradients are this same product becoming tiny or huge as more local factors are multiplied in. Residual connections, normalization, and careful initialization exist to keep it well behaved, and the Gradient flow through depth card measures what two of them do.",
    ],
  },

  "Loss versus weight tangent chart": {
    title: "The backward pass is a claim about this slope",
    summary:
      "The loss of this one-weight network for every weight from −3 to 3, with the current weight, a tangent whose slope is the backward pass's `∂L/∂w`, and a finite-difference check beside it. If the two slopes ever disagreed, the chain rule on the graph would be the thing that was wrong.",
    whatYouSee: [
      "The horizontal axis is `weight w` from −3 to 3. The vertical axis is `loss L` from 0 to the tallest point of the current curve, labelled at 0, half, and full; the scale refits when `Input x` or `Target y` changes.",
      "The loss curve sampled at 121 weights, a ball at the current weight with a dashed guide to the axis, and a dashed tangent through the ball.",
      "A green dashed line where ŷ = y, the weight at which the loss reaches zero, when it lies in view. At the defaults it sits at w 2.75.",
      "Metrics `Loss`, `Analytic slope` from the chain rule, and `Numeric check` from a central difference.",
    ],
    howItWorks: [
      "Each plotted point reruns the forward pass at that weight: `z = x · w`, `ŷ = σ(z)`, `L = (ŷ − y)²`. The curve belongs to the current x and y.",
      "`Analytic slope` is `2(ŷ − y) · ŷ(1 − ŷ) · x`, the same product as the tape. `Numeric check` is `(L(w + ε) − L(w − ε)) / 2ε` with ε = 10⁻⁵.",
      "The zero-loss weight is `w* = logit(y) / x = ln(y / (1 − y)) / x`. When y is 0 or 1 it is infinitely far away, so no line is drawn.",
      "The tangent is `L + (∂L/∂w)(w′ − w)` for w′ within 0.72 of the current weight, drawn through the same scales as the curve.",
    ],
    controls: [
      "This card has no controls of its own. `Weight w` moves the ball, `Input x` and `Target y` reshape the curve, and `Apply w ← w − η∂L/∂w` takes one step.",
      "Comparison worth running: drag `Weight w` across the range and watch `Analytic slope` and `Numeric check` agree in every printed digit. Then press the update button and watch the ball move downhill along the tangent.",
    ],
    notice: [
      "Where the sigmoid saturates, with x · w far from zero, the curve flattens and both slopes approach zero. That is not a minimum: at `Input x` 1.50 and `Weight w` −3.00 the loss is 0.790 and the slope only −0.029. It is a vanishing gradient.",
      "The ball only reaches the green line if you take enough steps or drag it there. Each step's size is η times the slope, so steps shrink as the curve flattens.",
    ],
    limits: [
      "In this lab: the curve is the exact loss of this network, sampled densely, with nothing fitted or interpolated beyond straight segments between samples.",
      "In this lab: the vertical scale refits to the tallest point, so the tangent's drawn angle is a faithful sign and rough steepness rather than a calibrated angle.",
      "In general: you cannot plot loss against a real model's parameters. This tangent stands in for one component of a gradient vector with millions of entries, all produced by the same backward pass.",
    ],
  },

  "Backpropagation controls and chain rule": {
    title: "Change a number, then spend the gradient",
    summary:
      "The four sliders that define the graph, the chain-rule product evaluated on those numbers, and one gradient-descent update that moves the weight. This is where the derivative stops being a picture and becomes a step.",
    whatYouSee: [
      "`Input x` from 0.10 to 1.50, `Weight w` from −3.00 to 3.00, `Target y` from 0.00 to 1.00, and `Learning rate` from 0.05 to 2.00.",
      "A `Chain rule` row printing the three factors `∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w`, their product `∂L/∂w`, and the relative error against the numeric check.",
      "A button `Apply w ← w − η∂L/∂w` that performs one update and announces the old weight, the new weight, and the new loss.",
    ],
    howItWorks: [
      "The factors are `2(ŷ − y)`, `ŷ(1 − ŷ)`, and `x`. Their product is the same number the graph's `∂L/∂w` pill, the tape, and the tangent use.",
      "The update is `w ← w − η · ∂L/∂w`, clamped to the slider's range of −3 to 3. There is no momentum and no clipping.",
      "Moving `Input x`, `Weight w`, or `Target y` restarts the trace at Multiply, so the graph never shows a gradient computed from stale values. `Learning rate` only scales the next press.",
      "The relative error is `|analytic − numeric| / max(|analytic|, |numeric|)`. At the defaults it reads 9.2e-10 percent: the two agree to about eleven significant digits.",
    ],
    controls: [
      "`Input x` appears twice: inside the forward product z = x · w, and again as the last chain-rule factor `∂z/∂w = x`.",
      "`Weight w` is the parameter. Drag it, or press the update button and let the gradient move it.",
      "`Target y` is the label. Moving it changes `∂L/∂ŷ` at once and reshapes the loss curve.",
      "Comparison worth running: at the defaults, one press at η 0.70 moves w from 0.600 to 0.675 and the loss from 0.0797 to 0.0720. Now set `Input x` 1.50, `Weight w` 3.00, and `Target y` 0.10: the loss is 0.790, yet the gradient is only 0.029 and the step barely moves.",
    ],
    notice: [
      "A negative `∂L/∂w` means the update increases the weight. The minus sign in the update is the whole of gradient descent.",
      "If you set `Target y` equal to the current prediction, `Loss`, `Analytic slope`, and the next update all become zero.",
      "The relative error grows when the gradient itself is near zero, because the finite difference then measures mostly rounding error. That is a limit of the check, not of the chain rule.",
    ],
    limits: [
      "In this lab: one parameter, one example, an exact gradient, and a learning rate you choose by hand.",
      "In general: automatic differentiation builds this product for every parameter from a tape of local derivatives recorded during the forward pass, which is why frameworks only ask you to write the forward pass.",
    ],
  },

  "Gradient flow through depth": {
    title: "How much gradient survives a deep stack",
    summary:
      "A multilayer perceptron with 1 to 12 hidden layers of 16 units and one sigmoid output is backpropagated once, at its random starting weights, on a fixed batch of 32 examples. Each bar is the exact norm of the loss's gradient with respect to one layer's weights, so you can watch the product of local derivatives shrink, hold steady, or blow up as you stack layers.",
    whatYouSee: [
      "`Network depth` (1 to 12), `Activation` (sigmoid, tanh, ReLU), `Initial weight scale` (small with standard deviation 0.1, Glorot, He, large with standard deviation 1.0), and `Skip connections` (off, on).",
      "One bar per hidden layer, Layer 1 nearest the input and the last layer nearest the output, with its gradient norm printed beside it. Bar length is the number of decades above 1e-13, up to 1e7, so equal steps mean equal ratios, not equal differences.",
      "Four metrics: the norm at Layer 1, the norm at the last layer, their ratio, and Loss at the start. A badge in the heading repeats the ratio.",
      "A collapsible table of the same numbers, with two more columns: the mean slope f′(z) of each layer's units over the batch, and the root-mean-square of the layer's output.",
    ],
    howItWorks: [
      "Forward: `h₁ = f(W₁x)`, then `h_l = f(W_l h_(l−1))`. With skip connections on, every layer after the first computes `h_l = h_(l−1) + f(W_l h_(l−1))`. The output is `σ(v·h_L)` and the loss is the mean binary cross-entropy over 32 fixed points in 16 dimensions.",
      "Backward: `∂L/∂logit = p − y`. Each layer multiplies the arriving gradient by `f′(z)` and then by `W_lᵀ`. With skips on the identity path adds the arriving gradient back, `∂L/∂h_(l−1) = ∂L/∂h_l + W_lᵀδ_l`. The gradient on `W_l` is `δ_l` times `h_(l−1)`, averaged over the batch.",
      "Weights are normal with standard deviation `√(2 / (fan-in + fan-out))` for Glorot, `√(2 / fan-in)` for He, and a fixed 0.1 or 1.0 for small and large. Biases are zero and left out of the norms. One seed draws the batch and the weights.",
      "A unit test checks every layer's gradient, and the output weights', against finite differences for each activation with and without skips.",
    ],
    controls: [
      "`Network depth`, `Activation`, `Initial weight scale`, and `Skip connections`. Nothing here moves the other cards.",
      "Comparison worth running: sigmoid, Glorot, skips off, depth 1 to 12. Layer 1's norm falls from 1.5e-1 to 2.6e-8 while the last layer's stays near 0.2. Then turn skips on, and then switch Activation to tanh.",
    ],
    notice: [
      "At the defaults (depth 6, sigmoid, Glorot) Layer 1's norm is 1.4e-4 and Layer 6's is 2.3e-2. At depth 12 each layer on the way back multiplies the gradient by about 0.24 on average: a sigmoid slope of at most 0.25, which the table's mean slope column shows, times a weight gain near one.",
      "Sigmoid vanishes at every starting scale: the first-to-last ratio is below 0.004 at all four, at every seed from 1 to 7. Glorot gives 1.2e-7 at depth 12 and He 2.3e-6.",
      "tanh with Glorot or He weights holds Layer 1 within a factor of three of the last layer. With large weights tanh saturates yet the gradient grows toward the input, more than ten times larger at Layer 1 at every seed.",
      "With ReLU the starting scale sets every layer's size: Layer 1's norm is 2.4e-7, 2.0e-2, 1.3, and 3.5e5 for small, Glorot, He, and large at seed 7, in that order at every seed.",
      "Skips on, sigmoid, Glorot, depth 12: Layer 1's norm rises from 2.6e-8 to 0.24 and the ratio from 1.2e-7 to 0.077, while Loss at the start rises from 0.90 to 7.30 because the additions make activations grow.",
    ],
    limits: [
      "In this lab: one random draw of the weights and batch, a 16-wide network on 32 synthetic points, no normalization layers, no training step, and weight-matrix norms only. Other seeds change the numbers but not the ordering in any comparison the lesson makes; tests cover seeds 1 to 7.",
      "In this lab: the loss scale, optimizer, and mini-batch noise are absent. A real run's gradient also depends on them, and it changes as training proceeds.",
      "In general: transformers and deep convolutional networks pair skip connections with normalization layers, which this card omits, so it shows what skips do alone, including the growth they cause. Gradient clipping, adaptive optimizers such as Adam, and mixed precision change what a gradient's size means in practice.",
    ],
  },
};

export default cardInfo;
