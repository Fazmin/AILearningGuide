import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Decision boundary and controls": {
    title: "The line where z = 0, and every knob that moves it",
    summary:
      "The input square shaded by the neuron's output at each point, with the line z = 0, the weight vector w drawn perpendicular to it, and all six controls. Every mark is computed from the same `z = w₁x₁ + w₂x₂ + b` the Neuron equation card prints.",
    whatYouSee: [
      "A 30 × 30 grid of cells over x₁ and x₂ from 0 to 1, shaded by the current activation's output: graded for Sigmoid, a grey zero side and a growing green side for ReLU, two flat halves for Step. Green is z > 0 and red is z < 0.",
      "A solid dark line, the decision boundary z = 0, and thin dashed lines at z = ±1 and ±2. They are parallel and spaced 1 / ‖w‖ apart.",
      "A blue arrow labelled w, starting on the boundary. It is perpendicular to the line and points toward growing z. Its drawn length grows with ‖w‖ but is scaled for display.",
      "The probe (dark dot) at `Input x₁`, `Input x₂`, and a dotted segment from it to the nearest point on the line.",
      "With `Corner task` on, the four corners (0,0), (0,1), (1,0), (1,1): a filled circle wants output 1, an open square wants 0, and each carries ✓ or ✗ for the current prediction.",
      "`‖w‖`, `Distance to boundary`, and `Corners correct` under the plane, plus a note that changes with the task.",
    ],
    howItWorks: [
      "Each cell's centre is evaluated with `z = w₁x₁ + w₂x₂ + b` and then `f(z)`. The boundary is where z = 0, clipped to the square; it disappears when both weights are 0.",
      "The gradient of z with respect to the input is (w₁, w₂), which is why w is perpendicular to every level line and why the z = k lines are 1 / ‖w‖ apart.",
      "`Distance to boundary` is `z / ‖w‖`, the signed Euclidean distance from the probe to the line, positive on the side w points to.",
      "A corner counts as predicted 1 when its z is strictly positive, the same rule as Step. For Sigmoid that is σ(z) > 0.5; for ReLU, an output above 0.",
      "An exhaustive search over the slider grid (steps of 0.1) finds settings that score 4 / 4 on AND and OR, and none better than 3 / 4 on XOR. That search is a unit test.",
    ],
    controls: [
      "`Activation` (Sigmoid, ReLU, Step) changes only the shading and the output. The line, the arrow, and the corner marks do not move, because they depend on the sign of z alone.",
      "`Input x₁` and `Input x₂` (0 to 1) move the probe only. They are data, not parameters.",
      "`Weight w₁` and `Weight w₂` (−2 to 2) turn the line: sweep `Weight w₁` from −2 to 2 and the arrow swings with it, always perpendicular.",
      "`Bias b` (−2 to 2) slides the line without turning it. At w = (1.2, −0.8), a bias above 0.8 or below −1.2 pushes the line out of the square.",
      "`Corner task` (Off, AND, OR, XOR) overlays the four truth-table corners and scores them.",
    ],
    notice: [
      "Scaling w₁, w₂ and b together keeps the line in place: at (1.0, −0.5, −0.2) and at (2.0, −1.0, −0.4) the probe's `Distance to boundary` is 0.291 in both, while z doubles from 0.325 to 0.650.",
      "Scaling only the weights does move the line, because b no longer scales with them.",
      "At the defaults AND and OR both score 3 / 4, and XOR 2 / 4. AND reaches 4 / 4 at w = (1, 1), b = −1.5; OR at w = (1, 1), b = −0.5. XOR never passes 3 / 4.",
      "The regions are always two half-planes. No setting of the three sliders bends the line.",
    ],
    limits: [
      "In this lab: nothing is trained. You set the weights by hand, and the four corners are a truth table, not a dataset with a loss.",
      "In this lab: inputs stay in 0 to 1 and the parameters in −2 to 2, so z stays between −6 and 6.",
      "In general: a real boundary lives in as many dimensions as there are inputs, so it can be drawn only for two or three of them.",
      "In general: one neuron inside a trained network is one of many half-planes that later layers combine. Reading its boundary alone over-explains the model.",
    ],
  },

  "Neuron equation": {
    title: "One weighted sum, then one activation",
    summary:
      "The forward pass of a two-input neuron with every intermediate number shown: each input times its weight, the sum plus the bias, and the activation of that sum.",
    whatYouSee: [
      "Two input circles with their values. Each line into the sum carries `× w = product`: the weight and that input's contribution to z.",
      "Line thickness grows with |w|. Positive weights are solid green lines; negative weights are dashed red lines, so sign is not shown by colour alone.",
      "The z circle holding the weighted sum, a dotted bias line from below, an activation box with a sketch of the chosen function, and the output a.",
      "Two formula rows: `Weighted sum`, written out with the current numbers, and `Activation`, naming the function and applying it.",
    ],
    howItWorks: [
      "`z = x₁ × w₁ + x₂ × w₂ + b`, evaluated exactly as printed. At the defaults: 0.70 × 1.2 + 0.35 × (−0.8) + (−0.1) = 0.840 − 0.280 − 0.100 = 0.460.",
      "Sigmoid computes `1 / (1 + e^−z)`, ReLU computes `max(0, z)`, and Step returns 1 when z > 0 and 0 otherwise, including at z = 0.",
      "Written for any width this is `z = w · x + b`, the dot product that appears in every attention projection and MLP later in the course.",
    ],
    controls: [
      "This card has no controls. The sliders and `Activation` on Decision boundary and controls drive it.",
      "Comparison worth running: with the default weights and bias, set `Input x₁` to 0.75 and `Input x₂` to 1.00 so z reads 0.000. Sigmoid gives 0.500, ReLU 0.000, Step 0.",
    ],
    notice: [
      "A negative weight turns a large input into a negative contribution. At the defaults x₂ subtracts 0.280 from z.",
      "Switching `Activation` changes a but never z: the three functions read the same weighted sum.",
      "Only w₁, w₂ and b are parameters. x₁ and x₂ are one example's features.",
    ],
    limits: [
      "In this lab: two inputs and hand-set parameters. There is no data, loss, or update rule.",
      "In general: a weight is not an importance score. Rescale an input feature and its weight compensates with no change in the neuron's behaviour.",
      "In general: production hidden layers mostly use GELU or SiLU, smooth relatives of ReLU. Sigmoid survives at binary outputs and inside gates.",
    ],
  },

  "Activation curve": {
    title: "Where this z sits on f, and how steep f is there",
    summary:
      "The chosen activation plotted for z from −6 to 6, the lab's full reachable range, with the current z marked and its tangent drawn. The slope readout is the number backpropagation would multiply by at this point.",
    whatYouSee: [
      "The selected function as a solid violet curve, and the other two as faint dashed curves for comparison. Step's curve breaks at the jump rather than drawing a vertical line.",
      "A dotted vertical guide and a dot at the current z, and a short pink tangent line whose steepness is f′(z). Step has no tangent.",
      "For Sigmoid, two shaded bands beyond z = ±2.89 labelled `slope < 0.05`: the saturated regions.",
      "`z`, `f(z)`, and `Slope f′(z)`. The slope turns red when it is below 0.05.",
    ],
    howItWorks: [
      "Sigmoid's slope is `σ(z)(1 − σ(z))`: 0.25 at z = 0, 0.2372 at the default z = 0.460, and 0.0025 at z = 6.",
      "ReLU's slope is 1 for z > 0 and 0 for z < 0. At exactly 0 the kink has no derivative; training libraries use 0 there by convention.",
      "Step's slope is 0 everywhere except the jump, where no derivative exists, so the readout shows 0.0000, or `none at z = 0` exactly on the jump. ReLU's kink at z = 0 reads the same way.",
    ],
    controls: [
      "`Activation` switches the solid curve.",
      "Push z to its extreme with `Input x₁` and `Input x₂` at 1 and `Weight w₁`, `Weight w₂`, `Bias b` at 2: z = 6, Sigmoid reads 0.998, and its slope 0.0025.",
    ],
    notice: [
      "The three functions disagree most near z = 0 and least far from it, except that ReLU keeps climbing.",
      "In the shaded bands Sigmoid barely moves: pushing z a further 1 unit outward shifts the output by less than 0.05.",
      "A slope of 0 means gradient descent gets no signal through this unit at this input, which is the whole case against Step.",
    ],
    limits: [
      "In this lab: the plot shows one neuron's activation for one z. No gradient flows anywhere, because nothing is trained.",
      "In general: saturation in deep sigmoid stacks multiplies many small slopes together, which is one reason gradients vanish. Normalization and careful initialization keep z in the responsive range.",
    ],
  },
};

export default cardInfo;
