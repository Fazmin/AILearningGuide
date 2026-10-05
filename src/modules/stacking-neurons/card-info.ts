import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Decision surface": {
    title: "What the trained network answers, everywhere",
    summary:
      "The network's output ŷ evaluated on a 40 × 40 grid over the input square, with the training points and the straight line each first-layer unit draws. Every shade comes from a real forward pass through the weights at the selected epoch.",
    whatYouSee: [
      "Background cells shaded by ŷ: green toward 1, red toward 0, pale near 0.5. The axes are x₁ and x₂ from −1 to 1.",
      "Training points: filled circles are label 1, open squares label 0. A red ring marks a point whose prediction (ŷ ≥ 0.5 means 1) is wrong.",
      "Dashed lines h1 … hN: where each first-layer unit's weighted sum `w·x + b` is zero. Each is a straight line, like the single neuron's boundary.",
      "A cross at the probe from Network builder, and a badge with the percentage of training points classified correctly.",
    ],
    howItWorks: [
      "Each cell centre is fed forward: `h = f(W₁x + b₁)` for every hidden layer, then `ŷ = σ(w·h + c)`. f is tanh, ReLU, or the identity for `none (linear)`.",
      "The weights are those after the selected number of epochs of full-batch gradient descent on mean binary cross-entropy (learning rate 0.5, momentum 0.9), starting from Glorot-uniform weights for the chosen seed.",
      "Datasets are fixed: XOR is 48 points in four clusters at (±0.6, ±0.6) with the diagonal pair labelled 0; Ring is 24 points inside radius 0.4 labelled 1 and 36 on a band from 0.66 to 0.9 labelled 0; Checkerboard is 64 points, four in the middle half of each cell of a 4 × 4 grid, with the label alternating from cell to cell.",
    ],
    controls: [
      "Everything on Network training controls changes this picture. `Training epochs` scrubs through the saved run without retraining.",
      "`Probe x₁` and `Probe x₂` on Network builder move the cross.",
    ],
    notice: [
      "At the defaults (one layer of 2 tanh units, seed 2) the two dashed lines run parallel to the diagonal, and the red band between them holds both label-0 clusters. Every point is correct from epoch 28.",
      "With `Neurons per layer` at 1 there is a single line and the best the network reaches is 75%: one unit, however deep the stack after it, gives one straight cut.",
      "With `Hidden activation` set to `none (linear)`, the shading becomes one faint straight split and accuracy sits at 50%.",
      "On Ring, two units can only carve a wedge (86.7% at seed 2); three can enclose the disc.",
      "On Checkerboard at seed 2 with tanh, one layer of 6 units ends at 54.7%, two layers of 6 at 93.8%, and three layers of 6 at 100%. Those networks have 25, 67, and 109 parameters, so the fair test of depth against width is the Depth against width card.",
    ],
    limits: [
      "In this lab: two inputs, at most 64 points, and training accuracy only. This card never scores a point the network did not train on, so 100% here says nothing about generalization; the Held-out data card does that for the Ring.",
      "In this lab: only first-layer boundaries are drawn as lines. Deeper units have bent boundaries, which Network builder shows as tiles.",
      "In general: real inputs have hundreds or thousands of dimensions, so a decision surface can be drawn only as a slice or a projection.",
    ],
  },

  "Network training controls": {
    title: "Change the architecture, then train it for real",
    summary:
      "Dataset, hidden activation, depth, width, and a real training run you can scrub through epoch by epoch. `Training loss`, `Accuracy`, and the curve are measured on the training set after each full-batch gradient step.",
    whatYouSee: [
      "`Dataset` (XOR, Ring, Checkerboard) and `Hidden activation` (tanh, ReLU, none (linear)).",
      "`Hidden layers` (1 to 3), `Neurons per layer` (1 to 6), and `Training epochs` (0 to 300).",
      "`Train 50 epochs` and `New initialization`, then `Training loss`, `Accuracy`, and `Parameters`.",
      "A loss curve for all 300 epochs with a marker at the selected epoch, a dotted line at 0.693, and, when reached, a green dashed line at the first epoch with every point correct.",
    ],
    howItWorks: [
      "Changing dataset, activation, depth, width, or seed retrains from scratch: 300 epochs of full-batch gradient descent with momentum, `v ← 0.9v − 0.5∇L`, `θ ← θ + v`. The weights after every epoch are kept, so the epoch slider replays the run.",
      "The loss is mean binary cross-entropy `−[y ln ŷ + (1−y) ln(1−ŷ)]`. The gradient is exact backpropagation; a unit test checks it against finite differences.",
      "`Parameters` is `Σ (n_in × n_out + n_out)` over the layers: 9 for 2 → 2 → 1, 109 for three layers of 6.",
      "0.693 is ln 2, the loss of answering 0.5 for every point. A linear stack on XOR settles exactly there.",
      "`New initialization` increments the seed, which redraws the random starting weights and resets to epoch 0.",
    ],
    controls: [
      "`Neurons per layer`: at 1, XOR stalls at 75% (loss 0.482 at epoch 150). At 2, seed 2 reaches 100% by epoch 28.",
      "`Hidden activation`: `none (linear)` at any depth and width leaves XOR at 50% and loss 0.693.",
      "`Training epochs` and `Train 50 epochs`: drag to 0 to see the initial random surface (47.9% correct at the default seed), then step forward.",
      "`New initialization`: at one layer of 2, seeds 1 and 3 get stuck near 0.35 loss with 62.5% and 68.8% correct.",
    ],
    notice: [
      "The loss falls steeply then flattens, and a stuck run flattens early, well above zero.",
      "More width makes training more reliable here: with 3 or more tanh units per layer, every seed from 1 to 6 solves XOR.",
      "Depth trains too: 2 layers of 3 reach 100% by epoch 10 at seed 2. Depth also adds ways to get stuck, which `New initialization` exposes.",
    ],
    limits: [
      "In this lab: the numbers on this card are training accuracy and training loss. Held-out points are scored only on the Held-out data card, which uses its own fixed networks.",
      "In this lab: one fixed learning rate, momentum, and 300 epochs. Other settings would change which seeds succeed and how fast.",
      "In general: real networks train on mini-batches with optimizers such as Adam, and deep stacks rely on normalization and residual connections to train at all. The Backpropagation lab's Gradient flow card measures why.",
      "In general: universal approximation says one wide enough layer can approximate any continuous function on a bounded domain. It proves existence, not that gradient descent will find those weights.",
    ],
  },

  "Network builder": {
    title: "Every unit's map of the input, and the forward pass at one point",
    summary:
      "The network drawn as columns of tiles. Each tile shades one unit's output over the whole input square, edges show the real weights, and the numbers under the tiles are the forward pass at the probe.",
    whatYouSee: [
      "Columns labelled input, hidden 1 … hidden 3, output. The two input tiles are x₁ (left to right) and x₂ (bottom to top).",
      "Each hidden tile shades that unit's activation over [−1, 1]²: green for positive, red for negative with tanh or none (linear, scaled to the tile's largest value), and grey for ReLU's zero region. The output tile shades ŷ.",
      "Edges from every unit to every unit in the next column. Thickness grows with |w| (capped at |w| = 3); positive weights are solid green, negative weights dashed red.",
      "Under each tile, the unit's name and its value at the probe. The small dot inside each tile marks the probe.",
      "`Probe x₁`, `Probe x₂`, and a formula row showing the output unit's weighted sum at the probe.",
    ],
    howItWorks: [
      "Each tile is a 14 × 14 grid of forward passes through the network at the selected epoch.",
      "A first-layer tile always has one straight edge between its green and red halves, because it is one neuron reading x₁ and x₂ directly. Later layers read earlier units, so their edges bend.",
      "The formula row is the output unit exactly: `ŷ = σ(Σ wᵢ·hᵢ + c)` with the trained weights and the last hidden layer's values at the probe.",
    ],
    controls: [
      "`Probe x₁` and `Probe x₂` (−1 to 1) move the probe; every value under a tile updates.",
      "The architecture and epoch come from Network training controls.",
    ],
    notice: [
      "At the defaults the hidden layer rewrites the input: both label-0 clusters land near (h1, h2) = (1, 1), while (−0.6, 0.6) lands at (1.00, −0.97) and (0.6, −0.6) at (−0.99, 1.00).",
      "In that new space one straight rule separates the classes: the output unit, σ(−6.39·h1 − 6.36·h2 + 5.74), fires only when h1 + h2 is below about 0.9.",
      "With three layers, compare columns: first-layer tiles are split by straight lines, later tiles by bent ones. That is composition, visible.",
    ],
    limits: [
      "In this lab: tiles make each unit legible because there are only two inputs. With real inputs, a unit's response can only be probed, not drawn whole.",
      "In this lab: the edge thickness cap hides differences between very large weights; read the formula row for exact values.",
      "In general: hidden units rarely have clean single meanings. Individual units often respond to several unrelated features, the superposition problem the interpretability modules take up.",
      "In general: specialized layers share or route weights: convolutions reuse one small filter across positions, recurrent layers reuse one layer over time, and mixture-of-experts layers send each token to a few of many sublayers.",
    ],
  },

  "Depth against width": {
    title: "The same parameter budget, shaped two ways",
    summary:
      "Two fixed networks with 85 parameters each train from the same seed on the dataset and activation chosen in Network training controls: one hidden layer of 21 units, and two hidden layers of 7. Everything shown is measured from real training, so you can see whether the shape of a budget changes what is learned.",
    whatYouSee: [
      "Two small decision surfaces, one per network, at the epoch set by `Training epochs`. Shade is the output ŷ, circles are label 1 and squares are label 0, as on Decision surface.",
      "Under each surface: `Accuracy`, `Training loss`, and `Parameters` at that epoch. A badge in the heading gives the two-layer network's accuracy minus the one-layer network's, in percentage points.",
      "A chart of training accuracy for epochs 0 to 300 for both networks, the one-layer curve dashed, with a guide at the current epoch. The legend prints each curve's accuracy at epoch 300, which is the chart's text alternative.",
    ],
    howItWorks: [
      "Parameters are `Σ (n_in × n_out + n_out)`: 2 → 21 → 1 is 63 + 22 = 85, and 2 → 7 → 7 → 1 is 21 + 56 + 8 = 85.",
      "Each network trains exactly as on Network training controls: full-batch gradient descent, learning rate 0.5, momentum 0.9, Glorot-uniform starting weights (He-uniform for ReLU), 300 epochs. Only the shape differs.",
      "The two runs depend on `Dataset`, `Hidden activation`, and the seed, which `New initialization` increments. Nothing else is random.",
    ],
    controls: [
      "`Dataset`, `Hidden activation`, `Training epochs`, and `New initialization` on Network training controls drive this card.",
      "Comparison worth running: Dataset Checkerboard, Hidden activation tanh, Training epochs 300. Then switch Hidden activation to ReLU and press New initialization through a few seeds.",
    ],
    notice: [
      "On XOR and Ring both shapes reach 100% at every seed from 1 to 6, so those datasets cannot tell width from depth.",
      "On Checkerboard with tanh the one-layer network ends between 46.9% and 57.8% across seeds 1 to 6 and the two-layer network between 81.3% and 100%. At seed 2 they end at 56.3% and 87.5%.",
      "With ReLU the order turns around: the one-layer network ends between 85.9% and 100% and is level with or ahead of the two-layer network (68.8% to 100%) on every seed.",
      "With none (linear) both networks stay at 51.6%, the share of the more common label, because each is one linear map.",
    ],
    limits: [
      "In this lab: one dataset where the shapes come apart, two shapes, one learning rate, and 300 epochs. The result is about this optimizer and budget. It does not show that one wide layer cannot represent the checkerboard, which universal approximation says a wide enough layer can.",
      "In this lab: surfaces are drawn over training points only. No held-out point is scored on this card.",
      "In general: whether depth or width wins at a fixed size depends on the task, the activation, the initialization, the optimizer, and how long training runs. Depth can make some functions far cheaper to represent, but a trained network of equal size does not always show the gain.",
    ],
  },

  "Held-out data": {
    title: "Points the network trained on against points it never saw",
    summary:
      "Two fixed networks, a narrow one of 3 units and a large one of two layers of 6, train on the Ring's 60 points and are scored at every epoch on 120 fresh ring points they never train on. Label noise flips 12 of the 60 training labels, so the card can show training accuracy and held-out accuracy pulling apart.",
    whatYouSee: [
      "`Training labels`: Clean, or 20% flipped, which swaps the label of 12 of the 60 training points, the same 12 every time. Held-out labels are always correct.",
      "`Curves`: Accuracy or Loss. For each network, a chart over epochs 0 to 300 with a solid training curve, a dashed held-out curve, and a guide at the current epoch. The legend prints each curve's value at epoch 300.",
      "Under each chart: `Training accuracy`, `Held-out accuracy`, `Gap` (training minus held-out, in percentage points), and `Held-out loss` at the current epoch.",
    ],
    howItWorks: [
      "Both networks are trained as on Network training controls, always with tanh units, from the lab's seed: full-batch gradient descent, learning rate 0.5, momentum 0.9. The narrow one is 2 → 3 → 1 with 13 parameters; the large one is 2 → 6 → 6 → 1 with 67.",
      "The held-out points come from the same ring generator with a different seed: 48 inside radius 0.4 labelled 1 and 72 on the band from 0.66 to 0.9 labelled 0. They are scored with the same forward pass and cross-entropy at every saved epoch, and they never contribute to a gradient.",
      "The 12 flipped labels are chosen by a fixed seeded shuffle, so a given seed always reproduces the same run.",
    ],
    controls: [
      "`Training labels` and `Curves` on this card; `Training epochs` and `New initialization` on Network training controls.",
      "Comparison worth running: Training epochs 300, then Training labels Clean against 20% flipped, with Curves on Accuracy and then on Loss.",
    ],
    notice: [
      "With clean labels both networks reach 100% on their training points and at least 99% on the held-out points at every seed from 1 to 6, so the gap is about zero.",
      "With 20% flipped at seed 2 and epoch 300, the large network scores 96.7% on its training points and 79.2% on the held-out points; the narrow network scores 90.0% and 91.7%.",
      "The large network's training loss keeps falling, to 0.07, while its held-out loss bottoms out near 0.28 at epoch 71 and then climbs to 0.99. Training loss alone would never show it.",
      "Across seeds 1 to 6 the large network's training-minus-held-out gap is at least 15 points; the narrow network's is at most 7.",
    ],
    limits: [
      "In this lab: one fixed set of 12 flipped labels, always tanh, and one dataset. Flipping a different 12 labels changes the size of the gap, and the effect needs the 300 epochs this optimizer takes to fit them.",
      "In this lab: the held-out set is 120 fresh points from the same generator, with correct labels. Real held-out data may be noisy too, and may not come from the same process as the data a model will meet.",
      "In general: a held-out set that is used over and over to pick models stops being held out. Real label noise is rarely a clean random flip, and other defenses, such as stopping early, more data, or regularization, change when the gap opens.",
    ],
  },
};

export default cardInfo;
