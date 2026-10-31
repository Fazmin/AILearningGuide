import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Optimizer race": {
    title: "Four optimizers, one recipe",
    summary:
      "Four complete training runs of the 30 × 30 bigram table on the first six Harbor sentences, one per optimizer, each at its own learning rate, scored at every epoch on that training text and on two held-out sentences no run ever sees. Only the optimizer and its rate differ between the runs.",
    whatYouSee: [
      "Four sliders, `SGD rate`, `Momentum rate`, `AdamW rate` and `Muon rate`, each with nine notches from a tenth of that optimizer's default to ten times it. The defaults are 1, 0.1, 0.03 and 0.05.",
      "A `Curves` switch between `Training loss` and `Held-out loss`, and a chart of that loss at each of the 26 epoch boundaries for all four runs: SGD solid, Momentum dashed, AdamW dotted and Muon dash-dot, each in its own colour, with the optimizer chosen on `Step size per row` drawn thicker. A dotted line marks ln 30 = 3.401, a model that knows nothing.",
      "A table with each run's rate, final training loss and held-out loss, the lowest held-out value marked. The AdamW row is renamed `Adam + L2` when decay goes into the gradient, and lists any setting changed on the Adam cards.",
      "Three metrics: `Lowest held out`, `Lowest training loss`, and `Held-out spread`, the highest held-out loss minus the lowest.",
    ],
    howItWorks: [
      "Each run is `trainTinyExtended` on the whole table: 25 epochs in batches of 16 over 378 training pairs, 600 steps, shuffle seed 1, zeroed starting weights and a constant rate. SGD steps by rate × gradient; momentum keeps a velocity with β 0.9; AdamW keeps two moments with β 0.9 and 0.999 and epsilon 1e-8; Muon keeps Nesterov momentum with β 0.95 and orthogonalizes it with 5 Newton–Schulz iterations.",
      "The trainer saves the weights at every epoch boundary. A training point is the cross-entropy on all 378 training pairs; a held-out point is the cross-entropy on the 118 pairs of the last two Harbor sentences.",
      "`Bias correction`, `Warmup`, `Weight decay` and `Decay mode` change only the AdamW run. SGD and momentum run without decay, and Muon always runs 5 iterations.",
      "Runs are cached by their settings, so a slider retrains only its own optimizer. A Muon run takes several times longer than the others, because each of its 600 steps runs five Newton–Schulz iterations on the 30 × 30 table.",
    ],
    controls: [
      "Each rate slider retrains its own optimizer. `Curves` changes what the chart plots; the table always shows both losses.",
      "Comparison worth running: at the defaults AdamW leads on held out, 2.041 against SGD's 2.116. Move `SGD rate` to 3 and SGD reaches 2.041 too.",
    ],
    notice: [
      "On `Training loss`, AdamW and Muon pull ahead early and finish lowest, 1.819 and 1.832, against about 2.040 for SGD and momentum.",
      "On `Held-out loss`, AdamW bottoms out at 2.025 at epoch 13 and drifts back to 2.041: it fits the six training sentences faster than what it learns carries over. SGD at its default is still improving at epoch 25.",
      "Momentum at 0.1 lands within 0.001 of SGD at 1. With β 0.9 a steady gradient builds a velocity ten times its size, so the two take about the same effective step.",
    ],
    limits: [
      "In this lab: one bigram table of 900 weights, 378 training pairs, one shuffle seed and 600 steps. The held-out text is two sentences, 118 pairs, so a gap of a few thousandths is inside the noise of the seed.",
      "In this lab: every optimizer trains the whole table, which is an embedding and an output layer at once. Muon is meant for hidden-layer matrices, which this model does not have.",
      "In general: rates do not carry over between optimizers. Adam's rate is roughly the step itself, while SGD's multiplies the gradient, so a race at default rates measures how good each default is. Real comparisons tune every optimizer and report on text kept apart from the text used to choose the settings.",
    ],
  },

  "Step size per row": {
    title: "Per-weight steps, read one row at a time",
    summary:
      "For the optimizer chosen in `Optimizer`, how far each row of the table travelled over the whole run, divided by the number of batches that held the row's character. It shows Adam's adaptive step: rows with rare, small gradients get steps about as big as busy rows do, so per visit they move much further.",
    whatYouSee: [
      "An `Optimizer` switch with SGD, Momentum, AdamW and Muon. It also thickens that optimizer's curve on the race.",
      "Bars for eight rows: space, e, t, a and o, the busiest inputs, and k, q and z, the rarest, each printed with its distance per visit and the number of visits. The rare rows use a second colour.",
      "Three metrics: `Space row per visit`, `q row per visit`, and `q ÷ space`.",
    ],
    howItWorks: [
      "For every step the trainer adds the length of the change to each row, and it counts the batches whose inputs included the row. A bar is that total divided by the count.",
      "SGD changes a row only on steps when it is in the batch, by rate × the row's averaged gradient. Momentum, AdamW and Muon change every row on every step, because their running averages keep moving a row after the gradient that started it is gone.",
      "Space starts 70 of the 378 training pairs and is in 580 of the 600 batches. q and z start one pair each and are in 25; k starts two and is in 50.",
    ],
    controls: [
      "`Optimizer` chooses the run. The rate sliders on `Optimizer race` change what each bar measures.",
      "Comparison worth running: at the default rates SGD reads `q ÷ space` 0.59 and AdamW 14.9.",
    ],
    notice: [
      "Under SGD the busy rows travel furthest per visit: space 0.101, q 0.059. A rare row's gradient is averaged with fifteen other pairs, so it is small, and SGD's step is proportional to it.",
      "Under AdamW the order flips: space 0.036, q 0.540. Adam's step is about the rate whatever the gradient's size, and its averages carry the row on between visits.",
      "Momentum's q row moves 0.059 per visit, the same as SGD's, while its space row drops to 0.022: momentum smooths the busy row's zig-zag rather than enlarging rare steps. Muon sits in between, at 9.4.",
    ],
    limits: [
      "In this lab: distance is a path length summed step by step, so a row that zig-zags travels further than one that heads straight to the same place. It measures how much the optimizer moved the row, not how far the row ended up from where it started.",
      "In general: Adam's per-weight scaling is part of why it handles rare features, such as the embeddings of rare tokens, better than plain SGD. The same scaling also lets rarely seen weights take full-size steps on very little evidence.",
    ],
  },

  "Bias correction and warmup": {
    title: "Adam's first steps, corrected and warmed up",
    summary:
      "The AdamW run's batch loss over its first 60 steps, with bias correction on or off and an optional warmup, plus the largest single-weight change on step 1. With the correction, Adam's first step equals the rate for every weight that has a gradient; without it, the step is about 3.16 times the rate.",
    whatYouSee: [
      "A `Bias correction` switch, On or Off, and a `Warmup` slider from off to 30% of the run in 5% notches, with its length in steps. A badge repeats the AdamW rate, which is set on `Optimizer race`.",
      "A chart of batch loss for steps 1 to 60: solid for the current run, dashed for the same run with bias correction flipped, a dotted line at ln 30 = 3.401, and a shaded strip over the warmup.",
      "Three metrics: `First-step move`, the largest change any weight made on step 1; `Early peak loss`, the highest batch loss in steps 1 to 60; and the run's `Held out` loss.",
    ],
    howItWorks: [
      "At step t, Adam's first moment becomes m = 0.9m + 0.1g and its second v = 0.999v + 0.001g². With the correction it divides them by 1 − 0.9ᵗ and 1 − 0.999ᵗ, so at step 1 the step is g ÷ |g|, the rate times ±1. Without it the step is 0.1g ÷ √(0.001g²) ≈ 3.16 times the rate.",
      "Warmup uses a warmup-then-constant schedule: the rate climbs in a straight line from 2% of its value to the full value over the chosen share of the run, then stays there. At 10% that is 60 steps.",
      "Batch loss is the cross-entropy of each step's 16 pairs, measured before that step's update, so it is noisier than the whole-text curves on the race.",
    ],
    controls: [
      "`Bias correction` and `Warmup` change only the AdamW run, and the race's AdamW row with it.",
      "Comparison worth running: set `AdamW rate` to 0.3. With the correction, `First-step move` reads 0.300 and `Early peak loss` 3.499. Turn it off: 0.949 and 7.853. Set `Warmup` to 10%: 0.019 and 3.849. At 20% the batch loss never rises above its starting 3.401.",
    ],
    notice: [
      "Without the correction at 0.3 the batch loss climbs through the first epoch and peaks at 7.853 on step 22, not on step 1. For a steady gradient the uncorrected step is 3.16 times the rate at step 1 but 6.6 times at step 12, because the second average fills far more slowly than the first, so the overshoot builds for a dozen steps before it eases; by step 600 it is still 1.5 times.",
      "At the default rate of 0.03 nothing spikes either way: the first step is 0.030 or 0.095. The uncorrected run still ends worse on held out, 2.103 against 2.041, because its steps stay inflated for the whole run while 1 − 0.999ᵗ is small. That is close to what a corrected run gets at three times the rate, 2.107 at 0.09.",
    ],
    limits: [
      "In this lab: one corpus, one seed, and a rate of 0.3 that is ten times AdamW's default. The spike is a deliberate overshoot on a 900-weight table, not a diverging deep network.",
      "In general: the correction removes the averages' bias but not their noise. Early in training each weight's second moment rests on a few gradients, which is the variance Liu et al. argue warmup reduces, so large models warm up for thousands of steps with the correction on.",
    ],
  },

  "Weight decay": {
    title: "L2 in the gradient against decoupled decay",
    summary:
      "The same pull toward zero applied to the AdamW run two ways: added to the gradient as L2 regularization, which Adam then divides by each weight's gradient size, or applied to the weights directly, as AdamW does. It reports the trained weights' size, the held-out loss, and how much of each row's size survives.",
    whatYouSee: [
      "A `Weight decay` slider with notches off, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1 and 0.3, and a `Decay mode` switch: `L2 in gradient` or `Decoupled`.",
      "Three metrics: `Weight norm`, the length of all 900 trained weights taken as one vector; `Held out`; and `Shrink per step`.",
      "A table of three rows, space, k and q, with the batches each appeared in and its size after training as a share of the same run without decay, and a callout naming how the current mode pulls.",
    ],
    howItWorks: [
      "Decoupled: each step multiplies every weight by 1 − rate × decay, outside the moment arithmetic, as PyTorch's AdamW does (the trainer's `adamw`). L2 in gradient: decay × weight is added to the gradient, which then feeds both moments (the trainer's `adam`).",
      "`Shrink per step` is a closed-form illustration, not a measurement of the run: one weight that starts at 1.0 and has no gradient of its own is stepped 10 times with the same arithmetic, and its average drop is reported. Decoupled gives about rate × decay. L2 gives about the full rate at any decay, because Adam divides the pull by its own size.",
      "`Size kept` divides each row's norm by the same row's norm in a second AdamW run that is identical except for zero decay.",
    ],
    controls: [
      "`Weight decay` and `Decay mode` change only the AdamW run; with decay off the two modes are the same run.",
      "Comparison worth running: at `AdamW rate` 0.03 and `Weight decay` 0.1, `L2 in gradient` gives `Weight norm` 1.19 and `Held out` 3.274, near the 3.401 of a model that knows nothing; `Decoupled` gives 59.45 and 2.061.",
    ],
    notice: [
      "Under L2 the rare rows collapse first. At decay 0.001 the space row keeps 39% of its size and q keeps 14%; decoupled decay takes about the same small share from every row, keeping 99% of each.",
      "Decoupled decay has a wide good range: from off to 0.1 the held-out loss stays between 2.031 and 2.061. L2 helps only at 0.0001; by 0.0003 it is already worse than no decay.",
      "L2 at 0.0001 scores 2.035, beating decoupled decay at the common 0.1 (2.061) on all five seeds. Decoupled at 0.03 scores 2.031, and over five seeds its range overlaps L2's, so the two best settings tie here.",
    ],
    limits: [
      "In this lab: one AdamW rate at a time, one seed, and two held-out sentences. `Weight norm` counts all 900 weights, including the four rows that never train and stay at zero.",
      "In general: Loshchilov and Hutter found that L2 inside Adam regularizes weights with large past gradients less than decoupled decay does, and that decoupling makes good decay values less dependent on the rate. Llama 2, for one, trains with AdamW and a decay of 0.1.",
    ],
  },

  "Orthogonalized update": {
    title: "Muon's update, before and after Newton–Schulz",
    summary:
      "The singular values of Muon's momentum direction at one captured step, normalized the way Newton–Schulz starts, beside the singular values after 0 to 8 iterations. Orthogonalizing keeps every direction of the update and makes the nonzero ones about the same size, while directions with no gradient stay at zero.",
    whatYouSee: [
      "A `Captured step` switch, `Step 1` or `Step 300`, and a `Newton–Schulz iterations` slider from 0 to 8, at 5 by default, Muon's own setting.",
      "Thirty pairs of bars, largest singular value first: grey for before and the accent colour for after. A dashed line marks 1, a shaded band runs up from 0.5, and a small dot marks each zero direction.",
      "Three metrics: `Directions with gradient`, `Between 0.5 and 1.5` counted among the nonzero directions after the iterations, and `Zero directions`.",
    ],
    howItWorks: [
      "The captured matrix is Muon's Nesterov direction, the gradient plus 0.95 × the momentum buffer, from a Muon run at the race's `Muon rate`. Newton–Schulz divides it by its Frobenius norm, then repeats X → 3.4445X − 4.775(XXᵀ)X + 2.0315(XXᵀ)²X.",
      "Each iteration is a polynomial in X, so it keeps the singular vectors and sends every singular value σ through σ → 3.4445σ − 4.775σ³ + 2.0315σ⁵. Zero goes to zero.",
      "Singular values are the square roots of the eigenvalues of MᵀM, found with a Jacobi eigen-solver in this module. A value at or below 1e-7 counts as zero; exact zeros come back at about 1e-8 or less.",
    ],
    controls: [
      "`Newton–Schulz iterations` reruns the iteration on the captured matrix only; the race's Muon run always uses 5. `Captured step` trains Muon up to that step.",
      "Comparison worth running: at `Step 300` the 26 nonzero values start between 0.543 and 2.1 × 10⁻⁶. After 5 iterations 22 of them sit between 0.68 and 1.20, and after 8, 25 do. At `Step 1` all 9 nonzero values land between 0.682 and 1.121, and the other 21 stay at zero.",
    ],
    notice: [
      "Before, a few directions dominate: at `Step 300` the largest is 0.543 and 14 of the 26 are below 0.07. After 5 iterations most are near 1, so a direction the gradient barely pointed along gets a step as big as the strongest one's.",
      "The values never settle exactly on 1. The quintic is tuned to grow small values fast, so its outputs bounce inside a band and never pass about 1.2.",
      "The weakest directions need more iterations: each one multiplies a tiny value by about 3.4, so 2.1 × 10⁻⁶ is still only 0.001 after 5.",
      "The 4 zero directions at `Step 300` come from the rows for j, x and the two marker symbols, which never start a training pair.",
    ],
    limits: [
      "In this lab: the matrix is the whole bigram table, both embedding and output layer, which is exactly where Muon's authors use AdamW instead. One captured step of one run is shown, not an average over training.",
      "In general: Muon is meant for the 2D weight matrices of hidden layers, while embeddings, the output layer and vector parameters such as biases go to AdamW. Each step costs a few extra matrix multiplications per matrix, and at scale its users add weight decay and per-parameter update scaling (Liu et al., 2025).",
    ],
  },

  "Tuned against tuned": {
    title: "A learning-rate search for every optimizer",
    summary:
      "`Tune each` trains every optimizer at five rates, a tenth, three tenths, one, three and ten times its default, with every other setting at the lab's defaults, and keeps the rate with the lowest held-out loss. It turns the race at defaults into a comparison of each optimizer at its best rate on the same grid.",
    whatYouSee: [
      "A `Tune each` button, a progress bar while the 20 runs train, and a badge counting the runs done.",
      "A table with a row per optimizer and a column per multiple of its default. Each cell is the held-out loss with its rate underneath, and the lowest in each row is marked.",
      "Three metrics, `Tuned winner`, `SGD, default to tuned` and `AdamW, default to tuned`, and a callout listing every optimizer's best against the noise of the shuffle seed.",
    ],
    howItWorks: [
      "Every run uses the race's recipe with a constant rate: no warmup, no decay, bias correction on, 5 Newton–Schulz iterations and seed 1. Settings changed on the other cards do not reach them.",
      "The 20 runs train one at a time between screen updates and are kept while the app is open. Muon's five take most of the time, because every one of their steps runs five Newton–Schulz iterations.",
      "The best rate is chosen by the same held-out loss the table reports. The seed figures in the callout were measured for the lesson, over shuffle seeds 1 to 5 at each tuned rate.",
    ],
    controls: [
      "`Tune each` starts the search; there is nothing else to set. A rate slider on `Optimizer race` reproduces any cell.",
      "Comparison worth running: after tuning, compare each row's default column with its marked cell.",
    ],
    notice: [
      "SGD's best rate is 3, three times its default: 2.041 against 2.116. Momentum's is 0.3: 2.035 against 2.115. AdamW's and Muon's defaults are already their best, 2.041 and 2.059.",
      "Tuned, SGD and AdamW differ by 0.0003 and momentum leads them by 0.006, while five shuffle seeds move these results by 0.006 to 0.011, so none of the three is ahead. Muon trails AdamW by 0.018, and all five of its seeds finish behind all five of AdamW's.",
      "Every row is U-shaped in the rate: too low and the run is unfinished after 600 steps, too high and it overshoots or overfits.",
    ],
    limits: [
      "In this lab: five rates per optimizer, one seed, and one 118-pair held-out set that both picks the rate and scores it, which flatters every row a little. Settings such as momentum's β or Adam's β2 are never tuned.",
      "In general: Schmidt et al. found no optimizer dominating across eight tasks, and that trying several at their defaults works about as well as tuning one. Wen et al. found that against a carefully tuned AdamW, many claimed 2× speedups were weak baselines, and the best gains shrink from about 1.4× at 0.1 billion parameters to 1.1× at 1.2 billion.",
    ],
  },

  "Optimizer state": {
    title: "Memory an optimizer keeps between steps",
    summary:
      "How many floats each optimizer keeps per weight between steps, counted from the trainer for this lab's 900 weights and multiplied out for a 7.5-billion-weight model. The lab column is measured from the trainer; the large-model column is arithmetic with the course's memory helper.",
    whatYouSee: [
      "A table with each optimizer's floats per weight, the bytes for this lab's 900 weights, and the bytes for 7.5 billion weights, with the optimizer chosen on `Step size per row` highlighted.",
      "Two metrics: `AdamW over SGD, 7.5 billion`, the extra state AdamW keeps, and `Mixed-precision Adam, all model states`.",
    ],
    howItWorks: [
      "The lab column is the trainer's own float count × 4 bytes: none for SGD, 900 floats for momentum and Muon, 1,800 for AdamW.",
      "The 7.5 billion column is the memory helper's fp32 row: AdamW counts as its Adam row, two buffers; momentum and Muon as its momentum row, one buffer; SGD as its SGD row, none.",
      "Mixed precision keeps 2-byte weights and gradients plus a 4-byte fp32 master copy, so Adam needs 16 bytes per weight in all (the ZeRO paper's accounting): 120 GB at 7.5 billion.",
    ],
    notice: [
      "AdamW's two buffers are 60 GB at 7.5 billion weights, twice the 30 GB of momentum or Muon; SGD keeps none.",
      "In mixed precision most of the memory is optimizer state: 12 of Adam's 16 bytes per weight, counting the master copy.",
    ],
    limits: [
      "In this lab: the table counts only the optimizer's buffers, at 4 bytes a float. It leaves out activations, temporary arrays such as Muon's Newton–Schulz matrices, and a real framework's overhead.",
      "In general: Muon runs only on hidden matrices, so a Muon-trained model still keeps Adam's two buffers for its embeddings and output layer. State can also be shrunk directly: Adafactor keeps per-row and per-column averages, 8-bit optimizers store each value in a byte, and ZeRO splits the state across devices.",
    ],
  },
};

export default cardInfo;
