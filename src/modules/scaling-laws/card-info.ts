import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Scaling law controls": {
    title: "Two resources, one compute bill",
    summary:
      "Two log-scale sliders set the parameter count N and the training-token count D. Training compute is C = 6ND, so every choice of the pair is also a choice of budget. Budget can hold that product fixed, and Fit picks which published constants the loss formula uses.",
    whatYouSee: [
      "`Parameters` from 100M to 1T and `Training tokens` from 1B to 30T, both on logarithmic tracks so each notch is the same percentage change.",
      "`Fit`, a two-way switch between `Hoffmann 2022` (the constants printed in the Chinchilla paper) and `Epoch 2024 refit` (the replication by Besiroglu et al.).",
      "`Budget`, with `Free` and `Hold compute`. The box under it states what the sliders are doing in words, and `Fit optimum here` names the best split at your current compute.",
      "`N from dimensions`: `Layers L` from 1 to 128, `Width d` from 64 to 16,384 in steps of 64, a line reading `12 × L × d² = N weights` with the weights per block, and a `Use as Parameters` button.",
    ],
    howItWorks: [
      "Compute is `C = 6ND` FLOPs: about 2 operations per parameter per token in the forward pass and 4 in the backward pass of a dense transformer.",
      "Under `Hold compute` the product N·D is kept fixed. Moving `Parameters` sets D = C / 6N, and the moving slider stops where the other would leave its range, so the budget never changes silently.",
      "`Fit optimum here` is the closed-form minimum of the loss formula along your isoFLOP curve: N* = G·(C/6)^a with a = β/(α+β) and G = (αA/βB)^(1/(α+β)), and D* = C/(6N*).",
      "`N from dimensions` evaluates 12 · L · d². Attention has four d × d matrices and a classic MLP has d × 4d and 4d × d, so a block holds 12d² weights and L blocks hold 12·L·d². `Use as Parameters` sets the `Parameters` slider to that count, clamped to 100M to 1T; under `Hold compute` the `Training tokens` slider moves with it.",
    ],
    controls: [
      "`Parameters` and `Training tokens`: set 10B and 2B, then 200M and 100B. Both cost 1.2 × 10²⁰ FLOPs, so they sit on the same curve, on opposite sides of its minimum.",
      "`Budget`: switch to `Hold compute` and drag `Parameters`; the dot slides along one curve instead of jumping between curves.",
      "`Fit`: switch fits at the default run and watch `Fit optimum here` move from 1.38B on 65B tokens to 2.02B on 44.3B tokens.",
      "`N from dimensions`: the defaults, 32 layers of width 4,096, give 6.44B. Double `Width d` and N becomes 25.8B, fourfold; double `Layers L` instead and it becomes 12.9B, twofold.",
    ],
    notice: [
      "Under `Free`, every slider move changes the budget. Under `Hold compute`, the only question left is the split, which is what compute-optimal means.",
      "The two fits agree on the shape and disagree on where the minimum sits, most sharply at large budgets.",
      "N follows the square of the width and only the first power of the layer count, so a wider model grows much faster than a deeper one.",
    ],
    limits: [
      "In this lab: nothing is trained. Every number is the published formula evaluated in closed form; the sliders reach 1T parameters and 30T tokens, far past the runs either fit was measured on (up to about 16B parameters and 500B tokens).",
      "In general: C = 6ND ignores attention FLOPs, which matter at long context, and real runs lose efficiency to communication and restarts, so the FLOPs actually paid are higher than 6ND.",
      "In this lab: `N from dimensions` leaves out embeddings, biases, norm scales and any other MLP shape, so it lands a few percent under a real model's total: 4% under LLaMA's published 6.7B shape and 1% under its 65.2B shape. It is a calculator, not a fit.",
      "In general: fits differ on what N counts. Kaplan et al. fitted non-embedding parameters and Hoffmann et al. total parameters, and Pearce and Song (2024) attribute much of the gap between the two fits to that choice. Both fits in this lab are of Hoffmann et al.'s parametric form.",
    ],
  },

  "IsoFLOP curves": {
    title: "Loss against model size at fixed compute",
    summary:
      "Each thin curve holds training compute fixed and sweeps model size, so moving right means a bigger model trained on fewer tokens. Every curve is U-shaped, and its lowest point is the compute-optimal size for that budget. The thick curve is your own budget.",
    whatYouSee: [
      "The x-axis is parameters N on a log scale from 100M to 1T; the y-axis is predicted loss in nats per token. Thin curves are budgets from 10¹⁹ to 10²⁵ FLOPs, labelled where they leave the plot.",
      "A small dot at the bottom of each thin curve, joined by the dashed compute-optimal frontier. The dotted line joins the points that spend each budget at 20 tokens per parameter.",
      "Your run is the filled circle on the thick curve. On that curve the diamond is the fit's optimum and the hollow square is the 20:1 split. Two crosses mark Chinchilla (70B on 1.4T tokens) and Gopher (280B on 300B tokens) at the loss this fit predicts for them.",
      "Readouts: `Predicted loss`, `Compute C = 6ND`, `Tokens per parameter`, and `Gap to optimum`, the loss above the diamond at the same compute. The legend repeats each marker's size, tokens and loss as text.",
    ],
    howItWorks: [
      "Loss is `L(N, D) = E + A/N^α + B/D^β` with N and D as raw counts. `Hoffmann 2022` uses E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (eq. 10 of the paper). `Epoch 2024 refit` uses E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658.",
      "An isoFLOP curve evaluates L(N, C/6N) across N. Its minimum has the closed form above, and the frontier is that minimum traced across budgets.",
      "Chinchilla and Gopher used almost the same compute (5.9 and 5.0 × 10²³ FLOPs). Under the printed fit Chinchilla scores 1.937 and Gopher 1.993: the smaller model trained on more tokens wins at equal cost.",
    ],
    controls: [
      "Drag `Parameters` with `Training tokens` fixed: the dot changes curve because compute changes.",
      "Set `Budget` to `Hold compute` and drag `Parameters` until `Gap to optimum` reads 0.000; the dot is then on the diamond.",
      "Switch `Fit` and watch the diamond, the frontier and the gap move while the square stays on the same N.",
    ],
    notice: [
      "The bottoms are flat. At the default budget the 20:1 square sits only 0.006 above the diamond although it uses a 54% larger model. Getting the ratio roughly right matters; getting it exactly right barely does.",
      "Under `Hoffmann 2022` the diamond drifts right of the square as compute grows: at Chinchilla's budget the printed fit prefers 32B parameters on 3.0T tokens (93 per parameter). Under `Epoch 2024 refit` the two nearly coincide, near 20 per parameter.",
      "Loss never reaches the floor: even at 1T parameters and 30T tokens the printed fit predicts 1.793 against E = 1.69.",
    ],
    limits: [
      "In this lab: the curves are a formula, not measurements, and the Chinchilla and Gopher crosses are what the fit predicts for their sizes, not their reported losses. Loss is in the fit's own units (nats per token on the MassiveText data and tokenizer), so it does not transfer to another model's loss.",
      "In general: Besiroglu et al. 2024 found the printed constants inconsistent with the paper's other two estimation methods and refitted them. Rounding alone matters: their table lists Hoffmann's estimates to more digits (α = 0.3392, β = 0.2849), which put the optimum at Chinchilla's budget near 40B rather than 32B. Fits also shift with architecture, data mix and learning-rate schedule.",
    ],
  },

  "Loss breakdown": {
    title: "Which term is holding the loss up",
    summary:
      "The same formula split into its parts: the floor E, the size term A/N^α and the data term B/D^β. The bars compare your run with the optimum at the same compute, and the readouts say how much each resource would buy if you doubled it.",
    whatYouSee: [
      "A worked line: the formula, your N and D in scientific notation, then the evaluated terms, with the size term in the forward colour, the data term in the gradient colour, and the total.",
      "Two bars, `Your run` and `Optimum, same C`, showing loss above the floor. The solid segment is the size term and the hatched segment is the data term; each segment prints its value and the row ends with their sum.",
      "`Doubling N lowers L by` and `Doubling D lowers L by`, with a sentence naming which one is larger.",
    ],
    howItWorks: [
      "Doubling N multiplies the size term by 2^−α, so it lowers L by (A/N^α)·(1 − 2^−α). Doubling D lowers L by (B/D^β)·(1 − 2^−β). Either doubling doubles C.",
      "At the optimum the marginal returns per log step match: α·(A/N^α) = β·(B/D^β). Because α and β differ, the two terms are not equal there; under the printed fit the size term is β/α ≈ 0.82 times the data term.",
    ],
    controls: [
      "At the default run (3.2B on 28B tokens) doubling tokens buys 0.086 and doubling parameters 0.050, so the next compute should go to data.",
      "Raise `Training tokens` until the two doubling readouts meet; `Gap to optimum` on the isoFLOP card falls toward zero at the same time.",
    ],
    notice: [
      "The bottleneck moves rather than disappearing. Push one slider far enough and the other term becomes the one worth buying.",
      "Both terms shrink slowly. With β = 0.28, ten times more tokens divides the data term by only 10^0.28 ≈ 1.9.",
    ],
    limits: [
      "In this lab: the terms come from a formula fitted to one family of runs. The split between size term and data term is only as trustworthy as that fit.",
      "In general: real models are not described by two knobs alone. Data quality, repetition, architecture and schedule all move the terms, and a lower loss does not guarantee better scores on a specific task.",
    ],
  },

  "Frontier power law": {
    title: "A straight line once the floor is removed",
    summary:
      "Along the compute-optimal frontier, loss minus the floor E falls as a pure power of compute. On log-log axes that is a straight line, which is what lets labs extrapolate from small runs to large ones.",
    whatYouSee: [
      "The x-axis is training compute from 10¹⁸ to 10²⁶ FLOPs; the y-axis is L − E. Both are logarithmic.",
      "The dashed frontier is straight. The dotted 20:1 line nearly overlaps it. Your run is the dot, with a short dashed drop to the frontier at the same compute.",
      "`Slope` and `10× compute keeps`, which is 10 to the power of the slope, written as the share of reducible loss that survives a tenfold budget.",
    ],
    howItWorks: [
      "At the optimum both reducible terms scale as C^(−αβ/(α+β)), so L*(C) − E is an exact power law. The slope is −0.154 under the printed fit and −0.178 under the refit.",
      "A slope of −0.154 means each tenfold increase in compute keeps 70% of the reducible loss; it takes about 90 times the compute to halve it.",
    ],
    controls: [
      "Drag either slider: the dot moves right with compute and sits above the line by your `Gap to optimum`.",
      "Switch `Fit` to see the slope and the floor change together.",
    ],
    notice: [
      "Straight on log-log axes means steady, predictable and slow returns. There is no point on the line where improvement accelerates.",
      "Loss itself, with the floor left in, would bend toward E and look like diminishing returns. The straight line only appears after subtracting E.",
    ],
    limits: [
      "In this lab: the line is exact because it is derived from the formula. Real frontiers are fitted to a few dozen runs and carry error bars that widen with every decade of extrapolation.",
      "In general: a smooth loss line can sit under a benchmark that jumps, because a thresholded metric such as exact match turns steady gains into sudden-looking ones. Loss is not capability.",
    ],
  },
};

export default cardInfo;
