import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Drawn digit": {
    title: "An 8×8 image and one receptive field",
    summary:
      "Sixty-four binary pixels that every other card reads: the filter slides over them, the RNN reads their columns left to right, and the expert router treats each column as a token. The outlined 3×3 window is the receptive field of the selected filter output.",
    whatYouSee: [
      "An 8×8 button grid; inked cells are filled. The default ink is a blocky 3.",
      "A blue outline on the 3×3 window around the selected cell, and a dot on the selected cell itself.",
      "Selected cell, a slider that names row and column, and Flip selected cell, the keyboard path for inking.",
    ],
    howItWorks: [
      "State stores `pixels` as 64 characters of 0 or 1 and `cell` as an index 0–63. Clicking a cell or pressing Flip selected cell writes the opposite bit.",
      "The window is rows r−1 to r+1 and columns c−1 to c+1 of the selected cell, cut at the edges, because the filter pads with zeros.",
    ],
    controls: [
      "Click any cell, or move Selected cell and press Flip selected cell. Clicking a cell in the filter response also selects it here.",
      "Comparison worth running: select a cell, flip a pixel inside its window, then one outside it. Only the first changes that output.",
    ],
    notice: [
      "An output of one convolution layer can see 9 pixels and nothing else. Locality is built in before any training.",
    ],
    limits: [
      "In this lab: 8×8 binary ink that you author. No image file, dataset, or trained classifier is involved.",
      "In general: real convnets read colour images with many channels, and stacked layers grow the receptive field until deep units see most of the image.",
    ],
  },

  "Live filter": {
    title: "One kernel, reused at every position",
    summary:
      "A 3×3 kernel is dotted with every 3×3 window of the pad. The same 9 weights produce all 64 outputs, which is weight sharing, and each output depends only on its own window, which is locality.",
    whatYouSee: [
      "Filter: vertical edge, horizontal edge, or center blob.",
      "The 3×3 kernel with signed weights, and an 8×8 grid of signed responses. Positive cells are tinted blue, negative cells orange, and the sign is always printed.",
      "A formula row that lists the nonzero pixel × weight products for the selected output, and metrics comparing 9 shared weights with a dense 64 → 64 layer.",
    ],
    howItWorks: [
      "response[r][c] = Σ over the 3×3 window of pixel × kernel weight, with zeros outside the pad. As in deep-learning libraries, the kernel is not flipped (cross-correlation).",
      "Vertical edge is the Prewitt kernel [−1 0 +1] on each row; horizontal edge is its transpose; center blob is a Laplacian, +4 at the centre and −1 at the four neighbours.",
      "Tint strength is |response| divided by the largest |response| on the grid.",
    ],
    controls: [
      "Filter. Clicking a response cell selects it and outlines its window on the pad.",
      "Comparison worth running: vertical edge against horizontal edge on the default 3. The bars light up under horizontal edge, the right-hand stroke under vertical edge.",
    ],
    notice: [
      "The top bar and the bottom bar get the same-shaped response from the same weights. A pattern detector learned in one place works everywhere.",
      "A dense layer mapping 64 pixels to 64 outputs would need 4,096 weights and would have to learn the edge separately at every position.",
    ],
    limits: [
      "In this lab: three hand-written kernels. Nothing is learned; the heatmap is a closed-form dot product.",
      "In general: a real convnet learns dozens to hundreds of kernels per layer and stacks many layers. Locality still means distant pixels only meet after enough layers, or through pooling, dilation, or attention.",
    ],
  },

  "Unrolled RNN": {
    title: "A chain of states, and a gradient that must cross every link",
    summary:
      "A one-number recurrent net reads the eight column means in order: h_t = tanh(w·h_(t−1) + x_t). The gradient of the final state h7 with respect to an early input is a product of one factor per later step, so it shrinks or grows geometrically with distance.",
    whatYouSee: [
      "Recurrent weight w and RNN step t.",
      "Eight boxes h0 to h7 with the hidden value and the input x below it, tinted by |h|. The small ×factor above each arrow is w·(1 − h²) for that step; the links from the selected step to h7 are drawn in orange.",
      "Bars of the gradient ∂h7/∂x_t for every column, a formula row that multiplies out the selected one, and metrics Hidden, ∂h7/∂x for the selected column, Hops to h7, and Column mean.",
    ],
    howItWorks: [
      "Inputs x_t are the fraction of inked pixels in column t + 1. h starts at 0; there is one weight on the input (fixed at 1) and one recurrent weight w.",
      "∂h_t/∂h_(t−1) = w·(1 − h_t²). ∂h7/∂x_t = (1 − h_t²) × the product of that factor for every later step. A unit test checks it against a finite difference.",
    ],
    controls: [
      "Recurrent weight from 0 to 2 and RNN step from 0 to 7. Clicking a box or a bar also sets the step.",
      "Comparison worth running: at w = 0.90 read column 1 (about 0.009) against column 8 (about 0.6). Then clear the pad and set w = 1.5: with h stuck at 0 each factor is 1.5, and column 1's gradient explodes to 1.5⁷ ≈ 17.",
    ],
    notice: [
      "Saturation matters as much as w: when |h| nears 1, 1 − h² nears 0 and the chain goes quiet even for large w.",
      "Flipping a pixel in column 1 barely moves h7, which is what vanishing gradients mean for learning: early inputs get little credit.",
    ],
    limits: [
      "In this lab: a single hidden unit with hand-set weights, no training, no LSTM gates.",
      "In general: real RNNs have vector states and Jacobian matrices, so the product's size is governed by their singular values. LSTM and GRU gates add an additive path that keeps gradients alive much longer, and gradient clipping tames explosions.",
    ],
  },

  "Who sees whom": {
    title: "Connectivity: which inputs each output can use",
    summary:
      "Each wiring is drawn as an 8×8 matrix over a sequence of eight positions: row t is an output, column s an input, and the number is how many layers or steps lie on the shortest path from s to t. The table beside it counts weights, sequential steps, and memory for each wiring.",
    whatYouSee: [
      "Wiring: Dense MLP, Per-position MLP, Convolution, Recurrent, Attention, State-space. Conv layers appears for Convolution.",
      "The matrix: filled cells marked 1 are reached in one hop, lighter numbered cells through a chain, dotted cells cannot be seen.",
      "A table at width d = 64, kernel k = 3, state size N = 16, with the current wiring highlighted.",
    ],
    howItWorks: [
      "Dense MLP: every cell 1. Per-position MLP: the diagonal only. Convolution: |t − s| ≤ layers, needing max(1, |t − s|) layers. Attention: s ≤ t, all 1. Recurrent and State-space: s ≤ t, t − s + 1 steps.",
      "Weights per layer: (n·d)², 2·d·4d, k·d², 2·d² (input and recurrent matrices), 4·d² (query, key, value, output), and 3·d·N (diagonal A, B, C of an S4D-style layer). Biases are ignored.",
      "Kept to extend is what must be stored to add one more position: the window, nothing, (k − 1)·d per conv layer, the hidden state d, the KV cache 2·n·d, or the SSM state d·N, shown at n = 8 and n = 4,096.",
    ],
    controls: [
      "Wiring, and Conv layers from 1 to 7 when Convolution is selected.",
      "Comparison worth running: Recurrent against Attention. The same cells are visible, but attention reaches input 1 in 1 hop and recurrence in 8.",
    ],
    notice: [
      "Recurrent and State-space draw the same matrix. The difference is in the table: a linear state update can be trained with a parallel scan, and both keep a fixed-size state.",
      "Attention's one-hop reach costs a KV cache that grows from 1,024 to 524,288 numbers per layer between n = 8 and n = 4,096.",
      "A transformer block is Attention followed by a Per-position MLP: one mixes across positions, the other within each position.",
    ],
    limits: [
      "In this lab: connectivity and parameter formulas, not trained models. The convolution is centred (it sees the future); causal convolutions used for text shift the window left.",
      "In general: path length and memory are only part of the story. Real models add residual connections, multiple heads, gating, and position encodings, and which wiring learns best depends on data and scale.",
    ],
  },

  "Sparse experts": {
    title: "Mixture of experts: stored weights versus active weights",
    summary:
      "A mixture-of-experts layer replaces one MLP with E expert MLPs and a router. Each token runs through only its top k experts, weighted by a softmax over those k router scores. Memory grows with E; compute per token grows with k.",
    whatYouSee: [
      "Experts E from 2 to 16 and Experts per token k from 1 to 4.",
      "A routing table: one row per token (each pad column as an 8-number vector), one column per expert, with the gate percentage where a token is routed. The load row counts tokens per expert; zero is marked.",
      "Metrics Total expert weights, Active per token, and Router weights.",
    ],
    howItWorks: [
      "Router scores = x·W + b with W (8 × 16) and b drawn from a fixed seeded generator; the first E columns are used. The top k scores are kept and softmaxed into gates, as in Mixtral.",
      "Each expert is a d → 4d → d MLP at d = 64: 2 × 64 × 256 = 32,768 weights. Total = E × 32,768, active = k × 32,768, router = 64 × E.",
    ],
    controls: [
      "Experts and Experts per token.",
      "Comparison worth running: raise Experts from 8 to 16 at k = 2. Total expert weights double to 524,288 while Active per token stays at 65,536.",
    ],
    notice: [
      "Identical pad columns always pick identical experts: routing is a deterministic function of the token.",
      "An untrained router leaves some experts idle. Stored, never used, which is why training adds a load-balancing loss.",
    ],
    limits: [
      "In this lab: an untrained router with seeded weights and no expert outputs; only the routing and the parameter arithmetic are computed.",
      "In general: the idea goes back to Shazeer et al. (2017) and Switch Transformer (k = 1, Fedus et al., 2021). Mixtral 8x7B has 46.7B parameters in total but uses about 12.9B per token, because attention and embeddings are shared. All experts must still sit in memory, and routing adds communication.",
    ],
  },
};

export default cardInfo;
