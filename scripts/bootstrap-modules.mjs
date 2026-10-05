import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const modules = [
  {
    slug: "tokens-embeddings",
    id: "module-01-tokens-embeddings",
    title: "Tokens & embeddings",
    group: "foundations",
    order: 1,
    icon: "Binary",
    accent: "#5a67d8",
    prerequisites: [],
    minutes: 12,
    steps: ["Split text", "Inspect token IDs", "Map meaning", "Try vector arithmetic"],
    tagline: "How language becomes numbers a model can work with.",
    objectives: ["Explain why models process tokens rather than words", "Interpret distance in an embedding space"],
    glossary: [
      ["Token", "A chunk of text assigned a numeric ID."],
      ["Embedding", "A learned vector that places useful meanings near one another."],
    ],
    checkpoint: [
      "Why might one unfamiliar word become several tokens?",
      ["The model is correcting a typo", "The tokenizer builds it from known pieces", "Embeddings require equal word length"],
      1,
      "A fixed vocabulary can represent unfamiliar text by combining smaller known pieces.",
    ],
    initial: { text: "attention turns context into meaning", vocabulary: 3, focus: 1, word: "queen" },
    standard: {
      what: "A tokenizer maps text into a finite vocabulary of IDs. An embedding table then turns every ID into a dense learned vector.",
      why: "Neural networks operate on numbers. Token IDs provide stable addresses; embeddings provide coordinates the network can transform.",
      play: "Edit the sentence and change vocabulary size. Select a token, then compare clusters and arithmetic in the projection.",
      notice: "Common strings often stay whole while rarer strings split. Nearby vectors tend to share use, not dictionary definitions.",
      breaks: "A 2D projection distorts a much larger space. Vector arithmetic is suggestive, not a universal law of meaning.",
      deep: "Modern tokenizers optimize vocabulary coverage and sequence length. Embeddings are updated by backpropagation with the rest of the model.",
    },
    plain: {
      analogy: "Tokens are fridge magnets; embeddings are where you place each magnet on a giant meaning-board.",
      misconception: "A token is not always a word. It can be punctuation, part of a word, or even whitespace.",
    },
  },
  {
    slug: "single-neuron",
    id: "module-02-single-neuron",
    title: "A single neuron",
    group: "foundations",
    order: 2,
    icon: "CircleDot",
    accent: "#0d9a94",
    prerequisites: ["module-01-tokens-embeddings"],
    minutes: 10,
    steps: ["Set inputs", "Tune weights", "Choose an activation", "Read the boundary"],
    tagline: "The tiny weighted decision at the heart of a network.",
    objectives: ["Compute a weighted sum", "Describe how activation functions shape outputs"],
    glossary: [
      ["Weight", "A learned multiplier controlling an input's influence."],
      ["Activation", "A function that transforms a neuron's weighted sum."],
    ],
    checkpoint: [
      "What does increasing a positive weight usually do?",
      ["Raises that input's influence", "Adds more training data", "Creates another neuron"],
      0,
      "A larger positive weight makes increases in that input push the weighted sum further upward.",
    ],
    initial: { x1: 0.7, x2: 0.35, w1: 1.2, w2: -0.8, bias: -0.1, activation: "sigmoid" },
    standard: {
      what: "A neuron multiplies inputs by weights, adds a bias, and passes the result through an activation function.",
      why: "This operation is simple, differentiable, and composable. Networks gain expressive power by arranging many of them.",
      play: "Move the inputs and weights. Switch activations and watch both the numeric output and decision boundary.",
      notice: "Weights rotate the boundary, while bias shifts it. Nonlinear activations change how sharply output responds.",
      breaks: "A single neuron draws only one linear boundary before activation. It cannot isolate curved or disconnected regions.",
      deep: "The same affine-transform-plus-nonlinearity pattern appears in MLPs, convolution channels, and transformer feed-forward blocks.",
    },
    plain: {
      analogy: "Imagine a vote where each input gets a louder or quieter microphone. The activation decides whether the final vote counts.",
      misconception: "A neuron does not store one human-readable fact; it participates in many patterns.",
    },
  },
  {
    slug: "stacking-neurons",
    id: "module-03-stacking-neurons",
    title: "Stacking neurons",
    group: "foundations",
    order: 3,
    icon: "Network",
    accent: "#a75bba",
    prerequisites: ["module-02-single-neuron"],
    minutes: 13,
    steps: ["Build a layer", "Add depth", "Train the network", "Inspect representations"],
    tagline: "Simple boundaries combine into useful representations.",
    objectives: ["Relate depth and width to model capacity", "Explain hidden representations"],
    glossary: [
      ["Hidden layer", "An intermediate transformation between input and output."],
      ["Representation", "A learned encoding that makes a useful pattern easier to separate."],
    ],
    checkpoint: [
      "Why add a nonlinear activation between layers?",
      ["To make file sizes smaller", "Without it, stacked linear layers collapse to one linear map", "To prevent all errors"],
      1,
      "Composing only linear maps is still linear. Nonlinearities let layers build more complex boundaries.",
    ],
    initial: { depth: 3, width: 4, epochs: 18 },
    standard: {
      what: "Multilayer perceptrons alternate learned linear maps with nonlinear activations, producing progressively transformed representations.",
      why: "Intermediate layers can reuse simple features to construct more complex features and decision surfaces.",
      play: "Change hidden depth and width, then train in small bursts. Compare capacity with the loss readout.",
      notice: "Width adds parallel features; depth composes features. More capacity helps only when optimization and data support it.",
      breaks: "Larger networks can overfit, train poorly, or waste compute. Architecture size alone does not guarantee useful learning.",
      deep: "Universal approximation describes possibility, not efficiency. Inductive bias and optimization determine what is learned in practice.",
    },
    plain: {
      analogy: "One stencil draws a line. A stack of stencils can combine edges into corners, shapes, and eventually a useful picture.",
      misconception: "More layers are not automatically better; they add room to learn and room to fail.",
    },
  },
  {
    slug: "loss-gradient-descent",
    id: "module-04-loss-gradient-descent",
    title: "Loss & gradient descent",
    group: "foundations",
    order: 4,
    icon: "Mountain",
    accent: "#dd6b45",
    prerequisites: ["module-02-single-neuron"],
    minutes: 14,
    steps: ["Measure error", "Read a slope", "Take a step", "Compare momentum"],
    tagline: "How a model turns mistakes into better parameters.",
    objectives: ["Interpret loss as an optimization objective", "Predict a gradient step's direction"],
    glossary: [
      ["Loss", "A scalar score measuring how wrong predictions are."],
      ["Gradient", "The direction and rate of fastest local increase."],
    ],
    checkpoint: [
      "To reduce loss, which way should a parameter move?",
      ["Along the gradient", "Against the gradient", "Always toward zero"],
      1,
      "The gradient points uphill, so gradient descent subtracts it to move downhill.",
    ],
    initial: { weight: -1.2, learningRate: 0.18, momentum: 0.8, velocity: 0, optimizer: "sgd" },
    standard: {
      what: "A loss function converts prediction quality into a scalar. Its gradient estimates how each parameter changes that loss locally.",
      why: "Optimization needs a repeatable signal connecting model behavior to parameter updates.",
      play: "Place the weight on the curve, adjust learning rate, and take steps with SGD or momentum.",
      notice: "Tiny steps are slow; oversized steps overshoot. Momentum smooths repeated directions but can carry past a minimum.",
      breaks: "Real loss surfaces have many dimensions, saddles, noisy batches, and changing curvature. This curve is a slice.",
      deep: "Adam and RMSProp rescale updates using gradient history. Their convenience can change generalization as well as speed.",
    },
    plain: {
      analogy: "You are hiking in fog. Feel the ground's slope, take a downhill step, and repeat.",
      misconception: "Gradient descent does not scan every possible answer; it follows local clues.",
    },
  },
  {
    slug: "backpropagation",
    id: "module-05-backpropagation",
    title: "Backpropagation",
    group: "foundations",
    order: 5,
    icon: "GitBranch",
    accent: "#d04f78",
    prerequisites: ["module-04-loss-gradient-descent"],
    minutes: 16,
    steps: ["Run forward", "Measure loss", "Apply the chain rule", "Update a weight"],
    tagline: "Trace responsibility backward through a computation.",
    objectives: ["Follow forward values and backward derivatives", "Use the chain rule on a small graph"],
    glossary: [
      ["Chain rule", "A way to multiply local derivatives along a dependency path."],
      ["Local derivative", "How one operation's output changes with one input."],
    ],
    checkpoint: [
      "What travels backward during backpropagation?",
      ["The original training examples", "Derivatives of loss", "New output tokens"],
      1,
      "Backpropagation propagates derivatives that assign local responsibility for the final loss.",
    ],
    initial: { x: 0.8, weight: 0.6, target: 0.9, phase: 0 },
    standard: {
      what: "Backpropagation applies reverse-mode automatic differentiation to efficiently compute loss derivatives for every parameter.",
      why: "A model may contain millions of parameters. Recomputing each parameter's effect independently would be prohibitively expensive.",
      play: "Advance one operation at a time. Change the input, target, and weight, then inspect each local derivative.",
      notice: "Forward values move with data flow. Backward gradients move against it and multiply through the chain.",
      breaks: "Long chains can shrink or explode gradients. Nondifferentiable operations and numerical precision also need care.",
      deep: "Frameworks record an operation graph during the forward pass, then execute vector-Jacobian products in reverse topological order.",
    },
    plain: {
      analogy: "A final bill is wrong. Walk backward through every receipt and multiplier to find how much each choice contributed.",
      misconception: "Backpropagation is not the optimizer; it computes gradients that an optimizer uses.",
    },
  },
  {
    slug: "training-dynamics",
    id: "module-06-training-dynamics",
    title: "Training dynamics",
    group: "foundations",
    order: 6,
    icon: "ChartNoAxesCombined",
    accent: "#2185a6",
    prerequisites: ["module-05-backpropagation"],
    minutes: 13,
    steps: ["Split the data", "Watch the curves", "Trigger overfitting", "Regularize"],
    tagline: "Learn to read the story told by training curves.",
    objectives: ["Distinguish training and validation performance", "Recognize and respond to overfitting"],
    glossary: [
      ["Generalization", "Performance on relevant examples the model did not train on."],
      ["Regularization", "A constraint or noise source that discourages brittle fitting."],
    ],
    checkpoint: [
      "Which pattern most clearly signals overfitting?",
      ["Both losses fall", "Training loss falls while validation loss rises", "Both losses are equal"],
      1,
      "Diverging curves show that improvements on training examples no longer transfer.",
    ],
    initial: { epoch: 35, capacity: 62, regularization: 28 },
    standard: {
      what: "Training dynamics describe how optimization, data splits, capacity, and regularization interact over time.",
      why: "Final accuracy hides when and why a model began memorizing, stalled, or became unstable.",
      play: "Move through epochs, increase capacity, and add regularization. Track the gap between curves.",
      notice: "Training loss can keep improving after validation performance peaks. That gap is actionable evidence.",
      breaks: "A noisy validation set can mislead. Distribution shift may make a clean holdout unrepresentative of deployment.",
      deep: "Early stopping, dropout, weight decay, augmentation, and data quality each change dynamics through different mechanisms.",
    },
    plain: {
      analogy: "Studying the answer key can perfect practice scores while making you worse at a new exam.",
      misconception: "A low training loss does not by itself mean the model learned a transferable rule.",
    },
  },
  {
    slug: "next-token-prediction",
    id: "module-07-next-token-prediction",
    title: "Next-token prediction",
    group: "inside-models",
    order: 7,
    icon: "TextCursorInput",
    accent: "#4974d1",
    prerequisites: ["module-01-tokens-embeddings", "module-06-training-dynamics"],
    minutes: 14,
    steps: ["Enter a prompt", "Compare models", "Shape probabilities", "Sample a token"],
    tagline: "Complex text emerges from one repeated prediction.",
    objectives: ["Describe autoregressive generation", "Explain temperature without calling it creativity"],
    glossary: [
      ["Logit", "An unnormalized score for a possible next token."],
      ["Temperature", "A divisor that changes the sharpness of a sampling distribution."],
    ],
    checkpoint: [
      "What happens immediately after a token is sampled?",
      ["Generation is complete", "It is appended and the model predicts again", "The model retrains"],
      1,
      "Autoregressive generation appends the sample to context and repeats the next-token calculation.",
    ],
    initial: { prompt: "The model learns from", temperature: 0.8, model: "transformer", sample: 2 },
    standard: {
      what: "An autoregressive language model maps the current context to logits over its vocabulary, samples one token, and repeats.",
      why: "Next-token prediction supplies a dense training target for every position in enormous unlabeled text corpora.",
      play: "Compare bigram, recurrent, and transformer distributions. Change temperature and sample repeatedly.",
      notice: "Architecture changes the logits; sampling settings reshape choices after the model has produced those logits.",
      breaks: "Locally plausible continuations can accumulate into globally false or inconsistent text. Prediction is not verification.",
      deep: "Decoding strategies include top-k, nucleus sampling, repetition penalties, constrained decoding, and speculative decoding.",
    },
    plain: {
      analogy: "It is an extremely capable autocomplete that writes one piece, rereads everything, then autocompletes again.",
      misconception: "Temperature does not add knowledge or reasoning; it only reshapes sampling odds.",
    },
  },
  {
    slug: "attention",
    id: "module-08-attention",
    title: "Attention",
    group: "inside-models",
    order: 8,
    icon: "ScanEye",
    accent: "#8a5bc2",
    prerequisites: ["module-07-next-token-prediction"],
    minutes: 17,
    steps: ["Choose a query", "Compare keys", "Scale scores", "Mix values"],
    tagline: "Let every token retrieve the context it needs.",
    objectives: ["Distinguish queries, keys, and values", "Read an attention pattern without over-interpreting it"],
    glossary: [
      ["Query", "A vector describing what the current position is looking for."],
      ["Attention weight", "A normalized score controlling how much one value contributes."],
    ],
    checkpoint: [
      "What does softmax do to attention scores?",
      ["Turns them into normalized positive weights", "Stores them in the KV cache", "Removes token order"],
      0,
      "Softmax converts scores into positive weights summing to one for each query.",
    ],
    initial: { text: "The animal did not cross the street because it was tired", head: 3, selected: 9 },
    standard: {
      what: "Self-attention derives query, key, and value vectors. Query-key scores become weights used to mix value vectors.",
      why: "It creates a content-dependent route for information between token positions, regardless of their distance.",
      play: "Select a query token, change heads, and follow score, scale, softmax, and value mixing.",
      notice: "Different heads produce different routing patterns. A strong arc shows association, not necessarily a complete explanation.",
      breaks: "Attention weights omit downstream transformations and can be unstable as explanations. Context cost also grows quadratically without optimizations.",
      deep: "Multi-head attention runs distinct learned projections in parallel, concatenates their outputs, and writes the result to the residual stream.",
    },
    plain: {
      analogy: "Each word writes a search request, checks every other word's label, then borrows information from the best matches.",
      misconception: "Attention is not the same as human focus, and a bright link does not prove causation.",
    },
  },
  {
    slug: "transformer-block",
    id: "module-09-transformer-block",
    title: "The transformer block",
    group: "inside-models",
    order: 9,
    icon: "Blocks",
    accent: "#387f8d",
    prerequisites: ["module-08-attention"],
    minutes: 15,
    steps: ["Normalize", "Route with attention", "Add the residual", "Transform with the MLP"],
    tagline: "The repeated circuit that powers modern language models.",
    objectives: ["Trace information through a transformer block", "Explain the role of the residual stream"],
    glossary: [
      ["Residual stream", "The shared vector channel that sublayers read from and write to."],
      ["MLP", "A position-wise network that transforms features at each token."],
    ],
    checkpoint: [
      "Why are residual connections useful?",
      ["They preserve an information path around each sublayer", "They remove all model errors", "They tokenize the input"],
      0,
      "Residual paths keep the existing representation while a sublayer contributes an update.",
    ],
    initial: { attention: true, mlp: true, norm: true, strength: 58 },
    standard: {
      what: "A transformer block combines normalization, multi-head attention, an MLP, and residual additions in a repeated architecture.",
      why: "Attention communicates across positions while MLPs transform features locally. Residual paths make deep composition trainable.",
      play: "Ablate sublayers and inspect their writes into the residual ribbon.",
      notice: "Removing a component changes one contribution but leaves the residual path. Layer order and normalization matter.",
      breaks: "An exploded diagram suggests clean stages, while real features are distributed and interact across many blocks.",
      deep: "Pre-norm and post-norm variants differ in optimization behavior. Gated MLPs and grouped-query attention are common modern changes.",
    },
    plain: {
      analogy: "Think of a shared whiteboard. Attention fetches notes from elsewhere; the MLP rewrites ideas; each adds without erasing the board.",
      misconception: "A block does not finish one linguistic task before the next starts; features overlap and evolve.",
    },
  },
  {
    slug: "scaling-laws",
    id: "module-10-scaling-laws",
    title: "Scaling laws",
    group: "inside-models",
    order: 10,
    icon: "ChartSpline",
    accent: "#b16b36",
    prerequisites: ["module-06-training-dynamics"],
    minutes: 12,
    steps: ["Set model size", "Allocate data", "Read predicted loss", "Balance compute"],
    tagline: "Predictable trade-offs between parameters, data, and compute.",
    objectives: ["Read a power-law trend", "Describe compute-optimal model and data balance"],
    glossary: [
      ["Scaling law", "An empirical power-law relation between resources and model performance."],
      ["Compute-optimal", "A resource allocation that minimizes predicted loss for a fixed budget."],
    ],
    checkpoint: [
      "For fixed compute, why not spend everything on a larger model?",
      ["The model also needs enough data to use its capacity", "Large models cannot use tokens", "Loss ignores model size"],
      0,
      "Under-training an oversized model wastes capacity; compute-optimal recipes balance parameters and data.",
    ],
    initial: { parameters: 3.2, data: 28 },
    standard: {
      what: "Scaling laws fit empirical power curves relating model loss to parameter count, training data, and compute.",
      why: "They help estimate outcomes and allocate expensive training budgets before committing to a run.",
      play: "Change model size and token count. Try to keep the balance indicator near the efficient frontier.",
      notice: "Returns are smooth but diminishing. Increasing only one resource eventually leaves the other as the bottleneck.",
      breaks: "Fits extrapolate from specific data, architectures, and regimes. Capability thresholds and data quality can violate simple predictions.",
      deep: "Compute-optimal conclusions depend on whether inference cost, data availability, and downstream utility are included in the objective.",
    },
    plain: {
      analogy: "A huge library with too few books wastes shelves; too many books in a tiny room wastes the collection.",
      misconception: "Scaling laws are measured trends, not physical guarantees about every future system.",
    },
  },
  {
    slug: "inference-kv-cache",
    id: "module-11-inference-kv-cache",
    title: "Inference & KV cache",
    group: "inside-models",
    order: 11,
    icon: "DatabaseZap",
    accent: "#1b8f72",
    prerequisites: ["module-09-transformer-block"],
    minutes: 14,
    steps: ["Prefill the prompt", "Decode one token", "Store keys and values", "Compare recomputation"],
    tagline: "Why remembering attention work makes generation faster.",
    objectives: ["Separate prefill from decode", "Explain the KV cache memory-speed trade-off"],
    glossary: [
      ["Prefill", "The parallel processing of all prompt tokens before generation."],
      ["KV cache", "Stored attention keys and values reused during autoregressive decoding."],
    ],
    checkpoint: [
      "What does a KV cache avoid during decoding?",
      ["Sampling new tokens", "Recomputing past keys and values", "Loading model weights"],
      1,
      "Past keys and values do not change, so caching them avoids repeated attention projections.",
    ],
    initial: { context: 7, generated: 4, useCache: true },
    standard: {
      what: "Inference first prefills a prompt, then decodes sequentially. A KV cache stores each layer's past attention keys and values.",
      why: "Without caching, every new token would recompute unchanged projections for the entire preceding sequence.",
      play: "Change prompt and output lengths, then toggle caching to compare work and latency.",
      notice: "Caching lowers repeated compute but memory grows with sequence length, layer count, and KV head dimensions.",
      breaks: "Very long contexts can exhaust memory. Cache eviction, quantization, and paging introduce their own quality and latency costs.",
      deep: "Grouped-query and multi-query attention reduce KV heads. Continuous batching and paged attention improve server utilization.",
    },
    plain: {
      analogy: "Keep your earlier calculations on a scratchpad instead of solving the whole worksheet again for every new answer.",
      misconception: "The KV cache is temporary working memory, not new long-term knowledge in model weights.",
    },
  },
  {
    slug: "post-training",
    id: "module-12-post-training",
    title: "Post-training",
    group: "frontiers",
    order: 12,
    icon: "MessagesSquare",
    accent: "#c65767",
    prerequisites: ["module-07-next-token-prediction"],
    minutes: 15,
    steps: ["Compare responses", "Trace supervision", "Collect preferences", "Stress-test rewards"],
    tagline: "Turn a text predictor into a useful assistant.",
    objectives: ["Differentiate pretraining, SFT, and preference optimization", "Identify reward-model shortcuts"],
    glossary: [
      ["SFT", "Supervised fine-tuning on demonstrations of desired responses."],
      ["Preference model", "A learned scorer trained from comparisons between outputs."],
    ],
    checkpoint: [
      "What data does preference optimization usually start from?",
      ["Pairs or rankings of candidate responses", "Only raw web text", "Model architecture diagrams"],
      0,
      "Human or synthetic comparisons provide a signal about which behavior is preferred.",
    ],
    initial: { preference: "none", strictness: 56, round: 1 },
    standard: {
      what: "Post-training uses demonstrations, preference data, and sometimes verifiable rewards to shape a pretrained model's behavior.",
      why: "Next-token pretraining learns broad patterns but does not directly teach instruction following, refusal policy, or conversational usefulness.",
      play: "Label response comparisons, adjust reward strictness, and inspect how a preference signal changes ranking.",
      notice: "The reward model is a proxy for human judgment. Models can exploit stable quirks in that proxy.",
      breaks: "Preference data can encode annotator bias, miss rare harms, and reward polished but false answers.",
      deep: "Common methods include SFT, RLHF, DPO, rejection sampling, constitutional feedback, and reinforcement learning with verifiable rewards.",
    },
    plain: {
      analogy: "Pretraining teaches language from a giant library; post-training is coaching on how to answer people helpfully.",
      misconception: "Post-training shapes behavior but does not reliably erase knowledge learned during pretraining.",
    },
  },
  {
    slug: "icl-reasoning",
    id: "module-13-icl-reasoning",
    title: "ICL & reasoning",
    group: "frontiers",
    order: 13,
    icon: "BrainCircuit",
    accent: "#6c68c2",
    prerequisites: ["module-08-attention", "module-12-post-training"],
    minutes: 15,
    steps: ["Give examples", "Infer the task", "Set a budget", "Check the answer"],
    tagline: "Teach temporary tasks inside the prompt.",
    objectives: ["Describe in-context learning without weight updates", "Separate visible reasoning from correctness"],
    glossary: [
      ["In-context learning", "Adapting behavior from examples in the prompt without updating weights."],
      ["Reasoning budget", "A limit on inference tokens or compute used before an answer."],
    ],
    checkpoint: [
      "During ordinary in-context learning, what changes?",
      ["The model's stored weights", "The active context and internal activations", "The tokenizer vocabulary"],
      1,
      "Examples change the context the fixed model processes; they do not run a training update.",
    ],
    initial: { examples: 2, budget: 45, strategy: "analogy" },
    standard: {
      what: "In-context learning uses prompt examples and instructions to induce temporary task behavior without updating parameters.",
      why: "A single pretrained model can adapt at inference time to formats, labels, and strategies not fixed during training.",
      play: "Vary the number of examples and reasoning budget. Compare analogy-first and explicit-rule prompts.",
      notice: "Examples communicate both the task and output format. More visible reasoning can help, plateau, or compound an error.",
      breaks: "Models copy spurious patterns, depend on example order, and may produce plausible reasoning disconnected from their true computation.",
      deep: "Research connects ICL to implicit Bayesian inference, meta-learning, induction circuits, and retrieval over learned representations.",
    },
    plain: {
      analogy: "Instead of changing the cook, place a few finished dishes on the counter so they can infer today's recipe.",
      misconception: "A written chain of thought is not guaranteed to be a faithful window into the model's internal process.",
    },
  },
  {
    slug: "diffusion-vaes",
    id: "module-14-diffusion-vaes",
    title: "Diffusion & VAEs",
    group: "frontiers",
    order: 14,
    icon: "WandSparkles",
    accent: "#c05b91",
    prerequisites: ["module-03-stacking-neurons"],
    minutes: 16,
    steps: ["Add noise", "Denoise a sample", "Decode a digit", "Move in latent space"],
    tagline: "Two routes from probability to generated samples.",
    objectives: ["Contrast iterative denoising with latent reconstruction", "Interpret a continuous latent space"],
    glossary: [
      ["Latent", "A compressed hidden variable used to represent data factors."],
      ["Diffusion", "A generative process learned by reversing gradual noise corruption."],
    ],
    checkpoint: [
      "What does a diffusion model learn during training?",
      ["To reverse controlled noise corruption", "To cache attention keys", "To sort embedding IDs"],
      0,
      "Training examples are noised at different levels and the model learns a denoising direction.",
    ],
    initial: { step: 62, latentX: 0.3, latentY: -0.25, mode: "diffusion" },
    standard: {
      what: "Diffusion models learn iterative denoising, while variational autoencoders learn a continuous latent distribution and a decoder.",
      why: "Both turn simple random variables into structured samples, with different speed, quality, and representation trade-offs.",
      play: "Scrub through denoising, then switch to the VAE and move its two latent coordinates.",
      notice: "Diffusion resolves global structure over steps. Nearby VAE points decode to smoothly related samples.",
      breaks: "Tiny latent spaces blur details; diffusion requires many evaluations unless distilled. Neither representation guarantees semantic controls.",
      deep: "Latent diffusion performs denoising in a learned compressed space. Classifier-free guidance trades diversity for prompt alignment.",
    },
    plain: {
      analogy: "Diffusion restores a picture from television static; a VAE navigates a compressed map where nearby addresses look alike.",
      misconception: "The model is not retrieving a stored training image pixel for pixel when it generates.",
    },
  },
  {
    slug: "interpretability-features",
    id: "module-15-interpretability-features",
    title: "Interpretability I: features",
    group: "frontiers",
    order: 15,
    icon: "Microscope",
    accent: "#2c8a8e",
    prerequisites: ["module-09-transformer-block"],
    minutes: 17,
    steps: ["Inspect activations", "Separate overlapping features", "Label a feature", "Test a threshold"],
    tagline: "Turn dense activations into tentative concepts.",
    objectives: ["Explain superposition and sparse feature extraction", "Treat feature labels as hypotheses"],
    glossary: [
      ["Superposition", "Representing more features than dimensions by using overlapping directions."],
      ["Sparse autoencoder", "A model that reconstructs activations using a small number of active learned features."],
    ],
    checkpoint: [
      "Why are sparse autoencoder feature labels tentative?",
      ["Activations and examples support an interpretation but do not prove a complete concept", "Features are always random", "Labels change model weights"],
      0,
      "Top examples reveal correlations. Causal tests and broader evaluation are needed before stronger claims.",
    ],
    initial: { feature: 12, threshold: 42, overlap: 35 },
    standard: {
      what: "Mechanistic interpretability studies internal representations. Sparse autoencoders decompose dense activations into a learned dictionary of sparse features.",
      why: "Individual neurons can be polysemantic. A sparse basis may provide cleaner units for description and intervention.",
      play: "Browse feature indices, raise the activation threshold, and increase superposition pressure.",
      notice: "Top-activating text supports a label, but edge cases often reveal a broader or narrower pattern.",
      breaks: "SAEs add reconstruction error and their features depend on training choices. Interpretability is not guaranteed by sparsity.",
      deep: "Feature absorption, splitting, dead latents, scaling, and cross-layer comparisons remain active research questions.",
    },
    plain: {
      analogy: "A chord mixes notes. A sparse feature tool tries to recover the few notes playing inside a dense sound.",
      misconception: "Finding a feature that correlates with a concept does not prove it exclusively controls that concept.",
    },
  },
  {
    slug: "interpretability-circuits",
    id: "module-16-interpretability-circuits",
    title: "Interpretability II: circuits",
    group: "frontiers",
    order: 16,
    icon: "CircuitBoard",
    accent: "#a65d38",
    prerequisites: ["module-15-interpretability-features"],
    minutes: 18,
    steps: ["Find an induction pattern", "Ablate a head", "Patch activations", "Steer a feature"],
    tagline: "Test causal stories about computation inside a model.",
    objectives: ["Use ablation and patching to test causality", "State the limits of circuit claims"],
    glossary: [
      ["Activation patching", "Replacing an internal activation to measure its causal effect on behavior."],
      ["Induction head", "An attention pattern that continues a sequence seen earlier in context."],
    ],
    checkpoint: [
      "What does activation patching add beyond observing an attention map?",
      ["A controlled causal intervention", "A larger tokenizer", "Guaranteed universal explanation"],
      0,
      "Patching changes a chosen internal state and measures the behavioral difference, providing causal evidence.",
    ],
    initial: { layer: 3, patch: 5, steering: 0, ablated: false },
    standard: {
      what: "Circuit analysis combines observed patterns with interventions such as ablation, activation patching, and feature steering.",
      why: "Causal tests distinguish components that merely correlate with behavior from components necessary or sufficient in a controlled setting.",
      play: "Select patch locations, ablate the induction head, and steer the measured feature.",
      notice: "The strongest patch can localize where clean information becomes useful. Ablation may reveal redundancy elsewhere.",
      breaks: "Results are prompt- and metric-dependent. Distributed backup mechanisms make simple circuit boundaries incomplete.",
      deep: "Path patching, causal scrubbing, attribution graphs, and transcoders extend interventions from locations toward computational pathways.",
    },
    plain: {
      analogy: "Do not just watch a machine's gears. Remove one, swap a signal, and see whether the behavior actually changes.",
      misconception: "One successful circuit explanation does not mean the whole model works through neat, isolated modules.",
    },
  },
];

function yamlList(items) {
  return `[${items.map((item) => JSON.stringify(item)).join(", ")}]`;
}

function standardMdx(module) {
  return `---
objectives: ${yamlList(module.objectives)}
terms: ${yamlList(module.glossary.map(([term]) => term))}
---

## What it is

${module.standard.what}

## Why it’s here

${module.standard.why}

## How to play with it

${module.standard.play}

## What to notice

${module.standard.notice}

## Where it breaks

${module.standard.breaks}

## Going deeper

${module.standard.deep}
`;
}

function plainMdx(module) {
  return `---
objectives: ${yamlList(module.objectives)}
terms: ${yamlList(module.glossary.map(([term]) => term))}
---

## What it is

${module.plain.analogy}

${module.standard.what}

## Why it’s here

${module.standard.why}

## How to play with it

**Try this:** ${module.standard.play}

## What to notice

${module.standard.notice}

## Where it breaks

${module.standard.breaks}

**A common mix-up:** ${module.plain.misconception}

## Going deeper

${module.standard.deep}
`;
}

for (const module of modules) {
  const directory = path.join(root, "src", "modules", module.slug);
  await mkdir(path.join(directory, "content"), { recursive: true });
  await writeFile(
    path.join(directory, "Explore.tsx"),
    `import { PlannedLab, type ModuleContext } from "@app/module-sdk";

export default function Explore({ currentStep }: ModuleContext) {
  return (
    <PlannedLab
      heading=${JSON.stringify(module.title)}
      description=${JSON.stringify(module.tagline)}
      controls={${JSON.stringify(module.steps)}}
      currentStep={currentStep}
    />
  );
}
`,
  );

  const [checkpointPrompt, checkpointOptions, checkpointAnswer, checkpointExplanation] = module.checkpoint;
  const glossary = module.glossary.map(([term, definition]) => ({ term, definition }));
  await writeFile(
    path.join(directory, "module.ts"),
    `import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = ${JSON.stringify(module.initial, null, 2)};

const definition: ModuleDefinition = {
  id: ${JSON.stringify(module.id)},
  slug: ${JSON.stringify(module.slug)},
  title: ${JSON.stringify(module.title)},
  group: ${JSON.stringify(module.group)},
  order: ${module.order},
  icon: ${JSON.stringify(module.icon)},
  accent: ${JSON.stringify(module.accent)},
  prerequisites: ${JSON.stringify(module.prerequisites)},
  estimatedMinutes: ${module.minutes},
  steps: ${JSON.stringify(module.steps)},
  stepInstructions: ${JSON.stringify(module.steps.map((step) => `Use the interactive to ${step.toLowerCase()}, then explain what changed.`))},
  stateVersion: 1,
  tagline: ${JSON.stringify(module.tagline)},
  objectives: ${JSON.stringify(module.objectives)},
  glossary: ${JSON.stringify(glossary, null, 2)},
  checkpoint: {
    prompt: ${JSON.stringify(checkpointPrompt)},
    options: ${JSON.stringify(checkpointOptions)},
    answer: ${checkpointAnswer},
    explanation: ${JSON.stringify(checkpointExplanation)}
  },
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      return { ...initialState, ...(JSON.parse(value) as ModuleState) };
    } catch {
      return { ...initialState };
    }
  }
};

export default definition;
`,
  );

  await writeFile(path.join(directory, "content", "standard.mdx"), standardMdx(module));
  await writeFile(path.join(directory, "content", "plain.mdx"), plainMdx(module));
  await writeFile(path.join(directory, "assets.json"), `${JSON.stringify({ assets: [] }, null, 2)}\n`);
}

console.log(`Created ${modules.length} module folders.`);
