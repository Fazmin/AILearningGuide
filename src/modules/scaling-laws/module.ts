import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { clampWidth, FIT_IDS, LAYER_RANGE, PARAMETER_RANGE, TOKEN_RANGE } from "./scaling";

const initialState: ModuleState = {
  parameters: 3.2,
  data: 28,
  fit: "hoffmann",
  budget: "free",
  layers: 32,
  width: 4096,
};

const finiteIn = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;

const definition: ModuleDefinition = {
  id: "module-10-scaling-laws",
  slug: "scaling-laws",
  title: "Scaling laws",
  group: "inside-models",
  order: 20,
  icon: "ChartSpline",
  accent: "#b16b36",
  prerequisites: ["module-06-training-dynamics", "module-09-transformer-block", "module-11-inference-kv-cache"],
  estimatedMinutes: 14,
  steps: ["Set model size", "Allocate data", "Read predicted loss", "Balance compute", "Read the frontier", "Build N from dimensions"],
  stepInstructions: [
    "Drag Parameters with Training tokens fixed. Your dot jumps between isoFLOP curves, because a bigger model on the same data costs more compute (C = 6ND).",
    "Drag Training tokens with Parameters fixed and watch Loss breakdown: the hatched data term shrinks until the size term is the larger one.",
    "Read Predicted loss, Compute C = 6ND and Gap to optimum together, then compare the diamond (fit optimum) and the square (20 tokens per parameter) on your curve.",
    "Set Budget to Hold compute, then drag Parameters along your highlighted curve until Gap to optimum reads 0.000 at the diamond.",
    "On Frontier power law, read Slope and 10× compute keeps, then switch Fit between Hoffmann 2022 and Epoch 2024 refit: the line stays straight but its slope changes, so each tenfold budget keeps a different share of L − E.",
    "Under N from dimensions, set Layers L and Width d, read 12 × L × d² and press Use as Parameters; then double Width d and compare the new N with the old one.",
  ],
  stateVersion: 2,
  tagline:
    "Explore how model size, training tokens and compute trade off under a published scaling-law fit, and find the compute-optimal split on an isoFLOP curve.",
  objectives: [
    "Read the compute-optimal frontier as a power law: a straight line on log-log axes once the floor is removed, with steady and slow returns",
    "Describe compute-optimal model and data balance",
    "Estimate N from a transformer's layer count and width, and say what that estimate leaves out",
  ],
  glossary: [
    {
      term: "Scaling law",
      definition:
        "An empirical fit relating loss to parameters, data and compute, here a floor plus two decaying power-law terms. It holds within the range of runs it was fitted on, for one architecture, dataset and tokenizer.",
    },
    {
      term: "Compute-optimal",
      definition:
        "The split of a fixed training budget between parameters and tokens that minimizes predicted loss. It is optimal for training cost only, not for serving cost.",
    },
    {
      term: "IsoFLOP curve",
      definition:
        "Predicted loss against model size with training compute held fixed, so every larger model sees fewer tokens. Its lowest point is the compute-optimal size for that budget.",
    },
    {
      term: "Power law",
      definition:
        "A relationship of the form y = a times x to a fixed power. Loss minus its floor falls as a power of compute, so it plots as a straight line on log-log axes, with steady and slow returns.",
    },
    {
      term: "Irreducible loss",
      definition:
        "The constant E in the fit, which no amount of scale removes. It is often read as the entropy of the text, but it is a fitted number and also absorbs limits of the model family.",
    },
    {
      term: "Chinchilla ratio",
      definition:
        "The rule of thumb of about 20 training tokens per parameter, from Hoffmann et al. 2022, who trained Chinchilla with 70B parameters on 1.4T tokens. The same paper's printed parametric fit puts the optimum at a higher ratio.",
    },
    {
      term: "FLOPs",
      definition:
        "Floating-point operations, the unit of training cost. For a dense transformer, training compute is approximately 6 times parameters times tokens: about 2 per parameter per token forward and 4 backward. A fixed compute budget is therefore one isoFLOP curve, and the choice is where on it to sit.",
    },
    {
      term: "Parameter count",
      definition:
        "N, the number of learned weights. For a dense transformer it is about 12 times layers times width squared with embeddings left out. Fits differ on embeddings: Kaplan et al. fitted non-embedding parameters and Hoffmann et al. total parameters, so a rule of thumb for one does not carry straight to the other.",
    },
    {
      term: "Token-to-parameter ratio",
      definition:
        "Training tokens divided by parameter count. It locates a run between a large model starved of data and a small model fed far more data than it can use.",
    },
    {
      term: "Overtraining",
      definition:
        "Training a smaller model well past its compute-optimal token count so it is cheaper to serve. It costs a little extra loss per training FLOP and is standard for widely deployed models.",
    },
    {
      term: "Emergent ability",
      definition:
        "A capability that seems to appear abruptly with scale. Many reported cases depend on a thresholded metric such as exact match, while the underlying loss improves smoothly.",
    },
    {
      term: "Data-constrained scaling",
      definition:
        "Scaling behavior when unique high-quality text runs short. Up to about four epochs, repeated tokens are nearly as useful as fresh ones; beyond that their value decays.",
    },
    {
      term: "Extrapolation",
      definition:
        "Using a fit beyond the range it was measured on. It is the main practical use of scaling laws and also their largest assumption.",
    },
    {
      term: "Test-time compute",
      definition:
        "Compute spent at answer time through longer reasoning, multiple samples or verification. It has its own scaling behavior and trades against training compute.",
    },
  ],
  references: [
    {
      authors: "Jared Kaplan, Sam McCandlish, Tom Henighan, et al.",
      title: "Scaling Laws for Neural Language Models",
      source: "arXiv preprint arXiv:2001.08361",
      year: 2020,
      url: "https://arxiv.org/abs/2001.08361",
      note: "The first big study to show that language model loss falls as a power law in model size, data and compute. It counts about 6 FLOPs per parameter per training token, the C = 6ND readout. It counts non-embedding parameters only, and its advice was to grow the model much faster than the data.",
    },
    {
      authors: "Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al.",
      title: "Training Compute-Optimal Large Language Models",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html",
      note: "The Chinchilla paper, listed in the proceedings as \"An empirical analysis of compute-optimal large language model training\". It finds the compute-optimal split three ways, including isoFLOP curves and the fit L = E + A/N^α + B/D^β that this lab uses as Hoffmann 2022, and it trains Chinchilla (70B parameters, 1.4T tokens) on Gopher's budget.",
    },
    {
      authors: "Tamay Besiroglu, Ege Erdil, Matthew Barnett, et al.",
      title: "Chinchilla Scaling: A replication attempt",
      source: "arXiv preprint arXiv:2404.10102",
      year: 2024,
      url: "https://arxiv.org/abs/2404.10102",
      note: "An Epoch AI team redoes the Chinchilla paper's third method and finds its printed constants do not match the paper's other two methods. Their refit is the Epoch 2024 refit option on Fit, and it backs the lesson's point that the printed fit and the 20-to-1 rule of thumb disagree.",
    },
    {
      authors: "Jason Wei, Yi Tay, Rishi Bommasani, et al.",
      title: "Emergent Abilities of Large Language Models",
      source: "Transactions on Machine Learning Research (TMLR), 2022",
      year: 2022,
      url: "https://arxiv.org/abs/2206.07682",
      note: "Defines an emergent ability as one that is missing in smaller models and present in larger ones, so it cannot be predicted by extending the trend from small models. It is the source of the term in this lesson's glossary.",
    },
    {
      authors: "Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo",
      title: "Are Emergent Abilities of Large Language Models a Mirage?",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html",
      note: "Shows that many sudden jumps come from all-or-nothing metrics such as exact match, and that smooth metrics on the same model outputs show steady change. It backs the lesson's warning that loss is not capability and that a smooth loss curve can sit under a metric that jumps.",
    },
    {
      authors: "Niklas Muennighoff, Alexander Rush, Boaz Barak, et al.",
      title: "Scaling Data-Constrained Language Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html",
      note: "Finds that up to about 4 epochs of repeated text change the loss very little compared with fresh text, and that the value of more repeats then decays. This is the data-constrained scaling result the lesson cites when it says the token supply is finite.",
    },
    {
      authors: "Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al.",
      title: "LLaMA: Open and Efficient Foundation Language Models",
      source: "arXiv preprint arXiv:2302.13971",
      year: 2023,
      url: "https://arxiv.org/abs/2302.13971",
      note: "Table 2 lists each model's width and layer count, such as 32 layers of width 4,096 for the 6.7B model and 80 layers of width 8,192 for the 65.2B model. The lesson uses these to check the 12 × L × d² rule on the N from dimensions card.",
    },
    {
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, et al.",
      title: "Language Models are Unsupervised Multitask Learners",
      source: "OpenAI technical report",
      year: 2019,
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      note: "The GPT-2 paper. Table 2 gives the smallest model 12 layers of width 768, and the text gives its vocabulary of 50,257 tokens, the numbers the lesson uses to show why the embedding table is a large share of a small model.",
    },
    {
      authors: "Tim Pearce and Jinyeop Song",
      title: "Reconciling Kaplan and Chinchilla Scaling Laws",
      source: "Transactions on Machine Learning Research (TMLR), 2024",
      year: 2024,
      url: "https://arxiv.org/abs/2406.12907",
      note: "Finds that much of the gap between the Kaplan and Chinchilla advice comes from Kaplan counting non-embedding parameters while Chinchilla counted all of them. It backs the lesson's point that which parameter count a fit uses is part of the fit.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "Its scaling-law section fits isoFLOP curves and notes they get flatter near the minimum. It also says the smaller Llama 3 models were trained for much longer than is compute-optimal so they do better at the same serving cost, the overtraining recipe the lesson describes for Llama 3 8B.",
    },
    {
      authors: "Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle",
      title: "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 43445–43460",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/sardana24a.html",
      note: "Adds serving cost to the Chinchilla fit and finds that a model expecting heavy use should be smaller and trained longer than compute-optimal. It backs the glossary's point that compute-optimal covers training cost only.",
    },
    {
      authors: "Aidan Clark, Diego De Las Casas, Aurelia Guy, et al.",
      title: "Unified Scaling Laws for Routed Language Models",
      source: "Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162, 4057–4086",
      year: 2022,
      url: "https://proceedings.mlr.press/v162/clark22a.html",
      note: "Fits scaling laws for models that route each token to only some of their experts, treating parameter count and per-token compute as two separate axes. It backs the lesson's note that mixture-of-experts fits separate active from total parameters and expert count.",
    },
    {
      authors: "Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar",
      title: "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html",
      note: "Studies how answers improve with more test-time compute, through search against a verifier and repeated revision. With FLOPs matched, a smaller model given extra answer-time compute can beat a much larger one on some problems, the trade against training compute the lesson mentions.",
    },
    {
      authors: "Greg Yang, Edward J. Hu, Igor Babuschkin, et al.",
      title: "Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
      source: "Advances in Neural Information Processing Systems 34 (NeurIPS 2021)",
      year: 2021,
      url: "https://proceedings.neurips.cc/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html",
      note: "Introduces muTransfer: with the μP parametrization, the best learning rate and other settings stay stable as width grows, so they can be tuned on a small model and reused on a large one. This is the μP the lesson names for setting hyperparameters at scale.",
    },
  ],
  checkpoint: [
    {
      prompt: "On Frontier power law the compute-optimal line is straight on log-log axes. What does that say about the returns on more compute?",
      options: [
        "Returns accelerate with scale, because each extra decade of compute removes more loss than the last",
        "Each tenfold increase in compute removes the same fraction of the reducible loss: steady, slow returns",
        "Returns hit a threshold budget, after which the loss drops sharply and then flattens at the floor",
      ],
      answer: 1,
      explanation: "A straight line on log-log axes is a power law: multiplying compute by a fixed factor multiplies the reducible loss by a fixed fraction. There is no point where improvement accelerates or collapses, and halving the reducible loss takes about ninety times the compute under the printed fit. The slope is the exponent.",
      objective: 0,
    },
    {
      prompt: "Even at the largest sizes the fit predicts a loss above E. What does E stand for, and why does the frontier look straight only after subtracting it?",
      options: [
        "The text's entropy, measured directly, so the loss above it is the model's remaining error",
        "The loss of the largest model trained so far, so the line is drawn downward from that point",
        "A fitted floor no scale removes, so only the loss above it falls as a pure power of compute",
      ],
      answer: 2,
      explanation: "E is the constant in L = E + A/N^α + B/D^β. Both power-law terms shrink toward zero, so loss bends toward E and looks like diminishing returns on raw axes. Subtracting E leaves a pure power law. E is a fitted number, often read as the text's entropy but also absorbing limits of the model family.",
      objective: 0,
    },
    {
      prompt: "Your compute budget is fixed. Why not spend it all on the largest model you can build?",
      options: [
        "Fixed compute means fewer tokens for a bigger model, and past the minimum that data term outgrows the size term",
        "Large models cannot learn from more than 20 tokens per parameter, so any extra tokens would be wasted on them",
        "Predicted loss does not depend on model size once compute is fixed, so a larger model would gain nothing at all",
      ],
      answer: 0,
      explanation: "C = 6ND ties the two together: at fixed C, doubling N halves D. Past the bottom of the isoFLOP curve the B/D^β term rises faster than the A/N^α term falls, so loss goes up. The 20:1 figure is a rule of thumb, not a limit on learning.",
      objective: 1,
    },
    {
      prompt: "With Budget on Hold compute you drag Parameters well past the diamond. What happens to Training tokens and to Predicted loss?",
      options: [
        "Tokens stay put and loss keeps falling, because a bigger model is always better at any budget",
        "Tokens fall to keep compute fixed, and loss rises once the dot passes its curve's bottom",
        "Tokens and loss both stay put, because Hold compute freezes the dot at the point you left it",
      ],
      answer: 1,
      explanation: "Hold compute keeps N times D constant, so the other slider moves with the one you drag and the dot slides along one isoFLOP curve. Beyond the minimum the starved data term grows faster than the size term shrinks, so Predicted loss and Gap to optimum rise while Compute C stays fixed.",
      objective: 1,
    },
    {
      prompt: "Loss breakdown shows the hatched data term larger than the solid size term. Where should the next doubling of compute go?",
      options: [
        "Mostly to more parameters, because the larger term should be fed with a bigger model each time",
        "Equally to both, because compute-optimal always means exactly twenty tokens for every parameter",
        "Mostly to more training tokens, because doubling D would lower the loss by more than doubling N",
      ],
      answer: 2,
      explanation: "The Doubling D and Doubling N readouts say which resource buys more loss per unit of compute. When the data term dominates, extra tokens help more; at the optimum the two gains match. The 20-to-1 ratio is a rule of thumb that only approximates where that balance falls.",
      objective: 1,
    },
    {
      prompt: "Labs often train a smaller model on far more tokens than the compute-optimal count. Why accept that?",
      options: [
        "The fit is wrong beyond the optimum, so extra tokens lower loss by more than the formula predicts",
        "Compute-optimal covers training cost only, and a smaller model costs less to serve per answer",
        "Smaller models cannot overfit, so they always gain from seeing the same data many more times over",
      ],
      answer: 1,
      explanation: "The optimum is defined for training compute alone. A model served to millions of users spends more on inference than on training, so a smaller model trained past its optimum costs a little extra loss per training FLOP but far less per answer. This is overtraining, and the fit still describes its predicted loss.",
      objective: 1,
    },
    {
      prompt:
        "Under N from dimensions, you double Width d and leave Layers L alone. Compared with doubling Layers L instead, what happens to N?",
      options: [
        "It grows fourfold, against twofold for doubling layers, since every matrix has d × d entries",
        "It grows twofold, the same as doubling layers, because both simply count the number of blocks",
        "It grows eightfold, against twofold for doubling layers, because width enters N as a cube",
        "It stays the same, because N depends on how many layers there are and not on how wide they are",
      ],
      answer: 0,
      explanation:
        "Per block, attention has four d × d matrices and the MLP has d × 4d and 4d × d, so 12d² weights, and N is that times the layer count. Doubling d multiplies every matrix by four; doubling L multiplies the number of blocks by two.",
      objective: 2,
    },
    {
      prompt:
        "A small model's N from 12 × L × d² sits well below its total parameter count, while a large model's sits close to it. Why?",
      options: [
        "The rule leaves out embeddings, and the embedding table is a much larger share of a small model",
        "The rule counts only the attention matrices, and attention is a much larger share of a small model",
        "Small models store their weights at higher precision, so each one of their parameters counts for more",
        "The rule double-counts the layers, and that double count matters less once there are many of them",
      ],
      answer: 0,
      explanation:
        "The embedding table has one row per vocabulary entry, so its size is the vocabulary times d, while the blocks grow with d squared times the layer count. For GPT-2 small the table alone is 38.6M against 84.9M in the blocks. The blocks grow with d squared while the table grows only with d, so its share falls as models widen. Hoffmann et al. count total parameters, so for small models the rule undercounts their N.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      return {
        ...initialState,
        ...parsed,
        parameters: finiteIn(parsed.parameters, PARAMETER_RANGE.min, PARAMETER_RANGE.max, 3.2),
        data: finiteIn(parsed.data, TOKEN_RANGE.min, TOKEN_RANGE.max, 28),
        fit: (FIT_IDS as readonly unknown[]).includes(parsed.fit) ? parsed.fit : "hoffmann",
        budget: parsed.budget === "hold" ? "hold" : "free",
        layers: Math.round(finiteIn(parsed.layers, LAYER_RANGE.min, LAYER_RANGE.max, 32)),
        width: clampWidth(typeof parsed.width === "number" ? parsed.width : 4096),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
