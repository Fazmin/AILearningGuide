import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateOptimizersState, initialState } from "./state";

export { hydrateOptimizersState };

const definition: ModuleDefinition = {
  id: "module-53-optimizers",
  slug: "optimizers",
  title: "Optimizers in practice",
  group: "pretraining",
  order: 43,
  icon: "Gauge",
  accent: "#2c7d8c",
  prerequisites: ["module-19-hyperparameters-schedules"],
  estimatedMinutes: 16,
  steps: [
    "Race four optimizers",
    "Per-weight step sizes",
    "Bias correction and warmup",
    "Decouple the decay",
    "Orthogonalize the update",
    "Tune fairly, count memory",
  ],
  stepInstructions: [
    "Leave SGD rate, Momentum rate, AdamW rate and Muon rate at their defaults and compare the four curves on Optimizer race, then the Held out column for SGD, Momentum, AdamW and Muon.",
    "On Step size per row, switch Optimizer between SGD and AdamW and compare how far the rare rows k, q and z move per visit with the space row, reading q ÷ space each time.",
    "Set AdamW rate to 0.3, turn Bias correction off and read First-step move and Early peak loss, then set Warmup to 10% and read them again.",
    "Set AdamW rate back to 0.03 and Weight decay to 0.1, then switch Decay mode between L2 in gradient and Decoupled and compare Weight norm, Held out and Shrink per step.",
    "On Orthogonalized update, step Newton–Schulz iterations from 0 to 5 and watch the after bars gather between 0.5 and 1.5 while the zero directions stay at zero, then switch Captured step to Step 1.",
    "Press Tune each and compare each optimizer's lowest held-out loss with its default column, then read Optimizer state for 900 weights and for 7.5 billion.",
  ],
  stateVersion: 1,
  tagline:
    "SGD, momentum, AdamW and Muon trained on the same text with the same batches, then taken apart: per-weight steps, bias correction, two kinds of weight decay, an orthogonalized update, a fair tuning race and the memory each optimizer keeps.",
  objectives: [
    "Explain how Adam gives every weight its own step size, and why rarely used weights move furthest under it",
    "Explain what bias correction fixes in Adam's first steps, and why Adam runs usually start with a warmup",
    "Tell L2 regularization inside Adam from decoupled weight decay, and predict which weights each one shrinks",
    "Describe what Muon does to an update matrix, and which weights it is and is not meant for",
    "Compare optimizers only after tuning each one, and count the optimizer-state memory each one costs",
  ],
  glossary: [
    {
      term: "Optimizer state",
      definition:
        "The numbers an optimizer carries from one step to the next on top of the weights and the gradient. Plain SGD keeps none, momentum and Muon keep one float per weight, and Adam keeps two, so in fp32 Adam's state alone is 8 bytes per weight.",
    },
    {
      term: "Momentum",
      definition:
        "A running sum of past gradients that the optimizer steps along instead of the raw gradient: each step the velocity becomes 0.9 times the old velocity plus the new gradient. A steady gradient builds a step ten times the rate, so a momentum rate is not comparable with a plain SGD rate.",
    },
    {
      term: "First moment",
      definition:
        "Adam's running average of each weight's gradient, decayed by beta1, 0.9 by default. It sets the step's direction and smooths the noise between batches, much as momentum does.",
    },
    {
      term: "Second moment",
      definition:
        "Adam's running average of each weight's squared gradient, decayed by beta2, 0.999 by default. Its square root says how large that weight's gradients usually are, and Adam divides the step by it.",
    },
    {
      term: "Bias correction",
      definition:
        "Dividing Adam's two running averages by 1 minus beta1 to the power t and 1 minus beta2 to the power t at step t. Both averages start at zero, so early on they are too small by exactly those factors; without the correction the first step is about 3.16 times the rate instead of equal to it.",
    },
    {
      term: "Epsilon",
      definition:
        "A small constant, 1e-8 by default, added to the square root of the second moment before Adam divides by it, so a weight whose gradients have all been zero does not divide by zero. It only matters when that square root is itself about as small as epsilon.",
    },
    {
      term: "Adaptive step size",
      definition:
        "A step size set separately for every weight from that weight's own gradient history. In Adam the step is about the rate times the first moment divided by the root of the second, which is close to the full rate for any weight whose gradient keeps its sign, whether that gradient is large or tiny.",
    },
    {
      term: "Adam",
      definition:
        "An optimizer that keeps a first and a second moment for every weight and moves each weight by the rate times the corrected first moment over the root of the corrected second moment plus epsilon. Its steps are roughly the rate in size, which is why its rates are far smaller than SGD's.",
    },
    {
      term: "AdamW",
      definition:
        "Adam with decoupled weight decay: each step also multiplies every weight by 1 minus the rate times the decay, outside the moment arithmetic. It has long been the default optimizer for pretraining language models.",
    },
    {
      term: "Decoupled weight decay",
      definition:
        "Shrinking every weight by the same small fraction each step, directly, instead of adding decay times the weight to the gradient. For plain SGD the two are the same step; under Adam they differ, because a pull added to the gradient is divided by each weight's own gradient size.",
    },
    {
      term: "Muon",
      definition:
        "An optimizer for the 2D weight matrices of hidden layers that keeps one momentum matrix and replaces it with an approximately orthogonal matrix before each step. Its authors train embeddings, the output layer and vector parameters such as biases with AdamW instead.",
    },
    {
      term: "Newton–Schulz iteration",
      definition:
        "A short loop of matrix multiplications that pushes every nonzero singular value of a normalized matrix toward 1 without computing a singular value decomposition. Muon runs five steps of a quintic version, which lands the values in a band around 1 rather than exactly on it and leaves zero directions at zero.",
    },
    {
      term: "Orthogonalized update",
      definition:
        "An update matrix whose singular values have all been pushed to about 1 while its singular vectors are kept, so every direction the gradient points along gets a step of similar size instead of a few dominant directions taking most of it. Directions with no gradient still get nothing.",
    },
  ],
  references: [
    {
      authors: "Diederik P. Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      source: "International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1412.6980",
      note: "Defines Adam's first and second moments, the defaults beta1 0.9, beta2 0.999 and epsilon 1e-8, and bias correction for averages that start at zero. It also says each step is roughly bounded by the rate, and that leaving the correction out gives very large early steps, the 3.16 times First-step move this lab measures.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "Decoupled Weight Decay Regularization",
      source: "International Conference on Learning Representations (ICLR 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1711.05101",
      note: "Shows that L2 regularization and weight decay are the same for SGD but not for Adam, where L2 regularizes weights with large past gradients less, and proposes AdamW's decoupled decay. This is the Decay mode switch and the lesson's account of which rows each mode shrinks.",
    },
    {
      authors: "Liyuan Liu, Haoming Jiang, Pengcheng He, et al.",
      title: "On the Variance of the Adaptive Learning Rate and Beyond",
      source: "International Conference on Learning Representations (ICLR 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/1908.03265",
      note: "Argues that Adam's per-weight step sizes vary too much early in training because they rest on few gradients, and that warmup works by reducing that variance. It backs the lesson's reason Adam runs start with a warmup even when bias correction is on.",
    },
    {
      authors: "Ilya Sutskever, James Martens, George Dahl, et al.",
      title: "On the importance of initialization and momentum in deep learning",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3), 1139–1147",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/sutskever13.html",
      note: "Describes classical momentum as a velocity that accumulates past gradients, decayed by a momentum coefficient, and its close relative Nesterov's accelerated gradient. That velocity is the Momentum run in the race and the buffer Muon keeps.",
    },
    {
      authors: "Keller Jordan, Yuchen Jin, Vlado Boza, et al.",
      title: "Muon: An optimizer for hidden layers in neural networks",
      source: "Blog post",
      year: 2024,
      url: "https://kellerjordan.github.io/posts/muon/",
      note: "Introduces Muon: SGD-momentum's update for each 2D parameter, replaced by the nearest semi-orthogonal matrix through five quintic Newton–Schulz steps with coefficients 3.4445, −4.7750 and 2.0315, tuned to land values roughly between 0.7 and 1.3. It says embeddings, the output layer and scalar or vector parameters should use AdamW, the limit the lesson draws.",
    },
    {
      authors: "Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al.",
      title: "Muon is Scalable for LLM Training",
      source: "arXiv preprint arXiv:2502.16982",
      year: 2025,
      url: "https://arxiv.org/abs/2502.16982",
      note: "Moonshot AI's report that adding weight decay and adjusting each parameter's update scale lets Muon scale, with about twice AdamW's computational efficiency in compute-optimal scaling experiments, and their 16-billion-parameter Moonlight model trained on 5.7 trillion tokens. It is the large-scale evidence the lesson sets against the toy.",
    },
    {
      authors: "Kimi Team",
      title: "Kimi K2: Open Agentic Intelligence",
      source: "arXiv preprint arXiv:2507.20534",
      year: 2025,
      url: "https://arxiv.org/abs/2507.20534",
      note: "Moonshot AI's technical report on Kimi K2, a mixture-of-experts model with 1 trillion total and 32 billion activated parameters, pretrained on 15.5 trillion tokens with MuonClip, Muon plus a QK-clip step against exploding attention logits. It backs the lesson's statement that Muon has trained a frontier-scale model.",
    },
    {
      authors: "Jeremy Bernstein and Laker Newhouse",
      title: "Old Optimizer, New Norm: An Anthology",
      source: "OPT 2024: 16th Annual Workshop on Optimization for Machine Learning",
      year: 2024,
      url: "https://arxiv.org/abs/2409.20325",
      note: "Reads Adam, Shampoo and Prodigy, with their running averages switched off, as steepest descent under different norms: Adam as sign descent, and Shampoo as a step to the nearest semi-orthogonal matrix. It is the Going deeper link between Adam's per-weight steps and Muon's per-direction steps.",
    },
    {
      authors: "Kaiyue Wen, David Hall, Tengyu Ma, et al.",
      title: "Fantastic Pretraining Optimizers and Where to Find Them",
      source: "arXiv preprint arXiv:2509.02046",
      year: 2025,
      url: "https://arxiv.org/abs/2509.02046",
      note: "Tunes about ten optimizers carefully at four scales from 0.1 to 1.2 billion parameters. It finds AdamW has long been the dominant pretraining optimizer, that many reported 2× speedups reflect weak baselines, and that the best matrix optimizers' lead shrinks from about 1.4× to 1.1× with scale, the tuned-against-tuned point of the last step.",
    },
    {
      authors: "Robin M. Schmidt, Frank Schneider, and Philipp Hennig",
      title: "Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 9367–9376",
      year: 2021,
      url: "https://arxiv.org/abs/2007.01547",
      note: "Benchmarks fifteen optimizers on eight problems and finds none clearly dominates, and that trying several optimizers at their defaults works about as well as tuning one. This is why the race at defaults and the Tune each table can tell different stories.",
    },
    {
      authors: "Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, et al.",
      title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
      source: "SC20: International Conference for High Performance Computing, Networking, Storage and Analysis",
      year: 2020,
      url: "https://arxiv.org/abs/1910.02054",
      note: "Counts mixed-precision Adam at 16 bytes per parameter, 12 of them optimizer state: an fp32 master copy, momentum and variance. Its 7.5-billion-parameter example needs 120 GB, the figure on the Optimizer state card.",
    },
    {
      authors: "Noam Shazeer and Mitchell Stern",
      title: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost",
      source: "Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80, 4596–4604",
      year: 2018,
      url: "https://arxiv.org/abs/1804.04235",
      note: "Keeps only per-row and per-column sums of the squared-gradient average for each weight matrix, cutting that buffer from one number per weight to one per row and column, and can switch momentum off. It is one of the memory savers in Going deeper.",
    },
    {
      authors: "Tim Dettmers, Mike Lewis, Sam Shleifer, et al.",
      title: "8-bit Optimizers via Block-wise Quantization",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2110.02861",
      note: "Stores optimizer state in 8 bits with block-wise quantization while matching 32-bit results. It puts 32-bit momentum and Adam at 4 and 8 bytes per parameter, 4 GB and 8 GB for a 1-billion-parameter model, cut to 1 GB and 2 GB, the other memory saver in Going deeper.",
    },
    {
      authors: "PyTorch Contributors",
      title: "torch.optim.AdamW",
      source: "PyTorch documentation",
      year: 2026,
      url: "https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
      note: "The standard implementation: defaults of rate 0.001, betas 0.9 and 0.999, epsilon 1e-8 and weight decay 0.01, with bias correction, and decay applied as a separate shrink of rate times decay times the weight. The lab's trainer applies decoupled decay the same way, which is what Shrink per step reports.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Trains with AdamW at beta1 0.9, beta2 0.95, epsilon 1e-5 and weight decay 0.1, a typical large-model recipe the lesson compares with the lab's settings in Toy versus real.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "On Step size per row the q row starts one training pair and the space row starts 70. You switch Optimizer from SGD to AdamW. What happens to q ÷ space?",
      options: [
        "It falls well below SGD's value, because AdamW's rate of 0.03 is so much smaller than SGD's 1",
        "It rises far above 1, because Adam sizes each weight's step from its own gradient history",
        "It stays near SGD's value, because both optimizers follow the same gradient and only scale it",
      ],
      answer: 1,
      explanation:
        "SGD moves a row by rate × gradient and only when the row is in the batch, so the rarely seen q row travels 0.59 times as far per visit as the space row. Adam divides each weight's running gradient by the root of its running squared gradient, so even a small, rare gradient produces a step of about the rate, and the running averages keep the row moving for several steps after each visit. Under AdamW the q row travels 14.9 times as far per visit as the space row, although its rate is 33 times smaller than SGD's.",
      objective: 0,
    },
    {
      prompt:
        "At AdamW rate 0.3 with Bias correction on, the first step moves every weight that has a gradient by exactly 0.300. You turn Bias correction off. What does First-step move read now?",
      options: [
        "Less than 0.300, because both running averages still start out close to zero",
        "About 0.949, because the averages are scaled down unevenly, the second one under a root",
        "Still exactly 0.300, because the correction only starts to matter once the averages fill up",
      ],
      answer: 1,
      explanation:
        "After one step the first moment is 0.1 × g and the second is 0.001 × g², each too small by the factor the correction divides out. Uncorrected, the step is 0.1g ÷ √(0.001g²) = 3.16 times the rate in the gradient's sign, so 0.949 at a rate of 0.3. Both averages are too small, but the second sits under a square root, so the ratio comes out too big, not too small. In the lab that larger first stride sends the early batch loss to 7.853, against 3.499 with the correction.",
      objective: 1,
    },
    {
      prompt:
        "Bias correction fixes Adam's first step exactly. Why do Adam runs still usually begin with a warmup?",
      options: [
        "Adam's steps grow larger as training goes on, so the rate has to start low to keep the last steps safe",
        "Early second-moment averages rest on very few gradients, so the per-weight step sizes are unreliable",
        "Warmup exists for SGD, and Adam runs keep it only out of habit, since it changes nothing measurable",
      ],
      answer: 1,
      explanation:
        "Correcting the averages removes their bias toward zero but not their noise: after a handful of batches each weight's second moment is an estimate from a few samples, so the division gives step sizes that vary far more than they will later. Warmup holds the rate down while those estimates fill in. In this lab, at AdamW rate 0.3 with the correction off, a 10% warmup lowers the early peak loss from 7.853 to 3.849, and 20% keeps it at the starting 3.401.",
      objective: 1,
    },
    {
      prompt:
        "At AdamW rate 0.03 and Weight decay 0.1, Decay mode L2 in gradient leaves the held-out loss near 3.27 while Decoupled leaves it near 2.06. Why does the same decay value do so much more damage in the gradient?",
      options: [
        "Adam divides the pull by each weight's own gradient size, so weakly driven weights are pulled at about the full rate",
        "L2 decay squares each weight before pulling it, so large weights are pulled far harder than under decoupled decay",
        "Decoupled decay only acts on steps where a weight's row is in the batch, so it pulls far less often than L2 in the gradient",
      ],
      answer: 0,
      explanation:
        "Decoupled decay multiplies every weight by 1 − 0.03 × 0.1 each step, the same 0.3% for all of them. L2 adds 0.1 × weight to the gradient before Adam divides by that gradient's running size, so for any weight whose own gradient is smaller than the pull, the step toward zero is about the full rate. The rare rows go first: the q row keeps 0.12% of its size against 60% with decoupled decay, and the weight norm falls to 1.19 against 59.45.",
      objective: 2,
    },
    {
      prompt:
        "On Orthogonalized update at Step 1, only 9 of the 30 rows had a gradient, so the update has 21 zero directions. What do five Newton–Schulz iterations do to them?",
      options: [
        "They rise to about 1 with the rest, because the iteration makes the whole update orthogonal",
        "They leave zero but stay small, because each iteration spreads a little of the update into them",
        "They stay at zero, because each iteration multiplies the matrix by a polynomial in itself",
      ],
      answer: 2,
      explanation:
        "Each iteration replaces X with aX + b(XXᵀ)X + c(XXᵀ)²X, a polynomial in X that keeps its singular vectors and maps every singular value through the same scalar polynomial. A singular value of zero maps to zero, so a row with no gradient gets no update. The nine nonzero values, which start between 0.211 and 0.537, end between 0.682 and 1.121: pushed near 1, not onto it.",
      objective: 3,
    },
    {
      prompt:
        "This lab runs Muon on the whole 30 × 30 table, and tuned Muon finishes last on held-out loss. What does that tell you about Muon?",
      options: [
        "Little, because the table is both the embedding and the output layer, the layers Muon's authors give to AdamW",
        "That Muon loses to AdamW on language models in general, since this bigram table is itself a small language model",
        "That Muon needs weight decay to work at all, since the race runs every optimizer without decay",
      ],
      answer: 0,
      explanation:
        "Muon is built for the 2D weight matrices of hidden layers; its authors optimize embeddings, the output layer and vector parameters with AdamW. The bigram table is nothing but an embedding and an output layer in one matrix, so the race tests Muon exactly where it is not meant to run, and tuned Muon's 2.059 against AdamW's 2.041 says nothing about hidden layers. At scale, Moonshot AI trained Kimi K2 with MuonClip, and Liu et al. report about twice AdamW's compute efficiency.",
      objective: 3,
    },
    {
      prompt:
        "At their default rates AdamW beats SGD on held-out loss by 0.075. Tune each then trains both at five rates. What is the fairest reading of the result?",
      options: [
        "AdamW is the better optimizer here, because it won from the defaults both runs started at",
        "SGD is the better optimizer here, because tuning improved it and left AdamW where it was",
        "They tie: SGD's default was a poor rate for SGD, and at its best rate it matches AdamW",
      ],
      answer: 2,
      explanation:
        "SGD's held-out loss falls from 2.116 at rate 1 to 2.041 at rate 3, while AdamW's default, 0.03, is already its best at 2.041. The gap is 0.0003, and five shuffle seeds move each result by 0.006 to 0.011. The race at defaults measured how good each default was, not how good each optimizer is, which is why careful benchmarks tune every optimizer before comparing them.",
      objective: 4,
    },
    {
      prompt:
        "A 7.5-billion-weight model trained in fp32 switches its hidden matrices from AdamW to Muon. What happens to the optimizer state for those matrices?",
      options: [
        "It halves, from two floats per weight to one, because Muon keeps only its momentum",
        "It disappears, because Muon rebuilds its update from the current gradient each step",
        "It doubles, because Muon stores the orthogonalized matrix beside its momentum buffer",
      ],
      answer: 0,
      explanation:
        "AdamW keeps a first and a second moment per weight, 8 bytes in fp32, 60 GB at 7.5 billion weights. Muon keeps only its momentum, 4 bytes, 30 GB; the orthogonalized matrix is recomputed each step and thrown away. In mixed precision both also keep a 4-byte master copy of every weight, so whole model states come to 120 GB with Adam and 90 GB with one buffer, and the embeddings and output layer keep Adam's two buffers either way.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateOptimizersState,
};

export default definition;
