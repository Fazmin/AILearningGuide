import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {
  rank: 6,
  temperature: 1,
  epochs: 18,
  labels: "soft",
  pruneFraction: 0.3,
  pruneMode: "unstructured",
  context: "h",
  /** Importance score for the compose card: how its prune stage decides which student values to drop. */
  importance: "magnitude",
  composeShare: 0.5,
  composeBits: 4,
};

const MODES = ["unstructured", "semi", "structured"];
const CONTEXTS = [" ", "h", "t", "e", "o", "s"];
const IMPORTANCE = ["magnitude", "activation", "ablation"];

/**
 * Version 1 had no context key and only two pruning modes; both are still valid values. Version 2 had no importance,
 * composeShare, or composeBits: each takes its default, and every field is clamped to its control.
 */
export function hydrateDistillationState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const number = (key: string, fallback: number) =>
      typeof parsed[key] === "number" && Number.isFinite(parsed[key]) ? (parsed[key] as number) : fallback;
    return {
      ...initialState,
      rank: Math.min(12, Math.max(1, Math.round(number("rank", 6)))),
      temperature: Math.min(6, Math.max(0.5, Math.round(number("temperature", 1) * 2) / 2)),
      epochs: Math.min(40, Math.max(1, Math.round(number("epochs", 18)))),
      labels: parsed.labels === "hard" ? "hard" : "soft",
      pruneFraction: Math.min(0.9, Math.max(0, Math.round(number("pruneFraction", 0.3) * 20) / 20)),
      pruneMode:
        typeof parsed.pruneMode === "string" && MODES.includes(parsed.pruneMode) ? parsed.pruneMode : "unstructured",
      context: typeof parsed.context === "string" && CONTEXTS.includes(parsed.context) ? parsed.context : "h",
      importance:
        typeof parsed.importance === "string" && IMPORTANCE.includes(parsed.importance) ? parsed.importance : "magnitude",
      composeShare: Math.min(0.9, Math.max(0, Math.round(number("composeShare", 0.5) * 20) / 20)),
      composeBits: Math.min(8, Math.max(2, Math.round(number("composeBits", 4)))),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-27-distillation-pruning",
  slug: "distillation-pruning",
  title: "Distillation & pruning",
  group: "training-adapting",
  order: 49,
  icon: "Scissors",
  accent: "#b04f77",
  prerequisites: ["module-26-quantization", "module-16-interpretability-circuits"],
  estimatedMinutes: 18,
  steps: [
    "Copy the teacher",
    "Tune the temperature",
    "Prune by magnitude",
    "Compare the three shapes",
    "Compose the stages",
    "Change the importance score",
  ],
  stepInstructions: [
    "Switch Training target between Teacher distribution and One-hot labels, and compare the two KL bars in Distillation and the student bars in Soft targets.",
    "Move Distillation temperature from 0.5 up to 6, reading KL to teacher at each setting, and watch Teacher mass on unseen next characters grow in Soft targets.",
    "With Pruning mode on Unstructured, raise Pruned share and watch perplexity, then find the share where the curve starts to bend.",
    "Set Pruned share to 50% and switch Pruning mode between Unstructured, 2:4 semi-structured, and Structured rows, comparing Pruned perplexity and the mask.",
    "In Compose the three steps, set Share pruned from the student to 50% and Final bit width to 4, then read the stored size and perplexity of the Teacher, Distilled student, Pruned, and Quantized rows.",
    "Switch Importance score between Magnitude, Activation-weighted, and Measured ablation at a 70% share, and compare the three perplexities after pruning with the passes each one needs.",
  ],
  stateVersion: 3,
  tagline:
    "Train a smaller model on a bigger one's distribution, remove weights by magnitude in three shapes, then run distil, prune, and quantize as one pipeline and measure what each stage and each importance score costs.",
  objectives: [
    "Explain what soft labels carry that one-hot labels do not",
    "Say what distillation temperature controls",
    "Distinguish unstructured, 2:4, and structured pruning by what each one actually saves",
    "Read perplexity and stored size after each stage of distil, prune, and quantize, and compare importance scores by what they keep and what they cost to compute",
  ],
  glossary: [
    {
      term: "Distillation",
      definition:
        "Training a small model to match a larger one's output distribution, usually as T² times the KL divergence between their temperature-softened distributions, often mixed with ordinary cross-entropy on the labels. The teacher is used at training time only.",
    },
    {
      term: "Soft labels",
      definition:
        "The teacher's full probability distribution over every token, including the ones it rejected. The relative sizes of those rejected probabilities are the extra signal.",
    },
    {
      term: "Temperature",
      definition:
        "A divisor on both models' logits before softmax in the distillation loss. It raises the weight the loss puts on small probabilities; it does not move the optimum for a student that can match the teacher exactly.",
    },
    {
      term: "Dark knowledge",
      definition:
        "The information in which wrong answers the teacher considered plausible. A one-hot label discards all of it; a teacher that learned little beyond label frequencies has little of it to give.",
    },
    {
      term: "Student capacity",
      definition:
        "How much of the teacher a smaller model can represent at all. Below some size, no amount of distillation closes the gap.",
    },
    {
      term: "Magnitude pruning",
      definition:
        "Removing the weights closest to zero, on the assumption that they contribute least. Cheap, effective, and only an approximation of importance.",
    },
    {
      term: "Unstructured pruning",
      definition:
        "Zeroing individual weights anywhere. The matrix keeps its shape, so it saves storage only with a sparse format and little time on dense hardware.",
    },
    {
      term: "Semi-structured sparsity",
      definition:
        "A fixed N:M pattern such as 2:4, keeping two of every four consecutive weights. Recent GPUs' sparse tensor cores skip the zeros, for up to twice the matmul throughput at exactly 50% sparsity.",
    },
    {
      term: "Structured pruning",
      definition:
        "Removing whole units: heads, neurons, channels, layers, or here whole context rows. The matrices get smaller, which saves real computation and costs more quality per weight removed.",
    },
    {
      term: "Sparsity",
      definition:
        "The share of weights that are exactly zero. A number worth reporting alongside quality, together with the pattern, since the pattern decides whether any time is saved.",
    },
    {
      term: "Importance score",
      definition:
        "Any estimate of how much a parameter matters: its magnitude, its magnitude times the input it multiplies, or the measured rise in loss when it is removed, the ablation from the interpretability labs. Better scores cost more passes over the data.",
    },
    {
      term: "Sparse format",
      definition:
        "A storage layout that keeps only the surviving values plus a record of where they were, here one mask bit per position. It is what turns unstructured zeros into saved bytes; real formats use indices or block metadata instead.",
    },
    {
      term: "Compression pipeline",
      definition:
        "Distilling to a smaller model, pruning it, then quantizing what is left, with size and quality measured after each stage. Quantizing goes last because it assumes the weights are final, and each stage inherits the error of the one before.",
    },
    {
      term: "Activation-weighted score",
      definition:
        "An importance score that multiplies a weight's magnitude by the size of the input it meets over calibration text, as Wanda does for each output. It needs one pass over the data, and helps when some inputs are far larger than others.",
    },
  ],
  references: [
    {
      authors: "Geoffrey Hinton, Oriol Vinyals, and Jeff Dean",
      title: "Distilling the Knowledge in a Neural Network",
      source: "NIPS 2014 Deep Learning Workshop; arXiv preprint arXiv:1503.02531",
      year: 2015,
      url: "https://arxiv.org/abs/1503.02531",
      note: "The paper behind the distillation loss in this lab. It divides the logits by a temperature, mixes the soft-target loss with ordinary cross-entropy, and multiplies the soft part by T squared, and it reports that a much smaller student does best at middle temperatures, like the shallow optimum in Distillation temperature.",
    },
    {
      authors: "Samuel Stanton, Pavel Izmailov, Polina Kirichenko, et al.",
      title: "Does Knowledge Distillation Really Work?",
      source: "Advances in Neural Information Processing Systems 34 (NeurIPS 2021), 6906–6919",
      year: 2021,
      url: "https://proceedings.neurips.cc/paper_files/paper/2021/hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html",
      note: "Separates how closely a student matches its teacher from how well it does on the task, and finds students often stay far from the teacher. It backs the lesson's point that fitting the labels and matching the teacher, the KL to teacher readout, are different questions.",
    },
    {
      authors: "Li Yuan, Francis E. H. Tay, Guilin Li, et al.",
      title: "Revisiting Knowledge Distillation via Label Smoothing Regularization",
      source: "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020), 3903–3911",
      year: 2020,
      url: "https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.html",
      note: "Argues that distillation works partly as a learned form of label smoothing, since even a weak teacher can help. This is the regularization account of soft targets in Going deeper.",
    },
    {
      authors: "Gemma Team, Google DeepMind",
      title: "Gemma 2: Improving Open Language Models at a Practical Size",
      source: "arXiv preprint arXiv:2408.00118",
      year: 2024,
      url: "https://arxiv.org/abs/2408.00118",
      note: "A real case of small open models learning from a larger one: the 2B and 9B models are trained on a larger teacher's next-token probabilities instead of one-hot next tokens. It shows the soft-label training of this lab at production scale.",
    },
    {
      authors: "Yoon Kim and Alexander M. Rush",
      title: "Sequence-Level Knowledge Distillation",
      source: "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), 1317–1327",
      year: 2016,
      url: "https://aclanthology.org/D16-1139/",
      note: "Introduces sequence-level distillation, where the student learns from whole outputs the teacher generates rather than one next-token distribution at a time. It also prunes the distilled student afterward, an early version of the compose pipeline.",
    },
    {
      authors: "Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, et al.",
      title: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2306.13649",
      note: "Shows that a student trained only on fixed text meets different sequences once it generates its own, and fixes this by having the teacher score the student's own outputs. It backs the lesson's point that matching next-token distributions alone does not transfer multi-step behaviour.",
    },
    {
      authors: "Song Han, Jeff Pool, John Tran, and William J. Dally",
      title: "Learning both Weights and Connections for Efficient Neural Networks",
      source: "Advances in Neural Information Processing Systems 28 (NIPS 2015)",
      year: 2015,
      url: "https://proceedings.neurips.cc/paper_files/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html",
      note: "The train, prune small weights, retrain recipe behind Magnitude pruning. It cut most of a large image model's weights without losing accuracy, but with a retraining step that this lab's one-shot pruning leaves out.",
    },
    {
      authors: "Asit Mishra, Jorge Albericio Latorre, Jeff Pool, et al.",
      title: "Accelerating Sparse Deep Neural Networks",
      source: "arXiv preprint arXiv:2104.08378",
      year: 2021,
      url: "https://arxiv.org/abs/2104.08378",
      note: "NVIDIA's description of the 2:4 pattern and the Sparse Tensor Cores in Ampere GPUs that skip its zeros for twice the math throughput. It also explains why scattered unstructured zeros rarely speed up matrix hardware, the tradeoff behind Pruning mode.",
    },
    {
      authors: "Xinyin Ma, Gongfan Fang, and Xinchao Wang",
      title: "LLM-Pruner: On the Structural Pruning of Large Language Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 21702–21720",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html",
      note: "Structured pruning on large language models: it removes whole connected groups of weights, scored with gradient information, then recovers quality with a short fine-tune. It is the real-scale version of the Structured rows mode.",
    },
    {
      authors: "Song Han, Huizi Mao, and William J. Dally",
      title: "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding",
      source: "International Conference on Learning Representations (ICLR 2016)",
      year: 2016,
      url: "https://arxiv.org/abs/1510.00149",
      note: "A staged compression pipeline that prunes first, then quantizes the surviving weights, and reports the saving after each stage. It is the pattern the Compose the three steps card follows, including quantizing last.",
    },
    {
      authors: "Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, et al.",
      title: "Compact Language Models via Pruning and Knowledge Distillation",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 41076–41102",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/4822991365c962105b1b95b1107d30e5-Abstract-Conference.html",
      note: "NVIDIA's Minitron work combines both levers of this lesson on large models: it prunes layers, heads, and neurons from a trained model, then retrains the smaller one by distilling from the original.",
    },
    {
      authors: "Yann LeCun, John Denker, and Sara Solla",
      title: "Optimal Brain Damage",
      source: "Advances in Neural Information Processing Systems 2 (NIPS 1989)",
      year: 1989,
      url: "https://proceedings.neurips.cc/paper_files/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html",
      note: "Questions the idea that magnitude equals importance and scores each weight by its estimated effect on the training error, using second derivatives. It is the older root of the Importance score control and of SparseGPT's second-order approach.",
    },
    {
      authors: "Elias Frantar and Dan Alistarh",
      title: "SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10323–10337",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/frantar23a.html",
      note: "Prunes very large GPT-style models in one shot, without retraining, by solving a layer-by-layer reconstruction problem. It also covers 2:4 patterns and works alongside quantization, the method named in Where it breaks.",
    },
    {
      authors: "Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter",
      title: "A Simple and Effective Pruning Approach for Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2306.11695",
      note: "The Wanda method: score each weight by its magnitude times the size of its input activation, compared per output, with no retraining. This is the Activation-weighted score option, and the paper reports it beats magnitude pruning on LLaMA, the result this toy reverses.",
    },
    {
      authors: "Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag",
      title: "What is the State of Neural Network Pruning?",
      source: "Proceedings of Machine Learning and Systems 2 (MLSys 2020)",
      year: 2020,
      url: "https://proceedings.mlsys.org/paper_files/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html",
      note: "A review of 81 pruning papers finding that results are hard to compare because papers report different size, speed, and quality numbers. It backs the lesson's advice to report sparsity and its pattern together with quality after each stage.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At rank 6 and T = 1 you switch Training target from Teacher distribution to One-hot labels, and KL to teacher rises. Why can the one-hot student not close the gap?",
      options: [
        "One-hot labels are harder to fit, so the training loss stays far above the soft student's loss at every step",
        "A one-hot update carries one sampled character, so the student never sees how the teacher ranks the others",
        "The one-hot student is given a smaller rank, so it has less capacity to match the teacher's whole distribution",
        "The one-hot student is scored against a different teacher, so its KL cannot be compared with the soft one's",
      ],
      answer: 1,
      explanation:
        "A soft update carries the teacher's whole next-character distribution, including how it ranks the characters it rejected. A one-hot update carries one sampled character, so the student averages noisier signals. The two training losses nearly overlap while KL differs about twofold: fitting the labels and matching the teacher are different questions.",
      objective: 0,
    },
    {
      prompt:
        "Soft labels carry what one-hot labels discard, yet this lab's teacher has only a thin tail of dark knowledge. Why?",
      options: [
        "Soft labels only help when the student has more capacity than the teacher, and every student here is smaller",
        "The lab fixes the temperature at 1, which switches the soft labels off and leaves one-hot training",
        "The teacher is a bigram that learned almost exactly the corpus's frequencies, so its rejected probabilities add little",
        "The student's rank is too high to benefit, because a large student matches the labels without help",
      ],
      answer: 2,
      explanation:
        "Dark knowledge is the information in which wrong answers the teacher found plausible. A teacher that learned little beyond the corpus's own character frequencies has only a thin smoothing tail to give, so its soft labels differ little from the labels. A large pretrained teacher carries far more. The lab shows the mechanism, not the size of the effect.",
      objective: 0,
    },
    {
      prompt:
        "You raise Distillation temperature from 1 to 4 at rank 6. What happens in Soft targets on “h” and in KL to teacher?",
      options: [
        "Teacher mass on characters that never follow “h” shrinks, and KL falls, because a high temperature sharpens the distribution",
        "Nothing changes in either, because temperature does not move the optimum for a student that can match the teacher",
        "KL falls to zero and the student matches the teacher exactly, because soft targets at high temperature are easier to fit",
        "Teacher mass on those characters grows, and KL ends higher, because the student spends capacity on the tail",
      ],
      answer: 3,
      explanation:
        "Temperature divides the logits, so a higher value flattens the teacher's distribution and lifts small probabilities, which is where its ranking of wrong answers lives. A student that could match the teacher would not care, but a rank-6 student cannot, so at high temperature it spends capacity on the tail and ends further from the teacher at T = 1.",
      objective: 1,
    },
    {
      prompt:
        "At the same share removed, Structured rows costs far more perplexity than Unstructured. Why is structured pruning used anyway?",
      options: [
        "Removing whole rows removes real computation, while a sparse matrix is usually just as slow to multiply",
        "Structured pruning gives better quality on the training set once the rows are gone, which hides the perplexity cost",
        "Unstructured pruning cannot reach high sparsity, so structured pruning is the only way to remove more weights",
        "Structured pruning needs no importance score, so it avoids the cost of deciding which weights to remove",
      ],
      answer: 0,
      explanation:
        "Zeroed weights still occupy their place in a dense matrix multiply, so unstructured sparsity buys storage and little speed without specialised hardware or a sparse format. Structured pruning changes the shape of the computation, which is the saving people actually want, and the extra quality cost per weight is the price.",
      objective: 2,
    },
    {
      prompt:
        "At a 50% pruned share, where does 2:4 semi-structured land between Unstructured and Structured rows, and what does it buy?",
      options: [
        "Better than both, because the regular pattern lets the optimizer recover the weights that were removed from each group",
        "Between them: a fixed pattern costs more than free choice but less than deleting rows, and sparse tensor cores skip its zeros",
        "Worse than Structured rows, because a fixed pattern forces the most important weights to be removed first in each group",
        "The same as Unstructured, because both remove the same half of the weights and only the mask is drawn differently here",
      ],
      answer: 1,
      explanation:
        "Forcing two zeros in every group of four constrains which weights can go, so perplexity sits above free-choice pruning but below deleting whole rows. In exchange the pattern is regular enough that sparse tensor cores on recent GPUs can skip the zeros for up to about twice the matmul throughput. The lab measures quality, not time.",
      objective: 2,
    },
    {
      prompt:
        "At 50% pruned and 4 bits, which stage of Compose the three steps gives up the least perplexity for the bytes it saves?",
      options: [
        "Prune: removing the smallest values is free, so it saves bytes without moving perplexity at all in any run",
        "Distil: the student copies the teacher, so shrinking to it costs no perplexity",
        "Quantize: it saves nearly as many bytes as pruning and adds the least perplexity of the three stages",
        "All three give up the same perplexity per byte, because the pipeline reports one running total",
      ],
      answer: 2,
      explanation:
        "Reading down the rows: distilling saves 2,160 B for +1.73 perplexity, pruning saves 675 B for +0.23, and quantizing saves 606 B for +0.06. Pruning is not free, a student is not a copy, and every stage adds its own error to the one before. Which stage wins depends on the settings: at 2 bits the quantize stage turns expensive.",
      objective: 3,
    },
    {
      prompt:
        "At a 70% pruned share, how do the three importance scores compare after the prune stage?",
      options: [
        "Activation-weighted leaves the lowest perplexity, because weighing each weight by its input always beats magnitude alone",
        "Magnitude leaves the lowest, because the simplest score is least likely to overfit the calibration text",
        "All three tie, because each removes the same number of values and so the same amount of information",
        "Measured ablation leaves the lowest, and activation-weighted ends worse than plain magnitude on this toy",
      ],
      answer: 3,
      explanation:
        "Ablation asks the loss directly what each value is worth, so it keeps the right ones. Activation-weighting here only measures how often a context occurs, which ignores how much the softmax cares about a given logit, so it falls behind magnitude. On large models the published result runs the other way, because their input activations have enormous outlier channels that magnitude cannot see.",
      objective: 3,
    },
    {
      prompt:
        "Measured ablation prunes best here, yet it is not how large models are pruned. Why not?",
      options: [
        "It needs one pass over the calibration text for every weight, 360 here and billions in a real model, so cheaper proxies stand in",
        "It needs the teacher model at inference time, so it cannot be used once the student has shipped to the people using it",
        "It only works for unstructured pruning, and large models are pruned in structured shapes that skip whole rows",
        "On a large model its scores come out the same as magnitude, so the extra work of computing them buys nothing at all",
      ],
      answer: 0,
      explanation:
        "The Passes to score readout is the bill: 0 for magnitude, 1 for activation-weighted, and one per factor entry for ablation. At billions of weights that is out of reach, which is why Wanda multiplies magnitude by an input-activation norm and SparseGPT solves a layer-wise reconstruction with second-order information instead. Ablation also measures each weight alone, so it misses interactions between weights.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateDistillationState,
};

export default definition;
