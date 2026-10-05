import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeState } from "./state";

/**
 * Version 1 held only problem and budget, version 2 added the sampling keys, and version 3 added the
 * group-relative training card (`grpo*` keys). `sanitizeState` fills anything an older payload lacks and
 * clamps every field, because the training run happens synchronously in a memo.
 */
export function hydrateReasoningState(value: string) {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-46-reasoning-models",
  slug: "reasoning-models",
  title: "Reasoning models",
  group: "training-adapting",
  order: 46,
  icon: "Lightbulb",
  accent: "#8a5a2a",
  prerequisites: ["module-13-icl-reasoning", "module-45-rl-foundations", "module-24-preference-optimization"],
  estimatedMinutes: 18,
  steps: [
    "Pick a practiced problem",
    "Train against a checker",
    "Add a length cost",
    "Raise the budget",
    "Inspect the extra tokens",
    "Sample and vote",
  ],
  stepInstructions: [
    "Set Problem to Sunday hours, then read the note on the Problem card: a prompt changes what the model reads, while training against a checker changes the weights.",
    "In Train against a checker, leave Length cost off and drag Training update from 0 to 100. Watch Mean length climb past Steps the task needs (k) while Checker pass rate rises toward 100%.",
    "Set Length cost to On and compare the length curve at update 100 with the cost off. Then set Group size to 8 and watch the policy collapse to one-step answers.",
    "Raise Thinking budget to 5, then to 6, and watch Padding and Unfaithful increment while Useful stays at 3 until the final step.",
    "Set Problem to 17 + 28 and read the first two trace lines: the carry is written down before the tens are added. Then drop the budget to 0 and note that the answer slot is authored.",
    "In Sampling and voting, set Samples to 16 and compare Majority vote with Verifier picks. Then set Wrong answers to Always the same one and watch Majority vote fall below One sample.",
  ],
  stateVersion: 3,
  tagline:
    "Train a toy reasoning policy against a checker, then price the two kinds of test-time compute: a longer trace, and more samples with a vote or a verifier.",
  objectives: [
    "Contrast a reasoning model, whose long traces come from reinforcement learning against a checker, with in-context chain-of-thought prompting",
    "Explain how group-relative training against a verifiable reward lengthens traces, and why a missing length cost lets them pad",
    "Treat test-time compute as extra generated tokens or search, not as proof of inner thoughts",
    "Predict when majority voting helps and when only a verifier does",
  ],
  glossary: [
    {
      term: "Chain of thought",
      definition:
        "Intermediate tokens generated before an answer. They can be trained, prompted, or distilled. Visible steps are not a window into a separate mind.",
    },
    {
      term: "Test-time compute",
      definition:
        "Extra work spent while answering: a longer trace, or several sampled traces plus a way to pick one. It costs tokens and latency; it does not rewrite the weights.",
    },
    {
      term: "Reasoning model",
      definition:
        "A next-token predictor post-trained, typically with reinforcement learning against a checker, to write long intermediate traces before answering. The architecture is unchanged; the post-training recipe is different.",
    },
    {
      term: "Verifiable reward",
      definition:
        "A reward computed by a program rather than a learned judge: an exact-match check on a final answer, or unit tests on code. It is cheap and hard to argue with, and it can only reward what the program can tell apart.",
    },
    {
      term: "RLVR",
      definition:
        "Reinforcement learning with verifiable rewards: sample several traces, score each with a checker, and make the traces that pass likelier. No human labels each trace, but the policy learns only what the checker distinguishes.",
    },
    {
      term: "GRPO (group-relative advantage)",
      definition:
        "A policy-gradient method that samples a group of G traces for one prompt and scores each against its group: advantage is reward minus the group mean, divided by the group standard deviation. The group is the baseline, so no value network is needed; a group with equal rewards gives zero advantage and no update.",
    },
    {
      term: "Overthinking",
      definition:
        "Trace length beyond what the answer needs. With a pure verifiable reward nothing penalises extra steps, and spare steps can still raise the pass chance a little, so length keeps drifting up. A per-step cost or a length budget trims it; the extra tokens are latency, not understanding.",
    },
    {
      term: "Process supervision",
      definition:
        "Scoring each step rather than only the final answer, usually with a trained process reward model. It can catch a wrong step that happens to reach a right answer, and it can still reward a fluent but unused step.",
    },
    {
      term: "Majority vote",
      definition:
        "Sampling several traces and returning the most common final answer, also called self-consistency. It needs no checker, and it amplifies whichever single answer is most likely, right or wrong.",
    },
    {
      term: "Verifier / best-of-n",
      definition:
        "A verifier checks a candidate answer: an exact-match test, unit tests, or a learned reward model. Best-of-n samples n candidates and keeps the one it scores highest; a perfect verifier succeeds whenever any sample is right, which is pass@n, while a learned one can be gamed.",
    },
    {
      term: "Distillation",
      definition:
        "Training a smaller or faster model to imitate traces from a stronger one. The student copies the text of the steps, not a guaranteed algorithm.",
    },
    {
      term: "Pass@n",
      definition:
        "The chance that at least one of n independent samples is correct, 1 minus (1 minus p) to the n for per-sample accuracy p. It is the ceiling for any way of choosing among the samples and what a perfect verifier reaches.",
    },
    {
      term: "Padding",
      definition:
        "A trace step that restates or recalls something the answer never uses. It adds tokens and latency without changing the answer; this lab places it by hand, where a real sampler interleaves it with useful steps.",
    },
    {
      term: "Unfaithful step",
      definition:
        "A trace step that does not match the work the answer needs, such as naming an unused fact, the wrong operation, or an invented rule. The lab labels these by hand; in real models faithfulness is tested by editing or truncating a trace and checking whether the answer changes.",
    },
  ],
  references: [
    {
      authors: "OpenAI",
      title: "Learning to reason with LLMs",
      source: "OpenAI research post",
      year: 2024,
      url: "https://openai.com/index/learning-to-reason-with-llms/",
      note: "The post that introduced o1, a model trained with reinforcement learning to write a chain of thought before it answers. It reports that accuracy rises both with more training and with more time spent thinking, the two ideas behind the Train against a checker card and the Thinking budget slider.",
    },
    {
      authors: "Daya Guo, Dejian Yang, Haowei Zhang, et al.",
      title: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
      source: "Nature 645(8081), 633–638",
      year: 2025,
      url: "https://www.nature.com/articles/s41586-025-09422-z",
      note: "Trains a reasoning model with GRPO and rule-based rewards that check final answers, with no human-written traces. Its response length grows during training as the model learns to think longer, which is the Mean length curve in the lab, and it distills the results into smaller models.",
    },
    {
      authors: "Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al.",
      title: "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
      source: "arXiv preprint arXiv:2411.15124",
      year: 2024,
      url: "https://arxiv.org/abs/2411.15124",
      note: "An open post-training recipe that names Reinforcement Learning with Verifiable Rewards (RLVR). Its RLVR section rewards the model only when a program confirms the answer, as the lesson's verifiable reward does, on math and instruction-following data.",
    },
    {
      authors: "Zhihong Shao, Peiyi Wang, Qihao Zhu, et al.",
      title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
      source: "arXiv preprint arXiv:2402.03300",
      year: 2024,
      url: "https://arxiv.org/abs/2402.03300",
      note: "Introduces GRPO. It drops the separate value model and uses the scores of a group of answers to the same question as the baseline, the group-relative advantage the Train card computes. It also adds the KL penalty to a reference model that the lesson's Going deeper section describes.",
    },
    {
      authors: "Zichen Liu, Changyu Chen, Wenjun Li, et al.",
      title: "Understanding R1-Zero-Like Training: A Critical Perspective",
      source: "arXiv preprint arXiv:2503.20783",
      year: 2025,
      url: "https://arxiv.org/abs/2503.20783",
      note: "Shows that GRPO's division by response length and by the group standard deviation biases training, so wrong answers grow longer and longer. It proposes a variant without those terms, which backs the lesson's point that published GRPO variants differ in how they normalise the loss.",
    },
    {
      authors: "Xingyu Chen, Jiahao Xu, Tian Liang, et al.",
      title: "Do NOT Think That Much for 2+3=? On the Overthinking of Long Reasoning Models",
      source: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267, 9487–9499",
      year: 2025,
      url: "https://proceedings.mlr.press/v267/chen25bx.html",
      note: "Studies overthinking: reasoning models spend many extra tokens on easy problems for little or no gain in accuracy. It measures how much of a trace actually helps and trains models to cut the waste, like the Length cost toggle in the lab.",
    },
    {
      authors: "Niklas Muennighoff, Zitong Yang, Weijia Shi, et al.",
      title: "s1: Simple test-time scaling",
      source: "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), 20275–20321",
      year: 2025,
      url: "https://aclanthology.org/2025.emnlp-main.1025/",
      note: "Controls a thinking budget by stopping the model's thinking early or making it keep going by appending \"Wait\". Accuracy rises with the budget, which backs the lesson's claim that reported accuracy tracks how long the model may think and that forcing an early stop lowers it.",
    },
    {
      authors: "Tamera Lanham, Anna Chen, Ansh Radhakrishnan, et al.",
      title: "Measuring Faithfulness in Chain-of-Thought Reasoning",
      source: "arXiv preprint arXiv:2307.13702",
      year: 2023,
      url: "https://arxiv.org/abs/2307.13702",
      note: "Tests whether a chain of thought is faithful by truncating it, adding mistakes, or paraphrasing it, then checking whether the answer changes. This is the test the Unfaithful step glossary entry describes for real models.",
    },
    {
      authors: "Yanda Chen, Joe Benton, Ansh Radhakrishnan, et al.",
      title: "Reasoning Models Don't Always Say What They Think",
      source: "arXiv preprint arXiv:2505.05410",
      year: 2025,
      url: "https://arxiv.org/abs/2505.05410",
      note: "Gives reasoning models hints in the prompt and finds their traces often do not mention a hint they used. Training against outcomes improves this only up to a point, which backs the lesson's warning that a visible trace is not proof of inner thoughts.",
    },
    {
      authors: "Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al.",
      title: "Training Verifiers to Solve Math Word Problems",
      source: "arXiv preprint arXiv:2110.14168",
      year: 2021,
      url: "https://arxiv.org/abs/2110.14168",
      note: "Introduces the GSM8K math dataset and trains a verifier that scores many sampled solutions, then keeps the highest-ranked one. This is best-of-n with a learned verifier, the method behind Verifier picks in Sampling and voting.",
    },
    {
      authors: "Xuezhi Wang, Jason Wei, Dale Schuurmans, et al.",
      title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2203.11171",
      note: "Samples several chains of thought and returns the most common final answer. This is the self-consistency method that the Majority vote readout computes.",
    },
    {
      authors: "Bradley Brown, Jordan Juravsky, Ryan Ehrlich, et al.",
      title: "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling",
      source: "arXiv preprint arXiv:2407.21787",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21787",
      note: "Finds that the share of problems solved by at least one sample keeps rising with the number of samples, which is pass@n. Where an automatic checker exists this turns into real gains, while majority voting and reward models level off, the gap the Sampling and voting card shows.",
    },
    {
      authors: "Hunter Lightman, Vineet Kosaraju, Yuri Burda, et al.",
      title: "Let's Verify Step by Step",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html",
      note: "Compares rewarding only the final answer with rewarding each step, and finds step-level process supervision works better on hard math problems. It is the source for the lesson's process supervision and process reward model terms.",
    },
    {
      authors: "Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar",
      title: "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html",
      note: "Compares ways to spend test-time compute, including best-of-n and beam search guided by a process reward model that extends the most promising partial answers. It finds the best choice depends on how hard the problem is, which backs the lesson's section on search beyond voting.",
    },
    {
      authors: "Leo Gao, John Schulman, and Jacob Hilton",
      title: "Scaling Laws for Reward Model Overoptimization",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/gao23h.html",
      note: "Measures what happens when best-of-n sampling or reinforcement learning pushes too hard on a learned reward model: the true quality first rises, then falls. It backs the lesson's warning that best-of-n with a learned reward model can pick an answer that games the reward.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "A base model writes a chain of thought when two worked examples sit in its prompt. A reasoning model writes longer ones with no examples. What differs?",
      options: [
        "Prompting updates the weights a little for that one answer, while training updates them far more",
        "A reasoning model has an extra reasoning layer in its architecture that a base model lacks",
        "Prompting changes only the context; training changes the weights so long traces become likelier",
        "A prompted chain is a faithful record of the work, while a trained chain is only decoration",
      ],
      answer: 2,
      explanation:
        "A prompt changes what the model reads and leaves the weights alone. Reinforcement learning against a checker changes the weights so that traces which end in a passing answer become likelier; that is why the Train card's policy moves and a prompt could not move it. The architecture is unchanged, and neither kind of trace is guaranteed to be faithful.",
      objective: 0,
    },
    {
      prompt:
        "In Train against a checker (k = 3, Length cost off) the pass rate is already close to 100% by update 50. What does Mean length do after that?",
      options: [
        "It keeps rising: nothing penalises extra steps and each one still adds a little pass chance",
        "It falls back toward k, because the checker only rewards the steps the task actually needs",
        "It stops moving, because a group where every trace passes always has zero advantage",
        "It falls, because every extra step is another chance to slip and fail the checker",
      ],
      answer: 0,
      explanation:
        "Advantage is measured against the group, so once almost every trace passes, the rare failures carry the signal and they tend to be the shorter traces. With no cost term the steps beyond k are free, and each still raises the chance that at least k of them land, so length creeps up. That drift is overthinking. Groups where every trace passes do give zero advantage, but not every group is like that.",
      objective: 1,
    },
    {
      prompt:
        "With Length cost on and Group size 8, seed 1 collapses to one-step answers and a pass rate near zero. Why?",
      options: [
        "The cost per step is bigger than the reward for passing, so every trace scores below zero",
        "The checker gets stricter when the cost is on, so fewer traces pass at every length",
        "Eight traces are too few to estimate a standard deviation, so every advantage comes out as zero",
        "No early group contains a pass, so cost is the only difference between traces and shorter wins",
      ],
      answer: 3,
      explanation:
        "Advantages are rewards relative to the group. When no trace in a group passes, the only thing that differs is the length cost, and dividing by the group standard deviation scales that small difference up to full-size advantages, so the update favours the shortest traces before any passing length has been found. The cost is 0.05 per step, so it never outweighs a pass; the failure is what normalisation does when there is no pass to compare against.",
      objective: 1,
    },
    {
      prompt: "A model writes numbered steps and then an answer. What is safe to conclude?",
      options: [
        "The numbered steps are a faithful record of the algorithm it actually ran",
        "It was trained or prompted to emit intermediate tokens before the answer",
        "In-context prompting and training a reasoning model are the same recipe",
        "Writing more steps always raises accuracy, so longer traces are always better",
      ],
      answer: 1,
      explanation:
        "Visible steps are generated tokens. They can be useful, because later tokens can read what earlier ones wrote, and still be unfaithful. Extra steps cost latency and help only where later steps need earlier results, so more is not always better. Prompting and training are different recipes for getting those tokens.",
      objective: 2,
    },
    {
      prompt:
        "At p = 0.40 with 16 samples you switch Wrong answers from Split over 3 to Always the same one. What happens?",
      options: [
        "Majority vote and Verifier picks both drop, because every wrong sample now counts against both",
        "Majority vote rises, because the wrong answers agree and the vote becomes more stable",
        "Majority vote falls below One sample, but Verifier picks does not change",
        "Neither changes, because both depend only on how many samples are drawn and on p",
      ],
      answer: 2,
      explanation:
        "A verifier needs only one correct sample, so its accuracy depends on p and n and not on how the wrong answers are spread. A vote needs the right answer to out-poll every wrong one: when all wrong samples are the same answer it collects about 60% of the votes against 40%, so more samples make the vote converge on the wrong answer.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateReasoningState,
};

export default definition;
