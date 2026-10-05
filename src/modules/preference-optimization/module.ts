import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { isReferenceChoice, sanitizePairsText } from "./dpo";

const initialState: ModuleState = {
  beta: 0.4,
  steps: 40,
  learningRate: 0.35,
  reference: "base",
  pairsText: [
    "the | fog settles over the harbor. | fog fog fog fog fog.",
    "warm the | pan and add a spoon of oil. | pan and the pan and the pan.",
    "simmer the | sauce until it thickens. | sauce and simmer the sauce.",
  ].join("\n"),
};

const numberIn = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;

const definition: ModuleDefinition = {
  id: "module-24-preference-optimization",
  slug: "preference-optimization",
  title: "Preference optimization in practice",
  group: "training-adapting",
  order: 45,
  icon: "Scale",
  accent: "#5d8a2f",
  prerequisites: ["module-12-post-training", "module-23-instruction-tuning"],
  estimatedMinutes: 18,
  steps: ["Author the comparisons", "Run the optimizer", "Read the margin", "Trade drift for control", "Test held-out pairs"],
  stepInstructions: [
    "Read the three default pairs, then write one of your own where the rejected completion repeats itself.",
    "Raise DPO steps from 0 and watch DPO loss fall from 0.6931 while the numbered pairs slide right along the loss curve.",
    "Compare the margin line with the chosen reward and rejected reward lines, and read which side moved further.",
    "Sweep Beta and compare the three drift curves, reading each one's gap and KL in the legend.",
    "Under Per-pair rewards, compare the three training pairs with the H1 to H3 held-out pairs, then switch Reference model to SFT checkpoint and read which pairs the reference already preferred.",
  ],
  stateVersion: 2,
  tagline:
    "Post-training explained preference learning. This one runs it: real DPO gradients on comparisons you write, with the reward margin and the drift from the reference model both measured.",
  objectives: [
    "Run preference optimization on comparisons you authored",
    "Read an implicit reward margin correctly",
    "Explain what beta trades away",
    "Judge whether a preference gain carries over by scoring pairs the optimizer never saw",
  ],
  glossary: [
    {
      term: "DPO",
      definition:
        "Direct preference optimization (Rafailov et al., 2023). A classification-style loss on preference pairs, derived from the closed-form optimum of KL-regularized RLHF, so it needs no separate reward model and no sampling loop.",
    },
    {
      term: "Reference model",
      definition:
        "A frozen copy of the starting model. Every reward in DPO is defined relative to it, which is what keeps the policy from drifting into nonsense.",
    },
    {
      term: "Implicit reward",
      definition:
        "Beta times the log of the policy's probability over the reference model's for a completion. It equals the reward DPO implicitly fits up to a per-prompt constant, so it compares completions of one prompt and is never a quality score.",
    },
    {
      term: "Margin",
      definition:
        "The implicit reward of the preferred completion minus that of the rejected one. The DPO loss is a decreasing function of exactly this number.",
    },
    {
      term: "Beta",
      definition:
        "The KL-penalty strength inherited from RLHF. In the DPO loss it scales the margin, so a higher beta saturates the loss with a smaller log-ratio change and ends nearer the reference. Under plain gradient descent it also scales every step, so early on a higher beta moves faster.",
    },
    {
      term: "KL divergence",
      definition:
        "How far one distribution sits from another, in nats. Here it is KL from policy to reference averaged evenly over all 30 context rows, a whole-table measure of drift; RLHF's penalty is instead taken over the policy's own sampled completions.",
    },
    {
      term: "Length bias",
      definition:
        "Two effects of length. Raw total log-probability always favours the shorter completion, and preference data in which longer answers tend to win teaches the policy to lengthen its answers.",
    },
    {
      term: "Degenerate repetition",
      definition:
        "Looping on a phrase. It is a high-probability failure for small models, which makes it a useful thing to label against.",
    },
    {
      term: "Reward hacking",
      definition:
        "Improving the measured objective while real quality stalls. In DPO the objective is the margin, so a rising margin is evidence about the margin and nothing else.",
    },
    {
      term: "Policy",
      definition:
        "The model being trained, as opposed to the frozen reference. Here it starts as an exact copy of the reference table and is the only set of weights DPO updates.",
    },
    {
      term: "RLHF",
      definition:
        "Reinforcement learning from human feedback: fit a reward model to preference comparisons, then train the model to raise that reward while a KL penalty holds it near a reference. DPO reaches the same optimum without the separate reward model or the sampling loop.",
    },
    {
      term: "Bradley-Terry",
      definition:
        "The model of pairwise comparisons in which the chance of preferring one completion is the sigmoid of the difference between the two rewards. DPO's loss is the negative log of that likelihood with the implicit reward substituted in, so it depends only on the margin.",
    },
    {
      term: "Held-out pairs",
      definition:
        "Comparisons the optimizer never trains on, scored by the finished policy with the same margin as the training pairs. A margin that is large on the training pairs and small or negative on held-out pairs means the policy fitted those pairs rather than the preference behind them.",
    },
    {
      term: "SFT checkpoint",
      definition:
        "A model already fine-tuned on instruction and response pairs. DPO normally starts from one and freezes a copy of it as the reference, so every reward measures movement away from the supervised model rather than from a raw base.",
    },
  ],
  references: [
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 27730–27744",
      year: 2022,
      url: "https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html",
      note: "The InstructGPT paper lays out the RLHF pipeline this lesson starts from: supervised fine-tuning first, then a reward model fitted to ranked outputs, then reinforcement learning with a KL penalty that holds the model near the SFT checkpoint.",
    },
    {
      authors: "Rafael Rafailov, Archit Sharma, Eric Mitchell, et al.",
      title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 53728–53741",
      year: 2023,
      url: "https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html",
      note: "The DPO paper. It derives the loss from the KL-regularized RLHF goal and the Bradley-Terry model, shows that the gradient pulls hardest on pairs the implicit reward orders wrongly, and says to fall back on fine-tuning on the preferred completions when no SFT model exists, which is how the SFT checkpoint option here is built.",
    },
    {
      authors: "Nathan Lambert",
      title: "Reinforcement Learning from Human Feedback: Alignment and post-training of LLMs",
      source: "Manning Publications, free to read online",
      year: 2026,
      url: "https://rlhfbook.com/c/08-direct-alignment",
      note: "The Direct-Alignment Algorithms chapter walks through the DPO derivation step by step, shows that DPO often lowers both the chosen and the rejected completion while the rejected one falls further, and surveys IPO, ORPO, SimPO and online DPO, the variants this lesson lists.",
    },
    {
      authors: "Hugging Face",
      title: "DPO Trainer",
      source: "TRL documentation",
      year: 2026,
      url: "https://huggingface.co/docs/trl/dpo_trainer",
      note: "The documentation for a widely used DPO trainer. Its logged metrics are the readouts this lab plots: the chosen and rejected implicit rewards, their margin, and the share of pairs where the chosen reward is higher, matching the ordered correctly badge.",
    },
    {
      authors: "Diederik P. Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      source: "International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1412.6980",
      note: "The paper behind the Adam optimizer, which scales each parameter's step by a running estimate of its gradient size. That rescaling is why the lesson says beta's step-size role, and the early flip in the Drift from the reference model chart, largely fades in production runs.",
    },
    {
      authors: "Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, et al.",
      title: "A General Theoretical Paradigm to Understand Learning from Human Preferences",
      source: "Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024), PMLR 238, 4447–4455",
      year: 2024,
      url: "https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html",
      note: "Introduces IPO and explains, in its section on weak regularisation and overfitting, that when each pair is labeled one way only, DPO keeps pushing the margin up and the KL anchor stops holding it back. That is the pattern behind the held-out pairs check, where training margins climb while H1 and H2 get worse.",
    },
    {
      authors: "Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, et al.",
      title: "Iterative Reasoning Preference Optimization",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 116617–116637",
      year: 2024,
      url: "https://papers.nips.cc/paper_files/paper/2024/hash/d37c9ad425fe5b65304d500c6edcba00-Abstract-Conference.html",
      note: "Shows that under plain DPO the log-probability of the chosen answers fell during training while the margin kept growing, most of all when the model started from SFT on those answers. Adding a likelihood term on the preferred side kept it rising, the fix the lesson names for the both-sides-fell outcome.",
    },
    {
      authors: "Yu Meng, Mengzhou Xia, and Danqi Chen",
      title: "SimPO: Simple Preference Optimization with a Reference-Free Reward",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 124198–124235",
      year: 2024,
      url: "https://papers.nips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html",
      note: "SimPO uses a completion's average log-probability per token as its reward, so longer answers are not penalized for their length, and it needs no reference model. It is the length-normalized, reference-free variant the lesson names.",
    },
    {
      authors: "Jiwoo Hong, Noah Lee, and James Thorne",
      title: "ORPO: Monolithic Preference Optimization without Reference Model",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), 11170–11189",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-main.626/",
      note: "ORPO folds the preference step into supervised fine-tuning with an odds-ratio penalty on the rejected answer, so there is no frozen reference model to keep in memory. It is the other reference-free variant the lesson names.",
    },
    {
      authors: "Shusheng Xu, Wei Fu, Jiaxuan Gao, et al.",
      title: "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 54983–54998",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/xu24h.html",
      note: "Finds that DPO can settle on solutions that favour answers outside the preference data, and that its results depend on the gap between the model's own outputs and that data. This backs the lesson's warning about off-policy pairs written by some other model.",
    },
    {
      authors: "Shangmin Guo, Biao Zhang, Tianlin Liu, et al.",
      title: "Direct Language Model Alignment from Online AI Feedback",
      source: "arXiv preprint arXiv:2402.04792",
      year: 2024,
      url: "https://arxiv.org/abs/2402.04792",
      note: "A Google DeepMind study of online DPO: at each step it samples two answers from the current model and has another model pick the better one. It is an example of the online variants the lesson describes, and it explains why fixed, off-policy pairs are a limit.",
    },
    {
      authors: "Ryan Park, Rafael Rafailov, Stefano Ermon, et al.",
      title: "Disentangling Length from Quality in Direct Preference Optimization",
      source: "Findings of the Association for Computational Linguistics: ACL 2024, 4998–5017",
      year: 2024,
      url: "https://aclanthology.org/2024.findings-acl.297/",
      note: "Shows that DPO exploits length: a small lean toward longer winners in the data makes the trained model write much longer answers. It also gives a simple length penalty. This is the length bias described under Where it breaks.",
    },
    {
      authors: "Prasann Singhal, Tanya Goyal, Jiacheng Xu, et al.",
      title: "A Long Way to Go: Investigating Length Correlations in RLHF",
      source: "Conference on Language Modeling (COLM 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.03716",
      note: "Finds that much of the reward gain from RLHF comes from longer answers, and traces it to length biases in the preference data. It backs the lesson's point that labelers' habits, such as favouring length, end up in the trained model.",
    },
    {
      authors: "Rafael Rafailov, Yaswanth Chittepu, Ryan Park, et al.",
      title: "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 126207–126242",
      year: 2024,
      url: "https://papers.nips.cc/paper_files/paper/2024/hash/e45caa3d5273d105b8d045e748636957-Abstract-Conference.html",
      note: "Shows that DPO and similar methods still over-optimize even with no separate reward model: quality gets worse as the policy moves further from the reference, sometimes within the first epoch. This backs the lesson's claim that reward hacking survives in DPO because the margin is now the quantity being gamed.",
    },
  ],
  checkpoint: [
    {
      prompt: "At DPO steps 0 the loss reads 0.6931. What is that number telling you?",
      options: [
        "The learning rate is too small for the loss to start moving, so the optimizer has not taken a step",
        "Every margin is zero because the policy is a copy of the reference, and the loss at zero margin is ln 2",
        "The chosen completions are exactly as likely as the rejected ones under the base model, so nothing is learned yet",
        "Every untrained policy scores 0.6931 whatever the data, because the loss is normalised to start there",
      ],
      answer: 1,
      explanation:
        "Each margin is the policy's log-ratio gap minus the reference's, and at step 0 the policy is the reference, so every margin is exactly zero. The loss is then −log sigmoid(0) = ln 2 = 0.6931 whatever the data. It does not mean the two completions are equally likely: in the default pairs the base model rates the repeating loop as the more likely one.",
      objective: 0,
    },
    {
      prompt:
        "After DPO the margin has grown, yet both the chosen and the rejected completion have lower log-probability than before. Is something wrong?",
      options: [
        "Yes. A correct run must raise the log-probability of the preferred completion on every pair it trains on",
        "Yes. It means beta was set too low for the data, so the margin could not be scaled up enough",
        "No. The loss rewards only the gap, so it is satisfied by pushing the rejected side down faster",
        "Yes. The reference model was updated by mistake, so the log-ratios no longer mean anything at all",
      ],
      answer: 2,
      explanation:
        "DPO optimizes a difference. Lowering both sides while widening the gap satisfies it perfectly, and it is a common outcome in practice. In this lab the chosen side rises a little and the rejected side falls much further, which is why the two sides are worth plotting separately rather than trusting the margin alone.",
      objective: 1,
    },
    {
      prompt: "Which comparison is an implicit reward built to support?",
      options: [
        "Rewards from different prompts, because beta scales every prompt's rewards onto one common quality scale",
        "Any two completions of any prompt, because the reward is the policy's log-probability alone",
        "The preferred completion against a human rating, because the reward estimates that rating directly",
        "Chosen against rejected on the same prompt, because each reward is relative to the reference's probability",
      ],
      answer: 3,
      explanation:
        "An implicit reward is beta times the log of the policy's probability over the reference's, which equals the fitted reward only up to a constant for each prompt. That constant cancels inside a pair, so a chosen-versus-rejected gap on one prompt is meaningful. Across prompts, or as a quality score, it is not.",
      objective: 1,
    },
    {
      prompt:
        "Sweeping Beta from 0.2 to 0.8 at 40 steps, the beta-scaled margin rises but the log-ratio gap falls. What does that show?",
      options: [
        "A higher beta trained the policy better, because the larger scaled margin means the pairs are separated further apart",
        "Beta scales the margin, so the loss saturates after a smaller log-ratio change and the policy ends nearer the reference",
        "Beta cancels out of the loss, so only the step size changed between runs and the gap fell by chance",
        "The optimizer diverged at the higher beta, so the gap shrank even though the scaled margin kept growing",
      ],
      answer: 1,
      explanation:
        "Beta multiplies the log-ratio gap inside the sigmoid, so a larger beta reaches any given loss with a smaller gap and the gradient flattens sooner. The scaled margin is therefore not comparable across betas: compare runs by the unscaled gap and by KL. A larger beta ends nearer the reference, which is the control it buys.",
      objective: 2,
    },
    {
      prompt:
        "In the lab a higher beta drifts less at 40 steps but more at 4 steps. A production run uses Adam instead of plain gradient descent. What happens to that early flip?",
      options: [
        "It gets stronger, because Adam takes larger steps early in training than plain gradient descent does",
        "It stays exactly as shown, because the step-size role comes from the loss and not from the optimizer",
        "It largely disappears: Adam rescales each step, so beta's step-size role fades and saturation dominates",
        "It reverses at every step count, because Adam makes a higher beta always drift farther from the reference",
      ],
      answer: 2,
      explanation:
        "Under plain gradient descent beta multiplies every gradient, so a higher beta also takes bigger steps, and that wins early. Adam normalises each parameter's step size, so that role fades and beta's other role, saturating the loss sooner, is what remains. The early flip is a property of the lab's optimizer, not of DPO.",
      objective: 2,
    },
    {
      prompt:
        "At the defaults every training pair ends ordered correctly, but fewer held-out pairs do. What does that show?",
      options: [
        "The optimizer fitted the pairs it trained on, and ordering new comparisons is a separate fact to check",
        "The held-out pairs are mislabelled, because a correct preference would be ordered like the training pairs",
        "Beta is set too high, because a lower beta would order every held-out pair just as well as the training ones",
        "DPO has not run long enough, because more steps always order more of the pairs it never trained on",
      ],
      answer: 0,
      explanation:
        "The loss grades only the training margins, so a high training margin is evidence about those pairs. The held-out pairs share some character transitions with them, which is why one held-out pair carries over while two end with their margin reversed. More steps make those two worse, not better. A check on unseen pairs is the only evidence that the preference itself was learned.",
      objective: 3,
    },
    {
      prompt:
        "You switch Reference model to SFT checkpoint. At DPO steps 0 the loss still reads 0.6931 although the reference now orders some training pairs correctly. Why?",
      options: [
        "The policy starts as a copy of whichever reference is chosen, so every margin is zero whatever it prefers",
        "The SFT checkpoint only changes the samples, so the loss ignores it until the first optimizer step",
        "The optimizer resets the loss to its starting value on every pair, so the reference cannot move it",
        "The SFT reference still prefers the loops on every pair, because SFT leaves raw probabilities as they were",
      ],
      answer: 0,
      explanation:
        "A margin is the policy's log-ratio gap minus the reference's, and at step 0 the policy is the reference, so every margin is exactly zero and the loss is ln 2 whatever the reference prefers. SFT does change the raw preferences: the reference now ranks two of the three training pairs correctly, which the Per-pair rewards rows state as already preferred it.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      // Version 2 added `reference`; a version 1 payload trained against the base table, which is the default.
      // Every key is validated and the pairs are cut to the editor's limits, because training runs in the render.
      return {
        beta: numberIn(parsed.beta, 0.05, 1.5, 0.4),
        steps: Math.round(numberIn(parsed.steps, 0, 120, 40)),
        learningRate: numberIn(parsed.learningRate, 0.05, 1, 0.35),
        reference: isReferenceChoice(parsed.reference) ? parsed.reference : "base",
        pairsText: typeof parsed.pairsText === "string" ? sanitizePairsText(parsed.pairsText) : initialState.pairsText,
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
