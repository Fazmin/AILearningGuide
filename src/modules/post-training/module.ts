import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { BETA_MAX, BETA_MIN, DEFAULT_LABELS, MAX_SFT_STEPS, encodeLabels, parseLabels } from "./prefs";

const initialState: ModuleState = {
  sftSteps: 3,
  labels: encodeLabels(DEFAULT_LABELS),
  beta: 2,
};

/** Version 1 stored { preference, strictness, round } for two closed-form lines; none of those keys survive. */
export function migratePostTrainingState(parsed: ModuleState): ModuleState {
  const sftSteps =
    typeof parsed.sftSteps === "number" && Number.isFinite(parsed.sftSteps)
      ? Math.min(MAX_SFT_STEPS, Math.max(0, Math.round(parsed.sftSteps)))
      : (initialState.sftSteps as number);
  const beta =
    typeof parsed.beta === "number" && Number.isFinite(parsed.beta) && parsed.beta > 0
      ? Math.min(BETA_MAX, Math.max(BETA_MIN, parsed.beta))
      : (initialState.beta as number);
  const labels = typeof parsed.labels === "string" ? encodeLabels(parseLabels(parsed.labels)) : (initialState.labels as string);
  return { sftSteps, labels, beta };
}

const definition: ModuleDefinition = {
  id: "module-12-post-training",
  slug: "post-training",
  title: "Post-training",
  group: "frontiers",
  order: 29,
  icon: "MessagesSquare",
  accent: "#c65767",
  prerequisites: ["module-07-next-token-prediction", "module-45-rl-foundations"],
  estimatedMinutes: 18,
  steps: ["Fine-tune on demonstrations", "Label preferences", "Optimize with a KL leash", "Break the length shortcut"],
  stepInstructions: [
    "Read the base bars in Post-training pipeline: the pretrained model favors A, another exam question. Raise SFT steps from 0 to 8 and watch mass move to the demonstrations B and F.",
    "In Preference pairs, click the response you prefer in each pair, then read the three weights and Agrees with gold in Reward model fit.",
    "Move KL coefficient β from 20 down to 0.02 and watch Reward-model score climb while Gold score rises, peaks, and falls.",
    "Label Pair 5 for B, the short answer that names the mechanism, then sweep KL coefficient β again and compare the Words ÷ 10 weight and the gold curve.",
  ],
  stateVersion: 2,
  tagline:
    "Run a whole post-training pipeline on one prompt: fine-tune on demonstrations, fit a Bradley–Terry reward model to your own labels, then optimize against it under a KL leash and watch a proxy score outrun the real one: the alignment problem in miniature.",
  objectives: [
    "Differentiate pretraining, supervised fine-tuning, reward modeling, and KL-regularized policy optimization",
    "Fit a Bradley–Terry reward model from pairwise preferences and read what it actually learned",
    "Identify reward hacking as a proxy score rising while true quality falls",
    "Explain alignment as the gap between what can be measured and what is wanted, and name the problems RLHF does not fix",
  ],
  glossary: [
    {
      term: "SFT",
      definition:
        "Supervised fine-tuning: ordinary next-token cross-entropy on curated prompt-and-response demonstrations, usually counted only on the response tokens. The loss is the pretraining loss; the data is what changed.",
    },
    {
      term: "Reward model",
      definition:
        "A learned function that maps a prompt and response to one scalar, trained on human comparisons and also called a preference model. Because it is an approximation fitted to limited comparisons, optimizing it hard rewards its mistakes along with the intended behavior.",
    },
    {
      term: "RLHF",
      definition:
        "Reinforcement learning from human feedback: fit a reward model to human comparisons, then optimize the language model to score well on it while a KL penalty keeps it near the fine-tuned starting point.",
    },
    {
      term: "DPO",
      definition:
        "Direct preference optimization: it uses the closed-form optimum of the KL-regularized objective to turn the reward-model loss into a loss on the policy itself, so it trains on preference pairs with no separate reward model and no sampling loop.",
    },
    {
      term: "KL divergence",
      definition:
        "The expected log-ratio between two distributions: the sum over outcomes of P times the natural log of P over Q, weighted by P. It is never negative, zero only when the two are identical, and not symmetric, so KL of P from Q differs from KL of Q from P. In nats when natural logs are used.",
    },
    {
      term: "KL penalty",
      definition:
        "A term, weighted by beta, that charges the policy for moving away from a reference model, usually the SFT model. It limits drift and slows reward hacking; it does not prevent it.",
    },
    {
      term: "Bradley-Terry",
      definition:
        "The pairwise model behind most reward-model training: the probability that A beats B is the sigmoid of r(A) minus r(B). Only reward differences matter, so adding a constant to every reward changes nothing.",
    },
    {
      term: "Reward hacking",
      definition:
        "Raising a learned or written reward without raising the quality it was meant to measure, an instance of Goodhart's law. Against a held-out gold scorer the proxy keeps rising while the gold score peaks and falls; a verifiable reward from a real checker, such as unit tests, removes the learned scorer as a target but exists only for checkable tasks.",
    },
    {
      term: "Constitutional AI",
      definition:
        "A method in which a model critiques and revises its own outputs against written principles to make fine-tuning data, then judges pairs of outputs against those principles to make preference data. Human effort shifts from labeling to writing the principles.",
    },
    {
      term: "Alignment",
      definition:
        "Getting a model's behavior to match what its developers and users intend, when intent can only be reached through proxies such as demonstrations, comparisons, and scores. Outer alignment asks whether the specified objective is the right one; inner alignment asks whether the trained model actually pursues it. Alignment training can also cost some benchmark performance, an alignment tax.",
    },
    {
      term: "Goal misgeneralization",
      definition:
        "A trained policy that pursues a different goal from the intended one while still matching the training reward on every training situation, so the difference only shows in new situations. It is documented in small reinforcement-learning environments; how far it explains the behavior of large language models is an open question.",
    },
    {
      term: "Scalable oversight",
      definition:
        "Methods for supervising outputs that people cannot easily check, such as long programs or subtle arguments, for example with model-written critiques for raters, debate between models, or feedback against written principles. It is active research: some methods help in controlled studies, and none is established as enough for a model more capable than its overseers.",
    },
    {
      term: "Sycophancy",
      definition:
        "Agreeing with the user against the evidence, including abandoning a correct answer after mild pushback. It is a measured behavior of preference-trained models: studies found that people and reward models sometimes rate agreeable answers above correct ones, so optimizing against them can raise it.",
    },
    {
      term: "Dangerous-capability evaluation",
      definition:
        "A test, run before release, of whether a model can do something seriously harmful if misused or if it acted on its own, such as giving real uplift toward weapons or carrying out cyber operations. Developers' safety frameworks tie the results to safeguards; failing a test under one attempt shows the model did not do it then, not that it cannot.",
    },
  ],
  references: [
    {
      authors: "Paul F. Christiano, Jan Leike, Tom Brown, et al.",
      title: "Deep Reinforcement Learning from Human Preferences",
      source: "Advances in Neural Information Processing Systems 30 (NIPS 2017)",
      year: 2017,
      url: "https://proceedings.neurips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html",
      note: "The early paper behind learning a reward from comparisons. People pick the better of two short clips, a reward model is fitted to those picks, and an agent learns games and robot tasks from it, the same comparisons-are-cheap idea as the Preference pairs card.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html",
      note: "The InstructGPT paper, which runs the lesson's three stages at full scale: fine-tuning on demonstrations, a reward model trained on rankings, and PPO with a per-token KL penalty from the SFT model. It also names the alignment tax, a drop on some public benchmarks after this training.",
    },
    {
      authors: "Ralph Allan Bradley and Milton E. Terry",
      title: "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons",
      source: "Biometrika 39(3/4)",
      year: 1952,
      url: "https://www.jstor.org/stable/2334029",
      note: "The original statistics paper for the Bradley-Terry model, written for ranking items from pairwise comparisons long before language models. The Reward model fit card uses the same idea: the chance that A beats B depends only on the gap between their scores.",
    },
    {
      authors: "John Schulman, Filip Wolski, Prafulla Dhariwal, et al.",
      title: "Proximal Policy Optimization Algorithms",
      source: "arXiv preprint arXiv:1707.06347",
      year: 2017,
      url: "https://arxiv.org/abs/1707.06347",
      note: "The PPO paper. Its clipped objective keeps each update small, which is how RLHF approximates the optimum that this lab computes exactly with no sampling or clipping.",
    },
    {
      authors: "Rafael Rafailov, Archit Sharma, Eric Mitchell, et al.",
      title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html",
      note: "The DPO paper. It uses the closed-form optimum of the KL-regularized objective, the same formula the lab computes, to train the policy straight from preference pairs with no separate reward model and no sampling.",
    },
    {
      authors: "Leo Gao, John Schulman, and Jacob Hilton",
      title: "Scaling Laws for Reward Model Overoptimization",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/gao23h.html",
      note: "The Gao et al. study the lesson cites. A large gold reward model stands in for people; as a policy is optimized against a smaller proxy, the proxy score keeps rising while the gold score peaks and falls, the curve on the Optimize against the reward model card. In their setup the KL penalty acted like stopping early rather than a cure.",
    },
    {
      authors: "Stephen Casper, Xander Davies, Claudia Shi, et al.",
      title: "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback",
      source: "Transactions on Machine Learning Research (TMLR), December 2023",
      year: 2023,
      url: "https://arxiv.org/abs/2307.15217",
      note: "A survey of what RLHF does not fix, from limits of human feedback to flaws in reward models and policies. It backs the lesson's sections Where it breaks and What alignment is, and what RLHF does not fix.",
    },
    {
      authors: "Lauro Langosco Di Langosco, Jack Koch, Lee D. Sharkey, et al.",
      title: "Goal Misgeneralization in Deep Reinforcement Learning",
      source: "Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162, 12004–12019",
      year: 2022,
      url: "https://proceedings.mlr.press/v162/langosco22a.html",
      note: "The Langosco et al. study the lesson cites for goal misgeneralization. In the CoinRun game an agent trained with the coin always at the far right keeps running right after the coin is moved: still skilled, but chasing the wrong goal.",
    },
    {
      authors: "Ethan Perez, Sam Ringer, Kamile Lukosiute, et al.",
      title: "Discovering Language Model Behaviors with Model-Written Evaluations",
      source: "Findings of the Association for Computational Linguistics: ACL 2023, 13387–13434",
      year: 2023,
      url: "https://aclanthology.org/2023.findings-acl.847/",
      note: "The Perez et al. study the lesson cites for sycophancy. It uses language models to write test sets and finds that larger models repeat back a user's preferred answer, and that more RLHF can make some behaviors worse.",
    },
    {
      authors: "Mrinank Sharma, Meg Tong, Tomek Korbak, et al.",
      title: "Towards Understanding Sycophancy in Language Models",
      source: "The Twelfth International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://proceedings.iclr.cc/paper_files/paper/2024/hash/0105f7972202c1d4fb817da9f21a9663-Abstract-Conference.html",
      note: "The Sharma et al. study the lesson cites. Five assistants showed sycophancy, and both people and preference models sometimes preferred a convincing agreeable answer over a correct one, the mechanism behind the lesson's sycophancy checkpoint.",
    },
    {
      authors: "Evan Hubinger, Carson Denison, Jesse Mu, et al.",
      title: "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training",
      source: "arXiv preprint arXiv:2401.05566",
      year: 2024,
      url: "https://arxiv.org/abs/2401.05566",
      note: "The Hubinger et al. study the lesson cites under Deceptive behavior. Models built on purpose with a hidden trigger, such as writing unsafe code when the year reads 2024, kept that behavior through fine-tuning, RL, and adversarial training.",
    },
    {
      authors: "Teun van der Weij, Felix Hofstätter, Oliver Jaffe, et al.",
      title: "AI Sandbagging: Language Models can Strategically Underperform on Evaluations",
      source: "The Thirteenth International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://proceedings.iclr.cc/paper_files/paper/2025/hash/b5e5753b0a0e440a6d8dc7e143617cec-Abstract-Conference.html",
      note: "The van der Weij et al. study of sandbagging the lesson cites. Models could be prompted or fine-tuned to do badly on dangerous-capability tests while doing well elsewhere, which is why a failed test does not prove a capability is absent.",
    },
    {
      authors: "Toby Shevlane, Sebastian Farquhar, Ben Garfinkel, et al.",
      title: "Model evaluation for extreme risks",
      source: "arXiv preprint arXiv:2305.15324",
      year: 2023,
      url: "https://arxiv.org/abs/2305.15324",
      note: "Researchers from several AI labs explain dangerous-capability evaluations and why their results should feed decisions about training, deployment, and security. It backs the lesson's section How developers check before release.",
    },
    {
      authors: "Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al.",
      title: "Constitutional AI: Harmlessness from AI Feedback",
      source: "arXiv preprint arXiv:2212.08073",
      year: 2022,
      url: "https://arxiv.org/abs/2212.08073",
      note: "The Constitutional AI paper. A model critiques and revises its own answers against written principles for fine-tuning data, then an AI judge picks the better of two answers to train a preference model, so human effort goes into writing the principles.",
    },
    {
      authors: "Daya Guo, Dejian Yang, Haowei Zhang, et al.",
      title: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
      source: "Nature 645, 633–638",
      year: 2025,
      url: "https://www.nature.com/articles/s41586-025-09422-z",
      note: "A large-scale example of the Going deeper ideas. It trains with GRPO, which scores a group of sampled answers to the same prompt against each other, and uses rule-based rewards such as answer checks for math and code because learned reward models were open to reward hacking.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "You raise SFT steps from 0 to 8 and change nothing else. What moves?",
      options: [
        "Probability shifts toward the demonstrations B and F, away from the base model's favorite, A",
        "The reward model's three weights shift toward whatever the demonstrations prefer",
        "The gold score of each response rises, because SFT re-grades the responses against the demonstrations",
        "KL coefficient β falls, because SFT sets how far the policy may drift from the base model",
      ],
      answer: 0,
      explanation:
        "SFT is ordinary next-token cross-entropy on the demonstrations, so each step moves probability toward B and F and away from A. It never looks at a reward: the reward model is fitted separately from your comparisons, gold is an authored grade, and β is a setting for the later optimization stage. SFT changes the starting point, π_SFT, that the KL leash later holds the policy near.",
      objective: 0,
    },
    {
      prompt:
        "You label Pair 5 for B, the short answer that names the mechanism, and leave the other labels alone. What happens to the Words ÷ 10 weight, and why?",
      options: [
        "It rises, because one more label is more evidence that longer answers are better in general",
        "It falls, because a longer response now loses a comparison and length stops explaining every win",
        "It stays the same, because the reward model reads only whether an answer names the mechanism at all",
        "It turns negative, because the reward model now treats length as something that it should avoid",
      ],
      answer: 1,
      explanation:
        "Bradley–Terry fits whatever feature differences separate winners from losers. In the default labels every winner is also the longer response, so length explains every pair. Pair 5 puts a longer, empty answer against a shorter, correct one, so length stops being a reliable reason to win and its weight falls, though it stays positive because other pairs still reward it.",
      objective: 1,
    },
    {
      prompt:
        "Every pair you labeled put the longer response on top, and the reward model gives words a large positive weight. You lower KL coefficient β until KL from the SFT model is large. What happens to the gold score?",
      options: [
        "It keeps rising, because the reward model was fit to your own preferences and so tracks them",
        "It falls from the start, because any move away from the SFT model must lower the quality of answers",
        "It stays where it was, because β only changes the reference model and leaves the policy alone",
        "It rises at first as junk answers lose mass, then falls once the policy piles onto the long, empty answer",
      ],
      answer: 3,
      explanation:
        "Lowering β lets exp(r/β) dominate the reference probabilities. Early on that removes mass from A and D, which both scorers dislike, so gold rises. Past the peak the policy piles onto E, the long response with no mechanism, because length was the feature your labels could not separate from quality. The proxy keeps rising; the gold score falls.",
      objective: 2,
    },
    {
      prompt:
        "In this lab the tuned policy is the exact optimum of the reward model. Which alignment problem therefore cannot appear in it?",
      options: [
        "A reward model that overrates length, so the proxy keeps rising while the gold score falls",
        "A policy that drifts far away from the SFT model when the KL coefficient is small",
        "A policy that learned a goal of its own, matching the training reward on training prompts only",
        "A reward model that fits your labels but disagrees with the careful rater's grades on other pairs",
      ],
      answer: 2,
      explanation:
        "Everything the lab shows is a failure of the objective: the policy does exactly what the proxy says and the proxy is wrong. A policy that internalized a different goal, called goal misgeneralization, needs a learned policy whose behavior can agree with the reward on training inputs and part from it on new ones. The lab's policy is defined by the reward, so it has no goal of its own that could diverge.",
      objective: 3,
    },
    {
      prompt:
        "Why can preference training encourage sycophancy, as the lesson describes it?",
      options: [
        "SFT demonstrations are full of users who are wrong, so the model learns to confirm whatever they say",
        "A KL penalty pays the policy for repeating whatever the user has just said to it in the prompt",
        "Reward models cannot read the user's message, so they score every possible reply as equally good",
        "People and reward models sometimes rate agreement above correctness, so optimizing the reward favors agreement",
      ],
      answer: 3,
      explanation:
        "It is the length shortcut again with a different feature. If raters or a reward model score agreement slightly higher than correctness, optimizing against that score moves the model toward agreeing, and the proxy rises while quality falls. Studies measured this behavior in preference-trained assistants. The KL penalty only limits drift, and a reward model does read the prompt.",
      objective: 3,
    },
    {
      prompt:
        "A developer's dangerous-capability evaluation finds that a model fails a test of a harmful skill. What does the result establish?",
      options: [
        "That the model cannot do it, because a model that could would always succeed on the test",
        "That the model did not do it on this attempt, and a stronger attempt might still draw it out",
        "That the skill is safe to ship, because such evaluations are run only on models known to be risky",
        "That the model is deceptive, because failing a test on purpose shows an intent to hide a skill",
      ],
      answer: 1,
      explanation:
        "A failed test is evidence about one attempt, not a proof of absence. Weak prompting, missing tools, or a model that underperforms on purpose can all produce a failure, so evaluators also try fine-tuning and scaffolding to draw the skill out. Safety frameworks use these results to decide which safeguards a developer commits to, but they are commitments, not guarantees.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      return migratePostTrainingState(parsed);
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
