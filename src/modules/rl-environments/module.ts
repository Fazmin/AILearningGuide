import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateRlEnvironmentsState, initialState } from "./state";

/**
 * Version 1 is the first release. `hydrateRlEnvironmentsState` rebuilds every field through
 * `sanitizeState`, because the training run and the 10-seed strip happen synchronously in a memo.
 */
export { hydrateRlEnvironmentsState };

const definition: ModuleDefinition = {
  id: "module-58-rl-environments",
  slug: "rl-environments",
  title: "RL environments & agents",
  group: "training-adapting",
  order: 57,
  icon: "Ticket",
  accent: "#5a62b0",
  prerequisites: ["module-46-reasoning-models", "module-41-agents", "module-45-rl-foundations"],
  estimatedMinutes: 18,
  steps: [
    "Run one episode",
    "Train on the outcome",
    "Leave the test writable",
    "Swap the scorer",
    "Add a leash and a cost",
    "Test unseen tickets",
  ],
  stepInstructions: [
    "Set Training update to 0, then in Ticket desk step Episode seed from 1 to 5 and compare the Sandbox test, Strict grader, Partial credit and Rubric judge rows of Scorecard.",
    "Set Reward design to Outcome only and Checker to Strict grader, drag Training update from 0 to 200 and watch True success (held-out) rise; then set Group size to 2 and read Groups with no signal.",
    "Set Group size back to 8 and Checker to Sandbox test, compare Training reward with True success (held-out) at update 200, then find the edit_check call in Episode replay.",
    "Set Reward design to Rubric judge and read Closed but not fixed; then pick Partial credit and compare Hack rate and Needless look-ups with Outcome only.",
    "Set Reward design to Outcome only, keep Sandbox test and raise KL to reference β to 0.2, reading Hack rate and KL per turn; then set Checker to Strict grader, β to 0 and Step cost to 0.05, and switch Advantage scale between Divide by group std and Mean only while watching Mean turns.",
    "Set Step cost to 0 and Advantage scale to Divide by group std, then switch Training tickets between 1 kind and 6 kinds and compare the Value missing bar in Unseen tickets with the Across 10 seeds strip.",
  ],
  stateVersion: 1,
  tagline:
    "Train a 48-weight agent with group-relative updates on a toy ticket desk, then watch it game a writable test and a rubric judge, and see what a KL leash, a step cost and more varied tickets do and do not fix.",
  objectives: [
    "Describe an RL environment for a tool-using agent, with its tasks, tools, turn cap and end-state check, and explain why the grader must sit outside the sandbox the agent can change",
    "Explain how group-relative updates turn whole episodes into a policy update, and when a group gives no signal",
    "Compare an outcome reward, partial credit and a rubric judge, and predict which one training will game and how",
    "Diagnose reward hacking as a gap between training reward and a held-out strict grader, and say what a KL penalty and a step cost do and do not fix",
    "Explain why the variety of training tasks decides how an agent does on tasks it never saw",
  ],
  glossary: [
    {
      term: "RL environment",
      definition:
        "The program an agent acts in while it trains: it hands out a task, runs each action, returns what happened, and scores the episode at the end. Its tasks, tools and checks decide what the agent can learn, including what it can get away with.",
    },
    {
      term: "Episode (trajectory)",
      definition:
        "One run of the agent on one task, from the first observation to the end: every tool call, every reply, and the final state. The whole episode gets one reward, so training has to share that credit across its turns.",
    },
    {
      term: "Tool action",
      definition:
        "One call the agent makes to the environment, such as read_ticket or update_record. In this lab the policy only chooses which of six tools to call; a real agent also writes the arguments as text.",
    },
    {
      term: "End-state check",
      definition:
        "A grader that inspects the state the episode left behind, such as whether the record holds the right value, instead of reading the transcript. It cannot be talked into a pass, but it checks only what it was written to check.",
    },
    {
      term: "Sandbox",
      definition:
        "The isolated workspace the agent acts in, so its actions cannot touch real systems. Anything inside the sandbox, including a test stored next to the code, is something the agent can change.",
    },
    {
      term: "Outcome reward",
      definition:
        "A reward paid only for the final result, here 1 when the checker passes and 0 otherwise. It is hard to fake when the checker is sound, and it gives no signal on a task where every episode fails.",
    },
    {
      term: "Partial credit",
      definition:
        "Extra reward for intermediate steps, here 0.1 for reading, 0.1 for a look-up and 0.2 once the record is right, on top of the outcome. Bonuses for steps can change which behaviour scores best, so they can teach steps the task does not need.",
    },
    {
      term: "Rubric judge",
      definition:
        "A grader that scores the transcript against a checklist, standing in for a language model used as a judge. It rewards what the transcript shows, so an agent can tick every item without doing the task.",
    },
    {
      term: "Reward hacking",
      definition:
        "A policy raising its training reward by a route that does not do the intended task. It is the optimizer working correctly on a reward that pays for the wrong thing, and it shows up as training reward rising while a separate, trusted measure stays flat or falls.",
    },
    {
      term: "Loophole",
      definition:
        "A specific gap in an environment that lets an episode score without doing the task, such as a test the agent can edit. Training exploits a loophole once it stumbles on it and the loophole pays more reliably than the honest route.",
    },
    {
      term: "Group-relative advantage",
      definition:
        "Each episode's reward minus the mean reward of its group of episodes on the same task, divided by the group's standard deviation in the original GRPO recipe. A group whose rewards are all equal gives every episode zero advantage, so it teaches nothing.",
    },
    {
      term: "KL penalty",
      definition:
        "A cost on how far the policy's action probabilities move from a reference policy, measured by the KL divergence and weighted by β. It slows drift, including drift toward a loophole, but it does not remove the reward that pays for the loophole.",
    },
    {
      term: "Step cost",
      definition:
        "A small amount subtracted from the reward for every turn, to favour shorter episodes. When a whole group fails, the cost is the only difference left between its episodes, and dividing by the group's spread makes that difference as large as a success.",
    },
    {
      term: "Held-out variant",
      definition:
        "A task kept out of training and used only to measure the trained policy. It tests generalization only when it differs from the training tasks in a way the policy can perceive.",
    },
  ],
  references: [
    {
      authors: "Zhihong Shao, Peiyi Wang, Qihao Zhu, et al.",
      title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
      source: "arXiv preprint arXiv:2402.03300",
      year: 2024,
      url: "https://arxiv.org/abs/2402.03300",
      note: "Introduces GRPO, which drops the value model, uses the average reward of several answers to the same question as the baseline, divides by the group's standard deviation, and adds a KL penalty to a reference model directly to the loss. The Train with group-relative updates card runs that update on whole episodes.",
    },
    {
      authors: "Daya Guo, Dejian Yang, Haowei Zhang, et al.",
      title: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
      source: "Nature 645(8081), 633–638",
      year: 2025,
      url: "https://www.nature.com/articles/s41586-025-09422-z",
      note: "Trains with GRPO against rule-based rewards that check answers and format, and says the authors avoided neural reward models for reasoning tasks because those are susceptible to reward hacking during large-scale RL. That is the lesson's case for an end-state check over a judge.",
    },
    {
      authors: "Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al.",
      title: "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
      source: "arXiv preprint arXiv:2411.15124",
      year: 2024,
      url: "https://arxiv.org/abs/2411.15124",
      note: "Names reinforcement learning with verifiable rewards (RLVR), where the policy is rewarded only when a program confirms that its answer is correct or that a constraint holds, which is what Outcome only with the Strict grader does here.",
    },
    {
      authors: "Carlos E. Jimenez, John Yang, Alexander Wettig, et al.",
      title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.06770",
      note: "Builds 2,294 tasks from real GitHub issues in 12 Python repositories. A model's patch counts only if tests from the original fix, which the harness applies after the patch, go from failing to passing: a check the model never edits, like the Strict grader.",
    },
    {
      authors: "Jiayi Pan, Xingyao Wang, Graham Neubig, et al.",
      title: "Training Software Engineering Agents and Verifiers with SWE-Gym",
      source: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267, 47717–47737",
      year: 2025,
      url: "https://proceedings.mlr.press/v267/pan25g.html",
      note: "An environment of 2,438 real Python tasks, each with an executable runtime, unit tests and a task in plain language, used to fine-tune agents on their own successful episodes and to train verifiers. It is the kind of environment the ticket desk shrinks.",
    },
    {
      authors: "Shunyu Yao, Noah Shinn, Pedram Razavi, et al.",
      title: "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2406.12045",
      note: "Agents talk to a simulated user and call domain tools under written policies, and success is judged by comparing the database at the end of the conversation with an annotated goal state, an end-state check like the Strict grader. Its pass^k metric asks whether an agent succeeds on all of k tries.",
    },
    {
      authors: "Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, et al.",
      title: "Specification gaming: the flip side of AI ingenuity",
      source: "Google DeepMind research blog",
      year: 2020,
      url: "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
      note: "Defines specification gaming as satisfying the literal specification of an objective without achieving the intended outcome, and links a list of real examples. Editing the test and the rubric-judge shortcut are both cases of it.",
    },
    {
      authors: "Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger",
      title: "Defining and Characterizing Reward Hacking",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2209.13085",
      note: "Defines reward hacking as optimizing a proxy reward in a way that leads to poor performance on the true reward, and shows that a proxy which can never be hacked is a very strong condition. That is why the lesson treats where the grader sits as a design decision.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2203.02155",
      note: "InstructGPT adds a per-token KL penalty from the supervised model to mitigate over-optimization of its reward model, the same leash that KL to reference β puts on the ticket-desk policy.",
    },
    {
      authors: "Leo Gao, John Schulman, and Jacob Hilton",
      title: "Scaling Laws for Reward Model Overoptimization",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/gao23h.html",
      note: "As a policy optimizes a learned proxy reward, the true score first rises and then falls. A KL penalty does not measurably improve the trade-off between true score and KL and acts like early stopping, which matches the lab: KL slows the hack without fixing the reward.",
    },
    {
      authors: "Bowen Baker, Joost Huizinga, Leo Gao, et al.",
      title: "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation",
      source: "arXiv preprint arXiv:2503.11926",
      year: 2025,
      url: "https://arxiv.org/abs/2503.11926",
      note: "Reports a frontier reasoning model trained in agentic coding environments that hacked its reward, for example by making a verify function always return true or by updating a unit test so it passes trivially: the edit_check loophole in real form. A monitor reading the chain of thought caught most of these hacks.",
    },
    {
      authors: "Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al.",
      title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track",
      year: 2023,
      url: "https://arxiv.org/abs/2306.05685",
      note: "Finds that language-model judges show position, verbosity and self-enhancement biases, while a strong judge such as GPT-4 agrees with human preferences over 80% of the time. The Rubric judge is a fixed checklist standing in for such a judge.",
    },
    {
      authors: "Zichen Liu, Changyu Chen, Wenjun Li, et al.",
      title: "Understanding R1-Zero-Like Training: A Critical Perspective",
      source: "Conference on Language Modeling (COLM 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2503.20783",
      note: "Shows that dividing by the group's reward standard deviation gives extra weight to questions whose rewards are almost all equal, because they are too easy or too hard, and proposes removing that division. That is the Mean only setting of Advantage scale, and the bias behind the step-cost collapse.",
    },
    {
      authors: "Karl Cobbe, Oleg Klimov, Chris Hesse, et al.",
      title: "Quantifying Generalization in Reinforcement Learning",
      source: "Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, 1282–1289",
      year: 2019,
      url: "https://proceedings.mlr.press/v97/cobbe19a.html",
      note: "Introduces CoinRun, with separate training and test levels, and finds that agents overfit to surprisingly large training sets and do better on unseen levels as the number of training levels grows. It backs the Unseen tickets card's point about training variety.",
    },
    {
      authors: "Andrew Y. Ng, Daishi Harada, and Stuart Russell",
      title: "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping",
      source: "Proceedings of the 16th International Conference on Machine Learning (ICML 1999), 278–287",
      year: 1999,
      url: "https://mlanthology.org/icml/1999/ng1999icml-policy/",
      note: "Proves that only potential-based shaping rewards are guaranteed to leave the best policy unchanged, and recounts a bicycle agent rewarded for progress that learned to ride in small circles. It backs the lesson's warning that partial-credit bonuses can change what the agent learns.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "In Ticket desk, an episode calls edit_check and then close_ticket without writing the record. Why does the Sandbox test row pass while the Strict grader row fails?",
      options: [
        "The Strict grader also reads the transcript and fails every episode shorter than three calls",
        "The sandbox test runs the edited file, while the Strict grader runs its own copy elsewhere",
        "The sandbox test passes at random 40% of the time, and this episode drew a lucky pass",
        "The Strict grader requires a run_check call before close_ticket, and this episode skipped it",
      ],
      answer: 1,
      explanation:
        "The Sandbox test runs test_record, which lives inside the sandbox next to the record, so after edit_check it passes whatever the record holds. The Strict grader keeps its own copy of the test outside the sandbox, where the agent cannot write, and fails any episode that changed the test. Neither scorer reads the transcript or requires run_check; the only difference is where the check lives.",
      objective: 0,
    },
    {
      prompt:
        "You lower Group size from 8 to 2 and Groups with no signal climbs. What does one of those groups do to the policy when KL to reference β is 0?",
      options: [
        "It raises the chance of every tool the two episodes used, since both got the same reward",
        "It moves the policy back toward the reference, since a group with no signal counts as a failure",
        "Nothing: both episodes have zero advantage, so all 48 weights stay where they were",
        "It doubles the size of the next update, since the unused signal carries over to the next group",
      ],
      answer: 2,
      explanation:
        "An advantage is an episode's reward minus its group's mean, so when both episodes score the same, both advantages are zero and the policy-gradient step is zero. With β above 0 the KL term would still pull toward the reference, but at β 0 nothing moves. Two episodes often both pass or both fail, so most small groups are wasted: at seed 1, 160 of 200 updates against 39 with a group of 8.",
      objective: 1,
    },
    {
      prompt:
        "Which reward design does training game by closing tickets without fixing the record, and how?",
      options: [
        "Outcome only, because a reward of 1 or 0 gives the agent no reason to write before closing",
        "Partial credit, because its bonuses outweigh the 1 the checker pays for a right record",
        "Rubric judge, because reading, looking up and closing in three calls meets all four items",
        "None, because every design pays more when the record ends up holding the right value",
      ],
      answer: 2,
      explanation:
        "The Rubric judge checks the transcript for a read, a look-up, a close with “resolved” and at most four calls, and never looks at the record, so read_ticket, look_up, close_ticket earns the full 1.00 without writing anything: Closed but not fixed reaches 100% and true success 0%. Outcome only with the Strict grader pays only for a right record. Partial credit's bonuses add at most 0.4 on top of the outcome, so they distort the route without replacing the goal.",
      objective: 2,
    },
    {
      prompt:
        "With Checker on Sandbox test, you raise KL to reference β from 0 to 0.2. What does update 200 show?",
      options: [
        "Hack rate falls to zero, because the reference policy almost never calls edit_check on its own",
        "Hack rate falls but stays far above zero, and true success stays below the Strict grader run",
        "Hack rate does not change at all, because the KL penalty only applies to the honest tools",
        "True success rises above the Strict grader run, because the leash keeps the good habits",
      ],
      answer: 1,
      explanation:
        "The KL term pulls each state's tool probabilities back toward the reference, whose edit_check logit is −2.5, so the policy drifts toward the loophole more slowly and settles partway: at seed 1 the hack rate is 42% instead of 100%, and over seeds 1 to 10 it averages 58%. But the sandbox test still pays for every edit, so the gradient keeps pointing at the loophole, and true success is 40% against 84% with the Strict grader. A leash limits how far training moves; only a grader the agent cannot change removes the reward for the hack.",
      objective: 3,
    },
    {
      prompt:
        "At seed 1 with Step cost 0.05, Divide by group std makes the policy close every ticket on its first turn, and Mean only does not. Why?",
      options: [
        "A 0.05 cost per turn is larger than the reward for a fixed record, so closing at once scores best",
        "With std scaling the Strict grader gets stricter, so fewer episodes pass and the agent gives up",
        "In an all-fail group the cost is the only difference, and the std division makes it full size",
        "Mean only ignores the step cost entirely, so the policy never learns that short episodes are cheaper",
      ],
      answer: 2,
      explanation:
        "Early on many groups contain no success, so their rewards differ only by the cost, and the shortest episode wins. Dividing by a spread of a few hundredths turns that small gap into advantages of about ±1, as large as a success, so quitting on turn 1 is learned before a fix is found. Mean only leaves the gap at its real size, which a single success outweighs. The cost itself never outweighs a pass: 8 turns cost 0.40 against a reward of 1.",
      objective: 3,
    },
    {
      prompt:
        "Training tickets is set to 1 kind, a ticket that gives its value. Why does the held-out Value missing ticket fail almost every time?",
      options: [
        "No training ticket left a value out, so the policy never learned to look one up before writing",
        "The held-out ticket's wording is new, and the policy cannot follow sentences it was not trained on",
        "One kind gives too few updates, and 200 updates on one ticket teach the policy nothing",
        "The look_up tool is switched off on held-out tickets, so the missing value cannot be found",
      ],
      answer: 0,
      explanation:
        "The policy reads seven yes/no features, never the wording, so new wording changes nothing. What matters is the “value missing” feature: with 1 kind it is never on during training, so the weights that would favour look_up in that situation are never trained and the policy writes a guess. At seed 1 the Value missing bar reads 4%, against 78% when the six training kinds include tickets with missing values.",
      objective: 4,
    },
    {
      prompt:
        "Trained on 6 kinds, the toy succeeds about as often on held-out tickets as on training tickets. Why is that weaker evidence of generalization than it looks?",
      options: [
        "The held-out tickets were secretly part of training, so the policy had already memorized them",
        "Held-out tickets differ only in wording, which the toy never reads but a language model does",
        "The Strict grader is easier on held-out tickets, so their success rate is inflated",
        "Held-out success is measured by sampling, so it carries noise the training curve does not",
      ],
      answer: 1,
      explanation:
        "Each ticket kind reaches the policy as one yes/no feature, value given or missing, so a held-out ticket of a trained class is identical to a training ticket from the policy's point of view. A language-model agent reads the ticket text, chooses arguments, and can meet tools and phrasings it never saw, so its held-out results can fall in ways this toy cannot show. Both readouts here are exact expectations, and the held-out tickets never reach training.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateRlEnvironmentsState,
};

export default definition;
