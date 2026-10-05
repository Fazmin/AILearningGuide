import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { CELLS, MAX_EPISODE_STEPS, MAX_SWEEPS, PG_MAX_RATE, PG_MIN_RATE } from "./mdp";

const initialState: ModuleState = {
  gamma: 0.95,
  sweeps: 30,
  slip: 0,
  hack: "off",
  step: MAX_EPISODE_STEPS,
  seed: 3,
  cell: 12,
  epsilon: 0.1,
  banditSeed: 3,
  pgRate: 0.2,
  pgBaseline: "on",
  pgSeed: 3,
};

const num = (value: unknown, fallback: number, low: number, high: number, integer = false) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const bounded = Math.min(high, Math.max(low, value));
  return integer ? Math.round(bounded) : bounded;
};

/**
 * Version 1 stored { policy, hack, step, seed } for a one-step heuristic; only hack, step, and seed still mean the same
 * thing. Version 2 lacked the policy-gradient card, so its payloads take pgRate, pgBaseline, and pgSeed from the defaults.
 */
export function migrateRlState(parsed: ModuleState): ModuleState {
  const hack = parsed.hack === "on" || parsed.hack === true ? "on" : "off";
  return {
    gamma: num(parsed.gamma, initialState.gamma as number, 0.5, 0.99),
    sweeps: num(parsed.sweeps, initialState.sweeps as number, 0, MAX_SWEEPS, true),
    slip: num(parsed.slip, initialState.slip as number, 0, 0.3),
    hack,
    step: num(parsed.step, initialState.step as number, 0, MAX_EPISODE_STEPS, true),
    seed: num(parsed.seed, initialState.seed as number, 1, 20, true),
    cell: num(parsed.cell, initialState.cell as number, 0, CELLS - 1, true),
    epsilon: num(parsed.epsilon, initialState.epsilon as number, 0, 0.5),
    banditSeed: num(parsed.banditSeed, initialState.banditSeed as number, 1, 20, true),
    pgRate: num(parsed.pgRate, initialState.pgRate as number, PG_MIN_RATE, PG_MAX_RATE),
    pgBaseline: parsed.pgBaseline === "off" ? "off" : "on",
    pgSeed: num(parsed.pgSeed, initialState.pgSeed as number, 1, 20, true),
  };
}

const definition: ModuleDefinition = {
  id: "module-45-rl-foundations",
  slug: "rl-foundations",
  title: "Reinforcement learning foundations",
  group: "frontiers",
  order: 28,
  icon: "Gamepad2",
  accent: "#2d7a5a",
  prerequisites: ["module-33-how-models-are-evaluated"],
  estimatedMinutes: 16,
  steps: [
    "Read one episode",
    "Sweep the backup",
    "Change the discount",
    "Explore or exploit",
    "Turn on the bonus",
    "Follow the policy gradient",
  ],
  stepInstructions: [
    "On Return along the episode, move Episode step from 0 to the end and compare Reward this step with Return so far and Full return G₀.",
    "Set Value-iteration sweeps to 0, then raise it one sweep at a time and watch value spread backward from G until V(S) stops changing.",
    "Move Discount γ from 0.95 down to 0.50 and read V(S), then set Slip chance to 20% and watch the arrows beside T turn away from it.",
    "On Explore versus exploit, set Exploration ε to 0, then 0.10, then 0.50, and compare Best-arm share, 300 runs with Same, ε = 0.",
    "Set Bonus tile H to on at Discount γ 0.95 and watch the episode circle H, then lower Discount γ to 0.85 and watch it walk to G again.",
    "On Policy gradient on the bandit, leave Baseline on at Learning rate α 0.20 and read Best-arm probability, 300 runs, 92.1%. Switch Baseline off and compare it with 84.9%, and Runs below 50% on best arm, 2 against 29 of 300.",
  ],
  stateVersion: 3,
  tagline:
    "Solve a tiny gridworld with real Bellman backups, watch value flow backward from a delayed reward, and see an exploit become the optimal policy, so RLHF later reads as one use of the same loop.",
  objectives: [
    "Identify state, action, reward, policy, and discounted return in a small episode",
    "Use the Bellman backup to explain credit assignment as value flowing backward from a delayed reward",
    "Explain why learning from samples needs exploration, and why a misspecified reward is maximized anyway",
    "Explain how a policy-gradient update turns sampled rewards into a better policy, and what a baseline and the learning rate change",
  ],
  glossary: [
    {
      term: "State",
      definition:
        "The information the policy is allowed to see before it acts. Here that is the cell index. If a needed fact is outside the state, no amount of training recovers it.",
    },
    {
      term: "Action",
      definition:
        "The choice the policy emits, here one of up, right, down, or left. In language-model training each generated token can be treated as an action.",
    },
    {
      term: "Reward",
      definition:
        "A scalar the designer chose to pay for a transition. It is not an objective fact about goodness, and a misspecified reward is maximized as faithfully as a good one.",
    },
    {
      term: "Policy",
      definition:
        "A rule, often a probability distribution, that maps each state to an action. Here it is the greedy action with respect to the current value table.",
    },
    {
      term: "Episode",
      definition:
        "One run from the start state until a terminal state or a step cap. This lab caps episodes at 24 steps so a loop cannot run forever.",
    },
    {
      term: "Return",
      definition:
        "The discounted sum of rewards from a time step onward, G = r1 + γ r2 + γ² r3 and so on. One episode's return is a single sample, not an average.",
    },
    {
      term: "Discount factor",
      definition:
        "The number γ between 0 and 1 that shrinks a reward k steps away by γ to the k. It sets how far ahead the agent cares, and keeps an endless stream of rewards finite.",
    },
    {
      term: "Value function",
      definition:
        "V(s), the expected return from state s when following a policy. With slip on it is an average over outcomes, so a single episode's return can land above or below it.",
    },
    {
      term: "Bellman equation",
      definition:
        "The recursion V(s) = max over actions of the expected immediate reward plus γ times V of the next state. It turns a long-horizon question into a one-step lookup.",
    },
    {
      term: "Value iteration",
      definition:
        "Applying the Bellman backup to every state repeatedly, starting from zero. It needs the full transition table, and each sweep shrinks the worst error by at least a factor of γ.",
    },
    {
      term: "Credit assignment",
      definition:
        "Deciding which earlier actions earned a later reward. In value iteration the answer travels backward one cell per sweep; with sparse, delayed rewards that is the hard part.",
    },
    {
      term: "Exploration",
      definition:
        "Trying actions whose value is uncertain, at a short-term cost. An agent that learns only from its own samples cannot improve its estimate of an action it never takes.",
    },
    {
      term: "Policy gradient and baseline",
      definition:
        "Policy gradient nudges a policy's parameters along the gradient of ln π(action) times (reward − baseline), so an action that beat the baseline becomes more likely and one that fell short less likely. REINFORCE is the basic form. It needs only sampled rewards, not a transition table. A baseline that ignores the current action leaves the expected update unchanged and reduces its noise.",
    },
    {
      term: "Reward hacking",
      definition:
        "A policy scoring highly on the written reward by a route the designer did not intend. It is correct optimization of the wrong scalar, not a bug in the optimizer.",
    },
  ],
  references: [
    {
      authors: "Richard S. Sutton and Andrew G. Barto",
      title: "Reinforcement Learning: An Introduction, 2nd edition",
      source: "MIT Press, free to read online",
      year: 2018,
      url: "http://incompleteideas.net/book/the-book-2nd.html",
      note: "The standard RL textbook. Chapters 3 and 4 define MDPs, return, discounting, the Bellman equation and value iteration, and Chapter 2 covers epsilon-greedy, upper-confidence bounds, and the gradient bandit, the same softmax update with an average-reward baseline that the Policy gradient on the bandit card runs.",
    },
    {
      authors: "Stuart Russell and Peter Norvig",
      title: "Artificial Intelligence: A Modern Approach, 4th edition",
      source: "Pearson",
      year: 2020,
      url: "https://aima.cs.berkeley.edu/contents.html",
      note: "Chapter 17, Making Complex Decisions, works through sequential decision problems on the small 4 by 3 grid world that this lab's per-move living cost comes from. It covers value iteration and bandit problems in the same chapter.",
    },
    {
      authors: "Ronald J. Williams",
      title: "Simple statistical gradient-following algorithms for connectionist reinforcement learning",
      source: "Machine Learning 8(3–4), 229–256",
      year: 1992,
      url: "https://link.springer.com/article/10.1007/BF00992696",
      note: "The REINFORCE paper the lesson names. It shows that nudging weights by the reward minus a baseline, times the gradient of the log-probability of the action taken, follows the gradient of expected reward, which is the update the Policy gradient on the bandit card applies.",
    },
    {
      authors: "OpenAI",
      title: "Part 3: Intro to Policy Optimization",
      source: "Spinning Up in Deep RL documentation",
      year: 2018,
      url: "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html",
      note: "A step-by-step derivation of the policy gradient from the log-probability trick. Its section on baselines shows why subtracting a baseline leaves the expected update unchanged while cutting its noise, the effect the Baseline switch shows.",
    },
    {
      authors: "Christopher J. C. H. Watkins and Peter Dayan",
      title: "Q-learning",
      source: "Machine Learning 8(3–4), 279–292",
      year: 1992,
      url: "https://link.springer.com/article/10.1007/BF00992698",
      note: "The Q-learning paper. It learns the value of each action in each state from sampled transitions, with no transition table, and proves this converges, which is the route the lesson gives for learning without the table.",
    },
    {
      authors: "John Schulman, Philipp Moritz, Sergey Levine, et al.",
      title: "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
      source: "International Conference on Learning Representations (ICLR 2016)",
      year: 2016,
      url: "https://arxiv.org/abs/1506.02438",
      note: "Uses a learned value function as the baseline to cut the noise in policy-gradient estimates, at the cost of some bias. It is the smarter baseline the lesson says real methods use in place of the card's running average.",
    },
    {
      authors: "John Schulman, Filip Wolski, Prafulla Dhariwal, et al.",
      title: "Proximal Policy Optimization Algorithms",
      source: "arXiv preprint arXiv:1707.06347",
      year: 2017,
      url: "https://arxiv.org/abs/1707.06347",
      note: "The PPO paper the lesson names. Its clipped objective stops one batch of samples from moving the policy too far, the clipping the lesson says the bandit card leaves out.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2203.02155",
      note: "The InstructGPT paper the lesson cites. It describes its RL stage as a bandit environment that shows a random prompt, scores one response with a reward model, and ends the episode, with a per-token KL penalty to the starting model.",
    },
    {
      authors: "Zhihong Shao, Peiyi Wang, Qihao Zhu, et al.",
      title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
      source: "arXiv preprint arXiv:2402.03300",
      year: 2024,
      url: "https://arxiv.org/abs/2402.03300",
      note: "Introduces GRPO, a variant of PPO that drops the learned value model and uses the average reward of several answers to the same question as the baseline. It is the group-average baseline the lesson mentions.",
    },
    {
      authors: "Peter Henderson, Riashat Islam, Philip Bachman, et al.",
      title: "Deep Reinforcement Learning that Matters",
      source: "Proceedings of the AAAI Conference on Artificial Intelligence 32(1) (AAAI 2018)",
      year: 2018,
      url: "https://arxiv.org/abs/1709.06560",
      note: "Shows that deep RL results can swing widely between random seeds, because the methods themselves are noisy. It backs the lesson's point that noisy sampled returns make variance and instability dominate real training, and why a method should be judged over many seeded runs.",
    },
    {
      authors: "Jack Clark and Dario Amodei",
      title: "Faulty reward functions in the wild",
      source: "OpenAI research blog",
      year: 2016,
      url: "https://openai.com/index/faulty-reward-functions/",
      note: "The boat-racing write-up the lesson retells. An agent in CoastRunners circles a lagoon hitting targets as they reappear instead of finishing the race, because the score pays for targets, the same story as circling Bonus tile H.",
    },
    {
      authors: "Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, et al.",
      title: "Specification gaming: the flip side of AI ingenuity",
      source: "Google DeepMind research blog",
      year: 2020,
      url: "https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
      note: "Defines specification gaming as meeting the literal objective without the intended outcome, and links a list of real examples. It backs the lesson's claim that a misspecified reward is maximized as faithfully as a good one.",
    },
    {
      authors: "Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger",
      title: "Defining and Characterizing Reward Hacking",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2209.13085",
      note: "Gives a formal definition of reward hacking: raising a proxy reward while the true reward falls. It shows that a proxy which can never be hacked this way is very hard to write, which is why the lesson treats reward specification as its own problem.",
    },
    {
      authors: "Leo Gao, John Schulman, and Jacob Hilton",
      title: "Scaling Laws for Reward Model Overoptimization",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/gao23h.html",
      note: "Optimizes language-model policies hard against a learned reward model and finds the proxy score keeps rising while the true score first improves and then falls. It backs the lesson's warning that more optimization makes a bad reward worse, not better.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "With no slip, you lower Discount γ from 0.95 to 0.50 on Return along the episode. What happens to Full return G₀ and to the path the policy takes?",
      options: [
        "G₀ falls and the path bends toward T, because a short horizon makes the −1 look closer",
        "G₀ rises because a smaller γ also shrinks the −0.04 cost that every move pays along the way",
        "G₀ stays the same because the six rewards are unchanged and only V(S) depends on the discount",
        "G₀ falls because the final +1 is shrunk by γ to the fifth power, and the path still ends at G",
      ],
      answer: 3,
      explanation:
        "A return weights the reward k steps ahead by γ to the k. The six rewards are the same, but at γ 0.50 the +1 at the end keeps only about 3% of its face value, less than the step costs paid on the way, so G₀ drops below zero. The greedy path does not change, because every alternative is worth even less.",
      objective: 0,
    },
    {
      prompt:
        "At the defaults, V(S) is not positive until Value-iteration sweep 6. What is the best explanation?",
      options: [
        "Each sweep carries the +1 back one move, and S is six moves from G",
        "Value is learned from experience, and the agent needs six episodes to reach G once",
        "The discount shrinks the +1 below the step costs until the sixth sweep",
        "The backup uses a small learning rate, so each sweep adds a fraction of the reward",
      ],
      answer: 0,
      explanation:
        "The Bellman backup reads only neighbours' current values, so the +1 at G reaches a cell one sweep per move of distance. S is six moves away, so V(S) first sees the reward at sweep 6. Value iteration reads the known transition table rather than learning from episodes, and γ is the same at every sweep.",
      objective: 1,
    },
    {
      prompt:
        "On Explore versus exploit you set Exploration ε to 0 and step through several Bandit seeds. What do you see?",
      options: [
        "Every run finds the 0.70 arm in the end, because greedy always follows the highest estimate",
        "Runs pull all four arms about equally, because nothing yet favors any arm",
        "Average reward is highest at ε = 0, because no pulls are wasted on random arms",
        "Each run locks onto whichever arm paid first, so the best arm is often never found",
      ],
      answer: 3,
      explanation:
        "Estimates start at 0, so the first arm that pays beats every untried arm and keeps winning. Averaged over 300 runs the best arm then gets only about a third of the pulls, against about 72% at ε 0.10, and average reward is lower too. A little exploration costs a few pulls and buys information the agent cannot get any other way.",
      objective: 2,
    },
    {
      prompt:
        "With Bonus tile H on at Discount γ 0.95, the greedy policy circles H and never reaches G. Which diagnosis is right?",
      options: [
        "Value iteration has not converged yet, so a few dozen more sweeps will eventually send the agent to G",
        "The reward pays more for circling than for finishing, so the policy is maximizing it correctly",
        "The agent has not explored enough to find out that G exists, so it settles for the nearby tile H",
        "The discount is too small, so the +1 at G is too far away to count for anything at all",
      ],
      answer: 1,
      explanation:
        "At γ = 0.95, stepping into H every other move is worth about (0.3γ − 0.04) / (1 − γ²) ≈ 2.5 from H, more than the single +1 at G. Value iteration reads the full transition table, so it is not missing information and more sweeps only confirm the loop. The scalar was misspecified; lowering γ to 0.88 or below makes finishing worth more again.",
      objective: 2,
    },
    {
      prompt:
        "Value iteration solves this gridworld exactly. Why can't the same method train a language model?",
      options: [
        "It needs a table of every state and transition, and text has far too many states to list",
        "The Bellman equation only holds on deterministic grids, and generated text is random from token to token",
        "A language model gets no reward signal at all, so there is no return to back up through the tokens",
        "Value iteration needs γ = 1, which would make the return of a long text infinite",
      ],
      answer: 0,
      explanation:
        "Value iteration sweeps every state using P(s′|s,a) as a lookup. A language model's state is the prompt plus the tokens so far, which cannot be enumerated, and its transitions are not given as a table, so it learns from sampled returns instead. The Bellman equation itself holds with randomness, as the slip chance shows, and rewards come from a reward model or a checker.",
      objective: 1,
    },
    {
      prompt:
        "On Policy gradient on the bandit, Learning rate α is 0.20 and Baseline is on. You switch Baseline off. What happens to the 300-run results?",
      options: [
        "The policy ends less sure of the best arm and more runs lock onto a worse one: any reward lifts the arm pulled",
        "The policy stops improving, because without a baseline the update cannot tell a good reward from a bad one at all",
        "The policy improves faster, because rewards that are not shifted down make every step larger than before",
        "Nothing changes on average, because subtracting a baseline cannot alter what the update does in expectation here",
      ],
      answer: 0,
      explanation:
        "A baseline that ignores the current action leaves the expected update the same but removes much of its noise. Without it, every reward of 1 raises the probability of whichever arm was pulled, so an arm that pays early can run away before the better arm is sampled. At α 0.20 the average probability on the best arm is 92.1% with the baseline and 84.9% without, and 2 of 300 runs end below 50% on it against 29.",
      objective: 3,
    },
    {
      prompt:
        "With Baseline on, you raise Learning rate α from 0.20 to 1.00. What do the 300 runs show?",
      options: [
        "The policy commits sooner, but more runs commit to a worse arm, so the average ends less sure of the best arm",
        "The policy gets steadily better, because a bigger step always finds the best arm in fewer pulls without overshooting",
        "The policy stops learning, because a step that large cannot be applied to a softmax over only four arms",
        "Nothing changes, because the baseline cancels the learning rate out of every update the policy takes",
      ],
      answer: 0,
      explanation:
        "A larger α moves the policy further on every sampled reward. After 50 pulls the average probability on the best arm is 65.1% at α 1.00 against 40.1% at 0.20, so it commits sooner. But a few early lucky pulls can then push it onto a worse arm for good: 33 of 300 runs end below 50% on the best arm, against 2, and the average final probability falls from 92.1% to 88.3%.",
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
      return migrateRlState(parsed);
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
