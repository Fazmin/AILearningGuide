import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Gridworld": {
    title: "A 5×4 MDP solved by value iteration",
    summary:
      "Twenty cells, four actions, rewards paid on entering a cell. The shading and numbers are V after the chosen number of synchronous Bellman backups from V = 0, and the arrows are the greedy action for that V. The agent's path is a rollout of that greedy policy from S.",
    whatYouSee: [
      "S (start, bottom left), G (+1, terminal), T (−1, terminal), and H when the bonus tile is on. Every other move costs −0.04.",
      "Each cell's top-right number is V(s). Tint strength is |V| relative to the largest |V| on the grid; greener is higher, redder is lower.",
      "Arrows are argmax over actions of Q(s, a). Two or more arrows mean a tie within 10⁻⁹; the rollout breaks ties up → right → down → left.",
      "The dark line is the episode up to Episode step; the faint line is the rest of it. \"slip\" marks a move that went sideways.",
      "The badge reports the episode outcome; V(S) and Residual repeat the start value and the last sweep's largest change.",
    ],
    howItWorks: [
      "V_{k+1}(s) = max_a Σ_{s′} P(s′|s,a)·[r(s,a,s′) + γ·V_k(s′)], applied to all 18 non-terminal cells at once. Terminal cells stay at 0.",
      "P(s′|s,a) sends the agent the intended way with probability 1 − slip and to each perpendicular neighbour with slip/2. A move into the wall leaves it in place and costs −0.04.",
      "The bonus tile pays +0.3 each time it is entered and does not end the episode, so stepping in and out of it is an endless stream of rewards.",
      "The rollout samples slips from a seeded generator and stops at G, at T, or after 24 steps.",
    ],
    controls: [
      "Value-iteration sweeps (0–60), Discount γ (0.50–0.99), Slip chance (0–30%), and Bonus tile H (off or on). Each change replays the episode to its end.",
      "Click a cell, or focus the grid and use the arrow keys, to inspect it on Bellman backup.",
      "Comparison worth running: sweeps 0 through 6 at γ 0.95. V(S) stays negative until sweep 6, when the +1 has traveled the six moves back to S.",
    ],
    notice: [
      "Value arrives at S exactly one sweep per move of distance from G. That is credit assignment made visible.",
      "At Slip chance 20% the arrow on the cell left of T turns away from it, and V(S) falls because some episodes now end in T.",
      "With the bonus on, the policy circles H at γ 0.89 and above and walks to G at 0.88 and below, at the default 30 sweeps with no slip.",
    ],
    limits: [
      "In this lab: the transition table is known and tiny, so value iteration can sweep every state. The rewards, the 24-step cap, and the +0.3 bonus are authored.",
      "In general: language models have far too many states to enumerate and no known transition table, so they are trained with sampled policy-gradient methods rather than value iteration. The vocabulary carries over; the algorithm does not.",
    ],
  },

  "Bellman backup": {
    title: "The one-step lookahead every sweep repeats",
    summary:
      "For the inspected cell, each row evaluates Q(s, a) = Σ P(s′|s,a)·(r + γ·V(s′)) with the current V. The largest Q is exactly what the next sweep writes into that cell. The chart shows how fast the largest change across the grid shrinks.",
    whatYouSee: [
      "A four-row table: the action, the expectation written out with the live probabilities, rewards, γ, and neighbour values, and the resulting Q. The best row is marked max.",
      "A formula card repeating the winning action's terms, its value, and the number the next sweep writes into this cell.",
      "A line chart of the measured residual max_s |V_k(s) − V_{k−1}(s)| for k = 1 to 60, a dotted γ^(k−1) × first-residual reference, and a guide at the current sweep.",
    ],
    howItWorks: [
      "Rewards depend on the cell entered: +1 for G, −1 for T, +0.3 for H when on, −0.04 otherwise, and −0.04 for bumping a wall.",
      "Value iteration is a γ-contraction: each sweep can shrink the worst-case error by at least a factor of γ. The dotted line is that reference, drawn from the first residual.",
      "With no slip the grid is deterministic and the residual drops to exactly 0 once value has reached every cell.",
    ],
    controls: [
      "Click any cell on Gridworld. Value-iteration sweeps, Discount γ, Slip chance, and Bonus tile H all change the numbers here.",
      "Comparison worth running: inspect cell r2c2, directly above T, at Slip chance 0% and then 30%. Right wins at 0%; at 30% a 15% chance of slipping down into T makes up the better move.",
    ],
    notice: [
      "The backup reads only neighbours' current values. Nothing looks further ahead, which is why a long corridor needs many sweeps.",
      "At γ 0.99 with the bonus on, the residual is still well above zero after 60 sweeps. A discount near 1 converges slowly.",
    ],
    limits: [
      "In this lab: the expectation is exact because P(s′|s,a) is a three-entry table. A 60-sweep cap is short enough that γ ≥ 0.97 with the bonus on has not converged.",
      "In general: deep RL replaces the table with a neural network and the exact expectation with sampled transitions, which brings bias, variance, and instability that exact value iteration does not have.",
    ],
  },

  "Return along the episode": {
    title: "A single sampled return, term by term",
    summary:
      "The bars are this episode's rewards r₁, r₂, … in order. The inner bars are the same rewards multiplied by γᵗ, which is what the discounted return adds up. With no slip and a converged V, G₀ equals V(S) exactly.",
    whatYouSee: [
      "Outer bars: raw rewards, +1 up to −1. Inner bars: γᵗ·r, green above zero and red below. Bars left of the cursor are counted in Return so far.",
      "Episode step (0 to the episode length) and Episode seed (1–20, which draws a different sequence of slips).",
      "Reward this step, Return so far, Full return G₀, and V(S) at the current sweep.",
    ],
    howItWorks: [
      "G₀ = Σₜ γᵗ·r₍ₜ₊₁₎ over the rewards actually received. The +1 at the end of a six-move episode is worth γ⁵ of its face value.",
      "V(S) is an expectation over all slip outcomes, while G₀ is one draw, so they differ whenever Slip chance is above zero.",
      "A looping episode is cut at 24 steps, so its G₀ undercounts the infinite-horizon value.",
    ],
    controls: [
      "Episode step scrubs the return; Episode seed resamples the slips. The Gridworld controls regenerate the episode.",
      "Comparison worth running: Slip chance 20%, then seeds 1 to 5. G₀ scatters around V(S) = 0.314.",
    ],
    notice: [
      "Most of an early step's contribution is the −0.04 step cost. The payoff that justifies the whole path arrives last and discounted.",
      "The note under the chart tells you why G₀ and V(S) do or do not agree for the current settings.",
    ],
    limits: [
      "In this lab: the episode follows the greedy policy, so the only randomness is slipping. There is no exploration inside the gridworld.",
      "In general: a real agent sees only sampled returns like this one and must average many of them, which is why return variance is the central practical difficulty of policy-gradient training.",
    ],
  },

  "Explore versus exploit": {
    title: "ε-greedy on a four-armed Bernoulli bandit",
    summary:
      "Four arms pay 1 with probabilities 0.30, 0.55, 0.45, and 0.70. The agent does not know them. It keeps a sample-average estimate per arm, pulls the best estimate, and with probability ε pulls a random arm instead. Without exploration it keeps the first arm that happens to pay.",
    whatYouSee: [
      "Bars: how many of 400 pulls each arm got in one seeded run, with its final estimate and its true mean.",
      "A line chart of mean reward so far, averaged over 300 seeded runs, for your ε and for ε = 0, with the best arm's 0.70 as a dotted ceiling.",
      "Best-arm share for this run, averaged over 300 runs, and the same average for ε = 0.",
    ],
    howItWorks: [
      "Estimates start at 0 and update as Q ← Q + (r − Q)/n. Greedy ties are broken uniformly at random.",
      "The 300 runs use fixed seeds, so the averages are deterministic. At ε = 0.10 the best arm gets about 72% of pulls; at ε = 0 about 33%; at ε = 0.50 about 56%.",
    ],
    controls: [
      "Exploration ε from 0 to 0.50, and Bandit seed for the single run in the bars.",
      "Comparison worth running: ε = 0 across several Bandit seeds. The run locks onto whichever arm paid first.",
    ],
    notice: [
      "Exploration has a price and a payoff: too little and the agent never learns the 0.70 arm; too much and it keeps pulling arms it already knows are worse.",
      "Value iteration on the gridworld never needed this, because it read the transition table directly instead of learning from its own samples.",
    ],
    limits: [
      "In this lab: one state, four arms, stationary Bernoulli payoffs, and ε-greedy only. Upper-confidence and Thompson-sampling strategies usually do better.",
      "In general: RLHF on a single prompt and response is close to a contextual bandit, so the same trade appears there; a model that is never sampled into a behavior cannot be rewarded for it. The next card replaces the value estimates with a learned policy.",
    ],
  },

  "Policy gradient on the bandit": {
    title: "REINFORCE on four arms",
    summary:
      "The same four Bernoulli arms, but the agent now keeps a softmax policy instead of value estimates. Each pull samples an arm from the policy and a reward from the arm, then nudges the policy toward arms that paid more than a baseline expects. The charts average 300 seeded runs; the bars show one run.",
    whatYouSee: [
      "`Learning rate α` (0.02 to 1.00), `Baseline` (off, so b = 0, or on, so b is the running average reward), and `Policy seed` (1 to 20).",
      "Bars: the policy's probability on each arm after 400 pulls in one seeded run, with the arm's true mean and how often it was pulled. Every arm starts at 25%.",
      "Two charts over 300 seeded runs, baseline on solid and off dashed: the probability on the best arm before each pull, and the mean reward so far with the best arm's 0.70 as a dotted ceiling. Their legends print the final values.",
      "Three metrics: the best-arm probability in this run, its average over 300 runs for your Baseline setting, and how many of the 300 runs end with less than half their probability on the best arm.",
    ],
    howItWorks: [
      "The policy is `π = softmax(θ)` over four preferences θ, all 0 at the start. Each pull samples an arm from π and a reward `r` of 0 or 1 from that arm's Bernoulli.",
      "The update is `θ_i ← θ_i + α·(r − b)·(1[i = arm] − π_i)`. The bracket is exactly `∂ ln π(arm) / ∂θ_i`, so this is REINFORCE (Williams, 1992): a sampled estimate of the gradient of expected reward. A unit test checks the bracket against finite differences, and checks that the expected update is that gradient for any constant baseline.",
      "The baseline `b` is the mean reward of the earlier pulls, or 0 when off. It does not depend on the current arm, so it leaves the expected update unchanged and changes only its noise.",
      "Both curves use the same 300 seeded runs that `Explore versus exploit` uses, so the only difference between them is b.",
    ],
    controls: [
      "`Learning rate α`, `Baseline`, and `Policy seed` on this card.",
      "Comparison worth running: α 0.20 with Baseline on, then off. Then Baseline on at α 0.02, 0.20, and 1.00.",
    ],
    notice: [
      "At α 0.20 the baseline-on policy ends with 92.1% on the best arm on average and 2 of 300 runs below 50%. With the baseline off, 84.9% and 29 of 300.",
      "With the baseline off a reward of 0 changes nothing and a reward of 1 raises the arm just pulled, so an arm that pays early can run away. At the default seed 3 the run without the baseline ends with 92% on arm 2, whose mean is 0.55.",
      "α 0.02 barely moves in 400 pulls: 37% on the best arm either way. α 1.00 commits fastest, with 65.1% after 50 pulls against 40.1% at 0.20, but 33 of 300 runs end below 50%.",
    ],
    limits: [
      "In this lab: one state, four stationary arms, a fixed 400 pulls, and a running-average baseline. There is no value function, no clipping, and no KL penalty, so this is the simplest policy-gradient estimator, not PPO or GRPO.",
      "In general: a language model's policy gives a probability to every possible response, and its rewards come from a learned reward model or a checker. Practical methods subtract a learned value estimate or a group average, and clip the step. The noise this card shows is the reason they do.",
    ],
  },
};

export default cardInfo;
