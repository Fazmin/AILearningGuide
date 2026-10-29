import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Inside `state`, `step` is the Episode step on
 * Return along the episode (24 means the end of the run), not the shell step the scene moves to.
 * Every scene starts from the defaults: γ 0.95, 30 sweeps, no slip, Bonus tile H off, cell r2c2
 * selected, ε 0.10, and the policy gradient at α 0.20 with the baseline on.
 */
const scenes: LabScene[] = [
  {
    id: "tiny-game",
    sections: ["A tiny game with a late score", "What it is"],
    title: "A tiny game with a late score",
    look: "S, T and G sit along the bottom row, and the path climbs over T to reach G, as the badge on the card says.",
    step: 0,
    focus: "Gridworld",
  },
  {
    id: "discounted-return",
    sections: ["Adding up the score: return and discount"],
    title: "Six rewards, discounted",
    look: "Each outer bar is a raw reward and the inner bar is what is left after the discount, and Full return G₀ adds the inner bars up.",
    step: 0,
    state: { step: 24 },
    focus: "Return along the episode",
  },
  {
    id: "bellman-backup",
    sections: ["The Bellman equation: one step of lookahead", "The Bellman equation"],
    title: "One cell, four moves",
    look: "The table scores the four moves from r2c2, right above T, and the row marked max becomes the cell’s value and its arrow.",
    state: { cell: 12 },
    focus: "Bellman backup",
  },
  {
    id: "credit-assignment",
    sections: ["Value iteration: passing the good news backward", "Credit assignment, made visible"],
    title: "Five sweeps in",
    look: "After five sweeps V(S) is still below zero; raise Value-iteration sweeps by one and watch it jump as the news from G reaches S.",
    step: 1,
    state: { sweeps: 5 },
    focus: "Gridworld",
  },
  {
    id: "greedy-trap",
    sections: ["Learning from tries: the bandit and exploration"],
    title: "Pure greed gets stuck",
    look: "With Exploration ε at 0, the bars show every pull going to one arm that is not the best; raise ε to 0.10 and compare.",
    step: 3,
    state: { epsilon: 0, banditSeed: 3 },
    focus: "Explore versus exploit",
  },
  {
    id: "policy-gradient",
    sections: ["Policy gradient: nudging the probabilities", "A policy gradient on the bandit"],
    title: "Nudging the probabilities",
    look: "Every arm starts at 25%, the bars show where one seeded run ended, and the solid line is the best arm’s average probability climbing pull by pull.",
    step: 5,
    state: { pgRate: 0.2, pgBaseline: "on", pgSeed: 3 },
    focus: "Policy gradient on the bandit",
  },
  {
    id: "read-episode",
    sections: ["1. Read one episode"],
    title: "Read one episode",
    look: "Drag Episode step from 0 to the end and watch Reward this step and Return so far change at every move.",
    step: 0,
    state: { step: 0 },
    focus: "Return along the episode",
  },
  {
    id: "sweep-backup",
    sections: ["2. Sweep the backup", "How to play with it"],
    title: "Start from zero sweeps",
    look: "Raise Value-iteration sweeps one at a time and watch the shading spread back from G while V(S) and Residual change.",
    step: 1,
    state: { sweeps: 0 },
    focus: "Gridworld",
  },
  {
    id: "low-discount",
    sections: ["3. Change the discount, then let moves slip"],
    title: "A discount of 0.50",
    look: "At Discount γ 0.50, V(S) is below zero though the path still ends at G; then set γ back to 0.95 and Slip chance to 20%.",
    step: 2,
    state: { gamma: 0.5 },
    focus: "Gridworld",
  },
  {
    id: "explore-exploit",
    sections: ["4. Explore or exploit"],
    title: "Explore or exploit",
    look: "Compare Best-arm share, 300 runs with Same, ε = 0, then move Exploration ε to 0 and to 0.50 and compare again.",
    step: 3,
    state: { epsilon: 0.1 },
    focus: "Explore versus exploit",
  },
  {
    id: "bonus-tile",
    sections: ["5. Turn on the bonus"],
    title: "The bonus tile",
    look: "The badge says the agent circles H instead of reaching G; lower Discount γ to 0.85 and watch the path walk to G again.",
    step: 4,
    state: { hack: "on" },
    focus: "Gridworld",
  },
  {
    id: "baseline-off",
    sections: ["6. Follow the policy gradient"],
    title: "Baseline off",
    look: "With Baseline off, read Best-arm probability, 300 runs and Runs below 50% on best arm, then switch Baseline on and compare.",
    step: 5,
    state: { pgRate: 0.2, pgBaseline: "off" },
    focus: "Policy gradient on the bandit",
  },
];

export default scenes;
