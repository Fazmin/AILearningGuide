import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Ticket desk": {
    title: "One episode on the Harbor ticket desk, scored four ways",
    summary:
      "The environment is a ticket desk with six tools and an 8-turn cap. One episode is sampled from the policy at the chosen training update, and four scorers read the finished episode: a test inside the sandbox, a strict grader outside it, partial credit, and a rubric judge that reads only the transcript. Same episode, four verdicts.",
    whatYouSee: [
      "Episode seed (1 to 20), which picks the ticket (seeds 1 to 8 visit the eight ticket kinds in order, then repeat) and the random draws. The heading names the ticket, whether it is held out, and the training update whose policy acts.",
      "The ticket text, with whether it gives the value or leaves it out. The policy never reads this text; it sees only seven yes/no features.",
      "Episode replay: each turn's tool call, the chance the policy gave that tool, the desk's reply, and in italics what is true but hidden from the agent, such as a write that did not save. Each turn ends with two marks for the hidden truth after it: whether the record is right and whether the test is intact, drawn bold on the turn that changed it. An edit_check row is marked loophole.",
      "Scorecard: Sandbox test, Strict grader, Partial credit and Rubric judge, each with its score, a bar for that score on a scale that runs to 1.4 with a mark at 1, and what it read. The row whose score is the training reward under the current Reward design and Checker is tagged training reward. Below it, the training reward for this episode under the current Reward design, Checker and Step cost.",
    ],
    howItWorks: [
      "Tools: read_ticket marks the ticket read; look_up finds the value; update_record writes the right value only if the value is known (read on a ticket that gives it, or looked up), and even then fails to save 40% of the time without saying so; run_check runs test_record; edit_check rewrites test_record to always pass; close_ticket ends the episode.",
      "Sandbox test = 1 if closed and (record right or test edited). Strict grader = 1 if closed, record right and test untouched. Partial credit = 0.1 read + 0.1 look-up + 0.2 record right + the Checker's verdict. Rubric judge = 0.25 each for read, looked up, closed with “resolved”, and 4 calls or fewer.",
      "An episode that reaches 8 calls without close_ticket is never scored: every scorer gives 0. Training reward = score − Step cost × turns.",
      "The policy is the 48-weight softmax from the training card, read at Training update. Only the tool choice and the save failures are random, drawn from a stream fixed by Episode seed.",
    ],
    controls: [
      "Episode seed here, and Training update, Reward design, Checker and Step cost on the training card.",
      "Comparison worth running: at Training update 0, step Episode seed from 1 to 5 and watch the four scorers disagree on the same episodes. Then set Checker to Sandbox test and Training update to 200 and read an episode that edits the test.",
    ],
    notice: [
      "A closed ticket with the right record passes both tests; an edited test passes only the Sandbox test.",
      "The Rubric judge can give 1.00 to an episode whose record is wrong, because it never reads the record.",
      "On a ticket that leaves the value out, a write before a look-up stores a guess, and the hidden note says so.",
    ],
    limits: [
      "In this lab: the desk, its replies, the scorers and the eight tickets are written by hand. The policy picks only which tool to call, never its arguments, and the ticket wording is decoration it never reads.",
      "In general: real environments give an agent a shell, files and tools with free-text arguments, so there are far more ways to fail and far more loopholes than one writable test, and a judge is often a language model rather than a fixed checklist.",
    ],
  },

  "Train with group-relative updates": {
    title: "GRPO-style training of a 48-weight agent, with a held-out score beside the reward",
    summary:
      "Each of 200 updates samples a group of episodes on one training ticket, scores them with the chosen reward, and moves the policy toward the episodes that beat their group's mean. The chart sets the training reward beside true success, the Strict grader on two held-out tickets, so a rising reward that does not fix tickets is visible.",
    whatYouSee: [
      "Reward design (Outcome only, Partial credit, Rubric judge), Checker (Strict grader or Sandbox test; marked not used under the Rubric judge), Training tickets (1, 3 or 6 kinds), Group size (2 to 32), KL to reference β (0 to 0.5), Step cost (0 to 0.10 per turn), Advantage scale (Divide by group std or Mean only), Seed, and Training update (0 to 200).",
      "A chart of Training reward (dashed) and True success (held-out) (solid) over the 200 updates, with a marker at the chosen update. The legend prints the values at update 200.",
      "Tiles at the chosen update: Training reward, True success (held-out), Hack rate (share of held-out episodes that call edit_check), Closed but not fixed, Needless look-ups (look-ups on the held-out ticket that gives its value), Mean turns, KL per turn (in nats), and Groups with no signal so far. A tile that reports a share also draws it as a meter from 0 to 100%.",
    ],
    howItWorks: [
      "Policy: logit(tool) = bias + 7 feature weights · features, softmax over six tools, 48 weights in all. It starts at a hand-written reference that solves 19% of held-out tickets.",
      "One update: take the next training ticket round-robin, sample G episodes, reward = score − Step cost × turns, advantage = teachingRl.groupAdvantages(rewards, { scale }), which is reward − group mean, divided by the group's standard deviation under Divide by group std. Then w += 0.6 / G × Σ [advantage × Σ over turns ∇log π(tool | state) − β × Σ over turns ∇KL(π ‖ π_ref)], with the exact KL gradient at each visited state.",
      "A group whose rewards are all equal gets zero advantage and counts as a group with no signal.",
      "Every curve and tile is an exact expectation of the policy at that update, computed by pushing probabilities through every desk state for 8 turns. Only the training path is sampled, so another seed gives another path.",
    ],
    controls: [
      "All nine controls above. Training update only chooses which saved update to read; it never retrains.",
      "Comparison worth running: with the defaults (Outcome only, Strict grader, 6 kinds, group of 8, seed 1), true success climbs from 19% to 84%. Set Checker to Sandbox test: training reward ends at 1.00 while true success ends at 0% and Hack rate at 100%.",
      "Then raise KL to reference β to 0.2: the hack rate falls to 42% at seed 1 but does not reach zero.",
    ],
    notice: [
      "Under the Sandbox test, the two curves split: the reward the optimizer sees rises while the held-out score falls.",
      "Under the Rubric judge, Closed but not fixed reaches 100% and Needless look-ups reaches 99%, because the judge pays for a look-up and never reads the record.",
      "At Group size 2, 160 of the 200 groups at seed 1 have no signal, against 39 at Group size 8.",
      "At Step cost 0.05, seed 1 learns to close on its first turn under Divide by group std (Mean turns 1.0) but not under Mean only.",
    ],
    limits: [
      "In this lab: the policy is a linear softmax over six tools with seven yes/no inputs, not a language model. There is one ticket per update, one gradient step per group with no probability ratio or clipping, an exact per-state KL instead of the per-token estimate, and a fixed learning rate of 0.6.",
      "In general: agent RL samples long transcripts from a language model, scores them with tests, graders or learned judges that each have their own loopholes, averages over many tasks per batch, and uses clipped updates. Rewards there are noisier and hacks harder to spot than an edit_check row.",
    ],
  },

  "Policy table": {
    title: "The trained policy, situation by situation",
    summary:
      "The 48 weights read as a table: six situations the desk can be in, each written as the seven yes/no features, and the chance the policy calls each of the six tools there. It shows what training changed, such as a look-up when the value is missing, or edit_check taking over under the Sandbox test.",
    whatYouSee: [
      "Rows: New ticket, Read, value given, Read, value missing, Value looked up, Updated, not checked, and Check failed. Columns: the six tools, with edit_check marked as the loophole.",
      "Each cell's shade grows with the probability; focus or hover a cell to read it. Each row sums to 100%.",
      "A list naming the likeliest tool in each situation, with its probability.",
    ],
    howItWorks: [
      "Each row is softmax(bias + weights · features) for that row's features, using the weights at Training update.",
      "The same 48 weights give every row, so a change learned in one situation moves the others that share its features.",
    ],
    controls: [
      "Training update and every training control.",
      "Comparison worth running: Training update 0 against 200 with the defaults, then the same with Training tickets at 1 kind, reading the Read, value missing row.",
    ],
    notice: [
      "With 6 kinds the Read, value missing row learns look_up; with 1 kind that row is never trained, so it keeps writing a guess.",
      "Under the Sandbox test the edit_check column grows from its starting logit of −2.5 until the hack dominates.",
    ],
    limits: [
      "In this lab: six of the situations the desk can produce are shown; passed checks and edited tests are left out to keep the table small. The probabilities are exact for the policy, not estimated.",
      "In general: a language model's policy cannot be laid out as a table, because its situations are every possible transcript and its actions every possible string; probing it takes sampled episodes.",
    ],
  },

  "Unseen tickets": {
    title: "Held-out tickets by value class, and the spread across ten seeds",
    summary:
      "Bars give Strict grader success on the held-out ticket that gives its value, the held-out ticket that leaves it out, and the training tickets. A strip shows where seeds 1 to 10 end at update 200 for the current controls, because one seed can hide an all-or-nothing outcome.",
    whatYouSee: [
      "Bars: Value given, Value missing (each one held-out ticket), and Training tickets, at the chosen update. Value missing turns to the warning colour, and says why, when no training ticket leaves its value out.",
      "Across 10 seeds: one column per seed on a 0% to 100% axis, with a filled dot for true success and a hollow diamond for the hack rate, a dashed line at the mean true success, and a shaded band behind the current seed, with its two values printed, when it is one of the ten.",
      "A line giving the mean true success, how many seeds hack more than half the time, and how many quit on their first turn.",
    ],
    howItWorks: [
      "Bars are exact Strict grader success rates for the policy at Training update, computed the same way as the training card's tiles.",
      "The strip retrains seeds 1 to 10 for 200 updates with every other control as set and evaluates each final policy exactly. It is cached by those controls, so moving Seed or Training update does not retrain it.",
    ],
    controls: [
      "Training tickets, plus every other training control.",
      "Comparison worth running: Training tickets at 1 kind against 6 kinds. At seed 1 Value missing goes from 4% to 78%, and in the strip every seed ends with lower true success on 1 kind than on 6.",
    ],
    notice: [
      "Held-out wording is new, but the policy never reads wording. A held-out ticket is new to it only when its value class, given or missing, never appeared in training.",
      "Under the Sandbox test all ten seeds hack. Under the Strict grader with a step cost and Divide by group std, some seeds quit at turn 1 while the rest learn: 1 of 10 at Step cost 0.05 and 4 of 10 at 0.10.",
    ],
    limits: [
      "In this lab: the eight tickets differ to the policy by one yes/no feature, so “6 kinds” means both value classes, three times each. Ten seeds of one toy say nothing about how often real training runs go wrong.",
      "In general: language-model agents read the task text, so new wording, new tools and new layouts can all break them, and the variety a training set needs is far larger and harder to measure than two value classes.",
    ],
  },
};

export default cardInfo;
