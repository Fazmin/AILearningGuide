import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Problem": {
    title: "A question the policy already practiced",
    summary:
      "Problem chooses Sunday hours, 17 + 28, or a dog on the ferry. They stand for items a reasoning model met during post-training, not few-shot examples pasted into a prompt. The contrast with the ICL lab is the recipe, not the harbor facts: a prompt changes what the model reads, while training against a checker changes its weights.",
    whatYouSee: [
      "Problem, three segments, and the question text.",
      "A note contrasting the two recipes: examples in a prompt (no weight change) and post-training with reinforcement learning against answers a program can check (weights change).",
    ],
    howItWorks: [
      "Switching Problem writes problem and resets Thinking budget to 3.",
      "No prompt with worked examples is built. The traces are authored to look like trained behaviour.",
    ],
    controls: [
      "Problem.",
      "Comparison worth running: Sunday hours against 17 + 28. The arithmetic trace has a step whose result the next step reads (the carry); the lookup trace mostly does not.",
    ],
    notice: [
      "The harbor facts match other labs so the course stays one world.",
      "Neither the Train against a checker card nor the sampling card depends on which problem is selected: one trains a toy policy over trace length, the other is arithmetic on an assumed accuracy.",
    ],
    limits: [
      "In this lab: three authored problems. No model samples the steps.",
      "In general: a reasoning model is still a next-token predictor with a different post-training recipe, not a new layer type.",
    ],
  },

  "Train against a checker": {
    title: "A toy policy learns from a checker to write longer traces",
    summary:
      "A one-parameter policy over trace length (1 to 8 steps) is trained by a group-relative policy gradient, GRPO-style, against a hand-written checker: a trace of L steps passes when at least k of its steps land, each landing with probability 0.80. It shows a verifiable reward making traces longer, an unpriced length drifting past what the task needs, and a per-step cost trimming it. It is not a language model.",
    whatYouSee: [
      "Controls Steps the task needs (k) from 1 to 4, Group size (G) from 2 to 32, Learning rate from 0.1 to 1.0, Seed, Training update (a scrubber over the 100 updates), and Length cost Off or On.",
      "Two learning curves with a marker at the chosen update: Checker pass rate (plus Mean reward after cost when the cost is on), and Mean trace length against a dotted line at k.",
      "Bars for the policy over trace length at that update: solid bars now, outlines for the starting policy, a dashed line for the checker's pass chance at each length, and a vertical rule left of which a trace is too short to ever pass.",
      "Metrics Checker pass rate, Mean length, Steps past k, Mean reward after cost, and Groups with no signal (updates so far whose sampled group had identical rewards).",
    ],
    howItWorks: [
      "Policy: a softmax over lengths 1 to 8 with logits −(L − μ)² / 2. The single trained number is μ, which starts at 1.5, so short traces dominate and the mean length is 1.8 steps.",
      "Checker: pass chance for length L is P(at least k of L steps land) with each step landing at 0.80. Reward is pass minus 0.05 × L when Length cost is on, and pass alone when it is off.",
      "Each of the 100 updates samples G lengths, runs the checker on each, and sets advantage = (reward − group mean) / (group standard deviation + 1e-6). Then μ += rate × (1/G) × Σ advantage × (L − mean length). A group with identical rewards has zero advantage and changes nothing.",
      "The curves and bars are exact expectations of the current policy. The path μ takes comes from the sampled groups, so a run is deterministic in the seed and the controls and a different seed gives a different path.",
    ],
    controls: [
      "Steps the task needs (k), Group size (G), Learning rate, Seed, Training update, and Length cost.",
      "Comparison worth running: with the defaults (k = 3, group of 16, rate 0.3, seed 1) and Length cost Off, mean length ends at 7.1 steps and the pass rate at 99%. Turn Length cost On: the length ends at 5.3 steps and the pass rate at 93%.",
      "Then set Group size to 8 with the cost on: the policy collapses to one-step answers (1.2 steps, 1% pass rate).",
    ],
    notice: [
      "Length emerges because the checker rewards it. A trace shorter than k never passes, and a trace of exactly 3 steps passes 51% of the time, so longer traces earn more.",
      "With the cost off, length keeps rising after the pass rate saturates: Steps past k reads 4.1 at update 100, and in 51 of 100 updates the sampled group had identical rewards and taught nothing. That is overthinking.",
      "The cost trims length but not to k. It ends 2.3 steps past k, because spare steps still insure against a slip.",
      "On a hard task (k = 4) with the cost on, no early group contains a pass, so the cost is the only difference between traces and length collapses.",
    ],
    limits: [
      "In this lab: the policy is one number over eight lengths, not a language model. The checker is a hand-written rule with a fixed step success of 0.80. There is one prompt, one gradient step per sampled group, no reference model, and no clipping. The traces on the other cards are not produced by this policy.",
      "In general: real RLVR and GRPO train a token-level policy over a language model, add a KL penalty to a reference model, clip probability ratios, average over many prompts per batch, and face checkers that can be gamed. Longer traces there help on some problems and are wasted tokens on others.",
    ],
  },

  "Thinking budget": {
    title: "How many authored steps are revealed",
    summary:
      "Thinking budget reveals 0 to 6 authored steps, useful first, then padding, then an unfaithful step, then a final useful one. The answer slot is authored and prints at every budget. Extra tokens are test-time compute, not a weight update.",
    whatYouSee: [
      "Thinking budget, a slider whose maximum is the trace length.",
      "Metrics Useful, Padding, Unfaithful, and Answer (authored).",
      "A note on why a trace can help: each token is another forward pass that can read what earlier steps wrote.",
    ],
    howItWorks: [
      "visible = trace.slice(0, budget). The three counts filter the kind field.",
      "Answer (authored) always prints the problem's canned answer. It is not computed from the visible steps.",
    ],
    controls: [
      "Thinking budget.",
      "Comparison worth running: 3 against 6 on Sunday hours, then 0 on 17 + 28.",
    ],
    notice: [
      "Padding does not change the answer. Unfaithful steps name unused facts or the wrong operation.",
      "On real hard problems, forcing a short trace usually lowers accuracy. This lab does not model that, which is why the answer is labelled authored.",
    ],
    limits: [
      "In this lab: the order useful → padding → unfaithful is authored. A real sampler interleaves them.",
      "In general: a fixed-depth transformer does a bounded amount of computation per token; a chain lets it spread serial work across many tokens, and that is where long traces help.",
    ],
  },

  "Inspectable trace": {
    title: "Tokens labelled useful, padding, or unfaithful",
    summary:
      "Each revealed line is a generated-looking sentence with a label this lab assigns. The labels are teaching marks, not a probe of an inner algorithm. A process reward model would score these steps; this page only displays them.",
    whatYouSee: [
      "An ordered list; an empty budget prints that no intermediate tokens ran.",
      "Each row has a kind label and the sentence.",
    ],
    howItWorks: [
      "The list is the prefix of the authored trace.",
      "Distillation, if this were production, would copy these strings into a student. The student would copy text, not an algorithm.",
    ],
    controls: [
      "This card has no controls.",
      "Comparison worth running: budget 4 on Dog on the ferry (padding about ticket price) against budget 5 (an invented leash rule).",
    ],
    notice: [
      "An unfaithful step can sit beside a correct answer. Faithfulness is a separate claim from accuracy.",
      "On 17 + 28, the first line writes the carry that the second line uses: the trace is working memory.",
    ],
    limits: [
      "In this lab: the trace is a template. No model writes it, and no process reward is computed.",
      "In general: visible steps can be useful and still unfaithful. Studies that edit or truncate a model's trace find the final answer sometimes ignores it.",
    ],
  },

  "Sampling and voting": {
    title: "More samples, and two ways to pick one",
    summary:
      "The other kind of test-time compute: draw n independent answers, each right with probability p, then either keep the most common answer (majority vote) or keep any answer a perfect checker accepts (verifier picks). Both curves are exact binomial arithmetic.",
    whatYouSee: [
      "Controls Samples (1 to 32), Single-sample accuracy (p from 0.05 to 0.95), and Wrong answers: Always the same one, Split over 3, or All different.",
      "A chart of accuracy against n: dotted one sample (flat at p), solid majority vote, dashed verifier picks, with markers at the current n.",
      "Bars for the binomial probability of k correct samples out of n. The solid part of each bar is the chance the vote then picks the right answer.",
      "Metrics One sample, Majority vote, Verifier picks, and Decode cost in multiples of one sample's tokens.",
    ],
    howItWorks: [
      "Verifier picks = 1 − (1 − p)^n, the chance at least one sample is right. That is pass@n, the ceiling for any way of choosing among the samples.",
      "Majority vote = Σₖ P(k correct) × P(the right answer wins the plurality | k), with ties broken at random. Wrong samples are spread evenly over 1 or 3 wrong answers, or are all different; the win probability is computed exactly by counting how the wrong votes can fall.",
    ],
    controls: [
      "Samples, Single-sample accuracy, and Wrong answers.",
      "Comparison worth running: at p = 0.40 and 16 samples, Split over 3 gives a majority vote of 70.8% against a verifier's 99.97%. Switch to Always the same one and the vote falls to 21.3%, below one sample.",
    ],
    notice: [
      "Voting amplifies whichever single answer is most likely. If one wrong answer beats the right one, more samples make the vote worse.",
      "Two samples never beat one under voting: a split vote is a coin toss.",
    ],
    limits: [
      "In this lab: p is your assumption, samples are independent, and the verifier is perfect. Real samples from one model are correlated, and learned verifiers make mistakes that selection can exploit.",
      "In general: sampling more helps only with a way to choose. Unit tests, proof checkers, and exact answer keys are the checkers that make best-of-n reliable.",
    ],
  },
};

export default cardInfo;
