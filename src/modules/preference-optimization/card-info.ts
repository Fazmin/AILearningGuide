import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Implicit reward margin during training": {
    title: "The quantity DPO actually optimizes",
    summary:
      "The implicit reward margin at every DPO step, with the chosen and rejected rewards that make it up. Every point comes from a real DPO gradient run against a frozen reference model trained on this page.",
    whatYouSee: [
      "A chart of DPO step against implicit reward, β times a log-probability shift from the reference, averaged over the pairs.",
      "Four series with their final values: `margin (chosen − rejected)` solid, `chosen reward` dashed, `rejected reward` dotted, and `held-out margin (never trained on)`, a solid line in a second colour. The margin is exactly the distance between the chosen and rejected lines.",
      "A kicker with the training pair count, the held-out count, beta and steps, and two badges: the share of training pairs whose margin is strictly positive, and how many of the three held-out pairs are.",
      "Four metrics: `DPO loss` averaged over pairs, `Margin, β-scaled`, `Log-ratio gap m = margin ÷ β`, the same gap with β divided back out, and `Held-out pairs ordered correctly`.",
    ],
    howItWorks: [
      "The loss is `−log σ(β · ((log πθ(y_w|x) − log πref(y_w|x)) − (log πθ(y_l|x) − log πref(y_l|x))))`, averaged over the pairs. It depends on the policy only through that margin.",
      "The reference is 50 epochs on Harbor weather notes and Recipe steps at seed 1, or, under `Reference model`, that base fine-tuned on your preferred completions. The policy starts as an exact copy of whichever it is, so at step 0 every reward is 0 and the loss is ln 2 = 0.6931.",
      "A completion's log-probability sums its character transitions, starting from the prompt's last character. Prompts keep a trailing space, so `the | fog …` is scored as `the fog …`.",
      "Each step is one full-batch update over every pair: learning rate times the gradient β·σ(−β·m)·∇m.",
      "The held-out line is β times the mean log-ratio gap of three fixed pairs the optimizer never uses, read at five evenly spaced step counts along one run of the same update. A test checks it against the trainer.",
    ],
    controls: [
      "`DPO steps`, `Beta`, `DPO learning rate` and the pair editor all retrain from the reference.",
      "Comparison worth running: at the defaults the chosen reward ends at +0.556 and the rejected at −2.940. Set `DPO steps` to 0 and every line collapses to 0.",
      "Then read the held-out line. At the defaults the training badge reads 100% and the held-out badge reads 1 of 3, and the held-out margin ends at +0.42 against +3.50 for the training pairs. At the default beta and learning rate no step count orders more than one held-out pair.",
    ],
    notice: [
      "The margin is bought lopsidedly: the rejected side moves about five times further than the chosen side.",
      "Both sides falling while the margin grows would not be a fault. The loss sees only the difference.",
      "At 0 steps the badge reads 0% because it counts strictly positive margins, and every margin is exactly zero.",
      "The training margin keeps climbing with more steps. The held-out line does not follow it: two of the three held-out pairs get worse as training continues.",
    ],
    limits: [
      "In this lab: the objective, gradients, loss and rewards are genuinely computed, but the model is a 30 × 30 character bigram. The mechanism is real; the capability is not.",
      "In this lab: by default the reference is a base table trained 50 epochs on Harbor and Recipes. The `SFT checkpoint` choice is a second model trained here on your preferred completions with a response-only loss; the weights from `Instruction tuning & templates` are never loaded.",
      "In this lab: the held-out set is three fixed comparisons of one kind, a corpus sentence against a loop, written before any run. It shows that ordering does not carry over automatically, not how often it does.",
      "In this lab: every series is multiplied by β. The `Log-ratio gap` metric removes β so runs at different betas can be compared.",
      "In general: a rising margin is evidence about the margin. DPO can over-optimize just as RLHF can, because the margin itself is now the exploitable quantity.",
      "In general: DPO starts from a supervised fine-tuned checkpoint and uses that same checkpoint as the frozen reference, so the margin measures movement away from the SFT model.",
    ],
  },

  "Where each pair sits on the loss": {
    title: "Why a pair stops pulling on the weights",
    summary:
      "The DPO loss −log σ(β·m) drawn against the log-ratio margin m for half, the current and double beta, with every pair placed on the current curve. As a pair slides right, its loss flattens and its gradient fades.",
    whatYouSee: [
      "Loss on the y axis, from 0 to 1.2, against the log-ratio margin m in nats on the x axis, with a dashed line at m = 0 where every pair starts at ln 2.",
      "Three curves: the current β solid, half β dotted, double β dashed. Numbered circles are the pairs at their current m.",
      "One row per pair with its m, its loss, and its pull σ(−β·m), the share of full gradient strength it still carries.",
      "A note giving the m each β needs to reach a loss of 0.1.",
    ],
    howItWorks: [
      "m is the log-ratio margin, `Δ log p(chosen) − Δ log p(rejected)`, measured from the trained weights for each pair.",
      "The gradient of the loss with respect to m is `−β·σ(−β·m)`. The pull column prints σ(−β·m): 50% at m = 0, falling towards 0 as the pair is separated.",
      "The margin for a target loss L is `−ln(e^L − 1) / β`, so doubling β halves the change needed to reach any given loss.",
    ],
    controls: [
      "No controls of its own. `Beta` reshapes the curves; `DPO steps` and `DPO learning rate` move the dots.",
      "Comparison worth running: at the defaults the three pairs sit at m = 9.50, 8.85 and 7.88 with 2.2%, 2.8% and 4.1% pull. Set `DPO steps` to 4 and they sit near m = 3, still pulling at around 20%.",
    ],
    notice: [
      "The pair that is least separated pulls hardest. DPO spends its steps on the comparisons it has not yet won.",
      "At a higher β the same m sits further along the flat part. That is the sense in which β keeps the policy near the reference.",
      "A dot left of the dashed line is a pair the policy currently orders wrongly.",
    ],
    limits: [
      "In this lab: three pairs drawn on one curve. Real training estimates the loss on minibatches of thousands of pairs.",
      "In general: a small pull is not proof the pair is learned in any useful sense, only that this loss has stopped asking for more.",
    ],
  },

  "Drift from the reference model": {
    title: "How far the policy wandered",
    summary:
      "Average KL divergence from the frozen reference model for three complete DPO runs at half, the current and double beta, with a before-and-after sample. Drift is what the reference constraint is meant to hold down.",
    whatYouSee: [
      "A chart of DPO step against mean KL(policy ‖ reference) over 30 rows, in nats, scaled to the largest curve.",
      "Three series, the current beta solid and the others muted, dotted and dashed. Each legend entry gives the run's final log-ratio gap and its final KL.",
      "A badge with the current run's KL, and two samples, `Reference model` and `After preference optimization`, from the same prompt and seed.",
    ],
    howItWorks: [
      "Drift is `tinyRowDivergence`: the KL from each policy row to the reference row, averaged evenly over all 30 context rows.",
      "The three curves are independent runs at β/2, β and 2β from the same reference with the same steps and learning rate. The outer one can exceed the 1.50 the `Beta` slider allows.",
      "Both samples decode 76 characters from `the ` at temperature 0.7, seed 8, so any difference comes from the weights.",
    ],
    controls: [
      "`Beta` moves all three runs. `DPO steps`, `DPO learning rate` and the pairs change them too.",
      "Comparison worth running: at 40 steps, β 0.20, 0.40 and 0.80 end at KL 0.0133, 0.0102 and 0.0053. At 4 steps the order flips: 0.0005, 0.0013 and 0.0017.",
      "Switch `Reference model` to `SFT checkpoint` at the defaults: the KL at β 0.40 rises from 0.0102 to 0.0161, and it is measured from the SFT checkpoint, not from the base.",
    ],
    notice: [
      "Beta has two effects here. It scales every gradient step, so early on a higher beta moves faster; it also saturates the loss sooner, so after enough steps a higher beta has moved less.",
      "The gap in the legend shrinks as beta grows: 10.2, 8.7 and 6.2 nats at the defaults. Less separation is needed, so less is bought.",
      "At the extremes the tidy story breaks. At β 3.0 with these settings the steps are large enough to overshoot, and drift rises again.",
    ],
    limits: [
      "In this lab: the KL is computed over every row of both tables, but it is an even average over contexts, not weighted by how often each context occurs.",
      "In this lab: plain full-batch gradient descent. With Adam, which normalizes step sizes, beta's effect on step size largely disappears and its saturating role dominates.",
      "In general: RLHF's KL penalty is taken over the policy's own sampled completions, and a small average can hide a large change on a narrow slice of behaviour. Production runs also track held-out evaluations.",
    ],
  },

  "Preference pair editor": {
    title: "Writing the comparisons and setting the run",
    summary:
      "The three-column format the optimizer parses, plus the three numbers that define a DPO run. Every other panel on the page is recomputed from what you type here.",
    whatYouSee: [
      "A text area labelled `Preference pairs` with the hint `prompt | preferred | rejected`, one comparison per line.",
      "A note on the size limit: up to 5 lines, each part up to 50 characters. Text past either limit is ignored.",
      "`Reference model`, a two-way switch between `Base model` and `SFT checkpoint`, with a line saying how the chosen one was trained.",
      "`DPO steps` from `untrained` at 0 to 120 in twos, `Beta` from 0.05 to 1.50, and `DPO learning rate` from 0.05 to 1.00.",
    ],
    howItWorks: [
      "Each line is split on the bars, trimmed, lowercased and cut to 50 characters a part, and kept only if both completions are non-empty. At most 5 pairs are kept. The prompt gets one trailing space.",
      "Text is encoded into the 30-symbol vocabulary: space, a–z, full stop and two reserved brackets. Other characters become a space, so digits and capitals vanish.",
      "Only the prompt's final character conditions the first completion character. With the trailing space, every prompt ends in the space row.",
      "The base reference is trained once and cached. The `SFT checkpoint` is that base plus 60 full-batch steps at learning rate 0.35 on the preferred completions: next-character loss on the completion only, with the prompt read as context and never graded. This is the fallback the DPO paper gives when no SFT model exists.",
      "The three DPO runs are recomputed on every change.",
    ],
    controls: [
      "`Preference pairs` defines the comparisons. `DPO steps` is the honest baseline: at 0 the loss reads 0.6931 and every reward is exactly 0.",
      "Comparison worth running: hold the pairs fixed and compare `Beta` 0.10 with 1.00 at the same step count, reading the gap and the KL together.",
      "Then switch `Reference model` to `SFT checkpoint`. The reference now already prefers the chosen completion on two of the three training pairs, where the base preferred none, and the loss at 0 steps still reads 0.6931. After 40 steps the held-out margins are −1.48, −1.76 and +2.70.",
    ],
    notice: [
      "Rewriting a prompt changes nothing unless its last character changes, because the model sees one character of context.",
      "A line with an empty prompt is scored from the space row.",
      "`Beta` moves the reward scale and the reference constraint at once; they are one parameter here.",
    ],
    limits: [
      "In this lab: three hand-written pairs updated full-batch is not a dataset, and most real preference data could not be expressed in this vocabulary.",
      "In this lab: the SFT checkpoint is trained on the same preferred completions the pairs contain, so it prefers them by construction. A real SFT set is separate and far larger.",
      "In general: preference datasets run to tens of thousands of comparisons, are usually collected from another model's outputs, and carry their annotators' biases toward length, confidence and formatting.",
    ],
  },

  "Per-pair rewards": {
    title: "The implicit reward for each comparison",
    summary:
      "One row per pair: both completions, a dumbbell placing their implicit rewards on an axis centred at the reference, and the margin between them. This is where you see whether every pair carries the average.",
    whatYouSee: [
      "A numbered block per training pair (numbers match the loss curve) with the prompt and both completions, then three blocks labelled `H1` to `H3` under a heading for the held-out pairs.",
      "A dumbbell: a circle for the chosen reward, a square for the rejected reward, a bar between them for the margin, and a dashed centre line at 0, the reference. All pairs share one scale.",
      "Both rewards printed with signs, and a footer with the margin, whether the reference already preferred the chosen completion, and the two lengths when they differ.",
      "A solid left edge and the words `ordered correctly` on pairs with a positive margin, a dashed edge and `not ordered` on the rest.",
    ],
    howItWorks: [
      "A reward is `β × (log πθ(completion|prompt) − log πref(completion|prompt))`, the same quantity the training chart averages.",
      "The margin is the chosen reward minus the rejected reward, the per-pair quantity the loss is a decreasing function of.",
      "The reference flag compares raw total log-probabilities, with no length normalization.",
    ],
    controls: [
      "No controls of its own. Every control in the pair editor moves these rewards.",
      "Comparison worth running: at the defaults the chosen rewards are +0.18, +0.51 and +0.97 and the rejected ones −3.61, −3.03 and −2.18.",
    ],
    notice: [
      "In all three default pairs the base reference preferred the repetitive rejected completion. DPO changed an ordering rather than reinforcing one.",
      "At the defaults `H1` and `H2` are not ordered and `H3` is, with β-scaled margins of −0.54, −1.12 and +2.90. `H3`'s loop contains ` and the `, as training pair 2's does: train on pair 2 alone and `H3` ends at +2.25, on pair 1 alone at +0.28.",
      "The first pair's chosen completion is longer, 28 characters against 20, which alone makes its raw log-probability lower.",
      "Raising `Beta` stretches every reward, even where the policy moved less.",
    ],
    limits: [
      "In this lab: character-level log-probabilities of short strings. Treat the ordering as real and the magnitudes as a toy's.",
      "In general: an implicit reward is reference-relative and defined per prompt. Rewards against different references, or for different prompts, are not comparable numbers.",
      "In general: length-normalized and reference-free variants exist precisely because raw log-probability and preference data both carry length bias.",
    ],
  },
};

export default cardInfo;
