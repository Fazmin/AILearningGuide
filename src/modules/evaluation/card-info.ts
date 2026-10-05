import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Leaderboard": {
    title: "Four checkpoints ranked, with the uncertainty drawn in",
    summary:
      "The same character bigram trained at four epoch budgets, ranked under a scoring rule you choose: by default how often it gives the correct continuation a higher per-character log-probability than the distractor. Every score carries a 95% interval, and a paired comparison says whether the top two are actually separated by these items. Changing the rule re-scores the checkpoints and can reorder them.",
    whatYouSee: [
      "A `Scoring rule` selector (Length-normalised, Raw log-probability, Sample and check) and, under Sample and check, an `Attempts allowed (k)` slider from 1 to 10, with a note describing the chosen rule.",
      "A forest plot: one row per checkpoint, the headline score with its 95% interval on the left (pass rate with a Wilson interval, or mean pass@k with a t interval), and the rule's second quantity on the right (mean margin, or mean pass@1). The leader's intervals are shaded across all rows, so overlap is visible at a glance.",
      "A table sorted by the headline score, then by the second quantity: `Passed` or `Mean pass@k`, `95% interval`, `Mean margin` or `Mean pass@1`, `Benchmark perplexity`, and `Control perplexity`. The leader row is highlighted.",
      "Metrics `Leader's standard error`, `Intervals overlapping the leader's`, and `Items for ±5 pts`.",
      "A callout with the leader-minus-runner-up margin difference, paired over the same items, and whether its interval excludes zero.",
    ],
    howItWorks: [
      "Budgets are 3, 10, 30, and 80 epochs times `Training budget multiplier`, each with batch 16, learning rate 0.6, seed 1, on Harbor weather notes minus its last three sentences. Changing the scoring rule never retrains them.",
      "Length-normalised: a candidate's score is the mean log-probability of the continuation “ candidate” after the context: the space and every candidate character, divided by that many transitions. Raw log-probability keeps the total instead, so each extra character multiplies in another probability below one and the shorter option gains a head start.",
      "Sample and check: the correct option's share is 1 / (1 + exp(distractor total − correct total)). Each item takes 20 draws from a seeded generator, a draw picking the correct option when a uniform number falls below that share. With c correct draws the item scores pass@k = 1 − C(20 − c, k) / C(20, k), and the checkpoint scores the mean over items. The seed depends on the item only, so every checkpoint reuses the same dice.",
      "Standard error is √(p(1 − p)/n). The drawn interval is Wilson's, which stays sensible at 0/n and n/n where p ± 1.96·SE collapses to zero width. Margin intervals use Student's t with n − 1 degrees of freedom.",
      "`Items for ±5 pts` is ⌈1.96² · p(1 − p) / 0.05²⌉ at the leader's pass rate, clamped to 5–95%: 385 items at 50%. Under Sample and check it uses the per-item variance of pass@k instead, with a floor of two items.",
    ],
    controls: [
      "`Scoring rule` and `Attempts allowed (k)` re-score every row. `Benchmark items`, `Training budget multiplier`, and the contamination switch retrain every row.",
      "Comparison worth running: with the defaults, the 80- and 30-epoch checkpoints both pass 3 of 6 with intervals 19%–81%, and the paired margin gap is +0.002 with an interval from −0.220 to +0.224. The ranking is a coin flip.",
      "Then switch to Raw log-probability: every checkpoint gains two passes (80 epochs goes from 3 of 6 to 5 of 6) and the 30-epoch checkpoint edges ahead. Under Sample and check, k = 1 puts the 10-epoch checkpoint first and the 80-epoch one third; at k = 5 the 3-epoch checkpoint leads at 94% and the 80-epoch one is last at 83%.",
    ],
    notice: [
      "A shaded band covering another checkpoint's point means these items cannot tell them apart. Under all three rules every interval still overlaps the leader's.",
      "The two extra raw passes come from the two items whose correct option is shorter, sails against trails and repeats against retreats: the rule, not the model, changed.",
      "Pass@k measures coverage. The 3-epoch checkpoint spreads probability over both options, so a few attempts nearly always include the key; the 80-epoch checkpoint is confidently wrong on port against town and stays near zero on it however many attempts it gets.",
      "The paired comparison is sharper than comparing two separate intervals, because both checkpoints answer the same items and item difficulty cancels.",
      "With contamination on, the leader passes 6/6 and its standard error reads ±0.0 points: the normal approximation claims certainty from six items. The Wilson interval, 61%–100%, does not.",
    ],
    limits: [
      "In this lab: each checkpoint is one training run at one seed, so the intervals cover item sampling only, not training noise. The model is a 30 × 30 character bigram. Sample and check draws a choice between the two options, not free text, because a bigram almost never writes a whole target word by chance, so its pass@k measures coverage over two options; the 20 draws per item add noise the interval does not include.",
      "In general: a scoring rule is a design decision. Likelihood, lettered options, generated answers with exact match, and judge models order the same models differently. A leaderboard gap smaller than about two standard errors is not evidence of a better model, and published comparisons should report the rule, intervals, and paired tests when models share items.",
    ],
  },

  "Item-level results": {
    title: "Both scores, for every item, at the largest budget",
    summary:
      "The strongest checkpoint scored item by item under the chosen scoring rule, with a bar for each item, so you can see which questions it passed, which it failed, and by how much. The callout restates the contamination experiment as a paired shift against the clean run.",
    whatYouSee: [
      "A kicker `Largest checkpoint · N epochs`.",
      "One row per item: the context, both scores joined by `›` or `‹` (raw totals or per-character means), the signed margin, a verdict, and a bar growing right of centre for a pass and left for a fail. Under Sample and check the row shows how many of the 20 draws picked the key, the item's pass@k, and a bar that grows from the left.",
      "Passing and failing rows differ in border colour, bar direction, and the verdict text.",
      "Metrics `Pass rate` with its 95% interval, `Mean margin`, and `Control perplexity`.",
    ],
    howItWorks: [
      "The list always shows the last of the four budgets, not the current leader.",
      "Each score is the one the chosen rule uses on the leaderboard (per-character mean, or total log-probability); the margin is correct minus distractor. Under Sample and check the item's value is its pass@k from 20 draws.",
      "With contamination on, the callout's margin shift is the paired mean of (leaked margin − clean margin) over the same items at the same budget and seed, with a t interval (pass@k under Sample and check). The control shift compares the two runs' Proverbs perplexity.",
    ],
    controls: [
      "No controls of its own. The scoring rule, editing items, the multiplier, or the contamination switch rewrites every row.",
      "Comparison worth running: “the boats stay in | port | town” fails by −0.840 when clean, because after a space this corpus usually writes “t”. Leak the source and it passes by +0.012.",
    ],
    notice: [
      "A pass by 0.01 and a pass by 1.2 count the same in `Passed`. The bars show the difference.",
      "The biggest margin, +1.213 on “the water stays”, is decided by the letters inside “warm” and “wide”, not by knowing about water.",
    ],
    limits: [
      "In this lab: only the largest checkpoint is expanded; the others are summarized in the table. Draws under Sample and check are seeded, so the counts are repeatable.",
      "In general: item-level inspection is how a benchmark solved by a surface cue — shared characters, a repeated template, a leaked n-gram — gets caught.",
    ],
  },

  "Benchmark editor": {
    title: "Write the items the leaderboard will score",
    summary:
      "The harness: one item per line as `context | correct | plausible distractor`, plus a multiplier that stretches all four training budgets together. The three character counts are the three corpora the page uses.",
    whatYouSee: [
      "A textarea labelled `context | correct | plausible distractor`.",
      "`Training budget multiplier` from 0.25× to 3.00× epochs.",
      "Metrics `Training characters`, `Benchmark source`, and `Control`.",
    ],
    howItWorks: [
      "A line becomes an item when it splits on `|` into at least three non-empty parts. Extra parts are ignored and case is folded.",
      "Each candidate is scored as the continuation “ candidate”: log p(space | last context character) plus each candidate character given the one before, divided by the candidate's length plus one. That is per-byte length normalization, the idea behind lm-evaluation-harness's acc_norm.",
      "At 0.25× the budgets become 1, 3, 8, and 20 epochs; at 3× they become 9, 30, 90, and 240.",
    ],
    controls: [
      "`Benchmark items` and `Training budget multiplier`.",
      "Comparison worth running: replace `morning | evening` with `morning | xyzzy` and watch the item become an easy pass. The item is only informative when the distractor is a real word.",
    ],
    notice: [
      "Training, benchmark source, and control are three different texts; only the first is trained on.",
      "Writing items from the training sentences instead of the held-out ones contaminates the clean leaderboard from the start.",
    ],
    limits: [
      "In this lab: scoring is likelihood-based, so a model that would never generate the right word can still pass by ranking it higher. Only the first 60 lines count, and each is cut at 160 characters, so a pasted benchmark cannot freeze the page.",
      "In general: likelihood, generation with exact match, and an LLM judge are different scoring rules and can rank the same models differently.",
    ],
  },

  "Contamination control": {
    title: "Leak the source text, not the questions",
    summary:
      "A switch that appends the three held-out harbor sentences to the training corpus, the way a scraped page leaks a public benchmark. Both conditions are always trained, so the leak's effect is measured as a paired shift against the clean run.",
    whatYouSee: [
      "A pressable row `Add the benchmark source sentences to the corpus` with a `held out` or `leaking` chip.",
      "The three held-out sentences.",
      "Metrics `Margin shift` (leaked minus clean mean margin, largest checkpoint; `Pass@k shift` under Sample and check) and `Control shift` (percent change in Proverbs perplexity).",
    ],
    howItWorks: [
      "Off: the training text is the harbor notes without the last three sentences. On: those sentences are appended.",
      "The items never enter training. A model that now answers them saw “in winter the same harbor freezes”, not the item line.",
      "The shifts compare two runs with the same budget and seed, so they do not change when you flip the switch.",
    ],
    controls: [
      "The switch.",
      "Comparison worth running: at the default budget, Margin shift is +0.519 per item while Control shift is −2.8%. The benchmark moves far more than general text.",
    ],
    notice: [
      "The control improves a little after a leak. Real English is real English; the signature is the disproportion, not a frozen control.",
      "At 30 epochs the leak takes the leaderboard from 3/6 to 5/6 and benchmark perplexity from 10.08 to 6.65.",
    ],
    limits: [
      "In this lab: the leak is a verbatim copy of three known sentences, the easiest kind to detect.",
      "In general: real leakage arrives as paraphrases, translations, and discussions of the test, which exact-match and n-gram checks miss.",
    ],
  },
};

export default cardInfo;
