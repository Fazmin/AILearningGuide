import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Task vectors": {
    title: "Two fine-tunes, written as differences from one base",
    summary:
      "A harbor base is trained once. Two fine-tunes start from that same base, one on Recipe steps and one on Proverbs. Each task vector is the fine-tuned table minus the base table, and the two maps draw both vectors cell by cell.",
    whatYouSee: [
      "A kicker naming the base and where the vectors came from, and a badge with the cosine similarity of the two task vectors.",
      "`Task vectors from` (`Full fine-tunes` or `LoRA rank 4`), `Recipe epochs`, and `Proverb epochs`, each 0 to 40 in steps of 2.",
      "Two 30 × 30 maps, `τ recipes` and `τ proverbs`. Rows are the previous character, columns the next character. Blue raises that logit, red lowers it, colour depth is |entry| on one shared scale, and grey means unchanged. A diagonal strike marks every entry whose sign disagrees with the other vector.",
      "Four metrics: each specialist's loss on its own corpus, how many of the 900 entries both vectors changed, and how many of those disagree in sign.",
    ],
    howItWorks: [
      "The base is `trainTinyModel` on Harbor weather notes for 48 epochs at seed 1. Full fine-tunes continue from it at learning rate 0.6, seeds 3 and 4. The LoRA option trains a rank-4, alpha-8 adapter instead and uses its product `(α / r)·B·A` as the task vector.",
      "`τ = θ_tuned − θ_base`, entry by entry. A specialist is `θ_base + τ`, scored with full-corpus cross-entropy.",
      "Rows for characters a corpus never uses as context (`j`, `q`, `x`, `z`, the brackets) receive no gradient, so they stay grey. That holds for LoRA too: the row of B for an unused context never receives gradient and stays at zero.",
      "Sign statistics count an entry as changed when |τ| > 1e-6. Cosine is the ordinary normalized dot product of the two 900-entry vectors.",
    ],
    controls: [
      "`Task vectors from` swaps the fine-tuning method. `Recipe epochs` and `Proverb epochs` set how far each fine-tune travels from the base.",
      "Comparison worth running: at 20 epochs each, read Sign conflicts with `Full fine-tunes` (96 of 720, 13%), then switch to `LoRA rank 4` (331 of 720, 46%).",
    ],
    notice: [
      "Both fine-tunes touch the same 720 entries, because both corpora use the same 24 context characters. Overlap alone does not mean conflict: 87% of those entries agree in sign.",
      "The cosine of 0.34 is positive. The two tasks share English structure, so part of each vector helps the other task.",
      "Set one epoch slider to 0 and that map goes grey: an untrained fine-tune is a zero task vector, and every merge reduces to the other task alone.",
    ],
    limits: [
      "In this lab: the model is a 30 × 30 bigram table with no hidden units, so the permutation problem that breaks merges of independently trained networks cannot occur here. Every vector shares the base on purpose.",
      "In this lab: a softmax row ignores a constant added to all its logits. Full fine-tunes never add one, since each row's gradient sums to zero, but LoRA products do, so LoRA maps and cosines include direction that changes no prediction.",
      "In general: merging assumes the checkpoints share a base, or at least sit in one linearly connected basin. Fine-tunes of one pretrained model usually do; two models trained from different random seeds usually do not.",
    ],
  },

  "Merge recipe": {
    title: "The merge as arithmetic you can check",
    summary:
      "Pick a method and its knobs. The formula prints the exact operation with current values, and the strip shows ten real coordinates: the two inputs, what each method keeps or discards, and the merged value.",
    whatYouSee: [
      "`Merge method`: `Linear blend`, `Task arithmetic`, `TIES`, `DARE`, `SLERP`. Below it, only that method's controls: `Blend t`, `Scale λ`, `Keep top`, `Drop rate p`, and `New random mask`.",
      "A `Served weights` formula with the current numbers substituted, and both merged losses as its result. A badge reports the merged vector's length.",
      "A strip of ten coordinates named `previous→next`: the five largest sign conflicts, then the five largest agreements, ranked by |τ_recipes| + |τ_proverbs|. Each has three bars in a fixed order, recipes, proverbs, merged, on one symmetric scale, with the merged value printed.",
      "For TIES, trimmed inputs are dashed outlines and the elected sign is printed above or below the merged bar. For DARE, dropped inputs are dashed outlines and a short line in the text colour marks each survivor's height before rescaling.",
    ],
    howItWorks: [
      "Linear blend: `θ_base + (1 − t)·τ_r + t·τ_p`, which equals averaging the two fine-tuned checkpoints. Task arithmetic: `θ_base + λ·(τ_r + τ_p)`.",
      "TIES: keep each vector's top `Keep top` share by magnitude, set each entry's sign to the sign of the summed kept values, average only kept entries with that sign, then scale by λ.",
      "DARE: drop each entry of each vector with probability p using a seeded mask, multiply survivors by `1 / (1 − p)`, add both, then scale by λ.",
      "SLERP: `(sin((1 − t)Ω)·τ_r + sin(tΩ)·τ_p) / sin Ω`, where Ω is the angle between the two task vectors. mergekit applies SLERP to whole checkpoints; this lab applies it to the task vectors so the curvature is visible.",
    ],
    controls: [
      "`Merge method` swaps the rule. `Blend t` moves along the recipes-to-proverbs line. `Scale λ` stretches the combined vector. `Keep top` sets the TIES trim. `Drop rate p` and `New random mask` set DARE's mask.",
      "Comparison worth running: pick TIES and move `Keep top` from 100% to 10%. With 100% kept, every conflict column is resolved by sign election alone; with fewer kept, small opposing entries are trimmed first.",
    ],
    notice: [
      "In a conflict column, Linear blend leaves the merged bar between the two inputs, partly cancelled. TIES leaves it at the size of the winning side, because the losing side is excluded from the mean.",
      "At `Keep top` 20% with the default vectors, 38 of the 96 conflicts survive trimming in both vectors and need an election; in the other 58, at least one side was trimmed away first.",
      "DARE's survivors grow by `1 / (1 − p)`. At p = 0.5 each kept entry doubles, which is why the strip's scale jumps.",
    ],
    limits: [
      "In this lab: two task vectors only, one seed each, and the strip shows 10 of 900 coordinates chosen for size. The other 890 still count in every loss.",
      "In this lab: SLERP is applied to task vectors, not to full checkpoints. With a shared base, full checkpoints are nearly parallel and SLERP on them is almost the same as a linear blend.",
      "In general: TIES and DARE were designed for many large fine-tunes whose small deltas are largely redundant. Their published gains come from that regime, with λ tuned on held-out data.",
    ],
  },

  "Two task losses": {
    title: "Where each merge lands on both tasks",
    summary:
      "Every merge scored on both corpora at once, plotted against the base, the two specialists, and the corner you would reach by keeping two separate models. A table lists every method at the current settings and splits each gap into interference and shrinkage.",
    whatYouSee: [
      "Recipe loss on the x axis and proverb loss on the y axis, both in nats per character. Lower and further left is better on both.",
      "Fixed markers: `harbor base` (hollow circle), `recipe specialist` (square), `proverb specialist` (triangle), and `two separate models` (cross), which sits at each specialist's own loss.",
      "The selected merge as a large filled circle, the other four methods as small hollow circles tagged Lin, TA, TIES, DARE, SL, a dotted line for Linear blend from t = 0 to 1, and a dashed line for Task arithmetic from λ = 0 to 1.5.",
      "A table with each method's recipe and proverb loss, its gap to the specialist, and its interference. A note below states what the other task did to each loss.",
    ],
    howItWorks: [
      "Each point is `tinyCrossEntropy` of `θ_base + τ_merged` on the full Recipe steps and Proverbs corpora.",
      "Interference on recipes is `loss(merge(τ_r, τ_p)) − loss(merge(τ_r, 0))`: the same method re-run with the proverb vector set to zero. Proverbs is scored the same way with the recipe vector removed.",
      "Gap minus interference is the cost of the method shrinking, trimming or dropping a task's own vector. SLERP has no one-vector version, so its interference is left blank.",
      "The plot stretches to hold the anchors, the linear path and the selected merge. Any other merge further out is pinned to the edge, drawn with a dashed outline, and labelled with its true losses.",
    ],
    controls: [
      "No controls of its own. Every control on the two cards above moves these points.",
      "Comparison worth running: on Linear blend at t = 0.5, the gaps are +0.134 and +0.161 while interference is −0.037 and −0.055. Switch to Task arithmetic at λ = 1 and interference turns positive, +0.156 and +0.089.",
    ],
    notice: [
      "Negative interference means the other task vector helped. In the 50/50 average, both gaps come from halving each task vector, not from the other task.",
      "The dashed Task arithmetic path bends back: both losses fall until about λ = 0.7 to 0.8, then both rise. Too much of both vectors overshoots.",
      "No merge reaches the `two separate models` cross. That gap is the price of serving one file instead of two.",
    ],
    limits: [
      "In this lab: the losses are training-set cross-entropy on eight-to-ten sentence corpora, with no held-out split. They say which merge fits these sentences, not which generalizes.",
      "In this lab: one DARE mask per setting. `New random mask` shows how much a single draw moves the point; at p = 0.5, masks #1 to #5 put recipe loss anywhere from 2.50 to 3.11.",
      "In general: merges are judged on held-out benchmarks for every task at once, and the scaling coefficient is tuned there. A merge that wins on two losses can still lose formatting, safety, or calibration behaviour that no loss here measures.",
    ],
  },
};

export default cardInfo;
