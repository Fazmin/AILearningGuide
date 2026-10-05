import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Training recipe": {
    title: "The first three stages, as six controls",
    summary:
      "The knobs for pretraining a base table, adapting it to a second task with a merged low-rank adapter, and aligning it with DPO. Each one changes what the later cards measure.",
    whatYouSee: [
      "`Base training epochs`, `LoRA rank`, `Adapter epochs`, `Replay sentences`, `DPO steps`, and `DPO beta`, each with its current value beside the label.",
      "A note listing which sentences each stage trains on and which two sentences of each corpus are held out.",
    ],
    howItWorks: [
      "Stage 1 trains the 900-weight character table on six harbor sentences with minibatch SGD (seed 1). Stage 2 trains a rank-`r` adapter on the first six recipe sentences plus the first `k` replayed harbor sentences, then merges it into the table. Stage 3 runs DPO on three preference pairs, with the adapted table as the frozen reference.",
      "Every run is deterministic in the controls, so the lab stores controls only and recomputes all four stages whenever one changes.",
      "Shared state from a link is clamped to the slider ranges before anything trains.",
    ],
    controls: [
      "Comparison worth running: with `Replay sentences` at none, raise `Adapter epochs` from 8 to 40. Harbor loss in `Stage by stage` rises at every setting, while recipe loss gains little and ends worse than the base model. Then set replay to four and repeat: harbor loss barely moves and recipe loss falls.",
      "`DPO steps` at 0 makes After aligning identical to After adapting, so the cost of alignment is one slider away.",
    ],
    notice: [
      "Replay and adapter epochs pull in opposite directions: more epochs fit the new task harder, more replay keeps the old one in the update.",
      "Rank 1 barely moves the model at all, which fails the gate for a different reason than forgetting does.",
    ],
    limits: [
      "In this lab: the model is a single table of 900 logits, so adapting and aligning edit the same weights, and a corpus of eight sentences stands in for a dataset. Nothing here is a transformer, a tokenizer, or an instruction-following model.",
      "In general: a real pipeline pretrains for weeks on trillions of tokens, adapts on curated data with a held-out split per task, and runs preference training against a reward signal collected from people or a verifier.",
    ],
  },

  "Stage by stage": {
    title: "What each stage did to both texts",
    summary:
      "Held-out loss on the recipe text and the harbor text after each of four stages, the six-item benchmark counts for both, and the preference-pair result from DPO. Changes between stages are written in words, not only signs.",
    whatYouSee: [
      "A table with a row per stage (Base model, After adapting, After aligning, As shipped) and columns for recipe-text loss, harbor-text loss, and benchmark items answered correctly out of six.",
      "Small notes beside each loss: the change from the stage above, labelled `worse` or `better` when it moves by more than 0.02 nats.",
      "Four metrics: base-to-shipped change in each loss, the share of preference pairs ordered correctly, and the reward margin.",
      "A callout listing every stage-to-stage rise above 0.02 nats, or saying there was none.",
    ],
    howItWorks: [
      "Loss is mean cross-entropy over the held-out characters, in nats. The two held-out texts are the last two sentences of each corpus, which no stage trains on.",
      "The benchmark scores each candidate as its continuation after the context, per transition, and counts an item correct when the right word outscores a wrong but ordinary one. All twelve items are held-out or freshly written phrases, and a test checks that none appears in a training text.",
      "`Preference pairs ordered` is the share of the three DPO pairs whose reference-relative reward margin is positive; `Reward margin` is the mean scaled margin.",
    ],
    notice: [
      "Adapting without replay for 16 epochs or more costs harbor loss more than it gains recipe loss. Aligning then raises both losses a little, which is the alignment tax in this toy.",
      "A stage can improve the benchmark count while worsening held-out loss, or the reverse. Six items move in jumps.",
    ],
    limits: [
      "In this lab: twelve benchmark items and two held-out sentences per text are far too few to separate close models, and a bigram reads each candidate only through its own characters and the space before it, so context barely matters.",
      "In general: release evaluation uses thousands of items, several domains, and checks for contamination, because a single pair of held-out texts can be overfit by the person tuning against it.",
    ],
  },

  "Eval gate": {
    title: "A bar set in advance, run on the shipped artifact",
    summary:
      "Two checks decided by held-out loss, read on the model as it would ship: the recipe text must improve by at least a set amount, and the harbor text may not get worse by more than a set amount. The benchmark appears beside them as advisory evidence.",
    whatYouSee: [
      "A `Gate: pass` or `Gate: fail` badge, two sliders for the thresholds, and one line per check with `Pass` or `Fail`, the loss it measured, and the base model's value.",
      "A note giving each benchmark's counts before and after, their 95% intervals, and whether the intervals overlap.",
    ],
    howItWorks: [
      "Target check: `base recipe loss − shipped recipe loss ≥ Required recipe-loss gain`. Control check: `shipped harbor loss − base harbor loss ≤ Allowed harbor-loss rise`. The gate passes only if both do.",
      "The intervals are Wilson 95% intervals for the item counts, from the same helper the Evaluation lab uses. They are advisory because six items cannot settle a close call.",
      "The gate reads the last stage, after quantizing and pruning, not the model you trained.",
    ],
    controls: [
      "Comparison worth running: with the default thresholds, find the smallest `Replay sentences` that passes the harbor check, then drag `Allowed harbor-loss rise` to its maximum and notice that a gate loosened until it passes decides nothing.",
    ],
    notice: [
      "The benchmark counts often fail to move while loss moves by tenths of a nat; that is the case for gating on loss at this scale.",
      "Both thresholds are yours, and that is the weakness: nothing stops you setting them after seeing the result.",
    ],
    limits: [
      "In this lab: the thresholds are sliders so a learner can feel how much the verdict depends on them. A real gate fixes them before the run and keeps them under review.",
      "In general: a gate is one layer. Releases also run safety and regression suites for the application, human review for high-stakes changes, and a staged rollout with monitoring.",
    ],
  },

  "Compress and serve": {
    title: "Shrink the weights, then fit the load",
    summary:
      "Quantize the aligned table to a named format and optionally prune it, which changes the model the gate reads; then size an imagined 8B-class deployment for the format, user count, and context length chosen.",
    whatYouSee: [
      "A segmented control for `Weight format` (F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0), and sliders for `Prune`, `Concurrent users`, and `Context length`.",
      "A first row of metrics from the toy: its file size and bits per weight at that format, and the recipe and harbor loss as shipped, coloured by whether they moved more than 0.02 nats from After aligning.",
      "A second row for the imagined deployment: 8B-class weights, KV cache for all streams, streams that fit, and tokens per second per user.",
      "A callout stating the headroom or the overshoot against a 24 GiB accelerator.",
    ],
    howItWorks: [
      "The toy is quantized with the format's block layout (bits, scope, group size) and its error is the rounding to that grid; pruning then zeroes the smallest-magnitude weights, a fraction of all 900.",
      "The deployment arithmetic is the serving lab's: weights = parameters × bits per weight ÷ 8, using the published whole-file size for each format; KV cache = 2 × layers × KV heads × head dimension × 2 bytes × context × users; per-user speed = 1 ÷ ((weights + cache) ÷ memory bandwidth) for a batched step.",
    ],
    controls: [
      "Comparison worth running: at a passing recipe, step `Weight format` from F16 to Q4_0 and watch the toy's losses barely move, then drag `Prune` to 90% and watch them jump.",
      "Set `Concurrent users` to 64 and `Context length` to 32,768 and switch between F16 and Q4_K_M: the cache, not the weights, is what overflows.",
    ],
    notice: [
      "Quantization at 4 bits or above costs almost nothing here; pruning is what hurts, and not gradually.",
      "A smaller format helps the budget twice: fewer bytes of weights, and fewer bytes read per decode step.",
    ],
    limits: [
      "In this lab: the quality numbers come from the toy and the budget from arithmetic about a different, imagined model, so no figure on this card is a measurement of a model that runs. The 8B shape is a stand-in for a real size.",
      "In general: compression quality depends on the model and the calibration data, and unstructured pruning saves time only on hardware that skips zeros. Real throughput also depends on kernels, batching policy, and the interconnect, which this arithmetic leaves out.",
    ],
  },

  "Release report": {
    title: "The decision, and what it cost to get there",
    summary:
      "One verdict that combines the quality gate and the serving budget, the artifact it applies to, every stage that made something worse, and, when the answer is hold, which condition failed and what to try.",
    whatYouSee: [
      "A `Verdict: ship` or `Verdict: hold` badge and a headline.",
      "A summary of the artifact (format, pruning, adapter rank), the gate result, the serving budget, and the stage-to-stage regressions.",
      "On hold, a callout with each failing condition's measured numbers and a suggested next move.",
    ],
    howItWorks: [
      "The verdict is `ship` only if both gate checks pass and the imagined deployment fits in memory at the chosen load.",
      "The regressions list is every stage-to-stage rise in either held-out loss above 0.02 nats, so a loss cannot hide inside a net gain.",
    ],
    notice: [
      "A hold with a one-line reason is usually cheaper to fix than a ship with a regression nobody read.",
      "Changing the thresholds changes the verdict without changing the model, which is why they belong in the plan, not in the run.",
    ],
    limits: [
      "In this lab: a pass means the toy cleared two held-out texts and fitted one imagined machine. It says nothing about a real model's safety, accuracy on real tasks, or behaviour under real traffic.",
      "In general: release decisions weigh quality, safety, cost, latency, and risk, usually with people accountable for the call. A verdict from a script is an input to that decision.",
    ],
  },
};

export default cardInfo;
