import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "One pass through the training loop": {
    title: "The same loop, named one stage at a time",
    summary:
      "A walkthrough of one training iteration on the bundled Harbor weather notes character bigram: dataset, batch, forward, loss, backward, optimizer step, checkpoint. Every metric on this card is measured from the run the sliders on the right just defined.",
    whatYouSee: [
      "A kicker `Stage N of 7` and a badge naming the corpus `Harbor weather notes`.",
      "A looping stage strip: Dataset, Batch, Forward, Loss, Backward, Optimizer step, Checkpoint. The current stage is marked; earlier ones are marked done.",
      "A title and a paragraph that change with the stage, plus three metrics that belong to that stage only.",
      "Dataset reports Characters, Training pairs, and Vocabulary 30. Batch reports batch size, batches per epoch, and total steps. Forward reports the top two next-character predictions after `the fo` and the row width. Loss reports this checkpoint's loss, its perplexity, and the uniform baseline 3.401. Backward reports the gradient norm of one real pair, the last logged batch norm, and the peak batch norm, and draws that pair's gradient as a strip of 30 bars. Optimizer step reports the fixed learning rate 0.6, how many of the 30 rows actually moved, and steps taken. Checkpoint reports how many snapshots were saved, the selected step, and its loss.",
    ],
    howItWorks: [
      "The model is a 30 × 30 logit table: the row for the previous character is softmaxed to get the next-character distribution. Training is minibatch SGD on cross-entropy at learning rate 0.6, seed 1.",
      "Text is encoded into ` abcdefghijklmnopqrstuvwxyz.<>`. Adjacent encoded characters become (context, next) pairs. The pair count is one less than the encoded length.",
      "A model that has learned nothing scores `ln 30 ≈ 3.401` nats per token, because that is the cross-entropy of the uniform distribution over 30 symbols.",
      "The forward, loss, and checkpoint stages, and the Backward strip, read the checkpoint selected on the right, not the end of the run. The batch norms and the optimizer stage quote the last recorded history point of the finished run.",
      "The Backward strip is the exact gradient of −log p(g | o) with respect to the 30 logits in row `o`: each character's predicted probability, minus 1 at `g`, because the corpus continues `the fo` as `the fog`.",
      "The strip is an indicator. Clicking a stage changes which explanation you see and nothing about the weights.",
    ],
    controls: [
      "Click a stage on the strip to change which part of the loop is described. The lesson stepper does not move this strip.",
      "`Batch size`, `Epochs`, and `Checkpoint` on `Run controls and checkpoint timeline` retrain the run or reload a snapshot, and every metric here follows.",
      "Comparison worth running: open Dataset and read Training pairs, then open Batch and raise `Batch size` from 16 to 64. Batches per epoch falls; total steps falls; the pair count does not.",
    ],
    notice: [
      "Forward's top prediction is the row for `o` — the last character of the probe `the fo` — taken from the selected checkpoint. It will not match a sample drawn from a different snapshot.",
      "A fall of `ln 2 ≈ 0.693` nats means the probability given to the correct next character doubled, as a geometric mean over the pairs. That conversion is why the Loss stage mentions 0.69.",
      "In the Backward strip the one large bar points down at `g`; every other bar is small and points up, the largest on the model's current favourite. The update raises `g` and lowers everything else, which is all learning means for this row.",
      "Only rows that appeared in some batch receive gradient. The optimizer stage's `Rows updated` reads 26 / 30 on this corpus: `j`, `x`, and the two reserved angle brackets never appear as a context character, so their rows stay at zero.",
    ],
    limits: [
      "In this lab: every count, loss, and gradient norm is measured from a real training run in the browser. The model is still a character bigram — one previous character of context, 900 weights, no layers.",
      "In this lab: the stage walkthrough is a labelling of one already-finished run. It does not pause mid-batch or show a live tape; the Backward strip is one pair's gradient at a saved checkpoint, not the batch gradient the optimizer actually applied at that step. The `Step one batch` card is where a single real update is replayed.",
      "In general: a production step also does mixed precision, gradient accumulation, data-parallel all-reduce, and an optimizer with per-parameter state. The seven names stay the same; the engineering around each one does not.",
    ],
  },

  "Measured training loss": {
    title: "Batch loss and whole-corpus loss over the whole run",
    summary:
      "Two measured curves against optimizer step: a thin, noisy batch-loss line, and a thick line through the whole-corpus loss at each saved checkpoint, with the selected checkpoint drawn as a large dot. Nothing on the chart is illustrated.",
    whatYouSee: [
      "The x axis is optimizer step, with small ticks at every epoch boundary; the y axis is loss in nats per token from 1.50 to 3.55. A badge repeats the total step count.",
      "A thin line of batch loss, a thick line with a dot per checkpoint for the whole-corpus loss, and a dotted `knows nothing` line at 3.401, which is `ln 30`.",
      "A dashed guide and a large dot at the checkpoint selected on the right, labelled with its step and loss. The legend prints the last batch value, the final whole-corpus loss, and the baseline.",
      "Three metrics: `Final loss` on the whole corpus after the last step, `Perplexity` as `exp` of that same loss, and `Better than chance by` as the percent of the way from 3.401 down to zero.",
    ],
    howItWorks: [
      "Each batch-loss point is the mean minibatch cross-entropy over a window of steps, so that line is noisier. Different batches are different slices of the shuffled pairs.",
      "Each checkpoint dot is cross-entropy over all 498 training pairs at the saved weights. The last dot is exactly `Final loss`; `Perplexity` is its exponential.",
      "The dotted line is the analytic uniform baseline, not a trained model. A curve above it is worse than guessing evenly among 30 characters.",
      "Changing `Batch size` or `Epochs` retrains from the same zeroed table at seed 1, so the curve is a new run, not a continuation.",
    ],
    controls: [
      "This card has no controls of its own. `Batch size` and `Epochs` redraw both curves; `Checkpoint` only moves the large dot.",
      "Comparison worth running: at 12 epochs, read Final loss at batch size 16 and at batch size 64. Fewer, less noisy steps are not automatically a better fit on a corpus this small.",
    ],
    notice: [
      "The large dot can sit well before the end of the run. The metrics still report the finished run; the stage card and the sample on the right follow the dot.",
      "Batch loss jitters above and below the smooth whole-corpus line. That jitter is sampling, not instability — unless the thick line climbs back toward the dotted one.",
      "Count the epoch ticks against the steps: at batch 16 an epoch is 32 steps, at batch 64 only 8. The same 12 epochs is a very different number of updates.",
      "`Better than chance by` is a distance to the uniform baseline, not an accuracy. 40% better than chance on 30 characters is still a weak model.",
    ],
    limits: [
      "In this lab: every point is measured from SGD on this corpus. The y axis starts at 1.50; the lowest value any setting reaches is about 1.69 for a batch-loss window and 1.81 for the whole corpus (batch 2, 60 epochs), so nothing is clipped.",
      "In this lab: there is no held-out curve on this card. `Final loss` is training loss. The Dataset building module is where a held-out number first appears in this track.",
      "In general: a training-loss curve reports fit to the data the optimizer saw. Real runs plot a validation curve beside it and treat divergence between the two as the stopping signal, not a low training loss.",
    ],
  },

  "Run controls and checkpoint timeline": {
    title: "Change the loop, then reopen a snapshot",
    summary:
      "The two knobs that define the run — how big a batch is, and how many passes it takes — plus a timeline that reloads saved weights and samples from them. A checkpoint is a copy of the table, not a different model.",
    whatYouSee: [
      "`Batch size` from 2 to 64 pairs, `Epochs` from 1 to 60 passes, and `Checkpoint` whose output reads `step N` for the selected snapshot.",
      "A sample paragraph drawn from the selected checkpoint at temperature 0.8, prompt `the `, length 96, seed 21.",
      "A bar list `Next character after \"the fo\" at this checkpoint`: the top five softmax probabilities from the row for `o`, with the leader highlighted.",
    ],
    howItWorks: [
      "The trainer saves nine or ten checkpoints including the untrained table at step 0. Moving `Checkpoint` loads that copy; it does not rewind the optimizer.",
      "Changing `Batch size` or `Epochs` retrains and jumps the timeline to the last snapshot. Snapshots are saved every `floor(steps / 8)` steps plus one at the very end, so most settings save nine and some save ten.",
      "The sample uses the same prompt and seed at every checkpoint, so differences in the text come from the weights. Temperature 0.8 is a softmax sharpening, not extra knowledge.",
      "The bar list is `tinyTopTokens` on the probe `the fo`. Because the model is a bigram, only the final `o` conditions the distribution.",
    ],
    controls: [
      "`Batch size` sets pairs per update and therefore steps per epoch, which is `ceil(pairs / batch size)`.",
      "`Epochs` sets how many shuffled passes run. Total steps equal epochs times batches per epoch.",
      "`Checkpoint` scrubs the saved snapshots. Stage metrics that say `this checkpoint` follow it; `Final loss` on the chart does not.",
      "Comparison worth running: leave Epochs at 12, scrub Checkpoint from 0 to the end, and read the top bar. The mass should move from a near-uniform spread toward characters that actually follow `o` in harbor notes: `r` reaches 23.5% (harbor, morning, storm), then `v` (over, moves).",
    ],
    notice: [
      "Checkpoint 0 is the zeroed table. Its sample is close to noise and its top-5 is nearly flat; that is the honest untrained baseline.",
      "A sample can look worse at a later checkpoint than an earlier one even while loss fell. One 96-character draw at temperature 0.8 is not a metric.",
      "Raising batch size makes each step see more pairs and makes the loss curve less jagged, at the cost of fewer steps for the same epoch count.",
    ],
    limits: [
      "In this lab: learning rate, seed, schedule, and corpus are fixed — 0.6, seed 1, constant step size, Harbor weather notes — so the only recipe knobs are batch size and epochs.",
      "In this lab: nine or ten checkpoints are spaced through the run rather than saved every step, so the timeline cannot land on an arbitrary step.",
      "In general: a real checkpoint also stores optimizer state, RNG state, and the data cursor, because resuming a run is not the same as evaluating a frozen table. Sampling from a checkpoint, as this card does, only needs the weights.",
    ],
  },

  "Step one batch": {
    title: "One real update, replayed from the selected checkpoint",
    summary:
      "Press the button and the next minibatch the run drew after the selected checkpoint is applied once to that checkpoint's weights. The card shows the pairs in the batch, the rows of the 900-weight table that moved, the loss on that batch before and after, the batch gradient norm, and the whole-corpus loss before and after.",
    whatYouSee: [
      "A kicker with the learning rate 0.6 and the seed, a badge counting presses (at most 24), and two buttons: `Step one batch` and `Back to the checkpoint`.",
      "A line naming the step (for example `Step 0 → 1`), the epoch and batch it came from, and how many pairs it held, then one chip per pair written as previous character, arrow, next character. A space is drawn as `␣`.",
      "Four metrics: `Batch loss, before → after` on the pairs in the batch; `Batch gradient norm`, the length of the whole batch gradient; `Whole-corpus loss, before → after` on all 498 pairs; and `Weights changed`, as a count out of 900 and a count of rows.",
      "A table of up to eight rows, largest gradient first. Each row is a previous character. It lists how many pairs in the batch began with it, the length of that row's gradient, the next character that row was asked for most often (its target), and that target's logit and probability before and after.",
    ],
    howItWorks: [
      "The trainer shuffles the 498 pairs once per epoch with a seeded generator and cuts the shuffle into batches. A checkpoint saved at step `s` therefore has a known next batch: batch `s mod batches-per-epoch` of epoch `floor(s / batches-per-epoch)`. The card rebuilds that batch from the same seed.",
      "The update is the trainer's own: every pair adds `(predicted probabilities − 1 at the true character) / batch size` to its row of the gradient, and each touched row then loses `0.6 × gradient`. Rows that no pair in the batch began with are not touched.",
      "Pressing again chains the next real batch onto the result, so after `n` presses the weights are exactly what the run would hold at step `s + n`. A test compares every one of these updates with the trainer's own per-step checkpoints, and they match to the last bit.",
      "Batch loss is measured on the batch's own pairs with the weights before and after. Whole-corpus loss is `tinyCrossEntropy` on all 498 pairs with the same two sets of weights.",
    ],
    controls: [
      "`Step one batch` applies the next update. `Back to the checkpoint` returns to the selected snapshot. Moving `Checkpoint`, `Batch size`, or `Epochs` also resets the count, because they change which weights and which batches are meant.",
      "Comparison worth running: set `Checkpoint` to step 0 and press once. Batch loss goes 3.401 → 3.362 and whole-corpus loss goes 3.401 → 3.396, with nine rows and 270 weights changed. Then set `Batch size` to 2, pick the checkpoint at step 1492, and press repeatedly.",
    ],
    notice: [
      "Exactly the rows a pair began with move, and every weight in each of them moves, because the softmax gives every character some probability. Nine rows touched is 270 of 900 weights; the table lists the eight with the largest gradient, and every other row is bit-for-bit unchanged.",
      "The loss on a step's own batch fell at every one of the steps in the 12-epoch runs at batch sizes 2, 16, and 64. The whole-corpus loss falls by a much smaller amount, and at `Batch size` 2 it rises on 17 of the first 24 presses from the checkpoint at step 1492. Over the whole run it rises on 1,174 of 2,988 steps at batch 2, on 5 of 384 at batch 16, and on none of 96 at batch 64.",
      "The batch gradient norm is the quantity a gradient clip compares with its threshold. For the first update from step 0 it is 0.2567; the Hyperparameters & schedules lab plots it at every step.",
    ],
    limits: [
      "In this lab: the learning rate (0.6, constant), the seed (1), and the corpus are fixed, so this replays the lab's own run rather than letting you choose a batch. There is no momentum, no clipping, and no weight decay in the update.",
      "In this lab: a press past the final checkpoint draws the next epoch's batch from the same generator, which is a step the saved run never took. The card says so on the step line.",
      "In general: a real step also runs through many layers, so one batch moves matrices in every layer, and the optimizer (Adam, usually) rescales each parameter's step by its own running statistics. The loss on the batch falling while the loss elsewhere does not is the same noise there; it is why runs average many batches.",
    ],
  },

  "Where the memory goes": {
    title: "Bytes per parameter, times parameters",
    summary:
      "A closed-form estimate of the memory the model state needs while training: weights, gradients, and optimizer state, for a chosen parameter count, precision, and optimizer. It is arithmetic on the per-parameter byte counts, not a measurement of any run, and it leaves activations out.",
    whatYouSee: [
      "Three segmented controls: `Model size` (this lab's 900 weights, then 125 million, 1.5 billion, 7.5 billion, and 70 billion), `Precision` (fp32 or Mixed 16-bit), and `Optimizer` (SGD, Momentum, or Adam).",
      "A stacked bar whose segments are weights, gradients, and optimizer state, with their sizes written inside. Segment width is proportional to bytes. With SGD in fp32 the optimizer-state segment disappears.",
      "A table with the same numbers: bytes per parameter and the total for each part, the optimizer state broken into its pieces (fp32 master copy, momentum, variance), a total row, and an `Activations` row that says they are not included. The badge repeats the total.",
      "Three metrics: bytes per parameter, optimizer state as a multiple of the weights, and how many 32 GB devices the model state alone would need.",
    ],
    howItWorks: [
      "Total = parameters × (weight bytes + gradient bytes + optimizer-state bytes). In fp32 the weights and gradients take 4 bytes each. In Mixed 16-bit they take 2 each and the optimizer also keeps a 4-byte fp32 master copy. SGD adds no buffers, Momentum adds one 4-byte buffer, and Adam adds two.",
      "Mixed-precision Adam is therefore 2 + 2 + (4 + 4 + 4) = 16 bytes per parameter, the figure in the ZeRO paper (Rajbhandari et al., 2019). That paper's own examples reappear here: 24 GB for 1.5 billion parameters and 120 GB for 7.5 billion.",
      "Units are decimal (1 GB is 10^9 bytes), as in the paper. The device count is `ceil(total / 32 GB)`, where 32 GB is the V100 size in the paper's Table 1.",
    ],
    controls: [
      "`Model size` multiplies every row; the proportions of the bar do not change. `Precision` and `Optimizer` change the bytes per parameter and so the proportions.",
      "Comparison worth running: at 7.5 billion parameters with Adam, read the total in fp32 and in Mixed 16-bit. They are the same 120 GB. Then switch to SGD and read what the optimizer-state segment becomes.",
    ],
    notice: [
      "Mixed precision does not shrink the model state under Adam: 16 bytes per parameter either way. What moves is where the bytes sit. In Mixed 16-bit, the optimizer state is six times the weights; in fp32 it is twice.",
      "The optimizer state is the largest part at 7.5 billion parameters: 90 GB of 120 GB under Adam in Mixed 16-bit. Dropping to SGD removes the momentum and variance but keeps the master copy, so 8 bytes per parameter remain.",
      "At 70 billion parameters the model state is 1.12 TB, which fills 35 devices of 32 GB before a single activation is stored. That is why state is divided across devices rather than held on one.",
    ],
    limits: [
      "In this lab: everything on the card is multiplication. No memory was allocated or measured, and the lab's own 900-weight trainer is the only real buffer: 3,600 bytes of fp32 weights and 3,600 bytes of gradient, with no optimizer state under plain SGD.",
      "In this lab: activations are left out on purpose, so the bar is a floor, not a forecast. The ZeRO paper's example, a 1.5 billion parameter model at sequence length 1,000 and batch size 32, needs about 60 GB of activations on top of its 24 GB of model state, or about 8 GB with activation recomputation.",
      "In this lab: data, tensor, and pipeline parallelism and ZeRO and FSDP sharding are described, not simulated. The card has no device count, no communication cost, and no per-device split.",
      "In general: real frameworks add buffers, fragmentation, and temporary memory, and mixed-precision recipes differ in which copies they keep. LoRA and QLoRA cut the gradient and optimizer rows by training only a small adapter; see the LoRA & adapters lab.",
    ],
  },
};

export default cardInfo;
