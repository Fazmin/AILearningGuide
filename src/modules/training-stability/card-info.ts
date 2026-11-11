import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Fault and fixes": {
    title: "Break one thing, then fix it",
    summary:
      "Picks one of four faults, or none, for the same rank-8 factored model on the same six Harbor sentences, and shows only the fixes that fault has. Every change retrains the whole 480-step run from step 1, with the same seed, so any difference on the dashboard comes from the fault and the fix.",
    whatYouSee: [
      "A `Fault` control with five options: None, Learning rate too high, fp16 overflow, Garbage batch and Logit growth.",
      "A note under it naming the fault and the recipe: the optimizer and its learning rate, the schedule, the clip, and the batch size of 16, the 480 planned steps and the seed of 9 that every run shares.",
      "The fixes for the chosen fault: `Gradient clip` and `Schedule` for Learning rate too high; `Loss scale` and `Scaling` for fp16 overflow; `Gradient clip` and `Roll back and skip` for Garbage batch; `z-loss` and `Learning rate` for Logit growth.",
      "A badge in the corner with the outcome: finished, finished worse than chance, or halted at a step.",
    ],
    howItWorks: [
      "None is plain SGD at a learning rate of 0.5 with a constant schedule. Learning rate too high is the same run at 8. fp16 overflow is the None run with every gradient multiplied by the loss scale, rounded to half precision and divided back. Garbage batch is Adam at 0.1 with five batches of `qqqq zzzz xxxx jjjj` in place of the real batches at steps 400 to 404. Logit growth is Adam at 1.",
      "Adam here has no weight decay, which makes it identical to AdamW with its decay set to zero.",
      "`Gradient clip` rescales any batch gradient longer than the threshold to exactly the threshold. It applies only to the two faults that show it; the other runs ignore it.",
      "`Warmup + cosine` is the SDK's one-cycle curve: a tenth of the rate at step 1, the full rate after 15 percent of the steps, then half a cosine down to 2 percent.",
      "`Dynamic` scaling skips any step whose gradient overflowed, halves the scale, and doubles it after 200 clean steps. `Static` keeps one scale, so an overflow reaches the weights.",
      "`Roll back and skip` reruns the Garbage batch run with steps 400 to 404 read but never applied, as if restarted from the checkpoint saved after step 350. It sets one flag; press it again to see the run without the rollback.",
    ],
    controls: [
      "`Fault` picks the run. Changing it keeps every fix where you left it, so a clip you set for one fault is still set when you come back.",
      "`Gradient clip` runs from off to 1.50 in steps of 0.05. `Loss scale` runs from 2⁰ to 2²⁴. `z-loss` steps through off, 0.0001, 0.001 and 0.01. `Learning rate` for Logit growth steps through 0.01, 0.03, 0.1, 0.3 and 1.",
      "Comparison worth running: at Learning rate too high, set Gradient clip to 0.50 and compare Constant with Warmup + cosine. The first ends at 3.172 nats, the second at 1.904, below the healthy run's 2.039.",
    ],
    notice: [
      "Neither fix for the high learning rate works alone. Warmup + cosine without a clip stays finite but ends at 19.152 nats; the clip without a schedule ends at 3.172.",
      "Garbage batch uses Adam because plain SGD barely reacts to the same garbage: at 0.5 it lifts the whole-corpus loss by only 0.044 nats, against 1.395 under Adam.",
    ],
    limits: [
      "In this lab: each fault is injected on purpose, one at a time, into a model with 480 weights and no attention, so the cause of every failure is known in advance. Real failures overlap and rarely announce their cause.",
      "In this lab: fp16 is emulated only where the gradient is cast; the forward pass, the weights and the update stay in 32-bit floats.",
      "In general: real runs combine these fixes from the start, with a warmup, a clip near 1.0, bf16 or a dynamic loss scale, z-loss or QK-norm, and checkpoints to roll back to, because finding the problem after the fact costs a large share of the run.",
    ],
  },

  "Run monitor": {
    title: "Four signals from every optimizer step",
    summary:
      "The dashboard a training team watches, measured on this run at every one of its steps: the loss, the gradient norm, the largest logit and the update-to-weight ratio. A shaded band on each panel is the normal range, the 5th to 95th percentile of a clean reference run, so a departure from it is the warning sign.",
    whatYouSee: [
      "Four panels sharing one x axis, the optimizer step from 1 to 480. Loss uses a linear axis; Gradient norm, Max logit and Update ÷ weight use log axes labelled in powers of ten, where equal distances are equal multiples.",
      "Loss: a thin line for each batch's own loss, a thick line for the whole-corpus loss every 4 steps, a dashed line for the reference run's whole-corpus loss, and a dotted line at ln 30 = 3.401, a model that knows nothing.",
      "Gradient norm: dark dots on steps the clip rescaled, a dashed green line at the clip threshold, and red triangles along the top on steps whose fp16 gradient overflowed.",
      "Update ÷ weight: a dashed line at 10⁻³, the rule of thumb for large networks.",
      "Markers on every panel: a red line and tinted zone after a halt, grey bars on skipped steps, an amber band on garbage batches, and a dashed green line at the checkpoint a rollback restarted from. With a rollback, a dotted amber line shows the whole-corpus loss without it.",
      "A stronger tint between a line and its band wherever the signal is outside the normal range, and the word normal beside each band.",
      "A legend under the panels naming the reference run and every marker that is showing.",
    ],
    howItWorks: [
      "Every value comes from the trace `trainTinyExtended` records at each step: the batch loss before the update, the global gradient norm before clipping, the largest logit in the batch, and the length of the weight change divided by the length of the weights after it.",
      "The whole-corpus loss is the cross-entropy of the saved weights on all 378 training pairs, every 4 steps.",
      "With z-loss on, the trainer adds the penalty's gradient, but every loss drawn here is the cross-entropy alone, so runs with and without z-loss are scored on the same scale.",
      "A value that is not a finite number is never drawn or printed. On the step a run halts, and on every skipped step, the panels leave a gap and draw a marker instead.",
      "Log axes cover whole powers of ten around the data, so their range changes with the run; read the labels before comparing two faults.",
      "The reference run is the healthy SGD run for Learning rate too high and fp16 overflow, the same Adam run with no garbage for Garbage batch, and Adam at 0.1 with no z-loss for Logit growth. With Fault on None the run is its own reference.",
    ],
    controls: [
      "`Steps shown` zooms all four panels to the first 40 steps or, for Garbage batch, to steps 380 to 440. It changes only the view; the panels redraw whenever a setting on Fault and fixes changes.",
      "Pointing at any panel draws a line at that step on all four and prints each signal's value there.",
      "Comparison worth running: with Fault on Learning rate too high, compare Update ÷ weight and Gradient norm on the first steps. The update ratio starts at 0.310, about fifteen times the healthy first step, while the gradient norm does not rise above its band until step 6.",
    ],
    notice: [
      "At Learning rate too high, gradient norm and max logit climb in nearly straight lines on their log axes, the mark of exponential growth, until a value overflows a 32-bit float and the run halts at step 154.",
      "The healthy run's Update ÷ weight band, about 0.021 to 0.040, sits far above the 10⁻³ line. In a small model trained with large steps the rule of thumb does not hold; the band does.",
      "In Garbage batch, the batch loss jumps on the first garbage step while the whole-corpus loss peaks 4 steps later, after the damage has been applied.",
    ],
    limits: [
      "In this lab: one model, one corpus, one seed and 480 steps. The bands come from one clean run, and a 5th to 95th percentile band is a description of that run, not a test of significance.",
      "In this lab: the model has no attention layers, so attention-logit growth, the instability QK-norm fixes, cannot appear; only the output logits can grow.",
      "In general: production dashboards log these signals per layer and per parameter group, add throughput, memory and loss-scale curves, and alert on sudden changes, because a global average can stay calm while one layer blows up.",
    ],
  },

  "Run report": {
    title: "What the dashboard is telling you",
    summary:
      "The run's outcome in numbers, measured on this run: its final whole-corpus loss or the step it halted on, the largest gradient norm and logit it reached, and how many steps were clipped, skipped or overflowed, with a sentence naming the mechanism behind what happened and the next thing to try.",
    whatYouSee: [
      "A scale from 1.5 to 4 nats with a dot for this run's final loss, a diamond for the clean reference run, and a dotted line at 3.401; the tinted stretch right of it is worse than a model that knows nothing. A halted run shows a cross instead of a dot.",
      "`Final loss` in nats per character, or `halted`; `Halted at`, a step number or `never`; and `Peak gradient norm`.",
      "`Clipped steps` out of the steps run, or `clip off`; `Skipped steps`; and `Overflow steps`, gradients that overflowed in fp16.",
      "`Peak max logit` and `Median update ÷ weight`, then one more readout that depends on the fault: `Loss scale` from start to end for fp16 overflow, `Spike` for Garbage batch, and `Median log-partition` otherwise.",
      "A callout: tinted red when the run halted, ended worse than chance or spiked, grey otherwise.",
    ],
    howItWorks: [
      "`Final loss` is the cross-entropy of the final weights on all 378 training pairs. It is `halted` when the run stopped because a weight or the loss was no longer a finite number.",
      "Each probability is floored at one in a trillion before taking its log, so one pair can cost at most about 27.6 nats; a loss near that cap is a lower bound.",
      "`Spike` is the highest whole-corpus loss from step 400 on minus the last value before it, with the step it peaked on. After a rollback it reads `skipped`.",
      "`Loss scale` reads the scale after the last step; dynamic scaling halves it on every overflow and doubles it after 200 clean steps.",
      "`Median log-partition` is the median over steps of each batch's mean log Z, the log of the softmax normalizer, which z-loss pulls toward zero.",
    ],
    controls: [
      "The report has no controls; it follows whatever Fault and fixes sets.",
      "Comparison worth running: at Garbage batch, read `Spike` with Gradient clip off (+1.395 nats at step 404) and at 1.00 (+0.734 at step 408), then press Roll back and skip and read `Final loss`: 1.993, against 2.022 without it.",
    ],
    notice: [
      "Dynamic scaling at 2²⁰ ends with `Loss scale` at 2¹⁷: three halvings settled the scale, and the doubling after 200 clean steps was halved again at step 430.",
      "At Logit growth the loss ends at 11.515 nats with every value still a finite number: logits of a few hundred do not overflow, so nothing halts, and the run quietly ends far worse than chance.",
    ],
    limits: [
      "In this lab: the report reads training loss on the text the model trains on. There is no held-out loss, so it cannot separate a good fit from memorizing.",
      "In general: a large run's report also needs evaluation scores, throughput and the cost of each restart. A rollback costs the steps since the checkpoint, so how often to save is itself a stability decision.",
    ],
  },

  "Number formats": {
    title: "Where fp16 runs out of room",
    summary:
      "A closed-form picture of three number formats on one log axis, fp32, bf16 and fp16, and of the narrow window of gradient values that survive the emulated fp16 cast at the current loss scale. With Fault on fp16 overflow it also outlines the range of this run's gradient norms against that window.",
    whatYouSee: [
      "Three bars on a log axis from 10⁻⁴⁶ to 10⁴⁰: the positive range of fp32, bf16 and fp16, with the lighter part on the left the subnormal range, where precision thins out.",
      "A green `survives` bar: the gradient values that come through the fp16 cast at the current `Loss scale`.",
      "A closer view from 10⁻²⁰ to 10¹⁰ that repeats the fp16 and `survives` bars, with an arrow for the division by the loss scale and tinted zones for values that round to zero or become infinity. With fp16 overflow selected, a black outline there marks the smallest to largest gradient norm of this run.",
      "A table of exponent and fraction bits, the largest value, and the smallest normal value of each format.",
      "A note with the survival window in numbers, such as 2.8 × 10⁻¹⁴ to 0.062 at 2²⁰.",
    ],
    howItWorks: [
      "Every number on this card is arithmetic on the format definitions, not a measurement. fp16 tops out at 65,504; fp32 and bf16 at about 3.4 × 10³⁸.",
      "The survival window is the fp16 range divided by the loss scale: an entry rounds to zero below 2⁻²⁵ divided by the scale, and becomes infinite from 65,520 divided by the scale up, the point where rounding passes 65,504.",
      "A gradient's norm is at least as large as any one of its entries, so a norm below the top of the window guarantees no overflow; a norm above it only makes one possible.",
    ],
    controls: [
      "`Loss scale` on Fault and fixes slides the window: each doubling moves both ends down by a factor of 2.",
      "Comparison worth running: at fp16 overflow, move `Loss scale` from 2¹⁷ to 2¹⁸. The top of the window drops from about 0.5 to 0.25, and the run goes from finishing to halting at step 184.",
    ],
    notice: [
      "bf16 reaches as far as fp32 in both directions, which is why runs in bf16 can usually drop loss scaling; it pays with 7 fraction bits instead of fp32's 23.",
      "This run never comes near the bottom of the window. Across the whole healthy run, no gradient entry is small enough to round to zero in fp16, even at a scale of 1.",
    ],
    limits: [
      "In this lab: fp16 is applied only to the gradient on its way to the update. The forward pass, the activations and the weights are 32-bit, so activation overflow, a common fp16 failure, cannot happen here.",
      "In general: modern accelerators train mostly in bf16, which avoids overflow but rounds more coarsely, so small updates to large weights can be lost; that is why an fp32 master copy of the weights is kept either way.",
    ],
  },
};

export default cardInfo;
