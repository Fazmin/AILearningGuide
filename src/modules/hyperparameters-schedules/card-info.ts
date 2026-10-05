import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Learning-rate schedule shapes": {
    title: "Step size as a function of progress",
    summary:
      "The learning rate each of the four schedules would use at every point in the run, given the peak rate and warmup fraction you set, with the selected schedule's phases shaded and one epoch inspected. This chart is the formula, not the training result — the loss chart is the result.",
    whatYouSee: [
      "A kicker `Schedule editor` and a badge repeating the current peak learning rate.",
      "A `Schedule` control with four options: Constant, Cosine decay, Warmup + decay, One cycle. The selected schedule is drawn solid and thick in the module colour; the other three are thin, grey, and dashed or dotted.",
      "A chart whose x axis is epoch 0 to 25 (progress through the run) and whose y axis is learning rate from 0 to just above the peak. Shaded bands mark the selected schedule's phases: a warmup band with its length in steps, then a decay band labelled with where it ends, such as `linear decay to 5% of peak (0.030)`.",
      "A dashed guide and dot at the `Inspect epoch` position, labelled with the rate at that step, and a note restating the schedule and its warmup length in steps.",
    ],
    howItWorks: [
      "Every curve is `peak × scheduleFactor(schedule, progress, warmup)`. Constant returns 1, so it is a flat line at the peak no matter what warmup is.",
      "Cosine decay is `0.03 + 0.97 × ½(1 + cos(π · progress))`. It ignores warmup and finishes at 3% of the peak.",
      "Warmup + decay ramps from near zero to the peak over the warmup fraction, then decays linearly to 5% of the peak. One cycle warms from 10% of peak to the peak, then cosines down to 2%.",
      "Warmup fraction is clamped inside the formula to [0.01, 0.90], so the slider's 2%–60% range is always respected.",
    ],
    controls: [
      "`Schedule` chooses which curve is emphasised and which run the metrics and gradient trace quote.",
      "`Inspect epoch` moves the guide here and on the loss chart together; it changes no training.",
      "`Peak learning rate` and `Warmup fraction` on `Optimizer controls` reshape every curve. Warmup only moves the two schedules that use it.",
      "Comparison worth running: select Warmup + decay, set Inspect epoch to 1, 3 and 4, and read the rate climbing through the shaded warmup (120 steps at the defaults). Then raise the peak to 30 — the shapes scale; they do not change.",
    ],
    notice: [
      "Constant and Cosine decay do not use warmup. Dragging `Warmup fraction` while those are selected will look like nothing happened on this chart, which is correct.",
      "One cycle ends below its start. That is the point of the schedule, not an undershoot bug.",
      "The selected schedule's colour is also its dash: solid only for the current choice, so the emphasis never depends on colour alone.",
    ],
    limits: [
      "In this lab: these four closed-form multipliers are the entire schedule zoo. There is no cosine-with-restarts, no ReduceLROnPlateau, and no per-layer rate.",
      "In this lab: the x axis is progress through planned steps, labelled in epochs, not wall time or loss. A run that diverges still has a well-defined schedule curve.",
      "In this lab: warmup-stable-decay (WSD), the schedule many recent runs use, is not one of the four. It would sit between Constant and Warmup + decay: flat at the peak for most of the run, then a short decay at the end.",
      "In general: a schedule is a bet that the right step size changes as the parameters move. The bet can be wrong for a given peak, which is why the loss card trains all four instead of trusting this picture.",
    ],
  },

  "Loss for all four schedules": {
    title: "Same data, same seed, same peak — four outcomes",
    summary:
      "Four complete training runs of 25 epochs on Harbor weather notes, one per schedule, each scored on the whole corpus at every epoch boundary, plus a dotted uniform baseline. A curve that climbs through that line is a run doing worse than a model that learned nothing. A Seeds switch repeats the same recipe over five shuffle seeds and shades the spread.",
    whatYouSee: [
      "A kicker with the epoch and step count, and a badge with the batch size.",
      "A chart of epoch against whole-corpus loss in nats per token. Four series — Constant solid, Cosine decay dashed, Warmup + decay dotted, One cycle dash-dot, each in its own colour — with the selected schedule drawn thicker, and a dotted `knows nothing` line at ln 30 = 3.401.",
      "A guide at the `Inspect epoch` position with a dot on every curve. The legend prints each schedule's loss at that epoch and at the end.",
      "Three metrics: the selected schedule's final loss (or `diverged`), `Best schedule here`, and that winner's final loss.",
      "A `Seeds` control: `Seed 1 only` or `Five seeds`. With five, each schedule gets a shaded band from its lowest to its highest loss across the seeds at each epoch, the legend adds `seeds low–high` for the final loss, and a table lists the final loss of every seed with a `Spread` column (highest minus lowest).",
      "Under the table, a callout that compares the two best schedules by their five-seed mean, and the two closest, each with the wider of their spreads, and says whether the gap is inside or outside the band.",
    ],
    howItWorks: [
      "Each run is minibatch SGD on the character bigram, 25 epochs, seed 1, sharing the current peak rate, warmup, clip, weight decay, and batch size. Only the schedule multiplier differs.",
      "The trainer saves the weights at every epoch boundary and each point is `tinyCrossEntropy` of those weights on all 498 training pairs, so the last point equals the final-loss metric. Noisy minibatch losses are not plotted.",
      "A run is treated as diverged when `finalLoss` is not finite. `Best schedule here` ignores those and picks the lowest finite final loss. With five seeds it picks the lowest mean over the five seeds, and the third metric becomes `Best mean of five seeds`.",
      "Five seeds means seeds 1 to 5. Seed 1 is the line drawn and the first column of the table. The weights start at zero, so the seed changes only the order the 498 training pairs are shuffled in. The band at each epoch is the lowest and highest of the five whole-corpus losses at that epoch boundary.",
      "The callout takes the gap between two schedules' five-seed means and compares it with the larger of their two spreads (highest minus lowest final loss). A gap above that spread is `outside` the band, and a gap below it is `inside`.",
      "The y range grows to fit the worst curve but stops at 7.5 nats; a value above that is pinned to the top edge while the legend prints its true number.",
    ],
    controls: [
      "`Schedule` on the shape card chooses which final-loss metric is highlighted and which band is drawn heavy; it does not hide the other three curves.",
      "`Seeds` switches between one run per schedule and five. Five seeds retrain twenty runs, so the sliders respond more slowly while it is on.",
      "Every slider on `Optimizer controls` retrains all four runs, or all twenty with five seeds.",
      "Comparison worth running: leave the defaults, then set `Peak learning rate` to 30. Constant climbs toward the dotted line (3.020 at the end) while the decaying schedules finish near 1.80. Then switch `Seeds` to `Five seeds`: Constant's final loss runs from 2.395 to 4.673 across the seeds, and seed 1's 3.020 is one point inside that range.",
    ],
    notice: [
      "At the default peak of 0.6 the three decaying curves almost coincide and Constant sits below them, because at this rate the decayed steps are simply too small. The card exists for the peak where that order flips.",
      "`Best schedule here` is a statement about this seed, this corpus, and this peak. It will flip as you move the sliders.",
      "A decaying schedule can finish higher than Constant at a low peak, because it spent the end of the run taking tiny steps. That is not instability; it is leftover budget.",
      "At the default peak the three decaying schedules end within 0.004 nats of each other, and each moves by about 0.004 between seeds, so their order is inside the band. Constant is 0.175 ahead of the best of them, about fifty times the band, so that ordering is real.",
      "The band widens sharply for Constant as the peak rises: 0.002 wide at 0.6, 0.128 at 6, 0.490 at 12, and 2.278 at 30. The decaying schedules stay under 0.011 wide at every peak checked. At a peak of 30, two of the five seeds already end worse than chance.",
    ],
    limits: [
      "In this lab: four runs, one corpus, 25 epochs, and a character bigram; one seed by default and five with the switch. The ranking is real for this page and not a schedule recommendation for a transformer.",
      "In this lab: five seeds is a small sample, and the band is the range of those five, not a confidence interval. It shows how much the shuffle order alone moves a result, and nothing about the corpus, the model size, or the learning-rate grid.",
      "In this lab: `diverged` means a non-finite final loss. A run that is merely worse than chance still prints a number and still counts in `Worse than chance` on the stability card.",
      "In general: schedule comparisons need more than one seed and a held-out curve. A schedule that wins on training loss can still overfit, and a schedule that looks unstable on one seed can be fine on the next. Real runs often cannot afford five seeds, which is why differences smaller than the seed noise are treated as unproven.",
    ],
  },

  "Optimizer controls": {
    title: "Four dials, one of them dangerous",
    summary:
      "Five dials: the peak learning rate, the warmup fraction, a gradient-norm clip, weight decay, and batch size. Every change retrains all four schedules. The peak is the dial that can make a run worse than chance.",
    whatYouSee: [
      "`Peak learning rate` from 0.10 to 60.00 and `Warmup fraction` from 2% to 60% of the run.",
      "`Gradient clip` from `off` to `norm ≤ 1.50`, `Weight decay` from `off` to 0.080, and `Batch size` from 2 to 64 pairs.",
    ],
    howItWorks: [
      "The step applied to a weight is `peak × scheduleFactor × (clipped gradient + weightDecay × weight)`. Clip 0 and decay 0 recover plain SGD.",
      "If a batch's gradient norm exceeds the clip threshold, the whole gradient is scaled so its norm equals the threshold. A threshold of 0 disables clipping.",
      "Weight decay adds `λw` inside the update, after clipping, so it is never clipped. For plain SGD that is the same step as an L2 penalty and as AdamW-style decoupled decay. It is applied only to rows that appeared in the batch.",
      "Batch size changes how many pairs a step averages and therefore how many steps 25 epochs contain. It does not change the peak rate.",
    ],
    controls: [
      "`Peak learning rate` is the dangerous one. Past about 6 the constant schedule's final loss starts rising, and at batch 16 it crosses the dotted baseline between 35 and 38.",
      "`Warmup fraction` only reshapes Warmup + decay and One cycle. It is ignored by Constant and Cosine decay.",
      "`Gradient clip` cannot bind above ~1.41 on this model — see the stability note — so the top of the slider is there to let you watch it never bind.",
      "`Weight decay` at 0.08 is a strong pull on a 900-weight table; small values are the useful ones.",
      "`Batch size` at 2 makes the noisiest gradients and is where a low clip helps most.",
      "Comparison worth running: set Peak learning rate to 40 with clip off, confirm Constant is worse than chance (3.687), then set `Gradient clip` to 0.40 and watch it come back below the line (3.186).",
    ],
    notice: [
      "Raising the peak scales every schedule's steps, including the decaying ones. Cosine at peak 30 still starts at 30.",
      "A clip that never binds is reported as such on the stability card. If `Peak gradient norm` sits at 1.2 and clip is 1.5, the slider is doing nothing.",
      "Weight decay and a decaying schedule both shrink effective steps late in training, for different reasons. Do not move them at the same time if you want to know which one did the work.",
    ],
    limits: [
      "In this lab: the optimizer is plain SGD plus an optional global-norm clip and weight decay. There is no Adam, no momentum, and no per-parameter state, so the difference between L2 regularization and AdamW's decoupled decay cannot appear here.",
      "In this lab: peak 60 on a character bigram is a teaching explosion, not a realistic transformer rate. The interesting region on this page is roughly 0.5–40.",
      "In general: the same names — peak rate, warmup, clip, decay, batch — appear in every modern recipe, but Adam's effective step size is not the number you typed, and clip thresholds are chosen against real activation norms, not against √2.",
    ],
  },

  "Stability readout": {
    title: "Which runs are still learning",
    summary:
      "The batch-gradient norm at every step of the selected run drawn against the clip threshold, a peak gradient-norm reading across all four runs, whether the clip is binding, and a count of schedules that finished worse than a uniform guess.",
    whatYouSee: [
      "A trace with one dot per optimizer step: x is the step, y the norm of that step's batch gradient before clipping, from 0 to 1.5. A dotted line marks the √2 bound; with `Gradient clip` on, a dashed line marks the threshold and steps above it are drawn darker and larger. A caption counts them.",
      "`Peak gradient norm` — the largest batch-gradient L2 seen in any of the four runs.",
      "`Clip` reading `off`, `N · binding`, or `N · never binds`, depending on whether the threshold is set and whether any recorded norm exceeded it.",
      "`Worse than chance` as `K / 4`, counting runs whose final loss is non-finite or greater than `ln 30 ≈ 3.401`.",
      "A callout that either names the unstable schedules, notes that decaying schedules are pulling ahead, or says all four are stable.",
      "A standing note: row gradients for softmax cross-entropy cannot exceed √2 ≈ 1.41, so a clip above that never binds on this model.",
    ],
    howItWorks: [
      "A run is unstable when `!Number.isFinite(finalLoss)` or `finalLoss > ln 30`. That is a worse-than-chance test, not a test that only looks for a non-numeric loss.",
      "Clip is binding when `clipNorm > 0` and `peakGradient > clipNorm`. The peak is taken across all four runs, so one noisy schedule can make the clip look binding for the page.",
      "The trace logs every step of the selected run: its norm is measured before clipping, so dots above the threshold are exactly the steps the clip rescaled.",
      "The callout's second sentence is the intended lesson: decaying the rate shrinks steps over time; clipping caps how far any single step can travel. This model overshoots rather than explodes.",
    ],
    controls: [
      "The card has no controls; it reports the four runs the sliders defined and traces the one `Schedule` selects.",
      "Comparison worth running: at peak 0.6, `Worse than chance` should be `0 / 4` and the callout should be quiet. Raise the peak until Constant appears by name, then decide whether you want to fix it with decay (already on the other three curves) or with a clip.",
    ],
    notice: [
      "At the default peak of 0.6 the interesting sentence is the quiet one: all four are stable. The card is working when it is boring.",
      "Past a peak of about 6 the callout switches to `Every run survives, but the decaying schedules are pulling ahead` even if nobody has crossed the baseline yet.",
      "At batch 16 the trace has a band near 0.2 and isolated dots near 0.8. The outliers are each epoch's last batch, which holds only 2 of the 498 pairs (498 = 31 × 16 + 2), so its gradient averages over far fewer pairs. That leftover batch sets `Peak gradient norm`.",
      "Clipping helps most at small batch sizes because a single pair's gradient is the noisiest estimate this trainer will make.",
    ],
    limits: [
      "In this lab: gradients are bounded by √2 per row, so the trained model on this page cannot show a genuine exploding gradient. The note says so on purpose. The `Gradient through depth` card shows one in closed form, as a product of per-layer gains, not as a trained network.",
      "In this lab: `Worse than chance` is a training-loss test on Harbor weather notes. A run can pass it and still be a bad model.",
      "In general: stability is not a single number. Production training watches grad-norm histograms, loss spikes, and whether tokens per second suddenly collapse — all of which can happen while a mean loss still looks fine.",
    ],
  },

  "Gradient through depth": {
    title: "A product of per-layer gains, and the clip that caps it",
    summary:
      "A closed-form model of a gradient passed back through a stack of layers. Every layer multiplies the gradient by the same gain, so with a gain above 1 the early layers see an exploding gradient, with a gain below 1 a vanishing one, and a global-norm clip can shorten the first but not lengthen the second. It is arithmetic, not a trained network.",
    whatYouSee: [
      "Three sliders: `Layers` from 2 to 48, `Gain per layer` from 0.70 to 1.40, and `Depth clip` from `off` to `total ≤ 10.0`.",
      "A chart whose x axis is the layer, from layer 1 (the input side) on the left to the output layer on the right, and whose y axis is gradient length on a log scale from 1e-8 to 1e8. A solid line is the gradient length at each layer before clipping, and with a clip set a dashed line shows it after. A dotted line marks the clip threshold, which applies to the total.",
      "A table of the same numbers: the output layer, the middle layer, the input layer, and the whole gradient, before and after the clip.",
      "Three metrics: `Input ÷ output gradient`, the length at layer 1 over the length at the output; `Whole gradient length`; and `Clip`, which reads `off`, `never binds`, or `binding` with the factor it scales by. A callout names the case: exploded, clipped, vanished, or ordinary.",
    ],
    howItWorks: [
      "The output layer's gradient has length 1. The layer `k` steps back has length `gain ^ k`, because each layer multiplies the gradient that reaches it by its gain. That is the shape of a product of Jacobians when every layer's Jacobian has the same size.",
      "The whole gradient has length `√(Σ length²)` over all layers. A clip at `τ` multiplies every layer's gradient by `min(1, τ ÷ whole length)`, so the direction across layers is kept and only the length is capped.",
      "The factor is never above 1. A clip therefore cannot lengthen a gradient, which is why it helps an exploding case and is silent on, or slightly harmful to, a vanishing one.",
    ],
    controls: [
      "`Layers` and `Gain per layer` set how far the gradient has to travel and how much each layer changes it. `Depth clip` caps the whole gradient; it is separate from `Gradient clip` on the optimizer card, whose range stops at the bigram's own bound of √2.",
      "Comparison worth running: leave 24 layers at a gain of 1.20, where the input layer sees 66 times the output layer and the whole gradient is about 120. Set `Depth clip` to 1.0 and the whole gradient becomes exactly 1.0, every layer scaled by 0.0083. Then set `Gain per layer` to 0.80 at 40 layers and raise the clip: nothing it does helps the input layer.",
    ],
    notice: [
      "On a log axis a gain gives a straight line, steeper as the gain moves from 1. At 40 layers a gain of 1.2 puts the input layer 1,225 times above the output layer; a gain of 0.8 puts it 0.000166 of the way, under a thousandth.",
      "With the clip on, the dashed line is the solid line shifted down by one constant factor, so the ratio between any two layers does not change. The clip fixes the size of the step and leaves its direction alone.",
      "At a gain of exactly 1.00 every layer has length 1, and the whole gradient is the square root of the layer count.",
    ],
    limits: [
      "In this lab: every layer has the same gain and the output gradient is set to 1. There are no weights, no activations, no data, and nothing is trained. A real network's per-layer factor varies with the weights, the activation function, and the input.",
      "In this lab: the log axis stops at 1e-8 and 1e8, so a length beyond that is pinned to the edge. Every length the sliders can produce stays finite.",
      "In general: real transformers keep the gradient in range with residual connections, normalization, and careful initialization, and still rely on a clip for occasional spikes. Both GPT-3 and Llama 2 report clipping the global gradient norm at 1.0 (Brown et al., 2020; Touvron et al., 2023).",
    ],
  },
};

export default cardInfo;
