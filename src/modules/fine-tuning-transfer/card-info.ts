import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Transfer against training from scratch": {
    title: "Same budget, different starting weights",
    summary:
      "Two runs that both spend your fine-tune epochs on Recipe steps: one initialised from a model already trained 60 epochs on Harbor weather notes, one started from a zeroed table. Only the starting weights differ.",
    whatYouSee: [
      "A kicker `Pretrained 60 epochs on harbor weather notes` and a title that restates the current fine-tune epoch count on recipe steps, plus a badge with the shared step count.",
      "A chart of fine-tuning progress against target-corpus loss. Solid `started from pretrained` and dashed `started from zero`, y axis in nats per token from 1.70 to about 3.50.",
      "Three metrics: `Transfer, target loss`, `Scratch, target loss`, and either `Transfer is lower by` or `Transfer is higher by` that gap in nats.",
      "A footnote: both curves are full-corpus loss on Recipe steps at each saved checkpoint, same seed, same budget.",
    ],
    howItWorks: [
      "The base model is trained once, cached, on Harbor weather notes for 60 epochs at seed 1. Transfer continues from those weights at seed 2. Scratch starts from zeros at seed 2. Both use your `Fine-tune learning rate` and `Fine-tune epochs`.",
      "Each point is `tinyCrossEntropy` on the entire Recipe steps corpus at that checkpoint, not the batch loss of the run.",
      "At 0 fine-tune epochs the transfer curve is a single point: the pretrained model's loss on recipes, which is already better than uniform because the alphabets overlap. Scratch at 0 epochs is the uniform 3.401 of a zeroed table.",
      "Frozen rows, set on the freeze grid, apply only to the transfer run. Scratch still trains every row.",
    ],
    controls: [
      "`Fine-tune epochs` and `Fine-tune learning rate` retrain both runs.",
      "The freeze grid retrains transfer only, so a large freeze can make transfer lose to scratch — that is the comparison the third metric is willing to flip for.",
      "Comparison worth running: at 20 epochs and learning rate 0.60, transfer reads 2.123 against scratch's 2.247. Set Fine-tune epochs to 0 and the gap is 0.845 (2.556 against 3.401): what pretraining bought before any adaptation.",
    ],
    notice: [
      "The first transfer point is not 3.401. Harbor and recipes share characters and short patterns (`the`, `and`, spaces), so a harbor model is already a better-than-chance recipe model.",
      "Scratch can catch up given enough epochs on a corpus this small. Transfer's usual win is speed, not an unbeatable floor.",
      "If you freeze every block, transfer cannot move and the solid curve is flat; scratch will pass it.",
    ],
    limits: [
      "In this lab: both losses are measured, but each run is one seed on two eight-sentence corpora. The gap is real for this page; it is not a claim about ImageNet pretraining.",
      "In this lab: the pretrained model is not the one you built in `Train one from scratch`. It is a fresh 900-weight table this lab trains itself, once, for a fixed 60 epochs on Harbor weather notes at seed 1, and nothing you did in an earlier lab carries into it.",
      "In this lab: `started from zero` is literally a zeroed 30 × 30 table, which is also a uniform next-character model. A real `from scratch` run still uses random initialisation, not zeros.",
      "In general: transfer helps when the source and target share structure. It can hurt when they do not — negative transfer — and nothing on this card searches for that failure except you, by reading the third metric's sign.",
    ],
  },

  "Forgetting meter": {
    title: "What adaptation costs on the original data",
    summary:
      "The same transfer run scored on both corpora at every checkpoint: Recipe steps falling as it adapts, Harbor weather notes usually rising as it forgets. Two samples, same prompt and seed, sit under the chart so you can hear the style shift.",
    whatYouSee: [
      "A kicker `Two corpora, one set of weights` and a badge with the signed forgetting in nats, `source_after − source_before`.",
      "A chart of fine-tuning progress against loss in nats per character, with the y axis fitted to the two curves. Solid `Recipe steps (adapting)` and dashed `Harbor weather notes (forgetting)`.",
      "Two sample panels, `Before adaptation` and `After adaptation`, each 76 characters from prompt `the ` at temperature 0.7, seed 12.",
    ],
    howItWorks: [
      "Every point is full-corpus cross-entropy of the transfer checkpoint on that corpus. Nothing is smoothed or interpolated.",
      "Forgetting is a single number, `CE(harbor, after) − CE(harbor, before)`, printed on the badge and again on the freeze grid.",
      "Both samples use the same prompt, temperature, and seed, so a change in the text is a change in the weights. Before is always the frozen 60-epoch harbor model.",
      "At 0 fine-tune epochs the two curves meet their starting values and the samples match.",
    ],
    controls: [
      "This card has no controls of its own. Epochs, learning rate, and the freeze grid all reshape both curves.",
      "Comparison worth running: at 20 epochs read the badge, then freeze several early-alphabet blocks and read it again. Forgetting should shrink if those rows were carrying harbor-specific predictions — and target adaptation may shrink with it.",
    ],
    notice: [
      "At very small budgets the harbor curve can dip by well under 0.001 nats before it rises; at the defaults it rises from the first checkpoint. Forgetting is the harbor curve's net rise, not a requirement that it rise on every step.",
      "A 900-weight table only learns which letter follows which, so the after sample is mostly letter fragments, not recipe words, and one draw is not a metric. The chart's loss is what measures the shift.",
      "A forgetting badge near 0 with a large target gain is the pleasant case. A large badge with a small target gain is the one the freeze grid exists to mitigate.",
    ],
    limits: [
      "In this lab: forgetting is training-set cross-entropy on the original eight sentences, not a held-out harbor evaluation. The model can `forget` a memorised weather note and still look fluent. The Replay and held-out target card splits the original corpus into replayed and not-replayed sentences to show how much this one number hides.",
      "In this lab: the two corpora are short and stylistically opposite on purpose. That exaggerates the style shift in the samples relative to fine-tuning a model on a nearby domain.",
      "In general: catastrophic forgetting is measured on held-out source tasks, often as a suite, and is the reason people freeze layers, replay old data, or use adapters that leave the base weights untouched — the next module.",
    ],
  },

  "Fine-tuning controls": {
    title: "How hard to push the pretrained weights",
    summary:
      "The adaptation budget and the step size, plus the three losses that tell you whether the push was worth it: harbor before, harbor after, and how much recipe loss you gained.",
    whatYouSee: [
      "`Fine-tune epochs` from `no adaptation` at 0 to 60 passes, and `Fine-tune learning rate` from 0.05 to 2.00.",
      "Three metrics: `Source before`, `Source after` (loss tone once forgetting exceeds 0.05 nats), and `Target gained` as `CE(recipes, before) − CE(recipes, after)`.",
    ],
    howItWorks: [
      "The Replay and held-out target card reads these same two sliders for its own fine-tune. Epochs at 0 short-circuits both trainers: transfer keeps the pretrained weights, scratch stays at zeros, and `Target gained` is 0.",
      "Both runs use this learning rate with a constant schedule, batch 16, seed 2. The pretrained base used the trainer default of 0.6 for its 60 epochs and is not affected by this slider.",
      "`Source before` is a constant for the page: the cached 60-epoch harbor model's loss on harbor. Only `Source after` and `Target gained` move.",
    ],
    controls: [
      "`Fine-tune epochs` is the honest first knob: 0 is the control, 20 is the default, 60 is enough for scratch to close a lot of the gap on this corpus.",
      "`Fine-tune learning rate` at 2.00 will move harbor numbers quickly and usually for the worse.",
      "Comparison worth running: hold epochs at 20 and drop the learning rate to 0.05, then raise it to 1.50, reading `Source after` and `Target gained` as a pair each time.",
    ],
    notice: [
      "`Target gained` stays positive across this page's ranges: even 5 epochs at learning rate 2.00 gains about 0.40 nats. What large rates buy is speed, and they spend it on harbor.",
      "`Source after` lighting up as a loss tone at +0.05 nats is a display threshold, not a scientific cutoff. The forgetting meter badge is the same quantity without that threshold.",
      "These three numbers describe the transfer run only. Scratch's losses live on the comparison chart.",
    ],
    limits: [
      "In this lab: the base model is one cached run at seed 1. You cannot change pretraining epochs or the source corpus.",
      "In this lab: `Target gained` is training-set improvement on eight recipe sentences. It will look large relative to any real fine-tune gain; the Replay and held-out target card holds some recipe sentences out to show how much.",
      "In general: the adaptation budget is chosen against a held-out target metric and a held-out source metric at the same time. One slider that only reports training loss cannot make that trade for you.",
    ],
  },

  "Layer freezing": {
    title: "Choose which rows may move",
    summary:
      "A freeze grid over the 30-character vocabulary, in blocks of six. Frozen rows still have gradients computed, then discarded, so they keep their harbor predictions while the unfrozen rows adapt to recipes.",
    whatYouSee: [
      "Five buttons, each labelled with the six vocabulary characters it covers (`␣abcde`, `fghijk`, …) and a status `trainable` or `frozen`.",
      "Three metrics: `Frozen rows` out of 30, `Trainable weights` as `(30 − frozen rows) × 30`, and `Forgetting` with the same +0.05 nats colour threshold as the controls card.",
      "A row strip with one bar per context character: the length of that row's change, ‖W_after − W_before‖, scaled to the largest. Frozen rows are hatched; trainable rows that never moved carry a dot.",
      "A note stating that this model has no layers, so the grid freezes context rows of the bigram table rather than transformer blocks.",
    ],
    howItWorks: [
      "Each button toggles a block id in `frozen`. Those row indices are passed to the transfer trainer as `trainableRows`, which computes the gradient for a frozen row and then skips its update.",
      "Scratch ignores the grid. Only transfer is affected, which is why freezing every block makes the comparison chart's solid curve go flat.",
      "Forgetting on this card is the same `source_after − source_before` as the badge on the forgetting meter.",
      "Thirty symbols split into five blocks of exactly six, so each frozen block removes 180 trainable weights.",
    ],
    controls: [
      "Click a block to freeze or unfreeze it. The pressed/frozen state is visible as `frozen` versus `trainable`.",
      "Comparison worth running: freeze nothing at 20 epochs and read Forgetting, then freeze the block that contains space and `e` (the first one) and read it again. Those rows are the most trafficked contexts in English-like text.",
    ],
    notice: [
      "Freezing a row means `after \"x\"` keeps its harbor next-character distribution forever. If recipes need a different continuation after `x`, the model cannot learn it.",
      "In the row strip, `j`, `q`, `x`, `z`, `<` and `>` never move even when trainable: none of them is ever followed by a character in Recipe steps, so their rows get no gradient.",
      "The mechanism, a masked update, is real; the grouping by letters instead of layers is the toy.",
    ],
    limits: [
      "In this lab: there are no layers. Freezing `␣abcde` is not analogous to freezing layer 0 of a transformer except in the narrow sense that some parameters stop moving.",
      "In this lab: frozen rows are still stored, still used at inference, and still have their gradients computed. There is no memory or compute saving, only a training-time mask.",
      "In general: people freeze early layers because those layers tend to compute features that transfer, and they leave late layers trainable because those layers tend to be task-specific. A character-row grid cannot show that structure; LoRA is the version that leaves every base weight frozen on purpose.",
    ],
  },

  "Replay and held-out target": {
    title: "What replay protects, and what held-out sentences reveal",
    summary:
      "A second fine-tune with two additions the first cards leave out: some original sentences mixed back into the training text, and some target sentences kept out of it. It shows whether a fine-tune generalized, and what replay protects and what it does not.",
    whatYouSee: [
      "Replay share, a slider from no replay to 100% of the original corpus in notches of 12.5%. Each notch is exactly one of Harbor weather notes' eight sentences, and the readout names the share and the sentence count. Held-out target sentences, a slider from 0 to 4, whose readout says how many of the eight recipe sentences are held out and how many trained on.",
      "A chart against fine-tuning progress of three losses in nats per character: the recipe sentences it trained on (solid), the recipe sentences held out (dotted), and Harbor weather notes (dashed, forgetting). With nothing held out the dotted line is absent.",
      "A table with a No replay column and, once Replay share is above 0, a With replay column. Its rows are Trained-on target, Held-out target, Forgetting on all original sentences, Forgetting on the replayed sentences, Forgetting on the sentences not replayed, and Optimizer steps.",
      "A kicker counting trained-on, held-out, and replayed sentences, and a badge with the optimizer steps of the run.",
    ],
    howItWorks: [
      "The recipe corpus is split by sentence: the first 8 minus k sentences enter the fine-tuning text and the last k are never trained on. Replay appends the first n sentences of Harbor weather notes, where n is the share times 8, rounded. The optimizer shuffles the pairs of the combined text as usual.",
      "The run starts from the same cached 60-epoch Harbor model, uses your Fine-tune epochs and Fine-tune learning rate, and runs at seed 2, batch 16, constant schedule, with every row free to move, so Layer freezing does not apply here. With nothing held out and no replay it reproduces the first card's transfer run exactly.",
      "Trained-on target is cross-entropy on the training sentences only. Held-out target is the same on the held-out ones. Each forgetting row is the loss on those Harbor sentences after the fine-tune minus the base model's loss on the same sentences.",
      "The No replay column is a second fine-tune with the same epochs, rate, and held-out split, scored on the same two groups of Harbor sentences, so the columns compare like with like.",
      "Replayed sentences are extra training pairs, so each epoch is longer and the step count rises with the share. The fine-tune is given epochs, not a fixed number of steps.",
    ],
    controls: [
      "`Replay share` sets how much of the original corpus is mixed back in. `Held-out target sentences` sets how many recipe sentences are kept out. `Fine-tune epochs` and `Fine-tune learning rate` on the controls card set the budget this card shares.",
      "Comparison worth running: at the defaults, 3 held out and no replay at 20 epochs, read Trained-on target (2.162) against Held-out target (2.424). Raise Fine-tune epochs to 60 and the gap widens from 0.262 to 0.472.",
      "Comparison worth running: at 20 epochs raise Replay share to 50% and read the two Forgetting rows, then to 100%.",
    ],
    notice: [
      "The gap between the first two rows is the part of the fit that does not carry to new sentences. It is 0.157 at 10 epochs, 0.262 at 20, and 0.472 at 60: trained-on loss keeps falling while held-out loss flattens near 2.40.",
      "Replay protects what it replays. At 50% the four replayed sentences end 0.038 nats better than they started, against +0.119 forgetting for those same sentences without replay. The four sentences not replayed forget +0.129 with replay and +0.128 without: no protection at all.",
      "The headline Forgetting falls from +0.123 to +0.042 at 50% and to +0.001 at 100% largely because the replayed sentences are part of what it scores. Only at 100% is there nothing left to compare.",
      "Replay costs fit and steps: Trained-on target rises from 2.162 to 2.232 at 50%, and the run takes 640 steps instead of 320. Held-out target barely moves, 2.424 to 2.417.",
      "With 4 sentences held out and 60 epochs the held-out line bottoms out at 2.541 and ends at 2.570 while trained-on loss falls to 1.865: small-data overfitting, visible only because those sentences were kept out.",
    ],
    limits: [
      "In this lab: each corpus is eight sentences, so a held-out set is at most four sentences and a few hundred characters, and every number is one seed. Differences under about 0.05 nats are unresolved. The replayed sentences are simply the first ones of the original corpus.",
      "In this lab: forgetting is still scored on text the model pretrained on, and the replayed sentences are part of it. A bigram can only memorize which characters follow which, so replay of a few sentences protects those sentences and very little else.",
      "In general: real replay samples a mixture of earlier data, sized so the old behaviour stays in the loss, and forgetting is measured on held-out old tasks and old behaviours, not on the replayed examples. Replay also costs compute, because it adds training data to every epoch.",
    ],
  },

  "Should you fine-tune? Rule of thumb": {
    title: "Four yes/no questions, one recommended route",
    summary:
      "A decision table someone wrote by hand. Your four answers pick one of five routes (prompt first, prompt with examples, retrieval, retrieval plus fine-tuning, fine-tuning) and print the reason. It is a rule of thumb, not a measurement: nothing on this card was trained or scored.",
    whatYouSee: [
      "A badge `rule N of 5 matched` and four fieldsets, each a question with `Yes` and `No` buttons: does the knowledge change often, must answers cite sources, do you need a consistent format, style or behaviour, and do you have hundreds of good examples and an evaluation set.",
      "A `Recommended route` box with the route name, the sentence that explains why this rule fired, and a `Next` line saying what to do first.",
      "Four printed lines, one per question, each starting with the question's short name and your answer, so you can read what each answer contributed.",
      "The whole rule table: five rows in order, the first that matches wins, and the matched row ends with `(matched)` and carries an outline as well as the text.",
    ],
    howItWorks: [
      "`recommend()` tests the rows in order. Row 1: facts change or need citations, and a consistent format is needed, and you have examples plus an evaluation set. Row 2: facts change or need citations. Row 3: a consistent format is needed and you have the examples and the evaluation set. Row 4: a consistent format is needed. Row 5: none of those.",
      "Because the first match wins, retrieval is part of the route whenever facts change or need citations, whatever the other answers say. Fine-tuning is only ever recommended when a consistent behaviour is wanted and you could both train it and check it.",
      "All 16 combinations of answers reach exactly one row, so every setting prints a route. The default is four `No` answers, which lands on `Prompt first`.",
      "The answers are stored as four true-or-false values in the lab's state, so a reload or a shared link restores them.",
    ],
    controls: [
      "Each `Yes`/`No` pair is a button group. Use Tab to reach it and Enter or Space to press a button; the pressed button is bold on a raised background, and the printed lines spell each answer out in words, so colour is never the only signal.",
      "Comparison worth running: set all four to `No`, then flip only `Do you need a consistent format, style or behaviour?` and note the route. Then flip `Do you have hundreds of good examples and an evaluation set?` as well and see what the second answer changes.",
      "Comparison worth running: with a format need and examples both `Yes`, flip `Does the knowledge the answers depend on change often?` and watch the route add retrieval.",
    ],
    notice: [
      "Having examples does not recommend fine-tuning by itself. The data answer only matters once a format or behaviour is needed, which is the point of asking for it second.",
      "A `Yes` to changing facts or to citations gives the same route as both together. The two questions ask for different things, but retrieval answers both.",
      "The route that asks for a consistent format without data is `Prompt with examples, then collect data`: the examples you would train on are the same ones you would put in the prompt.",
    ],
    limits: [
      "In this lab: the mapping from answers to a route is a hand-written rule of thumb. It was not fitted to data or measured on any model, and it is the only card in this lab whose output is not computed from training.",
      "In this lab: only four yes/no questions enter the rule, and each is answered in two words. Real situations come in degrees (how often facts change, how many examples) that the rule flattens into a yes or a no.",
      "In general: real decisions also weigh cost, latency, privacy and risk. A fine-tuned small model can cut per-request cost or latency at scale, a private corpus may rule out some options, and evaluation and maintenance cost money in every route. The rule leaves all of that to you.",
    ],
  },
};

export default cardInfo;
