import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Build a model in four choices": {
    title: "The decisions that define the run",
    summary:
      "The stage strip for the whole wizard plus the first real decision: which corpus the model reads. Choosing one swaps the training text and retrains both models on the page immediately, so nothing here is staged or deferred.",
    whatYouSee: [
      "The kicker reads `Choice N of 4` and the badge counts the factored model's trained weights, which is two times rank times 30 and is set by Model size on the controls card, not here.",
      "A numbered stage strip: Corpus (what it reads), Size (how many weights), Recipe (how it trains), Model card (what you built). The current stage is marked, earlier ones are marked done.",
      "Three corpus buttons — Harbor weather notes, Recipe steps, Proverbs — each with its one-line description and its length in characters after encoding. The pressed button is the corpus in use.",
      "The fourth stage, Model card, is a report rather than a choice: the run is fully defined by the corpus, the size, and the recipe.",
    ],
    howItWorks: [
      "Each button's character count is the corpus after `encodeTinyText`: lowercased, mapped into the 30-symbol vocabulary ` abcdefghijklmnopqrstuvwxyz.<>`, anything else replaced by a space, and runs of spaces collapsed. It is the real training length, not the raw string length.",
      "The chosen corpus is split at full stops and the last 25% of its sentences (at least one) are held out. Only the remainder is ever used for a gradient step.",
      "The model is a character bigram: one row of 30 logits per preceding character, and a softmax over that row is the next-character distribution. Training is minibatch SGD on cross-entropy, re-run synchronously on every change.",
      "Clicking a corpus also advances the stage indicator to at least Size and narrates the corpus name with its character count.",
      "The stage strip is an indicator only. Clicking a stage sets the index and nothing else; every control stays live in every stage, so no choice is gated behind another.",
    ],
    controls: [
      "The three corpus buttons: Harbor weather notes is repetitive descriptive prose, Recipe steps is imperative instructions, Proverbs is short unrelated sentences.",
      "The stage strip, which moves the `Choice N of 4` indicator and nothing else.",
      "Comparison worth running: keep Model size and Epochs fixed and switch between Harbor weather notes and Proverbs, then read Held-out loss on the model card. Same recipe, different fit.",
    ],
    notice: [
      "The character counts are the point: a few hundred characters is the entire evidence base for the run, which is why the corpus choice moves every other panel.",
      "Each button states a claim — Proverbs says it is hard to fit — and the model card lets you check it instead of taking it on trust.",
      "The badge changes when you move Model size, because the trained-weight count belongs to the factored model, not to the corpus.",
    ],
    limits: [
      "In this lab: the three corpora and their descriptions are hard-coded strings in the module SDK. You cannot paste your own text, and the held-out split is deterministic — the final sentences of the document, not a random sample.",
      "In this lab: the badge counts only the factored model's weights. The 900-weight full table trained alongside it for comparison is not in that number.",
      "In general: choosing data really is the first decision, but out there it includes provenance, licence, deduplication, and filtering. Only a small synthetic version of the cleaning step exists here, on the Training recipe card; a corpus is a handful of sentences with no licence to check.",
    ],
  },

  "Training progress and samples": {
    title: "Training and held-out loss, and what the model writes as it falls",
    summary:
      "Two real training runs on the same split — the rank-r factored model you are configuring and the full 900-weight table — each scored at every epoch on the training split and on the held-out sentences, with three text samples drawn from the factored model at the marked epochs.",
    whatYouSee: [
      "The x axis is epoch and the y axis is loss in nats per token, from just below the lowest curve up to 3.50.",
      "Four curves. Factored model: solid for the training split, dashed for held out, with the band between them shaded and the final gap printed. Full table: thin solid for training and thin dotted for held out. A dotted `knows nothing` line sits at ln 30 = 3.401.",
      "Circled markers A, B, and C on the solid curve at epoch 0, the halfway epoch, and the last epoch. The three sample boxes below carry the same letters, the epoch and step, and the training loss there.",
      "A legend printing all four final losses. They match the model card. If a run diverges, its curves stop at the last epoch whose loss was a number and its legend entries read `diverged`.",
    ],
    howItWorks: [
      "Both runs get the same training text and the same recipe from the Training recipe card, with seed 9. At the defaults that is batch 16, learning rate 0.5, a constant step size, and no clipping. The only difference between the runs is the parameterization: a 30 by r matrix times an r by 30 matrix against a single 30 by 30 table.",
      "Each run saves its weights at every epoch boundary. A training-curve point is the cross-entropy of those weights over the whole training split; a held-out point is the same formula on the held-out sentences. No noisy batch averages are plotted.",
      "One example moves 30 weights of the full table (its own row) but 31r weights of the factored model: its row of the input factor plus the whole r × 30 output factor that every character shares. That sharing, not capacity, is why the smaller model can sit lower.",
      "Each sample is drawn one character at a time from the softmax of the row for the previous character, at temperature 0.7 from the prompt `the ` with seed 31 for all three, so any difference between them comes from the weights alone.",
      "Because the model is a bigram, only the final character of the prompt conditions the first draw — `the ` conditions on the space. The reserved `<` and `>` are suppressed and the row renormalized, as a real decoder does with special tokens.",
    ],
    controls: [
      "This card has no controls of its own; the corpus buttons, Model size, Epochs, and the Training recipe card drive every curve and the samples.",
      "Comparison worth running: with the recipe at its defaults, choose Proverbs, set Model size to 12 and Epochs to 30, and watch the dashed held-out curve bottom out near epoch 18 (2.668) and climb to 2.818 while the solid curve keeps falling.",
    ],
    notice: [
      "The shaded band is the generalization gap at every epoch, not only at the end. On Harbor weather notes it stays thin; on Recipe steps and Proverbs it opens wide, because their held-out sentences share less with their training sentences.",
      "Samples move from letter soup to word-shaped fragments such as `the` and `and`, and stop there. One character of context cannot produce fluent text, and a sample illustrates the loss rather than measuring anything.",
      "Both training curves stall well above zero. The floor is the entropy of the next character given only the previous one, which is a property of the data.",
    ],
    limits: [
      "In this lab: every point on all four curves and every sample comes from real training — no fitted curve and no seeded noise anywhere. The held-out set is the last quarter of the document's sentences, roughly a hundred characters, so small wiggles in the dashed curves are noise.",
      "In this lab: the three samples come from the factored model and therefore follow Model size; the full table has curves but no samples.",
      "In general: real runs plot exactly this pair of curves and stop, or keep the best checkpoint, when the held-out curve turns up. Their held-out sets are large and drawn at random, so the curves are far smoother than these.",
    ],
  },

  "Size and epochs": {
    title: "Capacity first, then budget",
    summary:
      "The two dials that size the run once the corpus is fixed: how many weights the factored model gets, and how many passes it takes over the data. Both retrain everything as you drag them; the rest of the recipe lives on the Training recipe card.",
    whatYouSee: [
      "A Model size slider from 1 to 12 whose readout gives the rank and the resulting weight count, which is two times rank times 30.",
      "An Epochs slider from 1 to 30 whose readout counts passes over the training split.",
      "A callout naming which model currently fits better, with a paragraph explaining why: when the factored model wins, the shared output factor; when the full table wins, the factored model's missing capacity.",
      "A note restating the arithmetic — rank r keeps two times r times 30 weights against 900 — and warning that past rank 15 the factorization saves nothing at all.",
    ],
    howItWorks: [
      "Rank is the inner dimension of the factorization: each logit is the sum over r terms of left[row][j] times right[j][next], so the 900-entry table is stored as a 30 by r matrix times an r by 30 one, for two times r times 30 trained numbers.",
      "At rank 1 every logit is one number times another, so every context row is the same shaped distribution scaled up or down — sharper, flatter, or reversed when the row's number is negative, never any other ordering of characters. That is why rank 1 plateaus instead of training slowly.",
      "Epochs sets the step count: steps equal epochs times the number of batches per pass, which is the pair count divided by the batch size and rounded up. The batch size is 16 unless you change it on the Training recipe card, so Epochs is the budget dial and the recipe decides how each step is spent.",
      "Both losses quoted in the callout are full-split cross-entropy measured after each run finishes, recomputed on every change to either slider.",
      "Moving either slider also advances the wizard stage indicator on the first card — Model size to Size, Epochs to Recipe.",
    ],
    controls: [
      "`Model size` sets the rank and therefore the parameter count; it is the capacity dial.",
      "`Epochs` sets how many passes the optimizer takes; it is the budget dial.",
      "Comparison worth running: at Epochs 20, step Model size through 1, 4, 8, and 12 and read Train, Held out, and Gap on the model card each time. Then return Model size to 1 and push Epochs to 30 — thirty passes buy about 0.07 nats, because the missing thing is capacity, not steps.",
    ],
    notice: [
      "At the default recipe, Model size 8 and Epochs 20, the 480-weight factored model reports 2.039 nats on the training split while the 900-weight table reports 2.283. Fewer parameters, better fit, at this data scale and with this optimizer.",
      "The callout's headline and its explanation flip together. At rank 1 the full table is ahead and the paragraph blames capacity; from rank 4 at 20 epochs the factored model is ahead and the paragraph credits the shared output factor.",
      "Raising rank past 8 barely moves the training loss and widens the Gap on the model card (0.101 at rank 10, 0.128 at 12). That is the shape of enough capacity, then too much. Below 8 the gap is bumpy (0.095 at rank 6), so read the trend rather than single steps.",
      "Parameter count and capacity are separate quantities. The full table can express strictly more matrices than any rank-8 product and still loses on this data at this budget.",
    ],
    limits: [
      "In this lab: both numbers in the callout are measured on the training split, so `fitting better` means exactly that and says nothing about held-out behaviour, which lives on the Model card. Model size stops at 12 and Epochs at 30, so the rank-15 crossover the note mentions is out of reach.",
      "In this lab: the rest of the recipe lives on the Training recipe card, and every result is a single seed, 9. Weight decay is not offered, so budget here means epochs and the steps they imply.",
      "In general: the smaller-model-wins result depends on plain SGD and on a few hundred characters. Optimizers that scale each parameter's step separately reduce the update asymmetry, and at realistic data scale the usual finding runs the other way, which is why scaling laws pair parameter count with token count instead of maximizing either.",
    ],
  },

  "Training recipe": {
    title: "The settings that make a run reproducible",
    summary:
      "The settings the earlier labs fixed, now as controls: which text the model reads, the peak learning rate, the schedule, the batch size, and a gradient clip. Both models on the page retrain on every change, and the whole recipe is saved in this lab's state so it can be copied and shared.",
    whatYouSee: [
      "A Training text switch between Cleaned scrape, the default, and Raw scrape, with a sentence under it counting the scrape's lines and characters and what cleaning removes. For Harbor weather notes: 14 lines and 696 characters, of which cleaning removes 3 boilerplate, 2 too-short, 2 repeated and 1 near-repeated line, keeping 6 lines and 379 characters.",
      "Learning rate, a slider whose eight notches double from 0.125 to 16. Schedule, four buttons: Constant, Cosine decay, Warmup + decay, One cycle. Batch size, from 2 to 64 pairs. Gradient clip, from off to 1.50.",
      "Five readouts for the factored model: Held out, its held-out loss; Footer lines, its loss on the three boilerplate lines the raw scrape contains; Steps; Peak gradient norm, the largest batch gradient norm of the run measured before any clipping; and Clipped steps, how many steps the clip rescaled.",
      "A warning that appears only when the loss stops being a number. The run halts at that step and the charts, legend, and model card print `diverged` instead of a loss.",
      "A read-only box with the recipe as one line of text and a Copy my recipe button, then a box to paste a recipe someone sent you and a Load this recipe button.",
    ],
    howItWorks: [
      "The raw scrape is built deterministically from the corpus's own training sentences. After fixed sentence positions it inserts three footer lines, two exact repeats of earlier sentences, one copy of a sentence with a doubled letter, and two fragments. Nothing is random, so the same corpus always gives the same scrape.",
      "Cleaning is a small re-implementation of the dataset-building stages, run in this order, each line taking the first verdict that fires: a boilerplate phrase, a length under 12 characters, an exact duplicate of a kept line, or a near-duplicate of a kept line by character-trigram Jaccard similarity of at least 0.7. The kept lines are the training sentences again, so Cleaned scrape trains on exactly the text this lab always used.",
      "The step size at each step is the learning rate times a schedule factor from the same function as the hyperparameters lab. Constant is 1 throughout; Cosine decay starts at 1 and ends at 0.03; the warmup schedules ramp up over the first 15% of steps before decaying.",
      "Steps per epoch are the number of training pairs divided by the batch size, rounded up, and steps are that times Epochs. A smaller batch means more, noisier updates in the same passes.",
      "For each batch the lab measures one norm over the gradients of both factors together. If it exceeds the clip value, the whole gradient is multiplied by clip divided by the norm before the step; at off nothing is rescaled. Peak gradient norm is the largest value measured, so it is the unclipped one.",
      "The full 900-weight table trains with the same recipe. Its batch gradient norm cannot exceed the square root of 2, because each example contributes a probability vector minus a one-hot vector. The factored model has no such bound: each factor's gradient scales with the other factor, so a large step can grow the next gradient.",
      "The recipe is part of module state, which is why the Copy a link to this state button in the header already shares it. The Copy my recipe button writes only the recipe as a line beginning `tiny-lm-recipe v1`; Load reads such a line, ignores unknown keys and the seed, and clamps every value to what the controls can produce.",
    ],
    controls: [
      "`Training text` chooses between the raw scrape and the cleaned scrape. `Learning rate` sets the peak step size. `Schedule` shapes the step size over the run. `Batch size` sets the pairs per update. `Gradient clip` sets the norm above which a gradient is rescaled.",
      "Comparison worth running at the defaults: set Learning rate to 4 on Constant, read Held out, then switch Schedule to Cosine decay and read it again. Then set Learning rate to 8 on Constant, watch the run halt, and set Gradient clip to 0.50.",
      "Comparison worth running: set Batch size to 8, then 16, then 64, and read Steps and Held out. Then switch Training text to Raw scrape and compare Held out with Footer lines.",
    ],
    notice: [
      "The best learning rate depends on the schedule. At 0.5 Constant wins: 2.117 held out against 2.251 for Cosine decay. At 4 the order reverses: Constant ends at 3.016 and Cosine decay at 2.054.",
      "At 8 on Constant the factored run halts at step 154. With Gradient clip at 0.50 it finishes, with 459 of its 480 steps clipped, at 3.089 held out. With Cosine decay and no clip it finishes at 22.245 after a Peak gradient norm of about 1.2e12; with the clip it ends at 2.065. The full table at the same rate and schedule is untouched, ending at 2.066 with a peak norm of 0.34.",
      "Batch size trades step count against noise. At Epochs 20 the run takes 960 steps at batch 8, 480 at batch 16 and 120 at batch 64, and Held out reads 2.089, 2.117 and 2.618. The cleanest gradient comes with the fewest updates.",
      "Raw scrape adds steps, 880 against 480 for Harbor, and moves Footer lines from 3.090 to 2.509, so the model has learned the page furniture. Held out barely moves, 2.105 against 2.117, so held-out loss alone would not tell you to clean.",
    ],
    limits: [
      "In this lab: the optimizer is plain stochastic gradient descent with no momentum and no per-parameter scaling, so the learning rates here are not comparable to real ones. The scrape is synthetic, built from the same sentences with fixed furniture, and every result is one seed on about 380 characters, so differences under about 0.05 nats are unresolved.",
      "In this lab: the clip rescales the global norm over both factors together, and only the factored model can explode. Weight decay exists in the SDK but is not offered here, and the schedule's warmup share is fixed at 15%.",
      "In general: real recipes use an adaptive optimizer, a warmup, weight decay, mixed precision, and a data mixture rather than one corpus, and they are tuned on held-out sets thousands of times larger. Clipping and schedules are standard there, but the numbers do not transfer from this toy.",
    ],
  },

  "Model card": {
    title: "What you built, and what it does not promise",
    summary:
      "The finished artifact written down: the corpus and its split, the architecture, the recipe, and two losses that disagree. Every value is measured from the run the sliders just defined.",
    whatYouSee: [
      "Six rows: Corpus with its training and held-out character counts, Architecture as a character bigram with the rank and weight count, Recipe listing epochs, batch size, learning rate, schedule, clip, seed and the step count, then Training loss, Held-out loss with its perplexity, and the full table's two losses for comparison.",
      "A metric row of Train, Held out, and Gap, all in nats per token, where Gap is held-out loss minus training loss.",
      "Every value recomputes when you touch a corpus button or either slider; nothing on the card is cached from an earlier configuration.",
    ],
    howItWorks: [
      "Training loss is the mean cross-entropy in nats per token over the training split at the final weights. Held-out loss is the identical formula applied to the held-out sentences, which never entered a gradient step.",
      "Perplexity is the exponential of the held-out loss, so it is the same measurement on a different scale: roughly how many equally likely next characters the model is choosing among out of 30.",
      "The split keeps the last 25% of sentences (rounded, minimum one) out of training. For Harbor weather notes that is 379 training characters and 119 held out.",
      "The comparison row scores the full 900-weight table trained on the same text with the same recipe and seed, so the only difference between the two is how the weights are stored.",
      "The reported step count comes from the full-table run's counter. Both runs take the same number of steps at the same epoch setting, so it matches, but it is not the factored run's own number.",
    ],
    controls: [
      "The card has no controls; it reports whatever the corpus buttons, the Model size and Epochs sliders, and the Training recipe card selected. A diverged run prints `diverged` in place of its losses.",
      "Comparison worth running: at Epochs 20 read Gap at Model size 8 (0.077) and at Model size 12 (0.128). Training loss barely moves between them while the gap widens — extra capacity fitting the training text harder.",
    ],
    notice: [
      "A gap only means something next to a loss. Rank 1 has a modest gap and a bad loss, which is underfitting, not good generalization.",
      "At the defaults the held-out loss is about 2.117 nats, a perplexity near 8 against a 30-symbol vocabulary. That is the honest description of what was learned: 30 options narrowed to roughly 8.",
      "The full table generalizes most tightly here even though it fits worst, so the two losses rank the two models differently. Read them together or you can tell either story.",
      "Nothing on the card is a quality claim. It records scope — what was trained on what, with which recipe, and how it scored.",
    ],
    limits: [
      "In this lab: every value is measured from your run, and the Recipe row is the same recipe the Training recipe card copies as text, but the held-out set is roughly a hundred characters taken from the end of the document rather than a random sample, so treat differences under about 0.05 nats as unresolved.",
      "In this lab: the model conditions on exactly one preceding character, so `t` predicts the same distribution inside `the` as inside `it`. There are no layers, no attention, and no nonlinearity between the two factored matrices.",
      "In general: real model cards add data provenance and licence, the exact recipe, versioned evaluation suites, intended use, and known failure modes. None of it is a quality claim either — a card exists so someone else can judge whether your number applies to their problem.",
      "In general: cross-entropy is not capability. A good loss says nothing about whether a model refuses, cites, or follows an instruction, and a held-out number is honest only while the split is — near-duplicates across a real train and test boundary inflate scores for free.",
    ],
  },
};

export default cardInfo;
