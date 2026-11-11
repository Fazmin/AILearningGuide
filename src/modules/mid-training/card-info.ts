import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Data mixture": {
    title: "Weights become shares, shares become epochs",
    summary:
      "One bigram table trained live from zeros on a fixed budget of 9,600 character pairs, with every batch slot drawn from Harbor, Recipes, Proverbs or Arithmetic in proportion to the four weights. Each domain is scored on its own held-out sentences.",
    whatYouSee: [
      "A kicker with the budget (600 steps of 16 pairs), the optimizer (plain SGD at rate 3) and the schedule, and a badge with the seed.",
      "Mixture preset with Uniform (1, 1, 1, 1) and General-heavy (3, 3, 3, 1), and four sliders, Harbor weight to Arithmetic weight, each from 0 to 8 with its share of every batch beside it.",
      "Across the top, a bar split into each domain's share of one batch; when Late phase switches the mixture, a second bar shows the late batch. A table of Weight, Share, Pairs drawn, Training pairs and Epochs per domain.",
      "Under the sliders, a ruler for each domain whose blocks are full passes over that domain's own text, the same number as the Epochs column.",
      "Held-out loss by domain: one bar per domain plus General, the mean of the three general domains, out of 3.401, the loss of a table that knows nothing.",
      "Three metrics: General held-out, Arithmetic held-out, and the Late phase in force.",
    ],
    howItWorks: [
      "The run calls the SDK's trainTinyExtended in step mode with four sources. Each of the 16 slots in a batch picks a domain with probability equal to its share, then takes that domain's next pair from its own shuffled order.",
      "Share is a weight divided by the sum of the four. Epochs is Pairs drawn divided by Training pairs: 378 for Harbor, 292 for Recipes, 195 for Proverbs and 3,417 for Arithmetic.",
      "Every domain keeps its last quarter of sentences out of training: 2 of Harbor's 8, 2 of Recipes' 8, 3 of Proverbs' 10 and 40 of the 160 arithmetic sentences. Each bar is the cross-entropy on those, in nats per character.",
      "The schedule warms up over the first 5% of steps, holds rate 3, and decays linearly to zero over the last 20%. When Late phase is not Same mix, the shares change at step 480 and Pairs drawn counts both phases.",
    ],
    controls: [
      "Mixture preset sets all four weights at once; the four sliders set them one at a time. Weights of all zero fall back to Uniform.",
      "Comparison worth running: press Uniform, then General-heavy. Harbor improves from 2.197 to 2.097 and General from 2.365 to 2.311, while Arithmetic gets worse, from 1.967 to 2.348.",
    ],
    notice: [
      "Equal weights do not give equal epochs. Under Uniform, Proverbs gets 12.48 epochs and Arithmetic 0.70, because Proverbs holds 195 training pairs and Arithmetic 3,417.",
      "More epochs stop helping a small domain. General-heavy raises Proverbs from 12.48 to 14.72 epochs, and its held-out loss moves from 2.400 to 2.408.",
      "A share for one domain is taken from the others. The budget is fixed, so every bar that falls is paid for by a bar that rises.",
    ],
    limits: [
      "In this lab: the four domains are tiny, three of them are fixed paragraphs from the SDK and one is generated from three templates, so their sizes, not real-world value, decide where repetition sets in. Each number is one seed.",
      "In this lab: the model is a 900-weight character bigram, so a domain only differs in which letter tends to follow which. General is a plain mean of three losses; real reports weight domains by what the model is for.",
      "In general: real mixtures cover tens of sources at trillions of tokens, are tuned on smaller proxy runs, and are judged on benchmarks as well as per-domain loss. A held-out loss on a domain shows prediction of its text, not a skill.",
    ],
  },

  "Arithmetic weight sweep": {
    title: "Diminishing returns on the scarce domain",
    summary:
      "Nine full runs, one for each Arithmetic weight from 0 to 8 with the other three weights, the late phase and the schedule as set. It shows what each extra epoch of Arithmetic buys on its held-out loss and what General pays for it.",
    whatYouSee: [
      "A kicker naming the three general weights the sweep holds fixed, and a badge with the current Arithmetic weight.",
      "A chart of held-out loss against Arithmetic epochs, solid for Arithmetic and dashed for General, with a marker at the current weight. The legend gives each line's value at weight 8.",
      "A table with Arithmetic weight, Epochs, Arithmetic held-out, Gain per extra epoch and General held-out. The row of the run on the Data mixture card is highlighted, and each gain has a bar under it, so the shrinking return is visible down the column.",
    ],
    howItWorks: [
      "Each row is a separate training run with the same seed, so the row at the current weight is exactly the Data mixture card's run.",
      "Gain per extra epoch is the fall in Arithmetic held-out from the row above divided by the epochs added, in nats per epoch. The first row has no row above, so it shows a dash.",
      "If all three general weights are 0, the weight 0 row would be an empty mixture, so it reads no data.",
    ],
    controls: [
      "Arithmetic weight on the Data mixture card moves the highlight. The other three weights, Late phase and Schedule retrain all nine runs.",
      "Comparison worth running: at General-heavy, read the first and last rows. Arithmetic held-out falls from 2.832 to 1.706 while General held-out rises from 2.320 to 2.515.",
    ],
    notice: [
      "The gain per extra epoch shrinks as the weight grows: 1.75 nats per epoch from weight 0 to 1, 0.25 from weight 7 to 8.",
      "General pays at every step up but the first: from weight 1 to 8 it rises from 2.311 to 2.515.",
      "Even at weight 8 Arithmetic gets only 1.32 epochs, because it is the largest domain, so the shrinking gain here comes from the learning curve flattening, not yet from repetition. Repeated data isolates repetition.",
    ],
    limits: [
      "In this lab: nine runs at one seed on toy domains. At the default mixture, seeds 1 to 8 move Arithmetic held-out by up to 0.07 nats and General held-out by about 0.01, so small differences between neighbouring rows are not resolved. The weights are whole numbers from 0 to 8.",
      "In general: labs choose weights from sweeps of small proxy models, or fit a function of the proportions and optimize it, because a sweep at full size is unaffordable. The best weight at small scale need not be the best at large scale.",
    ],
  },

  "Answer probe": {
    title: "Does a lower loss mean it can add?",
    summary:
      "For each of the 40 held-out arithmetic facts, the probe scores all 21 answer words, zero to twenty, as the completion after is, using the Data mixture card's trained table, and counts how often the likeliest answer is the true one.",
    whatYouSee: [
      "A kicker with the counts, a title, and a badge with the number of right answers out of 40.",
      "Three metrics: Arithmetic held-out, Right answers and Different answers, the number of distinct answers the probe gave across all 40 questions.",
      "Bars for the five likeliest answers, each as a share of the probability the 21 answers receive together.",
      "A grid with one mark for each of the 40 held-out facts, a tick for a right answer and a cross for a wrong one, then three of those facts with the model's answer and the true one.",
    ],
    howItWorks: [
      "Each answer is scored by the log-probability the table gives to its letters and the full stop, starting from the space after is. The answers' probabilities are then renormalized so the 21 shares add up to 100%.",
      "A bigram predicts each character from the previous one only, and the character before every answer is the same space. So the ranking is identical for every question, and Different answers is always 1.",
    ],
    controls: [
      "Every control that changes the Data mixture card's run changes the table probed here: the four weights, Late phase and Schedule.",
      "Comparison worth running: set Arithmetic weight to 0 and then back to 1. With no Arithmetic in the mix the table answers one to every question and gets 3 of 40 right by luck; at the default it answers ten and gets 0 of 40 while its Arithmetic loss falls from 2.832 to 2.348.",
    ],
    notice: [
      "The loss and the probe move in opposite directions. Learning how number words are spelled lowers the loss and shifts the favourite answer to a short, common word, ten, which is never the answer to a held-out fact here.",
      "Short words win: every extra letter multiplies in one more probability below 1, so the five likeliest answers all have three or four letters, whatever the question.",
    ],
    limits: [
      "In this lab: the probe uses only the bigram table and the 40 held-out facts. A model that sees one character cannot see the numbers in the question, so this card shows a ceiling of the model as much as a limit of the loss.",
      "In general: per-domain loss and capability come apart in large models too, which is why labs judge math or code data on benchmarks of held-out problems as well as on loss, as OLMo 2 did with GSM8K after mid-training.",
    ],
  },

  "Repeated data": {
    title: "More passes over less data",
    summary:
      "Train on Arithmetic alone for a fixed Budget of pairs, drawing from only the first Unique sentences of its 120 training sentences, and score the 40 held-out ones. The same budget over fewer sentences means more passes over each.",
    whatYouSee: [
      "A kicker with the budget, a title, and a badge with the epochs the budget buys over the chosen sentences.",
      "Budget from 2,000 to 16,000 pairs in steps of 2,000, and Unique sentences from 2 to 120, each with its pairs.",
      "A chart of held-out loss against pairs drawn: solid for all 120 sentences, dashed for the chosen number.",
      "Bars for 120, 60, 30, 15, 8, 4 and 2 unique sentences at this budget, each with its epochs, and four metrics: Epochs, Held-out loss, Versus 120 unique and Lowest along the run.",
    ],
    howItWorks: [
      "Each run uses the mixture card's recipe with a single source: SGD at rate 3, batch 16, warmup-stable-decay with a 5% warmup and a 20% decay, seed 7. Steps are the budget divided by 16.",
      "Epochs is the budget divided by the pairs in the chosen sentences. The curve is the held-out loss at 21 checkpoints, and Lowest along the run is the best of them.",
      "Late phase and Schedule do not apply to this card.",
    ],
    controls: [
      "Comparison worth running: keep Budget at 8,000 and lower Unique sentences from 120 to 2. Held-out loss goes from 1.388 at 2.3 epochs to 2.542 at 135.6 epochs.",
      "Comparison worth running: at 2 unique sentences, read Lowest along the run, 2.217 at 27.1 epochs, against the final 2.542: the extra passes made it worse.",
    ],
    notice: [
      "Halving the data from 120 to 60 sentences doubles the epochs to 4.7 and costs only 0.011 nats at this budget; 30 sentences at 9.4 epochs cost 0.069. Versus 120 unique subtracts the unrounded losses, so it can differ by 0.001 from the rounded bars.",
      "With very few sentences the dashed line turns back up: the table keeps fitting those sentences' letter pairs and drifts from the held-out ones.",
    ],
    limits: [
      "In this lab: a bigram cannot memorize a sentence, so repetition costs here mainly through too little variety in which letter pairs appear. One seed; the sentences kept are always the first ones in the generated order.",
      "In general: large models memorize repeated text, so repetition costs more, and in more ways, than this card shows. Published results find a few epochs nearly as good as new data, and value falling toward zero with much more repetition.",
    ],
  },

  "Late phase and annealing": {
    title: "Switch the mixture while the rate decays",
    summary:
      "From step 480, the last 20% of the run, the mixture can switch to Target mix, half of every batch Arithmetic, or Target only. The table compares the switched run with one that spreads the same Arithmetic examples evenly, and with no switch at all.",
    whatYouSee: [
      "Late phase with Same mix, Target mix and Target only, and Schedule with Warmup-stable-decay and Constant rate.",
      "A chart of the learning-rate multiplier and Arithmetic's share of each batch across the 600 steps, with a marker where the decay starts.",
      "A chart of Arithmetic and General held-out loss every 30 steps from step 90, solid for this run and dashed for the even spread.",
      "A dot plot of the end-of-run losses on one axis: Arithmetic held-out and General held-out, with a disc for the switched run, a diamond for the even spread and a ring for no switch.",
      "A table of Arithmetic held-out, General held-out, Forgetting and Arithmetic pairs drawn, with columns Switched at decay, Spread evenly and No switch.",
    ],
    howItWorks: [
      "The switched run passes two mixture phases to the trainer: the stable shares until progress 0.8 and the late shares after it. Target mix keeps the general domains in their own proportions for the other half of each late batch.",
      "Spread evenly trains one constant mixture whose shares are 80% of the stable shares plus 20% of the late ones, so it draws the same pairs from each domain on average.",
      "Forgetting is General held-out at the end minus General held-out at the checkpoint at step 480, measured on the same run.",
      "Constant rate holds rate 3 for every step with no warmup and no decay.",
    ],
    controls: [
      "Comparison worth running: set Late phase to Target mix. Arithmetic held-out ends at 1.911 switched against 2.110 spread evenly, and General at 2.398 against 2.339.",
      "Comparison worth running: switch to Target only. Forgetting rises from +0.060 to +0.353 and General held-out from 2.398 to 2.691.",
      "Comparison worth running: with Target mix, change Schedule to Constant rate. Arithmetic ends lower still, 1.785, and General worse, 2.462.",
    ],
    notice: [
      "Timing matters even with the same data: the switched and even runs draw nearly the same Arithmetic pairs, 1,721 and 1,707, and end 0.199 nats apart on Arithmetic.",
      "With no switch, General improves during the decay (Forgetting −0.027); keeping general text in the late batches is what holds that loss down.",
      "At a constant rate the switch helps Arithmetic more and costs General more. In this toy most of the late-data gain comes from coming last, and the decay buys retention.",
    ],
    limits: [
      "In this lab: one seed, a single switch at a fixed point, and a toy target whose loss falls by learning spellings. The decay length, the target share and the switch point are fixed in code.",
      "In general: published mid-training and annealing runs report large benchmark gains from target data during the decay, measured on held-out tasks; this toy reproduces the advantage over an even spread but not a boost from the decay itself.",
    ],
  },
};

export default cardInfo;
