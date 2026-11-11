import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Bars that add to one": {
    title: "A belief spread over five answers",
    summary:
      "A probability distribution gives every possible next word a share of the chance, and the shares add to exactly one whole. Raising one bar must lower the others, because there is only one whole to share out.",
    whatYouSee: [
      "The prompt, the sky is ___, and five candidate words: blue, grey, dark, clear and green.",
      "A strip that is always full, cut into five parts as wide as each word's share.",
      "Five columns on a fixed 0 to 100% scale, each labelled with its share; a dark grip marks the top of a bar you can drag.",
      "Dashed outlines, once you have moved a bar, marking where the bars started.",
      "A formula line that adds the five shares to 1.00.",
    ],
    howItWorks: [
      "The bars are five whole percents that add to 100. Setting one bar to a new value shares the remainder among the other four in proportion to their current sizes, rounded to whole percents.",
      "No bar may go below 1% or above 96%, so no word is ever impossible.",
      "Show as changes only the labels: a share of 50% is printed as 50 of 100 stories, and the strip is ruled into 100 cells.",
    ],
    controls: [
      "Drag any bar, or focus it and press the arrow keys (Page Up and Page Down move 10 points).",
      "Chance of blue, Chance of grey and the other three sliders set the same bars.",
      "Show as switches between percent and counts out of 100 stories.",
      "Reset the bars returns to 50, 25, 12, 8 and 5.",
      "Comparison worth running: drag blue to 80% and check that grey is still twice dark.",
    ],
    notice: [
      "The strip never has a gap and never overflows, whatever you drag.",
      "The other bars keep their proportions to one another as they shrink or grow.",
    ],
    limits: [
      "In this lab: the five shares are invented, not measured from text or from a model, and they move in whole percents with a 1% floor.",
      "In general: a real model spreads its probability over tens of thousands of possible next tokens, most of them with tiny shares, and the distribution is different for every context.",
    ],
  },

  "Draw from the bars": {
    title: "Samples settle onto the bars",
    summary:
      "Drawing a sample picks one word at random, each as likely as its bar is tall. A few draws can sit far from the bars; as draws pile up, each word's share of the draws closes in on its probability. That is the law of large numbers.",
    whatYouSee: [
      "A prediction prompt about a tally of 10 draws.",
      "The most recent draws written out as words, with the followed word highlighted.",
      "Paired columns for each word: the solid bar is its chance and the dotted bar is its share of the draws so far, with the count on top and the fraction underneath.",
      "Three readouts: Draws so far, the share that were the followed word, and Biggest gap, tally to bars in percentage points.",
      "A line chart of the followed word's share of all draws so far, against a dashed line at its chance.",
    ],
    howItWorks: [
      "A seeded generator produces a fixed stream of numbers between 0 and 1. Each number is mapped to the word whose stretch of the 0 to 1 line it lands in, where the stretches are as long as the bars.",
      "Raising the number of draws reads further along the same stream, so the first 10 draws are also the first 10 of the 100.",
      "Biggest gap is the largest difference, over the five words, between a word's share of the draws and its chance.",
      "New luck changes the seed and draws the same number again.",
    ],
    controls: [
      "Number of draws sets the tally to none, 1, 10, 100 or 1,000 draws.",
      "Draw one more adds a single draw.",
      "New luck redraws with a different seed.",
      "Press a word's name under the bars to follow that word in the readout and the line chart.",
      "Comparison worth running: read Biggest gap at 10, 100 and 1,000 draws, then press New luck at 10 several times.",
    ],
    notice: [
      "With the starting seed, the gap is 20.0 points at 10 draws, 9.0 at 100 and 0.9 at 1,000.",
      "The line swings at the left and flattens onto the dashed line; it does not get there in a straight path.",
    ],
    limits: [
      "In this lab: the draws come from a fixed seed, so the same settings always give the same tally, and each draw is independent of the ones before it. The bars are the ones you set on the first card.",
      "In general: a language model's next draw depends on everything drawn so far, and you never get to see the true bars to compare a tally against.",
    ],
  },

  "Averages pile up in a bell": {
    title: "Mean, variance and the bell curve of a test score",
    summary:
      "A number attached to each outcome has a mean (its long-run average) and a variance (its average squared distance from the mean). The average of many draws piles up in a bell curve centred on the mean, whose width shrinks with the square root of the number of draws.",
    whatYouSee: [
      "A table for one answer: each word's chance, the points it earns (1 for the tallest bar, 0 otherwise), and the two sums that give the mean and the variance.",
      "A pile of 2,000 simulated tests: one bar per possible score, as tall as the share of tests that got it. Tinted bars lie within one standard deviation of the mean.",
      "A smooth bell (Gaussian) curve drawn from the pile's own mean and variance, a dashed line at the mean, and a shaded band one standard deviation either side.",
      "Readouts for the pile's mean, variance and standard deviation, and a formula line predicting the standard deviation from the bars.",
    ],
    howItWorks: [
      "The mean of one answer is the sum of chance × points. The variance is the sum of chance × (points − mean)².",
      "Each simulated test draws its questions independently with the hit chance of the tallest bar, and its score is hits ÷ questions. The seed is fixed.",
      "The pile's mean and variance are computed from the 2,000 scores, dividing by the count. The curve is the Gaussian density with that mean and variance, scaled to the width of one bar.",
      "The prediction is the square root of (variance of one answer ÷ questions).",
    ],
    controls: [
      "Questions per test switches between 10, 25, 100 and 400.",
      "Comparison worth running: read Standard deviation at 25, then at 100 and 400.",
    ],
    notice: [
      "Four times the questions halves the standard deviation; the mean does not move.",
      "The measured value sits close to the prediction but not on it, because 2,000 tests are a sample too.",
      "For 400 questions the scores are grouped four to a bar, so the pile can be drawn at the same scale.",
    ],
    limits: [
      "In this lab: the tests are simulated from a closed-form chance with a fixed seed, every question is equally hard and independent, and the bell is an approximation that fits worse for 10 questions or a mean near 0 or 1.",
      "In general: real benchmark questions differ in difficulty and come in related groups, which makes real scores vary more than this pile, and a bell curve's height is a density, not a probability.",
    ],
  },

  "Context changes the bars": {
    title: "One row of counts, rescaled",
    summary:
      "A conditional probability is a probability inside one situation. From a table of counts it is one row divided by that row's own total. The probability of the next word given the text so far is the one thing a language model outputs.",
    whatYouSee: [
      "A table of 200 made-up stories: prompts down the side, next words across the top, a count in each cell with a small bar under it, and each row's total.",
      "The row for the chosen prompt highlighted.",
      "Five columns directly under the table's five word columns: that row's counts divided by its total, with the division printed under each word.",
      "Dashed outlines of the bars for the sky is, when another prompt is chosen.",
      "A formula line for the row's most likely word.",
    ],
    howItWorks: [
      "The table is authored: 100 stories begin the sky is, 50 begin the grass is and 50 begin the sea is.",
      "Each bar is count ÷ row total, so each row's bars add to one whatever its total.",
    ],
    controls: [
      "Prompt switches between the three rows.",
      "Comparison worth running: follow green from 5% after the sky is to 70% after the grass is.",
    ],
    notice: [
      "The same five words get a different distribution for each prompt.",
      "The grass and sea rows have 50 stories each, so their counts are doubled to make percents.",
    ],
    limits: [
      "In this lab: the counts are invented and the prompts are three fixed strings. Nothing is read from a corpus or a model.",
      "In general: a model does not look probabilities up in a table. It computes them from the whole preceding context, and for most contexts no count exists because the exact text has never been seen.",
    ],
  },

  "Read the table backwards": {
    title: "Bayes' rule: prior times likelihood, rescaled",
    summary:
      "Given a clue, Bayes' rule updates a belief over explanations. Multiply each explanation's prior (how common it was) by its likelihood (how well it predicts the clue), then rescale so the results add to one. The result is the posterior.",
    whatYouSee: [
      "A prediction prompt about making one explanation rare.",
      "The count table with the clue word's column highlighted.",
      "An area picture: one column per prompt, as wide as its prior, filled up to the chance that the clue word follows it. Solid, hatched and dotted fills tell the three prompts apart.",
      "The product of width and height printed under each column, and a strip beneath that keeps only the filled areas and stretches them to one whole: the posterior.",
      "A table and a formula line with the same arithmetic in numbers.",
    ],
    howItWorks: [
      "The likelihoods are read from the count table: the clue word's count in each row divided by that row's total.",
      "The prior for the sky is comes from the slider; the other two prompts split the rest equally, as they do in the table.",
      "Posterior = prior × likelihood ÷ (sum of prior × likelihood over the three prompts).",
      "At a prior of 50% this matches counting down the column: 50 of the 75 stories that say blue began the sky is.",
    ],
    controls: [
      "The next word was chooses the clue.",
      "Prior: share of all stories that begin “the sky is” runs from 5% to 90%.",
      "Comparison worth running: with the clue blue, read the posterior for the sky is at a prior of 50%, 20% and 10%.",
    ],
    notice: [
      "Lowering the prior lowers the posterior although no likelihood changes: 67% at a prior of 50%, 18% at 10%.",
      "The chance that a sky story says blue (50%) is not the chance that a blue came from a sky story.",
    ],
    limits: [
      "In this lab: there are three explanations with authored counts, the two other priors are tied together, and the likelihoods are known exactly.",
      "In general: priors are often judgements and likelihoods are estimates, so the posterior is only as trustworthy as both. With many explanations the sum in the denominator can be the hard part to compute.",
    ],
  },

  "Surprise in bits": {
    title: "Surprise counts halvings",
    summary:
      "The surprise of an outcome is minus log base two of its probability, in bits. It counts how many halvings separate the outcome from certainty, so rare outcomes carry more. Probabilities of independent events multiply, and their surprises add.",
    whatYouSee: [
      "The five bars above a line, and under each one a hatched bar hanging down whose length is that word's surprise in bits.",
      "Lines across the hanging bars, one per bit, labelled 1 in 2, 1 in 4, 1 in 8 and so on.",
      "A formula line for the selected word.",
      "A second picture for two words in a row: a rectangle whose area is the product of the two chances, and a bar where the two surprises are laid end to end, with a dashed mark at the surprise of the product.",
    ],
    howItWorks: [
      "Surprise = −log₂(p). A probability of 50% gives 1 bit, 25% gives 2 bits, and 1% gives 6.64 bits.",
      "For two independent draws, P(both) = P(first) × P(second), and −log₂ of a product is the sum of the two surprises.",
      "The bars are the same ones as on the first card, so a change here carries back there.",
    ],
    controls: [
      "Drag a bar, or focus it and press the arrow keys; press a word's name to select it.",
      "Chance of the selected word sets its bar with a slider, and Reset the bars returns to 50, 25, 12, 8 and 5.",
      "First word and Second word choose the pair.",
      "Comparison worth running: drag blue from 50% to 25% and watch its hanging bar move down by exactly one line.",
    ],
    notice: [
      "Short probability bars have long surprise bars.",
      "The dashed mark for the pair always lands exactly at the end of the two stacked surprises.",
    ],
    limits: [
      "In this lab: the two draws are independent draws from the same bars, and the 1% floor keeps every surprise at or below 6.64 bits.",
      "In general: an outcome given probability zero has unbounded surprise, and words in real text are not independent, so a sentence's surprise is a sum of conditional surprises, each given the words before it.",
    ],
  },

  "Average surprise": {
    title: "Entropy is the area of the blocks",
    summary:
      "Entropy is the average surprise of a distribution: each probability times its surprise, added up. It is zero for a sure thing and largest when every outcome is equally likely. Two to the power of the entropy is the size of an equally uncertain fair choice.",
    whatYouSee: [
      "A prediction prompt comparing three shapes of bars.",
      "The bars at the current shape, with dashed outlines of the sky bars.",
      "A row of blocks, one per word, as wide as its probability and as tall as its surprise in bits, with a dashed line at the average height.",
      "Readouts for the entropy and for “as uncertain as a fair choice among”, and five boxes filled up to that number.",
      "A formula line listing chance × surprise for each word.",
    ],
    howItWorks: [
      "The slider blends the shares in a straight line: from one sure thing (all on blue) at the left, through the sky bars in the middle, to five equal bars at the right.",
      "Entropy = Σ p × (−log₂ p). A word with probability zero contributes nothing.",
      "The fair-choice number is 2 raised to the entropy, which later labs call perplexity.",
    ],
    controls: [
      "Shape of the bars moves between one sure thing, the sky bars and all equal.",
      "Comparison worth running: read the entropy at the left end, the middle and the right end.",
    ],
    notice: [
      "Entropy is 0.00 bits at the left, 1.87 in the middle and 2.32 at the right.",
      "The tallest blocks belong to the rarest words, so they are the narrowest and add least.",
    ],
    limits: [
      "In this lab: the shapes lie on one authored path between three fixed distributions, with five outcomes, so the maximum is log₂ 5.",
      "In general: the entropy of real language is not known. Only cross-entropy against a model can be measured, which is an upper bound on it.",
    ],
  },

  "Your guess against what happens": {
    title: "Cross-entropy and the bits a wrong belief wastes",
    summary:
      "Cross-entropy is the average surprise a model pays when outcomes happen with the true probabilities p but surprise is computed from the model's q. It equals the entropy of p plus the KL divergence, which is never negative, so it bottoms out at the entropy when q matches p.",
    whatYouSee: [
      "A prediction prompt asking whether the bill can beat the entropy.",
      "Paired columns: solid bars for what really happens (p) and hatched bars for the model's guess (q), with each word's cost in bits underneath.",
      "A row of blocks as wide as each word's true probability and as tall as the surprise the model pays for it. A dashed outline shows the height the truth's own bars would give.",
      "Two dashed levels, the bill and best possible, with the gap between them bracketed as wasted.",
      "A bar that splits the bill into best possible plus wasted, three readouts and a formula line.",
    ],
    howItWorks: [
      "The bill (cross-entropy) = Σ p × (−log₂ q). Best possible (entropy) = Σ p × (−log₂ p). Wasted (KL divergence) = the bill − best possible.",
      "p is fixed at the authored sky bars, 50, 25, 12, 8 and 5, whatever you did on earlier cards.",
      "Dragging one of the model's bars rescales its other four in proportion, as on the first card.",
    ],
    controls: [
      "Drag a hatched bar, or focus it and press the arrow keys; the Model: sliders set the same bars.",
      "Set the model to jumps to all equal, to 96% on green, or to the truth.",
      "Comparison worth running: raise the model's blue bar to 60% and read The bill, then drag every hatched bar onto its solid bar.",
    ],
    notice: [
      "Wasted reaches 0.00 only when the hatched bars match the solid ones, and never goes below it.",
      "A confident wrong model (96% on green) pays far more than one that knows nothing.",
    ],
    limits: [
      "In this lab: the true bars are known, so best possible can be printed, and the 1% floor caps the bill near 6.6 bits.",
      "In general: the true distribution of text is unknown, so only the bill is measured, on a sample of held-out text, and a low bill shows the model predicts that text, not that the text is true.",
    ],
  },

  "From scores to bars": {
    title: "Softmax, temperature and the sigmoid",
    summary:
      "A network produces one raw score (logit) per outcome. Softmax turns the scores into probabilities by raising e to each one and dividing by the total. Only the gaps between scores matter, temperature rescales those gaps, and with two outcomes softmax is the sigmoid of the gap.",
    whatYouSee: [
      "Three rows lined up by word: the scores drawn around a zero line (negative scores hang below it, hatched), e raised to each score, and the resulting bars.",
      "Captions between the rows for the three moves, with the live total.",
      "Dashed outlines of the bars at temperature 1, when the temperature is not 1.",
      "A formula line for blue.",
      "An S-shaped sigmoid curve for blue against grey, with a dot at the current score gap and readouts that compare softmax on the two scores with the sigmoid of the gap.",
    ],
    howItWorks: [
      "share = e^(score ÷ T) ÷ Σ e^(score ÷ T), where T is the temperature.",
      "Adding a constant to every score multiplies every exponential by the same factor, which cancels in the division.",
      "For two scores a and b, e^a ÷ (e^a + e^b) = 1 ÷ (1 + e^−(a − b)), the sigmoid of the gap.",
      "The middle row is drawn against its tallest bar, because its values can span many orders of magnitude.",
    ],
    controls: [
      "Score: blue and the other four sliders set each score from −6 to 6.",
      "Add 1 to every score and Take 1 from every score shift all five together.",
      "Temperature runs from 0.2 to 4.0.",
      "Reset scores returns to 2.0, 1.3, 0.6, 0.2 and −0.3 at temperature 1.",
      "Comparison worth running: press Add 1 to every score and watch the bars and the dot on the curve stay put.",
    ],
    notice: [
      "The starting scores give the sky bars to the nearest percent.",
      "Temperature below 1 sharpens the bars and above 1 flattens them, and the order of the words never changes.",
    ],
    limits: [
      "In this lab: the five scores are authored so that softmax reproduces the sky bars. No network computed them.",
      "In general: temperature reshapes the distribution a model already has and adds no knowledge, and a model's probabilities can be miscalibrated, meaning too sharp or too flat for how often it is right.",
    ],
  },

  "Challenge: learn it from samples": {
    title: "Estimate a distribution from 40 draws",
    summary:
      "The true bars are hidden and you see only a tally of 40 draws. The best estimate from samples is to count and divide. Your guess is scored by the bits it wastes against the hidden truth, and the goal is 0.05 bits or less.",
    whatYouSee: [
      "Three rounds, each with its own prompt and hidden bars.",
      "Paired columns: dotted bars for each word's share of the 40 draws, with the count on top, and hatched bars for your guess.",
      "A goal meter for Bits wasted, with a mark at the goal, that updates only when you press Score it.",
      "After scoring: your bill, the best possible bill and the wasted bits.",
      "Dashed outlines of the true bars, once you choose to reveal them.",
    ],
    howItWorks: [
      "Each round's tally is 40 seeded draws from an authored hidden distribution.",
      "Score it computes the cross-entropy of your bars against the hidden truth, the truth's entropy, and their difference, the KL divergence.",
      "Changing a bar clears the score until you press Score it again.",
    ],
    controls: [
      "Round chooses one of the three hidden distributions and resets your bars to 20% each.",
      "Drag a hatched bar, or focus it and press the arrow keys; the Guess: sliders set the same bars.",
      "Score it scores the current bars. Start this round again resets them. Reveal the true bars is available after the first scoring.",
    ],
    notice: [
      "Matching each hatched bar to its dotted bar meets the goal in every round.",
      "The winning score is small but not zero, because 40 draws do not pin the true bars exactly.",
    ],
    limits: [
      "In this lab: the three seeds were chosen so that counting and dividing meets the goal. For five outcomes and 40 draws the expected waste of that estimate is about 0.07 bits, so other draws can miss it.",
      "In general: counting gives zero probability to anything that did not appear in the sample, which costs unbounded surprise when it later occurs, so real count-based models smooth their estimates.",
    ],
  },
};

export default cardInfo;
