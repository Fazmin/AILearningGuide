import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrate, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-60-probability-surprise",
  slug: "probability-surprise",
  title: "Probability & surprise",
  group: "groundwork",
  order: 4,
  icon: "Dices",
  accent: "#ad4d98",
  prerequisites: ["module-29-what-ai-is"],
  estimatedMinutes: 20,
  steps: [
    "Share out one whole",
    "Draw from the bars",
    "Pile up the averages",
    "Let context change the bars",
    "Measure surprise in bits",
    "Score a guess",
    "Turn scores into bars",
    "Challenge: learn it from samples",
  ],
  stepInstructions: [
    "Drag the blue bar up to 80% and watch the other four bars shrink so the strip above stays one whole; then switch Show as to counts out of 100 stories.",
    "Set Number of draws to 10, then 100, then 1,000, and compare Biggest gap, tally to bars each time; press New luck to see a different run of the same size.",
    "Switch Questions per test from 25 to 100 and compare Standard deviation with the width of the pile; Mean of the pile should stay put.",
    "Switch Prompt to the grass is ___ and watch the bars change; then, on Read the table backwards, drag the Prior slider down to 10% and compare the posterior for the sky is.",
    "Drag the blue bar from 50% down to 25% and watch its hanging bar move from the 1-bit line to the 2-bit line; then, on Average surprise, drag Shape of the bars to each end and compare the dashed average line.",
    "Drag the model's hatched bars onto the solid bars and watch Wasted (KL divergence) fall to 0.00 while The bill stops at Best possible.",
    "Press Add 1 to every score and check that the bars do not move; then set Temperature to 0.5 and to 2.0 and compare the bars with their dashed outlines.",
    "Read the tally of 40 draws, set your five hatched bars, and press Score it; aim to get Bits wasted to 0.05 or less.",
  ],
  stateVersion: 1,
  tagline:
    "Turn chance into bars that add to one, draw samples from them, and measure in bits how surprised a guesser is: the same three moves score every language model.",
  objectives: [
    "Read a probability distribution as one whole shared among outcomes, and a conditional probability as one row of a count table rescaled to add to one",
    "Predict how a tally of samples settles onto the true bars, and read the mean, the variance and the bell curve of an average",
    "Update a belief with Bayes' rule by multiplying a prior by a likelihood and rescaling",
    "Measure surprise in bits, average it into entropy, and explain why cross-entropy can never fall below the entropy",
    "Turn raw scores into probabilities with softmax, and say what temperature and the sigmoid do",
  ],
  glossary: [
    {
      term: "Probability distribution",
      definition:
        "A share of the chance for every possible outcome, each from 0 to 1, that together add to exactly one. Raising one share must lower others. It describes how often outcomes happen in the long run, not what the next one will be.",
    },
    {
      term: "Sample",
      definition:
        "One outcome picked at random so that each is as likely as its probability says. By the law of large numbers, each outcome's share of many samples closes in on its probability; a handful of samples can sit far from it.",
    },
    {
      term: "Mean and variance",
      definition:
        "The mean, or expected value, is the long-run average of a number attached to each outcome: each chance times its number, added up. The variance is the average squared distance from the mean, and its square root, the standard deviation, is the typical distance. The average of n independent draws has the same mean and a variance n times smaller, so its spread shrinks only as the square root of n.",
    },
    {
      term: "Gaussian (normal) distribution",
      definition:
        "The bell curve, fixed by two numbers: its mean sets the centre and its variance sets the width, with about 68 percent of the area within one standard deviation of the mean. Averages of many independent draws pile up close to this shape. The fit is poor for short tests or averages near 0 or 1, and the curve's height is a density, not a probability.",
    },
    {
      term: "Conditional probability",
      definition:
        "The probability of an outcome within one situation, written P(word | prompt) and read as the probability of the word given the prompt. From a count table it is one row divided by that row's own total. It is what a language model outputs for the next token given the text so far.",
    },
    {
      term: "Bayes' rule",
      definition:
        "The way to turn a clue into an updated belief: multiply each explanation's prior by its likelihood, the chance of the clue under that explanation, then divide by the total so the results add to one. A likelihood on its own is not the answer; skipping the prior is the usual mistake.",
    },
    {
      term: "Prior",
      definition:
        "What you believed about each explanation before seeing the clue, such as each prompt's share of all the stories. In the lab it is a count you can see; in real problems it is often a judgement, and the conclusion depends on it.",
    },
    {
      term: "Posterior",
      definition:
        "The updated belief after the clue: each prior times its likelihood, rescaled to add to one. A rare explanation stays fairly unlikely after a clue unless the clue is far more likely under it than under the alternatives.",
    },
    {
      term: "Surprise",
      definition:
        "How unexpected an outcome is, measured as minus log base two of its probability, in bits. It counts halvings from certainty: 50 percent is 1 bit, 25 percent is 2 bits. Multiplying probabilities adds surprises. Later labs call it surprisal, and some use the natural logarithm, which gives nats (1 bit is about 0.69 nats).",
    },
    {
      term: "Entropy",
      definition:
        "The average surprise of a distribution when outcomes happen with their own probabilities: each probability times its surprise, added up. It is zero for a sure thing and largest when all outcomes are equally likely. Two to the power of the entropy is the perplexity, the size of an equally uncertain fair choice.",
    },
    {
      term: "Cross-entropy",
      definition:
        "The average surprise a model pays when outcomes happen with the true probabilities p but surprise is computed from the model's probabilities q. It is the loss language models are trained on. It can fall to the entropy of p and no lower, and a low value says the model predicts the text well, not that the text is true.",
    },
    {
      term: "KL divergence",
      definition:
        "The extra bits paid for believing q when the world follows p: the cross-entropy minus the entropy. It is never negative and is zero only when q equals p. It is not symmetric, so swapping p and q gives a different number.",
    },
    {
      term: "Softmax and temperature",
      definition:
        "Softmax turns raw scores (logits) into probabilities: raise e to each score, then divide by the total. Only the gaps between scores matter. Temperature divides every score first: below 1 sharpens the distribution, above 1 flattens it, and the ranking never changes. It reshapes what the model already believes and adds no knowledge.",
    },
    {
      term: "Sigmoid and logit",
      definition:
        "A logit is a raw score of any size or sign. The sigmoid, 1 / (1 + e^−z), squeezes a number z into a share between 0 and 1, with 50 percent at z = 0. With two outcomes, softmax is the sigmoid of the gap between their logits, and that gap is the log of the odds.",
    },
  ],
  references: [
    {
      authors: "Joseph K. Blitzstein and Jessica Hwang",
      title: "Introduction to Probability, Second Edition",
      source: "Chapman and Hall/CRC, free to read online",
      year: 2019,
      url: "https://stat110.hsites.harvard.edu/",
      note: "The textbook for Harvard's Stat 110 course. Its conditional probability chapter works through Bayes' rule and its common pitfalls, and later chapters cover variance, probability density functions, and the Normal (Gaussian) distribution that the bell card draws.",
    },
    {
      authors: "Kevin P. Murphy",
      title: "Probabilistic Machine Learning: An Introduction",
      source: "MIT Press",
      year: 2022,
      url: "https://mitpress.mit.edu/9780262046824/probabilistic-machine-learning/",
      note: "Section 2.3 applies Bayes' rule to a COVID-19 test and shows a low base rate keeping the posterior low, the same lesson as lowering the Prior slider. Chapter 2 also covers the sigmoid, softmax, and the Gaussian, and Chapter 6 defines entropy, cross-entropy, and KL divergence.",
    },
    {
      authors: "Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong",
      title: "Mathematics for Machine Learning",
      source: "Cambridge University Press, free to read online",
      year: 2020,
      url: "https://mml-book.github.io/",
      note: "Chapter 6, Probability and Distribution, is the textbook treatment of the ground this lab's first cards cover with bars, and it includes the Gaussian that the bell card draws.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 5 builds training losses from the likelihood and takes its log so a product of tiny probabilities becomes a sum, which is the surprise card's point. Its binary example turns a network output into a probability with the logistic sigmoid.",
    },
    {
      authors: "Michelle McDowell and Perke Jacobs",
      title: "Meta-analysis of the effect of natural frequencies on Bayesian reasoning",
      source: "Psychological Bulletin 143(12), 1273–1312",
      year: 2017,
      url: "https://www.semanticscholar.org/paper/ef69509ad2f141e7d045c2ced471b56dc9ae198d",
      note: "Reviews 20 years of studies finding that people solve Bayes' rule problems more often when the numbers come as counts than as conditional probabilities. That is why the lesson counts stories, 50 of 75, before it multiplies a prior by a likelihood.",
    },
    {
      authors: "Claude E. Shannon",
      title: "A Mathematical Theory of Communication",
      source: "The Bell System Technical Journal 27, 379–423 and 623–656",
      year: 1948,
      url: "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
      note: "The paper that introduced measuring information with a base-two logarithm, in units it calls bits, and that defines the entropy of a set of probabilities as minus the sum of p log p. The surprise and entropy cards compute exactly these quantities.",
    },
    {
      authors: "David J. C. MacKay",
      title: "Information Theory, Inference, and Learning Algorithms",
      source: "Cambridge University Press, free to view online",
      year: 2003,
      url: "https://www.inference.org.uk/mackay/itila/book.html",
      note: "A textbook that covers information theory and inference together, which is the route this lab takes from bars to Bayes' rule to bits. The author's page says you are welcome to view the whole book on screen.",
    },
    {
      authors: "Christopher Olah",
      title: "Visual Information Theory",
      source: "colah's blog",
      year: 2015,
      url: "https://colah.github.io/posts/2015-09-Visual-Information/",
      note: "Draws entropy and cross-entropy as areas, with probability along one side of each rectangle and a length in bits along the other, and defines KL divergence as the extra bits. The lab's row of blocks is the same picture with its two axes swapped.",
    },
    {
      authors: "Daniel Jurafsky and James H. Martin",
      title: "Speech and Language Processing, 3rd edition draft",
      source: "Chapter 3, N-gram Language Models (draft of August 19, 2026)",
      year: 2026,
      url: "https://web.stanford.edu/~jurafsky/slp3/",
      note: "Section 3.1.2 estimates next-word probabilities by counting and normalizing, the winning move of the challenge. Section 3.3 defines perplexity, Section 3.6 explains why unseen events must not get zero probability, and Section 3.7 ties perplexity to entropy and cross-entropy.",
    },
    {
      authors: "Geoffrey Hinton, Oriol Vinyals, and Jeff Dean",
      title: "Distilling the Knowledge in a Neural Network",
      source: "arXiv:1503.02531",
      year: 2015,
      url: "https://arxiv.org/abs/1503.02531",
      note: "Section 2 writes softmax with a temperature T that divides every logit and notes that a higher T gives a softer distribution over classes, the Temperature slider on the softmax card. It trains a small model on those softened probabilities with a cross-entropy loss.",
    },
    {
      authors: "Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger",
      title: "On Calibration of Modern Neural Networks",
      source: "Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321–1330",
      year: 2017,
      url: "https://proceedings.mlr.press/v70/guo17a.html",
      note: "Finds that modern networks' probabilities are often poorly calibrated and that fitting a single temperature after training corrects much of it. It backs the lesson's caution that temperature reshapes confidence without adding knowledge.",
    },
    {
      authors: "Evan Miller",
      title: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations",
      source: "arXiv:2411.00640",
      year: 2024,
      url: "https://arxiv.org/abs/2411.00640",
      note: "Treats the questions of an evaluation as a sample drawn from a larger population and gives formulas for analysing scores and comparing two models. It is the bell card applied to real benchmarks: a score is one noisy draw and needs an error bar.",
    },
  ],
  checkpoint: [
    {
      prompt: "On Bars that add to one you drag the blue bar from 50% up to 80%. What happens to the other four bars?",
      options: [
        "They stay where they are, so the five bars now add up to 130% in total",
        "They all shrink in proportion, sharing the 20% that is left",
        "Only the grey bar shrinks, because it is the next tallest",
        "They all drop to the same height, 5% each",
      ],
      answer: 1,
      explanation:
        "A distribution is one whole shared out, so the five bars must still add to 100%. The lab rescales the other four so they keep their proportions to one another: grey had half of what blue left over and still does, at 10%. Bars cannot add to more than one whole, and nothing singles out one bar to absorb the change.",
      objective: 0,
    },
    {
      prompt: "The count table has 50 stories that begin “the grass is”, and 35 of them go on to say green. What is P(green | the grass is)?",
      options: [
        "35 ÷ 200 = 0.175, because there are 200 stories in the table",
        "35 ÷ 47 = 0.74, because 47 stories in the table say green",
        "35 ÷ 50 = 0.70, because only the grass row counts",
      ],
      answer: 2,
      explanation:
        "A conditional probability is worked out inside one situation, so you keep the grass row and divide by that row's own total. Dividing by all 200 stories gives the chance of a grass story that says green among all stories. Dividing by the green column answers the reverse question, which prompt a green came from.",
      objective: 0,
    },
    {
      prompt: "On Draw from the bars you raise Number of draws from 10 to 1,000. What happens to Biggest gap, tally to bars?",
      options: [
        "It shrinks, because each word's share of the draws closes in on its probability",
        "It grows, because more draws give luck more chances to pile up",
        "It stays about the same, because every single draw is exactly as random as the one before it",
      ],
      answer: 0,
      explanation:
        "This is the law of large numbers. Each draw is as random as ever, but the shares are averages over all the draws, and averages of many draws wander less. In the lab the gap falls from 20.0 points at 10 draws to 0.9 points at 1,000. Luck does not pile up in one direction: misses on either side tend to cancel.",
      objective: 1,
    },
    {
      prompt: "On Averages pile up in a bell, a 25-question test has a standard deviation of about 0.10. You want it to be about 0.05. How many questions do you need?",
      options: [
        "50, because twice the questions gives half the spread",
        "100, because the spread shrinks with the square root of the number of questions",
        "625, because the spread shrinks with the square of the number of questions",
      ],
      answer: 1,
      explanation:
        "The variance of an average is the variance of one answer divided by the number of questions, and the standard deviation is its square root. Four times the questions gives a quarter of the variance and half the standard deviation. That slow shrinking is why small test sets give noisy scores and why later labs put error bars on them.",
      objective: 1,
    },
    {
      prompt: "On Read the table backwards the next word is blue, and you drag the Prior for “the sky is” from 50% down to 10%. What happens to the posterior for “the sky is”?",
      options: [
        "It stays at 67%, because blue is just as typical of sky stories as before",
        "It rises, because a rarer explanation makes the same clue more telling",
        "It falls to 10%, because the clue stops mattering once the prior changes",
        "It falls to about 18%, because there are fewer sky stories to say blue",
      ],
      answer: 3,
      explanation:
        "Bayes' rule multiplies each prior by its likelihood and rescales. The likelihoods (50%, 10% and 40%) have not changed, but the sky's product falls from 0.25 to 0.05 while the others grow, so its share of the total drops to 18%. The clue still helps: 18% is well above the 10% prior. A likelihood alone is never the posterior.",
      objective: 2,
    },
    {
      prompt: "A word's probability falls from 20% to 10%. What happens to its surprise in bits?",
      options: [
        "It doubles, because the word became half as likely",
        "It goes up by 10, because the probability fell by 10 points",
        "It goes up by exactly 1 bit, because the probability halved",
      ],
      answer: 2,
      explanation:
        "Surprise is minus log base two of the probability, which counts halvings from certainty. Every halving adds one bit wherever you start: 20% is 2.32 bits and 10% is 3.32 bits. Surprise only doubles when the probability is squared, and it does not follow the size of the drop in percentage points.",
      objective: 3,
    },
    {
      prompt: "On Your guess against what happens the truth's entropy is 1.87 bits. You make the model more confident by raising its blue bar to 60%. What can The bill do?",
      options: [
        "Fall below 1.87 bits, because blue is the most likely word and now costs less",
        "Stay above 1.87 bits, because the other words cost more and still happen half the time",
        "Stay exactly where it was, because the bill depends only on what really happens",
      ],
      answer: 1,
      explanation:
        "The bill is the cross-entropy: each word's true probability times the surprise the model assigns it. Raising blue to 60% makes blue cheaper (0.74 bits) but the others dearer (3.32 bits), and the bill is 2.03 bits. Cross-entropy equals the entropy only when the model's bars match the truth, and it is never lower.",
      objective: 3,
    },
    {
      prompt: "On From scores to bars you press Add 1 to every score. What happens to the bars?",
      options: [
        "Every bar grows, because every score went up",
        "The tallest bar grows and the others shrink, because larger scores mean more confidence",
        "Nothing, because softmax depends only on the gaps between scores",
      ],
      answer: 2,
      explanation:
        "Adding 1 to a score multiplies its “e to the score” by e. Every term in the fraction is multiplied by the same factor, so the shares are unchanged. Confidence comes from the gaps between scores, not their size. With two words this is the sigmoid: the share depends only on the gap between the two scores.",
      objective: 4,
    },
    {
      prompt: "You lower Temperature from 1.0 to 0.5 on the softmax card. What changes?",
      options: [
        "The bars sharpen towards the top word, and the order of the words stays the same",
        "The order of the words changes, because the model reconsiders its scores",
        "The bars flatten towards equal, because a lower temperature means less energy",
      ],
      answer: 0,
      explanation:
        "Temperature divides every score before softmax, so 0.5 doubles every gap and the largest share grows: blue rises from 50% to 74%. The ranking cannot change, because dividing by a positive number keeps the order of the scores. Temperature reshapes the distribution the model already has; it gives the model no new information.",
      objective: 4,
    },
    {
      prompt: "In the challenge, counting and dividing a tally of 40 draws scores 0.010 bits wasted, not 0.000. Why?",
      options: [
        "The lab's bars only move in whole percents, and nothing else is lost",
        "Counting and dividing is the wrong method here, and a cleverer way of guessing would bring the wasted bits to zero",
        "Forty draws are a noisy sample, so the tally's shares are close to the truth but not equal to it",
      ],
      answer: 2,
      explanation:
        "The tally is a small sample, and small samples miss the true probabilities, as the 10-draw tally did on the second card. Here the tally gives dark 20 of 40, or 50%, when the hidden truth is 55%. Counting is still the best use of the evidence. No method can recover what 40 draws did not capture, which is where the lab's toy stops and real estimation problems begin.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrate,
};

export default definition;
