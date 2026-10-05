import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Recursive training": {
    title: "Ten generations, each fitted only to the last one's text",
    summary:
      "Generation 0 is the character-bigram table fitted to 499 characters of human harbor notes. Every later generation is fitted to text that a generator wrote: by default the previous generation, or with the Generator control a fixed stronger teacher. The chart tracks what that loop keeps, for two regimes: synthetic text replacing the data, or synthetic text added on top of the human seed.",
    whatYouSee: [
      "The x axis is generation, 0 to 10. The y axis is whichever Measure is selected: Transitions (distinct character pairs with evidence), Coverage (share of the seed's character pairs still producible), Entropy (next-character uncertainty in nats), Real words (share of the generation's own writing that is seed words), or Unseen text (share of the character pairs in 586 characters of held-out harbor sentences that the generation can produce).",
      "Two families of lines. Synthetic only is drawn solid in the loss tone; Real + synthetic is drawn dashed in the forward tone. Each family has eight thin lines, one per sampling seed, and a thick line through their mean. The selected Training data family is drawn stronger.",
      "A dashed guide and enlarged dots at the Generation you selected. The legend and the three metrics repeat the generation-0 value and both means at that generation.",
    ],
    howItWorks: [
      "Each model is fitted exactly: row c holds the count of each character that followed c in the training text, divided by the row total. That is the maximum-likelihood bigram table, the optimum SGD would approach.",
      "To build generation g's data, generation g−1 drafts text at the chosen Temperature, the Filter keeps what it accepts, and drafting continues until Sample size characters are kept. Synthetic only fits generation g to that text alone; Real + synthetic fits it to the seed plus every batch kept so far.",
      "A pair the training text never contains gets probability zero, and a model cannot write what it assigns zero. So under Synthetic only the transition count can never rise. Under Real + synthetic the seed keeps every original pair in the counts.",
      "Entropy is each row's entropy weighted by how often the seed uses that character as context. Real words samples 2000 characters from the generation at the chosen temperature and counts tokens that are one of the seed's 57 words.",
      "Unseen text counts every adjacent pair in the held-out harbor sentences and asks whether the generation's training counts contain it. Those sentences are in no training text, the teacher's included. The seed-fitted model scores 92.6%.",
      "With Generator set to A stronger teacher, the draft in step 2 is written by a fixed teacher table instead of by generation g−1. Everything else is unchanged, so the only difference is who writes.",
      "The Transitions axis runs to 160 for the model itself, and grows to the next multiple of 40 when a teacher lifts a line past it.",
    ],
    controls: [
      "`Training data` switches which family is emphasised and which run feeds the sample and grid cards. Both families are always computed.",
      "`Measure` changes the y axis. `Generation` moves the guide and every per-generation readout on the page.",
      "`Generator` on `Generator and filter` chooses who writes the data. Both families recompute.",
      "Comparison worth running: at the defaults read Transitions at generation 10 for both families (51 against 157), then set Measure to Entropy and compare 0.912 with 1.713. Then set Measure to Unseen text and switch Generator to A stronger teacher with Real + synthetic: the line rises from 92.6% to 99.4% instead of staying flat.",
    ],
    notice: [
      "At the defaults the Synthetic only mean falls 157, 125, 105 over the first two generations and reaches 51 by generation 10; the eight runs end between 46 and 58. The drop is largest at first, when the rarest pairs are the easiest to miss.",
      "Real + synthetic holds Transitions at exactly 157 and Coverage at 100% at every setting, yet Entropy still drifts: from 1.787 to 1.713 at temperature 1, and to 1.341 at temperature 0.8. Keeping the seed stops loss, not narrowing.",
      "Real words can rise while Transitions collapse. Repeating `the the the` scores 100% real words with four transitions.",
      "Under Synthetic only the model itself loses Unseen text, from 92.6% to 55.6% by generation 10. A teacher stops that decay: its line is level from generation 1, because a fixed source cannot compound anyone's errors.",
      "A level line is not a high line. At 500 characters per generation the teacher's Synthetic only student sits near 87%, below the seed-fitted model's 92.6%, because one finite sample of the teacher still misses pairs. At 2000 characters it reaches 97.5%, above the seed, with no human data in any round.",
    ],
    limits: [
      "In this lab: every value is computed from real sampling and exact refits, and each line is eight seeds, not one. But the model has one character of context and 900 numbers, the seed is 499 characters, and there are only ten generations.",
      "In this lab: the fit is by counting, so an unseen pair is exactly zero. An SGD-trained table from zero weights keeps some probability on every pair and instead drifts toward noise when retrained on its own samples. That is a different face of the same loop, not shown here.",
      "In this lab: the held-out sentences were written by hand in the seed's style and share its theme, as the teacher's extra notes do. Unseen text asks whether a pair can be produced, not how likely it is, so it rewards covering pairs, not getting their odds right.",
      "In general: published collapse results come from repeatedly retraining language models and image generators on their own outputs (Shumailov et al., 2023), and accumulating synthetic data alongside the original real data avoided it in the models Gerstgrasser et al. (2024) tested. Real pipelines mix synthetic data with fresh human data, deduplicate it, and filter it with verifiers, which is why synthetic data can help rather than harm.",
    ],
  },

  "What a generation writes": {
    title: "The text the next generation will be fitted to",
    summary:
      "The first 240 characters that the selected generation writes, from run 1 of 8, at the chosen temperature. With the real-words verifier on, each token is marked kept or struck through, which is exactly what the filter does to drafts before they become training data.",
    whatYouSee: [
      "A kicker naming the run, the Training data regime, and the temperature, and a title naming the generation. Generation 0 is the model fitted to the human seed.",
      "The sample in monospace. When Filter is Real words only, rejected tokens are struck through and dimmed, so the verdict does not depend on colour.",
      "Three metrics: `Real words here` (the share of tokens in this excerpt that are seed words), `Filter keeps` (the share of this excerpt's characters the verifier would keep), and `Trained on` (how many human and synthetic characters this generation was fitted to).",
    ],
    howItWorks: [
      "The sample is drawn one character at a time from the generation's fitted row for the previous character, with every probability raised to the power 1/T and renormalized. It starts after a space.",
      "The verifier splits on spaces and keeps a token only if, ignoring full stops, it is one of the 57 words in the seed. It never edits a token and never adds one.",
      "Under Real + synthetic, `Trained on` adds the seed's 499 characters to every synthetic batch kept so far; under Synthetic only it is the latest batch alone.",
    ],
    controls: [
      "The card follows `Generation` and `Training data` on the chart and the three controls on `Generator and filter`.",
      "Comparison worth running: with Training data on Synthetic only, read generation 0, 5 and 10 at temperature 0.6. By generation 6 every run is writing `the the the`.",
    ],
    notice: [
      "Generation 0 is already mostly non-words: a bigram joins real fragments into invented words, so about 7% of its tokens are seed words at temperature 1. The loop starts from that, not from the human text.",
      "Later Synthetic only samples repeat a shrinking stock of fragments. Nothing new appears, because every pair the model writes had to be in the text it was fitted to.",
      "With the verifier on, early generations lose most of their draft. That discarded text is the price of rejection sampling.",
    ],
    limits: [
      "In this lab: one run and one 240-character excerpt are shown; the chart averages eight runs over 2000-character samples. A single excerpt can look better or worse than the mean.",
      "In general: a real synthetic-data pipeline samples prompts as well as completions, often from a stronger model than the one being trained, and checks outputs against far richer verifiers than a word list.",
    ],
  },

  "Generator and filter": {
    title: "Sharpen, verify, and size each round",
    summary:
      "Four settings that decide what each generation writes and keeps: who writes the data (the model itself or a fixed stronger teacher), the sampling temperature, whether a real-words verifier filters the drafts, and how many characters each generation is fitted to.",
    whatYouSee: [
      "`Generator`: The model itself, where each generation is drafted by the previous one, or A stronger teacher, a fixed model that writes every generation's data. With the teacher on, a note says how it was trained.",
      "`Temperature` from 0.5 to 1.2, labelled sharpened below 1, flattened above 1, and the model as fitted at exactly 1.",
      "`Filter`: No filter keeps every drafted character; Real words only keeps tokens that are one of the seed's 57 words.",
      "`Sample size`: 250, 500, or 2000 characters of kept text per generation. The seed itself is 499 characters.",
    ],
    howItWorks: [
      "The teacher is a character bigram trained by the track's SGD trainer, `trainTinyModel`, for 40 epochs at batch 16 and learning rate 6, seed 5, on the 499 seed characters plus 1,515 characters of extra harbor notes that the student never sees: 2,014 characters, about four times the seed. Its rows are softmaxed, the two reserved angle brackets are masked, and each row is renormalized. It is trained once and reused.",
      "SGD never drives a probability exactly to zero, so the teacher keeps about 1.0% of its probability, averaged over the rows its data uses, on pairs its data never contained. Its data holds 258 distinct pairs, and a student fed 10 generations of its text can list more, up to 275 at the defaults.",
      "Temperature raises each probability to 1/T before renormalizing, so a pair with probability zero stays zero at any temperature. Below 1 the likeliest continuation gains share each round, and the loss compounds.",
      "With the verifier on, drafting repeats in chunks until the kept text reaches Sample size, up to 40 times the sample size in drafts. The accepted share is recorded for every generation.",
      "Sample size sets how many draws each generation gets. A pair with probability 0.2% appears about once in 500 characters, so it is often missed at 250 and usually caught at 2000.",
    ],
    controls: [
      "`Generator`, `Temperature`, `Filter`, and `Sample size` each recompute all sixteen chains, eight per Training data regime.",
      "Comparison worth running: with Synthetic only and Measure on Transitions, read generation 10 at Sample size 250, 500 and 2000: about 34, 51 and 109. Then set Measure to Unseen text and Generator to A stronger teacher, and read the same three sizes again.",
      "Comparison worth running: with the teacher on and Real + synthetic, switch Filter to Real words only and watch Unseen text at generation 10 fall from 99.4% to 93.0%.",
    ],
    notice: [
      "Temperature 0.8 takes the Synthetic only mean at generation 10 from 51 transitions to about 11, and 0.6 reaches the four-pair `the ` loop by generation 6. Temperature 1.2 slows the loss (73 at generation 10) but cannot undo it.",
      "The verifier keeps about 5% of drafted characters at generation 1. Under Synthetic only it pushes Real words above 99% while Transitions fall to 16, fewer than with no filter at all.",
      "Under Real + synthetic the same verifier lifts Real words to 82% by generation 10 while all 157 transitions stay: verified synthetic data plus the original data is the combination that helps.",
      "The teacher's extra knowledge is mostly words the seed never used, so the seed-word verifier rejects it. With the verifier on, Synthetic only with the teacher scores 42.2% on Unseen text at generation 1, and Real + synthetic ends at 93.0%, close to the seed's 92.6%.",
      "A stronger teacher is not a free pass: it can only teach what it knows. The seed plus extra notes cover 99.5% of the held-out pairs, which is the most this teacher's data can offer a student.",
    ],
    limits: [
      "In this lab: the verifier is a 57-word list, a stand-in for a checker. It measures spelling against the seed, not truth or usefulness.",
      "In this lab: the teacher is a bigram too, only fitted to four times the text, so it is stronger in coverage and not in any kind of understanding. It is not the Distillation & pruning lab's setup, where a student is trained against the teacher's probabilities rather than its sampled text.",
      "In general: verifiers such as unit tests, proof checkers, or answer matching are what make synthetic data useful for code and math. They filter the generator's output; they do not extend what the generator can write.",
      "In general: a stronger model writing a weaker one's training data is common. Stanford's Alpaca fine-tuned a 7B LLaMA model on 52K instruction-following examples generated by OpenAI's text-davinci-003, for less than $500 of API use, and its authors restricted it to academic research partly because the data inherits the teacher's terms of use. The data's licence is part of what the student inherits.",
    ],
  },

  "Which transitions survive": {
    title: "The seed's character pairs, and which ones are gone",
    summary:
      "Every character pair the human seed contains, laid out as a grid, marked present or lost in the selected generation's training text for run 1. Lost cells cluster among the pairs the seed used least.",
    whatYouSee: [
      "A 28 × 28 grid: rows are the current character, columns the next one (space is shown as ␣). Only pairs that occur in the seed are drawn.",
      "A filled cell is a pair still present; darker means the seed used it more often. An outlined cell with a diagonal stroke is a pair that is gone.",
      "The title counts how many of the seed's 157 pairs remain. Three bars below split the lost pairs by how often the seed used them: once, 2–4 times, or 5+ times.",
    ],
    howItWorks: [
      "A cell is present when the selected generation's training counts contain that pair at least once. Under Real + synthetic the seed is part of those counts, so no cell can be lost.",
      "The grouping reads the seed's own counts: 72 pairs occur once, 58 occur 2–4 times, and 27 occur 5 or more times.",
    ],
    controls: [
      "The grid follows `Generation`, `Training data`, and the three generator settings. Hover a cell for the pair and its seed count.",
      "Comparison worth running: at the defaults step Generation from 1 to 10 with Synthetic only and watch the crossed cells spread from the pale ones toward the dark ones.",
    ],
    notice: [
      "At the defaults, run 1 at generation 10 has lost 63 of the 72 once-seen pairs, 35 of 58 pairs seen 2–4 times, and 8 of 27 common pairs. The tail goes first.",
      "No cell outside the seed ever appears. The loop can only lose pairs, never invent them, because a zero-probability pair cannot be sampled.",
    ],
    limits: [
      "In this lab: the grid shows one run of eight, so its counts can differ from the chart's mean.",
      "In general: a language model's tail is not a grid of character pairs but rare facts, dialects, styles, and edge cases. The mechanism is the same: what a finite sample misses, the next model does not learn.",
    ],
  },
};

export default cardInfo;
