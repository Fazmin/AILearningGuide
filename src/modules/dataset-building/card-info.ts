import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Source documents and cleaning verdicts": {
    title: "Every scraped line, and why it stayed or left",
    summary:
      "Twenty hard-coded lines that look like a scraped page — real harbor sentences mixed with copyright footers, subscribe prompts, an exact repeat, a spelling variant, and two fragments — each tagged with the cleaning stage that removed it, or `kept`. A switch can copy one held-out sentence into the page.",
    whatYouSee: [
      "A kicker counting the scraped lines (20, or 21 with the leak on) and how many the current pipeline removed, plus a `Show` control: `All lines` or `Kept only`.",
      "A pressable row, `Leak a held-out sentence into the scrape`, reading `off` or `leaked`.",
      "An ordered list. Each row is the line text, a verdict chip (`kept`, `boilerplate`, `too short`, `exact repeat`, `near-duplicate`, or `held-out overlap`), and the line's character count. A near-duplicate chip also prints the trigram Jaccard `J` with the line it matched; a leaked line that survives says `leaked`.",
      "Removed lines are struck through and dimmed, so a verdict never depends on the chip colour alone.",
    ],
    howItWorks: [
      "Verdicts are assigned in pipeline order, first match wins: boilerplate phrases (`all rights reserved`, `click here`, `subscribe`, `share this page`, `copyright`), then `Minimum line length`, then an exact match with a line already kept, then trigram Jaccard of at least 0.7 with a kept line, then any 8-word run shared with the held-out text.",
      "Duplicate checks compare only against lines that survived the earlier stages, so a stage's count depends on which stages ran before it.",
      "The leak switch inserts “the fog returns in the morning and the pattern repeats.”, one of the three held-out sentences, after the rain line. `Show` filters the list and changes nothing else.",
    ],
    controls: [
      "`Show` toggles between every line and the kept ones. The leak switch adds or removes the copied sentence.",
      "The four switches and `Minimum line length` on `Cleaning stages` rewrite every verdict live.",
      "Comparison worth running: with `Show` on `All lines`, toggle `Drop near-duplicates`. The harbour line flips to `near-duplicate · J 0.92`.",
    ],
    notice: [
      "`copyright 2019 all rights reserved.` appears four times, but with `Strip boilerplate` on every copy is tagged boilerplate first, so `Drop exact duplicates` only catches the repeated fog sentence and reads −1. Turn boilerplate off and it reads −7.",
      "The harbour spelling differs from the kept harbor line by one byte, so exact dedup keeps it. Only the near-duplicate stage removes it.",
      "`ok.` and `yes.` survive every default stage because they are not boilerplate and `Minimum line length` starts at off.",
    ],
    limits: [
      "In this lab: the twenty lines are a fixed teaching page, not a crawl. You cannot paste a URL, and the boilerplate list is five English phrases.",
      "In this lab: near-duplicate detection computes exact trigram Jaccard for every pair, which is only possible because there are twenty lines. The MinHash readout on `Cleaning stages` shows the estimate a real pipeline would use instead.",
      "In general: production cleaning also covers language ID, quality classifiers, toxicity filters, licence and PII stripping, and fuzzy dedup across billions of documents. A line-level switch list is the shape of that pipeline, not its scale.",
    ],
  },

  "Validation loss for both datasets": {
    title: "What the cleaning actually bought",
    summary:
      "Two real training runs — one on the raw scrape, one on whatever the current pipeline kept — scored at every checkpoint on the same three held-out sentences, plus a contamination check that searches the cleaned set for text copied from those sentences.",
    whatYouSee: [
      "A chart whose x axis is training progress from 0 to 1 and whose y axis is held-out loss in nats per token, pinned from 1.50 to 3.50.",
      "Two series named with their training-pair counts: dashed `raw (N pairs)` and solid `cleaned (N pairs)`.",
      "Three metrics: `Raw held-out`, `Cleaned held-out`, and `Difference`, signed with a minus when cleaning wins.",
      "A contamination check: the longest run of words the cleaned training set shares with the held-out text, the run itself, and a flag at 8 words or more. With the leak on it also prints the cleaned model's loss on the two held-out sentences the leak did not copy.",
    ],
    howItWorks: [
      "Each run is minibatch SGD on a character bigram, 25 epochs, batch 16, learning rate 0.6, seed 4, seven checkpoints. The only difference is the training text.",
      "Every plotted point is `tinyCrossEntropy` of that checkpoint's weights on the three held-out sentences, not the batch loss of the run.",
      "If cleaning removes every line, the cleaned run is trained on the raw text instead of an empty string, so the two curves coincide rather than leaving a missing number.",
      "The shared run is found by comparing every word position in the cleaned text with every position in the held-out text. Full stops are ignored.",
    ],
    controls: [
      "This card has no controls of its own. Every switch, the leak row, and `Minimum line length` retrain both runs.",
      "Comparison worth running: with the defaults read `Cleaned held-out` (2.300), turn on the leak (2.245), then read the untouched-sentence line (2.422, up from 2.397). Turn on `Decontaminate` and it returns to 2.300.",
    ],
    notice: [
      "A lower cleaned curve is not automatic. Dropping useful sentences can raise held-out loss, which is why the Difference metric is signed.",
      "The leak makes the held-out score better and the model worse on everything it did not copy. That is leakage: a better number without a better model.",
      "Without the leak the longest shared run is two words, `the morning`. Ordinary shared phrases are short; a copied sentence is long, which is why n-gram decontamination works.",
    ],
    limits: [
      "In this lab: both losses are genuinely computed, but the held-out set is three sentences and each run is one seed. Across seeds 1–8 the raw-to-cleaned gap ranges from 0.065 to 0.085, while gaps between the good settings stay under 0.01 and are unresolved.",
      "In this lab: the split bar on `Dataset statistics` is a display. It does not create the held-out text this chart uses.",
      "In general: a real cleaning win is measured on held-out distributions you care about, across seeds, against the cost of the tokens you threw away. Contamination checks run against every benchmark, and still miss paraphrases.",
    ],
  },

  "Survival funnel": {
    title: "How much of the scrape survives each stage",
    summary:
      "The same verdicts as the line list, counted stage by stage in pipeline order: how many lines and characters are left after each filter, and how much each one removed.",
    whatYouSee: [
      "One row per stage: Scraped, Strip boilerplate, Length floor, Exact dedup, Near-dup, Decontaminate. A centred bar's width is the number of lines left after that stage.",
      "Where a stage removed lines, a hatched ghost bar behind it shows the width before the stage.",
      "On the right, lines and characters left, and underneath the stage's own removal (`−9 lines · −319 chars`), `removes nothing here`, or `stage off`.",
    ],
    howItWorks: [
      "Rows are computed from the verdict list: each stage removes exactly the lines tagged with its verdict, so the removals always add up to the difference between the first and last rows.",
      "Characters are counted after `encodeTinyText`, the model's 30-symbol encoding, so the digits in `copyright 2019` do not count and the numbers match the `Characters` metric.",
    ],
    controls: [
      "The funnel follows every switch, the leak row, and `Minimum line length`.",
      "Comparison worth running: at the defaults, the funnel reads 20, 11, 11, 10, 10, 10 lines and 899 to 516 characters. Turn off `Strip boilerplate` and exact dedup takes 7 lines instead of 1.",
    ],
    notice: [
      "Boilerplate removes 9 of 20 lines but only 319 of 899 characters: the lines it removes are short. Counting documents and counting tokens tell different stories.",
      "A stage that is on but removes nothing still appears as a full-width row. That is not a bug: the earlier stages already took its targets.",
    ],
    limits: [
      "In this lab: twenty lines, five stages, and one order. The funnel cannot reorder stages.",
      "In general: published pipelines report funnels like this one in tokens, and the steepest drops usually come from language, quality, and deduplication filters, often removing most of a raw web crawl.",
    ],
  },

  "Cleaning stages": {
    title: "Toggle a filter and watch the verdicts move",
    summary:
      "Four pipeline switches plus a length floor. Each switch is live: it rewrites every verdict, both training runs, the funnel, and every statistic on the page. A note compares exact Jaccard with a MinHash estimate for the spelling-variant pair.",
    whatYouSee: [
      "Four pressable rows: `Strip boilerplate`, `Drop exact duplicates`, `Drop near-duplicates`, and `Decontaminate`, each with a one-line rule and a count reading `−N lines` when on or `off` when off.",
      "`Minimum line length` from 0 (`off`) to 60 characters.",
      "A note giving the harbour/harbor pair's exact trigram Jaccard (0.915) and a 64-hash MinHash estimate of it (0.922).",
    ],
    howItWorks: [
      "`Strip boilerplate` drops a line containing any of five phrases. `Drop exact duplicates` keeps the first surviving copy of a byte-identical line.",
      "`Drop near-duplicates` tags a line when its character-trigram Jaccard with an already-kept line is at least 0.7. Jaccard is shared trigrams divided by the union of both trigram sets.",
      "`Decontaminate` tags a line that shares any 8-word run with the held-out text. It runs last.",
      "MinHash hashes every trigram with 64 different hash functions and keeps each function's minimum. Two lines agree at a position with probability equal to their Jaccard, so the share of agreeing positions estimates it with error shrinking like one over the square root of the signature length.",
    ],
    controls: [
      "Each switch toggles that filter; the pressed state is `aria-pressed`.",
      "`Minimum line length` at 12 drops `ok.` and `yes.`. The shortest real sentence is 56 characters, so the floor starts eating real text at 57.",
      "Comparison worth running: leave boilerplate and exact-repeat on, then turn near-duplicates on and off while watching `Lines kept` and `Cleaned held-out` together.",
    ],
    notice: [
      "The `−N lines` count is what that stage tagged, not what it would have tagged if earlier stages had not already removed lines.",
      "Order matters. Boilerplate runs first, so most copyright lines never become exact-repeat candidates.",
      "`Decontaminate` reads −0 without the leak: no ordinary line shares 8 words with the held-out text.",
      "Turning every switch off and setting length to 0 makes the cleaned corpus identical to the raw one, the control the comparison chart needs.",
    ],
    limits: [
      "In this lab: four binary filters and a length floor are the entire cleaner. There is no language detector, no quality classifier, and no way to write your own rule. The MinHash note is computed but the near-duplicate stage itself uses exact Jaccard.",
      "In general: a production pipeline is this idea at corpus scale — cheap exact checks first, MinHash with locality-sensitive hashing for near-duplicates, n-gram decontamination against every benchmark, with every dropped document logged so someone can audit why it left.",
    ],
  },

  "Dataset statistics": {
    title: "Size, diversity, and a split that is only a picture",
    summary:
      "How much text survived, how repetitive it is, and a three-way split bar drawn by line. The bar is a teaching picture of a split; the validation loss chart does not use it.",
    whatYouSee: [
      "A first metric row: `Lines kept` as kept / scraped, `Characters` after encoding into the 30-symbol vocabulary, and `Training pairs` from the cleaned run.",
      "A second row: `Distinct trigrams` in the cleaned text, `Repetition` as one minus distinct/total trigrams, and `Raw repetition` of the unfiltered page.",
      "A split bar labelled `train N`, `val N`, and `test N`, sized by a 70% / 15% / remainder split of the kept lines, with validation getting at least one line.",
    ],
    howItWorks: [
      "A trigram here is three consecutive characters. Distinct is the number of unique ones; total is `max(1, length − 2)`; repetition is `1 − distinct / total`.",
      "`Characters` uses `encodeTinyText`, so case folds, unknown characters such as digits become spaces, and runs of spaces collapse. It will not match the sum of the raw line lengths.",
      "The split is `round(0.7n)` train, `max(1, round(0.15n))` validation, and the remainder test. It is not applied to either trainer.",
    ],
    controls: [
      "The card has no controls; it reports the current pipeline.",
      "Comparison worth running: read `Repetition` with defaults (45%), turn every switch off (60%), then turn only `Strip boilerplate` on (51%).",
    ],
    notice: [
      "`Repetition` uses the loss tone when the cleaned text is more repetitive than the raw page, which can happen if you drop the diverse sentences and keep the furniture.",
      "At the defaults `Distinct trigrams` falls from 366 to 285 while repetition falls from 60% to 45%: fewer distinct sequences and less repetition at once.",
      "If you keep one line, the bar still draws a validation slice of 1 because of the `max(1, …)` floor — do not treat that as a usable split.",
    ],
    limits: [
      "In this lab: the split bar never trains or scores a model. The only held-out number on the page is the three-sentence passage on the chart.",
      "In this lab: trigram diversity is a character-level toy. It calls a page diverse for using many letter sequences, not for covering many topics.",
      "In general: a real split is by document, not by line, and the test set is locked before you tune cleaning. Scoring on a passage written to resemble the training domain, as this page does, is closer to a development set than to a blind test.",
    ],
  },

  "Regurgitation probe": {
    title: "More context, and a model recites what it was trained on",
    summary:
      "Two count-based character models, one built from the raw scrape and one from whatever the current pipeline kept. Each starts from the opening characters of its own training text, writes 300 characters, and is scored on how much of that output is a verbatim copy. The `Context length` slider is the only thing that changes how much each model can copy.",
    whatYouSee: [
      "A `Context length` slider from 1 to 12 characters, and a badge that repeats it. At 1 character the model is the same bigram the rest of the track trains.",
      "Two columns, `Trained on the raw scrape` and `Trained on the cleaned set`. Each prints the first 170 characters of its sample and three metrics: `Copied verbatim (16+ chars)`, `Longest copied run`, and `Footer text in the sample`.",
      "A line under each sample saying how many of that model's contexts have only one possible next character, in which case it has no choice but to continue the text it saw.",
      "A table with the same sweep at context lengths 1, 2, 3, 4, 5, 6, 8, and 12: the share copied verbatim for each model, and the share of the raw model's sample that is footer text. The row for the current slider value is shaded.",
    ],
    howItWorks: [
      "The model is a table of counts, not a trained network. For a context length `k`, every run of `k` characters in the training text records which character came next. Sampling draws the next character in proportion to those counts, seeded with 7, after being given the first `k` characters of the training text.",
      "Text is encoded as the trainer sees it: lowercase, 30 symbols, runs of spaces collapsed. That makes the raw scrape 899 characters and the cleaned default set 516.",
      "A sample character counts as copied when it lies inside a 16-character window that appears verbatim in the training text. `Longest copied run` is the longest verbatim stretch of any length. `Footer text` counts the characters that belong to one of the five boilerplate phrases the cleaning stage looks for.",
    ],
    controls: [
      "`Context length` is the only control. The raw and cleaned columns also follow every switch on `Cleaning stages`, the leak row, and `Minimum line length`, because the cleaned set is rebuilt from them.",
      "Comparison worth running: at a context of 1, both samples are gibberish and 0% is copied. At 3, the raw model copies 52% and the cleaned one 39%. At 4, 98% and 93%. At 8, both copy 100%.",
    ],
    notice: [
      "Copying switches on over a narrow range. Nothing is copied at 1 or 2 characters of context, and by 5 almost every character is. The training text is only 516 to 899 characters long, so there is almost never a second continuation to choose between.",
      "The raw model writes the footer. At a context of 8, 21% of its sample is `copyright`, `subscribe`, and `share this page` text, and the cleaned model's is 0%. Repeated lines carry more counts, so they are sampled more often.",
      "The longest copied run is a stretch of one training line or several joined, so it can pass 160 characters in the raw sample at a context of 8, longer than any single scraped line.",
      "Both columns copy almost everything at a long context. Deduplication changes which text is recited and how often, not whether a model with enough context recites at all.",
    ],
    limits: [
      "In this lab: the model is a table of counts over a 516 to 899 character text, so every context longer than a few characters is unique and memorization is guaranteed. A real model with billions of parameters generalizes where this table cannot, and memorizes only part of what it saw.",
      "In this lab: the sample starts from the first characters of the training text, as an extraction attempt starts from a prefix. A random prompt would copy less. The seed is fixed, so one context length always prints the same text.",
      "In general: published work finds memorization growing log-linearly with model capacity, with how often a text was duplicated, and with the number of tokens of context used to prompt the model (Carlini et al., 2022). Deduplicating training data cut emitted memorized text about tenfold in one study (Lee et al., 2021), but sequences that appear in a single document have still been extracted, including names, phone numbers, and email addresses (Carlini et al., 2021).",
    ],
  },
};

export default cardInfo;
