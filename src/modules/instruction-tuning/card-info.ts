import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Chat template editor": {
    title: "The exact characters the model reads, and which ones it is graded on",
    summary:
      "Three switches wrap each instruction–response pair in the reserved markers `<system>`, `<user>`, `<assistant>` and `<end>`. `Loss on` decides which characters count in the loss. The strip draws the first example one character at a time, marks each as graded or context only, and shows how surprised the trained model is by it.",
    whatYouSee: [
      "A kicker counting templated training characters and a badge `N examples`.",
      "Three pressable rows — `System block`, `Role markers`, `End-of-turn marker` — each with an `in template` or `off` chip, and `Loss on` with `Assistant reply only` and `Every character`.",
      "The strip: one cell per character. A top rule marks the turn it belongs to (system dotted, user dashed, assistant solid). Graded cells are plain with a blue underline; context-only cells are hatched and greyed. Each cell's bar is −log p of that character under the trained templated model, from 0 to 5 nats.",
      "Three metrics: how many of the example's characters are graded, and the mean −log p of the graded and the context-only characters.",
    ],
    howItWorks: [
      "Segments are concatenated with markers against the text. If a switch removes the marker between two text segments, one space is inserted so words never fuse.",
      "With `Assistant reply only`, a character is graded when it belongs to the reply or to the `<end>` that closes it. Every other character is still read as the context for the character after it, but its own prediction is not in the loss.",
      "Each example is its own sequence, so no transition runs from one example into the next. Training is minibatch SGD from a zeroed 30 × 30 table on the graded transitions only: batch 16, learning rate 0.6, seed 6.",
      "Reserved characters `<` and `>` are stripped from anything typed in the dataset editor, so typed text cannot impersonate a marker.",
    ],
    controls: [
      "The three switches change the string, and `Loss on` changes the grading. Both retrain the two models.",
      "Comparison worth running: at the defaults, 52 of the first example's 137 predicted characters are graded, and the context-only ones average 2.92 nats against 2.11 for the graded ones. Switch to `Every character` and all 137 are graded, averaging 1.99.",
    ],
    notice: [
      "Under the mask, the prompt's markers and user text stay surprising. The model is never taught to write them, which is the point: it should produce replies, not user turns.",
      "The closing `<end>` is graded. That is how the model learns to stop.",
      "The first cell has no bar, because nothing precedes it to predict it.",
    ],
    limits: [
      "In this lab: markers are spelled in characters, so the letters of `<end>` share weight rows with ordinary text. A production template uses dedicated token ids, one per marker.",
      "In this lab: the system prompt is one fixed sentence, and the strip shows only the first parsed pair.",
      "In this lab: the switches change the strings the models train on. `Serve the trained models` then serves those weights other strings, but a one-character model hides most of what a template mismatch does; that card's limits say how.",
      "In general: a chat template is a contract between training and serving. A model trained with one set of markers will not stop or switch speakers reliably when served with another.",
    ],
  },

  "Two models, two formats": {
    title: "The same pairs, with and without the template",
    summary:
      "Two character bigrams trained on the same pairs, the same budget and the same `Loss on` rule: one on the templated sequences, one on `instruction + space + response`. Training loss, a per-character reply score and two samples sit side by side.",
    whatYouSee: [
      "A chart of optimizer step against batch loss on the graded characters. Solid `with template`, dashed `no template`, y from 0.60 to about 3.50, with a footnote giving each run's graded-target count.",
      "Three metrics: `Reply log-prob per char, templated`, `Reply log-prob per char, plain`, and the probability of `<` after the reply's last character for each model.",
      "Two samples at temperature 0.6, length 70, seed 17: the templated model continuing everything before the reply (markers shown), and the plain model continuing `instruction + space`.",
    ],
    howItWorks: [
      "Reply log-prob is the mean of log p over the reply characters of the first example, each conditioned on the character before it in its own format. Both models are scored on the same characters and divided by the same count.",
      "Under `Assistant reply only` the plain model is graded on its reply characters and the templated model on its reply plus `<end>`: 191 against 171 targets at the defaults.",
      "The samples use the SDK's sampler; the plain one suppresses reserved characters because that model never saw them.",
    ],
    controls: [
      "The template switches, `Loss on`, the dataset and `SFT epochs` all retrain both runs.",
      "Comparison worth running: with all markers on, read the reply scores under `Assistant reply only` (−2.228 against −2.168), then under `Every character` (−2.425 against −2.129). Grading the prompt costs the templated model more of its reply.",
    ],
    notice: [
      "The templated model scores the reply slightly lower even with the mask, because the letters of `<assistant>` and `<end>` compete with the reply's letters for the same rows.",
      "The curves train on different targets, so a lower curve is not a better assistant.",
      "The number that separates the formats is the `<` probability: 54% against 3% at the defaults.",
    ],
    limits: [
      "In this lab: both runs start from a zeroed table and see only these pairs. There is no pretrained base and no held-out instruction set.",
      "In this lab: the reply score covers one example of a bigram model, so it measures character statistics, not helpfulness.",
      "In general: SFT starts from a pretrained model and uses thousands to millions of pairs. The template is still what turns completions into a conversation.",
    ],
  },

  "Supervised dataset editor": {
    title: "One instruction and one response per line",
    summary:
      "The pairs both models train on and how many passes they get, with the templated character count and the number of graded targets under the current mask.",
    whatYouSee: [
      "A textarea labelled `instruction | response`.",
      "`SFT epochs` from 1 to 80 passes.",
      "A note on the size limit: up to 16 lines of 200 characters each. Text past either limit is ignored.",
      "Three metrics: `Examples` parsed, `Templated chars` after encoding, and `Graded targets` for the templated run.",
    ],
    howItWorks: [
      "Each line splits on the first `|`: the left side is the instruction and everything after is the response. Both are lowercased and stripped of `<` and `>`. Lines with an empty side are dropped.",
      "Encoding follows the SDK: unknown characters become a space and runs of spaces collapse, so digits and commas disappear.",
      "If no line parses, both trainers fall back to the single pair `the | the.` so nothing divides by zero.",
      "Training runs inside the page, so the editor and any shared link are cut to 16 lines of 200 characters before parsing.",
    ],
    controls: [
      "Edit the textarea to change the pairs. Raise `SFT epochs` to fit them harder.",
      "Comparison worth running: at 4 epochs the `<` probability after the full stop is 6%; at 30 it is 54%; at 80 it is 88%. The boundary is learned gradually, like any other statistic.",
    ],
    notice: [
      "Graded targets change with `Loss on`: 191 under the mask, 528 with every character graded.",
      "A line without a `|` silently drops out of `Examples`.",
      "Four short pairs are enough to teach a turn boundary and not enough to teach a useful assistant.",
    ],
    limits: [
      "In this lab: the pairs are the whole dataset, both models start from zero, and nothing is held out.",
      "In this lab: every example is short enough to be its own sequence, so there is no packing and no attention mask between packed examples.",
      "In general: supervised instruction data is necessary and not sufficient. Preference optimization, the next module, ranks two already-formatted replies.",
    ],
  },

  "What follows the end of a turn": {
    title: "Whether the model knows a reply just finished",
    summary:
      "The next-character distribution after the last character of the first reply, from both models. With `<end>` in the template, the templated model piles probability on the reserved `<`; the plain model has never seen anything follow a reply.",
    whatYouSee: [
      "A kicker naming the boundary character, a full stop for the default pairs.",
      "Two bar lists of the top five next characters: the templated model above, the plain model below, each with its probability.",
      "A note that states what each model saw after that character in training.",
    ],
    howItWorks: [
      "Both lists softmax one row of the trained table: the row for the reply's last character.",
      "Each example is its own sequence, so the plain model's full-stop row receives no gradient and stays uniform, 1/30 per character, unless a full stop also appears mid-reply.",
      "Under `Assistant reply only`, only a graded `<end>` can train the full-stop row of the templated model. Turn the end marker off and that row is untrained too.",
    ],
    controls: [
      "No knobs of its own. The switches, `Loss on`, the pairs and `SFT epochs` retrain both distributions.",
      "Comparison worth running: at the defaults the templated `<` reads 54.4%. Turn `End-of-turn marker` off and it falls to 3.3%, the uniform value. With `Every character` and the end marker off it stays at 55%, learned from the system prompt's full stop followed by `<user>`.",
    ],
    notice: [
      "A high `<` bar means the model wants to start a marker, not that it will spell `<end>` correctly. The rest of the marker is predicted character by character.",
      "The masked result is cleaner: the model learns end-of-reply from replies only, not from the system text.",
      "If the first reply does not end in a full stop, the probe moves to its last character.",
    ],
    limits: [
      "In this lab: the probe sees one character of context. A real stop token is one id predicted from the whole conversation.",
      "In this lab: only the first pair's boundary is probed.",
      "In general: stopping depends on the template and the serving engine together. If the stop id is missing from the engine's stop list, the model runs on into an invented next turn.",
    ],
  },

  "Serve the trained models": {
    title: "The same weights, served a string they did not train on",
    summary:
      "Both trained tables are probed with one serving string at a time: the chat template, the bare instruction, or the template with the user marker where the assistant marker belongs. Each model reads one character, so what changes is the table row of the last character sent. Next-character bars, a seeded sample cut at the first marker, and two readouts show what that row holds.",
    whatYouSee: [
      "`Serve with`, a three-way switch: `Chat template`, `No template` and `Wrong role marker`.",
      "A box with the exact string sent, its last character marked, and a line naming the one character each model reads.",
      "Two samples, templated above and plain below, at temperature 0.6, length 90 and seed 17 with markers shown. Each has a note on where its first marker falls, because a marker is the stop signal.",
      "Two bar lists of the five likeliest next characters, one per model, for the served string.",
      "Two readouts written `templated vs plain`: the probability of the first reply character, and the log-probability per character of a held-out reply.",
    ],
    howItWorks: [
      "The served string is the training prompt cut before the first reply character: `renderExample` with a stand-in reply, so the system block, role markers and closing marker follow the switches. `No template` is the instruction and one space. `Wrong role marker` replaces `<assistant>` with `<user>`, or appends `<user>` when role markers are off.",
      "The bars are the softmax row of the string's last character. The first-reply probability reads that row at the reply's first character. The held-out score is the mean log-probability of `the rain stops before noon.` written after the served string, computed by the same function as `Reply log-prob per char`.",
      "Both samples use the SDK sampler with reserved characters allowed and the same seed, so any difference comes from the weights and the string.",
    ],
    controls: [
      "`Serve with` changes only the string. The template switches, `Loss on`, the pairs and `SFT epochs` retrain both models first.",
      "Comparison worth running: at the defaults the first reply character `t` gets 18% from the templated model and 3% from the plain model under `Chat template`, and 22% and 25% under `No template`. The plain model, served a bracket it never saw, spreads probability evenly.",
      "Then set `Loss on` to `Every character` and serve the chat template: the templated model puts 37% on `<` after the bracket, because grading the prompt taught it that a bracket is followed by a marker.",
    ],
    notice: [
      "`Chat template` and `Wrong role marker` give identical bars and samples: both strings end in a closing bracket, which is all either model reads.",
      "Dropping the template does not hurt the templated model here, because its space row was trained on the replies' own words. That is a property of the toy, not of templates.",
      "The held-out reply scores −2.86 per character under the templated model, against −2.23 for the first trained reply: the gap between text a model trained on and text it did not.",
    ],
    limits: [
      "In this lab: each model reads one character of context, so a served string matters only through its last character. The role named by a marker, the system prompt and the instruction itself are invisible to both models, and the cost of reading a whole unfamiliar prompt cannot appear.",
      "In this lab: the first dataset line supplies the instruction and the reply behind the first-character readout, the held-out reply is one fixed sentence, and each sample is one seed at one temperature.",
      "In general: a template mismatch changes the context of every position, not one row. It is silent: nothing errors, and the model continues an unfamiliar string and answers a little worse. A stack should render the template from the model's own tokenizer configuration.",
    ],
  },
};

export default cardInfo;
