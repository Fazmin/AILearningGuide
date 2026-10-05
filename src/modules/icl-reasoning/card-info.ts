import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Prompt as context": {
    title: "Examples as evidence about the task",
    summary:
      "The literal prompt on the left is all a model gets. On the right, six candidate rules are each run on every example word. A rule that gets any example wrong drops out; the survivors share the weight equally, and their answers for the query become an answer distribution. No weights change anywhere.",
    whatYouSee: [
      "The prompt text: an optional instruction line, the shown examples, and the query line ending in ?.",
      "Examples (0–5), Prompt strategy (Examples only or State a rule), and Query (florin, letter, soravel).",
      "A rule table. Each cell is the rule's output for that example word, with ✓ or ✗. The first ✗ is bold. The next column is what the rule would answer for the query, and Weight is its share, or where it dropped out.",
      "Answer bars for the query; the letter count is highlighted. Metrics: P(query → its letter count), rules still fitting, and weights changed, which is always 0.",
    ],
    howItWorks: [
      "The rules are real functions: count the letters, count distinct letters, consonants + 1, vowels + 3, count consonants, count vowels (a, e, i, o, u).",
      "Posterior = prior × likelihood, normalized. The prior is uniform over six rules, or all on \"count the letters\" when the rule is stated. The likelihood is 1 if a rule reproduces every shown example and 0 otherwise.",
      "The example answers are the true letter counts: glim 4, plint 5, soravel 7, trelix 6, ballot 6.",
    ],
    controls: [
      "Examples adds rows to the prompt in order. Prompt strategy adds or removes the instruction line. Query changes the word being asked about.",
      "Comparison worth running: florin at 0, 1, 2, and 3 examples gives 33%, 50%, 67%, then 100% on 6.",
      "Then Query letter: at 3 or 4 examples it stays 50% on 6 and 50% on 4, because no example word repeats a letter. Ballot, the fifth, settles it.",
    ],
    notice: [
      "More examples help only when they rule something out. Trelix removes nothing that soravel had not already removed.",
      "An answer can be settled before the task is. For florin both surviving rules agree on 6, so the answer is certain while the rule is not.",
    ],
    limits: [
      "In this lab: six hand-written rules, noise-free examples, and a uniform prior. A real model's space of possible tasks is vast and its prior comes from pretraining.",
      "In general: the Bayesian reading of in-context learning is one useful model, not a description of the circuits. Real accuracy also moves with example order, formatting, and label wording.",
    ],
  },

  "Copy from context": {
    title: "A hand-set induction head over the prompt tokens",
    summary:
      "The last token of the prompt is the arrow. The head attends to tokens that came right after an earlier arrow, prefers the one whose line started with the same word, and copies what it finds. For a word already in the prompt that retrieves the answer; for a new word it can only offer every earlier answer.",
    whatYouSee: [
      "The prompt split into word tokens, with ⏎ for line breaks. Shading and the percentage under each token are the attention weight from the final arrow, marked now.",
      "Bars for the head's predicted next token, highlighting the query's letter count.",
      "The heading states whether the query appeared earlier in the prompt.",
    ],
    howItWorks: [
      "Score for position i: 1 if the token before it equals the current token (→), plus 1 more if the token two back equals the query word. All other positions score 0.",
      "Weights are softmax(4 · score) over every earlier position, and the prediction is the attention-weighted vote for the token at each position.",
      "With four examples and the new word florin, each earlier answer gets 24%. With soravel and at least three examples, 7 gets over 90%.",
    ],
    controls: [
      "Query, Examples, and Prompt strategy on Prompt as context.",
      "Comparison worth running: soravel, then florin, at four examples. Then florin at five examples, where 6 rises to 38% only because two earlier answers happen to be 6.",
    ],
    notice: [
      "Copying reproduces the format, a number after the arrow, without computing anything about the new word.",
      "A higher bar for the right answer can be a coincidence of which answers appear in the prompt.",
    ],
    limits: [
      "In this lab: the head's weights are set by hand and there is one head over whole words. Nothing is trained and no real tokenizer runs.",
      "In general: induction heads are a real, well-studied mechanism for copying from context, and their emergence in training coincides with a jump in in-context learning. Inferring a new rule relies on more than this one circuit.",
    ],
  },

  "Chain-of-thought scratchpad": {
    title: "Serial steps that do not fit in one pass",
    summary:
      "The task is to follow a lookup table k times from a start node, where each lookup needs the one before it. In this toy one forward pass can chain at most b lookups, a limit the lab imposes. With the scratchpad off the pass answers with where it got to; with it on, each pass writes the node it reached and the next pass starts from that written token.",
    whatYouSee: [
      "Hops k (1–8), Per-pass budget b (1–8), and Scratchpad (off or on).",
      "The context the model reads: the table and the task, then the tokens written so far. The scratch line holds what the early passes wrote; the answer line is the last token.",
      "One row per forward pass: where it starts (the prompt, or a token read back from the scratchpad), its silent lookups, which are never written down, and the one token it writes, marked scratch or answer.",
      "Answer, Correct answer from the table, and Passes. Each pass writes exactly one token, so passes and tokens written are the same count.",
      "A table of every hop count at the current b: the correct node, the no-scratchpad answer with ✓ or ✗, and the scratchpad answer with its pass count.",
    ],
    howItWorks: [
      "The table is one cycle through ten nodes, C → H → A → J → E → B → G → D → I → F → C, and every task starts at C. A cycle of ten means a wrong answer is never right by coincidence for k up to 8.",
      "A pass starts from C if nothing is written yet, or from the last written token when the scratchpad is on. It counts the tokens written so far, each worth b hops, to know how many lookups remain, chains the smaller of b and what remains, and writes the node it reached.",
      "With the scratchpad off there is one pass: it chains the smaller of b and k and answers with the node it reached. That rule is this toy's choice for a pass that runs out of budget, not a measured behavior.",
      "With it on, a pass answers when its lookups finish the chain and otherwise writes a scratch token, so passes needed = ceil(k / b).",
      "Resuming reads only the written text. A unit test changes one scratch token and confirms the final answer changes with it.",
    ],
    controls: [
      "Hops k, Per-pass budget b, and Scratchpad.",
      "Comparison worth running: k 6 and b 3. Off, the answer is J and wrong, since G is correct. On, pass 1 writes J, pass 2 starts from J and answers G, in 2 passes.",
      "Then raise b to 6. Both settings are right in one pass, so the scratchpad adds nothing.",
    ],
    notice: [
      "The silent lookups are never written. Only the token a pass writes reaches the next one, which is why a partial result has to be written down to be reused.",
      "The scratchpad helps exactly where the chain stops fitting in one pass, and its price is passes: ceil(k / b). At b 3, k 4 to 6 takes 2 passes and k 7 and 8 take 3; at b 1, k 8 takes 8.",
      "Without a scratchpad the answer is right only while k is at most b. Raising b moves the first failing row of the table down.",
    ],
    limits: [
      "In this lab: the per-pass limit b is imposed. It is a number the lab sets, not something measured from a model. The table, the task, and every step are exact and deterministic, no pass ever makes an error, and no model runs.",
      "In this lab: the trace is faithful by construction, because each pass reads only what was written. A real model's visible reasoning carries no such guarantee.",
      "In general: a real transformer's limit per pass comes from its depth and training and is not a clean integer. Theory shows that, under standard assumptions, fixed-depth transformers cannot solve some inherently serial problems in one pass but can with enough chain-of-thought steps (Li et al., 2024).",
      "In general: real traces can contain wrong steps that accumulate, can fail to match what the model computed, and cost latency and tokens. Reasoning models covers sampling, voting, and traces that are trained rather than prompted.",
    ],
  },
};

export default cardInfo;
