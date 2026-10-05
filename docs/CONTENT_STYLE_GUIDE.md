# Content style guide

Every explanation helps a learner predict the interactive, test that prediction, and name the limits of the result. Do not market the technology or use mystery as a substitute for mechanism.

## Shared rules

- Lead with the mechanism, not history or importance.
- Keep paragraphs under 80 words and sentences concrete.
- Introduce a term only when it lets the learner notice something more precisely.
- Tie claims to a visible control, value, or change in the lab.
- Say when a diagram is a projection, simplification, proxy, or toy model.
- Separate correlation, attribution, and causal evidence.
- Never describe sampling temperature as knowledge, intelligence, or creativity.
- Never treat an attention map or feature label as a complete explanation.

## The seven sections

**The big picture** opens every lesson, in both modes, before `## What it is`. It gives a complete
overview: what the module is about, where the mechanism shows up, why it matters, what the reader
will be able to explain by the end (every objective), and what the lab runs and lets them try,
including which parts are measured, closed-form, fixed, or synthetic. It does not teach the
mechanism in detail; that is the job of the sections after it. Standard writes it in the Standard
register, usually with the subheadings `### Where it shows up`, `### Why it matters`,
`### What you will be able to explain`, and `### What the lab shows`.

**What it is** gives the smallest accurate model of the mechanism.

**Why it’s here** names the problem this mechanism solves in the larger system. It goes deeper than
the overview without repeating it.

**How to play with it** points to a control and a deliberate comparison.

**What to notice** names the signal that distinguishes outcomes.

**Where it breaks** states at least one concrete failure or interpretive limit.

**Going deeper** connects the toy to the production mechanism without requiring the learner to follow the extra detail.

Depth is added by naming more sub-mechanisms, not by lengthening paragraphs. Use `###`
subheadings and lists to keep a long section scannable, and prefer a specific number,
formula, or failure over a general statement of importance.

`Where it breaks` carries two separate obligations: the limits of the interactive, including
which values are synthetic, and the limits of the real mechanism. Do not let one stand in for
the other.

## Card explanations

Each lab card carries its own info panel, authored in the module's `card-info.ts`. The
panel answers the same obligations as the lesson sections, scoped to one interactive:
`summary` is the smallest accurate model, `whatYouSee` decodes every axis, mark, and
readout, `howItWorks` states the formula the code actually evaluates, `controls` names
each control as the interface labels it, `notice` names the distinguishing signal, and
`limits` splits the interactive's limits from the mechanism's.

Because a card explanation sits directly on top of the visualization, it is the most
likely place for a learner to mistake a demonstration for inference. Name the synthetic
parts there, not only in the lesson: which readouts are closed-form, which strings are
fixed, and whether a value was measured or predicted.

## Standard mode

Use the accepted technical vocabulary, compact equations where useful, and precise qualifiers. Define specialized terms near first use. Standard does not mean academic or needlessly formal.

Good: “Softmax converts query-key scores into positive weights that sum to one for each query.”

Avoid: “Softmax is a normalization paradigm facilitating probabilistic attention allocation.”

## Plain mode

Plain mode covers the same objectives and limitations as Standard, written at a grade 8 to 10
reading level. It adds:

- one analogy that preserves the important structure;
- one bold **Try this:** nudge attached to a visible control;
- one bold **A common mix-up:** correction;
- concrete verbs, and terms explained before they are used.

Explain where an analogy stops matching. Do not replace a technical term that the interface displays; introduce it after the plain description.

Good: “Each word writes a search request, checks every other word’s label, then borrows information from the best matches.”

Avoid: “The words talk to each other magically.”

In Plain, **The big picture** is the easiest part of the lesson to read: everyday words, and
the objectives restated as things the reader will be able to explain.

### The grade 8 to 10 reader

Write for a curious student in grades 8 to 10, about 13 to 16 years old. Assume they know
arithmetic, fractions, decimals, percentages, negative numbers, averages, squares and square roots,
a letter standing for a number in a simple formula, and how to read a simple graph, and that they
have used a phone, a search engine, games, and maybe a chatbot. Do not assume logarithms, vectors,
matrices, calculus, statistics, programming, computer hardware, or any AI vocabulary.

- **Explain a word before you lean on it.** Describe the idea in everyday words first, then name
  it: “a list of numbers, called a **vector**”. Keep terms the interface or glossary shows; drop
  jargon the learner never needs.
- **Write connected sentences.** Link ideas with *because*, *so*, *which* and *while* instead of
  chopping every idea into its own short sentence. Most sentences run 12 to 25 words, and none
  runs past 30. Reach the reading level through fuller sentences, not through long words.
- **Use everyday words.** “Use”, not “utilize”; “about”, not “approximately”. Avoid *i.e.*,
  *e.g.*, *via*, *respectively*, *thereby*, and *vs.*
- **Make every number mean something.** Say what it measures in everyday terms (“99% sure”,
  “about 1 in 4”). Avoid stacking more than two numbers in a sentence. Explain a unit such as bits
  or GiB the first time it appears.
- **Show the math with small numbers.** Use a formula only if the lab shows it, and then work one
  example with real values.
- **Give a concrete example for every core idea**: a real sentence, a small worked number, or a
  step-by-step walk through the lab.
- **Use short anecdotes where they help an idea stick**: a test, a team, a game, a recipe, a
  group chat. Frame them as made up (“Picture a friend who…”). Never invent real people,
  events, quotes, or statistics; real-world facts come from the Standard lesson, where they are
  sourced.
- **Simplify without saying anything false.** Drop a minor detail rather than bend it. Plain and
  Standard never contradict each other, and figures pinned by a module's `quoted.test.ts` stay.
- **Talk to the reader, not down to them.** Use “you”. Skip “simply”, “just”, “obviously” and
  “easy”.
- **Take the room the reader needs.** A Plain lesson may run longer than Standard. Never cut a
  step, an example or a story to save words; never add a sentence that does not help.

`node scripts/readability.mjs <slug>` reports a lesson's Flesch-Kincaid grade, its long
sentences, and its most-used long words. `npm run check:readability` fails outside grades 8 to
10 or on any sentence over 35 words. The grade counts syllables and sentence length, not
unexplained ideas, so a lesson inside the band can still lose its reader: read it as the student
would.

## Checkpoints

Ask for a prediction, distinction, or causal explanation—not trivia. Distractors should represent plausible misconceptions already addressed in the lesson. Feedback explains the mechanism even when the answer is correct.

Each module carries at least three questions and every objective is tested by at least one. Beyond that:

- Prefer **"what will this control do?"** questions answerable from the lab (a number going up or down, a bar changing) over recall of a definition. Where a lab produces a measured number, name the comparison, not the value, so the question survives a retrain.
- Write each distractor as a misconception a learner could genuinely hold, and let the explanation say what that misconception gets wrong.
- Keep options similar in length and grammatical shape. The correct option must not be the longest more often than not; `registry.test.ts` checks this across every module that has migrated.
- Never write an option that refers to another option ("both A and B", "all of the above"): options are shuffled when shown.
- At least one question per module should test **where the mechanism or the lab breaks**, not only how it works.
- Spread questions across the objectives rather than testing the same idea three ways.

One checkpoint is the minimum. A module may add step-level checks, but scores should remain interpretable and retryable.

## Review

Before merging content, verify:

- both modes state the same facts;
- the plain analogy does not introduce a false mechanism;
- every “it” has an obvious referent;
- failure modes are specific;
- chart colours are never the only named distinction;
- UI labels and lesson terms use the same wording;
- claims remain true at both control extremes;
- every lab card has an explanation, and it names what is synthetic.
