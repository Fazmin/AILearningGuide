import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Grammar": {
    title: "Which language the mask enforces",
    summary:
      "Grammar picks one of two small output languages: a JSON object that must match a schema, or a Python-style tool call. The lab compiles the chosen pattern into a character-level matcher, and that matcher, not a list of hand-picked tokens, decides which tokens are legal at every step.",
    whatYouSee: [
      "`Grammar`, a two-way switch between `JSON object` and `Function call`. Switching resets `Decode step` to the first token.",
      "`Prompt`, the request the toy model is answering; `Allowed output`, the strings the grammar accepts; `Rule`, where the pattern comes from; `Vocabulary`, the 24 tokens the model can emit.",
    ],
    howItWorks: [
      "`JSON object` accepts exactly `{\"ok\":true}` and `{\"ok\":false}`: a JSON Schema with one required boolean key and no others, with whitespace dropped to keep the toy small. `Function call` accepts `get_weather(city=\"…\")` where the city is one or more letters.",
      "The pattern becomes a small automaton over characters. A token is legal when the text so far plus that token is still a prefix of some accepted string; end of sequence is legal only when the text is complete.",
    ],
    controls: [
      "`Grammar` is the only control here. Compare the two at step 1: the JSON walk allows `{\"` and `{`, the call walk allows only `get`.",
    ],
    notice: [
      "The JSON schema is stricter than JSON syntax: `null` and a string are valid JSON after the colon but illegal for a boolean field.",
      "The call's free string field accepts any letters, so `hello` or `Sure` would be a legal city. Parseable is not the same as correct.",
    ],
    limits: [
      "In this lab: two tiny patterns with a finite, hand-picked vocabulary. There is no JSON Schema compiler for arbitrary schemas, and no whitespace, nesting or escapes.",
      "In general: production engines compile full JSON Schema or context-free grammars into automata and precompute which vocabulary tokens each state allows, because checking 100,000 tokens character by character at every step would be too slow.",
    ],
  },

  "Prefix walk": {
    title: "The mask depends on the prefix",
    summary:
      "The walk is the greedy constrained decode: at each step the lab takes the most probable legal token. The emitted text so far is the prefix; the matcher's state after that prefix is what decides the next mask. Unconstrained pick shows what the model would have emitted without the mask.",
    whatYouSee: [
      "The emitted tokens as chips, followed by a dashed `?` for the position being decided. `Grammar accepts next` states the matcher's state in words, such as `true or false` or `more letters or \")`.",
      "`Decode step`, a step controller whose buttons are the tokens of the walk, including the end-of-sequence token `⟨eos⟩`.",
      "`Unconstrained pick` (the model's top token before masking, marked ✗ if illegal), `Pick after mask`, and `Legal tokens` out of 24.",
    ],
    howItWorks: [
      "Each step's model distribution is a softmax over authored logits. The mask is computed by running prefix plus token through the matcher, and the pick is the largest probability that survives.",
      "A test checks that the authored walk is exactly this greedy path, so the buttons are not a canned script.",
    ],
    controls: [
      "`Decode step`: at step 1 of the JSON walk the model's favourite is `Sure` (41.5%), which is masked, so the pick is `{\"`.",
      "At the last step of either walk only `⟨eos⟩` is legal: the grammar also decides when generation stops.",
    ],
    notice: [
      "The number of legal tokens swings from 1 to 13. Inside a free string almost every word-like token is legal; at punctuation often only one is.",
      "The model's top choice and the pick agree on most steps. Constrained decoding works best when the model already wants to produce the format, which is why schemas are also described in the prompt.",
    ],
    limits: [
      "In this lab: the walk always follows the greedy pick, and the logits are authored per step rather than produced by a trained model.",
      "In general: the mask is applied to whatever the model proposes. It does not teach the model the schema, and a model that has never seen the format can be forced through it into low-probability, low-quality text.",
    ],
  },

  "Masked distribution": {
    title: "Zero the illegal mass, then renormalize",
    summary:
      "The model's next-token probabilities before and after the mask. Illegal tokens drop to zero, and each legal token's probability is divided by Z, the total probability the legal tokens had. The result sums to one and keeps the model's ratios between legal tokens.",
    whatYouSee: [
      "Rows sorted by the model's probability. `model p` is the softmax over all 24 tokens; the bar is hatched red when the token is masked. `mask` says `✓ legal` or `✗ masked`, and a masked token's chip is struck through.",
      "`after mask` is the renormalized probability. The outlined row is the pick. A legal token that spans two grammar symbols carries a `↔` tag naming them, such as `{ and \"ok\"`.",
      "Tokens that are illegal and below 1% are folded into a final `+N more tokens` row with their combined mass.",
      "The formula line evaluates the pick: `p′ = p ÷ Z`. The kicker gives Z and the share of probability removed.",
    ],
    howItWorks: [
      "p = softmax(logits) over the whole vocabulary. Z = Σ p over legal tokens. p′ = p ÷ Z for legal tokens and 0 otherwise. This equals setting illegal logits to −∞ and taking the softmax again.",
      "Legality is checked on characters, so `\":` is legal after `{\"ok` although it closes the key and starts the separator in one token. A mask that only allowed tokens matching whole grammar symbols would forbid it.",
    ],
    controls: [
      "This card has no controls; `Grammar` and `Decode step` set it. At JSON step 1, Z = 0.320: 68% of the model's probability was on illegal tokens, mostly `Sure` and `The`.",
      "At JSON step 4, the model's second choice is `\"` (27.9%), a string, which the boolean field forbids; `false` keeps 21.4% after the mask.",
    ],
    notice: [
      "Renormalization adds no information. Whatever ratio the model had between `true` and `false` is exactly the ratio after the mask.",
      "At function-call step 6, `\")` keeps 6.3%: `get_weather(city=\"Os\")` would parse, run, and ask for the wrong city.",
    ],
    limits: [
      "In this lab: the logits are authored to tell a clear story, a baseline of −2 plus a few hand-set values per step. No trained model runs here.",
      "In general: when most of the mass is removed, the survivors may all be tokens the model found unlikely, so the output can be valid and poor. Mass removed is a useful warning signal.",
    ],
  },

  "Mask distortion": {
    title: "Per-step masking is not conditioning",
    summary:
      "A separate two-step example with an enum field that allows only Oslo or Bergen. Masking token by token commits to a prefix before seeing whether the model would continue it legally, so it samples whole strings in different proportions than the model's own probabilities restricted to valid strings.",
    whatYouSee: [
      "A token tree: the first value token splits into `\"O` 55%, `\"B` 30% and a masked `\"P` 15%. After `\"O` the model continues with `slo\"` or the masked `ttawa\"`; after `\"B`, with `ergen\"` 90% or the masked `ern\"`. Dashed red branches are masked.",
      "`P(slo\" after \"O)`, a slider for the model's probability of the legal continuation after `\"O`.",
      "Two bars: `Per-step mask`, the share of Oslo and Bergen that token-by-token masking produces, and `Model, conditioned on a valid string`, the share the model itself assigns once invalid strings are ruled out.",
    ],
    howItWorks: [
      "Per-step: the first step renormalizes over `\"O` and `\"B`, so Oslo gets 0.55 ÷ 0.85 = 65%. Each branch then has one legal continuation, which renormalizes to 100%. The slider cannot change this.",
      "Conditioned: P(Oslo) = 0.55 × q and P(Bergen) = 0.30 × 0.90, divided by their sum. At q = 10% that is 0.055 ÷ 0.325 = 17% Oslo.",
    ],
    controls: [
      "Drag `P(slo\" after \"O)` from 10% to 90%. The conditioned bar moves from 17% to 65% Oslo and meets the per-step bar, because both branches then keep 90% of their mass legal.",
    ],
    notice: [
      "The distortion comes from branches that differ in how much of their future is legal. The mask sees only one token ahead.",
      "Neither answer is guaranteed right. If the user meant Ottawa, the enum makes every valid output wrong.",
    ],
    limits: [
      "In this lab: the probabilities are authored and the tree has two levels, so the conditioned answer can be computed exactly by enumerating every string.",
      "In general: exact conditioning needs the probability of every legal future, which is intractable. Methods such as grammar-aligned decoding (Park et al., 2024) and sequential Monte Carlo approximate it at extra cost; most production systems accept the distortion.",
    ],
  },
};

export default cardInfo;
