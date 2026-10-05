import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "One context window": {
    title: "Everything the model reads, in one list",
    summary:
      "The host assembles one context: the developer's system prompt, the tool list, the user's message and the retrieved passages. Labelled by author, the hostile notice stands out. As the model receives it, it is just more text in the same sequence.",
    whatYouSee: [
      "Scenario: Indirect injection, Paraphrased injection, Direct injection, Image in the reply, Hallucination, or Sycophancy, with the failure's name and whether it is a security or an accuracy failure.",
      "View: who wrote each part — coloured bands, author labels, the source each retrieved passage came from, and a double red band for text carrying instructions — or what the model receives, one flat stream with [role] markers.",
      "Segments removed before the model sees them are struck through and faded, and the label says which control removed them: the keyword filter, or the allowlist of retrieval sources.",
    ],
    howItWorks: [
      "The system prompt includes the password only when Password in the system prompt is set to in the prompt. The tools line reflects the send_email setting.",
      "Retrieved passages are fixed per scenario: dock hours from the staff-edited harbor handbook, plus a hostile notice from the visitor notice board, which anyone can post to, for the three indirect scenarios (Indirect, Paraphrased, Image in the reply).",
      "With Retrieval sources on allowlist only, a passage whose source is not on the list is removed first. The keyword filter then reads whatever remains.",
    ],
    controls: [
      "Scenario and View.",
      "Comparison worth running: Indirect injection in both views — the notice's authority comes only from where it sits.",
    ],
    notice: [
      "Direct injection puts the attack in the user's own message; there is no retrieved notice at all, so neither the filter nor an allowlist has anything to remove.",
      "Image in the reply carries its instruction in a notice too, but asks for a markdown image instead of a tool call.",
      "Hallucination and Sycophancy contain nothing hostile. Their risk is a wrong answer. Hallucination's note points to Grounding check in the Retrieval-augmented generation lab.",
    ],
    limits: [
      "In this lab: no model runs, so the lab never shows whether an attack succeeds; the context is assembled from fixed strings.",
      "In general: chat templates wrap roles in special tokens, and models are trained to favour the system prompt, but that preference is learned and can be overridden. It is not an enforced boundary.",
    ],
  },

  "Host controls": {
    title: "Guarantees that do not depend on the model",
    summary:
      "The card assumes the worst: the model obeys any instruction that reaches it. Under that assumption, a leak needs private data in context, attacker text in context, and a way to send data out. The three toggles set those conditions, and the outcomes follow.",
    whatYouSee: [
      "Keyword filter on retrieved text, Retrieval sources (any page, or allowlist only), Password in the system prompt, send_email tool (runs automatically, needs approval, not offered) and Images in replies (rendered automatically, or not fetched).",
      "Three condition rows, lit when true: private data in context, attacker text reaches the model, and an unattended way to send data out, which for the image attack means an image the chat window loads and for the others means email.",
      "Metrics: Password in the reply, Password sent by email and Password in a rendered image — possible, blocked, only if a person approves, not used by this attack, or no attacker.",
      "For Image in the reply, a worst-case reply with the password in an image URL, and what the chat window then does with it.",
    ],
    howItWorks: [
      "Reply leak = attacker text reaches the model AND the password is in context. Email leak additionally needs send_email; with approval, a person must say yes. Image leak additionally needs Images in replies to render, because the chat window then requests the URL by itself.",
      "Each attack uses one outbound channel: the first three use email and Image in the reply uses an image. The other channel's readout says not used by this attack.",
      "The keyword filter removes a retrieved passage when it matches the pattern shown in Filter test bench. The allowlist removes a retrieved passage whose source is not on it, whatever the wording. Neither inspects the user's message.",
    ],
    controls: [
      "The five toggles, together with Scenario.",
      "Comparisons worth running: Paraphrased injection with the filter on, then set Password in the system prompt to kept out. Then Paraphrased injection with the filter off and Retrieval sources on allowlist only. Then Image in the reply with send_email tool not offered, against Images in replies not fetched.",
    ],
    notice: [
      "Keeping the password out of the prompt closes every leak for every scenario, whatever the wording of the attack.",
      "The allowlist closes the indirect attacks without any pattern, but not Direct injection, where the attacker is the user, and it costs every useful page outside the list.",
      "Closing the email channel does nothing for the image attack: each channel needs its own control, and the reply stays contaminated either way.",
      "For Hallucination and Sycophancy nothing changes, because there is no attacker.",
    ],
    limits: [
      "In this lab: the worst case is assumed, not measured, and the secret is a single word in a system prompt. The sources are two fixed labels, and the image fetch is described, not performed: nothing is requested from any server.",
      "In general: real systems hold data in tools, files and memory as well as prompts, and outbound channels include links, images and API calls, not only email. An allowlist has to be maintained and can be wrong, and a source on it can itself be poisoned. Defences for the image channel include not loading images from model output, and limiting which hosts images may load from, for example with a content security policy.",
    ],
  },

  "Filter test bench": {
    title: "Pattern matching, measured",
    summary:
      "Five fixed texts go through the same case-insensitive regular expression the keyword filter uses: three attacks and two benign texts. A sixth row takes your own text. The results are computed, and they show both a filter's misses and its false alarms.",
    whatYouSee: [
      "The regular expression.",
      "Five rows with their label, text, hostile or benign, whether the pattern matched, and a verdict: caught, missed, over-blocked or passed.",
      "Your text: a box you type in, and one row for it showing the filter's verdict (matches the pattern and is removed, or no match and is kept), whether the simulated assistant would act on it and why, and a verdict: caught, bypass, over-blocked or passes.",
      "Metrics: Attacks caught and Benign blocked. They count the five fixed texts only, never yours.",
    ],
    howItWorks: [
      "Pattern: ignore, optional all or any, then previous, prior or above, then instructions, with any whitespace between.",
      "caught = hostile and matched; missed = hostile and not matched; over-blocked = benign and matched; passed = benign and not matched.",
      "The simulated assistant is a second, small rule, not a model. It acts on your text when the text asks for something the lab protects: it contains a request word (tell, share, send, forward and similar) and the password, a passcode, a secret or the log, after look-alike spellings (ign0re, p@ssword) and spaced-out letters are read as letters, and with no never, not or don't in it. For your text, hostile means the assistant would act: caught = filter matched and it would act; bypass = filter did not match and it would act; over-blocked = filter matched and it would not act; passes = neither.",
    ],
    controls: [
      "Your text is the only control; the five rows are a fixed test set.",
      "Comparisons worth running: the literal injection against the staff security memo — the same phrase, opposite intent, same verdict from the filter. Then, in Your text, change Ignore to Disregard, then all previous to the previous, then add a never before the request.",
    ],
    notice: [
      "1 of 3 attacks caught: the paraphrase and the “ign0re” spelling both pass.",
      "1 of 2 benign texts blocked: the memo warning staff about the attack is itself removed. That is over-refusal.",
      "Case and extra spaces do not help an attacker, because the pattern ignores case and matches any run of whitespace. Another word, a missing letter, a digit for a letter, or spaced-out letters do, and the simulated assistant still acts, because its rule reads the request and not the phrase.",
    ],
    limits: [
      "In this lab: five hand-written texts and one regular expression. The assistant is a keyword-and-negation rule that does not understand meaning: it would act on less than a real model would (a vague request that names neither the password nor the log), and it would not act on some text a real model would obey (a negation inside an attack, such as “do not refuse”). Its verdict is a stand-in, so “passes” never means safe. A real evaluation needs many attacks and many ordinary documents.",
      "In general: learned classifiers catch more rewordings than regexes, but they also miss and over-block. Filters reduce risk; they do not remove the need to limit data and actions.",
    ],
  },
};

export default cardInfo;
