import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `scenario` is a scenario id from safety.ts:
 * indirect, paraphrase, direct, output (Image in the reply), hallucination or sycophancy. Every scene
 * starts from the lab's defaults: filter off, password in the prompt, send_email running automatically,
 * any retrieval source, images rendered, and the starter text in Your text.
 */
const scenes: LabScene[] = [
  {
    id: "one-stream",
    sections: ["Everything arrives as one stream of text", "What it is"],
    title: "What the model receives",
    look: "With View on what the model receives, the colours are gone: [system] and [retrieved 2] are just more text, and the notice sits a few lines below the rules.",
    step: 0,
    state: { scenario: "indirect", view: "model" },
    focus: "One context window",
  },
  {
    id: "direct-injection",
    sections: ["Prompt injection, direct and indirect"],
    title: "An attack the user types",
    look: "In Direct injection the attack sits in the user’s own message; switch Scenario back to Indirect injection to see it ride in on a fetched page.",
    state: { scenario: "direct" },
    focus: "One context window",
  },
  {
    id: "trifecta",
    sections: ["What a leak needs"],
    title: "Take away the way out",
    look: "With send_email tool not offered, the third check reads no and the email leak is blocked, yet Password in the reply still reads possible.",
    step: 4,
    state: { scenario: "indirect", email: "none" },
    focus: "Host controls",
  },
  {
    id: "no-attacker",
    sections: ["Wrong answers with no attacker"],
    title: "A question with no answer",
    look: "No part of this context is written by an attacker, and nothing in it names a lighthouse keeper; then pick Sycophancy.",
    step: 7,
    state: { scenario: "hallucination" },
    focus: "One context window",
  },
  {
    id: "see-context",
    sections: ["See one context window"],
    title: "Who wrote each part",
    look: "Read the band and label on each part, then set View to what the model receives and watch them disappear.",
    step: 0,
    state: { scenario: "indirect", view: "authors" },
    focus: "One context window",
  },
  {
    id: "filter-on",
    sections: ["Filter the literal attack"],
    title: "Keyword filter on",
    look: "The Visitor notice is struck through as removed by the keyword filter, and in Host controls both password readouts now read blocked.",
    step: 1,
    state: { scenario: "indirect", filter: "on" },
    focus: "One context window",
  },
  {
    id: "reworded",
    sections: ["Reword the attack"],
    title: "The same attack, new words",
    look: "The filter is still on, yet both password readouts read possible again, because the reworded notice has none of the words the pattern looks for.",
    step: 2,
    state: { scenario: "paraphrase", filter: "on" },
    focus: "Host controls",
  },
  {
    id: "your-text",
    sections: ["Type your own attack"],
    title: "Test the filter yourself",
    look: "Your text starts out caught; change IGNORE to Disregard and watch the verdict turn to bypass while the simulated assistant still acts.",
    step: 3,
    focus: "Filter test bench",
  },
  {
    id: "secret-out",
    sections: ["Remove a condition"],
    title: "Keep the password out",
    look: "With the password kept out, private data in context reads no and both leaks read blocked; put it back and try send_email tool instead.",
    step: 4,
    state: { scenario: "paraphrase", filter: "on", secret: "out" },
    focus: "Host controls",
  },
  {
    id: "allowlist",
    sections: ["Allowlist the sources"],
    title: "Only trusted sources",
    look: "With allowlist only, attacker text reaches the model reads no and both leaks read blocked; then pick Direct injection and watch them reopen.",
    step: 5,
    state: { scenario: "paraphrase", sources: "allowlist" },
    focus: "Host controls",
  },
  {
    id: "image-channel",
    sections: ["Close an output channel"],
    title: "A leak through a picture",
    look: "The reply ends with a picture whose address carries the password; set send_email tool to not offered, then Images in replies to not fetched, and watch Password in a rendered image.",
    step: 6,
    state: { scenario: "output" },
    focus: "Host controls",
  },
  {
    id: "accuracy",
    sections: ["Separate accuracy from security"],
    title: "Wrong, not leaked",
    look: "All three password readouts say no attacker; flip any switch and none of them moves, then try Sycophancy.",
    step: 7,
    state: { scenario: "hallucination" },
    focus: "Host controls",
  },
];

export default scenes;
