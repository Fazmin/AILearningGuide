import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Model output": {
    title: "The only thing the model produces",
    summary:
      "A tool call starts as text the model generates. Here that text is chosen from eleven authored first calls — three well-formed, six malformed, and two that fail and are then retried — or typed by you. Everything downstream is computed from exactly this string.",
    whatYouSee: [
      "The user message that prompted the call.",
      "Preset buttons in three rows: well-formed (valid call, add a note, email the log), malformed (missing field, bad enum, wrong type, extra field, not JSON, unknown tool) and error, then retry (wrong type, then fixed; missing field, then fixed).",
      "Model output, an editable textarea holding the raw text; its label says when you have edited it. The error-then-retry presets add a second textarea, Model output, turn 2, for the call written on turn 2.",
    ],
    howItWorks: [
      "Choosing a preset writes its text into Model output. Editing the text keeps the preset's user message and reruns every host check.",
      "Choosing an error-then-retry preset also fills the turn 2 textarea with an authored second call. The host runs it only if the first call did not execute, and checks it from scratch.",
      "Nothing samples these strings from a model. They are written to hit one failure each, and the retries are written to fix it.",
    ],
    controls: [
      "The preset buttons, the Model output textarea and, with a retry preset, the turn 2 textarea.",
      "Comparison worth running: valid call, then change \"north\" to \"North\" by hand — the enum is case-sensitive and validation fails.",
      "Second comparison: wrong type, then fixed, then edit the turn 2 textarea back to \"day\": 7 — the same mistake fails the same way.",
    ],
    notice: [
      "Well-formed and malformed outputs look almost identical as text. The difference is found by the host, not the model.",
      "Email the log is well-formed. Its problem is what it asks for, not its shape.",
      "In a retry preset the two calls differ in one value. Everything the model needed to know about that value arrived in the error message.",
    ],
    limits: [
      "In this lab: the outputs are authored, including both retries, which are not generated from the error text. There is one call per turn, with no parallel calls or streaming.",
      "In general: models produce tool calls as tokens; APIs often return them in a dedicated field, but the content is still generated text that the host must parse and check.",
    ],
  },

  "Host pipeline": {
    title: "Ten stages across a wall",
    summary:
      "The left lane is the model: it reads tool schemas and writes text. The right lane is host code: parse, look up the tool, validate, apply policy, execute, and append the result. The first failing stage stops the rest, and its error becomes the tool message.",
    whatYouSee: [
      "Two lanes with a dashed wall, arrows for the four crossings, and ten numbered stages.",
      "Host stages marked ✓ pass, ✕ fail or – skipped, with the detail that decided it.",
      "A header badge, executed or not executed, Approve send_email and, with a retry preset, a Turn switch.",
    ],
    howItWorks: [
      "Parse: JSON.parse, then require {name: string, arguments: object}. Lookup: the name must be one of the three advertised tools. Validate: the JSON Schema checks listed in Schema check.",
      "Policy: get_dock_hours and add_note run automatically; send_email runs only if Approve send_email is set to human approves. Execute: a lookup table, an in-memory note, or a simulated email.",
      "Turn chooses which call the stages show. Both calls go through the same ten stages; turn 2 exists only when turn 1 failed and there is a retry.",
    ],
    controls: [
      "Approve send_email: no approval or human approves. Turn: 1 · first call or 2 · retry.",
      "Comparison worth running: email the log with no approval, then with human approves. Stages 4–6 are identical; only stage 7 changes the outcome.",
      "Second comparison: wrong type, then fixed with Turn 1, then Turn 2 — stage 6 fails, then every host stage passes.",
    ],
    notice: [
      "Every failure still produces a stage 9 message. The loop continues; the next model call just reads an error.",
      "The model lane never changes when a check fails. The model does not know what the host did until stage 10.",
    ],
    limits: [
      "In this lab: the tools are a two-row table, a counter and a fake mailer. Nothing touches the network or disk.",
      "In general: host code is where permissions, rate limits, confirmations, logging and sandboxing live. If the host executes blindly, a validated call can still do damage.",
    ],
  },

  "Schema check": {
    title: "JSON Schema, checked for real",
    summary:
      "The schema of the tool the call names is shown as sent to the model. The validator walks the parsed arguments and reports each violation with a JSON-pointer path. Validation runs only if parsing and tool lookup passed.",
    whatYouSee: [
      "The tool's name, policy and JSON schema: types, enum lists, required properties, maxLength, pattern, and additionalProperties false.",
      "One row per validation error, such as /dock required property is missing, or a note that validation passed or did not run.",
    ],
    howItWorks: [
      "Checks: type (string, integer, number, boolean, object), enum membership, required keys, keys not listed when additionalProperties is false, maxLength and pattern on strings.",
      "All errors are collected, not just the first, and each becomes one line of the tool message's details. With a retry preset the card follows the turn chosen in Host pipeline.",
    ],
    controls: [
      "This card has no controls. It follows Model output.",
      "Comparison worth running: bad enum against wrong type — /dock “must be one of” versus /day “expected string, got integer”.",
    ],
    notice: [
      "“east” is a perfectly good string. The enum is what makes it wrong.",
      "The schema says nothing about whether emailing all@harbor.example is a good idea.",
    ],
    limits: [
      "In this lab: a subset of JSON Schema; no nested objects, arrays, formats or references.",
      "In general: providers accept different subsets of JSON Schema, and a schema is a contract about shape, not about intent or safety.",
    ],
  },

  "Transcript": {
    title: "The next call's whole view",
    summary:
      "The conversation is a list of messages. After a tool call, the host appends a tool message with the result or the error. The next model call reads the whole list and nothing else — not the host's code, logs or notes. With an error-then-retry preset the list holds both turns, so you can see the error sitting between the two calls.",
    whatYouSee: [
      "Five messages, or seven with a retry: system (the tool list), user, assistant (the raw call), tool (the host's JSON result), then for a retry another assistant call and tool result, and finally the next assistant turn, which is not generated here.",
      "Metrics: Executed (the outcome of the last call), Grammar blocks it (whether constrained decoding with these schemas could have prevented the first call), and Calls made.",
    ],
    howItWorks: [
      "Each tool message is exactly the stage 9 string for its call. Errors are returned as data so the model can correct itself on the next turn.",
      "Grammar blocks it is yes for parse, lookup and shape errors, no for a well-formed call stopped by policy, and n/a when the call ran. It describes the first call.",
    ],
    controls: [
      "This card has no controls. It follows the preset and both textareas in Model output.",
      "Comparison worth running: missing field against email the log — both not executed, but only one could have been prevented by a grammar.",
      "Second comparison: wrong type, then fixed — Calls made reads 2, Executed reads yes, and Grammar blocks it still reads yes for the first call.",
    ],
    notice: [
      "A saved note exists only in host memory. The model learns about it only through the tool message.",
      "In a retry, the second assistant message was written after the error above it. That message is all the model had to go on.",
      "The last assistant line is a placeholder. This lab does not generate the next call or an answer.",
    ],
    limits: [
      "In this lab: message roles are shown generically; real APIs name and structure them differently.",
      "In general: tool results are untrusted input to the next call. A web page or email returned by a tool can carry instructions, which Failure modes, safety, and security takes up.",
    ],
  },
};

export default cardInfo;
