import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Agent loop": {
    title: "The loop is host code",
    summary:
      "The host builds a context, calls the model, and checks the output. A plain-text answer is a terminal action and ends the loop. A tool call is executed by the host, its result is appended, and the host checks the step cap and any stop rule before calling the model again.",
    whatYouSee: [
      "Task: Look up hours, Recover from error, Retry forever, or Fix an argument.",
      "A loop diagram: Context → Model call → final answer? → Stop · answer, or → Host runs the tool → Host check → next call or Stop. Solid boxes are the model's side, dashed boxes are host code; the active step is filled.",
      "Replay step, which walks every event of the run, and a sentence describing the current event.",
    ],
    howItWorks: [
      "while calls < cap: output = policy(transcript); if final, stop; else result = host.run(output), append both; if the stop rule is on and the same call failed twice in a row, stop.",
      "The policy is a fixed rule standing in for the model: answer if the last tool result succeeded, otherwise issue the task's call again. On Fix an argument it instead reads the error text and corrects the argument the error names.",
    ],
    controls: [
      "Task and Replay step.",
      "Comparison worth running: Look up hours against Retry forever at the same Step cap — one exits at Stop · answer, the other at Stop · step cap.",
    ],
    notice: [
      "The model box never decides to stop the loop by itself except by writing an answer.",
      "With Step cap 1 on Look up hours, the host still runs the tool the first call requested; the answer never comes.",
    ],
    limits: [
      "In this lab: the policy is scripted and deterministic, and its repair is a string rule that matches the wrong value against the allowed values in the error text. A sampled model could answer early, pick a different tool, give up in words, or misread the error.",
      "In general: agent frameworks differ in detail, but each wraps a model in host code that executes actions and decides when to stop.",
    ],
  },

  "Trace": {
    title: "The transcript, event by event",
    summary:
      "Every model call and every tool result, in order, exactly as appended to the context. Model calls show their text and the tokens read and written. Tool results show the host's JSON, with failures outlined.",
    whatYouSee: [
      "The user's question in the kicker.",
      "One row per event: Model call n with its output and token counts, Host · tool result indented beneath it, and a final Stop row with the reason.",
      "The current replay event outlined; later events faded. Clicking a row jumps Replay step to it.",
    ],
    howItWorks: [
      "Rows are the run's events. Nothing is authored per row: the host produces them by running the loop.",
      "Tokens read by call n are the toy token count of the system prompt, the user message, and every earlier call and result.",
    ],
    controls: [
      "Click any row to replay to it.",
      "Comparison worth running: Recover from error — call 1 gets a timeout, call 2 gets 15:40, call 3 answers.",
      "Second comparison: Fix an argument — call 1 is rejected for \"Weekday\", call 2 sends \"weekday\" and gets 21:00. The two calls differ; in Recover from error they were identical.",
    ],
    notice: [
      "get_wind was never in the tool list. The policy asks for it anyway, and the host rejects it every time.",
      "The model learns about the timeout only because the host wrote it into the transcript.",
      "In Fix an argument the error names the argument and lists the allowed values. That text is all the second call has to go on.",
    ],
    limits: [
      "In this lab: two real tools with fixed answers; get_ferry_time fails on its first attempt by design, and get_dock_hours checks its two enum arguments the way the previous lab does.",
      "In general: observations can be incomplete, wrong or hostile, and a long trace makes it harder for the model to use early details.",
    ],
  },

  "Budgets": {
    title: "Three ways a loop ends",
    summary:
      "A run ends by a final answer, by the step cap, or by a host stop rule. The cap bounds the number of model calls; the stop rule ends a run early when the same call fails twice in a row. The metrics total what the run consumed.",
    whatYouSee: [
      "Step cap from 1 to 10 model calls, and Stop rule: none, or stop after 2 identical failures.",
      "Metrics: Model calls, Tool runs (and how many failed), Stop reason, Tokens read and Tokens written.",
    ],
    howItWorks: [
      "Tokens read = Σ over calls of that call's input. Tokens written = Σ of each output's toy token count.",
      "The stop rule compares each failing call's text with the previous failing call's text.",
    ],
    controls: [
      "Step cap and Stop rule.",
      "Comparison worth running: Retry forever at cap 4 then 10 — tokens read go from 418 to 1,975 for 2.5 times as many calls.",
    ],
    notice: [
      "The stop rule does not trip on Recover from error: one failure followed by a success is not two identical failures. It does not trip on Fix an argument either, because the second call is not identical to the first.",
      "A stop by cap or rule leaves the user without an answer. It limits damage; it does not solve the task.",
    ],
    limits: [
      "In this lab: cost is shown in toy tokens, not money, and every call has the same price per token.",
      "In general: production agents cap steps, wall-clock time, spend and tool permissions, and ask a person before irreversible actions.",
    ],
  },

  "Context growth": {
    title: "Input grows every turn",
    summary:
      "Each bar is one model call's input: the fixed system prompt and user message, plus every earlier call and tool result. Because the loop appends to the transcript every turn, each call reads more than the last, and the running total grows faster than the number of calls.",
    whatYouSee: [
      "One bar per allowed call: the light part is system prompt plus user message, the dark part is earlier turns. Dashed stubs are calls the cap allowed but the run did not use.",
      "The input size above each bar, the call number below it, and the total tokens read in the heading.",
    ],
    howItWorks: [
      "input(n) = base + Σ over k < n of (output(k) + result(k)). On Retry forever each turn adds the same 31 tokens, so input rises linearly and the total rises quadratically.",
      "Token counts come from a toy rule: runs of letters, runs of digits, and single symbols each count as one.",
    ],
    controls: [
      "This card has no controls. Task, Step cap and Stop rule change the bars; Replay step fades calls not yet reached.",
      "Comparison worth running: Retry forever at cap 10 — call 10 reads 337 tokens, almost six times call 1.",
    ],
    notice: [
      "The first call of every task reads the same base, give or take the user message.",
      "A host stop rule removes the tall bars on the right, which is where most of the tokens are.",
    ],
    limits: [
      "In this lab: real tokenizers split text differently, and nothing here is cached.",
      "In general: prompt caching can cut the price of re-reading an unchanged prefix, but the context still grows, and long transcripts can exceed the context window or bury early details.",
    ],
  },

  "Reliability": {
    title: "Per-step success, compounded",
    summary:
      "The chance that a run finishes with every step right is the per-step success probability multiplied by itself once per step. The card draws that as a curve over run length and as a table, and adds an approval gate: a reviewer who inspects the work after a chosen step and sends a wrong prefix back to be redone once.",
    whatYouSee: [
      "Four controls: Per-step success (50 to 100%), Steps in the run (1 to 30), Approval gate after step (0 means no gate) and Reviewer catches (0 to 100% of wrong runs).",
      "Metrics: Completes end to end, With the gate, Longest run that finishes half the time, and Expected steps run.",
      "A line chart of completion probability against run length: a solid line without a gate, a dashed line with it, and a marker at the current length. A table gives the same values for selected lengths.",
    ],
    howItWorks: [
      "No gate: P = p^n, where p is Per-step success and n is Steps in the run. With a gate after step k and catch rate d: s = p^k and P = (s + (1 − s)·d·s)·p^(n − k). A gate set beyond the run's last step sits after the last step.",
      "Expected steps run = n + k·d·(1 − s): the run itself plus the k steps redone when the reviewer sends a wrong prefix back. Longest run that finishes half the time is the largest n with p^n at least 0.5.",
      "The tests check these formulas against a seeded run-by-run simulation, so the numbers are exact formulas that a sample agrees with.",
    ],
    controls: [
      "Per-step success, Steps in the run, Approval gate after step and Reviewer catches.",
      "Comparison worth running: 95% against 99% at 20 steps — 35.8% against 81.8%.",
      "Second comparison: a gate after step 10 at 95% per step with Reviewer catches at 100% against 0% — 50.2% against 35.8%. The gate costs 4.0 expected steps at 100% and nothing at 0%.",
    ],
    notice: [
      "At 95% per step only runs of 13 steps or fewer finish more often than not. At 99% the limit is 68 steps.",
      "A gate after step 1 is a retry of step 1, and it barely helps: it protects one step of twenty (37.6% against 35.8%).",
      "A gate at the end gives the highest chance but the most redone work: 58.8% for 32.8 expected steps at 95% per step and 20 steps.",
    ],
    limits: [
      "In this lab: closed-form formulas with fixed assumptions — identical, independent steps; errors the agent cannot see; one check; a redo that is not rechecked. Nothing here runs an agent, and Per-step success is a number you set, not a measurement.",
      "In general: real steps differ in difficulty, their errors are linked, and some are never noticed. A reviewer can miss errors or be skipped under time pressure, and a retry can fail too. The per-step rate of a real agent has to be measured on its own tasks.",
    ],
  },
};

export default cardInfo;
