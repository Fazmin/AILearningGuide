import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeState } from "./state";

/** Reads a serialized state of any version; see state.ts for what each version stored. */
export function hydrateAgentsState(value: string) {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-41-agents",
  slug: "agents",
  title: "Agents",
  group: "building-with-models",
  order: 25,
  icon: "Workflow",
  accent: "#3d7a8c",
  prerequisites: ["module-40-tools-function-calling"],
  estimatedMinutes: 12,
  steps: [
    "Walk a successful loop",
    "Cap the horizon",
    "Recover from an error",
    "Loop without a stop rule",
    "Add a stop rule",
    "Repair an argument",
    "Compound the error",
  ],
  stepInstructions: [
    "Leave Task on Look up hours and move Replay step from 1 to the end. Watch the highlighted node move from Model call to Host runs the tool and back, then exit at Stop · answer.",
    "Set Step cap to 1. The host still runs the tool the first call asked for, but no second call is allowed, so the loop stops by step cap without an answer.",
    "Set Step cap to 4 and Task to Recover from error. The first get_ferry_time call times out; the second succeeds; the third call answers. Read Tool runs: 2 (1 failed).",
    "Set Task to Retry forever and raise Step cap to 10. Every call asks for a tool that does not exist, and in Context growth each bar is taller than the last.",
    "Keep Retry forever and set Stop rule to stop after 2 identical failures. The host ends the loop after two calls, whatever the step cap.",
    "Set Task to Fix an argument with the stop rule still on. Call 1 sends \"Weekday\" and the host rejects it; call 2 reads that error and sends \"weekday\"; call 3 answers. In Trace compare the two calls: they differ, so the rule never counts two identical failures.",
    "In Reliability keep Per-step success at 95% and Steps in the run at 20, then read Completes end to end. Raise Per-step success to 99% and compare. Back at 95%, set Approval gate after step to 10 and compare with and without it, then lower Reviewer catches to 0%.",
  ],
  stateVersion: 3,
  tagline:
    "Run a host loop around a model — call, run a tool, append the result, check the budget — and watch the step cap, a host stop rule and a growing context decide how it ends, a retry become a repair, and small per-step errors compound over a long run.",
  objectives: [
    "Describe an agent as a host loop that repeatedly calls a model, executes its tool calls, and appends the results, not as a different kind of model",
    "Explain how a terminal action, a step cap and a host stop rule each end the loop, and what happens when only the cap is left",
    "Explain why each model call's input grows with the loop, so cost rises faster than the number of steps",
    "Compute the chance that an agent completes every step of a run from its per-step success and the run length, and explain how an approval gate that sometimes catches errors changes it and what it costs in extra steps",
    "Tell a retry that repeats the same call from a repair that changes the call after reading the error, and say which of them a stop rule on identical failures can end",
  ],
  glossary: [
    {
      term: "Agent",
      definition:
        "A program that calls a model in a loop, executes the tool calls it proposes, feeds back the results, and decides when to stop. The model inside can be the same one used for plain chat.",
    },
    {
      term: "Agent loop",
      definition:
        "The repeated cycle: build the context, call the model, stop if it answered, otherwise run its tool call and append the result, then check the budget.",
    },
    {
      term: "Observation",
      definition:
        "What the loop feeds back after an action — a tool result or an error. The model knows only what reaches it as observations.",
    },
    {
      term: "Terminal action",
      definition:
        "An output that ends the loop, here a plain-text answer instead of a tool call. A model that never produces one never ends the loop by itself.",
    },
    {
      term: "Step cap",
      definition:
        "The host's maximum number of model calls. It guarantees the loop ends, but a loop that hits it stops without an answer.",
    },
    {
      term: "Stop rule",
      definition:
        "A host-side condition that ends the loop early, such as the same call failing twice in a row. It lives in host code, not in the model.",
    },
    {
      term: "Horizon",
      definition:
        "How many steps the loop may take. A longer horizon lets harder tasks finish and also lets errors, cost and side effects accumulate.",
    },
    {
      term: "Context growth",
      definition:
        "Each call reads the whole transcript so far, so input per call grows as the loop runs and the total tokens read grow faster than the number of calls.",
    },
    {
      term: "Planning",
      definition:
        "Writing out intended steps before or between actions. A written plan is generated text; nothing forces later calls to follow it.",
    },
    {
      term: "Compounding error",
      definition:
        "A mistake early in the loop that later steps build on. With many steps, even a small per-step error rate makes a fully correct run unlikely: the chance is the per-step success probability raised to the number of steps.",
    },
    {
      term: "Approval gate",
      definition:
        "A point in the loop where a person reviews the work so far before the agent goes on. It can catch an error before later steps build on it, at a cost in time and attention, and a reviewer can miss errors.",
    },
    {
      term: "Agent memory",
      definition:
        "What an agent can use beyond the current call: the transcript in its context window, which is exact but finite, or an external store such as a file or database that it writes to and reads from through tools.",
    },
    {
      term: "Multi-agent system",
      definition:
        "Several agent loops working on one task, often a lead agent that hands pieces to sub-agents and combines their results. It can run work in parallel and exceed one context window, but it costs more tokens and adds hand-offs where steps can go wrong.",
    },
  ],
  references: [
    {
      authors: "Stuart Russell and Peter Norvig",
      title: "Artificial Intelligence: A Modern Approach, 4th edition",
      source: "Pearson",
      year: 2020,
      url: "https://aima.cs.berkeley.edu/",
      note: "Chapter 2, Intelligent Agents, gives the classic definition: an agent takes in percepts from its surroundings and chooses actions, and its agent programs do this one percept at a time. A percept plays the role of this lesson's observation.",
    },
    {
      authors: "Lei Wang, Chen Ma, Xueyang Feng, et al.",
      title: "A survey on large language model based autonomous agents",
      source: "Frontiers of Computer Science 18(6), 186345",
      year: 2024,
      url: "https://arxiv.org/abs/2308.11432",
      note: "A broad review of agents built around language models. It sorts their parts into profile, memory, planning and action modules, which matches the lesson's terms Agent memory and Planning.",
    },
    {
      authors: "Shunyu Yao, Jeffrey Zhao, Dian Yu, et al.",
      title: "ReAct: Synergizing Reasoning and Acting in Language Models",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2210.03629",
      note: "The ReAct paper the lesson names. The model writes reasoning text and actions in turn, and the results of each action come back into the same transcript, the loop the Agent loop card draws.",
    },
    {
      authors: "Erik Schluntz and Barry Zhang",
      title: "Building effective agents",
      source: "Anthropic Engineering blog",
      year: 2024,
      url: "https://www.anthropic.com/engineering/building-effective-agents",
      note: "Describes agents as models using tools based on feedback from the environment in a loop. It recommends stopping conditions such as a maximum number of iterations, pauses for human feedback at checkpoints, and warns of higher costs and compounding errors, the same ideas as Step cap, Approval gate and Reliability.",
    },
    {
      authors: "Noah Shinn, Federico Cassano, Ashwin Gopinath, et al.",
      title: "Reflexion: Language Agents with Verbal Reinforcement Learning",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 8634–8652",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html",
      note: "Agents write a short reflection on why an attempt failed and keep it in memory for the next try, without changing the model's weights. It is a larger version of the repair in Fix an argument: read the feedback, then change what you do.",
    },
    {
      authors: "John Yang, Carlos Jimenez, Alexander Wettig, et al.",
      title: "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 50528–50652",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html",
      note: "A coding agent whose host checks each file edit, rejects a bad one, and shows the error so the model can try again. It also shortens old observations in the history to keep the context small, which ties to the lesson's points on repair and growing context.",
    },
    {
      authors: "Nelson F. Liu, Kevin Lin, John Hewitt, et al.",
      title: "Lost in the Middle: How Language Models Use Long Contexts",
      source: "Transactions of the Association for Computational Linguistics 12, 157–173",
      year: 2024,
      url: "https://aclanthology.org/2024.tacl-1.9/",
      note: "Shows that models often miss facts placed in the middle of a long input, even when the facts are there. It backs the lesson's warning that long transcripts bury early details the model needs.",
    },
    {
      authors: "Anthropic",
      title: "Prompt caching",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
      note: "Official docs for reusing an unchanged start of a prompt so it is cheaper and faster to read again, including a mode that moves the cache point forward as a conversation grows. It is the prompt caching that Going deeper mentions, which lowers the price of re-reading but does not stop the transcript growing.",
    },
    {
      authors: "Charles Packer, Sarah Wooders, Kevin Lin, et al.",
      title: "MemGPT: Towards LLMs as Operating Systems",
      source: "arXiv preprint arXiv:2310.08560",
      year: 2023,
      url: "https://arxiv.org/abs/2310.08560",
      note: "Treats the context window like a computer's fast memory and moves information to and from slower outside storage. It is a worked example of the external store described under Agent memory.",
    },
    {
      authors: "Jeremy Hadfield, Barry Zhang, Kenneth Lien, et al.",
      title: "How we built our multi-agent research system",
      source: "Anthropic Engineering blog",
      year: 2025,
      url: "https://www.anthropic.com/engineering/multi-agent-research-system",
      note: "The source the lesson cites for multi-agent systems. A lead agent saves its plan to memory because context past 200,000 tokens is cut off, hands pieces to sub-agents, and the system uses about 15 times the tokens of a chat.",
    },
    {
      authors: "Thomas Kwa, Ben West, Joel Becker, et al.",
      title: "Measuring AI Ability to Complete Long Software Tasks",
      source: "Advances in Neural Information Processing Systems 38 (NeurIPS 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2503.14499",
      note: "Measures how long a task, timed by how long it takes skilled people, AI agents can finish, and finds that success falls as tasks get longer. It also reports that newer models adapt to mistakes instead of repeating failed actions, which ties to Horizon, Reliability and Retry forever.",
    },
    {
      authors: "Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, et al.",
      title: "AI Agents That Matter",
      source: "Transactions on Machine Learning Research (TMLR)",
      year: 2025,
      url: "https://arxiv.org/abs/2407.01502",
      note: "Argues that agents should be judged on cost as well as accuracy, because an agent can cost far more than a single model call. It backs the lesson's point that tokens read grow faster than the number of calls, so long loops get expensive.",
    },
    {
      authors: "Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, et al.",
      title: "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection",
      source: "Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec 2023), 79–90",
      year: 2023,
      url: "https://arxiv.org/abs/2302.12173",
      note: "Shows that an attacker can hide instructions in data a model will later read, such as a web page, and steer which tools it calls. It backs the lesson's warning that an observation can carry hostile text.",
    },
    {
      authors: "Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, et al.",
      title: "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html",
      note: "A test bed of realistic agent tasks, such as email and banking, where data returned by tools tries to hijack the agent. It shows that observations can mislead and that tool access with side effects needs limits beyond a step cap.",
    },
  ],
  checkpoint: [
    {
      prompt: "On Retry forever with no stop rule, you raise Step cap from 4 to 10. What changes?",
      options: [
        "The agent learns from each failure and eventually finds the wind speed",
        "More failing calls are made, each reading a longer context",
        "The loop still ends after four calls, because the model decides when to stop",
      ],
      answer: 1,
      explanation:
        "The tool does not exist, so no number of retries helps. Without a terminal action or a host stop rule, only the cap ends the loop, and the scripted policy never writes an answer. Every call re-reads the growing transcript, so ten calls cost much more than two and a half times four.",
      objective: 1,
    },
    {
      prompt: "In the Agent loop diagram, which steps does the host perform rather than the model?",
      options: [
        "Choosing which tool to call, writing the arguments, and writing the final answer",
        "Running the tool, appending the result, and checking the cap and stop rule",
        "All of them, because the model runs the tool itself when it writes the call",
      ],
      answer: 1,
      explanation:
        "The model only emits text: a tool call or an answer. The host runs the tool, appends the observation, counts calls and decides whether the loop continues. That is why an agent is a loop around an unchanged model, and why every limit on the loop lives in host code.",
      objective: 0,
    },
    {
      prompt:
        "On Look up hours you set Step cap to 1. The tool runs but there is no answer. Why does the loop end this way?",
      options: [
        "The model judged the task impossible and decided to give up after one call",
        "The tool call failed on the first try, so the loop could not continue",
        "The host refuses a second model call, so the tool result is never read",
      ],
      answer: 2,
      explanation:
        "The first call asked for get_dock_hours and the host ran it, successfully. The cap then forbids another model call, so nobody reads the result and nobody writes the answer. A step cap guarantees the loop ends, not that it ends with an answer, and only a terminal action produces one.",
      objective: 1,
    },
    {
      prompt:
        "On Retry forever, raising Step cap from 4 to 10 raises Tokens read by more than a factor of 2.5. Why?",
      options: [
        "Each call re-reads the whole transcript, so inputs grow with each step",
        "Later calls use a larger model, which reads more tokens on every call",
        "The failing tool returns a longer error message each time it is called",
      ],
      answer: 0,
      explanation:
        "The error text is the same every time. What grows is the transcript: each call's input is the system prompt, the user message and every earlier call and result. Each turn adds a fixed amount, so the input per call rises linearly and the total rises roughly with the square of the number of calls.",
      objective: 2,
    },
    {
      prompt:
        "With the stop rule set to stop after 2 identical failures, you run Recover from error. What happens?",
      options: [
        "The rule ends the run after the first timeout, to be on the safe side",
        "The rule fires on the retry, because the same call is being repeated",
        "The retry succeeds, the count resets, and the model answers on call 3",
      ],
      answer: 2,
      explanation:
        "The rule counts consecutive identical failures. The first call times out, the identical retry succeeds, and the count resets, so the loop recovers and answers. The same rule ends Retry forever after two calls. A rule is a bet about which repeats are useless, and it cannot catch a loop whose calls keep changing.",
      objective: 1,
    },
    {
      prompt: "Why can this lab not tell you how often a real agent gets stuck in a loop?",
      options: [
        "Its model is a fixed rule that repeats the same call, unlike a sampled model",
        "Its loop code is a drawing, so the step cap and the stop rule are never applied",
        "Real agents never retry a failed call, so the Retry forever task cannot occur",
      ],
      answer: 0,
      explanation:
        "The loop, the cap, the stop rule and the token counts are computed, but the policy is scripted: answer after a good result, otherwise repeat the call. A sampled model could answer early, try another tool or give up in words, so how often real agents loop is an empirical question this lab does not measure.",
      objective: 0,
    },
    {
      prompt:
        "In Reliability you keep Steps in the run at 20 and raise Per-step success from 95% to 99%. What happens to Completes end to end?",
      options: [
        "It rises by about the same four points, because the gain is added once to the whole run",
        "It more than doubles, because the four-point gain applies at every one of the steps",
        "It barely moves, because only the first few steps of a long run can ruin the result",
      ],
      answer: 1,
      explanation:
        "The chance of finishing is the per-step probability raised to the number of steps, so a small gain per step is multiplied across all of them. At 20 steps it goes from about 36% to about 82%. A gain added once would be about four points, and a step late in the run fails as often as an early one.",
      objective: 3,
    },
    {
      prompt:
        "With Reviewer catches at 100% and Per-step success at 95%, a gate after step 10 beats a gate after step 1 on a 20-step run. Why?",
      options: [
        "It covers ten steps instead of one, so more of the run's errors get a second try",
        "It ends the run at step 10 and checks the answer, so the later steps can no longer fail",
        "It makes every step after the gate more reliable, because a person has looked at the work",
      ],
      answer: 0,
      explanation:
        "The gate sends a wrong prefix back to be redone once, so it protects exactly the steps before it. A gate after step 1 covers one step and one after step 10 covers ten, so it rescues many more runs. The price shows in Expected steps run: a later gate redoes more work. It neither stops the run nor changes the later steps, which still compound.",
      objective: 3,
    },
    {
      prompt:
        "Reliability says a 20-step run at 95% per step completes about 36% of the time. Why is that not a forecast for a real agent?",
      options: [
        "The 95% is an assumed value, and real steps differ in difficulty and fail together",
        "The curve comes from a simulation that is too short to give a trustworthy figure",
        "Real agents recheck every step, so their failures cancel out instead of compounding",
      ],
      answer: 0,
      explanation:
        "Every figure follows from assumptions you set: the same success chance at every step, independent steps, and errors the agent cannot see. In a real run some steps are easy and some hard, one mistake can make the next more likely, and the true per-step rate has to be measured. The card shows why length hurts, not how reliable a given agent is.",
      objective: 3,
    },
    {
      prompt:
        "Set Stop rule to stop after 2 identical failures. Why does it end Retry forever but leave Fix an argument alone?",
      options: [
        "Retry forever sends one call twice, but Fix an argument changes its call after the error",
        "Fix an argument has fewer failures in total, so it never comes close to the step cap",
        "The rule only watches get_wind calls, so any task that uses another tool is exempt from it",
      ],
      answer: 0,
      explanation:
        "The rule compares the text of consecutive failing calls. Retry forever sends the same get_wind call twice, so the rule fires. In Fix an argument the policy reads the error and changes \"Weekday\" to \"weekday\", so call 2 differs from call 1 and succeeds, and the count never reaches two. A rule on identical failures cannot end a loop whose calls keep changing but keep failing.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateAgentsState,
};

export default definition;
