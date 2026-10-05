import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeState } from "./state";

/** Reads a serialized state of any version; see state.ts for what each version stored. */
export function hydrateToolsState(value: string) {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-40-tools-function-calling",
  slug: "tools-function-calling",
  title: "Tools and function calling",
  group: "building-with-models",
  order: 24,
  icon: "Wrench",
  accent: "#b54d3a",
  prerequisites: ["module-37-structured-decoding"],
  estimatedMinutes: 12,
  steps: [
    "Read the contract",
    "Break the arguments",
    "Break the text",
    "Valid is not safe",
    "Read the transcript",
    "Repair a bad call",
  ],
  stepInstructions: [
    "Leave the preset on valid call. In Schema check read get_dock_hours: dock and day are required, each limited to an enum, and no other fields are allowed. Then follow stages 1–10 in Host pipeline.",
    "Click missing field, bad enum, wrong type and extra field in turn. Each passes parsing and fails at stage 6 with a JSON path in Schema check. Nothing executes.",
    "Click not JSON, then unknown tool. One fails at parse, the other at tool lookup. Then edit Model output yourself — change \"north\" to \"south\" — and watch every check rerun.",
    "Click email the log. Stages 4–6 all pass, and the host still refuses at policy. Set Approve send_email to human approves: only then does stage 8 run.",
    "Open Transcript and find the tool message. It is the only record of what happened that the next model call will read.",
    "Click wrong type, then fixed. In Host pipeline set Turn to 1, then 2: the first call fails at stage 6 and the retry passes every stage. In Transcript find the error the retry was written from, then edit the retry to repeat \"day\": 7 and watch the host reject it again.",
  ],
  stateVersion: 3,
  tagline:
    "Parse a model's tool call, validate it against a JSON schema, apply host policy, and return the result as a message — then watch a failed call come back as data and get corrected on the next turn. The model only writes text; the host runs everything.",
  objectives: [
    "Separate what the model does (emit text naming a tool and arguments) from what the host does (parse, validate, decide, execute, report)",
    "Use a JSON schema to name distinct failures — unparseable text, unknown tool, missing field, wrong type, bad enum, extra field — and what the host returns for each",
    "Explain why a call that passes every schema check can still need a policy decision, and why constrained decoding cannot provide one",
    "Trace an error-then-retry loop: the host returns the failure as data, the next call reads it and corrects its arguments, and the host validates the corrected call from scratch",
  ],
  glossary: [
    {
      term: "Tool",
      definition:
        "A function the host program is willing to run on the model's behalf, described to the model by a name, a description and a parameter schema.",
    },
    {
      term: "Function call",
      definition:
        "Text the model generates naming a tool and its arguments, usually as JSON. It is a request; nothing runs until the host acts on it.",
    },
    {
      term: "JSON Schema",
      definition:
        "A standard vocabulary for describing JSON: types, required properties, allowed values (enum), string limits, and whether extra properties are allowed. Tool parameters are usually written in it.",
    },
    {
      term: "Validation",
      definition:
        "Checking parsed arguments against the schema before running anything. It catches shape errors; it says nothing about whether a well-shaped request is wise.",
    },
    {
      term: "Enum",
      definition:
        "A schema rule that a value must be one of a listed set, such as north or south. It turns an open string into a closed choice.",
    },
    {
      term: "Orchestrator",
      definition:
        "The host code that sends tool schemas to the model, parses its output, validates, applies policy, executes, and appends the result. Policy lives here.",
    },
    {
      term: "Host policy",
      definition:
        "Rules the host applies after validation: which tools run automatically, which need a person to approve, and which are never exposed.",
    },
    {
      term: "Side effect",
      definition:
        "A change in the world — a saved note, a sent email. It happens only if the host executes the tool, and it cannot be taken back by the model.",
    },
    {
      term: "Tool result",
      definition:
        "The message the host appends after a call: the output, or an error. The next model call reads it as ordinary context.",
    },
    {
      term: "Constrained decoding",
      definition:
        "Masking the model's next-token choices so its output must match a grammar, such as one compiled from a JSON schema. It prevents shape errors, not bad decisions.",
    },
    {
      term: "Structured error",
      definition:
        "A failure reported as data, with the path and the reason, instead of a crash or a silent refusal. The next model call reads it as context and can change its arguments, which is how a repair loop works.",
    },
    {
      term: "Tool description",
      definition:
        "The prose a tool carries so the model can judge whether it fits the request and what each argument means. A vague one leaves the model guessing between tools; a tool-choice setting offered by some APIs can also limit which tool, if any, it may call.",
    },
    {
      term: "Parallel tool calls",
      definition:
        "Several tool calls written in one model turn. The host decides whether to run them together or in order, and each call still gets its own result message.",
    },
    {
      term: "Model Context Protocol",
      definition:
        "An open protocol, built on JSON-RPC 2.0 messages, for a host to list and call tools that separate servers expose. It standardizes the listing and the call; the host still decides what runs. This lab does not implement it.",
    },
  ],
  references: [
    {
      authors: "Anthropic",
      title: "Define tools",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools",
      note: "Shows how a tool is defined by a name, a description, and an input schema written in JSON Schema, as in stage 1 of Host pipeline. It calls detailed descriptions by far the most important factor in tool performance and lists the four tool_choice modes (auto, any, tool, none) that the lesson's Descriptions and tool choice section describes.",
    },
    {
      authors: "JSON Schema project",
      title: "JSON Schema reference: object",
      source: "JSON Schema documentation",
      year: 2026,
      url: "https://json-schema.org/understanding-json-schema/reference/object",
      note: "Explains the required keyword and how setting additionalProperties to false rejects any key the schema does not list. These are the rules behind the missing field and extra field presets that fail at stage 6 in Schema check.",
    },
    {
      authors: "Anthropic",
      title: "Handle tool calls",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls",
      note: "Shows the host running the tool and sending back a tool_result message, marked is_error when something went wrong. It shows a missing-parameter error returned as a result so the model can try again with the value filled in, and it advises writing error messages that say what went wrong instead of just \"failed\", like the Repair a bad call step.",
    },
    {
      authors: "OpenAI",
      title: "Function calling",
      source: "OpenAI API documentation",
      year: 2026,
      url: "https://developers.openai.com/api/docs/guides/function-calling",
      note: "Another provider's guide to the same round trip: the application, not the model, executes the code and sends the output back. Its Strict mode section explains that strict mode uses structured outputs, requires additionalProperties set to false, and does not support every JSON Schema feature, which backs the point that schemas differ by provider.",
    },
    {
      authors: "Anthropic",
      title: "Parallel tool use",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use",
      note: "Says the host decides whether to run several calls from one turn together or in order. Independent read-only calls are usually safe to run in parallel, tools with side effects may be better run one at a time, and every call gets its own result, as the lesson's Parallel calls section says.",
    },
    {
      authors: "Model Context Protocol project",
      title: "Specification",
      source: "Model Context Protocol specification, version 2026-07-28",
      year: 2026,
      url: "https://modelcontextprotocol.io/specification/2026-07-28",
      note: "The overview of the open protocol. It says MCP uses JSON-RPC 2.0 messages between hosts, clients, and servers, and its security principles say hosts must obtain explicit user consent before invoking any tool, the rule the lesson compares with this lab's policy stage.",
    },
    {
      authors: "Model Context Protocol project",
      title: "Tools",
      source: "Model Context Protocol specification, version 2026-07-28",
      year: 2026,
      url: "https://modelcontextprotocol.io/specification/2026-07-28/server/tools",
      note: "Defines tools/list, tools/call, and the inputSchema each tool carries. Its Error Handling section splits protocol errors, such as an unknown tool, from tool execution errors marked isError: true, which clients should pass to the model so it can correct itself.",
    },
    {
      authors: "Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, et al.",
      title: "Toolformer: Language Models Can Teach Themselves to Use Tools",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 68539–68551",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html",
      note: "The early study the lesson names. A model is trained to decide which tool to call, when, with what arguments, and how to use the result, with tools such as a calculator, a search engine, and a calendar.",
    },
    {
      authors: "Yujia Qin, Shihao Liang, Yining Ye, et al.",
      title: "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2307.16789",
      note: "Builds a training set of instructions paired with chains of real API calls, then fine-tunes an open model on it. It is a concrete example of the lesson's point that models learn to write tool calls in post-training on examples of calls and results.",
    },
    {
      authors: "Shunyu Yao, Jeffrey Zhao, Dian Yu, et al.",
      title: "ReAct: Synergizing Reasoning and Acting in Language Models",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2210.03629",
      note: "The ReAct paper the lesson names. The model writes reasoning steps between actions, such as calls to a simple Wikipedia API, and reads each observation before its next step: the call-result loop that Agents builds on.",
    },
    {
      authors: "Shishir G. Patil, Tianjun Zhang, Xin Wang, et al.",
      title: "Gorilla: Large Language Model Connected with Massive APIs",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 126544–126565",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html",
      note: "Reports that even strong models such as GPT-4 often write inaccurate arguments and make up wrong ways to use an API. That is why the host checks every call, as the unknown tool and bad-argument presets show.",
    },
    {
      authors: "Shishir G. Patil, Huanzhi Mao, Fanjia Yan, et al.",
      title: "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models",
      source: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267, 48371–48392",
      year: 2025,
      url: "https://proceedings.mlr.press/v267/patil25a.html",
      note: "A benchmark that measures how often models write correct single and parallel function calls, and whether they hold back when no call fits. It answers the question this lab cannot, since its model outputs are authored and say nothing about how often a real model writes a bad call.",
    },
    {
      authors: "Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, et al.",
      title: "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection",
      source: "Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec 2023), 79–90",
      year: 2023,
      url: "https://arxiv.org/abs/2302.12173",
      note: "Shows that text a model retrieves can carry instructions that change what the application does, including how and whether other APIs are called. It backs the lesson's warning that tool results are untrusted.",
    },
    {
      authors: "OWASP Gen AI Security Project",
      title: "LLM06:2025 Excessive Agency",
      source: "OWASP Top 10 for LLM Applications 2025",
      year: 2025,
      url: "https://genai.owasp.org/llmrisk/llm062025-excessive-agency/",
      note: "A security guide to giving a model more tools, permissions, or freedom than it needs. One fix it lists is requiring a human to approve high-impact actions, as the Approve send_email control does, and it backs the point that blind execution is a host bug.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "The model emits {\"name\": \"send_email\", \"arguments\": {\"to\": \"all@harbor.example\", \"body\": \"…\"}}. It parses and passes the schema. What happens next?",
      options: [
        "The email is sent, because a valid call is executed by the model runtime",
        "The host applies its policy, and sends the email only if that allows it",
        "Nothing can stop it now, because constrained decoding already approved the call",
      ],
      answer: 1,
      explanation:
        "The model only wrote text. Parsing and validation show the request is well-formed, not that it is acceptable. The host's policy — here, a human approval — decides whether the side effect happens, and the host performs it. Constrained decoding shapes output; it has no say in whether a valid request should run.",
      objective: 2,
    },
    {
      prompt:
        "In Model output you change \"north\" to \"North\" in the valid call. Where does the host stop it, and why?",
      options: [
        "At validation, because enum matching is exact and North is not listed",
        "At parsing, because a capital letter makes the whole text invalid JSON",
        "At policy, because the host refuses any call a person has edited",
      ],
      answer: 0,
      explanation:
        "The edited text is still valid JSON and still names a real tool, so parsing (stage 4) and lookup (stage 5) pass. Stage 6 compares /dock with the enum north, south and an enum match is exact, so North fails with a path. Policy is never reached, and the host does not know or care who typed the text.",
      objective: 1,
    },
    {
      prompt: "When a call fails validation, how does the model find out?",
      options: [
        "The model's runtime raises an error inside the generation and restarts it",
        "The model sees the validation errors while it is still writing the arguments",
        "The host appends the error as a tool message, which the next call reads",
      ],
      answer: 2,
      explanation:
        "The model lane never changes when a check fails: it wrote its text and stopped. The host returns the failure as data, a tool message holding the paths and reasons, and the next model call reads that message as ordinary context. That is also why the tool message is the only record of what the host did.",
      objective: 0,
    },
    {
      prompt:
        "Grammar blocks it reads yes for bad enum and no for email the log without approval. Why the difference?",
      options: [
        "A grammar can mask a bad enum value, but no schema says a valid email is unwise",
        "Policy is decided by the model itself, so a grammar has no say over it",
        "A grammar can only constrain short values, and an email body can be 200 characters",
      ],
      answer: 0,
      explanation:
        "A grammar compiled from the schema removes tokens that would break the schema: wrong enum values, missing keys, wrong types, even the length and pattern limits. The email call is well-formed, so the grammar allows it. Whether to send it is a decision about consequences, and that lives in the host's policy, not in the model or its decoder.",
      objective: 2,
    },
    {
      prompt: "In this lab, which part of the Host pipeline is computed and which is authored?",
      options: [
        "A small language model writes each call live, and only the email send is simulated",
        "The model outputs are authored; the host stages run for real on the text you edit",
        "Each preset carries fixed stage results, so editing the text changes nothing below",
      ],
      answer: 1,
      explanation:
        "The eleven first calls are fixed strings, two of them with a fixed retry, and no model is sampled. Everything after them is real code: the JSON parse, the tool lookup, a JSON Schema subset validator, the policy check and a simulated execution, all rerun on exactly what is in the boxes. So the lab shows the host's behaviour faithfully and says nothing about how often a model writes a bad call, or whether it would fix one.",
      objective: 0,
    },
    {
      prompt:
        "On wrong type, then fixed, you set Turn to 2 and every host stage passes. What let the second call differ from the first?",
      options: [
        "The host quietly rewrote the bad argument itself and resubmitted the call for the model",
        "The model read the error in the tool message and wrote a new call from it",
        "The validator reached into the model and changed the token that it had chosen earlier",
      ],
      answer: 1,
      explanation:
        "The host did not change the call. It returned the failure as a tool message naming /day and the reason, and the next model call read that message as ordinary context and wrote a different call. In this lab the second call is authored to show that step; in a real system it is generated, and nothing guarantees it will fix the problem.",
      objective: 3,
    },
    {
      prompt:
        "You edit the retry so that it repeats \"day\": 7, the same mistake as the first call. What does the host do with it?",
      options: [
        "It rejects the call with the same error, since each call is checked afresh",
        "It accepts the call, because the format was already seen once and is now allowed",
        "It repairs the call itself, because it remembers the error it reported earlier",
      ],
      answer: 0,
      explanation:
        "The host keeps no goodwill and does no repair. Validation looks only at the text of the call in front of it, so a repeated mistake fails at stage 6 again with the same path. A repair loop works only when the model changes its call in response to the error, which is why agents add step caps and stop rules for loops that do not.",
      objective: 3,
    },
    {
      prompt:
        "A call to get_dock_hours asks for the south dock when the user asked about the north dock. Why will the repair loop never trigger?",
      options: [
        "The call passes every schema check, so there is no error for the model to read",
        "Repair loops only run for email tools, since those are the ones that need approval",
        "The validator compares the call with the user's question and reports the mismatch",
      ],
      answer: 0,
      explanation:
        "Validation checks shape against the schema: south is a legal enum value, so the call is well-formed and the host runs it. The wrong answer comes back as a normal result. A repair loop can only react to errors the host detects, which is why checking intent needs other tools: tests, review, or a human approving the call.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateToolsState,
};

export default definition;
