import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { GRAMMAR_IDS, GRAMMARS, type GrammarId } from "./mask";

const initialState: ModuleState = {
  grammar: "json",
  step: 0,
  detour: 0.1,
};

const definition: ModuleDefinition = {
  id: "module-37-structured-decoding",
  slug: "structured-decoding",
  title: "Structured and constrained decoding",
  group: "inside-models",
  order: 21,
  icon: "Braces",
  accent: "#1a7f9c",
  prerequisites: ["module-07-next-token-prediction"],
  estimatedMinutes: 12,
  steps: ["Choose a grammar", "Step the prefix", "Read the mask", "Compare with conditioning"],
  stepInstructions: [
    "Set Grammar to JSON object, then to Function call, and read the Allowed output pattern; Decode step returns to the first token each time.",
    "Click through Decode step and compare Unconstrained pick with Greedy pick after mask at each prefix.",
    "On Masked distribution, find the most likely token marked masked, then check the formula: the greedy pick's probability divided by Z.",
    "On Mask distortion, drag P(slo\" after \"O) from 10% to 90% and watch the Per-step mask bar and the conditioned bar converge.",
  ],
  stateVersion: 1,
  tagline:
    "Mask next-token choices so a decode stays inside a grammar or JSON schema, renormalize what is left, and see that the constraint guarantees shape, not truth.",
  objectives: [
    "Describe constrained decoding as a token mask applied at each step",
    "Explain why a valid JSON string can still be a wrong answer",
    "Explain why masking one token at a time can sample different strings than conditioning the model on valid output",
  ],
  glossary: [
    {
      term: "Constrained decoding",
      definition:
        "Choosing each next token only from the set a grammar still allows. The model proposes a distribution; the mask removes illegal continuations before sampling or greedy choice.",
    },
    {
      term: "Grammar",
      definition:
        "A set of rules for which strings are legal, such as balanced braces or a function-call syntax. It is a filter, not a knowledge source.",
    },
    {
      term: "Schema",
      definition:
        "A description of fields and types, often JSON Schema, compiled into a grammar. A free string or number field still admits fluent nonsense.",
    },
    {
      term: "Token mask",
      definition:
        "A per-step yes-or-no over the whole vocabulary. Illegal tokens get probability zero, which is the same as setting their logits to minus infinity before the softmax.",
    },
    {
      term: "Renormalization",
      definition:
        "Dividing the surviving probabilities by their sum Z so they add to one again. It keeps the model's ratios among legal tokens and adds no new information.",
    },
    {
      term: "Valid prefix",
      definition:
        "Text that can still be completed into a legal string. The mask checks whether the prefix plus a candidate token is still a valid prefix.",
    },
    {
      term: "Token boundary",
      definition:
        "Where one vocabulary token ends and the next begins. Tokens rarely line up with grammar symbols, so a correct mask checks a token character by character, and a token may span two symbols.",
    },
    {
      term: "Greedy decoding",
      definition:
        "Taking the single most probable token at every step. Under a mask it takes the most probable legal token.",
    },
    {
      term: "Distribution distortion",
      definition:
        "The gap between per-step masking and sampling from the model conditioned on producing a valid string. Masking cannot see that a legal prefix leads mostly to illegal endings.",
    },
    {
      term: "Function call",
      definition:
        "A structured object naming a tool and its arguments. Constrained decoding can keep it parseable; it cannot check that the arguments are right.",
    },
  ],
  references: [
    {
      authors: "A. Wright, H. Andrews, and B. Hutton (eds.)",
      title: "JSON Schema Validation: A Vocabulary for Structural Validation of JSON",
      source: "IETF Internet-Draft draft-bhutton-json-schema-validation-01 (JSON Schema 2020-12)",
      year: 2022,
      url: "https://json-schema.org/draft/2020-12/json-schema-validation",
      note: "The standard that defines the type, enum, and required keywords a schema uses to list fields and the kind of value each holds. It is the format the lesson means by a JSON schema, including the enum field that allows only Oslo or Bergen on the Mask distortion card.",
    },
    {
      authors: "Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau",
      title: "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models",
      source: "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), 9895–9901",
      year: 2021,
      url: "https://aclanthology.org/2021.emnlp-main.779/",
      note: "An early method that parses the output as it is written and rejects tokens that would break the format, here SQL queries. It is the same valid-prefix check the lab's matcher runs at each Decode step.",
    },
    {
      authors: "Saibo Geng, Martin Josifoski, Maxime Peyrard, et al.",
      title: "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 10932–10952",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.674/",
      note: "Shows that a formal grammar can keep an off-the-shelf model's output in the required structure with no retraining, across tasks such as information extraction and parsing. It backs the lesson's point that constrained decoding leaves the weights alone and works at decode time.",
    },
    {
      authors: "Luca Beurer-Kellner, Marc Fischer, and Martin Vechev",
      title: "Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation",
      source: "Proceedings of the International Conference on Machine Learning (ICML 2024), PMLR 235, 3658–3673",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/beurer-kellner24a.html",
      note: "Finds that constrained decoders can hurt accuracy when they do not line up the model's sub-word tokens with the grammar, and presents a method (DOMINO) that does. It backs the lesson's Token boundary point that a correct mask must allow tokens like {\" that span two grammar symbols.",
    },
    {
      authors: "Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, et al.",
      title: "Grammar-Aligned Decoding",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 24547–24568",
      year: 2024,
      url: "https://papers.nips.cc/paper_files/paper/2024/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html",
      note: "The Park et al. (2024) paper the lesson cites. It shows that masking token by token can produce valid outputs at rates out of proportion to the model's own probabilities, and proposes a sampler (ASAp) that corrects this, the gap the Mask distortion card's two bars show.",
    },
    {
      authors: "Scott Lundberg and Marco Tulio Ribeiro",
      title: "Prompt Boundaries and Token Healing",
      source: "guidance library documentation notebook, GitHub",
      year: 2023,
      url: "https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb",
      note: "Written by the authors of the guidance library. It shows how a prompt that ends mid-token pushes a model toward odd continuations, and how token healing backs up one token and constrains the next pick to start with the removed text, as in the lesson's Token healing section.",
    },
    {
      authors: "Brandon T. Willard and Rémi Louf",
      title: "Efficient Guided Generation for Large Language Models",
      source: "arXiv preprint arXiv:2307.09702",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09702",
      note: "The paper behind the Outlines library named in the lesson. It turns a regular expression or grammar into a finite-state machine and builds an index of which vocabulary tokens are allowed in each state, the precomputation that makes masking cheap.",
    },
    {
      authors: "Yixin Dong, Charlie F. Ruan, Yaxing Cai, et al.",
      title: "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models",
      source: "Proceedings of Machine Learning and Systems 7 (MLSys 2025)",
      year: 2025,
      url: "https://proceedings.mlsys.org/paper_files/paper/2025/hash/5c20ca4b0b20b0bd2f1d839dc605e70f-Abstract-Conference.html",
      note: "The paper behind XGrammar, the other engine the lesson names. For context-free grammars it prechecks most tokens ahead of time and checks only the rest at each step, so the mask adds little time to decoding.",
    },
    {
      authors: "llama.cpp developers",
      title: "GBNF Guide",
      source: "llama.cpp documentation (grammars/README.md), GitHub",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md",
      note: "Shows how a grammar is written as rules in a BNF-style format to constrain a model's output, and how a JSON schema is converted into such a grammar. It also notes that the schema is not shown to the model, so the format should be described in the prompt too, as in the lesson's safety-net advice.",
    },
    {
      authors: "vLLM project",
      title: "Structured Outputs",
      source: "vLLM documentation",
      year: 2026,
      url: "https://docs.vllm.ai/en/latest/features/structured_outputs/",
      note: "Documents how a widely used inference engine accepts a JSON schema, regular expression, choice list, or grammar with a request, using XGrammar or guidance as the backend. It backs the lesson's Inference engines example of a mask running inside the decoding loop.",
    },
    {
      authors: "OpenAI",
      title: "Structured model outputs",
      source: "OpenAI API documentation",
      year: 2026,
      url: "https://developers.openai.com/api/docs/guides/structured-outputs",
      note: "A provider's guide to making responses follow a supplied JSON Schema. It warns that structured outputs can still contain mistakes and that a response can be incomplete at a max tokens limit, the lesson's points about shape, not truth, and about a length limit cutting the output off.",
    },
    {
      authors: "Anthropic",
      title: "Structured outputs",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/build-with-claude/structured-outputs",
      note: "Explains that structured outputs and strict tool use keep JSON responses and tool inputs schema-compliant through constrained decoding with compiled grammars. Its Invalid outputs section lists cases such as hitting max_tokens, where the output may not match the schema.",
    },
    {
      authors: "Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, et al.",
      title: "Let Me Speak Freely? A Study On The Impact Of Format Restrictions On Large Language Model Performance",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track (EMNLP 2024), 1218–1236",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-industry.91/",
      note: "Compares constrained JSON mode with format instructions in the prompt and with free answers. Strict formats often lowered scores on reasoning tasks while some classification tasks improved, which backs the lesson's warning that a valid output can still be a poor one.",
    },
  ],
  checkpoint: [
    {
      prompt: "On Masked distribution the model's favourite token is masked. What happens to the probabilities of the tokens that survive?",
      options: [
        "The masked token's probability is shared out equally among the legal tokens, whatever their ratios were",
        "They stay as they were, so the legal tokens now sum to less than one and the rest is simply ignored",
        "Each is divided by Z, the total the legal tokens had, so they sum to one and keep their ratios",
      ],
      answer: 2,
      explanation: "Masking sets illegal tokens to zero, equivalent to setting their logits to minus infinity before the softmax. Dividing the survivors by Z restores a total of one, and because every survivor is divided by the same number the model's ratios between legal tokens are unchanged. The removed mass is not redistributed evenly.",
      objective: 0,
    },
    {
      prompt: "A token such as { followed by a quote opens the object and starts the key in one step. How does the mask decide whether it is legal?",
      options: [
        "It allows only tokens that equal a single grammar symbol, so a token that spans two symbols is always masked",
        "It runs the token's characters through the matcher and checks the text is still a valid prefix",
        "It asks the model whether the token fits the format, and masks the ones the model scores below a threshold",
      ],
      answer: 1,
      explanation: "Tokens rarely line up with grammar symbols, so a correct mask checks character by character whether prefix plus token can still be completed into an accepted string. A mask limited to whole-symbol tokens would forbid common spellings the model rarely saw in training. The check is a matcher, not a judgement by the model.",
      objective: 0,
    },
    {
      prompt: "A constrained decoder returns valid JSON for your schema. What did the constraint guarantee?",
      options: [
        "Every token kept the text a valid prefix, so a finished output is guaranteed to fit the grammar",
        "The values inside the object are true, since the schema fixes each field's meaning",
        "The model's probabilities are unchanged by the mask, because only the output text is filtered",
      ],
      answer: 0,
      explanation: "The mask only removes tokens that would leave the grammar, so the text stays a valid prefix and can only end once it is complete. It says nothing about truth: a free string field accepts hello as a city. And the mask does change probabilities, both by renormalizing each step and by distorting which whole strings come out.",
      objective: 1,
    },
    {
      prompt: "The function-call grammar allows any letters for the city. What does that mean for get_weather(city=\"Os\")?",
      options: [
        "It cannot be produced, because a legal token is always a token the model believed was correct about the city",
        "It is guaranteed to parse, but nothing checks that Os is a real place, so the host must still validate it",
        "It will be rejected by the matcher, because a city that is not in the vocabulary is an illegal string",
      ],
      answer: 1,
      explanation: "Legality means the text is still a valid prefix of the grammar, and a free string field accepts any letters, so the call parses, runs, and asks about the wrong city. The grammar checks shape, never truth. Tools and agents therefore still validate arguments on the host side, as for any untrusted input.",
      objective: 1,
    },
    {
      prompt: "In Mask distortion you drag P(slo\" after \"O) from 10% to 90%. What happens to the two bars?",
      options: [
        "Both bars rise together, because both read the same probabilities from the slider control",
        "The per-step bar rises and the conditioned bar stays, because the mask uses the slider's probability",
        "The per-step bar does not move, while the conditioned bar rises until the two bars meet",
      ],
      answer: 2,
      explanation: "Per-step masking renormalizes over the legal first tokens and then over a single legal continuation, so the slider never enters it and it stays put. The conditioned bar multiplies the first-step probability by the slider's value, so it rises as that continuation becomes likely. At 90% both branches keep the same share legal and the bars meet.",
      objective: 2,
    },
    {
      prompt: "Why can masking token by token sample different strings than the model conditioned on producing valid output?",
      options: [
        "It commits to a prefix before seeing whether the model could continue it legally, over-weighting branches that are dead ends",
        "Masking nudges the model's weights a little at every step, so later tokens come from a slightly different model",
        "Renormalization shuffles the order of the legal tokens, so the draw no longer follows the ranking the model gave",
      ],
      answer: 0,
      explanation: "The mask looks one token ahead. A branch whose valid continuations are unlikely still gets its full share at the first step, because that step only compares legal first tokens, so such branches are over-represented. Weights never change and renormalization keeps the order and the ratios. Exact conditioning would need every legal future, which is intractable.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      const grammar: GrammarId = (GRAMMAR_IDS as readonly unknown[]).includes(parsed.grammar)
        ? (parsed.grammar as GrammarId)
        : "json";
      const maxStep = GRAMMARS[grammar].walk.length - 1;
      const step =
        typeof parsed.step === "number" && Number.isFinite(parsed.step)
          ? Math.min(maxStep, Math.max(0, Math.round(parsed.step)))
          : 0;
      const detour =
        typeof parsed.detour === "number" && Number.isFinite(parsed.detour)
          ? Math.min(1, Math.max(0.05, parsed.detour))
          : 0.1;
      return { ...initialState, ...parsed, grammar, step, detour };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
