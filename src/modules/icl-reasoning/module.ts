import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { EXAMPLES, QUERIES } from "./icl";
import { MAX_BUDGET, MAX_HOPS, MIN_BUDGET, MIN_HOPS } from "./scratchpad";

const initialState: ModuleState = {
  examples: 2,
  strategy: "analogy",
  query: "florin",
  hops: 6,
  budget: 3,
  scratchpad: "off",
};

const num = (value: unknown, fallback: number, low: number, high: number, integer = false) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const bounded = Math.min(high, Math.max(low, value));
  return integer ? Math.round(bounded) : bounded;
};

/**
 * Version 1 stored { examples, budget, strategy }; version 2 added the voting card's accuracy, samples,
 * spread, and drawSeed. Examples, strategy, and query keep their meaning. The voting keys moved to
 * Reasoning models and are dropped. `budget` is now the scratchpad card's per-pass budget, so a version 1
 * thinking budget, which counted something else, is ignored rather than reinterpreted.
 */
export function migrateIclState(parsed: ModuleState): ModuleState {
  // Only a payload saved by this version carries hops; anything older has a budget that meant something else.
  const current = "hops" in parsed;
  return {
    examples: num(parsed.examples, initialState.examples as number, 0, EXAMPLES.length, true),
    strategy: parsed.strategy === "rule" ? "rule" : "analogy",
    query: typeof parsed.query === "string" && (QUERIES as readonly string[]).includes(parsed.query) ? parsed.query : "florin",
    hops: num(parsed.hops, initialState.hops as number, MIN_HOPS, MAX_HOPS, true),
    budget: current
      ? num(parsed.budget, initialState.budget as number, MIN_BUDGET, MAX_BUDGET, true)
      : (initialState.budget as number),
    scratchpad: parsed.scratchpad === "on" ? "on" : "off",
  };
}

const definition: ModuleDefinition = {
  id: "module-13-icl-reasoning",
  slug: "icl-reasoning",
  title: "ICL & reasoning",
  group: "frontiers",
  order: 30,
  icon: "BrainCircuit",
  accent: "#6c68c2",
  prerequisites: ["module-08-attention", "module-12-post-training"],
  estimatedMinutes: 16,
  steps: ["Give examples", "Infer the task", "Copy versus rule", "Write the steps down", "Find where it stops helping"],
  stepInstructions: [
    "Set Examples to 0 and read the answer bars for florin, then add examples one at a time and watch rules drop out of the table in Prompt as context.",
    "Set Query to letter. At four examples two rules still fit and disagree; add the fifth example, ballot, or set Prompt strategy to State a rule.",
    "In Copy from context, set Query to soravel with at least three examples, then to florin, and compare what the induction head predicts.",
    "In Chain-of-thought scratchpad, set Hops k to 6 and Per-pass budget b to 3 with Scratchpad off and read the answer, then switch Scratchpad on and compare Answer, Passes, and the scratch line.",
    "Read the No scratchpad and Scratchpad columns of the table under the passes at Per-pass budget b 3, then raise b and watch the first row where No scratchpad fails move down.",
  ],
  stateVersion: 3,
  tagline:
    "Treat a prompt's examples as evidence about which task you mean, see what copying from context can and cannot do, and watch a written scratchpad let a fixed amount of computation per pass follow a longer chain of lookups.",
  objectives: [
    "Describe in-context learning as inference from the prompt, with no weight update",
    "Separate copying an answer from context from inferring a rule from it",
    "Explain how a scratchpad lets a fixed per-pass computation follow a longer chain of dependent steps, and what it costs in passes",
  ],
  glossary: [
    {
      term: "In-context learning",
      definition:
        "Adapting behavior from examples in the prompt with no weight update. Nothing persists: clearing the context removes the ability entirely.",
    },
    {
      term: "Few-shot prompting",
      definition:
        "Supplying a handful of worked examples so the intended task becomes identifiable. Accuracy is sensitive to which examples, their order, and their formatting.",
    },
    {
      term: "Zero-shot",
      definition:
        "Asking for a task with instructions but no examples. Instruction tuning makes this work far better, though base models can manage some tasks zero-shot.",
    },
    {
      term: "Demonstration",
      definition:
        "One example pair in the prompt. Min et al. (2022) found that the format and label space of demonstrations can matter more than whether each label is correct.",
    },
    {
      term: "Task inference",
      definition:
        "Treating the examples as evidence about which task is meant. Rules that contradict an example drop out, and several rules can survive when the examples cannot tell them apart.",
    },
    {
      term: "Induction head",
      definition:
        "An attention head that finds an earlier occurrence of the current token and predicts the token that followed it. Their appearance during training coincides with a jump in in-context learning (Olsson et al., 2022).",
    },
    {
      term: "Prompt sensitivity",
      definition:
        "The measurable dependence of accuracy on details that should be irrelevant, including example order, formatting, label wording, and whitespace.",
    },
    {
      term: "Context window",
      definition:
        "The maximum number of tokens the model can attend over. It is working space, not memory, and retrieval accuracy can degrade for content buried in the middle of a long context.",
    },
    {
      term: "Chain of thought",
      definition:
        "Intermediate tokens generated before the answer, which later tokens can read back. They help most on problems with several dependent steps, because each token gets a fixed amount of computation and a result written down can be built on by the next one.",
    },
    {
      term: "Scratchpad",
      definition:
        "Text a model writes for itself before answering. A partial result written there sits in the context, so the next forward pass can start from it instead of recomputing it; padding that carries no partial result adds tokens and no progress.",
    },
    {
      term: "Serial computation",
      definition:
        "Work in which each step needs the result of the one before it, such as following a chain of lookups. A network of fixed depth can chain only so many such steps in one forward pass, and writing intermediate results as tokens lets the chain continue across passes (Li et al., 2024).",
    },
    {
      term: "Faithfulness",
      definition:
        "Whether a stated reasoning trace reflects the computation that produced the answer. Traces can omit the features that actually drove the result. Not the same as circuit faithfulness in Interpretability II: circuits, which asks whether a subgraph of model components reproduces the behavior on its own.",
    },
    {
      term: "Test-time compute",
      definition:
        "Compute spent while answering rather than while training. A scratchpad spends it as extra serial passes; sampling, voting, and verification spend it across several attempts and are priced in Reasoning models.",
    },
  ],
  references: [
    {
      authors: "Tom B. Brown, Benjamin Mann, Nick Ryder, et al.",
      title: "Language Models are Few-Shot Learners",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 1877–1901",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html",
      note: "The GPT-3 paper. It sets out the zero-shot and few-shot settings and shows a large model doing new tasks from examples in the prompt with no gradient updates, which is the starting claim of this lesson.",
    },
    {
      authors: "Qingxiu Dong, Lei Li, Damai Dai, et al.",
      title: "A Survey on In-context Learning",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), 1107–1128",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-main.64/",
      note: "A survey that defines in-context learning and reviews how demonstrations are chosen, formatted, and ordered, and what is known about why it works. Use it as a map of the topics in this lesson.",
    },
    {
      authors: "Sang Michael Xie, Aditi Raghunathan, Percy Liang, et al.",
      title: "An Explanation of In-context Learning as Implicit Bayesian Inference",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2111.02080",
      note: "Argues that a model learns in context by inferring which hidden concept the prompt's examples share. That is the idea behind task inference and the rule weights on the Prompt as context card.",
    },
    {
      authors: "Catherine Olsson, Nelson Elhage, Neel Nanda, et al.",
      title: "In-context Learning and Induction Heads",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2022,
      url: "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
      note: "Describes induction heads, which complete [A][B] ... [A] with [B], and finds they form at the same point in training as a sudden jump in in-context learning. The Copy from context card runs a hand-set version of this head.",
    },
    {
      authors: "Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, et al.",
      title: "Show Your Work: Scratchpads for Intermediate Computation with Language Models",
      source: "arXiv preprint arXiv:2112.00114",
      year: 2021,
      url: "https://arxiv.org/abs/2112.00114",
      note: "Introduces the scratchpad: the model writes intermediate steps before answering. This helps on multi-step tasks such as long addition and running programs, the same idea as the Scratchpad switch on the Chain-of-thought scratchpad card.",
    },
    {
      authors: "Jason Wei, Xuezhi Wang, Dale Schuurmans, et al.",
      title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 24824–24837",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html",
      note: "The paper that named chain-of-thought prompting. Few-shot examples that show intermediate steps improve large models on arithmetic, commonsense, and symbolic reasoning tasks.",
    },
    {
      authors: "Zhiyuan Li, Hong Liu, Denny Zhou, et al.",
      title: "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2402.12875",
      note: "The theory result this lesson cites as Li et al., 2024. A transformer of fixed depth cannot do some serial computation in one pass but can with enough chain-of-thought steps, which the per-pass budget b on the scratchpad card turns into a toy.",
    },
    {
      authors: "Tony Z. Zhao, Eric Wallace, Shi Feng, et al.",
      title: "Calibrate Before Use: Improving Few-Shot Performance of Language Models",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 12697–12706",
      year: 2021,
      url: "https://proceedings.mlr.press/v139/zhao21c.html",
      note: "Shows that few-shot accuracy can swing from near chance to near the best results just by changing the prompt format, which examples are used, or their order. It backs the glossary entry on prompt sensitivity.",
    },
    {
      authors: "Melanie Sclar, Yejin Choi, Yulia Tsvetkov, et al.",
      title: "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.11324",
      note: "Changes only meaning-preserving formatting, such as separators, capitalization, and spacing, and finds large accuracy differences in few-shot prompts. It supports the lesson's point that formatting and even whitespace can move accuracy.",
    },
    {
      authors: "Sewon Min, Xinxi Lyu, Ari Holtzman, et al.",
      title: "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?",
      source: "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022), 11048–11064",
      year: 2022,
      url: "https://aclanthology.org/2022.emnlp-main.759/",
      note: "The Min et al. (2022) finding named in this lesson: swapping in random labels barely hurts few-shot accuracy, while the label space, the input text, and the format of the demonstrations matter more.",
    },
    {
      authors: "Nelson F. Liu, Kevin Lin, John Hewitt, et al.",
      title: "Lost in the Middle: How Language Models Use Long Contexts",
      source: "Transactions of the Association for Computational Linguistics 12, 157–173",
      year: 2024,
      url: "https://aclanthology.org/2024.tacl-1.9/",
      note: "Finds that models use information best at the start or end of a long input and worse in the middle. It backs the glossary point that a context window is working space, not reliable memory.",
    },
    {
      authors: "Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, et al.",
      title: "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2409.12183",
      note: "The comparison this lesson cites as Sprague et al., 2024. Across many papers and its own tests, chain of thought helps mostly on math and logic and much less on other tasks.",
    },
    {
      authors: "Miles Turpin, Julian Michael, Ethan Perez, et al.",
      title: "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 74952–74965",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html",
      note: "Adds a hidden bias to the prompt, such as making the answer to every example (A), and finds that models follow it but fail to mention it in their chain of thought. This is the faithfulness warning in Where it breaks.",
    },
    {
      authors: "Tamera Lanham, Anna Chen, Ansh Radhakrishnan, et al.",
      title: "Measuring Faithfulness in Chain-of-Thought Reasoning",
      source: "arXiv preprint arXiv:2307.13702",
      year: 2023,
      url: "https://arxiv.org/abs/2307.13702",
      note: "Tests faithfulness by editing a model's chain of thought and seeing whether the answer changes. Replacing the reasoning with filler dots gave no accuracy gain, which matches the lesson's point that padding adds tokens but no progress.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "The prompt shows glim → 4, plint → 5, soravel → 7, trelix → 6, then asks letter →. A model that fits every example answers 4. What is the best explanation?",
      options: [
        "It miscounted the word, because the examples clearly define the task as counting every letter",
        "Counting distinct letters fits every example as well as counting letters, and letter repeats letters",
        "Reading the examples updated its weights, so it switched to a different rule just for this one word",
        "It copied the earlier answer 4, because the arrow after letter matches the arrow after the glim line",
      ],
      answer: 1,
      explanation:
        "Every example word has no repeated letters, so \"count the letters\" and \"count distinct letters\" give identical answers on all four. They only disagree on a word like letter (6 versus 4). The examples underdetermine the task; an example such as ballot → 6, or a stated rule, resolves it. Nothing about the weights changed.",
      objective: 0,
    },
    {
      prompt:
        "On Copy from context you switch Query from soravel to florin at four examples. What does the induction head do, and what does that show?",
      options: [
        "For soravel it copies the earlier answer; for florin it can only spread its weight over the old answers",
        "For florin it computes the letter count from the word, because copying and counting are the same step",
        "For both words it learns the rule from the examples and updates its weights to apply it",
        "For soravel it fails, because the head only attends to words that have never appeared in the prompt",
      ],
      answer: 0,
      explanation:
        "The head finds an earlier line that starts with the same word and copies what followed it. For soravel that line exists, so it retrieves 7. For florin no earlier line starts with that word, so every earlier answer gets similar weight: copying explains the format, a number after the arrow, but not the count. Applying a rule to a new word needs more than this one circuit.",
      objective: 1,
    },
    {
      prompt:
        "In Chain-of-thought scratchpad, Hops k is 6 and Per-pass budget b is 3 with Scratchpad off, and the answer is wrong. What does switching Scratchpad on change, and why?",
      options: [
        "The model gets more computation per pass, so it can now chain all six lookups in a single pass",
        "The table is stored in the scratchpad, so each lookup becomes free and no extra passes are needed",
        "Extra tokens add computation by themselves, whatever they say, so any longer output would also work",
        "Pass 1 writes the node it reached, and pass 2 starts from that token and finishes the chain",
      ],
      answer: 3,
      explanation:
        "One pass can chain at most b lookups before it must emit a token. With the scratchpad on, pass 1 does three lookups and writes the node it reached; pass 2 reads that token from the context, does the remaining three, and answers. The extra computation comes from carrying a partial result across passes, not from the number of tokens, and padding that carries no result would leave the answer wrong.",
      objective: 2,
    },
    {
      prompt:
        "With Per-pass budget b at 3, for which hop counts k does switching the scratchpad on change whether the answer is right?",
      options: [
        "Every count, because a scratchpad always makes a model more accurate, whatever the task is",
        "Only counts below 3, because only a short chain is small enough for a scratchpad to store",
        "Only counts above 3: up to 3, one pass is already right, and past that only the scratchpad fixes it",
        "None, because the budget fixes what any model can compute, with or without a scratchpad at all",
      ],
      answer: 2,
      explanation:
        "Up to b hops one pass finishes the chain, so both settings agree and the scratchpad adds nothing. Past b, the single pass stops short and answers wrong, while the scratchpad stays right by spending ceil(k / b) passes, so the benefit comes with a cost in latency that grows with the length of the chain.",
      objective: 2,
    },
    {
      prompt:
        "Where does the scratchpad toy stop matching a real model?",
      options: [
        "The limit b is imposed as a clean integer, and the trace is faithful by construction",
        "A real model has no limit per pass, so a scratchpad can never help it on any task at all",
        "A real model's visible reasoning is always a complete and honest log of everything it computed",
        "A real model writes exactly one token for each hop of reasoning, so its cost is always k passes",
      ],
      answer: 0,
      explanation:
        "In the toy, b is a number the lab sets, and the next pass reads only what was written, so the written trace has to be what carries the answer. A real model's limit comes from its depth and training and is not a clean integer, and nothing forces its visible reasoning to match what it computed: it can omit the feature that drove the answer. Treat a trace as an output to check, and see faithfulness in the glossary and Reasoning models.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      return migrateIclState(parsed);
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
