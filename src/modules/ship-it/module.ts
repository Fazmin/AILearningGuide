import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_RECIPE } from "./pipeline";
import { normalizeRecipe, recipeToState } from "./state";

const initialState: ModuleState = recipeToState(DEFAULT_RECIPE);

const definition: ModuleDefinition = {
  id: "module-51-ship-it",
  slug: "ship-it",
  title: "Ship it",
  group: "training-adapting",
  order: 51,
  icon: "Workflow",
  accent: "#c2562f",
  prerequisites: [
    "module-20-train-tiny-lm",
    "module-22-lora-adapters",
    "module-24-preference-optimization",
    "module-25-evaluation",
    "module-26-quantization",
    "module-27-distillation-pruning",
    "module-28-deployment-serving",
  ],
  estimatedMinutes: 18,
  steps: ["Run the pipeline", "Read the gate", "Fix the regression", "Compress", "Budget and decide"],
  stepInstructions: [
    "In Stage by stage, compare Recipe-text loss and Harbor-text loss from Base model to After adapting, then to As shipped.",
    "In Eval gate, read which of the two checks fails and by how many nats, then compare it with the benchmark line underneath.",
    "Raise Replay sentences from none, one step at a time, and watch the Harbor check in Eval gate turn from fail to pass while recipe loss still falls.",
    "In Compress and serve, try each Weight format, then drag Prune toward 90% and watch As shipped drift away from After aligning.",
    "Set Concurrent users and Context length until the 8B-class budget fits, then read the verdict and its reasons in Release report.",
  ],
  stateVersion: 1,
  tagline:
    "Carry one small model through adapting, aligning, compressing and serving, and let a release gate decide whether it ships.",
  objectives: [
    "Trace one model through adapt, align, compress and serve, and name what each stage costs",
    "Gate a release on held-out loss for the target task and for a control task, and explain what a six-item benchmark cannot settle",
    "Fit the artifact you would ship to a memory and concurrency budget before calling it ready",
  ],
  glossary: [
    {
      term: "Release gate",
      definition:
        "A pass-or-fail check run on the artifact you would actually ship, with thresholds set before looking at the result. It decides whether a model goes out, so a gate that is loosened until it passes has stopped being a gate.",
    },
    {
      term: "Control set",
      definition:
        "Held-out text from a task the adaptation was not meant to change. It exists to catch what the target metric cannot see: a model can improve on the new task while quietly getting worse at the old one.",
    },
    {
      term: "Catastrophic forgetting",
      definition:
        "Loss of earlier ability when a model is trained on new data alone. Here it shows as harbor-text loss rising after the adapter trains on recipes only.",
    },
    {
      term: "Replay",
      definition:
        "Mixing some of the old data back into the adaptation data so the update has to keep serving it. It costs training signal for the new task and is the cheapest defence against forgetting.",
    },
    {
      term: "Alignment tax",
      definition:
        "Ability lost to preference training. Here, DPO raises the margin between the chosen and rejected answers while the held-out loss on both texts rises slightly.",
    },
    {
      term: "Held-out loss",
      definition:
        "Mean cross-entropy, in nats per character, on text the model never trained on. It is continuous and uses every character, so it separates close candidates that a handful of benchmark items cannot.",
    },
    {
      term: "Regression",
      definition:
        "A stage that makes something measured worse than the stage before it. The report lists every stage-to-stage rise above 0.02 nats so a loss does not hide inside a net gain.",
    },
    {
      term: "Not decisive",
      definition:
        "A benchmark difference whose 95% intervals overlap. Six items per benchmark means a real change of several items can still be inside the noise.",
    },
    {
      term: "Merged adapter",
      definition:
        "A low-rank update added into the base weights, so the shipped model is one table and costs nothing extra at inference. The adapter cannot be swapped out afterwards.",
    },
    {
      term: "Quantization",
      definition:
        "Storing weights in fewer bits per value. It shrinks the file and the memory read per token, at a rounding error that this lab measures through the gate.",
    },
    {
      term: "Pruning",
      definition:
        "Setting the smallest-magnitude weights to zero. Unstructured pruning keeps the file shape, so it saves speed only on hardware that skips zeros.",
    },
    {
      term: "Serving budget",
      definition:
        "Weights plus the KV cache for every concurrent stream plus runtime overhead, set against accelerator memory. The model fits only if all three do, at the load you actually expect.",
    },
  ],
  references: [
    {
      authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, et al.",
      title: "LoRA: Low-Rank Adaptation of Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2106.09685",
      note: "The paper behind Stage 2. It trains a small low-rank update while the base weights stay frozen, and shows the update can be merged into the weights so it adds no delay at inference, which is the merged adapter this lab ships.",
    },
    {
      authors: "Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, et al.",
      title: "LoRA Learns Less and Forgets Less",
      source: "Transactions on Machine Learning Research (TMLR), 2024",
      year: 2024,
      url: "https://arxiv.org/abs/2405.09673",
      note: "Fine-tunes real language models on code and math, then checks how much they lose on tasks outside that target. It is a full-size version of this lab's control set, and it shows that forgetting the old job is a real cost of adapting.",
    },
    {
      authors: "David Rolnick, Arun Ahuja, Jonathan Schwarz, et al.",
      title: "Experience Replay for Continual Learning",
      source: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1811.11682",
      note: "Explains catastrophic forgetting and shows that mixing stored old experience back into training greatly reduces it. The tests use game-playing agents rather than language models, but the idea is the same one the Replay sentences control applies.",
    },
    {
      authors: "Rafael Rafailov, Archit Sharma, Eric Mitchell, et al.",
      title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2305.18290",
      note: "The paper behind Stage 3, which the lesson names. It trains directly on pairs of preferred and rejected answers against a fixed reference model, with a beta setting like this lab's DPO beta.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2203.02155",
      note: "Uses the term alignment tax for the drop on standard tests that came with preference training. It also shows that mixing in some of the original training data shrinks that drop, much like replay does here.",
    },
    {
      authors: "Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh",
      title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2210.17323",
      note: "Rounds the weights of very large language models down to 3 or 4 bits with little loss in accuracy. It backs the lesson's finding that quantization is nearly free, while showing that a careful rounding method matters at real scale.",
    },
    {
      authors: "ggml-org and llama.cpp contributors",
      title: "llama.cpp quantize tool README",
      source: "llama.cpp documentation on GitHub",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md",
      note: "Shows how to turn a GGUF model into formats such as Q4_K_M, the starting Weight format in this lab, and lists file sizes before and after. It notes that the quality lost is measured with perplexity, a loss-based score like the one the gate reads.",
    },
    {
      authors: "Song Han, Jeff Pool, John Tran, and William J. Dally",
      title: "Learning both Weights and Connections for Efficient Neural Networks",
      source: "Advances in Neural Information Processing Systems 28 (NeurIPS 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1506.02626",
      note: "The standard source for magnitude pruning, which removes the smallest weights as the Prune slider does. Its networks keep their accuracy only because they are retrained after pruning, a step this lab skips.",
    },
    {
      authors: "Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter",
      title: "A Simple and Effective Pruning Approach for Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2306.11695",
      note: "Compares its method with plain magnitude pruning on large language models and reports that magnitude pruning breaks down fast as more weights are removed. It also measures real speedups only for a structured 2:4 pattern, which backs the point that unstructured zeros need special hardware to save time.",
    },
    {
      authors: "Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.",
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      source: "Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023), 611–626",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "The vLLM paper. It explains that the KV cache for each request is large and grows with the text, so it limits how many users can be served at once, which is what the Serving budget card adds up.",
    },
    {
      authors: "Evan Miller",
      title: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations",
      source: "arXiv preprint arXiv:2411.00640",
      year: 2024,
      url: "https://arxiv.org/abs/2411.00640",
      note: "Treats a benchmark as a sample of questions and shows how to put error bars on scores and on the gap between two models. It backs the benchmark line's 95% intervals and why a difference inside them is not decisive.",
    },
    {
      authors: "Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, et al.",
      title: "Lessons from the Trenches on Reproducible Evaluation of Language Models",
      source: "arXiv preprint arXiv:2405.14782",
      year: 2024,
      url: "https://arxiv.org/abs/2405.14782",
      note: "Written by the team behind EleutherAI's lm-evaluation-harness, the tool the lesson suggests. It explains how small changes in test setup move scores and gives practices for fair, repeatable comparisons between models.",
    },
    {
      authors: "Eric Breck, Shanqing Cai, Eric Nielsen, et al.",
      title: "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction",
      source: "Proceedings of IEEE Big Data 2017",
      year: 2017,
      url: "https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/",
      note: "A checklist of 28 tests from Google's production systems. It covers checking model quality before serving, canary releases to a small share of traffic, rolling a model back, and watching for drift after launch, the steps listed under What a real release adds.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1706.03762",
      note: "One of the four papers the lesson recommends. It introduced the Transformer, the design behind most of today's language models, including the kind of 8B-class model the Serving budget imagines.",
    },
    {
      authors: "Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al.",
      title: "Training Compute-Optimal Large Language Models",
      source: "arXiv preprint arXiv:2203.15556",
      year: 2022,
      url: "https://arxiv.org/abs/2203.15556",
      note: "Another of the four papers the lesson recommends. It works out how large a model should be and how much text it should see for a fixed training budget, the choice that comes before any of this lab's stages.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "With Replay sentences at none, the recipe text improves but the gate fails. What does the failing check catch?",
      options: [
        "The adapter made it worse on the harbor text it handled before",
        "The base model was trained for too few epochs to be a fair baseline",
        "The quantization format dropped too many bits to keep the recipe text",
      ],
      answer: 0,
      explanation:
        "The control check compares harbor-text loss with the base model's. Adapting on recipes alone moves the shared table away from harbor text, which is forgetting, and no target-task number would show it. The gate's second check exists for exactly this.",
      objective: 0,
    },
    {
      prompt:
        "Why does the gate read held-out loss rather than the count of benchmark items answered correctly?",
      options: [
        "Loss cannot be gamed, but a benchmark can always be gamed by the tuner",
        "Loss is continuous and uses every character; six items jump and overlap",
        "A quantized model cannot be scored on a benchmark, so loss is the only option",
      ],
      answer: 1,
      explanation:
        "A six-item benchmark can show the same count before and after a real improvement, and its 95% intervals overlap. Loss averages over every held-out character, so it separates close candidates. Loss can still be gamed by tuning on the held-out text, which is why that text is never trained on.",
      objective: 1,
    },
    {
      prompt:
        "A recipe passes the gate with 4-bit weights at the starting load. What must you still check before calling it ready to ship?",
      options: [
        "Whether the 4-bit file is smaller than the base model's file at full precision",
        "Whether the DPO margin is larger than the number of replay sentences used",
        "Whether weights, KV cache and overhead fit in memory at the load you expect",
      ],
      answer: 2,
      explanation:
        "Passing the quality gate says nothing about memory. The KV cache grows with users and context, so a model that fits at eight users can fail at sixty-four. The serving budget is a separate pass-or-fail condition on the same release.",
      objective: 2,
    },
    {
      prompt:
        "You raise the Prune slider to 90% and the gate fails on both checks. Which explanation matches what the lab shows?",
      options: [
        "Pruning only changes file size, so the gate failed because the control set was small",
        "Both losses rise, and recipe loss ends up nearly back at the base model's",
        "Pruning deletes the adapter first, which resets the model to the base exactly",
      ],
      answer: 1,
      explanation:
        "Magnitude pruning zeroes the smallest weights without regard to which text they serve. At 90% both losses rise, and the recipe gain nearly disappears. Pruning acts on the merged table, so it does not remove the adapter as a unit.",
      objective: 0,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      return recipeToState(normalizeRecipe(JSON.parse(value)));
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
