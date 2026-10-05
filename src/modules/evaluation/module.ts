import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeEvaluationState } from "./scoring";

const definition: ModuleDefinition = {
  id: "module-25-evaluation",
  slug: "evaluation",
  title: "Evaluation",
  group: "training-adapting",
  order: 47,
  icon: "ClipboardCheck",
  accent: "#3f7cc0",
  prerequisites: ["module-20-train-tiny-lm"],
  estimatedMinutes: 20,
  steps: [
    "Write the items",
    "Read the leaderboard",
    "Change the scoring rule",
    "Inspect the items",
    "Contaminate it",
  ],
  stepInstructions: [
    "Read the six default items in Benchmark items and note that each distractor is wrong but ordinary, then write one of your own.",
    "In Leaderboard, compare each checkpoint's pass rate with its 95% interval, then read the paired leader-versus-runner-up callout to see whether six items can separate them.",
    "In Leaderboard, switch Scoring rule to Raw log-probability and compare the pass counts with Length-normalised. Then choose Sample and check and raise Attempts allowed from 1 to 5 to see which checkpoint moves.",
    "In Item-level results, find the items the largest checkpoint gets wrong and read how large each margin is.",
    "Press Add the benchmark source sentences to the corpus, then compare Margin shift with Control shift.",
  ],
  stateVersion: 2,
  tagline:
    "Rank four checkpoints under three scoring rules with honest error bars, leak the benchmark into training and watch the leaderboard stop measuring ability, then see how the same ideas extend to applications.",
  objectives: [
    "Score a model with an explicit rule instead of an impression, and see how the choice of rule changes the ranking",
    "Read a benchmark score together with its uncertainty",
    "Recognize contamination from the gap between benchmark and control",
    "Choose what to measure for an application: golden sets for regressions, task metrics for retrieval, agents, and safety, and a checked judge",
  ],
  glossary: [
    {
      term: "Held-out set",
      definition:
        "Text excluded from training and used to score the result. Its only property that matters is that no part of it reached the training data.",
    },
    {
      term: "Perplexity",
      definition:
        "The exponential of mean cross-entropy on a corpus. Comparable across models only when the tokenizer and the corpus are identical.",
    },
    {
      term: "Likelihood scoring",
      definition:
        "Answering multiple choice by scoring each candidate continuation's log-probability and keeping the highest. No text is generated, so decoding settings cannot move the result. The raw total favours the shorter option, because every extra token multiplies in another probability below one; length normalization divides by the continuation's length, the idea behind acc_norm.",
    },
    {
      term: "Margin",
      definition:
        "The correct candidate's score minus the distractor's. It shows how nearly an item was missed, which a pass count hides.",
    },
    {
      term: "Confidence interval",
      definition:
        "A range that would contain the true pass rate in 95% of repeated samples of items. For accuracy p on n items the standard error is the square root of p(1 − p)/n, so six items at 50% give plus or minus 20 points.",
    },
    {
      term: "Contamination",
      definition:
        "Evaluation data present in the training data. It raises benchmark scores without raising capability, and usually arrives by duplication rather than intent. Overfitting the benchmark is its slower cousin: choosing data, settings, or checkpoints by one score until the score outruns the ability it was meant to measure.",
    },
    {
      term: "Control",
      definition:
        "A second unseen corpus that no leak touches. Without one, a benchmark gain and a contamination artifact look identical.",
    },
    {
      term: "Scoring rule",
      definition:
        "The function turning a model's output into a number: raw log-probability, length-normalised log-probability, or sampling answers and checking them are three rules that can order the same checkpoints differently. It is a design decision, so a score is comparable only under the same rule.",
    },
    {
      term: "Pass@k",
      definition:
        "For sampled answers a checker can verify: the chance that at least one of k samples passes. From n samples with c passing, the unbiased estimate is 1 − C(n − c, k) / C(n, k). It is the ceiling for a perfect checker, and it rewards a model that hedges as much as one that is right first time.",
    },
    {
      term: "Golden set",
      definition:
        "A fixed list of cases with known good outcomes, drawn from real traffic and past failures. Because it never changes, scores from different versions of a system are comparable, and its size limits how small a difference it can resolve.",
    },
    {
      term: "Regression suite",
      definition:
        "A golden set re-run on every change to the model, prompt, retriever, or tools, so a fix in one place cannot silently break another. Compare runs on the same cases and report an interval, not just a count.",
    },
    {
      term: "LLM-as-judge",
      definition:
        "Using a language model to grade outputs against a rubric. Cheap and scalable, and biased toward the first-listed answer, longer answers, and its own style. Check it against human labels on a sample, and swap the answer order to see whether verdicts flip.",
    },
    {
      term: "Human evaluation",
      definition:
        "Scoring by people, including arena-style pairwise votes turned into Elo ratings, where the winner gains K times how surprising the win was. It catches what rules miss and has its own taste for length, confidence, and formatting.",
    },
    {
      term: "Attack success rate",
      definition:
        "The share of attack prompts that elicit the prohibited behaviour. Read it with the false-refusal rate, the share of ordinary requests wrongly refused: a model that refuses everything scores 0% attack success and is useless.",
    },
  ],
  references: [
    {
      authors: "Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, et al.",
      title: "Lessons from the Trenches on Reproducible Evaluation of Language Models",
      source: "arXiv preprint arXiv:2405.14782",
      year: 2024,
      url: "https://arxiv.org/abs/2405.14782",
      note: "Written by the team behind lm-evaluation-harness. It explains likelihood scoring of answer choices, why the harness divides by answer length in bytes for acc_norm, and how one model's score on the same test can swing widely between prompt styles, which is why the lesson says a number needs its rule, harness, and version beside it.",
    },
    {
      authors: "Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta",
      title: "Interval Estimation for a Binomial Proportion",
      source: "Statistical Science 16(2)",
      year: 2001,
      url: "https://www.semanticscholar.org/paper/64596abdf95560ac4be87c71424d011eb1e51f78",
      note: "Shows that the usual p plus or minus 1.96 standard errors interval for a pass rate behaves badly, and recommends the Wilson interval when there are few items. That is the interval the Leaderboard draws for each pass rate.",
    },
    {
      authors: "Evan Miller",
      title: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations",
      source: "arXiv preprint arXiv:2411.00640",
      year: 2024,
      url: "https://arxiv.org/abs/2411.00640",
      note: "Treats a benchmark as a sample of questions and gives the standard error for a pass rate. It recommends comparing two models on paired, question-by-question differences, as the Leaderboard's paired callout does, and planning how many items a test needs, the idea behind Items for +/-5 pts.",
    },
    {
      authors: "Mark Chen, Jerry Tworek, Heewoo Jun, et al.",
      title: "Evaluating Large Language Models Trained on Code",
      source: "arXiv preprint arXiv:2107.03374",
      year: 2021,
      url: "https://arxiv.org/abs/2107.03374",
      note: "Introduces the HumanEval coding test, where generated programs are checked by unit tests. It gives the unbiased pass@k estimate, 1 - C(n - c, k) / C(n, k), used in the lesson and the Sample and check rule, and explains why the simpler plug-in formula is biased.",
    },
    {
      authors: "Oscar Sainz, Jon Campos, Iker García-Ferrero, et al.",
      title: "NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark",
      source: "Findings of the Association for Computational Linguistics: EMNLP 2023, 10776–10787",
      year: 2023,
      url: "https://aclanthology.org/2023.findings-emnlp.722/",
      note: "A short position paper on contamination: a model trained on a benchmark's test data scores higher than it should, and how often this happens is unknown because it is hard to measure. It backs the Contaminate it step and the glossary's point that a leak raises the score without raising ability.",
    },
    {
      authors: "Hugh Zhang, Jeff Da, Dean Lee, et al.",
      title: "A Careful Examination of Large Language Model Performance on Grade School Arithmetic",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
      year: 2024,
      url: "https://arxiv.org/abs/2405.00332",
      note: "Writes a fresh set of math problems in the same style as a popular benchmark and finds that some model families score lower on the new set, a sign of overfitting the benchmark. The fresh set plays the same role as the lab's control corpus.",
    },
    {
      authors: "Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, et al.",
      title: "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models",
      source: "Transactions on Machine Learning Research (TMLR), May 2023",
      year: 2023,
      url: "https://arxiv.org/abs/2206.04615",
      note: "The BIG-bench paper. Its section on including a canary string in all task files is a real example of the canary strings the lesson describes, a unique marker that helps people keep the test out of training data and check whether a model has seen it.",
    },
    {
      authors: "Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, et al.",
      title: "Are We Done with MMLU?",
      source: "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025), 5069–5096",
      year: 2025,
      url: "https://aclanthology.org/2025.naacl-long.262/",
      note: "Finds many wrong answer keys and flawed questions in a widely used multiple-choice benchmark, and shows that fixing them changes reported model scores. It backs the lesson's point that near the ceiling a benchmark starts measuring label noise.",
    },
    {
      authors: "Shahul Es, Jithin James, Luis Espinosa Anke, et al.",
      title: "RAGAs: Automated Evaluation of Retrieval Augmented Generation",
      source: "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL 2024), 150–158",
      year: 2024,
      url: "https://aclanthology.org/2024.eacl-demo.16/",
      note: "Scores a retrieval pipeline in separate parts, including faithfulness: whether each claim in the answer is supported by the retrieved text. It backs the lesson's advice to score retrieval and generation separately.",
    },
    {
      authors: "Carlos E. Jimenez, John Yang, Alexander Wettig, et al.",
      title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.06770",
      note: "A task set of real software issues. A model's fix counts as a success only if, after the patch is applied, the project's tests pass. That is the lesson's idea of checking an agent's task success on the end state.",
    },
    {
      authors: "Mantas Mazeika, Long Phan, Xuwang Yin, et al.",
      title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 35181–35224",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/mazeika24a.html",
      note: "A fixed suite of harmful behaviors used to compare attacks and defenses. It defines attack success rate as the share of test cases that get the behavior from the model, and shows that a setup detail, such as how many tokens the model may write, can change it.",
    },
    {
      authors: "Paul Röttger, Hannah Kirk, Bertie Vidgen, et al.",
      title: "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models",
      source: "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024), 5377–5400",
      year: 2024,
      url: "https://aclanthology.org/2024.naacl-long.301/",
      note: "Uses 250 safe prompts that only sound risky, plus unsafe prompts as a contrast, to measure how often models wrongly refuse. It backs the lesson's rule that attack success rate must be read with the false-refusal rate.",
    },
    {
      authors: "Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al.",
      title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track",
      year: 2023,
      url: "https://arxiv.org/abs/2306.05685",
      note: "Tests language models as graders and finds position, verbosity, and self-enhancement biases, measuring position bias by swapping the answer order. It compares judge-human agreement with human-human agreement, the check the lesson's LLM-as-judge section recommends.",
    },
    {
      authors: "Yann Dubois, Balázs Galambosi, Percy Liang, et al.",
      title: "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators",
      source: "Conference on Language Modeling (COLM 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2404.04475",
      note: "Shows that a popular model-graded benchmark favors longer answers, and removes that bias by asking what the verdict would be if both answers were the same length. It backs the lesson's warning that judges prefer longer answers.",
    },
    {
      authors: "Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, et al.",
      title: "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 8359–8388",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/chiang24b.html",
      note: "Describes the arena where people vote between two anonymous answers. It explains how the votes become a ranking with confidence intervals, using the Bradley-Terry model, a close relative of the Elo update in the lesson that the site showed earlier.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "In Leaderboard you switch Scoring rule from Length-normalised to Raw log-probability and several checkpoints gain passes. What is the most likely reason?",
      options: [
        "Raw log-probability favours the shorter option, and two correct answers are shorter than their distractors",
        "Raw log-probability removes the model's randomness, so each item is scored more reliably than before",
        "Raw log-probability makes each checkpoint train for more epochs before it is scored on the items",
        "Raw log-probability counts a tie between the two options as a pass, which the other rule does not",
      ],
      answer: 0,
      explanation:
        "Every extra character multiplies in another probability below one, so a total log-probability grows more negative with length and the shorter option wins more often whatever the model knows. Sails against trails and repeats against retreats are the two items here whose correct option is shorter. Dividing by length, the other rule, removes that head start.",
      objective: 0,
    },
    {
      prompt:
        "On Sample and check you raise Attempts allowed from 1 to 5, and the least-trained checkpoint climbs from last to first. Why does it gain the most?",
      options: [
        "Each extra attempt trains the checkpoint a little more, so by the fifth it has learned the items",
        "Extra attempts average out sampling noise, and noise is what held the least-trained model back",
        "A hesitant model gives every item some chance, so a few attempts nearly always include the key",
        "Pass@k rewards the checkpoint with the highest chance of being right on its very first attempt",
      ],
      answer: 2,
      explanation:
        "Pass@k asks only whether at least one of k samples is right, the ceiling a perfect checker would reach. A model that is unsure spreads probability over both options and hits within a few tries; a model that is confidently wrong on an item stays near zero however many attempts it gets. Pass@1 rewards the opposite. With two options the score is mostly coverage, which is why the rule is a design decision.",
      objective: 0,
    },
    {
      prompt:
        "Two checkpoints both pass 3 of 6 items, and the interval for the leader-minus-runner-up margin runs from below zero to above zero. What follows?",
      options: [
        "The two checkpoints are exactly as capable as each other, because their pass counts match",
        "These six items cannot say which is better; more items or a bigger gap would be needed",
        "The leader is better, because the middle of the interval sits just above zero",
        "The interval is too wide to use, so the standard error alone should be reported",
      ],
      answer: 1,
      explanation:
        "An interval that includes zero means the data are compatible with either checkpoint being ahead: the gap is unresolved, which is different from the two being equal. Pairing the items removes item difficulty from the comparison, but six items still leave it wide, and the standard error alone would hide that.",
      objective: 1,
    },
    {
      prompt:
        "After a data change, the benchmark margin grows five times over while the control corpus perplexity improves by two percent. What is the most likely explanation?",
      options: [
        "The benchmark text, or something close to it, reached the training data",
        "The model became better at exactly the skills the benchmark measures and nothing else",
        "Margins and perplexity measure different things, so nothing follows",
        "The control corpus is too small to move, so its two percent is only noise",
      ],
      answer: 0,
      explanation:
        "A real capability gain shows up on unseen text too, roughly in proportion. When the scored items improve far out of proportion to everything else, the most economical explanation is that those items stopped being unseen — which is why a control corpus is not optional.",
      objective: 2,
    },
    {
      prompt:
        "A retrieval-based assistant gives a wrong answer, and the passage that holds the answer was in its prompt. Which reading points at the generator rather than the retriever?",
      options: [
        "Context recall is low, so the retriever never fetched the supporting passage",
        "Attack success rate is high, so the generator is easy to push off task",
        "Pass@k is low at k = 1, so the generator needs more attempts per question",
        "Context recall is full, but answer correctness is low for those questions",
      ],
      answer: 3,
      explanation:
        "Context recall asks whether the facts needed reached the prompt; answer correctness asks whether the final answer matches the key. Full recall with a wrong answer means the evidence arrived and the reader failed to use it. Low recall is a retrieval failure, because the generator never saw the evidence. Faithfulness adds a third check, whether the answer's claims are supported by what was retrieved.",
      objective: 3,
    },
    {
      prompt:
        "A safety suite reports an attack success rate of 0% for a model. What must you check before calling it safe to ship?",
      options: [
        "Nothing more: a 0% rate means every attack in the suite failed, so the model is safe to ship as it stands",
        "Its false-refusal rate on ordinary requests, because a model that refuses everything also scores 0%",
        "Its perplexity on the control corpus, because successful attacks usually raise perplexity first",
        "Its pass@k at k = 1, because the first attempt is the one an attacker will use",
      ],
      answer: 1,
      explanation:
        "Attack success rate counts only how often the prohibited behaviour appears. A model that refuses everything never produces it, so 0% is also what an unusable model scores; the false-refusal rate on matched ordinary requests measures that cost, and the two are reported together. Neither number covers attacks that are not in the suite.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      // Every item is scored at every budget in a memo, so a shared link must not carry an unbounded
      // benchmark: the text is cut to MAX_ITEMS lines and every other field is validated.
      return sanitizeEvaluationState(JSON.parse(value));
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
