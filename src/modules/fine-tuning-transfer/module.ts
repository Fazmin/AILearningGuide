import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeState } from "./state";

const definition: ModuleDefinition = {
  id: "module-21-fine-tuning-transfer",
  slug: "fine-tuning-transfer",
  title: "Fine-tuning & transfer",
  group: "training-adapting",
  order: 41,
  icon: "Snowflake",
  accent: "#2b8ba8",
  prerequisites: ["module-20-train-tiny-lm"],
  estimatedMinutes: 19,
  steps: [
    "Compare the head start",
    "Push the adaptation",
    "Freeze some weights",
    "Read the forgetting meter",
    "Replay and hold out",
    "Decide whether to fine-tune",
  ],
  stepInstructions: [
    "At the default budget, compare the transfer curve with the from-scratch curve and note where the head start is largest.",
    "Move Fine-tune epochs from 0 to 60 and watch the target loss fall while the source loss climbs.",
    "Freeze one or two blocks on the Layer freezing card, then compare Transfer, target loss and Forgetting with the unfrozen run and check which rows stopped moving in the row strip.",
    "Read the forgetting meter: the two curves come from one model scored on two corpora.",
    "On the Replay and held-out target card, leave 3 target sentences held out and Replay share at 0, and compare Trained-on target with Held-out target. Then raise Replay share to 50% and compare the Forgetting rows for the replayed sentences and for the sentences not replayed.",
    "On the Should you fine-tune? card, start with all four answers on No, then flip one answer at a time and compare the recommended route and the row of the rule table that matched. It is a hand-written rule of thumb, not a measurement.",
  ],
  stateVersion: 3,
  tagline:
    "Start from pretrained weights instead of zero, then watch the same model gain on the new corpus and lose on the old one.",
  objectives: [
    "Explain why pretrained weights beat random ones on a small corpus",
    "Measure catastrophic forgetting rather than assume it",
    "Predict what freezing parameters does to adaptation and to forgetting",
    "Decide when fine-tuning is the right route and when prompting or retrieval should come first",
    "Use replay to protect old behaviour, and judge a fine-tune on target sentences it never trained on",
  ],
  glossary: [
    {
      term: "Transfer learning",
      definition:
        "Reusing weights trained on one distribution as the starting point for another. The benefit comes from structure the two distributions share.",
    },
    {
      term: "Full fine-tuning",
      definition:
        "Continuing ordinary training on new data with every parameter free to move. Simple, effective, and the most destructive option for prior behavior.",
    },
    {
      term: "Catastrophic forgetting",
      definition:
        "Loss on the original data rising while loss on the new data falls. There is one set of weights, and adapting them is the same operation as overwriting them.",
    },
    {
      term: "Freezing",
      definition:
        "Excluding parameters from the update. Frameworks usually skip computing their gradients too, which saves optimizer memory; either way it limits how far the model can drift, and also how well it can adapt.",
    },
    {
      term: "Forward transfer",
      definition:
        "The pretrained model already scoring better than chance on the new corpus before any adaptation, because the corpora share structure.",
    },
    {
      term: "Adaptation budget",
      definition:
        "Epochs times learning rate, roughly. Forgetting tracks the budget, so the cheapest defense against it is to stop early.",
    },
    {
      term: "Replay",
      definition:
        "Mixing some original data back into the fine-tuning set. The standard practical mitigation, at the cost of a slower fit to the new data. It protects what the replayed examples represent, and little else.",
    },
    {
      term: "Domain shift",
      definition:
        "The gap between the pretraining and target distributions. Descriptive prose to imperative instructions is a shift in style, not in vocabulary.",
    },
    {
      term: "Prompt engineering",
      definition:
        "Getting the behaviour you want from an unchanged model by changing its input: clearer instructions, worked examples, and what context is supplied. It needs no training data and is undone by editing text, which is why it comes before fine-tuning.",
    },
    {
      term: "Overfitting",
      definition:
        "Fitting the examples a model trained on more closely than new examples of the same kind. It shows as a gap between trained-on loss and held-out loss that widens with training, and it is largest when the fine-tuning set is small.",
    },
    {
      term: "Evaluation set",
      definition:
        "Examples held back from training and used to judge a model, covering both the new task and the old behaviour that must not be lost. Without one a fine-tune cannot be checked, so the decision rule counts it as part of being ready to fine-tune.",
    },
  ],
  references: [
    {
      authors: "Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson",
      title: "How transferable are features in deep neural networks?",
      source: "Advances in Neural Information Processing Systems 27 (NIPS 2014), 3320–3328",
      year: 2014,
      url: "https://arxiv.org/abs/1411.1792",
      note: "Measures how well each layer of a trained network carries over to a new task. Transferred weights beat random ones even when the two tasks are far apart, which is the forward transfer the lab shows at Fine-tune epochs 0.",
    },
    {
      authors: "Jeremy Howard and Sebastian Ruder",
      title: "Universal Language Model Fine-tuning for Text Classification",
      source: "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), 328–339",
      year: 2018,
      url: "https://aclanthology.org/P18-1031/",
      note: "Fine-tunes a pretrained language model and, with only 100 labeled examples, matches training from scratch on 100 times more data. It also introduces gradual unfreezing and discriminative learning rates, the softer forms of freezing in Going deeper.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online, Section 14.2",
      year: 2023,
      url: "https://d2l.ai/chapter_computer-vision/fine-tuning.html",
      note: "A worked fine-tuning example that races a pretrained model against one trained from scratch on a small dataset, like the lab's Transfer against training from scratch card. Its exercises also try freezing the pretrained layers.",
    },
    {
      authors: "James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, et al.",
      title: "Overcoming catastrophic forgetting in neural networks",
      source: "Proceedings of the National Academy of Sciences 114(13), 3521–3526",
      year: 2017,
      url: "https://arxiv.org/abs/1612.00796",
      note: "Introduces elastic weight consolidation, which slows learning on the weights that mattered most for old tasks. It is the weighted penalty against catastrophic forgetting that Going deeper describes.",
    },
    {
      authors: "Yun Luo, Zhen Yang, Fandong Meng, et al.",
      title: "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning",
      source: "arXiv preprint arXiv:2308.08747",
      year: 2023,
      url: "https://arxiv.org/abs/2308.08747",
      note: "Fine-tunes language models from 1 to 7 billion parameters on a sequence of instruction tasks and finds they lose domain knowledge, reasoning and reading comprehension. It shows the Forgetting meter's effect at real scale.",
    },
    {
      authors: "Ananya Kumar, Aditi Raghunathan, Robbie Jones, et al.",
      title: "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2202.10054",
      note: "Compares full fine-tuning with training only the last layer while the rest stays frozen. Each wins in a different setting, which backs the lesson's point that freezing is a dial between adaptation and keeping what was learned, not a fix.",
    },
    {
      authors: "Xuhong Li, Yves Grandvalet, and Franck Davoine",
      title: "Explicit Inductive Bias for Transfer Learning with Convolutional Networks",
      source: "Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80, 2825–2834",
      year: 2018,
      url: "https://proceedings.mlr.press/v80/li18a.html",
      note: "Tests penalties that keep fine-tuned weights close to the pretrained ones and recommends a simple squared-distance penalty. This is the weight-space regularizer in Going deeper; freezing is its extreme case.",
    },
    {
      authors: "David Rolnick, Arun Ahuja, Jonathan Schwarz, et al.",
      title: "Experience Replay for Continual Learning",
      source: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1811.11682",
      note: "Shows that mixing stored past experience back into training sharply cuts catastrophic forgetting while new tasks are still learned quickly. It is the idea behind the Replay share slider.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2203.02155",
      note: "The InstructGPT paper adds a per-token KL penalty that keeps the tuned model near its fine-tuned starting point. It also mixes pretraining data back into the updates to reduce regressions, a form of replay.",
    },
    {
      authors: "Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha",
      title: "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), 237–250",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-main.15/",
      note: "Finds that retrieval beats fine-tuning for adding knowledge and that models struggle to learn new facts by fine-tuning. It backs the Should you fine-tune? rule sending changing facts to retrieval.",
    },
    {
      authors: "Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al.",
      title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/2005.11401",
      note: "Pairs a language model with a searchable document index. It points out that knowledge stored in weights is hard to update or trace to a source, the reason the decision card picks retrieval when answers must cite.",
    },
    {
      authors: "Xiangyu Qi, Yi Zeng, Tinghao Xie, et al.",
      title: "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.03693",
      note: "Shows that fine-tuning a safety-trained chat model, even on harmless common datasets, can weaken its safety training and make it more willing to answer harmful requests. It backs the lesson's warning that forgetting reaches refusal behaviour that a loss on one corpus does not show.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Section 3.1 describes Llama 2-Chat's supervised fine-tune: 27,540 annotations, a starting learning rate of 2 x 10^-5, weight decay 0.1 and batch 64. These are the real-world numbers in Toy versus real.",
    },
    {
      authors: "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer",
      title: "QLoRA: Efficient Finetuning of Quantized LLMs",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2305.14314",
      note: "Fine-tunes large models with small LoRA adapters on a compressed base. Its hyperparameter table gives the learning rates quoted in Toy versus real, and its method leads into the next lab.",
    },
    {
      authors: "Hugging Face",
      title: "TRL documentation",
      source: "Hugging Face documentation",
      year: 2026,
      url: "https://huggingface.co/docs/trl/index",
      note: "The library behind SFTTrainer, with trainers for supervised fine-tuning, DPO and GRPO and a PEFT integration. It is the tool the lesson names for running this kind of fine-tune on real models.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Fine-tuning drives the target loss down and the source loss up. What is the mechanism?",
      options: [
        "The model deletes the old corpus from its memory to make room for the new corpus it is learning",
        "Every step optimizes only the target data, so the weights that held the source get overwritten",
        "Weights that go unused decay a little on every step, the way unrehearsed memories fade over time",
        "The source loss rises because the source data is no longer in the validation set",
      ],
      answer: 1,
      explanation:
        "Nothing in the objective mentions the original data, so nothing preserves it: the weights that encoded the source distribution are the same weights being overwritten. Freezing and replay both work by making some of the old behavior unavailable for overwriting or by putting the old data back in the loss.",
      objective: 1,
    },
    {
      prompt:
        "With Fine-tune epochs at 0, the pretrained model already beats the from-scratch model on Recipe steps. Why?",
      options: [
        "Recipe steps text was part of the pretraining data, so the model has effectively seen it already",
        "The pretrained table has more weights than the zeroed one, so it can fit more of the corpus right away",
        "Zero epochs still runs one pass over the target data, which gives the pretrained model a head start",
        "Both corpora share character statistics, so a Harbor-fitted table already beats chance on recipes",
      ],
      answer: 3,
      explanation:
        "A zeroed table gives every character equal probability, so its loss is exactly the log of the vocabulary size. Harbor and Recipe steps are both English at the character level, with similar letter frequencies and a dominant space, so weights fitted to Harbor already put sensible probability on recipe text. That is forward transfer: both tables hold the same 900 weights and neither has taken a step on recipes.",
      objective: 0,
    },
    {
      prompt:
        "You freeze the blocks labelled ␣abcde and fghijk on Layer freezing and keep the same epochs. What happens to Forgetting and to the transfer run's target loss?",
      options: [
        "Forgetting falls and target loss ends higher: frozen rows keep their Harbor predictions but cannot learn recipes",
        "Both improve, because fewer trainable weights means less noise in every update to the table",
        "Forgetting is unchanged, because the source loss depends only on the data and not on which weights move",
        "The target loss improves and forgetting grows, because the unfrozen rows have to move further to compensate for the frozen ones",
      ],
      answer: 0,
      explanation:
        "A frozen row keeps its pretrained next-character distribution, so what it predicted for Harbor survives, and what recipes need after those characters can no longer be learned. Each row's gradient depends only on that row, so the unfrozen rows move exactly as before and nothing compensates. Freezing is a dial between adaptation and forgetting, not a fix.",
      objective: 2,
    },
    {
      prompt:
        "Your assistant answers questions about a policy handbook that is revised every month. Each answer must cite its paragraph, and no special format is needed. What does the lab's rule of thumb recommend, and why?",
      options: [
        "Fine-tuning, because training the handbook into the weights makes answers faster and removes the need for sources",
        "A better prompt alone, because the handbook is only text and a prompt can hold any amount of it",
        "Retrieval, because a fine-tuned model's facts freeze when training stops and carry no source to cite",
        "Fine-tuning every month, because retraining is cheap once you have a few hundred examples",
      ],
      answer: 2,
      explanation:
        "Two answers on the decision card point the same way: the knowledge changes often and answers must cite sources. Weights are frozen when training stops, so every revision would need retraining, and a fine-tuned model cannot say which paragraph a fact came from. Retrieval reads the current text and can show the passage. The rule is a rule of thumb, and real decisions also weigh cost, latency, privacy and risk.",
      objective: 3,
    },
    {
      prompt:
        "The Forgetting readout is near zero after a fine-tune. Why can that still hide real damage in a production model?",
      options: [
        "It is computed on the target corpus rather than the source corpus, so it cannot see source behaviour at all",
        "It is reported only at the final checkpoint, so damage earlier in training goes unseen",
        "It measures how far the weights moved, and weights can move far without changing behaviour",
        "It is a loss on one corpus, while instruction following, format and refusals can degrade without moving it",
      ],
      answer: 3,
      explanation:
        "Forgetting here is the change in cross-entropy on one source corpus. Real forgetting also shows up in instruction following, output format, tool-call syntax, refusals and calibration, which a loss on generic text can miss, so production fine-tunes are checked against a suite of old behaviours rather than a single loss.",
      objective: 1,
    },
    {
      prompt:
        "On Replay and held-out target you raise Replay share from 0% to 50% and change nothing else. What do you expect to see?",
      options: [
        "Forgetting falls on the replayed sentences but barely moves on the ones not replayed, and trained-on target loss rises",
        "Forgetting falls equally on every original sentence, because the weights remember the whole original corpus once any of it is replayed",
        "Forgetting stays the same, because replay only adds training steps and leaves the loss unchanged",
        "Held-out target loss rises sharply, because replay takes training time away from the target sentences",
      ],
      answer: 0,
      explanation:
        "The loss now mentions the replayed sentences, so those are protected: their loss ends below where it started. Sentences that were not replayed are not in the loss, and they forget about as much as with no replay at all. Trained-on target loss rises a little because the weights are also pulled toward the replay, and the run takes twice the steps. Held-out target loss barely moves. Replay protects what it replays, so the replay set has to represent the behaviour you want to keep.",
      objective: 4,
    },
    {
      prompt:
        "As you add Fine-tune epochs, Trained-on target loss keeps falling while Held-out target loss flattens. What does the widening gap tell you?",
      options: [
        "The fine-tune is fitting the particular sentences it trained on, so the trained-on number overstates what it learned",
        "The held-out sentences are corrupted, because a loss that stops falling means the data is bad and should be cleaned",
        "The two lines cannot be compared, because held-out sentences are always harder than trained-on ones",
        "Falling trained-on loss proves the fine-tune generalizes, so the held-out line is the one to ignore",
      ],
      answer: 0,
      explanation:
        "Both lines score recipe sentences, but only one set ever entered a gradient step. The trained-on loss measures memory of those sentences; the held-out loss measures what carried over to new ones. With only a few training sentences the gap opens early and keeps widening. The earlier cards score the target corpus on the sentences it trained on, so their target loss is a training loss too.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      // Fine-tuning runs synchronously on every control change, so a shared link must
      // never be able to request an unbounded run: validate and clamp every field.
      return sanitizeState(JSON.parse(value));
    } catch {
      return sanitizeState(null);
    }
  },
};

export default definition;
