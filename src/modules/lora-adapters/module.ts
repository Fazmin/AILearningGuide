import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import { CALCULATOR_STATE, calculatorState, readCalculator } from "./calculator";
import Explore from "./Explore";

/**
 * Version 1 stored a heatmap toggle, `view`, which no longer exists. Version 3 adds the six keys of the
 * Adapter size calculator, which sits on a card of its own and changes nothing the other cards compute, so
 * an older payload takes their defaults and keeps its meaning.
 */
const initialState: ModuleState = {
  rank: 4,
  alpha: 8,
  epochs: 24,
  adapterOn: true,
  merged: false,
  probe: " ",
  ...CALCULATOR_STATE,
};

const PROBES = [" ", "e", "s", "l", "a"];

const definition: ModuleDefinition = {
  id: "module-22-lora-adapters",
  slug: "lora-adapters",
  title: "LoRA & adapters",
  group: "training-adapting",
  order: 42,
  icon: "Puzzle",
  accent: "#a8559b",
  prerequisites: ["module-21-fine-tuning-transfer"],
  estimatedMinutes: 17,
  steps: ["Size the update", "Train the adapter", "Hot-swap it", "Merge it", "Size a real adapter"],
  stepInstructions: [
    "Move Rank from 1 to 12 and watch B widen, A deepen, and the trainable count grow as r · (d + k) against the full 900.",
    "Set Adapter epochs to 0 and confirm B and ΔW are blank and the loss sits on the frozen-base line, then raise it and watch both change.",
    "Switch Serving between Base only and Base + adapter, and compare Target loss, Source loss and the sample text.",
    "Press Merge the adapter into W and read the logit gap, the unchanged loss, and what you can no longer do.",
    "On the Adapter size calculator, keep Llama 3 8B with every matrix targeted at rank 16, then switch off the MLP and compare Trainable parameters and Share of the base model. Change Adapter precision from 16-bit to 32-bit and read the size, then compare the toy's share with the real one in the callout.",
  ],
  stateVersion: 3,
  tagline:
    "Freeze the whole weight table and train two thin matrices instead. Then swap the adapter in and out, and fold it in for free inference.",
  objectives: [
    "Explain what B·A adds to a frozen weight matrix",
    "Trade rank against fit and parameter count",
    "Say what merging an adapter gains and what it forecloses",
    "Explain why a freshly initialized adapter changes nothing until it is trained",
    "Compute an adapter's trainable parameters, share of the base model, and file size from an architecture, and compare them with the toy",
  ],
  glossary: [
    {
      term: "LoRA",
      definition:
        "Low-rank adaptation. The base weights stay frozen and the update is constrained to the product of two thin matrices, so only those are trained and shipped.",
    },
    {
      term: "Rank",
      definition:
        "The inner dimension r shared by the two adapter matrices. The update B times A can have at most r independent directions, and a d by k weight matrix gets r times (d plus k) trainable numbers instead of d times k.",
    },
    {
      term: "Alpha",
      definition:
        "A constant that scales the update by alpha over rank. Holding alpha fixed while changing rank keeps the update's size roughly steady, so a learning rate tuned at one rank still works at another; it also means a rank sweep at fixed alpha changes the scale.",
    },
    {
      term: "Zero initialization",
      definition:
        "Starting one adapter matrix at zero so the initial update is exactly zero. Training therefore begins from the base model's behaviour rather than a perturbation of it.",
    },
    {
      term: "Hot-swapping",
      definition:
        "Attaching or detaching an adapter at serving time. Possible only while the adapter is stored separately from the base weights.",
    },
    {
      term: "Merging",
      definition:
        "Adding the update into the base matrix once. Inference then costs exactly what the base cost, and the adapter can no longer be removed.",
    },
    {
      term: "Target modules",
      definition:
        "Which matrices get adapters. In a transformer these are usually the attention projections, sometimes the MLP too; more targets means more capacity and more parameters.",
    },
    {
      term: "Adapter",
      definition:
        "Any small trainable module added to a frozen network. Earlier bottleneck adapters add layers that stay in the forward pass; a LoRA update is linear, so it can be merged and add nothing at inference.",
    },
    {
      term: "QLoRA",
      definition:
        "LoRA on top of a base model stored in 4-bit NormalFloat (NF4). The frozen weights are dequantized on the fly for each matrix multiply, and only the 16-bit adapter is trained, which cuts the memory for the base by about four times.",
    },
    {
      term: "PEFT",
      definition:
        "Parameter-efficient fine-tuning: training a small set of added or selected parameters while the base stays frozen. LoRA is one method, and prompt tuning, prefix tuning, IA3 and DoRA are others. Hugging Face PEFT is a library that implements them.",
    },
    {
      term: "Trainable fraction",
      definition:
        "Trainable parameters divided by the base model's parameters. A LoRA pair on one matrix costs rank times the sum of its dimensions against the product of them, so the fraction falls as matrices widen: 26.7% for the toy at rank 4, well under one percent on a real model.",
    },
    {
      term: "Multi-adapter serving",
      definition:
        "Keeping one copy of the base model in memory and choosing a different adapter for each request, instead of merging every adapter into its own copy of the weights. It trades the free inference of merging for the work of scheduling many small files.",
    },
  ],
  references: [
    {
      authors: "Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, et al.",
      title: "Parameter-Efficient Transfer Learning for NLP",
      source: "Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, 2790–2799",
      year: 2019,
      url: "https://proceedings.mlr.press/v97/houlsby19a.html",
      note: "The paper that introduced adapter modules: small trainable layers added to a frozen BERT, so each new task adds only a few parameters. These are the earlier bottleneck adapters the lesson contrasts with LoRA, because they stay in every forward pass.",
    },
    {
      authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, et al.",
      title: "LoRA: Low-Rank Adaptation of Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2106.09685",
      note: "The LoRA paper. Section 4.1 freezes W, trains B and A, starts B at zero so the update begins at zero, scales it by alpha over rank, and folds BA into W for no extra inference cost. Section 7 asks which attention matrices to adapt and reports the 10,000 times cut in trainable parameters for GPT-3 175B.",
    },
    {
      authors: "Hugging Face",
      title: "LoRA",
      source: "Hugging Face PEFT documentation, conceptual guides",
      year: 2026,
      url: "https://huggingface.co/docs/peft/main/en/conceptual_guides/lora",
      note: "A plain explanation of LoRA's two update matrices and the settings r, lora_alpha and target_modules, the same rank, alpha and target modules this lesson uses. It also shows merge_and_unload for folding the adapter into the base, and the use_rslora option.",
    },
    {
      authors: "Damjan Kalajdzievski",
      title: "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA",
      source: "arXiv preprint arXiv:2312.03732",
      year: 2023,
      url: "https://arxiv.org/abs/2312.03732",
      note: "The rsLoRA paper. It argues that dividing the update by the rank slows learning at large ranks and that dividing by the square root of the rank is steadier, which backs the lesson's note on what Alpha is for.",
    },
    {
      authors: "Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, et al.",
      title: "LoRA Learns Less and Forgets Less",
      source: "Transactions on Machine Learning Research (TMLR)",
      year: 2024,
      url: "https://arxiv.org/abs/2405.09673",
      note: "Compares LoRA with full fine-tuning on code and math. LoRA falls well behind on these targets but keeps more of the base model's other skills, and full fine-tuning learns updates of much higher rank. It backs the lesson's point that parity is a property of the task, and contrasts with the toy, where the adapter raises Source loss more.",
    },
    {
      authors: "Reece Shuttleworth, Jacob Andreas, Antonio Torralba, and Pratyusha Sharma",
      title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence",
      source: "Advances in Neural Information Processing Systems (NeurIPS 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2410.21228",
      note: "Shows that LoRA and full fine-tuning can score alike while changing the weights in very different ways, and links that difference to forgetting. It matches what the toy's spectrum chart shows: the adapter found a different update from the full fine-tune, not a copy of it.",
    },
    {
      authors: "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer",
      title: "QLoRA: Efficient Finetuning of Quantized LLMs",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 10088–10115",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html",
      note: "Trains LoRA adapters on a frozen base stored in 4-bit NormalFloat (NF4), with double quantization to save more memory. This is the source for the QLoRA glossary term and the lesson's claim that a 65B model can be fine-tuned on one 48 GB GPU.",
    },
    {
      authors: "Brian Lester, Rami Al-Rfou, and Noah Constant",
      title: "The Power of Scale for Parameter-Efficient Prompt Tuning",
      source: "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), 3045–3059",
      year: 2021,
      url: "https://aclanthology.org/2021.emnlp-main.243/",
      note: "Introduces prompt tuning: a frozen model is steered by learned soft prompts at the input. It backs the first entry under Other ways to adapt, and finds the method catches up with full tuning as models grow.",
    },
    {
      authors: "Xiang Lisa Li and Percy Liang",
      title: "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
      source: "Proceedings of the 59th Annual Meeting of the ACL and the 11th IJCNLP (ACL-IJCNLP 2021), 4582–4597",
      year: 2021,
      url: "https://aclanthology.org/2021.acl-long.353/",
      note: "Introduces prefix tuning, which keeps the model frozen and trains a short run of task vectors that later tokens attend to. It backs the lesson's prefix tuning entry, and trains about 0.1% of the parameters.",
    },
    {
      authors: "Haokun Liu, Derek Tam, Mohammed Muqeeth, et al.",
      title: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 1950–1965",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html",
      note: "The paper that introduced IA3, which trains small vectors that rescale a frozen model's activations. Its learned vectors rescale the keys, the values and the inner feed-forward activations, as the lesson's IA3 entry says.",
    },
    {
      authors: "Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, et al.",
      title: "DoRA: Weight-Decomposed Low-Rank Adaptation",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 32100–32121",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/liu24bn.html",
      note: "Splits each pretrained weight into a size and a direction, trains the direction with LoRA, and learns the size separately. It backs the lesson's DoRA entry under Other ways to adapt.",
    },
    {
      authors: "Hugging Face",
      title: "Quicktour",
      source: "Hugging Face PEFT documentation",
      year: 2026,
      url: "https://huggingface.co/docs/peft/quicktour",
      note: "Wraps Llama 3.2 1B with a rank-8 adapter on the q projections and prints 524,288 trainable parameters out of 1,236,338,688, the figure the Adapter size calculator reproduces. It also shows save_pretrained, and add_adapter and set_adapter for switching named adapters on one base.",
    },
    {
      authors: "vLLM project",
      title: "LoRA Adapters",
      source: "vLLM documentation",
      year: 2026,
      url: "https://docs.vllm.ai/en/latest/features/lora/",
      note: "Shows a server started with --enable-lora, adapters registered by name, and requests that pick an adapter through the model field, with settings such as max_loras. It backs the lesson's section on multi-adapter serving.",
    },
    {
      authors: "Ying Sheng, Shiyi Cao, Dacheng Li, et al.",
      title: "SLoRA: Scalable Serving of Thousands of LoRA Adapters",
      source: "Proceedings of Machine Learning and Systems 6 (MLSys 2024), 296–311",
      year: 2024,
      url: "https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html",
      note: "Describes a system that keeps all adapters in main memory and moves the ones in use onto the GPU, with batched kernels for many adapters at once. It reports up to four times the throughput of PEFT and vLLM baselines, the S-LoRA result the lesson quotes.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "Table 3 lists the layers, model dimension, feed-forward width and key/value heads of Llama 3 8B and 70B. These are the shapes the Adapter size calculator multiplies to get Trainable parameters and Share of the base model.",
    },
  ],
  checkpoint: [
    {
      prompt: "While the adapter trains, which terms of W + (alpha / rank) · B · A are ever updated?",
      options: [
        "W and B, while A stays at its random start so the update keeps its variety",
        "All three, with W changing the least because it was already trained",
        "Only W, with B · A recomputed from it after every optimizer step",
        "Only B and A: W is read on every forward pass and never written",
      ],
      answer: 3,
      explanation:
        "Only the two thin matrices receive gradients; the base table is read on every forward pass and written on none, which is why detaching the adapter restores the base exactly. The effective weights are a sum, so the model that is served changes even though W does not.",
      objective: 0,
    },
    {
      prompt: "At Alpha 8, rank 12 fits Recipe steps worse than rank 8. What is the most careful reading?",
      options: [
        "Rank 12 overfits, because an adapter always has more parameters than the full fine-tune",
        "Rank and the scale alpha over rank changed together, so the sweep cannot tell their effects apart",
        "Higher rank always means more capacity, so rank 12 must fit better and the sweep has a bug",
        "Extra rank directions each need their own training steps, so rank 12 has simply not converged yet at this budget",
      ],
      answer: 1,
      explanation:
        "At a fixed alpha, raising rank also lowers the scale alpha over rank, so every update is shrunk while the adapter gains directions. Two things moved at once. Pressing Set alpha = rank holds the scale at 1.00, and then the extra directions pay off: rank 12 edges below rank 8 instead of above it.",
      objective: 1,
    },
    {
      prompt: "After you press Merge the adapter into W, recipe loss is unchanged. What did merging buy, and what did it cost?",
      options: [
        "The merged model fits better than the separate pair, but the adapter can no longer be trained further",
        "The adapter file becomes smaller, but the base weights now have to be stored twice",
        "Inference needs no extra B · A multiply, but the adapter can no longer be detached or swapped",
        "Forgetting disappears from the merged table, but the rank is capped at what was trained",
      ],
      answer: 2,
      explanation:
        "B · A is linear, so its product can be folded into W once and the result is a single matrix: nothing extra to compute at inference, and the loss agrees with the unmerged path to float rounding. The cost is that the update is now mixed into every entry, so there is no adapter left to switch off or replace.",
      objective: 2,
    },
    {
      prompt: "Why does a freshly initialized LoRA adapter leave the model's output completely unchanged?",
      options: [
        "One matrix starts at zero, so B · A is zero and the effective weights equal the base weights",
        "The adapter is switched off by a flag, and the first training step switches it on",
        "The scaling factor alpha over rank starts at zero and grows during training",
        "Both matrices start as tiny random numbers, so their product is too small to change any prediction",
      ],
      answer: 0,
      explanation:
        "B starts at zero and A starts random, so B · A is exactly zero and the effective weights are exactly the base weights; that is why Adapter epochs at 0 reads the base loss. Training still moves, because B's gradient is proportional to the random A: B moves first, and A only receives gradient once B is non-zero. Adaptation begins from the pretrained behaviour rather than from a random perturbation of it.",
      objective: 3,
    },
    {
      prompt:
        "The base weights W are never written during LoRA training. Why does Source loss still rise when the adapter is on?",
      options: [
        "The optimizer secretly updates W through the product B · A on every step",
        "The served model is W plus the adapter's contribution, so behaviour changes while W does not",
        "The adapter erases Harbor from W, and detaching it only hides that damage",
        "Frozen weights are re-initialized at the start of every forward pass, which hurts old tasks too",
      ],
      answer: 1,
      explanation:
        "LoRA protects the base weights, not the base model's behaviour. Whatever the adapter adds to each logit is part of every prediction while it is attached, and at a scale of 2.00 it amplifies its update, so Harbor loss rises. Detaching the adapter returns Harbor loss to its original value exactly, because W was never touched.",
      objective: 0,
    },
    {
      prompt:
        "On the Llama 3 8B preset at rank 16 you target only q and v, then also switch on the MLP. What happens to Trainable parameters?",
      options: [
        "It rises more than fivefold, because the MLP's three matrices are far wider than the attention ones",
        "It roughly doubles, because the MLP is one more matrix alongside q and v",
        "It stays the same, because the rank alone sets the number of trainable parameters, whichever matrices are adapted",
        "It falls, because adapting more matrices lets a smaller rank do the same job",
      ],
      answer: 0,
      explanation:
        "Each adapted matrix costs rank times the sum of its input and output widths. The attention matrices are 4096 wide by 4096 or 1024, while the MLP block has three matrices between 4096 and 14336, so adding it multiplies the count by 5.15, from 6,815,744 to 35,127,296. Which matrices are targeted changes the count as much as the rank does.",
      objective: 4,
    },
    {
      prompt:
        "The toy's rank-4 adapter trains 27% of its table. At rank 4 on the Llama 3 8B preset with every matrix targeted, what does the calculator report, and why?",
      options: [
        "Well under one percent, because a pair costs rank times the sum of a matrix's sides while the matrix costs their product",
        "About 27% as well, because the share depends on the rank and not on how wide the matrices are",
        "Over 100%, because a real model has many more layers, and every layer needs its own pair of matrices on top of the base weights",
        "Exactly 4%, because a rank of 4 means four percent of the weights are trainable",
      ],
      answer: 0,
      explanation:
        "The pair holds r × (inputs + outputs) numbers and the matrix holds inputs × outputs, so the ratio shrinks as the sides grow. With 30 rows and columns it is 27%; with thousands it is 0.131% for every matrix targeted at rank 4, about 200 times smaller. Extra layers add pairs, but they add base matrices in the same proportion. The count is closed-form arithmetic, so it says nothing about how well such an adapter would fit.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      // Version 1 stored a heatmap toggle, `view`, which no longer exists. Drop it.
      const { view: _legacyView, ...rest } = parsed;
      void _legacyView;
      return {
        ...initialState,
        ...rest,
        probe: typeof rest.probe === "string" && PROBES.includes(rest.probe) ? rest.probe : " ",
        // The calculator's settings are clamped here, because a share link can carry any number.
        ...calculatorState(readCalculator(rest)),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
