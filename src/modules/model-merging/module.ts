import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {
  source: "full",
  epochsA: 20,
  epochsB: 20,
  method: "linear",
  mix: 0.5,
  lambda: 1,
  density: 0.2,
  dropRate: 0.5,
  maskSeed: 4,
};

const METHODS = ["linear", "task", "ties", "dare", "slerp"];

const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-47-model-merging",
  slug: "model-merging",
  title: "Model merging",
  group: "training-adapting",
  order: 43,
  icon: "GitMerge",
  accent: "#3a6a8a",
  prerequisites: ["module-22-lora-adapters"],
  estimatedMinutes: 14,
  steps: ["Build two task vectors", "Average, then split the cost", "Resolve sign conflicts", "Drop and rescale"],
  stepInstructions: [
    "Leave Recipe epochs and Proverb epochs at 20 and read the cosine badge, Overlapping entries and Sign conflicts under the two maps.",
    "With Merge method on Linear blend and Blend t at 50%, compare each merged loss with its specialist, then read Interference on recipes and on proverbs.",
    "Switch Merge method to TIES, move Keep top from 100% down to 10%, and watch the conflict columns in the strip take the elected sign.",
    "Switch Merge method to DARE, raise Drop rate p from 0 to 0.9, and follow the DARE point on the loss plane; press New random mask to see a second draw.",
  ],
  stateVersion: 2,
  tagline:
    "Add task vectors instead of training again: average, sum, trim-and-elect, or drop-and-rescale two real fine-tunes, and measure what each does to both tasks.",
  objectives: [
    "Describe a merge as arithmetic on task vectors that share one base, not as a second training run",
    "Separate the cost of shrinking each task vector from true interference between them",
    "Explain what TIES, DARE and SLERP each change about a plain average",
  ],
  glossary: [
    {
      term: "Model merge",
      definition:
        "Combining the parameters of two or more trained models or adapters into one set of weights with arithmetic alone. No gradient step runs, and no capability is guaranteed.",
    },
    {
      term: "Task vector",
      definition:
        "The fine-tuned weights minus the base weights they started from, one entry per parameter. Merging methods add, scale, trim or drop these vectors and then add the result back onto the base.",
    },
    {
      term: "Shared base",
      definition:
        "The requirement that every model being merged was fine-tuned from the same starting checkpoint. Task vectors from one base live in one coordinate system; two independently trained networks usually do not, because their hidden units can be permuted.",
    },
    {
      term: "Linear mode connectivity",
      definition:
        "The property that every point on the straight line between two solutions also has low loss. Fine-tunes of one pretrained model often have it, which is why averaging them works at all.",
    },
    {
      term: "Weight averaging",
      definition:
        "Taking a weighted mean of checkpoints, one parameter at a time. With a shared base it is exactly base plus the weighted mean of the task vectors, so each task keeps only its share of its own vector.",
    },
    {
      term: "Task arithmetic",
      definition:
        "Adding task vectors to the base with a scaling coefficient, and subtracting them to remove a behaviour (Ilharco et al., 2023). The algebra is exact; whether skills add like numbers is an empirical question.",
    },
    {
      term: "Interference",
      definition:
        "The change in one task's loss caused by adding the other task's vector. Measured by comparing a merge with the same merge where the other vector is set to zero, so shrinkage is not mistaken for it.",
    },
    {
      term: "Sign conflict",
      definition:
        "A parameter that one task vector pushes up and the other pushes down. Averaging a conflict cancels part of both pushes; TIES keeps only the side with more total mass.",
    },
    {
      term: "TIES-Merging",
      definition:
        "Trim, elect sign, and merge (Yadav et al., 2023): keep each vector's largest entries, choose one sign per parameter from the summed trimmed values, then average only the entries that agree with it.",
    },
    {
      term: "DARE",
      definition:
        "Drop And REscale (Yu et al., 2023): zero each task-vector entry with probability p and multiply the survivors by one over one minus p, so every entry keeps its expected value. It relies on the deltas being redundant.",
    },
    {
      term: "SLERP",
      definition:
        "Spherical linear interpolation: blend along the arc between two vectors instead of the straight chord, which keeps the blend's length when the vectors point in different directions. It combines exactly two inputs.",
    },
    {
      term: "Adapter",
      definition:
        "A small trained update, such as a LoRA pair, stored apart from the base. Its product is a task vector too, so the same merge arithmetic applies to it.",
    },
  ],
  references: [
    {
      authors: "Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, et al.",
      title: "Editing Models with Task Arithmetic",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2212.04089",
      note: "Defines a task vector as fine-tuned weights minus base weights, then adds task vectors to cover several tasks at once. Section 3 subtracts a toxic-text task vector to cut toxic generations, the removal trick this lesson mentions.",
    },
    {
      authors: "Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin",
      title: "Linear Mode Connectivity and the Lottery Ticket Hypothesis",
      source: "Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, 3259–3269",
      year: 2020,
      url: "https://proceedings.mlr.press/v119/frankle20a.html",
      note: "Trains two copies of a network from one shared point and checks the loss along the straight line between them. This is the test behind the glossary term linear mode connectivity.",
    },
    {
      authors: "Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang",
      title: "What is being transferred in transfer learning?",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper_files/paper/2020/hash/0607f4c705595b911a4f3e7a127b44e0-Abstract.html",
      note: "Finds that models trained from the same pretrained weights stay in the same low-loss basin and end up close in parameter space. That is why the lesson says fine-tunes of one shared base can be averaged at all.",
    },
    {
      authors: "Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur",
      title: "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2110.06296",
      note: "Argues that two networks trained from different random starts mostly differ by a reordering of their hidden units. It backs the lesson's warning that averaging such networks entry by entry mixes unrelated units.",
    },
    {
      authors: "Samuel K. Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa",
      title: "Git Re-Basin: Merging Models modulo Permutation Symmetries",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2209.04836",
      note: "Reorders the hidden units of one independently trained model to line up with another before merging them. It shows the permutation failure this lab's bigram table cannot stage, and one way around it when there is no shared base.",
    },
    {
      authors: "Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, et al.",
      title: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
      source: "Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162, 23965–23998",
      year: 2022,
      url: "https://proceedings.mlr.press/v162/wortsman22a.html",
      note: "Averages the weights of several fine-tunes of one pretrained model, at no extra cost when the model runs. It is the large-scale case of the Linear blend recipe, and notes that such fine-tunes often sit in one low-error basin.",
    },
    {
      authors: "Prateek Yadav, Derek Tam, Leshem Choshen, et al.",
      title: "TIES-Merging: Resolving Interference When Merging Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html",
      note: "The TIES method: reset small changes, elect one sign per parameter, and merge only the entries that agree with it. It names sign disagreement as a main source of interference, which the TIES step and the Sign conflicts count show.",
    },
    {
      authors: "Le Yu, Bowen Yu, Haiyang Yu, et al.",
      title: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 57755–57775",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/yu24p.html",
      note: "The DARE method: drop task-vector entries with rate p and rescale the rest by 1 / (1 - p). It works because fine-tuning changes in large language models are tiny and redundant, which is why the lesson says this 900-weight toy tilts against DARE.",
    },
    {
      authors: "Ken Shoemake",
      title: "Animating rotation with quaternion curves",
      source: "ACM SIGGRAPH Computer Graphics 19(3), 245–254 (SIGGRAPH 1985)",
      year: 1985,
      url: "https://www.semanticscholar.org/paper/8033e0edb3b43c4ba3605d70d0de14efbe69c976",
      note: "The computer-graphics paper that introduced slerp, which blends between two rotations along the surface of a sphere instead of a straight line. The SLERP recipe in this lab applies the same arc-not-chord idea to two task vectors.",
    },
    {
      authors: "Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, et al.",
      title: "Arcee's MergeKit: A Toolkit for Merging Large Language Models",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track (EMNLP 2024), 477–485",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-industry.36/",
      note: "The paper describing mergekit, the open-source library the lesson names for merging language model checkpoints without extra training. It describes how the library is designed to be efficient, extensible and usable on any hardware.",
    },
    {
      authors: "Arcee AI",
      title: "mergekit (README)",
      source: "GitHub repository documentation",
      year: 2026,
      url: "https://github.com/arcee-ai/mergekit",
      note: "The source of the lesson's mergekit quotes: merges set in a YAML file, the CPU or 8 GB VRAM claim, and mergekit-extract-lora. Its methods table lists Linear, SLERP for exactly two models, Task Arithmetic, TIES, DARE and DELLA.",
    },
    {
      authors: "Hugging Face",
      title: "LoRA (PEFT package reference)",
      source: "Hugging Face PEFT documentation",
      year: 2026,
      url: "https://huggingface.co/docs/peft/package_reference/lora",
      note: "Documents merge_and_unload(), which folds one adapter into its base, and add_weighted_adapter(), which merges several LoRA adapters with options that include ties and dare. It backs the lesson's point that an adapter is a task vector too.",
    },
    {
      authors: "Ying Sheng, Shiyi Cao, Dacheng Li, et al.",
      title: "SLoRA: Scalable Serving of Thousands of LoRA Adapters",
      source: "Proceedings of Machine Learning and Systems 6 (MLSys 2024), 296–311",
      year: 2024,
      url: "https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html",
      note: "The S-LoRA system named in the lesson, which keeps one base model and serves thousands of LoRA adapters side by side. It is the alternative to merging covered in Merge, or serve the adapters side by side.",
    },
    {
      authors: "Enneng Yang, Li Shen, Guibing Guo, et al.",
      title: "Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications, and Opportunities",
      source: "ACM Computing Surveys 58(8), 1–41",
      year: 2026,
      url: "https://arxiv.org/abs/2408.07666",
      note: "A survey that sorts many merging methods into families, covers the theory of why merging works, and lists open problems. Use it to place the five recipes in this lab among newer ones and to read more on where merges break.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "A 50/50 average of two fine-tunes is worse on both tasks than either specialist. What is the cleanest test of whether the two task vectors are interfering?",
      options: [
        "Merge again with the other task vector set to zero, so only what remains beyond that is interference",
        "Treat any rise above the specialist's loss as interference, so the gap itself is the whole measurement of it",
        "Train both fine-tunes for more epochs; if the gap stays after that, the gap was interference",
      ],
      answer: 0,
      explanation:
        "Averaging multiplies each task vector by 0.5, and half a fine-tune is worse than a whole one even with no other task present. Comparing against the merge with the other vector removed isolates what the other task actually did. In this lab the average's proverb vector slightly helps recipes; the full sum at scale 1 is where interference shows.",
      objective: 1,
    },
    {
      prompt:
        "This lab merges two fine-tunes of one base, and the merges mostly work. Which common real-world failure can this toy not show, and why?",
      options: [
        "Interference between tasks, because the lab only ever adds two task vectors",
        "Sign conflicts, because a 900-weight table is too small for two task vectors to disagree on any entry",
        "Averaging nets with different initializations, because a bigram table has no hidden units to reorder",
        "Any loss increase from merging, because each specialist is scored on its own training text",
      ],
      answer: 2,
      explanation:
        "Merging only makes sense when both fine-tunes start from one base, so that entry 417 means the same weight in both. Two networks trained from different random starts can compute the same function with hidden units in a different order, and averaging them mixes unrelated units. A bigram table has no hidden units, so the lab cannot stage that failure, although it does show interference and sign conflicts.",
      objective: 0,
    },
    {
      prompt:
        "On a weight where the two task vectors disagree in sign, what does TIES-Merging do that a plain average does not?",
      options: [
        "It averages the two entries, so they cancel toward zero and neither task is favoured",
        "It elects one sign for that weight and averages only the entries that agree with it",
        "It drops both entries and leaves the base value in place for that weight",
        "It rotates the pair along an arc so that their combined length is preserved",
      ],
      answer: 1,
      explanation:
        "TIES trims each vector to its largest entries, elects a single sign per weight, and averages only entries that agree with the elected sign, so a strong entry is not diluted by a weak opposing one. The plain average lets the two cancel. Rotating along an arc is SLERP, which changes the blend's length rather than resolving signs.",
      objective: 2,
    },
    {
      prompt:
        "Raising DARE's Drop rate p toward 0.9 rescales the surviving entries by 1 / (1 - p). What does that do to both losses in this lab?",
      options: [
        "They fall, because dropping entries removes the interference between the two task vectors for good",
        "They are unchanged, because the rescale restores each dropped entry exactly",
        "Only the recipe loss changes, because DARE drops entries from one vector only",
        "They rise sharply, because rescaling fixes each entry only on average, not individually",
      ],
      answer: 3,
      explanation:
        "The rescale makes the surviving sum right on average, not for any single entry, and the deltas of a 900-weight table carry no spare copies to absorb the loss. DARE was designed for large models whose fine-tuning deltas are small and redundant, so this toy tilts against it, and the result varies with the random mask.",
      objective: 2,
    },
    {
      prompt:
        "On Task arithmetic you sweep Scale lambda from 0 up to 1.5. Why do both losses first fall and then rise?",
      options: [
        "Small lambda undershoots each task vector; large lambda adds both whole, so interference outgrows the gain",
        "Large lambda pushes the merge beyond the base model, so the base's own loss starts to climb again and drags both tasks up",
        "Small lambda trains the merged model for fewer epochs, so it is still underfit at that setting",
        "The loss rising at large lambda is only noise from the DARE mask seed, which the sweep never fixes",
      ],
      answer: 0,
      explanation:
        "Lambda trades two effects. Below about 0.7 to 0.8 the merge is mostly paying the cost of scaling down each task's own vector, which more lambda repairs. Beyond that point both vectors are added nearly whole and the interference from the other task grows faster than the repair. No training happens at any lambda, and the mask seed only matters for DARE.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      // Version 1 held only epochsA, epochsB and mix, and its task vectors were rank-4 LoRA
      // deltas. Keep that meaning when an old snapshot arrives without a source.
      const legacy = !("source" in parsed) && ("epochsA" in parsed || "epochsB" in parsed || "mix" in parsed);
      return {
        ...initialState,
        ...parsed,
        source: legacy ? "lora" : parsed.source === "lora" ? "lora" : "full",
        epochsA: Math.round(clampNumber(parsed.epochsA, 20, 0, 40)),
        epochsB: Math.round(clampNumber(parsed.epochsB, 20, 0, 40)),
        method: METHODS.includes(parsed.method as string) ? parsed.method : "linear",
        mix: clampNumber(parsed.mix, 0.5, 0, 1),
        lambda: clampNumber(parsed.lambda, 1, 0, 1.5),
        density: clampNumber(parsed.density, 0.2, 0.05, 1),
        dropRate: clampNumber(parsed.dropRate, 0.5, 0, 0.9),
        maskSeed: Math.round(clampNumber(parsed.maskSeed, 4, 1, 999)),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
