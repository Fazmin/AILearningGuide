import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const REGIMES = ["replace", "accumulate"] as const;
const MEASURES = ["support", "coverage", "entropy", "realWords", "heldOut"] as const;
const GENERATORS = ["self", "teacher"] as const;
const FILTERS = ["none", "words"] as const;
const SAMPLE_SIZES = [250, 500, 2000] as const;

const initialState: ModuleState = {
  generation: 5,
  regime: "replace",
  measure: "support",
  temperature: 1,
  filter: "none",
  sampleSize: 500,
  generator: "self",
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const oneOf = <T extends string | number>(value: unknown, allowed: readonly T[], fallback: T): T =>
  (allowed as readonly unknown[]).includes(value) ? (value as T) : fallback;

/**
 * Version 1 stored { epochs, strict, mix } for a single student run. Its "mixed"
 * option (seed plus synthetic) is the accumulate regime here; the other two
 * options trained on one source, which is the replace regime. Version 2 had no
 * generator choice: every run was the model writing its own successor's data, so
 * an older payload takes "self" and keeps its meaning.
 */
export function hydrateSyntheticState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const legacy = !("regime" in parsed) && typeof parsed.mix === "string";
    const regime = legacy ? (parsed.mix === "mixed" ? "accumulate" : "replace") : parsed.regime;
    return {
      generation: Math.round(clampNumber(parsed.generation, 5, 0, 10)),
      regime: oneOf(regime, REGIMES, "replace"),
      measure: oneOf(parsed.measure, MEASURES, "support"),
      temperature: Math.round(clampNumber(parsed.temperature, 1, 0.5, 1.2) * 10) / 10,
      filter: oneOf(parsed.filter, FILTERS, "none"),
      sampleSize: oneOf(parsed.sampleSize, SAMPLE_SIZES, 500),
      generator: oneOf(parsed.generator, GENERATORS, "self"),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-44-synthetic-data",
  slug: "synthetic-data",
  title: "Synthetic data",
  group: "training-adapting",
  order: 38,
  icon: "Factory",
  accent: "#7a4a9a",
  prerequisites: ["module-18-dataset-building"],
  estimatedMinutes: 14,
  steps: [
    "Read one generation",
    "Recurse on synthetic only",
    "Keep the real data",
    "Filter and sharpen",
    "Bring in a stronger teacher",
  ],
  stepInstructions: [
    "With Generation at 0, read what the model fitted to the human seed writes, then set Generation to 1 and compare the two samples and the Real words here value.",
    "Keep Training data on Synthetic only and drag Generation from 0 to 10. Watch the Transitions line fall and the crossed cells spread in Which transitions survive.",
    "Switch Training data to Real + synthetic and drag Generation again. The Transitions line stays at 157; switch Measure to Entropy to see what still drifts.",
    "Set Filter to Real words only and Measure to Real words, then lower Temperature to 0.6. Compare Real words with Transitions for both Training data settings.",
    "Set Filter back to No filter and Measure to Unseen text. Switch Generator between The model itself and A stronger teacher under both Training data settings, then set Sample size to 2000 and compare the lines at generation 10.",
  ],
  stateVersion: 3,
  tagline:
    "Fit a model, let it write the next model's training text, and repeat: measure which patterns survive, what a filter adds, and what keeping the real data prevents.",
  objectives: [
    "Describe synthetic data as model-generated text that becomes another model's training set",
    "Explain model collapse as the loss of rare patterns when models train on finite samples of their own output",
    "Judge a filter or verifier by what it keeps and what it can never restore",
    "Explain what a fixed, stronger teacher changes: it breaks the loop and can add patterns the seed lacked, but only what it knows",
  ],
  glossary: [
    {
      term: "Synthetic data",
      definition:
        "Training examples written by a model or simulator rather than collected from the world. It can only contain what the generator can produce, so it inherits the generator's gaps along with its strengths.",
    },
    {
      term: "Seed data",
      definition:
        "The human-written text the first model is fitted to. Every later generation descends from it, so anything missing from the seed is missing from all of them.",
    },
    {
      term: "Recursive training",
      definition:
        "Fitting each new model to text the previous model wrote, generation after generation. The lab's Generation slider counts those rounds.",
    },
    {
      term: "Model collapse",
      definition:
        "The drift of recursively trained models away from the original distribution: rare patterns vanish first, then variety shrinks. It is a property of the data loop, not of any single model's size.",
    },
    {
      term: "Tail",
      definition:
        "The rare events of a distribution, here the character transitions the seed uses only once or twice. They matter out of proportion to their frequency because that is where unusual but real cases live.",
    },
    {
      term: "Finite-sample error",
      definition:
        "The gap between a distribution and any finite sample of it. A transition with probability 0.2 percent is absent from most 500-character samples, and a model fitted to a sample without it assigns it zero.",
    },
    {
      term: "Temperature",
      definition:
        "The exponent 1/T applied to every probability before sampling. Below 1 it sharpens toward the likeliest continuation; above 1 it flattens. It changes which text gets written, not what the model knows.",
    },
    {
      term: "Distribution narrowing",
      definition:
        "Probability mass concentrating on the most typical outputs over generations, measured here as falling next-character entropy. Sharpened sampling makes it faster.",
    },
    {
      term: "Verifier",
      definition:
        "A check that accepts or rejects generated examples, such as running code, checking a math answer, or here, requiring every token to be a real seed word. It raises the quality of what is kept but can only remove text.",
    },
    {
      term: "Rejection sampling",
      definition:
        "Drafting many candidates and keeping only those a verifier accepts. The accepted share is the cost: at generation 1 the real-words verifier here keeps about one character in twenty.",
    },
    {
      term: "Data accumulation",
      definition:
        "Adding each round's synthetic text to the original data instead of replacing it. The human seed keeps every original pattern represented, which is what prevents the collapse the replace loop shows.",
    },
    {
      term: "Contamination",
      definition:
        "Evaluation items or private seed strings reappearing in generated training text. A generator that saw a benchmark can write it back out, so synthetic data needs the same decontamination checks as scraped data.",
    },
    {
      term: "Teacher model",
      definition:
        "A stronger model whose output trains a smaller student. Because the teacher is fixed, the student's mistakes cannot feed back into it, and the student can gain patterns it never saw, up to what the teacher knows.",
    },
    {
      term: "Held-out coverage",
      definition:
        "The share of the character pairs in some unseen text that a model gives non-zero probability. It asks whether the model could produce those pairs at all, which is why a count-fitted model scores zero on a pair it never saw.",
    },
  ],
  references: [
    {
      authors: "Lin Long, Rui Wang, Ruixuan Xiao, et al.",
      title: "On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey",
      source: "Findings of the Association for Computational Linguistics: ACL 2024, 11065–11082",
      year: 2024,
      url: "https://aclanthology.org/2024.findings-acl.658/",
      note: "A survey of how language models are used to write training data, organized as generation, curation, and evaluation. Its sections on sample filtering and on faithfulness versus diversity give the wider picture behind this lesson's Synthetic data and Verifier terms.",
    },
    {
      authors: "Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, et al.",
      title: "AI models collapse when trained on recursively generated data",
      source: "Nature 631(8022), 755–759",
      year: 2024,
      url: "https://www.nature.com/articles/s41586-024-07566-y",
      note: "The paper the lesson cites for model collapse. It finds that training on model-generated content makes the tails of the original distribution disappear, the same loss of rare pairs that Which transitions survive shows under Synthetic only.",
    },
    {
      authors: "Elvis Dohmatob, Yunzhen Feng, Pu Yang, et al.",
      title: "A Tale of Tails: Model Collapse as a Change of Scaling Laws",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235:11165–11197",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/dohmatob24b.html",
      note: "Explains collapse as the cutting of a distribution's tail. Its first figure names finite samples, temperature, and top-p sampling as the causes, which matches the lesson's points that the tail goes first and that lower Temperature speeds the loss.",
    },
    {
      authors: "Ari Holtzman, Jan Buys, Li Du, et al.",
      title: "The Curious Case of Neural Text Degeneration",
      source: "International Conference on Learning Representations (ICLR 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/1904.09751",
      note: "Introduces top-p (nucleus) sampling and explains temperature sampling. It notes that a lower temperature makes text better at the cost of variety, the trade behind the lab's Temperature control and the Distribution narrowing term.",
    },
    {
      authors: "Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, et al.",
      title: "Self-Consuming Generative Models Go MAD",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2307.01850",
      note: "Studies image models trained in loops on their own output, with and without real data in each round. It finds that without enough real data, quality or variety falls, and that picking only the best-looking samples trades variety for quality, as the Real words filter does here.",
    },
    {
      authors: "Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, et al.",
      title: "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data",
      source: "arXiv preprint arXiv:2404.01413",
      year: 2024,
      url: "https://arxiv.org/abs/2404.01413",
      note: "The paper the lesson cites for data accumulation. Replacing the real data with each round's synthetic data tends toward collapse, while adding each round to the original data avoids it, the gap between the Synthetic only and Real + synthetic lines.",
    },
    {
      authors: "Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman",
      title: "STaR: Bootstrapping Reasoning With Reasoning",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html",
      note: "The STaR method named in Going deeper. A model writes step-by-step reasoning, only the reasoning that reaches a correct answer is kept, and the model is fine-tuned on it, which is rejection sampling with an answer check as the verifier.",
    },
    {
      authors: "Avi Singh, John D. Co-Reyes, Rishabh Agarwal, et al.",
      title: "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models",
      source: "Transactions on Machine Learning Research (TMLR)",
      year: 2024,
      url: "https://arxiv.org/abs/2312.06585",
      note: "A model drafts many solutions, keeps math answers that are correct and code that passes its test cases, and trains on what it kept. It shows the lesson's claim that synthetic data works best when a strong generator is paired with a real check.",
    },
    {
      authors: "Yoon Kim and Alexander M. Rush",
      title: "Sequence-Level Knowledge Distillation",
      source: "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), 1317–1327",
      year: 2016,
      url: "https://aclanthology.org/D16-1139/",
      note: "A small student is trained on text that a larger teacher generates, rather than on the teacher's probabilities. That is the setup the Generator control stages with A stronger teacher, and the contrast the lesson draws with the Distillation & pruning lab.",
    },
    {
      authors: "Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, et al.",
      title: "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
      source: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), 13484–13508",
      year: 2023,
      url: "https://aclanthology.org/2023.acl-long.754/",
      note: "Grows a large instruction dataset from a small set of human-written seed tasks, filtering out invalid or near-duplicate items before training. It is the method Alpaca's data followed, and a real-world case of the Seed data and filter steps in this lab.",
    },
    {
      authors: "Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, et al.",
      title: "Alpaca: A Strong, Replicable Instruction-Following Model",
      source: "Stanford CRFM blog post",
      year: 2023,
      url: "https://crfm.stanford.edu/2023/03/13/alpaca.html",
      note: "The Alpaca release the lesson describes: LLaMA 7B fine-tuned on 52K instruction-following examples written by text-davinci-003 for less than $500. It also gives the licence and terms-of-use reasons for the research-only release that the lesson's governance point cites.",
    },
    {
      authors: "Arnav Gudibande, Eric Wallace, Charlie Snell, et al.",
      title: "The False Promise of Imitating Proprietary Language Models",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2305.15717",
      note: "Fine-tunes smaller models on ChatGPT's outputs and finds they copy its style but gain little on tasks the imitation data does not cover. It backs the lesson's point that a student learns only what the teacher's data contains.",
    },
    {
      authors: "Nicholas Carlini, Florian Tramer, Eric Wallace, et al.",
      title: "Extracting Training Data from Large Language Models",
      source: "30th USENIX Security Symposium (USENIX Security 21), 2633–2650",
      year: 2021,
      url: "https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting",
      note: "Shows that a language model can write out word-for-word text from its training data, including personal details seen in only one document. It backs the Contamination term's warning that private strings can reappear in generated text.",
    },
    {
      authors: "Shahriar Golchin and Mihai Surdeanu",
      title: "Time Travel in LLMs: Tracing Data Contamination in Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2308.08493",
      note: "Detects benchmark contamination by asking a model to finish the start of a test item and checking whether it writes the rest exactly. It shows the lesson's point that a generator that saw a benchmark can write its items back out.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Each new model is trained only on text the previous model wrote, and every batch passes a strict quality check. After several rounds the outputs look clean but repetitive. What best explains it?",
      options: [
        "The quality check is too lenient, and a stricter check would bring the lost variety back",
        "Each round is a finite sample, so missed patterns are gone for good and a filter can only remove text",
        "Collapse only happens at low sampling temperature, so at temperature 1 every pattern would survive the loop",
      ],
      answer: 1,
      explanation:
        "A model fitted to a sample assigns zero to anything the sample missed, and it cannot write what it assigns zero, so losses compound. In the lab, temperature 1 still drops the mean from 157 transitions to 51 by generation 10, and the real-words verifier drops it faster, to 16. Keeping the human seed in every round is what holds all 157.",
      objective: 1,
    },
    {
      prompt: "With Training data on Synthetic only, what is the generation-3 model fitted to?",
      options: [
        "The human seed text, reweighted by the probabilities that generation 2 assigned to each pair in it",
        "The seed plus every batch of text written by generations 1 and 2, all counted together",
        "The generation-0 model's output, kept fixed as the training text for every later round",
        "A fresh batch sampled from the generation-2 model, after the optional verifier has filtered it",
      ],
      answer: 3,
      explanation:
        "Synthetic data is text a model wrote that becomes another model's training set. Under Synthetic only, each round replaces the data with the latest batch drafted by the previous table, after the optional verifier. Real + synthetic is the regime that adds every batch to the human seed instead, which is data accumulation.",
      objective: 0,
    },
    {
      prompt:
        "You switch Training data from Synthetic only to Real + synthetic and drag Generation to 10. What does the Transitions line do, and why?",
      options: [
        "It falls more slowly, because the synthetic share of the data grows with every round",
        "It rises above the seed's count, since the extra generated text adds new character pairs to the table",
        "It stays flat at the seed's count, because the seed is in every round, so its pairs are never missed",
        "It falls exactly as before, because the seed adds only text the model could already write",
      ],
      answer: 2,
      explanation:
        "A table fitted by counting gives a pair zero only if nothing in its training text contains it. With the human seed in every round, each of its pairs keeps a non-zero count, so none can be lost, and generated text can only repeat pairs the seed already had. Entropy still drifts, so accumulation slows narrowing without making the table a perfect copy.",
      objective: 1,
    },
    {
      prompt:
        "Under Synthetic only, a Real words verifier lifts Real words close to 100% yet leaves fewer Transitions than no filter at all. What does that show?",
      options: [
        "Quality and variety are separate: a verifier only removes text, so it can raise one and lower the other",
        "The verifier is broken, since higher quality must also mean more variety",
        "Real words measures transitions indirectly, so the two lines should have risen together",
        "The verifier just needs a higher sampling temperature, which would let it restore the pairs that went missing",
      ],
      answer: 0,
      explanation:
        "A verifier moves probability toward outputs that pass; it cannot restore a pair the generator never writes. A spelling check also rewards repetition, since the the the is all real words, so it keeps the loop clean while the set of distinct pairs shrinks. Variety has to come from the seed, the prompts, or fresh human data.",
      objective: 2,
    },
    {
      prompt:
        "You set Generator to A stronger teacher, Training data to Real + synthetic, and Measure to Unseen text, then drag Generation from 0 to 10. What happens to the line?",
      options: [
        "It falls, because any model trained on generated text loses patterns, whoever happened to write it",
        "It rises, because the teacher writes pairs the seed lacked and the seed stays in every round",
        "It stays flat, because a student can never produce a pair that its human seed text lacked",
        "It rises, but only if Filter is Real words only, since only verified text can add anything",
      ],
      answer: 1,
      explanation:
        "The collapse in the self loop comes from each generation sampling its own table, so a missed pair is gone for good. A fixed teacher is not refitted, so it keeps writing pairs the seed never had, and the student counts them. Keeping the seed in every round also stops any loss, so coverage of unseen harbor text climbs toward the teacher's. A seed-bound verifier would remove exactly those new pairs.",
      objective: 3,
    },
    {
      prompt:
        "With A stronger teacher and Filter set to Real words only, Unseen text hardly improves, though the teacher knows far more than the seed. Why?",
      options: [
        "The verifier keeps only words from the seed, so it discards the new text the teacher adds",
        "A teacher cannot improve a student once the data has been filtered by any check at all",
        "The filter retrains the teacher on the seed each round, so its extra knowledge is overwritten",
        "The student already covers all of the unseen text, so there is nothing left for it to gain",
      ],
      answer: 0,
      explanation:
        "A verifier accepts or rejects drafts against its own standard, here the seed's 57 words. Text full of words the seed never used fails that test, so the verifier throws away the very patterns that make the teacher stronger. A filter only removes text, and a filter written from the seed can only keep what the seed already contains.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateSyntheticState,
};

export default definition;
