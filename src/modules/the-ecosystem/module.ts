import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { LICENSES, USES } from "./terms";

const initialState: ModuleState = {
  use: "research",
  license: "research",
  code: "closed",
  data: "undisclosed",
};

const member = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/** Version 1 stored only { use } with research, product, or finetune; those keep their meaning. */
export function migrateEcosystemState(parsed: ModuleState): ModuleState {
  return {
    use: member(parsed.use, USES.map((item) => item.id), "research"),
    license: member(parsed.license, LICENSES.map((item) => item.id), "research"),
    code: member(parsed.code, ["closed", "released"], "closed"),
    data: member(parsed.data, ["undisclosed", "published"], "undisclosed"),
  };
}

const definition: ModuleDefinition = {
  id: "module-48-the-ecosystem",
  slug: "the-ecosystem",
  title: "The ecosystem",
  group: "frontiers",
  order: 32,
  icon: "Globe",
  accent: "#4a6fa8",
  prerequisites: ["module-33-how-models-are-evaluated", "module-10-scaling-laws", "module-12-post-training"],
  estimatedMinutes: 12,
  steps: ["Read the card", "Place it on the spectrum", "Check a use", "Separate permission from provenance"],
  stepInstructions: [
    "Read License, Data information, Evaluation harness, and Training energy on Sample model card before changing anything.",
    "Set License to Apache-2.0, then Training code to released and Data information to published, and watch the highlighted column move in Release spectrum.",
    "Set Intended use to Hosted product, then Train on its outputs, and compare which clause blocks each under Research only and Custom community.",
    "Set License to Apache-2.0 and Intended use to Hosted product. License match says yes; read what Still unknown says about the data.",
  ],
  stateVersion: 2,
  tagline:
    "Take apart a model release — weights, code, data information, license, harness, energy — so a download stops looking like a complete product, and open weights stop looking like open source.",
  objectives: [
    "Distinguish a base model, a fine-tune, an API, open weights, and open-source AI",
    "Check a proposed use against a license's clauses and separate permission from data provenance",
    "Read a model card's numbers critically, including evaluation error and estimated training energy",
  ],
  glossary: [
    {
      term: "Base model",
      definition:
        "A model trained for a general objective such as next-token prediction, before assistant or domain adaptation. It is a starting point, not a finished product.",
    },
    {
      term: "Fine-tune",
      definition:
        "A further training run on a narrower dataset. The name does not tell you whether the data were public, licensed, or synthetic, and the base model's license usually still applies.",
    },
    {
      term: "Open weights",
      definition:
        "A file of parameters you can download and run. It does not by itself include training data, training code, or permission for every use.",
    },
    {
      term: "Open source AI",
      definition:
        "Under the Open Source Initiative's definition (OSAID 1.0, October 2024): the weights, the complete training and inference code, and data information detailed enough to build a substantially equivalent system, all under terms that allow any purpose.",
    },
    {
      term: "API",
      definition:
        "A hosted interface that runs a model for you. You get behavior, not weights, and the terms of service decide what you may do with outputs, such as whether they may train a competing model.",
    },
    {
      term: "License",
      definition:
        "The legal terms on weights, code, data, or outputs. A research-only clause can sit on otherwise downloadable files, and a license cannot grant rights its authors never had.",
    },
    {
      term: "Permissive license",
      definition:
        "A license such as Apache-2.0 or MIT that allows any use, including commercial, with light conditions such as keeping notices. On weights alone it still does not make a release open-source AI.",
    },
    {
      term: "Acceptable-use policy",
      definition:
        "A list of banned uses attached to a license or an API. It makes the grant conditional, which is one reason custom model licenses do not meet the Open Source Definition.",
    },
    {
      term: "Model card",
      definition:
        "A structured document describing a model's intended uses, training data, evaluation, and limitations, proposed by Mitchell et al. (2019). It is only as good as what its authors chose to disclose.",
    },
    {
      term: "Data provenance",
      definition:
        "Where training data came from and under what terms or consent. Permission to use the weights does not repair missing provenance.",
    },
    {
      term: "Evaluation harness",
      definition:
        "A scripted set of tasks and scoring rules used to compare models. A leaderboard position is a score on that harness, not a complete quality claim.",
    },
    {
      term: "Standard error",
      definition:
        "The spread a measured score has from sampling alone: the square root of p(1 − p)/n for accuracy p on n items. On 40 items at 70% it is about 7 points.",
    },
  ],
  references: [
    {
      authors: "Irene Solaiman",
      title: "The Gradient of Generative AI Release: Methods and Considerations",
      source: "Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2023), 111–122",
      year: 2023,
      url: "https://arxiv.org/abs/2302.04844",
      note: "Lays out six levels of access to an AI system, from fully closed through API access and downloadable access to fully open. It backs the lesson's point that release is a range, not a yes-or-no, which the Release spectrum card squeezes into four columns.",
    },
    {
      authors: "Open Source Initiative",
      title: "The Open Source AI Definition – 1.0",
      source: "Open Source Initiative",
      year: 2024,
      url: "https://opensource.org/ai/open-source-ai-definition",
      note: "The definition this lesson calls OSAID 1.0. It lists the three parts an open-source AI system must share, data information, the complete training and running code, and the parameters, under terms that allow use for any purpose, and it asks for data information rather than always the data itself.",
    },
    {
      authors: "OpenAI",
      title: "GPT-4 Technical Report",
      source: "arXiv preprint arXiv:2303.08774",
      year: 2023,
      url: "https://arxiv.org/abs/2303.08774",
      note: "The report for the lesson's API-only example. It says plainly that it gives no further details about model size, hardware, training compute, dataset construction, or training method, which shows how little a closed release lets you check.",
    },
    {
      authors: "Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al.",
      title: "Mistral 7B",
      source: "arXiv preprint arXiv:2310.06825",
      year: 2023,
      url: "https://arxiv.org/abs/2310.06825",
      note: "The paper for the lesson's example of weights released under Apache-2.0. It releases the base model and an instruction-tuned version under Apache 2.0 with inference code, but says nothing about the data the base model was trained on, so it sits in the permissive open-weights column, short of open-source AI.",
    },
    {
      authors: "Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, et al.",
      title: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:2397–2430",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/biderman23a.html",
      note: "One of the lesson's examples of a release that ships more than weights. EleutherAI trained 16 models on public data in a fixed order and released the models, checkpoints, training code, and training data, the kind of kit the last column of Release spectrum describes.",
    },
    {
      authors: "Dirk Groeneveld, Iz Beltagy, Evan Walsh, et al.",
      title: "OLMo: Accelerating the Science of Language Models",
      source: "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 15789–15809",
      year: 2024,
      url: "https://aclanthology.org/2024.acl-long.841/",
      note: "AI2's paper for the lesson's other fully released example. It contrasts OLMo with releases that share only weights and inference code, and it releases the training data along with training and evaluation code.",
    },
    {
      authors: "The Apache Software Foundation",
      title: "Apache License, Version 2.0",
      source: "Apache Software Foundation",
      year: 2004,
      url: "https://www.apache.org/licenses/LICENSE-2.0",
      note: "The full text of the permissive license the lab offers as Apache-2.0. Its grant covers use, changes, and passing copies on, and its redistribution section asks you to include a copy of the license and keep the notices, the light conditions behind the Redistribute the weights row.",
    },
    {
      authors: "Meta",
      title: "Llama 2 Community License Agreement",
      source: "Meta Llama license page",
      year: 2023,
      url: "https://www.llama.com/llama2/license/",
      note: "A real custom model license of the kind the lab's Custom community license copies. It requires a separate license from Meta above 700 million monthly active users, ties use to an Acceptable Use Policy, and bars using the model's outputs to improve other large language models.",
    },
    {
      authors: "Stefano Maffulli",
      title: "Meta's LLaMa license is not Open Source",
      source: "Open Source Initiative blog",
      year: 2023,
      url: "https://opensource.org/blog/metas-llama-2-license-is-not-open-source",
      note: "The OSI statement behind the lesson's claim that Llama 2 is not open source. It explains that the commercial limit and the banned fields of use break the Open Source Definition, which does not allow limits on fields of use.",
    },
    {
      authors: "Shayne Longpre, Robert Mahari, Anthony Chen, et al.",
      title: "A large-scale audit of dataset licensing and attribution in AI",
      source: "Nature Machine Intelligence 6(8), 975–987",
      year: 2024,
      url: "https://www.nature.com/articles/s42256-024-00878-8",
      note: "The Data Provenance Initiative traced the sources, creators, and licenses of more than 1,800 text datasets and found licenses often missing or wrong on popular hosting sites. It backs the lesson's point that permission and data provenance are separate questions, the gap the Still unknown readout shows.",
    },
    {
      authors: "Margaret Mitchell, Simone Wu, Andrew Zaldivar, et al.",
      title: "Model Cards for Model Reporting",
      source: "Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019), 220–229",
      year: 2019,
      url: "https://arxiv.org/abs/1810.03993",
      note: "The paper that proposed model cards, the idea behind the lab's Sample model card. It describes short documents that state a model's intended uses, how it was evaluated, and how it performs across different groups of people.",
    },
    {
      authors: "Rishi Bommasani, Kevin Klyman, Sayash Kapoor, et al.",
      title: "The 2024 Foundation Model Transparency Index",
      source: "Transactions on Machine Learning Research (TMLR), 2025",
      year: 2025,
      url: "https://arxiv.org/abs/2407.12929",
      note: "Scores major model developers on 100 things they could disclose about data, compute, the model, and its use. It finds lasting gaps on copyright status, data access, and data labor, which backs the lesson's warning that a card is only as good as what its authors chose to disclose.",
    },
    {
      authors: "Evan Miller",
      title: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations",
      source: "arXiv preprint arXiv:2411.00640",
      year: 2024,
      url: "https://arxiv.org/abs/2411.00640",
      note: "Treats an evaluation as an experiment on a sample of questions and gives the standard error formula for right-or-wrong scores, the square root of p(1 - p)/n that the card uses. It also covers how to compare two models, which is why a 3-point gap on a 40-item harness means little.",
    },
    {
      authors: "Jared Kaplan, Sam McCandlish, Tom Henighan, et al.",
      title: "Scaling Laws for Neural Language Models",
      source: "arXiv preprint arXiv:2001.08361",
      year: 2020,
      url: "https://arxiv.org/abs/2001.08361",
      note: "The source of the rule of thumb behind the Training energy row: training costs about 6 floating point operations per parameter for each training token. The card multiplies this by 7B parameters and 2T tokens.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Meta's paper for Llama 2, trained on 2 trillion tokens. Its table of pretraining emissions lists 184,320 GPU-hours for the 7B model on A100-80GB chips rated at 400 W, the reported figure the lesson compares with the card's estimate of about 187,000.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "A model ships its weights under Apache-2.0, with no training code and no information about its training data. Which description is accurate?",
      options: [
        "Open source AI, because Apache-2.0 is an open-source license and the weights are public",
        "Closed, because a release that gives no data information cannot be reused in any way",
        "API-only, because training code that stays private means the model can only be reached remotely",
        "Open weights under a permissive license, but short of open-source AI as OSAID 1.0 defines it",
      ],
      answer: 3,
      explanation:
        "Apache-2.0 lets you use, modify, and redistribute the weights for any purpose, so they are permissively licensed open weights. OSAID 1.0 also asks for the complete training code and data information detailed enough to build a substantially equivalent system. The weights are downloadable, so it is not API-only, and a permissive license is exactly what lets you reuse them.",
      objective: 0,
    },
    {
      prompt:
        "You set License to Apache-2.0 and Intended use to Hosted product, and License match says yes. What has that settled?",
      options: [
        "That the product is safe to launch, because the license answers every legal question about the model",
        "That the training data was gathered with permission, since a license could not be granted otherwise",
        "That commercial use is allowed, while where the training data came from is still unknown",
        "That the release is open-source AI, because the license now allows any purpose",
      ],
      answer: 2,
      explanation:
        "License match looks up the clauses the use needs, here commercial use, and Apache-2.0 grants it. It cannot answer where the data came from or whether the people in it consented, because a license cannot grant rights its authors never had; Still unknown depends only on Data information. A permissive license on the weights also does not by itself make a release open-source AI.",
      objective: 1,
    },
    {
      prompt:
        "Model A scores 73% and model B scores 70% on the card's 40-item harness. What does the card's own arithmetic say about that 3-point gap?",
      options: [
        "It settles the ranking, because both models were scored on exactly the same 40 test items",
        "It is inside the harness's standard error of about 7 points, so this harness cannot rank them reliably",
        "It would become reliable once either score passes 80%, because noise vanishes near the ceiling",
        "It is a real difference, because a computed score has no sampling error to worry about",
      ],
      answer: 1,
      explanation:
        "A score measured on n items has a standard error of the square root of p(1 − p)/n, about 7 points at 70% on 40 items. Resampling the items would move either score by about that much, so a 3-point gap is inside the noise even though both models saw the same questions. The error shrinks only slowly near the ceiling (about 6 points at 80%) and mainly with more items.",
      objective: 2,
    },
    {
      prompt:
        "A real model card lists a strong harness score but says nothing about its training data. What is the sound way to read the omission?",
      options: [
        "As neutral, because a card only has to report scores and the data is the developer's own business",
        "As a gap in what you know: provenance, consent, and leakage cannot be judged from this card",
        "As reassurance, because developers are happy to disclose their data whenever it is clean",
        "As harmless, because OSAID 1.0 never asks for any information about how the data was built",
      ],
      answer: 1,
      explanation:
        "A card is only as good as what its authors chose to disclose, so a missing field is information, not a blank to fill with hope. Without data information you cannot check provenance or consent, and you cannot rule out that benchmark items leaked into training and inflated the score. OSAID 1.0 does ask for data information, though not necessarily the data itself.",
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
      return migrateEcosystemState(parsed);
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
