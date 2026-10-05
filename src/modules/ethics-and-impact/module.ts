import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEPLOYMENTS, PARAMETER_OPTIONS } from "./impact";

const initialState: ModuleState = {
  deployment: "school",
  claim: 0,
  shareB: 0.3,
  paramIndex: 2,
  trainTokens: 2,
  servedLog: 10,
};

const num = (value: unknown, fallback: number, low: number, high: number, integer = false) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const bounded = Math.min(high, Math.max(low, value));
  return integer ? Math.round(bounded) : bounded;
};

/** Version 1 stored { deployment, claim }; both keep their meaning. */
export function migrateEthicsState(parsed: ModuleState): ModuleState {
  const deployment =
    typeof parsed.deployment === "string" && DEPLOYMENTS.some((item) => item.value === parsed.deployment)
      ? parsed.deployment
      : "school";
  return {
    deployment,
    claim: num(parsed.claim, 0, 0, 4, true),
    shareB: num(parsed.shareB, initialState.shareB as number, 0, 0.6),
    paramIndex: num(parsed.paramIndex, initialState.paramIndex as number, 0, PARAMETER_OPTIONS.length - 1, true),
    trainTokens: num(parsed.trainTokens, initialState.trainTokens as number, 0.5, 15),
    servedLog: num(parsed.servedLog, initialState.servedLog as number, 6, 12),
  };
}

const definition: ModuleDefinition = {
  id: "module-49-ethics-and-impact",
  slug: "ethics-and-impact",
  title: "Ethics and impact",
  group: "frontiers",
  order: 33,
  icon: "Landmark",
  accent: "#6a5a3a",
  prerequisites: ["module-48-the-ecosystem"],
  estimatedMinutes: 14,
  steps: ["Pick a deployment", "Walk the claims", "Split the benchmark", "Weigh serving against training", "Notice the missing score"],
  stepInstructions: [
    "Set Deployment to School tutor and read that Score stays none, then read how many claims A benchmark speaks to.",
    "Move Claim through every For and Against line and read Who bears it and Can a benchmark measure it for each.",
    "In Who the benchmark represents, move Group B share of patients from 10% to 50% and compare the two accuracies with Missed urgent per 1,000, then read the Equal-opportunity gap, which stays at 35 points at every share.",
    "In Training versus serving, raise Tokens served per day and watch Days to break even fall, then change Parameters and see that it does not move.",
    "Switch Deployment to Newsroom draft, then Clinic intake, then Image generator, and name two costs that no leaderboard measures.",
  ],
  stateVersion: 2,
  tagline:
    "Connect training data, consent, likeness, labor, energy, dual use, and who a benchmark represents, so a higher score does not settle whether a deployment is justified.",
  objectives: [
    "Name at least two costs that never appear on a leaderboard",
    "Explain why a benchmark score cannot finish an ethics question, using who the benchmark represents",
    "Compare the one-time compute of training with the ongoing compute of serving",
    "Name the fairness criterion the triage gap breaks, and say how a risk-based rule treats a deployment without deciding whether it is justified",
  ],
  glossary: [
    {
      term: "Training data",
      definition:
        "The examples a model was fitted to. Much of it was collected without asking the people who wrote or appear in it, which is a consent problem rather than a math problem.",
    },
    {
      term: "Consent",
      definition:
        "Permission to collect, train on, or generate from someone's work or likeness. A public URL is not automatically consent, and a license on the weights cannot supply it.",
    },
    {
      term: "Dual use",
      definition:
        "A capability that can help and harm. The same extractor that fills in an intake form can compile a dossier; the mechanism does not choose.",
    },
    {
      term: "Externalities",
      definition:
        "Costs paid by people who did not choose the deployment, such as energy, scraped work, or a displaced task. They do not show up in a loss curve or a price.",
    },
    {
      term: "Labor",
      definition:
        "Human work in labeling, red-teaming, and content moderation, often low-paid and invisible behind the product. Alignment data is labor.",
    },
    {
      term: "Accountability",
      definition:
        "Being able to say who decided to deploy, on whom, and with what recourse when it goes wrong. A model card is not accountability by itself.",
    },
    {
      term: "Disaggregated evaluation",
      definition:
        "Reporting a metric separately for each group of people or inputs. An overall score can hide a group for whom the model fails far more often.",
    },
    {
      term: "Equal opportunity and equalized odds",
      definition:
        "Two group-fairness criteria defined on a classifier's error rates (Hardt, Price, and Srebro, 2016). Equal opportunity asks for equal true-positive rates across groups: the same share of truly positive cases flagged. Equalized odds, the stronger one, also asks for equal false-positive rates. Meeting either says nothing about whether the labels were right or the deployment is justified.",
    },
    {
      term: "Distribution shift",
      definition:
        "A difference between the data a model was evaluated on and the data it meets in use. A holdout drawn from last year's population cannot promise performance on next year's.",
    },
    {
      term: "Provenance",
      definition:
        "A record of where content came from and how it was made or changed, such as a machine-readable mark saying an image was generated. A mark helps only while it survives editing and someone checks it, so it supports judgment and does not replace it.",
    },
    {
      term: "Training compute",
      definition:
        "The arithmetic to train a model, about 6 × parameters × training tokens for a dense transformer. It is paid once per run, plus experiments and failed runs.",
    },
    {
      term: "Inference compute",
      definition:
        "The arithmetic to serve a model, about 2 × parameters for each token processed or generated. It is paid on every request, so at scale it can exceed training.",
    },
    {
      term: "Risk-based regulation",
      definition:
        "A rule that scales its duties with the harm a use could cause instead of treating every system alike. The EU AI Act sorts systems into unacceptable, high, transparency, and minimal risk levels, mostly by what they are used for. It sets duties and processes; it does not decide whether a given deployment is justified.",
    },
  ],
  references: [
    {
      authors: "Solon Barocas, Moritz Hardt, and Arvind Narayanan",
      title: "Fairness and Machine Learning: Limitations and Opportunities",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://fairmlbook.org/",
      note: "The Classification chapter defines demographic parity, equal opportunity, and equalized odds and shows why such criteria usually cannot all hold at once. The Datasets chapter looks at where training data comes from and the harms tied to it, which backs the Training data and Consent claims.",
    },
    {
      authors: "Laura Weidinger, John Mellor, Maribeth Rauh, et al.",
      title: "Ethical and social risks of harm from Language Models",
      source: "arXiv preprint arXiv:2112.04359",
      year: 2021,
      url: "https://arxiv.org/abs/2112.04359",
      note: "A DeepMind review that sorts 21 risks of language models into six areas, including misinformation, malicious uses, and environmental and job harms. It is a map of the kinds of claims the Tension explorer lists that no leaderboard scores, such as dual use.",
    },
    {
      authors: "Joy Buolamwini and Timnit Gebru",
      title: "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification",
      source: "Proceedings of the 1st Conference on Fairness, Accountability and Transparency, PMLR 81, 77–91",
      year: 2018,
      url: "https://proceedings.mlr.press/v81/buolamwini18a.html",
      note: "Finds that two face benchmarks were mostly lighter-skinned people and that commercial systems made far more errors on darker-skinned women. It is a real case of what Who the benchmark represents shows: a good overall score can hide a group the test left out.",
    },
    {
      authors: "Margaret Mitchell, Simone Wu, Andrew Zaldivar, et al.",
      title: "Model Cards for Model Reporting",
      source: "Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019), 220–229",
      year: 2019,
      url: "https://arxiv.org/abs/1810.03993",
      note: "Proposes short documents that report a model's results separately for groups such as age, race, or skin type, and say what it is meant to be used for. That per-group reporting is disaggregated evaluation, and the lesson adds that a model card alone is not accountability.",
    },
    {
      authors: "Moritz Hardt, Eric Price, and Nathan Srebro",
      title: "Equality of Opportunity in Supervised Learning",
      source: "Advances in Neural Information Processing Systems 29 (NIPS 2016)",
      year: 2016,
      url: "https://papers.nips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html",
      note: "The paper the lesson names for equal opportunity and equalized odds. It defines both criteria from a classifier's error rates in each group, which is how the lab computes the Equal-opportunity gap and the False-positive gap.",
    },
    {
      authors: "Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan",
      title: "Inherent Trade-Offs in the Fair Determination of Risk Scores",
      source: "8th Innovations in Theoretical Computer Science Conference (ITCS 2017), LIPIcs 67, 43:1–43:23",
      year: 2017,
      url: "https://doi.org/10.4230/LIPIcs.ITCS.2017.43",
      note: "Proves that three common fairness conditions cannot all be met at once except in narrow special cases. It backs the lesson's point that different fairness rules can disagree about the same model, so passing one does not mean passing the others.",
    },
    {
      authors: "Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan",
      title: "Dissecting racial bias in an algorithm used to manage the health of populations",
      source: "Science 366(6464), 447–453",
      year: 2019,
      url: "https://europepmc.org/article/MED/31649194",
      note: "A widely used health algorithm predicted spending instead of illness, so Black patients with the same score were sicker than White patients. It is a real case of the lesson's warning that a model can match its labels well and still be wrong if the labels are wrong.",
    },
    {
      authors: "Jared Kaplan, Sam McCandlish, Tom Henighan, et al.",
      title: "Scaling Laws for Neural Language Models",
      source: "arXiv preprint arXiv:2001.08361",
      year: 2020,
      url: "https://arxiv.org/abs/2001.08361",
      note: "Section 2.1 counts the arithmetic in a transformer: about 2 operations per parameter for each token in a forward pass, and about 6 per parameter per token in training. These are the 2·N and 6·N·D rules the Training versus serving card uses.",
    },
    {
      authors: "Emma Strubell, Ananya Ganesh, and Andrew McCallum",
      title: "Energy and Policy Considerations for Deep Learning in NLP",
      source: "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), 3645–3650",
      year: 2019,
      url: "https://aclanthology.org/P19-1355/",
      note: "Estimates the energy, carbon, and cloud cost of training several language models, and adds a case study of the full cost of developing one model, including all tuning and experiments. It backs the lesson's point that figures for the final run leave out the work before it.",
    },
    {
      authors: "Alexandra Sasha Luccioni, Yacine Jernite, and Emma Strubell",
      title: "Power Hungry Processing: Watts Driving the Cost of AI Deployment?",
      source: "Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2024), 85–99",
      year: 2024,
      url: "https://arxiv.org/abs/2311.16863",
      note: "Measures the energy and carbon of running models, not training them, across many tasks, and finds general-purpose generative models cost far more per answer than task-specific ones. It backs the lesson's point that serving is the recurring bill.",
    },
    {
      authors: "European Commission",
      title: "AI Act",
      source: "Shaping Europe's digital future, European Commission",
      year: 2026,
      url: "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
      note: "The Commission's plain summary of the Act's four risk levels, from banned uses to minimal risk, with examples of high-risk uses. It also lists when each set of rules applies, the schedule the lesson says to check before relying on a date.",
    },
    {
      authors: "European Parliament and Council of the European Union",
      title: "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)",
      source: "Official Journal of the European Union, L series",
      year: 2024,
      url: "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
      note: "The law as first published. Article 50 asks makers of tools that generate audio, images, video, or text to mark outputs as machine-readable and detectable as generated, and Annex III lists high-risk uses, including emergency healthcare patient triage and AI that evaluates learning outcomes. Later amendments are published separately.",
    },
    {
      authors: "National Institute of Standards and Technology",
      title: "Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
      source: "NIST AI 100-1",
      year: 2023,
      url: "https://www.nist.gov/itl/ai-risk-management-framework",
      note: "NIST's page for its voluntary framework, released in January 2023 and built around four functions: Govern, Map, Measure, and Manage. It backs the lesson's description of the NIST framework as a process, not a verdict on any one deployment.",
    },
    {
      authors: "National Institute of Standards and Technology",
      title: "Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile",
      source: "NIST AI 600-1",
      year: 2024,
      url: "https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence",
      note: "The July 2024 companion guide for generative AI. It lists risks such as confabulation, environmental impacts, harmful bias, information integrity, and intellectual property, and it discusses content provenance, which ties to the Image generator claims.",
    },
    {
      authors: "ISO/IEC JTC 1/SC 42",
      title: "ISO/IEC 42001:2023 Information technology - Artificial intelligence - Management system",
      source: "International Organization for Standardization",
      year: 2023,
      url: "https://www.iso.org/standard/81230.html",
      note: "The official page for the 2023 standard that sets requirements for how an organization sets up, runs, and keeps improving its own AI management. It backs the lesson's point that the standard is about an organization's processes, not any one model's behavior.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "In each of the four deployments the ledger tags at most one of the five claims as on a leaderboard. What follows?",
      options: [
        "A higher score cannot settle consent, labor, energy, or accountability, so those claims stay open",
        "The other claims count for nothing until someone builds a benchmark that can measure them",
        "The claims can be weighed against one another and reduced to a single ranking of the deployments",
        "Benchmarks add nothing to a deployment decision, so running them before launch is a wasted effort overall",
      ],
      answer: 0,
      explanation:
        "A benchmark answers the benchmark. Training text taken without consent, contractors' labeling work, serving energy, and who is accountable for a wrong output are real costs borne by people other than the buyer, and no harness number moves them. That does not make benchmarks useless: the one measurable claim is still worth knowing, and the lab never tallies the rest into a score.",
      objective: 0,
    },
    {
      prompt:
        "A triage model scores 93% on a holdout where 10% of patients are in group B. The clinic that will use it is half group B. What does the lab's arithmetic show?",
      options: [
        "Nothing changes, because the model and its accuracy are the same wherever it is deployed",
        "Accuracy dips a few points, but missed urgent cases roughly double, mostly in group B",
        "Accuracy rises, because the extra group B patients give the model more to learn from",
        "Accuracy collapses toward chance, because a clinic that is half group B is a different task",
      ],
      answer: 1,
      explanation:
        "Accuracy is a weighted average of the two groups, so moving from 10% to 50% group B takes it only from 93.1% to 89.7%. But the model misses 45% of group B's urgent cases against 10% of group A's, so missed urgent cases per 1,000 go from 27 to 55, and 82% of them are in group B. The benchmark answered the benchmark; it did not answer this clinic.",
      objective: 1,
    },
    {
      prompt:
        "In Training versus serving you move Parameters from 1B to 405B and leave the token counts alone. What happens to Days to break even?",
      options: [
        "It falls, because a bigger model answers more tokens per second and repays its training sooner",
        "It rises, because a larger model costs far more to train than it costs to serve",
        "It stays the same, because training and serving compute both scale with parameters",
        "It drops to zero for the largest models, because serving then outweighs training at once",
      ],
      answer: 2,
      explanation:
        "Training costs about 6·N·D operations and serving about 2·N per token, so their ratio is 3·D divided by the tokens served per day, and N cancels. Both bars grow with parameters, but they grow together, so the day on which serving overtakes training depends on training tokens and daily volume, not on model size.",
      objective: 2,
    },
    {
      prompt:
        "Which of these is a limit of the lab's training-versus-serving arithmetic, and not a feature of it?",
      options: [
        "It ignores how many training tokens were used, which would change the break-even day",
        "It ignores model size, which would shift the break-even day as parameters grow",
        "It measures energy at a real data center, so the figure cannot be repeated on other kinds of hardware",
        "It assumes a dense model at one utilization and leaves out cooling, the grid, and failed runs",
      ],
      answer: 3,
      explanation:
        "The break-even day uses training tokens as an input and model size cancels exactly, so neither is a gap. The estimate is computed, not measured: FLOP counts for a dense model at one utilization figure, converted to energy at an assumed power draw. Real energy use also depends on cooling, the grid, and the experiments and failed runs that published figures often leave out.",
      objective: 2,
    },
    {
      prompt:
        "In Who the benchmark represents, the triage model flags 90% of urgent cases in group A and 55% in group B, and 5% and 7% of non-urgent cases. Which fairness statement is right?",
      options: [
        "It breaks equal opportunity, which asks for equal true-positive rates, and so fails equalized odds too",
        "It meets equal opportunity, because the two groups' overall accuracies are within nine points of each other here",
        "It breaks equal opportunity only because of false alarms, since that criterion is about the non-urgent cases",
        "It meets equalized odds, because the false-positive rates are close and that is all the criterion requires of it",
      ],
      answer: 0,
      explanation:
        "Equal opportunity (Hardt, Price, and Srebro, 2016) asks that among truly urgent cases each group is flagged at the same rate: 90% against 55%, a gap of 35 points. Equalized odds asks for equal true-positive and equal false-positive rates, so the same gap breaks it; the false-positive rates, 5% and 7%, are only 2 points apart. Accuracy blends both error types, which is why it moves so little while the gap stays put.",
      objective: 3,
    },
    {
      prompt:
        "The lesson's governance section describes the EU AI Act's risk-based approach. What does it settle about the Clinic intake deployment?",
      options: [
        "It sets duties by use, and triage is on its high-risk list, but whether this deployment is justified stays open",
        "It bans AI in clinics, because health is the most sensitive area and so sits at the top of the Act's risk levels",
        "It exempts a system with high enough holdout accuracy, because a high score is treated as proof that it is safe",
        "It treats a model the same wherever it runs, because its duties attach to the architecture and not to the use",
      ],
      answer: 0,
      explanation:
        "The Act sorts systems into risk levels mostly by what they are used for. Annex III lists emergency healthcare patient triage among high-risk uses, which brings extra duties, but whether a given system falls under it depends on its intended use. Nothing in it answers the lab's question: who bears the costs and who was left out of the benchmark. The schedule has been amended since the Act was adopted, so check the current text.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      return migrateEthicsState(parsed);
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
