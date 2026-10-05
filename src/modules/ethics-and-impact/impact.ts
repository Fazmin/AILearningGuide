/**
 * Authored deployment claims, plus two computed instruments:
 * a disaggregated evaluation of a synthetic triage model, and the FLOP arithmetic
 * that compares one training run with ongoing serving.
 */

export type Side = "For" | "Against";
export type Measured = "yes" | "partly" | "no";

export interface Claim {
  side: Side;
  text: string;
  theme: string;
  bearer: string;
  /** Could a benchmark or leaderboard number speak to this claim? "partly" means only one built for it. */
  measured: Measured;
}

export interface Deployment {
  value: "school" | "news" | "clinic" | "image";
  label: string;
  claims: readonly Claim[];
}

export const DEPLOYMENTS: readonly Deployment[] = [
  {
    value: "school",
    label: "School tutor",
    claims: [
      { side: "For", text: "A tutor that never tires can sit with a student after the building closes.", theme: "access", bearer: "students", measured: "no" },
      { side: "Against", text: "The essays used to train it were collected without those students' consent.", theme: "consent", bearer: "past students", measured: "no" },
      { side: "For", text: "A published harness shows higher homework completion.", theme: "measured benefit", bearer: "district", measured: "yes" },
      { side: "Against", text: "Completion is not learning, and the harness does not represent the district that would pay.", theme: "representation", bearer: "students", measured: "partly" },
      { side: "Against", text: "Serving energy is billed to the school; labeling work was billed to contractors.", theme: "energy and labor", bearer: "school, contractors", measured: "no" },
    ],
  },
  {
    value: "news",
    label: "Newsroom draft",
    claims: [
      { side: "For", text: "A first draft can free a reporter for the phone call the model cannot make.", theme: "time", bearer: "reporters", measured: "no" },
      { side: "Against", text: "Fluent errors can reach readers in the desk's own voice.", theme: "misinformation", bearer: "readers", measured: "partly" },
      { side: "For", text: "The publisher already pays for an API, so the marginal article looks cheap.", theme: "cost", bearer: "publisher", measured: "no" },
      { side: "Against", text: "Cheap for the publisher is an externality for the people whose writing was scraped.", theme: "consent", bearer: "writers", measured: "no" },
      { side: "Against", text: "Accountability is unclear when a wrong date ships under a byline.", theme: "accountability", bearer: "readers, reporter", measured: "no" },
    ],
  },
  {
    value: "clinic",
    label: "Clinic intake",
    claims: [
      { side: "For", text: "Triage questions can be asked in more languages than the night desk speaks.", theme: "access", bearer: "patients", measured: "no" },
      { side: "Against", text: "A label set that under-represents some patients becomes an urgency score that misses them.", theme: "representation", bearer: "patients", measured: "partly" },
      { side: "For", text: "A documented harness beat a majority-class baseline on holdout notes.", theme: "measured benefit", bearer: "clinic", measured: "yes" },
      { side: "Against", text: "The holdout is last year's clinic, not the patients who walk in tomorrow.", theme: "distribution shift", bearer: "patients", measured: "partly" },
      { side: "Against", text: "Dual use: the same extractor can compile a dossier on a patient.", theme: "dual use", bearer: "patients", measured: "no" },
    ],
  },
  {
    value: "image",
    label: "Image generator",
    claims: [
      { side: "For", text: "A tool that renders a draft image in seconds lets a small studio test ideas it could not afford to commission.", theme: "access", bearer: "small studios, clients", measured: "no" },
      { side: "Against", text: "The training images were collected without the consent of the artists and photographers who made them.", theme: "consent", bearer: "illustrators, photographers", measured: "no" },
      { side: "For", text: "A published benchmark rates its images above the previous model's for matching the prompt.", theme: "measured benefit", bearer: "vendor", measured: "yes" },
      { side: "Against", text: "A realistic image of a real person can be made without that person's say, and the benchmark prompts never ask for one.", theme: "likeness", bearer: "people depicted", measured: "partly" },
      { side: "Against", text: "Without a mark of origin that survives editing, readers cannot tell a generated picture from a photograph.", theme: "provenance", bearer: "readers, newsrooms", measured: "partly" },
    ],
  },
];

/* ------------------------------------------------ disaggregated evaluation */

export interface GroupRates {
  name: string;
  /** Share of patients whose case is urgent. */
  prevalence: number;
  /** P(flagged urgent | urgent). */
  sensitivity: number;
  /** P(not flagged | not urgent). */
  specificity: number;
}

/** Synthetic rates for a triage classifier. Group B stands for notes the labels under-represent. */
export const GROUP_A: GroupRates = { name: "Group A", prevalence: 0.2, sensitivity: 0.9, specificity: 0.95 };
export const GROUP_B: GroupRates = { name: "Group B", prevalence: 0.2, sensitivity: 0.55, specificity: 0.93 };
export const HOLDOUT_SHARE_B = 0.1;

/** True-positive rate: the share of truly urgent cases the model flags. */
export const truePositiveRate = (group: GroupRates) => group.sensitivity;

/** False-positive rate: the share of non-urgent cases the model flags anyway. */
export const falsePositiveRate = (group: GroupRates) => 1 - group.specificity;

/**
 * Equal opportunity asks for equal true-positive rates across groups (Hardt, Price, and Srebro, 2016); this is how far
 * the model is from it. Equalized odds adds equal false-positive rates, and `falsePositiveGap` is that second distance.
 */
export const equalOpportunityGap = () => Math.abs(truePositiveRate(GROUP_A) - truePositiveRate(GROUP_B));
export const falsePositiveGap = () => Math.abs(falsePositiveRate(GROUP_A) - falsePositiveRate(GROUP_B));

export const accuracyOf = (group: GroupRates) =>
  group.prevalence * group.sensitivity + (1 - group.prevalence) * group.specificity;

/** Missed urgent cases per 1,000 patients of this group. */
export const missesPerThousand = (group: GroupRates) => 1000 * group.prevalence * (1 - group.sensitivity);

export function mixOutcome(shareB: number, patients = 1000) {
  const a = (1 - shareB) * patients;
  const b = shareB * patients;
  const missesA = (a / 1000) * missesPerThousand(GROUP_A);
  const missesB = (b / 1000) * missesPerThousand(GROUP_B);
  return {
    patientsA: a,
    patientsB: b,
    accuracy: (1 - shareB) * accuracyOf(GROUP_A) + shareB * accuracyOf(GROUP_B),
    missesA,
    missesB,
    misses: missesA + missesB,
    shareOfMissesB: missesA + missesB > 0 ? missesB / (missesA + missesB) : 0,
  };
}

/* ------------------------------------------------ training versus serving */

export const PARAMETER_OPTIONS = [1e9, 3e9, 7e9, 13e9, 34e9, 70e9, 180e9, 405e9] as const;
/** A100 dense BF16 peak and the utilization that reproduces Llama 2 7B's reported GPU-hours. */
export const PEAK_FLOPS = 312e12;
export const UTILIZATION = 0.4;
export const GPU_WATTS = 400;

export const trainingFlops = (parameters: number, tokens: number) => 6 * parameters * tokens;
export const servingFlopsPerDay = (parameters: number, tokensPerDay: number) => 2 * parameters * tokensPerDay;

/** Days of serving whose FLOPs equal one training run: 6ND / (2N·T) = 3D / T, independent of N. */
export const breakEvenDays = (trainingTokens: number, tokensPerDay: number) => (3 * trainingTokens) / tokensPerDay;

export const megawattHours = (flops: number, utilization = UTILIZATION) =>
  (flops / (PEAK_FLOPS * utilization) / 3600) * GPU_WATTS / 1e6;
