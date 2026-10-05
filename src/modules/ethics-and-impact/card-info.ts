import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Deployment": {
    title: "Four places, five claims each, no score",
    summary:
      "Deployment chooses a school tutor, a newsroom draft, clinic intake, or an image generator. Each has five authored claims, sorted into For and Against, and tagged with a theme, who bears it, and whether any benchmark number could speak to it. Score is fixed at none so a harness number cannot close the question.",
    whatYouSee: [
      "Deployment, four segments.",
      "A ledger: For on the left, Against on the right. Each entry shows its number, theme, who bears it, and a tag: on a leaderboard, partly measurable, or not on a leaderboard. Clicking an entry selects that claim.",
      "For claims, Against claims, A benchmark speaks to (how many claims a benchmark covers fully, and how many partly), and Score: none.",
    ],
    howItWorks: [
      "Every claim and tag is authored. Nothing is computed from a model or a benchmark, and the counts are not votes.",
      "\"Partly measurable\" means a benchmark could speak to the claim only if it were split by group, drawn from the people who will use the system, or built to test that case, such as a prompt set that asks for real people's likenesses.",
      "Switching Deployment resets Claim to the first entry.",
    ],
    controls: [
      "Deployment, and the ledger entries themselves.",
      "Comparison worth running: count the \"on a leaderboard\" tags in each deployment. It is at most one of five. Then read the consent, likeness, and provenance claims on Image generator.",
    ],
    notice: [
      "The one claim a leaderboard supports in the school, clinic, and image-generator lists is already answered by an Against line beside it.",
      "On Image generator the costs are consent of the people whose images trained it, the likeness of people depicted, and whether readers can tell a generated picture from a photograph. The only benchmark claim is prompt match.",
      "The costs are borne by people other than the buyer: past students, contractors, writers, readers, patients.",
    ],
    limits: [
      "In this lab: four fictional deployments, twenty authored claims, and authored tags. There is no survey, no stakeholder interview, and no ranking.",
      "In general: a benchmark answers the benchmark. Consent, labor, energy, and accountability stay outside it, however high the score.",
    ],
  },

  "Tension explorer": {
    title: "One claim at a time, left open",
    summary:
      "Claim steps through the selected deployment's five claims in order. Each is shown with its theme, who bears it, and whether a benchmark can measure it. The page never tallies them into a winner.",
    whatYouSee: [
      "Claim, a segmented control labeled by number and side.",
      "The claim text, bordered green for For and red for Against, with the side also written in the heading.",
      "Three tags: Theme, Who bears it, and Can a benchmark measure it, with yes, partly, or no and a one-line reason.",
    ],
    howItWorks: [
      "Claim is an integer index into the deployment's list; the text and tags are string constants.",
      "No weighting, voting, or \"ethics score\" is applied anywhere on the page.",
    ],
    controls: [
      "Claim here; Deployment on the card above changes the list.",
      "Comparison worth running: on School tutor, claim 3 (the harness) against claim 4 (what the harness does not represent).",
    ],
    notice: [
      "Two costs that never appear on a leaderboard: labeling labor and training text taken without consent.",
      "Accountability shows up as a missing name on a byline, not as a field on a model card.",
    ],
    limits: [
      "In this lab: the tensions are teaching sentences written for the lesson, not findings.",
      "In general: unresolved is the honest state. Understanding the mechanism does not choose the deployment; the people affected should have a say.",
    ],
  },

  "Who the benchmark represents": {
    title: "An average over whoever was in the test set",
    summary:
      "A synthetic triage model is evaluated on a holdout where 10% of patients are in group B, whose notes the labels under-represent. Group B share of patients moves the population the model will actually serve. Accuracy is a weighted average of the two groups, so it barely moves, while missed urgent cases pile up in group B.",
    whatYouSee: [
      "Group B share of patients, from 0% to 60%.",
      "Four stacked bars per 1,000 patients: patients in the holdout, patients at your share, missed urgent cases in the holdout, and missed urgent cases at your share. Group A is the left segment and group B the right, each labeled with its count.",
      "Accuracy on the holdout, accuracy at your share, and missed urgent cases per 1,000, holdout → yours.",
      "A second row of four readouts: the true-positive rate for each group (90% and 55%), the Equal-opportunity gap between them (35 points), the false-positive rate for each group (5% and 7%), and the False-positive gap (2 points). These are properties of the model, so they do not move with the share.",
    ],
    howItWorks: [
      "Both groups have 20% urgent cases. Group A: sensitivity 0.90, specificity 0.95, accuracy 94.0%. Group B: sensitivity 0.55, specificity 0.93, accuracy 85.4%.",
      "The true-positive rate is the sensitivity, the share of truly urgent cases flagged. The false-positive rate is 1 − specificity, the share of non-urgent cases flagged anyway. Equal opportunity asks that the true-positive rates match across groups, and equalized odds asks that the false-positive rates match too (Hardt, Price, and Srebro, 2016).",
      "Accuracy at share q = (1 − q)·94.0% + q·85.4%. Missed urgent cases per 1,000 = 1,000·[(1 − q)·0.2·0.10 + q·0.2·0.45].",
      "Holdout (10% B): 93.1% accurate, 27 misses per 1,000. At 50% B: 89.7% and 55, with 82% of the misses in group B.",
    ],
    controls: [
      "Group B share of patients.",
      "Comparison worth running: 10% against 50%. Accuracy drops 3.4 points; missed urgent cases double.",
    ],
    notice: [
      "Group B is 30% of patients at the default and 66% of the missed urgent cases.",
      "A single accuracy number cannot show this. Only a disaggregated evaluation, reported per group, can.",
      "The gap in true-positive rates is 35 points, so the model is far from equal opportunity. Its false-positive rates differ by only 2 points, so it fails equalized odds almost entirely because of the first gap.",
    ],
    limits: [
      "In this lab: the group rates are invented and fixed; nothing is trained, and real groups differ in prevalence as well as error rates. Two criteria are named here; others exist, such as demographic parity.",
      "In general: a model evaluated on last year's population can be deployed on a different one. Distribution shift and missing groups change who bears the errors, not just how many there are.",
    ],
  },

  "Training versus serving": {
    title: "Compute paid once against compute paid every day",
    summary:
      "Training a dense transformer takes about 6·N·D arithmetic operations; serving takes about 2·N per token. The chart compares one training run with the cumulative compute of serving. The two meet after 3·D ÷ tokens served per day, whatever the model size.",
    whatYouSee: [
      "Parameters (1B to 405B), Training tokens (0.5T to 15T), and Tokens served per day (1M to 1T, on a log scale).",
      "Training, once; Serving, per day; and Days to break even.",
      "A chart of one training run's compute (dashed, flat) and serving's cumulative compute (solid, rising), with a guide at the break-even day.",
    ],
    howItWorks: [
      "Training FLOP = 6·N·D. Serving FLOP per day = 2·N·T. Break-even days = 6ND ÷ 2NT = 3D ÷ T, so N cancels.",
      "Energy uses 40% of an A100's 312 TFLOP/s and 400 W per GPU, the utilization that reproduces Meta's reported 184,320 GPU-hours for Llama 2 7B. At the defaults that is about 75 MWh for training.",
      "At the defaults (2T training tokens, 10B tokens a day) serving matches training after 600 days; at 1T tokens a day, after 6 days.",
    ],
    controls: [
      "Parameters, Training tokens, and Tokens served per day.",
      "Comparison worth running: move Parameters from 1B to 405B. Both costs scale, and Days to break even does not move.",
    ],
    notice: [
      "A popular service can spend more compute answering questions than it spent learning to answer them.",
      "Decoding usually runs at lower utilization than training, so the energy break-even arrives sooner than the compute break-even shown here.",
    ],
    limits: [
      "In this lab: FLOP arithmetic for a dense model with one utilization figure. It ignores the attention cost of long contexts, mixture-of-experts sparsity, batching, cooling, and the energy source.",
      "In general: energy use depends on hardware, utilization, data-center efficiency, and grid mix, and published figures often cover only the final training run.",
    ],
  },
};

export default cardInfo;
