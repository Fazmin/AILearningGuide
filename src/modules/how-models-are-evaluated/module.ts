import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { GOALS, POPULATIONS } from "./metrics";

const BASELINES = ["learned", "always-negative", "always-positive"] as const;
const SPLITS = ["all", "train", "val", "test"] as const;
const TUNE = ["train", "val", "test"] as const;
const CURVES = ["underfit", "good", "overfit"] as const;

/**
 * Version 3 adds `population` (the 1-in-100 imbalance demo or the 1-in-5 population) and a third goal, F1.
 * A version 1 or 2 payload has no population, so it keeps meaning the rare, one-positive set it was saved on.
 */
const initialState: ModuleState = {
  threshold: 0.5,
  baseline: "always-negative",
  split: "all",
  tuneSplit: "val",
  goal: "accuracy",
  curve: "overfit",
  fold: 4,
  outlier: 12,
  population: "rare",
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

const definition: ModuleDefinition = {
  id: "module-33-how-models-are-evaluated",
  slug: "how-models-are-evaluated",
  title: "How we know a model works",
  group: "groundwork",
  order: 6,
  icon: "Target",
  accent: "#3f8a3d",
  prerequisites: ["module-50-supervised-learning-loop"],
  estimatedMinutes: 16,
  steps: [
    "Keep the never-fire baseline",
    "Move the threshold",
    "Switch to 1 in 5",
    "Change the split",
    "Choose t on a slice",
    "Read a learning curve",
    "Rotate a fold",
  ],
  stepInstructions: [
    "Leave Population on 1 in 100, Baseline on always negative and Scored split on all 100, then read Accuracy against Recall.",
    "Switch Baseline to score ≥ threshold and set Decision threshold to 0.42, then 0.60, then 0.90, and watch TP and FP on the score strip and the matrix.",
    "Set Population to 1 in 5, then slide Decision threshold from 0.30 to 0.70 and compare how TP, FP and FN change in the matrix and how Precision and Recall move in Four readings.",
    "Set Scored split to train, then to test, on each population. On 1 in 100, train and val hold no positive and recall reads n/a; on 1 in 5 every split holds positives.",
    "With Population on 1 in 5, set Maximize to F1, press Set t from val, then set Choose t using to test and press Set t from test. Compare Test and Fresh rows for the two paths, and Holdout?.",
    "Set Pattern to overfit, then to good fit, and compare where the lowest val loss sits and whether val turns up after it.",
    "Move Held-out fold from 0 to 4 on each population and notice which folds can measure recall: only fold 4 on 1 in 100, every fold on 1 in 5.",
  ],
  stateVersion: 3,
  tagline: "Read a confusion matrix on a deliberately imbalanced set and on a population with a realistic share of positives, then watch the precision-recall trade-off, leakage, learning curves, and MSE versus MAE so a tidy number cannot hide the rare class.",
  objectives: [
    "Read accuracy, precision, and recall from the same confusion matrix",
    "Explain why accuracy can look fine while the rare class is always missed, and why the same detector's precision falls as positives get rarer",
    "Say why choosing a threshold on the test slice is leakage that inflates the held-out score, and how train versus val curves diagnose overfitting",
    "Trace the precision-recall trade-off as the decision threshold moves on a population with a realistic share of positives",
  ],
  glossary: [
    {
      term: "Accuracy",
      definition: "The share of examples the model labeled correctly. On an imbalanced set it can stay high while the rare class is never found.",
    },
    {
      term: "Precision",
      definition: "Of the examples the model called positive, the share that truly are: TP over TP plus FP. High precision means few false alarms; it says nothing about the positives you missed, and it has no value (0/0) when nothing was called positive.",
    },
    {
      term: "Recall",
      definition: "Of the examples that truly are positive, the share the model found: TP over TP plus FN. High recall means few misses; it says nothing about how many false alarms you raised, and on a slice with no positives it has no value (0/0), which is not the same as zero.",
    },
    {
      term: "F1 score",
      definition: "The harmonic mean of precision and recall, 2PR/(P+R), which equals 2TP/(2TP+FP+FN). It stays low unless both are decent, so it follows the precision-recall trade-off as the threshold moves, and it ignores true negatives, so a never-firing rule scores F1 = 0 despite 99 percent accuracy on the 1-in-100 population.",
    },
    {
      term: "Confusion matrix",
      definition: "A table of true class versus predicted class. Every scalar score on this lab is a way of reading that one table, and the table changes with the decision threshold t: raising t removes false alarms first and then starts removing real positives.",
    },
    {
      term: "ROC curve and AUC",
      definition: "The ROC curve plots the true positive rate against the false positive rate as the threshold t falls. AUC is the area under it, which equals the chance that a random positive outscores a random negative. It grades the ranking across every t, so it can sit near 1 while the rule you publish at one t never fires.",
    },
    {
      term: "Class imbalance and base rate",
      definition: "Class imbalance is a set where one class is much rarer than another, and the base rate is the share of positives. Majority guessing then looks accurate and still fails the rare class, and the same detector's precision falls as the base rate falls, because its false alarms come from the much larger negative class.",
    },
    {
      term: "Held-out, validation, and test sets",
      definition: "Held-out data is kept out of fitting so the score is not just memory. The validation set is the slice you may look at while choosing settings such as the threshold t or the stop epoch. The test set is scored once, after those choices are final. Choosing t on the test set turns it into a second validation set, so the number you report stops being a test.",
    },
    {
      term: "Epoch",
      definition: "One full pass of training over every training example. The learning-curve card plots loss against epochs 0 to 20 from authored formulas, not from a trained run.",
    },
    {
      term: "Leakage",
      definition: "Using the holdout to make a choice (a threshold, a feature, a stop epoch) and then reporting that same holdout as if it were unseen. It inflates the reported score: the choice was fitted to those rows, so the number overstates what new rows will show.",
    },
    {
      term: "Overfitting",
      definition: "A fit that keeps improving on the training slice while the validation slice gets worse. Diagnosed by a diverging pair of curves, not by a low training number alone.",
    },
    {
      term: "Cross-validation",
      definition: "A rotation that lets each slice be the holdout once, refitting on the rest, then averages the scores. It still reports whatever metric you chose; accuracy on an imbalanced set remains a trap, and a rare class can miss whole folds unless the folds are stratified.",
    },
    {
      term: "Stratified split",
      definition: "A split that keeps each class's share the same in every slice, so a rare class shows up in train, val, and test alike. It needs at least one positive per slice: with a single positive in 100 rows no split can do it, while the 1-in-5 population here holds 20 percent positives in train, in val, and in test.",
    },
    {
      term: "MSE, MAE, and RMSE",
      definition: "Mean squared error (MSE) is the mean of squared residuals, so large misses dominate. Mean absolute error (MAE) is the mean of absolute residuals and grows linearly with a miss. RMSE is the square root of MSE, which returns to the units of y while still weighting large misses more.",
    },
  ],
  references: [
    {
      authors: "Moritz Hardt and Benjamin Recht",
      title: "Patterns, Predictions, and Actions: Foundations of Machine Learning",
      source: "Princeton University Press, free to read online",
      year: 2022,
      url: "https://mlstory.org/",
      note: "Chapter 2 builds the confusion table, precision, recall, F1, and the ROC curve, and gives the same base-rate formula for precision that the Four readings card uses. Chapter 8 explains why a test set that is reused again and again stops giving an honest score.",
    },
    {
      authors: "Sebastian Raschka",
      title: "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning",
      source: "arXiv preprint arXiv:1811.12808",
      year: 2018,
      url: "https://arxiv.org/abs/1811.12808",
      note: "A practical review of holdout sets and cross-validation. It explains stratified splits and the three-way train, validation, and test split that the Train, val, test card draws.",
    },
    {
      authors: "scikit-learn developers",
      title: "Metrics and scoring: quantifying the quality of predictions",
      source: "scikit-learn 1.9 user guide, section 3.4",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/model_evaluation.html",
      note: "The official guide for a widely used machine learning library. It defines accuracy, the confusion matrix, precision, recall, F1, ROC and AUC, MAE, and MSE, and it suggests checking a model against simple rules such as always guessing the most common class, like this lab's never-fire baseline.",
    },
    {
      authors: "scikit-learn developers",
      title: "Tuning the decision threshold for class prediction",
      source: "scikit-learn 1.9 user guide, section 3.3",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/classification_threshold.html",
      note: "Explains why the default cut-off of 0.5 is often the wrong one and how to pick the threshold that maximizes a chosen score, such as F1. It says to tune the threshold on data the model was not trained on, such as a separate validation set, which is what Set t from val does.",
    },
    {
      authors: "Takaya Saito and Marc Rehmsmeier",
      title: "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",
      source: "PLOS ONE 10(3), e0118432",
      year: 2015,
      url: "https://doi.org/10.1371/journal.pone.0118432",
      note: "Shows that when positives are rare, an ROC plot can make a classifier look better than it is, while a precision-recall plot gives a truer picture because it counts how many alarms are real. It backs up the lesson's warning that a high AUC is not enough on its own.",
    },
    {
      authors: "Sayash Kapoor and Arvind Narayanan",
      title: "Leakage and the reproducibility crisis in machine-learning-based science",
      source: "Patterns 4(9), 100804",
      year: 2023,
      url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/",
      note: "Finds leakage in 294 published papers across 17 fields and sorts it into eight types. One type, choosing features with help from the test set, is the same mistake as choosing t on test in this lab.",
    },
    {
      authors: "Cynthia Dwork, Vitaly Feldman, Moritz Hardt, et al.",
      title: "Generalization in Adaptive Data Analysis and Holdout Reuse",
      source: "Advances in Neural Information Processing Systems 28 (NIPS 2015)",
      year: 2015,
      url: "https://papers.nips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html",
      note: "Shows that checking one idea after another against the same holdout can overfit to the holdout itself. That is why a val slice used for many decisions wears out, and why the lesson keeps a test slice that no choice touches.",
    },
    {
      authors: "Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, et al.",
      title: "Do ImageNet Classifiers Generalize to ImageNet?",
      source: "Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, 5389–5400",
      year: 2019,
      url: "https://proceedings.mlr.press/v97/recht19a.html",
      note: "The authors built new test sets for two famous image benchmarks by following the original recipes, and every model they tried scored lower on them. Like the lab's Fresh rows, it checks a published score against new data; the authors trace the drop to slightly harder images rather than to test-set reuse.",
    },
    {
      authors: "Chuan Guo, Geoff Pleiss, Yu Sun, et al.",
      title: "On Calibration of Modern Neural Networks",
      source: "Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70, 1321–1330",
      year: 2017,
      url: "https://proceedings.mlr.press/v70/guo17a.html",
      note: "Calibration asks whether rows scored 0.8 turn out positive about 80 percent of the time, the check this lab leaves out. The paper finds that modern neural networks are often poorly calibrated, draws reliability diagrams, and shows a simple fix called temperature scaling.",
    },
    {
      authors: "Timothy O. Hodson",
      title: "Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not",
      source: "Geoscientific Model Development 15(14), 5481–5487",
      year: 2022,
      url: "https://doi.org/10.5194/gmd-15-5481-2022",
      note: "Explains that neither RMSE nor MAE is better in every case: RMSE suits errors that follow a normal bell curve, and MAE suits errors where extreme misses are more common. It backs up the Regression residuals card, where one outlier swamps MSE but not MAE.",
    },
  ],
  checkpoint: [
    {
      prompt: "A detector's confusion matrix on 100 rows reads TP 1, FP 1, FN 0, TN 98. Which readings follow?",
      options: [
        "Precision 1.00 and recall 0.50, because one of the two alarms was a real positive",
        "Precision 0.50 and recall 1.00: half the alarms are false, no positive is missed",
        "Precision 0.99 and recall 0.99, because accuracy sums up all the correct cells",
        "Precision 0.50 and recall 0.50, because both read the same two cells of the table",
      ],
      answer: 1,
      explanation: "Precision is TP/(TP+FP) = 1/2 and recall is TP/(TP+FN) = 1/1, so the one real positive was found but half the alarms were false. Accuracy is (TP+TN)/N = 0.99 and hides both. Precision ignores misses, recall ignores false alarms, and they read different cells of the same table.",
      objective: 0,
    },
    {
      prompt: "A detector never fires on a set that is 99 percent negative. What is true?",
      options: [
        "Accuracy is about 99 percent while recall of the rare class is exactly 0",
        "Precision and recall are both about 99 percent, since accuracy averages them",
        "Accuracy is 0 percent, because the rare class is the one that matters",
      ],
      answer: 0,
      explanation: "Majority guessing scores high accuracy on an imbalanced set, because the 99 negatives are all counted correct. Recall of the rare class is zero because no true positive was found, and precision has no value at all since nothing was called positive. Accuracy is a legal reading of the matrix that can still be the wrong number to publish.",
      objective: 1,
    },
    {
      prompt: "The same detector, with the same true positive rate and false positive rate, moves from a population where 1 row in 5 is positive to one where 1 row in 100 is. What happens to its precision?",
      options: [
        "It stays put, because precision describes the detector and not the rows it is given to score",
        "It rises, because a rarer class is easier to pick out from the crowd of negatives",
        "It falls sharply, because that false alarm rate now flags many negatives per positive",
        "It falls only a little, because recall is unchanged and precision follows recall",
      ],
      answer: 2,
      explanation: "Precision is TP/(TP+FP). Holding both rates fixed, TP scales with the number of positives and FP with the number of negatives, so a rarer positive class shrinks TP while the 99 negatives still raise their share of false alarms. At t = 0.50 the 1-in-5 population here reads precision 69.6 percent; the same rates at 1 in 100 give 8.5 percent.",
      objective: 1,
    },
    {
      prompt: "On 1 in 5, with Baseline on score ≥ threshold, you raise Decision threshold from 0.40 to 0.60. What happens to precision and recall?",
      options: [
        "Both rise, because a stricter threshold makes the detector more careful about everything",
        "Precision rises and recall falls, because weak alarms stop firing before real positives do",
        "Both fall, because firing on fewer rows can only lose information about the positives",
        "Precision falls and recall rises, because fewer firing rows leaves a larger share of misses",
      ],
      answer: 1,
      explanation: "The two score distributions overlap, so a higher t drops false alarms first and then starts dropping real positives. From 0.40 to 0.60 the false alarms fall from 22 to 1 while the true positives fall from 19 to 11, so precision climbs from 46.3 to 91.7 percent and recall drops from 95 to 55 percent. Moving t trades one reading for the other and cannot raise both.",
      objective: 3,
    },
    {
      prompt: "You press Set t from test with Maximize on recall, then report the test recall. Why is that number no longer a test?",
      options: [
        "Recall can only be computed on a validation slice, and never on a test slice at all",
        "Choosing a threshold changes the model's weights, which contaminates the test rows",
        "The threshold was chosen on the rows it is scored on, so test acted as validation",
      ],
      answer: 2,
      explanation: "The validation slice is where settings such as the threshold t may be chosen, and the test slice is scored once after every choice is final. Picking t on test lets those rows influence the rule, so the reported recall is optimistic and the holdout badge turns leaked. Setting a threshold does not change the weights.",
      objective: 2,
    },
    {
      prompt: "On 1 in 5 you maximize F1, then switch Choose t using from val to test, and Test F1 goes up. What does the Fresh rows reading for that same leaked t show?",
      options: [
        "It equals the leaked Test F1, because one t gives one F1 on every set of rows",
        "It is lower than the leaked Test F1, because that t was fitted to the fifteen test rows",
        "It is higher than the leaked Test F1, because fresh rows are a larger sample than test",
        "It stays blank, because fresh rows cannot be scored until the model is retrained on them",
      ],
      answer: 1,
      explanation: "Picking the best of 101 thresholds on fifteen rows keeps whatever is lucky about those rows. Scored on 400 rows that took no part in the choice, the leaked t reads 68.8 percent F1 against the 85.7 percent it claimed on test, while the t chosen on val claimed 66.7 and read 77.9. Leakage inflates the claim; it does not improve the detector.",
      objective: 2,
    },
    {
      prompt: "On Learning curves, train loss keeps falling while val loss bottoms out and then rises. What does that show?",
      options: [
        "The model is overfitting, and the best stop is where val loss is lowest",
        "The val slice is mislabeled, because val loss should always track train loss",
        "The model is underfitting, because val loss sits above train loss",
        "Training has broken, because a loss should never rise once it has fallen",
      ],
      answer: 0,
      explanation: "Overfitting is diagnosed by diverging curves, not by a low training number: the fit keeps improving on rows it has seen while getting worse on rows it has not. Early stopping keeps the weights from the lowest val point. These curves are authored formulas, not a trained run, so they show the pattern, not a real model.",
      objective: 2,
    },
    {
      prompt: "A detector's AUC is close to 1, yet at the threshold you published it never fires. What explains both facts?",
      options: [
        "AUC is measured on the training slice, so it cannot describe the rule you published",
        "AUC averages accuracy across every threshold, and accuracy on this kind of set is high",
        "The model is overfit, so its ranking and its thresholded predictions simply disagree",
        "AUC grades the ranking across all thresholds; the published one sits above every score",
      ],
      answer: 3,
      explanation: "AUC is the chance that a random positive outscores a random negative, so it grades the ranking across every threshold at once. The rule you publish uses one threshold, and if that sits above every score nothing fires and recall is 0. A good ranking is not a good decision rule until a threshold is chosen, on a validation slice.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      return {
        threshold: clampNumber(parsed.threshold, 0.5, 0, 1),
        baseline: asMember(parsed.baseline, BASELINES, initialState.baseline as string),
        split: asMember(parsed.split, SPLITS, initialState.split as string),
        tuneSplit: asMember(parsed.tuneSplit, TUNE, initialState.tuneSplit as string),
        goal: asMember(parsed.goal, GOALS, initialState.goal as string),
        curve: asMember(parsed.curve, CURVES, initialState.curve as string),
        fold: Math.round(clampNumber(parsed.fold, 4, 0, 4)),
        outlier: clampNumber(parsed.outlier, 12, 3, 16),
        population: asMember(parsed.population, POPULATIONS, initialState.population as string),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
