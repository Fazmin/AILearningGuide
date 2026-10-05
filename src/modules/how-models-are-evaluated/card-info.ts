import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Threshold and baseline": {
    title: "Two populations, two dumb rules, one threshold",
    summary:
      "One hundred scored rows in two populations. Population 1 in 100 is the deliberate imbalance demo: a single positive at row 99 with score 0.42 and a confident false alarm at row 12. Population 1 in 5 has twenty positives whose scores overlap the negatives' scores, so a threshold has real trade-offs to make. Baseline chooses always negative, always positive, or a threshold on those scores, and Scored split chooses which slice counts. The scores are not a trained loss; the metrics are readings of a thresholded matrix.",
    whatYouSee: [
      "A kicker with the number of scored rows and how many are truly positive, and a title that names the population.",
      "Population with two segments, 1 in 100 (rare) and 1 in 5 (about 20%); Baseline with three segments; Decision threshold from 0.00 to 1.00; and Scored split with all 100, train, val, and test.",
      "A score strip: every row as a dot at its detector score, stacked where scores repeat. Positives are diamonds and negatives are circles, so the classes differ by shape and not only by colour. On 1 in 100, row 99 is the diamond at 0.42 and row 12 a ringed circle at 0.88. The shaded side of the t line is what the rule predicts positive; rows outside the scored split are faded.",
      "A note that says what fires at the current setting and describes the population.",
    ],
    howItWorks: [
      "Train is rows 0–69, val 70–84, test 85–99 in both populations. On 1 in 100 the only positive is row 99, so train and val are all negatives. On 1 in 5 the twenty positives are placed four to every block of twenty rows, which leaves 14 in train, 3 in val, and 3 in test: exactly 20 percent in each slice.",
      "On 1 in 5 each score is drawn from a bell: positives centre on 0.64 and negatives on 0.30, with spreads of 0.16 and 0.14, clipped to 0.02–0.98 and rounded to three decimals, from a fixed seed. The two bells overlap, so no threshold separates the classes.",
      "Always negative predicts 0. Always positive predicts 1. Learned predicts 1 when score ≥ threshold. Threshold is ignored by the two constant rules.",
    ],
    controls: [
      "Population, Baseline, Decision threshold, and Scored split are the four controls.",
      "Comparison worth running on 1 in 100: always negative on all 100, then score ≥ threshold at 0.42, 0.60, and 0.90. At 0.42 both rows 99 and 12 fire; at 0.60 only row 12; at 0.90 nothing.",
      "Comparison worth running on 1 in 5: score ≥ threshold at 0.30, 0.50, and 0.70. False alarms leave first and real positives leave later, which is the precision-recall trade-off.",
    ],
    notice: [
      "Train and val can look perfect for a never-fire rule on 1 in 100 because they contain no positives. On 1 in 5 the same rule scores about 80 percent, because 80 percent of the rows are negative.",
      "The threshold only matters when Baseline is score ≥ threshold.",
    ],
    limits: [
      "In this lab: scores and labels are authored or drawn from a fixed seed, not outputs of the previous lab's classifier. There is no fitted model and no training loss on this card, and no model from the classical-ML or supervised-loop labs is scored.",
      "In general: a real holdout must stay unseen, and a real detector's score distributions are neither bells nor known in advance. Tuning the threshold on the test slice is the same leak the Evaluation lab later names contamination.",
    ],
  },

  "Train, val, test": {
    title: "The strip, and the leak",
    summary:
      "A 100-cell strip: train, val, test. Choose t using picks which slice searches for a threshold that maximizes accuracy, recall, or F1. Set t from that slice writes the threshold. Doing that on test marks the holdout leaked. A table then compares the intact path (t from val) with the leaked path (t from test) on the test slice and on 400 fresh rows from the same scoring process.",
    whatYouSee: [
      "A 100-cell strip. Train, val, and test use different fills. Positives are outlined; on 1 in 100, row 99 and row 12 have extra marks.",
      "Choose t using (train / val / test), Maximize (accuracy / recall / F1), and a Set t button that prints the winning threshold.",
      "Metrics t chosen, the goal's value on the tuning slice at that t, Test accuracy, recall, or F1, and Holdout? intact or leaked.",
      "A two-row table. Each row gives its threshold, its Test claim (the reading you would publish), its Fresh rows reading, and Claim − fresh, the amount the claim exceeds what fresh rows show. A metric prints Leak inflation: leaked minus intact test reading.",
    ],
    howItWorks: [
      "The search tries thresholds 0.00, 0.01, …, 1.00 on the chosen slice only, then scores the frozen t on the test slice. Ties go to the highest t.",
      "On 1 in 100, train and val have no positives, so recall is 0/0 there at every t, the search finds nothing, and it falls back to t = 1.00. Test contains the only positive; searching there finds t = 0.42 and is leakage.",
      "On 1 in 5 every slice has positives, so the val search is meaningful. The table always runs both searches: t from val, and t from test. Fresh rows are 400 more draws from the same scoring process, exactly 20 percent positive, which never take part in either choice. On 1 in 100 there is no fresh batch and the column reads n/a.",
    ],
    controls: [
      "Choose t using, Maximize, and Set t from …",
      "Comparison worth running on 1 in 100: choose t on val for recall (t = 1.00, Test recall 0%), then choose t on test for recall (t = 0.42, Test recall 100%) and watch Holdout? flip to leaked.",
      "Comparison worth running on 1 in 5: Maximize F1 with t from val, then with t from test. Test F1 rises, but the Fresh rows reading for the leaked t falls below the claim.",
    ],
    notice: [
      "Leakage inflates the claim, not the detector. The leaked threshold was fitted to fifteen rows, so its test number is the best of 101 tries on a small sample.",
      "Maximizing accuracy on the rare set prefers a t that never fires. Maximizing recall prefers a low t that fires on many rows. F1 balances both. Those are different searches.",
      "A leaked test number looks better and is the wrong claim.",
    ],
    limits: [
      "In this lab: a grid search over 101 thresholds, one authored positive on 1 in 100, and one seeded draw on 1 in 5. The seed was chosen so that the effect is visible; across 400 re-draws of the same process the leaked claim exceeded the intact one in most draws, and the size varies a great deal with fifteen test rows. No model is retrained.",
      "In general: leakage also includes features computed with test statistics, early stopping on the test curve, and benchmark text that sat in the training crawl. A fresh batch is a luxury; in practice the cure is to keep a test slice that no choice has seen.",
    ],
  },

  "Confusion matrix": {
    title: "Four counts the scalars are made from",
    summary:
      "A 2×2 of true class versus predicted class. The heatmap shows each cell as a share of the scored rows so the rare positive does not disappear visually, and the table prints the raw TN, FP, FN, and TP counts.",
    whatYouSee: [
      "A heatmap with rows true 0 / true 1 and columns pred 0 / pred 1, values as percents of the scored set.",
      "A table of the same four cells with integer counts. The true-1 row highlights when the positives are all missed.",
      "A note that every scalar on the next card is a reading of this table.",
    ],
    howItWorks: [
      "TN, FP, FN, and TP are tallied from the current baseline, threshold, and split.",
      "Heatmap values are those counts divided by the number of scored rows, so they sum to one.",
    ],
    controls: [
      "This card has no controls. The previous cards rewrite every cell.",
      "Comparison worth running: always negative versus always positive on all 100, and watch which off-diagonal fills. Then, on 1 in 5, slide the threshold and watch FP drain into TN before TP drains into FN.",
    ],
    notice: [
      "A 99% true-negative cell can hide a single false negative.",
      "When the detector never fires, TP is 0 and every positive is an FN: 1 of 1 on 1 in 100, 20 of 20 on 1 in 5.",
    ],
    limits: [
      "In this lab: two classes, and one authored positive on 1 in 100 or twenty seeded positives on 1 in 5. There is no cost matrix and no confidence interval on a cell.",
      "In general: a confusion matrix is still the object. Accuracy, F1, and AUC are different readings of related tables, not separate truths.",
    ],
  },

  "Four readings": {
    title: "Accuracy, precision, recall, and F1 from one table",
    summary:
      "Accuracy is (TP+TN)/N. Precision is TP/(TP+FP). Recall is TP/(TP+FN). F1 is 2PR/(P+R) = 2TP/(2TP+FP+FN). A reading whose denominator is 0 prints n/a. On 1 in 100, always negative prints 99% accuracy and 0% recall; on 1 in 5 it prints 80% and 0%. Two further readings hold the detector fixed and change only the share of positives.",
    whatYouSee: [
      "Four metrics: Accuracy, Precision, Recall, and F1, each as a percent or n/a.",
      "A note that names the never-found case, the n/a case, or the two fractions.",
      "A threshold sweep: accuracy (solid), precision (dashed), recall (dotted), and F1 (dash-dot) for every t from 0 to 1 on the scored rows, with labelled dots where the current t crosses each line.",
      "An ROC plot: true positive rate against false positive rate as t falls, the shaded area under it, the dashed chance diagonal, and a dot at the current t. Metrics AUC (ranking) and what it means for this slice.",
      "Two base-rate metrics: Precision if 1 row in 5 is positive and Precision if 1 row in 100 is positive, for the rule currently selected.",
      "A note that these scalars are not the training loss.",
    ],
    howItWorks: [
      "The scalars are computed from the same TP, FP, TN, FN as the matrix. When TP+FP = 0, precision is 0/0 and has no value; when TP+FN = 0, recall is 0/0 and has no value. The card prints n/a instead of pretending either is 0% or 100%.",
      "The sweep always evaluates the score ≥ t rule, even while a constant baseline is selected, so you can see what each threshold would give. A gap in a line is a range of t where that reading is 0/0.",
      "AUC is the Mann–Whitney probability that a random positive outscores a random negative, ties counting one half; it equals the area under the ROC steps. On 1 in 100 and all 100 rows it is 98/99 ≈ 0.990 because only row 12 outscores row 99. On train or val of 1 in 100 there is no positive and no ROC.",
      "The base-rate readings use Bayes' rule on the matrix: TPR·π / (TPR·π + FPR·(1−π)), with TPR and FPR taken from the current rule on the scored rows and π set to 0.20 or 0.01. At the rule's own base rate the formula returns its own precision.",
    ],
    controls: [
      "This card has no controls.",
      "Comparison worth running: on 1 in 5 with Baseline on score ≥ threshold, set t to 0.50 and read Precision against Precision if 1 row in 100 is positive. Same rule, same TPR and FPR, far lower precision.",
      "Comparison worth running: always negative on all 100 versus the same rule on train of 1 in 100. Both look accurate; only all 100 still has a positive to miss.",
    ],
    notice: [
      "Precision does not mention the misses. Recall does not mention the false alarms.",
      "A split with zero positives cannot measure recall of the rare class; it prints n/a.",
      "On 1 in 100 the accuracy line barely moves across t while recall drops off a cliff at 0.42. On 1 in 5 recall falls steadily, precision climbs, and accuracy peaks in the middle. That picture is why accuracy is the wrong headline on a rare class.",
      "AUC ≈ 0.99 and 0% recall at t = 0.50 can both be true on 1 in 100: AUC grades the ranking, recall grades one threshold.",
    ],
    limits: [
      "In this lab: on 1 in 100 one positive makes the ROC a single step and AUC hinges on one row; on 1 in 5 there are twenty positives and the curve has twenty steps. Percents on 100 rows are exact counts, not estimates with error bars. There is no calibration plot: neither population has enough rows to fill reliability bins.",
      "In general: the metric you publish is a choice. Accuracy on an imbalanced detector is often the wrong choice, a held-out set is required before any of the numbers mean generalization, and the base rate in production rarely equals the base rate in the evaluation set.",
    ],
  },

  "Learning curves": {
    title: "Three illustrative pairs of train versus val",
    summary:
      "Pattern chooses underfit, good fit, or overfit. The series are closed-form sketches that all start near the same loss: both flatten high; both fall and stay close; or train keeps falling while val bottoms out and turns up. They are not a model trained on the 100 rows.",
    whatYouSee: [
      "Pattern, a three-way segmented control.",
      "A line chart of train loss (solid) and val loss (dashed) over epochs 0 to 20, with a vertical guide at the epoch of lowest val loss. The legend prints both losses at epoch 20.",
      "Metrics Lowest val loss (epoch and value), Val rise since then, and Val − train at 20.",
      "A footnote that names the diagnosis, and a note that the curves are illustrative.",
    ],
    howItWorks: [
      "Underfit: train 0.45 + 0.55e^(−t/3), val 0.04 higher. Good fit: train 0.14 + 0.88e^(−t/5), val 0.19 + 0.86e^(−t/5). Overfit: train 0.05 + 0.9e^(−t/4); val 0.10 + 0.88e^(−t/4) + 0.0009t², lowest at epoch 10.",
      "The guide is the argmin of the val series — where early stopping would keep the weights.",
      "Nothing is optimized. The 100 detector scores do not enter this card.",
    ],
    controls: [
      "Pattern is the only control.",
      "Comparison worth running: overfit versus good fit. The tell is the val turn, not the final train number.",
    ],
    notice: [
      "A low training loss is not a diagnosis. The pair of curves is.",
      "Early stopping would use the val curve, not the test curve. Using test here would be leakage again.",
    ],
    limits: [
      "In this lab: three illustrative formulas. There is no optimizer and no random seed.",
      "In general: a real run can look like any of the three, and a val curve used for many decisions still needs a final untouched test.",
    ],
  },

  "Cross-validation": {
    title: "Five folds, one never-fire rule",
    summary:
      "The 100 rows are cut into five contiguous folds of 20. Held-out fold chooses which slice is scored. The rule is always negative. On 1 in 100 four folds print 100% accuracy and fold 4 prints 95%, the fold that contains row 99. On 1 in 5 every fold holds four positives and prints 80%.",
    whatYouSee: [
      "Held-out fold, a slider from 0 to 4 that names the row range.",
      "A strip that marks the held-out fold, a five-row table of always-negative accuracy, recall, and positives in each fold, and metrics for this fold, the CV mean, and this fold's recall.",
    ],
    howItWorks: [
      "Fold i is rows 20i through 20i+19. Always negative is the constant rule from the first card.",
      "CV mean accuracy is the unweighted mean of the five fold accuracies. Recall is 0/0, n/a, on every fold that lacks a positive, and 0% on a fold that has one, because the rule misses it.",
      "On 1 in 5 the twenty positives are placed four to a fold, so the folds are stratified by construction and recall is defined on all five.",
    ],
    controls: [
      "Held-out fold is the only control.",
      "Comparison worth running: fold 0 versus fold 4 on each population. On 1 in 100 mean accuracy barely moves and recall only exists on fold 4. On 1 in 5 recall exists on every fold.",
    ],
    notice: [
      "Cross-validation does not repair a bad metric. It averages that metric more fairly.",
      "On 1 in 100 only one fold can see the rare class; on 1 in 5 all five can.",
    ],
    limits: [
      "In this lab: contiguous folds over all 100 rows, a constant rule, and no model refit per fold. The 1 in 5 folds are stratified by how the positives were placed, and the 1 in 100 folds cannot be stratified at all. The folds include the test slice that a real protocol would keep out.",
      "In general: production CV refits the model on each training union. Stratification matters when the rare class can miss a fold entirely, as it does on 1 in 100.",
    ],
  },

  "Regression residuals": {
    title: "MAE versus MSE on one outlier",
    summary:
      "Five points and the identity model ŷ = x. Outlier y moves the fifth point. MAE is the mean absolute residual; MSE is the mean square. The outlier dominates MSE and barely moves MAE in comparison.",
    whatYouSee: [
      "Outlier y from 3 to 16, a plot of ŷ = x with a residual tick at each point and the outlier labelled, and a five-row table of residual, absolute residual, and squared residual.",
      "Metrics MAE, MSE, RMSE, and the outlier's share of the summed squared error.",
    ],
    howItWorks: [
      "The first four points are (0, 0.2), (1, 1.1), (2, 1.9), (3, 3.2). The fifth is (4, outlier). Prediction is ŷ = x, not a fitted line.",
      "MAE = mean |y−ŷ|. MSE = mean (y−ŷ)². RMSE is the square root of MSE.",
    ],
    controls: [
      "Outlier y is the only control.",
      "Comparison worth running: set the outlier to 4, then to 16, and watch MSE leave MAE behind.",
    ],
    notice: [
      "MSE is the usual training loss for a squared residual. It is not automatically the number you should publish.",
      "This card does not use the 100-row detector. It is a second toy so regression has its own matrix-like object: the residual list.",
    ],
    limits: [
      "In this lab: five authored points and an identity predictor. No line is fitted, so this is not least squares.",
      "In general: a fitted least-squares line would chase the outlier. That is a different demonstration of the same square.",
    ],
  },
};

export default cardInfo;
