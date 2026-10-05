import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Supervised loop": {
    title: "Five stages, one table",
    summary:
      "The loop is encode → split → choose a hypothesis class → drive a training loss down → read the holdout. Encoding, Algorithm, and Loss are independent knobs. Fit step scrubs real gradient updates; Tree depth is capacity for the CART tree.",
    whatYouSee: [
      "A five-stage flow whose detail row reprints the live vector, split, hypothesis, train loss, or holdout gap.",
      "A stage panel with the current tensors: the feature chip, w and b, or the two losses.",
      "Encoding (rooms + park / integer codes / one-hot), Algorithm (linear / logistic / tree), and Loss (log-loss / MSE).",
      "Fit step from 0 to 48, or grown depth when the algorithm is a tree, and Tree depth from 0 to 5.",
    ],
    howItWorks: [
      "Changing a stage chip only changes which payload is shown. The fit always uses the current encoding, algorithm, and loss.",
      "Linear and logistic start at w = 0, b = 0 and take mean-gradient steps on the twelve train rows. A tree is grown greedily to the chosen depth. Holdout rows never enter a gradient or a split.",
    ],
    controls: [
      "Stage chips, Encoding, Algorithm, Loss, Fit step, and Tree depth.",
      "Comparison worth running: keep logistic and log-loss, switch only Encoding; then keep rooms + park and switch only Loss.",
    ],
    notice: [
      "The pipeline is compositional. One knob can move the holdout number while the others stay put.",
      "Tree depth is ignored by linear and logistic. Fit step on a tree is grown depth, not gradient descent.",
    ],
    limits: [
      "In this lab: sixteen authored apartments, forty-eight GD steps, and a ridge of 10⁻⁴ only inside the least-squares solve. There is no minibatch and no neural net.",
      "In general: a production loop still has these stages. Scaling them up does not make the holdout safe to fit on.",
    ],
  },

  "Decision view": {
    title: "A 28×28 evaluation of the current parameters",
    summary:
      "Each cell evaluates the fitted rule at its (rooms, park) center. Logistic opacity is σ(w·x+b). The solid line is the 50% contour, which is a hyperplane in these features. Tree ticks are axis-aligned splits on rooms or park.",
    whatYouSee: [
      "A rooms × park plane. Filled circles are train; open circles are holdout; the larger mark is the selected row.",
      "A probability or ŷ wash, a solid hyperplane for linear and logistic, and dashed split segments for a tree, each bounded by the rectangle its parent owns.",
      "A note that names the color slice when color is in x, or says the plane is the whole feature space when it is not.",
    ],
    howItWorks: [
      "Linear MSE colors 1[ŷ ≥ 0.5] and draws ŷ = 0.5. Logistic, and linear with log-loss, color P(y=1|x) = σ(w·x+b) and draw w·x+b = 0.",
      "When color is in the encoding, the grid holds color at the selected apartment. That is a slice, not a projection of a fitted 2-D model. Rooms + park has no slice.",
      "Marks that share a (rooms, park) cell are nudged only on the plot. Gradients and splits use the exact vector.",
    ],
    controls: [
      "Click the canvas or a table name to change the selected row and, when color is in x, the slice.",
      "Comparison worth running: rooms + park logistic versus integer codes, then a depth-1 tree versus a depth-5 tree.",
    ],
    notice: [
      "The line moves because w and b moved. It is not an authored cartoon.",
      "A color split draws no line in this plane: the slice follows the branch its color falls into, so only the cuts that apply at this color are drawn.",
    ],
    limits: [
      "In this lab: park is binary, so the points sit on two rails. The wash is a 28×28 sample of a classical-scale rule, not a trained net.",
      "In general: a hyperplane is linear in the features you supplied. A different encoding is a different plane.",
    ],
  },

  "Loss over the fit": {
    title: "The actual sequence of mean losses",
    summary:
      "Train loss (solid) and holdout loss (dashed) are the mean of the selected loss after each real update, with the holdout chosen by Holdout drawn from. For a tree the x-axis is grown depth, not a gradient step. The least-squares dotted line is the closed-form train MSE when that solve exists.",
    whatYouSee: [
      "A line chart of train and holdout loss, a vertical marker at the current step or depth, and four metrics.",
      "On linear + MSE, a dotted least-squares train loss from (X̃ᵀX̃ + λI)⁻¹ X̃ᵀy with λ = 10⁻⁴.",
    ],
    howItWorks: [
      "Log-loss is −[y ln p + (1−y) ln(1−p)] with p = σ(w·x+b) or a leaf fraction. MSE is (prediction − y)². Both are means over the named slice.",
      "The gradient at each step is the exact derivative of the mean train loss: 2(ŷ − y)x averaged for linear MSE, 2(p − y)p(1 − p)x for logistic MSE, and (p − y)x for log-loss. Step sizes are 0.06, 0.7, and 0.85. Holdout is evaluated after the step.",
      "The legend prints each series at its final step or depth; the metrics below print the value at the marker.",
    ],
    controls: [
      "Fit step and Tree depth move the marker. Encoding, Algorithm, and Loss recompute the whole series.",
      "Comparison worth running: logistic log-loss from step 0 to 48, then tree depth 1 versus 5 on integer codes.",
    ],
    notice: [
      "A falling train curve with a rising holdout curve is extra capacity fitting the train slice, including Rowan 9. On integer codes a depth-5 tree reaches train loss 0 while holdout log-loss climbs to about 10.",
      "Accuracy metrics are 0–1 readings of the same predictions. They are not the training objective unless you chose 0–1 loss, which this page does not.",
    ],
    limits: [
      "In this lab: forty-eight full-batch steps from the origin, or six tree depths. There is no validation slice between train and holdout.",
      "In general: the published number still has to be chosen. A small training loss is not generalization, and a two-point holdout is a noisy estimate.",
    ],
  },

  "Train versus holdout": {
    title: "Which rows are allowed into the fit, and where the holdout comes from",
    summary:
      "Sixteen authored apartments. Twelve train, four holdout. The strip and the Split column are the same cut. Holdout drawn from decides whether the four holdout rows come from the same process as the train rows or from a different one, and a two-row table scores the same fit on both. p or ŷ is the current model on that row; it is not a reason to move the row into train.",
    whatYouSee: [
      "A 16-cell strip: train in one fill, holdout in another, high-price rows outlined.",
      "Holdout drawn from, two segments: the same process, or a different process.",
      "A table of name, split, rooms, color, park, y, and the live prediction.",
      "A two-row table that scores the current fit on both holdouts: L holdout, Holdout acc, and how many of the four rows are right. The row for the selected holdout is highlighted. Metrics give L train, which is the same for both, the ratio of the two holdout losses, and the change in holdout accuracy in points.",
    ],
    howItWorks: [
      "The split is a constant. Clicking a name writes selected and, when color is in x, changes the decision-view slice.",
      "Rowan 9 is labeled 0 despite 3 rooms and a park, an authored exception on train. The same-process holdout rows follow rooms ≥ 3 or (rooms = 2 and park).",
      "The different-process rows are Quay 3 (1 room, blue, no park, high), Reed 5 (2 rooms, green, park, high), Slate 7 (3 rooms, red, park, low), and Tarn 11 (4 rooms, blue, no park, low). Their labels follow high price when rooms ≤ 2, the reverse of the usual rule, so a rule learned on train gets three of the four wrong.",
      "The fit reads only the twelve train rows, so it and its train loss are identical whichever holdout is selected. Both holdouts are always scored at the current Fit step and Tree depth; the selected one also drives the strip, the decision view, the loss chart, and the metrics elsewhere.",
    ],
    controls: [
      "Click a row name, and use Holdout drawn from. The split itself has no slider, on purpose.",
      "Comparison worth running: on rooms + park, logistic, and log-loss, switch Holdout drawn from. Holdout accuracy falls from 100 percent to 25 percent and holdout log-loss rises from 0.31 to 1.39, while L train stays 0.45.",
      "Comparison worth running: Rowan 9 versus Laurel 2 under a depth-5 tree, with the same-process holdout.",
    ],
    notice: [
      "Red is mostly high on train and not on holdout. That is the spurious color cue a deep tree can memorize.",
      "A holdout prediction that looks good is still not evidence the row was used in the fit. It was not.",
      "A log-loss above 0.69, the loss of answering 0.5 for every row, means the fit is doing worse than no model at all on those rows.",
    ],
    limits: [
      "In this lab: the sixteen rows and the split are authored, and the shift is one extreme case, a single reversed rule on four rows. There is no random shuffle, no third validation slice, and a four-row holdout reads accuracy in steps of 25 points.",
      "In general: shift is usually partial and gradual, in the inputs, the labels, or the rule linking them. A holdout cannot reveal it, because it was drawn from the old process; it is caught by monitoring new data. A real holdout must also stay unseen for every choice, including encoding statistics and early stopping, which is the next lab.",
    ],
  },

  "Worked example": {
    title: "w, b, one x, one loss term",
    summary:
      "The selected apartment is expanded into x, the current w and b (or leaf p), the model output, and that row's contribution to the mean loss. The notation matches the curve.",
    whatYouSee: [
      "Metrics for b and each w_feature, a feature chip, and a FormulaWithValues block.",
      "On linear + MSE, the closed-form least-squares w and b printed as a note, separate from the GD weights.",
    ],
    howItWorks: [
      "ŷ = w·x+b for linear MSE. P(y=1|x) = σ(w·x+b) for logistic and for linear log-loss. A tree reports the train fraction in the leaf that contains x.",
      "The printed ℓ is that one example's loss. The chart averages the same formula over a slice.",
    ],
    controls: [
      "Click a row. Encoding, Algorithm, Loss, and Fit step rewrite every number.",
      "Comparison worth running: Alder 12 (train, y = 1) versus Laurel 2 (holdout, y = 0) at the same step.",
    ],
    notice: [
      "If the selected row is holdout, the note says so. Its ℓ is not in ∇L.",
      "Closed-form least squares and the GD path are different objects until the dotted line is met.",
    ],
    limits: [
      "In this lab: one example at a time, classical-scale arithmetic. There is no softmax over many classes.",
      "In general: the same objects — a vector, a score, a loss term — scale to a deep net. The scale is later labs, not this one.",
    ],
  },
};

export default cardInfo;
