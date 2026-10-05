import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Algorithm": {
    title: "Eight methods, one control",
    summary:
      "Algorithm chooses linear, logistic, svm, knn, tree, forest, naivebayes, or kmeans on the same 18 points. k is consumed by neighbors and k-means; Tree depth is consumed by the tree and the forest. Linear, logistic, SVM, and naive Bayes ignore both sliders.",
    whatYouSee: [
      "A kicker One dataset, eight methods and a badge that repeats the algorithm string.",
      "Algorithm, an eight-way segmented control.",
      "Sliders k from 1 to 7 and Tree depth from 1 to 5.",
      "A note that names which slider each method reads.",
    ],
    howItWorks: [
      "The 18 points are constants: an uneven XOR with twelve class-0 points on one diagonal, four class-1 points top-left, and two class-1 points bottom-right. Switching Algorithm refits that method and redraws the surface.",
      "Linear solves the least-squares normal equation for labels in {0,1}. Logistic runs 80 full-batch gradient steps on log-loss from w = 0. SVM runs 3000 primal subgradient steps on ½‖w‖² + 8·Σ hinge with a shrinking step and keeps the lowest-objective iterate. knn votes at query time. The tree splits on x or y by Gini gain 2p(1−p). The forest majority-votes 7 bootstrap trees. Naive Bayes fits a prior and a 1-D Gaussian per coordinate per class. k-means runs 12 assign-and-average steps and never reads a label.",
    ],
    controls: [
      "Algorithm switches the method. k and Tree depth only matter for the methods named in the note.",
      "Comparison worth running: linear versus knn with k = 1, then logistic versus SVM, then naive Bayes.",
    ],
    notice: [
      "The points do not move. Only the region colors change.",
      "k-means can look like a classifier while ignoring every label.",
    ],
    limits: [
      "In this lab: 18 authored 2-D points and closed-form or short-loop fits. There is no kernel SVM and no library solver; the SVM and logistic fits are close to, not exactly at, their optima.",
      "In general: a linear model can bend only after you add features. A deep tree can memorize. k is a choice you bring, not a fact in the data.",
    ],
  },

  "Decision surface": {
    title: "A 28×28 grid of predictions",
    summary:
      "Each cell is colored by the current method's prediction at its center. Logistic and naive Bayes also set opacity from P(y=1|x). Circles are the training points, always marked by their true label. The open circle is the probe.",
    whatYouSee: [
      "A square canvas of 784 cells with x and y ticks from 0 to 1, 18 labeled circles, and one probe ring.",
      "A dark edge wherever the predicted class (or cluster id) changes between neighbouring cells, and a small × over every training point the method misclassifies.",
      "A solid hyperplane for linear, logistic, and SVM; dashed margin lines and ringed support vectors for SVM; split segments for a tree (or tree 0 of the forest), each bounded by its parent's region; neighbor spokes for knn; numbered crosses for k-means centers.",
      "A legend that names shading, point labels, misclassification marks, and the probe, so color is never the only cue.",
    ],
    howItWorks: [
      "The grid evaluates the fitted rule at cell centers. Linear thresholds a 3-weight score at 0.5. Logistic and naive Bayes threshold a probability at 0.5, and opacity grows with |P(y=1|x) − 0.5|. SVM thresholds w·x+b at 0. knn majority-votes k neighbors. The tree walks axis-aligned splits. The forest majority-votes 7 trees. k-means shades by nearest center index.",
      "Point marks always use the authored label, even under k-means. Clicking the canvas writes probeX and probeY.",
    ],
    controls: [
      "Click the canvas to move the probe. Algorithm, k, and Tree depth redraw the surface.",
      "Comparison worth running: tree depth 1 versus depth 5, then forest at the same depth, then k-means with k = 2 versus k = 5.",
    ],
    notice: [
      "Linear, logistic, and linear SVM each draw one line that takes the top-left corner and crosses out the two bottom-right points. knn with k ≤ 4, deep trees, and the forest can wrap both corners.",
      "Logistic opacity is σ(z), not decoration. SVM's dashed slab is |w·x+b| < 1.",
    ],
    limits: [
      "In this lab: the grid is a display of a toy fit at 28×28 resolution, so the drawn edge is stepped. Forest split segments are from tree 0 only. SVM is linear soft-margin, not a kernel machine. k-means shading cycles four tones; the center numbers name the clusters.",
      "In general: a decision boundary is a property of the method plus the features. Change the encoding from the previous lab and the same method draws a different cut.",
    ],
  },

  "Probe and inductive bias": {
    title: "What the method assumes, at one query",
    summary:
      "Probe x and Probe y choose a query. The formula row is the number that method actually computes there: a linear score, σ(z), a margin, a k-vote, a forest vote, or a naive Bayes posterior. The note names the inductive bias.",
    whatYouSee: [
      "Sliders Probe x and Probe y from 0 to 1.",
      "A FormulaWithValues block whose expression matches the selected algorithm.",
      "A note that states the bias: hyperplane, probability, margin, local vote, axis-aligned cut, averaged trees, conditional independence, or Voronoi.",
    ],
    howItWorks: [
      "The probe is independent of fitting. Moving it does not refit the 18 points.",
      "Naive Bayes posterior uses the product of two 1-D Gaussians times the class prior — the independence assumption, written as a formula. Its detail line prints each class's prior × density of x × density of y; at the default probe (0.50, 0.50) that is 1.574 for class 0 and 0.527 for class 1, so the posterior is 0.251.",
      "The tree's detail line scores the root cut by Gini: the parent's 0.444, the size-weighted child Gini 0.333 for x < 0.15, and the gain 0.111, the largest of any single cut. The SVM's detail line adds the support-vector count, 12 of 18, to the street width 2/‖w‖ = 0.54.",
    ],
    controls: [
      "Probe x, Probe y, or a click on the canvas.",
      "Comparison worth running: put the probe in a XOR corner on linear, then on knn with k = 1, then on naive Bayes.",
    ],
    notice: [
      "Logistic's sigmoid does not bend the 50% line. Naive Bayes assumes x ⟂ y | class; class 1 sits in two opposite corners, so that is false, and its posterior passes 0.5 only in the far top-left.",
      "Forest votes can disagree with tree 0's drawn splits, because the other six trees saw different bootstrap samples.",
    ],
    limits: [
      "In this lab: one probe, closed-form readouts. There is no uncertainty band around a probability.",
      "In general: inductive bias is why two methods that both “learn from data” draw different pictures on the same points.",
    ],
  },

  "What it stored": {
    title: "Weights, points, or centers",
    summary:
      "After fitting, linear, logistic, and SVM keep three weights; knn keeps the 18 points; the tree keeps its splits and leaves; the forest keeps 7 trees; naive Bayes keeps a prior plus a mean and variance per coordinate per class; k-means keeps k centers. Accuracy is replaced by within-cluster sum of squares for k-means, because that method never saw a label.",
    whatYouSee: [
      "Metrics Keeps, Numbers stored, and Train accuracy (or Within-cluster SS for k-means).",
      "A note that names what the current method stored and, for trees, how many leaves the current depth grew.",
    ],
    howItWorks: [
      "Train accuracy is the count of the 18 points whose predicted class matches the authored label. k-means prints Σ‖x − nearest center‖² instead.",
      "Numbers stored counts parameters after fitting: 3 weights; 18 × 3 values for knn; 2 per split plus 1 per leaf for a tree, summed over 7 trees for the forest; 9 for naive Bayes; 2k for k-means.",
    ],
    controls: [
      "This card has no controls.",
      "Comparison worth running: read Numbers stored on linear, then on knn, then on naive Bayes. One compressed the data; one is the data; one stored a factored density.",
    ],
    notice: [
      "A high training accuracy on 18 points is not a test score.",
      "Depth 1 versus depth 5 is the bias-variance sketch on this page: one cut gets 14 of 18; seven leaves get all 18 by isolating single points.",
    ],
    limits: [
      "In this lab: accuracy is training accuracy on the same 18 points used to fit. There is no held-out set on this page.",
      "In general: storing the dataset (knn) and compressing it (linear) fail in different ways — memory and distance choice versus a boundary that cannot bend.",
    ],
  },
};

export default cardInfo;
