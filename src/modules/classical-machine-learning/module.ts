import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const ALGORITHMS = ["linear", "logistic", "svm", "knn", "tree", "forest", "naivebayes", "kmeans"] as const;

const initialState: ModuleState = {
  algorithm: "linear",
  k: 3,
  depth: 2,
  probeX: 0.5,
  probeY: 0.5,
};

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-32-classical-machine-learning",
  slug: "classical-machine-learning",
  title: "Classical machine learning",
  group: "groundwork",
  order: 4,
  icon: "Split",
  accent: "#c46a2d",
  prerequisites: ["module-31-data-features-representation"],
  estimatedMinutes: 16,
  steps: ["Switch the algorithm", "Change k or depth", "Place a probe", "Read what it stored"],
  stepInstructions: [
    "Set Algorithm to linear and find the two crossed points its straight cut misses, then switch to knn and watch the surface wrap both class-1 corners.",
    "On knn, move k from 1 to 7 and watch the bottom-right pair lose its corner at k = 5. Then switch Algorithm to tree and move Tree depth from 1 to 5.",
    "Move Probe x and Probe y, or click the canvas, and read the probe formula for logistic, SVM, naive Bayes, and the tree's root split.",
    "Read What it stored and compare Numbers stored: 3 weights for the linear methods, all 18 points for knn, and centers that ignore labels for kmeans.",
  ],
  stateVersion: 2,
  tagline: "Keep one dataset and switch linear, logistic, SVM, neighbors, trees, a forest, naive Bayes, and k-means so the boundary change is the lesson.",
  objectives: [
    "Match each classical method to the shape of boundary or grouping it can draw",
    "Say which methods store examples and which fit a small set of parameters",
    "Read logistic output as P(y=1|x), name the SVM margin, and name naive Bayes' independence assumption",
    "Say what a tree's Gini split, an SVM's margin and support vectors, and naive Bayes' product of per-feature likelihoods each compute on the 18 points",
  ],
  glossary: [
    {
      term: "Linear model",
      definition: "A prediction that is a weighted sum of features plus a bias. Its decision surface is a hyperplane, so it cannot bend around a XOR-shaped pattern without new features.",
    },
    {
      term: "Logistic regression",
      definition: "A linear model passed through a sigmoid so the output can be read as P(y=1|x) = σ(w·x+b). The 50% contour is still linear in the features you supplied.",
    },
    {
      term: "Support vector machine",
      definition: "A linear classifier that seeks a wide margin around its hyperplane, trained here on a soft-margin hinge loss. Points on or inside the margin, where t(w·x+b) ≤ 1 for a label t of plus or minus one, are the support vectors; they alone decide the weights.",
    },
    {
      term: "k-nearest neighbors",
      definition: "A predictor that stores the training points and votes with the k closest at query time. There is almost no fitting step; the dataset is the model.",
    },
    {
      term: "Decision tree",
      definition: "A model that splits the space with axis-aligned questions, choosing each cut greedily by Gini gain. Deep trees can isolate single points, which is why they memorize unless you limit depth.",
    },
    {
      term: "Gini impurity",
      definition: "A node's impurity 1 − p² − (1−p)² = 2p(1−p), where p is its share of class 1: 0 for a pure node and 0.5 for an even split. A tree keeps the cut whose size-weighted child impurity falls the most; that fall is the gain.",
    },
    {
      term: "Random forest",
      definition: "A majority vote of several trees, each grown on a bootstrap sample of the rows. Averaging usually lowers variance; the pieces stay axis-aligned. Production forests also sample which features each split may use.",
    },
    {
      term: "Naive Bayes",
      definition: "A classifier that applies Bayes' rule with a prior and a likelihood per feature, assuming features are independent given the class, so P(x,y|c) = P(x|c)P(y|c). The word naive names that assumption, which can be false and still useful.",
    },
    {
      term: "k-means",
      definition: "A clustering method that places k centers and assigns each point to the nearest center. It is unsupervised: it does not use labels, and the number k is a choice you bring, not a fact in the data.",
    },
    {
      term: "Decision boundary",
      definition: "The set of points where the model's predicted class changes. Its shape is a property of the method plus the features, not of the labels alone.",
    },
    {
      term: "Inductive bias",
      definition: "The restrictions a method brings before it sees the data: a line, a margin, a local vote, an axis-aligned cut, or an independence assumption.",
    },
  ],
  references: [
    {
      authors: "Kevin P. Murphy",
      title: "Probabilistic Machine Learning: An Introduction",
      source: "MIT Press, free to read online",
      year: 2022,
      url: "https://mitpress.mit.edu/9780262046824/probabilistic-machine-learning/",
      note: "A current textbook that covers every method in this lab: naive Bayes (Section 9.3), logistic regression (Chapter 10), least squares (11), nearest neighbors (16), SVMs (17), trees and random forests (18), and k-means (21). Section 21.3.7 covers how people pick the number of clusters k.",
    },
    {
      authors: "Tengyu Ma and Andrew Ng",
      title: "CS229 Lecture Notes",
      source: "Stanford University, CS229 Machine Learning course notes",
      year: 2026,
      url: "https://cs229.stanford.edu/main_notes.pdf",
      note: "Stanford's course notes work through the math behind this lab's deep dives: the sigmoid read as P(y=1|x), the SVM's margin and support vectors, the soft margin for data a line cannot separate, naive Bayes, and the k-means loop.",
    },
    {
      authors: "scikit-learn developers",
      title: "Classifier comparison",
      source: "scikit-learn example gallery",
      year: 2026,
      url: "https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html",
      note: "Fits nearest neighbors, a linear SVM, a decision tree, a random forest, naive Bayes, and others to the same small 2-D datasets and draws each decision boundary, the same side-by-side idea as this lab. It warns that what you see on toy data may not carry over to real datasets.",
    },
    {
      authors: "scikit-learn developers",
      title: "Decision Trees",
      source: "scikit-learn User Guide, section 1.10",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/tree.html",
      note: "Gives the Gini impurity formula and the size-weighted split score that this lab's tree uses to pick each cut. It also warns that deep trees overfit unless you limit depth, which the Tree depth slider shows.",
    },
    {
      authors: "scikit-learn developers",
      title: "Naive Bayes",
      source: "scikit-learn User Guide, section 1.9",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/naive_bayes.html",
      note: "States the naive assumption, that features are independent once the class is known, and the prior-times-likelihoods product the lab's probe prints, with one Gaussian per feature. It notes that naive Bayes often works well in spite of that simplified assumption.",
    },
    {
      authors: "Corinna Cortes and Vladimir Vapnik",
      title: "Support-vector networks",
      source: "Machine Learning 20(3), 273–297",
      year: 1995,
      url: "https://www.semanticscholar.org/paper/24e6cf0796237f21c780a3f0c996817f57b3a1bd",
      note: "The paper that extended the widest-margin classifier to data that cannot be split without errors, the soft-margin setting this lab's SVM uses on its XOR points. It also maps inputs to many new features first, the kernel step this lab leaves out.",
    },
    {
      authors: "Leo Breiman",
      title: "Random Forests",
      source: "Machine Learning 45(1), 5–32",
      year: 2001,
      url: "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf",
      note: "The paper that defined random forests: many trees, each grown from its own random draw, voting for the most popular class. It also picks a random set of features to try at each split, the production step this lab's forest leaves out.",
    },
    {
      authors: "S. Lloyd",
      title: "Least squares quantization in PCM",
      source: "IEEE Transactions on Information Theory 28(2), 129–137",
      year: 1982,
      url: "https://www.semanticscholar.org/paper/9241ea3d8cb85633d314ecb74b31567b8e73f6af",
      note: "The source of the name Lloyd's algorithm in the lab's k-means note. Written about coding signals with a few fixed levels, it works out where those levels must sit to make the average squared error smallest, the same goal as k-means' within-cluster sum of squares.",
    },
    {
      authors: "Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux",
      title: "Why do tree-based models still outperform deep learning on typical tabular data?",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track, 507–520",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html",
      note: "On a benchmark of 45 tabular datasets, tree-based models such as random forests and XGBoost still beat deep networks at medium size, about 10,000 rows, which backs the lesson's claim that classical methods stay strong on tables. The authors trace the gap to the different inductive biases of trees and neural networks.",
    },
    {
      authors: "Ravid Shwartz-Ziv and Amitai Armon",
      title: "Tabular data: Deep learning is not all you need",
      source: "Information Fusion 81, 84–90",
      year: 2022,
      url: "https://arxiv.org/abs/2106.03253",
      note: "Compares deep models built for tables with XGBoost, a boosted-tree method, on several datasets, including the ones those models were first tested on. XGBoost won and needed much less tuning, which supports the lesson's point that on tables a larger model has to beat these baselines.",
    },
  ],
  checkpoint: [
    {
      prompt: "Linear, logistic, and SVM each draw one straight line on the uneven XOR. What do they give up, and what can win it back?",
      options: [
        "The two bottom-right class-1 points; k-nearest neighbors with small k wraps both corners",
        "The four top-left class-1 points; k-nearest neighbors with a larger k wraps them back in",
        "Nothing at all, because the sigmoid in logistic regression bends the line around a corner",
        "The twelve class-0 points; naive Bayes wraps them with one Gaussian per class and feature",
      ],
      answer: 0,
      explanation: "A line can take the larger top-left corner but not both class-1 corners, so the two-point bottom-right group is the cost. A small k lets each neighborhood vote locally and keep that pair; from k = 5 up the pair is outvoted. The sigmoid only maps a score to a probability, and one Gaussian per class cannot bend around two opposite corners either.",
      objective: 0,
    },
    {
      prompt: "On What it stored, which method keeps the training points themselves, so its Numbers stored grows with the dataset?",
      options: [
        "Logistic regression, which keeps each point so it can recompute the sigmoid later",
        "Naive Bayes, which keeps the likelihood that each training point contributed",
        "k-nearest neighbors, which votes among the stored points when a query arrives",
        "k-means, which keeps every point's cluster assignment alongside its center",
      ],
      answer: 2,
      explanation: "Linear, logistic, and SVM compress the data into three weights, naive Bayes keeps a prior plus a mean and variance per coordinate per class, and k-means keeps k centers. k-nearest neighbors has almost no fitting step: the dataset is the model, so memory and query cost grow with it.",
      objective: 1,
    },
    {
      prompt: "Logistic regression outputs σ(w·x+b). What is still true of its decision boundary in the original features?",
      options: [
        "It can wrap an XOR pattern, because the S-shaped sigmoid bends the cut around a corner",
        "The 50% contour is still a hyperplane; the sigmoid only maps a score to a probability",
        "It stores the training points and lets the nearest ones vote when a new query arrives",
      ],
      answer: 1,
      explanation: "The sigmoid maps a linear score to a probability. The set where σ(w·x+b) = 0.5 is the same hyperplane as w·x+b = 0. Bending requires a different hypothesis class or new features.",
      objective: 2,
    },
    {
      prompt: "A decision tree compares candidate cuts such as x < 0.15 or y < 0.86. Which cut does it keep?",
      options: [
        "The one whose size-weighted child Gini is lowest, so the impurity falls the most",
        "The one that puts the most points on the correct side, with ties broken at random",
        "The one nearest the middle of the square, so both children stay about the same size",
        "The one that leaves the widest empty street between the two classes in the square",
      ],
      answer: 0,
      explanation: "Each cut is scored by the drop in Gini impurity from the parent to the size-weighted children. On the XOR set the best single cut only peels off two pure class-1 points, a drop of about a ninth, because no one cut separates the classes. Counting points is not the criterion, balance is not rewarded, and a wide street belongs to the SVM.",
      objective: 3,
    },
    {
      prompt: "You delete one point from the SVM's training set and refit. When does the line stay essentially where it was?",
      options: [
        "When the point sat on or inside the street, because those points set the line",
        "When the point is class 1, because the minority class has less pull on the line",
        "Always, because a line fitted through 18 points hardly depends on any one of them",
        "When the point lay well outside the street, with t(w·x+b) above 1 on its side",
      ],
      answer: 3,
      explanation: "A point with t(w·x+b) above 1 pays no hinge cost and exerts no pull, so removing it leaves the optimum unchanged. Points on or inside the street, the support vectors, carry the hinge cost and decide w and b, so deleting one shifts the line. On this non-separable set 12 of 18 points are support vectors, so the line is more sensitive than the textbook picture suggests.",
      objective: 3,
    },
    {
      prompt: "Why is naive Bayes called naive, and does the assumption hold on the XOR points?",
      options: [
        "It sets every class prior to one half, which fails because class 0 has twice as many points",
        "It ignores the labels the way k-means does, so no training accuracy can be reported at all",
        "It assumes a straight boundary, which the XOR arrangement of the corners breaks",
        "It treats x and y as independent within each class, though they correlate strongly here",
      ],
      answer: 3,
      explanation: "Naive Bayes multiplies the prior by a Gaussian likelihood for x and another for y, which is exact only if the two coordinates are independent given the class. Within each class they are strongly correlated, since the class-1 corners lie on an anti-diagonal. Dropping that correlation discards the structure that makes the set XOR-shaped. The priors are used, labels are read, and the boundary is curved, not straight.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      const algorithm =
        typeof parsed.algorithm === "string" && (ALGORITHMS as readonly string[]).includes(parsed.algorithm)
          ? parsed.algorithm
          : initialState.algorithm;
      return {
        ...initialState,
        ...parsed,
        algorithm,
        k: clampInt(parsed.k, 3, 1, 7),
        depth: clampInt(parsed.depth, 2, 1, 5),
        probeX: clampNumber(parsed.probeX, 0.5, 0, 1),
        probeY: clampNumber(parsed.probeY, 0.5, 0, 1),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
