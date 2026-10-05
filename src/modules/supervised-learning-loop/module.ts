import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { ALGORITHMS, ENCODINGS, HOLDOUT_SOURCES, LOSSES, TREE_MAX_DEPTH } from "./fit";

/**
 * Version 2 adds `holdoutSource` (the holdout from the same process as the train rows, or from a different one)
 * and a fifth `lastKnob` value. A version 1 payload has neither, so it keeps the same-process holdout it was
 * saved on.
 */
const initialState: ModuleState = {
  stage: 3,
  encoding: "numeric",
  algorithm: "logistic",
  loss: "logloss",
  step: 48,
  depth: 2,
  selected: 0,
  lastKnob: "algorithm",
  holdoutSource: "same",
};

const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-50-supervised-learning-loop",
  slug: "supervised-learning-loop",
  title: "The supervised learning loop",
  group: "groundwork",
  order: 5,
  icon: "Repeat",
  accent: "#3c7a8f",
  prerequisites: ["module-32-classical-machine-learning"],
  estimatedMinutes: 16,
  steps: [
    "Encode one row",
    "Read the split",
    "Choose a hypothesis",
    "Scrub the fit",
    "Swap one stage",
    "Raise tree depth",
    "Shift the holdout",
  ],
  stepInstructions: [
    "Set Encoding to rooms + park, then to integer codes, and read the feature chip for Alder 12.",
    "On Train versus holdout, click Rowan 9 (train, the exception) then Laurel 2 (holdout) and notice only train rows enter the fit.",
    "Set Algorithm to linear, then to logistic, then to tree, and watch the rooms × park surface change family.",
    "Move Fit step from 0 to 48 on logistic and read train loss against holdout loss on the curve.",
    "Keep Algorithm on logistic and switch only Encoding, then only Loss, and see which knob moves the holdout number.",
    "Set Encoding to integer codes and Algorithm to tree, then move Tree depth from 1 to 5. Training loss falls to 0 while holdout loss rises.",
    "On Train versus holdout, set Holdout drawn from to a different process and compare L holdout and Holdout acc in the two table rows while L train stays put.",
  ],
  stateVersion: 2,
  tagline:
    "Run one apartment table through encode, a frozen holdout, a hypothesis class, and a real fit, swap one stage and watch which number moves, then score the same fit on a holdout from a different process.",
  objectives: [
    "Name the supervised loop as encoding, split, hypothesis class, loss, and optimization, then a holdout reading, and say why a split is drawn at random and what train, validation, and test slices are for",
    "Show that changing only encoding, algorithm, or loss can move the holdout number while the other stages stay put",
    "Read a falling train loss against a rising holdout loss as extra capacity memorizing the train slice",
    "Show that a holdout drawn from a different process can collapse the score of an unchanged fit, so a holdout only estimates performance on rows like the training rows",
  ],
  glossary: [
    {
      term: "Hypothesis class",
      definition:
        "The family of functions you are willing to search. Here that is a linear score, a logistic sigmoid of that score, or an axis-aligned tree. Changing H is a different search, not a different dataset.",
    },
    {
      term: "Loss",
      definition:
        "The number the fit actually drives down on the training slice: mean squared error or mean log-loss. It is not automatically the number you should publish.",
    },
    {
      term: "Optimization",
      definition:
        "The procedure that searches inside H. Linear and logistic take real full-batch gradient steps on the mean training loss from (w,b) = 0. A tree grows greedy Gini or entropy splits. The holdout is never used to choose a step.",
    },
    {
      term: "Training set",
      definition:
        "The twelve apartments whose features and labels enter X and y. Rowan 9 is an authored exception in this slice so a deep tree has a point it can isolate.",
    },
    {
      term: "Holdout",
      definition:
        "The four apartments kept out of every gradient and every split. Their loss is a reading after the fit, not a signal the optimizer is allowed to see.",
    },
    {
      term: "Random split",
      definition:
        "Dividing the rows into train and holdout by shuffling them with a seed, so the cut is not chosen by hand and can be repeated. It makes a holdout that differs from train in some unintended way unlikely, not impossible: drawing 4 of this table's 16 rows, 7.7 percent of the draws leave all four on one side of the price label.",
    },
    {
      term: "Validation and test sets",
      definition:
        "Train fits the parameters. Validation is the slice you look at to choose settings such as encoding, algorithm, or tree depth. Test is scored once at the end. This lab has only train and holdout, so the holdout doubles as a validation set and its numbers are optimistic once you have chosen by them.",
    },
    {
      term: "Encoding",
      definition:
        "The map from a table row to the vector x the model sees. Integer codes include color as 1, 2, 3. One-hot gives each color a slot. Rooms + park drops color.",
    },
    {
      term: "Capacity",
      definition:
        "How flexible H is. Tree depth is the control on this page. Extra depth can drive training loss down by isolating a single train row.",
    },
    {
      term: "Generalization",
      definition:
        "Error on new rows from the same process. Holdout loss is the estimate this lab can compute. A small training loss does not, by itself, measure it.",
    },
    {
      term: "Distribution shift",
      definition:
        "The process that produces the rows you score differs from the one that produced the training rows: different inputs, different labels, or a different rule linking them. A holdout drawn from the old process cannot reveal it. Here the rule linking rooms to price flips, which is the concept-shift kind.",
    },
    {
      term: "Log-loss",
      definition:
        "Mean −[y ln p + (1−y) ln(1−p)] with p = σ(w·x+b) for logistic, or a tree leaf fraction. A confident wrong p is expensive.",
    },
    {
      term: "Decision boundary",
      definition:
        "The set of rooms × park points where the predicted class flips. For logistic that is the hyperplane w·x+b = 0, the 50% contour of σ(w·x+b).",
    },
  ],
  references: [
    {
      authors: "Moritz Hardt and Benjamin Recht",
      title: "Patterns, Predictions, and Actions: Foundations of Machine Learning",
      source: "Princeton University Press, free to read online",
      year: 2022,
      url: "https://mlstory.org/",
      note: "A textbook built around the same train-then-test routine this lab runs. The Supervised learning chapter treats fitting as lowering the average loss on a sample, and the Datasets chapter explains the holdout method and what goes wrong when one test set is reused many times.",
    },
    {
      authors: "scikit-learn developers",
      title: "Encoding categorical features",
      source: "scikit-learn User Guide, section 8.3.4",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features",
      note: "Explains that integer codes make a model treat categories as ordered even when the order was picked arbitrarily, and that one-hot encoding gives each category its own slot. This is the integer codes versus one-hot choice in the Encode step.",
    },
    {
      authors: "scikit-learn developers",
      title: "Cross-validation: evaluating estimator performance",
      source: "scikit-learn User Guide, section 3.1",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/cross_validation.html",
      note: "Shows how to cut a table into train and test slices with a fixed random seed, and how stratified splits keep the class share close to the same in each slice. It also warns that tuning settings against the test set leaks it into the model, which is why a separate validation set is held out.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapters 5 and 6 build the least-squares and binary cross-entropy (log-loss) losses and lower them with gradient descent, the same losses and downhill steps as the Fit stage. Chapter 8 explains why settings are chosen on a validation set and the test set is kept for the final score.",
    },
    {
      authors: "scikit-learn developers",
      title: "Decision Trees",
      source: "scikit-learn User Guide, section 1.10",
      year: 2026,
      url: "https://scikit-learn.org/stable/modules/tree.html",
      note: "Gives the Gini and entropy scores a tree uses to pick each split, the two this lab's tree switches between. It also warns that deep trees overfit and suggests limiting depth, the effect the Raise tree depth step shows.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, et al.",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online",
      year: 2023,
      url: "https://d2l.ai/",
      note: "Section 3.6 compares training error with generalization error and shows overfitting as a model gets more complex. Section 4.7 sorts distribution shift into covariate, label, and concept shift; the different-process holdout in this lab is concept shift.",
    },
    {
      authors: "Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al.",
      title: "Understanding deep learning (still) requires rethinking generalization",
      source: "Communications of the ACM 64(3), 107–115",
      year: 2021,
      url: "https://www.semanticscholar.org/paper/2f65c6ac06bfcd992d4dd75f0099a072f5c3cc8c",
      note: "Shows that large neural networks can fit training labels that were assigned at random, so they can memorize rather than learn. It backs this lesson's point that a small training loss does not, by itself, measure generalization.",
    },
    {
      authors: "Sayash Kapoor and Arvind Narayanan",
      title: "Leakage and the reproducibility crisis in machine-learning-based science",
      source: "Patterns 4(9), 100804",
      year: 2023,
      url: "https://arxiv.org/abs/2207.07048",
      note: "A survey of 17 research fields where data leakage led to overly hopeful results. Its list of leakage types includes preprocessing the training and test data together, the mistake this lesson warns about when encoding statistics are fitted on the holdout.",
    },
    {
      authors: "Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, et al.",
      title: "Do ImageNet Classifiers Generalize to ImageNet?",
      source: "Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97, 5389–5400",
      year: 2019,
      url: "https://proceedings.mlr.press/v97/recht19a.html",
      note: "The authors built new test sets for CIFAR-10 and ImageNet by following the original collection steps, and accuracy still fell by 3 to 15 percent. They trace the drop to slightly harder images rather than test-set reuse, showing how a small change in the data can lower a score that looked settled.",
    },
    {
      authors: "Jie Lu, Anjin Liu, Fan Dong, et al.",
      title: "Learning under Concept Drift: A Review",
      source: "IEEE Transactions on Knowledge and Data Engineering 31(12), 2346–2363",
      year: 2019,
      url: "https://arxiv.org/abs/2004.05785",
      note: "A review of methods for data whose underlying distribution changes over time. It splits the work into detecting drift, understanding it, and adapting to it, which is the monitoring of new data that this lesson says a fixed holdout cannot replace.",
    },
  ],
  checkpoint: [
    {
      prompt: "Which stage of the loop is allowed to read the four holdout apartments?",
      options: [
        "The final reading, which scores the holdout after the fit without feeding back",
        "Optimization, which averages the loss over all sixteen rows before every gradient step",
        "The hypothesis class, which picks its family by checking holdout loss",
        "Encoding, which fits its color codes on every row so they stay consistent",
      ],
      answer: 0,
      explanation: "The loop is encode, split, choose H, fit, then read the holdout. The holdout never enters a gradient or a split, and choosing a family or fitting an encoding on it would leak it. Its loss is a reading taken after the fit, not a signal the optimizer is allowed to see.",
      objective: 0,
    },
    {
      prompt: "Algorithm stays on logistic and Loss on log-loss. You switch only Encoding from rooms + park to one-hot color. Train loss falls. What happens to holdout loss?",
      options: [
        "It falls too, because more features always describe new rows better",
        "It stays put, because the holdout rows never enter the fit",
        "It rises, because color tracks price only on the train slice",
        "It rises, because the extra one-hot slots are added to the loss",
      ],
      answer: 2,
      explanation: "Encoding decides what the fit can see even though the holdout never enters it. Here color happens to line up with price on the train rows only, so the extra slots let the fit lean on a cue that does not carry over, and holdout loss rises while train loss falls. The loss is a mean over rows, not a sum over features.",
      objective: 1,
    },
    {
      prompt: "With rooms + park and logistic selected, you switch only Loss from log-loss to MSE. Why can't you rank the two holdout numbers against each other?",
      options: [
        "Holdout loss does not depend on Loss, because the holdout never enters the fit",
        "MSE is always smaller than log-loss, so it would always look better",
        "Loss only changes the optimizer's step size, so the fit stays the same",
        "A new loss drives a different fit and also reads it on a different scale",
      ],
      answer: 3,
      explanation: "The loss is both the objective the fit drives down and the unit in which the result is read. Switching it changes the fitted parameters and the yardstick at once, so a log-loss number and an MSE number are not comparable. Comparing models needs one agreed reading, taken on the holdout.",
      objective: 1,
    },
    {
      prompt: "You fit only on the twelve training apartments, then raise tree depth until training loss is tiny. What is still true?",
      options: [
        "Holdout loss must also be tiny, because both numbers are read off the same fitted tree",
        "Holdout loss can rise as training loss falls, because extra splits isolate train rows",
        "The holdout entered the Gini or entropy split, so its loss cannot possibly rise",
      ],
      answer: 1,
      explanation: "Depth is capacity. Once color is in x, a deeper tree can isolate Rowan 9, an exception that lives only in train. Those extra cuts are not in the holdout, so train loss can fall while holdout loss rises. The holdout never enters a split or a gradient on this page.",
      objective: 2,
    },
    {
      prompt: "On rooms + park alone, you raise Tree depth from 1 to 5 and training loss stalls above zero. Why can't more depth memorize the train slice?",
      options: [
        "A tree cannot reach zero training loss on any encoding of these twelve apartments",
        "Two rows with different prices share one vector, so no split can separate them",
        "Depth is capped by the number of training rows, which is twelve, so five is too deep",
        "Rooms + park drops the label, so the tree has nothing to fit on this encoding",
      ],
      answer: 1,
      explanation: "A split can only separate rows whose feature vectors differ. Rowan 9 and Cedar 9 both read 3 rooms with a park, so under rooms + park they share a vector and carry different labels. Capacity cannot memorize what the encoding cannot tell apart, which is why encoding is a stage of its own.",
      objective: 2,
    },
    {
      prompt: "A random 4-row holdout is drawn from this table's 16 apartments, 8 high-priced and 8 low-priced. Why is a shuffled split safer than choosing the four rows by hand?",
      options: [
        "It makes holdout loss smaller, because mixed rows are always easier to predict than chosen ones",
        "It guarantees two high and two low rows in the holdout, so no further care is needed",
        "It makes a holdout that differs from train in an unintended way unlikely, and a seed repeats it",
        "It lets the fit read each holdout row once, which helps the model generalize to new rows",
      ],
      answer: 2,
      explanation: "A shuffle takes the choice out of your hands, so a systematic difference between train and holdout has to arise by chance. It does not guarantee a balanced holdout: of the 1,820 ways to draw four rows, 140 (7.7 percent) leave all four on one side of the label, 784 (43.1 percent) give the lab's two and two, and 896 (49.2 percent) give three and one. The seed makes the split repeatable.",
      objective: 0,
    },
    {
      prompt: "You choose Encoding, Algorithm, and Tree depth by watching holdout loss. What does that do to the holdout numbers as an estimate for new rows?",
      options: [
        "It leaves them unbiased, because the holdout rows never enter a gradient or a split",
        "It makes them optimistic, because the winning setting was picked on those four rows",
        "It makes them pessimistic, because every setting you rejected scored worse on those rows",
        "It changes nothing, because only the train rows decide what the holdout measures",
      ],
      answer: 1,
      explanation: "Choosing among settings by a slice turns that slice into a validation set: the winner is partly lucky on exactly those rows. The rows never entering a gradient is not enough, because your choice reads them. A real project keeps a third test slice, scored once, for the honest number. This lab has no test slice.",
      objective: 0,
    },
    {
      prompt: "You keep the fit and its train loss fixed and switch Holdout drawn from to a different process. What happens to the holdout reading?",
      options: [
        "It stays put, because the holdout rows never enter the fit and so cannot change its score",
        "It improves, because rows from another process add information the twelve train rows lack",
        "It stays put, because a straight boundary is robust to whatever rows it is shown",
        "It collapses, because the price rule changed while the fit still follows the old one",
      ],
      answer: 3,
      explanation: "The fit was driven down on twelve rows from one process. On logistic with rooms + park and log-loss the same-process holdout reads 100 percent accuracy and log-loss 0.31, while the different-process rows read 25 percent and 1.39, worse than the 0.69 of answering 0.5 for every row. Holdout rows not entering the fit is why the score is honest about the same process, not about a new one.",
      objective: 3,
    },
    {
      prompt: "A holdout drawn from the same process as the train rows scores 100 percent accuracy on its four rows. Which reading is most defensible?",
      options: [
        "Meaningless: a holdout score of 100 percent can only come from a leaked holdout",
        "Conclusive: a perfect holdout score proves the rule will hold on any rows it meets later",
        "Encouraging but coarse: each row is 25 points, and it cannot speak for a moved process",
        "Uninterpretable: a holdout is never allowed to score above the slice the fit was trained on",
      ],
      answer: 2,
      explanation: "Four rows give accuracy in steps of 25 points, so one lucky or unlucky row moves the reading a long way. And a holdout from the same process can only speak for rows like the train rows; the different-process rows in this lab show what happens when the rule moves. A perfect score is not proof, and it is not evidence of leakage either: the holdout never entered the fit here.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      return {
        ...initialState,
        ...parsed,
        stage: clampInt(parsed.stage, initialState.stage as number, 0, 4),
        encoding: asMember(parsed.encoding, ENCODINGS, initialState.encoding as string),
        algorithm: asMember(parsed.algorithm, ALGORITHMS, initialState.algorithm as string),
        loss: asMember(parsed.loss, LOSSES, initialState.loss as string),
        step: clampInt(parsed.step, initialState.step as number, 0, 48),
        depth: clampInt(parsed.depth, initialState.depth as number, 0, TREE_MAX_DEPTH),
        selected: clampInt(parsed.selected, initialState.selected as number, 0, 15),
        lastKnob: asMember(parsed.lastKnob, ["encoding", "algorithm", "loss", "depth", "holdout"], initialState.lastKnob as string),
        holdoutSource: asMember(parsed.holdoutSource, HOLDOUT_SOURCES, initialState.holdoutSource as string),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
