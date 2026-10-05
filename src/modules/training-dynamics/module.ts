import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { CHECKPOINTS, DEGREE_MAX, DEGREE_MIN, WEIGHT_DECAYS } from "./sim";

const initialState: ModuleState = {
  degree: 5,
  epochIndex: CHECKPOINTS.indexOf(100),
  learningRateIndex: 3,
  weightDecayIndex: 0,
  batchSize: 16,
  seed: 1,
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const numberOr = (value: unknown, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

/**
 * Version 1 drew formula curves from three 0–100 sliders (`epoch`, `capacity`,
 * `regularization`). Version 2 trains a real model, so map each slider onto the
 * nearest control that plays the same role and drop the old keys.
 */
function migrate(parsed: ModuleState): ModuleState {
  const isVersion1 = "capacity" in parsed || "regularization" in parsed || "epoch" in parsed;
  if (!isVersion1) return { ...initialState, ...parsed };
  const { capacity, regularization, epoch, ...rest } = parsed;
  const lastCheckpoint = CHECKPOINTS.length - 1;
  return {
    ...initialState,
    ...rest,
    degree: Math.round(
      clamp(DEGREE_MIN + ((numberOr(capacity, 62) - 10) / 90) * (DEGREE_MAX - DEGREE_MIN), DEGREE_MIN, DEGREE_MAX),
    ),
    weightDecayIndex: Math.round(clamp(numberOr(regularization, 0) / 100, 0, 1) * (WEIGHT_DECAYS.length - 1)),
    epochIndex: Math.round(clamp(numberOr(epoch, 35) / 100, 0, 1) * lastCheckpoint),
  };
}

const definition: ModuleDefinition = {
  id: "module-06-training-dynamics",
  slug: "training-dynamics",
  title: "Training dynamics",
  group: "foundations",
  order: 12,
  icon: "ChartNoAxesCombined",
  accent: "#2185a6",
  prerequisites: ["module-05-backpropagation"],
  estimatedMinutes: 13,
  steps: ["Split the data","Watch the curves","Trigger overfitting","Regularize"],
  stepInstructions: [
    "Read the split strip on Training dynamics controls, then find the filled training dots and hollow validation dots on Fitted function.",
    "Drag Epoch from 0 to 10,000 and compare training and validation loss at the marker; note the epoch of the green lowest-validation ring.",
    "Set Polynomial degree to 11 and drag Epoch to 10,000: validation climbs from 0.098 to 8.63 while training loss keeps falling.",
    "At degree 11, raise Weight decay to 0.03, then set it back to 0 and press Stop at lowest validation, comparing validation loss each time.",
  ],
  stateVersion: 2,
  tagline: "Train a real model in the browser and watch training and validation loss part ways, then see how capacity, weight decay, learning rate, batch size, and early stopping change the outcome.",
  objectives: ["Distinguish training and validation performance","Recognize and respond to overfitting","Predict how capacity, weight decay, learning rate, and batch size change a training run"],
  glossary: [
  {
    "term": "Generalization",
    "definition": "Performance on relevant examples the model did not train on. It is the only evidence that a real pattern was learned rather than the sample memorized."
  },
  {
    "term": "Model capacity",
    "definition": "How flexible a model family is: how many different functions it can fit. Here it is the polynomial degree; in networks it grows with parameter count. More capacity can lower training loss without lowering held-out loss."
  },
  {
    "term": "Overfitting",
    "definition": "Training loss still falling while held-out loss rises. The extra fit describes noise and accidents of the training sample rather than structure that transfers."
  },
  {
    "term": "Underfitting",
    "definition": "Both losses staying high because the model has not captured structure plainly present in the data, either because it is too simple or because training has not gone far enough."
  },
  {
    "term": "Training set",
    "definition": "The examples the optimizer directly fits. Loss measured here is the cheapest number to improve and the least informative on its own."
  },
  {
    "term": "Validation set",
    "definition": "Held-out examples used to choose capacity, learning rate, and stopping point. Because decisions are made from it, its estimate becomes mildly optimistic."
  },
  {
    "term": "Test set",
    "definition": "Held-out examples touched once, at the end. It gives an unbiased estimate only until you start tuning against it."
  },
  {
    "term": "Generalization gap",
    "definition": "Held-out loss minus training loss. A steady gap is normal, especially on small data; a gap that keeps widening as training continues is the warning sign."
  },
  {
    "term": "Early stopping",
    "definition": "Keeping the model from the epoch with the lowest validation loss instead of the last one. It regularizes by budget: fewer updates leave less time to fit noise."
  },
  {
    "term": "Regularization",
    "definition": "Any constraint, penalty, or noise source that discourages brittle fitting. Weight decay, dropout, data augmentation, and early stopping are all instances."
  },
  {
    "term": "Weight decay",
    "definition": "A penalty of lambda times the sum of squared weights that pulls every weight toward zero on every step, favouring smaller weights and smoother functions. AdamW applies it directly to the parameters rather than through the gradient."
  },
  {
    "term": "Dropout",
    "definition": "Randomly zeroing a fraction of activations during training so no unit can depend on a specific partner, which forces redundant paths."
  },
  {
    "term": "Distribution shift",
    "definition": "A mismatch between held-out data and real inputs. A model can generalize well within its split and still fail in deployment."
  },
  {
    "term": "Data leakage",
    "definition": "Overlap between training and evaluation data, including near-duplicates and benchmark text present in pretraining. It makes a healthy-looking curve measure nothing."
  },
  {
    "term": "Double descent",
    "definition": "The observed pattern where held-out error worsens as capacity approaches the point of fitting the training set exactly, then improves again well beyond it. A similar rise and fall can appear across epochs."
  }
],
  references: [
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Sections 5.2 and 5.3 cover capacity, overfitting and underfitting, and why settings are chosen on a validation set. Section 7.8 on early stopping shows that, for a simple bowl-shaped loss, stopping early holds back the weakly curved directions most, the reason this lesson gives for why the damage arrives late.",
    },
    {
      authors: "Varun Godbole, George E. Dahl, Justin Gilmer, et al.",
      title: "Deep Learning Tuning Playbook",
      source: "Google Research, GitHub repository",
      year: 2023,
      url: "https://github.com/google-research/tuning_playbook",
      note: "A practical guide from Google researchers. It advises saving checkpoints and picking the best one by validation score afterwards, the same move as Stop at lowest validation, explains how to spot problematic overfitting in training curves, and warns that changing the batch size means re-tuning the learning rate.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "Decoupled Weight Decay Regularization",
      source: "International Conference on Learning Representations (ICLR 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1711.05101",
      note: "The AdamW paper. It shows that adding a squared-weight penalty to the loss and shrinking the weights directly are the same for plain gradient descent but not for Adam, and proposes applying weight decay to the parameters directly, as the Weight decay glossary entry says.",
    },
    {
      authors: "Sam McCandlish, Jared Kaplan, Dario Amodei, et al.",
      title: "An Empirical Model of Large-Batch Training",
      source: "arXiv preprint arXiv:1812.06162",
      year: 2018,
      url: "https://arxiv.org/abs/1812.06162",
      note: "Compares the noise in a minibatch gradient with the size of the full gradient, the same comparison as the Minibatch noise and Full-batch |∇L| readouts, and finds the noise grows as the loss falls. It also shows that a large step with a small batch can become unstable, as the batch-4 run does.",
    },
    {
      authors: "Priya Goyal, Piotr Dollár, Ross Girshick, et al.",
      title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
      source: "arXiv preprint arXiv:1706.02677",
      year: 2017,
      url: "https://arxiv.org/abs/1706.02677",
      note: "Introduces the linear scaling rule: when the batch size grows by some factor, grow the learning rate by the same factor. It is a well-known example of the lesson's point that learning rate and batch size have to be chosen together.",
    },
    {
      authors: "Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger",
      title: "On Calibration of Modern Neural Networks",
      source: "Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70, 1321–1330",
      year: 2017,
      url: "https://proceedings.mlr.press/v70/guo17a.html",
      note: "Finds that modern networks are often overconfident, and that a network can overfit its log loss while its error rate keeps improving. This backs the lesson's warning that validation loss can rise from overconfidence while accuracy still improves.",
    },
    {
      authors: "Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al.",
      title: "Scaling Data-Constrained Language Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html",
      note: "Trains language models on the same text for many epochs. Up to about 4 epochs costs little, but with more repeats the gains fade and the models overfit, so the authors judge them by held-out loss rather than training loss, as this lesson does.",
    },
    {
      authors: "Leo Gao, John Schulman, and Jacob Hilton",
      title: "Scaling Laws for Reward Model Overoptimization",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 10835–10866",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/gao23h.html",
      note: "Measures what happens when a model is optimized hard against a learned reward model: the proxy score keeps rising while the true score first improves and then falls. This is the reward-model case of overfitting the lesson describes for language models.",
    },
    {
      authors: "Preetum Nakkiran, Gal Kaplun, Yamini Bansal, et al.",
      title: "Deep Double Descent: Where Bigger Models and More Data Hurt",
      source: "International Conference on Learning Representations (ICLR 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/1912.02292",
      note: "Shows double descent in modern deep networks: test error gets worse and then better as models grow, and the same rise and fall can happen as training runs for more epochs. It backs the Double descent glossary entry and the end of Where it breaks.",
    },
    {
      authors: "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, et al.",
      title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
      source: "Journal of Machine Learning Research 15(56), 1929–1958",
      year: 2014,
      url: "https://jmlr.org/papers/v15/srivastava14a.html",
      note: "The original dropout paper. It randomly drops units and their connections during training so units cannot rely too much on each other, the regularizer described in the Dropout glossary entry and Going deeper.",
    },
    {
      authors: "Connor Shorten and Taghi M. Khoshgoftaar",
      title: "A survey on Image Data Augmentation for Deep Learning",
      source: "Journal of Big Data 6, 60",
      year: 2019,
      url: "https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0",
      note: "A survey of ways to make extra training images from existing ones, starting with simple changes such as horizontal flipping, as a fix for overfitting on limited data. It backs the lesson's data augmentation example of flipping a photo of a cat.",
    },
    {
      authors: "Rafael Müller, Simon Kornblith, and Geoffrey E. Hinton",
      title: "When does label smoothing help?",
      source: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
      year: 2019,
      url: "https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html",
      note: "Explains label smoothing, which trains on targets slightly softer than 100 percent, and shows that it keeps a classifier from becoming overconfident and improves calibration. It backs the label smoothing item in Going deeper.",
    },
    {
      authors: "Pang Wei Koh, Shiori Sagawa, Henrik Marklund, et al.",
      title: "WILDS: A Benchmark of in-the-Wild Distribution Shifts",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 5637–5664",
      year: 2021,
      url: "https://proceedings.mlr.press/v139/koh21a.html",
      note: "Ten real datasets, such as tumor images from different hospitals and wildlife photos from different cameras, where models score well on held-out data from the same source but substantially worse on new sources. It backs the lesson's Distribution shift warning.",
    },
    {
      authors: "Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al.",
      title: "Deduplicating Training Data Makes Language Models Better",
      source: "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022), 8424–8445",
      year: 2022,
      url: "https://aclanthology.org/2022.acl-long.577/",
      note: "Finds many near-duplicate examples in common text datasets, including overlap that affects over 4 percent of some validation sets, and removes them. It backs the lesson's Data leakage warning about near-copies landing on both sides of the split.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At the default degree 5, validation loss sits a little above training loss and the Gap stays steady as Epoch increases. What does that indicate?",
      options: [
        "Overfitting, because validation loss above training loss always means the model has memorized its data",
        "A normal difference between fitting seen and unseen points on a small set; a widening gap is the warning",
        "Underfitting, because any gap at all means the model has not yet fit the training points properly",
        "A bug in the split, because properly held-out data should score exactly the same as the training data",
      ],
      answer: 1,
      explanation:
        "The optimizer fits the training dots directly and has never seen the validation dots, so on 16 points the validation loss is expected to sit somewhat higher. A steady gap is normal. The warning is a gap that keeps widening while training continues, which is what degree 11 shows.",
      objective: 0,
    },
    {
      prompt:
        "At degree 11 you press Stop at lowest validation and report that validation loss as your model's final score. What is wrong with that?",
      options: [
        "You picked the epoch using that same data, so the score is mildly optimistic; an untouched test set gives the honest one",
        "Nothing is wrong: validation points never train the model, so any number taken from them is unbiased however you use it",
        "Validation loss may only be read after the last epoch, so a number from an earlier checkpoint is not valid evidence",
        "Training loss is the honest estimate, because it is measured on the points the model actually fit",
      ],
      answer: 0,
      explanation:
        "The validation set picked the epoch, so the loss you report is the best of many looks, and selecting on it makes it mildly optimistic. A test set that no decision has touched, scored once at the end, gives the unbiased estimate. This lab draws no test set, but real projects keep one.",
      objective: 0,
    },
    {
      prompt:
        "At degree 11, validation loss bottoms out within the first few dozen epochs, then climbs for thousands of epochs while training loss keeps falling. Which model should you keep?",
      options: [
        "The final model, because its training loss is the lowest of any epoch and is still improving",
        "Either one, because both fit the training points well and the difference is only noise",
        "The early model, because held-out loss is the evidence of what transfers",
      ],
      answer: 2,
      explanation:
        "Training loss only measures fit to the 16 points the optimizer saw, and after the early minimum it improves by fitting their noise. Validation loss is scored on points the model never trained on, and it says the early model transfers best. Choosing the epoch by validation loss spends that set's independence, which is why a separate test set gives the final estimate.",
      objective: 1,
    },
    {
      prompt:
        "At degree 11, raising Weight decay to 0.03 and pressing Stop at lowest validation both bring validation loss down. What do the two fixes share?",
      options: [
        "Each lowers training loss further, and that extra fit is what pulls validation loss down",
        "Each gives the model more data to learn from, so the noise in the training points averages out and the curve smooths",
        "Each lowers the learning rate, so the model trains more precisely and picks up less noise",
        "Each limits how far the curve can bend toward noise, by penalizing big weights or by stopping early",
      ],
      answer: 3,
      explanation:
        "Weight decay penalizes large weights, which keeps the curve smoother. Early stopping spends fewer updates, so there is less time to fit noise. Both are regularization, and both leave training loss a little higher than the unconstrained run, not lower. Neither adds data or changes the learning rate.",
      objective: 1,
    },
    {
      prompt:
        "Compare Polynomial degree 5 with degree 11 at epoch 13 and again at epoch 10,000. What do you expect to see?",
      options: [
        "Degree 11 is better at both epochs, because more capacity always fits the true curve more closely and generalizes better",
        "Similar at epoch 13, but far worse for degree 11 at epoch 10,000: capacity and training time act together",
        "Degree 11 is worse at both epochs, because extra capacity always hurts held-out loss",
        "Identical loss at both epochs, because the 16 training points fix the result whatever the model size",
      ],
      answer: 1,
      explanation:
        "Gradient descent fits the smooth, strongly curved directions first and the wiggly ones last. Early on the extra degrees have contributed little, so the models perform alike. Over thousands of epochs degree 11 fits the noise in the 16 points and its validation loss climbs. Capacity, data, and training time act together, so neither always better nor always worse holds.",
      objective: 2,
    },
    {
      prompt:
        "At degree 11 you lower Learning rate by a factor of ten. How does the run change?",
      options: [
        "Much the same run, only slower: validation bottoms out about ten times later and still climbs afterwards",
        "Overfitting disappears, because steps that small are too fine to fit any of the noise in the training points",
        "The best validation loss gets much lower, because a smaller step settles into a finer minimum",
      ],
      answer: 0,
      explanation:
        "The path is almost the same, but each step is a tenth as long, so the same events arrive about ten times later in epochs: on the log axis the curves slide one decade to the right. The best validation loss is about the same and the run still overfits afterwards. A smaller rate buys time to stop, not immunity.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      return migrate(parsed as ModuleState);
    } catch {
      return { ...initialState };
    }
  }
};

export default definition;
