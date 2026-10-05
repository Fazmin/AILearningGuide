import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DATASETS, MAX_DEPTH, MAX_EPOCHS, MAX_WIDTH, MIN_DEPTH, MIN_WIDTH } from "./mlp";

const initialState: ModuleState = {
  dataset: "xor",
  activation: "tanh",
  depth: 1,
  width: 2,
  epochs: 150,
  seed: 2,
  probeX: 0.6,
  probeY: -0.6,
  noise: "on",
  heldCurve: "accuracy",
};

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};
const clampNumber = (value: unknown, fallback: number, low: number, high: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;
const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/**
 * Version 1 stored depth 1–5, width 2–7 and epochs 0–80 for a closed-form loss curve. Those keys keep
 * their meaning, so they are clamped into the trainable ranges. Version 2 had no held-out card and only
 * the XOR and Ring datasets; its payloads gain `noise` and `heldCurve` from the defaults.
 */
export function hydrateStackingState(value: string): ModuleState {
  let parsed: Record<string, unknown> = {};
  try {
    const raw = JSON.parse(value) as unknown;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) parsed = raw as Record<string, unknown>;
  } catch {
    return { ...initialState };
  }
  return {
    dataset: asMember(parsed.dataset, DATASETS, "xor"),
    activation: asMember(parsed.activation, ["tanh", "relu", "linear"], "tanh"),
    depth: clampInt(parsed.depth, 1, MIN_DEPTH, MAX_DEPTH),
    width: clampInt(parsed.width, 2, MIN_WIDTH, MAX_WIDTH),
    epochs: clampInt(parsed.epochs, 150, 0, MAX_EPOCHS),
    seed: clampInt(parsed.seed, 2, 1, 999),
    probeX: clampNumber(parsed.probeX, 0.6, -1, 1),
    probeY: clampNumber(parsed.probeY, -0.6, -1, 1),
    noise: asMember(parsed.noise, ["off", "on"], "on"),
    heldCurve: asMember(parsed.heldCurve, ["accuracy", "loss"], "accuracy"),
  };
}

const definition: ModuleDefinition = {
  id: "module-03-stacking-neurons",
  slug: "stacking-neurons",
  title: "Stacking neurons",
  group: "foundations",
  order: 9,
  icon: "Network",
  accent: "#a75bba",
  prerequisites: ["module-02-single-neuron"],
  estimatedMinutes: 20,
  steps: [
    "Build a layer",
    "Add depth",
    "Train the network",
    "Inspect representations",
    "Remove the nonlinearity",
    "Need more width",
    "Match the size",
    "Hold data out",
  ],
  stepInstructions: [
    "With Hidden layers at 1, set Neurons per layer to 1: accuracy stalls at 75%. Raise it to 2 and watch two dashed lines, h1 and h2, carve the Decision surface into XOR's pattern.",
    "Raise Hidden layers to 2 and compare the tiles in Network builder: first-layer units have one straight edge, second-layer units combine them into bent ones.",
    "Drag Training epochs to 0, then press Train 50 epochs and watch the loss curve fall and the boundary form. Press New initialization a few times: some seeds get stuck.",
    "Move Probe x₁ and Probe x₂ in Network builder and read each unit's value under its tile. The output unit adds those values with its own weights.",
    "Set Hidden activation to none (linear). Try any Hidden layers and Neurons per layer: accuracy stays near 50% and the loss at 0.693, because the stack is one linear map.",
    "Set Dataset to Ring, Hidden activation to tanh and Hidden layers to 1. Two neurons plateau near 87%; three close the ring and reach 100%.",
    "Set Dataset to Checkerboard, Hidden activation to tanh and Training epochs to 300. In Depth against width both networks have 85 parameters: one layer of 21 units stays near chance, 56.3% at seed 2, while two layers of 7 reach 87.5%. Then switch Hidden activation to ReLU and press New initialization to see where the comparison turns around.",
    "In Held-out data, drag Training epochs to 300 and switch Training labels between Clean and 20% flipped. With flipped labels the large network trains to 96.7% yet scores 79.2% on held-out points, while the narrow network stays near 90% on both.",
  ],
  stateVersion: 3,
  tagline:
    "Train a real two-input network and watch a hidden layer do what one neuron cannot: solve XOR by combining straight boundaries into a bent one, see where depth beats width at the same size, and see extra capacity memorize noisy labels.",
  objectives: [
    "Relate width to model capacity, and show with real training a task where depth beats width at the same parameter count",
    "Explain hidden representations",
    "Show why a nonlinearity is required for depth to add anything",
    "Tell memorizing noisy labels from learning the pattern by comparing training and held-out accuracy",
  ],
  glossary: [
    {
      term: "Hidden layer",
      definition:
        "A set of neurons reading the same inputs with different weights, written h = f(Wx + b). Its output is consumed by the next layer, not by the user.",
    },
    {
      term: "Representation",
      definition:
        "The vector a layer hands forward: a rewritten version of the input that later layers find easier to use. It is a means, not an answer.",
    },
    {
      term: "Width and depth",
      definition:
        "Width is the number of units in a layer; depth is the number of stacked hidden layers. Width adds parallel detectors at one level and drives most of the parameter count; depth adds stages of composition, letting later layers build on features earlier layers invented. Which one wins at equal size depends on the task, the activation, and the training budget.",
    },
    {
      term: "Nonlinearity",
      definition:
        "The activation placed between layers. Without it, any stack of linear layers collapses algebraically into a single linear map.",
    },
    {
      term: "Universal approximation",
      definition:
        "The result that one sufficiently wide hidden layer with a non-polynomial activation can approximate any continuous function on a bounded domain. It proves existence, not trainability or efficiency.",
    },
    {
      term: "Feature hierarchy",
      definition:
        "The observed progression from simple to composite features across depth, such as edges to object parts in vision or surface form to syntax in language.",
    },
    {
      term: "Distributed representation",
      definition:
        "An encoding spread across many units rather than stored one concept per unit. It is why single hidden units rarely have a clean human label.",
    },
    {
      term: "Capacity",
      definition:
        "How complex a function a network could express, set mainly by depth and width. High capacity permits good fits and also permits memorizing noise.",
    },
    {
      term: "Parameter count",
      definition:
        "The total learned numbers in the model. Each layer contributes width-times-input weights plus one bias per unit, so adjacent wide layers dominate the total. Different shapes can share a count: one layer of 21 units and two layers of 7 both have 85.",
    },
    {
      term: "Forward pass and backpropagation",
      definition:
        "The forward pass is one left-to-right evaluation that turns an input into a prediction, each layer consuming the previous layer's output. Backpropagation runs back through the same layers with the chain rule to get the gradient for every weight; here it is exact, and it is taught in the lab Backpropagation.",
    },
    {
      term: "Binary cross-entropy",
      definition:
        "The loss for a yes-or-no output: the average of minus the log of the probability the network gave the correct label, so a confident wrong answer costs a lot. Answering 0.5 everywhere scores ln 2 = 0.693. It is taught in the lab Loss & gradient descent.",
    },
    {
      term: "Gradient descent, momentum, and epochs",
      definition:
        "Training repeatedly nudges every weight against the gradient of the loss, and momentum adds a running velocity so steps build along a steady slope. This lab uses a fixed learning rate of 0.5 with momentum 0.9 over the whole training set, so one epoch, a complete pass of the optimizer over the training set, is one full-batch step. Both ideas are taught in the lab Loss & gradient descent.",
    },
    {
      term: "Initialization",
      definition:
        "The random starting weights. Different seeds can end in different solutions, and with only two hidden units some seeds get stuck below 100% on XOR. The scale of the starting weights also sets how large gradients are at the start, which the Backpropagation lab's Gradient flow card measures.",
    },
    {
      term: "Overfitting and held-out data",
      definition:
        "Held-out data are points the network never trains on. Overfitting is fitting the training points, including any wrong labels among them, better than the pattern behind them, so accuracy on held-out points falls behind training accuracy. Label noise, wrong labels in the training set, gives a high-capacity network something to memorize that will not repeat.",
    },
  ],
  references: [
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Chapter 6 opens by solving XOR with a hidden layer of two units, the same problem as this lab's default, and shows how the hidden layer moves the points so one straight line separates them. Section 6.4.1 covers universal approximation and why depth can need fewer units.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 3 builds a shallow network one hidden unit at a time and states the universal approximation theorem. Chapter 4 composes networks into deep ones and compares shallow and deep networks at the same parameter count, the question behind the Depth against width card.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, et al.",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online",
      year: 2023,
      url: "https://d2l.ai/chapter_multilayer-perceptrons/mlp.html",
      note: "Section 5.1 adds a hidden layer to a linear model, shows that an affine map of an affine map is still affine, and so explains why an activation is needed between layers. It also covers the ReLU, sigmoid, and tanh activations, the choices in Hidden activation.",
    },
    {
      authors: "George Cybenko",
      title: "Approximation by superpositions of a sigmoidal function",
      source: "Mathematics of Control, Signals, and Systems 2(4), 303–314",
      year: 1989,
      url: "https://doi.org/10.1007/BF02551274",
      note: "An early proof that one hidden layer of sigmoid units, if wide enough, can approximate any continuous function on a bounded region. It is the universal approximation result in this lesson's glossary: it shows good weights exist, not how to find them.",
    },
    {
      authors: "Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, et al.",
      title: "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function",
      source: "Neural Networks 6(6), 861–867",
      year: 1993,
      url: "https://www.semanticscholar.org/paper/990504fc9e5a1f3619ace4fa7f5bf667069018b1",
      note: "Extends universal approximation beyond sigmoids: one hidden layer is enough for a broad class of activations, as long as the activation is not a polynomial. This is the non-polynomial condition in the glossary, and it covers the lab's tanh and ReLU.",
    },
    {
      authors: "Ronen Eldan and Ohad Shamir",
      title: "The Power of Depth for Feedforward Neural Networks",
      source: "Proceedings of the 29th Conference on Learning Theory (COLT 2016), PMLR 49, 907–940",
      year: 2016,
      url: "https://proceedings.mlr.press/v49/eldan16.html",
      note: "Proves there is a function a small network with two hidden layers can compute that one hidden layer can only approximate if it is exponentially wide. It is a theory version of the one-layer against two-layer comparison on the Depth against width card.",
    },
    {
      authors: "Matus Telgarsky",
      title: "Benefits of depth in neural networks",
      source: "Proceedings of the 29th Conference on Learning Theory (COLT 2016), PMLR 49, 1517–1539",
      year: 2016,
      url: "https://proceedings.mlr.press/v49/telgarsky16.html",
      note: "Shows that a deep network with only a few units per layer can compute functions that a much shallower network matches only with exponentially many units. It backs the Going deeper claim that for many functions depth needs far fewer units than width.",
    },
    {
      authors: "Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al.",
      title: "Understanding deep learning requires rethinking generalization",
      source: "International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1611.03530",
      note: "Shows that large image networks can fit training labels perfectly even when some or all are randomly corrupted, while test error rises with the noise. The Held-out data card shows the same effect in miniature with 20% flipped labels.",
    },
    {
      authors: "Xavier Glorot and Yoshua Bengio",
      title: "Understanding the difficulty of training deep feedforward neural networks",
      source: "Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS 2010), PMLR 9, 249–256",
      year: 2010,
      url: "https://proceedings.mlr.press/v9/glorot10a.html",
      note: "Studies why plain gradient descent from random starting weights struggles with deep networks, including how sigmoid units saturate and how gradients change across layers. It proposes a scale for the starting weights, the Initialization term in this lesson.",
    },
    {
      authors: "Sergey Ioffe and Christian Szegedy",
      title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
      source: "Proceedings of the 32nd International Conference on Machine Learning (ICML 2015), PMLR 37, 448–456",
      year: 2015,
      url: "https://proceedings.mlr.press/v37/ioffe15.html",
      note: "Introduces normalizing each layer's inputs inside the network, which allows higher learning rates and less care over starting weights. It is one of the two fixes named where the lesson says plain deep stacks were hard to train.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al.",
      title: "Deep Residual Learning for Image Recognition",
      source: "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), 770–778",
      year: 2016,
      url: "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
      note: "Adds residual connections, which let each layer learn a change to its input, and trains networks over 100 layers deep. It is the other fix named where the lesson says plain deep stacks were hard to train.",
    },
    {
      authors: "Nelson Elhage, Tristan Hume, Catherine Olsson, et al.",
      title: "Toy Models of Superposition",
      source: "Transformer Circuits Thread (Anthropic)",
      year: 2022,
      url: "https://transformer-circuits.pub/2022/toy_model/index.html",
      note: "Uses small networks to show how a layer can store more features than it has units, so single neurons respond to several unrelated features. It explains why a hidden unit rarely has one clean meaning, the distributed representation point in this lesson.",
    },
    {
      authors: "Matthew D. Zeiler and Rob Fergus",
      title: "Visualizing and Understanding Convolutional Networks",
      source: "Computer Vision – ECCV 2014 (European Conference on Computer Vision), 818–833",
      year: 2014,
      url: "https://arxiv.org/abs/1311.2901",
      note: "Shows what each layer of an image network responds to: corners and edge combinations in early layers, textures next, then object parts such as dog faces, then whole objects. It is the vision example of a feature hierarchy in this lesson.",
    },
    {
      authors: "Ian Tenney, Dipanjan Das, and Ellie Pavlick",
      title: "BERT Rediscovers the Classical NLP Pipeline",
      source: "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), 4593–4601",
      year: 2019,
      url: "https://aclanthology.org/P19-1452/",
      note: "Measures where a language model stores different kinds of information and finds them in order across its layers: parts of speech and grammar earlier, meaning-level tasks later. It is the language example of a feature hierarchy in this lesson.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Set Dataset to Ring with one hidden layer of 2 neurons, then raise Neurons per layer to 3. What changes, and why?",
      options: [
        "Accuracy rises only slightly, because extra neurons in the same layer mostly repeat one another's cut",
        "Accuracy reaches 100%, because three straight cuts can enclose the inner disc and two can only carve a wedge",
        "Accuracy stays the same, because only an extra layer can add a new kind of boundary",
        "Accuracy falls, because more neurons mean more weights to tune on the same few points",
      ],
      answer: 1,
      explanation:
        "Each first-layer neuron contributes one straight line, and the output unit combines them. Two lines can bound a wedge but not a closed region, while three can wall off the inner disc, so the number of units limits what a single hidden layer can express. Width adds more parallel detectors at one level.",
      objective: 0,
    },
    {
      prompt:
        "With Hidden layers at 2, the second-layer tiles in Network builder have bent edges while first-layer tiles have one straight edge. Why?",
      options: [
        "Later layers are trained for more epochs, and the extra training curves a unit's boundary away from a line",
        "Every extra input to a unit adds one bend, so units that read more inputs end up bending more",
        "Deeper units use a different activation function, and the shape of that function bends the edge",
        "First-layer units read x₁ and x₂ directly; second-layer units read the curved outputs of first-layer units",
      ],
      answer: 3,
      explanation:
        "A first-layer unit is a weighted sum of x₁ and x₂ followed by an activation, so its zero line is straight. A second-layer unit takes a weighted sum of those units' outputs, which are already bent responses over the square, so its zero line bends. This is what depth adds here: composition of earlier units, not extra straight cuts.",
      objective: 1,
    },
    {
      prompt:
        "At the defaults, set Probe x₁ and Probe x₂ to (0.6, 0.6), then to (−0.6, −0.6): two far-apart points with the same label. What do h1 and h2 do?",
      options: [
        "They swap signs, so the output unit has to learn a separate rule for each corner",
        "They change by as much as the inputs did, so the output unit sees two distant clusters to handle",
        "Both points land near the same values, so the output unit can treat them alike with one rule",
        "They both drop to zero, because these corners lie on the first-layer lines",
      ],
      answer: 2,
      explanation:
        "The trained hidden layer rewrites the input: both label-0 clusters land near the same corner of the (h1, h2) space, while the two label-1 clusters land in other corners. In that rewritten space one straight line separates the classes, and that line is the output unit. A hidden layer helps because it moves points into a layout the next layer finds easy, not because it labels them.",
      objective: 1,
    },
    {
      prompt:
        "Set Hidden activation to none (linear) and train XOR with 3 hidden layers of 6 neurons. What happens, and why?",
      options: [
        "Accuracy stays at chance and Training loss stays at 0.693, because the stacked layers compose into one linear map",
        "Accuracy still reaches 100%, only more slowly, because 18 neurons give far more capacity than 2",
        "Training blows up, because every extra linear layer multiplies the signal until the loss grows without any bound",
        "Only depth is wasted: with 6 neurons per layer the extra width still carves enough cuts to reach 100% here",
      ],
      answer: 0,
      explanation:
        "W₂(W₁x + b₁) + b₂ is still one affine map, so a linear stack draws one straight boundary however deep or wide it is. The loss stays at 0.693 = ln 2, the value for answering 0.5 for every point. The activation is what lets one layer's output be a bent function of the input, so width and depth both add nothing without it.",
      objective: 2,
    },
    {
      prompt:
        "Set Dataset to Checkerboard, Hidden activation to tanh, and Training epochs to 300. In Depth against width the two networks have the same number of parameters. What do you see?",
      options: [
        "Two layers of 7 fit most of the board while one layer of 21 stays near chance, with the same budget",
        "Both fit about equally well, because any two networks with the same parameter count have the same capacity",
        "One layer of 21 fits better, because a single wide layer holds more detectors than two narrow layers can",
        "Neither learns anything, because a checkerboard needs far more than 85 parameters for any network to fit it",
      ],
      answer: 0,
      explanation:
        "Both networks have 85 parameters. At seed 2 the one-layer network ends at 56.3% and the two-layer network at 87.5%; across seeds 1 to 6 the first never passes 58% and the second never falls below 81%. The lab cannot say whether the wide layer could ever fit the board, only that in 300 epochs of this optimizer it does not, while the stacked layers do.",
      objective: 0,
    },
    {
      prompt:
        "Which claim about the checkerboard result does the lab itself support?",
      options: [
        "With tanh and 300 epochs two layers beat one wide layer of equal size, but with ReLU the wide layer ties or wins",
        "Depth beats width at any equal size, because stacking layers always adds more than adding units to one layer does",
        "A single layer of 21 tanh units cannot represent a checkerboard, so no amount of training could ever make it fit",
        "Depth wins on every dataset in the lab, because XOR and the ring also pull the two shapes apart by accuracy here",
      ],
      answer: 0,
      explanation:
        "On XOR and the ring both shapes reach 100% on every seed, so there is no depth advantage there. On the checkerboard with tanh the stacked network wins, but switch Hidden activation to ReLU and the one-layer network fits at 86% to 100% across seeds 1 to 6. The advantage belongs to this task, activation, and training budget. It is not a law, and the lab does not show that a wide layer is unable to represent the board.",
      objective: 0,
    },
    {
      prompt:
        "In Held-out data with Training labels at 20% flipped and Training epochs at 300, the large network scores far higher on the training points than on the held-out points. What does that gap show?",
      options: [
        "It has fit some flipped labels, which do not repeat on new points, so training accuracy overstates what it learned",
        "The held-out points are harder than the training points, so every network scores lower on them whatever it learned",
        "It learned the ring better than the narrow network, because more units find the pattern in noisy labels more reliably",
        "Training corrected the flipped labels, so the training points became easier than the held-out points for the network",
      ],
      answer: 0,
      explanation:
        "A label that is wrong in training tells the network nothing about new points. The large network has the capacity to bend around those points, so its training accuracy climbs while its held-out accuracy and loss get worse; at seeds 1 to 6 the gap is at least 15 points, against 7 or fewer for the narrow network. Training accuracy alone would have hidden this, which is what the held-out points are for.",
      objective: 3,
    },
    {
      prompt:
        "Switch Training labels to Clean. What happens to the gap between the large network's training and held-out accuracy at epoch 300?",
      options: [
        "It nearly vanishes, because with clean labels everything the network fits also holds on new ring points",
        "It gets larger, because a bigger network always memorizes its training points whatever the labels say about them",
        "It stays the same, because the gap comes from the network's size and not from the labels it trains on",
        "It reverses, because clean labels make the held-out points easier than the training points it sees",
      ],
      answer: 0,
      explanation:
        "With clean labels the large network reaches 100% on both the training and the held-out ring points at seeds 1 to 6, so the gap is zero. Capacity alone does not overfit: it needs something in the training set that will not repeat, such as wrong labels. That is why the gap grows only once 12 of the 60 labels are flipped.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateStackingState,
};

export default definition;
