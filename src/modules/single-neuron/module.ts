import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {
  x1: 0.7,
  x2: 0.35,
  w1: 1.2,
  w2: -0.8,
  bias: -0.1,
  activation: "sigmoid",
  task: "off",
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;
const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/** Version 1 had the same keys minus `task`, and no bias control, so it hydrates as-is. */
export function hydrateNeuronState(value: string): ModuleState {
  let parsed: Record<string, unknown> = {};
  try {
    const raw = JSON.parse(value) as unknown;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) parsed = raw as Record<string, unknown>;
  } catch {
    return { ...initialState };
  }
  return {
    x1: clampNumber(parsed.x1, 0.7, 0, 1),
    x2: clampNumber(parsed.x2, 0.35, 0, 1),
    w1: clampNumber(parsed.w1, 1.2, -2, 2),
    w2: clampNumber(parsed.w2, -0.8, -2, 2),
    bias: clampNumber(parsed.bias, -0.1, -2, 2),
    activation: asMember(parsed.activation, ["sigmoid", "relu", "step"], "sigmoid"),
    task: asMember(parsed.task, ["off", "and", "or", "xor"], "off"),
  };
}

const definition: ModuleDefinition = {
  id: "module-02-single-neuron",
  slug: "single-neuron",
  title: "A single neuron",
  group: "foundations",
  order: 8,
  icon: "CircleDot",
  accent: "#0d9a94",
  prerequisites: ["module-01-tokens-embeddings"],
  estimatedMinutes: 12,
  steps: ["Set inputs", "Tune weights", "Choose an activation", "Read the boundary", "Try exclusive-or"],
  stepInstructions: [
    "Move Input x₁ and Input x₂. The probe crosses the plane, and the Weighted sum and Activation readouts update with it.",
    "Hold the inputs still and sweep Weight w₁ from −2 to 2: the line turns and the w arrow stays perpendicular to it. Then move Bias b: the line slides without turning.",
    "With w₁ 1.2, w₂ −0.8 and b −0.1, set Input x₁ to 0.75 and Input x₂ to 1.00 so z reads 0.000. Switch Activation between Sigmoid, ReLU, and Step and compare three outputs from one z.",
    "Set Weight w₁ 1.0, Weight w₂ −0.5 and Bias b −0.2, then double all three to 2.0, −1.0 and −0.4. The line stays put, the dashed z = ±1 lines move closer, z doubles, and Distance to boundary does not change.",
    "Set Corner task to AND and find weights that score 4 / 4. Then switch to XOR: no setting of the three sliders gets past 3 / 4.",
  ],
  stateVersion: 2,
  tagline:
    "See how a neuron combines inputs with learned weights, adds a bias, and passes the result through an activation function, and why its decision is always one straight line.",
  objectives: [
    "Compute a weighted sum",
    "Describe how activation functions shape outputs",
    "Relate the weights and bias to the position of the decision boundary",
  ],
  glossary: [
    {
      term: "Weight",
      definition:
        "A learned multiplier controlling how much one input contributes to the sum. Its sign says for or against; its size says how strongly.",
    },
    {
      term: "Bias",
      definition:
        "A learned constant added to the weighted sum. It slides the decision boundary without turning it, so the neuron can fire, or stay silent, when every input is 0.",
    },
    {
      term: "Weighted sum",
      definition:
        "The pre-activation value z = w·x + b. Every dense layer, attention projection, and MLP in this course is built from this one operation.",
    },
    {
      term: "Dot product",
      definition:
        "The sum of element-wise products of two vectors. It measures alignment, which is why it appears in both neurons and attention scores.",
    },
    {
      term: "Weight vector",
      definition:
        "The weights written as one arrow, w = (w₁, w₂). It is perpendicular to the decision boundary, points toward growing z, and its length sets how fast z grows away from the line.",
    },
    {
      term: "Activation",
      definition:
        "The scalar function applied to the weighted sum. It introduces the nonlinearity that makes stacked layers more expressive than one layer.",
    },
    {
      term: "Sigmoid",
      definition:
        "An activation squashing any input into the open range 0 to 1. Its slope peaks at 0.25 at z = 0 and falls below 0.05 once z is beyond about ±2.9.",
    },
    {
      term: "ReLU",
      definition:
        "An activation passing positive values unchanged and returning 0 otherwise. Its slope is 1 above zero and 0 below, so a unit that is negative for every input stops learning.",
    },
    {
      term: "Step function",
      definition:
        "An activation returning a hard 0 or 1. Its derivative is 0 everywhere except at the jump, where none exists, so gradient descent gets no signal from it.",
    },
    {
      term: "Decision boundary",
      definition:
        "The set of inputs where the weighted sum equals zero. For a single neuron it is always a straight line, plane, or hyperplane, perpendicular to the weight vector.",
    },
    {
      term: "Linear separability",
      definition:
        "The property that a single straight boundary can split two classes. Exclusive-or is the standard pattern that lacks it, which is why depth is needed.",
    },
    {
      term: "Saturation",
      definition:
        "The flat region of an activation where large changes in input barely change the output. It reads as a near-zero gradient during training.",
    },
    {
      term: "Parameter",
      definition:
        "Any number learned from data rather than supplied as input. In this lab only the two weights and the bias are parameters.",
    },
  ],
  references: [
    {
      authors: "Warren S. McCulloch and Walter Pitts",
      title: "A logical calculus of the ideas immanent in nervous activity",
      source: "The Bulletin of Mathematical Biophysics 5(4), 115–133",
      year: 1943,
      url: "https://doi.org/10.1007/BF02478259",
      note: "The first model of a neuron as an all-or-none unit, and a proof that nets of such units can compute logical rules. It is the ancestor of the Step activation and of the AND and OR corner tasks in this lab.",
    },
    {
      authors: "Frank Rosenblatt",
      title: "The perceptron: A probabilistic model for information storage and organization in the brain",
      source: "Psychological Review 65(6), 386–408",
      year: 1958,
      url: "https://www.semanticscholar.org/paper/5d11aad09f65431b5d3cb1d85328743c9e53ba96",
      note: "The paper that introduced the perceptron, a machine whose connection weights change with experience. The lesson's neuron with Activation set to Step is this kind of unit.",
    },
    {
      authors: "Stanford CS231n course staff",
      title: "Neural Networks Part 1: Setting up the Architecture",
      source: "CS231n: Deep Learning for Computer Vision, Stanford University course notes",
      year: 2026,
      url: "https://cs231n.github.io/neural-networks-1/",
      note: "Course notes that model one neuron as a dot product of inputs and weights, plus a bias, passed through an activation. The section on a single neuron as a linear classifier matches this lab's decision boundary, and the activation section covers sigmoid saturation and dying ReLUs.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, et al.",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online",
      year: 2023,
      url: "https://d2l.ai/chapter_multilayer-perceptrons/mlp.html",
      note: "Section 5.1 shows that an affine layer on top of an affine layer is still one affine map, which is why a nonlinear activation is needed. It then plots ReLU and sigmoid with their slopes, including the sigmoid slope's peak of 0.25 at zero.",
    },
    {
      authors: "Vinod Nair and Geoffrey E. Hinton",
      title: "Rectified Linear Units Improve Restricted Boltzmann Machines",
      source: "Proceedings of the 27th International Conference on Machine Learning (ICML 2010)",
      year: 2010,
      url: "https://icml.cc/Conferences/2010/papers/432.pdf",
      note: "An early paper on rectified linear units, the ReLU option in the Activation control. It finds that, unlike on-off units, they keep information about how strong an input is as it passes through layers.",
    },
    {
      authors: "Xavier Glorot and Yoshua Bengio",
      title: "Understanding the difficulty of training deep feedforward neural networks",
      source: "Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2010), PMLR 9, 249–256",
      year: 2010,
      url: "https://proceedings.mlr.press/v9/glorot10a.html",
      note: "Shows that sigmoid units in a deep network can be pushed into saturation, where learning can stall on long plateaus. It backs the lesson's point about flat slopes and proposes a starting weight scale that helps.",
    },
    {
      authors: "Lu Lu, Yeonjong Shin, Yanhui Su, et al.",
      title: "Dying ReLU and Initialization: Theory and Numerical Examples",
      source: "Communications in Computational Physics 28(5), 1671–1706",
      year: 2020,
      url: "https://arxiv.org/abs/1903.06733",
      note: "Defines a dying ReLU as a unit that outputs 0 for every input, the failure described in Where it breaks. It shows how the choice of starting weights and biases affects how often units die.",
    },
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Section 6.1, Example: Learning XOR, shows that a linear model cannot represent exclusive-or, the 3 / 4 ceiling in this lab's XOR corner task. It then solves XOR with one hidden layer and explains why that layer needs a nonlinear activation.",
    },
    {
      authors: "Marvin Minsky and Seymour A. Papert",
      title: "Perceptrons: An Introduction to Computational Geometry",
      source: "MIT Press (first published 1969; expanded edition reissued 2017)",
      year: 1969,
      url: "https://mitpress.mit.edu/9780262534772/perceptrons/",
      note: "The book-length mathematical analysis of what perceptrons can and cannot compute, named in Going deeper. The next lab, Stacking neurons, answers the limits it proves for one layer of units.",
    },
    {
      authors: "Dan Hendrycks and Kevin Gimpel",
      title: "Gaussian Error Linear Units (GELUs)",
      source: "arXiv preprint arXiv:1606.08415",
      year: 2016,
      url: "https://arxiv.org/abs/1606.08415",
      note: "Introduces GELU, one of the smooth relatives of ReLU named in Going deeper. Instead of gating an input by its sign like ReLU, it weights the input by a smooth curve of its value.",
    },
    {
      authors: "Stefan Elfwing, Eiji Uchibe, and Kenji Doya",
      title: "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning",
      source: "Neural Networks 107, 3–11",
      year: 2018,
      url: "https://arxiv.org/abs/1702.03118",
      note: "Introduces SiLU, the other smooth ReLU relative in Going deeper. Its output is the input multiplied by the sigmoid of that input, so it combines two of the curves from this lab.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al.",
      title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",
      source: "Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015), 1026–1034",
      year: 2015,
      url: "https://arxiv.org/abs/1502.01852",
      note: "Works out a starting weight scale designed for ReLU units, so very deep networks can train from scratch. It backs the lesson's point that initialization scale keeps z in the responsive part of the activation.",
    },
    {
      authors: "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton",
      title: "Layer Normalization",
      source: "arXiv preprint arXiv:1607.06450",
      year: 2016,
      url: "https://arxiv.org/abs/1607.06450",
      note: "Normalizes the summed inputs, the z values, of all neurons in a layer before the activation. This is the kind of normalization layer the lesson says keeps z in a useful range as networks get deeper.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Set Weight w₁ to 1.2, Weight w₂ to −0.8 and Bias b to −0.1, then Input x₁ to 1.00 and Input x₂ to 0.50. What does Weighted sum z read?",
      options: [
        "1.1: only the first product counts, because a negative weight cancels its input",
        "0.8: both products are added, with the bias left out of the sum",
        "0.7: both products are added, then the bias is added once",
        "1.5: the negative weight is counted as positive, then the bias is added",
      ],
      answer: 2,
      explanation:
        "z = x₁w₁ + x₂w₂ + b = 1.00 × 1.2 + 0.50 × (−0.8) + (−0.1) = 1.2 − 0.4 − 0.1 = 0.7. A negative weight subtracts its input's contribution instead of removing it, and the bias is added once after the products.",
      objective: 0,
    },
    {
      prompt:
        "Compare z = 3 with z = 6 on the Activation curve. Which description of the three outputs is right?",
      options: [
        "Sigmoid barely moves because it has flattened near 1, ReLU doubles, and Step stays at 1",
        "All three roughly double, because a bigger weighted sum always gives a proportionally bigger output",
        "Sigmoid roughly doubles while ReLU stays flat, because ReLU only reacts to values near zero",
        "Step climbs smoothly from 0.5 toward 1, while sigmoid jumps straight to 1 once z passes zero",
      ],
      answer: 0,
      explanation:
        "Sigmoid is already close to 1 at z = 3 and flattens further, so doubling z hardly changes it. ReLU returns z itself on the positive side, so its output doubles with z. Step is 1 for every positive z. The three functions disagree most near z = 0 and in how they treat large values.",
      objective: 1,
    },
    {
      prompt:
        "A unit's Slope f′(z) reads about 0 for every training input, either a saturated sigmoid or a ReLU with z below zero. What does that do to learning?",
      options: [
        "Its weights get larger corrections to make up for the flat part of the curve, so it learns faster than its neighbours",
        "Almost no gradient flows through it, so its weights barely change and it learns slowly or not at all",
        "Nothing: slope is only a display value, and the weights are updated from the loss alone",
      ],
      answer: 1,
      explanation:
        "Training updates a weight by multiplying the loss's slope by the activation's slope at that unit. When the activation's slope is about 0 the whole product is about 0, so the unit's weights stop moving. That is the cost of saturation and of a dead ReLU, and why a step function, with slope 0 almost everywhere, cannot be trained by gradient descent.",
      objective: 1,
    },
    {
      prompt: "You double w₁, w₂ and the bias together. What happens?",
      options: [
        "The boundary moves closer to the origin, as it does when only the weights are doubled",
        "The boundary moves twice as far from the origin, because every one of the three parameters doubled",
        "Nothing changes at all, because only the ratio of the weights matters",
        "The boundary stays where it is, and off the line the sigmoid output moves further from 0.5",
      ],
      answer: 3,
      explanation:
        "Doubling every parameter doubles z at every input. Its sign, and so the line z = 0, is unchanged, but |z| grows, so σ(z) moves toward 0 or 1. Doubling only the weights would move the line, because b would no longer scale with them.",
      objective: 2,
    },
    {
      prompt:
        "With Corner task on XOR the best score is 3 / 4. Would Weight and Bias sliders that reached beyond ±2 get 4 / 4?",
      options: [
        "No: one neuron draws one straight line, and no line has both 1-corners on one side and both 0-corners on the other",
        "Yes: larger weights would make the line steeper, so it could squeeze between the diagonal corners",
        "Yes: a large enough bias would slide the line to a spot that separates both diagonal pairs of corners",
      ],
      answer: 0,
      explanation:
        "The boundary z = 0 is a single straight line. XOR needs the two 1-corners on one side and the two 0-corners on the other, but each pair sits on a diagonal and the two diagonals cross, so any straight line leaves at least one corner on the wrong side. Scaling the weights does not move the line and a bias only slides it, so a wider slider range cannot help; XOR needs a second neuron.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateNeuronState,
};

export default definition;
