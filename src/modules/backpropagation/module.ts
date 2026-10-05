import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { FLOW_ACTIVATIONS, FLOW_INITS, MAX_FLOW_DEPTH, MIN_FLOW_DEPTH } from "./gradient-flow";

const initialState: ModuleState = {
  x: 0.8,
  weight: 0.6,
  target: 0.9,
  phase: 0,
  learningRate: 0.7,
  flowDepth: 6,
  flowActivation: "sigmoid",
  flowInit: "glorot",
  flowSkip: "off",
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number, integer = false) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const bounded = Math.min(high, Math.max(low, value));
  return integer ? Math.round(bounded) : bounded;
};
const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/**
 * Version 2 stored only the one-weight graph (x, weight, target, phase, learningRate) and spread whatever
 * it found. Those keys keep their meaning and are now clamped to their slider ranges; the four gradient-flow
 * keys arrive from the defaults.
 */
export function hydrateBackpropState(value: string): ModuleState {
  let parsed: Record<string, unknown> = {};
  try {
    const raw = JSON.parse(value) as unknown;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) parsed = raw as Record<string, unknown>;
  } catch {
    return { ...initialState };
  }
  return {
    x: clampNumber(parsed.x, initialState.x as number, 0.1, 1.5),
    weight: clampNumber(parsed.weight, initialState.weight as number, -3, 3),
    target: clampNumber(parsed.target, initialState.target as number, 0, 1),
    phase: clampNumber(parsed.phase, initialState.phase as number, 0, 5, true),
    learningRate: clampNumber(parsed.learningRate, initialState.learningRate as number, 0.05, 2),
    flowDepth: clampNumber(parsed.flowDepth, initialState.flowDepth as number, MIN_FLOW_DEPTH, MAX_FLOW_DEPTH, true),
    flowActivation: asMember(parsed.flowActivation, FLOW_ACTIVATIONS, "sigmoid"),
    flowInit: asMember(parsed.flowInit, FLOW_INITS, "glorot"),
    flowSkip: asMember(parsed.flowSkip, ["off", "on"], "off"),
  };
}

const definition: ModuleDefinition = {
  id: "module-05-backpropagation",
  slug: "backpropagation",
  title: "Backpropagation",
  group: "foundations",
  order: 11,
  icon: "GitBranch",
  accent: "#d04f78",
  prerequisites: ["module-04-loss-gradient-descent"],
  estimatedMinutes: 20,
  steps: [
    "Run the forward pass",
    "Measure the loss",
    "Trace the chain rule",
    "Update the weight",
    "Stack the layers",
    "Rescue the gradient",
  ],
  stepInstructions: [
    "Advance through Multiply and Activate, following values from the input and weight toward the prediction.",
    "Advance to Measure loss, then compare the loss node with the loss-versus-weight chart.",
    "Continue through Loss → output, Output → product, and Product → weight; on the tape, check that each gradient is the previous one times the edge's × factor.",
    "Choose a learning rate, apply the weight update, and verify that the new point has lower loss.",
    "In Gradient flow through depth, set Activation to sigmoid, Initial weight scale to Glorot, and Skip connections off, then raise Network depth from 1 to 12 and watch the Layer 1 bar fall by orders of magnitude while the last layer's stays put. Try tanh for comparison.",
    "At depth 12 with ReLU, step Initial weight scale from small to He to large and read Layer 1 norm. Then set Activation to sigmoid and turn Skip connections on, and compare Layer 1 ÷ last and Loss at the start.",
  ],
  stateVersion: 3,
  tagline: "Follow an error signal backward through a computation graph, using the chain rule to determine how much each weight contributed to the final loss.",
  objectives: [
    "Follow forward values and backward derivatives",
    "Use the chain rule on a small graph",
    "Predict how gradient size changes from layer to layer as depth, activation, initial weight scale, and skip connections change",
  ],
  glossary: [
  {
    "term": "Chain rule",
    "definition": "The rule that multiplies local derivatives along a dependency path. Here it gives dL/dw = 2(y_hat - y) times y_hat(1 - y_hat) times x."
  },
  {
    "term": "Local derivative",
    "definition": "How one operation's output changes with one of its inputs, computed without any knowledge of the rest of the graph."
  },
  {
    "term": "Computation graph",
    "definition": "The record of operations connecting inputs and parameters to the loss. Backpropagation is a traversal of this graph in reverse."
  },
  {
    "term": "Backward pass",
    "definition": "The right-to-left sweep that turns one scalar loss into a gradient for every parameter. It requires the forward values to still be available."
  },
  {
    "term": "Upstream gradient",
    "definition": "The gradient arriving at a node from the loss side. The node multiplies it by its own local derivative and passes the result on; when a node feeds several others, the arriving gradients are added first."
  },
  {
    "term": "Reverse-mode autodiff",
    "definition": "The algorithm frameworks implement to get every parameter's gradient in one backward pass. That pass costs a small constant multiple of the forward pass, typically about two times, however many parameters there are."
  },
  {
    "term": "Partial derivative",
    "definition": "The response of an output to one input while the others are held fixed. Written with the curly d, as in dL/dw."
  },
  {
    "term": "Gradient checking",
    "definition": "Comparing an analytic gradient against a finite-difference estimate. Disagreement almost always indicates a bug in a hand-written derivative."
  },
  {
    "term": "Vanishing gradient",
    "definition": "The decay toward zero when many local derivatives below one are multiplied. Sigmoid contributes at most 0.25 per layer, which compounds quickly with depth; the Gradient flow card measures it layer by layer."
  },
  {
    "term": "Exploding gradient",
    "definition": "The unbounded growth when many local derivatives above one are multiplied. Gradient clipping rescales the gradient whenever its norm passes a threshold."
  },
  {
    "term": "Weight initialization",
    "definition": "The random starting weights, drawn with a standard deviation that sets how much each layer scales the forward signal and the backward gradient. Glorot initialization (Glorot and Bengio, 2010) uses variance 2 over fan-in plus fan-out; He initialization (He et al., 2015) uses 2 over fan-in and was derived for rectifiers. Too small a scale shrinks every gradient and too large a scale inflates it."
  },
  {
    "term": "Skip connection",
    "definition": "A path that adds a layer's input to its output, so the backward pass has an identity route that bypasses the layer's weights and slope. Also called a residual connection. It keeps early layers' gradients from vanishing, though without normalization the stacked additions can make activations grow with depth."
  },
  {
    "term": "Activation memory",
    "definition": "The forward values held in memory for the backward pass. It grows with depth, batch size, and sequence length, and often limits training before compute does."
  },
  {
    "term": "Credit assignment",
    "definition": "Determining how much each parameter contributed to the final error. Backpropagation answers it with exact local sensitivities: how much the loss would change for a small nudge to each parameter."
  }
],
  references: [
    {
      authors: "David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams",
      title: "Learning representations by back-propagating errors",
      source: "Nature 323(6088), 533–536",
      year: 1986,
      url: "https://www.nature.com/articles/323533a0",
      note: "The short paper that made back-propagation the standard way to train networks of neuron-like units. It sends the error backward to adjust every weight, including weights in hidden layers, which is the credit assignment this lesson is about.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 7, Gradients and initialization, works the chain rule through a toy model, states the backpropagation algorithm, and shows how a bad starting weight scale leads to vanishing or exploding gradients, with He initialization as the fix. Chapter 11 explains residual connections, why stacking them makes forward values grow with depth, and how batch normalization holds that growth down.",
    },
    {
      authors: "Stanford CS231n course staff",
      title: "Backpropagation, Intuitions",
      source: "CS231n: Deep Learning for Computer Vision, Stanford University course notes",
      year: 2026,
      url: "https://cs231n.github.io/optimization-2/",
      note: "Walks through the chain rule on small circuits and a sigmoid example, the same kind of graph as the Backpropagation trace. It shows that a multiply gate uses the other input's value as its local gradient, that forward values must be cached for the backward pass, and that gradients add up where a value feeds two paths.",
    },
    {
      authors: "Stanford CS231n course staff",
      title: "Neural Networks Part 3: Learning and Evaluation",
      source: "CS231n: Deep Learning for Computer Vision, Stanford University course notes",
      year: 2026,
      url: "https://cs231n.github.io/neural-networks-3/",
      note: "The Gradient Checks section explains how to compare an analytic gradient with a finite-difference estimate, using a centered difference and a relative error. This is the check behind the Numeric check and Numeric agreement readouts.",
    },
    {
      authors: "Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, et al.",
      title: "Automatic Differentiation in Machine Learning: a Survey",
      source: "Journal of Machine Learning Research 18(153), 1–43",
      year: 2018,
      url: "https://jmlr.org/papers/v18/17-468.html",
      note: "Explains forward-mode and reverse-mode automatic differentiation and why one reverse pass gives the whole gradient of a single loss, while forward mode needs one pass per input. It also notes that reverse mode pays for this with memory for the stored intermediate values, the lesson's activation memory.",
    },
    {
      authors: "PyTorch contributors",
      title: "Autograd mechanics",
      source: "PyTorch documentation",
      year: 2026,
      url: "https://docs.pytorch.org/docs/stable/notes/autograd.html",
      note: "Describes how PyTorch records a graph of operations during the forward pass, saves the tensors each operation needs, and walks the graph backward with the chain rule. It also notes that gradients are accumulated into each parameter's .grad, which is why the lesson says they must be zeroed between steps.",
    },
    {
      authors: "Yoshua Bengio, Patrice Simard, and Paolo Frasconi",
      title: "Learning long-term dependencies with gradient descent is difficult",
      source: "IEEE Transactions on Neural Networks 5(2), 157–166",
      year: 1994,
      url: "https://www.semanticscholar.org/paper/d0be39ee052d246ae99c082a565aba25b811be2d",
      note: "Shows why gradient descent gets harder as an error signal must travel back over more and more steps of a recurrent network. It is an early account of the vanishing and exploding gradients that the Gradient flow card measures across layers.",
    },
    {
      authors: "Xavier Glorot and Yoshua Bengio",
      title: "Understanding the difficulty of training deep feedforward neural networks",
      source: "Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS 2010), PMLR 9, 249–256",
      year: 2010,
      url: "https://proceedings.mlr.press/v9/glorot10a.html",
      note: "Finds that the sigmoid is a poor fit for deep networks with random starting weights because its units saturate, and tracks how gradients change from layer to layer. It proposes the starting weight scale the lab calls Glorot.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al.",
      title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",
      source: "Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015), 1026–1034",
      year: 2015,
      url: "https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html",
      note: "Derives a starting weight scale for networks that use rectifier (ReLU) units, so very deep models can be trained from scratch. This is the He option in Initial weight scale, which the lesson's ReLU runs compare against small, Glorot, and large.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al.",
      title: "Deep Residual Learning for Image Recognition",
      source: "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), 770–778",
      year: 2016,
      url: "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
      note: "Introduces residual networks, where each block adds its input to its output, and shows they are easier to optimize as depth grows. These are the skip connections the Skip connections toggle adds to the Gradient flow card.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al.",
      title: "Identity Mappings in Deep Residual Networks",
      source: "Computer Vision – ECCV 2016, Lecture Notes in Computer Science 9908, 630–645",
      year: 2016,
      url: "https://arxiv.org/abs/1603.05027",
      note: "Analyzes how signals move through residual blocks and shows that with identity skips the gradient can pass directly from one block to any earlier one. This is the identity route in the lesson's backward formula for a skip connection.",
    },
    {
      authors: "Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio",
      title: "On the difficulty of training recurrent neural networks",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3), 1310–1318",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/pascanu13.html",
      note: "Studies the vanishing and exploding gradient problems and proposes gradient norm clipping, which rescales the gradient when its norm passes a threshold. This backs the lesson's definition of an exploding gradient and its clipping remedy.",
    },
    {
      authors: "Tianqi Chen, Bing Xu, Chiyuan Zhang, et al.",
      title: "Training Deep Nets with Sublinear Memory Cost",
      source: "arXiv preprint arXiv:1604.06174",
      year: 2016,
      url: "https://arxiv.org/abs/1604.06174",
      note: "Saves memory by storing only some forward values and recomputing the rest during the backward pass, at the cost of about one extra forward pass. It reports about 30 percent extra running time for a large memory cut, the trade the lesson gives for gradient checkpointing.",
    },
    {
      authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, et al.",
      title: "Mixed Precision Training",
      source: "International Conference on Learning Representations (ICLR 2018)",
      year: 2018,
      url: "https://arxiv.org/abs/1710.03740",
      note: "Trains networks with 16-bit (half-precision) numbers and scales the loss up before the backward pass so that small gradients are not lost. This is the loss-scaling factor the lesson describes for fp16 training.",
    },
    {
      authors: "Timothy P. Lillicrap, Adam Santoro, Luke Marris, et al.",
      title: "Backpropagation and the brain",
      source: "Nature Reviews Neuroscience 21(6), 335–346",
      year: 2020,
      url: "https://www.nature.com/articles/s41583-020-0277-3",
      note: "Explains why strict backpropagation is hard to square with the brain: it needs exact error signals sent back along feedback connections. The authors argue the brain may approximate it in other ways, which adds nuance to the lesson's point that backpropagation is not a model of biological learning.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Set Input x to 1.5, Weight w to 3 and Target y to 0.1. The loss is large, yet ∂L/∂w is small. Which factor of the chain-rule product is responsible?",
      options: [
        "The loss node's 2(ŷ − y), because a squared-error loss gives smaller gradients as the miss grows",
        "The sigmoid's ŷ(1 − ŷ), because ŷ has saturated near 1 and almost no gradient passes back through it",
        "The multiply node's x, because a bigger input always makes the gradient on the weight smaller",
        "None of them, because the loss and the gradient on a weight must always rise and fall together",
      ],
      answer: 1,
      explanation:
        "∂L/∂w = 2(ŷ − y) · ŷ(1 − ŷ) · x. Here the first factor is large because ŷ is far from 0.1, and x is 1.5, but the sigmoid has saturated, so ŷ(1 − ŷ) is far below its peak of 0.25. One tiny factor shrinks the whole product, so a large loss can sit beside a small gradient.",
      objective: 1,
    },
    {
      prompt:
        "At the Product → weight step the tape multiplies the running gradient by one local factor. Which one?",
      options: [
        "w, the weight itself, because its own value sets how strongly it changes z",
        "The learning rate, because the step size scales every gradient on the way back",
        "x, the input value from the forward pass, because z = x · w makes ∂z/∂w equal to x",
        "The target y, because the loss was measured against it and the target is part of the graph",
      ],
      answer: 2,
      explanation:
        "For a product z = x · w, the derivative with respect to w is x and the derivative with respect to x is w: each input's gradient uses the other input's forward value. That is why the forward values have to be kept in memory until the backward pass reaches them. The learning rate only enters later, when you press Apply.",
      objective: 0,
    },
    {
      prompt:
        "This lab's graph is one chain. In a real network one node's output feeds two later nodes. What does the backward pass do at that node?",
      options: [
        "It keeps the larger of the two arriving gradients, because the strongest path decides the blame",
        "It averages the two gradients, so a node's gradient does not grow with its number of paths",
        "It adds the gradients arriving along both paths, then multiplies by its own local derivative",
      ],
      answer: 2,
      explanation:
        "A change in the node's output reaches the loss along every path that uses it, and the total effect is the sum of the per-path effects. So gradients arriving from several consumers are added, then multiplied by the node's local derivative. A reused weight accumulates contributions the same way. The lab's single chain never needs this sum.",
      objective: 0,
    },
    {
      prompt:
        "The graph also computes a derivative toward the input x, although x has no weight to update. Why compute it?",
      options: [
        "In a deeper network that same branch carries the gradient on to the weights of earlier layers",
        "It updates the training example, so the next forward pass lands closer to the target",
        "It is used only to check that the Analytic slope matches the Numeric check",
        "It sets the learning rate, since a larger gradient on the input calls for a smaller step on every weight",
      ],
      answer: 0,
      explanation:
        "Here x is raw data, so nothing happens at the end of that branch. In a deeper model the node feeding a layer is the previous layer's output, which depends on that layer's weights. The chain rule keeps multiplying local derivatives down that branch, and that is how every earlier weight gets its gradient.",
      objective: 1,
    },
    {
      prompt:
        "In Gradient flow through depth, set Network depth to 12, Activation to sigmoid, Initial weight scale to Glorot, and Skip connections off. How does Layer 1's gradient norm compare with Layer 12's?",
      options: [
        "It is many orders of magnitude smaller, since each sigmoid on the way back passes at most a quarter of it",
        "About the same, because Glorot starting weights keep the signal the same size at every layer, however many there are",
        "Larger, because the gradient is added up over every layer above it as it travels back toward the input layer",
        "A little smaller, because the first layer sees less of the loss and the effect is gentle even at twelve layers",
      ],
      answer: 0,
      explanation:
        "Each step back multiplies the gradient by the layer's weights and by the sigmoid's slope, which is at most 0.25, so twelve layers compound a factor far below one. Glorot scaling fixes the weights' share, not the slope's. At seed 7 Layer 1's norm is 2.6e-8 and Layer 12's is 0.21; across seeds 1 to 7 the ratio stays between 9e-8 and 3e-7. Products multiply down the chain; they are not summed.",
      objective: 2,
    },
    {
      prompt:
        "With depth 12, sigmoid, and Glorot, you turn Skip connections on. What changes, and what does it cost?",
      options: [
        "Layer 1's norm rises by orders of magnitude and its ratio to Layer 12 is no longer tiny, but the loss jumps",
        "Nothing changes, because a skip connection only matters once training has begun and not at the starting weights",
        "Layer 1's norm falls even further, because each skip adds another factor below one to the product on the way back",
        "Every layer's norm becomes exactly equal, because the skip copies one unchanged gradient to all of the layers",
      ],
      answer: 0,
      explanation:
        "A skip adds an identity term to the backward step, so gradient reaches Layer 1 without passing every slope: Layer 1's norm goes from 2.6e-8 to 0.24 and the ratio from 1.2e-7 to 0.077 at seed 7. The price is forward growth: each skip adds a layer's output onto its input, so Loss at the start rises from 0.90 to 7.30. Real residual networks pair skips with normalization or small branch scales for this reason.",
      objective: 2,
    },
    {
      prompt:
        "At depth 12 with ReLU and Skip connections off, you step Initial weight scale from small to Glorot to He to large. What happens to Layer 1's norm?",
      options: [
        "It rises at every step, from far below 1 to far above 1, and every other layer's norm rises with it",
        "It stays the same, because a ReLU's slope is exactly 1 wherever a unit is active, so the starting weights cannot matter",
        "It falls at every step, because larger weights make the backward signal cancel itself out across layers",
        "Only Layer 1's norm rises, because only the layer next to the input depends on how the weights start",
      ],
      answer: 0,
      explanation:
        "The starting weights set a gain per layer, below one for small and Glorot, near one for He, and above one for large, and twelve layers compound it in the forward signal and the backward gradient, so the gradient on each weight matrix inherits it. At seed 7 Layer 1's norm is 2.4e-7 for small, 2.0e-2 for Glorot, 1.3 for He, and 3.5e5 for large, and at every seed from 1 to 7 the four ranges stay in that order. The scale moves every layer together, which a trained network feels as no learning or a blow-up.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateBackpropState,
};

export default definition;
