import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {
  startW: -1,
  startB: 1.2,
  learningRate: 0.1,
  momentum: 0.9,
  optimizer: "gd",
  step: 0,
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * Version 1 stored one weight on a closed-form curve (`weight`, `velocity`,
 * optimizer "sgd"). Version 2 runs descent on a real two-parameter loss, so the
 * old weight and learning rate have no counterpart. Keep the optimizer choice
 * and the momentum coefficient, which mean the same thing, and start fresh.
 */
function migrate(parsed: ModuleState): ModuleState {
  const isVersion1 = "weight" in parsed || "velocity" in parsed || parsed.optimizer === "sgd";
  if (!isVersion1) return { ...initialState, ...parsed };
  return {
    ...initialState,
    optimizer: parsed.optimizer === "momentum" ? "momentum" : "gd",
    momentum:
      typeof parsed.momentum === "number" && Number.isFinite(parsed.momentum)
        ? clamp(parsed.momentum, 0, 0.95)
        : initialState.momentum,
  };
}

const definition: ModuleDefinition = {
  id: "module-04-loss-gradient-descent",
  slug: "loss-gradient-descent",
  title: "Loss & gradient descent",
  group: "foundations",
  order: 10,
  icon: "Mountain",
  accent: "#dd6b45",
  prerequisites: ["module-02-single-neuron", "module-30-math-you-need"],
  estimatedMinutes: 14,
  steps: ["Measure error","Read the slope","Take steps","Compare momentum"],
  stepInstructions: [
    "At step 0, drag Start w and Start b and compare the squared-residual bars on Data and fit with Loss L; the loss is their mean.",
    "At step 0, read ∂L/∂w and ∂L/∂b, then check that the −∇L arrow on Loss surface crosses the nearest contour at a right angle and misses the minimum.",
    "Press Take one gradient step, then set Learning rate to 0.05, 0.20, 0.40 and 0.48 and compare the path, Settles by, and the regime badge.",
    "At Learning rate 0.05, switch Optimizer to Momentum and compare Settles by at Momentum β 0.8 and 0.95 with plain descent.",
  ],
  stateVersion: 2,
  tagline: "Watch gradient descent search a real loss surface: the loss measures the error, the gradient picks the direction, and the learning rate decides whether the search settles, zigzags, or blows up.",
  objectives: ["Interpret loss as an optimization objective","Predict a gradient step's direction","Explain how the learning rate and curvature decide whether descent settles, overshoots, or diverges"],
  glossary: [
  {
    "term": "Loss",
    "definition": "A single number measuring how wrong the model is on some data, built so that lower is better and so that it can be differentiated. It is a proxy for what you actually want, not the goal itself."
  },
  {
    "term": "Gradient",
    "definition": "The vector of partial derivatives of the loss with respect to every parameter. It points in the direction of steepest local increase and crosses the contour through the current point at a right angle, so descent subtracts it."
  },
  {
    "term": "Objective function",
    "definition": "The quantity being optimized. Mean squared error, the average of squared misses, suits numeric targets; cross-entropy, the negative log probability given to the correct class or token, suits classifiers and language models."
  },
  {
    "term": "Learning rate",
    "definition": "The multiplier on each step, written eta. The gradient chooses the direction and the learning rate the distance; past 2 divided by the largest curvature, plain descent overshoots further on every step and diverges."
  },
  {
    "term": "Gradient descent",
    "definition": "The update rule: parameters minus eta times the gradient, applied repeatedly. It needs only the local slope, which is why it scales to billions of parameters."
  },
  {
    "term": "Stochastic gradient descent",
    "definition": "Gradient descent with the gradient estimated from a random minibatch instead of the full dataset. Each step is cheaper and noisier; this lab uses the exact full-data gradient instead."
  },
  {
    "term": "Minibatch",
    "definition": "The small group of examples used to estimate one gradient. Larger batches give less noisy directions at a higher cost per step."
  },
  {
    "term": "Momentum",
    "definition": "An optimizer state that accumulates past gradients as a velocity, v = beta v + gradient. On a steady slope the step grows toward 1/(1 - beta) plain steps, while directions that keep reversing partly cancel."
  },
  {
    "term": "Curvature",
    "definition": "How fast the slope changes along a direction: the second derivative. The largest curvature sets the largest stable learning rate, and a big ratio between largest and smallest makes a narrow valley that plain descent crosses slowly."
  },
  {
    "term": "Convergence",
    "definition": "Settling where the loss stops improving. With a fixed learning rate the steps still shrink on their own, because the gradient shrinks near a minimum."
  },
  {
    "term": "Divergence",
    "definition": "Loss growing without bound, usually because the step is too large for the local curvature, so each step overshoots the valley floor by more than the last."
  },
  {
    "term": "Local minimum",
    "definition": "A point lower than everything nearby. In high dimensions most flat points at high loss are saddles rather than minima, and the minima large networks reach tend to have similar loss."
  },
  {
    "term": "Saddle point",
    "definition": "A flat point that is downhill in some directions and uphill in others. Saddles and long plateaus slow real training more often than bad local minima do."
  },
  {
    "term": "Learning-rate schedule",
    "definition": "A planned change of step size across a run, typically a short warmup from near zero followed by a decay. It stabilizes the early steps and lets the late steps settle."
  }
],
  references: [
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 5 builds losses such as least squares and cross-entropy, the two losses this lesson compares. Chapter 6, Fitting models, covers gradient descent, stochastic gradient descent, momentum, and Adam in the same order this lab and its Going deeper section do.",
    },
    {
      authors: "Gabriel Goh",
      title: "Why Momentum Really Works",
      source: "Distill 2(4)",
      year: 2017,
      url: "https://distill.pub/2017/momentum/",
      note: "An interactive article that splits a quadratic bowl into its curvature directions and shows each step multiplying the error by 1 - eta lambda, so steps work only while eta lambda stays under 2. It uses the same momentum rule as the Momentum setting, v = beta v + gradient, and shows how it speeds up a long, narrow valley.",
    },
    {
      authors: "Sebastian Ruder",
      title: "An overview of gradient descent optimization algorithms",
      source: "arXiv preprint arXiv:1609.04747",
      year: 2016,
      url: "https://arxiv.org/abs/1609.04747",
      note: "A short survey that compares full-batch, stochastic, and minibatch gradient descent, then explains why plain descent struggles in ravines and how momentum helps there. It also describes Adam and other adaptive optimizers, so it works as a map of the optimizers this lesson names.",
    },
    {
      authors: "Léon Bottou, Frank E. Curtis, and Jorge Nocedal",
      title: "Optimization Methods for Large-Scale Machine Learning",
      source: "SIAM Review 60(2), 223–311",
      year: 2018,
      url: "https://arxiv.org/abs/1606.04838",
      note: "A review of why stochastic gradient methods dominate large-scale training. It shows that a minibatch gradient costs more per step but has less noise, which backs the lesson's minibatch trade-off and the glossary entry on stochastic gradient descent.",
    },
    {
      authors: "Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, et al.",
      title: "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization",
      source: "Advances in Neural Information Processing Systems 27 (NIPS 2014)",
      year: 2014,
      url: "https://proceedings.neurips.cc/paper_files/paper/2014/hash/04192426585542c54b96ba14445be996-Abstract.html",
      note: "Argues that in high dimensions the main obstacle is a flood of saddle points surrounded by flat, high-loss plateaus, not bad local minima. This is the source for the Where it breaks claim that saddles slow real training more often than bad minima do.",
    },
    {
      authors: "Anna Choromanska, Mikael Henaff, Michael Mathieu, et al.",
      title: "The Loss Surfaces of Multilayer Networks",
      source: "Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS 2015), PMLR 38, 192–204",
      year: 2015,
      url: "https://proceedings.mlr.press/v38/choromanska15.html",
      note: "Finds that in large networks the local minima crowd into a narrow band of similar, low loss, while poor minima become rare as the network grows. It backs the glossary point that the minima large networks reach tend to have similar loss.",
    },
    {
      authors: "Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, et al.",
      title: "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability",
      source: "International Conference on Learning Representations (ICLR 2021)",
      year: 2021,
      url: "https://arxiv.org/abs/2103.00065",
      note: "Measures the largest curvature during real network training and finds it keeps rising until it reaches about 2 divided by the learning rate, the same limit this lab's regime badge marks. It backs the lesson's warning that curvature changes during training, so a learning rate that was stable early can become unstable.",
    },
    {
      authors: "Diederik P. Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      source: "3rd International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1412.6980",
      note: "The original Adam paper. It keeps running averages of each parameter's gradient and squared gradient and divides each step by the square root of the second, the per-parameter rescaling described in Going deeper.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "Decoupled Weight Decay Regularization",
      source: "International Conference on Learning Representations (ICLR 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1711.05101",
      note: "Introduces AdamW, which applies weight decay directly to the parameters instead of mixing it into the gradient that Adam rescales. This is the difference the lesson gives between AdamW and plain Adam.",
    },
    {
      authors: "Priya Goyal, Piotr Dollár, Ross Girshick, et al.",
      title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
      source: "arXiv preprint arXiv:1706.02677",
      year: 2017,
      url: "https://arxiv.org/abs/1706.02677",
      note: "Shows that large learning rates can upset the first steps of training and fixes this with a gradual warmup that ramps the rate up from a small value. It backs the warmup half of the learning-rate schedule in Going deeper.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "SGDR: Stochastic Gradient Descent with Warm Restarts",
      source: "International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1608.03983",
      note: "Proposes lowering the learning rate along a cosine curve during each run. It is the origin of the cosine decay that the lesson pairs with warmup in a learning-rate schedule.",
    },
    {
      authors: "Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio",
      title: "On the difficulty of training recurrent neural networks",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3), 1310–1318",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/pascanu13.html",
      note: "Proposes gradient norm clipping: when the gradient's size passes a threshold, scale it back down before the update. This is the gradient clipping the lesson describes as a guard against one odd minibatch throwing a run off.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "On Data and fit you drag Start w and Start b until Loss L is very small. What does that tell you?",
      options: [
        "The line will predict new points just as well, because loss measures how well a model generalizes",
        "The line misses these eight points by little on average; it says nothing yet about new points",
        "The surface is steepest here, because a small loss means the next gradient step will be large",
      ],
      answer: 1,
      explanation:
        "Loss L is the mean of the squared misses on these eight points, so a small value means a close fit to them. It is a proxy for what you want, not the goal itself: a line can fit its own points closely and still predict fresh ones poorly, which Training dynamics shows. And near the minimum the gradient shrinks toward zero, so the next step is small, not large.",
      objective: 0,
    },
    {
      prompt:
        "At step 0, ∂L/∂w and ∂L/∂b both read negative. What does Take one gradient step do?",
      options: [
        "Both w and b decrease, because a negative slope means the parameters should shrink to lower the loss",
        "Both increase and the step lands exactly on the minimum, because the gradient always points straight at it",
        "Both w and b increase, moving along the pink arrow, which crosses the nearest contour at a right angle",
      ],
      answer: 2,
      explanation:
        "Descent subtracts the gradient, so two negative slopes make both parameters grow: the loss falls as w and b rise. The pink −∇L arrow is perpendicular to the contour through the current point, which is the steepest local downhill, not the direction of the minimum. In a long narrow valley the two differ a lot, so one step lowers the loss but does not land on the minimum.",
      objective: 1,
    },
    {
      prompt:
        "At step 0 you double Learning rate before pressing Take one gradient step. How does the step change?",
      options: [
        "It points the same way and is twice as long, because the rate scales distance and never direction",
        "It turns toward the minimum, because a larger rate lets the step correct for the tilt of the valley",
        "It does not change at all, because the gradient alone decides both the direction and the distance",
      ],
      answer: 0,
      explanation:
        "The update is θ ← θ − η∇L, so η multiplies the same gradient vector: the step has the same direction and a length proportional to η. The learning rate cannot aim the step, only stretch it, which is why too large a rate overshoots across a valley instead of finding a better route.",
      objective: 1,
    },
    {
      prompt:
        "At learning rate 0.48 the path zigzags across the valley and each swing is wider than the last. What change stops the blow-up?",
      options: [
        "Take more steps so the swings have time to settle",
        "Start closer to the minimum so the first swing is small",
        "Raise the learning rate further so one step jumps clear across the valley",
        "Lower the learning rate below 2 divided by the largest curvature",
      ],
      answer: 3,
      explanation:
        "Each plain step multiplies the error across the valley by 1 − ηλ_max. At η = 0.48 that factor is −1.09, so the error flips sign and grows about 9 percent per step from any start except the exact minimum. More steps or a nearer start only delay the blow-up. Below 2/λ_max = 0.458 the factor's size drops under 1 and the zigzag shrinks.",
      objective: 2,
    },
    {
      prompt:
        "At Learning rate 0.20 the path drops fast, then crawls along the valley floor and takes tens of steps to settle. What limits the speed along the floor?",
      options: [
        "The learning rate decays automatically near the bottom, so a built-in schedule shortens the steps on purpose as the loss falls",
        "The gradient is exactly zero everywhere along the floor, so no gradient step can move the point that way",
        "Low curvature along the floor: each step shrinks that error only slightly, while the steep direction caps the rate",
        "The loss is not a simple bowl there, so descent keeps stalling at saddle points on its way to the minimum",
      ],
      answer: 2,
      explanation:
        "The learning rate has to stay under 2/λ_max for the steep direction, which fixes the step. Along the shallow direction the error shrinks by a factor of 1 − ηλ_min per step, and with λ_min small that factor stays close to 1, so progress there is slow. The lab uses a fixed rate, and its loss is a perfect bowl with one minimum and no saddles, so neither a schedule nor a stall is the cause.",
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
