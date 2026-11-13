import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-61-neural-networks",
  slug: "neural-networks",
  title: "Neural networks",
  group: "foundations",
  order: 10,
  icon: "CircleGauge",
  accent: "#2f8a68",
  prerequisites: ["module-59-slopes-gradients"],
  estimatedMinutes: 22,
  steps: ["Two knobs", "Seven knobs", "Turn hints", "More neurons", "Count the knobs", "Playground"],
  stepInstructions: [
    "Answer Predict first, then turn Knob w until every gap is the same length and Knob b until Loss reads 0.000. Move Input x and read the three boxes on the machine.",
    "Answer Predict first. Choose a knob under Selected knob, drag Turn the selected knob, and watch that neuron's thin line and the Loss. Then press Swap the two neurons and compare the curve before and after.",
    "Answer Predict first, then press Take one step and watch every knob move along its pink arrow. Press Run to the goal and compare the Now row of the table with the Made with row. Press Scramble the knobs and run again.",
    "Answer Predict first, then drag Hidden neurons from 1 to 6 and compare Loss after 600 steps with the bars. Leave it at 4 and press Scramble the knobs a few times, watching for a start that gets stuck.",
    "Answer Predict first. Press A small digit reader and read the Knobs line, then raise Neurons per layer and Hidden layers and compare your bar with the four published networks.",
    "With Dots on Wave, press Train and watch the Loss stall. Raise Neurons per layer to 4 and press Train again. Then change one thing at a time, Momentum to Off, Step size to 0.5, Hidden layers to 2, and compare Loss with the goal.",
  ],
  stateVersion: 1,
  tagline:
    "Open the machine up. A neural network is simple units wired together, and everything it does is set by numbers you can picture as knobs: turn two by hand, then seven, then watch training turn them all at once, and count how many a real model has.",
  objectives: [
    "Describe a neural network as fixed wiring whose behaviour is set entirely by its parameters: a weight on every wire and a bias in every neuron",
    "Explain why trying parameter settings one by one fails, and how a training step instead turns every parameter a little against its own slope of the loss",
    "Show that different parameter settings can compute the same function, so one parameter has no meaning on its own",
    "Relate the number of hidden neurons to the shapes a network can draw, and tell a network that cannot make a shape from one that got stuck while training",
    "Count a fully connected network's parameters from its layer sizes and place the count beside published models",
  ],
  glossary: [
    {
      term: "Neural network",
      definition:
        "A function built from many simple units, called neurons, wired in layers. Its wiring is fixed by the designer; what it computes is set by its parameters, which are found by training rather than written by hand.",
    },
    {
      term: "Parameter",
      definition:
        "Any number inside a model that training is allowed to change. In this lab every parameter is drawn as a knob, and a network's parameters are its weights and biases, nothing else.",
    },
    {
      term: "Weight",
      definition:
        "The parameter on one wire: the number a value is multiplied by as it travels from one neuron to the next. Its sign says whether the value pushes the next neuron up or down, and its size says how hard.",
    },
    {
      term: "Bias",
      definition:
        "A neuron's own parameter, added to its weighted inputs before anything else happens. It shifts where the neuron's curve sits, and lets a neuron send a value on even when every input is 0.",
    },
    {
      term: "Neuron",
      definition:
        "One unit of a network. It multiplies each input by a weight, adds the results and its bias, and passes the total through a fixed curve. Here that curve is tanh, an S shape running from −1 to 1. It is a piece of arithmetic, not a model of a brain cell.",
    },
    {
      term: "Hidden layer",
      definition:
        "A row of neurons between the input and the output. Nobody says what its neurons should send on; training settles that, which is why their values are called hidden.",
    },
    {
      term: "Forward pass",
      definition:
        "Running an input through the network from left to right, with the parameters held still, to get the output. It is the only thing a trained network does when you use it.",
    },
    {
      term: "Loss",
      definition:
        "One number for how wrong the network is on its examples; lower is better. Here it is the average of the squared gaps between the output and the dots, so 0 means every dot is hit. It measures the fit to these examples only.",
    },
    {
      term: "Gradient",
      definition:
        "The list of slopes of the loss, one for each parameter: how fast the loss would change if that parameter alone were turned a little. It says which way each parameter should move, not where it should end up.",
    },
    {
      term: "Training step",
      definition:
        "One update of every parameter at once: each moves against its own slope by the step size times that slope. Training is this step repeated, and the lab's playground takes at most 600 of them.",
    },
    {
      term: "Learning rate",
      definition:
        "The step size: the one number that scales every parameter's move in a training step. Too small and training crawls; too large and each step overshoots, so the loss rises instead of falling.",
    },
    {
      term: "Momentum",
      definition:
        "A change to the training step in which each parameter also keeps a share of its previous move, 0.9 of it in this lab. It carries the parameters through flat stretches where plain steps stall, at the cost of sometimes overshooting.",
    },
    {
      term: "Weight-space symmetry",
      definition:
        "The fact that different parameter settings can give exactly the same network. Swapping two hidden neurons, along with all their weights, changes many numbers and nothing the network computes.",
    },
    {
      term: "Capacity",
      definition:
        "The range of shapes a network's parameters could produce, set by how many neurons it has and how they are wired. Training can only pick from that range, and more capacity does not guarantee that training finds a good pick.",
    },
  ],
  references: [
    {
      authors: "Grant Sanderson",
      title: "But what is a neural network?",
      source: "3Blue1Brown, Deep Learning chapter 1",
      year: 2017,
      url: "https://www.3blue1brown.com/lessons/neural-networks",
      note: "A visual introduction that builds a digit reader with layers of 784, 16, 16 and 10 neurons and counts its 13,002 weights and biases, calling them knobs and dials. The Count the knobs card's A small digit reader button builds the same network and reaches the same count.",
    },
    {
      authors: "Michael A. Nielsen",
      title: "Neural Networks and Deep Learning, chapter 4: A visual proof that neural nets can compute any function",
      source: "Determination Press, free to read online",
      year: 2015,
      url: "http://neuralnetworksanddeeplearning.com/chap4.html",
      note: "Shows with interactive pictures how hidden neurons make step shapes, how two of them make a bump, and how adding bumps can approximate a function of one input. It is the idea behind the thin lines on the More neurons, more shapes card.",
    },
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 3 builds a network with one input and one output from a single row of hidden units, one unit at a time, and counts its parameters. It is the closest textbook match to the networks this lab draws.",
    },
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Chapter 6 defines the networks in this lab, layers of units with weights and biases, in full. Section 8.2.2 explains weight space symmetry: exchanging hidden units, with their weights, gives an equivalent network, which is what Swap the two neurons shows.",
    },
    {
      authors: "David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams",
      title: "Learning representations by back-propagating errors",
      source: "Nature 323(6088), 533–536",
      year: 1986,
      url: "https://www.nature.com/articles/323533a0",
      note: "The short paper that made it standard to train a network by repeatedly adjusting every weight to reduce a measure of error. It is the source of the step this lab takes on the Let the machine turn the knobs card.",
    },
    {
      authors: "George Cybenko",
      title: "Approximation by superpositions of a sigmoidal function",
      source: "Mathematics of Control, Signals, and Systems 2(4), 303–314",
      year: 1989,
      url: "https://doi.org/10.1007/BF02551274",
      note: "An early proof that one hidden layer of S-shaped units, if it has enough of them, can approximate any continuous function on a bounded range. It says good weights exist, not that training will find them, which is the gap the stuck runs in this lab fall into.",
    },
    {
      authors: "Daniel Smilkov, Shan Carter, D. Sculley, et al.",
      title: "Direct-Manipulation Visualization of Deep Networks",
      source: "arXiv:1708.03788",
      year: 2017,
      url: "https://arxiv.org/abs/1708.03788",
      note: "The paper behind TensorFlow Playground, a browser tool where people change a small network's settings and structure and watch it train. This lab's Playground card follows the same idea on a problem with one input.",
    },
    {
      authors: "Gabriel Goh",
      title: "Why Momentum Really Works",
      source: "Distill 2(4)",
      year: 2017,
      url: "https://distill.pub/2017/momentum/",
      note: "An interactive article on the momentum rule, in which each step keeps a share of the previous one. It explains why the Playground's Momentum setting gets through stretches where plain steps stall.",
    },
    {
      authors: "Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner",
      title: "Gradient-Based Learning Applied to Document Recognition",
      source: "Proceedings of the IEEE 86(11), 2278–2324",
      year: 1998,
      url: "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf",
      note: "The LeNet-5 paper. Its network for reading handwritten digits has about 60,000 trainable parameters, the smallest published network on the Count the knobs card.",
    },
    {
      authors: "Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton",
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      source: "Advances in Neural Information Processing Systems 25 (NeurIPS 2012)",
      year: 2012,
      url: "https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
      note: "The AlexNet paper. Its abstract gives the network 60 million parameters, the figure on the Count the knobs card, a thousand times LeNet-5's count fourteen years later.",
    },
    {
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, et al.",
      title: "Language Models are Unsupervised Multitask Learners",
      source: "OpenAI technical report",
      year: 2019,
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      note: "The GPT-2 paper. Table 2 lists four sizes, the largest with 1,542 million parameters, which is the 1.5 billion on the Count the knobs card.",
    },
    {
      authors: "Tom B. Brown, Benjamin Mann, Nick Ryder, et al.",
      title: "Language Models are Few-Shot Learners",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/2005.14165",
      note: "The GPT-3 paper. Its abstract describes a language model with 175 billion parameters, the largest count on the Count the knobs card.",
    },
  ],
  checkpoint: [
    {
      prompt: "On A machine with two knobs, you turn Knob b up by 0.5 and leave Knob w alone. What happens to the machine's line?",
      options: [
        "The whole line rises by 0.5 and keeps its tilt",
        "The line tilts more steeply around its middle point",
        "Only the right-hand end of the line rises, by 0.5",
      ],
      answer: 0,
      explanation:
        "The output is w × x + b, so b is added to every output whatever the input was: the whole line slides up by 0.5. The tilt comes from w, which multiplies the input, so a change to w moves the ends of the line more than its middle. Two parameters, two different jobs, and nothing else in the machine can change.",
      objective: 0,
    },
    {
      prompt: "On Let the machine turn the knobs, one knob reads +1.00 and its slope reads +0.80. The step size is 0.2. Where is that knob after one step?",
      options: ["At +1.16, because the knob follows its own slope up", "At +0.84, because the knob moves against its slope", "At +0.80, because the knob is set to its slope"],
      answer: 1,
      explanation:
        "A positive slope means turning the knob up would raise the loss, so the step turns it down: 1.00 − 0.2 × 0.80 = 0.84. The slope is not a destination. It only gives a direction and a size for one small move, and every other knob makes its own small move in the same step.",
      objective: 1,
    },
    {
      prompt: "A network has 7 knobs and you plan to try 10 settings of each. Then an eighth knob is added. What happens to the number of combinations to try?",
      options: ["It grows by 10 more combinations", "It grows by one eighth", "It becomes ten times as large"],
      answer: 2,
      explanation:
        "Each of the earlier combinations can be paired with any of the new knob's 10 settings, so the count multiplies: 10 million becomes 100 million. That multiplying is why nobody searches settings one by one, and why training instead asks each knob which way to turn from where it already is.",
      objective: 1,
    },
    {
      prompt: "On Seven knobs by hand, the curve follows the dots and you press Swap the two neurons. Six knob readings change. What happens to the Loss?",
      options: [
        "It rises, because knobs that were tuned have been moved",
        "It stays the same, because the output only adds the two neurons up",
        "It falls, because the swap undoes an earlier mistake",
      ],
      answer: 1,
      explanation:
        "Each neuron keeps its own three knobs; only their positions in the diagram trade. The output is a sum, and a sum does not depend on order, so the curve and the loss are untouched. Two different lists of seven numbers are the same network, which is why a single knob's value means nothing without the others.",
      objective: 2,
    },
    {
      prompt: "On More neurons, more shapes, the network with 1 hidden neuron ends with a high loss on the wave. Which change can bring its loss under 0.01?",
      options: ["Training it for ten times as many steps", "Giving it more hidden neurons", "Training it with a larger step size"],
      answer: 1,
      explanation:
        "One hidden neuron can only send on a single S-shaped step, so no setting of its four knobs draws a wave. Longer or bolder training still chooses among the same shapes. More hidden neurons add more steps for the output to combine, which puts the wave within reach of the knobs.",
      objective: 3,
    },
    {
      prompt: "On More neurons, more shapes, a network with 4 hidden neurons follows the wave from one scramble and ends with a high loss from another. What does the second run show?",
      options: [
        "Four hidden neurons cannot draw the wave",
        "The dots were different in the second run",
        "Enough knobs does not mean the steps will find a good setting",
      ],
      answer: 2,
      explanation:
        "The first run proves that a setting of those 13 knobs fits the wave. The second run started somewhere else, walked downhill from there, and stopped where the slopes ran out before reaching such a setting. A network can fail because its knobs cannot make the shape, or because training got stuck, and only the first is fixed by adding neurons.",
      objective: 3,
    },
    {
      prompt: "On Count the knobs, two hidden layers of 16 neurons sit side by side. You raise Neurons per layer to 64. Roughly how does the count of knobs between those two layers change?",
      options: ["It becomes about 4 times as large", "It becomes about 16 times as large", "It grows by 48 knobs for each of the layers"],
      answer: 1,
      explanation:
        "Every neuron in one layer is wired to every neuron in the next, so the wires number 16 × 16 = 256 at first and 64 × 64 = 4,096 afterwards. With the biases that is 272 knobs against 4,160, about 15 times as many. Making layers four times wider multiplies the wires between them by sixteen, which is how real networks reach billions of parameters.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState,
};

export default definition;
