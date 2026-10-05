import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_ERA_YEAR, eraYearFromState } from "./eras";
import { SCENARIO_IDS, SIGNAL_IDS, parseSignalPlacements } from "./signals";

const RINGS = ["ai", "ml", "dl", "gen", "llm"] as const;
const TERMS = ["chess", "linreg", "convnet", "gan", "chat"] as const;
const DISTINCTIONS = ["family", "scope", "job", "memory", "not"] as const;
const TERM_PLACEMENT = new RegExp(`^(${TERMS.join("|")}):(${RINGS.join("|")})$`);

/**
 * Version 4 replaces `era` (an index into seven stops) with `eraYear`, because the timeline now has eleven stops
 * and the early ones come first. A version 3 payload has only `era`, which hydrateState translates to that
 * stop's year.
 */
const initialState: ModuleState = {
  ring: "ai",
  eraYear: DEFAULT_ERA_YEAR,
  term: "chess",
  destination: "ai",
  placements: [],
  stage: 1,
  weight: 1,
  bias: 0.9,
  distinction: "family",
  scenario: "house",
  signal: "supervised",
  signalPlacements: [],
};

const asRing = (value: unknown, fallback: string) =>
  typeof value === "string" && (RINGS as readonly string[]).includes(value) ? value : fallback;
const asTerm = (value: unknown, fallback: string) =>
  typeof value === "string" && (TERMS as readonly string[]).includes(value) ? value : fallback;
const asDistinction = (value: unknown, fallback: string) =>
  typeof value === "string" && (DISTINCTIONS as readonly string[]).includes(value) ? value : fallback;
const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;
const clampNumber = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const definition: ModuleDefinition = {
  id: "module-29-what-ai-is",
  slug: "what-ai-is",
  title: "What AI actually is",
  group: "groundwork",
  order: 1,
  icon: "Atom",
  accent: "#3d6fd8",
  prerequisites: [],
  estimatedMinutes: 14,
  steps: [
    "Inspect a ring",
    "Fit a line by hand",
    "Name a distinction",
    "Scrub the timeline",
    "Pick a leftover term",
    "Place it in a ring",
    "Name the learning signal",
  ],
  stepInstructions: [
    "Use Inspected method, or click a circle, and read what that ring actually learns.",
    "On Learning problem, move Weight w and Bias b, then press Take a downhill step and watch L, the residual squares, and the dot walking down the contour map.",
    "Set Distinction to family, then to not, and read what the math does not claim.",
    "Move Era from the Dartmouth workshop to assistants and, at each stop, compare the key idea with why that era stalled; stop on Expert systems and note that no parameters were fitted.",
    "In Leftover terms, arrow to Hand-written chess rules and notice it is still unplaced.",
    "Set Place in ring to AI, press Place, then try Linear regression on ML versus DL and read the correction.",
    "In What is the learning signal, arrow to Raw sentences, set Learning signal to supervised and press Place to read the correction, then set it to self-supervised; select Blanked-out photos and compare its Output type.",
  ],
  stateVersion: 4,
  tagline: "Treat AI as a family of function-approximators, nest the modern methods, and name what the word does not claim.",
  objectives: [
    "Place machine learning, deep learning, and large language models inside AI rather than treating them as synonyms",
    "Name the learning problem as a hypothesis class, a loss, a search over parameters, and a generalization gap",
    "Separate symbolic methods, statistical fitting, and deep stacks, and say what those objects are not",
    "Name where a model's training signal comes from (human labels, structure in raw data, labels made from the data itself, or reward) and tell classification from regression by the kind of output",
  ],
  glossary: [
    {
      term: "Artificial intelligence",
      definition: "Any method that performs a task that used to need human judgment. The label names a goal, not a single algorithm, so a rule list and a neural net can both count. Symbolic AI, which runs on written rules, search, or logic, is AI but sits outside machine learning because nobody estimated θ from a table. AGI, artificial general intelligence, is the hoped-for or feared goal of a system that could do any cognitive task a person can; it is defined in different ways and has no agreed test, so it is not a measured property, and nothing in these labs claims it.",
    },
    {
      term: "Machine learning",
      definition: "A method that fits parameters from examples instead of writing the decision rule by hand. The fit is only as good as the data and the objective you chose.",
    },
    {
      term: "Deep learning",
      definition: "Machine learning that stacks differentiable layers so later layers can reuse earlier features. Depth is a wiring choice, not a guarantee of understanding.",
    },
    {
      term: "Generative model and large language model",
      definition: "A generative model is trained to produce new samples from a distribution, such as text, images, or audio. The Gen ring in the diagram holds the neural ones; an n-gram is generative without being deep and sits outside it. A large language model is a generative model trained to predict the next token in text, then often adapted into an assistant. Generating a fluent sample is not the same as checking that the sample is true.",
    },
    {
      term: "Hypothesis class",
      definition: "The set of functions you are willing to search, written H. On this page H is every line ŷ = wx + b. A different H (a tree, a deep net) is a different search.",
    },
    {
      term: "Loss",
      definition: "A number that says how wrong the current parameters are on the data you are fitting. Training searches for parameters that make this number small. It is not automatically the number you should publish.",
    },
    {
      term: "Generalization",
      definition: "Low error on new examples drawn from the same process as the training data. A small training loss does not, by itself, measure it.",
    },
    {
      term: "Model, training, and inference",
      definition: "A model is a fixed function f_θ: once its numbers θ are set, each input x maps to one output ŷ, or to one distribution p_θ(y|x). Training searches for θ by lowering the loss on examples; inference runs the finished model on a new input with θ held still. Finding θ so that f_θ(x) is close to a target y is function approximation, which is most of modern AI and not a model of a mind.",
    },
    {
      term: "Parameter and hyperparameter",
      definition: "A parameter is a number inside the model that training sets from data. Here w and b in ŷ = wx + b are the parameters; a large language model has billions. A hyperparameter is a setting you choose before the search starts, so training does not learn it. Here the 0.12 step size is one; later labs add learning rate, batch size, and epoch count.",
    },
    {
      term: "Supervised learning",
      definition: "Fitting a model on input-output pairs that someone labeled. The labels decide what counts as correct, so a biased label set becomes a biased rule. Semi-supervised learning is the same idea when only a few examples carry labels and many do not.",
    },
    {
      term: "Unsupervised learning",
      definition: "Learning from data that has no answer column. The model finds structure in the inputs themselves, such as clusters of similar records or readings far from the usual pattern. With no target there is nothing to score accuracy against, so a person still has to judge whether the structure means anything.",
    },
    {
      term: "Self-supervised learning",
      definition: "Learning where the label is made from the data itself: hide part of the data, such as the next word of a sentence or a blanked patch of a photo, and ask the model to restore it. The model still trains against labels, but they come free, so the training set can be as large as the raw collection. This is how large language models are pretrained.",
    },
    {
      term: "Reinforcement learning",
      definition: "Learning from a reward received after acting, with no example of the right action. The learner tries actions, sees the reward or penalty, and has to work out which earlier choices earned it. A reward scores an outcome; it is not a label for each step.",
    },
    {
      term: "Classification and regression",
      definition: "Two kinds of output for a model that predicts a target. Classification returns one category from a fixed set, such as spam or not spam, or the next word out of a vocabulary. Regression returns a number on a scale, such as a price or a pixel value. The output type is a separate axis from the learning signal: next-word prediction is self-supervised classification.",
    },
  ],
  references: [
    {
      authors: "Stuart Russell and Peter Norvig",
      title: "Artificial Intelligence: A Modern Approach, 4th edition",
      source: "Pearson",
      year: 2020,
      url: "https://aima.cs.berkeley.edu/",
      note: "The standard university textbook on AI. Its first chapter compares definitions of AI and settles on building agents that act well, which covers rule-based programs and learned models alike.",
    },
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Chapter 1 draws AI, machine learning, and deep learning as nested circles, the same picture as this lab's rings. Chapter 5 sets out the data, loss, search, and generalization that make up the learning problem.",
    },
    {
      authors: "John McCarthy, Marvin L. Minsky, Nathaniel Rochester, and Claude E. Shannon",
      title: "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence, August 31, 1955",
      source: "Reprinted in AI Magazine 27(4)",
      year: 2006,
      url: "https://doi.org/10.1609/aimag.v27i4.1904",
      note: "The proposal that named the field. It bets that every feature of learning and intelligence can be described precisely enough for a machine to simulate it, which is the first stop on the Era timeline.",
    },
    {
      authors: "Frank Rosenblatt",
      title: "The perceptron: A probabilistic model for information storage and organization in the brain",
      source: "Psychological Review 65(6), 386–408",
      year: 1958,
      url: "https://www.semanticscholar.org/paper/5d11aad09f65431b5d3cb1d85328743c9e53ba96",
      note: "The perceptron paper. It describes a machine that adjusts its connection weights from labeled examples, an early case of learning a rule instead of writing one by hand.",
    },
    {
      authors: "James Lighthill",
      title: "Artificial Intelligence: A General Survey",
      source: "Science Research Council (UK), Artificial Intelligence: A Paper Symposium; full text at the Chilton Computing archive",
      year: 1973,
      url: "https://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm",
      note: "The review that judged AI had not kept its early promises, partly because methods that worked on small problems blew up on large ones. It led to UK funding cuts in the first AI winter.",
    },
    {
      authors: "Judith Bachant and John McDermott",
      title: "R1 Revisited: Four Years in the Trenches",
      source: "AI Magazine 5(3)",
      year: 1984,
      url: "https://doi.org/10.1609/aimag.v5i3.445",
      note: "A first-hand account of XCON, the expert system that configured computers at Digital Equipment Corporation. It is AI with no fitted parameters, and it shows how much work a growing rule base took to keep correct.",
    },
    {
      authors: "Yann LeCun, Yoshua Bengio, and Geoffrey Hinton",
      title: "Deep learning",
      source: "Nature 521, 436–444",
      year: 2015,
      url: "https://doi.org/10.1038/nature14539",
      note: "A short review by three pioneers of the field. It explains why stacked layers can learn their own features from raw data, and what changed once large datasets and GPUs arrived.",
    },
    {
      authors: "Richard S. Sutton and Andrew G. Barto",
      title: "Reinforcement Learning: An Introduction, 2nd edition",
      source: "MIT Press, free to read online",
      year: 2018,
      url: "http://incompleteideas.net/book/the-book-2nd.html",
      note: "The standard textbook on learning from reward. Chapter 1 sets it apart from learning from labeled examples and from finding structure in unlabeled data, the split the learning-signal sort tests.",
    },
    {
      authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2203.02155",
      note: "The InstructGPT paper. A pretrained language model is tuned on human-written answers, then trained with reinforcement learning from human preference rankings, which is why one chat assistant blends three learning signals.",
    },
    {
      authors: "Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, et al.",
      title: "Levels of AGI for Operationalizing Progress on the Path to AGI",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2311.02462",
      note: "A proposal for grading progress toward AGI by performance and generality. It opens by comparing definitions of AGI that disagree, which is why this lab treats AGI as a goal with no agreed test.",
    },
  ],
  checkpoint: [
    {
      prompt: "A GAN draws new face images from a learned distribution. Which placement in the rings is correct?",
      options: [
        "Generative and deep learning, but not an LLM, since it does not predict the next token",
        "An LLM, because every generative model is trained to predict the next token in text",
        "Machine learning but not deep learning, because drawing samples needs no stacked layers",
        "Outside machine learning, because nobody wrote a label for each generated face",
      ],
      answer: 0,
      explanation: "The rings nest the common modern stack, but a GAN is generative without being an LLM: an LLM's training job is predicting the next token in text. A GAN's generator is a stack of differentiable layers, so it sits in deep learning, and fitting parameters from examples is what makes it machine learning. Labels are not required for that.",
      objective: 0,
    },
    {
      prompt: "On Learning problem you start at Weight w = 0 and Bias b = 0 and press Take a downhill step several times. What happens?",
      options: [
        "The three data points slide toward the line, which is what makes the residuals shrink",
        "The dot jumps straight to the bottom of the bowl on the first press, whatever the step size",
        "L stays the same until the slopes are exactly zero, then drops all at once",
        "L falls and the dot walks toward the bottom as the slopes flatten toward zero",
      ],
      answer: 3,
      explanation: "A downhill step moves (w, b), the dot on the parameter map, by minus 0.12 times the gradient of L. The data never move. Each press shrinks the squared residuals, so L falls, and the slopes flatten as the dot nears the bottom of the bowl. A fixed small step means several presses, not one jump.",
      objective: 1,
    },
    {
      prompt: "A line ŷ = wx + b is fitted by making mean squared error small on three points. What is still not guaranteed?",
      options: [
        "That the mean squared error is a well-defined number on these three points",
        "That a new x from the same process will also have a small prediction error",
        "That the hypothesis class contains at least one straight line through them",
      ],
      answer: 1,
      explanation: "Training loss is a reading of the fitted sample. Generalization is error on new draws from the same process. A small L on three authored points does not measure that, and nothing in the search requires consciousness or folk understanding.",
      objective: 1,
    },
    {
      prompt: "A hand-written checker applies a fixed list of rules to a chess position. How does it differ from fitting a line to data?",
      options: [
        "It must be machine learning, because any program that decides has learned from experience",
        "It is deep learning, because several rules are applied one after another",
        "It generalizes to new positions by construction, because written rules cannot be wrong",
        "It can be AI with no fitted parameters, because nobody estimated θ from examples",
      ],
      answer: 3,
      explanation: "Artificial intelligence names a goal, so a rule list counts. Machine learning means parameters fitted from examples, and a rule list has no θ to fit. Applying rules in sequence is not stacking differentiable layers, and a rule can still fail on a position its author never imagined.",
      objective: 2,
    },
    {
      prompt: "Next-word prediction can train on a huge pile of raw text that nobody labeled. Why?",
      options: [
        "It is unsupervised, so no target exists and the model only clusters similar words",
        "The next word is already in the text, so each sentence supplies its own answer key",
        "It is reinforcement learning, because the reader rewards each word after it appears",
      ],
      answer: 1,
      explanation: "Hiding the next word turns the text into labeled pairs for free: the context is x and the word that follows is y. That is self-supervised learning, and it scales with the amount of raw text rather than with the number of annotators. Unsupervised methods have no target at all, and no reader supplies a reward here.",
      objective: 3,
    },
    {
      prompt: "A model fills in photo patches that were blanked out by predicting the hidden pixel values. Which pair of descriptions fits?",
      options: [
        "Self-supervised signal and a regression output, since the answer is hidden numbers",
        "Supervised signal and a classification output, because a person labeled each patch",
        "Unsupervised signal and a regression output, because no target is being predicted",
        "Reinforcement signal and an action output, because a good repair earns a reward",
      ],
      answer: 0,
      explanation: "The hidden pixels are part of the original photo, so the data supplies its own label: that is the self-supervised signal. The output is a number for each pixel, so the task type is regression. The signal says where the target comes from; classification and regression say what kind of answer it is, and the two are separate choices.",
      objective: 3,
    },
    {
      prompt: "On Era timeline, the perceptron (1958) and the expert systems of 1980 both count as AI. What separates them?",
      options: [
        "A perceptron is the only one that is AI, since a rule list has no parameters and so cannot count",
        "Expert systems run rules people wrote, while a perceptron fits weights from labeled examples",
        "Both fit parameters from examples, and they differ only in the hardware they ran on",
        "Expert systems are deep learning, because several rules are applied in sequence and stacked",
      ],
      answer: 1,
      explanation: "An expert system such as XCON holds specialist knowledge as written rules, so no parameters are fitted and it is AI without being machine learning. A perceptron learns weights from labeled examples, which makes it machine learning. Applying rules in sequence is not a stack of differentiable layers, and AI names a goal, so a rule list counts.",
      objective: 2,
    },
    {
      prompt: "A chat assistant answers questions on many topics fluently. What do these labs say about calling it AGI?",
      options: [
        "It qualifies, because training on text from many topics makes a model skilled at every task",
        "It cannot qualify, because every large language model is narrow by definition and always will be",
        "It is a broad tool; AGI is a goal with no agreed test, and fluency does not measure it",
        "It qualifies, because answering any question in words is the definition of general intelligence",
      ],
      answer: 2,
      explanation: "AGI names a system that could do any cognitive task a person can, and people define it differently, so it is a goal rather than a measured property. A wide interface such as language makes a model look general; its training job is still next-token prediction plus later wraps. The rings do not contain AGI, and these labs neither claim it nor rule it out forever.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      const placements = Array.isArray(parsed.placements)
        ? parsed.placements.filter((item): item is string => typeof item === "string" && TERM_PLACEMENT.test(item))
        : [];
      return {
        ring: asRing(parsed.ring, initialState.ring as string),
        eraYear: eraYearFromState(parsed.eraYear, parsed.era),
        term: asTerm(parsed.term, initialState.term as string),
        destination: asRing(parsed.destination, initialState.destination as string),
        placements,
        stage: Math.round(clampNumber(parsed.stage, 1, 0, 4)),
        weight: clampNumber(parsed.weight, 1, 0, 2),
        bias: clampNumber(parsed.bias, 0.9, 0, 2.4),
        distinction: asDistinction(parsed.distinction, initialState.distinction as string),
        scenario: asMember(parsed.scenario, SCENARIO_IDS, initialState.scenario as string),
        signal: asMember(parsed.signal, SIGNAL_IDS, initialState.signal as string),
        signalPlacements: parseSignalPlacements(parsed.signalPlacements),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
