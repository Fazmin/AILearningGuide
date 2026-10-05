import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Nested methods": {
    title: "Five rings, one nesting test, and what a ring cannot draw",
    summary:
      "The common modern stack is drawn as nested rings: AI contains machine learning, which contains deep learning, which contains neural generative models, which contains large language models. Each ring also names the mathematical object (a goal, a fitted f_θ, a stacked map, a distribution, a next-token model). That is a vocabulary for these labs, not a law. Generative is a property that crosses the deep-learning boundary, which nested rings cannot draw, so an n-gram is marked inside ML and outside DL rather than inside the Gen ring.",
    whatYouSee: [
      "A kicker that writes the common modern stack in words, a title Five rings, one nesting, and a badge with the selected ring label.",
      "Inspected method with five segments: AI, ML, DL, Gen, LLM.",
      "Five concentric circles plus a key of five sentences, one per ring, stating the mathematical claim. A small dot labelled n-gram sits in the ML ring below the deep-learning circle.",
      "Two notes: what the selected ring learns, and what the drawing cannot show, which is that the Gen ring holds neural generative models only.",
    ],
    howItWorks: [
      "The rings teach the common modern stack: LLM ⊂ neural generative ⊂ deep learning ⊂ machine learning ⊂ AI. Selecting a ring only stores `ring` in module state.",
      "The caption and the key sentence are authored for that ring. No classifier, model, or dataset is evaluated.",
      "A method belongs inside another ring only when every example of the inner method is still an example of the outer one. The Gen ring is neural generative models, so it passes that test for deep learning. A generative model that is not neural, such as an n-gram that samples from counted word pairs, fails it, which is why the marker sits outside DL.",
    ],
    controls: [
      "Inspected method, a click on a circle, or a click on a key row sets the inspected ring.",
      "Comparison worth running: select AI, then LLM, and notice the caption narrows from a goal to next-token prediction. Then look for the n-gram dot and say which ring it is in.",
    ],
    notice: [
      "The same product can sit in several rings at once: an assistant is an LLM, which is a neural generative model, which is deep learning.",
      "A rule-based checker can be AI without being machine learning, because nobody fitted it from examples.",
      "The n-gram is generative and not deep, so no single ring can hold it and also mean generative.",
    ],
    limits: [
      "In this lab: the five rings, their captions, and the position of the n-gram marker are authored. Nothing here classifies a shipped system, and the diagram does not draw every generative or non-deep method.",
      "In general: product copy still collapses these words. A chatbot demo does not make every method underneath it true or safe, and AGI, the hoped-for system that could do any cognitive task, is not one of the rings.",
    ],
  },

  "Learning problem": {
    title: "Search θ so f_θ(x) matches y on data",
    summary:
      "Three authored points and the hypothesis class of all lines ŷ = wx + b. Loss is mean squared error on those three points. The sliders choose θ, which is one line in data space and one point in parameter space; Take a downhill step subtracts 0.12 times the closed-form gradient of that same MSE. Generalization is named and not measured.",
    whatYouSee: [
      "Learning object, a five-stage flow — Data, Hypothesis class, Loss, Optimization, Generalization — with each stage's formula under its name. Clicking a stage rewrites the paragraph below it.",
      "Sliders Weight w and Bias b and a Take a downhill step button.",
      "Data space: the three points, the current line, a vertical residual at each point, and a shaded square on each residual whose area is proportional to (ŷ − y)².",
      "Parameter space: contour ellipses of equal L over w from 0 to 2 and b from 0 to 2.4, each labelled with its L; the current (w, b) as a dot; an arrow to where the next downhill step lands; a dotted path of the steps taken; and a green × at the least-squares bottom.",
      "A five-column table of x, y, ŷ, residual, and square, metrics L MSE, ∂L/∂w, ∂L/∂b, and Minimum L (the least-squares bottom), and a formula row that reprints the current line and the three squared residuals.",
    ],
    howItWorks: [
      "The points are (0.5, 1.4), (1.5, 2.1), (2.5, 3.2). Prediction is ŷ = w x + b. L = mean (ŷ − y)². ∂L/∂w = mean 2(ŷ − y)x and ∂L/∂b = mean 2(ŷ − y).",
      "Take a downhill step writes w ← clip(w − 0.12 ∂L/∂w) and b ← clip(b − 0.12 ∂L/∂b). Clips are the slider ranges, not a production optimizer.",
      "L is exactly quadratic in (w, b): L = L* + dᵀA d with d = θ − θ* and A = (1/3)[[Σx², Σx], [Σx, 3]]. So every contour is an ellipse, sampled from that formula. The bottom θ* is the least-squares line w = 0.90, b ≈ 0.883, where L* ≈ 0.0089.",
      "Learning object only changes which authored paragraph you read. It does not change the arithmetic.",
    ],
    controls: [
      "Weight w, Bias b, Take a downhill step, and the Learning object stages.",
      "Comparison worth running: set w to 0 and b to 0, read L ≈ 5.54, then press Take a downhill step a few times. The squares shrink in data space while the dot walks down the contours toward the green ×.",
    ],
    notice: [
      "The search is over two numbers. No extra variable stands for understanding or consciousness.",
      "A small L on these three points is a fit to the sample. A fourth x is not scored here.",
    ],
    limits: [
      "In this lab: three authored points, one affine hypothesis class, and an analytic gradient of MSE. There is no held-out point and no automatic differentiation. The path of steps is view state; dragging a slider starts a new one.",
      "In general: modern training uses the same objects — H, L, an optimizer, a hope of generalization — in far more dimensions, with a loss that may not be squared error.",
    ],
  },

  "What the word hides": {
    title: "Five cuts through the same word",
    summary:
      "Distinction chooses family, scope, job, memory, or not. Each cut is three authored tiles. The card does not classify a product; it names a confusion the rings alone cannot.",
    whatYouSee: [
      "Distinction, a five-way segmented control whose values are family, scope, job, memory, and not.",
      "A title sentence for the cut and a three-tile grid.",
      "A note that states the limit of that cut.",
    ],
    howItWorks: [
      "State stores `distinction` as one of five literals. The tiles are authored strings for that literal.",
      "Family separates symbolic rules, statistical fitting, and stacked differentiable maps. Scope separates a single-task map from a wide interface. Job separates p_θ(y|x) from search that may call a predictor. Memory separates training-set interpolation from error on new draws. Not names claims the math does not make.",
    ],
    controls: [
      "Distinction is the only control.",
      "Comparison worth running: family versus not. One names methods; the other refuses extra claims.",
    ],
    notice: [
      "AGI, artificial general intelligence, names a system that could do any cognitive task a person can. It is a goal that people define differently, not a measured property, so the scope cut treats it as something these labs never claim.",
      "Symbolic AI still counts as AI. It fails the machine-learning subset test because no θ was fitted.",
      "A fluent continuation can look like reasoning and still be next-token prediction.",
    ],
    limits: [
      "In this lab: five authored cuts and three tiles each. No system is scored as conscious, general, or understanding; the scope cut defines AGI as a goal with no agreed test and does not claim any system reaches it.",
      "In general: a deployed stack can mix a rule filter, a fitted ranker, and a generative model. Each piece still belongs to one of these cuts.",
    ],
  },

  "Era timeline": {
    title: "Eleven stops, each with a stall",
    summary:
      "Era is a stop on a timeline from the 1956 Dartmouth workshop to 2022 assistants. Each stop names the key idea of that moment and the reason it stopped being enough, so history reads as a sequence of limits rather than a victory lap. The early stops are the symbolic thread: the Dartmouth proposal, expert systems, and the two AI winters.",
    whatYouSee: [
      "A kicker `YEAR · title` and the heading What changed, and why it stalled.",
      "Era, a slider whose output prints the year and the stop's name.",
      "Two metrics: Year and Key idea.",
      "Two notes: Idea and Why it stalled, both authored for that stop.",
    ],
    howItWorks: [
      "The slider stores the stop's year, one of 1956, 1958, 1969, 1974, 1980, 1986, 1987, 1997, 2012, 2017, or 2022. They are the Dartmouth workshop, the perceptron, the Perceptrons critique, the first AI winter, expert systems, backpropagation, the second AI winter, LSTM, AlexNet, the transformer, and assistants.",
      "Each stop is a pair of sentences written for this card. No archive, paper, or training run is loaded.",
      "Early facts follow named sources: the 1955 Dartmouth proposal by McCarthy, Minsky, Rochester, and Shannon, reprinted in AI Magazine 27(4), 2006; the Wikipedia articles on the Lighthill report and on AI winters, which cite Lighthill's 1973 report and the 1987 collapse of the Lisp-machine market; and the Wikipedia article on XCON, which cites McDermott's 1980 paper. The winter dates are approximate: about 1974 to 1980 and from about 1987.",
    ],
    controls: [
      "Move Era one stop at a time and read both notes before you move again.",
      "Comparison worth running: 1980 versus 2012. Expert systems fit no parameters and stalled on maintenance; AlexNet fit millions and stalled on needing labeled data for each task.",
    ],
    notice: [
      "Every stop has a stall. The timeline is not a claim that the last stop solved the field.",
      "Expert systems are AI without being machine learning: the knowledge was written, not fitted.",
      "2022 names post-training on a next-token model. Fluency is still not a truth check.",
    ],
    limits: [
      "In this lab: eleven authored stops. Dates are landmarks, not a complete history, and the card shows two threads, symbolic AI and neural nets, not every approach. Nothing on the card trains a model from that year.",
      "In general: a method can stall for funding, data, compute, or a mathematical limit, and the winters were funding downturns as much as technical ones. The next architecture does not erase the earlier failure mode.",
    ],
  },

  "Sort leftover terms": {
    title: "Place each leftover in its tightest ring",
    summary:
      "Five leftover systems must be assigned to the tightest ring that still contains them. A wrong ring is stored, then corrected in the note, so the lesson is the subset test rather than a free-form label.",
    whatYouSee: [
      "A kicker `N of 5 in the tightest ring` and a badge unplaced, correct, or corrected for the selected leftover.",
      "Leftover terms, a listbox of five buttons. Each row shows the term and whether it is unplaced, placed, or corrected.",
      "Place in ring, five segments, and a Place button that names the selected leftover.",
      "Three metrics: Placed, Tightest-ring matches, and Nesting depth of the selected term's correct ring.",
      "A note that either explains the correct ring or reports the correction.",
    ],
    howItWorks: [
      "State stores `term`, `destination`, and `placements` as strings like `chess:ai`. Placing a term overwrites its previous guess.",
      "Hand-written chess rules belong in AI, linear regression in ML, a convnet classifier in DL, a face GAN in generative, and a next-token assistant in LLM.",
      "A guess that is an outer ring still counts as wrong here. The tightest ring is the one the card scores.",
      "Arrow keys move the listbox selection. Enter places the selected term. Drag onto Place in ring is optional and writes the same state.",
    ],
    controls: [
      "Select a leftover in Leftover terms, choose Place in ring, then press Place.",
      "Comparison worth running: place Linear regression on house prices in DL, read the correction, then place it in ML.",
    ],
    notice: [
      "A GAN is generative without being an LLM. Fluency of a face sample is not token prediction.",
      "Chess rules can be AI and still have zero learned parameters.",
    ],
    limits: [
      "In this lab: five authored leftovers and one tightest ring each. Dragging is cosmetic; the listbox is the supported path.",
      "In general: real products sit in several rings at once. The tightest ring names the method, not the marketing category.",
    ],
  },

  "What is the learning signal": {
    title: "Where the training target comes from",
    summary:
      "Eight authored scenarios are sorted under the signal that trains the model: human labels (supervised), structure in raw data (unsupervised), labels made from the data itself (self-supervised), or a reward after acting (reinforcement). Each placement is checked at once and corrected with a reason. A separate readout names the output type, classification or regression, because the signal and the output type are different questions.",
    whatYouSee: [
      "A kicker `N of 8 sorted under the signal that trains them` and a badge unplaced, correct, or corrected for the selected scenario.",
      "Four tiles, one per signal, each naming where the answer comes from. The signal names are text, so no meaning rests on colour.",
      "Learning scenarios, a listbox of eight buttons. Each row shows a short name, a one-sentence description, and whether it is unplaced, placed, or corrected.",
      "Learning signal, four segments, and a Place button that names the selected scenario.",
      "Three metrics: Placed, Signal matches, and Output type for the selected scenario.",
      "Two notes: the verdict with its reason, and a reminder that output type is a separate question from the signal.",
    ],
    howItWorks: [
      "State stores `scenario`, `signal`, and `signalPlacements` as strings like `text:self`. Placing a scenario overwrites its earlier guess, and anything malformed is dropped when the state is loaded.",
      "Each scenario has one authored best signal and one authored output type. House prices and spam emails are supervised; customer records and sensor readings are unsupervised; raw sentences and blanked-out photos are self-supervised; the win-or-loss game and the charger robot are reinforcement.",
      "Output type is read off the answer the model gives: a category is classification, a number is regression, a grouping or score with no predicted answer is no target, and a choice of move is an action. Raw sentences are classification over a vocabulary; blanked-out photos are regression on pixel values.",
      "Arrow keys move the listbox selection. Enter places the selected scenario under the chosen signal. Dragging a row onto the Learning signal control is optional and writes the same state.",
    ],
    controls: [
      "Select a scenario in Learning scenarios, choose Learning signal, then press Place.",
      "Comparison worth running: place Raw sentences under supervised and read the correction, then under self-supervised. Then select Blanked-out photos and compare its Output type with that of Raw sentences: the same signal, different output types.",
    ],
    notice: [
      "Raw sentences and Blanked-out photos share a signal but not an output type. Self-supervised is not a synonym for classification.",
      "Customer records and Sensor readings have no answer column, so their output type reads no target. There is nothing to score accuracy against.",
      "The next word is already in the text, which is why next-word prediction needs no human labeling and can train on far more text than anyone could annotate.",
    ],
    limits: [
      "In this lab: eight authored scenarios, each with a single best signal and a single output type, and a wrong placement is corrected by a table lookup rather than by a model. Nothing is trained, and the card does not classify a real system.",
      "In general: real projects blend signals. Semi-supervised learning mixes a few labeled examples with many unlabeled ones, and a chat assistant is pretrained self-supervised, tuned on human-written answers, then trained on human-preference rewards.",
    ],
  },
};

export default cardInfo;
