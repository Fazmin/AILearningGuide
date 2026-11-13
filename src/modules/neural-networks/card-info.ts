import type { ModuleCardInfo } from "@app/module-sdk";

/**
 * Info-icon explanations for the six cards, keyed by the exact `label` of each <LabSurface>.
 * Every network here is real and every loss is measured, on dots made from short formulas; each
 * entry says which is which. The figures quoted are pinned in quoted.test.ts.
 */
const cardInfo: ModuleCardInfo = {
  "A machine with two knobs": {
    title: "The smallest machine that can be tuned",
    summary:
      "A machine that multiplies its input by one number and adds another. Those two numbers are its parameters, drawn as knobs, and every different pair of settings is a different machine. The loss says, as one number, how far the current machine is from the dots.",
    whatYouSee: [
      "The strip on the left is the machine: the input x enters, is multiplied by knob w, has knob b added, and leaves as the output. The three boxes show the number at each stage for the input chosen with Input x.",
      "Each knob is a dial that sweeps from −3, fully anticlockwise, through 0, straight up, to +3, fully clockwise. Its value is printed under it with its sign.",
      "The plot shows 17 dots, the outputs wanted at 17 inputs from −2 to 2, and the machine's output at every input as a line. A thin line joins each dot to the machine's line: the gap there. The filled marker is the input set by Input x.",
      "Loss prints the average of the squared gaps, and the bar fills as the loss falls from where the lab opens.",
    ],
    howItWorks: [
      "The output is `w × x + b`. The dots were made with w = 0.8 and b = 0.5, so that one setting passes through all of them.",
      "The loss is the mean squared error: square each of the 17 gaps, add them, divide by 17. The lab opens at w = −0.5, b = −1, where the loss is 4.785.",
      "The knobs click in tenths. On that grid only w = +0.80, b = +0.50 brings the loss to 0.000; one click away on b it reads 0.010, and one click away on w it reads 0.015.",
    ],
    controls: [
      "Knob w and Knob b: drag around a dial on the machine, focus it and use the arrow keys, or use the sliders underneath.",
      "Input x moves the marker and the three boxes. It does not change the loss, which always uses all 17 dots.",
    ],
    notice: [
      "Turning Knob w tilts the line about the point where it crosses x = 0; turning Knob b slides the whole line up or down without tilting it.",
      "When the tilt is right, every gap is the same length. When it is wrong, the gaps change in length from one end to the other.",
      "Changing Input x changes the output and nothing else. The knobs are what the machine keeps between one input and the next.",
    ],
    limits: [
      "In this lab: the dots lie exactly on a straight line made from a formula, with no noise, so a perfect setting exists and the loss can reach 0. Measured data would leave some loss at every setting.",
      "In this lab: only 17 inputs are scored, all between −2 and 2. The loss says nothing about what the machine does outside that range.",
      "In general: a model with two parameters can only draw straight lines. No setting of them fits a pattern that bends, which is the reason for the next card.",
    ],
  },

  "Seven knobs by hand": {
    title: "A network is small machines wired together",
    summary:
      "One input feeds two hidden neurons, and the output adds what they send on. Each neuron is a small machine with a knob on its incoming wire and a knob of its own, so the whole network has seven. Its curve can bend, which the two-knob machine could not do, and finding good settings by hand is already hard.",
    whatYouSee: [
      "The diagram: the input x on the left, two hidden neurons in the middle, the output on the right. A knob sits on each of the four wires (w₁, w₂, v₁, v₂) and on each neuron (b₁, b₂) and on the output (c).",
      "Inside each hidden neuron's box is the S-shaped curve it currently sends on, drawn over inputs from −2 to 2.",
      "The plot: 17 dots forming a hill, the network's output as a thick curve, and the gap at each dot. The two thin lines are each hidden neuron's share of the output, v₁ × its curve and v₂ × its curve.",
      "Loss and its bar, with the goal marked at 0.05, and the count of knobs: 4 on wires and 3 on neurons.",
    ],
    howItWorks: [
      "Neuron 1 sends `tanh(w₁ × x + b₁)` and neuron 2 sends `tanh(w₂ × x + b₂)`. tanh is a fixed S-shaped curve from −1 to 1. The output is `v₁ × neuron 1 + v₂ × neuron 2 + c`.",
      "The dots were made by this same network with w₁ = 2, b₁ = 2, w₂ = 2, b₂ = −2, v₁ = 1.5, v₂ = −1.5, c = −1, so a setting with a loss of exactly 0 is within the knobs' reach.",
      "The loss is the mean squared gap over the 17 dots. The lab opens at a scrambled setting where it is 1.790.",
      "Swap the two neurons trades w₁, b₁, v₁ with w₂, b₂, v₂. The output is a sum of the two neurons, so it cannot change. Scramble the knobs draws seven new values between −1.5 and 1.5 from a fixed list of seeds.",
    ],
    controls: [
      "Drag around any knob in the diagram, or focus it and use the arrow keys. Selected knob and Turn the selected knob do the same from the keyboard or a touch screen.",
      "Swap the two neurons and Scramble the knobs. Comparison worth running: note the Loss, press Swap the two neurons, and read the Loss again.",
    ],
    notice: [
      "w₁ makes neuron 1's S-curve steeper or flips it; b₁ slides it left and right; v₁ sets how tall its share is and which way up.",
      "A hill needs one neuron's share to rise on the left and the other's to fall on the right. The best spot for each knob moves when you turn another, which is what makes seven knobs so much harder than two.",
      "After a swap six of the seven readings are different and the curve has not moved.",
    ],
    limits: [
      "In this lab: the dots were made by a network of this exact shape, so a perfect fit exists. Real data is not made by the network that will be fitted to it.",
      "In this lab: knobs stop at ±3 and click in tenths when turned by hand, and there is one input and one output so the whole behaviour fits in one plot.",
      "In general: with more than a handful of parameters nobody can tune a network by hand or by trying settings in turn. Ten settings for each of seven knobs is already ten million combinations.",
      "In general: a hidden neuron's parameters have no fixed meaning. Swapping neurons, or flipping the sign of all of one tanh neuron's knobs, gives the same network with different numbers.",
    ],
  },

  "Let the machine turn the knobs": {
    title: "Training: a small turn for every knob, repeated",
    summary:
      "For each of the seven knobs the lab computes the slope of the loss: how the loss would change if that knob alone were turned a little. One step turns every knob against its slope at the same moment. Repeating that step is training, and it is the only way the knobs of a real network are ever set.",
    whatYouSee: [
      "The same diagram and the same seven knob settings as the previous card. A pink arrow on a knob shows the way it will turn on the next step; a longer arrow means the loss is more sensitive to that knob. A knob with no arrow has a slope near 0.",
      "The plot of the hill of dots and the network's curve, and the Loss bar with its goal at 0.01.",
      "The table: each knob's value now, its slope, and the value the dots were made with.",
      "Next turn works the step out for the selected knob. The chart draws the loss after every step taken so far.",
    ],
    howItWorks: [
      "The seven slopes are the gradient of the mean squared loss, computed exactly by the chain rule (backpropagation) through the network. A unit test checks them against nudging each knob.",
      "One step sets every knob to `knob − 0.2 × slope`, then stops it at ±3. The step size, 0.2, is fixed on this card.",
      "Run to the goal repeats the step until the loss is under 0.01 or 300 steps have been taken. From the setting the lab opens with, the loss goes 1.790, then 1.358 after one step, and is under 0.01 after 62.",
      "Turning a knob by hand, Scramble the knobs and Back to step 0 all restart the count of steps.",
    ],
    controls: [
      "Take one step, Take 20 steps, Run to the goal, Back to step 0, Scramble the knobs.",
      "Selected knob chooses which knob Next turn works out. Dragging a knob still turns it by hand.",
    ],
    notice: [
      "All seven knobs move on every step, each by its own amount. None of them is told its final value.",
      "The arrows shrink as the loss falls. Near a good setting every slope is close to 0, so the steps die away without anyone telling the machine to stop.",
      "From the opening setting the machine ends near w₁ = −1.4, b₁ = 1.3, v₁ = 1.7, which are not the numbers the dots were made with, although the curve matches. It has found a setting with the two neurons swapped and one of them sign-flipped.",
      "After Scramble the knobs the run reaches the goal at a different step and at different knob values: between step 46 and step 147 for the opening setting and the next nine scrambles.",
    ],
    limits: [
      "In this lab: seven knobs, 17 noise-free dots, every dot used on every step, and one fixed step size. The opening setting and the next nine scrambles all reach the goal; that is a property of this small problem, not of training.",
      "In this lab: the slopes are exact. A real training step estimates them from a small random batch of examples, so its steps are noisier.",
      "In general: a step uses only the slope where the knobs are now. It can stall on a flat stretch or settle at a setting that is worse than another one elsewhere, and it never knows which has happened.",
      "In general: a low loss on the training dots does not show the network is right anywhere else. That takes examples it was not trained on.",
    ],
  },

  "More neurons, more shapes": {
    title: "The knobs set which shapes are possible",
    summary:
      "Six networks, with one to six hidden neurons, are each trained on the same wave of dots by the same recipe. A network can only produce shapes its knobs can make, so the small ones are left with a high loss however they are trained, and the larger ones can follow the wave, though not from every start.",
    whatYouSee: [
      "The plot: 17 dots on a wave with crests at x = −1 and x = 1, the trained network's curve, the gap at each dot, and one thin line for each hidden neuron's share of the output.",
      "Hidden neurons chooses the network; its label also gives the knob count, 4 for one neuron up to 19 for six.",
      "The bars give the loss each of the six sizes is left with after training, and the chosen size is marked. Pressing a bar selects that size.",
      "The wall of small dials is every knob of the chosen network after training: straight up is 0, clockwise is positive.",
    ],
    howItWorks: [
      "Each network is 1 input → n hidden tanh neurons → 1 output, with `3n + 1` knobs. A neuron's share is `v × tanh(w × x + b)`, one S-shaped step, and the output is the sum of the shares plus a bias.",
      "The dots are `−1.2 × cos(π × x)` at 17 inputs from −2 to 2.",
      "Training is 600 steps from knobs scrambled between −1.5 and 1.5. Each step moves every knob by `0.9 × its last move − 0.05 × its slope`: plain steps with momentum. Knobs stop at ±8.",
      "Scramble the knobs moves to the next seed and retrains all six networks. Nothing else is random.",
    ],
    controls: [
      "Hidden neurons, or press a bar.",
      "Scramble the knobs. Comparison worth running: leave Hidden neurons at 4 and press it twice.",
    ],
    notice: [
      "At the opening start the six sizes end with losses of 0.648, 0.589, 0.299, 0.002, 0.003 and 0.003: three sizes miss, then four neurons fit.",
      "With one neuron the curve is a single gentle step whatever the start; its loss stays above 0.6 for the opening start and each of the next nine scrambles.",
      "After two presses of Scramble the knobs, four neurons end at 0.290 while three end at 0.018. The four-neuron network had the knobs it needed and got stuck on the way.",
      "No thin line looks like the wave. Each is one step; only their sum has crests.",
    ],
    limits: [
      "In this lab: one fixed recipe and 600 steps. A different step size or a longer run would change which starts get stuck, so a high bar for two or more neurons is not proof that the size cannot fit the wave. Only the one-neuron case is settled by its shape: a single step cannot rise and fall.",
      "In this lab: the loss is measured on the same 17 dots the network trained on. This card cannot show a network that has too many knobs and memorizes noise; the Stacking neurons lab does.",
      "In general: one hidden layer with enough neurons can approximate any continuous shape on a bounded range. That result says a good setting exists. It does not say how many neurons are enough, or that training will find the setting.",
    ],
  },

  "Count the knobs": {
    title: "Parameters are counted, and the count grows fast",
    summary:
      "In a fully connected network every neuron in one layer is wired to every neuron in the next, with a weight on each wire and a bias in each neuron. Multiply the two layer sizes, add one per neuron, and sum over the layers: that total is the parameter count quoted for a model.",
    whatYouSee: [
      "The sketch draws the network you built as columns of circles with every wire between the circles that are drawn. A layer too tall to draw shows six circles around a gap, with its true size printed underneath.",
      "The list under the sketch gives each layer's count as `wires + biases`, and Knobs adds them up.",
      "The bars compare your network with the seven-knob network of card 2 and four published networks. Each equal stretch of bar is ten times more knobs, so the bars compare the number of digits, not the amounts.",
      "The tiles give the count in words, how many of card 2's networks it equals, how long it would take to touch every knob once at one per second, and how many times larger GPT-3 is.",
    ],
    howItWorks: [
      "Knobs added by a layer = `neurons before × neurons in it + neurons in it`. The total is the sum over the hidden layers and the output layer.",
      "A small digit reader sets 784 inputs, two hidden layers of 16 and 10 outputs: 12,560 + 272 + 170 = 13,002 knobs.",
      "The largest network these sliders build, 784 inputs, six layers of 4,096 and 10 outputs, has about 87.2 million knobs.",
      "The published counts are the ones their papers report: about 60 thousand for LeNet-5, 60 million for AlexNet, 1.5 billion for the largest GPT-2, and 175 billion for GPT-3.",
    ],
    controls: [
      "Inputs (1, 2, 16 or 784), Hidden layers (1 to 6), Neurons per layer (2 to 4,096), Outputs (1 or 10).",
      "Card 2's network and A small digit reader set all four at once.",
    ],
    notice: [
      "With two or more hidden layers, making every layer twice as wide makes the count about four times as large: the wires between two layers are their sizes multiplied.",
      "Adding a hidden layer adds the same amount each time, so depth grows the count steadily while width grows it by squares.",
      "At one knob per second, the 13,002 knobs of the digit reader take 3.6 hours to touch once. GPT-3's take about 5,500 years.",
    ],
    limits: [
      "In this lab: nothing on this card is trained. It counts the knobs of a network shape; it does not say what a network of that shape could learn.",
      "In this lab: the count is for plain fully connected layers. The four published networks are not built that way: LeNet-5 and AlexNet reuse small sets of weights across an image, and GPT-2 and GPT-3 are transformers. Their counts are comparable as counts, not as designs.",
      "In general: a parameter count measures size, not ability. What a model can do also depends on its design, its training data and how long it was trained, and two models with the same count can differ widely.",
    ],
  },

  "Playground: train a network": {
    title: "Choose the dots, the network and the stepping, then watch",
    summary:
      "A real network trained in the browser on a set of dots you pick. You choose its size and how it steps; the lab scrambles its knobs and takes up to 600 training steps. Every curve, dial and loss on the card is computed from the network's knobs at the step shown.",
    whatYouSee: [
      "The plot: the chosen dots, the network's curve at the current step, and the gap at each dot.",
      "The wall of dials: every knob of the network, grouped by the layer it feeds. They turn as training runs.",
      "Loss with its goal at 0.01 on a bar where each third is a loss ten times smaller, and tiles for the knob count, steps taken, the loss now and the highest loss the run has reached.",
      "The chart: the loss after every step up to the current one. Training steps scrubs back and forth through the run.",
    ],
    howItWorks: [
      "The network is 1 input → one or two hidden layers of 1 to 6 tanh neurons → 1 output, so between 4 and 61 knobs.",
      "Dots: Slope is the line from card 1, Hill is card 2's hill, Wave is card 4's wave, and Step is −1 left of 0 and +1 right of it. Each has 17 dots and the loss is the mean squared gap.",
      "Every step moves each knob by `− step size × its slope`, plus `0.9 × its last move` when Momentum is On. Knobs start scrambled between −1.5 and 1.5 and stop at ±8.",
      "A run is fixed by the settings and the scramble, so the whole run is worked out once and Train only plays it. Changing any setting starts a new run from step 0.",
    ],
    controls: [
      "Dots, Hidden layers, Momentum, Neurons per layer, Step size (0.01 to 0.5).",
      "Train or Pause, Take 10 steps, Back to step 0, Scramble the knobs, and Training steps.",
      "Comparison worth running: Dots on Wave with one hidden layer, Neurons per layer at 2 and then at 4.",
    ],
    notice: [
      "As the lab opens, Wave with 2 neurons (7 knobs) ends at a loss of 0.589. With 4 neurons (13 knobs) the loss is under 0.01 from step 188.",
      "With those 4 neurons and Momentum Off, the same 600 steps end at 0.063: plain steps stall on the flat stretch that momentum carries the knobs across.",
      "With Step size at 0.5 and Momentum Off, the loss ends near 1,595 instead of near 0. Every step overshoots and the knobs run to their stops.",
      "Two hidden layers of 4 (33 knobs) pass the goal at step 76. Two hidden layers of 2 have 13 knobs, the same as one layer of 4, and end at 0.580: the same count of knobs, wired differently, is a different set of shapes.",
    ],
    limits: [
      "In this lab: across the opening start, the next seven scrambles and every setting on this card, no network with fewer than 13 knobs gets Wave under 0.01 in 600 steps. That is a measurement over those runs, not a proof about smaller networks.",
      "In this lab: one input, 17 noise-free dots used all at once on every step, and a loss measured on those same dots. There is no held-out data here, so the card cannot show overfitting.",
      "In general: real training differs in scale and in detail. It uses millions of examples in small random batches, optimizers such as Adam, and careful starting values, and it is judged on examples the network never trained on.",
      "In general: which run gets stuck depends on the starting knobs, and with real networks nobody can list the starts that will fail in advance.",
    ],
  },
};

export default cardInfo;
