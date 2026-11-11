import type { ModuleCardInfo } from "@app/module-sdk";

/**
 * Info-icon explanations for the six cards, keyed by the exact `label` of each <LabSurface>.
 * Every curve and map in this lab is a closed-form function, and each entry says so.
 */
const cardInfo: ModuleCardInfo = {
  "Nudge the dial": {
    title: "A slope is a ratio that settles",
    summary:
      "Nudge an input, measure how far the output moves, and divide. As the nudge shrinks the ratio stops changing, and the number it settles on is the slope of the curve at that point: the derivative.",
    whatYouSee: [
      "The curve is the meter L at every dial setting w from 0 to 6: a bowl with its bottom at w = 3, where L = 1.",
      "The large dot is the current setting and the small dot is the setting after the nudge. The blue leg is the nudge in the dial; the orange leg is the change it causes in the meter. Both changes are repeated as thick bars on the two axes.",
      "The solid pink line passes through both dots and its slope is the ratio. The dashed grey line is the line the pink one settles on. Once the nudge is below 1, a faint dotted line marks where the pink line was at a nudge of 1.",
      "The magnified box redraws the two dots at whatever zoom makes the nudge fill it, so a nudge of 0.01 is as easy to read as a nudge of 1. The dashed square on the main plot is the patch it shows.",
      "Ratio prints change in meter ÷ change in dial. The tiles give the meter at both settings, the ratio, and the slope it settles on.",
    ],
    howItWorks: [
      "The meter is L(w) = (w − 3)² + 1.",
      "The ratio is [L(w + nudge) − L(w)] ÷ nudge. For this curve that works out to exactly 2(w − 3) + nudge.",
      "The slope it settles on is the derivative, L′(w) = 2(w − 3), worked out by hand from the formula.",
    ],
    controls: [
      "Dial w moves the point; you can also drag anywhere on the plot, or focus the knob on the axis and use the arrow keys.",
      "Nudge size steps through 1, 0.5, 0.25, 0.1, 0.05 and 0.01. With Dial w at 4, read the ratio at each stop.",
    ],
    notice: [
      "Halving the nudge roughly halves the change in the meter, so the ratio barely moves.",
      "The curve inside the magnified box straightens as the nudge shrinks. On a straight line the ratio does not depend on the nudge at all.",
      "The gap between the ratio and the slope equals the nudge, so a nudge of 1 is off by 1.",
    ],
    limits: [
      "In this lab: the curve is an authored formula with one input, and every number is exact arithmetic on it. Nothing is measured from a model.",
      "In general: nudging and dividing gives only an estimate, and it needs one extra run of the function for every input. Training software does not find slopes this way; it applies the chain rule to the program that computes the loss.",
    ],
  },
  "A slope at every point": {
    title: "The slope is a curve of its own",
    summary:
      "Measure the slope at every dial setting and you get a second curve. Its sign says whether the meter is rising or falling, its size says how steeply, and its zeros mark level places, which are not always bottoms.",
    whatYouSee: [
      "The top picture is the meter L against the dial w. The bottom picture is the slope of that curve at the same w, with a darker zero line.",
      "Hatched bands mark stretches where the meter falls, and plain tinted bands mark where it rises. The same bands run through both pictures and are labelled along the top.",
      "A dotted line drops from each flat spot on the top curve to the place where the slope curve meets its zero line. Each flat spot is named: bottom, top or shelf.",
      "The vertical line is the current dial setting. On the top curve it carries a short pink stub with the slope of the curve there; on the bottom picture a pink bar reaches from zero to the slope's value.",
      "The sentence under the slider reads the slope in words, and the tiles give the meter, the slope, whether the meter is rising, and which way is downhill.",
    ],
    howItWorks: [
      "Two valleys is L(w) = 0.2u² + 0.9 cos(1.8u) − 0.3u + 2.2 with u = w − 3, and its slope is 0.4u − 1.62 sin(1.8u) − 0.3.",
      "A shelf is L(w) = 5 + 0.15u³, with slope 0.45u².",
      "Flat spots are found by scanning the slope for sign changes and closing in by bisection. Falling then rising is a bottom, rising then falling is a top, and a slope that touches zero without changing sign is a shelf.",
    ],
    controls: [
      "Dial w moves the cursor through both pictures; dragging on either picture or the arrow keys on the axis knob do the same. The dial snaps onto a flat spot when it comes within 0.03 of one.",
      "Meter curve switches between Two valleys and A shelf.",
    ],
    notice: [
      "Wherever the top curve falls, the slope curve is below its zero line, and wherever it rises, the slope curve is above.",
      "On Two valleys the slope is zero three times, and the middle one is the top of the hump.",
      "On A shelf the slope touches zero at w = 3 and is positive on both sides, so the level place is neither a bottom nor a top.",
    ],
    limits: [
      "In this lab: both curves are authored formulas, chosen so that one picture holds a bottom, a top and a second bottom, and the other holds a shelf.",
      "In general: a zero slope is a necessary sign of a minimum, not a sufficient one. With many inputs, level places that are not minima are the common case.",
    ],
  },
  "Two dials, one slope each": {
    title: "One slope per dial makes an arrow",
    summary:
      "With two dials there is a slope along each, measured while the other dial is held still. The pair of slopes is a vector called the gradient. It points straight uphill, crosses the contour line at a right angle, and is longer where the ground is steeper.",
    whatYouSee: [
      "A map seen from above: dial w runs left to right and dial b runs bottom to top. Each grey line joins settings with the same meter reading, printed on the line.",
      "The red point is the current setting, and the red line is the contour through it. A small hollow dot marks the centre of the map.",
      "With w only or b only, a dashed line shows the path the point can take when one dial turns, and a leg shows the slope along it: blue for w, purple for b. The leg points the way that dial raises the meter.",
      "With Both together, the legs are drawn tip to tail and the pink arrow from the point to their end is the gradient. A small square marks the right angle between it and the contour, and a short dashed arrow points the opposite way, downhill.",
      "The tiles give the meter, the two slopes, and the arrow's length. The Gradient line writes the two slopes as a list.",
    ],
    howItWorks: [
      "Round bowl is L = 1 + ½[(w − 3)² + (b − 3)²], Long valley is L = 1 + ½[0.25(w − 3)² + 2(b − 3)²], and Saddle is L = 5 + ½[(w − 3)² − (b − 3)²].",
      "Each slope is the derivative with respect to one dial with the other held fixed. On Long valley they are 0.25(w − 3) and 2(b − 3).",
      "The arrow is drawn 0.6 map units long for each unit of slope. Its length on the Steepness tile is the square root of the sum of the two slopes squared.",
      "Contour lines are traced by marching squares on a 48 by 48 grid over the map.",
    ],
    controls: [
      "Drag the point on the map, or use Dial w and Dial b. With the point focused, the arrow keys move it by 0.1.",
      "Dials turning chooses which legs are drawn: w only, b only, or Both together.",
      "Surface switches between Round bowl, Long valley and Saddle.",
    ],
    notice: [
      "On Round bowl the arrow points directly away from the lowest point. On Long valley at w = 1, b = 4 it points almost straight up the wall instead, so the downhill direction misses the lowest point.",
      "The arrow shrinks to nothing at the centre of every surface, including Saddle, where the centre is not a low point.",
    ],
    limits: [
      "In this lab: the three surfaces are authored quadratic formulas over two dials, so their contours are exact circles, ellipses and hyperbolas.",
      "In general: a gradient has one entry per parameter, often billions, and cannot be drawn. It describes only the ground near the current setting, and a short distance away it can point somewhere quite different.",
    ],
  },
  "Dials that drive dials": {
    title: "Along a chain, slopes multiply",
    summary:
      "When a dial reaches the meter through an intermediate value, each stage multiplies the change passing through it by its own slope. The overall slope is the product, so one nearly flat stage silences everything before it.",
    whatYouSee: [
      "Left: machine A, the middle reading u against the dial w. Right: machine B, the meter L against u. The purple marks on machine A's vertical axis and machine B's horizontal axis are the same value of u.",
      "Each curve carries a point where that machine is working and a short pink stub with its slope there.",
      "The row underneath is the chain: dial, machine A's slope, middle reading, machine B's slope, meter. Each value box has a bar for how far it moves when the dial is nudged by 0.1, and all three bars share one scale.",
      "Overall slope prints the product of the two slopes, with the change the product predicts and the change the two formulas really give.",
    ],
    howItWorks: [
      "Machine A is u = (w + 3)² ÷ 4, with slope (w + 3) ÷ 2. Machine B is L = 10 ÷ (1 + e^(−0.8(u − 9))), an S-shaped curve with slope 8s(1 − s), where s is L ÷ 10.",
      "The bars show nudge × slope of A, then that × slope of B. Overall slope is slope of A × slope of B.",
      "The real change is computed by running w + 0.1 through both formulas and subtracting.",
    ],
    controls: [
      "Dial w sets where both machines work; you can also drag on machine A's plot or use the arrow keys on its axis knob. Compare w = 3 with w = 5.",
    ],
    notice: [
      "At w = 3 the slopes are 3 and 2 and the overall slope is 6, not 5: they multiply.",
      "At w = 5 machine A is steeper but machine B is nearly flat, so the last bar almost disappears. The meter reads close to its maximum there, so a tiny slope does not mean a low meter.",
      "The predicted and real changes differ slightly, because a slope is exact only for a vanishingly small nudge.",
    ],
    limits: [
      "In this lab: two authored one-input functions stand in for the layers of a network, and the nudge is fixed at 0.1.",
      "In general: a network's chain has many links and each link is a whole grid of slopes, not one number. Backpropagation carries out this multiplication from the loss backwards. Many small factors make gradients vanish and many large ones make them explode.",
    ],
  },
  "Walk downhill": {
    title: "Step against the slope",
    summary:
      "Gradient descent reads the slope at the current setting and moves the dial the opposite way, by step size times slope. It needs no view of the whole curve, which is also why it settles in whichever valley it starts above.",
    whatYouSee: [
      "The meter L against the dial w. Flat spots are named along the top, each with a dotted line down to its place on the curve.",
      "The filled dot is the current setting, with a pink stub for the slope there. The hollow dot is the start, and the small dots joined by arcs are the settings visited so far.",
      "The dashed blue arc is the step that would be taken next, with its size. The knob on the axis is the start.",
      "Next step prints the update with the current numbers. The tiles give the number of steps, the dial, the meter and the slope.",
    ],
    howItWorks: [
      "Each step computes new w = w − step size × slope, using the exact slope of the chosen curve, and stops at the dial's ends, 0 and 6.",
      "Two valleys is the curve from card 2; Bowl is the curve from card 1, whose slope is 2(w − 3).",
      "On the Bowl each step multiplies the distance from the bottom by 1 − 2 × step size, which is why 0.5 lands in one step and anything above 1 moves away.",
      "The path is recomputed from Start w, Step size and the number of steps taken, so changing either slider begins a new walk.",
    ],
    controls: [
      "Take a step and Take 10 steps advance the walk, up to 40 steps. Start over returns to the start.",
      "Start w, or dragging on the plot, chooses where the walk begins. Step size runs from 0.05 to 1.2.",
      "Meter curve switches between Two valleys and Bowl.",
    ],
    notice: [
      "From Start w 2.6 on Two valleys the walk ends in the left valley at L = 2.28, although the right valley reaches 1.36.",
      "Steps shorten near a valley floor with no change to the step size, because the slope shrinks there.",
      "On the Bowl from Start w 5: 0.05 crawls, 0.5 lands in one step, 0.9 bounces across while closing in, and 1.1 climbs.",
    ],
    limits: [
      "In this lab: one dial, exact slopes, a curve drawn in advance, and dial stops that halt a runaway walk. The one-step landing at 0.5 is a property of this bowl.",
      "In general: the slope is estimated from a sample of data and is noisy, there are millions of dials, and no one can see the curve. A single-dial drawing also overstates how often a poor valley traps a large model.",
    ],
  },
  "Challenge: reach the valley floor": {
    title: "One step size for two directions",
    summary:
      "On a long valley the ground is steep across and gentle along. Each step moves both dials against their own slopes by the same step size, so the step must be small enough for the steep walls and large enough to make progress along the floor.",
    whatYouSee: [
      "The Long valley contour map. The hollow dot is the fixed start at w = 0.5, b = 4.5, and the green dashed ring around the centre is the goal: every setting where the meter is under 1.05.",
      "The red path is the current attempt, with a dot for each step. The grey dashed path is the previous attempt at a different step size.",
      "The goal bar fills as the meter falls from its starting value toward the valley floor, with a tick at the goal. The tiles give the steps left and the meter.",
      "Next step prints −step size × gradient with the current numbers, one entry for each dial.",
    ],
    howItWorks: [
      "The surface is L = 1 + ½[0.25(w − 3)² + 2(b − 3)²], starting at L = 4.03, with a floor of 1.",
      "Each step multiplies the distance from the centre by 1 − 0.25 × step size along w and by 1 − 2 × step size along b.",
      "The goal is met when the meter is under 1.05 at the current step. On the slider's grid that happens for step sizes from 0.45 to 0.90.",
      "The bar's length follows the square root of the meter's height above the floor, so the last stretch stays visible.",
    ],
    controls: [
      "Step size runs from 0.05 to 1.1. Changing it sends the walk back to the start and keeps the old path as the dashed one.",
      "Run all 12 steps finishes the attempt, Take one step advances it by one, and Back to the start resets it.",
    ],
    notice: [
      "At 0.10 the walk drops down the wall and then crawls along the floor, ending at 1.436.",
      "At 1.00 the walk reflects between the walls without closing in, ending at 3.25.",
      "The fastest attempts overshoot the floor slightly on every step.",
    ],
    limits: [
      "In this lab: an authored two-dial bowl, exact gradients, a fixed start and a 12-step budget. The range of step sizes that works belongs to this surface.",
      "In general: real loss surfaces have steep and gentle directions that differ by far more than the factor of 8 here, and they change as training moves. That is why learning rates are scheduled and why optimizers such as momentum and Adam exist.",
    ],
  },
};

export default cardInfo;
