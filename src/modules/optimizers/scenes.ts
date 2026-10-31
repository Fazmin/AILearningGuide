import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (SGD rate
 * 1, Momentum rate 0.1, AdamW rate 0.03, Muon rate 0.05, training-loss curves, bias correction on, no
 * warmup, no weight decay, Step 300 and 5 Newton–Schulz iterations), so the readouts show the lesson's
 * own numbers unless a scene sets a control.
 */
const scenes: LabScene[] = [
  {
    id: "momentum-matches-sgd",
    sections: ["Four ways to take a step", "Four update rules"],
    title: "Momentum at a tenth of the rate",
    look: "Momentum at 0.1 draws almost the same curve as SGD at 1, because its running total grows a steady push about ten times.",
    step: 0,
    state: { optimizer: "momentum" },
    focus: "Optimizer race",
  },
  {
    id: "race-defaults",
    sections: ["Race at the defaults"],
    title: "Four optimizers at their defaults",
    look: "All four rate sliders sit at their defaults, so compare the four curves and then the Held out column under the chart.",
    step: 0,
    focus: "Optimizer race",
  },
  {
    id: "held-out-curves",
    sections: ["On the race"],
    title: "Held-out loss turns back up",
    look: "With Curves on Held-out loss, the dotted AdamW curve dips lowest near epoch 13 and then creeps back up.",
    step: 0,
    state: { curves: "held" },
    focus: "Optimizer race",
  },
  {
    id: "volume-knob",
    sections: ["The picture: a volume knob for every weight", "Why Adam’s steps are per weight"],
    title: "Rare rows under AdamW",
    look: "With Optimizer on AdamW, the rare k, q and z rows move much further per visit than the busy space row.",
    step: 1,
    state: { optimizer: "adamw" },
    focus: "Step size per row",
  },
  {
    id: "rare-rows",
    sections: ["Rare rows and busy rows"],
    title: "Rare rows under SGD",
    look: "Read q ÷ space with Optimizer on SGD, then switch to AdamW and watch the rare rows' bars grow past the space row.",
    step: 1,
    focus: "Step size per row",
  },
  {
    id: "first-step-spike",
    sections: ["Break Adam’s first step"],
    title: "No bias correction at rate 0.3",
    look: "At AdamW rate 0.3 with Bias correction off, the solid curve spikes far above the dashed one, so raise Warmup to 10% and watch it shrink.",
    step: 2,
    state: { adamRate: 0.3, biasCorrection: "off" },
    focus: "Bias correction and warmup",
  },
  {
    id: "first-steps-default",
    sections: ["Adam’s first steps", "On the early steps"],
    title: "No bias correction at the default rate",
    look: "At the default AdamW rate, turning Bias correction off makes First-step move about three times bigger, and Held out gets worse without any spike.",
    step: 2,
    state: { biasCorrection: "off" },
    focus: "Bias correction and warmup",
  },
  {
    id: "decay-modes",
    sections: ["Two ways to decay"],
    title: "Weight decay 0.1 in the gradient",
    look: "At Weight decay 0.1 with L2 in gradient, Held out sits near the knows-nothing score, so switch Decay mode to Decoupled and compare.",
    step: 3,
    state: { weightDecay: 0.1, decayMode: "l2" },
    focus: "Weight decay",
  },
  {
    id: "decay-rare-rows",
    sections: ["Two ways to pull weights toward zero", "On weight decay"],
    title: "Which rows L2 shrinks",
    look: "At Weight decay 0.001 with L2 in gradient, the q row keeps far less of its size than the space row, so switch to Decoupled and compare.",
    step: 3,
    state: { weightDecay: 0.001, decayMode: "l2" },
    focus: "Weight decay",
  },
  {
    id: "newton-schulz",
    sections: ["Watch Newton–Schulz work"],
    title: "Zero Newton–Schulz iterations",
    look: "At 0 iterations the after bars match the before bars, so step Newton–Schulz iterations up to 5 and watch them rise toward 1.",
    step: 4,
    state: { nsSteps: 0 },
    focus: "Orthogonalized update",
  },
  {
    id: "muon-first-step",
    sections: ["Muon: even out the directions", "Muon: orthogonalize, then step"],
    title: "Muon's very first step",
    look: "At Step 1 only nine directions have a gradient, and after five iterations they all sit near 1 while the other 21 stay at zero.",
    step: 4,
    state: { capture: "first" },
    focus: "Orthogonalized update",
  },
  {
    id: "eight-iterations",
    sections: ["On the orthogonalized update"],
    title: "Eight Newton–Schulz iterations",
    look: "At 8 iterations almost every bar sits between 0.5 and 1.5, yet the four zero directions at the right still stay at zero.",
    step: 4,
    state: { nsSteps: 8 },
    focus: "Orthogonalized update",
  },
  {
    id: "tune-each",
    sections: ["Tune each, then count the bytes"],
    title: "A fair search for every optimizer",
    look: "Press Tune each and wait for the table, then compare each row's default column with its marked best.",
    step: 5,
    focus: "Tuned against tuned",
  },
  {
    id: "optimizer-memory",
    sections: ["What an optimizer remembers", "On tuning and memory"],
    title: "What each optimizer keeps",
    look: "The AdamW row keeps two floats per weight, twice Momentum and Muon, while SGD keeps none.",
    step: 5,
    focus: "Optimizer state",
  },
];

export default scenes;
