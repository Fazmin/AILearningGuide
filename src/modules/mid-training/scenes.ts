import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (the
 * General-heavy mixture 3, 3, 3, 1, Same mix, Warmup-stable-decay, all 120 unique sentences at a budget
 * of 8,000 pairs), so the readouts show the lesson's own numbers.
 */
const scenes: LabScene[] = [
  {
    id: "uniform-mix",
    sections: ["Step 1: set the mixture", "How to play with it"],
    title: "Equal weights, unequal epochs",
    look: "With every weight at 1, compare each domain's Epochs in the table with its bar in Held-out loss by domain.",
    step: 0,
    state: { harborWeight: 1, recipesWeight: 1, proverbsWeight: 1, arithmeticWeight: 1 },
    focus: "Data mixture",
  },
  {
    id: "shares-to-epochs",
    sections: ["From shares to epochs", "What it is"],
    title: "Pairs drawn over pairs held",
    look: "Read Pairs drawn and Training pairs for each domain, because dividing the first by the second gives the Epochs column.",
    step: 0,
    focus: "Data mixture",
  },
  {
    id: "arithmetic-eight",
    sections: ["Step 2: upweight the scarce domain"],
    title: "Arithmetic at weight 8",
    look: "The last row of the sweep is highlighted, so read Gain per extra epoch down its column and watch General held-out climb.",
    step: 1,
    state: { arithmeticWeight: 8 },
    focus: "Arithmetic weight sweep",
  },
  {
    id: "two-sentences",
    sections: ["Step 3: repeat or find more"],
    title: "Two sentences, many passes",
    look: "The dashed line for 2 unique sentences dips and then climbs back up, while the solid line for all 120 levels off at its lowest.",
    step: 2,
    state: { uniqueSentences: 2 },
    focus: "Repeated data",
  },
  {
    id: "target-mix",
    sections: ["Step 4: anneal on the target mix"],
    title: "Target mix from step 480",
    look: "Compare the Switched at decay and Spread evenly columns, and watch the solid Arithmetic line dive after the step 480 marker.",
    step: 3,
    state: { latePhase: "mix" },
    focus: "Late phase and annealing",
  },
  {
    id: "target-only",
    sections: ["Step 5: mid-train or fine-tune?"],
    title: "Nothing but Arithmetic at the end",
    look: "Every late batch is Arithmetic now, so read Forgetting and General held-out, then switch Late phase back to Target mix.",
    step: 4,
    state: { latePhase: "only" },
    focus: "Late phase and annealing",
  },
  {
    id: "answer-probe",
    sections: ["What the loss misses"],
    title: "Lower loss, no sums",
    look: "Read Right answers and Different answers, then set Arithmetic weight to 0 and watch the favourite answer change.",
    step: 1,
    focus: "Answer probe",
  },
  {
    id: "constant-rate",
    sections: ["Where the toy disagrees", "Where it breaks"],
    title: "The switch without the decay",
    look: "With Constant rate, the Switched at decay column gains more on Arithmetic and loses more on General than with the decay.",
    step: 3,
    state: { latePhase: "mix", schedule: "constant" },
    focus: "Late phase and annealing",
  },
  {
    id: "end-of-run",
    sections: ["Why the end of the run matters", "Why it’s here"],
    title: "The last batches count most",
    look: "The Arithmetic share jumps at the step 480 marker while the learning rate falls, and the held-out lines bend at the same point.",
    step: 3,
    state: { latePhase: "mix" },
    focus: "Late phase and annealing",
  },
];

export default scenes;
