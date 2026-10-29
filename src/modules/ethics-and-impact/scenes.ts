import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `claim` is 0-based, so `claim: 1` is the lab's
 * "2 Against". `shareB` is Group B share of patients as a fraction (default 0.3; the holdout is 0.1).
 * `servedLog` is the power of ten of Tokens served per day (10 is 10B), `trainTokens` counts trillions,
 * and `paramIndex` 2 is 7B. Every scene starts from those defaults, which are the lesson's worked example.
 */
const scenes: LabScene[] = [
  {
    id: "four-deployments",
    sections: ["Four deployments, five claims each", "What it is"],
    title: "Claims for and against",
    look: "The For and Against columns list each claim with who bears it, and the readouts below count the claims without ever giving a Score.",
    step: 0,
    focus: "Deployment",
  },
  {
    id: "one-claim",
    sections: ["Reading one claim, step by step"],
    title: "One claim, three tags",
    look: "Read Theme, Who bears it and Can a benchmark measure it? together, then pick 3 For and 4 Against and compare.",
    step: 1,
    state: { deployment: "school", claim: 1 },
    focus: "Tension explorer",
  },
  {
    id: "clinic-tomorrow",
    sections: ["Who the benchmark represents"],
    title: "Half the patients from group B",
    look: "Compare the two accuracy readouts with Missed urgent per 1,000, then see how much of the lower missed-cases bar is hatched group B.",
    step: 2,
    state: { shareB: 0.5 },
    focus: "Who the benchmark represents",
  },
  {
    id: "fairness-rules",
    sections: ["Two fairness rules, and the one this model breaks"],
    title: "Two gaps that never move",
    look: "On the second row of readouts, compare the True-positive rate and False-positive rate pairs, then move Group B share of patients and watch neither gap change.",
    step: 2,
    focus: "Who the benchmark represents",
  },
  {
    id: "training-serving",
    sections: ["Training versus serving"],
    title: "Paid once, paid daily",
    look: "Read Training, once, Serving, per day and Days to break even, then find where the solid serving line crosses the dashed training line.",
    step: 3,
    focus: "Training versus serving",
  },
  {
    id: "walk-claims",
    sections: ["2. Walk the claims"],
    title: "Walk the claims",
    look: "Step Claim from 1 For to 5 Against and read Who bears it and Can a benchmark measure it? each time.",
    step: 1,
    state: { claim: 0 },
    focus: "Tension explorer",
  },
  {
    id: "split-benchmark",
    sections: ["3. Split the benchmark"],
    title: "Split the benchmark",
    look: "Drag Group B share of patients down to 10% and then up to 50%, watching Missed urgent per 1,000 and the hatched part of the lower bars.",
    step: 2,
    focus: "Who the benchmark represents",
  },
  {
    id: "weigh-serving",
    sections: ["4. Weigh serving against training"],
    title: "Move the break-even day",
    look: "Raise Tokens served per day and watch Days to break even fall, then change Parameters and see that it stays put.",
    step: 3,
    focus: "Training versus serving",
  },
  {
    id: "missing-score",
    sections: ["5. Notice the missing score"],
    title: "Costs no leaderboard measures",
    look: "Step through Newsroom draft’s five claims, then switch Deployment to Clinic intake and Image generator and do the same.",
    step: 4,
    state: { deployment: "news", claim: 0 },
    focus: "Deployment",
  },
];

export default scenes;
