import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: Sunday
 * hours at a budget of 3, the training run at k = 3, a group of 16, learning rate 0.3, seed 1, Length
 * cost off and Training update at 100, and the vote at 8 samples with p = 0.40 split over 3 wrong answers.
 */
const scenes: LabScene[] = [
  {
    id: "chain-of-thought",
    sections: ["Working, one piece at a time"],
    title: "Working for 17 + 28",
    look: "The three lines here are the lesson’s working for 17 + 28, each one ordinary text written before the answer.",
    step: 4,
    state: { problem: "add" },
    focus: "Inspectable trace",
  },
  {
    id: "practiced-problem",
    sections: ["1. Pick a practiced problem"],
    title: "A practiced problem",
    look: "Switch Problem between the three questions and read the card’s note on how a prompt differs from training.",
    step: 0,
    focus: "Problem",
  },
  {
    id: "train-checker",
    sections: ["2. Train against a checker", "How to play with it"],
    title: "Training from update 0",
    look: "Training update starts at 0; drag it to 100 and watch Mean length climb past the dotted k line while Checker pass rate rises.",
    step: 1,
    state: { grpoView: 0 },
    focus: "Train against a checker",
  },
  {
    id: "length-cost",
    sections: ["3. Add a length cost"],
    title: "A price on every step",
    look: "Length cost is on, so the length curve levels off lower and a dashed Mean reward after cost line joins the pass rate chart.",
    step: 2,
    state: { grpoCost: true },
    focus: "Train against a checker",
  },
  {
    id: "raise-budget",
    sections: ["4. Raise the budget"],
    title: "Raise the budget",
    look: "Raise Thinking budget one step at a time and watch Padding, then Unfaithful, tick up while Useful waits for the last line.",
    step: 3,
    focus: "Thinking budget",
  },
  {
    id: "no-trace",
    sections: ["5. Inspect the extra tokens"],
    title: "A budget of zero",
    look: "Thinking budget is 0, so the trace is empty, yet the answer slot still prints the sum because the lab wrote it in advance.",
    step: 4,
    state: { problem: "add", budget: 0 },
    focus: "Inspectable trace",
  },
  {
    id: "sample-vote",
    sections: ["6. Sample and vote"],
    title: "Sixteen samples",
    look: "Compare Majority vote with Verifier picks, then set Wrong answers to Always the same one and watch the vote drop below One sample.",
    step: 5,
    state: { samples: 16 },
    focus: "Sampling and voting",
  },
  {
    id: "overthinking",
    sections: ["With no price on length, it keeps growing", "What to notice"],
    title: "Length keeps growing",
    look: "At update 100 with Length cost off, read Steps past k and Groups with no signal under the bar chart.",
    step: 1,
    focus: "Train against a checker",
  },
  {
    id: "small-group",
    sections: ["Why a small group collapses"],
    title: "A group of eight collapses",
    look: "With Length cost on and Group size (G) at 8, the solid bars pile up left of the rule, where traces never pass; set Seed to 4 to see a run escape.",
    step: 2,
    state: { grpoCost: true, grpoGroup: 8 },
    focus: "Train against a checker",
  },
  {
    id: "carry-memory",
    sections: ["Working helps most where a later step needs an earlier result"],
    title: "The whole trace for 17 + 28",
    look: "All six lines are shown: the carry line feeds the tens line, while the padding and unfaithful lines add nothing the answer uses.",
    step: 4,
    state: { problem: "add", budget: 6 },
    focus: "Inspectable trace",
  },
  {
    id: "one-right-try",
    sections: ["A checker needs only one right try"],
    title: "One right try is enough",
    look: "At 2 samples Verifier picks already beats One sample; drag Samples up and watch it close in on 100%.",
    step: 5,
    state: { samples: 2 },
    focus: "Sampling and voting",
  },
  {
    id: "same-wrong-answer",
    sections: ["A vote follows the most popular answer"],
    title: "One wrong answer wins the vote",
    look: "With Always the same one, only the bars past the halfway mark keep a solid part, so Majority vote sits below One sample; drag Samples up and watch it fall further.",
    step: 5,
    state: { samples: 16, spread: "same" },
    focus: "Sampling and voting",
  },
];

export default scenes;
