import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its headings. Each building beat opens its card as it first
 * appears; the limits scenes load the settings their paragraphs describe.
 */
const scenes: LabScene[] = [
  {
    id: "two-axes",
    sections: ["Two axes, not one ladder"],
    title: "The Levels of AGI table",
    look: "Pick a system, click the cell you think it belongs in, and watch the ASI corner at the top right.",
    step: 0,
    focus: "Narrow, general, and how good",
  },
  {
    id: "three-verdicts",
    sections: ["One profile, three verdicts"],
    title: "GPT-5's profile",
    look: "Switch the Rule and watch the same shape give three different answers.",
    step: 1,
    focus: "One profile, three verdicts",
  },
  {
    id: "faster-more",
    sections: ["Faster, more, or better"],
    title: "One expert, 100 years",
    look: "Raise Speed and Copies and watch the blue pin slide left along the time ruler.",
    step: 2,
    focus: "Faster, more, or better",
  },
  {
    id: "feedback-loop",
    sections: ["A loop that feeds itself"],
    title: "Returns at 1.5",
    look: "Move Returns to research up and down and watch the curve bend toward or away from the top.",
    step: 3,
    focus: "A loop that feeds itself",
  },
  {
    id: "forecasts",
    sections: ["Reading a forecast"],
    title: "Two questions, one survey",
    look: "Compare the two filled dots on the timeline, then change Assumed doubling time.",
    step: 4,
    focus: "Reading forecasts",
  },
  {
    id: "sort",
    sections: ["Your turn: sort the claims"],
    title: "Eight claims to sort",
    look: "Pick a kind for each claim and read the reason that appears under it.",
    step: 5,
    focus: "Challenge: sort the claims",
  },
  {
    id: "world-floor",
    sections: ["What the lab simplifies"],
    title: "A floor set by the world",
    look: "Drag Waits on the world down to 0% and back to 10% and compare Fastest possible here.",
    step: 2,
    state: { speed: 1000, copies: 1000000, world: 0.1 },
    focus: "Faster, more, or better",
  },
  {
    id: "capped-loop",
    sections: ["Where the arguments are open"],
    title: "A capped loop",
    look: "Compare the solid curve with the dashed one, then raise Returns to research to 3.",
    step: 3,
    state: { returns: 1.5, bottleneck: 0.2, prevReturns: 1.5, prevBottleneck: 0 },
    focus: "A loop that feeds itself",
  },
];

export default scenes;
