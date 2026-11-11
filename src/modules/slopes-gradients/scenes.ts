import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its headings. The five beats and the challenge open on the lab's
 * defaults, because each beat starts from the card as it first appears; the two scenes in the limits
 * section load the settings their paragraphs describe.
 */
const scenes: LabScene[] = [
  {
    id: "nudge-the-dial",
    sections: ["Nudge the dial"],
    title: "A nudge of 1",
    look: "Drag Nudge size to the left and watch Ratio at this nudge close in on Slope it settles on.",
    step: 0,
    focus: "Nudge the dial",
  },
  {
    id: "slope-everywhere",
    sections: ["A slope at every point"],
    title: "The meter and its slope",
    look: "Drag Dial w across the two pictures and compare the sign of Slope here with the band you are in.",
    step: 1,
    focus: "A slope at every point",
  },
  {
    id: "two-dials",
    sections: ["Two dials, one slope each"],
    title: "One leg at a time",
    look: "Press w only, b only and then Both together to build the arrow from its two legs.",
    step: 2,
    focus: "Two dials, one slope each",
  },
  {
    id: "chain",
    sections: ["Dials that drive dials"],
    title: "Two machines in a row",
    look: "Read the three bars under the curves, then slide Dial w up to 5 and watch the last bar.",
    step: 3,
    focus: "Dials that drive dials",
  },
  {
    id: "walk",
    sections: ["Walk downhill"],
    title: "Just left of the hump",
    look: "Read the Next step line, then press Take 10 steps and see which valley the walk ends in.",
    step: 4,
    focus: "Walk downhill",
  },
  {
    id: "challenge",
    sections: ["A first try", "Your turn: reach the valley floor"],
    title: "The challenge, at the start",
    look: "Press Run all 12 steps and read the goal bar, then change Step size and run again.",
    step: 5,
    focus: "Challenge: reach the valley floor",
  },
  {
    id: "lucky-step",
    sections: ["What the lab simplifies"],
    title: "One step to the bottom",
    look: "On the Bowl, one step from Start w 5 lands on the bottom. Move Step size a little and take the step again.",
    step: 4,
    state: { walkCurve: "bowl", walkStart: 5, walkStep: 0.5, walkSteps: 1 },
    focus: "Walk downhill",
  },
  {
    id: "flat-link",
    sections: ["Where the real idea runs into trouble"],
    title: "A flat link",
    look: "With Dial w at 5, machine B is nearly flat, so the last bar has almost vanished while the meter reads high.",
    step: 3,
    state: { chainW: 5 },
    focus: "Dials that drive dials",
  },
];

export default scenes;
