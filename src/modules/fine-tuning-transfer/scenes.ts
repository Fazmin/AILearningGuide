import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (20
 * epochs at learning rate 0.6, nothing frozen, 3 target sentences held out, no replay, every answer
 * on No), so the readouts show the lesson's own numbers. Block ids follow the freeze grid from
 * `block-0` (␣abcde) to `block-4` (xyz.<>).
 */
const scenes: LabScene[] = [
  {
    id: "head-start",
    sections: ["Step 1: the head start before any training", "How to play with it"],
    title: "The head start at zero epochs",
    look: "With no adaptation, compare Transfer, target loss with Scratch, target loss, and read the gap in Transfer is lower by.",
    step: 0,
    state: { fineTuneEpochs: 0 },
    focus: "Transfer against training from scratch",
  },
  {
    id: "default-run",
    sections: ["Step 2: the default run"],
    title: "The default run",
    look: "Read Source before, Source after and Target gained, then drag Fine-tune epochs slowly and watch the Forgetting meter lines cross.",
    step: 1,
    focus: "Fine-tuning controls",
  },
  {
    id: "freeze-two",
    sections: ["Step 3: freeze some pages"],
    title: "Two blocks frozen",
    look: "␣abcde and fghijk are frozen, so read Frozen rows and Trainable weights, and see their hatched bars sit at zero in the row strip.",
    step: 2,
    state: { frozen: ["block-0", "block-1"] },
    focus: "Layer freezing",
  },
  {
    id: "same-budget",
    sections: ["Step 4: spend the budget a different way"],
    title: "Half the epochs, twice the rate",
    look: "The sliders now give half the epochs at twice the rate, so read Forgetting on the Layer freezing card and compare it with the default run.",
    step: 1,
    state: { fineTuneEpochs: 10, learningRate: 1.2 },
    focus: "Fine-tuning controls",
  },
  {
    id: "held-out",
    sections: ["Step 5: test on sentences it never saw"],
    title: "Three sentences held out",
    look: "Compare Trained-on target with Held-out target in the table, then raise Fine-tune epochs and watch the gap between them grow.",
    step: 4,
    focus: "Replay and held-out target",
  },
  {
    id: "replay-half",
    sections: ["Step 6: mix old sentences back in"],
    title: "Half the old text replayed",
    look: "A With 50% replay column appears beside No replay, so compare them row by row, ending with the two split Forgetting rows.",
    step: 4,
    state: { replayShare: 50 },
    focus: "Replay and held-out target",
  },
  {
    id: "decision-card",
    sections: ["Step 7: the decision card", "Should you fine-tune?"],
    title: "Four questions, one route",
    look: "Flip one answer at a time and watch the Recommended route change and the (matched) mark move in the rule table.",
    step: 5,
    focus: "Should you fine-tune? Rule of thumb",
  },
  {
    id: "forgetting",
    sections: ["The cost: catastrophic forgetting", "Why it’s here"],
    title: "One model, two corpora",
    look: "The solid Recipe steps line falls while the dashed Harbor weather notes line climbs, and the badge counts how much Harbor got worse.",
    step: 3,
    focus: "Forgetting meter",
  },
  {
    id: "head-start-early",
    sections: ["On the first four cards", "What to notice"],
    title: "The head start shrinks",
    look: "The solid and dashed target curves start far apart and close in as training goes on, and raising Fine-tune epochs closes them further.",
    step: 0,
    focus: "Transfer against training from scratch",
  },
  {
    id: "held-out-overfit",
    sections: ["On the replay card"],
    title: "Held-out loss turns back up",
    look: "With four sentences held out, the dotted held-out line bottoms out halfway and turns back up while the trained-on line keeps falling.",
    step: 4,
    state: { heldOutSentences: 4, fineTuneEpochs: 60 },
    focus: "Replay and held-out target",
  },
  {
    id: "examples-alone",
    sections: ["On the decision card"],
    title: "Examples alone change nothing",
    look: "The examples question is on Yes, yet the Recommended route stays Prompt first until the format question is Yes as well.",
    step: 5,
    state: { hasExamples: true },
    focus: "Should you fine-tune? Rule of thumb",
  },
];

export default scenes;
