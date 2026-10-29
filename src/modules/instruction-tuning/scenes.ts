import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: the four
 * starting pairs, 30 epochs, all three template switches on, the loss on the assistant reply only and
 * the chat template served, so the cards show the lesson's own numbers unless a scene changes a setting.
 */
const scenes: LabScene[] = [
  {
    id: "one-long-line",
    sections: ["One long line of text", "What it is"],
    title: "One long line",
    look: "Each switch in this card reads in template, and the strip under them spells out the whole first example, one cell per character.",
    step: 0,
    focus: "Chat template editor",
  },
  {
    id: "roles-off",
    sections: ["1. Edit the template", "How to play with it"],
    title: "Role markers switched off",
    look: "Role markers is off, so the question runs straight into the answer in the strip; switch it back on and watch <user> and <assistant> return.",
    step: 0,
    state: { useRoles: false },
    focus: "Chat template editor",
  },
  {
    id: "graded-or-read",
    sections: ["Graded, or only read", "Which characters are graded"],
    title: "Graded or only read",
    look: "In the strip, grey hatched cells are only read, cells with a blue underline are graded, and Graded characters counts the graded ones.",
    step: 1,
    focus: "Chat template editor",
  },
  {
    id: "every-character",
    sections: ["2. Read the loss mask"],
    title: "Every character graded",
    look: "Loss on is set to Every character, so no cell is hatched and Mean −log p, context reads none; switch back to Assistant reply only to compare.",
    step: 1,
    state: { lossMask: "all" },
    focus: "Chat template editor",
  },
  {
    id: "few-epochs",
    sections: ["3. Fine-tune on your pairs"],
    title: "Training for four epochs",
    look: "Drag SFT epochs up from 4 passes and watch “<” after the full stop climb in the Two models, two formats card.",
    step: 2,
    state: { epochs: 4 },
    focus: "Supervised dataset editor",
  },
  {
    id: "end-marker-off",
    sections: ["4. Break the boundary"],
    title: "No end-of-turn marker",
    look: "End-of-turn marker is off, so the templated model’s upper list goes as flat as the plain model’s list below it.",
    step: 3,
    state: { useEnd: false },
    focus: "What follows the end of a turn",
  },
  {
    id: "serve-format",
    sections: ["5. Serve a different format", "Serve a different format"],
    title: "Serve a different format",
    look: "Switch Serve with between Chat template, No template and Wrong role marker, and read First reply character “t”, templated vs plain under the bars.",
    step: 4,
    focus: "Serve the trained models",
  },
  {
    id: "signal-to-stop",
    sections: ["A signal to stop", "Why it’s here"],
    title: "A learned stop",
    look: "The upper list is the templated model, and its top bar after the full stop is “<”, the start of a stop marker the plain model below never learned.",
    step: 3,
    focus: "What follows the end of a turn",
  },
  {
    id: "prompt-graded",
    sections: ["Why only the reply is graded"],
    title: "Grading the prompt too",
    look: "With Every character under Loss on, the templated model’s sample opens with a marker, and “<” tops its bars after the served string.",
    step: 4,
    state: { lossMask: "all" },
    focus: "Serve the trained models",
  },
  {
    id: "stop-from-prompt",
    sections: ["Where a reply ends", "What to notice"],
    title: "A stop learned from the prompt",
    look: "End-of-turn marker is off, yet with Every character under Loss on, “<” leads the upper list again, learned from the system prompt.",
    step: 3,
    state: { useEnd: false, lossMask: "all" },
    focus: "What follows the end of a turn",
  },
  {
    id: "template-costs",
    sections: ["What the template costs and buys"],
    title: "What the template costs",
    look: "Compare Reply log-prob per char, templated with the plain one, then read “<” after the full stop beside them.",
    step: 2,
    focus: "Two models, two formats",
  },
  {
    id: "wrong-role",
    sections: ["Serving a different format"],
    title: "The wrong role marker",
    look: "Wrong role marker sends <user> where <assistant> belongs, yet the bars and samples match Chat template because both strings end in “>”.",
    step: 4,
    state: { serveWith: "wrong-role" },
    focus: "Serve the trained models",
  },
];

export default scenes;
