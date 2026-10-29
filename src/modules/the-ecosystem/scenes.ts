import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults, the
 * Research only license with the training code and data information withheld and a Research paper as
 * the use, so a scene only names the controls its section changes. `license` is "research",
 * "community" or "apache"; `use` is "research", "product", "finetune" or "outputs".
 */
const scenes: LabScene[] = [
  {
    id: "base-and-fine-tune",
    sections: ["Base model and fine-tune", "What it is"],
    title: "Base model, no fine-tunes",
    look: "Read the Base model and Fine-tunes rows at the top of the model card, before any of the controls.",
    step: 0,
    focus: "Sample model card",
  },
  {
    id: "api-only",
    sections: ["Open weights and the API"],
    title: "What an API keeps back",
    look: "Find the Run on your own hardware row: the API only column is the one place it says no.",
    step: 1,
    focus: "Release spectrum",
  },
  {
    id: "full-kit",
    sections: ["Open source AI: the full kit", "Open weights are not open source"],
    title: "The full kit",
    look: "With Apache-2.0, the training code and the data information all released, the highlighted column is Open source AI, and its bottom row still does not promise the training data itself.",
    step: 1,
    state: { license: "apache", code: "released", data: "published" },
    focus: "Release spectrum",
  },
  {
    id: "fine-tune-and-ship",
    sections: ["The license, one rule at a time", "Permission is per use"],
    title: "One clause at a time",
    look: "Each clause the use needs gets its own row, and the one ◐ row sets License match even though the other rows show ✓.",
    step: 2,
    state: { license: "apache", use: "finetune" },
    focus: "Is this use allowed",
  },
  {
    id: "permission-not-provenance",
    sections: ["What a license cannot settle"],
    title: "Allowed, but still unknown",
    look: "License match says yes, yet Still unknown lists the data, so set Data information to published on the model card and only consent is left.",
    step: 3,
    state: { license: "apache", use: "product" },
    focus: "Is this use allowed",
  },
  {
    id: "card-numbers",
    sections: ["Reading a model card"],
    title: "Numbers the card works out",
    look: "Read the Evaluation harness and Training energy (estimated) rows: both are worked out from the card’s own figures.",
    step: 0,
    focus: "Sample model card",
  },
  {
    id: "research-only-product",
    sections: ["Why it’s here"],
    title: "A paid app under Research only",
    look: "Under Research only, the Commercial use row carries the ✗ that blocks a Hosted product before quality even comes up.",
    step: 2,
    state: { use: "product" },
    focus: "Is this use allowed",
  },
  {
    id: "spectrum-walk",
    sections: ["How to play with it"],
    title: "Walk along the spectrum",
    look: "Set License to Apache-2.0, then Training code to released, then Data information to published, and watch the highlighted column after each click.",
    step: 1,
    focus: "Release spectrum",
  },
  {
    id: "terms-matter",
    sections: ["What to notice"],
    title: "Every file, restrictive terms",
    look: "The training code and data information are both released, yet the highlighted column stays on the restricted license because Research only does not allow every purpose.",
    step: 1,
    state: { code: "released", data: "published" },
    focus: "Release spectrum",
  },
];

export default scenes;
