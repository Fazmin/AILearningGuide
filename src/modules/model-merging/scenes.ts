import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (full
 * fine-tunes at 20 epochs each, Linear blend at t = 0.5, λ = 1, Keep top 20%, drop rate 0.5, mask #4),
 * so the readouts show the lesson's own numbers. `density` is the Keep top share and `mix` is Blend t.
 */
const scenes: LabScene[] = [
  {
    id: "read-vectors",
    sections: ["Read the two task vectors", "How to play with it"],
    title: "Two task vectors",
    look: "Read Overlapping entries, Sign conflicts and the cosine badge, then find the struck-through cells on both maps.",
    step: 0,
    focus: "Task vectors",
  },
  {
    id: "two-maps",
    sections: ["Reading the two maps"],
    title: "Reading the maps",
    look: "Blue cells were raised and red cells lowered, and the rows that stay grey on both maps are characters neither text uses.",
    step: 0,
    focus: "Task vectors",
  },
  {
    id: "five-methods",
    sections: ["Five ways to combine the lists", "The five recipes"],
    title: "Five ways to merge",
    look: "Switch Merge method through all five and watch the Served weights formula and its controls change each time.",
    focus: "Merge recipe",
  },
  {
    id: "water-test",
    sections: ["The water test", "Why it’s here"],
    title: "The water test",
    look: "In the Linear blend row, Gap is the whole change from each specialist and Interference is the part the other list caused.",
    step: 1,
    focus: "Two task losses",
  },
  {
    id: "average",
    sections: ["Average, then split the cost"],
    title: "Average, then split the cost",
    look: "Find the large circle and the Linear blend row, then switch Merge method to Task arithmetic and watch Interference change sign.",
    step: 1,
    focus: "Two task losses",
  },
  {
    id: "full-strength",
    sections: ["Full strength is where they collide", "What to notice"],
    title: "Both lists at full strength",
    look: "In the Task arithmetic row Gap and Interference match, and sliding Scale λ moves the large circle along the dashed path.",
    step: 1,
    state: { method: "task" },
    focus: "Two task losses",
  },
  {
    id: "ties-sweep",
    sections: ["Resolve sign conflicts"],
    title: "Trim and elect",
    look: "Slide Keep top down from 100% to 10% and watch trimmed changes turn to dashed outlines while each conflict keeps its elected sign.",
    step: 2,
    state: { method: "ties", density: 1 },
    focus: "Merge recipe",
  },
  {
    id: "ties-winner",
    sections: ["TIES picks a winner in each conflict"],
    title: "A winner in each conflict",
    look: "In the strip, the h→a conflict’s merged bar keeps the whole proverb change instead of the small average.",
    step: 2,
    state: { method: "ties" },
    focus: "Merge recipe",
  },
  {
    id: "dare-sweep",
    sections: ["Drop and rescale"],
    title: "Drop and rescale",
    look: "Raise Drop rate p from 0 on the Merge recipe card and follow the large filled circle here, stopping at 0.50 to read the DARE row.",
    step: 3,
    state: { method: "dare", dropRate: 0 },
    focus: "Two task losses",
  },
  {
    id: "dare-breaks",
    sections: ["DARE breaks this model"],
    title: "DARE at a 90% drop",
    look: "The DARE point lands far above and to the right of every other marker, and each press of New random mask moves it.",
    step: 3,
    state: { method: "dare", dropRate: 0.9 },
    focus: "Two task losses",
  },
  {
    id: "slerp",
    sections: ["Curve instead of line"],
    title: "SLERP keeps the length",
    look: "Read the ‖τ_merged‖ badge and the angle Ω under the formula, then switch Merge method to Linear blend and read the badge again.",
    state: { method: "slerp" },
    focus: "Merge recipe",
  },
  {
    id: "lora-vectors",
    sections: ["Switch to LoRA task vectors"],
    title: "LoRA task vectors",
    look: "Read Sign conflicts and the cosine badge for the LoRA lists, then compare the Interference columns on Two task losses.",
    step: 0,
    state: { source: "lora" },
    focus: "Task vectors",
  },
];

export default scenes;
