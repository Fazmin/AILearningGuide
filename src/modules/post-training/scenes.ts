import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. `labels` holds one letter per pair, Pair 1 first:
 * `a` prefers the pair's first response, `b` the second, and `n` leaves it unlabelled. The default is
 * "aaaana", with Pair 5 open; "aaaaba" labels Pair 5 for B, and "aaabba" is what Label like the gold
 * rater writes. `beta` is the KL coefficient itself, not the slider's log position.
 */
const scenes: LabScene[] = [
  {
    id: "sft-eight-steps",
    sections: ["Stage 1: copy good answers", "Stage one: supervised fine-tuning"],
    title: "Eight steps of SFT",
    look: "Compare each answer card’s base and SFT bars: A’s share has moved to B and F, the two cards marked SFT demo.",
    step: 0,
    state: { sftSteps: 8 },
    focus: "Post-training pipeline",
  },
  {
    id: "scorer-features",
    sections: ["What the scorer can see", "Stage two: a reward model from comparisons"],
    title: "Three numbers per answer",
    look: "The three bars are the scorer’s weights, and the r on each answer card adds up that answer’s three numbers times these weights.",
    step: 1,
    focus: "Reward model fit",
  },
  {
    id: "bradley-terry",
    sections: ["Turning picks into chances: Bradley-Terry"],
    title: "From a reward gap to a chance",
    look: "The line under each pair turns the gap between the two answers’ r values into the chance that the first one wins.",
    step: 1,
    focus: "Preference pairs",
  },
  {
    id: "measure-kl",
    sections: ["Measuring how far the model moved"],
    title: "A middle-sized move",
    look: "Read KL from SFT under the slider, then compare A’s SFT and tuned bars on the first card.",
    step: 2,
    state: { beta: 3.5 },
    focus: "Optimize against the reward model",
  },
  {
    id: "exact-recipe",
    sections: ["The recipe the lab uses", "Stage three: optimize under a KL leash"],
    title: "The tilt, worked through",
    look: "The formula card works through the tuned model’s favorite answer, and its result is that answer’s tuned bar on the first card.",
    step: 2,
    focus: "Optimize against the reward model",
  },
  {
    id: "almost-no-leash",
    sections: ["Why the leash is needed"],
    title: "Almost no leash",
    look: "With KL coefficient β at its lowest, E’s tuned bar fills up while its SFT bar stays small.",
    step: 2,
    state: { beta: 0.02 },
    focus: "Post-training pipeline",
  },
  {
    id: "sft-from-zero",
    sections: ["1. Watch SFT move the chances", "How to play with it"],
    title: "Start SFT from zero",
    look: "With SFT steps at 0 the base and SFT bars match, so raise SFT steps to 8 and watch A’s SFT bar shrink while B and F grow.",
    step: 0,
    state: { sftSteps: 0 },
    focus: "Post-training pipeline",
  },
  {
    id: "be-the-judge",
    sections: ["2. Be the judge"],
    title: "You are the judge",
    look: "Click an answer to pick it, click it again to clear it, and watch the chance under every pair change, even Pair 5, which nobody has labeled.",
    step: 1,
    focus: "Preference pairs",
  },
  {
    id: "scorer-report",
    sections: ["3. Read what the scorer learned", "What to notice"],
    title: "What the scorer learned",
    look: "Compare Labels fitted with Agrees with gold: the first checks only your picks, the second checks every pair against gold.",
    step: 1,
    focus: "Reward model fit",
  },
  {
    id: "loosen-leash",
    sections: ["4. Loosen the leash"],
    title: "Loosen the leash",
    look: "Drag KL coefficient β slowly down to its lowest, and watch Reward-model score keep rising while Gold score rises and then falls.",
    step: 2,
    state: { beta: 20 },
    focus: "Optimize against the reward model",
  },
  {
    id: "pair-five",
    sections: ["5. Break the length shortcut"],
    title: "Pair 5 labeled for B",
    look: "Words ÷ 10 has shrunk and Names mechanism is now the biggest bar, so sweep KL coefficient β again and watch the gold curve stay high.",
    step: 3,
    state: { labels: "aaaaba" },
    focus: "Reward model fit",
  },
  {
    id: "gold-labels",
    sections: ["6. Compare with the gold rater"],
    title: "Labels from the gold rater",
    look: "Opens with praise now lowers reward, yet Agrees with gold still falls short of a full score.",
    step: 1,
    state: { labels: "aaabba" },
    focus: "Reward model fit",
  },
];

export default scenes;
