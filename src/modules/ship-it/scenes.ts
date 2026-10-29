import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (no replay, `Q4_K_M`, no
 * pruning, 8 users at 4,096 tokens), where the harbor check fails. The lesson's later runs keep four replayed harbor
 * sentences, so the scenes that follow them set `replay: 4` too.
 */
const scenes: LabScene[] = [
  {
    id: "five-stages",
    sections: ["The five stages", "What it is"],
    title: "Five stages, one model",
    look: "Each slider on this card belongs to one of the first three stages, and the Stage by stage table shows the model after each one.",
    focus: "Training recipe",
  },
  {
    id: "release-report",
    sections: ["The release gate"],
    title: "Gate and budget together",
    look: "The Release report adds the serving budget to the two Eval gate checks, and lists each reason the release is on hold.",
    focus: "Release report",
  },
  {
    id: "forgetting",
    sections: ["Getting better at one job can mean getting worse at another", "Why it’s here"],
    title: "Forgetting the old job",
    look: "On the After adapting row, the Harbor text column is marked worse while the Recipe text column is marked better.",
    step: 0,
    focus: "Stage by stage",
  },
  {
    id: "replay-four",
    sections: ["Replay: practising the old job alongside the new one"],
    title: "Four sentences replayed",
    look: "With four harbor sentences replayed, the As shipped row ends lower than Base model on both texts.",
    step: 2,
    state: { replay: 4 },
    focus: "Stage by stage",
  },
  {
    id: "test-what-ships",
    sections: ["Test the version you ship, not the one you trained"],
    title: "Pruned past the gate",
    look: "With Prune at 90%, the After aligning row still looks good, but As shipped has lost most of the recipe gain, and that is the row the gate reads.",
    step: 3,
    state: { replay: 4, prune: 0.9 },
    focus: "Stage by stage",
  },
  {
    id: "six-item-quiz",
    sections: ["A six-question quiz cannot settle a close call"],
    title: "A six-item quiz",
    look: "Read the benchmark line under the two checks: each count moved, yet its ranges overlap, so the lab calls the change not decisive.",
    step: 1,
    focus: "Eval gate",
  },
  {
    id: "does-not-fit",
    sections: ["A model that does not fit is not ready"],
    title: "Passes, but does not fit",
    look: "Both quality checks pass, yet the verdict is hold, and the only reason listed is the serving budget.",
    step: 4,
    state: { replay: 4, users: 64, context: 32768 },
    focus: "Release report",
  },
  {
    id: "run-pipeline",
    sections: ["1. Run the pipeline", "How to play with it"],
    title: "Run the pipeline",
    look: "Read the table from Base model down to As shipped, comparing the Recipe text and Harbor text columns at each row.",
    step: 0,
    focus: "Stage by stage",
  },
  {
    id: "read-gate",
    sections: ["2. Read the gate"],
    title: "Read the gate",
    look: "Read which check fails under the badge and by how many nats, measured against the pass mark on its slider.",
    step: 1,
    focus: "Eval gate",
  },
  {
    id: "fix-regression",
    sections: ["3. Fix the regression"],
    title: "Fix the regression",
    look: "Raise Replay sentences one step at a time on Training recipe and watch Keeps the harbor text turn from Fail to Pass.",
    step: 2,
    focus: "Eval gate",
  },
  {
    id: "compress",
    sections: ["4. Compress"],
    title: "Round, then prune",
    look: "Step Weight format from F16 down to Q4_0, then drag Prune toward 90% and watch Recipe loss, as shipped climb.",
    step: 3,
    state: { replay: 4, format: "F16" },
    focus: "Compress and serve",
  },
  {
    id: "budget-and-decide",
    sections: ["5. Budget and decide"],
    title: "Budget and decide",
    look: "Read the second row of numbers at the starting load, then push Concurrent users and Context length to the top and read the Release report.",
    step: 4,
    state: { replay: 4 },
    focus: "Compress and serve",
  },
];

export default scenes;
