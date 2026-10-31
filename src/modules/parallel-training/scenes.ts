import type { LabScene } from "@app/module-sdk";
import { LLAMA3_LAYOUT } from "./state";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults (Llama 3.1 8B on 64
 * data-parallel GPUs, ZeRO stage 0, one sequence per microbatch, 8 microbatches, Selective recompute, overlap on,
 * a checkpoint every 60 minutes with a 60-second save and a 10-minute restart), so the readouts show the lesson's
 * own numbers. Every value is one a control can set, which the module's hydrateState keeps as it is.
 */
const SEVENTY_B = { modelId: "70b", tensor: 8, pipeline: 4, data: 16, microbatches: 16, zero: 1 };
const SEVENTY_B_PIPELINE = { modelId: "70b", tensor: 8, pipeline: 8, data: 16, zero: 1 };

const scenes: LabScene[] = [
  {
    id: "four-parts",
    sections: ["Memory: four parts per GPU", "What one GPU has to hold", "What it is"],
    title: "Four parts of memory",
    look: "The bars are weights, gradients, optimizer state and activations, and together they run far past the 80 GB line.",
    step: 0,
    focus: "Memory per GPU",
  },
  {
    id: "four-splits",
    sections: ["Four ways to split", "Four ways to split the work"],
    title: "Four ways to split",
    look: "The list under the metrics says what each split divides, what it sends, and which link carries it.",
    step: 2,
    state: SEVENTY_B,
    focus: "Parallel layout",
  },
  {
    id: "step-parts",
    sections: ["Time: compute, traffic, and waiting", "Where the time goes"],
    title: "Where a step goes",
    look: "The strip splits one step into compute, traffic, bubble and exposed sync, and MFU counts only the compute.",
    step: 3,
    state: SEVENTY_B,
    focus: "Step time and MFU",
  },
  {
    id: "failure-rate",
    sections: ["Failures: a rate that grows with the job", "Why big jobs break more often"],
    title: "Failures grow with the job",
    look: "With 16,384 GPUs, Job MTBF falls to about three hours, so a long run is interrupted again and again.",
    step: 5,
    state: { ...LLAMA3_LAYOUT },
    focus: "Failures and checkpoints",
  },
  {
    id: "one-gpu",
    sections: ["Why it’s here"],
    title: "One GPU is not enough",
    look: "On a single GPU the 70B model needs more than ten times the 80 GB line before anything is split.",
    step: 0,
    state: { modelId: "70b", data: 1 },
    focus: "Memory per GPU",
  },
  {
    id: "shard-state",
    sections: ["Shard the state", "How to play with it"],
    title: "Shard the state",
    look: "Move ZeRO stage from 0 to 3 and watch the optimizer, gradient and weight bars shrink in turn while Activations stays put.",
    step: 0,
    focus: "Memory per GPU",
  },
  {
    id: "count-activations",
    sections: ["Count the activations"],
    title: "Four sequences per microbatch",
    look: "Activations now runs far past the line, so switch Recompute to Full and watch the bar shrink and a Recompute row appear.",
    step: 1,
    state: { zero: 3, microBatch: 4 },
    focus: "Memory per GPU",
  },
  {
    id: "split-layer",
    sections: ["Split each layer"],
    title: "A tensor group across two nodes",
    look: "At 16 GPUs per layer the TP all-reduce row is long, so drag Tensor parallel back to 8 and watch it shrink.",
    step: 2,
    state: { ...SEVENTY_B, tensor: 16 },
    focus: "Communication per step",
  },
  {
    id: "fill-pipeline",
    sections: ["Fill the pipeline"],
    title: "One microbatch in eight stages",
    look: "With one microbatch most of each row is hatched, so raise Microbatches and watch the idle corners shrink.",
    step: 3,
    state: { ...SEVENTY_B_PIPELINE, microbatches: 1 },
    focus: "Pipeline schedule",
  },
  {
    id: "pay-sync",
    sections: ["Pay for the sync"],
    title: "The once-per-step sync",
    look: "Raise Data parallel and watch DP gradient sync barely move while Tokens/s climbs, then switch Overlap gradient sync off.",
    step: 4,
    state: { ...SEVENTY_B_PIPELINE, microbatches: 32 },
    focus: "Communication per step",
  },
  {
    id: "plan-failure",
    sections: ["Plan for failure"],
    title: "How often to save",
    look: "Drag Checkpoint every to the right and watch Time lost fall, bottom out near the Young line, and climb again.",
    step: 5,
    state: { ...LLAMA3_LAYOUT, checkpointMinutes: 5 },
    focus: "Failures and checkpoints",
  },
  {
    id: "experts",
    sections: ["What to notice"],
    title: "Experts on eight GPUs",
    look: "With its experts spread over 8 GPUs Mixtral fits, and the EP all-to-all row shows what moving tokens to them costs.",
    step: 2,
    state: { modelId: "mixtral", zero: 1, expert: 8 },
    focus: "Communication per step",
  },
  {
    id: "no-headroom",
    sections: ["Where it breaks"],
    title: "A layout with no room to spare",
    look: "The 405B layout fits with only a sliver to spare, and the bar leaves out buffers that would use it up.",
    step: 5,
    state: { ...LLAMA3_LAYOUT },
    focus: "Memory per GPU",
  },
  {
    id: "interleaved",
    sections: ["Interleaved schedules", "Going deeper"],
    title: "A deep pipeline with few microbatches",
    look: "With 16 stages and 16 microbatches nearly half of each row is hatched, the bubble an interleaved schedule shrinks.",
    step: 3,
    state: { ...LLAMA3_LAYOUT },
    focus: "Pipeline schedule",
  },
];

export default scenes;
