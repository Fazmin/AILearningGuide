import type { ModuleGroupId } from "@app/module-sdk";

export interface ModuleGroup {
  id: ModuleGroupId;
  title: string;
  description: string;
}

export const moduleGroups: ModuleGroup[] = [
  {
    id: "groundwork",
    title: "Start with the groundwork",
    description: "What AI is, the math, data, classical methods, the supervised loop, and evaluation",
  },
  {
    id: "foundations",
    title: "Build the foundations",
    description: "Representations, neurons, optimization, and training",
  },
  {
    id: "inside-models",
    title: "Look inside models",
    description: "Generation, attention, transformers, and inference",
  },
  {
    id: "building-with-models",
    title: "Build with a trained model",
    description: "Search, retrieval, tools, agents, multimodal systems, and safety",
  },
  {
    id: "frontiers",
    title: "Explore the frontier",
    description: "Alignment, reasoning, generation, and interpretability",
  },
  {
    id: "pretraining",
    title: "Pretrain a model",
    description: "Training runs, parallelism, data, optimizers, stability, evaluation, and mid-training",
  },
  {
    id: "training-adapting",
    title: "Adapt and ship a model",
    description: "Fine-tuning, adapters, embeddings, alignment, RL environments, compression, and shipping",
  },
];
