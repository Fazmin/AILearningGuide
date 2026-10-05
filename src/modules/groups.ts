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
    id: "training-adapting",
    title: "Train and adapt a model",
    description: "Data, training runs, fine-tuning, evaluation, and shipping",
  },
];
