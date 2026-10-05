import type { ComponentType } from "react";
import type {
  MdxModule,
  ModuleCardInfo,
  ModuleDefinition,
  ModuleGroupId,
} from "@app/module-sdk";

const moduleFiles = import.meta.glob<{ default: ModuleDefinition }>("./*/module.ts", {
  eager: true,
});

const contentFiles = import.meta.glob<MdxModule>("./*/content/*.mdx", {
  eager: true,
});

const cardInfoFiles = import.meta.glob<{ default: ModuleCardInfo }>("./*/card-info.ts", {
  eager: true,
});

export interface RegisteredModule extends ModuleDefinition {
  content: {
    standard: ComponentType;
    plain: ComponentType;
  };
  /** Per-surface explanations, keyed by `LabSurface` label. */
  cardInfo: ModuleCardInfo;
}

export const modules: RegisteredModule[] = Object.values(moduleFiles)
  .map(({ default: definition }) => {
    const standardPath = `./${definition.slug}/content/standard.mdx`;
    const plainPath = `./${definition.slug}/content/plain.mdx`;
    const standard = contentFiles[standardPath]?.default;
    const plain = contentFiles[plainPath]?.default;

    if (!standard || !plain) {
      throw new Error(`Module "${definition.slug}" is missing standard.mdx or plain.mdx.`);
    }
    if (
      definition.steps.length === 0 ||
      definition.stepInstructions.length !== definition.steps.length ||
      definition.stepInstructions.some((instruction) => !instruction.trim())
    ) {
      throw new Error(
        `Module "${definition.slug}" must provide one non-empty instruction for every step.`,
      );
    }

    return {
      ...definition,
      content: { standard, plain },
      cardInfo: cardInfoFiles[`./${definition.slug}/card-info.ts`]?.default ?? {},
    };
  })
  .sort((a, b) => a.order - b.order);

export const modulesById = new Map(modules.map((module) => [module.id, module]));
export const modulesBySlug = new Map(modules.map((module) => [module.slug, module]));

export function modulesInGroup(group: ModuleGroupId) {
  return modules.filter((module) => module.group === group);
}

const orders = modules.map((module) => module.order);
const ids = modules.map((module) => module.id);

if (new Set(orders).size !== orders.length || new Set(ids).size !== ids.length) {
  throw new Error("Module IDs and order values must be unique.");
}
