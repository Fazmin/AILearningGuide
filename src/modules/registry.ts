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
    simple: ComponentType;
    plain: ComponentType;
    standard: ComponentType;
  };
  /** Per-surface explanations, keyed by `LabSurface` label. */
  cardInfo: ModuleCardInfo;
}

export const modules: RegisteredModule[] = Object.values(moduleFiles)
  .map(({ default: definition }) => {
    const standard = contentFiles[`./${definition.slug}/content/standard.mdx`]?.default;
    const plain = contentFiles[`./${definition.slug}/content/plain.mdx`]?.default;
    // TODO(simple-mode): require simple.mdx once every module has one; until then Plain stands in.
    const simple = contentFiles[`./${definition.slug}/content/simple.mdx`]?.default ?? plain;

    if (!standard || !plain || !simple) {
      throw new Error(`Module "${definition.slug}" is missing simple.mdx, plain.mdx or standard.mdx.`);
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
      content: { simple, plain, standard },
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
