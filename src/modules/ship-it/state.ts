import { DEFAULT_RECIPE, FORMAT_NAMES, RECIPE_BOUNDS, type ShipRecipe } from "./pipeline";

const INTEGER_KEYS = ["baseEpochs", "loraRank", "loraEpochs", "replay", "dpoSteps", "users", "context"] as const;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * Turns whatever arrives from a share link, snapshot, or hand-edited storage into a recipe the
 * lab can run: unknown keys are dropped, wrong types fall back to the default, numbers are
 * clamped to the slider range, and the format must be one the lab offers. Training runs
 * synchronously inside the render, so an unbounded epoch count must never get through.
 */
export function normalizeRecipe(raw: unknown): ShipRecipe {
  const source = raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
  const recipe: ShipRecipe = { ...DEFAULT_RECIPE };

  for (const key of Object.keys(RECIPE_BOUNDS) as Array<keyof typeof RECIPE_BOUNDS>) {
    const value = source[key];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const [low, high] = RECIPE_BOUNDS[key];
    const clamped = clamp(value, low, high);
    recipe[key] = (INTEGER_KEYS as readonly string[]).includes(key) ? Math.round(clamped) : clamped;
  }

  const format = source.format;
  if (typeof format === "string" && FORMAT_NAMES.includes(format)) recipe.format = format;
  return recipe;
}

export function recipeToState(recipe: ShipRecipe): Record<string, string | number> {
  return { ...recipe };
}
