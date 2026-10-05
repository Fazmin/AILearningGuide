import { TINY_SCHEDULES } from "@app/module-sdk";
import { CORPUS_IDS, DATA_MODES, readSettings, SEED, type LabSettings } from "./state";

/**
 * The recipe is the part of the lab's state that a learner would hand to someone else: corpus, size,
 * epochs, learning rate, schedule, batch, clip and training text. It is written as one line of
 * `key=value` pairs so it survives being pasted into a chat or a note.
 */
export const RECIPE_HEADER = "tiny-lm-recipe v1";
/** A pasted recipe longer than this is truncated before it is read. */
export const RECIPE_MAX_LENGTH = 400;

type RecipeKeys = Pick<LabSettings, "corpusId" | "rank" | "epochs" | "learningRate" | "schedule" | "batchSize" | "clipNorm" | "data">;

export function formatRecipe(settings: RecipeKeys): string {
  const clip = settings.clipNorm > 0 ? settings.clipNorm.toFixed(2) : "off";
  return [
    RECIPE_HEADER,
    `corpus=${settings.corpusId}`,
    `rank=${settings.rank}`,
    `epochs=${settings.epochs}`,
    `lr=${settings.learningRate}`,
    `schedule=${settings.schedule}`,
    `batch=${settings.batchSize}`,
    `clip=${clip}`,
    `data=${settings.data}`,
    `seed=${SEED}`,
  ].join("; ");
}

/** Short prose for the model card: "20 epochs · batch 16 · learning rate 0.5 · constant · no clipping". */
export function describeRecipe(settings: RecipeKeys): string {
  const schedule = TINY_SCHEDULES.find((entry) => entry.value === settings.schedule)?.label ?? settings.schedule;
  return [
    `${settings.epochs} epochs`,
    `batch ${settings.batchSize}`,
    `learning rate ${settings.learningRate}`,
    schedule.toLowerCase(),
    settings.clipNorm > 0 ? `clip ${settings.clipNorm.toFixed(2)}` : "no clipping",
    `seed ${SEED}`,
  ].join(" · ");
}

/**
 * Reads a pasted recipe. Unknown keys and the seed are ignored, a value that is not one of the
 * control's options is skipped, and a number is clamped or snapped like any other state, so a recipe
 * can only ever select settings the controls could produce. Returns null when no setting was
 * recognised.
 */
export function parseRecipe(text: string): Partial<LabSettings> | null {
  const pairs = new Map<string, string>();
  for (const part of text.slice(0, RECIPE_MAX_LENGTH).split(";")) {
    const at = part.indexOf("=");
    if (at > 0) pairs.set(part.slice(0, at).trim().toLowerCase(), part.slice(at + 1).trim().toLowerCase());
  }
  const patch: Record<string, unknown> = {};
  const choice = (key: string, field: string, allowed: ReadonlyArray<string>) => {
    const raw = pairs.get(key);
    if (raw !== undefined && allowed.includes(raw)) patch[field] = raw;
  };
  const numeric = (key: string, field: string, words: Record<string, number> = {}) => {
    const raw = pairs.get(key);
    if (raw === undefined || raw === "") return;
    const value = raw in words ? words[raw] : Number(raw);
    if (Number.isFinite(value)) patch[field] = value;
  };
  choice("corpus", "corpusId", CORPUS_IDS);
  numeric("rank", "rank");
  numeric("epochs", "epochs");
  numeric("lr", "learningRate");
  choice("schedule", "schedule", TINY_SCHEDULES.map((entry) => entry.value));
  numeric("batch", "batchSize");
  numeric("clip", "clipNorm", { off: 0 });
  choice("data", "data", DATA_MODES);
  const fields = Object.keys(patch) as Array<keyof LabSettings>;
  if (fields.length === 0) return null;
  const checked = readSettings(patch);
  const result: Record<string, unknown> = {};
  for (const field of fields) result[field] = checked[field];
  return result as Partial<LabSettings>;
}
