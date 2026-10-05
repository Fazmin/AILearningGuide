import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

/**
 * Static discovery of the `LabSurface` cards each module renders.
 *
 * Card explanations are keyed by surface label, so both `check:modules` and the
 * card-info test need the same list of labels per module. Modules either render
 * their own Explore.tsx or delegate to the PlannedLab placeholder, so both shapes
 * are resolved here.
 */

/** Finds the `>` that closes an opening JSX tag, ignoring braces and arrows. */
function openingTagEnd(source, from) {
  let depth = 0;
  for (let index = from; index < source.length; index += 1) {
    const character = source[index];
    if (character === "{") depth += 1;
    else if (character === "}") depth -= 1;
    else if (character === ">" && depth === 0 && source[index - 1] !== "=") return index;
  }
  return source.length;
}

/** Collects the literal `label` of every `<LabSurface>` in a source string. */
export function labSurfaceLabels(source) {
  const labels = [];
  const dynamic = [];
  const tag = "<LabSurface";

  for (let index = source.indexOf(tag); index !== -1; index = source.indexOf(tag, index + tag.length)) {
    const attributes = source.slice(index, openingTagEnd(source, index + tag.length));
    const literal = attributes.match(/\blabel="((?:[^"\\]|\\.)*)"/)?.[1];
    if (literal) labels.push(literal);
    else if (/\blabel=/.test(attributes)) dynamic.push(attributes.slice(0, 80));
  }

  return { labels, dynamic };
}

/**
 * Resolves every module's card labels.
 *
 * @param root absolute path to the repository root
 * @returns Map of module slug to `{ labels, dynamic }`
 */
export async function collectModuleLabels(root) {
  const modulesRoot = path.join(root, "src", "modules");
  const folders = (await readdir(modulesRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  const plannedSource = await readFile(
    path.join(root, "src", "module-sdk", "PlannedLab.tsx"),
    "utf8",
  );
  const plannedLabels = labSurfaceLabels(plannedSource);
  const byModule = new Map();

  for (const slug of folders) {
    let explore = "";
    try {
      explore = await readFile(path.join(modulesRoot, slug, "Explore.tsx"), "utf8");
    } catch {
      continue;
    }

    const own = labSurfaceLabels(explore);
    // A module that only delegates to PlannedLab renders that placeholder's surface.
    if (own.labels.length === 0 && /\bPlannedLab\b/.test(explore)) {
      byModule.set(slug, plannedLabels);
    } else {
      byModule.set(slug, own);
    }
  }

  return byModule;
}
