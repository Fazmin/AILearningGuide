import { access, readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { collectModuleLabels } from "./lib/lab-surfaces.mjs";
import { parseModuleCheckpoints } from "./lib/search-documents.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulesRoot = path.join(root, "src", "modules");
const entries = await readdir(modulesRoot, { withFileTypes: true });
const folders = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
const styles = await readFile(path.join(root, "src", "styles.css"), "utf8");
const surfacesByModule = await collectModuleLabels(root);
const ids = new Map();
const orders = new Map();
const failures = [];
const requiredSections = [
  "What it is",
  "Why it’s here",
  "How to play with it",
  "What to notice",
  "Where it breaks",
  "Going deeper",
];

for (const folder of folders) {
  const directory = path.join(modulesRoot, folder);
  const required = [
    "module.ts",
    "Explore.tsx",
    "card-info.ts",
    "content/standard.mdx",
    "content/plain.mdx",
    "assets.json",
  ];

  for (const file of required) {
    try {
      await access(path.join(directory, file));
    } catch {
      failures.push(`${folder}: missing ${file}`);
    }
  }

  let source = "";
  try {
    source = await readFile(path.join(directory, "module.ts"), "utf8");
  } catch {
    continue;
  }

  const id = source.match(/\bid:\s*"([^"]+)"/)?.[1];
  const slug = source.match(/\bslug:\s*"([^"]+)"/)?.[1];
  const order = Number(source.match(/\border:\s*(\d+)/)?.[1]);
  const stateVersion = Number(source.match(/\bstateVersion:\s*(\d+)/)?.[1]);
  const stepsSource = source.match(/\bsteps:\s*\[([\s\S]*?)\],\s*stepInstructions:/)?.[1];
  const instructionsSource = source.match(/\bstepInstructions:\s*\[([\s\S]*?)\],\s*stateVersion:/)?.[1];
  const countStrings = (value) =>
    value?.match(/"(?:\\.|[^"\\])*"/g)?.length ?? 0;

  if (!id) failures.push(`${folder}: module.ts has no literal id`);
  if (slug !== folder) failures.push(`${folder}: slug must match its folder name`);
  if (!Number.isInteger(order) || order < 1) failures.push(`${folder}: order must be a positive integer`);
  if (!Number.isInteger(stateVersion) || stateVersion < 1) failures.push(`${folder}: stateVersion must be a positive integer`);
  if (!stepsSource || !instructionsSource) {
    failures.push(`${folder}: steps and stepInstructions must be literal arrays`);
  } else if (countStrings(stepsSource) !== countStrings(instructionsSource)) {
    failures.push(`${folder}: stepInstructions must contain one entry per step`);
  } else {
    for (let step = 0; step < countStrings(stepsSource); step += 1) {
      const focusHook = `[data-module="${folder}"][data-current-step="${step}"]`;
      if (!styles.includes(focusHook)) {
        failures.push(`${folder}: step ${step + 1} has no visual focus target`);
      }
    }
  }
  if (id) {
    if (ids.has(id)) failures.push(`${folder}: duplicate id "${id}" also used by ${ids.get(id)}`);
    ids.set(id, folder);
  }
  if (Number.isInteger(order)) {
    if (orders.has(order)) failures.push(`${folder}: duplicate order ${order} also used by ${orders.get(order)}`);
    orders.set(order, folder);
  }

  // Checkpoint: a list of at least three questions with every objective covered. The question
  // text quality rules live in registry.test.ts, which can read the values rather than the source.
  {
    const { isList, questions } = parseModuleCheckpoints(source);
    const objectiveCount = countStrings(
      source.match(/\bobjectives:\s*\[([\s\S]*?)\],\s*glossary:/)?.[1],
    );
    if (questions.length === 0) {
      failures.push(`${folder}: checkpoint must be a literal question object or an array of them`);
    }
    questions.forEach((question, index) => {
      const label = `${folder}: checkpoint question ${index + 1}`;
      if (!question.prompt) failures.push(`${label} has no prompt (a question object must start with its prompt key)`);
      if (question.options.length < 2) failures.push(`${label} needs at least two options`);
      if (!Number.isInteger(question.answer) || question.answer < 0 || question.answer >= question.options.length) {
        failures.push(`${label}: answer must index one of its options`);
      }
      if (question.explanation.length <= 20) failures.push(`${label} needs an explanation`);
      if (question.objective !== undefined && question.objective >= objectiveCount) {
        failures.push(`${label}: objective ${question.objective} is not one of the module's ${objectiveCount} objectives`);
      }
    });
    if (!isList) failures.push(`${folder}: checkpoint must be a list of at least 3 questions (see MODULE_AUTHORING.md)`);
    if (isList) {
      if (questions.length < 3) failures.push(`${folder}: a checkpoint list needs at least 3 questions, found ${questions.length}`);
      const covered = new Set(questions.map((question) => question.objective));
      for (let objective = 0; objective < objectiveCount; objective += 1) {
        if (!covered.has(objective)) failures.push(`${folder}: no checkpoint question tests objective ${objective + 1}`);
      }
    }
  }

  const glossarySource = source.match(/\bglossary:\s*\[([\s\S]*?)\],\s*checkpoint:/)?.[1];
  const glossaryTerms = [
    ...(glossarySource ?? "").matchAll(/(?:"term"|term):\s*"((?:\\.|[^"\\])*)"/g),
  ].map((match) => match[1]);

  if (!glossarySource) {
    failures.push(`${folder}: glossary must be a literal array before checkpoint`);
  } else if (glossaryTerms.length < 6) {
    failures.push(`${folder}: glossary needs at least 6 terms, found ${glossaryTerms.length}`);
  }

  for (const mode of ["standard", "plain"]) {
    let content = "";
    try {
      content = await readFile(path.join(directory, "content", `${mode}.mdx`), "utf8");
    } catch {
      continue;
    }
    if (!content.startsWith("---\n")) failures.push(`${folder}/${mode}.mdx: frontmatter is required`);
    for (const section of requiredSections) {
      if (!content.includes(`## ${section}`)) failures.push(`${folder}/${mode}.mdx: missing "${section}" section`);
    }
    // Both modes open with an overview before the six shared sections (CONTENT_STYLE_GUIDE.md).
    const headings = [...content.matchAll(/^## (.+)$/gm)].map((match) => match[1].trim());
    const expected = ["The big picture", ...requiredSections];
    if (headings.join("\n") !== expected.join("\n")) {
      failures.push(`${folder}/${mode}.mdx: sections must be exactly, in order: ${expected.join(" / ")}`);
    }
    if (mode === "plain") {
      if (!content.includes("**Try this:**")) failures.push(`${folder}/plain.mdx: needs a "Try this" nudge`);
      if (!content.includes("**A common mix-up:**")) failures.push(`${folder}/plain.mdx: needs a corrected misconception`);
    }

    const termsSource = content.match(/^terms:\s*(\[.*\])\s*$/m)?.[1];
    let frontmatterTerms;
    try {
      frontmatterTerms = JSON.parse(termsSource);
    } catch {
      failures.push(`${folder}/${mode}.mdx: "terms" frontmatter must be a JSON array on one line`);
      continue;
    }
    if (glossarySource && JSON.stringify(frontmatterTerms) !== JSON.stringify(glossaryTerms)) {
      failures.push(`${folder}/${mode}.mdx: "terms" frontmatter must match the module glossary exactly`);
    }
  }

  // Every lab surface needs an info explanation, keyed by its exact label.
  const surfaces = surfacesByModule.get(folder);
  if (!surfaces || surfaces.labels.length === 0) {
    failures.push(`${folder}: no LabSurface cards were found to explain`);
  } else {
    for (const attributes of surfaces.dynamic) {
      failures.push(`${folder}: LabSurface label must be a string literal, found "${attributes}"`);
    }

    let cardInfo = "";
    try {
      cardInfo = await readFile(path.join(directory, "card-info.ts"), "utf8");
    } catch {
      cardInfo = "";
    }

    if (cardInfo) {
      const declarations = [...cardInfo.matchAll(/^ {2}"((?:[^"\\]|\\.)*)":\s*\{$/gm)];
      const declared = declarations.map((match) => match[1]);

      for (const label of surfaces.labels) {
        if (!declared.includes(label)) {
          failures.push(`${folder}/card-info.ts: no explanation for the "${label}" card`);
        }
      }
      for (const label of declared) {
        if (!surfaces.labels.includes(label)) {
          failures.push(`${folder}/card-info.ts: "${label}" matches no LabSurface label`);
        }
      }

      // Each explanation owes a mechanism, the visible signal, and its limits.
      declarations.forEach((declaration, position) => {
        const next = declarations[position + 1];
        const body = cardInfo.slice(
          declaration.index,
          next ? next.index : cardInfo.length,
        );
        for (const field of ["title", "summary", "whatYouSee", "howItWorks", "limits"]) {
          if (!new RegExp(`^ {4}${field}:`, "m").test(body)) {
            failures.push(`${folder}/card-info.ts: "${declaration[1]}" is missing ${field}`);
          }
        }
        for (const field of ["whatYouSee", "howItWorks", "limits"]) {
          const items = body.match(new RegExp(`^ {4}${field}: \\[([\\s\\S]*?)^ {4}\\]`, "m"))?.[1];
          const count = items?.match(/^ {6}"/gm)?.length ?? 0;
          if (count < 2) {
            failures.push(
              `${folder}/card-info.ts: "${declaration[1]}" needs at least 2 ${field} bullets, found ${count}`,
            );
          }
        }
        const limits = body.match(/^ {4}limits: \[([\s\S]*?)^ {4}\]/m)?.[1] ?? "";
        for (const prefix of ["In this lab: ", "In general: "]) {
          if (!limits.includes(`"${prefix}`)) {
            failures.push(
              `${folder}/card-info.ts: "${declaration[1]}" limits must include a "${prefix}" bullet`,
            );
          }
        }
      });
    }
  }

  try {
    const manifest = JSON.parse(await readFile(path.join(directory, "assets.json"), "utf8"));
    if (!Array.isArray(manifest.assets)) failures.push(`${folder}/assets.json: "assets" must be an array`);
    for (const asset of manifest.assets ?? []) {
      const relativePath = typeof asset === "string" ? asset : asset.path;
      if (!relativePath || relativePath.includes("..")) {
        failures.push(`${folder}/assets.json: invalid asset path`);
        continue;
      }
      try {
        await access(path.join(directory, relativePath));
      } catch {
        failures.push(`${folder}/assets.json: referenced asset "${relativePath}" does not exist`);
      }
    }
  } catch {
    failures.push(`${folder}/assets.json: must contain valid JSON`);
  }
}

if (failures.length) {
  console.error(`Module validation failed with ${failures.length} issue${failures.length === 1 ? "" : "s"}:`);
  failures.forEach((failure) => console.error(`  - ${failure}`));
  process.exit(1);
}

const cardCount = [...surfacesByModule.values()].reduce(
  (total, surfaces) => total + surfaces.labels.length,
  0,
);

console.log(
  `Validated ${folders.length} modules: unique IDs and orders, complete content, ` +
    `valid assets, ${cardCount} explained lab cards.`,
);
